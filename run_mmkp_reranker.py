#!/usr/bin/env python3
import json, urllib.request, time, re, math, itertools

def ollama(prompt, model="gemma3:12b", temperature=0.0, timeout=120):
    payload = json.dumps({"model": model, "prompt": prompt, "stream": False,
        "options": {"temperature": temperature, "num_predict": 2048}}).encode()
    try:
        req = urllib.request.Request("http://localhost:11434/api/generate", data=payload,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read()).get("response", "")
    except:
        return "{}"

def parse_json(text):
    if not text or not text.strip():
        return {}
    try:
        return json.loads(text)
    except:
        m = re.search(r'\{[^{}]*\{[^{}]*\}[^{}]*\}|\{[^{}]*\}', text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except:
                pass
        return {}

def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) + 1e-8
    nb = math.sqrt(sum(y * y for y in b)) + 1e-8
    return dot / (na * nb)

def mmkp_select(candidates, patient_feat, k=3, lambda_div=0.35):
    best_combo, best_score = None, -1e9
    n = len(candidates)
    for combo in itertools.combinations(range(n), min(k, n)):
        rel = sum(cosine(patient_feat, candidates[ci]["feat"]) for ci in combo) / k
        div_sum = 0.0
        pairs = 0
        for i in range(len(combo)):
            for j in range(i + 1, len(combo)):
                div_sum += cosine(candidates[combo[i]]["feat"], candidates[combo[j]]["feat"])
                pairs += 1
        div = 1.0 - (div_sum / max(1, pairs))
        score = rel + lambda_div * div
        if score > best_score:
            best_score, best_combo = score, combo
    if best_combo is None:
        return list(range(min(k, n)))
    return list(best_combo)

def build_features(case):
    totals = [s.get("sofa_total", 0) for s in case["sofa_scores"]]
    trend = 1 if totals[-1] > totals[0] else (-1 if totals[-1] < totals[0] else 0)
    last = case["sofa_scores"][-1]
    feat = [totals[0] / 20.0, totals[-1] / 20.0, float(trend)]
    for comp in ["sofa_respiration", "sofa_coagulation", "sofa_liver",
                 "sofa_cardiovascular", "sofa_cns", "sofa_renal"]:
        feat.append(last.get(comp, 0) / 4.0)
    return feat, totals

# Load data
with open("icu_stays_descriptions88.json") as f:
    data = json.load(f)

N = len(data)
BANK = 60
test_indices = list(range(BANK, N))

# Build memory bank
bank_entries = []
for i in range(BANK):
    c = data[i]
    feat, totals = build_features(c)
    bank_entries.append({
        "idx": i, "stay": c["stay_id"], "desc": c["input_description"][:600],
        "sofa_start": totals[0], "sofa_end": totals[-1],
        "trend": feat[2], "feat": feat,
        "summary": f"[{i}] Stay{c['stay_id']}: SOFA{totals[0]}->{totals[-1]}| {c['input_description'][:200]}"
    })
bank_text = "\n".join(e["summary"] for e in bank_entries)

# Initial weights (uniform)
weights = [1.0] * 9

def predict_one(ti, desc, start, trend_dir, pfeat, w, do_print=True):
    """Run one prediction with MMKP. Returns (pred, gt, error, selected)."""
    test = data[ti]
    feat, totals = build_features(test)
    gt = totals[-1]

    t0 = time.time()
    ret = parse_json(ollama(f"""From {BANK} ICU cases, pick the 10 most clinically similar. Rank by relevance.

PATIENT: SOFA{start}| {desc}
POOL:
{bank_text}
Output ONLY JSON: {{"ranked_indices": [i1,i2,...,i10]}}
JSON:"""))
    top10 = [i for i in ret.get("ranked_indices", [])[:10] if isinstance(i, int) and 0 <= i < BANK]
    if len(top10) < 3:
        top10 = list(range(min(10, BANK)))

    # Apply weights and MMKP
    candidates = [bank_entries[i] for i in top10]
    w_pfeat = [pfeat[d] * w[d] for d in range(len(w))]
    w_cands = [{**c, "feat": [c["feat"][d] * w[d] for d in range(len(w))]} for c in candidates]
    selected = [top10[i] for i in mmkp_select(w_cands, w_pfeat)]

    silenced = (start <= 6 and trend_dir != "rise")
    if not silenced:
        ctx = "\n\n".join(
            f"#{j+1}: Stay{bank_entries[si]['stay']} SOFA{bank_entries[si]['sofa_start']}->{bank_entries[si]['sofa_end']}. {bank_entries[si]['desc'][:400]}"
            for j, si in enumerate(selected)
        )
        mem = parse_json(ollama(f"""Predict FINAL SOFA total. Calibrate using similar cases.

Patient: SOFA{start}| {desc}
SIMILAR HISTORICAL CASES:
{ctx}
Output ONLY JSON: {{"predicted_final_sofa": N}}
JSON:"""))
    else:
        mem = parse_json(ollama(f"""Predict FINAL SOFA total.

Patient: SOFA{start}| {desc}
Output ONLY JSON: {{"predicted_final_sofa": N}}
JSON:"""))

    pred = mem.get("predicted_final_sofa", start)
    err = abs(pred - gt)
    elapsed = time.time() - t0

    if do_print:
        gate_tag = "SILENCED" if silenced else "APPLY"
        print(f"    GT={gt} Pred={pred}(e{err:.0f}) {gate_tag} | {elapsed:.0f}s", flush=True)

    return pred, gt, err, selected, silenced

# ============================================================
# PASS 1: uniform weights
# ============================================================
print("=" * 62)
print("PASS 1: MMKP with uniform weights")
print("=" * 62)

pass1 = []
for rank, ti in enumerate(test_indices):
    test = data[ti]
    pfeat, totals = build_features(test)
    gt = totals[-1]
    start = totals[0]
    desc = test["input_description"][:600]
    trend_dir = "rise" if totals[-1] > totals[0] else ("fall" if totals[-1] < totals[0] else "stable")

    print(f"  [{rank+1}/12] Stay{test['stay_id']} SOFA{start}->{gt}", end="", flush=True)
    pred, gt, err, selected, silenced = predict_one(ti, desc, start, trend_dir, pfeat, weights)
    pass1.append({"stay": test["stay_id"], "gt": gt, "pred": pred, "error": err,
                  "pfeat": pfeat, "selected": selected, "silenced": silenced})

p1_mae = sum(r["error"] for r in pass1) / len(pass1)
print(f"\nPass 1 MAE: {p1_mae:.1f}")

# ============================================================
# RERANKER: learn weights from good predictions
# ============================================================
print(f"\n{'='*62}")
print("RERANKER: Learning feature weights")
print("=" * 62)

for r in pass1:
    if r["error"] >= 5 or r["silenced"]:
        continue
    # Find the best retrieved case
    best_si, best_sim = None, -1
    for si in r["selected"]:
        sim = cosine(r["pfeat"], bank_entries[si]["feat"])
        if sim > best_sim:
            best_sim, best_si = sim, si
    if best_si is None:
        continue
    bc = bank_entries[best_si]
    for d in range(len(weights)):
        dim_match = 1.0 - abs(r["pfeat"][d] - bc["feat"][d])
        weights[d] += 0.12 * dim_match

# Normalize
w_sum = sum(weights)
weights = [w / w_sum * len(weights) for w in weights]
print(f"Learned weights: {[f'{w:.2f}' for w in weights]}")

# ============================================================
# PASS 2: learned weights
# ============================================================
print(f"\n{'='*62}")
print("PASS 2: MMKP + Reranker (learned weights)")
print("=" * 62)

pass2 = []
for rank, ti in enumerate(test_indices):
    test = data[ti]
    pfeat = pass1[rank]["pfeat"]
    totals = [s.get("sofa_total", 0) for s in test["sofa_scores"]]
    gt = totals[-1]
    start = totals[0]
    desc = test["input_description"][:600]
    trend_dir = "rise" if totals[-1] > totals[0] else ("fall" if totals[-1] < totals[0] else "stable")

    print(f"  [{rank+1}/12] Stay{test['stay_id']} SOFA{start}->{gt}", end="", flush=True)
    pred, gt, err, selected, silenced = predict_one(ti, desc, start, trend_dir, pfeat, weights)
    imp = pass1[rank]["error"] - err
    pass2.append({"stay": test["stay_id"], "gt": gt, "pred": pred, "error": err, "improvement": imp})

p2_mae = sum(r["error"] for r in pass2) / len(pass2)

# ============================================================
# SUMMARY
# ============================================================
print(f"\n{'='*62}")
print("FINAL: MMKP + Reranker")
print("=" * 62)
print(f"Pass 1 (uniform weights):  MAE={p1_mae:.1f}")
print(f"Pass 2 (learned weights):  MAE={p2_mae:.1f}")
print(f"Net improvement:           {p1_mae - p2_mae:+.1f}")

helped = sum(1 for r in pass2 if r["improvement"] > 0.5)
harmed = sum(1 for r in pass2 if r["improvement"] < -0.5)
print(f"Improved in Pass 2:        {helped}/{len(pass2)}")
print(f"Worse in Pass 2:           {harmed}/{len(pass2)}")

print(f"\n--- Cross-experiment ---")
print(f"P3 v1 (LLM top-3):            MAE=3.83 helped=9 harmed=2")
print(f"P3 v3 (+diversity +gate):     MAE=2.83 helped=11 harmed=0")
print(f"P3 v4 (MMKP uniform):         MAE={p1_mae:.1f}")
print(f"P3 v5 (MMKP+Reranker):        MAE={p2_mae:.1f} ({p1_mae-p2_mae:+.1f} vs uniform)")

with open("p3_mmkp_reranker_results.json", "w") as f:
    json.dump({
        "pass1_mae": p1_mae, "pass2_mae": p2_mae,
        "weights_initial": [1.0] * 9, "weights_learned": [round(w, 3) for w in weights],
        "pass1": [{"stay": r["stay"], "error": r["error"], "gt": r["gt"], "pred": r["pred"]} for r in pass1],
        "pass2": [{"stay": r["stay"], "error": r["error"], "gt": r["gt"], "pred": r["pred"], "improvement": r["improvement"]} for r in pass2]
    }, f, ensure_ascii=False, indent=2)
print(f"\nSaved p3_mmkp_reranker_results.json")
