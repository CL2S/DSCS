"""
Counterfactual Intervention Demo

Trains a small model on eICU sepsis data, then demonstrates the full
experience-memory pipeline on a single test patient:
  1. Current state assessment (clinical_state_signature, severity, trajectory)
  2. Factual forecast (base-only vs memory-enabled)
  3. Prototype retrieval (what similar cases are in the experience library)
  4. Counterfactual intervention plan (alternative donor interventions)
  5. Predicted outcome delta per candidate intervention

Usage:
  python scripts/demo_counterfactual.py

Quick smoke with 8 series:
  python scripts/demo_counterfactual.py --smoke
"""

import argparse
import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.persistent_memory_store import PersistentExperienceStore
from src.manifold_forecasting_trainer import (
    EndToEndForecastingManifoldTrainer,
    ForecastingTrainerConfig,
    PATTERN_LABELS,
    TRAJECTORY_LABELS,
)


def _point_metrics(truth, pred) -> dict:
    errs = [abs(t - p) for t, p in zip(truth, pred)]
    return {
        "mae": float(sum(errs) / max(1, len(errs))),
        "max_error": float(max(errs)) if errs else 0.0,
    }


def load_or_train(args):
    """Load dataset and train a small model."""
    from src.tsf_data import load_forecasting_dataset

    dataset = load_forecasting_dataset(
        format="eicu_sepsis3",
        eicu_max_series=args.eicu_max_series,
        history_length=args.history_length,
        forecast_horizon=args.forecast_horizon,
        max_train_windows_per_series=args.max_train_windows,
        device=args.device,
        enable_kg=not args.disable_kg,
    )

    mc = dataset._manifold_config()
    tc = ForecastingTrainerConfig(
        forecast_horizon=dataset.forecast_horizon,
        seasonality=dataset.seasonality,
        history_length=dataset.history_length,
        dataset_name=dataset.dataset_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
    )

    trainer = EndToEndForecastingManifoldTrainer(
        memory_config=mc,
        trainer_config=tc,
        static_feature_dim=dataset.static_feature_dim,
        kg_feature_dim=dataset.kg_feature_dim,
        intervention_feature_dim=dataset.intervention_feature_dim,
        intervention_sequence_dim=dataset.intervention_sequence_dim,
        formation_feature_dim=dataset.formation_feature_dim,
    )

    persistent_store = None
    if not args.disable_persistent:
        store_path = args.persistent_memory_store
        persistent_store = PersistentExperienceStore(Path(store_path))
        if args.build_store:
            print(f"[demo] Building persistent store at {store_path} ...")
            persistent_store.upsert_samples(
                dataset.train_samples, source="train", prototype_update_mode="incremental"
            )
        persistent_samples = persistent_store.load_samples(
            allowed_splits=("train", "external"), strict_no_test=True
        )
        trainer.prime_persistent_memory(persistent_samples)

    trainer.enable_transition_memory = not args.disable_transition
    trainer.enable_kg = not args.disable_kg

    print(f"[demo] Training: {len(dataset.train_samples)} train, "
          f"{len(dataset.val_samples)} val, {len(dataset.test_samples)} test, "
          f"{args.epochs} epochs ...")
    history = trainer.fit(
        train_samples=dataset.train_samples,
        val_samples=dataset.val_samples,
    )

    return trainer, dataset, persistent_store


def demo(args):
    trainer, dataset, persistent_store = load_or_train(args)

    test_samples = list(dataset.test_samples)
    if not test_samples:
        print("[demo] No test samples!")
        return

    # Pick a diverse sample: one with clear trend if possible
    target_patterns = ["down", "spike", "up", "flat"]
    sample = None
    for pat in target_patterns:
        for s in test_samples:
            if PATTERN_LABELS[int(s.pattern_label)] == pat:
                sample = s
                break
        if sample is not None:
            break
    if sample is None:
        sample = test_samples[0]

    meta = dict(sample.metadata or {})
    stay_id = meta.get("stay_id", meta.get("series_name", "unknown"))
    pattern_name = PATTERN_LABELS[int(sample.pattern_label)] if 0 <= int(sample.pattern_label) < len(PATTERN_LABELS) else "?"
    traj_name = TRAJECTORY_LABELS[int(sample.trajectory_label)] if 0 <= int(sample.trajectory_label) < len(TRAJECTORY_LABELS) else "?"

    # Clinical state
    state_payload = trainer._clinical_state_signature_payload(sample)
    future_dir = trainer._memory_manager._expected_direction_type(
        sample.formation_features, int(sample.trajectory_label)
    )

    print("\n" + "=" * 70)
    print("  COUNTERFACTUAL INTERVENTION DEMO")
    print("=" * 70)
    print(f"\n  Patient stay: {stay_id}")
    print(f"  Clinical state signature: {state_payload.get('signature')}")
    print(f"  Severity: {state_payload.get('severity_bin')}  "
          f"Level: {state_payload.get('level_bin')}  Trend: {state_payload.get('trend_bin')}")
    print(f"  Pattern: {pattern_name}  Trajectory: {traj_name}")
    print(f"  Expected direction: {future_dir}")
    print(f"  KG flags: {state_payload.get('active_kg_flags')}")
    print(f"  Context (last 4 steps): {[round(v,2) for v in sample.raw_context[-4:]]}")
    print(f"  True future: {[round(v,2) for v in sample.raw_target]}")

    # ---- 1. Factual forecast ----
    trainer.eval()
    import torch
    with torch.no_grad():
        base_pred, _ = trainer.predict([sample], use_memory=False)
        mem_pred, _ = trainer.predict([sample], use_memory=True)
    base_row = base_pred[0].detach().cpu().tolist()
    mem_row = mem_pred[0].detach().cpu().tolist()
    truth_row = sample.raw_target

    base_m = _point_metrics(truth_row, base_row)
    mem_m = _point_metrics(truth_row, mem_row)

    print(f"\n  --- Factual Forecast ---")
    print(f"  Base-only prediction:  {[round(v,2) for v in base_row]}")
    print(f"  Memory-enabled:        {[round(v,2) for v in mem_row]}")
    print(f"  Ground truth:          {[round(v,2) for v in truth_row]}")
    print(f"  Base MAE: {base_m['mae']:.4f}  Memory MAE: {mem_m['mae']:.4f}  "
          f"Gain: {base_m['mae'] - mem_m['mae']:+.4f}")

    # ---- 2. Prototype retrieval ----
    manager_result = trainer._memory_manager.read(
        encoding=trainer._encode_sample(sample),
        formation_features=sample.formation_features,
        pattern_label=sample.pattern_label,
        trajectory_label=sample.trajectory_label,
        experience_label=sample.experience_label,
        horizon=trainer.trainer_config.forecast_horizon,
        history_length=trainer.trainer_config.history_length,
        metadata=meta,
        kg_features=sample.kg_features,
        intervention_static=sample.intervention_static,
        intervention_sequence=sample.intervention_sequence,
    )
    sem = manager_result.semantic_result
    print(f"\n  --- Prototype Retrieval (top {sem.get('hit_count', 0)} hits) ---")
    print(f"  Direction alignment: {sem.get('direction_alignment', 1.0):.2f}")
    print(f"  Entropy penalty:     {sem.get('entropy_penalty', 1.0):.2f}")
    print(f"  Template confidence: {sem.get('template_confidence', 0):.3f}")
    print(f"  Template blend:      {sem.get('template_blend_weight', 0):.4f}")
    print(f"  Prototype IDs:       {sem.get('prototype_ids', [])}")
    print(f"  Template curve:      {[round(v,2) for v in sem.get('template_curve', [])]}")

    # ---- 3. Memory path audit ----
    from src.manifold_forecasting_trainer import PATTERN_LABELS, TRAJECTORY_LABELS
    fpa = trainer.audit_factual_memory_path([sample])
    trans_means = fpa.get("transition_means", {})
    dm = fpa.get("direct_memory_means", {})
    print(f"\n  --- Memory Path Audit ---")
    print(f"  Harm control scale:  {dm.get('memory_harm_control_scale', 'N/A')}")
    print(f"  Memory harm quality: {dm.get('memory_harm_quality', 'N/A')}")
    print(f"  Coord strength (pre): {dm.get('coordinated_memory_strength', 'N/A')}")
    print(f"  Coord strength (post):{dm.get('memory_harm_post_strength', 'N/A')}")
    print(f"  Transition blocked:  {trans_means.get('transition_gate_blocked', 'N/A')}")
    print(f"  Trans utility factor:{trans_means.get('transition_utility_factor', 'N/A')}")
    print(f"  Trans pattern factor:{trans_means.get('transition_pattern_factor', 'N/A')}")
    print(f"  Trans traj factor:   {trans_means.get('transition_trajectory_factor', 'N/A')}")
    print(f"  Gate reasons:        {fpa.get('transition_gate_blocked_reasons', {})}")

    # ---- 4. Counterfactual intervention plan ----
    if not args.disable_counterfactual:
        print(f"\n  --- Counterfactual Intervention Candidates ---")
        try:
            counterfactual_summary = trainer.evaluate_counterfactual_plan([sample])
            cases = counterfactual_summary.get("case_details", [])
            if cases:
                case = cases[0]
                candidates = case.get("donor_candidates", [])[:3]
                for i, cand in enumerate(candidates):
                    donor = cand.get("donor", {})
                    print(f"\n  Candidate #{i+1}:")
                    print(f"    Donor stay:     {donor.get('donor_stay_id', '?')}")
                    print(f"    Similarity:     {donor.get('donor_similarity', 0):.3f}")
                    print(f"    Guideline:      {donor.get('donor_guideline_compatibility', 0):.3f}")
                    print(f"    Total score:    {donor.get('donor_total_score', 0):.3f}")
                    print(f"    Predicted delta:{donor.get('counterfactual_predicted_delta', 'N/A')}")
                    print(f"    Intervention:   {donor.get('donor_intervention_summary', 'N/A')}")
            else:
                print("  (no donor candidates found for this sample)")

            # Show the selected recommendation
            selected = case.get("selected_candidate", {}) if cases else {}
            if selected:
                print(f"\n  >>> Recommended intervention:")
                sel_donor = selected.get("donor", {})
                print(f"  Donor: {sel_donor.get('donor_stay_id', '?')} "
                      f"score: {sel_donor.get('donor_total_score', 0):.3f}")
                print(f"  Predicted SOFA delta: {selected.get('counterfactual_predicted_delta', 'N/A')}")
        except Exception as exc:
            print(f"  (counterfactual evaluation skipped: {exc})")

    # ---- 5. Summary verdict ----
    print(f"\n  {'='*60}")
    print(f"  SUMMARY")
    print(f"  {'='*60}")
    print(f"  Patient {stay_id}: {pattern_name}|{traj_name}, severity={state_payload.get('severity_bin')}")
    print(f"  Factual forecast:   memory {'helps' if base_m['mae'] > mem_m['mae'] else 'does not help'} "
          f"(gain {base_m['mae'] - mem_m['mae']:+.4f} MAE)")
    print(f"  Prototype matches:  {sem.get('hit_count', 0)} prototypes, "
          f"direction_align={sem.get('direction_alignment', 1.0):.2f}")
    print(f"  Transition gate:    {'blocked' if trans_means.get('transition_gate_blocked', 0) > 0.5 else 'active'}")
    print(f"  Harm control:       memory scaled to {dm.get('memory_harm_control_scale', 1.0):.2f}x")
    print(f"  Experience library: {persistent_store.summarize() if persistent_store else 'N/A'}")
    print()


def parse_args():
    p = argparse.ArgumentParser(description="Counterfactual Intervention Demo")
    p.add_argument("--eicu-max-series", type=int, default=64)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--history-length", type=int, default=8)
    p.add_argument("--forecast-horizon", type=int, default=4)
    p.add_argument("--max-train-windows", type=int, default=12)
    p.add_argument("--device", default="cpu")
    p.add_argument("--persistent-memory-store", default=r".\output\persistent_memory\demo_store")
    p.add_argument("--disable-persistent", action="store_true")
    p.add_argument("--disable-kg", action="store_true")
    p.add_argument("--disable-transition", action="store_true")
    p.add_argument("--disable-counterfactual", action="store_true")
    p.add_argument("--build-store", action="store_true")
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.smoke:
        args.eicu_max_series = 8
        args.epochs = 1
        args.batch_size = 4
        args.max_train_windows = 1
        args.build_store = True
        print("[demo] Smoke mode: 8 series, 1 epoch")
    demo(args)


if __name__ == "__main__":
    main()
