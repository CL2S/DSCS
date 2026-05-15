import math
import random
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class AnchorMemoryItem:
    key: List[float]
    label: int
    activity: float
    support: int = 1


@dataclass
class CorrectionMemoryItem:
    key: List[float]
    label: int
    residual_logit: float
    activity: float
    support: int = 1
    residual_vector: Optional[List[float]] = None
    focus_source: int = -1
    focus_rival: int = -1


class BalancedOnlineLogisticRegressor:
    def __init__(
        self,
        epochs: int = 4,
        learning_rate: float = 0.08,
        l2: float = 1e-4,
        positive_weight_scale: float = 1.0,
        seed: int = 42,
    ):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.l2 = l2
        self.positive_weight_scale = positive_weight_scale
        self.seed = seed
        self.weights: List[float] = []
        self.bias = 0.0
        self._accum: List[float] = []
        self._bias_accum = 1e-6
        self.class_weights = {0: 1.0, 1: 1.0}

    def fit(self, x_train: List[List[float]], y_train: List[int]):
        if not x_train:
            raise ValueError("training data is empty")

        dim = len(x_train[0])
        self.weights = [0.0] * dim
        self.bias = 0.0
        self._accum = [1e-6] * dim
        self._bias_accum = 1e-6

        counts = Counter(y_train)
        total = len(y_train)
        self.class_weights = {
            0: total / max(1.0, 2.0 * counts.get(0, 0)),
            1: (total / max(1.0, 2.0 * counts.get(1, 0))) * self.positive_weight_scale,
        }

        rng = random.Random(self.seed)
        order = list(range(total))

        for _ in range(self.epochs):
            rng.shuffle(order)
            for idx in order:
                x_row = x_train[idx]
                y_true = y_train[idx]
                prob = self.predict_one(x_row)
                error = (prob - y_true) * self.class_weights[y_true]

                for j, value in enumerate(x_row):
                    grad = error * value + self.l2 * self.weights[j]
                    self._accum[j] += grad * grad
                    step = self.learning_rate / math.sqrt(self._accum[j])
                    self.weights[j] -= step * grad

                self._bias_accum += error * error
                self.bias -= self.learning_rate / math.sqrt(self._bias_accum) * error

    def predict_one(self, x_row: List[float]) -> float:
        score = self.bias + sum(weight * value for weight, value in zip(self.weights, x_row))
        if score >= 0:
            expo = math.exp(-score)
            return 1.0 / (1.0 + expo)
        expo = math.exp(score)
        return expo / (1.0 + expo)


class DynamicMemoryClassifier:
    def __init__(
        self,
        top_k: int = 8,
        sim_threshold: float = 0.92,
        merge_alpha: float = 0.2,
        decay: float = 0.997,
        forget_threshold: float = 0.1,
        max_memory: int = 256,
        base_epochs: int = 4,
        base_learning_rate: float = 0.08,
        base_l2: float = 1e-4,
        positive_weight_scale: float = 1.0,
        prototype_confidence: float = 0.84,
        memory_temperature: float = 0.12,
        correction_confidence: float = 0.78,
        uncertainty_low: float = 0.2,
        uncertainty_high: float = 0.8,
        correction_scale: float = 0.45,
        optimize_for: str = "f1_macro",
        min_recall: float = 0.0,
        min_precision: float = 0.0,
        min_specificity: float = 0.0,
        threshold_min: float = 0.15,
        threshold_max: float = 0.85,
        seed: int = 42,
    ):
        self.top_k = top_k
        self.sim_threshold = sim_threshold
        self.merge_alpha = merge_alpha
        self.decay = decay
        self.forget_threshold = forget_threshold
        self.max_memory = max_memory
        self.prototype_confidence = prototype_confidence
        self.memory_temperature = memory_temperature
        self.correction_confidence = correction_confidence
        self.uncertainty_low = uncertainty_low
        self.uncertainty_high = uncertainty_high
        self.correction_scale = correction_scale
        self.optimize_for = optimize_for
        self.min_recall = min_recall
        self.min_precision = min_precision
        self.min_specificity = min_specificity
        self.threshold_min = threshold_min
        self.threshold_max = threshold_max
        self.seed = seed

        self.num_classes = 0
        self.class_priors = {0: 0.5, 1: 0.5}
        self.decision_threshold = 0.5
        self.anchor_weight = 0.0
        self.correction_weight = 0.0
        self.anchor_direction = 1.0
        self.correction_direction = 1.0
        self.threshold_selection_summary: Dict[str, float] = {}
        self.memory_diagnostics: Dict[str, float] = {}

        self.base_model = BalancedOnlineLogisticRegressor(
            epochs=base_epochs,
            learning_rate=base_learning_rate,
            l2=base_l2,
            positive_weight_scale=positive_weight_scale,
            seed=seed,
        )
        self.base_models: List[BalancedOnlineLogisticRegressor] = []

        self._binary_hybrid_enabled = False
        self.anchor_memory_by_label: Dict[int, List[AnchorMemoryItem]] = {}
        self.correction_memory: List[CorrectionMemoryItem] = []
        self.correction_memory_by_label: Dict[int, List[CorrectionMemoryItem]] = {}
        self.memory: List[object] = []
        self.label_names: List[str] = []
        self.label_to_id: Dict[str, int] = {}
        self.priority_class_ids: List[int] = []
        self.priority_confusion_pairs: List[Tuple[int, int]] = []

    def _norm(self, vector: List[float]) -> float:
        return math.sqrt(sum(value * value for value in vector)) + 1e-12

    def _normalize(self, vector: List[float]) -> List[float]:
        norm = self._norm(vector)
        return [value / norm for value in vector]

    def _cosine(self, a: List[float], b: List[float]) -> float:
        return sum(x * y for x, y in zip(a, b))

    def _sigmoid(self, value: float) -> float:
        if value >= 0:
            expo = math.exp(-value)
            return 1.0 / (1.0 + expo)
        expo = math.exp(value)
        return expo / (1.0 + expo)

    def _logit(self, prob: float) -> float:
        safe_prob = min(max(prob, 1e-6), 1.0 - 1e-6)
        return math.log(safe_prob / (1.0 - safe_prob))

    def _target_prob(self, label: int) -> float:
        return 0.95 if label == 1 else 0.05

    def _target_prob_vector(self, label: int) -> List[float]:
        if self.num_classes <= 1:
            return [1.0]
        off_prob = 0.1 / max(1, self.num_classes - 1)
        probs = [off_prob] * self.num_classes
        probs[label] = 0.9
        return probs

    def _set_label_context(self, label_names: Optional[List[str]] = None):
        if label_names:
            self.label_names = label_names[:]
        else:
            self.label_names = [str(class_id) for class_id in range(self.num_classes)]
        self.label_to_id = {name: idx for idx, name in enumerate(self.label_names)}

        priority_names = ["worsen", "mixed"]
        self.priority_class_ids = [self.label_to_id[name] for name in priority_names if name in self.label_to_id]
        self.priority_confusion_pairs = []
        if len(self.priority_class_ids) >= 2:
            for source_id in self.priority_class_ids:
                for rival_id in self.priority_class_ids:
                    if source_id != rival_id:
                        self.priority_confusion_pairs.append((source_id, rival_id))

        mixed_id = self.label_to_id.get("mixed")
        improve_id = self.label_to_id.get("improve")
        worsen_id = self.label_to_id.get("worsen")
        if mixed_id is not None:
            for other_id in [improve_id, worsen_id]:
                if other_id is None or other_id == mixed_id:
                    continue
                self.priority_confusion_pairs.append((mixed_id, other_id))
                self.priority_confusion_pairs.append((other_id, mixed_id))

    def _is_priority_pair(self, source_id: int, rival_id: int) -> bool:
        return (source_id, rival_id) in self.priority_confusion_pairs

    def _correction_priority_bonus(self, label: int, focus_source: int, focus_rival: int) -> float:
        pair_bonus = 0.0
        if self._is_priority_pair(focus_source, focus_rival):
            pair_bonus += 0.45
        if label in self.priority_class_ids:
            pair_bonus += 0.18
        if focus_source in self.priority_class_ids or focus_rival in self.priority_class_ids:
            pair_bonus += 0.08
        if label in (focus_source, focus_rival):
            pair_bonus += 0.05
        return pair_bonus

    def _softmax(self, scores: List[float]) -> List[float]:
        if not scores:
            return []
        max_score = max(scores)
        exps = [math.exp(score - max_score) for score in scores]
        total = sum(exps) + 1e-12
        return [value / total for value in exps]

    def _binary_auc(self, y_true: List[int], y_score: List[float]) -> float:
        pairs = sorted(zip(y_score, y_true), key=lambda item: item[0])
        rank_sum = 0.0
        pos_count = 0
        neg_count = 0

        for rank, (_, label) in enumerate(pairs, start=1):
            if label == 1:
                rank_sum += rank
                pos_count += 1
            else:
                neg_count += 1

        if pos_count == 0 or neg_count == 0:
            return 0.5
        return (rank_sum - pos_count * (pos_count + 1) / 2.0) / (pos_count * neg_count)

    def _effective_memory_budget(self, train_size: int) -> int:
        suggested = max(96, int(math.sqrt(max(1, train_size)) * 1.8))
        return min(self.max_memory, suggested)

    def _anchor_limits(self, y_train: List[int], total_anchor_budget: int) -> Dict[int, int]:
        counts = Counter(y_train)
        sqrt_total = sum(math.sqrt(max(1, count)) for count in counts.values()) + 1e-12
        limits: Dict[int, int] = {}
        remaining = total_anchor_budget
        labels = sorted(counts)
        for label in labels[:-1]:
            raw = total_anchor_budget * math.sqrt(max(1, counts[label])) / sqrt_total
            limit = max(24, int(raw))
            limits[label] = limit
            remaining -= limit
        limits[labels[-1]] = max(24, remaining)
        return limits

    def _softmax_weights(self, sims: List[float]) -> List[float]:
        if not sims:
            return []
        max_sim = max(sims)
        temperature = max(self.memory_temperature, 1e-3)
        scores = [math.exp((sim - max_sim) / temperature) for sim in sims]
        total = sum(scores) + 1e-12
        return [score / total for score in scores]

    def _merge_anchor(self, bank: List[AnchorMemoryItem], x_norm: List[float], label: int, activity: float):
        best_idx = -1
        best_sim = -2.0
        for idx, item in enumerate(bank):
            sim = self._cosine(x_norm, item.key)
            if sim > best_sim:
                best_sim = sim
                best_idx = idx

        if best_idx >= 0 and best_sim >= self.sim_threshold:
            target = bank[best_idx]
            alpha = min(self.merge_alpha, 1.0 / (target.support + 1.0))
            target.key = [(1.0 - alpha) * left + alpha * right for left, right in zip(target.key, x_norm)]
            target.key = self._normalize(target.key)
            target.activity = max(target.activity, activity)
            target.support += 1
            return

        bank.append(AnchorMemoryItem(key=x_norm[:], label=label, activity=activity))

    def _merge_correction(self, bank: List[CorrectionMemoryItem], x_norm: List[float], label: int, residual_logit: float):
        best_idx = -1
        best_sim = -2.0
        for idx, item in enumerate(bank):
            if item.label != label:
                continue
            sim = self._cosine(x_norm, item.key)
            if sim > best_sim:
                best_sim = sim
                best_idx = idx

        if best_idx >= 0 and best_sim >= max(0.88, self.sim_threshold - 0.03):
            target = bank[best_idx]
            alpha = min(self.merge_alpha, 1.0 / (target.support + 1.0))
            target.key = [(1.0 - alpha) * left + alpha * right for left, right in zip(target.key, x_norm)]
            target.key = self._normalize(target.key)
            target.residual_logit = (1.0 - alpha) * target.residual_logit + alpha * residual_logit
            target.activity = max(target.activity, abs(target.residual_logit))
            target.support += 1
            return

        bank.append(
            CorrectionMemoryItem(
                key=x_norm[:],
                label=label,
                residual_logit=residual_logit,
                activity=abs(residual_logit),
            )
        )

    def _merge_multiclass_correction(
        self,
        bank: List[CorrectionMemoryItem],
        x_norm: List[float],
        label: int,
        residual_vector: List[float],
        activity: float,
        focus_source: int,
        focus_rival: int,
    ):
        best_idx = -1
        best_sim = -2.0
        for idx, item in enumerate(bank):
            if item.label != label or item.residual_vector is None:
                continue
            if item.focus_source != focus_source or item.focus_rival != focus_rival:
                continue
            sim = self._cosine(x_norm, item.key)
            if sim > best_sim:
                best_sim = sim
                best_idx = idx

        if best_idx >= 0 and best_sim >= max(0.88, self.sim_threshold - 0.03):
            target = bank[best_idx]
            alpha = min(self.merge_alpha, 1.0 / (target.support + 1.0))
            target.key = [(1.0 - alpha) * left + alpha * right for left, right in zip(target.key, x_norm)]
            target.key = self._normalize(target.key)
            target.residual_vector = [
                (1.0 - alpha) * left + alpha * right for left, right in zip(target.residual_vector or residual_vector, residual_vector)
            ]
            target.activity = max(target.activity, activity)
            target.support += 1
            return

        bank.append(
            CorrectionMemoryItem(
                key=x_norm[:],
                label=label,
                residual_logit=0.0,
                activity=activity,
                residual_vector=residual_vector[:],
                focus_source=focus_source,
                focus_rival=focus_rival,
            )
        )

    def _trim_anchor_memory(self, budget_by_label: Dict[int, int]):
        for label, bank in self.anchor_memory_by_label.items():
            bank.sort(key=lambda item: item.activity, reverse=True)
            self.anchor_memory_by_label[label] = bank[: budget_by_label.get(label, len(bank))]

    def _trim_correction_memory(self, budget: int):
        self.correction_memory.sort(key=lambda item: item.activity, reverse=True)
        self.correction_memory = self.correction_memory[:budget]

    def _build_binary_memory(self, x_train: List[List[float]], y_train: List[int]):
        total_budget = self._effective_memory_budget(len(x_train))
        anchor_budget = max(48, int(total_budget * 0.6))
        correction_budget = max(24, total_budget - anchor_budget)
        anchor_limits = self._anchor_limits(y_train, anchor_budget)

        self.anchor_memory_by_label = {label: [] for label in sorted(set(y_train))}
        self.correction_memory = []

        anchor_candidates: Dict[int, List[Tuple[float, List[float]]]] = {
            label: [] for label in self.anchor_memory_by_label
        }
        correction_candidates: List[Tuple[float, int, float, List[float]]] = []

        for x_row, y_true in zip(x_train, y_train):
            base_prob = self.base_model.predict_one(x_row)
            pred_label = 1 if base_prob >= 0.5 else 0
            true_conf = base_prob if y_true == 1 else 1.0 - base_prob

            anchor_threshold = self.prototype_confidence - (0.03 if y_true == 1 else 0.0)
            if pred_label == y_true and true_conf >= anchor_threshold:
                anchor_candidates[y_true].append((true_conf, x_row))

            if pred_label != y_true or true_conf < self.correction_confidence:
                residual_logit = self._logit(self._target_prob(y_true)) - self._logit(base_prob)
                correction_candidates.append((abs(residual_logit), y_true, residual_logit, x_row))

        for label, candidates in anchor_candidates.items():
            candidates.sort(key=lambda item: item[0], reverse=True)
            keep_count = max(anchor_limits.get(label, 0) * 4, anchor_limits.get(label, 0))
            for activity, x_row in candidates[:keep_count]:
                x_norm = self._normalize(x_row)
                self._merge_anchor(self.anchor_memory_by_label[label], x_norm, label, activity)

        correction_candidates.sort(key=lambda item: item[0], reverse=True)
        keep_corrections = max(correction_budget * 6, correction_budget)
        for _, label, residual_logit, x_row in correction_candidates[:keep_corrections]:
            x_norm = self._normalize(x_row)
            self._merge_correction(self.correction_memory, x_norm, label, residual_logit)

        self._trim_anchor_memory(anchor_limits)
        self._trim_correction_memory(correction_budget)
        self.memory = list(self.correction_memory)
        for bank in self.anchor_memory_by_label.values():
            self.memory.extend(bank)

    def _anchor_prob(self, x_row: List[float]) -> Optional[float]:
        anchor_items: List[Tuple[float, int]] = []
        x_norm = self._normalize(x_row)
        for label, bank in self.anchor_memory_by_label.items():
            for item in bank:
                anchor_items.append((self._cosine(x_norm, item.key), label))

        if not anchor_items:
            return None

        anchor_items.sort(key=lambda pair: pair[0], reverse=True)
        top_items = anchor_items[: min(self.top_k, len(anchor_items))]
        sims = [sim for sim, _ in top_items]
        if sims[0] < 0.2:
            return None
        weights = self._softmax_weights(sims)
        pos_score = sum(weight for weight, (_, label) in zip(weights, top_items) if label == 1)
        return pos_score

    def _correction_delta(self, x_row: List[float]) -> Optional[float]:
        if not self.correction_memory:
            return None

        x_norm = self._normalize(x_row)
        scored = [(self._cosine(x_norm, item.key), item.residual_logit) for item in self.correction_memory]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        top_items = scored[: min(self.top_k, len(scored))]
        sims = [sim for sim, _ in top_items]
        if sims[0] < 0.2:
            return None
        weights = self._softmax_weights(sims)
        return sum(weight * residual for weight, (_, residual) in zip(weights, top_items))

    def _multiclass_correction_delta(self, x_row: List[float], base_probs: Optional[List[float]] = None) -> Optional[List[float]]:
        if base_probs is None:
            base_probs = self._base_multiclass_probs(x_row)
        if len(base_probs) < 2:
            return None

        ranked = sorted(range(len(base_probs)), key=lambda idx: base_probs[idx], reverse=True)
        focus_source = ranked[0]
        focus_rival = ranked[1]
        valid_items: List[Tuple[float, CorrectionMemoryItem]] = []
        exact_pair_items = [
            item
            for item in self.correction_memory
            if item.residual_vector is not None and item.focus_source == focus_source and item.focus_rival == focus_rival
        ]
        for item in exact_pair_items:
            pair_weight = 1.0 if self._is_priority_pair(focus_source, focus_rival) else 0.9
            valid_items.append((pair_weight, item))

        fallback_labels = [focus_rival, focus_source]
        for label in fallback_labels:
            for item in self.correction_memory_by_label.get(label, []):
                if item.residual_vector is None:
                    continue
                if item.focus_source == focus_source and item.focus_rival == focus_rival:
                    continue
                fallback_weight = 0.55
                if item.focus_source == focus_source or item.focus_rival == focus_rival:
                    fallback_weight = 0.72
                if self._is_priority_pair(focus_source, focus_rival):
                    if item.label in self.priority_class_ids:
                        fallback_weight += 0.12
                    if item.focus_source in self.priority_class_ids or item.focus_rival in self.priority_class_ids:
                        fallback_weight += 0.08
                valid_items.append((fallback_weight, item))

        if not valid_items:
            return None

        x_norm = self._normalize(x_row)
        scored = [
            (self._cosine(x_norm, item.key) * pair_weight, item.residual_vector or [])
            for pair_weight, item in valid_items
        ]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        top_items = scored[: min(self.top_k, len(scored))]
        sims = [sim for sim, _ in top_items]
        if sims[0] < 0.2:
            return None
        weights = self._softmax_weights(sims)
        delta = [0.0] * self.num_classes
        for weight, (_, residual_vector) in zip(weights, top_items):
            for class_id, value in enumerate(residual_vector):
                delta[class_id] += weight * value
        return delta

    def _in_uncertainty_band(self, base_prob: float) -> bool:
        return self.uncertainty_low <= base_prob <= self.uncertainty_high

    def _compose_binary_prob(
        self,
        base_prob: float,
        anchor_prob: Optional[float],
        correction_delta: Optional[float],
        anchor_weight: Optional[float] = None,
        correction_weight: Optional[float] = None,
        anchor_direction: Optional[float] = None,
        correction_direction: Optional[float] = None,
    ) -> float:
        if not self._in_uncertainty_band(base_prob):
            return base_prob

        a_weight = self.anchor_weight if anchor_weight is None else anchor_weight
        c_weight = self.correction_weight if correction_weight is None else correction_weight
        a_dir = self.anchor_direction if anchor_direction is None else anchor_direction
        c_dir = self.correction_direction if correction_direction is None else correction_direction

        combined_logit = self._logit(base_prob)
        prior_logit = self._logit(self.class_priors.get(1, 0.5))

        if anchor_prob is not None and a_weight > 1e-12:
            centered_anchor = self._logit(anchor_prob) - prior_logit
            combined_logit += a_dir * a_weight * centered_anchor

        if correction_delta is not None and c_weight > 1e-12:
            combined_logit += c_dir * c_weight * self.correction_scale * correction_delta

        return self._sigmoid(combined_logit)

    def _base_multiclass_probs(self, x_row: List[float]) -> List[float]:
        if not self.base_models:
            return [self.class_priors.get(class_id, 1.0 / max(1, self.num_classes)) for class_id in range(self.num_classes)]

        logits = [self._logit(model.predict_one(x_row)) for model in self.base_models]
        return self._softmax(logits)

    def _multiclass_anchor_probs(self, x_row: List[float]) -> Optional[List[float]]:
        anchor_items: List[Tuple[float, int]] = []
        x_norm = self._normalize(x_row)
        for label, bank in self.anchor_memory_by_label.items():
            for item in bank:
                anchor_items.append((self._cosine(x_norm, item.key), label))

        if not anchor_items:
            return None

        anchor_items.sort(key=lambda pair: pair[0], reverse=True)
        top_items = anchor_items[: min(self.top_k, len(anchor_items))]
        sims = [sim for sim, _ in top_items]
        if sims[0] < 0.2:
            return None

        weights = self._softmax_weights(sims)
        class_probs = [0.0] * self.num_classes
        for weight, (_, label) in zip(weights, top_items):
            class_probs[label] += weight
        total = sum(class_probs) + 1e-12
        return [value / total for value in class_probs]

    def _compose_multiclass_probs(
        self,
        base_probs: List[float],
        anchor_probs: Optional[List[float]],
        anchor_weight: Optional[float] = None,
        correction_delta: Optional[List[float]] = None,
        correction_weight: Optional[float] = None,
    ) -> List[float]:
        a_weight = self.anchor_weight if anchor_weight is None else anchor_weight
        c_weight = self.correction_weight if correction_weight is None else correction_weight
        if (anchor_probs is None or a_weight <= 1e-12) and (correction_delta is None or c_weight <= 1e-12):
            return base_probs
        if max(base_probs) > self.uncertainty_high:
            return base_probs

        logits = [math.log(min(max(prob, 1e-6), 1.0)) for prob in base_probs]
        prior = [self.class_priors.get(class_id, 1.0 / max(1, self.num_classes)) for class_id in range(self.num_classes)]
        prior_logits = [math.log(min(max(prob, 1e-6), 1.0)) for prob in prior]

        if anchor_probs is not None and a_weight > 1e-12:
            anchor_logits = [math.log(min(max(prob, 1e-6), 1.0)) for prob in anchor_probs]
            logits = [
                logit + a_weight * (anchor_logit - prior_logit)
                for logit, anchor_logit, prior_logit in zip(logits, anchor_logits, prior_logits)
            ]

        if correction_delta is not None and c_weight > 1e-12:
            logits = [
                logit + c_weight * self.correction_scale * delta
                for logit, delta in zip(logits, correction_delta)
            ]

        return self._softmax(logits)

    def _f1_macro(self, y_true: List[int], y_pred: List[int]) -> float:
        classes = sorted(set(y_true) | set(y_pred))
        scores = []
        for class_id in classes:
            tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == class_id and yp == class_id)
            fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != class_id and yp == class_id)
            fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == class_id and yp != class_id)
            precision = tp / (tp + fp) if tp + fp > 0 else 0.0
            recall = tp / (tp + fn) if tp + fn > 0 else 0.0
            if precision + recall == 0:
                scores.append(0.0)
            else:
                scores.append(2.0 * precision * recall / (precision + recall))
        return sum(scores) / max(1, len(scores))

    def _selected_class_f1(self, y_true: List[int], y_pred: List[int], class_ids: List[int]) -> float:
        if not class_ids:
            return 0.0
        scores = []
        for class_id in class_ids:
            tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == class_id and yp == class_id)
            fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != class_id and yp == class_id)
            fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == class_id and yp != class_id)
            precision = tp / (tp + fp) if tp + fp > 0 else 0.0
            recall = tp / (tp + fn) if tp + fn > 0 else 0.0
            if precision + recall == 0:
                scores.append(0.0)
            else:
                scores.append(2.0 * precision * recall / (precision + recall))
        return sum(scores) / max(1, len(scores))

    def _binary_stats(self, y_true: List[int], probs: List[float], threshold: float) -> Dict[str, float]:
        y_pred = [1 if prob >= threshold else 0 for prob in probs]
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
        tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)

        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        specificity = tn / (tn + fp) if tn + fp > 0 else 0.0
        accuracy = (tp + tn) / max(1, len(y_true))
        balanced_accuracy = (recall + specificity) / 2.0
        macro_f1 = self._f1_macro(y_true, y_pred)
        return {
            "threshold": threshold,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "precision_pos": precision,
            "recall_pos": recall,
            "specificity": specificity,
            "balanced_accuracy": balanced_accuracy,
        }

    def _threshold_score(self, stats: Dict[str, float]) -> Tuple[float, ...]:
        violations = 0
        if stats["recall_pos"] < self.min_recall:
            violations += 1
        if stats["precision_pos"] < self.min_precision:
            violations += 1
        if stats["specificity"] < self.min_specificity:
            violations += 1

        objective = self.optimize_for.lower()
        if objective == "recall_priority":
            primary = (
                stats["recall_pos"],
                stats["balanced_accuracy"],
                stats["precision_pos"],
                stats["accuracy"],
            )
        elif objective == "balanced_accuracy":
            primary = (
                stats["balanced_accuracy"],
                stats["recall_pos"],
                stats["precision_pos"],
                stats["accuracy"],
            )
        elif objective == "clinical_warning":
            primary = (
                stats["recall_pos"],
                stats["precision_pos"],
                stats["balanced_accuracy"],
                stats["specificity"],
            )
        else:
            primary = (
                stats["macro_f1"],
                stats["balanced_accuracy"],
                stats["recall_pos"],
                stats["accuracy"],
            )

        return (-violations,) + primary + (-abs(stats["threshold"] - 0.5),)

    def _multiclass_stats(self, y_true: List[int], probs: List[List[float]]) -> Dict[str, float]:
        y_pred = [max(range(len(prob)), key=lambda idx: prob[idx]) for prob in probs]
        accuracy = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp) / max(1, len(y_true))
        macro_f1 = self._f1_macro(y_true, y_pred)
        stats = {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
        }
        if self.priority_class_ids:
            stats["priority_f1_macro"] = self._selected_class_f1(y_true, y_pred, self.priority_class_ids)
        return stats

    def _multiclass_score(self, stats: Dict[str, float]) -> Tuple[float, ...]:
        if self.priority_class_ids:
            return (stats["macro_f1"], stats.get("priority_f1_macro", 0.0), stats["accuracy"])
        return (stats["macro_f1"], stats["accuracy"])

    def _tune_binary_hyperparams(self, x_val: List[List[float]], y_val: List[int]):
        base_probs = [self.base_model.predict_one(x_row) for x_row in x_val]
        anchor_probs = [self._anchor_prob(x_row) for x_row in x_val]
        correction_deltas = [self._correction_delta(x_row) for x_row in x_val]

        anchor_branch_scores = [
            (self.class_priors.get(1, 0.5) if prob is None else prob)
            for prob in anchor_probs
        ]
        correction_branch_scores = []
        for base_prob, delta in zip(base_probs, correction_deltas):
            if delta is None:
                correction_branch_scores.append(base_prob)
            else:
                correction_branch_scores.append(
                    self._sigmoid(self._logit(base_prob) + self.correction_scale * delta)
                )

        anchor_auc = self._binary_auc(y_val, anchor_branch_scores)
        correction_auc = self._binary_auc(y_val, correction_branch_scores)
        self.anchor_direction = 1.0 if anchor_auc >= 0.5 else -1.0
        self.correction_direction = 1.0 if correction_auc >= 0.5 else -1.0

        uncertainty_coverage = sum(1 for prob in base_probs if self._in_uncertainty_band(prob)) / max(1, len(base_probs))
        memory_available = sum(
            1
            for anchor_prob, delta in zip(anchor_probs, correction_deltas)
            if anchor_prob is not None or delta is not None
        ) / max(1, len(base_probs))

        best_score: Optional[Tuple[float, ...]] = None
        best_stats: Dict[str, float] = {}
        best_anchor_weight = 0.0
        best_correction_weight = 0.0

        anchor_grid = [0.0, 0.1, 0.2, 0.35]
        correction_grid = [0.0, 0.1, 0.2, 0.35, 0.5]
        grid_min = int(self.threshold_min * 100)
        grid_max = int(self.threshold_max * 100)

        for anchor_weight in anchor_grid:
            for correction_weight in correction_grid:
                combined_probs = [
                    self._compose_binary_prob(
                        base_prob=base_prob,
                        anchor_prob=anchor_prob,
                        correction_delta=correction_delta,
                        anchor_weight=anchor_weight,
                        correction_weight=correction_weight,
                        anchor_direction=self.anchor_direction,
                        correction_direction=self.correction_direction,
                    )
                    for base_prob, anchor_prob, correction_delta in zip(base_probs, anchor_probs, correction_deltas)
                ]

                for threshold_idx in range(grid_min, grid_max + 1):
                    threshold = threshold_idx / 100.0
                    stats = self._binary_stats(y_val, combined_probs, threshold)
                    score = self._threshold_score(stats)
                    if best_score is None or score > best_score:
                        best_score = score
                        best_stats = stats
                        best_anchor_weight = anchor_weight
                        best_correction_weight = correction_weight

        self.anchor_weight = best_anchor_weight
        self.correction_weight = best_correction_weight
        self.decision_threshold = best_stats["threshold"]
        self.threshold_selection_summary = {
            "optimize_for": self.optimize_for,
            "threshold": self.decision_threshold,
            "anchor_weight": self.anchor_weight,
            "correction_weight": self.correction_weight,
            **best_stats,
        }
        self.memory_diagnostics = {
            "anchor_memory_size": float(sum(len(bank) for bank in self.anchor_memory_by_label.values())),
            "correction_memory_size": float(len(self.correction_memory)),
            "anchor_branch_auc": anchor_auc,
            "correction_branch_auc": correction_auc,
            "anchor_direction": self.anchor_direction,
            "correction_direction": self.correction_direction,
            "uncertainty_coverage": uncertainty_coverage,
            "memory_available_rate": memory_available,
        }

    def _fit_multiclass_base(self, x_train: List[List[float]], y_train: List[int]):
        self.base_models = []
        for class_id in range(self.num_classes):
            model = BalancedOnlineLogisticRegressor(
                epochs=self.base_model.epochs,
                learning_rate=self.base_model.learning_rate,
                l2=self.base_model.l2,
                positive_weight_scale=self.base_model.positive_weight_scale,
                seed=self.seed + class_id,
            )
            binary_targets = [1 if y_value == class_id else 0 for y_value in y_train]
            model.fit(x_train, binary_targets)
            self.base_models.append(model)

    def _build_multiclass_memory(
        self,
        x_train: List[List[float]],
        y_train: List[int],
        sample_groups: Optional[List[int]] = None,
    ):
        total_budget = self._effective_memory_budget(len(x_train))
        anchor_budget = max(48, int(total_budget * 0.65))
        correction_budget = max(24, total_budget - anchor_budget)
        anchor_limits = self._anchor_limits(y_train, anchor_budget)
        self.anchor_memory_by_label = {label: [] for label in sorted(set(y_train))}
        self.correction_memory = []
        self.correction_memory_by_label = {label: [] for label in sorted(set(y_train))}

        anchor_candidates: Dict[int, List[Tuple[float, int, List[float]]]] = {label: [] for label in self.anchor_memory_by_label}
        correction_candidates: List[Tuple[float, int, int, int, List[float], List[float]]] = []
        groups = sample_groups or list(range(len(x_train)))

        for x_row, y_true, group_id in zip(x_train, y_train, groups):
            base_probs = self._base_multiclass_probs(x_row)
            ranked = sorted(range(len(base_probs)), key=lambda idx: base_probs[idx], reverse=True)
            pred_label = ranked[0]
            runner_up = ranked[1] if len(ranked) > 1 else ranked[0]
            true_conf = base_probs[y_true]
            other_scores = [prob for idx, prob in enumerate(base_probs) if idx != y_true]
            margin = true_conf - max(other_scores or [0.0])

            if pred_label == y_true and true_conf >= max(self.prototype_confidence, 0.7) and margin >= 0.12:
                score = true_conf + margin
                anchor_candidates[y_true].append((score, group_id, x_row))

            correction_focus_rival = runner_up
            correction_label = None
            correction_activity = None
            if pred_label != y_true and runner_up == y_true:
                correction_label = y_true
                correction_activity = (base_probs[pred_label] - base_probs[y_true]) + (1.0 - true_conf)
            elif pred_label == y_true and (base_probs[pred_label] - base_probs[runner_up]) < 0.12:
                correction_label = y_true
                correction_activity = 0.5 * (0.12 - (base_probs[pred_label] - base_probs[runner_up])) + 0.25
            elif pred_label != y_true and true_conf < max(0.58, self.correction_confidence - 0.1):
                correction_label = y_true
                correction_focus_rival = pred_label
                correction_activity = (1.0 - true_conf) * 0.5

            if correction_label is not None and correction_focus_rival != correction_label:
                residual_vector = [0.0] * self.num_classes
                target_boost = math.log(0.88) - math.log(min(max(base_probs[correction_label], 1e-6), 1.0))
                rival_reduce = math.log(0.06) - math.log(min(max(base_probs[correction_focus_rival], 1e-6), 1.0))
                residual_vector[correction_label] = target_boost
                residual_vector[correction_focus_rival] = rival_reduce
                if self._is_priority_pair(pred_label, correction_focus_rival):
                    residual_vector[correction_label] *= 1.25
                    residual_vector[correction_focus_rival] *= 1.35
                priority_bonus = self._correction_priority_bonus(correction_label, pred_label, correction_focus_rival)
                correction_score = abs(target_boost) + abs(rival_reduce)
                correction_candidates.append(
                    (
                        correction_score + max(0.0, correction_activity or 0.0) + priority_bonus,
                        correction_label,
                        group_id,
                        pred_label,
                        correction_focus_rival,
                        residual_vector,
                        x_row,
                    )
                )

        anchor_group_counts: Dict[Tuple[int, int], int] = {}
        for label, candidates in anchor_candidates.items():
            candidates.sort(key=lambda item: item[0], reverse=True)
            keep_count = max(anchor_limits.get(label, 0) * 4, anchor_limits.get(label, 0))
            for activity, group_id, x_row in candidates[:keep_count]:
                group_key = (label, group_id)
                if anchor_group_counts.get(group_key, 0) >= 1:
                    continue
                x_norm = self._normalize(x_row)
                self._merge_anchor(self.anchor_memory_by_label[label], x_norm, label, activity)
                anchor_group_counts[group_key] = anchor_group_counts.get(group_key, 0) + 1

        correction_group_counts: Dict[Tuple[int, int], int] = {}
        correction_candidates.sort(key=lambda item: item[0], reverse=True)
        keep_corrections = max(correction_budget * 6, correction_budget)
        for activity, label, group_id, focus_source, focus_rival, residual_vector, x_row in correction_candidates[:keep_corrections]:
            group_key = (label, group_id, focus_source, focus_rival)
            if correction_group_counts.get(group_key, 0) >= 1:
                continue
            x_norm = self._normalize(x_row)
            self._merge_multiclass_correction(
                self.correction_memory,
                x_norm,
                label,
                residual_vector,
                activity,
                focus_source=focus_source,
                focus_rival=focus_rival,
            )
            correction_group_counts[group_key] = correction_group_counts.get(group_key, 0) + 1

        self._trim_anchor_memory(anchor_limits)
        self._trim_correction_memory(correction_budget)
        self.correction_memory_by_label = {label: [] for label in sorted(set(y_train))}
        for item in self.correction_memory:
            self.correction_memory_by_label.setdefault(item.label, []).append(item)
        self.memory = []
        self.memory.extend(self.correction_memory)
        for bank in self.anchor_memory_by_label.values():
            self.memory.extend(bank)

    def _tune_multiclass_hyperparams(self, x_val: List[List[float]], y_val: List[int]):
        base_probs = [self._base_multiclass_probs(x_row) for x_row in x_val]
        anchor_probs = [self._multiclass_anchor_probs(x_row) for x_row in x_val]
        correction_deltas = [self._multiclass_correction_delta(x_row, base_prob) for x_row, base_prob in zip(x_val, base_probs)]
        anchor_branch_probs = [
            ([self.class_priors.get(class_id, 1.0 / max(1, self.num_classes)) for class_id in range(self.num_classes)] if prob is None else prob)
            for prob in anchor_probs
        ]
        correction_branch_probs = [
            (
                base_prob
                if correction_delta is None
                else self._compose_multiclass_probs(base_prob, None, anchor_weight=0.0, correction_delta=correction_delta, correction_weight=1.0)
            )
            for base_prob, correction_delta in zip(base_probs, correction_deltas)
        ]

        best_score: Optional[Tuple[float, ...]] = None
        best_stats: Dict[str, float] = {}
        best_anchor_weight = 0.0
        best_correction_weight = 0.0

        for anchor_weight in [0.0, 0.1, 0.2, 0.35, 0.5]:
            for correction_weight in [0.0, 0.1, 0.2, 0.35, 0.5]:
                combined_probs = [
                    self._compose_multiclass_probs(
                        base_prob,
                        anchor_prob,
                        anchor_weight=anchor_weight,
                        correction_delta=correction_delta,
                        correction_weight=correction_weight,
                    )
                    for base_prob, anchor_prob, correction_delta in zip(base_probs, anchor_probs, correction_deltas)
                ]
                stats = self._multiclass_stats(y_val, combined_probs)
                score = self._multiclass_score(stats)
                if best_score is None or score > best_score:
                    best_score = score
                    best_stats = stats
                    best_anchor_weight = anchor_weight
                    best_correction_weight = correction_weight

        self.anchor_weight = best_anchor_weight
        self.correction_weight = best_correction_weight
        self.decision_threshold = 0.5
        self.threshold_selection_summary = {
            "optimize_for": self.optimize_for,
            "anchor_weight": self.anchor_weight,
            "correction_weight": self.correction_weight,
            **best_stats,
        }
        self.memory_diagnostics = {
            "anchor_memory_size": float(sum(len(bank) for bank in self.anchor_memory_by_label.values())),
            "correction_memory_size": float(len(self.correction_memory)),
            "priority_label_correction_memory_size": float(
                sum(
                    1
                    for item in self.correction_memory
                    if item.label in self.priority_class_ids or item.focus_rival in self.priority_class_ids
                )
            ),
            "priority_correction_memory_size": float(
                sum(
                    1
                    for item in self.correction_memory
                    if self._is_priority_pair(item.focus_source, item.focus_rival)
                )
            ),
            "anchor_branch_f1_macro": self._multiclass_stats(y_val, anchor_branch_probs)["macro_f1"],
            "anchor_branch_accuracy": self._multiclass_stats(y_val, anchor_branch_probs)["accuracy"],
            "correction_branch_f1_macro": self._multiclass_stats(y_val, correction_branch_probs)["macro_f1"],
            "correction_branch_accuracy": self._multiclass_stats(y_val, correction_branch_probs)["accuracy"],
            "uncertainty_coverage": sum(1 for prob in base_probs if max(prob) <= self.uncertainty_high) / max(1, len(base_probs)),
            "memory_available_rate": sum(
                1 for anchor_prob, correction_delta in zip(anchor_probs, correction_deltas) if anchor_prob is not None or correction_delta is not None
            ) / max(1, len(anchor_probs)),
        }

    def fit(
        self,
        x_train: List[List[float]],
        y_train: List[int],
        x_val: Optional[List[List[float]]] = None,
        y_val: Optional[List[int]] = None,
        sample_groups: Optional[List[int]] = None,
        label_names: Optional[List[str]] = None,
    ):
        self.num_classes = max(y_train) + 1
        self._set_label_context(label_names)
        counts = Counter(y_train)
        total = len(y_train)
        self.class_priors = {
            label: counts.get(label, 0) / max(1, total) for label in counts
        }

        if self.num_classes == 2:
            self._binary_hybrid_enabled = True
            self.base_model.fit(x_train, y_train)
            self.base_models = []
            self._build_binary_memory(x_train, y_train)
            self.anchor_weight = 0.0
            self.correction_weight = 0.0
            self.decision_threshold = 0.5
            if x_val is not None and y_val is not None:
                self._tune_binary_hyperparams(x_val, y_val)
            return

        self._binary_hybrid_enabled = False
        self._fit_multiclass_base(x_train, y_train)
        self._build_multiclass_memory(x_train, y_train, sample_groups=sample_groups)
        self.anchor_weight = 0.0
        self.correction_weight = 0.0
        self.decision_threshold = 0.5
        if x_val is not None and y_val is not None:
            self._tune_multiclass_hyperparams(x_val, y_val)

    def predict_proba(self, x_data: List[List[float]]) -> List[List[float]]:
        if self._binary_hybrid_enabled:
            probs = []
            for x_row in x_data:
                base_prob = self.base_model.predict_one(x_row)
                anchor_prob = self._anchor_prob(x_row)
                correction_delta = self._correction_delta(x_row)
                final_prob = self._compose_binary_prob(base_prob, anchor_prob, correction_delta)
                probs.append([1.0 - final_prob, final_prob])
            return probs

        probs = []
        for x_row in x_data:
            base_probs = self._base_multiclass_probs(x_row)
            anchor_probs = self._multiclass_anchor_probs(x_row)
            correction_delta = self._multiclass_correction_delta(x_row, base_probs)
            probs.append(self._compose_multiclass_probs(base_probs, anchor_probs, correction_delta=correction_delta))
        return probs

    def predict(self, x_data: List[List[float]]) -> List[int]:
        probs = self.predict_proba(x_data)
        if self._binary_hybrid_enabled:
            return [1 if prob[1] >= self.decision_threshold else 0 for prob in probs]
        return [max(range(len(prob)), key=lambda idx: prob[idx]) for prob in probs]

    def predict_base_proba(self, x_data: List[List[float]]) -> List[List[float]]:
        if not self._binary_hybrid_enabled:
            return [self._base_multiclass_probs(x_row) for x_row in x_data]
        return [[1.0 - self.base_model.predict_one(x_row), self.base_model.predict_one(x_row)] for x_row in x_data]
