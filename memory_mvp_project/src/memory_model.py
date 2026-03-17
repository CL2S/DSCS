import math
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class MemoryItem:
    key: List[float]
    value: List[float]
    label: int
    activity: float


class DynamicMemoryClassifier:
    def __init__(
        self,
        top_k: int = 16,
        sim_threshold: float = 0.8,
        merge_alpha: float = 0.2,
        decay: float = 0.997,
        forget_threshold: float = 0.1,
        max_memory: int = 5000,
    ):
        self.top_k = top_k
        self.sim_threshold = sim_threshold
        self.merge_alpha = merge_alpha
        self.decay = decay
        self.forget_threshold = forget_threshold
        self.max_memory = max_memory
        self.memory: List[MemoryItem] = []
        self.num_classes = 0

    def _norm(self, v: List[float]) -> float:
        return math.sqrt(sum(x * x for x in v)) + 1e-12

    def _normalize(self, v: List[float]) -> List[float]:
        n = self._norm(v)
        return [x / n for x in v]

    def _cosine(self, a: List[float], b: List[float]) -> float:
        na = self._normalize(a)
        nb = self._normalize(b)
        return sum(x * y for x, y in zip(na, nb))

    def _decay_and_forget(self):
        for item in self.memory:
            item.activity *= self.decay
        self.memory = [m for m in self.memory if m.activity >= self.forget_threshold]

    def _enforce_capacity(self):
        if len(self.memory) <= self.max_memory:
            return
        self.memory.sort(key=lambda x: x.activity, reverse=True)
        self.memory = self.memory[: self.max_memory]

    def _read(self, x: List[float]):
        if not self.memory:
            return None, None, None

        sims = [self._cosine(x, m.key) for m in self.memory]
        pairs = list(enumerate(sims))
        pairs.sort(key=lambda p: p[1], reverse=True)
        top_pairs = pairs[: min(self.top_k, len(pairs))]

        max_sim = top_pairs[0][1]
        exp_scores = [math.exp(s - max_sim) for _, s in top_pairs]
        denom = sum(exp_scores) + 1e-12
        weights = [e / denom for e in exp_scores]

        read_vec = [0.0] * len(x)
        class_scores = [0.0] * self.num_classes

        for (idx, _), w in zip(top_pairs, weights):
            mem = self.memory[idx]
            read_vec = [rv + w * mv for rv, mv in zip(read_vec, mem.value)]
            class_scores[mem.label] += w

        best_idx, best_sim = pairs[0]
        return read_vec, class_scores, (best_idx, best_sim)

    def _write(self, x: List[float], label: int, best_match: Optional[Tuple[int, float]]):
        if best_match is None:
            self.memory.append(MemoryItem(key=x[:], value=x[:], label=label, activity=1.0))
            return

        best_idx, best_sim = best_match
        if best_sim >= self.sim_threshold:
            target = self.memory[best_idx]
            target.key = [
                (1.0 - self.merge_alpha) * a + self.merge_alpha * b
                for a, b in zip(target.key, x)
            ]
            target.value = [
                (1.0 - self.merge_alpha) * a + self.merge_alpha * b
                for a, b in zip(target.value, x)
            ]
            target.key = self._normalize(target.key)
            target.value = self._normalize(target.value)
            target.label = label
            target.activity = 1.0
        else:
            self.memory.append(MemoryItem(key=x[:], value=x[:], label=label, activity=1.0))

    def fit(self, x_train: List[List[float]], y_train: List[int]):
        self.num_classes = max(y_train) + 1
        self.memory = []

        for x, y in zip(x_train, y_train):
            xn = self._normalize(x)
            _, _, best_match = self._read(xn)
            self._write(xn, y, best_match)
            self._decay_and_forget()
            self._enforce_capacity()

    def predict_proba(self, x_data: List[List[float]]) -> List[List[float]]:
        probs: List[List[float]] = []
        for x in x_data:
            xn = self._normalize(x)
            _, class_scores, _ = self._read(xn)
            if class_scores is None:
                prob = [1.0 / self.num_classes] * self.num_classes
            else:
                total = sum(class_scores)
                if total <= 1e-12:
                    prob = [1.0 / self.num_classes] * self.num_classes
                else:
                    prob = [s / total for s in class_scores]
            probs.append(prob)
        return probs

    def predict(self, x_data: List[List[float]]) -> List[int]:
        prob = self.predict_proba(x_data)
        return [max(range(len(p)), key=lambda i: p[i]) for p in prob]
