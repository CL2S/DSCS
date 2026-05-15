import copy
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch

from src.manifold_memory import (
    AttentionMemoryReader,
    ManifoldMemoryConfig,
    ManifoldMemoryItem,
    TorchAttentionReadout,
    TorchManifoldEncodingOutput,
    _normalize,
)


@dataclass
class ComponentReadResult:
    name: str
    readout: torch.Tensor
    confidence: float
    confidence_tensor: Optional[torch.Tensor]
    matched_indices: List[int]
    matched_labels: List[int]
    top_label: int
    max_similarity: float
    retrieval_source: str = "hot"
    archive_used: bool = False
    archive_confidence: float = 0.0
    archive_confidence_tensor: Optional[torch.Tensor] = None
    archive_weight: float = 0.0
    archive_weight_tensor: Optional[torch.Tensor] = None


class MemoryComponent:
    def __init__(self, name: str, config: ManifoldMemoryConfig):
        self.name = name
        self.config = copy.deepcopy(config)
        self.reader = AttentionMemoryReader(self.config)
        self.memory_bank: List[ManifoldMemoryItem] = []

    def _patient_id(self, metadata: Dict[str, object]) -> Optional[float]:
        if "stay_id" in metadata:
            return float(metadata["stay_id"])
        if "series_index" in metadata:
            return float(metadata["series_index"])
        return None

    def _similarity(self, left: Sequence[float], right: Sequence[float]) -> float:
        return sum(left_value * right_value for left_value, right_value in zip(left, right))

    def _label_count(self, label: int, bank: Optional[Sequence[ManifoldMemoryItem]] = None) -> int:
        items = bank if bank is not None else self.memory_bank
        return sum(1 for item in items if item.label == label)

    def _patient_label_count(
        self,
        label: int,
        patient_id: Optional[float],
        bank: Optional[Sequence[ManifoldMemoryItem]] = None,
    ) -> int:
        if patient_id is None:
            return 0
        items = bank if bank is not None else self.memory_bank
        return sum(
            1
            for item in items
            if item.label == label and self._patient_id(item.metadata) == patient_id
        )

    def _priority(self, item: ManifoldMemoryItem) -> float:
        freshness = float(item.metadata.get("freshness", 1.0)) if item.metadata else 1.0
        confidence = float(item.metadata.get("write_confidence", 1.0)) if item.metadata else 1.0
        prototype_bonus = 1.0 + 0.05 * float(item.metadata.get("prototype_children", 0.0))
        return (item.activity * freshness * confidence + 0.08 * math.log1p(item.support)) * prototype_bonus

    def _find_best_merge_candidate(
        self,
        key: Sequence[float],
        label: int,
        metadata: Dict[str, object],
        bank: Optional[Sequence[ManifoldMemoryItem]] = None,
        similarity_threshold: Optional[float] = None,
    ) -> tuple[int, float]:
        items = list(bank if bank is not None else self.memory_bank)
        patient_id = self._patient_id(metadata)
        best_index = -1
        best_similarity = -2.0
        best_adjusted_score = -2.0

        for index, item in enumerate(items):
            if self.config.same_label_merge_only and item.label != label:
                continue

            similarity = self._similarity(key, item.key)
            adjusted_score = similarity - self.config.support_penalty * math.log1p(item.support)
            if patient_id is not None and self._patient_id(item.metadata) == patient_id:
                adjusted_score -= self.config.collapse_penalty

            if adjusted_score > best_adjusted_score:
                best_index = index
                best_similarity = similarity
                best_adjusted_score = adjusted_score

        threshold = similarity_threshold if similarity_threshold is not None else self.config.similarity_threshold
        if best_index < 0 or best_similarity < threshold:
            return -1, best_similarity
        return best_index, best_similarity

    def _merge_item(self, target: ManifoldMemoryItem, key: Sequence[float], value: Sequence[float], metadata: Dict[str, object], activity: float):
        alpha = min(self.config.merge_alpha, 1.0 / (target.support + 1.0))
        target.key = _normalize([(1.0 - alpha) * left + alpha * right for left, right in zip(target.key, key)])
        target.value = [(1.0 - alpha) * left + alpha * right for left, right in zip(target.value, value)]
        target.activity = max(target.activity, activity) + 0.02
        target.support += 1
        merged_children = int(target.metadata.get("prototype_children", 0.0)) + int(metadata.get("prototype_children", 0.0))
        target.metadata.update(metadata)
        target.metadata["prototype_children"] = merged_children

    def write(self, encoding: TorchManifoldEncodingOutput, label: int, metadata: Dict[str, object], activity: float = 1.0):
        key = encoding.key.detach().cpu().view(-1).tolist()
        value = encoding.value.detach().cpu().view(-1).tolist()
        patient_id = self._patient_id(metadata)
        label_count = self._label_count(label)
        patient_label_count = self._patient_label_count(label, patient_id)

        force_new = (
            label_count < self.config.min_label_memory
            and patient_label_count < self.config.max_patient_label_memory
        )
        best_index, _ = self._find_best_merge_candidate(key, label, metadata)

        if not force_new and best_index >= 0:
            self._merge_item(self.memory_bank[best_index], key, value, metadata, activity)
            return

        self.memory_bank.append(
            ManifoldMemoryItem(
                key=_normalize(key),
                value=value[:],
                label=label,
                activity=activity,
                metadata=dict(metadata),
            )
        )
        self.trim()

    def _read_from_bank(
        self,
        encoding: TorchManifoldEncodingOutput,
        bank: Sequence[ManifoldMemoryItem],
        label_prior: Optional[torch.Tensor] = None,
    ) -> TorchAttentionReadout:
        return self.reader.read_torch(encoding.query, bank, class_prior=label_prior)

    def read(
        self,
        encoding: TorchManifoldEncodingOutput,
        label_prior: Optional[torch.Tensor] = None,
    ) -> ComponentReadResult:
        torch_readout = self._read_from_bank(encoding, self.memory_bank, label_prior=label_prior)
        return ComponentReadResult(
            name=self.name,
            readout=torch_readout.readout,
            confidence=float(torch_readout.memory_confidence),
            confidence_tensor=torch_readout.memory_confidence_tensor,
            matched_indices=list(torch_readout.matched_indices),
            matched_labels=list(torch_readout.matched_labels),
            top_label=int(torch_readout.top_label),
            max_similarity=float(torch_readout.max_similarity),
        )

    def decay(self):
        for item in self.memory_bank:
            item.activity *= self.config.decay
        self.memory_bank = [item for item in self.memory_bank if item.activity >= self.config.forget_threshold]
        self.trim()

    def _trim_bank(
        self,
        bank: Sequence[ManifoldMemoryItem],
        max_memory: int,
        max_label_memory: int,
    ) -> List[ManifoldMemoryItem]:
        if not bank:
            return []

        by_label: Dict[int, List[ManifoldMemoryItem]] = {}
        for item in bank:
            by_label.setdefault(item.label, []).append(item)

        trimmed: List[ManifoldMemoryItem] = []
        for label_items in by_label.values():
            sorted_items = sorted(label_items, key=self._priority, reverse=True)
            unique_patient_items: List[ManifoldMemoryItem] = []
            overflow_items: List[ManifoldMemoryItem] = []
            seen_patients = set()

            for item in sorted_items:
                patient_id = self._patient_id(item.metadata)
                if patient_id is not None and patient_id not in seen_patients and len(unique_patient_items) < max_label_memory:
                    unique_patient_items.append(item)
                    seen_patients.add(patient_id)
                else:
                    overflow_items.append(item)

            kept = unique_patient_items[:max_label_memory]
            if len(kept) < max_label_memory:
                kept.extend(overflow_items[: max_label_memory - len(kept)])
            trimmed.extend(kept)

        trimmed.sort(key=self._priority, reverse=True)
        return [copy.deepcopy(item) for item in trimmed[:max_memory]]

    def trim(self):
        self.memory_bank = self._trim_bank(
            self.memory_bank,
            max_memory=self.config.max_memory,
            max_label_memory=self.config.max_label_memory,
        )

    def summarize(self) -> Dict[str, float]:
        if not self.memory_bank:
            return {
                "memory_size": 0.0,
                "memory_unique_labels": 0.0,
                "memory_unique_stays": 0.0,
                "memory_max_label_share": 0.0,
                "memory_mean_support": 0.0,
            }

        label_counts: Dict[int, int] = {}
        patient_ids = set()
        support_sum = 0.0
        for item in self.memory_bank:
            label_counts[item.label] = label_counts.get(item.label, 0) + 1
            patient_id = self._patient_id(item.metadata)
            if patient_id is not None:
                patient_ids.add(patient_id)
            support_sum += float(item.support)

        max_label_share = max(label_counts.values()) / max(1, len(self.memory_bank))
        return {
            "memory_size": float(len(self.memory_bank)),
            "memory_unique_labels": float(len(label_counts)),
            "memory_unique_stays": float(len(patient_ids)),
            "memory_max_label_share": float(max_label_share),
            "memory_mean_support": float(support_sum / max(1, len(self.memory_bank))),
        }


class PatternMemory(MemoryComponent):
    pass


class TrajectoryMemory(MemoryComponent):
    pass


class ExperienceMemory(MemoryComponent):
    def __init__(self, name: str, config: ManifoldMemoryConfig):
        super().__init__(name=name, config=config)
        self.archive_bank: List[ManifoldMemoryItem] = []

    def _compress_by_prototype(
        self,
        bank: Sequence[ManifoldMemoryItem],
        similarity_threshold: float,
    ) -> List[ManifoldMemoryItem]:
        if len(bank) <= 1:
            return [copy.deepcopy(item) for item in bank]

        compressed: List[ManifoldMemoryItem] = []
        for item in sorted(bank, key=self._priority, reverse=True):
            merge_index, _ = self._find_best_merge_candidate(
                item.key,
                item.label,
                item.metadata,
                bank=compressed,
                similarity_threshold=similarity_threshold,
            )
            if merge_index >= 0:
                self._merge_item(
                    compressed[merge_index],
                    item.key,
                    item.value,
                    item.metadata,
                    item.activity,
                )
            else:
                compressed.append(copy.deepcopy(item))
        return compressed

    def trim(self):
        hot_limit = max(24, int(self.config.max_memory * 0.6))
        archive_limit = max(12, int(self.config.max_memory * 0.4))
        label_limit = max(4, min(self.config.max_label_memory, hot_limit))

        hot_candidates = self._trim_bank(self.memory_bank, max_memory=self.config.max_memory, max_label_memory=label_limit)
        hot_candidates = self._compress_by_prototype(
            hot_candidates,
            similarity_threshold=min(0.995, self.config.similarity_threshold + 0.03),
        )
        hot_candidates.sort(key=self._priority, reverse=True)
        if len(hot_candidates) <= 16:
            proactive_hot_keep = max(6, int(math.ceil(len(hot_candidates) * 0.6)))
        else:
            proactive_hot_keep = max(12, int(math.ceil(len(hot_candidates) * 0.7)))
        hot_keep = min(hot_limit, proactive_hot_keep, len(hot_candidates))
        self.memory_bank = hot_candidates[:hot_keep]

        overflow = [copy.deepcopy(item) for item in hot_candidates[hot_keep:]]
        archive_candidates = [copy.deepcopy(item) for item in self.archive_bank] + overflow
        archive_candidates = self._trim_bank(
            archive_candidates,
            max_memory=max(archive_limit * 2, archive_limit),
            max_label_memory=max(2, label_limit // 2),
        )
        archive_candidates = self._compress_by_prototype(
            archive_candidates,
            similarity_threshold=min(0.997, self.config.similarity_threshold + 0.05),
        )
        archive_candidates.sort(key=self._priority, reverse=True)
        self.archive_bank = archive_candidates[:archive_limit]
        for item in self.archive_bank:
            item.metadata["storage"] = "archive"
        for item in self.memory_bank:
            item.metadata["storage"] = "hot"

    def read(
        self,
        encoding: TorchManifoldEncodingOutput,
        label_prior: Optional[torch.Tensor] = None,
    ) -> ComponentReadResult:
        hot_read = self._read_from_bank(encoding, self.memory_bank, label_prior=label_prior)
        if not self.archive_bank:
            return ComponentReadResult(
                name=self.name,
                readout=hot_read.readout,
                confidence=float(hot_read.memory_confidence),
                confidence_tensor=hot_read.memory_confidence_tensor,
                matched_indices=list(hot_read.matched_indices),
                matched_labels=list(hot_read.matched_labels),
                top_label=int(hot_read.top_label),
                max_similarity=float(hot_read.max_similarity),
                retrieval_source="hot",
                archive_used=False,
                archive_confidence=0.0,
                archive_confidence_tensor=torch.zeros((), dtype=torch.float32, device=hot_read.readout.device),
                archive_weight=0.0,
                archive_weight_tensor=torch.zeros((), dtype=torch.float32, device=hot_read.readout.device),
            )

        archive_read = self._read_from_bank(encoding, self.archive_bank, label_prior=label_prior)
        hot_confidence = hot_read.memory_confidence_tensor
        archive_confidence = archive_read.memory_confidence_tensor
        archive_need = torch.relu(0.72 - hot_confidence) / 0.72
        archive_advantage = torch.relu(archive_confidence - hot_confidence + 0.04)
        archive_support = torch.relu(archive_confidence - 0.30)
        archive_weight_tensor = torch.clamp(
            0.30 * archive_need + 0.70 * archive_advantage + 0.20 * archive_support,
            min=0.0,
            max=0.45,
        )
        archive_weight = float(archive_weight_tensor.item())

        if archive_weight <= 0.02:
            return ComponentReadResult(
                name=self.name,
                readout=hot_read.readout,
                confidence=float(hot_read.memory_confidence),
                confidence_tensor=hot_read.memory_confidence_tensor,
                matched_indices=list(hot_read.matched_indices),
                matched_labels=list(hot_read.matched_labels),
                top_label=int(hot_read.top_label),
                max_similarity=float(hot_read.max_similarity),
                retrieval_source="hot",
                archive_used=False,
                archive_confidence=float(archive_read.memory_confidence),
                archive_confidence_tensor=archive_read.memory_confidence_tensor,
                archive_weight=archive_weight,
                archive_weight_tensor=archive_weight_tensor,
            )

        hot_weight = 1.0 - archive_weight_tensor
        blended_readout = hot_read.readout * hot_weight + archive_read.readout * archive_weight_tensor
        blended_confidence = max(float(hot_read.memory_confidence), float(archive_read.memory_confidence) * 0.9)
        matched_indices = list(hot_read.matched_indices[:3]) + [-(index + 1) for index in archive_read.matched_indices[:2]]
        matched_labels = list(hot_read.matched_labels[:3]) + list(archive_read.matched_labels[:2])
        top_label = int(hot_read.top_label if hot_read.memory_confidence >= archive_read.memory_confidence else archive_read.top_label)
        max_similarity = max(float(hot_read.max_similarity), float(archive_read.max_similarity))
        return ComponentReadResult(
            name=self.name,
            readout=blended_readout,
            confidence=blended_confidence,
            confidence_tensor=torch.maximum(hot_read.memory_confidence_tensor, archive_read.memory_confidence_tensor * 0.9),
            matched_indices=matched_indices,
            matched_labels=matched_labels,
            top_label=top_label,
            max_similarity=max_similarity,
            retrieval_source="blended",
            archive_used=True,
            archive_confidence=float(archive_read.memory_confidence),
            archive_confidence_tensor=archive_read.memory_confidence_tensor,
            archive_weight=archive_weight,
            archive_weight_tensor=archive_weight_tensor,
        )

    def summarize(self) -> Dict[str, float]:
        summary = super().summarize()
        archive_labels = {item.label for item in self.archive_bank}
        summary.update(
            {
                "archive_size": float(len(self.archive_bank)),
                "archive_unique_labels": float(len(archive_labels)),
                "hot_to_total_ratio": float(len(self.memory_bank) / max(1, len(self.memory_bank) + len(self.archive_bank))),
            }
        )
        return summary
