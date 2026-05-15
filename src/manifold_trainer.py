import copy
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from src.manifold_memory import ManifoldMemoryBlueprint, ManifoldMemoryConfig, TorchManifoldEncodingOutput


@dataclass
class ManifoldTrainerConfig:
    epochs: int = 12
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    aux_base_loss_weight: float = 0.35
    align_loss_weight: float = 0.05
    compact_loss_weight: float = 0.03
    separation_loss_weight: float = 0.03
    temporal_smoothness_weight: float = 0.04
    separation_margin: float = 1.0
    grad_clip: float = 1.0
    device: str = "cpu"
    seed: int = 42


class EndToEndManifoldTrainer(nn.Module):
    """
    First trainable manifold-memory trainer.

    Training strategy:
    - use the GRU manifold encoder to produce query/key/value
    - read from a dynamic memory bank during training
    - optimize both a base classifier and a memory-fused classifier
    - rebuild the memory bank at the end of each epoch using the latest encoder
    """

    def __init__(
        self,
        memory_config: ManifoldMemoryConfig,
        trainer_config: ManifoldTrainerConfig,
        num_classes: int,
        label_names: Optional[List[str]] = None,
    ):
        super().__init__()
        self.memory_config = memory_config
        self.trainer_config = trainer_config
        self.num_classes = num_classes
        self.label_names = label_names or [str(idx) for idx in range(num_classes)]
        self.device = torch.device(trainer_config.device)

        self.manifold = ManifoldMemoryBlueprint(memory_config)
        self.base_classifier = nn.Linear(self.manifold.encoder.embedding_dim, num_classes)
        self.fusion_classifier = nn.Linear(memory_config.fusion_hidden_dim, num_classes)
        self.to(self.device)

        self.sequence_mean: Optional[torch.Tensor] = None
        self.sequence_std: Optional[torch.Tensor] = None
        self.static_mean: Optional[torch.Tensor] = None
        self.static_std: Optional[torch.Tensor] = None
        self.class_weights: Optional[torch.Tensor] = None

        self.best_epoch = 0
        self.training_summary: Dict[str, float] = {}
        self.memory_diagnostics: Dict[str, float] = {}
        self.latest_loss_breakdown: Dict[str, float] = {}

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def _stack_sequences(self, sequences: Sequence[Sequence[Sequence[float]]]) -> torch.Tensor:
        return torch.tensor(sequences, dtype=torch.float32, device=self.device)

    def _stack_static(self, static_data: Sequence[Sequence[float]]) -> torch.Tensor:
        return torch.tensor(static_data, dtype=torch.float32, device=self.device)

    def _fit_normalizers(self, sequences: Sequence[Sequence[Sequence[float]]], static_data: Sequence[Sequence[float]]):
        sequence_tensor = self._stack_sequences(sequences)
        self.sequence_mean = sequence_tensor.mean(dim=(0, 1))
        self.sequence_std = sequence_tensor.std(dim=(0, 1), unbiased=False).clamp_min(1e-6)

        static_tensor = self._stack_static(static_data)
        self.static_mean = static_tensor.mean(dim=0)
        self.static_std = static_tensor.std(dim=0, unbiased=False).clamp_min(1e-6)

    def _normalize_sequence(self, sequence_steps: Sequence[Sequence[float]]) -> torch.Tensor:
        sequence_tensor = torch.tensor(sequence_steps, dtype=torch.float32, device=self.device)
        return (sequence_tensor - self.sequence_mean) / self.sequence_std

    def _normalize_static(self, static_vector: Sequence[float]) -> torch.Tensor:
        static_tensor = torch.tensor(static_vector, dtype=torch.float32, device=self.device)
        return (static_tensor - self.static_mean) / self.static_std

    def _compute_class_weights(self, y_train: Sequence[int]) -> torch.Tensor:
        counts = torch.bincount(torch.tensor(y_train, dtype=torch.long), minlength=self.num_classes).float()
        total = counts.sum().clamp_min(1.0)
        weights = total / (self.num_classes * counts.clamp_min(1.0))
        return weights.to(self.device)

    def _iter_batches(self, total_size: int) -> List[List[int]]:
        order = list(range(total_size))
        rng = random.Random(self.trainer_config.seed + self.best_epoch + total_size)
        rng.shuffle(order)
        return [order[start : start + self.trainer_config.batch_size] for start in range(0, total_size, self.trainer_config.batch_size)]

    def _detach_encoding(self, encoding: TorchManifoldEncodingOutput) -> TorchManifoldEncodingOutput:
        return TorchManifoldEncodingOutput(
            query=encoding.query.detach(),
            key=encoding.key.detach(),
            value=encoding.value.detach(),
            input_embedding=encoding.input_embedding.detach(),
            metadata=dict(encoding.metadata),
        )

    def _activity_for_label(self, label: int) -> float:
        label_name = self.label_names[label] if 0 <= label < len(self.label_names) else str(label)
        if label_name in {"worsen", "mixed"}:
            return 1.2
        return 1.0

    def _manifold_regularization(
        self,
        encodings: Sequence[TorchManifoldEncodingOutput],
        labels: Sequence[int],
        metadata: Sequence[Dict[str, float]],
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        if not encodings:
            zero = torch.zeros((), dtype=torch.float32, device=self.device)
            return zero, {
                "align_loss": 0.0,
                "compact_loss": 0.0,
                "separation_loss": 0.0,
                "temporal_smoothness_loss": 0.0,
                "manifold_regularization_loss": 0.0,
            }

        queries = torch.stack([encoding.query for encoding in encodings], dim=0)
        keys = torch.stack([encoding.key for encoding in encodings], dim=0)
        embeddings = torch.stack([encoding.input_embedding for encoding in encodings], dim=0)

        align_loss = (1.0 - F.cosine_similarity(queries, keys, dim=-1)).mean()

        pair_count = 0
        compact_sum = torch.zeros((), dtype=torch.float32, device=self.device)
        separation_sum = torch.zeros((), dtype=torch.float32, device=self.device)
        temporal_sum = torch.zeros((), dtype=torch.float32, device=self.device)
        temporal_pairs = 0

        for left_idx in range(len(encodings)):
            for right_idx in range(left_idx + 1, len(encodings)):
                distance = torch.norm(embeddings[left_idx] - embeddings[right_idx], p=2)
                if labels[left_idx] == labels[right_idx]:
                    compact_sum = compact_sum + distance.pow(2)
                    pair_count += 1
                else:
                    separation_sum = separation_sum + F.relu(self.trainer_config.separation_margin - distance).pow(2)

                same_patient = metadata[left_idx].get("stay_id") == metadata[right_idx].get("stay_id")
                if same_patient and "window_end_index" in metadata[left_idx] and "window_end_index" in metadata[right_idx]:
                    gap = abs(float(metadata[left_idx]["window_end_index"]) - float(metadata[right_idx]["window_end_index"]))
                    if gap <= 1.0:
                        temporal_sum = temporal_sum + distance.pow(2)
                        temporal_pairs += 1

        compact_loss = compact_sum / max(1, pair_count)
        separation_loss = separation_sum / max(1, pair_count)
        temporal_loss = temporal_sum / max(1, temporal_pairs)

        regularization = (
            self.trainer_config.align_loss_weight * align_loss
            + self.trainer_config.compact_loss_weight * compact_loss
            + self.trainer_config.separation_loss_weight * separation_loss
            + self.trainer_config.temporal_smoothness_weight * temporal_loss
        )

        return regularization, {
            "align_loss": float(align_loss.item()),
            "compact_loss": float(compact_loss.item()),
            "separation_loss": float(separation_loss.item()),
            "temporal_smoothness_loss": float(temporal_loss.item()),
            "manifold_regularization_loss": float(regularization.item()),
        }

    def _build_memory_bank(
        self,
        sequences: Sequence[Sequence[Sequence[float]]],
        static_data: Sequence[Sequence[float]],
        labels: Sequence[int],
        metadata: Sequence[Dict[str, float]],
    ):
        self.manifold.memory_bank = []
        self.eval()
        with torch.no_grad():
            for sequence_steps, static_vector, label, meta in zip(sequences, static_data, labels, metadata):
                encoding = self.manifold.encode_input_torch(
                    self._normalize_sequence(sequence_steps),
                    self._normalize_static(static_vector),
                    metadata=meta,
                )
                self.manifold.write_memory(encoding, label=label, activity=self._activity_for_label(label))

    def _forward_sample(
        self,
        sequence_steps: Sequence[Sequence[float]],
        static_vector: Sequence[float],
        metadata: Dict[str, float],
    ):
        normalized_sequence = self._normalize_sequence(sequence_steps)
        normalized_static = self._normalize_static(static_vector)
        encoding = self.manifold.encode_input_torch(
            normalized_sequence,
            normalized_static,
            metadata=metadata,
        )
        base_logits = self.base_classifier(encoding.input_embedding)
        base_prior = torch.softmax(base_logits.detach(), dim=-1)
        readout = self.manifold.read_memory_torch(encoding, class_prior=base_prior)
        fused_representation = self.manifold.fusion_head.fuse_torch(encoding.input_embedding, readout.readout)
        fusion_logits = self.fusion_classifier(fused_representation)
        return encoding, readout, base_logits, fusion_logits

    def _summarize_read_policy(
        self,
        sequences: Sequence[Sequence[Sequence[float]]],
        static_data: Sequence[Sequence[float]],
        metadata: Sequence[Dict[str, float]],
    ) -> Dict[str, float]:
        if not sequences:
            return {
                "read_confidence_mean": 0.0,
                "read_label_confidence_mean": 0.0,
                "read_attention_entropy_mean": 0.0,
                "read_top_label_matches_base_mean": 0.0,
            }

        confidence_sum = 0.0
        label_conf_sum = 0.0
        entropy_sum = 0.0
        margin_sum = 0.0
        label_match_sum = 0.0

        self.eval()
        with torch.no_grad():
            for sequence_steps, static_vector, meta in zip(sequences, static_data, metadata):
                normalized_sequence = self._normalize_sequence(sequence_steps)
                normalized_static = self._normalize_static(static_vector)
                encoding = self.manifold.encode_input_torch(
                    normalized_sequence,
                    normalized_static,
                    metadata=meta,
                )
                base_logits = self.base_classifier(encoding.input_embedding)
                base_prior = torch.softmax(base_logits, dim=-1)
                readout = self.manifold.read_memory_torch(encoding, class_prior=base_prior)
                base_label = int(torch.argmax(base_prior).item())

                confidence_sum += float(readout.memory_confidence)
                label_conf_sum += float(readout.label_confidence)
                entropy_sum += float(readout.attention_entropy)
                margin_sum += float(readout.score_margin)
                label_match_sum += 1.0 if readout.top_label == base_label else 0.0

        total = float(len(sequences))
        return {
            "read_confidence_mean": confidence_sum / total,
            "read_label_confidence_mean": label_conf_sum / total,
            "read_attention_entropy_mean": entropy_sum / total,
            "read_score_margin_mean": margin_sum / total,
            "read_top_label_matches_base_mean": label_match_sum / total,
        }

    def fit(
        self,
        train_sequences: Sequence[Sequence[Sequence[float]]],
        train_static: Sequence[Sequence[float]],
        y_train: Sequence[int],
        train_metadata: Sequence[Dict[str, float]],
        val_sequences: Sequence[Sequence[Sequence[float]]],
        val_static: Sequence[Sequence[float]],
        y_val: Sequence[int],
        val_metadata: Sequence[Dict[str, float]],
    ):
        self._fit_normalizers(train_sequences, train_static)
        self.class_weights = self._compute_class_weights(y_train)

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.trainer_config.learning_rate,
            weight_decay=self.trainer_config.weight_decay,
        )

        best_score = None
        best_state = None
        best_memory = None
        best_summary = {}

        for epoch in range(1, self.trainer_config.epochs + 1):
            self.train()
            self.manifold.memory_bank = []
            epoch_loss = 0.0
            batch_count = 0
            reg_running = {
                "align_loss": 0.0,
                "compact_loss": 0.0,
                "separation_loss": 0.0,
                "temporal_smoothness_loss": 0.0,
                "manifold_regularization_loss": 0.0,
            }

            for batch_indices in self._iter_batches(len(train_sequences)):
                optimizer.zero_grad()
                batch_losses = []
                delayed_writes: List[tuple[TorchManifoldEncodingOutput, int]] = []
                batch_encodings: List[TorchManifoldEncodingOutput] = []
                batch_labels: List[int] = []
                batch_metadata: List[Dict[str, float]] = []

                for sample_index in batch_indices:
                    encoding, _, base_logits, fusion_logits = self._forward_sample(
                        train_sequences[sample_index],
                        train_static[sample_index],
                        train_metadata[sample_index],
                    )
                    label_tensor = torch.tensor([y_train[sample_index]], dtype=torch.long, device=self.device)
                    fusion_loss = F.cross_entropy(fusion_logits.unsqueeze(0), label_tensor, weight=self.class_weights)
                    base_loss = F.cross_entropy(base_logits.unsqueeze(0), label_tensor, weight=self.class_weights)
                    batch_losses.append(fusion_loss + self.trainer_config.aux_base_loss_weight * base_loss)
                    delayed_writes.append((self._detach_encoding(encoding), y_train[sample_index]))
                    batch_encodings.append(encoding)
                    batch_labels.append(y_train[sample_index])
                    batch_metadata.append(train_metadata[sample_index])

                regularization, reg_breakdown = self._manifold_regularization(batch_encodings, batch_labels, batch_metadata)
                loss = torch.stack(batch_losses).mean() + regularization
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), self.trainer_config.grad_clip)
                optimizer.step()

                with torch.no_grad():
                    for detached_encoding, label in delayed_writes:
                        self.manifold.write_memory(detached_encoding, label=label, activity=self._activity_for_label(label))
                    self.manifold.decay_memory()

                epoch_loss += float(loss.item())
                batch_count += 1
                for key, value in reg_breakdown.items():
                    reg_running[key] += value

            self._build_memory_bank(train_sequences, train_static, y_train, train_metadata)
            val_probs = self.predict_proba(val_sequences, val_static, val_metadata)
            val_pred = [max(range(len(prob)), key=lambda idx: prob[idx]) for prob in val_probs]
            val_accuracy = sum(1 for yt, yp in zip(y_val, val_pred) if yt == yp) / max(1, len(y_val))
            val_macro_f1 = self._macro_f1(y_val, val_pred)
            val_priority_f1 = self._priority_f1(y_val, val_pred)
            score = (val_macro_f1, val_priority_f1, val_accuracy)

            if best_score is None or score > best_score:
                best_score = score
                self.best_epoch = epoch
                best_state = copy.deepcopy(self.state_dict())
                best_memory = copy.deepcopy(self.manifold.memory_bank)
                best_summary = {
                    "epoch": float(epoch),
                    "train_loss": epoch_loss / max(1, batch_count),
                    "val_accuracy": val_accuracy,
                    "val_f1_macro": val_macro_f1,
                    "val_priority_f1_macro": val_priority_f1,
                    "align_loss": reg_running["align_loss"] / max(1, batch_count),
                    "compact_loss": reg_running["compact_loss"] / max(1, batch_count),
                    "separation_loss": reg_running["separation_loss"] / max(1, batch_count),
                    "temporal_smoothness_loss": reg_running["temporal_smoothness_loss"] / max(1, batch_count),
                    "manifold_regularization_loss": reg_running["manifold_regularization_loss"] / max(1, batch_count),
                }

        if best_state is not None:
            self.load_state_dict(best_state)
        if best_memory is not None:
            self.manifold.memory_bank = best_memory

        memory_summary = self.manifold.summarize_memory()
        read_policy_summary = self._summarize_read_policy(val_sequences, val_static, val_metadata)
        self.latest_loss_breakdown = {
            "align_loss_weight": self.trainer_config.align_loss_weight,
            "compact_loss_weight": self.trainer_config.compact_loss_weight,
            "separation_loss_weight": self.trainer_config.separation_loss_weight,
            "temporal_smoothness_weight": self.trainer_config.temporal_smoothness_weight,
            "separation_margin": self.trainer_config.separation_margin,
        }
        self.training_summary = {
            "model_family": "manifold",
            "best_epoch": float(self.best_epoch),
            "trainable_parameter_count": float(self.parameter_count()),
            **best_summary,
        }
        self.memory_diagnostics = {
            "model_family": 1.0,
            "parameter_count": float(self.parameter_count()),
            "device_is_cuda": 1.0 if self.device.type == "cuda" else 0.0,
            "align_loss_weight": float(self.trainer_config.align_loss_weight),
            "compact_loss_weight": float(self.trainer_config.compact_loss_weight),
            "separation_loss_weight": float(self.trainer_config.separation_loss_weight),
            "temporal_smoothness_weight": float(self.trainer_config.temporal_smoothness_weight),
            **memory_summary,
            **read_policy_summary,
        }

    def predict_proba(
        self,
        sequences: Sequence[Sequence[Sequence[float]]],
        static_data: Sequence[Sequence[float]],
        metadata: Sequence[Dict[str, float]],
    ) -> List[List[float]]:
        self.eval()
        probs: List[List[float]] = []
        with torch.no_grad():
            for sequence_steps, static_vector, meta in zip(sequences, static_data, metadata):
                _, _, _, fusion_logits = self._forward_sample(sequence_steps, static_vector, meta)
                prob = torch.softmax(fusion_logits, dim=-1)
                probs.append(prob.detach().cpu().tolist())
        return probs

    def predict_base_proba(
        self,
        sequences: Sequence[Sequence[Sequence[float]]],
        static_data: Sequence[Sequence[float]],
        metadata: Sequence[Dict[str, float]],
    ) -> List[List[float]]:
        self.eval()
        probs: List[List[float]] = []
        with torch.no_grad():
            for sequence_steps, static_vector, meta in zip(sequences, static_data, metadata):
                encoding = self.manifold.encode_input_torch(
                    self._normalize_sequence(sequence_steps),
                    self._normalize_static(static_vector),
                    metadata=meta,
                )
                base_logits = self.base_classifier(encoding.input_embedding)
                prob = torch.softmax(base_logits, dim=-1)
                probs.append(prob.detach().cpu().tolist())
        return probs

    def predict(
        self,
        sequences: Sequence[Sequence[Sequence[float]]],
        static_data: Sequence[Sequence[float]],
        metadata: Sequence[Dict[str, float]],
    ) -> List[int]:
        probs = self.predict_proba(sequences, static_data, metadata)
        return [max(range(len(prob)), key=lambda idx: prob[idx]) for prob in probs]

    def _macro_f1(self, y_true: Sequence[int], y_pred: Sequence[int]) -> float:
        classes = sorted(set(y_true) | set(y_pred))
        scores = []
        for class_id in classes:
            tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == class_id and yp == class_id)
            fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != class_id and yp == class_id)
            fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == class_id and yp != class_id)
            precision = tp / (tp + fp) if tp + fp > 0 else 0.0
            recall = tp / (tp + fn) if tp + fn > 0 else 0.0
            scores.append(0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall))
        return sum(scores) / max(1, len(scores))

    def _priority_f1(self, y_true: Sequence[int], y_pred: Sequence[int]) -> float:
        priority_ids = [idx for idx, name in enumerate(self.label_names) if name in {"worsen", "mixed"}]
        if not priority_ids:
            return 0.0
        scores = []
        for class_id in priority_ids:
            tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == class_id and yp == class_id)
            fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != class_id and yp == class_id)
            fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == class_id and yp != class_id)
            precision = tp / (tp + fp) if tp + fp > 0 else 0.0
            recall = tp / (tp + fn) if tp + fn > 0 else 0.0
            scores.append(0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall))
        return sum(scores) / max(1, len(scores))
