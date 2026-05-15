import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn


@dataclass
class ManifoldMemoryConfig:
    sequence_feature_dim: int
    static_feature_dim: int
    manifold_dim: int = 32
    value_dim: int = 48
    fusion_hidden_dim: int = 64
    top_k: int = 6
    temperature: float = 0.12
    similarity_threshold: float = 0.9
    merge_alpha: float = 0.2
    decay: float = 0.995
    forget_threshold: float = 0.05
    max_memory: int = 256
    same_label_merge_only: bool = True
    min_label_memory: int = 8
    max_label_memory: int = 96
    max_patient_label_memory: int = 3
    support_penalty: float = 0.04
    collapse_penalty: float = 0.06
    rerank_strength: float = 0.18
    rerank_top_classes: int = 2
    rerank_candidates_per_class: int = 4
    confidence_floor: float = 0.15
    confidence_sharpness: float = 12.0
    confidence_margin_sharpness: float = 20.0
    uncertainty_floor: float = 0.35
    include_static_context: bool = True
    seed: int = 42
    encoder_type: str = "gru"
    gru_hidden_dim: int = 64
    gru_layers: int = 1
    gru_bidirectional: bool = True
    gru_dropout: float = 0.1
    transformer_d_model: int = 96
    transformer_layers: int = 2
    transformer_heads: int = 4
    transformer_ff_dim: int = 192
    transformer_dropout: float = 0.1
    transformer_max_length: int = 256
    static_hidden_dim: int = 16
    use_layer_norm: bool = True
    device: str = "cpu"


@dataclass
class ManifoldEncodingOutput:
    query: List[float]
    key: List[float]
    value: List[float]
    input_embedding: List[float]
    metadata: Dict[str, float] = field(default_factory=dict)


@dataclass
class TorchManifoldEncodingOutput:
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    input_embedding: torch.Tensor
    metadata: Dict[str, float] = field(default_factory=dict)


@dataclass
class ManifoldMemoryItem:
    key: List[float]
    value: List[float]
    label: int
    activity: float
    support: int = 1
    metadata: Dict[str, float] = field(default_factory=dict)


@dataclass
class AttentionReadout:
    readout: List[float]
    attention_weights: List[float]
    matched_indices: List[int]
    matched_labels: List[int]
    max_similarity: float
    memory_confidence: float
    label_confidence: float
    attention_entropy: float
    score_margin: float
    top_label: int


@dataclass
class TorchAttentionReadout:
    readout: torch.Tensor
    attention_weights: torch.Tensor
    matched_indices: List[int]
    matched_labels: List[int]
    max_similarity: float
    memory_confidence: float
    memory_confidence_tensor: torch.Tensor
    label_confidence: float
    attention_entropy: float
    score_margin: float
    top_label: int


def _norm(vector: Sequence[float]) -> float:
    return math.sqrt(sum(value * value for value in vector)) + 1e-12


def _normalize(vector: Sequence[float]) -> List[float]:
    inv = 1.0 / _norm(vector)
    return [value * inv for value in vector]


def _normalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor / tensor.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def _tensor_to_list(tensor: torch.Tensor) -> List[float]:
    return tensor.detach().cpu().view(-1).tolist()


class GRUManifoldEncoder(nn.Module):
    """
    Trainable manifold encoder.

    Input:
    - sequence_steps: [T, F] or [B, T, F]
    - static_vector: [S] or [B, S]

    Output:
    - query / key / value projections
    - a context embedding that downstream fusion heads can consume
    """

    def __init__(self, config: ManifoldMemoryConfig):
        super().__init__()
        self.config = config
        self.num_directions = 2 if config.gru_bidirectional else 1
        self.sequence_context_dim = config.gru_hidden_dim * self.num_directions
        self.static_context_dim = config.static_hidden_dim if config.include_static_context and config.static_feature_dim > 0 else 0
        self.embedding_dim = self.sequence_context_dim * 3 + self.static_context_dim

        self.input_norm = nn.LayerNorm(config.sequence_feature_dim) if config.use_layer_norm else nn.Identity()
        self.gru = nn.GRU(
            input_size=config.sequence_feature_dim,
            hidden_size=config.gru_hidden_dim,
            num_layers=config.gru_layers,
            batch_first=True,
            dropout=config.gru_dropout if config.gru_layers > 1 else 0.0,
            bidirectional=config.gru_bidirectional,
        )

        if self.static_context_dim > 0:
            self.static_projector = nn.Sequential(
                nn.Linear(config.static_feature_dim, config.static_hidden_dim),
                nn.GELU(),
                nn.LayerNorm(config.static_hidden_dim) if config.use_layer_norm else nn.Identity(),
            )
        else:
            self.static_projector = None

        self.context_norm = nn.LayerNorm(self.embedding_dim) if config.use_layer_norm else nn.Identity()
        self.query_head = nn.Linear(self.embedding_dim, config.manifold_dim)
        self.key_head = nn.Linear(self.embedding_dim, config.manifold_dim)
        self.value_head = nn.Linear(self.embedding_dim, config.value_dim)

    def _prepare_sequence(self, sequence_steps: Sequence[Sequence[float]] | torch.Tensor) -> torch.Tensor:
        if isinstance(sequence_steps, torch.Tensor):
            seq = sequence_steps.float()
        else:
            seq = torch.tensor(sequence_steps, dtype=torch.float32)
        if seq.dim() == 2:
            seq = seq.unsqueeze(0)
        return seq

    def _prepare_static(self, static_vector: Optional[Sequence[float] | torch.Tensor], batch_size: int, device: torch.device) -> Optional[torch.Tensor]:
        if self.static_projector is None:
            return None
        if static_vector is None:
            return torch.zeros(batch_size, self.config.static_feature_dim, dtype=torch.float32, device=device)
        if isinstance(static_vector, torch.Tensor):
            static_tensor = static_vector.float().to(device)
        else:
            static_tensor = torch.tensor(static_vector, dtype=torch.float32, device=device)
        if static_tensor.dim() == 1:
            static_tensor = static_tensor.unsqueeze(0)
        return static_tensor

    def _last_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = hidden.view(self.config.gru_layers, self.num_directions, hidden.size(1), self.config.gru_hidden_dim)
        last_layer = hidden[-1]
        if self.num_directions == 1:
            return last_layer[0]
        return torch.cat([last_layer[0], last_layer[1]], dim=-1)

    def forward(
        self,
        sequence_steps: Sequence[Sequence[float]] | torch.Tensor,
        static_vector: Optional[Sequence[float] | torch.Tensor] = None,
        metadata: Optional[Dict[str, float]] = None,
    ) -> TorchManifoldEncodingOutput:
        seq = self._prepare_sequence(sequence_steps).to(torch.device(self.config.device))
        seq = self.input_norm(seq)
        outputs, hidden = self.gru(seq)

        final_hidden = self._last_hidden(hidden)
        pooled_hidden = outputs.mean(dim=1)
        delta_hidden = outputs[:, -1, :] - outputs[:, 0, :]
        parts = [final_hidden, pooled_hidden, delta_hidden]

        static_tensor = self._prepare_static(static_vector, seq.size(0), seq.device)
        if static_tensor is not None:
            parts.append(self.static_projector(static_tensor))

        context = torch.cat(parts, dim=-1)
        context = self.context_norm(context)
        query = _normalize_tensor(self.query_head(context))
        key = _normalize_tensor(self.key_head(context))
        value = self.value_head(context)

        return TorchManifoldEncodingOutput(
            query=query.squeeze(0),
            key=key.squeeze(0),
            value=value.squeeze(0),
            input_embedding=context.squeeze(0),
            metadata=metadata or {},
        )


class TransformerManifoldEncoder(nn.Module):
    """
    Transformer-based encoder with the same downstream contract as the GRU
    encoder so memory read/write and fusion logic can remain unchanged.
    """

    def __init__(self, config: ManifoldMemoryConfig):
        super().__init__()
        self.config = config
        self.sequence_context_dim = config.transformer_d_model
        self.static_context_dim = config.static_hidden_dim if config.include_static_context and config.static_feature_dim > 0 else 0
        self.embedding_dim = self.sequence_context_dim * 3 + self.static_context_dim

        self.input_norm = nn.LayerNorm(config.sequence_feature_dim) if config.use_layer_norm else nn.Identity()
        self.input_projector = nn.Linear(config.sequence_feature_dim, config.transformer_d_model)
        self.position_embedding = nn.Embedding(config.transformer_max_length, config.transformer_d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.transformer_d_model,
            nhead=config.transformer_heads,
            dim_feedforward=config.transformer_ff_dim,
            dropout=config.transformer_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=config.use_layer_norm,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.transformer_layers)

        if self.static_context_dim > 0:
            self.static_projector = nn.Sequential(
                nn.Linear(config.static_feature_dim, config.static_hidden_dim),
                nn.GELU(),
                nn.LayerNorm(config.static_hidden_dim) if config.use_layer_norm else nn.Identity(),
            )
        else:
            self.static_projector = None

        self.context_norm = nn.LayerNorm(self.embedding_dim) if config.use_layer_norm else nn.Identity()
        self.query_head = nn.Linear(self.embedding_dim, config.manifold_dim)
        self.key_head = nn.Linear(self.embedding_dim, config.manifold_dim)
        self.value_head = nn.Linear(self.embedding_dim, config.value_dim)

    def _prepare_sequence(self, sequence_steps: Sequence[Sequence[float]] | torch.Tensor) -> torch.Tensor:
        if isinstance(sequence_steps, torch.Tensor):
            seq = sequence_steps.float()
        else:
            seq = torch.tensor(sequence_steps, dtype=torch.float32)
        if seq.dim() == 2:
            seq = seq.unsqueeze(0)
        return seq

    def _prepare_static(self, static_vector: Optional[Sequence[float] | torch.Tensor], batch_size: int, device: torch.device) -> Optional[torch.Tensor]:
        if self.static_projector is None:
            return None
        if static_vector is None:
            return torch.zeros(batch_size, self.config.static_feature_dim, dtype=torch.float32, device=device)
        if isinstance(static_vector, torch.Tensor):
            static_tensor = static_vector.float().to(device)
        else:
            static_tensor = torch.tensor(static_vector, dtype=torch.float32, device=device)
        if static_tensor.dim() == 1:
            static_tensor = static_tensor.unsqueeze(0)
        return static_tensor

    def forward(
        self,
        sequence_steps: Sequence[Sequence[float]] | torch.Tensor,
        static_vector: Optional[Sequence[float] | torch.Tensor] = None,
        metadata: Optional[Dict[str, float]] = None,
    ) -> TorchManifoldEncodingOutput:
        seq = self._prepare_sequence(sequence_steps).to(torch.device(self.config.device))
        seq = self.input_norm(seq)
        seq = self.input_projector(seq)

        if seq.size(1) > self.config.transformer_max_length:
            raise ValueError(
                f"sequence length {seq.size(1)} exceeds transformer_max_length={self.config.transformer_max_length}"
            )

        position_ids = torch.arange(seq.size(1), device=seq.device).unsqueeze(0).expand(seq.size(0), -1)
        seq = seq + self.position_embedding(position_ids)
        outputs = self.transformer(seq)

        final_hidden = outputs[:, -1, :]
        pooled_hidden = outputs.mean(dim=1)
        delta_hidden = outputs[:, -1, :] - outputs[:, 0, :]
        parts = [final_hidden, pooled_hidden, delta_hidden]

        static_tensor = self._prepare_static(static_vector, seq.size(0), seq.device)
        if static_tensor is not None:
            parts.append(self.static_projector(static_tensor))

        context = torch.cat(parts, dim=-1)
        context = self.context_norm(context)
        query = _normalize_tensor(self.query_head(context))
        key = _normalize_tensor(self.key_head(context))
        value = self.value_head(context)

        return TorchManifoldEncodingOutput(
            query=query.squeeze(0),
            key=key.squeeze(0),
            value=value.squeeze(0),
            input_embedding=context.squeeze(0),
            metadata=metadata or {},
        )


class AttentionMemoryReader(nn.Module):
    def __init__(self, config: ManifoldMemoryConfig):
        super().__init__()
        self.config = config

    def _compute_rerank_bias(
        self,
        memory_bank: Sequence[ManifoldMemoryItem],
        class_prior: Optional[torch.Tensor],
        device: torch.device,
        uncertainty_scale: float = 1.0,
    ) -> torch.Tensor:
        if class_prior is None or not memory_bank:
            return torch.zeros(len(memory_bank), dtype=torch.float32, device=device)

        prior = class_prior.float().to(device)
        if prior.dim() != 1:
            prior = prior.view(-1)
        prior = prior / prior.sum().clamp_min(1e-12)

        labels = torch.tensor([item.label for item in memory_bank], dtype=torch.long, device=device)
        label_bias = prior[labels]
        label_bias = label_bias - label_bias.mean()

        if self.config.rerank_top_classes > 0:
            top_classes = torch.topk(prior, k=min(self.config.rerank_top_classes, prior.numel())).indices
            top_mask = torch.zeros_like(prior)
            top_mask[top_classes] = 1.0
            label_bias = label_bias + 0.5 * top_mask[labels]

        return self.config.rerank_strength * uncertainty_scale * label_bias

    def _normalized_uncertainty(self, class_prior: Optional[torch.Tensor], device: torch.device) -> float:
        if class_prior is None:
            return 1.0
        prior = class_prior.float().to(device)
        if prior.dim() != 1:
            prior = prior.view(-1)
        prior = prior / prior.sum().clamp_min(1e-12)
        max_prior = float(torch.max(prior).item())
        return (1.0 - max_prior) * prior.numel() / max(1.0, float(prior.numel() - 1))

    def _label_distribution(
        self,
        matched_labels: Sequence[int],
        weights: torch.Tensor,
        class_prior: Optional[torch.Tensor],
        device: torch.device,
    ) -> tuple[torch.Tensor, float, int]:
        if class_prior is not None:
            num_classes = int(class_prior.numel())
        elif matched_labels:
            num_classes = max(matched_labels) + 1
        else:
            num_classes = 0

        if num_classes <= 0:
            return torch.zeros(0, dtype=torch.float32, device=device), 0.0, -1

        label_mass = torch.zeros(num_classes, dtype=torch.float32, device=device)
        for weight, label in zip(weights, matched_labels):
            label_mass[int(label)] += weight

        top_label = int(torch.argmax(label_mass).item()) if label_mass.numel() > 0 else -1
        top_mass = float(label_mass[top_label].item()) if top_label >= 0 else 0.0
        return label_mass, top_mass, top_label

    def _compute_memory_confidence(
        self,
        max_similarity: torch.Tensor,
        top_scores: torch.Tensor,
        weights: torch.Tensor,
        matched_labels: Sequence[int],
        class_prior: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, float, float, float, int]:
        device = max_similarity.device
        label_mass, label_confidence, top_label = self._label_distribution(matched_labels, weights, class_prior, device)

        if weights.numel() <= 1:
            attention_entropy = 0.0
        else:
            entropy = -(weights * torch.log(weights.clamp_min(1e-12))).sum()
            entropy = entropy / math.log(weights.numel())
            attention_entropy = float(entropy.item())

        similarity_conf = torch.sigmoid(
            (max_similarity - self.config.similarity_threshold) * self.config.confidence_sharpness
        )
        if top_scores.numel() <= 1:
            score_margin = 1.0
        else:
            score_margin = float((top_scores[0] - top_scores[1]).item())
        margin_conf = torch.sigmoid(torch.tensor(score_margin * self.config.confidence_margin_sharpness, device=device))

        if class_prior is not None and label_mass.numel() > 0:
            prior = class_prior.float().to(device)
            prior = prior / prior.sum().clamp_min(1e-12)
            max_prior = float(torch.max(prior).item())
            normalized_uncertainty = (1.0 - max_prior) * prior.numel() / max(1.0, float(prior.numel() - 1))
            uncertainty_scale = self.config.uncertainty_floor + (1.0 - self.config.uncertainty_floor) * normalized_uncertainty
        else:
            uncertainty_scale = 1.0

        combined = 0.45 * similarity_conf + 0.35 * float(label_confidence) + 0.20 * margin_conf
        confidence = self.config.confidence_floor + (1.0 - self.config.confidence_floor) * combined * uncertainty_scale
        confidence = torch.clamp(confidence, min=self.config.confidence_floor, max=1.0)
        return confidence, float(label_confidence), float(attention_entropy), score_margin, top_label

    def read_torch(
        self,
        query: torch.Tensor,
        memory_bank: Sequence[ManifoldMemoryItem],
        class_prior: Optional[torch.Tensor] = None,
    ) -> TorchAttentionReadout:
        if not memory_bank:
            return TorchAttentionReadout(
                readout=torch.zeros(self.config.value_dim, dtype=torch.float32, device=query.device),
                attention_weights=torch.zeros(0, dtype=torch.float32, device=query.device),
                matched_indices=[],
                matched_labels=[],
                max_similarity=0.0,
                memory_confidence=0.0,
                memory_confidence_tensor=torch.zeros((), dtype=torch.float32, device=query.device),
                label_confidence=0.0,
                attention_entropy=1.0,
                score_margin=0.0,
                top_label=-1,
            )

        keys = torch.tensor([item.key for item in memory_bank], dtype=torch.float32, device=query.device)
        values = torch.tensor([item.value for item in memory_bank], dtype=torch.float32, device=query.device)
        raw_similarities = torch.matmul(keys, query)
        uncertainty_scale = self._normalized_uncertainty(class_prior, query.device)
        rerank_bias = self._compute_rerank_bias(memory_bank, class_prior, query.device, uncertainty_scale=uncertainty_scale)

        candidate_indices = list(range(len(memory_bank)))
        if class_prior is not None and self.config.rerank_top_classes > 0:
            prior = class_prior.float().to(query.device)
            if prior.dim() != 1:
                prior = prior.view(-1)
            prior = prior / prior.sum().clamp_min(1e-12)
            top_classes = torch.topk(prior, k=min(self.config.rerank_top_classes, prior.numel())).indices.detach().cpu().tolist()
            candidate_set = set(torch.topk(raw_similarities, k=min(self.config.top_k, raw_similarities.numel()), dim=0).indices.detach().cpu().tolist())
            for class_id in top_classes:
                class_indices = [index for index, item in enumerate(memory_bank) if item.label == int(class_id)]
                if not class_indices:
                    continue
                class_scores = raw_similarities[class_indices]
                class_top = min(self.config.rerank_candidates_per_class, len(class_indices))
                selected_offsets = torch.topk(class_scores, k=class_top, dim=0).indices.detach().cpu().tolist()
                for offset in selected_offsets:
                    candidate_set.add(class_indices[offset])
            candidate_indices = sorted(candidate_set)

        candidate_tensor = torch.tensor(candidate_indices, dtype=torch.long, device=query.device)
        adjusted_scores = raw_similarities[candidate_tensor] + rerank_bias[candidate_tensor]
        top_k = min(self.config.top_k, adjusted_scores.numel())
        top_scores, local_top_indices = torch.topk(adjusted_scores, k=top_k, dim=0)
        top_indices = candidate_tensor[local_top_indices]
        weights = torch.softmax(top_scores / max(self.config.temperature, 1e-6), dim=0)
        top_indices_list = top_indices.detach().cpu().tolist()
        matched_labels = [memory_bank[index].label for index in top_indices_list]
        base_readout = (weights.unsqueeze(-1) * values[top_indices]).sum(dim=0)
        top_raw_similarity = raw_similarities[top_indices[0]] if top_k > 0 else torch.zeros((), dtype=torch.float32, device=query.device)
        memory_confidence_tensor, label_confidence, attention_entropy, score_margin, top_label = self._compute_memory_confidence(
            top_raw_similarity,
            top_scores,
            weights,
            matched_labels,
            class_prior,
        )
        readout = base_readout * memory_confidence_tensor

        return TorchAttentionReadout(
            readout=readout,
            attention_weights=weights,
            matched_indices=top_indices_list,
            matched_labels=matched_labels,
            max_similarity=float(top_raw_similarity.item()) if top_k > 0 else 0.0,
            memory_confidence=float(memory_confidence_tensor.item()),
            memory_confidence_tensor=memory_confidence_tensor,
            label_confidence=label_confidence,
            attention_entropy=attention_entropy,
            score_margin=score_margin,
            top_label=top_label,
        )

    def read(
        self,
        query: Sequence[float] | torch.Tensor,
        memory_bank: Sequence[ManifoldMemoryItem],
        class_prior: Optional[torch.Tensor] = None,
    ) -> AttentionReadout:
        if isinstance(query, torch.Tensor):
            query_tensor = query.float()
        else:
            query_tensor = torch.tensor(query, dtype=torch.float32)
        torch_readout = self.read_torch(query_tensor, memory_bank, class_prior=class_prior)
        return AttentionReadout(
            readout=_tensor_to_list(torch_readout.readout),
            attention_weights=_tensor_to_list(torch_readout.attention_weights),
            matched_indices=torch_readout.matched_indices,
            matched_labels=torch_readout.matched_labels,
            max_similarity=torch_readout.max_similarity,
            memory_confidence=torch_readout.memory_confidence,
            label_confidence=torch_readout.label_confidence,
            attention_entropy=torch_readout.attention_entropy,
            score_margin=torch_readout.score_margin,
            top_label=torch_readout.top_label,
        )


class FusionHeadBlueprint(nn.Module):
    """
    Trainable fusion head.

    It projects the memory readout into the same latent space as the current
    input embedding, then fuses input, memory, difference and interaction
    features through a small MLP.
    """

    def __init__(self, config: ManifoldMemoryConfig, input_embedding_dim: int):
        super().__init__()
        self.config = config
        self.input_embedding_dim = input_embedding_dim
        self.memory_projector = nn.Linear(config.value_dim, input_embedding_dim)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(input_embedding_dim * 4, config.fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.gru_dropout),
            nn.Linear(config.fusion_hidden_dim, config.fusion_hidden_dim),
            nn.GELU(),
        )

    def fuse_torch(self, input_embedding: torch.Tensor, memory_readout: torch.Tensor) -> torch.Tensor:
        projected_memory = self.memory_projector(memory_readout)
        diff = input_embedding - projected_memory
        prod = input_embedding * projected_memory
        fusion_input = torch.cat([input_embedding, projected_memory, diff, prod], dim=-1)
        return self.fusion_mlp(fusion_input)

    def fuse(self, encoding: ManifoldEncodingOutput, readout: AttentionReadout) -> List[float]:
        input_tensor = torch.tensor(encoding.input_embedding, dtype=torch.float32)
        readout_tensor = torch.tensor(readout.readout, dtype=torch.float32)
        return _tensor_to_list(self.fuse_torch(input_tensor, readout_tensor))


class ManifoldMemoryBlueprint(nn.Module):
    """
    Trainable GRU-based manifold memory skeleton.

    This is not the final end-to-end trainer yet, but it is no longer a
    placeholder projector. The encoder and fusion head are now genuine
    trainable PyTorch modules.
    """

    def __init__(self, config: ManifoldMemoryConfig):
        super().__init__()
        self.config = config
        if config.encoder_type == "gru":
            self.encoder = GRUManifoldEncoder(config)
        elif config.encoder_type == "transformer":
            self.encoder = TransformerManifoldEncoder(config)
        else:
            raise ValueError(f"Unsupported encoder_type: {config.encoder_type}")
        self.reader = AttentionMemoryReader(config)
        self.fusion_head = FusionHeadBlueprint(config, input_embedding_dim=self.encoder.embedding_dim)
        self.memory_bank: List[ManifoldMemoryItem] = []
        self.to(torch.device(config.device))

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def _similarity(self, left: Sequence[float], right: Sequence[float]) -> float:
        return sum(left_value * right_value for left_value, right_value in zip(left, right))

    def _patient_id(self, metadata: Dict[str, float]) -> Optional[float]:
        if "stay_id" not in metadata:
            return None
        return metadata["stay_id"]

    def _label_count(self, label: int) -> int:
        return sum(1 for item in self.memory_bank if item.label == label)

    def _patient_label_count(self, label: int, patient_id: Optional[float]) -> int:
        if patient_id is None:
            return 0
        return sum(
            1
            for item in self.memory_bank
            if item.label == label and self._patient_id(item.metadata) == patient_id
        )

    def _memory_priority(self, item: ManifoldMemoryItem) -> float:
        return item.activity + 0.08 * math.log1p(item.support)

    def summarize_memory(self) -> Dict[str, float]:
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

    def _find_best_merge_candidate(
        self,
        key: Sequence[float],
        label: int,
        metadata: Dict[str, float],
    ) -> Tuple[int, float]:
        patient_id = self._patient_id(metadata)
        best_index = -1
        best_similarity = -2.0
        best_adjusted_score = -2.0

        for index, item in enumerate(self.memory_bank):
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

        return best_index, best_similarity

    def encode_input_torch(
        self,
        sequence_steps: Sequence[Sequence[float]] | torch.Tensor,
        static_vector: Optional[Sequence[float] | torch.Tensor] = None,
        metadata: Optional[Dict[str, float]] = None,
    ) -> TorchManifoldEncodingOutput:
        return self.encoder(sequence_steps, static_vector=static_vector, metadata=metadata)

    def encode_input(
        self,
        sequence_steps: Sequence[Sequence[float]] | torch.Tensor,
        static_vector: Optional[Sequence[float] | torch.Tensor] = None,
        metadata: Optional[Dict[str, float]] = None,
    ) -> ManifoldEncodingOutput:
        torch_output = self.encode_input_torch(sequence_steps, static_vector=static_vector, metadata=metadata)
        return ManifoldEncodingOutput(
            query=_tensor_to_list(torch_output.query),
            key=_tensor_to_list(torch_output.key),
            value=_tensor_to_list(torch_output.value),
            input_embedding=_tensor_to_list(torch_output.input_embedding),
            metadata=torch_output.metadata,
        )

    def read_memory_torch(
        self,
        encoding: TorchManifoldEncodingOutput,
        class_prior: Optional[torch.Tensor] = None,
    ) -> TorchAttentionReadout:
        return self.reader.read_torch(encoding.query, self.memory_bank, class_prior=class_prior)

    def read_memory(
        self,
        encoding: ManifoldEncodingOutput,
        class_prior: Optional[torch.Tensor] = None,
    ) -> AttentionReadout:
        return self.reader.read(encoding.query, self.memory_bank, class_prior=class_prior)

    def forward_torch(
        self,
        sequence_steps: Sequence[Sequence[float]] | torch.Tensor,
        static_vector: Optional[Sequence[float] | torch.Tensor] = None,
        metadata: Optional[Dict[str, float]] = None,
        class_prior: Optional[torch.Tensor] = None,
    ) -> Tuple[TorchManifoldEncodingOutput, TorchAttentionReadout, torch.Tensor]:
        encoding = self.encode_input_torch(sequence_steps, static_vector=static_vector, metadata=metadata)
        readout = self.read_memory_torch(encoding, class_prior=class_prior)
        fused_representation = self.fusion_head.fuse_torch(encoding.input_embedding, readout.readout)
        return encoding, readout, fused_representation

    def forward(
        self,
        sequence_steps: Sequence[Sequence[float]] | torch.Tensor,
        static_vector: Optional[Sequence[float] | torch.Tensor] = None,
        metadata: Optional[Dict[str, float]] = None,
        class_prior: Optional[torch.Tensor] = None,
    ) -> Tuple[ManifoldEncodingOutput, AttentionReadout, List[float]]:
        encoding, readout, fused_representation = self.forward_torch(
            sequence_steps,
            static_vector=static_vector,
            metadata=metadata,
            class_prior=class_prior,
        )
        return (
            ManifoldEncodingOutput(
                query=_tensor_to_list(encoding.query),
                key=_tensor_to_list(encoding.key),
                value=_tensor_to_list(encoding.value),
                input_embedding=_tensor_to_list(encoding.input_embedding),
                metadata=encoding.metadata,
            ),
            AttentionReadout(
                readout=_tensor_to_list(readout.readout),
                attention_weights=_tensor_to_list(readout.attention_weights),
                matched_indices=readout.matched_indices,
                matched_labels=readout.matched_labels,
                max_similarity=readout.max_similarity,
                memory_confidence=readout.memory_confidence,
                label_confidence=readout.label_confidence,
                attention_entropy=readout.attention_entropy,
                score_margin=readout.score_margin,
                top_label=readout.top_label,
            ),
            _tensor_to_list(fused_representation),
        )

    def write_memory(
        self,
        encoding: ManifoldEncodingOutput | TorchManifoldEncodingOutput,
        label: int,
        activity: float = 1.0,
    ):
        key = _tensor_to_list(encoding.key) if isinstance(encoding, TorchManifoldEncodingOutput) else list(encoding.key)
        value = _tensor_to_list(encoding.value) if isinstance(encoding, TorchManifoldEncodingOutput) else list(encoding.value)
        metadata = dict(encoding.metadata)
        patient_id = self._patient_id(metadata)
        label_count = self._label_count(label)
        patient_label_count = self._patient_label_count(label, patient_id)

        force_new_memory = label_count < self.config.min_label_memory and patient_label_count < self.config.max_patient_label_memory
        best_index, best_similarity = self._find_best_merge_candidate(key, label, metadata)

        if not force_new_memory and best_index >= 0 and best_similarity >= self.config.similarity_threshold:
            target = self.memory_bank[best_index]
            alpha = min(self.config.merge_alpha, 1.0 / (target.support + 1.0))
            target.key = _normalize(
                [(1.0 - alpha) * left + alpha * right for left, right in zip(target.key, key)]
            )
            target.value = [(1.0 - alpha) * left + alpha * right for left, right in zip(target.value, value)]
            target.activity = max(target.activity, activity) + 0.02
            target.support += 1
            return

        self.memory_bank.append(
            ManifoldMemoryItem(
                key=_normalize(key),
                value=value[:],
                label=label,
                activity=activity,
                metadata=metadata,
            )
        )
        self._trim_memory()

    def decay_memory(self):
        for item in self.memory_bank:
            item.activity *= self.config.decay
        self.memory_bank = [item for item in self.memory_bank if item.activity >= self.config.forget_threshold]
        self._trim_memory()

    def _trim_memory(self):
        if not self.memory_bank:
            return

        by_label: Dict[int, List[ManifoldMemoryItem]] = {}
        for item in self.memory_bank:
            by_label.setdefault(item.label, []).append(item)

        trimmed: List[ManifoldMemoryItem] = []
        for label_items in by_label.values():
            sorted_items = sorted(label_items, key=self._memory_priority, reverse=True)
            unique_patient_items: List[ManifoldMemoryItem] = []
            overflow_items: List[ManifoldMemoryItem] = []
            seen_patients = set()

            for item in sorted_items:
                patient_id = self._patient_id(item.metadata)
                if patient_id is not None and patient_id not in seen_patients and len(unique_patient_items) < self.config.max_label_memory:
                    unique_patient_items.append(item)
                    seen_patients.add(patient_id)
                else:
                    overflow_items.append(item)

            kept_label_items = unique_patient_items[: self.config.max_label_memory]
            if len(kept_label_items) < self.config.max_label_memory:
                remaining_slots = self.config.max_label_memory - len(kept_label_items)
                kept_label_items.extend(overflow_items[:remaining_slots])

            trimmed.extend(kept_label_items)

        trimmed.sort(key=self._memory_priority, reverse=True)
        self.memory_bank = trimmed[: self.config.max_memory]
