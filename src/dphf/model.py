"""DPHF model; extracted without changing the frozen computational definitions."""
from __future__ import annotations
from typing import Optional
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence

def _masked_mean(sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(sequence.dtype).unsqueeze(-1)
    return (sequence * weights).sum(1) / weights.sum(1).clamp_min(1.0)


class DiscretePacketByteEncoder(nn.Module):
    """Encode categorical bytes without imposing a numeric byte ordering.

    The old encoder maps a byte to ``value / 255`` and flattens packet
    boundaries.  Here each byte is a vocabulary item, consecutive bytes are
    projected as an ordered chunk, and chunks are pooled inside each packet.
    This preserves the local protocol patterns that distinguish otherwise
    similar traffic classes.
    """

    def __init__(
        self,
        dim: int = 128,
        bytes_per_packet: int = 336,
        chunk_size: int = 8,
        byte_embedding_dim: int = 16,
        dropout: float = 0.1,
        categorical_bytes: bool = True,
        use_local_modeling: bool = True,
    ):
        super().__init__()
        if bytes_per_packet % chunk_size:
            raise ValueError("bytes_per_packet must be divisible by chunk_size")
        self.bytes_per_packet = int(bytes_per_packet)
        self.chunk_size = int(chunk_size)
        self.chunks_per_packet = self.bytes_per_packet // self.chunk_size
        self.categorical_bytes = bool(categorical_bytes)
        self.use_local_modeling = bool(use_local_modeling)
        if self.categorical_bytes:
            self.byte_embedding = nn.Embedding(256, byte_embedding_dim)
            self.continuous_projection = None
        else:
            self.byte_embedding = None
            self.continuous_projection = nn.Linear(1, byte_embedding_dim)
        self.chunk_projection = nn.Sequential(
            nn.Linear(byte_embedding_dim * self.chunk_size, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
        )
        self.direction_embedding = nn.Embedding(2, dim)
        self.chunk_position = nn.Parameter(torch.zeros(1, self.chunks_per_packet, dim))
        nn.init.normal_(self.chunk_position, std=0.02)
        if self.use_local_modeling:
            self.local_convolution = nn.Sequential(
                nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim),
                nn.Conv1d(dim, dim, kernel_size=1),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        else:
            self.local_convolution = None
        self.norm = nn.LayerNorm(dim)
        self.chunk_attention = nn.Linear(dim, 1) if self.use_local_modeling else None

    def forward(self, byte_packets: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
        batch, packets, width = byte_packets.shape
        if width != self.bytes_per_packet:
            raise ValueError(f"expected {self.bytes_per_packet} bytes per packet, got {width}")
        # Direction is represented separately.  abs() recovers the original
        # categorical byte IDs from the signed common-HDF5 adapter view.
        byte_values = byte_packets.abs().clamp(0, 255)
        if self.categorical_bytes:
            embedded = self.byte_embedding(byte_values.long())
        else:
            embedded = self.continuous_projection(
                (byte_values.float() / 255.0).unsqueeze(-1)
            )
        chunks = embedded.reshape(
            batch * packets,
            self.chunks_per_packet,
            self.chunk_size * embedded.shape[-1],
        )
        chunks = self.chunk_projection(chunks)
        direction_ids = (directions.reshape(batch * packets) >= 0).long()
        chunks = chunks + self.direction_embedding(direction_ids).unsqueeze(1)
        chunks = chunks + self.chunk_position
        if self.use_local_modeling:
            local = self.local_convolution(chunks.transpose(1, 2)).transpose(1, 2)
            chunks = self.norm(chunks + local)
            attention = self.chunk_attention(chunks).squeeze(-1).softmax(dim=1)
            packet_repr = (chunks * attention.unsqueeze(-1)).sum(1)
        else:
            packet_repr = self.norm(chunks).mean(1)
        return packet_repr.reshape(batch, packets, -1)


class PacketHierarchicalFusionNet(nn.Module):
    """Discrete packet-hierarchical byte/behaviour fusion candidate.

    Fusion happens at aligned packet positions before inter-packet sequence
    modelling.  The byte residual path is always retained, so an unreliable
    behaviour branch cannot erase byte evidence through competitive late
    fusion.  The output contract matches :class:`MSRABTGF`, allowing the same
    loss, evaluator and frozen sample protocol to be reused.
    """

    def __init__(
        self,
        num_classes: int,
        dim: int = 128,
        dropout: float = 0.1,
        chunk_size: int = 8,
        use_behavior: bool = True,
        categorical_bytes: bool = True,
        use_local_modeling: bool = True,
        packet_pooling: str = "attention",
    ):
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least two")
        if packet_pooling not in {"attention", "mean"}:
            raise ValueError("packet_pooling must be 'attention' or 'mean'")
        self.use_behavior = bool(use_behavior)
        self.packet_pooling = packet_pooling
        self.byte_encoder = DiscretePacketByteEncoder(
            dim=dim, chunk_size=chunk_size, dropout=dropout,
            categorical_bytes=categorical_bytes,
            use_local_modeling=use_local_modeling,
        )
        if self.use_behavior:
            self.behavior_projection = nn.Sequential(
                nn.Linear(7, dim), nn.LayerNorm(dim), nn.GELU()
            )
            self.behavior_convolution = nn.Conv1d(dim, dim, kernel_size=3)
            self.packet_gate = nn.Sequential(
                nn.Linear(dim * 3, dim), nn.GELU(), nn.Linear(dim, dim)
            )
            # Start as a conservative 0.5 residual injection.
            nn.init.zeros_(self.packet_gate[-1].weight)
            nn.init.zeros_(self.packet_gate[-1].bias)
        self.packet_norm = nn.LayerNorm(dim)
        self.packet_gru = nn.GRU(dim, dim, batch_first=True)
        self.packet_attention = (
            nn.Linear(dim, 1) if self.packet_pooling == "attention" else None
        )
        self.dropout = nn.Dropout(dropout)
        self.byte_head = nn.Linear(dim, num_classes)
        self.temporal_head = nn.Linear(dim, num_classes) if self.use_behavior else None
        self.fusion_head = nn.Linear(dim, num_classes)

    def forward(
        self,
        byte_packets: torch.Tensor,
        behavior_features: torch.Tensor,
        packet_count: torch.Tensor,
    ) -> dict[str, Optional[torch.Tensor]]:
        max_packets = byte_packets.shape[1]
        mask = (
            torch.arange(max_packets, device=byte_packets.device)[None, :]
            < packet_count.clamp(min=0, max=max_packets)[:, None]
        )
        byte_packets_repr = self.byte_encoder(byte_packets, behavior_features[..., 0])
        byte_packets_repr = byte_packets_repr * mask.unsqueeze(-1)
        byte_repr = _masked_mean(byte_packets_repr, mask)

        gate = None
        if self.use_behavior:
            temporal_packets = self.behavior_projection(behavior_features.float())
            # Left padding keeps the behaviour convolution causal.
            temporal_packets = self.behavior_convolution(
                F.pad(temporal_packets.transpose(1, 2), (2, 0))
            ).transpose(1, 2)
            temporal_packets = F.gelu(temporal_packets) * mask.unsqueeze(-1)
            temporal_repr = _masked_mean(temporal_packets, mask)
            gate = torch.sigmoid(
                self.packet_gate(
                    torch.cat(
                        (
                            byte_packets_repr,
                            temporal_packets,
                            (byte_packets_repr - temporal_packets).abs(),
                        ),
                        dim=-1,
                    )
                )
            )
            packet_sequence = self.packet_norm(
                byte_packets_repr + gate * temporal_packets
            )
        else:
            temporal_repr = None
            packet_sequence = self.packet_norm(byte_packets_repr)
        packet_sequence = packet_sequence * mask.unsqueeze(-1)

        lengths = packet_count.clamp(min=1, max=max_packets).to("cpu")
        packed = pack_padded_sequence(
            packet_sequence, lengths, batch_first=True, enforce_sorted=False
        )
        packed_output, _ = self.packet_gru(packed)
        sequence, _ = torch.nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True, total_length=max_packets
        )
        if self.packet_pooling == "attention":
            attention_logits = self.packet_attention(sequence).squeeze(-1)
            attention_logits = attention_logits.masked_fill(
                ~mask, torch.finfo(attention_logits.dtype).min
            )
            # All-empty rows are not expected in the datasets, but remain finite.
            empty = ~mask.any(dim=1)
            if empty.any():
                attention_logits = attention_logits.clone()
                attention_logits[empty, 0] = 0.0
            attention = attention_logits.softmax(dim=1)
            fused = (sequence * attention.unsqueeze(-1)).sum(1)
        else:
            attention = mask.to(sequence.dtype)
            attention = attention / attention.sum(1, keepdim=True).clamp_min(1.0)
            fused = (sequence * attention.unsqueeze(-1)).sum(1)

        return {
            "logits": self.fusion_head(self.dropout(fused)),
            "byte_logits": self.byte_head(byte_repr),
            "temporal_logits": (
                self.temporal_head(temporal_repr) if self.temporal_head is not None else None
            ),
            "byte_repr": byte_repr,
            "temporal_repr": temporal_repr,
            "fused_repr": fused,
            "gate": gate,
            "alpha": None,
            "packet_attention": attention,
        }
