"""Ablations retain the frozen initialization order and state-dict names."""
import torch
from torch import nn
from .model import PacketHierarchicalFusionNet
from .fusion import FusionComparison

class ConstantGateLogits(nn.Module):
    def __init__(self,dim):super().__init__();self.dim=dim
    def forward(self,x):return torch.zeros_like(x[...,:self.dim])

def shifted_indices(packet_count, width):
    n = packet_count.clamp(min=0, max=width).long()
    index = torch.arange(width, device=packet_count.device)[None].expand(len(n), -1)
    shift = torch.div(n, 2, rounding_mode='floor')
    mapped = (index - shift[:, None]) % n.clamp_min(1)[:, None]
    return torch.where(index < n[:, None], mapped, index)


def shift_encoded_behavior(value, packet_count):
    # Behavior convolution output is [batch, channels, packets]. Gather is
    # differentiable, preserves the per-flow valid feature multiset and padding.
    index = shifted_indices(packet_count, value.shape[-1])
    return value.gather(2, index[:, None].expand(-1, value.shape[1], -1))


class MisalignedDPHF(PacketHierarchicalFusionNet):
    misalignment_enabled = True

    def forward(self, byte_packets, behavior_features, packet_count):
        if not self.misalignment_enabled:
            return super().forward(byte_packets, behavior_features, packet_count)
        hook = self.behavior_convolution.register_forward_hook(
            lambda module, args, output: shift_encoded_behavior(output, packet_count))
        try:
            return super().forward(byte_packets, behavior_features, packet_count)
        finally:
            hook.remove()

MODES = ("full", "content_only", "behavior_only", "continuous_bytes", "continuous_content_only", "fixed_gate_05", "late_fusion", "packet_concat", "competitive_gate", "misaligned_behavior", "no_local_modeling")

def make_model(classes, mode="full", dim=128, dropout=0.1):
    if mode not in MODES:
        raise ValueError(f"Unknown variant: {mode}")
    if mode in ("behavior_only", "late_fusion", "packet_concat", "competitive_gate"):
        return FusionComparison(classes, mode, dim, dropout)
    if mode == "misaligned_behavior":
        return MisalignedDPHF(classes, dim=dim, dropout=dropout)
    model = PacketHierarchicalFusionNet(classes, dim=dim, dropout=dropout)
    if mode in ("content_only", "continuous_content_only"):
        model.use_behavior = False
        model.behavior_projection = None
        model.behavior_convolution = None
        model.packet_gate = None
        model.temporal_head = None
    if mode in ("continuous_bytes", "continuous_content_only"):
        width = model.byte_encoder.byte_embedding.embedding_dim
        with torch.random.fork_rng(devices=[]):
            projection = nn.Linear(1, width)
        model.byte_encoder.categorical_bytes = False
        model.byte_encoder.byte_embedding = None
        model.byte_encoder.continuous_projection = projection
    if mode == "fixed_gate_05":
        model.packet_gate = ConstantGateLogits(dim)
    if mode == "no_local_modeling":
        model.byte_encoder.use_local_modeling = False
        model.byte_encoder.local_convolution = None
        model.byte_encoder.chunk_attention = None
    return model
