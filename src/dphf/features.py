"""Deterministic feature construction for DPHF.

The behavior features are causal: row i depends only on packet i and its
history. Fixed logarithmic scales avoid normalization by future packets.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np


MAX_PACKETS = 50
BYTES_PER_PACKET = 336
RELATIVE_TIME_SCALE_US = 60_000_000.0
INTER_ARRIVAL_SCALE_US = 10_000_000.0
PACKET_RATE_SCALE_PPS = 1_000_000.0


def _log_scale(values: np.ndarray, maximum: float) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=np.float64), 0.0, maximum)
    return np.log1p(values) / np.log1p(maximum)


def _local_rate(times_us: np.ndarray, window: int) -> np.ndarray:
    count = len(times_us)
    result = np.zeros(count, dtype=np.float64)
    for index in range(count):
        start = max(0, index - window + 1)
        packet_span = index - start + 1
        if packet_span <= 1:
            continue
        elapsed_seconds = max((times_us[index] - times_us[start]) / 1_000_000.0, 1e-6)
        result[index] = packet_span / elapsed_seconds
    return result


def build_behavior_features(
    directions: np.ndarray,
    relative_time_us: np.ndarray,
    packet_count: int,
    max_packets: int = MAX_PACKETS,
) -> np.ndarray:
    """Return `[max_packets, 7]` packet-history behavior features."""
    directions = np.asarray(directions)
    relative_time_us = np.asarray(relative_time_us)
    if directions.ndim != 1 or relative_time_us.ndim != 1:
        raise ValueError("directions and relative_time_us must be one-dimensional")
    if len(directions) < max_packets or len(relative_time_us) < max_packets:
        raise ValueError("input arrays are shorter than max_packets")
    count = int(packet_count)
    if count < 0 or count > max_packets:
        raise ValueError(f"packet_count must be between 0 and {max_packets}")

    features = np.zeros((max_packets, 7), dtype=np.float32)
    if count == 0:
        return features

    direction = np.asarray(directions[:count], dtype=np.float64)
    time_us = np.asarray(relative_time_us[:count], dtype=np.float64)
    time_us = np.maximum.accumulate(np.maximum(time_us, 0.0))
    gap_us = np.maximum(np.diff(time_us, prepend=time_us[0]), 0.0)

    direction_change = np.zeros(count, dtype=np.float64)
    if count > 1:
        direction_change[1:] = direction[1:] != direction[:-1]

    same_direction_run = np.ones(count, dtype=np.float64)
    for index in range(1, count):
        same_direction_run[index] = (
            same_direction_run[index - 1] + 1.0
            if direction[index] == direction[index - 1]
            else 1.0
        )

    features[:count] = np.column_stack(
        (
            direction,
            _log_scale(time_us, RELATIVE_TIME_SCALE_US),
            _log_scale(gap_us, INTER_ARRIVAL_SCALE_US),
            direction_change,
            same_direction_run / float(max_packets),
            _log_scale(_local_rate(time_us, 3), PACKET_RATE_SCALE_PPS),
            _log_scale(_local_rate(time_us, 5), PACKET_RATE_SCALE_PPS),
        )
    ).astype(np.float32)
    if not np.isfinite(features).all():
        raise ValueError("behavior feature construction produced NaN or Inf")
    return features


class MSRAAdapter:
    """Convert one common-HDF5 sample to aligned byte and behavior views."""

    def __init__(self, max_packets: int = MAX_PACKETS, bytes_per_packet: int = BYTES_PER_PACKET):
        self.max_packets = int(max_packets)
        self.bytes_per_packet = int(bytes_per_packet)

    def __call__(self, sample: Mapping) -> dict:
        count = min(int(sample["packet_count"]), self.max_packets)
        headers = np.asarray(sample["headers"][:count], dtype=np.uint8)
        payloads = np.asarray(sample["payloads"][:count], dtype=np.uint8)
        directions = np.asarray(sample["directions"], dtype=np.int8)
        times = np.asarray(sample["relative_time_us"])

        raw_packets = np.concatenate((headers, payloads), axis=1)[:, : self.bytes_per_packet]
        signed = raw_packets.astype(np.int16) * directions[:count, None].astype(np.int16)
        byte_packets = np.zeros((self.max_packets, self.bytes_per_packet), dtype=np.int16)
        byte_packets[:count] = signed
        behavior = build_behavior_features(directions, times, count, self.max_packets)

        output = dict(sample)
        output.update(
            byte_packets=byte_packets,
            behavior_features=behavior,
            packet_count=count,
        )
        return output

