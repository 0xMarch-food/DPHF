# DPHF

PyTorch model implementation accompanying **DPHF: Compact Hierarchical Content–Behavior Learning for Robotic Network Intrusion Detection**.

Authors: Ting Su, Yutian Feng, Jingyi Xu, and Jiazhong Lu.

## Scope

This repository provides the DPHF architecture, causal input-feature construction and model variants. It does not include training/evaluation pipelines, dataset splits, raw or processed traffic, experiment result tables or pretrained checkpoints. These files alone do not reproduce the complete paper experiments. No journal acceptance or publication DOI is claimed.

## Datasets

Original dataset files must be downloaded from their creators' official Zenodo records:

- [NAVBOT25](https://zenodo.org/records/18210878)
- [ROSIDS23](https://zenodo.org/records/10014434)

Download and use the datasets in accordance with the licenses and terms of their respective records, and cite the original dataset publications. Dataset licenses are separate from this repository's MIT code license. No original or processed traffic data are redistributed here.

The paper evaluates a seven-class NAVBOT25 task (merging original port-scanning and reverse-shell labels into a connection-attempt class) and the original five-class ROSIDS23 task. Downloading these datasets alone does not reconstruct the paper's preprocessing or fixed splits; those pipelines are outside this model-only release.

## Installation

Use Python 3.10 or later and install from this directory:

```bash
python -m pip install .
```

Dependencies are PyTorch and NumPy. The original model experiments used PyTorch 2.1.2 and NumPy 1.26.4. DPHF uses a GRU and does not require Mamba, CUDA or a GPU for a forward pass.

## Model inputs and outputs

| Input | Shape | Meaning |
| --- | --- | --- |
| `byte_packets` | `[batch, packets, 336]` | 80 header bytes followed by 256 payload bytes; valid byte IDs are 0–255. Signed byte views are also accepted; direction is encoded separately. |
| `behavior_features` | `[batch, packets, 7]` | Direction, scaled elapsed time, scaled interarrival time, direction change, normalized same-direction run length, scaled local packet rates with windows 3 and 5. |
| `packet_count` | `[batch]` | Valid packet-prefix lengths, at most the input packet dimension. |

The paper uses up to 50 packets per flow. Padded positions follow the valid prefix. Use direction +1 or -1 for valid packets. `features.py` provides `build_behavior_features` for feature construction from directions and relative arrival times in microseconds; it uses fixed logarithmic scales and no future packets. `MSRAAdapter` constructs aligned inputs from header/payload arrays and packet metadata, without reading files.

```python
import torch
from dphf import make_model

model = make_model(classes=7, mode="full").eval()
byte_packets = torch.randint(0, 256, (2, 50, 336))
behavior_features = torch.zeros(2, 50, 7)
behavior_features[..., 0] = 1.0
packet_count = torch.tensor([10, 25], dtype=torch.long)

with torch.no_grad():
    output = model(byte_packets, behavior_features, packet_count)
    probabilities = output["logits"].softmax(dim=-1)

print(probabilities.shape)  # torch.Size([2, 7])
```

This example uses synthetic inputs and randomly initialized parameters; it does not produce meaningful attack predictions. Use `classes=5` for the five-class setting. Call `model.to(device)` and move all three inputs to the same device for GPU use.

All variants return a dictionary with `logits` of shape `[batch, classes]`; additional fields depend on the variant. The full model also returns intermediate representations, gate values and packet attention. Full-model registered parameter counts are 262,423 for seven classes and 261,649 for five classes. Auxiliary branch heads are retained for compatibility; the paper optimizes only the main classification cross-entropy.

## Variants

`make_model(classes, mode)` accepts:

| Mode | Description |
| --- | --- |
| `full` | Full DPHF |
| `content_only` | Content with direction; no independent behavior branch |
| `behavior_only` | Behavior branch and flow aggregation |
| `continuous_bytes` | Continuous byte projection with full fusion |
| `continuous_content_only` | Continuous bytes with no independent behavior branch |
| `fixed_gate_05` | Fixed 0.5 residual behavior gate |
| `late_fusion` | Separately aggregate views, then fuse |
| `packet_concat` | Concatenate aligned packet representations and project |
| `competitive_gate` | Competitive packet-level fusion |
| `misaligned_behavior` | Shift encoded behavior within each valid prefix |
| `no_local_modeling` | Remove local convolution and chunk attention together |

Content-only retains direction. The misalignment variant is an artificial correspondence/ordering control, not an online-causality claim. No-local-modeling removes a module bundle, not convolution alone.

## Files

```text
src/dphf/model.py      Main DPHF architecture
src/dphf/features.py   Causal feature construction and in-memory input adapter
src/dphf/variants.py   Model factory and ablation interventions
src/dphf/fusion.py     Alternative fusion models
src/dphf/__init__.py   Public imports
```

## License and citation

The author-owned code is distributed under the MIT license. See `LICENSE` and `THIRD_PARTY_NOTICES.md`. Software citation metadata are in `CITATION.cff`.
