# ViT Training Log: Quadtree vs Baseline

## Overview

Training log comparison between **Quadtree** (multi-resolution patch) and **Baseline** (fixed-resolution) approaches for video face recognition using ViT-Small on HEVC GOP32 data.

## Configuration

| Metric | Quadtree | Baseline |
|---|---|---|
| Dataset | `hevc_gop32_quadtree` | `hevc_gop32_video_codec` |
| dali_type | `quadtree` | `decord_hevc_gop32` |
| Total Steps | 468,750 | 468,750 |
| Batch Size | 32 | 32 |
| LR / Optimizer | 0.001 / AdamW | 0.001 / AdamW |
| Training Duration | ~93.6 hours | ~34.0 hours |
| Throughput (total its/s) | ~240-260 | ~490-500 |
| Initial Loss | 35.25 | 35.18 |
| **Final Loss** | **19.18** | **17.86** |
| Tokens/Batch | 62,720 (multi-res: 16/32/64px) | Fixed 196 patches |

## Visualizations

### 1. Loss Comparison
![Loss Comparison](figures/01_loss_comparison.png)

### 2. Learning Rate & Throughput
![LR and Throughput](figures/02_lr_throughput.png)

### 3. Quadtree-Specific Metrics
![Quadtree Metrics](figures/03_quadtree_metrics.png)

### 4. Summary Table
![Summary](figures/04_summary_table.png)

## How to Reproduce

```bash
python visualize.py
```

## File Structure

```
.
├── logs/
│   ├── quadtree_hevc_gop32.logger    # Quadtree full training log
│   ├── baseline_mvres_4gpu.logger    # Baseline full training log
│   ├── quadtree_data.csv             # Extracted quadtree metrics
│   ├── quadtree_loss.csv             # Extracted quadtree loss
│   ├── baseline_data.csv             # Extracted baseline metrics
│   └── baseline_loss.csv             # Extracted baseline loss
├── figures/
│   ├── 01_loss_comparison.png
│   ├── 02_lr_throughput.png
│   ├── 03_quadtree_metrics.png
│   └── 04_summary_table.png
├── visualize.py
└── README.md
```
