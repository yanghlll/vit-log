import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
import os

# Load data
qt_loss = pd.read_csv('logs/quadtree_loss.csv')
bl_loss = pd.read_csv('logs/baseline_loss.csv')
qt_data = pd.read_csv('logs/quadtree_data.csv')
bl_data = pd.read_csv('logs/baseline_data.csv')

# Downsample for plotting (every 100 steps)
qt_loss_ds = qt_loss.iloc[::100].copy()
bl_loss_ds = bl_loss.iloc[::100].copy()
qt_data_ds = qt_data.iloc[::100].copy()
bl_data_ds = bl_data.iloc[::100].copy()

os.makedirs('figures', exist_ok=True)

# ============================================================
# Figure 1: Loss Comparison
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 1a: Full loss curve
ax = axes[0]
ax.plot(qt_loss_ds['step'], qt_loss_ds['loss'], label='Quadtree', alpha=0.8, linewidth=0.8)
ax.plot(bl_loss_ds['step'], bl_loss_ds['loss'], label='Baseline', alpha=0.8, linewidth=0.8)
ax.set_xlabel('Step')
ax.set_ylabel('Loss')
ax.set_title('Training Loss Comparison')
ax.legend()
ax.grid(True, alpha=0.3)

# 1b: Loss difference
merged = pd.merge(qt_loss_ds[['step','loss']], bl_loss_ds[['step','loss']],
                   on='step', suffixes=('_qt','_bl'))
ax = axes[1]
ax.plot(merged['step'], merged['loss_qt'] - merged['loss_bl'], color='red', alpha=0.8, linewidth=0.8)
ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
ax.set_xlabel('Step')
ax.set_ylabel('Loss Difference (Quadtree - Baseline)')
ax.set_title('Loss Gap Over Training')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/01_loss_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved 01_loss_comparison.png")

# ============================================================
# Figure 2: Learning Rate & Throughput
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 2a: LR schedule
ax = axes[0]
ax.plot(qt_data_ds['step'], qt_data_ds['lr'], label='Quadtree', alpha=0.8)
ax.plot(bl_data_ds['step'], bl_data_ds['lr'], label='Baseline', alpha=0.8)
ax.set_xlabel('Step')
ax.set_ylabel('Learning Rate')
ax.set_title('Learning Rate Schedule')
ax.legend()
ax.grid(True, alpha=0.3)

# 2b: Throughput
ax = axes[1]
ax.plot(qt_data_ds['step'], qt_data_ds['total_its'], label='Quadtree', alpha=0.6, linewidth=0.8)
ax.plot(bl_data_ds['step'], bl_data_ds['total_its'], label='Baseline', alpha=0.6, linewidth=0.8)
ax.set_xlabel('Step')
ax.set_ylabel('Total Iterations/sec')
ax.set_title('Training Throughput')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/02_lr_throughput.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved 02_lr_throughput.png")

# ============================================================
# Figure 3: Quadtree-specific metrics
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 3a: Embedding norm
ax = axes[0][0]
ax.plot(qt_data_ds['step'], qt_data_ds['embed_norm'], color='purple', alpha=0.8, linewidth=0.8)
ax.set_xlabel('Step')
ax.set_ylabel('Embedding Norm')
ax.set_title('Quadtree: Embedding Norm')
ax.grid(True, alpha=0.3)

# 3b: Gradient norm (backbone & pfc)
ax = axes[0][1]
ax.plot(qt_data_ds['step'], qt_data_ds['grad_bb'], label='Backbone', alpha=0.7, linewidth=0.8)
ax.plot(qt_data_ds['step'], qt_data_ds['grad_pfc'], label='PFC', alpha=0.7, linewidth=0.8)
ax.set_xlabel('Step')
ax.set_ylabel('Gradient Norm')
ax.set_title('Quadtree: Gradient Norms')
ax.legend()
ax.grid(True, alpha=0.3)

# 3c: Token distribution (stacked area)
ax = axes[1][0]
total_tokens = qt_data_ds['p16'] + qt_data_ds['p32'] + qt_data_ds['p64']
ax.fill_between(qt_data_ds['step'], 0, qt_data_ds['p16']/total_tokens*100, alpha=0.7, label='16px')
ax.fill_between(qt_data_ds['step'], qt_data_ds['p16']/total_tokens*100,
                (qt_data_ds['p16']+qt_data_ds['p32'])/total_tokens*100, alpha=0.7, label='32px')
ax.fill_between(qt_data_ds['step'], (qt_data_ds['p16']+qt_data_ds['p32'])/total_tokens*100,
                100, alpha=0.7, label='64px')
ax.set_xlabel('Step')
ax.set_ylabel('Token Percentage (%)')
ax.set_title('Quadtree: Patch Size Distribution')
ax.legend(loc='center right')
ax.grid(True, alpha=0.3)
ax.set_ylim(85, 100)

# 3d: Token counts
ax = axes[1][1]
ax.plot(qt_data_ds['step'], qt_data_ds['p16'], label='16px', alpha=0.7, linewidth=0.8)
ax.plot(qt_data_ds['step'], qt_data_ds['p32'], label='32px', alpha=0.7, linewidth=0.8)
ax.plot(qt_data_ds['step'], qt_data_ds['p64'], label='64px', alpha=0.7, linewidth=0.8)
ax.set_xlabel('Step')
ax.set_ylabel('Token Count per Batch')
ax.set_title('Quadtree: Token Counts by Patch Size')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/03_quadtree_metrics.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved 03_quadtree_metrics.png")

# ============================================================
# Figure 4: Summary table as image
# ============================================================
fig, ax = plt.subplots(figsize=(12, 6))
ax.axis('off')

table_data = [
    ['Metric', 'Quadtree', 'Baseline'],
    ['Dataset', 'hevc_gop32_quadtree', 'hevc_gop32_video_codec'],
    ['dali_type', 'quadtree', 'decord_hevc_gop32'],
    ['Total Steps', '468,750', '468,750'],
    ['Batch Size', '32', '32'],
    ['LR / Optimizer', '0.001 / AdamW', '0.001 / AdamW'],
    ['Training Duration', '~93.6 hours', '~34.0 hours'],
    ['Throughput (total its/s)', '~240-260', '~490-500'],
    ['Initial Loss', '35.25', '35.18'],
    ['Final Loss', '19.18', '17.86'],
    ['Loss Reduction', '16.07 (45.6%)', '17.32 (49.2%)'],
    ['Tokens/Batch', '62,720 (multi-res)', 'N/A (fixed 196 patches)'],
]

table = ax.table(cellText=table_data, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 1.8)

# Style header
for j in range(3):
    table[0, j].set_facecolor('#4472C4')
    table[0, j].set_text_props(color='white', fontweight='bold')

# Alternate row colors
for i in range(1, len(table_data)):
    color = '#D9E2F3' if i % 2 == 0 else 'white'
    for j in range(3):
        table[i, j].set_facecolor(color)

ax.set_title('Quadtree vs Baseline Training Comparison', fontsize=16, fontweight='bold', pad=20)
plt.savefig('figures/04_summary_table.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved 04_summary_table.png")

print("\nAll figures generated successfully!")
