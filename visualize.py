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

# Deduplicate: quadtree has restart overlaps (step 330000-338040 appear twice)
# Keep the last occurrence (post-restart values)
qt_loss = qt_loss.drop_duplicates(subset='step', keep='last').reset_index(drop=True)
qt_data = qt_data.drop_duplicates(subset='step', keep='last').reset_index(drop=True)

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

ax = axes[0]
ax.plot(qt_loss_ds['step'], qt_loss_ds['loss'], label='Quadtree', alpha=0.8, linewidth=0.8)
ax.plot(bl_loss_ds['step'], bl_loss_ds['loss'], label='Baseline', alpha=0.8, linewidth=0.8)
ax.set_xlabel('Step')
ax.set_ylabel('Loss')
ax.set_title('Training Loss Comparison')
ax.legend()
ax.grid(True, alpha=0.3)

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
# Figure 2: Learning Rate (with restart annotation)
# ============================================================
fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(qt_data_ds['step'], qt_data_ds['lr'], label='Quadtree', alpha=0.8, linewidth=1.2)
ax.plot(bl_data_ds['step'], bl_data_ds['lr'], label='Baseline', alpha=0.8, linewidth=1.2)

# Mark restart points
ax.axvline(x=330000, color='red', linestyle='--', alpha=0.6, linewidth=1)
ax.axvline(x=338000, color='red', linestyle='--', alpha=0.6, linewidth=1)
ax.annotate('Restart #1\n(step 330000)', xy=(330000, 0.0006), fontsize=9,
            color='red', ha='center', va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
ax.annotate('Restart #2\n(step 338000)', xy=(338000, 0.0004), fontsize=9,
            color='red', ha='center', va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

ax.set_xlabel('Step')
ax.set_ylabel('Learning Rate')
ax.set_title('Learning Rate Schedule\n(Quadtree LR diverges after restart #2 due to scheduler re-initialization)')
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/02_learning_rate.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved 02_learning_rate.png")

# ============================================================
# Figure 3: Quadtree Embedding Norm (full view + zoomed)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

ax = axes[0]
ax.plot(qt_data_ds['step'], qt_data_ds['embed_norm'], color='purple', alpha=0.8, linewidth=0.8)
ax.axvline(x=330000, color='red', linestyle='--', alpha=0.4)
ax.axvline(x=338000, color='red', linestyle='--', alpha=0.4)
ax.set_xlabel('Step')
ax.set_ylabel('Embedding Norm')
ax.set_title('Quadtree: Embedding Norm (Full)')
ax.grid(True, alpha=0.3)

# Zoomed in after the spike
ax = axes[1]
mask = qt_data_ds['step'] > 50000
ax.plot(qt_data_ds.loc[mask, 'step'], qt_data_ds.loc[mask, 'embed_norm'],
        color='purple', alpha=0.8, linewidth=0.8)
ax.axvline(x=330000, color='red', linestyle='--', alpha=0.4, label='Restart')
ax.axvline(x=338000, color='red', linestyle='--', alpha=0.4)
ax.set_xlabel('Step')
ax.set_ylabel('Embedding Norm')
ax.set_title('Quadtree: Embedding Norm (After Step 50K)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/03_embedding_norm.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved 03_embedding_norm.png")

# ============================================================
# Figure 4: Quadtree Gradient Norms
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

ax = axes[0]
ax.plot(qt_data_ds['step'], qt_data_ds['grad_bb'], color='#2196F3', alpha=0.7, linewidth=0.8)
ax.axvline(x=330000, color='red', linestyle='--', alpha=0.4)
ax.axvline(x=338000, color='red', linestyle='--', alpha=0.4)
ax.set_xlabel('Step')
ax.set_ylabel('Gradient Norm')
ax.set_title('Quadtree: Backbone Gradient Norm')
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(qt_data_ds['step'], qt_data_ds['grad_pfc'], color='#FF9800', alpha=0.7, linewidth=0.8)
ax.axvline(x=330000, color='red', linestyle='--', alpha=0.4)
ax.axvline(x=338000, color='red', linestyle='--', alpha=0.4)
ax.set_xlabel('Step')
ax.set_ylabel('Gradient Norm')
ax.set_title('Quadtree: PFC (Partial FC) Gradient Norm')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/04_gradient_norms.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved 04_gradient_norms.png")

# ============================================================
# Figure 5: Quadtree Token / Patch Distribution
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 5a: Stacked area (zoomed to show 32px and 64px)
ax = axes[0][0]
total_tokens = qt_data_ds['p16'] + qt_data_ds['p32'] + qt_data_ds['p64']
pct_16 = qt_data_ds['p16'] / total_tokens * 100
pct_32 = qt_data_ds['p32'] / total_tokens * 100
pct_64 = qt_data_ds['p64'] / total_tokens * 100
ax.fill_between(qt_data_ds['step'], 0, pct_16, alpha=0.7, label='16px', color='#2196F3')
ax.fill_between(qt_data_ds['step'], pct_16, pct_16+pct_32, alpha=0.7, label='32px', color='#FF9800')
ax.fill_between(qt_data_ds['step'], pct_16+pct_32, 100, alpha=0.7, label='64px', color='#4CAF50')
ax.set_xlabel('Step')
ax.set_ylabel('Token Percentage (%)')
ax.set_title('Patch Size Distribution (Stacked %)')
ax.legend(loc='center right')
ax.grid(True, alpha=0.3)
ax.set_ylim(85, 100)

# 5b: Individual token counts
ax = axes[0][1]
ax.plot(qt_data_ds['step'], qt_data_ds['p16'], label='16px', alpha=0.7, linewidth=0.8, color='#2196F3')
ax.plot(qt_data_ds['step'], qt_data_ds['p32'], label='32px', alpha=0.7, linewidth=0.8, color='#FF9800')
ax.plot(qt_data_ds['step'], qt_data_ds['p64'], label='64px', alpha=0.7, linewidth=0.8, color='#4CAF50')
ax.set_xlabel('Step')
ax.set_ylabel('Token Count per Batch')
ax.set_title('Token Counts by Patch Size')
ax.legend()
ax.grid(True, alpha=0.3)

# 5c: 32px and 64px only (zoomed)
ax = axes[1][0]
ax.plot(qt_data_ds['step'], qt_data_ds['p32'], label='32px', alpha=0.7, linewidth=0.8, color='#FF9800')
ax.plot(qt_data_ds['step'], qt_data_ds['p64'], label='64px', alpha=0.7, linewidth=0.8, color='#4CAF50')
ax.set_xlabel('Step')
ax.set_ylabel('Token Count per Batch')
ax.set_title('32px & 64px Token Counts (Zoomed)')
ax.legend()
ax.grid(True, alpha=0.3)

# 5d: Total tokens per batch (should be constant)
ax = axes[1][1]
ax.plot(qt_data_ds['step'], total_tokens, color='black', alpha=0.7, linewidth=0.8)
ax.set_xlabel('Step')
ax.set_ylabel('Total Tokens per Batch')
ax.set_title('Total Tokens per Batch (Should be Constant = 62720)')
ax.grid(True, alpha=0.3)
ax.set_ylim(62000, 63500)

plt.tight_layout()
plt.savefig('figures/05_token_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved 05_token_distribution.png")

# ============================================================
# Figure 6: Throughput Comparison
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

ax = axes[0]
ax.plot(qt_data_ds['step'], qt_data_ds['total_its'], label='Quadtree', alpha=0.6, linewidth=0.8, color='#2196F3')
ax.plot(bl_data_ds['step'], bl_data_ds['total_its'], label='Baseline', alpha=0.6, linewidth=0.8, color='#FF9800')
ax.set_xlabel('Step')
ax.set_ylabel('Total Iterations/sec (all GPUs)')
ax.set_title('Training Throughput: Total its/s\n(= samples processed per second across all GPUs)')
ax.legend()
ax.grid(True, alpha=0.3)

# Throughput ratio
ax = axes[1]
merged_tp = pd.merge(qt_data_ds[['step','total_its']], bl_data_ds[['step','total_its']],
                      on='step', suffixes=('_qt','_bl'))
ratio = merged_tp['total_its_bl'] / merged_tp['total_its_qt']
ax.plot(merged_tp['step'], ratio, color='green', alpha=0.7, linewidth=0.8)
ax.axhline(y=ratio.median(), color='red', linestyle='--', alpha=0.6,
           label=f'Median ratio = {ratio.median():.2f}x')
ax.set_xlabel('Step')
ax.set_ylabel('Speedup (Baseline / Quadtree)')
ax.set_title('Baseline Speed Advantage Over Quadtree')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/06_throughput.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved 06_throughput.png")

# ============================================================
# Figure 7: Quadtree Loss vs Grad Norm vs Embed Norm (3-panel)
# ============================================================
fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)

ax = axes[0]
ax.plot(qt_loss_ds['step'], qt_loss_ds['loss'], color='#E91E63', alpha=0.8, linewidth=0.8)
ax.set_ylabel('Loss')
ax.set_title('Quadtree: Loss / Gradient Norm / Embedding Norm Over Training')
ax.grid(True, alpha=0.3)
ax.axvline(x=330000, color='red', linestyle='--', alpha=0.3)
ax.axvline(x=338000, color='red', linestyle='--', alpha=0.3)

ax = axes[1]
ax.plot(qt_data_ds['step'], qt_data_ds['grad_bb'], label='Backbone', alpha=0.7, linewidth=0.8, color='#2196F3')
ax.plot(qt_data_ds['step'], qt_data_ds['grad_pfc'], label='PFC', alpha=0.7, linewidth=0.8, color='#FF9800')
ax.set_ylabel('Gradient Norm')
ax.legend()
ax.grid(True, alpha=0.3)
ax.axvline(x=330000, color='red', linestyle='--', alpha=0.3)
ax.axvline(x=338000, color='red', linestyle='--', alpha=0.3)

ax = axes[2]
ax.plot(qt_data_ds['step'], qt_data_ds['embed_norm'], color='purple', alpha=0.8, linewidth=0.8)
ax.set_xlabel('Step')
ax.set_ylabel('Embedding Norm')
ax.grid(True, alpha=0.3)
ax.axvline(x=330000, color='red', linestyle='--', alpha=0.3, label='Restart')
ax.axvline(x=338000, color='red', linestyle='--', alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig('figures/07_quadtree_combined.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved 07_quadtree_combined.png")

# ============================================================
# Figure 8: Summary table
# ============================================================
fig, ax = plt.subplots(figsize=(14, 7))
ax.axis('off')

table_data = [
    ['Metric', 'Quadtree', 'Baseline'],
    ['Dataset', 'hevc_gop32_quadtree', 'hevc_gop32_video_codec'],
    ['dali_type', 'quadtree', 'decord_hevc_gop32'],
    ['Total Steps', '468,750', '468,750'],
    ['Batch Size', '32', '32'],
    ['LR / Optimizer', '0.001 / AdamW', '0.001 / AdamW'],
    ['Warmup Ratio', '0.1', '0.1'],
    ['Training Duration', '~93.6 hours', '~34.0 hours'],
    ['Throughput (total its/s)', '~240-260', '~490-500'],
    ['Restarts', '2 (step 330K, 338K)', '0'],
    ['LR Schedule Issue', 'Scheduler reset at 338K', 'None'],
    ['Initial Loss', '35.25', '35.18'],
    ['Final Loss', '19.18', '17.86'],
    ['Loss Reduction', '16.07 (45.6%)', '17.32 (49.2%)'],
    ['Tokens/Batch', '62,720 (16/32/64px)', 'Fixed 196 patches'],
]

table = ax.table(cellText=table_data, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 1.8)

for j in range(3):
    table[0, j].set_facecolor('#4472C4')
    table[0, j].set_text_props(color='white', fontweight='bold')

for i in range(1, len(table_data)):
    color = '#D9E2F3' if i % 2 == 0 else 'white'
    for j in range(3):
        table[i, j].set_facecolor(color)

ax.set_title('Quadtree vs Baseline Training Comparison', fontsize=16, fontweight='bold', pad=20)
plt.savefig('figures/08_summary_table.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved 08_summary_table.png")

print("\nAll figures generated successfully!")
