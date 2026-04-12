import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
import os

os.chdir('/tmp/vit-log')
os.makedirs('figures_v2', exist_ok=True)

# ── Load data ──────────────────────────────────────────────
qt5_loss = pd.read_csv('logs/qt_v5_loss.csv')
bl3_loss = pd.read_csv('logs/bl_v3_loss.csv')
qt5_data = pd.read_csv('logs/qt_v5_data.csv')
bl3_data = pd.read_csv('logs/bl_v3_data.csv')

# Also load old runs for 4-way comparison
qt_loss  = pd.read_csv('logs/quadtree_loss.csv')
bl_loss  = pd.read_csv('logs/baseline_loss.csv')
qt_data  = pd.read_csv('logs/quadtree_data.csv')
bl_data  = pd.read_csv('logs/baseline_data.csv')

# Deduplicate old quadtree (had restarts)
qt_loss = qt_loss.drop_duplicates('step', keep='last').reset_index(drop=True)
qt_data = qt_data.drop_duplicates('step', keep='last').reset_index(drop=True)

# Downsample every 100 steps
def ds(df): return df.iloc[::100].copy()
qt5_loss_ds = ds(qt5_loss); bl3_loss_ds = ds(bl3_loss)
qt5_data_ds = ds(qt5_data); bl3_data_ds = ds(bl3_data)
qt_loss_ds  = ds(qt_loss);  bl_loss_ds  = ds(bl_loss)
qt_data_ds  = ds(qt_data);  bl_data_ds  = ds(bl_data)

COLORS = {
    'qt_v5': '#E91E63',   # red-pink
    'bl_v3': '#2196F3',   # blue
    'qt_old': '#FF9800',  # orange (dashed)
    'bl_old': '#9C27B0',  # purple (dashed)
}

# ============================================================
# Fig 1: 4-way Loss Comparison
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

ax = axes[0]
ax.plot(qt5_loss_ds['step'], qt5_loss_ds['loss'], color=COLORS['qt_v5'], lw=1.2, label='Quadtree v5 (DALI)')
ax.plot(bl3_loss_ds['step'], bl3_loss_ds['loss'], color=COLORS['bl_v3'], lw=1.2, label='Baseline v3')
ax.plot(qt_loss_ds['step'],  qt_loss_ds['loss'],  color=COLORS['qt_old'], lw=0.8, ls='--', alpha=0.7, label='Quadtree old')
ax.plot(bl_loss_ds['step'],  bl_loss_ds['loss'],  color=COLORS['bl_old'], lw=0.8, ls='--', alpha=0.7, label='Baseline old')
ax.set_xlabel('Step'); ax.set_ylabel('Loss')
ax.set_title('Training Loss — 4 Runs')
ax.legend(); ax.grid(True, alpha=0.3)

# Focus on v5 vs v3 (new runs only)
ax = axes[1]
ax.plot(qt5_loss_ds['step'], qt5_loss_ds['loss'], color=COLORS['qt_v5'], lw=1.5, label='Quadtree v5')
ax.plot(bl3_loss_ds['step'], bl3_loss_ds['loss'], color=COLORS['bl_v3'], lw=1.5, label='Baseline v3')
merged = pd.merge(qt5_loss_ds[['step','loss']], bl3_loss_ds[['step','loss']], on='step', suffixes=('_qt','_bl'))
ax2 = ax.twinx()
ax2.plot(merged['step'], merged['loss_qt']-merged['loss_bl'], color='gray', lw=0.8, alpha=0.6, ls=':')
ax2.axhline(0, color='gray', ls='--', alpha=0.4)
ax2.set_ylabel('Loss Gap (Qt - Bl)', color='gray')
ax.set_xlabel('Step'); ax.set_ylabel('Loss')
ax.set_title('Quadtree v5 vs Baseline v3 (New Runs)')
ax.legend(loc='upper right'); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures_v2/01_loss_4way.png', dpi=150, bbox_inches='tight')
plt.close(); print("Saved 01_loss_4way.png")

# ============================================================
# Fig 2: Throughput Comparison (old vs new)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

ax = axes[0]
ax.plot(qt5_data_ds['step'], qt5_data_ds['total_its'], color=COLORS['qt_v5'], lw=1.2, label='Quadtree v5 (DALI opt)')
ax.plot(bl3_data_ds['step'], bl3_data_ds['total_its'], color=COLORS['bl_v3'], lw=1.2, label='Baseline v3')
ax.plot(qt_data_ds['step'],  qt_data_ds['total_its'],  color=COLORS['qt_old'], lw=0.8, ls='--', alpha=0.7, label='Quadtree old')
ax.plot(bl_data_ds['step'],  bl_data_ds['total_its'],  color=COLORS['bl_old'], lw=0.8, ls='--', alpha=0.7, label='Baseline old')
ax.set_xlabel('Step'); ax.set_ylabel('Total its/s')
ax.set_title('Throughput — All Runs')
ax.legend(); ax.grid(True, alpha=0.3)

# v5 vs v3 only
ax = axes[1]
merged_tp = pd.merge(qt5_data_ds[['step','total_its']], bl3_data_ds[['step','total_its']], on='step', suffixes=('_qt','_bl'))
ratio = merged_tp['total_its_qt'] / merged_tp['total_its_bl']
ax.plot(merged_tp['step'], ratio, color='green', lw=0.8, alpha=0.7)
ax.axhline(ratio.median(), color='red', ls='--', alpha=0.7, label=f'Median={ratio.median():.2f}x')
ax.axhline(1.0, color='black', ls='-', alpha=0.3)
ax.set_xlabel('Step'); ax.set_ylabel('Speed Ratio (Qt_v5 / Bl_v3)')
ax.set_title('Quadtree v5 Throughput vs Baseline v3')
ax.legend(); ax.grid(True, alpha=0.3)
ax.set_ylim(0.5, 1.5)

plt.tight_layout()
plt.savefig('figures_v2/02_throughput.png', dpi=150, bbox_inches='tight')
plt.close(); print("Saved 02_throughput.png")

# ============================================================
# Fig 3: Embed Norm — qt_v5 vs bl_v3
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

ax = axes[0]
ax.plot(qt5_data_ds['step'], qt5_data_ds['embed_norm'], color=COLORS['qt_v5'], lw=1.0, label='Quadtree v5')
ax.plot(bl3_data_ds['step'], bl3_data_ds['embed_norm'], color=COLORS['bl_v3'], lw=1.0, label='Baseline v3')
ax.set_xlabel('Step'); ax.set_ylabel('Embedding Norm')
ax.set_title('Embedding Norm — v5 vs v3 (Full)')
ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[1]
mask = qt5_data_ds['step'] > 20000
ax.plot(qt5_data_ds.loc[mask,'step'], qt5_data_ds.loc[mask,'embed_norm'], color=COLORS['qt_v5'], lw=1.0, label='Quadtree v5')
mask2 = bl3_data_ds['step'] > 20000
ax.plot(bl3_data_ds.loc[mask2,'step'], bl3_data_ds.loc[mask2,'embed_norm'], color=COLORS['bl_v3'], lw=1.0, label='Baseline v3')
ax.set_xlabel('Step'); ax.set_ylabel('Embedding Norm')
ax.set_title('Embedding Norm — After Warmup (Step >20K)')
ax.legend(); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures_v2/03_embed_norm.png', dpi=150, bbox_inches='tight')
plt.close(); print("Saved 03_embed_norm.png")

# ============================================================
# Fig 4: Gradient Norms — Backbone & PFC
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(18, 12))

for idx, (col, title, ax) in enumerate([
    ('grad_bb', 'Backbone Grad Norm', axes[0][0]),
    ('grad_pfc', 'PFC Grad Norm', axes[0][1]),
]):
    ax.plot(qt5_data_ds['step'], qt5_data_ds[col], color=COLORS['qt_v5'], lw=0.8, alpha=0.8, label='Quadtree v5')
    ax.plot(bl3_data_ds['step'], bl3_data_ds[col], color=COLORS['bl_v3'], lw=0.8, alpha=0.8, label='Baseline v3')
    ax.plot(qt_data_ds['step'],  qt_data_ds[col],  color=COLORS['qt_old'], lw=0.6, ls='--', alpha=0.5, label='Quadtree old')
    ax.set_xlabel('Step'); ax.set_ylabel('Grad Norm')
    ax.set_title(title); ax.legend(); ax.grid(True, alpha=0.3)

# Backbone grad: zoomed after step 50K
ax = axes[1][0]
m = qt5_data_ds['step'] > 50000
ax.plot(qt5_data_ds.loc[m,'step'], qt5_data_ds.loc[m,'grad_bb'], color=COLORS['qt_v5'], lw=0.8, label='Qt v5')
m2 = bl3_data_ds['step'] > 50000
ax.plot(bl3_data_ds.loc[m2,'step'], bl3_data_ds.loc[m2,'grad_bb'], color=COLORS['bl_v3'], lw=0.8, label='Bl v3')
ax.set_xlabel('Step'); ax.set_ylabel('Backbone Grad Norm')
ax.set_title('Backbone Grad Norm — After Step 50K'); ax.legend(); ax.grid(True, alpha=0.3)

# PFC grad: zoomed
ax = axes[1][1]
ax.plot(qt5_data_ds.loc[m,'step'], qt5_data_ds.loc[m,'grad_pfc'], color=COLORS['qt_v5'], lw=0.8, label='Qt v5')
ax.plot(bl3_data_ds.loc[m2,'step'], bl3_data_ds.loc[m2,'grad_pfc'], color=COLORS['bl_v3'], lw=0.8, label='Bl v3')
ax.set_xlabel('Step'); ax.set_ylabel('PFC Grad Norm')
ax.set_title('PFC Grad Norm — After Step 50K'); ax.legend(); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures_v2/04_grad_norms.png', dpi=150, bbox_inches='tight')
plt.close(); print("Saved 04_grad_norms.png")

# ============================================================
# Fig 5: Quadtree v5 Token Distribution
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(18, 12))

total = qt5_data_ds['p16'] + qt5_data_ds['p32'] + qt5_data_ds['p64']
pct16 = qt5_data_ds['p16']/total*100
pct32 = qt5_data_ds['p32']/total*100

ax = axes[0][0]
ax.fill_between(qt5_data_ds['step'], 0, pct16, alpha=0.7, label='16px', color='#2196F3')
ax.fill_between(qt5_data_ds['step'], pct16, pct16+pct32, alpha=0.7, label='32px', color='#FF9800')
ax.fill_between(qt5_data_ds['step'], pct16+pct32, 100, alpha=0.7, label='64px', color='#4CAF50')
ax.set_ylim(85, 100); ax.set_xlabel('Step'); ax.set_ylabel('%')
ax.set_title('Quadtree v5: Patch Size Distribution'); ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[0][1]
ax.plot(qt5_data_ds['step'], qt5_data_ds['p16'], color='#2196F3', lw=0.8, label='16px')
ax.plot(qt5_data_ds['step'], qt5_data_ds['p32'], color='#FF9800', lw=0.8, label='32px')
ax.plot(qt5_data_ds['step'], qt5_data_ds['p64'], color='#4CAF50', lw=0.8, label='64px')
ax.set_xlabel('Step'); ax.set_ylabel('Token Count')
ax.set_title('Quadtree v5: Token Counts'); ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[1][0]
ax.plot(qt5_data_ds['step'], qt5_data_ds['p32'], color='#FF9800', lw=0.8, label='32px')
ax.plot(qt5_data_ds['step'], qt5_data_ds['p64'], color='#4CAF50', lw=0.8, label='64px')
ax.set_xlabel('Step'); ax.set_ylabel('Token Count')
ax.set_title('32px & 64px Zoom'); ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[1][1]
ax.plot(qt5_data_ds['step'], total, color='black', lw=0.8)
ax.set_ylim(61000, 64000)
ax.set_xlabel('Step'); ax.set_ylabel('Total Tokens')
ax.set_title('Total Tokens/Batch (= 62720 constant)'); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures_v2/05_token_dist.png', dpi=150, bbox_inches='tight')
plt.close(); print("Saved 05_token_dist.png")

# ============================================================
# Fig 6: Quadtree v5 — Loss / GradNorm / EmbedNorm stacked
# ============================================================
fig, axes = plt.subplots(3, 1, figsize=(16, 14), sharex=True)

axes[0].plot(qt5_loss_ds['step'], qt5_loss_ds['loss'], color=COLORS['qt_v5'], lw=0.8)
axes[0].plot(bl3_loss_ds['step'], bl3_loss_ds['loss'], color=COLORS['bl_v3'], lw=0.8, alpha=0.7)
axes[0].set_ylabel('Loss')
axes[0].set_title('Quadtree v5 vs Baseline v3: Loss / Gradient / Embedding')
axes[0].legend(['Quadtree v5', 'Baseline v3']); axes[0].grid(True, alpha=0.3)

axes[1].plot(qt5_data_ds['step'], qt5_data_ds['grad_bb'], color=COLORS['qt_v5'], lw=0.7, label='Qt v5 BB')
axes[1].plot(bl3_data_ds['step'], bl3_data_ds['grad_bb'], color=COLORS['bl_v3'], lw=0.7, alpha=0.7, label='Bl v3 BB')
axes[1].plot(qt5_data_ds['step'], qt5_data_ds['grad_pfc'], color=COLORS['qt_v5'], lw=0.7, ls='--', alpha=0.6, label='Qt v5 PFC')
axes[1].plot(bl3_data_ds['step'], bl3_data_ds['grad_pfc'], color=COLORS['bl_v3'], lw=0.7, ls='--', alpha=0.6, label='Bl v3 PFC')
axes[1].set_ylabel('Gradient Norm'); axes[1].legend(ncol=2); axes[1].grid(True, alpha=0.3)

axes[2].plot(qt5_data_ds['step'], qt5_data_ds['embed_norm'], color=COLORS['qt_v5'], lw=0.8, label='Quadtree v5')
axes[2].plot(bl3_data_ds['step'], bl3_data_ds['embed_norm'], color=COLORS['bl_v3'], lw=0.8, alpha=0.8, label='Baseline v3')
axes[2].set_xlabel('Step'); axes[2].set_ylabel('Embedding Norm')
axes[2].legend(); axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures_v2/06_combined.png', dpi=150, bbox_inches='tight')
plt.close(); print("Saved 06_combined.png")

# ============================================================
# Fig 7: Summary comparison table (all 4 runs)
# ============================================================
fig, ax = plt.subplots(figsize=(16, 7))
ax.axis('off')

table_data = [
    ['Metric', 'Quadtree v5\n(DALI opt)', 'Baseline v3', 'Quadtree old', 'Baseline old'],
    ['Start Date', '2026-04-10', '2026-04-10', '2026-04-03', '2026-04-07'],
    ['Duration', '~38.9 h', '~34.8 h', '~93.6 h', '~34.0 h'],
    ['Throughput (median)', '~441 its/s', '~483 its/s', '~240 its/s', '~490 its/s'],
    ['Restarts', '0', '0', '2', '0'],
    ['LR Schedule Issue', 'None', 'None', 'Yes (restart #2)', 'None'],
    ['Initial Loss', '35.60', '35.24', '35.25', '35.18'],
    ['Loss @ 100K', '21.57', '21.79', '23.29', '21.71'],
    ['Loss @ 200K', '20.65', '20.67', '22.16', '20.85'],
    ['Loss @ 300K', '19.76', '19.92', '21.26', '19.96'],
    ['Loss @ 400K', '18.39', '18.78', '19.58', '18.89'],
    ['Loss @ 450K', '17.80', '18.23', '19.35', '18.09'],
    ['Final Loss', '17.69', '17.94', '19.18', '17.86'],
    ['Final embed_norm', '7.46', '20.7', '6.65', 'N/A'],
    ['dali_type', 'quadtree+DALI', 'decord_hevc', 'quadtree', 'decord_hevc'],
]

table = ax.table(cellText=table_data, loc='center', cellLoc='center')
table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1.0, 1.7)

header_colors = ['#37474F', '#E91E63', '#2196F3', '#FF9800', '#9C27B0']
for j, c in enumerate(header_colors):
    table[0, j].set_facecolor(c)
    table[0, j].set_text_props(color='white', fontweight='bold')

# Highlight best values in loss rows
loss_rows = list(range(7, 14))  # Loss @ xxx rows + Final Loss
for i in loss_rows:
    vals = []
    for j in range(1, 5):
        try: vals.append(float(table_data[i][j]))
        except: vals.append(float('inf'))
    best_j = vals.index(min(vals)) + 1
    table[i, best_j].set_facecolor('#C8E6C9')  # green highlight

for i in range(1, len(table_data)):
    color = '#F5F5F5' if i % 2 == 0 else 'white'
    for j in range(len(table_data[0])):
        if table[i, j].get_facecolor() == (1,1,1,1) or all(c == color_val for c,color_val in zip(table[i,j].get_facecolor()[:3], (1,1,1))):
            if table[i, j].get_facecolor() != (0.784, 0.902, 0.788, 1.0):
                table[i, j].set_facecolor(color)

ax.set_title('All 4 Runs Comparison  (green = best at each metric)',
             fontsize=14, fontweight='bold', pad=20)
plt.savefig('figures_v2/07_summary_table.png', dpi=150, bbox_inches='tight')
plt.close(); print("Saved 07_summary_table.png")

print("\nAll figures generated!")
