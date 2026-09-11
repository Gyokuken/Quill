"""Plot the loss curves of every run in runs/train_log.txt and save them to loss_curves.png.
Rerun after training (python plot_losses.py) to refresh the chart."""
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

HERE = Path(__file__).parent
LOG = HERE / 'runs' / 'train_log.txt'
OUT = HERE / 'loss_curves.png'

# categorical slots 1-4 in fixed order (color follows the run number), chrome from the same palette
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']
SURFACE, INK, INK_2, MUTED, GRID, AXIS = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
NAMES = {
    '1_bigram': ('Bigram (bigram.py)', 'Bigram'),
    '2_trainhere_tinytransformer': ('Your trainhere.ipynb model', 'trainhere.ipynb'),
    '3_gpt_small': ('Improved GPT, small', 'GPT small'),
    '4_gpt_medium': ('Improved GPT, medium', 'GPT medium'),
}
CONTEXT_SHOWN = {'1_bigram': 1}   # the bigram trains on 8-character windows but only looks at the current character
Y_MIN, Y_MAX = 1.0, 2.6

# --- parse the log ---------------------------------------------------------------------------
header = re.compile(r'^=== (\S+): ([\d,]+) parameters \| [\d,]+ iters x batch (\d+) x context (\d+) ===')
point = re.compile(r'^\[(\S+)\] step\s+([\d,]+)/[\d,]+ \| train ([\d.]+) \| val ([\d.]+)')
num = lambda s: int(s.replace(',', ''))

runs = {}
for line in LOG.read_text(encoding='utf-8').splitlines():
    if m := header.match(line):
        name, params, batch, context = m.groups()
        runs[name] = dict(params=num(params), context=int(context), chars_per_step=int(batch) * int(context),
                          x=[], train=[], val=[])
    elif m := point.match(line):
        r = runs[m.group(1)]
        r['x'].append(num(m.group(2)) * r['chars_per_step'])   # characters of training text seen
        r['train'].append(float(m.group(3)))
        r['val'].append(float(m.group(4)))
runs = {name: r for name, r in runs.items() if r['x']}

def series_color(name, i):
    return SERIES[int(name.split('_')[0]) - 1] if name[0].isdigit() else SERIES[i % len(SERIES)]

def params_text(p):
    return f'{p / 1e6:.1f}M' if p >= 1e5 else f'{p / 1e3:.0f}K'

# --- style -------------------------------------------------------------------------------------
plt.rcParams.update({
    'font.family': ['Segoe UI', 'DejaVu Sans'], 'font.size': 9.5,
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'axes.edgecolor': AXIS, 'axes.linewidth': 0.6, 'axes.labelcolor': INK_2,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'grid.linestyle': '-',
    'xtick.color': MUTED, 'ytick.color': MUTED, 'xtick.major.size': 0, 'ytick.major.size': 0,
    'lines.solid_capstyle': 'round', 'lines.solid_joinstyle': 'round',
})
chars_fmt = FuncFormatter(lambda v, _: '0' if v == 0 else f'{v / 1e6:g}M')
x_end = max(r['x'][-1] for r in runs.values())

fig = plt.figure(figsize=(12, 9.4), dpi=160)
gs = fig.add_gridspec(2, len(runs), height_ratios=[1.5, 1], left=0.06, right=0.985, top=0.855, bottom=0.085,
                      hspace=0.52, wspace=0.08)
fig.text(0.06, 0.965, 'Loss curves for the runs in results.txt', fontsize=15, fontweight='semibold', color=INK)
fig.text(0.06, 0.93, 'Cross-entropy per character, lower is better. The x-axis is characters of training text seen, '
         'so runs with different batch sizes line up.', fontsize=10, color=INK_2)
fig.text(0.06, 0.905, 'The y-axis is zoomed to 1.0-2.6: every run starts at 4.2-4.7 (random guessing = 4.17), '
         'so the first segment of each line drops in from the top.', fontsize=10, color=INK_2)

# --- top: validation loss of every run ------------------------------------------------------
ax = fig.add_subplot(gs[0, :])
legend_handles = []
for i, (name, r) in enumerate(runs.items()):
    c = series_color(name, i)
    long_name, short_name = NAMES.get(name, (name, name))
    ax.plot(r['x'], r['val'], color=c, lw=1.7, zorder=3)
    ax.plot(r['x'][-1], r['val'][-1], 'o', ms=7, color=c, mec=SURFACE, mew=1.4, zorder=4)
    ax.annotate(f"{short_name}  {r['val'][-1]:.2f}", (r['x'][-1], r['val'][-1]), xytext=(9, 0),
                textcoords='offset points', va='center', fontsize=9.5, color=INK)
    legend_handles.append(Line2D([], [], color=c, lw=2.2,
                                 label=f"{long_name}: {params_text(r['params'])} params, "
                                       f"context {CONTEXT_SHOWN.get(name, r['context'])}"))
ax.set_title('Validation loss: how well each model predicts text it never trained on', loc='left',
             fontsize=11.5, fontweight='semibold', color=INK, pad=10)
ax.set_xlim(0, x_end * 1.2)
ax.set_xticks([t for t in range(0, int(x_end) + 1, 5_000_000)])
ax.xaxis.set_major_formatter(chars_fmt)
ax.set_ylim(Y_MIN, Y_MAX)
ax.set_xlabel('characters of training text seen')
ax.set_ylabel('validation loss')
ax.legend(handles=legend_handles, loc='upper right', frameon=False, fontsize=9.5, labelcolor=INK,
          handlelength=2.2, borderaxespad=0.2)

# --- bottom: training vs validation, one small panel per run --------------------------------
axes = [fig.add_subplot(gs[1, i]) for i in range(len(runs))]
for a in axes[1:]:
    a.sharex(axes[0])
    a.sharey(axes[0])
    a.tick_params(labelleft=False)
for i, (a, (name, r)) in enumerate(zip(axes, runs.items())):
    c = series_color(name, i)
    a.plot(r['x'], r['train'], color=MUTED, lw=1.4, zorder=2)
    a.plot(r['x'], r['val'], color=c, lw=1.7, zorder=3)
    for ys, col in ((r['train'], MUTED), (r['val'], c)):
        a.plot(r['x'][-1], ys[-1], 'o', ms=6, color=col, mec=SURFACE, mew=1.3, zorder=4)
    gap = r['val'][-1] - r['train'][-1]
    a.set_title(NAMES.get(name, (name,))[0], loc='left', fontsize=10, fontweight='semibold', color=INK, pad=19)
    a.text(0, 1.035, f"last eval: train {r['train'][-1]:.2f} | val {r['val'][-1]:.2f} | gap {gap:.2f}",
           transform=a.transAxes, fontsize=9, color=INK_2)
axes[0].set_xlim(0, x_end * 1.05)
axes[0].set_xticks([t for t in range(0, int(x_end) + 1, 10_000_000)])
axes[0].xaxis.set_major_formatter(chars_fmt)
axes[0].set_ylim(Y_MIN, Y_MAX)
axes[0].set_ylabel('loss')
axes[0].legend(handles=[Line2D([], [], color=MUTED, lw=2, label='training loss'),
                        Line2D([], [], color=INK_2, lw=2, label="validation loss (run's color)")],
               loc='lower right', frameon=False, fontsize=9, labelcolor=INK, handlelength=2)
top_of_row = axes[0].get_position().y1
fig.text(0.06, top_of_row + 0.058, 'Training vs validation loss per run: a widening gap means the model is starting '
         'to memorize its training text', fontsize=11.5, fontweight='semibold', color=INK)
fig.text(0.5, 0.025, 'characters of training text seen', ha='center', fontsize=9.5, color=INK_2)

fig.savefig(OUT)
print(f'saved {OUT}')
