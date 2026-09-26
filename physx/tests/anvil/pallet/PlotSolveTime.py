"""Plot per-step time of the five-pallet conveyor for PhysX Anvil and MuJoCo.

usage: python PlotSolveTime.py <results-dir> [solve|step] [output=<file-stem>] [name=label ...]

The results directory holds `<name>-steps.csv` files written by SnippetPalletConveyor
and MujocoPalletConveyor. Without name=label arguments it plots `physx-new`, `mujoco` and
`physx-before`; otherwise it plots the named runs in order, and the first run marks the fall-off. `solve` plots the
`solve_ms` column, the constraint-solver stage of each engine: Anvil island preparation,
solve and writeback in PhysX, and mj_fwdConstraint in MuJoCo. `step` plots `step_ms`, the
whole physics step: simulate and fetchResults in PhysX, including collision detection and
the conveyor contact callback; mj_step1 through mj_Euler in MuJoCo, including collision
detection and the runner's conveyor adjustment. Plotting is outside all timers.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

INK = '#0b0b0b'
MUTED = '#52514e'
GRID = '#e4e3df'
SURFACE = '#fcfcfb'
# Categorical slots 1-3 of the validated reference palette, in fixed order.
COLORS = ['#2a78d6', '#eb6834', '#1baf7a']
SERIES = [
    ('physx-new', 'PhysX Anvil (current)', COLORS[0]),
    ('mujoco', 'MuJoCo 3.13 Anvil', COLORS[1]),
    ('physx-before', 'PhysX Anvil (before)', COLORS[2]),
]


def rolling_mean(values, window):
    kernel = np.ones(window) / window
    padded = np.concatenate([np.full(window - 1, values[0]), values])
    return np.convolve(padded, kernel, mode='valid')


METRICS = {
    'solve': ('solve_ms', 'Solve time', 'constraint-solver time per step', 'solve-time'),
    'step': ('step_ms', 'Step time', 'whole physics step time', 'step-time'),
}


def load(directory, name, column):
    data = np.genfromtxt(directory / f'{name}-steps.csv', delimiter=',', names=True)
    return data['time'], data[column], data['minimum_box_y']


def style(axis):
    axis.set_facecolor(SURFACE)
    axis.grid(color=GRID, linewidth=0.8, which='major')
    axis.tick_params(colors=MUTED, labelsize=9)
    for side in ('top', 'right'):
        axis.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        axis.spines[side].set_color(GRID)


def main():
    directory = Path(sys.argv[1])
    column, axisName, titleName, fileName = METRICS[sys.argv[2] if len(sys.argv) > 2 else 'solve']
    global SERIES
    arguments = sys.argv[3:]
    if arguments and arguments[0].startswith('output='):
        fileName = arguments.pop(0)[len('output='):]
    if arguments:
        SERIES = [(*argument.split('=', 1), COLORS[i]) for i, argument in enumerate(arguments)]
    runs = {name: load(directory, name, column) for name, _, _ in SERIES}
    time, _, lowest = runs[SERIES[0][0]]
    # The first box leaving the belt marks the fall-off.
    fall = time[np.argmax(lowest < 0.3)]

    fig, axes = plt.subplots(2, 1, figsize=(11, 8.2), layout='constrained', facecolor=SURFACE)
    overview, detail = axes

    # Overview: 50-step (0.5 s) rolling mean after the first second, whose start-up
    # settling transient would otherwise dominate the linear scale.
    settled = 1.0
    overviewTop = 0.0
    for name, label, color in SERIES:
        t, solve, _ = runs[name]
        smooth = rolling_mean(solve, 50)
        shown = t >= settled
        overview.plot(t[shown], smooth[shown], color=color, linewidth=2, label=label)
        overviewTop = max(overviewTop, smooth[shown].max())
    overview.axvline(fall, color=MUTED, linestyle='--', linewidth=1)
    overview.set_xlim(settled, time[-1])
    overview.set_ylim(0, overviewTop * 1.08)
    overview.set_ylabel(f'{axisName} per step (ms)', color=INK)
    overview.set_xlabel('Simulated time (s)', color=INK)
    overview.set_title('Whole run after the first second of settling: 0.5 s rolling mean', loc='left', fontsize=11, color=INK)
    overview.annotate('first box falls off the belt', (fall, 0.97), xycoords=('data', 'axes fraction'),
                      xytext=(-6, 0), textcoords='offset points', ha='right', va='top', fontsize=9, color=MUTED)
    style(overview)
    overview.legend(loc='upper left', fontsize=9, frameon=False)

    # Detail: every step around the fall-off, linear scale.
    start, end = fall - 1.0, fall + 1.5
    peaks = []
    for name, label, color in SERIES:
        t, solve, _ = runs[name]
        window = (t >= start) & (t <= end)
        detail.plot(t[window], solve[window], color=color, linewidth=2, label=label)
        peak = np.argmax(np.where(window, solve, -np.inf))
        peaks.append((solve[peak], t[peak], label, color))
    # Peak labels sit in the clear area right of the fall-off, highest first, with leader lines.
    top = max(value for value, _, _, _ in peaks)
    for rank, (value, when, label, color) in enumerate(sorted(peaks, reverse=True)):
        detail.plot(when, value, 'o', markersize=8, color=color, markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
        detail.annotate(f'{label}: peak {value:.1f} ms', (when, value), xytext=(fall + 0.45, top * (0.9 - 0.13 * rank)),
                        va='center', fontsize=9, color=INK,
                        arrowprops=dict(arrowstyle='-', color=MUTED, linewidth=0.8, shrinkA=2, shrinkB=5))
    detail.axvline(fall, color=MUTED, linestyle='--', linewidth=1)
    detail.set_xlim(start, end)
    detail.set_ylim(0, max(value for value, _, _, _ in peaks) * 1.12)
    detail.set_ylabel(f'{axisName} per step (ms)', color=INK)
    detail.set_xlabel('Simulated time (s)', color=INK)
    detail.set_title('Fall-off: every step', loc='left', fontsize=11, color=INK)
    style(detail)
    detail.legend(loc='upper left', fontsize=9, frameon=False)

    fig.suptitle(f'Five pallet conveyors (325 bodies, 10 ms steps, 8 workers): {titleName}',
                 fontsize=12, color=INK, x=0.01, ha='left')
    fig.savefig(directory / f'{fileName}.png', dpi=160, facecolor=SURFACE)
    fig.savefig(directory / f'{fileName}.svg', facecolor=SURFACE)
    plt.close(fig)


if __name__ == '__main__':
    main()
