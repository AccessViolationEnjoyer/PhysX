"""Bar chart of mean step time on the case conveyor benchmark.

usage: python PlotCases.py <results-dir> name=label [name=label ...]

Each name refers to `<name>-steps.csv` in the results directory. Bars show the mean step
time over steps 100-2000, after the cases have reached belt speed.
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
BAR = '#2a78d6'


def main():
    directory = Path(sys.argv[1])
    series = [argument.split('=', 1) for argument in sys.argv[2:]]
    means = []
    for name, _ in series:
        data = np.genfromtxt(directory / f'{name}-steps.csv', delimiter=',', names=True)
        means.append(data['step_ms'][100:].mean())
    labels = [label for _, label in series]
    fig, axis = plt.subplots(figsize=(10, 0.55 * len(series) + 1.6), layout='constrained', facecolor=SURFACE)
    positions = np.arange(len(series))[::-1]
    axis.barh(positions, means, height=0.6, color=BAR)
    for position, value in zip(positions, means):
        axis.annotate(f'{value:.2f} ms', (value, position), xytext=(6, 0), textcoords='offset points',
                      va='center', fontsize=9, color=INK)
    axis.set_yticks(positions, labels, fontsize=9, color=INK)
    axis.set_xlim(0, max(means) * 1.15)
    axis.set_xlabel('Mean step time, steps 100-2000 (ms)', color=INK)
    axis.set_facecolor(SURFACE)
    axis.grid(axis='x', color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    axis.tick_params(colors=MUTED, labelsize=9)
    for side in ('top', 'right'):
        axis.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        axis.spines[side].set_color(GRID)
    axis.set_title('2,000 cases on ten conveyors (10 ms steps, 8 workers)', loc='left', fontsize=12, color=INK)
    fig.savefig(directory / 'step-time.png', dpi=160, facecolor=SURFACE)
    fig.savefig(directory / 'step-time.svg', facecolor=SURFACE)
    plt.close(fig)


if __name__ == '__main__':
    main()
