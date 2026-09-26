"""Plot completed pallet-conveyor measurements; plotting is outside all timers."""
import json
import statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / 'physx/compiler/anvil/results/pallet/comparison'


def main():
    report = json.loads((OUT / 'comparison.json').read_text())
    names = ['pgs16-2', 'pgs98-2', 'mujoco', 'prototype', 'pgs98-2-1ms']
    labels = ['PGS 16+2', 'PGS 98+2', 'MuJoCo Newton', 'Our Anvil', 'PGS 98+2, 1 ms']
    colors = ['#b95739', '#d69b36', '#536db5', '#27856c', '#805ba3']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout='constrained')
    bars = []
    for name, label, color in zip(names, labels, colors):
        runs = [run for run in report['runs'] if run['case'] == name]
        bars.append(statistics.median(run['total_physics_seconds'] / run['simulated_seconds'] * 10 for run in runs))
        trajectory = np.genfromtxt(OUT / f'{name}-1-steps.csv', delimiter=',', names=True)
        axes[0].plot(trajectory['time'], np.maximum(1e-6, trajectory['sheet_overlap_m'] * 1000),
                     label=label, color=color, linewidth=1.4)
    axes[0].axhline(1, color='#777777', linestyle='--', linewidth=.8)
    axes[0].set(yscale='log', xlabel='Simulated time (s)', ylabel='Maximum sheet-plane intrusion (mm)',
                title='Thin-sheet behavior', ylim=(1e-5, 10))
    axes[0].legend(fontsize=8, loc='lower right')
    axes[0].grid(alpha=.2, which='both')
    axes[1].barh(labels, bars, color=colors)
    axes[1].invert_yaxis()
    axes[1].set(xlabel='Wall time per 10 ms of simulated time (ms)', title='Full simulation cost, including settling')
    axes[1].axvline(10, color='#777777', linestyle='--', linewidth=.8)
    axes[1].set_xlim(0, max(bars) * 1.2)
    for i, value in enumerate(bars):
        axes[1].text(value + max(bars) * .02, i, f'{value:.2f}', va='center', fontsize=9)
    fig.suptitle('Five pallet conveyors: 300 boxes + 20 slipsheets + 5 pallets; eight workers', fontsize=12)
    fig.savefig(OUT / 'pallet-comparison.png', dpi=170)
    fig.savefig(OUT / 'pallet-comparison.svg')
    plt.close(fig)


if __name__ == '__main__':
    main()
