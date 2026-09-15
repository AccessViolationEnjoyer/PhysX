"""Plot recorded sliding motion; this does not run or time a solver."""
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    variants = [('physical-pgs', 'PhysX PGS', '#228833', ':'),
                ('physical-newton', 'PhysX-compatible Newton', '#4477AA', '-'),
                ('associated-newton', 'Convex native experiment', '#CC6677', '-'),
                ('mujoco-flat', 'Official MuJoCo Newton', '#AA3377', '--')]
    figure, axes = plt.subplots(1, 3, figsize=(12.8, 4.2))
    for axis, speed in zip(axes, ['0.2', '1', '2']):
        for prefix, label, color, style in variants:
            path = args.directory / f'{prefix}-{speed}.csv'
            rows = list(csv.DictReader(path.open()))
            if prefix != 'mujoco-flat':
                rows = [row for row in rows if row['case'] == 'flat' and row['body'] == '0']
            rows = [row for row in rows if float(row['time']) <= .5]
            times = [0.] + [float(row['time']) for row in rows]
            heights = [0.] + [1000 * (float(row['y']) - .5) for row in rows]
            axis.plot(times, heights, label=label, color=color, linestyle=style, linewidth=1.9)
        axis.set_title(f'Initial sliding speed: {speed} m/s', fontsize=11)
        axis.set_xlabel('Time (s)')
        axis.set_ylabel('Rise above initial height (mm)')
        axis.set_xlim(0, .5)
        axis.grid(True, linewidth=.5, alpha=.3)
        axis.spines[['top', 'right']].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc='lower center', ncol=4, frameon=False, fontsize=9)
    figure.suptitle('Sliding-contact comparison at a 10 ms timestep', fontsize=14, fontweight='bold')
    figure.text(.5, .88, '1 kg box, rotation locked, friction 0.5. PGS and physical Newton stay on the surface.', ha='center', fontsize=10)
    figure.subplots_adjust(top=.76, bottom=.22, wspace=.35, left=.055, right=.98)
    for extension in ['svg', 'png']:
        figure.savefig(args.directory / f'sliding-rise.{extension}', dpi=160)
    plt.close(figure)


if __name__ == '__main__':
    main()