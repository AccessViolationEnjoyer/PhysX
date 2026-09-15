"""Compare complete Newton solves for one connected pile; runs never overlap."""
import argparse
import csv
import hashlib
import json
import math
import statistics
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
BUILD = ROOT / 'physx/compiler/newton'
EXE = BUILD / 'build/Release/NewtonPileBenchmark.exe'
SIZES = {125: (5, 5, 5), 500: (10, 10, 5), 1000: (10, 10, 10)}


def summarize(prefix, boxes, warmup):
    with Path(str(prefix) + '-steps.csv').open(newline='') as stream:
        all_rows = [{key: float(value) for key, value in row.items()} for row in csv.DictReader(stream)]
    rows = all_rows[warmup:]
    if not rows:
        raise RuntimeError('Warm-up leaves no measured frames')
    if not all(math.isfinite(value) for row in all_rows for value in row.values()):
        raise RuntimeError('Nonfinite simulation output')
    timing = sorted(row['solve_ms'] for row in rows)
    connected = [row for row in rows if row['islands'] == 1 and row['largest_island_bodies'] == boxes and row['unconstrained_bodies'] == 0]
    return dict(mean_ms=statistics.mean(timing), median_ms=statistics.median(timing),
                p95_ms=timing[math.ceil(.95 * len(timing)) - 1], max_ms=max(timing),
                first_solve_ms=all_rows[0]['solve_ms'],
                maximum_transient_solve_ms=max(row['solve_ms'] for row in all_rows[:warmup] or all_rows[:1]),
                mean_full_step_ms=statistics.mean(row['step_ms'] for row in rows),
                connected_frames=len(connected), measured_frames=len(rows),
                connected_fraction=len(connected) / len(rows),
                mean_connected_ms=statistics.mean(row['solve_ms'] for row in connected) if connected else None,
                minimum_largest_island=int(min(row['largest_island_bodies'] for row in rows)),
                maximum_islands=int(max(row['islands'] for row in all_rows)),
                mean_contacts=statistics.mean(row['contacts'] for row in rows),
                mean_rows=statistics.mean(row['rows'] for row in rows),
                mean_active_rows=statistics.mean(row['active_rows'] for row in rows),
                mean_iterations=statistics.mean(row['iterations'] for row in rows),
                max_steady_iterations=int(max(row['iterations'] for row in rows)),
                max_iterations=int(max(row['iterations'] for row in all_rows)),
                capped_steps=sum(row['iterations'] >= 100 for row in all_rows),
                max_stationarity=max(row['stationarity'] for row in rows),
                max_projection_error=max(row['projection_error'] for row in rows),
                max_penetration_m=max(row['penetration_m'] for row in all_rows),
                mean_max_linear_speed=statistics.mean(row['max_linear_speed'] for row in rows),
                final_max_linear_speed=rows[-1]['max_linear_speed'],
                final_minimum_height=rows[-1]['minimum_height'], final_maximum_height=rows[-1]['maximum_height'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--executable', type=Path, default=EXE)
    parser.add_argument('--impedance', nargs=2, type=float, default=[0.9999, 0.9999], metavar=('MIN', 'MAX'))
    parser.add_argument('--sizes', nargs='+', type=int, choices=SIZES, default=[125, 500, 1000])
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--warmup', type=int, default=200)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--audit-every', type=int, default=0)
    parser.add_argument('--snapshot-repeats', type=int, default=0)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--methods', nargs='+', choices=['prototype', 'mujoco'], default=['prototype', 'mujoco'])
    parser.add_argument('--output', type=Path, default=BUILD / 'results/pile/comparison')
    args = parser.parse_args()
    exe = args.executable.resolve()
    if not 0 < args.impedance[0] <= args.impedance[1] < 1:
        parser.error('Impedance must satisfy 0 < MIN <= MAX < 1')
    args.output.mkdir(parents=True, exist_ok=True)
    report = dict(steps=args.steps, warmup=args.warmup, repeats=args.repeats, threads=args.threads,
                  audit_every=args.audit_every, profile=args.profile, snapshot_repeats=args.snapshot_repeats,
                  primary_metric='solve_ms includes preparation, dispatch, solve, joins and writeback',
                  executable_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),
                  mujoco_dll_sha256=hashlib.sha256((exe.parent / 'mujoco.dll').read_bytes()).hexdigest(),
                  impedance=args.impedance, scenes={}, runs=[])
    for boxes in args.sizes:
        scene = args.output / f'pile{boxes}.xml'
        subprocess.run([str(exe), '--write-scene', str(scene), *map(str, SIZES[boxes])], check=True)
        if args.impedance != [0.9999, 0.9999]:
            tree = ET.parse(scene)
            tree.getroot().find('./default/geom').set('solimp', f'{args.impedance[0]:.12g} {args.impedance[1]:.12g} 0.001 0.5 2')
            tree.write(scene, encoding='unicode')
        report['scenes'][str(boxes)] = dict(dimensions_width_depth_layers=SIZES[boxes],
                                          sha256=hashlib.sha256(scene.read_bytes()).hexdigest(), path=str(scene))
        for repeat in range(args.repeats):
            methods = ['prototype'] if args.snapshot_repeats else (args.methods if repeat % 2 == 0 else list(reversed(args.methods)))
            for method in methods:
                prefix = args.output / f'{boxes}-{method}-{repeat + 1}'
                command = [str(exe), str(scene), str(prefix), method, str(args.steps), str(args.threads),
                           str(args.audit_every), str(int(args.profile)), str(args.snapshot_repeats)]
                print(f'{boxes} boxes: {method}, run {repeat + 1}, {args.threads} workers', flush=True)
                completed = subprocess.run(command, text=True, capture_output=True, timeout=1800)
                prefix.with_suffix('.log').write_text(completed.stdout + completed.stderr)
                if completed.returncode or 'WARNING' in completed.stdout + completed.stderr:
                    raise RuntimeError(completed.stdout + completed.stderr)
                if args.snapshot_repeats:
                    with Path(str(prefix) + '-snapshot.csv').open(newline='') as stream:
                        snapshots = list(csv.DictReader(stream))
                    if len(snapshots) != 2 * args.snapshot_repeats:
                        raise RuntimeError('Incomplete shared-snapshot comparison')
                    if not all(math.isfinite(float(value)) for row in snapshots
                               for key, value in row.items() if key != 'method'):
                        raise RuntimeError('Nonfinite snapshot output')
                    summaries = {}
                    for solver in ['prototype', 'mujoco']:
                        samples = [row for row in snapshots if row['method'] == solver]
                        times = sorted(float(row['solve_ms']) for row in samples)
                        summaries[solver] = dict(mean_ms=statistics.mean(times), median_ms=statistics.median(times),
                                                 min_ms=min(times), max_ms=max(times),
                                                 mean_iterations=statistics.mean(int(row['iterations']) for row in samples))
                    report['runs'].append(dict(boxes=boxes, repeat=repeat + 1, command=command,
                                               summaries=summaries, snapshots=snapshots))
                    (args.output / 'summary.json').write_text(json.dumps(report, indent=2))
                    print(f"  shared snapshot: prototype {summaries['prototype']['mean_ms']:.4f} ms; "
                          f"MuJoCo {summaries['mujoco']['mean_ms']:.4f} ms; {len(snapshots)} timed solves", flush=True)
                    continue
                result = dict(boxes=boxes, method=method, repeat=repeat + 1, command=command,
                              **summarize(prefix, boxes, args.warmup))
                report['runs'].append(result)
                (args.output / 'summary.json').write_text(json.dumps(report, indent=2))
                print(f"  {result['mean_ms']:.4f} ms mean; connected {result['connected_fraction']:.1%}; "
                      f"{result['mean_rows']:.0f} rows; iterations {result['mean_iterations']:.2f}", flush=True)
    print('Finished; raw CSVs, scene definitions, hashes and commands saved.', flush=True)


if __name__ == '__main__':
    main()
