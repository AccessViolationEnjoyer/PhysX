"""Run full pallet conveyor simulations sequentially, with eight workers each."""
import argparse
import csv
import hashlib
import json
import statistics
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / 'physx/tests/newton/pallet'
BUILD = ROOT / 'physx/compiler/newton'
EXE = BUILD / 'pallet-build/release'
OUT = BUILD / 'results/pallet/comparison'


def summarize(path):
    with path.open(newline='') as stream:
        rows = [{key: float(value) for key, value in row.items()} for row in csv.DictReader(stream)]
    steady = [row for row in rows if row['time'] > 2.0]
    measured = steady or rows
    timings = [row['step_ms'] for row in measured]
    solve_timings = [row['solve_ms'] for row in measured] if measured[0]['solve_ms'] >= 0 else None
    return dict(steps=len(rows), simulated_seconds=rows[-1]['time'],
                first_step_ms=rows[0]['step_ms'], all_mean_ms=statistics.mean(row['step_ms'] for row in rows),
                steady_median_ms=statistics.median(timings), steady_mean_ms=statistics.mean(timings),
                steady_p95_ms=sorted(timings)[int(.95 * (len(timings) - 1))],
                steady_mean_solve_ms=statistics.mean(solve_timings) if solve_timings else None,
                steady_median_solve_ms=statistics.median(solve_timings) if solve_timings else None,
                steady_p95_solve_ms=sorted(solve_timings)[int(.95 * (len(solve_timings) - 1))] if solve_timings else None,
                total_physics_seconds=sum(row['step_ms'] for row in rows) / 1000,
                maximum_sheet_overlap_mm=1000 * max(row['sheet_overlap_m'] for row in rows),
                final_sheet_overlap_mm=1000 * rows[-1]['sheet_overlap_m'],
                maximum_boxes_through_sheet=int(max(row['through_sheet'] for row in rows)),
                final_boxes_through_sheet=int(rows[-1]['through_sheet']),
                final_pallet_speed=rows[-1]['pallet_speed'], pallet_travel_m=rows[-1]['pallet_x'] + 3,
                minimum_box_height=min(row['minimum_box_y'] for row in rows),
                mean_contact_pairs=statistics.mean(row['contact_pairs'] for row in measured),
                mean_rows=statistics.mean(row['rows'] for row in measured) if measured[0]['solve_ms'] >= 0 else None,
                mean_maximum_island_iterations=statistics.mean(row['iterations'] for row in measured),
                steps_at_100_iterations=sum(row['iterations'] >= 100 for row in rows))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seconds', type=float, default=20.0)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--cases', nargs='+', choices=['pgs16-2', 'pgs98-2', 'mujoco', 'prototype', 'pgs98-2-1ms'])
    parser.add_argument('--output', type=Path, default=OUT)
    options = parser.parse_args()
    output = options.output
    output.mkdir(parents=True, exist_ok=True)
    cases = [('pgs16-2', .01, 16, 2), ('pgs98-2', .01, 98, 2),
             ('mujoco', .01, 100, None), ('prototype', .01, 100, None),
             ('pgs98-2-1ms', .001, 98, 2)]
    if options.cases:
        cases = [case for case in cases if case[0] in options.cases]
    report = dict(seconds=options.seconds, threads=options.threads, repeats=options.repeats, runs=[])
    for repeat in range(options.repeats):
        order = cases if repeat % 2 == 0 else list(reversed(cases))
        for name, timestep, iterations, velocity in order:
            prefix = output / f'{name}-{repeat + 1}'
            steps = round(options.seconds / timestep)
            if velocity is None:
                command = [str(EXE / 'MujocoPalletConveyor.exe'), str(SOURCE / 'PalletConveyor.xml'),
                           str(prefix), name, str(steps), str(timestep), str(iterations), str(options.threads)]
            else:
                command = [str(EXE / 'SnippetPalletConveyor.exe'), str(prefix), str(steps), str(timestep),
                           str(iterations), str(velocity), str(options.threads)]
            print(f'Running {name}, repetition {repeat + 1}', flush=True)
            result = subprocess.run(command, capture_output=True, text=True, timeout=600)
            prefix.with_suffix('.log').write_text(result.stdout + result.stderr)
            if result.returncode or 'WARNING' in result.stdout or 'error' in result.stderr.lower():
                raise RuntimeError(result.stdout + result.stderr)
            run = dict(case=name, repeat=repeat + 1, command=command, **summarize(Path(str(prefix) + '-steps.csv')))
            report['runs'].append(run)
            (output / 'comparison.json').write_text(json.dumps(report, indent=2))
            print(f"  {run['steady_median_ms']:.3f} ms/step; max overlap {run['maximum_sheet_overlap_mm']:.6f} mm; "
                  f"final speed {run['final_pallet_speed']:.6f} m/s", flush=True)
    report['hashes'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in list(SOURCE.glob('*.cpp')) + list(SOURCE.glob('*.h')) + list(SOURCE.glob('*.xml'))
                        + [EXE / 'SnippetPalletConveyor.exe', EXE / 'MujocoPalletConveyor.exe']}
    (output / 'comparison.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
