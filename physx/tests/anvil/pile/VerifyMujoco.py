"""Verify native MuJoCo with ordinary mj_step and the official Python wheel."""
import argparse
import csv
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'physx/compiler/anvil/vendor/python'))
import mujoco as mj
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('scene', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--warmup', type=int, default=10)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model = mj.MjModel.from_xml_path(str(args.scene.resolve()))
    data = mj.MjData(model)
    mj.set_mjcb_time(time.perf_counter)
    rows = []
    with args.output.with_suffix('.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['step', 'time', 'constraint_ms', 'full_step_ms',
            'contacts', 'rows', 'islands', 'largest_bodies', 'iterations', 'active_rows', 'solver_nnz', 'max_speed'])
        writer.writeheader()
        for step in range(args.steps):
            before = float(data.timer[mj.mjtTimer.mjTIMER_CONSTRAINT].duration)
            start = time.perf_counter()
            mj.mj_step(model, data)
            elapsed = 1000 * (time.perf_counter() - start)
            constraint_ms = 1000 * (float(data.timer[mj.mjtTimer.mjTIMER_CONSTRAINT].duration) - before)
            if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)) or any(data.warning.number):
                raise RuntimeError('Invalid native simulation or MuJoCo warning')
            row = dict(step=step + 1, time=data.time, constraint_ms=constraint_ms, full_step_ms=elapsed,
                contacts=data.ncon, rows=data.nefc, islands=data.nisland,
                largest_bodies=int(max(data.island_nv, default=0)) // 6,
                iterations=int(max(data.solver_niter)), active_rows=int(np.count_nonzero(data.efc_force > 0)),
                solver_nnz=int(sum(data.solver_nnz)), max_speed=float(max(np.linalg.norm(data.qvel.reshape(-1, 6)[:, :3], axis=1))))
            rows.append(row)
            writer.writerow(row)
            stream.flush()
            print(f"step {step + 1}: solve {constraint_ms:.3f} ms, full {elapsed:.3f} ms, rows {data.nefc}, iterations {row['iterations']}", flush=True)
    mj.set_mjcb_time(None)
    measured = rows[args.warmup:]
    if not measured:
        raise RuntimeError('Warmup leaves no measured steps')
    report = dict(version=mj.__version__, package=str(Path(mj.__file__).resolve()),
        dll_sha256=hashlib.sha256((Path(mj.__file__).parent / 'mujoco.dll').read_bytes()).hexdigest(),
        scene_sha256=hashlib.sha256(args.scene.read_bytes()).hexdigest(), command=sys.argv,
        method='Official Python wheel, ordinary mj_step, no contact/rest-distance modification',
        workers=1, timestep=model.opt.timestep, iterations=model.opt.iterations, tolerance=model.opt.tolerance,
        mean_constraint_ms=statistics.mean(row['constraint_ms'] for row in measured),
        median_constraint_ms=statistics.median(row['constraint_ms'] for row in measured),
        mean_full_step_ms=statistics.mean(row['full_step_ms'] for row in measured),
        mean_rows=statistics.mean(row['rows'] for row in measured),
        mean_iterations=statistics.mean(row['iterations'] for row in measured), rows=rows)
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2))
    print(f"Measured mean: solve {report['mean_constraint_ms']:.3f} ms; full step {report['mean_full_step_ms']:.3f} ms", flush=True)


if __name__ == '__main__':
    main()
