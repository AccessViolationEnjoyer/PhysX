"""Repeat a prepared-equation comparison with both processes on the same logical CPU."""
import argparse
import ctypes
import json
import statistics
from pathlib import Path
import CompareMujoco as bench


def run(batches, repeats, case, scale, output):
    bench.OUT = output or bench.BUILD / 'results/repeated-27'
    bench.OUT.mkdir(parents=True, exist_ok=True)
    args = bench.prepare(case, scale)
    model, data, pairs, masses, diagonal, jacobian, free = args
    path = bench.OUT / (case + '.txt')
    bench.export(path, args)
    reference = data.efc_aref.copy()
    data.efc_aref[:] = reference + .01*free/model.opt.timestep
    model.opt.tolerance = 1e-12
    bench.mj.mj_fwdConstraint(model, data)
    seed = data.qacc.copy()
    padded = bench.np.zeros((data.nefc, 3))
    padded[:, 2] = model.opt.timestep*data.efc_force
    primal = model.opt.timestep*bench.np.sqrt(diagonal)*(seed-data.qacc_smooth)
    seed_path = bench.OUT / 'seed.txt'
    bench.np.savetxt(seed_path, bench.np.concatenate([padded.ravel(), primal]), fmt='%.17g')
    data.efc_aref[:] = reference
    records = []
    for batch in range(batches):
        if batch % 2 == 0:
            native, native_impulse, native_primal = bench.native_solve(args, seed, 1e-8, repeats)
            ours, values = bench.prototype(path, seed_path, repeats=repeats, label=f'-{batch}')
        else:
            ours, values = bench.prototype(path, seed_path, repeats=repeats, label=f'-{batch}')
            native, native_impulse, native_primal = bench.native_solve(args, seed, 1e-8, repeats)
        impulse = values[:3*data.nefc].reshape(-1, 3)[:, 2]
        primal = values[3*data.nefc:]
        error = bench.comparison_error(primal, impulse, native_primal, native_impulse, diagonal, data.efc_R)
        assert error['scaled_velocity_error'] < 1e-10
        record = dict(native=native, anvil=ours, error=error, ratio=ours['median_ms']/native['median_ms'])
        records.append(record)
        print(batch, 'Anvil', ours['median_ms'], 'MuJoCo', native['median_ms'], 'ratio', record['ratio'], flush=True)
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batches', type=int, default=7)
    parser.add_argument('--repeats', type=int, default=100)
    parser.add_argument('--case', default='grid27-disturbed', choices=['grid27-disturbed', 'rest', 'mass-ratio'])
    parser.add_argument('--scale', default='native')
    parser.add_argument('--output', type=Path)
    options = parser.parse_args()
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    kernel.GetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
    kernel.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    process = kernel.GetCurrentProcess()
    original, system = ctypes.c_size_t(), ctypes.c_size_t()
    if not kernel.GetProcessAffinityMask(process, ctypes.byref(original), ctypes.byref(system)):
        raise ctypes.WinError(ctypes.get_last_error())
    mask = original.value & -original.value
    if not kernel.SetProcessAffinityMask(process, mask):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        records = run(options.batches, options.repeats, options.case, options.scale, options.output)
    finally:
        kernel.SetProcessAffinityMask(process, original.value)
    result = dict(case=options.case, scale=options.scale, affinity_mask=mask, batches=options.batches, repeats=options.repeats, records=records,
                  anvil_ms=statistics.median(r['anvil']['median_ms'] for r in records),
                  native_ms=statistics.median(r['native']['median_ms'] for r in records))
    (bench.OUT / 'comparison.json').write_text(json.dumps(result, indent=2))
    print('Median batch times:', result['anvil_ms'], result['native_ms'])


if __name__ == '__main__':
    main()
