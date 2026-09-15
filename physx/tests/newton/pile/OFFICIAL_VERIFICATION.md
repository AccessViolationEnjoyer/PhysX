# Verification with official MuJoCo 3.3.7

The high native times reproduce using the official MuJoCo wheel. The previous comparison
already called the actual MuJoCo solver (`mj_fwdConstraint`), built locally in Release mode.
It was not an implementation of MuJoCo written for this project.

## Official DLL replay

Copy the existing benchmark executable into an isolated directory and supply the official
Python wheel's `mujoco.dll`. The original build and solver are unchanged. The official DLL's
SHA-256 matches the wheel RECORD; the paths and both DLL hashes are saved in `manifest.json`.

Run the same 200-step prototype capture, followed by five alternating pairs of full solves
on the same state and restored warm start, with eight workers. The official-DLL replay has
exactly the same recorded non-timing snapshot fields as the previous locally built DLL run:
rows, active-row counts, islands, iteration counts, native H+L counts and velocity differences.

| Boxes | Previous native mean (ms) | Official native mean (ms) | Our mean in official-DLL run (ms) |
| ---: | ---: | ---: | ---: |
| 500 | 772.874 | 774.744 | 50.373 |
| 1,000 | 3019.694 | 2834.771 | 233.852 |

Both solvers take three iterations. Velocity increments agree within 1.2e-16. These are
complete fixed-equation solve times including our preparation and writeback; collision
detection and integration are outside the timer. All five repetitions remain in the means.
The official 1,000-box mean is somewhat lower than the earlier run; the seconds-scale cost
and large advantage of our solver persist. This replay alone does not separate compiler
performance from ordinary timing variability.

## Independent ordinary mj_step check

`VerifyMujoco.py` loads the unchanged 500-box XML through the official Python package and
calls `mujoco.mj_step(model, data)` for 30 steps. It does not invoke our solver or adjust
reference acceleration, contacts, forces or warm starts. It runs on the main thread with
no worker pool. A wall-clock callback supplies MuJoCo's built-in phase timers; diagnostics
and file output are outside the full-step timer.

The first step takes 41,233.549 ms in constraint solving (41,241.787 ms for the complete step).
Steps 11-30 average 7,188.559 ms in solving and 7,190.202 ms for the full step. Every step
reaches the 100-iteration cap, so this is not a converged resting-stack throughput result.
No nonfinite position/velocity or native warning is recorded.

This live run follows a different trajectory from the benchmark. Without the shared
rest-distance adjustment, MuJoCo's positive margin separates the initially touching surfaces;
the contact count changes and the pile can form multiple islands. The measured steps average
11,975 scalar rows. These times must not be combined with our fixed-snapshot speedup table.

## What these results establish

The earlier high times are real native-solver measurements, confirmed with the official
binary. They apply to a dense, nearly rigid contact stress test, not default contact settings.
Impedance is `0.9999 0.9999`, versus MuJoCo's default `0.9 0.95`. Both original methods use
these same hard settings. The snapshot contains 48,584 scalar pyramid rows for 500 boxes,
and 102,856 for 1,000 boxes. Timestep is 10 ms, versus MuJoCo's 2 ms default. Newton, the
100-iteration cap, 1e-8 tolerance and 0.01 line tolerance themselves are native defaults.

The local build audit found /O2, /Ob2, SIMD and NDEBUG, with no solver instrumentation or
arithmetic changes. Its only source changes are two export annotations. The local MSVC build
lacks /GL and /LTCG, which is why confirmation against the official wheel was useful.

## Reproduction

The isolated executable and official DLL are preserved under
`physx/compiler/newton/results/pile/official-mujoco-validation/bin`.
From the repository root:

```powershell
physx/compiler/newton/results/pile/official-mujoco-validation/bin/NewtonPileBenchmark.exe physx/compiler/newton/results/pile/shared-snapshots-8/pile500.xml physx/compiler/newton/results/pile/official-mujoco-validation/500-official prototype 201 8 0 0 5
physx/compiler/newton/results/pile/official-mujoco-validation/bin/NewtonPileBenchmark.exe physx/compiler/newton/results/pile/shared-snapshots-8/pile1000.xml physx/compiler/newton/results/pile/official-mujoco-validation/1000-official prototype 201 8 0 0 5
python -B physx/tests/newton/pile/VerifyMujoco.py physx/compiler/newton/results/pile/shared-snapshots-8/pile500.xml physx/compiler/newton/results/pile/official-mujoco-validation/500-ordinary-step
```

The ordinary live check takes several minutes. Raw CSVs, settings, package/version/hash
information, commands and `summary.json` are retained beside the executable directory.
No solver kernel, simulation settings in the maintained benchmark, or SDK source was changed.
