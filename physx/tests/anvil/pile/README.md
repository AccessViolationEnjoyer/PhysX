# Connected pile benchmark

The current Anvil prototype remains faster than native MuJoCo on a single connected pile,
but its absolute cost grows sharply with island size. Neither implementation currently
parallelizes the Anvil solve within an island. This benchmark does not change the solver
kernel or integrate it into PhysX.

The native timings were subsequently reproduced with the official MuJoCo 3.3.7 wheel DLL,
and the scene was independently run using ordinary Python `mj_step`. See
[OFFICIAL_VERIFICATION.md](OFFICIAL_VERIFICATION.md). The ordinary live run reaches the
iteration cap with these nearly rigid contacts; it is recorded separately from the snapshot comparison.

[SOFTER_CONTACTS.md](SOFTER_CONTACTS.md) compares the same geometry with lower contact
impedance for both solvers, using the official DLL. The expensive single-island system
persists with those softer settings.

## Scene and settings

Generate 125 (5x5x5), 500 (10x10x5), or 1,000 (10x10x10) separate 1 kg cubes,
with 0.3 m sides. Alternate layers are offset 0.075 m in both horizontal directions,
so overlapping boxes connect adjacent columns through actual contacts. A shared static floor
alone would not connect separate stacks. Every recorded frame and snapshot contains exactly
one island with all boxes and no unconstrained bodies.

The scene starts touching under gravity; it tests settling, not impact or disturbance recovery.
Both methods use a 10 ms timestep, warm starts, pyramidal friction (coefficient 0.5),
100-iteration cap, normalized tolerance 1e-8, line-search multiplier 0.01,
Euler integration, and eight workers unless specified. Contacts use solref `0.02 1`,
solimp `0.9999 0.9999 0.001 0.5 2`, and a positive 1 mm detection margin.
The same rest-distance adjustment as the pallet benchmark makes the intended surfaces touch;
it is applied to both methods. There is no contact rejection or scene-specific solver change.

Hardware: Intel Core i9-13900KS, Windows x64, Visual Studio 2022 Release, native MuJoCo 3.3.7.
Runs are sequential, with no affinity override, detailed phase profiling, or simultaneous audit.
The model arena is 2 GiB for both methods: the initial 512 MiB native 500-box pilot exhausted
its per-worker scratch allocation. Failed samples are not used.

## Identical settled equations

Capture equations after 200 integrated prototype steps (2 seconds), then run five alternating
pairs of complete solves without integrating either result. Both receive the same captured
`qacc_warmstart` before every call. The CSV retains every repetition, including the first.

| Boxes | Scalar pyramid rows | Our mean (ms) | MuJoCo mean (ms) | Speedup | Iterations, ours / native |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 125 | 10,812 | 4.065 | 20.657 | 5.08x | 4 / 4 |
| 500 | 48,584 | 54.653 | 772.874 | 14.14x | 3 / 3 |
| 1000 | 102,856 | 240.233 | 3019.694 | 12.57x | 3 / 3 |

Both methods produce identical active-row counts in every pair. Their maximum componentwise
velocity-increment difference is below 1.2e-16 (linear m/s or angular rad/s).
Runtime checks verify that position, velocity, simulation time and warm seed remain unchanged.
The native H+L nonzero counts are 133,584, 1,684,464 and 3,706,284, respectively.

These are repeated fixed-equation timings, not independently evolving simulation steps.
Our live capture has warmed our storage; native's first call may first-touch arena pages.
Both rebuild numeric factors, while our retained symbolic ordering can be reused for an
identical pattern. This distinction is why the independent live results are recorded below.

## Independent live simulations

Each method also ran 30 integrated steps (0.3 seconds) from the initial scene. Means below use
steps 11-30. These are short settling pilots, not long-run throughput estimates. Their
contact histories differ slightly; they are not the fixed state used in the preceding table.

| Boxes | Our mean (ms) | MuJoCo mean (ms) | Speedup | Our cold first (ms) | Native cold first (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 500 | 56.372 | 1910.076 | 33.88x | 136.333 | 41344.427 |
| 1000 | 355.110 | 4574.957 | 12.88x | 824.874 | 129952.951 |

For 500 boxes the mean row counts are 48,822 / 48,793; for 1,000 they are
103,322 / 103,289 (ours / native). Mean iterations are 3.25 / 3.40 and 3.70 / 3.90.
There are no capped solves or simulation warnings. Maximum penetration stays below 0.25 um;
final maximum body speed is below 1.9e-6 m/s. Absolute mass-scaled stationarity residuals
are comparable: maxima 0.00197 / 0.00202 for 500 and 0.00561 / 0.00394 for 1,000.
These dimensional residuals are not the normalized stopping-tolerance value.

## One versus eight workers

The 500-box capture was repeated with one worker, then eight again. All 200 non-timing
trajectory records are exactly identical across these runs, as are snapshot row counts,
active-row counts, iterations and paired velocity differences.

| Workers / order | Our mean (ms) | Our median (ms) | Native mean (ms) |
| --- | ---: | ---: | ---: |
| 8, before | 54.653 | 49.067 | 772.874 |
| 1 | 47.662 | 47.208 | 744.804 |
| 8, after | 48.061 | 48.088 | 753.708 |

These measurements predate within-island parallel factorization. The current solver uses PhysX
tasks for sufficiently large Cholesky factors while retaining the serial factor for smaller
islands. Current paired results are recorded in `../PHYSX_INTEGRATION.md`.

Source references: `MujocoAdapter.h` dispatches one task per island. `BlockCholesky.h` uses a
`ParallelExecutor` for large trailing updates, and `DyAnvilSolver.cpp` implements that executor
with PhysX tasks. Native `engine_forward.c` dispatches per island, `engine_island.c` preserves model
DOF ordering, and `engine_solver.c` calls its serial Cholesky routines in `engine_util_solve.c`.

## Timing boundary, reproduction and evidence

`solve_ms` starts after the host has generated equations. Our timing includes translation
into retained Anvil storage, dispatch, complete solving, joins and result writeback.
Native timing covers `mj_fwdConstraint`. Collision detection, host Jacobian/island construction,
integration, diagnostics and file I/O are excluded from both. `step_ms` additionally includes
host simulation work, excluding diagnostics and file I/O. No number here is a PhysX Anvil
simulation-step measurement; that integration remains pending.

From the repository root:

```powershell
cmake --build physx/compiler/anvil/build --config Release --target AnvilPileBenchmark
python physx/tests/anvil/pile/ComparePile.py --sizes 125 500 1000 --steps 201 --repeats 1 --threads 8 --snapshot-repeats 5 --output physx/compiler/anvil/results/pile/shared-snapshots-8
python physx/tests/anvil/pile/ComparePile.py --sizes 500 --steps 201 --repeats 1 --threads 1 --snapshot-repeats 5 --output physx/compiler/anvil/results/pile/shared-snapshots-1
python physx/tests/anvil/pile/ComparePile.py --sizes 500 --steps 201 --repeats 1 --threads 8 --snapshot-repeats 5 --output physx/compiler/anvil/results/pile/shared-snapshots-8-control
python physx/tests/anvil/pile/ComparePile.py --sizes 500 1000 --steps 30 --warmup 10 --repeats 1 --threads 8 --output physx/compiler/anvil/results/pile/pilot-2g
```

The native cold steps are particularly expensive: allow several minutes for the live command.
Omitting `--snapshot-repeats` selects independent live simulations. The executable can also
write arbitrary grid dimensions with `--write-scene output.xml width depth layers`.

Each output directory contains the generated scene, logs, raw per-solve CSVs, binary/scene
hashes and full commands. Snapshot settings include source step/time and stopping parameters.
`physx/compiler/anvil/results/pile/comparison-summary.json` joins the runs and source hashes.
CSV contact counts/residuals describe the solve leading into the recorded post-integration
position, velocity and time. Failed processes, warnings and nonfinite results invalidate runs.

The pallet adapter was mechanically extracted to `MujocoAdapter.h` for reuse by this test.
Comparison against the preserved executable found all 13,000 recorded pallet poses and all
non-timing fields identical over 400 steps. All three maintained CTests pass, including the
zero-allocation/frees audits. Evidence is in `results/pile/adapter-extraction-validation`.
No Anvil kernel, tolerance, factor-update policy, or SDK source was changed for this benchmark.
