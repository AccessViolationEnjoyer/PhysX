# Newton solver and native PhysX integration

The selected Newton solver uses incremental Cholesky updates and now has a native PhysX CPU
adapter, selected with `PxSceneDesc::solverType = PxSolverType::eNEWTON`. The numerical core lives
in `physx/source/lowleveldynamics/src/newton/core`; the native adapter is in its parent directory.
This test directory builds the same core for standalone validation and MuJoCo comparisons.
PGS and TGS retain their numerical kernels and compiler settings. The shared CPU task setup
selects Newton at the island boundary. Articulations, GPU dynamics and the direct GPU API are
rejected for Newton scenes. Current optimization experiments remain separate under
`physx/compiler/newton/experiments`.

`NewtonSolver.cpp` contains the convex objective, weighted pyramidal projection, stopping
criteria and safeguarded line search. `IncrementalCholesky.h` orders the body graph with AMD,
constructs the permuted sparse matrix in linear time, and chooses incremental updates or a
fresh factorization using structural work estimates. `BlockCholesky.h` supplies small rigid-body
block kernels, retaining scalar LLT for thin factors and failed block pivots. Positive updates
precede downdates. A numerical loss of rank triggers a complete current-Hessian rebuild.

A native island uses one ordinary `solveNewton` call. Each new island and timestep starts a
fresh numerical factor; there is no cross-timestep numerical-factor reuse. Scratch capacity and
symbolic analysis are retained, with symbolic reuse requiring an identical full sparse pattern.
Settings remain a 100-iteration cap, normalized tolerance `1e-8` and line-search multiplier `0.01`.
A converged warm start exits before Hessian assembly. The earlier five-pallet
MuJoCo-equation benchmark measured **2.03 ms** for preparation, dispatch, solving and writeback,
versus **6.02 ms** for native MuJoCo. These are historical measurements of that adapter and
equation set, not native PhysX timing claims. Native PhysX performance is still under
investigation. See [PERFORMANCE.md](PERFORMANCE.md) for the measured settings, tails and checks.

Storage is retained across solves and timesteps. `NewtonStorage.h` separates active vector/matrix
sizes from capacity; shrinking and regrowing within capacity does not allocate. Hessian assembly,
permutations, symbolic-analysis work arrays and line search also reuse storage. Capacity grows
geometrically when needed. The caller retains `Problem`, `Result` and an explicit `Workspace`,
then calls `solveNewton(problem, settings, result, workspace, previous)`. The call returns a
`SolveStatus`; `previous` may be `&result`. A workspace is owned by one solve at a time.
The native context retains a pool of these workspaces across islands and timesteps.

Scalar rows use 128-byte retained contact records; extra tangent rows are stored only for
three-row blocks. The full Contact is an input builder passed to problem.addContact(). Use
clearContacts() to rebuild a mixed problem while retaining both arrays' capacity. Producers
may accumulate nonzero column counts while emitting coefficients and call
prepareProblemFromColumnCounts(); reset the counts before each preparation, since that call
consumes them into CSC insertion cursors. The ordinary prepareProblem() counts its own entries.

Eigen's public wrappers allocate even when symbolic analysis is cached. At build time,
`PrepareStorageKernels.py` adapts the pinned Eigen AMD, symbolic-analysis and numeric LLT routines
to retained work arrays. It preserves their arithmetic and licenses; generated code lives in the
build directory. This removes the wrapper allocations without introducing another solver.

Prepared equations use mass-scaled body coordinates and positive diagonal constraint compliance.
Bilateral three-row blocks represent anchors and springs. Friction uses a square pyramid, with
matching tangent compliances. The MuJoCo comparison exports its four nonnegative pyramid-edge
variables directly; each scalar edge occupies one packed row in the solver. The prepared file format still reserves
three components, and the loader removes the two unused components.
These edge equations have MuJoCo's compliance, which is not interchangeable with a three-component
contact block's compliance. See PERFORMANCE.md for the measured comparison.

## PhysX integration

[PHYSX_INTEGRATION.md](PHYSX_INTEGRATION.md) records the backup and integration design, including
the shared PGS task pipeline and a possible future adaptive handoff. Adaptive PGS-to-Newton
switching is not implemented.

Native Newton uses the existing island startup and completion tasks, with its own constraint
preparation and solve between them. Scene settings default to `newtonMaxIterations = 100`,
`newtonTolerance = 1e-8` and `newtonRegularization = 1e-4`. Contacts use MuJoCo's four-edge
pyramidal formulation and reference-acceleration equation. Hard joint rows use the same reference
policy; spring rows retain their specified implicit stiffness and damping. Invalid settings and
unsupported articulation/GPU insertion paths are checked in Release as well as Checked builds.

The `native/` tests cover SDK rejection and allocation-recovery paths, joints, contact modification,
compliance, friction, rolling and scale/orientation cases. Native profile CSV fields named
`prepare_cpu_ms` and `solve_cpu_ms` sum elapsed task durations measured by `steady_clock`; they
are not scheduled CPU time and may overlap across workers. `solve_wall_ms` spans the parallel
solve interval, `island_wall_ms` includes preparation and writeback, and `step_ms` is the full step.

On the five-pallet conveyor, the MuJoCo formulation reduced mean island wall time from
34.964 ms to 2.352 ms over frames 100-220 and reduced the summed iteration count from
267.744 to 16.545; the parallel solve span was 1.820 ms. The 2,000-frame validation
averaged 3.042 ms over frames 200-2000,
with no slipsheet crossings and at most 0.73 um overlap. See
[PHYSX_INTEGRATION.md](PHYSX_INTEGRATION.md) for equations, limitations and commands.

## Build (Windows, Visual Studio 2022)

Run from the repository root. The core uses the maintained Eigen copy in
`physx/source/lowleveldynamics/src/newton/vendor/eigen`. MuJoCo comparison dependencies are
kept in `physx/compiler/newton/vendor`.
To reproduce them, use the preparation script; existing local sources allow an offline rebuild.
A fresh machine needs network access for the pinned source archives and Python wheel.

```powershell
python physx/tests/newton/PrepareDependencies.py
cmake -S physx/tests/newton -B physx/compiler/newton/build -G "Visual Studio 17 2022" -A x64
cmake --build physx/compiler/newton/build --config Release --parallel 2
```

Dependencies: Eigen 3.4.0, MuJoCo 3.3.7 (commit
`f1d45bd5422c74beddfb0d1deb590a02583d21de`), and NumPy 1.26.4.
The Newton kernel no longer requires MuJoCo; the pallet and reference harnesses use its public API.
The dependency preparation script no longer adds private-kernel exports. The preserved local DLL
still contains those historical visibility additions, with unchanged solver arithmetic. The official
Python wheel used for comparison is unmodified. Upstream licenses remain with the
vendor sources and generated kernel header. Building requires Python for that small local source
adapter; no download is needed when the pinned Eigen sources are present. Generated builds,
dependencies and results are ignored by Git.

## Validation and timing

```powershell
python physx/tests/newton/CompareMujoco.py
python physx/tests/newton/RepeatComparison.py
ctest --test-dir physx/compiler/newton/build -C Release --output-on-failure
physx/compiler/newton/build/Release/NewtonBenchmark.exe cable 20 1200 physx/compiler/newton/results/cable20.csv
physx/compiler/newton/build/Release/NewtonBenchmark.exe cable 120 1200 physx/compiler/newton/results/cable120.csv
```

The comparison runs seven cases: a resting box, a disturbed 3x3x3 stack, resting and disturbed
5x5x5 stacks, harder contacts with/without a warm start, and a 10,000:1 mass ratio. It compares
identical Jacobians, compliance, free motion and initial estimates against actual MuJoCo Newton.
Every measured run has a 10 ms timestep and 100-iteration cap. Warm seeds are computed at 99% of
the measured load and then held fixed across repeats. One initial run is discarded, followed by
seven timed runs. Fixture creation, collision detection, integration and file I/O are not timed.
The repeated 27-box comparison runs seven batches of 100 solves, pins both solvers to the same
logical CPU, and alternates execution order. Both use identical frozen warm starts.
Separate factor audits compare every updated Newton direction to an independently assembled
LDLT reference; this
reference is a validation check, not a selectable solver. The mixed-constraint CTest exercises
scalar rows, bilateral rows and coupled three-component friction together, including cold and
warm starts and changing island sizes to check workspace/pattern reuse. No timing threshold is part of that correctness test.

Two additional CTests instrument the Debug CRT heap, including C++ new and Eigen's malloc/realloc.
They verify zero allocations and frees after warm-up in 1,000 changing-topology mixed solves
and 400 live pallet steps with eight workers. The latter covers our adapter, task arrays and
solver, including work on all workers. The separately built release MuJoCo DLL is outside that
heap hook. Instrumentation is not linked into performance executables. See PERFORMANCE.md.

The cable is a nonlinear chain of 0.1 kg spheres spaced 5 cm apart, fixed at the root, with a 1 N
tip load and no gravity or collisions. Orientation springs use 100 N.m/degree stiffness and
20 N.m.s/degree damping. The run covers 12 seconds and rebuilds anchors/orientations each timestep.

The executable also accepts a prepared input, optional frozen seed and output solution:

```text
NewtonBenchmark input [seed|-] [solution|-] [repeats=8] [iterations=100] [check=0] [islands=1] [threads=1]
```

For 500 boxes, repeat one 125-box fixture as four independent islands, with eight available
workers. Parallelism is currently across islands; the sparse factorization within each island
is single threaded. `ms` includes the first island's complete per-call solve work and result
output writes. `wall_ms` includes dispatch, all island solves and result collection.
Optional residual diagnostics and file I/O are outside both timers. This is not a full PhysX
simulation-step timing. `Settings.profile` enables detailed phase timers; the normal path measures
only total solve time. Persistent worker storage is freed when its thread exits.

## Single connected pile

[pile/README.md](pile/README.md) documents the connected 125-, 500- and 1,000-box benchmark,
including native MuJoCo comparisons, identical settled snapshots and one/eight-worker controls.
Unlike the older four-island 500-box fixture, every box is part of one connected island.
The prototype is still faster than native MuJoCo, but a single large island remains expensive.

## Full pallet conveyor scene

`pallet/README.md` describes the five-conveyor, 325-body pallet/slipsheet test and its build commands.
It includes a stock PhysX PGS snippet, a MuJoCo XML scene and a full MuJoCo simulation runner
for native Newton or this prototype. `pallet/RESULTS.md` records the measured behavior and timings.
This MuJoCo full-simulation harness is separate from the native PhysX tests under `native/`;
its recorded timings must be distinguished from native SDK measurements.
