# Anvil solver and native PhysX integration

The selected Anvil solver uses incremental Cholesky updates and now has a native PhysX CPU
adapter, selected with `PxSceneDesc::solverType = PxSolverType::eANVIL`. The numerical core lives
in `physx/source/lowleveldynamics/src/anvil/core`; the native adapter is in its parent directory.
This test directory builds the same core for standalone validation and MuJoCo comparisons.
PGS and TGS retain their numerical kernels and compiler settings. The shared CPU task setup
selects Anvil at the island boundary. Articulations, GPU dynamics and the direct GPU API are
rejected for Anvil scenes. Current optimization experiments remain separate under
`physx/compiler/anvil/experiments`.

`AnvilSolver.cpp` contains the convex objective, weighted pyramidal projection, stopping
criteria and safeguarded line search. `IncrementalCholesky.h` orders small body graphs with a
retained minimum-degree implementation and larger graphs with METIS,
constructs the permuted sparse matrix in linear time, and chooses incremental updates or a
fresh factorization using structural work estimates. `BlockCholesky.h` supplies small rigid-body
block kernels for every factor, retaining scalar LLT only for failed block pivots. A one-body
problem factors its 6x6 Hessian directly and refactors instead of updating it. The serial block
factor reads the Hessian's 6x6 body-pair blocks directly; the scalar CSC Hessian is exported only
for symbolic analysis and the scalar or parallel paths. Positive updates
precede downdates. A numerical loss of rank triggers a complete current-Hessian rebuild.

Stiff contacts (small compliance R) make the objective's curvature jump by about 1/R at each
row's kink, so Anvil's exact line search can stop after only a few active-set changes. Such
solves needed 60-180 iterations on the pallet fall-off. Unilateral solves that are still
unconverged after `Settings::interiorPointSwitch` Anvil iterations, with repeated short steps,
continue with Mehrotra predictor-corrector interior-point steps. Their Anvil system has the same
`I + J' D J` shape, so they reuse the Hessian assembly and sparse factor; they need about 25
steps on those problems. Anvil finishes once the scaled gradient is below `interiorPointExit`.
Continuations of such a solve warm start the interior point from its previous multipliers.

A native island uses one ordinary `solveAnvil` call. Each new island and timestep starts a
fresh numerical factor; there is no cross-timestep numerical-factor reuse. Scratch capacity and
symbolic analysis are retained, with symbolic reuse requiring an identical full sparse pattern.
Settings remain a 100-iteration cap, normalized tolerance `1e-8` and line-search multiplier `0.01`.
A converged warm start exits before Hessian assembly. The earlier five-pallet
MuJoCo-equation benchmark measured **2.03 ms** for preparation, dispatch, solving and writeback,
versus **6.02 ms** for native MuJoCo. These are historical measurements of that adapter and
equation set, not native PhysX timing claims. Native PhysX performance is still under
investigation. See [PERFORMANCE.md](PERFORMANCE.md) for the measured settings, tails and checks.

Storage is retained across solves and timesteps. `AnvilStorage.h` separates active vector/matrix
sizes from capacity; shrinking and regrowing within capacity does not allocate. Hessian assembly,
permutations, symbolic-analysis work arrays and line search also reuse storage. Capacity grows
geometrically when needed. The caller retains `Problem`, `Result` and an explicit `Workspace`,
then calls `solveAnvil(problem, settings, result, workspace, previous)`. The call returns a
`SolveStatus`; `previous` may be `&result`. A workspace is owned by one solve at a time.
The native context retains a pool of these workspaces across islands and timesteps.

Scalar rows use 128-byte retained contact records; extra tangent rows are stored only for
three-row blocks. The full Contact is an input builder passed to problem.addContact(). Use
clearContacts() to rebuild a mixed problem while retaining both arrays' capacity. Producers
may accumulate nonzero column counts while emitting coefficients and call
prepareProblemFromColumnCounts(); reset the counts before each preparation, since that call
consumes them into CSC insertion cursors. The ordinary prepareProblem() counts its own entries.

The production solver has no Eigen dependency. Its fixed-size matrix operations, symmetric
eigensolver, retained CSC storage and sparse Cholesky implementation live in `core/`. Hot 6x6
factor updates use explicit packed-double kernels, while small changing islands use retained
ordering storage so repeated solves do not allocate.

Prepared equations use mass-scaled body coordinates and positive diagonal constraint compliance.
Bilateral three-row blocks represent anchors and springs. Friction uses a square pyramid, with
matching tangent compliances. The MuJoCo comparison exports its four nonnegative pyramid-edge
variables directly; each scalar edge occupies one packed row in the solver. The prepared file format still reserves
three components, and the loader removes the two unused components.
These edge equations have MuJoCo's compliance, which is not interchangeable with a three-component
contact block's compliance. See PERFORMANCE.md for the measured comparison.

## PhysX integration

[PHYSX_INTEGRATION.md](PHYSX_INTEGRATION.md) records the backup and integration design, including
the shared PGS task pipeline and a possible future adaptive handoff. Adaptive PGS-to-Anvil
switching is not implemented.

Native Anvil uses the existing island startup and completion tasks, with its own constraint
preparation and solve between them. Scene settings default to `anvilMaxIterations = 100`,
`anvilTolerance = 1e-8` and `anvilRegularization = 1e-4`, with contacts softened to 1e-2 at
first touch (see below). Contacts use MuJoCo's four-edge
pyramidal formulation and reference-acceleration equation. Hard joint rows use the same reference
policy; spring rows retain their specified implicit stiffness and damping. Invalid settings and
unsupported articulation/GPU insertion paths are checked in Release as well as Checked builds.

`anvilSurfaceRegularization` and `anvilStiffeningDepth` soften contacts at first touch, like
MuJoCo's position-dependent impedance: the log of the regularization follows MuJoCo's smooth
step (midpoint 0.5, power 2) from the surface value at zero penetration to `anvilRegularization`
at the stiffening depth, using each point's penetration at the start of the step. The defaults
are 1e-2 over 20 um (scaled by `PxTolerancesScale::length`); a depth of 0 restores constant
regularization. On the five-pallet conveyor this raised the slipsheet overlap from 0.72 to 9.4 um
and made the WebAssembly fall-off 28% faster; the connected 125-, 500- and 1,000-box piles took 3
instead of 5 Anvil iterations per step and ran 37-40% faster, with the bottom layer 5.5-7.3 um
into the ground and no added motion at rest. (Those speedups were measured with MuJoCo's
impedance step, below; the log step matched its times within 2% on both scenes.)

MuJoCo steps the impedance instead, which puts nearly all of a 100:1 regularization range in the
last fifth of the depth. That is unstable where a light body carries a heavy load, such as a
0.2 kg tote holding 40 kg (`cases/ToteConveyor`): the light body's contacts sink in one step by
their compliance times the load, the compliance is fixed at the start of the step, so the body
bounces or rocks with period two wherever the regularization falls faster than about the
seventh power of the depth. MuJoCo's own totes bounce 4 um this way; with the impedance step,
PhysX Anvil needed 1.8 iterations and 8 rank updates per tote every step, and 17% more step
time than with the log step, which needs one iteration. The log step's slope, in log
regularization against log depth, peaks at log(surface / regular), 4.6 for the defaults, so
ranges above about 1000:1 bounce again. Curves that are too steep for the load (surface 1e-1
over 2-10 um, or 1e-2 over 5 um) made contacts bob between steps and tripled iterations.

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

Run from the repository root. Some standalone validation programs use a maintained Eigen copy as
an independent numerical oracle; neither `PhysXAnvilCore` nor the native PhysX integration links
or includes it. MuJoCo comparison dependencies are kept in `physx/compiler/anvil/vendor`.
To reproduce them, use the preparation script; existing local sources allow an offline rebuild.
A fresh machine needs network access for the pinned source archives and Python wheel.

```powershell
python physx/tests/anvil/PrepareDependencies.py
cmake -S physx/tests/anvil -B physx/compiler/anvil/build -G "Visual Studio 17 2022" -A x64
cmake --build physx/compiler/anvil/build --config Release --parallel 2
```

The standalone benchmark enables `PX_ANVIL_USE_AVX2` by default. The core option defaults to
off in other PhysX builds so the SDK retains its existing CPU requirement. Enable it explicitly
on AVX2/FMA targets; the option applies only to the Anvil numerical core and does not change
PGS or TGS code generation.

Without AVX2, the kernels use 128-bit pairs of doubles: SSE2 on x86, and WebAssembly SIMD128 in
Emscripten builds, where the core adds `-msimd128`. `PX_ANVIL_USE_WASM_RELAXED_SIMD` also enables
relaxed SIMD, which fuses multiply-adds where the host supports it and so changes rounding between
hosts. Defining `ANVIL_NO_SIMD` selects the scalar reference kernels.

Emscripten builds of the SDK use the Linux platform files. PhysX's dispatcher threads need
`-pthread` on every object and a `-sPTHREAD_POOL_SIZE` of at least the worker count at link
time; Node.js runs such builds directly. Emscripten 6's Clang reports new warnings in the
existing SDK sources, so those builds need `-Wno-error`. The native adapter is compiled without
C++ exceptions there, since Emscripten's JavaScript exception support routes each potentially
throwing call through JavaScript. The SDK disables the solver's own clock reads
(`Settings::timing`), which also leave WebAssembly.

On the five-pallet conveyor with eight workers, a release WebAssembly build under Node.js runs
about 11-14% behind native SSE2, uniformly across the solver's phases. Relaxed SIMD
(`PX_ANVIL_USE_WASM_RELAXED_SIMD`) removes most of that gap (belt and fall-off steps about 11%
faster) but requires a host with relaxed SIMD and makes rounding host-dependent. LTO measured
within 1%.

Test-only dependencies: Eigen 3.4.0, MuJoCo 3.13.0 (commit
`123347c0eeab7e13c8da0828ab593bbd95bcf335`), and NumPy 1.26.4.
The Anvil kernel no longer requires MuJoCo. The comparison harnesses use an unmodified build of
the pinned upstream source, and the official Python wheel used for comparison is unmodified.
Upstream licenses remain with the vendor sources. No download is needed when the pinned test
dependencies are present. Generated builds, dependencies and results are ignored by Git.

## Validation and timing

```powershell
python physx/tests/anvil/CompareMujoco.py
python physx/tests/anvil/RepeatComparison.py
ctest --test-dir physx/compiler/anvil/build -C Release --output-on-failure
physx/compiler/anvil/build/Release/AnvilBenchmark.exe cable 20 1200 physx/compiler/anvil/results/cable20.csv
physx/compiler/anvil/build/Release/AnvilBenchmark.exe cable 120 1200 physx/compiler/anvil/results/cable120.csv
```

The comparison runs seven cases: a resting box, a disturbed 3x3x3 stack, resting and disturbed
5x5x5 stacks, harder contacts with/without a warm start, and a 10,000:1 mass ratio. It compares
identical Jacobians, compliance, free motion and initial estimates against actual MuJoCo Newton.
Every measured run has a 10 ms timestep and 100-iteration cap. Warm seeds are computed at 99% of
the measured load and then held fixed across repeats. One initial run is discarded, followed by
seven timed runs. Fixture creation, collision detection, integration and file I/O are not timed.
The repeated 27-box comparison runs seven batches of 100 solves, pins both solvers to the same
logical CPU, and alternates execution order. Both use identical frozen warm starts.
Separate factor audits compare every updated Anvil direction to an independently assembled
LDLT reference; this
reference is a validation check, not a selectable solver. The mixed-constraint CTest exercises
scalar rows, bilateral rows and coupled three-component friction together, including cold and
warm starts and changing island sizes to check workspace/pattern reuse. No timing threshold is part of that correctness test.

Two additional CTests instrument the Debug CRT heap, including C++ new and C allocation calls.
They verify zero solver allocations and frees after warm-up in 1,000 changing-topology mixed
solves and 400 live pallet steps with eight workers. MuJoCo 3.13's comparison-only `mju_dispatch`
creates one temporary task wrapper per island; the pallet audit permits exactly those known
external allocations and fails on any additional allocation. Instrumentation is not linked into
performance executables. See PERFORMANCE.md.

The cable is a nonlinear chain of 0.1 kg spheres spaced 5 cm apart, fixed at the root, with a 1 N
tip load and no gravity or collisions. Orientation springs use 100 N.m/degree stiffness and
20 N.m.s/degree damping. The run covers 12 seconds and rebuilds anchors/orientations each timestep.

The executable also accepts a prepared input, optional frozen seed and output solution:

```text
AnvilBenchmark input [seed|-] [solution|-] [repeats=8] [iterations=100] [check=0] [islands=1] [threads=1]
```

For 500 boxes, repeat one 125-box fixture as four independent islands, with eight available
workers. These factors remain below the internal parallel-work crossover, so parallelism in this
fixture is across islands. Larger connected factors can also parallelize their trailing block
updates with up to eight workers. `ms` includes the first island's complete per-call solve work and
result output writes. `wall_ms` includes dispatch, all island solves and result collection.
Optional residual diagnostics and file I/O are outside both timers. This is not a full PhysX
simulation-step timing. `Settings.profile` enables detailed phase timers; the normal path measures
only total solve time. Persistent worker storage is freed when its thread exits.

## Single connected pile

[pile/README.md](pile/README.md) documents the connected 125-, 500- and 1,000-box benchmark,
including native MuJoCo comparisons, identical settled snapshots and one/eight-worker controls.
Unlike the older four-island 500-box fixture, every box is part of one connected island. Factors
above the measured symbolic-work crossover use the parallel Cholesky path; smaller factors retain
the original serial path.

## Fixed-joint rods

`native/FixedJointTests.cpp` (`AnvilFixedJointTests`, ctest `AnvilNativeFixedJoints`) cantilevers
rods of 0.1 m cubes joined by fixed joints from a kinematic cube 0. Cube i weighs 0.1 * 2^i kg, so
every joint carries a heavier rod through a lighter cube: the 20-cube rod holds about 105 t
through a 0.2 kg cube. Rods of 10 and 20 cubes settle for 3 s, and a 100 kg cube falls 0.5 m onto
the tip of a settled 10-cube rod. `AnvilFixedJointTests pgs` reports PGS without checks, and a
second argument overrides the joint regularization. `native/MujocoFixedJointTests.cpp` runs the
same rods in MuJoCo 3.14 with weld equality constraints between free bodies and a static cube 0.

Hard joint rows once used the contacts' regularization, 1e-4 of their response. A row's response
is dominated by its lighter body, so a joint yields by that regularization times the load it
carries over that body's mass: the 10-cube rod sagged 215 mm and the 20-cube rod collapsed.
`PxSceneDesc::anvilJointRegularization` (default 1e-10) now sets joints apart; deflection scales
linearly with it (1e-6: 2.2 mm and 506 mm; 1e-8: 0.02 and 134 mm; 1e-12: 0 and 0.014 mm), and
conditioning depends only on it, not on the mass ratios. MuJoCo caps weld impedance at 0.9999,
the same 1e-4 relative softness, so it cannot hold these rods either.

Three seconds at 10 ms steps with one worker; mean step time excludes the first step:

| Scene | Anvil | PGS 16+2 | MuJoCo 3.14 (stiffest weld) |
| --- | --- | --- | --- |
| 10-cube rod, tip sag | 0.0001 mm | 537 mm, joints 136 mm apart | 545 mm, joints 18 mm apart |
| 20-cube rod, tip sag | 1.36 mm | collapses | unstable |
| 100 kg drop, peak / settled deflection | 0.0018 / 0.0007 mm | 1,052 / 484 mm, weight falls off | 680 / 586 mm, weight falls off |
| Step time, 10 / 20 cubes (native) | 28 / 37 us | 24 / 33 us | 16 / 35 us |
| Step time, 10 / 20 cubes (WebAssembly) | 51 / 57 us | 50 / 56 us | - |

MuJoCo's default weld impedance (solimp 0.9 0.95) diverges: it reports NaN accelerations and
resets. Anvil's step time here is mostly PhysX's fixed per-step cost; the joints add little.

## Gripper

`native/GripperTests.cpp` (`AnvilGripperTests`, ctest `AnvilNativeGripper`) closes two 0.5 kg
fingers on a row of four 1 kg, 0.1 m boxes (friction 0.5) and lifts them 0.3 m over 1.5 s. Each
finger is on a D6 joint with only X free, driven past the boxes by a force-limited linear drive,
so the drive's force limit is the grip. The outer faces carry two boxes each, so the row needs
2 m (g + a) / mu = 42.4 N at the lift's peak acceleration of 0.8 m/s^2. The default run checks
Anvil at 150 N, and at 10% below and above that requirement: below, the row must drop; above,
it must lift. `AnvilGripperTests anvil|pgs grip-force hold-seconds` reports a single run. Shapes
use a 0.5 mm contact offset, so pairs make contacts within 1 mm (the 2 cm default gave the same
Anvil results).

Anvil drops the row at 41 N and lifts it from 42 N. PGS (16+2) drops it up to 42 N and lifts it
at 45 N with 19 mm of slip and 2.6 degrees of tilt; at 150 N it still tilts the boxes 0.8 degrees
and squeezes them 0.5 mm into each other, against 0.01 mm for Anvil. A step takes about 35 us
natively for either solver (70 and 55 us in WebAssembly).

`native/MujocoGripperTests.cpp` builds the same gripper in MuJoCo 3.14: slide-joint fingers with
force-limited position actuators on a heavy arm whose position and velocity follow the lift.
MuJoCo drops the row in every variant tried (contact models matched to Anvil or MuJoCo's
defaults, pyramidal or elliptic friction, grips up to 1,000 N), and even a single box: the
fingers carry it about 0.11 m and then pass through it.

Anvil's friction is regularized, so a sustained friction load creeps: at 150 N the lifted boxes
slide down through the fingers at 0.36 mm/s (22 mm over 60 s), and faster with a lighter grip,
whose shallower contacts are softer (3.3 mm/s at 42 N). PGS's friction anchors hold position.

## Case conveyor scene

[cases/README.md](cases/README.md) documents 2,000 separate cases carried by ten long conveyors,
each case an independent island on its belt, run with PhysX PGS, PhysX Anvil and MuJoCo.

## Full pallet conveyor scene

`pallet/README.md` describes the five-conveyor, 325-body pallet/slipsheet test and its build commands.
It includes a stock PhysX PGS snippet, a MuJoCo XML scene and a full MuJoCo simulation runner
for native Anvil or this prototype. `pallet/RESULTS.md` records the measured behavior and timings.
This MuJoCo full-simulation harness is separate from the native PhysX tests under `native/`;
its recorded timings must be distinguished from native SDK measurements.
