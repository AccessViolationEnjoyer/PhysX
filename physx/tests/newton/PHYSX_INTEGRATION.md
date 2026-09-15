# Native PhysX Newton integration

The CPU backend is selected with `PxSolverType::eNEWTON`. PGS remains the default and
retains its existing preparation, solver kernels and compiler settings. TGS and a future
adaptive PGS-to-Newton handoff are unchanged.

```cpp
PxSceneDesc desc(physics->getTolerancesScale());
desc.solverType = PxSolverType::eNEWTON;
desc.newtonMaxIterations = 100;
desc.newtonTolerance = 1e-8f;
desc.newtonRegularization = 1e-4f;
```

The actor's PGS position and velocity iteration counts do not control Newton. Newton uses
one coupled solve per connected island and stops at the tolerance or iteration limit.

## Equations

Newton contacts use MuJoCo's convex pyramidal formulation. Each frictional contact point
emits four nonnegative scalar edges:

```text
n + mu t0, n - mu t0, n + mu t1, n - mu t1
```

`newtonRegularization` maps to impedance `d = 1 / (1 + regularization)`. With
`timeConstant = max(0.02, 2 timestep)`, the reference coefficients are
`B = 2 / (d timeConstant)` and `K = 1 / (d^2 timeConstant^2)`. Edge compliance uses
MuJoCo's pyramidal conversion, `R = 2 mu^2 regularization diagApprox`, where
`diagApprox` is the translational inverse-mass response scaled by `1 + mu^2`.

Body velocity is captured before PhysX applies gravity and damping. The prepared free term
therefore represents MuJoCo's acceleration-reference equation in velocity-increment units:

```text
freeSpeed - initialSpeed + timestep *
    (B * (initialSpeed - targetSpeed) + K * d * positionError)
```

Hard joint rows use the same reference policy. Native spring rows retain their exact implicit
stiffness and damping law. Positive restitution, compliant contacts, contact modification,
force caps and threshold reporting remain Newton-only PhysX extensions.

## Implementation

The maintained implementation lives under `source/lowleveldynamics/src/newton/`:

- `DyNewtonSolver.cpp` owns island preparation, reusable workspaces, warm starts and result
  conversion.
- `DyNewtonConstraintPrep.cpp` converts generic `PxConstraintSolverPrep` rows, bounds,
  springs and writeback.
- `DyNewtonContactPrep.cpp` converts native contact streams into pyramid edges and writes
  normal impulses and threshold events.
- `core/` contains the optimized incremental-Cholesky Newton solver used by both the native
  adapter and standalone comparisons.

The shared task pipeline selects
`existing start task -> Newton island task -> existing end task` before PGS row packing.
Independent islands run through PhysX's CPU dispatcher. A connected Newton island remains one
PhysX task. Large Cholesky factors parallelize independent trailing block updates with up to eight
OpenMP workers. The established left-looking factorization remains unchanged for smaller factors;
the selection uses symbolic update work rather than body count. At most one island recruits an
OpenMP team while other island tasks continue through the serial path.

The numeric factorization cannot enqueue child PhysX tasks and wait for them because a
`PxBaseTask::run()` implementation must not block. An asynchronous PhysX-task implementation would
instead require splitting the solver into continuations. Keeping OpenMP private to
`PhysXNewtonCore` avoids that pipeline change and does not add compiler flags or branches to PGS or
TGS.

Workspaces retain row, matrix, factor and result capacity between jobs. Warm starts store
physical body corrections and transform angular corrections into the current inertia basis.
Each island and timestep starts with an ordinary `solveNewton`; there is no cross-timestep
numeric-factor reuse. Scratch buffers and symbolic analysis storage are reused.

Newton captures pre-force velocity through a compile-time-specialized preintegration loop.
The PGS instantiation contains no per-body capture branch or write. Newton code is built in a
private target, so Eigen and exception options do not propagate into PGS or TGS.

## Current limitations

The pyramidal cone uses static friction while a contact sticks and changes to dynamic friction
when the solved contact retains tangential velocity. Equal coefficients use the original path
without friction-state storage or velocity writeback. The pyramid can introduce upward velocity
while sliding; this behavior is covered by bounded regression tests and will be addressed
separately if it is noticeable in practice.

An explicitly capped contact point currently retains its exact normal cap and omits friction,
because four independent edge bounds cannot express a cap on their summed normal impulse.
Native negative-restitution compliant contacts use their existing implicit scalar normal row.

This backend supports CPU rigid bodies. GPU dynamics, the Direct GPU API and articulations are
rejected. Constraint-local mass/inertia scaling and asymmetric contact dominance are also
unsupported and report an error for the affected island.

## Build and validation

```powershell
cmake -S physx/tests/newton/pallet -B physx/compiler/newton/native-build -G "Visual Studio 17 2022" -A x64
cmake --build physx/compiler/newton/native-build --config profile --parallel 4
cmake --build physx/compiler/newton/native-build/newton --config profile --target NewtonBenchmark NewtonAllocationAudit PalletAllocationAudit NewtonBoundedValidation NewtonPatchValidation NewtonPatchProjectionAudit NewtonContinuationValidation --parallel 4
ctest --test-dir physx/compiler/newton/native-build -C profile --output-on-failure
```

All twelve profile CTests pass. They cover native scenes, generic joints, contacts and contact
modification, compliant contacts, friction, rolling bodies, bounds, continuation and retained
storage. The PGS native scene suite also passes.

Profile CSVs report `step_ms`, `island_wall_ms`, `solve_wall_ms`, `prepare_cpu_ms`, `solve_cpu_ms`
and `rows`.
`island_wall_ms` spans the earliest Newton island start to the latest island completion.
`solve_wall_ms` spans the earliest island solve start to the latest solve completion. The CPU
fields sum task durations and can overlap across independent islands.

For the five-pallet conveyor at 10 ms, eight workers, regularization `1e-4` and a 100-iteration
limit, five 220-frame profile runs measured frames 100-220:

| Formulation | Rows | Island wall time | Total Newton iterations | Maximum slipsheet overlap |
| --- | ---: | ---: | ---: | ---: |
| Previous PhysX contact equations | about 20,400 | 34.964 ms | 267.744 | 4.67 um |
| MuJoCo pyramidal equations | 42,720 | 2.352 ms | 16.545 | 0.71 um |

The MuJoCo formulation is about 14.9 times faster in that matched interval despite emitting more
rows. The parallel solve span is 1.820 ms and mean full profile step time is 2.563 ms. A
2,000-frame run measured over frames 200-2000 averaged 3.042 ms of island wall time, 2.511 ms of
parallel solve time and 22.549 total iterations; no box crossed a slipsheet and maximum overlap
was 0.72 um. The conveyor maintained 0.2 m/s.

Three alternating 1,000-frame PGS runs averaged 0.751 ms before this change and 0.730 ms after
it. Their physical metrics were identical, so the difference is ordinary run-to-run variation.

On the connected pile fixture, 434,294 symbolic updates at 450 bodies took 59.776 ms serially and
60.085 ms with the parallel factor. At 500 bodies, 580,797 updates fell from 74.466 ms to 66.227 ms.
The crossover is therefore 500,000 symbolic updates. At 600 bodies the parallel path reduced the
Release full step from 105.590 ms to 79.777 ms, and at 700 bodies the profile solve span fell from
238.029 ms to 104.275 ms. At 1,000 bodies the Release full step fell from 660.640 ms to 277.669 ms;
both runs produced 6,769 contact pairs and matching settling metrics.

The pre-change source checkpoint is
`backups/Newton-native-before-mujoco-contacts-2026-09-14_22-22-21.zip`, SHA-256
`E87BB5D118E2DC31BCFACE2C0EDF335796DCDCDF1D522541AD2002A90F3E9F9C`.
