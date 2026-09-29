# Native PhysX Anvil integration

The CPU backend is selected with `PxSolverType::eANVIL`. PGS remains the default and
retains its existing preparation, solver kernels and compiler settings. TGS and a future
adaptive PGS-to-Anvil handoff are unchanged.

```cpp
PxSceneDesc desc(physics->getTolerancesScale());
desc.solverType = PxSolverType::eANVIL;
desc.anvilMaxIterations = 100;
desc.anvilTolerance = 1e-8f;
desc.anvilRegularization = 1e-4f;
```

The actor's PGS position and velocity iteration counts do not control Anvil. Anvil uses
one coupled solve per connected island and stops at the tolerance or iteration limit.

## Equations

Anvil contacts use lagged Coulomb friction, the Lagged approximation of Castro, Han and
Permenter, [Irrotational Contact Fields](https://arxiv.org/abs/2312.03908). Each frictional
contact point emits a nonnegative normal row and two tangent rows whose impulses are bounded by
`+/- mu N0`, where `N0` is the point's share of its pair's normal impulse in the previous step
(split equally over the pair's current points; a pair without one takes the impulse that stops
its approach). Friction therefore does not depend on the step's normal rows, so the problem
stays convex without MuJoCo's pyramidal coupling of normal and tangential velocity, and sliding
contacts do not separate. The lag is one step: a contact's friction limit follows a change of
its load in the next step. The tangent basis follows the contact's initial slip direction.

`anvilRegularization` maps to impedance `d = 1 / (1 + regularization)`. With
`timeConstant = max(0.02, 2 timestep)`, the reference coefficients are
`B = 2 / (d timeConstant)` and `K = 1 / (d^2 timeConstant^2)`. Row compliance is
`R = regularization diagApprox`, where `diagApprox` is the translational inverse-mass response;
the normal row of a frictional contact uses an eighth of it, the normal stiffness the earlier
four-edge pyramid had at friction 0.5. Softer normals cost iterations on large piles (a quarter:
12% slower on 1,000 boxes) and when many contacts change between sticking and sliding, and
penetrate further (pallet slipsheets 9.3 um at an eighth, 10.6 um at a quarter, 13.8 um without
scaling); much stiffer normals cost iterations on resting stacks.

Body velocity is captured before PhysX applies gravity and damping. The prepared free term
therefore represents MuJoCo's acceleration-reference equation in velocity-increment units:

```text
freeSpeed - initialSpeed + timestep *
    (B * (initialSpeed - targetSpeed) + K * d * positionError)
```

The shape's contact offset slop is not applied to Anvil rows.

Hard joint rows use the same reference policy. Native spring rows retain their exact implicit
stiffness and damping law. Positive restitution, compliant contacts, contact modification,
force caps and threshold reporting remain Anvil-only PhysX extensions.

## Implementation

The maintained implementation lives under `source/lowleveldynamics/src/anvil/`:

- `DyAnvilSolver.cpp` owns island preparation, reusable workspaces, warm starts and result
  conversion.
- `DyAnvilConstraintPrep.cpp` converts generic `PxConstraintSolverPrep` rows, bounds,
  springs and writeback.
- `DyAnvilContactPrep.cpp` converts native contact streams into normal and bounded friction
  rows, keeps each pair's friction state and writes normal impulses and threshold events.
- `core/` contains the optimized incremental-Cholesky Anvil solver used by both the native
  adapter and standalone comparisons.

The shared task pipeline selects
`existing start task -> Anvil island task -> existing end task` before PGS row packing.
Independent islands run through PhysX's CPU dispatcher. A connected Anvil island remains one
PhysX task. Islands smaller than 32 bodies share one task chain, up to 32 bodies in total, so
scattered resting objects do not each dispatch three tasks; every island in a batch is still
prepared, factored and solved separately, with results identical to unbatched tasks. A batch
task holds one solver workspace for all of its islands and groups the batch's constraint
descriptors by island once, so small islands share no lock or counter between workers. Large Cholesky factors parallelize independent trailing block updates with up to eight
workers from PhysX's CPU dispatcher. Pivots are grouped into panels of up to 16 (or 8,192 block
updates): a panel factors its own columns in order, then applies its updates to all later columns
in one parallel region, one task per target column, so a factor synchronizes once per panel rather
than once per pivot; each block still receives its updates in pivot order, and results are
unchanged. This took the 1,000-box pile's factorizations from 16.0 to about 12 ms per step with
eight workers. The Anvil island task participates in the work and waits at
the same cooperative barriers used by PhysX's parallel PGS solver. Helpers are launched lazily on
the first sufficiently large factorization and finish when that factorization ends. The established
left-looking factorization remains unchanged for smaller factors; selection uses symbolic update
work rather than body count. Both factorizations read the Hessian's body-pair blocks directly,
without exporting and permuting a scalar matrix. At most one island recruits helpers while other
island tasks continue through the serial path. Anvil adds no OpenMP dependency, and the PGS and TGS
paths are unchanged.

Islands of at least 500 bodies also run their other large phases on the same helpers: contact rows
are prepared in chunks of 256 pair descriptors and merged in descriptor order (islands with joints
keep serial preparation), and the Hessian blocks, `J v` and the gradient `J' lambda` are computed
per row, body or block. Each body and block gathers its rows in the order serial evaluation adds
them, so results are the same as serial evaluation. On the 1,000-box pile this took native steps
from 17.2 to 13.4 ms and WebAssembly from 26.0 to 22.4 ms with eight workers.

Large factors also solve their triangular systems on the helpers. The elimination tree's chains,
its maximal paths of bodies with one child each, are the separators of the nested dissection
order; chains of one depth are independent. The chains of the first four depths are solved in
stages and each subtree below them is one task. In the forward solve a chain's bodies first
subtract the blocks of deeper chains, every body independently, and then the chain solves in
order; the backward solve takes the chains from the top, one task per chain. Each stage's work
is split into at most 16 tasks of similar block counts, and a stage with a single task runs
without helpers. A row's blocks of deeper chains precede those of its own chain, so every solve
subtracts them in the serial order and gives the serial result. The schedule it replaces ran
one parallel region per set of independent bodies: 223 per solve on the 1,000-box pile, half of
them a single body, with half of the forward solve in the rows of the 100-body top chain, and
took as long with eight workers as without helpers. With 14 regions per solve, WebAssembly steps
of the 1,000-box pile take 5% less time and those of the 500-box pile 6% less.

The block factor stores its 6x6 blocks in single precision. Builds with 256-bit vectors pad each
column to eight floats, so one block column is one AVX register. Builds with 128-bit vectors,
WebAssembly among them, store 36 floats without padding, rows 0-3 of the six columns and then
their rows 4-5, so two columns' last rows share a vector and a block update takes nine vectors
instead of twelve; every entry receives the same operations in both layouts, and results are
unchanged. WebAssembly piles step 3-7% faster for it. Either way the block updates
that dominate factorization process twice the values per instruction. Diagonal blocks are
factored in double precision and then rounded, and triangular solves accumulate in double. The
factor only sets the Newton direction: the factor of a slightly perturbed positive definite
Hessian still gives a descent direction, and the line search and convergence tests use the
double-precision problem, so the solution is unchanged up to the solver tolerance; rank updates
continue on the exported double factor. Piles step 13-18% faster, native and WebAssembly, and the
platform scene 3-5%; penetration, overlap and slip metrics are unchanged.

Results do not depend on the worker count or on which island holds the helpers, so a
WebAssembly build gives the same results on every machine. The serial factorization updates each
block in ascending pivot order, the order of the parallel one, so both round alike, and the choice
between refactoring and rank updates assumes eight workers for every factor large enough to use
helpers, whatever the machine has. Parallel evaluation, assembly and row preparation keep the
serial order. The pile benchmark's `state_hash` column hashes every pose and velocity bit; all
benchmark scenes give identical outputs with one, three and eight workers.

`PxDefaultCpuDispatcher` workers in wait-for-work mode sleep on their own wake signal. A
submitted job wakes one sleeping worker, the one that slept most recently, instead of every
sleeper; busy workers make no event calls. This applies to PGS and TGS scenes too. A worker that
announced sleep but then found a job itself passes on any wake a submitter meant for it, so no
job waits in a queue while a worker sleeps. Without this, a parallel region's helpers, which wait
for each other, occasionally hung for good (about once in 300,000 regions in a dispatcher stress
test of that pattern, and once in the 1,000-box pile benchmark).

A worker that wakes also leaves the sleepers itself, whether or not a submitter claimed it. A
submitter claims a worker and signals it a moment later; in between, the worker may find a job
itself, run it, announce sleep again and begin to wait, so the late signal wakes it while it is
still announced. It then ran jobs while counted as sleeping, and a later job's wake went to that
busy worker instead of a sleeping one: a WebAssembly pile benchmark hung with the island task and
six helpers spinning and one worker asleep, about once in a hundred runs. An exhaustive model of
the protocol reaches that deadlock with three workers and one short job, and none with the rule.

Friction and normal rows are all scalar rows, so the core keeps its compact, vectorized
row kernels for them: rows carry `[lower, upper]` impulse bounds (`[0, cap]` for normals,
`[-mu N0, mu N0]` for friction), and the fused step evaluation, line search and curvature
updates clamp against them. The interior-point fallback remains for purely unilateral problems.
Each island takes one solve per step. The four-edge pyramid used before separated sliding
contacts (by `mu timestep` times the slip speed) unless up to four continuation solves biased
its edges by the solved slip speed; these corrections dominated steps in which many contacts
slide, such as pallets tipping off conveyors. `AnvilSlidingTests` (boxes and a loaded container
sliding down a ramp, and a loaded container pushed along a floor) measure no lift-off and sliding
accelerations within 0.5% of Coulomb's. On the five-pallet scene, the WebAssembly fall-off steps
take 1.2 ms on average instead of 2.6 ms, the 95th percentile 3.0 ms instead of 11.6 ms and the
slowest 5.1 ms instead of 15.3 ms.

Workspaces retain row, matrix, factor and result capacity between jobs. Warm starts store
physical body corrections and transform angular corrections into the current inertia basis.
Each island and timestep starts with an ordinary `solveAnvil`; there is no cross-timestep
numeric-factor reuse. Scratch buffers and symbolic analysis storage are reused.

Anvil captures pre-force velocity through a compile-time-specialized preintegration loop.
The PGS instantiation contains no per-body capture branch or write. Anvil code is built in a
private target with self-contained matrix and sparse-factor code. Its compiler options do not
propagate into PGS or TGS.

## Current limitations

Friction uses static friction while a contact sticks and changes to dynamic friction for the
next step once every loaded point of the pair reaches its friction limit: for scalar friction
rows, one row at its bound; three-row blocks compare each tangent impulse with friction times the
normal impulse. A slip-speed threshold is not used: soft friction lets a holding contact creep in
proportion to its load, so such a threshold failed at large scales. The two bounded tangent rows
form a square friction limit aligned with the initial slip direction, not a circle.

An explicitly capped contact point currently retains its exact normal cap and omits friction.
Native negative-restitution compliant contacts use their existing implicit scalar normal row.

This backend supports CPU rigid bodies. GPU dynamics, the Direct GPU API and articulations are
rejected. Constraint-local mass/inertia scaling and asymmetric contact dominance are also
unsupported and report an error for the affected island.

## Build and validation

```powershell
cmake -S physx/tests/anvil/pallet -B physx/compiler/anvil/native-build -G "Visual Studio 17 2022" -A x64 -DPX_ANVIL_USE_AVX2=ON
cmake --build physx/compiler/anvil/native-build --config profile --parallel 4
cmake --build physx/compiler/anvil/native-build/anvil --config profile --target AnvilBenchmark AnvilAllocationAudit PalletAllocationAudit AnvilBoundedValidation AnvilPatchValidation AnvilPatchProjectionAudit AnvilContinuationValidation --parallel 4
ctest --test-dir physx/compiler/anvil/native-build -C profile --output-on-failure
```

All twelve profile CTests pass. They cover native scenes, generic joints, contacts and contact
modification, compliant contacts, friction, rolling bodies, bounds, continuation and retained
storage. The PGS native scene suite also passes.

Profile CSVs report `step_ms`, `island_wall_ms`, `solve_wall_ms`, `prepare_cpu_ms`, `solve_cpu_ms`
and `rows`.
`island_wall_ms` spans the earliest Anvil island start to the latest island completion.
`solve_wall_ms` spans the earliest island solve start to the latest solve completion. The CPU
fields sum task durations and can overlap across independent islands.

For the five-pallet conveyor at 10 ms, eight workers, regularization `1e-4` and a 100-iteration
limit, five 220-frame profile runs measured frames 100-220:

| Formulation | Rows | Island wall time | Total Anvil iterations | Maximum slipsheet overlap |
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

On the connected pile fixture, the crossover is 500,000 symbolic updates. Paired Release runs of
the native PhysX-task implementation measured frames 11-30:

| Bodies | One worker | Eight workers | Speedup |
| ---: | ---: | ---: | ---: |
| 500 | 74.188 ms | 43.998 ms | 1.69x |
| 600 | 108.118 ms | 59.745 ms | 1.81x |
| 1,000 | 640.623 ms | 184.740 ms | 3.47x |

Contact counts, minimum height and settling speed match between the paired runs. The speedup is
below the worker count because the pivot dependency chain, matrix assembly, triangular solves,
Anvil evaluations and line search remain serial; only each pivot's independent trailing block
updates run across the task team.

The pre-change source checkpoint is
`backups/Anvil-native-before-mujoco-contacts-2026-09-14_22-22-21.zip`, SHA-256
`E87BB5D118E2DC31BCFACE2C0EDF335796DCDCDF1D522541AD2002A90F3E9F9C`.
