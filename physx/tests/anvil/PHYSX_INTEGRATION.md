# Native PhysX Anvil integration

The CPU backend is selected with `PxSolverType::eANVIL`. PGS remains the default and
retains its existing preparation, solver kernels and compiler settings. TGS and a future
adaptive PGS-to-Anvil handoff are unchanged.

```cpp
PxSceneDesc desc(physics->getTolerancesScale());
desc.solverType = PxSolverType::eANVIL;
desc.anvilMaxIterations = 1000;
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

Contact points whose gap cannot close within the step are left out of the problem: a point is
kept within 1 mm of its rest distance, or when its approach over the step (which already holds
this step's gravity), with a margin of one half for pushes from other contacts, covers the gap
(`ANVIL_SPECULATIVE_KEEP_GAP`, `ANVIL_SPECULATIVE_APPROACH_MARGIN`). This matters with
`PxRigidBodyFlag::eENABLE_SPECULATIVE_CCD`, which inflates a body's contact reach by its motion
per step: contacts then exist a step before surfaces meet, and a body arriving at 1 m/s is
stopped at the surface instead of embedding a centimetre and bouncing. On a 125-box pile with a
layer dropped from 5 cm every 0.2 s, the flag takes the landing step from 8-20 ms to 3-7 ms and
the twenty steps of re-settling after it from 3-7 ms to about 2 ms, against 1.3 ms at rest;
the same holds at 1,000 boxes. The price is PhysX's: every moving body's pairs within its
inflation are generated each step, and a scene of 2,000 totes riding a belt at 0.5 m/s, each
box 8 mm from its wall, spends 55% more per step (PGS pays the same), most of it in the narrow
phase before the filter discards the points. The test scenes set the flag on every body.

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

A large island refactors on nearly every step, and in a large irregular island most of that
work is lost: a staggered stack of 1,000 boxes of random size and density takes 23 steps and 17
factorizations per timestep after settling (a heap of 1,800 randomly oriented boxes 81 and 51,
against the regular pile's 1.2 and 1.3), and every other step or so the line search stops after
a few percent of the direction because some rows change between sticking and sliding or contact
and separation. Once a solve has taken its first step, a factor with at least 60,000 block
updates (the size from which the factor uses helpers; about 180 stacked boxes) is refactored
only when it must be. Where the cost model would refactor, the solve instead runs conjugate
gradients on the new Hessian, preconditioned by the factor it has, and takes the iterate as its
direction once the residual has fallen to 60% of the gradient, or after three iterations; a step
that took all three is followed by a refactorization, since the factor has fallen too far behind.
An iteration is a solve and two passes over the rows, and a factorization of such an island costs
eleven to twenty of them. Every iterate is a descent direction, so the exact line search applies
unchanged, with the current curvature rather than the factor's. This is an inexact step in the
sense of Dembo, Eisenstat and Steihaug, Inexact Newton methods, SIAM Journal on Numerical
Analysis 19 (1982); reusing a factor as the preconditioner of later systems, and choosing
between it and a new factor by their costs, follows Wang and O'Leary, Adaptive use of iterative
methods in predictor-corrector interior point methods for linear programming, Numerical
Algorithms 25 (2000). Looser directions cost more steps than they save: the retained factor's own
direction, one iteration, takes 80% more steps and 60-100% more time. Rows that change status
are always listed against the retained factor's curvature, so a later update or refactorization
covers everything that changed since it was made.

The same iterations decide convergence for an island near rest, whose second factorization
would only confirm that the next direction is within the displacement tolerance. The iterates
grow towards the direction, and one whose residual has not fallen can be far too short, so
convergence is declared only if the retained factor's own direction is within ten tolerances,
and once the residual has fallen to 7%, two successive iterates are below nine tenths of the
tolerances and the last grew by less than 3% of them, within ten iterations. On piles of 500 to
1,800 boxes of ten shapes this accepted 2,770 of 2,932 directions within tolerance and four
beyond it, by at most 6%. Conjugate-gradient error estimates from the energy norm (Meurant and
Tichý) bound the largest body velocity some 250 times too loosely to serve here.

The threshold was measured on regular piles of 27 to 1,000 boxes, staggered stacks of 125 to
1,000 boxes of random size and density, and the pallet scenes. Below about 30,000 block updates
the method costs time: a factorization of a 65-body pallet costs about as much as two or three
iterations, so steps on the retained factor make the platform scene 17% slower and the pallet
scene 5%, and irregular stacks of 200 to 300 boxes are neutral to 6% slower. From 60,000 updates
every scene measured gains: WebAssembly steps with eight workers take 20-30% less time on the
500- and 1,000-box piles and 15-20% less on piles of 180 to 343 boxes, and where an irregular
stack is large enough (1,000 boxes at 0.5 mm contact offsets; 343 at the 2 cm default) its
mean falls 11-28% and its 95th and 99th percentile steps by 15-36%, because the steps that
refactored most now refactor least. Results differ from those of the exact steps, as any change
of step does, but not systematically: settled heights, speeds and penetrations agree within their
noise. Results are bit-identical across worker counts, as before.

An exact line search often stops a step where a row reaches its kink, and rounding then puts the
row on either side of it. A row the step was pushing into contact, or into its friction bound,
that lands within a millionth of that step of its kink takes the curved branch of the Hessian
(`KINK_STEP_FRACTION`); the next direction otherwise pushes it straight through the kink and the
line search returns almost nothing, at the price of a factor update and a full line search. The
objective, impulses and gradient are the same on both branches, so this is a choice of
generalized Hessian at the kink, in the sense of semismooth Newton methods. A row the step was
carrying out of contact keeps the flat branch: giving every resting contact the curved branch
was tried and made a disturbed pile 40% slower, since unloaded touching contacts separate as
often as they close. On a 1,000-box pile with a layer of 100 dropped on it every 0.2 s the rule
takes 31% off the mean step (81 to 56 ms), and the worst 30 steps from 383 to 185 ms; the regular
pile, the pallet scenes and the irregular stacks are unchanged.

Steps in which many contacts are made or broken at once remain several times dearer than steps
at rest: a layer landing on the 125-box pile costs 3-7 ms against 1.3 ms settled, and on the
1,000-box pile up to 500 ms against 13 ms. The trace of such a solve shows thousands of rows
changing state in each of its first iterations and a handful in each of the next twenty, every
one with a factor update and a line search. An interior-point phase for bounded rows (a
two-sided extension of the unilateral one below) was tried and made these steps 60-75% slower,
as it needs 15-20 steps from a cold start and each is a fresh factorization with no rank updates;
so was taking the full step whenever it still lowered the objective, as a primal-dual active set
method does, at +15%. What does remove most of the disturbance is speculative contacts (above):
with them a landing is resolved before the surfaces meet and the pile below is hardly disturbed.

The default iteration limit is 1,000. A solve cut short by the limit hands its unconverged
velocities to the integrator, and an island of many colliding bodies can need several hundred
iterations: at 100, a staggered stack of 1,000 boxes of random size and density collapsing at
0.5 mm contact offsets reached 80-500 m/s, where PGS peaks at 3 m/s; at 1,000 it peaks at 3 m/s
too, for the same mean step time, since steps that converge never approach either limit.

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

The body ordering of islands of 192 bodies or more comes from METIS, whose nested dissection
draws random numbers, reseeded at every call. The vendored GKlib is built with `USE_GKRAND`
and its Mersenne Twister state is thread-local: with the C library's `rand` instead, MSVC's
per-thread state was repeatable but musl's single unlocked state was not, so two large islands
ordered at the same time in WebAssembly (a falling heap; never the pile or the standard scenes)
interleaved on it, their orderings varied from run to run, and rounding followed. Six runs of
1,800 randomly oriented falling boxes now give one hash at one, three and eight workers.

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
