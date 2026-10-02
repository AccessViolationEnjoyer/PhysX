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

A point of a sliding pair aligns its tangent rows with its slip `s`, its velocity relative to
the surface's target velocity at the start of the step. Coulomb friction keeps magnitude
`mu N0` and turns with the slip, so a velocity `v` across the slip turns it by `v / |s|`. The row
across the slip therefore has compliance `|s| / (mu N0)`, or the row compliance where that is
smaller, as inside the friction limit; both rows keep the `+/- mu N0` bound. As the slip
vanishes the rows become sticking's two rows, with no threshold between the cases. Two stiff
rows let the row across the slip hold while the other slides: a case straddling belts at 0.6
and 0.4 m/s then did not turn at all. A single row along the slip turned it, but with nothing
across the slip the lagged direction overshot every step when the slip was small against the
step's friction impulse. At 0.505/0.495 m/s the yaw rate then flipped sign each step. With the
compliant cross row the case turns at the Coulomb rate for its contact points at both speed
differences (ratio 1.0026 and 0.9997, `BehaviourTests split_conveyors`). On wasm the pallet
fall phase was 12% faster on the mean (4 repeats); belt, case, tote and pile-drop scenes were
unchanged in behaviour and within noise in time.

A sticking pair accumulates its slip, the tangential displacement at the contact centre, and its
friction rows correct it with twice the penetration stiffness, so a held load stays put. The
slip used to relax over 1 s, which let every held load creep at the slip that holds it divided
by that time: the PEEL card house sank 0.03 mm/s and a 150 N grip slipped 0.007 mm/s. The slip
no longer relaxes (card house 0.004 mm in 9 s, grip 0.0008 mm/s, near its limit 0.0005 instead of
0.017). Without relaxation, slip outlived its surfaces in collapsing hull piles: a hull that
tipped onto another face kept the slip of the face it left, and ten PEEL convex pile sizes took
9% longer on wasm. The slip now lapses when the contact normal, fixed in body 0's frame while the
pair sticks, turns far enough that body 0's material at the contact has moved by more than the
1 mm slip limit; the ten piles then take 4% less than with relaxation. Box scenes are unchanged
or faster (wasm platform -10%, pile 125 -10%, pallet and pile 1000 within noise); totes measured
+4% and -4% in separate runs. Four times the slip stiffness made the pallet five times slower.

PCM reuses a persistent box or hull manifold until the shapes have moved 0.375 times the
margin (15% of the smallest box half-extent), and PhysX placed reused points at their witness on
the second shape. On a static belt those trailed a sliding case by up to that distance. In the
split-belt test that drifted the case 2 mm in 4 s, dipped its speed 7.5% where its corners cross
the seam, and after 10 s, 5 mm off the seam, locked it onto the slow belt. Drift toward the slow
belt shifts load onto it, and once the case has turned to 45 degrees, Coulomb friction can hold
it there: a static check at the locked state needs 97-99.8% of each point's friction limit.

Box-box, box-convex and convex-convex clipping, and plane-box and plane-convex, therefore record
which shape owns each point (`PCMContactAnchor`, two bits per point in `mAnchors`, which fills
padding, so manifolds keep their size). A vertex inside the other shape's face belongs to its
shape; an edge crossing belongs to neither. `refreshContactPoints` moves the B witness of a point
owned by A onto A's witness projected on B's surface, which it already computes, so those points
cost nothing extra. Their drift is then a step's, and the manifold's relative-motion
invalidation, which acts before the drift limit, decides when they are regenerated. Points from
GJK's incremental path keep B's witness. A first version placed anchored points in the output
from both transforms. It cost tote 6%, 42,000 contacts each paying two quaternion transforms
per step.

Edge crossings move with neither shape. Box-box tags those clipped against a side of the
reference face `PCM_ANCHOR_CROSSING`; points clipped at the contact distance lie on B's edge and
keep B's witness. On reuse, `moveBoxBoxCrossings` moves both witnesses to where the two edges
cross now, seen along the point's normal, and updates the distance. A point on a box edge sits
at the extent on two axes, so the edge runs along the third and needs no storage. The edge is
taken from the two axes across the normal: at a corner every axis is at its extent, and choosing
a side's edge there, nearly parallel to the other box's edge as seen along the normal, moved the
crossing up that side and reported points centimetres apart; a pallet box whose edge overhung its
slipsheet by 60 um lost its edge's support that way and sank 52 um for 25 s. A crossing
that would leave either edge segment, as near-parallel edges of aligned boxes give, keeps its
witnesses until regeneration (moving it anyway blew up stacked cubes). Hull pairs report
crossings halfway between the witnesses (`PCM_ANCHOR_MIDPOINT`): exact placement there would need
the edges' hull indices, 16 more bytes per manifold.

Split belts, 0.6/0.4 m/s, 4 s:

| Contacts | Drift | Speed dip at the seam | Yaw / Coulomb |
|---|---|---|---|
| PhysX PCM | 2.0 mm | 7.5% | 1.0026 |
| Anchored, crossings halfway | 0.44 mm | 2.6% | 1.0017 |
| Anchored, box crossings exact | 0.0017 mm | 0.05% | 1.0009 |
| Regenerated every step | 0.0008 mm | 0.01% | 1.0009 |

Over 20 s the exact version drifts 0.25 mm through three seam crossings without locking. Wasm,
against PhysX PCM: pile drop -7%, 5x5x5 pile -4%, case -3%, pallet belt -4% and overall -2%,
tote within noise; the pallet fall phase is 4-5% slower on the mean and 9-10% on p95, which the
anchoring without exact crossings does not show.

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

A bouncing normal row (positive restitution, approach faster than the bounce threshold, and
closing within the step) instead targets the rebound speed directly: its free term is
`freeSpeed - targetSpeed`, with `targetSpeed` the restitution times the approach speed. The
relaxed form above loses part of the rebound at small timesteps. Its position term also adds
the depth at which the step finds the impact (without speculative contacts) or subtracts the
remaining gap (with them). At 10 ms the old form rebounded at 0.62/0.92 of the impact speed for
restitution 0.5/0.8 without speculative contacts, and at 0.37/0.67 with them. It now rebounds at
0.50/0.80 both ways (`BehaviourTests bouncing_balls`). PGS does the same: it drops the bias from
bouncing points.

A separated point that does not bounce pushes only to keep the step from carrying it through the
surface: its target is `targetSpeed - gap / timestep`, as PGS's speculative contacts. The spring
and damper act once the surfaces touch. The relaxed form used before damped a separated point's
approach too, slowing a body before it collided: a 64-sided hull rolling down a 5 degree ramp
reached 0.10 m/s in 1 s at any timestep, its next corner always within the contact offset, where a
rigid hull reaches 0.49 m/s; it now reaches 0.28 m/s at 10 ms and 0.42 m/s at 1 ms, as PGS does
(`BehaviourTests rolling`). On the wasm 125-box pile with a layer dropped every 0.2 s the mean step
is 2-4% slower, p95 4% and the worst step 13% faster, p99 22% slower (landings now arrive in one
step at full speed; an earlier solver measured +73% on p99). The pallet fall-off is faster (native
p95 2.5 -> 1.6 ms). A ball landing with speculative CCD penetrates 15-20 um instead of 4 um.

Contact points whose gap is unlikely to close within the step are deferred, left out of the
first solve: a point enters it within 1 mm of its rest distance, or when the approach over the
step of the motion at the step's start (last step's solution) toward the surface's target
velocity, with a margin of one half for pushes from other contacts, covers the gap
(`ANVIL_SPECULATIVE_KEEP_GAP`, `ANVIL_SPECULATIVE_APPROACH_MARGIN`). The solve then decides
the rest: after it, `addClosedAnvilContacts` tests the deferred points against the solved
velocities, and a point the step would carry through its surface (relative normal velocity below
the target speed less the gap per step, the bound its row enforces) joins the problem, which is
prepared again and solved from the solution so far. Passes repeat until no point closes. A late
point joins without friction rows: lagged friction bounds a point by its share of the pair's
normal impulse in the last step, and it had none; its normal impulse joins the pair's total,
which bounds the pair's friction from the next step. Without this, the FBF card house of real
playing cards (88 x 63 x 0.3 mm, `BehaviourTests card_house`) collapsed: its third-level tent
fell 14 mm with the stack onto a bridge, the landing spun both cards inward and their tops
closed 4.6 mm within the step through a 1.2 mm gap that had been dropped because the cards were
not approaching each other beforehand. Dropped points also let tumbling bodies sink 0.1-3 mm in
impacts (the convex pile 7,600 times in 800 steps). The deferral is summarized per pair
(`AnvilContactPair::deferredLimit`, the common normal and each body's reach to the farthest
deferred point), so the check reads a pair's contacts again only when its bodies' solved
velocities could close one; resting and sliding pairs pass in a few operations. The approach
test must be taken toward the target velocity: an earlier form ignored it, so a pallet
pivoting off a belt's end lost the edge contact, which closes by the belt speed's component
along the slanted edge normal (1 mm per step) and carries 14 times the island's next largest
impulse, and with the deferral in place every fall-off step re-solved. Taking the approach
after this step's gravity instead (the free velocity the row would hold back) made every hull
of a resting pile approach the points below it by one step's fall, and those inactive rows
cost the convex pile 16% without saving a re-solve; the start-of-step velocity is the natural
predictor of the solution for a body the contacts hold. Re-solves remain where impacts turn
bodies: the convex pile re-solves about 500 islands in 800 steps, 13 rows and 4 iterations
each against 113 for the first solve; the pallet and tote scenes none. A re-solve keeps the
bodies' ordering (`Settings::keepOrdering`): a late point usually adds a body pair with no kept
points, which changes the Hessian pattern, and finding a new ordering for 500 bodies cost 0.4 ms
of a 1.34 ms re-solve; the pattern is still analyzed (0.55 ms) and the factor rebuilt. The
re-solves took 12% of the pile's run before that and 9% after. The kept ordering serves only the
solve that added the rows: the next ordinary solve in that workspace forgets the analyzed
pattern, so the ordering stays a function of the pattern alone. (Without that, a workspace that
next solved the same pattern reused the kept ordering, and which workspace an island gets
depends on the scheduling: the convex pile differed between 1 and 8 workers and between runs.)
The symbolic analysis itself works on the body pairs: ordering, elimination tree and block
structure are built from the pairs and the permutation, and the permuted scalar Hessian with its
input maps is built only when the block factorization fails and the scalar fallback runs. The
scalar export, permutation and analysis had cost 0.3 ms of every pattern change on a 500-body
island (METIS 0.43 ms more), and the pile's pattern changes nearly every step while it moves.
The pile's collapse is chaotic
(rounding changes alone move its final speed between 50 and 140 mm/s), so single trajectories do
not compare its step totals; the per-island timing above and totals over ten pile sizes do.
Against the filter that dropped the points (native, interleaved): pallet fall-off mean 0.75 ->
0.66 ms, p95 1.59 -> 1.23, peak 2.36 -> 1.70 (the belt-edge contact now in the first solve);
totes 3.79 -> 3.75; the 5x5x20 convex pile 7.50 -> 7.42 mean, settled 2.21 -> 1.84, final speed
222 -> 53 mm/s. Ten convex pile sizes (3x3x20 to 6x6x20, 800 steps each) total 50.4 s dropping
the points, 53.4 s re-solving with the ordering kept, 52.1 s with the body-pair symbolic
analysis: the re-solves of real impacts cost collapsing hull piles about 3%. What remains of a
re-solve on a 500-body island is 0.57 ms against 7 ms for the step's first solve: re-preparation
0.06, pattern analysis 0.06, assembly 0.04, one factorization 0.12 and six iterations 0.35 ms,
each at its floor for a changed pattern. METIS options were tried for the ordering that every
pattern change pays (0.42 ms): no compression saves 7% of it at equal fill, fewer refinement
passes cost 15-24% more factor work; neither was kept. Wasm: pallet fall p95 1.89 -> 1.41 ms,
pile 1000 12.8 -> 12.6, the 125-box pile drop 2.49 -> 2.55, totes 3.99 ms (the per-point check
was 4.24).

The playing-card house creeps about 13 um/s and jolts by microns (`BehaviourTests card_house`
with `ANVIL_EXP_FRAMES=3000`). Both come from pairs whose every loaded point saturates for one
step: writeback marks them sliding, the next step prepares them sticking with their slip cleared,
and the released elastic slip snaps the structure; 175 such one-step flips in 30 s. They
saturate early because the pair's previous normal impulse is shared equally over its kept points,
including points 0.1-0.4 mm off the surface, so a bridge resting on one corner has that corner
capped at a quarter of the budget. Redistributing the budget by geometry (depth, touching points,
the pair's penetration as a margin, a plateau within it) collapses the house or sets a card
rattling: its tents are held by friction at points that do not touch (a card's top face is 0.3
mm wide and tilted 25 degrees, so its far edge, 0.13 mm clear, braces the near edge against
twisting, as a real card's crushed edge would), and shares that follow the gaps feed back through
them. Keeping the slip through a sliding step (as a bristle model would) lets the saturated pairs
slide instead. The remaining route is a contact margin in the normal law together with per-point
lagged impulses, which the slowed rolling hulls argue against at that reach.

PCM's box-box contact chose its reference axis by the smallest overlap alone. Two thin boxes
leaning against each other along an edge, the two cards of a tent, overlap almost equally along
either card's face axis, and each regeneration of the resting manifold could turn the normal by
52 degrees, which lapses the pair's slip and jolts: 6 times on loaded pairs in 30 s. The
generator now keeps the previous axis while it still interpenetrates and by no more than twice
the smallest (`doBoxBoxGenerateContacts`); the paper house's drift over 30 s fell from 0.048 to
0.012 mm and the playing cards' from 0.169 to 0.131 mm. The settling is chaotic, so the 10 s
window's extremes move between builds (max rotation 0.079 to 0.112 degrees); the test's turn
limit is the displacement limit as a turn, 0.25% of a radian. The creep stays, as a known
limitation of the equal friction split, by decision.

This matters with `PxRigidBodyFlag::eENABLE_SPECULATIVE_CCD`, which inflates a body's contact
reach by its motion per step: contacts then exist a step before surfaces meet, and a body
arriving at 1 m/s is stopped at the surface instead of embedding a centimetre and bouncing. On a
125-box pile with a layer dropped from 5 cm every 0.2 s, the flag takes the landing step from
8-20 ms to 3-7 ms and the twenty steps of re-settling after it from 3-7 ms to about 2 ms,
against 1.3 ms at rest; the same holds at 1,000 boxes. The price is PhysX's: every moving body's
pairs within its inflation are generated each step, and a scene of 2,000 totes riding a belt at
0.5 m/s, each box 8 mm from its wall, spends 55% more per step (PGS pays the same), most of it in
the narrow phase before the filter discards the points. The test scenes set the flag on every body.

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
descriptors by island once, so small islands share no lock or counter between workers.

Islands of more than 32 bodies, each a task of its own, are submitted after the small islands'
batches, ordered by bodies plus contacts with the largest last (`updatePostKinematic`). The
dispatcher's job queues are stacks (`PxSList`), so the last task submitted starts first, and a
large island that started behind the small ones would hold up the step while other threads idle.
In `tools/islands/MixedIslands.cpp` (a 294-box pile below the parallel-phase threshold, layers
dropped every 0.2 s, beside 8,000 single boxes), a pile listed before the singles took the step
from 5.0 to 5.6 ms; with the ordering both arrangements take 5.0-5.1 ms. Putting large islands
first instead made the step up to 16% slower. Results are bit-identical; tote, case and pallet
timings are unchanged within noise. The order costs one pass over the islands' node counts per
step, and a copy only when a large island is not already last.

Large Cholesky factors parallelize independent trailing block updates with up to eight
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
proportion to its load, so such a threshold failed at large scales. While a pair sticks, its
two bounded tangent rows form a square friction limit aligned with the initial slip direction,
not a circle; sliding points align the rows with the slip and soften the row across it (see
Equations).

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
