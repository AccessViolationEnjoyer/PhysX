# Anvil solver performance

This report records the pre-integration MuJoCo-equation benchmark. Its approximately
2 ms result is not native PhysX performance. See [PHYSX_INTEGRATION.md](PHYSX_INTEGRATION.md)
for the current integration, contact-model differences and measurements.

The complete five-pallet solver path averages **2.0327 ms**, versus
**2.4271 ms** for the saved solver and **6.0169 ms** for native
MuJoCo Newton. This pass reduces the complete cost by **16.2%**.
These times include preparation, dispatch, solving, joins and writeback.
The mean is still above 2 ms; the median is **1.8853 ms** and p95 is
**3.1209 ms**. This remains a standalone prototype; PhysX integration is pending.

## Timing scope and scene

All five conveyors and pallets are measured together: five 10 kg pallets, 300 separate 1 kg
boxes, and twenty rigid 1 mm sheets of 100 g each. Surface speed is 0.2 m/s. Settings remain
10 ms timestep, eight workers, warm starting, pyramidal friction, positive contact margin and
compliance, normalized tolerance 1e-8 and a 100-iteration cap. No contact filtering, physics
settings, stopping rules or numerical-factor reuse rules were changed.

Three sequential 20-second runs per variant alternate execution order. Each has 2,000 steps;
the first 200 are excluded from steady timing statistics. Mean, median and p95 below pool
5,400 samples. Hardware: Intel Core i9-13900KS, Windows x64, Visual Studio 2022 Release,
MuJoCo 3.3.7. Live runs use no affinity override or detailed phase profiling.

| Implementation | Complete mean (ms) | Median (ms) | p95 (ms) |
| --- | ---: | ---: | ---: |
| Saved optimized solver | 2.4271 | 2.3058 | 3.5223 |
| Current Anvil | 2.0327 | 1.8853 | 3.1209 |
| Native MuJoCo Newton | 6.0169 | 5.8745 | 8.8453 |

Current run means: 2.0307, 2.0270, 2.0404 ms.
The complete mean is 2.96x faster than native in this scene.
Cold first steps took 7.6838, 10.7262, 12.0997 ms; the largest
steady sample was 5.5510 ms. This is not a per-frame 2 ms guarantee or a universal speedup.

The common boundary starts after source equations exist. Our adapter includes translating
those inputs into retained Anvil storage and writing the result back. Native timing covers
mj_fwdConstraint, including its internal setup/writeback. Collision detection, geometry/Jacobian
generation in the host pipeline, integration, diagnostics and file I/O are outside both timers.

For the current fused task layout, global input preparation averages
0.1144 ms, the island task interval 1.8156 ms,
and writeback 0.1023 ms. Preparation and solving overlap across islands;
there is no separate global solve-only interval. Their old timing.csv columns now contain -1
(unavailable). steps.csv solve_ms now consistently means the complete adapter cost.

Earlier 1.68 ms figures excluded preparation/writeback. The saved version's complete cost was
2.4750 ms in that earlier session and 2.4271 ms in this paired run; use complete costs
for comparison. Historical narrower measurements remain under results/optimization2.

## Changes retained

- Scalar contacts retain one contiguous six-coordinate Jacobian per endpoint, reducing each
  retained record from 384 to 128 bytes. Three-row contacts own a separate 240-byte tangent
  block without duplicating their normal row. The full Contact structure is an input builder;
  obsolete packed-row/coupled metadata was removed from it.
- Producers can count nonzero column entries as they emit coefficients. Final preparation
  assigns row maps, scalar values and coupled metadata while filling CSC storage, avoiding
  repeated full scans. The ordinary preparation API still counts entries for other callers.
- Each island task prepares synchronously and immediately solves, preserving locality and
  eliminating a global prepare-all barrier. All tasks join before result scattering.
- Coupled Hessian assembly reconstructs each endpoint Jacobian once and reuses it for its
  diagonal and cross contributions. Scalar products retain the faster CSC kernels.

The earlier block Hessian assembly, block/scalar Cholesky, incremental updates, linear sparse
permutation, ordered triangular substitution and raw CSC products remain. No alternative
solver mode, new tuning constant, approximate Hessian or cross-step numerical-factor cache was
added. Capacity is reused across timesteps; factors are rebuilt numerically for each solve.

## PhysX task feasibility

The source check preceded the scheduling experiment. PhysX launches independent prepare/solve
chains for island batches: PGS DyDynamics.cpp:1914-1942 and TGS DyTGSDynamics.cpp:628-654.
The local chains are documented/connected at DyDynamics.cpp:1610-1666 and
DyTGSDynamics.cpp:3223-3239. There is no requirement that every island finish preparation before
any island starts solving.

Existing preparation can launch child tasks (PGS:2558-2580; TGS:1858-1866). Calling those routines
and immediately solving would be incorrect: retain their local continuation, or implement
synchronous Anvil preparation as this harness does. Common narrowphase/contact modification,
constant data, world/kinematic initialization and pre-solve event snapshots remain prerequisites.
Final writeback/reporting and memory recycling still wait for all islands. The future PhysX
implementation can use its task system without an all-island preparation barrier; parallel
preparation within one island still needs a local join. No PhysX SDK source was changed here.

## Validation and storage

All three maintained CTests pass, including independent mixed scalar/bilateral/coupled checks,
1,000 changing-topology allocation checks and 400 live steps after warm-up on eight workers.
Measured prototype scopes allocate/free zero times after capacity is established. The Debug CRT
hook includes our code and Eigen; the separate release MuJoCo DLL is outside that hook.

The preparation experiments compare 96 input/topology cases and 192 cold/warm solve pairs,
including producer-supplied counts, structural zeros, inactive/world-side entries and empty
contact sets. Packed arrays and solutions match exactly; the additional warmed 384-cycle audit
reports zero preparation/solve allocations and frees. The 108 randomized mixed solves retain
identical outputs and the same two pre-existing extreme iteration-cap cases.

All recorded poses and every non-timing step field match the saved solver across the three
20-second runs. Both retain 34740 scalar rows on average, and reach at most
10 iterations after warm-up. No boxes pass through sheets. Maximum
steady sheet overlap is 1.44866119e-06 m, unchanged from the saved solver. Native uses
34429 rows on average; this pass does not remove contacts to obtain its speedup.

The final eight prepared cases retain identical primal/impulse outputs. Prepared kernel
medians improve from 13.0354 to 12.7565 ms for the cold stiff 125-box case, 2.9375 to 2.8403 ms
for its warm case, and 0.1227 to 0.1223 ms for 27 boxes. The tiny mass-ratio case changes from
0.0024 to 0.0027 ms; this is not an across-the-board performance improvement.

Both 20- and 120-link cables preserve every physical and solver-result field over all three
1,200-step runs. Kernel medians are 0.0135 to 0.0133 ms (20 links) and 0.0823 to 0.0829 ms
(120 links). The full cable harness step rises from 0.1040 to 0.1144 ms for 120 links: the compact representation
adds overhead when preparing the cable's legacy three-row inputs. Scalar native
producers write the compact layout directly. This measured tradeoff remains visible; the pallet
speedup must not be presented as making every scene faster.

The optional scalar diagnostic now uses only the stored active normal row. Previously an
unused tangent entry in a manually filled scalar input could affect that diagnostic's scale;
unused tangents never entered the scalar equations or normalized stopping criterion.

## Remaining cost

The final VTune software profile and worker phase timers place numeric factorization and
Jacobian/gradient evaluation ahead of the other kernel phases. Averaged sums across all island
jobs are 1.700 ms factorization, 1.187 ms evaluation, 0.713 ms Hessian assembly, 0.605 ms line
search, 0.404 ms updates and 0.303 ms backsolves. These are aggregate worker intervals from an
instrumented run, not wall times to add to the 2.0327 ms benchmark. Preparation totals 1.061 ms
across the workers. Symbolic analysis averages 0.218 ms and can be nested in factor work.

Software hotspot sampling also attributes substantial CPU time to the host thread pool's
SwitchToThread waits. That aggregate wait activity is not Anvil wall time and is not treated
as an available solver speedup. Hardware-event sampling was unavailable. The capture and
exports are under results/optimization3/vtune-final-sw and vtune-hotspots.csv.

## Experiments and provenance

Initial two-run 12-second comparisons measured complete costs of 2.6105 ms for the saved
solver, 2.4734 ms for counted preparation alone, 2.2781 ms for compact storage alone, and
2.5598 ms for fused tasks alone. Combined staged/fused preparation measured 2.1386/2.1083 ms.
Most of the improvement comes from storage and preparation rather than scheduling.

Direct Jacobian-vector multiplication from compact rows was tested without extra storage.
It preserved exact results but increased complete cost from 2.1155 to 2.2298 ms (5.4% slower),
so it was rejected. A compile-time scalar-preparation specialization also failed to establish
an improvement: median run means were 2.0550 versus 2.0525 ms across three alternating pairs,
while frame medians were worse. The extra specialization was not retained.
Detailed earlier algorithm/kernel experiments remain under
physx/compiler/anvil/experiments and are excluded from the maintained build.

The pre-pass backup is backups/Anvil-full-pipeline-2026-09-14_17-09-09.zip,
SHA256 defc1749455a33bcba28539f8b7a92a88de9178faa79d2ac95327a0f3b2de309.
Source/binary hashes, commands, raw timing/pose CSVs and equivalence results are recorded in
physx/compiler/anvil/results/optimization3/summary.json and final-live/.

Reproduce the full comparison from the repository root with
results/optimization3/ComparePipeline.py (paths are under physx/compiler/anvil), using the
saved baseline/pallet executable and the current pallet-build/release executable. Request
--steps 2000 --batches 3 --threads 8; set the native variant to --mode native=mujoco.
