# Pallet conveyor results

The Newton timings below describe the earlier MuJoCo-hosted benchmark. Current native
PhysX results and their different contact equations are documented in
[PHYSX_INTEGRATION.md](../PHYSX_INTEGRATION.md).

**Current complete solver cost:** **2.0327 ms mean**, including preparation, dispatch,
solving and writeback. The saved implementation measures **2.4271 ms** and native
MuJoCo **6.0169 ms** in the same alternating comparison. The current median is
**1.8853 ms**, p95 **3.1209 ms**; the mean remains above 2 ms. Earlier 1.68 ms
figures excluded preparation/writeback. See [the current report](../PERFORMANCE.md).
All results below are historical.

## Historical results before the optimization pass

The requested test reproduces the thin-sheet problem in PGS. Both Newton implementations
keep the sheets intact at a 10 ms timestep. No Newton performance run reaches its
100-iteration cap. All pallets reach 0.2 m/s and travel approximately 4 m over 20 seconds.

These are full simulation measurements for **all five loaded pallets together**, with
325 dynamic bodies, automatic collision detection and eight workers. Three 20-second runs
were measured per configuration on an Intel Core i9-13900KS, Windows x64, Release builds.
Cases ran sequentially with alternating order. The table uses the median of the three
whole-run means; initialization, diagnostics, file I/O and rendering are excluded.

| Method | Timestep | Mean full step | Cost per 10 ms simulated | Maximum sheet intrusion |
| --- | ---: | ---: | ---: | ---: |
| PhysX PGS 16+2 | 10 ms | 0.79 ms | 0.79 ms | 2.626 mm |
| PhysX PGS 98+2 | 10 ms | 2.71 ms | 2.71 ms | 1.569 mm |
| MuJoCo Newton, cap 100 | 10 ms | 11.35 ms | 11.35 ms | 0.001469 mm |
| Our incremental Newton, cap 100 | 10 ms | 12.17 ms | 12.17 ms | 0.001473 mm |
| PhysX PGS 98+2 | 1 ms | 2.72 ms | 27.21 ms | 0.08048 mm |

The sheet thickness is 1 mm. At the end of the 10 ms runs, 179 original box/sheet
adjacencies in PGS 16+2 and 62 in PGS 98+2 intrude through that thickness. Neither Newton
run, nor the 1 ms PGS control, has any. This measures sheet-normal intrusion where the
projected footprints overlap, not an exact intersection volume; see README.md.

Our Newton is about **7.2% slower than native MuJoCo** on whole-run average full-step time
in this scene. It costs about **45% of the 1 ms PGS control** per simulated second and has
substantially smaller overlap. Both Newton modes exceed a 10 ms real-time budget slightly
on this machine; the result is not a claim that the scene already runs in real time.

After the first two simulated seconds, the median full-step times across repetitions were
0.780 ms, 2.671 ms, 11.179 ms, 11.806 ms and 2.675 ms respectively. The average maximum
island iteration count was 4.91 for native Newton and 5.04 for our Newton. Each engine had
approximately 2,669-2,670 contacting shape pairs. MuJoCo had about 8,600 contact points and
34,400 scalar pyramid-edge rows; PhysX uses its native patch-friction representation.

## Scope and contact settings

**The PhysX snippet uses stock PGS. Our Newton still runs through the MuJoCo pipeline;
it is not integrated into PhysX yet.** Its full-step timing includes MuJoCo collision
processing, contact-to-prototype conversion, task dispatch, the Newton solve and integration.
Native MuJoCo uses the same scene, contact model, thread pool and integration pipeline.

PhysX retains its default 2 cm contact offset per shape and zero rest offset, with
`eENABLE_FRICTION_EVERY_ITERATION`. MuJoCo uses a positive 1 mm contact-detection margin,
zero rest distance through the runner, constant impedance 0.9999 and `solref="0.02 1"`.
Copying PhysX's large pair envelope into MuJoCo produced an unstable simulation, even with
native Newton. That is not used in the reported results. The earlier zero-margin diagnostic
runs are also excluded. The two engines' collision and friction models differ; these results
do not isolate PGS versus Newton on identical PhysX constraints. The two Newton modes do
use identical equations and settings within MuJoCo.

The next PhysX integration needs to confirm this behavior with PhysX contact generation and
contact preparation. A direct translation of a collision margin between the engines is not
sufficient to reproduce their contact semantics.

## Checks

- Both executables build without compiler warnings in Release.
- Geometry validation confirms five pallets, 300 independent boxes, twenty 1 mm sheets,
  five static belts and 352 kg total dynamic mass.
- Each configuration's saved trajectories are byte-identical across all three repetitions.
- All five configurations reach the requested conveyor speed and retain all boxes on the pallets.
- A separate 200-step audit feeds our solution and native MuJoCo the same inputs. With a
  tightened normalized tolerance of 1e-12, their maximum next-velocity difference is
  7.92e-9 across linear components in m/s and angular components in rad/s. Three strict audit
  frames reached the 100-iteration cap; no performance frame did.
- At the production tolerance of 1e-8, the corresponding maximum difference is 5.80e-5.
  The stopping criterion controls normalized objective/gradient, not a fixed velocity error.
- The final task cleanup, logging and checkerboard-color changes leave the physics equations
  unchanged. The original benchmark hashes and current source hashes are retained in the
  generated JSON. Unavailable release-build PhysX row statistics are marked -1; timing and
  physical measurements are unchanged.

## Artifacts

Relative to `physx/compiler/newton/results/pallet/comparison`:

- `comparison.json`: all fifteen runs, commands, timing summaries and hashes.
- `pallet-comparison.png` / `.svg`: intrusion and full simulation cost plots.
- `newton.mp4`: replay of all five conveyors using recorded Newton poses.
- `newton-final.png`: overview after 20 simulated seconds.
- `pgs-final.png` / `newton-lane-final.png`: matching close views.
- `*-steps.csv`, `*-poses.csv`, `*-bodies.csv`: measurements and replay data.

See README.md for build and reproduction commands. PhysX 5.10.0 and MuJoCo 3.3.7 were used.
