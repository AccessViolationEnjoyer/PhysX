# Pallet conveyor comparison

This is a full simulation test with automatic collision detection, moving contact surfaces,
free rigid bodies and integration. The PhysX snippet supports PGS and the integrated Anvil
solver. A separate MuJoCo runner compares the same scene with native MuJoCo Newton and the
historical standalone Anvil adapter.

The scene contains five independent conveyors, each carrying one loaded pallet:

| Item | Dimensions (m) | Mass | Count per conveyor |
| --- | --- | --- | --- |
| Pallet | 1.2 x 0.15 x 1.0 | 10 kg | 1 |
| Boxes | 0.3 x 0.25 x 0.3 | 1 kg | 60: five layers of 3 x 4 |
| Slipsheets | 1.2 x 0.001 x 1.0 | 0.1 kg | 4, between layers |
| Conveyor | 12 x 0.2 x 1.2 | Static | 1 |

There are 325 dynamic bodies: 300 boxes, 20 sheets and five pallets. Dimensions are full
extents in x/y/z; y is up. The conveyors have a 1 m edge-to-edge gap. Their top surfaces are
at y=0.5 m. Pallets begin at x=-3 m and travel in +x at a surface speed of 0.2 m/s. Friction is
0.5. The sheets are rigid boxes, not flexible sheets. There is no CCD, no articulation and no
sleeping. Rendering and CSV output are outside the measured simulation steps.

`PalletScene.h` defines geometry and masses once. The PhysX snippet uses those definitions
directly, and `MujocoPalletConveyor --write-scene` generates `PalletConveyor.xml` from them.

## Rendered snippet

Generate the standard CPU-only PhysX solution, build the snippet and run it from the repository
root:

```powershell
cd physx
generate_projects.bat vc17win64-cpu-only
cmake --build compiler/vc17win64-cpu-only --config profile --target SnippetPalletConveyor --parallel 4
.\bin\win.x86_64.vc143.md\profile\SnippetPalletConveyor_64.exe
```

The rendered snippet uses Anvil with a 10 ms timestep, eight worker threads, regularization
`1e-4` and a 100-iteration limit. Press `P` to pause or resume, `O` to advance one timestep
while paused, and `R` to reset the scene. The render loop is intended for visual inspection;
use the separate benchmark below for timings without rendering overhead.

## Build

Run from the repository root, after preparing the existing Anvil dependencies:

```powershell
python physx/tests/anvil/PrepareDependencies.py
cmake -S physx/tests/anvil/pallet -B physx/compiler/anvil/native-build -G "Visual Studio 17 2022" -A x64 -DPX_ANVIL_USE_AVX2=ON
cmake --build physx/compiler/anvil/native-build --config profile --target SnippetPalletConveyor MujocoPalletConveyor --parallel 4
```

The executables are under `physx/compiler/anvil/native-build/profile`. This builds CPU-only
PhysX static libraries in the isolated benchmark build directory.

## Run

Output-prefix parent directories must already exist. From the repository root:

```powershell
New-Item -ItemType Directory -Force physx/compiler/anvil/results/pallet/manual
physx/compiler/anvil/native-build/profile/SnippetPalletConveyor.exe physx/compiler/anvil/results/pallet/manual/pgs 2000 .01 16 2 8 pgs
physx/compiler/anvil/native-build/profile/SnippetPalletConveyor.exe physx/compiler/anvil/results/pallet/manual/physx-anvil 2000 .01 16 2 8 anvil 1e-4 100
physx/compiler/anvil/native-build/profile/MujocoPalletConveyor.exe physx/tests/anvil/pallet/PalletConveyor.xml physx/compiler/anvil/results/pallet/manual/mujoco mujoco 2000 .01 100 8
physx/compiler/anvil/native-build/profile/MujocoPalletConveyor.exe physx/tests/anvil/pallet/PalletConveyor.xml physx/compiler/anvil/results/pallet/manual/anvil prototype 2000 .01 100 8
```

Arguments:

```text
SnippetPalletConveyor output-prefix [steps=2000] [dt=.01] [position=16] [velocity=2] [threads=8] [pgs|anvil] [regularization=1e-4] [Anvil-iterations=100] [friction-corrections=4] [surface-regularization=1e-2] [stiffening-depth=2e-5]
MujocoPalletConveyor scene.xml output-prefix mujoco|prototype [steps=2000] [dt=.01] [iterations=100] [threads=8] [impedance=0] [audit=0] [profile=0]
```

The optional impedance argument sets a constant contact impedance (both `solimp` ends); zero
means use the XML. The XML uses `solref="0.02 1"` and `solimp="0.990099 0.9999 0.00002 0.5 2"`,
with the endpoints of PhysX Anvil's defaults: regularization 1e-2 at first touch, stiffening to
1e-4 over 20 um of penetration (impedance 0.9999, or regularization 1e-4, is the former constant
setting). MuJoCo's smooth step is in impedance and PhysX's in log regularization, so at a given
depth PhysX contacts are stiffer. Both Anvil modes use a maximum of 100 iterations and MuJoCo-style
normalized stopping tolerance 1e-8. Warm starting is enabled. PhysX uses
`eENABLE_FRICTION_EVERY_ITERATION` and the requested position/velocity iterations.

`audit=1` additionally solves each prototype frame with native MuJoCo from the same input
state and warm start. It compares next velocities and records iteration counts in an audit
CSV. The native reference solve is excluded from timers; use ordinary runs for performance.
`profile=1` writes a `-profile.csv` containing summed worker preparation and solver phase times.
These are sums of per-island elapsed timers, not aggregate wall time or thread CPU time;
symbolic time is included in factor time.
Detailed solver timers are disabled in normal timing runs.

`-timing.csv` always records preparation, solver, scatter and complete adapter wall times.
For the prototype, `solve_ms` includes preparation, all Anvil jobs, dispatch/join and
result writeback. Preparation and solving overlap across islands; obsolete separate global
preparation/solver timing fields contain -1. Native `solve_ms` times `mj_fwdConstraint`, including its internal
setup/writeback. Other MuJoCo pipeline stages are outside both solver timers. `step_ms` still
records the full physics step. The current solver comparison is in [../PERFORMANCE.md](../PERFORMANCE.md).

To regenerate the checked-in XML after editing the shared dimensions:

```powershell
physx/compiler/anvil/native-build/profile/MujocoPalletConveyor.exe --write-scene physx/tests/anvil/pallet/PalletConveyor.xml
```

## Conveyor and contact semantics

PhysX uses `onContactModify` only for pairs involving a conveyor. The callback sets target
velocity with the sign appropriate to the actor order. Conveyors themselves remain static.
PhysX retains its default **2 cm contact offset per shape** and zero rest offset; the old
zero-margin experiments do not constrain this test. PGS uses its patch-friction model.
Anvil uses four nonnegative MuJoCo pyramid edges per frictional point and MuJoCo's
reference-acceleration equation.

MuJoCo does not expose a contact target-velocity setter. The runner adjusts the contact
reference acceleration after `mj_step1`, using the prescribed belt velocity projected onto
each pyramidal edge. Both Anvil implementations receive that adjustment. Conveyor body
positions remain fixed. The remaining stages follow Euler `mj_step2`, with only the
constraint solve replaced in prototype mode.

The MuJoCo XML uses a **positive 1 mm contact-detection margin**. The runner removes the
margin's positional bias from the reference acceleration, keeping the intended rest distance
at zero, as in PhysX. MuJoCo also measures `solimp` impedance from the margin, so the runner
recomputes each contact row's impedance from its penetration and rescales that row's
regularization and island copies to match; with constant `solimp` only the margin offset changes. Thus the margin does not artificially prop the sheets apart. MuJoCo's
large positive-margin behavior differs from PhysX's speculative-contact handling; copying
PhysX's 4 cm pair envelope made the MuJoCo scene unstable. This is a documented engine
setting difference, not an identical-equations comparison between PhysX and MuJoCo. The two
Anvil implementations *are* compared with identical settings in the same engine pipeline.

Opening the XML in a generic MuJoCo viewer shows the geometry, but the moving belt and
zero-rest-distance adjustment require this runner. The XML alone cannot encode contact
surface velocity. `ReplayPallet.py` renders the saved simulation poses from either engine.

## Repeat and inspect

```powershell
python physx/tests/anvil/pallet/ComparePallet.py
python physx/tests/anvil/pallet/PlotPallet.py
python physx/tests/anvil/pallet/ReplayPallet.py physx/compiler/anvil/results/pallet/comparison/prototype-1 physx/compiler/anvil/results/pallet/comparison/anvil.mp4
```

The comparison runs 20 simulated seconds, three repetitions, eight workers, with alternating
case order. Cases are PGS 16+2, PGS 98+2, native Anvil, prototype Anvil at 10 ms, plus PGS
98+2 at 1 ms. They run sequentially. Summary data are written after each completed run.
`--seconds`, `--repeats` and `--threads` can shorten or expand the comparison. Use
`--cases mujoco prototype --output <directory>` to compare only the two Anvil implementations
without overwriting the original five-method results.

Measured results are summarized in RESULTS.md. Unavailable PhysX point/row counts are recorded as -1.

Each runner writes:

- `-steps.csv`: full-step time, solve time when available, contact counts, maximum island
  iteration count, pallet travel/speed and sheet-plane intrusion.
- `-poses.csv`: every body's position and orientation every ten steps, for replay.
- `-bodies.csv`: names, dimensions, masses and original layer assignments.

The intrusion metric projects each box adjacent to a sheet onto that sheet's normal,
checking that their projected footprints overlap. It counts an intrusion through the 1 mm
thickness when the signed distance crosses the far surface. This is a useful failure
indicator for the aligned stack; it is not an exact intersection-volume calculation for
boxes that have already fallen off or rotated substantially. It also detects a sheet
escaping through a layer by retaining the original adjacency. Timed simulation does not
include these diagnostics, pose extraction, rendering or CSV I/O.

The prototype adapter is intentionally limited to centered free boxes with diagonal inertia
in their body frames and pyramidal contact rows. It uses MuJoCo's dynamically discovered
islands and thread pool, including when contacts change or islands split. The factorization
within each island is serial. Task arrays, preparation buffers and solver outputs retain capacity across steps. Every equation
is rebuilt on each step. Anvil retains assembly, ordering, factor and scratch storage, rebuilding
the symbolic pattern when needed and the numeric factor for each solve that requires an update. This harness is not a general PhysX scene integration.

## Pallet on a shuttling platform

`PalletPlatform.h` places one pallet of the conveyor scene (60 boxes, four slipsheets) on a
platform that shuttles along x over a 1 m stroke: it accelerates at 0.5 m/s^2 to 0.5 m/s,
cruises, and decelerates to rest at each end, so every 3 s stroke loads the whole stack
sideways at 0.05 g and then reverses. Each step records the pallet's slip on the platform, the
largest horizontal drift of any box or sheet in the pallet's frame, the largest speed of any
body relative to the platform, and the slipsheet overlap and crossings of the conveyor scene.

```text
AnvilPalletPlatform output-prefix [steps=6000] [dt=.01] [pgs|anvil] [threads=8] [position=16] [velocity=2] [contact-offset=5e-4]
MujocoPalletPlatform --write-scene PalletPlatform.xml
MujocoPalletPlatform PalletPlatform.xml output-prefix [steps=6000] [dt=.01] [iterations=100] [threads=8]
```

In PhysX the platform is a kinematic body driven by targets, and shapes use a 0.5 mm contact
offset (MuJoCo's 1 mm margin per pair) instead of PhysX's 2 cm default. MuJoCo (3.14, the
conveyor scenes' settings) has no kinematic bodies: its platform is a mocap body moved after
each step, whose contacts take the platform's velocity over the step through the reference
acceleration, as the conveyor belts' do.

Over 60 s (10 cycles; `compiler/anvil/results/platform/benchmark-20260926-optimized`, medians of
three interleaved runs; steps exclude the first):

| Variant | Mean step | Box drift, max / after 10 cycles | Sheet overlap |
|---|---|---|---|
| PhysX PGS 16+2, 8 workers | 0.30 ms (wasm 0.32) | 14 / 12 mm | 4.7 mm, 64 box-sheet crossings |
| PhysX Anvil, inline | 0.42 ms (wasm 0.56) | 0.72 / 0.05 mm | 9.6 um |
| PhysX Anvil, 8 workers | 0.50 ms (wasm 0.63) | 0.72 / 0.05 mm | 9.6 um |
| MuJoCo 3.14, one thread | 1.55 ms | 1.05 / 0.08 mm | 14 um |
| MuJoCo 3.14, 8 threads | 1.59 ms | 1.05 / 0.08 mm | 14 um |

Anvil and MuJoCo hold the stack. Boxes shift on the pallet while it accelerates, as friction
regularization lets loaded contacts creep (Anvil 0.77 mm/s, MuJoCo 1.1 mm/s with its softer
impedance step), and shift back while it decelerates. PGS boxes sink through the 1 mm slipsheets
from the first steps; with PhysX's 2 cm contact offset they still sink 2.5 mm but drift only
2 mm. The whole stack is one island, so worker threads only add overhead.

Anvil first took 0.72 ms natively and 1.07 ms in WebAssembly (`benchmark-20260926`). The boxes
touch their neighbours, and boxes on either side of a sheet are exactly one contact offset apart,
so 355 box pairs make 5,560 contact rows of which only 451 carry load. Single-precision poses make
those rows flip at the edge of contact every step, and Anvil spent 4.5 iterations chasing
corrections below the poses' resolution; `anvilDisplacementTolerance` stops it after one. The
block factor also skips the zero blocks of unloaded pairs.