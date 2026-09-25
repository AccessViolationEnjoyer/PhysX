# Pallet conveyor comparison

This is a full simulation test with automatic collision detection, moving contact surfaces,
free rigid bodies and integration. The PhysX snippet supports PGS and the integrated Newton
solver. A separate MuJoCo runner compares the same scene with native MuJoCo Newton and the
historical standalone Newton adapter.

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

The rendered snippet uses Newton with a 10 ms timestep, eight worker threads, regularization
`1e-4` and a 100-iteration limit. Press `P` to pause or resume, `O` to advance one timestep
while paused, and `R` to reset the scene. The render loop is intended for visual inspection;
use the separate benchmark below for timings without rendering overhead.

## Build

Run from the repository root, after preparing the existing Newton dependencies:

```powershell
python physx/tests/newton/PrepareDependencies.py
cmake -S physx/tests/newton/pallet -B physx/compiler/newton/native-build -G "Visual Studio 17 2022" -A x64 -DPX_NEWTON_USE_AVX2=ON
cmake --build physx/compiler/newton/native-build --config profile --target SnippetPalletConveyor MujocoPalletConveyor --parallel 4
```

The executables are under `physx/compiler/newton/native-build/profile`. This builds CPU-only
PhysX static libraries in the isolated benchmark build directory.

## Run

Output-prefix parent directories must already exist. From the repository root:

```powershell
New-Item -ItemType Directory -Force physx/compiler/newton/results/pallet/manual
physx/compiler/newton/native-build/profile/SnippetPalletConveyor.exe physx/compiler/newton/results/pallet/manual/pgs 2000 .01 16 2 8 pgs
physx/compiler/newton/native-build/profile/SnippetPalletConveyor.exe physx/compiler/newton/results/pallet/manual/physx-newton 2000 .01 16 2 8 newton 1e-4 100
physx/compiler/newton/native-build/profile/MujocoPalletConveyor.exe physx/tests/newton/pallet/PalletConveyor.xml physx/compiler/newton/results/pallet/manual/mujoco mujoco 2000 .01 100 8
physx/compiler/newton/native-build/profile/MujocoPalletConveyor.exe physx/tests/newton/pallet/PalletConveyor.xml physx/compiler/newton/results/pallet/manual/newton prototype 2000 .01 100 8
```

Arguments:

```text
SnippetPalletConveyor output-prefix [steps=2000] [dt=.01] [position=16] [velocity=2] [threads=8] [pgs|newton] [regularization=1e-4] [Newton-iterations=100] [dilatancy-corrections=4] [surface-regularization=1e-2] [stiffening-depth=2e-5]
MujocoPalletConveyor scene.xml output-prefix mujoco|prototype [steps=2000] [dt=.01] [iterations=100] [threads=8] [impedance=0] [audit=0] [profile=0]
```

The optional impedance argument sets a constant contact impedance (both `solimp` ends); zero
means use the XML. The XML uses `solref="0.02 1"` and `solimp="0.990099 0.9999 0.00002 0.5 2"`,
matching PhysX Newton's defaults: regularization 1e-2 at first touch, stiffening to 1e-4 over
20 um of penetration (impedance 0.9999, or regularization 1e-4, is the former constant setting). Both Newton modes use a maximum of 100 iterations and MuJoCo-style
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
For the prototype, `solve_ms` includes preparation, all Newton jobs, dispatch/join and
result writeback. Preparation and solving overlap across islands; obsolete separate global
preparation/solver timing fields contain -1. Native `solve_ms` times `mj_fwdConstraint`, including its internal
setup/writeback. Other MuJoCo pipeline stages are outside both solver timers. `step_ms` still
records the full physics step. The current solver comparison is in [../PERFORMANCE.md](../PERFORMANCE.md).

To regenerate the checked-in XML after editing the shared dimensions:

```powershell
physx/compiler/newton/native-build/profile/MujocoPalletConveyor.exe --write-scene physx/tests/newton/pallet/PalletConveyor.xml
```

## Conveyor and contact semantics

PhysX uses `onContactModify` only for pairs involving a conveyor. The callback sets target
velocity with the sign appropriate to the actor order. Conveyors themselves remain static.
PhysX retains its default **2 cm contact offset per shape** and zero rest offset; the old
zero-margin experiments do not constrain this test. PGS uses its patch-friction model.
Newton uses four nonnegative MuJoCo pyramid edges per frictional point and MuJoCo's
reference-acceleration equation.

MuJoCo does not expose a contact target-velocity setter. The runner adjusts the contact
reference acceleration after `mj_step1`, using the prescribed belt velocity projected onto
each pyramidal edge. Both Newton implementations receive that adjustment. Conveyor body
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
Newton implementations *are* compared with identical settings in the same engine pipeline.

Opening the XML in a generic MuJoCo viewer shows the geometry, but the moving belt and
zero-rest-distance adjustment require this runner. The XML alone cannot encode contact
surface velocity. `ReplayPallet.py` renders the saved simulation poses from either engine.

## Repeat and inspect

```powershell
python physx/tests/newton/pallet/ComparePallet.py
python physx/tests/newton/pallet/PlotPallet.py
python physx/tests/newton/pallet/ReplayPallet.py physx/compiler/newton/results/pallet/comparison/prototype-1 physx/compiler/newton/results/pallet/comparison/newton.mp4
```

The comparison runs 20 simulated seconds, three repetitions, eight workers, with alternating
case order. Cases are PGS 16+2, PGS 98+2, native Newton, prototype Newton at 10 ms, plus PGS
98+2 at 1 ms. They run sequentially. Summary data are written after each completed run.
`--seconds`, `--repeats` and `--threads` can shorten or expand the comparison. Use
`--cases mujoco prototype --output <directory>` to compare only the two Newton implementations
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
is rebuilt on each step. Newton retains assembly, ordering, factor and scratch storage, rebuilding
the symbolic pattern when needed and the numeric factor for each solve that requires an update. This harness is not a general PhysX scene integration.
