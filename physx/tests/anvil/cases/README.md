# Case conveyor benchmark

Ten 140 m conveyors carry 2,000 individual 0.4 m cube cases (200 per conveyor, 10 kg, friction
0.5) at 0.5 m/s. The belts accelerate from rest to that speed over the first second. Cases start at rest, 0.2 m apart, so none touches another: every case is an
independent island in contact only with its static belt. The belts move through contact target
velocities (PhysX contact modification, MuJoCo reference acceleration). The default run is
2,000 steps of 10 ms; the runners refuse runs longer than the 38 s before the leading case
reaches the end of its belt.

`CaseScene.h` holds the shared layout and per-step measurements: step time, case speeds, the
largest sink into and lift above the resting height, the smallest gap between neighbouring
cases, the largest sideways drift, and cases that left their belt. Each run writes
`<prefix>-steps.csv` and every case's final state in `<prefix>-final.csv`.

```text
CaseConveyor output-prefix [steps=2000] [dt=.01] [pgs|anvil] [threads=8] [position=4] [velocity=1] [surface-regularization=1e-2] [stiffening-depth=2e-5] [tolerance=1e-8] [belt-ramp=1]
MujocoCaseConveyor --write-scene CaseConveyor.xml
MujocoCaseConveyor CaseConveyor.xml output-prefix [steps=2000] [dt=.01] [iterations=100] [threads=8] [belt-ramp=1]
```

`CaseConveyor` runs PhysX with PGS (PhysX's default 4+1 iterations unless given) or Anvil
(default scene settings), both with `eENABLE_FRICTION_EVERY_ITERATION` as in the pallet
benchmark. `MujocoCaseConveyor` uses the pallet scene's solver settings, including contact
stiffening with PhysX Anvil's default endpoints, and the shared `MujocoConveyor.h` rest-distance
and belt-velocity adjustments. `belt-ramp` sets the seconds the belts take to reach full speed
from rest, 1 by default as on real conveyors; 0 starts them at full speed, so every case first
slides until friction brings it up to speed. It builds against MuJoCo 3.14.0 when the official Windows release
is unpacked in `compiler/anvil/vendor/mujoco-3.14.0` (to `native-build/mujoco-3.14.0`, since
both versions ship `mujoco.dll`), and `MujocoCaseConveyor313` against the pinned 3.13.

## Tote conveyor benchmark

The same ten conveyors, 180 m long, carry 2,000 open totes (200 per conveyor, 0.2 m apart) at
0.5 m/s, reached over the first second as in the case scene. Each tote is a 600 x 400 x 220 mm Euro container built from five box shapes, a 4 mm
base and four 4 mm walls, weighing 0.2 kg. It holds four 10 kg boxes (280 x 150 x 180 mm) in
a 2 x 2 grid, 8 mm from the walls and 16 mm from each other, covering 87% of the floor. Every
box-on-tote contact therefore has a 50:1 mass ratio, which PGS struggles with. Totes and boxes
start at rest; each tote and its boxes form an independent island touching only its belt.

`ToteScene.h` holds the layout and measurements. Tote rows match the case scene: speed, sink
into and lift above the belt, gaps, and sideways drift, plus the totes' tilt. Box rows are
measured in their tote's frame: speed, sink into and lift above the floor, horizontal slip,
and boxes that escaped over or through the walls. The runners refuse runs longer than the 38 s
before the leading tote reaches the end of its belt.

```text
ToteConveyor output-prefix [steps=2000] [dt=.01] [pgs|anvil] [threads=8] [position=4] [velocity=1] [surface-regularization=1e-2] [stiffening-depth=2e-5] [tolerance=1e-8] [contact-offset=5e-4] [belt-ramp=1]
MujocoToteConveyor --write-scene ToteConveyor.xml
MujocoToteConveyor ToteConveyor.xml output-prefix [steps=2000] [dt=.01] [iterations=100] [threads=8] [belt-ramp=1]
```

PhysX shapes use a 0.5 mm contact offset, so a pair makes contacts within 1 mm, matching
MuJoCo's margin. PhysX's default 2 cm per shape would add speculative contacts between every
box and its nearby walls and between the walls and the belt, five times Anvil's rows.

With 8 workers after the totes reach belt speed (belts starting at full speed), PGS (4+1) takes 2.43 ms per step natively and
2.73 ms in WebAssembly, Anvil 2.61 and 2.82 ms, and MuJoCo 3.14 42.2 ms
(`compiler/anvil/results/totes/benchmark-20260926-logcurve`). PGS boxes sink 4.3 mm through the
4 mm tote floor onto the belt, and the totes run up to 1.2% fast and tilt 0.29 degrees; 64
iterations still leave 3.4 mm. Anvil boxes sink 9.7 um and totes 12 um, at belt speed and level.
Anvil needs one iteration per tote once the stiffening curve is stable for light bodies under
heavy loads (see `../README.md`); with MuJoCo's curve it took 17% longer.

Belts that start at full speed (`belt-ramp` 0) make every tote slide for about 0.1 s while friction accelerates
it at 0.5 g, and its boxes slide inside it. Anvil then runs its dilatancy corrections on every
tote, so steps 2-15 take 10 ms, against 5.2 ms (native) and 3.6 ms (WebAssembly) with PGS. With
the default 1 s belt ramp (`benchmark-20260926-ramp1s`) nothing slides and every tote takes one Anvil
iteration from the first step: steps 2-15 take 3.2 ms natively and 3.9 ms in WebAssembly (PGS
2.8 and 3.9 ms). The first step of every solver is several times slower than the rest, since
the scene is set up then and WebAssembly is still compiling. The case scene also takes one
iteration per case throughout its ramp.
