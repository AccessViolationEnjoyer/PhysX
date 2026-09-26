# Case conveyor benchmark

Ten 140 m conveyors carry 2,000 individual 0.4 m cube cases (200 per conveyor, 10 kg, friction
0.5) at 0.5 m/s. Cases start at rest, 0.2 m apart, so none touches another: every case is an
independent island in contact only with its static belt. The belts move through contact target
velocities (PhysX contact modification, MuJoCo reference acceleration). The default run is
2,000 steps of 10 ms; the runners refuse runs longer than the 38 s before the leading case
reaches the end of its belt.

`CaseScene.h` holds the shared layout and per-step measurements: step time, case speeds, the
largest sink into and lift above the resting height, the smallest gap between neighbouring
cases, the largest sideways drift, and cases that left their belt. Each run writes
`<prefix>-steps.csv` and every case's final state in `<prefix>-final.csv`.

```text
CaseConveyor output-prefix [steps=2000] [dt=.01] [pgs|newton] [threads=8] [position=4] [velocity=1]
MujocoCaseConveyor --write-scene CaseConveyor.xml
MujocoCaseConveyor CaseConveyor.xml output-prefix [steps=2000] [dt=.01] [iterations=100] [threads=8]
```

`CaseConveyor` runs PhysX with PGS (PhysX's default 4+1 iterations unless given) or Newton
(default scene settings), both with `eENABLE_FRICTION_EVERY_ITERATION` as in the pallet
benchmark. `MujocoCaseConveyor` uses the pallet scene's solver settings, including the contact
stiffening that matches PhysX Newton's default, and the shared `MujocoConveyor.h` rest-distance
and belt-velocity adjustments. It builds against MuJoCo 3.14.0 when the official Windows release
is unpacked in `compiler/newton/vendor/mujoco-3.14.0` (to `native-build/mujoco-3.14.0`, since
both versions ship `mujoco.dll`), and `MujocoCaseConveyor313` against the pinned 3.13.
