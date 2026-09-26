# Softer contacts in the connected pile

Both methods were compared with MuJoCo's default solimp parameters,
`0.9 0.95 0.001 0.5 2`, and with an intermediate constant impedance
`0.99 0.99 0.001 0.5 2` for 500 boxes. The earlier hard case uses
`0.9999 0.9999 0.001 0.5 2`.

## Configuration and timing scope

The pile is a regular dense grid of separate 30 cm, 1 kg cubes. The 500-box case has
10 columns across, 10 deep, and 5 layers high; the 1,000-box case has 10 layers.
Alternating layers shift 7.5 cm in both horizontal directions, connecting neighboring
columns through actual contacts. There are no random positions, random orientations,
joints or ties. The floor is static. Every recorded frame and snapshot remains one island
containing all bodies.

Only the solimp attribute changes. XML structure, all other attributes and body transforms
were compared against the hard case. Timestep stays 10 ms, eight workers, warm starting,
pyramidal friction, 100-iteration cap, normalized tolerance 1e-8, solref `0.02 1`, and
1 mm contact margin with the same rest-distance correction for both methods. The official
MuJoCo 3.3.7 wheel DLL and benchmark executable hashes match the previous official verification.
No Anvil kernel or SDK source changes were made.

Each case advances our ordinary simulation for 200 steps (2 seconds), then compares five
alternating pairs of complete solves on identical host equations and a restored, fixed warm
start. Collision detection and integration are outside the solve timer; our preparation,
dispatch, solve, join and writeback are inside. These are captured-state comparisons, not
long-run throughput measurements. The hard rows below are the previous official-DLL run;
they were not timed again in this session.

## Results

| Boxes | Impedance endpoints | Our mean (ms) | Native mean (ms) | Our median (ms) | Native median (ms) | Iterations, ours / native |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 500 | 0.9999/0.9999 | 50.373 | 774.744 | 48.461 | 749.713 | 3 / 3 |
| 1,000 | 0.9999/0.9999 | 233.852 | 2834.771 | 230.679 | 2792.275 | 3 / 3 |
| 500 | 0.9/0.95 | 78.177 | 862.669 | 69.539 | 855.665 | 5 / 5 |
| 1,000 | 0.9/0.95 | 281.522 | 3217.759 | 281.475 | 3226.882 | 5 / 5 |
| 500 | 0.99/0.99 | 64.678 | 743.318 | 63.750 | 745.750 | 5 / 5 |

Both methods agree on velocity increments to better than 1e-15, with no capped snapshot
solves. The default-solimp snapshots both need five iterations, compared with three in the
hard snapshots. The intermediate 500-box case also retains the expensive coupled system.
The 500-box softer mean includes one 113.56 ms sample; no outliers were discarded.

The default-solimp captures have 47,812 scalar rows for 500 boxes and 101,540 for 1,000,
versus 48,584 and 102,856 in the hard captures. Native reported H+L storage remains
1,684,464 and 3,706,284 nonzeros respectively. Softness does not eliminate the large
connected system in this configuration. These structural statistics alone do not identify
which numeric phase dominates runtime.

The softer trajectories are also less stationary at capture: maximum speed immediately
before the snapshot is about 0.109 mm/s for 500 boxes and 0.194 mm/s for 1,000. Contact
penetration is about 0.057 mm and 0.123 mm, respectively, confirming increased compliance.
Consequently the hard and soft snapshots are not identical physical states or equally
converged equilibria. This result does not mean that softer contacts generally slow Anvil;
it establishes that changing softness alone did not improve these captured-state timings.
Within each comparison the two methods always solve exactly the same equations.

With this 1 mm margin and 1 mm impedance width, touching or penetrating contacts reach the
upper default impedance, 0.95. The rest-distance correction shifts reference acceleration
but leaves that impedance profile intact. Therefore these use default solimp parameters
with our existing margin/rest convention, rather than the complete MuJoCo default setup.
Changing solimp also changes the host's stiffness/damping normalization; both methods
receive the regenerated regularization and reference terms on every solve.

## Reproduction and evidence

From the repository root, using the isolated official-DLL executable already preserved:

```powershell
python -B physx/tests/anvil/pile/ComparePile.py --executable physx/compiler/anvil/results/pile/official-mujoco-validation/bin/AnvilPileBenchmark.exe --impedance 0.9 0.95 --sizes 500 1000 --steps 201 --repeats 1 --threads 8 --snapshot-repeats 5 --output physx/compiler/anvil/results/pile/softer-default-snapshots
python -B physx/tests/anvil/pile/ComparePile.py --executable physx/compiler/anvil/results/pile/official-mujoco-validation/bin/AnvilPileBenchmark.exe --impedance 0.99 0.99 --sizes 500 --steps 201 --repeats 1 --threads 8 --snapshot-repeats 5 --output physx/compiler/anvil/results/pile/softer-intermediate-snapshots
```

The driver now accepts --executable and --impedance; existing defaults preserve the original
hard case. It records the MuJoCo DLL hash alongside the executable and generated XML hashes.
Raw per-solve CSVs, capture settings and full commands are saved in each output directory.
`physx/compiler/anvil/results/pile/softer-comparison-summary.json` joins both new comparisons
and the earlier official hard results, including geometry checks and the driver hash.
