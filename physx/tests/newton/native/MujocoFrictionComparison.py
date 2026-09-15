"""Sliding reference using the official MuJoCo DLL and the pallet contact convention."""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'physx/compiler/newton/vendor/python'))
import mujoco as mj
import numpy as np


def run_case(case, speed, steps, directory):
    xml = '''<mujoco model="Sliding reference">
    <option timestep=".01" gravity="0 -9.81 0" integrator="Euler" solver="Newton"
      cone="pyramidal" jacobian="sparse" iterations="100" tolerance="1e-8"/>
    <default><geom friction=".5 0 0" condim="3" margin=".001" gap="0"
      solref=".02 1" solimp=".9999 .9999 .001 .5 2"/></default>
    <worldbody>
      <geom name="floor" type="box" pos="0 -.1 0" size="100 .1 10"/>
      <body name="box" pos="0 .5 0">
        <joint type="slide" axis="1 0 0"/><joint type="slide" axis="0 1 0"/>
        <joint type="slide" axis="0 0 1"/>
        <geom type="box" size=".5 .5 .5" mass="1"/>
      </body>
    </worldbody></mujoco>'''
    model = mj.MjModel.from_xml_string(xml)
    data = mj.MjData(model)
    if case == 'flat':
        data.qvel[0] = speed
    rows = []
    for step in range(steps):
        mj.mj_step1(model, data)
        data.efc_aref[:] -= data.efc_KBIP[:, 0] * data.efc_KBIP[:, 2] * data.efc_margin
        if case == 'conveyor':
            for contact in data.contact:
                if contact.efc_address < 0:
                    continue
                belt_speed = speed if contact.geom[0] == 0 else -speed
                for tangent in range(2):
                    for sign in range(2):
                        row = contact.efc_address + 2 * tangent + sign
                        target = belt_speed * (contact.frame[0] + (1 if sign == 0 else -1) *
                            contact.friction[tangent] * contact.frame[3 * (tangent + 1)])
                        data.efc_vel[row] -= target
                        data.efc_aref[row] += data.efc_KBIP[row, 1] * target
        mj.mj_fwdActuation(model, data)
        mj.mj_fwdAcceleration(model, data)
        mj.mj_fwdConstraint(model, data)
        impulse = model.opt.timestep * data.qfrc_constraint.copy()
        mj.mj_Euler(model, data)
        if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)) or any(data.warning.number):
            raise RuntimeError('Invalid MuJoCo state or warning')
        rows.append(dict(step=step + 1, time=float(data.time), x=float(data.qpos[0]),
            y=.5 + float(data.qpos[1]), rise=float(data.qpos[1]), vx=float(data.qvel[0]),
            vy=float(data.qvel[1]), normal_impulse=float(impulse[1]), tangent_impulse=float(impulse[0]),
            contacts=int(data.ncon), iterations=int(max(data.solver_niter))))
    path = directory / f'mujoco-{case}-{speed:g}.csv'
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return dict(case=case, speed=speed, steps=steps, first_rise=rows[0]['rise'],
        max_rise=max(row['rise'] for row in rows), max_abs_vy=max(abs(row['vy']) for row in rows),
        travel=rows[-1]['x'], final_speed=rows[-1]['vx'], final_rise=rows[-1]['rise'],
        total_normal_impulse=sum(row['normal_impulse'] for row in rows),
        total_tangent_impulse=sum(row['tangent_impulse'] for row in rows), trajectory=str(path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--speeds', type=float, nargs='+', default=[.2, 1, 2])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = [run_case(case, speed, args.steps, args.output) for case in ['flat', 'conveyor'] for speed in args.speeds]
    report = dict(version=mj.__version__, package=str(Path(mj.__file__).resolve()),
        dll_sha256=hashlib.sha256((Path(mj.__file__).parent / 'mujoco.dll').read_bytes()).hexdigest(),
        solver='Official MuJoCo Newton, pyramidal friction', timestep=.01, iterations=100,
        tolerance=1e-8, friction=.5, mass=1, half_extents=[.5, .5, .5],
        margin=.001, rest_distance=0, solimp=[.9999, .9999, .001, .5, 2], solref=[.02, 1],
        rotation='Locked using three translational joints',
        pipeline='Same manual stage sequence and margin/rest correction as the maintained pallet harness',
        results=rows)
    (args.output / 'mujoco-sliding.json').write_text(json.dumps(report, indent=2))
    for row in rows:
        print(json.dumps(row))


if __name__ == '__main__':
    main()