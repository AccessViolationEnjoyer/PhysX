"""Compare Newton against MuJoCo 3.3.7 on identical pyramidal equations.

Frozen warm starts come from a preceding solve at 99% of the measured load.
Collision detection, integration and fixture parsing are outside solve timings.
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BUILD = ROOT / 'physx/compiler/newton'
OUT = BUILD / 'results'
EXE = BUILD / 'build/Release/NewtonBenchmark.exe'
sys.path.insert(0, str(BUILD / 'vendor/python'))
import mujoco as mj
import numpy as np

def make_problem(case, dimensions=None, mass_ratio=1.0, friction=.5, disturbance=1.0, contact_points=4):
    side = 5 if '125' in case else 3 if '27' in case else 1
    height = 2 if case == 'mass-ratio' else side
    width, height, depth = dimensions if dimensions is not None else (side, height, side)
    poses = [(x, y + .5, z) for y in range(height) for z in range(depth) for x in range(width)]
    masses = np.geomspace(1.0, mass_ratio, len(poses))
    if case == 'mass-ratio':
        masses[-1] = 1.0 / float(np.float32(.0001))
    bodies = ''.join(f'<body pos="{x} {y} {z}"><freejoint/><geom type="box" size=".5 .5 .5" mass="{mass}" contype="0" conaffinity="0"/></body>'
                     for (x, y, z), mass in zip(poses, masses))
    xml = ('<mujoco><size memory="256M"/><option timestep=".01" gravity="0 -9.81 0" '
           'cone="pyramidal" jacobian="sparse" solver="Newton" iterations="100" tolerance="0" '
           'ls_iterations="50" ls_tolerance=".01"><flag warmstart="disable" island="disable"/></option>'
           '<worldbody><geom type="plane" size="20 20 .1" contype="0" conaffinity="0"/>' + bodies + '</worldbody></mujoco>')
    model = mj.MjModel.from_xml_string(xml)
    data = mj.MjData(model)
    for i, (x, y, z) in enumerate(poses):
        if 'disturbed' in case:
            data.qvel[6*i] = disturbance*(.3*(y-.5) + .03*np.sin(i))
            data.qvel[6*i+2] = disturbance*(.1*x - .04*z)
        if case == 'slide':
            data.qvel[6*i] = 1.0
    mj.mj_forward(model, data)
    pairs = []
    for i, (x, y, z) in enumerate(poses):
        faces = [(i-width*depth if y > .5 else -1, np.array([0., 1., 0.]), np.array([1., 0., 0.]))]
        if x:
            faces.append((i-1, np.array([1., 0., 0.]), np.array([0., 1., 0.])))
        if z:
            faces.append((i-width, np.array([0., 0., 1.]), np.array([1., 0., 0.])))
        for other, normal, tangent in faces:
            second = np.cross(normal, tangent)
            corners = [(0,0)] if contact_points == 1 else [(-1,-1),(1,1)] if contact_points == 2 else [(-1,-1),(-1,1),(1,-1),(1,1)]
            for a,b in corners:
                c = mj.MjContact()
                c.dim = 3
                c.includemargin = 1e-9
                c.pos[:] = np.array([x, y, z]) - .5*normal + .5*(a*tangent+b*second)
                c.frame[:] = np.concatenate([normal, tangent, second])
                c.friction[:] = [friction, friction, 0., 0., 0.]
                c.solref[:] = [.02, 1.]
                c.solimp[:] = [.9, .95, .001, .5, 2.]
                c.geom[:] = [other+1, i+1]
                c.flex[:] = -1
                c.elem[:] = -1
                c.vert[:] = -1
                assert mj.mj_addContact(model, data, c) == 0
                pairs.append((i, other))
    mj.mj_makeConstraint(model, data)
    mj.mj_projectConstraint(model, data)
    mj.mj_fwdVelocity(model, data)
    mj.mj_fwdActuation(model, data)
    mj.mj_fwdAcceleration(model, data)
    mj.mj_referenceConstraint(model, data)
    assert data.nefc == 4*len(pairs)
    jacobian = np.zeros((data.nefc, model.nv))
    for row in range(data.nefc):
        start = data.efc_J_rowadr[row]
        end = start + data.efc_J_rownnz[row]
        jacobian[row, data.efc_J_colind[start:end]] = data.efc_J[start:end]
    mass = np.zeros((model.nv, model.nv))
    mj.mj_fullM(model, mass, data.qM)
    diagonal = mass.diagonal().copy()
    assert np.max(np.abs(mass-np.diag(diagonal))) == 0
    scaled_jacobian = jacobian / np.sqrt(diagonal)
    free = model.opt.timestep*(jacobian @ data.qacc_smooth - data.efc_aref)
    return model, data, pairs, masses, diagonal, scaled_jacobian, free


def prepare(case,scale):
    model,data,pairs,masses,diagonal,_,_=make_problem(case)
    J = np.zeros((data.nefc,model.nv))
    for row in range(data.nefc):
        begin=data.efc_J_rowadr[row]
        end=begin+data.efc_J_rownnz[row]
        J[row,data.efc_J_colind[begin:end]]=data.efc_J[begin:end]
    scaled=J/np.sqrt(diagonal)
    if scale!='native':
        response=np.sum(scaled**2,axis=1).reshape(-1,4).mean(axis=1)
        data.efc_R[:]=np.repeat(float(scale)*response,4)
        data.efc_D[:]=1/data.efc_R
        data.efc_aref[:]=-(J@data.qvel)/model.opt.timestep
    free=model.opt.timestep*(J@data.qacc_smooth-data.efc_aref)
    return model,data,pairs,masses,diagonal,scaled,free



def export(path,args):
    model,data,pairs,masses,diagonal,J,free=args
    lines=[f'{len(masses)} {data.nefc} {model.opt.timestep:.17g}', ' '.join(format(x,'.17g') for x in 1/masses)]
    # Preserve the prepared-file format. The C++ loader packs each nonnegative
    # edge into one solver row and discards the two unused components.
    for edge in range(data.nefc):
        body,other=pairs[edge//4]
        values=[body,other,0,0,0,free[edge],1,1,data.efc_R[edge]]
        for end in [body,other]:
            block=np.zeros((3,6))
            if end>=0:
                block[2,:]=J[edge,6*end:6*end+6]
            values.extend(block.ravel())
        lines.append(' '.join(format(x,'.17g') for x in values))
    lines.append('mass_diagonal '+' '.join(format(x,'.17g') for x in diagonal))
    path.write_text('\n'.join(lines)+'\n')


def native_solve(args,seed,tolerance,repeats=7):
    model,data,_,_,diagonal,J,free=args
    model.opt.tolerance=tolerance
    model.opt.disableflags &= ~int(mj.mjtDisableBit.mjDSBL_WARMSTART)
    timings=[]
    for i in range(repeats+1):
        data.qacc_warmstart[:]=seed
        start=time.perf_counter_ns()
        mj.mj_fwdConstraint(model,data)
        dt=(time.perf_counter_ns()-start)*1e-6
        if i:
            timings.append(dt)
    impulse=model.opt.timestep*data.efc_force.copy()
    primal=model.opt.timestep*np.sqrt(diagonal)*(data.qacc-data.qacc_smooth)
    gradient=primal-J.T@impulse
    velocity=J@primal+free
    projected=np.maximum(-velocity/data.efc_R,0.)
    count=int(data.solver_niter[0])
    return dict(median_ms=statistics.median(timings),timings_ms=timings,tolerance=tolerance,
                iterations=count,gradient=float(np.max(np.abs(gradient))),
                projection_error=float(np.max(np.abs(impulse-projected))),
                last_scaled_gradient=float(data.solver[count-1].gradient) if count else None,
                hessian_nonzeros=int(data.solver_nnz[0])),impulse,primal



def comparison_error(primal,impulse,reference_primal,reference_impulse,diagonal,R):
    velocity = (primal-reference_primal)/np.sqrt(diagonal)
    reference_cost=.5*(reference_primal@reference_primal + np.dot(R,reference_impulse**2))
    cost=.5*(primal@primal+np.dot(R,impulse**2))
    return dict(scaled_velocity_error=float(np.max(np.abs(primal-reference_primal))),
                linear_velocity_error=float(np.max(np.abs(velocity.reshape(-1,6)[:,:3]))),
                angular_velocity_error=float(np.max(np.abs(velocity.reshape(-1,6)[:,3:]))),
                impulse_error=float(np.max(np.abs(impulse-reference_impulse))),
                relative_impulse_error=float(np.max(np.abs(impulse-reference_impulse))/max(1e-30,np.max(np.abs(reference_impulse)))),
                cost_error=float(cost-reference_cost))



def prototype(path, seed_path, repeats=7, check=False, islands=1, threads=1, label=""):

    tag = path.stem + ('-check' if check else '') + f'-{islands}islands' + label
    solution = OUT / (tag + '-solution.txt')
    command = [str(EXE), str(path), str(seed_path) if seed_path else '-', str(solution),
               str(repeats + 1), '100', str(int(check)), str(islands), str(threads)]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=240)
    (OUT / (tag + '.log')).write_text(completed.stdout + completed.stderr)
    if completed.returncode:
        raise RuntimeError(completed.stdout + completed.stderr)
    rows = [dict((k, float(v)) for k, v in (f.split('=', 1) for f in line.split(',')[1:]))
            for line in completed.stdout.splitlines() if line.startswith('RESULT,')]
    assert len(rows) == repeats + 1
    return dict(median_ms=statistics.median(r['ms'] for r in rows[1:]),
                wall_ms=statistics.median(r['wall_ms'] for r in rows[1:]),
                results=rows[1:], command=command), np.loadtxt(solution)


def run_case(case, scale, warm):
    args = prepare(case, scale)
    model, data, pairs, masses, diagonal, J, free = args
    path = OUT / f'{case}-{scale}-warm{warm}.txt'
    export(path, args)
    seed_path = None
    seed = data.qacc_smooth.copy()
    if warm:
        reference = data.efc_aref.copy()
        data.efc_aref[:] = reference + .01*free/model.opt.timestep
        model.opt.tolerance = 1e-12
        mj.mj_fwdConstraint(model, data)
        seed = data.qacc.copy()
        padded = np.zeros((data.nefc, 3))
        padded[:, 2] = model.opt.timestep*data.efc_force
        primal = model.opt.timestep*np.sqrt(diagonal)*(seed-data.qacc_smooth)
        seed_path = OUT / f'{path.stem}-seed.txt'
        np.savetxt(seed_path, np.concatenate([padded.ravel(), primal]), fmt='%.17g')
        data.efc_aref[:] = reference
    strict, ref_impulse, ref_primal = native_solve(args, seed, 1e-12)
    native, _, _ = native_solve(args, seed, 1e-8)
    ours, values = prototype(path, seed_path)
    impulse = values[:3*data.nefc].reshape(-1, 3)[:, 2]
    primal = values[3*data.nefc:]
    assert np.max(np.abs(values[:3*data.nefc].reshape(-1, 3)[:, :2])) == 0
    ours.update(comparison_error(primal, impulse, ref_primal, ref_impulse, diagonal, data.efc_R))
    assert ours['scaled_velocity_error'] < 1e-7, ours
    check, _ = prototype(path, seed_path, repeats=1, check=True)
    record = dict(case=case, scale=scale, warm=warm, bodies=len(masses), contacts=len(pairs),
                  active_rows=data.nefc, native_version=mj.__version__, native=native,
                  native_strict=strict, newton=ours, factor_check=check)
    print(case, scale, warm, 'native', round(native['median_ms'], 4),
          'Newton', round(ours['median_ms'], 4), 'velocity error', ours['linear_velocity_error'], flush=True)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--quick', action='store_true')
    options = parser.parse_args()
    OUT = options.output
    OUT.mkdir(parents=True, exist_ok=True)
    fixtures = [('rest', 'native', True), ('grid27-disturbed', 'native', True),
                ('grid125-rest', '.01', True), ('grid125-disturbed', '.01', True),
                ('grid125-disturbed', '.0001', True), ('grid125-disturbed', '.0001', False),
                ('mass-ratio', '1e-9', True)]
    records = []
    for case, scale, warm in fixtures[:2] if options.quick else fixtures:
        records.append(run_case(case, scale, warm))
        (OUT / 'comparison.json').write_text(json.dumps(records, indent=2))
