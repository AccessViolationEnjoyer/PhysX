"""Render recorded PhysX or MuJoCo body poses without re-running physics."""
import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[4]
SOURCE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'physx/compiler/anvil/vendor/python'))
import mujoco as mj
import numpy as np


def make_camera(xml, lane):
    target = np.array([-1.0, 1.1, 0.0 if lane is None else (lane - 2) * 2.2])
    position = target + (np.array([5.0, 8.0, 12.0]) if lane is None else np.array([2.0, 1.2, 3.0]))
    backward = position - target
    backward /= np.linalg.norm(backward)
    right = np.cross([0, 1, 0], backward)
    right /= np.linalg.norm(right)
    up = np.cross(backward, right)
    world = xml.find('worldbody')
    ET.SubElement(world, 'camera', name='replay', pos=' '.join(map(str, position)),
                  xyaxes=' '.join(map(str, np.concatenate((right, up)))))
    if lane is not None:
        for geom in world.findall('geom'):
            if geom.attrib['name'] != f'belt{lane + 1}':
                geom.set('group', '1')
        for body in world.findall('body'):
            if not body.attrib['name'].startswith(f'lane{lane + 1}_'):
                body.find('geom').set('group', '1')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('prefix', type=Path, help='Prefix of the recorded -poses.csv and -steps.csv files')
    parser.add_argument('output', type=Path, help='PNG of the final frame, or MP4 replay')
    parser.add_argument('--lane', type=int, choices=range(5))
    args = parser.parse_args()
    with Path(str(args.prefix) + '-steps.csv').open() as stream:
        steps = list(csv.DictReader(stream))
    timestep = float(steps[0]['time'])
    poses = np.loadtxt(str(args.prefix) + '-poses.csv', delimiter=',', skiprows=1)
    body_count = int(poses[:, 1].max()) + 1
    frames = poses.reshape(-1, body_count, poses.shape[1])
    xml = ET.parse(SOURCE / 'PalletConveyor.xml').getroot()
    make_camera(xml, args.lane)
    model = mj.MjModel.from_xml_string(ET.tostring(xml, encoding='unicode'))
    data = mj.MjData(model)
    camera_x = model.cam_pos[0, 0]
    option = mj.MjvOption()
    option.geomgroup[1] = 0
    renderer = mj.Renderer(model, height=720, width=1280)
    movie = args.output.suffix.lower() == '.mp4'
    pipe = None
    if movie:
        ffmpeg = shutil.which('ffmpeg')
        if not ffmpeg:
            raise RuntimeError('ffmpeg is required to write MP4')
        pipe = subprocess.Popen([ffmpeg, '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                                 '-s', '1280x720', '-r', '10', '-i', '-', '-an', '-c:v', 'libx264',
                                 '-crf', '20', '-pix_fmt', 'yuv420p', str(args.output)], stdin=subprocess.PIPE)
        times = np.arange(.1, float(steps[-1]['time']) + .001, .1)
        indices = np.minimum(np.searchsorted(frames[:, 0, 0] * timestep, times), len(frames) - 1)
    else:
        indices = [len(frames) - 1]
    for frame_index in indices:
        frame = frames[frame_index]
        if args.lane is not None:
            model.cam_pos[0, 0] = camera_x + frame[args.lane * 65, 2] + 1.0
        for i in range(body_count):
            data.qpos[7*i:7*i+3] = frame[i, 2:5]
            mj.mju_mat2Quat(data.qpos[7*i+3:7*i+7], frame[i, 5:14])
        mj.mj_kinematics(model, data)
        mj.mj_camlight(model, data)
        renderer.update_scene(data, camera='replay', scene_option=option)
        image = renderer.render()
        if movie:
            pipe.stdin.write(image.tobytes())
        else:
            from PIL import Image
            Image.fromarray(image).save(args.output)
    renderer.close()
    if pipe:
        pipe.stdin.close()
        if pipe.wait():
            raise RuntimeError('ffmpeg failed')


if __name__ == '__main__':
    main()
