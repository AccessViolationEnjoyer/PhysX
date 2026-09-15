"""Prepare pinned dependencies for the standalone Windows Newton benchmark."""
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
VENDOR = ROOT / "physx/compiler/newton/vendor"
MUJOCO_COMMIT = "f1d45bd5422c74beddfb0d1deb590a02583d21de"


def unpack(url, target):
    if target.exists():
        return
    with tempfile.TemporaryDirectory() as temporary:
        archive = Path(temporary) / "source.zip"
        urllib.request.urlretrieve(url, archive)
        with zipfile.ZipFile(archive) as source:
            source.extractall(Path(temporary) / "source")
        folders = list((Path(temporary) / "source").iterdir())
        assert len(folders) == 1 and folders[0].is_dir()
        shutil.copytree(folders[0], target)


def main():
    VENDOR.mkdir(parents=True, exist_ok=True)
    unpack("https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip", VENDOR / "eigen")
    source = VENDOR / "mujoco-source"
    unpack(f"https://github.com/google-deepmind/mujoco/archive/{MUJOCO_COMMIT}.zip", source)
    build = VENDOR / "mujoco-build"
    command = ["cmake", "-S", str(source), "-B", str(build), "-G", "Visual Studio 17 2022", "-A", "x64",
               "-DMUJOCO_BUILD_TESTS=OFF", "-DMUJOCO_BUILD_EXAMPLES=OFF", "-DMUJOCO_BUILD_SIMULATE=OFF"]
    for name in ("ccd", "lodepng", "marchingcubecpp", "qhull", "tinyobjloader", "tinyxml2", "trianglemeshdistance"):
        local = VENDOR / (name + "-src")
        if local.exists():
            command.append(f"-DFETCHCONTENT_SOURCE_DIR_{name.upper()}={local}")
    subprocess.run(command, check=True)
    subprocess.run(["cmake", "--build", str(build), "--config", "Release", "--target", "mujoco", "--parallel", "4"], check=True)
    destination = VENDOR / "mujoco"
    destination.mkdir(exist_ok=True)
    shutil.copytree(source / "include", destination / "include", dirs_exist_ok=True)
    shutil.copy2(source / "LICENSE", destination / "LICENSE")
    shutil.copy2(build / "bin/Release/mujoco.dll", destination / "mujoco.dll")
    shutil.copy2(build / "lib/Release/mujoco.lib", destination / "mujoco.lib")
    if not (VENDOR / "python/mujoco").exists():
        subprocess.run([sys.executable, "-m", "pip", "install", "--target", str(VENDOR / "python"),
                        "mujoco==3.3.7", "numpy==1.26.4"], check=True)


if __name__ == "__main__":
    main()
