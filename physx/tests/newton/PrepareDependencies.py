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
MUJOCO_VERSION = "3.13.0"
MUJOCO_COMMIT = "123347c0eeab7e13c8da0828ab593bbd95bcf335"


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
    source = VENDOR / f"mujoco-source-{MUJOCO_VERSION}"
    unpack(f"https://github.com/google-deepmind/mujoco/archive/{MUJOCO_COMMIT}.zip", source)
    build = VENDOR / f"mujoco-build-{MUJOCO_VERSION}"
    command = ["cmake", "-S", str(source), "-B", str(build), "-G", "Visual Studio 17 2022", "-A", "x64",
               "-DMUJOCO_BUILD_TESTS=OFF", "-DMUJOCO_BUILD_EXAMPLES=OFF", "-DMUJOCO_BUILD_SIMULATE=OFF"]
    for name in ("ccd", "lodepng", "marchingcubecpp", "qhull", "tinyobjloader", "tinyxml2", "trianglemeshdistance"):
        local = VENDOR / (name + "-src")
        if local.exists():
            command.append(f"-DFETCHCONTENT_SOURCE_DIR_{name.upper()}={local}")
    subprocess.run(command, check=True)
    subprocess.run(["cmake", "--build", str(build), "--config", "Release", "--target", "mujoco", "--parallel", "4"], check=True)
    destination = VENDOR / "mujoco"
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir()
    shutil.copytree(source / "include", destination / "include")
    shutil.copy2(source / "LICENSE", destination / "LICENSE")
    shutil.copy2(build / "bin/Release/mujoco.dll", destination / "mujoco.dll")
    shutil.copy2(build / "lib/Release/mujoco.lib", destination / "mujoco.lib")
    python = VENDOR / "python"
    for package in python.glob("mujoco*"):
        shutil.rmtree(package) if package.is_dir() else package.unlink()
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "--target", str(python),
                    f"mujoco=={MUJOCO_VERSION}", "numpy==1.26.4"], check=True)


if __name__ == "__main__":
    main()
