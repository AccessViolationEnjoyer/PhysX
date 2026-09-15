"""Compatibility entry point for the maintained production kernel generator."""
from pathlib import Path
import runpy
runpy.run_path(str(Path(__file__).resolve().parents[2] / "source/lowleveldynamics/src/newton/core/PrepareStorageKernels.py"), run_name="__main__")
