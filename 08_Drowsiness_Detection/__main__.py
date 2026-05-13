"""Run drowsiness detection when the folder is passed to Python.

Correct usage from repo root:
  python 08_Drowsiness_Detection

Or run the script file directly:
  python 08_Drowsiness_Detection\\drowsiness.py
"""
import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).resolve().parent / "drowsiness.py"), run_name="__main__")
