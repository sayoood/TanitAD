"""SHIM, not scipy. See ``r9e1_read.sh``: Thor's training venv has numpy 2.5 and torch but NO scipy (the system python3's scipy is
built for numpy 1.x and cannot share that venv). ``r9_e1_eval.nearfield_per_file`` needs exactly one thing from scipy,
``scipy.ndimage.binary_dilation(mask, iterations=n)``; this package provides that and nothing else, and it is put on PYTHONPATH
ONLY when the real scipy cannot be imported (never ahead of a working scipy). Held equal to scipy by test_r9_e1_floor.py."""
__version__ = "0+r9e1.shim"
IS_SHIM = True
