"""n5_env.py - bootstrap for the N5-L1 instruments. CPU only. Import this BEFORE numpy / n4_*.
Sets all BLAS thread vars to 1 and CUDA off (a previous agent measured a 2.8e-6 label drift otherwise), then puts the PINNED run tree
(F:/tanitad_scratch/n4nt/run_a1, never a copy) and its devkit on sys.path in the chain order. Nothing in the pinned tree is modified;
sys.dont_write_bytecode keeps it clean."""
import os
import sys

sys.dont_write_bytecode = True
RUN = os.environ.get("N4_RUN", "F:/tanitad_scratch/n4nt/run_a1")
DEVKIT = "F:/Archive/devbox-C/navsim/navsim-3e8291bfa89ff247231e0227778840cd0a036896"
NAVSIM_ROOT = "F:/Archive/devbox-C/navsim"
CACHE_ROOT = "F:/Archive/devbox-C/navsim/exp/w3_navtest_v1/metric_cache_navtest"
MAPS = "E:/Projects/TanitAD/data/nuplan-maps/nuplan-maps-v1.0"
os.environ.update({"CUDA_VISIBLE_DEVICES": "-1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                   "REFE_SIM_HZ": "10", "NUPLAN_MAPS_ROOT": MAPS, "PYTHONIOENCODING": "utf-8", "TANITAD_NAVSIM_ROOT": NAVSIM_ROOT,
                   "PYTHONDONTWRITEBYTECODE": "1", "N4_W": RUN + "/w/pk", "N4_P5R": RUN + "/w/p5r"})
for _p in (RUN + "/w/pk/refe", DEVKIT):
    while _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)
if RUN + "/n4" not in sys.path:
    sys.path.insert(0, RUN + "/n4")
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.append(HERE)
