"""n7_env.py - bootstrap for the N7 generator development (navtrain ONLY, CPU only). Import BEFORE numpy.
Pinned TF-N4 A1 run tree (read only) + devkit on sys.path; the L1 module is imported from the banked n5_l1 hold (read only)."""
import os
import sys

sys.dont_write_bytecode = True
RUN = "F:/tanitad_scratch/n4nt/run_a1"
DEVKIT = "F:/Archive/devbox-C/navsim/navsim-3e8291bfa89ff247231e0227778840cd0a036896"
L1_DIR = "E:/Projects/TanitAD-artifacts/scorer_gym_20261107/teacher_fidelity/n5_l1"
os.environ.update({"CUDA_VISIBLE_DEVICES": "-1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "REFE_SIM_HZ": "10",
                   "NUPLAN_MAPS_ROOT": "E:/Projects/TanitAD/data/nuplan-maps/nuplan-maps-v1.0", "PYTHONIOENCODING": "utf-8",
                   "TANITAD_NAVSIM_ROOT": "F:/Archive/devbox-C/navsim", "PYTHONDONTWRITEBYTECODE": "1",
                   "N4_W": RUN + "/w/pk", "N4_P5R": RUN + "/w/p5r"})
for _p in (L1_DIR, RUN + "/n4", RUN + "/w/pk/refe", DEVKIT):
    while _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.append(HERE)
