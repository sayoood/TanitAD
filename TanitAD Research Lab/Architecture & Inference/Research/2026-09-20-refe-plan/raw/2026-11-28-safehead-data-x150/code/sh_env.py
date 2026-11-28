"""Process environments for the SAFE-HEAD tools (same settings the TRK-SCORE chain and stage2_sft use; D: -> E: as trk_run_seam does)."""
from __future__ import annotations

import os

import sh_lib as L

NAVSIM_ROOT = "F:/Archive/devbox-C/navsim"


def denv(extra=None):
    """driverl-eval venv: model forward + nuPlan scenario builder (eval_checkpoint.env_driverl, D: -> E:)."""
    DZ = "C:/Users/Admin/dz/DriveZero/DriveRL"
    e = dict(os.environ)
    e.update({"DZ_ROOT": DZ, "NUPLAN_DATA_ROOT": "E:/Projects/TanitAD/data/nuplan", "NUPLAN_MAPS_ROOT": L.MAPS, "DRIVERL_EVAL_ROOT": DZ,
              "DRIVERL_EVAL_NUPLAN_ROOT": f"{DZ}/nuplan-devkit", "DRIVERL_EVAL_CONFIG_PATH": f"{DZ}/release/configs/driverl_teacher.yaml",
              "DRIVERL_EVAL_CHECKPOINT_PATH": f"{DZ}/release/checkpoints/checkpoint_2400.pt", "PYTHONPATH": f"{DZ}/nuplan-devkit;{DZ}/src",
              "REFE_BACKBONE_ROOT": "E:/Projects/TanitAD/data/backbones", "REFE_SIM_HZ": "10", "OMP_NUM_THREADS": "4", "PYTHONIOENCODING": "utf-8",
              "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1", "TANITAD_NAVSIM_ROOT": NAVSIM_ROOT, "TANITAD_REFE_PKG": L.PK, "NUPLAN_MAPS_ROOT_": L.MAPS})
    e.pop("CUDA_VISIBLE_DEVICES", None)
    if extra:
        e.update(extra)
    return e


def nenv(extra=None):
    """navsim venv: CPU labels (trk_chain.sh's ctrl invocation: PYTHONPATH NAV;refe;eval/instruments, GPU hidden; REFE_SIM_HZ unset = the programme labeller's 20 Hz default, p6_pod.sh)."""
    e = dict(os.environ)
    e.update({"CUDA_VISIBLE_DEVICES": "-1", "PYTHONPATH": f"{L.NAV};{L.PK}/refe;{L.PK}/eval/instruments", "TANITAD_NAVSIM_ROOT": NAVSIM_ROOT, "NUPLAN_MAPS_ROOT": L.MAPS,
              "NUPLAN_DATA_ROOT": "E:/Projects/TanitAD/data/nuplan", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
              "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1"})
    if extra:
        e.update(extra)
    return e
