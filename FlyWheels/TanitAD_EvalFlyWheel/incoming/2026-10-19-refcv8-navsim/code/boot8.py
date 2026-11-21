"""Bootstrap for the refcv8 NavSim suite (TANITAD VENV) -- ONE place decides which tree is imported.

The code under test is the refcv8 LAUNCH TREE ``C:/lgs/Lfb47a58376`` (``git archive fb47a58`` -- READ-ONLY, never
edited, never given a ``__pycache__``: ``sys.dont_write_bytecode`` is set before any import below), which carries
``stack``, ``taniteval`` and the refcv6 / refcv7 NavSim suites this port imports.  Override with ``TANITAD_REPO`` only
on purpose; the choice is ASSERTED (``tanitad.__file__`` must resolve inside it) and recorded by every caller.

Exports: ``TREE``, ``R7CODE`` (the refcv7 suite's ``code/``), ``R8BATT`` (the refcv8 battery's ``code/`` that holds
``r8_repoint.py``), ``loader()`` (``refcv7_loader`` as RE-POINTED at the refcv8 run by ``r8_repoint.apply``),
``load_retime_stage()`` and ``tree_record()``.

``R8_MUTATION`` (tests only, see ``refcv8_bridge`` section 7) is read here so that a mutation that belongs to the
refcv7 modules this suite imports (``wrong_grid`` / ``drop_mask`` / ``stride16``) is forwarded as ``R7_MUTATION``
BEFORE they are imported.
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
import sys

sys.dont_write_bytecode = True

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
TREE = os.path.abspath(os.environ.get("TANITAD_REPO", "C:/lgs/Lfb47a58376")).replace("\\", "/")
os.environ["TANITAD_REPO"] = TREE                      # frames416._find_repo / refcv7 boot7 read it
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
INC = f"{TREE}/FlyWheels/TanitAD_EvalFlyWheel/incoming"
R6PKG = f"{INC}/2026-09-23-refcv6-standard-tests/navsim"
R7PKG = f"{INC}/2026-09-28-refcv7-standard-tests/navsim"
R7CODE = f"{R7PKG}/code"
R7BAT = f"{INC}/2026-09-28-refcv7-standard-tests/battery/code"

MUTATION = os.environ.get("R8_MUTATION", "")
R7_FORWARDED = ("wrong_grid", "drop_mask", "stride16")
if MUTATION in R7_FORWARDED:
    os.environ["R7_MUTATION"] = MUTATION


def _battery_code() -> str:
    """The refcv8 battery's code dir (holds ``r8_repoint.py`` + ``refcv8_eval_remap_devbox.json``): $R8_BATTERY_CODE,
    else the repo-layout sibling of this package, else the dev-box staging hold.  Read-only."""
    cands = [os.environ.get("R8_BATTERY_CODE", ""),
             os.path.join(os.path.dirname(PKG), "2026-10-15-refcv8-milestone-battery", "code"),
             "C:/Users/Admin/qland/hold_r8batt/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/"
             "2026-10-15-refcv8-milestone-battery/code"]
    for c in cands:
        if c and os.path.isfile(os.path.join(c, "r8_repoint.py")) and \
                os.path.isfile(os.path.join(c, "refcv8_eval_remap_devbox.json")):
            return os.path.abspath(c).replace("\\", "/")
    raise SystemExit("[boot8] r8_repoint.py not found: set R8_BATTERY_CODE to the refcv8 battery's code/ directory")


R8BATT = _battery_code()
for _p in (R7CODE, f"{R6PKG}/code", f"{TREE}/taniteval", f"{TREE}/stack"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import tanitad  # noqa: E402

_tf = os.path.abspath(tanitad.__file__).replace("\\", "/").lower()
if not _tf.startswith((TREE + "/stack/").lower()):
    raise SystemExit(f"[boot8] tanitad imported from {tanitad.__file__}, not from {TREE}/stack -- an editable install "
                     f"or a stale PYTHONPATH is shadowing the tree under test")

_LOADER = None
_R8 = None


def r8_repoint():
    global _R8
    if _R8 is None:
        if R8BATT not in sys.path:
            sys.path.insert(0, R8BATT)
        spec = importlib.util.spec_from_file_location("r8_repoint", os.path.join(R8BATT, "r8_repoint.py"))
        m = importlib.util.module_from_spec(spec)
        sys.modules["r8_repoint"] = m
        spec.loader.exec_module(m)
        _R8 = m
    return _R8


def loader():
    """``tanitad.eval.refcv7_loader`` re-pointed at the refcv8 run (kit paths, remap overrides, the eval-only seam-guard
    patch).  The pose-sync patch is NOT enabled: it wraps ``build_eval_dataset`` and this suite never builds a dataset."""
    global _LOADER
    if _LOADER is None:
        _LOADER = r8_repoint().apply(pose_sync=False)
        if os.path.abspath(str(_LOADER.REPO)).replace("\\", "/").lower() != TREE.lower():
            raise SystemExit(f"[boot8] refcv7_loader.REPO={_LOADER.REPO} is not the tree {TREE}")
    return _LOADER


def sha256_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def load_retime_stage(path: str = ""):
    """``(module, path, sha256)`` of ``retime_stage.py``: ``path`` -> $RETIME_STAGE_FILE -> ``tanitad.retime_stage`` of
    the tree -> the dev-box build overlay.  The launch tree (fb47a58) PREDATES the stage, so a bare run needs one of
    these.  The module is loaded BY PATH and its sha256 is recorded; there is no default gains TABLE anywhere here."""
    cands = [path, os.environ.get("RETIME_STAGE_FILE", ""), f"{TREE}/stack/tanitad/retime_stage.py",
             "C:/lgs/retime_stage_scratch/stack/tanitad/retime_stage.py"]
    for c in cands:
        if c and os.path.isfile(c):
            spec = importlib.util.spec_from_file_location("_retime_stage_r8", c)
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            return m, os.path.abspath(c).replace("\\", "/"), sha256_file(c)
    raise SystemExit("[boot8] retime_stage.py not found: pass --retime-stage-file or set RETIME_STAGE_FILE")


def tree_record() -> dict:
    commit = None
    p = os.path.join(TREE, "TREE_COMMIT.txt")
    if os.path.exists(p):
        commit = open(p, encoding="utf-8").read().strip()
    return {"tree": TREE, "tree_commit": commit, "tanitad": tanitad.__file__, "r7_suite": R7CODE,
            "r8_battery_code": R8BATT, "r8_repoint_sha256": sha256_file(os.path.join(R8BATT, "r8_repoint.py")),
            "r8_remap_sha256": sha256_file(os.path.join(R8BATT, "refcv8_eval_remap_devbox.json"))}
