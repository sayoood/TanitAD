"""Bootstrap for the refcv7 NavSim suite (TANITAD VENV) — ONE place decides which tree is imported.

The code under test is the CLEAN TREE ``C:/Users/Admin/ev7nav`` = ``git archive 0c444082`` of
``stack taniteval tools`` + the NavSim predecessor packages (E2 / W3 / W7 / E1 / refcv6 suite), never
a working copy (the brief; CLAUDE.md "the editable install points at another tree"). Override with
``TANITAD_REPO`` only on purpose; the choice is ASSERTED here (``tanitad.__file__`` must resolve
inside it) and recorded by every caller.

Exports: ``TREE``, ``R6PKG`` (the refcv6 suite inside the tree), ``F4`` (frames416), ``R6``
(refcv6_bridge), ``RIG6`` (rig6), ``loader()`` (refcv7_loader, by path).
"""
from __future__ import annotations

import importlib.util
import os
import sys

TREE = os.path.abspath(os.environ.get("TANITAD_REPO", "C:/Users/Admin/ev7nav")).replace("\\", "/")
os.environ["TANITAD_REPO"] = TREE                      # frames416._find_repo reads it
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
R6PKG = (f"{TREE}/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-23-refcv6-standard-tests/navsim")
for _p in (f"{R6PKG}/code", f"{TREE}/taniteval", f"{TREE}/stack"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import tanitad  # noqa: E402

_tf = os.path.abspath(tanitad.__file__).replace("\\", "/").lower()
if not _tf.startswith((TREE + "/stack/").lower()):
    raise SystemExit(f"⛔ tanitad imported from {tanitad.__file__}, not from {TREE}/stack — an "
                     f"editable install or a stale PYTHONPATH is shadowing the tree under test")

import frames416 as F4  # noqa: E402  (refcv6 suite, from the clean tree)
import refcv6_bridge as R6  # noqa: E402
import rig6 as RIG6  # noqa: E402

if os.path.abspath(F4.REPO).replace("\\", "/").lower() != TREE.lower():
    raise SystemExit(f"⛔ frames416 resolved REPO={F4.REPO}, not {TREE}")

_LOADER = None


def loader():
    """``stack/tanitad/eval/refcv7_loader.py`` of THE TREE (its REPO = the tree it lives in)."""
    global _LOADER
    if _LOADER is None:
        p = f"{TREE}/stack/tanitad/eval/refcv7_loader.py"
        spec = importlib.util.spec_from_file_location("refcv7_loader", p)
        m = importlib.util.module_from_spec(spec)
        sys.modules["refcv7_loader"] = m
        spec.loader.exec_module(m)
        if os.path.abspath(str(m.REPO)).replace("\\", "/").lower() != TREE.lower():
            raise SystemExit(f"⛔ refcv7_loader.REPO={m.REPO} is not the tree {TREE}")
        _LOADER = m
    return _LOADER


def tree_record() -> dict:
    """What was imported, for every manifest."""
    commit = None
    p = os.path.join(TREE, "TREE_COMMIT.txt")
    if os.path.exists(p):
        commit = open(p, encoding="utf-8").read().strip()
    return {"tree": TREE, "tree_commit": commit, "tanitad": tanitad.__file__,
            "refcv6_suite": R6PKG, "frames416": F4.__file__, "refcv6_bridge": R6.__file__}
