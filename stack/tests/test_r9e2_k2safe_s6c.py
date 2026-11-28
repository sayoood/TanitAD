"""E2-K2SAFE / S1 -- the MAPSEL roll (S6c) of a K2 with a C1-banded map head.  Depends on the PARKED s6c hold (``mapsel_roll_r9_e2.py``): land that first.

The REAL ``own_planes`` / ``_row`` / ``MapSelProxy`` source of the UNCHANGED FlyWheels ``mapsel_roll.py`` is extracted with ``ast`` and exec'd in a synthetic namespace (the file itself cannot be
imported: it parses argv and imports the roll at module level).  Source: env ``R9E2_MAPSEL_ROLL_SRC`` or ``<repo>/FlyWheels/.../2026-10-25-refcv9-mapsel/code/mapsel_roll.py``.
Facts pinned: a banded head emits ``perception.map_hires_logits = None`` (refc_v3.py:2283-2286) => the stock proxy raises at ``lg is None`` (mapsel_roll.py:197-200) BEFORE ``own_planes`` (:202).
RED arms: stock (``require``) dies on banded; patching ONLY ``own_planes`` still dies; ``require`` on a banded K2 refuses with the cause named; ``skip`` keeps the plan + the GT raster and writes zero planes.
"""
from __future__ import annotations

import ast
import os
import sys
import time
import types
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]
R9DIR = ROOT / "ops" / "r9e2_read"
for _p in (str(ROOT), str(ROOT / "scripts"), str(R9DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

SRC = Path(os.environ.get("R9E2_MAPSEL_ROLL_SRC", ROOT.parent / "FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-25-refcv9-mapsel/code/mapsel_roll.py"))
# a skip MARK, not a module-level pytest.skip: the module must import without FlyWheels (the qland import probe counts a module-level skip as a fail)
pytestmark = pytest.mark.skipif(not SRC.is_file(), reason=f"the FlyWheels mapsel_roll.py is not at {SRC} (set R9E2_MAPSEL_ROLL_SRC)")
import mapsel_roll_r9_e2 as MR  # noqa: E402

H, W = 6, 5


class FakeBase:
    out = None

    def __init__(self, model, ctx):
        self.__dict__.update(model=model, ctx=ctx, conds_fed=["nav_true"])

    def __call__(self, frames, nav_cmd=None, v0=None, steps=0, ego_state=None, **kw):
        return type(self).out


def real_module(tmp_path, out):
    """-> the namespace of the REAL own_planes / _row / MapSelProxy, bound to fakes for everything else."""
    tree = ast.parse(SRC.read_text(encoding="utf-8"))
    keep = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name in ("own_planes", "_row")) or (isinstance(n, ast.ClassDef) and n.name == "MapSelProxy")]
    assert [getattr(n, "name") for n in keep] == ["_row", "own_planes", "MapSelProxy"]
    Base = type("Base", (FakeBase,), {"out": out})
    ST = {"W": 3, "t0": time.time(), "records": [], "present": {}, "c": {"calls": 0, "batch_gt1": 0, "navnull_calls": 0, "fan_n_seen": set(), "no_perception": 0, "decide_checked": 0,
                                                                         "decide_mismatch_cells": 0, "gt_windows": 0, "nogt_windows": 0, "rast_bytes": 0, "logit_shapes": set(), "rule_seen": set()}}
    ns = {"np": np, "torch": torch, "os": os, "time": time, "Path": Path, "ST": ST, "A": types.SimpleNamespace(save_every=10**9, budget_probe_windows=0), "RAST": tmp_path / "rast",
          "FIELDS": (("fan", "anchor_traj", np.float32),), "DRV_CODES": (1, 2, 3, 4, 6), "EDGE_CODE": 5, "SIDEWALK_CODE": 7, "_Base": Base, "_window_len": lambda m: 3, "save": lambda *a, **k: None}
    M = types.ModuleType("fake_mapsel_roll")
    M.__dict__.update(ns)
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(SRC), "exec"), M.__dict__)          # the functions' globals ARE the module dict, as in the real roll
    return M


def banded_out():
    return {"traj": torch.ones(1, 4, 3), "sel_idx": torch.tensor([2]), "perception": {"map_hires_logits": None, "map_banded": {"logits": [], "unobs": []}}}


def proxy(M, item=None):
    model = types.SimpleNamespace(_map_hires=types.SimpleNamespace(cfg=types.SimpleNamespace(out_hw=(H, W))), _map_hires_class_weight=None)
    ctx = types.SimpleNamespace(index=(7, 4), item=item if item is not None else {"map_fine": torch.zeros(H, W, dtype=torch.uint8), "map_fine_label": torch.tensor(True)})
    return M.MapSelProxy(model, ctx)


def call(p):
    return p(torch.zeros(1, 3, 2, 2), nav_cmd=torch.tensor([0]), v0=torch.tensor([1.0]))


def test_RED_the_stock_proxy_dies_on_a_banded_output_before_own_planes(tmp_path):
    M = real_module(tmp_path, banded_out())
    with pytest.raises(RuntimeError, match="map_hires_logits.*absent"):
        call(proxy(M))
    assert M.ST["c"]["no_perception"] == 1 and M.ST["c"]["calls"] == 1                      # the raise is the `lg is None` one: own_planes was never reached


def test_RED_patching_only_own_planes_still_dies(tmp_path):
    M = real_module(tmp_path, banded_out())
    M.own_planes = lambda model, lg: tuple(np.zeros((H, W), np.uint8) for _ in range(4))
    with pytest.raises(RuntimeError, match="map_hires_logits.*absent"):
        call(proxy(M))


def test_RED_require_on_a_banded_k2_refuses_naming_the_cause(tmp_path):
    M = real_module(tmp_path, banded_out())
    op = MR.apply_own_planes(M, "require")
    with pytest.raises(SystemExit, match="C1-banded.*--own-planes skip"):
        call(proxy(M))
    assert op["n_banded_refused"] == 1 and op["n_skipped"] == 0


def test_GREEN_skip_keeps_the_plan_and_the_gt_raster_and_writes_zero_planes(tmp_path):
    M = real_module(tmp_path, banded_out())
    op = MR.apply_own_planes(M, "skip")
    gt = torch.randint(0, 8, (H, W), dtype=torch.uint8)
    out = call(proxy(M, {"map_fine": gt, "map_fine_label": torch.tensor(True)}))
    assert torch.equal(out["traj"], banded_out()["traj"]) and out["perception"]["map_hires_logits"].shape == (1, 1, 1, 1)      # the plan is untouched; ONLY the logits slot holds the placeholder
    rec = M.ST["records"][0]
    assert np.array_equal(rec["plan"], np.ones((4, 3), np.float32)) and rec["sel"] == 2 and rec["gt_label"]
    z = np.load(tmp_path / "rast" / "e007_w0006.npz")
    assert sorted(z.files) == sorted(("gt",) + MR.OWN_PLANE_KEYS)
    assert np.array_equal(z["gt"], gt.numpy()) and all(z[k].shape == (H, W) and z[k].dtype == np.uint8 and not z[k].any() for k in MR.OWN_PLANE_KEYS)
    assert op["n_skipped"] == 1 and op["n_placeholder"] == 1 and M.ST["c"]["logit_shapes"] == {(1, 1, 1, 1)}


def test_require_on_an_8_class_k2_is_the_unchanged_roll_and_skip_discards_real_logits_visibly(tmp_path):
    from tanitad.models import map_head_hires as MH
    cw = torch.ones(MH.N_FINE_CLASSES)
    z8 = torch.randn(1, MH.N_FINE_CLASSES, H, W)
    out = {"traj": torch.ones(1, 4, 3), "sel_idx": torch.tensor([0]), "perception": {"map_hires_logits": z8}}
    for mode in ("require", "skip"):
        M = real_module(tmp_path / mode, out)
        orig_op = M.own_planes
        op = MR.apply_own_planes(M, mode)
        p = proxy(M)
        p.__dict__["model"]._map_hires.cfg.decision_rule = "prior_corrected"
        p.__dict__["model"]._map_hires_class_weight = cw
        call(p)
        z = np.load(tmp_path / mode / "rast" / "e007_w0006.npz")
        assert (M.own_planes is orig_op) == (mode == "require")                                  # require leaves the stock function in place
        assert bool(z["code"].any() or z["pdrv"].any()) == (mode == "require")                   # real planes vs zeros
        assert op["n_placeholder"] == 0 and op["n_skipped"] == (1 if mode == "skip" else 0)      # skip on real logits: n_skipped > n_placeholder is the visible discard


def test_mode_parsing_and_flag_plumbing():
    assert "--own-planes" in MR.OWN_FLAGS and MR.own_planes_mode({}) == "require" and MR.own_planes_mode({"--own-planes": "skip"}) == "skip"
    with pytest.raises(SystemExit, match="--own-planes must be one of"):
        MR.own_planes_mode({"--own-planes": "maybe"})
    own, pre, child = MR.split_argv(["--mapsel-code", "D", "--infer-seed", "1", "--own-planes", "skip", "--trace-out", "T", "--", "--x"])
    assert own["--own-planes"] == "skip" and "--own-planes" not in pre and pre == ["--trace-out", "T"] and child == ["--x"]
    with pytest.raises(SystemExit, match="cfg.out_hw"):
        MR.out_hw_of(types.SimpleNamespace())
