"""Tests for r8_repoint.py -- the refcv8 re-pointing of the refcv7 battery.

Run (dev box):  PYTHONPATH="C:/lgs/Lfb47a58376/stack;C:/lgs/Lfb47a58376/taniteval" \
                python -m pytest -q -p no:cacheprovider test_r8_repoint.py

Every expected value is a LITERAL (hand-worked in stack/tests/test_pose_sync.py: a = 2 m/s^2, delta 16.5 ms =>
v0 moves by 0.033 m/s), never an expression over the code under test.  Each guard ships a DELIBERATE-REGRESSION arm
(named `test_REGRESSION_*`) that reintroduces the real defect and must go RED on the very check that is green with the
fix: (1) the loader without the pose-sync patch; (2) a remap file missing the three flags the launch gate never needed.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r8_repoint as R  # noqa: E402

TREE = R.LAUNCH_TREE
KIT8 = Path(os.environ.get("R8_KIT", "E:/refcv8_eval_kit"))
CONFIG = KIT8 / "ckpt" / "config.json"
needs_tree = pytest.mark.skipif(not (TREE / "stack" / "scripts" / "refc_v3_train.py").exists(),
                                reason="launch tree C:/lgs/Lfb47a58376 absent (dev-box-only test)")
needs_kit = pytest.mark.skipif(not CONFIG.exists(), reason="E:/refcv8_eval_kit/ckpt/config.json absent (dev box only)")


# ------------------------------------------------------------------------------------- stub loader
class _StubDS:
    """Stands in for V3Dataset: records `enable_pose_sync` and exposes `.pose_sync`."""
    pose_sync = None
    calls: list = []

    def __init__(self, eps, **kw):
        self.eps = eps

    def enable_pose_sync(self, path, **kw):
        type(self).calls.append(str(path))
        self.pose_sync = object()
        return {"sidecar": str(path), "n_clips": 3, "n_covered": 3, "n_uncovered_read_unshifted": 0}


def _stub_loader():
    _StubDS.calls = []

    class _L:
        __file__ = "stub"
        seen = []

        @staticmethod
        def trainer():
            return SimpleNamespace(V3Dataset=_StubDS)

        @staticmethod
        def build_eval_dataset(model, cfg, args, config, *, with_perception_targets=True, dataset_cls=None):
            _L.seen.append(dataset_cls)
            cls = dataset_cls or _StubDS
            return cls(["ep"]), ["ep"], {}

    _L.seen = []
    return _L


def _args(sidecar):
    return SimpleNamespace(pose_sync_sidecar=sidecar, graft_lan=False, goal_str=False)


# ------------------------------------------------------------------------------------- the patch
def test_patch_attaches_the_sidecar_and_records_it():
    L = _stub_loader()
    R.patch_loader(L, sink={})
    e_ds, _eps, rec = L.build_eval_dataset(None, None, _args("S.jsonl"), {})
    assert _StubDS.calls == ["S.jsonl"]                       # exactly once, with the (remapped) path
    assert e_ds.pose_sync is not None
    assert rec["pose_sync"]["n_covered"] == 3 and rec["pose_sync"]["sidecar"] == "S.jsonl"
    R.assert_pose_sync_on(e_ds)                               # the positive check passes


def test_patch_preserves_the_callers_dataset_class_behaviour():
    """G0 / the rolls pass their own V3Dataset subclass; the wrapper must subclass THAT, not replace it."""
    L = _stub_loader()
    R.patch_loader(L, sink={})

    class G0Windows(_StubDS):
        def _window_u8(self):
            return "g0"

    e_ds, _e, _r = L.build_eval_dataset(None, None, _args("S.jsonl"), {}, dataset_cls=G0Windows)
    assert isinstance(e_ds, G0Windows) and e_ds._window_u8() == "g0" and e_ds.pose_sync is not None


def test_a_refcv7_argv_is_a_pass_through():
    """No --pose-sync-sidecar (refcv7) => the original call, byte for byte: the SAME dataset_cls object."""
    L = _stub_loader()
    R.patch_loader(L, sink={})
    e_ds, _e, rec = L.build_eval_dataset(None, None, _args(None), {})
    assert L.seen == [None] and e_ds.pose_sync is None and "pose_sync" not in rec and _StubDS.calls == []

    class Mine(_StubDS):
        pass

    L.build_eval_dataset(None, None, _args(None), {}, dataset_cls=Mine)
    assert L.seen[-1] is Mine


def test_patch_is_idempotent():
    L = _stub_loader()
    first = R.patch_loader(L, sink={})
    f = L.build_eval_dataset
    assert R.patch_loader(L, sink={}) is first and L.build_eval_dataset is f
    L.build_eval_dataset(None, None, _args("S.jsonl"), {})
    assert _StubDS.calls == ["S.jsonl"]                       # not attached twice


def test_REGRESSION_loader_without_the_patch_is_caught():
    """The real defect: the loader never calls enable_pose_sync. `assert_pose_sync_on` must go RED on it."""
    L = _stub_loader()
    R.patch_loader(L, enabled=False, sink={})                 # the omission, reintroduced
    e_ds, _e, _r = L.build_eval_dataset(None, None, _args("S.jsonl"), {})
    assert _StubDS.calls == [] and e_ds.pose_sync is None
    with pytest.raises(SystemExit, match="no pose_sync attached"):
        R.assert_pose_sync_on(e_ds)


# ------------------------------------------------------------------ the patch on the REAL V3Dataset
@needs_tree
def test_real_v3dataset_known_value_and_regression(tmp_path):
    """Real `V3Dataset` on the existing pose-sync test's synthetic clips: with the wrapper class the accelerating
    clip's v0 (= pose_last[3]) moves by a * delta = 2 * 0.0165 = 0.033 m/s (literal); the UNWRAPPED class is
    bit-identical to the correction OFF -- i.e. an instrument built from it scores the unshifted clock."""
    import torch
    sys.path[:0] = [str(TREE / "stack"), str(TREE / "stack" / "scripts")]
    from tanitad.data import pose_sync as PS
    from tanitad.data.v2_dataset import stable_episode_id
    import refc_v3_train as T
    rows_n, w, dt = 130, 8, 0.1006666

    def episode(clip, poses):
        actions = np.stack([np.linspace(0.0, 0.05, rows_n), np.linspace(0.0, 1.0, rows_n)], 1).astype(np.float32)
        return SimpleNamespace(frames=torch.zeros(rows_n, 9, 8, 8, dtype=torch.uint8),
                               actions=torch.from_numpy(actions), poses=torch.from_numpy(poses.copy()),
                               episode_id=stable_episode_id(clip))

    t = dt * np.arange(rows_n, dtype=np.float64)
    acc = np.zeros((rows_n, 4))
    acc[:, 0], acc[:, 3] = 5.0 * t + 0.5 * 2.0 * t * t, 5.0 + 2.0 * t          # a = 2 m/s^2, v0 = 5
    cv = np.zeros((rows_n, 4))
    cv[:, 0], cv[:, 3] = 12.0 * dt * np.arange(rows_n), 12.0
    eps = [episode("clip-aaa", acc.astype(np.float32)), episode("clip-bbb", cv.astype(np.float32))]
    sid_rows = [{"sid": int(e.episode_id), "dt_s": dt, "delta_us": [16500] * (rows_n + 2)} for e in eps]
    sidecar = str(tmp_path / "sync.jsonl")
    PS.write_pose_sync_sidecar(sidecar, sid_rows, {"test": True})

    def plain():
        ds = T.V3Dataset(eps, window=w, max_horizon=20, channels=9)
        ds.v7_by_sid, ds.v7_dt = {}, 0.1
        return ds

    sink: dict = {}
    wrapped_cls = R.make_pose_synced_cls(T.V3Dataset, sidecar, sink)
    synced = wrapped_cls(eps, window=w, max_horizon=20, channels=9)
    synced.v7_by_sid, synced.v7_dt = {}, 0.1
    off = plain()
    assert sink["n_covered"] == 2 and sink["n_uncovered_read_unshifted"] == 0
    R.assert_pose_sync_on(synced)
    a, b = off[3], synced[3]                                    # clip 0 (accelerating), t = 3
    assert float(b["pose_last"][3] - a["pose_last"][3]) == pytest.approx(0.033, abs=2e-5)
    # REGRESSION ARM: the class the unpatched loader would have built is bit-identical to OFF
    with pytest.raises(SystemExit, match="no pose_sync attached"):
        R.assert_pose_sync_on(plain())
    assert torch.equal(plain()[3]["pose_last"], a["pose_last"])


# ------------------------------------------------------------------ the remap covers the refcv8 argv
def _load_loader_fresh(monkeypatch, overrides_file: Path, name: str):
    for k in list(os.environ):
        if k.startswith("REFCV6_"):
            monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("REFCV7_KIT", R.EVAL_KIT)
    monkeypatch.setenv("REFCV7_REMAP_OVERRIDES", str(overrides_file))
    spec = importlib.util.spec_from_file_location(name, TREE / "stack" / "tanitad" / "eval" / "refcv7_loader.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _missing_eval_side_inputs(mod, argv) -> list[str]:
    out, _rec = mod.remap_argv(list(argv))
    miss = []
    for flag in R.EVAL_SIDE_FLAGS:
        if flag in out:
            p = out[out.index(flag) + 1]
            if not os.path.exists(p):
                miss.append(f"{flag} -> {p}")
        else:
            miss.append(f"{flag} (absent from argv)")
    return miss


@needs_tree
@needs_kit
def test_remap_resolves_every_eval_side_flag_of_the_refcv8_argv(monkeypatch):
    argv = json.load(open(CONFIG, encoding="utf-8"))["argv"]
    mod = _load_loader_fresh(monkeypatch, R.REMAP_OVERRIDES, "r8_loader_good")
    assert _missing_eval_side_inputs(mod, argv) == []
    # the launch gate's own four overrides are a SUBSET of ours, with the same flags
    gate = json.load(open(TREE / "stack" / "ops" / "refcv8_eval_remap_a2_thor.json", encoding="utf-8"))
    ours = json.load(open(R.REMAP_OVERRIDES, encoding="utf-8"))
    assert set(gate) <= set(ours) and set(ours) - set(gate) == {"--r8-v9-labels-eval", "--pose-sync-sidecar", "--join-defect-masks"}


@needs_tree
@needs_kit
def test_REGRESSION_a_remap_without_the_three_new_flags_is_caught(monkeypatch, tmp_path):
    ours = json.load(open(R.REMAP_OVERRIDES, encoding="utf-8"))
    for k in ("--r8-v9-labels-eval", "--pose-sync-sidecar", "--join-defect-masks"):
        ours.pop(k)
    mutant = tmp_path / "mutant_remap.json"
    mutant.write_text(json.dumps(ours), encoding="utf-8")
    argv = json.load(open(CONFIG, encoding="utf-8"))["argv"]
    mod = _load_loader_fresh(monkeypatch, mutant, "r8_loader_mutant")
    miss = _missing_eval_side_inputs(mod, argv)
    assert len(miss) == 3 and any("--r8-v9-labels-eval" in m for m in miss) \
        and any("--pose-sync-sidecar" in m for m in miss) and any("--join-defect-masks" in m for m in miss)


def test_configure_env_refuses_a_stale_shell_value(monkeypatch):
    monkeypatch.delitem(sys.modules, "tanitad.eval.refcv7_loader", raising=False)
    monkeypatch.setenv("REFCV7_KIT", "Z:/some/other/kit")
    with pytest.raises(SystemExit, match="disagrees"):
        R.configure_env()


def test_configure_env_refuses_after_the_loader_was_imported(monkeypatch):
    monkeypatch.setitem(sys.modules, "tanitad.eval.refcv7_loader", SimpleNamespace())
    with pytest.raises(SystemExit, match="already imported"):
        R.configure_env()
