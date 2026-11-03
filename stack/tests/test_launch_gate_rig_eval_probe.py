"""refcv9-e1rig G-EVAL: the probe dataset of a rig argv (GATE3, 2026-11-03).

MEASURED on Thor at dbc72ac6 (exploratory, non-token gate run on C0): every non-suite check PASSed except G-EVAL, which FAILED with
"the eval loader REFUSED: [refcv7_loader] the run's argv has no --eval-labels" -- the vendored loader's `build_eval_dataset` raises on
an argv with no label flag, and the rig argv carries none by design. `rig_eval_probe_dataset` is train()'s own construction of the rig's
held-out eval dataset (refc_v3_train.py: providers -> `dcls(e_eps, **kw)` -> `u8_frames` -> `newest_row_only` -> `enable_pose_sync`),
minus every block train() runs only `if args.eval_labels`.

  P1  a rig argv builds the dataset with train()'s settings and never calls the vendored `build_eval_dataset`
  P2  `newest_row_only` follows `u8_frames` (train(): bool(_r9rig_on and u8))
  P3  a rig argv that DOES carry --eval-labels is refused (no guessing)
  P4  REACHABILITY: the G-EVAL probe site calls the helper, and only under `r9_perc_rig`; the non-rig call is still the loader's
      (T1-GEVAL, 2026-11-03: the call now hands over the LOADER-built model, `model=model_l` -- the keyword is optional, so P1-P3 are unchanged;
      the T1 history attach is pinned in test_launch_gate_geval_t1.py)
"""
from __future__ import annotations

import inspect
import re
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import launch_gate as LG  # noqa: E402


class _DS:
    """a stand-in for the trainer's V3Dataset (the helper only sets attributes and calls enable_pose_sync)"""

    def __init__(self, eps, **kw):
        self.eps, self.kw, self.synced = eps, kw, None

    def __len__(self):
        return 7

    def enable_pose_sync(self, path):
        self.synced = path
        return {"n_shifted": 1}


def _loader():
    def boom(*a, **k):
        raise AssertionError("the rig probe must NOT call the vendored build_eval_dataset")
    return types.SimpleNamespace(trainer=lambda: types.SimpleNamespace(V3Dataset=_DS), build_eval_dataset=boom)


def _cfg():
    return types.SimpleNamespace(core=types.SimpleNamespace(window=8, encoder=types.SimpleNamespace(in_channels=9)))


def _args(**kw):
    base = dict(eval_cache="/x/eval", v2_lru=6, u8_batches=True, pose_sync_sidecar="/x/ps.jsonl", eval_labels=None)
    base.update(kw)
    return types.SimpleNamespace(**base)


@pytest.fixture()
def providers(monkeypatch):
    calls = {}
    mod = types.ModuleType("tanitad.data.v2_dataset")

    def build_v2_providers(caches, lru_size):
        calls["args"] = (list(caches), lru_size)
        return [object(), object()]
    mod.build_v2_providers = build_v2_providers
    monkeypatch.setitem(sys.modules, "tanitad.data.v2_dataset", mod)
    return calls


def test_P1_rig_probe_dataset_follows_train(providers):
    ds, eps, rec = LG.rig_eval_probe_dataset(_loader(), _cfg(), _args())
    assert providers["args"] == (["/x/eval"], 6)
    assert isinstance(ds, _DS) and ds.kw == {"window": 8, "max_horizon": 20, "channels": 9}
    assert ds.u8_frames is True and ds.newest_row_only is True
    assert ds.synced == "/x/ps.jsonl" and rec["pose_sync"] == {"n_shifted": 1}
    assert rec["rig"] is True and rec["labels"] is None and rec["n_episodes"] == 2 and rec["n_windows"] == 7 and len(eps) == 2


def test_P2_newest_row_only_follows_u8(providers):
    ds, _, rec = LG.rig_eval_probe_dataset(_loader(), _cfg(), _args(u8_batches=False, pose_sync_sidecar=None))
    assert ds.u8_frames is False and ds.newest_row_only is False and ds.synced is None and "pose_sync" not in rec


def test_P3_a_rig_argv_with_eval_labels_is_refused(providers):
    with pytest.raises(LG.GateError, match="eval-labels"):
        LG.rig_eval_probe_dataset(_loader(), _cfg(), _args(eval_labels="/x/labels.jsonl.gz"))


def test_P4_the_probe_site_reaches_the_helper_only_under_the_rig():
    sites = [(n, f) for n, f in inspect.getmembers(LG, inspect.isfunction)
             if "rig_eval_probe_dataset(L, cfg_l, args_l, model=model_l)" in inspect.getsource(f)]
    assert len(sites) == 1, [n for n, _ in sites]
    src = inspect.getsource(sites[0][1])
    m = re.search(r'if bool\(getattr\(args_l, "r9_perc_rig", False\)\):\s*\n(?:\s*#[^\n]*\n)+\s*e_ds, e_eps, drec = rig_eval_probe_dataset\(L, cfg_l, args_l, model=model_l\)\s*\n\s*else:\s*\n\s*e_ds, e_eps, drec = L\.build_eval_dataset\(', src)
    assert m, "the rig branch must wrap the helper and leave the non-rig call to the vendored loader"
