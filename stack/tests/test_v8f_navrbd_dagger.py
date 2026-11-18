"""PREREG_V8F_NAVRBD: the 15-column ``nav_rb`` rides the DAgger swap path EXACTLY as the 1-column one does (NAVRB2, ``test_v8f_navrb2.py``): the swap replaces own-state tensors only, ``nav_rb`` has no swap rule and is HELD AS
LOGGED (the same object, bit-equal), the dagger probe accepts ``nav_rb`` among the keys ``forward_infer`` reads, and the partition mutation (``NAV_RB_KEYS`` emptied) turns the probe RED for the wide channel too. CPU only.
Run the whole file (module-scoped fixtures)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "scripts"))
import tanitad  # noqa: E402
from tanitad.data import v8f_dagger as DG  # noqa: E402
from tanitad.data import v8f_mini_data as D  # noqa: E402
from tanitad.models import v8f_mini as M  # noqa: E402
import v8f_mini_train as TR  # noqa: E402
from test_v8f_dagger_train import GATE_ARGV, _woken, pack  # noqa: E402,F401  (pack: the module fixture: a trained checkpoint, its DAgger bank, the synthetic world)
from test_v8f_navrbd import sha, write_rbd_npz  # noqa: E402
from test_v8f_navrb2 import write_navrb_npz  # noqa: E402

assert Path(tanitad.__file__).resolve().is_relative_to(HERE.parents[1].resolve()), tanitad.__file__


def parts(pk, tmp_path, feat, p=0.5, n=24, seed=0):
    J = {k: v for k, v in pk.J.items()}
    rbp = write_rbd_npz(tmp_path / f"rb_{feat}.npz", J, p=0.4) if feat == "exitdist" else write_navrb_npz(tmp_path / f"rb_{feat}.npz", J, p=0.4)
    extra = ["--nav-rb-feat", "exitdist"] if feat == "exitdist" else []
    a = TR.build_parser().parse_args(["--out", str(tmp_path / "o"), *GATE_ARGV, "--nav-rb", "on", *extra, "--nav-rb-npz", str(rbp), "--nav-rb-sha256", sha(rbp), "--detach-upper"])
    cfg = TR.make_config(a)
    assert cfg.nav_rb == "on" and cfg.nav_rb_feat == feat
    J["nav_rb"], _ = D.nav_rb_arrays(J, str(rbp), sha(rbp), "train", "real", feat)
    store = D.ClipStore(pk.world, "train", lru=8, kin_mode="yaw", tracks=D.load_native_tracks(pk.world / "native_tracks.pt"))
    wins = D.train_windows(store, J, D.select_train_clips(list(map(str, J["clip"])), 1.0))
    ds = D.V8FMiniWindows(store, J, wins, hmax=cfg.hmax)
    idx = np.random.default_rng(seed).choice(len(wins), n, replace=False)
    batch = D.collate([ds.item(wins[i][0], wins[i][1], 1, 3) for i in idx])
    mu, sd = M.cons_stats(J)
    model = _woken(M.V8FMini(cfg, M.P10Prior.synthetic(0), mu, sd))
    return model, batch, cfg, DG.DaggerSwap(DG.DaggerBank.load(pk.bank, clip_names=J["clip"]), p=p, seed=1, kin_mode="yaw")


def test_the_15_column_channel_rides_the_swap_exactly_as_the_1_column_one_does(pack, tmp_path):
    seen = {}
    for feat, width in (("state", 1), ("exitdist", 15)):
        model, b, cfg, dag = parts(pack, tmp_path, feat)
        assert b["nav_rb"].shape[-1] == width
        res = TR.dagger_probe(model, b, cfg, dag)
        assert res["pass"] is True and "nav_rb" in res["keys_read"]
        pert = dag.apply(b, DG.DaggerSwap.PROBE_STEP, p=1.0)
        assert pert["nav_rb"] is b["nav_rb"] and torch.equal(pert["nav_rb"], b["nav_rb"])                       # no swap rule: HELD AS LOGGED (the same object)
        assert any(not torch.equal(pert[k], b[k]) for k in DG.REBUILT)                                          # the swap itself is live in this very call (the control)
        seen[feat] = (res["keys_read"], {k: pert[k] is b[k] for k in b})
    assert seen["state"] == seen["exitdist"]                                                                    # the same keys read, and the same keys replaced / kept, for both widths


def test_dagger_probe_mutation_without_the_partition_is_red_for_the_wide_channel(pack, tmp_path, monkeypatch):
    model, b, cfg, dag = parts(pack, tmp_path, "exitdist")
    monkeypatch.setattr(M, "NAV_RB_KEYS", frozenset())
    with pytest.raises(AssertionError):
        TR.dagger_probe(model, b, cfg, dag)
