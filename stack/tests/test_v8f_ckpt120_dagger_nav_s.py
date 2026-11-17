"""PREREG_V8F_CKPT120, DAgger swap x ``nav_s``: the trainer's ``dagger_probe`` (the G-DVB ``--selfcheck-only`` gate) on a batch that CARRIES ``nav_s`` -- the H2 arms' situation (``--nav-strat on`` with ``--dagger-p`` > 0).

The defect (pod, H2_s0 mint): ``dagger_probe`` admitted ``nav_s`` in its ``stray`` partition but its ``kept_moved`` check still demanded ``pert["nav_s"] is batch["nav_s"]``, while the swap REBUILDS ``nav_s``
channel 3 by the nav rule (``v8f_dagger.swap_rows``): the gate refused with "['nav_s'] are not the same objects after the swap". H1 arms carry no ``nav_s`` and were not affected.

The H2 flags used here are the pod argv's own CKPT120 levers (``F:/TanitAD-scratch/strat1/ckpt120build/pod/ckpt120_argvs_dry.json``, arm H2_s0): ``--s-far on --w-ckpt-far 0.3333 --s-far-lr-mult 10 --s-far-label-perm off
--s-adapter-detach off --nav-strat on --nav-strat-variant real --detach-upper`` plus ``--dagger-p`` / bank. Sizes are the test's (d 32, n_q 2).

Mutations (each must turn the probe RED): the swap changes ``nav_s`` channel 0 (token); changes ``nav_s`` on an UNSELECTED row; changes a frame that does not share NOW's token (the nav rule keeps it);
changes the NOW frame; leaves ``nav_s`` out of the swap's output keys."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_v8f_dagger_train import (D, DG, GATE_ARGV, M, TR, DAG, Pack, _woken, pack)  # noqa: E402,F401  (pack: the module fixture: a trained checkpoint, its DAgger bank, the synthetic world)
from test_v8f_ckpt120 import sha, write_nav_s_npz, write_rc120_table  # noqa: E402

H2_LEVERS = ["--s-far", "on", "--w-ckpt-far", "0.3333", "--s-far-lr-mult", "10", "--s-far-label-perm", "off", "--s-adapter-detach", "off", "--nav-strat-variant", "real", "--detach-upper"]


def h2_parts(pk, tmp_path, p=0.5, n=24, seed=0, nav_strat="on"):
    """(model, a real collated batch carrying ``nav_s`` (when nav_strat is on), cfg, the swap): the trainer's own parser -> ``make_config`` -> model, on the DAgger pack's world and bank."""
    J = {k: v for k, v in pk.J.items()}
    rc, ns = write_rc120_table(tmp_path / "rc120.npz", J), write_nav_s_npz(tmp_path / "navs.npz", J)
    extra = list(H2_LEVERS)
    if nav_strat == "on":
        extra += ["--nav-strat", "on", "--nav-strat-npz", str(ns), "--nav-strat-sha256", sha(ns)]
    a = TR.build_parser().parse_args(["--out", str(tmp_path / "o"), *GATE_ARGV, "--rc120-npz", str(rc), "--rc120-sha256", sha(rc), *extra])
    cfg = TR.make_config(a)
    assert (cfg.s_far, cfg.nav_strat) == ("on", nav_strat)
    J["rc120"], J["rc120_ok"], _ = D.rc120_arrays(J, str(rc), sha(rc), "train", "off")
    if nav_strat == "on":
        J["nav_s"], _ = D.nav_strat_arrays(J, str(ns), sha(ns), "train", "real")
    store = D.ClipStore(pk.world, "train", lru=8, kin_mode="yaw", tracks=D.load_native_tracks(pk.world / "native_tracks.pt"))
    wins = D.train_windows(store, J, D.select_train_clips(list(map(str, J["clip"])), 1.0))
    ds = D.V8FMiniWindows(store, J, wins, hmax=cfg.hmax)
    idx = np.random.default_rng(seed).choice(len(wins), n, replace=False)
    batch = D.collate([ds.item(wins[i][0], wins[i][1], 1, 3) for i in idx])
    mu, sd = M.cons_stats(J)
    model = _woken(M.V8FMini(cfg, M.P10Prior.synthetic(0), mu, sd))
    bank = DG.DaggerBank.load(pk.bank, clip_names=J["clip"])
    return model, batch, cfg, DG.DaggerSwap(bank, p=p, seed=1, kin_mode="yaw")


def test_the_probe_passes_on_a_batch_with_nav_s_and_dagger_on_and_records_the_nav_s_counts(pack, tmp_path):
    model, b, cfg, dag = h2_parts(pack, tmp_path)
    assert "nav_s" in b, "the H2 situation: the batch carries nav_s"
    res = TR.dagger_probe(model, b, cfg, dag)
    assert res["pass"] is True and "nav_s" in res["keys_read"]
    ns = res["nav_s"]
    assert ns["rows_changed"] >= 1 and ns["ch3_frames_changed"] >= 1, f"the swap must rebuild nav_s channel 3 on the selected rows: {ns}"
    assert ns["token_validity_bit_identical"] is True and ns["unselected_rows_bit_identical"] is True and ns["ch3_max_abs_err"] <= 1e-6
    assert ns["ch3_frames_changed"] <= ns["ch3_frames_rebuilt_by_rule"] and ns["rows_changed"] <= res["n_selected_at_p1"]


def test_a_batch_without_nav_s_behaves_as_before(pack, tmp_path):
    model, b, cfg, dag = h2_parts(pack, tmp_path, nav_strat="off")
    assert "nav_s" not in b
    r1 = TR.dagger_probe(model, b, cfg, dag)
    assert r1["pass"] is True and "nav_s" not in r1 and not any(k.startswith("nav_s") for k in r1), "no nav_s in the batch: no nav_s record, the result is the pre-change result"
    r2 = TR.dagger_probe(model, b, cfg, dag)
    drop = lambda r: {k: v for k, v in r.items() if k != "live_swapped"}       # noqa: E731  (live_check's own gradient sums move in the 6th digit between two calls: not part of this check)
    assert drop(r1) == drop(r2), "and the rest of the result is deterministic, bit for bit"


def _wrap_swap(monkeypatch, edit):
    real = DG.swap_rows

    def mut(batch, rows, rec):
        out = real(batch, rows, rec)
        edit(out, batch, rows)
        return out
    monkeypatch.setattr(DG, "swap_rows", mut)


def _expect_red(pk, tmp_path, monkeypatch, edit, match):
    model, b, cfg, dag = h2_parts(pk, tmp_path)
    TR.dagger_probe(model, b, cfg, dag)                                           # the unmutated swap passes the same probe on the same batch
    _wrap_swap(monkeypatch, edit)
    with pytest.raises(AssertionError, match=match):
        TR.dagger_probe(model, b, cfg, dag)


def test_mutation_the_swap_changes_nav_s_channel_0_is_refused(pack, tmp_path, monkeypatch):
    def edit(out, batch, rows):
        out["nav_s"] = out["nav_s"].clone()
        out["nav_s"][rows[:1], :, 0] = 1.0 - out["nav_s"][rows[:1], :, 0]
    _expect_red(pack, tmp_path, monkeypatch, edit, r"nav_s channels 0-2")


def test_mutation_the_swap_changes_nav_s_on_an_unselected_row_is_refused(pack, tmp_path, monkeypatch):
    def edit(out, batch, rows):
        un = [r for r in range(int(batch["nav_s"].shape[0])) if r not in set(rows.tolist())]
        assert un, "the fixture needs an unselected row"
        out["nav_s"] = out["nav_s"].clone()
        out["nav_s"][un[0], :, 3] = (out["nav_s"][un[0], :, 3] + 0.5).clamp(0.0, 3.0)
    _expect_red(pack, tmp_path, monkeypatch, edit, r"nav_s changed on a row the swap did not select")


def test_mutation_the_swap_moves_the_now_frame_of_nav_s_is_refused(pack, tmp_path, monkeypatch):
    def edit(out, batch, rows):
        out["nav_s"] = out["nav_s"].clone()
        out["nav_s"][rows, -1, 3] = (out["nav_s"][rows, -1, 3] + 0.25).clamp(0.0, 2.5) + 0.01
    _expect_red(pack, tmp_path, monkeypatch, edit, r"nav_s channel 3 differs from the nav rule")


def test_mutation_the_swap_changes_a_frame_outside_the_nav_rule_is_refused(pack, tmp_path, monkeypatch):
    def edit(out, batch, rows):
        ns0 = batch["nav_s"]
        out["nav_s"] = out["nav_s"].clone()
        hit = 0
        for r in rows.tolist():
            for f in range(int(ns0.shape[1]) - 1):
                if int(ns0[r, f, :3].argmax()) != int(ns0[r, -1, :3].argmax()) or float(ns0[r, f, 4]) != float(ns0[r, -1, 4]):
                    out["nav_s"][r, f, 3] = (ns0[r, f, 3] + 0.5) % 3.0 if ns0[r, f, 3] < 2.5 else ns0[r, f, 3] - 0.5
                    hit += 1
        if not hit:
            pytest.skip("this synthetic batch has no frame across a token / validity boundary on a selected row")
    _expect_red(pack, tmp_path, monkeypatch, edit, r"nav_s channel 3 differs from the nav rule")


def test_mutation_the_swap_drops_the_nav_s_key_is_refused(pack, tmp_path, monkeypatch):
    def edit(out, batch, rows):
        out["nav_s"] = batch["nav_s"]                                             # the swap forgets to rebuild it: every selected row keeps the logged countdown
    model, b, cfg, dag = h2_parts(pack, tmp_path)
    _wrap_swap(monkeypatch, edit)
    with pytest.raises(AssertionError, match=r"nav_s channel 3 differs from the nav rule"):
        TR.dagger_probe(model, b, cfg, dag)
