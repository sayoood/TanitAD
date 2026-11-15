"""PREREG_V8F_DIVRS: ``--wm-sampler {uniform,divrs}`` (default uniform). The data half (``window_act_means``, ``divrs_cells``, ``divrs_weights``, ``DiversitySampler``, ``check_cell_shares``, ``ClipStore.get_act``) and the trainer half
(the REAL ``v8f_mini_train.main`` on its synthetic world with tiny dimensions).

What this file pins, by the card's controls and the programme's "built, tested and unreachable" rule:

* The CELL RULE of the card: 5 x 5 grid, mean |yaw rate| x mean longitudinal acceleration over a window's transitions (the ``hmax - 1`` transitions the world model reads, front clamp included), the card's bin edges (a value ON
  an edge goes to the upper bin), weight 1 / count, cap 10 x the median window weight, empty cells no weight. ``window_act_means`` equals a brute-force mean over ``V8FMiniWindows.item``'s ``act[:-1]``; ``get_act`` equals ``get``.
* The SAMPLER: ``len(windows)`` draws per epoch (sums exactly), valid windows, the uniform sampler's age draws, I/O locality (a run of batches touches ~one chunk of clips, not the whole set), a seeded resumable stream.
  Its realised cell shares equal the card's targets whatever the chunk composition (the chunk allocation is by weight mass, systematic rounding).
* CARD CONTROL 1: the realised share of each cell over the first 2,000 batches within +-20 % (relative) of its target, computed from the cell table and the card's rule -- NOT from the sampler. The DELIBERATE-REGRESSION ARM
  (flag set, uniform draw) FAILS it; so does a sampler that allocates draws to chunks by chunk size, one without the weights, one without the cap.
* G-DVB: the flag has a ``FLAG_CHECKS`` entry; the built sampler's own record (kind, edges, cap multiple, cell table) is read back and compared with the card's literals; the gate refuses a flag set whose sampler was built
  uniform, a drifted bin edge, a missing world model, a non-yaw kin mode; G-LIVE: the control-1 probe on the BUILT sampler refuses a sampler whose record says divrs but whose draws are uniform.
* Default off: ``--wm-sampler uniform`` named and the flag absent give bit-equal checkpoints (and the uniform path is ``ChunkedSampler`` called as before); divrs moves the world model; a whole WM-only run and a whole M2b run
  reach the loss through the diversity sampler (spy on ``epoch``).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "stack", REPO / "taniteval", REPO / "stack" / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from tanitad.data import v8f_mini_data as D  # noqa: E402
from tanitad.models import v8f_mini as M  # noqa: E402
import v8f_mini_train as TR  # noqa: E402

torch.set_num_threads(1)
BASE = ["--synthetic", "--batch", "16", "--d", "32", "--n-q", "2", "--kin-mode", "yaw", "--seed", "5", "--arm", "M2b", "--threads", "1", "--warmup", "2"]
WMO = ["--hmax", "28", "--wm-only"]
CENTER_YAW = [0.01, 0.035, 0.075, 0.15, 0.30]       # a representative mean |yaw rate| inside each of the 5 yaw bins
CENTER_ACC = [-2.0, -0.65, 0.0, 0.65, 2.0]          # ... and mean acceleration inside each of the 5 acceleration bins


def feats_of_cells(cells) -> np.ndarray:
    cells = np.asarray(cells)
    return np.stack([np.array(CENTER_YAW)[cells // 5], np.array(CENTER_ACC)[cells % 5]], 1)


def fake_table(n_clips=60, per=50, seed=0):
    """3,000 windows over 60 clips, ~65 % in the straight, steady cell (12); every clip has a dominant cell, so chunks of clips differ a lot in composition (which is what a wrong chunk allocation gets wrong)."""
    rng = np.random.default_rng(seed)
    dom = np.where(rng.random(n_clips) < 0.7, 12, rng.integers(0, 25, n_clips))
    cells = np.where(rng.random((n_clips, per)) < 0.85, dom[:, None], rng.integers(0, 25, (n_clips, per)))
    windows = [(ci, 3 + j, 0, 0) for ci in range(n_clips) for j in range(per)]
    return windows, feats_of_cells(cells.reshape(-1)), cells.reshape(-1)


# ------------------------------------------------------------------------------------------------ the cell rule
def test_bin_edges_put_a_value_on_an_edge_into_the_upper_bin_and_cover_25_cells():
    yaw = [0.0, 0.0199, 0.02, 0.0499, 0.05, 0.0999, 0.10, 0.1999, 0.20, 9.0]
    want_y = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    acc = [-9.0, -1.0001, -1.0, -0.3001, -0.3, 0.2999, 0.3, 0.9999, 1.0, 9.0]
    want_a = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    got = D.divrs_cells(np.stack([yaw, acc], 1))
    assert got.tolist() == [y * 5 + a for y, a in zip(want_y, want_a)]
    assert D.DIVRS_N_CELLS == 25 and D.DIVRS_YAW_EDGES == (0.02, 0.05, 0.10, 0.20) and D.DIVRS_ACC_EDGES == (-1.0, -0.3, 0.3, 1.0) and D.DIVRS_CAP_MULT == 10.0
    assert TR.DIVRS_CARD == {"yaw_edges": list(D.DIVRS_YAW_EDGES), "acc_edges": list(D.DIVRS_ACC_EDGES), "cap_mult": D.DIVRS_CAP_MULT, "n_cells": D.DIVRS_N_CELLS}
    assert sorted(set(D.divrs_cells(feats_of_cells(np.arange(25))).tolist())) == list(range(25))


def test_weights_follow_the_card_rule_by_hand():
    """counts 100 / 50 / 8 / 1 -> raw weights .01 / .02 / .125 / 1; the median over the 159 windows is .01, so the cap is .1: the last two cells are capped; 21 cells are empty and carry nothing."""
    cells = np.array([0] * 100 + [1] * 50 + [2] * 8 + [3] * 1)
    W = D.divrs_weights(cells)
    assert W["counts"][:4].tolist() == [100, 50, 8, 1] and W["n_empty_cells"] == 21 and W["n_capped_cells"] == 2
    assert W["median_weight"] == pytest.approx(0.01) and W["cap"] == pytest.approx(0.1)
    assert W["cell_weight"][:4] == pytest.approx([0.01, 0.02, 0.1, 0.1]) and not W["cell_weight"][4:].any()
    assert W["target_share"][:4] == pytest.approx(np.array([1.0, 1.0, 0.8, 0.1]) / 2.9) and W["target_share"].sum() == pytest.approx(1.0)
    assert W["weight"].max() <= W["cap"] * (1 + 1e-12) and W["weight"][:100] == pytest.approx(0.01)
    # without the cap every non-empty cell has the same share (1/n_nonempty): the cap is what changes it
    assert D.divrs_weights(cells, cap_mult=np.inf)["target_share"][:4] == pytest.approx([0.25] * 4)


# ------------------------------------------------------------------------------------------------ the window table on the synthetic world
@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("divrs_world")
    w = D.make_synthetic_world(root, n_train=8, n_eval=2, T=120, seed=0)
    J = D.load_join(Path(root) / "join_train.npz")
    tracks = D.load_native_tracks(Path(root) / "native_tracks.pt")
    store = D.ClipStore(root, "train", lru=8, kin_mode="yaw", tracks=tracks)
    wins = D.train_windows(store, J, w["names"]["train"])
    return {"root": root, "J": J, "tracks": tracks, "store": store, "wins": wins, "names": w["names"]["train"]}


def test_get_act_equals_the_full_get(world):
    fresh = D.ClipStore(world["root"], "train", lru=8, kin_mode="yaw", tracks=world["tracks"])
    full = D.ClipStore(world["root"], "train", lru=8, kin_mode="yaw", tracks=world["tracks"])
    for c in world["names"]:
        a, ch = fresh.get_act(c)
        d = full.get(c)
        assert np.array_equal(a, d["a"]) and np.array_equal(ch, d["ch1"]) and ch.dtype == d["ch1"].dtype
    with pytest.raises(ValueError, match="yaw-mode"):
        D.ClipStore(world["root"], "train", lru=8).get_act(world["names"][0])


def test_window_act_means_equal_the_brute_force_mean_over_the_transitions_the_wm_reads(world):
    hmax = 28
    ds = D.V8FMiniWindows(world["store"], world["J"], world["wins"], hmax=hmax)
    fresh = D.ClipStore(world["root"], "train", lru=8, kin_mode="yaw", tracks=world["tracks"])
    got = D.window_act_means(fresh, world["J"], world["wins"], hmax)
    assert got.shape == (len(world["wins"]), 2) and np.isfinite(got).all()
    pick = sorted(set(np.random.default_rng(1).choice(len(world["wins"]), 60, replace=False).tolist()) | {0, 1, 2, len(world["wins"]) - 1})   # includes the first windows of a clip (front clamp at t < hmax - 1)
    assert min(world["wins"][i][1] for i in pick) < hmax - 1
    for i in pick:
        ci, t, _, _ = world["wins"][i]
        act = ds.item(ci, t)["act"][:-1].astype(np.float64)                  # the hmax - 1 transitions the WM reads: act[:, :-1]
        assert act.shape == (hmax - 1, 2)
        assert got[i, 0] == pytest.approx(np.abs(act[:, 1]).mean(), abs=1e-12) and got[i, 1] == pytest.approx(act[:, 0].mean(), abs=1e-12)


def test_the_diversity_sampler_over_the_synthetic_world_accounts_for_every_window(world):
    feats = D.window_act_means(world["store"], world["J"], world["wins"], 28)
    s = D.DiversitySampler(world["wins"], 16, feats, seed=3, chunk=4)
    b = s.built_levers()
    assert b["kind"] == "divrs" and sum(b["cell_counts"]) == b["n_windows"] == len(world["wins"]) and abs(sum(b["target_share"]) - 1.0) < 1e-12
    assert b["yaw_edges"] == [0.02, 0.05, 0.10, 0.20] and b["acc_edges"] == [-1.0, -0.3, 0.3, 1.0] and b["cap_mult"] == 10.0 and b["max_window_weight"] <= b["cap"] * (1 + 1e-12)


# ------------------------------------------------------------------------------------------------ the sampler
@pytest.fixture(scope="module")
def table():
    return fake_table()


def draws(sampler, ep):
    return [w for batch in sampler.epoch(ep) for w in batch]


def test_an_epoch_draws_exactly_n_windows_valid_ones_with_the_uniform_age_rule(table):
    windows, feats, _ = table
    s = D.DiversitySampler(windows, 16, feats, seed=1, chunk=8)
    valid = {(w[0], w[1]) for w in windows}
    for ep in range(6):
        batches = list(s.epoch(ep))
        flat = [w for b in batches for w in b]
        assert len(flat) == len(windows) and all(len(b) <= 16 for b in batches)
        assert all((ci, t) in valid for ci, t, _, _ in flat)
        assert all(0 <= a_t <= s.max_age_t and 0 <= a_s - a_t <= s.max_age_s - s.max_age_t for _, _, a_t, a_s in flat)
    assert draws(s, 2) == draws(s, 2) and draws(s, 2) != draws(s, 3)                                   # seeded by (seed, epoch): resumable
    assert draws(s, 2) != draws(D.DiversitySampler(windows, 16, feats, seed=2, chunk=8), 2)
    assert draws(s, 0) != draws(D.ChunkedSampler(windows, 16, seed=1, chunk=8), 0)                      # a different stream from the uniform sampler's


def test_io_locality_is_kept_a_run_of_batches_touches_about_one_chunk_of_clips(table):
    windows, feats, _ = table
    s = D.DiversitySampler(windows, 16, feats, seed=1, chunk=8)
    batches = list(s.epoch(0))
    worst = max(len({w[0] for b in batches[i:i + 6] for w in b}) for i in range(0, len(batches) - 6))   # 6 consecutive batches = 96 draws
    assert worst <= 3 * 8, worst                                                                       # a few chunks at most (a low-mass chunk gets few draws, so a run can cross two boundaries); a global draw touches ~47 of the 60 clips


def test_the_chunk_allocation_makes_the_expected_draws_exact_whatever_the_chunk_composition(table):
    windows, feats, cells = table
    s = D.DiversitySampler(windows, 16, feats, seed=0, chunk=8)
    row = {(w[0], w[1]): i for i, w in enumerate(windows)}
    cnt = np.zeros(25)
    n_ep = 40
    for ep in range(n_ep):
        for ci, t, _, _ in draws(s, ep):
            cnt[cells[row[(ci, t)]]] += 1
    share = cnt / cnt.sum()
    tgt = D.divrs_weights(cells)["target_share"]
    big = tgt > 0.02
    assert big.sum() >= 3 and np.all(np.abs(share[big] / tgt[big] - 1) < 0.05), (share[big], tgt[big])
    assert cnt[tgt == 0].sum() == 0


# ------------------------------------------------------------------------------------------------ card control 1
def test_control_1_realised_cell_shares_over_2000_batches_are_within_20_percent_of_the_target(table):
    windows, feats, _ = table
    s = D.DiversitySampler(windows, 16, feats, seed=0, chunk=8)
    res = D.check_cell_shares(s.epoch, windows, feats)
    assert res["pass"] and res["n_batches"] == 2000 and res["worst_rel_err_resolved"] <= 0.20 and res["n_resolved"] >= 5 and res["draws_in_zero_target_cells"] == 0
    assert all(r["expected_draws"] < D.DIVRS_MIN_EXPECTED for r in res["cells"] if r["target_share"] > 0 and not r["resolved"])      # an unresolved cell is listed with its expected count, never dropped
    assert {u["cell"] for u in res["unresolved"]} == {r["cell"] for r in res["cells"] if r["target_share"] > 0 and not r["resolved"]}


def test_control_1_fails_for_the_uniform_draw_the_deliberate_regression_arm(table):
    windows, feats, _ = table
    uni = D.ChunkedSampler(windows, 16, seed=0, chunk=8)
    res = D.check_cell_shares(uni.epoch, windows, feats)
    assert not res["pass"] and res["worst_rel_err_resolved"] > 0.5
    worst = max((r for r in res["cells"] if r["resolved"]), key=lambda r: abs(r["rel_err"]))
    assert worst["cell"] == 12 and worst["rel_err"] > 0.5                                                   # the straight, steady cell is drawn far above its target


def test_control_1_fails_for_a_chunk_allocation_by_size_and_for_a_draw_without_weights_or_cap(table, monkeypatch):
    windows, feats, _ = table
    # (a) every chunk gets draws in proportion to its SIZE (the per-chunk renormalisation): the straight-dominated chunks are over-drawn
    class BySize(D.DiversitySampler):
        def epoch(self, ep):
            rng = np.random.default_rng((self.seed, ep, 0xD1D5))
            by_clip = {}
            for i, w in enumerate(self.windows):
                by_clip.setdefault(w[0], []).append(i)
            order = rng.permutation(sorted(by_clip))
            for c0 in range(0, len(order), self.chunk):
                ch = np.array([i for c in order[c0:c0 + self.chunk] for i in by_clip[int(c)]])
                pool = rng.choice(ch, size=len(ch), replace=True, p=self.p[ch] / self.p[ch].sum())
                for b0 in range(0, len(pool), self.batch):
                    yield [(self.windows[int(i)][0], self.windows[int(i)][1], 0, 0) for i in pool[b0:b0 + self.batch]]
    # (the distortion is a ratio bias that grows as chunks shrink; chunk = 2 clips makes it plain, and the mass-proportional allocation must still pass there)
    assert D.check_cell_shares(D.DiversitySampler(windows, 16, feats, seed=0, chunk=2).epoch, windows, feats)["pass"]
    assert not D.check_cell_shares(BySize(windows, 16, feats, seed=0, chunk=2).epoch, windows, feats)["pass"]
    # (b) no cap: the target the check computes (capped) is not what an uncapped sampler draws
    assert not D.check_cell_shares(D.DiversitySampler(windows, 16, feats, seed=0, chunk=8, cap_mult=np.inf).epoch, windows, feats)["pass"]
    # (c) the weights not applied: the draw inside a chunk is uniform over its windows
    s = D.DiversitySampler(windows, 16, feats, seed=0, chunk=8)
    s.p = np.ones(len(windows))
    assert not D.check_cell_shares(s.epoch, windows, feats)["pass"]


# ------------------------------------------------------------------------------------------------ the trainer half
@pytest.fixture(scope="module")
def init(tmp_path_factory):
    root = tmp_path_factory.mktemp("divrs_tr")
    assert TR.main(["--out", str(root / "m2b"), *BASE, "--log-every", "1", "--ckpt-every", "100", "--steps", "2"]) == 0
    ck = root / "m2b" / "ckpt_last.pt"
    return {"root": root, "ck": ck, "blob": torch.load(ck, map_location="cpu", weights_only=False)}


def gate(init, name, extra=(), ck=True, base=BASE):
    try:
        rc = TR.main(["--out", str(init["root"] / name), *base, *(["--init-ckpt", str(init["ck"])] if ck else []), *extra, "--selfcheck-only"])
    except SystemExit as e:
        rc = e.code
    return rc


def sc_of(init, name):
    return json.loads((init["root"] / name / "selfcheck.json").read_text(encoding="utf-8"))


def refused(init, name, extra, ck=True, base=BASE, key="ZZDVB-REFUSEDZZ"):
    rc = gate(init, name, extra, ck, base)
    assert isinstance(rc, str) and key in rc, (name, rc)
    return rc


def test_the_flag_has_a_declared_vs_built_entry_and_defaults_to_uniform():
    dests = {act.dest for act in TR.build_parser()._actions if act.dest != "help"}
    assert "wm_sampler" in dests and dests == set(TR.FLAG_CHECKS) and TR.FLAG_CHECKS["wm_sampler"] == "data:wm_sampler"
    assert TR.build_parser().parse_args(["--out", "x"]).wm_sampler == "uniform"
    assert TR.build_parser().parse_args(["--out", "x", "--wm-sampler", "divrs"]).wm_sampler == "divrs"
    with pytest.raises(SystemExit):
        TR.build_parser().parse_args(["--out", "x", "--wm-sampler", "balanced"])


def test_the_gate_passes_with_divrs_reads_the_built_sampler_back_and_runs_control_1(init):
    assert gate(init, "g_divrs", [*WMO, "--wm-sampler", "divrs"]) == 0
    sc = sc_of(init, "g_divrs")
    rec = sc["levers"]["wm_sampler"]
    assert rec["kind"] == "divrs" and rec["yaw_edges"] == [0.02, 0.05, 0.10, 0.20] and rec["acc_edges"] == [-1.0, -0.3, 0.3, 1.0] and rec["cap_mult"] == 10.0
    assert sum(rec["cell_counts"]) == rec["n_windows"] == sc["info"]["n_windows"] and abs(sum(rec["target_share"]) - 1.0) < 1e-9
    pr = sc["wm_sampler_probe"]
    assert pr["pass"] and pr["n_batches"] == 2000 and pr["worst_rel_err_resolved"] <= 0.20 and pr["n_resolved"] >= 1
    assert sc["live"]["pass"] and sc["live"]["control_nonwm_grad_sum"] == 0.0


def test_an_argv_without_the_flag_builds_the_uniform_sampler_and_keeps_the_selfcheck_key_set(init):
    assert gate(init, "g_uni", WMO) == 0
    sc = sc_of(init, "g_uni")
    assert "wm_sampler" not in sc["levers"] and "wm_sampler_probe" not in sc and sc["info"]["wm_sampler"]["kind"] == "uniform" and sc["info"]["wm_sampler"]["class"] == "ChunkedSampler"


def test_the_gate_refuses_every_lie(init, monkeypatch):
    assert "yaw" in refused(init, "r_steer", [*WMO, "--wm-sampler", "divrs", "--kin-mode", "steer"])
    assert "no world model" in refused(init, "r_wm_off", ["--wm", "off", "--wm-sampler", "divrs"], ck=False, base=[b for b in BASE if b not in ("--arm", "M2b")] + ["--arm", "M1"])
    # the deliberate-regression arm: the flag is set, the sampler built is the uniform one
    monkeypatch.setattr(TR, "make_sampler", lambda a, cfg, wins, store, J, info: D.ChunkedSampler(wins, a.batch, seed=a.seed, chunk=max(4, min(48, info["lru"] - 2)), max_age_t=cfg.max_age_t, max_age_s=cfg.max_age_s))
    assert "declared divrs but built uniform" in refused(init, "r_uniform_built", [*WMO, "--wm-sampler", "divrs"])
    monkeypatch.undo()
    # a bin edge drifted in the data module: the sampler reports its own edges, the gate compares them with the card's
    monkeypatch.setattr(D, "DIVRS_YAW_EDGES", (0.02, 0.05, 0.10, 0.25))
    assert "yaw_edges" in refused(init, "r_edge", [*WMO, "--wm-sampler", "divrs"])
    monkeypatch.undo()
    monkeypatch.setattr(D, "DIVRS_CAP_MULT", 100.0)
    assert "cap_mult" in refused(init, "r_cap", [*WMO, "--wm-sampler", "divrs"])


def test_the_control_1_probe_refuses_a_sampler_whose_record_says_divrs_but_whose_draws_are_uniform(init, monkeypatch):
    """G-LIVE of the lever: the record is right (the gate passes), the draw is the uniform one (the line that applies the weights is gone); on a skewed cell table control 1 must refuse the launch."""
    skew = np.random.default_rng(0).random
    def skewed(store, J, wins, hmax):
        cells = np.where(skew(len(wins)) < 0.8, 12, np.arange(len(wins)) % 25)
        return feats_of_cells(cells)
    monkeypatch.setattr(D, "window_act_means", skewed)
    assert gate(init, "g_skew_ok", [*WMO, "--wm-sampler", "divrs"]) == 0                                    # the real sampler passes on the skewed table
    monkeypatch.setattr(D.DiversitySampler, "epoch", D.ChunkedSampler.epoch)                                 # ... and the echo (record says divrs, draw is uniform) is refused
    rc = refused(init, "r_echo", [*WMO, "--wm-sampler", "divrs"], key="wm-sampler control 1 failed")
    assert "ZZDVB-REFUSEDZZ" in rc


def test_a_whole_wm_only_run_reaches_the_loss_through_the_diversity_sampler_and_uniform_named_equals_absent(init, monkeypatch):
    seen = {"n": 0, "draws": 0}
    real = D.DiversitySampler.epoch
    def spy(self, ep):
        seen["n"] += 1
        for b in real(self, ep):
            seen["draws"] += len(b)
            yield b
    monkeypatch.setattr(D.DiversitySampler, "epoch", spy)
    common = [*BASE, *WMO, "--init-ckpt", str(init["ck"]), "--steps", "6", "--log-every", "2", "--ckpt-every", "100"]
    outs = {}
    for name, extra in (("absent", []), ("named_uniform", ["--wm-sampler", "uniform"]), ("divrs", ["--wm-sampler", "divrs"])):
        out = init["root"] / f"run_{name}"
        assert TR.main(["--out", str(out), *common, *extra]) == 0
        outs[name] = torch.load(out / "ckpt_last.pt", map_location="cpu", weights_only=False)["model"]
    # the spy ran for the divrs run only (the gate's probe draws 2,000 batches, the run draws 6 batches x 16)
    assert seen["n"] >= 2 and seen["draws"] >= 6 * 16
    a, n, d = outs["absent"], outs["named_uniform"], outs["divrs"]
    assert all(torch.equal(a[k], n[k]) for k in a), "--wm-sampler uniform named must be the flag absent, bit for bit"             # default off = identical
    assert any(not torch.equal(a[k], d[k]) for k in a if k.startswith("wm.")), "divrs did not move the world model"            # the lever reaches the WM loss
    base = init["blob"]["model"]
    rest = [k for k in base if not k.startswith("wm.")]
    assert rest and all(torch.equal(base[k], d[k]) for k in rest), "a non-WM tensor moved under divrs"                           # the planner is untouched
    log = [json.loads(ln) for ln in (init["root"] / "run_divrs" / "train_log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert log and all(np.isfinite(r["wm"]) for r in log)


def test_the_flag_reaches_a_full_m2b_run_through_the_loss(init):
    out = init["root"] / "full_divrs"
    assert TR.main(["--out", str(out), *BASE, "--wm-sampler", "divrs", "--steps", "2", "--log-every", "1", "--ckpt-every", "100"]) == 0
    sc = json.loads((out / "selfcheck.json").read_text(encoding="utf-8"))
    assert sc["levers"]["wm_sampler"]["kind"] == "divrs" and sc["wm_sampler_probe"]["pass"]
    s = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert s["done"] is True and s["levers"]["wm_sampler"]["kind"] == "divrs"
