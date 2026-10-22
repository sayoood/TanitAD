"""v8F-mini LONGITUDINAL distance keeping (code/v8f_dk_read.py): the reader that finally CALLS the programme's distance-keeping instrument
(`taniteval.lead_metrics`, BACKLOG R1) on the v8F-mini eval-141 grid. CPU only; no torch, no checkpoint, no repo data -- a synthetic straight-road world
whose answers are analytic.

What is pinned, and the control that must go red for each:
  * ANALYTIC. The ego drives v0 = 10 m/s behind a lead at 8 m/s with a rear-face gap g0 at t0. The human path's gap shrinks 0.4 m per 0.2 s step, so
    headway = g0 - 0.4 K, time-gap = headway / v0, min-TTC = (g0 - 0.4 (K - 1)) / 2.0 m/s. The reader must return exactly those.
  * EXACT ZERO. Two arms with identical paths give a delta of EXACTLY 0.0 [0.0, 0.0] on every metric, on the abs-error-vs-human contrast and on LEAD-VIOL.
  * DELIBERATE REGRESSION. A plan at 13 m/s that closes on the lead must read WORSE than the human-speed plan: lower headway, lower min-TTC, MORE
    LEAD-VIOL, farther from the human -- all separated; while the human path itself reads LEAD-VIOL 0 by construction.
  * NO LABEL IS NOT FREE FLOW. NO_LEAD / NOT_STRAIGHT / NO_LABEL windows are counted with their reasons and never enter a mean, even when the block
    carries a lead track on them; the time-gap is undefined (NaN, counted) at standstill.
  * THE JOIN IS BY KEY, never by position: shuffling the block's rows changes nothing.
  * REFUSALS (rc 2, with the message): missing eval / lead-block / P10 file, a positional block, a horizon-grid mismatch, a grid window with no block row,
    a failed speed proof, a failed kinematic-contract check, a pred in the wrong units, a pred that does not reproduce its saved ade6, an eid mismatch, a
    P10 that does not reproduce floors_perwindow.npz, a K above the block's.

Mutation check (run by hand, results in RESULT_DK.md): deleting the line that computes the gap (`lead_metrics.per_step_gap`), or the reader's lead
assignment / speed proof / contract check, must turn these tests RED.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
if not (REPO / "taniteval").is_dir():                                   # the scratch layout: <pkg>/tests, <pkg>/code, <pkg>/tree
    REPO = HERE.parents[1] / "tree"
for _p in (REPO / "stack", REPO / "taniteval"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

PKG = REPO / "TanitAD Research Lab" / "Architecture & Inference" / "Research" / "2026-10-17-v8f-mini"
READER = Path(os.environ.get("V8F_DK_READER") or (PKG / "code" / "v8f_dk_read.py"))
if not READER.exists():
    READER = HERE.parents[1] / "code" / "v8f_dk_read.py"
pytestmark = pytest.mark.skipif(not READER.exists(), reason="the v8F-mini package (code/v8f_dk_read.py) is not in this checkout")

from taniteval import lead_metrics as LM  # noqa: E402


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


R = _load("v8f_dk_read_under_test", READER) if READER.exists() else None     # importable without the package (the lander's stack-only probe)

DT, K, L = 0.2, 10, 4.5
V0, VL = 10.0, 8.0
NB = 120


# ============================================================================================ the synthetic world
def path_at(speed, n, kk=30):
    j = np.arange(1, kk + 1, dtype=np.float64)
    p = np.zeros((n, kk, 2))
    p[:, :, 0] = np.asarray(speed, np.float64).reshape(-1, 1) * DT * j
    return p


class World:
    """12 episodes x 6 windows on a straight road. Window w: clip ``clip-{e}``, cache step t = 3 + 4 i, RAW frame 2t."""

    def __init__(self, tmp, n_ep=12, n_win=6, g0=None, v0=V0, vl=VL, states=None, v0_over=None, block_k=K, lead_on_all=True):
        self.tmp = Path(tmp)
        self.n_ep, self.n_win = n_ep, n_win
        n = n_ep * n_win
        self.n = n
        self.eid = np.repeat(np.arange(n_ep), n_win)
        self.clip = np.array([f"clip-{e:03d}" for e in self.eid])
        self.t = np.tile(3 + 4 * np.arange(n_win), n_ep)
        self.frame = 2 * self.t
        self.g0 = np.asarray(g0 if g0 is not None else 14.5 + (np.arange(n) * 7 % 31) * 0.5, np.float64)
        self.v0 = np.full(n, v0, np.float64)
        if v0_over:
            for i, v in v0_over.items():
                self.v0[i] = v
        self.vl = np.full(n, vl, np.float64)
        self.states = np.array(["LEAD"] * n, dtype="U12")
        for i, s in (states or {}).items():
            self.states[i] = s
        self.gt = path_at(self.v0, n)
        self.block_k = block_k
        self.lead_on_all = lead_on_all

    # --- the lead block (one row per window + distractor rows, shuffled) ------------------------------------------------
    def block_arrays(self, shuffle_seed=None, drop=(), extra_distractors=True):
        n, k = self.n, self.block_k
        j = np.arange(1, k + 1, dtype=np.float64)
        leads = np.full((n, k, 2), np.nan)
        # a real block has NaN tracks / lens on non-LEAD rows; ``lead_on_all`` makes a NO_LABEL row CARRY a track, to test that the reader still refuses to score it
        keep_lead = np.ones(n, bool) if self.lead_on_all else (self.states == "LEAD")
        leads[keep_lead, :, 0] = (self.g0[keep_lead, None] + L / 2.0 + self.vl[keep_lead, None] * DT * j[None])
        leads[keep_lead, :, 1] = 0.0
        lens = np.where(keep_lead, L, np.nan)
        cols = dict(clip_id=self.clip.copy(), frame=self.frame.copy(), leads=leads, lead_lens=lens, speeds=self.v0.copy(), state=self.states.copy())
        hdk = LM.distance_keeping(self.gt[:, :k], np.where((self.states == "LEAD")[:, None, None], leads, np.nan),
                                  np.where(self.states == "LEAD", L, np.nan), self.v0, DT)
        cols["gt_headway_min_m"], cols["gt_time_gap_min_s"], cols["gt_min_ttc_s"] = hdk["headway_min_m"], hdk["time_gap_min_s"], hdk["min_ttc_s"]
        if extra_distractors:                                       # rows the grid never asks for (other frames), so the join must be by KEY
            m = n_extra = 40
            dis = dict(clip_id=np.array([f"clip-{i % self.n_ep:03d}" for i in range(m)]), frame=np.array([1001 + 2 * i for i in range(m)]),
                       leads=np.full((m, k, 2), 7.0), lead_lens=np.full(m, 3.0), speeds=np.full(m, 99.0), state=np.array(["LEAD"] * m, dtype="U12"),
                       gt_headway_min_m=np.full(m, 1.0), gt_time_gap_min_s=np.full(m, 1.0), gt_min_ttc_s=np.full(m, 1.0))
            cols = {kk: np.concatenate([cols[kk], dis[kk]]) for kk in cols}
        keepr = np.ones(len(cols["frame"]), bool)
        for d in drop:
            keepr[d] = False
        cols = {kk: v[keepr] for kk, v in cols.items()}
        if shuffle_seed is not None:
            perm = np.random.default_rng(shuffle_seed).permutation(len(cols["frame"]))
            cols = {kk: v[perm] for kk, v in cols.items()}
        cols["ts_rel_s"] = DT * j
        cols["dt_s"] = np.array([DT])
        cols["meta_json"] = np.frombuffer(json.dumps({"version": "synthetic", "tool": "test", "built_utc": "0"}).encode(), np.uint8)
        return cols

    def write_block(self, name="lead.npz", mut=None, **kw):
        cols = self.block_arrays(**kw)
        if mut:
            cols = mut(cols)
        p = self.tmp / name
        np.savez(p, **cols)
        return p

    def write_gt(self, name="gt.npz"):
        p = self.tmp / name
        np.savez(p, gt=self.gt, eid=self.eid, clip=self.clip, frame=self.frame, v0=self.v0, dt=np.float64(DT))
        return p

    def write_arm(self, d, tag, pred, ade=True, eid=None):
        d = Path(d)
        d.mkdir(parents=True, exist_ok=True)
        kw = dict(pred=pred.astype(np.float32), eid=self.eid if eid is None else eid)
        if ade:
            kw["ade6"] = np.linalg.norm(pred.astype(np.float32).astype(np.float64) - self.gt, axis=-1).mean(1)
        np.savez(d / f"eval_{tag}.npz", **kw)

    def preds(self, speeds):
        return {nm: path_at(np.full(self.n, s), self.n) if np.isscalar(s) else s for nm, s in speeds.items()}

    def read(self, speeds, block=None, k=K, n_boot=NB, **kw):
        blk = R.load_block(block if block is not None else self.write_block())
        pr = self.preds(speeds)
        return R.read_dk(preds=pr, gt=self.gt, eid=self.eid, clip=self.clip, frame=self.frame, v0=self.v0, block=blk, k=k, n_boot=n_boot, **kw)["read"]


FULL = {"M1_s0": 10.0, "M1_s1": 10.0, "M2b_s0": 10.0, "M2b_s1": 10.0, "P10": 10.0}


def human_analytic(w, k=K):
    hw = w.g0 + DT * (VL - V0) * k
    tg = hw / V0
    ttc = (w.g0 + DT * (VL - V0) * (k - 1)) / ((V0 - VL))
    return hw, tg, ttc


# ============================================================================================ analytic / exact-zero / regression
def test_analytic_straight_road_answers(tmp_path):
    w = World(tmp_path)
    r = w.read(FULL)
    hw, tg, ttc = human_analytic(w)
    h = r["human_reference"]["metrics"]
    assert h["headway_min_m"]["n"] == w.n and h["headway_min_m"]["mean"] == pytest.approx(hw.mean(), abs=1e-4)
    assert h["time_gap_min_s"]["mean"] == pytest.approx(tg.mean(), abs=1e-4)
    assert h["min_ttc_s"]["mean"] == pytest.approx(ttc.mean(), abs=1e-4) and h["min_ttc_s"]["n_closing"] == w.n
    # a plan at the human's own speed IS the human path: every metric equals the reference, and its distance to the human is exactly 0
    a = r["arms"]["P10"]
    for kk, v in zip(R.DK_KEYS, (hw, tg, ttc)):
        assert a["metrics"][kk]["mean"] == pytest.approx(v.mean(), abs=1e-4)
        assert a["vs_human"][kk]["abs_error"]["mean"] == 0.0
    assert a["lead_viol"]["share"] == 0.0 and a["lead_viol"]["n_viol"] == 0, "the human path reads LEAD-VIOL 0 by construction"
    assert r["human_reference"]["kinematic_contract"]["status"] == "OK" and r["human_reference"]["kinematic_contract"]["median_abs_headway_diff_m"] == 0.0
    # the 6 s = 30 step horizon on a 30-step block reads the analytic 30-step answers too
    (tmp_path / "k30").mkdir()
    w30 = World(tmp_path / "k30", block_k=30)
    r30 = w30.read(FULL, k=30)
    hw30, _, _ = human_analytic(w30, 30)
    assert r30["human_reference"]["metrics"]["headway_min_m"]["mean"] == pytest.approx(hw30.mean(), abs=1e-4)
    assert hw30.min() < hw.min(), "a longer horizon sees a tighter minimum"


def test_identical_arms_give_an_exact_zero(tmp_path):
    w = World(tmp_path)
    r = w.read({**FULL, "M2b_s0": 11.0, "P10": 11.0})            # M2b_s0 and P10 are the SAME path
    p = r["pairs"]["M2b_s0__minus__P10"]
    for kk in R.DK_KEYS:
        d = p["dk"][kk]
        assert d["delta"] == 0.0 and d["lo"] == 0.0 and d["hi"] == 0.0, kk
        e = p["abs_error_vs_human"][kk]
        assert e["delta"] == 0.0 and e["lo"] == 0.0 and e["hi"] == 0.0, kk
    for blk in (p["lead_viol"]["all_lead_windows"], p["lead_viol"]["at_risk"]):
        assert blk["delta"] == 0.0 and blk["lo"] == 0.0 and blk["hi"] == 0.0
    ic = r["identity_control"]
    assert all(ic[kk]["delta"] == 0.0 and ic[kk]["lo"] == 0.0 and ic[kk]["hi"] == 0.0 for kk in R.DK_KEYS) and ic["lead_viol"]["delta"] == 0.0


def test_a_broken_paired_instrument_is_caught_by_the_identity_control(tmp_path, monkeypatch):
    """The reader pairs an arm with ITSELF and refuses unless every metric reads exactly +0 [0, 0]: a paired bootstrap that returns noise for identical
    inputs would otherwise print separated deltas for nothing."""
    w = World(tmp_path)
    blk = R.load_block(w.write_block())
    real = LM.paired_distance_keeping

    def noisy(dk_a, dk_b, eid, **kw):
        out = real(dk_a, dk_b, eid, **kw)
        for v in out["metrics"].values():
            if v.get("status") == "OK":
                v["delta"], v["lo"], v["hi"] = 1e-3, 5e-4, 2e-3
        return out
    monkeypatch.setattr(LM, "paired_distance_keeping", noisy)
    with pytest.raises(R.Refusal, match="IDENTITY CONTROL FAILED"):
        R.read_dk(preds=w.preds(FULL), gt=w.gt, eid=w.eid, clip=w.clip, frame=w.frame, v0=w.v0, block=blk, k=K, n_boot=NB)


def test_deliberate_regression_closing_on_the_lead_reads_worse(tmp_path):
    w = World(tmp_path)
    r = w.read({**FULL, "M2b_s0": 13.0, "M2b_s1": 13.0})
    reg, base = r["arms"]["M2b_s0"], r["arms"]["M1_s0"]
    assert reg["metrics"]["headway_min_m"]["mean"] < base["metrics"]["headway_min_m"]["mean"] - 5.0
    assert reg["metrics"]["min_ttc_s"]["mean"] < base["metrics"]["min_ttc_s"]["mean"] - 5.0
    p = r["pairs"]["M2b_s0__minus__P10"]
    assert p["dk"]["headway_min_m"]["delta"] == pytest.approx(-6.0, abs=1e-6) and p["dk"]["headway_min_m"]["separated"]
    assert p["dk"]["min_ttc_s"]["delta"] < 0 and p["dk"]["min_ttc_s"]["separated"]
    assert p["dk"]["time_gap_min_s"]["delta"] < 0 and p["dk"]["time_gap_min_s"]["separated"]
    assert p["abs_error_vs_human"]["headway_min_m"]["delta"] > 0 and p["abs_error_vs_human"]["headway_min_m"]["separated"]
    lv = p["lead_viol"]["at_risk"]
    assert lv["delta"] > 0 and lv["separated"] and lv["x7_direction_met_at_risk"] is False, "MORE violations"
    assert reg["lead_viol"]["share_of_at_risk"] > 0.2 and base["lead_viol"]["share_of_at_risk"] == 0.0
    # the human-speed plan keeps the progress control at exactly 1; the 13 m/s plan out-runs the human (clipped at 1.0)
    assert base["lead_viol"]["lead_progress_ratio_clipped"] == pytest.approx(1.0) and base["lead_viol"]["progress"]["meets_min_progress"] is True
    # the programme's rule for an effect: separated AND beyond 2F at both seeds (F = 0 here: both seeds identical)
    assert r["exceeds_2F_both_seeds"]["M2b_minus_P10"]["headway_min_m"] is True
    assert r["exceeds_2F_both_seeds"]["M1_minus_P10"]["headway_min_m"] is False


def test_a_crawling_plan_fails_the_progress_control(tmp_path):
    w = World(tmp_path)
    r = w.read({**FULL, "M1_s0": 3.0})                           # far slower than the human: no violation, but it did not drive
    lv = r["arms"]["M1_s0"]["lead_viol"]
    assert lv["share_of_at_risk"] == 0.0
    assert lv["progress"]["meets_min_progress"] is False and lv["progress"]["lead_progress_ratio_clipped"] == pytest.approx(0.3, abs=1e-6)


# ============================================================================================ labels are not free flow
def test_no_label_never_enters_a_mean_and_every_window_is_accounted(tmp_path):
    states = {0: "NO_LEAD", 1: "NOT_STRAIGHT", 2: "NO_LABEL", 3: "NO_LABEL", 7: "NO_LEAD"}
    w = World(tmp_path, states=states, lead_on_all=True)          # the block carries a lead TRACK on those rows; the reader must not score it
    r = w.read(FULL)
    lead = np.array([s == "LEAD" for s in w.states])
    hw, tg, ttc = human_analytic(w)
    assert r["join"]["window_states"] == {"LEAD": int(lead.sum()), "NO_LEAD": 2, "NOT_STRAIGHT": 1, "NO_LABEL": 2}
    assert sum(r["join"]["window_states"].values()) == w.n
    m = r["arms"]["P10"]["metrics"]["headway_min_m"]
    assert m["n"] == int(lead.sum()) and m["mean"] == pytest.approx(hw[lead].mean(), abs=1e-4)
    rs = r["arms"]["P10"]["no_value_reasons"]["headway_min_m"]
    assert (rs["NO_LEAD"], rs["NOT_STRAIGHT"], rs["NO_LABEL"], rs["n_with_value"]) == (2, 1, 2, int(lead.sum()))
    assert r["arms"]["P10"]["n_lead_windows"] == int(lead.sum())


def test_time_gap_is_undefined_at_standstill_and_counted(tmp_path):
    w = World(tmp_path, v0_over={5: 0.3, 9: 0.2})
    r = w.read(FULL)
    a = r["arms"]["P10"]
    assert a["metrics"]["time_gap_min_s"]["n"] == a["metrics"]["headway_min_m"]["n"] - 2
    assert a["no_value_reasons"]["time_gap_min_s"]["standstill_v0_below_0p5_mps"] == 2
    assert a["no_value_reasons"]["headway_min_m"].get("standstill_v0_below_0p5_mps") is None


def test_a_plan_that_leaves_the_lane_loses_the_lead_and_is_counted(tmp_path):
    w = World(tmp_path)
    pr = w.preds(FULL)
    lat = pr["M1_s0"].copy()
    lat[:3, :, 1] = 5.0                                           # 3 windows run 5 m to the left of the lead's lane for the whole horizon
    blk = R.load_block(w.write_block())
    r = R.read_dk(preds={**pr, "M1_s0": lat}, gt=w.gt, eid=w.eid, clip=w.clip, frame=w.frame, v0=w.v0, block=blk, k=K, n_boot=NB)["read"]
    a = r["arms"]["M1_s0"]
    assert a["no_value_reasons"]["headway_min_m"]["lead_window_but_path_never_in_corridor"] == 3
    assert a["metrics"]["headway_min_m"]["n"] == w.n - 3
    assert a["lead_viol"]["n_plan_lost_lead"] == 3, "a lane-leaving plan has no time gap: counted, not a free pass"


def test_the_join_is_by_key_not_position(tmp_path):
    w = World(tmp_path)
    a = w.read(FULL, block=w.write_block("a.npz"))
    b = w.read(FULL, block=w.write_block("b.npz", shuffle_seed=3))
    for kk in R.DK_KEYS:
        assert a["arms"]["P10"]["metrics"][kk] == b["arms"]["P10"]["metrics"][kk]
    assert a["join"]["speed_proof_max_mps"] == b["join"]["speed_proof_max_mps"] == 0.0


def test_ttc_stride2_sensitivity_exists_and_matches_the_instrument(tmp_path):
    w = World(tmp_path)
    r = w.read(FULL)
    s = r["ttc_stride2_sensitivity"]
    # constant-speed paths: the closing rate is the same at any spacing, so the stride-2 TTC equals the stride-1 TTC at the SAME instants only up to
    # which step carries the minimum; it must exist, be finite for every arm, and its identity pair must be exactly 0
    assert s["dt_s"] == pytest.approx(0.4) and set(s["arms"]) == set(FULL) | {"human"}
    assert all(np.isfinite(v["mean"]) for v in s["arms"].values())
    assert s["pairs"]["M2b_s0__minus__P10"]["delta"] == 0.0


# ============================================================================================ the CLI and its refusals
def _cli_world(tmp, **kw):
    w = World(tmp, **kw)
    d = Path(tmp) / "eval"
    pr = {"M1_s0": 9.0, "M1_s1": 9.2, "M2b_s0": 10.0, "M2b_s1": 10.5}
    for tag, s in pr.items():
        w.write_arm(d, tag, path_at(np.full(w.n, s), w.n))
    p10 = path_at(np.full(w.n, 10.0), w.n).astype(np.float32)
    np.save(Path(tmp) / "p10.npy", p10)
    np.savez(d / "floors_perwindow.npz", P10_refit__ade6=np.linalg.norm(p10.astype(np.float64) - w.gt, axis=-1).mean(1))
    return w, d


def _main(w, d, block, extra=(), out=None, gt=None):
    out = out or (Path(d).parent / "out.json")
    argv = ["--eval-dir", str(d), "--out", str(out), "--lead-block", str(block), "--p10-pred", str(Path(d).parent / "p10.npy"),
            "--gt-npz", str(gt or w.write_gt()), "--n-boot", "60", *extra]
    return R.main(argv), out


def test_cli_end_to_end_is_strict_json_and_stamps_the_tier(tmp_path, capsys):
    w, d = _cli_world(tmp_path)
    rc, out = _main(w, d, w.write_block(), extra=["--perwindow-out", str(tmp_path / "pw.npz")])
    assert rc == 0, capsys.readouterr().err

    def _no_const(c):
        raise ValueError(f"non-strict JSON constant {c}")
    j = json.loads(out.read_text(encoding="utf-8"), parse_constant=_no_const)
    assert j["schema"] == R.SCHEMA and "T1-class open loop" in j["tier"] and "episode" in j["estimator"] and "k10" in j["reads"]
    assert j["no_bar_registered"].startswith("no distance-keeping bar")
    r = j["reads"]["k10"]
    assert set(r["arms"]) == {"M1_s0", "M1_s1", "M2b_s0", "M2b_s1", "P10"} and len(r["pairs"]) == 6
    assert r["join"]["n_windows"] == w.n and r["join"]["n_episodes"] == w.n_ep
    pw = np.load(tmp_path / "pw.npz")
    assert pw["k10__P10__headway_min_m"].shape == (w.n,) and pw["k10__human__headway_min_m"].shape == (w.n,)
    assert "taniteval" in r["code"]["lead_metrics"].replace("\\", "/")


@pytest.mark.parametrize("name,setup,needle", [
    ("missing_block", lambda w, d: dict(block=Path(w.tmp) / "nope.npz"), "does not exist"),
    ("positional_block", lambda w, d: dict(block=w.write_block("p.npz", mut=lambda c: {k: v for k, v in c.items() if k != "clip_id"})), "not a per-frame B1 block"),
    ("horizon_grid", lambda w, d: dict(block=w.write_block("h.npz", mut=lambda c: {**c, "ts_rel_s": 0.1 * np.arange(1, K + 1)})), "horizon grid"),
    ("window_without_row", lambda w, d: dict(block=w.write_block("r.npz", drop=[4])), "GRID MISMATCH"),
    ("speed_proof", lambda w, d: dict(block=w.write_block("s.npz", mut=lambda c: {**c, "speeds": c["speeds"] + 0.5})), "SPEED PROOF"),
    ("contract_broken", lambda w, d: dict(block=w.write_block("c.npz", mut=lambda c: {**c, "gt_headway_min_m": c["gt_headway_min_m"] + 40.0})), "KINEMATIC-CONTRACT"),
    ("k_above_block", lambda w, d: dict(block=w.write_block("k.npz"), extra=["--horizon-k", "20"]), "horizon grid"),
    ("missing_arm", lambda w, d: (os.remove(Path(d) / "eval_M2b_s1.npz"), dict(block=w.write_block()))[1], "eval_M2b_s1.npz is missing"),
    ("eid_mismatch", lambda w, d: (w.write_arm(d, "M1_s1", path_at(np.full(w.n, 9.0), w.n), eid=w.eid[::-1].copy()), dict(block=w.write_block()))[1], "eid differs"),
    ("ade_mismatch", lambda w, d: (np.savez(Path(d) / "eval_M1_s0.npz", pred=path_at(np.full(w.n, 9.0), w.n).astype(np.float32), eid=w.eid,
                                           ade6=np.zeros(w.n) + 0.5), dict(block=w.write_block()))[1], "saved ade6"),
    ("wrong_units", lambda w, d: (w.write_arm(d, "M1_s0", 1000.0 * path_at(np.full(w.n, 9.0), w.n), ade=False), dict(block=w.write_block()))[1], "first-step"),
    ("p10_not_floor", lambda w, d: (np.savez(Path(d) / "floors_perwindow.npz", P10_refit__ade6=np.zeros(w.n) + 1.0), dict(block=w.write_block()))[1], "P10_refit__ade6"),
    ("missing_p10", lambda w, d: (os.remove(Path(w.tmp) / "p10.npy"), dict(block=w.write_block()))[1], "P10 predictions"),
])
def test_refusals_exit_2_with_a_clear_message(tmp_path, capsys, name, setup, needle):
    w, d = _cli_world(tmp_path)
    kw = setup(w, d)
    rc, out = _main(w, d, kw["block"], extra=kw.get("extra", ()))
    err = capsys.readouterr().err
    assert rc == 2 and ("REFUSED" in err or "INPUT ERROR" in err), (name, rc, err)
    assert needle in err, (name, err)
    assert not out.exists(), "a refused read writes nothing"


def test_a_missing_gt_input_is_a_refusal_not_a_traceback(tmp_path, capsys):
    w, d = _cli_world(tmp_path)
    rc, _ = _main(w, d, w.write_block(), gt=tmp_path / "no_gt.npz")
    assert rc == 2 and "does not exist" in capsys.readouterr().err
