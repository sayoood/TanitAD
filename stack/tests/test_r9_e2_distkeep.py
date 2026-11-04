"""Tests for ``stack/scripts/r9_e2_distkeep.py`` -- the LONGITUDINAL distance-keeping rows of the refcv9 E2 judge.  Synthetic dumps + lead blocks with hand-computable answers.

Geometry of the fixture (hand-computable): the ego drives straight along +x at speed s (window origin frame, rig origin), human path x_k = s * 0.5 * (k + 1) at 0.5 / 1.0 / 1.5 / 2.0 s.
The lead is STATIONARY on the lane, lead_len 4 m, REAR face at 3.4 * s (centre 3.4 * s + 2).  An arm = a path scale factor over the human path.
  s = 10, scale 1.0 : gaps 29 24 19 14 -> headway 14, time gap 1.4 s, TTC min 1.9 s, human at-risk (tg >= 1.0)
  s = 10, scale 1.6 : gaps 26 18 10  2 -> headway  2, time gap 0.2 s, TTC min 0.625 s (rate 16 m/s), arc 32 / 20 = 1.6
RED arms: an arm that closes on the lead reads separated WORSE in the JUDGE (clause d FAIL); identical arms read exactly 0 with CI [0, 0]; a lead-less set reads NOT-COMPUTABLE with its n;
a window-set mismatch / GT disagreement / failed speed proof / wrong families.json refuses.
"""
from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

import tanitad

_ROOT = Path(tanitad.__file__).resolve().parents[2]           # <tree>/stack
sys.path.insert(0, str(_ROOT / "scripts"))
import r9_e2_distkeep as D                                      # noqa: E402
import r9_e2_verdict as V                                       # noqa: E402

N_DEV, WS, K = 12, (10, 20, 30, 40), 4
OFF = 1                                                         # provider -> raw frame offset
DEV_CLIPS = [f"clip{e:02d}" for e in range(N_DEV)]
HOLD_CLIPS = ["clipH0", "clipH1"]
LEAD_ROWS = ("LON_dk_headway_min_m", "LON_dk_time_gap_min_s", "LON_dk_min_ttc_s", "LON_dk_headway_abs_err_m", "LON_dk_ttc_viol_rate", "LON_dk_lead_viol_at_risk",
             "LON_dk_lead_progress_shortfall")


def speed_of(clip: str) -> float:
    e = int(clip[-2:]) if clip.startswith("clip") and clip[-2:].isdigit() else 0
    return 8.0 + 2.0 * (e % 4)


def human(s: float) -> np.ndarray:
    return np.stack([s * 0.5 * (np.arange(K) + 1.0), np.zeros(K)], -1)


def write_dump(root: Path, name: str, scale: float, clips=tuple(DEV_CLIPS + HOLD_CLIPS), g_shift: float = 0.0, raw_off: int = OFF) -> Path:
    d = root / name
    d.mkdir(parents=True)
    eps = []
    for fi, c in enumerate(clips):
        s = speed_of(c)
        g = np.repeat(human(s)[None], len(WS), 0) + g_shift
        np.savez(d / f"ep{fi:03d}.npz", g=g, os=g * scale, ws=np.asarray(WS), v0=np.full(len(WS), s), eid=np.asarray([fi]))
        eps.append({"file_index": fi, "clip_id": c, "n_windows": len(WS)})
    (d / "manifest.json").write_text(json.dumps({"episodes": eps, "grid": {"dt_s": 0.5, "k": K, "n_windows": len(WS) * len(clips)},
                                                 "corpus": {"frames": {"provider_to_raw_frame_offset": raw_off}}}), encoding="utf-8")
    return d


def write_block(path: Path, *, state: str = "LEAD", dt: float = 0.5, k: int = K, clips=tuple(DEV_CLIPS + HOLD_CLIPS), speed_bias: float = 0.0, frame_shift: int = 0) -> Path:
    cid, fr, leads, lens, sp, st = [], [], [], [], [], []
    ts = dt * (np.arange(k) + 1.0)
    for c in clips:
        s = speed_of(c)
        for w in WS:
            cid.append(c)
            fr.append(w + OFF + frame_shift)
            leads.append(np.tile([3.4 * s + 2.0, 0.0], (k, 1)) if state == "LEAD" else np.full((k, 2), np.nan))
            lens.append(4.0 if state == "LEAD" else np.nan)
            sp.append(s + speed_bias)
            st.append(state)
    np.savez(path, clip_id=np.asarray(cid), frame=np.asarray(fr, dtype=np.int64), ts_rel_s=ts, leads=np.asarray(leads), lead_lens=np.asarray(lens), speeds=np.asarray(sp),
             state=np.asarray(st), eid=np.asarray(cid))
    return path


@pytest.fixture()
def dev():
    import r9_e2_families as F
    return {F.sha12(c) for c in DEV_CLIPS}


@pytest.fixture()
def block(tmp_path):
    return write_block(tmp_path / "block_dt0p5_k4.npz")


def run(tmp_path, dev, block, a_scale, b_scale, **kw):
    a = write_dump(tmp_path, f"a_{a_scale}_{b_scale}", a_scale)
    b = write_dump(tmp_path, f"b_{a_scale}_{b_scale}", b_scale)
    return D.build(a, b, dev=dev, lead_block=block, n_boot=300, **kw), a, b


def fams_json(doc, a, b):
    ok = {"delta": -0.1, "lo": -0.2, "hi": -0.05, "separated": True, "better": "lower", "n": 10, "n_episodes": 5}
    return {"mode": "families", "labels": doc["labels"], "n_windows_paired": doc["n_windows_paired"], "a": {"dir": str(a)}, "b": {"dir": str(b)},
            "rows": {"LONGITUDINAL": {"LON_speed_mae_mps": dict(ok)}, "LATERAL": {"LAT_heading_mae_deg": dict(ok)}, "TACTICAL": {"TAC_traj_lat_correct": {**ok, "better": "higher", "delta": 0.1,
                                                                                                                                         "lo": 0.05, "hi": 0.2}}}}


# ---- hand-computable per-window values -------------------------------------------------------------------------------------------------------------------
def one_window(scale, state="LEAD", s=10.0, lead_x=36.0, lens=4.0):
    g = human(s)[None]
    return {"P": g * scale, "G": g, "leads": np.tile([lead_x, 0.0], (1, K, 1)), "lens": np.asarray([lens]), "speeds": np.asarray([s]), "state": np.asarray([state]),
            "cols": list(range(K)), "dt": 0.5}


def test_hand_computed_window_values_closing_arm():
    v, ex = D.window_values(one_window(1.6))
    assert v["LON_dk_headway_min_m"][0] == pytest.approx(2.0)
    assert v["LON_dk_time_gap_min_s"][0] == pytest.approx(0.2)
    assert v["LON_dk_min_ttc_s"][0] == pytest.approx(0.625)
    assert v["LON_dk_headway_abs_err_m"][0] == pytest.approx(12.0)             # |2 - 14|
    assert v["LON_dk_ttc_viol_rate"][0] == 1.0 and v["LON_dk_lead_viol_at_risk"][0] == 1.0
    assert v["LON_dk_lead_progress_shortfall"][0] == 0.0 and ex["mean_progress_ratio"] == pytest.approx(1.6)      # over-progress is not a shortfall
    assert ex["n_lead_windows"] == 1 and ex["n_at_risk"] == 1


def test_hand_computed_window_values_human_path():
    v, _ = D.window_values(one_window(1.0))
    assert v["LON_dk_headway_min_m"][0] == pytest.approx(14.0) and v["LON_dk_time_gap_min_s"][0] == pytest.approx(1.4)
    assert v["LON_dk_min_ttc_s"][0] == pytest.approx(1.9) and v["LON_dk_headway_abs_err_m"][0] == pytest.approx(0.0)
    assert v["LON_dk_ttc_viol_rate"][0] == 0.0 and v["LON_dk_lead_viol_at_risk"][0] == 0.0 and v["LON_dk_lead_progress_shortfall"][0] == 0.0


@pytest.mark.parametrize("state", ["NO_LEAD", "NOT_STRAIGHT", "NO_LABEL"])
def test_leadless_window_is_nan_never_zero(state):
    w = one_window(1.6, state=state)
    w["leads"][:] = np.nan
    w["lens"][:] = np.nan
    v, ex = D.window_values(w)
    assert all(np.isnan(v[r][0]) for r in LEAD_ROWS), {r: v[r][0] for r in LEAD_ROWS}
    assert ex["n_lead_windows"] == 0


def test_a_state_other_than_LEAD_never_scores_even_if_a_track_is_present():
    v, _ = D.window_values(one_window(1.6, state="NOT_STRAIGHT"))               # the track is there, the label says the bend hides it: NOT-COMPUTABLE
    assert all(np.isnan(v[r][0]) for r in LEAD_ROWS)


def test_standstill_has_no_time_gap_and_the_window_is_dropped_from_that_row_only():
    v, _ = D.window_values(one_window(1.0, s=0.2, lead_x=60.0))
    assert math.isnan(v["LON_dk_time_gap_min_s"][0]) and math.isfinite(v["LON_dk_headway_min_m"][0])


# ---- the pipeline: join + DEV + pairing + bootstrap --------------------------------------------------------------------------------------------------------
def test_RED_closing_arm_reads_separated_worse_and_the_judge_fails_clause_d(tmp_path, dev, block):
    doc, a, b = run(tmp_path, dev, block, 1.6, 1.0)                              # K2 closes on the lead, K0 holds the human gap
    rows = doc["rows"]["LONGITUDINAL"]
    assert doc["n_windows_paired"] == N_DEV * len(WS) and doc["lead"]["grid_complete"] is True and doc["lead"]["n_lead_windows"] == 48
    r = rows["LON_dk_headway_min_m"]
    assert r["better"] == "higher" and r["separated"] and r["delta"] < 0 and r["hi"] < 0
    assert r["a_mean"] == pytest.approx(np.mean([3.4 * s - 3.2 * s for s in (8, 10, 12, 14)]))     # 0.2 s per window, equal episode mix
    assert r["b_mean"] == pytest.approx(np.mean([1.4 * s for s in (8, 10, 12, 14)]))
    for rid in ("LON_dk_time_gap_min_s", "LON_dk_min_ttc_s"):
        assert rows[rid]["separated"] and rows[rid]["delta"] < 0
    for rid in ("LON_dk_headway_abs_err_m", "LON_dk_ttc_viol_rate", "LON_dk_lead_viol_at_risk"):
        assert rows[rid]["better"] == "lower" and rows[rid]["delta"] > 0
    assert rows["LON_dk_lead_viol_at_risk"]["a_mean"] == 1.0 and rows["LON_dk_lead_viol_at_risk"]["b_mean"] == 0.0
    assert rows["LON_dk_lead_progress_shortfall"]["a_mean"] == 0.0 and doc["a"]["mean_progress_ratio"] == pytest.approx(1.6)
    merged = D.merge_into(_write(tmp_path, "fam.json", fams_json(doc, a, b)), doc)
    c, rep = V.families_clause(merged)
    assert c["status"] == "FAIL" and any("LON_dk_headway_min_m" in x for x in c["failing_rows"]), c
    assert rep["LONGITUDINAL"]["status"] == "FAIL"


def test_the_good_direction_passes_the_judge(tmp_path, dev, block):
    doc, a, b = run(tmp_path, dev, block, 1.0, 1.6)                              # K2 keeps distance, K0 closes: nothing separated worse
    assert doc["rows"]["LONGITUDINAL"]["LON_dk_headway_min_m"]["delta"] > 0
    c, _ = V.families_clause(D.merge_into(_write(tmp_path, "fam.json", fams_json(doc, a, b)), doc))
    assert c["status"] == "PASS", c


def test_RED_identical_arms_read_exactly_zero_with_ci_0_0(tmp_path, dev, block):
    doc, a, b = run(tmp_path, dev, block, 1.2, 1.2)
    rows = doc["rows"]["LONGITUDINAL"]
    assert set(rows) == set(LEAD_ROWS) and not any(r.get("undefined") for r in rows.values())
    for rid, r in rows.items():
        assert r["delta"] == 0.0 and r["lo"] == 0.0 and r["hi"] == 0.0 and r["separated"] is False and r["a_mean"] == r["b_mean"], (rid, r)
    c, _ = V.families_clause(D.merge_into(_write(tmp_path, "fam.json", fams_json(doc, a, b)), doc))
    assert c["status"] == "PASS"


def test_RED_leadless_window_set_reads_not_computable_with_its_n(tmp_path, dev):
    blk = write_block(tmp_path / "noleads.npz", state="NO_LEAD")
    doc, a, b = run(tmp_path, dev, blk, 1.6, 1.0)
    rows = doc["rows"]["LONGITUDINAL"]
    assert doc["lead"]["n_lead_windows"] == 0 and doc["lead"]["state_counts_paired_dev"]["NO_LEAD"] == 48
    for rid, r in rows.items():
        assert r["undefined"] is True and r["n"] == 0 and "NOT-COMPUTABLE" in r["reason"] and "delta" not in r, (rid, r)
    assert set(doc["omitted_rows"]) == set(LEAD_ROWS)
    c, rep = V.families_clause(D.merge_into(_write(tmp_path, "fam.json", fams_json(doc, a, b)), doc))
    assert c["status"] == "NOT-COMPUTABLE" and rep["LONGITUDINAL"]["status"] == "NOT-COMPUTABLE"          # never a pass, never a 0


def test_wrong_frame_map_is_not_scored_as_free_flow(tmp_path, dev):
    blk = write_block(tmp_path / "shifted.npz", frame_shift=-OFF)               # block keyed on ws, dump says raw = ws + 1 -> no row for any window
    doc, _, _ = run(tmp_path, dev, blk, 1.6, 1.0)
    assert doc["lead"]["state_counts_paired_dev"]["NO_LABEL"] == 48 and all(r["undefined"] for r in doc["rows"]["LONGITUDINAL"].values())


def test_RED_failed_speed_proof_refuses(tmp_path, dev):
    blk = write_block(tmp_path / "badspeed.npz", speed_bias=0.5)
    with pytest.raises(SystemExit, match="speed proof"):
        run(tmp_path, dev, blk, 1.6, 1.0)


def test_RED_window_set_mismatch_refuses(tmp_path, dev, block):
    a = write_dump(tmp_path, "a", 1.0)
    b = write_dump(tmp_path, "b", 1.0, clips=tuple(DEV_CLIPS[:-1] + HOLD_CLIPS))     # one DEV clip missing in B
    with pytest.raises(SystemExit, match="window-set mismatch"):
        D.build(a, b, dev=dev, lead_block=block, n_boot=100)


def test_RED_gt_disagreement_refuses(tmp_path, dev, block):
    a, b = write_dump(tmp_path, "a", 1.0), write_dump(tmp_path, "b", 1.0, g_shift=0.5)
    with pytest.raises(SystemExit, match="GT"):
        D.build(a, b, dev=dev, lead_block=block, n_boot=100)


def test_hold_windows_are_dropped_and_counted_and_never_scored(tmp_path, dev, block):
    doc, a, _ = run(tmp_path, dev, block, 1.6, 1.0)
    assert doc["a"]["n_non_dev_dropped"] == len(HOLD_CLIPS) * len(WS) and doc["a"]["n_dev"] == N_DEV * len(WS) and doc["n_windows_paired"] == 48
    with pytest.raises(SystemExit, match="not DEV"):
        D.build(a, tmp_path / "b_1.6_1.0", dev=dev, lead_block=block, non_dev="refuse", n_boot=100)


def test_unpowered_row_is_a_placeholder_with_its_n(tmp_path, dev, block):
    doc, _, _ = run(tmp_path, dev, block, 1.6, 1.0, min_n=1000)
    for r in doc["rows"]["LONGITUDINAL"].values():
        assert r["undefined"] and r["n"] == 48 and "UNPOWERED" in r["reason"]


def test_coarse_common_grid_is_flagged_and_scored_on_the_shared_instants(tmp_path, dev):
    blk = write_block(tmp_path / "block_dt0p2_k10.npz", dt=0.2, k=10)           # the banked 0.2 s block: shares 1.0 s and 2.0 s only with the 0.5 s roll grid
    doc, _, _ = run(tmp_path, dev, blk, 1.6, 1.0)
    assert doc["lead"]["grid_complete"] is False and doc["lead"]["instants_s"] == pytest.approx([1.0, 2.0]) and doc["lead"]["dt_s"] == pytest.approx(1.0) and doc["lead"]["grid_note"]
    r = doc["rows"]["LONGITUDINAL"]["LON_dk_headway_min_m"]
    assert r["a_mean"] == pytest.approx(np.mean([0.2 * s for s in (8, 10, 12, 14)]))      # min over the 2 shared instants = the 2.0 s gap
    one, _ = D.window_values({**one_window(1.6), "cols": [1, 3], "dt": 1.0, "leads": np.tile([36.0, 0.0], (1, 2, 1))})
    assert one["LON_dk_min_ttc_s"][0] == pytest.approx(1.125)                  # (18 - 2) / 1.0 s = 16 m/s; 18 / 16: the coarse-grid TTC, not 0.625


def test_cross_check_against_lead_metrics_lead_viol(tmp_path, dev, block):
    """The row's a_mean is the registered LEAD-VIOL `share_of_at_risk` (lead_metrics.lead_viol), not a re-derivation."""
    from taniteval import lead_metrics as lm
    A, _ = D.restrict_dev(D.load_arm(write_dump(tmp_path, "a", 1.6), block), dev)
    vals, _ = D.window_values({"P": A["P"], "G": A["G"], "leads": A["leads"], "lens": A["lens"], "speeds": A["speeds"], "state": A["state"], "cols": A["cols"], "dt": A["dt"]})
    c = A["cols"]
    dkp, dkg = (lm.distance_keeping(x[:, c], A["leads"], A["lens"], A["speeds"], A["dt"]) for x in (A["P"], A["G"]))
    lv = lm.lead_viol(dkp, dkg, [k[0] for k in A["keys"]], lead_valid=A["state"] == "LEAD", n_boot=50)
    assert lv["share_of_at_risk"] == pytest.approx(float(np.nanmean(vals["LON_dk_lead_viol_at_risk"])))


# ---- merge --------------------------------------------------------------------------------------------------------------------------------------------------
def _write(tmp_path, name, obj):
    p = tmp_path / name
    p.write_text(json.dumps(obj), encoding="utf-8")
    return p


@pytest.mark.parametrize("what", ["n_windows", "labels", "sha", "dir_a", "mode"])
def test_RED_merge_refuses_a_families_json_from_other_windows_or_arms(tmp_path, dev, block, what):
    doc, a, b = run(tmp_path, dev, block, 1.6, 1.0)
    fj = fams_json(doc, a, b)
    if what == "n_windows":
        fj["n_windows_paired"] -= 4
    elif what == "labels":
        fj["labels"] = "K0 - K2"
    elif what == "sha":
        fj["window_set_sha256"] = "0" * 64
    elif what == "dir_a":
        fj["a"]["dir"] = str(tmp_path / "some_other_dump")
    else:
        fj["mode"] = "camblind"
    p = _write(tmp_path, "fam.json", fj)
    before = p.read_text(encoding="utf-8")
    with pytest.raises(SystemExit, match="REFUSED"):
        D.merge_into(p, doc)
    assert p.read_text(encoding="utf-8") == before                                 # a refusal writes nothing


def test_merge_adds_rows_under_longitudinal_and_keeps_the_speed_rows(tmp_path, dev, block):
    doc, a, b = run(tmp_path, dev, block, 1.0, 1.6)
    fj = fams_json(doc, a, b)
    fj["window_set_sha256"] = doc["window_set_sha256"]
    out = D.merge_into(_write(tmp_path, "fam.json", fj), doc, out=tmp_path / "merged.json")
    lon = json.loads((tmp_path / "merged.json").read_text(encoding="utf-8"))["rows"]["LONGITUDINAL"]
    assert "LON_speed_mae_mps" in lon and set(LEAD_ROWS) <= set(lon) and out["distance_keeping_merge"]["rows_added"] == sorted(LEAD_ROWS)
    assert not (tmp_path / "merged.json.tmp").exists()


def test_cli_end_to_end_and_refusal_exit_code(tmp_path, dev, block):
    import r9_e2_families as F
    a, b = write_dump(tmp_path, "a", 1.6), write_dump(tmp_path, "b", 1.0)
    split = _write(tmp_path, "split.json", {"DEV": {"sha12": sorted(dev)}, "HOLD": {"sha12": [F.sha12(c) for c in HOLD_CLIPS]}})
    doc0 = D.build(a, b, dev=dev, lead_block=block, n_boot=50)
    fam = _write(tmp_path, "fam.json", fams_json(doc0, a, b))
    rc = D.main(["--a-dump", str(a), "--b-dump", str(b), "--dev-split", str(split), "--lead-block", str(block), "--boot", "50", "--out", str(tmp_path / "o.json"), "--merge-into", str(fam)])
    assert rc == 0 and json.loads((tmp_path / "o.json").read_text(encoding="utf-8"))["schema"] == D.SCHEMA
    assert "LON_dk_headway_min_m" in json.loads(fam.read_text(encoding="utf-8"))["rows"]["LONGITUDINAL"]
    bad = write_dump(tmp_path, "bad", 1.0, clips=tuple(DEV_CLIPS[:-1] + HOLD_CLIPS))
    assert D.main(["--a-dump", str(a), "--b-dump", str(bad), "--dev-split", str(split), "--lead-block", str(block), "--out", str(tmp_path / "o2.json")]) == 2
    assert not (tmp_path / "o2.json").exists()


def test_directions_are_pinned_literally():
    assert dict(D.ROWS) == {"LON_dk_headway_min_m": "higher", "LON_dk_time_gap_min_s": "higher", "LON_dk_min_ttc_s": "higher", "LON_dk_headway_abs_err_m": "lower",
                            "LON_dk_ttc_viol_rate": "lower", "LON_dk_lead_viol_at_risk": "lower", "LON_dk_lead_progress_shortfall": "lower"}
    assert D.TTC_VIOL_S == 1.5


def test_hand_computed_crawling_window_has_a_progress_shortfall():
    v, ex = D.window_values(one_window(0.5))                                      # plan arc 10 m vs human 20 m on an at-risk window
    assert v["LON_dk_lead_progress_shortfall"][0] == pytest.approx(0.5) and ex["mean_progress_ratio"] == pytest.approx(0.5)
    assert v["LON_dk_headway_min_m"][0] == pytest.approx(24.0)                    # it "keeps distance" -- the loophole the control exists for


def test_RED_a_crawling_arm_wins_the_distance_rows_but_the_progress_control_fails_the_judge(tmp_path, dev, block):
    doc, a, b = run(tmp_path, dev, block, 0.5, 1.0)                              # K2 crawls, K0 = the human gap
    rows = doc["rows"]["LONGITUDINAL"]
    assert rows["LON_dk_headway_min_m"]["delta"] > 0 and rows["LON_dk_headway_min_m"]["separated"]            # better on distance ...
    r = rows["LON_dk_lead_progress_shortfall"]
    assert r["better"] == "lower" and r["delta"] == pytest.approx(0.5) and r["separated"] and doc["a"]["mean_progress_ratio"] == pytest.approx(0.5)
    c, _ = V.families_clause(D.merge_into(_write(tmp_path, "fam.json", fams_json(doc, a, b)), doc))
    assert c["status"] == "FAIL" and "LONGITUDINAL/LON_dk_lead_progress_shortfall" in c["failing_rows"], c
    assert not {"LONGITUDINAL/LON_dk_headway_min_m", "LONGITUDINAL/LON_dk_time_gap_min_s", "LONGITUDINAL/LON_dk_min_ttc_s"} & set(c["failing_rows"])     # the distance rows alone would have passed it
