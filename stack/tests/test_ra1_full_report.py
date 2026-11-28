"""PREREG_RA1_FULL's report generator (code/ra1_full_report.py). CPU only, < 30 s.

Run: PYTHONPATH=<repo>/stack;<repo>/taniteval pytest -q stack/tests/test_ra1_full_report.py

What is pinned, and the control that must go red for each:
  * --interim renders every run and all four families (STRATEGIC = PAUSED, distance-keeping UNAVAILABLE, the longitudinal
    admissibility caveat) and makes NO selection: no S*, no outcome;
  * the data-scale row is the PAIRED mean difference full - 540 on the same windows (equal to mean(a) - mean(b)), flagged beyond 2F
    only when it exceeds twice the larger training-seed floor; misaligned windows are REFUSED;
  * mixed provenance is REFUSED: another holdout (in one run, or shared by all runs but not the Amendment-1 list), a steer-unit run,
    a run from the 600-clip cache, two different P10-yaw-540s;
  * final mode copies S* / the outcome from the verdict's JSON and REFUSES a full1.json whose seed means are not the runs' (it would be
    reporting files the verdict did not read), and REFUSES a gate delta the A1x / A1 runs do not reproduce;
  * the R0 -> A1x -> A1 decomposition at S*_A1 is the paired mean difference of each pair; --notes lands under its own labelled
    heading, before the generated tables;
  * PAIRED FAMILY COLUMNS: when the runs' eval.json carries `families.paired_arm_minus_p10` the families table gains an
    "arm - P10 (PAIRED)" column per seed (the evaluator's own delta [lo, hi] and **sep**, curvature at 6 dp, - for an unpaired component,
    n/a for a run that predates the block) on EVERY row, the note that no family difference is claimed is replaced, and a failed
    P10 - P10 control is flagged; when the block is absent the note and the 5-column table are today's, byte for byte.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
PKG = REPO / "TanitAD Research Lab" / "Architecture & Inference" / "Research" / "2026-10-17-refav1-hj-mini"
REPORT = PKG / "code" / "ra1_full_report.py"
HOLDOUT_SHA = "f6a1dee02af0b96f815416bfef9bc11a04554702525135aa75a740942d7750f9"
RUN = {"R0": "R0_s{s}", "A1": "A1_b0.08g50_s{s}"}
pytestmark = pytest.mark.skipif(not REPORT.exists(), reason="report generator not in this checkout")


def _mod():
    spec = importlib.util.spec_from_file_location("ra1_full_report", REPORT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _ci(m):
    return {"mean": m, "lo": m - 0.1, "hi": m + 0.1}


def _fams(v):
    blk = {"longitudinal": {"ci": {"components": {"speed_mae_mps": _ci(v)}}, "distance_keeping": {"status": "UNAVAILABLE", "n": 0}},
           "lateral": {"ci": {"components": {"yaw_rate_mae_degps": _ci(v)}}},
           "tactical": {"lateral_decision": {"ci": {"accuracy": _ci(0.8), "kappa": _ci(0.5)}}, "goal_setting": {"goal_range_ratio": 1.0}},
           "strategic": {"status": "PAUSED", "ruling": "PI 2026-09-19", "underlying": {"status": "UNAVAILABLE", "reason": "no route"}},
           "_longitudinal_claim_admissible": False}
    return blk


def _paired(delta, ok=True):
    """A `families.paired_arm_minus_p10` block as the evaluator writes it (only the fields the report reads)."""
    c = lambda d: {"delta": d, "lo": d - 0.05, "hi": d + 0.05, "separated": True, "arm": 1.0, "p10": 1.0 - d}   # noqa: E731
    return {"status": "OK", "longitudinal": {"speed_mae_mps": c(delta), "along_mae_m": {"status": "UNAVAILABLE"}},
            "lateral": {"yaw_rate_mae_degps": c(delta - 0.1111),
                        "curvature_mae_1pm": {"delta": -0.000321, "lo": -0.00045, "hi": -0.00019, "separated": True}},
            "controls": {"p10_minus_p10_ok": ok}}


def _run(d, arm, S, s, ade6, hade6, p10=None, hp10=None, kin="yaw", n_train=4572, holdout=HOLDOUT_SHA, t_shift=0, paired=None):
    rng = np.random.default_rng(100 + s)
    eid = np.repeat(np.arange(20), 10)
    t = np.tile(np.arange(10), 20) + t_shift
    p10 = np.full(eid.size, 3.0) if p10 is None else p10
    hp10 = np.full(eid.size, 3.2) if hp10 is None else hp10
    d.mkdir(parents=True)
    np.savez_compressed(d / "perwindow.npz", eid=eid, t=t, clips=np.array([f"c{i:02d}" for i in range(20)]), ade6=ade6, p10_ade6=p10,
                        hold_eid=eid, hold_t=t, hold_ade6=hade6, hold_p10_ade6=hp10, _r=rng.normal(size=3))
    dl = {"delta": float(ade6.mean() - p10.mean()), "lo": -0.2, "hi": -0.1, "separated": True}
    ev = dict(arm=arm, kin_mode=kin, ckpt_step=S, instrument_ok=True, cfg_sha12="c" * 12, seconds=1.0,
              disjointness={"n_train": n_train, "n_eval": 141, "overlap": 0},
              controls={"p10_minus_p10_ok": True, "gt_ade_max": 0.0}, param_audit={"untrained": []},
              holdout={"clips_sha256": holdout, "train_sha256": "t" * 64, "ade6_arm": {"mean": float(hade6.mean())},
                       "ade6_p10": {"mean": float(hp10.mean())}, "ade6_arm_minus_p10": dl},
              K={"ade6_arm": {"mean": float(ade6.mean())}, "ade6_arm_minus_p10": dl},
              families={"arm": _fams(1.0), "p10": _fams(2.0), "lead_braking_stratum_and_distance_keeping": "NOT COMPUTABLE",
                        **({"paired_arm_minus_p10": paired} if paired is not None else {})},
              strata={"turn": {"n_windows": 40, "n_episodes": 4, "ade6_arm_minus_p10": dl}, "stop": {"status": "n too small"},
                      "brake": {"status": "n too small"}},
              COST={"L2_minus_L1_native": dl} if arm == "A1" else {"status": "n/a"})
    (d / "eval.json").write_text(json.dumps(ev), encoding="utf-8")


def _world(tmp, gain=0.3, seed_gap=0.01, mini_gap=0.01, every=None, **over):
    """full runs at 500 and 1000 (R0, A1 x 2 seeds) + 540-clip runs at 500; full = mini - gain on every window.
    over: applied to ONE full run (R0 seed 1 at 1000); every: applied to ALL full runs."""
    rng = np.random.default_rng(0)
    base = 2.6 + rng.normal(0, 0.3, 200) + np.repeat(rng.normal(0, 0.2, 20), 10)
    for S in (500, 1000):
        for arm in RUN:
            for s in (0, 1):
                kw = dict(every or {}, **(over if (S, arm, s) == (1000, "R0", 1) else {}))
                _run(tmp / "full" / "runs" / f"st{S}" / RUN[arm].format(s=s) / "eval", arm, S, s,
                     base - gain + s * seed_gap - (0.05 if arm == "A1" else 0), base + 0.4 + s * seed_gap - 0.001 * S / 100, **kw)
    # A1x seed 0 at 1000 (the gate's control): A1x - A1 = +0.02 on every eval-141 window
    _run(tmp / "full" / "runs" / "st1000" / "A1x_b0.08g50_s0" / "eval", "A1x", 1000, 0, base - gain - 0.03, base + 0.4 - 0.01,
         **dict(every or {}))
    for arm in RUN:
        for s in (0, 1):
            _run(tmp / "mini" / f"st500" / RUN[arm].format(s=s) / "eval", arm, 500, s, base + s * mini_gap - (0.05 if arm == "A1" else 0),
                 base + 0.4 + s * mini_gap, n_train=600, p10=np.full(200, 3.1), hp10=np.full(200, 3.3))
    return tmp / "full", tmp / "mini"


def _main(full, mini, *extra):
    out = full / "R.md"
    js = full / "R.json"
    mr = ["--mini-root", str(mini)] if mini is not None else []
    rc = _mod().main(["--root", str(full), *mr, "--out", str(out), "--json-out", str(js), *extra])
    return rc, (out.read_text(encoding="utf-8") if rc == 0 else ""), (json.loads(js.read_text(encoding="utf-8")) if rc == 0 else None)


def test_interim_renders_all_four_families_and_reads_nothing(tmp_path):
    full, mini = _world(tmp_path)
    rc, md, js = _main(full, mini, "--interim")
    assert rc == 0
    assert "INTERIM" in md and "⭐S*" not in md and "outcome **" not in md
    for must in ("LONGITUDINAL", "LATERAL", "TACTICAL", "**PAUSED**", "distance_keeping | **UNAVAILABLE**", "fidelity diagnostic",
                 "no arm − P10 family difference is claimed"):
        assert must in md, must
    assert js["interim"] is True and js["S_star"] == {}
    # no paired block anywhere -> today's 5-column table, no PAIRED column, and no new JSON keys
    assert "| family | metric | A1 s0 | A1 s1 | P10-yaw-4512 |\n|---|---|---|---|---|\n" in md
    assert "(PAIRED) |" not in md.replace("(PAIRED; n ", "") and "family_paired" not in js


def test_paired_family_columns_print_when_the_eval_carries_the_paired_block(tmp_path):
    full, mini = _world(tmp_path, every=dict(paired=_paired(-0.1234)))
    rc, md, js = _main(full, mini, "--interim")
    assert rc == 0
    assert "A1 s0 − P10 (PAIRED) | A1 s1 − P10 (PAIRED) |" in md and "R0 s0 − P10 (PAIRED)" in md
    assert "no arm − P10 family difference is claimed" not in md and "families.paired_arm_minus_p10" in md
    assert "TACTICAL decision agreement is not paired" in md, "what is still unpaired must stay visible"
    # the cells: the evaluator's own delta [lo, hi] and **sep**; curvature in 6 dp; an unpaired component is —, an UNAVAILABLE one says so
    assert "| LONGITUDINAL | speed_mae_mps |" in md
    row = next(l for l in md.splitlines() if l.startswith("| LONGITUDINAL | speed_mae_mps |"))
    assert row.endswith("| -0.1234 [-0.1734, -0.0734] **sep** | -0.1234 [-0.1734, -0.0734] **sep** |"), row
    assert next(l for l in md.splitlines() if l.startswith("| LATERAL | yaw_rate_mae_degps |")).count("-0.2345 [-0.2845, -0.1845] **sep**") == 2
    assert next(l for l in md.splitlines() if l.startswith("| LATERAL | curvature_mae_1pm |")).count("-0.000321 [-0.000450, -0.000190] **sep**") == 2
    assert next(l for l in md.splitlines() if l.startswith("| LONGITUDINAL | speed_bias_mps |")).endswith("| — | — |")
    assert next(l for l in md.splitlines() if l.startswith("| LONGITUDINAL | along_mae_m |")).endswith("| UNAVAILABLE | UNAVAILABLE |")
    assert next(l for l in md.splitlines() if l.startswith("| LONGITUDINAL | accel_mae_mps2 |")).endswith("| — | — |")
    # every table row has the same number of cells (the two new columns on EVERY row, not only the paired ones)
    tbl = [l for l in md.split("### A1 @ 1000")[1].split("\n\n")[1].splitlines() if l.startswith("|")]
    assert len({l.count("|") for l in tbl}) == 1 and tbl[0].count("|") == 8, tbl[:2]
    # the machine-readable copy
    assert js["family_paired"]["A1@1000/s0"]["longitudinal"]["speed_mae_mps"]["delta"] == -0.1234
    assert "along_mae_m" not in js["family_paired"]["A1@1000/s0"]["longitudinal"]            # UNAVAILABLE components carry no delta
    assert js["family_paired_control_failed"] == []


def test_a_run_without_the_paired_block_shows_na_beside_one_that_has_it(tmp_path):
    full, mini = _world(tmp_path, every=dict(paired=_paired(-0.1234)), paired=None)      # R0 seed 1 @ 1000 predates the block
    rc, md, js = _main(full, mini, "--interim")
    assert rc == 0
    row = next(l for l in md.split("### R0 @ 1000")[1].splitlines() if l.startswith("| LONGITUDINAL | speed_mae_mps |"))
    assert row.endswith("| -0.1234 [-0.1734, -0.0734] **sep** | n/a |"), row
    assert "R0@1000/s1" not in js["family_paired"] and "R0@1000/s0" in js["family_paired"]


def test_a_failed_paired_control_is_flagged_not_silently_rendered(tmp_path):
    full, mini = _world(tmp_path, every=dict(paired=_paired(-0.1234)), paired=_paired(-0.1234, ok=False))
    rc, md, js = _main(full, mini, "--interim")
    assert rc == 0
    assert "paired-family control FAILED for R0 @ 1000 seed 1" in md and js["family_paired_control_failed"] == ["R0@1000/s1"]


def test_data_scale_row_is_the_paired_mean_difference_and_2F_gates_the_flag(tmp_path):
    full, mini = _world(tmp_path / "a", gain=0.3)
    rc, md, js = _main(full, mini, "--interim")
    assert rc == 0
    for arm in RUN:
        for s in (0, 1):
            fa = np.load(full / "runs" / "st500" / RUN[arm].format(s=s) / "eval" / "perwindow.npz")["ade6"]
            ma = np.load(mini / "st500" / RUN[arm].format(s=s) / "eval" / "perwindow.npz")["ade6"]
            row = js["data_scale"][f"{arm}@500/s{s}"]
            assert row["eval141"]["delta"] == pytest.approx(round(float(fa.mean() - ma.mean()), 4), abs=1e-4)
            assert row["beyond_2F"] is True
    assert "st1000" not in json.dumps(js["data_scale"])          # 1000 exists at one scale only: no matched row
    full2, mini2 = _world(tmp_path / "b", gain=0.3, seed_gap=0.2)  # F = 0.2 -> 2F = 0.4 > |0.3|: separated, but not beyond 2F
    rc, _, js2 = _main(full2, mini2, "--interim")
    assert rc == 0 and not any(r["beyond_2F"] for r in js2["data_scale"].values())


def test_misaligned_windows_are_refused(tmp_path):
    full, mini = _world(tmp_path)
    d = mini / "st500" / "R0_s0" / "eval"
    z = dict(np.load(d / "perwindow.npz"))
    z["t"] = z["t"] + 1
    np.savez_compressed(d / "perwindow.npz", **z)
    assert _main(full, mini, "--interim")[0] == 2


@pytest.mark.parametrize("over", [dict(holdout="0" * 64), dict(kin="steer"), dict(n_train=600)])
def test_mixed_provenance_is_refused(tmp_path, over):
    full, mini = _world(tmp_path, **over)
    assert _main(full, mini, "--interim")[0] == 2


def test_a_shared_holdout_that_is_not_the_amendment_1_list_is_refused(tmp_path):
    full, mini = _world(tmp_path, every=dict(holdout="0" * 64))   # ONE holdout across all runs, but not f6a1dee0...
    assert _main(full, None, "--interim")[0] == 2                  # no 540-clip root: the window-alignment guard cannot catch it
    full2, _ = _world(tmp_path / "ok")
    assert _main(full2, None, "--interim")[0] == 0                 # control: the same call on the right holdout renders


def test_two_different_p10_yaw_540s_are_refused(tmp_path):
    full, mini = _world(tmp_path)
    d = mini / "st500" / "A1_b0.08g50_s1" / "eval"
    z = dict(np.load(d / "perwindow.npz"))
    z["p10_ade6"] = z["p10_ade6"] + 0.01
    np.savez_compressed(d / "perwindow.npz", **z)
    assert _main(full, mini, "--interim")[0] == 2


def _full1(full, nudge=0.0):
    b = {}
    for arm in RUN:
        for S in (500, 1000):
            h = [json.loads((full / "runs" / f"st{S}" / RUN[arm].format(s=s) / "eval" / "eval.json").read_text())["holdout"]["ade6_arm"]["mean"]
                 for s in (0, 1)]
            b.setdefault(arm, {})[str(S)] = {"seed_mean_holdout_ade6": sum(h) / 2 + nudge}
    (full / "full1.json").write_text(json.dumps(dict(status="OK", S_star={"R0": 1000, "A1": 1000}, edge={"R0": "ABOVE", "A1": "ABOVE"},
                                                     K_arm="A1", reading="r", next="full2 1000 1000 0.08 50", budgets=b)))


def _final(full, gate_delta=0.02):
    cand = {"seed0": {"ade6": 2.3, "p10_ade6": 2.5, "arm_minus_p10": {"delta": -0.2, "lo": -0.3, "hi": -0.1, "separated": True}}}
    (full / "full_final.json").write_text(json.dumps(dict(
        outcome="K_PASS", reading="READING 4 PASS", K_arm="A1", training_seed_floor_A1=0.01,
        A1x_control={"verdict": "OK: A1x loses", "delta_A1x_minus_A1_ade6": {"delta": gate_delta, "lo": 0.01, "hi": 0.03, "separated": True}},
        K_read={"bar": "b", "K_arm": "A1", "PASS": True, "K_arm_row": cand, "other_arm_DESCRIPTIVE": cand})))


def test_final_mode_copies_S_star_and_the_outcome_and_decomposes_R0_A1x_A1(tmp_path):
    full, mini = _world(tmp_path)
    _full1(full)
    _final(full)
    notes = tmp_path / "notes.md"
    notes.write_text("HAND-WRITTEN LINE", encoding="utf-8")
    rc, md, js = _main(full, mini, "--notes", str(notes))
    assert rc == 0
    assert "FINAL" in md and "K_PASS" in md and "→ **PASS**" in md and "OK: A1x loses" in md
    assert md.count("⭐S*") == 2 and "| A1 | 1000 ⭐S*" in md and "| R0 | 1000 ⭐S*" in md
    assert js["K_arm"] == "A1" and js["S_star"] == {"R0": 1000, "A1": 1000}
    rows = js["decomposition"]["rows"]
    assert js["decomposition"]["S"] == 1000
    assert rows["A1x_s0-A1_s0"]["eval141"]["delta"] == pytest.approx(0.02, abs=1e-4)    # the gate's own quantity, re-derived
    assert rows["A1x_s0-R0_s0"]["eval141"]["delta"] == pytest.approx(-0.03, abs=1e-4)
    assert rows["A1_s1-R0_s1"]["eval141"]["delta"] == pytest.approx(-0.05, abs=1e-4)
    i = md.index("## The author's reading")
    assert "NOT generated" in md[i:i + 200] and "HAND-WRITTEN LINE" in md[i:] and i < md.index("## Reading 1")


def test_final_mode_refuses_a_gate_delta_the_runs_do_not_give(tmp_path):
    full, mini = _world(tmp_path)
    _full1(full)
    _final(full, gate_delta=0.05)                                    # the runs give +0.02
    assert _main(full, mini)[0] == 2
    _final(full, gate_delta=0.02)                                    # control: the true value renders
    assert _main(full, mini)[0] == 0


def test_final_mode_refuses_a_full1_that_read_other_files(tmp_path):
    full, mini = _world(tmp_path)
    _full1(full, nudge=1e-6)
    assert _main(full, mini)[0] == 2
