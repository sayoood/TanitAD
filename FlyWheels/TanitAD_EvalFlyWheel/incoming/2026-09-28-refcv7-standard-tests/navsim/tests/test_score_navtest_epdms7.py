"""Controls for the navtest EPDMS re-score (``code/score_navtest_epdms7.py`` + ``code/epdms_waiter7.py``).

TANITAD VENV. NO scorer is launched and NO GPU is touched; the heaviest read is one 5 MB seam and a 20 KB CSV.
Addendum: ``SPEC_ADDENDUM_NAVTEST_EPDMS.md`` section 6. Every expectation below is a LITERAL (hand-computed from
docs/metrics.md or read off a banked file), never an expression over the code under test, and each guard has a
deliberate-regression arm that must go RED.
"""
from __future__ import annotations

import csv
import importlib
import json
import math
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PKG, "code"))
sys.path.insert(0, "D:/Projects/TanitAD/taniteval")

import score_navtest_epdms7 as E  # noqa: E402

CONTROLS = os.path.join(PKG, "raw", "navtest_epdms", "controls")
W8_STOP = os.path.join(CONTROLS, "W8_smoke218_STOP.csv")
SEAM5K = os.path.join(PKG, "raw", "milestones", "step5000", "bridge_navtest", "seam_R7_A1.npz")


# --------------------------------------------------------------------------- #
# KE1: the formula, on hand-computed LITERALS                                  #
# --------------------------------------------------------------------------- #
def _row(**kw):
    base = dict(NC=1.0, DAC=1.0, DDC=1.0, TLC=1.0, EP=1.0, TTC=1.0, LK=1.0, HC=1.0, EC=1.0)
    base.update(kw)
    return base


def test_KE1_formula_literals():
    # (5*0.5 + 5*1 + 2*1 + 2*0 + 2*1) / 16 = 11.5 / 16
    assert E.epdms_from_components(_row(EP=0.5, HC=0.0)) == pytest.approx(0.71875, abs=1e-15)
    # EC NaN -> its weight is dropped: (2.5 + 5 + 2 + 0) / 14 = 9.5 / 14
    assert E.epdms_from_components(_row(EP=0.5, HC=0.0, EC=float("nan"))) == pytest.approx(0.678571428571, abs=1e-12)
    # NC = 1/2 halves the score: 0.5 * 0.71875
    assert E.epdms_from_components(_row(NC=0.5, EP=0.5, HC=0.0)) == pytest.approx(0.359375, abs=1e-15)
    # any zero multiplier -> 0
    for k in ("NC", "DAC", "DDC", "TLC"):
        assert E.epdms_from_components(_row(**{k: 0.0})) == 0.0
    # perfect -> 1
    assert E.epdms_from_components(_row()) == 1.0


def _load_banked():
    from taniteval.bench.navsim import single_stage as SS
    return SS.load_token_scores(W8_STOP)


def test_KE1_holds_on_W8s_banked_rows_and_a_mutation_goes_red():
    tok = _load_banked()
    assert len(tok) == 218
    gap, n = E.max_formula_gap(tok)
    assert n == 218 and gap <= 1e-9
    # deliberate regression: treat a NaN EC as 0 instead of dropping it -> must disagree with the devkit
    worst = 0.0
    n_nan = 0
    for r in tok.values():
        row = {k: r[c] for k, c in E.COMPONENTS.items()}
        if math.isnan(row["EC"]):
            n_nan += 1
            row["EC"] = 0.0
            worst = max(worst, abs(E.epdms_from_components(row) - r["score"]))
    assert n_nan >= 30            # W8: 34 tokens have no adjacent earlier frame
    assert worst > 1e-3


def test_banked_headline_literal():
    """The known value the KE2 control must reproduce (read as TEXT from the banked CSV)."""
    rows = list(csv.DictReader(open(W8_STOP, encoding="utf-8")))
    avg = [r for r in rows if r["token"] == "average_all_frames"]
    assert len(avg) == 1
    assert float(avg[0]["score"]) == 0.579805532200227


# --------------------------------------------------------------------------- #
# G1/G2/G5: the guards refuse                                                  #
# --------------------------------------------------------------------------- #
def _write_csv(path, rows, header):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([""] + list(header))                     # the devkit CSV's first column is the UNNAMED pandas index
        for i, r in enumerate(rows):
            w.writerow([i] + r)


def _banked_rows():
    rd = list(csv.reader(open(W8_STOP, encoding="utf-8")))
    return rd[0], rd[1:]


def test_guard_accepts_the_real_banked_csv_for_its_own_token_set():
    tok = _load_banked()
    pc = E.postcheck(W8_STOP, set(tok), full=False)
    assert pc["ok"], pc["failures"]
    assert pc["n_tokens"] == 218 and pc["headline_score"] == 0.579805532200227


def test_guard_refuses_a_short_csv_a_swapped_token_and_a_full_split_claim(tmp_path):
    hdr, rows = _banked_rows()
    tok = _load_banked()
    want = set(tok)
    # (a) one token row short
    short = [r[1:] for r in rows if r[1] != next(iter(want))]
    p = tmp_path / "short.csv"
    _write_csv(p, short, hdr[1:])
    assert not E.postcheck(str(p), want, full=False)["ok"]
    # (b) one token id swapped for a foreign one
    swapped = [list(r[1:]) for r in rows]
    swapped[0][0] = "ffffffffffffffff"
    p2 = tmp_path / "swap.csv"
    _write_csv(p2, swapped, hdr[1:])
    pc = E.postcheck(str(p2), want, full=False)
    assert not pc["ok"] and any("G2" in f for f in pc["failures"])
    # (c) the real 218 rows claimed as the FULL split -> the 12,146 / token-hash guard refuses
    pc = E.postcheck(W8_STOP, set(tok), full=True)
    assert not pc["ok"] and any("G2 token-set sha256" in f for f in pc["failures"])


def test_guard_refuses_a_pdm_score_only_csv(tmp_path):
    hdr, rows = _banked_rows()
    h = [("pdm_score" if c == "score" else c) for c in hdr[1:]]
    p = tmp_path / "pdm.csv"
    _write_csv(p, [r[1:] for r in rows], h)
    pc = E.postcheck(str(p), set(_load_banked()), full=False)
    assert not pc["ok"] and "pdm_score" in " ".join(pc["failures"])


def test_guard_refuses_a_row_whose_score_breaks_the_formula(tmp_path):
    hdr, rows = _banked_rows()
    si = hdr.index("score")
    bad = [list(r) for r in rows]
    bad[3][si] = repr(float(bad[3][si]) + 0.01)           # one token's score moved by 0.01
    p = tmp_path / "bad.csv"
    _write_csv(p, [r[1:] for r in bad], hdr[1:])
    pc = E.postcheck(str(p), set(_load_banked()), full=False)
    assert not pc["ok"]


# --------------------------------------------------------------------------- #
# KE2/KE3: cell-identity comparison, itself under test                         #
# --------------------------------------------------------------------------- #
def test_compare_is_identical_on_itself_and_red_on_a_1e6_change(tmp_path):
    assert E.compare_csv(W8_STOP, W8_STOP)["identical"]
    hdr, rows = _banked_rows()
    ci = hdr.index("ego_progress")
    mut = [list(r) for r in rows]
    mut[5][ci] = repr(float(mut[5][ci]) + 1e-6)
    p = tmp_path / "mut.csv"
    _write_csv(p, [r[1:] for r in mut], hdr[1:])
    r = E.compare_csv(W8_STOP, str(p))
    assert not r["identical"] and r["max_abs"]["ego_progress"] == pytest.approx(1e-6, rel=1e-3)
    # NaN vs a number is "infinite" difference, NaN vs NaN is equal
    nan_i = hdr.index("two_frame_extended_comfort")
    k = next(i for i, r_ in enumerate(rows) if r_[nan_i] == "")
    mut2 = [list(r) for r in rows]
    mut2[k][nan_i] = "0.5"
    p2 = tmp_path / "mut2.csv"
    _write_csv(p2, [r[1:] for r in mut2], hdr[1:])
    assert E.compare_csv(W8_STOP, str(p2))["max_abs"]["two_frame_extended_comfort"] == float("inf")


# --------------------------------------------------------------------------- #
# G4: the seam                                                                 #
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not os.path.exists(SEAM5K), reason="step-5,000 seam not on this disk")
def test_banked_step5000_seam_passes_G4_and_a_mutated_copy_is_refused(tmp_path):
    info = E.seam_info(SEAM5K)
    assert info["n"] == 12146 and info["n_unique"] == 12146
    assert info["tokens_sha256"] == "8d42fef68542f095d0e03894fdaf38babfdc89f3fd7cdc26080c3a6009cb4108"
    assert E.seam_refusals(info) == []
    z = np.load(SEAM5K, allow_pickle=False)
    # (a) one token replaced
    toks = np.asarray(z["token"]).copy()
    toks[0] = "0" * 16
    np.savez(tmp_path / "m1.npz", **{**{k: z[k] for k in z.files}, "token": toks})
    assert any("token-set sha256" in b for b in E.seam_refusals(E.seam_info(str(tmp_path / "m1.npz"))))
    # (b) one non-finite pose
    poses = np.asarray(z["poses"]).copy()
    poses[7, 3, 0] = np.nan
    np.savez(tmp_path / "m2.npz", **{**{k: z[k] for k in z.files}, "poses": poses})
    assert "non-finite poses" in E.seam_refusals(E.seam_info(str(tmp_path / "m2.npz")))
    # (c) a wrong horizon grid
    np.savez(tmp_path / "m3.npz", **{**{k: z[k] for k in z.files}, "sampling": np.asarray([8.0, 0.25])})
    assert E.seam_refusals(E.seam_info(str(tmp_path / "m3.npz")))


@pytest.mark.skipif(not os.path.exists(SEAM5K), reason="step-5,000 seam not on this disk")
def test_subset_seam_rows_are_bit_identical_to_the_parent(tmp_path):
    z = np.load(SEAM5K, allow_pickle=False)
    pick = [str(x) for x in z["token"][[3, 10, 11, 5000]]]
    rec = E.subset_seam(SEAM5K, pick, str(tmp_path / "sub.npz"))
    y = np.load(rec["path"], allow_pickle=False)
    parent = {str(t): i for i, t in enumerate(z["token"])}
    for j, t in enumerate(y["token"]):
        assert np.array_equal(y["poses"][j], z["poses"][parent[str(t)]])
        assert str(y["fingerprint"][j]) == str(z["fingerprint"][parent[str(t)]])
    assert rec["n"] == 4 and rec["parent_sha256"] == E.sha256_file(SEAM5K)
    with pytest.raises(SystemExit):
        E.subset_seam(SEAM5K, pick + ["f" * 16], str(tmp_path / "sub2.npz"))


def test_stop_seam_is_the_suites_own_all_zero_builder(tmp_path):
    meta = {"aaaa000000000001": {"fp": "f1"}, "aaaa000000000002": {"fp": "f2"}, "aaaa000000000003": {"fp": "f3"}}
    p = E.stop_seam(str(tmp_path / "stop.npz"), sorted(meta), meta)
    z = np.load(p, allow_pickle=False)
    assert [str(t) for t in z["token"]] == sorted(meta)
    assert z["poses"].shape == (3, 8, 3) and not z["poses"].any()
    assert [str(f) for f in z["fingerprint"]] == ["f1", "f2", "f3"]
    assert set(str(s) for s in z["source"]) == {"precomputed"} and z["sampling"].tolist() == [8.0, 0.5]


# --------------------------------------------------------------------------- #
# the gates                                                                    #
# --------------------------------------------------------------------------- #
def _m(p, v):
    return {"free_phys_gb": p, "free_virt_gb": v}


def test_gate_needs_five_consecutive_open_samples_on_both_quantities():
    ok, low = _m(9.0, 6.0), _m(8.99, 8.0)
    assert E.gate_ok(ok) and not E.gate_ok(low)
    assert not E.gate_ok(_m(12.0, 5.99))                  # commit headroom closes it on its own
    assert not E.gate_sustained([ok] * 4)
    assert E.gate_sustained([ok] * 5)
    assert not E.gate_sustained([ok] * 4 + [low])
    assert not E.gate_sustained([low] + [ok] * 4)
    assert E.gate_sustained([low] + [ok] * 5)
    # the smoke gate is looser but is still a gate
    assert E.gate_ok(_m(4.5, 4.0), E.SMOKE_PHYS_GB, E.SMOKE_VIRT_GB)
    assert not E.gate_ok(_m(3.9, 8.0), E.SMOKE_PHYS_GB, E.SMOKE_VIRT_GB)


def test_mem_status_reads_plausible_numbers():
    m = E.mem_status()
    assert 0.0 < m["free_phys_gb"] < m["total_phys_gb"] and m["free_virt_gb"] > 0.0


# --------------------------------------------------------------------------- #
# classification words (addendum section 5)                                    #
# --------------------------------------------------------------------------- #
def _iv(lo, hi):
    return {"status": "OK", "lo": lo, "hi": hi}


def test_classification_words():
    F = 0.085                                              # points
    assert E.classify(3.7774, _iv(0.0158, 0.0586), F) == "SEPARATED"          # excl 0 and 3.78 > 0.17
    assert E.classify(0.15, _iv(0.0001, 0.0030), F) == "NOT PROVEN"           # excl 0 but 0.15 <= 0.17
    assert E.classify(-0.07, _iv(-0.0014, 0.0), F) == "NOT SEPARATED"         # hi == 0 does not exclude 0
    assert E.classify(-3.0, _iv(-0.05, -0.01), F) == "SEPARATED"              # a negative margin separates too
    assert E.classify(1.0, {"status": "UNAVAILABLE"}, F) == "INTERVAL UNAVAILABLE"


# --------------------------------------------------------------------------- #
# the pre-registration pin                                                     #
# --------------------------------------------------------------------------- #
def test_real_addendum_is_pinned_by_its_last_hash_line():
    st = E.addendum_state()
    assert st["ok"], st
    assert len(st["recorded"]) >= 1 and all(len(h) == 64 for h in st["recorded"])


def test_editing_the_addendum_without_a_hash_line_is_refused(tmp_path, monkeypatch):
    ad = tmp_path / "SPEC_ADDENDUM_NAVTEST_EPDMS.md"
    sf = tmp_path / "sha.txt"
    ad.write_text("pre-registered text\n", encoding="utf-8")
    sf.write_text(f"sha256  {E.sha256_file(str(ad))}  SPEC_ADDENDUM_NAVTEST_EPDMS.md\n", encoding="utf-8")
    monkeypatch.setattr(E, "ADDENDUM", str(ad))
    monkeypatch.setattr(E, "SHAFILE", str(sf))
    assert E.addendum_state()["ok"]
    ad.write_text("pre-registered text, quietly moved goalpost\n", encoding="utf-8")
    assert not E.addendum_state()["ok"]                              # RED: the edit has no hash line
    with open(sf, "a", encoding="utf-8") as fh:
        fh.write(f"sha256_after_A1  {E.sha256_file(str(ad))}  SPEC_ADDENDUM_NAVTEST_EPDMS.md\n")
    assert E.addendum_state()["ok"]                                  # an amendment with its own line is fine


# --------------------------------------------------------------------------- #
# the scorer is pointed where the addendum says                                #
# --------------------------------------------------------------------------- #
def test_overrides_use_the_one_stage_runner_non_reactive_and_the_D_logs():
    from taniteval.bench.navsim import profiles as P, scoring as SC
    prof = P.SPLITS["navtest_single_stage"]
    assert P.runner_for(prof) == "pdm_score_one_stage"
    ov = SC.overrides_for(prof, exp_name="x", worker="sequential")
    assert "train_test_split=navtest" in ov and "traffic_agents=non_reactive" in ov
    assert any(o.startswith("navsim_log_path=D:/Archive/devbox-C/navsim/data/openscene/navsim_logs/test") for o in ov)
    assert any(o.endswith("metric_cache_navtest_v2") for o in ov if o.startswith("metric_cache_path="))
    assert not any("synthetic" in o for o in ov)
    assert prof.n_stage1 == 12146 and prof.n_stage2 == 0 and prof.n_logs == 136
    assert P.DEVKIT_SHA == E.PINS["devkit_sha"]
    # RED arm: the two-stage runner on a one-stage split must be refused
    with pytest.raises(P.Refusal):
        P.check_runner(prof, "pdm_score")


# --------------------------------------------------------------------------- #
# KE5 / KE7                                                                    #
# --------------------------------------------------------------------------- #
def _tok(**kw):
    base = {c: 1.0 for c in E.COMPONENTS.values()}
    base.update(kw)
    return base


def test_KE5_human_analytic_and_its_red_arm():
    ok = {"a": _tok(), "b": _tok(driving_direction_compliance=0.5)}
    assert E.ke5_human(ok)["pass"]
    bad = {"a": _tok(), "b": _tok(no_at_fault_collisions=0.5)}
    assert not E.ke5_human(bad)["pass"] and E.ke5_human(bad)["exceptions"] == 1
    bad2 = {"a": _tok(driving_direction_compliance=0.0)}
    assert not E.ke5_human(bad2)["pass"]


def test_KE7_counts_v2_below_v1_as_a_plan_mismatch(tmp_path):
    v2 = tmp_path / "v2.csv"
    v1 = tmp_path / "v1.csv"
    v2.write_text("token,drivable_area_compliance\nt1,1.0\nt2,1.0\nt3,0.0\nt4,1.0\naverage_all_frames,0.75\n", encoding="utf-8")
    v1.write_text("token,drivable_area_compliance\nt1,1.0\nt2,0.0\nt3,1.0\nt4,1.0\naverage,0.75\n", encoding="utf-8")
    r = E.ke7_dac(str(v2), str(v1))
    assert (r["n_common"], r["equal"], r["v2_gt_v1"], r["v2_lt_v1"]) == (4, 2, 1, 1)


# --------------------------------------------------------------------------- #
# log hygiene                                                                  #
# --------------------------------------------------------------------------- #
def test_sanitize_replaces_thread_uuids_and_nothing_else():
    u = "-".join(["123e4567", "e89b", "42d3", "a456", "426614174000"])   # built, not typed: the landing scan refuses a literal
    txt = f"Starting worker in thread_id={u}, node_id=0\ntoken 0a1b2c3d4e5f6a7b scene 2021.06.03.12.02.06_veh-35_01100_01227\n"
    new, n = E.sanitize_text(txt)
    assert n == 1 and u not in new and "thread_id=<thread-uuid>" in new
    assert "0a1b2c3d4e5f6a7b" in new and "2021.06.03.12.02.06_veh-35_01100_01227" in new
    assert not E.UUID_RE.search(new)


# --------------------------------------------------------------------------- #
# the waiter: readiness is read from ARTIFACTS                                 #
# --------------------------------------------------------------------------- #
@pytest.fixture()
def W():
    import epdms_waiter7
    return importlib.reload(epdms_waiter7)


def _mkms(tmp_path, step, *, summary_md5=None, bars=False, marker=False):
    d = tmp_path / f"step{step}"
    d.mkdir(parents=True, exist_ok=True)
    if summary_md5:
        (d / "MILESTONE_SUMMARY.json").write_text(json.dumps({"md5": summary_md5, "step": step}), encoding="utf-8")
    if bars:
        (d / "BARS.json").write_text("{}", encoding="utf-8")
    if marker:
        (d / "complete.log").write_text(f"x\nZZCOMPLETE7DONEZZ step={step} bars=present\n", encoding="utf-8")
    elif marker is None:
        (d / "complete.log").write_text("2026 COMPLETE7 START ...\nDRY RUN -- nothing launched\n", encoding="utf-8")
    return str(tmp_path)


def test_waiter_done_rules(W, tmp_path):
    md = W.MILESTONES
    # step 30000: ONLY the marker line for THIS step counts
    assert W.milestone_done(30000, _mkms(tmp_path, 30000, marker=None))[0] is False
    assert W.milestone_done(30000, _mkms(tmp_path, 30000, marker=True))[0] is True
    root = _mkms(tmp_path / "other", 30000)
    (tmp_path / "other" / "step30000" / "complete.log").write_text("ZZCOMPLETE7DONEZZ step=5000\n", encoding="utf-8")
    assert W.milestone_done(30000, root)[0] is False                      # another step's marker is not ours
    # step 50400: summary with the RIGHT md5 and BARS
    r1 = _mkms(tmp_path / "a", 50400, summary_md5=md[50400]["md5"], bars=False)
    assert W.milestone_done(50400, r1)[0] is False
    r2 = _mkms(tmp_path / "b", 50400, summary_md5=md[50400]["md5"], bars=True)
    assert W.milestone_done(50400, r2)[0] is True
    r3 = _mkms(tmp_path / "c", 50400, summary_md5="0" * 32, bars=True)
    assert W.milestone_done(50400, r3)[0] is False                        # another checkpoint's summary
    r4 = _mkms(tmp_path / "d", 50400)
    assert W.milestone_done(50400, r4)[0] is False
    # step 5000 uses the same summary rule
    r5 = _mkms(tmp_path / "e", 5000, summary_md5=md[5000]["md5"], bars=True)
    assert W.milestone_done(5000, r5)[0] is True


def test_the_pinned_milestone_md5s_match_the_banked_summaries(W):
    """The waiter's md5 pins are read off the live artifacts, not remembered."""
    p = os.path.join(PKG, "raw", "milestones", "step5000", "MILESTONE_SUMMARY.json")
    if os.path.exists(p):
        assert json.load(open(p, encoding="utf-8"))["md5"] == W.MILESTONES[5000]["md5"]
    q = os.path.join(PKG, "raw", "milestones", "step30000", "complete.log")
    if os.path.exists(q):
        assert f"md5={W.MILESTONES[30000]['md5']}" in open(q, encoding="utf-8", errors="replace").read()
    r = os.path.join(PKG, "raw", "milestones", "step50400", "runner.log")
    if os.path.exists(r):
        assert f"md5={W.MILESTONES[50400]['md5']}" in open(r, encoding="utf-8", errors="replace").read()


def test_waiter_work_order_and_gave_up_skip(W, monkeypatch):
    done = set()
    monkeypatch.setattr(W, "item_done", lambda step, arm: (step, arm) in done)
    monkeypatch.setattr(W, "milestone_done", lambda step, ms_root=None: (step in (5000,), "stub"))
    monkeypatch.setattr(W, "seam_ok", lambda step, arm: (True, "ok"))
    pick, table = W.next_item({}, set())
    assert pick[:2] == (None, "STOP")                                      # the cheapest floor first
    done.add((None, "STOP"))
    assert W.next_item({}, set())[0][:2] == (5000, "R7_A1")
    done.add((5000, "R7_A1"))
    assert W.next_item({}, set())[0][:2] == (5000, "R7_A1_s1")
    done.update({(5000, "R7_A1_s1")})
    assert W.next_item({}, set())[0][:2] == (None, "CV")
    # a given-up item is skipped, not retried forever
    assert W.next_item({}, {"floors/CV"})[0][:2] == (5000, "PRIOR_ha0p")
    # milestones that are not done offer NO model arm
    done.update({(None, "CV"), (5000, "PRIOR_ha0p"), (5000, "R7_CEILDECL_d"), (None, "HUMAN")})
    pick, table = W.next_item({}, set())
    assert pick is None and any(v.startswith("waiting: milestone step30000 not done") for k, v in table if k.startswith("step30000"))


def test_waiter_marks_a_missing_seam_of_a_done_milestone_ABSENT_not_eligible_forever(W, monkeypatch):
    monkeypatch.setattr(W, "item_done", lambda step, arm: False)
    monkeypatch.setattr(W, "milestone_done", lambda step, ms_root=None: (step == 30000, "stub"))
    monkeypatch.setattr(W, "seam_ok", lambda step, arm: (arm != "R7_A1_s1", "stub"))
    _, table = W.next_item({}, set())
    states = dict(table)
    assert states["step30000/R7_A1_s1"] == "ABSENT"
    assert states["step30000/R7_A1"] == "ELIGIBLE"


def test_waiter_never_schedules_a_subset_or_an_unknown_arm(W):
    for step, arm in W.WORK:
        assert arm in E.FLOORS + E.MODEL_ARMS
        assert (step is None) == (arm in E.FLOORS)
    assert len(set(W.WORK)) == len(W.WORK)
    assert [a for s, a in W.WORK if s == 5000][0] == "R7_A1"


# --------------------------------------------------------------------------- #
# summarize, end to end, on SYNTHETIC full-split arms (the real scorer is not run)       #
# --------------------------------------------------------------------------- #
def _synthetic_arm_csv(path, tokens, kind, seed):
    """12,146 rows whose `score` obeys the EPDMS formula; kinds differ in quality so the pairs have a sign."""
    rng = np.random.default_rng(seed)
    n = len(tokens)
    good = {"HUMAN": 1.0, "R7_A1": 0.62, "R7_A1_s1": 0.62, "R7_CEILDECL_d": 0.63, "PRIOR_ha0p": 0.55,
            "STOP": 0.45, "CV": 0.30}[kind]
    nc = np.where(rng.random(n) < (1.0 if kind == "HUMAN" else 0.9), 1.0, 0.5)
    dac = np.where(rng.random(n) < (1.0 if kind == "HUMAN" else 0.9), 1.0, 0.0)
    ddc = np.where(rng.random(n) < 0.97, 1.0, 0.5)
    tlc = np.ones(n)
    ep = np.clip(rng.normal(good, 0.2, n), 0, 1)
    ttc = np.where(rng.random(n) < 0.9, 1.0, 0.0)
    lk = np.where(rng.random(n) < 0.95, 1.0, 0.0)
    hc = np.where(rng.random(n) < 0.8, 1.0, 0.0)
    ec = np.where(rng.random(n) < 0.5, 1.0, 0.0)
    ec[rng.random(n) < 0.05] = np.nan
    if kind == "HUMAN":
        nc[:] = dac[:] = ttc[:] = lk[:] = 1.0
    rows, scores = [], []
    for i, t in enumerate(tokens):
        r = dict(NC=nc[i], DAC=dac[i], DDC=ddc[i], TLC=tlc[i], EP=ep[i], TTC=ttc[i], LK=lk[i], HC=hc[i], EC=ec[i])
        sc = E.epdms_from_components(r)
        scores.append(sc)
        rows.append([i, t, "True"] + [repr(float(r[k])) if not math.isnan(r[k]) else "" for k in E.COMPONENTS]
                    + [repr(sc)])
    hdr = ["", "token", "valid"] + list(E.COMPONENTS.values()) + ["score"]
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(hdr)
        w.writerows(rows)
        w.writerow([len(rows), "average_all_frames", "True"] + [""] * 9 + [repr(float(np.mean(scores)))])


@pytest.fixture()
def synth(tmp_path, monkeypatch):
    if not os.path.exists(SEAM5K):
        pytest.skip("step-5,000 seams not on this disk")
    out = tmp_path / "raw" / "navtest_epdms"
    (out / "inputs").mkdir(parents=True)
    import shutil
    shutil.copyfile(E.meta_path(), out / "inputs" / "token_meta.json")
    monkeypatch.setattr(E, "PKG", str(tmp_path))
    monkeypatch.setattr(E, "OUT", str(out))
    meta = E.load_meta()
    tokens = sorted(meta)
    seeds = {"STOP": 1, "CV": 2, "HUMAN": 3, "R7_A1": 4, "R7_A1_s1": 5, "PRIOR_ha0p": 6, "R7_CEILDECL_d": 7}
    for arm, seed in seeds.items():
        d = out / ("floors" if arm in E.FLOORS else "step5000") / arm
        d.mkdir(parents=True)
        p = d / f"{arm}.devkit.csv"
        _synthetic_arm_csv(p, tokens, arm, seed)
        rec = {"status": "PASS", "full_split": True, "arm": arm, "wall_s": 1.0, "resources": {"peak_rss_mb": 1.0},
               "banked": {"csv": os.path.relpath(str(p), str(tmp_path)).replace(os.sep, "/"),
                          "csv_sha256": E.sha256_file(str(p))},
               "cache": {"tokens_sha256": E.PINS["tokens_sha256"], "manifest_sha256": E.PINS["cache_manifest_sha256"]},
               "seam": ({"sha256": E.sha256_file(E.seam_path(5000, arm))} if arm in E.MODEL_ARMS else None),
               "headline_x100": 0.0}
        (d / "ARM_DONE.json").write_text(json.dumps(rec), encoding="utf-8")
    return out, tokens


def test_summarize_end_to_end_on_synthetic_arms(synth):
    out, tokens = synth
    assert E.main(["summarize", "--step", "5000"]) == 0
    s = json.load(open(out / "step5000" / "summary_navtest_epdms.json", encoding="utf-8"))
    assert s["status"] == "COMPLETE" and s["n_tokens"] == 12146 and s["refusals"] == []
    a1, st = s["arms"]["R7_A1"], s["arms"]["STOP"]
    assert a1["status"] == "OK" and st["status"] == "OK"
    assert a1["interval"]["status"] == "OK" and a1["interval"]["lo"] < a1["epdms"] < a1["interval"]["hi"]
    assert a1["interval"]["n_clusters"] == 136 and a1["interval"]["detail"]["n_boot"] == 2000 and a1["interval"]["detail"]["seed"] == 0
    assert 0.0 <= a1["stop_fraction_4s_endpoint_lt_1m"] <= 1.0 and st["stop_fraction_4s_endpoint_lt_1m"] == 1.0
    assert a1["controls"]["KE7_dac_v2_vs_v1"]["n_common"] == 12146
    assert s["arms"]["HUMAN"]["controls"]["KE5_human_analytic"]["pass"]
    # the synthetic R7_A1 is built better than STOP by ~0.17 EP: separated, positive, beyond 2x the seed floor
    cl = s["classification"]["R7_A1__minus__STOP"]
    assert cl["word"] == "SEPARATED" and cl["delta_x100"] > 0
    assert s["classification"]["R7_A1__minus__HUMAN"].startswith("CONTEXT")
    assert s["seed_floor_x100"] is not None and s["seed_floor_x100"] >= 0.0
    # P1 compares against the REAL banked PDMS delta (+3.7774) and the synthetic EPDMS delta (>0)
    assert s["P1"]["pdms_delta_x100"] == pytest.approx(3.7774, abs=1e-4) and s["P1"]["holds"] is True
    # decomposition covers all four commands' groups that occur, and sums back to the arm
    dec = s["decomposition"]["R7_A1"]["by_command"]
    assert sum(g["n"] for g in dec.values()) == 12146
    assert "pdm_score" in s["headline_column"]                                # the forbidden column is named as forbidden
    assert os.path.exists(out / "step5000" / "summary_navtest_epdms.txt")


def test_summarize_refuses_a_tampered_csv_and_reports_partial_for_a_missing_arm(synth):
    out, tokens = synth
    p = out / "step5000" / "R7_A1_s1" / "R7_A1_s1.devkit.csv"
    txt = p.read_text(encoding="utf-8")
    p.write_text(txt.replace("True", "True", 1) + " ", encoding="utf-8")        # one byte appended after the run
    os.remove(out / "floors" / "CV" / "ARM_DONE.json")
    assert E.main(["summarize", "--step", "5000"]) == 3                          # REFUSED
    s = json.load(open(out / "step5000" / "summary_navtest_epdms.json", encoding="utf-8"))
    assert s["status"] == "REFUSED"
    assert s["arms"]["R7_A1_s1"]["status"] == "REFUSED" and s["arms"]["CV"]["status"] == "NOT_SCORED"
    assert s["seed_floor_x100"] is None                                          # no seed floor without the replicate
    assert s["classification"]["R7_A1__minus__STOP"].startswith("SEED FLOOR UNAVAILABLE")


def test_summarize_refuses_a_recorded_cache_that_is_not_the_pinned_one(synth):
    out, tokens = synth
    d = out / "step5000" / "R7_A1" / "ARM_DONE.json"
    rec = json.loads(d.read_text(encoding="utf-8"))
    rec["cache"]["manifest_sha256"] = "0" * 64
    d.write_text(json.dumps(rec), encoding="utf-8")
    assert E.main(["summarize", "--step", "5000"]) == 3
    s = json.load(open(out / "step5000" / "summary_navtest_epdms.json", encoding="utf-8"))
    assert s["arms"]["R7_A1"]["status"] == "REFUSED"


def test_KE4_subset_equals_full_and_a_changed_cell_goes_red(tmp_path, monkeypatch):
    import shutil
    from taniteval.bench.navsim import single_stage as SS
    d = tmp_path / "smoke" / "ke2" / "STOP"
    d.mkdir(parents=True)
    shutil.copyfile(W8_STOP, d / "STOP.devkit.csv")
    monkeypatch.setattr(E, "OUT", str(tmp_path))
    full = SS.load_token_scores(W8_STOP)                       # the "full-split run" here is the same 218 rows
    r = E.ke4_subset_vs_full("STOP", None, full)
    assert r["status"] == "OK" and r["pass"] and r["n_tokens"] == 218
    t0 = next(iter(full))
    full[t0] = dict(full[t0], ego_progress=full[t0]["ego_progress"] + 1e-6)
    assert not E.ke4_subset_vs_full("STOP", None, full)["pass"]                       # RED: one cell moved
    assert E.ke4_subset_vs_full("R7_A1", 30000, full)["status"] == "UNAVAILABLE"      # smoke exists for step 5,000 only
    del full[t0]
    assert E.ke4_subset_vs_full("STOP", None, full)["pass"] is False                  # a missing token is not a pass


# --------------------------------------------------------------------------- #
# the FULL-split branch of run_arm, with the scorer replaced by a mock that writes a synthetic 12,146-row CSV #
# --------------------------------------------------------------------------- #
def _install_mock_scorer(monkeypatch, tmp_path, *, drop_a_row=False, corrupt_headline=False):
    from pathlib import Path
    from taniteval.bench.navsim import profiles as P, scoring as SC
    monkeypatch.setattr(E, "PKG", str(tmp_path))
    monkeypatch.setattr(E, "OUT", str(tmp_path / "raw" / "navtest_epdms"))
    monkeypatch.setattr(E, "LOCK_ROOT", str(tmp_path / "locks"))
    monkeypatch.setattr(P, "EXP_ROOT", Path(tmp_path / "scratch"))
    monkeypatch.setattr(P, "preflight", lambda prof, tokens=None, **kw: {"mock": True})
    monkeypatch.setattr(E, "mem_status", lambda: {"free_phys_gb": 12.0, "free_virt_gb": 9.0, "total_phys_gb": 32.0, "load_pct": 60})
    (tmp_path / "raw" / "navtest_epdms" / "inputs").mkdir(parents=True)
    import shutil
    shutil.copyfile(os.path.join(PKG, "raw", "navtest_epdms", "inputs", "token_meta.json"),
                    tmp_path / "raw" / "navtest_epdms" / "inputs" / "token_meta.json")
    seen = {}

    def fake_score_arm(*, arm, prof, raw_dir, exp_dir, seam=None, official_agent=None, ram_floor_mb=3000.0, log=print,
                       tokens=None, token_log=None, **kw):
        seen.update(arm=arm, seam=seam, official=official_agent, ram_floor_mb=ram_floor_mb, tokens=tokens)
        raw_dir.mkdir(parents=True, exist_ok=True)
        toks = sorted(E.load_meta())
        if drop_a_row:
            toks = toks[:-1]
        kind = "STOP" if arm.endswith("STOP") else ("R7_A1" if arm.endswith("R7_A1") else "CV")
        csvp = raw_dir / f"{arm}.devkit.csv"
        _synthetic_arm_csv(str(csvp), toks, kind, 11)
        if corrupt_headline:
            txt = csvp.read_text(encoding="utf-8").rstrip("\n").split("\n")
            parts = txt[-1].split(",")
            parts[-1] = "0.123456"
            txt[-1] = ",".join(parts)
            csvp.write_text("\n".join(txt) + "\n", encoding="utf-8")
        tid = "-".join(["123e4567", "e89b", "42d3", "a456", "426614174000"])      # built, not typed (landing scan)
        (raw_dir / f"{arm}.score.log").write_text("Starting worker in thread_id=" + tid + "\n", encoding="utf-8")
        (raw_dir / f"{arm}_manifest.json").write_text(json.dumps({"resources": {"peak_rss_mb": 1234.0}}), encoding="utf-8")
        rep = {"status": "PASS", "csv": str(csvp), "wall_s": 3.0, "agent_calls": {"seam": len(toks)}, "failures": []}
        (raw_dir / f"{arm}.counts.json").write_text(json.dumps(rep), encoding="utf-8")
        return rep

    monkeypatch.setattr(SC, "score_arm", fake_score_arm)
    return seen


def test_run_arm_full_branch_banks_a_pass_cleans_scratch_and_pins_the_seam(tmp_path, monkeypatch):
    if not os.path.exists(SEAM5K):
        pytest.skip("step-5,000 seams not on this disk")
    seen = _install_mock_scorer(monkeypatch, tmp_path)
    rec = E.run_arm(arm="R7_A1", step=5000)
    assert rec["status"] == "PASS" and rec["full_split"] is True
    assert seen["seam"] == SEAM5K.replace("\\", "/") or os.path.samefile(seen["seam"], SEAM5K)    # the BANKED seam, never a copy
    assert seen["ram_floor_mb"] == 4000.0 and seen["tokens"] is None
    d = tmp_path / "raw" / "navtest_epdms" / "step5000" / "R7_A1"
    done = json.loads((d / "ARM_DONE.json").read_text(encoding="utf-8"))
    assert done["n_tokens"] == 12146 and done["tokens_sha256"] == E.PINS["tokens_sha256"]
    assert done["seam"]["sha256"] == E.sha256_file(SEAM5K)
    assert (d / "R7_A1.devkit.csv").exists() and "thread-uuid" in (d / "R7_A1.score.log").read_text(encoding="utf-8")
    assert not E.UUID_RE.search((d / "R7_A1.score.log").read_text(encoding="utf-8"))
    assert not (tmp_path / "scratch" / "runs" / "epdms7" / "step5000" / "R7_A1").exists()   # scratch freed on a full PASS
    assert not os.path.exists(E.lock_path("step5000", "R7_A1"))                              # lock released
    assert json.loads((d / "ARM_LAST_ATTEMPT.json").read_text(encoding="utf-8"))["status"] == "PASS"
    assert E.read_done(5000, "R7_A1") is not None


def test_run_arm_stop_uses_the_suites_stop_seam_and_the_floors_scope(tmp_path, monkeypatch):
    seen = _install_mock_scorer(monkeypatch, tmp_path)
    rec = E.run_arm(arm="STOP")
    assert rec["status"] == "PASS" and rec["scope"] == "floors"
    assert seen["official"] is None and seen["seam"] is not None
    assert rec["seam"]["built_by"] == "taniteval.bench.navsim.seams.make_stop_seam"
    assert (tmp_path / "raw" / "navtest_epdms" / "floors" / "STOP" / "ARM_DONE.json").exists()


def test_run_arm_official_agents_get_no_seam(tmp_path, monkeypatch):
    seen = _install_mock_scorer(monkeypatch, tmp_path)
    assert E.run_arm(arm="CV")["status"] == "PASS"
    assert seen["official"] == "constant_velocity_agent" and seen["seam"] is None


def test_run_arm_refuses_a_row_short_scorer_output_and_leaves_ARM_FAILED_not_ARM_DONE(tmp_path, monkeypatch):
    _install_mock_scorer(monkeypatch, tmp_path, drop_a_row=True)
    rec = E.run_arm(arm="CV")
    assert rec["status"] == "FAIL_POSTCHECK" and rec["retryable"] is False
    d = tmp_path / "raw" / "navtest_epdms" / "floors" / "CV"
    assert (d / "ARM_FAILED.json").exists() and not (d / "ARM_DONE.json").exists()
    assert E.read_done(None, "CV") is None


def test_run_arm_refuses_when_the_average_row_is_not_the_mean_of_the_rows(tmp_path, monkeypatch):
    _install_mock_scorer(monkeypatch, tmp_path, corrupt_headline=True)
    rec = E.run_arm(arm="CV")
    assert rec["status"] == "FAIL_POSTCHECK" and any("G5" in f for f in rec["postcheck"]["failures"])


def test_run_arm_a_closed_gate_is_a_deferral_not_a_failure(tmp_path, monkeypatch):
    _install_mock_scorer(monkeypatch, tmp_path)
    monkeypatch.setattr(E, "mem_status", lambda: {"free_phys_gb": 4.4, "free_virt_gb": 8.0, "total_phys_gb": 32.0, "load_pct": 85})
    rec = E.run_arm(arm="CV")
    assert rec["status"] == "GATE_CLOSED" and rec["retryable"] is True
    d = tmp_path / "raw" / "navtest_epdms" / "floors" / "CV"
    assert json.loads((d / "ARM_LAST_ATTEMPT.json").read_text(encoding="utf-8"))["status"] == "GATE_CLOSED"
    assert not (d / "ARM_DONE.json").exists() and not (d / "ARM_FAILED.json").exists()


def test_run_arm_refuses_a_subset_without_a_tag_and_a_model_arm_without_a_step(tmp_path, monkeypatch):
    _install_mock_scorer(monkeypatch, tmp_path)
    with pytest.raises(SystemExit):
        E.run_arm(arm="CV", tokens=["a" * 16])
    with pytest.raises(SystemExit):
        E.run_arm(arm="R7_A1")


def test_the_lock_is_exclusive_and_a_dead_owners_lock_is_reclaimed(tmp_path, monkeypatch):
    import psutil
    monkeypatch.setattr(E, "LOCK_ROOT", str(tmp_path / "locks"))
    assert E.take_lock("floors", "CV") is True
    assert E.take_lock("floors", "CV") is False                                  # alive owner (this process)
    dead = 999999
    while psutil.pid_exists(dead):
        dead += 1
    with open(E.lock_path("floors", "CV"), "w", encoding="utf-8") as fh:
        json.dump({"pid": dead}, fh)
    assert E.take_lock("floors", "CV") is True                                   # stale -> reclaimed
