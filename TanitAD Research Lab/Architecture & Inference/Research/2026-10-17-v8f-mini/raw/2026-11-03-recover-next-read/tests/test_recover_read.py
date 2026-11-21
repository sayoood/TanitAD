"""Known-answer, rule and mutation tests of code/recover_read.py (the RECOVER-FT + COMPOSITE reader).

Run:  C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I -m pytest -q tests/test_recover_read.py        (CPU only; the slow tests take ~3 minutes together)
The reader under test is loaded BY FILE PATH (env RECOVER_READ_PATH overrides, which is how the mutation tests aim at a mutated copy).  No second stack root is ever put on sys.path.
Known answers are computed on EXISTING arrays: KAT1 (R := C0 := the DAG-OACC SREP runs), KAT2 (R := HISTSWAP's logged-history run, C0 := SREP), KAT3 (composite path: RC := FEAS2's KBOX runs).
"""
import importlib.util
import json
import os
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
PK = os.path.dirname(HERE)
SRC = f"{PK}/code/recover_read.py"
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
ART = "E:/Projects/TanitAD-artifacts"
DL = f"{ART}/dagland"


def _load(path):
    spec = importlib.util.spec_from_file_location("recover_read_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = _load(os.environ.get("RECOVER_READ_PATH", SRC))


@pytest.fixture(scope="module", autouse=True)
def heavy():
    R.heavy_init()


# ------------------------------------------------------------------------------------------------ helpers
def cell(better, share, xy=-0.03, yp=0.05):
    return {"better": better, "xy_delta": xy, "yp_delta": yp, "yp_separated_worse": True, "share_closed": share}


def rows(delta=-1.0, lo=-2.0, hi=-0.5, sep_worse=(), unsep=()):
    """19 rows, all separated on the BETTER side (TAC rows: positive), except ``sep_worse`` (separated worse) and ``unsep`` (CI straddles 0)."""
    out = {}
    for lab in R.GD.LABELS:
        up = lab in R.GD.HIGHER_BETTER
        d = -delta if up else delta
        lo_, hi_ = (-hi, -lo) if up else (lo, hi)
        out[lab] = {"delta": d, "lo": lo_, "hi": hi_, "separated": True}
        if lab in unsep:
            out[lab] = {"delta": 0.001, "lo": -0.5, "hi": 0.5, "separated": False}
        if lab in sep_worse:
            out[lab] = {"delta": -d, "lo": -hi_, "hi": -lo_, "separated": True}
    return out


# ------------------------------------------------------------------------------------------------ rule tests (pure layer)
def test_primary_rule_edges():
    ok = {"s0": cell(True, 0.5), "s1": cell(True, 0.5)}
    assert R.primary_rule(ok)["PASS"] is True                                   # mean exactly 0.5 passes
    assert R.primary_rule({"s0": cell(True, 0.4999), "s1": cell(True, 0.4999)})["PASS"] is False
    assert R.primary_rule({"s0": cell(True, 0.9), "s1": cell(False, 0.9)})["PASS"] is False   # BOTH seeds must be separated-better
    und = R.primary_rule({"s0": cell(True, 0.9), "s1": cell(True, None)})       # an undefined share counts 0: mean 0.45
    assert und["PASS"] is False and abs(und["mean_share_closed"] - 0.45) < 1e-12


def test_m1_cell_share_definition():
    c = R.m1_cell({"delta": -0.03, "lo": -0.04, "hi": -0.02, "separated": True}, {"delta": 0.06, "lo": 0.04, "hi": 0.08, "separated": True})
    assert c["better"] and abs(c["share_closed"] - 0.5) < 1e-12
    assert R.m1_cell({"delta": -0.03, "lo": -0.04, "hi": -0.02, "separated": True}, {"delta": 0.0, "lo": -0.01, "hi": 0.01, "separated": False})["share_closed"] is None


def test_guards_rule():
    ok = R.SR.read_xy(R.GD, rows(delta=0.0, lo=-1.0, hi=1.0, unsep=R.GD.LABELS))
    assert R.guards_rule({"s0": ok, "s1": ok})["PASS"] is True
    over = R.SR.read_xy(R.GD, {**rows(unsep=R.GD.LABELS), "ADE6": {"delta": 0.0605, "lo": -1.0, "hi": 1.0, "separated": False}})
    assert R.guards_rule({"s0": ok, "s1": over})["PASS"] is False               # ADE6 point above +0.0604 at ONE seed
    worse = R.SR.read_xy(R.GD, rows(unsep=R.GD.LABELS, sep_worse=("heading MAE",)))
    assert R.guards_rule({"s0": worse, "s1": ok})["PASS"] is False              # a separated-worse non-crawl row
    crawl = R.SR.read_xy(R.GD, rows(unsep=R.GD.LABELS, sep_worse=("M1 crawl share",)))
    assert R.guards_rule({"s0": crawl, "s1": ok})["PASS"] is True               # M1 itself is a crawl row, not a guard


def test_floor_rule_is_strict_and_per_seed():
    assert R.floor_rule({"s0": -0.03, "s1": -0.03}, {"s0": 0.01, "s1": 0.01})["PASS"] is True
    assert R.floor_rule({"s0": -0.03, "s1": -0.03}, {"s0": 0.03, "s1": 0.01})["PASS"] is False      # equal is not greater
    assert R.floor_rule({"s0": -0.03, "s1": -0.01}, {"s0": 0.01, "s1": 0.02})["PASS"] is False      # one seed inside the floor


def test_label_table_every_row():
    assert R.label(True, True, True, "ROBUST") == "RECOVERS"
    assert R.label(True, True, True, "ARTEFACT") == "ARTEFACT"
    assert R.label(True, True, True, "PARTIAL") == "PARTIAL"
    assert R.label(True, False, True, "ROBUST") == "HARMS"                      # G fails: HARMS, whatever P says
    assert R.label(False, False, False, "PARTIAL") == "HARMS"
    assert R.label(False, True, True, "ROBUST") == "NOT MOVED"


def test_label_inside_floor():
    """P and G hold, F fails at one or both seeds: INSIDE-FLOOR, never RECOVERS (the floor clause; mutated below)."""
    assert R.label(True, True, False, "ROBUST") == "INSIDE-FLOOR"
    assert R.label(True, True, False, "ARTEFACT") == "INSIDE-FLOOR"


def test_freeze_reading_rules():
    robust = R.freeze_reading(True, {"s0": cell(True, 0.5), "s1": cell(True, 0.6)})
    assert robust["reading"] == "ROBUST"
    assert R.freeze_reading(True, {"s0": cell(True, 0.9), "s1": cell(False, 0.9)})["reading"] == "ARTEFACT"        # separated-better at fewer than 2 seeds
    assert R.freeze_reading(True, {"s0": cell(True, 0.3), "s1": cell(True, 0.1)})["reading"] == "ARTEFACT"         # mean 0.2 < 0.25 (both separated, ROBUST fails on the mean)
    assert R.freeze_reading(True, {"s0": cell(True, 0.3), "s1": cell(True, 0.3)})["reading"] == "PARTIAL"          # 0.25 <= mean < 0.5
    assert R.freeze_reading(False, {"s0": cell(False, 0.0), "s1": cell(False, 0.0)})["reading"] == "PARTIAL"       # P failed: ARTEFACT is not defined
    assert R.freeze_reading(True, {"s0": cell(True, 0.5), "s1": cell(True, 0.5)})["reading"] == "ROBUST"           # mean exactly 0.5


def test_read_a_block_strategic_paused_and_failing_terms():
    ok = R.read_a_block(R.SR, R.GD, rows())
    assert ok["DOMINANT"] and ok["DOMINANT_three_families_strategic_paused"] and not ok["failing_terms"]
    nostrat = R.read_a_block(R.SR, R.GD, rows(unsep=("S-POS@30", "S-POS@50", "S-POS@80")))
    assert nostrat["DOMINANT"] is False and nostrat["D3_by_family"]["STRATEGIC"] is False                  # the pause never lowers a bar
    assert nostrat["DOMINANT_three_families_strategic_paused"] is True and any("STRATEGIC" in t for t in nostrat["failing_terms"])
    assert "PAUSED" in nostrat["strategic_family"]
    d2 = R.read_a_block(R.SR, R.GD, rows(sep_worse=("M1 crawl share",)))
    assert d2["D2"] is False and d2["separated_worse_rows"] == ["M1 crawl share"] and d2["failing_terms"] == ["D2 (separated-worse: M1 crawl share)"]
    weak = R.read_a_block(R.SR, R.GD, rows(delta=-0.2))
    assert weak["D1"] is False                                                                              # ADE6 point -0.2 is above the -0.2415 bar


def _dom(v=True):
    return {"DOMINANT": v, "D1": v, "D2": v, "D3": v, "D3_by_family": {f: v for f in R.FAMILIES}, "separated_worse_rows": [], "DOMINANT_three_families_strategic_paused": v, "failing_terms": []}


def test_composite_rule_counts_and_attribution():
    rc = {"s0": _dom(), "s1": _dom()}
    nc = {"s0": _dom(False), "s1": _dom(False)}
    c = R.composite_rule(rc, nc, {"reading": "ROBUST"}, True)
    assert c["status"] == "DOMINANT, COUNTED" and c["counts"] is True and c["attribution"] == "R-SPECIFIC"
    assert R.composite_rule(rc, nc, {"reading": "PARTIAL"}, True)["counts"] is True
    art = R.composite_rule(rc, nc, {"reading": "ARTEFACT"}, True)
    assert art["status"] == "DOMINANT, RIG ARTEFACT" and art["counts"] is False
    assert R.composite_rule(rc, nc, None, True)["status"].startswith("DOMINANT, UNREAD")
    assert R.composite_rule(rc, nc, {"reading": "ROBUST"}, False)["status"].startswith("DOMINANT, UNREAD")
    assert R.composite_rule({"s0": _dom(), "s1": _dom(False)}, nc, {"reading": "ROBUST"}, True)["status"] == "NOT DOMINANT"     # BOTH seeds
    assert R.composite_rule(rc, rc, {"reading": "ROBUST"}, True)["attribution"] == "RECIPE-LEVEL"
    assert R.composite_rule(rc, {"s0": _dom(), "s1": _dom(False)}, {"reading": "ROBUST"}, True)["attribution"] == "MIXED"


def test_composite_artefact_makes_it_a_rig_artefact():
    """Dedicated to the ARTEFACT clause (mutated below): an ARTEFACT R must never leave the composite COUNTED."""
    c = R.composite_rule({"s0": _dom(), "s1": _dom()}, None, {"reading": "ARTEFACT"}, True)
    assert c["counts"] is False and "ARTEFACT" in c["status"]


# ------------------------------------------------------------------------------------------------ control tests on existing evidence
def _tokens(name):
    return open(f"{DL}/recover_ft_launch/box2_files/argv/{name}.train.txt", encoding="utf-8").read().split()


@pytest.mark.parametrize("s", [0, 1])
def test_argv_audit_on_the_box2_argv_files(s):
    a = R.argv_audit(_tokens(f"R_{s}"), _tokens(f"C0_{s}"), _tokens(f"C0rep_{s}"), s)
    assert a["PASS"], [k for k, v in a.items() if v is False]
    bad = _tokens(f"C0_{s}") + ["--recover-target", "rejoin"]                       # C0 must name --recover-p 0 ONLY
    assert R.argv_audit(_tokens(f"R_{s}"), bad, _tokens(f"C0rep_{s}"), s)["PASS"] is False
    rep = _tokens(f"C0rep_{s}")
    wrong_seed = [("0" if (i > 0 and rep[i - 1] == "--seed") else t) for i, t in enumerate(rep)]
    assert R.argv_audit(_tokens(f"R_{s}"), _tokens(f"C0_{s}"), wrong_seed, s)["PASS"] is False                      # C0rep seed must be 10 + s


def test_kr3_on_the_gate_evidence_of_R_0():
    sc = json.load(open(f"{DL}/recover_ft_launch/evidence/R_0/selfcheck.json", encoding="utf-8"))
    c = R.kr3_check(sc, 0)
    assert c["PASS"] and c["table_n"] == 433301 and c["overlap_with_eval"] == 0
    sc["info"]["recover"]["kb3"]["overlap_with_eval"] = 1
    assert R.kr3_check(sc, 0)["PASS"] is False
    assert R.kr3_check(json.load(open(f"{DL}/recover_ft_launch/evidence/R_0/selfcheck.json", encoding="utf-8")), 1)["PASS"] is False    # s1 expects the other bank sha


def test_kr1_and_kp1():
    r, c0 = {"recover_n": 90, "recover_eligible": 300, "recover_p": 0.3}, {"recover_n": 0, "recover_p": 0.0}
    assert R.kr1_check(r, c0, c0)["PASS"]
    assert R.kr1_check({"recover_n": 120, "recover_eligible": 300, "recover_p": 0.3}, c0, c0)["PASS"] is False        # 0.40 is outside [0.25, 0.35]
    assert R.kr1_check(r, {"recover_n": 1, "recover_p": 0.0}, c0)["PASS"] is False
    ref = json.load(open(f"{ART}/v8f-dagoacc/M2b_s0_dagoacc/summary.json", encoding="utf-8"))
    assert R.kp1_check(dict(ref), ref)["PASS"]
    assert R.kp1_check({**ref, "train_window_list_sha256": "0" * 64}, ref)["PASS"] is False


def test_run_table_is_the_cards_runs():
    t = R.run_table()
    assert len(t) == 15 and len({r["tag"] for r in t}) == 15
    assert [r["tag"] for r in t[:2]] == ["R_s0", "C0_s0"]                           # the primary contrast first
    roles = {}
    for r in t:
        roles.setdefault(r["role"], []).append(r)
    assert {k: len(v) for k, v in roles.items()} == {"R": 2, "C0": 2, "C0REP": 2, "KA1R": 1, "FZR": 2, "FZC0": 2, "RC": 2, "C0C": 2}
    assert all(r["feasible"] == "kbox" for r in roles["RC"] + roles["C0C"]) and all(r["feasible"] is None for r in t if r["role"] not in ("RC", "C0C"))
    assert all(r["scene"] == "hold_now" for r in roles["FZR"] + roles["FZC0"]) and all(r["scene"] == "logged" for r in t if r["role"] not in ("FZR", "FZC0"))
    assert all("sg72" in R.run_flags(r) for r in t)
    assert R.run_flags(roles["KA1R"][0]) == R.run_flags(roles["R"][0]) and roles["KA1R"][0]["ckpt_run"] == roles["R"][0]["ckpt_run"]
    assert roles["C0REP"][0]["ckpt_run"] == "M2b_s0_recft_C0rep" and roles["RC"][1]["ckpt_run"] == "M2b_s1_recft_R" and roles["C0C"][0]["ckpt_run"] == "M2b_s0_recft_C0"


def test_kk2_on_feas2_kbox_ticks_and_is_sensitive():
    p = "F:/TanitAD-scratch/feas2_t1/out/KBOX_DAGOACC_s0/ticks/ticks_KBOX_DAGOACC_s0.npz"
    if not os.path.exists(p):
        pytest.skip("FEAS2 scratch ticks absent")
    t = {k: v for k, v in np.load(p, allow_pickle=True).items() if k not in ("tag", "ckpt")}
    assert R.kk2_check(t)["PASS"]
    t2 = dict(t)
    t2["plan"] = t["plan_s"]                                                        # kbox not applied: the control must fail
    assert R.kk2_check(t2)["PASS"] is False


# ------------------------------------------------------------------------------------------------ known answers on existing arrays (B = 2000, seed 0)
@pytest.fixture(scope="module")
def kat(tmp_path_factory):
    out = {}
    for m in ("kat1", "kat2", "kat3"):
        d = tmp_path_factory.mktemp(m)
        assert R.main(["--mode", m, "--raw", str(d)]) == 0
        out[m] = json.load(open(f"{d}/{m.upper()}.json", encoding="utf-8"))
    return out


def test_kat1_R_equals_C0_every_delta_is_exactly_zero(kat):
    k = kat["kat1"]
    for s in ("s0", "s1"):
        for lab, r in k["seeds"][s]["R_minus_C0"].items():
            assert r["delta"] == 0.0 and r["lo"] == 0.0 and r["hi"] == 0.0 and r["separated"] is False, (s, lab, r)
    assert k["primary"]["PASS"] is False and k["primary"]["n_seeds_separated_better"] == 0 and k["primary"]["mean_share_closed"] == 0.0
    assert k["label"] == "NOT MOVED" and k["guards"]["PASS"] is True
    assert k["floor"]["PASS"] is False and all(p["abs_R_minus_C0"] == 0.0 and p["abs_C0rep_minus_C0"] > 0.0 for p in k["floor"]["per_seed"].values())      # the floor check is DEFINED (the REP arm differs)
    assert "PAUSED" in k["strategic_family"] and k["controls"]["file_controls"].startswith("NOT APPLICABLE")


def test_kat2_logged_history_closes_M1_and_reproduces_H_HISTSWAP_1(kat):
    k = kat["kat2"]
    hs = json.load(open(f"{DL}/histswap/raw/HISTSWAP_T1.json", encoding="utf-8"))["arms"]
    for s in ("s0", "s1"):
        mine, theirs = k["seeds"][s]["m1_cell_logged"], hs[f"CTRL_{s}"]["m1_cell"]
        assert mine["better"] and abs(mine["share_closed"] - theirs["share_closed"]) < 1e-12 and abs(mine["xy_delta"] - theirs["xy_delta"]) < 1e-12 and abs(mine["yp_delta"] - theirs["yp_delta"]) < 1e-12
    assert k["primary"]["PASS"] and 0.92 < k["primary"]["share_closed"]["s1"] < 0.93 and abs(k["primary"]["mean_share_closed"] - 0.9546) < 5e-4
    # the four-cell mean of H-HISTSWAP-1 (DAG-OACC and SOUP) reproduces 0.9395 through this reader's own cell function
    cells = [mine for mine in (k["seeds"][s]["m1_cell_logged"] for s in ("s0", "s1"))]
    for s in (0, 1):
        z, y = R.read_arrays(f"{DL}/histswap/raw/arrays/HSW_SOUP_s{s}.npz"), R.read_arrays(f"{DL}/kappa_t1/raw/arrays/SREP_SOUP_s{s}.npz")
        cls, _ = R.GD.classify(z["v0"], z["g"])
        xy, _ = R.rows_of(R.Aof(z), R.Aof(y), z["eid"], cls, 2000)
        yp, _ = R.rows_of(R.Aof(y), R.Bof(y), z["eid"], cls, 2000)
        cells.append(R.m1_cell(xy[R.M1], yp[R.M1]))
    assert abs(float(np.mean([c["share_closed"] for c in cells])) - 0.9394581303675301) < 1e-9
    assert k["label"].startswith("UNREAD")                                          # no frame-freeze roles here: the label refuses to name RECOVERS


def test_kat3_composite_path_reproduces_FEAS2_kbox_D2_rows(kat):
    k = kat["kat3"]
    fe = json.load(open(f"{DL}/feas2_t1/raw/FEAS2_T1.json", encoding="utf-8"))["arms"]
    for s in ("s0", "s1"):
        for mine, theirs in ((k["seeds"][s]["RC_minus_P10"], fe[f"CTRL_{s}:kbox"]["rows"]["X_minus_P10"]), (k["seeds"][s]["RC_minus_R"], fe[f"CTRL_{s}:kbox"]["rows"]["X_minus_Y"])):
            for lab, r in theirs.items():
                assert all(abs(mine[lab][q] - r[q]) < 1e-12 for q in ("delta", "lo", "hi")) and mine[lab]["separated"] == r["separated"], (s, lab)
        b = k["composite"]["by_seed"][s]
        assert b["separated_worse_rows"] == ["M1 crawl share"] == fe[f"CTRL_{s}:kbox"]["read_a"]["separated_worse_rows"]      # M1 is the ONLY D2-failing row
        assert b["D1"] and b["D3"] and not b["D2"] and not b["DOMINANT"]
    assert k["composite"]["status"] == "NOT DOMINANT" and k["composite"]["DOMINANT"] is False


def test_label_end_to_end_on_real_arrays_and_freeze_refusal():
    """R := HISTSWAP's logged-history run, C0 := SREP, FZR := R and FZC0 := C0 (no frame motion difference: the gain survives the freeze) -> P, G, F hold and the freeze reads ROBUST -> RECOVERS;
    with the freeze controls failed the contrast is REFUSED: the label says UNREAD and a DOMINANT composite is not counted."""
    arrays = R.kat_arrays("kat2")
    for s in R.SEEDS:
        arrays[s]["FZR"], arrays[s]["FZC0"] = arrays[s]["R"], arrays[s]["C0"]
    ok, _ = R.evaluate(arrays, 300, log=lambda *a: None)
    assert ok["primary"]["PASS"] and ok["guards"]["PASS"] and ok["floor"]["PASS"] and ok["frame_freeze"]["reading"] == "ROBUST" and ok["label"] == "RECOVERS"
    assert ok["frame_freeze"]["applies_to_label"] is True
    refused, _ = R.evaluate(arrays, 300, log=lambda *a: None, freeze_usable=False, freeze_refusal=["KFZ4"])
    assert refused["label"].startswith("UNREAD") and refused["frame_freeze"]["reading"] is None and "REFUSED" in refused["frame_freeze"]["refused"]
    assert refused["primary"]["PASS"] == ok["primary"]["PASS"]                      # only the freeze contrast is refused; the rest is still read
    # an ARTEFACT freeze (the gain vanishes when frames are held) names the label ARTEFACT
    for s in R.SEEDS:
        arrays[s]["FZR"] = arrays[s]["FZC0"]                                          # FZ-R == FZ-C0: no M1 gain under the freeze
    art, _ = R.evaluate(arrays, 300, log=lambda *a: None)
    assert art["frame_freeze"]["reading"] == "ARTEFACT" and art["label"] == "ARTEFACT"


@pytest.mark.slow
def test_extraction_wiring_reproduces_a_banked_array(tmp_path):
    """extract() on KAPPA-T1's SREP_DAGOACC_s0 rollout reproduces its banked per-window array bit for bit (proves the extractor command line and its determinism)."""
    if not os.path.exists("F:/TanitAD-scratch/kappa_t1/out/SREP_DAGOACC_s0/ticks/ticks_SREP_DAGOACC_s0.npz"):
        pytest.skip("KAPPA-T1 scratch rollout absent")
    dst = R.extract("SREP_DAGOACC_s0", 0, "F:/TanitAD-scratch/kappa_t1/out", str(tmp_path))
    a, b = R.read_arrays(dst), R.read_arrays(f"{DL}/kappa_t1/raw/arrays/SREP_DAGOACC_s0.npz")
    same = lambda x, y: np.array_equal(x, y, equal_nan=(x.dtype.kind in "fc"))   # noqa: E731  (the labels array is a string array)
    assert set(a) == set(b) and all(same(a[k], b[k]) for k in a)


# ------------------------------------------------------------------------------------------------ mutation tests: delete a clause, a named test must go RED
MUTATIONS = {
    "floor clause in label": ('    if not F:\n        return "INSIDE-FLOOR"\n', "", ["test_label_inside_floor"]),
    "floor comparison in floor_rule": ('p["exceeds_floor"] = bool(p["abs_R_minus_C0"] > p["abs_C0rep_minus_C0"])', 'p["exceeds_floor"] = True', ["test_floor_rule_is_strict_and_per_seed"]),
    "share bar in primary_rule": ('mean >= SHARE_BAR)}', 'True)}', ["test_primary_rule_edges"]),
    "ARTEFACT clause in freeze_reading": ('elif p_logged and (n < 2 or mean < ARTEFACT_BAR):', 'elif False:', ["test_freeze_reading_rules"]),
    "ARTEFACT clause in composite_rule": ('elif rd == "ARTEFACT":', 'elif False:', ["test_composite_artefact_makes_it_a_rig_artefact"]),
    "ADE6 guard in guards_rule": ('v["ade6_delta"] <= ADE_GUARD_M', 'True', ["test_guards_rule"]),
}


@pytest.mark.parametrize("name", list(MUTATIONS))
def test_mutation_goes_red(name, tmp_path):
    old, new, nodes = MUTATIONS[name]
    src = open(SRC, encoding="utf-8").read()
    assert src.count(old) == 1, f"the clause to delete is not present exactly once: {name}"        # a vacuous mutation proves nothing
    mut = tmp_path / "recover_read_mut.py"
    mut.write_text(src.replace(old, new, 1), encoding="utf-8", newline="\n")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONDONTWRITEBYTECODE="1")
    cmd = lambda path: [PY, "-I", "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider"] + [f"{HERE}/test_recover_read.py::{n}" for n in nodes]   # noqa: E731
    clean = subprocess.run(cmd(SRC), env=dict(env, RECOVER_READ_PATH=SRC), capture_output=True, text=True)
    assert clean.returncode == 0, "the control (unmutated reader) must be GREEN:\n" + clean.stdout[-600:]
    red = subprocess.run(cmd(str(mut)), env=dict(env, RECOVER_READ_PATH=str(mut)), capture_output=True, text=True)
    assert red.returncode != 0, f"mutation '{name}' left {nodes} GREEN: the test cannot see this clause"
