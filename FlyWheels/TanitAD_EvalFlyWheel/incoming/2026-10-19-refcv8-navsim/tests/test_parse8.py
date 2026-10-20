"""parse8 against LITERAL targets (TANITAD VENV, CPU, no GPU, no checkpoint, no network):

    pytest -q -p no:cacheprovider tests/test_parse8.py -k "not control"      # the fast synthetic + e2e-on-synthetic part
    pytest -q -p no:cacheprovider tests/test_parse8.py                       # + the refcv7 banked-input controls

Expectations are LITERALS written here (hand arithmetic on tiny synthetic splits, and numbers read ONCE from refcv7's banked
summaries -- the paths are cited at the literal), never an expression over the code under test -- CLAUDE.md: "a check that shares
the defect it checks for is green forever".  ``tests/test_mutation_parse8.py`` re-runs this file in fresh interpreters with
``PARSE8_MUTATION`` set and proves the named tests go RED.

Tests with ``control`` in the name read the banked refcv7 step-30000 NavSim inputs on D: / C: and SKIP (stated, not a pass) when those
disks are absent.  Tests with ``e2e`` import ``navsim_bars8`` (hold or landed copy) and SKIP when it is absent.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PKG, "code"))
import parse8 as P  # noqa: E402

# ------------------------------------------------------------------------------------------------------------------------------
# synthetic splits (hand-computable)
# ------------------------------------------------------------------------------------------------------------------------------
SUBCOLS = list(P.SUB.values())
S2_TOK = ["a00", "a01", "b00", "b01", "a10", "a11", "b10", "b11"]          # warmup stage-2 tokens (2 keys x 2 groups x 2)
S1_TOK = ["o0", "p0", "o1", "p1"]
W_MAPPING = [["o0", "p0", [["a00", "b00"], ["a01", "b01"]]], ["o1", "p1", [["a10", "b10"], ["a11", "b11"]]]]
# LEGAL stage-2 scores: groups [a00,a01]=.2,.4 -> .3 ; [b00,b01]=.6,.8 -> .7 ; [a10,a11]=.5,.5 -> .5 ; [b10,b11]=.1,.3 -> .2
LEGAL_S2 = dict(zip(S2_TOK, [0.2, 0.4, 0.6, 0.8, 0.5, 0.5, 0.1, 0.3]))        # S2-EPDMS-u = (.3+.7+.5+.2)/4 = 0.425
WLOG = "2021.01.01.00.00.00_veh-1_00000_00001"


def _const(v, toks=S2_TOK + S1_TOK):
    return {t: v for t in toks}


def _arm_scores(s2: dict, s1_val=0.9):
    d = dict(s2)
    d.update(_const(s1_val, S1_TOK))
    return d


def _write_csv(path, scores: dict, official=None, zero_nc=()):
    rows = []
    for t, s in scores.items():
        r = {"token": t, "valid": True, "score": s}
        for c in SUBCOLS:
            r[f"{c}_stage_one"] = 1.0
            r[f"{c}_stage_two"] = 1.0
        if t in zero_nc:
            r["no_at_fault_collisions_stage_two"] = 0.0
        rows.append(r)
    if official is not None:
        r = {"token": "extended_pdm_score_combined", "valid": True, "score": official}
        for c in SUBCOLS:
            r[f"{c}_stage_one"] = np.nan
            r[f"{c}_stage_two"] = np.nan
        rows.append(r)
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_seam(bridge, arm, tokens, sources, end_x):
    poses = np.zeros((len(tokens), 2, 3))
    poses[:, -1, 0] = end_x
    np.savez(os.path.join(bridge, f"seam_{arm}.npz"), source=np.array(sources), token=np.array(tokens), poses=poses)
    man = {"model": {"step": 48852, "ckpt_md5": "0123456789abcdef0123456789abcdef"}, "device": ["cuda"], "precision": ["bf16"]}
    with open(os.path.join(bridge, f"seam_{arm}.manifest.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(man, fh)


def _write_counts(scores_dir, stem, status="PASS"):
    with open(os.path.join(scores_dir, f"score_{stem}.counts.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump({"status": status}, fh)


def build_warmup(root, *, omit=(), counts_fail=(), own_stage1=False):
    """A 2-key / 4-group warmup-shaped split.  Returns the kwargs of ``parse8.build_summary``."""
    root = str(root)
    scores, floors, bridge = (os.path.join(root, d) for d in ("scores_warmup", "floors", "bridge_warmup"))
    for d in (scores, floors, bridge):
        os.makedirs(d, exist_ok=True)
    inputs = os.path.join(root, "navsim_agent_inputs.json")
    toks = {t: {"stage": 1 if t in S1_TOK else 2, "log_name": WLOG} for t in S1_TOK + S2_TOK}
    with open(inputs, "w", encoding="utf-8", newline="\n") as fh:
        json.dump({"tokens": toks, "reactive_all_mapping": W_MAPPING}, fh)
    plus = lambda d, x: {t: v + x for t, v in d.items()}                      # noqa: E731
    model = {"R8_LEGAL": (LEGAL_S2, 0.5), "R8_LEGAL_s1": (plus(LEGAL_S2, 0.01), 5.0),       # u = .435 -> seed floor .01
             "R8_BLIND": (_const(0.1, S2_TOK), 5.0), "R8_NAVOFF": (plus(LEGAL_S2, 0.0005), 5.0),   # u = .4255
             "PRIOR_ha0p": (_const(0.2, S2_TOK), 5.0)}
    end_legal = [0.5, 3.0, 0.2, 10.0, 5.0, 5.0, 5.0, 5.0]                    # per S2 token -> 2/8 below 1 m, median 5.0
    for arm, (s2, _) in model.items():
        if arm in omit:
            continue
        sc = _arm_scores(s2)
        _write_csv(os.path.join(scores, f"score_{arm}.csv"), sc, official=0.123, zero_nc=("a00",) if arm == "R8_LEGAL" else ())
        _write_counts(scores, arm, "FAIL" if arm in counts_fail else "PASS")
        tk = S1_TOK + S2_TOK
        src = ["model" if own_stage1 else "cv_standin"] * len(S1_TOK) + ["model"] * len(S2_TOK)
        _write_seam(bridge, arm, tk, src, [5.0] * len(S1_TOK) + (end_legal if arm == "R8_LEGAL" else [5.0] * len(S2_TOK)))
    for arm, v, off in (("CV_official", 0.3, 0.11), ("STOP_zero", 0.4, 0.3), ("ECHO_ha0_ext", 0.35, 0.22)):
        if arm in omit:
            continue
        _write_csv(os.path.join(floors, f"score_{arm}.csv"), _arm_scores(_const(v, S2_TOK), s1_val=v), official=off)
    return dict(split="warmup_two_stage", scores=scores, bridge=bridge, inputs=inputs, label="RESULT", floors=floors)


NK = 8                                                                          # 8 mapping keys -> 8 logs -> 8 drives (= the floor)
N_MAPPING = [[f"o{k}", f"p{k}", [[f"a{k}", f"b{k}"]]] for k in range(NK)]
N_LOG = lambda k: f"2021.01.01.00.{k:02d}.00_veh-1_00000_00001"                 # noqa: E731  one drive per log


def build_navhard(root, *, omit=(), wrapper_omit=()):
    """8 keys, constant per-arm scores s => every key contributes (s*s + s*s)/2 = s^2 (written as LITERAL officials)."""
    root = str(root)
    sfx = "__navhard_two_stage"
    scores, floors, bridge = (os.path.join(root, d) for d in ("scores_navhard", "floors", "bridge_navhard"))
    for d in (scores, floors, bridge):
        os.makedirs(d, exist_ok=True)
    toks, log_of = {}, {}
    for k in range(NK):
        for t, st in ((f"o{k}", 1), (f"p{k}", 1), (f"a{k}", 2), (f"b{k}", 2)):
            toks[t] = {"stage": st, "log_name": N_LOG(k)}
    inputs = os.path.join(root, "navsim_agent_inputs.json")
    with open(inputs, "w", encoding="utf-8", newline="\n") as fh:
        json.dump({"tokens": toks, "reactive_all_mapping": N_MAPPING}, fh)
    all_t = list(toks)
    spec = {  # arm -> (constant score s, official two-stage EPDMS = s^2 written as a literal)
        "R8_LEGAL": (0.6, 0.36), "R8_LEGAL_s1": (0.59, 0.3481), "PRIOR_ha0p": (0.45, 0.2025),
        "CV_official": (0.2, 0.04), "STOP_zero": (0.5, 0.25), "ECHO_ha0_ext": (0.3, 0.09)}
    for arm, (s, off) in spec.items():
        if arm in omit:
            continue
        model = arm in P.MODEL_ARMS
        d = scores if model else floors
        _write_csv(os.path.join(d, f"score_{arm}{sfx}.csv"), _const(s, all_t), official=off)
        wd = os.path.join(d, f"score_{arm}{sfx}_wrapper")
        os.makedirs(wd, exist_ok=True)
        if arm not in wrapper_omit:
            pd.DataFrame({"token": all_t, "score": [s] * len(all_t), "weight": [1.0] * len(all_t),
                          "log_name": [toks[t]["log_name"] for t in all_t]}).to_csv(
                os.path.join(wd, f"{arm}{sfx}_final_scores_frame.csv"), index=False)
        if model:
            _write_counts(scores, f"{arm}{sfx}")
            _write_seam(bridge, arm, all_t, ["model"] * len(all_t), [5.0] * len(all_t))
    return dict(split="navhard_two_stage", scores=scores, bridge=bridge, inputs=inputs, label="RESULT", floors=floors,
                csv_suffix=sfx)


def approx(v):
    return pytest.approx(v, abs=1e-12)


# ------------------------------------------------------------------------------------------------------------------------------
# 1. warmup, synthetic, literal
# ------------------------------------------------------------------------------------------------------------------------------
def test_warmup_s2_epdms_u_literal(tmp_path):
    out = P.build_summary(**build_warmup(tmp_path))
    u = {a: r["S2_EPDMS_u"]["value"] for a, r in out["arms"].items()}
    # hand arithmetic: LEGAL groups .3 .7 .5 .2 -> .425 ; s1 = +.01 -> .435 ; NAVOFF = +.0005 -> .4255 ; BLIND/CV/STOP/ECHO/PRIOR constant
    assert u["R8_LEGAL"] == approx(0.425) and u["R8_LEGAL_s1"] == approx(0.435) and u["R8_NAVOFF"] == approx(0.4255)
    assert u["R8_BLIND"] == approx(0.1) and u["PRIOR_ha0p"] == approx(0.2)
    assert u["CV_official"] == approx(0.3) and u["STOP_zero"] == approx(0.4) and u["ECHO_ha0_ext"] == approx(0.35)
    assert out["arms"]["R8_LEGAL"]["S2_EPDMS_u"]["n_groups"] == 4 and out["arms"]["R8_LEGAL"]["n_stage2"] == 8
    assert out["arms"]["R8_LEGAL"]["n_stage1"] == 4


def test_arm_names_are_r8_and_no_r7_name_survives(tmp_path):
    out = P.build_summary(**build_warmup(tmp_path))
    names = set(out["arms"]) | {p for k in out["pairs"] for p in k.split("__minus__")}
    assert not [n for n in names if n.startswith("R7_")], names
    assert {"R8_LEGAL", "R8_LEGAL_s1", "R8_BLIND", "CV_official", "STOP_zero", "ECHO_ha0_ext"} <= set(out["arms"])
    assert "R8_LEGAL__minus__STOP_zero" in out["pairs"] and "R8_LEGAL__minus__R8_LEGAL_s1" in out["pairs"]


def test_seed_floor_literal_and_the_exceeds_flags(tmp_path):
    out = P.build_summary(**build_warmup(tmp_path))
    assert out["seed_floor_S2_EPDMS_u"] == approx(0.01)                     # |.425 - .435|
    assert out["pairs"]["R8_LEGAL__minus__R8_LEGAL_s1"]["S2_EPDMS_u_delta"] == approx(-0.01)
    ex = {k.split("__minus__")[1]: v["exceeds_seed_floor"] for k, v in out["pairs"].items()
          if k.startswith("R8_LEGAL__minus__") and "exceeds_seed_floor" in v}
    # deltas: CV .125, STOP .025, ECHO .075, BLIND .325, PRIOR .225 all > .01 ; NAVOFF |.425-.4255| = .0005 < .01
    assert ex == {"CV_official": True, "STOP_zero": True, "ECHO_ha0_ext": True, "R8_BLIND": True, "PRIOR_ha0p": True,
                  "R8_NAVOFF": False}


def test_seed_arms_are_not_swapped(tmp_path):
    out = P.build_summary(**build_warmup(tmp_path))
    assert out["arms"]["R8_LEGAL"]["S2_EPDMS_u"]["value"] == approx(0.425)          # seed 0 is the BAR arm
    assert out["arms"]["R8_LEGAL_s1"]["S2_EPDMS_u"]["value"] == approx(0.435)       # seed 1 is the floor arm
    assert out["arms"]["R8_LEGAL"]["stop_fraction"]["frac_endpoint_lt_1m"] == approx(0.25)    # only the BAR seam has 2/8 < 1 m


def test_blind_arm_is_read(tmp_path):
    out = P.build_summary(**build_warmup(tmp_path))
    assert out["arms"]["R8_BLIND"]["S2_EPDMS_u"]["value"] == approx(0.1)
    assert out["pairs"]["R8_LEGAL__minus__R8_BLIND"]["S2_EPDMS_u_delta"] == approx(0.325)
    assert out["arms"]["R8_BLIND"].get("status") is None


def test_wtl_counts_literal(tmp_path):
    out = P.build_summary(**build_warmup(tmp_path))
    # LEGAL - s1: every stage-2 delta = -.01 -> 0/0/8 ; LEGAL - STOP(.4): -.2 0 +.2 +.4 +.1 +.1 -.3 -.1 -> 4 wins, 1 tie, 3 losses
    p1, p2 = out["pairs"]["R8_LEGAL__minus__R8_LEGAL_s1"], out["pairs"]["R8_LEGAL__minus__STOP_zero"]
    assert (p1["wins"], p1["ties"], p1["losses"], p1["n_scenes"]) == (0, 0, 8, 8)
    assert (p2["wins"], p2["ties"], p2["losses"]) == (4, 1, 3)
    assert p2["scene_mean_delta"] == approx(0.025)


def test_stop_fraction_and_zero_rate_literal(tmp_path):
    out = P.build_summary(**build_warmup(tmp_path))
    sf = out["arms"]["R8_LEGAL"]["stop_fraction"]
    # endpoints 0.5 3.0 0.2 10 5 5 5 5 : two below 1 m -> 2/8 ; sorted .2 .5 3 5 5 5 5 10 -> median 5.0.  Stage-1 rows are cv_standin: excluded.
    assert sf["n"] == 8 and sf["frac_endpoint_lt_1m"] == approx(0.25) and sf["median_4s_distance_m"] == approx(5.0)
    assert out["arms"]["STOP_zero"]["stop_fraction"]["frac_endpoint_lt_1m"] == 1.0
    assert out["arms"]["R8_LEGAL"]["S2_multiplier_zero_rates"]["NC"] == approx(0.125)       # a00 zeroed: 1 of 8 stage-2 tokens
    assert out["arms"]["R8_LEGAL"]["S2_multiplier_zero_rates"]["DAC"] == 0.0


def test_cv_standin_stage1_is_not_reported_as_a_real_two_stage_epdms(tmp_path):
    out = P.build_summary(**build_warmup(tmp_path))
    r = out["arms"]["R8_LEGAL"]
    assert r["stage1_rows_own"] is False
    assert isinstance(r["official_two_stage_EPDMS"], dict) and r["official_two_stage_EPDMS"]["status"] == "UNAVAILABLE"
    assert r["official_rows_HYBRID"]["extended_pdm_score_combined"] == approx(0.123) and "official_rows" not in r
    assert out["arms"]["STOP_zero"]["official_two_stage_EPDMS"] == approx(0.3) and out["arms"]["STOP_zero"]["stage1_rows_own"] is True
    own = P.build_summary(**build_warmup(tmp_path / "own", own_stage1=True))["arms"]["R8_LEGAL"]
    assert own["stage1_rows_own"] is True and own["official_two_stage_EPDMS"] == approx(0.123)


def test_missing_expected_arm_reads_unavailable_never_a_value(tmp_path):
    out = P.build_summary(**build_warmup(tmp_path, omit=("R8_BLIND",)))
    b = out["arms"]["R8_BLIND"]
    assert b["status"] == "UNAVAILABLE" and "score_R8_BLIND.csv" in b["reason"] and "S2_EPDMS_u" not in b
    pr = out["pairs"]["R8_LEGAL__minus__R8_BLIND"]
    assert pr["status"] == "UNAVAILABLE" and "S2_EPDMS_u_delta" not in pr
    assert out["arms"]["R8_LEGAL"]["S2_EPDMS_u"]["value"] == approx(0.425)             # the others are untouched
    assert out["seed_floor_S2_EPDMS_u"] == approx(0.01)


def test_missing_floor_arm_reads_unavailable_never_a_value(tmp_path):
    out = P.build_summary(**build_warmup(tmp_path, omit=("STOP_zero",)))
    assert out["arms"]["STOP_zero"]["status"] == "UNAVAILABLE" and "S2_EPDMS_u" not in out["arms"]["STOP_zero"]
    assert out["pairs"]["R8_LEGAL__minus__STOP_zero"]["status"] == "UNAVAILABLE"


def test_missing_seed_floor_arm_gives_a_null_floor_with_a_reason(tmp_path):
    out = P.build_summary(**build_warmup(tmp_path, omit=("R8_LEGAL_s1",)))
    assert out["arms"]["R8_LEGAL_s1"]["status"] == "UNAVAILABLE"
    assert out["seed_floor_S2_EPDMS_u"] is None                                         # null, never a dict (bars8 does abs(floor))
    assert out["seed_floor_S2_EPDMS_u_status"]["status"] == "UNAVAILABLE"
    assert not any("exceeds_seed_floor" in v for v in out["pairs"].values())            # no lever is read against a floor that was not measured


def test_count_guard_failure_refuses_the_arm(tmp_path):
    out = P.build_summary(**build_warmup(tmp_path, counts_fail=("R8_LEGAL_s1",)))
    assert out["arms"]["R8_LEGAL_s1"]["status"] == "UNAVAILABLE" and "count guard FAIL" in out["arms"]["R8_LEGAL_s1"]["reason"]
    assert "R8_LEGAL_s1" in out["arms_refused"]
    assert out["seed_floor_S2_EPDMS_u"] is None


def test_a_pdm_score_column_is_refused(tmp_path):
    kw = build_warmup(tmp_path)
    p = os.path.join(kw["scores"], "score_R8_LEGAL.csv")
    df = pd.read_csv(p)
    df["pdm_score"] = 0.0
    df.to_csv(p, index=False)
    with pytest.raises(SystemExit, match="pdm_score"):
        P.build_summary(**kw)


def test_an_unknown_split_is_refused(tmp_path):
    kw = build_warmup(tmp_path)
    kw["split"] = "navtest"
    with pytest.raises(P.ParseError, match="unknown split"):
        P.build_summary(**kw)


def test_warmup_never_carries_an_interval(tmp_path):
    out = P.build_summary(**build_warmup(tmp_path))
    assert out["estimator"]["status"] == "UNAVAILABLE" and out["estimator"]["n"] == 7
    assert "intervals" not in out and "paired_intervals" not in out


def test_summary_is_json_roundtrippable_with_lf_endings(tmp_path):
    out = P.build_summary(**build_warmup(tmp_path))
    p = tmp_path / "s.json"
    P.write_summary(out, str(p))
    raw = p.read_bytes()
    assert b"\r\n" not in raw
    assert json.loads(raw.decode("utf-8"))["arms"]["R8_LEGAL"]["S2_EPDMS_u"]["value"] == approx(0.425)


def test_main_refuses_to_write_while_a_mutation_is_set(monkeypatch, tmp_path):
    monkeypatch.setattr(P, "MUTATION", "swap_seed_arms")
    kw = build_warmup(tmp_path)
    with pytest.raises(SystemExit, match="deliberate-regression"):
        P.main(["--split", "warmup_two_stage", "--scores", kw["scores"], "--bridge", kw["bridge"], "--inputs", kw["inputs"],
                "--label", "x", "--out", str(tmp_path / "o.json")])


def test_the_mutation_list_matches_the_module():
    assert set(P.MUTATIONS) == {"swap_seed_arms", "drop_blind_arm", "skip_count_guard", "silent_skip_missing", "ignore_cv_standin",
                                "s1_pair_reuses_s0"}


def test_warmup_seed1_mirror_pairs_are_point_deltas_only(tmp_path):
    out = P.build_summary(**build_warmup(tmp_path))
    # s1 = LEGAL + .01 on every stage-2 token -> u = .435.  vs STOP (.4): u delta .035 ; per-token deltas -.19 +.01 +.21 +.41 +.11 +.11 -.29 -.09
    # -> 5 wins, 0 ties, 3 losses.  (The seed-0 row is .025 and 4/1/3: a mirror fed from seed 0 would read those instead.)
    p = out["pairs"]["R8_LEGAL_s1__minus__STOP_zero"]
    assert p["S2_EPDMS_u_delta"] == approx(0.035) and (p["wins"], p["ties"], p["losses"]) == (5, 0, 3)
    assert out["pairs"]["R8_LEGAL_s1__minus__CV_official"]["S2_EPDMS_u_delta"] == approx(0.135)        # .435 - .3
    assert out["pairs"]["R8_LEGAL_s1__minus__ECHO_ha0_ext"]["S2_EPDMS_u_delta"] == approx(0.085)       # .435 - .35
    assert out["pairs"]["R8_LEGAL__minus__STOP_zero"]["S2_EPDMS_u_delta"] == approx(0.025)             # the seed-0 row is untouched
    assert "paired_intervals" not in out                                                                 # 7 logs < 8: warmup never has an interval


def test_warmup_seed1_mirror_reads_unavailable_when_the_s1_arm_is_absent(tmp_path):
    out = P.build_summary(**build_warmup(tmp_path, omit=("R8_LEGAL_s1",)))
    for y in ("STOP_zero", "CV_official", "ECHO_ha0_ext"):
        r = out["pairs"][f"R8_LEGAL_s1__minus__{y}"]
        assert r["status"] == "UNAVAILABLE" and "R8_LEGAL_s1" in r["reason"] and "S2_EPDMS_u_delta" not in r


# ------------------------------------------------------------------------------------------------------------------------------
# 2. navhard, synthetic, literal (constant arms: every identity below is exact, no bootstrap noise)
# ------------------------------------------------------------------------------------------------------------------------------
def test_navhard_constant_arms_literal(tmp_path):
    out = P.build_summary(**build_navhard(tmp_path))
    off = {a: r["official_two_stage_EPDMS"] for a, r in out["arms"].items()}
    assert off == {"R8_LEGAL": approx(0.36), "R8_LEGAL_s1": approx(0.3481), "PRIOR_ha0p": approx(0.2025),
                   "CV_official": approx(0.04), "STOP_zero": approx(0.25), "ECHO_ha0_ext": approx(0.09)}
    iv = out["intervals"]["R8_LEGAL"]
    # a constant arm: every resample of logs gives the same mean -> zero-width interval on the exact value; 8 logs = the floor
    assert iv["status"] == "OK" and iv["point"] == 0.36 and iv["lo"] == 0.36 and iv["hi"] == 0.36
    assert iv["n_clusters"] == 8 and iv["n_units"] == 8
    pi = out["paired_intervals"]["R8_LEGAL__minus__STOP_zero"]
    assert pi["status"] == "OK" and pi["lo"] == 0.11 and pi["hi"] == 0.11 and pi["n_clusters"] == 8     # .36 - .25
    assert out["pairs"]["R8_LEGAL__minus__STOP_zero"]["official_two_stage_EPDMS_delta"] == approx(0.11)


def test_navhard_seed_floor_is_the_official_delta_of_the_two_seed_arms(tmp_path):
    out = P.build_summary(**build_navhard(tmp_path))
    assert out["pairs"]["R8_LEGAL__minus__R8_LEGAL_s1"]["official_two_stage_EPDMS_delta"] == approx(0.0119)     # .36 - .3481
    assert out["seed_floor_S2_EPDMS_u"] == approx(0.01)                                                           # |.6 - .59|


def test_navhard_seed1_mirror_rows_literal(tmp_path):
    """A-R8-NS1 AMENDMENT 3: the bars hold on BOTH inference seeds.  s1 is the constant .59 arm (official .3481), s0 the .6 arm (.36):
    mirror deltas vs STOP .25 / CV .04 / ECHO .09 are .0981 / .3081 / .2581, and the seed-0 rows stay .11 / .32 / .27."""
    out = P.build_summary(**build_navhard(tmp_path))
    for y, d0, d1 in (("STOP_zero", 0.11, 0.0981), ("CV_official", 0.32, 0.3081), ("ECHO_ha0_ext", 0.27, 0.2581)):
        assert out["pairs"][f"R8_LEGAL_s1__minus__{y}"]["official_two_stage_EPDMS_delta"] == approx(d1), y
        assert out["pairs"][f"R8_LEGAL__minus__{y}"]["official_two_stage_EPDMS_delta"] == approx(d0), y
        pi = out["paired_intervals"][f"R8_LEGAL_s1__minus__{y}"]
        # constant arms -> zero-width interval on the exact paired delta, at BOTH levels (8 logs = 8 drives = the floor)
        assert pi["status"] == "OK" and pi["lo"] == d1 and pi["hi"] == d1 and pi["delta"] == d1, y
        assert pi["separated"] is True and pi["separated_log_name"] is True and pi["n_clusters"] == 8, y
        dr = pi["sensitivity_nuplan_drive"]
        assert dr["status"] == "OK" and dr["lo"] == d1 and dr["hi"] == d1 and dr["n_clusters"] == 8, y
        # SAME record shape as the R8_LEGAL row (same function), not a lookalike
        assert set(pi) == set(out["paired_intervals"][f"R8_LEGAL__minus__{y}"]), y
        assert set(dr) == set(out["paired_intervals"][f"R8_LEGAL__minus__{y}"]["sensitivity_nuplan_drive"]), y


def test_navhard_seed1_mirror_rows_read_unavailable_when_the_s1_arm_is_absent(tmp_path):
    out = P.build_summary(**build_navhard(tmp_path, omit=("R8_LEGAL_s1",)))
    for y in ("STOP_zero", "CV_official", "ECHO_ha0_ext"):
        for blk in ("pairs", "paired_intervals"):
            r = out[blk][f"R8_LEGAL_s1__minus__{y}"]
            assert r["status"] == "UNAVAILABLE" and "R8_LEGAL_s1" in r["reason"], (blk, y)
            assert "lo" not in r and "official_two_stage_EPDMS_delta" not in r
        assert out["paired_intervals"][f"R8_LEGAL__minus__{y}"]["status"] == "OK"                 # the seed-0 rows are untouched


def test_navhard_estimator_is_the_pinned_refcv7_one(tmp_path):
    out = P.build_summary(**build_navhard(tmp_path))
    e = out["estimator"]
    assert (e["name"], e["unit"], e["B"], e["seed"], e["aggregation"]) == (
        "navsim_log_cluster_bootstrap", "log_name", 2000, 0, "two_stage_mapping_key_mean")
    assert len(e["module_sha256"]) == 64


def test_navhard_a_missing_floor_arm_makes_its_paired_interval_unavailable(tmp_path):
    out = P.build_summary(**build_navhard(tmp_path, omit=("STOP_zero",)))
    assert out["arms"]["STOP_zero"]["status"] == "UNAVAILABLE"
    assert out["intervals"]["STOP_zero"]["status"] == "UNAVAILABLE"
    pi = out["paired_intervals"]["R8_LEGAL__minus__STOP_zero"]
    assert pi["status"] == "UNAVAILABLE" and "lo" not in pi
    assert out["paired_intervals"]["R8_LEGAL__minus__CV_official"]["status"] == "OK"     # the others are untouched


def test_navhard_a_missing_wrapper_frame_is_an_error_record_not_a_silent_skip(tmp_path):
    out = P.build_summary(**build_navhard(tmp_path, wrapper_omit=("R8_LEGAL",)))
    assert out["intervals"]["R8_LEGAL"]["status"] == "ERROR"
    assert out["paired_intervals"]["R8_LEGAL__minus__STOP_zero"]["status"] == "UNAVAILABLE"


# ------------------------------------------------------------------------------------------------------------------------------
# 3. e2e: the consumer (navsim_bars8) accepts parse8's output
# ------------------------------------------------------------------------------------------------------------------------------
_BARS_CANDIDATES = (
    os.environ.get("PARSE8_BARS8_CODE", ""),
    os.path.join(PKG, "code"),
    "C:/Users/Admin/qland/hold_r8nsbars/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-19-refcv8-navsim/code",
    "C:/Users/Admin/qland/landed_hold_r8nsbars/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-19-refcv8-navsim/code",
    "C:/lgs/Lfb47a58376/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-19-refcv8-navsim/code",
    "D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-19-refcv8-navsim/code",
)
ELIGIBLE = {"label": "refcv8 final read", "step": 48852, "weights": "ema"}


def _bars():
    for d in _BARS_CANDIDATES:
        if d and os.path.exists(os.path.join(d, "navsim_bars8.py")):
            if d not in sys.path:
                sys.path.insert(0, d)
            import navsim_bars8 as B
            return B
    pytest.skip("navsim_bars8.py not found (set PARSE8_BARS8_CODE) -- a skipped e2e is stated, it is not a pass")


def test_e2e_navhard_synthetic_n2_passes(tmp_path):
    B = _bars()
    out = P.build_summary(**build_navhard(tmp_path))
    bar = B.navhard_bar(out, ELIGIBLE, B.DEFAULT_CFG)
    # margin .36 - max(.04, .25, .09) = .11 ; paired lo = .11 > 0 ; seed floor .0119 -> .11 > 2 x .0119
    assert bar["verdict"] == "PASS"
    assert bar["values"]["R8_LEGAL"] == approx(0.36) and bar["margin"] == approx(0.11) and bar["seed_floor"] == approx(0.0119)


def test_e2e_navhard_unavailable_arm_propagates_as_unavailable(tmp_path):
    B = _bars()
    out = P.build_summary(**build_navhard(tmp_path, omit=("ECHO_ha0_ext",)))
    assert B.navhard_bar(out, ELIGIBLE, B.DEFAULT_CFG)["verdict"].startswith("UNAVAILABLE")


def test_e2e_warmup_synthetic_n3_passes_and_the_blind_panel_is_ok(tmp_path):
    B = _bars()
    out = P.build_summary(**build_warmup(tmp_path))
    bar = B.warmup_bar(out, ELIGIBLE, B.DEFAULT_CFG)
    # margin .425 - max(.3, .4, .35) = .025 ; seed floor .01 -> .025 > .02 PASS ; BLIND .1 - .4 < 0 -> FAILED -> panel OK
    assert bar["verdict"] == "PASS" and bar["blind_control_verdict"] == "FAILED" and bar["panel"].startswith("OK")
    assert bar["margin"] == approx(0.025) and bar["seed_floor"] == approx(0.01)


def test_e2e_warmup_without_the_blind_arm_cannot_pass(tmp_path):
    B = _bars()
    out = P.build_summary(**build_warmup(tmp_path, omit=("R8_BLIND",)))
    bar = B.warmup_bar(out, ELIGIBLE, B.DEFAULT_CFG)
    assert bar["verdict"].startswith("UNAVAILABLE") and bar["panel"].startswith("UNAVAILABLE")


def test_e2e_warmup_missing_seed_arm_is_not_proven_not_a_crash(tmp_path):
    B = _bars()
    out = P.build_summary(**build_warmup(tmp_path, omit=("R8_LEGAL_s1",)))
    assert B.warmup_bar(out, ELIGIBLE, B.DEFAULT_CFG)["verdict"].startswith("NOT PROVEN")


def test_e2e_a_blind_arm_that_passes_voids_the_panel(tmp_path):
    B = _bars()
    kw = build_warmup(tmp_path)
    p = os.path.join(kw["scores"], "score_R8_BLIND.csv")                                 # make the deliberate regression BEAT the floors
    df = pd.read_csv(p)
    for t in S2_TOK:
        df.loc[df.token == t, "score"] = 0.9
    df.to_csv(p, index=False)
    assert B.warmup_bar(P.build_summary(**kw), ELIGIBLE, B.DEFAULT_CFG)["verdict"].startswith("VOID")


# ------------------------------------------------------------------------------------------------------------------------------
# 4. CONTROL: parse8 over refcv7's banked step-30000 NavSim inputs, R7 arm names read as the mapped R8 names
# ------------------------------------------------------------------------------------------------------------------------------
_R7RAW = "FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/navsim/raw"
_R7_ROOTS = tuple(f"{r}/{_R7RAW}" for r in ("D:/Projects/TanitAD", "C:/lgs/Lfb47a58376"))
_NAVHARD_INPUTS = "C:/Users/Admin/navsim-crun/exp/tanitad_bench/exports/navhard_two_stage/navsim_agent_inputs.json"
_WARMUP_INPUTS = tuple(f"{r}/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-navsim-refcv4b-bridge/raw/navsim_agent_inputs.json"
                       for r in ("D:/Projects/TanitAD", "C:/lgs/Lfb47a58376"))


def _control_paths(sp: str):
    """-> dict for ``build_summary`` + the banked summary path, or skip."""
    split = f"{sp}_two_stage"
    for root in _R7_ROOTS:
        base = f"{root}/milestones/step30000"
        if os.path.exists(f"{base}/summary_{sp}.json") and os.path.isdir(f"{base}/scores_{sp}") and \
                os.path.isdir(f"{root}/floors/{split}"):
            inputs = _NAVHARD_INPUTS if sp == "navhard" else next((p for p in _WARMUP_INPUTS if os.path.exists(p)), "")
            if inputs and os.path.exists(inputs):
                kw = dict(split=split, scores=f"{base}/scores_{sp}", bridge=f"{base}/bridge_{sp}", inputs=inputs,
                          label="CONTROL", floors=f"{root}/floors/{split}", alias=dict(P.CONTROL_R7_ALIAS),
                          csv_suffix="__navhard_two_stage" if sp == "navhard" else "")
                return kw, f"{base}/summary_{sp}.json"
    pytest.skip(f"refcv7 banked step-30000 {sp} inputs not on this disk -- a skipped control is stated, it is not a pass")


# banked numbers: read ONCE from refcv7's summaries (parse7's output) -- the CONTROL's literal targets
#   D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/navsim/raw/milestones/step30000/summary_warmup.json
#   (R7 names on the left of each comment; the parser is asked for the mapped R8 name)
WARMUP_U = {"R8_LEGAL": 0.5223666823715276,         # R7_A1
            "R8_LEGAL_s1": 0.5292943037372295,      # R7_A1_s1
            "R8_BLIND": 0.21932603572915813,        # R7_BLIND
            "R8_NT": 0.521096210181534,             # R7_A1NT
            "R8_NAVOFF": 0.5189898596005245,        # R7_NAVOFF
            "PRIOR_ha0p": 0.4585525017515276, "CV_official": 0.39713167136274696,
            "STOP_zero": 0.5212469877807624, "ECHO_ha0_ext": 0.4287471809626652}
WARMUP_SEED_FLOOR = 0.006927621365701908          # |R7_A1 - R7_A1_s1| ; pair delta -0.006927621365701908, W/T/L 48/94/62
#   .../step30000/summary_navhard.json
NAVHARD_OFFICIAL = {"R8_LEGAL": 0.2269259218512275,  # R7_A1
                    "R8_LEGAL_s1": 0.2232962025910765, "PRIOR_ha0p": 0.1300400827024679, "CV_official": 0.11481608441648,
                    "STOP_zero": 0.2985321254269824, "ECHO_ha0_ext": 0.1429384814411483}
NAVHARD_U = {"R8_LEGAL": 0.42939530720905833, "R8_LEGAL_s1": 0.42596966263582864, "STOP_zero": 0.4828269144777932}
NAVHARD_SEED_FLOOR_U = 0.0034256445732296914
NAVHARD_SEED_FLOOR_OFFICIAL_DELTA = 0.0036297192601510153        # official two-stage EPDMS(R7_A1) - (R7_A1_s1)


def test_control_warmup_reproduces_the_banked_refcv7_numbers():
    kw, _ = _control_paths("warmup")
    out = P.build_summary(**kw)
    for arm, v in WARMUP_U.items():
        assert out["arms"][arm]["S2_EPDMS_u"]["value"] == approx(v), arm
        assert out["arms"][arm]["S2_EPDMS_u"]["n_groups"] == 16
    assert out["seed_floor_S2_EPDMS_u"] == approx(WARMUP_SEED_FLOOR)
    p = out["pairs"]
    assert p["R8_LEGAL__minus__STOP_zero"]["S2_EPDMS_u_delta"] == approx(0.001119694590765219)      # W/T/L 113/32/59
    assert (p["R8_LEGAL__minus__STOP_zero"]["wins"], p["R8_LEGAL__minus__STOP_zero"]["ties"],
            p["R8_LEGAL__minus__STOP_zero"]["losses"]) == (113, 32, 59)
    assert p["R8_LEGAL__minus__R8_BLIND"]["S2_EPDMS_u_delta"] == approx(0.3030406466423695)
    assert (p["R8_LEGAL__minus__R8_BLIND"]["wins"], p["R8_LEGAL__minus__R8_BLIND"]["losses"]) == (116, 18)
    assert p["R8_LEGAL__minus__R8_LEGAL_s1"]["S2_EPDMS_u_delta"] == approx(-WARMUP_SEED_FLOOR)
    assert out["arms"]["R8_LEGAL"]["stop_fraction"]["frac_endpoint_lt_1m"] == approx(0.029411764705882353)
    assert out["arms"]["R8_BLIND"]["stop_fraction"]["frac_endpoint_lt_1m"] == approx(0.09313725490196079)
    assert out["arms"]["R8_BLIND"]["official_two_stage_EPDMS"] == 0.0                              # BLIND's stage 1 is its own
    assert out["arms"]["STOP_zero"]["official_two_stage_EPDMS"] == approx(0.3009023137456225)
    assert out["arms"]["R8_LEGAL"]["official_two_stage_EPDMS"]["status"] == "UNAVAILABLE"           # CV stand-in stage 1: stated, not headlined
    assert out["checkpoint_step"] == 30000


def test_control_navhard_reproduces_the_banked_refcv7_numbers():
    kw, _ = _control_paths("navhard")
    out = P.build_summary(**kw)
    for arm, v in NAVHARD_OFFICIAL.items():
        assert out["arms"][arm]["official_two_stage_EPDMS"] == approx(v), arm
    for arm, v in NAVHARD_U.items():
        assert out["arms"][arm]["S2_EPDMS_u"]["value"] == approx(v), arm
    assert out["seed_floor_S2_EPDMS_u"] == approx(NAVHARD_SEED_FLOOR_U)
    assert out["pairs"]["R8_LEGAL__minus__R8_LEGAL_s1"]["official_two_stage_EPDMS_delta"] == approx(NAVHARD_SEED_FLOOR_OFFICIAL_DELTA)
    p = out["pairs"]["R8_LEGAL__minus__STOP_zero"]
    assert p["S2_EPDMS_u_delta"] == approx(-0.05343160726873486) and (p["wins"], p["ties"], p["losses"]) == (2533, 1077, 1852)
    # the log-cluster bootstrap (B = 2000, seed 0): the interval strings are 4-dp rounded, so these are EXACT
    iv = out["intervals"]["R8_LEGAL"]
    assert (iv["status"], iv["point"], iv["lo"], iv["hi"], iv["n_clusters"], iv["n_units"]) == ("OK", 0.2269, 0.1922, 0.261, 76, 225)
    iv = out["intervals"]["STOP_zero"]
    assert (iv["point"], iv["lo"], iv["hi"]) == (0.2985, 0.2737, 0.3253)
    pi = out["paired_intervals"]["R8_LEGAL__minus__STOP_zero"]
    assert (pi["status"], pi["delta"], pi["lo"], pi["hi"], pi["separated"]) == ("OK", -0.0716, -0.1112, -0.0346, True)
    pi = out["paired_intervals"]["R8_LEGAL__minus__R8_LEGAL_s1"]
    assert (pi["delta"], pi["lo"], pi["hi"], pi["separated"]) == (0.0036, -0.0107, 0.018, False)


def _r7_to_r8(k):
    m = {"R7_A1": "R8_LEGAL", "R7_A1_s1": "R8_LEGAL_s1", "R7_BLIND": "R8_BLIND", "R7_A1NT": "R8_NT", "R7_NAVOFF": "R8_NAVOFF"}
    return "__minus__".join(m.get(p, p) for p in k.split("__minus__"))


def _diff(a, b, path=""):
    """Every difference between two JSON trees; floats compared at 1e-12 (the bootstrap is seeded, so equality is expected)."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            out += [(path + "/" + k, "only in one")] if (k not in a or k not in b) else _diff(a[k], b[k], path + "/" + k)
        return out
    if isinstance(a, list) and isinstance(b, list):
        return [(path, "len")] if len(a) != len(b) else [d for i, (x, y) in enumerate(zip(a, b)) for d in _diff(x, y, f"{path}[{i}]")]
    if isinstance(a, float) and isinstance(b, float):
        return [] if (a == b or (math.isnan(a) and math.isnan(b)) or abs(a - b) <= 1e-12) else [(path, (a, b))]
    return [] if a == b else [(path, (a, b))]


@pytest.mark.parametrize("sp", ["warmup", "navhard"])
def test_control_whole_summary_equals_the_banked_one_apart_from_labels(sp):
    """Everything the bars layer could read -- arms, pairs, intervals, paired intervals, estimator -- equals parse7's banked
    output after the R7->R8 rename.  Only free-text labels (``what``, ``_label``, ``model_as_trained``) and parse8's additive
    provenance are excluded; the R7-recipe-only arms (VMAXOFF, FILTOFF, CEILDECL_d) have no R8 counterpart."""
    kw, banked_path = _control_paths(sp)
    new = P.build_summary(**kw)
    banked = json.load(open(banked_path, encoding="utf-8"))
    drop = {"R7_VMAXOFF", "R7_FILTOFF", "R7_CEILDECL_d"}
    b = {}
    for k, v in banked.items():
        if k in ("arms", "pairs", "intervals", "paired_intervals"):
            b[k] = {_r7_to_r8(kk): vv for kk, vv in v.items() if not any(p in drop for p in kk.split("__minus__"))}
        elif k not in ("_label", "model_as_trained", "arms_refused"):
            b[k] = v
    n = {k: v for k, v in new.items() if k in b}
    for blk in ("pairs", "paired_intervals"):
        if blk in n:        # parse8 ADDS the seed-1 mirror rows (AMENDMENT 3); parse7's banked output has no such key
            n[blk] = {k: v for k, v in n[blk].items() if k in b[blk]}
    assert all(f"R8_LEGAL_s1__minus__{y}" in new["pairs"] for y in ("STOP_zero", "CV_official", "ECHO_ha0_ext"))
    for blk in ("pairs",):
        for rec in list(n.get(blk, {}).values()) + list(b.get(blk, {}).values()):
            rec.pop("what", None)
    if "estimator" in n and "module" in n["estimator"]:
        n["estimator"] = {k: v for k, v in n["estimator"].items() if k not in ("module", "module_sha256")}
        b["estimator"] = {k: v for k, v in b["estimator"].items() if k not in ("module", "module_sha256")}
    assert _diff(n, b) == []
    extra = set(new) - set(b)
    assert extra <= {"_label", "model_as_trained", "arms_refused", "arms_not_run", "_parse8", "seed_floor_S2_EPDMS_u_status"}, extra


def test_control_cli_writes_the_same_summary_through_main(tmp_path):
    kw, _ = _control_paths("warmup")
    outp = tmp_path / "summary_warmup.json"
    r = subprocess.run([sys.executable, "-X", "utf8", os.path.join(PKG, "code", "parse8.py"), "--split", "warmup_two_stage",
                        "--scores", kw["scores"], "--floors", kw["floors"], "--bridge", kw["bridge"], "--inputs", kw["inputs"],
                        "--label", "CONTROL", "--control-r7-names", "--out", str(outp)],
                       capture_output=True, text=True, env={k: v for k, v in os.environ.items() if k != "PARSE8_MUTATION"})
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
    got = json.loads(outp.read_bytes().decode("utf-8"))
    assert got["arms"]["R8_BLIND"]["S2_EPDMS_u"]["value"] == approx(0.21932603572915813) and b"\r\n" not in outp.read_bytes()


# banked (parse7) R7_A1 paired records vs the floors -- the targets of the IDENTITY control
#   .../step30000/summary_navhard.json : paired_intervals["R7_A1__minus__<floor>"] and pairs[...]
IDENTITY = {  # floor -> (delta, lo, hi, drive_lo, drive_hi, S2_EPDMS_u_delta, official_two_stage_EPDMS_delta)
    "CV_official": (0.1121, 0.0716, 0.1546, 0.055, 0.155, 0.09048158072024265, 0.1121098374347475),
    "STOP_zero": (-0.0716, -0.1112, -0.0346, -0.1247, -0.035, -0.05343160726873486, -0.07160620357575487),
    "ECHO_ha0_ext": (0.084, 0.0481, 0.12, 0.0297, 0.1259, 0.0881150933788416, 0.08398744041007922)}


def test_control_identity_seed1_mirror_fed_with_seed0_rows_equals_the_banked_seed0_records():
    """IDENTITY control on real data: read the refcv7 R7_A1 (seed-0) files as BOTH R8_LEGAL and R8_LEGAL_s1.  The new
    ``R8_LEGAL_s1__minus__<floor>`` record must equal the banked ``R7_A1__minus__<floor>`` record numerically (delta, lo, hi, drive
    lo/hi) -- the mirror runs the same function on the same kind of rows.  NB this control cannot see a mirror that silently reuses
    the seed-0 rows (here the two are identical by construction); that defect is caught by the synthetic s0 != s1 tests."""
    kw, _ = _control_paths("navhard")
    kw["alias"] = {"R8_LEGAL": "R7_A1", "R8_LEGAL_s1": "R7_A1"}
    out = P.build_summary(**kw)
    for y, (d, lo, hi, dlo, dhi, ud, od) in IDENTITY.items():
        pi = out["paired_intervals"][f"R8_LEGAL_s1__minus__{y}"]
        dr = pi["sensitivity_nuplan_drive"]
        assert (pi["status"], pi["delta"], pi["lo"], pi["hi"]) == ("OK", d, lo, hi), y
        assert (dr["status"], dr["lo"], dr["hi"], dr["n_clusters"]) == ("OK", dlo, dhi, 34), y
        assert pi["separated"] is True and pi["separated_log_name"] is True and dr["separated"] is True, y
        pr = out["pairs"][f"R8_LEGAL_s1__minus__{y}"]
        assert pr["S2_EPDMS_u_delta"] == approx(ud) and pr["official_two_stage_EPDMS_delta"] == approx(od), y
        assert pi == out["paired_intervals"][f"R8_LEGAL__minus__{y}"], y            # s1 mirror == s0 row, field for field


def test_e2e_control_bars_accept_the_summaries_and_give_a_verdict():
    B = _bars()
    nh = P.build_summary(**_control_paths("navhard")[0])
    wu = P.build_summary(**_control_paths("warmup")[0])
    n2 = B.navhard_bar(nh, ELIGIBLE, B.DEFAULT_CFG)
    # refcv7-step-30000 numbers under R8 names: LEGAL .2269 < STOP .2985 -> margin < 0 -> FAILED (an honest FAILED, not UNAVAILABLE)
    assert n2["verdict"] == "FAILED" and n2["margin"] == approx(0.2269259218512275 - 0.2985321254269824)
    assert n2["paired_vs_STOP"]["status"] == "OK" and n2["seed_floor"] == approx(NAVHARD_SEED_FLOOR_OFFICIAL_DELTA)
    n3 = B.warmup_bar(wu, ELIGIBLE, B.DEFAULT_CFG)
    # margin .5223667 - max(.3971, .5212, .4287) = .0011197 ; floor .0069276 -> inside 2x -> NOT PROVEN ; BLIND .2193 fails -> panel OK
    assert n3["verdict"].startswith("NOT PROVEN") and n3["panel"].startswith("OK") and n3["blind_control_verdict"] == "FAILED"
    assert n3["margin"] == approx(0.001119694590765219) and n3["seed_floor"] == approx(WARMUP_SEED_FLOOR)
