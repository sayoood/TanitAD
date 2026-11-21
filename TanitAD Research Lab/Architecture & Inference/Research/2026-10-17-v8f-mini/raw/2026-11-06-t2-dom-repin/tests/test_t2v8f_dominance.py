"""Tests of `code/read_t2_dominance.py` (the read of PREREG_T2_DOMINANCE) on SYNTHETIC arrays and synthetic pulls only: no stage-2/3/4 row is read. CPU only.
Every outcome of card section 5 is produced; every gate (G1..G4) has a RED arm; G4 also runs the real reproduction of the banked T1-kin Read A block."""
from __future__ import annotations

import json
import os
import shutil
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code"))
import numpy as np  # noqa: E402
import pytest  # noqa: E402

import read_t2_dominance as RD  # noqa: E402

TAC = ("TAC-LAT", "TAC-LON")
STRAT = ("S-POS@30", "S-POS@50", "S-POS@80")


def synth(effects=None, n=121, seed=0):
    """Reader-format arrays: C0C row = P10 row + effect (constant, so the paired bootstrap interval is a point: separated iff effect != 0). `effects` {label: delta}, default 0."""
    rr = RD.init()
    GD = rr.GD
    rng = np.random.default_rng(seed)
    eid = np.array([i for i in range(62) for _ in range(2)])[:n]
    z = {"labels": np.array(GD.LABELS), "eid": eid, "v0": np.full(n, 10.0), "g": np.stack([np.stack([np.arange(1, 31) * 2.0, np.zeros(30)], 1)] * n)}
    for i, lab in enumerate(GD.LABELS):
        b = rng.uniform(1.0, 2.0, n)
        z[f"a_{i}"], z[f"b_{i}"] = b + float((effects or {}).get(lab, 0.0)), b
    return z


def dom_effects(skip=()):
    GD = RD.init().GD
    return {lab: (0.5 if lab in TAC else -0.5) for lab in GD.LABELS if lab not in skip}


def blocks_of(e0, e1):
    return RD.read_blocks({0: synth(e0, seed=1), 1: synth(e1, seed=2)}, n_boot=200)


def test_outcome_dominant():
    d = RD.decide(True, blocks_of(dom_effects(), dom_effects()))
    assert d["outcome"] == "DOMINANT-T2", d
    assert all(p["DOMINANT"] and p["D1"] and p["D2"] and p["D3"] for p in d["per_seed"].values())


def test_outcome_not_dominant_d2_one_seed():
    e1 = dom_effects()
    e1["speed MAE"] = 0.5                                              # separated-worse at seed 1 only
    d = RD.decide(True, blocks_of(dom_effects(), e1))
    assert d["outcome"] == "NOT-DOMINANT" and d["per_seed"]["s0"]["DOMINANT"] and not d["per_seed"]["s1"]["D2"], d
    assert any("D2 (separated-worse: speed MAE)" in t for t in d["failing_terms"]["s1"]) and d["failing_terms"]["s0"] == []


def test_outcome_not_dominant_d1_only():
    e = dom_effects()
    e["ADE6"] = -0.1                                                   # separated-better but above -0.2415
    d = RD.decide(True, blocks_of(e, e))
    assert d["outcome"] == "NOT-DOMINANT" and not d["per_seed"]["s0"]["D1"] and d["per_seed"]["s0"]["D2"] and d["per_seed"]["s0"]["D3"], d
    assert d["failing_terms"]["s0"][0].startswith("D1 (ADE6 not separated-better with point <= -0.2415 m")


def test_outcome_not_dominant_d3_and_three_family_form_beside():
    e = dom_effects(skip=STRAT)                                        # STRATEGIC has no separated-better row
    b = blocks_of(e, e)
    d = RD.decide(True, b)
    assert d["outcome"] == "NOT-DOMINANT" and "D3 (STRATEGIC: no separated-better row)" in d["failing_terms"]["s0"]
    assert d["per_seed"]["s0"]["DOMINANT_three_families_strategic_paused"] is True                      # the three-family form is reported beside, and it holds


def test_outcome_underpowered_is_reachable():
    d = RD.decide(True, blocks_of({}, {}))
    assert d["outcome"] == "UNDERPOWERED" and all(p["n_rows_separated"] == 0 for p in d["per_seed"].values())
    one = dom_effects(skip=[l for l in RD.init().GD.LABELS if l != "ADE6"])
    one["ADE6"] = -0.5                                                 # one separated row at one seed: not UNDERPOWERED
    assert RD.decide(True, blocks_of({}, one))["outcome"] == "NOT-DOMINANT"


def test_outcome_gate_fail_reads_no_row():
    d = RD.decide(False, None)
    assert d["outcome"] == "GATE-FAIL" and d["per_seed"] is None
    with pytest.raises(ValueError):
        RD.decide(True, {0: blocks_of({}, {})[0]})


def test_family_table_and_seed_gap():
    b = blocks_of(dom_effects(), dom_effects())
    t = RD.family_table(b[0]["rows"])
    assert set(t) >= set(RD.FAMILIES) and all(r["verdict"] == "separated-better" for r in t["STRATEGIC"])
    g = RD.seed_gap_table(b)
    assert g["ADE6"]["separated_better_both_seeds"] and g["ADE6"]["gap"] < 1e-9 and g["ADE6"]["margin_lt_gap"] is False


# ------------------------------------------------------------------------------------------------ G1, G2
def _g1(tmp_path, **kw):
    d = {"gate": "G1_C0C_ID", "pass": True, "red_arms": {a: "RED" for a in RD.RED_ARMS}, "off_off_reproduces_C0": True}
    d.update(kw)
    p = tmp_path / "g1.json"
    p.write_text(json.dumps(d))
    return str(p)


def test_g1_positive_and_red_arms(tmp_path):
    assert RD.g1_check(_g1(tmp_path))["PASS"]
    assert not RD.g1_check(_g1(tmp_path, **{"pass": False}))["PASS"]
    assert not RD.g1_check(_g1(tmp_path, red_arms={"smooth_off": "RED", "feasible_off": "GREEN", "prior_zeros": "RED"}))["PASS"]
    assert not RD.g1_check(_g1(tmp_path, off_off_reproduces_C0=False))["PASS"]
    assert not RD.g1_check(_g1(tmp_path, gate="G1_OTHER"))["PASS"]
    assert not RD.g1_check(str(tmp_path / "nope.json"))["PASS"]


def test_g2_positive_and_red_arm(tmp_path):
    def res(name, gate):
        p = tmp_path / name
        p.write_text(json.dumps({"gates": {"preflight": "PASS", "gate": gate}}))
        return str(p)
    assert RD.g2_check({2: res("a.json", "PASS"), 3: res("b.json", "PASS"), 4: res("c.json", "PASS")})["PASS"]
    assert not RD.g2_check({2: res("a.json", "PASS"), 3: res("b2.json", "FAIL"), 4: res("c.json", "PASS")})["PASS"]
    assert not RD.g2_check({2: res("a.json", "PASS"), 3: str(tmp_path / "missing.json")})["PASS"]


# ------------------------------------------------------------------------------------------------ G3 on a synthetic pull
SIDS = {0: "aaaaaaaa-1", 1: "bbbbbbbb-2", 2: "cccccccc-3"}
EXP = {"scenes_launch": 2, "scenes_guard": 1, "starts_launch": 3, "starts_guard": 2}
STARTS = {0: (20, 30), 1: (25,), 2: (40, 50)}


def make_pulls(root, c0c_shift=0, drop_done=None, overall4="COMPLETE", skip_p10_scene=None):
    def cell(r, idx, arm, shift=0):
        d = f"{r}/rollouts/{SIDS[idx][:8]}/objects/T2/{arm}"
        os.makedirs(d, exist_ok=True)
        recs = [{"scene": idx, "f0": f + shift} for f in STARTS[idx]]
        open(f"{d}/records.jsonl", "w").write("".join(json.dumps(x) + "\n" for x in recs))
        if d.replace("\\", "/") != drop_done:
            open(f"{d}/DONE", "w").write(f"rc=0 records={len(recs)}/{len(recs)}\n")
    for st, idxs in ((2, (0, 1)), (3, (2,))):
        for i in idxs:
            if i != skip_p10_scene:
                cell(f"{root}/p{st}", i, "P10")
    for i in (0, 1, 2):
        for arm in ("C0C_s0", "C0C_s1"):
            cell(f"{root}/p4", i, arm, c0c_shift if (arm == "C0C_s0" and i == 0) else 0)
    out = {}
    for st, ov in ((2, "COMPLETE"), (3, "COMPLETE"), (4, overall4)):
        os.makedirs(f"{root}/p{st}", exist_ok=True)
        open(f"{root}/p{st}/STAGE{st}_RESULT.json", "w").write(json.dumps({"overall": ov, "gates": {"preflight": "PASS", "gate": "PASS"}}))
        out[st] = f"{root}/p{st}/rollouts"
    return out


def g3(tmp_path, name, **kw):
    r = str(tmp_path / name)
    roots = make_pulls(r, **kw)
    return RD.g3_check(roots, {s: f"{r}/p{s}/STAGE{s}_RESULT.json" for s in roots}, SIDS, EXP)


def test_g3_positive_and_red_arms(tmp_path):
    gate, recs = g3(tmp_path, "ok")
    assert gate["PASS"] and recs and gate["pairing"] == {"s0": "PAIRED", "s1": "PAIRED"}, gate
    bad, none = g3(tmp_path, "shift", c0c_shift=1)
    assert not bad["PASS"] and none is None and any("G3_PAIRING seed 0" in w for w in bad["why"]), bad
    p4 = str(tmp_path / "nodone" / "p4" / "rollouts" / "aaaaaaaa" / "objects" / "T2" / "C0C_s1")
    bad, _ = g3(tmp_path, "nodone", drop_done=p4.replace("\\", "/"))
    assert any("G3_CELL_NOT_DONE" in w and "no DONE marker" in w for w in bad["why"]), bad
    bad, _ = g3(tmp_path, "incomplete", overall4="INCOMPLETE")
    assert any("G3_OVERALL_NOT_COMPLETE stage 4" in w for w in bad["why"]), bad
    bad, _ = g3(tmp_path, "nop10", skip_p10_scene=2)
    assert any("G3_CELLS stage 3 P10" in w for w in bad["why"]), bad


# ------------------------------------------------------------------------------------------------ G4: the reader reproduces the banked Read A block
def test_g4_reproduces_banked_block_and_red_arms(tmp_path):
    ok = RD.g4_check()
    assert ok["PASS"], ok
    assert not RD.g4_check(pins={**RD.BANK_PINS, RD.BANK_JSON: "0" * 64})["PASS"]                       # a pin that does not hold
    def bump(s, ar):
        ar["a_0"] = ar["a_0"] + 1e-3 if s == 0 else ar["a_0"]
    red = RD.g4_check(mutate=bump)
    assert not red["PASS"] and any("G4_ROWS_DIFFER seed 0" in w for w in red["why"]), red               # one perturbed array breaks bit identity
    fake = tmp_path / "recover_read.py"
    shutil.copyfile(RD.RR, fake)
    fake.write_text(fake.read_text(encoding="utf-8") + "\n# one byte more\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        RD.init(rr_path=str(fake))                                                                      # the reader's sha256 pin refuses an edited reader


def test_card_is_frozen_and_untouched(tmp_path):
    """M, 2026-11-06: the card gained its 17:42 log entry, which moved the WHOLE-FILE sha256 (8f2bbbd1...) off the frozen one; the check is now on the frozen prefix (section 8 is append-only)."""
    assert RD.card_frozen_ok()
    b = open(RD.CARD, "rb").read()
    ok, bad = tmp_path / "appended.md", tmp_path / "edited.md"
    ok.write_bytes(b + b"* 2026-11-07: a later dated entry.\n")
    assert RD.card_frozen_ok(str(ok))                                                    # an appended entry is allowed
    bad.write_bytes(b[:100] + b"X" + b[101:])
    assert not RD.card_frozen_ok(str(bad))                                               # one edited byte above the log is a refused control
    assert not RD.card_frozen_ok(str(tmp_path / "missing.md"))
