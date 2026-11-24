"""Tests for scorerft_read.py (PREREG_SCORER_FT.md sections 5-10). Run the WHOLE file (module fixtures; never -k).

Positive arms: the clauses P1 / G1 / P2 / P3, the outcome order O1, O2, O5, O6, O3, O4, the controls.
Deliberate-regression (RED) arms, each must turn the read INVALID / the clause false:
  RED-SWAP    T and C swapped -> P1 FAILS, no Stage-2 trigger
  RED-DROP    one token dropped from one arm -> CTRL-SET FAILS and per_token refuses
  RED-PERTURB one per-token PDMS perturbed by 1e-6 in a table -> CTRL-T FAILS
  RED-FLOOR   a Cs / Ts pair whose F makes d <= 2F -> P3 FAILS, O2 instead of O1
  RED-ARGV    an argv set with ONE extra flag -> CTRL-ARGV FAILS
"""
import argparse
import csv
import json
import math
import os

import numpy as np
import pytest

import scorerft_read as S
import p5c_read as PR
import field_read as FR

METRICS = PR.METRICS


# ------------------------------------------------------------------------------------------------------ synthetic data
def mkP(n=400, L=20, offsets=None, seed=0, sub_off=None):
    """P as p5c_read.per_token builds it: offsets {arm: pdms offset}; every sub-score identical across arms unless sub_off {arm: {metric: off}}."""
    rng = np.random.default_rng(seed)
    base = {m: rng.uniform(40, 99, n) for m in METRICS}
    noise = {a: rng.normal(0, 0.05, n) for a in offsets}
    P = {"tokens": [f"{i:016x}" for i in range(n)], "logs": [f"log{i % L}" for i in range(n)]}
    for a, off in offsets.items():
        P[a] = {s: {m: base[m] + (off + noise[a] if m == "pdms" else (sub_off or {}).get(a, {}).get(m, 0.0)) for m in METRICS} for s in S.SCORERS}
    return P


def con_of(P, x, y, n_boot=300):
    bt = PR.Boot(P["logs"], n_boot)
    return bt, PR.contrast(bt, P, x, y)


def fake_con(pdms=(0.5, 0.2, 0.8), adverse=()):
    z = lambda pt, lo, hi: {"point": pt, "lo": lo, "hi": hi}    # noqa: E731
    out = {m: {"tracked": z(0.0, -0.1, 0.1), "transparent": z(0.0, -0.1, 0.1), "TD": z(0.0, -0.1, 0.1)} for m in METRICS}
    out["pdms"]["tracked"] = z(*pdms)
    for m in adverse:
        out[m]["tracked"] = z(-1.0, -1.5, -0.5)
    return out


def fake_mech(m1=False, m3=False, m4=False):
    return {"m4_damaged": m4, "mechanism_moved": bool(m1 or m3)}


# ------------------------------------------------------------------------------------------------------ outcomes, in the card's order
def test_stage1_p1_and_g1_trigger_stage2_never_adopt():
    v = S.decide(fake_con(), fake_mech(), True, None)
    assert v["verdict"] == "STAGE-2-TRIGGERED" and v["outcome"] == "PENDING"
    assert v["clauses"]["P2"] == "PENDING (Stage 2)" and v["clauses"]["P3"] == "PENDING (Stage 2)"
    assert "ADOPT" not in v["verdict"] and "Never ADOPT" in v["wording"] and "training variance was NOT measured" in v["wording"]


def test_outcome_order_O5_O6_O3_O4_stage1():
    o5 = S.decide(fake_con(adverse=("ep",)), fake_mech(), True)
    assert o5["outcome"] == "O5" and "EP adverse => `gate` only" in o5["next_arm"] and "training variance" in o5["wording"]
    o5b = S.decide(fake_con(adverse=("dac",)), fake_mech(), True)
    assert "`ep` only" in o5b["next_arm"] and "gate` only" not in o5b["next_arm"]
    nop1 = fake_con(pdms=(0.1, -0.3, 0.5))
    assert S.decide(nop1, fake_mech(m4=True), True)["outcome"] == "O6"
    assert S.decide(nop1, fake_mech(m4=True, m1=True), True)["outcome"] == "O6"          # O6 precedes O3
    assert S.decide(nop1, fake_mech(m3=True), True)["outcome"] == "O3"
    assert S.decide(nop1, fake_mech(m1=True), True)["outcome"] == "O3"
    o4 = S.decide(nop1, fake_mech(), True)
    assert o4["outcome"] == "O4" and "training variance" in o4["wording"] and "TEACHER-N7" in o4["next_arm"]


def test_every_outcome_carries_its_committed_next_arm():
    assert set(S.NEXT_ARM) == {"O1", "O2", "O3", "O4", "O5", "O6"}
    for o, t in S.NEXT_ARM.items():
        assert t and S.OUTCOME_VERDICT[o]


def test_p1_ci_lo_exactly_zero_is_not_p1():
    assert S.decide(fake_con(pdms=(0.5, 0.0, 1.0)), fake_mech(), True)["clauses"]["P1_lo_gt_0"] is False


def test_invalid_when_a_control_fails_or_is_missing():
    for ok in (False, None):
        v = S.decide(fake_con(), fake_mech(), ok)
        assert v["verdict"] == "INVALID" and v["outcome"] is None


def test_not_p1_without_tables_is_incomplete_not_a_guess():
    v = S.decide(fake_con(pdms=(0.1, -0.3, 0.5)), None, True)
    assert v["verdict"] == "INCOMPLETE"


# ------------------------------------------------------------------------------------------------------ Stage 2: P2 and P3 (RED-FLOOR)
def stage2_P(cs_off, ts_off=None, t_off=1.0):
    ts_off = cs_off + 1.0 if ts_off is None else ts_off
    return mkP(offsets={"C": 0.0, "T": t_off, "Cs": cs_off, "Ts": ts_off})


def test_p2_p3_pass_gives_O1_adopt_with_caveat():
    P = stage2_P(0.1)
    bt, c0 = con_of(P, "T", "C")
    rep = S.pair_rep(bt, P, "C", "T", "Cs", "Ts")
    v = S.decide(c0, fake_mech(), True, rep)
    assert v["outcome"] == "O1" and v["verdict"] == "O1" and v["clauses"]["P2"] == "PASS" and v["clauses"]["P3"] == "PASS"
    assert "GUARD-FAIL" in v["wording"]
    assert rep["s_p"]["value"] >= 0 and "decides nothing" in rep["s_p"]["definition"]


def test_RED_FLOOR_a_seed_pair_that_triggers_p3_gives_O2():
    P = stage2_P(0.6)                      # F = |Cs - C| = 0.6 -> 2F = 1.2 > d = 1.0
    bt, c0 = con_of(P, "T", "C")
    rep = S.pair_rep(bt, P, "C", "T", "Cs", "Ts")
    assert abs(rep["F"] - 0.6) < 0.01
    v = S.decide(c0, fake_mech(), True, rep)
    assert v["clauses"]["P2"] == "PASS" and v["clauses"]["P3"] == "FAIL" and v["outcome"] == "O2"
    assert v["clauses"]["P3_2F"] > v["clauses"]["P3_d"]


def test_p2_fails_when_replicate_ci_lo_not_positive_or_g1_adverse_on_it():
    P = mkP(offsets={"C": 0.0, "T": 1.0, "Cs": 0.0, "Ts": 0.0})
    bt, c0 = con_of(P, "T", "C")
    assert S.decide(c0, fake_mech(), True, S.pair_rep(bt, P, "C", "T", "Cs", "Ts"))["outcome"] == "O2"
    P2 = mkP(offsets={"C": 0.0, "T": 1.0, "Cs": 0.0, "Ts": 1.0}, sub_off={"Ts": {"ep": -2.0}})
    bt, c0 = con_of(P2, "T", "C")
    rep = S.pair_rep(bt, P2, "C", "T", "Cs", "Ts")
    v = S.decide(c0, fake_mech(), True, rep)
    assert v["clauses"]["P2"] == "FAIL" and v["clauses"]["P2_G1_adverse"] == ["ep"] and v["outcome"] == "O2"


def test_stage2_does_not_rescue_a_failed_stage1():
    P = stage2_P(0.1)
    bt, c0 = con_of(P, "C", "T")           # swapped: P1 fails
    rep = S.pair_rep(bt, P, "T", "C", "Cs", "Ts")
    assert S.decide(c0, fake_mech(), True, rep)["outcome"] == "O4"


# ------------------------------------------------------------------------------------------------------ RED-SWAP
def test_RED_SWAP_T_and_C_makes_P1_fail_and_no_stage2_trigger():
    P = mkP(offsets={"C": 0.0, "T": 1.0})
    bt, good = con_of(P, "T", "C")
    bt, swapped = con_of(P, "C", "T")
    assert S.decide(good, fake_mech(), True)["verdict"] == "STAGE-2-TRIGGERED"
    sv = S.decide(swapped, fake_mech(), True)
    assert sv["clauses"]["P1_lo_gt_0"] is False and sv["verdict"] != "STAGE-2-TRIGGERED" and sv["outcome"] == "O4"


# ------------------------------------------------------------------------------------------------------ RED-DROP
def test_RED_DROP_one_token_from_one_arm_fails_ctrl_set_and_per_token():
    toks = [f"{i:016x}" for i in range(10)]
    ok = FR.ctrl_set({"rows:T": toks, "rows:C": toks, "dump:T": toks}, None)
    assert ok["pass"]
    bad = FR.ctrl_set({"rows:T": toks, "rows:C": toks[:-1], "dump:T": toks}, None)
    assert not bad["pass"] and "rows:C" in bad["unequal"]
    full = FR.ctrl_set({"rows:T": toks}, 12146)
    assert not full["pass"] and full["population_problems"]
    rows = {t: {"token": t, "log": "l", "arms": {a: {s: {m: 0.5 for m in METRICS} for s in S.SCORERS} for a in ("T", "C")}} for t in toks}
    del rows[toks[3]]["arms"]["C"]
    with pytest.raises(PR.Refused):
        PR.per_token(rows, ["T", "C"])


# ------------------------------------------------------------------------------------------------------ synthetic files for CTRL-T / the end-to-end read
def mk_arm(n, rng, good=False):
    tr, tp = {}, {}
    for nm in ("tr", "tp"):
        d = {k: (rng.random((n, 64)) < 0.9).astype(float) for k in ("nc", "dac", "ttc", "c")}
        d["nc"] = rng.choice([0.0, 0.5, 1.0], size=(n, 64), p=[0.04, 0.06, 0.90])      # NC = 0.5 occurs in NAVSIM (partial credit): it is not a gate pass
        d["ddc"] = np.ones((n, 64))
        d["ep"] = rng.uniform(0.3, 1.0, (n, 64))
        d["pdms"] = d["nc"] * d["dac"] * (5 * d["ep"] + 5 * d["ttc"] + 2 * d["c"]) / 12
        (tr if nm == "tr" else tp).update(d)
    pick = tr["pdms"].argmax(1) if good else rng.integers(0, 64, n)
    return tr, tp, pick


def write_arm(dirp, name, n, rng, good, logs, tokens, perturb_table=False):
    tr, tp, pick = mk_arm(n, rng, good)
    ar = np.arange(n)
    prop = rng.normal(0, 1, (n, 64, 8, 3))
    logits = rng.normal(0, 1, (n, 64, 6)).astype(np.float32)
    np.savez(dirp / f"{name}.proposals.npz", token=np.array(tokens), proposals=prop, pick=pick, logits=logits)
    np.savez(dirp / f"{name}.npz", token=np.array(tokens), poses=prop[ar, pick], sampling=np.array([8.0, 0.5]))
    tab = dirp / f"{name}_table.jsonl"
    rows = []
    with open(tab, "w", encoding="utf-8") as f:
        for i, t in enumerate(tokens):
            trr = {k: v[i].tolist() for k, v in tr.items()}
            if perturb_table and i == 5:
                trr["pdms"][pick[i]] += 1e-6
            f.write(json.dumps({"token": t, "log": logs[i], "pick": int(pick[i]), "tr": trr, "tp": {k: v[i].tolist() for k, v in tp.items()},
                                "tp20_pick": {k: float(v[i, pick[i]]) for k, v in tp.items()}}) + "\n")
    with open(dirp / f"{name}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["", "token", "valid", "no_at_fault_collisions", "drivable_area_compliance", "ego_progress", "time_to_collision_within_bound", "comfort",
                    "driving_direction_compliance", "score"])
        for i, t in enumerate(tokens):
            w.writerow([i, t, True] + [float(tr[k][i, pick[i]]) for k in ("nc", "dac", "ep", "ttc", "c", "ddc", "pdms")])
    return {"tr": tr, "tp": tp, "pick": pick}


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    d = tmp_path_factory.mktemp("sft")
    rng = np.random.default_rng(7)
    n = 90
    tokens = [f"{i:016x}" for i in range(n)]
    logs = [f"log{i % 9}" for i in range(n)]
    A = {"T": write_arm(d, "T", n, rng, True, logs, tokens), "C": write_arm(d, "C", n, rng, False, logs, tokens)}
    with open(d / "rows.jsonl", "w", encoding="utf-8") as f:
        for i, t in enumerate(tokens):
            arms = {}
            for a, X in A.items():
                p = X["pick"][i]
                arms[a] = {"tracked": {k: float(X["tr"][k][i, p]) for k in ("nc", "dac", "ddc", "ttc", "ep", "c", "pdms")},
                           "transparent": {k: float(X["tp"][k][i, p]) for k in ("nc", "dac", "ddc", "ttc", "ep", "c", "pdms")}}
            f.write(json.dumps({"token": t, "log": logs[i], "arms": arms}) + "\n")
    return {"dir": d, "tokens": tokens, "logs": logs, "A": A, "n": n}


def ns(synth, **kw):
    d = synth["dir"]
    base = dict(rows=[str(d / "rows.jsonl")], t="T", c="C", ts=None, cs=None, w0_digest_c=None, w0_ref=None, det_seam_t=None, det_full_seam_t=None, seam_ledger=None,
                convert_report=None, fam_ci=None, fam_ci_s=None, fam_labels=None, tactical=None, strategic=None, dry_run="unit test", n_boot=200, out="x", text=None, verdict_line=None)
    for k in ("t", "c", "ts", "cs"):
        for f in ("seam", "dump", "csv", "argv"):
            base[f"{f}_{k}"] = None
        base[f"proptable_{k}"] = None
    for k in ("t", "c"):
        nm = k.upper()
        base.update({f"seam_{k}": str(d / f"{nm}.npz"), f"dump_{k}": str(d / f"{nm}.proposals.npz"), f"csv_{k}": str(d / f"{nm}.csv"), f"proptable_{k}": [str(d / f"{nm}_table.jsonl")]})
    base.update(kw)
    return argparse.Namespace(**base)


# ------------------------------------------------------------------------------------------------------ CTRL-T and RED-PERTURB
def test_ctrl_t_passes_and_RED_PERTURB_one_pdms_fails(synth, tmp_path):
    d, tokens, n = synth["dir"], synth["tokens"], synth["n"]
    T = S.load_table_arrays([str(d / "T_table.jsonl")], tokens)
    dump = S.load_dump(str(d / "T.proposals.npz"), tokens)
    ok = S.ctrl_t(T, str(d / "T.csv"), tokens, dump["pick"], "T")
    assert ok["pass"] and ok["n_over_bar"] == 0
    rng = np.random.default_rng(7)
    write_arm(tmp_path, "P", n, rng, True, synth["logs"], tokens, perturb_table=True)
    Tp = S.load_table_arrays([str(tmp_path / "P_table.jsonl")], tokens)
    bad = S.ctrl_t(Tp, str(tmp_path / "P.csv"), tokens, S.load_dump(str(tmp_path / "P.proposals.npz"), tokens)["pick"], "P")
    assert not bad["pass"] and bad["n_over_bar"] == 1 and bad["max_abs_diff"]["pdms"] > 9e-7


def test_ctrl_t_fails_when_the_tables_pick_is_not_the_dumps(synth):
    d, tokens = synth["dir"], synth["tokens"]
    T = S.load_table_arrays([str(d / "T_table.jsonl")], tokens)
    wrong = (S.load_dump(str(d / "T.proposals.npz"), tokens)["pick"] + 1) % 64
    assert not S.ctrl_t(T, str(d / "T.csv"), tokens, wrong, "T")["pass"]


def test_table_missing_a_token_is_refused(synth, tmp_path):
    d, tokens = synth["dir"], synth["tokens"]
    lines = open(d / "T_table.jsonl", encoding="utf-8").read().splitlines()
    (tmp_path / "short.jsonl").write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(PR.Refused):
        S.load_table_arrays([str(tmp_path / "short.jsonl")], tokens)


# ------------------------------------------------------------------------------------------------------ CTRL-ARGV and RED-ARGV
BASE = ["--scorer-sees-trunk", "--nav-mode", "command", "--init-from", "/w/c1/model_final.pt", "--steps", "1000", "--batch", "4", "--accum", "64", "--lr", "5e-5"]
ARGV_C = BASE + ["--rank-w", "0.0", "--out", "/w/c"]
ARGV_T = BASE + ["--rank-w", "0.37", "--rank-heads", "ep,gate", "--rank-ep-gap", "0.02", "--rank-gate-topk", "8", "--out", "/w/t"]


def argvs(**over):
    d = {"C": ARGV_C, "T": ARGV_T, "Cs": ARGV_C + ["--seed", "1"], "Ts": ARGV_T + ["--seed", "1"]}
    d["Cs"] = [x if x != "/w/c" else "/w/cs" for x in d["Cs"]]
    d["Ts"] = [x if x != "/w/t" else "/w/ts" for x in d["Ts"]]
    d.update(over)
    return {k: S.parse_argv(v) for k, v in d.items()}


def test_ctrl_argv_passes_on_exactly_the_cards_difference():
    r = S.ctrl_argv(argvs())
    assert r["pass"], r["fails"]
    assert set(r["detail"]["T_minus_C_flags"]) == S.ALLOWED_TC and set(r["detail"]["Cs_minus_C_flags"]) == S.ALLOWED_SEED
    assert S.ctrl_argv({k: v for k, v in argvs().items() if k in ("C", "T")})["pass"]


def test_RED_ARGV_one_extra_flag_fails_in_every_position():
    assert not S.ctrl_argv(argvs(T=ARGV_T + ["--w-field", "2.0"]))["pass"]
    assert not S.ctrl_argv(argvs(C=ARGV_C + ["--label-workers", "13"]))["pass"]
    assert not S.ctrl_argv(argvs(Ts=ARGV_T + ["--seed", "1", "--lr", "1e-4"]))["pass"]


def test_ctrl_argv_other_deviations_fail():
    assert not S.ctrl_argv(argvs(C=[x for x in ARGV_C if x not in ("--rank-w", "0.0")]))["pass"]                                   # C must NAME --rank-w 0.0
    assert not S.ctrl_argv(argvs(T=[("5" if x == "0.02" else x) for x in ARGV_T]))["pass"]                                         # wrong gap
    assert not S.ctrl_argv(argvs(T=[x for x in ARGV_T if x not in ("--rank-heads", "ep,gate")]))["pass"]                          # heads dropped
    assert not S.ctrl_argv(argvs(Ts=ARGV_T + ["--seed", "2"]))["pass"]                                                              # seed must be 1
    assert not S.ctrl_argv(argvs(C=[("2000" if x == "1000" else x) for x in ARGV_C]))["pass"]                                      # steps
    assert not S.ctrl_argv({"C": S.parse_argv(ARGV_C)})["pass"]                                                                    # T missing
    assert not S.ctrl_argv(argvs(T=[("/w/c" if x == "/w/t" else x) for x in ARGV_T]))["pass"]                                      # shared --out


def test_parse_argv_forms(tmp_path):
    assert S.parse_argv("--a 1 --b x y --c") == {"--a": ("1",), "--b": ("x", "y"), "--c": ()}
    assert S.parse_argv(["--a=1", "--b", "-1"]) == {"--a": ("1",), "--b": ("-1",)}
    p = tmp_path / "argv.json"
    p.write_text(json.dumps({"argv": ["--steps", "1000"]}), encoding="utf-8")
    assert S.parse_argv(str(p)) == {"--steps": ("1000",)}


# ------------------------------------------------------------------------------------------------------ CTRL-W0
H1, H2 = "f09949e2a5f2f9661ad47457e77ddfd10f49117d64e095487809c1d04f7a34a4", "a" * 64


def test_ctrl_w0(tmp_path):
    assert S.ctrl_w0(H1, H1)["pass"]
    f = tmp_path / "step0.json"
    f.write_text(json.dumps({"step": 0, "model_sha256": H1}), encoding="utf-8")
    assert S.ctrl_w0(str(f), H1)["pass"]
    assert not S.ctrl_w0(H2, H1)["pass"]
    assert not S.ctrl_w0(None, H1)["pass"] and not S.ctrl_w0(H1, None)["pass"] and not S.ctrl_w0("no digest here", H1)["pass"]
    assert not S.ctrl_w0(H1, H1, {"T": H2})["pass"]


# ------------------------------------------------------------------------------------------------------ CTRL-DET
def test_ctrl_det(tmp_path):
    n = 260
    rng = np.random.default_rng(1)
    toks = np.array([f"{i:016x}" for i in range(n)])
    poses = rng.normal(0, 1, (n, 8, 3))
    np.savez(tmp_path / "full.npz", token=toks, poses=poses, sampling=np.array([8.0, 0.5]))
    sub = np.arange(0, 220)
    np.savez(tmp_path / "re.npz", token=toks[sub], poses=poses[sub], sampling=np.array([8.0, 0.5]))
    assert S.ctrl_det(str(tmp_path / "full.npz"), str(tmp_path / "re.npz"))["pass"]
    bad = poses[sub].copy()
    bad[7, 3, 1] += 1e-12
    np.savez(tmp_path / "bad.npz", token=toks[sub], poses=bad, sampling=np.array([8.0, 0.5]))
    r = S.ctrl_det(str(tmp_path / "full.npz"), str(tmp_path / "bad.npz"))
    assert not r["pass"] and r["tokens_not_bit_identical"] == 1
    np.savez(tmp_path / "few.npz", token=toks[:50], poses=poses[:50], sampling=np.array([8.0, 0.5]))
    assert not S.ctrl_det(str(tmp_path / "full.npz"), str(tmp_path / "few.npz"))["pass"]
    np.savez(tmp_path / "alien.npz", token=np.array(["zz" + t[2:] for t in toks[sub]]), poses=poses[sub], sampling=np.array([8.0, 0.5]))
    assert not S.ctrl_det(str(tmp_path / "full.npz"), str(tmp_path / "alien.npz"))["pass"]


# ------------------------------------------------------------------------------------------------------ the instruments as arrays equal the originals
def test_spearman_and_auc_match_the_definitions():
    a, b = np.array([1.0, 2.0, 3.0, 4.0]), np.array([1.0, 3.0, 2.0, 4.0])
    assert abs(S.spearman(a, b) - 0.8) < 1e-12
    assert math.isnan(S.spearman(a[:2], b[:2])) and math.isnan(S.spearman(np.ones(4), b))
    assert S.auc_within(np.array([.9, .8, .1, .2]), np.array([1, 1, 0, 0.])) == 1.0
    assert S.auc_within(np.array([.5, .5]), np.array([1, 0.])) == 0.5
    assert math.isnan(S.auc_within(np.array([.5, .5]), np.array([1, 1.])))


def test_mech_arrays_regret_split_sums_to_regret_and_zero_identities(synth):
    d, tokens = synth["dir"], synth["tokens"]
    A = S.mech_arrays(S.load_table_arrays([str(d / "C_table.jsonl")], tokens), S.load_dump(str(d / "C.proposals.npz"), tokens))
    tot = A["regret_gate"] + A["regret_ep"] + A["regret_ttc"] + A["regret_c"] + A["regret_other"]
    assert np.allclose(tot, A["regret"], atol=1e-9) and (A["regret"] >= -1e-12).all()
    assert (A["top1"][A["regret"] > 1e-7] == 0).all()
    T = S.mech_arrays(S.load_table_arrays([str(d / "T_table.jsonl")], tokens), S.load_dump(str(d / "T.proposals.npz"), tokens))
    assert (T["top1"] == 100.0).all() and np.allclose(T["regret"], 0.0)       # T picks argmax PDMS in the synthetic data


def test_regret_split_equals_regret_decomp_lines_verbatim(synth):
    d, tokens = synth["dir"], synth["tokens"]
    Tb = S.load_table_arrays([str(d / "C_table.jsonl")], tokens)
    A = S.mech_arrays(Tb, S.load_dump(str(d / "C.proposals.npz"), tokens))
    n = len(tokens); ar = np.arange(n); pk = Tb["pick"]; tr = {k: Tb[f"tr_{k}"] for k in S.TAB_KEYS}        # regret_decomp.py lines 8-14, verbatim
    o = tr["pdms"].argmax(1)
    reg = (tr["pdms"].max(1) - tr["pdms"][ar, pk]) * 100
    z = (tr["nc"][ar, pk] < .5) | (tr["dac"][ar, pk] < .5)
    lin = ~z & (tr["nc"][ar, o] > .5) & (tr["dac"][ar, o] > .5) & (tr["nc"][ar, pk] > .5) & (tr["dac"][ar, pk] > .5)
    comp = {k: ((w * (tr[k][ar, o] - tr[k][ar, pk])) / 12 * 100) for k, w in (("ep", 5), ("ttc", 5), ("c", 2))}
    assert (~z & ~lin).any(), "the synthetic data must contain an NC = 0.5 pick, else this test proves nothing"
    assert np.isclose(A["regret"].mean(), reg.mean())
    assert np.isclose(A["regret_gate"].sum() / n, reg[z].sum() / n) and np.isclose(A["regret_other"].sum() / n, reg[~z & ~lin].sum() / n)
    for k in ("ep", "ttc", "c"):
        assert np.isclose(A[f"regret_{k}"].sum() / n, comp[k][lin].sum() / n)


def test_mech_matches_head_auc_when_importable(synth):
    sk = pytest.importorskip("sklearn")
    del sk
    import importlib.util
    here = os.path.dirname(os.path.abspath(__file__))
    p = os.path.join(here, "..", "..", "raw", "2026-11-20-p5c-next")
    if not os.path.exists(os.path.join(p, "head_auc.py")):
        pytest.skip("head_auc.py not in this tree")
    import sys
    sys.path.insert(0, p)
    HA = importlib.import_module("head_auc")
    rng = np.random.default_rng(3)
    for _ in range(20):
        a, b = rng.normal(size=12), rng.normal(size=12)
        x = HA.spearman(a, b), S.spearman(a, b)
        assert (math.isnan(x[0]) and math.isnan(x[1])) or x[0] == x[1]
        s, y = rng.normal(size=64), (rng.random(64) < 0.7).astype(float)
        y1, y2 = HA.auc_within(s, y), S.auc_within(s, y)
        assert (math.isnan(y1) and math.isnan(y2)) or y1 == y2


def test_wdraws_with_unit_weights_equals_boot_draws():
    P = mkP(offsets={"A": 0.0})
    bt = PR.Boot(P["logs"], 200)
    v = P["A"]["tracked"]["pdms"]
    assert np.array_equal(S.wdraws(bt, v, np.ones(len(v))), bt.draws(v))
    w = np.zeros(len(v))
    assert S.wci(bt, v, w)["status"] == "NOT COMPUTABLE"


def test_mech_decide_rules():
    n1 = np.ones(5)
    z = lambda pt, lo, hi: {"point": pt, "lo": lo, "hi": hi, "n_tokens": 5}   # noqa: E731
    con = {"ep_spearman_both_defined": z(0.12, 0.05, 0.2), "regret": z(0.1, -0.2, 0.3), "oracle": z(0.0, -0.1, 0.1), "zeroed": z(0, 0, 0), "zeroed_by_dac": z(0, 0, 0),
           "zeroed_by_nc": z(0, 0, 0), "conf_wrong_dac": z(0, 0, 0), "conf_wrong_nc": z(0, 0, 0), "regret_gate": z(0, 0, 0), "regret_ep": z(0, 0, 0), "regret_ttc": z(0, 0, 0),
           "regret_c": z(0, 0, 0), "regret_other": z(0, 0, 0)}
    m = S.mech_decide(con)
    assert m["M1_ep_spearman"]["moved"] and m["mechanism_moved"] and not m["m4_damaged"]
    con["ep_spearman_both_defined"] = z(0.09, 0.05, 0.2)                        # below +0.10
    assert not S.mech_decide(con)["M1_ep_spearman"]["moved"]
    con["ep_spearman_both_defined"] = z(0.2, -0.01, 0.4)                        # CI-lo <= 0
    assert not S.mech_decide(con)["M1_ep_spearman"]["moved"]
    con["regret"] = z(-0.3, -0.5, -0.1)
    assert S.mech_decide(con)["M3_regret"]["moved"]
    con["oracle"] = z(-0.5, -0.9, -0.1)
    assert S.mech_decide(con)["m4_damaged"]
    _ = n1


# ------------------------------------------------------------------------------------------------------ CTRL-ID, goal, strategic
def test_ctrl_id_zero_exact_and_red_detects_nonzero(synth):
    P = mkP(offsets={"T": 0.0})
    r = S.ctrl_id(P, P["logs"], None, "T", 200)
    assert r["pass"] and r["nonzero_deltas"] == [] and r["P1"] is False
    assert FR.zero_exact({"a": {"point": 0.0, "lo": 0.0, "hi": 1e-12}}) == ["a.hi"]                 # the detector can fail


def test_goal_flag_below_goal_under_94_55():
    lv = {"point": 92.55, "lo": 91.8, "hi": 93.26}
    assert S.goal_flag(lv)["status"] == "BELOW GOAL" and S.goal_flag(lv)["ci_hi_below_goal"]
    assert S.goal_flag({"point": 94.60, "lo": 94.0, "hi": 95.0})["status"].startswith("AT/ABOVE")
    assert S.goal_flag({"point": 94.54, "lo": 94.0, "hi": 95.0})["status"] == "BELOW GOAL"


def test_strategic_is_reported_in_full_never_paused_and_a_paused_json_is_refused():
    j = {"instrument": "strategic_family", "arms": {"T": {"s_dec_acc": 0.5}}, "paired_vs_base": {"T_minus_C": {}}, "controls": {"ok": 1}}
    b = S.strategic_block(j, 12146)
    assert b["status"].startswith("REPORTED IN FULL") and "PAUSED" not in json.dumps(b)
    with pytest.raises(PR.Refused):
        S.strategic_block({"arms": {}, "note": "PAUSED"}, 5)
    assert S.strategic_block(None, 5)["status"] == "NOT SUPPLIED"
    assert S.strategic_block({"arms": {}, "controls": {"x": "INVALID"}}, 5)["status"].startswith("INVALID")
    import p5c_tactical as PT
    assert "PAUSED" not in json.dumps(PT.STRATEGIC) and "PAUSED" not in json.dumps(PR.STRATEGIC)


def test_relabel_tactical_drops_strategic_stub_and_renames():
    t = {"kind": "P5C_TACTICAL", "strategic": {"status": "REPORTED_SEPARATELY"}, "command_adherence": {"arms": {"C1": {}, "C0": {}, "base": {}}, "paired_C1_minus_x": {}}}
    r = S.relabel_tactical(t)
    s = json.dumps(r)
    assert '"strategic"' not in s and '"C1"' not in s and '"T"' in s and '"C"' in s and "paired_T_minus_x" in s and r["pooled"] is False


def test_fam_cells_direction_label_check_and_lead_components():
    e = lambda lo, hi, pt: {"status": "OK", "n_tokens": 9, "A": {"point": 1, "lo": 0, "hi": 2}, "B": {"point": 1, "lo": 0, "hi": 2}, "diff_B_minus_A": {"point": pt, "lo": lo, "hi": hi}}   # noqa: E731
    j = {"label_a": "C", "label_b": "T", "estimator": "x", "frames": {"executed_tracked": {"families": {
        "longitudinal": {"min_ttc_s": e(-0.5, -0.1, -0.3), "headway_min_m": {"status": "UNAVAILABLE", "n_tokens": 0}, "speed_mae_mps": e(0.1, 0.5, 0.3)},
        "lateral": {"cross_mae_m": e(-0.1, 0.1, 0.0)}, "tactical": {"lateral_decision.accuracy": e(-0.2, -0.01, -0.1), "x.per_class.a.recall": e(0.1, 0.2, 0.15)},
        "strategic": {"status": "PAUSED"}}}}}
    f = S.fam_cells(j)
    assert f["status"] == "COMPUTED"
    assert set(f["adverse"]) == {"LONGITUDINAL.min_ttc_s", "LONGITUDINAL.speed_mae_mps", "TACTICAL.lateral_decision.accuracy"}      # higher-is-better min_ttc fell; recall is not judged
    assert "headway_min_m" in f["lead_agent_components"]["components"] and "PAUSED" not in json.dumps(f)
    rise = {"status": "OK", "n_tokens": 9, "A": {"point": 13.6, "lo": 13.0, "hi": 14.0}, "B": {"point": 13.9, "lo": 13.3, "hi": 14.3}, "diff_B_minus_A": {"point": 0.3, "lo": 0.15, "hi": 0.43}}
    assert S.adverse_cell("min_ttc_s", rise) is False and S.adverse_cell("headway_min_m", rise) is False       # a RISE in a safety margin is not adverse (the inherited rule said it was)
    assert S.adverse_cell("speed_mae_mps", rise) is True                                                        # a rise in an error IS adverse
    assert S.fam_cells(dict(j, label_a="T", label_b="C"))["status"] == "NOT-EVALUATED"
    assert S.fam_cells(None)["status"] == "NOT SUPPLIED"


def test_polarity_is_the_one_shared_rule_and_red_arm(monkeypatch):
    """v2 (2026-11-20): scorerft_read carries NO polarity table of its own; it uses field_read.entry_adverse_pol. RED arm: break the shared rule -> this read breaks."""
    assert not hasattr(S, "POLARITY_HIGHER")
    rise = {"status": "OK", "n_tokens": 9, "A": {"point": 13.6, "lo": 13.0, "hi": 14.0}, "B": {"point": 13.9, "lo": 13.3, "hi": 14.3}, "diff_B_minus_A": {"point": 0.3, "lo": 0.15, "hi": 0.43}}
    for k in ("min_ttc_s", "headway_min_m", "time_gap_min_s"):
        assert S.adverse_cell(k, rise) is False and S.adverse_cell(k, rise) == S.FR.entry_adverse_pol(k, rise)
    old = lambda m: ("higher" if ("within_" in m or "accuracy" in m or "kappa" in m) else "magnitude" if "bias" in m else "closer_to_1" if "progress_ratio" in m else "lower")   # noqa: E731
    monkeypatch.setattr(S.PR, "fam_direction", old)                                  # an old p5c_read export underneath
    assert S.adverse_cell("min_ttc_s", rise) is False                                # still corrected: the rule lives in field_read
    monkeypatch.setattr(S.FR, "SAFETY_MARGIN_HIGHER", ())                            # the deliberate regression
    assert S.adverse_cell("min_ttc_s", rise) is True                                 # the defect is back, so the assertions above would fail


# ------------------------------------------------------------------------------------------------------ the end-to-end read on synthetic files
def test_end_to_end_dry_read_stage1_stamps_and_controls(synth):
    r = S.run_read(ns(synth))
    assert r["STAMP"] == S.DRY_STAMP and r["verdict"]["verdict"].startswith("DRY RUN:")
    names = [c["control"] for c in r["controls"]]
    assert {"CTRL-SET", "CTRL-T", "CTRL-CSV", "CTRL-SEAM", "CTRL-ID"} <= set(names)
    assert all(c["pass"] for c in r["controls"]), [c for c in r["controls"] if not c["pass"]]
    assert {u["control"] for u in r["controls_not_available"]} == {"CTRL-ARGV", "CTRL-W0", "CTRL-DET"}
    assert r["stage"] == 1 and r["replicate"]["status"].startswith("PENDING") and r["verdict"]["clauses"]["P2"].startswith("PENDING")
    assert r["comfort_transparent"]["forms"].keys() == {"transparent_20_pose_plan", "transparent_8_submitted_poses"}
    assert max(r["comfort_transparent"]["rows_transparent_c_equals_20_pose"].values()) < 1e-12
    f = r["families"]
    assert f["pooled"] is False and f["TACTICAL"]["(c)_goal_setting"]["status"] == "NOT COMPUTABLE" and f["TACTICAL"]["(c)_goal_setting"]["n"] == 90
    assert f["STRATEGIC"]["status"] == "NOT SUPPLIED" and "PAUSED" not in json.dumps(r["families"])
    assert "(b)_selection_quality" in f["TACTICAL"] and "T_minus_C" in f["TACTICAL"]["(b)_selection_quality"]
    line = S.verdict_line(r)
    assert line.startswith(S.DRY_STAMP) and "look 1 of H-SCORERFT-1" in line and "multiplicity" in line
    txt = S.text_report(r)
    assert "NOT AVAILABLE: CTRL-ARGV" in txt and "DDC: reported, not a guard" in txt and "BELOW GOAL" in txt


def test_end_to_end_self_read_is_all_zero_and_no_p1(synth):
    r = S.run_read(ns(synth, c="T", seam_c=str(synth["dir"] / "T.npz"), dump_c=str(synth["dir"] / "T.proposals.npz"), csv_c=str(synth["dir"] / "T.csv"),
                      proptable_c=[str(synth["dir"] / "T_table.jsonl")]))
    assert r["self_read"] and r["verdict"]["clauses"]["P1_lo_gt_0"] is False
    assert not S.FR.zero_exact({m: {s: r["T_minus_C"][m][s] for s in ("tracked", "transparent", "TD")} for m in METRICS})


def test_real_read_refuses_loudly_without_its_controls_inputs(synth):
    with pytest.raises(PR.Refused) as e:
        S.run_read(ns(synth, dry_run=None))
    msg = str(e.value)
    for need in ("--argv-t", "--argv-c", "--w0-digest-c", "--w0-ref", "--det-seam-t", "--seam-ledger", "--fam-ci", "--strategic", "--tactical"):
        assert need in msg, need


def test_stage2_needs_both_arms_and_their_inputs(synth):
    with pytest.raises(PR.Refused):
        S.run_read(ns(synth, ts="T"))


def test_every_control_failure_turns_the_read_invalid(synth, tmp_path):
    d = synth["dir"]
    rng = np.random.default_rng(7)
    write_arm(tmp_path, "P", synth["n"], rng, True, synth["logs"], synth["tokens"], perturb_table=True)
    r = S.run_read(ns(synth, proptable_t=[str(tmp_path / "P_table.jsonl")]))
    assert r["controls_all_pass"] is False and r["verdict"]["verdict"].startswith("DRY RUN:INVALID")
    _ = d


def test_main_writes_json_text_and_verdict_line(synth, tmp_path, capsys):
    d = synth["dir"]
    argv = ["--rows", str(d / "rows.jsonl"), "--t", "T", "--c", "C", "--dry-run", "t", "--n-boot", "100", "--out", str(tmp_path / "o.json"), "--text", str(tmp_path / "o.txt"),
            "--verdict-line", str(tmp_path / "o.line")]
    for k in ("t", "c"):
        nm = k.upper()
        argv += [f"--seam-{k}", str(d / f"{nm}.npz"), f"--dump-{k}", str(d / f"{nm}.proposals.npz"), f"--csv-{k}", str(d / f"{nm}.csv"), f"--proptable-{k}", str(d / f"{nm}_table.jsonl")]
    import sys
    old = sys.argv
    sys.argv = ["scorerft_read.py"] + argv
    try:
        assert S.main() == 0
    finally:
        sys.argv = old
    assert json.load(open(tmp_path / "o.json", encoding="utf-8"))["STAMP"] == S.DRY_STAMP
    assert open(tmp_path / "o.line", encoding="utf-8").read().startswith(S.DRY_STAMP)
    assert "ZZSFT_READ" in capsys.readouterr().out


def test_prereg_constants_are_the_cards():
    assert S.PREREG_SHA256 == "adc1b2ae8d3cc9b3dfed00a20a3c39197428cda8492be34af3bb9ed24010ec91"
    assert (S.SEED, S.NBOOT, S.GOAL, S.N_NAVTEST, S.L_NAVTEST, S.DET_N_MIN, S.STEPS_EXPECTED) == (20260927, 10000, 94.55, 12146, 136, 200, 1000)
    assert S.G1_SUBS == ("nc", "dac", "ttc", "ep", "c")


# ====================================================================================================== card PICK (PREREG_SCORERFT_PICK.md, H-SCORERFT-PICK-1)
# Deliberate-regression (RED) arms added with the pick card, each must turn the read INVALID / the control FAIL:
#   RED-SWAP-P  `--t C --c P` under pick is REFUSED          RED-ARGV-P  one extra flag, --rank-ep-gap on P, or heads pick,ep => CTRL-ARGV FAILS
#   RED-S0      one step-0 metric one ulp off => FAIL        RED-ORDER   two steps' `pos` swapped => FAIL
#   RED-CODE    model.py sha, device or args.lr differs, or P's train.py equals C's => FAIL
#   RED-HARNESS a different tip or `rule` => FAIL            RED-WRULE   status K2, heads ep,gate, or rank_w off by one candidate => FAIL
#   RED-CREUSE  one C row perturbed by 1e-6 or a ledger sha changed => FAIL
import hashlib
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
EVID = "E:/Projects/TanitAD-artifacts/scorer_gym_20261107/scorer_ft/o3/card/evidence"
S0_KEYS = ["assign_d", "batch", "bce_c", "bce_dac", "bce_ddc", "bce_ep", "bce_nc", "bce_ttc", "epoch", "lr", "pos", "pos_L1", "samples", "score", "scorer_cov", "step", "traj_L1", "winners", "yaw_L1"]
BASEP = ["--scorer-sees-trunk", "--nav-mode", "command", "--init-from", "/w/c1/model_final.pt", "--steps", "1000", "--batch", "4", "--accum", "64", "--lr", "5e-5"]
ARGV_PP = BASEP + ["--rank-w", "0.25", "--rank-heads", "pick", "--out", "/w/p"]


def argvs_p(**over):
    d = {"C": ARGV_C, "T": ARGV_PP, "Cs": [x if x != "/w/c" else "/w/cs" for x in ARGV_C + ["--seed", "1"]], "Ts": [x if x != "/w/p" else "/w/ps" for x in ARGV_PP + ["--seed", "1"]]}
    d.update(over)
    return {k: S.parse_argv(v) for k, v in d.items()}


def fake_mech_pick(m3=False, m4=False, zup=False, m1=False):
    return {"m4_damaged": m4, "m3_moved": m3, "mechanism_moved": m3, "Zup": zup, "M1_ep_spearman": {"moved_flag": m1, "decisive": False}}


# ---- outcomes
def test_parent_card_is_the_default_and_unchanged(synth):
    r = S.run_read(ns(synth))                                                          # no `card` attribute at all: the parent card
    assert r["card_key"] == "H-SCORERFT-1" and r["card"] == S.CARDS["scorer_ft"]["card"] and r["card_sha256"] == S.PREREG_SHA256
    assert "T_minus_C" in r and "P_minus_C" not in r and "T" in r["levels"] and "P" not in r["levels"]
    assert S.CARDS["scorer_ft"]["next_arm"] is S.NEXT_ARM and S.CARDS["scorer_ft"]["verdict"] is S.OUTCOME_VERDICT and S.CARDS["scorer_ft"]["allowed_tc"] is S.ALLOWED_TC
    assert set(S.CARDS["pick"]["labels"].values()) == {"P", "C", "Ps", "Cs"}
    with pytest.raises(PR.Refused, match="belongs to card pick"):
        S.run_read(ns(synth, metrics_c="x"))                                           # a pick-only flag under the parent card


def test_pick_outcome_order_OP1_to_OP7():
    nop1 = fake_con(pdms=(-0.5, -1.0, -0.1))                                           # P1 false, d0 CI-hi < 0
    flat = fake_con(pdms=(0.0, -0.3, 0.3))                                             # P1 false, d0 CI-hi >= 0
    assert S.decide_pick(fake_con(adverse=("ttc",)), fake_mech_pick(), True)["outcome"] == "OP3"
    assert S.decide_pick(flat, fake_mech_pick(m3=True, m4=True), True)["outcome"] == "OP4"                      # OP4 wins over OP5
    assert S.decide_pick(nop1, fake_mech_pick(m3=True), True)["outcome"] == "OP5"                               # OP5 wins over OP6
    assert S.decide_pick(flat, fake_mech_pick(m3=True), True)["outcome"] == "OP5"
    z = S.decide_pick(nop1, fake_mech_pick(zup=True), True)
    assert z["outcome"] == "OP6" and "pick8" in z["next_arm"]                                                    # OP6 with Z-up: the P-top8 arm
    nz = S.decide_pick(nop1, fake_mech_pick(zup=False), True)
    assert nz["outcome"] == "OP6" and "TEACHER-N7" in nz["next_arm"] and "pick8" not in nz["next_arm"]         # without Z-up: OP7's next
    o7 = S.decide_pick(flat, fake_mech_pick(m1=True), True)
    assert o7["outcome"] == "OP7" and "TEACHER-N7" in o7["next_arm"]                                            # M1 alone is not decisive
    for o in ("OP3", "OP4", "OP5", "OP6", "OP7"):
        v = S.decide_pick(fake_con(adverse=("ttc",)) if o == "OP3" else (nop1 if o in ("OP5", "OP6") else flat), fake_mech_pick(m4=(o == "OP4"), m3=(o == "OP5")), True)
        assert "training variance was NOT measured" in v["wording"], o
    assert S.decide_pick(flat, None, True)["verdict"] == "INCOMPLETE"
    assert S.decide_pick(fake_con(), fake_mech_pick(), None)["verdict"] == "INVALID"
    P = stage2_P(0.1)
    bt, c0 = con_of(P, "T", "C")
    rep = S.pair_rep(bt, P, "C", "T", "Cs", "Ts")
    v1 = S.decide_pick(c0, fake_mech_pick(), True, rep)
    assert v1["outcome"] == "OP1" and "ADOPT" in v1["wording"] and "GUARD-FAIL" in v1["wording"]
    P2 = stage2_P(0.6)
    bt, c0 = con_of(P2, "T", "C")
    assert S.decide_pick(c0, fake_mech_pick(), True, S.pair_rep(bt, P2, "C", "T", "Cs", "Ts"))["outcome"] == "OP2"


def test_pick_stage1_p1_g1_triggers_stage2_never_adopt():
    v = S.decide_pick(fake_con(), fake_mech_pick(), True, None)
    assert v["verdict"] == "STAGE-2-TRIGGERED" and v["outcome"] == "PENDING" and "Cs then Ps" in v["next_arm"] and "ADOPT" not in v["verdict"]
    assert v["clauses"]["P2"] == "PENDING (Stage 2)" and "Never ADOPT" in v["wording"]


def test_pick_every_outcome_carries_its_committed_next_arm():
    assert set(S.NEXT_ARM_PICK) == {"PL1", "PL2", "OP1", "OP2", "OP3", "OP4", "OP5", "OP6_Zup", "OP6_noZup", "OP7"} and all(S.NEXT_ARM_PICK.values())
    assert set(S.OUTCOME_VERDICT_PICK) == {"OP1", "OP2", "OP3", "OP4", "OP5", "OP6", "OP7"} and all(S.OUTCOME_VERDICT_PICK.values())
    for key, phrase in (("PL1", "P-log"), ("PL2", "--rank-detach-ctx"), ("OP4", "--rank-detach-ctx"), ("OP3", "--rank-pick-guard"), ("OP5", "successor card"), ("OP6_Zup", "pick8"), ("OP7", "TEACHER-N7")):
        assert phrase in S.NEXT_ARM_PICK[key], (key, phrase)
    assert not any("--rank-pick-gate-detach" in v for v in S.NEXT_ARM_PICK.values())                          # subsumed by decision 10


def test_pick_op3_names_the_adverse_guard_set():
    v = S.decide_pick(fake_con(adverse=("ttc", "c")), fake_mech_pick(), True)
    assert v["outcome"] == "OP3" and "--rank-pick-guard ttc,c" in v["next_arm"] and "<h," not in v["next_arm"] and v["clauses"]["G1_adverse"] == ["ttc", "c"]


# ---- CTRL-ARGV, pick form
def test_pick_ctrl_argv_passes_on_exactly_the_cards_difference():
    r = S.ctrl_argv(argvs_p(), card="pick")
    assert r["pass"], r["fails"]
    assert set(r["detail"]["P_minus_C_flags"]) == {"--rank-w", "--rank-heads", "--out"} == S.ALLOWED_TC_PICK and "T_minus_C_flags" not in r["detail"]
    assert S.ctrl_argv({k: v for k, v in argvs_p().items() if k in ("C", "T")}, card="pick")["pass"]


def test_RED_ARGV_P_variants_fail():
    def bad(**over):
        return not S.ctrl_argv(argvs_p(**over), card="pick")["pass"]
    assert bad(T=ARGV_PP + ["--w-field", "2.0"]) and bad(C=ARGV_C + ["--label-workers", "13"]) and bad(Ts=ARGV_PP + ["--seed", "1", "--lr", "1e-4"]) and bad(Cs=ARGV_C + ["--seed", "1", "--lr", "1e-4"])
    assert bad(T=ARGV_PP + ["--rank-ep-gap", "0.02"]) and bad(T=ARGV_PP + ["--rank-gate-topk", "8"])
    assert bad(T=[("pick,ep" if x == "pick" else x) for x in ARGV_PP]) and bad(T=[("ep,gate" if x == "pick" else x) for x in ARGV_PP])
    assert bad(T=[("0" if x == "0.25" else x) for x in ARGV_PP]) and bad(T=[("nan" if x == "0.25" else x) for x in ARGV_PP])
    assert bad(C=ARGV_C + ["--rank-heads", "pick"]) and bad(C=[("2000" if x == "1000" else x) for x in ARGV_C]) and bad(T=[("/w/c" if x == "/w/p" else x) for x in ARGV_PP])
    assert bad(Ts=ARGV_PP + ["--seed", "2"]) and bad(T=[x for x in ARGV_PP if x not in ("--rank-heads", "pick")])
    assert any("P " in f or "P:" in f for f in S.ctrl_argv(argvs_p(T=ARGV_PP + ["--w-field", "2.0"]), card="pick")["fails"])      # the label, not the role


def test_cross_card_argv_is_refused():
    assert not S.ctrl_argv(argvs(), card="pick")["pass"]                                # ARGV_T (ep,gate + gap + top-k) under pick
    assert not S.ctrl_argv(argvs_p())["pass"]                                           # ARGV_P under the parent card


def test_argv_script_and_code_tag(tmp_path):
    x = "/w/data/refe_p5c/code_83411c28/refe/train.py --steps 1000"
    assert S.argv_script(x) == "/w/data/refe_p5c/code_83411c28/refe/train.py" and S.code_tag(x) == "code_83411c28"
    assert S.argv_script("--steps 1000") is None and S.code_tag("--steps 1000") is None and S.argv_script(ARGV_PP) is None
    p = tmp_path / "argv.json"
    p.write_text(json.dumps({"argv": x.split()}), encoding="utf-8")
    assert S.code_tag(str(p)) == "code_83411c28" and S.parse_argv(str(p)) == {"--steps": ("1000",)}


# ---- CTRL-S0 / CTRL-ORDER
def mk_metrics(path, n_steps=1000, rank=False, seed0=1, seed_rest=2, rank_w=0.25, extra_row_at0=None):
    """a metrics.jsonl: start + bank events, a training row per step (every key C logs, train.py:2054-2077), onpolicy rows, and P-only rank_* / *_v1 keys when `rank`. Step 0 is a function of seed0 ALONE (so a P and a C
    built with the same seed0 agree at step 0 and, with another seed_rest, differ from step 1 on); ORDER_KEYS are a function of the step alone."""
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps({"event": "start", "at": "2026-11-22T23:01:48Z"}) + "\n" + json.dumps({"event": "bank", "epoch": 0, "n_tuples": 10}) + "\n")
        for s in range(n_steps):
            rng = np.random.default_rng((seed0 if s == 0 else seed_rest) * 1000 + s)
            r = {"step": s, "epoch": 0, "pos": 256 * (s + 1), "traj_L1": float(rng.random()), "score": float(rng.random()), "winners": 4, "batch": 4, "assign_d": 0.0, "scorer_cov": 1.0, "lr": 5e-5 * (1 - s / 2000),
                 "elapsed_s": float(rng.random()) + s, "samples": 256 * (s + 1), "cuda_max_mem_gb": 41.0 + float(rng.random()), "pos_L1": float(rng.random()), "yaw_L1": float(rng.random())}
            for k in ("nc", "dac", "ep", "ttc", "c", "ddc"):
                r[f"bce_{k}"] = float(rng.random())
            if rank:
                r.update(rank_L=float(rng.random()), rank_w=rank_w, rank_pairs_pick=int(rng.integers(5, 50)), rank_sets_pick=3)
            if extra_row_at0 and s == 0:
                f.write(json.dumps(dict(r, traj_L1=r["traj_L1"] + 1.0)) + "\n")              # a resumed run re-logs step 0 first with another value; the LAST row counts
            f.write(json.dumps(r) + "\n")
            if s < 2:
                r2 = np.random.default_rng((seed0 if s == 0 else seed_rest) * 1000 + s + 500000)
                o = {"step": s, "event": "onpolicy", "sets": 256, "pick": float(r2.random()), "random": float(r2.random()), "best": float(r2.random()), "skill": float(r2.random()), "drift_m": 0.0, "lag_steps": 0.0, "cov_norm": 64.0}
                if rank:
                    o.update(pick_v1=float(rng.random()), random_v1=0.5, best_v1=0.9, skill_v1=0.7)
                f.write(json.dumps(o) + "\n")
    return path


def _rewrite(path, fn):
    rows = [json.loads(ln) for ln in open(path, encoding="utf-8").read().splitlines()]
    rows = fn(rows)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_ctrl_s0_full_and_RED(tmp_path):
    c, p = mk_metrics(tmp_path / "c.jsonl"), mk_metrics(tmp_path / "p.jsonl", rank=True)
    r = S.ctrl_s0(str(c), str(p), rank_w=0.25, check_c_sha=False)
    assert r["pass"], r["fails"]
    assert r["n_keys_compared"]["training"] == 19 and sorted(r["keys_compared"]["training"]) == S0_KEYS and r["traj_L1_c"] == r["traj_L1_p"]
    # wall-clock / allocator keys are not model state: their differences are ignored (they differ here, by construction)
    assert r["excluded_present"] == ["cuda_max_mem_gb", "elapsed_s"]

    def edit(fn, **kw):
        q = tmp_path / "q.jsonl"
        shutil.copyfile(p, q)
        _rewrite(str(q), fn)
        return S.ctrl_s0(str(c), str(q), rank_w=0.25, check_c_sha=False, **kw)

    def at0(f_):
        return lambda rows: [f_(x) if (x.get("step") == 0 and "event" not in x) else x for x in rows]
    assert not edit(at0(lambda x: dict(x, traj_L1=float(np.nextafter(x["traj_L1"], 2.0)))))["pass"]                 # RED-S0: one ulp
    assert not edit(at0(lambda x: {k: v for k, v in x.items() if k != "bce_nc"}))["pass"]                                # a C key missing in P
    assert not edit(at0(lambda x: dict(x, mystery_key=1.0)))["pass"]                                                  # a P-only key outside S0_RANK_ONLY
    assert not edit(at0(lambda x: {k: v for k, v in x.items() if k != "rank_L"}))["pass"]                             # P without rank_L
    assert not edit(at0(lambda x: dict(x, rank_L=float("nan"))))["pass"]                                              # rank_L NaN
    assert not edit(at0(lambda x: dict(x, rank_w=0.5)))["pass"]                                                       # rank_w != argv
    assert not edit(lambda rows: [x for x in rows if not (x.get("event") == "onpolicy" and x["step"] == 0)])["pass"]   # an onpolicy row in one arm only
    # a resumed run whose step-0 row is re-logged: the LAST row is the one that counts
    pr = mk_metrics(tmp_path / "pr.jsonl", rank=True, extra_row_at0=True)
    assert S.ctrl_s0(str(c), str(pr), rank_w=0.25, check_c_sha=False)["pass"]
    assert not S.ctrl_s0(str(c), str(p), rank_w=0.25, check_c_sha=True)["pass"]                                       # the C log is not the banked one
    p3 = mk_metrics(tmp_path / "p3.jsonl", rank=True, seed_rest=3)
    assert not S.ctrl_s0(str(c), str(p3), rank_w=0.25, step=1, check_c_sha=False)["pass"]                              # non-vacuity: from step 1 the arms differ (other seed_rest)


def test_ctrl_s0_on_the_stage1_rows(tmp_path):
    fx = json.load(open(os.path.join(HERE, "stage1_s0_rows.json"), encoding="utf-8"))
    assert fx["sha256"]["c"] == S.C_METRICS_SHA256 and fx["sha256"]["t"] == S.T_METRICS_SHA256
    for a in ("c", "t"):
        with open(tmp_path / f"{a}.jsonl", "w", encoding="utf-8", newline="\n") as f:
            for r in fx["rows"][a]:
                f.write(json.dumps(r) + "\n")
    r0 = S.ctrl_s0(str(tmp_path / "c.jsonl"), str(tmp_path / "t.jsonl"), rank_w=0.03125, step=0, check_c_sha=False)
    assert r0["pass"], r0["fails"]
    assert sorted(r0["keys_compared"]["training"]) == sorted(S.S0_KEYS_TRAIN_STAGE1) and len(r0["keys_compared"]["training"]) == 19
    assert sorted(r0["keys_compared"]["onpolicy"]) == sorted(S.S0_KEYS_ONPOLICY_STAGE1) and len(r0["keys_compared"]["onpolicy"]) == 10
    assert all(S.S0_RANK_ONLY.match(k) for k in r0["p_only_keys"]["training"] + r0["p_only_keys"]["onpolicy"]) and r0["p_only_keys"]["training"]
    assert r0["traj_L1_c"] == r0["traj_L1_p"] == 0.2121780663728714
    assert S.ctrl_s0(str(tmp_path / "c.jsonl"), str(tmp_path / "t.jsonl"), rank_w=0.03125, step=0, check_c_sha=False, pin_keys=True)["pass"]       # the pinned key sets are the measured ones
    r1 = S.ctrl_s0(str(tmp_path / "c.jsonl"), str(tmp_path / "t.jsonl"), rank_w=0.03125, step=1, check_c_sha=False)
    assert not r1["pass"] and set(r1["keys_differing"]["training"]) == {"bce_c", "bce_dac", "bce_ddc", "bce_ep", "bce_nc", "bce_ttc", "pos_L1", "score", "traj_L1", "yaw_L1"}
    assert set(r1["keys_differing"]["onpolicy"]) == {"best", "pick", "random", "skill"}
    if os.path.exists(os.path.join(EVID, "metrics_scorerft_c.jsonl")):                                                  # the banked logs, when this machine holds them
        c, t = os.path.join(EVID, "metrics_scorerft_c.jsonl"), os.path.join(EVID, "metrics_scorerft_t.jsonl")
        assert FR.sha256_file(c) == S.C_METRICS_SHA256 and FR.sha256_file(t) == S.T_METRICS_SHA256
        full = S.ctrl_s0(c, t, rank_w=0.03125)                                                                           # the positive control: T's log in P's place, C's sha enforced
        assert full["pass"] and full["n_keys_compared"] == {"training": 19, "onpolicy": 10}, full["fails"]
        o = S.ctrl_order(c, t)
        assert o["pass"] and o["first_difference"] is None, o["fails"]


def test_ctrl_order_and_RED(tmp_path):
    c, p = mk_metrics(tmp_path / "c.jsonl"), mk_metrics(tmp_path / "p.jsonl", rank=True)
    assert S.ctrl_order(str(c), str(p))["pass"]

    def edit(fn):
        q = tmp_path / "q.jsonl"
        shutil.copyfile(p, q)
        _rewrite(str(q), fn)
        return S.ctrl_order(str(c), str(q))

    def swap(rows):
        i = [k for k, x in enumerate(rows) if x.get("step") == 7 and "event" not in x][0]
        j = [k for k, x in enumerate(rows) if x.get("step") == 8 and "event" not in x][0]
        rows[i]["pos"], rows[j]["pos"] = rows[j]["pos"], rows[i]["pos"]
        return rows
    r = edit(swap)
    assert not r["pass"] and r["first_difference"]["step"] == 7 and "pos" in r["first_difference"]["keys"]            # RED-ORDER
    r = edit(lambda rows: [x for x in rows if not (x.get("step") == 500 and "event" not in x)])
    assert not r["pass"] and any("500" in f for f in r["fails"])                                                         # a missing step
    pd = mk_metrics(tmp_path / "pd.jsonl", rank=True, extra_row_at0=True)
    assert S.ctrl_order(str(c), str(pd))["pass"]                                                                         # a duplicate step from a resume is de-duplicated
    assert not S.ctrl_order(str(c), str(p), steps=999)["pass"]                                                           # rows beyond the expected step count


# ---- CTRL-CODE
def mk_config(arm="P", **over):
    """a config.json with the keys pod_config_c.json carries (device, torch, warm_start, source_sha256, args, identity, meta, argv)."""
    p = arm == "P"
    args = {"lr": 5e-5, "steps": 1000, "seed": 0, "rank_w": 0.25 if p else 0.0, "rank_heads": "pick" if p else "", "rank_ep_gap": 0.02, "rank_gate_topk": 8, "out": "/w/p" if p else "/w/c"}
    d = {"argv": ["/w/data/refe_p5c/code_%s/refe/train.py" % ("aaaaaaaa" if p else "83411c28")] + ["--steps", "1000"], "args": args,
         "identity": {"args": {k: v for k, v in args.items() if k != "out"}, "targets_sha256": "a" * 64, "scorer_sha256": "b" * 64, "calib_sha256": "c" * 64, "frozen_sha256": "d" * 64, "traj_label_sha256": "e" * 64,
                      "n_tuples": 150603, "n_scenes": 92896, "per_rank": {"0": 92896, "1": 57707}},
         "meta": {"frozen_sha256": "d" * 64, "traj_label_table_sha256": "e" * 64, "traj_label_coverage": {"rows": 150603, "covered": 150603}},
         "warm_start": {"sha256": S.C1_SHA256}, "torch": S.POD_TORCH, "device": S.POD_DEVICE, "started_at": "2026-11-22T23:01:48Z",
         "source_sha256": {"train.py": ("f" * 64) if p else S.C_TRAIN_SHA256, "model.py": "1" * 64, "load_dinov3.py": "2" * 64, "ckpt_io.py": "3" * 64}}
    for k, v in over.items():
        if k in ("device", "torch"):
            d[k] = v
        elif k.startswith("src_"):
            d["source_sha256"][k[4:].replace("_py", ".py")] = v
        elif k.startswith("arg_"):
            d["args"][k[4:]] = v
            if k[4:] != "out":
                d["identity"]["args"][k[4:]] = v
        elif k.startswith("id_"):
            d["identity"][k[3:]] = v
        else:
            d[k] = v
    return d


def test_ctrl_code_and_RED(monkeypatch):
    monkeypatch.setattr(S, "PICK_TRAIN_SHA256", "f" * 64)
    ok_ = S.ctrl_code(mk_config("C"), mk_config("P"), check_c_sha=False)
    assert ok_["pass"], ok_["fails"]
    assert set(ok_["detail"]["args_differing"]) == {"rank_w", "rank_heads", "out"}

    def bad(c_over=None, p_over=None, **kw):
        return not S.ctrl_code(mk_config("C", **(c_over or {})), mk_config("P", **(p_over or {})), check_c_sha=False, **kw)["pass"]
    assert bad(p_over={"src_model_py": "9" * 64}) and bad(p_over={"device": "NVIDIA A40"}) and bad(c_over={"torch": "2.8.0"}, p_over={"torch": "2.8.0"})
    assert bad(p_over={"arg_lr": 1e-4}) and bad(p_over={"arg_rank_ep_gap": 0.05}) and bad(p_over={"id_targets_sha256": "0" * 64})
    assert bad(p_over={"src_train_py": S.C_TRAIN_SHA256})                                                              # RED-CODE: P ran the OLD train.py
    assert bad(p_over={"warm_start": {"sha256": "0" * 64}}) and bad(p_over={"src_train_py": "0" * 64})
    assert bad(p_over={"arg_rank_heads": "ep,gate"}) and bad(p_over={"arg_rank_w": 0.0})
    assert bad(script_tags=("code_83411c28", "code_bbbbbbbb"), tag_pins=("code_83411c28", "code_aaaaaaaa")) and not bad(script_tags=("code_83411c28", "code_aaaaaaaa"), tag_pins=("code_83411c28", "code_aaaaaaaa"))
    assert bad(script_tags=("code_deadbeef", "code_aaaaaaaa"), tag_pins=("code_83411c28", None))                      # C's code dir is not the pinned one
    # the seed form: Xs vs X allows {seed, out} ONLY
    seed = dict(allowed_args=("seed", "out"), kinds=("P", "P"), pins=("f" * 64, "f" * 64), label_c="P", label_p="Ps", check_c_sha=False)
    assert S.ctrl_code(mk_config("P"), mk_config("P", arg_seed=1, arg_out="/w/ps"), **seed)["pass"]
    assert not S.ctrl_code(mk_config("P"), mk_config("P", arg_seed=1, arg_out="/w/ps", arg_lr=1e-4), **seed)["pass"]
    monkeypatch.setattr(S, "PICK_TRAIN_SHA256", None)
    assert not S.ctrl_code(mk_config("C"), mk_config("P"), check_c_sha=False)["pass"]                                 # a real read refuses an unfilled pin
    d = S.ctrl_code(mk_config("C"), mk_config("P"), check_c_sha=False, real=False)
    assert d["pass"] and "unfilled" in json.dumps(d["detail"])                                                          # a dry run says so and goes on


# ---- CTRL-HARNESS
def mk_report(**over):
    d = dict(S.HARNESS_KEYS, tokens_asked=12146, rows=12146, misses=0, driving_command_set=12146)
    d.update(over)
    return d


def test_ctrl_harness_and_RED(tmp_path):
    log = tmp_path / "CHAIN.log"
    log.write_text("[t] SFTC chain start tip=751f3ae7 prefix=a\n[t] READ1 start tip=429086ac\n[t] SFTP chain start tip=751f3ae7 prefix=b\n", encoding="utf-8")
    assert S.ctrl_harness(mk_report(), mk_report(), str(log))["pass"]
    assert not S.ctrl_harness(mk_report(), mk_report(rule="navsim_v2"), str(log))["pass"]
    assert not S.ctrl_harness(mk_report(), mk_report(misses=1), str(log))["pass"]
    assert not S.ctrl_harness(mk_report(), mk_report(driving_command_set=0), str(log))["pass"]
    assert not S.ctrl_harness(mk_report(rows=12145), mk_report(), str(log))["pass"]
    log.write_text(log.read_text(encoding="utf-8") + "[t] SFTP chain start tip=0c5edd9e prefix=b\n", encoding="utf-8")                  # the LAST line counts
    r = S.ctrl_harness(mk_report(), mk_report(), str(log))
    assert not r["pass"] and r["tips"]["SFTP"] == "0c5edd9e"
    log.write_text("[t] SFTC chain start tip=751f3ae7\n", encoding="utf-8")
    assert not S.ctrl_harness(mk_report(), mk_report(), str(log))["pass"]                                               # no SFTP line at all


# ---- CTRL-WRULE
def mk_wrule(**over):
    d = {"rule": {"status": "SELECT", "rank_w": 0.25}, "heads": ["pick"], "dead_heads": [], "label_backend": "pdm", "NOT_A_DECISION": False, "init_from": "/w/c1/model_final.pt",
         "r_by_w": {"0.25": 1.4}, "rho_by_w": {"0.25": 0.2}, "pairs_by_head": {"pick": 100}, "r1_per_micro_min_median_max": [1, 2, 3]}
    d.update(over)
    return d


def test_ctrl_wrule_and_RED():
    ap = S.parse_argv(ARGV_PP)
    r = S.ctrl_wrule(mk_wrule(), ap)
    assert r["pass"], r["fails"]
    assert r["r_by_w"] and r["pairs_by_head"] == {"pick": 100}
    for over in ({"rule": {"status": "K2", "rank_w": 0.25}}, {"rule": {"status": "K2b", "rank_w": 0.25}}, {"heads": ["ep", "gate"]}, {"dead_heads": ["pick"]}, {"label_backend": "synthetic"}, {"NOT_A_DECISION": True},
                 {"init_from": "/w/other.pt"}, {"rule": {"status": "SELECT", "rank_w": 0.5}}, {"rule": {"status": "SELECT", "rank_w": 0.25000001}}):
        assert not S.ctrl_wrule(mk_wrule(**over), ap)["pass"], over
    assert not S.ctrl_wrule(mk_wrule(), S.parse_argv([("0.5" if x == "0.25" else x) for x in ARGV_PP]))["pass"]            # argv --rank-w = 2x the rule's value


# ---- the synthetic pick arms and Stage 1
@pytest.fixture(scope="module")
def synth_pick(tmp_path_factory):
    d = tmp_path_factory.mktemp("sftpick")
    rng = np.random.default_rng(7)
    n = 90
    tokens = [f"{i:016x}" for i in range(n)]
    logs = [f"log{i % 9}" for i in range(n)]
    A = {"P": write_arm(d, "P", n, rng, True, logs, tokens), "C": write_arm(d, "C", n, rng, False, logs, tokens)}
    with open(d / "rows.jsonl", "w", encoding="utf-8") as f:
        for i, t in enumerate(tokens):
            arms = {}
            for a, X in A.items():
                p = X["pick"][i]
                arms[a] = {"tracked": {k: float(X["tr"][k][i, p]) for k in ("nc", "dac", "ddc", "ttc", "ep", "c", "pdms")}, "transparent": {k: float(X["tp"][k][i, p]) for k in ("nc", "dac", "ddc", "ttc", "ep", "c", "pdms")}}
            f.write(json.dumps({"token": t, "log": logs[i], "arms": arms}) + "\n")
    return {"dir": d, "tokens": tokens, "logs": logs, "A": A, "n": n}


def nsp(sp, **kw):
    """a namespace for `--card pick` on the synthetic arms P (the treatment) and C."""
    d = sp["dir"]
    base = dict(card="pick", rows=[str(d / "rows.jsonl")], t="P", c="C", ts=None, cs=None, w0_digest_c=None, w0_ref=None, det_seam_t=None, det_full_seam_t=None, seam_ledger=None, convert_report=None,
                fam_ci=None, fam_ci_s=None, fam_labels=None, tactical=None, strategic=None, dry_run="unit test", n_boot=200, out="x", text=None, verdict_line=None)
    for k in ("t", "c", "ts", "cs"):
        for f in ("seam", "dump", "csv", "argv"):
            base[f"{f}_{k}"] = None
        base[f"proptable_{k}"] = None
    for k in S.PICK_FLAGS:
        base[k] = None
    for k, nm_ in (("t", "P"), ("c", "C")):
        base.update({f"seam_{k}": str(d / f"{nm_}.npz"), f"dump_{k}": str(d / f"{nm_}.proposals.npz"), f"csv_{k}": str(d / f"{nm_}.csv"), f"proptable_{k}": [str(d / f"{nm_}_table.jsonl")]})
    base.update(kw)
    return argparse.Namespace(**base)


def sha_of(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def mk_stage1(sp, d):
    """Stage 1 as the PARENT card would have written it: run_read under card scorer_ft on the P files as T and the C files, dry, n_boot 200; its ledger is recorded at first sight."""
    led = d / "ledger_s1.json"
    r = S.run_read(argparse.Namespace(**dict(vars(nsp(sp)), card="scorer_ft", seam_ledger=str(led))))
    f = d / "read_s1.json"
    f.write_text(json.dumps(r, default=float), encoding="utf-8")
    return f, led, r


def pin_c(monkeypatch, sp, d, s1, led, argv_c, w0_c):
    monkeypatch.setattr(S, "STAGE1_READ_SHA256", sha_of(s1))
    monkeypatch.setattr(S, "STAGE1_LEDGER_SHA256", sha_of(led))
    monkeypatch.setattr(S, "C_SEAM_SHA256", sha_of(sp["dir"] / "C.npz"))
    monkeypatch.setattr(S, "C_DUMP_SHA256", sha_of(sp["dir"] / "C.proposals.npz"))
    monkeypatch.setattr(S, "C_CSV_SHA256", sha_of(sp["dir"] / "C.csv"))
    monkeypatch.setattr(S, "C_ARGV_SHA256", sha_of(argv_c))
    monkeypatch.setattr(S, "C_W0_SHA256", sha_of(w0_c))


def mk_probe(state="CLIP-SAME occ=2/8", **over):
    """a banked CTRL-CLIP probe JSON as clip_probe.py writes it (the fields the read consumes)."""
    v = {k: {"id": k, "pass": True, "fails": []} for k in ("V1", "V2", "V3", "V4")}
    cl = lambda args, out: [x if x != "/w/p" and x != "/w/c" else out for x in args]    # noqa: E731
    d = {"instrument": "clip_probe.py", "k": 8, "valid": True, "state": state, "validity": v, "n_on": 0, "n_off": 0, "occ_P": 2, "occ_C": 2, "cmin_P": 0.5, "cmin_C": 0.5, "N0_P": 2.0, "N0_C": 2.0, "pl": None,
         "scope": "steps 0-7 only; P's full-run clip path is UNMEASURED",
         "forms": {"P": {"argv": cl(ARGV_PP, "/scratch/p")}, "C": {"argv": cl(ARGV_C, "/scratch/c")}}}
    d.update(over)
    return d


def pick_inputs(sp, d, monkeypatch=None):
    """every input a pick read takes, synthesised; returns the kwargs for nsp(...) and the paths."""
    d.mkdir(exist_ok=True)
    s1, led, _ = mk_stage1(sp, d)
    argv_c, argv_p = d / "argv_c.json", d / "argv_p.json"
    argv_c.write_text(json.dumps({"argv": ["/w/data/refe_p5c/code_83411c28/refe/train.py"] + ARGV_C}), encoding="utf-8")
    argv_p.write_text(json.dumps({"argv": ["/w/data/refe_p5c/code_aaaaaaaa/refe/train.py"] + ARGV_PP}), encoding="utf-8")
    w0_c, w0_p = d / "w0_c.json", d / "w0_p.json"
    for f_ in (w0_c, w0_p):
        f_.write_text(json.dumps({"step0_digest": S.C1_SHA256}), encoding="utf-8")
    mk_metrics(d / "m_c.jsonl")
    mk_metrics(d / "m_p.jsonl", rank=True)
    (d / "cfg_c.json").write_text(json.dumps(mk_config("C")), encoding="utf-8")
    (d / "cfg_p.json").write_text(json.dumps(mk_config("P")), encoding="utf-8")
    for nm_ in ("c", "p"):
        (d / f"rep_{nm_}.json").write_text(json.dumps(mk_report(tokens_asked=sp["n"], rows=sp["n"], driving_command_set=sp["n"])), encoding="utf-8")
    (d / "CHAIN.log").write_text("[t] SFTC chain start tip=751f3ae7\n[t] SFTP chain start tip=751f3ae7\n", encoding="utf-8")
    (d / "wrule.json").write_text(json.dumps(mk_wrule()), encoding="utf-8")
    (d / "clip_probe.json").write_text(json.dumps(mk_probe()), encoding="utf-8")
    if monkeypatch is not None:
        monkeypatch.setattr(S, "CLIP_PROBE_SHA256", sha_of(d / "clip_probe.json"))
        pin_c(monkeypatch, sp, d, s1, led, argv_c, w0_c)
        monkeypatch.setattr(S, "PICK_TRAIN_SHA256", "f" * 64)
    return dict(argv_c=str(argv_c), argv_t=str(argv_p), w0_digest_c=str(w0_c), w0_digest_p=str(w0_p), w0_ref=S.C1_SHA256, metrics_c=str(d / "m_c.jsonl"), metrics_p=str(d / "m_p.jsonl"),
                pod_config_c=str(d / "cfg_c.json"), pod_config_p=str(d / "cfg_p.json"), seam_report_c=str(d / "rep_c.json"), seam_report_p=str(d / "rep_p.json"), seam_chain_log=str(d / "CHAIN.log"),
                weight_rule_p=str(d / "wrule.json"), clip_probe=str(d / "clip_probe.json"), stage1_read=str(s1), stage1_ledger=str(led), seam_ledger=str(d / "ledger_p.json"))


# ---- CTRL-CREUSE
def test_ctrl_creuse_and_RED(synth_pick, tmp_path, monkeypatch):
    kw = pick_inputs(synth_pick, tmp_path / "in", monkeypatch)
    r = S.run_read(nsp(synth_pick, **kw))
    cr = [c for c in r["controls"] if c["control"] == "CTRL-CREUSE"]
    assert len(cr) == 1 and cr[0]["pass"], cr
    assert cr[0]["max_abs_delta"]["levels"] <= S.CREUSE_BAR and cr[0]["n_fields_compared"]["levels"] == 7 * 3 * 3 and cr[0]["n_fields_compared"]["selection_levels"] > 0
    # --- direct RED arms on ctrl_creuse
    sp = synth_pick
    tokens = sp["tokens"]
    geo = S.FR.seam_geometry(str(sp["dir"] / "C.npz"), str(sp["dir"] / "C.proposals.npz"), tokens)
    P_ = PR.per_token(FR.load_slim([str(sp["dir"] / "rows.jsonl")], ["P", "C"])[0], ["P", "C"])
    bt = PR.Boot(P_["logs"], 200)
    levels_c = {m: PR.level(bt, P_, "C", m) for m in METRICS}
    MAc = S.mech_arrays(S.load_table_arrays([str(sp["dir"] / "C_table.jsonl")], tokens), S.load_dump(str(sp["dir"] / "C.proposals.npz"), tokens))
    sel = S.mech_levels(bt, MAc)
    args = (kw["stage1_read"], kw["stage1_ledger"], geo, S.C_CSV_SHA256, kw["argv_c"], kw["w0_digest_c"], levels_c, sel, 200)
    assert S.ctrl_creuse(*args)["pass"]
    lv = json.loads(json.dumps(levels_c, default=float))
    lv["pdms"]["tracked"]["point"] += 1e-6                                                                        # RED-CREUSE: one C level moved by 1e-6
    bad = S.ctrl_creuse(args[0], args[1], geo, args[3], args[4], args[5], lv, sel, 200)
    assert not bad["pass"] and bad["first_field_over_bar"] == "levels.C.pdms.tracked.point"
    sl = json.loads(json.dumps(sel, default=float))
    sl["regret"]["point"] += 1e-6
    assert not S.ctrl_creuse(args[0], args[1], geo, args[3], args[4], args[5], levels_c, sl, 200)["pass"]            # a selection level moved
    led2 = tmp_path / "led2.json"                                                                                   # a ledger whose arms.C.seam_sha256 is changed
    j = json.load(open(kw["stage1_ledger"], encoding="utf-8"))
    j["arms"]["C"]["seam_sha256"] = "0" * 64
    led2.write_text(json.dumps(j), encoding="utf-8")
    assert not S.ctrl_creuse(args[0], str(led2), geo, *args[3:])["pass"]
    s2 = tmp_path / "s2.json"                                                                                       # a stage-1 file edited by one byte: the sha pin
    s2.write_text(open(kw["stage1_read"], encoding="utf-8").read() + " ", encoding="utf-8")
    assert not S.ctrl_creuse(str(s2), args[1], geo, *args[3:])["pass"]
    assert not S.ctrl_creuse(args[0], args[1], geo, "0" * 64, *args[4:])["pass"]                                      # a C CSV sha != pin
    assert not S.ctrl_creuse(*args[:8], 300)["pass"]                                                                # another draw count
    # a dry read with another n_boot: NOT AVAILABLE, not a pass and not a fail
    r2 = S.run_read(nsp(synth_pick, **dict(kw, n_boot=300)))
    assert "CTRL-CREUSE" in [u["control"] for u in r2["controls_not_available"]] and "CTRL-CREUSE" not in [c["control"] for c in r2["controls"]]


# ---- labels in every output
def test_pick_labels_in_every_output(synth_pick, tmp_path):
    r = S.run_read(nsp(synth_pick))
    blob = json.dumps(r, default=float)
    assert "P_minus_C" in r and "P" in r["levels"] and "P" in r["goal"] and "P" in r["families"]["TACTICAL"]["(b)_selection_quality"]["levels"] and "P_minus_C" in r["families"]["TACTICAL"]["(b)_selection_quality"]
    assert "T_minus_C" not in blob and '"T"' not in blob and "Ts_minus_Cs" not in blob and '"Ts"' not in blob, [m for m in ('"T"', "T_minus_C", '"Ts"') if m in blob]
    assert r["card_key"] == "H-SCORERFT-PICK-1" and r["card"] == "PREREG_SCORERFT_PICK.md" and set(r["arms"]) == {"P", "C"}
    line, txt = S.verdict_line(r), S.text_report(r)
    assert "P - C" in txt and "H-SCORERFT-PICK-1" in line and "shared control" in line and " T - C" not in txt and "T_minus_C" not in txt
    assert line.startswith(S.DRY_STAMP) and "M3_moved=" in line and "Zup=" in line and "M1(reported)=" in line and "M1_moved" not in line


def test_relabel_tactical_to_P():
    t = {"kind": "P5C_TACTICAL", "strategic": {"status": "REPORTED_SEPARATELY"}, "command_adherence": {"arms": {"C1": {}, "C0": {}, "base": {}}, "paired_C1_minus_x": {}, "C1-C0": 1}}
    r = S.relabel_tactical(t, {"C1": "P", "C0": "C"}, "P")
    s = json.dumps(r)
    assert '"strategic"' not in s and '"C1"' not in s and '"P"' in s and '"C"' in s and "paired_P_minus_x" in s and "P-C" in s and r["pooled"] is False and "C1 -> P" in r["labelling"]
    assert '"T"' not in s


# ---- refusals
def test_pick_real_read_refuses_without_its_inputs(synth_pick, monkeypatch):
    monkeypatch.setattr(S, "PICK_PREREG_SHA256", "1" * 64)
    monkeypatch.setattr(S, "PICK_TRAIN_SHA256", "2" * 64)
    with pytest.raises(PR.Refused) as e:
        S.run_read(nsp(synth_pick, dry_run=None))
    msg = str(e.value)
    for need in ("--w0-digest-p", "--pod-config-c", "--pod-config-p", "--seam-report-c", "--seam-report-p", "--seam-chain-log", "--weight-rule-p", "--stage1-read", "--stage1-ledger", "--metrics-c", "--metrics-p", "--clip-probe",
                 "--argv-p", "--argv-c", "--w0-digest-c", "--w0-ref", "--det-seam-p", "--seam-ledger", "--fam-ci", "--strategic", "--tactical"):
        assert need in msg, need
    assert "PICK_PREREG_SHA256" not in msg and "--argv-t" not in msg and "--det-seam-t" not in msg
    with pytest.raises(PR.Refused) as e2:
        S.run_read(nsp(synth_pick, dry_run=None, ts="P", cs="C"))
    for need in ("--pod-config-cs", "--pod-config-ps", "--metrics-cs", "--metrics-ps", "--fam-ci-s"):
        assert need in str(e2.value), need


def test_pick_unfrozen_pins_refuse_a_real_read(synth_pick, tmp_path, monkeypatch):
    for pin in ("PICK_PREREG_SHA256", "PICK_TRAIN_SHA256"):
        monkeypatch.setattr(S, "PICK_PREREG_SHA256", "1" * 64)
        monkeypatch.setattr(S, "PICK_TRAIN_SHA256", "2" * 64)
        monkeypatch.setattr(S, pin, None)
        with pytest.raises(PR.Refused) as e:
            S.run_read(nsp(synth_pick, dry_run=None))
        assert pin in str(e.value)
    assert S.run_read(nsp(synth_pick))["STAMP"] == S.DRY_STAMP                                                        # a dry run with unfilled pins is allowed and stamped
    # a --metrics-c whose sha != C_METRICS_SHA256 fails CTRL-S0 in a real read
    c, p = mk_metrics(tmp_path / "c.jsonl"), mk_metrics(tmp_path / "p.jsonl", rank=True)
    r = S.ctrl_s0(str(c), str(p), rank_w=0.25, check_c_sha=True)
    assert not r["pass"] and any("sha256" in f for f in r["fails"])


def test_RED_SWAP_P_names_refused(synth_pick):
    with pytest.raises(PR.Refused, match="RED-SWAP-P"):
        S.run_read(nsp(synth_pick, t="C", c="P"))
    with pytest.raises(PR.Refused, match="RED-SWAP-P"):
        S.run_read(nsp(synth_pick, t="C", c="P", dry_run=None))


# ---- CTRL-ID and the self-read
def test_pick_ctrl_id_and_self_read(synth_pick):
    d = synth_pick["dir"]
    sp = synth_pick
    Tab = S.load_table_arrays([str(d / "P_table.jsonl")], sp["tokens"])
    MA = S.mech_arrays(Tab, S.load_dump(str(d / "P.proposals.npz"), sp["tokens"]))
    P_ = PR.per_token(FR.load_slim([str(d / "rows.jsonl")], ["P"])[0], ["P"])
    r = S.ctrl_id(P_, P_["logs"], MA, "P", 200, card="pick")
    assert r["pass"] and r["nonzero_deltas"] == [] and r["P1"] is False and r["verdict"] == "OP7"
    bt = PR.Boot(P_["logs"], 200)
    mc = S.mech_contrast(bt, MA, MA)
    assert all(mc[k]["point"] == 0.0 and mc[k]["lo"] == 0.0 and mc[k]["hi"] == 0.0 for k in ("regret", "oracle", "zeroed"))
    assert S.decide_pick(PR.contrast(bt, P_, "P", "P"), S.mech_decide_pick(mc), True)["outcome"] == "OP7"          # CI-hi = 0.0 is not < 0: a self-read cannot reach OP6
    r2 = S.run_read(nsp(sp, c="P", seam_c=str(d / "P.npz"), dump_c=str(d / "P.proposals.npz"), csv_c=str(d / "P.csv"), proptable_c=[str(d / "P_table.jsonl")]))
    assert r2["self_read"] and r2["verdict"]["clauses"]["P1_lo_gt_0"] is False
    assert not S.FR.zero_exact({m: {s: r2["P_minus_C"][m][s] for s in ("tracked", "transparent", "TD")} for m in METRICS})


# ---- the end-to-end dry read with every new input
def test_end_to_end_pick_dry_read(synth_pick, tmp_path, monkeypatch):
    kw = pick_inputs(synth_pick, tmp_path / "e2e", monkeypatch)
    r = S.run_read(nsp(synth_pick, **kw))
    names = [c["control"] for c in r["controls"]]
    assert {"CTRL-SET", "CTRL-T", "CTRL-CSV", "CTRL-SEAM", "CTRL-ARGV", "CTRL-W0", "CTRL-S0", "CTRL-ORDER", "CTRL-CODE", "CTRL-HARNESS", "CTRL-WRULE", "CTRL-CREUSE", "CTRL-CLIP", "CTRL-ID"} <= set(names), names
    assert all(c["pass"] for c in r["controls"]), [(c["control"], c.get("fails")) for c in r["controls"] if not c["pass"]]
    assert {u["control"] for u in r["controls_not_available"]} == {"CTRL-DET"}                                       # 90 tokens cannot hold the 200-token re-run
    assert r["STAMP"] == S.DRY_STAMP and r["verdict"]["verdict"].startswith("DRY RUN:") and r["stage"] == 1
    f = r["families"]
    assert f["STRATEGIC"]["status"] == "NOT SUPPLIED" and "PAUSED" not in json.dumps(f)
    assert f["TACTICAL"]["(c)_goal_setting"]["status"] == "NOT COMPUTABLE" and f["TACTICAL"]["(c)_goal_setting"]["n"] == 90
    m = r["mechanism"]
    assert m["M1_ep_spearman"]["decisive"] is False and m["mechanism_moved"] == m["m3_moved"] and "M5_train_monitor" in m and m["M5_train_monitor"]["status"].startswith("REPORTED")
    cc = next(c for c in r["controls"] if c["control"] == "CTRL-CLIP")
    assert cc["state"] == "CLIP-SAME occ=2/8" and "clip: CLIP-SAME occ=2/8" in S.verdict_line(r) and "CTRL-CLIP state" in S.text_report(r)
    cd = next(c for c in r["controls"] if c["control"] == "CTRL-CODE")
    assert cd["detail"]["code_tag"] == {"C": "code_83411c28", "P": "code_aaaaaaaa"}
    ar = next(c for c in r["controls"] if c["control"] == "CTRL-ARGV")
    assert ar["detail"]["P_minus_C_flags"] == sorted(S.ALLOWED_TC_PICK)
    # every control failure turns the read INVALID: P's argv with one extra flag
    argv_bad = tmp_path / "e2e" / "argv_p_bad.json"
    argv_bad.write_text(json.dumps({"argv": ARGV_PP + ["--w-field", "2.0"]}), encoding="utf-8")
    rb = S.run_read(nsp(synth_pick, **dict(kw, argv_t=str(argv_bad))))
    assert rb["controls_all_pass"] is False and rb["verdict"]["verdict"].startswith("DRY RUN:INVALID")


def test_pick_constants_are_the_cards():
    assert S.C_METRICS_SHA256 == "97a8631cc863d95cf646932ef3833206affbc7a0857d21bff5d5e8b78ff04eed" and S.T_METRICS_SHA256 == "09cdaf83c611fba9af7bafef250affc7195af71aa9c40bfcf461888d712b0f6a"
    assert S.C_TRAIN_SHA256 == "59a737f43947175001202d0114e3d4c83f8a30c984b171c3d284ca1b4684a755" and S.C_CODE_TAG == "code_83411c28"
    assert S.C_POD_CONFIG_SHA256 == "a66f6750416207cbfefa5a58a51088f5e44068b4d212bc5f5d38b9a8b78c5491" and S.C_ARGV_SHA256 == "00729d56aa734e370d394385b516d9b8abae94bfe1e3f1df304e323772069e03"
    assert S.C_W0_SHA256 == "2dd2dcd25cb582715d1f8846bc352f4c2e767afc061acd36b4a80415c54351ec" and S.C_SEAM_SHA256 == "914bdd0aab4459a4ebb61d9fb0d82c537c3b1ec86ebc427882066ddee279b94f"
    assert S.C_DUMP_SHA256 == "848e6561987b2971bd63f4cdcf9f17c3e4ec76d3aa19da405f7d7d5788492e56" and S.C_CSV_SHA256 == "2df7f856b46b9dc7f3540c056b15ffd2fc42a7ba39c94e0aad37d4f907f3c81f"
    assert S.STAGE1_READ_SHA256 == "d301d38c6c358a4220a0cdb4cd7641f86c9c4a01d123b546050305e7fda32d41" and S.STAGE1_LEDGER_SHA256 == "e627df2af876624588c93e9759210d7c52063e155d4231d87412a7dcf6e5f9b2"
    assert S.C1_SHA256 == "f09949e2a5f2f9661ad47457e77ddfd10f49117d64e095487809c1d04f7a34a4" and (S.POD_DEVICE, S.POD_TORCH, S.HARNESS_TIP) == ("NVIDIA L40", "2.7.1+cu128", "751f3ae7")
    assert S.HARNESS_KEYS == {"select": "best", "rule": "navsim_v1", "repair_last_heading": True, "sanitize_goal": False, "goal_fix": "pdm_route", "nav_mode": "command"} and S.CREUSE_BAR == 1e-9
    assert S.ALLOWED_TC_PICK == {"--rank-w", "--rank-heads", "--out"} and S.ORDER_KEYS == ("epoch", "pos", "samples", "batch", "lr") and S.S0_EXCLUDE == {"elapsed_s", "cuda_max_mem_gb", "at"}
    assert len(S.S0_KEYS_TRAIN_STAGE1) == 19 and len(S.S0_KEYS_ONPOLICY_STAGE1) == 10 and list(S.S0_KEYS_TRAIN_STAGE1) == sorted(S0_KEYS)
    # the pins the MM fills at freeze / land / launch (a real read refuses while they are None); the MM edits this test to the filled values, as test_prereg_constants_are_the_cards does for the parent
    assert S.PICK_PREREG_SHA256 is None or len(S.PICK_PREREG_SHA256) == 64
    assert S.PICK_TRAIN_SHA256 is None or len(S.PICK_TRAIN_SHA256) == 64
    assert S.PICK_CODE_TAG is None or S.PICK_CODE_TAG.startswith("code_")


# ---- CTRL-CLIP, read form (card 5.1; the amended card, not READ_TOOL_SPEC.md)
def test_ctrl_clip_and_RED(tmp_path, monkeypatch):
    ap, ac = S.parse_argv(ARGV_PP), S.parse_argv(ARGV_C)
    r = S.ctrl_clip(mk_probe(), ap, ac, real=False)
    assert r["pass"], r["fails"]
    assert r["state"] == "CLIP-SAME occ=2/8" and r["pl"] is None and "UNMEASURED" in r["scope"]
    t = S.ctrl_clip(mk_probe(state="CLIP-T on=2 off=0 occ_P=3/8 occ_C=1/8 cmin_P=0.5 cmin_C=1", n_on=2, pl="PL4"), ap, ac, real=False)
    assert t["pass"] and t["state"].startswith("CLIP-T on=2") and t["n_on"] == 2                              # CLIP-T does not veto: the read carries the string
    z = S.ctrl_clip(mk_probe(state="CLIP-0"), ap, ac, real=False)
    assert z["pass"] and z["state"] == "CLIP-0"
    bad = lambda **o: not S.ctrl_clip(mk_probe(**o), ap, ac, real=False)["pass"]                             # noqa: E731
    v = mk_probe()
    for k in ("V1", "V2", "V3", "V4"):
        vv = json.loads(json.dumps(v["validity"]))
        vv[k]["pass"] = False
        assert bad(validity=vv), k
    assert bad(valid=False) and bad(state="CLIP-INVALID") and bad(k=7)
    assert not S.ctrl_clip(mk_probe(), S.parse_argv(ARGV_PP + ["--w-field", "2.0"]), ac, real=False)["pass"]              # the probe's P-form argv is not P's run argv
    assert not S.ctrl_clip(mk_probe(), ap, S.parse_argv(ARGV_C + ["--seed", "1"]), real=False)["pass"]                    # ... nor C's
    alt = mk_probe()
    alt["forms"]["P"]["argv"] = [("/elsewhere" if x == "/w/p" else x) for x in ARGV_PP]
    assert S.ctrl_clip(alt, ap, ac, real=False)["pass"]                                                                     # --out is the one flag that may differ
    f = tmp_path / "probe.json"
    f.write_text(json.dumps(mk_probe()), encoding="utf-8")
    monkeypatch.setattr(S, "CLIP_PROBE_SHA256", None)
    assert not S.ctrl_clip(str(f), ap, ac, real=True)["pass"]                                                               # an unfilled pin refuses a real read
    monkeypatch.setattr(S, "CLIP_PROBE_SHA256", sha_of(f))
    assert S.ctrl_clip(str(f), ap, ac, real=True)["pass"]
    f.write_text(json.dumps(mk_probe()) + " ", encoding="utf-8")                                                            # the file edited by one byte
    assert not S.ctrl_clip(str(f), ap, ac, real=True)["pass"]


def test_pick_read_without_a_clip_probe_is_not_a_verdict(synth_pick, tmp_path, monkeypatch):
    kw = pick_inputs(synth_pick, tmp_path / "noclip", monkeypatch)
    kw.pop("clip_probe")
    r = S.run_read(nsp(synth_pick, **kw))
    assert "CTRL-CLIP" in [u["control"] for u in r["controls_not_available"]] and r["controls_all_pass"] is True       # dry: reported as NOT AVAILABLE
    monkeypatch.setattr(S, "PICK_PREREG_SHA256", "1" * 64)
    with pytest.raises(PR.Refused, match="--clip-probe"):
        S.run_read(nsp(synth_pick, dry_run=None, **kw))                                                                      # real: refused, named


def test_clip_t_is_carried_into_the_verdict_notes(synth_pick, tmp_path, monkeypatch):
    kw = pick_inputs(synth_pick, tmp_path / "clipt", monkeypatch)
    pj = tmp_path / "clipt" / "clip_probe_t.json"
    pj.write_text(json.dumps(mk_probe(state="CLIP-T on=1 off=0 occ_P=2/8 occ_C=1/8 cmin_P=0.4 cmin_C=0.6", n_on=1, pl="PL4")), encoding="utf-8")
    monkeypatch.setattr(S, "CLIP_PROBE_SHA256", sha_of(pj))
    r = S.run_read(nsp(synth_pick, **dict(kw, clip_probe=str(pj))))
    assert "clip: CLIP-T on=1" in S.verdict_line(r)
    if r["verdict"]["outcome"] in ("OP4", "OP5", "OP6", "OP7"):
        assert "UNEXCLUDED CO-MEDIATOR" in r["verdict"]["wording"]
