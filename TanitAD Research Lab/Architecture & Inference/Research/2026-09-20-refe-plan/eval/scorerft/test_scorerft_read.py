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
