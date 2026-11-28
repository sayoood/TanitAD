#!/usr/bin/env python3
"""Tests of `amend1_concordance.py` (L2 Step B, AMENDMENT 1).  PLANTED worlds only, CPU only: NO real Step B output, label or verdict is read, no checkpoint is needed (a tiny set-conditional fake scorer stands in
for the model-identity / re-score path).  Run (never on the real run dir):

    CUDA_VISIBLE_DEVICES=-1 python -m pytest test_amend1.py -q -p no:cacheprovider --basetemp=C:/Users/Admin/par18/l2stepb/test_amend1/tmp

(a) a planted run dir with known pairs reproduces the HAND-computed A3 / A4 / A5 (a literal table, derived on paper from the amendment's text), plus an independent pure-python reference + bootstrap oracle
(b) an output identical to the pick contributes to no pair (with a decoy that WOULD change the numbers if included, and a control row that does)
(c) RED: corrupt one banked pick logit -> the identity refuses (NOT COMPUTABLE), with a mutation arm proving the identity is what refuses
(d) the bootstrap is deterministic under its seed (and a different seed changes it)
(e) the script refuses if PREREG_L2_STEPB_AMENDMENT1.sha256 does not match the file
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.dont_write_bytecode = True
import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import amend1_concordance as AM  # noqa: E402
import fake_scorer  # noqa: E402
import l2_scorer as L2S  # noqa: E402
import stepB  # noqa: E402

A, T = stepB.A, stepB.T
FAKE = fake_scorer.FakeScorerModule()
SEEDS = np.asarray(T.SEARCH_SEEDS, np.int64)
H64 = lambda c: c * 64                                                                    # a planted 64-hex "sha256"

try:
    import torch
    torch.set_num_threads(2)
except Exception:  # pragma: no cover
    pass


# ------------------------------------------------------------------------------------------------------------------------------ planting helpers
def sha_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def mk_bundle(n, logs, K=4, pick=None, seed=3, traj=False):
    rng = np.random.default_rng(seed)
    pick = np.arange(n) % K if pick is None else np.asarray(pick, np.int64)
    sub = np.clip(rng.uniform(0.0, 1.0, (n, K, 6)), 0, 1).round(2)
    pdms = np.clip(rng.uniform(0.0, 1.0, (n, K)), 0, 1).round(2)
    for t in range(n):                                                                     # the pick's BANKED label: all sub-scores 1, PDMS 0.8 (the literal table's anchor)
        sub[t, pick[t]] = 1.0
        pdms[t, pick[t]] = 0.8
    b = {"token": np.array([f"t{i}" for i in range(n)]), "log": np.asarray(logs), "pick": pick.astype(np.int64), "sub": sub, "pdms": pdms, "valid": np.ones((n, K), bool), "logits": np.zeros((n, K, 6))}
    if traj:
        b["traj_raw"] = _plans(rng, n)
    return b


def _plans(rng, n_tok, K=64):
    out = np.zeros((n_tok, K, 20, 3), np.float32)
    for t in range(n_tok):
        for k in range(K):
            v, kappa = rng.uniform(6.0, 14.0), rng.normal(0, 0.01)
            s = v * 0.2 * np.arange(1, 21)
            yaw = kappa * s
            out[t, k, :, 0], out[t, k, :, 1], out[t, k, :, 2] = s * np.cos(yaw / 2), s * np.sin(yaw / 2), yaw
            out[t, k, -1, 2] += 0.3
    return out


def table_arm(bundle, table, S=3, pick_logits=None, ep_pick=None):
    """a banked arm from a literal table {(token, seed): (shift, pdms, {col: value})}: every other (token, seed) is UNMOVED (its output IS the pick: same logits, the pick's banked label).  The anchor's logits are all
    zeros (aggregate 0.125); a candidate's logits are `shift` in all six columns (aggregate sigmoid(shift)^3)."""
    n = len(bundle["token"])
    pl = np.zeros((n, 6)) if pick_logits is None else np.array(pick_logits, np.float64)
    if ep_pick is not None:
        pl[:, L2S.EP] = ep_pick
    out = np.repeat(pl[:, None, :], S, axis=1).copy()
    moved = np.zeros((n, S), bool)
    pick = bundle["pick"]
    lab_sub = np.repeat(bundle["sub"][np.arange(n), pick][:, None, :], S, axis=1).copy()
    lab_pdms = np.repeat(bundle["pdms"][np.arange(n), pick][:, None], S, axis=1).copy()
    for (i, s), (shift, pd, over) in table.items():
        moved[i, s] = True
        out[i, s] = shift
        lab_pdms[i, s] = pd
        for c, v in (over or {}).items():
            lab_sub[i, s, c] = v
    return {"pick_logits": pl, "out_logits": out, "moved": moved, "lab_sub": lab_sub, "lab_pdms": lab_pdms}


def write_world(root, bundle, arm_specs, *, done="ZZSTEPB_DONE NOT_CLEARED", pins_over=None, meta_over=None, exec_fn=None, extra_so=None):
    """a planted Step B run dir + bundle + sealed pkg.  arm_specs[name] = dict(pick_logits, out_logits, moved, lab_sub, lab_pdms) (+ optional scorer_in / exec / theta / pick_exec).  Returns the paths."""
    root = str(root)
    run, pkg = os.path.join(root, "run"), os.path.join(root, "pkg")
    os.makedirs(run, exist_ok=True)
    os.makedirs(pkg, exist_ok=True)
    bp = os.path.join(root, "bundle.npz")
    np.savez(bp, **bundle)
    n = len(bundle["token"])
    carve = stepB.tokens_sha256(bundle["token"])
    pins = {"carve_sha256": carve, "head_s0_sha256": H64("a"), "head_s1_sha256": H64("b"), "ckpt_sha256": H64("c"), "config_sha256": H64("d"), **(pins_over or {})}
    heads = {"L2s0": pins["head_s0_sha256"], "L2s1": pins["head_s1_sha256"], "P2HEAD": None}
    rng = np.random.default_rng(99)
    pick_exec = rng.uniform(-5, 5, (n, 20, 3)).astype(np.float32)
    for name, kind, s in AM.ARMS:
        sp = arm_specs[name]
        moved = np.asarray(sp["moved"], bool)
        S = moved.shape[1]
        pe = np.asarray(sp.get("pick_exec", pick_exec), np.float32)
        ex = np.repeat(pe[:, None], S, axis=1).copy()
        ex[moved] += 0.25
        if "exec" in sp:
            ex = np.asarray(sp["exec"], np.float32)
        meta = {"arm": kind, "head_sha256": heads[name], "ckpt_sha256": pins["ckpt_sha256"], "config_sha256": pins["config_sha256"], "bundle_sha256": sha_file(bp), "data_clause": {"carve_sha256": carve},
                "n_tokens": n, **((meta_over or {}).get(name, {}))}
        so = {"token": bundle["token"], "bundle_index": np.arange(n, dtype=np.int64), "pick": bundle["pick"], "seeds": SEEDS, "exec": ex, "pick_exec": pe, "moved": moved,
              "scorer_in": np.asarray(sp.get("scorer_in", np.zeros((n, S, 20, 3), np.float32))), "theta": np.zeros((n, S, 2)), "pred_logit_pick": np.asarray(sp["pick_logits"], np.float64),
              **({"pred_logit_out": np.asarray(sp["out_logits"], np.float64)} if sp.get("out_logits") is not None else {}), "meta": np.array(json.dumps(meta)), **(extra_so or {})}
        spth = os.path.join(run, f"search_{name}.npz")
        np.savez(spth, **so)
        np.savez(os.path.join(run, f"labels_{name}.npz"), sub=np.asarray(sp["lab_sub"], np.float64), pdms=np.asarray(sp["lab_pdms"], np.float64), token=bundle["token"],
                 bundle_index=np.arange(n, dtype=np.int64), search_sha256=np.array(sha_file(spth)), identity=np.array(json.dumps({"passed": True, "control_ok": True, "max_abs_diff": 0.0})))
    with open(os.path.join(run, "chain_stepB.log"), "w", encoding="utf-8") as f:
        f.write("ZZSTEPB_START pid=1\nZZSTEPB_SEAL_OK x\n" + (done + "\n" if done else ""))
    main_text = "# planted main prereg\nStatus: SEALED\n<!-- PINS-BEGIN -->\n```\n" + "\n".join(f"{k} {v}" for k, v in pins.items()) + "\n```\n<!-- PINS-END -->\n"
    am_text = "# planted amendment 1\nREPORTED ONLY.\n"
    for md, sh, text in ((AM.MAIN_MD, AM.MAIN_SHA, main_text), (AM.AMEND_MD, AM.AMEND_SHA, am_text)):
        with open(os.path.join(pkg, md), "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        with open(os.path.join(pkg, sh), "w", encoding="utf-8", newline="\n") as f:
            f.write(hashlib.sha256(text.encode()).hexdigest() + f" *{md}\n")
    return {"root": root, "run": run, "pkg": pkg, "bundle": bp, "out": os.path.join(root, "out"), "pins": pins}


def rebind_labels(w, name):
    """after editing a search file: re-bind its labels' search_sha256 (so the label-binding check is not what fires)"""
    lp = os.path.join(w["run"], f"labels_{name}.npz")
    z = {k: v for k, v in np.load(lp, allow_pickle=False).items()}
    z["search_sha256"] = np.array(sha_file(os.path.join(w["run"], f"search_{name}.npz")))
    np.savez(lp, **z)


def edit_search(w, name, **changes):
    sp = os.path.join(w["run"], f"search_{name}.npz")
    z = {k: v for k, v in np.load(sp, allow_pickle=False).items()}
    z.update(changes)
    np.savez(sp, **z)
    rebind_labels(w, name)


def run_compute(w, **kw):
    kw.setdefault("expect_logs", None)
    kw.setdefault("planted_carve", True)
    return AM.compute(w["run"], w["bundle"], pkg=w["pkg"], **kw)


def cli(w, monkeypatch, *extra, expect_logs=2):
    """the CLI on a planted world: the sealed carve constant is the ONLY thing patched (a planted bundle cannot be the real carve); no CLI flag reaches it"""
    monkeypatch.setattr(stepB, "SEALED_CARVE_SHA256", w["pins"]["carve_sha256"])
    return AM.main(["--run-dir", w["run"], "--bundle", w["bundle"], "--out-dir", w["out"], "--pkg", w["pkg"], "--expect-logs", str(expect_logs), *extra])


# ------------------------------------------------------------------------------------------------------------------------------ the HAND-COMPUTED literal world (W1)
# anchor: logits 0 in every column (aggregate .125), banked label nc = dac = ... = 1, PDMS 0.8.  Candidate: logits = shift in every column (aggregate sigmoid(shift)^3: +2 -> .683, +1 -> .391, 0 -> .125, -1 -> .0195,
# -2 -> .0017), so a rise of >= 0.02 happens for shift +2 / +1 only, and a tie only for shift 0.
L2S0_TABLE = {
    (0, 0): (2, 0.90, None),          # rise, dPDMS +0.10: A3 concordant
    (0, 1): (2, 0.60, {0: 0.0}),      # rise, dPDMS -0.20 (fall), NC 1->0 (flip): A3 DIScordant, A4 hit, A5 hit
    (0, 2): (-2, 0.60, {0: 0.0}),     # aggregate falls, dPDMS -0.20: A3 concordant; NC 1->0 but NO rise, so A5 does NOT count it (decoy)
    (1, 0): (0, 0.90, None),          # scorer TIE, dPDMS +0.10: A3 counts 0.5
    (1, 1): (2, 0.82, None),          # rise, dPDMS +0.02: |dPDMS| < 0.05 -> not an A3 pair; a rise without fall / flip
    # (1, 2) UNMOVED: the output IS the pick -> no pair
    (2, 0): (2, 0.69, {1: 0.0}),      # rise, dPDMS -0.11 (fall), DAC 1->0 (flip): A3 discordant, A4 hit, A5 hit
    (2, 1): (1, 0.78, None),          # rise, dPDMS -0.02: not an A3 pair, no fall, no flip
    (2, 2): (-1, 0.95, None),         # aggregate falls, dPDMS +0.15: A3 discordant
    (3, 0): (2, 0.86, None),          # rise, dPDMS +0.06: A3 concordant
    (3, 1): (2, 0.40, {2: 0.0}),      # rise, dPDMS -0.40 (fall), EP 1->0 only (NOT NC / DAC): A3 discordant, A4 hit, no A5
    (3, 2): (-2, 0.40, None),         # aggregate falls, dPDMS -0.40: A3 concordant
}
# HAND RESULTS (pooled): A3 pairs = t0:3 + t1:1 + t2:2 + t3:3 = 9, concordant weight 1+0+1 + .5 + 0+0 + 1+0+1 = 4.5 -> 0.5 ; rise pairs = 7 (t0s0 t0s1 t1s1 t2s0 t2s1 t3s0 t3s1), falls 3 -> 3/7 ; flips 2 -> 2/7
HAND_L2S0 = {"A3": (4.5, 9), "A4": (3, 7), "A5": (2, 7)}
# per seed: A3 s0 = (1+.5+0+1)/4 = .625 ; s1 = 0/2 = 0 ; s2 = (1+0+1)/3 = 2/3.  A4 s0 = 1/3, s1 = 2/4, s2 = undefined (no rise).  A5 s0 = 1/3, s1 = 1/4, s2 = undefined
HAND_L2S0_SEED = {"A3": [(2.5, 4), (0.0, 2), (2.0, 3)], "A4": [(1, 3), (2, 4), (0, 0)], "A5": [(1, 3), (1, 4), (0, 0)]}
# L2-s1: every (token, seed) is a rise, a fall (-0.2) and an NC flip -> A3 0/12, A4 12/12, A5 12/12
L2S1_TABLE = {(i, s): (2, 0.60, {0: 0.0}) for i in range(4) for s in range(3)}
# P2HEAD: two moved outputs: t0s0 (rise, +0.10: concordant) and t3s1 (rise, -0.40: discordant fall): A3 1/2, A4 1/2, A5 0/2
P2_TABLE = {(0, 0): (2, 0.90, None), (3, 1): (2, 0.40, None)}


def w1_world(root, l2s0=None, p2=None, l2s1=None, **kw):
    b = mk_bundle(4, ["A", "A", "B", "B"])
    specs = {"L2s0": table_arm(b, L2S0_TABLE if l2s0 is None else l2s0), "L2s1": table_arm(b, L2S1_TABLE if l2s1 is None else l2s1), "P2HEAD": table_arm(b, P2_TABLE if p2 is None else p2)}
    return write_world(root, b, specs, **kw), b, specs


def close(a, b, tol=1e-12):
    return a is not None and abs(a - b) <= tol


# ------------------------------------------------------------------------------------------------------------------------------ (a) hand-computed A3 / A4 / A5
def test_a_planted_run_dir_reproduces_the_hand_computed_a3_a4_a5(tmp_path):
    w, _, _ = w1_world(tmp_path)
    r = run_compute(w, expect_logs=2)
    assert r["status"] == "COMPUTED" and r["reported_only"] is True and r["gated"] is False and r["changes_step_b_verdict"] is False
    a = r["arms"]["L2s0"]
    for k, (num, den) in HAND_L2S0.items():
        assert close(a["pooled"][k]["point"], num / den), (k, a["pooled"][k])
        assert a["pooled"][k]["den"] == den and close(a["pooled"][k]["num"], num)
    assert a["counts"]["n_pairs"] == 11 and a["counts"]["n_identical_excluded"] == 1 and a["counts"]["n_pairs_total"] == 12
    for k, rows in HAND_L2S0_SEED.items():
        for s, (num, den) in enumerate(rows):
            got = a["per_seed"][s][k]
            if den == 0:
                assert got["point"] is None and got["den"] == 0                           # an undefined ratio is None, never 0 and never NaN
            else:
                assert close(got["point"], num / den), (k, s, got)
    assert [row["seed"] for row in a["per_seed"]] == [int(x) for x in SEEDS]


def test_a_the_other_two_arms_are_not_mixed_up_and_the_paired_differences_are_right(tmp_path):
    w, _, _ = w1_world(tmp_path)
    r = run_compute(w, expect_logs=2)
    s1, p2 = r["arms"]["L2s1"], r["arms"]["P2HEAD"]
    assert (s1["pooled"]["A3"]["point"], s1["pooled"]["A4"]["point"], s1["pooled"]["A5"]["point"]) == (0.0, 1.0, 1.0)
    assert (p2["pooled"]["A3"]["point"], p2["pooled"]["A4"]["point"], p2["pooled"]["A5"]["point"]) == (0.5, 0.5, 0.0)
    assert [row["A3"]["point"] for row in p2["per_seed"]] == [1.0, 0.0, None]                # P2HEAD seed 2 has no pair at all
    d0, d1 = r["differences_L2_minus_P2HEAD"]["L2s0"]["pooled"], r["differences_L2_minus_P2HEAD"]["L2s1"]["pooled"]
    assert close(d0["A3"]["point"], 0.0) and close(d0["A4"]["point"], 3 / 7 - 0.5) and close(d0["A5"]["point"], 2 / 7)
    assert close(d1["A3"]["point"], -0.5) and close(d1["A4"]["point"], 0.5) and close(d1["A5"]["point"], 1.0)
    ds = r["differences_L2_minus_P2HEAD"]["L2s0"]["per_seed"]
    assert close(ds[0]["A3"]["point"], 0.625 - 1.0) and close(ds[1]["A3"]["point"], 0.0 - 0.0) and ds[2]["A3"]["point"] is None      # seed 2: P2HEAD has no pair -> the difference is undefined
    assert r["arms"]["P2HEAD"]["counts"]["n_pairs"] == 2


def test_a_the_terms_come_from_l1s_unchanged_code_not_a_copy():
    import inspect
    assert AM.A is stepB.A and os.path.normcase(os.path.dirname(inspect.getsourcefile(A.admission_terms))) == os.path.normcase(stepB.L1_CODE)
    assert stepB.verify_l1_code() == AM.check_l1_code()                                   # the four L1 files are the landed ones (LF sha256)
    src = open(os.path.join(HERE, "amend1_concordance.py"), encoding="utf-8").read()
    assert "def admission_terms" not in src and "def concordance_counts" not in src      # never copied


def test_a_red_arm_a_broken_stepA_term_changes_the_numbers_so_the_check_is_wired_to_it(tmp_path, monkeypatch):
    w, _, _ = w1_world(tmp_path)
    base = run_compute(w, expect_logs=2)
    real = A.concordance_counts

    def flipped(s_a, s_p, t_a, t_p, thr=A.THR_TRUE_PAIR):
        conc, m = real(s_a, s_p, t_a, t_p, thr)
        return m - conc, m                                                                 # concordant <-> discordant

    monkeypatch.setattr(A, "concordance_counts", flipped)
    red = run_compute(w, expect_logs=2)
    assert base["arms"]["L2s0"]["pooled"]["A3"]["point"] == 0.5 and red["arms"]["L2s0"]["pooled"]["A3"]["point"] == 0.5     # 4.5/9 is symmetric: look at the arm that is not
    assert base["arms"]["L2s1"]["pooled"]["A3"]["point"] == 0.0 and red["arms"]["L2s1"]["pooled"]["A3"]["point"] == 1.0


def test_a_anchor_and_candidate_are_not_swapped(tmp_path):
    """one pair: the anchor's logits are low and the candidate's high, and the true PDMS RISES -> concordant.  A swap of anchor and candidate would read discordant."""
    b = mk_bundle(2, ["A", "B"])
    specs = {k: table_arm(b, {} if k != "L2s0" else {(0, 0): (2, 0.95, None)}) for k in ("L2s0", "L2s1", "P2HEAD")}
    w = write_world(tmp_path, b, specs)
    r = run_compute(w, expect_logs=2)
    assert r["arms"]["L2s0"]["pooled"]["A3"]["point"] == 1.0 and r["arms"]["L2s0"]["pooled"]["A4"]["point"] == 0.0 and r["arms"]["L2s0"]["pooled"]["A5"]["point"] == 0.0


# ------------------------------------------------------------------------------------------------------------------------------ a bigger random world with an independent pure-python oracle (W3, 24 logs)
def mk_w3(root, seed=5, n=48, S=3):
    rng = np.random.default_rng(seed)
    logs = [f"L{i % 24:02d}" for i in range(n)]
    b = mk_bundle(n, logs, K=4, pick=rng.integers(0, 4, n), seed=seed)
    b["sub"] = (rng.uniform(size=b["sub"].shape) < 0.8).astype(np.float64)
    b["pdms"] = rng.uniform(0.2, 1.0, b["pdms"].shape).round(3)
    base = rng.normal(0, 1.0, (n, 6))
    specs = {}
    for name, _, _ in AM.ARMS:
        pl = base.copy()
        if name != "P2HEAD":
            pl[:, L2S.EP] = rng.normal(0, 1.0, n)                                        # only the EP column differs from P2HEAD at the pick
        moved = rng.uniform(size=(n, S)) < 0.8
        out = np.repeat(pl[:, None], S, axis=1).copy()
        out[moved] = rng.normal(0, 1.0, (int(moved.sum()), 6))
        ls = (rng.uniform(size=(n, S, 6)) < 0.8).astype(np.float64)
        lp = rng.uniform(0.0, 1.0, (n, S)).round(3)
        pick = b["pick"]
        for i in range(n):
            for s in range(S):
                if not moved[i, s]:
                    ls[i, s], lp[i, s] = b["sub"][i, pick[i]], b["pdms"][i, pick[i]]
        specs[name] = {"pick_logits": pl, "out_logits": out, "moved": moved, "lab_sub": ls, "lab_pdms": lp}
    return write_world(root, b, specs), b, specs


def sigm(x):
    return 1.0 / (1.0 + math.exp(-x))


def agg_ref(l):
    p = [sigm(v) for v in l]
    return p[0] * p[1] * (5 * p[2] + 5 * p[3] + 2 * p[4]) / 12.0                          # NC x DAC x (5 EP + 5 TTC + 2 C) / 12, from the amendment's text


def sgn(x):
    return int(x > 0) - int(x < 0)


def oracle(b, spec, n_logs=24):
    """pure-python pairs -> per-log (num, den) for A3 / A4 / A5, pooled and per seed (the amendment's rules, written from its text)"""
    n, S = spec["moved"].shape
    logs = sorted(set(b["log"]))
    li = {l: i for i, l in enumerate(logs)}
    z = lambda: {k: [np.zeros(n_logs), np.zeros(n_logs)] for k in AM.KEYS}
    pooled, per_seed = z(), [z() for _ in range(S)]
    for i in range(n):
        pk = b["pick"][i]
        for s in range(S):
            if not spec["moved"][i, s]:
                continue                                                                   # an output identical to the pick contributes to no pair
            ag, pg = agg_ref(spec["pick_logits"][i]), agg_ref(spec["out_logits"][i, s])
            a_pd, p_pd = b["pdms"][i, pk], spec["lab_pdms"][i, s]
            a_sub, p_sub = b["sub"][i, pk], spec["lab_sub"][i, s]
            terms = {}
            if abs(p_pd - a_pd) >= 0.05 - 1e-12:
                ds, dt = sgn(pg - ag), sgn(p_pd - a_pd)
                terms["A3"] = (0.5 if ds == 0 else float(ds == dt), 1.0)
            rise = (pg - ag) >= 0.02 - 1e-12
            if rise:
                fall = (p_pd - a_pd) <= -0.10 + 1e-12
                flip = (a_sub[0] == 1 and p_sub[0] == 0) or (a_sub[1] == 1 and p_sub[1] == 0)
                terms["A4"] = (float(fall), 1.0)
                terms["A5"] = (float(flip), 1.0)
            for k, (nu, de) in terms.items():
                for tgt in (pooled, per_seed[s]):
                    tgt[k][0][li[b["log"][i]]] += nu
                    tgt[k][1][li[b["log"][i]]] += de
    return pooled, per_seed


def oracle_ci(num, den, idx):
    pt = num.sum() / den.sum() if den.sum() > 0 else float("nan")
    n, d = num[idx].sum(1), den[idx].sum(1)
    ok = d > 0
    r = n[ok] / d[ok]
    return pt, float(np.quantile(r, 0.025)), float(np.quantile(r, 0.975))


def oracle_diff(a, b, idx):
    na, da = a
    nb, db = b
    ok = (da[idx].sum(1) > 0) & (db[idx].sum(1) > 0)
    r = na[idx].sum(1)[ok] / da[idx].sum(1)[ok] - nb[idx].sum(1)[ok] / db[idx].sum(1)[ok]
    return na.sum() / da.sum() - nb.sum() / db.sum(), float(np.quantile(r, 0.025)), float(np.quantile(r, 0.975))


def test_a_random_world_matches_an_independent_oracle_including_the_bootstrap_and_the_differences(tmp_path):
    w, b, specs = mk_w3(tmp_path)
    r = run_compute(w, expect_logs=24)
    idx = np.random.default_rng(20260927).integers(0, 24, size=(10000, 24))                # the sealed draws, written literally: 24 logs, 10,000 resamples, seed 20260927
    assert np.array_equal(idx, A.boot_indices(24, 10000, 20260927)) and r["estimator"]["B"] == 10000 and r["estimator"]["seed"] == 20260927 and r["estimator"]["n_logs"] == 24
    O = {n: oracle(b, specs[n]) for n in specs}
    for n in specs:
        pooled, per_seed = O[n]
        for k in AM.KEYS:
            pt, lo, hi = oracle_ci(*pooled[k], idx)
            got = r["arms"][n]["pooled"][k]
            assert close(got["point"], pt) and close(got["lo"], lo) and close(got["hi"], hi), (n, k, got, (pt, lo, hi))
            for s in range(3):
                pt, lo, hi = oracle_ci(*per_seed[s][k], idx)
                g = r["arms"][n]["per_seed"][s][k]
                assert close(g["point"], pt) and close(g["lo"], lo) and close(g["hi"], hi)
    for n in ("L2s0", "L2s1"):
        for k in AM.KEYS:
            pt, lo, hi = oracle_diff(O[n][0][k], O["P2HEAD"][0][k], idx)
            g = r["differences_L2_minus_P2HEAD"][n]["pooled"][k]
            assert close(g["point"], pt) and close(g["lo"], lo) and close(g["hi"], hi), (n, k, g, (pt, lo, hi))
            for s in range(3):
                pt, lo, hi = oracle_diff(O[n][1][s][k], O["P2HEAD"][1][s][k], idx)
                g = r["differences_L2_minus_P2HEAD"][n]["per_seed"][s][k]
                assert close(g["point"], pt) and close(g["lo"], lo) and close(g["hi"], hi)
    assert 0.0 <= r["arms"]["L2s0"]["pooled"]["A3"]["lo"] <= r["arms"]["L2s0"]["pooled"]["A3"]["point"] <= r["arms"]["L2s0"]["pooled"]["A3"]["hi"] <= 1.0


# ------------------------------------------------------------------------------------------------------------------------------ (b) an output identical to the pick contributes to no pair
def test_b_an_unmoved_output_contributes_to_no_pair_even_with_a_decoy_label(tmp_path):
    base_w, b, specs = w1_world(os.path.join(str(tmp_path), "base"))
    base = run_compute(base_w, expect_logs=2)
    # decoy: t1 s2 is UNMOVED (identical to the pick: same logits) but its label is wildly different (PDMS 0.10, NC 0).  If it were a pair it would be an A3 tie (+0.5, den +1).
    dec = {k: dict(v) for k, v in specs.items()}
    dec["L2s0"]["lab_pdms"] = specs["L2s0"]["lab_pdms"].copy()
    dec["L2s0"]["lab_pdms"][1, 2] = 0.10
    dec["L2s0"]["lab_sub"] = specs["L2s0"]["lab_sub"].copy()
    dec["L2s0"]["lab_sub"][1, 2, 0] = 0.0
    w2 = write_world(os.path.join(str(tmp_path), "decoy"), b, dec)
    r = run_compute(w2, expect_logs=2)
    for k in AM.KEYS:
        assert r["arms"]["L2s0"]["pooled"][k] == base["arms"]["L2s0"]["pooled"][k]       # unchanged to the last bit: the decoy is in no pair
    assert r["arms"]["L2s0"]["counts"]["n_identical_excluded"] == 1 and r["arms"]["L2s0"]["counts"]["n_pairs"] == 11
    # control: the SAME row, but MOVED (a real candidate that happens to have the pick's logits) IS a pair: A3 counts it as a tie and the denominator grows
    ctl = {k: dict(v) for k, v in dec.items()}
    ctl["L2s0"]["moved"] = specs["L2s0"]["moved"].copy()
    ctl["L2s0"]["moved"][1, 2] = True
    w3 = write_world(os.path.join(str(tmp_path), "ctl"), b, ctl)
    rc = run_compute(w3, expect_logs=2)
    assert rc["arms"]["L2s0"]["pooled"]["A3"]["den"] == base["arms"]["L2s0"]["pooled"]["A3"]["den"] + 1 and close(rc["arms"]["L2s0"]["pooled"]["A3"]["num"], 4.5 + 0.5)
    assert rc["arms"]["L2s0"]["counts"]["n_identical_excluded"] == 0 and rc["arms"]["L2s0"]["counts"]["n_pairs"] == 12


def test_b_a_moved_flag_on_an_output_whose_plan_is_the_picks_is_still_no_pair_and_an_inconsistent_unmoved_flag_is_refused(tmp_path):
    w, b, specs = w1_world(os.path.join(str(tmp_path), "a"))
    n = 4
    z = np.load(os.path.join(w["run"], "search_L2s0.npz"))
    ex = np.array(z["exec"])
    ex[0, 0] = z["pick_exec"][0]                                                           # t0 s0 is flagged moved, but its executed plan IS the pick's
    edit_search(w, "L2s0", exec=ex)
    r = run_compute(w, expect_logs=2)
    assert r["arms"]["L2s0"]["counts"]["n_identical_excluded"] == 2 and r["arms"]["L2s0"]["counts"]["n_pairs"] == 10
    w2, _, _ = w1_world(os.path.join(str(tmp_path), "b"))
    z = np.load(os.path.join(w2["run"], "search_L2s0.npz"))
    ex = np.array(z["exec"])
    ex[1, 2] += 1.0                                                                        # t1 s2 is flagged UNMOVED but its executed plan is not the pick's: the chain's own invariant is broken
    edit_search(w2, "L2s0", exec=ex)
    with pytest.raises(AM.NotComputable, match="unmoved output's exec is not the pick's exec"):
        run_compute(w2, expect_logs=2)


# ------------------------------------------------------------------------------------------------------------------------------ (c) RED: corrupt one banked pick logit
def test_c_structural_identity_refuses_a_corrupted_banked_pick_logit(tmp_path):
    w, _, _ = w1_world(os.path.join(str(tmp_path), "ok"))
    assert run_compute(w, expect_logs=2)["status"] == "COMPUTED"                           # green control
    w1, _, _ = w1_world(os.path.join(str(tmp_path), "c1"))
    z = np.load(os.path.join(w1["run"], "search_L2s0.npz"))
    pl = np.array(z["pred_logit_pick"])
    pl[0, 3] += 1e-9                                                                       # a NON-EP column at token t0: P2HEAD's differs by 1e-9 -> only the EP row may differ between arms
    edit_search(w1, "L2s0", pred_logit_pick=pl)
    with pytest.raises(AM.NotComputable, match=r"identity FAILED \(L2s0\).*non-EP columns"):
        run_compute(w1, expect_logs=2)
    w2, _, _ = w1_world(os.path.join(str(tmp_path), "c2"))
    z = np.load(os.path.join(w2["run"], "search_L2s0.npz"))
    pl = np.array(z["pred_logit_pick"])
    pl[1, L2S.EP] += 1e-9                                                                  # the EP column of a token whose seed 2 is UNMOVED (its output logits must equal the pick's, bit for bit)
    edit_search(w2, "L2s0", pred_logit_pick=pl)
    with pytest.raises(AM.NotComputable, match=r"UNMOVED .* banked output logits differ from the banked pick logits"):
        run_compute(w2, expect_logs=2)


@pytest.fixture(scope="module")
def model_world(tmp_path_factory):
    """a planted Step B run made by the REAL `stepB.run_search` on the fake set-conditional scorer: 6 tokens, 3 logs, heads s0 / s1 = P2's EP row replaced, P2HEAD = plain"""
    d = str(tmp_path_factory.mktemp("w2"))
    rng = np.random.default_rng(11)
    n = 6
    b = mk_bundle(n, [f"log{i % 3}" for i in range(n)], K=64, pick=rng.integers(0, 64, n), seed=12, traj=True)
    ck = os.path.join(d, "model_final.pt")
    open(ck, "wb").write(b"not a checkpoint: the fake scorer ignores it")
    ns, _ = FAKE.load_scorer(ck)
    W, bb = ns.score_head.weight.detach().numpy().copy(), ns.score_head.bias.detach().numpy().copy()
    heads_dir = os.path.join(d, "heads")
    os.makedirs(heads_dir)
    shas = {}
    for s, (dw, db) in (("s0", (0.6, -0.25)), ("s1", (-0.7, 0.3))):
        W2, b2 = W.copy(), bb.copy()
        W2[L2S.EP] = W[L2S.EP] * -1.5 + dw
        b2[L2S.EP] = bb[L2S.EP] + db
        p = os.path.join(heads_dir, f"final_head_{s}.npz")
        np.savez(p, weight=W2.astype(np.float32), bias=b2.astype(np.float32), kind=np.array("test"))
        shas[s] = L2S.sha256_file(p)
    cfg = stepB.c8_from_grid(m_pop=16, n_elite=2)                                          # a small population: the mechanics are the real ones, the cost is seconds
    pins = {"head_s0_sha256": shas["s0"], "head_s1_sha256": shas["s1"], "ckpt_sha256": L2S.sha256_file(ck), "config_sha256": H64("d")}
    specs, outs = {}, {}
    for name, kind, s in AM.ARMS:
        head = L2S.read_head_file(os.path.join(heads_dir, f"final_head_{s}.npz"), shas[s]) if s else None
        fn = L2S.build_score_fn(FAKE, ck, "unused", np.asarray(b["token"]), head, refe_dir=None, device="cpu")
        out = stepB.run_search(fn, b, range(n), cfg)
        outs[name] = out
        lab_sub = np.clip(rng.uniform(0, 1, (n, 3, 6)), 0, 1).round(2)
        lab_pdms = np.clip(rng.uniform(0, 1, (n, 3)), 0, 1).round(2)
        unm = ~out["moved"]
        for i, sd in zip(*np.nonzero(unm)):
            lab_sub[i, sd], lab_pdms[i, sd] = b["sub"][i, b["pick"][i]], b["pdms"][i, b["pick"][i]]
        specs[name] = {"pick_logits": out["pred_logit_pick"], "out_logits": out["pred_logit_out"], "moved": out["moved"], "lab_sub": lab_sub, "lab_pdms": lab_pdms, "scorer_in": out["scorer_in"], "exec": out["exec"],
                       "pick_exec": out["pick_exec"]}
    assert all(sp["moved"].any() for sp in specs.values()), "the planted search must move at least one output per arm (else the model identity would be vacuous)"
    return {"dir": d, "bundle": b, "specs": specs, "pins": pins, "heads_dir": heads_dir, "ckpt": ck, "outs": outs}


def mw(model_world, root, **kw):
    mwd = model_world
    w = write_world(root, mwd["bundle"], mwd["specs"], pins_over=mwd["pins"], **kw)
    w["heads_dir"], w["ckpt"] = mwd["heads_dir"], mwd["ckpt"]
    return w


def model_kw(w):
    return dict(scorer_mod=FAKE, heads_dir=w["heads_dir"], ckpt=w["ckpt"], ctx=os.path.join(w["root"], "ctx_unused.npy"), refe_dir=None, threads=2, expect_logs=3)


def test_c_the_model_identity_reproduces_the_banked_logits_bit_for_bit_and_rescore_equals_banked(model_world, tmp_path):
    w = mw(model_world, tmp_path)
    banked = run_compute(w, logits="banked", identity_tokens=6, **model_kw(w))
    assert banked["identity"]["logits_source"] == "banked" and set(banked["identity"]["model"]) == {"L2s0", "L2s1", "P2HEAD"}
    assert all(v["pick_logits_bitwise_equal"] and v["out_logits_bitwise_equal"] and v["tokens_checked"] == 6 for v in banked["identity"]["model"].values())
    resc = run_compute(w, logits="rescore", **model_kw(w))
    assert resc["identity"]["logits_source"] == "rescore"
    for n in banked["arms"]:
        assert banked["arms"][n]["pooled"] == resc["arms"][n]["pooled"] and banked["arms"][n]["per_seed"] == resc["arms"][n]["per_seed"]
    assert banked["differences_L2_minus_P2HEAD"] == resc["differences_L2_minus_P2HEAD"]


def test_the_files_the_real_shard_writer_and_merge_produce_are_read(model_world, tmp_path):
    """format fidelity: the search files here are NOT planted by this test: they are `stepB.run_search`'s own output, written as shards the way `run_search_cli` writes them and merged by `stepB.cmd_merge`"""
    w = mw(model_world, tmp_path)
    n = len(model_world["bundle"]["token"])
    for name, kind, sd in AM.ARMS:
        sdir = os.path.join(w["root"], f"shards_{name}")
        os.makedirs(sdir)
        meta = {"arm": kind, "head_sha256": model_world["pins"][f"head_{sd}_sha256"] if sd else None, "ckpt_sha256": model_world["pins"]["ckpt_sha256"], "config_sha256": model_world["pins"]["config_sha256"],
                "bundle_sha256": sha_file(w["bundle"]), "data_clause": {"carve_sha256": w["pins"]["carve_sha256"]}}
        for lo, hi in ((0, 4), (4, n)):
            out = {k: (v if k == "seeds" else v[lo:hi]) for k, v in model_world["outs"][name].items()}
            np.savez(os.path.join(sdir, f"shard_{kind}_{lo:04d}.npz"), meta=np.array(json.dumps({**meta, "lo": lo, "hi": hi})), **out)
        assert stepB.cmd_merge(["--out-dir", sdir, "--arm", kind, "--out", os.path.join(w["run"], f"search_{name}.npz"), "--n", str(n)]) == 0
        rebind_labels(w, name)
    r = run_compute(w, logits="banked", identity_tokens=6, **model_kw(w))
    assert r["status"] == "COMPUTED" and all(v["pick_logits_bitwise_equal"] for v in r["identity"]["model"].values())
    planted = run_compute(mw(model_world, os.path.join(str(tmp_path), "planted")), logits="banked", **model_kw(w))
    for k in r["arms"]:
        assert r["arms"][k]["pooled"] == planted["arms"][k]["pooled"]


def test_c_the_rescore_path_works_when_the_output_logits_are_not_banked(model_world, tmp_path):
    specs = {k: {kk: vv for kk, vv in v.items() if kk != "out_logits"} for k, v in model_world["specs"].items()}
    mwd = dict(model_world, specs=specs)
    w = mw(mwd, tmp_path)
    r = run_compute(w, logits="auto", **model_kw(w))
    assert r["identity"]["logits_source"] == "rescore" and all(v["out_logits_bitwise_equal"] is None for v in r["identity"]["model"].values())
    full = run_compute(mw(model_world, os.path.join(str(tmp_path), "full")), logits="banked", **{**model_kw(w), "heads_dir": w["heads_dir"]})
    for n in r["arms"]:
        assert r["arms"][n]["pooled"] == full["arms"][n]["pooled"]                          # the re-scored outputs are the banked ones, so the numbers agree
    with pytest.raises(AM.NotComputable, match="pred_logit_out is not banked"):
        run_compute(w, logits="banked", **model_kw(w))


@pytest.mark.parametrize("mode,tokens", [("rescore", 0), ("banked", 6)])
def test_c_red_a_corrupted_banked_pick_logit_makes_the_model_identity_refuse(model_world, tmp_path, mode, tokens):
    w = mw(model_world, tmp_path)
    assert run_compute(w, logits=mode, identity_tokens=tokens, **model_kw(w))["status"] == "COMPUTED"      # green control (nothing corrupted)
    z = np.load(os.path.join(w["run"], "search_L2s0.npz"))
    pl = np.array(z["pred_logit_pick"])
    pl[2, L2S.EP] += 1e-9                                                                   # ONE logit of ONE token in the arm's EP column, a 1e-9 corruption: bit for bit, not allclose
    out = np.array(z["pred_logit_out"])
    um = ~np.asarray(z["moved"])
    out[um] = pl[np.nonzero(um)[0]]                                                         # keep the model-free identities satisfied: only the MODEL identity can see it
    edit_search(w, "L2s0", pred_logit_pick=pl, pred_logit_out=out)
    with pytest.raises(AM.NotComputable, match=r"identity FAILED \(L2s0\): the re-scored logits at P2's pick differ from the banked pred_logit_pick at token t2"):
        run_compute(w, logits=mode, identity_tokens=tokens, **model_kw(w))
    # the arm of the control: with the model identity REMOVED (a mutation) the very same corrupted file sails through and gives numbers: the identity is what refuses
    import amend1_concordance
    orig = amend1_concordance.model_identity
    try:
        amend1_concordance.model_identity = lambda arm, bundle, scorer, rows: ({"tokens_checked": 0, "pick_logits_bitwise_equal": True, "out_logits_bitwise_equal": None}, np.array(arm["so"]["pred_logit_out"]))
        red = run_compute(w, logits=mode, identity_tokens=tokens, **model_kw(w))
    finally:
        amend1_concordance.model_identity = orig
    assert red["status"] == "COMPUTED"


def test_c_a_corrupted_banked_output_logit_refuses_under_the_model_identity(model_world, tmp_path):
    w = mw(model_world, tmp_path)
    z = np.load(os.path.join(w["run"], "search_P2HEAD.npz"))
    out = np.array(z["pred_logit_out"])
    i, s = [int(x[0]) for x in np.nonzero(np.asarray(z["moved"]))]
    out[i, s, 0] += 1e-9
    edit_search(w, "P2HEAD", pred_logit_out=out)
    with pytest.raises(AM.NotComputable, match=r"identity FAILED \(P2HEAD\): the re-scored logits at the search output differ from the banked pred_logit_out"):
        run_compute(w, logits="banked", identity_tokens=6, **model_kw(w))


def test_c_the_model_free_identities_alone_do_not_see_an_isolated_output_logit_corruption_which_is_why_the_run_carries_identity_tokens(model_world, tmp_path, monkeypatch):
    """an honest limit, pinned: with --identity-tokens 0 only the structural identities run, and a 1e-9 change of ONE moved output's logit is consistent with everything model-free"""
    w = mw(model_world, tmp_path)
    z = np.load(os.path.join(w["run"], "search_P2HEAD.npz"))
    out = np.array(z["pred_logit_out"])
    i, s = [int(x[0]) for x in np.nonzero(np.asarray(z["moved"]))]
    out[i, s, 0] += 1e-9
    edit_search(w, "P2HEAD", pred_logit_out=out)
    before = sorted(os.listdir(w["run"]))
    assert cli(w, monkeypatch, expect_logs=3) == 0
    assert sorted(os.listdir(w["run"])) == before                                           # the script never writes into the run dir


def test_c_cli_not_computable_writes_the_section_and_exits_nonzero(tmp_path, capsys, monkeypatch):
    w, _, _ = w1_world(tmp_path)
    z = np.load(os.path.join(w["run"], "search_L2s0.npz"))
    pl = np.array(z["pred_logit_pick"])
    pl[0, 3] += 1e-9
    edit_search(w, "L2s0", pred_logit_pick=pl)
    rc = cli(w, monkeypatch)
    assert rc == AM.EXIT["not_computable"] == 3
    res = json.load(open(os.path.join(w["out"], "amend1_result.json"), encoding="utf-8"))
    assert res["status"] == "NOT_COMPUTABLE" and "identity FAILED" in res["reason"] and "arms" not in res                  # no number is quoted
    md = open(os.path.join(w["out"], "AMEND1.md"), encoding="utf-8").read()
    assert md.startswith(f"## {AM.SECTION}") and "NOT COMPUTABLE" in md and "non-EP columns" in md
    assert "ZZAMEND1_NOT_COMPUTABLE" in capsys.readouterr().out


# ------------------------------------------------------------------------------------------------------------------------------ (d) the bootstrap is deterministic under its seed
def test_d_the_bootstrap_is_deterministic_under_its_seed_and_the_seed_matters(tmp_path):
    w, _, _ = mk_w3(tmp_path)
    r1, r2 = run_compute(w, expect_logs=24), run_compute(w, expect_logs=24)
    for r in (r1, r2):
        r.pop("elapsed_s"), r.pop("generated")
    assert r1 == r2                                                                         # identical to the last bit, every arm, every seed, every difference
    r3 = run_compute(w, expect_logs=24, seed=20260928)
    assert r3["arms"]["L2s0"]["pooled"]["A3"]["point"] == r1["arms"]["L2s0"]["pooled"]["A3"]["point"]              # the point estimate does not depend on the draws
    assert (r3["arms"]["L2s0"]["pooled"]["A3"]["lo"], r3["arms"]["L2s0"]["pooled"]["A3"]["hi"]) != (r1["arms"]["L2s0"]["pooled"]["A3"]["lo"], r1["arms"]["L2s0"]["pooled"]["A3"]["hi"])   # the control: the seed is read
    assert np.array_equal(A.boot_indices(24, 10000, 20260927), A.boot_indices(24, 10000, 20260927))


# ------------------------------------------------------------------------------------------------------------------------------ (e) the seal
def test_e_refuses_when_the_amendment_sha_does_not_match_its_text(tmp_path, capsys, monkeypatch):
    w, _, _ = w1_world(tmp_path)
    assert cli(w, monkeypatch) == 0                                                         # green control
    shutil.rmtree(w["out"])
    p = os.path.join(w["pkg"], AM.AMEND_MD)
    open(p, "a", encoding="utf-8").write("one edited line after sealing\n")
    rc = cli(w, monkeypatch)
    assert rc == AM.EXIT["seal"] == 5 and not os.path.exists(w["out"])                      # nothing is written: no number from an unsealed protocol
    assert "SEAL_MISMATCH" in capsys.readouterr().out
    with pytest.raises(AM.SealRefused, match="SEAL_MISMATCH"):
        AM.check_seals(w["pkg"])


def test_e_refuses_a_tampered_sha_record_a_missing_file_and_an_edited_main_text(tmp_path):
    for case in ("record", "missing_sha", "missing_md", "main_edited", "bad_record"):
        w, _, _ = w1_world(os.path.join(str(tmp_path), case))
        AM.check_seals(w["pkg"])                                                            # green
        if case == "record":
            p = os.path.join(w["pkg"], AM.AMEND_SHA)
            h = open(p).read()
            open(p, "w").write(("0" if h[0] != "0" else "1") + h[1:])
        elif case == "missing_sha":
            os.remove(os.path.join(w["pkg"], AM.AMEND_SHA))
        elif case == "missing_md":
            os.remove(os.path.join(w["pkg"], AM.AMEND_MD))
        elif case == "main_edited":
            open(os.path.join(w["pkg"], AM.MAIN_MD), "a").write("edited\n")
        else:
            open(os.path.join(w["pkg"], AM.AMEND_SHA), "w").write("not-a-sha *x\n")
        with pytest.raises(AM.SealRefused):
            AM.check_seals(w["pkg"])
        with pytest.raises(AM.SealRefused):
            run_compute(w, expect_logs=2)


def test_e_the_crlf_checkout_of_the_sealed_text_still_matches_and_a_draft_main_text_is_refused(tmp_path):
    w, _, _ = w1_world(tmp_path)
    p = os.path.join(w["pkg"], AM.AMEND_MD)
    t = open(p, "rb").read()
    open(p, "wb").write(t.replace(b"\n", b"\r\n"))                                          # a Windows checkout of the same sealed text
    assert AM.check_seals(w["pkg"])["amendment_sha256"] == hashlib.sha256(t).hexdigest()
    pm = os.path.join(w["pkg"], AM.MAIN_MD)
    txt = open(pm).read().replace("Status: SEALED", "Status: DRAFT")
    open(pm, "w").write(txt)
    open(os.path.join(w["pkg"], AM.MAIN_SHA), "w").write(hashlib.sha256(txt.encode()).hexdigest() + "\n")
    with pytest.raises(AM.SealRefused, match="SEAL_NOT_SEALED"):
        AM.check_seals(w["pkg"])


def test_e_the_real_sealed_amendment_matches_its_recorded_sha():
    pkg = os.path.normpath(os.path.join(HERE, ".."))
    if not os.path.exists(os.path.join(pkg, AM.AMEND_MD)):
        pytest.skip("not on the box that holds the sealed text")
    s = AM.check_seals(pkg)                                                                 # reads the SEALED TEXT only (never a Step B output)
    assert s["amendment_sha256"].startswith("a2e0c2fe") and s["main_sha256"].startswith("c661588c")


# ------------------------------------------------------------------------------------------------------------------------------ the preconditions and the other refusals
def test_the_chain_must_be_done_and_the_marker_is_anchored(tmp_path, capsys, monkeypatch):
    w, _, _ = w1_world(os.path.join(str(tmp_path), "nodone"), done=None)
    assert cli(w, monkeypatch) == AM.EXIT["precondition"] == 4 and not os.path.exists(w["out"])
    assert cli(w, monkeypatch, "--allow-no-done") == 0                                      # a stopped chain, explicitly allowed
    shutil.rmtree(w["out"])
    open(os.path.join(w["run"], "chain_stepB.log"), "a").write("heartbeat: still waiting for ZZSTEPB_DONE to appear\n  ZZSTEPB_DONE indented is not a marker either\n")
    assert cli(w, monkeypatch) == 4                                                         # a line that merely MENTIONS the marker does not count
    open(os.path.join(w["run"], "chain_stepB.log"), "a").write("ZZSTEPB_DONE TRADE\n")
    assert cli(w, monkeypatch) == 0
    out = capsys.readouterr().out
    assert "TRADE" not in out and "NOT_CLEARED" not in out and "O1_PASS" not in out         # the verdict word after the marker is never read into any output
    for f in ("amend1_result.json", "AMEND1.md"):
        assert "TRADE" not in open(os.path.join(w["out"], f), encoding="utf-8").read()


@pytest.mark.parametrize("what", ["head_pin", "ckpt_pin", "bundle", "labels_not_bound", "token_order", "pick", "nonfinite_label", "identity_record", "n_logs", "arm_kind", "seeds", "shape", "carve"])
def test_other_refusals_are_not_computable_with_a_reason(tmp_path, what):
    w, b, specs = w1_world(tmp_path)
    kw = {"expect_logs": 2}
    run = w["run"]
    if what == "head_pin":
        w, _, _ = w1_world(tmp_path / "x", meta_over={"L2s1": {"head_sha256": H64("f")}})
    elif what == "ckpt_pin":
        w, _, _ = w1_world(tmp_path / "x", meta_over={"P2HEAD": {"ckpt_sha256": H64("f")}})
    elif what == "bundle":
        w, _, _ = w1_world(tmp_path / "x", meta_over={"L2s0": {"bundle_sha256": H64("f")}})
    elif what == "carve":
        w, _, _ = w1_world(tmp_path / "x", meta_over={"L2s0": {"data_clause": {"carve_sha256": H64("f")}}})
    elif what == "arm_kind":
        w, _, _ = w1_world(tmp_path / "x", meta_over={"L2s0": {"arm": "P2HEAD"}})
    elif what == "labels_not_bound":
        z = {k: v for k, v in np.load(os.path.join(run, "labels_L2s0.npz")).items()}
        z["search_sha256"] = np.array(H64("e"))
        np.savez(os.path.join(run, "labels_L2s0.npz"), **z)
    elif what == "token_order":
        z = {k: v for k, v in np.load(os.path.join(run, "labels_L2s1.npz")).items()}
        z["token"] = z["token"][::-1].copy()
        np.savez(os.path.join(run, "labels_L2s1.npz"), **z)
    elif what == "pick":
        edit_search(w, "P2HEAD", pick=(b["pick"] + 1) % 4)
    elif what == "nonfinite_label":
        z = {k: v for k, v in np.load(os.path.join(run, "labels_L2s1.npz")).items()}
        z["pdms"][0, 0] = np.nan
        np.savez(os.path.join(run, "labels_L2s1.npz"), **z)
    elif what == "identity_record":
        z = {k: v for k, v in np.load(os.path.join(run, "labels_L2s0.npz")).items()}
        z["identity"] = np.array(json.dumps({"passed": True, "control_ok": False}))
        np.savez(os.path.join(run, "labels_L2s0.npz"), **z)
    elif what == "n_logs":
        kw = {"expect_logs": 24}
    elif what == "seeds":
        edit_search(w, "L2s1", seeds=SEEDS + 1)
    elif what == "shape":
        z = np.load(os.path.join(run, "search_L2s0.npz"))
        edit_search(w, "L2s0", pred_logit_out=np.array(z["pred_logit_out"])[:, :2])
    with pytest.raises(AM.NotComputable):
        run_compute(w, **kw)


def test_the_outputs_are_strict_json_with_the_exact_section_title_and_nothing_is_gated(tmp_path, monkeypatch):
    w, _, _ = w1_world(tmp_path)
    assert cli(w, monkeypatch) == 0
    raw = open(os.path.join(w["out"], "amend1_result.json"), encoding="utf-8").read()

    def bad(c):
        raise ValueError(f"non-strict JSON constant {c}")
    res = json.loads(raw, parse_constant=bad)                                               # no NaN / Infinity anywhere: an undefined ratio is null
    assert res["status"] == "COMPUTED" and res["gated"] is False and res["reported_only"] is True and res["section"] == "AMENDMENT 1: aggregate concordance of the search outputs (reported only)"
    md = open(os.path.join(w["out"], "AMEND1.md"), encoding="utf-8").read()
    assert md.splitlines()[0] == "## AMENDMENT 1: aggregate concordance of the search outputs (reported only)"
    assert md.count("## AMENDMENT 1: aggregate concordance of the search outputs (reported only)") == 1
    for needle in ("REPORTED ONLY", "Pooled over the 3 search seeds", "L2 minus P2HEAD", "Per search seed", "n/a (den 0)", "another draw of LOGS", "SEEN-SCENE"):
        assert needle in md, needle
    assert "PASS" not in md.replace("passed", "")                                           # no verdict language
    assert res["estimator"]["B"] == 10000 and res["estimator"]["seed"] == 20260927 and res["inputs"]["script_sha256"] == sha_file(os.path.join(HERE, "amend1_concordance.py"))


def test_the_cli_exposes_no_flag_that_changes_the_sealed_estimator():
    src = open(os.path.join(HERE, "amend1_concordance.py"), encoding="utf-8").read()
    for flag in ("--seed", "--n-boot", "--B", "--carve"):
        assert f'"{flag}"' not in src                                                       # the estimator and the carve are sealed: reachable from Python (tests) only
