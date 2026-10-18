#!/usr/bin/env python3
"""Tests of the P2 held-out driver's GATE LOGIC on a synthetic world (no GPU, no model, no nuPlan).  Every gate has a deliberate-regression arm that must read RED.

    C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe test_p2ho.py        # prints ZZP2HOTEST_PASS n  /  ZZP2HOTEST_FAIL <what>
"""
from __future__ import annotations

import json
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p2ho_driver as D  # noqa: E402

TAG = "unittest"
RNG = np.random.default_rng(7)
N_OK = 0


def check(name, cond):
    global N_OK
    if not cond:
        print(f"ZZP2HOTEST_FAIL {name}")
        sys.exit(1)
    N_OK += 1
    print(f"  ok  {name}")


def make_world(ctx, n=5, bad=None):
    """one unit, n tokens: a synthetic dump + features + scores that are mutually consistent unless `bad` mutates one thing."""
    toks = sorted(f"{i:016x}" for i in RNG.integers(1, 2 ** 40, n))
    u = {"key": "unitkey001", "log": "LOGX", "tokens": toks, "n": n}
    logits = RNG.normal(size=(n, 64, 6)).astype(np.float32) * 2
    pick = D.pick_from_logits(logits)
    P = RNG.normal(size=(n, 64, 8, 3)).astype(np.float32)
    sub = RNG.uniform(0.2, 1.0, size=(n, 64, 6))
    pdms = sub[..., 0] * sub[..., 1] * (5 * sub[..., 2] + 5 * sub[..., 3] + 2 * sub[..., 4]) / 12.0
    feat = RNG.normal(size=(n, 64, 256)).astype(np.float16)
    rule, rep, ck = "navsim_v1", True, D.CKPT
    if bad == "rule":
        rule = "v2_shape"
    if bad == "nan_feat":
        feat[0, 0, 0] = np.nan
    if bad == "nan_cell":
        sub[1, 3, 2] = np.nan
    if bad == "wrong_pick":
        pick = (pick + 1) % 64
    if bad == "wrong_formula":
        pdms = pdms * 0.99
    if bad == "missing_token":
        toks_dump = toks[:-1]
    else:
        toks_dump = toks
    k = len(toks_dump)
    np.savez(D.dump_p(ctx, u), token=np.array(toks_dump), fingerprint=np.array(["fp"] * k), proposals=P[:k], proposals20=np.zeros((k, 64, 20, 3), np.float32), logits=logits[:k],
             pick=pick[:k].astype(np.int64), log=np.array(["LOGX"] * k), select=np.array("best"), rule=np.array(rule), repair_last_heading=np.array(rep), ckpt=np.array(ck),
             sampling=np.array([8, 0.5]))
    np.savez(D.feat_p(ctx, u), token=np.array(toks_dump), log=np.array(["LOGX"] * k), prehead_features=feat[:k])
    json.dump({"misses": 0}, open(os.path.splitext(D.dump_p(ctx, u))[0] + ".report.json", "w"))
    np.savez(D.score_p(ctx, u), token=np.array(toks_dump), sub=sub[:k], pdms=pdms[:k], valid_token=np.ones(k, bool), sub_names=np.array(D.SUB))
    return u


def main():
    a = D.make_parser().parse_args(["--tag", TAG, "--device", "cpu"])
    ctx = D.Ctx(a)
    try:
        # G-DUMP gate: green on a consistent world, RED on each mutation
        u = make_world(ctx)
        check("gate_unit green on a consistent dump", D.gate_unit(ctx, u)["PASS"])
        for bad in ("rule", "nan_feat", "missing_token"):
            u = make_world(ctx, bad=bad)
            check(f"gate_unit RED on mutation {bad}", not D.gate_unit(ctx, u)["PASS"])
        # table gates G3 / G4
        u = make_world(ctx)
        plan = {"tokens": u["tokens"], "token_log": {t: "LOGX" for t in u["tokens"]}}
        G, out, _pr, _ft = D.stage_table(ctx, plan, [u])
        check("stage_table green on a consistent world", G["PASS"] and G["G4a_pick_from_logits_f64_equal"] == u["n"] and G["G4b_pdms_formula_max_abs_diff"] < 1e-9)
        with np.load(out) as T:
            check("table schema = table_p2_full.npz schema", {"token", "log", "fingerprint", "pdms", "sub", "valid", "logits", "proposals", "pick"} <= set(T.files)
                  and T["pdms"].shape == (u["n"], 64) and T["sub"].shape == (u["n"], 64, 6) and T["proposals"].shape == (u["n"], 64, 8, 3) and list(T["token"]) == sorted(T["token"]))
        check("DDC-in-product control does NOT reproduce pdms (the formula test can fail)", G["G4b_ddc_in_product_control_max_abs_diff"] > 1e-3)
        for bad, key in (("nan_cell", "G3_all_valid"), ("wrong_pick", "G4a_pick_from_logits_f64_equal"), ("wrong_formula", "G4b_pdms_formula_max_abs_diff")):
            u = make_world(ctx, bad=bad)
            plan = {"tokens": u["tokens"], "token_log": {t: "LOGX" for t in u["tokens"]}}
            G, *_ = D.stage_table(ctx, plan, [u])
            check(f"stage_table RED on mutation {bad}", not G["PASS"])
        # units: a log is never split, key depends on the token list
        plan = {"tokens": [f"t{i:03d}" for i in range(7)], "token_log": {f"t{i:03d}": ("A" if i < 4 else "B") for i in range(7)}}
        us = D.make_units(plan, 250)
        check("make_units: one unit per log, deterministic keys", [x["log"] for x in us] == ["A", "B"] and us == D.make_units(plan, 250))
        plan2 = dict(plan, tokens=plan["tokens"] + ["t999"], token_log=dict(plan["token_log"], t999="A"))
        check("make_units: a log that gains a token gets a NEW key", D.make_units(plan2, 250)[0]["key"] != us[0]["key"] and D.make_units(plan2, 250)[1]["key"] == us[1]["key"])
        # disk guard: stops cleanly below the floor, passes above it
        real = D.C.free_gb
        D.C.free_gb = lambda p: 1.0
        try:
            D.disk_guard(ctx, "x")
            check("disk_guard raises Stop below 5 GiB", False)
        except D.Stop as e:
            check("disk_guard raises Stop below 5 GiB (lowdisk)", str(e).startswith("lowdisk"))
        D.C.free_gb = lambda p: 100.0
        realmb = D.C.free_mb
        D.C.free_mb = lambda: 20000
        try:
            D.disk_guard(ctx, "x")
            check("disk_guard passes with space and RAM", True)
        finally:
            D.C.free_gb, D.C.free_mb = real, realmb
        # sets index: rank-0 and rank-1 sets of one token share the row
        keys = [("L", "a", 0, 0), ("L", "a", 0, 1), ("L", "b", 0, 0), ("L", "zz", 0, 0)]
        real_k = D.PF.heldout_keys
        D.PF.heldout_keys = lambda: (keys, {})
        try:
            idx = D.sets_index(ctx, {"a": 0, "b": 1}, os.path.join(ctx.work, "final", "idx.npz"))
        finally:
            D.PF.heldout_keys = real_k
        z = np.load(os.path.join(ctx.work, "final", "idx.npz"))
        check("sets_index: twin sets share their token's row, unplanned token -> -1", list(z["row"]) == [0, 0, 1, -1] and idx["n_sets_with_row"] == 3)
        # the lock constants of the landed module are the landed ones
        check("landed lock path + owner override", D.C.LOCK.endswith("devbox_gpu.lock") and D.C.OWNER == "p2heldout_table")
    finally:
        shutil.rmtree(ctx.wd, ignore_errors=True)
        shutil.rmtree(ctx.work, ignore_errors=True)
    print(f"ZZP2HOTEST_PASS {N_OK}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
