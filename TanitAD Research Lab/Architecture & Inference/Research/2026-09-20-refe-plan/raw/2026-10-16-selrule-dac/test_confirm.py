#!/usr/bin/env python3
"""Tests of confirm.py's wiring on a SYNTHETIC full table (random logits / labels over the REAL 12,146-token, 136-log list; nothing here is P2's data).

    python test_confirm.py [--families]        -> ZZTEST_CONFIRM_PASS <checks>

It proves the gates, the STOP paths and the output schema before the one real read.  The deliberate-regression arms are the gates themselves: a missing
marker, an overlapping RL-val log, a tampered FIT.json, an existing CONFIRM.json, a wrong mean at the table pick and a mis-ordered sub_names must each STOP.
--families additionally runs the family leg (families6.py in the TanitAD venv, ~minutes) on synthetic straight-line poses: wiring only, the values mean nothing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import confirm as CF  # noqa: E402
import selrule as SR  # noqa: E402

N_CHECKS = 0


def chk(cond, msg):
    global N_CHECKS
    N_CHECKS += 1
    if not cond:
        raise AssertionError(msg)


def world(tmp, overlap=0, spec=("V", 0.8), mean_at_pick=0.8882, wrong_names=False, seed=1):
    """A fake P2FULL table + a synthetic FIT.json (+ sidecar) under tmp.  -> (table_dir, fit_path, args-namespace)."""
    toks, tl = CF.load_token_map()
    N, M = len(toks), CF.M
    rng = np.random.default_rng(seed)
    logits = rng.normal(0.5, 2.0, size=(N, M, 6)).astype(np.float32)
    P = SR.RV.sigmoid(logits)
    B = SR.picks(("B",), P)
    sub = np.stack([rng.choice([0.0, 0.5, 1.0], size=(N, M), p=[.1, .05, .85]), rng.choice([0.0, 1.0], size=(N, M), p=[.15, .85]), rng.uniform(0, 1, size=(N, M)),
                    rng.choice([0.0, 1.0], size=(N, M)), rng.choice([0.0, 1.0], size=(N, M), p=[.05, .95]), np.ones((N, M))], -1)
    pdms = SR.RR.pdms_v1(sub[..., 0], sub[..., 1], sub[..., 2], sub[..., 3], sub[..., 4])
    pdms[np.arange(N), B] = mean_at_pick                      # so the "P2 banked 88.82" gate can pass on synthetic data
    steps = (np.arange(1, 9, dtype=np.float32) * 2.0)[None, None, :]
    prop = np.zeros((N, M, 8, 3), np.float32)
    prop[..., 0] = steps + rng.normal(0, 0.3, size=(N, M, 8)).astype(np.float32)
    prop[..., 1] = rng.normal(0, 0.3, size=(N, M, 8)).astype(np.float32)
    names = list(CF.SUB_NAMES)
    if wrong_names:
        names[0], names[1] = names[1], names[0]
    td = os.path.join(tmp, "table")
    os.makedirs(td, exist_ok=True)
    np.savez(os.path.join(td, "table.npz"), token=np.asarray(toks), pdms=pdms, sub=sub, valid=np.ones((N, M), bool), logits=logits, proposals=prop, pick=B,
             sub_names=np.asarray(names), pdms_at_pick_banked=pdms[np.arange(N), B])
    fp = np.asarray([hashlib.sha1(t.encode()).hexdigest() for t in toks])
    np.savez(os.path.join(td, "proposals.npz"), token=np.asarray(toks), fingerprint=fp, sampling=np.array([8.0, 0.5]), proposals=prop, logits=logits, pick=B)
    open(os.path.join(td, "P2FULL_DONE"), "w").write("synthetic\n")
    navlogs = sorted(set(tl.values()))
    fake = [f"synthetic_rlval_log_{i:02d}" for i in range(75)]
    for i in range(overlap):
        fake[i] = navlogs[i]
    fit = {"frozen": True, "tool": "synthetic", "chosen": {"name": SR.name(spec), "spec": list(spec), "delta_vs_B": {"mean": 0.0, "ci95": [0.0, 0.0]}},
           "rl_val": {"logs": fake}, "prereg": {"selrule_py_sha256": "synthetic"}}
    fp_path = os.path.join(tmp, "FIT_synth.json")
    with open(fp_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(fit, f, sort_keys=True)
    with open(fp_path + ".sha256", "w", encoding="utf-8", newline="\n") as f:
        f.write(f"{SR.sha256_file(fp_path)}  FIT_synth.json\n")
    a = types.SimpleNamespace(table_dir=td, table_name="table.npz", proposals_name="proposals.npz", fit=fp_path, out_dir=os.path.join(tmp, "out"), expect_fit_sha256="", no_families=True)
    return td, fp_path, a


def stops(a, needle):
    try:
        CF.full(a)
    except CF.Stop as e:
        return needle in str(e)
    return False


def test_gates_and_stop_paths():
    for label, kw, needle in (("marker missing", {}, "P2FULL_DONE does not exist"), ("RL-val INTERSECT navtest overlap", {"overlap": 2}, "share 2 logs"),
                              ("wrong mean at the table pick", {"mean_at_pick": 0.80}, "mean_PDMS_at_table_pick_not_P2_banked"),
                              ("sub_names mis-ordered", {"wrong_names": True}, "sub_names")):
        with tempfile.TemporaryDirectory() as tmp:
            td, fit_path, a = world(tmp, **{k: v for k, v in kw.items()})
            if label == "marker missing":
                os.remove(os.path.join(td, "P2FULL_DONE"))
            chk(stops(a, needle), f"DELIBERATE REGRESSION [{label}] must STOP with {needle!r}")
    with tempfile.TemporaryDirectory() as tmp:
        td, fit_path, a = world(tmp)
        with open(fit_path, "r+", encoding="utf-8") as f:
            s = f.read().replace('"V"', '"K"')
            f.seek(0)
            f.write(s)
            f.truncate()
        chk(stops(a, "altered"), "DELIBERATE REGRESSION [tampered FIT.json] must STOP (sidecar sha256 differs)")
    with tempfile.TemporaryDirectory() as tmp:
        td, fit_path, a = world(tmp)
        open(os.path.join(td, "P2FULL_FAILED"), "w").write("x")
        chk(stops(a, "P2FULL_FAILED"), "DELIBERATE REGRESSION [P2FULL_FAILED beside DONE] must STOP")
    with tempfile.TemporaryDirectory() as tmp:
        td, fit_path, a = world(tmp)
        a.expect_fit_sha256 = "0" * 64
        chk(stops(a, "--expect-fit-sha256"), "a wrong --expect-fit-sha256 must STOP")


def test_full_run_on_the_synthetic_world():
    with tempfile.TemporaryDirectory() as tmp:
        td, fit_path, a = world(tmp)
        rc = CF.full(a)
        chk(rc == 0, "full() returns 0")
        out = json.load(open(os.path.join(a.out_dir, "CONFIRM.json"), encoding="utf-8"))
        chk(out["disjointness"]["intersection"] == 0 and out["disjointness"]["navtest_logs"] == 136, "disjointness recorded: 136 navtest logs, 0 shared")
        chk(out["result"]["n_tokens"] == 12146 and out["result"]["n_logs"] == 136, "12,146 tokens over 136 logs")
        chk(out["result"]["B_vs_table_pick"]["equal"] == 12146, "B == table pick on every synthetic token (the table pick is B by construction)")
        chk(out["gates"]["mean_PDMS_x100_at_table_pick"] == 88.82 or abs(out["gates"]["mean_PDMS_x100_at_table_pick"] - 88.82) < 1e-6, "gate: mean at the table pick")
        chk(out["outcome"]["outcome"] in ("REFUTED", "NOT PROVEN") or out["outcome"]["outcome"].startswith(("UNRESOLVED", "NOT ADOPTED")) or out["outcome"]["outcome"] == "ADOPT", "an outcome from the committed table")
        chk(out["families"]["computed"] is False, "--no-families -> the family leg is reported NOT computed")
        chk(out["outcome"]["outcome"] != "ADOPT", "families not computed can never read ADOPT")
        chk(out["result"]["per_command"]["available"] and out["result"]["per_command"]["tokens_covered"] == 12146, "per-command splits cover all 12,146 tokens (real export)")
        chk(set(out["result"]["oracles"]) == {"O-DAC", "O-NC", "O-both"}, "three oracle controls")
        chk("STRATEGIC" in open(os.path.join(a.out_dir, "CONFIRM.md"), encoding="utf-8").read() and "PAUSED" in out["four_families"]["strategic"], "STRATEGIC reported PAUSED, never a pass")
        chk(out["provenance"]["tokens_md5"] == CF.TOKENS_MD5 and out["fit_sha256"] == SR.sha256_file(fit_path), "provenance: token-list md5 and FIT sha256 recorded")
        a2 = types.SimpleNamespace(**vars(a))
        chk(stops(a2, "ONE read"), "a SECOND confirm into the same out-dir STOPS (the read is one read)")


def test_family_leg_wiring():
    with tempfile.TemporaryDirectory() as tmp:
        td, fit_path, a = world(tmp)
        a.no_families = False
        rc = CF.full(a)
        out = json.load(open(os.path.join(a.out_dir, "CONFIRM.json"), encoding="utf-8"))
        chk(rc == 0 and out["families"]["computed"] is True, f"family leg ran: {out['families']}")
        chk(out["families"]["components_compared"] and out["families"]["components_compared"] > 0, "family components were compared")
        chk(os.path.exists(os.path.join(a.out_dir, "families", "selrule_B_families.json")) and os.path.exists(os.path.join(a.out_dir, "families", "selrule_chosen_families.json")), "both families JSONs exist")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", action="store_true")
    a = ap.parse_args()
    test_gates_and_stop_paths()
    print("  ok  test_gates_and_stop_paths", flush=True)
    test_full_run_on_the_synthetic_world()
    print("  ok  test_full_run_on_the_synthetic_world", flush=True)
    if a.families:
        test_family_leg_wiring()
        print("  ok  test_family_leg_wiring", flush=True)
    print(f"ZZTEST_CONFIRM_PASS {N_CHECKS} checks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
