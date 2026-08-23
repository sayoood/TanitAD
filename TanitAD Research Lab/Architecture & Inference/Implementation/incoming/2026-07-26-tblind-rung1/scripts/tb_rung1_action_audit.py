#!/usr/bin/env python3
"""RUNG 1 — the ACTION AUDIT. ~3 GPU-min on pod2. Reads WHAT THE LOOP FED.

The mechanism separation needs the action tensor itself, and ``bi_run._run_arms``
stores ``psi`` / ``pred_speed`` per arm but **not** ``fed_actions``. Two things are
therefore done here on a 40-episode subset of the SAME ordered val split (a
strict subset of the sweep's 599 windows, verified by ``eid``/``t0``):

1. ⛔ **The reconstruction gate.** ``blindimag.reconstruct_kinematic_actions``
   recovers (steer, accel) from ``psi``/``pred_speed``/``v_last``. It is pinned on
   a CPU fixture in ``test_blindimag.py``; here it is held against the REAL
   model's ``fed_actions`` before it is used to characterise all 599 windows.
   Both directions: the correct input must match, a deliberately wrong one must
   not.
2. **Dense ``fed_actions`` for the FILTERED arms**, where the reconstruction does
   not apply by construction (a filter is exactly the difference between the
   action the inverse produced and the action fed).

``bi_run._prepare`` is reused verbatim — encode, window build and parity block
are the sweep's own.

Usage (pod2):
    PYTHONPATH=/root/bi:/root/taniteval:/root/TanitAD/stack:/root/TanitAD/stack/scripts \
    OMP_NUM_THREADS=8 python3 tb_rung1_action_audit.py \
        --out /root/tbr1/perwindow --episodes 40 --kmax 185
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

for _p in ("/root/bi", "/root/taniteval", "/root/TanitAD/stack",
           "/root/TanitAD/stack/scripts"):
    if Path(_p).is_dir() and _p not in sys.path:
        sys.path.insert(0, _p)

import torch                                          # noqa: E402

import bi_run as R                                    # noqa: E402
from taniteval import blindimag as bi                 # noqa: E402

STR = "str"

#: (name, state_source, action_source, readout_level). a-arms only: the audit
#: characterises the ACTION, and the action does not depend on the comparator.
AUDIT_ARMS = [
    ("own",         "imagination", "own_kinematic",            STR),
    ("hold",        "imagination", "hold_last",                STR),
    ("true",        "imagination", "true_future",              STR),
    ("gtkin",       "imagination", "gt_kinematic",             STR),
    ("own_op",      "imagination", "own_kinematic",            "op"),
    ("blend0.5",    "imagination", "own_kinematic|blend=0.5",  STR),
    ("ema0.8",      "imagination", "own_kinematic|ema=0.8",    STR),
    ("every5",      "imagination", "own_kinematic|every=5",    STR),
    ("accelclip0.3", "imagination", "own_kinematic|accel_clip=0.3", STR),
    ("own_frozen",  "frozen_last", "own_kinematic",            STR),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", default="flagship-30k")
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--kmax", type=int, default=185)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--enc-batch", type=int, default=32)
    a = ap.parse_args()
    torch.manual_seed(0)
    outd = Path(a.out)
    outd.mkdir(parents=True, exist_ok=True)
    dev = "cuda"

    h, eps, cache, poses_by_ep, recs, parity, t_enc = R._prepare(a, dev)
    readouts = h["grounding"].step
    sro = h["step_readout"]
    k = a.kmax

    store = {n: {"fed": [], "psi": [], "spd": [], "dp": []} for n, *_ in AUDIT_ARMS}
    eid, t0s, vlast = [], [], []
    t_roll = time.time()
    for b in bi.batch_windows(recs, poses_by_ep, k, batch=a.batch):
        eid += b["eid"]
        t0s += b["t0"]
        vlast.append(b["v_last"].cpu().float())
        gtd = b["gt_dpose"].to(b["states"].device)
        vl = b["v_last"].to(b["states"].device)
        for name, ss, asrc, lvl in AUDIT_ARMS:
            r = bi.blind_rollout(
                h["model"].predictor, b["states"], b["actions"],
                sro if lvl == "op" else readouts[lvl], k,
                state_source=ss, action_source=asrc,
                future_actions=b["future_actions"], obs_states=b["obs_states"],
                gt_step_dpose=gtd, v_last=vl)
            store[name]["fed"].append(r["fed_actions"].cpu().float())
            store[name]["psi"].append(r["psi"].cpu().float())
            store[name]["spd"].append(r["pred_speed"].cpu().float())
            store[name]["dp"].append(r["step_dpose"].cpu().float())
    t_roll = time.time() - t_roll
    out = {n: {kk: torch.cat(vv) for kk, vv in d.items()}
           for n, d in store.items()}
    v_last = torch.cat(vlast)

    # ---- ⛔ THE RECONSTRUCTION GATE, both directions --------------------- #
    gate = {}
    for name in ("own", "own_op", "gtkin", "own_frozen"):
        st, ac = bi.reconstruct_kinematic_actions(out[name]["psi"],
                                                  out[name]["spd"], v_last)
        fed = out[name]["fed"]
        d_st = float((st[:, :k - 1] - fed[:, :k - 1, 0]).abs().max())
        d_ac = float((ac[:, :k - 1] - fed[:, :k - 1, 1]).abs().max())
        gate[name] = {"max_abs_diff_steer": d_st, "max_abs_diff_accel": d_ac,
                      "reproduces": bool(d_st < 1e-5 and d_ac < 1e-5)}
    # `gtkin` derives its action from the TRUE dposes, so the reconstruction
    # (which reads the model's own decoded motion) MUST NOT reproduce it. That
    # is the failing direction, on real data.
    st, ac = bi.reconstruct_kinematic_actions(out["own"]["psi"],
                                              out["own"]["spd"] * 1.5, v_last)
    gate["deliberate_failure_wrong_speed"] = {
        "max_abs_diff_accel": float((ac[:, :k - 1]
                                     - out["own"]["fed"][:, :k - 1, 1]).abs().max()),
        "must_be_large": True}
    gate["GATE_PASS"] = bool(
        gate["own"]["reproduces"] and gate["own_op"]["reproduces"]
        and gate["own_frozen"]["reproduces"]
        and not gate["gtkin"]["reproduces"]
        and gate["deliberate_failure_wrong_speed"]["max_abs_diff_accel"] > 1e-3)

    meta = {"block": "taniteval.blindimag/rung1_action_audit",
            "arm_ckpt": a.arm, "ckpt_step": h.get("step"), "kmax": k,
            "stride": a.stride, "episodes_requested": a.episodes,
            "n_windows": len(recs), "val_parity": parity,
            "wheelbase_used": bi.WHEELBASE, "steer_clamp": bi.STEER_CLAMP,
            "accel_clamp": bi.ACCEL_CLAMP, "dt_s": bi.DT,
            "arms": {n: {"state_source": s, "action_source": asrc,
                         "readout_level": lv} for n, s, asrc, lv in AUDIT_ARMS},
            "reconstruction_gate": gate,
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "seconds_encode": round(t_enc, 1),
            "seconds_rollout": round(t_roll, 1)}
    torch.save({"fed_actions": {n: d["fed"] for n, d in out.items()},
                "psi": {n: d["psi"] for n, d in out.items()},
                "pred_speed": {n: d["spd"] for n, d in out.items()},
                "step_dpose": {n: d["dp"] for n, d in out.items()},
                "v_last": v_last, "eid": eid, "t0": t0s, "meta": meta},
               str(outd / f"action_audit_K{k}.pt"))
    (outd / f"action_audit_meta_K{k}.json").write_text(
        json.dumps(meta, indent=2, default=str), encoding="utf-8")
    print("RECONSTRUCTION GATE:", json.dumps(gate, indent=1), flush=True)
    print(f"GATE_PASS = {gate['GATE_PASS']}  n_windows={len(recs)}", flush=True)
    return 0 if gate["GATE_PASS"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
