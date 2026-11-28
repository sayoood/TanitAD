#!/usr/bin/env python3
"""ENS-SCORE controls that read the run's own artifacts (PREREG_ENSSCORE.md section 6). NO PDMS is read here.

    ens_ctrl.py --ens-dump <ens.npz> --ens-seam <seam.npz> --ens-props <seam.proposals.npz>
                --banked-seam <primary's banked seam.npz> --banked-props <primary's banked proposals.npz>
                --banked-seam-b <member's banked seam.npz> --banked-props-b <member's banked proposals.npz>
                --out <ctrl.json> [--expect-n 12146] [--det-dump <a second ENS dump, same primary + member>]

CTRL-SET   tokens of dump, seam and props identical; inside both banked sets; n == --expect-n and == the primary's banked set when given.
CTRL-ID    (a) k_id (the primary ensembled with ITSELF) == k_ship on every token; (b) k_b_own_id (the member's re-score of its OWN set
           with its own context) == k_b_own on every token. Max |logit difference| of both re-score paths reported.
CTRL-DET   the primary vs its banked seam, bit for bit: k_ship == banked pick; ship row == banked seam row; the run's 64 planned props ==
           banked proposals; the run's forward logits == banked logits.
CTRL-DET-B the member's own forward vs ITS banked seam: max |d logits| <= 1e-4 and max |d props| <= 1e-4 m on every token (bitwise, pick
           and ship-row agreement reported). It proves the member received the primary's exact inputs (image, ego + its own yaw, nav, calib).
CTRL-RED   member B misaligned by one proposal index (k_red) differs from k_ens on >= 1 % of tokens: a path that ignores B fails.
CTRL-REACH (i) the dumped device aggregate a_ens == an independent numpy float32 recompute of the navsim_v1 rule on (sig(s_a)+sig(s_b))/2
           to <= 1e-6; (ii) argmax a_ens == k_ens on every token; (iii) on >= 99 % of tokens a_ens differs from the primary-ALONE aggregate
           by > 1e-5 somewhere in the set (the member term is present). Removing the member term fails (iii).
CTRL-SUB   (submitted = the primary's PLANNED proposal k_ens) seam row == props[k_ens] == to_navsim(pick20) bit for bit; props pick == k_ens.
CTRL-SEAM  report rows == n, misses 0, rule navsim_v1, A7 on, frame control (full set: 6.18063877792441 m), seam log device=cuda.
CTRL-CFG   the launcher's CTRL-CFG passed (member config == primary config) and both or neither model used the ego-yaw shim.
CTRL-CODE  the dump's recorded sha256 of ens_lib.py / ens_select.py == the files next to this script.
CTRL-DET-RUN (--det-dump) every shared token: k_ship, k_ens, k_b_own, s_a, s_b, a_ens bit for bit; shared >= min(50, n).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.environ.get("TANITAD_REFE_PKG", "F:/TanitAD_p5cread_tip/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan")
V1_W = (5.0, 5.0, 2.0)          # refe/planner.py:520 (navsim_v1: NC x DAC x (5 EP + 5 TTC + 2 C) / 12), re-typed ON PURPOSE: an independent check


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def idx(tokens):
    return {str(t): i for i, t in enumerate(tokens)}


def sig(x):
    x = np.asarray(x, np.float32)
    with np.errstate(over="ignore"):                  # exp(-x) -> inf for x << 0 gives exactly 0.0, which is the right limit
        return (np.float32(1.0) / (np.float32(1.0) + np.exp(-x))).astype(np.float32)


def agg_np(p):
    p = np.asarray(p, np.float32)
    w = np.asarray(V1_W, np.float32)
    return (p[..., 0] * p[..., 1] * (p[..., 2] * w[0] + p[..., 3] * w[1] + p[..., 4] * w[2]) / w.sum()).astype(np.float32)


def seam_mod():
    import importlib.util
    spec = importlib.util.spec_from_file_location("seam_c", os.path.join(PKG, "eval", "refe_navtest_seam.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    for k in ("--ens-dump", "--ens-seam", "--ens-props", "--banked-seam", "--banked-props", "--banked-seam-b", "--banked-props-b", "--out"):
        ap.add_argument(k, required=True)
    ap.add_argument("--expect-n", type=int, default=0)
    ap.add_argument("--det-dump", default=None)
    a = ap.parse_args()
    d = np.load(a.ens_dump, allow_pickle=False)
    ts = np.load(a.ens_seam, allow_pickle=False)
    tp = np.load(a.ens_props, allow_pickle=False)
    bs, bp = np.load(a.banked_seam, allow_pickle=False), np.load(a.banked_props, allow_pickle=False)
    bsb, bpb = np.load(a.banked_seam_b, allow_pickle=False), np.load(a.banked_props_b, allow_pickle=False)
    meta = json.loads(str(d["meta"]))
    toks = [str(t) for t in ts["token"]]
    di, pi, tsi = idx(d["token"]), idx(tp["token"]), idx(ts["token"])
    bsi, bpi, bsbi, bpbi = idx(bs["token"]), idx(bp["token"]), idx(bsb["token"]), idx(bpb["token"])
    out = {"meta": {k: meta[k] for k in ("n", "code", "member", "wrapped_rc", "t_ens_mean_s", "cuda_max_mem_mb") if k in meta}}
    C = {}
    # CTRL-SET
    inside = all(t in bsi and t in bpi and t in bsbi and t in bpbi for t in toks)
    set_ok = (set(toks) == set(di) == set(pi)) and inside and len(toks) == len(set(toks))
    if a.expect_n:
        set_ok = set_ok and len(toks) == a.expect_n and set(toks) == set(bsi) == set(bsbi)
    C["CTRL-SET"] = {"pass": bool(set_ok), "n_seam": len(toks), "n_dump": len(di), "n_props": len(pi), "n_banked": len(bsi), "n_banked_b": len(bsbi),
                     "expect_n": a.expect_n}
    if not set_ok:
        toks = [t for t in toks if t in di and t in pi and t in bsi and t in bpi and t in bsbi and t in bpbi]
    n = len(toks)
    D = np.array([di[t] for t in toks])
    P = np.array([pi[t] for t in toks])
    S = np.array([tsi[t] for t in toks])
    BS, BP, BSB, BPB = (np.array([m[t] for t in toks]) for m in (bsi, bpi, bsbi, bpbi))
    g = {k: d[k][D] for k in ("k_ship", "k_ens", "k_id", "k_red", "k_a_only", "k_b_only", "k_b_own", "k_b_own_id")}
    # CTRL-ID
    C["CTRL-ID"] = {"pass": bool((g["k_id"] == g["k_ship"]).all() and (g["k_b_own_id"] == g["k_b_own"]).all()),
                    "a_self_ensemble_match": int((g["k_id"] == g["k_ship"]).sum()), "a_rescore_max_abs_logit_diff": float(d["id_maxdiff_a"][D].max()),
                    "b_own_rescore_match": int((g["k_b_own_id"] == g["k_b_own"]).sum()), "b_rescore_max_abs_logit_diff": float(d["id_maxdiff_b"][D].max()),
                    "n": n}
    # CTRL-DET (primary, bitwise)
    pick_eq = bp["pick"][BP] == g["k_ship"]
    ship_eq = np.array([np.array_equal(d["ship8"][i], bs["poses"][j]) for i, j in zip(D, BS)])
    props_eq = np.array([np.array_equal(tp["proposals"][i], bp["proposals"][j]) for i, j in zip(P, BP)])
    logit_eq = np.array([np.array_equal(tp["logits"][i], bp["logits"][j]) for i, j in zip(P, BP)])
    C["CTRL-DET"] = {"pass": bool(pick_eq.all() and ship_eq.all() and props_eq.all() and logit_eq.all()), "pick_eq": int(pick_eq.sum()),
                     "ship_row_eq": int(ship_eq.sum()), "props_eq": int(props_eq.sum()), "logits_eq": int(logit_eq.sum()),
                     "props_max_abs_diff": float(max(np.abs(tp["proposals"][i].astype(np.float64) - bp["proposals"][j]).max() for i, j in zip(P, BP))),
                     "logits_max_abs_diff": float(max(np.abs(tp["logits"][i].astype(np.float64) - bp["logits"][j]).max() for i, j in zip(P, BP))), "n": n}
    # CTRL-DET-B (the member's own forward vs its banked seam)
    lb = np.array([np.abs(d["s_b_fwd"][i].astype(np.float64) - bpb["logits"][j]).max() for i, j in zip(D, BPB)])
    pb_ = np.array([np.abs(d["props8_b"][i].astype(np.float64) - bpb["proposals"][j]).max() for i, j in zip(D, BPB)])
    C["CTRL-DET-B"] = {"pass": bool(lb.max() <= 1e-4 and pb_.max() <= 1e-4), "logits_max_abs_diff": float(lb.max()), "props_max_abs_diff": float(pb_.max()),
                       "logits_bitwise_tokens": int((lb == 0).sum()), "props_bitwise_tokens": int((pb_ == 0).sum()),
                       "pick_eq_banked": int((bpb["pick"][BPB] == g["k_b_own"]).sum()),
                       "ship_row_eq_banked": int(sum(np.array_equal(d["ship8_b"][i], bsb["poses"][j]) for i, j in zip(D, BSB))), "n": n}
    # CTRL-RED
    sw = int((g["k_red"] != g["k_ens"]).sum())
    C["CTRL-RED"] = {"pass": bool(n > 0 and sw >= math.ceil(0.01 * n)), "k_red_ne_k_ens": sw, "need_at_least": math.ceil(0.01 * n) if n else None,
                     "what": "member B's probabilities rolled by one proposal index before the mean"}
    # CTRL-REACH
    sa, sb, ae = d["s_a"][D], d["s_b"][D], d["a_ens"][D]
    a_np = agg_np((sig(sa) + sig(sb)) / np.float32(2.0))
    a_a = agg_np(sig(sa))
    r1 = float(np.abs(ae.astype(np.float64) - a_np).max()) if n else None
    r2 = int((ae.argmax(-1) == g["k_ens"]).sum())
    r3 = int((np.abs(ae.astype(np.float64) - a_a).max(-1) > 1e-5).sum())
    C["CTRL-REACH"] = {"pass": bool(n > 0 and r1 <= 1e-6 and r2 == n and r3 >= math.ceil(0.99 * n)), "a_ens_vs_numpy_max_abs": r1,
                       "argmax_a_ens_eq_k_ens": r2, "tokens_member_term_present": r3, "need_at_least": math.ceil(0.99 * n) if n else None, "n": n}
    # CTRL-SUB
    sm = seam_mod()
    row_eq = np.array([np.array_equal(ts["poses"][si], tp["proposals"][pj][k]) for si, pj, k in zip(S, P, g["k_ens"])])
    pick_is = tp["pick"][P] == g["k_ens"]
    p20 = bool(sm.pick20_equals_seam(tp["pick20"][P], ts["poses"][S]))
    C["CTRL-SUB"] = {"pass": bool(row_eq.all() and pick_is.all() and p20), "row_eq_planned_props_k_ens": int(row_eq.sum()),
                     "props_pick_eq_k_ens": int(pick_is.sum()), "pick20_equals_seam": p20, "n": n}
    # CTRL-SEAM
    rep = json.load(open(os.path.splitext(a.ens_seam)[0] + ".report.json", encoding="utf-8"))
    lg = os.path.splitext(a.ens_dump)[0] + ".seam.log"
    cuda = os.path.isfile(lg) and "device=cuda" in open(lg, encoding="utf-8", errors="replace").read()
    fc = float(rep["frame_control"]["max_m"])
    fc_ok = abs(fc - 6.18063877792441) <= 1e-6 if a.expect_n == 12146 else True
    C["CTRL-SEAM"] = {"pass": bool(rep.get("rows") == n and rep.get("misses") == 0 and rep.get("rule") == "navsim_v1" and rep.get("repair_last_heading") is True
                                   and fc_ok and cuda), "rows": rep.get("rows"), "misses": rep.get("misses"), "frame_control_max_m": fc,
                      "frame_control_declared_ok": fc_ok, "seam_log_device_cuda": cuda, "ckpt": rep.get("ckpt"), "wrapped_rc": meta.get("wrapped_rc")}
    # CTRL-CFG
    mem = meta.get("member") or {}
    y = mem.get("yaw", "absent")
    C["CTRL-CFG"] = {"pass": bool(mem.get("cfg_equal") is True and y != "absent"), "member": mem}
    # CTRL-CODE
    code = meta.get("code") or {}
    mine = {f: sha256_file(os.path.join(HERE, f)) for f in ("ens_lib.py", "ens_select.py")}
    C["CTRL-CODE"] = {"pass": all(code.get(f) == h for f, h in mine.items()), "dump": {f: code.get(f) for f in mine}, "staged": mine}
    # CTRL-DET-RUN
    if a.det_dump:
        e = np.load(a.det_dump, allow_pickle=False)
        ei = idx(e["token"])
        sh = [t for t in toks if t in ei]
        ok = len(sh) >= min(50, n)
        bad = {}
        for k in ("k_ship", "k_ens", "k_b_own", "s_a", "s_b", "a_ens"):
            m = sum(not np.array_equal(d[k][di[t]], e[k][ei[t]]) for t in sh)
            bad[k] = int(m)
            ok = ok and m == 0
        C["CTRL-DET-RUN"] = {"pass": bool(ok), "shared_tokens": len(sh), "mismatching_tokens": bad, "other": os.path.abspath(a.det_dump)}
    out["descriptive_no_pdms"] = {"switch_k_ens_ne_k_ship": int((g["k_ens"] != g["k_ship"]).sum()),
                                  "member_alone_ne_k_ship": int((g["k_b_only"] != g["k_ship"]).sum()), "n": n}
    out["controls"] = C
    out["all_pass"] = all(v["pass"] for v in C.values())
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1)
    print("ZZENS_CTRL " + " ".join(f"{k}={'PASS' if v['pass'] else 'FAIL'}" for k, v in C.items()) + f" all_pass={out['all_pass']}", flush=True)
    return 0 if out["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
