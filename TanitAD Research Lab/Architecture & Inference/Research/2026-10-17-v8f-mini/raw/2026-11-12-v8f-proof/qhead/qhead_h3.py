#!/usr/bin/env python
"""qhead_h3.py -- card PREREG_V8F_PROOF_ADD42, control h3 (outside the rollout), for ONE QH arm:

  at the first tick (k == 0), on the ticks the head calls present, the gap the rule used in the QH run (rule log, g_src) equals min(g_head, q_head) recomputed from the banked
  states of the evaluation windows (the eval state file of the checkpoint, the banked head tensors, the quantile head tensors), within 1e-4 m.

  python qhead_h3.py --arm LSQH_s0 --rule-log rule_log_LSQH_s0.npz --join join_eval141.npz --eval-states LS_s0_eval.npz --banked-head LS_s0_eval_lead.npz.head.npz
                     --qhead LS_s0.qhead.npz --out H3_LSQH_s0.json [--side rule_log_LSQH_s0_qh.npz] [--tol 1e-4]

Three readings, all printed: (a) in the rollout, g_src equals the logged min(g_head, q_head) of the side file; (b) the card form, g_src against the recomputation from the banked states (this is PASS);
(c) beside, naming nothing: the rollout's head gap against the recomputed head gap (the eval state file holds float16 states, the rollout float32 ones, so (c) shows how much of (b) is that rounding).
Also written, for the read: the mean amount taken off the read at the first tick (g_head - g_used) and the coverage on the evaluation windows (the share of first-tick present windows with a labelled
gap whose q_head lies above the label gap). Tier T1-kin for the rule log, T0 for the recomputation; none is driving performance.
Exit 0 = PASS, 2 = a failed reading, 3 = refused.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import qhead_model as QM  # noqa: E402

GAP_MAX_M = 60.0
HIST = 5


def history_rows(clip_idx, t, want_clip, want_t, hist):
    """For each wanted (clip, t): the rows of the frames t-hist+1..t in the state arrays, -1 where absent. -> [n, hist]"""
    key = {(int(c), int(g)): i for i, (c, g) in enumerate(zip(clip_idx, t))}
    out = np.full((len(want_t), hist), -1, np.int64)
    for i, (c, g) in enumerate(zip(want_clip, want_t)):
        row = [key.get((int(c), int(g) - (hist - 1 - j)), -1) for j in range(hist)]
        if all(r >= 0 for r in row):
            out[i] = row
    return out


def check(arm, rule_log, side, join, eval_states, banked, qtens, tol=1e-4):
    ci, k, j = np.asarray(rule_log["ci"], np.int64), np.asarray(rule_log["k"], np.int64), np.asarray(rule_log["j"], np.int64)
    present = np.asarray(rule_log["present"]).astype(bool)
    g_src = np.asarray(rule_log["g_src"], np.float64)
    n_rows = len(ci)
    r = {"arm": arm, "n_rule_log_rows": int(n_rows), "n_side_rows": int(len(side["g_head"])), "tol_m": tol}
    if len(side["g_head"]) != n_rows:
        r.update(PASS=False, why="the side file has %d rows, the rule log %d: not row for row" % (len(side["g_head"]), n_rows))
        return r
    first = (k == 0) & present
    r["n_first_tick_present"] = int(first.sum())
    if first.sum() == 0:
        r.update(PASS=False, why="no first-tick present row")
        return r
    gh, qh, gu = (np.asarray(side[x], np.float64)[first] for x in ("g_head", "q_head", "g_used"))
    gs = g_src[first]
    ra = {"max_abs_g_src_vs_g_used": float(np.max(np.abs(gs - gu))), "max_abs_g_used_vs_min": float(np.max(np.abs(gu - np.minimum(gh, qh))))}
    r["a_in_rollout"] = dict(ra, PASS=bool(max(ra.values()) <= tol))
    names = np.asarray(join["clip"]).astype(str)
    s_names = np.asarray(eval_states["clips"]).astype(str)
    s_index = {n: i for i, n in enumerate(s_names)}
    cl = np.array([s_index.get(names[c], -1) for c in ci[first]], np.int64)
    ok_clip = cl >= 0
    rows = np.full((int(first.sum()), HIST), -1, np.int64)
    if ok_clip.any():
        rows[ok_clip] = history_rows(eval_states["clip_idx"], eval_states["t"], cl[ok_clip], j[first][ok_clip], HIST)
    have = (rows >= 0).all(1)
    r["n_recomputed"], r["n_without_history_or_clip"] = int(have.sum()), int((~have).sum())
    if have.sum() == 0:
        r.update(PASS=False, why="no first-tick row could be matched to the eval state file (clip name or frame mapping)")
        return r
    z = np.asarray(eval_states["Z"])[rows[have]].astype(np.float32)
    g_re = QM.functional_forward(banked, z)[:, 1] * QM.GAP_SCALE_M
    q_re = QM.functional_forward(qtens, z)[:, 0] * QM.GAP_SCALE_M
    u_re = np.minimum(g_re, q_re)
    rb = {"max_abs_g_src_vs_recomputed_min": float(np.max(np.abs(gs[have] - u_re))), "mean_abs": float(np.mean(np.abs(gs[have] - u_re)))}
    r["b_card_form"] = dict(rb, PASS=bool(rb["max_abs_g_src_vs_recomputed_min"] <= tol))
    r["c_beside_head_gap_rollout_vs_recomputed_max_abs_m"] = float(np.max(np.abs(gh[have] - g_re)))
    r["c_beside_quantile_rollout_vs_recomputed_max_abs_m"] = float(np.max(np.abs(qh[have] - q_re)))
    # beside, naming nothing
    off = np.asarray(join["off"], np.int64)
    lab_row = off[ci[first]] + j[first]
    valid, lg = np.asarray(join["lead_valid"]), np.asarray(join["lead_gap"], np.float64)
    lab = np.where(valid[lab_row] == 1, lg[lab_row], np.nan)
    lab_ok = np.isfinite(lab) & (lab >= 0) & (lab <= GAP_MAX_M)
    r["beside"] = {"mean_amount_taken_off_read_m": float(np.mean(gh - gu)), "n_labelled_first_tick": int(lab_ok.sum()),
                   "coverage_eval_windows_share_q_above_label": float(np.mean(qh[lab_ok] > lab[lab_ok])) if lab_ok.any() else None,
                   "share_first_tick_cut": float(np.mean(gu < gh))}
    r["PASS"] = bool(r["a_in_rollout"]["PASS"] and r["b_card_form"]["PASS"])
    return r


def main(argv=None):
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--rule-log", required=True)
    ap.add_argument("--side", default="")
    ap.add_argument("--join", required=True)
    ap.add_argument("--eval-states", required=True)
    ap.add_argument("--banked-head", required=True)
    ap.add_argument("--qhead", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tol", type=float, default=1e-4)
    a = ap.parse_args(argv)
    side_p = a.side or (a.rule_log[:-4] if a.rule_log.endswith(".npz") else a.rule_log) + "_qh.npz"
    for p in (a.rule_log, side_p, a.join, a.eval_states, a.banked_head, a.qhead):
        if not os.path.isfile(p):
            print("REFUSED: no such file", p)
            return 3
    if QM.is_eval_path(a.qhead):
        print("REFUSED: a head file of the evaluation split")
        return 3
    side = dict(np.load(side_p, allow_pickle=False))
    rl = dict(np.load(a.rule_log, allow_pickle=False))
    join = dict(np.load(a.join, allow_pickle=True))
    es = dict(np.load(a.eval_states, allow_pickle=False))
    bz = np.load(a.banked_head, allow_pickle=False)
    qz = np.load(a.qhead, allow_pickle=False)
    banked = {kk: bz[kk] for kk in QM.TENSOR_KEYS}
    qt = {kk: qz[kk] for kk in QM.TENSOR_KEYS}
    r = check(a.arm, rl, side, join, es, banked, qt, a.tol)
    r["files_sha256"] = {p.replace("\\", "/"): QM.sha256_file(p) for p in (a.rule_log, side_p, a.join, a.eval_states, a.banked_head, a.qhead)}
    r["tier"] = "T1-kin (rule log) against T0 (recomputation from logged states); never driving performance"
    QM.dump_json(a.out, r)
    print("h3 %s: %s | %s" % (a.arm, "PASS" if r["PASS"] else "FAIL", json.dumps({k: r[k] for k in r if k in ("a_in_rollout", "b_card_form", "n_first_tick_present", "n_recomputed", "why")})[:400]))
    return 0 if r["PASS"] else 2


if __name__ == "__main__":
    sys.exit(main())
