"""check_prev_read.py -- card PREREG_V8F_PROOF_ADD50 (GAPRATE10K), pod side, CPU: the identities of one run of the form RRd and the earlier reads recomputed by an
independent path. Reads the rule log and its side file (`<rule log minus .npz>_rd.npz`), the reader checkpoint, the cache files of the evaluation clips and the join.
  i1  the side file holds the columns of the form (p_prev, g_prev, ok_prev, have, c_fed, id0) row for row with the rule log
  i2  the gather at K = 0 equals the read of the batch tokens at every tick (id0 <= TOL; an unequal history flag reads inf)
  i3  the closing speed in the rule log (c_src) equals the fed value at every tick; it is exactly 0 where no closing read exists; no value came from the labels
  i4  recomputed from the cache file of the clip by the reader's own clip path (tok_reader.read_clip, the path of the banked rows): the read at the tick (p, g_read)
      and the earlier read at cache step j-K (p_prev, g_prev, ok_prev) agree with the logged ones
  i5  the fed value equals slope * (g_read - g_prev) / (0.2 K) where `have`, with `have` = reader present at both ends
Exit code 0 only if every identity holds. Names nothing; reads no outcome of the card. Tier T0 (logged frames)."""
import argparse
import json
import os
import sys

import numpy as np

TOL_ID, TOL_RE = 1e-4, 1e-3
P_MIN, DT_S = 0.5, 0.2


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--rule-log", required=True)
    ap.add_argument("--reader", required=True)
    ap.add_argument("--eval-cache", required=True)
    ap.add_argument("--join", required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--build", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    sys.path.insert(0, a.build)
    import torch
    import tok_reader as TR
    L = np.load(a.rule_log, allow_pickle=True)
    S = np.load(a.rule_log[:-4] + "_rd.npz", allow_pickle=True)
    need = ("p", "g_read", "g_used", "ok", "p_prev", "g_prev", "ok_prev", "have", "c_fed", "id0")
    miss = [k for k in need if k not in S.files]
    n = len(L["ci"])
    res = {"rule_log": a.rule_log, "k": a.k, "n_ticks": int(n), "tier": "T0"}
    res["i1_columns"] = {"missing": miss, "rows_equal": bool(not miss and all(len(S[k]) == n for k in need)), "PASS": bool(not miss and all(len(S[k]) == n for k in need))}
    if not res["i1_columns"]["PASS"]:
        res["PASS"] = False
        json.dump(res, open(a.out, "w", encoding="utf-8", newline="\n"), indent=1)
        print("ZZ PREV-CHECK FAIL columns", miss)
        return 4
    ci, j = np.asarray(L["ci"], np.int64), np.asarray(L["j"], np.int64)
    p, g, ok = np.asarray(S["p"], np.float64), np.asarray(S["g_read"], np.float64), np.asarray(S["ok"]).astype(bool)
    pp, gp, okp = np.asarray(S["p_prev"], np.float64), np.asarray(S["g_prev"], np.float64), np.asarray(S["ok_prev"]).astype(bool)
    have, cf, id0 = np.asarray(S["have"]).astype(bool), np.asarray(S["c_fed"], np.float64), np.asarray(S["id0"], np.float64)
    slope, kd = float(L["slope"]), int(L["k_diff"])
    mx0 = float(np.max(id0)) if n else float("nan")
    res["i2_gather_at_zero"] = {"max_abs": mx0, "PASS": bool(n > 0 and np.isfinite(mx0) and mx0 <= TOL_ID)}
    cs = np.asarray(L["c_src"], np.float64)
    cnt = json.loads(str(L["mix_counts"]))
    d_feed = float(np.max(np.abs(cs - cf)))
    res["i3_feed"] = {"max_abs_c_src_minus_fed": d_feed, "max_abs_fed_where_no_read": float(np.max(np.abs(cf[~have]))) if (~have).any() else 0.0, "closing_from_other": int(cnt.get("closing_from_other", -1)),
                      "k_diff": kd, "slope": slope}
    res["i3_feed"]["PASS"] = bool(d_feed <= TOL_ID and res["i3_feed"]["max_abs_fed_where_no_read"] == 0.0 and res["i3_feed"]["closing_from_other"] == 0 and kd == a.k)
    # ---- i4: the reader's own clip path on the cache files (float16 as the store holds the tokens)
    rd = TR.TokReader.from_ckpt(a.reader)
    J = np.load(a.join, allow_pickle=True)
    jclip = [str(c) for c in J["clip"]]
    P_all, G_all, OK_all = {}, {}, {}
    for c in sorted(set(int(v) for v in ci)):
        d = torch.load(os.path.join(a.eval_cache, "eval", "clips", jclip[c] + ".pt"), map_location="cpu")
        x = d["pool4x10"].to(torch.float16).float().numpy()
        P_all[c], G_all[c], OK_all[c] = TR.read_clip(rd, x)
    T = {c: len(v) for c, v in P_all.items()}
    now = np.array([(OK_all[c][t] if 0 <= t < T[c] else False) for c, t in zip(ci, j)], bool)
    e = j - a.k
    prv = np.array([(OK_all[c][t] if 0 <= t < T[c] else False) for c, t in zip(ci, e)], bool)
    dn_p = np.array([abs(P_all[c][t] - a_) if o else 0.0 for c, t, a_, o in zip(ci, j, p, now)])
    dn_g = np.array([abs(G_all[c][t] - a_) if o else 0.0 for c, t, a_, o in zip(ci, j, g, now)])
    dp_p = np.array([abs(P_all[c][t] - a_) if o else 0.0 for c, t, a_, o in zip(ci, e, pp, prv)])
    dp_g = np.array([abs(G_all[c][t] - a_) if o else 0.0 for c, t, a_, o in zip(ci, e, gp, prv)])
    res["i4_recomputed"] = {"n_now": int(now.sum()), "n_prev": int(prv.sum()), "flag_now_equal": bool(np.array_equal(now, ok)), "flag_prev_equal": bool(np.array_equal(prv, okp)),
                            "max_abs_p_now": float(dn_p.max()), "max_abs_g_now_m": float(dn_g.max()), "max_abs_p_prev": float(dp_p.max()), "max_abs_g_prev_m": float(dp_g.max()),
                            "prev_nan_where_no_history": bool(np.all(~np.isfinite(gp[~okp]))) if (~okp).any() else True}
    r4 = res["i4_recomputed"]
    r4["PASS"] = bool(r4["n_prev"] >= 1 and r4["flag_now_equal"] and r4["flag_prev_equal"] and r4["prev_nan_where_no_history"]
                      and max(r4["max_abs_p_now"], r4["max_abs_g_now_m"], r4["max_abs_p_prev"], r4["max_abs_g_prev_m"]) <= TOL_RE)
    # ---- i5: the fed value from the logged reads
    present = (p >= P_MIN) & ok
    with np.errstate(invalid="ignore"):
        have2 = present & okp & (pp >= P_MIN) & np.isfinite(g) & np.isfinite(gp)
        want = np.where(have2, slope * (g - gp) / (DT_S * a.k), 0.0)
    res["i5_formula"] = {"have_equal": bool(np.array_equal(have2, have)), "max_abs": float(np.max(np.abs(want - cf))), "n_have": int(have.sum()), "n_reader_present": int(present.sum()),
                         "share_of_present_with_closing_read": float(have.sum() / max(1, present.sum()))}
    res["i5_formula"]["PASS"] = bool(res["i5_formula"]["have_equal"] and res["i5_formula"]["max_abs"] <= TOL_ID and res["i5_formula"]["n_have"] >= 1)
    res["PASS"] = bool(all(res[k]["PASS"] for k in ("i1_columns", "i2_gather_at_zero", "i3_feed", "i4_recomputed", "i5_formula")))
    json.dump(res, open(a.out, "w", encoding="utf-8", newline="\n"), indent=1)
    for k in ("i1_columns", "i2_gather_at_zero", "i3_feed", "i4_recomputed", "i5_formula"):
        print("   %s %s %s" % (k, "PASS" if res[k]["PASS"] else "FAIL", json.dumps(res[k])[:420]))
    print("ZZ PREV-CHECK %s" % ("PASS" if res["PASS"] else "FAIL"))
    return 0 if res["PASS"] else 4


if __name__ == "__main__":
    sys.exit(main())
