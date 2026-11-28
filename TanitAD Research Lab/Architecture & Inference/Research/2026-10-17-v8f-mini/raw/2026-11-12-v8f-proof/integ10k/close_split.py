# close_split.py : the split named in section 5 of card PREREG_V8F_PROOF_ADD48 for the outcome CUTS-UNSAFE with READER-NO-GAIN: on which ticks do the reader and the
# label differ. Descriptive, from the rule logs only (numpy); it names nothing and ranks the levers for the next arm. Tier T1-kin logs; the reads themselves are T0
# (the reader reads tokens of logged frames, the label gap is that of the logged frame).
# Per model seed, on the ticks of the runs RLc (label presence and gap) and RQ50c (reader presence and fed gap, same ticks):
#   in range = the label has a lead at the tick with a gap of at most 60 m (the rule is inactive above 60 m);
#   both / miss (label only) / false (reader only) on the ticks, the gap error on the ticks where both are present, the reader's probability on the missed ticks,
#   and per window with an in-range label tick: reader present at the first such tick (on time), only at a later one (late) or at none (missed).
# Roots from PS_ROOT / PS_OUT. Usage: python close_split.py
import json
import os

import numpy as np

ROOT = os.environ.get("PS_ROOT", "F:/TanitAD-scratch/v8proof/integ10k/pull")
OUT = os.environ.get("PS_OUT", "F:/TanitAD-scratch/v8proof/integ10k/read_close/CLOSE_SPLIT.json")
G_MAX = 60.0
BANDS = ((0.0, 10.0), (10.0, 20.0), (20.0, 40.0), (40.0, 60.0))


def L(arm, k, suf=""):
    return np.load(os.path.join(ROOT, "%s_s%d" % (arm, k), "ticks", "rule_log_%s_s%d%s.npz" % (arm, k, suf)))


def q(x, p):
    return float(np.percentile(x, p)) if len(x) else None


res = {"card": "PREREG_V8F_PROOF_ADD48", "what": "split of the ticks where reader and label differ (section 5, outcome CUTS-UNSAFE with READER-NO-GAIN); names nothing", "per_seed": {}}
for k in (0, 1):
    a, b, r = L("I10RLc", k), L("I10RQ50c", k), L("I10RQ50c", k, "_rd")
    assert all(np.array_equal(a[c], b[c]) for c in ("ci", "t0", "tick")), "tick grids differ"
    assert len(r["p"]) == len(a["ci"]), "side log length"
    lp, rp = np.asarray(a["present"]).astype(bool), np.asarray(b["present"]).astype(bool)
    gl, gf, gr = (np.asarray(x, np.float64) for x in (a["g_src"], b["g_src"], r["g_read"]))
    p, ok = np.asarray(r["p"], np.float64), np.asarray(r["ok"]).astype(bool)
    rng = lp & np.isfinite(gl) & (gl <= G_MAX)
    both, miss, false_ = rng & rp, rng & ~rp, ~lp & rp
    e_fed, e_read = (gf - gl)[both], (gr - gl)[both]
    S = {"n_ticks": int(len(lp)), "n_label_present": int(lp.sum()), "n_in_range": int(rng.sum()), "n_reader_present": int(rp.sum()),
         "in_range": {"both": int(both.sum()), "miss": int(miss.sum()), "recall": float(both.sum() / max(1, rng.sum()))},
         "false_presence_ticks": int(false_.sum()), "false_presence_share_of_no_label_ticks": float(false_.sum() / max(1, (~lp).sum())),
         "gap_error_fed_minus_label_m": {"mean": float(e_fed.mean()) if len(e_fed) else None, "p10": q(e_fed, 10), "median": q(e_fed, 50), "p90": q(e_fed, 90),
                                         "share_above_plus_5m": float((e_fed > 5).mean()) if len(e_fed) else None, "share_below_minus_5m": float((e_fed < -5).mean()) if len(e_fed) else None},
         "gap_error_read_minus_label_m": {"mean": float(e_read.mean()) if len(e_read) else None, "median": q(e_read, 50), "mae": float(np.abs(e_read).mean()) if len(e_read) else None},
         "missed_ticks": {"n": int(miss.sum()), "history_flag_false": int((miss & ~ok).sum()), "p_median": q(p[miss], 50), "share_p_at_least_0.3": float((p[miss] >= 0.3).mean()) if miss.any() else None,
                          "share_p_at_least_0.1": float((p[miss] >= 0.1).mean()) if miss.any() else None, "label_gap_median_m": q(gl[miss], 50)},
         "recall_by_label_gap_band": {}}
    for lo, hi in BANDS:
        m = rng & (gl >= lo) & (gl < hi if hi < G_MAX else gl <= hi)
        S["recall_by_label_gap_band"]["%g-%g" % (lo, hi)] = {"n": int(m.sum()), "recall": float((m & rp).sum() / max(1, m.sum())),
                                                               "fed_minus_label_median_m": q((gf - gl)[m & rp], 50)}
    # windows
    key = np.asarray(a["ci"], np.int64) * 1000000 + np.asarray(a["t0"], np.int64)
    tick = np.asarray(a["tick"], np.int64)
    on_time = late = missed = 0
    for w in np.unique(key[rng]):
        m = key == w
        o = np.argsort(tick[m], kind="stable")
        r_in, p_in = rng[m][o], rp[m][o]
        idx = np.flatnonzero(r_in)
        if p_in[idx[0]]:
            on_time += 1
        elif p_in[idx].any():
            late += 1
        else:
            missed += 1
    nw = on_time + late + missed
    S["windows_with_in_range_label_tick"] = {"n": int(nw), "reader_on_time": int(on_time), "reader_late": int(late), "reader_missed": int(missed),
                                              "share_on_time": float(on_time / max(1, nw)), "share_missed": float(missed / max(1, nw))}
    S["rule_changed_plan_share_on_in_range_ticks"] = {"RLc": float(np.asarray(a["changed"]).astype(bool)[rng].mean()) if rng.any() else None,
                                                      "RQ50c": float(np.asarray(b["changed"]).astype(bool)[rng].mean()) if rng.any() else None}
    res["per_seed"][str(k)] = S
    print("s%d ticks %d | label present %d, in range %d | reader present %d | in range: both %d miss %d recall %.3f | false presence %d (%.3f of no-label ticks)" % (
        k, S["n_ticks"], S["n_label_present"], S["n_in_range"], S["n_reader_present"], S["in_range"]["both"], S["in_range"]["miss"], S["in_range"]["recall"],
        S["false_presence_ticks"], S["false_presence_share_of_no_label_ticks"]))
    g = S["gap_error_fed_minus_label_m"]
    print("s%d fed gap minus label gap (both present): mean %+.2f m, p10 %+.2f, median %+.2f, p90 %+.2f, share above +5 m %.3f, below -5 m %.3f | read gap MAE %.2f m" % (
        k, g["mean"], g["p10"], g["median"], g["p90"], g["share_above_plus_5m"], g["share_below_minus_5m"], S["gap_error_read_minus_label_m"]["mae"]))
    ms = S["missed_ticks"]
    print("s%d missed ticks %d: history flag false %d, reader p median %s, share p >= 0.3 %s, p >= 0.1 %s, label gap median %s m" % (
        k, ms["n"], ms["history_flag_false"], ms["p_median"], ms["share_p_at_least_0.3"], ms["share_p_at_least_0.1"], ms["label_gap_median_m"]))
    print("s%d recall by label gap band: %s" % (k, json.dumps(S["recall_by_label_gap_band"])))
    print("s%d windows: %s | rule changed the plan on in-range ticks: %s" % (k, json.dumps(S["windows_with_in_range_label_tick"]), json.dumps(S["rule_changed_plan_share_on_in_range_ticks"])))
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(res, open(OUT, "w", encoding="utf-8", newline="\n"), indent=1)
print("ZZ CLOSE-SPLIT DONE")
