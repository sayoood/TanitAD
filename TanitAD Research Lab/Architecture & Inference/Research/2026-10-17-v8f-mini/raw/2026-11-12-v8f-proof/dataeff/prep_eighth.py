"""prep_eighth.py -- the clip list of S-1/8 for DATAEFF (PREREG_V8F_PROOF_ADD13 section 8: if R-1/4 is not worse than full, Q2 and Q3 move to one eighth with the same rules).
CPU, label statistics only (train labels; no eval data, no model). Every function is imported unchanged from dataeff_prep.py.
Reading choice E1 (moves no threshold, bar or outcome rule): the rule of section 3 scaled by one half: 423 clips with the highest score (ties by hash order) plus the first 141 of the
remaining clips in hash order (the same three quarters / one quarter split as 846 + 282), 564 clips, the count of R-1/8.
Writes lists/S_eighth.txt, G1_EIGHTH.json (design gate g1 at one eighth: mean score of S-1/8 at least 1.3 times that of R-1/8), CONTROLS_EIGHTH.json.
"""
import json
import os
import sys

sys.dont_write_bytecode = True
import numpy as np  # noqa: E402
import dataeff_prep as DP  # noqa: E402

TOP8, FLOOR8 = 423, 141


def main():
    out = DP.HERE
    man = DP.read_manifest(DP.TRAIN_MANIFEST)
    clips = [e["clip"] for e in man]
    nfr = {e["clip"]: int(e["n_frames"]) for e in man}
    order = DP.hash_order(clips)
    N = len(clips)
    B8 = DP.range_end(N, DP.KEPT["R_eighth"])
    R8 = DP.apply_exclude(order, DP.HOLD[0], B8)
    B4 = DP.range_end(N, DP.KEPT["R_quarter"])
    R4 = DP.apply_exclude(order, DP.HOLD[0], B4)
    excluded60 = set(order[DP.HOLD[0]:DP.HOLD[1]])
    tracks_pt = __import__("torch").load(DP.TRACKS, map_location="cpu", weights_only=True)["tracks"]
    tracks = {c: t.double().numpy() for c, t in tracks_pt.items()}
    stats = DP.clip_stats(order, [nfr[c] for c in order], tracks)
    sc = DP.scores_of(stats)
    S8 = DP.select_S(order, sc, excluded60, TOP8, FLOOR8)
    S4 = DP.select_S(order, sc, excluded60)
    assert len(S8) == TOP8 + FLOOR8 == len(R8) == 564 and not (set(S8) & excluded60)
    h = DP.write_list(out + "/lists/S_eighth.txt", DP.hash_order(S8))
    with open(out + "/lists/R_eighth.txt", encoding="utf-8") as f:
        r8_file = [ln.strip() for ln in f if ln.strip()]
    s8, r8 = np.array([sc[c] for c in S8]), np.array([sc[c] for c in R8])
    mS, mR = float(s8.mean()), float(r8.mean())
    ratio = mS / mR if mR > 0 else float("inf")
    top_set = set(DP.select_S(order, sc, excluded60, TOP8, 0))
    g1 = {"rule": "mean score(S-1/8) >= 1.3 x mean score(R-1/8)", "mean_score_S_eighth": mS, "mean_score_R_eighth": mR, "ratio": ratio, "threshold": DP.G1_RATIO,
          "PASS": bool(ratio >= DP.G1_RATIO), "top": TOP8, "floor": FLOOR8,
          "S_top_mean": float(np.mean([sc[c] for c in S8 if c in top_set])), "S_floor_mean": float(np.mean([sc[c] for c in S8 if c not in top_set])),
          "share_at_1_S": float((s8 == 1).mean()), "share_at_0_S": float((s8 == 0).mean()), "share_at_1_R": float((r8 == 1).mean()), "share_at_0_R": float((r8 == 0).mean()),
          "list_sha256": h, "n_windows_S_eighth": int(sum(stats[c][0] for c in S8)), "n_windows_R_eighth": int(sum(stats[c][0] for c in R8)),
          "exclude_R_eighth": "%d:%d" % (DP.HOLD[0], B8)}
    with open(out + "/G1_EIGHTH.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(g1, f, indent=1)
    eval_ent = DP.read_manifest(DP.EVAL_MANIFEST)
    ids = set(e["clip"] for e in eval_ent)
    ctl = {"e0_count": {"S_eighth": len(S8), "R_eighth": len(R8), "PASS": bool(len(S8) == 564 and len(R8) == 564)},
           "e1_R_eighth_equals_banked_list": {"PASS": bool(sorted(r8_file) == sorted(R8))},
           "e2_eval_overlap": {"S_eighth": len(ids & set(S8)), "n_eval": len(ids), "PASS": bool(len(ids & set(S8)) == 0)},
           "e3_no_holdout_clip": {"PASS": bool(not (set(S8) & excluded60))},
           "e4_S_quarter_reproduced": {"PASS": bool(sorted(S4) == sorted(ln.strip() for ln in open(out + "/lists/S_quarter.txt", encoding="utf-8") if ln.strip()))},
           "e5_overlap_reported": {"S_eighth_in_S_quarter": len(set(S8) & set(S4)), "S_eighth_in_R_eighth": len(set(S8) & set(R8)), "S_eighth_in_R_quarter": len(set(S8) & set(R4))}}
    ctl["ALL_PASS"] = bool(all(v.get("PASS", True) for v in ctl.values() if isinstance(v, dict)))
    with open(out + "/CONTROLS_EIGHTH.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(ctl, f, indent=1)
    print("G1 eighth: S %.5f / R %.5f = %.4f -> %s" % (mS, mR, ratio, "PASS" if g1["PASS"] else "FAIL"))
    print("windows S %d R %d | list sha256 %s | exclude R-1/8 %s" % (g1["n_windows_S_eighth"], g1["n_windows_R_eighth"], h[:16], g1["exclude_R_eighth"]))
    print("controls", json.dumps(ctl))
    return 0 if ctl["ALL_PASS"] else 2


if __name__ == "__main__":
    sys.exit(main())
