"""Pick path-length telemetry for the REFe 'slow-pick' failure -- CPU only, reads two banked arrays.

CONTROL FIRST: the same statistics on the STORED pick of the final model (`proposals.npz::pick`) must
reproduce REVIEW_7 §2.3 / gap_selection.json (pick_path/fan_mean 0.8868, median length rank 49.0,
75.15 % of picks shorter than the fan median).  Only then are the SFT-4 numbers read.

SFT-4 u400 changed ONLY the scorer (eval/RESULT_SFT4.md: built on model_final.pt), so its picks are
members of the SAME 64-proposal fan; the pick index is recovered by matching the seam's executed xy to
the stored proposals.  A token whose best xy match is > 1 mm is COUNTED, never silently matched.
"""
import json
import sys
import numpy as np

PKG = "D:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
PROPS = "E:/Projects/TanitAD/data/refe_navtest/proptable/navtest_final/proposals.npz"
SEAM = PKG + "/raw/2026-10-05-sft-stage2/sft4_B_u400_navtest_full/sft4_B_u400_navtest_full_seam.npz"
DEP = PKG + "/raw/2026-10-05-sft-stage2/sft4_B_u400_navtest_full/deployed_seam.npz"


def plen(p):                                   # [..., 8, 3] -> path length through the origin
    xy = p[..., :2].astype(np.float64)
    first = np.linalg.norm(xy[..., 0, :], axis=-1)
    return first + np.linalg.norm(np.diff(xy, axis=-2), axis=-1).sum(-1)


def stats(L, pick):
    n = L.shape[0]
    pl = L[np.arange(n), pick]
    order = (-L).argsort(1).argsort(1)                       # 0 = longest
    rk = order[np.arange(n), pick]
    q = np.clip((rk // 13), 0, 4)                            # 5 bins of ~13 ranks; bin 4 = shortest
    # quintile of the PICK by length, shortest first, as in REVIEW_7
    share_short_first = [float((q == k).mean()) for k in (4, 3, 2, 1, 0)]
    return dict(n=int(n), pick_over_fan_mean_path=float((pl / L.mean(1)).mean()),
                median_rank_by_length_0_longest=float(np.median(rk)),
                share_pick_shorter_than_fan_median=float((pl < np.median(L, 1)).mean()),
                pick_length_quintile_shares_shortest_first=share_short_first,
                mean_pick_path_m=float(pl.mean()), mean_fan_path_m=float(L.mean()))


def main():
    P = np.load(PROPS, allow_pickle=True)
    prop, pick0, tok = P["proposals"], P["pick"], P["token"]
    L = plen(prop)
    out = {"control_stored_final_pick_all_tokens": stats(L, pick0)}
    res = {}
    for name, path in (("sft4_B_u400", SEAM), ("deployed_seam", DEP)):
        S = np.load(path, allow_pickle=True)
        assert (S["token"] == tok).all(), "token order differs"
        d = np.linalg.norm(prop[..., :2] - S["poses"][:, None, :, :2], axis=-1).max(-1)   # [n, 64]
        res[name] = (d.argmin(1), d.min(1))
    # The seam applies the ROUTE FIX, which re-runs the model on a corrected route goal, so on some tokens the
    # executed pick is NOT a member of the stored fan (xy mismatch up to 41 m).  Those tokens are EXCLUDED
    # from every row below (and counted), never matched to their nearest stranger.
    ok = np.ones(len(tok), bool)
    for name, (idx, dmin) in res.items():
        ok &= dmin <= 1e-3
    out["tokens_in_common_fan"] = int(ok.sum())
    out["tokens_excluded_route_fix_changed_fan"] = int((~ok).sum())
    out["control_stored_final_pick_common_tokens"] = stats(L[ok], pick0[ok])
    for name, (idx, dmin) in res.items():
        r = stats(L[ok], idx[ok])
        r["same_pick_as_stored_final"] = float((idx[ok] == pick0[ok]).mean())
        out[name + "_common_tokens"] = r
    json.dump(out, open("raw/pick_length_telemetry.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


main()
