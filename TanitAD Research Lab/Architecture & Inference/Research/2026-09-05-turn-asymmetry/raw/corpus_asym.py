"""P4 candidate 3 -- is the CORPUS asymmetric? GT-only, zero GPU."""
import json, sys, collections
import numpy as np

rows = json.load(open(sys.argv[1]))
LAT = {0: "lane_keep", 1: "turn_left", 2: "turn_right"}
for r in rows:
    r["lat_name"] = LAT[r["lat"]]

def block(sel, title):
    print("\n=== %s ===" % title)
    print("  %-11s %5s %5s | %8s %8s %8s | %8s %8s | %8s" %
          ("stratum", "n", "n_ep", "|dyaw|med", "|k0|med", "|kmax|med", "v0 med", "v0 mean", "latext"))
    out = {}
    for k in ("turn_left", "turn_right", "lane_keep"):
        s = [r for r in sel if r["lat_name"] == k]
        if not s:
            continue
        dy = np.abs([r["dyaw"] for r in s]); k0 = np.abs([r["k0"] for r in s])
        km = np.abs([r["kmax_signed"] for r in s]); v0 = np.array([r["v0"] for r in s])
        le = np.abs([r["lat_ext"] for r in s])
        out[k] = dict(n=len(s), dy=dy, k0=k0, km=km, v0=v0, le=le)
        print("  %-11s %5d %5d | %8.4f %8.5f %8.5f | %8.3f %8.3f | %8.3f" %
              (k, len(s), len({r["ei"] for r in s}), np.median(dy), np.median(k0),
               np.median(km), np.median(v0), v0.mean(), np.median(le)))
    if "turn_left" in out and "turn_right" in out:
        L, R = out["turn_left"], out["turn_right"]
        print("  -> L/R ratio of medians: |dyaw| %.3f  |k0| %.3f  |kmax| %.3f  v0 %.3f" %
              (np.median(L["dy"]) / np.median(R["dy"]),
               np.median(L["k0"]) / max(np.median(R["k0"]), 1e-9),
               np.median(L["km"]) / max(np.median(R["km"]), 1e-9),
               np.median(L["v0"]) / np.median(R["v0"])))
        # Mann-Whitney U, two-sided normal approx (windows are NOT independent;
        # this is a DESCRIPTIVE flag, never a decision statistic -- the decision
        # estimator is the episode-cluster bootstrap.)
        from scipy import stats
        for nm in ("dy", "k0", "km", "v0"):
            u = stats.mannwhitneyu(L[nm], R[nm], alternative="two-sided")
            print("     MWU %-4s p=%.4g   (DESCRIPTIVE ONLY: windows overlap)"
                  % (nm, u.pvalue))
    return out

W = 4
allw = rows
sel16 = [r for r in rows if (r["t"] - (W - 1)) % 16 == 0]
block(allw, "ALL 535 available windows")
block(sel16, "the banked stride-16 panel (n=40)")

print("\n=== turn duration: consecutive runs of the same turn label per episode ===")
for k in (1, 2):
    runs = []
    for ei in sorted({r["ei"] for r in rows}):
        sub = sorted([r for r in rows if r["ei"] == ei], key=lambda r: r["t"])
        cur = 0
        for r in sub:
            if r["lat"] == k:
                cur += 1
            else:
                if cur:
                    runs.append(cur)
                cur = 0
        if cur:
            runs.append(cur)
    if runs:
        print("  %-11s runs=%d  median=%.1f  mean=%.1f  max=%d  (window ticks, 0.2 s each)"
              % (LAT[k], len(runs), np.median(runs), np.mean(runs), max(runs)))

print("\n=== SIGN of GT curvature vs the dyaw label (consistency control) ===")
for k in (1, 2):
    s = [r for r in rows if r["lat"] == k]
    pos = sum(1 for r in s if r["kmax_signed"] > 0)
    print("  %-11s n=%d  kmax_signed > 0 on %d (%.3f)" % (LAT[k], len(s), pos, pos / len(s)))
