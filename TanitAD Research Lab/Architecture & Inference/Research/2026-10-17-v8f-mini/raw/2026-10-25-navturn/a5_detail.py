"""a5_detail.py -- distinct events, episode concentration and parameter distributions behind the cause table (EXPLORATORY, MEASURED eval-141; reads navturn_windows.csv)."""
import csv
import numpy as np
from nt_common import *

J, K, T, grid = load_inputs()
rows = list(csv.DictReader(open(HERE + "navturn_windows.csv", encoding="utf-8")))
f = lambda s: float(s) if s not in ("", "nan") else np.nan
N = len(T["ci"])
ci = T["ci"]
G = T["gt_std"]
cols = list(range(1, 10))


def sq(arm):
    r = T[f"T_{arm}"] - G
    return np.where(np.isfinite(r), r ** 2, 0.0)


print("events = distinct (episode, absolute segment start); a window belongs to the event its label cells describe (A) or the next announced turn (B)")
for typ in ("A", "B"):
    print(f"\n[{typ}]")
    cz = {}
    for r in rows:
        if r["type"] == typ:
            cz.setdefault(r["cause"], []).append(r)
    for name, lst in sorted(cz.items(), key=lambda kv: -len(kv[1])):
        # absolute start proxy: now_s(rowT or rowN) + t_start (A) ; for B use the clip + (rowN - t index) bucket by the announced distance
        ev = set()
        for r in lst:
            if typ == "A":
                now = float(J["now_s"][int(r["rowT"])])
                ev.add((r["clip"], round(now + f(r["ts"]), 0)))
            else:
                now = float(J["now_s"][int(r["rowN"])])
                ts = f(r["ts"])
                ev.add((r["clip"], round(now + ts, 0) if np.isfinite(ts) else -1))
        eps = {r["clip"] for r in lst}
        print(f"  {name:12s} windows {len(lst):4d}  episodes {len(eps):3d}  distinct events {len(ev):3d}  windows/episode median {np.median(list(np.bincount([sorted(eps).index(r['clip']) for r in lst]))) :.1f}")

print("\n== CURVE (A): parameters of the label segment (join cells at rowT; vmin from the re-run turn table)")
cv = [r for r in rows if r["type"] == "A" and r["cause"] == "CURVE"]
for nm, key in (("|dyaw| deg", "dyaw"), ("R_arc m", "Rarc"), ("vmin m/s", "vmin"), ("t_start s", "ts"), ("t_end-t_start s", None)):
    if key:
        v = np.abs(np.array([f(r[key]) for r in cv])) if key == "dyaw" else np.array([f(r[key]) for r in cv])
    else:
        v = np.array([f(r["te"]) - f(r["ts"]) for r in cv])
    print(f"  {nm:16s} p10 {np.percentile(v, 10):7.1f}  p25 {np.percentile(v, 25):7.1f}  median {np.median(v):7.1f}  p75 {np.percentile(v, 75):7.1f}  p90 {np.percentile(v, 90):7.1f}")
vm = np.array([f(r["vmin"]) for r in cv])
print(f"  vmin bands: (8,10] {int(((vm > 8) & (vm <= 10)).sum())}  (10,12] {int(((vm > 10) & (vm <= 12)).sum())}  (12,15] {int(((vm > 12) & (vm <= 15)).sum())}  >15 {int((vm > 15).sum())}   [(8,10] = within 25 % of the gate = genuinely ambiguous]")
dy = np.abs(np.array([f(r["dyaw"]) for r in cv]))
print(f"  |dyaw| bands: <30 {int((dy < 30).sum())}  30-60 {int(((dy >= 30) & (dy < 60)).sum())}  60-120 {int(((dy >= 60) & (dy < 120)).sum())}  >=120 {int((dy >= 120).sum())}")
# episode concentration of the curve SSE
for arm in ("M2b_s0", "M2b_s1"):
    E = sq(arm)
    w = np.array([int(r["w"]) for r in cv])
    sse = E[w][:, cols].sum(1)
    eid = np.array([r["clip"] for r in cv])
    byep = {e: sse[eid == e].sum() for e in np.unique(eid)}
    tot = sum(byep.values())
    top = sorted(byep.values(), reverse=True)
    print(f"  {arm}: CURVE turn-cell SSE {tot:.1f} over {len(byep)} episodes; top-1 {top[0] / tot:.1%}, top-3 {sum(top[:3]) / tot:.1%}, top-5 {sum(top[:5]) / tot:.1%}")

print("\n== SUPPRESSED (A)")
sp = [r for r in rows if r["type"] == "A" and r["cause"] == "SUPPRESSED"]
print("  episodes:", sorted({r['clip'][:8] for r in sp}), " t_start median", np.median([f(r['ts']) for r in sp]))

print("\n== HORIZON (A): label t_start of the 16 silent is_turn windows (s)")
hz = [r for r in rows if r["type"] == "A" and r["cause"] == "HORIZON"]
print("  t_start:", np.round(sorted(f(r['ts']) for r in hz), 1))

print("\n== B causes: speed and time facts")
for name in ("BAND_START", "HORIZON", "LABEL_MASKED", "UNRESOLVED", "BAND_EDGE"):
    lst = [r for r in rows if r["type"] == "B" and r["cause"] == name]
    if not lst:
        continue
    v = np.array([f(r["v_N"]) for r in lst]); d = np.array([f(r["navd_N"]) for r in lst]); da = np.array([f(r["d_ann_N"]) for r in lst])
    print(f"  {name:12s} n {len(lst):3d}: v_now median {np.median(v):5.1f} m/s (p10 {np.percentile(v, 10):.1f}, p90 {np.percentile(v, 90):.1f}); nav d_next median {np.nanmedian(d):6.1f} m; d_ann median {np.median(da):5.1f} m; "
          f"windows at the 30 m floor (v<5 m/s) {int((v < 5).sum())}")
bs = [r for r in rows if r["type"] == "B" and r["cause"] == "BAND_START"]
te = np.array([f(r["te"]) for r in bs]); ts_ = np.array([f(r["ts"]) for r in bs]); dyb = np.abs(np.array([f(r["dyaw"]) for r in bs]))
print(f"  BAND_START: segment ends in {np.median(te):.1f} s (p10 {np.percentile(te, 10):.1f}, p90 {np.percentile(te, 90):.1f}); started {np.median(-ts_):.1f} s ago; |dyaw| median {np.median(dyb):.0f} deg; "
      f"dpsi6 (6 s path heading change at the tick) median |{np.median(np.abs([f(r['dpsi6']) for r in bs])):.1f}| deg")
hb = [r for r in rows if r["type"] == "B" and r["cause"] == "HORIZON"]
print(f"  B HORIZON evidence: {sum(1 for r in hb if r['ev'] == 'TRACK')} by pose-track end (>=8 s ahead), {sum(1 for r in hb if r['ev'] == 'REPRO')} by re-run table; {len({r['clip'] for r in hb})} episodes")
print("  B HORIZON episodes:", sorted({r['clip'][:8] for r in hb}))

print("\n== cause -> the lead's classes")
mapping = {("A", "CURVE"): "threshold mismatch", ("A", "SUPPRESSED"): "label/nav policy defect", ("A", "HORIZON"): "horizon mismatch", ("A", "UNRESOLVED"): "unresolved (event beyond pose track)",
           ("B", "HORIZON"): "horizon mismatch", ("B", "BAND_EDGE"): "horizon mismatch (band edge)", ("B", "BAND_START"): "horizon/definition (label excludes the first 2 s)",
           ("B", "TIMING"): "timing offset (row pairing artefact)", ("B", "LABEL_MASKED"): "label censoring (reversing mask)", ("B", "SIGN_THETA"): "label rule (sign/theta)", ("B", "UNRESOLVED"): "unresolved"}
for (typ, cz), cls in mapping.items():
    n = sum(1 for r in rows if r["type"] == typ and r["cause"] == cz)
    print(f"  {typ} {cz:12s} {n:4d}  -> {cls}")
