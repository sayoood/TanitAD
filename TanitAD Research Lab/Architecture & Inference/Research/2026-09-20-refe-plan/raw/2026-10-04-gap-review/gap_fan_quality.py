#!/usr/bin/env python3
"""REVIEW 7 -- does the pick fail because the FAN is dirty? Pick outcome by the fan's share of NAVSIM-safe proposals.
CPU only (navsim venv python). Reads the lane census (GT per proposal) and the final proposal table (pick)."""
import json
import numpy as np
import gap_selection as GS

OUT = GS.PKG + "/raw/2026-10-04-gap-review/gap_fan_quality.json"
z = np.load(GS.PROPS, allow_pickle=True)
pos = {t: i for i, t in enumerate(z["token"])}
pick = z["pick"]
N = len(pick)
share = np.zeros(N); pz = np.zeros(N); ps = np.zeros(N); best = np.zeros(N); mean64 = np.zeros(N)
for line in open(GS.CENSUS):
    r = json.loads(line)
    i = pos[r["token"]]
    nc = np.asarray(r["nc"][1:65]); dac = np.asarray(r["dac"][1:65]); pd = np.asarray(r["pdms"][1:65])
    share[i] = np.mean(nc * dac == 1.0)
    ps[i] = pd[pick[i]]; pz[i] = float(pd[pick[i]] == 0); best[i] = pd.max(); mean64[i] = pd.mean()
bins = [0, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0001]
tab = {}
for lo, hi in zip(bins[:-1], bins[1:]):
    m = (share >= lo) & (share < hi)
    if m.sum() == 0:
        continue
    tab[f"[{lo:.2f},{min(hi,1):.2f}{']' if hi > 1 else ')'}"] = {
        "tokens": int(m.sum()), "share_of_tokens": float(m.mean()),
        "pick_PDMS_x100": float(ps[m].mean() * 100), "pick_zero_rate": float(pz[m].mean()),
        "best_of_64_x100": float(best[m].mean() * 100), "mean_of_64_x100": float(mean64[m].mean() * 100)}
out = {"n": N, "fan_safe_share_mean": float(share.mean()), "by_fan_safe_share": tab,
       "corr_token_fan_safe_share_vs_pick_pdms": float(np.corrcoef(share, ps)[0, 1])}
json.dump(out, open(OUT, "w"), indent=1)
print(json.dumps(out, indent=1))

# ---- appended: what limits the pick on CLEAN fans (>= 0.95 safe)? weighted-part components of the pick vs human
ep_p = np.zeros(N); ttc_p = np.zeros(N); c_p = np.zeros(N); ep_h = np.zeros(N); m_p = np.zeros(N)
for line in open(GS.CENSUS):
    r = json.loads(line)
    i = pos[r["token"]]
    j = 1 + pick[i]
    def ep_of(k):
        m = r["nc"][k] * r["dac"][k]
        return 0.0 if m <= 0 else max(0.0, min(1.0, (12 * r["pdms"][k] / m - 5 * r["ttc"][k] - 2 * r["c"][k]) / 5))
    ep_p[i] = ep_of(j); ttc_p[i] = r["ttc"][j]; c_p[i] = r["c"][j]; ep_h[i] = ep_of(65); m_p[i] = r["nc"][j] * r["dac"][j]
clean = share >= 0.95
dirty = share < 0.5
def blk(m):
    mm = m & (m_p > 0)
    return {"tokens": int(m.sum()), "pick_PDMS_x100": float(ps[m].mean() * 100),
            "pick_EP_x100_M>0": float(ep_p[mm].mean() * 100), "human_EP_x100": float(ep_h[m].mean() * 100),
            "pick_TTC_x100_M>0": float(ttc_p[mm].mean() * 100), "pick_C_x100_M>0": float(c_p[mm].mean() * 100),
            "weighted_part_loss_points_M>0": float(((1 - (5 * ep_p[mm] + 5 * ttc_p[mm] + 2 * c_p[mm]) / 12)).mean() * 100),
            "of_which_EP": float((5 * (1 - ep_p[mm]) / 12).mean() * 100),
            "of_which_TTC": float((5 * (1 - ttc_p[mm]) / 12).mean() * 100)}
out["clean_fan_ge_0.95"] = blk(clean)
out["dirty_fan_lt_0.50"] = blk(dirty)
out["all"] = blk(np.ones(N, bool))
json.dump(out, open(OUT, "w"), indent=1)
print(json.dumps({k: out[k] for k in ("clean_fan_ge_0.95", "dirty_fan_lt_0.50", "all")}, indent=1))
