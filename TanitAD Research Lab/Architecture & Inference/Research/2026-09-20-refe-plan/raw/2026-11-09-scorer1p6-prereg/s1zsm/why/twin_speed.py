"""Descriptive, CPU, navtrain FIT only (8,489 sets, P6 in-sample; no navtest): the speed direction of S1Z's twin pairs under P6's own logits.
Pairs as PREREG_S1Z_P6 section 1 (clean i, hard negative j of head h in the rule's top 8, ADE <= 1.0 m), on the 8-pose 0.5 s FIT geometry (the card's ADE uses 20 raw poses).
v4 ~ chord pose6->pose7 / 0.5 s (t = 3.5..4.0 s); v0 ~ median over the set's valid proposals of the first chord (origin->pose0) / 0.5 s.  Approximations: this informs the definition, decides no bar."""
import json, numpy as np
Z = np.load("F:/tanitad_scratch/p6fit/fit_pack8.npz", allow_pickle=False)
ok = Z["has_pdm"].astype(bool)
L = Z["logits"][ok].astype(np.float64); PD = Z["pdm"][ok].astype(np.float64); XY = Z["traj_xy"][ok].astype(np.float64)
NC, DAC, EP, TTC, C = 0, 1, 2, 3, 4
P = 1 / (1 + np.exp(-L)); R = P[..., NC] * P[..., DAC] * (5 * P[..., EP] + 5 * P[..., TTC] + 2 * P[..., C]) / 12
V = ~np.isnan(PD[..., 6]) & np.isfinite(XY).all((-1, -2))
v4 = np.linalg.norm(XY[:, :, 7] - XY[:, :, 6], axis=-1) / 0.5
vfirst = np.linalg.norm(XY[:, :, 0], axis=-1) / 0.5
out = {}
for hn, h in (("NC", NC), ("DAC", DAC), ("TTC", TTC)):
    dv_all, rel_all, sets_pair, sets_bal, sets_band = [], [], 0, 0, 0
    for s in range(len(L)):
        v = V[s]
        if not v.any(): continue
        v0 = float(np.median(vfirst[s][v]))
        clean = v & (PD[s, :, NC] == 1) & (PD[s, :, DAC] == 1) & (PD[s, :, TTC] == 1)
        rank = np.where(v, R[s], -np.inf); top = np.argsort(-rank, kind="stable")[:8]
        hard = np.zeros(64, bool); hard[top] = True; hard &= v & (PD[s, :, h] < 1)
        ade = np.linalg.norm(XY[s][:, None] - XY[s][None, :], axis=-1).mean(-1)
        ii, jj = np.nonzero(clean[:, None] & hard[None, :] & (ade <= 1.0))
        if not len(ii): continue
        sets_pair += 1
        dv = v4[s, ii] - v4[s, jj]; rel = np.abs(dv) / max(v0, 1.0)
        dv_all.append(dv); rel_all.append(rel)
        band = rel < 0.15
        sets_band += int(band.any())
        sets_bal += int((dv[band] > 0).any() and (dv[band] < 0).any())
    dv = np.concatenate(dv_all); rel = np.concatenate(rel_all); band = rel < 0.15
    out[hn] = {"sets_with_pair": sets_pair, "pairs": int(len(dv)),
               "share_clean_slower": float((dv < 0).mean()), "share_clean_faster": float((dv > 0).mean()), "median_dv_mps": float(np.median(dv)),
               "share_in_band_rel_lt_0.15": float(band.mean()),
               "in_band": {"pairs": int(band.sum()), "share_clean_slower": float((dv[band] < 0).mean()), "share_clean_faster": float((dv[band] > 0).mean()), "median_dv_mps": float(np.median(dv[band])), "sets": sets_band},
               "out_of_band": {"pairs": int((~band).sum()), "share_clean_slower": float((dv[~band] < 0).mean()) if (~band).any() else None},
               "sets_with_both_directions_in_band": sets_bal}
print(json.dumps(out, indent=1))
json.dump(out, open("C:/Users/Admin/AppData/Local/Temp/claude/4ac3a056b343bb35/e07efd49-d158-4484-85f3-d8888196bc46/scratchpad/twin_speed.json", "w"), indent=1)
