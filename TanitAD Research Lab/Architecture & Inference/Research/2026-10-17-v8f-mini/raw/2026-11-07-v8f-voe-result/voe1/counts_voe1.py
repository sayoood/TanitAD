"""VOE1 pre-statistic counts (labels / ego / bank geometry only; NO latent statistic): qualifying windows per manipulation, donors, event onsets and step-label rates.
Writes counts_voe1.json. Run after build_bank_voe1.py (no tanitad import needed)."""
import json, os
import numpy as np, torch
import voe1_core as C

HERE = os.path.dirname(os.path.abspath(__file__))
Bk = torch.load(os.path.join(HERE, "bank_voe1.pt"), weights_only=False)
w = Bk["win"]; keep = np.asarray(w["keep"])
eid = np.array(w["eid"])[keep]; base = np.asarray(w["base"])[keep]; g0 = np.asarray(w["g0"])[keep]
T_clip = np.array([Bk["T"][e] for e in eid])
ep, n_ep = C.episode_codes(eid)
act, spd = Bk["act"].numpy(), Bk["speed"].numpy()
wt = C.window_table(act, spd, base, g0, T_clip, ep)
out = {"n_windows": int(len(base)), "n_episodes": int(n_ep), "qualify": {}, "spd_min_pct": [float(x) for x in np.nanpercentile(wt["spd_min"], [5, 25, 50, 75, 95])]}
for k, q in wt["q"].items():
    out["qualify"][k] = {"n_windows": int(q.sum()), "n_episodes": int(len(set(eid[q])))}
sw = wt["swap_ncand"]
out["action_swap_candidates"] = {"min": int(sw.min()), "median": float(np.median(sw)), "windows_without_donor": int((sw == 0).sum())}
out["swap_lon_gap_pct"] = [float(x) for x in np.percentile(np.abs(wt["lon"][wt["donor"]["ACTION-SWAP"][wt["q"]["ACTION-SWAP"]]] - wt["lon"][wt["q"]["ACTION-SWAP"]]), [5, 50, 95])]
out["frames"] = {"n_frames": int(len(act)), "n_clips": len(Bk["clips"])}
clip_lo = np.array([Bk["off"][e] for e in eid]); clip_hi = clip_lo + T_clip
out["events"] = {}
for ev, key in (("brake", "ev_brake"), ("turn", "ev_turn"), ("lead", "ev_lead")):
    on = Bk[key]
    rec = {"onsets_in_bank": int(on.sum()), "clips_with_onset": int(sum(on[Bk["off"][c]:Bk["off"][c] + Bk["T"][c]].any() for c in Bk["clips"])), "levels": {}}
    for l, S in ((1, 15), (2, 7), (3, 3)):
        Y = C.event_step_labels(on, base, clip_lo, clip_hi, l, S)
        rec["levels"][str(l)] = {"n_points": int(Y.size), "positives": int(Y.sum()), "rate": float(Y.mean()), "windows_with_positive": int(Y.any(1).sum())}
    out["events"][ev] = rec
# the two clips for the along-clip figure: the most onsets of the three types combined, ties by sorted clip id
tot = {c: int(sum(Bk[k][Bk["off"][c]:Bk["off"][c] + Bk["T"][c]].sum() for k in ("ev_brake", "ev_turn", "ev_lead"))) for c in Bk["clips"]}
best = sorted(Bk["clips"], key=lambda c: (-tot[c], c))[:2]
out["figure_clips"] = {c: {"onsets_total": tot[c], "T": int(Bk["T"][c]), "brake": int(Bk["ev_brake"][Bk["off"][c]:Bk["off"][c]+Bk["T"][c]].sum()),
                           "turn": int(Bk["ev_turn"][Bk["off"][c]:Bk["off"][c]+Bk["T"][c]].sum()), "lead": int(Bk["ev_lead"][Bk["off"][c]:Bk["off"][c]+Bk["T"][c]].sum())} for c in best}
out["clip_length"] = {"min": int(min(Bk["T"].values())), "median": float(np.median(list(Bk["T"].values()))), "max": int(max(Bk["T"].values()))}
a = act
out["act_ranges"] = {"a_pct_1_50_99": [float(x) for x in np.percentile(a[:, 0], [1, 50, 99])], "yaw_pct_1_50_99": [float(x) for x in np.percentile(a[:, 1], [1, 50, 99])]}
json.dump(out, open(os.path.join(HERE, "counts_voe1.json"), "w", encoding="utf-8", newline="\n"), indent=1)
print(json.dumps(out, indent=1))
