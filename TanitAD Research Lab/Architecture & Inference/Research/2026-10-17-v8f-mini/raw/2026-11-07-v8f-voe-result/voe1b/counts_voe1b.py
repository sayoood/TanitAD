"""VOE1B pre-statistic counts (labels / ego / bank geometry only; NO latent or surprise statistic): qualifying windows per manipulation (incl. ACTION-SWAP-M), donors, event
onsets and step-label rates. Writes counts_voe1b.json. Run after build_bank_voe1b.py (no tanitad import needed)."""
import json, os
import numpy as np, torch
import voe1b_core as C

HERE = os.path.dirname(os.path.abspath(__file__))
BANK = os.environ.get("VOE1B_BANK", "F:/TanitAD-scratch/voe1b_bank/bank_voe1b.pt")
Bk = torch.load(BANK, weights_only=False)
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
# ACTION-SWAP-M (card: magnitude within +-10 %, opposite sign where the mean is at least 0.5 m/s^2 / 0.05 rad/s): qualifying n from labels only, under the committed reading
# (sign tested on a channel only if BOTH signed means reach the threshold) and, DISCLOSED ONLY, the looser window-only reading (donor's signed mean any size)
qm = wt["q"]["ACTION-SWAP-M"]
mn = wt["swapm_ncand"]
ml, my, sl, sy = wt["mabs_lon"], wt["mabs_yaw"], wt["lon"], wt["yaw"]
loose = np.zeros(len(ep), bool)
for i in range(len(ep)):
    ok = (ep != ep[i]) & (np.abs(ml - ml[i]) <= C.SWM_REL * ml[i]) & (np.abs(my - my[i]) <= C.SWM_REL * my[i])
    ol = (np.abs(sl[i]) >= C.SWM_LON_MIN) & (np.sign(sl) == -np.sign(sl[i]))
    oy = (np.abs(sy[i]) >= C.SWM_YAW_MIN) & (np.sign(sy) == -np.sign(sy[i]))
    loose[i] = bool((ok & (ol | oy)).any())
mag_only = np.zeros(len(ep), bool)
for i in range(len(ep)):
    mag_only[i] = bool(((ep != ep[i]) & (np.abs(ml - ml[i]) <= C.SWM_REL * ml[i]) & (np.abs(my - my[i]) <= C.SWM_REL * my[i])).any())
don = wt["donor"]["ACTION-SWAP-M"]
out["action_swap_m"] = {"committed_reading": {"n_windows": int(qm.sum()), "n_episodes": int(len(set(eid[qm]))), "underpowered_below": C.SWM_MIN_N, "UNDERPOWERED": bool(qm.sum() < C.SWM_MIN_N),
                                              "candidates_per_qualifying_window": {"min": int(mn[qm].min()) if qm.any() else 0, "median": float(np.median(mn[qm])) if qm.any() else 0.0, "max": int(mn.max())}},
                        "disclosure_window_only_sign_reading": {"n_windows": int(loose.sum())}, "disclosure_magnitude_match_only": {"n_windows": int(mag_only.sum())},
                        "windows_with_threshold_lon": int((np.abs(sl) >= C.SWM_LON_MIN).sum()), "windows_with_threshold_yaw": int((np.abs(sy) >= C.SWM_YAW_MIN).sum()),
                        "donor_vs_window_mean_abs_lon_ratio_pct_5_50_95": [float(x) for x in np.percentile(ml[don[qm]] / ml[qm], [5, 50, 95])] if qm.any() else None,
                        "donor_vs_window_mean_abs_yaw_ratio_pct_5_50_95": [float(x) for x in np.percentile(my[don[qm]] / my[qm], [5, 50, 95])] if qm.any() else None,
                        "donor_is_other_episode": bool((ep[don[qm]] != ep[qm]).all()) if qm.any() else None,
                        "qualifying_windows_with_lon_sign_flip": int((np.sign(sl[don[qm]]) == -np.sign(sl[qm]))[(np.abs(sl[qm]) >= C.SWM_LON_MIN) & (np.abs(sl[don[qm]]) >= C.SWM_LON_MIN)].sum()) if qm.any() else 0,
                        "qualifying_windows_with_yaw_sign_flip": int((np.sign(sy[don[qm]]) == -np.sign(sy[qm]))[(np.abs(sy[qm]) >= C.SWM_YAW_MIN) & (np.abs(sy[don[qm]]) >= C.SWM_YAW_MIN)].sum()) if qm.any() else 0}
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
json.dump(out, open(os.path.join(HERE, "counts_voe1b.json"), "w", encoding="utf-8", newline="\n"), indent=1)
print(json.dumps(out, indent=1))
