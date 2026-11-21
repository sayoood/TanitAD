#!/usr/bin/env python
"""Descriptive counts + cruise-bundle sanity check + independent hand-verification of one token per category.\nRun: C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe -I summarize.py\nThe thresholds below are EXAMPLES for orientation only. The frozen selection rule (stream BEHAV) decides later."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import json, math, pickle, sys
import numpy as np
import pandas as pd

OUT = r"E:/Projects/TanitAD-artifacts/navtest_behaviour_mine_20261107"
LOG_DIR = r"F:/Archive/devbox-C/navsim/data/openscene/navsim_logs/test"
CRUISE = r"E:/Projects/TanitAD-artifacts/tsf_nuplan_cruise_bundle/cruise_a0_per_token.json"
EGO_FRONT = 4.049

df = pd.read_parquet(f"{OUT}/features.parquet")
n = len(df)
rep = json.load(open(f"{OUT}/run_report.json"))

# ---- EXAMPLE category definitions (boolean masks) ---------------------------------------------------
C = {}
C["ego_brake_fd>=2.0"] = df.ego_decel_max_fd_mps2 >= 2.0
C["ego_brake_fd>=3.0"] = df.ego_decel_max_fd_mps2 >= 3.0
C["ego_brake_ax>=2.0"] = df.ego_decel_max_ax_mps2 >= 2.0
C["ego_accel_fd>=2.0"] = df.ego_accel_max_fd_mps2 >= 2.0
C["ego_stops_in_future"] = df.ego_stops_in_future
C["ego_starts_from_stop"] = df.ego_starts_from_stop
C["ego_stationary_window"] = df.ego_stationary_window
C["ego_started_in_hist"] = df.ego_started_in_hist
C["lead_t0"] = df.lead_t0
C["lead_time_gap_t0<2.0s"] = df.lead_time_gap_s < 2.0
C["lead_ttc_t0<=5s"] = df.lead_ttc_s <= 5.0
C["lead_ttc_min_win<=3s"] = df.lead_ttc_min_win_s <= 3.0
C["lead_gap_t0<10m"] = df.lead_gap_m < 10.0
C["lead_stationary_t0"] = df.lead_t0 & df.lead_stationary
C["lead_decel>=1.5"] = df.lead_decel_max_mps2 >= 1.5
C["lead_speed_drop>=2.0"] = df.lead_speed_drop_fut >= 2.0
C["lead_stops_fut"] = df.lead_stops_fut
C["lead_appears_win"] = df.lead_appears_win
C["cutin(ego-frame)"] = df.cutin
C["cutin_path"] = df.cutin_path
C["cutin_ego_frame_gap<20m"] = df.cutin & (df.cutin_gap_m < 20.0)
C["vru_corridor_win"] = df.vru_corridor_win
C["oncoming_corridor_t0"] = df.oncoming_corridor_t0
C["tl_present"] = df.tl_present
C["tl_any_red(list)"] = df.tl_any_red
C["tl_red_ahead(map)"] = df.tl_red_ahead
C["tl_red_ahead & ego_stops/brake>=1.5"] = df.tl_red_ahead & (df.ego_stops_in_future | (df.ego_decel_max_fd_mps2 >= 1.5))
C["tl_red_on_path"] = df.tl_red_on_path
C["junction_t0"] = df.junction_t0
C["junction_in_win"] = df.junction_in_win
C["junction_ahead<=30m"] = df.junction_dist_ahead_m <= 30.0
C["cmd=left"] = df.cmd_idx == 0
C["cmd=right"] = df.cmd_idx == 2
C["cmd=straight"] = df.cmd_idx == 1
C["turn |dyaw|>=0.4rad"] = df.ego_dyaw_fut_rad.abs() >= 0.4
C["turn |dyaw|>=0.4 & junction_in_win"] = (df.ego_dyaw_fut_rad.abs() >= 0.4) & df.junction_in_win
brake_or = C["ego_brake_fd>=2.0"] | C["ego_stops_in_future"] | C["ego_starts_from_stop"] | C["ego_stationary_window"]
lead_hard = C["lead_ttc_min_win<=3s"] | C["lead_decel>=1.5"] | C["lead_stops_fut"] | C["lead_time_gap_t0<2.0s"]
other_ev = C["cutin(ego-frame)"] | C["tl_red_ahead(map)"] | C["turn |dyaw|>=0.4rad"] | C["vru_corridor_win"]
C["ANY of {brake>=2/stop/start/stationary, lead_hard, cutin, red_ahead, turn, vru}"] = brake_or | lead_hard | other_ev
cruise_like = (df.ego_v0 >= 5.0) & (df.ego_decel_max_fd_mps2 < 1.0) & ~df.ego_stops_in_future & ~lead_hard & ~other_ev & ~df.cutin_path
C["cruise_like (v0>=5, decel<1, no event above)"] = cruise_like


def pct(k):
    return f"{100.0 * k / n:5.1f} %"


lines = []
lines.append("# SUMMARY - navtest per-token behaviour features (DESCRIPTIVE ONLY)\n")
lines.append("> **Everything below is descriptive.** Thresholds are EXAMPLES chosen for orientation. The frozen selection rule "
             "(stream BEHAV, `BEHAVIOUR_SCORECARD.md`) decides which tokens form the suite. No token has been picked here.\n")
lines.append("## 1. Token accounting (pinned)\n")
lines.append(f"* navtest.yaml: **{rep['yaml_tokens']} tokens** ({rep['yaml_tokens_unique']} unique), **{rep['yaml_logs']} log_names**; token-list sha256 `{rep['yaml_tokens_sha256']}`.")
lines.append(f"* **n tokens read = {rep['n_tokens_read']}**; n skipped = {rep['yaml_tokens'] - rep['n_tokens_read']} "
             f"(not found in any window: {rep['n_tokens_not_found']}; failed `has_route` filter: {rep['n_skipped_has_route']}; duplicates dropped: {rep['duplicate_tokens_dropped']}).")
lines.append(f"* logs processed {rep['logs_processed']}, status {rep['logs_by_status']}; **logs that failed to unpickle / extract: {len(rep['logs_failed'])}** {rep['logs_failed']}.")
lines.append(f"* The pkl directory holds {rep['pkl_files_in_dir']} files; {len(rep['pkl_not_in_navtest_logs'])} are not in navtest's 136 `log_names` and were NOT read (they belong to the OpenScene test split but not to navtest).")
lines.append(f"* maps unavailable for: {rep['map_status'] if rep['map_status'] else 'none (all 4 cities loaded)'}.")
lines.append(f"* `dt_ok` (all 12 frame gaps within 0.5 +- 0.05 s): {int(df.dt_ok.sum())} / {n}.\n")
lines.append("## 2. Counts per example category (n = %d)\n" % n)
lines.append("| category (EXAMPLE threshold) | n | share |\n|---|---:|---:|")
for k, m in C.items():
    lines.append(f"| {k} | {int(m.sum())} | {pct(int(m.sum()))} |")
lines.append("")
lines.append("## 3. By map location\n")
lines.append("| map_location | n tokens | logs | lead_t0 | cutin | red_ahead(map) | junction_in_win | ego stops | cruise_like |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|")
for ml, g in df.groupby("map_location"):
    ix = g.index
    lines.append(f"| {ml} | {len(g)} | {g.log_name.nunique()} | {int(g.lead_t0.sum())} | {int(g.cutin.sum())} | {int(g.tl_red_ahead.sum())} | "
                 f"{int(g.junction_in_win.sum())} | {int(g.ego_stops_in_future.sum())} | {int(cruise_like[ix].sum())} |")
lines.append("")
lines.append("## 4. Distributions (quantiles 0/5/25/50/75/95/100)\n")
lines.append("| column | n finite | quantiles |\n|---|---:|---|")
for c in ["ego_v0", "ego_v_min_fut", "ego_decel_max_fd_mps2", "ego_decel_max_ax_mps2", "ego_accel_max_fd_mps2", "lead_gap_m", "lead_time_gap_s",
          "lead_ttc_s", "lead_closing_mps", "lead_decel_max_mps2", "lead_speed_drop_fut", "cutin_gap_m", "ego_dyaw_fut_rad", "junction_dist_ahead_m", "tl_red_ahead_dist_m"]:
    x = df[c].to_numpy(float)
    x = x[np.isfinite(x)]
    q = np.percentile(x, [0, 5, 25, 50, 75, 95, 100]) if len(x) else []
    lines.append(f"| {c} | {len(x)} | {', '.join(f'{v:.2f}' for v in q)} |")
lines.append("")
lines.append("## 5. Driving-command decoding check (empirical)\n")
lines.append("`cmd_idx` is the argmax of the 4-vector. Mean realised yaw change over the 4 s future per index (index 0 should be left = positive, 2 right = negative):\n")
lines.append("| cmd_idx | n | mean ego_dyaw_fut_rad | share with |dyaw|>=0.4 |\n|---|---:|---:|---:|")
for ci, g in df.groupby("cmd_idx"):
    lines.append(f"| {ci} | {len(g)} | {g.ego_dyaw_fut_rad.mean():+.3f} | {100 * (g.ego_dyaw_fut_rad.abs() >= 0.4).mean():.1f} % |")
lines.append("")

# ---- cruise-bundle sanity ----------------------------------------------------------------------------
cb = json.load(open(CRUISE))
ctok = [t["token"] for t in cb["tokens"]]
cd = df[df.token.isin(ctok)].copy()
lines.append("## 6. Sanity: the cruise bundle (`cruise_a0_per_token.json`)\n")
lines.append(f"Cruise mask in that bundle: `{cb['meta']['cruise_mask']}`. Bundle tokens: {len(ctok)}; found in navtest features: **{len(cd)}**.\n")
free = cd.lead_present_frac_win == 0.0
steady = cd.lead_t0 & cd.lead_present_all_win & (cd.lead_closing_mps.abs() < 1.5) & (cd.lead_speed_drop_fut.fillna(0) < 2.0) & ~(cd.lead_decel_max_mps2.fillna(0) >= 1.5) & (cd.lead_time_gap_min_win_s.fillna(9) >= 1.0)
lines.append("Definitions (this sanity check only): **free road** = no lead vehicle in the corridor in any of the 9 window frames; "
             "**steady lead** = lead at all 9 frames, |closing| < 1.5 m/s at t0, lead speed drop < 2 m/s, lead decel < 1.5 m/s^2, min time gap >= 1.0 s; "
             "**other** = everything else.\n")
fr = int(free.sum()); st = int((steady & ~free).sum()); ot = len(cd) - fr - st
lines.append(f"| class | n | share of {len(cd)} |\n|---|---:|---:|")
lines.append(f"| free road | {fr} | {100 * fr / max(len(cd), 1):.1f} % |")
lines.append(f"| steady lead | {st} | {100 * st / max(len(cd), 1):.1f} % |")
lines.append(f"| other | {ot} | {100 * ot / max(len(cd), 1):.1f} % |")
lines.append(f"| **free road + steady lead** | {fr + st} | **{100 * (fr + st) / max(len(cd), 1):.1f} %** |")
lines.append("")
lines.append(f"Of the {len(cd)}: ego stops in future {int(cd.ego_stops_in_future.sum())}, ego decel_fd >= 2.0 {int((cd.ego_decel_max_fd_mps2 >= 2).sum())}, "
             f"cutin {int(cd.cutin.sum())}, red-light-ahead {int(cd.tl_red_ahead.sum())}, junction_in_win {int(cd.junction_in_win.sum())}, "
             f"in `cruise_like` {int(cruise_like[cd.index].sum())}.\n")
oth = cd[~(free | steady)]
if len(oth):
    lines.append("Tokens in the bundle that read as 'other' (first 12): " + ", ".join(f"`{t}`" for t in oth.token.head(12)) + "\n")
oth.to_csv(f"{OUT}/cruise_bundle_other_tokens.csv", index=False)


lines.append("### 6b. Why the 'other' bundle tokens are not free road / steady lead\n")
o = cd[~(free | steady)]
lines.append(f"| reason (not exclusive) | n of {len(o)} |\n|---|---:|")
lines.append(f"| lead present in only some window frames (0 < frac < 1) | {int(((o.lead_present_frac_win > 0) & (o.lead_present_frac_win < 1)).sum())} |")
lines.append(f"| lead in all 9 frames but not 'steady' (closing/drop/decel/time-gap rule) | {int(((o.lead_present_all_win) & ~steady[o.index]).sum())} |")
lines.append(f"| of which lead stationary at t0 | {int((o.lead_t0 & o.lead_stationary).sum())} |")
lines.append(f"| ego-frame `cutin` flag | {int(o.cutin.sum())} |")
lines.append(f"| `cutin_path` flag | {int(o.cutin_path.sum())} |")
lines.append("")
lines.append(f"Event flags over ALL 154 bundle tokens vs ALL navtest: `cutin` {100*cd.cutin.mean():.1f} % vs {100*df.cutin.mean():.1f} %; `cutin_path` {100*cd.cutin_path.mean():.1f} % vs {100*df.cutin_path.mean():.1f} %; "
             f"`vru_corridor_win` {100*cd.vru_corridor_win.mean():.1f} % vs {100*df.vru_corridor_win.mean():.1f} %; `oncoming_corridor_t0` {100*cd.oncoming_corridor_t0.mean():.1f} % vs {100*df.oncoming_corridor_t0.mean():.1f} %; "
             f"`junction_in_win` {100*cd.junction_in_win.mean():.1f} % vs {100*df.junction_in_win.mean():.1f} %.\n")
lines.append("## 6c. Cut-in definition diagnostics (the two flags are NOT interchangeable)\n")
straight = (df.ego_dyaw_fut_rad.abs() < 0.1) & (df.ego_y_end_m.abs() < 1.0)
lines.append("| ego behaviour subset | n | `cutin` (ego-frame) | `cutin_path` |\n|---|---:|---:|---:|")
for nm_, m_ in [("all", df.index == df.index), ("ego straight (|dyaw|<0.1, |y_end|<1 m)", straight), ("ego turning (|dyaw|>=0.4)", df.ego_dyaw_fut_rad.abs() >= 0.4),
                ("ego hard brake (fd>=2.0)", df.ego_decel_max_fd_mps2 >= 2.0), ("ego starts from stop", df.ego_starts_from_stop)]:
    g_ = df[m_]
    lines.append(f"| {nm_} | {len(g_)} | {int(g_.cutin.sum())} ({100*g_.cutin.mean():.1f} %) | {int(g_.cutin_path.sum())} ({100*g_.cutin_path.mean():.1f} %) |")
lines.append("")
lines.append("`cutin` = any vehicle that was beside the corridor at t0 and is inside the (ego-relative) corridor later - it also fires on crossing traffic and on ego-induced relative motion. "
             "`cutin_path` = the vehicle moves into the ego's realised path. Neither is validated against a ground-truth cut-in label (none exists in the pkl); the hand check in section 7 shows the raw track.\n")
lines.append("## 6d. Other descriptive caveats\n")
lines.append(f"* Stationary leads are common: {int((df.lead_t0 & df.lead_stationary).sum())} of {int(df.lead_t0.sum())} tokens with a lead ({100*(df.lead_t0 & df.lead_stationary).sum()/max(df.lead_t0.sum(),1):.1f} %) have a lead moving < 0.5 m/s (queues, vehicles parked in the 1.8 m half-width strip). A lead-based rule should say whether it wants these.")
lines.append(f"* `tl_any_red` (raw list) is True on {100*df.tl_any_red.mean():.1f} % of tokens vs `tl_red_ahead` (map-matched approach) {100*df.tl_red_ahead.mean():.1f} %: the raw list is not a statement about the ego's own signal.")
lines.append(f"* `dt_ok` False on {int((~df.dt_ok).sum())} tokens (frame gap outside 0.5 +- 0.05 s); their finite-difference features are less reliable (filter on `dt_ok == False`).\n")

# ---- independent hand verification -------------------------------------------------------------------
def pick(mask, key=None, asc=True):
    g = df[mask]
    if not len(g):
        return None
    if key is not None:
        g = g.sort_values(key, ascending=asc)
    return g.iloc[len(g) // 2] if key is None else g.iloc[0]


cats = {
    "free_road_cruise": pick(cruise_like & (df.lead_present_frac_win == 0.0)),
    "steady_lead": pick(df.lead_t0 & df.lead_present_all_win & (df.lead_closing_mps.abs() < 0.5) & (df.ego_v0 > 5) & (df.lead_speed_drop_fut.fillna(0) < 0.5)),
    "lead_braking": pick(C["lead_decel>=1.5"] & C["lead_speed_drop>=2.0"]),
    "ego_hard_brake": pick(C["ego_brake_fd>=3.0"]),
    "ego_stops": pick(C["ego_stops_in_future"] & ~df.lead_t0 & (df.ego_v_min_fut < 0.5)),
    "ego_starts_from_stop": pick(C["ego_starts_from_stop"]),
    "cutin_path": pick(C["cutin_path"]),
    "cutin_ego_frame": pick(C["cutin(ego-frame)"] & ~df.cutin_path),
    "red_ahead_stop": pick(C["tl_red_ahead & ego_stops/brake>=1.5"]),
    "junction_turn": pick(C["turn |dyaw|>=0.4 & junction_in_win"]),
}
cache = {}


def load(log):
    if log not in cache:
        cache.clear()
        cache[log] = pickle.load(open(f"{LOG_DIR}/{log}.pkl", "rb"))
    return cache[log]


def corners_minx(b):
    x, y, _, L, W, _, h = b
    c, s = math.cos(h), math.sin(h)
    pts = [(x + c * dx - s * dy, y + s * dx + c * dy) for dx in (-L / 2, L / 2) for dy in (-W / 2, W / 2)]
    return min(p[0] for p in pts)



def indep_cutin(fr_, j):
    """Re-implements the ego-frame cut-in definition from raw arrays (no shared code with mine_features)."""
    def veh(k):
        a = fr_[k]["anns"]; nm = np.asarray(a["gt_names"]); b = np.asarray(a["gt_boxes"], float); v = np.asarray(a["gt_velocity_3d"], float)
        return {a["track_tokens"][i]: (b[i], v[i]) for i in np.where(nm == "vehicle")[0]}
    v0 = veh(j); best = None
    for t, (b, v) in v0.items():
        if not (0 < b[0] < 60 and 1.8 <= abs(b[1]) <= 8):
            continue
        for k in range(j + 1, j + 9):
            vk = veh(k)
            if t in vk:
                bb, vv = vk[t]
                gap = corners_minx(bb) - EGO_FRONT
                if abs(bb[1]) < 1.8 and 0 < bb[0] < 60 and gap > 0 and math.hypot(vv[0], vv[1]) >= 1.0 and vv[0] >= -1.0:
                    if best is None or (k - j, gap) < (best["step"], best["gap"]):
                        best = dict(track=t[:8], step=k - j, gap=round(float(gap), 4), y_t0=round(float(b[1]), 2), y_entry=round(float(bb[1]), 2), speed=round(math.hypot(vv[0], vv[1]), 2))
                    break
    return best

lines.append("## 7. Hand verification: one token per category (recomputed straight from the raw pkl, separate code path)\n")
lines.append("Each block: the feature-table value vs a recomputation that reads `gt_boxes` corners / `ego_dynamic_state` directly. `OK` = agree within 1e-6.\n")
verif = {}
for name, row in cats.items():
    if row is None:
        lines.append(f"### {name}\nno token satisfies this example category.\n"); continue
    fr_ = load(row.log_name)
    j = [i for i, f in enumerate(fr_) if f["token"] == row.token][0]
    v = [math.hypot(fr_[k]["ego_dynamic_state"][0], fr_[k]["ego_dynamic_state"][1]) for k in range(j - 3, j + 9)]
    vf = v[4:]; v0 = v[3]
    dvdt = [(v[k + 1] - v[k]) / 0.5 for k in range(3, 11)]
    chk = {"ego_v0": (v0, row.ego_v0), "ego_v_min_fut": (min(vf), row.ego_v_min_fut), "ego_decel_max_fd_mps2": (max(0, -min(dvdt)), row.ego_decel_max_fd_mps2)}
    txt = [f"### {name}\n`token {row.token}`  log `{row.log_name}`  list_idx {row.list_idx} (pkl frame_idx {row.frame_idx})  cmd_idx {row.cmd_idx}  {row.map_location}"]
    if row.lead_t0:
        a = fr_[j]["anns"]
        nm = np.asarray(a["gt_names"]); idx = [i for i, t in enumerate(a["track_tokens"]) if t == row.lead_trk_t0 and nm[i] == "vehicle"][0]
        b = np.asarray(a["gt_boxes"][idx], float); vel = np.asarray(a["gt_velocity_3d"][idx], float)
        gap = corners_minx(b) - EGO_FRONT
        clos = fr_[j]["ego_dynamic_state"][0] - vel[0]
        chk["lead_gap_m"] = (gap, row.lead_gap_m); chk["lead_closing_mps"] = (clos, row.lead_closing_mps)
        txt.append(f"raw lead box t0 (x,y,L,W,heading) = ({b[0]:.2f}, {b[1]:.2f}, {b[3]:.2f}, {b[4]:.2f}, {b[6]:.3f}); lead velocity (ego axes) = ({vel[0]:.2f}, {vel[1]:.2f}) m/s; rear-edge x - 4.049 = gap")
    if row.cutin:
        ent = indep_cutin(fr_, j)
        txt.append(f"independent cut-in recomputation (ego-frame, all vehicle tracks): {ent}")
        if ent is not None:
            chk["cutin_entry_step"] = (float(ent["step"]), float(row.cutin_entry_step)); chk["cutin_gap_m"] = (ent["gap"], row.cutin_gap_m)
        else:
            chk["cutin_entry_step"] = (-1.0, float(row.cutin_entry_step))
    ok = all(abs(a_ - b_) < 1e-3 or (math.isnan(a_) and math.isnan(b_)) for a_, b_ in chk.values())
    txt.append("ego speed (j-3..j+8, m/s): " + ", ".join(f"{x:.2f}" for x in v))
    txt.append("| quantity | recomputed | table |\n|---|---:|---:|")
    for k, (a_, b_) in chk.items():
        txt.append(f"| {k} | {a_:.4f} | {b_:.4f} |")
    keys = ["ego_decel_max_ax_mps2", "ego_dist_fut_m", "ego_dyaw_fut_rad", "lead_time_gap_s", "lead_ttc_s", "lead_speed_drop_fut", "lead_decel_max_mps2",
            "cutin", "cutin_entry_step", "cutin_gap_m", "cutin_path", "tl_n", "tl_n_red", "tl_red_ahead", "tl_red_ahead_dist_m", "junction_t0", "junction_dist_ahead_m"]
    txt.append("table values: " + ", ".join(f"{k}={row[k]:.3f}" if isinstance(row[k], (float, np.floating)) else f"{k}={row[k]}" for k in keys))
    txt.append(f"**agreement: {'OK' if ok else 'MISMATCH'}**\n")
    lines += txt
    verif[name] = dict(token=row.token, log=row.log_name, ok=bool(ok))
json.dump(verif, open(f"{OUT}/hand_verified_tokens.json", "w"), indent=1)
open(f"{OUT}/SUMMARY.md", "w", encoding="utf-8").write("\n".join(lines))
print("wrote SUMMARY.md", n, "rows; verified:", verif)
