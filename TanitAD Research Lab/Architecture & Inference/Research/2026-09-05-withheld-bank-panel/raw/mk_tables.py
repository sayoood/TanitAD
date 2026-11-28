"""Render the panel's markdown tables FROM panel_report.json — RESULT.md quotes
these, never hand-typed numbers. usage: python mk_tables.py <panel_report.json> [out.md]
"""
import json
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")   # cp1252 console on the dev box
except Exception:
    pass

J = json.load(open(sys.argv[1], encoding="utf-8"))
OUT = []


def w(s=""):
    OUT.append(s)


def f(x, d=3):
    try:
        if x is None:
            return "—"
        return f"{float(x):.{d}f}"
    except Exception:
        return str(x)


def pr(r, d=3):
    """paired/level dict -> 'delta [lo, hi] SEP/ns'"""
    if not isinstance(r, dict) or "delta" not in r and "mean" not in r:
        return "n/a" if not isinstance(r, dict) else r.get("status", "n/a")
    if "delta" in r:
        return f"{f(r['delta'], d)} [{f(r['lo'], d)}, {f(r['hi'], d)}] {'**SEP**' if r.get('separated') else 'ns'}"
    return f"{f(r['mean'], d)} [{f(r.get('lo'), d)}, {f(r.get('hi'), d)}]"


arms = [a for a, r in J["arms"].items() if r.get("status") == "MEASURED"]
FAM = [("LON_speed_mae_mps", "speed MAE (m/s)"), ("LON_along_mae_m", "along MAE (m)"),
       ("LON_accel_mae_mps2", "accel MAE (m/s²)"), ("LAT_cross_mae_m", "cross MAE (m)"),
       ("LAT_heading_mae_deg", "heading MAE (°)"), ("LAT_yaw_rate_mae_radps", "yaw-rate MAE (rad/s)"),
       ("TAC_traj_lat_correct", "tactical lat agree"), ("TAC_traj_lon_correct", "tactical lon agree"),
       ("ade_m", "ADE (m)")]

w(f"_scope: {J['_scope']}_  \n_n_windows {J['n_windows']} from {J['n_episodes_pool']} val episodes; "
  f"estimator {J['estimator']}; n_boot {J['n_boot']}; device {J['device']}_")
w()
w("### The echo instrument (kept regime = deployed; H-ECHO-8's gate 2b, `min_degradation` 0.05)")
w()
w("| arm | bank policy | gate2 (structural) | gate2b verdict | wrong SCENE hurts | wrong EGO hurts | gate raised | regress re-probed with REAL frames |")
w("|---|---|---|---|---|---|---|---|")
for a in arms:
    r = J["arms"][a]
    g2b = r["gate2b"]["sources"]
    real = r.get("gate2b_realframes")
    real_s = "—" if not real else f"{real['verdict']} (scene {real['sources']['scene']['degradation_rel']:+.4f})"
    w(f"| {a} | {r['withheld_bank'].get('mode')} | {r['gate2'].get('verdict')} | **{r['gate2b']['verdict']}** | "
      f"{g2b['scene']['degradation_rel']:+.4f} ({'sep' if g2b['scene']['separated'] else 'ns'}) | "
      f"{g2b['ego']['degradation_rel']:+.4f} ({'sep' if g2b['ego']['separated'] else 'ns'}) | "
      f"{r['GATE'].get('raised')} | {real_s} |")
w()
w("Withheld-regime scene half (ego is zero there, so only the scene column reads):")
w()
w("| arm | wrong SCENE hurts (withheld regime) | constant-predictor control (scene / ego) |")
w("|---|---|---|")
for a in arms:
    r = J["arms"][a]
    gw = r["gate2b_withheld_regime"]["sources"]["scene"]
    gc = r["gate2b_constant_predictor_control"]["sources"]
    w(f"| {a} | {gw['degradation_rel']:+.4f} ({'sep' if gw['separated'] else 'ns'}) | "
      f"{gc['scene']['degradation_rel']:+.4f} / {gc['ego']['degradation_rel']:+.4f} |")
w()

for reg, title in (("w", "WITHHELD regime (keep = 0; the training-time dropout regime, each arm under its OWN bank)"),
                   ("k", "KEPT regime (keep = 1; the deployed regime)")):
    w(f"### Four families on the SELECTED 2 s plan — {title}")
    w()
    w("Levels (episode-cluster bootstrap mean [lo, hi]); paired vs A0 on the same windows (negative = better than A0, except the tactical-agreement rows where positive = better).")
    w()
    hdr = "| family / metric | " + " | ".join(arms) + " |"
    w(hdr); w("|---|" + "---|" * len(arms))
    for mk, name in FAM:
        cells = []
        for a in arms:
            lv = J["arms"][a]["regimes"][reg]["plan_2s_families"]["levels"].get(mk, {})
            cells.append(pr(lv))
        w(f"| {name} — level | " + " | ".join(cells) + " |")
        cells = []
        for a in arms:
            if a == "A0_fixed":
                cells.append("(ref)")
            else:
                pv = J["arms"][a].get("paired_vs_A0", {}).get(reg, {}).get("plan_2s", {}).get(mk)
                cells.append(pr(pv))
        w(f"| {name} — Δ vs A0 | " + " | ".join(cells) + " |")
    w()
    w("Model-free references on the same 2 s grid (ha / ha0 / ha0_ext) — the arm's paired delta vs `ha` (negative = beats hold-action):")
    w()
    w("| metric | " + " | ".join(arms) + " | ha level | ha0_ext level |")
    w("|---|" + "---|" * (len(arms) + 2))
    for mk, name in FAM[:2] + FAM[3:4] + FAM[-1:]:
        cells = [pr(J["arms"][a]["regimes"][reg]["plan_2s_families"]["paired_arm_minus_ha"].get(mk)) for a in arms]
        ref = J["controls"]["plan_2s_families"]
        w(f"| {name} | " + " | ".join(cells) + f" | {f(ref['ha'][mk])} | {f(ref['ha0_ext'][mk])} |")
    w()

# A1's separating regimes + A1 vs A2
if "A1_pred" in arms:
    w("### A1 (`pred`): separating the WEIGHTS from the BANK, and the shuffle control")
    w()
    w("| regime of A1 | what it is | speed MAE Δ vs A0[w] | along MAE Δ vs A0[w] | ADE Δ vs A0[w] |")
    w("|---|---|---|---|---|")
    desc = {"w": "own weights, own `pred` bank", "w_fixed": "own weights, bank swapped back to fixed 10 m/s",
            "w_shuf": "own weights, `pred` bank at a PERMUTED row's speed (gain must vanish)"}
    for tag in ("w", "w_fixed", "w_shuf"):
        pv = J["arms"]["A1_pred"].get("paired_vs_A0", {}).get(tag, {}).get("plan_2s", {})
        if not pv:
            continue
        w(f"| {tag} | {desc[tag]} | {pr(pv.get('LON_speed_mae_mps'))} | {pr(pv.get('LON_along_mae_m'))} | {pr(pv.get('ade_m'))} |")
    w()
if "A1_minus_A2" in J:
    w("### A1 − A2 (the control that must NOT gain the same; negative = A1 better)")
    w()
    w("| regime | speed MAE | along MAE | cross MAE | ADE |")
    w("|---|---|---|---|---|")
    for tag in ("w", "k"):
        d = J["A1_minus_A2"][tag]
        w(f"| {tag} | {pr(d.get('LON_speed_mae_mps'))} | {pr(d.get('LON_along_mae_m'))} | "
          f"{pr(d.get('LAT_cross_mae_m'))} | {pr(d.get('ade_m'))} |")
    w()

w("### Withheld ceilings (T0, model-free oracle-in-vocabulary of the bank the withheld regime would decode), by v0 band")
w()
for a in arms:
    c = J["arms"][a]["withheld_ceilings"]
    lv = c["levels"]
    w(f"**{a}** — fixed 10 m/s {pr(lv['fixed_10ms'])} · own prediction {pr(lv['own_pred_speed_w'])} · "
      f"true v0 (LEAK bound) {pr(lv['true_v0_LEAK_BOUND'])}; pred − fixed {pr(c['paired_pred_minus_fixed'])}; "
      f"pred − true {pr(c['paired_pred_minus_true'])}")
    w()
    w("| v0 band (m/s) | n | fixed | own pred | true v0 | decoded bank (w) |")
    w("|---|---|---|---|---|---|")
    dec_b = {tuple(r_["v0_band_ms"]): r_["oiv_m"] for r_ in J["arms"][a]["regimes"]["w"]["oiv_decoded_bank_by_v0_band"]}
    for r_ in c["by_v0_band"]:
        w(f"| {r_['v0_band_ms'][0]}–{r_['v0_band_ms'][1]} | {r_['n']} | {f(r_['fixed_10ms'])} | {f(r_['own_pred_speed_w'])} | "
          f"{f(r_['true_v0_LEAK_BOUND'])} | {f(dec_b.get(tuple(r_['v0_band_ms'])))} |")
    w()

w("### The model's own 2 s speed (`g_tac`), selection profile, tactical heads")
w()
w("| arm | regime | speed MAE vs GT 2 s (m/s) | speed MAE vs v0 | straight-ahead share (sel) | a* straight share | lat acc / majority | lon acc / majority |")
w("|---|---|---|---|---|---|---|---|")
for a in arms:
    for reg in J["arms"][a]["regimes"]:
        r = J["arms"][a]["regimes"][reg]
        t = r["tactical_head"]
        lat = f"{f(t['lat'].get('accuracy'))} / {f(t['lat'].get('majority_class_control'))}" if "accuracy" in t.get("lat", {}) else t.get("lat", {}).get("status", "—")
        lon = f"{f(t['lon'].get('accuracy'))} / {f(t['lon'].get('majority_class_control'))}" if "accuracy" in t.get("lon", {}) else t.get("lon", {}).get("status", "—")
        w(f"| {a} | {reg} | {pr(r['bank_speed_pred_mae_vs_gt2s_mps'])} | {f(r['bank_speed_pred_mae_vs_v0_mps'])} | "
          f"{f(r['selection']['straight_ahead_idx67_frac'])} | {f(r['selection']['astar_straight_frac'])} | {lat} | {lon} |")
w()

w("### D3 (endpoint gradient) and D4 (plan predictability from (v0, a0, yaw-rate) alone)")
w()
hum = J["controls"]["D4_human_plan"]
w(f"D4 on the HUMAN plan: 1−R² = {f(hum['rel_err_1_minus_r2'])} (λ {hum['lambda']}, n_fit {hum['n_fit']}, n_score {hum['n_score']}, d_in {hum['d_in']}, d_out {hum['d_out']}; constant-only control {f(hum['constant_only_control'])}). A plan MORE self-predictable than the human (smaller) is the copycat signature.")
w()
w("| arm | D3 ‖J‖ mean | D3 median | D3 per channel (v0 / a / yaw-rate / κ) | D4 kept 1−R² | D4 withheld 1−R² |")
w("|---|---|---|---|---|---|")
for a in arms:
    d3 = J["arms"][a]["D3_endpoint_gradient"]; d4 = J["arms"][a]["D4_plan_predictability"]
    pc = d3["per_channel_mean"]
    w(f"| {a} | {f(d3['mean'])} | {f(d3['median'])} | {f(pc['v0'])} / {f(pc['a_long'])} / {f(pc['yaw_rate'])} / {f(pc['curvature'])} | "
      f"{f(d4['k']['rel_err_1_minus_r2'])} (λ {d4['k']['lambda']}) | {f(d4['w']['rel_err_1_minus_r2'])} (λ {d4['w']['lambda']}) |")
w()

w("### Gate 1 on the goal rows (kept regime): paired vs the kinematic references (the primary slot is the LONGEST horizon)")
w()
w("| arm | slot (τ) | n | arm ADE | vs ha: Δ [CI] passes | vs ha0_ext: Δ [CI] passes | vs constant_only |")
w("|---|---|---|---|---|---|---|")
for a in arms:
    for row in J["arms"][a]["regimes"]["k"]["gate1"]["slots"]:
        vs = row["vs"]
        w(f"| {a} | {row['slot']} ({row['tau_s']} s) | {row['n_windows']} | {f(row['arm_mean'])} | "
          f"{f(vs['ha']['delta_ref_minus_arm'])} [{f(vs['ha']['ci'][0])}, {f(vs['ha']['ci'][1])}] {vs['ha']['passes']} | "
          f"{f(vs['ha0_ext']['delta_ref_minus_arm'])} [{f(vs['ha0_ext']['ci'][0])}, {f(vs['ha0_ext']['ci'][1])}] {vs['ha0_ext']['passes']} | "
          f"{f(vs['constant_only']['delta_ref_minus_arm'])} |")
w()
w("### Controls that must read known values")
w()
c = J["controls"]
w("| slot (τ) | n | ha | ha0 | ha0_ext | constant_only |")
w("|---|---|---|---|---|---|")
for r_ in c["goal_rows_by_slot"]:
    w(f"| {r_['slot']} ({r_['tau_s']} s) | {r_['n']} | {f(r_['ha'])} | {f(r_['ha0'])} | {f(r_['ha0_ext'])} | {f(r_['constant_only'])} |")
w()
w(f"`ha` vs `ha0_ext` max |Δ| on the goal rows: {f(c['ha_equals_ha0_ext_max_abs_m'], 4)} m. {c['_reads']}")
w()
if arms:
    pf = J["arms"][arms[0]]["regimes"]["k"].get("raw_pixel_floor", {})
    if pf:
        w(f"Raw-pixel floor (ridge, {pf['meta']}):")
        w()
        w("| slot | n | floor ADE | " + " | ".join(f"{a} ADE" for a in arms) + " |")
        w("|---|---|---|" + "---|" * len(arms))
        for i, r_ in enumerate(pf["per_slot"]):
            cells = [f(J["arms"][a]["regimes"]["k"]["raw_pixel_floor"]["per_slot"][i]["arm_ade"]) for a in arms]
            w(f"| {r_['slot']} ({r_['tau_s']} s) | {r_['n']} | {f(r_['floor_ade'])} | " + " | ".join(cells) + " |")
w()
text = "\n".join(OUT)
if len(sys.argv) > 2:
    open(sys.argv[2], "w", encoding="utf-8").write(text)
print(text)
