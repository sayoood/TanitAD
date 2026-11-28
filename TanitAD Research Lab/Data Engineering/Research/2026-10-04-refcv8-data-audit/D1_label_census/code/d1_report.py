"""Render the numbers_<tag>.json files into markdown tables (no hand transcription of numbers)."""
from __future__ import annotations

import json
import sys
from pathlib import Path


def f(x, d=2):
    if x is None:
        return "n/a"
    if isinstance(x, float):
        if x != x:
            return "n/a"
        return f"{x:.{d}f}"
    return str(x)


def n(x):
    return f"{int(x):,}"


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def render(tr, ev):
    L = []
    # T1
    keys = list(tr["T1_channels"].keys())
    rows = []
    for k in keys:
        a, b = tr["T1_channels"][k], ev["T1_channels"][k]
        rows.append([k, n(a["n"]), f(a["pct_of_windows"]) + " %", n(b["n"]), f(b["pct_of_windows"]) + " %"])
    L.append("### T1. Channel coverage per window\n")
    L.append(table(["channel", "TRAIN n", "TRAIN %", "EVAL139 n", "EVAL139 %"], rows))
    ci = ev.get("T1_boot_ci95", {})
    if ci:
        L.append("\nEVAL139 episode-cluster bootstrap 95 % CI (139 clips, B=2000): " +
                 "; ".join(f"{k} {f(v[0],1)}-{f(v[1],1)} %" for k, v in ci.items()))
    L.append(f"\nTRAIN: {n(tr['n_windows'])} windows over {n(tr['n_clips'])} clips (windows per clip min/median/max "
             f"{tr['windows_per_clip']['min']}/{tr['windows_per_clip']['median']:.0f}/{tr['windows_per_clip']['max']}); "
             f"clips with >=1 tactical window {n(tr['clips_with_any_tactical_window'])}; tactical windows per clip "
             f"min/median/max {tr['in_band_windows_per_clip']['min']}/{tr['in_band_windows_per_clip']['median']:.0f}/"
             f"{tr['in_band_windows_per_clip']['max']}. EVAL139: {n(ev['n_windows'])} windows over {n(ev['n_clips'])} clips; "
             f"clips with >=1 tactical window {n(ev['clips_with_any_tactical_window'])}.")
    # T2
    L.append("\n### T2. Tactical lateral / longitudinal classes (windows carrying the class)\n")
    for fam, key in (("lateral (`lat_v7`)", "T2_lat"), ("longitudinal (`lon_v7`)", "T2_lon")):
        rows = []
        for k in tr[key]:
            a, b = tr[key][k], ev[key][k]
            rows.append([k, n(a["windows"]), f(a["pct_of_all"]) + " %", f(a["pct_of_in_band"], 1) + " %", n(a["clips"]),
                         n(b["windows"]), f(b["pct_of_all"]) + " %", f(b["pct_of_in_band"], 1) + " %"])
        L.append(f"**{fam}**\n")
        L.append(table(["class", "TRAIN windows", "% of all", "% of in-band", "TRAIN clips", "EVAL windows", "% of all", "% of in-band"], rows))
        L.append("")
    # T3
    L.append("### T3. The 22 goal tokens: positives and scored cells (windows)\n")
    rows = []
    for k in tr["T3_goal_tokens"]:
        a, b = tr["T3_goal_tokens"][k], ev["T3_goal_tokens"][k]
        rows.append([k, n(a["positive_windows"]), f(a["pct_of_all_positive"], 3) + " %", n(a["positive_clips"]),
                     n(a["scored_windows_as_run"]), f(a["pct_of_all_scored"], 2) + " %", n(a["scored_windows_census_state"]),
                     n(b["positive_windows"]), n(b["scored_windows_as_run"])])
    L.append(table(["token", "TRAIN positive windows", "% of all", "pos. clips", "TRAIN scored (as the workers saw it)", "% of all",
                    "TRAIN scored (train-blob census state)", "EVAL positive", "EVAL scored"], rows))
    gs = tr["goal_state_difference"]
    L.append(f"\nTwo label-module states are tabulated for TRAIN (see FINDINGS): windows whose scored-cell count differs between them: "
             f"{n(gs['n_windows_scored_count_differs'])}; cells differing by token: {json.dumps(gs['bits_differing'])}.")
    # T4
    L.append("\n### T4. In-band fraction as a function of t_now (1-s bins over the clip)\n")
    rows = []
    for k in tr["T4_inband_by_tnow"]:
        a = tr["T4_inband_by_tnow"][k]
        b = ev["T4_inband_by_tnow"].get(k, {"windows": 0, "in_band": 0, "pct_in_band": float("nan")})
        rows.append([k, n(a["windows"]), n(a["in_band"]), f(a["pct_in_band"], 1) + " %", n(b["windows"]), n(b["in_band"]), f(b["pct_in_band"], 1) + " %"])
    L.append(table(["t_now bin (s)", "TRAIN windows", "TRAIN in-band", "TRAIN %", "EVAL windows", "EVAL in-band", "EVAL %"], rows))
    # T5
    L.append("\n### T5. Turn windows (|dyaw over [0,6] s| >= 30 deg, windows with the full 6 s in the clip) and the tactical lateral label\n")
    rows = []
    for nm in ("all_turn", "left", "right"):
        a, b = tr["T5_turn_windows"][nm], ev["T5_turn_windows"][nm]
        rows.append([nm, n(a["n"]), n(a["lat_GT_present"]), f(a["pct_present"], 1) + " %", f(a["pct_side_agree_where_present"], 1) + " %",
                     n(b["n"]), n(b["lat_GT_present"]), f(b["pct_present"], 1) + " %", f(b["pct_side_agree_where_present"], 1) + " %"])
    L.append(table(["turn windows", "TRAIN n", "lat GT present", "% present", "side agrees where present", "EVAL n", "lat GT present", "% present", "side agrees where present"], rows))
    for nm, lab in (("gentle_10_30", "10-30 deg"), ("straight_lt_10", "< 10 deg")):
        L.append(f"\n{lab}: TRAIN n {n(tr['T5_turn_windows'][nm]['n'])}, lat GT present {f(tr['T5_turn_windows'][nm]['pct_lat_present'],1)} %; "
                 f"EVAL n {n(ev['T5_turn_windows'][nm]['n'])}, {f(ev['T5_turn_windows'][nm]['pct_lat_present'],1)} %")
    a, b = tr["T5_route_definition_all_windows"], ev["T5_route_definition_all_windows"]
    L.append(f"\nRoute-following package definition (terminal heading of the slot-50 -> 60 segment, >= 30 deg, path >= 5 m) on ALL windows: "
             f"TRAIN GT-turn {n(a['GT-turn'])} (L {n(a['L'])}, R {n(a['R'])}), lat IGNORE on {f(a['pct_lat_IGNORE_on_GT-turn'],1)} %; "
             f"EVAL139 GT-turn {n(b['GT-turn'])} (L {n(b['L'])}, R {n(b['R'])}), lat IGNORE on {f(b['pct_lat_IGNORE_on_GT-turn'],1)} %.")
    # T6
    L.append("\n### T6. The fed nav token against where a turn actually is\n")
    A, B = tr["T6_nav"], ev["T6_nav"]
    rows = [
        ["windows with nav token left / right / follow", f"{n(A['windows_nav_left'])} / {n(A['windows_nav_right'])} / {n(A['windows_nav_follow'])}",
         f"{n(B['windows_nav_left'])} / {n(B['windows_nav_right'])} / {n(B['windows_nav_follow'])}"],
        ["clips with nav token left / right / follow", f"{tr['T6_clip_nav_counts']['left']} / {tr['T6_clip_nav_counts']['right']} / {tr['T6_clip_nav_counts']['follow']}",
         f"{ev['T6_clip_nav_counts']['left']} / {ev['T6_clip_nav_counts']['right']} / {ev['T6_clip_nav_counts']['follow']}"],
        ["L/R windows (denominator)", n(A["windows_nav_LR"]), n(B["windows_nav_LR"])],
        ["L/R windows, NO turn START within the next 6 s (labels)", f"{n(A['LR_no_turn_start_within_6s'])} ({f(A['pct_LR_no_turn_start_within_6s'],1)} %)",
         f"{n(B['LR_no_turn_start_within_6s'])} ({f(B['pct_LR_no_turn_start_within_6s'],1)} %)"],
        ["... and none IN PROGRESS either (6 s)", f"{n(A['LR_no_turn_start_or_in_progress_within_6s'])} ({f(A['pct_LR_no_turn_start_or_in_progress_within_6s'],1)} %)",
         f"{n(B['LR_no_turn_start_or_in_progress_within_6s'])} ({f(B['pct_LR_no_turn_start_or_in_progress_within_6s'],1)} %)"],
        ["L/R windows, NO turn START within the next 10 s (labels)", f"{n(A['LR_no_turn_start_within_10s'])} ({f(A['pct_LR_no_turn_start_within_10s'],1)} %)",
         f"{n(B['LR_no_turn_start_within_10s'])} ({f(B['pct_LR_no_turn_start_within_10s'],1)} %)"],
        ["... and none IN PROGRESS either (10 s)", f"{n(A['LR_no_turn_start_or_in_progress_within_10s'])} ({f(A['pct_LR_no_turn_start_or_in_progress_within_10s'],1)} %)",
         f"{n(B['LR_no_turn_start_or_in_progress_within_10s'])} ({f(B['pct_LR_no_turn_start_or_in_progress_within_10s'],1)} %)"],
        ["L/R windows with 6 s of future in the clip: NO realised >= 30 deg turn in 6 s (poses)", f"{n(A['realised_LR_no_30deg_turn_in_6s'])} of {n(A['realised_LR_windows_full6'])} ({f(A['pct_realised_LR_no_30deg_turn_in_6s'],1)} %)",
         f"{n(B['realised_LR_no_30deg_turn_in_6s'])} of {n(B['realised_LR_windows_full6'])} ({f(B['pct_realised_LR_no_30deg_turn_in_6s'],1)} %)"],
        ["... and not even 10 deg", f"{f(A['pct_realised_LR_no_10deg_in_6s'],1)} %", f"{f(B['pct_realised_LR_no_10deg_in_6s'],1)} %"],
        ["realised >= 30 deg turn in the next 6 s: windows", n(A["realised_turn_windows_full6"]), n(B["realised_turn_windows_full6"])],
        ["... of which the fed token is FOLLOW", f"{n(A['realised_turn_with_nav_follow'])} ({f(A['pct_realised_turn_with_nav_follow'],1)} %)",
         f"{n(B['realised_turn_with_nav_follow'])} ({f(B['pct_realised_turn_with_nav_follow'],1)} %)"],
        ["... of which the fed token matches the turn side", f"{n(A['realised_turn_with_matching_token'])} ({f(A['pct_realised_turn_with_matching_token'],1)} %)",
         f"{n(B['realised_turn_with_matching_token'])} ({f(B['pct_realised_turn_with_matching_token'],1)} %)"],
        ["... of which the fed token is the WRONG side", n(A["realised_turn_with_wrong_side_token"]), n(B["realised_turn_with_wrong_side_token"])],
        ["median / p5-p95 seconds to the next labelled turn start (L/R windows that have one)",
         f"{f(A['ttn_s_quantiles_5_25_50_75_95_on_LR_with_turn'][2],1)} ({f(A['ttn_s_quantiles_5_25_50_75_95_on_LR_with_turn'][0],1)}-{f(A['ttn_s_quantiles_5_25_50_75_95_on_LR_with_turn'][4],1)})",
         f"{f(B['ttn_s_quantiles_5_25_50_75_95_on_LR_with_turn'][2],1)} ({f(B['ttn_s_quantiles_5_25_50_75_95_on_LR_with_turn'][0],1)}-{f(B['ttn_s_quantiles_5_25_50_75_95_on_LR_with_turn'][4],1)})"],
    ]
    L.append(table(["quantity", "TRAIN", "EVAL139"], rows))
    # T7
    L.append("\n### T7. The per-clip speed ceiling against the window's own realised speed ([+2,+6] s ahead; windows with the full 6 s)\n")
    A, B = tr["T7_ceiling"], ev["T7_ceiling"]
    rows = [["all fed bins", n(A["n_windows_full6_with_ceiling"]), f"{n(A['realised_max_26_exceeds_fed_ceiling'])} ({f(A['pct_exceeds'],1)} %)",
             f"{n(A['realised_bin_below_fed_bin'])} ({f(A['pct_below'],1)} %)", f"{n(A['realised_bin_equals_fed_bin'])} ({f(A['pct_equal'],1)} %)",
             n(B["n_windows_full6_with_ceiling"]), f"{f(B['pct_exceeds'],1)} %", f"{f(B['pct_below'],1)} %", f"{f(B['pct_equal'],1)} %"]]
    for k in A["per_fed_bin"]:
        a, b = A["per_fed_bin"][k], B["per_fed_bin"][k]
        rows.append([f"fed {k} km/h", n(a["windows"]), f"{n(a['exceeds'])} ({f(a['pct_exceeds'],1)} %)", f"{n(a['below'])} ({f(a['pct_below'],1)} %)",
                     f"{n(a['equal'])} ({f(a['pct_equal'],1)} %)", n(b["windows"]), f"{f(b['pct_exceeds'],1)} %", f"{f(b['pct_below'],1)} %", f"{f(b['pct_equal'],1)} %"])
    L.append(table(["fed ceiling", "TRAIN windows", "realised max EXCEEDS ceiling", "realised bin BELOW fed bin", "same bin", "EVAL windows", "exceeds", "below", "same"], rows))
    L.append(f"\nMargin: realised max exceeds the fed ceiling by > 5 km/h on {f(A['exceeds_by_gt_5_kmh']['pct'],1)} % of TRAIN windows ({n(A['exceeds_by_gt_5_kmh']['n'])}), "
             f"by > 10 km/h on {f(A['exceeds_by_gt_10_kmh']['pct'],1)} % ({n(A['exceeds_by_gt_10_kmh']['n'])}); it is more than 20 km/h BELOW the fed ceiling on "
             f"{f(A['realised_max_below_fed_ceiling_by_gt_20_kmh']['pct'],1)} % ({n(A['realised_max_below_fed_ceiling_by_gt_20_kmh']['n'])}); median realised-minus-ceiling "
             f"{f(A['median_realised_minus_ceiling_kmh'],1)} km/h (EVAL139: >5 km/h {f(B['exceeds_by_gt_5_kmh']['pct'],1)} %, >10 km/h {f(B['exceeds_by_gt_10_kmh']['pct'],1)} %).")
    L.append(f"\nIn-band windows only (the span the ceiling's own definition describes): TRAIN n {n(A['in_band_only']['n'])}: exceeds {f(A['in_band_only']['pct_exceeds'],1)} %, "
             f"below {f(A['in_band_only']['pct_below'],1)} %, same {f(A['in_band_only']['pct_equal'],1)} %; EVAL n {n(B['in_band_only']['n'])}: exceeds "
             f"{f(B['in_band_only']['pct_exceeds'],1)} %, below {f(B['in_band_only']['pct_below'],1)} %, same {f(B['in_band_only']['pct_equal'],1)} %. "
             f"Current speed v0 above the fed ceiling: TRAIN {f(A['v0_above_fed_ceiling']['pct'],1)} %, EVAL {f(B['v0_above_fed_ceiling']['pct'],1)} % of windows.")
    # T8
    L.append("\n\n### T8. The per-clip constants against t_now (2-s bins; the nav token and the ceiling are ONE value per clip on every window)\n\n")
    rows = []
    for k in tr["T8_by_tnow_2s"]:
        a = tr["T8_by_tnow_2s"][k]
        b = ev["T8_by_tnow_2s"].get(k)
        rows.append([k, n(a["windows"]), f(a["pct_tactical_GT"], 1) + " %", f(a["pct_nav_LR"], 1) + " %",
                     f(a["pct_LR_with_turn_start_within_6s"], 1) + " %", f(a["pct_LR_no_30deg_turn_in_6s_poses"], 1) + " %",
                     f(a["pct_turn_with_matching_token"], 1) + " %", f(a["pct_turn_with_follow_token"], 1) + " %",
                     f(a["pct_exceeds"], 1) + " %", f(a["pct_below"], 1) + " %", f(a["pct_same"], 1) + " %",
                     ("n/a" if b is None else f(b["pct_LR_with_turn_start_within_6s"], 1) + " %"),
                     ("n/a" if b is None else f(b["pct_exceeds"], 1) + " %")])
    L.append(table(["t_now bin", "TRAIN windows", "tactical GT", "nav = L/R", "L/R with a turn start <= 6 s (labels)",
                    "L/R with NO >=30 deg turn in 6 s (poses)", "realised turns: token matches", "realised turns: token FOLLOW",
                    "ceiling exceeded", "realised bin below fed", "same bin", "EVAL L/R with turn start <= 6 s", "EVAL ceiling exceeded"], rows))
    L.append("\n\n**In-band windows, lateral label against the window's own realised heading, by distance from the anchor**\n\n")
    rows = []
    for k in tr["T5b_label_vs_own_heading_by_distance_from_anchor"]:
        a = tr["T5b_label_vs_own_heading_by_distance_from_anchor"][k]
        b = ev["T5b_label_vs_own_heading_by_distance_from_anchor"][k]
        rows.append([k, n(a["in_band_windows"]), n(a["turn_windows"]), f(a["pct_label_is_TURN_x_on_turn"], 1) + " %", f(a["pct_label_is_LANE_KEEP_on_turn"], 1) + " %",
                     f(a["side_agree_pct"], 1) + " %", n(b["turn_windows"]), f(b["side_agree_pct"], 1) + " %"])
    L.append(table(["window distance from the 8.0 s anchor", "TRAIN in-band", "TRAIN turn windows", "label = TURN_x", "label = LANE_KEEP", "label side == realised side",
                    "EVAL turn windows", "EVAL side agree"], rows))
    return "\n".join(L)


if __name__ == "__main__":
    d = Path(sys.argv[1])
    tr = json.load(open(d / "numbers_train.json", encoding="utf-8"))
    ev = json.load(open(d / "numbers_eval139.json", encoding="utf-8"))
    out = render(tr, ev)
    (d / "COVERAGE_tables_generated.md").write_text(out, encoding="utf-8")
    print("wrote", d / "COVERAGE_tables_generated.md", len(out))
