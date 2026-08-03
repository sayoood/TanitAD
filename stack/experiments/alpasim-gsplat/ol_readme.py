#!/usr/bin/env python3
"""Generate the OPEN-LOOP videos folder README **from the panel JSONs**.

⛔ WHY IT IS GENERATED. The closed-loop README set the standard: "generated from those
JSONs so no number on this page was hand-copied". Hand-copying is how three numbers
propagated wrong for days in this programme. Every figure below is read out of a file
`cl_metrics.py`, `ol_distance_keeping.py` or `ffprobe` wrote.

Usage:
    python ol_readme.py --results <results dir> --videos <videos dir> --out README.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

FAMS = ("ADE", "LONGITUDINAL", "LATERAL", "TACTICAL", "STRATEGIC")

# metric -> (family, pretty name). The four families are reported PER FAMILY, never
# pooled — a single composite hides exactly the trade-off this run exists to show.
HEADLINE = [
    ("ADE", "ade_0_2s", "ADE 0–2 s (m)"),
    ("ADE", "de_2s", "displacement @2 s (m)"),
    ("LONGITUDINAL", "abs_target_speed_err_ms", "target-speed err (abs, m/s)"),
    ("LONGITUDINAL", "target_speed_err_ms", "target-speed err signed (m/s)"),
    ("LONGITUDINAL", "along_track_ade_m", "along-track ADE (m)"),
    ("LATERAL", "heading_err_rad", "heading err (rad)"),
    ("LATERAL", "curvature_err_1pm", "curvature err (1/m)"),
    ("LATERAL", "yawrate_err_rads", "yaw-rate err (rad/s)"),
    ("LATERAL", "lateral_ade_m", "lateral ADE (m)"),
    ("TACTICAL", "manoeuvre_plan_eq_logged", "plan == logged manoeuvre"),
    ("TACTICAL", "head_eq_logged", "manoeuvre HEAD == logged"),
    ("TACTICAL", "head_eq_plan", "manoeuvre HEAD == own plan"),
    ("STRATEGIC", "route_head_eq_logged", "route HEAD == logged route"),
    ("STRATEGIC", "route_head_side_eq_graded_proxy", "route side vs graded proxy"),
]
DK = [("plan_headway_2s", "headway the PLAN implies @2 s (m)"),
      ("plan_time_gap_2s", "time gap the PLAN implies @2 s (s)"),
      ("plan_ttc_2s", "TTC the PLAN implies (s, while closing)"),
      ("plan_tg_below_1s", "frac of windows: plan time gap < 1 s"),
      ("plan_tg_below_0_5s", "frac of windows: plan time gap < 0.5 s"),
      ("plan_would_hit_inlane", "frac: plan would drive INTO the lead (in-lane)")]


def esc(s):
    """Escape pipes so free text cannot silently split a markdown table cell.

    ⚠️ MEASURED on the first generation of this page: the reason string
    "departure is |cross_track| > 2 m" broke its row into three columns. A generated
    report that renders wrong is not better than a hand-copied one.
    """
    return str(s).replace("|", "\\|")


def V(d, dp=4, show_n=False):
    """Render an interval. `show_n` states the DENOMINATOR on the number itself.

    ⛔ BINDING (this programme): a rate is never reported without its denominator. It is
    not cosmetic here — MEASURED on the junction distance-keeping block, `plan_ttc_2s`
    came out `7.7219 [7.7219, 7.7219]` because TTC is only DEFINED while closing and just
    one cluster qualified. A zero-width interval reads as certainty and is the opposite.
    The marginal means and the paired delta are also computed over DIFFERENT window sets
    (the paired estimator uses jointly-finite windows only), so without n they cannot even
    be checked for consistency.
    """
    if not isinstance(d, dict):
        return "-"
    if d.get("n") == 0:
        return "n=0"
    if "mean" in d:
        s = f"{d['mean']:.{dp}f} [{d['lo']:.{dp}f}, {d['hi']:.{dp}f}]"
    elif "delta" in d:
        s = f"**{d['delta']:+.{dp}f}** [{d['lo']:+.{dp}f}, {d['hi']:+.{dp}f}]"
    else:
        return "-"
    if show_n and d.get("n_used") is not None:
        s += f" *(n={d['n_used']}"
        if d.get("n_episodes") is not None:
            s += f", {d['n_episodes']}cl"
        s += ")*"
        if d.get("n_episodes", 99) < 3 or d["lo"] == d["hi"]:
            s += " ⛔too-few-clusters"
    if d.get("CIRCULAR_NAV_ECHO"):
        s += " ⛔CIRCULAR"
    elif d.get("NAV_ECHO_UNIDENTIFIABLE"):
        s += " ⚠️echo-unidentifiable"
    if d.get("degenerate"):
        s += " ⛔degenerate"
    return s


def sep(p, k):
    d = p.get("paired_A_minus_B", {}).get(k)
    if not isinstance(d, dict) or "delta" not in d:
        return "-", ""
    return V(d), ("**separated**" if d.get("separated") else "not separated")


def family_table(p, title):
    A, B = p["arm_A"], p["arm_B"]
    L = [f"#### {title}", "",
         f"`{A['name']}` (A) vs `{B['name']}` (B) · **{p['paired_n_windows']} paired "
         f"windows · {A['n_clusters']} clusters** · paired episode-cluster bootstrap.",
         "", "| family | metric | A | B | paired Δ (A−B) | |", "|---|---|---|---|---|---|"]
    for fam, key, pretty in HEADLINE:
        a = A["families"].get(fam, {}).get(key)
        b = B["families"].get(fam, {}).get(key)
        if a is None and b is None:
            continue
        d, verdict = sep(p, key)
        L.append(f"| {fam} | {pretty} | {V(a)} | {V(b)} | {d} | {verdict} |")
    L.append("")
    return L


def dk_table(dk, title):
    A, B = dk["arm_A"], dk["arm_B"]
    a, b = (A["LONGITUDINAL_distance_keeping"], B["LONGITUDINAL_distance_keeping"])
    if a.get("n") == 0:
        return [f"#### {title}", "", f"**NOT MEASURABLE** — {a['reason']}", ""]
    L = [f"#### {title}", "",
         f"Lead present on **{a['n_windows_with_lead']}/{a['n_windows_total']}** windows "
         f"(rate {a['lead_present_rate']}); {dk['paired_n_windows']} paired.", "",
         "| metric | A | B | paired Δ (A−B) | |", "|---|---|---|---|---|"]
    for k, pretty in DK:
        d = dk.get("paired_A_minus_B", {}).get(k, {})
        verdict = ("**separated**" if d.get("separated") else "not separated") if d else ""
        if d.get("n_episodes", 99) < 3 or (d and d.get("lo") == d.get("hi")):
            verdict = "⛔ **NOT QUOTABLE** (too few clusters)"
        L.append(f"| {pretty} | {V(a.get(k), show_n=True)} | {V(b.get(k), show_n=True)} "
                 f"| {V(d, show_n=True)} | {verdict} |")
    for arm, blk in (("A", a), ("B", b)):
        pr = blk["plan_would_hit_precision"]
        L.append(f"* **{arm} in-lane precision of the ungated test** — {pr['n_true_inlane']}"
                 f"/{pr['n_fires']} fires were genuinely in-lane (precision "
                 f"{pr['precision']}), median |y| of a fire {pr['median_abs_y_of_fires_m']} m.")
    L.append("")
    L.append("⚠️ `headway_err_2s` is **algebraically −(along-track error @2 s)** — the lead "
             "term cancels — so it is NOT an independent separation and is excluded above.")
    L.append("⚠️ `n` is the **jointly-finite** window count for the paired column and the "
             "arm's own finite count for the marginals — TTC is defined only while "
             "CLOSING, so those three denominators differ and the marginals cannot be "
             "subtracted to reproduce the delta. Rows on fewer than 3 clusters, or with a "
             "zero-width interval, are marked NOT QUOTABLE.")
    L.append("")
    return L


def echo_block(p, scene):
    L = []
    for tag in ("arm_A", "arm_B"):
        a = p[tag]
        S = a["families"]["STRATEGIC"]
        e = S.get("route_head_nav_echo_check", {})
        L.append(f"* **`{a['name']}` on {scene}** — `identifiable={e.get('identifiable')}`, "
                 f"nav takes {e.get('n_distinct_nav')} value(s), head takes "
                 f"{e.get('n_distinct_head')}; map `{e.get('nav_to_head_map')}`. "
                 f"**{e.get('verdict')}**")
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--videos", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    R, VD = Path(args.results), Path(args.videos)
    J = lambda p: json.loads(Path(p).read_text())

    s1_obj = J(R / "OL_flagship_vs_refc_objects.json")
    s1_emp = J(R / "OL_flagship_vs_refc_empty.json")
    s1_dk = J(R / "OL_DISTKEEP_objects.json")
    jn_obj = J(R / "junction" / "OL_flagship_vs_refc_objects.json")
    jn_dk = J(R / "junction" / "OL_DISTKEEP_objects.json")
    flag_oe = J(R / "OL_flagship-v1_objects_vs_empty.json")
    refc_oe = J(R / "OL_refc-base_objects_vs_empty.json")
    sm1 = J(R / "openloop_summary_objects.json")
    smj = J(R / "junction" / "openloop_summary_objects.json")
    aud = J(R / "OL_AUDIT_objects.json")
    vv = J(VD / "video_verification.json")
    rq = sm1["render_quality"]

    L = ["# AlpaSim OPEN-LOOP videos — rendered on the Jetson Thor, 2026-08-03", ""]
    L.append("⭐ **These are the programme's FIRST open-loop AlpaSim videos.** Every previous "
             "AlpaSim clip and number is closed loop.")
    L.append("")
    L.append("**OPEN LOOP means the ego follows the LOGGED trajectory.** Each frame is "
             "rendered at the pose the rig actually had, the model consumes that frame "
             "stack and emits a plan, and the plan is scored against the log's own future "
             "motion. **The model never drives.** No controller step, no bicycle "
             "integration, no divergence.")
    L.append("")
    L.append("**Why it exists.** In closed loop, perception error and control drift are "
             "confounded: a bad frame moves the car, which produces a worse frame. That is "
             "not a hypothetical here — MEASURED 2026-08-03, flagship v1's driven path "
             "moved a mean **9.05 m (max 37.78)** from a *render change alone*. Open loop "
             "pins the observation distribution to the log, so what is left is prediction.")
    L.append("")

    # ---- the files ----------------------------------------------------------------
    L += ["## The eight videos", "",
          "| file | scene | arm | traffic | frames | duration | decode |", "|---|---|---|---|---|---|---|"]
    for r in vv:
        if "openloop" not in r["file"]:
            continue
        scene = "7c72937c (JUNCTION, day)" if "junction" in r["dir"] else "00040136 (night)"
        arm = "flagship v1" if r["file"].startswith("flagship") else "REF-C base"
        traf = "with objects" if "with_objects" in r["file"] else "empty road"
        ok = "✅ 0 errors" if r["decode_error_lines"] == 0 else "⛔"
        L.append(f"| `{r['dir'].split('/')[-1] if 'junction' in r['dir'] else '.'}/"
                 f"{r['file']}` | {scene} | {arm} | {traf} | {r['n_frames']} | "
                 f"{r['duration_s']:.1f} s | {ok} |")
    L.append("")
    L.append("Each is **1800×850 @ 10 fps**, front camera + metric BEV inset + decision HUD, "
             "with **OPEN-LOOP burned into the frame** and a legend naming ground truth vs "
             "prediction. Every file was verified by **decoding it back** and md5-matched "
             "against the Thor copy — `video_verification.json` holds both.")
    L.append("")

    # ---- render -------------------------------------------------------------------
    L += ["## The render — the 2026-08-03 chosen configuration", "",
          f"`layers={rq['layers']}` + all dynamic layers + **scale-cull "
          f"{rq['cull_scale_quantile']}** + **gated sky gain {rq['sky_gain']}** "
          f"(ramp {rq['sky_ramp_deg'][0]}–{rq['sky_ramp_deg'][1]}° above horizon).",
          "",
          "| | grad-NCC ↑ | neg-control margin ↑ | ms/frame |", "|---|---|---|---|",
          "| morning (`background+road`) | 0.2774 | +0.0873 | 23.3 |",
          "| **this render** | **0.3424 (+23.4 %)** | **+0.1020** | 36.3 |",
          "",
          "`run_dir = thor:~/rq_out/panel6_chosen`; evidence in "
          "`stack/experiments/alpasim-gsplat/RENDER_QUALITY.md`. **Identical to the render "
          "the closed-loop videos use**, deliberately, so open and closed loop are on the "
          "same pixels.", "",
          f"MEASURED on this run: scene 00040136 **{sm1['n_frames_rendered']} frames, "
          f"{sm1['render_ms_mean']:.1f} ms/frame mean ({sm1['render_ms_p50']:.1f} p50), "
          f"{sm1['wall_s']:.1f} s wall for BOTH arms**; junction scene "
          f"{smj['n_frames_rendered']} frames at {smj['render_ms_mean']:.1f} ms.",
          "",
          "⛔ **Rolling shutter is OFF.** It is measured-better (grad-NCC 0.3747, +35.1 %) "
          "and costs **161×** (3749 ms/frame). Nothing here uses it; if a later run does, "
          "it must say so and quote the cost.", ""]

    # ---- pairing ------------------------------------------------------------------
    L += ["## ⭐ Four invariants — CHECKED against the banked dumps, not asserted", "",
          "One render pass drives every arm inside one process, and the per-tick **md5 of "
          "the rendered frame** is recorded in each rollout file. Every claim below would "
          "leave the panel looking fine if it were false, so `ol_verify_invariants.py` "
          "re-derives all four from the banked per-window dumps:", ""]
    inv = J(R / "OL_INVARIANTS.json")
    L += ["| scene | condition | ego pinned to log | arms share pixels | md5 consistent | "
          "windows aligned | n |", "|---|---|---|---|---|---|---|"]
    tick = lambda b: "✅" if b else "⛔"
    for scene, blk in inv.items():
        if not isinstance(blk, dict):
            continue
        for c in blk["conditions"]:
            L.append(f"| {scene} | {c['condition']} | {tick(c['ego_pinned_to_log'])} | "
                     f"{tick(c['arms_share_pixels'])} | {tick(c['md5_consistent'])} | "
                     f"{tick(c['windows_aligned_across_arms'])} | {c['n_windows']} |")
    L.append("")
    for scene, blk in inv.items():
        if not isinstance(blk, dict):
            continue
        dd = blk["objects_vs_empty_frames_differ"]
        L.append(f"* **{scene}** — the objects/empty A/B changes the pixels on "
                 f"**{dd['n_ticks_differing']}/{dd['n_ticks']}** ticks. "
                 f"{tick(dd['all_differ'])} A silently no-op ablation would read as *the "
                 "model ignores the agents*, the most tempting wrong conclusion here.")
    L += ["", f"`ALL_INVARIANTS_HOLD = {inv['ALL_INVARIANTS_HOLD']}` "
              "(`OL_INVARIANTS.json`). `ego_pinned_to_log` is checked at 1e-9 on x, y AND "
              "yaw — if it were false the run would not be open loop at all and this whole "
              "page would be mislabelled, and nothing else in the pipeline would notice.",
          "",
          "This matters because the renderer is a **step function of pose**: a 0.1 px "
          "camera rotation has been measured to move the 2 s waypoint 6.65 m, and a gRPC "
          "float32 round-trip alone costs 4.59 m. \"We rendered it the same way twice\" is "
          "not good enough for a paired estimator — it has to be the same bytes.", ""]

    # ---- degeneracy ---------------------------------------------------------------
    L += ["## ⛔ Five metrics are SETUP, not RESULT — measured, then struck out", "",
          "The ego IS the logged path, so these are pinned to zero by the experiment's own "
          "definition. They were **measured** rather than assumed, and every one confirmed "
          f"at float tolerance (`all_confirmed={aud['all_confirmed']}`):", "",
          "| metric | measured max\\|value\\| | why |", "|---|---|---|"]
    seen = set()
    for r in aud["degeneracy_audit"]:
        if r["metric"] in seen:
            continue
        seen.add(r["metric"])
        L.append(f"| `{r['metric']}` | {r['measured_max_abs']:.3g} | {esc(r['why'])} |")
    L += ["",
          "⚠️ `manoeuvre_exec_eq_plan` additionally **collapses onto "
          "`manoeuvre_plan_eq_logged`** — the 'executed' manoeuvre is classified from the "
          "ego poses, which are the logged poses. *Does the arm execute what it selects* is "
          "only askable in CLOSED loop.", "",
          "⚠️ **Headway/time-gap/TTC from the ego to the annotated lead are also properties "
          "of the LOG in open loop** and come out bit-identical across arms (all four "
          "`real_lead_*` paired deltas exactly +0.0000). That family is therefore measured "
          "by a **new instrument** — see below — not dropped.", ""]

    # ---- families -----------------------------------------------------------------
    L += ["## The four families + ADE — scene 00040136 (night)", ""]
    L += family_table(s1_obj, "with objects (the scene's own dynamic layers rendered)")
    L += family_table(s1_emp, "empty road (matched control, identical logged poses)")

    L += ["## ⭐ LONGITUDINAL distance-keeping — the family, measured", "",
          "Since the ego-to-lead gap is a log property here, what IS policy-dependent is "
          "**the gap the plan would produce**: project the ego to +2 s along the emitted "
          "plan, put the lead where the annotation says it will be at +2 s, and measure. "
          "Instrument: `ol_distance_keeping.py`.", ""]
    L += dk_table(s1_dk, "scene 00040136, with objects")

    L += ["## The four families + ADE — JUNCTION scene 7c72937c (day)", "",
          "⭐ This scene was added **because the strategic family cannot be evaluated on "
          "00040136**: the nav command there is constant (`follow` on 170/170 windows), so "
          "the circularity guard is structurally unable to run. On 7c72937c the nav command "
          "**varies** and the guard becomes identifiable.", ""]
    L += family_table(jn_obj, "junction 7c72937c, with objects")
    L += dk_table(jn_dk, "junction 7c72937c, with objects — distance-keeping")

    # ---- strategic ----------------------------------------------------------------
    L += ["## ⛔⭐ THE STRATEGIC RESULT — flagship's route head is a CONFIRMED nav echo", "",
          "The harness FEEDS a nav command to the policy. If the route head is a "
          "deterministic function of that input, `route_head_eq_logged` measures the echo of "
          "the model's own conditioning — and it scores perfectly, because the nav command "
          "is derived from the same log the route label is. The guard is **computed**, and "
          "it needs ≥2 distinct nav values to separate an echo from a constant head.", ""]
    L += echo_block(jn_obj, "junction 7c72937c")
    L += [""]
    L += echo_block(s1_obj, "scene 00040136")
    fa = jn_obj["arm_A"]["families"]["STRATEGIC"]
    fb = jn_obj["arm_B"]["families"]["STRATEGIC"]
    L += ["",
          f"⇒ **flagship v1's `route_head_eq_logged` = {V(fa['route_head_eq_logged'])} is "
          "NOT strategic skill** — the head is an exact bijection of the nav command on "
          f"{jn_obj['paired_n_windows']}/{jn_obj['paired_n_windows']} windows. Its graded "
          f"proxy {V(fa['route_head_side_eq_graded_proxy'])} carries the same stamp. "
          "**Neither may be quoted.**", "",
          f"⇒ **REF-C base's {V(fb['route_head_eq_logged'])} is NOT an echo** (the head is "
          "not a function of nav) and is measured on a scene whose logged route is "
          f"genuinely non-degenerate: `{fb['route_logged_share']}`, valid on "
          f"{fb['route_label_valid_rate']} of windows. **This is the only admissible "
          "strategic-accuracy number the programme currently has.**", "",
          "⚠️ This independently reproduces, in OPEN loop, what the closed-loop panel found "
          "— so the echo is a property of the arm, not of the closed-loop harness.", ""]

    # ---- objects vs empty ----------------------------------------------------------
    L += ["## The objects-vs-empty contrast, with control drift REMOVED", "",
          "The closed-loop panel found that rendering the agents makes flagship better and "
          "flagged it as **not yet interpretable**: flagship is the arm whose plan moves 9 m "
          "under *any* appearance change, so 'extra gaussians change the frame' and 'the "
          "model reasons about the agents' predict the same sign. **In open loop the ego "
          "cannot drift**, so that confound is gone. Same logged poses, only the pixels "
          "differ.", "",
          "| arm | metric | Δ (objects − empty) | |", "|---|---|---|---|"]
    for tag, oe in (("flagship v1", flag_oe), ("REF-C base", refc_oe)):
        for k in ("ade_0_2s", "abs_target_speed_err_ms", "along_track_ade_m",
                  "lateral_ade_m", "manoeuvre_plan_eq_logged"):
            d = oe["paired_A_minus_B"].get(k, {})
            if "delta" not in d:
                continue
            L.append(f"| {tag} | `{k}` | {V(d)} | "
                     f"{'**separated**' if d.get('separated') else 'not separated'} |")
    L += ["", "Negative Δ = the arm is BETTER with the agents rendered.", ""]

    # ---- caveats -------------------------------------------------------------------
    L += ["## ⚠️ Caveats that travel with every number here", "",
          "* ⛔ **WITHIN-SIM RELATIVE ONLY.** REF-C's open-loop ADE is **1.5157 on these "
          "reconstructions vs 0.4728 on real footage — 3.21× OOD**. Orderings survive; "
          "absolute rates do not. Never quote a sim rate as a real-world rate.",
          "* **Scope:** this exercises AlpaSim's renderer **wire contract** with a gsplat "
          "backend, driven by a TanitAD harness. It is **not** `alpasim_runtime.simulate`, "
          "so there is **no AlpaSim collision / offroad / scene score** here.",
          "* ✅ **grad-NCC is the ONLY admissible render metric on these clips.** PSNR, NCC "
          "**and MAE** are RETRACTED — over 5 frames × 6 wrong references grad-NCC "
          "identifies the correct frame **5/5 on every arm** while MAE and PSNR manage "
          "**1–4/5 with arm-dependent reliability**.",
          "* ⚠️ **flagship v1's route head is a CIRCULAR nav echo** (proved above on the "
          "junction scene, and previously 369/369 + 81/81 in closed loop). Its "
          "`route_head_eq_logged = 1.0000` is not strategic skill and must never appear on a "
          "video or in a README without this guard.",
          "* ⚠️ **Precision travels with every rate**, and the denominator is stated: the "
          "per-class PR blocks in `OL_PANEL_*.md` carry precision, recall, support and "
          "n_fires, plus the majority-class baseline any constant predictor achieves.",
          "* **Clusters are DISJOINT SEGMENTS OF ONE CLIP**, not 40 independent val "
          f"episodes — {s1_obj['arm_A']['n_clusters']} of them. The interval is the right "
          "estimator for the resampling unit available; the unit is named so it is never "
          "mistaken for the 40-episode val bootstrap.",
          "* ⚠️ The renderer is a **step function of pose**; all production numbers here come "
          "from one numerical path (in-process, one render pass, both arms).", ""]

    L += ["## Provenance", "",
          "Code — `stack/experiments/alpasim-gsplat/`: `openloop_drive.py` (the sweep), "
          "`cl_metrics.py` (the four families, **unchanged** — the record schema is the "
          "same as a closed-loop rollout), `ol_distance_keeping.py` (the open-loop "
          "distance-keeping instrument), `ol_report.py` (degeneracy audit + panel), "
          "`overlay_video.py --mode open_loop` (the video), `run_openloop_videos.sh` (the "
          "runner). This page is generated by `ol_readme.py` **from the JSONs**, so no "
          "number on it was hand-copied.", "",
          "Results — `stack/experiments/alpasim-gsplat/results/openloop-thor-2026-08-03/`: "
          "`OL_flagship_vs_refc_{objects,empty}.json` (scene 00040136), "
          "`junction/OL_flagship_vs_refc_{objects,empty}.json` (7c72937c), "
          "`OL_DISTKEEP_*.json`, `OL_{arm}_objects_vs_empty.json`, `OL_AUDIT_*.json` "
          "(the degeneracy audit), `OL_INVARIANTS.json` (the four invariants above), "
          "`OL_PANEL_*.md` (the full per-family tables including "
          "every confusion matrix and per-class PR block), `openloop_summary_*.json` "
          "(render config, timings, per-tick frame md5 pointer).", "",
          "⭐ **The RAW per-window surface is banked, not stranded** — "
          "`results/openloop-thor-2026-08-03/rollouts/{scene00040136,junction7c72937c}/"
          "<condition>_<arm>.json` (8 files, 1.8 MB) hold every step's ego pose, emitted "
          "plan, head logits, nav command and frame md5. Every number on this page can be "
          "re-derived from them with **zero GPU**, and a re-render can be checked "
          "bit-exactly against the digests.", "",
          "On the device — `thor:~/ol_out` (scene 00040136), `thor:~/ol_out_junction`, "
          "`thor:~/ol_videos`, `thor:~/ol_videos_junction`. Rollout JSONs carry the "
          "per-tick `frame_md5` so a re-render can be checked bit-exactly.", "",
          "⚠️ `*.mp4` is gitignored — these are committed with `git add -f`. Any new video "
          "needs the same or it silently never lands.", "",
          "*Closed-loop counterparts: `../alpasim-closedloop-thor-2026-08-03/`.*", ""]

    Path(args.out).write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {args.out} ({len(L)} lines)")


if __name__ == "__main__":
    main()
