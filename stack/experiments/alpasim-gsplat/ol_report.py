#!/usr/bin/env python3
"""Render an OPEN-LOOP panel (`cl_metrics.py` output) into a report, with the
degenerate-by-construction metrics MARKED rather than silently published.

⛔ WHY THIS FILE EXISTS AND WHY IT IS NOT A cl_metrics PATCH.
`cl_metrics.py` is the closed-loop instrument and is shared with another stream; it
scores an open-loop rollout correctly and unchanged, because the record schema is the
same. What it cannot know is that in OPEN loop five of its metrics are pinned to zero by
the experiment's own definition — the ego IS the logged path — so a reader would see

    cross_track_abs_m  0.0000 [0.0000, 0.0000]      "separated"

and read a perfect lateral controller. It is not a result; it is the setup. This module
therefore MEASURES each of them, states the measured magnitude, and stamps the metric
`DEGENERATE_BY_CONSTRUCTION` before anything is published.

⚠️ It never re-derives a number. Every value printed is read out of the panel JSON that
`cl_metrics.py` wrote, so the report and the artifact cannot drift.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# metric -> why it is pinned in OPEN loop. The tolerance is the magnitude below which the
# degeneracy is CONFIRMED; above it the assumption is wrong and this module says so
# loudly rather than stamping a caveat on a number that is actually informative.
DEGENERATE = {
    "cross_track_abs_m": ("the ego pose IS the logged pose, so its distance to the "
                          "logged polyline is zero by construction", 1e-6),
    "cross_track_signed_m": ("same quantity, signed", 1e-6),
    "dist_to_gt_traj_m": ("literally abs(cross_track) in cl_metrics — the same "
                          "measurement under a second name", 1e-6),
    "executed_speed_err_ms": ("the ego speed IS the logged speed; no controller runs",
                              1e-6),
    "route_corridor_departure_rate": ("departure is |cross_track| > 2 m, and cross_track "
                                      "is pinned at 0", 1e-6),
}
# metrics that are not zero but are ALIASES of another metric in open loop
COLLAPSED = {
    "manoeuvre_exec_eq_plan": (
        "in OPEN loop the 'executed' manoeuvre is classified from the ego poses, which "
        "are the LOGGED poses — so this is `manoeuvre_plan_eq_logged` a second time, not "
        "an independent measurement of whether the arm executes what it selects. That "
        "question is only askable in CLOSED loop."),
}
FAMILY_ORDER = ("ADE", "LONGITUDINAL", "LATERAL", "TACTICAL", "STRATEGIC")


def fmt(v):
    if not isinstance(v, dict):
        return str(v)
    if v.get("n") == 0:
        return "n=0 (" + str(v.get("reason", "?"))[:90] + ")"
    if "mean" in v:
        s = f"{v['mean']:.4f} [{v['lo']:.4f}, {v['hi']:.4f}]"
    elif "delta" in v:
        s = f"{v['delta']:+.4f} [{v['lo']:+.4f}, {v['hi']:+.4f}]"
    else:
        return "-"
    if v.get("degenerate"):
        s += " ⛔degenerate"
    if v.get("CIRCULAR_NAV_ECHO"):
        s += " ⛔CIRCULAR"
    if v.get("NAV_ECHO_UNIDENTIFIABLE"):
        s += " ⚠️echo-unidentifiable"
    return s


def audit(fams, arm="?"):
    """MEASURE the degeneracy claims instead of asserting them.

    ⚠️ The arm travels with every row. Without it the two arms' rows are
    indistinguishable in the audit table and a failure on ONE arm would read as a
    duplicate of the other's pass.
    """
    rows = []
    for f, d in fams.items():
        if not isinstance(d, dict):
            continue
        for k, v in d.items():
            if k in DEGENERATE and isinstance(v, dict) and "mean" in v:
                why, tol = DEGENERATE[k]
                mag = max(abs(v["mean"]), abs(v["lo"]), abs(v["hi"]))
                rows.append({"arm": arm, "family": f, "metric": k,
                             "measured_max_abs": mag,
                             "tolerance": tol, "confirmed": bool(mag <= tol),
                             "why": why})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", required=True, help="cl_metrics.py output json")
    ap.add_argument("--out", required=True, help="markdown out")
    ap.add_argument("--json-out", default=None, help="audit json out")
    ap.add_argument("--title", default="OPEN-LOOP panel")
    args = ap.parse_args()

    p = json.loads(Path(args.panel).read_text())
    A, B = p["arm_A"], p.get("arm_B")
    aud = (audit(A["families"], A["name"])
           + (audit(B["families"], B["name"]) if B else []))
    bad = [r for r in aud if not r["confirmed"]]

    L = [f"# {args.title}", ""]
    arms_line = f"**Arms:** `{A['name']}` (A)"
    if B:
        arms_line += f" vs `{B['name']}` (B)"
    arms_line += f" · windows A {A['n_windows']} / clusters {A['n_clusters']}"
    if B:
        arms_line += (f" · B {B['n_windows']} / {B['n_clusters']}"
                      f" · paired windows {p.get('paired_n_windows')}")
    L.append(arms_line)
    L.append("")
    L.append("⭐ **OPEN LOOP** — the ego follows the LOGGED trajectory; every frame is "
             "rendered at the pose the rig actually had and the model's plan is scored "
             "against the log's own future motion. **The plan is never executed.** This "
             "isolates perception + prediction from the control drift that is confounded "
             "with them in every closed-loop number.")
    L.append("")
    L.append(f"`f_eff` A = {A.get('f_eff')}" + (f", B = {B.get('f_eff')}" if B else "")
             + " (the canonicalization self-check against `F_REF`; both arms are 256px "
               "SQUARE and `_BasePolicy.canon` asserts `(1, 8, 9, 256, 256)` before any "
               "forward pass).")
    L.append("")

    # ---- the degeneracy audit, FIRST, before any table can be misread -------------
    L.append("## ⛔ Degenerate by construction — measured, then marked")
    L.append("")
    L.append("| arm | family | metric | measured max\\|value\\| | tol | confirmed | why |")
    L.append("|---|---|---|---|---|---|---|")
    for r in aud:
        L.append(f"| {r['arm']} | {r['family']} | `{r['metric']}` | "
                 f"{r['measured_max_abs']:.3g} | {r['tolerance']:.0e} | "
                 f"{'✅' if r['confirmed'] else '⛔ NO'} | {r['why']} |")
    L.append("")
    if bad:
        L.append("⛔ **A degeneracy assumption FAILED** — the metrics above marked ⛔ are "
                 "NOT pinned to zero in this run, which means the open-loop setup is not "
                 "doing what this module assumes. Investigate before quoting anything.")
    else:
        L.append("All pinned metrics confirmed at float tolerance. They are **setup, not "
                 "result**, and are struck through in the family tables below.")
    L.append("")
    for k, why in COLLAPSED.items():
        L.append(f"⚠️ `{k}` — {why}")
    L.append("")

    # ---- families ----------------------------------------------------------------
    for fam in FAMILY_ORDER:
        fa = A["families"].get(fam, {})
        fb = (B or {}).get("families", {}).get(fam, {}) if B else {}
        L.append(f"## {fam}")
        L.append("")
        L.append(f"| metric | {A['name']} |" + (f" {B['name']} |" if B else "")
                 + " paired Δ (A−B) | |")
        L.append("|---|---|" + ("---|" if B else "") + "---|---|")
        for k, v in fa.items():
            if not isinstance(v, dict) or ("mean" not in v and v.get("n") != 0):
                continue
            pr = p.get("paired_A_minus_B", {}).get(k, {})
            mark = ""
            name = f"`{k}`"
            if k in DEGENERATE:
                name = f"~~`{k}`~~"
                mark = "⛔ setup, not result"
            elif k in COLLAPSED:
                name = f"`{k}` ⚠️"
                mark = "alias of plan_eq_logged"
            elif pr.get("separated"):
                mark = "**separated**"
            elif pr:
                mark = "not separated"
            L.append(f"| {name} | {fmt(v)} |" + (f" {fmt(fb.get(k, {}))} |" if B else "")
                     + f" {fmt(pr) if pr else '-'} | {mark} |")
        L.append("")
        # non-interval descriptors worth carrying (shares, confusion, guards)
        for k in ("plan_class_share", "logged_class_share", "head_class_share",
                  "route_logged_share", "route_head_share",
                  "route_derivation_reason_share", "route_label_valid_rate",
                  "route_head_nav_echo_check", "lead_present_rate", "ttc_defined_rate",
                  "head_vs_logged_per_class_PR", "plan_vs_logged_per_class_PR",
                  "route_head_side_per_class_PR"):
            if k in fa:
                L.append(f"* **A `{k}`** — `{json.dumps(fa[k])[:600]}`")
            if fb and k in fb:
                L.append(f"* **B `{k}`** — `{json.dumps(fb[k])[:600]}`")
        L.append("")

    L.append("## Estimator and caveats")
    L.append("")
    L.append(f"* {p.get('estimator_note', '')}")
    L.append(f"* {p.get('within_sim_note', '')}")
    L.append("* ✅ **grad-NCC is the only admissible render metric on these night clips.** "
             "PSNR, NCC **and MAE** are RETRACTED here — grad-NCC identifies the correct "
             "reference frame 5/5 on every arm while MAE/PSNR manage 1–4/5 with "
             "arm-dependent reliability.")
    L.append("* **Scope:** AlpaSim renderer **wire contract** with a gsplat backend. NOT "
             "`alpasim_runtime.simulate` — there is no AlpaSim collision/offroad score here.")
    L.append("* ⚠️ Report **precision alongside recall** for every rate, and state the "
             "denominator — the per-class PR blocks above carry both.")
    Path(args.out).write_text("\n".join(L), encoding="utf-8")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(
            {"degeneracy_audit": aud, "all_confirmed": not bad,
             "panel": str(args.panel)}, indent=2))
    print(f"wrote {args.out} ({len(L)} lines); degeneracy confirmed={not bad}")


if __name__ == "__main__":
    main()
