# -*- coding: utf-8 -*-
"""Emit the RESULT tables straight from gate_report.json + the arms' logs.

The tables are GENERATED, never transcribed: a number in the report and a
number in the artifact cannot drift apart if only one of them is typed.
"""
import json
import pathlib

W = pathlib.Path(r"C:\Users\Admin\run_refcv4")
G = json.loads((W / "gate_report.json").read_text(encoding="utf-8"))

ORDER = ["A_v3", "B_v4_noguard", "C_v4_drop", "D_v4_full", "E_regress"]
LABEL = {"A_v3": "A `v3`", "B_v4_noguard": "B `v4_noguard`",
         "C_v4_drop": "C `v4_drop`", "D_v4_full": "D `v4_full`",
         "E_regress": "E `regress` \u26d4"}
out = []


def f(x, n=4):
    return "n/a" if x is None else ("%.*f" % (n, x))


def last_metrics(arm):
    p = W / "arms" / arm / "metrics.jsonl"
    if not p.exists():
        return {}
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    return rows[-1] if rows else {}


out.append("### Training summary (2,000 steps each, identical everything but the lever)\n")
out.append("| arm | lever added | final loss | `goal2s_err_m` | `echo_ratio` | "
           "`ego_keep_frac` | `ego_injected` | wall-clock |")
out.append("|---|---|---|---|---|---|---|---|")
LEV = {"A_v3": "\u2014 (incumbent)",
       "B_v4_noguard": "`ego_state_inject`, dropout **0.0**",
       "C_v4_drop": "+ `ego_dropout 0.5`",
       "D_v4_full": "+ `echo_base` (E14)",
       "E_regress": "+ `--ablate-frames`"}
for a in ORDER:
    m = last_metrics(a)
    s = W / "arms" / a / "summary.json"
    wc = json.loads(s.read_text(encoding="utf-8")).get("wallclock_s") if s.exists() else None
    out.append("| %s | %s | %s | %s | %s | %s | %s | %s |" % (
        LABEL[a], LEV[a], f(m.get("loss"), 3), f(m.get("goal2s_err_m"), 3),
        f(m.get("echo_ratio"), 5), f(m.get("ego_keep_frac"), 3),
        f(m.get("ego_injected"), 0), ("%.0f s" % wc) if wc else "n/a"))

out.append("\n### The controls, on the shared scored windows (endpoint ADE at each goal slot)\n")
out.append("| slot | tau | n windows | `constant_only` | `ha0` | `ha` | `ha0_ext` |")
out.append("|---|---|---|---|---|---|---|")
for c in G.get("controls_by_slot", []):
    out.append("| %d | %.1f s | %d | %s | %s | %s | %s |" % (
        c["slot"], c["tau_s"], c["n"], f(c["constant_only"]), f(c["ha0"]),
        f(c["ha"]), f(c["ha0_ext"])))

out.append("\n### GATE 1 \u2014 paired episode-cluster bootstrap vs each reference\n")
out.append("| arm | slot | arm ADE | vs `ha` (rel, sep, pass) | "
           "vs `ha0_ext` (rel, sep, pass) | vs `ha0` | n win / n ep |")
out.append("|---|---|---|---|---|---|---|")
for a in ORDER:
    r = G["arms"].get(a, {})
    for row in r.get("gate1", {}).get("slots", []):
        if "vs" not in row:
            out.append("| %s | %d | \u2014 | %s | | | n=%d |" % (
                LABEL[a], row["slot"], row.get("status"), row.get("n", 0)))
            continue
        def cell(k):
            v = row["vs"][k]
            return "%s (%+.4f, %s, %s)" % (f(v["ref_mean"]),
                                           v["relative_margin"],
                                           "sep" if v["separated"] else "NOT sep",
                                           "PASS" if v["passes"] else "fail")
        out.append("| %s | %d (%.1f s) | %s | %s | %s | %s | %d / %d |" % (
            LABEL[a], row["slot"], row["tau_s"], f(row["arm_mean"]),
            cell("ha"), cell("ha0_ext"), cell("ha0"),
            row["n_windows"], row["n_episodes"]))

out.append("\n### GATE 2 (structural) / GATE 2b (functional) / the verdict\n")
out.append("| arm | GATE 2 | GATE 2b: wrong SCENE hurts | GATE 2b: wrong EGO hurts | "
           "GATE 2b verdict | gate |")
out.append("|---|---|---|---|---|---|")
for a in ORDER:
    r = G["arms"].get(a)
    if not r or r.get("status") != "MEASURED":
        out.append("| %s | %s | | | | |" % (LABEL[a], (r or {}).get("status")))
        continue
    g2, g2b, gate = r["gate2"], r["gate2b"], r["GATE"]
    sc, eg = g2b["sources"]["scene"], g2b["sources"]["ego"]
    gv = ("n/a (%s)" % gate.get("reason", "")[:40] if not gate.get("applicable", True)
          else ("**RAISED**" if gate.get("raised") else "passed"))
    out.append("| %s | %s | %+.4f (%s) | %+.4f (%s) | %s | %s |" % (
        LABEL[a], g2.get("verdict") or g2.get("status"),
        sc["degradation_rel"], "sep" if sc["separated"] else "NOT sep",
        eg["degradation_rel"], "sep" if eg["separated"] else "NOT sep",
        g2b["verdict"], gv))
    if r.get("gate2b_realframes"):
        rr = r["gate2b_realframes"]
        s2, e2 = rr["sources"]["scene"], rr["sources"]["ego"]
        out.append("| %s \u2014 **re-probed with the REAL frames** | \u2014 | %+.4f (%s) | "
                   "%+.4f (%s) | %s | |" % (
                       LABEL[a], s2["degradation_rel"],
                       "sep" if s2["separated"] else "NOT sep",
                       e2["degradation_rel"],
                       "sep" if e2["separated"] else "NOT sep", rr["verdict"]))

out.append("\n### The BINDING four families (per slot; ADE alone is an incomplete eval)\n")
out.append("| arm | slot | ADE arm/`ha0_ext` | LONG speed err arm/`ha0_ext` | "
           "LAT heading | LAT curvature | LAT yaw-rate | LAT cross-track |")
out.append("|---|---|---|---|---|---|---|---|")
for a in ORDER:
    for fam in G["arms"].get(a, {}).get("four_families", []) or []:
        lat = fam["lateral"]
        out.append("| %s | %.1f s | %s / %s | %s / %s | %s / %s | %s / %s | %s / %s | %s / %s |" % (
            LABEL[a], fam["tau_s"],
            f(fam["ade_m"]["arm"], 3), f(fam["ade_m"]["ha0_ext"], 3),
            f(fam["longitudinal"]["speed_err_mps"]["arm"], 3),
            f(fam["longitudinal"]["speed_err_mps"]["ha0_ext"], 3),
            f(lat["heading_err_rad"]["arm"]), f(lat["heading_err_rad"]["ha0_ext"]),
            f(lat["curvature_err_invm"]["arm"]), f(lat["curvature_err_invm"]["ha0_ext"]),
            f(lat["yaw_rate_err_radps"]["arm"]), f(lat["yaw_rate_err_radps"]["ha0_ext"]),
            f(lat["cross_track_m"]["arm"], 3), f(lat["cross_track_m"]["ha0_ext"], 3)))

out.append("\n\u26a0\ufe0f A goal ROW cannot carry every family, and the ones it "
           "cannot are reported with their reason and their n rather than "
           "dropped. **headway / TTC**: needs a lead agent in frame, and the "
           "`obstacle.offline` join is not built for these episodes. "
           "**TACTICAL**: needs the head logits, not a goal row \u2014 so it is "
           "computed in the next table from `lat_logits_tac` / "
           "`lon_logits_tac` against the trainer OWN label function "
           "(`refc_tactical.window_factored_labels`). **STRATEGIC**: needs "
           "the LAN corridor, which is training-only by E12 and is not "
           "emitted on these windows \u2014 absent, WITH the reason.\n")

out.append("\n### The TACTICAL family \u2014 factored decision accuracy vs the MAJORITY-CLASS control\n")
out.append("| arm | axis | n | accuracy | majority-class control | beats it? | "
           "per-class recall (support) |")
out.append("|---|---|---|---|---|---|---|")
for a in ORDER:
    t = G["arms"].get(a, {}).get("tactical_family") or {}
    for axis in ("lat", "lon"):
        v = t.get(axis)
        if not v or v.get("status"):
            out.append("| %s | %s | %s | \u2014 | | | %s |" % (
                LABEL[a], axis, (v or {}).get("n", "n/a"),
                (v or {}).get("status", "absent")))
            continue
        pc = " \u00b7 ".join(
            "c%d %s (%.2f)" % (
                c["class"],
                "n/a" if c["recall"] is None else "%.3f" % c["recall"],
                c["support_frac"])
            for c in v["per_class"])
        out.append("| %s | %s | %d | %s | %s | %s | %s |" % (
            LABEL[a], axis, v["n"], f(v["accuracy"]),
            f(v["majority_class_control"]),
            "**yes**" if v["beats_majority"] else "**no**", pc))
_t0 = None
for a in ORDER:
    _t0 = G["arms"].get(a, {}).get("tactical_family")
    if _t0:
        break
_t0 = _t0 or {}
if _t0.get("_strategic_note"):
    out.append("\n\u26a0\ufe0f **STRATEGIC family, absent WITH its reason:** "
               + _t0["_strategic_note"])
if _t0.get("_selected_vs_executed_note"):
    out.append("\n\u26a0\ufe0f **Selected-vs-executed manoeuvre, filed as a work item:** "
               + _t0["_selected_vs_executed_note"])

out.append("### S6 \u2014 manoeuvre-stratified ADE (the aggregate hides behavioural collapse)\n")
mc = G.get("manoeuvre_census") or {}
out.append("Census over the scored windows: **straight %s** / **non-straight %s** / "
           "unlabelled %s.\n" % (mc.get("straight"), mc.get("non_straight"),
                                 mc.get("unlabelled")))
out.append("| arm | subset | slot | n | arm ADE | `ha0_ext` | `ha` | `ha0` |")
out.append("|---|---|---|---|---|---|---|---|")
for a in ORDER:
    for r in G["arms"].get(a, {}).get("manoeuvre_stratified", []) or []:
        if r.get("status"):
            continue
        out.append("| %s | %s | %.1f s | %d | %s | %s | %s | %s |" % (
            LABEL[a], r["subset"], r["tau_s"], r["n"], f(r["arm_ade"], 3),
            f(r["ha0_ext_ade"], 3), f(r["ha_ade"], 3), f(r["ha0_ade"], 3)))

out.append("\n### The raw-input floor (ridge from downsampled pixels), with n and d\n")
for a in ORDER:
    rp = G["arms"].get(a, {}).get("gate1", {}).get("raw_pixel_floor")
    if not rp:
        continue
    m = rp["meta"]
    out.append("* **%s** \u2014 n_fit %d, n_score %d, **d %d**, lambda %s (%s), "
               "underpowered=%s" % (LABEL[a], m["n_fit"], m["n_score"], m["d"],
                                    m["lambda"], m["lambda_selected_on"],
                                    m["underpowered_n_lt_d"]))
    for r in rp["per_slot"]:
        out.append("  * slot %.1f s (n=%d): floor **%s** vs arm **%s**" % (
            r["tau_s"], r["n"], f(r["floor_ade"], 3), f(r["arm_ade"], 3)))

(W / "RESULT_TABLES.md").write_text("\n".join(out) + "\n", encoding="utf-8")
print("\n".join(out))
