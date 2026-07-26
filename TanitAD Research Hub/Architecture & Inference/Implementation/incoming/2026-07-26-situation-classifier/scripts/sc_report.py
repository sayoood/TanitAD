"""Situation classifier — render every results table from the JSON. No number is hand-typed.

usage:  python sc_report.py <artifacts_dir> <out_tables.md> [<doc.md> ...]
        (with a doc argument, the tables are spliced at the <!-- TABLES:X --> markers)
"""
from __future__ import annotations

import json
import os
import re
import sys

SITS = ("lane_change", "roundabout", "intersection")
PRETTY = {"lane_change": "LANE CHANGE", "roundabout": "ROUNDABOUT",
          "intersection": "INTERSECTION"}


def ci(d, k="point", n=5):
    if d is None:
        return "—"
    lo, hi = d.get("lo"), d.get("hi")
    if lo is None:
        return f"{d.get(k, float('nan')):.{n}f}"
    return f"{d.get(k, d.get('delta', float('nan'))):.{n}f} [{lo:.{n}f}, {hi:.{n}f}]"


def sep(d):
    if d is None:
        return "—"
    return "**YES**" if d.get("separated") else "no"


def main():
    art, out = sys.argv[1], sys.argv[2]
    R = json.load(open(os.path.join(art, "sc_results.json")))
    U = json.load(open(os.path.join(art, "universe.json")))
    V = json.load(open(os.path.join(art, "label_validation.json")))
    X = json.load(open(os.path.join(art, "cross_summary.json")))
    blocks = {}

    # ---------------------------------------------------------------- SUBSTRATE
    L = ["### The universe and the label, per situation\n",
         "| situation | TRAIN events / pos-frames / pos-clips | HELD-OUT events / pos-frames / "
         "pos-clips | held-out base rate | **C-POW** |", "|---|---|---|---|---|"]
    for s in SITS:
        u = U["per_situation"][s]
        t, h = u["TRAIN"], u["HELDOUT"]
        L.append(f"| **{PRETTY[s]}** | {t['events']} / {t['pos_frames']:,} / {t['pos_clips']} "
                 f"| {h['events']} / {h['pos_frames']:,} / {h['pos_clips']} | {h['base_rate']:.5f} "
                 f"| {'✅ OK' if h['pos_clips'] >= 40 else '⛔ **UNDERPOWERED**'} "
                 f"({h['pos_clips']} clusters) |")
    L.append("")
    L.append(f"*Universe: {U['n_clips']:,} cached parity episodes, {U['n_chunks']} chunks "
             f"({U['n_chunks_train']} TRAIN / {U['n_chunks']-U['n_chunks_train']} HELD-OUT); "
             f"TRAIN {U['per_side']['TRAIN']['clips']:,} clips / "
             f"{U['per_side']['TRAIN']['frames']:,} frames, HELD-OUT "
             f"{U['per_side']['HELDOUT']['clips']:,} clips / "
             f"{U['per_side']['HELDOUT']['frames']:,} frames. "
             f"Anticipation lead {U['lead_s']} s; minimum useful lead "
             f"{U['min_useful_lead_s']} s (registered before measurement).*")
    blocks["SUBSTRATE"] = "\n".join(L)

    # ---------------------------------------------------------------- VALIDATION
    v1 = V["V1_lane_change_heading_gate"]
    L = ["### V1 — the lane-change heading gate, swept (the ×89-collapse test)\n",
         "| net-heading gate | events (with S-shape) | clip rate | median lateral | "
         "events (NO S-shape) | clip rate | median lateral |", "|---|---|---|---|---|---|---|"]
    for g, a in v1["curve"]["with_S_shape"].items():
        b = v1["curve"]["without_S_shape"][g]
        mark = " ⭐" if float(g) == v1["operating_gate_deg"] else ""
        L.append(f"| {g}°{mark} | {a['events']} | {a['clip_rate']:.5f} | {a['median_lateral_m']} m "
                 f"| {b['events']} | {b['clip_rate']:.5f} | {b['median_lateral_m']} m |")
    L += ["",
          f"**Collapse ratio 10° → 1°: {v1['collapse_ratio_10_to_1_with_S_shape']}× with the "
          f"S-shape clause, {v1['collapse_ratio_10_to_1_without_S_shape']}× without.** The S-shape "
          f"clause rejects **{v1['S_shape_rejection_at_operating_gate']*100:.1f} %** of candidates "
          f"at the operating gate. Lateral offset at the operating gate: median "
          f"{v1['lateral_offset_at_operating_gate_m']['median_m']} m "
          f"[p10 {v1['lateral_offset_at_operating_gate_m']['p10_m']}, "
          f"p90 {v1['lateral_offset_at_operating_gate_m']['p90_m']}].", ""]

    v2 = V["V2_roundabout"]
    L += ["### V2 — the roundabout direction test (the corpus has **0** left-hand-traffic clips)\n",
          "| variant | TRAIN (in-sample: the selection target) | HELD-OUT (**out-of-sample**) |",
          "|---|---|---|"]
    for nm, key in (("`ROUND` (pre-registered)", "ccw_purity"), ("`ROUND_core`", "ccw_purity_core")):
        a, b = v2[key]["TRAIN"], v2[key]["HELDOUT"]
        L.append(f"| {nm} | {a['ccw_frac']} (n={a['n']}) | **{b['ccw_frac']}** (n={b['n']}) |")
    L += ["", f"Maximum same-sign sweep anywhere in this universe: "
              f"**{v2['max_same_sign_sweep_deg_any_radius']}°** "
              f"({v2['max_same_sign_sweep_deg_radius_le_50m']}° at radius ≤ 50 m).", ""]

    v3 = V["V3_turn_direction_balance"]
    L += ["### V3 — turn left/right balance (a junction population must be ~50/50)\n",
          "| side | n | left fraction |", "|---|---|---|"]
    for k, d in v3.items():
        L.append(f"| {k} | {d['n']} | {d['left_frac']} |")
    L.append("")

    v4 = V["V4_turn_is_junction"]
    L += ["### ⭐ V4 — is the TURN half a junction detector, or just a curve detector?\n",
          "*Pre-registered in `PRE_REGISTRATION.md` §6.2 with both outcomes committed. "
          "If the ratio's CI included 1.0, the turn half would NOT be a junction detector and the "
          "intersection label would fall back to `CROSS`-only.*\n",
          "| quantity | value |", "|---|---|",
          f"| P(perpendicular cross traffic \\| tight TURN, R ≤ 25 m) | **{v4['P_cross_given_TURN']}** "
          f"({v4['n_turn_frames']:,} frames) |",
          f"| P(… \\| matched-heading LARGE-RADIUS curve, R > 40 m) | {v4['P_cross_given_LARGE_RADIUS_CURVE']} "
          f"({v4['n_curve_frames']:,} frames) |",
          f"| **ratio** | **{v4['ratio']}×** {v4['ci95']} |",
          f"| separated from 1.0? | {'✅ **YES**' if v4['separated_from_1'] else '❌ no'} |", "",
          f"*{v4['estimator']}.*", ""]

    L += ["### C-ALIGN — the obstacle clock map, in METRES\n", "| quantity | value |", "|---|---|",
          f"| clips attempted / admitted | {X['n_clips']} / {X['admitted']} |",
          f"| median position residual | **{X['C_ALIGN']['quantiles_m']['0.5']} m** |",
          f"| p90 / p99 residual | {X['C_ALIGN']['quantiles_m']['0.9']} m / "
          f"{X['C_ALIGN']['quantiles_m']['0.99']} m |",
          f"| failure fraction (> {X['C_ALIGN']['admission_floor_m']} m) | "
          f"{X['C_ALIGN']['fail_frac']*100:.2f} % |",
          f"| **obstacle-window overlap (median)** | "
          f"**{X['OBSTACLE_CLOCK_JOIN_PROOF']['overlap_quantiles']['0.5']}** |",
          f"| cross-traffic frame rate (path-crossing) | {X['cross_frame_rate']} |",
          f"| perpendicular-agent-present frame rate | {X['perp_present_frame_rate']} |", ""]
    blocks["VALIDATION"] = "\n".join(L)

    # ---------------------------------------------------------------- RESULTS
    L = []
    for s in SITS:
        r = R["situations"][s]
        L += [f"### {PRETTY[s]} — held-out discrimination\n",
              f"*{r['n_windows']:,} scored windows · {r['n_pos']:,} positives · "
              f"{r['n_pos_clusters']} positive clip clusters of {r['n_clusters']} · "
              f"base rate {r['base_rate']:.5f} · **C-POW {r['C_POW']}**.*\n",
              "| arm | AP (episode-cluster bootstrap CI95) | AP / base | AUROC | ΔAP vs CHANCE | "
              "above chance? |", "|---|---|---|---|---|---|"]
        order = [k for k in ("head_img_ego", "ridge_img_ego", "head_img", "ridge_img", "head_ego",
                             "ridge_ego", "head_img_ego_concat", "heur_kin", "head_priv",
                             "head_img_shuf", "ridge_img_shuf") if k in r["AP"]]
        for k in order:
            d = r["AP"][k]
            ac = r["above_chance"].get(k)
            star = " ⭐" if k == "head_img_ego" else (" ⭐C-POS" if k == "head_priv" else
                                                     (" ⭐C-NEG" if "shuf" in k else ""))
            L.append(f"| `{k}`{star} | {ci(d)} | {d['ap_over_base']}× | {d['auroc']} | "
                     f"{ci(ac, 'delta')} | {sep(ac)} |")
        L.append("")
        L += ["| contrast | Δ | CI95 | separated? |", "|---|---|---|---|"]
        for k, d in r["vs_head_ego"].items():
            if k in ("head_img_shuf", "ridge_img_shuf", "head_priv"):
                continue
            L.append(f"| `{k}` − `head_ego` | {d['delta']:+.5f} | [{d['lo']:+.5f}, {d['hi']:+.5f}] "
                     f"| {sep(d)} |")
        for k, d in r.get("paired_recall_delta", {}).items():
            L.append(f"| recall: {k} | {d['delta']:+.4f} | [{d['lo']:+.4f}, {d['hi']:+.4f}] "
                     f"| {sep(d)} |")
        L.append("")

        L += [f"#### {PRETTY[s]} — operating point at B\\* = {R['B_star']} extra cams/frame\n",
              "| arm | firing rate | recall | precision | precision lift | caught / total | θ\\* |",
              "|---|---|---|---|---|---|---|"]
        for k in order:
            o = r["operating_point"].get(k)
            if not o:
                continue
            L.append(f"| `{k}` | {ci(o['firing_rate'],'point',4)} | {ci(o['recall'],'point',4)} | "
                     f"{ci(o['precision'],'point',5)} | {o['precision_lift']:.2f}× | "
                     f"{o['n_caught']}/{r['n_pos']} | {o['theta']:.4f} |")
        b = r["baselines"]
        L += [f"| **(a) always-escalate** | 1.0 | 1.0 | {b['always_escalate']['precision']:.5f} "
              f"| 1.00× | {r['n_pos']}/{r['n_pos']} | — |",
              "| **(b) never-escalate** | 0.0 | **0.0** | — | — | 0/" + str(r['n_pos']) + " | — |",
              f"| **(c) random @ matched rate** | {b['random_at_matched_rate']['matched_rate']:.4f} "
              f"| {b['random_at_matched_rate']['recall_mean']:.4f} "
              f"[{b['random_at_matched_rate']['recall_p2.5']:.4f}, "
              f"{b['random_at_matched_rate']['recall_p97.5']:.4f}] "
              f"| {b['random_at_matched_rate']['precision_mean']:.5f} | 1.00× | — | 200 seeds |",
              f"| **oracle** | {b['oracle']['firing_rate']:.5f} | 1.0 | 1.0 | "
              f"{1/max(r['base_rate'],1e-9):.0f}× | {r['n_pos']}/{r['n_pos']} | — |", ""]

        L += [f"#### ⭐ {PRETTY[s]} — LEAD TIME (registered minimum "
              f"**{R['min_useful_lead_s']} s**; a high-AP zero-lead trigger FAILS)\n",
              "| arm | events detected | event recall | **median lead** | p25 / p75 | "
              "frac of events at ≥ min lead | PASS? |", "|---|---|---|---|---|---|---|"]
        for k in order:
            lt = r["lead_time"].get(k)
            if not lt:
                continue
            q = lt["lead_quantiles_s"]
            L.append(f"| `{k}` | {lt['n_detected']}/{lt['n_events']} | {lt['event_recall']} | "
                     f"**{lt['median_lead_s']} s** | {q['0.25']} / {q['0.75']} s | "
                     f"{lt['frac_events_at_or_above_min_lead']} | "
                     f"{'✅' if lt['PASS_min_lead'] else '❌'} |")
        L.append("")

        # ---- the controls, each with its MDE ----
        c = r["controls"]
        cb = r.get("C_BLIND", {})
        mde = cb.get("MDE_AUDIT", {})
        L += [f"#### {PRETTY[s]} — the controls, each with its MDE\n",
              "| control | what it catches | result | can it fail? |", "|---|---|---|---|",
              f"| **C-POS** `head_priv` | instrument insensitivity | ΔAP vs chance "
              f"{ci(c['C_POS_head_priv'], 'delta')} → {sep(c['C_POS_head_priv'])} | it is a probe on "
              f"the label's own defining quantity; a null here would mean UNPOWERED |",
              f"| **C-NEG** `head_img_shuf` | a pipeline leak | {ci(c['C_NEG_head_img_shuf'],'delta')} "
              f"→ {sep(c['C_NEG_head_img_shuf'])} | features permuted ACROSS clips — it must NOT "
              f"separate |",
              f"| **C-NEG** `ridge_img_shuf` | a pipeline leak (closed form) | "
              f"{ci(c['C_NEG_ridge_img_shuf'],'delta')} → {sep(c['C_NEG_ridge_img_shuf'])} | as above |",
              f"| **MDE** (upper 95 % bound of the C-NEG ΔAP) | the smallest effect this run can "
              f"distinguish from nothing | **{c['MDE_from_C_NEG']:.5f}** | — |",
              f"| **C-POW** | reading a small-n null as a refutation | {r['n_pos_clusters']} "
              f"positive clusters vs a 40 bar → **{r['C_POW']}** | measured before any score |",
              f"| **C-BLIND** (packaged firewall, imported) | the target being a function of the "
              f"conditioning | verdict **{cb.get('verdict','—')}**, but "
              f"`blind_skill_over_majority` = **{mde.get('blind_skill_over_majority','—')}** | "
              f"⛔ **NO** — positive rate {mde.get('positive_rate','—')} < deterministic_eps "
              f"{mde.get('deterministic_eps','—')}, so the majority-class predictor alone forces "
              f"`CIRCULAR`. Max possible accuracy gain from ANY context: "
              f"{mde.get('max_possible_accuracy_gain_over_majority','—')}. |", "",
              "> ⚠️ **The C-BLIND verdict is degenerate on this target and must not be quoted "
              "alone.** Its own companion numbers refute it: the ego context adds **exactly zero** "
              "skill over the base rate. The informative form of the same question — *does vision "
              "buy anything the ego state did not already give?* — is the AP-based "
              "`− head_ego` contrast above, which is the pre-registered primary comparison.", ""]

        L += [f"#### {PRETTY[s]} — the efficiency curve (recall at a fixed camera budget)\n",
              "| budget B (extra cams/frame) | realised firing rate | **head recall** | "
              "head precision | ego-head recall | kinematic-rule recall | saving vs always-on-7 |",
              "|---|---|---|---|---|---|---|"]
        for i, b_ in enumerate(R["efficiency_budgets"] if "efficiency_budgets" in R
                               else [c["budget"] for c in r["efficiency_curve"]["head_img_ego"]]):
            hh = r["efficiency_curve"]["head_img_ego"][i]
            eg = r["efficiency_curve"]["head_ego"][i]
            hk = r["efficiency_curve"]["heur_kin"][i]
            cams = 1.0 + hh["extra_cams_per_frame"]
            L.append(f"| {b_} | {hh['firing_rate']['point']:.4f} | **{hh['recall']['point']:.4f}** "
                     f"| {hh['precision']['point']:.5f} | {eg['recall']['point']:.4f} | "
                     f"{hk['recall']['point']:.4f} | {(1-cams/7.0)*100:.1f} % |")
        L.append("")
        # --- the efficiency LEDGER: BOOST_PROGRAM 7.3, saving beside recall, never instead of it
        op = r["operating_point"]["head_img_ego"]
        b = r["baselines"]
        oc = 1.0 + 2.0 * r["base_rate"]
        L += [f"#### {PRETTY[s]} — the efficiency ledger (⚠️ read the RECALL column, not the saving)\n",
              "| policy | extra cams/frame | cams/frame | saving vs always-on-7 | **recall** |",
              "|---|---|---|---|---|",
              "| **never escalate** (free and useless) | 0 | 1.000 | 85.7 % | **0.000** |",
              f"| **ORACLE** — fires exactly on the label | {2*r['base_rate']:.4f} | {oc:.4f} | "
              f"{(1-oc/7.0)*100:.1f} % | **1.000** |",
              f"| **`head_img_ego` @ B\\*** | {op['extra_cams_per_frame']:.4f} | "
              f"{1+op['extra_cams_per_frame']:.4f} | "
              f"{(1-(1+op['extra_cams_per_frame'])/7.0)*100:.1f} % | "
              f"**{op['recall']['point']:.4f}** [{op['recall']['lo']:.4f}, {op['recall']['hi']:.4f}] |",
              "| **always escalate** | 6.000 | 7.000 | 0.0 % | **1.000** |", "",
              f"> The span of *saving* between a useless gate and a perfect oracle is "
              f"**{abs((1-oc/7.0)-(1-1/7.0))*100:.2f} percentage points** — so **no compute-saving "
              f"number here can distinguish a good gate from a useless one** (BOOST_PROGRAM §7.3). "
              f"The axis that carries information is **recall at the budget**, and the lead time.", ""]
    blocks["RESULTS"] = "\n".join(L)

    # ---------------------------------------------------------------- CAMERA NEED
    L = []
    if "camera_need" in R:
        L += ["### The multi-camera need — MEASURED, per situation\n",
              "*An agent that projects into camera X but **not** into the canonical 51.4° encoder "
              "crop (H2's `T_off` machinery, per-clip `(cx,cy)` + per-clip extrinsics). "
              "Reported against the matched NOT-in-situation baseline on the same clips, because a "
              "camera that always sees something extra proves nothing.*\n",
              "| situation | camera | in situation | not in situation | **lift [CI95]** | "
              "separated? |", "|---|---|---|---|---|---|"]
        for s in SITS:
            cn = R["camera_need"].get(s, {})
            for c, d in sorted(cn.get("per_camera", {}).items(),
                               key=lambda kv: -(kv[1].get("lift") or 0)):
                nm = c.replace("camera_", "").replace("_120fov", "").replace("_70fov", "") \
                      .replace("_30fov", "")
                cc = d.get("lift_ci95")
                L.append(f"| {PRETTY[s]} | `{nm}` | {d['in_situation']} | "
                         f"{d['not_in_situation']} | **{d['lift']}×** "
                         f"{cc if cc else ''} | "
                         f"{'✅ **YES**' if d.get('separated_from_1') else 'no'} |")
        L += ["", "*Estimator: paired episode-cluster bootstrap over the clips carrying each "
                  "situation, B = 400. ⚠️ **Read the LIFT, not the raw rate.** The not-in-situation "
                  "rates are 0.4–0.9, i.e. an agent outside the front crop is visible in some other "
                  "camera almost all the time — a raw 'need' rate is close to information-free "
                  "(BOOST_PROGRAM §7.3). Only the lift over the matched baseline carries evidence.*",
              ""]
    blocks["CAMERA"] = "\n".join(L)

    # ---------------------------------------------------------------- VERDICT (computed, not typed)
    L = ["| situation | C-POW | image arm above chance? | vision over ego? | median lead | "
         "**PRE-REGISTERED VERDICT** |", "|---|---|---|---|---|---|"]
    verdicts = {}
    for s in SITS:
        r = R["situations"][s]
        pos = r["controls"]["C_POS_head_priv"]
        neg_ok = not any(r["above_chance"].get(k, {}).get("separated") and
                         r["above_chance"][k]["delta"] > 0
                         for k in ("head_img_shuf", "ridge_img_shuf") if k in r["above_chance"])
        img_arms = [k for k in ("head_img_ego", "ridge_img_ego", "head_img", "ridge_img")
                    if k in r["above_chance"]]
        above = [k for k in img_arms
                 if r["above_chance"][k]["separated"] and r["above_chance"][k]["delta"] > 0]
        over_ego = [k for k in img_arms
                    if r["vs_head_ego"].get(k, {}).get("separated")
                    and r["vs_head_ego"][k]["delta"] > 0]
        lead = r["lead_time"].get("head_img_ego", {}).get("median_lead_s")
        lead_ok = bool(lead is not None and lead >= R["min_useful_lead_s"])
        pos_ok = bool(pos and pos.get("separated") and pos["delta"] > 0)
        if r["C_POW"] != "OK" or not pos_ok:
            v = "**UNPOWERED**"
        elif above and over_ego and lead_ok and neg_ok:
            v = "**A — the classifier works AND vision contributes**"
        elif above and lead_ok and neg_ok:
            v = "**A− — predictable, but not *from the camera* beyond ego state**"
        elif not above and pos_ok:
            v = "**B — the frozen v1 front-camera state does not expose it**"
        else:
            v = "**A− (lead-time or control condition unmet — see the row)**"
        verdicts[s] = v
        L.append(f"| **{PRETTY[s]}** | {r['C_POW']} ({r['n_pos_clusters']} clusters) | "
                 f"{', '.join('`'+k+'`' for k in above) if above else '**none**'} | "
                 f"{', '.join('`'+k+'`' for k in over_ego) if over_ego else '**none**'} | "
                 f"{lead} s {'✅' if lead_ok else '❌'} | {v} |")
    L += ["", f"*Evaluated in code by the rule fixed in `PRE_REGISTRATION.md` §7 — "
              f"C-POS must separate, C-NEG must not, the median lead time must reach "
              f"{R['min_useful_lead_s']} s, and a situation with fewer than 40 held-out positive "
              f"clusters is `UNPOWERED` and gets no verdict at all.*"]
    blocks["VERDICT"] = "\n".join(L)

    # ------------------------------------------------- CV / the rank ladder (training side only)
    T = json.load(open(os.path.join(art, "train_summary.json")))
    L = ["### ⭐ The rank ladder, replicated on THESE targets (training-side CV, out-of-fold)\n",
         "*A sibling stream MEASURED a monotone swamping dose-response on this same frozen v1 state "
         "(ego 3.659× → +k16 3.685× → +k64 3.000× → +k256 2.116× → +k2048 1.59×; INHERITED). "
         "This is the independent replication on the PI's three situations — **CV-AP is out-of-fold "
         "on TRAIN, grouped by chunk, and is NOT a result** (only held-out output is quotable); it "
         "is shown because it is where the ordering first appears, before the held-out side was "
         "touched.*\n",
         "| arm \\| config | mean CV-AP | selected? |", "|---|---|---|"]
    sel = T["selected"]
    for key in sorted(T["cv"]):
        arm = key.split("|")[0]
        vals = T["cv"][key]
        if isinstance(next(iter(vals.values())), list):
            m = sum(max(v) for v in vals.values()) / len(vals)
        else:
            m = sum(vals.values()) / len(vals)
        s = sel.get(arm, {})
        chosen = (f"pw{s.get('cfg',{}).get('pw')}|d{s.get('cfg',{}).get('d')}|"
                  f"r{s.get('cfg',{}).get('r')}" if "cfg" in s else f"lam{s.get('lam')}")
        mark = "⭐" if key.split("|", 1)[1].startswith(chosen) or (
            "lam" in chosen and f"lam{s.get('lam'):g}" in key) else ""
        L.append(f"| `{key}` | {m:.5f} | {mark} |")
    L += ["", "**Every one of the ten arms selected rank 16 over rank 64.** The raw-2048 "
              "concatenation arm (`head_img_ego_concat`) is the far end of the same ladder and it "
              "is where the degradation is largest — see the held-out tables in §5.", ""]
    L += ["| fold | chunks |", "|---|---|"]
    for f, ch in sorted(T["folds"].items()):
        L.append(f"| {f} | {len(ch)} chunks |")
    L.append("")
    blocks["CV"] = "\n".join(L)

    open(out, "w", encoding="utf-8", newline="\n").write(
        "\n\n".join(f"<!-- TABLES:{k} -->\n{v}\n<!-- /TABLES:{k} -->" for k, v in blocks.items()))
    for doc in sys.argv[3:]:
        t = open(doc, encoding="utf-8").read()
        for k, v in blocks.items():
            t = re.sub(rf"<!-- TABLES:{k} -->.*?<!-- /TABLES:{k} -->",
                       lambda _m, _v=v, _k=k: f"<!-- TABLES:{_k} -->\n{_v}\n<!-- /TABLES:{_k} -->",
                       t, flags=re.S)
        open(doc, "w", encoding="utf-8", newline="\n").write(t)
        print(f"[report] spliced -> {doc}")
    print(f"[report] -> {out}")


if __name__ == "__main__":
    main()
