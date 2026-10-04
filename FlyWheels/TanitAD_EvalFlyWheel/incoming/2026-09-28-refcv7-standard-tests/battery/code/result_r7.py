"""The milestone RESULT (JSON + a short markdown section) from the battery's own artifacts.

    python result_r7.py --tag step5000

Reads `raw/<tag>/battery_summary.json` (G0, rolls, bars R7), `raw/<tag>/cross_paired_s{0,1}.json`
(the four families, paired), `raw/<tag>/perc_score.json` (BAR-M7 / BAR-B7) and writes
`raw/<tag>/RESULT_<tag>.json` + `raw/<tag>/RESULT_SECTION_<tag>.md`. Every number is copied from those
files with its tier, estimator, n and interval; nothing is recomputed here. A missing artifact is
written as NOT RUN / NOT EVALUABLE with the reason, never silently dropped.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
FAMS = {"ADE": ("ade_m", "fde_m"),
        "longitudinal": ("LON_speed_mae_mps", "LON_along_mae_m", "LON_accel_mae_mps2"),
        "lateral": ("LAT_cross_mae_m", "LAT_heading_mae_deg", "LAT_yaw_rate_mae_radps_valid"),
        "tactical": ("TAC_traj_lat_correct", "TAC_traj_lon_correct")}


def _load(p):
    try:
        return json.load(open(p, encoding="utf-8"))
    except Exception:                                          # noqa: BLE001 -- recorded as absent
        return None


def _cell(c):
    if not c:
        return "absent"
    if not isinstance(c, dict) or c.get("delta") is None or c.get("lo") is None:
        return str((c or {}).get("status") or "undefined") if isinstance(c, dict) else str(c)
    return (f"{c['delta']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}]{' SEP' if c.get('separated') else ''}"
            f" (n {c.get('n_windows')}/{c.get('n_episodes')})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--root", default="D:/refcv7_eval_kit/battery")
    a = ap.parse_args()
    R = Path(a.root) / a.tag
    s = _load(R / "battery_summary.json") or {}
    ps = _load(R / "perc_score.json")
    res = {"tool": "result_r7.py", "tag": a.tag, "written": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "ckpt": s.get("ckpt"), "ckpt_md5": s.get("ckpt_md5"), "step": s.get("step"),
           "spec_sha256": s.get("spec_sha256"), "is_milestone": s.get("is_milestone"),
           "tier": "T1 self-action OPEN LOOP (status UNRULED for an action-free model); never closed "
                   "loop, never driving performance",
           "estimator": "FULL-SET point estimate; paired episode-cluster bootstrap, n_boot 2000, seed 0",
           "g0": (s.get("stages") or {}).get("g0"),
           "bars_planner": s.get("bars"), "inference_seed_replicate": s.get("inference_seed_replicate"),
           "baselines_missing": s.get("baselines_missing"),
           "strategic": s.get("strategic") or {"status": "NOT APPLICABLE", "n": 0,
                                               "reason": "strategic layer OFF"},
           "families": {}, "bars_map": None, "bars_box": None}
    for seed in (0, 1):
        cp = _load(R / f"cross_paired_s{seed}.json")
        if not cp:
            res["families"][str(seed)] = "NOT RUN"
            continue
        blk = {}
        for pair in ("os_minus_ha0ext", "os_minus_refcv6_38k_s%d" % seed, "filteroff_minus_os"):
            fam = ((cp.get("pairs") or {}).get(pair) or {}).get("families")
            if not fam:
                blk[pair] = "ABSENT"
                continue
            blk[pair] = {f: {m: (fam.get(f) or {}).get(m) for m in ms} for f, ms in FAMS.items()}
        res["families"][str(seed)] = blk
    if ps:
        res["bars_map"] = {k: v.get("verdict") for k, v in ((ps.get("map") or {}).get("bars") or {}).items()
                           if isinstance(v, dict) and k.startswith("BAR")} or (ps.get("map") or {}).get("bars")
        res["bars_box"] = {k: {"verdict": v.get("verdict"), "cell": v.get("cell")}
                           for k, v in ((ps.get("box") or {}).get("bars") or {}).items()
                           if isinstance(v, dict) and k.startswith("BAR")} or (ps.get("box") or {}).get("bars")
        res["box_conf_ratio_watch"] = (ps.get("box") or {}).get("conf_ratio_watch")
    else:
        res["bars_map"] = res["bars_box"] = "NOT RUN (no perc_score.json)"
    json.dump(res, open(R / f"RESULT_{a.tag}.json", "w", encoding="utf-8"), indent=1, default=str)
    L = [f"## RESULT — refcv7 {a.tag} ({res['written']})", ""]
    if not res["is_milestone"]:
        L += ["> ⛔ **PIPELINE VALIDATION ONLY — not a result** (SPEC §6). No bar is evaluated.", ""]
    g0 = res["g0"] or {}
    md = g0.get("mutation_detection") or {}
    L += [f"* **G0 as registered: {g0.get('G0_as_registered')}**"
          f"{' — ' + '; '.join((g0.get('reasons_as_registered') or [])[:3]) if g0.get('reasons_as_registered') else ''}",
          *([f"* **G0-A2 (seeds 0..7): {g0.get('G0_A2')}**"
             f"{' — ' + '; '.join((g0.get('reasons_A2') or [])[:3]) if g0.get('reasons_A2') else ''}"]
            if g0.get("G0_A2") is not None else []),
          f"* **G0-{g0.get('amendment') or 'A2'} (THE GATE, SPEC {g0.get('amendment') or 'A2'}; "
          f"{g0.get('n_seeds') or 8} inference seeds): {g0.get('G0')}**; mutation terms moved: "
          + ", ".join(f"{m.upper()} {d.get('n_terms_out')} ({'detected' if d.get('detected') else 'NOT detected'})"
                      for m, d in md.items())
          + f"; wrapper clause {g0.get('wrapper_clause')}; reasons: {(g0.get('reasons') or [])[:5]}"
          + (" ⛔ M1 moved 0 terms → G0 is VOID" if (md.get('m1') or {}).get('n_terms_out') == 0 else ""),
          *([f"* **G0-A5 (24 seeds; reported beside the A6 gate): {g0.get('G0_A5')}**"
             f"{' — ' + '; '.join((g0.get('reasons_A5') or [])[:3]) if g0.get('reasons_A5') else ''}"]
            if g0.get("amendment") == "A6" and g0.get("G0_A5") is not None else []),
          *([f"* **G0-A6 ({'REGISTERED' if (g0.get('a6_registration') or {}).get('registered') else 'DRAFT -- reported, NOT the gate'}"
             f"; measured numerics floor): {g0.get('G0_A6')}**"
             f"{' — ' + '; '.join((g0.get('reasons_A6') or [])[:3]) if g0.get('reasons_A6') else ''}"
             f"; floor-rescued terms: {[r.get('term') for r in (g0.get('a6_rescued') or [])]}"
             f"; threshold-target terms: "
             + "; ".join(f"{k} in-run {v.get('inrun')} interval [{v.get('a6_lo')}, {v.get('a6_hi')}] "
                         f"{v.get('verdict')}" for k, v in (g0.get('a6_threshold_terms') or {}).items())]
            if g0.get("G0_A6") is not None else []),
          # Master Mind ruling 2026-10-04: BOTH records, whenever the gate rests on the registered A6 text
          *([f"* ⭐ **G0 RECORD (gate source: {(g0.get('text_override') or {}).get('gate_source')}):** "
             + " · ".join(f"**{x}**" for x in ((g0.get('text_override') or {}).get('record_lines') or []))
             + f" (`{os.path.basename(str((g0.get('text_override') or {}).get('text_verdict_file')))}`, sha256 "
             f"{str((g0.get('text_override') or {}).get('text_verdict_sha256'))[:12]}…; re-verified by the judge in use)"]
            if g0.get("text_override") else []),
          f"* **Tier:** {res['tier']}. **Estimator:** {res['estimator']}.",
          f"* **Inference-seed floor:** {(res['inference_seed_replicate'] or {}).get('floor_m')} m ADE "
          f"(training-seed floor NOT measured).", ""]
    L += ["| bar | verdict | seed 0 | seed 1 |", "|---|---|---|---|"]
    for b in res["bars_planner"] or []:
        per = b.get("per_inference_seed") or {}
        L.append(f"| {b['id']} | {b.get('verdict')} | {_cell(per.get(0) or per.get('0'))} | "
                 f"{_cell(per.get(1) or per.get('1'))} |")
    if isinstance(res["bars_map"], dict):
        for k, v in res["bars_map"].items():
            L.append(f"| {k} | {v} | (per band in perc_score.json) | — |")
    if isinstance(res["bars_box"], dict):
        for k, v in res["bars_box"].items():
            L.append(f"| {k} | {v.get('verdict')} | {_cell(v.get('cell'))} | — |")
    L += ["", "| BAR-R7-N1 (NavSim) | not this battery | — | — |", "",
          "**Four families, paired `os − ha0_ext` (seed 0):** "]
    f0 = (res["families"].get("0") or {})
    if isinstance(f0, dict) and isinstance(f0.get("os_minus_ha0ext"), dict):
        for f, ms in f0["os_minus_ha0ext"].items():
            L.append(f"* {f}: " + "; ".join(f"{m} {_cell(c)}" for m, c in ms.items()))
    # ---- family LEVELS per arm (seed 0), from the same analysis JSON; n and tier travel with them ---- #
    an = _load(R / "analysis_s0.json")
    tac = _load(R / "tactical_v6_s0.json")
    if an:
        L += ["", "**Family levels, seed 0 (T1; FULL-SET mean; intervals in `analysis_s0.json`):**", "",
              "| arm | speed MAE m/s | target-speed acc | along MAE m | min headway m / time gap s / min TTC s (n, n_closing) | heading ° | curvature 1/m | yaw-rate °/s | cross-track m | tac lat / lon (traj) |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for arm in ("os", "os_filteroff", "ha0_ext", "ha", "ha0", "b_refcv6_38k_s0"):
            ff = (((an.get("arms") or {}).get(arm) or {}).get("four_families") or {})
            if not ff:
                continue
            lo, la, ta = ff.get("longitudinal") or {}, ff.get("lateral") or {}, ff.get("tactical") or {}
            dk = lo.get("distance_keeping") or {}

            def f(x, n=3):
                return "—" if x is None else (f"{x:.{n}f}" if isinstance(x, (int, float)) else str(x))
            ld, od = ta.get("lateral_decision") or {}, ta.get("longitudinal_decision") or {}
            tsa = lo.get("target_speed_acc")
            tsa = tsa.get("within_0.5_mps") if isinstance(tsa, dict) else tsa
            L.append(f"| {arm} | {f(lo.get('speed_mae_mps'))} | {f(tsa)} (±0.5 m/s) | "
                     f"{f(lo.get('along_mae_m'))} | {f(dk.get('mean_headway_min_m'), 2)} / "
                     f"{f(dk.get('mean_time_gap_min_s'), 2)} / {f(dk.get('mean_min_ttc_s'), 1)} "
                     f"(n {dk.get('n')}, closing {dk.get('n_closing')}) | {f(la.get('heading_mae_deg'), 2)} | "
                     f"{f(la.get('curvature_mae_1pm'), 4)} | {f(la.get('yaw_rate_mae_degps'), 2)} | "
                     f"{f(la.get('cross_mae_m'))} | {f(ld.get('accuracy', ld.get('acc')))} / "
                     f"{f(od.get('accuracy', od.get('acc')))} |")
    if tac:
        for head in ("lat", "lon"):
            blk = tac.get(head) or {}
            for surf in ("v6_behaviour_decoder", "z_tac_v7_heads"):
                s_ = blk.get(surf) or {}
                acc = (s_.get("acc") or {}).get("mean") if isinstance(s_.get("acc"), dict) else s_.get("acc")
                L.append(f"* TACTICAL {head} decision, {surf}: acc {acc}, kappa {s_.get('kappa')}, "
                         f"n in band {blk.get('n_in_band')} (label clock: the refcv7 loader's TRUE clip "
                         f"clock, `enable_clip_clock` + G3)")
        g = tac.get("goal_22") or {}
        pc = (g.get("per_class") or {}).get("nav_true") or {}
        n_sc = sum(1 for v in pc.values() if isinstance(v, dict) and (v.get("n_pos") or 0) >= g.get(
            "scoreability_floor_n_pos", 200))
        L.append(f"* TACTICAL goal selection (22 tokens): {n_sc} scoreable classes (n_pos ≥ "
                 f"{g.get('scoreability_floor_n_pos')}); per-class AUROC / AP / P / R in `tactical_v6_s0.json`")
    L.append(f"* strategic: {res['strategic'].get('status')} (n {res['strategic'].get('n')}): "
             f"{res['strategic'].get('reason')}")
    if res.get("baselines_missing"):
        L.append(f"* ⚠️ baselines missing: {res['baselines_missing']}")
    (R / f"RESULT_SECTION_{a.tag}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
