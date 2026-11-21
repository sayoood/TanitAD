"""ss_score.py -- SELFSHARP mechanical scorer (PREREG_SELFSHARP.md).  CPU only.

    python ss_score.py --out <dir> --bank-ss <dir with trace_selfsharp_s0.npz> [--selftest]

BASE-os = L1's un-hooked os-only bank, ORACLE = L2's banked trace_oracle_s0.  Same pipeline as pg_score.py / l2_score.py
(arm_cells -> pair_read_r8.paired_cell, B 2000, seed 0, over EPISODES), shipped stage on.  A missing arm is PENDING, never PASS.
--selftest substitutes the ORACLE trace for SELFSHARP (plumbing only; verdict meaningless).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
L2 = HERE.parent
sys.path.insert(0, str(L2 / "code"))
sys.path.insert(0, str(L2 / "probegate"))
import l2_common as C  # noqa: E402
import l2_score as S  # noqa: E402
from pg_score import ceiling_share, fixed_broken, load_trace_full, ESTIMATOR  # noqa: E402

LON6_V7_IDS = (0, 1, 3, 4, 5, 7)
MD_BASE_PREREG, FD_BASE_PREREG = 0.4695, 0.1395          # literals from the prereg (ORACLE_ATTR.json), K3 checks them to 4 dp
NEXT = {
    "PASS": "refcv9 trains the lon head toward confident posteriors (or ships a calibrated temperature); SHIP-R8 re-reads the arm on the per-frame bank.",
    "PASS-WITH-CHANNEL-COST": "as PASS, but the worse rows are the forcing channel's own cost (ORACLE pays them too): the deployable form is a softer sharpen "
                              "(temperature sweep on the lon logits) read against the same rows.",
    "FAIL_P1": "sharpening the head's own choice does not catch decelerations on its own: the sharpen gain needs the head to be RIGHT, so the lever is head "
               "accuracy (train the lon head toward the z_tac probe on the clean L1C split).",
    "FAIL_P2": "sharpening converts cruise into false decels: read the temperature sweep (partial sharpen) and the per-class FD split before training anything.",
    "FAIL_G": "sharpening degrades rows the ORACLE force does not: the cost is in the head's wrong choices, not the channel -- gate the sharpen on head margin.",
    "INVALID": "a control failed: fix the instrument, re-score from the banked trace (no GPU), then read the bars.",
}


def sha256_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--bank-ss", default="F:/refcv9_selfsharp")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    want = (HERE / "PREREG_SELFSHARP.sha256").read_text().split()[0]
    pre = sha256_file(HERE / "PREREG_SELFSHARP.md")
    if pre != want and not a.selftest:
        raise SystemExit("[ss_score] PREREG_SELFSHARP.md does not match its seal: refusing")
    st = C.stack()
    PR = st["PR"]
    rows = S.metric_rows(PR)
    metrics_keys = [k for (_f, k, _l, _u, _lw, _s) in PR.METRICS]
    E = C.load_eval_pf(0)
    ex = np.load(C.ARCH / C.EVAL_TREE / "dump_s0" / "refcv6_extras.npz", allow_pickle=True)
    clip_r, ws_r = ex["clip_index"].astype(np.int64), ex["ws"].astype(np.int64)
    n = len(ws_r)
    ctl, pending = {}, []
    res = {"tool": "ss_score.py", "prereg_sha256": pre, "script_sha256": sha256_file(__file__), "estimator": ESTIMATOR, "selftest": bool(a.selftest),
           "tier": "T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)", "controls": ctl, "pending": pending,
           "families_not_computed": {"LONGITUDINAL distance-keeping (headway/TTC)": "no lead-agent track in the banked refcv8 dumps; n = 0; named work item",
                                     "STRATEGIC": "refcv8 runs --no-strategic; n = 0",
                                     "S2 rows": "not computed in this run (prereg: PENDING)"}}
    b1 = S.load_l1_base("C:/Users/Admin/refcv8_pf/refcv9_l1/bank", 0)
    if not b1 or b1.get("incomplete"):
        raise SystemExit("[ss_score] BASE-os (L1 un-hooked os-only bank) absent/incomplete")
    orc_p = L2 / "bank" / "trace_oracle_s0.npz"
    ss_p = orc_p if a.selftest else Path(a.bank_ss) / "trace_selfsharp_s0.npz"
    lab_p = (L2 / "bank" / "oracle_labels.npz") if a.selftest else HERE / "labels_selfsharp_s0.npz"
    traces = {}
    for nm, p in (("ORACLE", orc_p), ("SELFSHARP", ss_p)):
        t = S.load_trace(p)
        if t is None or t.get("incomplete"):
            pending.append(f"{nm} ({'incomplete' if t else 'not rolled'})")
        else:
            traces[nm] = t
    plan = {"BASE-os": S.join_plan8(b1, E)}
    for nm, t in traces.items():
        plan[nm] = S.join_plan8(t, E)
    cells = {nm: S.arm_cells(E, p8) for nm, p8 in plan.items()}
    # ---- controls ----------------------------------------------------------------------------------------------------------------------
    ctl["K0_seal"] = bool(pre == want)
    ctl["K_stage_bit_equal_banked_s0"] = bool(np.array_equal(C.apply_stage(E["plan8"], E["prior8"]), E["stage_banked"]))
    orc_raw = np.load(orc_p)["raw_lon_logits"]
    # K1: the labels on disk are exactly argmax(raw) over the LON6 v7 columns, recomputed from the banked ORACLE trace
    lz = np.load(lab_p)
    lab = lz["lon_v7"].astype(np.int64)
    if not a.selftest:
        L6 = np.array(LON6_V7_IDS)
        re = L6[orc_raw[:, L6].argmax(1)]
        ctl["K1_labels_equal_argmax_raw"] = {"pass": bool(np.array_equal(re, lab) and np.array_equal(lz["clip_index"], clip_r) and np.array_equal(lz["ws"], ws_r)),
                                             "n": int(len(lab)), "n_forced": int(np.isin(lab, LON6_V7_IDS).sum())}
    if "SELFSHARP" in traces:
        z, meta = load_trace_full(ss_p)
        k = {}
        forced = np.isin(lab, LON6_V7_IDS)
        k["order_equals_extras"] = bool(np.array_equal(z["e_i"], clip_r) and np.array_equal(z["ws"], ws_r))
        k["n_forced"], k["n_unforced"] = int(forced.sum()), int((~forced).sum())
        c = meta["counters"]
        k["counters"] = {kk: c[kk] for kk in ("calls", "hook_applied", "hook_native_fallthrough", "hook_calls_unarmed", "multi_hook_in_call", "batch_gt1", "navnull_calls", "oracle_labelled")}
        k["counters_ok"] = bool(c["calls"] == n and c["hook_applied"] == k["n_forced"] and c["oracle_labelled"] == k["n_forced"] and c["hook_native_fallthrough"] == k["n_unforced"]
                                and c["hook_calls_unarmed"] == 0 and c["multi_hook_in_call"] == 0 and c["batch_gt1"] == 0 and c["navnull_calls"] == 0)
        app, raw = z["applied_lon_logits"], z["raw_lon_logits"]
        k["applied_onehot_on_forced"] = bool(forced.any() and (app[forced].argmax(1) == lab[forced]).all() and (app[forced].max(1) == 20.0).all())
        k["raw_equals_ORACLE_raw"] = bool(np.array_equal(raw, orc_raw))
        k["pass"] = bool(k["order_equals_extras"] and k["counters_ok"] and k["applied_onehot_on_forced"] and k["raw_equals_ORACLE_raw"])
        ctl["K_hook_SELFSHARP"] = k
        # K2 (diagnostic, stated): windows whose raw head already put >= 20 of margin on its argmax are (near) unchanged vs BASE-os
        jb = np.array([b1["idx"][(int(cc), int(w))] for cc, w in zip(clip_r, ws_r)])
        L6 = list(LON6_V7_IDS)
        srt = np.sort(raw[:, L6], 1)
        conf = (srt[:, -1] - srt[:, -2]) >= 20.0
        dmax = np.abs(z["plan8"] - b1["plan8"][jb]).reshape(n, -1).max(1)
        res["K2_diagnostic_confident_windows"] = {"n_margin_ge_20": int(conf.sum()), "max_abs_plan8_change_m": float(dmax[conf].max()) if conf.any() else None,
                                                  "median_abs_plan8_change_all_m": float(np.median(dmax))}
    # K3: BASE-os rates reproduce the prereg's literals (ORACLE_ATTR.json) to 4 dp
    md_b, fd_b = S.rate(cells["BASE-os"]["stage"], "missed_decel"), S.rate(cells["BASE-os"]["stage"], "false_decel")
    ctl["K3_base_rates_reproduce_prereg"] = {"pass": bool(round(md_b, 4) == MD_BASE_PREREG and round(fd_b, 4) == FD_BASE_PREREG), "md": md_b, "fd": fd_b}
    ident = S.compare(PR, E, cells["BASE-os"]["stage"], cells["BASE-os"]["stage"], rows)
    ctl["K4_identity_BASE_vs_BASE_reads_zero"] = {"pass": bool(all(c["delta"] == 0 and c["lo"] == 0 and c["hi"] == 0 and not c["separated"] for c in ident.values() if "delta" in c)),
                                                  "n_rows": len(ident)}
    l2r = json.loads((L2 / "result" / "final" / "L2_RESULT.json").read_text())
    ref = l2r["seeds"]["0"]["comparisons"]["oracle|stage|vs BASE-os"]
    comps = {}
    for nm in ("SELFSHARP", "ORACLE"):
        if nm in cells:
            comps[f"{nm}|stage|vs BASE-os"] = S.compare(PR, E, cells[nm]["stage"], cells["BASE-os"]["stage"], rows)
    if "ORACLE" in cells:
        mine = comps["ORACLE|stage|vs BASE-os"]
        ctl["K5_oracle_vs_base_reproduces_L2"] = {"pass": bool(all(mine[k][f] == ref[k][f] for k in ("missed_decel", "false_decel", "ade_m") for f in ("delta", "lo", "hi", "separated")))}
    res["rates"] = {f"{nm}|{sk}": {"false_decel": S.rate(c[sk], "false_decel"), "missed_decel": S.rate(c[sk], "missed_decel"),
                                   "n_gt_cruise": int(np.isfinite(c[sk]["false_decel"]).sum()), "n_gt_decel": int(np.isfinite(c[sk]["missed_decel"]).sum())}
                    for nm, c in cells.items() for sk in ("stage", "nostage")}
    res["comparisons"] = comps
    # ---- bars ----------------------------------------------------------------------------------------------------------------------------
    bars = {"P1": None, "P2": None, "G_strict": None}
    if "SELFSHARP" in cells:
        cb = comps["SELFSHARP|stage|vs BASE-os"]
        bars["P1"] = bool(cb["missed_decel"]["hi"] < 0)
        bars["P2"] = bool(not (cb["false_decel"]["lo"] > 0))
        worse = sorted(k for k in metrics_keys if k in cb and cb[k].get("worse") is True)
        if cb["ade_m"].get("worse") is True and "ade_m" not in worse:
            worse.append("ade_m")
        bars["G_rows_separated_worse"] = worse
        bars["G_strict"] = bool(not worse)
        if "ORACLE|stage|vs BASE-os" in comps:
            cr = comps["ORACLE|stage|vs BASE-os"]
            orc_worse = sorted(k for k in list(metrics_keys) + ["ade_m"] if k in cr and cr[k].get("worse") is True)
            bars["G_rows_separated_worse_ORACLE"] = orc_worse
            bars["G_channel_all_worse_also_ORACLE"] = bool(set(worse) <= set(orc_worse))
        bars["P1_cell"], bars["P2_cell"] = cb["missed_decel"], cb["false_decel"]
    res["bars"] = bars
    desc = {}
    if "SELFSHARP" in cells and "ORACLE" in cells:
        desc["ceiling_share_missed_decel"] = ceiling_share(cells["BASE-os"]["stage"]["missed_decel"], cells["SELFSHARP"]["stage"]["missed_decel"],
                                                           cells["ORACLE"]["stage"]["missed_decel"], E["eid"], a.n_boot)
        desc["changed_windows_vs_base"] = {"missed_decel (GT-decel)": fixed_broken(cells["BASE-os"]["stage"]["missed_decel"], cells["SELFSHARP"]["stage"]["missed_decel"]),
                                           "false_decel (GT-cruise)": fixed_broken(cells["BASE-os"]["stage"]["false_decel"], cells["SELFSHARP"]["stage"]["false_decel"])}
    res["descriptive"] = desc
    res["seed_floor_context"] = {"L2 BASE-os inference-seed floor |MD s0 - s1| (stage)": abs(l2r["seeds"]["0"]["rates"]["BASE-os|stage"]["missed_decel"] - l2r["seeds"]["1"]["rates"]["BASE-os|stage"]["missed_decel"]),
                                 "evidence": "INHERITED (L2_RESULT.json); context for effect size only"}
    # ---- verdict -----------------------------------------------------------------------------------------------------------------------
    ctl_fail = sorted(kk for kk, v in ctl.items() if (isinstance(v, bool) and not v) or (isinstance(v, dict) and v.get("pass") is False))
    if a.selftest:
        verdict, nl = "SELFTEST (plumbing only; stand-in trace; verdict meaningless)", "-"
    elif ctl_fail:
        verdict, nl = f"INVALID (control failed: {ctl_fail})", NEXT["INVALID"]
    elif bars["P1"] is None:
        verdict, nl = "PENDING (" + "; ".join(pending) + ")", "roll the arm"
    elif bars["P1"] and bars["P2"] and bars["G_strict"]:
        verdict, nl = "PASS", NEXT["PASS"]
    elif bars["P1"] and bars["P2"] and bars.get("G_channel_all_worse_also_ORACLE"):
        verdict, nl = "PASS-WITH-CHANNEL-COST", NEXT["PASS-WITH-CHANNEL-COST"]
    else:
        failing = [kk for kk in ("P1", "P2", "G_strict") if bars[kk] is False]
        verdict = "FAIL (" + ", ".join(failing) + ")"
        nl = NEXT["FAIL_P1"] if not bars["P1"] else NEXT["FAIL_P2"] if not bars["P2"] else NEXT["FAIL_G"]
    res["verdict"], res["next_lever"], res["wall_s"] = verdict, nl, round(time.time() - t0, 1)
    C.jdump(res, out / "SS_RESULT.json")
    r = res["rates"]

    def cell(cc):
        return f"{cc['delta']:+.4f} [{cc['lo']:+.4f}, {cc['hi']:+.4f}] separated={cc['separated']}" if cc else "-"
    L = [f"{verdict}", "", f"# SELFSHARP RESULT (mechanical; PREREG_SELFSHARP sha256 {pre[:12]})", "", f"tier: {res['tier']}", f"estimator: {ESTIMATOR}", ""]
    if "SELFSHARP|stage" in r:
        sh = desc.get("ceiling_share_missed_decel")
        L += [f"MD (S6 compliant, stage on): BASE-os {r['BASE-os|stage']['missed_decel']:.4f} -> SELFSHARP {r['SELFSHARP|stage']['missed_decel']:.4f}"
              + (f" (ORACLE {r['ORACLE|stage']['missed_decel']:.4f})" if "ORACLE|stage" in r else ""),
              f"FD: BASE-os {r['BASE-os|stage']['false_decel']:.4f} -> SELFSHARP {r['SELFSHARP|stage']['false_decel']:.4f}"
              + (f" (ORACLE {r['ORACLE|stage']['false_decel']:.4f})" if "ORACLE|stage" in r else ""),
              (f"ceiling share {sh['point']:.3f} [{sh['lo']:.3f}, {sh['hi']:.3f}] (descriptive)" if sh else ""), "",
              "| bar | value | cell |", "|---|---|---|",
              f"| P1 MD separated BELOW BASE-os | {bars['P1']} | {cell(bars.get('P1_cell'))} |",
              f"| P2 FD NOT separated above BASE-os | {bars['P2']} | {cell(bars.get('P2_cell'))} |",
              f"| G-strict no S6 row separated worse | {bars['G_strict']} | {', '.join(bars.get('G_rows_separated_worse', [])) or '-'} |",
              f"| (G-channel) rows ORACLE also degrades | {bars.get('G_channel_all_worse_also_ORACLE')} | {', '.join(bars.get('G_rows_separated_worse_ORACLE', [])) or '-'} |", ""]
    L += [f"controls failed: {ctl_fail or 'none'}", f"pending: {pending or 'none'}", f"families not computed: {res['families_not_computed']}", "", f"NEXT LEVER: {nl}", "", "ZZSSVERDICTENDZZ"]
    (out / "VERDICT.md").write_text("\n".join(L), encoding="utf-8")
    print(json.dumps({"verdict": verdict, "bars": {kk: bars[kk] for kk in ("P1", "P2", "G_strict")}, "ctl_fail": ctl_fail, "pending": pending, "wall_s": res["wall_s"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
