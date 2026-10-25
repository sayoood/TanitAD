"""pg_score.py -- PROBEGATE mechanical scorer (PREREG_PROBEGATE.md sections 5-8).  CPU only.

    python pg_score.py --out <dir> [--bank-pg <dir with trace_probegate_s0.npz / trace_randgate_s0.npz>] [--selftest]

BASE = L1's un-hooked os-only bank (BASE-os), ORACLE = L2's banked trace_oracle_s0 (bit-compatibility proven in the prereg and re-proven here, K3).
Every arm is read through the SAME pipeline as l2_score.py (arm_cells -> pair_read_r8.paired_cell, B 2000, seed 0, over EPISODES), with the shipped stage.
A missing arm is PENDING, never PASS.  --selftest substitutes the ORACLE trace for both new arms to exercise the plumbing (its verdict is meaningless).
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
sys.path.insert(0, str(HERE))
import l2_common as C  # noqa: E402
import l2_score as S  # noqa: E402
import pg_policy as P  # noqa: E402

AUC_HEAD, AUC_PROBE = 0.6850606753548109, 0.7623863391074416     # lonprobe/LONPROBE.json seeds.0.auc (literals; context for the extrapolation line only)
ESTIMATOR = ("paired episode-cluster bootstrap (pair_read_r8.paired_cell -> taniteval.ci), B 2000, seed 0, over EPISODES: answers 'would another draw of EPISODES say this?' ONLY; "
             "NOT another training run (one refcv8 training seed), NOT another inference run (one inference seed, 0)")


def sha256_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_trace_full(p):
    z = np.load(p, allow_pickle=False)
    meta = json.loads(str(z["meta_json"]))
    return {k: z[k] for k in z.files if k != "meta_json"}, meta


def ceiling_share(md_base, md_new, md_orc, eid, n_boot=2000, seed=0):
    """(MD_base - MD_new) / (MD_base - MD_orc) with an episode-cluster bootstrap (DESCRIPTIVE; not a bar)."""
    fin = np.isfinite(md_base)
    e = np.asarray(eid)[fin]
    ue, inv = np.unique(e, return_inverse=True)
    cnt = np.bincount(inv).astype(np.float64)
    sums = [np.bincount(inv, weights=a[fin]) for a in (md_base, md_new, md_orc)]
    pt = [s.sum() / cnt.sum() for s in sums]
    point = (pt[0] - pt[1]) / (pt[0] - pt[2]) if pt[0] != pt[2] else float("nan")
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for b in range(n_boot):
        ii = rng.integers(0, len(ue), len(ue))
        c = cnt[ii].sum()
        mb, mn, mo = (s[ii].sum() / c for s in sums)
        out[b] = (mb - mn) / (mb - mo) if mb != mo else np.nan
    return {"point": float(point), "lo": float(np.nanpercentile(out, 2.5)), "hi": float(np.nanpercentile(out, 97.5)), "md": {"base": pt[0], "new": pt[1], "oracle": pt[2]}}


def fixed_broken(base, new):
    m = np.isfinite(base) & np.isfinite(new)
    return {"fixed": int(((base == 1) & (new == 0) & m).sum()), "broken": int(((base == 0) & (new == 1) & m).sum()), "n": int(m.sum())}


NEXT = {
    "PASS": "L3 -- train refcv9's lon head toward the probe (cross-fitted decel posterior as an auxiliary target / distillation); the gate is the existence proof and the "
            "ceiling share bounds what to expect.",
    "P1_thin": "THIN GATE (P1 failed, but the missed-decel point estimate improved): the gate is directionally right and short of information.  Next lever = a more informative "
               "gate, screened at 0 GPU FIRST on the banked material (nonlinear cross-fit on [z_tac PCA | v0 | head logits | plan decel], same folds; admit to a GPU re-gate only "
               "if its cross-fit AUC beats 0.7624 by a paired-bootstrap-separated margin, and use the AUC* line below as the power target).",
    "P1_inert": "GATE INERT (P1 failed and the missed-decel point estimate did not improve): flipping the head's decel/cruise group at probe quality does not move the plan.  "
                "Next lever = stop gating post hoc; attribute the oracle gap per forced class (BRAKE_TO / CREEP / HOLD) at 0 GPU from the banked oracle trace, then train toward "
                "the probe's posterior (L3) only if one class carries the gap.",
    "P3": "NOT PROBE-SPECIFIC (P3 failed): random flips of the same size reproduce the gain, so the effect is 'forcing more decel-group mass', not the probe's information.  "
          "Next lever = a gate whose flips are conditioned on a MARGIN (flip only where |score - tau| is large) and re-run with the same RANDGATE control.",
    "P2": "FALSE-BRAKE COST (P2 failed): up-flips on GT-cruise windows raise false-decel significantly.  Next lever = an UP-ONLY gate with a margin (drop the down-flips and "
          "the lowest-margin up-flips), same RANDGATE control.",
    "G": "COLLATERAL DAMAGE (G failed): a four-family row or S6 ADE got significantly worse.  Next lever = restrict the force to the longitudinal channel the rows implicate "
         "(see the named rows) or shrink the flip set to the highest-margin windows.",
    "G_channel": "CHANNEL-LEVEL G FAILURE (G failed on rows that the privileged ORACLE force ALSO degrades): forcing the lon class trips these rows even with perfect labels, so G as "
                 "written cannot be passed by any lon-class-force lever.  Next lever = pair the gate with a lateral guard (hold the lateral candidate / selection fixed while the lon "
                 "class is forced) and re-roll; the probe is not what failed.",
}


def next_lever(p1, p2, p3, g, share_point, controls_ok, g_in_oracle=False):
    if not controls_ok:
        return "none -- a control failed; repair the instrument first (see controls)."
    if p1 and p2 and p3 and g:
        return NEXT["PASS"]
    if not p1:
        return NEXT["P1_thin"] if (share_point is not None and share_point > 0) else NEXT["P1_inert"]
    if p3 is False:
        return NEXT["P3"]
    if not p2:
        return NEXT["P2"]
    if not g:
        return NEXT["G_channel"] if g_in_oracle else NEXT["G"]
    return NEXT["P1_inert"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--bank-pg", default="F:/refcv9_probegate")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    t0 = time.time()
    if a.selftest and not (HERE / "PREREG_PROBEGATE.sha256").exists():
        want = pre = "UNSEALED-SELFTEST"
    else:
        want = (HERE / "PREREG_PROBEGATE.sha256").read_text().split()[0]
        pre = sha256_file(HERE / "PREREG_PROBEGATE.md")
        if want != pre:
            raise SystemExit("[pg_score] PREREG_PROBEGATE.md does not match its seal: refusing")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    st = C.stack()
    PR = st["PR"]
    rows = S.metric_rows(PR)
    metrics_keys = [k for (_f, k, _l, _u, _lw, _s) in PR.METRICS]
    E = C.load_eval_pf(0)
    ex = np.load(C.ARCH / C.EVAL_TREE / "dump_s0" / "refcv6_extras.npz", allow_pickle=True)
    clip_r, ws_r = ex["clip_index"].astype(np.int64), ex["ws"].astype(np.int64)
    n = len(ws_r)
    ctl, pending = {}, []
    res = {"tool": "pg_score.py", "prereg_sha256": pre, "script_sha256": sha256_file(__file__), "estimator": ESTIMATOR, "selftest": bool(a.selftest),
           "tier": "T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)", "controls": ctl, "pending": pending,
           "families_not_computed": {"LONGITUDINAL distance-keeping (headway/TTC)": "no lead-agent track in the banked refcv8 dumps; n = 0; named work item",
                                     "STRATEGIC": "refcv8 runs --no-strategic; n = 0"}}
    # ---- the arms ----------------------------------------------------------------------------------------------------------------------
    b1 = S.load_l1_base("C:/Users/Admin/refcv8_pf/refcv9_l1/bank", 0)
    if not b1 or b1.get("incomplete"):
        raise SystemExit("[pg_score] BASE-os (L1 un-hooked os-only bank) absent/incomplete")
    orc_p = L2 / "bank" / "trace_oracle_s0.npz"
    pg_p, rg_p = Path(a.bank_pg) / "trace_probegate_s0.npz", Path(a.bank_pg) / "trace_randgate_s0.npz"
    lab_pg, lab_rg = HERE / "labels_probegate_s0.npz", HERE / "labels_randgate_s0.npz"
    if a.selftest:
        pg_p = rg_p = orc_p
        lab_pg = lab_rg = L2 / "bank" / "oracle_labels.npz"
    traces = {}
    for nm, p in (("ORACLE", orc_p), ("PROBEGATE", pg_p), ("RANDGATE", rg_p)):
        t = S.load_trace(p)
        if t is None or t.get("incomplete"):
            pending.append(f"{nm} ({'incomplete' if t else 'not rolled'})")
        else:
            traces[nm] = t
    plan = {"BASE-bat": E["plan8"], "BASE-os": S.join_plan8(b1, E)}
    for nm, t in traces.items():
        plan[nm] = S.join_plan8(t, E)
    cells = {nm: S.arm_cells(E, p8) for nm, p8 in plan.items()}
    # refcv7 is not part of this experiment (no P4-type bar); not read.
    # ---- controls ----------------------------------------------------------------------------------------------------------------------
    ctl["K1_stage_bit_equal_banked_s0"] = bool(np.array_equal(C.apply_stage(E["plan8"], E["prior8"]), E["stage_banked"]))
    ctl["K2_population_s0"] = bool(int(E["compliant"].sum()) == 3251 and len(set(E["eid"][E["compliant"]].tolist())) == 138)
    ctl["K0_seals"] = {"prereg": pre == want}
    # K3: unforced windows of every new trace are bit-equal to the un-hooked BASE-os plans (the in-situ proof that BASE is reusable)
    jb = np.array([b1["idx"][(int(c), int(w))] for c, w in zip(clip_r, ws_r)])
    base_plan8_s2, base_sel_s2 = b1["plan8"][jb], b1["sel"][jb]
    heads = {}
    for nm, labp in (("ORACLE", L2 / "bank" / "oracle_labels.npz"), ("PROBEGATE", lab_pg), ("RANDGATE", lab_rg)):
        if nm not in traces:
            continue
        z, meta = load_trace_full(pg_p if nm == "PROBEGATE" else rg_p if nm == "RANDGATE" else orc_p)
        lz = np.load(labp)
        k = {}
        k["order_equals_extras"] = bool(np.array_equal(z["e_i"], clip_r) and np.array_equal(z["ws"], ws_r) and np.array_equal(lz["clip_index"], clip_r) and np.array_equal(lz["ws"], ws_r))
        lab = lz["lon_v7"].astype(np.int64)
        forced = np.isin(lab, P.LON6_V7_IDS)
        unforced = ~forced
        k["n_forced"], k["n_unforced"] = int(forced.sum()), int(unforced.sum())
        k["unforced_plan8_max_abs_vs_BASE-os"] = float(np.abs(z["plan8"][unforced] - base_plan8_s2[unforced]).max()) if unforced.any() else 0.0
        k["unforced_sel_equal_BASE-os"] = bool(np.array_equal(z["sel"][unforced], base_sel_s2[unforced]))
        c = meta["counters"]
        k["counters"] = {kk: c[kk] for kk in ("calls", "hook_applied", "hook_native_fallthrough", "hook_calls_unarmed", "multi_hook_in_call", "batch_gt1", "navnull_calls", "oracle_labelled")}
        k["counters_ok"] = bool(c["calls"] == n and c["hook_applied"] == k["n_forced"] and c["oracle_labelled"] == k["n_forced"] and c["hook_native_fallthrough"] == k["n_unforced"]
                                and c["hook_calls_unarmed"] == 0 and c["multi_hook_in_call"] == 0 and c["batch_gt1"] == 0 and c["navnull_calls"] == 0)
        app, raw = z["applied_lon_logits"], z["raw_lon_logits"]
        k["applied_onehot_on_forced"] = bool(forced.any() and (app[forced].argmax(1) == lab[forced]).all() and (app[forced].max(1) == 20.0).all())
        k["applied_equals_raw_on_unforced"] = bool(np.array_equal(app[unforced], raw[unforced]))
        k["raw_equals_ORACLE_raw"] = bool(np.array_equal(raw, np.load(orc_p)["raw_lon_logits"]))
        k["pass"] = bool(k["order_equals_extras"] and k["unforced_plan8_max_abs_vs_BASE-os"] <= 1e-5 and k["unforced_sel_equal_BASE-os"] and k["counters_ok"]
                         and k["applied_onehot_on_forced"] and k["applied_equals_raw_on_unforced"] and k["raw_equals_ORACLE_raw"])
        ctl[f"K3_hook_and_identity_{nm}"] = k
        heads[nm] = {"lab": lab, "forced": forced, "raw": raw}
    # K4: policy re-derivation from the saved scores (real mode) -- the labels on disk are what pg_policy says
    if not a.selftest:
        try:
            zs = np.load(HERE / "probe_scores_s0.npz")
            lp, up, dn, info = P.probegate(zs["score"], np.load(orc_p)["raw_lon_logits"])
            lr, ur, dr, _ = P.randgate(np.load(orc_p)["raw_lon_logits"], int(up.sum()), int(dn.sum()), seed=0)
            ctl["K4_labels_rederived"] = {"probegate_equal": bool(np.array_equal(lp, np.load(lab_pg)["lon_v7"].astype(np.int64))),
                                          "randgate_equal": bool(np.array_equal(lr, np.load(lab_rg)["lon_v7"].astype(np.int64))), "n_up": int(up.sum()), "n_down": int(dn.sum())}
            ctl["K4_labels_rederived"]["pass"] = ctl["K4_labels_rederived"]["probegate_equal"] and ctl["K4_labels_rederived"]["randgate_equal"]
            cj = json.loads((HERE / "LABELS_CHECK.json").read_text())
            vj = json.loads((HERE / "LABELS_VERIFY.json").read_text())
            sh = {f: sha256_file(HERE / f) for f in ("probe_scores_s0.npz", "labels_probegate_s0.npz", "labels_randgate_s0.npz")}
            ctl["K5_labels_checks"] = {"LABELS_CHECK_ALL_PASS": cj.get("ALL_PASS"), "LABELS_VERIFY_ALL_PASS": vj.get("ALL_PASS"), "sha256_equal_recorded": sh == cj.get("sha256"),
                                       "prereg_sha_in_check": cj.get("prereg_sha256") == pre}
            ctl["K5_labels_checks"]["pass"] = bool(cj.get("ALL_PASS") is True and vj.get("ALL_PASS") is True and sh == cj.get("sha256") and cj.get("prereg_sha256") == pre)
        except Exception as exc:                                                       # noqa: BLE001 -- recorded; a missing check is a FAILED control
            ctl["K4_labels_rederived"] = {"pass": False, "error": f"{type(exc).__name__}: {exc}"[:300]}
    # K6/K7: known-null and known-positive controls of the SCORER itself
    ident = S.compare(PR, E, cells["BASE-os"]["stage"], cells["BASE-os"]["stage"], rows)
    ctl["K6_identity_BASE_vs_BASE_reads_zero"] = {"pass": bool(all(c["delta"] == 0 and c["lo"] == 0 and c["hi"] == 0 and not c["separated"] for c in ident.values() if "delta" in c)),
                                                  "n_rows": len(ident)}
    l2r = json.loads((L2 / "result" / "final" / "L2_RESULT.json").read_text())
    ref = l2r["seeds"]["0"]["comparisons"]["oracle|stage|vs BASE-os"]
    if "ORACLE" in cells:
        mine = S.compare(PR, E, cells["ORACLE"]["stage"], cells["BASE-os"]["stage"], rows)
        ctl["K7_oracle_vs_base_reproduces_L2"] = {"pass": bool(all(mine[k][f] == ref[k][f] for k in ("missed_decel", "false_decel", "ade_m") for f in ("delta", "lo", "hi", "separated"))),
                                                  "missed_decel_cell": {f: mine["missed_decel"][f] for f in ("delta", "lo", "hi", "separated")}}
    ctl["K8_oracle_rates_equal_L2"] = {"pass": bool(abs(S.rate(cells["BASE-os"]["stage"], "missed_decel") - l2r["seeds"]["0"]["rates"]["BASE-os|stage"]["missed_decel"]) < 1e-12 and
                                                    abs(S.rate(cells["ORACLE"]["stage"], "missed_decel") - l2r["seeds"]["0"]["rates"]["oracle|stage"]["missed_decel"]) < 1e-12)} if "ORACLE" in cells else {"pass": None}
    # ---- rates -------------------------------------------------------------------------------------------------------------------------
    res["rates"] = {f"{nm}|{sk}": {"false_decel": S.rate(c[sk], "false_decel"), "missed_decel": S.rate(c[sk], "missed_decel"),
                                   "n_gt_cruise": int(np.isfinite(c[sk]["false_decel"]).sum()), "n_gt_decel": int(np.isfinite(c[sk]["missed_decel"]).sum())}
                    for nm, c in cells.items() for sk in ("stage", "nostage")}
    # ---- comparisons and bars ----------------------------------------------------------------------------------------------------------
    comps = {}
    for nm in ("PROBEGATE", "RANDGATE", "ORACLE"):
        if nm in cells:
            comps[f"{nm}|stage|vs BASE-os"] = S.compare(PR, E, cells[nm]["stage"], cells["BASE-os"]["stage"], rows)
    if "PROBEGATE" in cells and "RANDGATE" in cells:
        comps["PROBEGATE|stage|vs RANDGATE"] = S.compare(PR, E, cells["PROBEGATE"]["stage"], cells["RANDGATE"]["stage"], rows)
    res["comparisons"] = comps
    bars = {"P1": None, "P2": None, "P3": None, "G": None}
    if "PROBEGATE" in cells:
        cb = comps["PROBEGATE|stage|vs BASE-os"]
        bars["P1"] = bool(cb["missed_decel"]["hi"] < 0)                               # separated BELOW base: the whole interval is under 0
        bars["P2"] = bool(not (cb["false_decel"]["lo"] > 0))                          # NOT separated above base
        worse = sorted(k for k in metrics_keys if k in cb and cb[k].get("worse") is True)
        bars["G_rows_separated_worse"] = worse
        bars["G_ade_worse"] = bool(cb["ade_m"].get("worse") is True)
        bars["G"] = bool(not worse and not bars["G_ade_worse"])
        for ref_nm in ("ORACLE", "RANDGATE"):                                           # attribution context: does the FORCING CHANNEL itself trip G?
            cr = comps.get(f"{ref_nm}|stage|vs BASE-os")
            if cr:
                bars[f"G_rows_separated_worse_{ref_nm}"] = sorted(k for k in metrics_keys if k in cr and cr[k].get("worse") is True)
        if "G_rows_separated_worse_ORACLE" in bars:
            bars["G_failure_subset_of_ORACLE_rows"] = bool(set(worse) <= set(bars["G_rows_separated_worse_ORACLE"]))
        bars["P1_cell"], bars["P2_cell"] = cb["missed_decel"], cb["false_decel"]
    if "PROBEGATE" in cells and "RANDGATE" in cells:
        cp = comps["PROBEGATE|stage|vs RANDGATE"]["missed_decel"]
        bars["P3"] = bool(cp["hi"] < 0)
        bars["P3_cell"] = cp
    else:
        pending.append("P3 (RANDGATE not rolled)")
    res["bars"] = bars
    # ---- descriptive: ceiling share, flip precision, AUC* line ------------------------------------------------------------------------------
    share = None
    desc = {}
    if "PROBEGATE" in cells and "ORACLE" in cells:
        share = ceiling_share(cells["BASE-os"]["stage"]["missed_decel"], cells["PROBEGATE"]["stage"]["missed_decel"], cells["ORACLE"]["stage"]["missed_decel"], E["eid"], a.n_boot)
        if "RANDGATE" in cells:
            share["randgate"] = ceiling_share(cells["BASE-os"]["stage"]["missed_decel"], cells["RANDGATE"]["stage"]["missed_decel"], cells["ORACLE"]["stage"]["missed_decel"], E["eid"], a.n_boot)
        desc["ceiling_share_missed_decel"] = share
        s = share["point"]
        if s and s > 0:
            auc_star = AUC_HEAD + 0.5 * (AUC_PROBE - AUC_HEAD) / s
            desc["auc_star_for_50pct_ceiling_share"] = {"value": float(auc_star), "evidence": "ESTIMATED (two-point linear extrapolation: head AUC 0.6851 -> share 0, probe AUC 0.7624 -> measured share)",
                                                        "admissible": bool(auc_star <= AUC_PROBE + (AUC_PROBE - AUC_HEAD))}
    for nm in ("PROBEGATE", "RANDGATE"):
        if nm in cells:
            desc[f"{nm}_changed_windows_vs_base"] = {"missed_decel (GT-decel)": fixed_broken(cells["BASE-os"]["stage"]["missed_decel"], cells[nm]["stage"]["missed_decel"]),
                                                     "false_decel (GT-cruise)": fixed_broken(cells["BASE-os"]["stage"]["false_decel"], cells[nm]["stage"]["false_decel"])}
    if "PROBEGATE" in heads:
        zc = json.loads((HERE / "LABELS_CHECK.json").read_text()).get("descriptive_flips_on_S6_compliant") if not a.selftest else None
        desc["flips_on_S6_compliant_from_labels_check"] = zc
    res["descriptive"] = desc
    res["seed_floor_context"] = {"L2 BASE-os inference-seed floor (seed 0 vs 1, stage)": {"false_decel": l2r["inference_seed_floor_fd"]["per_baseline"].get("BASE-os"),
                                                                                          "missed_decel": abs(l2r["seeds"]["0"]["rates"]["BASE-os|stage"]["missed_decel"] - l2r["seeds"]["1"]["rates"]["BASE-os|stage"]["missed_decel"])},
                                 "evidence": "INHERITED (L2_RESULT.json); context for effect sizes only"}
    # ---- verdict -----------------------------------------------------------------------------------------------------------------------
    ctl_fail = sorted(k for k, v in ctl.items() if (isinstance(v, bool) and not v) or (isinstance(v, dict) and v.get("pass") is False))
    controls_ok = not ctl_fail
    have_all = all(bars[k] is not None for k in ("P1", "P2", "P3", "G"))
    failing = [k for k in ("P1", "P2", "P3", "G") if bars[k] is False]
    if a.selftest:
        verdict = "SELFTEST (plumbing only; stand-in traces; verdict meaningless)"
    elif not controls_ok:
        verdict = f"INVALID (control failed: {ctl_fail})"
    elif not have_all and not failing:
        verdict = "PENDING (" + "; ".join(pending) + ")"
    elif failing:
        verdict = "FAIL (" + ", ".join(failing) + ")" + (" [P3 PENDING]" if bars["P3"] is None else "")
    else:
        verdict = "PASS"
    nl = next_lever(bool(bars["P1"]), bool(bars["P2"]), bars["P3"], bool(bars["G"]), (share or {}).get("point"), controls_ok, bool(bars.get("G_failure_subset_of_ORACLE_rows")))
    res["verdict"], res["next_lever"] = verdict, nl
    res["wall_s"] = round(time.time() - t0, 1)
    C.jdump(res, out / "PG_RESULT.json")
    # ---- md ----------------------------------------------------------------------------------------------------------------------------
    r = res["rates"]

    def rt(nm, sk="stage"):
        v = r.get(f"{nm}|{sk}")
        return (f"{v['false_decel']:.4f} | {v['missed_decel']:.4f}") if v else "- | -"
    L = [f"# PROBEGATE RESULT (mechanical; PREREG_PROBEGATE sha256 {pre[:12]})", "", f"**Verdict: {verdict}**", "", f"estimator: {ESTIMATOR}", f"tier: {res['tier']}", "",
         "## rates (S6 compliant, with the shipped stage; false-decel on GT-cruise n 1,856, missed-decel on GT-decel n 639)", "", "| arm | false-decel | missed-decel |", "|---|---|---|"]
    for nm in ("BASE-os", "BASE-bat", "ORACLE", "PROBEGATE", "RANDGATE"):
        if nm in cells:
            L.append(f"| {nm} | {rt(nm)} |")
    L += ["", "## bars", "", "| clause | reading | cell (delta [lo, hi]) |", "|---|---|---|"]

    def cell(c):
        return f"{c['delta']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}] separated={c['separated']}" if c else "-"
    L += [f"| P1 PROBEGATE missed-decel separated BELOW base | {bars['P1']} | {cell(bars.get('P1_cell'))} |", f"| P2 PROBEGATE false-decel NOT separated above base | {bars['P2']} | {cell(bars.get('P2_cell'))} |",
          f"| P3 PROBEGATE - RANDGATE missed-decel separated below 0 | {bars['P3']} | {cell(bars.get('P3_cell'))} |",
          f"| G no METRICS row / S6 ADE separated worse | {bars['G']} | rows: {', '.join(bars.get('G_rows_separated_worse', [])) or '-'} |",
          f"| (context) rows the ORACLE force itself degrades | - | {', '.join(bars.get('G_rows_separated_worse_ORACLE', [])) or '-'} |",
          f"| (context) rows RANDGATE degrades | - | {', '.join(bars.get('G_rows_separated_worse_RANDGATE', [])) or '-'} |", ""]
    if share:
        L += [f"ceiling share (MD_base - MD_probegate)/(MD_base - MD_oracle) = {share['point']:.3f} [{share['lo']:.3f}, {share['hi']:.3f}] (DESCRIPTIVE; MD base {share['md']['base']:.4f}, probegate "
              f"{share['md']['new']:.4f}, oracle {share['md']['oracle']:.4f})", ""]
    L += ["## descriptive", "", "```", json.dumps(desc, indent=1, default=str), "```", "", "## controls", "", "```", json.dumps(ctl, indent=1, default=str), "```", "",
          "## next lever (pre-registered mapping, section 8)", "", nl, ""]
    (out / "PG_RESULT.md").write_text("\n".join(L), encoding="utf-8")
    gt_lines = [f"{verdict}", ""]
    if bars["P1"] is not None:
        gt_lines += [f"P1={bars['P1']}  P2={bars['P2']}  P3={bars['P3']}  G={bars['G']}", f"MD base {r['BASE-os|stage']['missed_decel']:.4f} -> probegate {r['PROBEGATE|stage']['missed_decel']:.4f}"
                     + (f", randgate {r['RANDGATE|stage']['missed_decel']:.4f}" if "RANDGATE|stage" in r else "") + f", oracle {r['ORACLE|stage']['missed_decel']:.4f}",
                     f"FD base {r['BASE-os|stage']['false_decel']:.4f} -> probegate {r['PROBEGATE|stage']['false_decel']:.4f}" + (f", randgate {r['RANDGATE|stage']['false_decel']:.4f}" if "RANDGATE|stage" in r else ""),
                     (f"ceiling share {share['point']:.3f} [{share['lo']:.3f}, {share['hi']:.3f}]" if share else ""), f"G rows worse: {', '.join(bars.get('G_rows_separated_worse', [])) or '-'}  (ORACLE force degrades: {', '.join(bars.get('G_rows_separated_worse_ORACLE', [])) or '-'})", ""]
    gt_lines += [f"estimator: {ESTIMATOR}", "", f"NEXT LEVER: {nl}", ""]
    (out / "VERDICT.md").write_text("\n".join(gt_lines), encoding="utf-8")
    print(json.dumps({"verdict": verdict, "bars": {k: bars[k] for k in ("P1", "P2", "P3", "G")}, "ctl_fail": ctl_fail, "pending": pending, "wall_s": res["wall_s"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
