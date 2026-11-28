#!/usr/bin/env python3
"""YAXN P2 analysis: PREREG_AXD.md section 6 -> section 5 evaluated mechanically (IMPL_P2.md fixes the reading). CPU only.
    python axn_analyze.py --root <run root> --tokens <w3 token json> [--smoke]
Per seed s: T = AXN_s (the AXNOISE checkpoint, ax fed honestly = the seam's own A0 call), Ref = YAX's banked X1_AXON_s (same recipe K1, seed, permutation, no noise).
  PDMS clause (a): paired CI-lo(PDMS(T) - PDMS(Ref)) > -0.5.   Gain clauses (b)-(e): g_dn(T) = (v4(IDT) - v4(IDT@-)) / 0.5 on the CRUISE tokens, read on the AXNOISE checkpoint's own IDT arms,
  against u400's banked IDT gain (YAXD P1 run, same tokens/instrument): (b) <= 0.55 (c) CI-hi < 0.92 (d) CI-hi < CI-lo(AXON_s in-run gain) (e) <= 0.5 x AXON_s in-run gain [AMENDMENT_1; u400's banked gain is descriptive only].   (f) D_lon <= +0.3 vs Ref.  (g) CRUISE floor.
Overall: CANDIDATE iff BOTH seeds CANDIDATE; REFUTED iff BOTH REFUTED; any red gate -> VOID; else NOT PROVEN.
Writes <root>/analysis/P2.json + P2.md. Last line: ZZAXNVERDICT P2 <verdict> s0=<..> s1=<..>
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import axn_common as N  # noqa: E402

A, C = N.A, N.C
sys.path.insert(0, str(A.HERE))
import axd_analyze as AN  # noqa: E402  (YAXD: gate_idt, gate_memo, gate_reach_idt_gain, gains, decide -- the P1 code, pinned)
import ax_stats as S  # noqa: E402
import ax_analyze as AA  # noqa: E402


def load_records(root, s):
    return load_records_dir(Path(root) / f"seam_arms_s{s}")


def load_records_dir(d):
    d = Path(d)
    z = np.load(d / "records.npz", allow_pickle=False)
    rec = {k: z[k] for k in z.files}
    for arm in ("IDT", "IDTm"):
        sz = np.load(d / arm / "seam.npz")
        assert [str(x) for x in sz["token"]] == [str(x) for x in rec["token"]], arm
        rec[f"poses__{arm}"] = sz["poses"]
    return rec, json.load(open(d / "seam_meta.json"))


def train_gates(root, s) -> dict:
    """Everything the training side proved for seed s, as booleans (any False -> VOID)."""
    root = Path(root)
    g = {}
    gp, ap = root / "arms" / f"K1_AXNOISE_s{s}" / "gates.json", root / "arms" / f"K1_AXNOISE_s{s}" / "axnoise.json"
    gj = N.jload(gp) if gp.exists() else {}
    g["train_gates_pass(G-REACH,G-DECL,GA_FEED,GB_STEP0,AXN_NOISE_STD,AXN_HONEST_GATE)"] = bool(gj.get("pass"))
    g["G_DECL"] = bool(gj.get("G_DECL"))
    g["G_REACH"] = bool(gj.get("G_REACH"))
    aj = N.jload(ap) if ap.exists() else {}
    g["post_gates_pass(G_NOISE_N,G_NOISE_STD_TRAIN,G_HONEST_VAL)"] = bool(aj.get("pass"))
    g["sigma_recorded_equals_prereg"] = bool(aj.get("sigma") == N.SIGMA and aj.get("noise_seed") == s + N.NOISE_SEED_OFFSET)
    kp = root / "ckpt" / f"AXN_s{s}.pt.gates.json"
    kj = N.jload(kp) if kp.exists() else {}
    g["compose_K1_K5"] = bool(kj) and all(v for k, v in kj.items() if k.startswith("K") and v is not None)
    ip = root / "identity" / f"AXN_s{s}.json"
    g["G_CKPT_identity"] = bool(ip.exists() and N.jload(ip).get("pass"))
    return g


def seam_gates(rec, meta, toks, pre="") -> dict:
    """the seam-level gates for one checkpoint's records (AXNOISE_s and, AMENDMENT_1, the in-run AXON_s reference): pure functions of the records."""
    rt = [str(t) for t in rec["token"]]
    return {f"{pre}G-SEAM": {"rc": meta["rc"], "missing_records": meta["missing_records"], "n": len(rt), "PASS": meta["rc"] == 0 and not meta["missing_records"] and sorted(rt) == sorted(toks)},
            f"{pre}G-IDT": AN.gate_idt(rec), f"{pre}G-MEMO": AN.gate_memo(rec), f"{pre}G-REACH:IDTgain": AN.gate_reach_idt_gain(rec)}


def cruise_gain(rec, logs_by_tok):
    """(gain block of the IDT arms, CRUISE tokens in record order) -- bootstrap over the logs of those tokens."""
    pos = {str(t): i for i, t in enumerate(rec["token"])}
    ct = [str(x) for x in rec["cruise_token"]]
    csel = np.asarray([pos[t] for t in ct], dtype=np.int64)
    return AN.gains(rec, S.Boot([logs_by_tok[t] for t in ct]), "IDT", csel), ct


def run_seed(root, s, toks, rec_meta, g_u400, bt_cache):
    root = Path(root)
    rec, meta = load_records(root, s)
    arm = AA.load_arm(root, f"AXN_s{s}", toks)
    ctl = AA.load_arm(N.YAX_EVAL, N.CONTROL[s], toks)
    G = seam_gates(rec, meta, toks)
    G["G-HARNESS:AXN"] = {"PASS": arm["seam_misses"] == 0 and len(arm["csv"]) >= len(toks)}
    G["G-HARNESS:CONTROL"] = {"PASS": ctl["seam_misses"] == 0 and len(ctl["csv"]) >= len(toks)}
    # AMENDMENT_1: the in-run AXON_s reference (re-composed from the banked K1 decoder tensors, same seam, same IDT/gain arms)
    rrec, rmeta = load_records_dir(root / f"seam_arms_axon_s{s}")
    G.update(seam_gates(rrec, rmeta, toks, "REF:"))
    kp = root / "ckpt" / f"AXON_s{s}.pt.gates.json"
    kj = N.jload(kp) if kp.exists() else {}
    ip = root / "identity" / f"AXON_s{s}.json"
    G["REF:compose_K1_K5"] = {"PASS": bool(kj) and all(v for k, v in kj.items() if k.startswith("K") and v is not None)}
    G["REF:G_CKPT_identity"] = {"PASS": bool(ip.exists() and N.jload(ip).get("pass"))}
    tg = train_gates(root, s)
    red = [k for k, v in G.items() if not v["PASS"]] + [k for k, v in tg.items() if not v] + ([] if rec_meta["gates_axn"] else ["gates_axn(sigma0-identity, noise-reach, determinism)"])
    logs = [str(arm["pw"]["log"][i]) for i in range(len(toks))]
    bt = bt_cache.setdefault("bt", S.Boot(logs))
    lbt = dict(zip(toks, logs))
    v0, a0, vg, g2, cr = AA.cruise_inputs(toks)
    g_own, ct_own = cruise_gain(rec, lbt)                                     # the AXNOISE checkpoint's own gain (its IDT arms)
    g_ref, ct_ref = cruise_gain(rrec, lbt)                                    # AXON_s in-run gain: the reference of clauses (d), (e)
    if set(ct_own) != set(ct_ref):
        red.append("REF:cruise_token_set_differs")
    n_cr, n_cl = int(cr.sum()), len(set(np.asarray(logs)[cr]))
    floor_ok = n_cr >= 60 and n_cl >= 15
    sT, sR = arm["sub"]["score"], ctl["sub"]["score"]
    dd = sT - sR
    pdms = {"point": float(dd.mean()), "ci95": S.ci(bt.stat(dd)), "T_x100": float(sT.mean()), "Ref_AXON_x100": float(sR.mean())}
    subs = {k: {"point": float((arm["sub"][k] - ctl["sub"][k]).mean()), "ci95": S.ci(bt.stat(arm["sub"][k] - ctl["sub"][k]))} for k in AA.SUB}
    bT, bR = A.v4(arm["poses"]) - vg, A.v4(ctl["poses"]) - vg
    d_lon = abs(bT[cr].mean()) - abs(bR[cr].mean())
    dl = np.abs(bt.stat(bT * cr, cr.astype(float))) - np.abs(bt.stat(bR * cr, cr.astype(float)))

    def lat(num, den, scale=1.0):
        c = S.contrast(bt, [(arm["pw"][num], arm["pw"][den])], [(ctl["pw"][num], ctl["pw"][den])])
        return {"point": c["point"] * scale, "ci95": [x * scale for x in c["ci95"]]}
    fam = {"LONGITUDINAL": {"cruise_b4_T": float(bT[cr].mean()), "cruise_b4_Ref": float(bR[cr].mean()), "D_lon": float(d_lon), "D_lon_ci95": S.ci(dl),
                            "speed_bias_4s_all_T": float(bT.mean()), "speed_mae_4s_all_T": float(np.abs(bT).mean()), "speed_mae_4s_all_Ref": float(np.abs(bR).mean()),
                            "along_mae_m(T-Ref)": lat("along_abs_sum", "along_n"), "along_bias_m(T-Ref)": lat("along_signed_sum", "along_n"),
                            "headway_ttc": "n/a: the NAVSIM seam carries no lead-agent headway; PDMS TTC sub-score is the proxy (subscores)"},
           "LATERAL(T-Ref)": {"curvature_mae_1pm": lat("curv_abs_sum", "curv_n"), "heading_mae_deg": lat("head_abs_sum_rad", "head_n", 180 / math.pi),
                              "yaw_rate_mae_degps": lat("yaw_abs_sum_radps", "yaw_n", 180 / math.pi), "cross_mae_m": lat("cross_abs_sum", "cross_n")},
           "TACTICAL": N.TACTICAL, "STRATEGIC": N.STRATEGIC}
    dec = AN.decide("AXN", red, pdms, g_own, g_ref, d_lon, floor_ok)         # AMENDMENT_1: relative clauses (d), (e) read AXON_s's in-run gain
    dec_u400 = AN.decide("AXN", red, pdms, g_own, g_u400, d_lon, floor_ok)    # DESCRIPTIVE ONLY: the pre-amendment reference (u400's banked P1 gain)
    rp = root / "ref_axon" / f"AXON_s{s}" / "repro.json"
    return {"gates": G, "train_gates": tg, "gates_red": red, "pdms_delta_x100": pdms, "subscores_delta_x100": subs, "gain_own_IDT": g_own, "gain_AXON_ref_IDT(in-run)": g_ref,
            "gain_ratio_to_AXON_ref": g_own["down"]["point"] / g_ref["down"]["point"], "gain_ratio_to_u400": g_own["down"]["point"] / g_u400["down"]["point"], "families": fam,
            "share_plans_changed_vs_Ref": float(np.mean([not np.array_equal(arm["poses"][i], ctl["poses"][i]) for i in range(len(toks))])), "decision": dec,
            "descriptive_decision_vs_u400_banked_gain(not the rule)": dec_u400["verdict"], "n_cruise": n_cr, "n_cruise_logs": n_cl,
            "AXON_ref_repro_vs_banked_X1_seam(reported)": (N.jload(rp) if rp.exists() else None)}


def overall(v0, v1):
    vs = (v0, v1)
    if "VOID" in vs:
        return "VOID"
    if all(v == "CANDIDATE" for v in vs):
        return "CANDIDATE"
    if all(v == "REFUTED" for v in vs):
        return "REFUTED"
    return "NOT PROVEN"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    root = Path(a.root)
    toks = sorted(C.jload(a.tokens)["tokens"])
    gp = root / "analysis" / "gates_axn.json"
    rec_meta = {"gates_axn": bool(gp.exists() and N.jload(gp).get("pass"))}
    g_u400 = N.u400_gain()
    seeds, bt_cache = {}, {}
    for s in N.SEEDS:
        seeds[f"s{s}"] = run_seed(root, s, toks, rec_meta, g_u400, bt_cache)
    v = {k: seeds[k]["decision"]["verdict"] for k in seeds}
    verdict = overall(v["s0"], v["s1"])
    F = {"pdms_delta": abs(seeds["s0"]["pdms_delta_x100"]["point"] - seeds["s1"]["pdms_delta_x100"]["point"]),
         "gain_down": abs(seeds["s0"]["gain_own_IDT"]["down"]["point"] - seeds["s1"]["gain_own_IDT"]["down"]["point"])}
    out = {"verdict": verdict, "per_seed_verdict": v, "tier": N.TIER, "variance": N.VARIANCE, "tactical": N.TACTICAL, "strategic": N.STRATEGIC, "smoke": a.smoke, "seeds": seeds,
           "seed_floor_F": F, "u400_IDT_gain_DESCRIPTIVE_ONLY(P1.json, INHERITED-banked, same tokens; not a clause reference since AMENDMENT_1)": g_u400,
           "relative_gain_reference": "AXON_s in-run IDT gain, per seed (AMENDMENT_1)", "u400_P1_json_sha256": N.sha256_file(N.P1_JSON), "n_tokens": len(toks),
           "hypothesis_attenuation(PREREG6, not a clause)": {"predicted_gain_ratio": round(0.813 ** 2 / (0.813 ** 2 + N.SIGMA ** 2), 3), "measured_ratio_to_u400": {k: seeds[k]["gain_ratio_to_u400"] for k in seeds}, "measured_ratio_to_AXON_ref": {k: seeds[k]["gain_ratio_to_AXON_ref"] for k in seeds}},
           "prereg_sha256": C.sha256_file(A.PREREG), "written": N.now()}
    an = root / "analysis"
    an.mkdir(exist_ok=True)
    C.jdump(out, an / "P2.json")
    L = [f"# YAXN P2 AXNOISE (sigma {N.SIGMA}, K1 recipe, seeds 0 and 1){' -- SMOKE, mechanics only, not decision data' if a.smoke else ''}: **{verdict}**", "", N.TIER, "", N.VARIANCE, "",
         f"Relative-gain reference = AXON_s in-run (AMENDMENT_1). u400's banked P1 down-gain, descriptive only: {g_u400['down']['point']:.3f} [{g_u400['down']['ci95'][0]:.3f}, {g_u400['down']['ci95'][1]:.3f}]. Attenuation hypothesis predicts ratio ~{out['hypothesis_attenuation(PREREG6, not a clause)']['predicted_gain_ratio']}.", "",
         "| seed | verdict | dPDMS x100 vs AXON [CI] | own g_dn [CI] | AXON_s in-run g_dn [CI] | g_dn/AXON_s | g_dn/u400 (desc.) | g_sym | D_lon | curvature dMAE | failing |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, r in seeds.items():
        p, gd, d = r["pdms_delta_x100"], r["gain_own_IDT"]["down"], r["decision"]
        L.append(f"| {k} | {d['verdict']} | {p['point']:+.3f} [{p['ci95'][0]:+.3f}, {p['ci95'][1]:+.3f}] | {gd['point']:.3f} [{gd['ci95'][0]:.3f}, {gd['ci95'][1]:.3f}] | {r['gain_AXON_ref_IDT(in-run)']['down']['point']:.3f} [{r['gain_AXON_ref_IDT(in-run)']['down']['ci95'][0]:.3f}, {r['gain_AXON_ref_IDT(in-run)']['down']['ci95'][1]:.3f}] | {r['gain_ratio_to_AXON_ref']:.2f} | {r['gain_ratio_to_u400']:.2f} | "
                 f"{r['gain_own_IDT']['sym']['point']:.3f} | {r['families']['LONGITUDINAL']['D_lon']:+.3f} | {r['families']['LATERAL(T-Ref)']['curvature_mae_1pm']['point']:+.5f} | {d.get('failing', d.get('why', ''))} |")
    L += ["", f"Seed floor F: {json.dumps(F)}", N.TACTICAL, N.STRATEGIC, "", "Red gates: " + json.dumps({k: r["gates_red"] for k, r in seeds.items()})]
    (an / "P2.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"ZZAXNVERDICT P2 {verdict} s0={v['s0'].replace(' ', '_')} s1={v['s1'].replace(' ', '_')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
