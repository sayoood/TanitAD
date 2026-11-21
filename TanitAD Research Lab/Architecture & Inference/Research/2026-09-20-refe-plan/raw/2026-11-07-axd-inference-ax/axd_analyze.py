#!/usr/bin/env python3
"""YAXD P1 analysis: the gates and the COMMITTED rule of PREREG_AXD.md sections 4-5, evaluated mechanically per transform. CPU only.
    python axd_analyze.py --root <run root> --tokens <w3 token json> [--smoke]
Reads <root>/seam_arms/{records.npz, seam_meta.json} and <root>/eval/<ARM>/ (A0 + every CANDIDATE: arm.json, perwindow.npz, seam.npz, families.json).
Writes <root>/analysis/P1.json and P1.md. Last line: ZZAXDVERDICT P1 <T>=<verdict> ...
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
import axd_common as A  # noqa: E402

C = A.C
import ax_stats as S  # noqa: E402  (YAX: Boot = paired log-cluster bootstrap, 10,000 resamples, seed 20260927)
import ax_analyze as AA  # noqa: E402  (YAX: read_csv, cruise_inputs, load_arm)

TIER = ("TIER: NAVSIM v1.1 PDMS + families6 of an OPEN-LOOP plan against logged agents on the A5 tokens (T0-family) -- NOT the action-closed-loop tier. Open loop has no ax feedback, "
        "so it shows what a transform COSTS and how much it cuts the plan's ax GAIN; it is never proof that the stall is fixed. The closed-loop ADOPT is PREREG_AXOFF section 6c, run by MM.")
VARIANCE = ("Q1 another draw of LOGS: paired log-cluster bootstrap, 10,000 resamples, percentile 95 %, seed 20260927. Q2 another TRAINING run: not applicable (P1 trains nothing; one "
            "checkpoint, u400). Q3 another INFERENCE run: MEASURED here by G-IDT (a fresh-trunk re-run of the identity arm must equal A0 bit-for-bit on every token).")
TACTICAL = "TACTICAL: n/a as a test -- REFe has no tactical layer; an ax transform changes no decision head (families6 decisions derive from executed geometry; not gated)."
STRATEGIC = "STRATEGIC: PAUSED (PI ruling 2026-09-19, PI_DECISION_QUEUE.md item 25) -- not a pass, not omitted."


# ---------------------------------------------------------------- gates (pure functions of the seam records; unit-tested with RED arms in test_axd.py)
def gate_idt(rec) -> dict:
    fed_eq = bool(np.array_equal(rec["fed__IDT"], rec["raw"]))
    pose_eq = bool(np.array_equal(rec["a0_poses"], _arm_poses(rec, "IDT")))
    seam_eq = bool(np.array_equal(rec["a0_poses"], rec["a0_seam_poses"]))
    return {"fed_equal_raw": fed_eq, "poses_equal_A0": pose_eq, "a0_record_equals_seam_file": seam_eq, "PASS": fed_eq and pose_eq and seam_eq}


def gate_memo(rec) -> dict:
    fe = bool(np.array_equal(rec["fed__IDTm"], rec["raw"]))
    pe = bool(np.array_equal(rec["a0_poses"], _arm_poses(rec, "IDTm")))
    return {"fed_equal_raw": fe, "poses_equal_A0": pe, "PASS": fe and pe}


def _arm_poses(rec, arm):
    return rec[f"poses__{arm}"]


def expected_fed(t, raw, hist, d=0.0, mode="inst"):
    return np.asarray([A.transform(t, float(r), h, d, mode) for r, h in zip(raw[:, A.AX], hist)], np.float32)


def gate_reach(rec, t) -> dict:
    raw, fed = rec["raw"], rec[f"fed__{t}"]
    others = [i for i in range(raw.shape[1]) if i != A.AX]
    exp = expected_fed(t, raw, rec["hist"])
    o_eq = bool(np.array_equal(fed[:, others], raw[:, others]))
    v_eq = bool(np.array_equal(fed[:, A.AX], exp))
    n_ch = int((fed[:, A.AX] != raw[:, A.AX]).sum())
    g_ok, g_bad = True, []
    for n, tt, d, mode in A.gain_arms():
        if tt != t:
            continue
        gf = rec[f"gfed__{n}"]
        if len(gf) and not np.array_equal(gf[:, A.AX], expected_fed(tt, rec["cruise_raw"], rec["cruise_hist"], d, mode)):
            g_ok = False
            g_bad.append(n)
    return {"other_slots_unchanged": o_eq, "slot2_equals_transform": v_eq, "n_tokens_changed": n_ch, "gain_arms_fed_ok": g_ok, "gain_arms_bad": g_bad,
            "PASS": o_eq and v_eq and n_ch >= 1 and g_ok}


def gate_reach_idt_gain(rec) -> dict:
    bad = [n for n, tt, d, mode in A.gain_arms() if tt == "IDT" and len(rec[f"gfed__{n}"])
           and not np.array_equal(rec[f"gfed__{n}"][:, A.AX], expected_fed("IDT", rec["cruise_raw"], rec["cruise_hist"], d, mode))]
    return {"bad": bad, "n": int(len(rec["cruise_raw"])), "PASS": not bad and len(rec["cruise_raw"]) > 0}


def gate_zero(rec, a0x_cruise_poses) -> dict:
    p, m = rec["gpose__AX0@+"], rec["gpose__AX0@-"]
    eq = bool(np.array_equal(p, m) and np.array_equal(p, a0x_cruise_poses))
    return {"AX0_plus_minus_unperturbed_bit_identical": eq, "n": int(len(p)), "PASS": eq and len(p) > 0}


def gate_hist(rec, tol=1e-3) -> dict:
    d = np.abs(rec["ax_export_t0"] - rec["raw"][:, A.AX].astype(np.float64))
    return {"max_abs_export_t0_minus_db_ax": float(d.max()), "tol": tol, "PASS": bool(d.max() <= tol)}


# ---------------------------------------------------------------- gains
def gains(rec, bt_c, t, cr_sel):
    """per-CRUISE-token gains (m/s of planned v@4s per m/s^2): down = (v4(T) - v4(T@-)) / d, up, sym; SMOOTH4's down/sym use the SUSTAINED arms (PREREG section 4)."""
    sfx = "s" if t == "SMOOTH4" else ""
    v0 = A.v4(rec[f"poses__{t}"][cr_sel])
    vm, vp = A.v4(rec[f"gpose__{t}@-{sfx}"]), A.v4(rec[f"gpose__{t}@+{sfx}"])
    out = {}
    for k, g in (("down", (v0 - vm) / A.DELTA), ("up", (vp - v0) / A.DELTA), ("sym", (vp - vm) / (2 * A.DELTA))):
        out[k] = {"point": float(g.mean()), "ci95": S.ci(bt_c.stat(g))}
    if t == "SMOOTH4":
        vmi, vpi = A.v4(rec["gpose__SMOOTH4@-"]), A.v4(rec["gpose__SMOOTH4@+"])
        out["down_instant"] = {"point": float(((v0 - vmi) / A.DELTA).mean()), "ci95": S.ci(bt_c.stat((v0 - vmi) / A.DELTA))}
    if t == "IDT":
        g = (A.v4(rec["gpose__IDT@abs+"]) - A.v4(rec["gpose__IDT@abs-"])) / (2 * A.DELTA)
        out["probe_literal_sym"] = {"point": float(g.mean()), "ci95": S.ci(bt_c.stat(g)), "what": "ax SET to +-0.5 (refe_ego_probe.py:312), the 1.10 instrument on these tokens"}
    return out


def decide(t, gates_red, pdms, g, g_idt, d_lon, floor_ok) -> dict:
    if gates_red:
        return {"verdict": "VOID", "why": f"red gates {gates_red}"}
    gd = g["down"]
    cl = {"pdms_ni": pdms["ci95"][0] > A.PDMS_MARGIN, "gain_point": gd["point"] <= A.GAIN_BAR, "gain_ci_abs": gd["ci95"][1] < A.GAIN_CI_BAR,
          "gain_ci_rel": gd["ci95"][1] < g_idt["down"]["ci95"][0], "gain_half": gd["point"] <= 0.5 * g_idt["down"]["point"], "d_lon": d_lon <= A.LON_BAR, "cruise_floor": floor_ok}
    if all(cl.values()):
        return {"verdict": "CANDIDATE", "clauses": cl}
    if pdms["ci95"][1] < A.PDMS_MARGIN or gd["ci95"][0] > A.GAIN_BAR:
        return {"verdict": "REFUTED", "clauses": cl, "why": "PDMS cost certainly beyond the margin" if pdms["ci95"][1] < A.PDMS_MARGIN else "gain certainly not cut to the bar"}
    return {"verdict": "NOT PROVEN", "clauses": cl, "failing": [k for k, v in cl.items() if not v]}


def load_records(root):
    z = np.load(Path(root) / "seam_arms" / "records.npz", allow_pickle=False)
    rec = {k: z[k] for k in z.files}
    for arm in ("IDT", "IDTm") + A.CANDIDATES:
        s = np.load(Path(root) / "seam_arms" / arm / "seam.npz")
        assert [str(x) for x in s["token"]] == [str(x) for x in rec["token"]]
        rec[f"poses__{arm}"] = s["poses"]
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    root = Path(a.root)
    rec = load_records(root)
    toks = [str(t) for t in rec["token"]]
    meta = json.load(open(root / "seam_arms" / "seam_meta.json"))
    want = sorted(C.jload(a.tokens)["tokens"])
    G = {"G-SEAM": {"rc": meta["rc"], "missing_records": meta["missing_records"], "n": len(toks), "asked": len(want), "PASS": meta["rc"] == 0 and not meta["missing_records"] and sorted(toks) == want}}
    G["G-IDT"], G["G-MEMO"], G["G-HIST"] = gate_idt(rec), gate_memo(rec), gate_hist(rec)
    for t in A.CANDIDATES:
        G[f"G-REACH:{t}"] = gate_reach(rec, t)
    ct = [str(x) for x in rec["cruise_token"]]
    cpos = {t: i for i, t in enumerate(toks)}
    csel = np.asarray([cpos[t] for t in ct], dtype=np.int64)
    G["G-ZERO"] = gate_zero(rec, rec["poses__AX0"][csel])
    G["G-REACH:IDTgain"] = gate_reach_idt_gain(rec)
    arms = {}
    for n in ("A0",) + A.CANDIDATES:
        try:
            arms[n] = AA.load_arm(root, n, toks)
            G[f"G-HARNESS:{n}"] = {"PASS": arms[n]["seam_misses"] == 0 and len(arms[n]["csv"]) >= len(toks)}
        except Exception as e:
            G[f"G-HARNESS:{n}"] = {"PASS": False, "error": repr(e)[:300]}
    # reported controls (never voiding): A0 here vs the banked seam / CSV
    try:
        bz = np.load(C.BANKED_SEAM)
        bi = {str(t): i for i, t in enumerate(bz["token"])}
        bp = np.asarray([bz["poses"][bi[t]] for t in toks])
        bcsv = AA.read_csv(C.BANKED_CSV)
        rep = {"seam_bit_identical": bool(np.array_equal(bp, rec["a0_seam_poses"])), "seam_max_abs_m": float(np.abs(bp.astype(np.float64) - rec["a0_seam_poses"]).max())}
        if "A0" in arms:
            d = np.asarray([float(arms["A0"]["csv"][t]["score"]) - float(bcsv[t]["score"]) for t in toks])
            rep.update({"pdms_max_abs_token_diff": float(np.abs(d).max()), "pdms_A0_here_x100": float(100 * np.mean([float(arms['A0']['csv'][t]['score']) for t in toks])),
                        "pdms_banked_x100_same_tokens": float(100 * np.mean([float(bcsv[t]['score']) for t in toks]))})
        G["R-A0BANK(reported)"] = rep
    except Exception as e:
        G["R-A0BANK(reported)"] = {"error": repr(e)[:300]}
    common = [k for k in ("G-SEAM", "G-IDT", "G-MEMO", "G-ZERO", "G-REACH:IDTgain") if not G[k]["PASS"]] + ([] if G["G-HARNESS:A0"]["PASS"] else ["G-HARNESS:A0"])
    logs = [str(arms["A0"]["pw"]["log"][i]) for i in range(len(toks))] if "A0" in arms else None
    bt = S.Boot(logs) if logs else None
    clogs = [logs[cpos[t]] for t in ct] if logs else None
    bt_c = S.Boot(clogs) if clogs else None
    v0, a0, vg, g2, cr = AA.cruise_inputs(toks)
    n_cr, n_cl = int(cr.sum()), len(set(np.asarray(logs)[cr])) if logs else 0
    floor_ok = n_cr >= 60 and n_cl >= 15
    out = {"tier": TIER, "variance": VARIANCE, "tactical": TACTICAL, "strategic": STRATEGIC, "gates": G, "n_tokens": len(toks), "n_cruise": n_cr, "n_cruise_logs": n_cl,
           "cruise_floor_ok": floor_ok, "prereg_sha256": C.sha256_file(A.PREREG) if A.PREREG.exists() else None, "smoke": a.smoke, "arms": {}}
    g_idt = gains(rec, bt_c, "IDT", csel) if bt_c is not None and len(ct) else None
    out["gain_IDT_u400"] = g_idt
    if g_idt is not None:
        G["G-POS"] = {"down_ci_lo": g_idt["down"]["ci95"][0], "bar": A.POS_CTRL_MIN, "PASS": g_idt["down"]["ci95"][0] > A.POS_CTRL_MIN}
        if not G["G-POS"]["PASS"]:
            common.append("G-POS")
    else:
        common.append("G-POS")
    b_a0 = A.v4(arms["A0"]["poses"]) - vg if "A0" in arms else None
    for t in A.CANDIDATES:
        red = list(common) + ([] if G[f"G-REACH:{t}"]["PASS"] else [f"G-REACH:{t}"]) + ([] if G[f"G-HARNESS:{t}"]["PASS"] else [f"G-HARNESS:{t}"])
        if t == "SMOOTH4" and not G["G-HIST"]["PASS"]:
            red.append("G-HIST")
        if t not in arms or "A0" not in arms or g_idt is None:
            out["arms"][t] = {"decision": {"verdict": "VOID", "why": f"missing inputs; red {red}"}}
            continue
        sT, s0 = arms[t]["sub"]["score"], arms["A0"]["sub"]["score"]
        dd = sT - s0
        pdms = {"point": float(dd.mean()), "ci95": S.ci(bt.stat(dd)), "T_x100": float(sT.mean()), "A0_x100": float(s0.mean())}
        subs = {k: {"point": float((arms[t]["sub"][k] - arms["A0"]["sub"][k]).mean()), "ci95": S.ci(bt.stat(arms[t]["sub"][k] - arms["A0"]["sub"][k]))} for k in AA.SUB}
        g = gains(rec, bt_c, t, csel)
        bT = A.v4(arms[t]["poses"]) - vg
        d_lon = abs(bT[cr].mean()) - abs(b_a0[cr].mean())
        dl_bt = np.abs(bt.stat(bT * cr, cr.astype(float))) - np.abs(bt.stat(b_a0 * cr, cr.astype(float)))

        def lat(num, den, scale=1.0):
            c = S.contrast(bt, [(arms[t]["pw"][num], arms[t]["pw"][den])], [(arms["A0"]["pw"][num], arms["A0"]["pw"][den])])
            return {"point": c["point"] * scale, "ci95": [x * scale for x in c["ci95"]]}
        fam = {"LONGITUDINAL": {"cruise_b4_T": float(bT[cr].mean()), "cruise_b4_A0": float(b_a0[cr].mean()), "D_lon": float(d_lon), "D_lon_ci95": S.ci(dl_bt),
                                "speed_bias_4s_all_T": float(bT.mean()), "speed_mae_4s_all_T": float(np.abs(bT).mean()), "speed_mae_4s_all_A0": float(np.abs(b_a0).mean()),
                                "along_mae_m": lat("along_abs_sum", "along_n"), "along_bias_m": lat("along_signed_sum", "along_n"),
                                "headway_ttc": "n/a: the NAVSIM seam carries no lead-agent headway; PDMS TTC sub-score is the proxy (subscores)"},
               "LATERAL": {"curvature_mae_1pm": lat("curv_abs_sum", "curv_n"), "heading_mae_deg": lat("head_abs_sum_rad", "head_n", 180 / math.pi),
                           "yaw_rate_mae_degps": lat("yaw_abs_sum_radps", "yaw_n", 180 / math.pi), "cross_mae_m": lat("cross_abs_sum", "cross_n")},
               "TACTICAL": TACTICAL, "STRATEGIC": STRATEGIC}
        pick_changed = float(np.mean([not np.array_equal(arms[t]["poses"][i], arms["A0"]["poses"][i]) for i in range(len(toks))]))
        out["arms"][t] = {"pdms_delta_x100": pdms, "subscores_delta_x100": subs, "gain": g, "families": fam, "share_plans_changed_vs_A0": pick_changed,
                          "decision": decide(t, red, pdms, g, g_idt, d_lon, floor_ok)}
    cand = [t for t in A.CANDIDATES if out["arms"][t]["decision"]["verdict"] == "CANDIDATE"]
    out["P1_candidates"] = cand
    an = root / "analysis"
    an.mkdir(exist_ok=True)
    C.jdump(out, an / "P1.json")
    lines = [f"# YAXD P1 (inference-only ax transforms on u400){' -- SMOKE, not decision data' if a.smoke else ''}", "", TIER, "", VARIANCE, "",
             f"n tokens {len(toks)}, CRUISE {n_cr} over {n_cl} logs (floor met: {floor_ok})", f"u400 IDT gain (this run): {json.dumps(g_idt)}", "",
             "| T | verdict | dPDMS x100 [CI] | gain down [CI] | gain sym | D_lon | curvature dMAE |", "|---|---|---|---|---|---|---|"]
    for t in A.CANDIDATES:
        r = out["arms"][t]
        if "pdms_delta_x100" not in r:
            lines.append(f"| {t} | {r['decision']['verdict']} | - | - | - | - | - |")
            continue
        p, gd = r["pdms_delta_x100"], r["gain"]["down"]
        lines.append(f"| {t} | {r['decision']['verdict']} | {p['point']:+.3f} [{p['ci95'][0]:+.3f}, {p['ci95'][1]:+.3f}] | {gd['point']:.3f} [{gd['ci95'][0]:.3f}, {gd['ci95'][1]:.3f}] | "
                     f"{r['gain']['sym']['point']:.3f} | {r['families']['LONGITUDINAL']['D_lon']:+.3f} | {r['families']['LATERAL']['curvature_mae_1pm']['point']:+.5f} |")
    lines += ["", TACTICAL, STRATEGIC, "", "Gates: " + json.dumps({k: v.get("PASS") for k, v in G.items()}), f"P1 CANDIDATES: {cand}"]
    (an / "P1.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print("ZZAXDVERDICT P1 " + " ".join(f"{t}={out['arms'][t]['decision']['verdict'].replace(' ', '_')}" for t in A.CANDIDATES))
    return 0


if __name__ == "__main__":
    sys.exit(main())
