"""lh_bars.py -- PREREG_LONHEAD_C s2-s3 (+ AMENDMENT1): compose LHS_FACT8 / DER_FACT8 EXACTLY as FACTORED_DEPLOY8 (Amendment 5:
P = SF.compose(RT, P_os8, P_sharp8) = eval_polyline(polyline_geometry(P_os8), arc(P_sharp8)) on 8 slots, staged by RS.retime_plan with the registered
per-frame RETIME deploy table, S6 slots [1,3,4,5,6,7]), then bars F1 / F2 / G / A at inference seed 0 and the replicate (F-bar + G) at seed 1, with the
beside rows (compliance, E1 / E3, the four families, LH's per-class confusion on eval).  CPU only.  NEW file (refcv9 LONHEAD phase-2 agent, 2026-10-26).

Reused, imported unedited: hsgate_common.lane() (the battery lane: RT / RS / SF / registered deploy table, sha-verified), l2_common (load_eval_pf,
lon_indicators, arm_metrics, apply_stage), l2_score (load_l1_base, load_trace, join_plan8, metric_rows, compare), pair_read_r8.paired_cell
(taniteval.ci paired episode-cluster bootstrap B 2000, seed 0, alpha 0.05).
LHS_FACT8 is ALSO produced by the full SHIP-R8 Amendment-5 forward (code/sub_fx_lhs, 3-line diff: the trace source) -> fwd_lhs/; this script asserts
the light composer == that forward bit for bit on S6 (K-LHS-FWD), and == the LANDED FACTORED_DEPLOY8 when fed the SELFSHARP traces (K-FX-REBUILD).
DER_FACT8 (seed 0 only) comes from the same light composer.

    python lh_bars.py --validate-only     # the composer controls only (no LHS / DER trace read)
    python lh_bars.py                     # everything -> result/LONHEAD_VERDICT.{json,md}
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in ("C:/Users/Admin/refcv8_pf/refcv9_hsgate/code", "C:/Users/Admin/refcv8_pf/refcv9_l2/code", str(HERE)):
    sys.path.insert(0, _p)
import hsgate_common as HG  # noqa: E402
import k_head as K  # noqa: E402

TREE = "step48852ema_pf"
S6 = [1, 3, 4, 5, 6, 7]
ROLLS = Path("F:/refcv9_lonhead/rolls")
SS_TRACE = "F:/refcv9_selfsharp/trace_selfsharp_s{s}.npz"
FX_PKG = Path("C:/Users/Admin/refcv8_pf/ship_r8_sub_fx/out")
H8_PKG = Path("C:/Users/Admin/refcv8_pf/ship_r8_sub/out")
LANDED = {"FACTORED_DEPLOY8": str(FX_PKG / "arms/factored_scored/FACTORED_DEPLOY8" / f"DEPLOY8_POS_{TREE}_s{{s}}.npz"),
          "BASE_OS": str(FX_PKG / "arms/factored_scored/OSONLY_DEPLOY8" / f"DEPLOY8_POS_{TREE}_s{{s}}.npz"),
          "H2CN_DEPLOY8": str(H8_PKG / "arms/h2cn/H2CN_DEPLOY8" / f"DEPLOY8_POS_{TREE}_s{{s}}.npz"),
          "RETIME_DEPLOY8": str(H8_PKG / "arms/h2cn/RETIME_DEPLOY8" / f"DEPLOY8_POS_{TREE}_s{{s}}.npz")}
FWD_LHS = ROOT / "fwd_lhs" / "out"
LHS_FWD_POS = str(FWD_LHS / "arms/factored_scored/FACTORED_DEPLOY8" / f"DEPLOY8_POS_{TREE}_s{{s}}.npz")
K_NONE = {0: (0.46948, 0.13955), 1: (0.47105, 0.13739)}      # BASE-os|stage (MD, FD), literals (PREREG_SELCOR s4 / PREREG_SSGUARD_S1 s2)
FAMILIES = {"ADE": ("ade_m", "fde_m"),
            "LONGITUDINAL": ("LON_speed_mae_mps", "LON_along_mae_m", "LON_accel_mae_mps2", "LON_along_abs_final_m", "LON_speed_within_0p5"),
            "LATERAL": ("LAT_cross_mae_m", "LAT_heading_mae_deg", "LAT_yaw_rate_mae_radps_valid", "LAT_curvature_mae_1pm_valid", "LAT_cross_abs_final_m"),
            "TACTICAL": ("TAC_traj_lat_correct", "TAC_traj_lon_correct")}
TIER = "T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)"
ESTIMATOR = ("paired episode-cluster bootstrap (pair_read_r8.paired_cell -> taniteval.ci), B 2000, bootstrap seed 0, alpha 0.05, over EPISODES only "
             "(models fixed): answers 'another draw of EPISODES' only -- blind to training variance (ONE refcv8 training seed), to head-fit variance "
             "(ONE LH fit, ONE DER fit) and to inference variance within a seed; inference seeds 0 and 1 are reported separately, never pooled")
NEXT = {"PASS+REPLICATED": "the first lever today that moves BEYOND the frontier: a SHIP-R8 amendment on LHS_FACT8 needs a PI go; refcv9 trains the lon head on a "
                           "clean split as its own component (design input with this number).",
        "PASS+NOT_REPLICATED": "the seed-0 frontier gain does not hold at inference seed 1: report both words; no SHIP-R8 request (prereg s3: REPLICATED is required).",
        "HEAD-NOT-BETTER": "the clean split does not reproduce the probe's headroom: the lon decision needs NEW supervision (richer labels or more fresh clips), "
                           "a data decision for the PI.",
        "FAIL_F": "head accuracy is not the binding constraint for the factored composition: the progress the sharpened class implies is (next: a "
                  "class-conditioned progress head on the fresh split, pre-registered first).",
        "FAIL_A": "the gain is the head's class RATE, not its per-window decisions: report it; no ship.",
        "FAIL_G": "a frontier bar passes but an S6 row is separated WORSE than BASE-os: name the rows; no ship (not a committed s5 branch -- reported as written).",
        "INVALID": "a control failed: fix the instrument, re-score (0 GPU)."}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def log(*a):
    print("[lh_bars]", *a, flush=True)


def sep_below(c):
    return bool(c.get("separated") and c.get("hi") is not None and c["hi"] < 0)


def sep_above(c):
    return bool(c.get("separated") and c.get("lo") is not None and c["lo"] > 0)


def fmt(c):
    return (f"{c['delta']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}]" + (" SEP" if c.get("separated") else "")
            + f" ({c['mean_new']:.4f} vs {c['mean_ref']:.4f}, n={c['n_windows']}/{c['n_episodes']} ep)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate-only", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    env = HG.env_check()
    hashes = K.assert_registered()
    kh = json.loads((ROOT / "result" / "K_HEAD.json").read_text(encoding="utf-8"))
    assert kh["go"] is True and kh["controls"]["ALL_PASS"] is True
    L = HG.lane()
    RT, RS, SF, table = L["RT"], L["RS"], L["SF"], L["table"]
    import l2_common as L2
    import l2_score as S2
    st_ = L2.stack()
    PR, ra = st_["PR"], st_["ra"]
    ctl, per_seed, beside = {}, {}, {}
    lab_rec = json.loads((ROOT / "labels" / "LABELS_INFO.json").read_text(encoding="utf-8"))

    def load_tr(p):
        t = S2.load_trace(p)
        if t is None or t.get("incomplete") or t["n"] != 4754:
            raise SystemExit(f"[lh_bars] {p}: absent / not final / n != 4754")
        return t

    def compose_stage(p_os8, p_sh8, prior8):
        P = SF.compose(RT, np.ascontiguousarray(p_os8).astype(np.float32), np.ascontiguousarray(p_sh8).astype(np.float32))
        pos8 = RS.retime_plan(P, prior8, SF.T8, table, cast=True)
        return P, pos8, np.ascontiguousarray(pos8[:, S6])

    def k_hot(t, lab_file, tag):
        """K-FRESH-SS's assertion on the eval roll: applied lon logits one-hot +-20 on the label, every window; hook counters."""
        lz = np.load(lab_file)
        lab = {(int(c), int(w)): int(v) for c, w, v in zip(lz["clip_index"], lz["ws"], lz["lon_v7"])}
        z = np.load(t["_path"])
        keys = list(zip(z["e_i"].tolist(), z["ws"].tolist()))
        labs = np.array([lab.get(k, -100) for k in keys])
        L6 = np.array(HG.LON6_V7_IDS)
        all_lab = bool(np.isin(labs, L6).all())
        exp = np.full((len(keys), 8), -20.0, np.float32)
        if all_lab:
            exp[np.arange(len(keys)), labs] = 20.0
        app_ok = bool(all_lab and np.array_equal(np.asarray(z["applied_lon_logits"], np.float32), exp))
        c = t["meta"].get("counters", {})
        cnt_ok = bool(c.get("calls") == 4754 and c.get("hook_applied") == 4754 and c.get("hook_calls_unarmed") == 0 and c.get("batch_gt1") == 0
                      and c.get("multi_hook_in_call") == 0 and c.get("oracle_labelled") == 4754)
        lab_sha_meta = t["meta"].get("labels_sha256")
        return {"tag": tag, "n": len(keys), "keys_unique": len(set(keys)) == len(keys), "all_windows_labelled_lon6": all_lab,
                "applied_onehot_pm20_on_label_every_window": app_ok, "counters": {k: c.get(k) for k in ("calls", "hook_applied", "hook_calls_unarmed", "batch_gt1",
                                                                                                         "multi_hook_in_call", "oracle_labelled", "oracle_unlabelled_or_masked")},
                "counters_ok": cnt_ok, "labels_npz_in_meta": t["meta"].get("labels_npz"), "labels_sha256_in_meta": lab_sha_meta,
                "trace_sha256": sha(t["_path"]), "PASS": bool(app_ok and cnt_ok and len(set(keys)) == len(keys))}

    def cells(E, P6):
        comp = np.asarray(E["compliant"], bool)
        c = L2.arm_metrics(E, P6)
        fd, md, _gt, _pc = L2.lon_indicators(P6, E, comp)
        d = {k: np.where(comp, v, np.nan) for k, v in c.items()}
        d["false_decel"], d["missed_decel"] = fd, md
        return d

    rows = S2.metric_rows(PR)
    gkeys = [k for (_f, k, _l, _u, _lw, _s) in PR.METRICS]
    arms6 = {}
    for s in (0, 1):
        E = L2.load_eval_pf(s)
        comp = np.asarray(E["compliant"], bool)
        eid = list(E["eid"])
        Z = {k: np.load(v.format(s=s), allow_pickle=False) for k, v in LANDED.items()}
        al = {k: bool(np.array_equal(z["S6_ws"], E["ws"]) and np.array_equal(z["S6_clip"], E["clip"])) for k, z in Z.items()}
        prior8 = Z["FACTORED_DEPLOY8"]["S6_prior8"]
        pri = {k: bool(np.array_equal(z["S6_prior8"], prior8)) for k, z in Z.items()}
        pri["E_prior8"] = bool(np.array_equal(np.asarray(E["prior8"]), prior8))
        b1 = S2.load_l1_base(str(HG.EVAL_L1_BANK), s)
        p_os8 = S2.join_plan8(b1, E)
        # K-FX-REBUILD: the light composer on the SELFSHARP trace == the LANDED FACTORED_DEPLOY8 (plan8 / pos8 / S6 pos), bit for bit
        tss = load_tr(SS_TRACE.format(s=s))
        Pfx, pos8fx, pos6fx = compose_stage(p_os8, S2.join_plan8(tss, E), prior8)
        kfx = {"plan8": bool(np.array_equal(Pfx, Z["FACTORED_DEPLOY8"]["S6_plan8"])), "pos8": bool(np.array_equal(pos8fx, Z["FACTORED_DEPLOY8"]["S6_pos8"])),
               "S6_pos": bool(np.array_equal(pos6fx, Z["FACTORED_DEPLOY8"]["S6_pos"])), "selfsharp_trace_sha256": sha(SS_TRACE.format(s=s))}
        # K-OS: the os-only plan staged == the LANDED OSONLY_DEPLOY8 (= BASE-os|stage), via the lane AND via l2_common.apply_stage
        pos_os = np.ascontiguousarray(RS.retime_plan(np.ascontiguousarray(p_os8).astype(np.float32), prior8, SF.T8, table, cast=True)[:, S6])
        kos = {"lane": bool(np.array_equal(pos_os, Z["BASE_OS"]["S6_pos"])), "l2_apply_stage": bool(np.array_equal(L2.apply_stage(p_os8, E["prior8"]), Z["BASE_OS"]["S6_pos"]))}
        # K-MUT (discriminating): the composer fed the os plan as its own progress source must give BASE-os exactly AND differ from FACTORED on >0 windows
        _P, _p8, pos_ssos = compose_stage(p_os8, p_os8, prior8)
        kos["ssos_identity_equals_BASE_OS"] = bool(np.array_equal(pos_ssos, Z["BASE_OS"]["S6_pos"]))
        kos["ssos_differs_from_FACTORED_windows"] = int((pos_ssos != Z["FACTORED_DEPLOY8"]["S6_pos"]).any((1, 2)).sum())
        kos["mutation_detected"] = bool(kos["ssos_differs_from_FACTORED_windows"] > 0)
        ctl[f"s{s}"] = {"K-ALIGN": al, "K-PRIOR8": pri, "K-FX-REBUILD": kfx, "K-OS": kos,
                        "landed_sha256": {k: sha(v.format(s=s)) for k, v in LANDED.items()}}
        log(f"s{s} K-ALIGN {al} K-PRIOR8 {pri} K-FX-REBUILD {kfx} K-OS {kos}")
        if a.validate_only:
            continue
        A = {"FACTORED_DEPLOY8": Z["FACTORED_DEPLOY8"]["S6_pos"], "H2CN_DEPLOY8": Z["H2CN_DEPLOY8"]["S6_pos"], "BASE_OS": Z["BASE_OS"]["S6_pos"],
             "RETIME_DEPLOY8": Z["RETIME_DEPLOY8"]["S6_pos"]}
        tl = load_tr(ROLLS / f"trace_LHS_s{s}.npz"); tl["_path"] = ROLLS / f"trace_LHS_s{s}.npz"
        _P, _p8, A["LHS_FACT8"] = compose_stage(p_os8, S2.join_plan8(tl, E), prior8)
        ctl[f"s{s}"]["K-HOT_LHS"] = k_hot(tl, ROOT / "labels" / "labels_LHS.npz", f"LHS_s{s}")
        fw = np.load(LHS_FWD_POS.format(s=s), allow_pickle=False)
        ctl[f"s{s}"]["K-LHS-FWD"] = {"S6_pos_bit_equal_full_forward": bool(np.array_equal(fw["S6_pos"], A["LHS_FACT8"])),
                                     "S6_ws_clip_aligned": bool(np.array_equal(fw["S6_ws"], E["ws"]) and np.array_equal(fw["S6_clip"], E["clip"])),
                                     "forward_npz_sha256": sha(LHS_FWD_POS.format(s=s))}
        ctl[f"s{s}"]["sharp_vs_os_windows_differ"] = int((np.abs(S2.join_plan8(tl, E).astype(np.float64) - p_os8.astype(np.float64)).max(axis=(1, 2)) > 0).sum())
        if s == 0:
            td = load_tr(ROLLS / "trace_DER_s0.npz"); td["_path"] = ROLLS / "trace_DER_s0.npz"
            _P, _p8, A["DER_FACT8"] = compose_stage(p_os8, S2.join_plan8(td, E), prior8)
            ctl["s0"]["K-HOT_DER"] = k_hot(td, ROOT / "labels" / "labels_DER.npz", "DER_s0")
        arms6[s] = A
        Cc = {k: cells(E, v) for k, v in A.items()}
        r5 = lambda c: (round(float(np.nanmean(c["missed_decel"])), 5), round(float(np.nanmean(c["false_decel"])), 5))  # noqa: E731
        ctl[f"s{s}"]["K-NONE"] = {"got": r5(Cc["BASE_OS"]), "want": K_NONE[s], "pass": r5(Cc["BASE_OS"]) == K_NONE[s]}
        zc = PR.paired_cell(Cc["FACTORED_DEPLOY8"]["false_decel"], Cc["FACTORED_DEPLOY8"]["false_decel"], eid)
        ctl[f"s{s}"]["K-ZERO"] = bool(zc["delta"] == 0.0 and zc["lo"] == 0.0 and zc["hi"] == 0.0)
        pc = lambda x, y: {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in PR.paired_cell(x, y, eid).items()  # noqa: E731
                           if k in ("delta", "lo", "hi", "separated", "n_windows", "n_episodes", "mean_new", "mean_ref", "n_dropped_nonfinite", "status")}
        rates = {k: {"MD": float(np.nanmean(c["missed_decel"])), "FD": float(np.nanmean(c["false_decel"])),
                     "n_decel": int(np.isfinite(c["missed_decel"]).sum()), "n_cruise": int(np.isfinite(c["false_decel"]).sum())} for k, c in Cc.items()}
        cl = {"F1_FD_LHS_vs_FX": pc(Cc["LHS_FACT8"]["false_decel"], Cc["FACTORED_DEPLOY8"]["false_decel"]),
              "F1_MD_LHS_vs_FX": pc(Cc["LHS_FACT8"]["missed_decel"], Cc["FACTORED_DEPLOY8"]["missed_decel"]),
              "F2_MD_LHS_vs_H8": pc(Cc["LHS_FACT8"]["missed_decel"], Cc["H2CN_DEPLOY8"]["missed_decel"]),
              "F2_FD_LHS_vs_H8": pc(Cc["LHS_FACT8"]["false_decel"], Cc["H2CN_DEPLOY8"]["false_decel"])}
        F1 = bool(sep_below(cl["F1_FD_LHS_vs_FX"]) and not sep_above(cl["F1_MD_LHS_vs_FX"]))
        F2 = bool(sep_below(cl["F2_MD_LHS_vs_H8"]) and not sep_above(cl["F2_FD_LHS_vs_H8"]))
        cmpG = S2.compare(PR, {"eid": eid}, Cc["LHS_FACT8"], Cc["BASE_OS"], rows)
        worse = sorted(k for k in gkeys + ["ade_m"] if k in cmpG and cmpG[k].get("worse") is True)
        G = not worse
        ps = {"rates_S6_compliant": rates, "cells": cl, "F1": F1, "F2": F2, "G": G, "G_rows_separated_worse": worse, "G_table": cmpG}
        if s == 0:
            cmpGd = S2.compare(PR, {"eid": eid}, Cc["DER_FACT8"], Cc["BASE_OS"], rows)
            ps["beside_DER_G_rows_separated_worse"] = sorted(k for k in gkeys + ["ade_m"] if k in cmpGd and cmpGd[k].get("worse") is True)
            cl["A1_FD_LHS_vs_DER"] = pc(Cc["LHS_FACT8"]["false_decel"], Cc["DER_FACT8"]["false_decel"])
            cl["A2_MD_LHS_vs_DER"] = pc(Cc["LHS_FACT8"]["missed_decel"], Cc["DER_FACT8"]["missed_decel"])
            cl["beside_DER_F1_FD_vs_FX"] = pc(Cc["DER_FACT8"]["false_decel"], Cc["FACTORED_DEPLOY8"]["false_decel"])
            cl["beside_DER_F1_MD_vs_FX"] = pc(Cc["DER_FACT8"]["missed_decel"], Cc["FACTORED_DEPLOY8"]["missed_decel"])
            cl["beside_DER_F2_MD_vs_H8"] = pc(Cc["DER_FACT8"]["missed_decel"], Cc["H2CN_DEPLOY8"]["missed_decel"])
            cl["beside_DER_F2_FD_vs_H8"] = pc(Cc["DER_FACT8"]["false_decel"], Cc["H2CN_DEPLOY8"]["false_decel"])
            ps["A1"] = sep_below(cl["A1_FD_LHS_vs_DER"])
            ps["A2"] = sep_below(cl["A2_MD_LHS_vs_DER"])
        # four families (beside), LHS_FACT8 vs FACTORED / H2CN / BASE-os, S6 compliant
        fam = {}
        for fname, keys in FAMILIES.items():
            fam[fname] = {key: {f"vs_{ref}": pc(Cc["LHS_FACT8"][key], Cc[ref][key]) for ref in ("FACTORED_DEPLOY8", "H2CN_DEPLOY8", "BASE_OS")}
                          for key in keys if key in Cc["LHS_FACT8"]}
        fam["TACTICAL"]["false_decel_FD"] = {f"vs_{ref}": pc(Cc["LHS_FACT8"]["false_decel"], Cc[ref]["false_decel"]) for ref in ("FACTORED_DEPLOY8", "H2CN_DEPLOY8", "BASE_OS")}
        fam["TACTICAL"]["missed_decel_MD"] = {f"vs_{ref}": pc(Cc["LHS_FACT8"]["missed_decel"], Cc[ref]["missed_decel"]) for ref in ("FACTORED_DEPLOY8", "H2CN_DEPLOY8", "BASE_OS")}
        fam["LONGITUDINAL"]["distance_keeping_headway_TTC"] = "n/a: no lead-agent track in the banked refcv8 eval dumps this battery reads; n = 0"
        fam["STRATEGIC"] = ("n/a: eval139 carries no strategic decision / route-goal label this battery reads (refcv8 runs --no-strategic; SHIP-R8 clause 6 N/A); "
                            "LH does not touch the route input; n = 0")
        ps["families"] = fam
        per_seed[str(s)] = ps
        log(f"s{s} rates {json.dumps({k: (round(v['MD'], 4), round(v['FD'], 4)) for k, v in rates.items()})}")
        log(f"s{s} F1 {F1} F2 {F2} G {G} worse {worse} | " + " | ".join(f"{k}: {fmt(v)}" for k, v in cl.items()))
    if a.validate_only:
        ok = all(all(c["K-ALIGN"].values()) and all(c["K-PRIOR8"].values()) and all(v for k, v in c["K-FX-REBUILD"].items() if k != "selfsharp_trace_sha256")
                 and all(c["K-OS"].values()) for c in ctl.values())
        out = {"tool": "lh_bars.py --validate-only", "controls": ctl, "ALL_PASS": bool(ok), "env": env, "lane": L["rec"], "wall_s": round(time.time() - t0, 1)}
        K_path = ROOT / "result" / "COMPOSER_VALIDATION.json"
        K_path.write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
        log("VALIDATE", "ALL_PASS" if ok else "FAIL")
        return 0 if ok else 2
    # ---- LH per-class confusion on eval (T0, diagnostic) ----
    beside["LH_confusion_eval_T0"] = confusion(L2)
    # ---- compliance row, E1 / E3 (from the full SHIP-R8 forward + verdict on LHS_FACT8; LANDED FACTORED / H2CN beside) ----
    beside.update(ship_r8_beside())
    # ---- controls roll-up ----
    def cpass(c, s):
        ok = (all(c["K-ALIGN"].values()) and all(c["K-PRIOR8"].values()) and all(v for k, v in c["K-FX-REBUILD"].items() if k != "selfsharp_trace_sha256")
              and all(c["K-OS"].values()) and c["K-HOT_LHS"]["PASS"] and c["K-LHS-FWD"]["S6_pos_bit_equal_full_forward"] and c["K-LHS-FWD"]["S6_ws_clip_aligned"]
              and c["K-NONE"]["pass"] and c["K-ZERO"])
        if s == 0:
            ok = ok and c["K-HOT_DER"]["PASS"]
        return bool(ok)
    ctl_pass = bool(cpass(ctl["s0"], 0) and cpass(ctl["s1"], 1) and beside.get("forward_ALL_PASS") is True)
    p0, p1 = per_seed["0"], per_seed["1"]
    a_ok = bool((p0["F1"] and p0["A1"]) or (p0["F2"] and p0["A2"]))
    PASS = bool((p0["F1"] or p0["F2"]) and p0["G"] and a_ok)
    rep_bars = [b for b in ("F1", "F2") if p0[b] and p0["A" + b[1]]]
    REPL = bool(PASS and p1["G"] and any(p1[b] for b in rep_bars))
    if not ctl_pass:
        word, nxt = "INVALID", NEXT["INVALID"]
    elif PASS and REPL:
        word, nxt = "PASS+REPLICATED", NEXT["PASS+REPLICATED"]
    elif PASS:
        word, nxt = "PASS+NOT_REPLICATED", NEXT["PASS+NOT_REPLICATED"]
    elif not (p0["F1"] or p0["F2"]):
        word, nxt = "FAIL_F (head better, F1 and F2 FAIL)", NEXT["FAIL_F"]
    elif not a_ok:
        word, nxt = "FAIL_A", NEXT["FAIL_A"]
    else:
        word, nxt = "FAIL_G", NEXT["FAIL_G"]
    res = {"tool": "lh_bars.py", "script_sha256": sha(__file__), "prereg": {"PREREG_LONHEAD_C.md": hashes["PREREG_LONHEAD_C.md"],
                                                                          "PREREG_LONHEAD_C_AMENDMENT1.md": hashes["PREREG_LONHEAD_C_AMENDMENT1.md"]},
           "registered_sha256": hashes, "tier": TIER, "estimator": ESTIMATOR,
           "word": word, "PASS_s0": PASS, "REPLICATED_s1": REPL, "A_s0": a_ok, "committed_next_step_prereg_s5": nxt,
           "K_HEAD": {"LH_auc": kh["LH_point_auc"], "ci95": kh["result"]["LH"]["ci95"], "reference": 0.685, "native_reproduced": kh["controls"]["K-REF_0.685"]["native_head_renorm_auc"],
                      "DER_auc": kh["result"]["DER"]["auc"], "LH_minus_native": kh["result"]["LH_minus_native"], "n_union": kh["population"]["union_n"]},
           "labels": lab_rec, "controls": ctl, "controls_ALL_PASS": ctl_pass, "per_seed": per_seed, "beside": beside,
           "bar_definitions": {"F1": "FD(LHS_FACT8) - FD(FACTORED_DEPLOY8) separated BELOW 0 and MD(LHS_FACT8) - MD(FACTORED_DEPLOY8) NOT separated above 0",
                               "F2": "MD(LHS_FACT8) - MD(H2CN_DEPLOY8) separated BELOW 0 and FD(LHS_FACT8) - FD(H2CN_DEPLOY8) NOT separated above 0",
                               "G": "no S6 row of PR.METRICS + ade_m separated WORSE than BASE-os (= OSONLY_DEPLOY8, the os-only plan staged)",
                               "A": "on the F-bar that passes: LHS_FACT8 - DER_FACT8 separated in the same direction (FD below for F1, MD below for F2), seed 0",
                               "PASS": "(F1 or F2) and G and A at s0", "REPLICATED": "the same F-bar (one that passed with A at s0) and G hold at s1"},
           "families_not_computed": {"LONGITUDINAL distance-keeping (headway / TTC)": "no lead-agent track in the banked refcv8 eval dumps; n = 0",
                                     "STRATEGIC": "refcv8 runs --no-strategic; no route-goal label on eval139; n = 0",
                                     "DER_FACT8 compliance / E1 / E3": "not computed: the SHIP-R8 forward reads BOTH inference seeds and DER was rolled at s0 only (prereg s2)"},
           "env": env, "lane": L["rec"], "generated": time.strftime("%FT%T%z"), "wall_s": round(time.time() - t0, 1)}
    (ROOT / "result").mkdir(exist_ok=True)
    tmp = ROOT / "result" / "LONHEAD_VERDICT.json.tmp"
    tmp.write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    os.replace(tmp, ROOT / "result" / "LONHEAD_VERDICT.json")
    write_md(res)
    log("WORD", word, "| controls", ctl_pass)
    return 0 if ctl_pass else 2


def confusion(L2) -> dict:
    """LH (and the native head, beside) argmax vs the battery's GT lon label (extras lon_v7) on S6 windows, inference seed 0. T0 diagnostic."""
    E = L2.load_eval_pf(0)
    comp = np.asarray(E["compliant"], bool)
    zb = K.load_bank(K.L1_BANK / "ztac_eval_s0.npz")
    keys = list(zip(E["ex_clip"][E["rows"]].tolist(), E["ex_ws"][E["rows"]].tolist()))
    jz = np.array([zb["idx"][k] for k in keys])
    LH = K.LF.load_frozen(ROOT / "LH_FROZEN.pt", K.REG["LH_FROZEN.pt"])
    L6 = np.asarray(K.LF.LON6_V7, np.int64)
    pred = L6[K.LF.predict_logits6(LH, zb["z"][jz], np.asarray(zb["v0_in"], np.float64)[jz]).argmax(1)]
    nat = L6[L2.lon6(E["logits"]).argmax(1)]
    gt = np.asarray(E["lon_v7"]).astype(np.int64)
    out = {"tier": "T0 diagnostic (head read on logged frames; no planner)", "classes_v7": L6.tolist(), "names": list(K.LF.KL.LON7[i] for i in L6)}
    for nm, m in (("S6_compliant", comp), ("S6_all", np.ones(len(comp), bool))):
        mm = m & np.isin(gt, L6)
        g, p, q = gt[mm], pred[mm], nat[mm]
        cm = np.zeros((6, 6), np.int64); cn = np.zeros((6, 6), np.int64)
        for i, a_ in enumerate(L6):
            for j, b_ in enumerate(L6):
                cm[i, j] = int(((g == a_) & (p == b_)).sum()); cn[i, j] = int(((g == a_) & (q == b_)).sum())
        maj = int(L6[np.argmax([(g == c).sum() for c in L6])])
        out[nm] = {"n_windows_gt_in_lon6": int(mm.sum()), "n_windows_gt_outside_lon6": int((m & ~np.isin(gt, L6)).sum()),
                   "LH_confusion_rows_gt_cols_pred": cm.tolist(), "native_confusion_rows_gt_cols_pred": cn.tolist(),
                   "LH_acc": float((p == g).mean()), "native_acc": float((q == g).mean()), "constant_majority_acc": float((g == maj).mean()), "majority_class_v7": maj,
                   "LH_recall_per_class": {int(c): (float((p[g == c] == c).mean()) if (g == c).any() else None) for c in L6},
                   "native_recall_per_class": {int(c): (float((q[g == c] == c).mean()) if (g == c).any() else None) for c in L6},
                   "LH_pred_counts": {int(c): int((p == c).sum()) for c in L6}, "gt_counts": {int(c): int((g == c).sum()) for c in L6}}
    return out


def ship_r8_beside() -> dict:
    out = {}
    fwd_p = FWD_LHS / "forward" / f"FORWARD_FACTORED_SCORED_{TREE}.json"
    ver_p = FWD_LHS / "ship_r8" / "SHIP_R8_FACTORED_DEPLOY8.json"
    fwd = json.loads(fwd_p.read_text(encoding="utf-8"))
    ver = json.loads(ver_p.read_text(encoding="utf-8"))
    fx_ver = json.loads((FX_PKG / "ship_r8" / "SHIP_R8_FACTORED_DEPLOY8.json").read_text(encoding="utf-8"))
    fx_fwd = json.loads((FX_PKG / "forward" / f"FORWARD_FACTORED_SCORED_{TREE}.json").read_text(encoding="utf-8"))
    h8_ver_p = H8_PKG / "ship_r8" / "SHIP_R8_H2CN_DEPLOY8.json"
    h8_ver = json.loads(h8_ver_p.read_text(encoding="utf-8")) if h8_ver_p.exists() else {}
    out["forward_ALL_PASS"] = (fwd.get("controls") or {}).get("ALL_PASS")
    out["forward_file"] = {"path": str(fwd_p), "sha256": sha(fwd_p)}
    out["ship_r8_LHS_FACT8"] = {"path": str(ver_p), "sha256": sha(ver_p), "overall": ver.get("overall"),
                                "note": "SHIP-R8 Amendment-5 clauses read on LHS_FACT8 (the machinery's arm name is FACTORED_DEPLOY8 inside fwd_lhs/)"}

    def e13(v):
        cl = {str(c["id"]): c for c in v.get("clauses", [])}
        return {"E1_clause2_S2_ADE_minus_ha0_ext_pose": (cl.get("2", {}).get("evidence") or {}).get("cells"),
                "E3_clause3_S6_ADE_minus_ha0_ext_pose": (cl.get("3", {}).get("evidence") or {}).get("cells"),
                "clause_verdicts": {k: c.get("pass", c.get("verdict")) for k, c in cl.items()}, "overall": v.get("overall")}
    out["E1_E3"] = {"LHS_FACT8": e13(ver), "FACTORED_DEPLOY8_landed": e13(fx_ver), "H2CN_DEPLOY8_landed": e13(h8_ver) if h8_ver else None}

    def comprow(f, arm):
        r = {}
        for k_, v_ in ((f.get("compliance_row") or {}).get("cells") or {}).items():
            a_ = (v_.get("arms") or {}).get(arm) or {}
            r[k_] = (a_.get("full_plan_0_6s") or {}).get("compliant_set")
        return r
    out["compliance_row"] = {"LHS_FACT8": comprow(fwd, "FACTORED_DEPLOY8"), "FACTORED_DEPLOY8_landed": comprow(fx_fwd, "FACTORED_DEPLOY8"),
                             "OSONLY_DEPLOY8": comprow(fwd, "OSONLY_DEPLOY8"), "RETIME_DEPLOY8": comprow(fwd, "RETIME_DEPLOY8"),
                             "rule": (fwd.get("compliance_row") or {}).get("rule")}
    return out


def write_md(res: dict) -> None:
    p0, p1 = res["per_seed"]["0"], res["per_seed"]["1"]
    L = [f"# LONHEAD_C phase 2 -- {res['word']}", "",
         f"**Tier:** {res['tier']}.  ", f"**Estimator:** {res['estimator']}.  ",
         f"**Prereg:** PREREG_LONHEAD_C.md {res['prereg']['PREREG_LONHEAD_C.md'][:8]} + AMENDMENT1 {res['prereg']['PREREG_LONHEAD_C_AMENDMENT1.md'][:8]} "
         "(FIT rows = the 292 fresh clips with obstacle labels); LH = logistic lambda 100 (C 0.01, the sealed grid's strongest regularisation, grid edge, disclosed).  ",
         f"**Committed outcome (prereg s5):** {res['committed_next_step_prereg_s5']}", "",
         f"## K-HEAD (T0, go/no-go): LH decel AUC {res['K_HEAD']['LH_auc']} {res['K_HEAD']['ci95']} vs reference 0.685 (native reproduced "
         f"{res['K_HEAD']['native_reproduced']}); DER {res['K_HEAD']['DER_auc']}; LH - native {res['K_HEAD']['LH_minus_native']} (episode-resampling bootstrap, "
         f"n {res['K_HEAD']['n_union']} union windows) -> GO", "",
         f"## Bars (S6 compliant)  PASS_s0 = {res['PASS_s0']}  A_s0 = {res['A_s0']}  REPLICATED_s1 = {res['REPLICATED_s1']}  controls_ALL_PASS = {res['controls_ALL_PASS']}", ""]
    for s, p in (("0", p0), ("1", p1)):
        L.append(f"### inference seed {s}: F1 {p['F1']} | F2 {p['F2']} | G {p['G']} (worse rows: {p['G_rows_separated_worse'] or 'none'})"
                 + (f" | A1 {p.get('A1')} | A2 {p.get('A2')}" if s == "0" else ""))
        L.append("")
        L.append("| arm | MD | FD | n decel | n cruise |")
        L.append("|---|---|---|---|---|")
        for k, v in p["rates_S6_compliant"].items():
            L.append(f"| {k} | {v['MD']:.4f} | {v['FD']:.4f} | {v['n_decel']} | {v['n_cruise']} |")
        L.append("")
        for k, v in p["cells"].items():
            L.append(f"* {k}: {fmt(v)}")
        L.append("")
    b = res["beside"]
    L.append("## Beside rows")
    L.append(f"* SHIP-R8 Amendment-5 machinery on LHS_FACT8: overall {b['ship_r8_LHS_FACT8']['overall']} (forward ALL_PASS {b['forward_ALL_PASS']})")
    for arm, e in b["E1_E3"].items():
        if not e:
            continue
        for nm in ("E1_clause2_S2_ADE_minus_ha0_ext_pose", "E3_clause3_S6_ADE_minus_ha0_ext_pose"):
            cells_ = e.get(nm) or {}
            L.append(f"* {arm} {nm.split('_')[0]}: " + " | ".join(f"{s}: {fmt(x)}" for s, x in cells_.items() if isinstance(x, dict) and "delta" in x))
    for arm, r in b["compliance_row"].items():
        if arm == "rule" or not r:
            continue
        L.append(f"* compliance ({arm}): " + " | ".join(f"{k}: " + json.dumps({kk: (vv.get('mean') if isinstance(vv, dict) else vv)
                                                                              for kk, vv in (v or {}).items() if kk.startswith('exceed_rate_strict') or kk.startswith('n_exceed_strict')})
                                                     for k, v in r.items()))
    cf = b["LH_confusion_eval_T0"]["S6_compliant"]
    L.append(f"* LH confusion on eval (T0, S6 compliant, n {cf['n_windows_gt_in_lon6']}): LH acc {cf['LH_acc']:.4f} vs native {cf['native_acc']:.4f} vs constant "
             f"{cf['constant_majority_acc']:.4f}; LH recall {json.dumps({k: (round(v, 3) if v is not None else None) for k, v in cf['LH_recall_per_class'].items()})}; "
             f"native recall {json.dumps({k: (round(v, 3) if v is not None else None) for k, v in cf['native_recall_per_class'].items()})}")
    L.append("* four families: per_seed.<s>.families in LONHEAD_VERDICT.json (ADE / LONGITUDINAL / LATERAL / TACTICAL vs FACTORED_DEPLOY8, H2CN_DEPLOY8, BASE-os); "
             "STRATEGIC and distance-keeping n/a with reasons (families_not_computed)")
    L.append("")
    L.append("ZZLHVERDICTENDZZ")
    (ROOT / "result" / "LONHEAD_VERDICT.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
