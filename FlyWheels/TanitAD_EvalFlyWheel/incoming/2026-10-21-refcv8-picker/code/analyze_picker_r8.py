"""analyze_picker_r8.py -- PREREG R8-PICKER (sha256 e6ec30e7..., Addendum 1 994bd849...): the MECHANICAL read -> PICKER_R8.json + RESULT.md.

CPU only.  It edits no instrument.  Every statistic is the battery's own: pair_read_r8.components / paired_cell (taniteval.ci paired episode-cluster
bootstrap, B 2000, seed 0, 95 % percentile), compliant_reread_r8.prepare_seed / plan_variants / pmax (the landed cap, the fed ceiling, the compliant
mask), clamp_posthoc_r8.cluster_rate.  The archive root of those modules is re-pointed to the picker archive (a PATH, as retime_trainfit_r8 does).

    python analyze_picker_r8.py k9            # control K9: the identity tree's lane verdict == the banked SHIP_R8_step48852ema_pf.json
    python analyze_picker_r8.py full          # everything (needs runs/R0, runs/R1, the three picker trees and their lanes)

STAMPS: T1 self-action OPEN LOOP on logged eval139 frames (not driving performance); ONE training seed (refcv8-r101-s0); inference seeds 0 and 1 are the
replicate; the CI answers ONLY 'another draw of EPISODES'.  Picker - SHIP cells are within ONE DDIM draw per seed (all arms share the capture's fan).
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

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
P = "C:/Users/Admin/refcv8_pf/picker"
ARCH = f"{P}/archive"
LANE = f"{P}/lane"
RUNS = os.environ.get("PK_RUNS", f"{P}/runs")
PKGW = "C:/Users/Admin/qland/hold_r8batt/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-15-refcv8-milestone-battery/code"
HOLDW = "C:/Users/Admin/qland/hold_r8pf/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-20-refcv8-perframe-final"
BANK_TREE = "C:/Users/Admin/refcv8_eval_archive/battery/step48852ema_pf"
BANK_VERDICT = f"{HOLDW}/raw/ship_r8/SHIP_R8_step48852ema_pf.json"
BANK_VERDICT_SHA256 = "893b69a8cd8da312b89c31206db2bf637232468d89711e8bd00e68e34cc48c39"
FANCEIL_S1 = "C:/Users/Admin/refcv8_pf/fanceil/runs/s1/dump/fan_extras.npz"
PREREG = {"PREREG_R8_PICKER.md": "e6ec30e7bc03b3dc4811674352f94c9f128203cbf814f284876f239ae867b258",
          "PREREG_R8_PICKER_ADDENDUM1.md": "994bd84910dd920deec948bdc2217b6948e8729121cf0eea3becf8329de1ba89",
          "PREREG_R8_PICKER_ADDENDUM2.md": "a4982749af9b2adadc69ff276989c62dbc8f28bf2572efafd6bdab399e517b9f"}
TREES = json.loads(os.environ["PK_TREES"]) if os.environ.get("PK_TREES") else {"SHIP": "pkship_pf", "BASE": "pkbase_pf", "MBR32": "pkmbr32_pf"}
OUT_PREFIX = os.environ.get("PK_OUT_PREFIX", "")          # DRY-RUN ONLY: a non-empty prefix never writes PICKER_R8.json / RESULT.md
SEEDS = (0, 1)
SURFS = ("S2", "S6")
N_BOOT = 2000
E1_TOL, E3_TOL = 0.005, 0.02
TIER = "T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)"
QUESTION = ("paired episode-cluster bootstrap (taniteval.ci via pair_read_r8.paired_cell), B 2000, seed 0, 95 % percentile; models and RETIME gains FIXED: "
            "answers ONLY 'another draw of EPISODES'; ONE training seed; inference seeds 0 and 1 are the inference replicate; picker - SHIP is within one DDIM draw")


def sha256_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def md5_file(p) -> str:
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def jload(p):
    return json.load(open(p, encoding="utf-8"))


def tools():
    if PKGW not in sys.path:
        sys.path.insert(0, PKGW)
    import clamp_posthoc_r8 as CP
    import compliant_reread_r8 as CR
    CP.ARCHIVE = CR.ARCHIVE = ARCH
    T = CR.tools_load()
    return CP, CR, T


# ------------------------------------------------------------------------------------------ K9 (identity lane == banked verdict)
def _clause_cells(v: dict, cid: str) -> dict:
    c = next(c for c in v["clauses"] if str(c["id"]) == cid)
    return {s: {k: x.get(k) for k in ("delta", "lo", "hi", "separated", "n_windows")} for s, x in (c.get("evidence", {}).get("cells") or {}).items()}


def k9() -> dict:
    mine_p = f"{LANE}/pkident_pf/ship_r8/SHIP_R8_pkident_pf.json"
    out = {"bank_verdict": BANK_VERDICT, "bank_verdict_sha256_ok": sha256_file(BANK_VERDICT) == BANK_VERDICT_SHA256, "identity_verdict": mine_p}
    if not os.path.exists(mine_p):
        return {**out, "PASS": False, "status": "the identity lane verdict does not exist"}
    b, m = jload(BANK_VERDICT), jload(mine_p)
    out["per_clause_bank"], out["per_clause_identity"] = b["per_clause"], m["per_clause"]
    out["overall_bank"], out["overall_identity"] = b["overall"], m["overall"]
    out["cells_equal"] = {}
    for cid in ("2", "3"):
        cb, cm = _clause_cells(b, cid), _clause_cells(m, cid)
        out["cells_equal"][cid] = {"equal": cb == cm, "bank": cb, "identity": cm}
    pb = jload(f"{BANK_TREE}/PAIRED_READ_step48852ema_pf.json")
    pm = jload(f"{ARCH}/pkident_pf/PAIRED_READ_pkident_pf.json")
    neq = []
    for surf in SURFS:
        for s in ("0", "1"):
            rb = pb["surfaces"][surf]["seeds"][s].get("rows", {})
            rm = pm["surfaces"][surf]["seeds"][s].get("rows", {})
            for key in rb:
                for cell in rb[key]:
                    if isinstance(rb[key][cell], dict) and "delta" in rb[key][cell]:
                        a_, b_ = rb[key][cell], (rm.get(key) or {}).get(cell) or {}
                        if any(a_.get(f) != b_.get(f) for f in ("delta", "lo", "hi", "n_windows")):
                            neq.append(f"{surf}|s{s}|{key}|{cell}")
    out["paired_read_regenerated_equals_bank"] = {"n_cells_differing": len(neq), "first": neq[:10]}
    out["PASS"] = bool(out["bank_verdict_sha256_ok"] and b["per_clause"] == m["per_clause"] and b["overall"] == m["overall"]
                       and all(v["equal"] for v in out["cells_equal"].values()) and not neq)
    return out


# ------------------------------------------------------------------------------------------ helpers
def align(src_c, src_w, dst_c, dst_w) -> np.ndarray:
    k = {(int(c), int(w)): i for i, (c, w) in enumerate(zip(src_c, src_w))}
    return np.asarray([k[(int(c), int(w))] for c, w in zip(dst_c, dst_w)], np.int64)


def sep_worse(cell: dict, lower_is_better: bool) -> bool | None:
    if not isinstance(cell, dict) or cell.get("lo") is None:
        return None
    return bool(cell["lo"] > 0) if lower_is_better else bool(cell["hi"] < 0)


def slim(c: dict) -> dict:
    return {k: c.get(k) for k in ("delta", "lo", "hi", "separated", "n_windows", "n_episodes", "mean_new", "mean_ref", "n_dropped_nonfinite", "status")}


def capture_paths(seed: int) -> dict:
    d = f"{RUNS}/R{seed}"
    return {"dir": d, "fan": f"{d}/dump/fan_extras.npz", "side": f"{d}/dump/refcv6_extras.npz", "meta": f"{d}/fan_roll_s{seed}.json",
            "roll": f"{d}/roll_s{seed}.json", "md5_thor": f"{d}/MD5_R{seed}.txt"}


# ------------------------------------------------------------------------------------------ controls on the captures
def controls_captures() -> dict:
    import controls_pk as K
    res = {}
    for s in SEEDS:
        cp = capture_paths(s)
        r = {"paths": cp}
        # K11: pulled md5 (Thor list) == dev md5
        try:
            lines = [ln.split() for ln in open(cp["md5_thor"], encoding="utf-8").read().splitlines() if ln.strip()]
            chk = {f.lstrip("*"): {"thor": m, "dev": md5_file(f"{cp['dir']}/{f.lstrip('*')}")} for m, f in lines}   # '*' = md5sum binary-mode marker
            r["K11"] = {"files": chk, "PASS": bool(lines and all(len(v["thor"]) == 32 and v["thor"] == v["dev"] for v in chk.values()))}
        except Exception as e:  # noqa: BLE001
            r["K11"] = {"PASS": False, "error": repr(e)}
        z = np.load(cp["fan"])
        side = np.load(cp["side"])
        bankx = np.load(f"{BANK_TREE}/dump_s{s}/refcv6_extras.npz")
        if not (np.array_equal(side["clip_index"], z["clip_index"]) and np.array_equal(side["ws"], z["ws"])):
            raise SystemExit(f"[pk] R{s}: capture rows != the roll's sidecar rows")
        ib = align(bankx["clip_index"], bankx["ws"], z["clip_index"], z["ws"])
        r["capture"] = K.capture_controls(z, side["nav_true.traj"], bankx["v_max_ms"][ib], bankx["v_max_valid"][ib])
        sel = np.asarray(z["sel_idx"]).astype(np.int64).reshape(-1)
        bsel = bankx["nav_true.sel_idx"].astype(np.int64)[ib]
        r["K7_vs_bank"] = {"sel_agree_frac": float((sel == bsel).mean()), "n_disagree": int((sel != bsel).sum()), "bar": 0.99,
                           "PASS": bool((sel == bsel).mean() >= 0.99), "blocking": False,
                           "plan_vs_bank_median_m_on_agreeing": float(np.median(np.linalg.norm(np.asarray(z["traj"], np.float64)[sel == bsel]
                                                                                              - bankx["nav_true.traj"].astype(np.float64)[ib][sel == bsel], axis=-1).max(-1)))}
        if s == 1:
            f1 = np.load(FANCEIL_S1)
            i1 = align(f1["clip_index"], f1["ws"], z["clip_index"], z["ws"])
            s1 = np.asarray(f1["sel_idx"]).astype(np.int64).reshape(-1)[i1]
            ag = s1 == sel
            dtr = np.abs(np.asarray(f1["traj"], np.float64)[i1][ag] - np.asarray(z["traj"], np.float64)[ag]).max() if ag.any() else None
            r["K2_vs_fanceil_thor_s1"] = {"fanceil_fan": FANCEIL_S1, "fanceil_fan_sha256": sha256_file(FANCEIL_S1), "sel_agree_frac": float(ag.mean()),
                                          "n_disagree": int((~ag).sum()), "max_abs_traj_on_agreeing_m": (None if dtr is None else float(dtr)),
                                          "bar": 0.99, "PASS": bool(ag.mean() >= 0.99)}
        r["meta"] = {k: v for k, v in jload(cp["meta"]).items() if k in ("n_windows", "conds_seen", "n_calls_without_nav_true", "keys_missing_counts", "wall_s",
                                                                        "cuda_max_memory_allocated_gib", "torch", "host", "this_script_sha256", "refc_v3_sha256",
                                                                        "os_only", "child_argv", "K3a_max_abs_fan_at_sel_minus_traj_m", "K3b_traj_equals_sidecar_nav_true",
                                                                        "K4_rolltime_e9_argmax_equals_sel_idx_frac")}
        rr = jload(cp["roll"])
        r["roll_pf_ok"] = (rr.get("pf") or {}).get("ok")
        r["form"] = "FULL ARM" if s == 0 else "--os-only"
        r["form_ok"] = bool(r["meta"].get("os_only") == (s == 1))
        res[str(s)] = r
    return res


# ------------------------------------------------------------------------------------------ R4 / R5 per seed
def deploy8(tree: str, seed: int) -> dict:
    z = np.load(f"{LANE}/{tree}/pfpkg/raw/retime_deploy/DEPLOY8_POS_{tree}_s{seed}.npz")
    return {k: z[k] for k in z.files if k != "meta_json"}


def family_reads(CP, CR, T) -> dict:
    PR, V, ra = T["PR"], T["V"], T["ra"]
    rc = CP.run_cfg()
    METRICS = PR.METRICS
    out = {"metrics": [{"family": f, "key": k, "label": lab, "unit": u, "lower_is_better": lo} for f, k, lab, u, lo, _s in METRICS], "seeds": {}}
    ship_floor = {}
    for s in SEEDS:
        ctx = {nm: CR.prepare_seed(tree, s, T, rc) for nm, tree in TREES.items()}
        pv = {nm: CR.plan_variants(V, ctx[nm], rc) for nm in TREES}
        d8 = {nm: deploy8(tree, s) for nm, tree in TREES.items()}
        c0 = ctx["SHIP"]
        for nm in TREES:                                                      # every tree carries the bank's windows / mask / ceiling
            assert np.array_equal(ctx[nm]["clip"], c0["clip"]) and np.array_equal(ctx[nm]["ws"], c0["ws"]) and np.array_equal(ctx[nm]["comp"], c0["comp"])
        sd = {"surfaces": {}, "compliance": {}, "K12_gt_identical": {}}
        for surf in SURFS:
            Pn = {nm: ctx[nm]["panels"][surf] for nm in TREES}
            P0 = Pn["SHIP"]
            rows, slots, dt, eid = P0["rows"], P0["slots"], P0["dt"], np.asarray(P0["eid"])
            g = P0["A"]["g"]
            bank_g = []
            import glob as _g
            pdir = f"{BANK_TREE}/{'panel_s%d' % s if surf == 'S2' else 's6_s%d' % s}"
            for f in sorted(_g.glob(os.path.join(pdir, "ep*.npz"))):
                bank_g.append(np.load(f)["g"])
            bank_g = np.concatenate(bank_g)
            sd["K12_gt_identical"][surf] = bool(all(np.array_equal(Pn[nm]["A"]["g"], g) and np.array_equal(Pn[nm]["rows"], rows) for nm in TREES)
                                                and np.array_equal(bank_g, g))
            kc, kw = c0["clip"][rows], c0["ws"][rows]
            comp = c0["comp"][rows]
            plans = {}
            for nm in TREES:
                ia = align(d8[nm][f"{surf}_clip"], d8[nm][f"{surf}_ws"], kc, kw)
                if not np.array_equal(d8[nm][f"{surf}_compliant"][ia], comp):
                    raise SystemExit(f"[pk] {nm} s{s} {surf}: DEPLOY8 compliant mask != the compliant re-read mask")
                if not np.array_equal(d8[nm][f"{surf}_os"][ia], pv[nm]["uncapped"][rows][:, slots]):
                    raise SystemExit(f"[pk] {nm} s{s} {surf}: DEPLOY8 os != the tree's emitted plan at the slots")
                plans[f"{nm}+RT" if nm != "SHIP" else "SHIP"] = d8[nm][f"{surf}_pos"][ia].astype(np.float32)
                plans[f"{nm}_raw_capped"] = pv[nm]["capped"][rows][:, slots]
                plans[f"{nm}_raw_uncapped"] = pv[nm]["uncapped"][rows][:, slots]
            comps = {k: PR.components(ra, v, g, dt) for k, v in plans.items()}
            arms = ("BASE+RT", "MBR32+RT", "BASE_raw_capped", "MBR32_raw_capped", "SHIP_raw_capped", "BASE_raw_uncapped", "MBR32_raw_uncapped", "SHIP")
            srec = {"n_windows": int(len(rows)), "n_compliant": int(comp.sum()), "n_episodes": int(len(np.unique(eid))), "grid_slots": slots,
                    "means_compliant": {a: {k: float(np.nanmean(comps[a][k][comp])) for f, k, *_r in METRICS} for a in arms}, "cells": {}}
            for a in arms:
                srec["cells"][a] = {}
                for setn, m in (("compliant", comp), ("all", np.ones(len(comp), bool))):
                    srec["cells"][a][setn] = {k: slim(PR.paired_cell(comps[a][k][m], comps["SHIP"][k][m], eid[m], N_BOOT)) for f, k, *_r in METRICS}
            sd["surfaces"][surf] = srec
            if surf == "S2":
                ship_floor[s] = float(np.nanmean(comps["SHIP"]["ade_m"][comp]))
        # R5-i COMPLIANCE on the full emitted 8-slot plan, dump order, all 4,754 windows (compliant set headline)
        lim, v0 = c0["lim"], c0["v0"]
        eid_d = c0["clip"]
        comp_d = c0["comp"]
        emitted = {}
        for nm in TREES:
            ia = align(d8[nm]["S2_clip"], d8[nm]["S2_ws"], c0["clip"], c0["ws"])
            emitted[f"{nm}+RT" if nm != "SHIP" else "SHIP"] = d8[nm]["S2_pos8"][ia].astype(np.float32)
            emitted[f"{nm}_raw_uncapped"] = pv[nm]["uncapped"]
            emitted[f"{nm}_raw_capped"] = pv[nm]["capped"]
        fin = np.isfinite(lim)
        exc = {k: (fin & (CR.pmax(V, v, rc["horizons"], rc["tick_s"]) > lim)).astype(np.float64) for k, v in emitted.items()}
        crec = {"n_windows": int(len(lim)), "n_compliant": int(comp_d.sum()), "rates": {}, "cells_vs_SHIP": {}}
        for k, x in exc.items():
            crec["rates"][k] = {"compliant": CP.cluster_rate(x[comp_d], eid_d[comp_d]), "all": CP.cluster_rate(x, eid_d)}
            if k != "SHIP":
                crec["cells_vs_SHIP"][k] = {"compliant": slim(PR.paired_cell(x[comp_d], exc["SHIP"][comp_d], eid_d[comp_d], N_BOOT)),
                                            "all": slim(PR.paired_cell(x, exc["SHIP"], eid_d, N_BOOT))}
        sd["compliance"] = crec
        out["seeds"][str(s)] = sd
        del ctx, pv, d8
    out["ship_inference_floor_S2_ade_compliant"] = {"s0": ship_floor.get(0), "s1": ship_floor.get(1),
                                                    "floor_m": (abs(ship_floor[0] - ship_floor[1]) if len(ship_floor) == 2 else None)}
    # SHIP - SHIP must read exactly 0 (an analytic control on the machinery)
    z = [c for s in out["seeds"].values() for surf in s["surfaces"].values() for c in surf["cells"]["SHIP"]["compliant"].values()]
    out["K13_ship_minus_ship_exactly_zero"] = bool(z and all(c.get("delta") == 0 and c.get("lo") == 0 and c.get("hi") == 0 for c in z if c.get("delta") is not None))
    return out


# ------------------------------------------------------------------------------------------ verdict reads
def verdicts() -> dict:
    out = {}
    for nm, tree in TREES.items():
        p = f"{LANE}/{tree}/ship_r8/SHIP_R8_{tree}.json"
        if not os.path.exists(p):
            out[nm] = {"status": "ABSENT", "file": p}
            continue
        v = jload(p)
        out[nm] = {"file": p, "sha256": sha256_file(p), "overall": v["overall"], "per_clause": v["per_clause"], "label": v.get("label"),
                   "E1": _clause_cells(v, "2"), "E3": _clause_cells(v, "3"),
                   "reasons": {str(c["id"]): c["reason"][:300] for c in v["clauses"]}}
        cr = f"{LANE}/{tree}/pfpkg/raw/compliant_reread/trees/{tree}.json"
        if os.path.exists(cr):
            b = (jload(cr).get("bars") or {}).get("compliant", {}).get("capped", {})
            out[nm]["raw_bars"] = {"BAR-R8-E1": b.get("BAR-R8-E1"), "BAR-R8-E3": b.get("BAR-R8-E3"),
                                   "BAR-R8-NR_per_family": (b.get("BAR-R8-NR") or {}).get("per_family")}
    return out


# ------------------------------------------------------------------------------------------ the decision rule (section 7)
def decide(fam: dict, ver: dict, ctl: dict, k8: dict, k9r: dict, k1: dict) -> dict:
    METRICS = fam["metrics"]
    blocking = {"K1_smoke": bool(k1.get("ALL_PASS")), "K8_identity_splice": bool(k8.get("K8_PASS")), "K9_lane_reproduces_bank": bool(k9r.get("PASS")),
                "K13_ship_minus_ship_zero": bool(fam.get("K13_ship_minus_ship_exactly_zero"))}
    for s, c in ctl.items():
        blocking[f"K11_md5_R{s}"] = bool(c["K11"]["PASS"])
        blocking[f"K3_R{s}"] = bool(c["capture"]["K3"]["PASS"])
        blocking[f"K4_R{s}"] = bool(c["capture"]["K4"]["PASS"])
        blocking[f"K5_R{s}"] = bool(c["capture"]["K5"]["PASS"])
        blocking[f"K6b_R{s}"] = c["capture"]["K6b"]["PASS"] is not False
        blocking[f"K6c_R{s}"] = bool(c["capture"]["K6c"]["PASS"])
        blocking[f"roll_pf_ok_R{s}"] = bool(c["roll_pf_ok"])
        blocking[f"roll_form_R{s}"] = bool(c["form_ok"])
        if "K2_vs_fanceil_thor_s1" in c:
            blocking["K2_R1_vs_fanceil_thor"] = bool(c["K2_vs_fanceil_thor_s1"]["PASS"])
        for surf, ok in fam["seeds"][s]["K12_gt_identical"].items():
            blocking[f"K12_gt_{surf}_s{s}"] = bool(ok)
    ship = ver.get("SHIP", {})
    k10 = {"overall": ship.get("overall"), "PASS_blocking": ship.get("overall") == "PASS", "drift": {}}
    bank = jload(BANK_VERDICT)
    for cid, nmE, tol in (("2", "E1", E1_TOL), ("3", "E3", E3_TOL)):
        cb = _clause_cells(bank, cid)
        for sk, c in (ship.get(nmE) or {}).items():
            d = None if (c.get("delta") is None or cb.get(sk, {}).get("delta") is None) else abs(c["delta"] - cb[sk]["delta"])
            k10["drift"][f"{nmE}|{sk}"] = {"ship_tree": c.get("delta"), "bank": cb.get(sk, {}).get("delta"), "abs_diff": d, "tol": tol,
                                          "within": (None if d is None else bool(d <= tol))}
    blocking["K10_ship_verdict_PASS"] = bool(k10["PASS_blocking"])
    controls_ok = all(blocking.values())
    floor = fam["ship_inference_floor_S2_ade_compliant"]["floor_m"]
    arms = {}
    for arm, tree_nm, rt in (("BASE+RT", "BASE", True), ("MBR32+RT", "MBR32", True), ("BASE", "BASE", False), ("MBR32", "MBR32", False)):
        key = arm if rt else f"{tree_nm}_raw_capped"
        worse, worse_both = [], []
        for f in METRICS:
            for surf in SURFS:
                flags = {s: sep_worse(fam["seeds"][s]["surfaces"][surf]["cells"][key]["compliant"][f["key"]], f["lower_is_better"]) for s in ("0", "1")}
                if any(flags.values()):
                    worse.append(f"{f['family']}:{surf}:{f['key']}@" + ",".join(f"s{s}" for s, v in flags.items() if v))
                if all(flags.values()):
                    worse_both.append(f"{f['family']}:{surf}:{f['key']}")
        ckey = arm if rt else f"{tree_nm}_raw_uncapped"
        cflags = {s: sep_worse(fam["seeds"][s]["compliance"]["cells_vs_SHIP"][ckey]["compliant"], True) for s in ("0", "1")}
        if any(cflags.values()):
            worse.append("LONGITUDINAL:FULL8:COMPLIANCE@" + ",".join(f"s{s}" for s, v in cflags.items() if v))
        if all(cflags.values()):
            worse_both.append("LONGITUDINAL:FULL8:COMPLIANCE")
        a3c = {s: fam["seeds"][s]["surfaces"]["S2"]["cells"][key]["compliant"]["ade_m"] for s in ("0", "1")}
        a3_sep = all(c.get("hi") is not None and c["hi"] < 0 for c in a3c.values())
        a3_floor = (floor is not None and all(abs(c["delta"]) > 2.0 * floor for c in a3c.values()))
        rec = {"A2_separated_worse_cells": worse, "A2_PASS": not worse, "A2_beside_worse_at_both_seeds": worse_both,
               "A3_cells": {s: slim(c) for s, c in a3c.items()}, "A3_separated_below_0_both": a3_sep, "A3_beyond_2x_floor": a3_floor,
               "A3_floor_m": floor, "A3_PASS": bool(a3_sep and a3_floor)}
        if rt:
            v = ver.get(tree_nm, {})
            rec["A1_overall"] = v.get("overall")
            rec["A1_PASS"] = v.get("overall") == "PASS"
            if not controls_ok:
                word = "INCOMPLETE"
            elif not rec["A1_PASS"]:
                bad = [k for k, x in (v.get("per_clause") or {}).items() if x != "PASS" and x != "N/A"]
                word = f"FAIL (A1: SHIP-R8 {v.get('overall')}; clauses {bad})"
            elif not rec["A2_PASS"]:
                word = f"FAIL (A2: separated worse than SHIP: {worse[0]}{' +%d more' % (len(worse) - 1) if len(worse) > 1 else ''})"
            elif not a3_sep:
                word = "FAIL (A3: S2 ADE vs SHIP not separated below 0 at both seeds)"
            elif not a3_floor:
                word = "NOT PROVEN (A3: separated but |delta| within 2x the SHIP inference floor)"
            else:
                word = "ADOPTABLE"
            rec["verdict"] = word
        else:
            rb = ver.get(tree_nm, {}).get("raw_bars") or {}
            e1 = str((rb.get("BAR-R8-E1") or {}).get("verdict", ""))
            e3 = str((rb.get("BAR-R8-E3") or {}).get("verdict", ""))
            nr = rb.get("BAR-R8-NR_per_family") or {}
            rec["raw_bars"] = {"E1": e1, "E3": e3, "NR_per_family": nr}
            raw_ok = e1.startswith("PASS") and e3.startswith("PASS") and nr and all(x == "PASS" for x in nr.values())
            rec["verdict"] = ("INCOMPLETE" if not controls_ok else
                              ("RAW-ELIGIBLE (a PI item: it would drop the shipped RETIME stage, D-R8-SHIP-RETIME)" if (raw_ok and rec["A2_PASS"] and rec["A3_PASS"])
                               else f"NOT RAW-ELIGIBLE (E1 {e1[:14]}, E3 {e3[:14]}, NR {nr}, A2 {rec['A2_PASS']}, A3 {rec['A3_PASS']})"))
        arms[arm] = rec
    adopt = [a for a in ("MBR32+RT", "BASE+RT") if arms[a]["verdict"] == "ADOPTABLE"]
    if "MBR32+RT" in adopt:
        rec_ = "MBR32+RT (keeps the E9 graft; prereg section 7 preference fixed in advance)" + ("; BASE+RT also ADOPTABLE -> PI item (hierarchy)" if "BASE+RT" in adopt else "")
    elif "BASE+RT" in adopt:
        rec_ = "BASE+RT ADOPTABLE -> its adoption is a PI decision WHATEVER (removes the E9 goal-distance graft + drivable critic: hierarchy thesis)"
    else:
        rec_ = "NONE ADOPTABLE"
    return {"blocking_controls": blocking, "controls_ALL_PASS": controls_ok, "K10": k10, "arms": arms, "adoptable": adopt, "recommendation": rec_}


def next_levers(dec: dict, picks: dict, fam: dict) -> list:
    """Section 8 triggers, evaluated mechanically."""
    out = []
    m = dec["arms"]["MBR32+RT"]
    outside = max(picks.get(s, {}).get("pick_outside_e9_keep_frac", {}).get("mbr32", 0.0) for s in ("0", "1"))
    comp_fail = any("COMPLIANCE" in x for x in m["A2_separated_worse_cells"])
    if outside > 0 and comp_fail and m["verdict"] != "ADOPTABLE":
        out.append("L1 TRIGGERED: MBR32 picks outside e9_keep on %.4f of windows AND MBR32+RT fails A2 on COMPLIANCE -> MBR32m (risk argmin restricted to e9_keep), "
                   "sealed as an ADDENDUM before it is computed (CPU, same capture)" % outside)
    for raw, rt in (("BASE", "BASE+RT"), ("MBR32", "MBR32+RT")):
        if dec["arms"][raw]["A3_separated_below_0_both"] and not dec["arms"][rt]["A3_separated_below_0_both"]:
            out.append(f"L2 TRIGGERED ({rt}): the raw S2 gain vs SHIP is separated but the +RT arm's is not -> the frozen RETIME table absorbs the picker's "
                       "longitudinal gain; next = RETIME re-fit on PICKER train rolls (H-R8-RETIME-TRAINFIT-1). BLOCKED on Thor train rolls + a PI decision (new registered table)")
    for rt in ("BASE+RT", "MBR32+RT"):
        a = dec["arms"][rt]
        bad4 = any(k in ("4", "5") and v == "FAIL" for k, v in ((dec.get("_per_clause") or {}).get(rt, {}) or {}).items())
        lat_tac = any(x.split(":")[0] in ("LATERAL", "TACTICAL") for x in a["A2_separated_worse_cells"])
        if lat_tac or bad4:
            out.append(f"L3 TRIGGERED ({rt}): a LATERAL / TACTICAL cell is separated worse (or clause 4/5 failed) -> refcv9 selector item (rank the extras). "
                       "BLOCKED on training compute + a spec")
    if not out:
        out.append("no section-8 trigger fired" + (" (an arm is ADOPTABLE)" if dec["adoptable"] else ""))
    return out


# ------------------------------------------------------------------------------------------ RESULT.md
def f4(c, k="delta"):
    if not isinstance(c, dict) or c.get(k) is None:
        return "n/a"
    return f"{c['delta']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}]" + (" sep" if c.get("separated") else "")


def render(R: dict) -> str:
    dec, fam, ver, ctl = R["decision"], R["families"], R["verdicts"], R["controls"]
    L = ["# R8-PICKER: refcv8 EMA-48,852 plus a GT-free picker, against the shipped refcv8 + RETIME", "",
         f"* **Pre-registration.** `PREREG_R8_PICKER.md`, sha256 `{R['prereg']['PREREG_R8_PICKER.md']['sha256'][:16]}…`, plus Addendum 1 "
         f"`{R['prereg']['PREREG_R8_PICKER_ADDENDUM1.md']['sha256'][:16]}…` and Addendum 2 `{R['prereg']['PREREG_R8_PICKER_ADDENDUM2.md']['sha256'][:16]}…` "
         f"(the K1 rule). All three seals were verified at read time: **{R['prereg_ok']}**.",
         "* **How this file was written.** Mechanically, by `code/analyze_picker_r8.py`, at " + R["generated"] + ".",
         f"* **Tier.** {TIER}.",
         "* **Evidence class.** MEASURED.",
         f"* **Estimator.** {QUESTION}.",
         "* **Rolls.** R0 = seed 0 **FULL ARM**, which reproduces the banked s0 draw. R1 = seed 1 `--os-only`. Both ran on Thor.",
         "", "## Verdict", "",
         f"**Recommendation: {dec['recommendation']}**", "",
         f"Every blocking control passed: **{dec['controls_ALL_PASS']}**.", "",
         "| arm | verdict | A1: SHIP-R8 | A2: no family cell separated worse vs SHIP | A3: S2 ADE − SHIP, s0 / s1 |", "|---|---|---|---|---|"]
    for a in ("MBR32+RT", "BASE+RT", "MBR32", "BASE"):
        r = dec["arms"][a]
        L.append(f"| {a} | **{r['verdict']}** | {r.get('A1_overall', 'n/a (raw arm)')} | {r['A2_PASS']} ({len(r['A2_separated_worse_cells'])} cells) | "
                 f"{f4(r['A3_cells']['0'])} / {f4(r['A3_cells']['1'])} |")
    L += ["", f"The SHIP inference floor (|S2 ADE s0 − s1| on compliant windows) is {fam['ship_inference_floor_S2_ade_compliant']['floor_m']:.4f} m. "
          "BASE's adoption is a PI decision whatever its verdict, because it removes the E9 goal-distance graft and the drivable critic.", "",
          "**Next levers (prereg section 8, evaluated mechanically):**", ""] + [f"* {x}" for x in R["next_levers"]]
    L += ["", "## SHIP-R8 (AMENDMENT 3, registered table 2aacc9ff…) per tree", "", "| tree | overall | per clause | E1 s0 / s1 | E3 s0 / s1 |", "|---|---|---|---|---|"]
    for nm in TREES:
        v = ver.get(nm, {})
        L.append(f"| {nm} ({TREES[nm]}) | **{v.get('overall')}** | {v.get('per_clause')} | {f4((v.get('E1') or {}).get('s0'))} / {f4((v.get('E1') or {}).get('s1'))} | "
                 f"{f4((v.get('E3') or {}).get('s0'))} / {f4((v.get('E3') or {}).get('s1'))} |")
    L += ["", "**Raw-plan bars.** These are capped, on compliant windows, minus `ha0_ext_pose`; NR is vs refcv7-50,400.", "", "| raw arm | E1 | E3 | NR per family |", "|---|---|---|---|"]
    for nm in TREES:
        rb = ver.get(nm, {}).get("raw_bars") or {}
        e1 = rb.get("BAR-R8-E1") or {}
        e3 = rb.get("BAR-R8-E3") or {}
        L.append(f"| {nm}_raw | {e1.get('verdict')} ({', '.join(f's{k} {x.get('delta')}' for k, x in (e1.get('per_inference_seed') or {}).items())}) | "
                 f"{e3.get('verdict')} ({', '.join(f's{k} {x.get('delta')}' for k, x in (e3.get('per_inference_seed') or {}).items())}) | {rb.get('BAR-R8-NR_per_family')} |")
    L += ["", "## The four metric families, arm − SHIP (paired, compliant windows)", "",
          "Each cell is delta [lo, hi]. `sep` marks a separated interval; ⚠ marks a cell that is separated WORSE. "
          "STRATEGIC is **NOT APPLICABLE, n = 0**: the run's argv carries `--no-strategic`, so the route head is untrained and "
          "unscored, and this battery has no route or goal readout.", ""]
    for surf in SURFS:
        L += [f"### {surf} (compliant n = {fam['seeds']['0']['surfaces'][surf]['n_compliant']} windows)", "",
              "| family | metric | " + " | ".join(f"{a} s{s}" for a in ("MBR32+RT", "BASE+RT", "MBR32_raw_capped", "BASE_raw_capped") for s in ("0", "1")) + " |",
              "|---|---|" + "---|" * 8]
        for m in fam["metrics"]:
            row = []
            for a in ("MBR32+RT", "BASE+RT", "MBR32_raw_capped", "BASE_raw_capped"):
                for s in ("0", "1"):
                    c = fam["seeds"][s]["surfaces"][surf]["cells"][a]["compliant"][m["key"]]
                    w = sep_worse(c, m["lower_is_better"])
                    row.append(("⚠ " if w else "") + f4(c))
            L.append(f"| {m['family']} | {m['key']} | " + " | ".join(row) + " |")
        L.append("")
    L += ["## Speed cap", "", "**R5-i COMPLIANCE.** A window counts as exceeding when the full 8-slot emitted plan is above the fed ceiling (strict). "
          "Rates are on compliant windows; arm − SHIP is paired.", "", "| arm | s0 rate | s1 rate | − SHIP s0 | − SHIP s1 |", "|---|---|---|---|---|"]
    for k in fam["seeds"]["0"]["compliance"]["rates"]:
        r0, r1 = fam["seeds"]["0"]["compliance"]["rates"][k]["compliant"], fam["seeds"]["1"]["compliance"]["rates"][k]["compliant"]
        c0 = (fam["seeds"]["0"]["compliance"]["cells_vs_SHIP"].get(k) or {}).get("compliant")
        c1 = (fam["seeds"]["1"]["compliance"]["cells_vs_SHIP"].get(k) or {}).get("compliant")
        L.append(f"| {k} | {r0['mean']:.4f} [{r0['lo']:.4f}, {r0['hi']:.4f}] | {r1['mean']:.4f} [{r1['lo']:.4f}, {r1['hi']:.4f}] | {f4(c0)} | {f4(c1)} |")
    L += ["", "**R5-ii.** The fraction of windows whose pick lies outside the model's own `e9_keep`: " +
          "; ".join(f"s{s}: " + ", ".join(f"{k} {v:.4f}" for k, v in R["picks"][s]["pick_outside_e9_keep_frac"].items()) for s in R["picks"]), "",
          "## Controls (prereg section 6 and Addendum 1)", "", "| control | value | pass |", "|---|---|---|"]
    for k, v in dec["blocking_controls"].items():
        L.append(f"| {k} | | {v} |")
    for s, c in ctl.items():
        cap = c["capture"]
        L.append(f"| R{s} detail | K3a {cap['K3']['K3a_max_abs_m']}, K4 disagreements {cap['K4']['n_disagree']} (ties {cap['K4']['n_ties']}), "
                 f"K5 disagreements {cap['K5']['n_disagree']} (ties {cap['K5']['n_ties']}), K6a n {cap['K6a']['n_windows_v_max_valid_0']}, "
                 f"K6b n_nonbinding {cap['K6b']['n_nonbinding_windows']}, K7 vs bank {c['K7_vs_bank']['sel_agree_frac']:.4f}"
                 + (f", K2 vs fanceil-Thor {c['K2_vs_fanceil_thor_s1']['sel_agree_frac']:.4f} / max traj {c['K2_vs_fanceil_thor_s1']['max_abs_traj_on_agreeing_m']}" if "K2_vs_fanceil_thor_s1" in c else "")
                 + " | |")
    L.append(f"| K10 drift vs the banked verdict (not blocking) | {dec['K10']['drift']} | |")
    L += ["", "## Artifacts", "", "* `PICKER_R8.json` holds every number in this file.",
          "* The trees are `archive/pk{ship,base,mbr32}_pf`, and their lanes are `lane/<tree>/` (pfpkg, ship_r8, feed_parity).",
          "* The captures are `runs/R{0,1}/` (Thor originals: `/home/nvidia/r8ns/picker/out/R{0,1}`).",
          "* The picks are `archive/PICKS_s{0,1}.npz`.", ""]
    return "\n".join(L)


def cmd_full(a) -> int:
    t0 = time.time()
    pre = {f: {"sha256": sha256_file(f"{P}/{f}"), "expected": h} for f, h in PREREG.items()}
    pre_ok = all(v["sha256"] == v["expected"] for v in pre.values())
    CP, CR, T = tools()
    k1 = jload(f"{P}/K1_SMOKE.json") if os.path.exists(f"{P}/K1_SMOKE.json") else {}
    k8 = jload(f"{P}/K8_IDENTITY_SPLICE.json") if os.path.exists(f"{P}/K8_IDENTITY_SPLICE.json") else {}
    k9r = k9()
    ctl = controls_captures()
    build = jload(os.environ.get("PK_BUILD_JSON", f"{P}/BUILD_PICKER_TREES.json"))
    picks = {s: {k: v for k, v in r.items() if k != "fan"} for s, r in build["picks"].items()}
    fam = family_reads(CP, CR, T)
    ver = verdicts()
    dec = decide(fam, ver, ctl, k8, k9r, k1)
    dec["_per_clause"] = {"BASE+RT": ver.get("BASE", {}).get("per_clause"), "MBR32+RT": ver.get("MBR32", {}).get("per_clause")}
    R = {"tool": "analyze_picker_r8.py", "generated": time.strftime("%F %T %z"), "prereg": pre, "prereg_ok": pre_ok, "tier": TIER, "estimator": QUESTION,
         "controls": ctl, "K1_smoke": {k: k1.get(k) for k in ("ALL_PASS", "generated")}, "K8": {k: k8.get(k) for k in ("K8_PASS", "K8_identity", "K8_mutation")},
         "K9": k9r, "picks": picks, "families": fam, "verdicts": ver, "decision": dec, "build": {k: build.get(k) for k in ("generated", "frozen_cell", "script_sha256")},
         "script_sha256": sha256_file(__file__)}
    R["next_levers"] = next_levers(dec, picks, fam)
    if not pre_ok:
        R["decision"]["recommendation"] = "INCOMPLETE: the pre-registration seal does not verify"
    json.dump(R, open(f"{P}/{OUT_PREFIX}PICKER_R8.json", "w", encoding="utf-8"), indent=1, default=str)
    Path(f"{P}/{OUT_PREFIX}RESULT.md").write_text(render(R), encoding="utf-8")
    chk = jload(f"{P}/{OUT_PREFIX}PICKER_R8.json")                                        # assert on the ARTIFACT
    assert chk["decision"]["arms"]["MBR32+RT"]["verdict"] == dec["arms"]["MBR32+RT"]["verdict"]
    print("ZZPKANALYSISDONEZZ", {k: v["verdict"] for k, v in dec["arms"].items()}, "controls", dec["controls_ALL_PASS"], "wall", round(time.time() - t0, 1), flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("k9", "full"))
    a = ap.parse_args()
    if a.cmd == "k9":
        r = k9()
        json.dump(r, open(f"{P}/K9_LANE_IDENTITY.json", "w", encoding="utf-8"), indent=1, default=str)
        print("K9", r.get("PASS"), r.get("per_clause_identity"), r.get("paired_read_regenerated_equals_bank"), "ZZPKK9" + ("PASS" if r.get("PASS") else "FAIL") + "ZZ")
        return 0 if r.get("PASS") else 4
    return cmd_full(a)


if __name__ == "__main__":
    import pk_peak
    pk_peak.arm("analyze_picker_r8 " + " ".join(sys.argv[1:]))
    sys.exit(main())
