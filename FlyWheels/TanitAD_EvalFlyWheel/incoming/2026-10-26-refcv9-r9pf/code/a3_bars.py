"""a3_bars.py -- PREREG_REFCV9_INMODEL_PA PHASE A3: bars PA-1a / PA-1b / PA-2 / PA-G / PA-R, controls K-CONSTZ / K-SHUF (+ the K-EQ / K-OFF record of
a3_keq_roll.py and the geometry record of a3_compose.py), the four families, the compliance row and the beside rows -> result/R9PF_PA_VERDICT.{json,md}.
CPU only.  Packages in (F:/refcv9_r9pf_scratch/arms), json + md out.

Estimator: pair_read_r8.paired_cell (taniteval.ci paired episode-cluster bootstrap, B 2000, bootstrap seed 0, windows finite on BOTH sides), over EPISODES
with the models fixed -- blind to training and inference variance; fit seeds 0/1/2 are the training-variance read, inference seeds 0/1 reported apart.
Populations: S6 compliant (MD on GT-decel, FD on GT-cruise: l2_common.lon_indicators exactly as mdfix_bars.py) and S2 compliant; tier T1 open loop.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r9pf_common as RC  # noqa: E402

import numpy as np  # noqa: E402

TREE = "step48852ema_pf"
LANDED = Path("C:/Users/Admin/refcv8_pf/refcv9_hsgate/out_mdfix/arms/h2cn_mdfix")
LANDED_MAN = Path("C:/Users/Admin/refcv8_pf/refcv9_hsgate/MANIFEST_MDFIX.sha256")
BESIDE = {"H8": "H2CN_DEPLOY8", "FX": "FACTORED_DEPLOY8", "D8": "RETIME_DEPLOY8"}
PKG = RC.SCRATCH / "arms"
TIER = "T1 self-action OPEN LOOP on logged eval139 frames, not driving performance"
EST = ("paired episode-cluster bootstrap over EPISODES, B 2000 seed 0 — blind to training and inference variance; fit seeds 0/1/2 = "
       "training-variance read. (pair_read_r8.paired_cell -> taniteval.ci, models fixed; inference seeds 0 / 1 reported separately, never pooled; "
       "ONE refcv8 trunk seed)")


def cell(PR, a, b, eid) -> dict:
    c = PR.paired_cell(np.asarray(a, np.float64), np.asarray(b, np.float64), list(eid))
    keep = ("delta", "lo", "hi", "separated", "n_windows", "n_episodes", "mean_new", "mean_ref", "n_dropped_nonfinite", "status")
    return {k: (float(c[k]) if isinstance(c.get(k), (float, np.floating)) else c.get(k)) for k in keep if k in c}


def sep_below(c) -> bool:
    return bool(c.get("separated") and c.get("hi") is not None and c["hi"] < 0)


def sep_above(c) -> bool:
    return bool(c.get("separated") and c.get("lo") is not None and c["lo"] > 0)


def fmt(c) -> str:
    if not c or c.get("delta") is None:
        return "n/a"
    return (f"{c['delta']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}]" + (" SEP" if c.get("separated") else "")
            + f" ({c['mean_new']:.4f} vs {c['mean_ref']:.4f}, n={c['n_windows']}/{c['n_episodes']} ep)")


def manifest_sha(rel: str) -> str:
    for ln in LANDED_MAN.read_text(encoding="utf-8").splitlines():
        p = ln.split(None, 1)
        if len(p) == 2 and p[1].lstrip("*").strip() == rel:
            return RC.hex64(p[0])
    return ""


def main() -> int:
    t0 = time.time()
    prereg = RC.check_prereg()
    amend1 = RC.check_amendment1()
    fixpin = RC.check_fix_pin()
    RC.setup_paths()
    sys.path.insert(0, "C:/Users/Admin/refcv8_pf/refcv9_l2/code")
    import l2_common as L2
    import compliant_reread_r8 as CR
    import clamp_posthoc_r8 as CP
    import l1_fit_score as F
    st = L2.stack()
    PR, ra = st["PR"], st["ra"]
    METRICS = [(m[0], m[1], bool(m[4])) for m in PR.METRICS]           # (family, key, lower_is_better)
    tools = F.RT.tools_load()
    rc = tools["rc"]
    V_cmp, vrec = CP.load_vendored()
    comp_rec = json.loads((RC.RESULT / "R9PF_COMPOSE.json").read_text(encoding="utf-8"))
    keq_path = RC.RESULT / "R9PF_KEQ.json"
    keq = json.loads(keq_path.read_text(encoding="utf-8")) if keq_path.is_file() else None
    tags = sorted(p.name for p in PKG.iterdir() if p.is_dir())
    ctl, R = {"compose_ALL_PASS": bool(comp_rec["controls"]["ALL_PASS"])}, {}
    # ---- beside packages pinned to the landed manifest
    pins = {}
    for k, nm in BESIDE.items():
        for s in (0, 1):
            rel = f"out_mdfix/arms/h2cn_mdfix/{nm}/DEPLOY8_POS_{TREE}_s{s}.npz"
            got, want = RC.sha256_file(LANDED / nm / f"DEPLOY8_POS_{TREE}_s{s}.npz"), manifest_sha(rel)
            pins[f"{nm}|s{s}"] = {"sha256": got, "manifest": want, "match": bool(len(got) == 64 and got == want)}
    ctl["K-BESIDE-PINNED"] = {"PASS": all(v["match"] for v in pins.values()), "pins": pins}
    landed_fwd = json.loads((LANDED / "H2CN_DEPLOY8" / f"FORWARD_DEPLOY8_{TREE}.json").read_text(encoding="utf-8"))
    for s in (0, 1):
        E = L2.load_eval_pf(s)
        I = np.load(PKG / f"EVAL_INPUTS_{TREE}_s{s}.npz", allow_pickle=False)
        eid6 = list(E["eid"])
        comp6 = np.asarray(E["compliant"], bool)
        B = {k: np.load(LANDED / nm / f"DEPLOY8_POS_{TREE}_s{s}.npz", allow_pickle=False) for k, nm in BESIDE.items()}
        pk = {t: np.load(PKG / t / f"R9_POS_{TREE}_s{s}.npz", allow_pickle=False) for t in tags}
        al = {t: bool(np.array_equal(z["S6_ws"], E["ws"]) and np.array_equal(z["S6_clip"], E["clip"]) and np.array_equal(z["S2_ws"], B["H8"]["S2_ws"])
                      and np.array_equal(z["S2_clip"], B["H8"]["S2_clip"])) for t, z in pk.items()}
        al.update({f"beside_{k}": bool(np.array_equal(z["S6_ws"], E["ws"]) and np.array_equal(z["S2_ws"], I["S2_ws"])) for k, z in B.items()})
        al["inputs_S6_eq_E"] = bool(np.array_equal(I["S6_ws"], E["ws"]) and np.array_equal(I["S6_clip"], E["clip"]))
        ctl[f"K-ALIGN|s{s}"] = {"PASS": all(al.values()), "detail": al}
        # surface data
        surf = {"S2": {"G": I["S2_G"].astype(np.float32), "dt": float(I["S2_dt"]), "eid": list(I["S2_eid"]), "mask": I["S2_compliant"].astype(bool),
                       "HP": I["S2_HP"].astype(np.float32), "OS": I["S2_plan_battery"][:, RC.SLOTS["S2"]].astype(np.float32)},
                "S6": {"G": E["g"].astype(np.float32), "dt": float(E["dt"]), "eid": eid6, "mask": comp6, "HP": I["S6_HP"].astype(np.float32),
                       "OS": I["S6_plan_battery"][:, RC.SLOTS["S6"]].astype(np.float32)}}
        ctl[f"K-GT|s{s}"] = {"PASS": bool(np.allclose(I["S6_G"].astype(np.float64), E["g"], atol=1e-6) and comp6.sum() == int(I["S6_compliant"].sum()))}
        arms_pos = {t: {sf: z[f"{sf}_pos"] for sf in ("S2", "S6")} for t, z in pk.items()}
        arms_pos.update({k: {sf: z[f"{sf}_pos"] for sf in ("S2", "S6")} for k, z in B.items()})
        arms_pos["OS"] = {sf: surf[sf]["OS"] for sf in ("S2", "S6")}
        arms_pos["HP"] = {sf: surf[sf]["HP"] for sf in ("S2", "S6")}
        comps = {a: {sf: PR.components(ra, np.asarray(p[sf], np.float32), surf[sf]["G"], surf[sf]["dt"]) for sf in ("S2", "S6")} for a, p in arms_pos.items()}
        # ---- K-E13: H2CN_DEPLOY8 - ha0_ext_pose ADE reproduces the landed forward's cell (4 dp)
        ke = {}
        for sf in ("S2", "S6"):
            m = surf[sf]["mask"]
            c = cell(PR, comps["H8"][sf]["ade_m"][m], comps["HP"][sf]["ade_m"][m], [e for e, k in zip(surf[sf]["eid"], m) if k])
            ref = landed_fwd["seeds"][str(s)]["surfaces"][sf]["sets"]["compliant"]["arms"]["H2CN_DEPLOY8"]["minus_ha0_ext_pose"]
            ke[sf] = {"mine": c, "landed": {k: ref.get(k) for k in ("delta", "lo", "hi", "n_windows")},
                      "PASS": bool(all(abs(round(c[k], 4) - ref[k]) < 1e-9 for k in ("delta", "lo", "hi")) and c["n_windows"] == ref["n_windows"])}
        ctl[f"K-E13|s{s}"] = {"PASS": all(v["PASS"] for v in ke.values()), "detail": ke}
        # ---- MD / FD indicators (S6 compliant)
        ind = {a: L2.lon_indicators(np.asarray(p["S6"], np.float32), E, comp6) for a, p in arms_pos.items() if a not in ("HP",)}
        fdI, mdI = {a: v[0] for a, v in ind.items()}, {a: v[1] for a, v in ind.items()}
        ctl[f"K-ZERO|s{s}"] = {"PASS": bool(cell(PR, mdI["H8"], mdI["H8"], eid6)["delta"] == 0.0)}
        # ---- exceedance (full 8-slot plan, all windows; the registered compliance row's rule)
        lim2 = I["S2_lim"].astype(np.float32)
        full8 = {t: z["S2_pos8"] for t, z in pk.items()}
        full8.update({k: z["S2_pos8"] for k, z in B.items()})
        full8["OS"] = I["S2_plan_battery"].astype(np.float32)
        exc = {}
        for a, p8 in full8.items():
            pm = CR.pmax(V_cmp, np.asarray(p8, np.float32), [int(h) for h in rc["horizons"]], rc["tick_s"])
            exc[a] = (np.isfinite(lim2) & ((pm - lim2).astype(np.float32) > 0)).astype(np.float64)
        lc = landed_fwd["compliance_row"]["cells"][f"S2|s{s}"]["arms"]["H2CN_DEPLOY8"]["full_plan_0_6s"]["all_windows"]
        ctl[f"K-EXC|s{s}"] = {"PASS": bool(int(exc["H8"].sum()) == int(lc["n_exceed_strict"])), "mine": int(exc["H8"].sum()), "landed": lc["n_exceed_strict"]}
        eid2 = surf["S2"]["eid"]
        # ---- per-arm cells
        for t in tags + ["FX", "D8", "OS"]:
            r = R.setdefault(t, {}).setdefault(f"s{s}", {})
            r["MD_vs_H8"], r["FD_vs_H8"] = cell(PR, mdI[t], mdI["H8"], eid6), cell(PR, fdI[t], fdI["H8"], eid6)
            r["MD_vs_FX"], r["FD_vs_FX"] = cell(PR, mdI[t], mdI["FX"], eid6), cell(PR, fdI[t], fdI["FX"], eid6)
            r["MD_vs_D8"], r["FD_vs_D8"] = cell(PR, mdI[t], mdI["D8"], eid6), cell(PR, fdI[t], fdI["D8"], eid6)
            r["MD_rate"], r["FD_rate"] = float(np.nanmean(mdI[t])), float(np.nanmean(fdI[t]))
            r["MD_n"], r["FD_n"] = int(np.isfinite(mdI[t]).sum()), int(np.isfinite(fdI[t]).sum())
            for sf in ("S2", "S6"):
                m = surf[sf]["mask"]
                em = [e for e, k in zip(surf[sf]["eid"], m) if k]
                r[f"ADE_{sf}_vs_D8"] = cell(PR, comps[t][sf]["ade_m"][m], comps["D8"][sf]["ade_m"][m], em)
                r[f"ADE_{sf}_vs_HP"] = cell(PR, comps[t][sf]["ade_m"][m], comps["HP"][sf]["ade_m"][m], em)
                r[f"fam_{sf}_vs_H8"] = {k: {**cell(PR, comps[t][sf][k][m], comps["H8"][sf][k][m], em), "family": fam, "lower_is_better": lib}
                                        for fam, k, lib in METRICS if k in comps[t][sf]}
                if t in ("R9PF_fs0",):
                    r[f"fam_{sf}_vs_D8"] = {k: {**cell(PR, comps[t][sf][k][m], comps["D8"][sf][k][m], em), "family": fam, "lower_is_better": lib}
                                            for fam, k, lib in METRICS if k in comps[t][sf]}
            r["EXC_vs_H8"] = cell(PR, exc[t], exc["H8"], eid2)
            r["EXC_rate"] = float(exc[t].mean())
            r["EXC_n"] = int(exc[t].sum())
            if t.startswith("R9PF"):
                dz = pk[t]["S6_decision"][comp6]
                r["decision_counts_S6_compliant"] = np.bincount(dz, minlength=3).tolist()
        for k in ("H8",):
            R.setdefault("H8", {}).setdefault(f"s{s}", {}).update({"MD_rate": float(np.nanmean(mdI["H8"])), "FD_rate": float(np.nanmean(fdI["H8"])),
                                                                     "EXC_rate": float(exc["H8"].mean()), "EXC_n": int(exc["H8"].sum())})
        print(f"[a3b] seed {s} cells done ({time.time() - t0:.0f} s)", flush=True)
    # ---------------------------------------------------------------- bars
    def pa1a(t):
        return {f"s{s}": bool(sep_below(R[t][f"s{s}"]["MD_vs_H8"]) and not sep_above(R[t][f"s{s}"]["FD_vs_H8"])) for s in (0, 1)}

    def pa1b(t):
        return {f"s{s}": bool(sep_below(R[t][f"s{s}"]["FD_vs_FX"]) and not sep_above(R[t][f"s{s}"]["MD_vs_FX"])) for s in (0, 1)}

    def pa2(t):
        out = {}
        for s in (0, 1):
            r = R[t][f"s{s}"]
            out[f"s{s}"] = {"vs_RETIME": bool(sep_below(r["ADE_S2_vs_D8"]) and sep_below(r["ADE_S6_vs_D8"])),
                            "E1_E3_vs_ha0_ext_pose": bool(sep_below(r["ADE_S2_vs_HP"]) and sep_below(r["ADE_S6_vs_HP"]))}
            out[f"s{s}"]["holds"] = bool(out[f"s{s}"]["vs_RETIME"] and out[f"s{s}"]["E1_E3_vs_ha0_ext_pose"])
        return out

    def pag(t):
        worse = []
        for s in (0, 1):
            r = R[t][f"s{s}"]
            for sf in ("S2", "S6"):
                for k, c in r[f"fam_{sf}_vs_H8"].items():
                    if (c["lower_is_better"] and sep_above(c)) or (not c["lower_is_better"] and sep_below(c)):
                        worse.append(f"s{s}|{sf}|{c['family']}|{k}: {fmt(c)}")
            if sep_above(r["EXC_vs_H8"]):
                worse.append(f"s{s}|exceedance (full plan, all windows): {fmt(r['EXC_vs_H8'])}")
        return {"holds": not worse, "worse_rows": worse}

    bars = {}
    for t in tags:
        a, b, c2, g = pa1a(t), pa1b(t), pa2(t), pag(t)
        bars[t] = {"PA-1a": all(a.values()), "PA-1a_per_seed": a, "PA-1b": all(b.values()), "PA-1b_per_seed": b,
                   "PA-2": all(v["holds"] for v in c2.values()), "PA-2_detail": c2, "PA-G": g["holds"], "PA-G_worse_rows": g["worse_rows"]}
    prim = "R9PF_fs0"
    pa_r = {f"fs{k}": {x: bars[f"R9PF_fs{k}"][x] for x in ("PA-1a", "PA-1b", "PA-2")} for k in RC.FIT_SEEDS}
    pa_r_holds = all(all(v.values()) for v in pa_r.values())
    k_constz = {f"fs{k}": {"PA-1a": bars[f"R9PF-CONSTZ_fs{k}"]["PA-1a"], "PA-1b": bars[f"R9PF-CONSTZ_fs{k}"]["PA-1b"]} for k in RC.FIT_SEEDS}
    k_shuf = {f"fs{k}": bars[f"R9PF-SHUF_fs{k}"]["PA-1a"] for k in RC.FIT_SEEDS}
    ctl["K-CONSTZ"] = {"PASS": not any(v["PA-1a"] or v["PA-1b"] for v in k_constz.values()), "constz_passes": k_constz,
                       "reading": "CONSTZ passing PA-1a OR PA-1b at ANY fit seed voids the run (the conservative reading of 'must NOT pass PA-1')"}
    ctl["K-SHUF"] = {"PASS": not any(k_shuf.values()), "shuf_passes_PA-1a": k_shuf}
    if keq is not None:
        for k in ("K-EQ", "K-OFF"):
            ctl[k] = keq.get(k, {"PASS": False, "reason": "absent from R9PF_KEQ.json"})
    else:
        ctl["K-EQ"] = ctl["K-OFF"] = {"PASS": False, "reason": "R9PF_KEQ.json does not exist (the subset roll did not run)"}
    geo_keys = [k for k in comp_rec["controls"] if k.startswith(("K-GEO", "K-LAT"))]
    ctl["K-GEO"] = {"PASS": all(comp_rec["controls"][k]["PASS"] for k in geo_keys if k.startswith("K-GEO")),
                    "max_abs_m": max(comp_rec["controls"][k]["max_abs_torch64_vs_numpy_m"] for k in geo_keys if k.startswith("K-GEO"))}
    ctl["K-LAT"] = {"PASS": all(comp_rec["controls"][k]["PASS"] for k in geo_keys if k.startswith("K-LAT")),
                    "max_dist_f64_m": max(comp_rec["controls"][k]["max_dist_f64_m"] for k in geo_keys if k.startswith("K-LAT")),
                    "max_dist_after_f32_cast_m": max(comp_rec["controls"][k]["max_dist_after_f32_cast_m"] for k in geo_keys if k.startswith("K-LAT"))}
    void_keys = ["compose_ALL_PASS", "K-BESIDE-PINNED", "K-CONSTZ", "K-SHUF", "K-EQ", "K-OFF", "K-GEO", "K-LAT"] + \
                [k for k in ctl if k.startswith(("K-ALIGN", "K-GT", "K-E13", "K-ZERO", "K-EXC"))]
    failed_ctl = [k for k in void_keys if not (ctl[k] if isinstance(ctl[k], bool) else ctl[k].get("PASS"))]
    # ---- INERT read: R9PF vs R9PF-TL per fit seed against the fit-seed floor
    inert = {}
    for s in (0, 1):
        for met in ("MD", "FD"):
            rates = [R[f"R9PF_fs{k}"][f"s{s}"][f"{met}_rate"] for k in RC.FIT_SEEDS]
            floor = float(max(rates) - min(rates))
            dl = [R[f"R9PF_fs{k}"][f"s{s}"][f"{met}_rate"] - R[f"R9PF-TL_fs{k}"][f"s{s}"][f"{met}_rate"] for k in RC.FIT_SEEDS]
            inert[f"s{s}|{met}"] = {"fit_seed_floor": floor, "R9PF_minus_TL": dl, "within_floor": bool(all(abs(x) <= floor for x in dl))}
    inert_flag = all(v["within_floor"] for v in inert.values())
    p = bars[prim]
    PASS = bool(p["PA-1a"] and p["PA-1b"] and p["PA-2"] and p["PA-G"] and pa_r_holds)
    if failed_ctl:
        word = "VOID"
    elif PASS:
        word = "PASS"
    elif p["PA-1a"] != p["PA-1b"]:
        word = "PARTIAL"
    elif inert_flag:
        word = "INERT"
    elif not p["PA-G"]:
        word = "FAIL-GUARD"
    else:
        word = "FAIL"
    # inference-seed floor beside every primary effect
    isf = {k: abs(R[prim]["s0"][k]["mean_new"] - R[prim]["s1"][k]["mean_new"]) for k in ("MD_vs_H8", "FD_vs_H8", "MD_vs_FX", "FD_vs_FX", "ADE_S2_vs_D8", "ADE_S6_vs_D8")}
    out = {"tool": "a3_bars.py", "prereg": prereg, "amendment1": amend1, "fix_pin": fixpin, "word": word, "tier": TIER, "estimator": EST, "primary_arm": prim,
           "bars_primary": {x: p[x] for x in ("PA-1a", "PA-1b", "PA-2", "PA-G")}, "PA-R": {"holds": pa_r_holds, "per_fit_seed": pa_r},
           "PASS": PASS, "bars_all_heads": bars, "controls": ctl, "failed_controls": failed_ctl, "inert_read": {"INERT": inert_flag, "detail": inert},
           "inference_seed_floor_primary": isf, "cells": R, "keq_record": (None if keq is None else {"file": str(keq_path), "sha256": RC.sha256_file(keq_path)}),
           "compose_record": {"file": str(RC.RESULT / "R9PF_COMPOSE.json"), "sha256": RC.sha256_file(RC.RESULT / "R9PF_COMPOSE.json")},
           "vendored_selection": vrec, "families_note": "ADE / LONGITUDINAL / LATERAL / TACTICAL = pair_read_r8.METRICS (14 rows); STRATEGIC n/a (refcv8 runs "
           "--no-strategic: no strategic head exists to score); distance-keeping (headway / TTC) is not instrumented in this battery (n/a, no lead-agent "
           "join in the eval139 panel)", "script_sha256": RC.sha256_file(__file__), "wall_s": round(time.time() - t0, 1), "generated": time.strftime("%FT%T%z")}
    # without the K-EQ / K-OFF record the run is PROVISIONAL and is written to scratch only (never to result/)
    RC.jdump(out, (RC.RESULT / "R9PF_PA_BARS.json") if keq is not None else (RC.SCRATCH / "R9PF_PA_BARS_PROVISIONAL_NO_KEQ.json"))
    print(f"[a3b] word {word} PASS={PASS} failed_controls={failed_ctl}  ZZA3BARS_{word}ZZ", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
