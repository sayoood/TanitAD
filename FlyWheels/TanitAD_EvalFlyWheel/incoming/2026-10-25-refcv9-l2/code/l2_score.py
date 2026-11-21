"""l2_score.py -- refcv9 L2: the MECHANICAL reading of PREREG_L2.md (+ Amendment 1) sections 6-8 from the banked rolls.  CPU only.

    python l2_score.py --bank <dir> --out <dir> [--l1-bank <dir>] [--gate after_const|after_cal0]

Reads: the banked 48,852 per-frame battery (BASE-bat, refcv7, GT, prior, banked stage), L1's un-hooked os-only bank (BASE-os), and this chain's traces
(`trace_{native300,ident300}_s0.npz`, `trace_{cal,const_cruise,oracle}_s{seed}.npz`).  Writes L2_RESULT.json / L2_RESULT.md / VERDICT.md.
Every arm is read with and without the shipped stage; the bars read on "+stage".  A missing arm/seed is PENDING, never PASS.
Reading of the B9-0 guard G: STRICT -- a METRICS row separated worse at ANY rolled seed fails G (the lenient 'both seeds' form is printed beside).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import l2_common as C  # noqa: E402
import l2_fit as F  # noqa: E402

BAR_FD, BAR_MD_LITERAL = 0.10, 0.358
SEEDS = (0, 1)
ARMS_ROLLED = ("cal", "const_cruise", "oracle")


def load_trace(path):
    p = Path(path)
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=False)
    meta = json.loads(str(z["meta_json"]))
    if meta.get("final") is not True:
        return {"incomplete": True, "meta": meta}
    idx = {(int(e), int(w)): j for j, (e, w) in enumerate(zip(z["e_i"], z["ws"]))}
    return {"idx": idx, "plan8": z["plan8"].astype(np.float32), "sel": z["sel"], "raw": z["raw_lon_logits"], "app": z["applied_lon_logits"], "meta": meta, "n": len(z["ws"])}


def load_l1_base(l1_bank, seed):
    p = Path(l1_bank) / f"ztac_eval_s{seed}.npz"
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=False)
    meta = json.loads(str(z["meta_json"]))
    if meta.get("final") is not True or int(meta.get("n_windows", 0)) != 4754:
        return {"incomplete": True, "meta": meta}
    W = int(meta["window_W"])
    ws = z["t"].astype(np.int64) + W - 1
    idx = {(int(e), int(w)): j for j, (e, w) in enumerate(zip(z["e_i"], ws))}
    return {"idx": idx, "plan8": z["traj_own"].astype(np.float32), "sel": z["sel_own"], "meta": meta, "n": len(ws), "path": str(p)}


def join_plan8(tr, E):
    keys = list(zip(E["ex_clip"][E["rows"]].tolist(), E["ex_ws"][E["rows"]].tolist()))
    j = np.array([tr["idx"].get(k, -1) for k in keys])
    if (j < 0).any():
        raise SystemExit(f"[l2_score] {int((j < 0).sum())} S6 windows have no entry in the trace")
    return tr["plan8"][j]


def arm_cells(E, plan8):
    """Per-window arrays (NaN outside the compliant mask) for one arm, with and without the stage."""
    out = {}
    comp = E["compliant"]
    for sk, P in (("nostage", plan8[:, C.S6_SLOTS]), ("stage", C.apply_stage(plan8, E["prior8"]))):
        c = C.arm_metrics(E, P)
        fd, md, gt, pc = C.lon_indicators(P, E, comp)
        cruise = comp & (gt == "cruise")
        d = {k: np.where(comp, v, np.nan) for k, v in c.items()}
        d["false_decel"], d["missed_decel"] = fd, md
        d["along_signed_final_m@GTcruise"] = np.where(cruise, c["along_signed_final_m"], np.nan)
        P64 = np.asarray(P, np.float64)

        def arclen(X):
            Z = np.concatenate([np.zeros_like(X[:, :1]), X], axis=1)
            return np.linalg.norm(np.diff(Z, axis=1), axis=-1).sum(1)
        sp = (arclen(P64) - arclen(E["g"])) / (P.shape[1] * E["dt"])
        d["speed_err_signed_mps@GTcruise"] = np.where(cruise, sp, np.nan)
        out[sk] = d
    return out


def metric_rows(PR):
    rows = [(f, k, lab, unit, lower) for (f, k, lab, unit, lower, _s) in PR.METRICS]
    rows += [("LONGITUDINAL", "false_decel", "false-decel rate (GT-cruise)", "frac", True), ("LONGITUDINAL", "missed_decel", "missed-decel rate (GT-decel)", "frac", True),
             ("LONGITUDINAL", "along_signed_final_m@GTcruise", "signed along-track at 6 s (GT-cruise)", "m", None),
             ("LONGITUDINAL", "speed_err_signed_mps@GTcruise", "signed mean speed error (GT-cruise)", "m/s", None)]
    return rows


def compare(PR, E, A, B, rows):
    out = {}
    for fam, k, lab, unit, lower in rows:
        if k not in A or k not in B:
            continue
        c = PR.paired_cell(A[k], B[k], list(E["eid"]))
        c = {kk: (float(v) if isinstance(v, (float, np.floating)) else v) for kk, v in c.items() if kk in ("delta", "lo", "hi", "n_windows", "n_episodes", "mean_new", "mean_ref", "separated", "status", "reason")}
        c.update(family=fam, label=lab, unit=unit, lower_is_better=lower)
        if lower is None or "delta" not in c:
            c["worse"] = None
        else:
            c["worse"] = bool((lower and c["lo"] > 0) or ((not lower) and c["hi"] < 0))
        out[k] = c
    return out


def rate(cells, key):
    return float(np.nanmean(cells[key]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--l1-bank", default="C:/Users/Admin/refcv8_pf/refcv9_l1/bank")
    ap.add_argument("--gate", choices=("after_const", "after_cal0"), default=None)
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    t0 = time.time()
    root = Path(__file__).resolve().parent.parent
    seals = {}
    for nm, f, s in (("prereg", "PREREG_L2.md", "PREREG_L2.sha256"), ("amendment1", "PREREG_L2_AMENDMENT1.md", "PREREG_L2_AMENDMENT1.sha256")):
        seals[nm] = C.sha256_file(root / f)
        if seals[nm] != (root / s).read_text().split()[0]:
            raise SystemExit(f"[l2_score] {f} does not match its seal: refusing")
    st = C.stack()
    PR = st["PR"]
    rows = metric_rows(PR)
    bank = Path(a.bank)
    res = {"tool": "l2_score.py", "seals": seals, "tier": "T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)",
           "estimator": "paired episode-cluster bootstrap (pair_read_r8.paired_cell -> taniteval.ci), B 2000, seed 0, over EPISODES: answers 'another draw of episodes' only; "
                        "NOT another training run (single training seed), NOT another inference run (seeds 0 / 1 shown separately, never pooled)",
           "families_not_computed": {"LONGITUDINAL distance-keeping (headway/TTC)": "no lead-agent track in the banked refcv8 dumps; n = 0; named work item",
                                     "STRATEGIC": "refcv8 runs --no-strategic; n = 0"},
           "script_sha256": C.sha256_file(__file__), "seeds": {}, "controls": {}, "pending": []}
    Es, cells, plans = {}, {}, {}
    for s in SEEDS:
        E = C.load_eval_pf(s)
        Es[s] = E
        arms = {"BASE-bat": E["plan8"]}
        b1 = load_l1_base(a.l1_bank, s)
        if b1 and not b1.get("incomplete"):
            arms["BASE-os"] = join_plan8(b1, E)
        else:
            res["pending"].append(f"BASE-os s{s} (L1 os-only bank {'incomplete' if b1 else 'absent'})")
        for arm in ARMS_ROLLED:
            tr = load_trace(bank / f"trace_{arm}_s{s}.npz")
            if tr and not tr.get("incomplete"):
                arms[arm] = join_plan8(tr, E)
                plans[(arm, s)] = tr
            else:
                res["pending"].append(f"{arm} s{s} ({'incomplete' if tr else 'not rolled'})")
        cells[s] = {nm: arm_cells(E, p8) for nm, p8 in arms.items()}
        # refcv7: the panel's embedded plan (6 instants, no stage)
        c7 = C.arm_metrics(E, E["r7"])
        fd7, md7, gt, _pc = C.lon_indicators(E["r7"], E, E["compliant"])
        d7 = {k: np.where(E["compliant"], v, np.nan) for k, v in c7.items()}
        d7["false_decel"], d7["missed_decel"] = fd7, md7
        cells[s]["r7"] = {"nostage": d7, "stage": d7}
    # ---- controls ---------------------------------------------------------------------------------------------
    ctl = res["controls"]
    for s in SEEDS:
        E = Es[s]
        ctl[f"K1_stage_bit_equal_banked_s{s}"] = bool(np.array_equal(C.apply_stage(E["plan8"], E["prior8"]), E["stage_banked"]))
        pop = {"n": int(E["compliant"].sum()), "ep": len(set(E["eid"][E["compliant"]].tolist()))}
        ctl[f"K2_population_s{s}"] = bool(pop["n"] == 3251 and pop["ep"] == 138)
    nat, idn = load_trace(bank / "trace_native300_s0.npz"), load_trace(bank / "trace_ident300_s0.npz")
    if nat and idn and not nat.get("incomplete") and not idn.get("incomplete"):
        keys = sorted(set(nat["idx"]) & set(idn["idx"]))
        jn, ji = [nat["idx"][k] for k in keys], [idn["idx"][k] for k in keys]
        dmax = float(np.abs(nat["plan8"][jn] - idn["plan8"][ji]).max())
        ctl["K3_ident_vs_native"] = {"n": len(keys), "max_abs_m": dmax, "sel_equal": bool(np.array_equal(nat["sel"][jn], idn["sel"][ji])),
                                     "hook_calls_armed": idn["meta"]["counters"]["hook_calls_armed"], "pass": bool(dmax <= 1e-5 and len(keys) >= 250)}
    else:
        ctl["K3_ident_vs_native"] = {"pass": None, "why": "smoke traces absent/incomplete"}
    for arm in ARMS_ROLLED:
        for s in SEEDS:
            tr = plans.get((arm, s))
            if not tr:
                continue
            if arm == "cal":
                cal = C.jload(tr["meta"]["calibrator"]["file"])
                want = F.apply8(tr["raw"].astype(np.float64), b8=cal["b8"], T=cal["T"])
                ctl[f"K4_cal_trace_vs_apply8_s{s}"] = {"max_abs": float(np.abs(want - tr["app"]).max()), "pass": bool(np.abs(want - tr["app"]).max() <= 1e-4)}
            if arm == "const_cruise":
                ctl[f"K4_const_all_cruise_s{s}"] = {"frac_argmax_cruise": float((tr["app"].argmax(1) == 1).mean()), "pass": bool((tr["app"].argmax(1) == 1).all())}
    # K5: BASE-os vs BASE-bat rates
    if all("BASE-os" in cells[s] for s in SEEDS):
        k5 = {}
        for key in ("false_decel", "missed_decel"):
            r = {s: (rate(cells[s]["BASE-os"]["stage"], key), rate(cells[s]["BASE-bat"]["stage"], key)) for s in SEEDS}
            tol = max(0.01, 2 * abs(r[0][1] - r[1][1]))
            k5[key] = {"BASE-os": {s: r[s][0] for s in SEEDS}, "BASE-bat": {s: r[s][1] for s in SEEDS}, "tol": tol,
                       "pass": bool(all(abs(r[s][0] - r[s][1]) <= tol for s in SEEDS))}
        ctl["K5_base_os_vs_battery_rates"] = k5
    fitst = Path(root / "result" / "fit_selftest.json")
    ctl["K7_fit_selftest"] = (C.jload(fitst).get("ALL_PASS") if fitst.exists() else None)
    # ---- comparisons ------------------------------------------------------------------------------------------
    primary = "BASE-os" if all("BASE-os" in cells[s] for s in SEEDS) else "BASE-bat"
    res["primary_comparator"] = primary
    comps = {}
    fast = a.gate is not None                                  # gate mode: only what the gate token needs (the GPU is held while this runs)
    fd_md_rows = [r for r in rows if r[1] in ("false_decel", "missed_decel")]
    for s in SEEDS:
        comps[s] = {}
        for arm in ARMS_ROLLED:
            if arm not in cells[s]:
                continue
            for sk in (("stage",) if fast else ("stage", "nostage")):
                for cmp_name in ((primary, "r7") if fast else ("BASE-os", "BASE-bat", "r7")):
                    if cmp_name not in cells[s]:
                        continue
                    ref = cells[s][cmp_name][sk if cmp_name != "r7" else "stage"]
                    rr = rows if (not fast or arm == "cal") else fd_md_rows
                    comps[s][f"{arm}|{sk}|vs {cmp_name}"] = compare(PR, Es[s], cells[s][arm][sk], ref, rr)
        if not fast:
            for sk in ("stage", "nostage"):                    # the two refcv8 baselines against refcv7
                for cmp_name in ("BASE-os", "BASE-bat"):
                    if cmp_name in cells[s]:
                        comps[s][f"{cmp_name}|{sk}|vs r7"] = compare(PR, Es[s], cells[s][cmp_name][sk], cells[s]["r7"]["stage"], rows)
        res["seeds"][str(s)] = {"rates": {f"{nm}|{sk}": {"false_decel": rate(c[sk], "false_decel"), "missed_decel": rate(c[sk], "missed_decel"),
                                                         "n_gt_cruise": int(np.isfinite(c[sk]["false_decel"]).sum()), "n_gt_decel": int(np.isfinite(c[sk]["missed_decel"]).sum())}
                                          for nm, c in cells[s].items() for sk in ("stage", "nostage")},
                                "comparisons": comps[s]}
    # ---- floor ------------------------------------------------------------------------------------------------
    floors = {}
    for nm in ("BASE-os", "BASE-bat"):
        if all(nm in cells[s] for s in SEEDS):
            floors[nm] = abs(rate(cells[0][nm]["stage"], "false_decel") - rate(cells[1][nm]["stage"], "false_decel"))
    F_floor = max(floors.values()) if floors else None
    res["inference_seed_floor_fd"] = {"per_baseline": floors, "F": F_floor}
    # ---- bars -------------------------------------------------------------------------------------------------
    bars = {}
    for s in SEEDS:
        if "cal" not in cells[s] or primary not in cells[s]:
            bars[s] = None
            continue
        cmp_b = comps[s][f"cal|stage|vs {primary}"]
        cmp_7 = comps[s]["cal|stage|vs r7"]
        fd = rate(cells[s]["cal"]["stage"], "false_decel")
        md = rate(cells[s]["cal"]["stage"], "missed_decel")
        d = cmp_b["false_decel"]
        p2 = bool(d["hi"] < 0 and (F_floor is not None and abs(d["delta"]) > F_floor))
        g_rows = {k: v["worse"] for k, v in cmp_b.items() if v.get("worse") is not None}
        bars[s] = {"fd": fd, "md": md, "P1": bool(fd <= BAR_FD), "P2": p2, "P2_cell": d, "P3": bool(md <= BAR_MD_LITERAL),
                   "P3L": bool(cmp_b["missed_decel"]["lo"] <= 0), "P3L_cell": cmp_b["missed_decel"], "P4": bool(cmp_7["false_decel"]["lo"] <= 0), "P4_cell": cmp_7["false_decel"],
                   "G_rows_separated_worse": sorted(k for k, w in g_rows.items() if w), "G": bool(not any(g_rows.values()))}
    res["bars"] = {str(s): bars[s] for s in SEEDS}
    rolled = [s for s in SEEDS if bars[s] is not None]
    common = lambda b: b["P1"] and b["P2"] and b["P4"] and b["G"]        # noqa: E731
    fails_common = any(not common(bars[s]) for s in rolled)
    if not rolled:
        verdict = "PENDING (no CAL roll banked)"
    elif len(rolled) < 2:
        verdict = "FAIL (seed %d fails a clause common to both PASS labels; the other seed cannot change it)" % rolled[0] if fails_common else "PENDING (one seed rolled)"
    else:
        if all(bars[s]["P1"] and bars[s]["P2"] and bars[s]["P3"] and bars[s]["P4"] and bars[s]["G"] for s in rolled):
            verdict = "PASS"
        elif all(common(bars[s]) and bars[s]["P3L"] for s in rolled):
            verdict = "PASS-AT-SHIPPED-BASELINE"
        else:
            verdict = "FAIL"
    res["verdict_bars"] = verdict
    # K6 and the outcome label
    k6 = None
    if "const_cruise" in cells[0] and primary in cells[0]:
        c = comps[0][f"const_cruise|stage|vs {primary}"]["false_decel"]
        k6 = {"cell": c, "pass": bool(c["hi"] < 0)}
        ctl["K6_const_potency"] = k6
    const_fd = {s: (rate(cells[s]["const_cruise"]["stage"], "false_decel") if "const_cruise" in cells[s] else None) for s in SEEDS}
    const_lo = (comps[0][f"const_cruise|stage|vs r7"]["false_decel"] if "const_cruise" in cells[0] else None)
    # CONST FD lower CI bound against the 0.10 bar: the single-arm interval of the rate (paired vs the constant 0.10 is not a paired cell) via a bootstrap on the arm alone
    const_ci = None
    if "const_cruise" in cells[0]:
        from taniteval import ci as _ci
        x = cells[0]["const_cruise"]["stage"]["false_decel"]
        keep = np.isfinite(x)
        r = _ci.paired_episode_cluster_bootstrap(x[keep], np.full(int(keep.sum()), BAR_FD), [e for e, k in zip(Es[0]["eid"], keep) if k], n_boot=a.n_boot, seed=0)
        const_ci = {"fd": float(np.mean(x[keep])), "lo_vs_bar": float(r["lo"]) + BAR_FD, "hi_vs_bar": float(r["hi"]) + BAR_FD}
    res["const_fd_ci_seed0"] = const_ci
    ctl_fail = [k for k, v in ctl.items() if (isinstance(v, bool) and not v) or (isinstance(v, dict) and v.get("pass") is False and not k.startswith("K6"))]
    if ctl_fail:
        outcome = f"O6 INVALID (control failed: {ctl_fail})"
    elif k6 is not None and not k6["pass"]:
        outcome = "O5 CHANNEL-INERT (K6: forcing CRUISE does not move the plan)"
    elif verdict == "PASS":
        outcome = "O1 PASS"
    elif verdict == "PASS-AT-SHIPPED-BASELINE":
        outcome = "O2 PASS-AT-SHIPPED-BASELINE"
    elif verdict.startswith("FAIL") and const_ci is not None:
        outcome = ("O4 FAIL, CONST does NOT reach FD <= 0.10 (lower bound %.3f > 0.10)" % const_ci["lo_vs_bar"] if const_ci["lo_vs_bar"] > 0
                   else "O3 FAIL, CONST reaches FD <= 0.10 (point %.3f)" % const_ci["fd"] if const_ci["fd"] <= BAR_FD else "FAIL, CONST point %.3f > 0.10 but its interval covers 0.10 (O3/O4 undecided)" % const_ci["fd"])
    else:
        outcome = verdict
    res["outcome"] = outcome
    res["wall_s"] = round(time.time() - t0, 1)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    C.jdump(res, out / "L2_RESULT.json")
    # ---- md ---------------------------------------------------------------------------------------------------
    L = [f"# L2 RESULT (mechanical, PREREG_L2 sha256 {seals['prereg'][:12]}, Amendment 1 {seals['amendment1'][:12]})", "",
         f"**Verdict on the bars: {verdict}**  |  **Outcome: {outcome}**", "",
         f"tier: {res['tier']}", f"estimator: {res['estimator']}", f"primary comparator: {primary}; inference-seed floor on FD: {res['inference_seed_floor_fd']}", "",
         f"pending: {res['pending']}", "", "## rates (S6 compliant; false-decel on GT-cruise n 1,856, missed-decel on GT-decel n 639)", "",
         "| seed | arm | stage | false-decel | missed-decel |", "|---|---|---|---|---|"]
    for s in SEEDS:
        for k, v in res["seeds"][str(s)]["rates"].items():
            nm, sk = k.split("|")
            L.append(f"| {s} | {nm} | {sk} | {v['false_decel']:.4f} | {v['missed_decel']:.4f} |")
    L += ["", "## bars", "", "| seed | FD | MD | P1 | P2 | P3 | P3L | P4 | G | rows separated worse |", "|---|---|---|---|---|---|---|---|---|---|"]
    for s in SEEDS:
        b = bars[s]
        L.append(f"| {s} | " + (" | ".join([f"{b['fd']:.4f}", f"{b['md']:.4f}", str(b["P1"]), str(b["P2"]), str(b["P3"]), str(b["P3L"]), str(b["P4"]), str(b["G"]), ", ".join(b["G_rows_separated_worse"]) or "-"]) if b else "PENDING | - | - | - | - | - | - | - | - |") + " |")
    L += ["", "## controls", "", "```", json.dumps(ctl, indent=1, default=str), "```", "",
          "Full four-family tables for every pair: `L2_RESULT.json` -> seeds -> comparisons."]
    (out / "L2_RESULT.md").write_text("\n".join(L), encoding="utf-8")
    (out / "VERDICT.md").write_text(f"{verdict}\n{outcome}\n", encoding="utf-8")
    # ---- gate tokens for the chain -----------------------------------------------------------------------------
    if a.gate == "after_cal0":
        print("ZZL2GATEZZ", "SKIP_CAL_S1" if (bars[0] is not None and not common(bars[0])) else "RUN_CAL_S1")
    if a.gate == "after_const":
        print("ZZL2GATEZZ", "CONST_BLOCKS" if (const_ci and const_ci["lo_vs_bar"] > 0) else "CONST_OPEN")
    print(json.dumps({"verdict": verdict, "outcome": outcome, "pending": res["pending"], "wall_s": res["wall_s"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
