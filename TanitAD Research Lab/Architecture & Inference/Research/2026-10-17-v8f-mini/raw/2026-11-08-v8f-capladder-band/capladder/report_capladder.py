"""CAPLADDER analysis + report: reads rungs/*.npz and the control files, writes RESULT_CAPLADDER.json and RESULT_CAPLADDER.md (banner VALID / INVALID by controls 1-3 first, then control 4, the
readability table of lead_gap per rung and seed, the LOSS table per step, the outcomes, the beside rows). Run again after figs_capladder.py to list the figure. CPU, numpy only."""
import json, os, sys, time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import capladder_lib as CL

FEAT = "F:/TanitAD-scratch/capladder_feat"
RDIR = os.environ.get("CAPLADDER_RDIR", os.path.join(HERE, "rungs"))
OUT = os.environ.get("CAPLADDER_OUT", HERE)               # the real run writes beside the code; a dry run may redirect
SEEDS = ("s0", "s1")
DIMS = {"R0": 655360, "R1": 163840, "R2": 40960, "R2m": 1024, "R3": 3072, "R4": 384, "R3s": 1024, "R3t": 1024, "R3o": 1024, "L1": 128, "L2": 128, "L3": 128}
NAMES_AB = ["R0", "R1", "R2", "R2m"] + [f"{b}_{s}" for b in ("R3", "R4", "R3s", "R3t", "R3o", "L1", "L2", "L3") for s in SEEDS] + [f"{b}_r{r}" for b in ("R3", "R4") for r in (100, 101)]


def load(name):
    p = os.path.join(RDIR, f"{name}_A.npz")
    return np.load(p) if os.path.exists(p) else None


def pairs_all(have_r0):
    P = [("R1", "R2")] + ([("R0", "R1")] if have_r0 else [])
    for s in SEEDS:
        P += [("R2", f"R3_{s}"), ("R2", f"R4_{s}"), (f"R3_{s}", f"R4_{s}"), ("R2", f"R3s_{s}"), ("R2", f"R3t_{s}"), ("R2", f"R3o_{s}"),
              ("R2", f"L1_{s}"), (f"L1_{s}", f"L2_{s}"), (f"L2_{s}", f"L3_{s}")]
    return P


def fmt(x, d=3):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}f}"


def comb(a, b):
    return "READABLE" if (a and b) else "UNREADABLE" if (not a and not b) else "SEEDS-DISAGREE"


def combl(a, b):
    return "LOSS" if (a and b) else "no loss" if (not a and not b) else "SEEDS-DISAGREE"


def main():
    t0 = time.time()
    s = np.load(os.path.join(HERE, "sets_capladder.npz"), allow_pickle=False)
    uniq, nA, nD, n_ep = s["uniq"], int(s["nA"]), int(s["nD"]), len(s["uniq"])
    W = CL.boot_weights_local(uniq)
    w_check = "not checked"
    try:
        import wmact_lib
        W2 = wmact_lib.boot_weights(s["beid"])[1]
        w_check = "equal to wmact_lib.boot_weights" if np.array_equal(W, W2) else "DIFFERENT from wmact_lib.boot_weights"
    except Exception as e:                                                               # the analysis must not need the code root
        w_check = f"not checked ({type(e).__name__})"
    R = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "tier": "T0", "strategic": "PAUSED (PI item 25)", "bootstrap_weights_check": w_check,
         "sets": json.load(open(os.path.join(HERE, "sets_capladder.report.json")))}
    # ----- R0 status and control 4
    st = json.load(open(f"{FEAT}/R0_STATUS.json")) if os.path.exists(f"{FEAT}/R0_STATUS.json") else {"status": "NOT RUN", "reason": "no status file"}
    c4 = json.load(open(f"{FEAT}/control4_capladder.json")) if os.path.exists(f"{FEAT}/control4_capladder.json") else None
    r0_ran = load("R0") is not None
    r0_adm = bool(r0_ran and c4 is not None and c4["pass"])
    R["r0"] = {"status": st.get("status"), "reason": st.get("reason"), "ran": r0_ran, "admissible": r0_adm, "control4": c4,
               "label": "ADMISSIBLE" if r0_adm else ("NOT COMPARABLE (control 4 failed)" if r0_ran and c4 is not None else "NOT RUN" + (f": {st.get('reason')}" if st.get("reason") else ""))}
    # ----- rungs
    rg, raw = {}, {}
    for nm in NAMES_AB:
        z = load(nm)
        if z is not None:
            raw[nm] = z
            rg[nm] = CL.Rung(z["sse_e"], z["d_e"], W)
    G = {}
    for d in (655360, 163840, 40960, 3072, 1024, 384, 128):
        z = load(f"G_{d}")
        if z is not None:
            G[d] = CL.Rung(z["sse_e"], z["d_e"], W)
    pairs = [(a, b) for a, b in pairs_all("R0" in rg) if a in rg and b in rg]
    # ----- control 1
    cal = CL.calibration_share(rg, pairs) if rg else {}
    cal_seed = {}
    for sd in SEEDS:
        sub = {k: v for k, v in rg.items() if k.endswith(sd) or "_" not in k or k in ("R0", "R1", "R2", "R2m")}
        pp = [(a, b) for a, b in pairs if b.endswith(sd)]
        if sub and pp:
            cal_seed[sd] = CL.calibration_share(sub, pp)
    c1_pass = bool(cal) and all(cal[k]["share"] <= 0.05 for k in cal)
    # ----- control 2
    c2 = {}
    for d, g in G.items():
        ok, info = CL.readable_row(g)
        c2[str(d)] = {"nmse": float(g.point[0]), "readable": bool(ok), **info, "rungs": [k for k in rg if DIMS.get(k.split("_")[0], -1) == d or (k.startswith(("R3_r", "R4_r")) and (3072 if k.startswith("R3") else 384) == d)]}
    need = {d for k in rg for d in [DIMS.get(k.split("_")[0], 0) if not k.startswith(("R3_r", "R4_r")) else (3072 if k.startswith("R3") else 384)]}
    c2_missing = sorted(d for d in need if d not in G)
    c2_pass = bool(c2) and not c2_missing and not any(v["readable"] for v in c2.values())
    # ----- control 3 (the test)
    tdone = os.path.join(HERE, "TEST_CAPLADDER.done")
    tlog = open(os.path.join(HERE, "TEST_CAPLADDER.log"), encoding="utf-8", errors="replace").read() if os.path.exists(os.path.join(HERE, "TEST_CAPLADDER.log")) else ""
    summ = [l for l in tlog.split(chr(10)) if (" passed" in l or " failed" in l) and " in " in l]
    c3_pass = bool(os.path.exists(tdone) and open(tdone).read().strip().startswith("rc=0") and summ and " passed" in summ[-1] and "failed" not in summ[-1] and "error" not in summ[-1])
    valid = bool(c1_pass and c2_pass and c3_pass)
    R["controls"] = {"c1_calibration": {"pass": c1_pass, "all": cal, "by_seed": cal_seed, "bar": "at most 5 % separated per row kind"},
                     "c2_dimension": {"pass": c2_pass, "rows": c2, "missing_dims": c2_missing},
                     "c3_planted": {"pass": bool(c3_pass), "log_tail": summ[-1].strip("= ") if summ else "no log"},
                     "c4_r0_identity": c4, "valid": valid}
    # ----- readability (lead_gap) and the LOSS rows
    rows = {}
    for nm, g in rg.items():
        ok, info = CL.readable_row(g)
        lo, hi = np.percentile(g.boot[:, 0], [2.5, 97.5])
        rows[nm] = {"n": nA, "nmse": float(g.point[0]), "lo": float(lo), "hi": float(hi), "readable": bool(ok), **info,
                    "log_gap": {"nmse": float(g.point[220]), "lo": float(np.percentile(g.boot[:, 220], 2.5)), "hi": float(np.percentile(g.boot[:, 220], 97.5))} if g.point.shape[0] > 220 else None}
    R["rungs"] = rows
    loss_rows = {}
    for a, b in pairs:
        ok, info = CL.loss_row(rg[a], rg[b])
        dlo, dhi = np.percentile(rg[b].boot[:, 0] - rg[a].boot[:, 0], [2.5, 97.5])
        loss_rows[f"{a}->{b}"] = {"delta": info["delta"], "lo": float(dlo), "hi": float(dhi), "loss": bool(ok), "ref_p97_5": info["ref_p97_5"], "lb": info["lb"]}
    R["loss"] = loss_rows
    # ----- outcomes (IMPL_CHOICES section 8)
    rd = {k: v["readable"] for k, v in rows.items()}
    ls = {k: v["loss"] for k, v in loss_rows.items()}
    dl = {k: v["delta"] for k, v in loss_rows.items()}
    g = lambda d, k: d.get(k, None)
    seeds_ok = all(f"R3_{sd}" in rd and f"R4_{sd}" in rd for sd in SEEDS)
    out = {}
    if r0_adm:
        out["O-ENC"] = {"holds": (not rd["R0"]), "read_at": "R0"}
    else:
        out["O-ENC"] = {"holds": (not rd["R1"]) if "R1" in rd else None, "read_at": "R1 (R0 " + R["r0"]["label"] + ")"}
    out["O-POOL8"] = {"holds": (rd["R0"] and ls.get("R0->R1", False)) if r0_adm and "R0->R1" in ls else None, "delta": g(dl, "R0->R1")}
    out["O-POOL4"] = {"holds": (rd["R1"] and ls["R1->R2"]) if "R1->R2" in ls and "R1" in rd else None, "delta": g(dl, "R1->R2")}
    if seeds_ok and "R2" in rd and all(f"R2->R3_{sd}" in ls and f"R2->R4_{sd}" in ls for sd in SEEDS):
        per = {sd: {"R2->R3": ls[f"R2->R3_{sd}"], "R2->R4": ls[f"R2->R4_{sd}"]} for sd in SEEDS}
        net = rd["R2"] and all(per[sd]["R2->R3"] or per[sd]["R2->R4"] for sd in SEEDS)
        sizes = [max([dl[f"R2->{x}_{sd}"] for x in ("R3", "R4") if ls[f"R2->{x}_{sd}"]] or [0.0]) for sd in SEEDS]
        out["O-NET"] = {"holds": bool(net), "per_seed_steps": per, "delta": float(min(sizes)) if net else None}
        allp = ["R2->R3", "R2->R4"]
        dif = rd["R2"] and all(not rd[f"R4_{sd}"] for sd in SEEDS) and all(
            not any(ls.get(f"{a}_{sd}", False) for a in ("R2->R3", "R2->R4", "R2->L1", "L1->L2", "L2->L3")) for sd in SEEDS)
        out["O-DIFFUSE"] = {"holds": bool(dif)}
        out["O-NONE"] = {"holds": bool(all(rd[f"R4_{sd}"] for sd in SEEDS))}
    else:
        for k in ("O-NET", "O-DIFFUSE", "O-NONE"):
            out[k] = {"holds": None, "note": "a rung or step is missing"}
    trig = [k for k, v in out.items() if v["holds"]]
    ranked = sorted([k for k in trig if out[k].get("delta") is not None], key=lambda k: -out[k]["delta"])
    R["outcomes"] = out
    R["triggered"] = {"list": trig, "ranked_by_separated_delta_nmse": ranked}
    # ----- beside rows
    be = {}
    yD, posD, ypred_fold = s["present"].astype(float), s["posD"], s["foldD"]
    for nm in ("R0", "R1", "R2", "R2m", "R3_s0", "R3_s1", "R4_s0", "R4_s1"):
        z = raw.get(nm)
        if z is None:
            continue
        row = {}
        row["log_gap"] = rows[nm]["log_gap"]
        if "B_sse_e" in z.files:
            p, b = CL.nmse_stats(z["B_sse_e"], z["B_d_e"], W)
            for i, q in enumerate(("lead_closing", "ttc")):
                row[q] = {"nmse": float(p[i]), "lo": float(np.percentile(b[:, i], 2.5)), "hi": float(np.percentile(b[:, i], 97.5))}
            p, b = CL.nmse_stats(z["C_sse_e"], z["C_d_e"], W)
            row["gap_le_30"] = {"nmse": float(p[0]), "lo": float(np.percentile(b[:, 0], 2.5)), "hi": float(np.percentile(b[:, 0], 97.5))}
            p, b = CL.nmse_stats(z["mlp_sse_e"], z["mlp_d_e"], W)
            row["mlp_pca256_lead_gap"] = {"nmse": float(p[0]), "lo": float(np.percentile(b[:, 0], 2.5)), "hi": float(np.percentile(b[:, 0], 97.5))}
        zd = os.path.join(RDIR, f"{nm}_D.npz")
        if os.path.exists(zd):
            pr = np.load(zd)["pred"] >= 0.5
            f = lambda m: np.bincount(posD, weights=m.astype(float), minlength=n_ep)
            TP, P, TN, N = f(pr & (yD == 1)), f(yD == 1), f(~pr & (yD == 0)), f(yD == 0)
            ba = lambda w: 0.5 * ((w @ TP) / (w @ P) + (w @ TN) / (w @ N)) - 0.5
            pt = 0.5 * (TP.sum() / P.sum() + TN.sum() / N.sum()) - 0.5
            bs = 0.5 * ((W @ TP) / (W @ P) + (W @ TN) / (W @ N)) - 0.5
            row["lead_present_bal_acc_minus_0.5"] = {"score": float(pt), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5)), "n": nD}
        be[nm] = row
    R["beside"] = be
    json.dump(R, open(os.path.join(OUT, "RESULT_CAPLADDER.json"), "w", encoding="utf-8"), indent=1, default=float)
    write_md(R, rows, loss_rows, out, c2, cal, cal_seed, be, valid, t0)
    print(f"[report] valid={valid} c1={c1_pass} c2={c2_pass} c3={c3_pass} r0={R['r0']['label']} triggered={trig} {time.time() - t0:.0f}s", flush=True)


def write_md(R, rows, loss_rows, out, c2, cal, cal_seed, be, valid, t0):
    C = R["controls"]
    L = []
    w = L.append
    fig = os.path.join(HERE, "fig", "ladder_lead_gap.png")
    w("# RESULT_CAPLADDER: where does the lead gap get lost between DINOv3 patch tokens and the world-model levels of M2b?\n")
    w(f"Card `../PREREG_V8F_CAPLADDER.md` (frozen prefix 7,830 bytes, sha256 `e17a4d90...`), `IMPL_CHOICES_CAPLADDER.md` (sha256 recorded in `IMPL_CHOICES_CAPLADDER.sha256`). Tier **T0**. STRATEGIC: PAUSED (PI item 25). Eval-141 frames; deciding target `lead_gap` on {R['sets']['nA']:,} frames of {R['sets']['A_episodes']} episodes; 5 outer episode folds; dual ridge; bootstrap B 2,000 over 141 episodes (weights {R['bootstrap_weights_check']}). Written {R['time']}.\n".replace("{R['sets']['nA']:,}", f"{R['sets']['nA']:,}").replace("{R['sets']['A_episodes']}", str(R['sets']['A_episodes'])).replace("{R['bootstrap_weights_check']}", R['bootstrap_weights_check']))
    c1, c2d, c3 = C["c1_calibration"], C["c2_dimension"], C["c3_planted"]
    w(f"## Banner: {'VALID' if valid else 'INVALID'}\n")
    w("| control | result | detail |\n|---|---|---|")
    if c1["all"]:
        w(f"| 1 calibration (20 deranged draws, at most 5 % separated per row kind) | {'pass' if c1['pass'] else 'FAIL'} | READABLE rows {c1['all']['READABLE']['separated']}/{c1['all']['READABLE']['rows']} ({100 * c1['all']['READABLE']['share']:.1f} %), LOSS rows {c1['all']['LOSS']['separated']}/{c1['all']['LOSS']['rows']} ({100 * c1['all']['LOSS']['share']:.1f} %) |")
    else:
        w("| 1 calibration | NOT RUN | no rung result |")
    w(f"| 2 dimension (Gaussian n x d, seed 0, must not be READABLE) | {'pass' if c2d['pass'] else 'FAIL'} | " + "; ".join(f"d {int(d):,}: NMSE {fmt(v['nmse'])}, bootstrap upper {fmt(v['ub'])}, reference 2.5th {fmt(v['ref_p2_5'])}, {'READABLE' if v['readable'] else 'not readable'}" for d, v in c2.items()) + (f"; missing dims {c2d['missing_dims']}" if c2d['missing_dims'] else "") + " |")
    w(f"| 3 planted (synthetic a, b, c and the regression arms) | {'pass' if c3['pass'] else 'FAIL'} | `TEST_CAPLADDER.log`: {c3['log_tail']} |")
    r0 = R["r0"]
    w(f"| 4 R0 identity (decides only R0) | {r0['label']} | " + (f"8x20: {100 * r0['control4']['share_ok_8x20']:.2f} % of {r0['control4']['n_bank_frames']:,} bank frames within 0.08 (median {r0['control4']['err_8x20']['p50']:.4f}, 99th {r0['control4']['err_8x20']['p99']:.4f}, max {r0['control4']['err_8x20']['p100']:.4f}); 4x10: {100 * r0['control4']['share_ok_4x10']:.2f} % (median {r0['control4']['err_4x10']['p50']:.4f}, max {r0['control4']['err_4x10']['p100']:.4f}); after the fp8 cast the pooled field equals the cache bit for bit on {100 * r0['control4']['frames_fully_bit_equal_8x20']:.1f} % (8x20) and {100 * r0['control4']['frames_fully_bit_equal_4x10']:.1f} % (4x10) of frames |" if r0["control4"] else f"{r0.get('reason')} |"))
    w("")
    if not valid:
        w("**INVALID: only the control rows are written.** Cheapest next lever: " + ("control 1 failed (" + ", ".join(f"{k} {100 * v['share']:.1f} %" for k, v in c1["all"].items() if v["share"] > 0.05) + "): inspect which rungs / pairs are separated in the calibration columns of `rungs/*.npz` (columns 200 to 219) and compare the bootstrap-only share" if not c1["pass"] and c1["all"] else "") + ("control 2 failed: the Gaussian of the listed d is READABLE; check the standardisation of that rung and the reference" if not c2d["pass"] else "") + ("control 3 failed: see `TEST_CAPLADDER.log`" if not c3["pass"] else "") + ".\n")
        open(os.path.join(OUT, "RESULT_CAPLADDER.md"), "w", encoding="utf-8").write("\n".join(L))
        return
    w("## Readability of `lead_gap` at every rung (NMSE; 1 = the fit-fold mean; READABLE = bootstrap upper bound < 1 AND point below the 2.5th percentile of 199 refit-under-permutation draws)\n")
    w("| rung | what | d | NMSE | 95 % CI | upper < 1 | reference 2.5th | label |\n|---|---|---|---|---|---|---|---|")
    desc = {"R0": "patch tokens 16 x 40", "R1": "pool 8 x 20 (cached)", "R2": "pool 4 x 10 (the model's input)", "R2m": "mean token (beside)"}
    for nm in ["R0", "R1", "R2", "R2m"]:
        if nm in rows:
            v = rows[nm]
            lab = ("READABLE" if v["readable"] else "UNREADABLE") + ("" if nm != "R0" or r0["admissible"] else " (R0 " + r0["label"] + ")")
            w(f"| {nm} | {desc[nm]} | {DIMS[nm]:,} | {fmt(v['nmse'])} | [{fmt(v['lo'])}, {fmt(v['hi'])}] | {'yes' if v['hi'] < 1 else 'no'} | {fmt(v['ref_p2_5'])} | {lab} |")
        elif nm == "R0":
            w(f"| R0 | {desc[nm]} | {DIMS[nm]:,} | | | | | {r0['label']} |")
    w("")
    w("| rung | what | d | seed | NMSE | 95 % CI | upper < 1 | reference 2.5th | label per seed | combined |\n|---|---|---|---|---|---|---|---|---|---|")
    mdesc = {"R3": "readers S, T, O", "R4": "levels L1, L2, L3", "R3s": "read_s (beside)", "R3t": "read_t (beside)", "R3o": "read_o (beside)", "L1": "level 1 (beside)", "L2": "level 2 (beside)", "L3": "level 3 (beside)"}
    for b in ("R3", "R4", "R3s", "R3t", "R3o", "L1", "L2", "L3"):
        if all(f"{b}_{sd}" in rows for sd in SEEDS):
            lab = [rows[f"{b}_{sd}"]["readable"] for sd in SEEDS]
            for i, sd in enumerate(SEEDS):
                v = rows[f"{b}_{sd}"]
                w(f"| {b} | {mdesc[b]} | {DIMS[b]:,} | {sd} | {fmt(v['nmse'])} | [{fmt(v['lo'])}, {fmt(v['hi'])}] | {'yes' if v['hi'] < 1 else 'no'} | {fmt(v['ref_p2_5'])} | {'READABLE' if lab[i] else 'UNREADABLE'} | {comb(*lab) if i == 0 else ''} |")
    for b in ("R3", "R4"):
        for r in (100, 101):
            nm = f"{b}_r{r}"
            if nm in rows:
                v = rows[nm]
                w(f"| {b} re-initialised (beside) | seed {r} | {DIMS[b]:,} | | {fmt(v['nmse'])} | [{fmt(v['lo'])}, {fmt(v['hi'])}] | {'yes' if v['hi'] < 1 else 'no'} | {fmt(v['ref_p2_5'])} | {'READABLE' if v['readable'] else 'UNREADABLE'} | |")
    w("\nFigure: " + (f"`fig/ladder_lead_gap.png`" if os.path.exists(fig) else "not written") + " (NMSE per rung with the bootstrap band, both seeds).\n")
    w("## LOSS table per step (Delta NMSE = NMSE(after) - NMSE(before); LOSS = paired bootstrap lower bound > 0 AND point above the 97.5th percentile of the 199 reference Deltas)\n")
    w("| step | seed | Delta NMSE | 95 % paired CI | reference 97.5th | label | in an outcome |\n|---|---|---|---|---|---|---|")
    sx = lambda x: x[:-3] if x[-3:] in ("_s0", "_s1") else x
    for k, v in loss_rows.items():
        a, b = k.split("->")
        sd = b[-2:] if b[-3:] in ("_s0", "_s1") else ""
        a0, b0 = sx(a), sx(b)
        used = "yes" if k in ("R0->R1", "R1->R2") or (a0 == "R2" and b0 in ("R3", "R4", "L1")) or (a0, b0) in (("L1", "L2"), ("L2", "L3")) else "beside"
        w(f"| {a0} -> {b0} | {sd} | {fmt(v['delta'])} | [{fmt(v['lo'])}, {fmt(v['hi'])}] | {fmt(v['ref_p97_5'])} | {'LOSS' if v['loss'] else 'no loss'} | {used} |")
    for b in ("R3_", "R4_"):
        pass
    w("\nSeed agreement for the model steps: " + "; ".join(f"R2 -> {x}: {combl(loss_rows.get(f'R2->{x}_s0', {}).get('loss', False), loss_rows.get(f'R2->{x}_s1', {}).get('loss', False))}" for x in ("R3", "R4") if f"R2->{x}_s0" in loss_rows) + ". R3 -> R4 is not a step of the real dataflow (L1 reads R2; IMPL_CHOICES section 3) and is used in no outcome.\n")
    w("## Outcomes the card commits (read on `lead_gap`)\n")
    w("| outcome | holds | note |\n|---|---|---|")
    for k, v in out.items():
        h = "yes" if v["holds"] else ("no" if v["holds"] is False else "not evaluable")
        note = []
        if v.get("delta") is not None:
            note.append(f"separated Delta NMSE {fmt(v['delta'])}")
        if v.get("read_at"):
            note.append("read at " + v["read_at"])
        if v.get("per_seed_steps"):
            note.append("per seed (LOSS of R2->R3 / R2->R4): " + "; ".join(f"{sd}: {st['R2->R3']} / {st['R2->R4']}" for sd, st in v["per_seed_steps"].items()))
        if v.get("note"):
            note.append(v["note"])
        w(f"| {k} | {h} | {'; '.join(note)} |")
    tr = R["triggered"]
    w("\n**Triggered:** " + (", ".join(tr["list"]) if tr["list"] else "none") + ". Ranked by the separated Delta NMSE (Rule Zero 5): " + (" > ".join(f"{k} ({fmt(out[k]['delta'])})" for k in tr["ranked_by_separated_delta_nmse"]) if tr["ranked_by_separated_delta_nmse"] else "no ranked step") + ".\n")
    levers = {"O-ENC": "E8-RES (higher-resolution pass on the lead frames, gap bins beside) with a label check of `lead_gap` against its cuboid source", "O-POOL8": "E8 UNPOOL with patch tokens or learned attention pooling",
              "O-POOL4": "E8 UNPOOL with the 8 x 20 grid as the model's input", "O-NET": "E8 SCNAUX: a gap and TTC auxiliary target from the cuboids at training (inference stays vision-only)",
              "O-DIFFUSE": "E8 SCNAUX, labelled diffuse", "O-NONE": "re-read IMAG1B's IA-2 gap row on frames before any arm"}
    for k in tr["list"]:
        w(f"* {k} triggers: {levers[k]}.")
    w("")
    w("## Beside rows (no bar; point and 95 % episode-cluster interval; deciding rungs and R2m only)\n")
    w("| rung | log(max(gap, 0.1 m)) NMSE | `lead_gap` <= 30 m NMSE | `lead_closing` NMSE | `ttc` NMSE | `lead_gap` MLP on PCA-256 NMSE | `lead_present` balanced accuracy - 0.5 |\n|---|---|---|---|---|---|---|")
    ci = lambda v: "n/a" if not v else f"{fmt(v['nmse'])} [{fmt(v['lo'])}, {fmt(v['hi'])}]"
    for nm, r_ in be.items():
        lp = r_.get("lead_present_bal_acc_minus_0.5")
        w(f"| {nm} | {ci(r_.get('log_gap'))} | {ci(r_.get('gap_le_30'))} | {ci(r_.get('lead_closing'))} | {ci(r_.get('ttc'))} | {ci(r_.get('mlp_pca256_lead_gap'))} | {('%s [%s, %s] (n %d, balanced)' % (fmt(lp['score']), fmt(lp['lo']), fmt(lp['hi']), lp['n'])) if lp else 'n/a'} |")
    w("\n## Disclosures\n* 5 outer folds by `sha256(episode)[:8] % 5`: IMAG1B has only a two-fold split (IMPL_CHOICES section 1). `lead_present` is scored on a balanced 4,978-frame set at every rung, not on all frames. Beside targets on subsets use set A's standardisation. R3 carries the model's ego / nav conditioning as well as the scene tokens. Tier T0: a representation diagnostic, never a driving claim.\n* A synthetic smoke before the statistics is disclosed in `IMPL_ADDENDUM_CAPLADDER.md`.\n")
    open(os.path.join(OUT, "RESULT_CAPLADDER.md"), "w", encoding="utf-8").write("\n".join(L))


if __name__ == "__main__":
    main()
