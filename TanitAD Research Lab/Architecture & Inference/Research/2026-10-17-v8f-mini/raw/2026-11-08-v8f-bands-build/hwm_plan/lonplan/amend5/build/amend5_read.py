"""AMEND_HWMPLAN_5 readout (sections 5, 6, 7), computed MECHANICALLY from the arm folders. Nothing here is tuned; every constant is the card's.
Pure logic (importable, tested with synthetic inputs): sep_better, worse, chassis_label, combine, ceil_oracle, finalize, next_step, check_score_identity, multiplicity.
Commands:
  train  --chassis k     TRAIN-stage report (beside, in-sample, NEVER a result): every RA arm vs its base and vs HOLD on the 564 TRAIN windows, clip-cluster bootstrap over the 141 clips; C-SHUF sign, C-PLANT (ii)
  score                  contrasts (contrast_l1full_strat.py = the strategic-audit reader, B 2,000 seed 0), labels, the four families -> <ROOT>/readout/RESULT_AMEND5.json + .md
Estimator: paired episode-cluster bootstrap (answers 'another draw of EPISODES' only). F_I answers 'another INFERENCE run'. Tier T0 (oracle goal): a world-model diagnostic, never driving performance.
Labels (section 6): RAL1-<outcome> / RAL2-<outcome>; a label that differs between the chassis reads SEEDS-DISAGREE and the LESS favourable label's next step applies."""
import argparse, json, os, subprocess, sys
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import amend5_stage as ST
sys.path.insert(0, ST.LONPLAN_BUILD)
import lonplan_common as _C            # puts the code of record on sys.path (hwm_eval, hwm_plan, ...)
import numpy as np

ORDER = ("CEILING", "NARROWS", "NULL", "HARM")           # most to least favourable
D_MIN = 0.10


# ---------------------------------------------------------------------------------------------- pure logic
def sep_better(row, floor, seed1_delta, drop=()):
    """'Separated-better beyond floor': CI wholly below 0, |delta| > F_I, and the seed-1 point delta also below 0."""
    return bool(("ci" in drop or row["hi"] < 0) and ("floor" in drop or abs(row["delta"]) > floor) and ("seed1" in drop or seed1_delta < 0))


def worse(row, floor, drop=()):
    """'WORSE': CI wholly above 0 and delta > F_I."""
    return bool(("ci" in drop or row["lo"] > 0) and ("floor" in drop or row["delta"] > floor))


def chassis_label(c, drop=()):
    """c: dH, dB, dM (rows: delta, lo, hi; dM may be None), their floors dH_F / dB_F / dM_F and seed-1 points dH_seed1 / dB_seed1 / dM_seed1, gx (X - BASE xtrack 0-2 s row) with gx_F, and D.
    Precedence HARM > CEILING > NARROWS > NULL. HARM: dB WORSE or the guard G_x WORSE. CEILING: dH separated-better, D >= 0.10 (G_x not WORSE already by precedence). NARROWS: dB separated-better;
    qualifier INFORMATIVE if dM separated-better beyond floor, else MEAN-ONLY. `drop`: names among harm_dB_ci / harm_dB_floor / harm_guard / ceil_ci / ceil_floor / ceil_seed1 / ceil_D / narrows_ci / narrows_floor / narrows_seed1 /
    inf_ci / inf_floor / inf_seed1: that ONE condition is ignored (the mutation hook of the tests)."""
    dr = lambda pre: tuple(x[len(pre):] for x in drop if x.startswith(pre))
    if worse(c["dB"], c["dB_F"], dr("harm_dB_")) or ("harm_guard" not in drop and worse(c["gx"], c["gx_F"])):
        return dict(label="HARM", qualifier=None)
    if sep_better(c["dH"], c["dH_F"], c["dH_seed1"], dr("ceil_")) and ("ceil_D" in drop or c["D"] >= D_MIN):
        return dict(label="CEILING", qualifier=None)
    if sep_better(c["dB"], c["dB_F"], c["dB_seed1"], dr("narrows_")):
        inf = c.get("dM") is not None and sep_better(c["dM"], c["dM_F"], c["dM_seed1"], dr("inf_"))
        return dict(label="NARROWS", qualifier="INFORMATIVE" if inf else "MEAN-ONLY")
    return dict(label="NULL", qualifier=None)


def combine(labels, order=ORDER):
    """Both chassis equal -> that label; otherwise SEEDS-DISAGREE with the LESS favourable label as the effective one."""
    ls = list(labels)
    if len(set(ls)) == 1:
        return dict(label=ls[0], effective=ls[0], seeds_disagree=False)
    return dict(label="SEEDS-DISAGREE", effective=max(ls, key=order.index), seeds_disagree=True)


def combine_qualifier(per):
    """NARROWS qualifiers: INFORMATIVE only if INFORMATIVE at every chassis that reads NARROWS."""
    qs = [q for q in per if q is not None]
    return None if not qs else ("INFORMATIVE" if all(q == "INFORMATIVE" for q in qs) else "MEAN-ONLY")


def ceil_oracle(per):
    """CEIL-ORACLE holds iff RA-ORACLE - HOLD is separated-better at BOTH chassis. per: {k: dict(row, F, seed1)}."""
    return bool(per) and all(sep_better(v["row"], v["F"], v["seed1"]) for v in per.values())


def read_limited(effective, ceil_ok):
    """A NULL, NARROWS or HARM row with CEIL-ORACLE failed is READ-LIMITED (section 6)."""
    return bool(effective in ("NULL", "NARROWS", "HARM") and not ceil_ok)


def next_step(effective, ceil_ok, base):
    """The committed next lever (section 6), as a key + one line; the MM writes the reading."""
    if effective == "CEILING":
        return "CEILING: (1) MM candidate level-1 longitudinal cost after the multiplicity guard; (2) the 2-6 s band, a K = 30 readout on the same TRAIN clips; (3) the T1-kin read, blocked on a goal head (Q2a)"
    if effective == "HARM":
        return "HARM: first C-OFF5, C-YAW-ID, C-GRAD, C-PLANT and C-RA-FIT are re-run on the SCORE argv (wiring first), then the CEIL-ORACLE row as NULL: " + next_step("NULL", ceil_ok, base)
    if not ceil_ok:
        return "READ-LIMITED: (1) R35's response along RA's direction (ACTRSP Option B / action-sensitivity training; blocked on box 2 and the PI, Q2b); (2) RA's precision (more TRAIN clips' level-1 latents from box 2; blocked, PI P3)"
    if effective == "NARROWS":
        return "NARROWS, CEIL-ORACLE holds: the source's 2 s intent is the limit: (a) X + REG with lambda re-selected on TRAIN (AMEND_HWMPLAN_3 section 3) ; (b) action-sensitivity training (Q2b)"
    other = {"P": "P23", "P23": "P", "A0": None}[base]
    return "NULL, CEIL-ORACLE holds: the hierarchy's 2 s intent is what fails; the lever is the source: " + (f"RA-L2 under {other}, " if other else "") + "then action-sensitivity training (PI, Q2b)"


def check_score_identity(js):
    """C-YAW-ID / C-BAND-ID on SCORE: X - BASE reads exactly 0.0 [0, 0] for yaw_mae and head_err in every band and for acc_mae in 2-6 s (per-knot). js: the X - BASE contrast file."""
    bad, n = [], 0
    for r in js["rows"]:
        if r["a"] == "A" and r["b"] == "B" and ((r["metric"] in ("yaw_mae", "head_err")) or (r["metric"] == "acc_mae" and r["band"] == "2-6s")):
            n += 1
            if not (r["delta"] == 0.0 and r["lo"] == 0.0 and r["hi"] == 0.0):
                bad.append((r["band"], r["metric"], r["delta"], r["lo"], r["hi"]))
    return dict(ok=bool(n >= 7 and not bad), n_cells=n, nonzero=bad)


def multiplicity(effective_by_arm, abs_dH_below_2F):
    """Section 6: under B1 / B2 two lever arms read against one bar. A CEILING at ONE arm only changes the solver of record after the NULL_s2 / NULL_s3 read agrees; the same read is needed if |dH| < 2 F_I at either chassis."""
    ceil = [a for a, e in effective_by_arm.items() if e == "CEILING"]
    return dict(two_arms=len(effective_by_arm) == 2, ceiling_arms=ceil, guard_applies=bool(len(effective_by_arm) == 2 and len(ceil) == 1),
                needs_null_replicates=bool(ceil and any(abs_dH_below_2F.get(a, False) for a in ceil)))


def finalize(lever_inputs, oracle_inputs, controls_ok, extra_invalid=()):
    """lever_inputs: {arm: {k: chassis_label input}}; oracle_inputs: {k: dict(row, F, seed1)}. -> the labels block."""
    ceil_ok = ceil_oracle(oracle_inputs)
    out = dict(CEIL_ORACLE=ceil_ok, CEIL_ORACLE_per_chassis={k: sep_better(v["row"], v["F"], v["seed1"]) for k, v in oracle_inputs.items()}, arms={})
    eff = {}
    for arm, per in lever_inputs.items():
        ks = sorted(per)
        pl = {k: chassis_label(per[k]) for k in ks}
        comb = combine([pl[k]["label"] for k in ks])
        q = combine_qualifier([pl[k]["qualifier"] for k in ks])
        name = ("RAL2" if arm.startswith("RAL2") else "RAL1")
        label = comb["label"] if controls_ok and not extra_invalid else "INVALID"
        rl = read_limited(comb["effective"], ceil_ok)
        out["arms"][arm] = dict(per_chassis={k: pl[k]["label"] for k in ks}, qualifier=q, **{x: comb[x] for x in ("label", "effective", "seeds_disagree")}, name=f"{name}-{label}", effective_name=f"{name}-{comb['effective']}",
                                READ_LIMITED=rl, hold_collapsed={k: bool(per[k]["D"] < D_MIN) for k in ks}, abs_dH_below_2F={k: bool(abs(per[k]["dH"]["delta"]) < 2 * per[k]["dH_F"]) for k in ks},
                                next_step=next_step(comb["effective"], ceil_ok, per[ks[0]].get("base", "A0")))
        eff[arm] = comb["effective"]
    out["multiplicity"] = multiplicity(eff, {a: any(v["abs_dH_below_2F"].values()) for a, v in out["arms"].items()})
    if not controls_ok or extra_invalid:
        out["INVALID"] = "; ".join(["a section 8 control is RED"] * (not controls_ok) + list(extra_invalid))
    return out


# ---------------------------------------------------------------------------------------------- extraction from the contrast JSONs
def find_row(js, kind, family, band, metric, a, b=None, arm=None):
    for r in js["rows" if kind == "contrast" else "replicate"]:
        if kind == "contrast":
            if (r["family"], r["band"], r["metric"], r["a"], r["b"]) == (family, band, metric, a, b):
                return r
        elif (r["family"], r["band"], r["metric"], r["arm"]) == (family, band, metric, arm):
            return r
    raise KeyError((kind, family, band, metric, a, b, arm))


def cell(js, family, band, metric, ref="B"):
    """From a contrast file (a = X, b = ref arm): X - HOLD (dH), X - ref (dr), the floor F = max(|X seed1 - seed0|, |ref seed1 - seed0|) and the seed-1 points."""
    dH = find_row(js, "contrast", family, band, metric, "A", "HOLD")
    d0 = find_row(js, "contrast", family, band, metric, "A", "B")
    repX = find_row(js, "rep", family, band, metric, None, arm="A")["delta"]
    repB = find_row(js, "rep", family, band, metric, None, arm="B")["delta"]
    return dict(dH=dH, d0=d0, F=max(abs(repX), abs(repB)), dH_seed1=dH["delta"] + repX, d0_seed1=d0["delta"] + repX - repB, repX=repX, repB=repB)


def lever_input(js_xb, js_xm, D, base):
    """js_xb: the X - BASE file; js_xm: the X - MEAN-src file. All cells 0-2 s: acc_mae (deciding) and xtrack (the guard)."""
    c0 = cell(js_xb, "LONGITUDINAL", "0-2s", "acc_mae")
    cg = cell(js_xb, "LATERAL", "0-2s", "xtrack")
    cm = cell(js_xm, "LONGITUDINAL", "0-2s", "acc_mae")
    return dict(dH=c0["dH"], dH_F=c0["F"], dH_seed1=c0["dH_seed1"], dB=c0["d0"], dB_F=c0["F"], dB_seed1=c0["d0_seed1"], dM=cm["d0"], dM_F=c0["F"], dM_seed1=cm["d0"]["delta"] + cm["repX"] - cm["repB"],
                gx=cg["d0"], gx_F=cg["F"], D=D, base=base)


# ---------------------------------------------------------------------------------------------- the run side
def run_contrast(windows_dir, a, b, out, rep_a=None, rep_b=None, n_boot=2000, py=sys.executable):
    cmd = [py, os.path.join(ST.LONPLAN_BUILD, "contrast_l1full_strat.py"), "--windows", windows_dir, "--a", a, "--b", b, "--out", out, "--n-boot", str(n_boot)]
    for k, v in (("--rep-a", rep_a), ("--rep-b", rep_b)):
        if v:
            cmd += [k, v]
    e = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run(cmd, env=e, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise RuntimeError(f"contrast {os.path.basename(out)} rc {r.returncode}: {r.stderr[-800:]}")
    return json.load(open(out, encoding="utf-8"))


def assemble_2_0(plans_npz, out_npz):
    """P2-0 (zero compute): knots 1-10 = level 2's own chosen primitives, knots 11-30 = the base's."""
    with np.load(plans_npz, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    plan = d["plan_raw"].copy()
    plan[:, :10] = d["l2_prim_raw"][:, :10]
    np.savez(out_npz, plan_raw=plan)
    return out_npz


def arm_sets(solver):
    """-> (base, levers [arm...], mean map, committed arm, beside list)."""
    if solver["branch"] == "B3":
        return "A0", ["RAL1"], {"RAL1": "MEANL1"}, "RAL1", ["RAL2"]
    return solver["base"], ["RAL2", "RAL1"], {"RAL2": "MEANL2", "RAL1": "MEANL1"}, "RAL2", []


def score(root=None, n_boot=2000):
    import lonplan_common as C
    import lonplan_select as S
    import lonplan_diracc as DA
    root = root or ST.ROOT
    solver = ST.read_base_solver()
    base, levers, meanof, committed, beside = arm_sets(solver)
    ro = f"{root}/readout"
    os.makedirs(ro, exist_ok=True)
    res = dict(kind="RESULT_AMEND5", tier="T0", evidence="MEASURED", estimator="paired_episode_cluster_bootstrap", B=n_boot, seed=0, base_solver=dict(base=base, branch=solver["branch"], file_sha256=solver["sha256"]),
               committed=committed, levers=levers, chassis={})
    lever_in, oracle_in, extra = {x: {} for x in levers}, {}, []
    f = lambda d, s=0: f"{d}/plans_hwm_oracle_s{s}.npz"
    for k in ("s0", "s1"):
        Bd, A1d = ST.base_dir("score", base, k), f"{ST.L0}/score/A1_{k}"
        w = f"{Bd}/windows.npz"
        dirs = {a: ST.out_dir("score", a, base, k, root) for a in levers + list(meanof.values()) + ["NEG", "SHUF", "ORACLE", "OFF"]}
        for a, d in dirs.items():
            if C.sha256_file(f"{d}/windows.npz") != C.sha256_file(w):
                raise RuntimeError(f"{a}_{k} windows differ from the base's")
        pairs = {}
        for x in levers:
            m = meanof[x]
            pairs[f"{x}_BASE"] = (f(dirs[x]), f(Bd), f(dirs[x], 1), f(Bd, 1))
            pairs[f"{x}_{m}"] = (f(dirs[x]), f(dirs[m]), f(dirs[x], 1), f(dirs[m], 1))
            pairs[f"{x}_A1"] = (f(dirs[x]), f(A1d), f(dirs[x], 1), f(A1d, 1))
        pairs[f"SHUF_{committed}"] = (f(dirs["SHUF"]), f(dirs[committed]), None, None)
        pairs["SHUF_BASE"] = (f(dirs["SHUF"]), f(Bd), None, None)
        pairs[f"NEG_{committed}"] = (f(dirs["NEG"]), f(dirs[committed]), None, None)
        pairs["ORACLE_BASE"] = (f(dirs["ORACLE"]), f(Bd), f(dirs["ORACLE"], 1), f(Bd, 1))
        if len(levers) == 2:
            pairs["RAL2_RAL1"] = (f(dirs["RAL2"]), f(dirs["RAL1"]), None, None)
        if base in ("P", "P23"):
            p20 = assemble_2_0(f(Bd), f"{ro}/P2-0_{base}_{k}.npz")
            pairs["P2-0_BASE"] = (p20, f(Bd), None, None)
            for x in levers:
                pairs[f"{x}_P2-0"] = (f(dirs[x]), p20, None, None)
        bes = {}
        for x in beside:
            d = ST.out_dir("score", x, "P", k, root)
            if os.path.exists(f"{d}/DONE"):
                bes[x] = d
                pairs[f"{x}_P_BASEP"] = (f(d), f(ST.base_dir("score", "P", k)), f(d, 1), f(ST.base_dir("score", "P", k), 1))
        js = {n: run_contrast(Bd, a, b, f"{ro}/c_{n}_{k}.json", ra, rb, n_boot) for n, (a, b, ra, rb) in pairs.items()}
        sel = json.load(open(f"{ST.L0}/sel_{k}/LAMBDA_SELECTION_R35_{k}.json", encoding="utf-8"))
        std_raw = np.array(sel["refs"]["prior_std_raw"] if "refs" in sel else sel["prior_std_raw"], np.float64)
        wz = np.load(w, allow_pickle=False)
        A, eid = wz["A"].astype(np.float64), wz["eid"]
        Dv = {x: float(S.distance_D(np.load(f(dirs[x]))["plan_raw"].astype(np.float64), A, std_raw)[0]) for x in levers}
        Dv["BASE"] = float(S.distance_D(np.load(f(Bd))["plan_raw"].astype(np.float64), A, std_raw)[0])
        for x in levers:
            lever_in[x][k] = lever_input(js[f"{x}_BASE"], js[f"{x}_{meanof[x]}"], Dv[x], base)
            ident = check_score_identity(js[f"{x}_BASE"])
            if not ident["ok"]:
                extra.append(f"C-YAW-ID / C-BAND-ID on SCORE: {x}_{k} {ident['nonzero'][:2]}")
        cO = cell(js["ORACLE_BASE"], "LONGITUDINAL", "0-2s", "acc_mae")
        Fc = lever_in[committed][k]["dH_F"]
        oracle_in[k] = dict(row=cO["dH"], F=cO["F"], seed1=cO["dH_seed1"])
        shuf = check_shuf(js[f"SHUF_{committed}"])
        if not shuf["ok"]:
            extra.append(f"C-SHUF on SCORE: SHUF - {committed} is not separated-worse at {k}")
        off = {"ok": True}
        with np.load(f(dirs["OFF"])) as zo, np.load(f(Bd)) as zb:
            for key in ("plan_raw", "a1_raw", "cost1"):
                off[key] = bool(np.array_equal(zo[key], zb[key])); off["ok"] = off["ok"] and off[key]
        if not off["ok"]:
            extra.append(f"C-OFF5 on SCORE: RA-OFF differs from the base at {k}")
        arms = [("BASE", f(Bd))] + [(x, f(dirs[x])) for x in levers + list(meanof.values()) + ["NEG", "SHUF", "ORACLE"]] + ([("P2-0", f"{ro}/P2-0_{base}_{k}.npz")] if base in ("P", "P23") else [])
        dir_acc = DA.diracc_table(w, arms, n_boot, 0)
        mech = {}
        for x in levers + ["NEG", "SHUF", "ORACLE"]:
            z = np.load(f(dirs[x]))
            a_raw = z["plan_raw"][:, :10, 0].mean(1) - A[:, 0, 0]
            b_raw = np.load(f(Bd))["plan_raw"][:, :10, 0].mean(1) - A[:, 0, 0]
            mech[x] = dict(DEP_mean_abs_2s_mean_departure=float(np.abs(a_raw).mean()), DEP_base=float(np.abs(b_raw).mean()), HIT_median_final_cost=float(np.median(z["ra_cost"])), HIT_share_hit_B1=float(z["ra_hit_box"].mean()))
        res["chassis"][k] = dict(contrasts={n: f"{ro}/c_{n}_{k}.json" for n in pairs}, D=Dv, dir_acc=dir_acc, mechanism=mech, n_windows=int(len(eid)), n_episodes=int(len(set(eid))), windows_sha256=C.sha256_file(w),
                                 families={n: dict(strategic=js[n]["strategic"], tactical_kappa=js[n]["tactical_kappa"]) for n in (f"{x}_BASE" for x in levers)},
                                 C_SHUF_score=shuf, C_OFF5_score=off, beside={x: pairs.get(f"{x}_P_BASEP") and f"{ro}/c_{x}_P_BASEP_{k}.json" for x in bes},
                                 plan_sha256={a: C.sha256_file(f(d)) for a, d in dirs.items()})
    ctl = json.load(open(f"{root}/controls/CONTROLS.json", encoding="utf-8"))
    tr = {k: json.load(open(f"{root}/readout/TRAIN_STAGE_{k}.json", encoding="utf-8")) for k in ("s0", "s1")}
    ok_train = all(t["checks_ok"] for t in tr.values())
    res["controls_ok"] = bool(ctl["ok"] and ok_train)
    res["labels"] = finalize(lever_in, oracle_in, res["controls_ok"], extra)
    res["train_stage_checks_ok"] = ok_train
    for x, per in lever_in.items():
        res["labels"]["arms"][x]["cells"] = {k: dict(dH=per[k]["dH"], dB=per[k]["dB"], dM=per[k]["dM"], gx=per[k]["gx"], F_I=per[k]["dH_F"], D=per[k]["D"]) for k in per}
    with open(f"{ro}/RESULT_AMEND5.json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(res, fh, indent=1, default=float)
    with open(f"{ro}/RESULT_AMEND5.md", "w", encoding="utf-8", newline="\n") as fh:
        fh.write(render_md(res))
    return res


def check_shuf(js_row_file, separated=True):
    from amend5_controls import check_shuf_sign
    r = find_row(js_row_file, "contrast", "LONGITUDINAL", "0-2s", "acc_mae", "A", "B")
    return check_shuf_sign(r, separated)


# ---------------------------------------------------------------------------------------------- TRAIN stage (beside)
def train_stage(root, k, n_boot=2000):
    import hwm_eval as E
    import hwm_plan as H
    import amend5_controls as K
    w = np.load(ST.windows_of("train", k), allow_pickle=False)
    A, v0, eid = w["A"].astype(np.float64), w["v0"], w["eid"]
    base = "A0"
    arms = {a: ST.out_dir("train", a, base, k, root) for a in ("RAL1", "MEANL1", "NEG", "SHUF", "ORACLE")}
    plans = {"BASE": dict(np.load(ST.base_plan("train", base, k, 0)))}
    for a, d in arms.items():
        plans[a] = dict(np.load(f"{d}/plans_hwm_oracle_s0.npz"))
    M = {x: E.band_metrics(plans[x]["plan_raw"], A, v0, None, None)[0] for x in plans}
    M["HOLD"] = E.band_metrics(H.hold_plan(A[:, 0], 30), A, v0, None, None)[0]
    out = dict(kind="AMEND5_TRAIN_STAGE", chassis=k, tier="T0, TRAIN windows, IN-SAMPLE for R35 and for RA: beside, never a result", n_windows=int(len(eid)), n_clips=int(len(set(eid))), rows=[],
               estimator="paired episode-cluster (here clip-cluster) bootstrap over the TRAIN clips, B 2,000 seed 0")
    for a_, b_ in (("RAL1", "BASE"), ("RAL1", "HOLD"), ("MEANL1", "BASE"), ("MEANL1", "HOLD"), ("RAL1", "MEANL1"), ("NEG", "RAL1"), ("SHUF", "RAL1"), ("SHUF", "BASE"), ("ORACLE", "HOLD"), ("ORACLE", "RAL1"), ("BASE", "HOLD")):
        for fam, mets in E.FAMILIES.items():
            for band in E.BANDS:
                for m in mets + (E.BESIDE if fam == "LATERAL" else ()):
                    out["rows"].append(dict(a=a_, b=b_, family=fam, band=band, metric=m, **E.contrast(M[a_][band][m], M[b_][band][m], eid, n_boot)))
    out["strategic"] = "n/a on TRAIN windows: no navigation-text row joined to TRAIN windows; reported on SCORE only"
    out["headway"] = "n = 0 on TRAIN windows (no lead-state join); reported on SCORE only"
    pick = lambda a_, b_: [r for r in out["rows"] if (r["a"], r["b"], r["band"], r["metric"]) == (a_, b_, "0-2s", "acc_mae")][0]
    shuf = K.check_shuf_sign(pick("SHUF", "RAL1"), separated=False)
    plant = {a: K.check_plant_real(plans[a]["ra_cost"]) for a in ("RAL1", "ORACLE")}
    out["C-SHUF_train_sign"] = shuf
    out["C-PLANT_ii"] = plant
    out["checks_ok"] = bool(shuf["ok"] and all(v["ok"] for v in plant.values()))
    out["mechanism"] = {a: dict(median_final_cost=float(np.median(plans[a]["ra_cost"])), share_hit_B1=float(plans[a]["ra_hit_box"].mean())) for a in ("RAL1", "NEG", "SHUF", "ORACLE")}
    with open(f"{root}/readout/TRAIN_STAGE_{k}.json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, indent=1, default=float)
    return out


# ---------------------------------------------------------------------------------------------- the .md generator
def _f(v, d=4):
    return "n/a" if v is None or (isinstance(v, float) and v != v) else f"{v:+.{d}f}"


def render_md(res):
    L = ["# RESULT_AMEND5 (generated from RESULT_AMEND5.json; the MM writes the reading)", "",
         "Tier T0 (oracle goal): a world-model diagnostic, never driving performance. Estimator: paired episode-cluster bootstrap, B 2,000 seed 0 (another draw of EPISODES only); F_I answers another INFERENCE run.", "",
         f"Base solver: {res['base_solver']['base']} ({res['base_solver']['branch']}). Committed arm: {res['committed']}. CEIL-ORACLE: {res['labels']['CEIL_ORACLE']} (per chassis {res['labels']['CEIL_ORACLE_per_chassis']}).", ""]
    if "INVALID" in res["labels"]:
        L += [f"**INVALID.** {res['labels']['INVALID']}", ""]
    L += ["## Labels (section 6)", "", "| arm | s0 | s1 | label | effective | qualifier | READ-LIMITED | HOLD-collapsed | abs dH < 2 F_I |", "|---|---|---|---|---|---|---|---|---|"]
    for x, lb in res["labels"]["arms"].items():
        L.append(f"| {x} | {lb['per_chassis']['s0']} | {lb['per_chassis']['s1']} | {lb['name']} | {lb['effective_name']} | {lb['qualifier']} | {lb['READ_LIMITED']} | {lb['hold_collapsed']} | {lb['abs_dH_below_2F']} |")
    L += ["", f"Multiplicity guard: {res['labels']['multiplicity']}", ""]
    for x, lb in res["labels"]["arms"].items():
        L += [f"Next step ({x}): {lb['next_step']}", ""]
        for k, c in lb["cells"].items():
            L.append(f"* {x} {k}: dH {_f(c['dH']['delta'])} [{_f(c['dH']['lo'])}, {_f(c['dH']['hi'])}]; dB {_f(c['dB']['delta'])} [{_f(c['dB']['lo'])}, {_f(c['dB']['hi'])}]; dM {_f(c['dM']['delta'])}; guard xtrack {_f(c['gx']['delta'])}; F_I {c['F_I']:.4f}; D {c['D']:.3f}")
        L.append("")
    for k, c in res["chassis"].items():
        L += [f"## Chassis R35_{k}: {c['n_windows']} windows, {c['n_episodes']} episodes (windows sha256 {c['windows_sha256'][:12]})", ""]
        for n, p in c["contrasts"].items():
            js = json.load(open(p, encoding="utf-8"))
            L += [f"### {n} ({os.path.basename(p)}; {js['n_windows']} windows / {js['n_episodes']} episodes)", "", "| family | band | metric | a - b | CI | label | n windows | n episodes |", "|---|---|---|---|---|---|---|---|"]
            for r in js["rows"]:
                if r["a"] == "A" and r["b"] == "B":
                    L.append(f"| {r['family']} | {r['band']} | {r['metric']} | {_f(r['delta'])} | [{_f(r['lo'])}, {_f(r['hi'])}] | {r['label']} | {r['n_windows']} | {r['n_episodes']} |")
            if n.endswith("_BASE"):
                L += ["", "a - HOLD:", "", "| family | band | metric | a - HOLD | CI | label | n |", "|---|---|---|---|---|---|---|"]
                for r in js["rows"]:
                    if r["a"] == "A" and r["b"] == "HOLD":
                        L.append(f"| {r['family']} | {r['band']} | {r['metric']} | {_f(r['delta'])} | [{_f(r['lo'])}, {_f(r['hi'])}] | {r['label']} | {r['n_windows']} |")
                L += ["", f"STRATEGIC / TACTICAL: see `families` in RESULT_AMEND5.json: {json.dumps(js['strategic'])[:400]}"]
            L.append("")
        L += ["### Mechanism rows (beside, no bar)", "", "| arm | DEP mean abs 2 s departure (base) | HIT median final cost | HIT share at B1 |", "|---|---|---|---|"]
        for x, m in c["mechanism"].items():
            L.append(f"| {x} | {m['DEP_mean_abs_2s_mean_departure']:.4f} ({m['DEP_base']:.4f}) | {m['HIT_median_final_cost']:.3e} | {m['HIT_share_hit_B1']:.3f} |")
        L += ["", "### DIR-ACC (beside)", "", "| arm | r | CI | n windows | n episodes |", "|---|---|---|---|---|"]
        for a, r in c["dir_acc"]["arms"].items():
            L.append(f"| {a} | {_f(r['r'])} | [{_f(r['lo'])}, {_f(r['hi'])}] | {r['n_windows']} | {r.get('n_episodes', 'n/a')} |")
        L += ["", f"C-SHUF on SCORE: {c['C_SHUF_score']}; C-OFF5 on SCORE: {c['C_OFF5_score']}", ""]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    t = sp.add_parser("train"); t.add_argument("--chassis", required=True)
    sp.add_parser("score")
    a = ap.parse_args()
    os.makedirs(f"{ST.ROOT}/readout", exist_ok=True)
    if a.cmd == "train":
        r = train_stage(ST.ROOT, a.chassis)
        print("C-SHUF_train_sign", r["C-SHUF_train_sign"], "C-PLANT_ii", r["C-PLANT_ii"])
        sys.exit(0 if r["checks_ok"] else 2)
    r = score(ST.ROOT)
    print(json.dumps({x: {k: v for k, v in r["labels"]["arms"][x].items() if k != "cells"} for x in r["labels"]["arms"]}, indent=1, default=float))


if __name__ == "__main__":
    main()
