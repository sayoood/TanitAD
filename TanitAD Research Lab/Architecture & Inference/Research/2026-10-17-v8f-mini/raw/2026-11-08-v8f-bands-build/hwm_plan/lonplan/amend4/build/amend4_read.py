"""AMEND_HWMPLAN_4 readout (sections 5, 6, 7), computed MECHANICALLY from the arm folders. Nothing here is tuned; every constant is the card's.
Pure logic (importable, tested with synthetic inputs): sep_better, worse, chassis_label, combine, band_label, base_solver, read_labels.
Commands:
  train  --chassis k            TRAIN-stage report (beside, in-sample, never a result): P / P23 / RS against LON-PLAN's banked A0 on the 564 TRAIN windows, episode-cluster bootstrap over the 141 clips; C-RS sign check
  score                         contrasts (contrast_l1full_strat.py, the strategic-audit reader, B 2,000 seed 0), the 2-0 rows, DIR-ACC, mechanism rows, labels -> <ROOT>/readout/RESULT_AMEND4.json + .md skeleton
Estimator: paired episode-cluster bootstrap (answers 'another draw of EPISODES' only). The inference floor F_I answers 'another INFERENCE run'. Tier T0 (oracle goal): a world-model diagnostic, never driving performance.
Label wording follows section 6: L2IMG-<outcome> for P, L23IMG-<outcome> for P23; a label that differs between the chassis reads SEEDS-DISAGREE and the LESS favourable label's next step applies."""
import argparse, json, math, os, subprocess, sys
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np

ORDER = ("CEILING", "NARROWS", "NULL", "HARM")          # most to least favourable
BAND_ORDER = ("BAND-ADDS", "BAND-NEUTRAL", "BAND-HURTS")
D_MIN = 0.10                                            # X is HOLD-collapsed if D < 0.10 (section 6)


# ---------------------------------------------------------------------------------------------- pure logic
def sep_better(row, floor, seed1_delta, drop=()):
    """'Separated-better beyond floor': the CI lies wholly below 0, |delta| > F_I, and the seed-1 point delta is also below 0. `drop` (tests only): conditions among ci / floor / seed1 to ignore."""
    return bool(("ci" in drop or row["hi"] < 0) and ("floor" in drop or abs(row["delta"]) > floor) and ("seed1" in drop or seed1_delta < 0))


def worse(row, floor, drop=()):
    """'WORSE': the CI lies wholly above 0 and delta > F_I."""
    return bool(("ci" in drop or row["lo"] > 0) and ("floor" in drop or row["delta"] > floor))


def chassis_label(c, drop=()):
    """c: dict(dH=row, dH_F, dH_seed1, d0=row, d0_F, d0_seed1, gyaw=row, gyaw_F, D). Rows are the paired contrast rows (delta, lo, hi) of X - HOLD (dH), X - A0 (d0) and X - A0 on yaw_mae (gyaw), all 0-2 s.
    Precedence HARM > CEILING > NARROWS > NULL (a CEILING or NARROWS needs G_yaw not WORSE, so a worse guard is HARM).
    `drop` (the mutation hook of the tests): names among harm_d0_ci, harm_d0_floor, harm_guard, ceil_ci, ceil_floor, ceil_seed1, ceil_D, narrows_ci, narrows_floor, narrows_seed1: that ONE condition is ignored."""
    dr = lambda pre: tuple(x[len(pre):] for x in drop if x.startswith(pre))
    harm = worse(c["d0"], c["d0_F"], dr("harm_d0_")) or ("harm_guard" not in drop and worse(c["gyaw"], c["gyaw_F"]))
    if harm:
        return "HARM"
    if sep_better(c["dH"], c["dH_F"], c["dH_seed1"], dr("ceil_")) and ("ceil_D" in drop or c["D"] >= D_MIN):
        return "CEILING"
    if sep_better(c["d0"], c["d0_F"], c["d0_seed1"], dr("narrows_")):
        return "NARROWS"
    return "NULL"


def combine(labels, order=ORDER):
    """Both chassis equal -> that label; otherwise SEEDS-DISAGREE with the LESS favourable label as the effective one (its next step applies)."""
    ls = list(labels)
    if len(set(ls)) == 1:
        return dict(label=ls[0], effective=ls[0], seeds_disagree=False)
    eff = max(ls, key=order.index)
    return dict(label="SEEDS-DISAGREE", effective=eff, seeds_disagree=True)


def outcome(label, controls_ok):
    """Section 6: any control failing (C-RS included) reads INVALID, whatever the numbers say."""
    return label if controls_ok else "INVALID"


def band_label(c):
    """Secondary 2-6 s label (beside): BAND-ADDS if X - A0 on 2-6 s acc_mae is separated-better beyond floor; BAND-HURTS if WORSE; else BAND-NEUTRAL. Per chassis, then combine()."""
    if sep_better(c["d0"], c["d0_F"], c["d0_seed1"]):
        return "BAND-ADDS"
    if worse(c["d0"], c["d0_F"]):
        return "BAND-HURTS"
    return "BAND-NEUTRAL"


def base_solver(eff, dH_point, d0_point):
    """The RA read's base solver (section 6). eff: {arm: effective label}; dH_point / d0_point: {arm: [s0, s1]} points. If either arm reads CEILING that arm (both: P23 if its dH point is lower at BOTH chassis, else P);
    else if either reads NARROWS that arm (the same tie rule on d0); else A0's upper levels."""
    for want, pts in (("CEILING", dH_point), ("NARROWS", d0_point)):
        hit = [x for x in ("P", "P23") if eff.get(x) == want]
        if len(hit) == 1:
            return hit[0]
        if len(hit) == 2:
            return "P23" if all(a < b for a, b in zip(pts["P23"], pts["P"])) else "P"
    return "A0"


def read_labels(inp):
    """inp: {arm: {chassis: dict for chassis_label (+ band dict 'band')}} -> the labels block."""
    out, eff, dHp, d0p = {}, {}, {}, {}
    for arm, per in inp.items():
        ks = sorted(per)
        per_label = {k: chassis_label(per[k]) for k in ks}
        comb = combine([per_label[k] for k in ks])
        band = {k: band_label(per[k]["band"]) for k in ks}
        bcomb = combine([band[k] for k in ks], order=BAND_ORDER)
        pre = "L2IMG" if arm == "P" else "L23IMG"
        out[arm] = dict(per_chassis=per_label, **comb, name=f"{pre}-{comb['label']}", effective_name=f"{pre}-{comb['effective']}", band_per_chassis=band, band=bcomb,
                        hold_collapsed={k: bool(per[k]["D"] < D_MIN) for k in ks}, abs_dH_below_2F={k: bool(abs(per[k]["dH"]["delta"]) < 2 * per[k]["dH_F"]) for k in ks})
        eff[arm] = comb["effective"]
        dHp[arm] = [per[k]["dH"]["delta"] for k in ks]
        d0p[arm] = [per[k]["d0"]["delta"] for k in ks]
    out["base_solver_for_RA_read"] = base_solver(eff, dHp, d0p) if {"P", "P23"} <= set(inp) else "A0"
    out["multiplicity_guard_applies"] = bool(sum(1 for x in ("P", "P23") if eff.get(x) == "CEILING") == 1)
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


def cell(js_xa0, family, band, metric):
    """From the X - A0 contrast file (a = X, b = A0): dH (X - HOLD), d0 (X - A0), the floor F_I and the seed-1 points."""
    dH = find_row(js_xa0, "contrast", family, band, metric, "A", "HOLD")
    d0 = find_row(js_xa0, "contrast", family, band, metric, "A", "B")
    repX = find_row(js_xa0, "rep", family, band, metric, None, arm="A")["delta"]
    repA = find_row(js_xa0, "rep", family, band, metric, None, arm="B")["delta"]
    return dict(dH=dH, d0=d0, F=max(abs(repX), abs(repA)), dH_seed1=dH["delta"] + repX, d0_seed1=d0["delta"] + repX - repA, repX=repX, repA0=repA)


def chassis_input(js_xa0, D):
    c0 = cell(js_xa0, "LONGITUDINAL", "0-2s", "acc_mae")
    cy = cell(js_xa0, "LATERAL", "0-2s", "yaw_mae")
    cb = cell(js_xa0, "LONGITUDINAL", "2-6s", "acc_mae")
    return dict(dH=c0["dH"], dH_F=c0["F"], dH_seed1=c0["dH_seed1"], d0=c0["d0"], d0_F=c0["F"], d0_seed1=c0["d0_seed1"], gyaw=cy["d0"], gyaw_F=cy["F"], D=D,
                band=dict(d0=cb["d0"], d0_F=cb["F"], d0_seed1=cb["d0_seed1"]))


# ---------------------------------------------------------------------------------------------- the run side (needs the arm folders)
def _paths(root):
    import amend4_stage as ST
    return ST, root


def env_for_subprocess():
    e = dict(os.environ)
    e.update(CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1")
    return e


def run_contrast(windows_dir, a, b, out, rep_a=None, rep_b=None, py=sys.executable):
    lp = os.environ.get("AMEND4_LONPLAN_BUILD", "F:/TanitAD-scratch/actrsp/lonplan/build")
    cmd = [py, os.path.join(lp, "contrast_l1full_strat.py"), "--windows", windows_dir, "--a", a, "--b", b, "--out", out]
    if rep_a:
        cmd += ["--rep-a", rep_a]
    if rep_b:
        cmd += ["--rep-b", rep_b]
    r = subprocess.run(cmd, env=env_for_subprocess(), capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise RuntimeError(f"contrast {os.path.basename(out)} rc {r.returncode}: {r.stderr[-800:]}")
    return json.load(open(out, encoding="utf-8"))


def assemble_2_0(plans_npz, out_npz):
    """P2-0 / P23-2-0 (zero compute): knots 1-10 = level 2's own chosen primitives, knots 11-30 = the arm's."""
    with np.load(plans_npz, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    plan = d["plan_raw"].copy()
    plan[:, :10] = d["l2_prim_raw"][:, :10]
    np.savez(out_npz, plan_raw=plan)
    return out_npz


def sha(p):
    import lonplan_common as C
    return C.sha256_file(p)


def score(root, n_boot=2000):
    import amend4_stage as ST
    import lonplan_common as C
    import lonplan_select as S
    import lonplan_diracc as DA
    import hwm_eval as E
    L0 = ST.L0
    ro = f"{root}/readout"
    os.makedirs(ro, exist_ok=True)
    res = dict(kind="RESULT_AMEND4", tier="T0", evidence="MEASURED", estimator="paired_episode_cluster_bootstrap", B=n_boot, seed=0, chassis={})
    inp = {"P": {}, "P23": {}}
    for k in ("s0", "s1"):
        A0d, A1d = f"{L0}/score/A0_{k}", f"{L0}/score/A1_{k}"
        w = f"{A0d}/windows.npz"
        arm_dir = {x: f"{root}/score/{x}_{k}" for x in ("P", "P23", "RS")}
        for x, d in arm_dir.items():
            if C.sha256_file(f"{d}/windows.npz") != C.sha256_file(w):
                raise RuntimeError(f"{x}_{k} windows differ from A0_{k}")
        f = lambda d, s=0: f"{d}/plans_hwm_oracle_s{s}.npz"
        p20 = {x: assemble_2_0(f(arm_dir[x]), f"{ro}/{x}_2-0_{k}.npz") for x in ("P", "P23")}
        pairs = {"P_A0": (f(arm_dir["P"]), f(A0d), f(arm_dir["P"], 1), f(A0d, 1)), "P23_A0": (f(arm_dir["P23"]), f(A0d), f(arm_dir["P23"], 1), f(A0d, 1)),
                 "RS_A0": (f(arm_dir["RS"]), f(A0d), None, None), "RS_P": (f(arm_dir["RS"]), f(arm_dir["P"]), None, None),
                 "P23_P": (f(arm_dir["P23"]), f(arm_dir["P"]), f(arm_dir["P23"], 1), f(arm_dir["P"], 1)),
                 "P_A1": (f(arm_dir["P"]), f(A1d), f(arm_dir["P"], 1), f(A1d, 1)), "P23_A1": (f(arm_dir["P23"]), f(A1d), f(arm_dir["P23"], 1), f(A1d, 1)),
                 "P2-0_P": (p20["P"], f(arm_dir["P"]), None, None), "P23-2-0_P23": (p20["P23"], f(arm_dir["P23"]), None, None)}
        js = {n: run_contrast(A0d, a, b, f"{ro}/c_{n}_{k}.json", ra, rb) for n, (a, b, ra, rb) in pairs.items()}
        sel = json.load(open(f"{L0}/sel_{k}/LAMBDA_SELECTION_R35_{k}.json", encoding="utf-8"))
        std_raw = np.array(sel["refs"]["prior_std_raw"] if "refs" in sel else sel["prior_std_raw"], np.float64)
        wz = np.load(w, allow_pickle=False)
        A = wz["A"].astype(np.float64)
        D = {x: float(S.distance_D(np.load(f(arm_dir[x]))["plan_raw"].astype(np.float64), A, std_raw)[0]) for x in ("P", "P23")}
        D["A0"] = float(S.distance_D(np.load(f(A0d))["plan_raw"].astype(np.float64), A, std_raw)[0])
        for x in ("P", "P23"):
            inp[x][k] = chassis_input(js[f"{x}_A0"], D[x])
        dir_acc = DA.diracc_table(w, [("A0", f(A0d)), ("P", f(arm_dir["P"])), ("P23", f(arm_dir["P23"])), ("RS", f(arm_dir["RS"])), ("P2-0", p20["P"]), ("P23-2-0", p20["P23"])], n_boot, 0)
        eid = wz["eid"]
        mech = {}
        c1_a0 = np.load(f(A0d))["cost1"]
        for x in ("P", "P23", "RS"):
            c1 = np.load(f(arm_dir[x]))["cost1"]
            z = np.load(f(arm_dir[x]))
            mech[x] = dict(REACH=dict(median_ratio_cost1=float(np.median(c1 / c1_a0)), mean_log_ratio=E.contrast(np.log(c1), np.log(c1_a0), eid, n_boot)),
                           IMG_residual=dict(level2_max=float(z["image_resid"].max()), level2_median=float(np.median(z["image_resid"])), level3_max=(float(np.nanmax(z["image_resid3"])) if np.isfinite(z["image_resid3"]).any() else None)))
            a0b = np.load(f"{root}/controls/a0_b0_{k}_s0.npz")
            n0 = len(a0b["cost2"])
            mech[x]["COST2_batch0"] = dict(median_ratio_vs_A0=float(np.median(z["cost2"][:n0] / a0b["cost2"])), n=int(n0))
        res["chassis"][k] = dict(contrasts={n: f"{ro}/c_{n}_{k}.json" for n in pairs}, D=D, dir_acc=dir_acc, mechanism=mech,
                                 n_windows=int(len(eid)), n_episodes=int(len(set(eid))), windows_sha256=C.sha256_file(w),
                                 families={n: dict(strategic=js[n]["strategic"], tactical_kappa=js[n]["tactical_kappa"]) for n in ("P_A0", "P23_A0", "RS_A0")},
                                 plan_sha256={x: C.sha256_file(f(arm_dir[x])) for x in arm_dir})
        res["chassis"][k]["C-RS_score"] = {kk: find_row(js["RS_A0"], "contrast", "LONGITUDINAL", "0-2s", "acc_mae", "A", "B")[kk] for kk in ("delta", "lo", "hi", "label")}
        res["chassis"][k]["C-RS_score"]["ok"] = bool(res["chassis"][k]["C-RS_score"]["lo"] > 0)
    res["labels"] = read_labels(inp)
    ctl = json.load(open(f"{root}/controls/CONTROLS.json", encoding="utf-8"))
    res["controls_ok"] = bool(ctl["ok"])
    if not ctl["ok"]:
        res["labels"]["INVALID"] = f"a section 8 control is RED: {ctl['red']}"
    res["C-RS_score_ok_both_chassis"] = bool(all(res["chassis"][k]["C-RS_score"]["ok"] for k in ("s0", "s1")))
    if not res["C-RS_score_ok_both_chassis"]:
        res["labels"]["INVALID"] = "C-RS: RS - A0 on 0-2 s acc_mae is not separated-worse at both chassis; nothing is read (section 8.2)"
    with open(f"{ro}/RESULT_AMEND4.json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(res, fh, indent=1, default=float)
    with open(f"{ro}/RESULT_AMEND4.md", "w", encoding="utf-8", newline="\n") as fh:
        fh.write(render_md(res))
    return res


# ---------------------------------------------------------------------------------------------- TRAIN stage (beside)
def train_stage(root, k, n_boot=2000):
    import amend4_stage as ST
    import hwm_eval as E
    L0 = ST.L0
    w = np.load(ST.sel_windows(k), allow_pickle=False)
    A, v0, eid = w["A"].astype(np.float64), w["v0"], w["eid"]
    plans = {"A0": np.load(f"{L0}/sel_{k}/lam_0/plans_hwm_oracle_s0.npz"), **{x: np.load(f"{root}/train/{x}_{k}/plans_hwm_oracle_s0.npz") for x in ("P", "P23", "RS")}}
    M = {x: E.band_metrics(plans[x]["plan_raw"], A, v0, None, None)[0] for x in plans}
    out = dict(kind="AMEND4_TRAIN_STAGE", chassis=k, tier="T0, TRAIN windows, in-sample: beside, never a result", n_windows=int(len(eid)), n_clips=int(len(set(eid))), estimator="paired_episode_cluster_bootstrap over TRAIN clips", rows=[])
    for a_, b_ in (("P", "A0"), ("P23", "A0"), ("RS", "A0"), ("RS", "P")):
        for fam, mets in E.FAMILIES.items():
            for band in E.BANDS:
                for m in mets + (E.BESIDE if fam == "LATERAL" else ()):
                    r = E.contrast(M[a_][band][m], M[b_][band][m], eid, n_boot)
                    out["rows"].append(dict(a=a_, b=b_, family=fam, band=band, metric=m, **r))
    out["strategic"] = "n/a on TRAIN windows: no navigation-text row joined to TRAIN windows; reported on SCORE only"
    out["headway"] = "n = 0 on TRAIN windows (no lead-state join); reported on SCORE only"
    rs = [r for r in out["rows"] if (r["a"], r["b"], r["band"], r["metric"]) == ("RS", "A0", "0-2s", "acc_mae")][0]
    out["C-RS_train"] = dict(delta=rs["delta"], lo=rs["lo"], hi=rs["hi"], ok=bool(rs["delta"] > 0))
    with open(f"{root}/readout/TRAIN_STAGE_{k}.json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, indent=1, default=float)
    return out


# ---------------------------------------------------------------------------------------------- the .md skeleton generator
def _f(v, d=4):
    return "n/a" if v is None or (isinstance(v, float) and v != v) else f"{v:+.{d}f}"


def render_md(res):
    L = ["# RESULT_AMEND4 (skeleton generated from RESULT_AMEND4.json; the MM writes the reading)", "",
         "Tier T0 (oracle goal): a world-model diagnostic, never driving performance. Estimator: paired episode-cluster bootstrap, B 2,000 seed 0 (another draw of EPISODES only); F_I answers another INFERENCE run.", ""]
    if "INVALID" in res["labels"]:
        L += [f"**INVALID.** {res['labels']['INVALID']}", ""]
    L += ["## Labels (section 6)", "", "| arm | s0 | s1 | label | effective (next step applies) | 2-6 s band | HOLD-collapsed | abs dH < 2 F_I |", "|---|---|---|---|---|---|---|---|"]
    for x in ("P", "P23"):
        lb = res["labels"][x]
        L.append(f"| {x} | {lb['per_chassis']['s0']} | {lb['per_chassis']['s1']} | {lb['name']} | {lb['effective_name']} | {lb['band']['label']} ({lb['band_per_chassis']['s0']} / {lb['band_per_chassis']['s1']}) | {lb['hold_collapsed']} | {lb['abs_dH_below_2F']} |")
    L += ["", f"Base solver of the RA read: **{res['labels']['base_solver_for_RA_read']}**. Multiplicity guard applies: {res['labels']['multiplicity_guard_applies']}.", ""]
    for k, c in res["chassis"].items():
        L += [f"## Chassis R35_{k}: {c['n_windows']} windows, {c['n_episodes']} episodes (windows sha256 {c['windows_sha256'][:12]})", "",
              "Deciding cells, 0-2 s, with the family tables in the contrast files named below. Rows: delta [lo, hi] label.", ""]
        for n, p in c["contrasts"].items():
            js = json.load(open(p, encoding="utf-8"))
            L += [f"### {n}  ({os.path.basename(p)}; {js['n_windows']} windows / {js['n_episodes']} episodes)", "", "| family | band | metric | a - b | CI | label | n windows | n episodes |", "|---|---|---|---|---|---|---|---|"]
            for r in js["rows"]:
                if r["a"] == "A" and r["b"] == "B":
                    L.append(f"| {r['family']} | {r['band']} | {r['metric']} | {_f(r['delta'])} | [{_f(r['lo'])}, {_f(r['hi'])}] | {r['label']} | {r['n_windows']} | {r['n_episodes']} |")
            if n.endswith("_A0") and n[:2] != "RS":
                L += ["", "X - HOLD (the CEILING cell and its family):", "", "| family | band | metric | X - HOLD | CI | label | n |", "|---|---|---|---|---|---|---|"]
                for r in js["rows"]:
                    if r["a"] == "A" and r["b"] == "HOLD":
                        L.append(f"| {r['family']} | {r['band']} | {r['metric']} | {_f(r['delta'])} | [{_f(r['lo'])}, {_f(r['hi'])}] | {r['label']} | {r['n_windows']} |")
            L.append("")
        L += ["### Mechanism rows (beside, no bar)", "", "| arm | REACH median cost1 ratio vs A0 | REACH mean log-ratio [CI] | COST2 median ratio (batch 0) | image resid level 2 max | level 3 max |", "|---|---|---|---|---|---|"]
        for x, m in c["mechanism"].items():
            r = m["REACH"]["mean_log_ratio"]
            L.append(f"| {x} | {m['REACH']['median_ratio_cost1']:.3f} | {_f(r['delta'])} [{_f(r['lo'])}, {_f(r['hi'])}] | {m['COST2_batch0']['median_ratio_vs_A0']:.3f} | {m['IMG_residual']['level2_max']:.2e} | {m['IMG_residual']['level3_max']} |")
        L += ["", "### DIR-ACC (beside, no bar)", "", "| arm | r | CI | n windows | n episodes |", "|---|---|---|---|---|"]
        for a, r in c["dir_acc"]["arms"].items():
            L.append(f"| {a} | {_f(r['r'])} | [{_f(r['lo'])}, {_f(r['hi'])}] | {r['n_windows']} | {r.get('n_episodes', 'n/a')} |")
        L += ["", f"C-RS on SCORE (RS - A0, 0-2 s acc_mae): {c['C-RS_score']}", "", "STRATEGIC / TACTICAL blocks: see `families` in RESULT_AMEND4.json (strategic_block with n per cell and the not-computed entries with reasons).", ""]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    t = sp.add_parser("train"); t.add_argument("--chassis", required=True)
    sp.add_parser("score")
    a = ap.parse_args()
    import amend4_stage as ST
    os.makedirs(f"{ST.ROOT}/readout", exist_ok=True)
    if a.cmd == "train":
        r = train_stage(ST.ROOT, a.chassis)
        print("C-RS_train", r["C-RS_train"])
        sys.exit(0 if r["C-RS_train"]["ok"] else 2)
    r = score(ST.ROOT)
    print(json.dumps(r["labels"], indent=1, default=float))


if __name__ == "__main__":
    main()
