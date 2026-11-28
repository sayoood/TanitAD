"""AMEND_HWMPLAN_7 FIT stage CLI (sections 2-5). CPU only, OMP_NUM_THREADS=2, zero world-model training.
  python fit7.py crossfit --chassis k --arm RA|MLPA|MLPB|KCTL|PLANT|SHUFT|ZERO|CONST   writes amend7/fit/<arm>_<k>.npz + .json (+ _models.pt)
  python fit7.py final    --chassis k --seed-set A|B                                    writes readout_in/readout7_MLP_R35_<k>[_B].pt and updates RA7_FIT_<k>.json (sha256)
  python fit7.py gain     --chassis k                                                   GAIN7 (RA5 / RA6 / RA7-A) and RA5's r_wm on the 859 new clips -> fit/GAIN7_<k>.json
  python fit7.py gate                                                                   G7-FIT: readout/G7_FIT.json + .md and readout/RESULT_AMEND7_FIT.json + .md
Evidence class MEASURED, tier T0 (true-future latents and R35 rollouts under the logged actions, teacher-forced); estimator: paired clip-cluster percentile bootstrap, B 2,000, seed 0, 1,000 clips."""
import amend7_env  # noqa: F401  (first)
import argparse, hashlib, json, os, sys, time
sys.dont_write_bytecode = True
import numpy as np
import torch
from amend7_env import ROOT, ST, A5, A6, readout_path7
import fit_ra6 as F
import mlp7 as M
import pairs7 as PR

FITD = ROOT + "/fit"
RO = ROOT + "/readout"
ARMS = ("RA", "MLPA", "MLPB", "KCTL", "PLANT", "SHUFT", "ZERO", "CONST")
S_A = {"s0": 0.20076, "s1": 0.20323}                      # AMEND_OPTB.md section 1: Option B's own s_a
B_BOOT = 2000
STAMP = dict(tier="T0", evidence="MEASURED", estimator="paired clip-cluster percentile bootstrap, B 2000, seed 0, over the 1,000 clips (answers another draw of CLIPS only)")


def jdump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(obj, open(path, "w", encoding="utf-8", newline="\n"), indent=1, default=float)


def table(k):
    return PR.load_table(f"{PR.PAIRS_DIR}/pairs7_{k}.npz", f"{ROOT}/PAIRS7_{k}.json")


def pairs_sha(k):
    return json.load(open(f"{ROOT}/PAIRS7_{k}.json", encoding="utf-8"))["sha256"]


def swap_end(T):
    z0, z10, p = T["z0"].astype(np.float64), T["z10"].astype(np.float64), T["part"]
    return z0 + (z10[p] - z0[p])


def s_swap(pred_real, pred_swap, y, part):
    """amend1: s_swap = Cov(R(real) - R(swap), y - y') / Var(y - y'), population moments."""
    d = y - y[part]
    u = pred_real - pred_swap
    return float(((u - u.mean()) * (d - d.mean())).mean() / d.var())


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def crossfit_arm(arm, T, k, y_override=None, learn=None, kind=None, lams=None, mutate=None, rolls=True):
    z0, z10 = T["z0"], T["z10"]
    y = T["y"] if y_override is None else y_override
    fo, oc = T["fold"], T["oc"]
    if learn is None:
        learn = dict(RA=M.ridge_learner(), MLPA=M.mlp_learner(M.S0_A), MLPB=M.mlp_learner(M.S0_B), ZERO=M.mlp_learner(M.S0_A), SHUFT=M.mlp_learner(M.S0_A), CONST=M.const_learner(),
                     KCTL=(M.lind_learner(lams) if lams is not None else None))[arm]
    kind = kind or dict(RA="D", MLPA="z0z1", MLPB="z0z1", ZERO="z0z0", SHUFT="z0z1", CONST="D", KCTL="z0z1")[arm]
    use = rolls and arm in ("RA", "MLPA", "MLPB", "KCTL", "ZERO")
    return M.crossfit(learn, kind, z0, z10, y, fo, oc, T["F_true"] if use else None, T["F_alt"] if use else None, swap_end(T) if use else None, mutate=mutate)


def save_arm(name, k, res, y_used, T, extra=None, models=True):
    base = f"{FITD}/{name}_{k}"
    os.makedirs(FITD, exist_ok=True)
    arrs = dict(pred=res["pred"], y_used=y_used, fold=T["fold"])
    for key in ("dR", "rswap"):
        if res[key] is not None:
            arrs[key] = res[key]
    np.savez(base + ".npz", **arrs)
    if models:
        torch.save(res["models"], base + "_models.pt")
    rec = dict(arm=name, chassis=k, n=int(len(y_used)), pairs_sha256=pairs_sha(k), r2_point=M.r2(res["pred"], y_used), folds=res["folds"], **STAMP, **(extra or {}))
    jdump(rec, base + ".json")
    return rec


def cmd_crossfit(k, arm):
    t0 = time.time()
    T = table(k)
    log("crossfit", k, arm, "n", len(T["y"]))
    if arm == "PLANT":
        D = T["z10"].astype(np.float64) - T["z0"].astype(np.float64)
        yp, r2o, _ = M.planted(T["z0"].astype(np.float64), D)
        ra = crossfit_arm("RA", T, k, y_override=yp, rolls=False)
        lams = [f["lam"] for f in ra["folds"]]
        kc = crossfit_arm("KCTL", T, k, y_override=yp, lams=lams, rolls=False)
        ml = crossfit_arm("MLPA", T, k, y_override=yp, rolls=False)
        np.savez(f"{FITD}/PLANT_{k}.npz", y_plant=yp, pred_RA=ra["pred"], pred_KCTL=kc["pred"], pred_MLP=ml["pred"], fold=T["fold"])
        jdump(dict(arm="PLANT", chassis=k, n=int(len(yp)), pairs_sha256=pairs_sha(k), r2_oracle=r2o, r2_RA=M.r2(ra["pred"], yp), r2_KCTL=M.r2(kc["pred"], yp), r2_MLP=M.r2(ml["pred"], yp),
                   folds_MLP=ml["folds"], folds_RA=ra["folds"], wall_s=round(time.time() - t0, 1), **STAMP), f"{FITD}/PLANT_{k}.json")
        log("PLANT", k, "r2 oracle", round(r2o, 4), "MLP", round(M.r2(ml["pred"], yp), 4), "RA", round(M.r2(ra["pred"], yp), 4), "KCTL", round(M.r2(kc["pred"], yp), 4))
        return
    y_used = M.shuffled(T["y"]) if arm == "SHUFT" else T["y"]
    lams = None
    if arm == "KCTL":
        lams = [f["lam"] for f in json.load(open(f"{FITD}/RA_{k}.json", encoding="utf-8"))["folds"]]
    res = crossfit_arm(arm, T, k, y_override=y_used, lams=lams)
    rec = save_arm(arm, k, res, y_used, T, extra=dict(wall_s=round(time.time() - t0, 1), S0=(M.S0_B if arm == "MLPB" else M.S0_A)))
    log("crossfit", k, arm, "done R2", round(rec["r2_point"], 5), "wall", rec["wall_s"])


def cmd_final(k, seed_set):
    t0 = time.time()
    T = table(k)
    S0 = M.S0_A if seed_set == "A" else M.S0_B
    X = M.build_input("z0z1", T["z0"], T["z10"])
    log("final", k, seed_set, "selection over all five folds, refit on", len(X), "pairs")
    m, rec = M.mlp_learner(S0)(X, T["y"], T["fold"], T["oc"], M.SEED_FINAL_SEL)
    e, wd = rec["chosen"]
    p = readout_path7(k, "" if seed_set == "A" else "_B")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    torch.save(dict(kind="mlp", amend="AMEND_HWMPLAN_7", chassis=k, seed_set=seed_set, S0=S0, width=m["width"], epochs=int(e), wd=float(wd), mu=m["mu"], sd=m["sd"], net_state=m["state"],
                    n_pairs=int(len(X)), n_clips=int(T["oc"].max() + 1), refit_seed=rec["refit_seed"], input="[z0, z10] (256), standardised by the fit rows"), p)
    sha = hashlib.sha256(open(p, "rb").read()).hexdigest()
    jp = f"{ROOT}/RA7_FIT_{k}.json"
    J = json.load(open(jp, encoding="utf-8")) if os.path.exists(jp) else dict(chassis=k)
    suf = "" if seed_set == "A" else "_B"
    J.update({f"readout{suf}": p, f"sha256{suf}": sha, f"epochs{suf}": int(e), f"wd{suf}": float(wd), f"S0{suf}": S0, f"selection{suf}": rec["selection"], f"refit_seed{suf}": rec["refit_seed"], "n_pairs": int(len(X)), "n_clips": int(T["oc"].max() + 1),
              "pairs_sha256": pairs_sha(k), "tier": "T0", "evidence": "MEASURED", f"wall_s{suf}": round(time.time() - t0, 1)})
    jdump(J, jp)
    log("final", k, seed_set, "chosen epochs", e, "wd", wd, "sha", sha[:12], "wall", round(time.time() - t0, 1))


# ---------------------------------------------------------------------------------------------- GAIN7
def _ridge_np(path):
    r = torch.load(path, map_location="cpu", weights_only=False)
    return tuple(np.asarray(r["RA"][x], np.float64) for x in ("mu", "sd", "W", "ym"))


def gain_summary(G, oc_anc, n_clips, B=1000):
    bt = M.Boot(oc_anc, n_clips, dict(G=G))
    pt, lo, hi = bt.run(lambda T: np.array([T["G"] / T["n"]]), B, 0)
    return dict(n_anchors=int(len(G)), mean=dict(point=float(pt[0]), lo=float(lo[0]), hi=float(hi[0])), q10=float(np.quantile(G, 0.1)), q50=float(np.quantile(G, 0.5)), q90=float(np.quantile(G, 0.9)),
                share_below_0p25=float((G < 0.25).mean()), estimator="clip-cluster percentile bootstrap, B 1000, seed 0, on the mean")


def cmd_gain(k):
    t0 = time.time()
    import ra7_l1
    T = table(k)
    z0, anc, zp, zm = T["z0"].astype(np.float64), T["anc"], T["zp"].astype(np.float64), T["zm"].astype(np.float64)
    Z0, oca = z0[anc], T["oc"][anc]
    ra5 = _ridge_np(f"{A5}/readout_in/readout_K10_R35_{k}.pt")
    ra6 = _ridge_np(f"{A6}/readout_in/readout_K10_R35_{k}.pt")
    J7 = json.load(open(f"{ROOT}/RA7_FIT_{k}.json", encoding="utf-8"))
    r7 = ra7_l1.load_readout7(J7["readout"], k, J7["sha256"])
    rd = lambda m, ze: F.ridge_pred(m, ze - Z0)[:, 0]
    with torch.no_grad():
        rd7 = lambda ze: r7.read(torch.tensor(ze - Z0, dtype=torch.float32), torch.tensor(Z0, dtype=torch.float32)).numpy().astype(np.float64)
    a_ra, a_ml = np.load(f"{FITD}/RA_{k}.npz"), np.load(f"{FITD}/MLPA_{k}.npz")
    s6 = s_swap(a_ra["pred"], a_ra["rswap"], T["y"], T["part"])
    s7 = s_swap(a_ml["pred"], a_ml["rswap"], T["y"], T["part"])
    dl = PR.DELTA
    out = dict(chassis=k, delta=dl, s_used=dict(RA5=S_A[k], RA6=s6, RA7A=s7), anchors_per_clip_rule="every 10 frames from the clip's first table frame (amend1_kstep.py:195)", **STAMP)
    G = {}
    G["RA5"] = (rd(ra5, zp) - rd(ra5, zm)) / (2 * dl * S_A[k])
    G["RA6"] = (rd(ra6, zp) - rd(ra6, zm)) / (2 * dl * s6)
    G["RA7A"] = (rd7(zp) - rd7(zm)) / (2 * dl * s7)
    out["gain"] = {n: gain_summary(g, oca, int(T["oc"].max() + 1)) for n, g in G.items()}
    new = oca >= 141
    out["gain_RA5_new_clips_only"] = gain_summary(G["RA5"][new], oca[new] - 141, 859)
    out["gain_RA5_new_clips_only"]["note"] = "RA5 held out here (859 new clips); in-sample on the 141"
    out["gain_note"] = "RA6 and RA7-A are final readouts, in-sample on these clips; s of RA6 / RA7-A = that readout's own cross-fit s_swap at N = 1,000 (fit/RA and fit/MLPA)"
    # r_wm of RA5 on the 859 new clips (RA5 held out): same-state contrast, Pearson r(dR, y)
    rows = T["oc"] >= 141
    dR = F.ridge_pred(ra5, T["F_true"].astype(np.float64)[rows] - z0[rows])[:, 0] - F.ridge_pred(ra5, T["F_alt"].astype(np.float64)[rows] - z0[rows])[:, 0]
    y = T["y"][rows]
    oc = T["oc"][rows] - 141
    bt = M.Boot(oc, 859, dict(y=y, yy=y * y, x=dR, xx=dR * dR, xy=dR * y))
    pt, lo, hi = bt.run(lambda t: np.array([M.t_corr(t, "x", "xx", "xy")]), B_BOOT, 0)
    out["r_wm_RA5_new_clips"] = dict(point=float(pt[0]), lo=float(lo[0]), hi=float(hi[0]), n_pairs=int(rows.sum()), n_clips=859, note="RA5 frozen readout, held out on the 859 new clips")
    out["wall_s"] = round(time.time() - t0, 1)
    jdump(out, f"{FITD}/GAIN7_{k}.json")
    log("gain", k, {n: round(v["mean"]["point"], 4) for n, v in out["gain"].items()}, "r_wm RA5 new", round(float(pt[0]), 4))


# ---------------------------------------------------------------------------------------------- the gate
def chassis_stats(k, B=B_BOOT):
    T = table(k)
    y, oc, part = T["y"], T["oc"], T["part"]
    A = {n: np.load(f"{FITD}/{n}_{k}.npz") for n in ("RA", "MLPA", "MLPB", "ZERO", "KCTL", "CONST")}
    cols = dict(y=y, yy=y * y, d=y - y[part], dd=(y - y[part]) ** 2)
    for n in ("RA", "MLPA", "MLPB", "ZERO", "KCTL", "CONST"):
        cols[f"e2_{n}"] = (A[n]["pred"] - y) ** 2
    for n in ("RA", "MLPA", "MLPB"):
        x = A[n]["dR"]
        cols.update({f"x_{n}": x, f"xx_{n}": x * x, f"xy_{n}": x * y})
    for n in ("RA", "MLPA", "MLPB"):
        u = A[n]["pred"] - A[n]["rswap"]
        cols.update({f"u_{n}": u, f"uu_{n}": u * u, f"ud_{n}": u * (y - y[part])})
    bt = M.Boot(oc, int(oc.max() + 1), cols)

    def corr_wm(t, n):
        return M.t_corr(t, f"x_{n}", f"xx_{n}", f"xy_{n}")

    def sw(t, n):
        nn = t["n"]; mu_u, mu_d = t[f"u_{n}"] / nn, t["d"] / nn
        cov = t[f"ud_{n}"] / nn - mu_u * mu_d
        vd = t["dd"] / nn - mu_d ** 2
        vu = t[f"uu_{n}"] / nn - mu_u ** 2
        return np.array([cov / vd, cov / np.sqrt(vu * vd)])

    names = ["r2_RA", "r2_MLPA", "r2_MLPB", "r2_ZERO", "r2_KCTL", "r2_CONST", "dR2_A", "dR2_B", "r_wm_RA", "r_wm_MLPA", "r_wm_MLPB", "dr_A", "dr_B", "s_swap_RA", "r_swap_RA", "s_swap_MLPA", "r_swap_MLPA", "s_swap_MLPB", "r_swap_MLPB"]

    def fn(t):
        r = {n: t_ for n, t_ in ((f"r2_{a}", M.t_r2(t, f"e2_{a}")) for a in ("RA", "MLPA", "MLPB", "ZERO", "KCTL", "CONST"))}
        w = {f"r_wm_{a}": corr_wm(t, a) for a in ("RA", "MLPA", "MLPB")}
        v = [r["r2_RA"], r["r2_MLPA"], r["r2_MLPB"], r["r2_ZERO"], r["r2_KCTL"], r["r2_CONST"], r["r2_MLPA"] - r["r2_RA"], r["r2_MLPB"] - r["r2_RA"], w["r_wm_RA"], w["r_wm_MLPA"], w["r_wm_MLPB"],
             w["r_wm_MLPA"] - w["r_wm_RA"], w["r_wm_MLPB"] - w["r_wm_RA"]]
        for a in ("RA", "MLPA", "MLPB"):
            v += list(sw(t, a))
        return np.array(v)
    pt, lo, hi = bt.run(fn, B, 0)
    S = {n: dict(point=float(pt[i]), lo=float(lo[i]), hi=float(hi[i])) for i, n in enumerate(names)}
    S["F_T_r2"] = abs(S["r2_MLPB"]["point"] - S["r2_MLPA"]["point"])
    S["F_T_r_wm"] = abs(S["r_wm_MLPB"]["point"] - S["r_wm_MLPA"]["point"])
    return S, A


def cmd_gate():
    t0 = time.time()
    ctl_p = f"{ROOT}/controls/CONTROLS_FIT.json"
    ctl = json.load(open(ctl_p, encoding="utf-8")) if os.path.exists(ctl_p) else None
    controls_ok = bool(ctl and ctl["ok"])
    per, S_all = {}, {}
    for k in ("s0", "s1"):
        S, _ = chassis_stats(k)
        S_all[k] = S
        per[k] = M.chassis_gate(dict(dR2_A=S["dR2_A"], dR2_B=S["dR2_B"], dr_A=S["dr_A"], dr_B=S["dr_B"]), M.M_K[k], S["F_T_r2"], S["F_T_r_wm"])
        per[k].update(M_k=M.M_K[k], F_T_r2=S["F_T_r2"], F_T_r_wm=S["F_T_r_wm"])
    out_ = M.fit_outcome(per, controls_ok)
    gain = {k: json.load(open(f"{FITD}/GAIN7_{k}.json", encoding="utf-8")) for k in ("s0", "s1") if os.path.exists(f"{FITD}/GAIN7_{k}.json")}
    res = dict(kind="RESULT_AMEND7_FIT", spec_sha256=ST.SPEC_SHA, **STAMP, outcome=out_["text"], outcome_label=out_["label"], seeds_disagree=out_["seeds_disagree"], trans_only=out_["trans_only"], controls_ok=controls_ok,
               controls=(dict(ok=ctl["ok"], red=ctl["red"]) if ctl else "CONTROLS_FIT.json missing: R7-FIT-INVALID"), per_chassis=per, stats=S_all, gain7=gain,
               families=dict(LONGITUDINAL="FIT rows below are readout diagnostics of the longitudinal channel (T0, MEASURED); plan cells: NOT COMPUTABLE (n = 0; a readout fit produces no trajectory)",
                             LATERAL="NOT COMPUTABLE (n = 0; a readout fit produces no trajectory)", TACTICAL="NOT COMPUTABLE (n = 0; a readout fit produces no trajectory)",
                             STRATEGIC="NOT COMPUTABLE (n = 0; a readout fit produces no trajectory); the item 25 pause ended by PI ruling 2026-11-16, so it is reported in full on the SCORE read"),
               next_lever=next_lever(out_), wall_s=round(time.time() - t0, 1), finished=time.strftime("%Y-%m-%dT%H:%M:%S"))
    os.makedirs(RO, exist_ok=True)
    for nme in ("G7_FIT", "RESULT_AMEND7_FIT"):
        jdump(res, f"{RO}/{nme}.json")
        open(f"{RO}/{nme}.md", "w", encoding="utf-8", newline="\n").write(render_fit_md(res))
    log("GATE", out_["text"], {k: (v["a"], v["b"]) for k, v in per.items()})
    return out_


def next_lever(o):
    if o["label"] == "R7-FIT-INVALID":
        return "nothing is read; the failing control is fixed and the FIT stage rerun (PLANT P1 alone: the grid is re-carded by an amendment first)"
    if o["label"] == "R7-FIT-PASS":
        return "the SCORE read (sections 6-7) runs on the final readouts A and B (MM launches: run_amend7.sh score)"
    if o["label"] == "R7-STATE-ONLY":
        return "no SCORE read (the gain lies in the part of the read that cancels in the pass); next lever: ACTRSP Option B, handed section 5.2"
    return "RY-shape readout ELIMINATED as a lever at N = 1,000 at the FIT stage (logged); next lever: ACTRSP Option B (card AMEND_OPTB.md), handed section 5.2" + ("; TRANS-ONLY: a SCORE read of RA7 is an MM decision (about 1.3 h of CPU)" if o["trans_only"] else "")


def _ci(v, d=4):
    return f"{v['point']:+.{d}f} [{v['lo']:+.{d}f}, {v['hi']:+.{d}f}]"


def render_fit_md(res):
    L = ["# RESULT_AMEND7_FIT (generated; the MM writes the reading)", "",
         f"Tier T0 (true-future latents and R35 rollouts under the logged actions, teacher-forced): a world-model diagnostic, never driving performance. Evidence MEASURED. Estimator: {STAMP['estimator']}. F_T answers another READOUT-TRAINING run; neither answers another training run of R35.", "",
         f"**Outcome: {res['outcome']}.** Controls: {res['controls']}. Next lever: {res['next_lever']}", "",
         "## G7-FIT (card section 4), per chassis", "", "| chassis | (a) R2 part | (b) pass-contrast part | label | M_k | F_T(R2) | F_T(r_wm) |", "|---|---|---|---|---|---|---|"]
    for k, v in res["per_chassis"].items():
        L.append(f"| {k} | {v['a']} | {v['b']} | {v['label']} | {v['M_k']} | {v['F_T_r2']:.4f} | {v['F_T_r_wm']:.4f} |")
    for k, S in res["stats"].items():
        L += ["", f"## FIT rows, chassis R35_{k} (58,934 pairs, 1,000 clips; held-out, cross-fit; T0, MEASURED)", "", "| quantity | point [95% CI] |", "|---|---|"]
        for n in ("r2_RA", "r2_MLPA", "r2_MLPB", "dR2_A", "dR2_B", "r_wm_RA", "r_wm_MLPA", "r_wm_MLPB", "dr_A", "dr_B", "r2_ZERO", "r2_KCTL", "r2_CONST", "s_swap_RA", "r_swap_RA", "s_swap_MLPA", "r_swap_MLPA"):
            L.append(f"| {n}{' (STATE7)' if n == 'r2_ZERO' else ''} | {_ci(S[n])} |")
    for k, g in res["gain7"].items():
        L += ["", f"## GAIN7, chassis {k} (beside, no bar; T0, MEASURED; {g['estimator'] if 'estimator' in g else ''})", "", "| readout | mean G_aa [CI] | q10 | q50 | q90 | share G < 0.25 | n anchors |", "|---|---|---|---|---|---|---|"]
        for n, v in g["gain"].items():
            L.append(f"| {n} | {_ci(v['mean'])} | {v['q10']:.3f} | {v['q50']:.3f} | {v['q90']:.3f} | {v['share_below_0p25']:.3f} | {v['n_anchors']} |")
        v = g["gain_RA5_new_clips_only"]
        L.append(f"| RA5, 859 new clips (held out) | {_ci(v['mean'])} | {v['q10']:.3f} | {v['q50']:.3f} | {v['q90']:.3f} | {v['share_below_0p25']:.3f} | {v['n_anchors']} |")
        L.append(f"\nr_wm of RA5 on the 859 new clips: {_ci(g['r_wm_RA5_new_clips'])}.")
    L += ["", "## Four families (all four are named; the FIT stage defines only the longitudinal readout diagnostics above)", ""] + [f"* {k}: {v}" for k, v in res["families"].items()]
    return "\n".join(L) + "\n"


def main():
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    c = sp.add_parser("crossfit"); c.add_argument("--chassis", required=True); c.add_argument("--arm", required=True, choices=ARMS)
    f = sp.add_parser("final"); f.add_argument("--chassis", required=True); f.add_argument("--seed-set", required=True, choices=["A", "B"])
    g = sp.add_parser("gain"); g.add_argument("--chassis", required=True)
    sp.add_parser("gate")
    a = ap.parse_args()
    if a.cmd == "crossfit":
        cmd_crossfit(a.chassis, a.arm)
    elif a.cmd == "final":
        cmd_final(a.chassis, a.seed_set)
    elif a.cmd == "gain":
        cmd_gain(a.chassis)
    else:
        cmd_gate()


if __name__ == "__main__":
    main()
