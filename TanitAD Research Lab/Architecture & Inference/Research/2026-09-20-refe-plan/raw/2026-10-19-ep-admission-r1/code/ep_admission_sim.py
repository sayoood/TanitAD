"""EP-ADMISSION-R1: simulate the false-refusal rate and the power of the Revision-1 admission gate on the REAL estimator (design: ../SIM_DESIGN.md, hashed before the first run).  CPU only.

Pipeline (each step writes to --workdir on D:; every step is restartable):
    python ep_admission_sim.py precompute --arrays-npz <pickrule arrays.npz> --landed-result <ep-reliab result.json> --workdir D:/epadm1/run     # per-log sufficient statistics + PC3 (the shortcut IS the estimator)
    python ep_admission_sim.py cost       --workdir D:/epadm1/run                                                                                  # MEASURE the cost on 3 replicates first
    python ep_admission_sim.py run        --workdir D:/epadm1/run --n-reps 1000 --workers 8                                                         # all conditions + PC1/PC2/PC4/PC6 + flags -> sim_results.json
    python ep_admission_sim.py repro      --workdir D:/epadm1/run                                                                                  # PC5: 20 replicates in a SEPARATE process, bit-identical to run's first 20
`run` / `repro` take `--rep-offset INT` (default 0, EP-ADMISSION-R2-CONFIRM): replicate `rep` is drawn with `rep + offset` in BOTH generators (`default_rng([OUTER_SALT, rep + offset])`, `default_rng([WILD_SALT, rep + offset])`).
Offset 0 is the banked run bit for bit; a non-zero offset is a FRESH set of replicates of the same conditions, estimator and registered bootstrap draws (`BOOT_SEED` is NOT offset).
The decision functions are imported from `ep_admission.py` (rules r1 and the draft, for comparison); the estimator is `fastneg_lib.diff_ci` / `ratio_ci` with the registered draws.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import multiprocessing as mp
import os
import subprocess
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.abspath(os.path.join(HERE, "..", ".."))
ADM_DIR = os.environ.get("EP_ADMISSION_DIR") or os.path.join(RAW, "2026-10-19-p6-read-prep", "code")
DESIGN = os.path.join(HERE, "..", "SIM_DESIGN.md")
sys.path.insert(0, ADM_DIR)
import ep_admission as A                                                                    # noqa: E402

N_LOGS, BOOT_B, BOOT_SEED = 117, 10000, 20260927
OUTER_SALT, WILD_SALT, SUBSET_SEED = 20261027, 20261028, 20261029
SEEDS = (0, 1)
HEADS = ("NC", "DAC", "TTC")
HCOL = {"NC": 0, "DAC": 1, "TTC": 3}
EP = 2
GEOMS = {"G78": 1.0, "G25": 0.32, "G05": 0.065}
DELTAS = {"G78": (0.01, 0.02, 0.05, 0.10, 0.135), "G25": (0.05, 0.135), "G05": (0.05, 0.135)}
RHOS = (1.25, 1.5)
Z95 = 1.959964
# PC1 / PC2 tolerances (fixed in SIM_DESIGN.md section 7 BEFORE any output)
TOL_EP_M, TOL_DM, TOL_A2_RATIO, TOL_PLANT_DM = 0.01, 0.005, 0.05, 0.005


def libs():
    return A.libs()


def sha256_file(p):
    return A.sha256_file(p)


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# small pure helpers
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def wilson(k: int, n: int, z: float = Z95) -> list:
    """Wilson score interval for a binomial proportion k / n."""
    if n <= 0:
        return [float("nan"), float("nan")]
    p = k / n
    den = 1.0 + z * z / n
    c = (p + z * z / (2.0 * n)) / den
    h = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / den
    return [max(0.0, c - h), min(1.0, c + h)]


def rate(k: int, n: int) -> dict:
    return {"k": int(k), "n": int(n), "rate": (k / n if n else float("nan")), "wilson95": wilson(int(k), int(n))}


def _finite(x):
    x = float(x)
    return x if math.isfinite(x) else None


def ci_of(row) -> dict:
    return {"point": float(row[0]), "lo": float(row[1]), "hi": float(row[2])}


def per_log(v, log_idx, n_logs=N_LOGS):
    return np.bincount(log_idx, weights=np.asarray(v, np.float64), minlength=n_logs).astype(np.float64)


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# per-log sufficient statistics of one (geometry, seed)
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def components(z6: np.ndarray, zc: np.ndarray, lab: np.ndarray, log_idx: np.ndarray, n_logs: int = N_LOGS) -> dict:
    """The 117-vectors the replicates are built from, from the SAME terms `epreliab_lib.pair_terms` / `reliability_stats` use.
       y_p, s_p, y_e, s_e : the direct per-log sums of the EP label / candidate sigma at P2's pick and at the candidate's pick (the estimator's own inputs: the 'raw' condition)
       Yu Su Ypc Spc Yec Sec : the same, split into UNCHANGED sets (one shared pick) and CHANGED sets (P2's pick / the candidate's pick)
       OBS_h PRED_h : per-log observed (1 - y) and predicted (1 - sigma) unsafe mass of NC / DAC / TTC at the candidate's picks, from the candidate's rows"""
    L, P, F = libs()
    pk_p, pk_c = P.rule_pick(z6), P.rule_pick(zc)
    sig, y = P.sigmoid64(zc[..., EP]), lab[..., EP].astype(np.float64)
    t = L.pair_terms(sig, y, pk_p, pk_c)
    ch = t["changed"]
    nc = (~ch).astype(np.float64)
    c = ch.astype(np.float64)
    pl = lambda v: per_log(v, log_idx, n_logs)               # noqa: E731
    out = {"y_p": pl(t["y_p"]), "s_p": pl(t["s_p"]), "y_e": pl(t["y_e"]), "s_e": pl(t["s_e"]),
           "Yu": pl(t["y_p"] * nc), "Su": pl(t["s_p"] * nc), "Ypc": pl(t["y_p"] * c), "Spc": pl(t["s_p"] * c), "Yec": pl(t["y_e"] * c), "Sec": pl(t["s_e"] * c)}
    for h in HEADS:
        yh = lab[..., HCOL[h]].astype(np.float64)
        sg = P.sigmoid64(zc[..., HCOL[h]])
        out[f"OBS_{h}"] = pl(1.0 - P.at(yh, pk_c))
        out[f"PRED_{h}"] = pl(1.0 - P.at(sg, pk_c))
    meta = {"n_changed": int(ch.sum()), "n_sets": int(len(pk_p)), "s_e_changed": t["s_e"][ch]}
    return {"vec": out, "meta": meta}


def scalars(v: dict) -> dict:
    """The calibration scalars of section 4 of the design (totals over the 117 real logs)."""
    return {"a_u": v["Yu"].sum() / v["Su"].sum() if v["Su"].sum() > 0 else 1.0, "a_p": v["Ypc"].sum() / v["Spc"].sum() if v["Spc"].sum() > 0 else 1.0,
            "a_e": v["Yec"].sum() / v["Sec"].sum() if v["Sec"].sum() > 0 else 1.0, **{f"k_{h}": v[f"OBS_{h}"].sum() / v[f"PRED_{h}"].sum() for h in HEADS}}


def plant_factor(v: dict, delta: float) -> float:
    """f such that, after calibration, inflating the candidate's predicted EP mass at its CHANGED picks by f gives M_e = 1 - delta (so D_M = +delta exactly; M_p stays 1)."""
    yu, yec = float(v["Yu"].sum()), float(v["Yec"].sum())
    return ((yu + yec) / (1.0 - delta) - yu) / yec


def build_condition(v: dict, kind: str, delta: float = 0.0, head: str | None = None, rho: float = 1.0) -> dict:
    """Per-log vectors of one condition of one seed.  kind: 'raw' (the real, UNcalibrated data), 'cal' (calibrated by construction), 'plant_ep' (+delta over-prediction), 'plant_a2' (`head` under-predicts risk by rho).
    Returns {Yp, Sp, Ye, Se, OBS_h, PRED_h} (the generator-A inputs) and the components for generator B: {Yc: {u, pc, ec}, Sc: {u, pc, ec}, Scal: {u, pc, ec}}."""
    if kind == "raw":
        d = {"Yp": v["y_p"], "Sp": v["s_p"], "Ye": v["y_e"], "Se": v["s_e"], **{f"OBS_{h}": v[f"OBS_{h}"] for h in HEADS}, **{f"PRED_{h}": v[f"PRED_{h}"] for h in HEADS}}
        d["Yc"] = {"u": v["Yu"], "pc": v["Ypc"], "ec": v["Yec"]}
        d["Sc"] = d["Scal"] = {"u": v["Su"], "pc": v["Spc"], "ec": v["Sec"]}
        return d
    sc = scalars(v)
    Scal = {"u": sc["a_u"] * v["Su"], "pc": sc["a_p"] * v["Spc"], "ec": sc["a_e"] * v["Sec"]}
    Sc = dict(Scal)
    if kind == "plant_ep":
        Sc = {**Scal, "ec": plant_factor(v, delta) * Scal["ec"]}
    Yc = {"u": v["Yu"], "pc": v["Ypc"], "ec": v["Yec"]}
    d = {"Yp": Yc["u"] + Yc["pc"], "Sp": Sc["u"] + Sc["pc"], "Ye": Yc["u"] + Yc["ec"], "Se": Sc["u"] + Sc["ec"], "Yc": Yc, "Sc": Sc, "Scal": Scal}
    for h in HEADS:
        d[f"OBS_{h}"] = v[f"OBS_{h}"]
        d[f"PRED_{h}"] = sc[f"k_{h}"] * v[f"PRED_{h}"] / (rho if (kind == "plant_a2" and h == head) else 1.0)
    return d


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# a replicate: ONE draw of logs shared by both seeds and all heads; the REAL estimator on it
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def outer_idx(rep: int, offset: int = 0) -> np.ndarray:
    return np.random.default_rng([OUTER_SALT, rep + offset]).integers(0, N_LOGS, N_LOGS)


def wild_signs(rep: int, offset: int = 0) -> np.ndarray:
    return np.random.default_rng([WILD_SALT, rep + offset]).choice(np.array([-1.0, 1.0]), N_LOGS)


def replicate(cond: dict, rep: int, draws: np.ndarray, offset: int = 0) -> dict:
    """-> {dm [2, 3] (point, lo, hi) per seed, a2 [2, 3, 3] (head, point/lo/hi) per seed, mp [2], me [2]}.  Generator A resamples the logs; generator B flips the sign of each log's calibrated residual.
    Replicate `rep` uses the draws of index `rep + offset` (offset 0 = the banked run; the registered bootstrap `draws` are shared by every replicate and are not offset)."""
    _, _, F = libs()
    gen, heads = cond["gen"], cond.get("heads", HEADS)
    idx = outer_idx(rep, offset) if gen == "A" else None
    w = wild_signs(rep, offset) if gen == "B" else None
    dm = np.full((2, 3), np.nan)
    a2 = np.full((2, 3, 3), np.nan)
    mp, me = np.full(2, np.nan), np.full(2, np.nan)
    for s in SEEDS:
        v = cond["vec"][s]
        if gen == "A":
            Yp, Sp, Ye, Se = v["Yp"][idx], v["Sp"][idx], v["Ye"][idx], v["Se"][idx]
        else:
            Ys = {k: v["Scal"][k] + w * (v["Yc"][k] - v["Scal"][k]) for k in ("u", "pc", "ec")}
            Yp, Ye = Ys["u"] + Ys["pc"], Ys["u"] + Ys["ec"]
            Sp, Se = v["Sc"]["u"] + v["Sc"]["pc"], v["Sc"]["u"] + v["Sc"]["ec"]
        mp[s], me[s] = Yp.sum() / Sp.sum(), Ye.sum() / Se.sum()
        if cond["a1"]:
            ci = F.diff_ci(Yp, Sp, Ye, Se, draws)
            dm[s] = (ci["point"], ci["lo"], ci["hi"])
        if cond["a2"]:
            for j, h in enumerate(HEADS):
                if h in heads:
                    ci = F.ratio_ci(v[f"OBS_{h}"][idx], v[f"PRED_{h}"][idx], draws)
                    a2[s, j] = (ci["point"], ci["lo"], ci["hi"])
    return {"dm": dm, "a2": a2, "mp": mp, "me": me}


_CONDS: dict = {}
_DRAWS = None
_REP_OFFSET = 0


def _init(conds, rep_offset=0):
    global _CONDS, _DRAWS, _REP_OFFSET
    _CONDS = conds
    _REP_OFFSET = int(rep_offset)
    _DRAWS = libs()[2].boot_draws(N_LOGS, BOOT_B, BOOT_SEED)


def _chunk(task):
    name, a, b = task
    recs = [replicate(_CONDS[name], r, _DRAWS, _REP_OFFSET) for r in range(a, b)]
    return name, a, {k: np.stack([r[k] for r in recs]) for k in ("dm", "a2", "mp", "me")}


def run_condition_set(conds: dict, n_reps: int, workers: int, chunk: int = 25, log=print, rep_offset: int = 0) -> dict:
    """All replicates of all conditions, in `workers` spawned processes (each loads the module and the registered draws once).  Replicate `r` is the draw of index `r + rep_offset`."""
    tasks = [(name, a, min(a + chunk, n_reps)) for name in conds for a in range(0, n_reps, chunk)]
    parts: dict = {name: {} for name in conds}
    t0 = time.time()
    done = {name: 0 for name in conds}
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, initializer=_init, initargs=(conds, int(rep_offset))) as pool:
        for name, a, rec in pool.imap_unordered(_chunk, tasks):
            parts[name][a] = rec
            done[name] += len(rec["mp"])
            if done[name] == n_reps:
                log(f"  {name}: {n_reps} replicates done ({time.time() - t0:.0f} s elapsed)")
    out = {}
    for name in conds:
        keys = sorted(parts[name])
        out[name] = {k: np.concatenate([parts[name][a][k] for a in keys]) for k in ("dm", "a2", "mp", "me")}
    return out


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# decisions (the MODULE's functions, rules r1 and draft) and an inline restatement of the prose rule (PC4)
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def summarize(cond: dict, rec: dict) -> dict:
    dm, a2, mpv, mev = rec["dm"], rec["a2"], rec["mp"], rec["me"]
    n = len(dm)
    out = {"name": cond["name"], "gen": cond["gen"], "geom": cond["geom"], "n": int(n), "means": {"M_p": mpv.mean(axis=0).tolist(), "M_e": mev.mean(axis=0).tolist()}}
    if cond["a1"]:
        sd = np.array([[A.a1_verdict_r1(ci_of(dm[i, s])) != "ADMIT-A1" for s in SEEDS] for i in range(n)])                 # [n, 2] seed-level FAIL, module r1
        fail_r1 = np.array([A.a1_verdict_r1_seeds([ci_of(dm[i, 0]), ci_of(dm[i, 1])]) == "FAIL-A1" for i in range(n)])
        inline = np.array([any((dm[i, s, 1] > 0.0) or (dm[i, s, 0] > 0.05) for s in SEEDS) for i in range(n)])
        lo_hit = dm[:, :, 1] > 0.0
        pt_hit = dm[:, :, 0] > 0.05
        fail_draft = np.array([any(A.a1_verdict(ci_of(dm[i, s])) != "ADMIT-A1" for s in SEEDS) for i in range(n)])
        sd_draft = np.array([[A.a1_verdict(ci_of(dm[i, s])) != "ADMIT-A1" for s in SEEDS] for i in range(n)])
        out["A1"] = {
            "r1_any_seed": rate(fail_r1.sum(), n), "r1_seed0": rate(sd[:, 0].sum(), n), "r1_seed1": rate(sd[:, 1].sum(), n),
            "r1_both_seeds": rate((sd[:, 0] & sd[:, 1]).sum(), n), "r1_indep_expected_any": float(1 - (1 - sd[:, 0].mean()) * (1 - sd[:, 1].mean())),
            "r1_seed_level_by_conjunct": {"lower_ci_gt_0_only": int((lo_hit & ~pt_hit).sum()), "point_gt_0.05_only": int((pt_hit & ~lo_hit).sum()), "both": int((lo_hit & pt_hit).sum()), "n_seed_evaluations": int(2 * n)},
            "r1_any_seed_by_conjunct": {"lower_ci_gt_0_any_seed": rate(lo_hit.any(axis=1).sum(), n), "point_gt_0.05_any_seed": rate(pt_hit.any(axis=1).sum(), n)},
            "draft_any_seed": rate(fail_draft.sum(), n), "draft_seed0": rate(sd_draft[:, 0].sum(), n), "draft_seed1": rate(sd_draft[:, 1].sum(), n),
            "pc4_inline_disagreements": int((inline != fail_r1).sum()),
            "D_M_point_mean": dm[:, :, 0].mean(axis=0).tolist(), "D_M_point_sd": dm[:, :, 0].std(axis=0, ddof=1).tolist(), "D_M_half_width_mean": ((dm[:, :, 2] - dm[:, :, 1]) / 2.0).mean(axis=0).tolist(),
            "seed_corr_of_D_M_point": _finite(np.corrcoef(dm[:, 0, 0], dm[:, 1, 0])[0, 1])}
    if cond["a2"]:
        heads = cond.get("heads", HEADS)
        j_of = {h: j for j, h in enumerate(HEADS)}
        def cis(i, s):
            return {h: ci_of(a2[i, s, j_of[h]]) for h in heads}
        fail_r1, fail_draft = [], []
        per_head = {h: np.zeros((n, 2), bool) for h in heads}
        for i in range(n):
            f1 = f2 = False
            for s in SEEDS:
                c = cis(i, s)
                if A.a2_verdict_r1(c, tuple(heads)) != "ADMIT-A2":
                    f1 = True
                if A.a2_verdict(c, tuple(heads)) != "ADMIT-A2":
                    f2 = True
                for h in heads:
                    per_head[h][i, s] = c[h]["lo"] > 1.0
            fail_r1.append(f1)
            fail_draft.append(f2)
        fail_r1, fail_draft = np.array(fail_r1), np.array(fail_draft)
        inline = np.array([any(a2[i, s, j_of[h], 1] > 1.0 for s in SEEDS for h in heads) for i in range(n)])
        out["A2"] = {"heads": list(heads), "r1_any_head_any_seed": rate(fail_r1.sum(), n), "draft_any_head_any_seed": rate(fail_draft.sum(), n),
                     "r1_per_head_per_seed": {h: {f"seed{s}": rate(per_head[h][:, s].sum(), n) for s in SEEDS} for h in heads},
                     "r1_per_head_any_seed": {h: rate(per_head[h].any(axis=1).sum(), n) for h in heads},
                     "ratio_point_mean": {h: a2[:, :, j_of[h], 0].mean(axis=0).tolist() for h in heads}, "pc4_inline_disagreements": int((inline != fail_r1).sum())}
        if cond["a1"]:
            out["gate_r1_A1_or_A2"] = rate((np.array([A.a1_verdict_r1_seeds([ci_of(dm[i, 0]), ci_of(dm[i, 1])]) == "FAIL-A1" for i in range(n)]) | fail_r1).sum(), n)
    return out


def pc1_calibrated(summary: dict) -> dict:
    """PC1: a calibrated condition's replicate means: M_p, M_e within TOL_EP_M of 1, mean D_M point within TOL_DM of 0 (both seeds); A2 ratio means within TOL_A2_RATIO of 1."""
    ok_m = all(abs(m - 1.0) <= TOL_EP_M for m in summary["means"]["M_p"] + summary["means"]["M_e"])
    ok_d = all(abs(m) <= TOL_DM for m in summary["A1"]["D_M_point_mean"])
    ok_a = True
    if "A2" in summary:
        ok_a = all(abs(m - 1.0) <= TOL_A2_RATIO for h in summary["A2"]["heads"] for m in summary["A2"]["ratio_point_mean"][h])
    return {"pass": bool(ok_m and ok_d and ok_a), "M_p": summary["means"]["M_p"], "M_e": summary["means"]["M_e"], "D_M_point_mean": summary["A1"]["D_M_point_mean"],
            "A2_ratio_point_mean": summary.get("A2", {}).get("ratio_point_mean"), "tolerances": {"M": TOL_EP_M, "D_M": TOL_DM, "A2_ratio": TOL_A2_RATIO}}


def pc2_planted(summary: dict, delta: float) -> dict:
    return {"pass": bool(all(abs(m - delta) <= TOL_PLANT_DM for m in summary["A1"]["D_M_point_mean"])), "D_M_point_mean": summary["A1"]["D_M_point_mean"], "delta": delta}


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# steps
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def design_guard() -> str:
    if not os.path.exists(DESIGN):
        raise SystemExit(f"REFUSED: {os.path.abspath(DESIGN)} does not exist (the design must be written and hashed before the simulation runs)")
    return sha256_file(DESIGN)


def eq_ci(a: dict, b: dict, tol: float = 0.0) -> bool:
    return all(abs(float(a[k]) - float(b[k])) <= tol for k in ("point", "lo", "hi"))


def cmd_precompute(a) -> int:
    L, P, F = libs()
    dsha = design_guard()
    os.makedirs(a.workdir, exist_ok=True)
    sha = sha256_file(a.arrays_npz)
    assert sha == L.ARRAYS_SHA256, f"arrays.npz sha256 {sha} != the pinned {L.ARRAYS_SHA256}"
    z = np.load(a.arrays_npz, allow_pickle=False)
    z6, lab, log_idx = z["z6"], z["lab"], z["log_idx"]
    N = len(z6)
    draws = F.boot_draws(N_LOGS, BOOT_B, BOOT_SEED)
    u = np.random.default_rng(SUBSET_SEED).random(N)
    by_log = [np.nonzero(log_idx == l)[0] for l in range(N_LOGS)]
    S_full = A.make_stats(log_idx, N_LOGS)
    pk_p = P.rule_pick(z6)
    store, meta, pc3 = {}, {}, {"identity_vs_library": {}, "materialised_resample_vs_shortcut": {}, "analyze_candidate_vs_shortcut": None, "landed": {}}
    ident_ok, mat_ok = True, True
    landed = json.load(open(a.landed_result, encoding="utf-8"))["per_seed"] if a.landed_result else None
    for g, q in GEOMS.items():
        sub = u < q
        for s in SEEDS:
            zc = z6.copy()
            zc[..., EP] = np.where(sub[:, None], z[f"ep_OOF_s{s}"], z6[..., EP])
            comp = components(z6, zc, lab, log_idx)
            for k, v in comp["vec"].items():
                store[f"{g}__s{s}__{k}"] = v
            sc = scalars(comp["vec"])
            sec = comp["meta"]["s_e_changed"]
            fr = {}
            for d in sorted({x for ds in DELTAS.values() for x in ds}):
                f = plant_factor(comp["vec"], d)
                fr[str(d)] = {"f": f, "implied_sigma_gt_1_frac_of_changed_picks": float(((sc["a_e"] * f * sec) > 1.0).mean()) if len(sec) else float("nan")}
            meta[f"{g}|s{s}"] = {"n_changed": comp["meta"]["n_changed"], "n_sets": comp["meta"]["n_sets"], "frac_changed": comp["meta"]["n_changed"] / N, "scalars": sc,
                                 "calibrated_sigma_gt_1_frac_of_changed_picks": float(((sc["a_e"] * sec) > 1.0).mean()) if len(sec) else float("nan"), "plant": fr}
            # PC3 (i): the identity draw equals the library's set-level pipeline, bit for bit
            pk_c = P.rule_pick(zc)
            sig, y = P.sigmoid64(zc[..., EP]), lab[..., EP].astype(np.float64)
            t = L.pair_terms(sig, y, pk_p, pk_c)
            lib_dm = L.core_stats(S_full, t)["D_M"]
            my_dm = F.diff_ci(comp["vec"]["y_p"], comp["vec"]["s_p"], comp["vec"]["y_e"], comp["vec"]["s_e"], draws)
            rel = L.reliability_stats(S_full, lab, zc, z6, pk_p, pk_c)
            row = {"D_M_equal": eq_ci(lib_dm, my_dm), "D_M": {k: my_dm[k] for k in ("point", "lo", "hi")}}
            for h in HEADS:
                lib_h = rel[h]["composite_picks"]["deficit_form_sum_1my_over_sum_1msigma"]
                my_h = F.ratio_ci(comp["vec"][f"OBS_{h}"], comp["vec"][f"PRED_{h}"], draws)
                row[f"{h}_equal"] = eq_ci(lib_h, my_h)
            pc3["identity_vs_library"][f"{g}|s{s}"] = row
            ident_ok &= all(v for k, v in row.items() if k.endswith("_equal"))
            if g == "G78" and landed is not None:
                ld = landed[str(s)]["G"]["D_M"]
                pc3["landed"][f"s{s}"] = {"landed_D_M": {k: ld[k] for k in ("point", "lo", "hi")}, "max_abs_diff": max(abs(float(ld[k]) - float(my_dm[k])) for k in ("point", "lo", "hi"))}
            # PC3 (ii): three materialised set-level log-resamples equal the shortcut (raw condition)
            rawv = build_condition(comp["vec"], "raw")
            for r in (0, 1, 2):
                idx = outer_idx(r)
                sets = np.concatenate([by_log[l] for l in idx])
                new_li = np.repeat(np.arange(N_LOGS), [len(by_log[l]) for l in idx])
                St = P.Stats(new_li, draws, F, n_logs=N_LOGS)
                tr = L.pair_terms(sig[sets], y[sets], pk_p[sets], pk_c[sets])
                m_dm = L.core_stats(St, tr)["D_M"]
                s_dm = F.diff_ci(rawv["Yp"][idx], rawv["Sp"][idx], rawv["Ye"][idx], rawv["Se"][idx], draws)
                relr = L.reliability_stats(St, lab[sets], zc[sets], z6[sets], pk_p[sets], pk_c[sets])
                okr = eq_ci(m_dm, s_dm, 1e-12)
                for h in HEADS:
                    okr &= eq_ci(relr[h]["composite_picks"]["deficit_form_sum_1my_over_sum_1msigma"], F.ratio_ci(rawv[f"OBS_{h}"][idx], rawv[f"PRED_{h}"][idx], draws), 1e-12)
                pc3["materialised_resample_vs_shortcut"][f"{g}|s{s}|rep{r}"] = bool(okr)
                mat_ok &= okr
            if g == "G78" and s == 0:
                an = A.analyze_candidate(z6, zc, lab, S_full)             # PC3 (iii): the module's own analysis
                okc = eq_ci(an["G"]["D_M"], my_dm) and all(eq_ci(an["A2_reliability"][h], F.ratio_ci(comp["vec"][f"OBS_{h}"], comp["vec"][f"PRED_{h}"], draws)) for h in HEADS)
                pc3["analyze_candidate_vs_shortcut"] = bool(okc)
            print(f"precompute {g} seed {s}: n_changed {comp['meta']['n_changed']} ({comp['meta']['n_changed'] / N:.3%})", flush=True)
    pc3["pass"] = bool(ident_ok and mat_ok and pc3["analyze_candidate_vs_shortcut"] and all(v["max_abs_diff"] <= 1e-9 for v in pc3["landed"].values()))
    np.savez(os.path.join(a.workdir, "stats.npz"), **store)
    json.dump({"design_sha256": dsha, "arrays_sha256": sha, "meta": meta, "PC3": pc3, "ep_admission_sha256": sha256_file(os.path.join(ADM_DIR, "ep_admission.py"))},
              open(os.path.join(a.workdir, "precompute.json"), "w", encoding="utf-8", newline="\n"), indent=1, sort_keys=True)
    print(f"PRECOMPUTE PC3 pass={pc3['pass']} (identity {ident_ok}, materialised {mat_ok}, analyze_candidate {pc3['analyze_candidate_vs_shortcut']})")
    return 0 if pc3["pass"] else 3


def load_vec(workdir: str) -> dict:
    z = np.load(os.path.join(workdir, "stats.npz"), allow_pickle=False)
    out: dict = {}
    for k in z.files:
        g, s, name = k.split("__")
        out.setdefault(g, {}).setdefault(int(s[1:]), {})[name] = z[k]
    return out


def make_conditions(vec: dict) -> dict:
    conds = {}

    def add(name, gen, geom, a1, a2, kind, heads=None, **kw):
        conds[name] = {"name": name, "gen": gen, "geom": geom, "a1": a1, "a2": a2, "kind": kind, "kw": kw, **({"heads": heads} if heads else {}),
                       "vec": {s: build_condition(vec[geom][s], kind, **kw) for s in SEEDS}}
    for g in GEOMS:
        add(f"{g}|A|CAL", "A", g, True, True, "cal")
        for d in DELTAS[g]:
            add(f"{g}|A|PLANT_EP_{d}", "A", g, True, False, "plant_ep", delta=d)
    add("G78|A|RAW", "A", "G78", True, True, "raw")
    for h in HEADS:
        for rho in RHOS:
            add(f"G78|A|PLANT_A2_{h}_{rho}", "A", "G78", False, True, "plant_a2", heads=[h], head=h, rho=rho)
    add("G78|B|CAL", "B", "G78", True, False, "cal")
    for d in (0.05, 0.135):
        add(f"G78|B|PLANT_EP_{d}", "B", "G78", True, False, "plant_ep", delta=d)
    return conds


def identity_condition(vec: dict) -> dict:
    """PC6: P2 as the candidate: every per-log difference is zero (the candidate's picks are P2's)."""
    v = {k: vec["G78"][0][k] for k in vec["G78"][0]}
    for k in ("y_e", "s_e"):
        v[k] = v["y_p"] if k == "y_e" else v["s_p"]
    d = {"Yp": v["y_p"], "Sp": v["s_p"], "Ye": v["y_p"], "Se": v["s_p"]}
    return {"name": "IDENTITY", "gen": "A", "geom": "G78", "a1": True, "a2": False, "kind": "identity", "vec": {0: d, 1: d}}


def cmd_cost(a) -> int:
    design_guard()
    vec = load_vec(a.workdir)
    conds = make_conditions(vec)
    _, _, F = libs()
    draws = F.boot_draws(N_LOGS, BOOT_B, BOOT_SEED)
    res = {}
    for name in ("G78|A|CAL", "G78|A|PLANT_EP_0.135", "G78|A|PLANT_A2_NC_1.25", "G78|B|CAL"):
        t0 = time.time()
        for r in range(3):
            replicate(conds[name], r, draws)
        res[name] = (time.time() - t0) / 3.0
    n_conds = len(conds)
    est = sum(res[n] if n in res else (res["G78|A|CAL"] if c["a1"] and c["a2"] else (res["G78|A|PLANT_EP_0.135"] if c["a1"] else res["G78|A|PLANT_A2_NC_1.25"])) for n, c in conds.items())
    out = {"seconds_per_replicate_by_condition": res, "n_conditions": n_conds, "estimated_cpu_seconds_per_replicate_index_all_conditions": est,
           "estimated_cpu_hours_for_1000_reps": est * 1000 / 3600.0, "note": "single-process timing of 3 replicates each; the run uses --workers processes"}
    json.dump(out, open(os.path.join(a.workdir, "cost.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))
    return 0


def cmd_run(a) -> int:
    dsha = design_guard()
    L, P, F = libs()
    pre = json.load(open(os.path.join(a.workdir, "precompute.json"), encoding="utf-8"))
    assert pre["PC3"]["pass"], "PC3 did not pass: the per-log shortcut is not proven to be the estimator; refusing"
    assert pre["design_sha256"] == dsha, "SIM_DESIGN.md changed since precompute"
    vec = load_vec(a.workdir)
    conds = make_conditions(vec)
    conds["IDENTITY"] = identity_condition(vec)
    n_reps = {k: a.n_reps for k in conds}
    n_reps["IDENTITY"] = 50
    t0 = time.time()
    print(f"run: {len(conds)} conditions x {a.n_reps} replicates (IDENTITY x 50), {a.workers} workers, B = {BOOT_B}, seed {BOOT_SEED}, rep_offset {a.rep_offset}", flush=True)
    recs = {}
    for group in (["IDENTITY"], [k for k in conds if k != "IDENTITY"]):
        part = {k: conds[k] for k in group}
        recs.update(run_condition_set(part, n_reps[group[0]] if len(group) == 1 else a.n_reps, a.workers, log=lambda m: print(m, flush=True), rep_offset=a.rep_offset))
    wall = time.time() - t0
    np.savez_compressed(os.path.join(a.workdir, "replicates.npz"), **{f"{n}|{k}".replace("|", "__"): v for n, r in recs.items() for k, v in r.items()})
    summ = {n: summarize(conds[n], recs[n]) for n in conds}
    # the controls
    pc = {"PC1_calibrated_by_construction": {}, "PC2_check_has_teeth": {}, "PC3_shortcut_is_the_estimator": pre["PC3"], "PC4_decision_code_is_the_modules": {}, "PC6_identity_candidate": {}}
    for n in ("G78|A|CAL", "G25|A|CAL", "G05|A|CAL", "G78|B|CAL"):
        pc["PC1_calibrated_by_construction"][n] = pc1_calibrated(summ[n])
    raw1 = pc1_calibrated(summ["G78|A|RAW"])
    plant1 = pc1_calibrated(summ["G78|A|PLANT_EP_0.135"])
    pc["PC2_check_has_teeth"] = {"uncalibrated_real_replicates_FAIL_the_calibrated_check": {"expected": False, "got": raw1["pass"], "D_M_point_mean": raw1["D_M_point_mean"], "pass": (not raw1["pass"])},
                                 "plant_0.135_FAILs_the_calibrated_check": {"expected": False, "got": plant1["pass"], "pass": (not plant1["pass"])},
                                 "plant_0.135_PASSes_the_planted_form": {**pc2_planted(summ["G78|A|PLANT_EP_0.135"], 0.135)},
                                 "plant_0.05_PASSes_the_planted_form": {**pc2_planted(summ["G78|A|PLANT_EP_0.05"], 0.05)},
                                 "generator_B_plant_0.135_PASSes_the_planted_form": {**pc2_planted(summ["G78|B|PLANT_EP_0.135"], 0.135)}}
    pc["PC2_check_has_teeth"]["pass"] = bool(all(v["pass"] for v in pc["PC2_check_has_teeth"].values()))
    dis = {n: (s.get("A1", {}).get("pc4_inline_disagreements", 0) + s.get("A2", {}).get("pc4_inline_disagreements", 0)) for n, s in summ.items()}
    pc["PC4_decision_code_is_the_modules"] = {"disagreements_by_condition": dis, "pass": bool(sum(dis.values()) == 0), "ep_admission_sha256": sha256_file(os.path.join(ADM_DIR, "ep_admission.py"))}
    idn = recs["IDENTITY"]["dm"]
    pc["PC6_identity_candidate"] = {"all_D_M_exactly_0_0_0": bool((idn == 0.0).all()), "verdicts": summ["IDENTITY"]["A1"]["r1_any_seed"], "pass": bool((idn == 0.0).all() and summ["IDENTITY"]["A1"]["r1_any_seed"]["k"] == 0)}
    pc["PC1_pass"] = bool(all(v["pass"] for v in pc["PC1_calibrated_by_construction"].values()))
    pc["PC5_determinism"] = "run `repro` (separate process); filled by RESULT.md"
    ctrl_ok = bool(pc["PC1_pass"] and pc["PC2_check_has_teeth"]["pass"] and pre["PC3"]["pass"] and pc["PC4_decision_code_is_the_modules"]["pass"] and pc["PC6_identity_candidate"]["pass"])
    # the flags committed in advance (SIM_DESIGN.md section 8)
    cal = summ["G78|A|CAL"]
    f = {"F1_A1_false_refusal_wilson_upper_gt_0.10_G78": bool(cal["A1"]["r1_any_seed"]["wilson95"][1] > 0.10),
         "F2_A2_false_refusal_wilson_lower_gt_0.10_any_geometry": {g: bool(summ[f"{g}|A|CAL"]["A2"]["r1_any_head_any_seed"]["wilson95"][0] > 0.10) for g in GEOMS},
         "F3_power_0.135_wilson_lower_lt_0.95_G78": bool(summ["G78|A|PLANT_EP_0.135"]["A1"]["r1_any_seed"]["wilson95"][0] < 0.95),
         "F3_power_0.05_wilson_lower_lt_0.80_G78": bool(summ["G78|A|PLANT_EP_0.05"]["A1"]["r1_any_seed"]["wilson95"][0] < 0.80),
         "F4_generators_A_B_A1_false_refusal_differ_gt_3pp": bool(abs(cal["A1"]["r1_any_seed"]["rate"] - summ["G78|B|CAL"]["A1"]["r1_any_seed"]["rate"]) > 0.03),
         "F5_a_control_failed": bool(not ctrl_ok)}
    res = {"schema": "ep-admission-r1-sim-v1", "created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"), "design_sha256": dsha, "n_reps": a.n_reps, "rep_offset": int(a.rep_offset), "boot": {"B": BOOT_B, "seed": BOOT_SEED, "unit": "log"},
           "ep_admission_sha256": sha256_file(os.path.join(ADM_DIR, "ep_admission.py")), "wall_seconds": wall, "workers": a.workers, "controls": pc, "controls_ok_excluding_PC5": ctrl_ok, "flags": f, "conditions": summ,
           "geometry_meta": pre["meta"], "environment": {"python": sys.version.split()[0], "numpy": np.__version__, "arrays_sha256": pre["arrays_sha256"]}}
    json.dump(res, open(os.path.join(a.workdir, "sim_results.json"), "w", encoding="utf-8", newline="\n"), indent=1, sort_keys=True)
    print(f"RUN done in {wall:.0f} s; controls_ok_excluding_PC5={ctrl_ok}; -> {os.path.join(a.workdir, 'sim_results.json')}")
    return 0 if ctrl_ok else 3


def cmd_repro(a) -> int:
    """PC5: the first 20 replicates of 3 conditions, recomputed in this separate process with `--rep-offset`, equal the stored ones bit for bit.
    `--against <replicates.npz>` compares with another run's file instead of the workdir's (with `--rep-offset 0` and the banked EP-ADMISSION-R1 file: the offset-0 identity proof)."""
    design_guard()
    vec = load_vec(a.workdir)
    conds = make_conditions(vec)
    _, _, F = libs()
    draws = F.boot_draws(N_LOGS, BOOT_B, BOOT_SEED)
    z = np.load(a.against or os.path.join(a.workdir, "replicates.npz"))
    ok = True
    for name in ("G78|A|CAL", "G78|B|CAL", "G78|A|PLANT_EP_0.135"):
        for r in range(20):
            rec = replicate(conds[name], r, draws, a.rep_offset)
            for k in ("dm", "a2", "mp", "me"):
                ok &= bool(np.array_equal(rec[k], z[f"{name}|{k}".replace("|", "__")][r], equal_nan=True))
    out = {"bit_identical_first_20_replicates_of_3_conditions": bool(ok), "pid": os.getpid(), "rep_offset": int(a.rep_offset), "against": a.against or "workdir replicates.npz"}
    json.dump(out, open(os.path.join(a.workdir, a.out or "repro.json"), "w"), indent=1)
    print("REPRO", out)
    return 0 if ok else 3


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("precompute")
    p.add_argument("--arrays-npz", required=True)
    p.add_argument("--landed-result", default=None)
    p.add_argument("--workdir", required=True)
    p.set_defaults(fn=cmd_precompute)
    c = sub.add_parser("cost")
    c.add_argument("--workdir", required=True)
    c.set_defaults(fn=cmd_cost)
    r = sub.add_parser("run")
    r.add_argument("--workdir", required=True)
    r.add_argument("--n-reps", type=int, default=1000)
    r.add_argument("--workers", type=int, default=8)
    r.add_argument("--rep-offset", type=int, default=0, help="replicate r uses the draws of index r + offset in both generators; 0 = the banked EP-ADMISSION-R1 draws, 1000000 = the EP-ADMISSION-R2-CONFIRM fresh set")
    r.set_defaults(fn=cmd_run)
    q = sub.add_parser("repro")
    q.add_argument("--workdir", required=True)
    q.add_argument("--rep-offset", type=int, default=0, help="the offset the compared replicates were run with")
    q.add_argument("--against", default=None, help="replicates.npz to compare with (default: the workdir's own)")
    q.add_argument("--out", default=None, help="file name under --workdir for the result JSON (default repro.json)")
    q.set_defaults(fn=cmd_repro)
    return ap


def main(argv=None) -> int:
    ap = build_parser()
    a = ap.parse_args(argv)
    if os.path.splitdrive(os.path.abspath(a.workdir))[0].upper() == "C:":
        raise SystemExit("--workdir must be on D: (C: is nearly full)")
    return a.fn(a)


if __name__ == "__main__":
    mp.freeze_support()
    sys.exit(main())
