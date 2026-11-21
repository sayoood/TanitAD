"""EP-ADMISSION R1 re-simulation at the S0 OWN-FAN geometry of a search-output bank (EPADM-SEARCHBANK-S0; design: ../SIM_DESIGN_S0.md, hashed BEFORE the first run that reads a rate).  CPU only.

A thin SIBLING of `epadm_sim_c1.py` (itself a sibling of the frozen R1 simulator `ep_admission_sim.py`): it imports their calibration / planting / Wilson functions UNCHANGED and the frozen gate's decision
functions, and re-runs the replicate loop at N = 41 or 117 logs with the registered bootstrap `boot_draws(N, 10000, 20260927)`.  Two bases: C (carve-B clones, 3 seeds, real search outputs) and X (the real S0
stored-bank G25 statistics of EP-ADMISSION-R2-CONFIRM, 2 seeds).  Reads no label and no search output of any new token.

    python epadm_sim_s0.py cost  --pool P --stats S --geometry G --workdir W                         # 3 replicates per class (reads no rate)
    python epadm_sim_s0.py run   --job C-GS41-A --pool P --stats S --geometry G --workdir W [--rep-offset 0] [--n-reps N]
    python epadm_sim_s0.py repro --job C-GS41-A ...                                                    # PC5 (separate process)
    python epadm_sim_s0.py pc8   --stats S --banked B --workdir W                                      # Base X = the R1 simulator, bit for bit
    python epadm_sim_s0.py report --workdir W --geometry G [--pc3 ..] [--repro-glob ..]
Jobs: <base C|X>-<geometry GS41|GS117|GS43>-<generator A|T>.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
C1_DIR = os.environ.get("EPADM_C1_DIR") or os.path.join(HERE, "..", "..", "2026-10-19-epadm-searchbank", "code")
sys.path.insert(0, C1_DIR)
import epadm_sim_c1 as C1                                                              # noqa: E402

SB, A, R1 = C1.SB, C1.A, C1.R1
HEADS = R1.HEADS
DESIGN = os.path.join(HERE, "..", "SIM_DESIGN_S0.md")
GEOMETRY_SHA256 = "7a2e71e6d8e8c9707faab9bafa57fb6dd54b53c6ce6a471ffbf9e7b75124a7ca"
STATS_SHA256 = "4c34b63960ee0b2f3b5662e47f1a4a978a263f29c33963593cb9001b99d514a8"          # R2-CONFIRM stats.npz
BOOT_B, BOOT_SEED = 10000, 20260927
S0_BASE_SALT, S0_OUTER_SALT = 20261301, 20261302                                           # NEW (R1 20261027/28/29, EP-RELIAB 20261019, C1 20261201/02 are not reused)
USED_SALTS = (R1.OUTER_SALT, R1.WILD_SALT, R1.SUBSET_SEED, 20261019, C1.BASE_SALT, C1.OUTER_SALT)
SEEDS = {"C": (0, 1, 2), "X": (0, 1)}
TOKENS = {"GS41": 2782, "GS117": 8922, "GS43": 2919}
MARGINS, BAR_MARGIN = C1.MARGINS, C1.BAR_MARGIN
BAR_A1_FR_UPPER, BAR_A2_FR_UPPER, BAR_DAC_POWER_125 = C1.BAR_A1_FR_UPPER, C1.BAR_A2_FR_UPPER, C1.BAR_DAC_POWER_125     # the committed bars; do not edit
TOL_EP_M, TOL_DM, TOL_PLANT_DM, TOL_BASE_EXACT = R1.TOL_EP_M, R1.TOL_DM, R1.TOL_PLANT_DM, 1e-12
T_NAMES = ("CAL", "PLANT_A2_DAC_1.25")
JOBS = ("C-GS41-A", "C-GS41-T", "C-GS117-A", "C-GS117-T", "X-GS41-A", "X-GS117-A")


def sha256_file(p):
    return SB.sha256_file(p)


assert S0_BASE_SALT not in USED_SALTS and S0_OUTER_SALT not in USED_SALTS


# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# geometry and bases
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def load_geometry(path: str, name: str) -> dict:
    if sha256_file(path) != GEOMETRY_SHA256:
        raise SystemExit("REFUSED: geometry_s0.json is not the pinned file")
    g = json.load(open(path, encoding="utf-8"))
    pos = g["geometries"][name]
    logs = [g["logs"][p] for p in pos]
    counts = np.array([l["n_tokens"] for l in logs], np.int64)
    if int(counts.sum()) != TOKENS[name] or [l["pos"] for l in logs] != list(pos):
        raise SystemExit(f"REFUSED: geometry {name}: {int(counts.sum())} tokens != {TOKENS[name]} (PC7)")
    return {"name": name, "pos": np.array(pos, np.int64), "counts": counts, "n": len(pos), "logs": [l["log"] for l in logs]}


def pin_design() -> str:
    if not os.path.exists(DESIGN):
        raise SystemExit(f"REFUSED: {os.path.abspath(DESIGN)} does not exist")
    side = DESIGN[:-3] + ".sha256"
    if not os.path.exists(side) or open(side, encoding="utf-8").read().split()[0].lower() != sha256_file(DESIGN):
        raise SystemExit("REFUSED: SIM_DESIGN_S0.sha256 is missing or does not equal SIM_DESIGN_S0.md (hash it BEFORE the first run)")
    return sha256_file(DESIGN)


def base_c(pool: dict, counts: np.ndarray, rep: int, offset: int = 0) -> dict:
    """Base C, one replicate: every S0 log l draws ONE source carve-B log (probability proportional to its set count), then counts[l] sets WITH REPLACEMENT from that log's sets; the same draw for all 3 seeds."""
    rng = np.random.default_rng([S0_BASE_SALT, rep + offset])
    n_src = len(pool["log_names"])
    sizes = np.bincount(pool["log_idx"], minlength=n_src).astype(np.float64)
    src = rng.choice(n_src, size=len(counts), p=sizes / sizes.sum())
    ids_by = [np.nonzero(pool["log_idx"] == l)[0] for l in range(n_src)]
    sel, sel_log = [], []
    for l in range(len(counts)):
        ids = ids_by[int(src[l])]
        sel.append(ids[rng.integers(0, len(ids), int(counts[l]))])
        sel_log.append(np.full(int(counts[l]), l, np.int64))
    sel, sel_log = np.concatenate(sel), np.concatenate(sel_log)
    base = {s: {k: np.bincount(sel_log, weights=v[sel], minlength=len(counts)).astype(np.float64) for k, v in pool["tok"][s].items()} for s in SEEDS["C"]}
    return {"base": base, "src": src, "sel": sel, "sel_log": sel_log}


def check_clone(pool: dict, bc: dict, counts: np.ndarray) -> bool:
    """PC7 (runtime on replicate 0, and tested): every simulated S0 log took ALL its sets from ONE source log, and exactly counts[l] of them."""
    for l in range(len(counts)):
        ids = bc["sel"][bc["sel_log"] == l]
        if len(ids) != int(counts[l]) or set(pool["log_idx"][ids].tolist()) != {int(bc["src"][l])}:
            return False
    return True


def load_base_x(stats_npz: str, pos: np.ndarray) -> dict:
    if sha256_file(stats_npz) != STATS_SHA256:
        raise SystemExit("REFUSED: stats.npz is not the R2-CONFIRM file")
    z = np.load(stats_npz, allow_pickle=False)
    out = {}
    for s in SEEDS["X"]:
        pre = f"G25__s{s}__"
        out[s] = {k[len(pre):]: z[k][pos].astype(np.float64) for k in z.files if k.startswith(pre)}
    return out


def pool_events(pool: dict) -> dict:
    """R-EVENTS / R-DOSE: the distinct unsafe-pick counts and the changed-pick share of the Base C pool, per seed."""
    return {str(s): {"unsafe_picks": {h: int((pool["tok"][s][f"OBS_{h}"] > 0).sum()) for h in HEADS}, "changed": int(pool["n_changed"][s]), "changed_share": pool["n_changed"][s] / pool["n_sets"]} for s in SEEDS["C"]}


# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# a replicate: the REAL estimator on one draw of logs (generator A) or on the fixed logs (generator T)
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def outer_idx(rep: int, n: int, gen: str, offset: int = 0) -> np.ndarray:
    if gen == "T":
        return np.arange(n)
    return np.random.default_rng([S0_OUTER_SALT, rep + offset]).integers(0, n, n)


def replicate(cond: dict, base: dict, seeds: tuple, gen: str, rep: int, draws: np.ndarray, offset: int = 0, idx: np.ndarray | None = None) -> dict:
    """-> {dm [S,3], a2 [S,3,3], mp [S], me [S], k0 [S,3] (replicate base has no observed unsafe pick for the head: the NaN-policy flag), br [S,3] (observed / predicted at the base, before the log redraw)}."""
    _, _, F = R1.libs()
    n = len(base[seeds[0]]["y_p"])
    idx = outer_idx(rep, n, gen, offset) if idx is None else idx
    S = len(seeds)
    dm, a2 = np.full((S, 3), np.nan), np.full((S, 3, 3), np.nan)
    mp, me, br = np.full(S, np.nan), np.full(S, np.nan), np.full((S, 3), np.nan)
    k0 = np.zeros((S, 3), bool)
    for si, s in enumerate(seeds):
        v = base[s]
        if cond["kind"] == "identity":
            d = {"Yp": v["y_p"], "Sp": v["s_p"], "Ye": v["y_p"], "Se": v["s_p"]}
        else:
            d = R1.build_condition(v, cond["kind"], **cond["kw"])
            for j, h in enumerate(HEADS):
                rho_h = cond["kw"].get("rho", 1.0) if (cond["kind"] == "plant_a2" and cond["kw"].get("head") == h) else 1.0
                if float(v[f"OBS_{h}"].sum()) == 0.0:                                   # SIM_DESIGN_S0 section 6: k_h = 0 would wipe the prediction (0/0); the real gate sees PRED > 0 -> ratio 0
                    k0[si, j] = True
                    d[f"PRED_{h}"] = v[f"PRED_{h}"] / rho_h
                else:
                    br[si, j] = float(v[f"OBS_{h}"].sum()) / float(d[f"PRED_{h}"].sum())
        Yp, Sp, Ye, Se = d["Yp"][idx], d["Sp"][idx], d["Ye"][idx], d["Se"][idx]
        mp[si], me[si] = Yp.sum() / Sp.sum(), Ye.sum() / Se.sum()
        if cond["a1"]:
            ci = F.diff_ci(Yp, Sp, Ye, Se, draws)
            dm[si] = (ci["point"], ci["lo"], ci["hi"])
        for j, h in enumerate(HEADS):
            if cond["a2"] and h in cond["heads"]:
                ci = F.ratio_ci(d[f"OBS_{h}"][idx], d[f"PRED_{h}"][idx], draws)
                a2[si, j] = (ci["point"], ci["lo"], ci["hi"])
    return {"dm": dm, "a2": a2, "mp": mp, "me": me, "k0": k0, "br": br}


def parse_job(job: str) -> tuple:
    b, g, gen = job.split("-")
    if b not in SEEDS or g not in TOKENS or gen not in ("A", "T") or (b == "X" and gen == "T"):
        raise SystemExit(f"REFUSED: unknown job {job!r}")
    return b, g, gen


def job_conditions(job: str) -> dict:
    b, g, gen = parse_job(job)
    c = C1.conditions()
    if gen == "T":
        return {n: {**c[n], "n": 1000} for n in T_NAMES}
    return c


def make_sampler(job: str, a) -> tuple:
    """-> (geometry, seeds, base_fn(rep, offset) -> base dict, extras)."""
    b, g, gen = parse_job(job)
    geo = load_geometry(a.geometry, g)
    if b == "C":
        pool = C1.load_pool(a.pool)
        if not check_clone(pool, base_c(pool, geo["counts"], 0, 0), geo["counts"]):
            raise SystemExit("REFUSED: PC7 clone integrity failed on replicate 0")
        return geo, SEEDS["C"], (lambda rep, off=0: base_c(pool, geo["counts"], rep, off)["base"]), {"pool_events": pool_events(pool), "pool_sets": pool["n_sets"], "PC7_clone_integrity_rep0": True,
                                                                                                    "bank_manifests": {n: sha256_file(os.path.join(a.pool, n, "MANIFEST.sha256.json")) for n in ("p2", "s0", "s1", "s2")}}
    bx = load_base_x(a.stats, geo["pos"])
    return geo, SEEDS["X"], (lambda rep, off=0: bx), {"stats_sha256": STATS_SHA256}


def run_job(job: str, a, n_reps_override: int = 0, log=print) -> dict:
    geo, seeds, base_fn, extras = make_sampler(job, a)
    _, _, gen = parse_job(job)
    _, _, F = R1.libs()
    draws = F.boot_draws(geo["n"], BOOT_B, BOOT_SEED)
    conds, out = job_conditions(job), {}
    for name, cond in conds.items():
        t0 = time.time()
        nr = n_reps_override or cond["n"]
        recs = [replicate(cond, base_fn(r, a.rep_offset), seeds, gen, r, draws, a.rep_offset) for r in range(nr)]
        out[name] = {k: np.stack([r[k] for r in recs]) for k in ("dm", "a2", "mp", "me", "k0", "br")}
        log(f"  {job} {name}: {nr} replicates ({time.time() - t0:.0f} s)")
    return {"recs": out, "extras": extras, "geo": geo, "seeds": seeds}


# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# decisions: the frozen gate's functions and an inline restatement (PC4); a NaN bound fails closed in both
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def ci_of(row) -> dict:
    return {"point": float(row[0]), "lo": float(row[1]), "hi": float(row[2])}


def a1_refuse_module(dm_i) -> bool:
    return A.a1_verdict_r1_seeds([ci_of(r) for r in dm_i]) == "FAIL-A1"


def a1_refuse_inline(dm_i) -> bool:
    return any(not (r[1] <= 0.0 and r[0] <= 0.05) for r in dm_i)


def a2_refuse_module(a2_i, heads, margin) -> bool:
    j_of = {h: j for j, h in enumerate(HEADS)}
    cis = [{h: ci_of(a2_i[si, j_of[h]]) for h in heads} for si in range(a2_i.shape[0])]
    if margin == 1.0:
        if tuple(heads) == tuple(A.A2_HEADS):
            return A.a2_verdict_r1_seeds(cis) == "FAIL-A2"
        return any(A.a2_verdict_r1(c, tuple(heads)) != "ADMIT-A2" for c in cis)
    return any(not all(A.a2_head_ok_r1(c[h], margin) for h in heads) for c in cis)


def a2_refuse_inline(a2_i, heads, margin) -> bool:
    j_of = {h: j for j, h in enumerate(HEADS)}
    return any(not (a2_i[si, j_of[h], 1] <= margin) for si in range(a2_i.shape[0]) for h in heads)


def failclosed_extra(k0_i, heads) -> bool:
    """The second NaN convention (SIM_DESIGN_S0 section 6): a replicate-seed-head with no observed unsafe pick in the base counts as a refusal."""
    j_of = {h: j for j, h in enumerate(HEADS)}
    return bool(any(k0_i[si, j_of[h]] for si in range(k0_i.shape[0]) for h in heads))


def summarize(name: str, cond: dict, rec: dict, seeds: tuple) -> dict:
    dm, a2, k0, n, S = rec["dm"], rec["a2"], rec["k0"], len(rec["dm"]), len(seeds)
    R = R1.rate
    out = {"name": name, "n": int(n), "means": {"M_p": rec["mp"].mean(axis=0).tolist(), "M_e": rec["me"].mean(axis=0).tolist()}}
    dis = 0
    if cond["a1"]:
        mod = np.array([a1_refuse_module(dm[i]) for i in range(n)])
        inl = np.array([a1_refuse_inline(dm[i]) for i in range(n)])
        dis += int((mod != inl).sum())
        lo_hit, pt_hit = dm[:, :, 1] > 0.0, dm[:, :, 0] > 0.05
        seed_fail = np.array([[A.a1_verdict_r1(ci_of(dm[i, si])) != "ADMIT-A1" for si in range(S)] for i in range(n)])
        out["A1"] = {"r1_any_seed": R(mod.sum(), n), "r1_per_seed": [R(seed_fail[:, si].sum(), n) for si in range(S)],
                     "by_conjunct_seed_evaluations": {"lower_ci_gt_0_only": int((lo_hit & ~pt_hit).sum()), "point_gt_0.05_only": int((pt_hit & ~lo_hit).sum()), "both": int((lo_hit & pt_hit).sum()), "n": int(S * n)},
                     "D_M_point_mean": dm[:, :, 0].mean(axis=0).tolist(), "D_M_point_sd": dm[:, :, 0].std(axis=0, ddof=1).tolist() if n > 1 else None,
                     "D_M_half_width_mean": ((dm[:, :, 2] - dm[:, :, 1]) / 2.0).mean(axis=0).tolist(), "n_nan_ci": int(np.isnan(dm[:, :, 1]).sum())}
    if cond["a2"]:
        heads, j_of = cond["heads"], {h: j for j, h in enumerate(HEADS)}
        out["A2"] = {"heads": heads, "n_nan_ci": int(np.isnan(a2[:, :, [j_of[h] for h in heads], 1]).sum()), "ratio_point_mean": {h: a2[:, :, j_of[h], 0].mean(axis=0).tolist() for h in heads},
                     "k0_replicate_seed_heads": {h: int(k0[:, :, j_of[h]].sum()) for h in HEADS}, "by_margin": {}}
        for t in MARGINS:
            mod = np.array([a2_refuse_module(a2[i], heads, t) for i in range(n)])
            inl = np.array([a2_refuse_inline(a2[i], heads, t) for i in range(n)])
            fc = np.array([bool(mod[i] or failclosed_extra(k0[i], heads)) for i in range(n)])
            dis += int((mod != inl).sum())
            out["A2"]["by_margin"][f"{t:.2f}"] = {"any_head_any_seed": R(mod.sum(), n), "any_head_any_seed_failclosed": R(fc.sum(), n),
                                                  "per_head_any_seed": {h: R(np.array([a2_refuse_inline(a2[i], [h], t) for i in range(n)]).sum(), n) for h in heads},
                                                  "per_head_any_seed_failclosed": {h: R(np.array([bool(a2_refuse_inline(a2[i], [h], t) or failclosed_extra(k0[i], [h])) for i in range(n)]).sum(), n) for h in heads}}
    out["base_ratio_obs_over_pred_min_max"] = [float(np.nanmin(rec["br"])), float(np.nanmax(rec["br"]))] if np.isfinite(rec["br"]).any() else None
    out["pc4_module_vs_inline_disagreements"] = int(dis)
    return out


def decide(a1_fr_upper: float, a2_fr_upper: float, dac_power_125: float) -> dict:
    """The committed bars as a pure function of three rates (the same constants as C1's `decide`)."""
    bars = {"A1_r1_false_refusal_wilson_upper_le_5pct": a1_fr_upper <= BAR_A1_FR_UPPER, "A2_1.05_false_refusal_wilson_upper_le_5pct": a2_fr_upper <= BAR_A2_FR_UPPER,
            "DAC_power_x1.25_margin1.05_point_ge_70pct": dac_power_125 >= BAR_DAC_POWER_125}
    return {"bars": bars, "all_met": bool(all(bars.values())), "missed": [k for k, v in bars.items() if not v]}


def bar_status(c_ok: bool, x_ok: bool) -> str:
    """SIM_DESIGN_S0 section 7: MET = met on Base C AND on Base X; MISSED = missed on Base C; BASE-DEPENDENT = met on C, missed on X."""
    if not c_ok:
        return "MISSED"
    return "MET" if x_ok else "BASE-DEPENDENT"


def bar_inputs(summ: dict, conv: str = "primary") -> tuple:
    cal, dac = summ["CAL"], summ["PLANT_A2_DAC_1.25"]
    a1 = cal["A1"]["r1_any_seed"]
    key = "any_head_any_seed" if conv == "primary" else "any_head_any_seed_failclosed"
    a2 = cal["A2"]["by_margin"]["1.05"][key]
    pk = "per_head_any_seed" if conv == "primary" else "per_head_any_seed_failclosed"
    pw = dac["A2"]["by_margin"]["1.05"][pk]["DAC"]
    return a1, a2, pw


# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# controls on one job's records
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def job_controls(summ: dict, recs: dict) -> dict:
    pc = {}
    cal = summ["CAL"]
    pc["PC1a_calibrated_by_construction"] = {"pass": bool(all(abs(m - 1.0) <= TOL_EP_M for m in cal["means"]["M_p"] + cal["means"]["M_e"]) and all(abs(m) <= TOL_DM for m in cal["A1"]["D_M_point_mean"])),
                                             "M_p": cal["means"]["M_p"], "M_e": cal["means"]["M_e"], "D_M_point_mean": cal["A1"]["D_M_point_mean"], "tolerances": {"M": TOL_EP_M, "D_M": TOL_DM}}
    br = recs["CAL"]["br"]
    pc["PC1b_base_exactness"] = {"pass": bool(np.all(np.abs(br[np.isfinite(br)] - 1.0) <= TOL_BASE_EXACT)), "max_abs_dev": float(np.nanmax(np.abs(br - 1.0))) if np.isfinite(br).any() else None,
                                 "REPORTED_A2_ratio_point_mean_after_redraw": cal["A2"]["ratio_point_mean"]}
    teeth = {}
    for d in C1.DELTAS:
        if f"PLANT_EP_{d}" in summ:
            s = summ[f"PLANT_EP_{d}"]
            teeth[f"plant_{d}_reads_the_planted_delta"] = {"pass": bool(all(abs(m - d) <= TOL_PLANT_DM for m in s["A1"]["D_M_point_mean"])), "D_M_point_mean": s["A1"]["D_M_point_mean"]}
            teeth[f"plant_{d}_FAILs_the_calibrated_check"] = {"pass": bool(not all(abs(m) <= TOL_DM for m in s["A1"]["D_M_point_mean"]))}
    pc["PC2_check_has_teeth"] = {**teeth, "pass": bool(all(v["pass"] for v in teeth.values())) if teeth else None}
    dis = {n: s["pc4_module_vs_inline_disagreements"] for n, s in summ.items()}
    pc["PC4_decision_code_is_the_modules"] = {"disagreements_by_condition": dis, "pass": bool(sum(dis.values()) == 0)}
    if "IDENTITY" in recs:
        idn = recs["IDENTITY"]["dm"]
        pc["PC6_identity_candidate"] = {"all_D_M_exactly_0": bool((idn == 0.0).all()), "pass": bool((idn == 0.0).all() and summ["IDENTITY"]["A1"]["r1_any_seed"]["k"] == 0)}
    return pc


# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def cmd_cost(a) -> int:
    pin_design()
    out = {}
    for job in a.jobs.split(","):
        geo, seeds, base_fn, _ = make_sampler(job, a)
        _, _, gen = parse_job(job)
        _, _, F = R1.libs()
        draws = F.boot_draws(geo["n"], BOOT_B, BOOT_SEED)
        conds = job_conditions(job)
        sec = {}
        for name in ("CAL", "PLANT_EP_0.135", "PLANT_A2_DAC_1.25"):
            if name not in conds:
                continue
            t0 = time.time()
            for r in range(3):
                replicate(conds[name], base_fn(r), seeds, gen, r, draws)
            sec[name] = (time.time() - t0) / 3.0
        est = sum(c["n"] * (sec["CAL"] if c["a1"] and c["a2"] else sec.get("PLANT_EP_0.135", sec["CAL"]) if c["a1"] else sec["PLANT_A2_DAC_1.25"]) for c in conds.values())
        out[job] = {"seconds_per_replicate": sec, "estimated_cpu_seconds": est}
    os.makedirs(a.workdir, exist_ok=True)
    json.dump(out, open(os.path.join(a.workdir, "cost.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))
    return 0


def cmd_run(a) -> int:
    dsha = pin_design()
    t0 = time.time()
    print(f"run {a.job}: B = {BOOT_B}, seed {BOOT_SEED}, rep_offset {a.rep_offset}, salts {S0_BASE_SALT}/{S0_OUTER_SALT}", flush=True)
    r = run_job(a.job, a, a.n_reps, log=lambda m: print(m, flush=True))
    os.makedirs(a.workdir, exist_ok=True)
    np.savez_compressed(os.path.join(a.workdir, f"replicates_{a.job}.npz"), **{f"{n}__{k}": v for n, rec in r["recs"].items() for k, v in rec.items()})
    json.dump({"design_sha256": dsha, "job": a.job, "wall_seconds": time.time() - t0, "rep_offset": a.rep_offset, "n_logs": r["geo"]["n"], "n_tokens": int(r["geo"]["counts"].sum()), "extras": r["extras"],
               "code_sha256": {"epadm_sim_s0.py": sha256_file(os.path.abspath(__file__)), "epadm_sim_c1.py": sha256_file(os.path.join(C1_DIR, "epadm_sim_c1.py")), "ep_admission.py": SB.GATE_SHA256,
                               "ep_admission_sim.py": sha256_file(os.path.join(C1.R1_DIR, "ep_admission_sim.py"))}}, open(os.path.join(a.workdir, f"run_{a.job}.json"), "w"), indent=1)
    print(f"JOB {a.job} DONE_MARKER ({time.time() - t0:.0f} s)")
    return 0


def cmd_repro(a) -> int:
    pin_design()
    geo, seeds, base_fn, _ = make_sampler(a.job, a)
    _, _, gen = parse_job(a.job)
    _, _, F = R1.libs()
    draws = F.boot_draws(geo["n"], BOOT_B, BOOT_SEED)
    z = np.load(os.path.join(a.workdir, f"replicates_{a.job}.npz"))
    conds, ok = job_conditions(a.job), True
    for name in T_NAMES:
        for r in range(20):
            rec = replicate(conds[name], base_fn(r, a.rep_offset), seeds, gen, r, draws, a.rep_offset)
            for k in ("dm", "a2", "mp", "me", "k0", "br"):
                ok &= bool(np.array_equal(rec[k], z[f"{name}__{k}"][r], equal_nan=True))
    out = {"job": a.job, "pass": bool(ok), "bit_identical_first_20_replicates_of_2_conditions": bool(ok), "pid": os.getpid()}
    json.dump(out, open(os.path.join(a.workdir, f"repro_{a.job}.json"), "w"), indent=1)
    print("REPRO", out)
    return 0 if ok else 3


def cmd_pc8(a) -> int:
    """PC8: with R1's own outer draws (offset 1,000,000) the Base X / GS117 DAC x1.25 replicates equal the banked R2-CONFIRM replicates bit for bit: this code is the R1 simulator."""
    geo = load_geometry(a.geometry, "GS117")
    bx = load_base_x(a.stats, geo["pos"])
    _, _, F = R1.libs()
    draws = F.boot_draws(117, BOOT_B, BOOT_SEED)
    banked = np.load(a.banked, allow_pickle=False)
    cond = C1.conditions()["PLANT_A2_DAC_1.25"]
    ok, n = True, 20
    for r in range(n):
        rec = replicate(cond, bx, SEEDS["X"], "A", r, draws, 0, idx=R1.outer_idx(r, 1000000))
        for k in ("dm", "a2", "mp", "me"):
            ok &= bool(np.array_equal(rec[k], banked[f"G25__A__PLANT_A2_DAC_1.25__{k}"][r], equal_nan=True))
    out = {"pass": bool(ok), "replicates_compared": n, "banked_sha256": sha256_file(a.banked), "stats_sha256": STATS_SHA256, "offset": 1000000}
    os.makedirs(a.workdir, exist_ok=True)
    json.dump(out, open(os.path.join(a.workdir, "pc8.json"), "w"), indent=1)
    print("PC8", out)
    return 0 if ok else 3


def load_job(workdir: str, job: str) -> tuple:
    z = np.load(os.path.join(workdir, f"replicates_{job}.npz"))
    recs = {}
    for k in z.files:
        n, key = k.split("__")
        recs.setdefault(n, {})[key] = z[k]
    run = json.load(open(os.path.join(workdir, f"run_{job}.json")))
    return recs, run


def cmd_report(a) -> int:
    dsha = pin_design()
    seeds_of = {"C": SEEDS["C"], "X": SEEDS["X"]}
    res = {"schema": "ep-admission-s0bank-sim-v1", "created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"), "design_sha256": dsha, "jobs": {}, "geometries": {}}
    ctrl_ok = True
    for job in a.jobs.split(","):
        b, g, gen = parse_job(job)
        recs, run = load_job(a.workdir, job)
        conds = job_conditions(job)
        summ = {n: summarize(n, conds[n], recs[n], seeds_of[b]) for n in conds}
        entry = {"summaries": summ, "run": run}
        if gen == "A":
            pc = job_controls(summ, recs)
            rp = os.path.join(a.workdir, f"repro_{job}.json")
            pc["PC5_determinism"] = json.load(open(rp)) if os.path.exists(rp) else {"pass": None}
            entry["controls"] = pc
            ok = all(pc[k].get("pass") is True for k in pc if k not in ("PC1b_base_exactness",)) and pc["PC1b_base_exactness"]["pass"]
            entry["controls_ok"] = bool(ok)
            ctrl_ok &= bool(ok)
        a1, a2, pw = bar_inputs(summ, "primary")
        a1f, a2f, pwf = bar_inputs(summ, "failclosed")
        entry["bars_primary"] = {"A1_r1_false_refusal": a1, "A2_1.05_false_refusal": a2, "DAC_power_x1.25_margin1.05": pw, "decision": decide(a1["wilson95"][1], a2["wilson95"][1], pw["rate"])}
        entry["bars_failclosed"] = {"A1_r1_false_refusal": a1f, "A2_1.05_false_refusal": a2f, "DAC_power_x1.25_margin1.05": pwf, "decision": decide(a1f["wilson95"][1], a2f["wilson95"][1], pwf["rate"])}
        res["jobs"][job] = entry
    for g in sorted({parse_job(j)[1] for j in res["jobs"]}):
        c, x = res["jobs"].get(f"C-{g}-A"), res["jobs"].get(f"X-{g}-A")
        if not (c and x):
            continue
        st = {k: bar_status(c["bars_primary"]["decision"]["bars"][k], x["bars_primary"]["decision"]["bars"][k]) for k in c["bars_primary"]["decision"]["bars"]}
        fl = {k: bar_status(c["bars_failclosed"]["decision"]["bars"][k], x["bars_failclosed"]["decision"]["bars"][k]) for k in st}
        t = res["jobs"].get(f"C-{g}-T")
        res["geometries"][g] = {"status_primary_convention": st, "status_failclosed_convention": fl, "all_met": bool(all(v == "MET" for v in st.values())), "R-NAN": bool(st != fl),
                                "R-GEN_T_vs_A_A1_fr_pp": (abs(t["bars_primary"]["A1_r1_false_refusal"]["rate"] - c["bars_primary"]["A1_r1_false_refusal"]["rate"]) * 100 if t else None),
                                "R-SEED_C_vs_X_pp": {"A1_fr": abs(c["bars_primary"]["A1_r1_false_refusal"]["rate"] - x["bars_primary"]["A1_r1_false_refusal"]["rate"]) * 100,
                                                     "DAC_power": abs(c["bars_primary"]["DAC_power_x1.25_margin1.05"]["rate"] - x["bars_primary"]["DAC_power_x1.25_margin1.05"]["rate"]) * 100}}
    p3 = json.load(open(a.pc3)) if a.pc3 and os.path.exists(a.pc3) else {"pass": None}
    p8 = json.load(open(os.path.join(a.workdir, "pc8.json"))) if os.path.exists(os.path.join(a.workdir, "pc8.json")) else {"pass": None}
    res["controls_global"] = {"PC3_shortcut_is_the_estimator": p3, "PC8_code_is_the_R1_simulator": p8}
    res["controls_ok"] = bool(ctrl_ok and p3.get("pass") is True and p8.get("pass") is True)
    json.dump(res, open(os.path.join(a.workdir, "sim_results.json"), "w", encoding="utf-8", newline="\n"), indent=1, sort_keys=True, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(f"REPORT controls_ok={res['controls_ok']} " + " ".join(f"{g}:all_met={v['all_met']}" for g, v in res["geometries"].items()))
    return 0 if res["controls_ok"] else 3


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for nm, fn in (("cost", cmd_cost), ("run", cmd_run), ("repro", cmd_repro)):
        p = sub.add_parser(nm)
        p.add_argument("--pool", default=None); p.add_argument("--stats", default=None); p.add_argument("--geometry", required=True); p.add_argument("--workdir", required=True)
        p.add_argument("--rep-offset", type=int, default=0)
        if nm == "cost":
            p.add_argument("--jobs", default=",".join(JOBS))
        else:
            p.add_argument("--job", required=True)
        if nm == "run":
            p.add_argument("--n-reps", type=int, default=0)
        p.set_defaults(fn=fn)
    q = sub.add_parser("pc8")
    q.add_argument("--stats", required=True); q.add_argument("--banked", required=True); q.add_argument("--geometry", required=True); q.add_argument("--workdir", required=True)
    q.set_defaults(fn=cmd_pc8)
    r = sub.add_parser("report")
    r.add_argument("--workdir", required=True); r.add_argument("--jobs", default=",".join(JOBS)); r.add_argument("--pc3", default=None)
    r.set_defaults(fn=cmd_report)
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
