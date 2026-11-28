"""EP-ADMISSION N1 re-simulation: the frozen A2 statistic restricted to the CHANGED picks, at the S0 own-fan geometry of a search-output bank (EPADM-S0BANK DQ3, arm N1; design: ../SIM_DESIGN_N1.md, hashed
BEFORE the first run that reads a rate).  CPU only.

A thin SIBLING of `epadm_sim_s0.py` (itself a sibling of `epadm_sim_c1.py` and the frozen R1 simulator): it imports their calibration / planting / Wilson / decision functions UNCHANGED and the statistic itself from
`epadm_n1.py`.  What is new: (i) the changed-pick split of the per-log sufficient statistics (Base C from the carve-B bank read through the gate's loaders, Base X recomputed from the banked S0 arrays and CHECKED
bit for bit against the R2-CONFIRM stats), (ii) calibration of the null AT THE CHANGED PICKS, (iii) NEW replicate salts, (iv) a non-degenerate cross-check generator P.  Reads no label or search output of any NEW token.

    python epadm_sim_n1.py basex  --arrays A --stats S --workdir W                       # per-log changed / unchanged split of the Base X statistics + PC9 (+ its RED arm)
    python epadm_sim_n1.py cost   --pool P --stats S --geometry G --basex B --workdir W   # 3 replicates per class (reads no rate)
    python epadm_sim_n1.py run    --job C-GS41-A --pool P --stats S --geometry G --basex B --workdir W [--rep-offset 0] [--n-reps N]
    python epadm_sim_n1.py repro  --job C-GS41-A ...                                      # PC5 (separate process)
    python epadm_sim_n1.py pc3    --workdir W                                             # N1 shortcut == the gate's Stats path; all-changed N1 == A2; zero-changed refuses (SYNTHETIC bank)
    python epadm_sim_n1.py report --workdir W [--pc3 ..] [--pc8 ..]
Jobs: <base C|X>-<geometry GS41|GS117>-<generator A|P>   (P: Base C only).
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import epadm_sim_s0 as S0                                                              # noqa: E402
import epadm_n1 as N1                                                                  # noqa: E402

C1, SB, A, R1 = S0.C1, S0.SB, S0.A, S0.R1
HEADS = R1.HEADS
DESIGN = os.path.join(HERE, "..", "SIM_DESIGN_N1.md")
BOOT_B, BOOT_SEED = 10000, 20260927
N1_BASE_SALT, N1_OUTER_SALT = 20261401, 20261402                                       # NEW (R1 20261027/28/29, EP-RELIAB 20261019, C1 20261201/02, S0BANK 20261301/02 are not reused)
USED_SALTS = tuple(S0.USED_SALTS) + (S0.S0_BASE_SALT, S0.S0_OUTER_SALT)
SEEDS = S0.SEEDS
GEOS = ("GS41", "GS117")
MARGINS, BAR_MARGIN = C1.MARGINS, C1.BAR_MARGIN
RHOS = (1.25, 1.5)
BAR_A1_FR_UPPER, BAR_A2_FR_UPPER, BAR_DAC_POWER_125 = C1.BAR_A1_FR_UPPER, C1.BAR_A2_FR_UPPER, C1.BAR_DAC_POWER_125     # the committed bars; do not edit
TOL_EP_M, TOL_DM, TOL_PLANT_DM, TOL_BASE_EXACT = R1.TOL_EP_M, R1.TOL_DM, R1.TOL_PLANT_DM, 1e-12
JOBS = ("C-GS41-A", "C-GS41-P", "C-GS117-A", "C-GS117-P", "X-GS41-A", "X-GS117-A")
P_NAMES = ("CAL", "PLANT_N1_DAC_1.25")
REPRO_NAMES = ("CAL", "PLANT_N1_DAC_1.25")
CLASS_KEYS = ("CH", "NSETS") + tuple(f"{p}_{h}" for h in HEADS for p in ("OBSc", "PREDc", "OBSu", "PREDu"))        # the changed / unchanged split carried by every base
REC_KEYS = ("dm", "n1", "a2l", "mp", "me", "k0", "und", "br", "nch", "eobs")
SUBSET_SEED_WRONG = R1.SUBSET_SEED + 1                                                  # the RED arm of PC9

assert N1_BASE_SALT != N1_OUTER_SALT and N1_BASE_SALT not in USED_SALTS and N1_OUTER_SALT not in USED_SALTS


def sha256_file(p):
    return SB.sha256_file(p)


def pin_design() -> str:
    if not os.path.exists(DESIGN):
        raise SystemExit(f"REFUSED: {os.path.abspath(DESIGN)} does not exist")
    side = DESIGN[:-3] + ".sha256"
    if not os.path.exists(side) or open(side, encoding="utf-8").read().split()[0].lower() != sha256_file(DESIGN):
        raise SystemExit("REFUSED: SIM_DESIGN_N1.sha256 is missing or does not equal SIM_DESIGN_N1.md (hash it BEFORE the first run)")
    return sha256_file(DESIGN)


# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# bases with the changed / unchanged split
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def split_vectors(obs: dict, pred: dict, ch: np.ndarray) -> dict:
    """The changed / unchanged split of per-set (or per-log) OBS / PRED vectors by the 0/1 vector `ch` (CH itself is carried): OBSc = OBS * ch, OBSu = OBS * (1 - ch), the same for PRED."""
    cf = np.asarray(ch, np.float64)
    out = {"CH": cf}
    for h in HEADS:
        out[f"OBSc_{h}"], out[f"PREDc_{h}"] = obs[h] * cf, pred[h] * cf
        out[f"OBSu_{h}"], out[f"PREDu_{h}"] = obs[h] * (1.0 - cf), pred[h] * (1.0 - cf)
    return out


def load_pool_n1(bank_dir: str) -> dict:
    """Base C's pool = C1's pool (the carve-B bank THROUGH THE GATE'S LOADERS) + the changed split of every token vector (changed = this seed's pick != P2's pick, the gate's rule)."""
    pool = C1.load_pool(bank_dir)
    p2, cands, _ = SB.load_with_gate(bank_dir)
    P = SB.pickrule()
    pk_p = P.rule_pick(p2["z6"])
    for s in SEEDS["C"]:
        ch = N1.changed_mask(pk_p, P.rule_pick(cands[s]["z6"]))
        if int(ch.sum()) != int(pool["n_changed"][s]):
            raise SystemExit(f"REFUSED: seed {s}: changed picks {int(ch.sum())} != the pool's n_changed {pool['n_changed'][s]}")
        t = pool["tok"][s]
        t.update(split_vectors({h: t[f"OBS_{h}"] for h in HEADS}, {h: t[f"PRED_{h}"] for h in HEADS}, ch))
    return pool


def base_c_n1(pool: dict, counts: np.ndarray, rep: int, offset: int = 0) -> dict:
    """Base C, one replicate (S0.base_c with the N1 salt): every S0 log l draws ONE source carve-B log (probability proportional to its set count), then counts[l] sets WITH REPLACEMENT from it; the same draw
    for all 3 seeds.  Every per-token vector (the changed split included) is summed per log."""
    rng = np.random.default_rng([N1_BASE_SALT, rep + offset])
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


def basex_vectors(z6, lab, log_idx, ep_oof: dict, subset_seed: int, q: float, n_logs: int, seeds=(0, 1)) -> dict:
    """Base X, the R1 simulator's G25 construction recomputed (candidate = the EP-led rows on a Bernoulli(q) subset of sets), returned as {seed: {all-pick vectors of `R1.components`, changed split per log}}."""
    _, P, _ = R1.libs()
    N = len(z6)
    sub = np.random.default_rng(subset_seed).random(N) < q
    pk_p = P.rule_pick(z6)
    out = {}
    for s in seeds:
        zc = z6.copy()
        zc[..., R1.EP] = np.where(sub[:, None], ep_oof[s], z6[..., R1.EP])
        comp = R1.components(z6, zc, lab, log_idx, n_logs)["vec"]
        pk_c = P.rule_pick(zc)
        ch = N1.changed_mask(pk_p, pk_c)
        pl = lambda v: R1.per_log(v, log_idx, n_logs)                                    # noqa: E731
        obs, pred = {}, {}
        for h in HEADS:
            yh, sg = lab[..., R1.HCOL[h]].astype(np.float64), P.sigmoid64(zc[..., R1.HCOL[h]])
            obs[h], pred[h] = 1.0 - P.at(yh, pk_c), 1.0 - P.at(sg, pk_c)
        split = split_vectors(obs, pred, ch)
        split["NSETS"] = np.ones(N)
        out[s] = {"allpick": comp, "split": {k: pl(v) for k, v in split.items()}, "n_changed": int(ch.sum()), "n_sets": int(N), "n_subset": int(sub.sum())}
    return out


def cmd_basex(a) -> int:
    dsha = pin_design()
    L, _, _ = R1.libs()
    if sha256_file(a.arrays) != L.ARRAYS_SHA256:
        raise SystemExit("REFUSED: arrays.npz is not the pinned S0 array file")
    if sha256_file(a.stats) != S0.STATS_SHA256:
        raise SystemExit("REFUSED: stats.npz is not the R2-CONFIRM file")
    z, ref = np.load(a.arrays, allow_pickle=False), np.load(a.stats, allow_pickle=False)
    z6, lab, log_idx = z["z6"], z["lab"], z["log_idx"]
    oof = {s: z[f"ep_OOF_s{s}"] for s in (0, 1)}
    q = R1.GEOMS["G25"]

    def compare(vec):
        return all(np.array_equal(vec[s]["allpick"][k], ref[f"G25__s{s}__{k}"]) for s in (0, 1) for k in vec[s]["allpick"])

    vec = basex_vectors(z6, lab, log_idx, oof, R1.SUBSET_SEED, q, R1.N_LOGS)
    pc9_equal = bool(compare(vec))
    decomp = bool(all(np.allclose(vec[s]["split"][f"OBSc_{h}"] + vec[s]["split"][f"OBSu_{h}"], vec[s]["allpick"][f"OBS_{h}"], rtol=1e-12, atol=1e-12)
                      and np.allclose(vec[s]["split"][f"PREDc_{h}"] + vec[s]["split"][f"PREDu_{h}"], vec[s]["allpick"][f"PRED_{h}"], rtol=1e-12, atol=1e-12) for s in (0, 1) for h in HEADS))
    red = basex_vectors(z6, lab, log_idx, oof, SUBSET_SEED_WRONG, q, R1.N_LOGS)
    red_mismatch = bool(not compare(red))                                              # RED arm: another subset seed must NOT reproduce the banked statistics
    res = {"design_sha256": dsha, "arrays_sha256": sha256_file(a.arrays), "stats_sha256": sha256_file(a.stats),
           "PC9_allpick_vectors_equal_R2CONFIRM_stats_bit_for_bit": pc9_equal, "PC9_split_sums_to_allpick": decomp, "PC9_RED_wrong_subset_seed_does_not_reproduce": red_mismatch,
           "pass": bool(pc9_equal and decomp and red_mismatch), "n_changed_by_seed": {str(s): vec[s]["n_changed"] for s in (0, 1)}, "n_sets": int(len(z6)),
           "changed_share_by_seed": {str(s): vec[s]["n_changed"] / len(z6) for s in (0, 1)}, "subset_share": vec[0]["n_subset"] / len(z6)}
    os.makedirs(a.workdir, exist_ok=True)
    np.savez(os.path.join(a.workdir, "basex_n1.npz"), **{f"s{s}__{k}": v for s in (0, 1) for k, v in vec[s]["split"].items()})
    json.dump(res, open(os.path.join(a.workdir, "basex_n1.json"), "w", encoding="utf-8", newline="\n"), indent=1, sort_keys=True)
    print("BASEX", {k: v for k, v in res.items() if k.startswith("PC9") or k == "pass"})
    return 0 if res["pass"] else 3


def load_base_x_n1(stats_npz: str, basex_npz: str, pos: np.ndarray) -> dict:
    """Base X for the N1 jobs: the R2-CONFIRM per-log statistics (S0.load_base_x) + the changed split, restricted to the geometry's logs.  REFUSES unless PC9 (and its RED arm) passed when the split was built."""
    js = basex_npz[:-4] + ".json"
    if not os.path.exists(js) or json.load(open(js, encoding="utf-8")).get("pass") is not True:
        raise SystemExit("REFUSED: basex_n1.json is missing or PC9 did not pass")
    base = S0.load_base_x(stats_npz, pos)
    z = np.load(basex_npz, allow_pickle=False)
    for s in S0.SEEDS["X"]:
        for k in CLASS_KEYS:
            base[s][k] = z[f"s{s}__{k}"][pos].astype(np.float64)
    return base


# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# conditions and a replicate (the REAL estimator on one draw of logs [A] or on the fixed logs of a population-calibrated draw [P])
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def conditions() -> dict:
    c = {"CAL": dict(kind="cal", a1=True, n1=True, a2l=False, heads=list(HEADS), kw={}, n=2000)}
    for d in C1.DELTAS:
        c[f"PLANT_EP_{d}"] = dict(kind="plant_ep", a1=True, n1=False, a2l=False, heads=[], kw={"delta": d}, n=1000)
    for h in HEADS:
        for rho in RHOS:
            c[f"PLANT_N1_{h}_{rho}"] = dict(kind="plant_n1", a1=False, n1=True, a2l=False, heads=[h], kw={"head": h, "rho": rho}, n=(2000 if (h, rho) == ("DAC", 1.25) else 1000))
    c["PLANT_A2L_DAC_1.25"] = dict(kind="plant_a2l", a1=False, n1=False, a2l=True, heads=["DAC"], kw={"head": "DAC", "rho": 1.25}, n=1000)
    c["IDENTITY"] = dict(kind="identity", a1=True, n1=True, a2l=False, heads=list(HEADS), kw={}, n=50)
    return c


def job_conditions(job: str) -> dict:
    b, g, gen = parse_job(job)
    c = conditions()
    if gen == "P":
        return {"CAL": {**c["CAL"], "a1": False, "n": 1000}, "PLANT_N1_DAC_1.25": {**c["PLANT_N1_DAC_1.25"], "n": 1000}}
    return c


def parse_job(job: str) -> tuple:
    b, g, gen = job.split("-")
    if b not in SEEDS or g not in GEOS or gen not in ("A", "P") or (b == "X" and gen == "P"):
        raise SystemExit(f"REFUSED: unknown job {job!r}")
    return b, g, gen


def outer_idx(rep: int, n: int, gen: str, offset: int = 0) -> np.ndarray:
    """A: the geometry's n logs redrawn with replacement (the R1 / S0 generator, N1 salt).  P: the logs are FIXED (the draw is of tokens only, in `base_c_n1`)."""
    if gen == "P":
        return np.arange(n)
    return np.random.default_rng([N1_OUTER_SALT, rep + offset]).integers(0, n, n)


def calib(obs_sum: float, pred_sum: float):
    """The scalar that puts a class's observed / predicted unsafe mass at exactly 1; None (= leave the prediction unscaled, SIM_DESIGN_N1 section 6) when no unsafe mass was observed."""
    return (obs_sum / pred_sum) if (obs_sum > 0.0 and pred_sum > 0.0) else None


def replicate(cond: dict, base: dict, seeds: tuple, gen: str, rep: int, draws: np.ndarray, offset: int = 0, idx: np.ndarray | None = None, kfix: dict | None = None) -> dict:
    """-> {dm [S,3], n1 [S,3,3] (point, lo, hi per head), a2l [S,3,3], mp [S], me [S], k0 [S,3] (no observed unsafe mass at the changed picks: prediction left unscaled), und [S,3] (N1 undefined: no predicted mass),
    br [S,3] (observed / predicted at the replicate base, before the log redraw), nch [S] (changed picks in the base), eobs [S,3] (observed unsafe mass at the changed picks in the base)}."""
    _, _, F = R1.libs()
    n = len(base[seeds[0]]["y_p"])
    idx = outer_idx(rep, n, gen, offset) if idx is None else idx
    S = len(seeds)
    nan = np.nan
    dm, n1c, a2l = np.full((S, 3), nan), np.full((S, 3, 3), nan), np.full((S, 3, 3), nan)
    mp, me, br, eobs, nch = np.full(S, nan), np.full(S, nan), np.full((S, 3), nan), np.full((S, 3), nan), np.zeros(S)
    k0, und = np.zeros((S, 3), bool), np.zeros((S, 3), bool)
    kind = cond["kind"]
    for si, s in enumerate(seeds):
        v = base[s]
        if cond["a1"]:
            if kind == "identity":
                d = {"Yp": v["y_p"], "Sp": v["s_p"], "Ye": v["y_p"], "Se": v["s_p"]}
            else:
                d = R1.build_condition(v, kind, **cond["kw"])
            Yp, Sp, Ye, Se = d["Yp"][idx], d["Sp"][idx], d["Ye"][idx], d["Se"][idx]
            mp[si], me[si] = Yp.sum() / Sp.sum(), Ye.sum() / Se.sum()
            ci = F.diff_ci(Yp, Sp, Ye, Se, draws)
            dm[si] = (ci["point"], ci["lo"], ci["hi"])
        nch[si] = 0.0 if kind == "identity" else float(v["CH"].sum())
        for j, h in enumerate(HEADS):
            if h not in cond["heads"]:
                continue
            rho_h = cond["kw"].get("rho", 1.0) if (kind in ("plant_n1", "plant_a2l") and cond["kw"].get("head") == h) else 1.0
            if cond["n1"]:
                obsc, predc = (np.zeros_like(v["y_p"]), np.zeros_like(v["y_p"])) if kind == "identity" else (v[f"OBSc_{h}"], v[f"PREDc_{h}"])
                so, sp = float(obsc.sum()), float(predc.sum())
                eobs[si, j] = so
                if not sp > 0.0:                                               # no changed pick in this replicate base: N1 is undefined
                    und[si, j] = True
                    continue
                k = calib(so, sp) if kfix is None else kfix[(si, j)]
                k0[si, j] = k is None                                          # SIM_DESIGN_N1 section 6: no observed unsafe mass at the changed picks -> the prediction is left UNSCALED (real sigma)
                pred = predc * (1.0 if k is None else k) / rho_h
                if not k0[si, j]:
                    br[si, j] = so / float(pred.sum())
                try:
                    ci = N1.n1_ci_logvec(obsc[idx], pred[idx], draws)
                    n1c[si, j] = (ci["point"], ci["lo"], ci["hi"])
                except N1.N1Undefined:                                         # the redrawn logs hold no changed pick
                    und[si, j] = True
            if cond["a2l"]:
                oc, pc, ou, pu = v[f"OBSc_{h}"], v[f"PREDc_{h}"], v[f"OBSu_{h}"], v[f"PREDu_{h}"]
                kc, ku = calib(float(oc.sum()), float(pc.sum())), calib(float(ou.sum()), float(pu.sum()))
                pred_all = (1.0 if ku is None else ku) * pu + (1.0 if kc is None else kc) * pc / rho_h
                ci = F.ratio_ci((oc + ou)[idx], pred_all[idx], draws)
                a2l[si, j] = (ci["point"], ci["lo"], ci["hi"])
    return {"dm": dm, "n1": n1c, "a2l": a2l, "mp": mp, "me": me, "k0": k0, "und": und, "br": br, "nch": nch, "eobs": eobs}


def pool_k(pool: dict) -> dict:
    """Generator P: the POOL-level calibration scalars k[(seed index, head index)] = sum OBSc / sum PREDc over every carve-B set (None if no observed unsafe mass at the pool's changed picks)."""
    out = {}
    for si, s in enumerate(SEEDS["C"]):
        for j, h in enumerate(HEADS):
            out[(si, j)] = calib(float(pool["tok"][s][f"OBSc_{h}"].sum()), float(pool["tok"][s][f"PREDc_{h}"].sum()))
    return out


def class_summary(v_obs_c, v_pred_c, v_obs_u, v_pred_u, v_ch, n_sets) -> dict:
    return {"changed_picks": float(v_ch), "changed_share": float(v_ch) / n_sets, "observed_unsafe_mass_at_changed": float(v_obs_c), "predicted_unsafe_mass_at_changed": float(v_pred_c),
            "N1_pooled_point_ratio": (float(v_obs_c) / float(v_pred_c)) if v_pred_c > 0 else None, "observed_unsafe_mass_at_unchanged": float(v_obs_u), "predicted_unsafe_mass_at_unchanged": float(v_pred_u),
            "unchanged_pooled_point_ratio": (float(v_obs_u) / float(v_pred_u)) if v_pred_u > 0 else None}


def pool_events(pool: dict) -> dict:
    """R-EVENTS / R-DOSE / R-POOLRATIO for Base C (descriptive, point ratios only): per seed, the changed-pick count and, per head, the distinct changed picks with unsafe mass, the masses and the pooled ratios."""
    out = {}
    for s in SEEDS["C"]:
        t = pool["tok"][s]
        out[str(s)] = {"changed": int(t["CH"].sum()), "changed_share": float(t["CH"].mean()), "n_sets": int(pool["n_sets"]),
                       "heads": {h: {**class_summary(t[f"OBSc_{h}"].sum(), t[f"PREDc_{h}"].sum(), t[f"OBSu_{h}"].sum(), t[f"PREDu_{h}"].sum(), t["CH"].sum(), pool["n_sets"]),
                                     "distinct_changed_picks_with_unsafe_mass": int((t[f"OBSc_{h}"] > 0).sum()), "distinct_unchanged_picks_with_unsafe_mass": int((t[f"OBSu_{h}"] > 0).sum())} for h in HEADS}}
    return out


def base_x_events(bx: dict) -> dict:
    """R-EVENTS / R-DOSE / R-POOLRATIO for Base X on the geometry's logs (descriptive, point ratios only)."""
    ns = {s: float(bx[s]["NSETS"].sum()) for s in bx}
    return {str(s): {"changed": float(bx[s]["CH"].sum()), "n_sets": ns[s],
                     "heads": {h: class_summary(bx[s][f"OBSc_{h}"].sum(), bx[s][f"PREDc_{h}"].sum(), bx[s][f"OBSu_{h}"].sum(), bx[s][f"PREDu_{h}"].sum(), bx[s]["CH"].sum(), ns[s]) for h in HEADS}} for s in bx}


def make_sampler(job: str, a) -> tuple:
    """-> (geometry, seeds, base_fn(rep, offset) -> base dict, extras, kfix)."""
    b, g, gen = parse_job(job)
    geo = S0.load_geometry(a.geometry, g)
    if b == "C":
        pool = load_pool_n1(a.pool)
        if not S0.check_clone(pool, base_c_n1(pool, geo["counts"], 0, 0), geo["counts"]):
            raise SystemExit("REFUSED: PC7 clone integrity failed on replicate 0")
        extras = {"pool_events": pool_events(pool), "pool_sets": pool["n_sets"], "PC7_clone_integrity_rep0": True,
                  "bank_manifests": {n: sha256_file(os.path.join(a.pool, n, "MANIFEST.sha256.json")) for n in ("p2", "s0", "s1", "s2")}}
        kfix = pool_k(pool) if gen == "P" else None
        if kfix is not None:
            extras["P_pool_k"] = {f"s{SEEDS['C'][si]}|{HEADS[j]}": k for (si, j), k in kfix.items()}
        return geo, SEEDS["C"], (lambda rep, off=0: base_c_n1(pool, geo["counts"], rep, off)["base"]), extras, kfix
    bx = load_base_x_n1(a.stats, a.basex, geo["pos"])
    return geo, SEEDS["X"], (lambda rep, off=0: bx), {"stats_sha256": S0.STATS_SHA256, "basex_events_geometry_logs": base_x_events(bx)}, None


def run_job(job: str, a, n_reps_override: int = 0, log=print) -> dict:
    geo, seeds, base_fn, extras, kfix = make_sampler(job, a)
    _, _, gen = parse_job(job)
    _, _, F = R1.libs()
    draws = F.boot_draws(geo["n"], BOOT_B, BOOT_SEED)
    out = {}
    for name, cond in job_conditions(job).items():
        t0 = time.time()
        nr = n_reps_override or cond["n"]
        recs = [replicate(cond, base_fn(r, a.rep_offset), seeds, gen, r, draws, a.rep_offset, kfix=kfix) for r in range(nr)]
        out[name] = {k: np.stack([r[k] for r in recs]) for k in REC_KEYS}
        log(f"  {job} {name}: {nr} replicates ({time.time() - t0:.0f} s)")
    return {"recs": out, "extras": extras, "geo": geo, "seeds": seeds}


# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# decisions (module path and an inline restatement, PC4) and summaries
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def n1_verdict_module(n1_i, und_i, heads, margin) -> str:
    """The frozen-module path: `epadm_n1.n1_verdict` (-> `a2_head_ok_r1`) on one replicate's [S,3,3] intervals; a seed with any undefined head in `heads` is an undefined seed."""
    j_of = {h: j for j, h in enumerate(HEADS)}
    per_seed = []
    for si in range(n1_i.shape[0]):
        per_seed.append(None if any(und_i[si, j_of[h]] for h in heads) else {h: S0.ci_of(n1_i[si, j_of[h]]) for h in heads})
    return N1.n1_verdict(per_seed, margin, tuple(heads))


def n1_verdict_inline(n1_i, und_i, heads, margin) -> str:
    j_of = {h: j for j, h in enumerate(HEADS)}
    S = n1_i.shape[0]
    if S == 0:
        return "UNDEFINED-N1"
    seed_und = [any(und_i[si, j_of[h]] for h in heads) for si in range(S)]
    fail = any((not seed_und[si]) and any(not (n1_i[si, j_of[h], 1] <= margin) for h in heads) for si in range(S))
    if fail:
        return "FAIL-N1"
    return "UNDEFINED-N1" if any(seed_und) else "ADMIT-N1"


def head_refuse(n1_i, und_i, h, margin) -> bool:
    """Power's per-head event: ANY defined seed has the head's lower CI above the margin (an undefined seed never counts as a refusal here)."""
    j = HEADS.index(h)
    return any((not und_i[si, j]) and not A.a2_head_ok_r1(S0.ci_of(n1_i[si, j]), margin) for si in range(n1_i.shape[0]))


def summarize(name: str, cond: dict, rec: dict, seeds: tuple) -> dict:
    n, S = len(rec["dm"]), len(seeds)
    R = R1.rate
    out = {"name": name, "n": int(n)}
    dis = 0
    if cond["a1"]:
        base = S0.summarize(name, {"a1": True, "a2": False, "heads": []}, {"dm": rec["dm"], "a2": np.full((n, S, 3, 3), np.nan), "mp": rec["mp"], "me": rec["me"], "k0": rec["k0"], "br": np.full((n, S, 3), np.nan)}, seeds)
        out["means"], out["A1"] = base["means"], base["A1"]
        dis += base["pc4_module_vs_inline_disagreements"]
    if cond["n1"]:
        heads, jh = list(cond["heads"]), [HEADS.index(h) for h in cond["heads"]]
        blk = {"heads": heads, "k0_replicate_seed_heads": {h: int(rec["k0"][:, :, HEADS.index(h)].sum()) for h in HEADS},
               "und_replicate_seed_heads": {h: int(rec["und"][:, :, HEADS.index(h)].sum()) for h in HEADS},
               "base_changed_picks_mean_by_seed": rec["nch"].mean(axis=0).tolist(), "base_observed_unsafe_mass_at_changed_mean": {h: rec["eobs"][:, :, HEADS.index(h)].mean(axis=0).tolist() for h in heads},
               "ratio_point_mean": {h: np.nanmean(rec["n1"][:, :, HEADS.index(h), 0], axis=0).tolist() if np.isfinite(rec["n1"][:, :, HEADS.index(h), 0]).any() else None for h in heads},
               "n_nan_lower_ci": int(np.isnan(rec["n1"][:, :, jh, 1]).sum() - rec["und"][:, :, jh].sum()), "by_margin": {}}
        for t in MARGINS:
            vm = [n1_verdict_module(rec["n1"][i], rec["und"][i], heads, t) for i in range(n)]
            vi = [n1_verdict_inline(rec["n1"][i], rec["und"][i], heads, t) for i in range(n)]
            dis += int(sum(a != b for a, b in zip(vm, vi)))
            nonadmit = np.array([v != "ADMIT-N1" for v in vm])
            fail = np.array([v == "FAIL-N1" for v in vm])
            fc = np.array([bool(nonadmit[i] or any(rec["k0"][i, si, j] for si in range(S) for j in jh)) for i in range(n)])
            ph = {h: np.array([head_refuse(rec["n1"][i], rec["und"][i], h, t) for i in range(n)]) for h in heads}
            phc = {h: np.array([bool(ph[h][i] or any(rec["k0"][i, si, HEADS.index(h)] for si in range(S))) for i in range(n)]) for h in heads}
            blk["by_margin"][f"{t:.2f}"] = {"any_head_any_seed": R(nonadmit.sum(), n), "any_head_any_seed_fail_only": R(fail.sum(), n), "undefined_verdicts": int(sum(v == "UNDEFINED-N1" for v in vm)),
                                            "any_head_any_seed_failclosed": R(fc.sum(), n), "per_head_any_seed": {h: R(ph[h].sum(), n) for h in heads}, "per_head_any_seed_failclosed": {h: R(phc[h].sum(), n) for h in heads}}
        out["N1"] = blk
    if cond["a2l"]:
        heads = list(cond["heads"])
        blk = {"heads": heads, "ratio_point_mean": {h: np.nanmean(rec["a2l"][:, :, HEADS.index(h), 0], axis=0).tolist() for h in heads}, "by_margin": {}}
        for t in MARGINS:
            mod = np.array([S0.a2_refuse_module(rec["a2l"][i], heads, t) for i in range(n)])
            inl = np.array([S0.a2_refuse_inline(rec["a2l"][i], heads, t) for i in range(n)])
            dis += int((mod != inl).sum())
            blk["by_margin"][f"{t:.2f}"] = {"any_head_any_seed": R(mod.sum(), n), "per_head_any_seed": {h: R(np.array([S0.a2_refuse_inline(rec["a2l"][i], [h], t) for i in range(n)]).sum(), n) for h in heads}}
        out["A2L"] = blk
    out["pc4_module_vs_inline_disagreements"] = int(dis)
    return out


decide, bar_status = S0.decide, S0.bar_status


def bar_inputs(summ: dict, conv: str = "primary") -> tuple:
    """(A1-r1 false refusal, A2-N1 false refusal, N1 DAC power at x1.25) at margin 1.05.  Primary: a NON-admission (FAIL or UNDEFINED) on the calibrated CAL replicates counts as a false refusal; the second convention also
    counts every replicate-seed-head with no observed unsafe mass at the changed picks (the C1 / S0BANK fail-closed convention)."""
    cal, dac = summ["CAL"], summ["PLANT_N1_DAC_1.25"]
    a1 = cal["A1"]["r1_any_seed"] if "A1" in cal else None
    m = cal["N1"]["by_margin"][f"{BAR_MARGIN:.2f}"]
    pm = dac["N1"]["by_margin"][f"{BAR_MARGIN:.2f}"]
    if conv == "primary":
        return a1, m["any_head_any_seed"], pm["per_head_any_seed"]["DAC"]
    return a1, m["any_head_any_seed_failclosed"], pm["per_head_any_seed_failclosed"]["DAC"]


# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# controls on one job's records
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def job_controls(summ: dict, recs: dict) -> dict:
    pc = {}
    cal = summ["CAL"]
    pc["PC1a_calibrated_by_construction"] = {"pass": bool(all(abs(m - 1.0) <= TOL_EP_M for m in cal["means"]["M_p"] + cal["means"]["M_e"]) and all(abs(m) <= TOL_DM for m in cal["A1"]["D_M_point_mean"])),
                                             "M_p": cal["means"]["M_p"], "M_e": cal["means"]["M_e"], "D_M_point_mean": cal["A1"]["D_M_point_mean"], "tolerances": {"M": TOL_EP_M, "D_M": TOL_DM}}
    br, jd = recs["CAL"]["br"], HEADS.index("DAC")
    bp = recs["PLANT_N1_DAC_1.25"]["br"][:, :, jd]
    ok_cal = bool(np.all(np.abs(br[np.isfinite(br)] - 1.0) <= TOL_BASE_EXACT))
    ok_plant = bool(np.all(np.abs(bp[np.isfinite(bp)] - 1.25) <= TOL_BASE_EXACT)) and bool(np.isfinite(bp).any())
    pc["PC1b_N1_base_exactness_at_the_changed_picks"] = {"pass": bool(ok_cal and ok_plant), "CAL_max_abs_dev_from_1": float(np.nanmax(np.abs(br - 1.0))) if np.isfinite(br).any() else None,
                                                         "PLANT_DAC_1.25_max_abs_dev_from_1.25": float(np.nanmax(np.abs(bp - 1.25))) if np.isfinite(bp).any() else None, "tolerance": TOL_BASE_EXACT}
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
        idn = recs["IDENTITY"]
        all_und = bool(idn["und"].all())
        verd = [n1_verdict_module(idn["n1"][i], idn["und"][i], list(HEADS), BAR_MARGIN) for i in range(len(idn["dm"]))]
        pc["PC6_identity_candidate"] = {"all_D_M_exactly_0": bool((idn["dm"] == 0.0).all()), "N1_undefined_on_every_replicate_seed_head": all_und, "every_N1_verdict_is_UNDEFINED_never_ADMIT_never_FAIL": bool(all(v == "UNDEFINED-N1" for v in verd)),
                                       "pass": bool((idn["dm"] == 0.0).all() and summ["IDENTITY"]["A1"]["r1_any_seed"]["k"] == 0 and all_und and all(v == "UNDEFINED-N1" for v in verd))}
    return pc


# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def cmd_cost(a) -> int:
    pin_design()
    out = {}
    for job in a.jobs.split(","):
        geo, seeds, base_fn, _, kfix = make_sampler(job, a)
        _, _, gen = parse_job(job)
        _, _, F = R1.libs()
        draws = F.boot_draws(geo["n"], BOOT_B, BOOT_SEED)
        conds, sec = job_conditions(job), {}
        for name in ("CAL", "PLANT_EP_0.135", "PLANT_N1_DAC_1.25", "PLANT_A2L_DAC_1.25"):
            if name not in conds:
                continue
            t0 = time.time()
            for r in range(3):
                replicate(conds[name], base_fn(r), seeds, gen, r, draws, kfix=kfix)
            sec[name] = (time.time() - t0) / 3.0
        est = sum(c["n"] * sec.get(nm, sec["PLANT_N1_DAC_1.25"]) for nm, c in conds.items())
        out[job] = {"seconds_per_replicate": sec, "estimated_cpu_seconds_upper": est}
    os.makedirs(a.workdir, exist_ok=True)
    json.dump(out, open(os.path.join(a.workdir, "cost.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))
    return 0


def cmd_run(a) -> int:
    dsha = pin_design()
    t0 = time.time()
    print(f"run {a.job}: B = {BOOT_B}, seed {BOOT_SEED}, rep_offset {a.rep_offset}, salts {N1_BASE_SALT}/{N1_OUTER_SALT}", flush=True)
    r = run_job(a.job, a, a.n_reps, log=lambda m: print(m, flush=True))
    os.makedirs(a.workdir, exist_ok=True)
    np.savez_compressed(os.path.join(a.workdir, f"replicates_{a.job}.npz"), **{f"{n}__{k}": v for n, rec in r["recs"].items() for k, v in rec.items()})
    code = {"epadm_sim_n1.py": sha256_file(os.path.abspath(__file__)), "epadm_n1.py": sha256_file(os.path.join(HERE, "epadm_n1.py")), "epadm_sim_s0.py": sha256_file(os.path.join(HERE, "epadm_sim_s0.py")),
            "epadm_sim_c1.py": sha256_file(os.path.join(S0.C1_DIR, "epadm_sim_c1.py")), "ep_admission.py": SB.GATE_SHA256, "ep_admission_sim.py": sha256_file(os.path.join(C1.R1_DIR, "ep_admission_sim.py"))}
    json.dump({"design_sha256": dsha, "job": a.job, "wall_seconds": time.time() - t0, "rep_offset": a.rep_offset, "n_logs": r["geo"]["n"], "n_tokens": int(r["geo"]["counts"].sum()), "extras": r["extras"], "code_sha256": code},
              open(os.path.join(a.workdir, f"run_{a.job}.json"), "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(f"JOB {a.job} DONE_MARKER ({time.time() - t0:.0f} s)")
    return 0


def cmd_repro(a) -> int:
    pin_design()
    geo, seeds, base_fn, _, kfix = make_sampler(a.job, a)
    _, _, gen = parse_job(a.job)
    _, _, F = R1.libs()
    draws = F.boot_draws(geo["n"], BOOT_B, BOOT_SEED)
    z = np.load(os.path.join(a.workdir, f"replicates_{a.job}.npz"))
    conds, ok = job_conditions(a.job), True
    for name in REPRO_NAMES:
        for r in range(20):
            rec = replicate(conds[name], base_fn(r, a.rep_offset), seeds, gen, r, draws, a.rep_offset, kfix=kfix)
            for k in REC_KEYS:
                ok &= bool(np.array_equal(rec[k], z[f"{name}__{k}"][r], equal_nan=True))
    out = {"job": a.job, "pass": bool(ok), "bit_identical_first_20_replicates_of_2_conditions": bool(ok), "pid": os.getpid()}
    json.dump(out, open(os.path.join(a.workdir, f"repro_{a.job}.json"), "w"), indent=1)
    print("REPRO", out)
    return 0 if ok else 3


def pc3_n1(bank_dir_all: str, bank_dir_none: str, bank_dir_part: str) -> dict:
    """PC3 (N1): on SYNTHETIC adapted banks (planted data) -- (a) the per-log shortcut `n1_ci_logvec` equals `n1_reliability` through the gate's `Stats` object to 0.0 for 3 seeds x 3 heads; (b) on a bank where EVERY pick is
    changed, N1 equals the frozen `analyze_candidate` A2 to 0.0 (so N1 is the same statistic, restricted); (c) on the partial-change bank N1 DIFFERS from A2 (the restriction is real); (d) on a bank with ZERO changed picks
    `n1_reliability` raises N1Undefined and the verdict is UNDEFINED-N1."""
    L, P, F = R1.libs()
    out, ok = {}, True
    for tag, bd in (("partial", bank_dir_part), ("all_changed", bank_dir_all), ("none_changed", bank_dir_none)):
        p2, cands, _ = SB.load_with_gate(bd)
        nl = p2["n_logs"]
        S = A.make_stats(p2["log_idx"], nl)
        draws = F.boot_draws(nl, BOOT_B, BOOT_SEED)
        pl = lambda x: np.bincount(p2["log_idx"], weights=x, minlength=nl).astype(np.float64)       # noqa: E731
        pk_p = P.rule_pick(p2["z6"])
        row = {}
        for s in SEEDS["C"]:
            pk_c = P.rule_pick(cands[s]["z6"])
            n_ch = int((pk_c != pk_p).sum())
            r = {"n_changed": n_ch, "n_sets": int(len(pk_p))}
            if tag == "none_changed":
                try:
                    N1.n1_reliability(S, p2["lab"], cands[s]["z6"], pk_p, pk_c)
                    r["refuses"] = False
                except N1.N1Undefined:
                    r["refuses"] = True
                r["ok"] = bool(r["refuses"] and n_ch == 0)
            else:
                mod = N1.n1_reliability(S, p2["lab"], cands[s]["z6"], pk_p, pk_c)
                v = C1.token_vectors(p2["z6"], cands[s]["z6"], p2["lab"])
                cf = (pk_c != pk_p).astype(np.float64)
                eq_short = {h: R1.eq_ci(mod["ci"][h], N1.n1_ci_logvec(pl(v[f"OBS_{h}"] * cf), pl(v[f"PRED_{h}"] * cf), draws)) for h in HEADS}
                an = A.analyze_candidate(p2["z6"], cands[s]["z6"], p2["lab"], S)
                eq_a2 = {h: R1.eq_ci(mod["ci"][h], an["A2_reliability"][h]) for h in HEADS}
                r.update({"shortcut_equals_Stats_path": eq_short, "equals_A2_all_picks": eq_a2})
                r["ok"] = bool(all(eq_short.values()) and ((all(eq_a2.values()) and n_ch == len(pk_p)) if tag == "all_changed" else (not all(eq_a2.values()))))
            row[f"s{s}"] = r
            ok &= r["ok"]
        out[tag] = row
    cis_none = [None, None, None]
    out["verdict_of_undefined_seeds_is_UNDEFINED_not_ADMIT"] = bool(N1.n1_verdict(cis_none) == "UNDEFINED-N1")
    ok &= out["verdict_of_undefined_seeds_is_UNDEFINED_not_ADMIT"]
    return {"pass": bool(ok), "banks": out, "on": "SYNTHETIC adapted banks (planted data; no real candidate statistic)"}


def cmd_pc3(a) -> int:
    import fixture_synth as FX
    dirs = {}
    for tag, share, sd in (("all", 1.0, 5), ("none", 0.0, 6), ("part", 0.3, 3)):
        fx = FX.make_fixture(os.path.join(a.workdir, f"pc3_fixture_{tag}"), n_tokens=120, n_logs=24, seed=sd, moved_share=share)
        out = os.path.join(a.workdir, f"pc3_bank_{tag}")
        SB.build(fx["bundle"], fx["search"], fx["labels"], out, {"token_sha256": fx["token_sha256"]})
        dirs[tag] = out
    res = pc3_n1(dirs["all"], dirs["none"], dirs["part"])
    json.dump(res, open(os.path.join(a.workdir, "pc3.json"), "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("PC3", res["pass"])
    return 0 if res["pass"] else 3


def load_job(workdir: str, job: str) -> tuple:
    z = np.load(os.path.join(workdir, f"replicates_{job}.npz"))
    recs = {}
    for k in z.files:
        n, key = k.split("__")
        recs.setdefault(n, {})[key] = z[k]
    return recs, json.load(open(os.path.join(workdir, f"run_{job}.json")))


def cmd_report(a) -> int:
    dsha = pin_design()
    res = {"schema": "ep-admission-n1-sim-v1", "created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"), "design_sha256": dsha, "jobs": {}, "geometries": {}}
    ctrl_ok = True
    for job in a.jobs.split(","):
        b, g, gen = parse_job(job)
        recs, run = load_job(a.workdir, job)
        conds = job_conditions(job)
        summ = {n: summarize(n, conds[n], recs[n], S0.SEEDS[b]) for n in conds}
        entry = {"summaries": summ, "run": run}
        if gen == "A":
            pc = job_controls(summ, recs)
            rp = os.path.join(a.workdir, f"repro_{job}.json")
            pc["PC5_determinism"] = json.load(open(rp)) if os.path.exists(rp) else {"pass": None}
            entry["controls"] = pc
            entry["controls_ok"] = bool(all(pc[k].get("pass") is True for k in pc))
            ctrl_ok &= entry["controls_ok"]
        else:
            rp = os.path.join(a.workdir, f"repro_{job}.json")
            pr = json.load(open(rp)) if os.path.exists(rp) else {"pass": None}
            entry["controls"] = {"PC5_determinism": pr, "PC4_decision_code_is_the_modules": {"pass": bool(sum(s["pc4_module_vs_inline_disagreements"] for s in summ.values()) == 0)}}
            entry["controls_ok"] = bool(all(v.get("pass") is True for v in entry["controls"].values()))
            ctrl_ok &= entry["controls_ok"]
        if gen == "A":
            a1, a2, pw = bar_inputs(summ, "primary")
            a1f, a2f, pwf = bar_inputs(summ, "failclosed")
            entry["bars_primary"] = {"A1_r1_false_refusal": a1, "A2_N1_1.05_false_refusal": a2, "N1_DAC_power_x1.25_margin1.05": pw, "decision": decide(a1["wilson95"][1], a2["wilson95"][1], pw["rate"])}
            entry["bars_failclosed"] = {"A1_r1_false_refusal": a1f, "A2_N1_1.05_false_refusal": a2f, "N1_DAC_power_x1.25_margin1.05": pwf, "decision": decide(a1f["wilson95"][1], a2f["wilson95"][1], pwf["rate"])}
        else:                                                                    # generator P carries no A1: only the two N1 bars are read
            _, a2, pw = bar_inputs(summ, "primary")
            entry["bars_primary"] = {"A2_N1_1.05_false_refusal": a2, "N1_DAC_power_x1.25_margin1.05": pw,
                                     "bars": {"A2_N1_fr_upper_le_5pct": bool(a2["wilson95"][1] <= BAR_A2_FR_UPPER), "N1_DAC_power_ge_70pct": bool(pw["rate"] >= BAR_DAC_POWER_125)}}
        res["jobs"][job] = entry
    for g in GEOS:
        c, x, p = res["jobs"].get(f"C-{g}-A"), res["jobs"].get(f"X-{g}-A"), res["jobs"].get(f"C-{g}-P")
        if not (c and x):
            continue
        keys = list(c["bars_primary"]["decision"]["bars"])
        st = {k: bar_status(c["bars_primary"]["decision"]["bars"][k], x["bars_primary"]["decision"]["bars"][k]) for k in keys}
        fl = {k: bar_status(c["bars_failclosed"]["decision"]["bars"][k], x["bars_failclosed"]["decision"]["bars"][k]) for k in keys}
        cells = {"C": c["bars_primary"]["decision"], "X": x["bars_primary"]["decision"]}
        pp = {}
        if p:
            pa2, ppw = p["bars_primary"]["A2_N1_1.05_false_refusal"], p["bars_primary"]["N1_DAC_power_x1.25_margin1.05"]
            ca2, cpw = c["bars_primary"]["A2_N1_1.05_false_refusal"], c["bars_primary"]["N1_DAC_power_x1.25_margin1.05"]
            pp = {"R-GEN_P_vs_A_pp": {"A2_N1_fr": abs(pa2["rate"] - ca2["rate"]) * 100, "DAC_power": abs(ppw["rate"] - cpw["rate"]) * 100},
                  "P_bar_status": {"A2_N1_fr_met": p["bars_primary"]["bars"]["A2_N1_fr_upper_le_5pct"], "DAC_power_met": p["bars_primary"]["bars"]["N1_DAC_power_ge_70pct"]},
                  "R-GEN_status_differs_from_A": bool(p["bars_primary"]["bars"]["A2_N1_fr_upper_le_5pct"] != c["bars_primary"]["decision"]["bars"]["A2_1.05_false_refusal_wilson_upper_le_5pct"]
                                                      or p["bars_primary"]["bars"]["N1_DAC_power_ge_70pct"] != c["bars_primary"]["decision"]["bars"]["DAC_power_x1.25_margin1.05_point_ge_70pct"])}
        res["geometries"][g] = {"status_primary_convention": st, "status_failclosed_convention": fl, "all_met": bool(all(v == "MET" for v in st.values())), "cells_all_met": {k: bool(v["all_met"]) for k, v in cells.items()},
                                "cells_missed": {k: v["missed"] for k, v in cells.items()}, "R-NAN": bool(st != fl),
                                "R-SEED_C_vs_X_pp": {"A1_fr": abs(c["bars_primary"]["A1_r1_false_refusal"]["rate"] - x["bars_primary"]["A1_r1_false_refusal"]["rate"]) * 100,
                                                     "A2_N1_fr": abs(c["bars_primary"]["A2_N1_1.05_false_refusal"]["rate"] - x["bars_primary"]["A2_N1_1.05_false_refusal"]["rate"]) * 100,
                                                     "DAC_power": abs(c["bars_primary"]["N1_DAC_power_x1.25_margin1.05"]["rate"] - x["bars_primary"]["N1_DAC_power_x1.25_margin1.05"]["rate"]) * 100}, **pp}
    rd = lambda n: json.load(open(os.path.join(a.workdir, n))) if os.path.exists(os.path.join(a.workdir, n)) else {"pass": None}    # noqa: E731
    res["controls_global"] = {"PC3_N1_shortcut_and_restriction": rd("pc3.json"), "PC8_s0_code_reproduces_banked_R2CONFIRM": rd("pc8.json"), "PC9_BaseX_split_equals_banked_stats": rd("basex_n1.json")}
    res["controls_ok"] = bool(ctrl_ok and all(v.get("pass") is True for v in res["controls_global"].values()))
    json.dump(res, open(os.path.join(a.workdir, "sim_results.json"), "w", encoding="utf-8", newline="\n"), indent=1, sort_keys=True, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(f"REPORT controls_ok={res['controls_ok']} " + " ".join(f"{g}:all_met={v['all_met']}" for g, v in res["geometries"].items()))
    return 0 if res["controls_ok"] else 3


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for nm, fn in (("cost", cmd_cost), ("run", cmd_run), ("repro", cmd_repro)):
        p = sub.add_parser(nm)
        p.add_argument("--pool", default=None); p.add_argument("--stats", default=None); p.add_argument("--basex", default=None); p.add_argument("--geometry", required=True); p.add_argument("--workdir", required=True)
        p.add_argument("--rep-offset", type=int, default=0)
        if nm == "cost":
            p.add_argument("--jobs", default=",".join(JOBS))
        else:
            p.add_argument("--job", required=True)
        if nm == "run":
            p.add_argument("--n-reps", type=int, default=0)
        p.set_defaults(fn=fn)
    q = sub.add_parser("basex")
    q.add_argument("--arrays", required=True); q.add_argument("--stats", required=True); q.add_argument("--workdir", required=True)
    q.set_defaults(fn=cmd_basex)
    q = sub.add_parser("pc3")
    q.add_argument("--workdir", required=True)
    q.set_defaults(fn=cmd_pc3)
    r = sub.add_parser("report")
    r.add_argument("--workdir", required=True); r.add_argument("--jobs", default=",".join(JOBS))
    r.set_defaults(fn=cmd_report)
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
