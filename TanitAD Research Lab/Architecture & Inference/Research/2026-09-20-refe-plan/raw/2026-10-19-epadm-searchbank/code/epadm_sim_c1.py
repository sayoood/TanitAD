"""EP-ADMISSION R1 re-simulation at C1's geometry (SEARCH-P2-CONFIRM DQ6; design: ../SIM_DESIGN.md, hashed BEFORE the first run).  CPU only.

The R1 simulator (`ep_admission_sim.py`, frozen) is hard-wired to 117 logs and 2 seeds, so this module is a thin SIBLING that imports its calibration / planting / Wilson functions UNCHANGED and re-runs its
replicate loop at 24 logs x 3 search seeds with the registered bootstrap (`fastneg_lib.diff_ci` / `ratio_ci`, `boot_draws(24, 10000, 20260927)`) and the frozen gate's decision functions.  The estimator, the
decision functions and the bars are not touched; only the geometry, the base population and the replicate seeds are new.

    python epadm_sim_c1.py cost    --bank D --workdir W                 # 3 replicates per timing class (reads no rate)
    python epadm_sim_c1.py run     --bank D --workdir W --part I --nparts 2 [--rep-offset 0]
    python epadm_sim_c1.py report  --workdir W                          # merge parts, controls PC1-PC6, the bars -> sim_results.json
    python epadm_sim_c1.py repro   --bank D --workdir W                 # PC5: 20 replicates of 2 conditions in a SEPARATE process equal the stored ones
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
import epadm_searchbank as SB                                                               # noqa: E402

A = SB.gate()                                                                                # the frozen, sha-checked ep_admission (also what ep_admission_sim imports)
R1_DIR = os.environ.get("EP_ADMISSION_R1_DIR") or os.path.join(HERE, "..", "..", "2026-10-19-ep-admission-r1", "code")
sys.path.insert(0, R1_DIR)
import ep_admission_sim as R1                                                                # noqa: E402

DESIGN = os.path.join(HERE, "..", "SIM_DESIGN.md")
N_LOGS, SEEDS, HEADS = 24, (0, 1, 2), R1.HEADS
BOOT_B, BOOT_SEED = 10000, 20260927
BASE_SALT, OUTER_SALT = 20261201, 20261202                                                    # NEW replicate seeds (R1 used 20261027 / 20261028 / 20261029; EP-RELIAB's null 20261019)
R1_SALTS = (R1.OUTER_SALT, R1.WILD_SALT, R1.SUBSET_SEED)
N_BAR, N_OTHER, N_ID = 2000, 1000, 50
MARGINS = (1.00, 1.05, 1.10)
BAR_MARGIN = 1.05
RHOS, DELTAS = (1.25, 1.5), (0.135, 0.05)
# the committed bars (MM_RULING_EPADM_R1.md section 2, ratified unchanged by MM_RULING_SPC.md DQ6); do not edit
BAR_A1_FR_UPPER, BAR_A2_FR_UPPER, BAR_DAC_POWER_125 = 0.05, 0.05, 0.70
TOL_EP_M, TOL_DM, TOL_A2_RATIO, TOL_PLANT_DM = R1.TOL_EP_M, R1.TOL_DM, R1.TOL_A2_RATIO, R1.TOL_PLANT_DM


def sha256_file(p):
    return SB.sha256_file(p)


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# the base: token-level vectors of the adapted banks, and the C1-sized synthetic base of one replicate
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def token_vectors(z6: np.ndarray, zc: np.ndarray, lab: np.ndarray) -> dict:
    """Per-SET versions of `ep_admission_sim.components` (same terms, same pick rule); summing them per log with np.bincount reproduces `components` bit for bit (tested)."""
    L, P, F = R1.libs()
    pk_p, pk_c = P.rule_pick(z6), P.rule_pick(zc)
    sig, y = P.sigmoid64(zc[..., R1.EP]), lab[..., R1.EP].astype(np.float64)
    t = L.pair_terms(sig, y, pk_p, pk_c)
    c = t["changed"].astype(np.float64)
    nc = 1.0 - c
    v = {"y_p": t["y_p"], "s_p": t["s_p"], "y_e": t["y_e"], "s_e": t["s_e"], "Yu": t["y_p"] * nc, "Su": t["s_p"] * nc, "Ypc": t["y_p"] * c, "Spc": t["s_p"] * c, "Yec": t["y_e"] * c, "Sec": t["s_e"] * c}
    for h in HEADS:
        yh, sg = lab[..., R1.HCOL[h]].astype(np.float64), P.sigmoid64(zc[..., R1.HCOL[h]])
        v[f"OBS_{h}"] = 1.0 - P.at(yh, pk_c)
        v[f"PRED_{h}"] = 1.0 - P.at(sg, pk_c)
    return v


def load_pool(bank_dir: str) -> dict:
    """The adapted carve-B (or any) bank read THROUGH THE GATE'S LOADERS -> token vectors per seed + the log of each set.  No admission statistic is computed here."""
    p2, cands, rows = SB.load_with_gate(bank_dir)
    return {"log_idx": p2["log_idx"], "log_names": p2["log_names"], "n_sets": p2["n_sets"], "tok": {s: token_vectors(p2["z6"], cands[s]["z6"], p2["lab"]) for s in SEEDS},
            "n_changed": {s: int((SB.pickrule().rule_pick(cands[s]["z6"]) != SB.pickrule().rule_pick(p2["z6"])).sum()) for s in SEEDS}}


def load_geometry(path: str, log_names: list) -> np.ndarray:
    """C1's tokens per log, in the bank's log order (derived from token lists only: heldout minus Step A minus carve B; sha256 of the 905-token list pinned)."""
    g = json.load(open(path, encoding="utf-8"))
    cnt = g["b_rest_tokens_per_log"]
    if sorted(cnt) != sorted(log_names):
        raise SystemExit("REFUSED: the geometry's logs are not the bank's logs")
    if g["n"] != sum(cnt.values()) or g["b_rest_tokens_sha256"] != "3a85cc8124b3dd02297fffa176787b629369a75d82ed52c4440b32aa4cc70c4b":
        raise SystemExit("REFUSED: the geometry is not the sealed 905-token B-REST list")
    return np.array([cnt[l] for l in log_names], np.int64)


def outer_idx(rep: int, offset: int = 0) -> np.ndarray:
    return np.random.default_rng([OUTER_SALT, rep + offset]).integers(0, N_LOGS, N_LOGS)


def base_vectors(pool: dict, counts: np.ndarray, rep: int, offset: int = 0) -> dict:
    """One synthetic C1 base: for every log l, counts[l] sets drawn WITH REPLACEMENT from the pool's sets of that log (the same draw for all 3 seeds: the seeds share the tokens), summed per log."""
    rng = np.random.default_rng([BASE_SALT, rep + offset])
    sel, sel_log = [], []
    for l in range(N_LOGS):
        ids = np.nonzero(pool["log_idx"] == l)[0]
        sel.append(ids[rng.integers(0, len(ids), int(counts[l]))])
        sel_log.append(np.full(int(counts[l]), l))
    sel, sel_log = np.concatenate(sel), np.concatenate(sel_log)
    return {s: {k: np.bincount(sel_log, weights=v[sel], minlength=N_LOGS).astype(np.float64) for k, v in pool["tok"][s].items()} for s in SEEDS}


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# conditions and a replicate (the REAL estimator on one draw of logs)
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def conditions() -> dict:
    c = {"CAL": dict(kind="cal", a1=True, a2=True, heads=list(HEADS), kw={}, n=N_BAR)}
    for d in DELTAS:
        c[f"PLANT_EP_{d}"] = dict(kind="plant_ep", a1=True, a2=False, heads=[], kw={"delta": d}, n=N_OTHER)
    for h in HEADS:
        for rho in RHOS:
            c[f"PLANT_A2_{h}_{rho}"] = dict(kind="plant_a2", a1=False, a2=True, heads=[h], kw={"head": h, "rho": rho}, n=(N_BAR if (h, rho) == ("DAC", 1.25) else N_OTHER))
    c["IDENTITY"] = dict(kind="identity", a1=True, a2=False, heads=[], kw={}, n=N_ID)
    return c


def replicate(cond: dict, pool: dict, counts: np.ndarray, rep: int, draws: np.ndarray, offset: int = 0) -> dict:
    """-> {dm [3, 3] (point, lo, hi) per seed, a2 [3, 3, 3] (head, point / lo / hi) per seed, mp [3], me [3]}."""
    _, _, F = R1.libs()
    base = base_vectors(pool, counts, rep, offset)
    idx = outer_idx(rep, offset)
    dm, a2 = np.full((3, 3), np.nan), np.full((3, 3, 3), np.nan)
    mp, me = np.full(3, np.nan), np.full(3, np.nan)
    for s in SEEDS:
        if cond["kind"] == "identity":
            v = base[s]
            d = {"Yp": v["y_p"], "Sp": v["s_p"], "Ye": v["y_p"], "Se": v["s_p"]}
        else:
            d = R1.build_condition(base[s], cond["kind"], **cond["kw"])
        Yp, Sp, Ye, Se = d["Yp"][idx], d["Sp"][idx], d["Ye"][idx], d["Se"][idx]
        mp[s], me[s] = Yp.sum() / Sp.sum(), Ye.sum() / Se.sum()
        if cond["a1"]:
            ci = F.diff_ci(Yp, Sp, Ye, Se, draws)
            dm[s] = (ci["point"], ci["lo"], ci["hi"])
        for j, h in enumerate(HEADS):
            if cond["a2"] and h in cond["heads"]:
                ci = F.ratio_ci(d[f"OBS_{h}"][idx], d[f"PRED_{h}"][idx], draws)
                a2[s, j] = (ci["point"], ci["lo"], ci["hi"])
    return {"dm": dm, "a2": a2, "mp": mp, "me": me}


def run_conditions(pool, counts, names, n_reps, offset=0, log=print) -> dict:
    _, _, F = R1.libs()
    draws = F.boot_draws(N_LOGS, BOOT_B, BOOT_SEED)
    conds = conditions()
    out = {}
    for name in names:
        t0 = time.time()
        recs = [replicate(conds[name], pool, counts, r, draws, offset) for r in range(n_reps[name])]
        out[name] = {k: np.stack([r[k] for r in recs]) for k in ("dm", "a2", "mp", "me")}
        log(f"  {name}: {n_reps[name]} replicates ({time.time() - t0:.0f} s)")
    return out


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# decisions: the frozen gate's functions, and an inline restatement (PC4); NaN fails closed in both
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def ci_of(row) -> dict:
    return {"point": float(row[0]), "lo": float(row[1]), "hi": float(row[2])}


def a1_refuse_module(dm_i) -> bool:
    return A.a1_verdict_r1_seeds([ci_of(dm_i[s]) for s in SEEDS]) == "FAIL-A1"


def a1_refuse_inline(dm_i) -> bool:
    return any(not (dm_i[s, 1] <= 0.0 and dm_i[s, 0] <= 0.05) for s in SEEDS)


def a2_refuse_module(a2_i, heads, margin) -> bool:
    """The frozen module's per-head test `a2_head_ok_r1(ci, bar)` with the margin as the bar; margin 1.00 is also checked through `a2_verdict_r1_seeds` (the shipped default)."""
    j_of = {h: j for j, h in enumerate(HEADS)}
    cis = [{h: ci_of(a2_i[s, j_of[h]]) for h in heads} for s in SEEDS]
    if margin == 1.0:
        if tuple(heads) == tuple(A.A2_HEADS):
            return A.a2_verdict_r1_seeds(cis) == "FAIL-A2"
        return any(A.a2_verdict_r1(c, tuple(heads)) != "ADMIT-A2" for c in cis)
    return any(not all(A.a2_head_ok_r1(c[h], margin) for h in heads) for c in cis)


def a2_refuse_inline(a2_i, heads, margin) -> bool:
    j_of = {h: j for j, h in enumerate(HEADS)}
    return any(not (a2_i[s, j_of[h], 1] <= margin) for s in SEEDS for h in heads)


def summarize(name: str, cond: dict, rec: dict) -> dict:
    dm, a2, n = rec["dm"], rec["a2"], len(rec["dm"])
    R = R1.rate
    out = {"name": name, "n": int(n), "means": {"M_p": rec["mp"].mean(axis=0).tolist(), "M_e": rec["me"].mean(axis=0).tolist()}}
    dis = 0
    if cond["a1"]:
        mod = np.array([a1_refuse_module(dm[i]) for i in range(n)])
        inl = np.array([a1_refuse_inline(dm[i]) for i in range(n)])
        dis += int((mod != inl).sum())
        lo_hit, pt_hit = dm[:, :, 1] > 0.0, dm[:, :, 0] > 0.05
        seed_fail = np.array([[A.a1_verdict_r1(ci_of(dm[i, s])) != "ADMIT-A1" for s in SEEDS] for i in range(n)])
        out["A1"] = {"r1_any_seed": R(mod.sum(), n), "r1_per_seed": [R(seed_fail[:, s].sum(), n) for s in SEEDS], "r1_all_three_seeds": R(seed_fail.all(axis=1).sum(), n),
                     "r1_indep_expected_any": float(1 - np.prod([1 - seed_fail[:, s].mean() for s in SEEDS])),
                     "by_conjunct_seed_evaluations": {"lower_ci_gt_0_only": int((lo_hit & ~pt_hit).sum()), "point_gt_0.05_only": int((pt_hit & ~lo_hit).sum()), "both": int((lo_hit & pt_hit).sum()), "n": int(3 * n)},
                     "D_M_point_mean": dm[:, :, 0].mean(axis=0).tolist(), "D_M_point_sd": dm[:, :, 0].std(axis=0, ddof=1).tolist() if n > 1 else None,
                     "D_M_half_width_mean": ((dm[:, :, 2] - dm[:, :, 1]) / 2.0).mean(axis=0).tolist(), "n_nan_ci": int(np.isnan(dm[:, :, 1]).sum())}
    if cond["a2"]:
        heads = cond["heads"]
        j_of = {h: j for j, h in enumerate(HEADS)}
        out["A2"] = {"heads": heads, "n_nan_ci": int(np.isnan(a2[:, :, [j_of[h] for h in heads], 1]).sum()), "ratio_point_mean": {h: a2[:, :, j_of[h], 0].mean(axis=0).tolist() for h in heads}, "by_margin": {}}
        for t in MARGINS:
            mod = np.array([a2_refuse_module(a2[i], heads, t) for i in range(n)])
            inl = np.array([a2_refuse_inline(a2[i], heads, t) for i in range(n)])
            dis += int((mod != inl).sum())
            ent = {"any_head_any_seed": R(mod.sum(), n)}
            ent["per_head_any_seed"] = {h: R(np.array([a2_refuse_inline(a2[i], [h], t) for i in range(n)]).sum(), n) for h in heads}
            ent["per_head_per_seed"] = {h: [R(int((~(a2[:, s, j_of[h], 1] <= t)).sum()), n) for s in SEEDS] for h in heads}
            out["A2"]["by_margin"][f"{t:.2f}"] = ent
        if cond["a1"]:
            out["gate_A1_or_A2_at_1.05"] = R(sum(1 for i in range(n) if a1_refuse_module(dm[i]) or a2_refuse_module(a2[i], heads, BAR_MARGIN)), n)
    out["pc4_module_vs_inline_disagreements"] = int(dis)
    return out


def decide(a1_fr_upper: float, a2_fr_upper: float, dac_power_125: float) -> dict:
    """The committed rule, exactly: ALL bars hold => admission is read on C1's bank; ANY missed => fallback (b) (the Master Mind's call).  Pure function of three rates."""
    bars = {"A1_r1_false_refusal_wilson_upper_le_5pct": a1_fr_upper <= BAR_A1_FR_UPPER, "A2_1.05_false_refusal_wilson_upper_le_5pct": a2_fr_upper <= BAR_A2_FR_UPPER,
            "DAC_power_x1.25_margin1.05_point_ge_70pct": dac_power_125 >= BAR_DAC_POWER_125}
    return {"bars": bars, "all_met": bool(all(bars.values())), "missed": [k for k, v in bars.items() if not v],
            "consequence": ("ALL MET: admission is read on C1's bank (DQ6 a)" if all(bars.values()) else "MISSED: fallback (b), an S0-geometry bank of the search, with the miss reported; never (c). The Master Mind's call.")}


def controls(summ: dict, recs: dict) -> dict:
    pc = {}
    cal = summ["CAL"]
    ok1_m = all(abs(m - 1.0) <= TOL_EP_M for m in cal["means"]["M_p"] + cal["means"]["M_e"])
    ok1_d = all(abs(m) <= TOL_DM for m in cal["A1"]["D_M_point_mean"])
    ok1_a = all(abs(m - 1.0) <= TOL_A2_RATIO for h in HEADS for m in cal["A2"]["ratio_point_mean"][h])
    pc["PC1_calibrated_by_construction"] = {"pass": bool(ok1_m and ok1_d and ok1_a), "M_p": cal["means"]["M_p"], "M_e": cal["means"]["M_e"], "D_M_point_mean": cal["A1"]["D_M_point_mean"],
                                            "A2_ratio_point_mean": cal["A2"]["ratio_point_mean"], "tolerances": {"M": TOL_EP_M, "D_M": TOL_DM, "A2_ratio": TOL_A2_RATIO}}
    teeth = {}
    for d in DELTAS:
        s = summ[f"PLANT_EP_{d}"]
        teeth[f"plant_{d}_reads_the_planted_delta"] = {"pass": bool(all(abs(m - d) <= TOL_PLANT_DM for m in s["A1"]["D_M_point_mean"])), "D_M_point_mean": s["A1"]["D_M_point_mean"]}
        teeth[f"plant_{d}_FAILs_the_calibrated_check"] = {"pass": bool(not all(abs(m) <= TOL_DM for m in s["A1"]["D_M_point_mean"]))}
    pc["PC2_check_has_teeth"] = {**teeth, "pass": bool(all(v["pass"] for v in teeth.values()))}
    for h in HEADS:
        s = summ[f"PLANT_A2_{h}_1.25"]
        pc["PC2_check_has_teeth"][f"A2_plant_{h}_1.25_ratio_mean"] = s["A2"]["ratio_point_mean"][h]
    dis = {n: s["pc4_module_vs_inline_disagreements"] for n, s in summ.items()}
    pc["PC4_decision_code_is_the_modules"] = {"disagreements_by_condition": dis, "pass": bool(sum(dis.values()) == 0), "ep_admission_sha256": SB.GATE_SHA256}
    idn = recs["IDENTITY"]["dm"]
    pc["PC6_identity_candidate"] = {"all_D_M_exactly_0": bool((idn == 0.0).all()), "refusals": summ["IDENTITY"]["A1"]["r1_any_seed"], "pass": bool((idn == 0.0).all() and summ["IDENTITY"]["A1"]["r1_any_seed"]["k"] == 0)}
    pc["PC5_determinism"] = "run `repro` (separate process)"
    return pc


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def design_guard() -> str:
    if not os.path.exists(DESIGN):
        raise SystemExit(f"REFUSED: {os.path.abspath(DESIGN)} does not exist (the design must be written and hashed before the simulation runs)")
    side = DESIGN[:-3] + ".sha256"
    if not os.path.exists(side) or open(side, encoding="utf-8").read().split()[0].lower() != sha256_file(DESIGN):
        raise SystemExit("REFUSED: SIM_DESIGN.sha256 is missing or does not equal SIM_DESIGN.md (hash it BEFORE the first run)")
    return sha256_file(DESIGN)


def setup(a):
    pool = load_pool(a.bank)
    counts = load_geometry(a.geometry, pool["log_names"])
    return pool, counts


def part_names(i: int, n: int) -> list:
    names = list(conditions())
    return [x for j, x in enumerate(names) if j % n == i]


def pc3_shortcut(bank_dir: str) -> dict:
    """PC3: the per-log shortcut (bincount vectors -> F.diff_ci / F.ratio_ci with the registered draws) equals the frozen gate's own `analyze_candidate` on the SAME adapted bank, to 0.0, for all 3 seeds.
    Run it on a SYNTHETIC adapted bank only: it computes the candidate's A1 / A2 statistics."""
    L, P, F = R1.libs()
    p2, cands, rows = SB.load_with_gate(bank_dir)
    nl = p2["n_logs"]
    S = A.make_stats(p2["log_idx"], nl)
    draws = F.boot_draws(nl, BOOT_B, BOOT_SEED)
    pl = lambda x: np.bincount(p2["log_idx"], weights=x, minlength=nl).astype(np.float64)       # noqa: E731
    out, ok = {}, True
    for s in SEEDS:
        an = A.analyze_candidate(p2["z6"], cands[s]["z6"], p2["lab"], S)
        v = token_vectors(p2["z6"], cands[s]["z6"], p2["lab"])
        eq = {"D_M": R1.eq_ci(an["G"]["D_M"], F.diff_ci(pl(v["y_p"]), pl(v["s_p"]), pl(v["y_e"]), pl(v["s_e"]), draws))}
        for h in HEADS:
            eq[h] = R1.eq_ci(an["A2_reliability"][h], F.ratio_ci(pl(v[f"OBS_{h}"]), pl(v[f"PRED_{h}"]), draws))
        out[f"s{s}"] = eq
        ok &= all(eq.values())
    return {"pass": bool(ok), "per_seed": out, "n_logs": nl, "on": "SYNTHETIC adapted bank (planted data; no real candidate statistic)"}


def cmd_pc3(a) -> int:
    import fixture_synth as FX
    fx = FX.make_fixture(os.path.join(a.workdir, "pc3_fixture"), n_tokens=120, n_logs=24, seed=3)
    out = os.path.join(a.workdir, "pc3_bank")
    SB.build(fx["bundle"], fx["search"], fx["labels"], out, {"token_sha256": fx["token_sha256"]})
    res = pc3_shortcut(out)
    json.dump(res, open(os.path.join(a.workdir, "pc3.json"), "w"), indent=1)
    print("PC3", res["pass"])
    return 0 if res["pass"] else 3


def cmd_cost(a) -> int:
    design_guard()
    pool, counts = setup(a)
    _, _, F = R1.libs()
    draws = F.boot_draws(N_LOGS, BOOT_B, BOOT_SEED)
    conds, res = conditions(), {}
    for name in ("CAL", "PLANT_EP_0.135", "PLANT_A2_DAC_1.25"):
        t0 = time.time()
        for r in range(3):
            replicate(conds[name], pool, counts, r, draws)
        res[name] = (time.time() - t0) / 3.0
    est = sum(c["n"] * (res["CAL"] if c["a1"] and c["a2"] else res["PLANT_EP_0.135"] if c["a1"] else res["PLANT_A2_DAC_1.25"]) for c in conds.values())
    out = {"seconds_per_replicate": res, "estimated_cpu_seconds_all_conditions": est, "estimated_wall_seconds_2_processes": est / 2.0, "tokens_per_log": counts.tolist(), "n_tokens": int(counts.sum()), "pool_sets": pool["n_sets"]}
    os.makedirs(a.workdir, exist_ok=True)
    json.dump(out, open(os.path.join(a.workdir, "cost.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))
    return 0


def cmd_run(a) -> int:
    dsha = design_guard()
    pool, counts = setup(a)
    names = part_names(a.part, a.nparts)
    n_reps = {k: (a.n_reps if a.n_reps else v["n"]) for k, v in conditions().items()}
    t0 = time.time()
    print(f"run part {a.part}/{a.nparts}: {names}, B = {BOOT_B}, seed {BOOT_SEED}, rep_offset {a.rep_offset}", flush=True)
    recs = run_conditions(pool, counts, names, n_reps, a.rep_offset, log=lambda m: print(m, flush=True))
    os.makedirs(a.workdir, exist_ok=True)
    np.savez_compressed(os.path.join(a.workdir, f"replicates_part{a.part}.npz"), **{f"{n}__{k}": v for n, r in recs.items() for k, v in r.items()})
    json.dump({"design_sha256": dsha, "part": a.part, "names": names, "wall_seconds": time.time() - t0, "rep_offset": a.rep_offset, "n_changed_by_seed_in_pool": pool["n_changed"], "pool_sets": pool["n_sets"],
               "bank_manifests": {n: sha256_file(os.path.join(a.bank, n, "MANIFEST.sha256.json")) for n in ("p2", "s0", "s1", "s2")}},
              open(os.path.join(a.workdir, f"run_part{a.part}.json"), "w"), indent=1)
    print(f"PART {a.part} DONE_MARKER ({time.time() - t0:.0f} s)")
    return 0


def load_parts(workdir: str, nparts: int = 2) -> dict:
    recs = {}
    for i in range(nparts):
        z = np.load(os.path.join(workdir, f"replicates_part{i}.npz"))
        for k in z.files:
            n, key = k.split("__")
            recs.setdefault(n, {})[key] = z[k]
    return recs


def cmd_report(a) -> int:
    dsha = design_guard()
    recs = load_parts(a.workdir)
    conds = conditions()
    if sorted(recs) != sorted(conds):
        raise SystemExit(f"REFUSED: parts do not cover every condition: {sorted(set(conds) - set(recs))}")
    summ = {n: summarize(n, conds[n], recs[n]) for n in conds}
    pc = controls(summ, recs)
    pc3 = json.load(open(a.pc3)) if a.pc3 else {"pass": None, "note": "not supplied"}
    pc["PC3_shortcut_is_the_estimator"] = pc3
    rep = json.load(open(a.repro)) if a.repro else {"pass": None}
    pc["PC5_determinism"] = rep
    ctrl_ok = bool(pc["PC1_calibrated_by_construction"]["pass"] and pc["PC2_check_has_teeth"]["pass"] and pc["PC4_decision_code_is_the_modules"]["pass"] and pc["PC6_identity_candidate"]["pass"]
                   and pc3.get("pass") is True and rep.get("pass") is True)
    cal, dac = summ["CAL"], summ["PLANT_A2_DAC_1.25"]
    a1 = cal["A1"]["r1_any_seed"]
    a2 = cal["A2"]["by_margin"]["1.05"]["any_head_any_seed"]
    pw = dac["A2"]["by_margin"]["1.05"]["per_head_any_seed"]["DAC"]
    dec = decide(a1["wilson95"][1], a2["wilson95"][1], pw["rate"])
    run0 = json.load(open(os.path.join(a.workdir, "run_part0.json")))
    res = {"schema": "ep-admission-searchbank-sim-v1", "created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"), "design_sha256": dsha, "controls": pc, "controls_ok": ctrl_ok,
           "decision": dec, "bar_inputs": {"A1_r1_false_refusal": a1, "A2_1.05_false_refusal": a2, "DAC_power_x1.25_margin1.05": pw},
           "conditions": summ, "geometry": {"n_logs": N_LOGS, "n_tokens": 905, "seeds": 3, "n_changed_by_seed_in_pool": run0["n_changed_by_seed_in_pool"], "pool_sets": run0["pool_sets"]},
           "boot": {"B": BOOT_B, "seed": BOOT_SEED, "unit": "log", "draws": f"boot_draws({N_LOGS}, {BOOT_B}, {BOOT_SEED})"}, "salts": {"base": BASE_SALT, "outer": OUTER_SALT, "r1_salts_not_reused": list(R1_SALTS)},
           "code_sha256": {"ep_admission.py": SB.GATE_SHA256, "ep_admission_sim.py": sha256_file(os.path.join(R1_DIR, "ep_admission_sim.py")), "epadm_sim_c1.py": sha256_file(os.path.abspath(__file__)),
                           "epadm_searchbank.py": sha256_file(os.path.join(HERE, "epadm_searchbank.py"))},
           "wall_seconds_part0": run0["wall_seconds"], "bank_manifests": run0["bank_manifests"]}
    json.dump(res, open(os.path.join(a.workdir, "sim_results.json"), "w", encoding="utf-8", newline="\n"), indent=1, sort_keys=True, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    np.savez_compressed(os.path.join(a.workdir, "replicates.npz"), **{f"{n}__{k}": v for n, r in recs.items() for k, v in r.items()})
    print(f"REPORT controls_ok={ctrl_ok} all_bars_met={dec['all_met']} missed={dec['missed']}")
    return 0 if ctrl_ok else 3


def cmd_repro(a) -> int:
    design_guard()
    pool, counts = setup(a)
    _, _, F = R1.libs()
    draws = F.boot_draws(N_LOGS, BOOT_B, BOOT_SEED)
    recs = load_parts(a.workdir)
    conds, ok = conditions(), True
    for name in ("CAL", "PLANT_A2_DAC_1.25"):
        for r in range(20):
            rec = replicate(conds[name], pool, counts, r, draws, a.rep_offset)
            for k in ("dm", "a2", "mp", "me"):
                ok &= bool(np.array_equal(rec[k], recs[name][k][r], equal_nan=True))
    out = {"pass": bool(ok), "bit_identical_first_20_replicates_of_2_conditions": bool(ok), "pid": os.getpid()}
    json.dump(out, open(os.path.join(a.workdir, "repro.json"), "w"), indent=1)
    print("REPRO", out)
    return 0 if ok else 3


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for nm, fn in (("cost", cmd_cost), ("run", cmd_run), ("repro", cmd_repro)):
        p = sub.add_parser(nm)
        p.add_argument("--bank", required=True, help="the adapter's output dir (p2/, s0/, s1/, s2/, ranked_logs.tsv)")
        p.add_argument("--geometry", default=os.path.join(HERE, "..", "raw", "geometry_c1.json"))
        p.add_argument("--workdir", required=True)
        p.add_argument("--rep-offset", type=int, default=0)
        if nm == "run":
            p.add_argument("--part", type=int, required=True); p.add_argument("--nparts", type=int, default=2); p.add_argument("--n-reps", type=int, default=0)
        p.set_defaults(fn=fn)
    q = sub.add_parser("pc3")
    q.add_argument("--workdir", required=True)
    q.set_defaults(fn=cmd_pc3)
    r = sub.add_parser("report")
    r.add_argument("--workdir", required=True); r.add_argument("--pc3", default=None); r.add_argument("--repro", default=None)
    r.set_defaults(fn=cmd_report)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
