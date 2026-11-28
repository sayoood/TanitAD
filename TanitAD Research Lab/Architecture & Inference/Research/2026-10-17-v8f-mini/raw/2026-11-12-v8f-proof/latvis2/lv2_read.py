"""lv2_read.py: the DEV-SIDE reader of the tool `latvis2` (card PREREG_V8F_PROOF_ADD38 sections 4 and 5). Numpy only.

  python lv2_read.py read --dump DUMP_LV2_G_s0.npz DUMP_LV2_G_s1.npz --arm G --seeds 0,1 --out-dir DIR --ref-windows LATENTS_CTX_G_s0_R1.npz
  python lv2_read.py pair --a-dir DIR_RR --b-dir DIR_G --out-dir DIR_PAIR

Statistic D = error(present state) - error(imagined state) per window and cell (above 0: the imagination is closer to the true tokens of t + 2.0 s than the copy); headroom H = error(present)
- error(true future state). On all 40 cells and on the MOVING cells (per window the 10 of 40 with the largest true token change). Per seed: episode-cluster bootstrap (B 2,000, rng 0); pooled:
the two-level draw of imaghead's `boot_plan` (seeds with replacement, then episodes). Tier T0; never driving performance."""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, os.path.join(HERE, "..", "imaghead")):
    if os.path.exists(os.path.join(_p, "imag_read.py")) and _p not in sys.path:
        sys.path.append(_p)

TIER = "T0 (a world-model diagnostic on logged frames, never driving performance)"
STRAT = "STRATEGIC: PAUSED (PI item 25)"
FAMILIES = "LONGITUDINAL, LATERAL, TACTICAL: not measured by this read"
NCELL, NMOV, NROW, NCOL = 40, 10, 4, 10
ROW_NAMES = ("sky", "far", "near", "hood")
B_DEFAULT = 2000
K1_BAR, TOL_K5 = 0.02, 1e-4
KEYS = ("D_all", "D_mov", "H_all", "H_mov") + tuple(f"D_row{r}" for r in range(NROW))


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def boot_plan(K, n_ep, B, rng_seed=0):
    """imag_read.boot_plan (seeds with replacement, then episodes inside each drawn seed); reproduced bit for bit, and tested equal to it when imaghead is importable."""
    rng = np.random.default_rng(rng_seed)
    pick = rng.integers(0, K, size=(B, K))
    C = rng.multinomial(n_ep, np.full(n_ep, 1.0 / n_ep), size=(B, K)).astype(np.float64)
    return pick, C


def ci(x):
    lo, hi = np.nanpercentile(x, [2.5, 97.5])
    return [float(lo), float(hi)]


def load_dump(path):
    z = np.load(path, allow_pickle=True)
    d = {k: z[k] for k in z.files}
    d["eid"] = d["eid"].astype(str)
    with open(path + ".json", encoding="utf-8") as fh:
        d["meta"] = json.load(fh)
    d["path"] = path.replace(os.sep, "/")
    return d


def moving_mask(chg, n_mov=NMOV):
    """[n, 40] bool: per window the n_mov cells with the largest true token change (ties: the lower cell index)."""
    order = np.argsort(-np.asarray(chg, np.float64), axis=1, kind="stable")[:, :n_mov]
    m = np.zeros(chg.shape, bool)
    np.put_along_axis(m, order, True, axis=1)
    return m


def window_values(e_ii, e_i, e_iii, mov):
    """-> dict key -> [n] per-window value (mean over the cells of the set)."""
    D, H = e_ii.astype(np.float64) - e_i, e_ii.astype(np.float64) - e_iii
    out = {"D_all": D.mean(1), "H_all": H.mean(1), "D_mov": (D * mov).sum(1) / mov.sum(1), "H_mov": (H * mov).sum(1) / mov.sum(1)}
    for r in range(NROW):
        out[f"D_row{r}"] = D[:, r * NCOL:(r + 1) * NCOL].mean(1)
    return out


def episode_sums(v, epi, n_ep):
    return np.bincount(epi, weights=v, minlength=n_ep), np.bincount(epi, minlength=n_ep).astype(np.float64)


def seed_stats(d, imagined="imag", mutate_mov=None):
    """Per-episode sums of every statistic for one seed. imagined: 'imag' (the tool's), 'pres' (k3: the present state given as the imagined state), 'true' (k4)."""
    e_i = {"imag": d["err_imag"], "pres": d["err_pres"], "true": d["err_true"]}[imagined]
    mov = moving_mask(d["tok_change"]) if mutate_mov is None else mutate_mov
    vals = window_values(d["err_pres"], e_i, d["err_true"], mov)
    ueid, epi = np.unique(d["eid"], return_inverse=True)
    S, N = {}, None
    for k in KEYS:
        S[k], N = episode_sums(vals[k], epi, len(ueid))
    return {"S": S, "N": N, "ueid": ueid, "mov": mov}


def point(St, k):
    return float(St["S"][k].sum() / St["N"].sum())


def seed_ci(S, N, B, rng_seed=0):
    n_ep = len(N)
    C = np.random.default_rng(rng_seed).multinomial(n_ep, np.full(n_ep, 1.0 / n_ep), size=B).astype(np.float64)
    return ci((C @ S) / (C @ N))


def pool_draws(S, N, plan):
    """S [K, n_ep] numerators, N [n_ep]: the pooled statistic of every draw (mean over the K slots of the ratio of the drawn seed)."""
    pick, C = plan
    B, K = pick.shape
    out = np.zeros(B)
    for j in range(K):
        k, c = pick[:, j], C[:, j, :]
        out += (c * S[k]).sum(1) / (c * N).sum(1)
    return out / K


def summarize(sts, plan, B):
    """sts: per-seed seed_stats. -> {key: {per_seed: [...], per_seed_ci: [...], pooled: x, pooled_ci: [lo, hi]}} and the share of the headroom."""
    out = {}
    N = sts[0]["N"]
    draws = {}
    for k in KEYS:
        S = np.stack([s["S"][k] for s in sts])
        dr = pool_draws(S, N, plan)
        draws[k] = dr
        pts = [point(s, k) for s in sts]
        out[k] = {"per_seed": pts, "per_seed_ci": [seed_ci(s["S"][k], N, B) for s in sts], "pooled": float(np.mean(pts)), "pooled_ci": ci(dr)}
    for a, b, name in (("D_all", "H_all", "share_all"), ("D_mov", "H_mov", "share_mov")):
        dr = np.divide(draws[a], draws[b], out=np.full_like(draws[a], np.nan), where=draws[b] != 0)
        sd = lambda x, y: float(x / y) if y != 0 else float("nan")
        out[name] = {"per_seed": [sd(out[a]["per_seed"][i], out[b]["per_seed"][i]) for i in range(len(sts))], "pooled": sd(out[a]["pooled"], out[b]["pooled"]), "pooled_ci": ci(dr) if np.isfinite(dr).any() else [float("nan")] * 2}
    return out, draws


# ------------------------------------------------------------------------------------------------ controls
def control_k1(dumps):
    v = [float(np.max(d["skill_heldout_permuted"])) for d in dumps]
    return {"name": "k1", "rule": f"held-out skill of the token map with the states permuted across frames <= {K1_BAR}", "value_max_per_seed": v, "pass": bool(all(x <= K1_BAR for x in v))}


def control_k2(dumps):
    ok_tr = [bool(np.all(d["skill_train"] > d["skill_heldout"])) for d in dumps]
    ok_pr = [bool(np.all(d["skill_heldout_present"] > 0)) for d in dumps]
    return {"name": "k2", "rule": "training-fold skill > held-out skill in every fold, and held-out skill of the true present state on the present tokens > 0 in every fold",
            "train_gt_heldout": ok_tr, "present_heldout_gt0": ok_pr, "skill_train": [d["skill_train"].tolist() for d in dumps], "skill_heldout": [d["skill_heldout"].tolist() for d in dumps],
            "skill_heldout_present": [d["skill_heldout_present"].tolist() for d in dumps], "pass": bool(all(ok_tr) and all(ok_pr))}


def control_k3(dumps, substitute=True):
    """The present state given as the imagined state: D exactly 0 (all cells and moving cells). `substitute=False` is the deliberate regression (the real imagined state is read)."""
    mx = []
    for d in dumps:
        st = seed_stats(d, "pres" if substitute else "imag")
        mx.append(float(max(np.abs(st["S"]["D_all"]).max(), np.abs(st["S"]["D_mov"]).max())))
    return {"name": "k3", "rule": "with the present state given as the imagined state D is exactly 0", "max_abs_episode_sum_of_D": mx, "pass": bool(all(x == 0.0 for x in mx))}


def control_k4(dumps, plan, B):
    """Known positive: the true future state given as the imagined state: D equals the headroom exactly and the pooled two-level interval lies above 0 (all cells and moving cells)."""
    sts = [seed_stats(d, "true") for d in dumps]
    eq = [bool(np.array_equal(s["S"]["D_all"], s["S"]["H_all"]) and np.array_equal(s["S"]["D_mov"], s["S"]["H_mov"])) for s in sts]
    summ, _ = summarize(sts, plan, B)
    lo = {k: summ[k]["pooled_ci"][0] for k in ("D_all", "D_mov")}
    return {"name": "k4", "rule": "with the true future state given as the imagined state D equals the headroom and its two-level interval lies above 0", "D_equals_headroom": eq,
            "pooled": {k: summ[k]["pooled"] for k in ("D_all", "D_mov")}, "pooled_ci": {k: summ[k]["pooled_ci"] for k in ("D_all", "D_mov")}, "pass": bool(all(eq) and all(v > 0 for v in lo.values()))}


def control_k5(dumps):
    v = [float(d["k5_max_abs"]) for d in dumps]
    run = [bool(np.isfinite(x)) for x in v]
    return {"name": "k5", "rule": f"the rollout of the tool equals the banked rollout (pred_2s) of the same checkpoint to {TOL_K5}", "max_abs_per_seed": v, "run": run,
            "files": [(d["meta"].get("k5") or {}).get("file") for d in dumps], "pass": bool(all(run) and all(x <= TOL_K5 for x in v))}


def control_k6(dumps, ref_path=None):
    if not ref_path:
        return {"name": "k6", "rule": "windows, folds and episodes equal those of the r-b read", "pass": False, "status": "NOT-RUN (no --ref-windows)"}
    R = np.load(ref_path, allow_pickle=True)
    reid, rg0 = R["eid"].astype(str), R["g0"].astype(np.int64)
    rfold = R["w_fold"].astype(np.int64) if "w_fold" in R.files else None
    res = []
    for d in dumps:
        same_n = len(reid) == len(d["eid"])
        same_ids = bool(same_n and np.array_equal(reid, d["eid"]) and np.array_equal(rg0, d["g0"].astype(np.int64)))
        same_fold = bool(rfold is not None and same_n and np.array_equal(rfold, d["fold"].astype(np.int64)))
        same_ep = bool(set(reid.tolist()) == set(d["eid"].tolist()))
        res.append({"n": int(len(d["eid"])), "ref_n": int(len(reid)), "same_ids_and_order": same_ids, "same_folds": same_fold, "same_episodes": same_ep, "n_episodes": int(len(set(d["eid"])))})
    return {"name": "k6", "rule": "the windows, folds and episodes equal those of the r-b read (same ids, same count)", "ref": ref_path.replace(os.sep, "/"), "per_seed": res,
            "pass": bool(all(r["same_ids_and_order"] and r["same_folds"] and r["same_episodes"] for r in res))}


# ------------------------------------------------------------------------------------------------ names
def name_arm(summ, n_seeds):
    if n_seeds < 2:
        return None, "needs both seeds"
    s = summ["D_mov"]
    ok = all(x > 0 for x in s["per_seed"]) and s["pooled_ci"][0] > 0
    return ("TOKENS-IMAGINED" if ok else "TOKENS-FLAT"), "D on the moving cells above 0 at both seeds and the two-level interval above 0" if ok else "not both seeds above 0 with the two-level interval above 0"


def name_pair(a_arm, b_arm, diff_per_seed, diff_ci, n_seeds, controls_ok):
    if not controls_ok:
        return None, "a control of an arm failed"
    if n_seeds < 2:
        return None, "needs both seeds"
    if (a_arm, b_arm) != ("RR", "G"):
        return None, "the card names RR minus G only"
    ok = all(x > 0 for x in diff_per_seed) and diff_ci[0] > 0
    return ("RR-IMAGINES-TOKENS-BETTER" if ok else None), ("D of RR minus D of G above 0 at both seeds, paired two-level interval above 0" if ok else "nothing is named for the pair")


# ------------------------------------------------------------------------------------------------ commands
def hash_arr(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


K5P_BARS = {"a_rollout_fn_max_abs": 1e-4, "c_share_min": 0.99, "d_cosine_min": 0.999, "e_tie_max": 1e-6}


def control_k5_parts(dumps, parts):
    """Note PREREG_V8F_PROOF_ADD38_N2: k5 read in parts from the pod probe (ops/lv2_k5_probe.py), one json file per seed."""
    if len(parts) != len(dumps):
        raise SystemExit("REFUSED: --k5-parts needs one probe file per dump")
    rows, ok = [], True
    for d, path in zip(dumps, parts):
        with open(path, encoding="utf-8") as fh:
            q = json.loads(fh.read().strip().splitlines()[-1])
        r = {"probe": path.replace(os.sep, "/"), "a_rollout_fn_max_abs": float(q["a_rollout_fn_on_banked_inputs_max_abs"]), "b_actions_max_abs": float(q["d_actions_max_abs"]),
             "b_same_window_list": bool(q["same_window_list"]), "c_start_frame_share": float(q["c_nearest_frame_to_banked_start_is_4_share"]),
             "c_future_frame_share": float(q["c_nearest_frame_to_banked_future_is_14_share"]), "d_cosine_start": float(q["b_start_state_pod_vs_banked"]["cosine_mean"]),
             "d_cosine_future": float(q["b_future_state_pod_vs_banked"]["cosine_mean"]), "e_tie_to_dump": abs(float(q["rollout_pod_vs_banked"]["max_abs"]) - float(d["k5_max_abs"])),
             "e_same_ckpt": str(q.get("ckpt")) == str(d["meta"].get("ckpt")), "rms_diff_start": float(q["b_start_state_pod_vs_banked"]["rms_diff"]),
             "rms_state": float(q["b_start_state_pod_vs_banked"]["rms_state"])}
        r["pass"] = bool(r["a_rollout_fn_max_abs"] <= K5P_BARS["a_rollout_fn_max_abs"] and r["b_actions_max_abs"] == 0.0 and r["b_same_window_list"]
                         and r["c_start_frame_share"] >= K5P_BARS["c_share_min"] and r["c_future_frame_share"] >= K5P_BARS["c_share_min"]
                         and r["d_cosine_start"] >= K5P_BARS["d_cosine_min"] and r["d_cosine_future"] >= K5P_BARS["d_cosine_min"]
                         and r["e_tie_to_dump"] <= K5P_BARS["e_tie_max"] and r["e_same_ckpt"])
        ok = ok and r["pass"]
        rows.append(r)
    return {"name": "k5p", "rule": "k5 in parts (note ADD38_N2): rollout function on the banked inputs to 1e-4; same windows and actions; frames aligned (share >= 0.99); "
            "pod and banked states agree (mean cosine >= 0.999); the probe is tied to the dump (1e-6, same checkpoint)", "per_seed": rows, "bars": K5P_BARS, "pass": bool(ok)}


def read_dumps(dumps, arm, seeds, B=B_DEFAULT, ref_windows=None, mutate=None, k5_parts=None):
    """-> result dict, stats arrays. `mutate` (tests): 'k3_not_substituted'."""
    n_ep = len(np.unique(dumps[0]["eid"]))
    plan = boot_plan(len(dumps), n_ep, B, 0)
    ctrl = [control_k1(dumps), control_k2(dumps), control_k3(dumps, substitute=(mutate != "k3_not_substituted")), control_k4(dumps, plan, B), control_k5(dumps), control_k6(dumps, ref_windows)]
    passed_written = bool(all(c["pass"] for c in ctrl))
    passed = passed_written
    if k5_parts:
        ctrl.append(control_k5_parts(dumps, k5_parts))
        passed = bool(all(c["pass"] for c in ctrl if c["name"] != "k5"))
    sts = [seed_stats(d) for d in dumps]
    for s in sts[1:]:
        if not np.array_equal(s["ueid"], sts[0]["ueid"]):
            raise SystemExit("REFUSED: the seeds hold different episodes")
    summ, _ = summarize(sts, plan, B)
    nm, why = name_arm(summ, len(dumps))
    res = {"arm": arm, "seeds": seeds, "tier": TIER, "families": FAMILIES, "strategic": STRAT, "card": "PREREG_V8F_PROOF_ADD38", "B": B, "bootstrap": "per seed: episode-cluster; pooled: two-level (seeds, then episodes), rng seed 0",
           "n_windows": int(len(dumps[0]["eid"])), "n_episodes": int(n_ep), "moving_cells_per_window": NMOV, "controls": ctrl, "controls_pass": passed, "controls_pass_as_written": passed_written,
           "k5_form": ("in parts (note PREREG_V8F_PROOF_ADD38_N2); k5 as written: " + ("PASS" if ctrl[4]["pass"] else "FAIL") if k5_parts else "as written"), "stats": summ,
           "D_per_row": {ROW_NAMES[r]: summ[f"D_row{r}"] for r in range(NROW)},
           "name": (nm if passed else None), "name_rule": why, "name_withheld_reason": (None if passed else "a control failed: " + ",".join(c["name"] for c in ctrl if not c["pass"] and not (k5_parts and c["name"] == "k5"))),
           "dumps": [{"path": d["path"], "ckpt_sha256": str(d["meta"].get("ckpt_sha256")), "sha256": (sha256_file(d["path"]) if os.path.exists(d["path"]) else None)} for d in dumps]}
    arrs = {"ueid": sts[0]["ueid"], "N": sts[0]["N"], "windows_sha": np.array(hash_arr(np.char.add(dumps[0]["eid"], dumps[0]["g0"].astype(str)))),
            "mov_sha": np.array([hash_arr(s["mov"]) for s in sts])}
    for i, s in enumerate(sts):
        for k in KEYS:
            arrs[f"S{i}_{k}"] = s["S"][k]
    return res, arrs


def fmt(x, p=4):
    return f"{x:+.{p}f}"


def verdict(r):
    L = [f"# VERDICT_LV2_{r['arm']}: the token-level read of what the rollout transports (card PREREG_V8F_PROOF_ADD38)", "",
         f"Tier {TIER}. {r['families']}. {r['strategic']}.", "", "## Controls (all must pass or no name is printed)", ""]
    for c in r["controls"]:
        L.append(f"* **{c['name']}**: {'PASS' if c['pass'] else 'FAIL'} - {c['rule']}" + (f" ({c['status']})" if c.get("status") else ""))
    L += ["", f"controls_pass: {r['controls_pass']} (k5 {r.get('k5_form', 'as written')}; all controls as written: {r.get('controls_pass_as_written', r['controls_pass'])})", "", "## Statistic D (error of the present state minus error of the imagined state; above 0 = the imagination is closer to the true tokens of t + 2.0 s than the copy)", "",
          f"{r['n_windows']} windows, {r['n_episodes']} episodes, seeds {r['seeds']}; moving cells = per window the {NMOV} of {NCELL} with the largest true token change. Units: squared error in the 64-component token space, per cell.", "",
          "| statistic | per seed | pooled | pooled 95 % interval (two-level) |", "|---|---|---|---|"]
    for k, lab in (("D_all", "D, all cells"), ("D_mov", "D, moving cells"), ("H_all", "headroom, all cells"), ("H_mov", "headroom, moving cells"), ("share_all", "share of headroom, all"), ("share_mov", "share of headroom, moving")):
        s = r["stats"][k]
        L.append(f"| {lab} | {', '.join(fmt(x) for x in s['per_seed'])} | {fmt(s['pooled'])} | [{fmt(s['pooled_ci'][0])}, {fmt(s['pooled_ci'][1])}] |")
    L += ["", "D per row of the grid (all cells of the row):", "", "| row | per seed | pooled | pooled interval |", "|---|---|---|---|"]
    for nm, s in r["D_per_row"].items():
        L.append(f"| {nm} | {', '.join(fmt(x) for x in s['per_seed'])} | {fmt(s['pooled'])} | [{fmt(s['pooled_ci'][0])}, {fmt(s['pooled_ci'][1])}] |")
    L += ["", "## Name", "", f"* {r['name'] or 'NONE'}" + (f" ({r['name_rule']})" if r["name"] else f" ({r['name_withheld_reason'] or r['name_rule']})"), "",
          "Limits: linear maps show what is linearly readable; the token grid is a pooled 4 x 10 read of the picture; two seeds per arm; one corpus; 139 episodes. A separated interval over episodes is necessary, not sufficient (the bootstrap does not see training variance). No picture is evidence for a claim."]
    return "\n".join(L) + "\n"


def cmd_read(a):
    dumps = [load_dump(p) for p in a.dump]
    seeds = [int(x) for x in a.seeds.split(",")]
    if len(seeds) != len(dumps):
        raise SystemExit("REFUSED: --seeds and --dump differ in length")
    res, arrs = read_dumps(dumps, a.arm, seeds, a.B, a.ref_windows, k5_parts=a.k5_parts)
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, f"RESULT_LV2_{a.arm}.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(res, fh, indent=1)
    with open(os.path.join(a.out_dir, f"VERDICT_LV2_{a.arm}.md"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(verdict(res))
    np.savez_compressed(os.path.join(a.out_dir, f"STATS_LV2_{a.arm}.npz"), **arrs)
    print(f"lv2_read {a.arm}: controls_pass {res['controls_pass']}; name {res['name']}; D_mov pooled {res['stats']['D_mov']['pooled']:+.5f}")
    return 0


def pair(sa, sb, ra, rb, a_arm, b_arm, B=B_DEFAULT):
    for k in ("ueid", "N", "windows_sha", "mov_sha"):
        if not np.array_equal(sa[k], sb[k]):
            raise SystemExit(f"REFUSED: the two arms differ in {k} (not the same windows or moving cells)")
    K = len([k for k in sa.files if k.startswith("S") and k.endswith("_D_all")])
    N = sa["N"]
    plan = boot_plan(K, len(N), B, 0)
    out = {}
    for key in ("D_mov", "D_all"):
        diff = np.stack([sa[f"S{i}_{key}"] - sb[f"S{i}_{key}"] for i in range(K)])
        pts = [float(diff[i].sum() / N.sum()) for i in range(K)]
        out[key] = {"per_seed": pts, "per_seed_ci": [seed_ci(diff[i], N, B) for i in range(K)], "pooled": float(np.mean(pts)), "pooled_ci": ci(pool_draws(diff, N, plan))}
    ok = bool(ra["controls_pass"] and rb["controls_pass"])
    nm, why = name_pair(a_arm, b_arm, out["D_mov"]["per_seed"], out["D_mov"]["pooled_ci"], K, ok)
    return {"a": a_arm, "b": b_arm, "tier": TIER, "families": FAMILIES, "strategic": STRAT, "card": "PREREG_V8F_PROOF_ADD38", "B": B, "estimator": "paired (same windows), per seed episode-cluster, pooled two-level",
            "controls_pass": {"a": ra["controls_pass"], "b": rb["controls_pass"]}, "diff": out, "name": nm, "name_rule": why}


def cmd_pair(a):
    ra = json.load(open(next(os.path.join(a.a_dir, f) for f in os.listdir(a.a_dir) if f.startswith("RESULT_LV2_")), encoding="utf-8"))
    rb = json.load(open(next(os.path.join(a.b_dir, f) for f in os.listdir(a.b_dir) if f.startswith("RESULT_LV2_")), encoding="utf-8"))
    sa = np.load(os.path.join(a.a_dir, f"STATS_LV2_{ra['arm']}.npz"), allow_pickle=True)
    sb = np.load(os.path.join(a.b_dir, f"STATS_LV2_{rb['arm']}.npz"), allow_pickle=True)
    r = pair(sa, sb, ra, rb, ra["arm"], rb["arm"], a.B)
    os.makedirs(a.out_dir, exist_ok=True)
    stem = f"PAIR_LV2_{ra['arm']}_minus_{rb['arm']}"
    with open(os.path.join(a.out_dir, stem + ".json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(r, fh, indent=1)
    L = [f"# {stem}", "", f"Tier {TIER}. {FAMILIES}. {STRAT}.", "", f"Controls pass: {r['controls_pass']}", "", "| statistic (D of A minus D of B) | per seed | pooled | pooled paired two-level interval |", "|---|---|---|---|"]
    for k, s in r["diff"].items():
        L.append(f"| {k} | {', '.join(fmt(x) for x in s['per_seed'])} | {fmt(s['pooled'])} | [{fmt(s['pooled_ci'][0])}, {fmt(s['pooled_ci'][1])}] |")
    L += ["", f"Name: {r['name'] or 'NONE'} ({r['name_rule']})"]
    with open(os.path.join(a.out_dir, stem + ".md"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"lv2_read pair {stem}: name {r['name']}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("read")
    r.add_argument("--dump", nargs="+", required=True)
    r.add_argument("--arm", required=True)
    r.add_argument("--seeds", required=True)
    r.add_argument("--out-dir", required=True)
    r.add_argument("--ref-windows", default=None, help="LATENTS_CTX_<ARM>_R1.npz of the r-b read (k6)")
    r.add_argument("--B", type=int, default=B_DEFAULT)
    r.add_argument("--k5-parts", nargs="+", default=None, help="note ADD38_N2: one probe json per dump (ops/lv2_k5_probe.py); k5 is then read in parts")
    p = sub.add_parser("pair")
    p.add_argument("--a-dir", required=True)
    p.add_argument("--b-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--B", type=int, default=B_DEFAULT)
    a = ap.parse_args(argv)
    return cmd_read(a) if a.cmd == "read" else cmd_pair(a)


if __name__ == "__main__":
    sys.exit(main())
