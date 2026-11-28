r"""hierrep_read.py -- the read of PREREG_V8F_PROOF_ADD11 (HIERREP).  CPU, zero training, no model pass.  Tier T1-kin (kinematic rollout on the model's own actions,
logged frames): a diagnostic, never driving performance.  Evidence class MEASURED once run on real dumps.  Definitions: HIERREP_DEFS.md (R1..R17), written before this file.

  python hierrep_read.py read --t1-root DIR --runs-root DIR --out-dir DIR [--n-boot 2000] [--seeds 0,1,2,3,4,5] [--r1-result FILE] [--no-card-counts] [--expect-files 141]
  python hierrep_read.py r1   --dump DIR --ref r1_reference.json [--out FILE] [--tol 0.005] [--expect-files 141]      exit 0 PASS, 3 FAIL, 4 PENDING (no reference file)

Layout read: <t1-root>/<ARM>_s<S>/dump_<ARM>_s<S>_m10_causal_heads/ep*.npz ; <runs-root>/<ARM>_s<S>/summary.json ; ARM in HIER FLAT DAG FLATDAG.
Estimators: (A) two-level bootstrap seeds then episodes (B 2,000, rng 0) decides; (B) exact sign-flip over the seed differences; (C) t interval, decides nothing.
"""
import argparse
import glob
import itertools
import json
import math
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np  # noqa: E402

if os.environ.get("TANITEVAL_ROOT"):
    sys.path.insert(0, os.environ["TANITEVAL_ROOT"])
import torch  # noqa: E402
from taniteval import ci as CI  # noqa: E402
from taniteval import four_families as FF  # noqa: E402

DT = 0.2
ARMS = ("HIER", "FLAT", "DAG", "FLATDAG")
STAGES = {1: ("HIER", "FLAT"), 2: ("DAG", "FLATDAG")}
STAGE_NAME = {1: "Stage 1 (from scratch, 19,164 steps)", 2: "Stage 2 (own-state fine-tune, 3,000 steps) = PRIMARY protocol"}
EXPECT_STEPS = {"HIER": 19164, "FLAT": 19164, "DAG": 3000, "FLATDAG": 3000}
EXPECT_PARAMS = 7214495
CARD_COUNTS = {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}
THR = {"stop_mps": 0.5, "dv_mps": 2.0, "head_deg": 10.0}
KEEP_STRAIGHT = 3 * 3 + 2
N_SEEDS_CARD = 6
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}
FAM_ROWS = (("LONGITUDINAL", "speed_mae_mps"), ("LONGITUDINAL", "along_mae_m"), ("LATERAL", "cross_mae_m"), ("LATERAL", "heading_mae_deg"),
            ("LATERAL", "curvature_mae_1pm"), ("LATERAL", "yaw_rate_mae_degps"), ("TACTICAL", "lon_err_rate"), ("TACTICAL", "lat_err_rate"),
            ("TACTICAL", "joint_err_rate"), ("TACTICAL", "ade_tac_m"))
GAPS = ("LONGITUDINAL headway and time gap: GAP (R14a) no per-window lead block in the T1-kin dumps and none on the pod",
        "TACTICAL goal-setting (stop-goal decision): GAP (R14b) no per-arm goal label file",
        "STRATEGIC: PAUSED (PI item 25), never a pass, never omitted")


# ------------------------------------------------------------------------------------------------ geometry (HIERWHERE copy, HIER_DEFS section 2)
def kinematics(traj):
    p = np.asarray(traj, np.float64)
    p = np.concatenate([np.zeros_like(p[:, :1]), p], axis=1)
    d = p[:, 1:] - p[:, :-1]
    ds = np.linalg.norm(d, axis=-1)
    speed = ds / DT
    valid = ds > FF.MIN_DS_MPS * DT
    head = np.arctan2(d[..., 1], d[..., 0])
    dh = np.zeros(len(p))
    for i in range(len(p)):
        v = np.flatnonzero(valid[i])
        if len(v) >= 2:
            u = np.unwrap(head[i, v])
            dh[i] = u[-1] - u[0]
    return speed, np.degrees(dh)


def classify(speed, dhead_deg):
    below = speed < THR["stop_mps"]
    suffix_all = np.logical_and.accumulate(below[:, ::-1], axis=1)[:, ::-1]
    stop = suffix_all[:, 9:].any(axis=1)
    dv = speed[:, 29] - speed[:, 0]
    lon = np.full(len(speed), 3, np.int64)
    lon[dv >= THR["dv_mps"]] = 2
    lon[dv <= -THR["dv_mps"]] = 1
    lon[stop] = 0
    lat = np.full(len(speed), 2, np.int64)
    lat[dhead_deg >= THR["head_deg"]] = 0
    lat[dhead_deg <= -THR["head_deg"]] = 1
    return lon, lat


def comps(X, G):
    Xt, Gt = torch.as_tensor(np.asarray(X, np.float64)), torch.as_tensor(np.asarray(G, np.float64))
    P, Q = FF._seq_geometry(Xt, DT), FF._seq_geometry(Gt, DT)
    e = torch.linalg.norm(Xt - Gt, dim=-1)
    both, bp = P["valid"] & Q["valid"], P["pair_valid"] & Q["pair_valid"]
    dh = P["heading"] - Q["heading"]
    dh = ((dh + math.pi) % (2 * math.pi) - math.pi).abs()

    def mm(x, m):
        s, c = (x * m).sum(1), m.sum(1)
        return torch.where(c > 0, s / c.clamp_min(1), torch.full_like(s, float("nan")))
    out = {"ade6_m": e.mean(1), "ade_tac_m": e[:, 10:30].mean(1),
           "speed_mae_mps": (P["speed"] - Q["speed"]).abs().mean(1), "along_mae_m": (P["along"] - Q["along"]).abs().mean(1),
           "cross_mae_m": (P["cross"] - Q["cross"]).abs().mean(1),
           "heading_mae_deg": torch.rad2deg(mm(dh, both)), "yaw_rate_mae_degps": torch.rad2deg(mm((P["yaw_rate"] - Q["yaw_rate"]).abs(), bp)),
           "curvature_mae_1pm": mm((P["curvature"] - Q["curvature"]).abs(), bp)}
    return {k: v.numpy().astype(np.float64) for k, v in out.items()}


# ------------------------------------------------------------------------------------------------ loading and completeness (R1, R2, R3)
def load_dump(d, expect_files):
    fs = sorted(glob.glob(d + "/ep*.npz"))
    if len(fs) != expect_files:
        raise ValueError(f"{d}: {len(fs)} ep files, expected {expect_files}")
    out = {k: [] for k in ("g", "cl", "ws", "eid", "clip_index", "v0")}
    for f in fs:
        with np.load(f) as z:
            for k in out:
                out[k].append(np.asarray(z[k]))
    return {k: np.concatenate(v) for k, v in out.items()}


def dump_dir(t1_root, arm, s):
    return f"{t1_root}/{arm}_s{s}/dump_{arm}_s{s}_m10_causal_heads"


def arm_status(t1_root, runs_root, arm, s, expect_files):
    """-> (ok, reason, summary_or_None).  R2: dump complete AND summary done with the final step."""
    sm = f"{runs_root}/{arm}_s{s}/summary.json"
    if not os.path.isfile(sm):
        return False, f"no summary.json ({sm})", None
    try:
        with open(sm, encoding="utf-8") as f:
            S = json.load(f)
    except Exception as e:  # noqa: BLE001
        return False, f"summary unreadable: {e}", None
    if S.get("done") is not True or S.get("steps") != EXPECT_STEPS[arm]:
        return False, f"summary done={S.get('done')} steps={S.get('steps')} (expected {EXPECT_STEPS[arm]})", S
    n = len(glob.glob(dump_dir(t1_root, arm, s) + "/ep*.npz"))
    if n != expect_files:
        return False, f"dump has {n} ep files, expected {expect_files}", S
    return True, "complete", S


# ------------------------------------------------------------------------------------------------ the estimator engine (R6 to R10)
def episode_units(eid):
    uniq, inv = np.unique(np.asarray([str(x) for x in eid]), return_inverse=True)
    return uniq, inv


def seed_arrays(hier, flat, mask, inv, n_ep):
    """hier, flat: lists over seeds of per-window arrays.  -> sumE [S,E], cntE [S,E] of the paired window difference hier - flat over finite class windows."""
    sums, cnts = [], []
    for x, y in zip(hier, flat):
        d = np.asarray(x, np.float64) - np.asarray(y, np.float64)
        ok = np.isfinite(d) & mask
        sums.append(np.bincount(inv, weights=np.where(ok, d, 0.0), minlength=n_ep))
        cnts.append(np.bincount(inv, weights=ok.astype(np.float64), minlength=n_ep))
    return np.asarray(sums), np.asarray(cnts)


def boot_two_level(sumE, cntE, n_boot, seed=0):
    S, E = sumE.shape
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, S, size=(n_boot, S))
    counts = rng.multinomial(E, np.full(E, 1.0 / E), size=(n_boot, S)).astype(np.float64)
    num = np.einsum("bse,bse->bs", counts, sumE[idx])
    den = np.einsum("bse,bse->bs", counts, cntE[idx])
    with np.errstate(invalid="ignore", divide="ignore"):
        v = np.where(den > 0, num / den, np.nan)
    D = np.nanmean(v, axis=1)
    D = D[np.isfinite(D)]
    if len(D) == 0:
        return float("nan"), float("nan"), 0
    lo, hi = np.percentile(D, [2.5, 97.5])
    return float(lo), float(hi), int(len(D))


def sign_flip_p(d):
    d = np.asarray(d, np.float64)
    obs = abs(d.mean())
    ge = 0
    tot = 0
    for sg in itertools.product((-1.0, 1.0), repeat=len(d)):
        tot += 1
        ge += abs((np.asarray(sg) * d).mean()) >= obs - 1e-12
    return ge / tot, tot


def t_interval(d):
    n = len(d)
    if n < 2:
        return float("nan"), float("nan")
    m, sd = float(np.mean(d)), float(np.std(d, ddof=1))
    t = T975.get(n - 1, 1.96)
    h = t * sd / math.sqrt(n)
    return m - h, m + h


def label_of(d, lo, hi, n_req=N_SEEDS_CARD):
    """R10.  -> (label, flags)."""
    d = np.asarray(d, np.float64)
    inc = len(d) < n_req
    if inc or not np.all(np.isfinite(d)):
        return "OPEN", ["INCOMPLETE"] if inc else ["NON-FINITE"]
    if np.all(d < 0) and hi < 0:
        return "HIER", []
    if np.all(d > 0) and lo > 0:
        return "FLAT", []
    return "OPEN", []


def read_cell(sumE, cntE, n_boot, label_n=N_SEEDS_CARD):
    den = cntE.sum(1)
    d = np.where(den > 0, sumE.sum(1) / np.maximum(den, 1), np.nan)
    lo, hi, nfin = boot_two_level(sumE, cntE, n_boot)
    p, nassign = sign_flip_p(d) if np.all(np.isfinite(d)) else (float("nan"), 0)
    tlo, thi = t_interval(d)
    lab, flags = label_of(d, lo, hi, label_n)
    return {"d_s": [float(x) for x in d], "D": float(np.mean(d)), "A_lo": lo, "A_hi": hi, "A_excludes_0": bool(lo > 0 or hi < 0), "A_n_boot_finite": nfin,
            "B_signflip_p": p, "B_n_assignments": nassign, "C_t_lo": tlo, "C_t_hi": thi, "label": lab, "flags": flags,
            "n_windows_per_seed": [int(x) for x in den], "n_episodes": int((cntE.sum(0) > 0).sum()), "n_seeds": int(len(d))}


def outcome(crit_label, all_label):
    if crit_label == "HIER":
        return "HIER-DOMINATES" if all_label == "HIER" else "HIER-ON-CRITICAL"
    if crit_label == "FLAT":
        return "FLAT-AHEAD"
    return "NOT-ESTABLISHED"


def r4_swap_control(sumE, cntE, n_boot, n_draws=20):
    """R12.  Labels of the swapped-label critical cell, 20 draws."""
    labs = []
    for k in range(n_draws):
        rng = np.random.default_rng(1000 + k)
        sw = rng.random(sumE.shape[0]) < 0.5
        sg = np.where(sw, -1.0, 1.0)[:, None]
        c = read_cell(sumE * sg, cntE, n_boot)
        labs.append({"draw": k, "n_swapped": int(sw.sum()), "label": c["label"], "D": c["D"]})
    n_open = sum(x["label"] == "OPEN" for x in labs)
    return {"PASS": bool(n_open >= 18), "n_open": n_open, "n_draws": n_draws, "draws": labs}


# ------------------------------------------------------------------------------------------------ noise floor (R11)
def noise_floor(V, arm, seeds, masks, eid, n_boot):
    out = {}
    for cname, m in masks.items():
        diffs, sep = [], 0
        pairs = list(itertools.combinations(seeds, 2))
        for a, b in pairs:
            x, y = V[(arm, a)]["ade6_m"][m], V[(arm, b)]["ade6_m"][m]
            ok = np.isfinite(x) & np.isfinite(y)
            r = CI.paired_episode_cluster_bootstrap(x[ok], y[ok], np.asarray(eid)[m][ok], n_boot=n_boot, seed=0)
            diffs.append(abs(float(np.mean(x[ok]) - np.mean(y[ok]))))
            sep += bool(r["separated"])
        out[cname] = {"n_pairs": len(pairs), "median_abs_m": float(np.median(diffs)) if diffs else float("nan"), "n_separated": int(sep)}
    return out


# ------------------------------------------------------------------------------------------------ r1
def cmd_r1(a):
    res = {"control": "r1 harness", "tol_m": a.tol}
    if not os.path.isfile(a.ref):
        res.update({"status": "PENDING", "reason": f"reference file {a.ref} absent: recorded as PENDING, never as a pass"})
        rc = 4
    else:
        with open(a.ref, encoding="utf-8") as f:
            ref = json.load(f)
        D = load_dump(a.dump, a.expect_files)
        c = comps(D["cl"], D["g"])
        ade = float(np.nanmean(c["ade6_m"]))
        dev = abs(ade - float(ref["ade6_all_windows_m"]))
        ok = bool(dev <= a.tol and int(ref.get("n_windows", len(D["g"]))) == len(D["g"]))
        res.update({"status": "PASS" if ok else "FAIL", "pod_ade6_all_windows_m": ade, "ref_ade6_all_windows_m": float(ref["ade6_all_windows_m"]), "abs_dev_m": dev,
                    "n_windows_pod": int(len(D["g"])), "n_windows_ref": ref.get("n_windows")})
        rc = 0 if ok else 3
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1)
    print("r1", json.dumps(res))
    return rc


# ------------------------------------------------------------------------------------------------ read
def fmt(c, nd=4):
    return f"{c['D']:+.{nd}f} [{c['A_lo']:+.{nd}f}, {c['A_hi']:+.{nd}f}]"


def cmd_read(a):
    seeds_all = [int(x) for x in a.seeds.split(",")]
    res = {"card": "PREREG_V8F_PROOF_ADD11.md sha256 94647519b965e5d0ac5b6decc0475719ed16a57757decd6a1994ebb25ee29d1c", "tier": "T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance)",
           "evidence_class": "MEASURED (ours, T1-kin dumps)",
           "estimator": f"(A) two-level bootstrap seeds then episodes, B {a.n_boot}, rng 0 (decides); (B) exact sign-flip over seed differences; (C) t interval (decides nothing)",
           "STRATEGIC": "PAUSED (PI item 25)", "gaps": list(GAPS), "controls": {}, "completeness": {}, "stages": {}}
    # ---- r5 completeness
    S_ok, V, sums = [], {}, {}
    for s in seeds_all:
        reasons = {}
        for arm in ARMS:
            ok, why, summ = arm_status(a.t1_root, a.runs_root, arm, s, a.expect_files)
            sums[(arm, s)] = summ
            if not ok:
                reasons[arm] = why
        res["completeness"][f"s{s}"] = {"complete": not reasons, "reasons": reasons}
        if not reasons:
            S_ok.append(s)
    res["seeds_read"] = S_ok
    res["seeds_excluded"] = [s for s in seeds_all if s not in S_ok]
    res["controls"]["r5_completeness"] = {"PASS": True, "note": "an arm without its final step is not read; a seed enters only with all four arms", "seeds_read": S_ok, "seeds_excluded": res["seeds_excluded"]}
    if not S_ok:
        res["controls"]["ALL_PASS"] = False
        _write(res, a.out_dir)
        print("no complete seed: nothing to read")
        return 2
    # ---- r3 parameter count
    bad = []
    for s in S_ok:
        for arm in ARMS:
            n = (sums[(arm, s)].get("levers") or {}).get("n_params_trainable")
            if n != EXPECT_PARAMS:
                bad.append(f"{arm}_s{s}: {n}")
    res["controls"]["r3_param_count"] = {"PASS": not bad, "expected": EXPECT_PARAMS, "mismatch": bad}
    # ---- load
    ref = None
    for s in S_ok:
        for arm in ARMS:
            D = load_dump(dump_dir(a.t1_root, arm, s), a.expect_files)
            if ref is None:
                ref = D
            for k in ("g", "ws", "eid", "clip_index", "v0"):
                if not np.array_equal(D[k], ref[k]):
                    print(f"windows differ between arms: {arm}_s{s} key {k}")
                    return 3
            c = comps(D["cl"], D["g"])
            lon, lat = classify(*kinematics(D["cl"]))
            c["_lon"], c["_lat"] = lon, lat
            V[(arm, s)] = c
    eid = ref["eid"]
    uniq, inv = episode_units(eid)
    n_ep = len(uniq)
    glon, glat = classify(*kinematics(ref["g"]))
    joint_g = glon * 3 + glat
    crit = joint_g != KEEP_STRAIGHT
    masks = {"critical": crit, "bulk": ~crit, "all": np.ones(len(crit), bool)}
    for arm in ARMS:
        for s in S_ok:
            c = V[(arm, s)]
            c["lon_err_rate"] = (c["_lon"] != glon).astype(float)
            c["lat_err_rate"] = (c["_lat"] != glat).astype(float)
            c["joint_err_rate"] = ((c["_lon"] * 3 + c["_lat"]) != joint_g).astype(float)
    n_c = {"n_windows": int(len(crit)), "n_episodes": int(n_ep), "n_crit": int(crit.sum()), "n_bulk": int((~crit).sum())}
    cc_ok = all(n_c[k] == v for k, v in CARD_COUNTS.items())
    res["window_counts"] = n_c
    res["controls"]["window_counts"] = {"PASS": bool(cc_ok or a.no_card_counts), "card": CARD_COUNTS, "read": n_c, "checked": not a.no_card_counts}
    # ---- cells
    keyset = {"ade6_m": ("critical", "bulk", "all"), "ade_tac_m": ("critical", "bulk", "all")}
    for stg, (ha, fa) in STAGES.items():
        st = {"name": STAGE_NAME[stg], "arms": [ha, fa], "ade6": {}, "tactical_band_ade": {}, "families": {}, "noise_floor": {}}
        for key, dest in (("ade6_m", "ade6"), ("ade_tac_m", "tactical_band_ade")):
            for cname in keyset[key]:
                sE, cE = seed_arrays([V[(ha, s)][key] for s in S_ok], [V[(fa, s)][key] for s in S_ok], masks[cname], inv, n_ep)
                st[dest][cname] = read_cell(sE, cE, a.n_boot)
                if stg == 2 and key == "ade6_m" and cname == "critical":
                    res["controls"]["r4_label_swap"] = r4_swap_control(sE, cE, a.n_boot)
        for fam, key in FAM_ROWS:
            sE, cE = seed_arrays([V[(ha, s)][key] for s in S_ok], [V[(fa, s)][key] for s in S_ok], masks["all"], inv, n_ep)
            st["families"][f"{fam}:{key}"] = read_cell(sE, cE, a.n_boot)
        st["outcome"] = outcome(st["ade6"]["critical"]["label"], st["ade6"]["all"]["label"])
        for arm in (ha, fa):
            st["noise_floor"][arm] = noise_floor(V, arm, S_ok, masks, eid, a.n_boot)
        res["stages"][f"stage{stg}"] = st
    res["PRIMARY_OUTCOME"] = res["stages"]["stage2"]["outcome"]
    res["SECONDARY_OUTCOME_STAGE1"] = res["stages"]["stage1"]["outcome"]
    if a.r1_result and os.path.isfile(a.r1_result):
        with open(a.r1_result, encoding="utf-8") as f:
            res["controls"]["r1_harness"] = json.load(f)
    else:
        res["controls"]["r1_harness"] = {"status": "PENDING", "reason": "no r1 result file given"}
    res["controls"]["r2_gate"] = {"note": "read from the chain's token files and STAGEB_STATUS.json, not from the dumps", "status": "see chain log"}
    flags = [v.get("PASS") for k, v in res["controls"].items() if isinstance(v, dict) and "PASS" in v]
    res["controls"]["ALL_PASS"] = bool(all(f is not False for f in flags))
    _write(res, a.out_dir)
    print(f"PRIMARY {res['PRIMARY_OUTCOME']} | stage1 {res['SECONDARY_OUTCOME_STAGE1']} | seeds {S_ok} | controls {'PASS' if res['controls']['ALL_PASS'] else 'FAILED'}")
    return 0 if res["controls"]["ALL_PASS"] else 2


def _write(res, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with open(out_dir + "/RESULT_HIERREP.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    with open(out_dir + "/VERDICT_HIERREP.md", "w", encoding="utf-8") as f:
        f.write(verdict(res))


def verdict(res):
    L = []
    P = L.append
    C = res["controls"]
    valid = C.get("ALL_PASS")
    P("# VERDICT_HIERREP (PREREG_V8F_PROOF_ADD11; MEASURED, tier T1-kin, a diagnostic, never driving performance)")
    P("")
    if not valid:
        P("**READ NOT VALID: a control failed or no seed was complete (see Controls).** The tables below are printed for the record only.")
        P("")
    P(f"Estimator: {res['estimator']}. Seeds read: {res.get('seeds_read')} (n = {len(res.get('seeds_read', []))}); excluded: {res.get('seeds_excluded')}. Windows: {res.get('window_counts')}.")
    P("Difference = hierarchical arm minus flat arm, ADE6 in metres, lower is better; negative = hierarchy better. Each cell: D [95 % interval A]; d_s are the per-seed differences.")
    P("")
    if "PRIMARY_OUTCOME" in res:
        P(f"## Outcome (primary protocol, Stage 2 DAG vs FLATDAG): **{res['PRIMARY_OUTCOME']}**; Stage 1 (secondary): **{res['SECONDARY_OUTCOME_STAGE1']}**")
        P("")
        for sk, st in res["stages"].items():
            P(f"## {st['name']}: {st['arms'][0]} - {st['arms'][1]}  (outcome {st['outcome']})")
            P("| cell | class | n windows per seed (ep) | seeds | D [A] | d_s | sign-flip p (B) | t interval (C) | label |")
            P("|---|---|---|---|---|---|---|---|---|")
            for dest, nm in (("ade6", "ADE6"), ("tactical_band_ade", "tactical-band ADE 2.2-6.0 s")):
                for cname, c in st[dest].items():
                    P(f"| {nm} | {cname} | {c['n_windows_per_seed'][0] if c['n_windows_per_seed'] else 0} ({c['n_episodes']}) | {c['n_seeds']} | {fmt(c)} | {', '.join(f'{x:+.3f}' for x in c['d_s'])} | {c['B_signflip_p']:.5f} | [{c['C_t_lo']:+.4f}, {c['C_t_hi']:+.4f}] | {c['label']}{' ' + ','.join(c['flags']) if c['flags'] else ''} |")
            P("")
            P("Noise floor beside the differences (same arm type, 15 seed pairs, ADE6): median |difference| m, pairs separated by the paired episode bootstrap")
            P("| arm | class | median abs m | separated / pairs |")
            P("|---|---|---|---|")
            for arm, nf in st["noise_floor"].items():
                for cname, v in nf.items():
                    P(f"| {arm} | {cname} | {v['median_abs_m']:.4f} | {v['n_separated']} / {v['n_pairs']} |")
            P("")
            P("Four families on all windows (hierarchy minus flat; speed, along-track, cross-track, heading, curvature, yaw rate, mode error rates, tactical-band ADE):")
            P("| row | n windows (ep) | D [A] | sign-flip p | label |")
            P("|---|---|---|---|---|")
            for k, c in st["families"].items():
                P(f"| {k} | {c['n_windows_per_seed'][0] if c['n_windows_per_seed'] else 0} ({c['n_episodes']}) | {fmt(c, 5)} | {c['B_signflip_p']:.5f} | {c['label']} |")
            P("")
    P("## Named gaps")
    for g in res["gaps"]:
        P(f"* {g}")
    P("")
    P("## Controls")
    for k, v in C.items():
        if k == "ALL_PASS":
            continue
        if k == "r4_label_swap":
            P(f"* r4 label swap (20 draws, Stage 2 critical, must read OPEN in >= 18): {'PASS' if v['PASS'] else 'FAIL'}; OPEN in {v['n_open']} of {v['n_draws']}")
        elif k == "r1_harness":
            P(f"* r1 harness: {v.get('status')} {json.dumps({x: v[x] for x in v if x != 'status'})}")
        else:
            P(f"* {k}: {json.dumps(v, default=str)[:700]}")
    P("")
    P("## What it cannot show")
    P("Six seeds, one corpus (4,512 training clips), one evaluation set of 141 episodes, a kinematic rollout on logged frames. Nothing about another corpus, a re-perception simulator or a larger model. The intervals answer whether another draw of seeds and episodes would say this; inference-seed variance is not sampled (the planner is deterministic).")
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("read")
    r.add_argument("--t1-root", required=True)
    r.add_argument("--runs-root", required=True)
    r.add_argument("--out-dir", required=True)
    r.add_argument("--n-boot", type=int, default=2000)
    r.add_argument("--seeds", default="0,1,2,3,4,5")
    r.add_argument("--r1-result", default="")
    r.add_argument("--no-card-counts", action="store_true")
    r.add_argument("--expect-files", type=int, default=141)
    q = sub.add_parser("r1")
    q.add_argument("--dump", required=True)
    q.add_argument("--ref", required=True)
    q.add_argument("--out", default="")
    q.add_argument("--tol", type=float, default=0.005)
    q.add_argument("--expect-files", type=int, default=141)
    a = ap.parse_args(argv)
    try:
        return cmd_read(a) if a.cmd == "read" else cmd_r1(a)
    except (ValueError, OSError, KeyError) as e:    # an input error is exit 3, never a pass
        print(f"INPUT ERROR: {type(e).__name__}: {e}")
        return 3


if __name__ == "__main__":
    sys.exit(main())
