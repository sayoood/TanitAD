#!/usr/bin/env python3
"""RL-S1 priority 3: the RL-val LOOKUP evaluator.

A scorer's per-proposal scores on RL-val sets + the sets' PDM labels (+ the rl_terms side files) -> the TRUE PDMS of the scorer's pick, its six
sub-scores, the reward R and the drivable excess D at the pick, and the slow-pick telemetry of review 3.4 (g3).  No model runs here: the labels
are computed once (frozen fan), so reading any scorer is a table lookup.  Two scorers are PAIRED on identical sets with proposal-for-proposal
identical labels.

    python eval/rl_val_eval.py --scores C0=c0.npz R0=r0.npz --base C0 --pdm <pdm_*.jsonl dir> --terms <rl_terms_*.jsonl dir> \\
        --sets <onpolicy_*.jsonl dir> --rl-val-scenes rl_val_scenes.tsv --rule navsim_v1_lane --out result.json

INPUT FORMATS (scorer_finetune.py has NO score dump today -- its evaluate() computes picks in-process and writes only aggregates; see HANDOFF_RL_S1):
  scores  .npz  arrays log, token (str), step, rank, ckpt_step (int), scores float [N, M, C] (C = 6 or 7), scalar `space` = "logit" (default; the
                trainer's raw output, sigmoid is applied here) or "prob".   write_scores_npz() writes it.   .jsonl: one {"kind":"rl_scores",
                "key":[log,token,step,rank],"ckpt_step":c,"scores":[[C]*M],"space":"logit"} per set.
  pdm     <dir>/pdm_*.jsonl     onpolicy_relabel_pdm.py's {"kind":"pdm_targets","key","ckpt_step","nc","dac","ep","ttc","c","ddc","pdms"}.
  terms   <dir>/rl_terms_*.jsonl  rl_terms_label.py's {"kind":"rl_terms","key","ckpt_step","d","k","lc_executed","label_fires","dither"}.
  sets    <dir>/onpolicy_*.jsonl  only for the path-length telemetry: the frozen-fan dump's lines (traj [M][T][2]); read for matched keys only.
A record counts only for the set it was computed on (same key AND ckpt_step, OnPolicyBank's rule): a mismatch is COUNTED as stale and the set is
dropped from EVERY arm, never read from the wrong labels.  Ties in the selection score resolve to the lowest proposal index.

ESTIMATOR: paired log-cluster bootstrap (cluster unit log_name), 10,000 resamples, 95 %, seed 20260927 = proxy_eval.boot, the estimator of
eval/stage3_full.py.  It answers 'another draw of LOGS' only (not another training run, not another inference run).
TIER: a lookup of the NAVSIM v1.1 PDMS of an open-loop plan on navtrain RL-val sets (frozen fan): hyper-parameter / early-stop use only, never an endpoint.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "refe"))
import proxy_eval as PE  # noqa: E402  -- the SAME estimator stage 3 uses
import rl_reward  # noqa: E402

TOOL = "rl_val_eval/1"
N_BOOT, SEED = 10000, 20260927
SWAP_TOL = 1e-12
RULES = ("navsim_v1", "navsim_v1_lane")
SUBS = ("NC", "DAC", "EP", "TTC", "C", "DDC")
# review 3.4 (g3)/(g5), K4 -- baselines are the review's, measured on navtest (10,070 tokens), not on RL-val
G3_PICK_FAN_MEAN_MIN = 0.914 - 0.02
G3_SHORTEST_SHARE_MAX = 0.55
G5_SUBSCORE_FALL_MAX = 0.5
K4_PDMS_FALL_MAX = 0.5


# ----------------------------------------------------------------------------------------------------------------- selection
def sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(x, np.float64)))                 # stable for large |x|


def select(scores, rule, space="logit"):
    """-> (pick index, selection score [M]).  scores [M, C] with C >= 6 (7 = the teacher-lane output).  navsim_v1 = NC*DAC*(5EP+5TTC+2C)/12;
    navsim_v1_lane = that x p_lane (planner.py rule of the same name)."""
    if rule not in RULES:
        raise ValueError(f"rule must be one of {RULES}")
    s = np.asarray(scores, np.float64)
    if s.ndim != 2 or s.shape[-1] < 6:
        raise ValueError(f"scores need shape [M, C>=6], got {s.shape}")
    p = sigmoid(s) if space == "logit" else s
    sel = p[:, 0] * p[:, 1] * (5.0 * p[:, 2] + 5.0 * p[:, 3] + 2.0 * p[:, 4]) / 12.0
    if rule == "navsim_v1_lane":
        if s.shape[-1] < 7:
            raise ValueError("rule navsim_v1_lane needs a 7-output scorer")
        sel = sel * p[:, 6]
    return int(np.argmax(sel)), sel


def path_len(traj):
    """[M, T, 2(+)] -> path length through the origin, per proposal (pick_length_telemetry.plen)."""
    xy = np.asarray(traj, np.float64)[..., :2]
    return np.linalg.norm(xy[:, 0, :], axis=-1) + np.linalg.norm(np.diff(xy, axis=-2), axis=-1).sum(-1)


def pick_telemetry(L, i):
    """One set's slow-pick statistics (review 3.4): ratio of the pick's path to the fan mean, its length rank (0 = longest), 5 bins of ceil(M/5)
    ranks (13 for M = 64), shorter-than-fan-median flag."""
    L = np.asarray(L, np.float64)
    M = len(L)
    rk = int((-L).argsort().argsort()[i])
    width = int(math.ceil(M / 5.0))
    return {"ratio": float(L[i] / L.mean()), "rank": rk, "bin": int(min(rk // width, 4)), "shorter": bool(L[i] < np.median(L)),
            "pick_m": float(L[i]), "fan_m": float(L.mean())}


def telemetry_summary(rows):
    if not rows:
        return None
    b = np.asarray([r["bin"] for r in rows])
    return {"n": len(rows), "pick_over_fan_mean_path": float(np.mean([r["ratio"] for r in rows])),
            "median_rank_by_length_0_longest": float(np.median([r["rank"] for r in rows])),
            "share_pick_shorter_than_fan_median": float(np.mean([r["shorter"] for r in rows])),
            "pick_length_quintile_shares_shortest_first": [float((b == k).mean()) for k in (4, 3, 2, 1, 0)],
            "shortest_bin_share": float((b == 4).mean()), "longest_bin_share": float((b == 0).mean()),
            "mean_pick_path_m": float(np.mean([r["pick_m"] for r in rows])), "mean_fan_path_m": float(np.mean([r["fan_m"] for r in rows]))}


# ----------------------------------------------------------------------------------------------------------------- I/O
def key_of(k):
    k = tuple(k)
    if len(k) != 4:
        raise ValueError(f"a key is [log, token, step, rank], got {k!r}")
    return (str(k[0]), str(k[1]), int(k[2]), int(k[3]))


def kstr(k):
    return f"{k[0]}|{k[1]}|{k[2]}|{k[3]}"


def write_scores_npz(path, keys, ckpt_steps, scores, space="logit"):
    """keys: [(log, token, step, rank)], ckpt_steps: [int], scores: [N, M, C] -> the evaluator's .npz format."""
    ks = [key_of(k) for k in keys]
    np.savez(path, log=np.asarray([k[0] for k in ks]), token=np.asarray([k[1] for k in ks]), step=np.asarray([k[2] for k in ks], np.int64),
             rank=np.asarray([k[3] for k in ks], np.int64), ckpt_step=np.asarray(ckpt_steps, np.int64), scores=np.asarray(scores, np.float64),
             space=np.asarray(space))


def load_scores(path):
    """-> {key: {"ckpt": int, "scores": [M, C] float64, "space": str}}.  A duplicated key is an error."""
    out = {}

    def put(k, ck, sc, sp):
        if k in out:
            raise ValueError(f"duplicate scores for set {kstr(k)} in {path}")
        if sp not in ("logit", "prob"):
            raise ValueError(f"space must be logit or prob, got {sp!r}")
        out[k] = {"ckpt": int(ck), "scores": np.asarray(sc, np.float64), "space": sp}

    if path.endswith(".npz"):
        z = np.load(path, allow_pickle=False)
        sp = str(z["space"]) if "space" in z.files else "logit"
        for i in range(len(z["token"])):
            put(key_of((z["log"][i], z["token"][i], z["step"][i], z["rank"][i])), z["ckpt_step"][i], z["scores"][i], sp)
    else:
        with open(path, encoding="utf-8") as fh:
            for ln in fh:
                d = json.loads(ln)
                if d.get("kind") == "rl_scores":
                    put(key_of(d["key"]), d["ckpt_step"], d["scores"], d.get("space", "logit"))
    return out


def _jsonl(dir_, pat):
    for fp in sorted(glob.glob(os.path.join(dir_, pat))):
        with open(fp, encoding="utf-8") as fh:
            for ln in fh:
                try:
                    d = json.loads(ln)
                except json.JSONDecodeError:
                    continue
                yield d


def load_pdm(dir_):
    """-> ({key: {"ckpt", "tg" [M,6]}}, stats).  tg is rl_reward.tg_from_pdm_record's stacking; a record whose own pdms column disagrees is dropped."""
    out, st = {}, {"lines": 0, "bad_pdms": 0, "duplicates": 0}
    for d in _jsonl(dir_, "pdm_*.jsonl"):
        if d.get("kind") != "pdm_targets":
            continue
        st["lines"] += 1
        try:
            k = key_of(d["key"])
            tg = rl_reward.tg_from_pdm_record(d)
            if "pdms" in d:
                rl_reward.check_pdms(tg, d["pdms"])
        except (KeyError, ValueError, TypeError):
            st["bad_pdms"] += 1
            continue
        if k in out:
            st["duplicates"] += 1
        out[k] = {"ckpt": int(d["ckpt_step"]), "tg": tg}
    return out, st


def load_terms(dir_):
    out, st = {}, {"lines": 0, "bad": 0}
    for d in _jsonl(dir_, "rl_terms_*.jsonl"):
        if d.get("kind") != "rl_terms":
            continue
        st["lines"] += 1
        try:
            k = key_of(d["key"])
            rec = {"ckpt": int(d["ckpt_step"]), "d": np.asarray(d["d"], np.float64), "k": np.asarray(d["k"], np.float64) if "k" in d else None,
                   "lc": np.asarray(d["lc_executed"], bool) if "lc_executed" in d else None,
                   "fires": bool(d["label_fires"]) if "label_fires" in d else None,
                   "dither": np.asarray(d["dither"], np.float64) if "dither" in d else None}
        except (KeyError, ValueError, TypeError):
            st["bad"] += 1
            continue
        out[k] = rec
    return out, st


def load_paths(dir_, keys, ckpts):
    """Path lengths [M] for the wanted keys, from the dump's set lines (head parse first, json only for a wanted key)."""
    import select_bank_subset as SB
    want, out = set(keys), {}
    for fp in sorted(glob.glob(os.path.join(dir_, "onpolicy_*.jsonl"))):
        with open(fp, encoding="utf-8") as fh:
            for ln in fh:
                f = SB.head_fields(ln[:2000] + ln[-400:])
                if "log_name" not in f:
                    continue
                k = (f["log_name"], f.get("token", ""), int(f.get("step", 0)), int(f.get("rank", 0)))
                if k not in want or int(f.get("ckpt_step", -1)) != ckpts[k]:
                    continue
                out[k] = path_len(json.loads(ln)["traj"])
    return out


def read_scenes(path):
    """RL-val scenes TSV (token, log) from rl_partition.py -> {token: log}."""
    out = {}
    with open(path, encoding="utf-8") as fh:
        head = fh.readline().rstrip("\n").split("\t")
        it, il = head.index("token"), head.index("log")
        for ln in fh:
            f = ln.rstrip("\n").split("\t")
            out[f[it]] = f[il]
    return out


# ----------------------------------------------------------------------------------------------------------------- one scorer on one set
def eval_set(score_rec, lk, rule, cfg):
    """The lookup for one set.  lk: {"tg" [M,6], "terms" or None, "L" [M] or None}."""
    tg = lk["tg"]
    s = rl_reward.split_components(tg)
    pdms = rl_reward.pdms_v1(s["nc"], s["dac"], s["ep"], s["ttc"], s["c"])
    i, _sel = select(score_rec["scores"], rule, score_rec["space"])
    t = lk["terms"]
    d = k = lane = None
    if t is not None:
        d = t["d"]
        k = t["k"]
        if t["lc"] is not None:
            lane = {"lc_executed": t["lc"], "label_fires": t["fires"], "dither": t["dither"]}
    res = rl_reward.reward(tg, cfg, d=d, k=k, lane=lane)
    out = {"PDMS": float(pdms[i]), "NC": float(s["nc"][i]), "DAC": float(s["dac"][i]), "EP": float(s["ep"][i]), "TTC": float(s["ttc"][i]),
           "C": float(s["c"][i]), "DDC": float(s["ddc"][i]), "R": float(res["R"][i]), "R_fan_mean": float(res["R"].mean()),
           "zero": float(pdms[i] == 0.0), "rand": float(pdms.mean()), "best": float(pdms.max()), "pick": i,
           "safe": float(res["safe"][i])}
    if d is not None:
        out["D"] = float(np.asarray(d)[i])
    if lk["L"] is not None:
        out["tele"] = pick_telemetry(lk["L"], i)
    return out


def summarise(per):
    """per: {key: eval_set output} -> the arm's aggregate (PDMS family x100)."""
    ks = sorted(per)
    m = lambda f: float(np.mean([per[k][f] for k in ks]))                            # noqa: E731
    pk, rd, bs = m("PDMS"), m("rand"), m("best")
    out = {"n_sets": len(ks), "n_logs": len({k[0] for k in ks}), "PDMS_x100": 100 * pk, "subscores_x100": {n: 100 * m(n) for n in SUBS},
           "fan_random_x100": 100 * rd, "fan_best_x100": 100 * bs, "skill": (pk - rd) / max(bs - rd, 1e-9), "zero_picks": int(sum(per[k]["zero"] for k in ks)),
           "unsafe_picks": int(len(ks) - sum(per[k]["safe"] for k in ks)), "R_pick_x100": 100 * m("R"), "R_fan_mean_x100": 100 * m("R_fan_mean")}
    if all("D" in per[k] for k in ks):
        out["D_at_pick_mean"] = m("D")
    rows = [per[k]["tele"] for k in ks if "tele" in per[k]]
    out["slow_pick_telemetry"] = telemetry_summary(rows) if len(rows) == len(ks) else None
    if out["slow_pick_telemetry"] is not None:
        t = out["slow_pick_telemetry"]
        out["g3"] = {"pick_over_fan_mean_path": t["pick_over_fan_mean_path"], "min_allowed": G3_PICK_FAN_MEAN_MIN, "pick_ok": t["pick_over_fan_mean_path"] >= G3_PICK_FAN_MEAN_MIN,
                     "shortest_bin_share": t["shortest_bin_share"], "max_allowed": G3_SHORTEST_SHARE_MAX, "share_ok": t["shortest_bin_share"] <= G3_SHORTEST_SHARE_MAX,
                     "note": "bounds are the review's, derived from a navtest baseline (0.914 pick/fan-mean, shortest-bin 0.506); applied to RL-val here"}
    return out


# ----------------------------------------------------------------------------------------------------------------- pairing
QTY = SUBS + ("PDMS", "R")


def read_one(arm, base, ks, key, n=N_BOOT, seed=SEED):
    scale = 100.0
    d = {kstr(k): scale * (arm[k][key] - base[k][key]) for k in ks}
    tl = {kstr(k): k[0] for k in ks}
    mu, lo, hi = PE.boot(d, tl, n=n, seed=seed)
    v = list(d.values())
    return {"mean": mu, "ci95": [lo, hi], "arm_mean_x100": scale * float(np.mean([arm[k][key] for k in ks])),
            "base_mean_x100": scale * float(np.mean([base[k][key] for k in ks])), "better": int(sum(x > 0 for x in v)), "worse": int(sum(x < 0 for x in v)),
            "tied": int(sum(x == 0 for x in v)), "n": len(ks), "n_logs": len({k[0] for k in ks})}


def read_pair(arm, base, n=N_BOOT, seed=SEED):
    """arm minus base on the sets both have, PDMS + six sub-scores + R, one estimator.  adverse <=> the interval lies entirely below 0."""
    ks = sorted(set(arm) & set(base))
    res = {q: read_one(arm, base, ks, q, n, seed) for q in QTY}
    for q in res:
        res[q]["adverse"] = bool(res[q]["ci95"][1] < 0)
    if all("D" in arm[k] and "D" in base[k] for k in ks):
        res["D"] = read_one(arm, base, ks, "D", n, seed)
    return {"n_sets": len(ks), "primary": res["PDMS"], "quantities": res,
            "adverse": sorted(q for q in res if res[q].get("adverse")),
            "g5": {"NC_fall_ok": res["NC"]["mean"] >= -G5_SUBSCORE_FALL_MAX, "TTC_fall_ok": res["TTC"]["mean"] >= -G5_SUBSCORE_FALL_MAX,
                   "bound": G5_SUBSCORE_FALL_MAX, "note": "point deltas, x100 units; review g5 'NC and TTC may not fall by > 0.5 each versus control'"},
            "K4": {"pdms_delta": res["PDMS"]["mean"], "breach": bool(res["PDMS"]["mean"] < -K4_PDMS_FALL_MAX), "bound": K4_PDMS_FALL_MAX,
                   "note": "K4 stops an arm on TWO consecutive checks; one breach here is a flag, not the stop"}}


def check_selfpair(rd):
    p = rd["primary"]
    zeros = all(v["mean"] == 0.0 and v["ci95"] == [0.0, 0.0] for v in rd["quantities"].values())
    ok = (p["mean"] == 0.0 and p["ci95"] == [0.0, 0.0] and p["better"] == 0 and p["worse"] == 0 and p["tied"] == p["n"] and zeros)
    return {"ok": bool(ok), "mean": p["mean"], "ci95": p["ci95"], "all_quantities_exactly_zero": bool(zeros)}


def check_swap(fwd, rev):
    """arm<->base swapped: the sign flips; the point estimate bit-exact, CI ends to SWAP_TOL (np.percentile is not bitwise antisymmetric)."""
    per = {}
    for q in fwd["quantities"]:
        f, r = fwd["quantities"][q], rev["quantities"][q]
        d_lo, d_hi = abs(r["ci95"][0] + f["ci95"][1]), abs(r["ci95"][1] + f["ci95"][0])
        per[q] = bool(r["mean"] == -f["mean"] and max(d_lo, d_hi) <= SWAP_TOL and r["better"] == f["worse"] and r["worse"] == f["better"])
    return {"ok": all(per.values()), "per_quantity": per, "tolerance": SWAP_TOL}


# ----------------------------------------------------------------------------------------------------------------- the run
def run(arms, pdm, terms, paths, rule, cfg, scenes=None, require_full=False, base=None, n_boot=N_BOOT, seed=SEED):
    """arms: {name: load_scores dict}.  Returns the result dict (raises on a violated input contract)."""
    elig = {"scored_sets_per_arm": {n: len(a) for n, a in arms.items()}}
    need_terms = cfg.lam_d != 0.0 or cfg.use_k or cfg.use_l
    common = set.intersection(*[set(a) for a in arms.values()])
    elig["sets_scored_by_every_arm"] = len(common)
    foreign = []
    if scenes is not None:
        foreign = sorted(k for k in common if scenes.get(k[1]) != k[0])
        if foreign:
            raise ValueError(f"{len(foreign)} scored sets are NOT RL-val scenes (first: {kstr(foreign[0])}): refusing to read a train scene as validation")
    stale = missing_pdm = missing_terms = 0
    use, tt = [], {}
    for k in sorted(common):
        cks = {a[k]["ckpt"] for a in arms.values()}
        p = pdm.get(k)
        if p is None:
            missing_pdm += 1
            continue
        if cks != {p["ckpt"]}:
            stale += 1
            continue
        t = terms.get(k) if terms is not None else None
        t_ok = t is not None and t["ckpt"] == p["ckpt"]
        if need_terms and not t_ok:
            if t is None:
                missing_terms += 1
            else:
                stale += 1
            continue
        tt[k] = t if t_ok else None
        use.append(k)
    elig.update({"missing_pdm": missing_pdm, "stale_or_mismatched_ckpt": stale, "missing_terms": missing_terms, "sets_used": len(use)})
    if scenes is not None:
        covered = {k[1] for k in use}
        elig["rl_val_scenes"] = len(scenes)
        elig["rl_val_scenes_with_a_used_set"] = len(covered & set(scenes))
        elig["rl_val_scene_coverage"] = len(covered & set(scenes)) / max(len(scenes), 1)
        elig["rl_val_logs"] = len(set(scenes.values()))
        if require_full and elig["rl_val_scenes_with_a_used_set"] < len(scenes):
            raise ValueError(f"--require-full: {len(scenes) - elig['rl_val_scenes_with_a_used_set']} RL-val scenes have no usable set ({elig})")
    if not use:
        raise ValueError(f"no usable set: {elig}")
    if paths is not None:
        elig["path_lengths_missing"] = sum(1 for k in use if k not in paths)          # any gap -> the telemetry row is None (counted here, never silently averaged over fewer sets)
    lks = {}
    for k in use:
        lks[k] = {"tg": pdm[k]["tg"], "terms": tt[k], "L": (paths.get(k) if paths is not None else None)}
    per = {n: {k: eval_set(a[k], lks[k], rule, cfg) for k in use} for n, a in arms.items()}
    res = {"tool": TOOL, "rule": rule, "reward_cfg": {"lam_d": cfg.lam_d, "w_prog": cfg.w_prog, "guard": cfg.guard, "use_k": cfg.use_k, "use_l": cfg.use_l},
           "eligibility": elig, "arms": {n: summarise(p) for n, p in per.items()},
           "estimator": f"paired log-cluster bootstrap (cluster unit log_name), {n_boot} resamples, 95 %, seed {seed}; proxy_eval.boot, the estimator of eval/stage3_full.py. "
                        "It answers 'another draw of LOGS' only; training variance (one seed per arm) and inference variance are NOT measured here.",
           "tier": "lookup of the true NAVSIM v1.1 PDMS of the scorer's pick on RL-val sets of the frozen fan (open-loop plan, pseudo-simulation); selection of hyper-parameters / early stops only, never an endpoint",
           "not_computed": ["four metric families (read on the full navtest endpoint)", "g4 longitudinal bias rows (need plan-vs-reference traces, not in the lookup)"],
           "paths_loaded": paths is not None}
    if len(arms) > 1:
        b = base if base is not None else sorted(arms)[0]
        res["base"] = b
        res["paired_vs_base"] = {n: read_pair(per[n], per[b], n_boot, seed) for n in sorted(arms) if n != b}
        res["selfcheck"] = {"selfpair": check_selfpair(read_pair(per[b], per[b], n_boot, seed))}
        n0 = sorted(n for n in arms if n != b)[0]
        res["selfcheck"]["swap"] = check_swap(read_pair(per[n0], per[b], n_boot, seed), read_pair(per[b], per[n0], n_boot, seed))
    else:
        n = next(iter(arms))
        res["selfcheck"] = {"selfpair": check_selfpair(read_pair(per[n], per[n], n_boot, seed))}
    return res


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", nargs="+", required=True, help="name=path (.npz or .jsonl); several = paired")
    ap.add_argument("--base", default=None, help="the control the others are paired against (default: the first by name)")
    ap.add_argument("--pdm", required=True)
    ap.add_argument("--terms", default=None)
    ap.add_argument("--sets", default=None, help="dump dir for the path-length telemetry")
    ap.add_argument("--rl-val-scenes", default=None)
    ap.add_argument("--require-full", action="store_true")
    ap.add_argument("--rule", required=True, choices=RULES)
    ap.add_argument("--lam-d", type=float, default=0.15)
    ap.add_argument("--w-prog", type=float, default=0.5)
    ap.add_argument("--use-k", action="store_true")
    ap.add_argument("--use-l", action="store_true")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    arms, hashes = {}, {}
    for spec in a.scores:
        name, _, path = spec.partition("=")
        arms[name] = load_scores(path)
        hashes[name] = sha256_file(path)
    pdm, pst = load_pdm(a.pdm)
    terms, tst = (load_terms(a.terms) if a.terms else (None, None))
    cfg = rl_reward.RewardCfg(lam_d=a.lam_d, w_prog=a.w_prog, guard=True, use_k=a.use_k, use_l=a.use_l)
    if (cfg.lam_d != 0.0 or cfg.use_k or cfg.use_l) and terms is None:
        print("REFUSED: the reward needs the rl_terms side files (--terms) unless --lam-d 0 is NAMED and K / L are off", flush=True)
        return 4
    scenes = read_scenes(a.rl_val_scenes) if a.rl_val_scenes else None
    common = set.intersection(*[set(x) for x in arms.values()])
    paths = None
    if a.sets:
        paths = load_paths(a.sets, common, {k: next(iter(arms.values()))[k]["ckpt"] for k in common})
    try:
        res = run(arms, pdm, terms, paths, a.rule, cfg, scenes, a.require_full, a.base, a.n_boot)
    except ValueError as e:
        print(f"ZZRLVAL_REFUSED {e}", flush=True)
        return 3
    res["inputs"] = {"scores_sha256": hashes, "pdm_dir": os.path.basename(os.path.normpath(a.pdm)), "pdm_stats": pst, "terms_stats": tst,
                     "rl_val_scenes_sha256": sha256_file(a.rl_val_scenes) if a.rl_val_scenes else None}
    res["rl_val_scenes_checked"] = scenes is not None
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1, sort_keys=True)
    sc = res["selfcheck"]
    ok = sc["selfpair"]["ok"] and sc.get("swap", {"ok": True})["ok"]
    print(f"ZZRLVAL_{'OK' if ok else 'SELFCHECK_FAILED'} sets {res['eligibility']['sets_used']} arms {sorted(arms)}", flush=True)
    return 0 if ok else 5


if __name__ == "__main__":
    sys.exit(main())
