"""D-NAVROUTE-1 §4 -- did intervening on g_str MOVE THE PLAN?

Compares each intervention arm's emitted plan against the REPLICATE baseline,
window by window, on the SAME matched episode subset. The replicate is the
rig's own run-to-run noise floor: an intervention effect inside it is NOT an
effect, however separated its CI looks.

ASCII-only output (cp1252-safe). Estimator: paired episode-cluster bootstrap.
"""
import glob
import json
import os
import sys

import numpy as np

ROOT = os.environ.get("NAVROUTE", "/home/nvidia/navroute")
NBOOT = int(os.environ.get("NBOOT", "2000"))
SEED = int(os.environ.get("SEED", "0"))
BASE_TAG = "REPL"
ARMS = ["GZERO", "BLIND", "GSHUF"]


def _clip_index_for(dumpdir, base):
    """⛔ The join key is the CLIP INDEX, read from the top-level dump.

    The `decisions/` sidecar does NOT carry `clip_index`, and joining on the
    FILENAME would silently assume both runs enumerated episodes in the same
    order. A wrong-clip join is worse than a crash: it would compare one clip's
    plan against another's and read as a large intervention effect. So the key
    comes from the top-level `ep*.npz`, and a missing one is skipped loudly.
    """
    top = os.path.join(dumpdir, base)
    if not os.path.exists(top):
        return None
    t = np.load(top, allow_pickle=True)
    if "clip_index" not in t.files:
        return None
    return int(np.asarray(t["clip_index"]).reshape(-1)[0])


def load_dump(tag, explicit_dir=None):
    """-> dict keyed by (clip_index, ws) with the plan and g_str per window."""
    dumpdir = explicit_dir or os.path.join(ROOT, f"dump_{tag}")
    d = os.path.join(dumpdir, "decisions")
    if not os.path.isdir(d):
        return None
    out, skipped = {}, 0
    for f in sorted(glob.glob(os.path.join(d, "*.npz"))):
        z = np.load(f, allow_pickle=True)
        if "plan_full_nav_true" not in z.files:
            continue
        ci = _clip_index_for(dumpdir, os.path.basename(f))
        if ci is None:
            skipped += 1
            continue
        ws = np.asarray(z["ws"]).reshape(-1)
        plan = np.asarray(z["plan_full_nav_true"])
        g = (np.asarray(z["gstr_nav_true"]) if "gstr_nav_true" in z.files
             else np.full((len(ws), 3), np.nan, np.float32))
        for i, w in enumerate(ws):
            out[(ci, int(w))] = (plan[i], g[i], ci)
    if skipped:
        print(f"# WARN {tag}: {skipped} episodes skipped (no clip_index)",
              file=sys.stderr)
    return out


def boot_mean(ep, x, nboot, seed):
    rng = np.random.default_rng(seed)
    ue = np.unique(ep)
    idx = {e: np.where(ep == e)[0] for e in ue}
    draws = np.empty(nboot)
    for i in range(nboot):
        pick = rng.choice(ue, size=ue.size, replace=True)
        sel = np.concatenate([idx[e] for e in pick])
        draws[i] = x[sel].mean()
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"mean": round(float(x.mean()), 4),
            "lo": round(float(lo), 4), "hi": round(float(hi), 4),
            "median": round(float(np.median(x)), 4),
            "max": round(float(x.max()), 4),
            "frac_exactly_zero": round(float((x == 0).mean()), 4),
            "n_windows": int(x.size), "n_episodes": int(ue.size)}


def compare(base, arm, nboot, seed):
    keys = sorted(set(base) & set(arm))
    if not keys:
        return {"ERROR": "no shared windows"}
    ep = np.array([base[k][2] for k in keys])
    dplan = np.array([float(np.linalg.norm(arm[k][0][-1] - base[k][0][-1]))
                      for k in keys])
    dg = np.array([float(np.linalg.norm(arm[k][1] - base[k][1])) for k in keys])
    return {"n_shared_windows": len(keys),
            "plan_terminal_displacement_m": boot_mean(ep, dplan, nboot, seed),
            "gstr_displacement": boot_mean(ep, dg, nboot, seed)}


def main():
    base = load_dump(BASE_TAG)
    res = {"tool": "D-NAVROUTE-1 g_str intervention comparison",
           "root": ROOT, "baseline": BASE_TAG,
           "tier": "T1 (self-action OPEN loop, 2026-09-02 ruling)",
           "estimator": f"paired episode-cluster bootstrap, {NBOOT} resamples, "
                        f"seed {SEED}",
           "variance_answered": "EPISODE DRAW; the run-to-run floor is the "
                                "REPL-vs-banked-os row below"}
    if base is None:
        print(json.dumps({"ERROR": f"no baseline dump for {BASE_TAG}"}, indent=1))
        return
    res["baseline_windows"] = len(base)

    # ---- THE NOISE FLOOR: replicate vs the BANKED os arm, same ckpt/config ---
    bank = load_dump("__banked__",
                     explicit_dir="/home/nvidia/navpred/navflip_dump")
    if bank:
        res["NOISE_FLOOR_replicate_vs_banked_os"] = compare(bank, base,
                                                            NBOOT, SEED)
        res["_floor_reads"] = ("the SAME flags and the SAME seed, run again. Any "
                               "intervention effect at or below this is NOT an "
                               "effect -- measured replicate false-positive rate "
                               "on this programme's rigs is 6/42 = 14.3 %")

    res["arms"] = {}
    for tag in ARMS:
        a = load_dump(tag)
        res["arms"][tag] = ({"STATUS": "not present"} if a is None
                            else compare(base, a, NBOOT, SEED))
    print(json.dumps(res, indent=1))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
