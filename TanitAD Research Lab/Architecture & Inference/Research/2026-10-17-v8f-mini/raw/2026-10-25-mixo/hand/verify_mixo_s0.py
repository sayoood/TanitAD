"""Independent re-derivation of PREREG_V8F_MIXO stage s0 from the per-window eval arrays (does not import the reader).
PRIMARY: own-code ADE6 X - B, paired episode-cluster bootstrap (own implementation), bar <= -2F = -0.2415 and separated.
D-M: r(a=0.5) = d(0.5) / d(1), d(a) = mean ADE6(constraints moved a of the way to GT) - mean ADE6(T's own code, tacpred); rise r_X - r_B(committed 0.36) >= 0.10.
HARM: GT-code oracle (tacorc) ADE6 X - B, beside."""
import json
import sys

import numpy as np

E = sys.argv[1]
B_BOOT, SEED = 2000, 0


def load(name):
    z = np.load(f"{E}/eval_{name}.npz")
    return {k: z[k] for k in z.keys()}


def ade6(z):   # the eval's own per-window ADE6 (the npz holds no GT: `pred` == `traj30` == the plan); B's mean is a known value (2.1753, MODEL_REGISTRY)
    return z["ade6"].astype(np.float64)


def boot(delta, eid):
    """Paired episode-cluster bootstrap of the window-mean of `delta`: resample episodes with replacement, keep each episode's windows."""
    rng = np.random.default_rng(SEED)
    eps = np.unique(eid)
    idx = [np.flatnonzero(eid == e) for e in eps]
    s = np.array([delta[i].sum() for i in idx]); n = np.array([len(i) for i in idx])
    draws = rng.integers(0, len(eps), size=(B_BOOT, len(eps)))
    m = s[draws].sum(1) / n[draws].sum(1)
    lo, hi = np.percentile(m, [2.5, 97.5])
    return {"point": float(delta.mean()), "lo": float(lo), "hi": float(hi), "separated": bool(lo > 0 or hi < 0), "n_windows": int(len(delta)), "n_episodes": int(len(eps))}


X, Bz = load("M2b-mixo_s0"), load("M2b_s0")
assert np.array_equal(X["eid"], Bz["eid"])
eid = X["eid"]
out = {"pred_equals_traj30": bool(np.array_equal(X["pred"], X["traj30"]) and np.array_equal(Bz["pred"], Bz["traj30"]))}
out["PRIMARY"] = boot(ade6(X) - ade6(Bz), eid)
out["PRIMARY"]["ade6_X"], out["PRIMARY"]["ade6_B"] = float(ade6(X).mean()), float(ade6(Bz).mean())
out["PRIMARY"]["bar"] = -0.2415
out["PRIMARY"]["met"] = bool(out["PRIMARY"]["point"] <= -0.2415 and out["PRIMARY"]["hi"] < 0)
r = {}
for arm in ("M2b-mixo_s0", "M2b_s0"):
    pred, full, half = (ade6(load(f"{arm}-{s}")) for s in ("tacpred", "tacorc-constraints", "tacorc-constraints-a50"))
    std = ade6(load(arm))
    r[arm] = {"tacpred_vs_standard_max_m": float(np.abs(pred - std).max()), "d_full": float((full - pred).mean()), "d_half": float((half - pred).mean())}
    r[arm]["r_half"] = r[arm]["d_half"] / r[arm]["d_full"]
out["DM"] = r
out["DM"]["rise_over_committed_0.36"] = r["M2b-mixo_s0"]["r_half"] - 0.36
out["DM"]["rise_over_measured_B"] = r["M2b-mixo_s0"]["r_half"] - r["M2b_s0"]["r_half"]
out["HARM"] = boot(ade6(load("M2b-mixo_s0-tacorc")) - ade6(load("M2b_s0-tacorc")), eid)
p = out["PRIMARY"]["met"]
rise = out["DM"]["rise_over_committed_0.36"]
out["VERDICT_BY_HAND"] = "PASS(seed0 only; seed 1 needed)" if p else ("FAIL-O-LEARNED" if rise >= 0.10 else "FAIL-O-DID-NOT-LEARN")
print(json.dumps(out, indent=1))
