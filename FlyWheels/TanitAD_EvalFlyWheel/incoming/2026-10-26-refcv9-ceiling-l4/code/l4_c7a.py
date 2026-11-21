"""Step 3d (Rule Zero next lever): C7a = C7 lifted ONE 4-way step when the IMU a0 at t0 (past-only 1-s backward difference)
exceeds a threshold. Threshold chosen on TRAIN only (max compliance s.t. leak < 0.05), then applied to eval139 unchanged.
Leak base = [1, v0, a0] (a0 is an admissible t0 input; information it carries is not a leak)."""
import json, sys, numpy as np
sys.path.insert(0, "F:/refcv9_ceiling/code")
import l4_common as C
from l4_candidates import train_windows, eval_windows, oof_recovered, onehot_of

STEPS = [30.0, 50.0, 100.0, 120.0]
def lift(c, on):
    out = c.copy()
    for a, b in zip(STEPS[:-1], STEPS[1:]):
        out = np.where(on & (np.abs(c - a) < 1e-6), b, out)
    return out

def prep(w, sc_path, v9_path):
    sc = C.load_sidecar(sc_path); v9 = C.V9(v9_path); a0 = np.full(len(w["kn"]), np.nan); cache = {}
    for j, (c, k) in enumerate(zip(w["cid"], w["kn"])):
        if c not in cache:
            cache[c] = v9.clip(sc[c]["sid"])["V"]
        V = cache[c]
        if k - 10 >= 0 and k < len(V):
            a0[j] = (V[k] - V[k - 10]) / 1.0
    n2 = w["n2"]; ok = np.isfinite(n2) & (n2 > 0)
    c7 = np.where(ok, C.snap_up_kmh(np.where(ok, n2, 0) / 3.6, C.L4_KMH, clamp_value=True), np.inf)
    return np.nan_to_num(a0), c7

def _main():
  res = {}
  wt = train_windows(); a0t, c7t = prep(wt, C.SIDECAR_TRAIN, C.V9_TRAIN)
  we = eval_windows(); a0e, c7e = prep(we, C.SIDECAR_EVAL, C.V9_EVAL)
  def leak(w, a0, ceil):
      m = np.isfinite(w["y26"]) & np.isfinite(w["v0"])
      folds = np.array([int(C.sha12(c), 16) % 5 for c in w["cid"][m]])
      base = np.stack([np.ones(m.sum()), w["v0"][m], a0[m]], 1)
      return round(oof_recovered(w["y26"][m], base, onehot_of(np.round(np.minimum(ceil[m], 131), 3)), folds)[2], 4)
  res["train_sweep"] = {}
  for th in (0.25, 0.5, 0.75, 1.0, 1.5):
      c = lift(c7t, a0t > th)
      res["train_sweep"][str(th)] = {"compliance": round(float((wt["gt"] <= c / 3.6).mean()), 4), "leak_vs_[1,v0,a0]": leak(wt, a0t, c),
                                     "frac_lifted": round(float(((a0t > th) & (c != c7t)).mean()), 4)}
  res["train_C7_baseline"] = {"compliance": round(float((wt["gt"] <= c7t / 3.6).mean()), 4), "leak_vs_[1,v0,a0]": leak(wt, a0t, c7t)}
  ok = {k: v for k, v in res["train_sweep"].items() if v["leak_vs_[1,v0,a0]"] < 0.05}
  th = max(ok, key=lambda k: ok[k]["compliance"]) if ok else None
  res["chosen_on_train"] = th
  if th is not None:
      ce = lift(c7e, a0e > float(th))
      res["eval139_applied"] = {"threshold_ms2": float(th), "compliance": round(float((we["gt"] <= ce / 3.6).mean()), 4),
                                "n_exceed": int((we["gt"] > ce / 3.6).sum()), "leak_vs_[1,v0,a0]": leak(we, a0e, ce),
                                "C7_eval_compliance": round(float((we["gt"] <= c7e / 3.6).mean()), 4)}
  json.dump(res, open("F:/refcv9_ceiling/raw/l4_c7a.json", "w", encoding="utf-8"), indent=1)
  print(json.dumps(res, indent=1))


if __name__ == "__main__":
    _main()
