"""wprobe_score.py -- scoring half of PREREG_WEATHERPROBE (dev box, numpy only).  Same labels, mask, constant baseline and estimator as strat1_ctx_read.py (imported, not re-written).
Usage: python wprobe_score.py OUT_NPZ META_JSON OUT_JSON [--n-boot 2000]"""
import json
import sys

import numpy as np

sys.path.insert(0, "F:/TanitAD-scratch/strat1/read")
import strat1_lib as L  # noqa: E402
import strat1_ctx_read as R  # noqa: E402

npz, meta_p, out_p = sys.argv[1:4]
nb = int(sys.argv[sys.argv.index("--n-boot") + 1]) if "--n-boot" in sys.argv else L.NB
CI = L.load_ci()
join = dict(np.load(L.pinned("join_eval141.npz")))
tab = dict(np.load(L.pinned("strat_labels_table.npz")))
plat = dict(np.load(L.INP + "/strat_platform_by_clip.npz"))
ci, t = L.grid_windows(L.pinned("eval_grid_windows.json"), join["clip"])
R.check_alignment(tab, plat, join)
y = R.label_rows(tab, L.rows_of(tab["eval_off"], ci, t))
base = R.table_baselines(tab)
K = R.CTX_VARS.index("weather")
yw = y[:, K]
const = base["weather"]["constant_class"]
z = np.load(npz)
assert np.array_equal(z["eval_ci"], ci) and np.array_equal(z["eval_t"], t), "windows differ from the grid"
meta = json.load(open(meta_p))
rep = {"tool": "wprobe_score.py", "tier": "T0 (representation read at the NOW frame)", "evidence_class": "MEASURED (ours)",
       "estimator": f"taniteval.ci.paired_episode_cluster_bootstrap, B={nb}, seed 0 ({CI.SOURCE}); 'another draw of EPISODES' only",
       "constant_class": int(const), "meta": meta, "arms": {}}


def row(logits):
    r = R.var_rows(CI, logits.argmax(1), yw, ci, const, 4, nb)
    r["detects"] = R.detects(r)
    return r


# constant predictor control (must read exactly the constant baseline)
cl = np.full((len(yw), 4), -9.0)
cl[:, const] = 9.0
rc = row(cl)
rep["control_constant"] = {"accuracy": rc["accuracy"], "constant_accuracy": rc["constant_accuracy"], "delta": rc["model_minus_constant"]["delta"], "pass": rc["accuracy"] == rc["constant_accuracy"]}
for k in z.files:
    if k.startswith("S1_"):
        rep["arms"][k] = row(z[k])
res = {}
ok_ctrl = rep["control_constant"]["pass"] and all(meta.get(f"identity_s{s}_weather_argmax_agree", 0) >= 0.99 for s in (0, 1)) and all(v == 0 for v in meta["disjointness"].values())
for s in (0, 1):
    real, r1 = rep["arms"].get(f"S1_s{s}_real"), rep["arms"].get(f"S1_s{s}_R1_shuffle")
    if real is None or r1 is None:
        res[f"S1_s{s}"] = "NOT-RUN"
        continue
    res[f"S1_s{s}"] = "DETECTS" if (real["detects"] and not r1["detects"]) else ("INSTRUMENT-FAIL (R1 detects)" if r1["detects"] else "NOT-DETECTED")
if not ok_ctrl:
    rep["label"] = "INSTRUMENT-FAIL"
elif any("INSTRUMENT-FAIL" in v for v in res.values()):
    rep["label"] = "INSTRUMENT-FAIL"
elif all(v == "DETECTS" for v in res.values()):
    rep["label"] = "DETECTS(weather)-by-probe at both seeds"
else:
    rep["label"] = "NOT-DETECTED(weather)-by-probe"
rep["per_seed"] = res
rep["controls_ok"] = bool(ok_ctrl)
json.dump(rep, open(out_p, "w"), indent=1, default=float)
print("LABEL:", rep["label"], res)
for k, v in rep["arms"].items():
    m = v["model_minus_constant"]
    print(f"{k:24s} acc {v['accuracy']:.4f} const {v['constant_accuracy']:.4f} delta {m['delta']:+.4f} [{m['lo']:+.4f}, {m['hi']:+.4f}] sep {m['separated']} bal_acc {v['balanced_accuracy']:.4f} n {v['n_windows']}/{v['n_episodes']}")
