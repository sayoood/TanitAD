"""CONTROL: re-read the banked lateral-head agreement from the per-frame battery dumps; must read 0.9415 at seed 0
(TAC_CONTROLS_step48852ema_pf.json).  Also reads the longitudinal head (0.6103) and compares the constant control.
CPU only; read-only."""
import glob, json, sys
import numpy as np
sys.path.insert(0, "E:/Projects/TanitAD/taniteval")
from taniteval import ci as TCI

ROOT = "C:/Users/Admin/refcv8_eval_archive/battery/step48852ema_pf"
BANK = json.load(open("C:/Users/Admin/refcv8_pf/post48852/TAC_CONTROLS_step48852ema_pf.json"))
out = {}
for seed in (0, 1):
    files = sorted(glob.glob(f"{ROOT}/dump_s{seed}/decisions/ep*.npz"))
    rec = {"n_files": len(files)}
    for head, key_p, key_l in (("lat", "lat_pred_nav_true", "lat_label"), ("lon", "lon_pred_nav_true", "lon_label")):
        P, L, E = [], [], []
        for i, f in enumerate(files):
            z = np.load(f, allow_pickle=True)
            P.append(z[key_p]); L.append(z[key_l]); E.append(np.full(len(z[key_l]), i))
        P, L, E = np.concatenate(P), np.concatenate(L), np.concatenate(E)
        ok = L >= 0
        cls, cnt = np.unique(L[ok], return_counts=True)
        const = int(cls[np.argmax(cnt)])
        # the banked control's validity rule: windows whose label is an EXACT class (>=0) -- confirm by count
        acc = float((P[ok] == L[ok]).mean()); accc = float((L[ok] == const).mean())
        d = TCI.paired_episode_cluster_bootstrap((P[ok] == L[ok]).astype(float), (L[ok] == const).astype(float), E[ok], n_boot=2000, seed=0)
        b = BANK["seeds"][str(seed)][head]
        rec[head] = {"n_windows": int(ok.sum()), "n_episodes": int(len(np.unique(E[ok]))), "acc": round(acc, 4), "acc_constant": round(accc, 4),
                     "constant_class": const, "delta": d["delta"], "lo": d["lo"], "hi": d["hi"],
                     "banked": {"acc": b["acc_nav_true"], "acc_constant": b["acc_constant"], "n_windows": b["n_windows"], "n_episodes": b["n_episodes"],
                                "delta": b["head_minus_constant"]["delta"], "lo": b["head_minus_constant"]["lo"], "hi": b["head_minus_constant"]["hi"]},
                     "reads_banked_value": bool(abs(acc - b["acc_nav_true"]) < 5e-5 and ok.sum() == b["n_windows"] and abs(accc - b["acc_constant"]) < 5e-5)}
    out[f"seed{seed}"] = rec
json.dump(out, open("C:/Users/Admin/refcv8_pf/tacconstr/CONTROL_lat_lon_agreement_reread.json", "w"), indent=1)
for s, r in out.items():
    for h in ("lat", "lon"):
        x = r[h]; print(s, h, "acc", x["acc"], "const", x["acc_constant"], "n", x["n_windows"], x["n_episodes"], "delta", x["delta"], x["lo"], x["hi"], "| banked", x["banked"]["acc"], x["banked"]["delta"], "| reads_banked:", x["reads_banked_value"])
