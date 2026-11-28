import sys, numpy as np
PKG = "E:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
sys.path.insert(0, PKG + "/refe"); sys.path.insert(0, PKG + "/eval")
import rls1_checks as RC, rl_val_eval as RV
p = "C:/Users/Admin/AppData/Local/Temp/rls1dl/trunc_600.npz"
a, b = RV.load_scores(p), RC.load_scores_light(p)
assert list(a) == list(b) and len(a) == 600
for k in a:
    assert a[k]["ckpt"] == b[k]["ckpt"] and a[k]["space"] == b[k]["space"] and a[k]["scores"].dtype == b[k]["scores"].dtype
    assert a[k]["scores"].shape == b[k]["scores"].shape and a[k]["scores"].tobytes() == b[k]["scores"].tobytes()
print("EQUAL loaders: keys in the same order, ckpt, space, dtype, shape and the BYTES of every scores row (600 sets)")
