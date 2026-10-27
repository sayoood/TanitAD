"""BESIDE: DAG-HACC - DAG at T0 (fresh ages, eval-141, T1-class open loop = own held outputs) on the four families, paired episode-cluster bootstrap B = 2000, per seed.
Per-window metrics = hblend.arm_pw (the eval's own family functions) on each arm's plan against the eval GT; rows = hblend.rows_for / taniteval paired_row. TACTICAL decision heads and
STRATEGIC checkpoint outputs are checked for EXACT equality with plain DAG's (the post-map touches the plan only)."""
import json
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TANITEVAL_STACK_OVERRIDE", "E:/Projects/TanitAD-artifacts/dagt1_tree_b43c7a9a/stack")
sys.dont_write_bytecode = True
PK = "E:/Projects/TanitAD-artifacts/daghacc"
sys.path.insert(0, f"{PK}/hb")
import numpy as np  # noqa: E402
import v8f_hblend as H  # noqa: E402
from tanitad.data import v8f_mini_data as D  # noqa: E402
from taniteval import v8f_beta_eval as BE  # noqa: E402

RAW = "E:/Projects/TanitAD-artifacts/dagt1_tree_b43c7a9a/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw"
grid = json.load(open(f"{RAW}/eval_grid_windows.json"))
Jn = D.load_join(f"{RAW}/join_eval141.npz")
X = D.eval_context(grid, f"{RAW}/eval141_kin.npz", Jn, kin_mode="yaw")
gt = np.asarray(X["gt"], np.float64)
out = {"tier": "T0 / T1-class open loop (eval-141, own held outputs, fresh ages)", "estimator": "paired episode-cluster bootstrap, B=2000, seed 0 (taniteval.ci via v8f_beta_eval.paired_row)", "seeds": {}}
for s in (0, 1):
    a = np.load(f"{PK}/t0/eval_M2b-daghacc_s{s}-fresh.npz")
    b = np.load(f"E:/Projects/TanitAD-artifacts/v8f-dagger-t1/t0fresh/eval_M2b-dag_s{s}-fresh.npz")
    assert np.array_equal(a["eid"], b["eid"])
    eid = a["eid"]
    pa, pb = H.arm_pw(a["pred"], gt, BE), H.arm_pw(b["pred"], gt, BE)
    rows = H.rows_for(pa, pb, eid, BE, 2000)
    for k in ("viol_any", "viol_SPEED", "viol_STOP", "viol_TURN", "viol_LEAD"):
        rows["consistency_" + k] = BE.paired_row(a[k].astype(np.float64), b[k].astype(np.float64), eid, 2000, True)
    ident = {k: (float(np.abs(a[k].astype(np.float64) - b[k].astype(np.float64)).max())) for k in ("ckpt", "goal_prob", "lat_logits", "lon_logits", "cons_pred", "ck_pos_err")}
    means = {k: [float(np.nanmean(pa[k])), float(np.nanmean(pb[k]))] for k in pa}
    out["seeds"][f"s{s}"] = {"rows_DAGHACC_minus_DAG": rows, "means_[DAGHACC,DAG]": means, "tactical_and_strategic_outputs_max_abs_diff_vs_DAG": ident,
                             "n_windows": int(len(eid)), "n_episodes": int(len(np.unique(eid)))}
    sel = ("ade1", "ade2", "ade6", "fde6", "fw_lon_speed_mae", "fw_lon_accel_mae", "fw_lon_along_mae", "fw_lat_cross_mae", "fw_lat_heading_mae_deg", "fw_lat_yaw_rate_mae_degps",
           "fw_lat_curvature_mae_1pm", "fw_tac_lat_acc", "fw_tac_lon_acc")
    print(f"s{s} tactical/strategic outputs max|diff| vs DAG:", ident)
    for k in sel:
        r = rows[k]
        print(f"s{s} {k:28s} DAGHACC - DAG {r['mean']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}] sep={r['separated']}  means {means[k][0]:.4f} / {means[k][1]:.4f}")
json.dump(out, open(f"{PK}/raw/t0_beside_DAGHACC_minus_DAG.json", "w"), indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x))
print("ZZT0BESIDE-DONE")
