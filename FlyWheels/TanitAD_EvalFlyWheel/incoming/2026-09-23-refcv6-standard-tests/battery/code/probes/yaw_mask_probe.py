"""Probe: is the paired LAT_yaw_rate cell (refav1_arm._components) scored on stopped/crawling steps?
Compares, per arm, the UNMASKED per-window yaw-rate MAE (as _components computes it) against the same
quantity restricted to four_families' own pair_valid mask (pred AND gt both valid). Read-only on a panel."""
import sys, json, math
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import refcv6_panel as P  # noqa: E402
import torch  # noqa: E402
from taniteval import four_families as ff  # noqa: E402

panel = sys.argv[1]
arms = sys.argv[2].split(",")
man, A, eid = P.load_panel(panel)
dt = float(man["grid"]["dt_s"])
G = torch.as_tensor(A["g"]).float()
Gg = ff._seq_geometry(G, dt)
out = {"panel": panel, "dt_s": dt, "n_windows": int(len(eid)), "min_ds_m": float(Gg["min_ds_m"]), "arms": {}}
for a in arms:
    Pp = torch.as_tensor(A[a]).float()
    Pg = ff._seq_geometry(Pp, dt)
    err = (Pg["yaw_rate"] - Gg["yaw_rate"]).abs()             # [n, H-1] rad/s
    unm = err.mean(1).numpy()
    m = Pg["pair_valid"] & Gg["pair_valid"]
    nv = m.sum(1)
    msk = torch.where(nv > 0, (err * m).sum(1) / nv.clamp_min(1), torch.nan).numpy()
    pred_invalid = (~Pg["pair_valid"]).any(1).numpy()
    gt_invalid = (~Gg["pair_valid"]).any(1).numpy()
    big = unm > 1.0
    out["arms"][a] = {
        "unmasked_mean_radps": float(np.mean(unm)),
        "masked_mean_radps": float(np.nanmean(msk)),
        "masked_n_windows": int(np.isfinite(msk).sum()),
        "windows_with_pred_invalid_step": int(pred_invalid.sum()),
        "windows_with_gt_invalid_step": int(gt_invalid.sum()),
        "windows_unmasked_gt_1radps": int(big.sum()),
        "share_of_unmasked_sum_from_those": float(unm[big].sum() / max(unm.sum(), 1e-12)),
    }
print(json.dumps(out, indent=1))
