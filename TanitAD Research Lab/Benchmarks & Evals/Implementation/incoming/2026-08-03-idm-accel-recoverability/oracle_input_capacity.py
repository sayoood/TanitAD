"""STREAM D — the CAPACITY CONTROL, done properly.

The whole "UNRECOVERABLE" reading rests on one control: *the same architectures,
the same optimiser, the same loss and the same epoch protocol DO learn
``long_accel`` when the input carries it.* If that fails, the null is about the
head and not about the representation.

⛔ THE DEFECT THIS FIXES, caught by reading the control's own positive channel.
In the headline run the oracle-input arms were fed the RAW CAN speed window in
m/s (0–40). A head whose input literally IS the speed then scored ``speed``
R² **+0.4678** (transformer) / **+0.4777** (MLP) — a direct copy should be ≈1.0.
That is an input-SCALING artifact of the control, not a property of the
architecture, and it makes the control weaker than it should be in exactly the
direction that would flatter a null.

So run the oracle-input arms twice — RAW and STANDARDISED — and report both, so
the artifact is visible rather than quietly fixed. The standardised arms are the
capacity control the verdict leans on; the raw arms stay as the record of why.

usage: python oracle_input_capacity.py --out raw/oracle_input_capacity.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
sys.path.insert(0, str(REPO / "stack"))
sys.path.insert(0, str(REPO / "stack" / "scripts"))
sys.path.insert(0, str(REPO / "taniteval"))

T0 = time.time()
SCALARS = ("speed", "yaw_rate", "steer", "long_accel")
HORIZONS = (5, 10, 15, 20)
K = 4


def log(m):
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


def main() -> int:
    from tanitad.eval import accel_probe as AP
    from tanitad.eval import ap_ci as APCI
    from tanitad.eval import idm_families as FF
    import idm_head as ih
    import run_accel_recoverability as R

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "raw" / "oracle_input_capacity.json"))
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--max-epochs", type=int, default=40)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")

    sub = R.load_substrate()
    eps = sub["episodes"]
    tr_eps, ho_eps = R.split_episodes(eps)
    fit_eps = [e for i, e in enumerate(tr_eps) if i % 3 != 0]
    sel_eps = [e for i, e in enumerate(tr_eps) if i % 3 == 0]
    _, Yf, _ = R.stack(fit_eps)
    _, Ys, _ = R.stack(sel_eps)
    _, Ytr, _ = R.stack(tr_eps)
    _, Yho, eid = R.stack(ho_eps)

    def q(es):
        return torch.cat([e["Q"] for e in es]).float().unsqueeze(-1)
    Qf, Qs, QT, Qho = q(fit_eps), q(sel_eps), q(tr_eps), q(ho_eps)
    mu, sd = float(QT.mean()), float(QT.std())
    log(f"oracle input: true CAN speed window [N,9,1]; train mean {mu:.3f} "
        f"std {sd:.3f} m/s")

    Sf, Tf = Yf[:, :4], Yf[:, 4:].reshape(len(Yf), len(HORIZONS), 2)
    Ss, Ts = Ys[:, :4], Ys[:, 4:].reshape(len(Ys), len(HORIZONS), 2)
    ST, TT = Ytr[:, :4], Ytr[:, 4:].reshape(len(Ytr), len(HORIZONS), 2)
    gs = Yho[:, :4].numpy().astype(np.float64)
    gt = Yho[:, 4:].numpy().astype(np.float64).reshape(len(Yho), len(HORIZONS), 2)

    res = {"_what": "capacity control: same heads, ORACLE (true speed) input, "
                    "raw vs standardised",
           "input": "true CAN speed at the 9 window positions",
           "train_mean_mps": round(mu, 4), "train_std_mps": round(sd, 4),
           "n_heldout_windows": int(len(Qho)), "seeds": a.seeds,
           "estimator": "episode-cluster bootstrap (paired vs the train-mean null)",
           "arms": {}}
    NULL = np.tile(Ytr.mean(0, keepdim=True).numpy(), (len(Yho), 1))

    specs = [
        ("transformer_d256_L3",
         lambda: ih.IDMHead(state_dim=1, d_model=256, depth=3, n_heads=4,
                            window=2 * K + 1, horizons=HORIZONS)),
        ("mlp_window_h512", lambda: AP.MLPHead(1, 512, 2, mode="window")),
        ("gru_d128_L2", lambda: AP.GRUHead(1, 128, 2)),
        # BLIND BY CONSTRUCTION: v(t) alone cannot contain its own derivative.
        # If this one ever scored, the pipeline would be leaking.
        ("mlp_centreONLY_blind", lambda: AP.MLPHead(1, 512, 2, mode="centre")),
    ]
    for scale in ("raw", "standardised"):
        f = (lambda X: X) if scale == "raw" else (lambda X: (X - mu) / sd)
        for name, make in specs:
            tag = f"ORACLEIN_{scale}_{name}"
            t = time.time()
            ho, tr, ho_seeds, m = R.neural_arm(
                AP, make, (f(Qf), Sf, Tf), (f(Qs), Ss, Ts), (f(QT), ST, TT),
                f(Qho), Yho[:, :4], seeds=a.seeds, max_epochs=a.max_epochs,
                device=dev)
            ps = ho[:, :4]
            pt = ho[:, 4:].reshape(len(ho), len(HORIZONS), 2)
            m.update({
                "input_scaling": scale,
                "fit_seconds": round(time.time() - t, 1),
                "per_seed_heldout_long_accel_r2":
                    [round(AP.r2_score(s[:, 3], gs[:, 3]), 4) for s in ho_seeds],
                "r2": {SCALARS[j]: APCI.stat_episode_cluster_bootstrap(
                    (lambda s, j=j: AP.r2_score(ps[s, j], gs[s, j])), eid,
                    n_boot=a.n_boot, name=f"r2_{SCALARS[j]}") for j in range(4)},
                "r2_train_insample": {
                    SCALARS[j]: round(AP.r2_score(tr[:, j], ST[:, j].numpy()), 4)
                    for j in range(4)},
                "paired_vs_train_mean_null": {
                    SCALARS[j]: APCI.paired_stat_episode_cluster_bootstrap(
                        (lambda s, j=j: AP.r2_score(ps[s, j], gs[s, j])),
                        (lambda s, j=j: AP.r2_score(NULL[s, j], gs[s, j])), eid,
                        n_boot=a.n_boot, name=f"d_r2_{SCALARS[j]}")
                    for j in range(4)},
                "four_families": FF.all_families(pt, gt, FF.IDM_DT_S,
                                                 pred_scalars=ps, gt_scalars=gs),
            })
            m["four_families"]["TACTICAL"]["from_long_accel_scalar"] = \
                R.accel_tactical(FF, ps[:, 3], gs[:, 3])
            res["arms"][tag] = m
            log(f"  {tag}: speed {m['r2']['speed']['point']:+.4f}  accel "
                f"{m['r2']['long_accel']['point']:+.4f} "
                f"[{m['r2']['long_accel']['lo']:+.4f},"
                f"{m['r2']['long_accel']['hi']:+.4f}]  d-vs-null "
                f"{m['paired_vs_train_mean_null']['long_accel']['delta']:+.4f}"
                f"{'*' if m['paired_vs_train_mean_null']['long_accel']['separated'] else ''}"
                f"  ({m['fit_seconds']}s)")

    # ---- the CLOSED-FORM capacity control, matched to the family that carries
    # the verdict. The ridge ladder is what rules out "the head was too small";
    # its own capacity control therefore has to be a RIDGE on an input that
    # carries the answer, scored through the identical selection protocol.
    sp, hp, tp, keys, meta = R.ridge_arm(
        AP, Qf, Yf, Qs, QT, Ytr, Qho, feat="window", kernel="linear", device=dev)
    chosen = R.select_hparam(AP, sp, Ys, keys, meta)
    P = R.assemble(hp, chosen, meta)
    res["arms"]["ORACLEIN_ridge_linear_window"] = {
        "kind": "closed_form_capacity_control",
        "input": "true CAN speed window, 9 features",
        "n_features": meta["n_features"],
        "selected_alpha_long_accel": R.jsonable(chosen[3][0][1]),
        "inner_val_r2_long_accel": round(chosen[3][2], 4),
        "r2": {SCALARS[j]: APCI.stat_episode_cluster_bootstrap(
            (lambda s, j=j: AP.r2_score(P[s, j], gs[s, j])), eid,
            n_boot=a.n_boot, name=f"r2_{SCALARS[j]}") for j in range(4)},
        "paired_vs_train_mean_null": {
            SCALARS[j]: APCI.paired_stat_episode_cluster_bootstrap(
                (lambda s, j=j: AP.r2_score(P[s, j], gs[s, j])),
                (lambda s, j=j: AP.r2_score(NULL[s, j], gs[s, j])), eid,
                n_boot=a.n_boot, name=f"d_r2_{SCALARS[j]}") for j in range(4)},
    }
    r = res["arms"]["ORACLEIN_ridge_linear_window"]
    log(f"  ORACLEIN_ridge_linear_window: speed {r['r2']['speed']['point']:+.4f}"
        f"  accel {r['r2']['long_accel']['point']:+.4f} "
        f"[{r['r2']['long_accel']['lo']:+.4f},{r['r2']['long_accel']['hi']:+.4f}]"
        f"  d-vs-null {r['paired_vs_train_mean_null']['long_accel']['delta']:+.4f}"
        f"{'*' if r['paired_vs_train_mean_null']['long_accel']['separated'] else ''}")
    del sp, hp, tp

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=str))
    log(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
