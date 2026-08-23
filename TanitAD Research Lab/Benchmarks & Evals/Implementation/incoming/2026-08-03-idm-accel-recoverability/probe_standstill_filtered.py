"""STREAM D, the last alternative explanation — is the null caused by LABEL NOISE?

MEASURED here (`raw/substrate_verification.json` + this script's own preflight):
two of the fifty "content-clean" comma2k19 episodes are a **PARKED CAR**.
``ep_00045`` has speed 0.0003–0.015 m/s across all 138 of its windows, its CAN
``long_accel`` is identically **0.0**, and its heading-repair ``yaw_rate`` reaches
**15.275 rad/s** (876 °/s) because the heading is arctan2 of a velocity vector
that is essentially zero. ``ep_00080`` is likewise 100 % stationary. Ten of the
fifty episodes have more than 5 % stationary windows.

So before "the representation does not carry ``long_accel``" can stand, one
alternative has to be killed: **the target is partly a constant-zero label from
parked cars, and the probe is fitting that instead.** Refit the ladder with
stationary windows removed from TRAIN, and score twice —

  * on the FULL held-out set (rows identical to the headline run, so the numbers
    are directly comparable), and
  * on the MOVING-ONLY held-out subset (the domain the IDM would actually label).

Both with the matched shuffled-latent control and the paired episode-cluster
bootstrap. If ``long_accel`` becomes recoverable, the finding is a DATA finding
and the fix is a filter. If it does not, the label-noise explanation is dead.

usage: python probe_standstill_filtered.py --out raw/standstill_filtered.json
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

V_MIN = 0.5          # m/s — the same floor the heading repair uses
T0 = time.time()
SCALARS = ("speed", "yaw_rate", "steer", "long_accel")


def log(m):
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


def main() -> int:
    from tanitad.eval import accel_probe as AP
    from tanitad.eval import ap_ci as APCI
    import run_accel_recoverability as R

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "raw" / "standstill_filtered.json"))
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")

    sub = R.load_substrate()
    eps = sub["episodes"]
    tr_eps, ho_eps = R.split_episodes(eps)
    fit_eps = [e for i, e in enumerate(tr_eps) if i % 3 != 0]
    sel_eps = [e for i, e in enumerate(tr_eps) if i % 3 == 0]

    def blocks(es, shuffle=None):
        Z, Y, eid = R.stack(es, shuffle)
        mv = (Y[:, 0].numpy() >= V_MIN)
        return Z, Y, eid, mv

    Zf, Yf, _, mf = blocks(fit_eps)
    Zs, Ys, _, ms = blocks(sel_eps)
    Ztr, Ytr, _, mt = blocks(tr_eps)
    Zho, Yho, eid, mh = blocks(ho_eps)
    res = {
        "_what": ("does removing STATIONARY windows from TRAIN make long_accel "
                  "recoverable? The last alternative to 'the representation does "
                  "not carry it'."),
        "v_min_mps": V_MIN,
        "standstill_census": {
            "train_windows_total": int(len(Ztr)),
            "train_windows_stationary": int((~mt).sum()),
            "train_frac_stationary": round(float((~mt).mean()), 4),
            "heldout_windows_total": int(len(Zho)),
            "heldout_windows_stationary": int((~mh).sum()),
            "heldout_frac_stationary": round(float((~mh).mean()), 4),
            "fully_parked_episodes": [
                e["name"] for e in eps
                if float((e["S"].numpy()[:, 0] < V_MIN).mean()) == 1.0],
        },
        "estimator": "episode-cluster bootstrap; paired for arm-vs-control",
        "n_boot": a.n_boot, "arms": {}, "paired_vs_control": {},
    }
    log(f"stationary: train {res['standstill_census']['train_frac_stationary']:.1%} "
        f"heldout {res['standstill_census']['heldout_frac_stationary']:.1%} "
        f"fully parked {res['standstill_census']['fully_parked_episodes']}")

    gs_full = Yho[:, :4].numpy().astype(np.float64)
    preds = {}
    for feat in ("centre", "window"):
        for tag, shuffle in (("", None), ("__CTRL", 101)):
            Zf_ = Zf if shuffle is None else R.stack(fit_eps, shuffle)[0]
            ZT_ = Ztr if shuffle is None else R.stack(tr_eps, shuffle)[0]
            sp, hp, tp, keys, meta = R.ridge_arm(
                AP, Zf_[mf], Yf[mf], Zs[ms], ZT_[mt], Ytr[mt], Zho,
                feat=feat, kernel="linear", device=dev)
            chosen = R.select_hparam(AP, sp, Ys[ms], keys, meta)
            name = f"FILTERED_ridge_linear_{feat}{tag}"
            preds[name] = R.assemble(hp, chosen, meta)
            res["arms"][name] = {
                "feature": feat, "shuffled_latents": shuffle is not None,
                "n_train_windows_after_filter": int(mt.sum()),
                "inner_val_r2_best_over_grid":
                    {SCALARS[j]: round(chosen[j][2], 4) for j in range(4)},
                "ORACLE_SELECTED_heldout_r2_UPPER_BOUND_cheating":
                    R.oracle_selected_r2(AP, hp, Yho, keys, meta)}
            del sp, hp, tp
            log(f"  {name}: inner accel {chosen[3][2]:+.4f}")

    # the empirical null on the FILTERED train set
    preds["FILTERED_NULL_train_mean"] = np.tile(
        Ytr[mt].mean(0, keepdim=True).numpy(), (len(Yho), 1))
    res["arms"]["FILTERED_NULL_train_mean"] = {
        "kind": "empirical_null_on_filtered_train", "shuffled_latents": False}

    for scope, mask in (("heldout_FULL_same_rows_as_headline", np.ones(len(Zho), bool)),
                        ("heldout_MOVING_ONLY", mh)):
        e_s = eid[mask]
        for name, P in preds.items():
            key = f"{name}|{scope}"
            res["arms"].setdefault(name, {})
            res["arms"][name][scope] = {
                "n_windows": int(mask.sum()),
                "n_episodes": int(len(np.unique(e_s))),
                "r2": {SCALARS[j]: APCI.stat_episode_cluster_bootstrap(
                    (lambda s, j=j, PP=P: AP.r2_score(PP[mask][s, j],
                                                      gs_full[mask][s, j])),
                    e_s, n_boot=a.n_boot, name=f"r2_{SCALARS[j]}")
                    for j in range(4)}}
        for feat in ("centre", "window"):
            base = f"FILTERED_ridge_linear_{feat}"
            for other, tag in ((base + "__CTRL", "minus_CTRL"),
                               ("FILTERED_NULL_train_mean", "minus_NULLMEAN")):
                out = {}
                for j, ch in enumerate(SCALARS):
                    r = APCI.paired_stat_episode_cluster_bootstrap(
                        (lambda s, j=j, PP=preds[base]: AP.r2_score(
                            PP[mask][s, j], gs_full[mask][s, j])),
                        (lambda s, j=j, PP=preds[other]: AP.r2_score(
                            PP[mask][s, j], gs_full[mask][s, j])),
                        e_s, n_boot=a.n_boot, name=f"d_r2_{ch}")
                    spread = float(np.abs(preds[base][mask][:, j]
                                          - preds[other][mask][:, j]).max())
                    if spread <= 1e-10 * (gs_full[mask][:, j].std() + 1e-12):
                        r["separated"], r["degenerate"] = False, True
                    out[f"r2_{ch}"] = r
                res["paired_vs_control"][f"{base}_{tag}|{scope}"] = out
                log(f"  PAIRED {base}_{tag}|{scope}: accel "
                    f"{out['r2_long_accel']['delta']:+.4f}"
                    f"{'*' if out['r2_long_accel'].get('separated') else ''}"
                    f"  speed {out['r2_speed']['delta']:+.4f}"
                    f"{'*' if out['r2_speed'].get('separated') else ''}")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=str))
    log(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
