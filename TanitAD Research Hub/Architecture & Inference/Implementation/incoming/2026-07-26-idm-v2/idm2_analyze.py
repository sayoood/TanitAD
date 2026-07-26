"""IDM-v2 -- the decision-grade comparison.

Every contrast is a PAIRED episode-cluster bootstrap over the 36 val episodes
(taniteval.ci.paired_episode_cluster_bootstrap, n_boot=2000).  Single-arm
intervals are episode_cluster_bootstrap.  overlapping_holdout_se is not used.

Writes /root/idm2/out/compare.json
"""
from __future__ import annotations
import sys
import numpy as np

sys.path.insert(0, "/root/idm2")
import idm2_lib as L  # noqa: E402

CH = {"speed": (0, "S"), "yaw_rate": (1, "S"), "steer": (2, "S"),
      "long_accel": (3, "S")}


def main():
    import glob
    P = {}
    for f in (["/root/idm2/out/preds.npy"] +
              sorted(glob.glob("/root/idm2/out/*_preds.npy"))):
        P.update(np.load(f, allow_pickle=True).item())
    print("arms loaded:", sorted(P))
    G = np.load("/root/idm2/out/val_gt.npy", allow_pickle=True).item()
    eid, dom = G["eid"], G["dom"]
    gt = G["S"].astype(np.float64)
    akin = G["Akin"].astype(np.float64)
    # ---- ADMISSIBLE-LABEL mask ------------------------------------------ #
    # 0.30 % of comma2k19 val windows carry a PHYSICALLY IMPOSSIBLE yaw_rate
    # (up to 15.5 rad/s, all at v ~ 0, from arctan2 of an undefined ENU velocity).
    # A model cannot be scored against an impossible label, so yaw is ALSO
    # reported on the admissible subset -- with the exclusion count stated.
    adm = np.abs(gt[:, 1]) <= 1.5
    out = {"n_val_windows": int(gt.shape[0]),
           "yaw_admissible": {"kept": int(adm.sum()),
                              "excluded": int((~adm).sum()),
                              "excluded_frac": float((~adm).mean()),
                              "rule": "|yaw_rate_gt| <= 1.5 rad/s (86 deg/s)"},
           "n_val_episodes": int(len(np.unique(eid))),
           "estimator": "paired_episode_cluster_bootstrap (taniteval.ci), "
                        "n_boot=2000, resampling unit = val EPISODE",
           "single_arm": {}, "paired_vs_A0": {}, "paired_vs_B0": {}}

    arms = list(P.keys())
    for arm in arms:
        e = {}
        for nm, (j, _) in CH.items():
            p = P[arm]["S"][:, j]
            g = gt[:, j]
            m = L.chan_metrics(p, g)
            m["boot_r2"] = L.boot_r2(p, g, eid)
            m["boot_mae"] = L.boot_mae(p, g, eid, reduce="mean")
            m["boot_medae"] = L.boot_mae(p, g, eid, reduce="median")
            m["per_domain"] = {}
            for d in ("pai", "cm"):
                dm = dom == d
                md = L.chan_metrics(p[dm], g[dm])
                md["boot_medae"] = L.boot_mae(p[dm], g[dm], eid[dm], reduce="median")
                m["per_domain"][d] = md
            if nm == "yaw_rate":
                m["admissible"] = L.chan_metrics(p[adm], g[adm])
                m["admissible"]["boot_r2"] = L.boot_r2(p[adm], g[adm], eid[adm])
                m["admissible"]["per_domain"] = {
                    d: L.chan_metrics(p[adm & (dom == d)], g[adm & (dom == d)])
                    for d in ("pai", "cm")}
            e[nm] = m
        # accel vs the KINEMATIC target
        p = P[arm]["S"][:, 3]
        e["long_accel_vs_kinematic"] = L.chan_metrics(p, akin)
        e["long_accel_vs_kinematic"]["per_domain"] = {
            d: L.chan_metrics(p[dom == d], akin[dom == d]) for d in ("pai", "cm")}
        out["single_arm"][arm] = e

    for ref, key in (("A0", "paired_vs_A0"), ("B0", "paired_vs_B0")):
        if ref not in P:
            continue
        for arm in arms:
            if arm == ref:
                continue
            e = {}
            for nm, (j, _) in CH.items():
                a, b = P[arm]["S"][:, j], P[ref]["S"][:, j]
                g = gt[:, j]
                e[nm] = {
                    "d_mae": L.paired_mae(a, b, g, eid, reduce="mean"),
                    "d_medae": L.paired_mae(a, b, g, eid, reduce="median"),
                    "mae_arm": float(np.abs(a - g).mean()),
                    "mae_ref": float(np.abs(b - g).mean()),
                    "r2_arm": L.chan_metrics(a, g)["r2"],
                    "r2_ref": L.chan_metrics(b, g)["r2"]}
                for d in ("pai", "cm"):
                    dm = dom == d
                    e[nm][f"d_medae_{d}"] = L.paired_mae(
                        a[dm], b[dm], g[dm], eid[dm], reduce="median")
                if nm == "yaw_rate":
                    e[nm]["d_medae_admissible"] = L.paired_mae(
                        a[adm], b[adm], g[adm], eid[adm], reduce="median")
                    e[nm]["d_mae_admissible"] = L.paired_mae(
                        a[adm], b[adm], g[adm], eid[adm], reduce="mean")
            # trajectory ADE, paired
            da = np.linalg.norm(P[arm]["Traj"] - G["Traj"], axis=-1).mean(1)
            db = np.linalg.norm(P[ref]["Traj"] - G["Traj"], axis=-1).mean(1)
            e["ade_2s"] = L.paired_mae(da, np.zeros_like(da), np.zeros_like(da), eid)
            e["ade_2s_paired_delta"] = L.paired_mae(da, db, np.zeros_like(da), eid)
            out[key][arm] = e

    # derived-steer vs regressed-steer, within each arm
    out["steer_derived_vs_regressed"] = {}
    for arm in arms:
        kap = P[arm]["S"][:, 1] / np.clip(P[arm]["S"][:, 0], 3.0, None)
        out["steer_derived_vs_regressed"][arm] = {"note":
            "derived steer is computed in idm2_v2.py with TRAIN-fit per-domain "
            "bicycle coefficients; see v2_results.json channels.steer_derived"}
    L.jdump(out, "/root/idm2/out/compare.json")

    # ---- console table ---------------------------------------------------- #
    print(f"\n{'arm':<5}{'speedR2':>9}{'speedMAE':>10}{'yawR2':>9}"
          f"{'yawMedAE':>10}{'yawNMAD':>9}{'steerR2':>9}{'accR2':>8}"
          f"{'accKinR2':>10}")
    for arm in arms:
        e = out["single_arm"][arm]
        print(f"{arm:<5}{e['speed']['r2']:>+9.4f}{e['speed']['mae']:>10.3f}"
              f"{e['yaw_rate']['r2']:>+9.4f}{e['yaw_rate']['medae']:>10.4f}"
              f"{e['yaw_rate']['nmedae']:>9.3f}{e['steer']['r2']:>+9.4f}"
              f"{e['long_accel']['r2']:>+8.4f}"
              f"{e['long_accel_vs_kinematic']['r2']:>+10.4f}")
    print("\nper-domain yaw_rate (R2 | medAE | nMedAE):")
    for arm in arms:
        e = out["single_arm"][arm]["yaw_rate"]["per_domain"]
        print(f"  {arm:<5} pai {e['pai']['r2']:+.4f} {e['pai']['medae']:.4f} "
              f"{e['pai']['nmedae']:.3f}   cm {e['cm']['r2']:+.4f} "
              f"{e['cm']['medae']:.4f} {e['cm']['nmedae']:.3f}")
    print("\nper-domain speed (R2 | MAE):")
    for arm in arms:
        e = out["single_arm"][arm]["speed"]["per_domain"]
        print(f"  {arm:<5} pai {e['pai']['r2']:+.4f} {e['pai']['mae']:.3f}   "
              f"cm {e['cm']['r2']:+.4f} {e['cm']['mae']:.3f}")
    if "A0" in P:
        print("\nPAIRED vs A0 (negative delta = v2 BETTER; separated = CI excludes 0):")
        for arm, e in out["paired_vs_A0"].items():
            for nm in ("speed", "yaw_rate"):
                d = e[nm]["d_mae"]
                print(f"  {arm:<5}{nm:<10} dMAE {d['delta']:+.4f} "
                      f"[{d['lo']:+.4f},{d['hi']:+.4f}] sep={d['separated']}")
            d = e["yaw_rate"]["d_medae_pai"]
            print(f"  {arm:<5}{'yaw@pai':<10} dMedAE {d['delta']:+.5f} "
                  f"[{d['lo']:+.5f},{d['hi']:+.5f}] sep={d['separated']}")
            d = e["ade_2s_paired_delta"]
            print(f"  {arm:<5}{'ADE@2s':<10} dADE {d['delta']:+.4f} "
                  f"[{d['lo']:+.4f},{d['hi']:+.4f}] sep={d['separated']}")


if __name__ == "__main__":
    main()
