"""IDM v3 — scoring. Every contrast is a PAIRED episode-cluster bootstrap on the
identical 4,195 val windows (B = 2000, unit = the 36 val episodes).
`overlapping_holdout_se` is never called.

Produces, in order of the brief's priority:
  1. LABEL FIX before/after — every channel, per corpus, for the DEPLOYED head
     and for a retrained recipe control.
  2. The comma.ai protocol gate (their own `calib_challenge` discards frames
     below 4 m/s) scored as a separate, clearly-labelled protocol variant.
  3. GEOMETRY CONDITIONING vs its three controls — corpus one-hot, rig one-hot,
     shuffled geometry.
  4. The PRE-REGISTERED DISCRIMINATOR: geometry may help SPEED (which scales
     with f*h, unmatched across clips) and must NOT help YAW (which scales with
     f alone, already canonicalised to ~266 everywhere). A method that helps
     both equally is reading the corpus, not the geometry.
  5. long_accel as classification vs regression.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, "/root/idm2")
sys.path.insert(0, "/root/taniteval")
sys.path.insert(0, "/root/v4eval/stack")
sys.path.insert(0, "/root/v4eval/stack/scripts")

import idm2_lib as L                 # noqa: E402
from taniteval import ci as tci      # noqa: E402

CH = {"speed": 0, "yaw_rate": 1, "steer": 2, "long_accel": 3}
OUT = Path("/workspace/idm3/out")


def paired(pa, pb, g, eid):
    """delta = MAE(a) - MAE(b). Negative => a better."""
    a = np.abs(pa - g)
    b = np.abs(pb - g)
    d = tci.paired_episode_cluster_bootstrap(a, b, eid, n_boot=2000, seed=0,
                                             reduce="mean")
    return {"delta_mae": float(d["delta"]), "lo": float(d["lo"]),
            "hi": float(d["hi"]),
            "separated": bool(d["lo"] > 0 or d["hi"] < 0),
            "mae_a": float(a.mean()), "mae_b": float(b.mean())}


def r2ci(p, g, eid):
    d = L.boot_r2(p, g, eid, n_boot=2000, seed=0)
    return {"r2": float(1 - ((p - g) ** 2).sum() / max(((g - g.mean()) ** 2).sum(), 1e-12)),
            "lo": float(d["lo"]), "hi": float(d["hi"])}


def chan_block(p, g, eid, dom, mask=None):
    if mask is None:
        mask = np.ones(len(g), bool)
    out = {"pooled": {**L.chan_metrics(p[mask], g[mask]),
                      **{"r2_ci": r2ci(p[mask], g[mask], eid[mask])}}}
    for d in ("pai", "cm"):
        m = mask & (dom == d)
        out[d] = {**L.chan_metrics(p[m], g[m]),
                  **{"r2_ci": r2ci(p[m], g[m], eid[m])}}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default=str(OUT / "arms_v3_preds.npy"))
    ap.add_argument("--extra", default="")
    ap.add_argument("--out", default=str(OUT / "compare_v3.json"))
    a = ap.parse_args()

    P = np.load(a.preds, allow_pickle=True).item()
    if a.extra and Path(a.extra).exists():
        P.update(np.load(a.extra, allow_pickle=True).item())
    gtd = np.load(OUT / "val_gt_v3.npy", allow_pickle=True).item()
    G, Gleg = gtd["S"], gtd["S_leg"]
    Akin, eid, dom = gtd["Akin"], gtd["eid"], gtd["dom"]
    vspeed = G[:, 0]
    res = {"arms_present": sorted(P), "n_windows": int(G.shape[0]),
           "n_episodes": int(len(set(eid)))}

    # ---- A0 under both label sets -----------------------------------------
    a0 = np.load(OUT / "a0_preds.npy", allow_pickle=True).item()["S"]
    res["LABEL_FIX_deployed_head"] = {}
    for nm, j in CH.items():
        blk = {"legacy": chan_block(a0[:, j], Gleg[:, j], eid, dom),
               "repaired": chan_block(a0[:, j], G[:, j], eid, dom)}
        if nm == "yaw_rate":
            blk["legacy_admissible_only"] = chan_block(
                a0[:, j], Gleg[:, j], eid, dom, mask=np.abs(Gleg[:, j]) <= 1.5)
            # comma.ai's OWN calib_challenge protocol: discard frames < 4 m/s
            blk["repaired_comma_4mps_gate"] = chan_block(
                a0[:, j], G[:, j], eid, dom, mask=vspeed >= 4.0)
            blk["legacy_comma_4mps_gate"] = chan_block(
                a0[:, j], Gleg[:, j], eid, dom, mask=vspeed >= 4.0)
        res["LABEL_FIX_deployed_head"][nm] = blk
    ch_m = np.abs(G[:, 1] - Gleg[:, 1]) > 1e-9
    res["label_defect_extent"] = {
        "n_windows_changed_by_repair": int(ch_m.sum()),
        "frac": float(ch_m.mean()),
        "n_impossible_legacy": int((np.abs(Gleg[:, 1]) > 1.5).sum()),
        "n_impossible_repaired": int((np.abs(G[:, 1]) > 1.5).sum()),
        "n_below_4mps": int((vspeed < 4.0).sum()),
    }

    # ---- AUDIT: is the repair a real correction or a statistical artifact? --
    # R2 has the label's own variance in its denominator, so shrinking a wild
    # label can flatter R2 without the model getting any better. Three checks:
    #   (a) the UNCHANGED windows must be bit-identical (isolation);
    #   (b) on the CHANGED windows, show what the label was, what it became, and
    #       what the model said — if the model was already predicting ~0 while
    #       the label said +-15 rad/s at a standstill, the MODEL was right;
    #   (c) outlier-proof statistics (medAE, nMedAE, Spearman) must move the
    #       same way as R2, or the gain is an R2-denominator artifact.
    unch = ~ch_m
    res["REPAIR_AUDIT"] = {
        "a_unchanged_windows": {
            "n": int(unch.sum()),
            "max_abs_label_delta": float(np.abs(G[unch, 1] - Gleg[unch, 1]).max()),
            "identical": bool(np.array_equal(G[unch, 1], Gleg[unch, 1])),
        },
        "a_physicalai_untouched": {
            "n_pai_changed": int((ch_m & (dom == "pai")).sum()),
            "must_be_zero": True,
        },
        "b_changed_windows": {
            "n": int(ch_m.sum()),
            "legacy_label_absmax": float(np.abs(Gleg[ch_m, 1]).max()),
            "legacy_label_absmean": float(np.abs(Gleg[ch_m, 1]).mean()),
            "repaired_label_absmax": float(np.abs(G[ch_m, 1]).max()),
            "repaired_label_absmean": float(np.abs(G[ch_m, 1]).mean()),
            "A0_pred_absmean": float(np.abs(a0[ch_m, 1]).mean()),
            "A0_mae_vs_legacy": float(np.abs(a0[ch_m, 1] - Gleg[ch_m, 1]).mean()),
            "A0_mae_vs_repaired": float(np.abs(a0[ch_m, 1] - G[ch_m, 1]).mean()),
            "gt_speed_at_these_windows_max": float(vspeed[ch_m].max()),
            "gt_speed_at_these_windows_mean": float(vspeed[ch_m].mean()),
            "reading": "a road vehicle at v~0 has yaw_rate ~0; if A0_pred_absmean "
                       "is small while legacy_label_absmax is huge, the MODEL was "
                       "right and the LABEL was wrong",
        },
        "c_outlier_proof": {
            k: {"legacy": {m: res["LABEL_FIX_deployed_head"]["yaw_rate"]["legacy"][k][m]
                           for m in ("medae", "nmedae", "rho", "mae")},
                "repaired": {m: res["LABEL_FIX_deployed_head"]["yaw_rate"]["repaired"][k][m]
                             for m in ("medae", "nmedae", "rho", "mae")}}
            for k in ("pooled", "pai", "cm")},
    }

    # ---- per-arm headline --------------------------------------------------
    res["arms"] = {}
    for arm, p in P.items():
        gt = Gleg if arm.endswith("LEG") else G
        e = {}
        for nm, j in CH.items():
            e[nm] = chan_block(p["S"][:, j], gt[:, j], eid, dom)
        e["long_accel_vs_kinematic"] = chan_block(p["S"][:, 3], Akin, eid, dom)
        # every arm ALSO scored against the repaired labels, so LEG is comparable
        e["yaw_rate_on_repaired"] = chan_block(p["S"][:, 1], G[:, 1], eid, dom)
        if "acc_cls" in p:
            e["long_accel_BINNED"] = chan_block(p["acc_cls"], gt[:, 3], eid, dom)
            e["long_accel_BINNED_vs_kinematic"] = chan_block(p["acc_cls"], Akin,
                                                            eid, dom)
        res["arms"][arm] = e

    # ---- the contrasts -----------------------------------------------------
    def cmp_block(a_, b_, chans=("speed", "yaw_rate", "steer", "long_accel")):
        if a_ not in P or b_ not in P:
            return None
        o = {}
        for nm in chans:
            j = CH[nm]
            o[nm] = {"pooled": paired(P[a_]["S"][:, j], P[b_]["S"][:, j],
                                      G[:, j], eid)}
            for d in ("pai", "cm"):
                m = dom == d
                o[nm][d] = paired(P[a_]["S"][m, j], P[b_]["S"][m, j],
                                  G[m, j], eid[m])
        return o

    res["CONTRASTS"] = {}
    pairs = [
        # label layer
        ("R0", "R0LEG", "label repair, same recipe"),
        # geometry WITHOUT clip-context — the informative layer
        ("G1n", "R0", "GEOMETRY token vs nothing"),
        ("G1h", "R0", "camera-height ONLY vs nothing"),
        ("Ccorpn", "R0", "CONTROL corpus one-hot vs nothing"),
        ("Crign", "R0", "CONTROL rig one-hot vs nothing"),
        ("Cshufn", "R0", "NEGATIVE CONTROL shuffled geometry vs nothing"),
        ("Sctxn", "R0", "clip-context (the v2 lever) vs nothing"),
        ("G1n", "Ccorpn", "geometry vs the corpus-embedding CONTROL"),
        ("G1n", "Crign", "geometry vs the rig CONTROL"),
        ("G1n", "Cshufn", "geometry vs the SHUFFLED control"),
        ("G1n", "Sctxn", "geometry vs clip-context"),
        # geometry ON TOP of clip-context
        ("G1", "V2R", "geometry on top of clip-context"),
        ("Ccorp", "V2R", "CONTROL corpus one-hot on top of clip-context"),
        ("Cshuf", "V2R", "NEGATIVE CONTROL shuffled on top of clip-context"),
        ("G1", "Cshuf", "geometry vs shuffled, both on clip-context"),
        # physics arm (pre-registered to FAIL)
        ("G2", "V2R", "PHYSICS v=(f*h)*PHI vs plain regression"),
        # overall
        ("V2R", "R0", "v2 recipe vs v1 recipe, both on repaired labels"),
    ]
    for x, y, why in pairs:
        c = cmp_block(x, y)
        if c:
            res["CONTRASTS"][f"{x}_vs_{y}"] = {"why": why, **c}

    # ---- the pre-registered discriminator ---------------------------------
    disc = {}
    for arm in ("G1n", "G1h", "Ccorpn", "Crign", "Cshufn", "Sctxn"):
        c = res["CONTRASTS"].get(f"{arm}_vs_R0")
        if not c:
            continue
        sd_, yd_ = c["speed"]["pooled"], c["yaw_rate"]["pooled"]
        # delta = MAE(arm) - MAE(R0); NEGATIVE means the arm is BETTER.
        # (Bug fixed 2026-07-27: the first version tested `separated` without
        # the SIGN, so it labelled arms that were significantly WORSE as
        # "uses geometry". The underlying CIs were correct; only this verdict
        # string was wrong. Recorded rather than silently corrected.)
        s_better = sd_["separated"] and sd_["delta_mae"] < 0
        s_worse = sd_["separated"] and sd_["delta_mae"] > 0
        y_better = yd_["separated"] and yd_["delta_mae"] < 0
        disc[arm] = {
            "speed_delta_mae": sd_["delta_mae"],
            "speed_separated": sd_["separated"],
            "yaw_delta_mae": yd_["delta_mae"],
            "yaw_separated": yd_["separated"],
            "VERDICT": ("SIGNIFICANTLY WORSE than no conditioning" if s_worse
                        else "helps both -> reads the corpus, not the geometry"
                        if s_better and y_better
                        else "USES GEOMETRY (speed improves, yaw does not)"
                        if s_better else "no separated speed effect"),
        }
    res["PREREGISTERED_DISCRIMINATOR"] = {
        "rule": "geometry must help SPEED (scales with f*h, unmatched per clip) "
                "and must NOT help YAW (scales with f alone, already ~266 on "
                "every corpus). Helping both equally = corpus memorisation.",
        "arms": disc}

    L.jdump(res, a.out)

    # ---- console summary ---------------------------------------------------
    print("\n" + "=" * 78)
    print("LABEL FIX — the DEPLOYED head, yaw_rate, nothing retrained")
    y = res["LABEL_FIX_deployed_head"]["yaw_rate"]
    for k in ("legacy", "legacy_admissible_only", "repaired",
              "legacy_comma_4mps_gate", "repaired_comma_4mps_gate"):
        if k in y:
            print("  %-26s pooled R2 %+.4f   pai %+.4f   cm %+.4f   n=%d"
                  % (k, y[k]["pooled"]["r2"], y[k]["pai"]["r2"],
                     y[k]["cm"]["r2"], y[k]["pooled"]["n"]))
    print("\nARM HEADLINES (speed R2 / yaw R2 / steer R2 / accel R2), per corpus")
    print("  %-8s %-22s %-22s %-16s %s" % ("arm", "speed pool/pai/cm",
                                           "yaw pool/pai/cm", "steer p/c", "accel"))
    for arm in sorted(res["arms"]):
        e = res["arms"][arm]
        print("  %-8s %+.3f/%+.3f/%+.3f    %+.3f/%+.3f/%+.3f    %+.3f/%+.3f   %+.3f"
              % (arm, e["speed"]["pooled"]["r2"], e["speed"]["pai"]["r2"],
                 e["speed"]["cm"]["r2"], e["yaw_rate"]["pooled"]["r2"],
                 e["yaw_rate"]["pai"]["r2"], e["yaw_rate"]["cm"]["r2"],
                 e["steer"]["pai"]["r2"], e["steer"]["cm"]["r2"],
                 e["long_accel"]["pooled"]["r2"]))
    print("\nCONTRASTS — paired dMAE [95% CI], negative = first arm better")
    for k, c in res["CONTRASTS"].items():
        s, yv = c["speed"]["pooled"], c["yaw_rate"]["pooled"]
        print("  %-18s speed %+.4f [%+.4f,%+.4f]%s | yaw %+.5f [%+.5f,%+.5f]%s  (%s)"
              % (k, s["delta_mae"], s["lo"], s["hi"], " *" if s["separated"] else "  ",
                 yv["delta_mae"], yv["lo"], yv["hi"], " *" if yv["separated"] else "  ",
                 c["why"]))
    print("\nPRE-REGISTERED DISCRIMINATOR")
    for arm, d in disc.items():
        print("  %-8s %s" % (arm, d["VERDICT"]))


if __name__ == "__main__":
    main()
