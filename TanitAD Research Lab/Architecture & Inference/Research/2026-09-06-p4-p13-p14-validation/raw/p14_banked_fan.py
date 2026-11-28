"""P14 -- THE SAMPLER RANKS A FAN IT HAS NOT SAMPLED. Banked-fan validation.

ZERO GPU. Executes PREREG.md section 1 exactly:
  B1  ceiling      ADE(shipped) - ADE(fan-best), paired episode-cluster CI
  B2  >2x share    fraction of windows where fan-best beats shipped by >2x
  B3  vacuity      manoeuvre rate beside every number
  R   regression   rank_shuffled MUST be separated-worse or the gate is VOID
  controls         rank_shipped reproduces `sel` exactly (n_mismatch == 0)
                   rank_oracle == per-window min by construction
                   cv = raw-input floor ; rank_constant = no-information
  families         ADE / LONGITUDINAL / LATERAL(curvature MAE + straight floor)
                   / TACTICAL(agreement + manoeuvre rate) ; STRATEGIC = N/A+reason

ASCII ONLY in print() -- cp1252 dev box.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import torch

sys.path.insert(0, "taniteval")
from taniteval.ci import paired_episode_cluster_bootstrap  # noqa: E402

DT = 0.1  # 10 Hz corpus; wp_steps are frame indices


# ------------------------------------------------------------------ metrics #
def path_with_origin(p):
    """Prepend the ego origin so a 4-waypoint path has 5 points and a
    curvature can be formed at all. [..., S, 2] -> [..., S+1, 2]."""
    z = np.zeros(p.shape[:-2] + (1, 2), dtype=p.dtype)
    return np.concatenate([z, p], axis=-2)


def curvature(p, ts):
    """Menger-free finite-difference curvature along a path. [..., S+1, 2]
    with sample times `ts` -> [..., S-1] curvature in 1/m.

    kappa = |x' y'' - y' x''| / (x'^2 + y'^2)^{3/2}
    """
    d1 = np.gradient(p, ts, axis=-2)
    d2 = np.gradient(d1, ts, axis=-2)
    num = np.abs(d1[..., 0] * d2[..., 1] - d1[..., 1] * d2[..., 0])
    den = np.power(d1[..., 0] ** 2 + d1[..., 1] ** 2, 1.5)
    return num / np.maximum(den, 1e-6)


def families(traj, gt, ts):
    """Per-window metric families for ONE selected trajectory per window.
    traj/gt [N, S, 2] -> dict of [N] arrays."""
    err = traj - gt
    ade = np.linalg.norm(err, axis=-1).mean(axis=-1)
    along = np.abs(err[..., 0]).mean(axis=-1)          # LONGITUDINAL
    cross = np.abs(err[..., 1]).mean(axis=-1)          # LATERAL (cross-track)
    kt = curvature(path_with_origin(traj), ts)
    kg = curvature(path_with_origin(gt), ts)
    kmae = np.abs(kt - kg).mean(axis=-1)               # LATERAL (curvature)
    # terminal speed, from the last inter-waypoint step
    sp = np.linalg.norm(traj[:, -1] - traj[:, -2], axis=-1) / (ts[-1] - ts[-2])
    spg = np.linalg.norm(gt[:, -1] - gt[:, -2], axis=-1) / (ts[-1] - ts[-2])
    manoeuvre = (np.abs(traj[:, -1, 1]) > 1.0).astype(np.float64)
    return {"ade": ade, "along": along, "cross": cross, "kmae": kmae,
            "speed_err": np.abs(sp - spg), "manoeuvre": manoeuvre}


def fan_ade(fan, gt):
    """[N, K] ADE of every candidate."""
    return np.linalg.norm(fan - gt[:, None], axis=-1).mean(axis=-1)


# -------------------------------------------------------------------- main #
def run(path, seed=0, out=None):
    d = torch.load(path, map_location="cpu", weights_only=False)
    fan = d["fan"].float().numpy()          # [N, K, S, 2]
    gt = d["gt"].float().numpy()            # [N, S, 2]
    cv = d["cv"].float().numpy()            # [N, S, 2]  raw-input floor
    logits = d["logits"].float().numpy()    # [N, K]
    sel = d["sel"].numpy()                  # [N]
    eid = list(d["eid"])
    wp = list(d["wp_steps"])
    ts = np.array([w * DT for w in wp], dtype=np.float64)
    ts0 = np.concatenate([[0.0], ts])
    n, k = fan.shape[0], fan.shape[1]

    print("=" * 78)
    print("P14 BANKED-FAN VALIDATION -- %s" % path)
    print("=" * 78)
    print("n (windows) = %d   d (fan width K) = %d   episodes = %d"
          % (n, k, len(set(eid))))
    print("waypoint frames %s -> times %s s (horizon %.1f s)"
          % (wp, list(ts), ts[-1]))
    print("ckpt = %s   step = %s   decoder steps = %s"
          % (d.get("ckpt"), d.get("ckpt_step"), d.get("steps")))

    # ---- CONTROL 1: my harness must reproduce the shipped decision ------- #
    ship_idx = logits.argmax(axis=1)
    n_mis = int((ship_idx != sel).sum())
    print("")
    print("[CONTROL rank_shipped] argmax(logits) vs dumped sel: n_mismatch = "
          "%d / %d  -> %s" % (n_mis, n, "PASS" if n_mis == 0 else "FAIL"))
    if n_mis:
        raise SystemExit("VOID: harness does not reproduce the shipped "
                         "ranking; no re-ranking result is admissible.")

    # ---- the selectors ---------------------------------------------------- #
    a_all = fan_ade(fan, gt)                                   # [N, K]
    rng = np.random.default_rng(seed)
    shuf = np.stack([rng.permutation(logits[i]) for i in range(n)])
    sels = {
        "rank_shipped": ship_idx,
        "rank_oracle": a_all.argmin(axis=1),
        "rank_constant": np.zeros(n, dtype=np.int64),
        "rank_shuffled": shuf.argmax(axis=1),
    }

    # ---- CONTROL 2: oracle == per-window minimum, by construction --------- #
    o = a_all[np.arange(n), sels["rank_oracle"]]
    assert np.allclose(o, a_all.min(axis=1)), "oracle is not the fan minimum"
    print("[CONTROL rank_oracle] equals per-window fan minimum: PASS "
          "(max |d| = %.3e)" % float(np.abs(o - a_all.min(axis=1)).max()))

    # ---- the four metric families, per selector + the CV floor ------------ #
    rows = {}
    for name, idx in sels.items():
        rows[name] = families(fan[np.arange(n), idx], gt, ts0)
    rows["cv_floor"] = families(cv, gt, ts0)          # RAW-INPUT FLOOR
    # the straight-line floor for LATERAL: a plan that never steers
    straight = np.zeros_like(gt)
    straight[..., 0] = gt[..., 0]                    # same along-track, y == 0
    rows["straight_floor"] = families(straight, gt, ts0)

    print("")
    print("-" * 78)
    print("FOUR METRIC FAMILIES -- per family, NEVER pooled.  n = %d" % n)
    print("-" * 78)
    hdr = ("%-16s %9s %9s %9s %11s %9s %9s"
           % ("arm", "ADE m", "ALONG m", "CROSS m", "kappaMAE", "spdErr", "man.rate"))
    print(hdr)
    print("%-16s %9s %9s %9s %11s %9s %9s"
          % ("", "(accomp.)", "LONGIT", "LATERAL", "LATERAL", "LONGIT", "VACUITY"))
    for name in ("rank_oracle", "rank_shipped", "rank_shuffled",
                 "rank_constant", "cv_floor", "straight_floor"):
        r = rows[name]
        print("%-16s %9.4f %9.4f %9.4f %11.5f %9.4f %8.2f%%"
              % (name, r["ade"].mean(), r["along"].mean(), r["cross"].mean(),
                 r["kmae"].mean(), r["speed_err"].mean(),
                 100.0 * r["manoeuvre"].mean()))
    print("STRATEGIC: N/A -- the banked fan carries no route/goal label. n = 0.")

    # ---- B1: the ceiling, paired episode-cluster bootstrap ---------------- #
    print("")
    print("-" * 78)
    print("BAR P14-B1 -- THE CEILING (paired episode-cluster bootstrap)")
    print("-" * 78)
    res = {}
    for fam in ("ade", "along", "cross", "kmae", "speed_err"):
        r = paired_episode_cluster_bootstrap(
            rows["rank_shipped"][fam], rows["rank_oracle"][fam], eid, seed=seed)
        res[fam] = r
        print("%-10s shipped-minus-oracle = %+.4f   95%% CI [%+.4f, %+.4f]   "
              "separated=%s" % (fam, r["delta"], r["lo"], r["hi"],
                                r["separated"]))
    gap = res["ade"]["delta"]
    b1 = (gap >= 0.05) and bool(res["ade"]["separated"])
    print("")
    print("BAR: PASS if ADE gap >= 0.05 m AND CI excludes zero.")
    print("MEASURED gap = %.4f m ; separated = %s  ==>  %s"
          % (gap, res["ade"]["separated"], "PASS" if b1 else "FAIL"))

    # ---- B2: the >2x share ------------------------------------------------ #
    a_ship = a_all[np.arange(n), ship_idx]
    a_orc = a_all[np.arange(n), sels["rank_oracle"]]
    share2 = float((a_ship > 2.0 * np.maximum(a_orc, 1e-9)).mean())
    print("")
    print("-" * 78)
    print("BAR P14-B2 -- IS THE DEFECT RANKING, OR FAN QUALITY?")
    print("-" * 78)
    print("windows where fan-best beats shipped by >2x: %.2f%% (n = %d)"
          % (100.0 * share2, n))
    print("BAR: PASS if > 25%%  ==>  %s" % ("PASS" if share2 > 0.25 else "FAIL"))
    print("  (the fan CONTAINS a >2x-better trajectory this often, and the "
          "shipped ranking does not pick it -- that is recoverable by ranking, "
          "not by a better fan.)")

    # ---- R: the deliberate-regression arm --------------------------------- #
    print("")
    print("-" * 78)
    print("ARM P14-R -- DELIBERATE REGRESSION (rank_shuffled)")
    print("-" * 78)
    rr = paired_episode_cluster_bootstrap(
        rows["rank_shuffled"]["ade"], rows["rank_shipped"]["ade"], eid, seed=seed)
    valid = bool(rr["separated"]) and rr["delta"] > 0
    print("shuffled-minus-shipped ADE = %+.4f   95%% CI [%+.4f, %+.4f]   "
          "separated = %s" % (rr["delta"], rr["lo"], rr["hi"], rr["separated"]))
    print("GATE VALIDITY: the instrument must see a knowingly-broken ranking.")
    print("  ==> %s" % ("VALID -- a broken ranking IS detected, so a PASS above "
                        "means something" if valid else
                        "VOID -- the gate cannot fail a shuffled ranking; every "
                        "P14 number here is inadmissible"))

    # ---- B3: the vacuity gate --------------------------------------------- #
    print("")
    print("-" * 78)
    print("BAR P14-B3 -- VACUITY GATE (manoeuvre rate beside every number)")
    print("-" * 78)
    m_ship = rows["rank_shipped"]["manoeuvre"].mean()
    for name in ("rank_oracle", "rank_shipped", "rank_shuffled",
                 "rank_constant", "cv_floor"):
        m = rows[name]["manoeuvre"].mean()
        flag = ""
        if name != "rank_shipped":
            flag = ("  <-- BELOW 50%% OF SHIPPED (VACUOUS)"
                    if m < 0.5 * m_ship else "")
        print("  %-16s manoeuvre rate %6.2f%%%s" % (name, 100.0 * m, flag))
    print("GT manoeuvre rate (|y_terminal| > 1 m): %.2f%%"
          % (100.0 * (np.abs(gt[:, -1, 1]) > 1.0).mean()))

    # ---- TACTICAL family: agreement --------------------------------------- #
    print("")
    print("-" * 78)
    print("TACTICAL FAMILY -- selection agreement (n = %d, d = %d)" % (n, k))
    print("-" * 78)
    print("  shipped picks the fan-best anchor : %.2f%%"
          % (100.0 * float((ship_idx == sels["rank_oracle"]).mean())))
    print("  shipped is in the fan-best TOP-5  : %.2f%%"
          % (100.0 * float(
              (np.argsort(a_all, axis=1)[:, :5] ==
               ship_idx[:, None]).any(axis=1).mean())))
    print("  shuffled picks the fan-best anchor: %.2f%%  (no-information "
          "reference = 1/K = %.2f%%)"
          % (100.0 * float((sels["rank_shuffled"] == sels["rank_oracle"]).mean()),
             100.0 / k))

    if out:
        blob = {"path": path, "n": n, "K": k, "episodes": len(set(eid)),
                "b1_ade_gap": gap, "b1_ci": [res["ade"]["lo"], res["ade"]["hi"]],
                "b1_separated": bool(res["ade"]["separated"]), "b1_pass": b1,
                "b2_share_2x": share2, "b2_pass": share2 > 0.25,
                "regression_valid": valid,
                "regression_delta": rr["delta"],
                "means": {a: {f: float(v.mean()) for f, v in r.items()}
                          for a, r in rows.items()},
                "families_paired_vs_oracle": {
                    f: {"delta": r["delta"], "lo": r["lo"], "hi": r["hi"],
                        "separated": bool(r["separated"])}
                    for f, r in res.items()}}
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(blob, fh, indent=2)
        print("")
        print("[banked] %s" % out)
    return b1, share2 > 0.25, valid


if __name__ == "__main__":
    run(sys.argv[1], out=(sys.argv[2] if len(sys.argv) > 2 else None))
