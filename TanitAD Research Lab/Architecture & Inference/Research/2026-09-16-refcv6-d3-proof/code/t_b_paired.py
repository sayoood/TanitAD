#!/usr/bin/env python3
"""T-B / T-A read: PAIRED distance-keeping between two banked dumps, plus ADE.

    python t_b_paired.py --dump-a <A> --label-a lead --dump-b <B> --label-b rand \
                         [--arm os] [--out out.json]

⛔ ONE ESTIMATOR. The lead join, the metrics and the paired bootstrap are the harness's
own -- `refcv3_arm.lead_block_common_grid`, `refav1_arm.join_lead_block`,
`lead_metrics.distance_keeping`, `lead_metrics.paired_distance_keeping`,
`taniteval.ci.paired_episode_cluster_bootstrap`. Only the half-frame shim that
`_distance_keeping` builds inline is reproduced here (15 lines, marked), because that
function also patches a full record this script does not have.

⭐ SIGN CONVENTION. `paired_distance_keeping(dk_a, dk_b)` returns **A - B**. This script
is called with A = the LEAD-masked arm and B = the RANDOM-masked arm, so a NEGATIVE
`time_gap_min_s` delta means masking the lead makes the plan approach it MORE than an
area-matched mask elsewhere does -- the UNSAFE direction the pre-registration names.
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import sys
import tempfile

import numpy as np

REPO = os.environ.get("D3_REPO", r"C:\Users\Admin\refcv5cmp\repo")
for p in (os.path.join(REPO, "stack"), os.path.join(REPO, "taniteval"), REPO):
    if p not in sys.path:
        sys.path.insert(0, p)
LEAD_BLOCK = r"C:\Users\Admin\refcv5cmp\data\b1_eval_lead_block.npz"


def _mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def load_dump(d: str, arm: str):
    files = sorted(glob.glob(os.path.join(d, "ep*.npz")))
    if not files:
        raise SystemExit(f"{d}: no ep*.npz")
    with open(os.path.join(d, "manifest.json"), encoding="utf-8") as fh:
        man = json.load(fh)
    P, G, WS, EID = [], [], [], []
    for f in files:
        with np.load(f) as z:
            if arm not in z.files:
                raise SystemExit(f"{f}: no arm {arm!r} (has {z.files})")
            P.append(np.asarray(z[arm], dtype=np.float64))
            G.append(np.asarray(z["g"], dtype=np.float64))
            WS.append(np.asarray(z["ws"], dtype=np.int64).reshape(-1))
            EID.append(np.full(len(z["ws"]), int(z["eid"][0])))
    abl = ""
    ap = os.path.join(d, "ABLATION.txt")
    if os.path.exists(ap):
        abl = open(ap, encoding="utf-8").read().strip()
    return (files, man, np.concatenate(P), np.concatenate(G),
            np.concatenate(WS), np.concatenate(EID).astype(object), abl)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-a", required=True)
    ap.add_argument("--dump-b", required=True)
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--arm", default="os")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--restrict-changed-vs", default=None, metavar="BASELINE_DUMP",
                    help="keep ONLY the windows whose arm-A plan differs from this "
                         "baseline dump -- i.e. the windows the intervention actually "
                         "touched. ⛔ Pooling the untouched windows in DILUTES the "
                         "effect: on T-B only 885 of 4,823 windows carry a lead within "
                         "30 m, so 82 %% of the denominator is a guaranteed zero.")
    ap.add_argument("--restrict-mode", choices=("a", "both"), default="a",
                    help="'a': windows where arm A moved. 'both': windows where BOTH "
                         "arms moved -- the right set for lead-vs-random, because the "
                         "random arm cannot place an off-agent patch in every window "
                         "(147 rand_fallback) and an unmasked twin is not a control.")
    a = ap.parse_args()

    arm_mod = _mod("refcv3_arm_real", os.path.join(REPO, "taniteval", "tools",
                                                   "refcv3_arm.py"))
    ra = _mod("refav1_arm_real", os.path.join(REPO, "taniteval", "tools",
                                              "refav1_arm.py"))
    from taniteval import ci as _ci
    from taniteval import lead_metrics as lm

    fA, manA, PA, GA, wsA, eidA, ablA = load_dump(a.dump_a, a.arm)
    fB, manB, PB, GB, wsB, eidB, ablB = load_dump(a.dump_b, a.arm)
    if not (np.array_equal(wsA, wsB) and np.array_equal(GA, GB)):
        raise SystemExit("the two dumps are not on the same windows / GT — not paired")
    n_ep = len(set(eidA.tolist()))
    print(f"[paired] {a.label_a} vs {a.label_b} · arm={a.arm} · "
          f"n_win={len(wsA)} n_ep={n_ep} · ablation A={ablA!r} B={ablB!r}")

    # -- the lead join, on the grid the dump was rolled on --------------------
    dt, k = 0.5, 4
    raw_off = int(manA["corpus"]["frames"]["provider_to_raw_frame_offset"])
    view, idx, meta, info = arm_mod.lead_block_common_grid(LEAD_BLOCK, dt, k)
    if view is None:
        raise SystemExit(f"lead block unusable on this grid: {info}")
    # ---- harvested verbatim from refcv3_arm._distance_keeping (the half-frame
    #      shim; join_lead_block is told frame_of_t = identity) ---------------
    shim_dir = tempfile.mkdtemp(prefix="d3_leadjoin_")
    shim_files = []
    for f in fA:
        with np.load(f) as d:
            ws = np.asarray(d["ws"]).astype(np.int64).reshape(-1)
            v0 = np.asarray(d["v0"], dtype=np.float32).reshape(-1)
        sf = os.path.join(shim_dir, os.path.basename(f))
        np.savez(sf, ws=ws + raw_off, v0=v0)
        shim_files.append(sf)
    lead = ra.join_lead_block(shim_files, manA, view, idx, k=info["k"],
                              dt=info["dt_s"], frame_of_t=lambda t: t)
    cov = lead.pop("coverage")
    cols, dtv = info["dump_cols"], info["dt_s"]
    eid_w = np.asarray(lead["eid"], dtype=object)
    print(f"[lead] {cov['counts']} · eps OK {cov['n_episodes_ok']}/{cov['n_episodes']}")

    # -- optional restriction to the TOUCHED windows -------------------------- #
    touched = np.ones(len(wsA), dtype=bool)
    if a.restrict_changed_vs:
        _, _, PR, _, wsR, _, _ = load_dump(a.restrict_changed_vs, a.arm)
        if not np.array_equal(wsR, wsA):
            raise SystemExit("--restrict-changed-vs is not on the same windows")
        touched = (np.abs(PA - PR).reshape(len(wsA), -1).max(1) > 0)
        if a.restrict_mode == "both":
            touched &= (np.abs(PB - PR).reshape(len(wsA), -1).max(1) > 0)
        print(f"[restrict] {int(touched.sum())} of {len(touched)} windows differ from "
              f"{os.path.basename(a.restrict_changed_vs)}")

    dkA = lm.distance_keeping(PA[:, cols], lead["leads"], lead["lead_lens"],
                              lead["speeds"], dtv)
    dkB = lm.distance_keeping(PB[:, cols], lead["leads"], lead["lead_lens"],
                              lead["speeds"], dtv)
    dkG = lm.distance_keeping(GA[:, cols], lead["leads"], lead["lead_lens"],
                              lead["speeds"], dtv)
    if not touched.all():
        for _dk in (dkA, dkB):
            for _k in ("headway_min_m", "time_gap_min_s", "min_ttc_s"):
                v = np.asarray(_dk[_k], dtype=np.float64).copy()
                v[~touched] = np.nan
                _dk[_k] = v
    paired = lm.paired_distance_keeping(dkA, dkB, eid_w,
                                        names=(a.label_a, a.label_b),
                                        n_boot=a.n_boot, seed=a.seed)

    # -- ADE, paired, on ALL windows (the same estimator) ---------------------
    def ade(P):
        return np.linalg.norm(P - GA, axis=-1).mean(axis=1)
    ade_paired = _ci.paired_episode_cluster_bootstrap(
        ade(PA), ade(PB), list(eidA), n_boot=a.n_boot, seed=a.seed)
    ade_paired_touched = _ci.paired_episode_cluster_bootstrap(
        ade(PA)[touched], ade(PB)[touched], list(eidA[touched]),
        n_boot=a.n_boot, seed=a.seed) if touched.any() else None

    # -- planned deceleration + speed MAE in CLOSING windows ------------------
    # closing = the GT lead is approaching (rel_speed < 0) at t0, on the joined grid
    # `distance_keeping` returns the per-window arrays at the TOP level (it also
    # accepts a `_per_window` sub-dict from four_families; this call does not go
    # through that wrapper).
    _pw = dkG.get("_per_window", dkG)
    closing = np.asarray(_pw["min_ttc_s"], dtype=np.float64)
    closing = np.isfinite(closing) & (closing < 30.0)
    T = dtv * len(cols)

    def plan_accel(P):
        s = np.zeros(P.shape[0])
        prev = np.zeros((P.shape[0], 2))
        for j in range(P.shape[1]):
            s += np.linalg.norm(P[:, j] - prev, axis=-1)
            prev = P[:, j]
        return 2.0 * (s - lead["speeds"] * T) / (T * T)

    accA, accB, accG = plan_accel(PA[:, cols]), plan_accel(PB[:, cols]), \
        plan_accel(GA[:, cols])

    def spd_mae(P):
        return np.abs(plan_accel(P) * T - accG * T)      # |Delta v over the horizon|

    sub = {}
    for nm, m in (("closing", closing), ("all_lead", np.isfinite(
            np.asarray(_pw["headway_min_m"], dtype=np.float64)))):
        if int(m.sum()) < 5:
            sub[nm] = {"status": "UNPOWERED", "n": int(m.sum())}
            continue
        sub[nm] = {
            "n": int(m.sum()),
            "n_ep": len(set(np.asarray(eid_w)[m].tolist())),
            "planned_accel_mps2": {
                a.label_a: float(accA[m].mean()), a.label_b: float(accB[m].mean()),
                "human": float(accG[m].mean()),
                "paired_A_minus_B": _ci.paired_episode_cluster_bootstrap(
                    accA[m], accB[m], list(np.asarray(eid_w)[m]),
                    n_boot=a.n_boot, seed=a.seed)},
            "speed_err_mps_vs_human": {
                a.label_a: float(spd_mae(PA[:, cols])[m].mean()),
                a.label_b: float(spd_mae(PB[:, cols])[m].mean()),
                "paired_A_minus_B": _ci.paired_episode_cluster_bootstrap(
                    spd_mae(PA[:, cols])[m], spd_mae(PB[:, cols])[m],
                    list(np.asarray(eid_w)[m]), n_boot=a.n_boot, seed=a.seed)}}

    def means(dk):
        return {
            "mean_headway_min_m": dk.get("mean_headway_min_m"),
            "mean_time_gap_min_s": dk.get("mean_time_gap_min_s"),
            "mean_min_ttc_s": dk.get("mean_min_ttc_s"),
            "n": dk.get("n"), "n_time_gap": dk.get("n_time_gap"),
            "n_closing": dk.get("n_closing")}

    out = {
        "_what": f"paired {a.label_a} - {a.label_b}; arm {a.arm}",
        "_sign": ("NEGATIVE time_gap / headway / ttc delta = the LEAD-masked arm "
                  "approaches the lead MORE than the area-matched random mask: "
                  "the UNSAFE direction"),
        "tier": "T1 (self-action OPEN loop)",
        "n_windows": int(len(wsA)), "n_episodes": int(n_ep),
        "dumps": {a.label_a: {"dir": a.dump_a, "ablation": ablA},
                  a.label_b: {"dir": a.dump_b, "ablation": ablB}},
        "lead_join": {"coverage": cov, "grid": info,
                      "block_sha256": ra._sha256(LEAD_BLOCK)},
        "per_arm": {a.label_a: means(dkA), a.label_b: means(dkB),
                    "gt_reference": means(dkG)},
        "paired_distance_keeping": paired,
        "paired_ade_m": ade_paired,
        "paired_ade_m_touched_windows": ade_paired_touched,
        "n_touched_windows": int(touched.sum()),
        "restrict_changed_vs": a.restrict_changed_vs,
        "restrict_mode": a.restrict_mode,
        "longitudinal_subsets": sub,
        "estimator": ("lead_metrics.distance_keeping + "
                      "lead_metrics.paired_distance_keeping + "
                      "taniteval.ci.paired_episode_cluster_bootstrap; "
                      f"n_boot={a.n_boot} seed={a.seed}; clusters = clips"),
    }
    txt = json.dumps(out, indent=1, default=str)
    print(json.dumps({
        "n_touched": int(touched.sum()),
        "ade_A_minus_B_touched": ade_paired_touched,
        "ade_A_minus_B": ade_paired,
        **{k: {kk: v[kk] for kk in ("delta", "lo", "hi", "separated", "n_used")
               if kk in v}
           for k, v in paired["metrics"].items()}}, indent=1, default=str))
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as fh:
            fh.write(txt)
        print("[out]", a.out)


if __name__ == "__main__":
    main()
