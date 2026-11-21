#!/usr/bin/env python3
"""Close the LONGITUDINAL family's DISTANCE-KEEPING sub-block on the gated path.

⛔ A missing metric is a WORK ITEM, not an excuse (binding, PI 2026-08-02, clause 3), and
distance keeping is the half of LONGITUDINAL that ADE cannot see: 88.7 % of the programme's
oracle gap is longitudinal, and an arm can win ADE while setting the wrong speed or closing
on the lead.

`kin_gate_eval.py` reports this sub-block UNAVAILABLE because it passes `lead=None`. This
script supplies it from the SAME banked tensors -- headway, time gap and min-TTC per rule --
so the family is complete rather than complete-with-an-excuse. Zero GPU, zero model calls.

⚠️ THE MASK IS LOAD-BEARING. `rl_refcv3_min.lead_track` returns a FAR SENTINEL
(FAR_LEAD_X_M held over the horizon) for a window with no lead, so scoring every window would
compute a headway to a vehicle that is not there. Only `has_lead` windows are scored, and the
n is printed beside every number.

⚠️ THE GRID JOIN IS ALSO LOAD-BEARING. The banked lead track is on the 5-point reward grid
(0, 0.5, 1.0, 1.5, 2.0 s), which is exactly the path grid used here, so dt = 0.5 s and
path_steps = 0..4. TTC scales as 1/dt through the closing rate; headway and time gap do not.
ASCII-only output.
"""
import argparse
import importlib.util
import json
import os
import sys

import numpy as np
import torch

_REPO = os.environ.get("TANITAD_REPO") or "C:/Users/Admin/refcv4b_repo"


def _load_driver():
    p = os.path.join(_REPO, "stack", "scripts", "rl_refcv3_min.py")
    spec = importlib.util.spec_from_file_location("_rl_refcv3_min_lead", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


D = _load_driver()
RW = D.RW
FS = D.FS
import taniteval.ci as CI                                              # noqa: E402
from taniteval.lead_metrics import distance_keeping                    # noqa: E402

GATE_KS = (1, 2, 4, 8, 16, 32, 128)
KEYS = ("headway_min_m", "time_gap_min_s", "min_ttc_s")
MEANK = {"headway_min_m": "mean_headway_min_m",
         "time_gap_min_s": "mean_time_gap_min_s",
         "min_ttc_s": "mean_min_ttc_s"}


def pboot(a, b, e, *, n_boot, seed):
    a, b, e = np.asarray(a, float), np.asarray(b, float), np.asarray(e)
    ok = np.isfinite(a) & np.isfinite(b)
    if int(ok.sum()) < 3:
        return {"delta": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "separated": False, "n_windows": int(ok.sum())}
    r = CI.paired_episode_cluster_bootstrap(a[ok], b[ok], e[ok], n_boot=n_boot, seed=seed)
    return {"delta": float(r["delta"]), "lo": float(r["lo"]), "hi": float(r["hi"]),
            "separated": bool(r["separated"]), "n_windows": int(r["n_windows"]),
            "n_episodes": int(r["n_episodes"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()

    z = np.load(a.npz)
    bk = {k: torch.from_numpy(z[k]) for k in z.files}
    W, N = bk["fan8"].shape[0], bk["fan8"].shape[1]
    NS = D.N_REWARD_SLOTS
    has = bk["has_lead"].bool().numpy()
    eid = bk["eid"].numpy()

    fan2 = D.with_origin(bk["fan8"][..., :NS, :])
    b = {"v0": bk["v0"], "lead_track": bk["lead5"], "lead_xy": bk["lead_xy"]}
    ctx = D.reward_ctx(b, S5=fan2.shape[-2], cand_dims=1)
    r_kin = (RW.COMPONENTS["feasibility"](fan2, ctx) + RW.COMPONENTS["comfort"](fan2, ctx))
    # THE RANKING THE DEPLOYED MODEL ACTUALLY USES: on the `hier` arm refc.py:1763
    # argmaxes `sel_score_v3` (goal-seam-grafted) masked by reach_keep, NOT `sel_score`.
    _rk = "sel_score_v3" if "sel_score_v3" in bk else "sel_score"
    rank = bk[_rk].float().masked_fill(~bk["reach"].bool(), float("-inf"))
    order = rank.argsort(dim=1, descending=True)

    pick = {"model": bk["sel_idx"].long()}
    for k in GATE_KS:
        idx = torch.empty(W, dtype=torch.long)
        for j in range(W):
            cand = order[j, :min(k, N)]
            idx[j] = cand[r_kin[j][cand].argmax()]
        pick["gate%d" % k] = idx
    pick["kin_only"] = r_kin.argmax(dim=1)
    ade = (bk["fan8"][..., :NS, :] - bk["gt4"][:, None]).norm(dim=-1).mean(dim=-1)
    pick["oracle"] = ade.argmin(dim=1)

    # the human's own logged path, as the reference the arms are read against
    ar = torch.arange(W)
    paths = {r: D.with_origin(bk["fan8"][ar, i][:, :NS]).numpy()
             for r, i in pick.items()}
    paths["human_gt"] = D.with_origin(bk["gt4"]).numpy()

    leads = bk["lead5"].numpy().astype(np.float64).copy()      # [W, 5, 2]
    leads[~has] = np.nan                                       # ⛔ the sentinel is NOT a lead
    lead_lens = np.full(W, float(D.LEAD_LEN_DEFAULT_M))
    lead_lens[~has] = np.nan
    speeds = bk["v0"].numpy().astype(np.float64)

    res, per = {}, {}
    for rule, p in paths.items():
        dk = distance_keeping(p, leads, lead_lens, speeds, 0.5)
        res[rule] = {k: dk.get(MEANK[k]) for k in KEYS}
        res[rule]["n"] = int(dk.get("n", 0))
        res[rule]["n_closing"] = dk.get("n_closing")
        res[rule]["n_time_gap"] = dk.get("n_time_gap")
        res[rule]["status"] = dk.get("status", "OK")
        res[rule]["censoring_note"] = dk.get("censoring_note")
        res[rule]["gap_convention"] = dk.get("gap_convention")
        pw = dk.get("_per_window", dk)
        per[rule] = {k: np.asarray(pw.get(k, [np.nan] * W), dtype=float) for k in KEYS}

    paired = {}
    for rule in [r for r in paths if r not in ("model",)]:
        paired[rule] = {k: pboot(per[rule][k], per["model"][k], eid,
                                 n_boot=a.n_boot, seed=a.seed) for k in KEYS}

    out = {"tool": "2026-09-05-kinematic-gate/raw/longitudinal_lead.py", "tier": "T0",
           "family": "LONGITUDINAL / distance keeping",
           "n_windows_total": W, "n_lead_windows": int(has.sum()),
           "n_lead_episodes": int(len(set(eid[has].tolist()))),
           "dt_s": 0.5, "grid": "0, 0.5, 1.0, 1.5, 2.0 s (the reward grid = the path grid)",
           "no_lead_masked_to_nan": True, "npz": a.npz,
           "ranking_key_used": _rk,
           "gate1_index_disagreements": int((pick["gate1"] != pick["model"]).sum()),
           "abs": res, "paired_vs_model": paired,
           "keys_present": sorted(set().union(*[set(v.keys()) for v in per.values()]))}
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)

    print("=== LONGITUDINAL / DISTANCE KEEPING on the selected path (T0) ===")
    print("  lead windows %d of %d (%d episodes); dt 0.50 s; no-lead windows masked to NaN"
          % (out["n_lead_windows"], W, out["n_lead_episodes"]))
    print("  ranking used: %s ; gate1 index disagreements with model: %d"
          % (_rk, out["gate1_index_disagreements"]))
    print("  %-12s %12s %12s %12s %8s %8s" % ("rule", "headway_min_m",
                                                "time_gap_min_s", "min_ttc_s", "n",
                                                "n_closng"))
    for rule in ["human_gt", "model"] + ["gate%d" % k for k in GATE_KS] + ["kin_only",
                                                                          "oracle"]:
        r = res.get(rule)
        if not r:
            continue
        def fv(x):
            try:
                return "%12.4f" % float(x)
            except (TypeError, ValueError):
                return "%12s" % "n/a"
        print("  %-12s %s %s %s %8s %8s" % (rule, fv(r.get("headway_min_m")),
                                             fv(r.get("time_gap_min_s")),
                                             fv(r.get("min_ttc_s")), r.get("n"),
                                             r.get("n_closing")))
    print("")
    print("  paired vs model (negative = the rule keeps LESS distance / less time):")
    for rule in ("gate1", "gate2", "gate4", "kin_only", "human_gt"):
        if rule not in paired:
            continue
        for k in KEYS:
            d = paired[rule][k]
            print("    %-10s %-12s %+10.4f [%+.4f, %+.4f] sep=%-5s n=%s"
                  % (rule, k, d["delta"], d["lo"], d["hi"], d["separated"],
                     d.get("n_windows")))
    print("")
    print("[lead] wrote %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
