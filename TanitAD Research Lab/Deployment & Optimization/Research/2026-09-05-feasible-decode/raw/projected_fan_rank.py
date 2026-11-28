#!/usr/bin/env python3
"""P1c -- does the reward's ranking defect SURVIVE a feasibility-aware decode?

ZERO GPU. Re-runs the rank probe on the PROJECTED fan.

THE POINT, and it is a change of QUESTION, not a change of bar.
`D-RL-REWARD-RANK-1` asks whether the reward ranks envelope-VIOLATING candidates above
feasible ones. That presupposes a population: windows whose fan contains BOTH. After the
projection the fan's `envelope` / `kamm_over` / `infeasible` rates are 0.0000 exactly, so
a Spearman against those flags is UNDEFINED -- there is nothing to rank. This tool asserts
that (P1c-C0) rather than letting an undefined value be averaged in as a null.

The residual, WEAKER question -- the only one with a population left -- is whether the
reward still prefers the HARDER-DRIVING candidate inside a fan that is feasible by
construction, i.e. rho(term, peak_g) with BOTH sides computed on the projected fan.
⚠️ Different stake: preferring 0.6 g over 0.3 g is margin; preferring 4 g over 0.5 g was
a safety defect. The two must never be reported as one number.

⛔ C-OBJECT (RETRACTION #30): the artifact re-derives envelope/kamm_over on the ranked
tensor through `fan_safety.score_paths` -- the CONSUMER -- and prints the npz md5, so it
says WHICH object it ranked rather than only that its arithmetic closed.
ASCII-only output.
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np
import torch

_REPO = os.environ.get("TANITAD_REPO") or "C:/Users/Admin/refcv4b_repo"
sys.path.insert(0, os.path.join(_REPO, "stack"))
sys.path.insert(0, os.path.join(_REPO, "taniteval"))
sys.path.insert(0, os.path.join(_REPO, "taniteval", "tools"))

from tanitad.rl import rewards as RW                # noqa: E402
from tanitad.refs import feasible_decode as FD      # noqa: E402
import fan_safety as FS                             # noqa: E402
import taniteval.ci as CI                           # noqa: E402

TOOL = "2026-09-05-feasible-decode/raw/projected_fan_rank.py"
DT = 0.5
FLAGS = ("envelope", "kamm_over", "infeasible", "off_reach", "ttc_below", "contact")
LEAD_ONLY = set(FS.LEAD_ONLY)
COMPS = ["progress", "collision", "headway", "feasibility", "comfort"]


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def spearman(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    if a.size != b.size or a.size < 3:
        return float("nan")
    from scipy.stats import rankdata
    ra, rb = rankdata(a), rankdata(b)
    if ra.std() == 0.0 or rb.std() == 0.0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def boot(vals, eids, n_boot, seed):
    v = np.asarray(vals, np.float64)
    e = np.asarray(eids)
    ok = np.isfinite(v)
    if ok.sum() < 3:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "n_windows": int(ok.sum()), "n_episodes": 0,
                "n_undefined": int((~ok).sum()), "separated": False,
                "estimator": "episode_cluster_bootstrap",
                "note": "UNDEFINED on %d/%d windows -- reported, never averaged as 0"
                        % (int((~ok).sum()), v.size)}
    r = CI.episode_cluster_bootstrap(v[ok], e[ok], n_boot=n_boot, seed=seed)
    o = {k: r[k] for k in ("mean", "lo", "hi", "n_windows", "n_episodes", "estimator")}
    o["n_undefined"] = int((~ok).sum())
    o["separated"] = bool(o["lo"] > 0 or o["hi"] < 0)
    return o


def exceedance(t, dt=DT):
    k = RW.kinematics(t, dt)
    return ((k.accel.abs() / RW.A_MAX_MPS2 - 1).clamp_min(0).amax(-1)
            + (k.kappa.abs() / RW.KAPPA_MAX_1PM - 1).clamp_min(0).amax(-1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--fan-key", default="fan8", choices=("fan2", "fan8"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()

    z = np.load(a.npz)
    m5 = md5(a.npz)
    if a.fan_key == "fan2":
        raw5 = torch.from_numpy(z["fan2"]).float()
    else:
        raw5 = FS.with_origin(torch.from_numpy(z["fan8"]).float()[..., :4, :])
    W, N = raw5.shape[0], raw5.shape[1]
    v0 = torch.from_numpy(z["v0"]).float()
    lead5 = torch.from_numpy(z["lead5"]).float().reshape(W, 1, 5, 2)
    eid, has = z["eid"], z["has_lead"]

    print("=== P1c: does the reward's ranking defect SURVIVE the feasibility-aware decode? ===")
    print("  object: %s [%s] %s  md5 %s" % (os.path.basename(a.npz), a.fan_key,
                                            tuple(raw5.shape), m5))

    arms = {"emitted": raw5,
            "projected_mu0.70": FD.project_feasible(raw5, v0, mu=FD.MU_KAMM),
            "projected_mu0.70_entry": FD.project_feasible(raw5, v0, mu=FD.MU_KAMM,
                                                          clamp_entry=True)}
    out = {"_tool": TOOL, "_tier": "T0 readout on a fan -- NOT a training result and NOT "
                                   "a driving claim",
           "_evidence_class": "MEASURED (ours)", "npz": a.npz, "npz_md5": m5,
           "n_windows": W, "n_candidates": N,
           "n_episodes": int(len(set(eid.tolist()))),
           "estimator": "episode_cluster_bootstrap over per-window Spearman rho",
           "n_boot": a.n_boot, "seed": a.seed, "arms": {}}

    for tag, fan in arms.items():
        ctx = {"dt": DT, "v0": v0.reshape(W, 1),
               "lead_len_m": FS.LEAD_LEN_DEFAULT_M, "lead_path": lead5}
        sc = FS.score_paths(fan, v0, lead5, lead_len_m=FS.LEAD_LEN_DEFAULT_M)
        comp = {n: RW.COMPONENTS[n](fan, ctx).double().numpy() for n in COMPS}
        ex = exceedance(fan).double().numpy()
        nz = ex[ex > 0]
        ex_s = float(np.median(nz)) if nz.size else 1.0
        terms = {
            "progress": comp["progress"],
            "progress_v2": (RW._progress(fan, ctx).clamp(max=1.0).clamp_min(-1.0)
                            .double().numpy() * (1.0 / (1.0 + ex / max(ex_s, 1e-9)))),
            "DEFAULT": sum(RW.DEFAULT_WEIGHTS[n] * comp[n] for n in COMPS),
        }
        flags = {f: sc[f].numpy().astype(np.float64) for f in FS.FLAGS}
        pg = sc["peak_g"].double().numpy()

        rows = []
        for j in range(W):
            r = {}
            for k, vv in terms.items():
                r["rho__%s__peak_g" % k] = spearman(vv[j], pg[j])
                for f in FLAGS:
                    if f in LEAD_ONLY and not bool(has[j]):
                        r["rho__%s__%s" % (k, f)] = float("nan")
                    else:
                        r["rho__%s__%s" % (k, f)] = spearman(vv[j], flags[f][j])
            r["ctrl_self"] = spearman(terms["progress"][j], terms["progress"][j])
            r["ctrl_const"] = spearman(np.ones(N), pg[j])
            r["ctrl_rand"] = spearman(np.random.default_rng(1000 + j).random(N), pg[j])
            rows.append(r)

        panel = {}
        for k in terms:
            for f in ("peak_g",) + FLAGS:
                nm = "rho__%s__%s" % (k, f)
                panel[nm] = boot([r[nm] for r in rows], eid, a.n_boot, a.seed)
        ctr = {c: boot([r[c] for r in rows], eid, a.n_boot, a.seed)
               for c in ("ctrl_self", "ctrl_const", "ctrl_rand")}
        ctr["ctrl_const"]["n_undefined_is_all_windows"] = bool(
            ctr["ctrl_const"]["n_undefined"] == W)

        obj = {"re_derived_through": "fan_safety.score_paths (the CONSUMER)",
               "envelope_rate": float(sc["envelope"].float().mean()),
               "kamm_over_rate": float(sc["kamm_over"].float().mean()),
               "infeasible_rate": float(sc["infeasible"].float().mean()),
               "peak_g_mean": float(sc["peak_g"].mean()),
               "ex_s": ex_s}
        out["arms"][tag] = {"C_OBJECT": obj, "controls": ctr, "panel": panel,
                            "ex_s": ex_s}

    p = out["arms"]["projected_mu0.70"]
    c0 = {"criterion": "on the projected fan, rho vs `envelope` and `kamm_over` is "
                       "UNDEFINED on 100% of windows (there is nothing to rank)",
          "envelope_n_undefined": p["panel"]["rho__progress__envelope"]["n_undefined"],
          "kamm_n_undefined": p["panel"]["rho__progress__kamm_over"]["n_undefined"],
          "n_windows": W,
          "fan_envelope_rate": p["C_OBJECT"]["envelope_rate"],
          "fan_kamm_over_rate": p["C_OBJECT"]["kamm_over_rate"]}
    c0["PASS"] = bool(c0["envelope_n_undefined"] == W and c0["kamm_n_undefined"] == W
                      and p["C_OBJECT"]["envelope_rate"] == 0.0
                      and p["C_OBJECT"]["kamm_over_rate"] == 0.0)
    c1 = {"criterion": "on the projected fan, rho(progress stock, peak_g) < +0.05",
          "value": p["panel"]["rho__progress__peak_g"]["mean"],
          "ci": [p["panel"]["rho__progress__peak_g"]["lo"],
                 p["panel"]["rho__progress__peak_g"]["hi"]],
          "emitted_reference":
              out["arms"]["emitted"]["panel"]["rho__progress__peak_g"]["mean"]}
    c1["PASS"] = bool(c1["value"] < 0.05)
    c3 = {"committed_prediction": "the projection does NOT touch the lead geometry, so "
                                  "rho vs ttc_below should be essentially unchanged; if "
                                  "it is not, something other than feasibility moved",
          "emitted":
              out["arms"]["emitted"]["panel"]["rho__progress__ttc_below"]["mean"],
          "projected": p["panel"]["rho__progress__ttc_below"]["mean"]}
    c3["abs_change"] = abs(c3["projected"] - c3["emitted"])
    out["P1c_C0"], out["P1c_C1"], out["P1c_C3"] = c0, c1, c3

    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)

    print("")
    print("  %-24s %-14s %10s %10s %10s %10s %10s"
          % ("arm", "term", "peak_g", "envelope", "kamm_over", "ttc_below", "contact"))
    for tag in arms:
        A = out["arms"][tag]
        for k in ("progress", "progress_v2", "DEFAULT"):
            def fmt(f):
                b = A["panel"]["rho__%s__%s" % (k, f)]
                if b["n_undefined"] == W:
                    return "  UNDEFINED"
                return "%10.4f" % b["mean"]
            print("  %-24s %-14s %s %s %s %s %s"
                  % (tag, k, fmt("peak_g"), fmt("envelope"), fmt("kamm_over"),
                     fmt("ttc_below"), fmt("contact")))
        print("      [object] envelope %.4f  kamm_over %.4f  infeasible %.4f  peak_g %.4f"
              % (A["C_OBJECT"]["envelope_rate"], A["C_OBJECT"]["kamm_over_rate"],
                 A["C_OBJECT"]["infeasible_rate"], A["C_OBJECT"]["peak_g_mean"]))
    print("")
    print("  P1c-C0 rho vs envelope UNDEFINED on %d/%d windows, kamm on %d/%d -> %s"
          % (c0["envelope_n_undefined"], W, c0["kamm_n_undefined"], W,
             "PASS" if c0["PASS"] else "FAIL"))
    print("  P1c-C1 rho(progress, peak_g | projected) = %+.4f [%+.4f, %+.4f] "
          "(emitted %+.4f) -> %s"
          % (c1["value"], c1["ci"][0], c1["ci"][1], c1["emitted_reference"],
             "PASS" if c1["PASS"] else "FAIL"))
    print("  P1c-C3 rho vs ttc_below emitted %+.4f -> projected %+.4f (|change| %.4f; "
          "predicted ~unchanged)" % (c3["emitted"], c3["projected"], c3["abs_change"]))
    print("[p1c] wrote %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
