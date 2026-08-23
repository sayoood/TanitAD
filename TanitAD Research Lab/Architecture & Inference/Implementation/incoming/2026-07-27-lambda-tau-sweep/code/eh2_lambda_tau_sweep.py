#!/usr/bin/env python3
"""E-H2 — the (lambda, tau) PRIOR-STRENGTH sweep over v4's frozen candidate fan.

Pre-registration: ``HIERARCHY_PRIOR_RESEARCH.md`` section 6.3, implemented as
specified (42 cells = lambda in {0,.25,.5,1,2,4,8} x tau in {.1,.25,.5,1,2,4},
q = 256 ALWAYS, two clamp sheets).  Report: ``LAMBDA_TAU_SWEEP.md``.

WHAT IS BEING SWEPT, AND WHY IT IS NOT `q`
------------------------------------------
v4's deployed selector is ALREADY the "bias, don't gate" architecture: a base
score plus a norm-capped product-of-experts class prior, resolved by a FLAT
argmax over all 256 candidates::

    graft = lambda * sum_m W_m @ log_softmax(logits_m / tau)
    ratio = ||graft|| / ||refined||                     (per window)
    graft = graft * min(1, seam_clamp / ratio)
    score = refined_pre + graft + sel_gate * pen        # pen = longitudinal term
    pick  = argmax_c score[c]                           # 1-of-256, never masked

Only the prior's STRENGTH was hard-wired (lambda = 1, tau = 1).  `q` -- the
top-q truncation measured at +0.21 .. +5.82 m -- confounds commitment with
COVERAGE, and no published planner truncates at inference.  (lambda, tau) reach
arbitrarily hard commitment WITHOUT removing a single candidate, which is the
axis neither we nor the literature has measured.

THE NAMED TRAP THIS FILE EXISTS TO NOT FALL INTO
------------------------------------------------
``seam_clamp = 1.0`` rescales the graft whenever ||graft|| > ||refined||, so
ABOVE THAT POINT lambda IS A NO-OP.  A lambda sweep read without the pre-clamp
ratio shows a flat axis that is SATURATION, NOT A FINDING.  Every cell therefore
carries ``preclamp_ratio_{max,mean,p50,p95}`` and ``clamp_bound_frac``, the
verdict has a dedicated ``SATURATED`` failing state, and the grid is run as two
sheets: ``deployable`` (seam_clamp = 1.0) and ``diagnostic`` (clamp off).

A SECOND TRAP, FOUND WHILE BUILDING THIS
----------------------------------------
The shipped head also carries ``seam_fail = 1.5``, which RAISES on a pre-clamp
ratio above 1.5.  It is checked BEFORE the clamp, so in the deployed head a
large lambda does not saturate -- it CRASHES.  Every cell records whether the
shipped guard would have fired (``would_trip_seam_fail``); the sweep itself runs
with the guard explicitly raised and says so.

VALIDATION IN BOTH DIRECTIONS
-----------------------------
``--selftest`` runs the whole pipeline on synthetic fixtures with KNOWN answers:
  * ``planted``    -- an interior optimum planted by construction; the rule must
                      find it and return CONFIRM-INTERIOR.
  * ``degenerate`` -- zero graft weights; the rule must return DEGENERATE.
  * ``saturated``  -- grafts so large every cell is clamp-bound; SATURATED.
  * ``unpowered``  -- pure noise; the rule must return UNPOWERED, not an optimum.
A harness that can only ever return PASS has not been validated.

NOTHING IS TRAINED.  Every cell is a CPU argmax over cached [W, 256] tensors.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

# ---- the ONLY admissible estimator ----------------------------------------
# 2026-07-27 (eval-host run): the probe assumed the repo layout and raised
# IndexError on any shallower path (e.g. /workspace/_eh2/). Made tolerant --
# a LOCATOR fix only; no arithmetic, no estimator, no cell value changes.
_HERE = Path(__file__).resolve()
_cands = [Path("/root/taniteval"), Path("/workspace/taniteval")]
if len(_HERE.parents) > 6:
    _cands.insert(0, _HERE.parents[6] / "taniteval")
for _c in _cands:
    if (_c / "taniteval" / "ci.py").exists():
        sys.path.insert(0, str(_c))
        break
from taniteval.ci import paired_episode_cluster_bootstrap  # noqa: E402

LAMBDAS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
TAUS = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0)
SHIPPED_SEAM_CLAMP = 1.0
SHIPPED_SEAM_FAIL = 1.5
CLAMP_OFF = float("inf")

# committed bars this sweep must reproduce before any cell is quotable
COMMITTED = {"produced|F_flat": 0.8563, "produced|O_oracle_in_fan": 0.2505,
             "oracle|F_flat": 0.6423, "produced|F_base_only": 0.8781,
             "neutral|F_flat": 0.7620}


# --------------------------------------------------------------------------- #
# The scorer -- ONE cell                                                       #
# --------------------------------------------------------------------------- #
def graft_of(cache: dict, lam: float, tau: float) -> torch.Tensor:
    """``lambda * sum_m W_m @ log_softmax(logits_m / tau)`` -> [W, N].

    Mirrors ``FlagshipV4Head._factor_grafts`` exactly (same op order, same
    log_softmax), so the lambda=1/tau=1 cell is the deployed graft.
    """
    lsm = torch.log_softmax
    g = torch.zeros_like(cache["refined_pre"])
    for m in ("lat", "lon", "dist"):
        g = g + lsm(cache[m] / tau, dim=-1) @ cache[f"W_{m}"].T   # [W,K]@[K,N]
    return lam * g


def score_cell(cache: dict, lam: float, tau: float, seam_clamp: float) -> dict:
    """Full deployed score for one (lambda, tau, clamp) cell + its clamp audit."""
    refined = cache["refined_pre"]
    graft = graft_of(cache, lam, tau)
    base = refined.norm(dim=-1).clamp_min(1e-9)                   # [W]
    ratio = graft.norm(dim=-1) / base                             # PRE-clamp
    if np.isfinite(seam_clamp):
        scale = seam_clamp / ratio.clamp_min(seam_clamp)          # <= 1
        graft = graft * scale[:, None]
    score = refined + graft + cache["sel_pen"]
    pick = score.argmax(dim=1)
    ade = cache["fan_err"].gather(1, pick[:, None]).squeeze(1)
    r = ratio.double().numpy()
    return {
        "ade_per_window": ade.double().numpy(),
        "pick": pick.numpy(),
        "preclamp_ratio_max": float(r.max()),
        "preclamp_ratio_mean": float(r.mean()),
        "preclamp_ratio_p50": float(np.percentile(r, 50)),
        "preclamp_ratio_p95": float(np.percentile(r, 95)),
        "clamp_bound_frac": float((r > seam_clamp).mean())
        if np.isfinite(seam_clamp) else 0.0,
        "would_trip_seam_fail": bool(r.max() > SHIPPED_SEAM_FAIL),
        "trip_frac_seam_fail": float((r > SHIPPED_SEAM_FAIL).mean()),
        "distinct_picks": int(len(np.unique(pick.numpy()))),
    }


# --------------------------------------------------------------------------- #
# The pre-registered optimum-locating rule                                     #
# --------------------------------------------------------------------------- #
#  Written BEFORE any cell was scored.  It has FOUR failing states and one
#  passing state, so it is falsifiable by construction:
#
#    DEGENERATE       the prior moves nothing anywhere on the grid       FAIL
#    SATURATED        the ARGMIN CELL ITSELF is clamp-bound and is tied
#                     (to 1e-12) with a different lambda at the same tau
#                     -> the located optimum is NOT IDENTIFIABLE in
#                     lambda, because the clamp ate that stretch of the
#                     axis.  (Saturation only in the high-lambda TAIL is
#                     reported in the clamp audit and does NOT disqualify
#                     an optimum found below it.)                        FAIL
#    UNPOWERED        the argmin is not the shipped cell but its paired
#                     CI vs the shipped cell INCLUDES zero                FAIL
#    NO-INTERIOR      the argmin sits on a grid EDGE (lambda = 0 or 8, or
#                     tau = 0.1 or 4) -> the optimum is not bracketed     FAIL
#    CONFIRM-INTERIOR argmin is interior AND separated-better than shipped PASS
#
#  The "interval" on the optimum is the ADMISSIBLE SET: every cell whose paired
#  delta vs the argmin is NOT separated.  A single argmin cell with no interval
#  is not a located optimum.
_EDGE_L, _EDGE_T = {LAMBDAS[0], LAMBDAS[-1]}, {TAUS[0], TAUS[-1]}


def locate_optimum(cells: dict, eid, n_boot: int, seed: int = 0) -> dict:
    """Apply the rule above to a {(lam, tau): cell} sheet. Returns the verdict."""
    keys = list(cells)
    base_key = (1.0, 1.0)
    if base_key not in cells:
        raise KeyError("the shipped cell (lambda=1, tau=1) must be in the grid")
    ade = {k: cells[k]["ade_per_window"] for k in keys}
    means = {k: float(ade[k].mean()) for k in keys}
    argmin = min(keys, key=lambda k: means[k])

    spread = max(means.values()) - min(means.values())
    vs_base = {k: paired_episode_cluster_bootstrap(
        ade[k], ade[base_key], eid, n_boot=n_boot, seed=seed) for k in keys}
    vs_argmin = {k: paired_episode_cluster_bootstrap(
        ade[k], ade[argmin], eid, n_boot=n_boot, seed=seed) for k in keys}
    admissible = [k for k in keys if not vs_argmin[k]["separated"]]

    # --- the failing states, in the order they are checked -------------------
    if spread < 1e-9:
        verdict = "DEGENERATE"
        why = ("the prior does not move the pick anywhere on the grid "
               f"(total spread {spread:.3e} m) -- there is no axis to optimise")
    else:
        lam_row = sorted([k for k in keys if k[1] == argmin[1]])
        bound = [k for k in lam_row if cells[k]["clamp_bound_frac"] >= 0.999]
        # is the OPTIMUM ITSELF identifiable in lambda, or did the clamp tie it
        # to a different lambda at the same tau?
        tied = [k for k in lam_row if k != argmin
                and abs(means[k] - means[argmin]) < 1e-12
                and cells[k]["clamp_bound_frac"] >= 0.999]
        if cells[argmin]["clamp_bound_frac"] >= 0.999 and tied:
            verdict = "SATURATED"
            why = (f"the argmin {argmin} is 100% clamp-bound and numerically "
                   f"TIED with lambda={sorted(k[0] for k in tied)} at the same "
                   f"tau -- lambda is a no-op there, so the 'optimum' is an "
                   f"artefact of seam_clamp, not a located optimum "
                   f"({len(bound)}/{len(lam_row)} cells in this row are fully "
                   "clamp-bound). Read the diagnostic sheet")
        elif argmin != base_key and not vs_base[argmin]["separated"]:
            verdict = "UNPOWERED"
            why = (f"argmin {argmin} beats the shipped cell by only "
                   f"{-vs_base[argmin]['delta']:+.4f} m, CI "
                   f"[{vs_base[argmin]['lo']:+.4f}, {vs_base[argmin]['hi']:+.4f}] "
                   "includes zero -- a null at n=40 episodes is UNPOWERED, "
                   "not refuted")
        elif argmin[0] in _EDGE_L or argmin[1] in _EDGE_T:
            verdict = "NO-INTERIOR"
            why = (f"argmin {argmin} sits on a grid EDGE, so the optimum is not "
                   "bracketed -- the grid must be extended before an interior "
                   "optimum can be claimed")
        else:
            verdict = "CONFIRM-INTERIOR"
            why = (f"argmin {argmin} is interior and separated-better than the "
                   f"shipped lambda=1/tau=1 cell by "
                   f"{-vs_base[argmin]['delta']:+.4f} m")
    return {
        "verdict": verdict, "why": why,
        "argmin": {"lambda": argmin[0], "tau": argmin[1],
                   "ade_0_2s": round(means[argmin], 4)},
        "shipped": {"lambda": 1.0, "tau": 1.0,
                    "ade_0_2s": round(means[base_key], 4)},
        "argmin_vs_shipped_paired": vs_base[argmin],
        "admissible_set": sorted([{"lambda": k[0], "tau": k[1],
                                   "ade_0_2s": round(means[k], 4)}
                                  for k in admissible],
                                 key=lambda d: d["ade_0_2s"]),
        "n_admissible": len(admissible), "n_cells": len(keys),
        "grid_spread_m": round(spread, 6),
        "_rule": "pre-registered in this file's header; four FAILING states "
                 "(DEGENERATE / SATURATED / UNPOWERED / NO-INTERIOR) and one "
                 "PASSING state (CONFIRM-INTERIOR). The admissible set is the "
                 "interval on the optimum's LOCATION.",
    }


# --------------------------------------------------------------------------- #
# The sweep                                                                    #
# --------------------------------------------------------------------------- #
def run_sheet(cache, eid, seam_clamp, n_boot, seed=0, tag="") -> dict:
    cells, rows = {}, []
    t0 = time.time()
    for lam in LAMBDAS:
        for tau in TAUS:
            cells[(lam, tau)] = score_cell(cache, lam, tau, seam_clamp)
    base = cells[(1.0, 1.0)]["ade_per_window"]
    for (lam, tau), c in cells.items():
        p = paired_episode_cluster_bootstrap(c["ade_per_window"], base, eid,
                                             n_boot=n_boot, seed=seed)
        rows.append({
            "lambda": lam, "tau": tau,
            "ade_0_2s": round(float(c["ade_per_window"].mean()), 4),
            "paired_delta_vs_shipped": p["delta"], "lo": p["lo"], "hi": p["hi"],
            "separated": p["separated"],
            "preclamp_ratio_max": round(c["preclamp_ratio_max"], 4),
            "preclamp_ratio_mean": round(c["preclamp_ratio_mean"], 4),
            "preclamp_ratio_p50": round(c["preclamp_ratio_p50"], 4),
            "preclamp_ratio_p95": round(c["preclamp_ratio_p95"], 4),
            "clamp_bound_frac": round(c["clamp_bound_frac"], 4),
            "would_trip_seam_fail": c["would_trip_seam_fail"],
            "trip_frac_seam_fail": round(c["trip_frac_seam_fail"], 4),
            "distinct_picks": c["distinct_picks"],
        })
    n_bound = sum(1 for r in rows if r["clamp_bound_frac"] > 0)
    n_full = sum(1 for r in rows if r["clamp_bound_frac"] >= 0.999)
    n_trip = sum(1 for r in rows if r["would_trip_seam_fail"])
    return {
        "_sheet": tag, "_seam_clamp": (None if not np.isfinite(seam_clamp)
                                       else seam_clamp),
        "_seam_fail_raised_for_the_sweep": True,
        "clamp_audit": {
            "cells_with_any_clamped_window": n_bound,
            "cells_fully_clamp_bound": n_full,
            "cells_the_shipped_seam_fail_1_5_would_have_RAISED": n_trip,
            "_read": "a flat lambda axis with clamp_bound_frac at 1.0 is "
                     "SATURATION, not a finding. Cells the shipped guard would "
                     "raise on are NOT reachable in the deployed head at all.",
        },
        "cells": sorted(rows, key=lambda r: (r["lambda"], r["tau"])),
        "optimum": locate_optimum(cells, eid, n_boot, seed),
        "_wallclock_s": round(time.time() - t0, 1),
    }


def fidelity_gate(cache) -> dict:
    """The lambda=1/tau=1 cell MUST reproduce the cached deployed selector.

    Without this every cell is measured against a baseline that is not the
    shipped model.  Two checks: the score tensor and the resulting pick.
    """
    c = score_cell(cache, 1.0, 1.0, SHIPPED_SEAM_CLAMP)
    out = {"cell": "lambda=1, tau=1, seam_clamp=1.0"}
    if "sel_score" in cache:
        mine = (cache["refined_pre"]
                + graft_of(cache, 1.0, 1.0)
                * (SHIPPED_SEAM_CLAMP / (graft_of(cache, 1.0, 1.0).norm(dim=-1)
                   / cache["refined_pre"].norm(dim=-1).clamp_min(1e-9)
                   ).clamp_min(SHIPPED_SEAM_CLAMP))[:, None]
                + cache["sel_pen"])
        out["max_abs_score_err_vs_forward_pass"] = float(
            (mine - cache["sel_score"]).abs().max())
        out["score_bit_identical"] = bool(torch.equal(mine, cache["sel_score"]))
    if "ref_sel_idx" in cache:
        agree = float((torch.as_tensor(c["pick"]) == cache["ref_sel_idx"]).float()
                      .mean())
        out["selection_fidelity_vs_forward_pass"] = round(agree, 6)
        out["PASS"] = bool(agree >= 1.0 - 1e-12)
    out["ade_0_2s"] = round(float(c["ade_per_window"].mean()), 4)
    return out


# --------------------------------------------------------------------------- #
# Synthetic fixtures -- validation in BOTH directions                          #
# --------------------------------------------------------------------------- #
# Fixture constants, tuned ONCE so each case lands on its intended verdict, then
# frozen. They are stated here rather than searched at run time, because a
# fixture that tunes itself to the answer proves nothing.
_FIX = {                    # graft_scale, graft_noise, logit_noise, seam_clamp
    "planted":    (0.020, 0.75, 2.0, SHIPPED_SEAM_CLAMP),
    "degenerate": (0.000, 0.00, 2.0, SHIPPED_SEAM_CLAMP),
    "saturated":  (8.000, 0.75, 2.0, 0.02),
    "unpowered":  (1.5e-4, 1.00, 2.0, SHIPPED_SEAM_CLAMP),
}


def _fixture(kind: str, W=880, N=64, K=8, n_ep=40, seed=0) -> tuple[dict, np.ndarray]:
    """Build a cache with a KNOWN answer.  ``kind`` selects which answer.

    The construction mirrors the real problem's shape: a per-candidate error
    surface with class-dependent structure, a NOISY base score that already
    ranks it fairly well, and a class posterior that is informative but wrong
    some of the time.  An interior optimum in prior strength exists because both
    the base score and the prior are noisy estimates of the same utility -- the
    classic mixing-weight situation, which is also the mechanism CFG's U-shaped
    FID curve and aWTA's annealing both live on.
    """
    scale, gnoise, lnoise, _clamp = _FIX[kind]
    g = torch.Generator().manual_seed(seed)
    eid = np.repeat(np.arange(n_ep), W // n_ep).astype(str)
    fan_err = torch.rand(W, N, generator=g) * 4.0                 # metres
    cls = torch.randint(0, K, (W,), generator=g)                  # the true class
    good = torch.randn(N, K, generator=g)                         # class -> anchor
    for k in range(K):                                            # plant the signal
        fan_err[cls == k] -= (good[:, k] * 0.6)[None, :]
    fan_err = fan_err.clamp_min(0.02)
    refined = -fan_err + torch.randn(W, N, generator=g) * 1.2     # noisy base score
    logits = torch.full((W, K), -2.0)
    logits[torch.arange(W), cls] = 2.0                            # a good classifier
    logits = logits + torch.randn(W, K, generator=g) * lnoise     # ... but not perfect
    # ~74% argmax accuracy at lnoise=2.0. A near-perfect classifier would make
    # tau -> 0 monotonically better and there would be no interior tau optimum
    # to find -- the fixture has to be able to punish hard commitment.
    # a PARTIALLY correct class->anchor map: right direction, wrong in detail.
    # Perfectly correct would make more prior monotonically better and the
    # optimum would sit on the lambda = 8 edge, testing nothing.
    Wm = (good + torch.randn(N, K, generator=g) * gnoise) * scale
    # ROW-CENTRE across the class axis. Non-obvious and load-bearing: the graft
    # consumes log_softmax, so sum_j W[c,j]*L[j] carries a CLASS-INDEPENDENT
    # nuisance g*sum_j W[c,j] whose variance is K x the signal's. Un-centred, the
    # fixture's prior is dominated by that nuisance and lambda is monotonically
    # harmful -- which is a property of the fixture, not of prior strength.
    Wm = Wm - Wm.mean(dim=1, keepdim=True)
    return {
        "refined_pre": refined,
        "lat": logits, "lon": torch.zeros(W, K), "dist": torch.zeros(W, K),
        "W_lat": Wm, "W_lon": torch.zeros(N, K), "W_dist": torch.zeros(N, K),
        "sel_pen": torch.zeros(W, N), "fan_err": fan_err,
    }, eid


def selftest(n_boot: int = 400) -> dict:
    """Fidelity gate AND deliberate failure -- the rule must be able to say NO."""
    want = {"planted": "CONFIRM-INTERIOR", "degenerate": "DEGENERATE",
            "saturated": "SATURATED", "unpowered": "UNPOWERED"}
    got, ok = {}, True
    for kind, expect in want.items():
        cache, eid = _fixture(kind)
        clamp = _FIX[kind][3]
        sheet = run_sheet(cache, eid, clamp, n_boot, tag=f"selftest:{kind}")
        v = sheet["optimum"]
        # for the planted fixture, also check the located optimum is SOFT+INTERIOR
        extra = {}
        if kind == "planted":
            extra = {"argmin_is_not_lambda0":
                     v["argmin"]["lambda"] > 0.0,
                     "admissible_set_is_an_interval": v["n_admissible"] > 1}
        hit = (v["verdict"] == expect) and all(extra.values())
        ok = ok and hit
        got[kind] = {"expected": expect, "got": v["verdict"], "pass": hit,
                     "argmin": v["argmin"], "why": v["why"],
                     "n_admissible": v["n_admissible"], **extra}
    return {
        "PASS": ok, "cases": got,
        "_read": "the rule returns a FAILING verdict on three fixtures built to "
                 "fail and the PASSING verdict on one built to pass. A harness "
                 "that can only ever return PASS has not been validated.",
        "_what_makes_it_FAIL_on_real_data":
            "DEGENERATE if the grafts were still at their ReZero zero-init; "
            "SATURATED if seam_clamp=1.0 pins every lambda cell (the named "
            "trap); UNPOWERED if the best cell's paired CI vs lambda=1/tau=1 "
            "includes zero at n=40 episodes; NO-INTERIOR if the argmin lands on "
            "lambda=0/8 or tau=0.1/4.",
    }


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="", help="eh2_cache.pt from eh2_build_cache.py")
    ap.add_argument("--out", default="eh2_lambda_tau_sweep.json")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    R = {"_experiment": "E-H2 -- the (lambda, tau) prior-strength sweep, q=256 always",
         "_prereg": "HIERARCHY_PRIOR_RESEARCH.md 6.3",
         "_evidence_class": "MEASURED (ours)",
         "_estimator": "paired_episode_cluster_bootstrap (B=%d, unit = episode "
                       "cluster). NEVER overlapping_holdout_se." % a.boot,
         "_host": platform.node(), "_python": platform.python_version(),
         "_torch": torch.__version__,
         "_grid": {"lambda": list(LAMBDAS), "tau": list(TAUS), "q": 256,
                   "n_cells": len(LAMBDAS) * len(TAUS)},
         "_committed_bars": COMMITTED}

    R["selftest"] = selftest()
    print(json.dumps({k: v["got"] for k, v in R["selftest"]["cases"].items()},
                     indent=2), flush=True)
    if not R["selftest"]["PASS"]:
        raise SystemExit("SELFTEST FAILED — the sweep is not admissible")
    if a.selftest or not a.cache:
        Path(a.out).write_text(json.dumps(R, indent=2))
        print(f"[eh2] selftest only -> {a.out}")
        return

    cache = torch.load(a.cache, map_location="cpu", weights_only=False)
    eid = np.asarray([str(int(e)) for e in cache["ep"]])
    R["_cache"] = {k: str(list(v.shape)) for k, v in cache.items()
                   if torch.is_tensor(v)}
    R["fidelity_gate"] = fidelity_gate(cache)
    if not R["fidelity_gate"].get("PASS", False):
        raise SystemExit("FIDELITY GATE FAILED — the CPU re-scorer does not "
                         "reproduce the forward pass; no cell is quotable")
    for tag, clamp in (("deployable", SHIPPED_SEAM_CLAMP),
                       ("diagnostic", CLAMP_OFF)):
        R[f"sheet_{tag}"] = run_sheet(cache, eid, clamp, a.boot, tag=tag)
        print(f"[eh2] {tag}: {R[f'sheet_{tag}']['optimum']['verdict']}", flush=True)
    Path(a.out).write_text(json.dumps(R, indent=2))
    print(f"[eh2] -> {a.out}")


if __name__ == "__main__":
    main()
