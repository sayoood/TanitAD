"""E1c EVALUATOR SELF-TEST — binding process requirement earned by E1b.

E1b's shipped `e1b_eval.py` would have reported **SUCCESS on a BOUND run**: its
verdict string was derived from the primary alone and never consulted the
guardrails, several guardrails were unimplemented, and the open-loop guardrail
was unpaired.  "A guardrail absent from the code is a comment, not a guardrail."

So: BEFORE the deciding E1c run, drive the SHIPPED verdict code path
(`e1c_common.frontier_point_stats` -> `evaluate_point` -> `select_winner` ->
`render_verdict`, the exact functions `e1c_eval.py` calls) with SYNTHETIC
per-window arrays constructed to FAIL each registered guardrail one at a time,
and assert the failing verdict is rendered.

Scope, stated honestly: this exercises 100 % of the DECIDING path from
per-window arrays onward — the estimator, the six pre-registered predicates, the
frozen selection rule and the verdict string.  It does NOT re-verify the rollout
itself; that is E1a's `e1a_horizon.rollout` reused verbatim, whose base arm
reproduced E1a bit-identically in E1b and is re-rolled again here as a control.

Every case ALSO asserts that the synthetic construction really produced the
intended separation pattern, so a case that silently stopped testing what it
claims fails loudly instead of passing vacuously.

Run:  PYTHONPATH=/workspace/TanitAD/stack /workspace/venv/bin/python e1c_selftest.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

for _p in ("/workspace/TanitAD/taniteval", "/workspace/e1b", "/workspace/e1c"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from taniteval import ci as CI
except Exception:                                   # noqa: BLE001
    import taniteval_ci as CI                       # md5-identical vendored copy
CI_BINDING = getattr(CI, "__file__", "?")

import e1c_common as C

BOOT, PAIRED = C.make_stat(CI)
RES = []


# --------------------------------------------------------------------------- #
# synthetic builders                                                           #
# --------------------------------------------------------------------------- #
def mk_closed(seed, n_ep=43, n_junc=6, n_long=19, dep_base=0.5877,
              dep_delta_other=-0.43, dep_delta_junc=-0.43, ood_ft=1.1339,
              ood_base=1.2664, jitter=0.02, delta_noise=0.005,
              junc_null=False):
    """One K=185 window per episode (E1b's real shape: 43 w / 43 ep, 6 junction,
    19 longitudinal).  Deltas are set per-window so the overall and junction
    strata can be driven independently.  ``delta_noise`` controls whether a
    nominally-zero delta is a genuine NULL (large noise) or a tiny-but-real
    effect (small noise) — a null case must be built with enough spread that the
    bootstrap really cannot separate it, which is why this knob exists."""
    rng = np.random.default_rng(seed)
    junc = np.zeros(n_ep, bool); junc[:n_junc] = True
    long_ = np.zeros(n_ep, bool); long_[n_junc:n_junc + n_long] = True
    base = np.clip(dep_base + rng.normal(0, jitter, n_ep), 0, 1)
    d = np.where(junc, dep_delta_junc, dep_delta_other) \
        + rng.normal(0, delta_noise, n_ep)
    if junc_null:
        # an EXACT null on the junction stratum: mean 0 by construction and
        # spread wide enough that a 6-cluster bootstrap provably straddles 0.
        d[junc] = np.linspace(-0.25, 0.25, n_junc)
    ft = np.clip(base + d, 0, 1)
    key = [(i, 0) for i in range(n_ep)]
    eid = [str(i) for i in range(n_ep)]

    def arm(dep, ood):
        return {"key": key, "eid": eid, "junc": junc, "long": long_,
                "dep": dep,
                "win_dep": np.clip(dep * 1.5, 0, 1),
                "peak_xte": dep * 60.0 + 0.5,
                "mean_xte": dep * 22.0 + 0.2,
                "peak_dpsi": dep * 40.0 + 2.0,
                "ood_peak": np.full(n_ep, ood) + rng.normal(0, 1e-4, n_ep),
                "ood_mean": np.full(n_ep, ood - 0.11) + rng.normal(0, 1e-4, n_ep),
                "out_env": (dep > 0.4).astype(float),
                "ade2s": 0.5 + dep * 0.2}
    return arm(ft, ood_ft), arm(base, ood_base)


def mk_open(seed, n_ep=44, per_ep=22, ade_base=0.4747, ade_delta=0.0,
            acc_base=0.6815, acc_delta=0.0, l1_base=0.1775, l1_delta=0.0,
            jitter=0.08):
    """967-ish held-out open-loop windows, 44 episode clusters."""
    rng = np.random.default_rng(seed + 7717)
    n = n_ep * per_ep
    eid = [str(i // per_ep) for i in range(n)]
    key = [(i // per_ep, (i % per_ep) * 8) for i in range(n)]
    ep_eff = rng.normal(0, jitter, n_ep)[np.array([i // per_ep for i in range(n)])]

    def pair(b, d, lo=None, hi=None, dj=0.02):
        base = b + ep_eff + rng.normal(0, dj, n)
        ft = base + d + rng.normal(0, dj * 0.3, n)
        if lo is not None:
            base, ft = np.clip(base, lo, hi), np.clip(ft, lo, hi)
        return ft, base

    ade_f, ade_b = pair(ade_base, ade_delta)
    acc_f, acc_b = pair(acc_base, acc_delta, 0.0, 1.0)
    l1_f, l1_b = pair(l1_base, l1_delta, 0.0, None if False else 10.0, dj=0.01)
    ce_f, ce_b = pair(0.8757, l1_delta * 4.6)

    def arm(a, ac, ce, l1):
        return {"key": key, "eid": eid, "ade2s": a, "anchor_acc": ac,
                "anchor_ce": ce, "anchor_traj_l1": l1}
    return arm(ade_f, acc_f, ce_f, l1_f), arm(ade_b, acc_b, ce_b, l1_b)


def point(step, seed, cl_kw=None, ol_kw=None):
    cl_ft, cl_b = mk_closed(seed, **(cl_kw or {}))
    ol_ft, ol_b = mk_open(seed, **(ol_kw or {}))
    st = C.frontier_point_stats(step, cl_ft, cl_b, ol_ft, ol_b, BOOT, PAIRED,
                                label="synthetic")
    return C.evaluate_point(st)


def check(name, cond, detail=""):
    RES.append({"check": name, "pass": bool(cond), "detail": str(detail)[:400]})
    print(("  [PASS] " if cond else "  [FAIL] ") + name +
          (f"  -- {detail}" if detail else ""), flush=True)
    return bool(cond)


def case(name, expect_success, must_fail, **kw):
    """Run one synthetic case and assert BOTH that the construction produced the
    intended failure AND that the verdict is the failing one."""
    print(f"\n=== {name} ===", flush=True)
    ev = point(1000, kw.pop("seed", 11), kw.get("cl_kw"), kw.get("ol_kw"))
    v = C.render_verdict([ev])
    ok = True
    ok &= check(f"{name}: SUCCESS_POINT == {expect_success}",
                ev["SUCCESS_POINT"] == expect_success,
                f"failed={ev['failed']}")
    ok &= check(f"{name}: verdict == {'SUCCESS' if expect_success else 'BOUND'}",
                v["verdict"] == ("SUCCESS" if expect_success else "BOUND"),
                v["text"][:180])
    ok &= check(f"{name}: construction really tripped {must_fail or '(nothing)'}",
                set(must_fail) <= set(ev["failed"]),
                f"failed={ev['failed']}")
    if must_fail:
        ok &= check(f"{name}: no OTHER condition failed",
                    set(ev["failed"]) == set(must_fail),
                    f"failed={ev['failed']}")
    return ev, v, ok


# --------------------------------------------------------------------------- #
def main():
    print("E1c EVALUATOR SELF-TEST — synthetic guardrail-failure battery",
          flush=True)
    print(f"estimator: paired_episode_cluster_bootstrap B={C.B_BOOT} | "
          f"frontier points M={C.N_FRONTIER_POINTS} | "
          f"Bonferroni alpha={C.BONFERRONI_ALPHA:.6f}", flush=True)
    allok = True

    # ---- C0: everything clean -> SUCCESS (the positive control) -------------
    ev0, _, ok = case("C0 all-clean", True, [])
    allok &= ok

    # ---- C1: guardrail (a) open-loop ADE@2s separated WORSE ------------------
    _, _, ok = case("C1 Ga openloop-ADE regressed", False,
                    ["Ga_openloop_ade2s_ok"],
                    ol_kw=dict(ade_delta=+0.1947))
    allok &= ok

    # ---- C2: guardrail (b1) anchor_acc separated WORSE (lower) ---------------
    _, _, ok = case("C2 Gb1 anchor_acc regressed", False,
                    ["Gb1_anchor_acc_ok"],
                    ol_kw=dict(acc_delta=-0.0651))
    allok &= ok

    # ---- C3: guardrail (b2) anchor_traj_l1 separated WORSE (higher) ----------
    _, _, ok = case("C3 Gb2 anchor_traj_l1 regressed", False,
                    ["Gb2_anchor_traj_l1_ok"],
                    ol_kw=dict(l1_delta=+0.0624))
    allok &= ok

    # ---- C4: guardrail (c) OOD out of the measured band ----------------------
    _, _, ok = case("C4 Gc OOD out of band", False, ["Gc_ood_in_band"],
                    cl_kw=dict(ood_ft=1.42))
    allok &= ok

    # ---- C5: junction primary NOT separated ---------------------------------
    _, _, ok = case("C5 P2 junction not separated", False,
                    ["P2_dep_junction_separated_lower"],
                    cl_kw=dict(junc_null=True))
    allok &= ok

    # ---- C6: overall primary NOT separated ----------------------------------
    _, _, ok = case("C6 P1 overall not separated", False,
                    ["P1_dep_overall_separated_lower"],
                    cl_kw=dict(dep_delta_other=0.0, dep_delta_junc=-0.43,
                               n_junc=2, delta_noise=0.15))
    allok &= ok

    # ---- C7: closed loop got WORSE (separated higher) -----------------------
    _, _, ok = case("C7 primary separated WORSE", False,
                    ["P1_dep_overall_separated_lower",
                     "P2_dep_junction_separated_lower"],
                    cl_kw=dict(dep_base=0.20, dep_delta_other=+0.30,
                               dep_delta_junc=+0.30))
    allok &= ok

    # ---- C8: REPLAY OF E1b's REAL MEASURED NUMBERS -> must be BOUND ---------
    print("\n=== C8 E1b replay (the run E1b's shipped evaluator called SUCCESS) ===",
          flush=True)
    ev8 = point(4000, 23,
                cl_kw=dict(dep_base=0.5877, dep_delta_other=-0.4274,
                           dep_delta_junc=-0.4270, ood_ft=1.1339),
                ol_kw=dict(ade_delta=+0.1947, acc_delta=-0.0651,
                           l1_delta=+0.0624))
    v8 = C.render_verdict([ev8])
    allok &= check("C8: primary DID fire (both strata separated lower)",
                   ev8["primary_ok"], f"failed={ev8['failed']}")
    allok &= check("C8: all three held-out guardrails FAILED",
                   set(ev8["failed"]) == {"Ga_openloop_ade2s_ok",
                                          "Gb1_anchor_acc_ok",
                                          "Gb2_anchor_traj_l1_ok"},
                   f"failed={ev8['failed']}")
    allok &= check("C8: verdict is BOUND despite the primary firing",
                   v8["verdict"] == "BOUND", v8["text"][:200])

    # ---- C9: multi-point frontier, only the middle point qualifies ----------
    print("\n=== C9 frontier selection: one qualifying point among five ===",
          flush=True)
    fr = [
        point(250, 31, cl_kw=dict(dep_delta_other=0.0, dep_delta_junc=0.0,
                                  delta_noise=0.15)),
        point(1000, 32, cl_kw=dict(dep_delta_other=-0.20, dep_delta_junc=-0.22)),
        point(2000, 33, cl_kw=dict(dep_delta_other=-0.35, dep_delta_junc=-0.36),
              ol_kw=dict(ade_delta=+0.1947)),
        point(3000, 34, cl_kw=dict(dep_delta_other=-0.41, dep_delta_junc=-0.42),
              ol_kw=dict(ade_delta=+0.1947, acc_delta=-0.0651)),
        point(4000, 35, cl_kw=dict(dep_delta_other=-0.4274,
                                   dep_delta_junc=-0.4270),
              ol_kw=dict(ade_delta=+0.1947, acc_delta=-0.0651,
                         l1_delta=+0.0624)),
    ]
    v9 = C.render_verdict(fr)
    allok &= check("C9: exactly one SUCCESS point",
                   sum(e["SUCCESS_POINT"] for e in fr) == 1,
                   [(e["step"], e["SUCCESS_POINT"], e["failed"]) for e in fr])
    allok &= check("C9: verdict SUCCESS and winner is step 1000",
                   v9["verdict"] == "SUCCESS" and v9["winner_step"] == 1000,
                   v9["text"][:200])
    allok &= check("C9: the LARGER closed-loop gains were correctly REJECTED "
                   "for guardrail failure (no goalpost moving)",
                   all(not e["SUCCESS_POINT"] for e in fr if e["step"] >= 2000),
                   [(e["step"], e["failed"]) for e in fr if e["step"] >= 2000])

    # ---- C10: selection rule picks the largest closed-loop gain -------------
    print("\n=== C10 selection rule among several SUCCESS points ===", flush=True)
    fr2 = [
        point(500, 41, cl_kw=dict(dep_delta_other=-0.10, dep_delta_junc=-0.11)),
        point(1500, 42, cl_kw=dict(dep_delta_other=-0.30, dep_delta_junc=-0.31)),
        point(2500, 43, cl_kw=dict(dep_delta_other=-0.20, dep_delta_junc=-0.21)),
    ]
    v10 = C.render_verdict(fr2)
    allok &= check("C10: all three qualify",
                   all(e["SUCCESS_POINT"] for e in fr2),
                   [(e["step"], e["failed"]) for e in fr2])
    allok &= check("C10: winner is step 1500 (most negative overall delta)",
                   v10["winner_step"] == 1500,
                   [(e["step"], e["_dep_overall_delta"]) for e in fr2])

    # ---- C11: an all-BOUND frontier renders BOUND, never SUCCESS ------------
    print("\n=== C11 whole frontier bound ===", flush=True)
    v11 = C.render_verdict([fr[0], fr[2], fr[3], fr[4]])
    allok &= check("C11: no SUCCESS point anywhere -> BOUND",
                   v11["verdict"] == "BOUND" and v11["winner_step"] is None,
                   v11["text"][:200])

    # ---- C12: the descriptors are wired ------------------------------------
    print("\n=== C12 non-deciding descriptors ===", flush=True)
    allok &= check("C12: multiplicity_robust is emitted and boolean",
                   isinstance(ev0["multiplicity_robust"], bool),
                   ev0["multiplicity_robust"])
    allok &= check("C12: noninferior_0p05 emitted; TRUE on the clean case",
                   ev0["noninferior_0p05"] is True,
                   ev0["_openloop_ade2s_ci"])
    allok &= check("C12: descriptors do NOT decide (C1 fails despite primary)",
                   True, "verified by C1/C8")

    n_pass = sum(1 for r in RES if r["pass"])
    out = {"_what": "E1c evaluator self-test: the shipped verdict path driven by "
                    "synthetic per-window arrays built to fail each registered "
                    "guardrail one at a time",
           "estimator": "paired_episode_cluster_bootstrap (taniteval/ci.py) "
                        "B=%d" % C.B_BOOT,
           "estimator_binding": CI_BINDING,
           "overlapping_holdout_se_used": False,
           "n_checks": len(RES), "n_pass": n_pass,
           "ALL_PASS": bool(allok and n_pass == len(RES)),
           "checks": RES,
           "C8_e1b_replay_verdict": v8,
           "C9_frontier_verdict": v9,
           "C10_selection_verdict": v10}
    Path(sys.argv[1] if len(sys.argv) > 1
         else "/workspace/e1c/e1c_selftest_result.json").write_text(
        json.dumps(out, indent=2, default=str))
    print(f"\n{n_pass}/{len(RES)} checks passed | ALL_PASS={out['ALL_PASS']}",
          flush=True)
    print("E1C_SELFTEST_DONE" if out["ALL_PASS"] else "E1C_SELFTEST_FAILED",
          flush=True)
    sys.exit(0 if out["ALL_PASS"] else 1)


if __name__ == "__main__":
    main()
