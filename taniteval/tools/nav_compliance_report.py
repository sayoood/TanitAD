#!/usr/bin/env python3
"""nav_compliance_report.py — the NAV-COMPLIANCE block from a refcv3_arm dump,
as a standalone artifact (0 GPU, re-analysis only).

    python taniteval/tools/nav_compliance_report.py --dump-dir <dump> \
        --labels <s2_labels_v7.2_eval.jsonl.gz> --out navcomp_<arm>.json \
        --arm refcv4b-9500 --tag "EARLY-TRAINING DIAGNOSTIC"

The dump must have been rolled by a ``refcv3_arm.py`` that writes the
nav-compliance sidecar keys (``plan_full_*``, ``gstr_*``, ``sel_bank_*``,
``fan_term_heading_*``, ``reach_keep_*``, ``gt_future_ext``, ``pose_last``,
``ego_t0``, ``ep_poses``); older dumps get an UNAVAILABLE block with the reason.

The artifact is written in the ``taniteval.driving`` schema so
``tools/criteria_check.py`` reads it: it carries the STRATEGIC family's
``nav_compliance`` block and REFUSES the other families with a reason pointing
at the companion ``<arm>.ARM.json`` record, where they live.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_TE = os.path.dirname(_HERE)                       # <repo>/taniteval
for p in (_TE, os.path.join(os.path.dirname(_TE), "stack")):
    if os.path.isdir(p) and p not in sys.path:
        sys.path.append(p)

from taniteval import nav_compliance as nc      # noqa: E402


def _fmt(r):
    if not isinstance(r, dict) or r.get("mean") is None and r.get("delta") is None:
        return "n/a"
    if "delta" in r:
        return (f"{r['delta']:+.4f} [{r.get('lo')}, {r.get('hi')}] "
                f"{'SEP' if r.get('separated') else 'n.s.'}"
                + (" DEGENERATE" if r.get("degenerate") else ""))
    return f"{r['mean']:.4f} [{r.get('lo')}, {r.get('hi')}]"


def render(rep: dict) -> str:
    if rep.get("status") == "UNAVAILABLE":
        return f"nav_compliance UNAVAILABLE: {rep.get('reason')}"
    L = [f"NAV-COMPLIANCE  tier={rep['tier']}  tag={rep.get('tag')}  "
         f"n_windows={rep['n_windows']} n_episodes={rep['n_episodes']}  "
         f"left/right/follow={rep['n_true_left']}/{rep['n_true_right']}/{rep['n_follow']}",
         f"  informative: plan n={rep['informative']['plan']['n_windows']} "
         f"({rep['informative']['plan']['n_episodes']} eps) · gstr "
         f"n={rep['informative']['gstr']['n_windows']} "
         f"({rep['informative']['gstr']['n_episodes']} eps)",
         f"  time base: {rep.get('time_base_control', {}).get('chosen')} "
         f"({rep.get('time_base_control', {}).get('hypotheses')})",
         f"  tolerance: {rep['tolerance']['values']}",
         f"  panel_ok={rep['panel_ok']}"]
    for name, blk in rep["readouts"].items():
        L.append(f"  [{name}]  powered={blk['powered']}  n={blk['n_informative_windows']}")
        for c, cb in blk["conditionings"].items():
            L.append(f"     {c:13s} C(true)={_fmt(cb.get('compliance_with_TRUE_command'))}")
        L.append(f"     true-shuffled {_fmt(blk.get('paired_true_minus_shuffled'))}   "
                 f"ceiling {(blk.get('always_commanded_ceiling') or {}).get('delta_true_minus_shuffled')}")
        L.append(f"     true-zero     {_fmt(blk.get('paired_true_minus_zero'))}")
        cs = blk.get("changed_subset") or {}
        if cs.get("follows_FED_command"):
            L.append(f"     changed n={cs['n_changed_fed_a_turn']}: follows FED "
                     f"{_fmt(cs['follows_FED_command'])}  follows TRUE "
                     f"{_fmt(cs['follows_TRUE_command'])}")
        v = rep["verdicts"].get(name) or {}
        L.append(f"     verdict: {v.get('verdict')} — {v.get('reason')}")
    for cname, cb in rep["controls"].items():
        L.append(f"  control {cname:8s} C(true)={_fmt(cb['compliance_with_TRUE_command'])}"
                 f"  delta≡{cb['delta_true_minus_shuffled']}")
    if "fan_coverage" in rep:
        for c, fc in rep["fan_coverage"].items():
            L.append(f"  fan[{c}] has-compliant={_fmt(fc['fan_has_compliant_candidate'])} "
                     f"mean#={fc['compliant_candidates_mean']}")
    if "seam" in rep:
        L.append(f"  SEAM: {rep['seam']['reading']} — {rep['seam']['_reads']}")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-dir", required=True)
    ap.add_argument("--labels", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", default="refcv3")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-delta", type=float, default=0.0,
                    help="the pre-registered minimum shuffle drop for FOLLOWS_NAV")
    ap.add_argument("--registry", default=None,
                    help="CRITERIA_REGISTRY.json (default: the repo's), used to decline "
                         "every non-nav-compliance criterion at its own key with a reason")
    a = ap.parse_args(argv)
    rep = nc.from_refcv3_dump(a.dump_dir, labels_path=a.labels, n_boot=a.n_boot,
                              seed=a.seed, tag=a.tag, min_delta=a.min_delta)
    art = _artifact(rep, a.arm, a.tag, registry_path=a.registry)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(art, fh, indent=1, default=str)
    print(render(rep))
    print(f"[out] {a.out}")


_REGISTRY_DEFAULT = os.path.join(os.path.dirname(_TE), "products", "P7-TanitEval",
                                 "CRITERIA_REGISTRY.json")


def _set_path(obj: dict, dotted: str, value) -> None:
    """Create ``obj[a][b]...[z] = value`` without clobbering an existing value."""
    parts = dotted.split(".")
    cur = obj
    for p in parts[:-1]:
        nxt = cur.get(p)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[p] = nxt
        cur = nxt
    cur.setdefault(parts[-1], value)


def _artifact(rep: dict, arm: str, tag, *, registry_path: str | None = None) -> dict:
    """The standalone artifact in the ``taniteval.driving`` schema the criteria
    checker reads (``tools/criteria_check.py``, registry v2.6.0).

    This artifact carries ONE family's block. Every OTHER binding criterion is
    declined the way the registry can read it: most criteria have no
    ``refused_as`` name, so a top-level ``refused`` block cannot reach them —
    the honest idiom is an INLINE ``{status: UNAVAILABLE, reason, n}`` at the
    criterion's own key, which the checker classifies as a WORK ITEM pointing
    at the companion record where those families live. The tier travels in
    ``block`` (``taniteval.driving/tier1``), and the nav-blind floors are
    exposed as ``floors`` / ``vs_floor_paired`` (the ``ctrl.floor_comparison``
    criterion: a number without its trivial control is unreadable)."""
    companion = f"{arm}.ARM.json (refcv3_arm.py record on the SAME dump)"
    tier = rep.get("tier", "T1")
    n = int(rep.get("n_windows") or 0)
    ctl = rep.get("controls") or {}
    plan_true = (((rep.get("readouts") or {}).get("plan") or {}).get("conditionings") or {}
                 ).get("nav_true", {}).get("compliance_with_TRUE_command") or {}
    floors, vs = {}, {}
    for cname in ("ha0", "ha0_ext", "ha"):
        c = ctl.get(cname)
        if not c:
            continue
        r = c.get("compliance_with_TRUE_command") or {}
        floors[cname] = {"plan_compliance": r.get("mean"),
                         "_is": ("nav-blind: the straight line" if cname == "ha0" else
                                 "nav-blind: constant-(a, kappa) extrapolation of the measured "
                                 "t0 state" if cname == "ha0_ext" else
                                 "nav-blind: the held action closing at t0")}
        if plan_true.get("mean") is not None and r.get("mean") is not None:
            d = float(plan_true["mean"]) - float(r["mean"])
            vs[cname] = {"plan_compliance": {
                "delta_model_minus_floor": round(d, 4),
                "model_ci_lo_above_floor": bool(plan_true.get("lo", 0) > r["mean"]),
                "favours": "model" if plan_true.get("lo", 0) > r["mean"] else
                           ("floor" if plan_true.get("hi", 1) < r["mean"] else "tie"),
                "_note": ("the floor is deterministic given the windows; 'model' requires the "
                          "model's CI lower bound to clear it")}}
    art = {
        "block": f"taniteval.driving/{ {'T0': 'tier0', 'T1': 'tier1', 'T2': 'tier2'}.get(tier, 'tier1') }",
        "tier": tier,
        "arm": arm, "tag": tag, "tool": "taniteval/tools/nav_compliance_report.py",
        "companion_record": companion,
        "n_windows": n, "n_episodes": rep.get("n_episodes"),
        "estimator": {"interval": "episode_cluster_bootstrap",
                      "paired": "paired_episode_cluster_bootstrap",
                      "deprecated_and_refused": "overlapping_holdout_se"},
        "protocol": {
            "inference_inputs": ("the OBSERVED window's frames; the MEASURED ego state at "
                                 "t0 (v0, a_long, yaw_rate, curvature — PI 2026-09-02/03); "
                                 "the clip's v7.2 nav token under nav_true / permuted under "
                                 "nav_shuffled / withheld under nav_zero / left<->right under "
                                 "nav_flipped"),
            "vision_only": "vision + ego(t0) + nav token; no future",
            "goal_source": ("g_str is the model's own strategic head; the plan is the model's "
                            "own selection; nothing here reads a label at inference"),
            "goal_situation_disjoint": True,
            "corpus": ((rep.get("join") or {}).get("labels")),
            "parity_key": "the v7.2 EVAL clip set (labels blob md5 in the companion record)",
            "tier": tier,
        },
        "floors": floors, "vs_floor_paired": vs,
        "strategic": {"nav_compliance": rep,
                      "echo_test": {"kind": "intervention pair (nav_shuffled + nav_zero"
                                            " [+ nav_flipped])",
                                    "bijection_with_input": False,
                                    "_reads": nc.FALSIFIABILITY}},
    }
    # ---- decline every other binding criterion AT ITS KEY, with the reason -- #
    reg_path = registry_path or _REGISTRY_DEFAULT
    try:
        reg = json.load(open(reg_path, encoding="utf-8"))
    except OSError:
        reg = None
        art["_registry_not_read"] = reg_path
    if reg:
        why_other = (f"not this artifact's family — this artifact carries the STRATEGIC "
                     f"nav-compliance block only; the four families on the SAME dump and "
                     f"windows live in the companion record {companion}")
        why_old = ("the route/decision label is a bijection of the fed nav token "
                   "(D-REFAV1-ROUTE-LABEL-IS-THE-NAV, 141/141): the head metric cannot be "
                   "falsified and is superseded by strategic.nav_compliance; the old block "
                   f"remains in {companion} as an echo diagnostic")
        for fam, blk in (reg.get("families") or {}).items():
            for crit in blk.get("criteria", []):
                cid = crit.get("id", "")
                if cid.startswith("strat.nav_compliance"):
                    continue
                keys = crit.get("keys") or []
                if not keys:
                    continue
                key = next((k for k in keys if not k.startswith("four_families.")
                            and not k.startswith("refcv3.")), keys[0])
                reason = why_old if cid in ("strat.decision", "strat.route_goal") else why_other
                _set_path(art, key, {"status": "UNAVAILABLE", "reason": reason, "n": n})
    return art


if __name__ == "__main__":
    main()
