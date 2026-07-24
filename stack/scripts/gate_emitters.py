"""gate_emitters.py -- the three v4-gate KILL secondaries that had NO emitter.

WHY THIS EXISTS (2026-07-23)
----------------------------
``flagship-v4.card.json`` lists 8 KILL secondaries. ``run_gate.py check`` marks
any card secondary with no supplied ``--secondary-value`` as ``pass: None`` and
the whole verdict as **INCOMPLETE** (run_gate.py:617). Five of the eight had an
emitter; **three did not**, so no v4 gate could render a COMPLETE formal verdict
(v4.1's real 10 k gate came out INCOMPLETE for exactly this reason):

  * ``deploy_tick_p99_ms``          <= 50   the arm is undeployable if it breaches
  * ``speed_benefit_recovered_frac`` >= 0.70  the quiet v3enc plateau the canary misses
  * ``nonav_route_beats_majority``   >= 1    strategic route value, or a relabelled echo

STEP-0 RECONCILIATION (banked). LOOP_STATE called these "P7, report-only,
NON-blocking"; the registry agent read the card as "KILL". **The registry agent is
right.** ``flagship-v4.card.json`` puts all three in the ``secondary`` array, and
every on-card secondary is KILL by construction (run_gate has NO report-only flag
for card secondaries -- report-only is a separate OFF-card ``--secondary-value``
channel, run_gate.py:607-615). V4_FLAGSHIP_DESIGN.md's §9 split-card table marks
all three **KILL** explicitly. The "P7 report-only" set is a DIFFERENT, off-card
group -- ``imag_win_at_5s`` / ``strat_subspace_{sufficiency,compression}`` /
``longh_5s_beats_persistence`` / ``cruise_delta_vs_holdv0`` -- and the two sets
were conflated.

DESIGN §17.3 says each is emitted by an EXISTING panel; the gap was surfacing the
one gate-named number and passing it to ``run_gate --secondary-value``:

  | secondary                      | panel (emitter)                        |
  |--------------------------------|----------------------------------------|
  | deploy_tick_p99_ms             | ``taniteval.efficiency`` lever panel   |
  | nonav_route_beats_majority     | ``taniteval.hierarchy`` (JSON key      |
  |                                |   ``vision_route_beats_majority``)     |
  | speed_benefit_recovered_frac   | NEW ``tanitad.eval.speed_benefit``     |

This module READS those panels' committed/produced JSON and emits the gate value +
provenance + evidence class, and (``gate-values``) prints the exact
``--secondary-value name=value`` strings ``run_gate.py check`` consumes. It never
re-implements a panel; the numbers are the panels' own.

Validated against the deployed flagship **v1** (``flagship-30k``, step 29999) --
the §17.1b dry-run fixture whose every number is known:
  * deploy_tick_p99_ms   = 18.76 ms  (all_levers composed tick, A40, PASS <=50)
  * speed_benefit_frac   = 0.8184     (8-10 k, PASS >=0.70 -- design's 81.8 %)
  * nonav_route_beats_maj = 0         (route_acc_follow 0.7083 == majority 0.7083,
                                       the pure command echo -- FAILS, correctly)
So the honest v1 verdict is COMPLETE/RESTART (nonav_route fails); COMPLETE, not
INCOMPLETE, is the deliverable -- the machinery now renders a full verdict.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ============================================================================ #
# deploy_tick_p99_ms  --  taniteval.efficiency lever panel                     #
# ============================================================================ #
# The DEPLOYED tick is the fully-composed inference variant the car runs: encode
# + rollout with every accuracy-preserving lever on (fp16 weights + rolling
# encoder cache + whole-rollout CUDA graph). Preference order picks the most-
# composed lever first, each falling back if a build failed on the measuring pod.
# ``all_levers`` is the deployment config; the graph variants below it are the
# precision-exact alternatives, in decreasing composition.
DEPLOY_LEVER_PREFERENCE = (
    "all_levers", "fp16_enc_cache_graph", "enc_cache_graph",
    "fp16_graph_rollout", "graph_rollout", "graph_fulltick", "graph_step")
DEPLOY_TICK_THRESHOLD_MS = 50.0
# an equivalence guard: a composed tick that is NOT accuracy-equivalent to the
# eager reference is a fast WRONG answer -- reject it as the deploy tick.
DEPLOY_ADE_DELTA_TOL_M = 0.05


def _pick_lever(levers: dict, preference=DEPLOY_LEVER_PREFERENCE):
    """First lever in ``preference`` that BUILT (has a ``tick`` with p99) and is
    accuracy-equivalent (finite equivalence, |ade_0_2s_delta| <= tol). Returns
    ``(name, node)`` or ``(None, None)``."""
    for name in preference:
        node = levers.get(name)
        if not isinstance(node, dict):
            continue
        tick = node.get("tick") or node.get("plan_step")
        if not (isinstance(tick, dict) and tick.get("p99_ms") is not None):
            continue
        eq = node.get("equivalence") or {}
        d = eq.get("ade_0_2s_delta_m")
        finite = eq.get("finite", True)
        if finite is False:
            continue
        if d is not None and abs(float(d)) > DEPLOY_ADE_DELTA_TOL_M:
            continue                        # a fast WRONG tick -- not deployable
        return name, node
    return None, None


def deploy_tick_from_eff_json(path, precision="fp32",
                              preference=DEPLOY_LEVER_PREFERENCE,
                              threshold=DEPLOY_TICK_THRESHOLD_MS) -> dict:
    """Emit ``deploy_tick_p99_ms`` from a ``taniteval.efficiency`` LEVER panel
    JSON (``eff_levers_<key>.json``). Reads the composed deployed tick's p99, NOT
    the eager baseline (which is the un-optimised ~100 ms tick, not deployed)."""
    ev = json.loads(Path(path).read_text(encoding="utf-8"))
    r = deploy_tick_from_eff_json_dict(ev, precision, preference, threshold)
    r.setdefault("provenance", {})["eff_lever_panel"] = str(path)
    if r.get("value") is not None:              # stamp the artifact filename
        block = ev.get(precision) or ev.get("fp32") or ev.get("tf32") or {}
        r["evidence_class"] = (f"MEASURED ({block.get('env', {}).get('gpu', 'GPU')}"
                               f"; artifact = {Path(path).name})")
    return r


def deploy_tick_from_eff_json_dict(ev: dict, precision="fp32",
                                   preference=DEPLOY_LEVER_PREFERENCE,
                                   threshold=DEPLOY_TICK_THRESHOLD_MS) -> dict:
    """Logic core of :func:`deploy_tick_from_eff_json`, on an already-loaded
    panel dict (so it is testable without a file)."""
    block = ev.get(precision) or ev.get("fp32") or ev.get("tf32") or {}
    levers = block.get("levers")
    if not isinstance(levers, dict):
        return {"gate_metric": "deploy_tick_p99_ms", "value": None, "pass": None,
                "note": f"no '{precision}'.levers block -- is this a LEVER panel "
                        "(eff_levers_*.json)? the eager baseline panel is not the "
                        "deployed tick"}
    name, node = _pick_lever(levers, preference)
    if node is None:
        return {"gate_metric": "deploy_tick_p99_ms", "value": None, "pass": None,
                "note": "no accuracy-equivalent composed lever built in this panel"}
    tick = node.get("tick") or node.get("plan_step")
    p99 = round(float(tick["p99_ms"]), 4)
    eq = node.get("equivalence") or {}
    contam = block.get("contamination_check") or {}
    return {
        "gate_metric": "deploy_tick_p99_ms",
        "value": p99,
        "threshold": threshold,
        "direction": "<=",
        "pass": bool(p99 <= threshold),
        "evidence_class": f"MEASURED ({block.get('env', {}).get('gpu', 'GPU')}; "
                          "efficiency lever panel)",
        "deployed_lever": name,
        "lever_desc": (node.get("meta") or {}).get("desc"),
        "weights_dtype": (node.get("meta") or {}).get("weights_dtype"),
        "precision_block": precision,
        "tick_ms": {k: tick.get(k) for k in
                    ("mean_ms", "p50_ms", "p95_ms", "p99_ms", "std_ms",
                     "iters", "warmup")},
        "accuracy_equivalence": {
            "ade_0_2s_delta_m": eq.get("ade_0_2s_delta_m"),
            "cosine": eq.get("cosine"), "finite": eq.get("finite"),
            "note": "the deployed composed tick must decode the SAME trajectory "
                    "as the eager reference (a fast wrong tick is worthless)"},
        "gpu_exclusive": contam.get("valid"),
        "v4_delta_note": ("v4's operative predictor is v1-verbatim; the anchored-"
                          "diffusion head adds `diffusion_steps` truncated-denoise "
                          "passes to the tick (the tick KNOB, V4_FLAGSHIP_DESIGN "
                          "§8: ~25-28 ms floor with the imagination probe, first "
                          "thing cut if the arm breaches 50). Measure the composed "
                          "tick on the v4 ckpt to include the head."),
        "provenance": {"emitter": "taniteval.efficiency (lever panel)"},
    }


# ============================================================================ #
# nonav_route_beats_majority  --  taniteval.hierarchy panel                     #
# ============================================================================ #
# §17.3: the emitter is hierarchy.py's ``vision_route_beats_majority`` JSON key
# (under seam_nav_to_strategic), vs ``majority_straight_rate``. The gate value is
# that boolean as an int (>=1 passes). §7A.5: with the command WITHHELD (follow),
# route accuracy on the valid subset must beat the majority-class (straight) base
# rate -- v1's route head is a pure command echo (route_skill_vs_chance 0.0), so
# with the command gone it collapses to constant-straight and cannot clear the bar.
NONAV_ROUTE_MARGIN = 0.03                 # hierarchy.py's practical margin


def nonav_route_from_hierarchy_json(path) -> dict:
    """Emit ``nonav_route_beats_majority`` (int 0/1) from a ``taniteval.hierarchy``
    result JSON."""
    ev = json.loads(Path(path).read_text(encoding="utf-8"))
    r = nonav_route_from_hierarchy_dict(ev)
    r["provenance"]["hierarchy_panel"] = str(path)
    return r


def nonav_route_from_hierarchy_dict(ev: dict) -> dict:
    """Logic core of :func:`nonav_route_from_hierarchy_json`, on an already-loaded
    hierarchy dict (testable without a file)."""
    sn = ev.get("seam_nav_to_strategic") or {}
    beats = sn.get("vision_route_beats_majority")
    acc_follow = sn.get("route_acc_follow")
    straight = sn.get("majority_straight_rate")
    if beats is None and acc_follow is not None and straight is not None:
        beats = acc_follow > straight + NONAV_ROUTE_MARGIN
    value = None if beats is None else int(bool(beats))
    return {
        "gate_metric": "nonav_route_beats_majority",
        "value": value,
        "threshold": 1,
        "direction": ">=",
        "pass": (None if value is None else bool(value >= 1)),
        "evidence_class": "MEASURED (ours; hierarchy panel on the ckpt)",
        "route_acc_follow": acc_follow,
        "route_acc_nav_commanded": sn.get("route_acc_nav"),
        "route_acc_zeronav": sn.get("route_acc_zeronav"),
        "majority_straight_rate": straight,
        "margin": NONAV_ROUTE_MARGIN,
        "follow_pred_distribution": sn.get("follow_pred_distribution"),
        "n_valid": sn.get("n_valid"),
        "reading": ("1 = the produced (no-command) route head beats always-"
                    "straight -> a real strategic level; 0 = command echo / "
                    "constant-straight (route_acc_follow == majority)"),
        "provenance": {"emitter": "taniteval.hierarchy "
                                  "(seam_nav_to_strategic.vision_route_beats_majority)"},
    }


# ============================================================================ #
# speed_benefit_recovered_frac  --  tanitad.eval.speed_benefit                   #
# ============================================================================ #
def speed_benefit_emit(arm_log, nospeed_log=None, repo_root=None) -> dict:
    """Thin adapter to ``tanitad.eval.speed_benefit.emit`` so all three emitters
    share one CLI. Imported lazily so a torch-less environment can still run the
    deploy-tick / nonav-route readers."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # stack/
    import importlib.util
    sb_path = Path(__file__).resolve().parents[1] / "tanitad" / "eval" / "speed_benefit.py"
    spec = importlib.util.spec_from_file_location("_speed_benefit", sb_path)
    sb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sb)                        # no torch import -> safe
    kw = {}
    if nospeed_log:
        kw["nospeed_log"] = nospeed_log
    if repo_root:
        kw["repo_root"] = repo_root
    return sb.emit(arm_log, **kw)


# ============================================================================ #
# gate-values : assemble all three -> the --secondary-value strings run_gate eats#
# ============================================================================ #
GATE_NAMES = ("deploy_tick_p99_ms", "speed_benefit_recovered_frac",
              "nonav_route_beats_majority")


def _fmt_value(name, v):
    """run_gate's ``_KV`` action parses ``float(val)``; a bool int stays an int
    string, a float stays a float string."""
    if isinstance(v, bool):
        return str(int(v))
    return repr(v)


def gate_values(eff_json=None, hierarchy_json=None, arm_log=None,
                nospeed_log=None, repo_root=None, precision="fp32") -> dict:
    """Compute all three secondaries and the exact ``--secondary-value`` args."""
    out: dict = {"emitted_utc": None, "secondaries": {}}
    from datetime import datetime, timezone
    out["emitted_utc"] = datetime.now(timezone.utc).isoformat()
    if eff_json:
        out["secondaries"]["deploy_tick_p99_ms"] = deploy_tick_from_eff_json(
            eff_json, precision=precision)
    if arm_log:
        out["secondaries"]["speed_benefit_recovered_frac"] = speed_benefit_emit(
            arm_log, nospeed_log, repo_root)
    if hierarchy_json:
        out["secondaries"]["nonav_route_beats_majority"] = \
            nonav_route_from_hierarchy_json(hierarchy_json)

    args, missing = [], []
    for name in GATE_NAMES:
        row = out["secondaries"].get(name)
        if row and row.get("value") is not None:
            args.append(f"{name}={_fmt_value(name, row['value'])}")
        else:
            missing.append(name)
    out["secondary_value_args"] = args
    out["missing"] = missing
    out["all_three_emitted"] = (len(missing) == 0)
    return out


# ============================================================================ #
# CLI                                                                          #
# ============================================================================ #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        "gate_emitters", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("deploy-tick", help="deploy_tick_p99_ms from an eff lever panel")
    d.add_argument("--eff-json", required=True)
    d.add_argument("--precision", default="fp32", choices=["fp32", "tf32", "amp16"])
    d.add_argument("--out", default=None)

    r = sub.add_parser("nonav-route", help="nonav_route_beats_majority from a hierarchy JSON")
    r.add_argument("--hierarchy-json", required=True)
    r.add_argument("--out", default=None)

    s = sub.add_parser("speed-benefit", help="speed_benefit_recovered_frac from train logs")
    s.add_argument("--arm-log", required=True)
    s.add_argument("--nospeed-log", default=None)
    s.add_argument("--repo-root", default=None)
    s.add_argument("--out", default=None)

    g = sub.add_parser("gate-values", help="all three + the --secondary-value args")
    g.add_argument("--eff-json", default=None)
    g.add_argument("--hierarchy-json", default=None)
    g.add_argument("--arm-log", default=None)
    g.add_argument("--nospeed-log", default=None)
    g.add_argument("--repo-root", default=None)
    g.add_argument("--precision", default="fp32", choices=["fp32", "tf32", "amp16"])
    g.add_argument("--out", default=None)

    a = ap.parse_args(argv)
    if a.cmd == "deploy-tick":
        res = deploy_tick_from_eff_json(a.eff_json, precision=a.precision)
    elif a.cmd == "nonav-route":
        res = nonav_route_from_hierarchy_json(a.hierarchy_json)
    elif a.cmd == "speed-benefit":
        res = speed_benefit_emit(a.arm_log, a.nospeed_log, a.repo_root)
    else:
        res = gate_values(a.eff_json, a.hierarchy_json, a.arm_log, a.nospeed_log,
                          a.repo_root, a.precision)
        if res.get("secondary_value_args"):
            print("# run_gate.py check ... --secondary-value \\")
            print("    " + " ".join(res["secondary_value_args"]))
    print(json.dumps(res, indent=2))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
