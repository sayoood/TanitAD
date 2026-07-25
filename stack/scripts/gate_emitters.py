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

THE CO-PRIMARY EMITTER (added 2026-07-26, Tier-1 #1)
----------------------------------------------------
``run_gate.py`` gained a horizon-honest **co-primary**
(``corridor_departure_rate`` at a pre-registered K) and demoted ``ade_0_2s`` to
a diagnostic. That co-primary needs an emitter for the same reason the three
secondaries above did: without one, every gate renders INCOMPLETE.

``corridor`` / ``corridor-arg`` below read a :mod:`taniteval.corridor` result
JSON -- or compute one from a persisted ``windows_<arm>.pt`` -- and print the
exact ``--corridor-json`` argument ``run_gate.py check`` consumes.

⚠️ MEASURED 2026-07-26, and this is the current blocker: **0 of the 30 committed
``windows_*.pt`` dumps carry ``pred_dense``/``gt_dense``**; every one is the
4-waypoint sparse view with ``wp_steps = [5, 10, 15, 20]``. So
:func:`corridor_from_windows` returns ``taniteval.corridor``'s self-describing
``skipped`` node for all of them, and the open-loop dense surface caps at
**K=20 (2.0 s)** even once the dense keys land -- which is the blind horizon
itself. **A K>=100 co-primary needs a CLOSED-LOOP rollout (E1a's surface), i.e.
GPU.** The emitter says so instead of fabricating a number at the wrong horizon.
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
# corridor_departure_rate @ K  --  taniteval.corridor  (THE CO-PRIMARY)         #
# ============================================================================ #
CORRIDOR_METRIC = "corridor_departure_rate"
CORRIDOR_HALFWIDTH_M = 1.75            # taniteval.corridor.CORRIDOR_HALFWIDTH_M
HORIZON_CEILING_K = 190                # 190-199-frame clips, T - W - K >= 1
ADE2S_K = 20                           # the blind horizon, named not implied


def corridor_from_corridor_json(path, stratum="overall",
                                metric=CORRIDOR_METRIC) -> dict:
    """Emit the co-primary from an already-computed ``taniteval.corridor`` JSON.

    Pure JSON, no torch: this is the path that works on a dev box. The value is
    the panel's own -- nothing is recomputed -- and the horizon travels with it,
    because a corridor number without its K is exactly the defect the co-primary
    replaces."""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    r = corridor_from_corridor_dict(doc, stratum=stratum, metric=metric)
    r.setdefault("provenance", {})["corridor_panel"] = str(path)
    if r.get("value") is not None:
        r["run_gate_arg"] = f"--corridor-json {path}"
    return r


def corridor_from_corridor_dict(doc: dict, stratum="overall",
                                metric=CORRIDOR_METRIC) -> dict:
    """Logic core of :func:`corridor_from_corridor_json` (testable without a file)."""
    out = {"gate_metric": metric, "stratum": stratum, "value": None,
           "provenance": {"emitter": "taniteval.corridor"}}
    if not isinstance(doc, dict):
        out["note"] = "not a corridor result JSON"
        return out
    if doc.get("skipped"):
        out["note"] = f"corridor panel SKIPPED: {doc['skipped']}"
        out["dense_surface_available"] = doc.get("dense_surface_available")
        out["evidence_class"] = "NOT MEASURED (no dense/closed-loop surface)"
        return out
    node = doc
    for key in ("corridor", "co_primary", "driving"):
        if not isinstance(node.get(stratum), dict) and isinstance(node.get(key), dict):
            node = node[key]
    blk = node.get(stratum)
    if not isinstance(blk, dict):
        out["note"] = (f"no {stratum!r} stratum in this document "
                       f"(keys: {sorted(node)[:12]})")
        return out
    m = blk.get(metric)
    if not isinstance(m, dict) or "mean" not in m:
        out["note"] = f"{metric!r} is not an interval node in stratum {stratum!r}"
        return out
    K = blk.get("horizon_K")
    junction = node.get("junction") if isinstance(node.get("junction"), dict) else None
    out.update({
        "value": float(m["mean"]), "lo": m.get("lo"), "hi": m.get("hi"),
        "estimator": m.get("estimator"), "n_boot": m.get("n_boot"),
        "horizon_K": K,
        "horizon_s": blk.get("horizon_s",
                             None if K is None else round(float(K) * 0.1, 2)),
        "corridor_primary_m": blk.get("corridor_primary_m"),
        "n_windows": blk.get("n_windows"), "n_episodes": blk.get("n_episodes"),
        "surface": blk.get("surface"),
        "evidence_class": (f"MEASURED (ours; taniteval.corridor, "
                           f"{m.get('estimator')}, n={blk.get('n_windows')} "
                           f"windows / {blk.get('n_episodes')} episodes)"),
        # reported SEPARATELY, always -- 0.8414 vs 0.5877 at K=185
        "junction": None if junction is None else {
            "value": (junction.get(metric) or {}).get("mean"),
            "lo": (junction.get(metric) or {}).get("lo"),
            "hi": (junction.get(metric) or {}).get("hi"),
            "n_windows": junction.get("n_windows"),
            "n_episodes": junction.get("n_episodes")},
        "horizon_note": (
            "a corridor number is meaningless without its K: on E1a's own 43 "
            "windows the SAME trajectories give 0.0035 at K=20 and 0.5877 at "
            f"K=185. The corpus ceiling is K={HORIZON_CEILING_K} "
            f"({HORIZON_CEILING_K * 0.1:.1f} s)."),
    })
    if K is not None and int(K) <= ADE2S_K:
        out["WARNING_blind_horizon"] = (
            f"K={K} is at or below ade_0_2s' own horizon ({ADE2S_K} = "
            f"{ADE2S_K * 0.1:.1f} s). run_gate.py will REFUSE this as a "
            f"co-primary.")
    return out


def corridor_from_windows(windows_path, halfwidth=CORRIDOR_HALFWIDTH_M,
                          n_boot=2000, out_json=None) -> dict:
    """Compute the co-primary from a persisted ``windows_<arm>.pt``.

    Lazily imports :mod:`taniteval.corridor` (torch) so the torch-free emitters
    above keep working on a box without it. Returns the same shape as
    :func:`corridor_from_corridor_json`, including the honest ``skipped`` node
    when the dump carries no dense path -- which, MEASURED 2026-07-26, is all 30
    of them."""
    repo = Path(__file__).resolve().parents[2]
    for p in (repo / "taniteval", repo / "stack", repo / "stack" / "scripts"):
        sys.path.insert(0, str(p))
    from taniteval import corridor as C          # noqa: E402  (lazy: needs torch)
    from taniteval import rollout                # noqa: E402
    res = C.from_windows(rollout.load_windows(str(windows_path)),
                         primary=halfwidth, n_boot=n_boot)
    if out_json:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(out_json).write_text(json.dumps(res, indent=2, default=str),
                                  encoding="utf-8")
    r = corridor_from_corridor_dict(res)
    r["provenance"]["windows"] = str(windows_path)
    if out_json:
        r["provenance"]["corridor_panel"] = str(out_json)
        r["run_gate_arg"] = f"--corridor-json {out_json}"
    if res.get("skipped"):
        r["blocker"] = (
            "This dump has no dense path, so no corridor number exists at ANY "
            "horizon. Note that even with pred_dense/gt_dense the open-loop "
            f"surface caps at K={ADE2S_K} ({ADE2S_K * 0.1:.1f} s) -- the blind "
            "horizon. A K>=100 co-primary requires a CLOSED-LOOP rollout "
            "(E1a's surface, e1a_horizon.py), which needs GPU.")
    return r


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
                nospeed_log=None, repo_root=None, precision="fp32",
                corridor_json=None) -> dict:
    """Compute all three secondaries and the exact ``--secondary-value`` args.

    ``corridor_json`` additionally emits the CO-PRIMARY into ``co_primary`` and
    the matching ``--corridor-json`` argument. It is deliberately kept OUT of
    ``GATE_NAMES``/``secondary_value_args``: the co-primary is not a secondary
    and must never be passed through the ``--secondary-value`` channel, where
    an off-card value silently becomes report-only."""
    out: dict = {"emitted_utc": None, "secondaries": {}}
    from datetime import datetime, timezone
    out["emitted_utc"] = datetime.now(timezone.utc).isoformat()
    if corridor_json:
        out["co_primary"] = corridor_from_corridor_json(corridor_json)
        out["co_primary_arg"] = out["co_primary"].get("run_gate_arg")
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

    k = sub.add_parser("corridor",
                       help="THE CO-PRIMARY: corridor_departure_rate @ K, from a "
                            "taniteval.corridor JSON or a windows_<arm>.pt dump")
    k.add_argument("--corridor-json", default=None,
                   help="an already-computed taniteval.corridor result JSON")
    k.add_argument("--windows", default=None,
                   help="a persisted windows_<arm>.pt (needs torch); writes the "
                        "corridor panel to --out-corridor")
    k.add_argument("--out-corridor", default=None,
                   help="where to write the computed corridor panel")
    k.add_argument("--stratum", default="overall",
                   choices=["overall", "junction", "longitudinal", "other"])
    k.add_argument("--halfwidth", type=float, default=CORRIDOR_HALFWIDTH_M)
    k.add_argument("--n-boot", type=int, default=2000)
    k.add_argument("--out", default=None)

    g = sub.add_parser("gate-values", help="all three + the --secondary-value args")
    g.add_argument("--eff-json", default=None)
    g.add_argument("--hierarchy-json", default=None)
    g.add_argument("--arm-log", default=None)
    g.add_argument("--nospeed-log", default=None)
    g.add_argument("--repo-root", default=None)
    g.add_argument("--corridor-json", default=None,
                   help="the CO-PRIMARY panel; emitted as --corridor-json, never "
                        "as a --secondary-value")
    g.add_argument("--precision", default="fp32", choices=["fp32", "tf32", "amp16"])
    g.add_argument("--out", default=None)

    a = ap.parse_args(argv)
    if a.cmd == "deploy-tick":
        res = deploy_tick_from_eff_json(a.eff_json, precision=a.precision)
    elif a.cmd == "nonav-route":
        res = nonav_route_from_hierarchy_json(a.hierarchy_json)
    elif a.cmd == "speed-benefit":
        res = speed_benefit_emit(a.arm_log, a.nospeed_log, a.repo_root)
    elif a.cmd == "corridor":
        if not (a.corridor_json or a.windows):
            raise SystemExit("[gate_emitters] corridor needs --corridor-json or "
                             "--windows")
        res = (corridor_from_corridor_json(a.corridor_json, stratum=a.stratum)
               if a.corridor_json else
               corridor_from_windows(a.windows, halfwidth=a.halfwidth,
                                     n_boot=a.n_boot, out_json=a.out_corridor))
        if res.get("run_gate_arg"):
            print("# run_gate.py check ... \\")
            print("    " + res["run_gate_arg"])
    else:
        res = gate_values(a.eff_json, a.hierarchy_json, a.arm_log, a.nospeed_log,
                          a.repo_root, a.precision, corridor_json=a.corridor_json)
        if res.get("co_primary_arg"):
            print("# run_gate.py check ... \\")
            print("    " + res["co_primary_arg"] + " \\")
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
