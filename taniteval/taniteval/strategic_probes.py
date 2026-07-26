"""TanitEval — HP-3: the COUNTERFACTUAL ROUTE-SWAP probe.

THE PREDICTION, AND WHY IT IS THE CHEAPEST DISCRIMINATING EXPERIMENT
--------------------------------------------------------------------
``01_EXECUTION_PLAN.md`` Part A, HP-3:

    **Route-conditionality: same scene + different ``nav_cmd`` ⇒ different,
    correct trajectory.** A flat marginal model *cannot* do this.
    *Falsifier:* trajectories identical under different commands ⇒ still a
    command echo (PC1 regression).

This is the discriminating prediction with the **best cost/information ratio in
the whole battery**: it needs **zero training**, zero new forward-pass
machinery, and one encode per window. ``hierarchy.py`` already runs three
strategic passes on a single encode (real ``nav`` / ``follow`` / zeroed-nav) —
this module keeps that structure and changes the question from *route
ACCURACY* to *trajectory DIVERGENCE*, which is what HP-3 actually asks
(HPP-0 audit §3.2: *"absent; hierarchy.py:277-283 is the hook to build it on"*).

A flat model that has learned the marginal over trajectories emits the **same**
path whatever the command. It fails HP-3 **by construction** — that is the
point, and it is why a null here is informative rather than merely
disappointing.

⚠️ WHAT TODAY'S ARMS WILL SCORE, AND WHY THAT IS NOT THE MODEL'S FAULT
----------------------------------------------------------------------
**Every arm scored to date is expected to score ~0 divergence.** Three
independent, MEASURED reasons — none of them evidence against the hierarchy:

1. **The label circularity.** ``refb_labels.route_target(nav_cmd) =
   _NAV_TO_ROUTE[nav_cmd]`` — the route TARGET is a deterministic lookup of the
   route INPUT, so the head reaches route CE **exactly 0.0** by step ~14.5 k by
   copying its own conditioning embedding to its logits. Nothing in training
   ever rewarded the command *changing the trajectory*, only the *route logit*.
   ``route_skill = 0.0000`` on two checkpoints, two window counts, two
   architectures (HPP-0 §1.5).
2. **Coverage.** ``nav_command`` returns ``(NAV_FOLLOW, False)`` on ~73 % of
   windows and ``NAV_FOLLOW`` is still **fed to the model**, so the strategic
   level is told "follow" on three quarters of windows including mid-turn ones.
3. **The scored operative rollout is intent-free by design**
   (``metric_dynamics.rollout_decode`` takes no ``intent``/``ctx``/``nav``), so
   on the *deployed* surface the route **structurally cannot** reach the
   trajectory at all. :func:`run` reports this as
   ``route_can_reach_scored_trajectory`` rather than letting a 0 be read as a
   model verdict.

**So this probe's first real use is AFTER HPP-1's label fix** — it is the
instrument that will show whether the fix worked. Running it before then
establishes the pre-fix baseline, which is exactly what a pre-registered
experiment needs. Both outcomes are committed in advance:

* divergence CI-separated from 0 **and** directionally correct above chance
  ⇒ **HP-3 PASSES**: the strategic command is causally steering the trajectory.
* divergence not separated ⇒ **HP-3 FAILS**: still a command echo. Report it as
  a PC1 regression, never as "the hierarchy does not help".

TWO SURFACES, AND ONLY ONE IS IN-REGIME
---------------------------------------
* ``tactical_waypoints`` — **the primary.** The tactical head's 2 s waypoints
  under each command. In-regime: the head is conditioned on the strategic ctx
  during training, so this is where a route effect is *supposed* to appear.
* ``grounded_rollout`` — an **OUT-OF-REGIME diagnostic**, threading the branch's
  intent into the 20-step recursive grounded rollout whose step-readout was
  calibrated intent-free. ``hierarchy.py`` carries the identical caveat under
  ``diagnostic_intent_in_grounded_rollout_OUT_OF_REGIME``. A blow-up here shows
  the intent FiLM *perturbs* the operative predictor; it is not evidence about
  route quality.

THE CHANNEL: CROSS-TRACK, NOT ADE
---------------------------------
Route choice is a **lateral** phenomenon. ``LATERAL_VS_LONGITUDINAL_ANALYSIS.md``
§M6 states it directly: HP-2 and HP-3 are measured in the **cross-track
channel**, because ADE is 98.6 % longitudinal by squared-error energy and would
dilute a route effect into invisibility. Every divergence here is therefore
reported **both** as a full L2 and as its lateral component
(:mod:`taniteval.lateral`), and the lateral one is the headline.
"""
from __future__ import annotations

import sys

import numpy as np
import torch

sys.path.insert(0, "/root/TanitAD/stack")
sys.path.insert(0, "/root/TanitAD/stack/scripts")
sys.path.insert(0, "/root/taniteval")

import refb_labels as rl  # noqa: E402
from driving_diagnostic import WP_STEPS  # noqa: E402

from taniteval import ci as _ci  # noqa: E402
from taniteval import driving as _drv  # noqa: E402
from taniteval import hierarchy as _hier  # noqa: E402
from taniteval import lateral as _lat  # noqa: E402

BLOCK = "taniteval.strategic_probes/hp3_route_counterfactual"
VERSION = "1.0.0"
SPEC = ("Project Steering/Reviews/2026-07-25-independent-chief-scientist-review/"
        "01_EXECUTION_PLAN.md Part A, HP-3")

WIN = 8
K_MAX = max(WP_STEPS)                 # 20 steps = 2 s @ 10 Hz
IDX = [k - 1 for k in WP_STEPS]
DT = 0.1
N_BOOT = _ci.DEFAULT_N_BOOT

# The counterfactual commands. ``follow`` is the deploy-realistic control
# (`hierarchy.py` feeds `zeros(b)` for the deploy read), left/right are the
# alternatives. NAV_STRAIGHT exists in the vocabulary but `_NAV_TO_ROUTE` has no
# entry for it, so it is never fed by the trainer and is excluded here.
BRANCHES = {"follow": rl.NAV_FOLLOW, "left": rl.NAV_LEFT, "right": rl.NAV_RIGHT}
BRANCH_ORDER = ("follow", "left", "right")
# route class each command maps to under the v1 labeler — the ECHO target.
BRANCH_ROUTE = {k: rl.route_target(v) for k, v in BRANCHES.items()}
# Expected SIGN of the lateral deviation relative to the `follow` branch, in the
# repo's `_ego` convention (+y = LEFT, matching refb_labels and driving.frenet).
BRANCH_LAT_SIGN = {"left": +1.0, "right": -1.0}

# A divergence below this is not a behavioural difference, whatever its CI: at
# n~880 an interval can separate ~1 mm. Same guard, same reason, as
# `hierarchy.MIN_ADE_M`.
MIN_DIVERGENCE_M = 0.05
# Both signs correct by chance = 0.25; the per-window score is the FRACTION of
# the two signs that are correct, whose chance value is 0.5.
DIRECTION_CHANCE = 0.5

DECISION_ESTIMATORS = _drv.DECISION_ESTIMATORS
DEPRECATED_ESTIMATOR = _drv.DEPRECATED_ESTIMATOR
ESTIMATOR_NOTE = _drv.ESTIMATOR_NOTE


# Names other arms use for the same two levels. The probe drives the flagship
# 4-brain API (``strategic_policy(states, nav) -> {"ctx", "route_logits"}`` /
# ``tactical_policy(states, ctx) -> {"waypoints", "intent", ...}``) and cannot
# call these without an adapter — but it MUST NOT report their absence.
_ALIAS_HINTS = ("strateg", "tactic", "route", "maneuv", "nav", "intent", "ctx",
                "hier", "graft", "plan")


def _skip_report(model):
    """A SKIP, with the evidence for WHY — never a bare "no strategic level".

    ⚠️ MEASURED 2026-07-26, and it is exactly the CLAUDE.md rule-2 failure
    (*"absence found at ONE location is not absence"*): ``refc-base-30k``
    returns None for ``strategic_policy`` **and has a strategic level** —
    ``RefCModel`` names it ``model.strategic`` (GRU + proj) and carries
    ``route_head``, ``maneuver_head`` and ``decoder.ctx_to_cond`` /
    ``decoder.maneuver_to_anchor``. Reporting that arm as "no strategic level"
    would have manufactured a false structural claim about the very
    architecture this program is comparing itself to.

    So the skip enumerates what it DID find, and says plainly that the missing
    thing is an **adapter**, not a brain."""
    found = []
    try:
        names = [n for n, _ in model.named_modules() if n]
        found = [n for n in names
                 if any(h in n.lower() for h in _ALIAS_HINTS)][:24]
    except Exception:                                    # not an nn.Module
        found = []
    return {
        "block": BLOCK, "version": VERSION, "spec": SPEC,
        "skipped": ("this arm does not expose the flagship 4-brain API "
                    "(strategic_policy + tactical_policy) that HP-3 drives. "
                    "A SKIP IS NOT A PASS, and it is NOT a claim that the arm "
                    "has no strategic level -- see "
                    "`hierarchy_like_modules_found`."),
        "has_strategic_policy": getattr(model, "strategic_policy", None)
        is not None,
        "has_tactical_policy": getattr(model, "tactical_policy", None)
        is not None,
        "model_class": type(model).__name__,
        "hierarchy_like_modules_found": found,
        "strategic_level_absent": not found,
        "_read": ("`strategic_level_absent` is the ONLY field that may be read "
                  "as a structural absence, and it is False whenever any "
                  "hierarchy-shaped module was found under another name. In "
                  "that case HP-3 is UNMEASURED on this arm and needs a "
                  "per-arch adapter; do not record a 0."),
    }


def _wp_tensor(head_out):
    """``{step: [b,2]}`` -> ``[b, 4, 2]`` at WP_STEPS, in WP_STEPS order."""
    return torch.stack([head_out["waypoints"][k] for k in WP_STEPS], dim=1)


def _cos(a, b):
    return torch.nn.functional.cosine_similarity(a, b, dim=-1)


def _lat_endpoint(wp):
    """Signed cross-track of the 2 s endpoint, ego frame (+ = LEFT).

    The ego frame IS the reference frame here: all branches share one encode and
    one observed pose, so their endpoints are directly comparable on axis1
    without any path projection. That is the ``lateral.decompose(mode="ego")``
    convention (axis0 along, axis1 cross)."""
    return wp[:, -1, _lat.CROSS_AXIS]


# ========================================================================== #
# the probe                                                                    #
# ========================================================================== #
@torch.no_grad()
def run(model, episodes, device, step_readout=None, speed_input=False,
        yaw_input=False, dyn_input=False, max_eps=40, stride=8, batch=16,
        n_boot=N_BOOT, seed=0, grounded=False):
    """Encode each window ONCE, branch the strategic command, compare trajectories.

    ``grounded=True`` additionally threads each branch's intent through the
    20-step grounded rollout — the OUT-OF-REGIME diagnostic; it needs
    ``step_readout`` and costs one rollout per branch. Default OFF: the
    in-regime tactical surface is the primary and is ~3x cheaper.

    Returns the assembled block; ``{"skipped": ...}`` for an arm without both a
    trained strategic and tactical policy (a SKIP IS NOT A PASS)."""
    strat = getattr(model, "strategic_policy", None)
    tac = getattr(model, "tactical_policy", None)
    if strat is None or tac is None:
        return _skip_report(model)
    if grounded and step_readout is None:
        raise ValueError("grounded=True needs a step_readout")
    model.eval()
    # PC2 (2026-07-26): HP-3 is a claim about the STRATEGIC command reaching the
    # trajectory, so the probe must prove its own scored pass traversed the
    # seams it is reporting on. `operative_intent` is required only in
    # --grounded mode, which is the only mode that threads an intent into the
    # operative predictor.
    from taniteval import hierarchy_guard as _hg
    _need = (("strategic", "tactical", "operative_intent") if grounded
             else ("strategic", "tactical"))
    _trace = _hg.HierarchyTrace(model)
    _trace.__enter__()
    rec = {k: [] for k in ("eid", "nav_true", "nav_valid")}

    for ep in episodes[:max_eps]:
        fr = ep.feats
        T = fr.shape[0]
        for i0 in range(0, max(0, T - WIN - K_MAX), stride * batch):
            ch = list(range(i0, min(i0 + stride * batch, T - WIN - K_MAX),
                            stride))
            if not ch:
                continue
            b = len(ch)
            last = torch.tensor([t + WIN - 1 for t in ch])
            fw = torch.stack([torch.as_tensor(fr[t:t + WIN]) for t in ch]
                             ).to(device).float()
            if fr.dtype == torch.uint8:
                fw = fw.div_(255.0)
            states = model.encode_window(fw)                  # ONE encode
            aw = torch.stack([ep.actions[t:t + WIN] for t in ch]).to(device)
            fa = torch.stack([ep.actions[t + WIN:t + WIN + K_MAX] for t in ch]
                             ).to(device)
            ego = _hier._ego_channels(ep, last, speed_input, yaw_input,
                                      dyn_input, device)
            if ego is not None:
                aw = torch.cat([aw, ego[:, None].expand(-1, aw.shape[1], -1)], -1)
                fa = torch.cat([fa, ego[:, None].expand(-1, fa.shape[1], -1)], -1)

            br = {}
            for name, cmd in BRANCHES.items():
                nav = torch.full((b,), int(cmd), dtype=torch.long, device=device)
                s = strat(states, nav)
                t_out = tac(states, s["ctx"])
                wp = _wp_tensor(t_out)
                d = {"ctx": s["ctx"], "route_pred": s["route_logits"].argmax(-1),
                     "intent": t_out["intent"], "wp": wp,
                     "man": t_out["maneuver_logits"].argmax(-1),
                     "lat": _lat_endpoint(wp)}
                if grounded:
                    g_wp, _ = _hier._rollout_intent(
                        model.predictor, states, aw, fa, step_readout, K_MAX,
                        t_out["intent"])
                    d["grounded_wp"] = g_wp[:, IDX]
                    d["grounded_lat"] = _lat_endpoint(g_wp[:, IDX])
                br[name] = d

            for t in ch:
                cmd, valid = rl.nav_command(ep.poses, t + WIN - 1)
                rec["nav_true"].append(int(cmd))
                rec["nav_valid"].append(bool(valid))
            rec["eid"] += [ep.episode_id] * b
            _accumulate(rec, br, grounded)

    _trace.__exit__()
    if not rec["eid"]:
        return {"block": BLOCK, "version": VERSION,
                "skipped": "no eligible windows (episodes too short)"}
    pc2 = _hg.assert_hierarchy_traversed(
        _trace, block=BLOCK, claim="HP-3 counterfactual route swap",
        require=_need)
    out = _assemble(rec, n_boot=n_boot, seed=seed, grounded=grounded,
                    intent_free_scored_rollout=True)
    out["pc2"] = pc2
    return out


def _accumulate(rec, br, grounded):
    """Per-window counterfactual comparisons for one batch."""
    def push(k, v):
        rec.setdefault(k, []).extend(
            v.detach().float().cpu().tolist() if torch.is_tensor(v) else list(v))

    for a, b in (("left", "right"), ("left", "follow"), ("right", "follow")):
        wa, wb = br[a]["wp"], br[b]["wp"]
        push(f"wp_div_{a}_vs_{b}", torch.linalg.norm(wa - wb, dim=-1).mean(1))
        push(f"wp_div_final_{a}_vs_{b}",
             torch.linalg.norm(wa[:, -1] - wb[:, -1], dim=-1))
        # THE HP-3 CHANNEL: the lateral half of the divergence (M6)
        push(f"lat_div_{a}_vs_{b}", (br[a]["lat"] - br[b]["lat"]).abs())
        push(f"ctx_cos_{a}_vs_{b}", _cos(br[a]["ctx"], br[b]["ctx"]))
        push(f"intent_cos_{a}_vs_{b}", _cos(br[a]["intent"], br[b]["intent"]))
        push(f"man_changed_{a}_vs_{b}", (br[a]["man"] != br[b]["man"]).float())
        if grounded:
            ga, gb = br[a]["grounded_wp"], br[b]["grounded_wp"]
            push(f"grounded_div_{a}_vs_{b}",
                 torch.linalg.norm(ga - gb, dim=-1).mean(1))
            push(f"grounded_lat_div_{a}_vs_{b}",
                 (br[a]["grounded_lat"] - br[b]["grounded_lat"]).abs())

    # signed lateral response, and whether it points the commanded way
    correct = []
    for name, sign in BRANCH_LAT_SIGN.items():
        delta = br[name]["lat"] - br["follow"]["lat"]
        push(f"lat_signed_delta_{name}", delta)
        correct.append((torch.sign(delta) == sign).float())
    push("direction_score", torch.stack(correct).mean(0))     # in {0, .5, 1}
    push("direction_both_correct",
         (torch.stack(correct).sum(0) == len(correct)).float())
    for name in BRANCH_ORDER:
        push(f"route_pred_{name}", br[name]["route_pred"])


# ========================================================================== #
# assembly                                                                     #
# ========================================================================== #
def _assemble(rec, n_boot=N_BOOT, seed=0, grounded=False,
              intent_free_scored_rollout=True):
    A = {k: np.asarray(v) for k, v in rec.items()}
    eids = [str(x) for x in rec["eid"]]
    B = _hier._Boot(eids, n_boot=n_boot, seed=seed)
    valid = np.asarray(rec["nav_valid"], dtype=bool)

    def I(key, mask=None, reduce="mean"):                     # noqa: E743
        return _hier._interval(B, A[key], mask, reduce)

    pairs = (("left", "right"), ("left", "follow"), ("right", "follow"))
    div = {}
    for a, b in pairs:
        tag = f"{a}_vs_{b}"
        div[tag] = {
            "wp_l2_mean_m": I(f"wp_div_{tag}"),
            "wp_l2_final_m": I(f"wp_div_final_{tag}"),
            # the headline: route choice is a LATERAL phenomenon
            "cross_track_2s_m": I(f"lat_div_{tag}"),
            "cross_track_2s_p90_m": I(f"lat_div_{tag}", reduce="p90"),
            "cross_track_tail": _lat.tail_stats(A[f"lat_div_{tag}"]),
            "ctx_cosine": I(f"ctx_cos_{tag}"),
            "intent_cosine": I(f"intent_cos_{tag}"),
            "maneuver_changed_rate": I(f"man_changed_{tag}"),
        }
        if grounded:
            div[tag]["OUT_OF_REGIME_grounded_l2_m"] = I(f"grounded_div_{tag}")
            div[tag]["OUT_OF_REGIME_grounded_cross_track_m"] = I(
                f"grounded_lat_div_{tag}")

    lr = div["left_vs_right"]
    direction = {
        "score": I("direction_score"),
        "both_correct_rate": I("direction_both_correct"),
        "chance": DIRECTION_CHANCE,
        "separated_above_chance": bool(
            I("direction_score").get("lo") is not None
            and I("direction_score")["lo"] > DIRECTION_CHANCE),
        "signed_lateral_delta_left_m": I("lat_signed_delta_left"),
        "signed_lateral_delta_right_m": I("lat_signed_delta_right"),
        "_read": ("per-window score = the fraction of the two commands whose "
                  "lateral response points the commanded way (left => +y, "
                  "right => -y, ego convention). Chance is 0.5; the verdict "
                  "needs the CI to clear it, not the point estimate."),
    }
    echo = _route_echo(A, valid, B)

    out = {
        "block": BLOCK, "version": VERSION, "spec": SPEC,
        "n_windows": int(len(eids)), "n_episodes": B.full.n_episodes
        if B.full is not None else len(set(eids)),
        "branches": {k: int(v) for k, v in BRANCHES.items()},
        "branch_route_targets": {k: int(v) for k, v in BRANCH_ROUTE.items()},
        "surface": ("tactical head 2 s waypoints (IN-REGIME)"
                    + (" + grounded rollout with the branch intent "
                       "(OUT-OF-REGIME diagnostic)" if grounded else "")),
        "estimator": {
            "interval": "episode_cluster_bootstrap",
            "delta": "paired_episode_cluster_bootstrap",
            "n_boot": int(n_boot), "seed": int(seed),
            "resampling_unit": "val episode",
            "deprecated_and_refused": DEPRECATED_ESTIMATOR,
            "estimator_note": ESTIMATOR_NOTE},
        "thresholds": {"min_divergence_m": MIN_DIVERGENCE_M,
                       "min_divergence_mark": "PROPOSED",
                       "direction_chance": DIRECTION_CHANCE},
        "divergence": div,
        "direction": direction,
        "route_head_echo": echo,
        "route_can_reach_scored_trajectory": not intent_free_scored_rollout,
        "_pc2_note": (
            "the DEPLOYED scored rollout (metric_dynamics.rollout_decode) takes "
            "no intent/ctx/nav, so on the leaderboard surface a route command "
            "structurally CANNOT change the trajectory whatever this probe "
            "reports. A 0 here is therefore a statement about the tactical "
            "head, and PC2 is a separate, prior defect."),
    }
    out["verdict"] = _verdict(out)
    _drv.assert_no_deprecated_estimator(out, _path=BLOCK)
    return out


def _route_echo(A, valid, B):
    """Does the ROUTE HEAD follow the command? (It does — by construction.)

    Kept as a positive control, and labelled as one. Under the v1 labeler the
    route target IS ``_NAV_TO_ROUTE[nav_cmd]``, so a head that copies its own
    conditioning embedding scores 1.0 here while changing nothing about the
    trajectory. **A high echo rate next to a zero trajectory divergence is the
    exact signature of the defect HPP-1 fixes** — the command reaches the
    logits and stops there."""
    rate = np.mean([
        float((A[f"route_pred_{n}"] == BRANCH_ROUTE[n]).mean())
        for n in BRANCH_ORDER])
    per = {n: round(float((A[f"route_pred_{n}"] == BRANCH_ROUTE[n]).mean()), 4)
           for n in BRANCH_ORDER}
    follows = np.mean(
        [(A[f"route_pred_{n}"] == BRANCH_ROUTE[n]).astype(float)
         for n in BRANCH_ORDER], axis=0)
    return {
        "route_logit_follows_command_rate": round(float(rate), 4),
        "by_branch": per,
        "interval": _hier._interval(B, follows),
        "n_nav_valid": int(valid.sum()),
        "_read": ("route target == _NAV_TO_ROUTE[nav_cmd] under the v1 "
                  "labeler, so this is ~1.0 BY CONSTRUCTION and is NOT "
                  "evidence of route understanding. High echo + zero "
                  "trajectory divergence == the command reaches the logits "
                  "and stops there."),
    }


def _verdict(o):
    lr = o["divergence"]["left_vs_right"]
    xt, l2 = lr["cross_track_2s_m"], lr["wp_l2_mean_m"]
    sep = bool(xt.get("lo") is not None and xt["lo"] > 0
               and xt["mean"] >= MIN_DIVERGENCE_M)
    dirok = bool(o["direction"]["separated_above_chance"])
    o["HP3_route_conditional"] = bool(sep and dirok)
    o["HP3_divergence_separated"] = sep
    o["HP3_direction_correct"] = dirok
    if sep and dirok:
        head = ("HP-3 PASSES — the strategic command causally changes the "
                "trajectory, in the commanded direction")
    elif sep:
        head = ("HP-3 PARTIAL — the command changes the trajectory but NOT "
                "reliably in the commanded direction (divergence without "
                "correctness is not route-following)")
    else:
        head = ("HP-3 FAILS — trajectories are effectively IDENTICAL under "
                "left vs right. This is a PC1 regression (still a command "
                "echo), NOT evidence that hierarchy does not help")
    return (f"{head} · left-vs-right cross-track@2s "
            f"{xt['mean']} [{xt['lo']}, {xt['hi']}] m "
            f"(L2 {l2['mean']} m, floor {MIN_DIVERGENCE_M}) · direction score "
            f"{o['direction']['score']['mean']} vs chance "
            f"{DIRECTION_CHANCE} · route-logit echo "
            f"{o['route_head_echo']['route_logit_follows_command_rate']} · "
            f"n={o['n_windows']}/{o['n_episodes']} eps")


# ========================================================================== #
# invocation                                                                   #
# ========================================================================== #
INVOCATION = """\
HP-3 counterfactual route swap — run when a pod frees (needs 1 GPU, ~2 min).
NOT run against a real checkpoint by the authoring agent: pod1/2/3 were training
and the eval pod was mid-transfer.

    cd /root/taniteval && PYTHONPATH=/root/TanitAD/stack \\
      python -m taniteval.strategic_probes \\
        --model flagship-30k --episodes 40 --stride 8 \\
        --out results/hp3_flagship-30k.json

  --grounded    additionally thread each branch's intent through the 20-step
                grounded rollout (OUT-OF-REGIME diagnostic, ~3x the cost)

READ IT LIKE THIS
  * `HP3_route_conditional: false` on any arm trained before HPP-1 is the
    EXPECTED, pre-registered outcome — `route_target = _NAV_TO_ROUTE[nav_cmd]`
    makes `route_skill = 0.0` by construction. Record it as the pre-fix
    baseline; do NOT report it as a hierarchy result.
  * The number that matters after HPP-1 lands is
    `divergence.left_vs_right.cross_track_2s_m` clearing 0 (CI) and
    MIN_DIVERGENCE_M, together with `direction.separated_above_chance`.
  * `route_head_echo` near 1.0 next to a zero divergence IS the defect, stated
    in one line. Both must move for the fix to have worked.
  * Compare arms with `paired_hp3_delta` on the SAME windows — never two
    single-arm intervals in quadrature.
"""


def paired_hp3_delta(a_lat_div, b_lat_div, eid, n_boot=N_BOOT, seed=0):
    """PAIRED Δ left-vs-right cross-track divergence between two arms.

    The admissible arm-vs-arm form for HP-3: both arms score the same windows,
    so the comparison is paired. Oriented ``a − b``, so POSITIVE means arm ``a``
    is MORE route-conditional — the opposite orientation to the error metrics,
    because here **more divergence is better**."""
    d = _ci.paired_episode_cluster_bootstrap(
        np.asarray(a_lat_div, dtype=np.float64),
        np.asarray(b_lat_div, dtype=np.float64), [str(x) for x in eid],
        n_boot=n_boot, seed=seed)
    d["_orientation"] = ("a - b on left-vs-right cross-track divergence; "
                         "POSITIVE = arm `a` is MORE route-conditional. Note "
                         "the inverted sense: for HP-3, MORE divergence is "
                         "BETTER.")
    return d


def main():
    import argparse
    import json
    from pathlib import Path
    sys.path.insert(0, "/root/taniteval")
    from taniteval import data, loaders
    from taniteval.registry import MODELS
    ap = argparse.ArgumentParser("taniteval.strategic_probes")
    ap.add_argument("--model", default="flagship-30k")
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--grounded", action="store_true")
    ap.add_argument("--val-dir", default="/root/valdata/physicalai-val-0c5f7dac3b11")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    e = [m for m in MODELS if m["key"] == a.model][0]
    L = loaders.load(e, "cuda")
    files = data.list_val_episodes(a.val_dir, a.episodes)
    eps = (data.load_frames(files) if L["feed"] == "frames"
           else data.load_features(files, L["feed"], "cuda"))
    res = run(L["model"], eps, "cuda", step_readout=L["step_readout"],
              speed_input=bool(e.get("speed_input")),
              yaw_input=bool(e.get("yaw_input")),
              dyn_input=bool(e.get("dyn_input")),
              max_eps=a.episodes, stride=a.stride, n_boot=a.n_boot,
              grounded=a.grounded)
    print(res.get("verdict") or res.get("skipped"))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=2, default=str))
        print(f"[hp3] wrote {a.out}")


if __name__ == "__main__":
    main()
