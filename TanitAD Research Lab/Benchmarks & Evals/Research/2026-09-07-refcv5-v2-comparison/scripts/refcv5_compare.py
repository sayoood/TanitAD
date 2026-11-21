#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""refcv5_compare.py — the ONE-COMMAND refcv5-v2 vs refcv4b comparison.

⭐ WHAT THIS IS FOR. ``taniteval/tools/refcv3_arm.py`` already rolls one
checkpoint and emits the four families per arm. It does NOT emit (a) a
cross-checkpoint paired margin, (b) the STRATEGIC family — which its own
``four_families`` reports UNAVAILABLE even though ``route_pred_*``/``route_label``
sit in the decisions dump it just wrote, (c) the straight-line curvature floor
beside the curvature it prints, (d) a machine-written VERDICT against a bar that
was fixed BEFORE the checkpoint existed, or (e) the statement of WHICH variance
question each separated interval actually answered. This does all five, so that
none of them is a sentence a human adds to a report afterwards.

⛔⛔ THE PRE-REGISTERED BAR (written here, 2026-09-07, while refcv5-v2 was at
step ~2,600 of 40,284 and could not be scored).

    refcv4b @40,284 only TIES the do-nothing baselines:
        os - ha        -0.0021 [-0.0178, +0.0154]   NOT separated
        os - ha0_ext   +0.0101 [-0.0050, +0.0273]   NOT separated
    ⇒ refcv5-v2 PASSES only if ``os - ha0_ext`` on ADE is NEGATIVE **and
      SEPARATED** (the whole interval below zero). Beating refcv4b is NOT the
      bar; refcv4b did not clear it either. ``BAR_PRIMARY`` below is that
      sentence in code.

⛔ ESTIMATOR. ``taniteval.ci.paired_episode_cluster_bootstrap`` only.
``overlapping_holdout_se`` is FORBIDDEN — it biases the POINT ESTIMATE
bidirectionally, up to a sign flip, and is not merely an interval choice.

⛔ TIER. Every number is T1 = **self-action open loop**. A planner feeding its
own predictor is STILL open loop (PI ruling 2026-09-02). ⛔ None of this is
"driving performance"; there is no simulator and no vehicle in this path.

⛔⛔ A SEPARATED CI FROM A ONE-SEED ARM IS NECESSARY, NOT SUFFICIENT. A pure
replicate produced "separated" differences on 6 of 42 cells — a 14.3 %
false-positive rate — and a one-seed lateral "direction" was refuted by its own
second seed with the deranged control beating the real arm. So every separated
cell here carries ``variance_questions``: the episode draw is ANSWERED, the
TRAINING-RUN draw is NOT MEASURED at one seed, and the INFERENCE-RUN draw is
answered only for a deterministic sampler. A cell whose separation is fragile is
stamped ``NOT YET MEASURED`` and must never be reported as "a direction".

⚠️ ORACLE-NAV. Both arms consume the v7.2 nav command, and all 4,719 v7.2 nav
records are ``ego-future`` — derived from the ego's own future path. Per the
binding PI ruling of 2026-09-04 that is a first-class ROUTE INPUT (what a map
router supplies), not a leak; but it is noiseless and perfectly timed where a
real router is coarse, so every panel is stamped and no arm's nav may be read as
a production command.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

from taniteval.ci import (episode_cluster_bootstrap,          # noqa: E402
                          paired_episode_cluster_bootstrap)

# --------------------------------------------------------------------------- #
# THE PRE-REGISTERED BAR — fixed 2026-09-07, BEFORE the checkpoint existed.
# --------------------------------------------------------------------------- #
BAR_PRIMARY = {
    "id": "BAR-REFCV5V2-1",
    "registered_utc": "2026-09-07",
    "registered_before": "refcv5-v2 landing (it was at step ~2,600 of 40,284)",
    "statement": ("refcv5-v2 must BEAT the echo control `ha0_ext` on ADE, "
                  "SEPARATED: paired `os - ha0_ext` delta < 0 AND the whole "
                  "episode-cluster interval below zero."),
    "why": ("refcv4b @40,284 only TIED the do-nothing baselines "
            "(os-ha -0.0021 [-0.0178,+0.0154]; os-ha0_ext +0.0101 "
            "[-0.0050,+0.0273], neither separated). An arm that ties a plan "
            "holding its own t0 action and curvature has not learned to drive "
            "either, whatever it does against another model."),
    "arm": "os", "control": "ha0_ext", "metric": "ade_m",
    "predicate": "delta < 0 and hi < 0",
    #: ⛔ `stack/tanitad/eval/echo_gate.py::echo_gate` requires BOTH the CI to
    #: exclude zero AND a RELATIVE point margin committed in advance — "a
    #: separated CI on a 0.3 % margin is a real but useless difference, and the
    #: prereg committed to margins for that reason" (its own `_reads`). 0.10 is
    #: echo_gate's own docstring example, taken rather than invented, and it is
    #: ~28x the ~0.001 m cross-hardware roll noise MEASURED between the A40
    #: landing (os 0.2975) and the Thor re-roll (0.2965).
    "relative_margin_required": 0.10,
}
BAR_SECONDARY = {
    "id": "BAR-REFCV5V2-2",
    "statement": ("refcv5-v2 must BEAT the hold-action control `ha` on ADE, "
                  "SEPARATED. `ha` and `ha0_ext` are the bar TOGETHER "
                  "(stack/tanitad/eval/echo_gate.py)."),
    "arm": "os", "control": "ha", "metric": "ade_m",
    "predicate": "delta < 0 and hi < 0",
}
BAR_TERTIARY = {
    "id": "BAR-REFCV5V2-3",
    "statement": ("ARM DELTA (informational, NOT the bar): refcv5-v2 `os` vs "
                  "refcv4b `os` on the same 4,823 windows. refcv4b beat refcv3 "
                  "by -0.1444 m and still failed BAR-1, so a win here is not a "
                  "pass."),
    "arm": "os", "control": "os@baseline", "metric": "ade_m",
    "predicate": "informational",
}

#: MEASURED false-positive rate of "separated" on a PURE REPLICATE, one seed:
#: 6 separated cells of 42. Any panel whose separated fraction sits at or under
#: this is, as a panel, indistinguishable from a replicate.
REPLICATE_FP_RATE = 6.0 / 42.0

#: A separated cell is FRAGILE when its interval reaches back to within this
#: fraction of the point estimate. Pre-registered here, not chosen after seeing
#: the data.
FRAGILE_FRAC = 0.25

TIER_NOTE = ("T1 = SELF-ACTION OPEN LOOP. One forward pass at t0; the path is "
             "the model's own selection. A planner feeding its own predictor is "
             "STILL open loop (PI ruling 2026-09-02). ⛔ NOT driving "
             "performance — no simulator, no vehicle, nothing closed.")

NAV_STAMP = ("⚠️ ORACLE-NAV: both arms consume the v7.2 nav command and all "
             "4,719 v7.2 nav records are `ego-future` (derived from the ego's "
             "own future path). FAIR — refcv4b shares the identical input — and "
             "a first-class ROUTE INPUT under the PI ruling of 2026-09-04, but "
             "noiseless and perfectly timed where a real router is coarse. ⛔ "
             "NEITHER ARM's nav may be read as a production command.")

TRIVIAL_ARMS = ("ha", "ha0", "ha0_ext")


def _p(*a):
    print(*a, flush=True)


# --------------------------------------------------------------------------- #
# dumps
# --------------------------------------------------------------------------- #
def load_dump(d: str, arms=("os", "ha", "ha0", "ha0_ext")) -> dict:
    """Per-window ADE for each arm, plus eid/ws — the pairing keys.

    ⛔ A dump that reads ZERO windows is a MOUNT FLAP, not an empty corpus.
    Refuse; a control reading 0 must never be believed.
    """
    fs = sorted(glob.glob(os.path.join(d, "ep*.npz")))
    if not fs:
        raise SystemExit("[compare] ⛔ no ep*.npz under %r. If this path is on "
                         "a network mount, RETRY — a control reading 0 means "
                         "the mount flapped, it does not mean 'no data'." % d)
    ade = {a: [] for a in arms}
    per_slot = {a: [] for a in arms}          # [N, K] — echo_gate scores PER SLOT
    paths = {a: [] for a in arms}             # [N, K, 2] — the geometry itself
    present, eid, ws, v0 = None, [], [], []
    for f in fs:
        z = np.load(f, allow_pickle=True)
        if present is None:
            present = [a for a in arms if a in z.files]
        g = np.asarray(z["g"], dtype=np.float64)
        for a in present:
            p = np.asarray(z[a], dtype=np.float64)
            d = np.linalg.norm(p - g, axis=-1)        # [N, K]
            per_slot[a].append(d)
            paths[a].append(p)
            ade[a].append(d.mean(axis=1))
        eid.append(np.full(g.shape[0], int(np.asarray(z["eid"]).ravel()[0])))
        ws.append(np.asarray(z["ws"]).ravel())
        v0.append(np.asarray(z["v0"]).ravel() if "v0" in z.files
                  else np.full(g.shape[0], np.nan))
    out = {"ade": {a: np.concatenate(ade[a]) for a in present},
           "per_slot": {a: np.concatenate(per_slot[a], axis=0) for a in present},
           "paths": {a: np.concatenate(paths[a], axis=0) for a in present},
           "eid": np.concatenate(eid), "ws": np.concatenate(ws),
           "v0": np.concatenate(v0), "dir": d,
           "arms_present": present, "arms_missing": [a for a in arms if a not in present]}
    n = out["eid"].size
    if n == 0:
        raise SystemExit("[compare] ⛔ 0 windows read from %r — RETRY." % d)
    out["n_windows"], out["n_episodes"] = int(n), len(set(out["eid"].tolist()))
    return out


def _read_manifest(dump_dir: str) -> dict | None:
    """The dump's own manifest, or None. ⛔ A missing manifest is reported as
    None and resolved elsewhere from another witness — it is never a licence to
    invent a bank size."""
    p = os.path.join(dump_dir, "manifest.json")
    if not os.path.exists(p):
        return None
    try:
        return json.load(open(p, encoding="utf-8"))
    except Exception:
        return None


def assert_same_grid(a: dict, b: dict, la: str, lb: str) -> dict:
    """⛔ Pairing is a FICTION unless the window grids are identical. Refuse
    rather than align — aligning is how a 'paired' margin becomes a comparison
    of different windows wearing the same name."""
    if a["eid"].shape != b["eid"].shape:
        raise SystemExit("[compare] ⛔ %s has %d windows, %s has %d — NOT the "
                         "same grid, refusing to pair."
                         % (la, a["eid"].size, lb, b["eid"].size))
    if not np.array_equal(a["eid"], b["eid"]):
        raise SystemExit("[compare] ⛔ episode ids differ between %s and %s." % (la, lb))
    if not np.array_equal(a["ws"], b["ws"]):
        raise SystemExit("[compare] ⛔ window origins differ between %s and %s." % (la, lb))
    return {"status": "PAIRING OK", "n_windows": a["n_windows"],
            "n_episodes": a["n_episodes"], "checked": ["eid", "ws", "shape"]}


# --------------------------------------------------------------------------- #
# variance questions — what a separated interval DID and DID NOT answer
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# ⛔ READING THE RUN'S OWN config.json — NESTED, and REFUSING rather than
# defaulting. MEASURED 2026-09-07 on the LIVE refcv5-v2 config (35 top-level
# keys, 206 leaves): every fact this harness wants about the two new levers is
# NESTED, and the bank size is not under the name `n_anchors` at all.
#
#   fact                      where it ACTUALLY is                 refcv4b
#   sampler_ranks_the_fan     /selection/sampler_ranks_the_fan     absent
#   sel_refined               /selection/sel_refined               absent
#   sel_score_emitted         /selection/sel_score_emitted         absent
#   tac_goal_tok_head         /seams/tac_goal_tok_head/built       /seams is null
#   anchors v0_conditioned    /anchors/v0_conditioned              same
#   anchor control units      /anchors/control_units               same
#   ⛔ n_anchors              NOWHERE — it is /anchors/shape[0]     same
#
# ⭐ EVERY fact is resolved from its nested path AND corroborated against `argv`,
# which is an independently authored witness (what the operator asked for, vs
# what the trainer recorded). Agreement between a value and a re-derivation of
# ITSELF would measure determinism, not correctness.
# --------------------------------------------------------------------------- #
def _pos_int(v):
    """A positive whole number, or None. ⛔ Never coerces a string."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    if not (v == v) or v <= 0 or float(v) != int(v):
        return None
    return int(v)


def _first(x):
    return x[0] if isinstance(x, (list, tuple)) and x else None


def _argv_of(cfg):
    a = (cfg or {}).get("argv") or []
    return a if isinstance(a, list) else str(a).split()


def _argv_flag(cfg, flag):
    return flag in _argv_of(cfg)


def _argv_value(cfg, flag):
    a = _argv_of(cfg)
    if flag not in a:
        return None
    i = a.index(flag)
    return a[i + 1] if i + 1 < len(a) else None


#: ``name -> (nested path, argv flag)``. The nested path is the DECLARATION the
#: trainer wrote; the argv flag is what the operator asked for. Both are read;
#: a CONTRADICTION is a refusal, an agreed absence is a corroborated ``False``.
NESTED_LEVERS = (
    ("sel_refined",          ("selection", "sel_refined"),        "--sel-refined"),
    ("sel_score_emitted",    ("selection", "sel_score_emitted"),  "--sel-score-emitted"),
    ("sampler_ranks_the_fan", ("selection", "sampler_ranks_the_fan"), None),
    ("tac_goal_tok_head",    ("seams", "tac_goal_tok_head", "built"), "--tac-goal-tok-head"),
    ("goal_str",             ("seams", "goal_str"),               "--goal-str"),
    ("anchor_v0_conditioned", ("anchors", "v0_conditioned"),      "--anchor-v0-conditioned"),
)

#: Facts that are VALUES, not flags: ``name -> (nested path, argv flag)``.
NESTED_VALUES = (
    ("anchor_control_units", ("anchors", "control_units"),   "--anchor-control-units"),
    ("sampler",              ("seams", "sampler"),           "--sampler"),
)


def _dig(d, path):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def config_facts(cfg: dict | None) -> dict:
    """Every lever this panel names, resolved from its NESTED path and
    corroborated by ``argv``. ⛔ A declaration that CONTRADICTS argv is a
    REFUSAL: the two describe the same run and cannot both be right."""
    if cfg is None:
        return {"_status": "NO CONFIG SUPPLIED — pass --new-config; the levers "
                           "below are NOT DECLARED, which is not the same as OFF",
                "levers": {}, "values": {}}
    out, conflicts = {}, []
    for name, path, flag in NESTED_LEVERS:
        decl = _dig(cfg, path)
        asked = _argv_flag(cfg, flag) if flag else None
        val = bool(decl) if decl is not None else (bool(asked) if asked is not None else None)
        if decl is not None and asked is not None and bool(decl) != bool(asked):
            conflicts.append("%s: /%s = %r but argv %s %s"
                             % (name, "/".join(path), decl, flag,
                                "present" if asked else "absent"))
        out[name] = {"value": val, "declared_at": "/" + "/".join(path),
                     "declared": decl, "argv_flag": flag, "argv_says": asked,
                     "source": ("nested declaration" if decl is not None else
                                "argv only (the config block is absent)"
                                if asked else "NOT DECLARED anywhere")}
    vals = {}
    for name, path, flag in NESTED_VALUES:
        decl = _dig(cfg, path)
        asked = _argv_value(cfg, flag) if flag else None
        if decl is not None and asked is not None and str(decl) != str(asked):
            conflicts.append("%s: /%s = %r but argv %s %r"
                             % (name, "/".join(path), decl, flag, asked))
        vals[name] = {"value": decl if decl is not None else asked,
                      "declared_at": "/" + "/".join(path),
                      "declared": decl, "argv_says": asked}
    if conflicts:
        raise SystemExit("[compare] ⛔ the run's config CONTRADICTS its own argv "
                         "on %d fact(s) — refusing rather than choosing one:\n  "
                         % len(conflicts) + "\n  ".join(conflicts))
    return {"_status": "OK", "levers": out, "values": vals,
            "_rule": ("every fact is read from its NESTED path and corroborated "
                      "against argv; a contradiction is a refusal and an agreed "
                      "absence is a corroborated False, never a default")}


#: ⭐ THE FIELDS THAT CAN WITNESS A BANK SIZE, in order of AUTHORITY. The
#: CHECKPOINT TENSOR is first because it is THE OBJECT THE MODEL USES: the
#: anchors are a persistent buffer, the deployed selection is an argmax over
#: exactly that tensor, and `anchor_acc`'s denominator IS its first dimension.
N_ANCHORS_WITNESS_FIELDS = (
    "checkpoint:model['core.decoder.anchors'].shape[0]",
    "dump/manifest.json:model.n_anchors",
    "config.json:anchors.shape[0]",
    "config.json:anchors.controls_shape[0]",
    "argv:--n-anchors",
)


def ckpt_n_anchors(ckpt_path: str | None):
    """The bank size READ OUT OF THE WEIGHTS, cheaply (mmap: 0.03 s on a 428 MB
    checkpoint, MEASURED). ⛔ Returns None only when the file or the key is not
    there; it never guesses, and a read failure is reported by the caller."""
    if not ckpt_path or not os.path.exists(ckpt_path):
        return None, ("no checkpoint given" if not ckpt_path
                      else "checkpoint not found: %s" % ckpt_path)
    try:
        import torch
    except Exception as ex:                                   # pragma: no cover
        return None, "torch unavailable: %s" % ex
    for kw in ({"mmap": True, "weights_only": True},
               {"mmap": True, "weights_only": False},
               {"weights_only": False}):
        try:
            ck = torch.load(ckpt_path, map_location="cpu", **kw)
            t = (ck.get("model") or {}).get("core.decoder.anchors")
            if getattr(t, "ndim", 0) != 3:
                return None, "no core.decoder.anchors [N,S,2] in %s" % ckpt_path
            return int(tuple(t.shape)[0]), "torch.load(%s)" % ",".join(sorted(kw))
        except Exception:
            continue
    return None, "could not read %s with any torch.load mode" % ckpt_path


def resolve_n_anchors(manifest: dict | None, cfg: dict | None,
                      ckpt_path: str | None = None) -> tuple:
    """``(n_anchors, provenance)`` — or a REFUSAL THAT NAMES THE FIELD.

    ⛔ This exists because a hardcoded ``1/128`` was written into banked sidecars
    while the live bank was 117. ``anchor_chance(n)`` made the DERIVATION single;
    this makes its INPUT single. MEASURED 2026-09-07: the live refcv5-v2
    ``config.json`` has NO key named ``n_anchors`` in any of its 206 leaves — the
    bank size lives at ``anchors.shape[0]``, at ``anchors.controls_shape[0]`` and
    in ``argv``. ⇒ a resolver that looked only for the NAME would have found
    nothing and had to either fail or default; this one finds three witnesses.

    ⛔ There is NO default. Unresolvable is a refusal naming every field probed;
    two witnesses that disagree is a refusal naming both.
    """
    n_ck, ck_note = ckpt_n_anchors(ckpt_path)
    anch = (cfg or {}).get("anchors")
    anch = anch if isinstance(anch, dict) else {}
    argv_n = _argv_value(cfg, "--n-anchors")
    try:
        argv_n = int(argv_n) if argv_n is not None else None
    except (TypeError, ValueError):
        argv_n = None
    cand = {
        "checkpoint:model['core.decoder.anchors'].shape[0]": n_ck,
        "dump/manifest.json:model.n_anchors":
            _dig(manifest or {}, ("model", "n_anchors")),
        "config.json:anchors.shape[0]": _first(anch.get("shape")),
        "config.json:anchors.controls_shape[0]": _first(anch.get("controls_shape")),
        "argv:--n-anchors": argv_n,
    }
    seen, absent = {}, {}
    for k in N_ANCHORS_WITNESS_FIELDS:
        n = _pos_int(cand.get(k))
        if n is None:
            absent[k] = cand.get(k)
        else:
            seen[k] = n
    if not seen:
        raise SystemExit(
            "[compare] ⛔ n_anchors IS UNRESOLVABLE — every witness is absent: "
            + ", ".join(N_ANCHORS_WITNESS_FIELDS)
            + ". The anchor chance level is 1/n_anchors and there is NO default: "
              "a hardcoded 1/128 was banked into several sidecars while the live "
              "bank was 117, which is the defect this refusal exists to prevent. "
              "Pass --new-ckpt, or supply a config/manifest that declares the "
              "bank. (checkpoint read: %s)" % ck_note)
    vals = sorted(set(seen.values()))
    if len(vals) > 1:
        raise SystemExit(
            "[compare] ⛔ n_anchors WITNESSES DISAGREE: "
            + "; ".join("%s = %d" % (k, v) for k, v in seen.items())
            + ". Refusing rather than picking one — the weights, the config and "
              "the dump must describe ONE bank.")
    n = vals[0]
    return n, {"n_anchors": n, "chance": round(1.0 / n, 6),
               "primary": next(k for k in N_ANCHORS_WITNESS_FIELDS if k in seen),
               "witnesses_agreeing": seen, "witnesses_absent": sorted(absent),
               "checkpoint_read": ck_note,
               "_rule": ("chance = 1/n_anchors, DERIVED from this run's own bank. "
                         "⛔ Never a remembered constant; the tensor is "
                         "authoritative and every other field corroborates it.")}


def anchor_selection_block(res: dict, n_anchors: int, prov: dict) -> dict:
    """⭐ THE TACTICAL FAMILY'S GOAL/ANCHOR-SELECTION HALF, which this panel used
    to omit entirely. The binding rule is four families NEVER POOLED and a
    missing metric is a WORK ITEM, not an excuse — an `anchor_acc` printed with
    no chance level beside it is exactly the silent drop that rule forbids.

    ⛔ The chance level here is the RESOLVED one, not the one the arm JSON
    happens to carry: an arm that was analysed from a manifest with no bank size
    publishes ``chance: null`` and this panel would otherwise print a bare
    accuracy against no bar."""
    r = res.get("refcv3") or {}
    asel = ((r.get("tactical_declared") or {}).get("anchor_selection") or {})
    sp = r.get("selection_profile") or {}
    acc = asel.get("anchor_acc") or {}
    chance = round(1.0 / n_anchors, 6)
    mean = acc.get("mean")
    out = {"n_anchors": n_anchors, "n_anchors_provenance": prov,
           "chance": chance,
           "anchor_acc": {k: acc.get(k) for k in
                          ("mean", "lo", "hi", "n_windows", "n_episodes",
                           "estimator")},
           "lift_over_chance": (round(float(mean) / chance, 2)
                                if isinstance(mean, (int, float)) and chance else None),
           "deployed_selection_agrees_oracle":
               (asel.get("deployed_selection_agrees_oracle") or {}).get("mean"),
           "n_distinct_selected": asel.get("n_distinct_selected"),
           "selection_profile": {k: sp.get(k) for k in
                                 ("n_anchors", "modal_anchor", "modal_frac",
                                  "entropy_nats", "max_entropy_nats",
                                  "entropy_ratio", "degenerate", "status",
                                  "reason")},
           "arm_json_declared_chance": asel.get("chance"),
           "_estimator": "episode_cluster_bootstrap (taniteval/ci.py)"}
    # ⛔ the arm's own chance, when it published one, must EQUAL the resolved
    # one. A disagreement means two derivations of the same denominator drifted,
    # which is the entire defect class this row exists to close.
    ac = asel.get("chance")
    if isinstance(ac, (int, float)) and abs(float(ac) - chance) > 1e-9:
        raise SystemExit("[compare] ⛔ the arm JSON declares chance %r but the "
                         "resolved bank size %d gives %r — two derivations of "
                         "one denominator have drifted; refusing."
                         % (ac, n_anchors, chance))
    if ac is None:
        out["_note"] = ("⚠️ the arm JSON published NO chance (its dump manifest "
                        "declared no `model.n_anchors`), so the chance printed "
                        "here comes from %s. The arm's own selection_profile is "
                        "REFUSED and stays refused — this row does not repair "
                        "it, it only stops the panel from printing an accuracy "
                        "with no bar." % out["n_anchors_provenance"]["primary"])
    return out


def sampler_is_stochastic(cfg: dict | None) -> tuple[bool, str]:
    """refcv4b zeroes the decoder noise at eval (refc.py:1599) ⇒ deterministic.
    refcv5's WP-4 lever is ``--sampler ddim``, an anchored Gaussian in control
    space ⇒ STOCHASTIC AT INFERENCE, and the inference-seed question that is
    CLOSED BY CONSTRUCTION for refcv4b is OPEN for it."""
    argv = _argv_of(cfg)
    # ⛔ argv is what was ASKED FOR; `/seams/sampler` is what the trainer
    # RECORDED. Read both and refuse on a contradiction — reading only one is how
    # a run gets stamped "deterministic" because a flag moved.
    decl = _dig(cfg or {}, ("seams", "sampler"))
    if "--sampler" in argv and decl is not None and str(decl) != str(
            argv[argv.index("--sampler") + 1] if argv.index("--sampler") + 1 < len(argv)
            else "?"):
        raise SystemExit("[compare] ⛔ /seams/sampler = %r but argv says %r — the "
                         "config contradicts itself about the sampler; refusing."
                         % (decl, argv[argv.index("--sampler") + 1]))
    if decl is not None and "--sampler" not in argv:
        if str(decl) != "truncated":
            return True, ("/seams/sampler = %r: samples are drawn at inference, "
                          "so a re-roll of this same checkpoint can move every "
                          "number below" % decl)
    if "--sampler" in argv:
        i = argv.index("--sampler")
        s = argv[i + 1] if i + 1 < len(argv) else "?"
        if s != "truncated":
            return True, ("--sampler %s: samples are drawn at inference, so a "
                          "re-roll of this same checkpoint can move every "
                          "number below" % s)
    return False, ("no --sampler flag: the decoder's noise is zeroed when not "
                   "training (refc.py:1599), so inference is deterministic and "
                   "the inference-run question is CLOSED BY CONSTRUCTION")


def variance_block(stochastic: bool, why: str, n_boot: int) -> dict:
    return {
        "episode_draw": ("ANSWERED — paired episode-cluster bootstrap "
                         "(taniteval/ci.py), cluster = episode, n_boot %d, "
                         "seed 0. This is the ONLY question the interval "
                         "answers." % n_boot),
        "training_run": ("⛔ NOT MEASURED — one seed per arm. A pure replicate "
                         "produced 'separated' on 6/42 cells (14.3 %). "
                         "H-ESTIM-SEED-1 is OPEN; a second training seed is the "
                         "only thing that answers it."),
        "inference_run": (("⛔ NOT MEASURED — %s" % why) if stochastic
                          else ("CLOSED BY CONSTRUCTION — %s" % why)),
    }


def classify(cell: dict) -> dict:
    """Add the honesty fields to one paired-bootstrap result."""
    d, lo, hi = cell.get("delta"), cell.get("lo"), cell.get("hi")
    sep = bool(cell.get("separated"))
    out = dict(cell)
    if d is None or lo is None or hi is None:
        out["reading"] = "UNAVAILABLE"
        return out
    margin = min(abs(lo), abs(hi)) if sep else 0.0
    out["separation_margin"] = round(float(margin), 6)
    if not sep:
        out["reading"] = "NOT SEPARATED"
        out["fragile"] = False
    elif abs(d) > 0 and margin < FRAGILE_FRAC * abs(d):
        out["reading"] = ("⛔ NOT YET MEASURED — separated but FRAGILE (the "
                          "interval reaches back to within %.0f %% of the point "
                          "estimate). At ONE seed this is not a direction."
                          % (100 * FRAGILE_FRAC))
        out["fragile"] = True
    else:
        out["reading"] = ("SEPARATED on the EPISODE DRAW ONLY — necessary, "
                          "NOT sufficient at one seed")
        out["fragile"] = False
    return out


def paired_cell(a, b, eid, n_boot, seed, direction) -> dict:
    r = paired_episode_cluster_bootstrap(a, b, eid, n_boot=n_boot, seed=seed)
    r["direction"] = direction
    return classify(r)


# --------------------------------------------------------------------------- #
# TRIVIAL-CONTROL EXACTNESS — the instrument check, before any result
# --------------------------------------------------------------------------- #
def constant_arm_checks(dmp: dict, ff_by_arm: dict) -> dict:
    """⭐ The constant-only arm must read the NO-INFORMATION value EXACTLY.

    ``ha0`` is a straight line at constant speed (a = 0, kappa = 0). Three
    quantities are then known in closed form and are not estimates:
      * its curvature is EXACTLY 0 at every step, so its curvature MAE IS the
        straight-line floor mean|kappa_gt|;
      * its tactical decisions are constant, so Cohen's kappa is EXACTLY 0 —
        the no-information value;
      * ha0 vs ha0 paired against itself is EXACTLY 0 with a zero-width interval.
    A departure means the instrument is wrong and nothing downstream is
    admissible. ⛔ These are FAILURES, not warnings.
    """
    checks = []
    self_pair = paired_episode_cluster_bootstrap(
        dmp["ade"]["ha0"], dmp["ade"]["ha0"], dmp["eid"], n_boot=200, seed=0)
    ok = (self_pair["delta"] == 0.0 and self_pair["lo"] == 0.0
          and self_pair["hi"] == 0.0 and not self_pair["separated"])
    checks.append({"check": "ha0 paired against ITSELF is exactly zero-width",
                   "expected": "delta=0 lo=0 hi=0 separated=False",
                   "got": {k: self_pair[k] for k in ("delta", "lo", "hi", "separated")},
                   "pass": bool(ok)})
    tac = (ff_by_arm.get("ha0", {}).get("tactical") or {})
    for key in ("lateral_decision", "longitudinal_decision"):
        k = (tac.get(key) or {}).get("kappa")
        checks.append({"check": "ha0 tactical %s kappa is the NO-INFORMATION value" % key,
                       "expected": 0.0, "got": k,
                       "pass": (k is not None and abs(float(k)) < 1e-12)})
    # ⛔ THE CLOSED-FORM LATERAL CHECK — on ha0's OWN geometry, not on its error.
    #
    # ⚠️ RETRACTED 2026-09-07, IN THIS FILE: the first version of this check
    # asserted |curvature_bias_1pm| == curvature_mae_1pm, reasoning that a
    # straight-line arm's error is exactly -kappa_gt. That is WRONG, and it
    # FAILED on the first real panel (mae 0.006802 vs bias 0.001672). kappa_gt is
    # SIGNED: MAE = mean|kappa_gt| but bias = -mean(kappa_gt), and left and right
    # turns cancel in the second. Equality holds only on a corpus that turns one
    # way. The check was the error, not the instrument -- and it is recorded here
    # rather than quietly deleted.
    #
    # What IS closed-form: `four_families._seq_geometry` returns EXACTLY 0
    # curvature on a straight line at any speed and on any grid
    # (test_four_families_curvature_analytic.py::
    #  test_straight_line_has_exactly_zero_curvature_on_every_grid). So run the
    # real estimator over ha0's OWN paths and demand exactly zero back.
    lat0 = (ff_by_arm.get("ha0", {}).get("lateral") or {})
    kmax, kn = None, 0
    try:
        import torch
        from taniteval import four_families as _FF
        wp = torch.from_numpy(np.ascontiguousarray(
            dmp["paths"]["ha0"], dtype=np.float32))
        geo = _FF._seq_geometry(wp, float(dmp.get("dt_s") or 0.5))
        k = geo["curvature"][geo["pair_valid"]]
        kmax, kn = float(k.abs().max().item()) if k.numel() else 0.0, int(k.numel())
    except Exception as e:                                     # pragma: no cover
        kmax = "UNAVAILABLE: %r" % (e,)
    checks.append({"check": "ha0's OWN curvature is EXACTLY 0 under the real "
                            "masked estimator (so its MAE IS the straight-line floor)",
                   "expected": "max |kappa(ha0)| == 0.0 exactly",
                   "got": {"max_abs_kappa": kmax, "n_valid_pairs": kn,
                           "reported_mae_is_the_floor": lat0.get("curvature_mae_1pm")},
                   "pass": (isinstance(kmax, float) and kmax == 0.0 and kn > 0)})
    # the weaker invariant that must ALSO hold: |bias| <= MAE, always.
    cb, cm = lat0.get("curvature_bias_1pm"), lat0.get("curvature_mae_1pm")
    checks.append({"check": "ha0 |curvature_bias| <= curvature_MAE (an identity, "
                            "since bias = -mean(kappa_gt) and MAE = mean|kappa_gt|)",
                   "expected": "|bias| <= MAE",
                   "got": {"mae": cm, "bias": cb},
                   "pass": (cb is not None and cm is not None
                            and abs(float(cb)) <= float(cm) + 1e-12)})
    return {"n": len(checks), "n_pass": sum(1 for c in checks if c["pass"]),
            "all_pass": all(c["pass"] for c in checks), "checks": checks,
            "⛔": ("A failed check invalidates every number in this panel. The "
                   "constant arm's values are known in closed form; if the "
                   "instrument cannot reproduce a value it cannot get wrong, "
                   "it cannot be trusted on one it can.")}


def run_echo_gate(dmp: dict, n_boot: int, seed: int) -> dict:
    """⛔ THE PRE-COMMITTED ANTI-ECHO GATE, per horizon slot.

    ``stack/tanitad/eval/echo_gate.py`` REFUSES a panel missing either of
    ``("ha", "ha0_ext")`` (:85), and its ``passes`` requires BOTH CI separation
    AND the relative point margin committed in advance — "a separated CI on a
    0.3 %% margin is a real but useless difference, and the prereg committed to
    margins for that reason".

    ⚠️ This is GATE 1 only. ``assert_not_echoing`` additionally needs GATE 2
    (ego intervention) and GATE 2b (source ablation); omitting GATE 2b yields a
    verdict explicitly marked ``STRUCTURAL_ONLY``, which is NOT an anti-echo
    pass — MEASURED 2026-09-03, a deliberate-regression arm PASSED gate 2's
    scene direction because a live encoder moves the output for any model,
    including one that ignores the scene. Both need a live model, so they are
    named here as WORK ITEMS rather than silently skipped.
    """
    try:
        from tanitad.eval.echo_gate import echo_gate as _gate
    except Exception as e:                                     # pragma: no cover
        return {"status": "UNAVAILABLE", "reason": "import failed: %r" % (e,)}
    refs = {k: dmp["per_slot"][k] for k in ("ha", "ha0", "ha0_ext")
            if k in dmp["per_slot"]}
    try:
        g = _gate(arm_ade=dmp["per_slot"]["os"], references=refs, eid=dmp["eid"],
                  margins={"ha0_ext": BAR_PRIMARY["relative_margin_required"],
                           "ha": BAR_PRIMARY["relative_margin_required"]},
                  n_boot=n_boot, seed=seed)
    except ValueError as e:
        return {"status": "REFUSED", "reason": str(e)[:400]}
    g["status"] = "OK"
    g["margins_prereg"] = {"ha0_ext": BAR_PRIMARY["relative_margin_required"],
                           "ha": BAR_PRIMARY["relative_margin_required"]}
    g["gate1_only"] = True
    g["⛔_not_an_anti_echo_pass_alone"] = (
        "GATE 1 (trivial-control margins) only. assert_not_echoing also needs "
        "GATE 2 (ego_intervention_test) and ⛔ GATE 2b (source_ablation_test); "
        "with gate2b=None the verdict is marked STRUCTURAL_ONLY and is NOT an "
        "anti-echo pass. Both need a live model in the loop — WORK ITEM.")
    per_ref = {}
    for name in refs:
        rows = [s["vs"][name] for s in g["slots"]]
        per_ref[name] = {
            "n_slots": len(rows),
            "n_slots_passing": sum(1 for r in rows if r["passes"]),
            "all_slots_pass": all(r["passes"] for r in rows),
            "relative_margin_by_slot": [round(r["relative_margin"], 5) for r in rows],
            "separated_by_slot": [r["separated"] for r in rows]}
    g["per_reference"] = per_ref
    return g


def vacuity_gate(ff_by_arm: dict, arm: str = "os") -> dict:
    """⛔ §5.8 — the manoeuvre RATE beside every decision number.

    MEASURED: one zero-violation result was bought by a ``turn_left`` recall of
    EXACTLY 0.0000. An accuracy or a kappa that never fires the rare class is
    vacuous, and the recall + support is the only thing that shows it.
    """
    tac = (ff_by_arm.get(arm, {}).get("tactical") or {})
    out = {"arm": arm, "status": "OK", "decisions": {}}
    for key in ("lateral_decision", "longitudinal_decision"):
        dd = tac.get(key) or {}
        pc = dd.get("per_class") or {}
        rows, zero = {}, []
        for cls, v in pc.items():
            if not isinstance(v, dict):
                continue
            r = v.get("recall")
            n = v.get("n_true", v.get("support"))
            rows[cls] = {"recall": r, "n_true": n}
            if r is not None and float(r) == 0.0 and n:
                zero.append(cls)
        out["decisions"][key] = {
            "accuracy": dd.get("accuracy"), "kappa": dd.get("kappa"),
            "n": dd.get("n"), "per_class": rows,
            "classes_with_EXACTLY_zero_recall": zero,
            "vacuous": bool(zero),
            "reads": ("⛔ VACUOUS — class(es) %s are never predicted at all; any "
                      "constraint-satisfaction number on this arm was bought by "
                      "not manoeuvring." % zero) if zero else
                     "every labelled class fires at least once"}
    out["any_vacuous"] = any(v["vacuous"] for v in out["decisions"].values())
    return out


# --------------------------------------------------------------------------- #
# STRATEGIC — computed here, because refcv3_arm reports it UNAVAILABLE while
# writing route_pred_*/route_label into its own decisions dump.
# --------------------------------------------------------------------------- #
def cohen_kappa(y_true, y_pred, n_cls) -> float | None:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if y_true.size == 0:
        return None
    cm = np.zeros((n_cls, n_cls), dtype=np.float64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < n_cls and 0 <= p < n_cls:
            cm[int(t), int(p)] += 1
    n = cm.sum()
    if n == 0:
        return None
    po = np.trace(cm) / n
    pe = float((cm.sum(0) * cm.sum(1)).sum()) / (n * n)
    if abs(1.0 - pe) < 1e-12:
        return None
    return float((po - pe) / (1.0 - pe))


def strategic_family(dump_dir: str, n_boot: int, seed: int) -> dict:
    """Route prediction from the decisions dump: accuracy + Cohen's kappa with
    an episode-cluster interval, plus the ECHO test.

    ⛔ REPORTED WITH ITS REASON AND ITS n IF IT CANNOT BE COMPUTED — never
    dropped. ⚠️ ``route_pred`` indexes ROUTE_CLASSES (3-wide) while ``nav_cmd``
    indexes NAV_COMMANDS (4-wide) and _ROUTE_TO_NAV = {0:1, 1:0, 2:2} is NOT the
    identity; comparing them raw is the TYPE ERROR that published a wrong echo
    index on 2026-09-06 (RETRACTION_LOG). This maps before comparing.
    """
    dd = os.path.join(dump_dir, "decisions")
    fs = sorted(glob.glob(os.path.join(dd, "ep*.npz")))
    if not fs:
        return {"status": "UNAVAILABLE", "n": 0,
                "reason": ("no decisions/ep*.npz under %r — refcv3_arm.py writes "
                           "them only when it traverses the hierarchy. Re-roll "
                           "with --dump-dir." % dump_dir),
                "tier": "T1"}
    ROUTE_TO_NAV = {0: 1, 1: 0, 2: 2}
    lab, prd, prd_shuf, prd_zero, nav, eid = [], [], [], [], [], []
    for f in fs:
        z = np.load(f, allow_pickle=True)
        need = ("route_label", "route_pred_nav_true")
        if any(k not in z.files for k in need):
            return {"status": "UNAVAILABLE", "n": 0,
                    "reason": ("decisions dump lacks %s (has %d files); this "
                               "build predates the route head's dump contract."
                               % (need, len(z.files))), "tier": "T1"}
        L = np.asarray(z["route_label"]).ravel()
        lab.append(L)
        prd.append(np.asarray(z["route_pred_nav_true"]).ravel())
        prd_shuf.append(np.asarray(z["route_pred_nav_shuffled"]).ravel()
                        if "route_pred_nav_shuffled" in z.files else np.full(L.size, -1))
        prd_zero.append(np.asarray(z["route_pred_nav_zero"]).ravel()
                        if "route_pred_nav_zero" in z.files else np.full(L.size, -1))
        nav.append(np.asarray(z["nav_cmd"]).ravel() if "nav_cmd" in z.files
                   else np.full(L.size, -1))
        eid.append(np.full(L.size, int(os.path.basename(f)[2:5])))
    lab, prd = np.concatenate(lab), np.concatenate(prd)
    prd_shuf, prd_zero = np.concatenate(prd_shuf), np.concatenate(prd_zero)
    nav, eid = np.concatenate(nav), np.concatenate(eid)
    m = (lab >= 0) & (prd >= 0)
    n_lab = int(m.sum())
    if n_lab == 0:
        return {"status": "UNAVAILABLE", "n": 0,
                "reason": "route_label is -1 on every window (no labelled route).",
                "tier": "T1"}
    hit = (lab[m] == prd[m]).astype(np.float64)
    acc = episode_cluster_bootstrap(hit, eid[m], n_boot=n_boot, seed=seed)
    kap = cohen_kappa(lab[m], prd[m], 3)
    out = {"status": "OK", "tier": "T1", "n": n_lab,
           "n_episodes": len(set(eid[m].tolist())),
           "route_acc": round(float(hit.mean()), 4),
           "route_acc_ci": {"lo": acc.get("lo"), "hi": acc.get("hi")},
           "route_kappa": (round(kap, 4) if kap is not None else None),
           "route_chance_1_over_3": round(1.0 / 3.0, 6),
           "estimator": ("episode_cluster_bootstrap (taniteval/ci.py), "
                         "n_boot %d, seed %d, cluster = episode" % (n_boot, seed)),
           "_computed_by": ("refcv5_compare.strategic_family — refcv3_arm.py's "
                            "own four_families reports STRATEGIC UNAVAILABLE "
                            "while writing route_pred_*/route_label into the "
                            "decisions dump it produced. Computed, not dropped.")}
    mn = m & (nav >= 0)
    if mn.any():
        nav_of_route = np.array([ROUTE_TO_NAV.get(int(r), -1) for r in prd[mn]])
        out["nav_echo_index"] = round(float((nav_of_route == nav[mn]).mean()), 4)
        out["_echo_index_is"] = ("fraction of windows where the route head's "
                                 "prediction, MAPPED THROUGH _ROUTE_TO_NAV "
                                 "{0:1,1:0,2:2}, equals the fed nav token. ⚠️ "
                                 "Comparing the two raw indices is a TYPE ERROR "
                                 "(3-wide ROUTE_CLASSES vs 4-wide NAV_COMMANDS) "
                                 "and published a wrong number on 2026-09-06.")
    ms = m & (prd_shuf >= 0)
    if ms.any():
        same = float((prd[ms] == prd_shuf[ms]).mean())
        out["route_pred_identical_under_nav_shuffle"] = round(same, 6)
        out["_shuffle_reading"] = (
            "1.000000 is a STRUCTURAL identity, not a noisy null: the route head "
            "reads the observed window only and never the nav token. ⭐ It is "
            "therefore NOT an echo of its input." if same == 1.0 else
            "below 1.0 ⇒ the route head DOES consume the nav token; the echo "
            "question is LIVE and route_acc is not admissible as vision-only "
            "route prediction.")
    mz = m & (prd_zero >= 0)
    if mz.any():
        out["route_pred_identical_under_nav_zero"] = round(
            float((prd[mz] == prd_zero[mz]).mean()), 6)
    return out


# --------------------------------------------------------------------------- #
# the four-family table
# --------------------------------------------------------------------------- #
def _g(d, *ks, default=None):
    for k in ks:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def four_family_table(res: dict, dump_dir: str, n_boot: int, seed: int) -> dict:
    """Per arm, the four families NEVER POOLED, each with its estimator and CI.
    A family that cannot be computed carries its reason AND its n."""
    ff_by_arm, table = {}, {}
    for arm, a in (res.get("arms") or {}).items():
        ff = a.get("four_families") or {}
        ff_by_arm[arm] = ff
        iv_m = _g(a, "intervals", "metrics", default={}) or {}
        ade = iv_m.get("ade_dense_m") or iv_m.get("ade_m") or {}
        lon, lat = ff.get("longitudinal") or {}, ff.get("lateral") or {}
        tac, st = ff.get("tactical") or {}, ff.get("strategic") or {}
        table[arm] = {
            "tier": a.get("tier"), "tier_note": TIER_NOTE,
            "ade_m": {"mean": ade.get("mean"), "lo": ade.get("lo"), "hi": ade.get("hi"),
                      "_is": "ONE ROW OF FOUR FAMILIES, never the result"},
            "LONGITUDINAL": {
                "speed_mae_mps": lon.get("speed_mae_mps"),
                "speed_bias_mps": lon.get("speed_bias_mps"),
                "target_speed_acc": lon.get("target_speed_acc"),
                "along_mae_m": lon.get("along_mae_m"),
                "accel_mae_mps2": lon.get("accel_mae_mps2"),
                "distance_keeping": lon.get("distance_keeping"),
                "estimator": _g(a, "intervals", "estimator"),
            },
            "LATERAL": {
                "heading_mae_deg": lat.get("heading_mae_deg"),
                "yaw_rate_mae_degps": lat.get("yaw_rate_mae_degps"),
                "curvature_mae_1pm": lat.get("curvature_mae_1pm"),
                "curvature_bias_1pm": lat.get("curvature_bias_1pm"),
                "cross_mae_m": lat.get("cross_mae_m"),
                "n_steps_curvature": lat.get("n_steps_curvature"),
                "excluded_below_min_ds": lat.get("excluded_below_min_ds"),
                "min_ds_m": lat.get("min_ds_m"),
                "⛔_masked_estimator": (
                    "curvature = |d1 x d2| / max(|d1|^3, eps) is SINGULAR as "
                    "speed -> 0. `pair_valid` (min_ds_m above) is NOT optional: "
                    "MEASURED 2026-09-07, dropping it let 43 stopped windows "
                    "(4.9 %%, v0 = 0.000) publish '84x worse than a never-steer "
                    "plan', which reverses to 0.64x — BETTER, separated — once "
                    "masked. Pinned by "
                    "taniteval/tests/test_four_families_curvature_analytic.py "
                    "against an analytic circle (1/R) and a line (exactly 0). "
                    "⛔ A curvature number without a validity mask is "
                    "INADMISSIBLE. excluded_below_min_ds is the mask's own n."),
            },
            "TACTICAL": {
                "status": tac.get("status"),
                "n": tac.get("n"),
                "lateral_decision": tac.get("lateral_decision"),
                "longitudinal_decision": tac.get("longitudinal_decision"),
                "goal_setting": tac.get("goal_setting"),
                "estimator": tac.get("_estimator"),
            },
            "STRATEGIC": st,
        }
    # STRATEGIC, computed from the decisions dump for the model arm only
    strat = strategic_family(dump_dir, n_boot, seed)
    for arm in ("os",):
        if arm in table:
            declared = table[arm]["STRATEGIC"] or {}
            table[arm]["STRATEGIC"] = {
                "as_declared_by_refcv3_arm": {
                    "status": declared.get("status"), "n": declared.get("n"),
                    "reason": declared.get("reason")},
                "computed_here": strat}
    # the straight-line floor printed BESIDE every curvature
    floor = _g(ff_by_arm, "ha0", "lateral", "curvature_mae_1pm")
    echo = _g(ff_by_arm, "ha0_ext", "lateral", "curvature_mae_1pm")
    for arm, row in table.items():
        row["LATERAL"]["straight_line_floor_1pm"] = floor
        row["LATERAL"]["echo_control_ha0_ext_1pm"] = echo
        k = row["LATERAL"]["curvature_mae_1pm"]
        if k is not None and floor:
            row["LATERAL"]["ratio_to_straight_line_floor"] = round(float(k) / float(floor), 4)
            row["LATERAL"]["reads"] = (
                "IS the straight-line floor (kappa == 0 by construction — the "
                "reference, not a result)" if arm == "ha0" else
                "BELOW the straight-line floor (tracks the road better than a "
                "plan that never steers)" if float(k) < float(floor) else
                "⛔ ABOVE the straight-line floor — a SHAPE defect: a plan that "
                "never steers tracks the road better")
    return {"table": table, "ff_by_arm": ff_by_arm, "strategic_computed": strat}


# --------------------------------------------------------------------------- #
def render(rep: dict) -> str:
    L, W = [], 108
    A, B = rep["labels"]["new"], rep["labels"]["base"]
    L.append("=" * W)
    L.append("REFCV5-V2 vs REFCV4B — FOUR-FAMILY PANEL   [%s  vs  %s]" % (A, B))
    L.append("  n_windows=%d  n_episodes=%d   grid: dt=%s s, K=%s   T-TIER: T1 on every number"
             % (rep["grid"]["n_windows"], rep["grid"]["n_episodes"],
                rep["grid"]["dt_s"], rep["grid"]["horizon_steps"]))
    L.append("  " + TIER_NOTE)
    L.append("  " + NAV_STAMP)
    L.append("=" * W)
    L.append("")
    L.append("[0] INSTRUMENT — the constant-only arm must read the NO-INFORMATION value EXACTLY")
    cc = rep["constant_arm_checks"]
    for c in cc["checks"]:
        L.append("    %-4s %s" % ("PASS" if c["pass"] else "FAIL", c["check"]))
        L.append("         expected %s   got %s" % (c["expected"], json.dumps(c["got"], default=str)[:90]))
    L.append("    => %s (%d/%d)" % ("ALL PASS" if cc["all_pass"] else "⛔ FAILED",
                                    cc["n_pass"], cc["n"]))
    L.append("")
    L.append("[0b] ANTI-ECHO GATE 1 — stack/tanitad/eval/echo_gate.py, PER HORIZON SLOT")
    L.append("     ⛔ `passes` = CI excludes zero AND the relative point margin committed in "
             "advance (%.0f %%)." % (100 * BAR_PRIMARY["relative_margin_required"]))
    for tag in ("new", "base"):
        gt = rep["echo_gate1"][tag]
        if gt.get("status") != "OK":
            L.append("    %-12s %s: %s" % (rep["labels"][tag], gt.get("status"),
                                           str(gt.get("reason"))[:90]))
            continue
        for name, r in (gt.get("per_reference") or {}).items():
            L.append("    %-12s vs %-8s slots passing %s/%s   relative margin by slot %s"
                     % (rep["labels"][tag], name, r["n_slots_passing"], r["n_slots"],
                        r["relative_margin_by_slot"]))
    L.append("     ⚠️ GATE 1 ONLY. GATE 2 (ego intervention) and ⛔ GATE 2b (source ablation) "
             "need a live model — WORK ITEMS.")
    L.append("        With gate2b absent, assert_not_echoing returns STRUCTURAL_ONLY, which is "
             "NOT an anti-echo pass.")
    L.append("")
    L.append("[0c] VACUITY GATE — the manoeuvre RATE beside every decision number")
    for tag in ("new", "base"):
        v = rep["vacuity_gate"][tag]
        L.append("    %-12s any_vacuous=%s" % (rep["labels"][tag], v.get("any_vacuous")))
        for kk, dd in (v.get("decisions") or {}).items():
            L.append("        %-22s %s" % (kk, dd.get("reads")))
    L.append("")
    for tag in ("new", "base"):
        t = rep["four_families"][tag]["table"]
        L.append("-" * W)
        L.append("FOUR FAMILIES — %s   (n and d printed per row; ⛔ ADE alone is an INCOMPLETE result)"
                 % rep["labels"][tag])
        L.append("-" * W)
        for arm in ("os", "ha", "ha0", "ha0_ext"):
            r = t.get(arm)
            if not r:
                continue
            ade = r["ade_m"]
            L.append("  ARM %-8s TIER %-4s  ADE %s m  [%s, %s]"
                     % (arm, r["tier"], _f(ade.get("mean")), _f(ade.get("lo")), _f(ade.get("hi"))))
            lo_ = r["LONGITUDINAL"]
            dk = lo_.get("distance_keeping") or {}
            tsa = lo_.get("target_speed_acc")
            tsa = (tsa.get("within_0.5_mps") if isinstance(tsa, dict) else tsa)
            L.append("    LONGITUDINAL  speed_MAE %s m/s  bias %s  target_speed_acc@0.5 %s  along_MAE %s m"
                     % (_f(lo_.get("speed_mae_mps")), _f(lo_.get("speed_bias_mps")),
                        _f(tsa), _f(lo_.get("along_mae_m"))))
            L.append("                  distance-keeping/TTC: status=%s  n=%s of %s windows  "
                     "mean_headway_min %s m  mean_time_gap_min %s s (n %s)  mean_min_TTC %s s (n_closing %s)"
                     % (dk.get("status"), dk.get("n"), dk.get("n_windows"),
                        _f(dk.get("mean_headway_min_m")),
                        _f(dk.get("mean_time_gap_min_s")), dk.get("n_time_gap"),
                        _f(dk.get("mean_min_ttc_s")), dk.get("n_closing")))
            cens = dk.get("censoring_note")
            if cens:
                L.append("                  ⚠️ %s" % str(cens)[:120])
            la = r["LATERAL"]
            L.append("    LATERAL       heading %s deg  yaw_rate %s deg/s  cross_track %s m"
                     % (_f(la.get("heading_mae_deg")), _f(la.get("yaw_rate_mae_degps")),
                        _f(la.get("cross_mae_m"))))
            L.append("       ⭐ CURVATURE_MAE %s 1/m   straight-line floor (ha0) %s   ratio %s  -> %s"
                     % (_f(la.get("curvature_mae_1pm"), 6), _f(la.get("straight_line_floor_1pm"), 6),
                        _f(la.get("ratio_to_straight_line_floor"), 3), la.get("reads", "")))
            L.append("          MASKED estimator: n_steps=%s  excluded_below_min_ds=%s (min_ds=%s m)"
                     % (la.get("n_steps_curvature"), la.get("excluded_below_min_ds"),
                        la.get("min_ds_m")))
            ta = r["TACTICAL"]
            L.append("    TACTICAL      status=%s n=%s" % (ta.get("status"), ta.get("n")))
            for kk in ("lateral_decision", "longitudinal_decision"):
                dd = ta.get(kk) or {}
                L.append("       %-22s acc %s  kappa %s  n %s"
                         % (kk, _f(dd.get("accuracy")), _f(dd.get("kappa")), dd.get("n")))
                # ⛔ VACUITY GATE (§5.8): the manoeuvre RATE beside every
                # decision number. A zero-violation result once cost exactly a
                # turn_left recall of 0.0000.
                pc = dd.get("per_class") or {}
                bits = ["%s r=%s(n%s)" % (c, _f(v.get("recall"), 3),
                                          v.get("n_true", v.get("support")))
                        for c, v in pc.items() if isinstance(v, dict)]
                if bits:
                    L.append("           per-class recall: " + "  ".join(bits))
            if arm == "os":
                asl = (rep.get("anchor_selection") or {}).get(tag) or {}
                if asl:
                    aa = asl.get("anchor_acc") or {}
                    L.append("       anchor_selection       acc %s [%s, %s]  "
                             "chance 1/%s = %s  lift %sx  n %s"
                             % (_f(aa.get("mean")), _f(aa.get("lo")),
                                _f(aa.get("hi")), asl.get("n_anchors"),
                                _f(asl.get("chance"), 6), asl.get("lift_over_chance"),
                                aa.get("n_windows")))
                    sp_ = asl.get("selection_profile") or {}
                    L.append("           bank size from %s | distinct %s of %s, "
                             "modal %s (%s), entropy_ratio %s%s"
                             % (((asl.get("n_anchors_provenance") or {})
                                 .get("primary") or "?"),
                                asl.get("n_distinct_selected"), asl.get("n_anchors"),
                                sp_.get("modal_anchor"), _f(sp_.get("modal_frac"), 3),
                                _f(sp_.get("entropy_ratio"), 3),
                                ("  ⛔ selection_profile REFUSED: "
                                 + str(sp_.get("reason"))[:60])
                                if sp_.get("status") else ""))
                    if asl.get("_note"):
                        L.append("           " + str(asl["_note"])[:150])
            gs = ta.get("goal_setting") or {}
            L.append("       goal_setting           FDE %s m  bearing_MAE %s deg  n %s"
                     % (_f(gs.get("goal_point_error_m")), _f(gs.get("goal_bearing_mae_deg")),
                        gs.get("n")))
            st = r["STRATEGIC"]
            if arm == "os" and isinstance(st, dict) and "computed_here" in st:
                s = st["computed_here"]
                if s.get("status") == "OK":
                    L.append("    STRATEGIC     route_acc %s [%s, %s]  kappa %s  n %s / %s eps  "
                             "(chance 1/3 = 0.3333)"
                             % (_f(s.get("route_acc")), _f(s.get("route_acc_ci", {}).get("lo")),
                                _f(s.get("route_acc_ci", {}).get("hi")), _f(s.get("route_kappa")),
                                s.get("n"), s.get("n_episodes")))
                    L.append("                  nav_echo_index %s   route_pred identical under "
                             "nav-shuffle %s" % (_f(s.get("nav_echo_index")),
                                                 s.get("route_pred_identical_under_nav_shuffle")))
                else:
                    L.append("    STRATEGIC     ⛔ %s  n=%s  reason: %s"
                             % (s.get("status"), s.get("n"), str(s.get("reason"))[:110]))
            else:
                L.append("    STRATEGIC     %s  n=%s%s"
                         % (st.get("status"), st.get("n"),
                            ("  reason: " + str(st.get("reason"))[:100])
                            if st.get("status") != "OK" else ""))
            L.append("")
    L.append("=" * W)
    L.append("PAIRED MARGINS — paired episode-cluster bootstrap (taniteval/ci.py), "
             "n_boot=%d, seed=%d, cluster=EPISODE" % (rep["n_boot"], rep["seed"]))
    L.append("⛔ overlapping_holdout_se is NOT used anywhere: it biases the POINT ESTIMATE, "
             "not merely the interval.")
    L.append("=" * W)
    for name, c in rep["paired"].items():
        L.append("  %-34s delta %s  [%s, %s]  separated=%s"
                 % (name, _f(c.get("delta"), 4), _f(c.get("lo"), 4), _f(c.get("hi"), 4),
                    c.get("separated")))
        L.append("       %s" % c.get("reading"))
        if c.get("separated"):
            vq = c.get("variance_questions", {})
            L.append("       variance answered  | episode draw : %s" % vq.get("episode_draw", "")[:88])
            L.append("       variance NOT       | training run : %s" % vq.get("training_run", "")[:88])
            L.append("       variance           | inference run: %s" % vq.get("inference_run", "")[:88])
    m = rep["multiplicity"]
    L.append("")
    L.append("  MULTIPLICITY: %d of %d cells read SEPARATED = %.1f %%; a PURE REPLICATE "
             "produced %.1f %% (6/42)." % (m["n_separated"], m["n_cells"],
                                           100 * m["frac"], 100 * REPLICATE_FP_RATE))
    L.append("  => %s" % m["reading"])
    L.append("")
    L.append("=" * W)
    L.append("VERDICT against the bar registered %s, BEFORE the checkpoint existed"
             % BAR_PRIMARY["registered_utc"])
    L.append("=" * W)
    for v in rep["verdicts"]:
        L.append("  [%s] %s" % (v["id"], v["statement"]))
        L.append("      measured: delta %s [%s, %s] separated=%s"
                 % (_f(v.get("delta"), 4), _f(v.get("lo"), 4), _f(v.get("hi"), 4),
                    v.get("separated")))
        L.append("      ==> %s" % v["verdict"])
    L.append("")
    L.append("  " + rep["headline"])
    L.append("=" * W)
    return "\n".join(L)


def _f(v, nd=4):
    if v is None:
        return " -- "
    try:
        return ("%." + str(nd) + "f") % float(v)
    except Exception:
        return str(v)


# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--new-json", required=True)
    ap.add_argument("--new-dump", required=True)
    ap.add_argument("--new-label", default="refcv5-v2")
    ap.add_argument("--new-config", default=None,
                    help="the run's config.json — read ONLY to decide whether "
                         "the sampler is stochastic at inference")
    ap.add_argument("--base-json", required=True)
    ap.add_argument("--base-dump", required=True)
    ap.add_argument("--base-label", default="refcv4b")
    ap.add_argument("--base-config", default=None)
    ap.add_argument("--new-ckpt", default=None,
                    help="the new arm's checkpoint — read ONLY for "
                         "core.decoder.anchors.shape[0], the AUTHORITATIVE bank "
                         "size (mmap, 0.03 s). Optional: the dump manifest and "
                         "the config are witnesses too, but the tensor is the "
                         "object the model actually uses.")
    ap.add_argument("--base-ckpt", default=None)
    ap.add_argument("--prior-dump", default=None,
                    help="refcv3 @40,284 dump — the CONTROL that must reproduce "
                         "the published -0.1444 m")
    ap.add_argument("--prior-label", default="refcv3")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-prefix", required=True)
    a = ap.parse_args(argv)

    t0 = time.time()
    new_res = json.load(open(a.new_json, encoding="utf-8"))
    base_res = json.load(open(a.base_json, encoding="utf-8"))
    new_d, base_d = load_dump(a.new_dump), load_dump(a.base_dump)
    grid = assert_same_grid(new_d, base_d, a.new_label, a.base_label)
    _p("[compare] %s" % json.dumps(grid))

    new_cfg = json.load(open(a.new_config, encoding="utf-8")) if a.new_config else None
    base_cfg = json.load(open(a.base_config, encoding="utf-8")) if a.base_config else None
    facts_new = config_facts(new_cfg)
    facts_base = config_facts(base_cfg)
    new_man = _read_manifest(a.new_dump)
    base_man = _read_manifest(a.base_dump)
    # ⛔ REFUSES rather than defaulting. See resolve_n_anchors.
    n_anch_new, n_anch_prov_new = resolve_n_anchors(new_man, new_cfg, a.new_ckpt)
    n_anch_base, n_anch_prov_base = resolve_n_anchors(base_man, base_cfg, a.base_ckpt)
    anchor_new = anchor_selection_block(new_res, n_anch_new, n_anch_prov_new)
    anchor_base = anchor_selection_block(base_res, n_anch_base, n_anch_prov_base)
    stoch, why = sampler_is_stochastic(new_cfg)
    vq = variance_block(stoch, why, a.n_boot)

    ff_new = four_family_table(new_res, a.new_dump, a.n_boot, a.seed)
    ff_base = four_family_table(base_res, a.base_dump, a.n_boot, a.seed)
    cc = constant_arm_checks(new_d, ff_new["ff_by_arm"])
    gate1 = run_echo_gate(new_d, a.n_boot, a.seed)
    gate1_base = run_echo_gate(base_d, a.n_boot, a.seed)
    vac = vacuity_gate(ff_new["ff_by_arm"])
    vac_base = vacuity_gate(ff_base["ff_by_arm"])

    eid = new_d["eid"]
    paired = {}
    for ctrl in TRIVIAL_ARMS:
        if ctrl in new_d["ade"]:
            paired["%s: os - %s" % (a.new_label, ctrl)] = paired_cell(
                new_d["ade"]["os"], new_d["ade"][ctrl], eid, a.n_boot, a.seed,
                "os - %s" % ctrl)
    for ctrl in TRIVIAL_ARMS:
        if ctrl in base_d["ade"]:
            paired["%s: os - %s" % (a.base_label, ctrl)] = paired_cell(
                base_d["ade"]["os"], base_d["ade"][ctrl], eid, a.n_boot, a.seed,
                "os - %s" % ctrl)
    paired["%s - %s  (os)" % (a.new_label, a.base_label)] = paired_cell(
        new_d["ade"]["os"], base_d["ade"]["os"], eid, a.n_boot, a.seed,
        "%s.os - %s.os" % (a.new_label, a.base_label))
    # ⛔ the model-free CONTROL: the trivial arms are functions of the corpus,
    # not of the checkpoint, so they MUST read exactly 0 across two runs.
    for ctrl in TRIVIAL_ARMS:
        if ctrl in new_d["ade"] and ctrl in base_d["ade"]:
            paired["CONTROL %s - %s  (%s, must be 0)"
                   % (a.new_label, a.base_label, ctrl)] = paired_cell(
                new_d["ade"][ctrl], base_d["ade"][ctrl], eid, a.n_boot, a.seed,
                "%s.%s - %s.%s" % (a.new_label, ctrl, a.base_label, ctrl))
    prior = None
    if a.prior_dump:
        prior = load_dump(a.prior_dump)
        assert_same_grid(base_d, prior, a.base_label, a.prior_label)
        paired["%s - %s  (os)" % (a.base_label, a.prior_label)] = paired_cell(
            base_d["ade"]["os"], prior["ade"]["os"], eid, a.n_boot, a.seed,
            "%s.os - %s.os" % (a.base_label, a.prior_label))
        paired["%s - %s  (os)" % (a.new_label, a.prior_label)] = paired_cell(
            new_d["ade"]["os"], prior["ade"]["os"], eid, a.n_boot, a.seed,
            "%s.os - %s.os" % (a.new_label, a.prior_label))
        for ctrl in ("ha", "ha0"):
            if ctrl in prior["ade"] and ctrl in base_d["ade"]:
                paired["CONTROL %s - %s  (%s, must be 0)"
                       % (a.base_label, a.prior_label, ctrl)] = paired_cell(
                    base_d["ade"][ctrl], prior["ade"][ctrl], eid, a.n_boot, a.seed,
                    "%s.%s - %s.%s" % (a.base_label, ctrl, a.prior_label, ctrl))
    for k, c in paired.items():
        if c.get("separated"):
            c["variance_questions"] = vq

    real = {k: v for k, v in paired.items() if not k.startswith("CONTROL")}
    n_sep = sum(1 for v in real.values() if v.get("separated"))
    frac = n_sep / max(1, len(real))
    mult = {"n_cells": len(real), "n_separated": n_sep, "frac": frac,
            "replicate_fp_rate": REPLICATE_FP_RATE,
            "reading": (("⛔ the panel's separated fraction (%.1f %%) is AT OR "
                         "BELOW the pure-replicate rate (%.1f %%) — as a PANEL "
                         "this is not distinguishable from a replicate."
                         % (100 * frac, 100 * REPLICATE_FP_RATE))
                        if frac <= REPLICATE_FP_RATE else
                        ("the panel's separated fraction (%.1f %%) EXCEEDS the "
                         "pure-replicate rate (%.1f %%); individual cells still "
                         "answer only the EPISODE-DRAW question."
                         % (100 * frac, 100 * REPLICATE_FP_RATE)))}

    verdicts = []
    for bar in (BAR_PRIMARY, BAR_SECONDARY):
        c = paired.get("%s: os - %s" % (a.new_label, bar["control"]))
        v = dict(bar)
        if c is None:
            v["verdict"] = ("⛔ NOT EVALUABLE — the control arm %r is absent "
                            "from the dump." % bar["control"])
        else:
            v.update({k: c.get(k) for k in ("delta", "lo", "hi", "separated")})
            # ⛔ echo_gate's own criterion: CI separation AND the pre-registered
            # relative margin, on EVERY horizon slot. Either alone is not it.
            gref = (gate1.get("per_reference") or {}).get(bar["control"]) or {}
            v["echo_gate_all_slots_pass"] = gref.get("all_slots_pass")
            v["echo_gate_slots_passing"] = "%s/%s" % (gref.get("n_slots_passing"),
                                                      gref.get("n_slots"))
            v["relative_margin_by_slot"] = gref.get("relative_margin_by_slot")
            ci_ok = c.get("delta", 0) < 0 and c.get("hi", 0) < 0
            if ci_ok and gref.get("all_slots_pass") and not c.get("fragile"):
                v["verdict"] = ("✅ PASS — beats %s SEPARATED on the episode draw "
                                "AND clears the pre-registered %.0f %% relative "
                                "margin on every horizon slot"
                                % (bar["control"],
                                   100 * BAR_PRIMARY["relative_margin_required"]))
            elif ci_ok and c.get("fragile"):
                v["verdict"] = "⛔ NOT YET MEASURED — separated but FRAGILE at one seed"
            elif ci_ok and gref.get("all_slots_pass") is False:
                v["verdict"] = ("⛔ FAIL — CI separates but the pre-registered "
                                "%.0f %% relative margin does NOT hold on every "
                                "slot (%s). A separated CI on a small margin is "
                                "a real but useless difference."
                                % (100 * BAR_PRIMARY["relative_margin_required"],
                                   v["echo_gate_slots_passing"]))
            else:
                v["verdict"] = ("⛔ FAIL — does NOT beat %s separated. %s"
                                % (bar["control"], c.get("reading")))
        verdicts.append(v)
    c = paired.get("%s - %s  (os)" % (a.new_label, a.base_label))
    v = dict(BAR_TERTIARY)
    v.update({k: (c or {}).get(k) for k in ("delta", "lo", "hi", "separated")})
    v["verdict"] = ("INFORMATIONAL — %s. ⛔ Beating %s is NOT the bar: %s itself "
                    "failed BAR-REFCV5V2-1."
                    % ((c or {}).get("reading", "unavailable"), a.base_label, a.base_label))
    verdicts.append(v)

    primary = verdicts[0]
    headline = ("HEADLINE: %s — %s"
                % (primary["verdict"].split(" — ")[0],
                   ("%s beats a plan that holds its own t0 action and curvature, "
                    "separated (T1, open loop, EPISODE-DRAW variance only — the "
                    "training-run and, for a stochastic sampler, the "
                    "inference-run questions are still open)." % a.new_label
                    if primary["verdict"].startswith("✅") else
                    "%s has NOT cleared the pre-registered bar: it does not "
                    "separate from the do-nothing baselines on this surface, "
                    "which is exactly where refcv4b also stands."
                    % a.new_label)))

    rep = {
        "tool": "refcv5_compare.py",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "labels": {"new": a.new_label, "base": a.base_label, "prior": a.prior_label},
        "ckpts": {"new": new_res.get("ckpt"), "base": base_res.get("ckpt")},
        "dumps": {"new": a.new_dump, "base": a.base_dump, "prior": a.prior_dump},
        "grid": {"n_windows": new_d["n_windows"], "n_episodes": new_d["n_episodes"],
                 "dt_s": new_res.get("dt_s"), "horizon_steps": new_res.get("horizon_steps"),
                 "pairing": grid},
        "n_boot": a.n_boot, "seed": a.seed,
        "tier": "T1", "tier_note": TIER_NOTE, "nav_stamp": NAV_STAMP,
        "bars": [BAR_PRIMARY, BAR_SECONDARY, BAR_TERTIARY],
        "constant_arm_checks": cc,
        "echo_gate1": {"new": gate1, "base": gate1_base},
        "vacuity_gate": {"new": vac, "base": vac_base},
        "four_families": {"new": ff_new, "base": ff_base},
        "paired": paired,
        "multiplicity": mult,
        "variance_questions": vq,
        "sampler_stochastic_at_inference": stoch,
        "config_facts": {"new": facts_new, "base": facts_base},
        "anchor_selection": {"new": anchor_new, "base": anchor_base},
        "verdicts": verdicts,
        "headline": headline,
        "estimator": ("paired_episode_cluster_bootstrap (taniteval/ci.py); "
                      "point estimates are FULL-SET pooled means over windows. "
                      "⛔ overlapping_holdout_se is NOT used anywhere."),
        "arms_missing_from_dump": {"new": new_d["arms_missing"],
                                   "base": base_d["arms_missing"]},
        "wall_s": round(time.time() - t0, 1),
    }
    with open(a.out_prefix + ".json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(rep, fh, indent=1, default=float, ensure_ascii=False)
    txt = render(rep)
    with open(a.out_prefix + ".txt", "w", encoding="utf-8", newline="\n") as fh:
        fh.write(txt + "\n")
    _p(txt)
    _p("\n[compare] wrote %s.json and %s.txt (%.1f s)"
       % (a.out_prefix, a.out_prefix, rep["wall_s"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
