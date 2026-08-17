#!/usr/bin/env python3
"""v6_chain.py — THE STAGE CHAIN: S-W -> S-T{goal,mlp} -> S-S -> S-J.

WHY THIS FILE EXISTS, AND WHY IT IS NOT A SHELL SCRIPT
======================================================
`stack/scripts/` holds five `*_chain.sh` files, and every one of them
orchestrates *evaluation tools*: fixed steps, no branching, no state. The v6
ladder is a different animal — it BRANCHES on a gate verdict, it carries a
two-arm comparison, and each edge of it has already produced a defect. Six were
found in the two days before this file was written, and **all six were found by
EXECUTING a transition; none by reading code**:

  * S-S invalidates S-T's frozen selector certificate — no gate existed (dc50dbc)
  * `"goal"` had no capacity control, so a win would be unattributable (b12c190)
  * `--init-from` refused the designed selector introduction — **both S-T arms
    were unlaunchable** (8e215b3)
  * `--resume auto` had NO stage check; the only barrier was `torch.optim`
    complaining about list lengths, and it is skipped for the fp16 snapshot
  * `--init-from` + `--resume auto` recorded the WRONG ancestor's md5
  * `--init-from <fp16 snapshot>` blamed geometry for an unopened container

That history is the design brief. This driver is Python because its logic is
*adjudication*, and adjudication has to be pinned by tests — `tests/
test_v6_chain.py` executes every refusal in this file. A bash chain would move
the ladder's branch points into a place no test can reach, which is exactly how
the six defects above survived.

THE FIVE PROPERTIES THIS CHAIN GUARANTEES
=========================================
1. ⛔ **Every stage and every arm gets its OWN `--out`.** :func:`assert_plan`
   refuses a plan in which two steps share one. The trainer's stage-labelled
   `--resume auto` refusal is the *last* line of defence; a chain that leans on
   it has already created the situation. `--resume auto` inside one stage's own
   directory then means exactly what it says.
2. **Every stage after S-W carries BOTH `--init-from <prev ckpt.pt>` and
   `--prev-gate <prev stage_gate.json>`** — the weights and the certificate.
   Missing either is a refusal, in the chain and again in the trainer.
3. ⛔ **The chain REFUSES to advance on a non-PASS gate.** FAIL has no override
   at all. INCONCLUSIVE is NOT a pass; it needs `--allow-inconclusive-gate`
   AND a recorded `--gate-off-reason`. There is exactly ONE adjudicator —
   `train_v6_staged.assert_stage_precondition` — imported, never re-implemented,
   because two copies of a gate rule is how a ladder drifts.
4. ⛔ **S-T's DEFAULT IS `--selector none`, because SEL-1 IS REFUSED.** E-WC2
   fired 2026-08-16 against a threshold pre-registered in `V6F_PLANNER_DESIGN`
   §5.2 with both outcomes committed in advance — see :data:`SEL1_ADMISSION`.
   A selector arm is therefore reachable only as an OPT-IN (`--st-arms`), and
   only once the S-W latent surface has been measured and reaches the
   thresholds pre-registered HERE, in code, before that measurement is taken.
   When arms ARE opted into, `"goal"` and `"mlp"` are an ARM PAIR: a `"goal"`
   result judged without its capacity control is unattributable between
   MECHANISM and CAPACITY (the C6 confound), so a `"goal"`-only ladder is not
   launchable without a recorded `--unpaired-arm-reason`.
5. **The selector GEOMETRY is carried forward.** S-S and S-J must be launched
   with the S-T lineage's `--selector`, or `--init-from` dies on unexpected
   `cand_score.*` keys. MEASURED 2026-08-16 — see `assert_geometry_carry`.

WHAT THIS CHAIN DELIBERATELY DOES NOT DO
========================================
⛔ **It never runs the four real stages inside one process.** Each stage is a
multi-day GPU job; a long-lived orchestrator on a pod dies with its ssh session
and takes the ladder's state with it. The state lives in the FILESYSTEM — the
gate files and the done-markers — and `status` / `next` recompute it from
scratch every time. `run` exists for the CPU dry ladder (and small local
ladders); on Thor you use `next` + `commands` and launch one stage.

⚠️ **It must never be a supervisor's `TRAIN_CMD`.** `supervise_run.sh` SOURCES
ITS MANIFEST ONCE at supervisor startup and replays the command it captured, so
a supervised *chain* would replay stage 1 after a mid-ladder crash. `manifests`
therefore emits ONE MANIFEST PER STAGE whose `TRAIN_CMD` is the TRAINER, and
`tests/test_v6_chain.py` pins that no emitted manifest names this file.

⚠️ **torch is imported LAZILY.** `uv pip install <anything>` has twice replaced
a pod's torch with a wheel its driver cannot run, and on that pod you still want
`plan`, `commands`, `status` and `manifests` to work. Anything that genuinely
needs torch says so when torch is missing instead of dying at import.

USAGE
=====
    python3 scripts/v6_chain.py plan                    # the resolved ladder
    python3 scripts/v6_chain.py commands --step S-T:goal
    python3 scripts/v6_chain.py status
    python3 scripts/v6_chain.py next                    # what may launch NOW
    python3 scripts/v6_chain.py manifests --dest ops/runs.d
    python3 scripts/v6_chain.py verify --step S-T:goal  # the /proc probe
    python3 scripts/v6_chain.py run --dry --root /tmp/ladder   # CPU, ~1 min
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import posixpath
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

_HERE = Path(__file__).resolve().parent                  # stack/scripts
_STACK = _HERE.parent                                    # stack
sys.path.insert(0, str(_HERE))
sys.path.insert(1, str(_STACK))

TRAINER = "scripts/train_v6_staged.py"

# ---------------------------------------------------------------------------
# Thor, MEASURED. Every one of these is counter-intuitive and every one of them
# has a measurement behind it, so the chain carries the numbers rather than the
# folklore.
# ---------------------------------------------------------------------------
#: s/step on the Jetson Thor at the production geometry. MEASURED 2026-08-16 on
#: the live v6F S-W run, marginal over steps 6300->6400; three statistics with
#: different startup exposure agree to 0.5 % (50-step marginal 27.21,
#: 100-step marginal 27.18, cumulative-with-startup 27.32). A40 = 20.46 (1.329x).
THOR_S_PER_STEP = 27.18
A40_S_PER_STEP = 20.46
#: ⛔ Thor's 20 SMs SATURATE AT BATCH 8: throughput is FLAT at 12.3-14.1
#: windows/s across a 6x batch range, so a bigger batch buys nothing and only
#: costs memory. This inverts the A40 instinct (16), which is why it is a
#: constant with a citation and not a default someone can drift.
THOR_BATCH = 8
A40_BATCH = 16
#: ⚠️ The "each dataloader worker costs ~8.6 GB host RAM" rule does NOT bind
#: this trainer: `train_v6_staged.train()` collates SYNCHRONOUSLY in the main
#: process (`default_collate([ds_train[i] for i in idx])`) and never constructs
#: a DataLoader. There are zero workers to tune. The host-RAM knob here is
#: `--v2-lru`, and on Thor host RAM IS device memory (unified), so it competes
#: with the model rather than sitting beside it.
THOR_V2_LRU = 6
#: ⛔ The ONLY admissible memory probe on Thor. mem_get_info / free / tegrastats
#: / VmRSS all misreport on unified memory, in BOTH directions (MEASURED
#: 2026-08-03). The trainer logs `cuda_max_mem_gb` from this call per log row.
THOR_MEMORY_PROBE = "torch.cuda.max_memory_allocated()"


class ChainRefusal(SystemExit):
    """The chain refuses to launch a step. Its own class so a caller can tell a
    ladder refusal from a subprocess failure."""


# ============================================================================
# ⛔ SEL-1's ADMISSION — the pre-registration, in code, as data
# ============================================================================
#: E-WC2 RAN 2026-08-16 and **FIRED REFUSED**. `V6F_PLANNER_DESIGN.md` §5.2 had
#: pre-registered both outcomes in advance — *σ/ADE ≤ 1.7 ⇒ SEL-1 is funded and
#: S-T launches with it; **σ/ADE ≥ 3.0 ⇒ SEL-1 is REFUSED before launch** and
#: the work moves to `ANCHOR_GOAL` supervision.* It is not close: the CI's LOWER
#: bound is still 2.48x the refusal threshold.
#:
#: ⭐ AND THE REASON MATTERS MORE THAN THE VERDICT. A **0-parameter
#: constant-yaw-rate** goal reaches σ(2 s) = **1.1888 m — 3.96x BETTER** than a
#: ridge fitted on frozen REF-C latents. So this is NOT "a 6 s goal point is
#: unpredictable"; it is "**these latents are the wrong surface**". SEL-1's
#: ESTIMAND survives — a candidate-independent reference still has no
#: degenerate minimiser — but its INPUT does not. ⚠️ And the other half must be
#: said with it: even that kinematic floor lands at σ/ADE **2.52**, which is
#: still NOT FUNDED. Neither surface is good enough today.
#:
#: ⚠️ SCOPE, stated so it cannot be overstated: this is the **REF-C** surface,
#: not the frozen S-W latents §5.2 nominally names — those have NEVER been
#: dumped. The RATIOS transfer (REF-C is the arm the 1.7/3.0 thresholds were
#: derived on, and the denominators are §3.1's own published fan references);
#: the absolute metres are REF-C's. Tier **T0-DIAGNOSTIC** — no T1 claim may
#: cite it.
SEL1_ADMISSION: dict = {
    "verdict": "REFUSED",
    "fired": "2026-08-16",
    "pre_registered_in": "Project Steering/V6F_PLANNER_DESIGN.md §5.2",
    "surface": ("REF-C-XL `refc-xl-30k` step 29999, 881 windows / 40 episodes; "
                "point estimates full_set, intervals episode-cluster bootstrap "
                "over the 40 val episodes, 2000 draws; overlapping_holdout_se "
                "used nowhere"),
    "tier": "T0-DIAGNOSTIC",
    "sigma_2s_m": 4.7104, "sigma_2s_ci": (3.8087, 5.6860),
    "sigma_6s_m": 18.3519, "sigma_6s_ci": (15.8621, 20.9608), "n_6s": 681,
    "sigma_over_ade": 9.9915, "sigma_over_ade_ci": (7.4492, 13.5119),
    "sigma_over_oracle": 28.7307,
    "incumbent_selected_ade_m": 0.4714, "fan_oracle_m": 0.1639,
    "funded_at_sigma_over_ade": 1.7, "refused_at_sigma_over_ade": 3.0,
    "replication": {"REF-C-base sigma_over_ade": 9.6337},
    "kinematic_floor": {
        "rule": "0-parameter constant-yaw-rate goal",
        "sigma_2s_m": 1.1888, "better_than_the_ridge_by": 3.96,
        "sigma_over_ade": 2.52,
        "_read": "BETTER than the learned ridge and STILL NOT FUNDED. It is "
                 "what makes 'the latents are the wrong surface' the reading, "
                 "rather than 'the 6 s goal is unpredictable'."},
    # ⛔ §5.3's own refutation row fired too: "a σ* re-measured at 6 s exceeds
    # 3x the 2 s value" ⇒ the ratio form does NOT transfer and the threshold
    # must be RE-DERIVED, not scaled. So there is no 6 s threshold to publish.
    "sigma_6s_over_2s": 3.7481,
    "ratio_form_transfers": False,
    "threshold_6s_m": None,
    "threshold_6s_reason": ("σ(6 s)/σ(2 s) = 3.7481 > 3 ⇒ §5.3's REDERIVE row "
                            "fires. Scaling the 2 s threshold across a 3x "
                            "horizon is exactly the ≤2x extrapolation rule's "
                            "prohibition; `threshold_6s` stays null until it "
                            "is re-derived at 6 s."),
    "committed_next": ("the pre-registered fallback: ANCHOR_GOAL supervision "
                       "(PH0 + obstacle.offline agent slots), NOT another "
                       "cost. A goal-distance selector whose goal is "
                       "unsupervised is being asked to invent its own "
                       "reference."),
}

#: ⭐ THE ONE INPUT THAT IS STILL MISSING, AND ITS THRESHOLDS — PRE-REGISTERED
#: HERE, IN CODE, BEFORE THE MEASUREMENT IS TAKEN. §5.2 names the FROZEN S-W
#: latents, and those have never been dumped; the S-W -> S-T boundary is the
#: natural (and only) cheap pause at which to dump them (~10-25 GPU-min; the
#: instrument, the endpoint backfill and the val40 poses are all in place).
#:
#: ⛔ These numbers are committed BEFORE the pause is spent, so the decision
#: cannot be made after seeing the number:
#:   σ(2 s) ≤ 0.80 m -> **FUNDED**   (5.89x better than REF-C's ridge, and
#:                                    1.49x better than the 0-param kinematic
#:                                    floor — from VISION ALONE)
#:   0.80 < σ ≤ 1.41 -> **INCONCLUSIVE** (leaves REFUSED; the capacity control
#:                                    is the first re-run, not a weight sweep)
#:   σ > 1.41 m      -> **REFUSED stands**
#: ⚠️ 2 s only. There is no 6 s threshold — see ``threshold_6s_reason``.
SW_LATENT_ADMISSION: dict = {
    "probe": "E-WC2-SW — ridge on FROZEN S-W latents, LOEO over the 40 val "
             "episodes, 1σ per-axis endpoint error at 2 s",
    "artifact": "ewc2_sw_latents.json",
    #: ⛔ CORRECTED 2026-08-17 (E4 stream) — this read `"sigma_2s_m"`, a key the
    #: instrument DOES NOT EMIT, at a level it does not emit it. See
    #: :data:`SW_SIGMA_LOCATIONS`. The reopening path could not return FUNDED.
    "field": "references_and_ratios.sigma_perax_2s_m",
    "funded_at_or_below_m": 0.80,
    "refused_above_m": 1.41,
    "horizon_s": 2.0,
    "cost": "~10-25 GPU-min at the S-W -> S-T boundary",
    "pre_registered": "2026-08-16, BEFORE the dump was taken",
    "_read": "σ ≤ 0.80 FUNDED · 0.80 < σ ≤ 1.41 INCONCLUSIVE (REFUSED stands) "
             "· σ > 1.41 REFUSED. No 6 s threshold is emitted: σ(6 s)/σ(2 s) "
             "= 3.7481 on REF-C, past the 3x line, so the ratio form does not "
             "transfer and a scaled 6 s number would be a fabricated one.",
}


#: ⛔ WHERE THE σ ACTUALLY LIVES IN THE INSTRUMENT'S OWN ARTIFACT, as data.
#:
#: **MEASURED 2026-08-17 BY EXECUTION (E4 stream), and this was a DEAD GATE.**
#: ``SW_LATENT_ADMISSION["field"]`` read ``"sigma_2s_m"`` and
#: :func:`read_sw_admission` looked for it at the TOP LEVEL of the JSON.
#: ``scripts/e_wc2_sigma_star.py`` emits neither: its result's top level is
#: ``{item, _prereg, _evidence_class, _class, _tier, _estimator, _ridge,
#: surface, features, sigma, references_and_ratios, rederive_check_5_3,
#: decision, four_families}``, and the per-axis 2 s σ is at
#: ``references_and_ratios.sigma_perax_2s_m``. **Name AND level both differed.**
#:
#: Executed end-to-end on a synthetic dump with a PLANTED σ of **0.30 m** — deep
#: inside the FUNDED band — through the real estimator and the real reader:
#: the estimator recovered **0.3026** (0.9 % of planted) and
#: ``read_sw_admission`` still returned ``verdict: null`` (*"the probe did not
#: report the quantity the threshold is defined on"*), so
#: :func:`assert_selector_admissible` **REFUSED the launch**.
#:
#: ⇒ **SEL-1's pre-registered reopening path could not return FUNDED for ANY
#: measurement.** It is the exact mirror of E4 — E4 is a gate that cannot read
#: PASS; this is an admission gate that cannot read FUNDED — and together they
#: made "eventually we need a tactical selector" unreachable by construction on
#: both sides of the same question. *A gate that cannot report PASS and a gate
#: that cannot report FAIL are the same defect.*
#:
#: ⚠️ **ROOT-CAUSE CLASS: A FIXTURE THAT MODELS THE CONSUMER'S EXPECTATION
#: INSTEAD OF THE PRODUCER'S OUTPUT.** ``test_v6_chain.py`` wrote the artifact by
#: hand as ``{"sigma_2s_m": sigma}`` — the shape the READER wanted — so the join
#: between instrument and reader was never exercised. Same family as
#: ``touch_ancestor`` writing a ``stage_gate.json`` with no ``config.json``
#: (`ST_LAUNCH_FIXES.md` §9), and as ``assert_geometry_carry`` comparing 2 of
#: the 76 fields it had in hand. The replacement pin RUNS THE REAL ESTIMATOR and
#: resolves its real output.
#:
#: Searched in order; the FIRST that resolves to a float wins. The legacy
#: top-level name is kept LAST so a hand-written artifact still reads.
SW_SIGMA_LOCATIONS: tuple[tuple[str, ...], ...] = (
    ("references_and_ratios", "sigma_perax_2s_m"),
    ("references_and_ratios", "sigma_perax_2s_matched_m"),
    ("sigma_2s_m",),
)

#: ⛔ THE UNIT TRAP, REFUSED BY NAME. ``sigma_radial_rms_m`` is **√2 × larger**
#: than the per-axis σ the threshold is defined on (``e_wc2_sigma_star.py``
#: states this and pins the unit: 0.8 / 0.4714 = 1.70 reproduces §3.1's
#: published ratio exactly). Reading the radial form against 0.80 m would
#: **inflate σ by 1.414 and flip FUNDED → INCONCLUSIVE on arithmetic alone** —
#: a true 0.56 m surface would read 0.79 and squeak in, a true 0.79 would read
#: 1.12 and be refused. So the resolver refuses these keys explicitly rather
#: than never happening to look at them.
SW_SIGMA_FORBIDDEN: dict[str, str] = {
    "sigma_radial_rms_m": "the RADIAL RMS, sqrt(2)x the per-axis sigma the "
                          "0.80/1.41 thresholds are defined on",
    "sigma_6s_m": "the 6 s horizon. There is NO 6 s threshold: "
                  "sigma(6s)/sigma(2s) = 3.7481 on REF-C is past the 3x line, "
                  "so the ratio form does not transfer (SEL1_ADMISSION"
                  "['threshold_6s_reason']) and a scaled 6 s number would be "
                  "fabricated.",
}


def resolve_sw_sigma(d: dict) -> dict:
    """Find the PER-AXIS 2 s σ in an E-WC2 artifact, or say exactly what failed.

    Returns ``{"ok": True, "sigma_2s_m": float, "at": "a.b.c"}`` or
    ``{"ok": False, "searched": [...], "reason": ...}``. ⛔ It never guesses and
    it never falls back to a differently-scaled quantity — see
    :data:`SW_SIGMA_FORBIDDEN`. An admission gate that reads the wrong unit is
    worse than one that reads nothing, because it looks like an answer.
    """
    searched = [".".join(p) for p in SW_SIGMA_LOCATIONS]
    for path in SW_SIGMA_LOCATIONS:
        node: object = d
        for k in path:
            if not isinstance(node, dict) or k not in node:
                node = None
                break
            node = node[k]
        if isinstance(node, (int, float)) and not isinstance(node, bool):
            return {"ok": True, "sigma_2s_m": float(node),
                    "at": ".".join(path),
                    "unit": "PER-AXIS 1-sigma, metres (NOT the radial RMS, "
                            "which is sqrt(2)x larger)"}
    present = sorted(k for k in SW_SIGMA_FORBIDDEN
                     if k in d or k in (d.get("references_and_ratios") or {}))
    return {"ok": False, "searched": searched,
            "top_level_keys": sorted(d) if isinstance(d, dict) else None,
            "refused_alternatives": {k: SW_SIGMA_FORBIDDEN[k] for k in present},
            "reason": (
                f"none of {searched} resolved to a number. The producer is "
                f"`scripts/e_wc2_sigma_star.py --dump <d> --out <this file>`, "
                f"which writes the per-axis 2 s sigma at "
                f"`references_and_ratios.sigma_perax_2s_m`. "
                + (f"⛔ {sorted(present)} IS present and was REFUSED: "
                   f"{[SW_SIGMA_FORBIDDEN[k] for k in present]}. "
                   if present else "")
                + "Absence of the number is NOT an admission.")}


def adjudicate_sw_admission(sigma_2s_m: float) -> dict:
    """Apply :data:`SW_LATENT_ADMISSION`'s PRE-REGISTERED thresholds."""
    s = float(sigma_2s_m)
    if s <= SW_LATENT_ADMISSION["funded_at_or_below_m"]:
        v = "FUNDED"
    elif s <= SW_LATENT_ADMISSION["refused_above_m"]:
        v = "INCONCLUSIVE"
    else:
        v = "REFUSED"
    return {"sigma_2s_m": s, "verdict": v,
            "thresholds": {k: SW_LATENT_ADMISSION[k] for k in
                           ("funded_at_or_below_m", "refused_above_m")},
            "pre_registered": SW_LATENT_ADMISSION["pre_registered"],
            "admits_a_selector_arm": v == "FUNDED"}


def admission_path(cfg: "ChainConfig") -> str:
    return posixpath.join(cfg.path(cfg.sw_dir),
                          SW_LATENT_ADMISSION["artifact"])


def read_sw_admission(cfg: "ChainConfig") -> dict:
    """Read the S-W latent-surface probe, or say precisely what is missing."""
    p = Path(admission_path(cfg))
    if not p.exists():
        return {"present": False, "path": str(p),
                "verdict": None,
                "_read": "the E-WC2-SW dump has not been taken. Absence of the "
                         "probe is NOT an admission."}
    try:
        d = json.loads(p.read_text())
    except Exception as e:
        return {"present": True, "path": str(p), "verdict": None,
                "error": f"{type(e).__name__}: {e}"}
    # ⛔ The σ is resolved through :func:`resolve_sw_sigma`, which knows where
    # the INSTRUMENT actually writes it. This used to be a bare `field not in d`
    # against a key `e_wc2_sigma_star.py` never emitted — see
    # :data:`SW_SIGMA_LOCATIONS` for the executed proof that the gate was dead.
    hit = resolve_sw_sigma(d)
    if not hit["ok"]:
        return {"present": True, "path": str(p), "verdict": None,
                "resolve": hit,
                "_read": f"{p} carries no per-axis 2 s sigma. {hit['reason']}"}
    return {"present": True, "path": str(p),
            **adjudicate_sw_admission(hit["sigma_2s_m"]),
            "read_at": hit["at"], "unit": hit["unit"], "raw": d}


def sw_admission_recipe(cfg: "ChainConfig") -> dict:
    """⭐ THE EMITTED PATH TO A TACTICAL SELECTOR — the PI's "eventually", as a
    runnable sequence with its one missing piece NAMED rather than assumed.

    ⛔ **This exists because the reopening path lived only in a docstring.**
    ``assert_selector_admissible`` tells an operator the arm is refused and
    names an artifact; nothing anywhere emitted the commands that produce it,
    and the one step that does not exist yet was recorded nowhere at all. That
    is the "please merge in a README" failure — a request nobody re-reads —
    applied to the programme's own next decision.

    ⭐ **STEP 1 IS BUILT (2026-08-18) — ``scripts/v6_dump_sw_latents.py``.** It
    was emitted as ``⛔ NOT BUILT`` from 2026-08-17, when three probes MEASURED
    that no script dumped v6 S-W latents in ``e_wc2_sigma_star.DUMP_CONTRACT``
    (``refc_dump_latents.build_model`` builds a ``RefCModel`` and cannot load a
    v6 checkpoint; ``probe_latent_state.py`` reads v6 checkpoints but emits
    P1/P2 retention). All four steps are now runnable commands, and the whole
    path — producer → estimator → this reader → :func:`assert_selector_
    admissible` — is pinned end-to-end on a PLANTED σ by
    ``tests/test_v6_dump_sw_latents.py``, never by a hand-written dump.

    ⚠️ **The measurement itself has still NOT been taken.** The commands exist;
    the S-W → S-T boundary is when they are cheap. Absence of the probe is not
    an admission, and ``current`` below says which it is.
    """
    sw = cfg.path(cfg.sw_dir)
    art = admission_path(cfg)
    dump = posixpath.join(sw, "ewc2_sw_dump.pt")
    return {
        "why": "SEL-1 is REFUSED; this is the ONLY thing that lifts it, and it "
               "was PRE-REGISTERED on 2026-08-16 BEFORE the measurement "
               "existed. It is not a veto on a tactical selector — it is the "
               "measurement that decides whether this surface can carry one.",
        "when": "the S-W -> S-T boundary. That is the only cheap pause in the "
                "ladder and it is the moment S-W reaches 30 000.",
        "cost": SW_LATENT_ADMISSION["cost"],
        "steps": [
            {"n": 1, "status": "✅ built 2026-08-18 — THE GPU STEP",
             "what": "dump the FROZEN S-W latents on the canonical val40 grid "
                     "in the E-WC2 dump contract",
             "cmd": (f"PYTHONPATH={cfg.pythonpath} OMP_NUM_THREADS=6 "
                     f"{cfg.python} scripts/v6_dump_sw_latents.py "
                     f"--ckpt {shlex.quote(posixpath.join(sw, 'ckpt.pt'))} "
                     f"--v2-val-cache {shlex.quote(cfg.val_cache)} "
                     f"--v2-lru {cfg.v2_lru} --batch {cfg.batch} "
                     f"--out {shlex.quote(dump)}"),
             "contract": "scripts/e_wc2_sigma_star.py --print-contract "
                         "(requires: eid [n], >=1 VISION-ONLY feature block "
                         "[n, F], gt_endpoint [n, He, 2] with axis0 = "
                         "LONGITUDINAL, endpoint_valid) — AND `fan`/`gt`/"
                         "`wp_steps`, because the estimator writes "
                         f"{SW_LATENT_ADMISSION['field']} only inside its "
                         "fan-references branch, so a latent-ONLY dump leaves "
                         "this gate with nothing to read.",
             "grid": "40 episodes, WINDOW=8, STRIDE=8, 881 windows. "
                     "Re-selecting episodes breaks parity and is refused. The "
                     "grid stays at WINDOW=8 and the model is fed the LAST "
                     "predictor.window frames — same last frame, same rows.",
             "why_not_reusable": "refc_dump_latents.build_model builds a "
                                 "RefCModel and cannot load a v6 checkpoint; "
                                 "probe_latent_state.py reads v6 checkpoints "
                                 "but emits P1/P2 retention, not an E-WC2 "
                                 "dump. MEASURED at three probes 2026-08-17 — "
                                 "which is why v6_dump_sw_latents.py exists "
                                 "rather than a flag on either of them.",
             "⚠️ on a --selector none arm": "the dump carries NO `sel` (there "
                 "is no incumbent selection to record, and fabricating one "
                 "would manufacture the σ/ADE denominator). e_wc2_sigma_star "
                 "then emits NO_VERDICT for its OWN §5.2 ratio test and still "
                 "writes the per-axis σ this gate adjudicates in ABSOLUTE "
                 "metres. Both are expected; neither is a failed dump.",
             "input": posixpath.join(sw, "ckpt.pt"),
             "output": dump},
            {"n": 2, "status": "✅ built — CPU, no GPU, no model",
             "what": "run the estimator (ridge = the P1/P2 battery's, LOEO "
                     "over the 40 val episodes)",
             "cmd": (f"{cfg.python} scripts/e_wc2_sigma_star.py "
                     f"--dump {shlex.quote(dump)} "
                     f"--out {shlex.quote(art)} --features pooled,ctx")},
            {"n": 3, "status": "✅ built",
             "what": "adjudicate against the PRE-REGISTERED thresholds",
             "cmd": f"{cfg.python} scripts/v6_chain.py admission "
                    f"--root {shlex.quote(cfg.root)} "
                    f"--sw-dir {shlex.quote(cfg.sw_dir)}",
             "reads": f"{SW_LATENT_ADMISSION['field']} (PER-AXIS metres)",
             "thresholds": {
                 "FUNDED": f"sigma <= {SW_LATENT_ADMISSION['funded_at_or_below_m']} m",
                 "INCONCLUSIVE": f"{SW_LATENT_ADMISSION['funded_at_or_below_m']} "
                                 f"< sigma <= {SW_LATENT_ADMISSION['refused_above_m']} m "
                                 f"(REFUSED stands)",
                 "REFUSED": f"sigma > {SW_LATENT_ADMISSION['refused_above_m']} m"}},
            {"n": 4, "status": "✅ built — only reachable if step 3 says FUNDED",
             "what": "schedule the selector arm; sel_gap then BINDS at the "
                     "S-T gate (train_v6_staged.GATE_APPLICABILITY)",
             "cmd": f"{cfg.python} scripts/v6_chain.py commands --step S-T:goal "
                    f"--st-arms goal --root {shlex.quote(cfg.root)}"},
        ],
        "current": read_sw_admission(cfg),
        "what_binds_if_funded": (
            "GATE_APPLICABILITY['S-T']['sel_gap'] = 'has_scorer'. On a FUNDED "
            "arm the stack has a cand_score, so sel_gap is REQUIRED at the S-T "
            "gate and sel_gap_revalidated is REQUIRED at S-S. On the refused "
            "arm both are NOT APPLICABLE and the certificate says so, with "
            "this recipe as the way to change it."),
        "_evidence_class": "PRE-REGISTERED (the thresholds, 2026-08-16) + "
                           "MEASURED (ours, 2026-08-17: the step-1 absence at "
                           "three probes; the reader/instrument key mismatch "
                           "executed end-to-end)",
    }


def assert_selector_admissible(step, cfg: "ChainConfig") -> dict:
    """⛔ No stage may build a scorer while SEL-1 stands REFUSED.

    The chain does not merely decline to schedule a selector arm — it refuses to
    LAUNCH one, because "we did not put it in the default plan" is a preference
    and this is a fired pre-registration. The only thing that lifts it is the
    S-W latent measurement reaching the threshold committed in
    :data:`SW_LATENT_ADMISSION` before that measurement existed.
    """
    if step.selector in ("none", "?"):
        return {"applies": False, "selector": step.selector}
    adm = read_sw_admission(cfg)
    if adm.get("verdict") == "FUNDED":
        return {"applies": True, "ok": True, "admission": adm}
    raise ChainRefusal(
        f"[chain] ⛔ {step.key} would launch with --selector {step.selector!r}, "
        f"and SEL-1 is REFUSED.\n"
        f"  E-WC2 fired {SEL1_ADMISSION['fired']} against a threshold "
        f"pre-registered in {SEL1_ADMISSION['pre_registered_in']} with BOTH "
        f"outcomes committed in advance: σ/ADE ≥ "
        f"{SEL1_ADMISSION['refused_at_sigma_over_ade']} ⇒ refuse before "
        f"launch. MEASURED σ/ADE = {SEL1_ADMISSION['sigma_over_ade']} "
        f"{SEL1_ADMISSION['sigma_over_ade_ci']} — the interval's LOWER bound "
        f"is still 2.48x the refusal threshold, and REF-C-base agrees "
        f"({SEL1_ADMISSION['replication']['REF-C-base sigma_over_ade']}).\n"
        f"  ⭐ The reading is NOT 'a 6 s goal is unpredictable': a "
        f"0-parameter constant-yaw-rate goal reaches σ(2 s) = "
        f"{SEL1_ADMISSION['kinematic_floor']['sigma_2s_m']} m, "
        f"{SEL1_ADMISSION['kinematic_floor']['better_than_the_ridge_by']}x "
        f"BETTER than the ridge. THESE LATENTS ARE THE WRONG SURFACE. The "
        f"estimand survives; its input does not. ⚠️ And that floor is itself "
        f"only σ/ADE {SEL1_ADMISSION['kinematic_floor']['sigma_over_ade']} — "
        f"not funded either.\n"
        f"  ⚠️ SCOPE: measured on the REF-C surface (T0-DIAGNOSTIC), not on "
        f"frozen S-W latents, which have never been dumped. The RATIOS "
        f"transfer; the metres are REF-C's.\n"
        f"  ⇒ The committed next step is {SEL1_ADMISSION['committed_next']}\n"
        f"  ⇒ To reopen a selector arm, take the E-WC2-SW dump at the S-W→S-T "
        f"boundary ({SW_LATENT_ADMISSION['cost']}) and write "
        f"{admission_path(cfg)} with {SW_LATENT_ADMISSION['field']!r}. "
        f"PRE-REGISTERED {SW_LATENT_ADMISSION['pre_registered']}: "
        f"σ ≤ {SW_LATENT_ADMISSION['funded_at_or_below_m']} m FUNDED · "
        f"σ > {SW_LATENT_ADMISSION['refused_above_m']} m REFUSED stands. "
        f"Current state: {adm.get('_read', adm.get('verdict'))}\n"
        f"  ⭐ THE RECIPE IS EMITTED, not described: `v6_chain.py admission` "
        f"prints all four steps (`recipe`), and since 2026-08-18 ALL FOUR are "
        f"runnable commands — step 1, the v6 S-W latent dump in E-WC2's "
        f"contract, is `scripts/v6_dump_sw_latents.py`. ⚠️ Until "
        f"then the S-T certificate carries `sel_gap` as NOT APPLICABLE with "
        f"this same path attached (train_v6_staged.UNMEASURED_BY_CONSTRUCTION), "
        f"so the tactical-selection question is recorded as OPEN, never as "
        f"answered.")


# ============================================================================
# the plan, as data
# ============================================================================

@dataclass(frozen=True)
class ChainStep:
    """One launch. ``key`` is what an operator types; ``out`` is unique to it."""
    key: str                       # "S-W" | "S-T:goal" | "S-T:mlp" | "S-S" ...
    stage: str                     # the trainer's --stage
    arm: str | None                # the selector arm this step IS, if any
    out: str                       # ⛔ unique per step, always
    steps: int
    lr: float
    selector: str                  # geometry carried forward, not just an arm
    w_select: float
    init_from_key: str | None      # which step's ckpt seeds this one
    prev_gate_key: str | None      # which step's gate certifies it
    max_horizon: int | None
    pair_with: tuple[str, ...] = ()
    extra: tuple[str, ...] = ()
    note: str = ""
    #: ⭐ the g_str->P_T port (F-1). GEOMETRY, exactly like ``selector``: once
    #: S-T trains with it, every consumer of that lineage must build it or
    #: `load_stage_init` dies on unexpected `cond_tac_dyn.*` keys.
    tac_goal_cond: bool = False
    #: ⛔ Set when this step consumes an S-T checkpoint and no ``--st-winner``
    #: was declared. The ladder is NOT a straight line at S-T — it forks into
    #: the arm pair — so "which arm does S-S continue" is a DECISION, and a
    #: chain that picks one silently has made the PI's decision for them.
    needs_st_winner: bool = False

    @property
    def ckpt(self) -> str:
        return posixpath.join(self.out, "ckpt.pt")

    @property
    def gate(self) -> str:
        return posixpath.join(self.out, "stage_gate.json")

    @property
    def done_marker(self) -> str:
        return posixpath.join(self.out, "summary.json")

    @property
    def run_id(self) -> str:
        return posixpath.basename(self.out)


@dataclass
class ChainConfig:
    """Everything the ladder needs that is not a per-step fact."""
    root: str = "/workspace/experiments"
    sw_dir: str = "v6F-SW-30k"            # the LIVE run; the chain never writes here
    train_cache: str = ("/workspace/data/"
                        "physicalai-train-e438721ae894-w120-256x640cyl")
    val_cache: str = ("/workspace/data/"
                      "physicalai-val-0c5f7dac3b11-w120-256x640cyl")
    workdir: str = "/workspace/TanitAD/stack"
    python: str = "python3"
    batch: int = THOR_BATCH
    v2_lru: int = THOR_V2_LRU
    lr: float = 1e-4
    sj_lr: float = 3e-5                   # V6F_PLANNER_DESIGN §4.4
    sw_steps: int = 30_000
    st_steps: int = 10_000                # §4.1
    ss_steps: int = 8_000                 # §4.2
    sj_steps: int = 3_000                 # §4.4
    #: ⛔ EMPTY BY DEFAULT. SEL-1 is REFUSED (:data:`SEL1_ADMISSION`), so the
    #: default ladder runs ONE S-T with `--selector none`. Opting in requires
    #: BOTH `--st-arms` and an S-W latent surface that reaches the
    #: pre-registered threshold; `assert_selector_admissible` enforces the
    #: second, so "not in the default plan" is never the only barrier.
    st_arms: tuple[str, ...] = ()
    st_winner: str | None = None          # declared, never guessed
    #: ⛔ ON BY DEFAULT — F-1 (DIAGRAM_CONFORMANCE.md 2026-08-16, the top 🟥):
    #: the binding diagram, HIERARCHY_VOCABULARY §5 and V6Stack's own docstring
    #: all spec `P_T(z_tac, a_tac | g_str)`, and an S-T launched without the
    #: port would never train the strategic->tactical DYNAMICS downlink in its
    #: own stage — the `intent_proj` defect one level up. The port is
    #: ZERO-INIT (+33,024 params, MEASURED at the production d_goal_embed=128)
    #: and S-T's `STAGE_MAY_INTRODUCE` admits its fresh keys, so turning it on
    #: over the S-W checkpoint is loss-continuous. `--no-tac-goal-cond`
    #: reproduces the pre-F-1 ladder as a declared decision, never a default.
    tac_goal_cond: bool = True
    w_select: float = 1.0
    n_candidates: int = 8
    selector_tau_m: float = 1.0
    selector_mlp_hidden: int = 256
    plan_wta_eps: float = 0.05
    s_per_step: float = THOR_S_PER_STEP
    tiny: bool = False                    # the CPU dry ladder's geometry
    dry: bool = False
    dry_steps: int = 2
    extra_common: tuple[str, ...] = ()
    #: ⭐ E1's escape hatch, and the ONLY one. A banked copy of the ancestor's
    #: own `config.json` (or its directory), used when the real one is not
    #: readable from the box generating the command — the normal case, since
    #: commands are written on the dev box for a pod path. ⛔ It is never a
    #: place to TYPE a geometry: the fields are read out of a run's record.
    geometry_from: str | None = None
    #: ⭐ E5 — bank the 60-step plan at every save boundary (F-16's X2_seam
    #: instrument, ZERO extra GPU: `L["out"]["plan"]` is already computed for
    #: the step's loss). ⚠️ NOT emitted for S-W: the emission head is at its
    #: zero-init there, so the plan is all-zero and `seam_dump_from_plan`
    #: correctly refuses it as DEGENERATE. A degenerate dump cannot answer the
    #: seam question and must not be produced to make the row non-empty.
    dump_seam_plan: bool = True
    #: the live S-W run's cadence and the validated S-T line's; the trainer's
    #: own default is 1000, and the seam dump fires at this boundary.
    save_every: int = 250

    def path(self, *parts: str) -> str:
        return posixpath.join(self.root.replace("\\", "/"), *parts)

    @property
    def pythonpath(self) -> str:
        """⛔ BOTH ROOTS. `taniteval` is a SIBLING of `stack/`, not a member of
        it, so `PYTHONPATH=<workdir>` alone makes `import taniteval.seam_dump`
        a `ModuleNotFoundError` — MEASURED 2026-08-17 on Thor AND the dev box.
        The trainer caught that broadly and printed *"training continues"*, so
        `--dump-seam-plan` banked NOTHING, silently, at every save boundary,
        with the first attempt ~1.8 h in. Derived from `workdir` rather than
        configured, because the two directories are siblings in the repo by
        construction and a second knob is a second thing to forget."""
        parent = posixpath.dirname(self.workdir.replace("\\", "/").rstrip("/"))
        return f"{self.workdir}:{posixpath.join(parent, 'taniteval')}"


#: The tiny geometry the CPU dry ladder builds — the SAME wiring and the SAME
#: seams as production, ~1000x fewer parameters (the fixture `tests/
#: test_v6_ladder_edges.py` uses, so a dry-ladder failure is reproducible from
#: the suite).
TINY_GEOMETRY: tuple[str, ...] = (
    "--in-channels", "3", "--frame-h", "32", "--frame-w", "32", "--patch", "16",
    "--enc-dim", "32", "--enc-depth", "1", "--enc-heads", "2",
    "--readout-grid", "4", "--readout-dim", "8",
    "--pred-dim", "32", "--pred-depth", "1", "--pred-heads", "2",
    "--window", "4", "--horizons", "1", "2",
    "--d-tac", "32", "--d-str", "16", "--d-goal-embed", "16",
    "--adapter-hidden", "32", "--sigreg-slices", "8",
)

#: ⛔ The reason stamped on every dry-ladder gate override. It is a CONSTANT so
#: it reads identically in every artifact and a real launch can never be
#: mistaken for one — and `assert_stage_precondition(dry_run=False)` refuses a
#: `_dry_run` gate anyway, so this is the second lock, not the only one.
DRY_OFF_REASON = ("CPU DRY LADDER — no corpus, no battery, synthetic tensors. "
                  "Every dry gate is INCONCLUSIVE by construction because no "
                  "frozen-battery probe ran; advancing through the RECORDED "
                  "override is the honest path and is what makes this an "
                  "end-to-end execution of the real gate machinery.")


def assert_no_tilde(cfg: ChainConfig) -> dict:
    """⛔ A `~` in any path makes the emitted launch line silently WRONG.

    MEASURED 2026-08-16 while generating the Thor commands: every path in a
    launch line is `shlex.quote`d (correctly — `--gate-off-reason 'PI directive:
    …'` is exactly why), and quoting SUPPRESSES tilde expansion. `mkdir -p
    '~/experiments/v6F-ST-10k'` creates a DIRECTORY LITERALLY NAMED `~`, the
    trainer writes into it, and `--init-from '~/experiments/v6F-SW-30k/ckpt.pt'`
    then does not exist — a launch that fails in a way that looks like a missing
    checkpoint.

    ⛔ And the chain must NOT expand it: `~` on the dev box that generates the
    command is not `~` on the pod that runs it. Refuse and make the operator
    paste what `echo $HOME` prints ON THE POD.
    """
    bad = {name: v for name, v in
           (("--root", cfg.root), ("--workdir", cfg.workdir),
            ("--train-cache", cfg.train_cache), ("--val-cache", cfg.val_cache))
           if "~" in str(v)}
    if bad:
        raise ChainRefusal(
            f"[chain] ⛔ these paths contain '~': {bad}.\n"
            f"  Every path in an emitted launch line is shell-quoted (which it "
            f"must be — `--gate-off-reason 'PI directive: …'` is why), and "
            f"quoting SUPPRESSES tilde expansion: `mkdir -p '~/experiments/…'` "
            f"creates a directory literally named `~`, and the next stage's "
            f"--init-from then points at a file that does not exist.\n"
            f"  ⚠️ The chain will not expand it for you: `~` on the box that "
            f"GENERATES the command is not `~` on the pod that RUNS it. Paste "
            f"what `echo $HOME` prints ON THE POD (e.g. /root/experiments).")
    return {"ok": True}


def build_plan(cfg: ChainConfig) -> tuple[ChainStep, ...]:
    """Resolve the ladder. Pure — no filesystem, no torch."""
    assert_no_tilde(cfg)
    for arm in cfg.st_arms:
        if arm not in ("goal", "mlp"):
            raise ChainRefusal(
                f"[chain] ⛔ unknown S-T arm {arm!r}; the pre-registered pair is "
                f"'goal' (the mechanism, +267 params) and 'mlp' (the CAPACITY "
                f"CONTROL, +33,801 = 126.6x, information-MATCHED).")
    steps: list[ChainStep] = [ChainStep(
        key="S-W", stage="S-W", arm=None,
        out=cfg.path(cfg.sw_dir), steps=cfg.sw_steps, lr=cfg.lr,
        selector="none", w_select=0.0,
        init_from_key=None, prev_gate_key=None, max_horizon=None,
        note="THE WORLD STAGE. Live on Thor. lambda_plan == 0 by construction "
             "and the trainer REFUSES --selector here: a scorer built in S-W "
             "would be untrainable dead weight AND would change the "
             "state_dict, breaking the live run's strict resume.")]

    if not cfg.st_arms:
        # ⛔ THE DEFAULT LADDER. One S-T, `--selector none`, `--w-select 0`.
        # SEL-1 fired REFUSED on 2026-08-16 against a threshold pre-registered
        # with both outcomes committed in advance, so scheduling a selector arm
        # here would be running an experiment the programme has already
        # declined. The plan loss, the ε-relaxed WTA and the tactical layer
        # still train — what does NOT train is a scorer.
        steps.append(ChainStep(
            key="S-T", stage="S-T", arm=None,
            out=cfg.path(f"v6F-ST-{cfg.st_steps // 1000}k"),
            steps=cfg.st_steps, lr=cfg.lr, selector="none", w_select=0.0,
            init_from_key="S-W", prev_gate_key="S-W", max_horizon=60,
            tac_goal_cond=cfg.tac_goal_cond,
            extra=("--plan-wta-eps", str(cfg.plan_wta_eps), "--w-t1", "1.0"),
            note="THE PLANNER STAGE, WITHOUT A SELECTOR. ⛔ SEL-1 is REFUSED: "
                 "E-WC2 measured σ/ADE 9.9915 [7.4492, 13.5119] against a "
                 "pre-registered refusal line of 3.0 — the CI's LOWER bound is "
                 "2.48x the threshold. ⭐ The reading is that these LATENTS "
                 "are the wrong surface, not that a 6 s goal is unpredictable "
                 "(a 0-param constant-yaw-rate goal is 3.96x better and still "
                 "not funded). The committed branch is ANCHOR_GOAL "
                 "supervision. Selector arms stay reachable via --st-arms, "
                 "gated on the E-WC2-SW dump."))
        winner_key = "S-T"
    else:
        arm_keys = tuple(f"S-T:{a}" for a in cfg.st_arms)
        for arm in cfg.st_arms:
            key = f"S-T:{arm}"
            extra = ["--selector", arm,
                     "--plan-wta-eps", str(cfg.plan_wta_eps), "--w-t1", "1.0"]
            if arm == "goal":
                extra += ["--selector-tau-m", str(cfg.selector_tau_m)]
            else:
                extra += ["--selector-mlp-hidden",
                          str(cfg.selector_mlp_hidden)]
            steps.append(ChainStep(
                key=key, stage="S-T", arm=arm,
                out=cfg.path(f"v6F-ST-{arm}-{cfg.st_steps // 1000}k"),
                steps=cfg.st_steps, lr=cfg.lr, selector=arm,
                w_select=cfg.w_select,
                init_from_key="S-W", prev_gate_key="S-W", max_horizon=60,
                tac_goal_cond=cfg.tac_goal_cond,
                pair_with=tuple(k for k in arm_keys if k != key),
                extra=tuple(extra),
                note=("⚠️ OPT-IN ARM while SEL-1 stands REFUSED. MECHANISM "
                      "arm: a candidate-INDEPENDENT reference, +267 params"
                      if arm == "goal" else
                      "⚠️ OPT-IN ARM while SEL-1 stands REFUSED. ⭐ CAPACITY "
                      "CONTROL: +33,801 params (126.6x the goal rule) on "
                      "EXACTLY the goal rule's inputs — the candidate "
                      "ENDPOINT and e_g_tac. Without it a 'goal' win is "
                      "unattributable between mechanism and capacity (C6).")))
        winner_key = f"S-T:{cfg.st_winner}" if cfg.st_winner else None
    # ⛔ The selector GEOMETRY, not the arm's loss: S-S freezes the planner but
    # its stack MUST still build the scorer the S-T checkpoint carries.
    carried = "none" if not cfg.st_arms else (cfg.st_winner or "?")
    ss_extra = ["--w-s1", "1.0"]
    if carried not in ("none", "?"):
        ss_extra += ["--selector", carried]
        if carried == "mlp":
            ss_extra += ["--selector-mlp-hidden", str(cfg.selector_mlp_hidden)]
        else:
            ss_extra += ["--selector-tau-m", str(cfg.selector_tau_m)]
    steps.append(ChainStep(
        key="S-S", stage="S-S", arm=None,
        out=cfg.path(f"v6F-SS-{cfg.ss_steps // 1000}k"),
        steps=cfg.ss_steps, lr=cfg.lr, selector=carried,
        w_select=0.0,                     # for_stage('S-S') zeroes it anyway
        init_from_key=winner_key, prev_gate_key=winner_key, max_horizon=None,
        # the port GEOMETRY is carried exactly like the selector's: S-S
        # freezes layer_tac, but its stack must still BUILD cond_tac_dyn or
        # the S-T checkpoint's keys become unexpected and --init-from is fatal.
        tac_goal_cond=cfg.tac_goal_cond,
        needs_st_winner=winner_key is None, extra=tuple(ss_extra),
        note="Trains layer_str ONLY; the planner is FROZEN and w_select is 0 "
             "(for_stage('S-S') zeroes it, so a non-zero flag would advertise "
             "a loss that is not in force). ⛔ Its gate REQUIRES "
             "sel_gap_revalidated + TACTICAL_revalidated: S-S moves e_g_str -> "
             "e_g_tac, which is the frozen selector's ONLY input, so S-T's "
             "certificate stops applying the moment S-S starts."))

    sj_extra = ["--w-t1", "1.0", "--w-s1", "1.0",
                "--plan-wta-eps", str(cfg.plan_wta_eps)]
    if carried not in ("none", "?"):
        sj_extra += ["--selector", carried]
        sj_extra += (["--selector-mlp-hidden", str(cfg.selector_mlp_hidden)]
                     if carried == "mlp" else
                     ["--selector-tau-m", str(cfg.selector_tau_m)])
    steps.append(ChainStep(
        key="S-J", stage="S-J", arm=None,
        out=cfg.path(f"v6F-SJ-{cfg.sj_steps // 1000}k"),
        steps=cfg.sj_steps, lr=cfg.sj_lr, selector=carried,
        w_select=cfg.w_select if carried not in ("none", "?") else 0.0,
        init_from_key="S-S", prev_gate_key="S-S", max_horizon=60,
        tac_goal_cond=cfg.tac_goal_cond,
        needs_st_winner=winner_key is None, extra=tuple(sj_extra),
        note="OPTIONAL. Run only if S-T/S-S plateau. Trains everything with "
             "isolation still ON; the gate is the frozen battery FLAT across "
             "the joint phase (H-COTRAIN) plus zero live forbidden X3 edges."))
    return tuple(steps)


def step_by_key(plan, key: str) -> ChainStep:
    for s in plan:
        if s.key == key:
            return s
    raise ChainRefusal(f"[chain] ⛔ no step {key!r}; the ladder is "
                       f"{[s.key for s in plan]}")


# ============================================================================
# refusal 1 — the plan itself. A shared --out must be IMPOSSIBLE, not caught.
# ============================================================================

def assert_plan(plan) -> dict:
    """⛔ No two steps may share an ``--out``, and no step may point at
    another's checkpoint as its own.

    The trainer's `--resume auto` now refuses a checkpoint written by a
    different stage — but that refusal exists because a shared directory ALREADY
    happened. A chain that relies on it has built the trap and installed a
    warning sign. This makes the trap unbuildable: the plan is refused before a
    single command is emitted.
    """
    seen: dict[str, str] = {}
    for s in plan:
        norm = posixpath.normpath(s.out)
        if norm in seen:
            raise ChainRefusal(
                f"[chain] ⛔ steps {seen[norm]!r} and {s.key!r} share --out "
                f"{s.out}.\n"
                f"  Each stage AND each arm gets its own directory. A shared "
                f"--out plus `--resume auto` is how a stage silently resumes "
                f"another stage's checkpoint: every stage saves the WHOLE "
                f"V6Stack, so the state_dict load succeeds key-for-key and the "
                f"only accidental barrier (the optimiser's param-group size) "
                f"holds ONLY because the stages happen to train different "
                f"tensor counts — S-W 240 / S-T 80 / S-S 54 / S-J 374, "
                f"MEASURED. One STAGE_GROUPS edit and it passes silently.")
        seen[norm] = s.key
    for s in plan:
        if s.init_from_key and s.init_from_key == s.key:
            raise ChainRefusal(f"[chain] ⛔ {s.key} would --init-from itself")
    return {"ok": True, "n_steps": len(plan),
            "out_dirs": {s.key: s.out for s in plan}}


# ============================================================================
# refusal 2 — the arm pair. A "goal" result with no capacity control is
# UNATTRIBUTABLE, and no amount of downstream care fixes it afterwards.
# ============================================================================

def assert_arm_pair(step, plan, *, unpaired_arm_reason: str = "") -> dict:
    """Anything that consumes an S-T checkpoint requires EVERY S-T arm's gate.

    ⭐ `GoalDistanceScorer` selects with **+267** parameters. If that arm wins,
    two stories fit — MECHANISM (a candidate-independent reference has no
    degenerate minimiser) and CAPACITY (the head was simply underpowered). A
    `"goal"`-only experiment cannot separate them, and reading capacity as
    mechanism is the **C6 confound** verbatim. `"mlp"` is +33,801 (126.6x) on
    exactly the goal rule's inputs — information-MATCHED, so it moves capacity
    and nothing else.

    ⚠️ The pair is checked HERE, at the consumer, and not merely recommended in
    a document, because by the time S-S has run the S-T GPU-days are spent and
    the comparison can no longer be made from the same trunk.
    """
    if step.stage in ("S-W", "S-T"):
        return {"applies": False,
                "reason": f"{step.stage} is not an S-T consumer"}
    arms = [s for s in plan if s.arm is not None]
    if not arms:
        return {"applies": False,
                "reason": "no selector arms in this ladder — SEL-1 is REFUSED "
                          "and the default S-T runs --selector none, so there "
                          "is no mechanism/capacity attribution to protect"}
    have = [s.key for s in arms if Path(s.gate).exists()]
    missing = [s.key for s in arms if not Path(s.gate).exists()]
    if not missing:
        return {"applies": True, "ok": True, "arms": [s.key for s in arms],
                "gates_present": have}
    if unpaired_arm_reason.strip():
        return {"applies": True, "ok": True, "override": "unpaired-arm-reason",
                "reason": unpaired_arm_reason, "arms_without_a_gate": missing,
                "_read": "⚠️ RECORDED DECISION: this ladder advanced past S-T "
                         "without the full arm pair. Any 'goal' result from it "
                         "is NOT attributable between mechanism and capacity."}
    raise ChainRefusal(
        f"[chain] ⛔ {step.key} consumes an S-T checkpoint, but these S-T arms "
        f"have no stage_gate.json yet: {missing}.\n"
        f"  ⭐ 'goal' (+267 params) and 'mlp' (+33,801, 126.6x, "
        f"information-MATCHED to the goal rule's inputs) are an ARM PAIR. A "
        f"'goal' win judged without the capacity control cannot be attributed "
        f"between MECHANISM and CAPACITY — that is the C6 confound, and this "
        f"programme has already designed a hierarchy away on one.\n"
        f"  ⚠️ It cannot be fixed later: after S-S the S-T GPU-days are spent "
        f"and the two arms can no longer be compared on the same trunk.\n"
        f"  ⇒ run the missing arm(s), or make it a RECORDED decision with "
        f"--unpaired-arm-reason '<why>'.")


# ============================================================================
# refusal 3 — the selector geometry must be carried forward
# ============================================================================

def _read_run_args(out_dir: str) -> dict:
    """The predecessor's own `config.json['args']`, or ``{}``.

    Reads JSON, never the checkpoint: it costs nothing, needs no torch, and on
    a RAM-bound pod a metadata question must not transiently allocate 3.5 GB.
    """
    p = Path(out_dir) / "config.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text()).get("args") or {}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# ⭐ THE GEOMETRY SET IS **DERIVED FROM THE TRAINER'S OWN SOURCE**, NEVER LISTED
# ---------------------------------------------------------------------------
# ⛔ WHY A HAND-LIST WOULD BE THE BUG AGAIN. The version of
# `assert_geometry_carry` this replaces read the predecessor's `config.json
# ["args"]` — which contains EVERY geometry field — and compared exactly TWO of
# them (`selector`, `tac_goal_cond`). It had the data in hand and did not look,
# so `v6_chain.py commands --step S-T` emitted a line with NO model geometry at
# all and the stack built at the trainer's defaults: 87.93 M params against a
# 336.54 M checkpoint, `RuntimeError … size mismatch for encoder.pos: [1, 640,
# 768] vs [1, 640, 384]` (MEASURED 2026-08-17).
#
# Adding the missing fields by hand fixes today's two omissions and installs
# tomorrow's — which is precisely the `LADDER_UNTRAINED_GROUPS` lesson (718855f:
# a stage list that had to be edited in lockstep with the architecture and was
# not). So the set is READ OFF `build_stack_from_args` BY AST: every argparse
# dest that function consumes IS the model geometry, by construction, and a new
# `V6Config` lever wired to a new flag joins this check the moment it is wired.

def _trainer_source() -> str:
    """The trainer's source text. Read, never imported — `import
    train_v6_staged` pulls torch, and `plan`/`commands`/`status`/`manifests`
    must keep working on a pod whose torch a `uv pip install` has broken
    (MEASURED twice on pod4)."""
    p = _HERE / "train_v6_staged.py"
    if not p.exists():
        raise ChainRefusal(
            f"[chain] ⛔ the trainer source {p} is missing, so the model "
            f"geometry cannot be derived and no launch line may be emitted. "
            f"This file must sit beside {TRAINER}.")
    return p.read_text(encoding="utf-8")


def _fn_node(src: str, name: str):
    import ast
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise ChainRefusal(
        f"[chain] ⛔ `{name}` was not found in {TRAINER}. The chain derives the "
        f"model geometry from that function's body; a rename means this guard "
        f"is silently checking nothing, which is the failure it exists to "
        f"prevent. ⇒ update `_fn_node`'s target, do not delete the check.")


@functools.lru_cache(maxsize=1)
def geometry_dests() -> frozenset[str]:
    """Every argparse dest ``build_stack_from_args`` reads = the model geometry.

    Three read shapes are collected, all of them present in the real function:

    ==============================  ============================
    ``a.enc_dim``                   plain attribute
    ``getattr(a, "mpc_topk", 2)``   the default-tolerant idiom
    ``resolve_gc(a, "enc_gc")``     ``a`` handed to a helper with the dest name
    ==============================  ============================
    """
    import ast
    fn = _fn_node(_trainer_source(), "build_stack_from_args")
    arg0 = fn.args.args[0].arg if fn.args.args else "a"
    found: set[str] = set()
    for n in ast.walk(fn):
        if (isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
                and n.value.id == arg0):
            found.add(n.attr)
        elif isinstance(n, ast.Call) and n.args:
            first = n.args[0]
            if (isinstance(first, ast.Name) and first.id == arg0
                    and len(n.args) > 1
                    and isinstance(n.args[1], ast.Constant)
                    and isinstance(n.args[1].value, str)):
                found.add(n.args[1].value)
    if not found:
        raise ChainRefusal(
            "[chain] ⛔ derived ZERO geometry dests from "
            "build_stack_from_args — the AST shapes changed and this guard is "
            "now checking nothing. An empty check that returns OK is worse "
            "than no check.")
    return frozenset(found)


@dataclass(frozen=True)
class _ArgSpec:
    """One ``add_argument`` as the trainer declares it."""
    dest: str
    opt: str                       # the long option, e.g. "--enc-dim"
    action: str = ""               # "" | "store_true" | "store_false"
    nargs: str = ""                # "" | "+" | "*" | "?"
    default: object = None
    has_default: bool = False


@functools.lru_cache(maxsize=1)
def trainer_arg_spec() -> dict[str, _ArgSpec]:
    """dest -> :class:`_ArgSpec`, as the trainer's ``build_parser`` declares it.

    The AST read is the FLOOR (it works on a pod whose torch a `uv pip install`
    has broken, where `plan`/`commands`/`status` must still run); when the real
    parser IS importable its defaults win, because ``ast.literal_eval`` cannot
    evaluate a default that is an imported NAME.

    ⚠️ MEASURED 2026-08-17, by the ladder suite failing: `--a-max`,
    `--kappa-max` and `--plan-steps` default to imported constants (`A_MAX`,
    `KAPPA_MAX`), so the AST-only spec read them as *unknown* and the geometry
    diff then compared an ancestor's real 4.0 against a successor's `None` —
    a FALSE REFUSAL on the dry ladder. Unknown-default dests are now excluded
    from the diff unless BOTH sides state a value; a guard that cannot see a
    field must say so, never invent a mismatch.
    """
    spec = dict(_ast_arg_spec())
    try:                                        # exact, when torch is present
        from train_v6_staged import build_parser
        for act in build_parser()._actions:     # noqa: SLF001
            if act.dest in spec and act.option_strings:
                spec[act.dest] = _ArgSpec(
                    dest=act.dest, opt=spec[act.dest].opt,
                    action=spec[act.dest].action, nargs=spec[act.dest].nargs,
                    default=act.default, has_default=True)
    except Exception:                                         # noqa: BLE001
        pass
    return spec


@functools.lru_cache(maxsize=1)
def _ast_arg_spec() -> dict[str, _ArgSpec]:
    """The torch-free floor: ``add_argument`` calls read straight out of source.

    ⚠️ This is a READ of argparse's declaration, not a reimplementation of
    argparse. `tests/test_v6_chain.py` pins it against the REAL
    ``build_parser()`` — every dest, option string, action and nargs — so the
    torch-free path cannot drift from the parser that actually runs.
    """
    import ast
    fn = _fn_node(_trainer_source(), "build_parser")
    out: dict[str, _ArgSpec] = {}
    for n in ast.walk(fn):
        if not (isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and n.func.attr == "add_argument"):
            continue
        opts = [x.value for x in n.args
                if isinstance(x, ast.Constant) and isinstance(x.value, str)]
        longs = [o for o in opts if o.startswith("--")]
        if not longs:
            continue
        kw = {k.arg: k.value for k in n.keywords if k.arg}
        dest = None
        if "dest" in kw and isinstance(kw["dest"], ast.Constant):
            dest = kw["dest"].value
        else:
            dest = longs[0][2:].replace("-", "_")
        action = (kw["action"].value if isinstance(kw.get("action"),
                                                   ast.Constant) else "")
        nargs = (str(kw["nargs"].value) if isinstance(kw.get("nargs"),
                                                      ast.Constant) else "")
        default, has_default = None, "default" in kw
        if has_default:
            try:
                default = ast.literal_eval(kw["default"])
            except Exception:                                 # noqa: BLE001
                has_default = False
        if not has_default:
            default = {"store_true": False, "store_false": True}.get(action)
            has_default = action in ("store_true", "store_false")
        out[dest] = _ArgSpec(dest=dest, opt=longs[0], action=action,
                             nargs=nargs, default=default,
                             has_default=has_default)
    return out


def _norm(v):
    """Compare JSON round-trips and parsed argv on equal terms."""
    if isinstance(v, (list, tuple)):
        return [_norm(x) for x in v]
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return float(v)
    return v


def _flags_declared(argv) -> set[str]:
    """The dests an argv EXPLICITLY names — a declared change, not an omission."""
    by_opt = {s.opt: s.dest for s in trainer_arg_spec().values()}
    return {by_opt[tok] for tok in argv if tok in by_opt}


def parse_argv_geometry(argv) -> dict:
    """dest -> value for every geometry dest, as the emitted ``argv`` sets it.

    Torch-free, spec-driven, and deliberately narrow: it resolves ONLY the
    geometry dests, because that is the only question this guard asks.
    """
    spec = trainer_arg_spec()
    by_opt = {s.opt: s for s in spec.values()}
    vals = {d: spec[d].default for d in geometry_dests() if d in spec}
    i = 0
    argv = list(argv)
    while i < len(argv):
        tok = argv[i]
        s = by_opt.get(tok)
        if s is None:
            i += 1
            continue
        if s.action in ("store_true", "store_false"):
            if s.dest in vals:
                vals[s.dest] = s.action == "store_true"
            i += 1
            continue
        j = i + 1
        got: list[str] = []
        while j < len(argv) and argv[j] not in by_opt:
            got.append(argv[j])
            j += 1
            if s.nargs not in ("+", "*"):
                break
        if s.dest in vals and got:
            if s.nargs in ("+", "*"):
                # ⚠️ COERCE THE ELEMENTS. `--horizons 1 2 4` parses to STRINGS
                # here while the ancestor's config.json holds INTS, so a bare
                # comparison reports a difference that does not exist — and for
                # an undeclared dest that is a FALSE REFUSAL. MEASURED
                # 2026-08-17: `horizons` showed as a changed field on a launch
                # line that carries the ancestor's own value.
                like = (s.default[0] if isinstance(s.default, (list, tuple))
                        and s.default else None)
                vals[s.dest] = [_coerce(g, like) for g in got]
            else:
                vals[s.dest] = _coerce(got[0], s.default)
        i = max(j, i + 1)
    return vals


def _coerce(text: str, like):
    if isinstance(like, bool):
        return text.lower() in ("1", "true", "yes")
    for cast in ((int, float) if not isinstance(like, str) else ()):
        try:
            return cast(text)
        except ValueError:
            continue
    if like is None:
        for cast in (int, float):
            try:
                return cast(text)
            except ValueError:
                continue
    return text


def geometry_source(step, cfg, plan) -> tuple[dict, str]:
    """``(args, where)`` — the record this step's geometry is carried FROM.

    ⛔ The predecessor's own ``config.json`` wins when it is readable: it is the
    only authority on what shapes its checkpoint actually holds. ``--geometry-
    from`` is the off-box escape (commands are usually generated on the dev box
    for a pod path that does not exist there), and it points at a BANKED COPY
    of exactly that file, never at a hand-written geometry.
    """
    prev = step_by_key(plan, step.init_from_key) if step.init_from_key else None
    if prev is not None:
        args = _read_run_args(prev.out)
        if args:
            return args, f"{prev.out}/config.json"
    if cfg.geometry_from:
        p = Path(cfg.geometry_from)
        if p.is_dir():
            p = p / "config.json"
        if p.exists():
            try:
                doc = json.loads(p.read_text(encoding="utf-8"))
            except Exception as e:                            # noqa: BLE001
                raise ChainRefusal(
                    f"[chain] ⛔ --geometry-from {p} is not readable JSON "
                    f"({type(e).__name__}: {e}).")
            args = doc.get("args") or {}
            if args:
                return args, str(p)
            raise ChainRefusal(
                f"[chain] ⛔ --geometry-from {p} has no 'args' block. The "
                f"geometry is carried from a run's OWN config.json, which is "
                f"the record of what its checkpoint was built as.")
        raise ChainRefusal(f"[chain] ⛔ --geometry-from {p} does not exist.")
    raise ChainRefusal(
        f"[chain] ⛔ {step.key} needs its ancestor's MODEL GEOMETRY and cannot "
        f"read it.\n"
        f"  Tried: "
        f"{prev.out + '/config.json' if prev is not None else '(no ancestor)'}"
        f" — not readable from here.\n"
        f"  Emitting the launch anyway is exactly the E1 defect: with no "
        f"geometry flags the stack builds at the trainer's DEFAULTS (enc 384x8, "
        f"pred 768x6, d_tac 512, d_str 256) and `--init-from` dies with "
        f"`size mismatch for encoder.pos: [1, 640, 768] vs [1, 640, 384]` "
        f"(MEASURED 2026-08-17: 87.93 M built against a 336.54 M checkpoint).\n"
        f"  ⇒ generate the command ON the pod, or pass --geometry-from "
        f"<banked config.json> (a copy of the ancestor's own file).")


def carried_geometry(step, cfg, plan, *, skip: set[str] = frozenset()
                     ) -> list[str]:
    """Everything one non-dry step inherits from its ancestor's own record.

    ``skip`` is the set of dests THIS LAUNCH ALREADY NAMES — derived from the
    argv being built, never listed — so a per-stage decision (`--max-horizon`,
    `--w-t1`, `--selector`, the data flags) is never overwritten by the carry.
    """
    if not step.init_from_key and not cfg.geometry_from:
        return []              # S-W starts the ladder; it HAS no ancestor
    args, _where = geometry_source(step, cfg, plan)
    return carried_argv(args, skip=set(skip) | set(GEOMETRY_INTRODUCIBLE))


def carried_argv(args: dict, *, skip: set[str] = frozenset()) -> list[str]:
    """The flags that reproduce ``args``' MODEL GEOMETRY, from a predecessor
    run's own ``config.json["args"]`` — the only authority on what its
    checkpoint holds.

    ⛔ **THE CARRY IS EXACTLY :func:`geometry_dests`, AND NOTHING ELSE.** A
    blanket "carry every recorded arg" is tempting and wrong: the ancestor's
    record also holds RUN CONTROL (`--force-rerun`, `--resume`,
    `--gate-off-reason`, its own `--out`/`--init-from` paths), and carrying
    `--force-rerun` forward would let a successor overwrite a DONE run. Every
    non-geometry knob is a per-stage decision and belongs in the plan, where it
    is visible and a test can reach it.

    ⛔ **GEOMETRY IS EMITTED EVEN WHEN IT EQUALS TODAY'S DEFAULT.** An "omit the
    defaults" line means whatever the trainer's defaults happen to be ON THE DAY
    IT RUNS: change `--enc-dim`'s default from 384 and every banked command
    silently builds a different model. MEASURED 2026-08-17 while building this:
    the omit-defaults version dropped `--frame-h/--frame-w/--patch/--window/
    --horizons/--d-goal-embed/--adapter-hidden/--dt/--uplink/--ema-decay/
    --sigreg-slices/--n-registers` from the S-T line — all correct that day, all
    silently wrong after one default moves.

    ``skip`` holds the dests the launch already names, so a per-stage flag
    always wins; the splice order in :func:`trainer_argv` makes that true for
    ``step.extra`` too, since argparse takes the last occurrence.
    """
    spec = trainer_arg_spec()
    av: list[str] = []
    for dest in sorted(geometry_dests()):
        if dest in skip or dest not in args or dest not in spec:
            continue
        s, v = spec[dest], args[dest]
        if v is None or v == "" or (isinstance(v, (list, tuple)) and not v):
            continue                       # cannot be expressed as a flag
        if s.action == "store_true":
            if v:
                av.append(s.opt)
        elif s.action == "store_false":
            if not v:
                av.append(s.opt)
        elif isinstance(v, (list, tuple)):
            av += [s.opt] + [str(x) for x in v]
        else:
            av += [s.opt, str(v)]
    return av


def assert_geometry_carry(step, plan, argv=None, cfg=None) -> dict:
    """⛔ This step's MODEL GEOMETRY must equal its ancestor's.

    MEASURED 2026-08-16, on a real state_dict round-trip: an S-T checkpoint
    carries four `cand_score.*` keys. Launch S-S with the default
    `--selector none` against it and `load_stage_init` is FATAL —
    ``unexpected (the ckpt has keys this stack does not): ['cand_score.
    cand_bias', 'cand_score.goal_point.bias', 'cand_score.goal_point.weight',
    'cand_score.log_tau']``. That refusal is correct; discovering it at 3 a.m.
    after the corpus has mounted is not. The chain checks it from a JSON file
    before anything is built.

    ⭐ WHEN ``argv`` IS GIVEN, THE CHECK IS THE FULL DERIVED DIFF and not the
    two fields this function used to compare. `selector` and `tac_goal_cond`
    keep their bespoke refusals below because they carry INTRODUCTION semantics
    (`STAGE_MAY_INTRODUCE`) that a value diff cannot express; every OTHER dest
    in :func:`geometry_dests` is compared generically, so a field added to
    ``build_stack_from_args`` tomorrow is covered without editing this file.

    ⚠️ A dest the step EXPLICITLY NAMES in its own flags is a DECLARED change
    and is reported, not refused — an operator who writes `--enc-dim 384` has
    made a visible decision and `load_stage_init` will adjudicate it. What is
    refused is the SILENT OMISSION, which is what E1 actually was: an argv with
    no geometry at all, building 87.93 M against a 336.54 M checkpoint.
    """
    if not step.init_from_key:
        return {"applies": False}
    prev = step_by_key(plan, step.init_from_key)
    # ⛔ THE GUARD READS THE SAME SOURCE THE EMITTER CARRIED FROM. MEASURED
    # 2026-08-17 while building this fix: reading only `<prev.out>/config.json`
    # made the whole check evaporate to `ok: None` on the dev box — where
    # commands are generated and where that path does not exist — so the
    # geometry-free argv passed the guard. `geometry_source` prefers the REAL
    # predecessor config whenever it is readable (the pod case, where the check
    # is against the true record) and falls back to `--geometry-from` (the
    # off-box case) instead of falling back to NOT CHECKING.
    args, source = {}, f"{prev.out}/config.json"
    if cfg is not None:
        try:
            args, source = geometry_source(step, cfg, plan)
        except ChainRefusal:
            args = {}
    else:
        args = _read_run_args(prev.out)
    if not args:
        return {"applies": True, "ok": None,
                "_read": f"{source} not readable and no --geometry-from — the "
                         f"geometry carry-forward could NOT be verified here. "
                         f"load_stage_init will still refuse a mismatch, but "
                         f"only after the stack is built."}
    full = _assert_geometry_diff(step, prev, args, argv) | {"source": source}
    prev_sel = args.get("selector", "none")
    # ---- F-1: the g_str->P_T port carries EXACTLY like the selector --------
    # Checked first-class rather than folded into `extra`: the failure mode is
    # identical (the S-T checkpoint carries cond_tac_dyn.* keys; a flagless
    # successor makes them UNEXPECTED and `load_stage_init` is fatal), and it
    # must be caught from a JSON read before anything is built.
    prev_tgc = bool(args.get("tac_goal_cond", False))
    port: dict = {"tac_goal_cond": step.tac_goal_cond}
    if prev_tgc != step.tac_goal_cond:
        if not prev_tgc and step.stage == "S-T":
            port |= {"introduces_port": "cond_tac_dyn.",
                     "_read_port": "S-T is where the g_str->P_T port is BUILT "
                                   "(STAGE_MAY_INTRODUCE['S-T'] admits "
                                   "'cond_tac_dyn.'), so the predecessor not "
                                   "carrying it is the design (F-1)."}
        elif prev_tgc:
            raise ChainRefusal(
                f"[chain] ⛔ {step.key} would launch WITHOUT --tac-goal-cond, "
                f"but its ancestor {prev.key} trained WITH the g_str->P_T "
                f"port (config.json args.tac_goal_cond = true).\n"
                f"  Every stage saves the WHOLE V6Stack, so cond_tac_dyn.* is "
                f"IN the checkpoint; a stack built without it makes those "
                f"keys UNEXPECTED and `load_stage_init` is fatal — the same "
                f"carry rule as --selector.\n"
                f"  ⇒ carry the port geometry forward: launch {step.key} with "
                f"--tac-goal-cond (the chain does this by default; "
                f"--no-tac-goal-cond only reproduces a pre-F-1 lineage).")
        else:
            raise ChainRefusal(
                f"[chain] ⛔ {step.key} ({step.stage}) would launch with "
                f"--tac-goal-cond, but its ancestor {prev.key} never built "
                f"the port and {step.stage} may introduce nothing "
                f"(STAGE_MAY_INTRODUCE[{step.stage!r}] == ()). The g_str->P_T "
                f"port is introduced at S-T (F-1) — arriving above S-T "
                f"without it means S-T never trained the downlink, which is a "
                f"mis-ordered ladder, not an introduction.")
    if prev_sel == step.selector:
        return {"applies": True, "ok": True,
                "selector": step.selector} | port | full
    if prev_sel == "none" and step.stage == "S-T":
        return {"applies": True, "ok": True, "selector": step.selector,
                "introduces": "cand_score.",
                "_read": "S-T is where the selector is BUILT "
                         "(STAGE_MAY_INTRODUCE['S-T'] includes "
                         "'cand_score.'), so "
                         "the predecessor having none is the design."
                } | port | full
    raise ChainRefusal(
        f"[chain] ⛔ {step.key} would launch with --selector {step.selector!r} "
        f"but its ancestor {prev.key} ran with --selector {prev_sel!r}.\n"
        f"  Every stage saves the WHOLE V6Stack, so the scorer's parameters "
        f"are IN the checkpoint. A stack built without it makes them "
        f"UNEXPECTED keys and `load_stage_init` is fatal (MEASURED: 4 "
        f"cand_score.* keys); a stack built with a DIFFERENT scorer is a "
        f"mis-specified arm, not an introduction.\n"
        f"  ⇒ carry the winning arm forward: --st-winner {prev_sel}")


#: ⛔ The two dests with INTRODUCTION semantics. They are excluded from the
#: generic value diff because `STAGE_MAY_INTRODUCE` makes "predecessor differs"
#: legitimate at exactly one stage each — a fact a value comparison cannot
#: express. They keep their own refusals above; nothing else is exempt.
GEOMETRY_INTRODUCIBLE: tuple[str, ...] = ("selector", "tac_goal_cond")


def _assert_geometry_diff(step, prev, args: dict, argv) -> dict:
    """The generic half of :func:`assert_geometry_carry`: every derived
    geometry dest, predecessor vs emitted argv."""
    if argv is None:
        return {"full_diff": None,
                "_read_full": "no argv was supplied, so ONLY the selector and "
                              "port carry-rules ran. assert_may_launch passes "
                              "the argv it is about to launch; a direct caller "
                              "that does not is checking two fields, which is "
                              "exactly the E1 defect."}
    spec = trainer_arg_spec()
    got = parse_argv_geometry(argv)
    declared = _flags_declared(argv)
    missing_flag, mismatch, declared_diffs = [], [], {}
    for dest in sorted(geometry_dests()):
        if dest in GEOMETRY_INTRODUCIBLE or dest not in spec:
            continue
        # ⛔ AN UNKNOWABLE DEFAULT IS NOT A MISMATCH. When `spec[dest]` has no
        # resolvable default (an imported constant, on a box without torch) the
        # value a FLAGLESS launch takes is unknown here — so a dest that neither
        # side states explicitly is SKIPPED rather than compared against None.
        if not spec[dest].has_default and (dest not in args
                                           or spec[dest].opt not in argv):
            continue
        want = _norm(args[dest]) if dest in args else _norm(spec[dest].default)
        have = _norm(got.get(dest, spec[dest].default))
        if want == have:
            continue
        if dest in declared:
            declared_diffs[dest] = {
                "ancestor": args.get(dest, spec[dest].default),
                "ancestor_is_default": dest not in args, "this": have}
        elif dest not in args:
            continue                   # predecessor predates the flag entirely
        else:
            (mismatch if spec[dest].opt in argv else missing_flag).append(
                {"dest": dest, "flag": spec[dest].opt,
                 "ancestor": args.get(dest), "this_launch": got.get(dest)})
    if missing_flag or mismatch:
        rows = "\n".join(
            f"    {r['flag']:28s} ancestor {r['ancestor']!r:>22}  "
            f"this launch {r['this_launch']!r}"
            for r in (missing_flag + mismatch))
        raise ChainRefusal(
            f"[chain] ⛔ {step.key}'s MODEL GEOMETRY does not match its "
            f"ancestor {prev.key}, on {len(missing_flag) + len(mismatch)} of "
            f"the {len(geometry_dests())} fields "
            f"`build_stack_from_args` reads:\n{rows}\n"
            f"  {len(missing_flag)} of those are flags the launch NEVER "
            f"EMITS, so the stack silently builds at the TRAINER'S DEFAULTS. "
            f"MEASURED 2026-08-17: that produced an 87.93 M stack against a "
            f"336.54 M checkpoint and `RuntimeError … size mismatch for "
            f"encoder.pos: [1, 640, 768] vs [1, 640, 384]`.\n"
            f"  ⇒ the geometry is CARRIED from {prev.out}/config.json — make "
            f"that file readable from here, or point the chain at a banked "
            f"copy with --geometry-from <path>. Every field above comes from "
            f"the ancestor's own record; none of it is typed by hand.")
    return {"full_diff": {"n_checked": len(geometry_dests()),
                          "declared_changes": declared_diffs},
            "_read_full": f"every dest build_stack_from_args reads was "
                          f"compared against {prev.out}/config.json['args']; "
                          f"{len(declared_diffs)} were changed by an explicit "
                          f"flag on this step (a declared decision, reported "
                          f"not refused)."}


# ============================================================================
# refusal 4 — S-S INVALIDATES S-T's certificate, and the re-measurement is
# REQUIRED, not reported
# ============================================================================

#: The two probes `STAGE_GATE_SPEC["S-S"]["required"]` adds for exactly one
#: reason: S-S trains `layer_str`, whose output flows
#: `goal_head_str -> e_g_str -> goal_head_tac(cond=e_g_str) -> e_g_tac`, and
#: `e_g_tac` is the SELECTOR'S ONLY INPUT. The selector is FROZEN in S-S — but
#: its input distribution MOVES. S-T certified `sel_gap` and the TACTICAL family
#: against the S-T-era `e_g_tac`; that certificate does not survive S-S.
S_S_REVALIDATIONS: tuple[str, ...] = ("sel_gap_revalidated",
                                      "TACTICAL_revalidated")


def assert_revalidations_present(step, plan, *, allow_inconclusive=False,
                                 off_reason: str = "") -> dict:
    """⛔ Nothing may consume S-S's checkpoint on a gate that skipped the
    re-measurement.

    The generic gate rule already refuses an INCONCLUSIVE verdict, and a gate
    missing these probes IS inconclusive — but a chain that only says
    "INCONCLUSIVE" sends the operator to look for any missing probe. This names
    the two and the seam, so the fix is obvious and the override is a conscious
    act rather than a shrug at an unexplained key.

    ⚠️ The waiver is the SAME one the design commits to and no wider: omitting
    the re-measurement is overridable by `--allow-inconclusive-gate` WITH a
    recorded reason. Running it and REGRESSING is a FAIL, and a FAIL has no
    override anywhere in this ladder — it is the trigger for the S-S' planner
    refit.
    """
    if step.prev_gate_key is None:
        return {"applies": False}
    prev = step_by_key(plan, step.prev_gate_key)
    if prev.stage != "S-S":
        return {"applies": False, "reason": f"{prev.key} is not S-S"}
    p = Path(prev.gate)
    if not p.exists():
        return {"applies": True, "ok": None,
                "_read": "S-S has no gate yet; the generic precondition check "
                         "owns that refusal."}
    try:
        probes = (json.loads(p.read_text()).get("probes") or {})
    except Exception as e:
        return {"applies": True, "ok": None,
                "error": f"{type(e).__name__}: {e}"}
    absent = [k for k in S_S_REVALIDATIONS
              if probes.get(k, {}).get("pass") is None]
    if not absent:
        return {"applies": True, "ok": True,
                "revalidated": list(S_S_REVALIDATIONS)}
    if allow_inconclusive and off_reason.strip():
        return {"applies": True, "ok": True, "override": "allow-inconclusive-gate",
                "off_reason": off_reason, "not_revalidated": absent,
                "_read": "⚠️ RECORDED WAIVER: S-J is launching on a selector "
                         "certificate that S-S invalidated and nobody "
                         "re-earned. Any sel_gap / TACTICAL number carried "
                         "forward from S-T is stale."}
    raise ChainRefusal(
        f"[chain] ⛔ {step.key} would consume S-S's checkpoint, but S-S's gate "
        f"never re-measured {absent}.\n"
        f"  S-S trains `layer_str`, and its output flows goal_head_str -> "
        f"e_g_str -> goal_head_tac(cond=e_g_str) -> e_g_tac, which is the "
        f"SELECTOR'S ONLY INPUT. The selector and goal_head_tac are FROZEN in "
        f"S-S — but their INPUT DISTRIBUTION MOVES. S-T certified sel_gap and "
        f"the TACTICAL family against the S-T-era e_g_tac, and that "
        f"certificate does not survive S-S. This is registry §1.14's "
        f"consumer-invalidation one level up: you cannot repair a trunk and "
        f"keep its planner.\n"
        f"  ⚠️ They are `required`, not `reported`, precisely so that omitting "
        f"them reads INCONCLUSIVE and never PASS — a reported-only probe is "
        f"what a silent carry-forward looks like.\n"
        f"  ⇒ re-run the two probes under the POST-S-S g_tac (paired "
        f"episode-cluster bootstrap, same windows) and re-gate. A measured "
        f"REGRESSION is a FAIL, and a FAIL here is the trigger for the S-S' "
        f"planner-refit micro-stage — not a carry-forward.")


# ============================================================================
# refusal 5 — the out directory must not already belong to something else
# ============================================================================

def out_dir_state(step) -> dict:
    """What is in this step's ``--out`` right now. Never raises."""
    out = Path(step.out)
    state = {"out": step.out, "exists": out.exists(),
             "has_ckpt": Path(step.ckpt).exists(),
             "has_gate": Path(step.gate).exists(),
             "done": False, "gate_verdict": None, "ckpt_stage": None,
             "gate_is_dry_run": None, "step": None}
    if Path(step.done_marker).exists():
        try:
            m = json.loads(Path(step.done_marker).read_text())
            state["done"] = m.get("done") is True
            state["done_marker"] = m
        except Exception as e:
            state["done_marker_error"] = f"{type(e).__name__}: {e}"
    if state["has_gate"]:
        try:
            g = json.loads(Path(step.gate).read_text())
            state["gate_verdict"] = g.get("verdict")
            state["gate_is_dry_run"] = bool(g.get("_dry_run"))
        except Exception as e:
            state["gate_error"] = f"{type(e).__name__}: {e}"
    if state["has_ckpt"]:
        try:
            from train_v6_staged import read_ckpt_provenance
            prov = read_ckpt_provenance(step.ckpt)
            state["ckpt_stage"] = prov.get("stage")
            state["step"] = prov.get("step")
            state["ckpt_provenance"] = prov
        except Exception as e:
            # ⚠️ Absence at one location is not absence: say WHAT was not
            # reachable rather than reporting "no provenance".
            state["ckpt_provenance"] = {
                "readable": None,
                "error": f"{type(e).__name__}: {e}",
                "_read": "torch (or train_v6_staged) could not be imported "
                         "here, so the checkpoint's stage label was NOT read. "
                         "This is an instrument limit, not a finding."}
    return state


def assert_out_dir_free(step) -> dict:
    """⛔ Refuse a launch into a directory that holds another stage's run."""
    st = out_dir_state(step)
    if st["ckpt_stage"] and st["ckpt_stage"] != step.stage:
        raise ChainRefusal(
            f"[chain] ⛔ {step.key} would write to {step.out}, which already "
            f"holds a ckpt.pt written by stage {st['ckpt_stage']!r} at step "
            f"{st['step']}.\n"
            f"  `--resume auto` would find it, and the trainer would refuse — "
            f"but the fix is not to rely on that refusal. Give this step its "
            f"own directory. A resume that adopted the other stage's step "
            f"would replay the LR schedule to the wrong point.")
    if st["done"]:
        return {"ok": True, "already_done": True, "state": st}
    return {"ok": True, "already_done": False, "state": st}


# ============================================================================
# THE ADJUDICATOR — one gate rule, imported, never re-implemented
# ============================================================================

def assert_may_launch(step, plan, cfg: "ChainConfig | None" = None, *,
                      allow_inconclusive: bool = False,
                      off_reason: str = "", dry: bool = False,
                      unpaired_arm_reason: str = "") -> dict:
    """Everything that must hold before ``step`` is launched, in cost order.

    ⛔ The gate rule itself is `train_v6_staged.assert_stage_precondition` —
    imported, not re-implemented. FAIL has NO override. INCONCLUSIVE is not a
    pass and needs `--allow-inconclusive-gate` WITH a recorded reason. A second
    copy of that rule living in a chain script is how the two drift apart, and
    the ladder then has two different opinions about what a pass is.
    """
    report: dict = {"step": step.key, "stage": step.stage, "out": step.out}
    if step.needs_st_winner:
        raise ChainRefusal(
            f"[chain] ⛔ {step.key} continues ONE of the two S-T arms, and no "
            f"--st-winner was declared.\n"
            f"  The ladder forks at S-T: 'goal' (the mechanism) and 'mlp' (the "
            f"capacity control) produce two different checkpoints, and which "
            f"lineage the strategic stage continues is a DECISION about what "
            f"v6f IS — not a default a chain may pick. It also fixes the "
            f"scorer GEOMETRY every stage above must build.\n"
            f"  ⇒ read both arms' gates, then pass --st-winner goal|mlp.")
    report["plan"] = assert_plan(plan)
    # ⛔ FIRST, and before anything expensive: a scorer may not be built while
    # SEL-1 stands REFUSED. This is a FIRED pre-registration, not a preference,
    # so leaving it out of the default plan is not sufficient — it is refused
    # at launch too.
    report["sel1_admission"] = assert_selector_admissible(
        step, cfg if cfg is not None else ChainConfig())
    report["out_dir"] = assert_out_dir_free(step)
    report["arm_pair"] = assert_arm_pair(
        step, plan, unpaired_arm_reason=unpaired_arm_reason)
    # the CHEAP half — a JSON read, the selector and the F-1 port
    report["geometry"] = assert_geometry_carry(step, plan, None, cfg)
    report["revalidations"] = assert_revalidations_present(
        step, plan, allow_inconclusive=allow_inconclusive,
        off_reason=off_reason)

    # ⛔ THE CERTIFICATE BEFORE THE WEIGHTS. When the stage below never ran,
    # both its gate and its checkpoint are missing — and "the stage below did
    # not pass a gate" is the diagnosis, while "a file is missing" sends the
    # operator to look for a path. X5 orders these two, so this does too.
    if step.prev_gate_key is None:
        report["precondition"] = {"ok": True, "precondition": None,
                                  "reason": "S-W starts the ladder"}
    else:
        prev_gate = step_by_key(plan, step.prev_gate_key)
        from train_v6_staged import assert_stage_precondition  # THE adjudicator
        report["precondition"] = assert_stage_precondition(
            step.stage, prev_gate.gate, allow_inconclusive=allow_inconclusive,
            off_reason=off_reason, dry_run=dry)

    if step.init_from_key:
        prev = step_by_key(plan, step.init_from_key)
        init = _init_from_path(step, plan, dry=dry)
        if not Path(init).exists():
            raise ChainRefusal(
                f"[chain] ⛔ {step.key} must --init-from {init}, which does not "
                f"exist. A gate saying {prev.key} passed is worthless if this "
                f"stage then trains on a randomly-initialised trunk while its "
                f"log looks healthy — that is not a staged protocol, it is "
                f"four unrelated models with a gate between them.")
        report["init_from"] = init
    else:
        report["init_from"] = None

    # ⭐ LAST, AND ON THE ARGV THAT IS ACTUALLY ABOUT TO BE LAUNCHED. E1 was a
    # two-field comparison against a config.json holding EVERY geometry field,
    # so the real check has to see the emitted command, not just the step.
    # ⛔ It runs AFTER the certificate and the weights on purpose: the chain's
    # own ordering rule is that "the stage below did not pass a gate" is the
    # diagnosis an operator needs first, and a geometry refusal on a step whose
    # gate never passed would send them to the wrong problem.
    if cfg is not None:
        report["geometry"] = assert_geometry_carry(
            step, plan,
            trainer_argv(step, cfg, plan, allow_inconclusive=allow_inconclusive,
                         off_reason=off_reason), cfg)
    return report


def _init_from_path(step, plan, *, dry: bool) -> str:
    """Where this step's weights come from.

    In a dry ladder the predecessor is `dry_ckpt.pt`, written by
    :func:`write_dry_predecessor` — a DIFFERENT filename on purpose, so no real
    `--resume auto` can ever find it and no dry artifact can be mistaken for a
    run's checkpoint.
    """
    prev = step_by_key(plan, step.init_from_key)
    return posixpath.join(prev.out, "dry_ckpt.pt" if dry else "ckpt.pt")


# ============================================================================
# emitting the launch
# ============================================================================

def trainer_argv(step, cfg: ChainConfig, plan, *, allow_inconclusive=False,
                 off_reason: str = "") -> list[str]:
    """The exact argv for one stage. This is the ONLY place a v6 launch line is
    constructed, so 'the command someone improvised at 3 a.m.' has nowhere to
    come from."""
    # ⛔ `--n-candidates` IS EMITTED ON EVERY STEP, S-W INCLUDED. MEASURED
    # 2026-08-16 by the dry ladder, on its first execution: S-W took the
    # trainer's default 8 while S-T ran the tiny fan of 3, and `--init-from`
    # died with `size mismatch for cand_queries.weight: [8, 256] vs [3, 256]`.
    # ⚠️ That failure does NOT go through `load_stage_init`'s adjudication —
    # `load_state_dict(strict=False)` tolerates missing/unexpected KEYS but
    # still RAISES on a shape mismatch, so `STAGE_MAY_INTRODUCE` never sees it.
    # The fan size is a ladder-wide constant (SEL-5 makes it a declared ARM, and
    # an arm is declared once for the whole ladder, not per stage).
    head = ["--stage", step.stage, "--out", step.out,
            "--steps", str(step.steps), "--batch", str(cfg.batch),
            "--lr", str(step.lr), "--n-candidates", str(cfg.n_candidates)]
    av: list[str] = []
    if step.init_from_key:
        av += ["--init-from", _init_from_path(step, plan, dry=cfg.dry)]
    if step.prev_gate_key:
        av += ["--prev-gate", step_by_key(plan, step.prev_gate_key).gate]
    if step.max_horizon:
        av += ["--max-horizon", str(step.max_horizon)]
    if step.w_select:
        av += ["--w-select", str(step.w_select)]
    # ⭐ F-1: the g_str->P_T port, on every stage after S-W (S-T builds and
    # TRAINS it; S-S/S-J carry the GEOMETRY, or the S-T checkpoint's
    # cond_tac_dyn.* keys become unexpected and --init-from is fatal — the
    # same carry rule as --selector, enforced by assert_geometry_carry).
    if step.tac_goal_cond:
        av += ["--tac-goal-cond"]
    av += list(step.extra)
    if cfg.dry:
        av += ["--dry-run", "--device", "cpu", "--dry-steps", str(cfg.dry_steps),
               "--dry-batch", "2", "--dry-k", "12", "--o5-k", "12"]
        if cfg.tiny:
            av += list(TINY_GEOMETRY)
    else:
        # ⛔ `--v2-subframe` IS DELIBERATELY ABSENT. In `train_v6_staged` the
        # sub-frame moves the DATA and NOT the MODEL: `build_stack_from_args`
        # sizes the encoder from `--frame-h/--frame-w`, and `resolve_eval_frames`
        # applies the sub-frame to a flagship-v4 EVAL config 120 lines later.
        # MEASURED 2026-08-17: `--v2-subframe 176x624` against a 256x640
        # encoder lets `--init-from` SUCCEED and parity PASS, then dies at the
        # first forward with `encoder input is (176, 624) but the config
        # declares (256, 640)` — after the corpus has mounted. The live S-W run
        # carries `v2_subframe: null`, and the ladder carries it forward.
        # (`train_v6_staged.subframe_desync` now refuses this at STARTUP too.)
        # ⚠️ `--save-every` is EXPLICIT, not left to the trainer's default of
        # 1000: the live S-W run and the validated S-T line both use 250, and
        # the SEAM DUMP fires at exactly this boundary — so the default would
        # quarter the number of banked plans without anyone choosing that.
        av += ["--v2-cache", cfg.train_cache, "--v2-val-cache", cfg.val_cache,
               "--v2-lru", str(cfg.v2_lru), "--save-every", str(cfg.save_every),
               "--frame-hfov", "120", "--projection", "cylindrical",
               "--require-parity"]
    # ⭐ E5 — X2_seam has read "not-run" for three stages running, and the
    # reason is that NOTHING TURNED THE DUMP ON: `--dump-seam-plan` appeared in
    # no `ChainStep.extra` and in no `trainer_argv` branch, so the module was
    # never even imported. Zero extra GPU (the plan is already computed for the
    # step's loss); S-W is excluded because its emission head is zero-init and
    # the dump would be DEGENERATE.
    if not cfg.dry and cfg.dump_seam_plan and step.stage != "S-W":
        av += ["--dump-seam-plan", seam_dir(step)]
    if allow_inconclusive:
        av += ["--allow-inconclusive-gate", "--gate-off-reason", off_reason]
    av += list(cfg.extra_common)
    # ⭐ E1 — THE ANCESTOR'S BUILD, CARRIED FROM ITS OWN RECORD, SPLICED IN
    # FIRST. Before this, `trainer_argv` emitted NO geometry flag on a non-dry
    # step (`TINY_GEOMETRY` was added only under `cfg.dry and cfg.tiny`), so the
    # stack built at the trainer's DEFAULTS and `--init-from` died with
    # `size mismatch for encoder.pos: [1, 640, 768] vs [1, 640, 384]` — 87.93 M
    # built against a 336.54 M checkpoint (MEASURED 2026-08-17).
    # ⛔ THE SKIP SET IS DERIVED FROM THE ARGV, NOT LISTED. Every flag this
    # function already decided — the per-stage ones (`--max-horizon`, `--w-t1`,
    # `--selector`), the data ones, the identity block — is excluded because it
    # is PRESENT, so a stage decision can never be overwritten by the carry and
    # a new per-stage flag needs no edit here. Splicing after `head` (rather
    # than appending) also means `step.extra` wins on any collision, since
    # argparse takes the LAST occurrence.
    # ⭐ THE DRY LADDER CARRIES TOO. It exists to be "the SAME wiring and the
    # SAME seams as production", and E1 lived in a seam no test reached; running
    # the carry here makes the CPU dry ladder a real rehearsal of it.
    return head + carried_geometry(
        step, cfg, plan, skip=_flags_declared(head + av)) + av


def seam_dir(step) -> str:
    """Where this step banks its seam dumps."""
    return posixpath.join(step.out, "seam")


def launch_line(step, cfg: ChainConfig, plan, **kw) -> str:
    """The copy-pasteable pod-side line, with the two env vars that are not
    optional: PYTHONPATH (or it dies with ``ModuleNotFound: tanitad``) and
    OMP_NUM_THREADS (torch spawns ~113 threads/process; 7 concurrent arms sat
    at 0-6 % sm for 50 minutes)."""
    argv = trainer_argv(step, cfg, plan, **kw)
    body = " ".join(shlex.quote(x) for x in argv)
    mk = seam_dir(step) if "--dump-seam-plan" in argv else step.out
    return (f"mkdir -p {shlex.quote(mk)} && cd {shlex.quote(cfg.workdir)} "
            f"&& PYTHONPATH={cfg.pythonpath} OMP_NUM_THREADS=6 "
            f"PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
            f"setsid nohup {cfg.python} -u {TRAINER} {body} "
            f"> {shlex.quote(posixpath.join(step.out, 'train.out'))} 2>&1 "
            f"< /dev/null &")


# ============================================================================
# the supervisor seam — one manifest PER STAGE, and never for this file
# ============================================================================

def manifest_text(step, cfg: ChainConfig, plan, **kw) -> str:
    """A `supervise_run.sh` manifest for ONE stage.

    ⚠️ `supervise_run.sh` SOURCES ITS MANIFEST ONCE, at supervisor startup, and
    replays the `TRAIN_CMD` it captured. Two consequences, both baked in here:

    1. ⛔ **`TRAIN_CMD` is the TRAINER, never `v6_chain.py`.** A supervised
       chain would replay stage 1 after a mid-ladder crash.
    2. **Editing a manifest under a live supervisor changes NOTHING.** To change
       a supervised run: edit the manifest -> kill the SUPERVISOR first ->
       kill the trainer -> start a fresh supervisor. Killing the trainer first
       just makes the supervisor restore the stale command. And verify by
       grepping the flags out of the RUNNING PROCESS (`v6_chain.py verify`),
       never by reading the manifest back.
    """
    argv = trainer_argv(step, cfg, plan, **kw)
    body = " ".join(shlex.quote(x) for x in argv)
    return "\n".join([
        f"# {step.run_id} — v6 ladder step {step.key}. Generated by "
        f"scripts/v6_chain.py.",
        "# ⛔ ONE MANIFEST PER STAGE. Never supervise v6_chain.py itself: the",
        "#    supervisor replays the command it captured at STARTUP, so a",
        "#    supervised chain would replay stage 1 after a mid-ladder crash.",
        "# ⚠️ The manifest is SOURCED ONCE. To change this run: edit here ->",
        "#    kill the SUPERVISOR -> kill the trainer -> start a fresh",
        "#    supervisor. Verify with `v6_chain.py verify --step "
        f"{step.key}` (reads /proc), never by reading this file back.",
        f"RUN_ID={step.run_id}",
        f"OUT={step.out}",
        f"WORKDIR={cfg.workdir}",
        f"TRAIN_MATCH='train_v6_staged\\.py.*{step.run_id}'",
        # ⛔ BOTH ROOTS — see ChainConfig.pythonpath. A supervisor that
        # restores this run must restore the seam dump's import too.
        f"export PYTHONPATH={cfg.pythonpath}",
        "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
        "export OMP_NUM_THREADS=6",
        f"TRAIN_CMD={shlex.quote(cfg.python + ' -u ' + TRAINER + ' ' + body)}",
        "",
    ])


def train_cmd_of(manifest: str) -> str:
    """The `TRAIN_CMD` line's VALUE, unquoted.

    The manifest's comment block deliberately names `v6_chain.py` (to say it
    must never be supervised), so "does the manifest mention the chain" is the
    wrong question and answers `True` for a correct manifest. The question that
    matters is what the supervisor will REPLAY, which is exactly this string.
    """
    for line in manifest.splitlines():
        if line.startswith("TRAIN_CMD="):
            return " ".join(shlex.split(line[len("TRAIN_CMD="):]))
    raise ChainRefusal("[chain] ⛔ manifest has no TRAIN_CMD "
                       "(supervise_run.sh requires RUN_ID OUT WORKDIR "
                       "TRAIN_CMD)")


def verify_probe(step) -> str:
    """The shell text that proves WHICH command is actually running.

    ⛔ Not `pgrep -f` and not `ps | grep`. Both put the searched token into the
    searching process's own command line, so the probe MATCHES ITSELF — a trap
    this programme has measured three times, most recently as a monitor that
    reported `Traceback CUDA out of memory` for a healthy run 3 minutes in.
    This reads `/proc/*/cmdline` from a heredoc (whose own cmdline is just
    `python3 -`, containing neither token) and emits an OPAQUE marker, so the
    emitted token is disjoint from the searched one.
    """
    return (
        "python3 - <<'PROBE'\n"
        "import glob, os\n"
        "want_a, want_b = 'train_v6_staged' '.py', " + repr(step.out) + "\n"
        "hits = []\n"
        "for c in glob.glob('/proc/[0-9]*/cmdline'):\n"
        "    try:\n"
        "        parts = open(c, 'rb').read().decode('utf-8', 'replace')\\\n"
        "                    .split('\\x00')\n"
        "    except Exception:\n"
        "        continue\n"
        "    if any(want_a in p for p in parts) and want_b in parts:\n"
        "        hits.append((os.path.basename(os.path.dirname(c)),\n"
        "                     ' '.join(p for p in parts if p)))\n"
        "print('ZZ%dZZ' % len(hits))\n"
        "for pid, cmd in hits:\n"
        "    print('ZZPID%sZZ %s' % (pid, cmd))\n"
        "PROBE")


# ============================================================================
# the dry ladder — EXECUTED, because a chain validated by reading is exactly
# what the last six defects defeated
# ============================================================================

def write_dry_predecessor(step, cfg: ChainConfig, plan, **kw) -> str:
    """Write ``<out>/dry_ckpt.pt`` so the NEXT dry stage has a real ancestor.

    A dry-run deliberately writes no `ckpt.pt` (a synthetic checkpoint that
    looks real is worse than none). But `--init-from` cannot be exercised
    without a predecessor FILE, and exercising it is the whole point: the S-W ->
    S-T edge is where `strict=True` refused the designed `cand_score.*`
    introduction and made both arms unlaunchable. So the chain writes a
    *differently named* artifact, built with this step's own geometry and
    stamped `_dry_run`, and the next stage really loads it through
    `load_stage_init`.
    """
    import torch
    from train_v6_staged import (_save_ckpt, build_parser,
                                 build_stack_from_args)
    from tanitad.models.v6 import apply_stage_freeze
    ap = build_parser()
    ap.add_argument("--i-know-this-is-the-control-arm", action="store_true",
                    dest="control_arm_ack")
    a = ap.parse_args(trainer_argv(step, cfg, plan, **kw))
    torch.manual_seed(0)
    stack = build_stack_from_args(a)
    apply_stage_freeze(stack, step.stage)
    tr = [p for p in stack.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(tr, lr=step.lr)
    sum((p * p).sum() for p in tr).backward()
    opt.step()
    for p in stack.parameters():
        p.grad = None
    dest = Path(step.out) / "dry_ckpt.pt"
    _save_ckpt(dest, stack=stack, opt=opt, step=step.steps, cfg_json={
        "stage": step.stage, "run": f"v6-staged-{step.stage}", "_dry_run": True,
        "args": {"selector": step.selector,
                 "tac_goal_cond": step.tac_goal_cond},
        "_read": "DRY-LADDER ancestor. Synthetic weights, written by "
                 "v6_chain.py so --init-from can be EXECUTED on CPU. Named "
                 "dry_ckpt.pt, never ckpt.pt, so --resume auto can never find "
                 "it and it can never be mistaken for a run."})
    return str(dest)


def run_chain(cfg: ChainConfig, *, stop_after: str | None = None,
              allow_inconclusive: bool = False, off_reason: str = "",
              unpaired_arm_reason: str = "", echo=print) -> dict:
    """Execute the ladder, one stage per subprocess, adjudicating between them.

    ⛔ For the REAL ladder you do not use this: each stage is a multi-day GPU
    job and belongs under its own supervisor. This is the CPU dry ladder's
    driver — and it is what makes the chain's advance logic *executed* rather
    than reasoned about.
    """
    plan = build_plan(cfg)
    assert_plan(plan)
    if cfg.dry:
        allow_inconclusive, off_reason = True, DRY_OFF_REASON
    env = dict(os.environ)
    env["PYTHONPATH"] = (str(_STACK) + os.pathsep
                         + env.get("PYTHONPATH", "")).rstrip(os.pathsep)
    env.setdefault("OMP_NUM_THREADS", "6")
    env["PYTHONUTF8"] = "1"
    transcript: list[dict] = []
    t_all = time.time()
    for step in plan:
        Path(step.out).mkdir(parents=True, exist_ok=True)
        kw = dict(allow_inconclusive=allow_inconclusive and
                  step.prev_gate_key is not None, off_reason=off_reason)
        echo(f"\n=== {step.key} ({step.stage}) -> {step.out} ===")
        pre = assert_may_launch(
            step, plan, cfg, allow_inconclusive=allow_inconclusive,
            off_reason=off_reason, dry=cfg.dry,
            unpaired_arm_reason=unpaired_arm_reason)
        echo(f"[chain] may-launch OK · precondition="
             f"{pre['precondition'].get('prev_verdict', 'n/a')} · "
             f"init_from={pre['init_from']}")
        argv = trainer_argv(step, cfg, plan, **kw)
        t0 = time.time()
        proc = subprocess.run(
            [sys.executable, "-u", str(_HERE / "train_v6_staged.py")] + argv,
            cwd=str(_STACK), env=env, capture_output=True, text=True,
            encoding="utf-8", errors="replace")
        row = {"step": step.key, "stage": step.stage, "out": step.out,
               "argv": argv, "returncode": proc.returncode,
               "elapsed_s": round(time.time() - t0, 2),
               "stdout_tail": proc.stdout.strip().splitlines()[-6:],
               "stderr_tail": proc.stderr.strip().splitlines()[-6:],
               "precondition": pre["precondition"],
               "arm_pair": pre["arm_pair"], "geometry": pre["geometry"]}
        for line in proc.stdout.strip().splitlines():
            echo("    " + line)
        transcript.append(row)
        if proc.returncode != 0:
            echo(f"[chain] ⛔ {step.key} FAILED (rc={proc.returncode})")
            for line in proc.stderr.strip().splitlines()[-12:]:
                echo("    ! " + line)
            break
        gate = json.loads(Path(step.gate).read_text())
        row["gate_verdict"] = gate["verdict"]
        row["gate_is_dry_run"] = bool(gate.get("_dry_run"))
        echo(f"[chain] {step.key} gate {gate['verdict']}"
             f"{' (_dry_run)' if gate.get('_dry_run') else ''}")
        if cfg.dry:
            row["dry_ckpt"] = write_dry_predecessor(step, cfg, plan, **kw)
            echo(f"[chain] dry ancestor -> {row['dry_ckpt']}")
        if stop_after and step.key == stop_after:
            echo(f"[chain] stopping after {step.key} as asked")
            break
    return {"steps": transcript, "elapsed_s": round(time.time() - t_all, 2),
            "dry": cfg.dry, "root": cfg.root,
            "_evidence_class": "MEASURED (ours; executed transcript)"}


# ============================================================================
# reporting
# ============================================================================

def wallclock(step, cfg: ChainConfig) -> dict:
    s = step.steps * cfg.s_per_step
    return {"steps": step.steps, "s_per_step": cfg.s_per_step,
            "hours": round(s / 3600, 2), "days": round(s / 86400, 2)}


def status(cfg: ChainConfig) -> dict:
    plan = build_plan(cfg)
    rows = []
    for s in plan:
        st = out_dir_state(s)
        st.update(key=s.key, stage=s.stage, arm=s.arm,
                  wallclock=wallclock(s, cfg))
        rows.append(st)
    return {"root": cfg.root, "steps": rows,
            "_evidence_class": "MEASURED (ours; filesystem read)"}


def next_launchable(cfg: ChainConfig, **kw) -> dict:
    """The first step that is not done, plus its verdict. Never guesses."""
    plan = build_plan(cfg)
    assert_plan(plan)
    for s in plan:
        st = out_dir_state(s)
        if st["done"]:
            continue
        try:
            rep = assert_may_launch(s, plan, cfg, dry=cfg.dry, **kw)
            return {"next": s.key, "may_launch": True, "report": rep,
                    "command": launch_line(s, cfg, plan)}
        except SystemExit as e:
            return {"next": s.key, "may_launch": False, "refusal": str(e)}
    return {"next": None, "may_launch": False,
            "refusal": "every step in the ladder has a done-marker"}


# ============================================================================
# CLI
# ============================================================================

def _cfg_from_args(a) -> ChainConfig:
    cfg = ChainConfig()
    for f in ("root", "sw_dir", "train_cache", "val_cache", "workdir", "python",
              "batch", "v2_lru", "lr", "sj_lr", "sw_steps", "st_steps",
              "ss_steps", "sj_steps", "st_winner", "w_select", "n_candidates",
              "selector_tau_m", "selector_mlp_hidden", "plan_wta_eps",
              "s_per_step", "dry_steps", "geometry_from"):
        v = getattr(a, f, None)
        if v is not None:
            setattr(cfg, f, v)
    if getattr(a, "no_seam_dump", False):
        cfg.dump_seam_plan = False
    if getattr(a, "st_arms", None):
        cfg.st_arms = tuple(a.st_arms)
    if getattr(a, "no_tac_goal_cond", False):
        cfg.tac_goal_cond = False
    cfg.dry = bool(getattr(a, "dry", False))
    cfg.tiny = bool(getattr(a, "tiny", False)) or cfg.dry
    if cfg.dry and getattr(a, "n_candidates", None) is None:
        cfg.n_candidates = 3          # the tiny geometry's fan
    # ⚠️ --steps is NOT reduced in the dry ladder: `--dry-steps` governs the
    # synthetic loop, and keeping --steps at its real value keeps the dry
    # ladder's OUT-DIR NAMES identical to production's, so the transcript
    # exercises the same plan shape the pod will.
    if getattr(a, "a40", False):
        cfg.batch, cfg.s_per_step = A40_BATCH, A40_S_PER_STEP
    return cfg


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="v6 stage chain (S-W -> S-T{goal,mlp} -> S-S -> S-J)")
    ap.add_argument("cmd", choices=("plan", "commands", "status", "next",
                                    "manifests", "verify", "run", "admission"))
    ap.add_argument("--root", default=None)
    ap.add_argument("--sw-dir", default=None)
    ap.add_argument("--train-cache", default=None)
    ap.add_argument("--val-cache", default=None)
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--python", default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--v2-lru", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--sj-lr", type=float, default=None)
    ap.add_argument("--sw-steps", type=int, default=None)
    ap.add_argument("--st-steps", type=int, default=None)
    ap.add_argument("--ss-steps", type=int, default=None)
    ap.add_argument("--sj-steps", type=int, default=None)
    ap.add_argument("--st-arms", nargs="+", default=None,
                    help="⛔ OFF by default: SEL-1 is REFUSED (E-WC2, "
                         "2026-08-16, σ/ADE 9.9915 vs a pre-registered "
                         "refusal line of 3.0). Opting in ALSO requires the "
                         "E-WC2-SW dump to reach the threshold pre-registered "
                         "in SW_LATENT_ADMISSION. If used, 'goal' and 'mlp' "
                         "are a PAIR — dropping one makes a 'goal' result "
                         "unattributable (C6).")
    ap.add_argument("--st-winner", choices=("goal", "mlp"), default=None,
                    help="required before S-S/S-J: which arm's lineage the "
                         "strategic stage continues. Never guessed.")
    ap.add_argument("--no-tac-goal-cond", action="store_true",
                    help="⛔ drop the g_str->P_T port from the ladder (F-1 "
                         "says S-T must NOT launch without it — this exists "
                         "only to reproduce a pre-F-1 lineage as a DECLARED "
                         "decision). Default: every stage after S-W carries "
                         "--tac-goal-cond; S-T introduces the zero-init "
                         "cond_tac_dyn keys, S-S/S-J carry the geometry.")
    ap.add_argument("--w-select", type=float, default=None)
    ap.add_argument("--n-candidates", type=int, default=None)
    ap.add_argument("--selector-tau-m", type=float, default=None)
    ap.add_argument("--selector-mlp-hidden", type=int, default=None)
    ap.add_argument("--plan-wta-eps", type=float, default=None)
    ap.add_argument("--s-per-step", type=float, default=None)
    ap.add_argument("--geometry-from", default=None,
                    help="a BANKED COPY of the ancestor run's config.json (or "
                         "its directory), for generating commands off-pod. ⛔ "
                         "The real <prev>/config.json ALWAYS wins when it is "
                         "readable; this is not a place to type a geometry. "
                         "Without one of the two, a step with --init-from "
                         "REFUSES to emit — a launch line with no geometry "
                         "builds 87.93 M against a 336.54 M checkpoint (E1).")
    ap.add_argument("--no-seam-dump", action="store_true",
                    help="do NOT emit --dump-seam-plan. The dump costs zero "
                         "extra GPU and is the only thing that feeds X2_seam, "
                         "so turning it off is a declared choice.")
    ap.add_argument("--a40", action="store_true",
                    help="A40 constants (batch 16, 20.46 s/step) instead of "
                         "Thor's (batch 8, 27.18 s/step)")
    ap.add_argument("--dry", action="store_true",
                    help="the CPU dry ladder: tiny geometry, --dry-run, no "
                         "corpus, no GPU")
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--dry-steps", type=int, default=None)
    ap.add_argument("--step", default=None)
    ap.add_argument("--stop-after", default=None)
    ap.add_argument("--dest", default=None)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--allow-inconclusive-gate", action="store_true")
    ap.add_argument("--gate-off-reason", default="")
    ap.add_argument("--unpaired-arm-reason", default="")
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    cfg = _cfg_from_args(a)
    plan = build_plan(cfg)
    assert_plan(plan)
    steps = ([step_by_key(plan, a.step)] if a.step else list(plan))
    out: dict | list | None = None

    if a.cmd == "plan":
        out = {"root": cfg.root, "config": asdict(cfg),
               "steps": [asdict(s) | {"gate": s.gate, "ckpt": s.ckpt,
                                      "wallclock": wallclock(s, cfg)}
                         for s in plan],
               "thor": {"s_per_step": THOR_S_PER_STEP, "batch": THOR_BATCH,
                        "memory_probe": THOR_MEMORY_PROBE,
                        "_read": "MEASURED 2026-08-16 (marginal, steps "
                                 "6300->6400 of the live S-W run). Thor "
                                 "saturates at batch 8; a bigger batch buys "
                                 "nothing and costs memory."},
               "total_wallclock_days": round(
                   sum(s.steps for s in plan if s.key != "S-W")
                   * cfg.s_per_step / 86400, 2),
               "_evidence_class": "MEASURED (s/step) + ESTIMATED (wall-clock, "
                                  "S-W's rate applied to stages that have "
                                  "never run)"}
        print(json.dumps(out, indent=1))
    elif a.cmd == "commands":
        kw = dict(allow_inconclusive=a.allow_inconclusive_gate,
                  off_reason=a.gate_off_reason)
        for s in steps:
            print(f"\n# ---- {s.key} · {s.stage} · {s.steps} steps · "
                  f"{wallclock(s, cfg)['days']} d at {cfg.s_per_step} s/step "
                  f"----")
            print(f"# {s.note}")
            print(launch_line(s, cfg, plan, **kw))
    elif a.cmd == "admission":
        out = {"sel1": SEL1_ADMISSION,
               "sw_latent_pre_registration": SW_LATENT_ADMISSION,
               "current": read_sw_admission(cfg),
               "selector_arms_scheduled": list(cfg.st_arms),
               # ⭐ the emitted path to a tactical selector — a reopening route
               # that only exists in a docstring is one nobody executes.
               "recipe": sw_admission_recipe(cfg),
               "_evidence_class":
                   "MEASURED (E-WC2, REF-C surface, T0-DIAGNOSTIC) + "
                   "PRE-REGISTERED (the S-W latent thresholds, committed "
                   "before that dump is taken)"}
        print(json.dumps(out, indent=1))
    elif a.cmd == "status":
        out = status(cfg)
        print(json.dumps(out, indent=1))
    elif a.cmd == "next":
        out = next_launchable(
            cfg, allow_inconclusive=a.allow_inconclusive_gate,
            off_reason=a.gate_off_reason,
            unpaired_arm_reason=a.unpaired_arm_reason)
        print(json.dumps(out, indent=1))
        if not out["may_launch"]:
            return 3
    elif a.cmd == "manifests":
        dest = Path(a.dest or (_STACK / "ops" / "runs.d"))
        dest.mkdir(parents=True, exist_ok=True)
        written = []
        for s in steps:
            if s.key == "S-W":
                continue          # the live run already has its own supervisor
            p = dest / f"{s.run_id}.env"
            p.write_text(manifest_text(s, cfg, plan), encoding="utf-8")
            written.append(str(p))
        print(json.dumps({"written": written,
                          "_read": "ONE MANIFEST PER STAGE. Never supervise "
                                   "v6_chain.py: supervise_run.sh replays the "
                                   "TRAIN_CMD it captured at startup."},
                         indent=1))
    elif a.cmd == "verify":
        for s in steps:
            print(f"# ---- is {s.key} really running? ----")
            print(verify_probe(s))
    elif a.cmd == "run":
        out = run_chain(cfg, stop_after=a.stop_after,
                        allow_inconclusive=a.allow_inconclusive_gate,
                        off_reason=a.gate_off_reason,
                        unpaired_arm_reason=a.unpaired_arm_reason)
        if a.out_json:
            Path(a.out_json).write_text(json.dumps(out, indent=1),
                                        encoding="utf-8")
        bad = [r for r in out["steps"] if r["returncode"] != 0]
        print(f"\n[chain] {len(out['steps'])} step(s), "
              f"{len(bad)} failure(s), {out['elapsed_s']} s")
        return 1 if bad else 0
    if a.out_json and out is not None and a.cmd != "run":
        Path(a.out_json).write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":                                # pragma: no cover
    raise SystemExit(main())
