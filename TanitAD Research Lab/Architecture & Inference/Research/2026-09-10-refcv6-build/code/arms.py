#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""refcv6 -- the SINGLE SOURCE OF TRUTH for every arm's argv.

⛔ NOTHING here launches anything. This module only BUILDS argv lists. The launch
scripts import it; the one-variable checker imports it; the pre-launch gate imports
it. There is exactly one place an arm's flags are written down, so a launch script
and a pre-registration table cannot drift apart.

# Provenance of BASE  (MEASURED, 2026-09-10)

`BASE_V5V2` below is refcv5-v2's OWN argv, read back verbatim from the banked
`config.json['argv']` of the finished 40,284-step run:

    C:/Users/Admin/refcv5v2_final/config.json          (12,393 bytes, 65 argv tokens)
    also in the rescue tarball C:/Users/Admin/a40-rescue/a40_records.tgz

⚠️ It is NOT hand-transcribed prose. `verify_base_against_banked_config()` re-reads
that JSON and asserts token equality, so a typo here is a loud failure and not a
silently different experiment.

# ⛔ THE ONE DELIBERATE DEVIATION FROM refcv5-v2, AND WHY

`BASE` = `BASE_V5V2` **plus `--agent-join <join>`**, carried by EVERY arm including
the control.

Reason, and it is the WP-D precedent (`PREREG_WPD_BEV_AUX.md` §4) verbatim: *"the
join must be in both arms or the A/B also changes the dataset."* `V3Dataset`
emits `agent_box/yaw/cls/valid/occ/label/ep` per window when the join is enabled
(`refc_v3_train.py:1527-1700`). If only arm A carried the join, arm A would differ
from the control in the DATASET as well as in the lever, invisibly, in every log.

⛔ **CONSEQUENCE, STATED RATHER THAN HIDDEN: `V0` IS NOT A REPLICATION OF
refcv5-v2.** It is a fresh control on the refcv6 BASE at the refcv6 step budget.
Every refcv6 delta is therefore paired against `V0`, never against refcv5-v2's
banked 40,284-step numbers -- pairing across a different BASE *and* a different
step budget would be a two-factor confound. refcv5-v2's banked values are the
PROVENANCE of the non-regression clauses (they say which metrics matter and in
which direction), not their comparison operand. See `PREREG_REFCV6.md` §5B.

# The four levers  (MEASURED present, `stack/scripts/refc_v3_train.py`)

    D  --w-u0 0            lever key `w_u0`         decl :5095   (default U0_WEIGHT_DEFAULT = 0.0, :122)
    A  --agents head       lever key `agents`       decl :5151   (default "off")
    B  --wp-index on       lever key `wp_index`     decl :5340   (default "off")
    C  --w-tac-goal <PI>   lever key `w_tac_goal`   decl :4922   (default 0.0)

# ⛔ WHY ARM A's NAMESPACE DIFF IS TWO KEYS AND NOT ONE

The trainer REFUSES `--agents head` unless `--w-agent > 0`:

    refc_v3_train.py:583
        if args.agents == "head" and float(getattr(args, "w_agent", 0.0)) <= 0.0:
            raise SystemExit("... builds a detector that is never supervised ...
                              the arm would read as 'agent tokens do not help' --
                              a REFUTATION manufactured by a missing loss.")

and `AGENT_WEIGHT_DEFAULT = 0.0` (:121). ⇒ **there exists NO argv in which `agents`
moves alone.** `w_agent` is therefore classified CONSTITUTIVE, not a second lever:
it is the flag without which the named lever cannot be built at all. The checker
(`check_one_variable.py`) allows exactly the constitutive set declared here and
refuses any third key -- so this is an enforced classification, not a note.

(The sibling guard at :591 -- `--agents head` without `--agent-join` has NO LABELS
-- is already satisfied for every arm, because the join is in BASE.)
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Iterable

# --------------------------------------------------------------------------- #
# 1. refcv5-v2's argv, VERBATIM from the banked config.json                    #
# --------------------------------------------------------------------------- #

#: MEASURED 2026-09-10 -- `json.load(open(...))['argv']`, 65 tokens, order preserved.
BASE_V5V2: tuple[str, ...] = (
    "--arm", "hier",
    "--size", "base",
    "--v2-cache", "/root/data/train",
    "--v7-labels", "/workspace/TanitAD/data/s2_labels_v7.2_train.jsonl.gz",
    "--eval-cache", "/root/data/eval",
    "--eval-labels", "/workspace/TanitAD/data/s2_labels_v7.2_eval.jsonl.gz",
    "--eval-every", "500",
    "--eval-batches", "8",
    "--image-hw", "256", "640",
    "--steps", "40284",
    "--batch", "20",
    "--workers", "6",
    "--prefetch-factor", "1",
    "--v2-lru", "24",
    "--lr", "1e-4",
    "--warmup", "2000",
    "--seed", "0",
    "--log-every", "50",
    "--save-every", "500",
    "--u8-batches",
    "--out", "/workspace/experiments/refcv5-v2-noagents-b1-v72-40k",
    "--nav-from-v7",
    "--ego-state-inject",
    "--ego-dropout", "0.5",
    "--anchors", "/workspace/experiments/refcv5-v2-noagents-b1-v72-40k/anchors.pt",
    "--n-anchors", "117",
    "--anchor-v0-conditioned",
    "--anchor-control-units", "alat",
    "--sel-accel-max", "2.0",
    "--sampler", "ddim",
    "--w-u0", "0.5",
    "--sel-refined",
    "--sel-score-emitted",
    "--goal-str",
    "--tac-goal-tok-head",
    "--agents", "off",
)

BANKED_CONFIG_CANDIDATES = (
    r"C:\Users\Admin\refcv5v2_final\config.json",
    "/root/refcv5v2_final/config.json",
    "/workspace/experiments/refcv5-v2-noagents-b1-v72-40k/config.json",
)

# --------------------------------------------------------------------------- #
# 2. Paths that are per-BOX, not per-arm                                       #
# --------------------------------------------------------------------------- #

#: The B1 TRAIN agent join. MEASURED (`D-B1TRAIN-JOIN-1`): 4,427 / 4,572 clips,
#: 849,263 labelled frames, 28,053,187 boxes, md5 1c985e6d6ad34e605c4ebd30cb353558.
#: ⚠️ ~2.1 GB RSS when fully loaded (`JoinFileReader` docstring) -- budget it.
DEFAULT_AGENT_JOIN = "/root/data/joins/train2400_agents.jsonl.xz"

#: ⛔ ONE anchors file for EVERY arm. A per-arm rebuild would make the anchor
#: vocabulary a hidden second lever. The gate md5s it (`prelaunch_gate.py`).
DEFAULT_ANCHORS = "/root/data/refcv6/anchors.pt"

DEFAULT_OUT_ROOT = "/workspace/experiments"

# --------------------------------------------------------------------------- #
# 3. Step budgets                                                              #
# --------------------------------------------------------------------------- #

#: `full` reproduces refcv5-v2's own budget, so a full-budget arm is additionally
#: comparable to the banked 40,284-step numbers (as a SECONDARY read -- see the
#: BASE deviation above). `cut` is the WP-D precedent: the shortest budget past
#: `--warmup 2000` at which the run's own 500-step milestone gate reads a curve.
STEP_BUDGETS = {"full": 40284, "cut": 12000}

# --------------------------------------------------------------------------- #
# 4. The lever algebra                                                          #
# --------------------------------------------------------------------------- #

#: key -> the ONE parsed-namespace key the arm is allowed to move.
LEVER_KEY = {"D": "w_u0", "A": "agents", "B": "wp_index", "C": "w_tac_goal"}

#: key -> parsed-namespace keys that the trainer's own guards FORCE to move with
#: the lever. ⛔ Not "other levers we bundled" -- keys without which the lever
#: cannot be built, each with the file:line of the guard that forces it.
CONSTITUTIVE = {
    "D": (),
    "A": ("w_agent",),      # refc_v3_train.py:583
    "B": (),
    "C": (),
}

CONSTITUTIVE_GUARD = {
    "w_agent": "refc_v3_train.py:583 -- '--agents head with --w-agent 0 builds a "
               "detector that is never supervised ... a REFUTATION manufactured "
               "by a missing loss'",
}

#: Keys that differ between arms by CONSTRUCTION and carry no experimental
#: meaning. ⛔ The checker still reports them; it does not ignore them silently.
BOOKKEEPING_KEYS = frozenset({"out", "seed"})

#: ⛔ NOT bookkeeping: `anchors`. Every arm points at the SAME file, and the gate
#: asserts one md5 across arms. It is listed here so a reviewer can see it was
#: considered and deliberately excluded from the tolerated-difference set.
PARITY_PINNED_KEYS = frozenset({"anchors"})


class PIDecisionRequired(RuntimeError):
    """Raised when an arm needs a value only the PI may choose."""


# --------------------------------------------------------------------------- #
# 5. Building an arm                                                            #
# --------------------------------------------------------------------------- #

def _replace_flag(argv: list[str], flag: str, values: Iterable[str]) -> list[str]:
    """Replace `flag`'s value(s) in-place, preserving token order.

    ⛔ Refuses if the flag is absent -- a silent append would put the new token at
    the END of argv while the pre-registration table shows it in the middle, and
    two 'identical' commands would then diff as strings.
    """
    values = list(values)
    try:
        i = argv.index(flag)
    except ValueError as exc:  # pragma: no cover - defensive
        raise KeyError(
            f"{flag} is not in BASE; refusing to append it silently. "
            f"Add it to BASE explicitly so its position is recorded."
        ) from exc
    n_old = 0
    j = i + 1
    while j < len(argv) and not argv[j].startswith("--"):
        n_old += 1
        j += 1
    if n_old != len(values):
        raise ValueError(
            f"{flag} takes {n_old} value(s) in BASE, got {len(values)}"
        )
    out = list(argv)
    out[i + 1:i + 1 + n_old] = values
    return out


def _append_flag(argv: list[str], flag: str, values: Iterable[str] = ()) -> list[str]:
    return list(argv) + [flag] + [str(v) for v in values]


def base_argv(
    *,
    steps: int,
    out: str,
    anchors: str = DEFAULT_ANCHORS,
    agent_join: str = DEFAULT_AGENT_JOIN,
    seed: int = 0,
) -> list[str]:
    """BASE = refcv5-v2's argv, re-pointed, plus `--agent-join` (see module docstring)."""
    argv = list(BASE_V5V2)
    argv = _replace_flag(argv, "--steps", [str(steps)])
    argv = _replace_flag(argv, "--out", [out])
    argv = _replace_flag(argv, "--anchors", [anchors])
    argv = _replace_flag(argv, "--seed", [str(seed)])
    # ⛔ carried by EVERY arm, control included -- otherwise the A/B also moves the dataset.
    argv = _append_flag(argv, "--agent-join", [agent_join])
    return argv


def arm_argv(
    arm: str,
    *,
    steps: int,
    out_root: str = DEFAULT_OUT_ROOT,
    anchors: str = DEFAULT_ANCHORS,
    agent_join: str = DEFAULT_AGENT_JOIN,
    w_agent: float | None = None,
    w_tac_goal: float | None = None,
    run_tag: str = "refcv6",
) -> list[str]:
    """Return the full argv for one arm.

    Arms
    ----
    ``V0``  control (BASE, seed 0)            ``V0b`` its replicate (seed 1)
    ``D``   ``--w-u0 0``                      ``Db``  replicate
    ``A``   ``--agents head`` (+ w_agent)     ``Ab``  replicate
    ``B``   ``A`` + ``--wp-index on``         ``Bb``  replicate
    ``C``   ``--w-tac-goal <PI>``             ``Cb``  replicate

    ⛔ ``B`` is built ON TOP OF ``A``, not on top of ``V0`` -- ``--wp-index on``
    REFUSES ``--agents off`` by design (`refc_v3_train.py:5340`: *"with no agent
    tokens there is nothing to address and the arm would read as 'the index does
    not help' while never having had one"*). ⇒ B's one-variable comparison is
    **B vs A**, never B vs V0. Enforced by `PAIRING` below.
    """
    if arm not in ARMS:
        raise KeyError(f"unknown arm {arm!r}; known: {sorted(ARMS)}")
    seed = ARMS[arm]["seed"]
    out = f"{out_root}/{run_tag}-{arm}"
    argv = base_argv(steps=steps, out=out, anchors=anchors,
                     agent_join=agent_join, seed=seed)

    lever = ARMS[arm]["lever"]
    if lever is None:
        return argv

    if lever == "D":
        # ⛔ set EXPLICITLY to 0 rather than dropping the flag: the run record must
        # say the operator chose 0, not that a default happened to be 0.
        # (`U0_WEIGHT_DEFAULT` is 0.0, so dropping it would be silently equivalent
        #  and would leave `config.json` unable to distinguish intent from default.)
        return _replace_flag(argv, "--w-u0", ["0"])

    if lever in ("A", "B"):
        if w_agent is None:
            raise PIDecisionRequired(
                "arm A/B needs --w-agent > 0 (refc_v3_train.py:583 refuses "
                "--agents head at w_agent <= 0). No value is hardcoded here: "
                "pass --w-agent explicitly so the record shows who chose it."
            )
        if float(w_agent) <= 0.0:
            raise ValueError("--w-agent must be > 0 for --agents head")
        argv = _replace_flag(argv, "--agents", ["head"])
        argv = _append_flag(argv, "--w-agent", [repr(float(w_agent))])
        if lever == "B":
            argv = _append_flag(argv, "--wp-index", ["on"])
        return argv

    if lever == "C":
        if w_tac_goal is None:
            raise PIDecisionRequired(
                "⛔ --w-tac-goal is a PI DECISION (queue item 10 / `D-TACGOAL-2`) "
                "and is deliberately NOT chosen in this repo. The register records "
                "the two options: a third tactical term at 0.05 either RAISES the "
                "budget to MANEUVER_WEIGHT 0.15 (leaving lat/lon untouched, so the "
                "arm stays paired with the banked one) or forces a /3.0 RE-SPLIT "
                "(which changes lat/lon pressure and is a SEPARATE ARM, not a "
                "tweak). Pass --w-tac-goal once the PI has ruled."
            )
        if float(w_tac_goal) <= 0.0:
            raise ValueError("--w-tac-goal must be > 0 for arm C")
        return _replace_flag(argv, "--w-tac-goal", [repr(float(w_tac_goal))]) \
            if "--w-tac-goal" in argv \
            else _append_flag(argv, "--w-tac-goal", [repr(float(w_tac_goal))])

    raise AssertionError(f"unhandled lever {lever!r}")  # pragma: no cover


#: arm -> {lever, seed, role}. `lever=None` is the control.
ARMS: dict[str, dict] = {
    "V0":  {"lever": None, "seed": 0, "role": "control"},
    "V0b": {"lever": None, "seed": 1, "role": "replicate of V0"},
    "D":   {"lever": "D",  "seed": 0, "role": "lever: --w-u0 0"},
    "Db":  {"lever": "D",  "seed": 1, "role": "replicate of D"},
    "A":   {"lever": "A",  "seed": 0, "role": "lever: --agents head"},
    "Ab":  {"lever": "A",  "seed": 1, "role": "replicate of A"},
    "B":   {"lever": "B",  "seed": 0, "role": "lever: --wp-index on (on top of A)"},
    "Bb":  {"lever": "B",  "seed": 1, "role": "replicate of B"},
    "C":   {"lever": "C",  "seed": 0, "role": "lever: --w-tac-goal <PI>"},
    "Cb":  {"lever": "C",  "seed": 1, "role": "replicate of C"},
}

#: arm -> the arm it is compared against. ⛔ B pairs with A, not with V0.
#: A replicate pairs with its own treatment, and that pair MEASURES THE NOISE
#: FLOOR -- it is not a lever comparison.
PAIRING: dict[str, str] = {
    "D": "V0", "A": "V0", "C": "V0",
    "B": "A",
    "V0b": "V0", "Db": "D", "Ab": "A", "Bb": "B", "Cb": "C",
}

#: The arms whose pair is a NOISE-FLOOR read, not a lever read.
REPLICATE_ARMS = frozenset({"V0b", "Db", "Ab", "Bb", "Cb"})


# --------------------------------------------------------------------------- #
# 6. Verification against the banked record                                    #
# --------------------------------------------------------------------------- #

def verify_base_against_banked_config(path: str | None = None) -> dict:
    """Assert `BASE_V5V2` equals the banked run's own `config.json['argv']`.

    ⛔ Returns a verdict dict; NEVER returns a bare bool that a caller could read
    through a pipe. `ok` is only True when the file was actually READ and the
    tokens actually MATCHED -- a file that could not be opened reports
    `INCONCLUSIVE`, never agreement. (The empty-string blob-comparison hole:
    both operands failing identically must not read as a match.)
    """
    candidates = [path] if path else list(BANKED_CONFIG_CANDIDATES)
    for cand in candidates:
        if not cand or not os.path.isfile(cand):
            continue
        try:
            with open(cand, "r", encoding="utf-8") as fh:
                cfg = json.load(fh)
        except Exception as exc:
            return {"verdict": "INCONCLUSIVE", "ok": False, "path": cand,
                    "reason": f"unreadable: {exc.__class__.__name__}: {exc}"}
        banked = cfg.get("argv")
        if not isinstance(banked, list) or not banked:
            return {"verdict": "INCONCLUSIVE", "ok": False, "path": cand,
                    "reason": "config.json carries no non-empty 'argv'"}
        same = list(banked) == list(BASE_V5V2)
        return {
            "verdict": "MATCH" if same else "MISMATCH",
            "ok": bool(same),
            "path": cand,
            "n_banked": len(banked),
            "n_here": len(BASE_V5V2),
            "first_difference": None if same else next(
                (i for i, (x, y) in enumerate(zip(banked, BASE_V5V2)) if x != y),
                min(len(banked), len(BASE_V5V2)),
            ),
        }
    return {"verdict": "INCONCLUSIVE", "ok": False, "path": None,
            "reason": f"none of {candidates} exists on this box"}


def _cli() -> int:
    ap = argparse.ArgumentParser(description="print one refcv6 arm's argv")
    ap.add_argument("arm", choices=sorted(ARMS))
    ap.add_argument("--budget", default="full", choices=sorted(STEP_BUDGETS))
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    ap.add_argument("--anchors", default=DEFAULT_ANCHORS)
    ap.add_argument("--agent-join", default=DEFAULT_AGENT_JOIN)
    ap.add_argument("--w-agent", type=float, default=None)
    ap.add_argument("--w-tac-goal", type=float, default=None)
    ap.add_argument("--verify-base", action="store_true")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    if a.verify_base:
        print(json.dumps(verify_base_against_banked_config(), indent=2))

    steps = a.steps if a.steps is not None else STEP_BUDGETS[a.budget]
    argv = arm_argv(a.arm, steps=steps, out_root=a.out_root, anchors=a.anchors,
                    agent_join=a.agent_join, w_agent=a.w_agent,
                    w_tac_goal=a.w_tac_goal)
    if a.json:
        print(json.dumps({"arm": a.arm, "steps": steps, "argv": argv}, indent=2))
    else:
        print(" ".join(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
