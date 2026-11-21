#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""refcv6 -- ⛔ ONE LEVER PER ARM, asserted on PARSED NAMESPACES.

    python check_one_variable.py --budget full --w-agent <W> [--w-tac-goal <W>] \
        --json raw/one_variable.json

# Why this exists

*"Diffed as parsed namespaces, not as intent."* A past sweep was invalidated
because an arm silently changed the effective lambda alongside its named lever.
Diffing the argv STRINGS cannot catch that: two different strings can parse to the
same namespace (a flag at its default) and, far worse, the same-looking string can
parse differently once a `dest=`, a `type=` coercion or a mutually-exclusive group
is involved.

⛔ **This module does NOT reimplement the trainer's parser.** It imports
`refc_v3_train.build_parser()` -- the real one, the one the launch will use. A
checker with its own copy of the flags is *a check that shares the defect it checks
for*: it would go green against a trainer whose parser had moved underneath it.

# What it asserts, per arm pair

For `PAIRING[arm] -> base`, every key whose parsed value differs is classified:

    LEVER         the ONE key the arm is declared to move          (must be exactly 1)
    CONSTITUTIVE  a key the trainer's OWN GUARD forces to move      (allow-listed per
                  with the lever -- not a second lever               lever, with file:line)
    BOOKKEEPING   `out`, `seed` -- differ by construction           (reported, not ignored)
    VIOLATION     anything else                                     ⇒ ⛔ REFUSE

⭐ The discriminating half, and the one a positive-only check would miss: it ALSO
asserts the lever key's value actually MOVED to the declared target. A pair that
differs in exactly one key is not evidence the lever is on -- the key could have
moved the wrong way, or to the default. `expected_from`/`expected_to` are literals
here, never expressions over the code under test.

# ⛔ Exit code is NOT the verdict

The verdict is the JSON artifact's `"verdict"` field. `$?` after a pipe reports the
LAST element's status (`cmd | tail` reports tail's), which manufactured four false
successes in this programme in two days. Callers MUST read the file.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import traceback

import arms as A  # the single source of truth for every arm's argv

# --------------------------------------------------------------------------- #

#: For each lever, what the key must move FROM and TO. ⛔ Written as LITERALS,
#: never as an expression over the trainer's own defaults -- an expectation
#: computed from the code under test measures determinism, not correctness.
EXPECTED_MOVE = {
    "D": ("w_u0", 0.5, 0.0),
    "A": ("agents", "off", "head"),
    "B": ("wp_index", "off", "on"),
    # C's TO value is a PI decision; only the FROM value and the direction
    # (> 0.0) are pre-registered. Checked as "moved from 0.0 to something > 0".
    "C": ("w_tac_goal", 0.0, None),
}


def load_trainer(trainer_path: str):
    """Import the REAL `refc_v3_train` module from an explicit path.

    Returns `(module, None)` or `(None, reason)`. ⛔ Never raises past the caller:
    an import failure must become an INCONCLUSIVE verdict in the artifact, not a
    traceback that a wrapper script reads as "the check did not find problems".
    """
    if not os.path.isfile(trainer_path):
        return None, f"trainer not found at {trainer_path}"
    try:
        spec = importlib.util.spec_from_file_location("refc_v3_train", trainer_path)
        if spec is None or spec.loader is None:
            return None, f"could not build an import spec for {trainer_path}"
        mod = importlib.util.module_from_spec(spec)
        sys.modules["refc_v3_train"] = mod
        spec.loader.exec_module(mod)
    except Exception as exc:
        return None, (f"import failed: {exc.__class__.__name__}: {exc}\n"
                      + traceback.format_exc(limit=6))
    if not hasattr(mod, "build_parser"):
        return None, "module imported but has no build_parser()"
    return mod, None


def parse_arm(parser_factory, argv: list[str]) -> tuple[dict | None, str | None]:
    """Parse one arm's argv through the trainer's own parser.

    A fresh parser per arm: argparse parsers hold mutable state (`set_defaults`,
    seen-actions), so reusing one across arms can leak a value between them --
    which would be this checker committing the exact defect it exists to catch.
    """
    ap = parser_factory()
    try:
        ns = ap.parse_args(argv)
    except SystemExit as exc:            # argparse exits on a bad flag
        return None, f"argparse refused this argv (exit {exc.code})"
    except Exception as exc:
        return None, f"{exc.__class__.__name__}: {exc}"
    out = {}
    for k, v in sorted(vars(ns).items()):
        if k.startswith("_"):
            continue
        out[k] = list(v) if isinstance(v, (list, tuple)) else v
    return out, None


def diff_namespaces(a: dict, b: dict) -> dict[str, tuple]:
    keys = set(a) | set(b)
    sentinel = object()
    return {
        k: (a.get(k, sentinel), b.get(k, sentinel))
        for k in sorted(keys)
        if a.get(k, sentinel) != b.get(k, sentinel)
    }


def check_pair(arm: str, base: str, ns_arm: dict, ns_base: dict) -> dict:
    lever_id = A.ARMS[arm]["lever"]
    base_lever_id = A.ARMS[base]["lever"]
    is_replicate = arm in A.REPLICATE_ARMS

    lever_key = A.LEVER_KEY.get(lever_id) if lever_id else None
    # A replicate moves NO lever -- only the seed. B pairs with A, and both
    # carry the A lever, so A's lever key is not expected to differ there.
    if is_replicate or lever_id == base_lever_id:
        expected_lever_keys: set[str] = set()
        constitutive: set[str] = set()
    else:
        expected_lever_keys = {lever_key} if lever_key else set()
        constitutive = set(A.CONSTITUTIVE.get(lever_id, ()))
        # B is built on A; A's constitutive keys are already in the base.
        if base_lever_id:
            constitutive -= set(A.CONSTITUTIVE.get(base_lever_id, ()))

    d = diff_namespaces(ns_arm, ns_base)
    classified: dict[str, str] = {}
    for k in d:
        if k in expected_lever_keys:
            classified[k] = "LEVER"
        elif k in constitutive:
            classified[k] = "CONSTITUTIVE"
        elif k in A.BOOKKEEPING_KEYS:
            classified[k] = "BOOKKEEPING"
        elif k in A.PARITY_PINNED_KEYS:
            # ⛔ anchors must be the SAME file across arms; a difference here is a
            # hidden second lever (a different anchor vocabulary), never bookkeeping.
            classified[k] = "VIOLATION_PARITY"
        else:
            classified[k] = "VIOLATION"

    violations = {k: v for k, v in classified.items() if v.startswith("VIOLATION")}
    n_lever = sum(1 for v in classified.values() if v == "LEVER")

    problems: list[str] = []
    if violations:
        problems.append(
            "unexpected keys differ: "
            + ", ".join(f"{k} ({classified[k]}): {d[k][1]!r} -> {d[k][0]!r}"
                        for k in sorted(violations))
        )
    if len(expected_lever_keys) != n_lever:
        problems.append(
            f"expected exactly {len(expected_lever_keys)} LEVER key(s), found {n_lever}"
        )

    # ⭐ The discriminating control: the lever must have moved to the DECLARED
    # target. Differing in one key is not evidence the lever is ON.
    move_check = None
    if lever_id and expected_lever_keys:
        key, want_from, want_to = EXPECTED_MOVE[lever_id]
        got_arm, got_base = ns_arm.get(key), ns_base.get(key)
        ok_from = (got_base == want_from)
        ok_to = (got_arm > 0.0 if want_to is None else got_arm == want_to)
        move_check = {
            "key": key, "expected_from": want_from, "expected_to": want_to,
            "got_from": got_base, "got_to": got_arm,
            "from_ok": bool(ok_from), "to_ok": bool(ok_to),
        }
        if not ok_from:
            problems.append(f"{key}: base is {got_base!r}, pre-registered "
                            f"starting value is {want_from!r}")
        if not ok_to:
            problems.append(f"{key}: arm is {got_arm!r}, pre-registered "
                            f"target is {want_to!r}")

    # A replicate must move the seed and NOTHING else that matters.
    if is_replicate and ns_arm.get("seed") == ns_base.get("seed"):
        problems.append("replicate arm carries the SAME seed as its treatment -- "
                        "it would measure nothing")

    return {
        "arm": arm, "base": base, "role": A.ARMS[arm]["role"],
        "is_replicate": is_replicate,
        "n_keys_differing": len(d),
        "classified": classified,
        "differences": {k: {"arm": _j(d[k][0]), "base": _j(d[k][1])} for k in d},
        "expected_lever_keys": sorted(expected_lever_keys),
        "constitutive_keys": sorted(constitutive),
        "constitutive_guards": {k: A.CONSTITUTIVE_GUARD.get(k, "")
                                for k in sorted(constitutive)},
        "lever_move_check": move_check,
        "problems": problems,
        "verdict": "PASS" if not problems else "REFUSE",
    }


def _j(v):
    """JSON-safe, and never lets a sentinel leak into the artifact."""
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, (list, tuple)):
        return [_j(x) for x in v]
    return "<ABSENT>" if type(v).__name__ == "object" else repr(v)


def main() -> int:
    ap = argparse.ArgumentParser(description="refcv6 one-variable check")
    ap.add_argument("--trainer", default=None,
                    help="path to refc_v3_train.py (default: found next to "
                         "$TANITAD_STACK or ../../../../stack/scripts)")
    ap.add_argument("--budget", default="full", choices=sorted(A.STEP_BUDGETS))
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--out-root", default=A.DEFAULT_OUT_ROOT)
    ap.add_argument("--anchors", default=A.DEFAULT_ANCHORS)
    ap.add_argument("--agent-join", default=A.DEFAULT_AGENT_JOIN)
    ap.add_argument("--w-agent", type=float, default=None)
    ap.add_argument("--w-tac-goal", type=float, default=None)
    ap.add_argument("--arms", default=None,
                    help="comma-separated subset; default = every arm that can "
                         "be built with the values supplied")
    ap.add_argument("--json", default=None, help="write the verdict artifact here")
    a = ap.parse_args()

    trainer = a.trainer
    if trainer is None:
        here = os.path.dirname(os.path.abspath(__file__))
        for cand in (
            os.environ.get("TANITAD_TRAINER", ""),
            os.path.join(os.environ.get("TANITAD_STACK", ""), "scripts", "refc_v3_train.py"),
            os.path.abspath(os.path.join(here, "..", "..", "..", "..", "..",
                                         "stack", "scripts", "refc_v3_train.py")),
            "/workspace/TanitAD/stack/scripts/refc_v3_train.py",
        ):
            if cand and os.path.isfile(cand):
                trainer = cand
                break

    report = {
        "tool": "check_one_variable.py",
        "trainer_path": trainer,
        "budget": a.budget,
        "steps": a.steps if a.steps is not None else A.STEP_BUDGETS[a.budget],
        "base_provenance": A.verify_base_against_banked_config(),
        "pairs": [],
        "arms_skipped": {},
        "verdict": "INCONCLUSIVE",
    }

    mod, why = load_trainer(trainer or "")
    if mod is None:
        report["verdict"] = "INCONCLUSIVE"
        report["reason"] = (
            f"could not import the trainer ({why}). ⛔ This is NOT a pass: the "
            f"namespaces were never parsed. Set --trainer or PYTHONPATH="
            f"<stack> and re-run."
        )
        _emit(report, a.json)
        return 3

    steps = report["steps"]
    want = set(a.arms.split(",")) if a.arms else set(A.ARMS)

    namespaces: dict[str, dict] = {}
    for arm in sorted(A.ARMS):
        if arm not in want:
            continue
        try:
            argv = A.arm_argv(arm, steps=steps, out_root=a.out_root,
                              anchors=a.anchors, agent_join=a.agent_join,
                              w_agent=a.w_agent, w_tac_goal=a.w_tac_goal)
        except A.PIDecisionRequired as exc:
            report["arms_skipped"][arm] = f"PI_DECISION_REQUIRED: {exc}"
            continue
        except Exception as exc:
            report["arms_skipped"][arm] = f"{exc.__class__.__name__}: {exc}"
            continue
        ns, err = parse_arm(mod.build_parser, argv)
        if ns is None:
            report["arms_skipped"][arm] = f"PARSE_FAILED: {err}"
            continue
        namespaces[arm] = ns
        report.setdefault("argv", {})[arm] = argv

    for arm in sorted(namespaces):
        base = A.PAIRING.get(arm)
        if base is None or base not in namespaces:
            continue
        report["pairs"].append(check_pair(arm, base, namespaces[arm], namespaces[base]))

    checked = [p for p in report["pairs"]]
    if not checked:
        report["verdict"] = "INCONCLUSIVE"
        report["reason"] = "no arm pair could be built and parsed"
    elif any(p["verdict"] == "REFUSE" for p in checked):
        report["verdict"] = "REFUSE"
    else:
        report["verdict"] = "PASS"

    report["summary"] = {
        "n_pairs_checked": len(checked),
        "n_refused": sum(1 for p in checked if p["verdict"] == "REFUSE"),
        "n_arms_skipped": len(report["arms_skipped"]),
    }
    _emit(report, a.json)
    return 0 if report["verdict"] == "PASS" else 1


def _emit(report: dict, path: str | None) -> None:
    text = json.dumps(report, indent=2, ensure_ascii=False, default=str)
    if path:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
        print(f"[one-variable] wrote {path}")
    print(f"[one-variable] verdict = {report['verdict']}")
    for p in report.get("pairs", []):
        mark = "OK " if p["verdict"] == "PASS" else "REFUSE"
        keys = ", ".join(f"{k}={v}" for k, v in sorted(p["classified"].items()))
        print(f"  [{mark}] {p['arm']:>3} vs {p['base']:<3} "
              f"({p['n_keys_differing']} key(s)): {keys}")
        for prob in p["problems"]:
            print(f"          ⛔ {prob}")
    for arm, why in sorted(report.get("arms_skipped", {}).items()):
        print(f"  [skip ] {arm}: {why.splitlines()[0][:120]}")


if __name__ == "__main__":
    raise SystemExit(main())
