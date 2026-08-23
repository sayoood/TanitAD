"""The size comparison, normalised — apples to apples, after the fact.

`measure_options.py` sized the two options BEFORE either was written, in physical
line spans. This re-measures both on the SAME footing once option A actually
exists, because a physical-line count flatters whichever side has more comments
and this codebase comments heavily on purpose.

Metric: **non-blank, non-comment, non-docstring lines** ("code lines").

  * Option A — the REAL diff of the chosen implementation (`git diff` + the new
    test file), split into product code and tests.
  * Option B — the code lines of the machinery `train_flagship4b` would have to
    ACQUIRE to run the mid-run held-out gate, because its selected plan is a
    4-point 0.5 s trajectory and pseudo-simulation refuses a non-dense plan.

⚠️ Option B's number is a FLOOR, not an estimate: it counts only existing
definitions that would have to move, and none of the wiring, tests or the second
optimizer group they would need.

Run:  python measure_diff.py --out ../raw/diff_size_<date>.json
"""
from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[6]
STACK = REPO / "stack"


def _code_lines(text: str) -> int:
    """Non-blank, non-comment, non-docstring lines."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        tree = None
    doc_spans: set[int] = set()
    if tree is not None:
        for node in ast.walk(tree):
            body = getattr(node, "body", None)
            # ⚠️ `.body` is a LIST on modules/defs/classes but a single node on
            # IfExp/Lambda — indexing it there raises. Guard on the type, not on
            # truthiness (this bit once, on FlagshipV15Head).
            if not isinstance(body, list) or not body or \
                    not isinstance(body[0], ast.Expr):
                continue
            v = body[0].value
            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                doc_spans |= set(range(v.lineno, (v.end_lineno or v.lineno) + 1))
    n = 0
    for i, ln in enumerate(text.splitlines(), 1):
        s = ln.strip()
        if not s or s.startswith("#") or i in doc_spans:
            continue
        n += 1
    return n


def _added_lines(path: str) -> list[str]:
    # ⚠️ `HEAD` explicitly, not a bare `git diff`: these deliverables are STAGED
    # (per the operating standard), and a bare `git diff` compares the worktree
    # against the INDEX — which returns 0 added lines once the work is staged,
    # silently reporting the change as empty. Measured the hard way.
    raw = subprocess.run(["git", "diff", "-U0", "HEAD", "--", path],
                         cwd=str(REPO), capture_output=True).stdout
    out = raw.decode("utf-8", "replace")
    return [ln[1:] for ln in out.splitlines()
            if ln.startswith("+") and not ln.startswith("+++")]


def _def_code_lines(path: Path, name: str) -> int:
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)) and node.name == name:
            seg = "\n".join(src.splitlines()[node.lineno - 1:node.end_lineno])
            return _code_lines(seg)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    a_prod = {}
    for rel in ("stack/scripts/train_flagship_v4.py",
                "stack/tanitad/data/parity.py"):
        added = _added_lines(rel)
        a_prod[rel] = {"added_physical": len(added),
                       "added_code_lines": _code_lines("\n".join(
                           ln.lstrip() for ln in added))}
    test_f = STACK / "tests" / "test_v5_trainer_v2_val.py"
    test_src = test_f.read_text(encoding="utf-8")

    b_need = {
        "tanitad/models/flagship_v4.py::FlagshipV4Head": _def_code_lines(
            STACK / "tanitad" / "models" / "flagship_v4.py", "FlagshipV4Head"),
        "tanitad/models/flagship_v15.py::FlagshipV15Head": _def_code_lines(
            STACK / "tanitad" / "models" / "flagship_v15.py", "FlagshipV15Head"),
        "scripts/train_flagship_v4.py::v4_loss_step": _def_code_lines(
            STACK / "scripts" / "train_flagship_v4.py", "v4_loss_step"),
        "scripts/flagship_v4_data.py::FlagshipV4Dataset": _def_code_lines(
            STACK / "scripts" / "flagship_v4_data.py", "FlagshipV4Dataset"),
        "scripts/train_flagship_v4.py::_training_loop": _def_code_lines(
            STACK / "scripts" / "train_flagship_v4.py", "_training_loop"),
        "scripts/train_flagship_v4.py::canary_rollout": _def_code_lines(
            STACK / "scripts" / "train_flagship_v4.py", "canary_rollout"),
        "scripts/train_flagship_v4.py::evaluate_planner": _def_code_lines(
            STACK / "scripts" / "train_flagship_v4.py", "evaluate_planner"),
    }

    # ⚠️ A raw "lines added" number flatters option B, because most of option A's
    # diff is NOT the v2 wiring — it is the geometry binding, the train/val leak
    # guard and the legible missing-manifest refusal, every one of which option B
    # would ALSO have to write. Split it, or the comparison is rhetoric.
    CORE = {                      # what "teach v4 to read v2" actually costs
        ("scripts/train_flagship_v4.py", "assert_corpus_args"),
        ("scripts/train_flagship_v4.py", "_assert_parity_v2"),
    }
    STRENGTHENING = {             # required by the brief, needed on EITHER path
        ("tanitad/data/parity.py", "assert_v2_splits_disjoint"),
        ("tanitad/data/parity.py", "assert_v2_geometry_matches"),
        ("tanitad/data/parity.py", "_missing_entry_lines"),
        ("tanitad/data/parity.py", "_sibling_candidate_key"),
        ("scripts/train_flagship_v4.py", "_geometry_report"),
    }
    split = {"core_v2_wiring": {}, "strengthening_needed_on_either_path": {}}
    for group, keys in (("core_v2_wiring", CORE),
                        ("strengthening_needed_on_either_path", STRENGTHENING)):
        for rel, name in sorted(keys):
            split[group][f"{rel}::{name}"] = _def_code_lines(STACK / rel, name)

    rec = {
        "metric": "non-blank, non-comment, non-docstring lines",
        "option_a_new_definitions_by_purpose": {
            **{k: {"defs": v, "total": sum(v.values())}
               for k, v in split.items()},
            "note": "the remainder of the diff is in-place branches (the data "
                    "block, the geometry call, the CLI flags, preflight, the "
                    "staged command), which no separate definition owns.",
        },
        "option_a_CHOSEN": {
            "what": "teach train_flagship_v4 to read v2 + bind the geometry",
            "product": a_prod,
            "product_code_lines_total": sum(v["added_code_lines"]
                                            for v in a_prod.values()),
            "tests": {"file": "stack/tests/test_v5_trainer_v2_val.py",
                      "physical": len(test_src.splitlines()),
                      "code_lines": _code_lines(test_src)},
            "NEW_MACHINERY": 0,
            "new_machinery_note": (
                "every capability it uses already existed: build_v2_providers "
                "(the v2 loader), assert_v2_parity_cache (the membership "
                "proof), apply_geometry_args (the frame), HeldoutGate (the "
                "mid-run gate). The diff is wiring and refusals."),
        },
        "option_b_REJECTED": {
            "what": "port the held-out val loop into train_flagship4b",
            "machinery_it_must_acquire": b_need,
            "code_lines_total": sum(b_need.values()),
            "floor_not_estimate": (
                "counts only existing definitions that would have to move — "
                "not the wiring, the tests, the second optimizer group, or the "
                "v2 val-cache branch it ALSO needs (which is option A's work "
                "done twice)."),
            "why_it_is_not_a_port": (
                "train_flagship4b's selected plan is 4 points at 0.5 s spacing "
                "(tactical waypoint_horizons (5,10,15,20)); the gate's "
                "pseudo-simulation differentiates at dt=0.1 s and raises "
                "NonDensePlanError on it. A dense plan means FlagshipV4Head and "
                "the loss that trains it — i.e. making 4b into v4."),
        },
    }
    rec["ratio_new_machinery"] = (
        f"{rec['option_b_REJECTED']['code_lines_total']} : 0")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(json.dumps(rec, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
