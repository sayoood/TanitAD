"""⛔ EVERY GUARD IN THIS PACKAGE, PROVEN BY MUTATION — never by inspection.

A green test proves nothing about a guard: it is equally green on code that
cannot fail. This reintroduces each defect the patch closes, ONE AT A TIME, and
asserts the named test **goes RED for the right reason**. A mutation whose test
stays GREEN is reported as ``BLIND`` — a guard that is a decoration.

⭐ THE CONTROL THAT MAKES THE TABLE READABLE. Before any mutation, the same
tests are run UNMUTATED and must all pass. Measured on a companion package: an
argv typo made every "guard fires" row fire for the wrong reason, and a RED-only
table could not tell a guard from a brick.

⚠️ IT EDITS THE WORKTREE AND RESTORES IT. Every original is held in memory and
written back in a ``finally``, and the restoration is VERIFIED by re-reading the
bytes — an exit code is not evidence that a file came back.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

#: (name, file, old, new, test, expected-substring-in-failure)
#: ⛔ Each `old` is a distinctive slice of the REAL fix; `new` is the state the
#: tip was in, or the obvious wrong thing. A mutation that does not APPLY is
#: reported as NOT-APPLIED and is a failure of this harness, never a pass.
MUTATIONS = [
    (
        "feed_gate_removed",
        "stack/tanitad/refs/refc_v3.py",
        '            if not _feed:\n'
        '                pout = dict(pout)\n'
        '                pout.pop("bev_tokens", None)\n'
        '                pout["bev_tokens_fed"] = False\n'
        '                return pout\n',
        '            if False:\n'
        '                pout = dict(pout)\n'
        '                pout.pop("bev_tokens", None)\n'
        '                pout["bev_tokens_fed"] = False\n'
        '                return pout\n',
        "test_GREEN_bev_tokens_REACH_the_behaviour_decoder",
        "d_bev = 0",
    ),
    (
        "declared_bev_without_branch_guard_removed",
        "stack/tanitad/refs/refc_v3.py",
        "        if _dbev_cfg > 0 and _bh is None:",
        "        if False and _dbev_cfg > 0 and _bh is None:",
        "test_MUTATION_a_declared_bev_source_with_NO_branch_REFUSES",
        "DID NOT RAISE",
    ),
    (
        "bev_hook_never_passed_to_core",     # ⭐ THE TIP'S OWN STATE
        "stack/tanitad/refs/refc_v3.py",
        '        if _bh is not None:\n            _core_kw["bev_hook"] = _bh',
        '        if False:\n            _core_kw["bev_hook"] = _bh',
        "test_GREEN_bev_tokens_REACH_the_behaviour_decoder",
        "",
    ),
    (
        "detach_flag_ignored",
        "stack/tanitad/refs/refc_v3.py",
        '            if bool(getattr(self.cfg, "tac_decoder_bev_detach", False)) \\\n'
        '                    and "bev_tokens" in pout:',
        '            if False \\\n'
        '                    and "bev_tokens" in pout:',
        "test_R3_the_tactical_loss_REACHES_the_bev_branch_and_the_detach_STOPS_it",
        "did not cut the path",
    ),
    (
        "flat_arm_refusal_removed",
        "stack/tanitad/refs/refc_v3.py",
        '            if getattr(self, "_perception", None) is not None:',
        '            if False:',
        "test_MUTATION_a_perception_branch_on_the_FLAT_arm_REFUSES",
        "DID NOT RAISE",
    ),
    (
        "two_suppliers_guard_removed",
        "stack/tanitad/refs/refc.py",
        "            if bev_tokens is not None:\n"
        "                raise ValueError(\n"
        '                    "refcv6: BOTH `bev_hook` and `bev_tokens` were supplied. "',
        "            if False:\n"
        "                raise ValueError(\n"
        '                    "refcv6: BOTH `bev_hook` and `bev_tokens` were supplied. "',
        "test_bev_hook_and_explicit_bev_tokens_TOGETHER_are_refused",
        "DID NOT RAISE",
    ),
    (
        "trainer_d_bev_precondition_removed",
        "stack/scripts/refc_v3_train.py",
        "            if _wmap <= 0.0:",
        "            if False:",
        "test_RED_each_dead_configuration_refuses_for_its_own_reason",
        "",
    ),
    (
        "stamp_hardcoded_false_again",       # ⭐ the tip's literal
        "stack/scripts/refc_v3_train.py",
        '            "bev_tokens_reach_decoder": bool(\n'
        '                getattr(cfg, "tac_decoder_v6", False)\n'
        '                and int(getattr(cfg.tac_decoder_cfg, "d_bev", 0) or 0) > 0),',
        '            "bev_tokens_reach_decoder": False,',
        "test_the_bev_seam_stamp_is_the_ARM_S_OWN_answer_not_a_constant",
        "",
    ),
    (
        "conflict_detector_loses_the_tactical_term",
        "stack/scripts/refc_v3_train.py",
        '    ("tac_v6", lambda m: float(getattr(m, "_w_tac_v6", 0.0))),'
        '    # refcv6 §4\n',
        '',
        "test_the_conflict_detector_counts_the_TACTICAL_term_as_an_aux_gradient",
        "aux table",
    ),
    (
        "tactical_loss_not_exposed_to_the_detector",
        "stack/scripts/refc_v3_train.py",
        '        extra["tac_v6"] = _t6_loss\n',
        '',
        "test_every_conflict_term_HAS_A_PRODUCER_in_the_loss_dict",
        "NO producer",
    ),
]

TESTFILES = ("tests/test_refcv6_bev_tactical_wiring.py",
             "tests/test_refcv6_tactical_training.py")


def _pytest(root: Path, node: str) -> tuple[bool, str]:
    """-> (passed, output). ⛔ Never `$?` through a pipe: `capture_output` reads
    the process's own status."""
    f = next((t for t in TESTFILES
              if node in (root / "stack" / t).read_text(encoding="utf-8")),
             TESTFILES[0])
    p = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
         f"{f}::{node}"],
        cwd=str(root / "stack"), capture_output=True, text=True,
        encoding="utf-8", errors="replace")
    return p.returncode == 0, (p.stdout or "") + (p.stderr or "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="the worktree root")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root = Path(a.root)

    rows: list[dict] = []
    # ---- THE CONTROL: unmutated, every named test must PASS --------------- #
    control = {}
    for name in sorted({m[4] for m in MUTATIONS}):
        ok, _ = _pytest(root, name)
        control[name] = ok
    if not all(control.values()):
        Path(a.out).write_text(json.dumps(
            {"verdict": "INCONCLUSIVE",
             "why": "a named test fails BEFORE any mutation, so every RED below "
                    "would be unreadable — a brick, not a guard",
             "control": control}, indent=2), encoding="utf-8")
        print(json.dumps(control, indent=2))
        return 1

    originals: dict[str, str] = {}
    try:
        for name, rel, old, new, node, needle in MUTATIONS:
            p = root / rel
            src = originals.setdefault(rel, p.read_text(encoding="utf-8"))
            n_hits = src.count(old)
            if n_hits != 1:
                rows.append({"mutation": name, "file": rel,
                             "verdict": "NOT-APPLIED",
                             "why": f"the anchor occurs {n_hits} times, not 1"})
                continue
            p.write_text(src.replace(old, new), encoding="utf-8")
            ok, out = _pytest(root, node)
            p.write_text(src, encoding="utf-8")          # restore immediately
            rows.append({
                "mutation": name, "file": rel, "test": node,
                "test_passed_under_mutation": ok,
                "reason_matched": (bool(needle) and needle in out) or not needle,
                "verdict": ("BLIND" if ok else "DETECTED"),
                "tail": out.strip().splitlines()[-1] if out.strip() else ""})
    finally:
        for rel, src in originals.items():
            (root / rel).write_text(src, encoding="utf-8")
        # ⛔ VERIFY THE RESTORATION BY READING THE BYTES BACK. An exit code is
        # not evidence that a file came back; a half-restored worktree would
        # make every later result in this session a lie.
        restored = {rel: (root / rel).read_text(encoding="utf-8") == src
                    for rel, src in originals.items()}

    doc = {"control": control, "mutations": rows, "restored": restored,
           "verdict": {
               "all_named_tests_pass_unmutated": all(control.values()),
               "n_mutations": len(rows),
               "n_detected": sum(1 for r in rows if r["verdict"] == "DETECTED"),
               "blind": [r["mutation"] for r in rows if r["verdict"] == "BLIND"],
               "not_applied": [r["mutation"] for r in rows
                               if r["verdict"] == "NOT-APPLIED"],
               "worktree_restored": all(restored.values())}}
    Path(a.out).write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(json.dumps(doc["verdict"], indent=2))
    return 0 if not doc["verdict"]["blind"] \
        and not doc["verdict"]["not_applied"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
