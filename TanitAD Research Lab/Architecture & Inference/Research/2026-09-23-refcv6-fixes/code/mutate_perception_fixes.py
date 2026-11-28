"""The deliberate-regression arms: reintroduce each MEASURED defect and require RED.

⛔⛔ **WHY A FIX WITHOUT THIS IS NOT A FIX.** A test that passes against the fixed code
proves only that the fixed code is self-consistent. The question a guard must answer is
*"would this have caught the defect?"*, and the only admissible answer is a run in which
the defect is back and the guard is RED. Four instances in one night (CLAUDE.md,
2026-09-07) were green forever because the check shared the defect it checked for.

⛔ **ANCHORS ARE WHOLE LINES MATCHED BY EQUALITY** (after stripping the line terminator),
never substrings. These files are CRLF, and a substring anchor collides across indentation
levels -- ``m = match or match_slots(pred, tgt)`` appears once here and its shape appears in
three other modules.

⛔ **A NON-APPLYING ARM ABORTS AS INVALID, NEVER AS A PASS.** An anchor that matches zero
lines, or more than one, means the source moved; the arm then reports ``INVALID`` and the
run's exit code is non-zero. A mutation harness that silently skips is the `grep -c` hole:
"0 hits" from a file that could not be read is indistinguishable from a genuine absence.

⭐ **AND EACH ARM NAMES THE TESTS THAT MUST GO RED AND THE ONES THAT MUST STAY GREEN.**
The second half is the discriminating control: a mutation that reddens the WHOLE suite has
proved nothing about the specific guard.

Run (CPU, ~2 min):
  PYTHONPATH=D:/Projects/TanitAD/stack python mutate_perception_fixes.py
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time

REPO = pathlib.Path(__file__).resolve()
while REPO.name != "TanitAD" or not (REPO / "stack").is_dir():
    REPO = REPO.parent
HERE = pathlib.Path(__file__).resolve().parent
RAW = HERE.parent / "raw"
TESTS = "stack/tests/test_refcv6_perception_supervision_fixes.py"

BOX = "stack/tanitad/models/box3d_head.py"
BRANCH = "stack/tanitad/models/refcv6_perception_branch.py"
SLOTS = "stack/tanitad/models/agent_slots.py"
ENC = "stack/tanitad/models/bev_encoder.py"

#: ``id -> (file, anchor line, replacement line(s), must-go-RED, must-stay-GREEN)``
#: Anchors carry their exact leading whitespace and are compared with ``==``.
ARMS = [
    dict(
        id="M1_D1_filter_defaults_OFF",
        what="D-1 verbatim: the refcv6 box path applies NO visibility filter unless a "
             "caller names one. This IS the pre-2026-09-23 behaviour of "
             "refc_v3_train.py:4372 -> refcv6_perception_branch.py:503 -> "
             "box3d_head.py:327.",
        file=BOX,
        anchor="                   visible_filter: bool = True,",
        replace="                   visible_filter: bool = False,",
        # ⛔ ONLY the no-keyword test can see this. The explicit-keyword tests stay GREEN
        # under it BY CONSTRUCTION, and that is what this arm measured on its first run.
        red=["test_D1_THE_DEFAULT_IS_FILTERED_because_the_trainer_never_names_it"],
        green=["test_D1_three_boxes_whose_visibility_is_known_before_any_code_runs",
               "test_D1_the_boundary_is_60_DEGREES_and_60_METRES_exactly",
               "test_D4_the_RAW_vector_is_UNMOVED_by_this_session"],
    ),
    dict(
        id="M2_D1_filter_declared_but_not_plumbed",
        what="The `occ_from_geometry` class: the flag parses, is stamped, and reaches "
             "nothing. Worse than M1 because the run record would claim a filtered arm.",
        file=BOX,
        anchor="    if visible_filter:",
        replace="    if False:   # MUTATION: declared, never plumbed",
        red=["test_D1_three_boxes_whose_visibility_is_known_before_any_code_runs",
             "test_D1_THE_DEFAULT_IS_FILTERED_because_the_trainer_never_names_it",
             "test_D1_a_supplied_match_with_the_filter_on_is_REFUSED",
             "test_D2_the_budget_now_runs_over_the_IN_FIELD_set"],
        green=["test_D3_the_lift_mask_narrows_the_supervised_cells_to_a_counted_number"],
    ),
    dict(
        id="M3_D2_budget_ranks_the_RAW_set",
        what="D-2 verbatim: the filter runs, but `match_slots` still ranks the UNFILTERED "
             "targets, so the nearest-N is chosen over a set that is half behind the ego. "
             "This is the arm that separates 'filter present' from 'filter BEFORE the "
             "budget' -- the ordering is the finding, not the filter alone.",
        file=BOX,
        anchor="    m = match or match_slots(pred, tgt)",
        replace="    m = match or match_slots(pred, tgt_raw)",
        red=["test_D2_the_budget_now_runs_over_the_IN_FIELD_set"],
        green=["test_D1_the_boundary_is_60_DEGREES_and_60_METRES_exactly",
               "test_D3_the_lift_mask_narrows_the_supervised_cells_to_a_counted_number"],
    ),
    dict(
        id="M4_D3_map_seen_is_the_clip_lifetime_mask",
        what="D-3 verbatim: `map_soft_ce` supervises every `seen` cell, including the "
             "11.048 % that BEVLift has already zeroed and replaced with its `unobserved` "
             "constant.",
        file=BRANCH,
        anchor="        m_seen = seen & lift_valid.to(seen.device)",
        replace="        m_seen = seen   # MUTATION: the clip-lifetime mask",
        red=["test_D3_the_lift_mask_narrows_the_supervised_cells_to_a_counted_number"],
        green=["test_D3_the_mask_is_the_SAME_predicate_the_lift_substitutes_on",
               "test_D1_three_boxes_whose_visibility_is_known_before_any_code_runs"],
    ),
    dict(
        id="M5_D3_branch_stops_emitting_the_mask",
        what="The fix's INPUT disappears: the branch no longer emits `map_valid`, so the "
             "trainer has nothing to pass and the loss silently falls back to `seen`.",
        file=BRANCH,
        anchor="            out[\"map_valid\"] = map_valid_from_lift(valid)",
        replace="            pass   # MUTATION: the mask is never emitted",
        red=["test_the_branch_emits_map_valid_and_it_is_the_lift_predicate"],
        green=["test_D3_the_lift_mask_narrows_the_supervised_cells_to_a_counted_number"],
    ),
    dict(
        id="M6_D4_expectation_comes_from_the_key_again",
        what="D-4 verbatim: the arm's ACTUAL corpus stops being an input, so the "
             "expectation and the artifact are selected by one key and the guard cannot "
             "go red on the operator error it names (p5: guard_blocks_the_operator_error "
             "= false).",
        file=SLOTS,
        anchor="    base = _pathlib.Path(str(join_path)).name",
        replace="    base = \"__never_matches__\"   # MUTATION: no derivation from the arm",
        red=["test_D4_the_corpus_line_is_DERIVED_FROM_THE_JOIN_not_from_the_weight_key",
             "test_D4_the_OPERATOR_ERROR_that_previously_loaded_is_now_REFUSED"],
        green=["test_D4_the_RAW_vector_is_UNMOVED_by_this_session",
               "test_D4_the_VISIBLE_vector_is_the_measured_post_filter_census"],
    ),
    dict(
        id="M7_D4_filtered_arm_served_the_RAW_vector",
        what="THE NEW DEFECT THE REVIEW WARNS FIXING D-1 WOULD CREATE: a filtered loss "
             "running the raw-join frequencies. 1.746x on other_vehicle, 0.663x on "
             "stroller, 2.63x end to end -- the anchors.pt units error in a frequency "
             "costume, and it is SILENT.",
        file=SLOTS,
        anchor="        _wkey, _dkey, _ckey = _CLS_WEIGHT_KEYS_VISIBLE",
        replace="        _wkey, _dkey, _ckey = _CLS_WEIGHT_KEYS_RAW   # MUTATION",
        red=["test_D4_the_VISIBLE_vector_is_the_measured_post_filter_census",
             "test_D4_a_FILTERED_arm_cannot_load_an_artifact_with_no_visible_vector"],
        green=["test_D4_the_RAW_vector_is_UNMOVED_by_this_session"],
    ),
    dict(
        id="M8_map_weight_denominator_stops_following_the_weights",
        what="The `slot_set_loss` denominator rule, broken on the map head: the term's "
             "SCALE would then move with the weighting as well as its per-class emphasis "
             "-- two variables in a one-variable arm. ⭐ Note the identity control at "
             "w = ones CANNOT see this (both denominators equal n there), which is "
             "exactly why the hand-computed reference exists.",
        file=ENC,
        anchor="        denom = (p * cwv * mf).sum().clamp_min(1e-12)",
        replace="        denom = float(max(n, 1))   # MUTATION",
        red=["test_map_class_weight_matches_a_HAND_COMPUTED_weighted_cross_entropy"],
        green=["test_map_class_weight_at_ONES_is_BIT_IDENTICAL_to_passing_nothing"],
    ),
]


def apply_arm(text: str, anchor: str, replace: str) -> tuple[str | None, int]:
    """Whole-line equality. Returns ``(new_text, n_matches)``; ``None`` unless n == 1."""
    lines = text.split("\n")
    hits = [i for i, ln in enumerate(lines) if ln.rstrip("\r") == anchor]
    if len(hits) != 1:
        return None, len(hits)
    keep_cr = lines[hits[0]].endswith("\r")
    lines[hits[0]] = replace + ("\r" if keep_cr else "")
    return "\n".join(lines), 1


#: pytest's own exit codes -- the ONLY honest classifier here. ⛔ The first version of
#: this function grepped the output tail for "ERROR" and misread a genuine
#: `Failed: DID NOT RAISE` as `missing`, turning a CAUGHT arm into an ESCAPED one. A
#: substring over a captured stream is not a status; the status is the status.
_PYTEST_EXIT = {0: "passed", 1: "failed", 2: "interrupted", 3: "internal-error",
                4: "usage-error", 5: "missing"}


def run_tests(names: list[str]) -> dict:
    """-> {name: 'passed'|'failed'|'missing'|...} by running each nodeid separately."""
    out = {}
    for n in names:
        p = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "--no-header", "--tb=no",
             "-p", "no:cacheprovider", f"{TESTS}::{n}"],
            cwd=str(REPO), capture_output=True, text=True,
            env={**_env(), "PYTHONPATH": str(REPO / "stack")})
        out[n] = _PYTEST_EXIT.get(p.returncode, f"exit-{p.returncode}")
    return out


def _env() -> dict:
    import os
    return dict(os.environ)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RAW / "mutation_perception_fixes.json"))
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    t0 = time.time()

    # BASELINE FIRST. ⛔ A red baseline makes every "RED under mutation" meaningless.
    base = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider",
         TESTS],
        cwd=str(REPO), capture_output=True, text=True,
        env={**_env(), "PYTHONPATH": str(REPO / "stack")})
    baseline_green = base.returncode == 0
    results = []

    for arm in ARMS:
        if a.only and arm["id"] != a.only:
            continue
        src = REPO / arm["file"]
        original = src.read_text(encoding="utf-8", newline="")
        new, n_hits = apply_arm(original, arm["anchor"], arm["replace"])
        if new is None:
            results.append({**{k: arm[k] for k in ("id", "what", "file", "anchor")},
                            "verdict": "INVALID",
                            "reason": f"anchor matched {n_hits} lines, expected exactly 1 "
                                      f"-- the source moved; this arm proves nothing"})
            continue
        backup = pathlib.Path(tempfile.mkdtemp()) / src.name
        shutil.copy2(src, backup)
        try:
            src.write_text(new, encoding="utf-8", newline="")
            red = run_tests(arm["red"])
            green = run_tests(arm["green"])
        finally:
            shutil.copy2(backup, src)
            shutil.rmtree(backup.parent, ignore_errors=True)
        ok_red = all(v == "failed" for v in red.values())
        ok_green = all(v == "passed" for v in green.values())
        results.append({
            **{k: arm[k] for k in ("id", "what", "file", "anchor", "replace")},
            "must_go_RED": red, "must_stay_GREEN": green,
            "verdict": ("CAUGHT" if ok_red and ok_green else
                        ("ESCAPED" if not ok_red else "OVERBROAD")),
        })

    # ⛔ THE FILE MUST BE BIT-IDENTICAL AFTERWARDS. A harness that leaves a mutation on
    # disk would poison every later run, and the later run would look like a real failure.
    restored = {f: __import__("hashlib").sha256(
        (REPO / f).read_bytes()).hexdigest()[:16]
        for f in sorted({arm["file"] for arm in ARMS})}
    after = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider",
         TESTS],
        cwd=str(REPO), capture_output=True, text=True,
        env={**_env(), "PYTHONPATH": str(REPO / "stack")})

    res = {
        "_evidence_class": "MEASURED (ours; artifact = this JSON + the code beside it)",
        "_what": "deliberate-regression arms for the 2026-09-22 refcv6 perception review",
        "tests": TESTS,
        "baseline_green_before": baseline_green,
        "baseline_green_after_restore": after.returncode == 0,
        "file_sha256_12_after_restore": restored,
        "n_arms": len(results),
        "n_caught": sum(1 for r in results if r["verdict"] == "CAUGHT"),
        "n_escaped": sum(1 for r in results if r["verdict"] == "ESCAPED"),
        "n_overbroad": sum(1 for r in results if r["verdict"] == "OVERBROAD"),
        "n_invalid": sum(1 for r in results if r["verdict"] == "INVALID"),
        "arms": results,
        "wall_s": round(time.time() - t0, 1),
    }
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False),
                                   encoding="utf-8")
    for r in results:
        print(f"{r['verdict']:9s} {r['id']}")
    print(json.dumps({k: res[k] for k in (
        "baseline_green_before", "baseline_green_after_restore", "n_arms", "n_caught",
        "n_escaped", "n_overbroad", "n_invalid", "wall_s")}, indent=1))
    print("wrote", a.out)
    bad = res["n_escaped"] + res["n_overbroad"] + res["n_invalid"]
    if not baseline_green or after.returncode != 0:
        bad += 1
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
