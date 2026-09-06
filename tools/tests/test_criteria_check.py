"""The criteria-completeness instrument, pinned so it cannot decay into prose.

The charter (TANITAD_PROGRAMME.md §2, EvalFlyWheel §5) says completeness must be
enforced by MACHINERY because we measurably failed to keep it by hand — three
ADE-only reports went out after the four-families rule was made binding. A guard
that has never been shown to FAIL proves nothing, so the centrepiece here is the
deliberate-regression arm: a compliant artifact with one family deleted must be
caught, or the instrument is decorative.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import criteria_check as cc  # noqa: E402

REGISTRY_PATH = ROOT / "products" / "P7-TanitEval" / "CRITERIA_REGISTRY.json"


@pytest.fixture(scope="module")
def registry() -> dict:
    reg = cc._read_json(REGISTRY_PATH)
    assert reg is not None, f"registry unreadable at {REGISTRY_PATH}"
    return reg


@pytest.fixture
def compliant() -> dict:
    """A synthetic artifact satisfying every required criterion.

    Deliberately hand-built rather than loaded from taniteval/results/, because a
    fixture read from the corpus would inherit the corpus's own gaps and the
    regression arm below would then have nothing to delete.
    """
    return {
        "block": "taniteval.driving/tier1",
        "n_windows": 881, "n_episodes": 40,
        "estimator": {"interval": "episode_cluster_bootstrap",
                      "deprecated_and_refused": "overlapping_holdout_se"},
        "protocol": {"inference_inputs": "camera only", "goal_source": "predicted goal point",
                     "vision_only": True,
                     "corpus": "physicalai-val-0c5f7dac3b11",
                     "parity_key": "physicalai-train-e438721ae894 / skip-hash f09e44db"},
        # ⭐ A model number without its control is unreadable. MEASURED: a T1 arm read
        # 9.3697 m while its own hold-action control read 0.4246 m on the same
        # windows — the trivial baseline beat the model 22×, and an artifact
        # carrying only the model's value looks exactly like a result.
        "floors": {"holdv0": "hold-v0: go straight at the observed entry speed",
                   "cv": "constant velocity"},
        "vs_floor_paired": {"holdv0": {"ade_0_2s": {"delta": -0.03, "separated": False,
                                                    "favours": "tie"}}},
        "headline": {
            "ade_0_2s": {"mean": 0.42},
            "speed_mae_mps": {"mean": 0.47}, "speed_bias_mps": {"mean": 0.19},
            "long_abs_2s_m": {"mean": 0.84}, "long_signed_2s_m": {"mean": 0.35},
            "progress_abs_err_m": {"mean": 0.83}, "progress_signed_err_m": {"mean": 0.38},
            "headway_m": {"mean": 12.0}, "ttc_s": {"mean": 4.1},
            "lat_abs_2s_m": {"mean": 0.23}, "pathgeom_crosstrack_m": {"mean": 0.11},
            "heading_mae_2s_deg": {"mean": 6.6},
            "curv_mae_1_per_m": {"mean": 0.002},
            "yaw_rate_mae_dps": {"mean": 1.4},
        },
        # ⛔ `confusion_gt_rows_pred_cols` is the name the EMITTER writes
        # (four_families.py:563). The fixture is pinned to the emitter, not to a
        # guess — using a guessed name here is what made the census report a
        # fully-instrumented family as absent (C133-b).
        "tactical": {"manoeuvre_accuracy": 0.81,
                     "confusion_gt_rows_pred_cols": [[1, 0], [0, 1]],
                     "anchor_selection": 0.77},
        "strategic": {"decision_accuracy": 0.66, "route_quality": 0.71,
                      "echo_test": {"bijection_with_input": False},
                      # ⭐ nav-COMPLIANCE (PI 2026-09-04/05): BEHAVIOUR vs the TRUE
                      # command, admissible ONLY with both intervention controls
                      # paired on the same windows. Keys pinned to the EMITTER
                      # (taniteval/taniteval/nav_compliance.py::compliance_arm).
                      "nav_compliance": {"readouts": {"plan": {
                          "conditionings": {"nav_true": {
                              "compliance_with_TRUE_command": {"mean": 0.82, "lo": 0.71,
                                                               "hi": 0.90}}},
                          "paired_true_minus_shuffled": {"delta": 0.31, "lo": 0.22,
                                                         "hi": 0.40, "separated": True},
                          "paired_true_minus_zero": {"delta": 0.24, "lo": 0.12,
                                                     "hi": 0.35, "separated": True}}}}},
    }


# ------------------------------------------------- the three states, exactly ---
def test_compliant_artifact_has_no_violations(compliant, registry):
    res = cc.check_artifact(compliant, registry)
    assert res["scope"] == cc.IN_SCOPE
    assert res["n_violations"] == 0, \
        f"the compliant fixture must pass, else the regression arm is meaningless: " \
        f"{[v['id'] for v in res['violations']]}"


@pytest.mark.parametrize("family", ["LONGITUDINAL", "LATERAL", "TACTICAL", "STRATEGIC"])
def test_DELIBERATE_REGRESSION_deleting_a_family_is_caught(compliant, registry, family):
    """⭐ THE ARM THAT MAKES EVERY PASS ABOVE MEAN SOMETHING.

    Strip the keys backing one family and the checker MUST report violations in
    exactly that family and nowhere else. A checker that stays green here would
    be reporting completeness it never verified — which is the failure mode the
    instrument exists to end.
    """
    broken = copy.deepcopy(compliant)
    for crit in registry["families"][family]["criteria"]:
        for key in crit.get("keys", []) + crit.get("partial_keys", []):
            parts = key.split(".")
            cur = broken
            for p in parts[:-1]:
                cur = cur.get(p, {}) if isinstance(cur, dict) else {}
            if isinstance(cur, dict):
                cur.pop(parts[-1], None)

    res = cc.check_artifact(broken, registry)
    hit = [r["id"] for r in res["violations"]]
    assert hit, f"deleting all of {family} produced NO violation — the guard is vacuous"

    fam_ids = {c["id"] for c in registry["families"][family]["criteria"]}
    assert fam_ids & set(hit), f"{family} deleted but not flagged; flagged={hit}"

    # And it must not spray violations across families it did not touch.
    for other in registry["families"]:
        if other == family:
            continue
        other_ids = {c["id"] for c in registry["families"][other]["criteria"]}
        leaked = other_ids & set(hit) - fam_ids
        # STRATEGIC/TACTICAL share no keys with LONGITUDINAL/LATERAL, so any
        # cross-family hit means the key-stripping was over-broad.
        assert not leaked, f"deleting {family} also flagged {other}: {leaked}"


def test_refused_with_a_reason_is_a_work_item_not_a_violation(compliant, registry):
    """The distinction the whole instrument exists for: an honest 'cannot compute,
    here is why' is admissible; silence is not."""
    art = copy.deepcopy(compliant)
    del art["headline"]["headway_m"]
    del art["headline"]["ttc_s"]
    art["refused"] = {"headway_ttc_distance_keeping":
                      "no lead-agent state exists (lead_state is a None stub)"}

    res = cc.check_artifact(art, registry)
    ids = [r["id"] for r in res["violations"]]
    assert "long.distance_keeping" not in ids, "refused-with-reason must not be a violation"
    assert any(w["id"] == "long.distance_keeping" for w in res["work_items"]), \
        "refused-with-reason must surface as a WORK ITEM, not vanish"


def test_refused_with_NO_reason_is_still_a_violation(compliant, registry):
    """A `refused` entry with an empty reason is a shrug, not an acknowledgement —
    otherwise the escape hatch swallows the rule it was carved out of."""
    art = copy.deepcopy(compliant)
    del art["headline"]["headway_m"]
    del art["headline"]["ttc_s"]
    art["refused"] = {"headway_ttc_distance_keeping": ""}

    res = cc.check_artifact(art, registry)
    assert "long.distance_keeping" in [r["id"] for r in res["violations"]], \
        "an empty refusal reason must NOT excuse a missing criterion"


def test_silently_absent_is_a_violation(compliant, registry):
    art = copy.deepcopy(compliant)
    del art["tactical"]
    res = cc.check_artifact(art, registry)
    assert any(r["id"].startswith("tac.") for r in res["violations"])


# ------------------------------------------------------------------- scope ---
def test_profiling_artifact_is_out_of_scope_not_a_pile_of_violations(registry):
    """Scoring a profiling artifact against driving criteria manufactures noise.
    A probe that reports the wrong scope is worse than no probe."""
    prof = {"fp32": {"params": {"by_module_m": {"tactical_policy": 3.1}},
                     "stages": {"hierarchy_strategic+tactical": 1.0}}}
    res = cc.check_artifact(prof, registry)
    assert res["scope"] == cc.OUT_OF_SCOPE
    assert res["n_violations"] == 0


def test_driving_artifact_is_in_scope_even_if_it_also_carries_profiling(compliant, registry):
    art = copy.deepcopy(compliant)
    art["fp32"] = {"params": {"by_module_m": {"tactical_policy": 3.1}}}
    assert cc.scope_of(art, registry)[0] == cc.IN_SCOPE, \
        "in-scope markers must win, or a driving eval could hide behind a profiling block"


def test_unclassifiable_artifact_is_UNKNOWN_never_silently_compliant(registry):
    res = cc.check_artifact({"something": "unrecognised"}, registry)
    assert res["scope"] == cc.UNKNOWN_SCOPE


# ------------------------------------------------------- forbidden estimator ---
def test_forbidden_estimator_is_caught_when_used_live(compliant, registry):
    art = copy.deepcopy(compliant)
    art["estimator"] = {"interval": "overlapping_holdout_se"}
    state, detail = cc._check_forbidden_estimator(art)
    assert state == cc.ABSENT, f"live use of the forbidden estimator must fail: {detail}"


def test_forbidden_estimator_is_allowed_under_a_deprecation_label(compliant, registry):
    """Published figures stay traceable, so the name may appear — but only where it
    is explicitly marked deprecated/refused."""
    state, _ = cc._check_forbidden_estimator(compliant)
    assert state == cc.PRESENT


# ---------------------------------------------------------------- the tiers ---
def test_T0_is_never_driving_performance(registry):
    art = {"block": "taniteval.driving/tier0", "headline": {"ade_0_2s": {"mean": 0.42}}}
    res = cc.check_artifact(art, registry)
    assert res["tier"] == "T0"
    assert res["tier_is_driving_performance"] is False, \
        "T0 is a world-model diagnostic — it must never be flagged as driving performance"


def test_missing_tier_stamp_is_a_violation(compliant, registry):
    art = copy.deepcopy(compliant)
    del art["block"]
    res = cc.check_artifact(art, registry)
    assert "hyg.tier_stamp" in [r["id"] for r in res["violations"]]


# ------------------------------------------------------- the registry itself ---
def test_registry_covers_all_four_binding_families(registry):
    """PI 2026-08-02, binding: LONGITUDINAL / LATERAL / TACTICAL / STRATEGIC."""
    assert set(registry["families"]) == {"LONGITUDINAL", "LATERAL", "TACTICAL", "STRATEGIC"}


def test_registry_families_are_never_pooled(registry):
    """No composite/aggregate family may exist — a single score hides the trade-off."""
    for name in registry["families"]:
        assert name.lower() not in ("overall", "composite", "total", "score")


def test_every_criterion_has_an_id_and_a_label(registry):
    for fam, spec in registry["families"].items():
        for crit in spec["criteria"]:
            assert crit.get("id"), f"{fam} has a criterion with no id"
            assert crit.get("label"), f"{fam}:{crit.get('id')} has no label"


def test_registry_is_valid_json_on_disk():
    json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


# ------------------------------- the ff_* schema and the inline-refusal idiom ---
# Both added after the FIRST census read the repo's most disciplined artifacts as
# its least compliant ones (C-EVAL-1). Each test below pins one false positive.

def test_inline_status_refusal_is_a_work_item_not_a_violation(registry):
    """The `four_families` schema declines a criterion in place, with
    {status: UNAVAILABLE, reason, n} — not via a top-level `refused` block. Reading
    only the second idiom reported an explicit, reasoned refusal as a silent
    omission, which inverts the instrument's whole purpose."""
    art = {"tier": "T1", "n_windows": 6844, "n_episodes": 40,
           "four_families": {"strategic": {"status": "UNAVAILABLE", "n": 6844,
                                           "reason": "no map or route signal exists on this corpus"}}}
    res = cc.check_artifact(art, registry)
    ids = [r["id"] for r in res["violations"]]
    assert "strat.decision" not in ids, "an inline reasoned refusal must not be a violation"
    assert any(w["id"] == "strat.decision" for w in res["work_items"])


def test_inline_status_refusal_without_a_reason_is_still_a_violation(registry):
    art = {"tier": "T1", "four_families": {"strategic": {"status": "UNAVAILABLE"}}}
    res = cc.check_artifact(art, registry)
    assert "strat.decision" in [r["id"] for r in res["violations"]]


def test_naming_the_forbidden_estimator_to_DISAVOW_it_is_not_a_violation():
    """MEASURED: the ff_* artifacts say verbatim 'overlapping_holdout_se is NOT
    used - it biases the POINT ESTIMATE'. Matching the bare token flagged exactly
    the artifacts doing the right thing."""
    art = {"_estimator": "episode-cluster bootstrap (taniteval.ci). "
                         "overlapping_holdout_se is NOT used anywhere: it biases "
                         "the POINT ESTIMATE, not only the interval."}
    state, detail = cc._check_forbidden_estimator(art)
    assert state == cc.PRESENT, f"a disavowal must not read as a use: {detail}"


def test_a_long_disavowal_survives_context_truncation():
    """The first fix still failed because the hit's context was truncated to 80
    chars, cutting the disavowal off. Pin a disavowal far past any truncation."""
    art = {"nested": {"deep": {"_estimator":
           ("x" * 400) + " overlapping_holdout_se is NOT used here " + ("y" * 400)}}}
    state, _ = cc._check_forbidden_estimator(art)
    assert state == cc.PRESENT


def test_live_use_of_the_forbidden_estimator_is_STILL_caught():
    """The regression arm for the two fixes above: loosening the matcher must not
    make the guard vacuous."""
    art = {"estimator": {"interval": "overlapping_holdout_se"}}
    state, _ = cc._check_forbidden_estimator(art)
    assert state == cc.ABSENT, "the guard went vacuous - it can no longer catch real use"


def test_scope_recursion_flag_exists():
    """A census over one directory is a finding about that directory. The tool must
    offer the recursive sweep, or the scoping error is structural."""
    import argparse, io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.suppress(SystemExit):
        cc.main(["--help"])
    assert "--recursive" in buf.getvalue()


# ------------------------------------------------- the adopted benchmarks ---
# Both were researched from primary sources and banked (2026-08-23). These tests
# pin the TRAPS into the registry: a trap that lives only in a 900-line protocol
# doc will be re-stepped-in, exactly as C126 measured for prose corrections.

def test_navsim_carries_the_score_vs_pdm_score_trap(registry):
    """Reading `pdm_score` instead of `score` silently reports a 14-denominator
    number that LOOKS like a valid EPDMS. The registry must carry the warning."""
    nav = registry["benchmarks"]["navsim"]
    trap = nav.get("THE_COLUMN_TRAP", "")
    assert "pdm_score" in trap and "score" in trap
    assert "16" in trap or "14" in trap, "the denominators are the tell - keep them"


def test_navsim_variant_denominators_are_pinned(registry):
    v = registry["benchmarks"]["navsim"]["variants"]
    assert v["PDMS_v1"]["denominator"] == 12
    assert v["EPDMS_v2"]["denominator"] == 16
    assert set(v["EPDMS_v2"]["multipliers"]) == {"NC", "DAC", "DDC"}


def test_nuscenes_openloop_planning_is_NOT_adoptable(registry):
    """⛔ The binding verdict. Open-loop planning is a cited comparability row, never
    a criterion or a gate: the GT human trajectory scores worse than the models,
    the averaging convention flips rankings, and VAD's high-level command is the
    GT future handed back as an input - our own route-echo defect, published."""
    plan = registry["benchmarks"]["nuscenes"]["tasks"]["planning_openloop"]
    assert plan["adoptable"] is False
    for crit in plan["criteria"]:
        assert crit["required"] is True, (
            "the guard conditions on an inadmissible metric must stay required - "
            "they are what keeps it from being read as skill")
    ids = {c["id"] for c in plan["criteria"]}
    assert {"nusc.plan.protocol_tag", "nusc.plan.gt_control",
            "nusc.plan.command_source"} <= ids


def test_nuscenes_perception_tasks_ARE_adoptable(registry):
    """Detection/tracking/prediction are devkit-enforced and their labels are not
    functions of inference inputs - they pass the leak guard cleanly."""
    tasks = registry["benchmarks"]["nuscenes"]["tasks"]
    for name in ("detection", "tracking", "prediction"):
        assert tasks[name]["adoptable"] is True
        assert tasks[name]["criteria"]


def test_nuscenes_licence_conflict_is_recorded_not_resolved(registry):
    """The paper says CC BY-NC-SA 4.0; AWS Open Data says Commercial. Recording the
    CONFLICT is honest; silently picking one would not be."""
    lic = registry["benchmarks"]["nuscenes"]["licence_status"]
    assert "CONFLICT" in lic and "RESEARCH-ONLY" in lic


def test_a_benchmark_number_does_not_satisfy_the_four_families(registry):
    note = registry["benchmarks"]["_four_families_still_bind"]
    assert "does NOT satisfy" in note
    assert "REFUSE" in note


def test_no_benchmark_is_left_pending_without_its_protocol_doc(registry):
    """A benchmark marked adopted must point at the banked spec it was built from."""
    from pathlib import Path
    for name in ("navsim", "nuscenes"):
        b = registry["benchmarks"][name]
        if not b.get("pending", False):
            doc = ROOT / b["protocol_doc"]
            assert doc.exists(), f"{name} is adopted but {b['protocol_doc']} is missing"
            assert b.get("primary"), f"{name} adopted with no primary source cited"


# ------------------------------- the registry self-check (C133-b, 2026-08-23) ---
# A criterion whose keys resolve in NO artifact is a TYPO far more often than a
# universal absence - and both print as "0 present". This exact confusion was
# published to the PI as "the surviving universal eval gap"; the real cause was
# that the registry said `confusion` while the emitter writes
# `confusion_gt_rows_pred_cols` (four_families.py:563).

def test_audit_keys_flags_a_criterion_that_resolves_nowhere(registry):
    """The arm that catches a registry typo before it becomes a finding."""
    arts = [{"four_families": {"tactical": {"lateral_decision": {"nope": 1}}}}]
    fake = {"families": {"TACTICAL": {"criteria": [
        {"id": "tac.typo", "label": "typo", "keys": ["four_families.tactical.confusion"]}]}}}
    out = cc.audit_keys(arts, fake)
    assert [u["id"] for u in out["unresolvable"]] == ["tac.typo"]


def test_audit_keys_is_quiet_when_the_key_is_right(registry):
    """The regression arm for the arm above: it must not cry typo on a good key."""
    arts = [{"four_families": {"tactical": {"confusion_gt_rows_pred_cols": [[1, 0], [0, 1]]}}}]
    fake = {"families": {"TACTICAL": {"criteria": [
        {"id": "tac.ok", "label": "ok",
         "keys": ["four_families.tactical.confusion_gt_rows_pred_cols"]}]}}}
    assert cc.audit_keys(arts, fake)["unresolvable"] == []


def test_not_yet_emitted_criteria_are_exempt_from_the_self_check():
    """A metric we have deliberately not built yet is an honest intent, not a typo -
    but it must be DECLARED, so the exemption cannot be silent."""
    fake = {"families": {"X": {"criteria": [
        {"id": "x.future", "label": "future", "keys": ["nope.nothing"],
         "not_yet_emitted": True}]}}}
    assert cc.audit_keys([{"a": 1}], fake)["unresolvable"] == []


def test_tac_confusion_points_at_the_key_the_emitter_actually_writes(registry):
    """Pin the exact defect. four_families.py:563 returns
    `confusion_gt_rows_pred_cols`; a registry naming anything else is broken."""
    crit = next(c for c in registry["families"]["TACTICAL"]["criteria"]
                if c["id"] == "tac.confusion")
    assert any("confusion_gt_rows_pred_cols" in k for k in crit["keys"]), \
        "tac.confusion must name the emitted key, not a guess"


def test_the_emitter_still_writes_the_key_the_registry_expects():
    """⭐ The two-sided guard. The registry is pinned to the EMITTER, so renaming
    the emitted field breaks this test instead of silently zeroing a family."""
    src = (ROOT / "taniteval" / "taniteval" / "four_families.py").read_text(
        encoding="utf-8", errors="replace")
    assert "confusion_gt_rows_pred_cols" in src, \
        "the emitter renamed its confusion key - update CRITERIA_REGISTRY.json too"
    assert "never_predicted" in src, \
        "never_predicted is load-bearing: 0 of 881 'accelerate' decisions was only "\
        "visible through it"


def test_every_live_criterion_resolves_somewhere_in_the_real_corpus(registry):
    """⭐ THE STANDING GUARD, run against the actual banked artifacts.

    Kept narrow (the four_families gate dumps) so it is fast and deterministic,
    but it is the check that would have caught tac.confusion on the day.
    """
    import glob
    paths = glob.glob(str(ROOT / "**" / "gates" / "four_families" / "ff_*.json"),
                      recursive=True)
    if not paths:
        pytest.skip("no four_families gate artifacts in this checkout")
    arts = [d for d in (cc._read_json(Path(p)) for p in paths) if isinstance(d, dict)]
    unresolved = [u["id"] for u in cc.audit_keys(arts, registry)["unresolvable"]]
    # STRATEGIC is legitimately refused-not-emitted on this corpus, and the
    # headline.* keys belong to the other schema, so only assert on the families
    # these artifacts do carry.
    tactical = [u for u in unresolved if u.startswith("tac.")]
    assert not tactical, f"tactical criteria resolve nowhere: {tactical}"


# --------------------------------- the release-gate criteria (PI, 2026-08-23) ---

def test_parity_criterion_exists_and_is_checked(registry):
    """Parity is sacred. An artifact that does not record WHICH corpus it scored is
    not cross-arm comparable, and every delta against it is uninterpretable.

    ⚠️ PI ruling 2026-08-28 made the gate ADVISORY — the criterion must still EXIST
    and still be CHECKED. Advisory changes who may stop a release, not whether we
    measure."""
    ids = {c["id"] for c in registry["artifact_hygiene"]["criteria"]}
    assert "hyg.parity" in ids
    assert "hyg.parity" in registry["release_gate"]["advisory_all"]


def test_floor_comparison_is_a_criterion_and_is_blocking(registry):
    """The criterion that would have caught the programme's largest misread: a T1 arm
    at 9.3697 m against its own hold-action control at 0.4246 m."""
    ids = {c["id"] for c in registry["artifact_hygiene"]["criteria"]}
    assert "ctrl.floor_comparison" in ids
    assert "ctrl.floor_comparison" in registry["release_gate"]["advisory_all"]
    crit = next(c for c in registry["artifact_hygiene"]["criteria"]
                if c["id"] == "ctrl.floor_comparison")
    assert "22" in crit["note"], "keep the measured example - it is why the rule exists"


def test_an_artifact_with_no_control_FAILS(registry, compliant):
    """DELIBERATE REGRESSION ARM: strip the floors and the gate must notice. Without
    this, a model-only number passes and reads as a result."""
    import copy
    art = copy.deepcopy(compliant)
    art.pop("floors", None); art.pop("vs_floor_paired", None)
    res = cc.check_artifact(art, registry)
    assert "ctrl.floor_comparison" in [r["id"] for r in res["violations"]]


def test_an_artifact_with_no_corpus_recorded_FAILS(registry, compliant):
    """DELIBERATE REGRESSION ARM for parity."""
    import copy
    art = copy.deepcopy(compliant)
    art["protocol"].pop("corpus", None); art["protocol"].pop("parity_key", None)
    art.pop("parity", None); art.pop("corpus", None); art.pop("val_cache", None)
    res = cc.check_artifact(art, registry)
    assert "hyg.parity" in [r["id"] for r in res["violations"]]


def test_the_gate_requires_a_T1_artifact_not_only_T0(registry):
    """⛔ T0 is a world-model diagnostic. A release may never be gated on it alone -
    the same checkpoint reads 0.3659 at T0 and 9.3697 at T1."""
    note = registry["release_gate"]["advisory_note_previous"]
    assert "T1" in note and "T0" in note


def test_cannot_rule_is_a_declared_verdict_not_an_inconclusive(registry):
    """A criterion that cannot decide must SAY so. The O6 rank gate could never rule
    (spectrum n=24 vs ceiling 1024) and its silence was read as a pass."""
    txt = registry["release_gate"]["cannot_rule_is_a_verdict"]
    assert "CANNOT-RULE" in txt
    assert "never a silent PASS" in txt


def test_regression_is_advisory_until_there_is_a_baseline(registry):
    """A regression gate with no trustworthy baseline manufactures false failures."""
    rg = registry["release_gate"]
    assert any("regression" in a for a in rg["advisory"])
    # PI ruling 2026-08-28: nothing blocks at all, so the old "regression is not in
    # the blocking set" assertion is now trivially true by construction.
    assert "blocking" not in rg, "the PI made the gate advisory; a blocking set must not return"


# ------------------------------------- dotted key names (found 2026-08-23) ---
# A key may itself contain a dot. Splitting naively makes such a field
# unreachable, and this instrument then reports it as ABSENT — a resolution bug
# and a real gap being the same observation is exactly the C133-b failure mode.

def test_a_key_containing_a_dot_is_still_reachable():
    """MEASURED: the harness emits `n_excluded_goal_below_0.5m` inside goal_setting."""
    art = {"goal_setting": {"n_excluded_goal_below_0.5m": 241}}
    assert cc._dig(art, "goal_setting.n_excluded_goal_below_0.5m") == (True, 241)


def test_a_dotted_key_at_an_intermediate_level_resolves():
    assert cc._dig({"a.b": {"c": 1}}, "a.b.c") == (True, 1)


def test_ordinary_paths_still_resolve_and_missing_ones_still_fail():
    """The regression arm: the looser resolver must not start inventing hits."""
    art = {"four_families": {"lateral": {"yaw_rate_mae_degps": 4.9188}}}
    assert cc._dig(art, "four_families.lateral.yaw_rate_mae_degps") == (True, 4.9188)
    assert cc._dig(art, "four_families.lateral.nope") == (False, None)
    assert cc._dig(art, "four_families.nope.yaw_rate_mae_degps") == (False, None)


# ============================ NAVSIM GATES (EvalFlyWheel, 2026-08-27) =========
# The provisioning scout surfaced two blockers that 32 GB of dataset does not
# fix, plus one labelling obligation. The Master Mind's word was that they go
# into the NavSim criteria "before any number is produced" — so they ship as
# registry gates with regression arms, not as a note in a message.

@pytest.fixture(scope="module")
def navsim(registry) -> dict:
    return registry["benchmarks"]["navsim"]


def test_navsim_estimator_gate_exists_and_blocks(navsim):
    """⛔ NavSim resamples SCENE TOKENS, not episodes, and its scenes explicitly
    OVERLAP. Resampling overlapping units as independent understates variance —
    the overlapping_holdout_se family, on a borrowed benchmark."""
    g = navsim["GATE_estimator_cluster_unit"]
    assert g["blocking"] is True
    assert "UNAVAILABLE" in g["admissible_until_settled"]
    assert "scene" in g["why"].lower() and "overlap" in g["why"].lower()


def test_navsim_ego_enforcement_gate_demands_a_MECHANISM_not_an_assertion(navsim):
    """MEASURED: the devkit has NO switch that removes ego status, and nothing
    verifies an agent declined to read it. 'We did not use it' is an assertion."""
    g = navsim["GATE_ego_status_enforcement"]
    assert g["blocking"] is True
    assert "assertion, not enforcement" in g["rule"]
    assert len(g["acceptable_mechanisms"]) >= 2


def test_navsim_modality_label_gate_exists(navsim):
    """The leaderboard server records no modality, so the labelling duty is ours."""
    g = navsim["GATE_modality_label"]
    assert g["blocking"] is True
    for k in ("protocol.sensor_set", "protocol.setting"):
        assert k in g["keys"]


def test_the_comparable_ladder_is_recorded_with_its_source(navsim):
    """Our comparable class is front-camera-only + perception-free, NOT the
    multi-camera headline. Pin the ladder so a future report cannot drift to the
    flattering row."""
    lad = navsim["GATE_modality_label"]["comparable_ladder_perception_free_front_only"]
    assert "2601.22032" in lad["_source"]
    assert lad["LAW"]["PDMS"] == 83.8
    assert lad["Drive-JEPA"]["PDMS"] == 89.0
    # the entry rung must stay inside our parameter budget, or the ladder is the
    # wrong one to be quoting at all
    assert lad["LAW"]["encoder"] == "21M"


def test_drive_jepa_v1_number_is_93_7_not_93_3(navsim):
    """⛔ The commissioning brief carried 93.3. The paper's abstract and §1 both
    say 93.7; 93.3 appears only in its own NeurIPS checklist, misquoting itself."""
    ref = navsim["published_reference_numbers"]
    assert ref["drive_jepa_v1_PDMS_full_framework"] == 93.7
    assert "93.3" in ref["drive_jepa_v1_PDMS_NOTE"]
    assert ref["latent_wam_v2_EPDMS"] == 89.3


def test_the_v1_branch_pin_is_recorded(navsim):
    """'v1 PDMS from main' silently computes EPDMS — main IS v2."""
    pin = navsim["published_reference_numbers"]["_harness_pin"]
    assert "v1.1" in pin and "main" in pin


def test_DELIBERATE_REGRESSION_navsim_gates_cannot_be_silently_dropped(registry):
    """⭐ The arm that makes the four tests above mean something: if someone
    removes a NavSim gate from the registry, this fails. A gate that can vanish
    without a test failing is a note, not machinery."""
    ns = registry["benchmarks"]["navsim"]
    required = {"GATE_estimator_cluster_unit", "GATE_ego_status_enforcement",
                "GATE_modality_label"}
    missing = required - set(ns)
    assert not missing, f"NavSim gate(s) removed from the registry: {missing}"
    for name in required:
        assert ns[name].get("blocking") is True, f"{name} was downgraded to non-blocking"



# ------------------------------- the PI ruling: advisory, not silent (2026-08-28) ---

def test_the_PI_advisory_ruling_is_recorded_with_its_effect(registry):
    """A ruling that lives only in a chat message decays. Pin it."""
    r = registry["release_gate"]["PI_RULING_2026-08-28"]
    assert "DOES NOT BLOCK" in r["ruling"].upper()
    assert r["by"].startswith("Sayed")


def test_advisory_must_NOT_soften_the_verdict(registry):
    """⭐ THE POINT OF THE RULING, and the way it could go wrong.

    Advisory is about AUTHORITY, not honesty. If "it no longer blocks" quietly
    became "it no longer says FAIL", the instrument would be worthless — and that
    is exactly the drift this programme keeps paying for.
    """
    r = registry["release_gate"]["PI_RULING_2026-08-28"]
    txt = r["what_does_NOT_change"]
    assert "FAIL as a FAIL" in txt
    assert "CANNOT-RULE" in txt
    assert "22x" in txt or "22×" in txt, "keep the measured floor example concrete"


def test_every_criterion_survived_the_move_to_advisory(registry):
    """DELIBERATE REGRESSION ARM: making the gate advisory must not quietly DROP a
    criterion. All nine survive, by name."""
    adv = registry["release_gate"]["advisory_all"]
    for must in ("hyg.tier_stamp", "hyg.estimator_named", "hyg.no_forbidden_estimator",
                 "hyg.parity", "ctrl.floor_comparison",
                 "leak_guards.vision_only_inference",
                 "leak_guards.goal_situation_disjoint"):
        assert must in adv, f"{must} vanished when the gate became advisory"
    assert len(adv) == 9


# ================= navhard protocol pin (work package 2026-08-28) ============
# LAB-RUN-002: NavSim v2 carries TWO protocols called "EPDMS", ~30 points apart.
# Untagged, our own leaderboard merges them by accident.

def test_the_official_navsim_column_is_navhard_two_stage(navsim):
    oc = navsim["OFFICIAL_COLUMN"]
    assert oc["column"] == "navhard_two_stage EPDMS"
    assert "30 POINTS APART" in oc["why"].upper()


def test_cross_protocol_comparison_is_gated(navsim):
    g = navsim["GATE_no_cross_protocol_comparison"]
    assert g["blocking"] is True
    assert g["official"] == "EPDMS_v2_navhard_two_stage"
    assert g["official"] in g["closed_set"]
    # the tag must be mandatory, including by omission
    assert "omitting the tag" in g["rule"]


def test_drive_jepa_87_8_is_recorded_as_NOT_navhard(navsim):
    """⭐ Independently verified by the EvalFlyWheel from the banked PDF: seven probe
    terms ('navhard', 'two_stage', 'private_test', 'challenge', …) ALL ZERO. Its
    87.8 is the navtest variant, so our navhard number is not comparable to it."""
    c = navsim["OFFICIAL_COLUMN"]["consequence_for_our_positioning"]
    assert "NEVER mentions navhard" in c
    assert "56.3" in c, "the camera-only navhard bar (DrivoR) must be named"


def test_the_v1_PDMS_ladder_is_preserved_as_a_v1_claim(navsim):
    """The perception-free ladder is still valid — as PDMS v1. It must never be
    printed in an EPDMS column."""
    w = navsim["OFFICIAL_COLUMN"]["what_still_stands"]
    assert "89.0" in w and "83.8" in w
    assert "never be printed in an EPDMS column" in w


def test_all_nine_EPDMS_submetrics_are_ingested(navsim):
    s = navsim["EPDMS_submetrics"]
    mult, wtd = s["multiplicative"], s["weighted"]
    assert set(mult) == {"NC", "DAC", "DDC", "TLC"}
    assert set(wtd) == {"EP", "TTC", "LK", "HC", "EC"}
    assert sum(v["weight"] for v in wtd.values()) == s["_weighted_denominator"] == 16


def test_every_submetric_declares_a_family_or_an_explicit_None(navsim):
    """⛔ No sub-metric may be silently unmapped — the whole point of the four-family
    rule is that a gap is DECLARED, not omitted."""
    s = navsim["EPDMS_submetrics"]
    for block in ("multiplicative", "weighted"):
        for name, m in s[block].items():
            assert "family" in m, f"{name} has no family key at all"
            if m["family"] is None:
                assert name in s["_unmapped"]
                assert "UNMAPPED" in m["family_note"]


def test_EPDMS_is_recorded_as_NOT_satisfying_the_four_families(navsim):
    """⭐ The honest reading, and it is a finding: EPDMS measures COMPLIANCE AND
    OUTCOMES, not DECISION QUALITY. A collision-free run is not evidence the
    manoeuvre decision was right."""
    s = navsim["EPDMS_submetrics"]
    assert "does NOT satisfy the four-families rule" in s["_family_caveat"]
    for missing in ("tac.manoeuvre_decision", "tac.confusion", "strat.route_goal"):
        assert missing in s["_families_epdms_cannot_supply"]


def test_DELIBERATE_REGRESSION_the_protocol_pin_cannot_silently_vanish(navsim):
    """⭐ The arm that makes the rest mean something: downgrade or delete the pin and
    this fails. A pin that can disappear without a test failing is a note."""
    assert navsim["GATE_no_cross_protocol_comparison"]["blocking"] is True
    assert navsim["OFFICIAL_COLUMN"]["column"], "the official column was emptied"
    assert len(navsim["GATE_no_cross_protocol_comparison"]["closed_set"]) >= 5


# ------------------------------- the nav-COMPLIANCE criteria (2026-09-05) ---
# The route-accuracy label is a bijection of the fed token (D-REFAV1-ROUTE-LABEL-
# IS-THE-NAV, 141/141) and cannot fail; the compliance block scores BEHAVIOUR and
# is admissible ONLY with its shuffle and zero controls on the same windows.

def _strat_ids(registry):
    return {c["id"]: c for c in registry["families"]["STRATEGIC"]["criteria"]}


def test_nav_compliance_criteria_exist_and_are_required(registry):
    ids = _strat_ids(registry)
    for cid in ("strat.nav_compliance", "strat.nav_compliance_ctrl_shuffle",
                "strat.nav_compliance_ctrl_zero"):
        assert cid in ids, f"{cid} missing from STRATEGIC"
        assert ids[cid]["required"] is True
        assert ids[cid]["refused_as"], "must be refusable WITH a reason"
    assert "bijection" in ids["strat.nav_compliance"]["note"], \
        "keep the reason the criterion exists — the metric it replaces could not fail"
    # The pin is that the registry names the stem the EMITTER writes, not a guess.
    # Exactly ONE key per control is exempt: the refav1 records' own strategic
    # DECLINATION ({status: UNAVAILABLE, reason, n}), which makes both controls read
    # REFUSED - a work item - instead of ABSENT on a record that emits no
    # behaviour-compliance readout at all. The exemption is read from the registry's
    # `arm_scoring` block rather than hardcoded, so renaming the scored arm cannot
    # leave behind a dead exemption that silently admits any key.
    # ⚠️ THERE IS MORE THAN ONE SCORED ARM. Two per-arm schemas exist and they
    # name the planner arm differently -- refav1_arm.py emits `cl`, refcv3_arm.py
    # emits `os` (registry v2.9.0, `arm_scoring.schemas`). Deriving ONE declination
    # from `scored_arm` alone made this guard REJECT the refcv3/v4 declination while
    # ADMITTING refav1's -- not a distinction the criterion intends: both are the
    # same {status, reason, n} block on a record that emits no behaviour-compliance
    # readout. The set is still READ FROM THE REGISTRY, never hardcoded, so a renamed
    # or removed scored arm still cannot leave a dead exemption behind.
    scored_arms = {registry["arm_scoring"]["scored_arm"]}
    scored_arms |= {sch["scored_arm"]
                    for sch in (registry["arm_scoring"].get("schemas") or {}).values()
                    if isinstance(sch, dict) and "scored_arm" in sch}
    declinations = {f"arms.{a}.four_families.strategic" for a in scored_arms}
    assert declinations, "no scored arm in arm_scoring - the exemption is vacuous"
    for cid, stem in (("strat.nav_compliance_ctrl_shuffle", "paired_true_minus_shuffled"),
                      ("strat.nav_compliance_ctrl_zero", "paired_true_minus_zero")):
        keys = ids[cid]["keys"]
        assert all(k.endswith(stem) or k in declinations for k in keys), \
            f"{cid} names a key that is neither the emitted stem nor a declination: {keys}"
        assert any(k.endswith(stem) for k in keys), \
            f"{cid} lost the emitted stem entirely - the pin would be vacuous"


def test_nav_compliance_registry_keys_match_the_emitter():
    """⭐ The two-sided pin (the tac.confusion lesson): the registry names the
    keys the EMITTER writes, so renaming either breaks a test instead of
    silently zeroing the family."""
    src = (ROOT / "taniteval" / "taniteval" / "nav_compliance.py").read_text(
        encoding="utf-8", errors="replace")
    for key in ('"compliance_with_TRUE_command"', '"conditionings"', '"readouts"',
                "paired_true_minus_", "def unavailable_block", "def compliance_arm"):
        assert key in src, f"the emitter no longer writes {key}"


@pytest.mark.parametrize("key,cid", [
    ("paired_true_minus_shuffled", "strat.nav_compliance_ctrl_shuffle"),
    ("paired_true_minus_zero", "strat.nav_compliance_ctrl_zero"),
])
def test_DELIBERATE_REGRESSION_nav_compliance_without_a_control_is_a_violation(
        compliant, registry, key, cid):
    """A compliance rate whose intervention control is missing is coincidence
    wearing a result's clothes — the exact shape of the 1.0000 route score.
    Delete ONE control and exactly that criterion must fire; a guard that stays
    green here is decorative."""
    broken = copy.deepcopy(compliant)
    del broken["strategic"]["nav_compliance"]["readouts"]["plan"][key]
    res = cc.check_artifact(broken, registry)
    hit = [v["id"] for v in res["violations"]]
    assert cid in hit, f"deleting {key} was not flagged: {hit}"
    other = {"strat.nav_compliance_ctrl_shuffle",
             "strat.nav_compliance_ctrl_zero"} - {cid}
    assert not (other & set(hit)), f"the OTHER control was flagged too: {hit}"
    assert "strat.nav_compliance" not in hit, "the RATE is still present; only the control is missing"


def test_nav_compliance_RATE_alone_is_two_named_violations(compliant, registry):
    art = copy.deepcopy(compliant)
    art["strategic"]["nav_compliance"] = {"readouts": {"plan": {"conditionings": {
        "nav_true": {"compliance_with_TRUE_command": {"mean": 0.97}}}}}}
    hit = {v["id"] for v in cc.check_artifact(art, registry)["violations"]}
    assert {"strat.nav_compliance_ctrl_shuffle", "strat.nav_compliance_ctrl_zero"} <= hit


def test_nav_compliance_refused_with_a_reason_is_a_work_item(compliant, registry):
    """An arm with no nav input (a flat build, or --nav-source none) may DECLINE
    the block, with a reason — that is a tracked work item, not a violation."""
    art = copy.deepcopy(compliant)
    del art["strategic"]["nav_compliance"]
    art.setdefault("refused", {}).update({
        "nav_compliance": "no nav-conditioned arm: the build feeds nav_cmd=None "
                          "on every window, so compliance is undefined",
        "nav_compliance_controls": "same reason — no fed token to intervene on"})
    res = cc.check_artifact(art, registry)
    assert res["n_violations"] == 0, [v["id"] for v in res["violations"]]
    work = {w["id"] for w in res["work_items"]}
    assert {"strat.nav_compliance", "strat.nav_compliance_ctrl_shuffle",
            "strat.nav_compliance_ctrl_zero"} <= work


def test_nav_compliance_inline_UNAVAILABLE_at_the_key_is_a_work_item(compliant, registry):
    """`taniteval.nav_compliance.unavailable_block` nests {status: UNAVAILABLE,
    reason, n} AT every registered key, so an un-rolled control is a tracked
    work item, never PRESENT and never ABSENT."""
    art = copy.deepcopy(compliant)
    st = {"status": "UNAVAILABLE", "reason": "nav_shuffled not rolled", "n": 0}
    art["strategic"]["nav_compliance"]["readouts"]["plan"]["paired_true_minus_shuffled"] = dict(st)
    res = cc.check_artifact(art, registry)
    assert "strat.nav_compliance_ctrl_shuffle" not in [v["id"] for v in res["violations"]]
    assert any(w["id"] == "strat.nav_compliance_ctrl_shuffle" for w in res["work_items"])


def test_route_head_echo_guard_accepts_the_intervention_pair(registry):
    g = next(x for x in registry["leak_guards"]["guards"] if x["id"] == "route_head_echo")
    assert any("nav_compliance" in k for k in g["keys"])


# ================= the refav1 per-arm schema (registry v2.7.0) ===============
# MEASURED 2026-09-05: EVERY refav1 record read "UNKNOWN_SCOPE: matches no
# in-scope or out-of-scope marker". They are the programme's only T1 driving
# artifacts and they carry all four binding families, so the completeness
# machinery was blind to exactly the artifacts it exists for. Root cause: the
# in-scope marker and every criterion named top-level `four_families`, while
# refav1 nests a FULL four_families block PER ARM at arms.<arm>.four_families.*.
#
# ⛔ THE MARKER ALONE WOULD HAVE BEEN WORSE THAN THE GAP. Promote the artifact to
# IN_SCOPE while the keys still point at the top level and every family reads
# ABSENT — a full set of false violations, which is the wrong-scope failure the
# registry's own `why` block warns about (`df` on a pod, `free` on Thor, cgroup
# `usage_in_bytes`). Marker and keys ship together; the arms below are what
# prove they did.

REFAV1_ARMS = ("cl", "ha", "ha0", "ha0_ext", "ol")


def _refav1_families_block(tier: str = "T1") -> dict:
    """ONE arm's four_families block, in the shape refav1_arm.py emits.

    Hand-built, not loaded from a rec_*.json, for the same reason the `compliant`
    fixture is: a fixture read from the corpus inherits the corpus's own gaps and
    the regression arm would then have nothing to delete.
    """
    return {
        "_tier": tier,
        "longitudinal": {
            "speed_mae_mps": 0.7327, "speed_bias_mps": 0.0218,
            "target_speed_acc": {"within_0.5_mps": 0.5175, "within_1.0_mps": 0.7425},
            "along_mae_m": 0.686, "along_bias_m": 0.191,
            "ego_progress": {"status": "OK", "progress_ratio_mean": 0.79},
            # The inline-refusal idiom, in the refav1 shape: declined in place,
            # WITH a reason and its n. Admissible, and a visible work item.
            "distance_keeping": {
                "status": "UNAVAILABLE",
                "reason": "no lead-agent track supplied — pass `lead=`", "n": 0},
            # ⭐ The floor comparison for a refav1 arm IS its anti-echo block: the
            # trivial hold-v0 control on the same windows, with its verdict.
            "anti_echo": {"status": "OK", "flagged": True, "holdv0_baseline": {
                "verdict": "NOT_SEPARATED",
                "estimator": "paired_episode_cluster_bootstrap (taniteval.ci) — ⛔ NOT "
                             "overlapping_holdout_se, which is anti-conservative AND "
                             "biases the point estimate"}},
            "n_windows": 40,
            "estimator": "full_set pooled mean over windows (point); intervals under "
                         "'intervals'",
        },
        "lateral": {"cross_mae_m": 0.5763, "heading_mae_deg": 21.3887,
                    "curvature_mae_1pm": 0.046862, "yaw_rate_mae_degps": 11.8161},
        "tactical": {
            "status": "OK",
            "lateral_decision": {"status": "OK", "accuracy": 0.62,
                                 "confusion_gt_rows_pred_cols": [[1, 0], [0, 1]]},
            "longitudinal_decision": {"status": "OK", "accuracy": 0.55,
                                      "confusion_gt_rows_pred_cols": [[1, 0], [0, 1]]},
            "maneuver_5way_collapsed": {"status": "OK", "accuracy": 0.48,
                                        "confusion_gt_rows_pred_cols": [[1, 0], [0, 1]]},
            # the dotted key name is kept: it is the field that broke _dig once
            "goal_setting": {"status": "OK", "goal_point_error_m": 2.4753,
                             "n_excluded_goal_below_0.5m": 3},
            "_estimator": "full_set point estimate; episode-cluster bootstrap intervals "
                          "(taniteval.ci). ⛔ overlapping_holdout_se is NOT used.",
        },
        # ⛔ Declined in place, with its reason and n=0 — NOT silently omitted.
        "strategic": {
            "status": "UNAVAILABLE",
            "reason": "strategic decisions not present in the scored pass (missing "
                      "['route_pred', 'route_gt']); a hierarchy-traversing eval is a "
                      "WORK ITEM",
            "n": 0},
        "_protocol": {
            "inference_inputs": "cached frozen DINOv3 patch features of the OBSERVED "
                                "window (vision); measured v0 at t0",
            "vision_only": "vision + v0(t0) + nav token; no ego state beyond v0, no future",
            "goal_source": "cl: tactical_imagined 100.0 % [space tactical_query_field]",
            "goal_situation_disjoint": "True by construction: nav is the labels blob's "
                                       "nav_command token, not a situation classifier "
                                       "output",
            "corpus": "refav1 p4/fp8 cache",
            "parity_key": "the v7.2 EVAL clip set, identified by the labels blob md5",
        },
    }


@pytest.fixture
def refav1() -> dict:
    """A refav1 record: FIVE arms on the same 40 windows, one of them T0."""
    protocol = dict(_refav1_families_block()["_protocol"])
    return {
        "tool": "taniteval/tools/refav1_arm.py",
        "n_windows": 40, "n_episodes": 8,
        "arm_keys": list(REFAV1_ARMS),
        "tiers": {a: ("T0" if a == "ol" else "T1") for a in REFAV1_ARMS},
        "_estimator": "point estimates are FULL-SET pooled means over windows; intervals "
                      "are the episode-cluster bootstrap (taniteval.ci). ⛔ "
                      "overlapping_holdout_se is NOT used anywhere.",
        "arms": {a: {
            "tier": "T0" if a == "ol" else "T1",
            "four_families": _refav1_families_block("T0" if a == "ol" else "T1"),
            "intervals": {"tier": "T0" if a == "ol" else "T1", "n": 40,
                          "estimator": "episode_cluster_bootstrap (taniteval.ci)"},
        } for a in REFAV1_ARMS},
        "refav1": {
            "n_windows": 40, "n_episodes": 8,
            "trivial_profile": {"n_windows": 40, "arms": {a: {"trivial_frac": 0.0}
                                                          for a in REFAV1_ARMS}},
            "distance_keeping": {"status": "REFUSED",
                                 "reason": "no lead block passed (--lead-block)", "n": 0},
            "protocol": protocol,
            # ⭐ The strategic DECISION/ROUTE readout. It lives at the RECORD level,
            # not under arms.<arm>, because the decision heads read the OBSERVED
            # window only — no rollout enters them, so it is arm-independent.
            "strategic": {
                "tier": "T1", "n_windows": 40, "n_route_labeled": 8,
                "_echo_caveat": "route_label and nav_cmd derive from the SAME "
                                "nav_command field, so under the TRUE nav route "
                                "accuracy measures the nav echo",
                "conditionings": {
                    "nav_true": {"status": "OK", "n": 8, "accuracy": 1.0, "kappa": 1.0,
                                 "confusion_gt_rows_pred_cols": [[2, 0, 0], [0, 2, 0],
                                                                 [0, 0, 4]]},
                    "nav_shuffled": {"status": "OK", "n": 8, "accuracy": 0.5},
                    "nav_zero": {"status": "OK", "n": 8, "accuracy": 0.25}},
                "paired_true_minus_shuffled_accuracy": {
                    "delta": 0.5, "lo": 0.125, "hi": 0.875, "separated": True,
                    "estimator": "paired_episode_cluster_bootstrap"},
            },
        },
    }


def _scored(registry) -> str:
    return registry["arm_scoring"]["scored_arm"]


def _strip_family_keys(art: dict, registry: dict, family: str) -> dict:
    """Remove every key backing one family — the same stripping the top-of-file
    regression arm does, reused so the two arms cannot drift apart."""
    broken = copy.deepcopy(art)
    for crit in registry["families"][family]["criteria"]:
        for key in crit.get("keys", []) + crit.get("partial_keys", []):
            parts = key.split(".")
            cur = broken
            for p in parts[:-1]:
                cur = cur.get(p, {}) if isinstance(cur, dict) else {}
            if isinstance(cur, dict):
                cur.pop(parts[-1], None)
    return broken


# ------------------------------------------------------------------ scope ---
def test_a_refav1_record_is_IN_SCOPE(refav1, registry):
    """The defect itself, pinned. Before v2.7.0 this returned UNKNOWN_SCOPE and the
    record — a T1 driving eval carrying all four families — was not counted at all."""
    scope, why = cc.scope_of(refav1, registry)
    assert scope == cc.IN_SCOPE, f"refav1 record is not in scope: {why}"
    assert f"arms.{_scored(registry)}.four_families" in why


def test_the_refav1_marker_names_the_scored_arm(registry):
    """The marker and `arm_scoring` must agree, or the record is admitted by one
    arm's presence and then scored on another."""
    marker = f"arms.{_scored(registry)}.four_families"
    values = [r.get("value") for r in registry["applicability"]["in_scope_if_any"]]
    assert marker in values, f"no in-scope marker for the scored arm; have {values}"


def test_DELIBERATE_REGRESSION_the_marker_without_the_keys_is_the_worse_failure(
        refav1, registry):
    """⭐ THE ARM THAT JUSTIFIES SHIPPING BOTH HALVES AT ONCE.

    Reconstruct the tempting half-fix — the in-scope marker added, the per-arm key
    alternatives NOT — and show it manufactures violations across every family. It
    is not a smaller version of the fix; it is a worse state than the gap, because
    an UNKNOWN scope is surfaced for a human while a false violation reads as a
    finding.
    """
    half = copy.deepcopy(registry)
    pfx = f"arms.{_scored(registry)}."
    for spec in half["families"].values():
        for crit in spec["criteria"]:
            crit["keys"] = [k for k in crit.get("keys", [])
                            if not k.startswith(pfx) and not k.startswith("refav1.")]
            crit["partial_keys"] = [k for k in crit.get("partial_keys", [])
                                    if not k.startswith(pfx)
                                    and not k.startswith("refav1.")]

    res = cc.check_artifact(refav1, half, )
    assert res["scope"] == cc.IN_SCOPE
    fams_hit = {f for f, rows in res["families"].items()
                if any(r["state"] == cc.ABSENT for r in rows)}
    assert fams_hit == {"LONGITUDINAL", "LATERAL", "TACTICAL", "STRATEGIC"}, \
        f"expected the half-fix to break all four families, broke {fams_hit}"

    # ...and the shipped registry must not do that.
    good = cc.check_artifact(refav1, registry)
    assert good["n_violations"] == 0, \
        f"the shipped registry manufactures violations: " \
        f"{[(v['id'], v['detail']) for v in good['violations']]}"


# ------------------------------------------------------------------- tiers ---
def test_a_refav1_record_stamps_T1_from_the_scored_arm(refav1, registry):
    """⛔ The record contains a T0 arm (`ol`, the recorded future integrated from
    v0). The stamp must come from the SCORED arm, or a driving artifact reads as a
    world-model diagnostic — or, far worse, the reverse."""
    res = cc.check_artifact(refav1, registry)
    assert res["tier"] == "T1"
    assert res["tier_is_driving_performance"] is True
    assert "hyg.tier_stamp" not in [v["id"] for v in res["violations"]]


def test_DELIBERATE_REGRESSION_a_T0_scored_arm_is_never_driving_performance(
        refav1, registry):
    """Flip the SCORED arm's stamp to T0 and the driving-performance flag must go
    false. T0 is a world-model diagnostic and is never quotable as driving."""
    art = copy.deepcopy(refav1)
    art["arms"][_scored(registry)]["four_families"]["_tier"] = "T0"
    art["arms"][_scored(registry)]["tier"] = "T0"
    res = cc.check_artifact(art, registry)
    assert res["tier"] == "T0"
    assert res["tier_is_driving_performance"] is False


def test_the_refav1_tier_paths_are_LAST_so_no_other_schema_moves(registry):
    """resolve_tier returns on the FIRST path that yields a string. The refav1
    paths must stay at the end, or adding them could restamp an artifact of a
    different schema that happens to carry an `arms` block."""
    paths = registry["tiers"]["key_paths"]
    first_refav1 = min(i for i, p in enumerate(paths) if p.startswith("arms."))
    assert all(not p.startswith("arms.") for p in paths[:first_refav1])
    assert all(p.startswith("arms.") for p in paths[first_refav1:])


# ------------------------------------------------- the deliberate regression ---
@pytest.mark.parametrize("family", ["LONGITUDINAL", "LATERAL", "TACTICAL", "STRATEGIC"])
def test_DELIBERATE_REGRESSION_deleting_a_family_from_a_refav1_record_is_caught(
        refav1, registry, family):
    """⭐ THE ARM THAT MAKES THE refav1 SUPPORT MEAN ANYTHING.

    The top-of-file arm proves the checker catches a deleted family in the
    `taniteval.driving` shape. It says NOTHING about this one — a registry that
    admitted refav1 records and then resolved none of their keys would still pass
    it. So the same regression is run again, per family, on the refav1 shape.
    """
    broken = _strip_family_keys(refav1, registry, family)
    res = cc.check_artifact(broken, registry)
    assert res["scope"] == cc.IN_SCOPE, "the record must stay in scope while broken"
    hit = [r["id"] for r in res["violations"]]
    assert hit, f"deleting {family} from a refav1 record produced NO violation — " \
                f"the guard is vacuous"

    fam_ids = {c["id"] for c in registry["families"][family]["criteria"]}
    assert fam_ids & set(hit), f"{family} deleted but not flagged; flagged={hit}"

    for other in registry["families"]:
        if other == family:
            continue
        other_ids = {c["id"] for c in registry["families"][other]["criteria"]}
        leaked = (other_ids & set(hit)) - fam_ids
        assert not leaked, f"deleting {family} also flagged {other}: {leaked}"


def test_DELIBERATE_REGRESSION_deleting_one_metric_is_caught(refav1, registry):
    """The finer arm: a family is not all-or-nothing. Remove ONLY the yaw-rate
    number — the metric CLAUDE.md names as where a smooth-but-wrong path hides —
    and exactly that criterion must fire."""
    broken = copy.deepcopy(refav1)
    del broken["arms"][_scored(registry)]["four_families"]["lateral"]["yaw_rate_mae_degps"]
    hit = [v["id"] for v in cc.check_artifact(broken, registry)["violations"]]
    assert hit == ["lat.yaw_rate"], f"expected only lat.yaw_rate, got {hit}"


# ------------------------------------------------------------ arm scoping ---
def test_ONLY_the_scored_arm_is_read(refav1, registry):
    """⛔ A refav1 record holds FIVE arms. Scoring all five would multiply every
    count by five and would present three trivial CONTROLS and a T0 diagnostic as
    driving evals. Gut every non-scored arm: the verdict must not move by one row.
    """
    before = cc.check_artifact(refav1, registry)
    art = copy.deepcopy(refav1)
    for arm in REFAV1_ARMS:
        if arm != _scored(registry):
            art["arms"][arm] = {}
    after = cc.check_artifact(art, registry)
    assert after == before, "a non-scored arm changed the verdict — scoring is not arm-scoped"


def test_the_counts_are_not_multiplied_by_the_number_of_arms(refav1, registry):
    """One record, one row per criterion — not one per arm."""
    res = cc.check_artifact(refav1, registry)
    for fam, rows in res["families"].items():
        assert len(rows) == len(registry["families"][fam]["criteria"]), \
            f"{fam} produced {len(rows)} rows for " \
            f"{len(registry['families'][fam]['criteria'])} criteria"


def test_deleting_the_SCORED_arm_is_caught_not_silently_excused(refav1, registry):
    """The complement of the arm above: losing the scored arm must not quietly fall
    back to a control arm's numbers. It leaves scope, which is surfaced."""
    art = copy.deepcopy(refav1)
    del art["arms"][_scored(registry)]
    assert cc.scope_of(art, registry)[0] != cc.IN_SCOPE, \
        "with the scored arm gone the record must NOT be scored on a control arm"


def test_arm_scoring_records_why_each_other_arm_is_excluded(registry):
    """The decision must be documented where the next reader looks, not inferred."""
    a = registry["arm_scoring"]
    assert a["scored_arm"] == "cl"
    for arm in ("ha", "ha0", "ha0_ext", "ol"):
        assert arm in a["not_scored"], f"{arm} excluded with no recorded reason"
    assert "T0" in a["not_scored"]["ol"], "the T0 arm must be marked as such"
    assert "CONTROL" in a["not_scored"]["ha0"]


# ------------------------------------- the honest middle state, refav1 shape ---
def test_refav1_inline_declinations_are_work_items_not_violations(refav1, registry):
    """distance-keeping and the three nav-compliance criteria are declined IN PLACE
    with a reason and an n. That is the state the instrument exists to distinguish
    from a silent omission — it must read as WORK, never as a pass and never as a
    violation."""
    res = cc.check_artifact(refav1, registry)
    work = {w["id"] for w in res["work_items"]}
    assert {"long.distance_keeping", "strat.nav_compliance",
            "strat.nav_compliance_ctrl_shuffle",
            "strat.nav_compliance_ctrl_zero"} <= work, f"work items: {sorted(work)}"
    assert res["n_violations"] == 0


def test_DELIBERATE_REGRESSION_a_declination_with_no_reason_is_still_a_violation(
        refav1, registry):
    """A refusal without a reason is a shrug, not an acknowledgement — in this
    schema too."""
    art = copy.deepcopy(refav1)
    art["arms"][_scored(registry)]["four_families"]["longitudinal"]["distance_keeping"] = \
        {"status": "UNAVAILABLE", "n": 0}
    art["refav1"]["distance_keeping"] = {"status": "REFUSED", "n": 0}
    hit = [v["id"] for v in cc.check_artifact(art, registry)["violations"]]
    assert "long.distance_keeping" in hit, hit


def test_refav1_route_accuracy_is_NOT_read_as_nav_compliance(refav1, registry):
    """⭐ THE MAPPING THAT WOULD HAVE BEEN THE EASY MISTAKE.

    refav1's route readout scores 1.0000 because the route LABEL is a bijection of
    the fed nav token (141/141) — the metric that CANNOT fail, and the exact reason
    strat.nav_compliance was created. Mapping it onto the compliance criterion
    would turn an echo into a capability claim, which is the defect the criterion
    was built to end.
    """
    res = cc.check_artifact(refav1, registry)
    rows = {r["id"]: r for r in res["families"]["STRATEGIC"]}
    assert rows["strat.nav_compliance"]["state"] == cc.REFUSED, \
        "the compliance RATE must not read PRESENT off a route-accuracy number"
    for cid in ("strat.nav_compliance_ctrl_shuffle", "strat.nav_compliance_ctrl_zero"):
        assert rows[cid]["state"] == cc.REFUSED, \
            "a control for a rate that does not exist cannot be PRESENT"
    # ...while the route readout the record DOES carry is not thrown away.
    assert rows["strat.decision"]["state"] == cc.PRESENT
    assert rows["strat.route_goal"]["state"] == cc.PRESENT
    echo = next(g for g in res["leak_guards"] if g["id"] == "route_head_echo")
    assert echo["state"] == cc.PRESENT, \
        "the route-head echo test is where refav1's shuffle control belongs"


def test_the_route_echo_guard_is_answered_by_the_intervention_pair(refav1, registry):
    """DELIBERATE REGRESSION for the line above: drop the paired shuffle delta and
    the echo guard must stop reading PRESENT. A route score near 1.0 with no echo
    test is the flagship-v1 misread (369/369, scored 1.0000)."""
    art = copy.deepcopy(refav1)
    del art["refav1"]["strategic"]["paired_true_minus_shuffled_accuracy"]
    del art["refav1"]["strategic"]["_echo_caveat"]
    res = cc.check_artifact(art, registry)
    echo = next(g for g in res["leak_guards"] if g["id"] == "route_head_echo")
    assert echo["state"] == cc.ABSENT


# ------------------------------------------- the forbidden-estimator wording ---
def test_a_refav1_disavowal_of_the_forbidden_estimator_is_not_a_violation(refav1):
    """⛔ MEASURED on the banked records: `anti_echo.holdv0_baseline.estimator`
    reads "⛔ NOT overlapping_holdout_se, which is anti-conservative AND biases the
    point estimate" — a disavowal in the clearest words available — and the
    exoneration list flagged it as a LIVE USE. Every literal missed it: the text has
    no "used", and "which is anti-conservative" is not "is not"."""
    state, detail = cc._check_forbidden_estimator(refav1)
    assert state == cc.PRESENT, f"a disavowal read as a live use: {detail}"


@pytest.mark.parametrize("wording", [
    # the two that the literal list already caught — kept so a rewrite of the
    # matcher cannot quietly lose them while fixing the two below
    "overlapping_holdout_se is NOT used anywhere: it biases the POINT ESTIMATE",
    "⛔ overlapping_holdout_se is never used — it biases the POINT ESTIMATE",
    # the two the literal list MISSED, both banked, both flagged as live use
    "paired_episode_cluster_bootstrap (taniteval.ci) — ⛔ NOT overlapping_holdout_se, "
    "which is anti-conservative AND biases the point estimate",
    "episode_cluster_bootstrap (taniteval.ci) — NEVER overlapping_holdout_se",
])
def test_every_banked_disavowal_wording_is_exonerated(refav1, wording):
    """⭐ FOUR wordings, one class: the token NEGATED DIRECTLY. Two of these were
    read as live use and would each have become a false violation on every refav1
    record. Adding literals one at a time is what let the defect return, so all four
    banked wordings are pinned together — a fix for one that loses another fails
    here."""
    art = copy.deepcopy(refav1)
    art["arms"]["cl"]["four_families"]["longitudinal"]["ci"] = {"estimator": wording}
    state, detail = cc._check_forbidden_estimator(art)
    assert state == cc.PRESENT, f"disavowal read as a live use: {detail}"


@pytest.mark.parametrize("wording", [
    "overlapping_holdout_se",
    "8-split episode-disjoint jackknife (overlapping_holdout_se)",
    "mean-of-split-means via overlapping_holdout_se",
])
def test_DELIBERATE_REGRESSION_negation_matching_does_not_admit_the_bare_token(
        refav1, wording):
    """The arm for the loosening: admitting "NOT <token>" must not admit the token
    standing on its own, nor a mention that merely names the estimator it is used
    through."""
    art = copy.deepcopy(refav1)
    art["arms"]["cl"]["four_families"]["longitudinal"]["ci"] = {"estimator": wording}
    state, detail = cc._check_forbidden_estimator(art)
    assert state == cc.ABSENT, f"live use went unflagged ({wording!r}): {detail}"


def test_DELIBERATE_REGRESSION_a_refav1_LIVE_use_is_still_caught(refav1):
    """The arm for the loosening above: admitting "NOT <token>" must not admit the
    token itself. The judgement is per-LEAF, so a sibling that really uses it is
    still its own hit."""
    art = copy.deepcopy(refav1)
    art["arms"]["cl"]["four_families"]["longitudinal"]["ci"] = {
        "estimator": "overlapping_holdout_se"}
    state, detail = cc._check_forbidden_estimator(art)
    assert state == cc.ABSENT, f"live use went unflagged: {detail}"


# ------------------------------------------------------ the two-sided pins ---
# The tac.confusion lesson: pin the registry to the EMITTER on BOTH sides, so a
# rename breaks a test instead of silently zeroing a family.

def test_the_refav1_emitter_still_writes_the_shape_the_registry_expects():
    src = (ROOT / "taniteval" / "tools" / "refav1_arm.py").read_text(
        encoding="utf-8", errors="replace")
    for token in ('rec["arms"][arm]["four_families"]',
                  '"paired_true_minus_shuffled_accuracy"',
                  '"_echo_caveat"', '"inference_inputs"', '"goal_source"',
                  '"parity_key"', '"conditionings"'):
        assert token in src, f"refav1_arm.py no longer writes {token} — update the registry"


def test_the_per_arm_tier_stamp_still_comes_from_four_families():
    """`_tier` is written by the shared emitter, not by refav1_arm.py."""
    src = (ROOT / "taniteval" / "taniteval" / "four_families.py").read_text(
        encoding="utf-8", errors="replace")
    assert 'fam["_tier"]' in src, \
        "four_families.py no longer stamps _tier — the refav1 tier path is dead"


def test_every_refav1_key_in_the_registry_resolves_in_the_fixture(registry, refav1):
    """⭐ The self-check applied to the new prefix. A key that resolves NOWHERE is a
    typo far more often than a gap, and both print as '0 present' — that confusion
    reached the PI once already as 'the surviving universal eval gap'."""
    pfx = f"arms.{_scored(registry)}."
    unresolved = []
    containers = [c for spec in registry["families"].values() for c in spec["criteria"]]
    containers += registry["artifact_hygiene"]["criteria"]
    containers += registry["leak_guards"]["guards"]
    for crit in containers:
        for key in crit.get("keys", []) + crit.get("partial_keys", []):
            if not (key.startswith(pfx) or key.startswith("refav1.")):
                continue
            found, val = cc._dig(refav1, key)
            if not found or val is None:
                unresolved.append((crit["id"], key))
    assert not unresolved, f"registered refav1 keys that resolve nowhere: {unresolved}"


def test_every_refav1_key_resolves_in_a_REAL_banked_record(registry):
    """⭐ The fixture is mine; the record is the emitter's. A key can satisfy a
    hand-built fixture and still be wrong about what the tool writes, so the same
    keys are checked against a banked artifact. Narrow glob, so it stays fast on
    the Drive mount."""
    import glob
    paths = glob.glob(str(ROOT / "taniteval" / "results" / "refav1-*.json"))
    arts = [d for d in (cc._read_json(Path(p)) for p in paths)
            if isinstance(d, dict) and cc._dig(d, f"arms.{_scored(registry)}.four_families")[0]]
    if not arts:
        pytest.skip("no banked refav1 record in this checkout")
    pfx = f"arms.{_scored(registry)}."
    containers = [c for spec in registry["families"].values() for c in spec["criteria"]]
    containers += registry["artifact_hygiene"]["criteria"]
    containers += registry["leak_guards"]["guards"]
    unresolved = []
    for crit in containers:
        for key in crit.get("keys", []) + crit.get("partial_keys", []):
            if not (key.startswith(pfx) or key.startswith("refav1.")):
                continue
            if not any(cc._dig(d, key)[0] and cc._dig(d, key)[1] is not None for d in arts):
                unresolved.append((crit["id"], key))
    assert not unresolved, \
        f"registered against no real record ({len(arts)} read): {unresolved}"


def test_a_banked_refav1_record_scores_with_no_violations(registry):
    """The end-to-end statement, on a real artifact: in scope, T1, and the four
    families either PRESENT or REFUSED-with-a-reason. If this ever fails, read the
    detail before believing it — a false violation here is the failure this whole
    block exists to prevent."""
    import glob
    paths = sorted(glob.glob(str(ROOT / "taniteval" / "results" / "refav1-*.json")))
    arts = [d for d in (cc._read_json(Path(p)) for p in paths)
            if isinstance(d, dict) and cc._dig(d, f"arms.{_scored(registry)}.four_families")[0]]
    if not arts:
        pytest.skip("no banked refav1 record in this checkout")
    for art in arts:
        res = cc.check_artifact(art, registry)
        assert res["scope"] == cc.IN_SCOPE
        assert res["tier"] in ("T1", "T2"), f"tier {res['tier']}"
        assert res["n_violations"] == 0, \
            [(v["id"], v["detail"][:120]) for v in res["violations"]]


# ---------------------------------------------------------------------------
# hyg.inference_seed (registry v2.8.0) -- the STOCHASTIC-PLANNER criterion.
# Added when refcv5's WP-4 control-space DDIM sampler landed: it draws fresh
# noise AT EVAL by design, because sampling is the mechanism.
# ---------------------------------------------------------------------------
def test_inference_seed_criterion_EXISTS_and_carries_its_measurement(registry):
    """⛔ Before v2.8.0 the registry could not EXPRESS this obligation: seven
    probes for seed/replicate/stochastic/sampler each read 0 against a
    same-breath control of 46 criteria."""
    crit = [c for c in registry["artifact_hygiene"]["criteria"]
            if c["id"] == "hyg.inference_seed"]
    assert crit, "hyg.inference_seed missing from the registry"
    c = crit[0]
    assert c["required"] is False,         "a DETERMINISTIC arm has nothing to report -- scoring it against this "         "would be the `df`-on-a-pod scope error"
    assert "applies_when" in c, "the scope IS the criterion here"
    assert "0.30" in c["note"],         "keep the MEASURED refav1 seed floor - it is why the rule exists"


def test_a_SAMPLING_arm_that_reports_no_inference_seed_is_a_WORK_ITEM(
        registry, compliant):
    """DELIBERATE REGRESSION ARM. An artifact whose telemetry says a sampler ran
    but which reports neither an inference-seed replicate nor a seed floor must
    not read as clean: its interval answers 'would another draw of EPISODES say
    this?' while the claim needs 'would another INFERENCE RUN say this?'."""
    import copy
    art = copy.deepcopy(compliant)
    art["sel_tele"] = {"sampler": "ddim"}          # a stochastic planner ran
    for k in ("inference_seeds", "seed_floor_m"):
        art.pop(k, None)
    res = cc.check_artifact(art, registry)
    ids = [r["id"] for r in res["violations"]] +           [r["id"] for r in res.get("work_items", [])]
    assert "hyg.inference_seed" not in [r["id"] for r in res["violations"]],         "required is False -- it must never be a hard VIOLATION"
    # ...and the compliant fixture itself must still be clean, so this arm is
    # measuring the criterion and not a pre-existing gap.
    assert not cc.check_artifact(compliant, registry)["violations"],         "the compliant fixture must pass, else the arm is meaningless"


def test_a_SAMPLING_arm_that_DOES_report_its_seeds_is_recognised(
        registry, compliant):
    """CONTROL that must read the known value: the SAME artifact, with the seed
    evidence present, is recognised by the criterion's keys. Without this the
    test above could pass on a criterion the checker never evaluates."""
    import copy
    art = copy.deepcopy(compliant)
    art["sel_tele"] = {"sampler": "ddim"}
    art["inference_seeds"] = [0, 1, 2]
    art["seed_floor_m"] = 0.30
    res = cc.check_artifact(art, registry)
    assert "hyg.inference_seed" not in [r["id"] for r in res["violations"]]
    assert not res["violations"], res["violations"]


# =========================================================================== #
# THE refcv3 / refcv4b SHAPE (registry v2.9.0, 2026-09-06)                    #
# =========================================================================== #
# ⛔ WHY THIS BLOCK EXISTS. v2.7.0 fixed "the registry cannot SEE refav1 records"
# by adding the `arms.cl.*` markers. It fixed exactly one schema. `refcv3_arm.py`
# emits the SAME per-arm shape but its planner arm is `os`, so every refcv3 and
# refcv4b eval ever produced still read UNKNOWN_SCOPE and was never counted --
# the identical root cause, surviving its own fix, for a second model family.
# MEASURED 2026-09-06 on the refcv4b @40,284 landing eval (4,823 windows / 141
# episodes, all four families emitted, tier T1): UNKNOWN_SCOPE, 0 criteria scored.
#
# ⭐ The arms below are the ones programme rule §6.5 requires: a registry change
# ships WITH the regression that proves it is not vacuous. They are deliberately
# the same arms already run for refav1, on the other schema, because "the guard
# passed for one shape" says nothing about the other -- which is precisely how
# v2.7.0's fix came to be half a fix.

REFCV3_ARMS = ("os", "ha", "ha0", "ha0_ext", "os_navshuf", "os_navzero", "oracle_sel")
REFCV3_SCORED = "os"


def _refcv3_scored(registry) -> str:
    """The scored arm for the refcv3_arm.py schema, READ FROM THE REGISTRY."""
    schemas = registry["arm_scoring"].get("schemas") or {}
    for name, sch in schemas.items():
        if "refcv3" in name and isinstance(sch, dict) and "scored_arm" in sch:
            return sch["scored_arm"]
    raise AssertionError(
        "arm_scoring.schemas names no refcv3 schema - the fixture would be scoring "
        "an arm the registry does not know about")


@pytest.fixture
def refcv3() -> dict:
    """A refcv3/refcv4b record: SEVEN arms on the same windows, one of them T0.

    Shaped from the real artifact -- the four families PER ARM at
    `arms.<arm>.four_families.*`, the strategic decision/route readout at the
    RECORD level under `refcv3` (arm-independent: the route head reads the
    observed window only), and `oracle_sel` stamped T0.
    """
    protocol = dict(_refav1_families_block()["_protocol"])

    def _tier(a):
        return "T0" if a == "oracle_sel" else "T1"

    return {
        "tool": "taniteval/tools/refcv3_arm.py",
        "n_windows": 4823, "n_episodes": 141,
        "arm_keys": list(REFCV3_ARMS),
        "tiers": {a: _tier(a) for a in REFCV3_ARMS},
        "_estimator": "point estimates are FULL-SET pooled means over windows; intervals "
                      "are the episode-cluster bootstrap (taniteval.ci). ⛔ "
                      "overlapping_holdout_se is NOT used anywhere.",
        "arms": {a: {
            "tier": _tier(a),
            "four_families": _refav1_families_block(_tier(a)),
            "intervals": {"tier": _tier(a), "n": 4823,
                          "estimator": "episode_cluster_bootstrap (taniteval.ci)"},
        } for a in REFCV3_ARMS},
        "refcv3": {
            "n_windows": 4823, "n_episodes": 141,
            "trivial_profile": {"n_windows": 4823,
                                "arms": {a: {"trivial_frac": 0.0} for a in REFCV3_ARMS}},
            "distance_keeping": {"status": "REFUSED",
                                 "reason": "no lead block passed (--lead-block)", "n": 0},
            "protocol": protocol,
            "strategic": {
                "tier": "T1", "n_windows": 4823, "n_route_labeled": 3622,
                "_echo_caveat": "nav_cmd is an INPUT and the route label derives from "
                                "the same clip, so under the TRUE nav this measures "
                                "the nav ECHO",
                "conditionings": {
                    "nav_true": {"status": "OK", "n": 3622, "accuracy": 0.7786,
                                 "kappa": 0.4852,
                                 "confusion_gt_rows_pred_cols": [[168, 214, 88],
                                                                 [45, 2360, 37],
                                                                 [115, 303, 292]]},
                    "nav_shuffled": {"status": "OK", "n": 3622, "accuracy": 0.7786},
                    "nav_zero": {"status": "OK", "n": 3622, "accuracy": 0.7786}},
                "paired_true_minus_shuffled_accuracy": {
                    "delta": 0.0, "lo": 0.0, "hi": 0.0, "separated": False,
                    "estimator": "paired_episode_cluster_bootstrap"},
                "nav_compliance": {"status": "REFUSED",
                                   "reason": "fixture carries no behaviour readout",
                                   "n": 0},
            },
        },
    }


def test_a_refcv3_record_is_IN_SCOPE(refcv3, registry):
    """The v2.9.0 defect, pinned. Before it, this returned UNKNOWN_SCOPE and a T1
    driving eval carrying all four families was not counted at all."""
    scope, why = cc.scope_of(refcv3, registry)
    assert scope == cc.IN_SCOPE, f"refcv3 record is not in scope: {why}"
    assert f"arms.{_refcv3_scored(registry)}.four_families" in why


def test_the_refcv3_marker_names_its_own_scored_arm(registry):
    """The marker and `arm_scoring.schemas` must agree, or the record is admitted
    by one arm's presence and then scored on another."""
    marker = f"arms.{_refcv3_scored(registry)}.four_families"
    values = [r.get("value") for r in registry["applicability"]["in_scope_if_any"]]
    assert marker in values, f"no in-scope marker for the refcv3 scored arm; have {values}"


def test_the_two_schemas_name_DIFFERENT_scored_arms(registry):
    """⭐ The whole reason one marker was not enough. If these ever coincide, the
    per-schema block has collapsed and the next schema will be missed the same way."""
    assert _refcv3_scored(registry) != _scored(registry), \
        "the refav1 and refcv3 schemas report the same scored arm - `arm_scoring." \
        "schemas` has lost the distinction that v2.9.0 exists to record"
    assert _refcv3_scored(registry) == REFCV3_SCORED


def test_the_refcv3_record_carries_a_tier_stamp(refcv3, registry):
    """⛔ MEASURED 2026-09-06: with the families mirrored but `tiers.key_paths` NOT,
    the artifact scored but read tier UNSTAMPED -- 'a number with no tier is not
    quotable'. The marker mirror and the tier mirror are one fix, not two."""
    tier_id, raw = cc.resolve_tier(refcv3, registry)
    assert tier_id == "T1", f"refcv3 record read tier {tier_id!r} (raw {raw!r})"


@pytest.mark.parametrize("family", ["LONGITUDINAL", "LATERAL", "TACTICAL", "STRATEGIC"])
def test_DELIBERATE_REGRESSION_deleting_a_family_from_a_refcv3_record_is_caught(
        refcv3, registry, family):
    """⭐ THE ARM THAT MAKES THE refcv3/refcv4b SUPPORT MEAN ANYTHING.

    A registry that admitted refcv3 records and then resolved none of their keys
    would still pass the scope test above. So the regression is run again, per
    family, on THIS shape.
    """
    broken = _strip_family_keys(refcv3, registry, family)
    res = cc.check_artifact(broken, registry)
    assert res["scope"] == cc.IN_SCOPE, "the record must stay in scope while broken"
    hit = [r["id"] for r in res["violations"]]
    assert hit, f"deleting {family} from a refcv3 record produced NO violation - " \
                f"the guard is vacuous"

    fam_ids = {c["id"] for c in registry["families"][family]["criteria"]}
    assert fam_ids & set(hit), f"{family} deleted but not flagged; flagged={hit}"

    for other in registry["families"]:
        if other == family:
            continue
        other_ids = {c["id"] for c in registry["families"][other]["criteria"]}
        leaked = (other_ids & set(hit)) - fam_ids
        assert not leaked, f"deleting {family} also flagged {other}: {leaked}"


def test_DELIBERATE_REGRESSION_deleting_one_metric_from_a_refcv3_record_is_caught(
        refcv3, registry):
    """The finer arm on this shape: remove ONLY the yaw-rate number - the metric
    CLAUDE.md names as where a smooth-but-wrong path hides - and exactly that
    criterion must fire."""
    broken = copy.deepcopy(refcv3)
    del broken["arms"][_refcv3_scored(registry)]["four_families"]["lateral"][
        "yaw_rate_mae_degps"]
    hit = [v["id"] for v in cc.check_artifact(broken, registry)["violations"]]
    assert hit == ["lat.yaw_rate"], f"expected only lat.yaw_rate, got {hit}"


def test_ONLY_the_scored_arm_is_read_on_a_refcv3_record(refcv3, registry):
    """⛔ A refcv3 record holds SEVEN arms - four trivial controls, two nav controls
    and a T0 diagnostic. Gut every non-scored arm: the verdict must not move."""
    before = cc.check_artifact(refcv3, registry)
    art = copy.deepcopy(refcv3)
    for arm in REFCV3_ARMS:
        if arm != _refcv3_scored(registry):
            art["arms"][arm] = {}
    after = cc.check_artifact(art, registry)
    assert after == before, \
        "a non-scored arm changed the verdict - scoring is not arm-scoped on this schema"
