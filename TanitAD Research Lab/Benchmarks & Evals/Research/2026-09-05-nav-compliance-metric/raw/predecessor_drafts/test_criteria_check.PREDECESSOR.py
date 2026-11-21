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
                      # ⭐ nav-COMPLIANCE (PI 2026-09-05): behaviour vs the TRUE
                      # command, admissible ONLY with both intervention controls
                      # paired on the same windows. Keys pinned to the emitter
                      # (taniteval/taniteval/nav_compliance.py::compliance_report).
                      "nav_compliance": {
                          "readouts": {"plan": {"nav_true": {"imminent": {"rate": 0.8}}}},
                          "controls": {"nav_shuffle": {"plan": {"imminent": {"delta": 0.3}}},
                                       "nav_zero": {"plan": {"imminent": {"delta": 0.2}}}},
                          "verdict": {"nav_effect": "FOLLOWS_NAV"}}},
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


# ----------------------------- nav-COMPLIANCE (PI 2026-09-05) ------------------
# "we are not evaluating the nav command itself, we are evaluating the fact that
# the model is following the nav command in consistency to the strategic goals."
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


def test_nav_compliance_registry_keys_match_the_emitter():
    """⭐ The two-sided pin (the tac.confusion lesson): the registry names the
    keys the EMITTER writes, so renaming either breaks a test instead of
    silently zeroing the family."""
    src = (ROOT / "taniteval" / "taniteval" / "nav_compliance.py").read_text(
        encoding="utf-8", errors="replace")
    for key in ('"readouts"', '"controls"', '"consistency"', '"verdict"',
                '"known_value_controls"'):
        assert key in src, f"the emitter no longer writes {key}"
    assert "def compliance_report" in src


@pytest.mark.parametrize("control,cid", [
    ("nav_shuffle", "strat.nav_compliance_ctrl_shuffle"),
    ("nav_zero", "strat.nav_compliance_ctrl_zero"),
])
def test_DELIBERATE_REGRESSION_nav_compliance_without_a_control_is_a_violation(
        compliant, registry, control, cid):
    """A compliance rate whose intervention control is missing is coincidence
    wearing a result's clothes. Delete ONE control and exactly that criterion
    must fire — a guard that stays green here is decorative."""
    broken = copy.deepcopy(compliant)
    del broken["strategic"]["nav_compliance"]["controls"][control]
    res = cc.check_artifact(broken, registry)
    hit = [v["id"] for v in res["violations"]]
    assert cid in hit, f"deleting the {control} control was not flagged: {hit}"
    other = {"strat.nav_compliance_ctrl_shuffle",
             "strat.nav_compliance_ctrl_zero"} - {cid}
    assert not (other & set(hit)), f"the OTHER control was flagged too: {hit}"


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
    refused = {r["id"] for r in res.get("work_items", res.get("refused", []))} \
        if isinstance(res.get("work_items", res.get("refused")), list) else set()
    states = {r["id"]: r["state"] for r in res["results"]} if "results" in res else {}
    if states:
        assert states["strat.nav_compliance"] == cc.REFUSED
        assert states["strat.nav_compliance_ctrl_zero"] == cc.REFUSED


def test_nav_compliance_inline_UNAVAILABLE_is_a_work_item(compliant, registry):
    """The emitter's own idiom for an empty stratum: {status: UNAVAILABLE,
    reason, n} at the key. Read as REFUSED, never as PRESENT or ABSENT."""
    art = copy.deepcopy(compliant)
    art["strategic"]["nav_compliance"]["readouts"] = {
        "status": "UNAVAILABLE", "reason": "no informative window", "n": 0}
    res = cc.check_artifact(art, registry)
    assert not [v for v in res["violations"] if v["id"] == "strat.nav_compliance"]
