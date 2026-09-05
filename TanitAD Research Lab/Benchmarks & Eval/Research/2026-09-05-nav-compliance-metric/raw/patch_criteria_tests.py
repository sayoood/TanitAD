"""Patch tools/tests/test_criteria_check.py (HEAD content) with the nav-compliance
fixture + tests. Structure from the predecessor's draft, keys from the emitter."""
import sys

p = sys.argv[1]
src = open(p, encoding="utf-8", newline="").read()
if "strat.nav_compliance" in src:
    raise SystemExit("already patched")
old_fix = '''        "strategic": {"decision_accuracy": 0.66, "route_quality": 0.71,
                      "echo_test": {"bijection_with_input": False}},
    }
'''
new_fix = '''        "strategic": {"decision_accuracy": 0.66, "route_quality": 0.71,
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
'''
assert src.count(old_fix) == 1, src.count(old_fix)
src = src.replace(old_fix, new_fix)
src = src.rstrip("\n") + '''


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
    assert "bijection" in ids["strat.nav_compliance"]["note"], \\
        "keep the reason the criterion exists — the metric it replaces could not fail"
    assert all(k.endswith("paired_true_minus_shuffled")
               for k in ids["strat.nav_compliance_ctrl_shuffle"]["keys"])
    assert all(k.endswith("paired_true_minus_zero")
               for k in ids["strat.nav_compliance_ctrl_zero"]["keys"])


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
'''
open(p, "w", encoding="utf-8", newline="").write(src + "\n")
print("criteria tests patched")
