"""The release gate, pinned by DELIBERATE REGRESSION — one arm per criterion.

⭐ A GATE THAT HAS NEVER BEEN SHOWN TO FAIL PROVES NOTHING.

That is not a slogan here, it is the programme's measured history. The O6 rank
gate returned INCONCLUSIVE forever because it compared a spectrum of n=24 against
a ceiling of 1024 — structurally incapable of ruling, and nobody noticed because
it never failed either. `nonav_route_beats_majority` sat in a KILL set where it
could not pass, for a label-bug reason nothing to do with any checkpoint. And the
banked T1 corpus is four-families COMPLETE while a hold-action control beats the
model by 22x on ADE — completeness alone was blind to that.

So every criterion in `release_gate.py` gets an arm below that breaks exactly it
and asserts the gate FAILS on exactly it. Where a breakage legitimately cascades,
the cascade is DECLARED in `collateral` — an undeclared extra failure fails the
test, so the blast radius of each criterion stays pinned too.
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
import release_gate as rg  # noqa: E402

REGISTRY_PATH = ROOT / "products" / "P7-TanitEval" / "CRITERIA_REGISTRY.json"
GRID = {"n_windows": 6844, "horizon_steps": 20, "dt_s": 0.1,
        "n_episodes": 40, "eid_sha1": "deadbeefcafe0001", "gt_sha1": "deadbeefcafe0002"}


@pytest.fixture(scope="module")
def registry() -> dict:
    reg = cc._read_json(REGISTRY_PATH)
    assert reg is not None, f"registry unreadable at {REGISTRY_PATH}"
    return reg


# --------------------------------------------------------------- fixtures ---
def _arm(arm: str, arm_key: str, *, tier="T1", scale=1.0, kappa=0.55) -> dict:
    """One four_families artifact. `scale` multiplies every ERROR metric, so a
    control or a baseline is built by scaling a single knob rather than by hand-
    typing numbers that could silently disagree with each other."""
    return {
        "arm": arm, "arm_key": arm_key, "tier": tier,
        "n_windows": 6844, "n_episodes": 40,
        "grid_fingerprint": dict(GRID),
        "parity": {"corpus": rg.CANON_CORPUS, "skip_hash": rg.CANON_SKIP_HASH},
        "protocol": {
            "inference_inputs": "front-wide camera frames only",
            "vision_only": True,
            "goal_source": "predicted goal point from the vision trunk; the situation "
                           "head is information-disjoint at inference",
        },
        "_estimator": "point estimates are full_set means; intervals are the "
                      "episode_cluster_bootstrap (taniteval.ci). "
                      "overlapping_holdout_se is NOT used.",
        "four_families": {
            "longitudinal": {
                "speed_mae_mps": 0.40 * scale, "speed_bias_mps": 0.11 * scale,
                "along_mae_m": 0.62 * scale, "along_bias_m": 0.20 * scale,
                "ego_progress": {"status": "OK", "progress_ratio_mean": 1.02, "n": 6844},
                "distance_keeping": {"status": "UNAVAILABLE", "n": 0,
                                     "reason": "no lead-agent track supplied; "
                                               "PhysicalAI-AV lead_state is a None stub"},
                "n_windows": 6844, "tier": tier,
            },
            "lateral": {
                "heading_mae_deg": 2.10 * scale, "yaw_rate_mae_degps": 2.60 * scale,
                "curvature_mae_1pm": 0.0090 * scale, "cross_mae_m": 0.14 * scale,
                "n_windows": 6844, "tier": tier,
            },
            "tactical": {
                "status": "OK", "n": 6844,
                "lateral_decision": {"status": "OK", "accuracy": 0.82, "kappa": kappa,
                                     "confusion_gt_rows_pred_cols": [[9, 1], [1, 9]],
                                     "never_predicted": [], "n": 6844},
                "longitudinal_decision": {"status": "OK", "accuracy": 0.61,
                                          "kappa": kappa - 0.10,
                                          "confusion_gt_rows_pred_cols": [[7, 3], [2, 8]],
                                          "never_predicted": [], "n": 6844},
                "maneuver_5way_collapsed": {"status": "OK", "accuracy": 0.58,
                                            "kappa": 0.31, "never_predicted": [],
                                            "confusion_gt_rows_pred_cols": [[5, 1], [1, 5]],
                                            "n": 6844},
                "goal_setting": {"status": "OK", "goal_point_error_m": 1.05 * scale,
                                 "anchor_selection": {"top1": 0.71}, "n": 6844},
                "_estimator": "episode_cluster_bootstrap; overlapping_holdout_se is "
                              "NOT used",
                "tier": tier,
            },
            # PRESENT, not refused: a PASS fixture must not lean on a refusal, or the
            # regression arm that deletes the family would have nothing to delete.
            "strategic": {"status": "OK", "decision_accuracy": 0.64,
                          "route_quality": 0.68, "n": 6844, "tier": tier},
            "_tier": tier,
        },
        "strategic": {"decision_accuracy": 0.64, "route_quality": 0.68,
                      "echo_test": {"bijection_with_input": False,
                                    "matched": 41, "n": 369}},
        # `ctrl.floor_comparison` (registry v2.2.0): the artifact must DECLARE its
        # trivial floor. The gate then does the cross-artifact comparison the
        # declaration alone cannot do.
        "floors": {
            "hold_v0": {"ade_dense_m": 0.44 * 3.0, "arm": f"{arm.split(':')[0]}:ha"},
            "_estimator": "paired_episode_cluster_bootstrap on the same windows",
        },
        "intervals": {
            "tier": tier, "estimator": "episode_cluster_bootstrap",
            "n_windows": 6844,
            "metrics": {
                "ade_dense_m": {"mean": 0.44 * scale, "lo": 0.39, "hi": 0.50,
                                "n_episodes": 40, "n_windows": 6844,
                                "estimator": "episode_cluster_bootstrap"},
                "fde_last_m": {"mean": 1.05 * scale, "lo": 0.90, "hi": 1.22,
                               "n_episodes": 40, "n_windows": 6844,
                               "estimator": "episode_cluster_bootstrap"},
            },
        },
    }


def _write(d: Path, docs: dict) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    for name, doc in docs.items():
        (d / name).write_text(json.dumps(doc, ensure_ascii=False, indent=1),
                              encoding="utf-8")
    return d


def _release_docs() -> dict:
    """A synthetic PASSING release: the model arm, its control, its baseline.

    The control is 3x WORSE and the baseline 1.4x worse, so the passing state is
    the model genuinely beating both — not an accident of a missing comparison.
    """
    return {
        "ff_relX_cl.json": _arm("relX:cl", "cl", tier="T1", scale=1.0, kappa=0.55),
        "ff_relX_ha.json": _arm("relX:ha", "ha", tier="T1", scale=3.0, kappa=0.20),
        "ff_relX_ol.json": _arm("relX:ol", "ol", tier="T0", scale=0.9, kappa=0.57),
        "ff_baseW_cl.json": _arm("baseW:cl", "cl", tier="T1", scale=1.4, kappa=0.50),
        "ff_baseW_ha.json": _arm("baseW:ha", "ha", tier="T1", scale=3.0, kappa=0.20),
    }


def _run(tmp_path, registry, docs, *, model="relX", baseline="baseW", **kw):
    d = _write(tmp_path / "arts", docs)
    cfg = rg.Cfg(model=model, baseline=baseline,
                 primary_arm_key=kw.pop("primary_arm_key", "cl"),
                 control_margin=kw.pop("control_margin", 0.0),
                 regression_tol=kw.pop("regression_tol", 0.0),
                 accept_regression=kw.pop("accept_regression", {}),
                 parity_key=rg.CANON_CORPUS, parity_skip_hash=rg.CANON_SKIP_HASH,
                 echo_threshold=kw.pop("echo_threshold", 0.99),
                 episode_floor=kw.pop("episode_floor", rg.EPISODE_CLUSTER_FLOOR),
                 regression_blocking=kw.pop("regression_blocking", False))
    assert not kw, f"unused kwargs {kw}"
    return rg.run_gate(str(d), cfg, registry)


def _states(res) -> dict:
    return {r.id: r.state for r in res["_rows"]}


def _failed(res) -> set:
    return {r.id for r in res["_rows"] if r.state == rg.FAIL}


def _cannot(res) -> set:
    return {r.id for r in res["_rows"] if r.state == rg.CANNOT_RULE}


# --------------------------------------------------- the passing reference ---
def test_the_synthetic_release_PASSES(tmp_path, registry):
    """If this ever fails the regression arms below mean nothing — they would be
    detecting the fixture's own defects instead of the injected one."""
    res = _run(tmp_path, registry, _release_docs())
    assert _failed(res) == set(), \
        json.dumps([r.as_dict() for r in res["_rows"] if r.state == rg.FAIL], indent=1)
    assert _cannot(res) == set(), \
        json.dumps([r.as_dict() for r in res["_rows"] if r.state == rg.CANNOT_RULE],
                   indent=1)
    assert res["verdict"].startswith("ADVISORY-PASS")


def test_the_verdict_is_never_a_pooled_score(tmp_path, registry):
    """⛔ Per-family, always. A single composite hides the trade-off we are trying
    to see (PI, 2026-08-02, binding)."""
    res = _run(tmp_path, registry, _release_docs())
    assert isinstance(res["verdict"], str) and not res["verdict"].replace(
        "-", "").replace(".", "").isdigit()
    fams = {r.family for r in res["_rows"]}
    assert {"LONGITUDINAL", "LATERAL", "TACTICAL", "STRATEGIC"} <= fams
    txt = rg.render(res)
    assert "PER-FAMILY PANEL" in txt and "never pooled" in txt


def test_the_scope_it_swept_is_printed(tmp_path, registry):
    """A gate that silently swept one directory reports THAT DIRECTORY's verdict,
    not the model's. Every file read must be named, including the excluded ones."""
    res = _run(tmp_path, registry, _release_docs())
    txt = rg.render(res)
    assert "SCOPE SWEPT" in txt
    for name in _release_docs():
        assert name in txt, f"{name} was read but not disclosed in the scope block"
    assert "NOT-THIS-MODEL" in txt, "excluded files must be visible, not silently dropped"


# =================================================================================
#  ⭐ DELIBERATE REGRESSION — one arm per criterion. THE CORE OF THIS FILE.
# =================================================================================
def _break_rg01(docs):        # no artifact matches the model key
    return docs, {"model": "no-such-release"}


def _break_rg02(docs):        # a whole family silently deleted
    del docs["ff_relX_cl.json"]["four_families"]["tactical"]
    return docs, {}


def _break_rg03(docs):        # tier stamp stripped everywhere it lives
    d = docs["ff_relX_ha.json"]
    d.pop("tier", None)
    d["four_families"].pop("_tier", None)
    d["intervals"].pop("tier", None)
    for blk in ("longitudinal", "lateral", "tactical", "strategic"):
        d["four_families"][blk].pop("tier", None)
    return docs, {}


def _break_rg04(docs):        # T0 only — a WM diagnostic is NEVER driving performance
    for name in ("ff_relX_cl.json", "ff_relX_ha.json"):
        docs[name]["tier"] = "T0"
        docs[name]["four_families"]["_tier"] = "T0"
        docs[name]["intervals"]["tier"] = "T0"
    return docs, {}


def _break_rg05(docs):        # an estimator that is named but is the WRONG one
    d = docs["ff_relX_cl.json"]
    d["intervals"]["estimator"] = "iid_window_bootstrap"
    d["_estimator"] = "iid_window_bootstrap over windows"
    for m in d["intervals"]["metrics"].values():
        m["estimator"] = "iid_window_bootstrap"
    d["four_families"]["tactical"]["_estimator"] = "iid_window_bootstrap"
    return docs, {}


def _break_rg06(docs):        # the forbidden estimator, used and not disavowed
    docs["ff_relX_cl.json"]["intervals"]["interval_method"] = "overlapping_holdout_se"
    return docs, {}


def _break_rg07(docs):        # ego state at inference — a LEAK, not a capability
    docs["ff_relX_cl.json"]["protocol"]["inference_inputs"] = \
        "camera frames + ego_state (v0, yaw rate)"
    return docs, {}


def _break_rg08(docs):        # the goal carries the situation classifier's output
    docs["ff_relX_cl.json"]["protocol"]["goal_source"] = \
        "goal point conditioned on the situation classifier argmax"
    return docs, {}


def _break_rg09(docs):        # a 1.0000 route score with no echo test = the v1 defect
    docs["ff_relX_cl.json"]["strategic"]["route_quality"] = 1.0
    docs["ff_relX_cl.json"]["strategic"].pop("echo_test")
    return docs, {}


def _break_rg10(docs):        # the control beats the model — the 22x case
    docs["ff_relX_ha.json"] = _arm("relX:ha", "ha", tier="T1", scale=0.25, kappa=0.90)
    return docs, {}


def _break_rg11(docs):        # a family regresses vs the baseline, no trade declared
    docs["ff_baseW_cl.json"] = _arm("baseW:cl", "cl", tier="T1", scale=0.5, kappa=0.50)
    return docs, {"regression_blocking": True}


def _break_rg12(docs):        # parity stamp gone — the corpus is unverifiable
    docs["ff_relX_cl.json"].pop("parity")
    return docs, {}


def _break_rg13(docs):        # the control sits on a DIFFERENT window set
    docs["ff_relX_ha.json"]["grid_fingerprint"]["eid_sha1"] = "0000000000000000"
    return docs, {}


# id -> (breaker, DECLARED collateral). An undeclared extra failure fails the test,
# so each criterion's blast radius is pinned as tightly as the criterion itself.
REGRESSION_ARMS = {
    "RG-01": (_break_rg01, {"RG-01"}),
    "RG-02": (_break_rg02, {"RG-02"}),
    "RG-03": (_break_rg03, {"RG-03"}),
    "RG-04": (_break_rg04, {"RG-04"}),
    "RG-05": (_break_rg05, {"RG-05"}),
    # the forbidden estimator is ALSO a registry completeness criterion
    # (hyg.no_forbidden_estimator), so RG-02 legitimately fires with it
    "RG-06": (_break_rg06, {"RG-06", "RG-02"}),
    "RG-07": (_break_rg07, {"RG-07"}),
    "RG-08": (_break_rg08, {"RG-08"}),
    "RG-09": (_break_rg09, {"RG-09"}),
    # breaking the floor breaks it in every family at once, by construction
    "RG-10.LON": (_break_rg10, {"RG-10.LON", "RG-10.LAT", "RG-10.TAC", "RG-10.CMP"}),
    "RG-11.LON": (_break_rg11, {"RG-11.LON", "RG-11.LAT", "RG-11.TAC"}),
    # parity is ALSO a registry completeness criterion (hyg.parity)
    "RG-12": (_break_rg12, {"RG-12", "RG-02"}),
    "RG-13": (_break_rg13, {"RG-13"}),
}


@pytest.mark.parametrize("cid", sorted(REGRESSION_ARMS))
def test_DELIBERATE_REGRESSION_each_criterion_can_actually_fail(tmp_path, registry, cid):
    """Break exactly one criterion; the gate must FAIL on exactly that one.

    Both halves matter. If the target does not fail, the criterion is decorative.
    If something ELSE fails too and was not declared, the criterion is not isolated
    and a future reader cannot tell which defect a red gate is pointing at.
    """
    breaker, collateral = REGRESSION_ARMS[cid]
    docs, kw = breaker(copy.deepcopy(_release_docs()))
    res = _run(tmp_path, registry, docs, **kw)
    failed = _failed(res)
    assert cid in failed, (
        f"{cid} was deliberately broken and the gate stayed green on it. "
        f"states={json.dumps(_states(res), indent=1)}")
    assert failed <= collateral, (
        f"{cid} broke UNDECLARED collateral {sorted(failed - collateral)} — "
        f"either the criteria are entangled or the breaker is too coarse")
    assert res["verdict"].startswith("ADVISORY-FAIL")


def test_the_registry_decides_what_blocks_not_this_tool(tmp_path, registry):
    """The criterion set is `CRITERIA_REGISTRY.json:release_gate.advisory_all`. A
    rule living only in the tool would drift from the census the first time either
    was edited — the decay the registry exists to stop.

    ⚠️ PI ruling 2026-08-28: the gate is ADVISORY. The rows must still be PRODUCED
    and still carry their `blocking_why` provenance — the tool's internal
    `blocking` flag now means "this is a criterion the registry named", not "this
    stops a release". Authority moved to the PI; measurement did not move."""
    res = _run(tmp_path, registry, _release_docs())
    by_id = {r.id: r for r in res["_rows"]}
    for entry in registry["release_gate"]["advisory_all"]:
        for rid in rg._REGISTRY_TO_RG.get(entry, ()):
            rows = [r for r in res["_rows"] if r.id.split(".")[0] == rid]
            assert rows, f"registry blocks on {entry!r} but no {rid} row exists"
            assert all(r.blocking for r in rows), f"{entry!r} must block"
            assert entry in rows[0].blocking_why
    assert by_id["RG-04"].blocking, "a T1 artifact is REQUIRED"


def test_regression_is_ADVISORY_until_the_registry_says_otherwise(tmp_path, registry):
    """registry release_gate.advisory: regression is advisory until two clean
    releases exist to compare, or it manufactures false failures. The criterion
    still RUNS and still says FAIL — it just does not block."""
    docs, _ = _break_rg11(copy.deepcopy(_release_docs()))
    res = _run(tmp_path, registry, docs)          # note: no regression_blocking
    row = next(r for r in res["_rows"] if r.id == "RG-11.LON")
    assert row.state == rg.FAIL and row.blocking is False
    assert "advisory" in row.blocking_why
    assert res["counts"]["ADVISORY_FAIL"] >= 1
    assert "[ADVISORY]" in rg.render(res)

    # ⛔ CHANGED BY THE PI RULING OF 2026-08-28, deliberately.
    # Before the ruling this asserted the verdict stayed "RELEASE-CLEARED" when only
    # an advisory criterion failed — because "cleared" then meant "nothing BLOCKING
    # failed". Now nothing blocks at all, so a verdict of PASS would have to mean
    # "nothing failed". Reporting PASS while a regression genuinely failed is exactly
    # the softening the ruling must not cause: advisory is about AUTHORITY, not
    # honesty. So the verdict names the failure...
    assert res["verdict"].startswith("ADVISORY-FAIL"),         "a real failure must be named in the verdict even when nothing blocks"
    # ...while the escalation path stays clear: nothing registry-named failed, so the
    # exit-code contract still reports 0 and no process is asked to stop.
    assert res["counts"]["BLOCKING_FAIL"] == 0,         "an advisory-only failure must not escalate to the registry-named count"


def test_a_broken_gate_exits_non_zero(tmp_path, registry, monkeypatch, capsys):
    """The exit code is the only thing CI reads."""
    docs, _ = _break_rg02(copy.deepcopy(_release_docs()))
    d = _write(tmp_path / "arts", docs)
    code = rg.main(["--model", "relX", "--artifacts", str(d), "--baseline", "baseW"])
    assert code == 1, "a FAILing release must exit 1"
    assert "ADVISORY-FAIL" in capsys.readouterr().out


def test_a_clean_release_exits_zero(tmp_path, registry, capsys):
    d = _write(tmp_path / "arts", _release_docs())
    code = rg.main(["--model", "relX", "--artifacts", str(d), "--baseline", "baseW"])
    assert code == 0
    assert "ADVISORY-PASS" in capsys.readouterr().out


# =================================================================================
#  ⭐ CANNOT-RULE — the state the O6 rank gate needed and did not have
# =================================================================================
def test_CANNOT_RULE_when_n_episodes_is_below_the_cluster_floor(tmp_path, registry):
    """n=3 episodes cannot support an episode-cluster bootstrap however many
    windows sit inside them. The gate must SAY SO — not pass, not hang on a
    perpetual INCONCLUSIVE, and not quietly report an interval it cannot resolve."""
    docs = copy.deepcopy(_release_docs())
    for name in ("ff_relX_cl.json", "ff_relX_ha.json", "ff_baseW_cl.json"):
        docs[name]["n_episodes"] = 3
        docs[name]["grid_fingerprint"]["n_episodes"] = 3
    res = _run(tmp_path, registry, docs)
    assert "RG-14" in _cannot(res), json.dumps(_states(res), indent=1)
    row = next(r for r in res["_rows"] if r.id == "RG-14")
    assert row.can_rule is False
    assert "n_episodes=3" in row.detail and "floor" in row.detail
    assert "RG-10" in row.detail and "RG-11" in row.detail, \
        "an unrulable criterion must name what it DISARMS downstream"
    assert res["verdict"].startswith("ADVISORY-INCONCLUSIVE")
    assert "CANNOT RULE" in rg.render(res)


def test_CANNOT_RULE_when_no_control_was_banked(tmp_path, registry):
    """No floor => the headline cannot be read. MEASURED: a complete artifact whose
    control beat it 22x. Absence of the control is not absence of the problem."""
    docs = copy.deepcopy(_release_docs())
    docs.pop("ff_relX_ha.json")
    res = _run(tmp_path, registry, docs)
    assert "RG-10" in _cannot(res), json.dumps(_states(res), indent=1)
    assert res["verdict"].startswith("ADVISORY-INCONCLUSIVE")


def test_CANNOT_RULE_when_the_named_baseline_is_not_in_scope(tmp_path, registry):
    """A named-but-absent baseline is not a pass — that is how a regression check
    silently stops checking."""
    docs = copy.deepcopy(_release_docs())
    docs.pop("ff_baseW_cl.json")
    docs.pop("ff_baseW_ha.json")
    res = _run(tmp_path, registry, docs)
    assert "RG-11" in _cannot(res)


def test_CANNOT_RULE_when_the_baseline_sits_on_a_different_grid(tmp_path, registry):
    """A delta across different windows is not a delta."""
    docs = copy.deepcopy(_release_docs())
    docs["ff_baseW_cl.json"]["grid_fingerprint"]["gt_sha1"] = "1111111111111111"
    docs["ff_baseW_ha.json"]["grid_fingerprint"]["gt_sha1"] = "1111111111111111"
    res = _run(tmp_path, registry, docs)
    assert "RG-11" in _cannot(res)


def test_cannot_rule_exits_two_not_zero(tmp_path, registry, capsys):
    """⛔ CANNOT-RULE must never be reported through a success exit code."""
    docs = copy.deepcopy(_release_docs())
    docs.pop("ff_relX_ha.json")
    d = _write(tmp_path / "arts", docs)
    code = rg.main(["--model", "relX", "--artifacts", str(d), "--baseline", "baseW"])
    assert code == 2
    assert "ADVISORY-INCONCLUSIVE" in capsys.readouterr().out


# =================================================================================
#  the doctrine's finer edges
# =================================================================================
def test_a_refusal_WITH_a_reason_is_admissible(tmp_path, registry):
    """An eval that cannot compute a family and SAYS so, with its reason and its n,
    is doing the right thing. That is the whole REFUSED/ABSENT distinction."""
    docs = copy.deepcopy(_release_docs())
    docs["ff_relX_cl.json"]["four_families"]["strategic"] = {
        "status": "UNAVAILABLE", "n": 6844,
        "reason": "PhysicalAI-AV carries no map, no lane graph and no route signal"}
    docs["ff_relX_cl.json"].pop("strategic")
    res = _run(tmp_path, registry, docs)
    assert "RG-02" not in _failed(res), json.dumps(_states(res), indent=1)


def test_a_refusal_with_an_EMPTY_reason_is_a_shrug_and_fails(tmp_path, registry):
    docs = copy.deepcopy(_release_docs())
    docs["ff_relX_cl.json"]["four_families"]["strategic"] = {
        "status": "UNAVAILABLE", "n": 6844, "reason": ""}
    docs["ff_relX_cl.json"].pop("strategic")
    res = _run(tmp_path, registry, docs)
    assert "RG-02" in _failed(res)


def test_a_declared_trade_converts_a_regression_FAIL_into_a_recorded_PASS(
        tmp_path, registry):
    """'must not regress, OR must state the trade explicitly' — the second clause,
    exercised. A blank trade is rejected at the CLI (see below)."""
    docs, _ = _break_rg11(copy.deepcopy(_release_docs()))
    res = _run(tmp_path, registry, docs,
               accept_regression={"LONGITUDINAL": "traded for the tactical fix; "
                                                  "pre-registered in D-0xx",
                                  "LATERAL": "same trade",
                                  "TACTICAL": "same trade"})
    assert not any(r.id.startswith("RG-11") and r.state == rg.FAIL
                   for r in res["_rows"])
    row = next(r for r in res["_rows"] if r.id == "RG-11.LON")
    assert "TRADE DECLARED" in row.detail and "REGRESSED" in row.detail


def test_a_blank_trade_is_refused(tmp_path, registry):
    with pytest.raises(SystemExit):
        rg._kv_list(["LATERAL="])


def test_naming_the_forbidden_estimator_to_DISAVOW_it_is_not_a_violation(
        tmp_path, registry):
    """The best-disciplined artifacts in the repo say 'overlapping_holdout_se is
    NOT used' verbatim. Flagging those would punish exactly the right behaviour."""
    res = _run(tmp_path, registry, _release_docs())
    blob = json.dumps(_release_docs()["ff_relX_cl.json"])
    assert "overlapping_holdout_se" in blob
    assert "RG-06" not in _failed(res)


def test_an_unknown_scope_artifact_is_disclosed_and_never_counted_compliant(
        tmp_path, registry):
    """The ff_comparison.full.json shape in the real corpus. It must be visible."""
    docs = copy.deepcopy(_release_docs())
    docs["ff_relX_comparison.json"] = {"tool": "ff_rescore.py",
                                       "arms": {"relX:cl": {"tier": "T1"}}}
    res = _run(tmp_path, registry, docs)
    row = next(s for s in res["scope"] if "comparison" in s["file"])
    assert row["role"] == rg.UNKNOWN
    assert "UNKNOWN-SCOPE" in rg.render(res)
    assert "RG-01" not in _failed(res), "an UNKNOWN artifact must not be counted in"


def test_T0_alone_cannot_clear_a_release(tmp_path, registry):
    """⛔ Stated twice because it has to be: T0 is a world-model diagnostic and is
    NEVER driving performance."""
    docs, _ = _break_rg04(copy.deepcopy(_release_docs()))
    res = _run(tmp_path, registry, docs)
    row = next(r for r in res["_rows"] if r.id == "RG-04")
    assert row.state == rg.FAIL
    assert "T0" in row.detail and "NEVER driving performance" in row.detail


def test_the_wholly_unmeasured_family_is_announced_loudly(tmp_path, registry):
    """A family whose every criterion is N/A is invisible in a table of passes —
    and the hierarchy is the programme's thesis."""
    docs = copy.deepcopy(_release_docs())
    for name in list(docs):
        docs[name]["four_families"]["strategic"] = {
            "status": "UNAVAILABLE", "n": 6844, "reason": "no map in this corpus"}
        docs[name].pop("strategic", None)
    res = _run(tmp_path, registry, docs)
    txt = rg.render(res)
    assert "STRATEGIC IS WHOLLY UNMEASURED" in txt, txt[-2500:]


def test_the_json_report_carries_every_criterion_and_the_scope(tmp_path, registry):
    docs = _release_docs()
    d = _write(tmp_path / "arts", docs)
    out = tmp_path / "verdict.json"
    rg.main(["--model", "relX", "--artifacts", str(d), "--baseline", "baseW",
             "--json", str(out)])
    rep = json.loads(out.read_text(encoding="utf-8"))
    assert rep["verdict"].startswith("ADVISORY-PASS")
    assert len(rep["scope"]) == len(docs)
    assert {c["id"] for c in rep["criteria"]} >= {
        "RG-01", "RG-02", "RG-03", "RG-04", "RG-05", "RG-06", "RG-07", "RG-08",
        "RG-09", "RG-12", "RG-13", "RG-14"}
    assert "NOT a pooled score" in rep["verdict_is"]


def test_every_registry_family_has_a_gate_metric_or_is_declared_absent(registry):
    """Registry hygiene, the release-gate half: a family the registry binds but the
    gate never reads would be silently ungated — the same shape as the criteria
    registry's own UNRESOLVABLE self-check."""
    covered = {s.family for s in rg.FAMILY_METRICS}
    assert set(registry["families"]) <= covered, \
        f"registry families with no gate metric: {set(registry['families']) - covered}"



# ------------------------- the PI ruling: advisory, still honest (2026-08-28) ---

def test_the_verdict_no_longer_claims_blocking_authority(tmp_path, registry):
    """⛔ PI ruling 2026-08-28: the gate does not block a release. The word must go
    from the verdict — a tool that keeps announcing an authority it does not have is
    lying about the process."""
    docs, _ = _break_rg11(copy.deepcopy(_release_docs()))
    res = _run(tmp_path, registry, docs)
    assert "BLOCKED" not in res["verdict"].upper().replace("DOES NOT BLOCK", "")
    assert res["verdict"].startswith("ADVISORY")


def test_advisory_did_NOT_soften_the_counts(tmp_path, registry):
    """⭐ THE REGRESSION ARM FOR THE RULING ITSELF.

    Advisory is about AUTHORITY, not honesty. A FAIL must still be counted and named
    a FAIL. If making the gate advisory had quietly reduced the failure count, the
    instrument would be worthless — and that drift is exactly what this programme
    keeps paying for.
    """
    docs, _ = _break_rg11(copy.deepcopy(_release_docs()))
    res = _run(tmp_path, registry, docs)
    n_fail = res["counts"]["FAIL"]
    assert n_fail > 0, "the broken fixture must still produce FAILs"
    assert str(n_fail) in res["verdict"], "the verdict must carry the honest count"


def test_the_exit_code_contract_survives_the_ruling(tmp_path, registry):
    """The exit code is INFORMATION, not authority — automation still needs to see a
    failure, even though no process is required to stop on it."""
    import tools.release_gate as _rg  # noqa: F401
    assert "exit 1" in rg.__doc__ and "exit 2" in rg.__doc__
