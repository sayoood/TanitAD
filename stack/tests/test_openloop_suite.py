"""``taniteval/tools/openloop_suite.py`` — the layer above the arm tool, end to end.

The fixture is ``test_refcv3_arm.py``'s (a random-init ``RefCV3Model`` over a
synthetic 3-episode v2 slice, CPU only), so this suite exercises the REAL path —
rollout → dump → ``analyze_refcv3`` → suite artifact → criteria check → MD + HTML —
without touching Thor, a pod or a real checkpoint.

WHAT IS PINNED, and the failure each line exists against
--------------------------------------------------------
  (1) ⭐ THE CONSTANT-ONLY CONTROL READS ITS KNOWN VALUE, and the exact one is
      EXACT. ``const0`` paired against itself must be ``0.0 [0.0, 0.0]``, not
      separated — no tolerance. A probe that tunes on the data it scores, or a
      metric with a wrong normalisation, produces a confident number for anything;
      the ONLY thing that catches it is a control whose answer is known first.
  (2) ⛔ THE REPORT IS OPEN LOOP AND SAYS SO. The forbidden regime phrase (PI
      ruling 2026-09-02) may not appear in either rendering, and the guard is
      absolute — it refused this tool's own first report, at the very table that
      REPORTS the stale labels, which is why those entries describe the text
      instead of quoting it.
  (3) THE CRITERIA CHECKER PASSES: zero required criteria ABSENT. Three separate
      defects were found by running it rather than by reading it — an artifact
      KIND in the tier slot, an audit record that re-introduced the token it
      audits, and a disavowal the checker's wording list does not recognise.
  (4) THE FOUR FAMILIES ARE ALL PRESENT AND NEVER POOLED. No composite over
      families exists anywhere in the artifact, and STRATEGIC — which has no
      paired block against the floor — is still in the headline table with its
      reason, because a family that DISAPPEARS reads exactly like one with no gap.
  (5) A REFUSED FAMILY CARRIES A REASON AND AN ``n``. Silence is the failure.
  (6) EVERY ARM CARRIES ITS TIER AND ITS EVIDENCE CLASS.
  (7) THE FLOOR VERDICTS HAVE THE RIGHT POLARITY: a random-init model must LOSE to
      constant velocity, and an interval excluding zero in the FLOOR's favour must
      render LOST — never TIED.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("torchvision.io")

_REPO = Path(__file__).resolve().parents[2]
SUITE = _REPO / "taniteval" / "tools" / "openloop_suite.py"
ARM_TEST = Path(__file__).resolve().parent / "test_refcv3_arm.py"


def _by_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


os_mod = _by_path("openloop_suite_under_test", SUITE)
arm_t = _by_path("refcv3_arm_fixture_for_suite", ARM_TEST)


@pytest.fixture(scope="module")
def suite(tmp_path_factory):
    root = tmp_path_factory.mktemp("openloop_suite_e2e")
    eps, lp, ck = arm_t._fixture(root / "fx")
    out = root / "out"
    rc = os_mod.main([
        "--ckpt", str(ck), "--episodes", str(eps), "--labels", str(lp),
        "--nav-source", "v72", "--grid", "2s", "--device", "cpu",
        "--action-units", "steer", "--with-oracle-sel",
        "--window-stride", "1", "--lru", "4",
        "--dump-dir", str(root / "dump"),
        "--n-boot", "120", "--seed", "0",
        "--out-dir", str(out), "--tag", "TEST",
        "--registry", str(_REPO / "products" / "P7-TanitEval"
                          / "CRITERIA_REGISTRY.json"),
        "--corpus", "SYNTHETIC fixture",
        "--parity-key", "NON-PARITY — synthetic fixture",
        "--parity-status", "NON-PARITY — synthetic fixture, no v2_parity block",
        "--expect-step", "11",
        "--train-eval-disjoint", "N/A — synthetic fixture",
        "--tiers", "os=T1,os_navshuf=T1,os_navzero=T1,oracle_sel=T0",
        "--strict",
    ])
    art = json.loads((out / "TEST.json").read_text(encoding="utf-8"))
    md = (out / "TEST.md").read_text(encoding="utf-8")
    html = (out / "TEST.html").read_text(encoding="utf-8")
    return rc, art, md, html


# =========================================================================== #
# (1) the constant-only control reads its KNOWN value                          #
# =========================================================================== #
def test_constant_only_control_reads_its_known_value_exactly(suite):
    _, art, _, _ = suite
    c0 = art["controls"]["const0"]
    assert c0["status"] == "OK", c0
    ec = c0["exact_check"]
    # ⛔ NO TOLERANCE HERE. The paired bootstrap of an arm against ITSELF has one
    # correct answer and it is exactly zero, with a zero-width interval that does
    # not separate. Anything else means the estimator is not doing what it says.
    assert ec["measured"]["delta"] == 0.0
    assert ec["measured"]["lo"] == 0.0
    assert ec["measured"]["hi"] == 0.0
    assert ec["measured"]["separated"] is False
    assert ec["pass"] is True
    assert ec["tolerance"].startswith("NONE")
    # the two analytic cross-checks, against values derived in float64 numpy
    # independently of the production float32 geometry path
    for k, v in c0["analytic_checks"].items():
        assert v["pass"] is True, (k, v)
        assert abs(v["measured"] - v["expected"]) <= v["rel_tol"] * max(
            1.0, abs(v["expected"])), (k, v)


def test_a_failing_control_is_a_harness_verdict_not_a_model_one(suite):
    """The headline must say the HARNESS is wrong, not the model, and it must be
    the FIRST thing said — ahead of any family row."""
    _, art, md, _ = suite
    hl = art["headline"]
    assert hl["harness_controls_pass"] is True
    assert "THE HARNESS IS WRONG, NOT THE MODEL" in json.dumps(art)
    assert md.index("## 1. Headline") < md.index("## 3. The four families")


# =========================================================================== #
# (2) the regime is OPEN LOOP and the guard is absolute                        #
# =========================================================================== #
def test_neither_rendering_uses_the_superseded_regime_phrase(suite):
    _, art, md, html = suite
    for what, text in (("markdown", md), ("html", html)):
        bad = os_mod._guard_loop_vocabulary(text, what)
        assert not bad, bad
    assert art["loop_class"]["value"] == "OPEN LOOP"
    assert art["protocol"]["loop_class"] == "OPEN LOOP"
    for name, row in art["arms"].items():
        if not name.startswith("_"):
            assert row["loop_class"] == "OPEN LOOP", name


def test_the_guard_actually_fires_on_the_phrase():
    """⛔ A GUARD NOBODY HAS SEEN FIRE IS NOT A GUARD. This one refused the tool's
    own first report; the deliberate-regression case is pinned so a later edit
    cannot quietly neuter it."""
    hits = os_mod._guard_loop_vocabulary(
        "line one\nthe arm is scored closed loop here\nline three", "unit")
    assert len(hits) == 1 and hits[0]["line"] == 2


def test_stale_loop_labels_are_reported_not_edited(suite):
    _, art, _, _ = suite
    labels = art["loop_class"]["stale_loop_labels"]
    assert len(labels) >= 3
    for s in labels:
        assert "REPORTED, not edited" in s["action"]
        # and the entry itself must not carry the phrase — that is what let the
        # guard stay absolute instead of growing an exemption
        assert not os_mod._guard_loop_vocabulary(json.dumps(s), "label")


# =========================================================================== #
# (3) the criteria checker passes                                              #
# =========================================================================== #
def test_criteria_checker_reports_zero_violations(suite):
    rc, art, _, _ = suite
    crit = art["criteria"]
    assert crit["scope"] == "IN_SCOPE", crit
    assert crit["tier"] == "T1", crit          # not the artifact KIND
    assert crit["n_violations"] == 0, crit["violations"]
    assert rc == 0                              # --strict was passed


def test_forbidden_estimator_is_never_a_decision_grade_interval(suite):
    _, art, _, _ = suite
    hyg = {r["id"]: r for r in art["criteria"]["hygiene"]}
    assert hyg["hyg.no_forbidden_estimator"]["state"] == "PRESENT"
    ann = art["four_families"]["_disavowal_annotations"]
    # the audit trail must not re-introduce the token it audits
    for site in ann["sites"]:
        assert "overlapping_holdout" not in json.dumps(site)
    assert art["estimator"]["interval"].startswith("episode_cluster_bootstrap")
    assert art["estimator"]["paired"].startswith("paired_episode_cluster_bootstrap")


# =========================================================================== #
# (4) four families, separate, never pooled                                    #
# =========================================================================== #
def test_all_four_families_present_and_never_pooled(suite):
    _, art, md, _ = suite
    fam = art["four_families"]
    for k in ("longitudinal", "lateral", "tactical", "strategic"):
        assert k in fam, k
    crit_f = art["criteria"]["families"]
    for name in ("LONGITUDINAL", "LATERAL", "TACTICAL", "STRATEGIC"):
        rows = crit_f[name]
        assert rows, name
        assert all(r["state"] in ("PRESENT", "REFUSED", "PARTIAL") for r in rows), \
            (name, rows)
    # no composite anywhere
    blob = json.dumps(art).lower()
    for banned in ("composite_score", "overall_score", "pooled_score"):
        assert banned not in blob, banned
    assert "never pooled" in md.lower()


def test_strategic_is_in_the_headline_even_with_no_paired_block(suite):
    """⛔ A FAMILY THAT DISAPPEARS FROM THE TABLE READS LIKE A FAMILY WITH NO GAP.
    STRATEGIC has no contrast against `ha0` — the floor has no route head — so it
    carries its OWN contrast and says so, rather than being dropped."""
    _, art, md, _ = suite
    st = art["headline"]["families"]["strategic"]
    assert "no paired block against `ha0`" in st["_never_pooled"]
    assert "majority" in st["contrast"].lower()
    assert "**STRATEGIC**" in md


def test_ade_and_the_longitudinal_lateral_split_are_both_reported(suite):
    _, art, md, _ = suite
    fams = art["headline"]["families"]
    assert "ade_m" in fams["ADE"]["metrics"]
    assert "LON_speed_mae_mps" in fams["longitudinal"]["metrics"]
    assert "LAT_cross_mae_m" in fams["lateral"]["metrics"]
    assert "LAT_yaw_rate_mae_radps" in fams["lateral"]["metrics"]
    # and every absolute reading carries its interval, not a bare point estimate
    rows = os_mod._abs_rows(art["four_families"]["lateral"])
    scored = [r for r in rows if not str(r[0]).startswith(("n_steps", "min_ds",
                                                           "excluded"))]
    assert scored
    for name, _v, ci in scored:
        assert ci.startswith("[") or ci == "REFUSED", (name, ci)


# =========================================================================== #
# (5) a refused family carries a reason and an n                               #
# =========================================================================== #
def test_refused_criteria_carry_a_reason_and_are_named_as_gaps(suite):
    _, art, md, _ = suite
    dk = art["four_families"]["longitudinal"]["distance_keeping"]
    assert dk["status"] in ("UNAVAILABLE", "REFUSED")
    assert dk.get("reason")
    assert "n" in dk
    ids = [g["criterion"] for g in art["gaps"]["items"]]
    assert "long.distance_keeping" in ids, ids
    for g in art["gaps"]["items"]:
        assert g["reason"], g
    assert "## 7. Honest gaps" in md


# =========================================================================== #
# (6) every arm carries its tier and its evidence class                        #
# =========================================================================== #
def test_every_arm_carries_tier_and_evidence_class(suite):
    _, art, _, _ = suite
    arms = {k: v for k, v in art["arms"].items() if not k.startswith("_")}
    assert {"os", "os_navshuf", "os_navzero", "ha", "ha0", "oracle_sel",
            "const0"} <= set(arms)
    for name, row in arms.items():
        assert row["tier"] in ("T0", "T1"), (name, row)
        assert row["evidence_class"], name
    assert arms["oracle_sel"]["tier"] == "T0"
    assert arms["ha0"]["tier"] == "T1"
    assert art["arms"]["_tier_ruling"], "the OPEN tier ruling must travel"


def test_the_two_nav_controls_are_both_present_and_distinguished(suite):
    _, art, _, _ = suite
    nc = art["controls"]["nav_controls"]
    assert "PAIRING" in nc["os_navshuf"]
    assert "SIGNAL" in nc["os_navzero"]
    assert "cannot stand in" in nc["not_interchangeable"]
    assert art["headline"]["deployment_families"], \
        "the deployment margin (nav withheld) must be reported beside the oracle one"


# =========================================================================== #
# (7) polarity: a random-init model must LOSE to constant velocity             #
# =========================================================================== #
def test_random_init_loses_to_the_constant_velocity_floor(suite):
    _, art, _, _ = suite
    ade = art["headline"]["families"]["ADE"]["metrics"]["ade_m"]
    assert ade["delta"] > 0 and ade["separated"] is True
    assert ade["verdict"] == "LOST", ade


def test_a_floor_win_renders_LOST_never_TIED():
    """⛔ THE REGISTRY'S LOUDEST RULE. A paired interval that excludes zero while
    favouring the FLOOR means the trivial baseline won — rendering that as a tie is
    how a 22x loss once looked like a result."""
    lost = {"delta": 7.9, "lo": 6.1, "hi": 9.7, "separated": True}
    won = {"delta": -0.12, "lo": -0.18, "hi": -0.04, "separated": True}
    tied = {"delta": -0.05, "lo": -0.43, "hi": 0.43, "separated": False}
    assert os_mod.verdict_of("ade_m", lost)[0] == "LOST"
    assert os_mod.verdict_of("ade_m", won)[0] == "WON"
    assert os_mod.verdict_of("ade_m", tied)[0] == "TIED"
    # higher-is-better metrics invert, and a degenerate interval is UNREADABLE
    assert os_mod.verdict_of("TAC_traj_lat_correct",
                             {"delta": 0.2, "lo": .1, "hi": .3,
                              "separated": True})[0] == "WON"
    assert os_mod.verdict_of("ade_m", {"delta": 0.0, "lo": 0.0, "hi": 0.0,
                                       "separated": True,
                                       "degenerate": True})[0] == "UNREADABLE"


def test_family_verdict_names_every_non_empty_bucket():
    """A tally that hides a bucket is the pooling failure in miniature."""
    v = os_mod._family_verdict({"WON": 1, "LOST": 1, "TIED": 1, "UNREADABLE": 0})
    assert "1/3 WON" in v and "1/3 LOST" in v and "1/3 TIED" in v


# =========================================================================== #
# provenance                                                                   #
# =========================================================================== #
def test_provenance_records_which_tree_produced_the_numbers(suite):
    """Presence proves transfer, md5 proves bytes, a successful import proves
    loading — none of them proves CURRENCY. The resolved paths are what let a
    reader tell, after the fact, which code ran."""
    _, art, _, _ = suite
    tr = art["provenance"]["resolved_trees"]
    for k in ("tanitad", "taniteval", "torch", "numpy"):
        assert tr.get(k), k
    assert art["provenance"]["source"]
    assert art["parity"]["parity_key"]
    assert art["parity"]["train_eval_disjoint"]


def test_arm_json_roundtrip_is_the_contract_with_another_stream(suite, tmp_path):
    """A record banked by ANOTHER stream goes through `--arm-json` and produces the
    same compliant report — that IS the interchange contract, so it is exercised
    rather than asserted about the source."""
    _, art, _, _ = suite
    dump = Path(art["provenance"]["dump_dir"])
    arm = _by_path("refcv3_arm_for_roundtrip",
                   _REPO / "taniteval" / "tools" / "refcv3_arm.py")
    rec = arm.analyze_refcv3(str(dump), n_boot=60, seed=0)
    banked = tmp_path / "armrecord.json"
    banked.write_text(json.dumps(rec, indent=1, ensure_ascii=False, default=str),
                      encoding="utf-8")

    out = tmp_path / "out"
    rc = os_mod.main([
        "--arm-json", str(banked), "--dump-dir", str(dump),
        "--out-dir", str(out), "--tag", "RT", "--n-boot", "60",
        "--registry", str(_REPO / "products" / "P7-TanitEval"
                          / "CRITERIA_REGISTRY.json"),
        "--corpus", "SYNTHETIC fixture",
        "--parity-status", "NON-PARITY — synthetic fixture",
        "--strict"])
    assert rc == 0
    rt = json.loads((out / "RT.json").read_text(encoding="utf-8"))
    assert rt["criteria"]["n_violations"] == 0
    assert rt["controls"]["const0"]["status"] == "OK"
    assert rt["provenance"]["source"].startswith("BANKED RECORD")
    assert rt["provenance"]["sha256"]


def test_arm_json_without_a_dump_refuses_the_control_rather_than_dropping_it(
        tmp_path):
    """⛔ Without the per-window arrays the constant-only control cannot be built.
    The one check whose answer is known in advance must be REFUSED WITH A REASON,
    never silently omitted — a suite missing it looks exactly like one that passed
    it."""
    miss = os_mod.constant_only_control(str(tmp_path), 0.5, 10, 0)
    assert miss["status"] == "REFUSED"
    assert "constant-only control" in miss["reason"]
    assert "WORK ITEM" in miss["estimator"]


def test_checkpoint_is_verified_by_content_not_by_filename(suite):
    """⛔ `ckpt.pt` IS THE ROLLING FILE AND ITS NAME NEVER CHANGES.

    `refc_v3_train.py:103` sets `MILESTONES = (5000, 15000, 20000, 30000)`, so no
    `ckpt_40284_FINAL.pt` is ever written; the final checkpoint is plain `ckpt.pt`
    (`:1191`), which is also overwritten every `--save-every` steps. A report of
    "the final read" computed from step 38,450 is not wrong in any way a reader
    could detect — so the step is checked against the CHECKPOINT, never the name.
    """
    _, art, _, _ = suite
    # the positive case ran inside the fixture (--expect-step 11 on a step-11 ckpt)
    assert art["provenance"]["source"]
    # ⭐ and the NEGATIVE control: a guard nobody has seen refuse is not a guard
    import argparse as _ap
    with pytest.raises(SystemExit) as ex:
        os_mod._verify_ckpt_step({"model": {"step": 38450}},
                                 _ap.Namespace(expect_step=40284))
    assert "CHECKPOINT STEP MISMATCH" in str(ex.value)
    assert "38450" in str(ex.value) and "40284" in str(ex.value)
    # absent --expect-step must NOT raise; it reports and moves on
    os_mod._verify_ckpt_step({"model": {"step": 7}},
                             _ap.Namespace(expect_step=None))


def test_non_parity_is_stated_not_left_to_be_assumed(suite):
    """The run's own config.json carries `v2_parity.parity false` / `checked false`,
    so refcv3 is NOT cross-arm comparable with refc-base or refc-xl. The default is
    NON-PARITY precisely because ASSUMING parity is the failure mode."""
    _, art, md, html = suite
    pa = art["parity"]
    assert pa["parity_status"].upper().startswith("NON")
    warn = pa["⛔_cross_arm_comparability"]
    assert "NOT" in warn and "cross-arm comparable" in warn
    for text in (md, html):
        assert "NON-PARITY" in text
    # and the escape hatch is the margin over the shared floor, not a level
    assert "ha0" in warn


def test_the_deployment_margin_has_equal_billing_with_the_oracle_one(suite):
    """⛔ The nav token is an ORACLE (provenance ego-future) that will not exist at
    deployment. Quoting only `os − ha0` overstates the system, so `os_navzero − ha0`
    is printed beside it everywhere — headline, per family, and in its own section."""
    _, art, md, html = suite
    assert art["vs_floor_paired_deployment"], "the deployment margin must be computed"
    dep = art["headline"]["deployment_families"]
    assert {"ADE", "longitudinal", "lateral"} <= set(dep)
    for fam in ("ADE", "longitudinal", "lateral"):
        assert dep[fam]["metrics"], fam
    for text in (md, html):
        assert "equal billing" in text.lower()
        assert "os_navzero" in text
        assert "overstates the system" in text.lower()


def test_html_is_a_complete_standalone_document(suite):
    _, _, _, html = suite
    assert html.lstrip().lower().startswith("<!doctype html>")
    assert '<meta charset="utf-8">' in html
    assert "<title>" in html and "</body></html>" in html
    # self-contained: no external asset of any kind
    for banned in ("<script", "src=\"http", "href=\"http", "@import"):
        assert banned not in html, banned
