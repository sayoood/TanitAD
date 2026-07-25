"""``run_gate`` — the HORIZON CORRECTION (Tier-1 #1, 2026-07-26).

``corridor_departure_rate`` at a pre-registered horizon K becomes a **gate
co-primary**; ``ade_0_2s`` is **demoted to a proposal-quality diagnostic**.

WHY THESE TESTS ARE SHAPED THIS WAY
-----------------------------------
The change is only worth anything if the gate now *decides differently* on the
case that motivated it. So the load-bearing test is not "the field exists" — it
is :func:`test_the_inversion_the_change_exists_for`, which feeds the gate two
numbers MEASURED on the **same arm** (``refc-diffusion-base-v21-30k`` @ 29999)
and asserts the verdict flips:

  * ``ade_0_2s`` 0.4728 [0.3835, 0.5699] — open-loop 4wp,
    ``driving_refc-base-30k.json`` — **PASSES** a 0.60 bar
  * ``corridor_departure_rate`` 0.5877 [0.5107, 0.6622] @ K=185 — closed loop,
    ``e1a_horizon_heldout44_K185.json`` — **FAILS** a 0.35 bar

Old gate: CONTINUE. Corrected gate: RESTART. That is the whole point.

The other families pin the properties that make the co-primary trustworthy:
K is explicit and refused when blind or impossible; the junction stratum is
always reported; the deprecated estimator can never adjudicate; a missing
corridor artifact is INCOMPLETE and never a pass; and — the back-compat pin —
a card written before this change still renders its historical verdict, so
``v1_g1_dryrun_gate_FIXED.json``'s 8-KILL / 5-report-only / CONTINUE split
survives untouched.

Pure-Python: constructed dicts + the two committed JSON artifacts. No torch, no
window load, no GPU.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_gate as rg                                              # noqa: E402

_REPO = Path(__file__).resolve().parents[2]
_E1A = (_REPO / "TanitAD Research Hub" / "Architecture & Inference"
        / "Implementation" / "incoming" / "2026-07-25-closedloop-horizon-and-shift"
        / "e1a_horizon_heldout44_K185.json")
_REFC_DRIVING = _REPO / "taniteval" / "results" / "driving_refc-base-30k.json"

# MEASURED, from the two artifacts above. Named here so a drift in either goes
# red rather than quietly changing what these tests mean.
E1A_CDR_K185_OVERALL = 0.5877
E1A_CDR_K185_JUNCTION = 0.8414
E1A_CDR_K20_OVERALL = 0.0035
REFC_ADE_2S = 0.4728


# =========================================================================== #
# helpers                                                                     #
# =========================================================================== #
def _cb(mean, lo, hi, n_w=43, n_e=43):
    return {"mean": mean, "lo": lo, "hi": hi, "ci95": round((hi - lo) / 2, 4),
            "reducer": "mean", "n_windows": n_w, "n_episodes": n_e,
            "n_boot": 2000, "estimator": "episode_cluster_bootstrap"}


def _stratum(mean, K=185, n_w=43, n_e=43, lo=None, hi=None, primary_m=1.75):
    lo = mean * 0.85 if lo is None else lo
    hi = min(1.0, mean * 1.15) if hi is None else hi
    return {"block": "taniteval.corridor", "version": "1.0.0",
            "surface": "closed_loop", "horizon_K": K, "horizon_s": round(K * 0.1, 2),
            "n_windows": n_w, "n_episodes": n_e,
            "corridor_primary_m": primary_m,
            "corridor_thresholds_m": [1.0, 1.75, 2.5],
            "corridor_departure_rate": _cb(mean, lo, hi, n_w, n_e),
            "corridor_departure_rate_by_threshold_m": {
                "1": _cb(min(1.0, mean * 1.2), lo, hi, n_w, n_e),
                "1.75": _cb(mean, lo, hi, n_w, n_e),
                "2.5": _cb(mean * 0.8, lo, hi, n_w, n_e)},
            "window_departure_rate": _cb(min(1.0, mean * 1.4), lo, hi, n_w, n_e),
            "peak_xte_m": _cb(38.9445, 26.0, 52.0, n_w, n_e),
            "mean_xte_m": _cb(2.1, 1.5, 2.9, n_w, n_e)}


def _corridor_doc(overall=E1A_CDR_K185_OVERALL, junction=E1A_CDR_K185_JUNCTION,
                  K=185, primary_m=1.75):
    return {"overall": _stratum(overall, K, 43, 43, primary_m=primary_m),
            "junction": _stratum(junction, K, 6, 6, primary_m=primary_m),
            "n_by_stratum": {"overall": 43, "junction": 6}}


def _card(tmp_path, **over):
    """A horizon-honest card, written directly (bypassing `register`) so a test
    can express a shape `register` would refuse."""
    card = {"run": "t", "gate_step": 10000, "primary_metric": "ade_0_2s",
            "primary_threshold": 0.60, "primary_direction": "<=",
            "primary_source": "held-out", "secondary": [],
            "reference_run": None, "reference_log": None, "compare_metric": None,
            "tau": 1.5, "lever_family": "fam", "restarts_used": 0,
            "restart_cap": 2, "registered_utc": "2026-07-26T00:00:00+00:00",
            "note": "",
            "co_primary_metric": "corridor_departure_rate",
            "co_primary_threshold": 0.35, "co_primary_direction": "<=",
            "co_primary_horizon_K": 185, "co_primary_corridor_m": 1.75,
            "co_primary_stratum": "overall",
            "co_primary_junction_threshold": None, "co_primary_source": "",
            "primary_role": "diagnostic", "no_co_primary_reason": ""}
    card.update(over)
    p = tmp_path / "card.json"
    p.write_text(json.dumps(card))
    return p


def _run(tmp_path, card_path, ade=REFC_ADE_2S, corridor=None, secondary_values=(),
         corridor_path=None):
    (tmp_path / "eval.json").write_text(json.dumps(
        {"cluster_bootstrap": {"model": {"ade_0_2s": _cb(ade, ade * 0.8, ade * 1.2)}}}))
    (tmp_path / "log.jsonl").write_text(
        '{"step": 0, "step_s": 0}\n{"step": 10000, "step_s": 100}\n')
    argv = ["check", "--card", str(card_path), "--log", str(tmp_path / "log.jsonl"),
            "--eval-json", str(tmp_path / "eval.json"),
            "--json", str(tmp_path / "gate.json")]
    if corridor is not None:
        cp = corridor_path or (tmp_path / "corridor.json")
        Path(cp).write_text(json.dumps(corridor))
        argv += ["--corridor-json", str(cp)]
    if secondary_values:
        argv += ["--secondary-value", *secondary_values]
    rg.main(argv)
    return json.loads((tmp_path / "gate.json").read_text())


# =========================================================================== #
# 1. THE INVERSION — the reason the change exists                             #
# =========================================================================== #
def test_the_inversion_the_change_exists_for(tmp_path):
    """Same arm, same day: ADE@2s passes, the corridor co-primary fails.

    Under the pre-2026-07-26 gate this arm CONTINUEs on 0.4728 <= 0.60. Under
    the corrected gate it RESTARTs on 0.5877 > 0.35 at K=185 — a failure the
    2 s instrument cannot see (0.0035 on the SAME windows at K=20)."""
    g = _run(tmp_path, _card(tmp_path), ade=REFC_ADE_2S,
             corridor=_corridor_doc())
    assert g["primary"]["pass"] is True, "ade_0_2s passes its bar"
    assert g["primary"]["role"] == "diagnostic"
    assert g["co_primary"]["pass"] is False
    assert g["co_primary"]["value"] == E1A_CDR_K185_OVERALL
    assert g["verdict"] == "RESTART", (g["verdict"], g["reason"])
    assert g["verdict_adjudicated_by"][0] == "co_primary.corridor_departure_rate"


def test_the_same_windows_at_the_blind_horizon_would_have_passed(tmp_path):
    """The counterfactual, in two halves.

    (a) On E1a's OWN windows the corridor rate at K=20 is 0.0035 — it clears the
    same 0.35 bar the K=185 value (0.5877) fails by a factor of 1.7. A gate that
    read the metric at the short horizon would have said CONTINUE.
    (b) So a K=20 co-primary is refused not only by ``register`` but by
    ``check`` too — a hand-forged blind card cannot be smuggled past the gate."""
    blind = _corridor_doc(overall=E1A_CDR_K20_OVERALL, junction=0.025, K=20)
    read = rg.read_corridor(blind, expect_K=20)
    assert read["value"] == E1A_CDR_K20_OVERALL
    assert read["value"] <= 0.35 < E1A_CDR_K185_OVERALL, "the blind spot, exactly"

    with pytest.raises(SystemExit, match="below ade_0_2s' own horizon"):
        _run(tmp_path, _card(tmp_path, co_primary_horizon_K=20), corridor=blind)


def test_a_demoted_primary_failure_cannot_kill_but_is_never_hidden(tmp_path):
    """ade_0_2s FAILS while the co-primary passes -> CONTINUE, with a qualifier.

    "Demoted" has to mean something: a diagnostic that still kills is not a
    diagnostic. The failure is recorded on the verdict, not swallowed."""
    g = _run(tmp_path, _card(tmp_path, co_primary_threshold=0.70), ade=0.95,
             corridor=_corridor_doc())
    assert g["primary"]["pass"] is False and g["co_primary"]["pass"] is True
    assert g["verdict"] == "CONTINUE"
    assert "did not kill the run" in g["qualifier"]
    assert "0.95" in g["qualifier"]


# =========================================================================== #
# 2. The horizon is explicit, recorded, and bounded                           #
# =========================================================================== #
def test_every_verdict_names_its_horizon_and_n(tmp_path):
    """The standing rule, mirroring "never quote an exponent bare"."""
    g = _run(tmp_path, _card(tmp_path), corridor=_corridor_doc())
    h = g["horizon"]
    assert h["registered_horizon_K"] == 185 and h["registered_horizon_s"] == 18.5
    assert h["measured_horizon_K"] == 185
    assert h["n_windows"] == 43 and h["n_episodes"] == 43
    assert h["n_junction_windows"] == 6
    assert h["surface"] == "closed_loop"
    for token in ("K=185", "18.5 s", "n=43", "43 episodes", "closed_loop"):
        assert token in h["rendered"], (token, h["rendered"])
    assert g["horizon_honest"] is True


@pytest.mark.parametrize("K", [1, 4, 19, 20])
def test_register_refuses_a_horizon_at_or_below_the_blind_one(K):
    with pytest.raises(SystemExit) as ei:
        rg.validate_horizon_K(K)
    assert "0.0035" in str(ei.value) and "0.5877" in str(ei.value)


@pytest.mark.parametrize("K", [191, 200, 300])
def test_register_refuses_a_structurally_impossible_horizon(K):
    with pytest.raises(SystemExit, match="structural ceiling"):
        rg.validate_horizon_K(K)


def test_the_ceiling_is_the_corpus_ceiling_not_a_round_number():
    """190-199-frame clips, ``T - W - K >= 1`` -> K <= 190 (19.0 s); K=200 is
    structurally impossible. Same constant as taniteval.corridor."""
    assert rg.HORIZON_CEILING_K == 190
    assert rg.horizon_seconds(190) == 19.0
    assert rg.validate_horizon_K(185)["frac_of_ceiling"] == round(185 / 190, 4)
    assert rg.validate_horizon_K(185)["horizon_honest"] is True
    assert rg.validate_horizon_K(50)["horizon_honest"] is False


def test_a_block_at_the_wrong_horizon_is_refused_not_silently_used(tmp_path):
    """Rendering at a horizon other than the pre-registered one is a garden of
    forking paths with extra steps."""
    with pytest.raises(SystemExit) as ei:
        _run(tmp_path, _card(tmp_path, co_primary_horizon_K=185),
             corridor=_corridor_doc(K=120))
    assert "K=120" in str(ei.value) and "185" in str(ei.value)


def test_a_block_without_horizon_K_is_refused(tmp_path):
    doc = _corridor_doc()
    doc["overall"].pop("horizon_K")
    with pytest.raises(SystemExit, match="does not record"):
        _run(tmp_path, _card(tmp_path), corridor=doc)


def test_a_different_corridor_halfwidth_is_refused(tmp_path):
    with pytest.raises(SystemExit, match="knife-edge|half-width"):
        _run(tmp_path, _card(tmp_path), corridor=_corridor_doc(primary_m=2.5))


# =========================================================================== #
# 3. The junction stratum — reported separately, ALWAYS                       #
# =========================================================================== #
def test_junction_is_reported_even_when_it_does_not_adjudicate(tmp_path):
    """0.8414 vs 0.5877: the failure concentrates there, so it is never folded
    into the overall number."""
    g = _run(tmp_path, _card(tmp_path, co_primary_junction_threshold=None),
             corridor=_corridor_doc())
    j = g["co_primary"]["junction"]
    assert j["measured"] is True and j["value"] == E1A_CDR_K185_JUNCTION
    assert j["adjudicated"] is False and j["pass"] is None
    assert j["n_windows"] == 6 and j["n_episodes"] == 6
    assert "kinematic" not in j["definition"].lower() or "topology" in j["definition"]
    assert "10 deg" in j["definition"]


def test_junction_can_adjudicate_and_a_junction_failure_kills(tmp_path):
    """Overall PASSES, junction FAILS -> the gate fails. Without the stratum the
    arm would have walked through on a diluted average."""
    g = _run(tmp_path, _card(tmp_path, co_primary_threshold=0.70,
                             co_primary_junction_threshold=0.50),
             corridor=_corridor_doc())
    j = g["co_primary"]["junction"]
    assert j["adjudicated"] is True and j["pass"] is False
    assert g["co_primary"]["pass"] is False
    assert g["verdict"] == "RESTART"


def test_a_junction_stratum_too_small_to_bootstrap_is_not_a_pass(tmp_path):
    """``taniteval.corridor`` returns None below 2 windows / 2 episodes. A None
    is a NOT-MEASURED; treating it as a pass is how a stratum disappears."""
    doc = _corridor_doc()
    doc["junction"] = None
    g = _run(tmp_path, _card(tmp_path, co_primary_junction_threshold=0.50),
             corridor=doc)
    j = g["co_primary"]["junction"]
    assert j["measured"] is False and j["pass"] is None
    assert "NOT a pass" in j["note"]


# =========================================================================== #
# 4. The estimator — and the live `heldout` tripwire                          #
# =========================================================================== #
def test_the_deprecated_estimator_can_never_adjudicate_the_co_primary(tmp_path):
    doc = _corridor_doc()
    doc["overall"]["corridor_departure_rate"]["estimator"] = "overlapping_holdout_se"
    with pytest.raises(SystemExit) as ei:
        _run(tmp_path, _card(tmp_path), corridor=doc)
    msg = str(ei.value)
    assert "overlapping_holdout_se" in msg and "may not decide a gate" in msg


def test_an_unnamed_estimator_is_refused_not_assumed(tmp_path):
    doc = _corridor_doc()
    doc["overall"]["corridor_departure_rate"].pop("estimator")
    with pytest.raises(SystemExit, match="decision-grade set"):
        _run(tmp_path, _card(tmp_path), corridor=doc)


def test_a_bare_corridor_number_with_no_interval_is_refused(tmp_path):
    doc = _corridor_doc()
    doc["overall"]["corridor_departure_rate"] = 0.12
    with pytest.raises(SystemExit, match="no interval"):
        _run(tmp_path, _card(tmp_path), corridor=doc)


def test_the_heldout_tripwire_is_still_armed():
    """⚠️ LIVE TRIPWIRE. ``_deprecated_present`` keys on the literal name
    ``heldout``, which a sibling deliberately kept as a back-compat alias. If
    this key is ever renamed the gate silently stops refusing the biased
    estimator — so the refusal is pinned here as well as in
    ``test_run_gate_eval_metric.py``."""
    ev = {"full_set": {"model": {"ade_0_2s": 0.4271}},
          "heldout": {"model": {"ade_0_2s": {
              "mean": 0.4522, "ci95": 0.0312,
              "estimator": "overlapping_holdout_se"}}}}
    assert rg._deprecated_present(ev, ("ade_0_2s",)) is True
    with pytest.raises(SystemExit, match="overlapping_holdout_se"):
        rg._read_eval_metric(ev, "ade_0_2s")


def test_only_the_paired_bootstrap_is_admissible_for_an_arm_vs_arm_delta():
    ok = {"delta": 0.5842, "lo": 0.5071, "hi": 0.6565, "separated": True,
          "p_delta_gt0": 1.0, "estimator": "paired_episode_cluster_bootstrap"}
    assert rg.read_corridor_paired(ok)["overall"]["delta"] == 0.5842
    for bad in ("episode_cluster_bootstrap", "overlapping_holdout_se", None):
        with pytest.raises(SystemExit, match="paired_episode_cluster_bootstrap"):
            rg.read_corridor_paired(dict(ok, estimator=bad))


# =========================================================================== #
# 5. Not-measured is INCOMPLETE, never a pass and never ADE-alone             #
# =========================================================================== #
def test_a_registered_co_primary_with_no_artifact_is_INCOMPLETE(tmp_path):
    """The honest state of every arm on disk today. ade_0_2s passing does NOT
    buy a CONTINUE."""
    g = _run(tmp_path, _card(tmp_path), ade=0.20, corridor=None)
    assert g["primary"]["pass"] is True
    assert g["co_primary"]["measured"] is False and g["co_primary"]["pass"] is None
    assert g["verdict"] == "INCOMPLETE"
    assert "CO-PRIMARY" in g["reason"] and "K=185" in g["reason"]
    assert g["horizon_honest"] is False


def test_a_skipped_corridor_artifact_is_INCOMPLETE_not_a_crash(tmp_path):
    """``corridor.from_windows`` on a dump with no dense path emits a
    self-describing ``skipped`` node — that is exactly the state of all 30
    committed ``windows_*.pt``. It must become INCOMPLETE, not an exception and
    never a pass."""
    g = _run(tmp_path, _card(tmp_path),
             corridor={"block": "taniteval.corridor", "version": "1.0.0",
                       "dense_surface_available": False,
                       "skipped": "no dense path in this dump "
                                  "(pred_dense/gt_dense)."})
    assert g["co_primary"]["measured"] is False
    assert "SKIPPED" in g["co_primary"]["note"]
    assert g["verdict"] == "INCOMPLETE"


def test_a_missing_corridor_file_is_INCOMPLETE_not_a_crash(tmp_path):
    card = _card(tmp_path)
    (tmp_path / "eval.json").write_text(json.dumps(
        {"cluster_bootstrap": {"model": {"ade_0_2s": _cb(0.2, 0.1, 0.3)}}}))
    (tmp_path / "log.jsonl").write_text('{"step": 10000, "step_s": 1}\n')
    rg.main(["check", "--card", str(card), "--log", str(tmp_path / "log.jsonl"),
             "--eval-json", str(tmp_path / "eval.json"),
             "--corridor-json", str(tmp_path / "nope.json"),
             "--json", str(tmp_path / "gate.json")])
    g = json.loads((tmp_path / "gate.json").read_text())
    assert g["verdict"] == "INCOMPLETE"
    assert "does not exist" in g["co_primary"]["note"]


# =========================================================================== #
# 6. `register` refuses to write a new horizon-blind card                     #
# =========================================================================== #
def _register(tmp_path, *extra):
    return rg.main(["register", "--run", "r", "--gate-step", "10000",
                    "--primary-threshold", "0.6",
                    "--card", str(tmp_path / "c.json"), "--force", *extra])


def test_register_refuses_a_card_with_no_co_primary(tmp_path):
    with pytest.raises(SystemExit) as ei:
        _register(tmp_path)
    assert "horizon-blind" in str(ei.value) and "--co-primary-horizon-K" in str(ei.value)


def test_register_writes_a_blind_card_only_with_a_written_reason(tmp_path):
    _register(tmp_path, "--no-co-primary", "no closed-loop rollout exists yet")
    card = json.loads((tmp_path / "c.json").read_text())
    assert card["co_primary_horizon_K"] is None
    assert card["no_co_primary_reason"] == "no closed-loop rollout exists yet"
    assert card["primary_role"] == "kill"          # nothing else can adjudicate


def test_register_demotes_the_primary_whenever_a_co_primary_exists(tmp_path):
    _register(tmp_path, "--co-primary-threshold", "0.35",
              "--co-primary-horizon-K", "185")
    card = json.loads((tmp_path / "c.json").read_text())
    assert card["primary_role"] == "diagnostic"
    assert card["co_primary_metric"] == "corridor_departure_rate"
    assert card["co_primary_horizon_K"] == 185
    assert card["co_primary_corridor_m"] == 1.75
    assert rg.GateCard(**card).has_co_primary is True


def test_register_refuses_mixing_no_co_primary_with_a_co_primary(tmp_path):
    with pytest.raises(SystemExit, match="mutually exclusive"):
        _register(tmp_path, "--no-co-primary", "x",
                  "--co-primary-horizon-K", "185")


# =========================================================================== #
# 7. BACK-COMPAT — the pre-2026-07-26 dry-run must still render               #
# =========================================================================== #
LEGACY_CARD = {
    "run": "flagship-v4-dryrun", "gate_step": 10000, "primary_metric": "ade_0_2s",
    "primary_threshold": 0.60, "primary_direction": "<=",
    "primary_source": "held-out full-set",
    "secondary": ["wm_canary_ade_2s<=0.55", "speed_benefit_recovered_frac>=0.70",
                  "oracle_in_fan<=0.30", "miss_at_2m<=0.10",
                  "seam_norm_ratio_max<=1.0", "encoder_touching_levers<=2",
                  "deploy_tick_p99_ms<=50", "nonav_route_beats_majority>=1"],
    "reference_run": None, "reference_log": None, "compare_metric": None,
    "tau": 1.5, "lever_family": "joint-planner-wm", "restarts_used": 0,
    "restart_cap": 2, "registered_utc": "2026-07-22T00:00:00+00:00", "note": ""}

# v1_g1_dryrun_gate_FIXED.json, fixes_verified.split_8_KILL_5_REPORT
DRYRUN_KILL = ["wm_canary_ade_2s=0.4522", "speed_benefit_recovered_frac=0.8169",
               "oracle_in_fan=0.29", "miss_2m=0.0602", "seam_norm_ratio_max=0.9",
               "encoder_touching_levers=2", "deploy_tick_p99_ms=11.5",
               "nonav_route_beats_majority=1"]
DRYRUN_REPORT = ["imag_win_at_5s=1", "strat_subspace_sufficiency=0.92",
                 "strat_subspace_compression=0.45",
                 "longh_5s_beats_persistence=1", "cruise_delta_vs_holdv0=-0.2122"]


def test_a_pre_horizon_correction_card_still_loads(tmp_path):
    """``GateCard(**json)`` on a card with none of the new fields."""
    card = rg.GateCard(**LEGACY_CARD)
    assert card.has_co_primary is False
    assert card.primary_role == "kill"


def test_the_committed_dryrun_split_survives_the_change(tmp_path):
    """PIN: ``kill_adjudicated: 8, report_only: 5, verdict_from_kill_only:
    CONTINUE`` (``taniteval/results/v1_g1_dryrun_gate_FIXED.json``). The gate
    could already render a verdict; the horizon correction must not break it."""
    p = tmp_path / "legacy.json"
    p.write_text(json.dumps(LEGACY_CARD))
    g = _run(tmp_path, p, ade=0.4271, corridor=None,
             secondary_values=DRYRUN_KILL + DRYRUN_REPORT)
    assert len(g["secondary"]) == 8
    assert all(r["pass"] is True for r in g["secondary"])
    assert len(g["report_only"]) == 5
    assert all(r["adjudicated"] is False for r in g["report_only"])
    assert g["verdict"] == "CONTINUE"


def test_a_legacy_card_is_stamped_horizon_blind_with_the_reason(tmp_path):
    """Back-compat is bounded: the verdict value is unchanged, but it can no
    longer be quoted as if it were horizon-honest."""
    p = tmp_path / "legacy.json"
    p.write_text(json.dumps(LEGACY_CARD))
    g = _run(tmp_path, p, ade=0.4271, corridor=None,
             secondary_values=DRYRUN_KILL)
    assert g["horizon_honest"] is False
    assert g["co_primary"]["registered"] is False
    w = g["horizon"]["warning"]
    assert "0.0035" in w and "0.5877" in w and "K=20" in w
    assert g["primary"]["role"] == "kill"          # nothing else can adjudicate
    assert g["horizon"]["horizon_K"] == rg.ADE2S_K == 20


def test_a_legacy_card_still_restarts_on_a_failing_secondary(tmp_path):
    p = tmp_path / "legacy.json"
    p.write_text(json.dumps(LEGACY_CARD))
    g = _run(tmp_path, p, ade=0.4271, corridor=None,
             secondary_values=[s.replace("oracle_in_fan=0.29", "oracle_in_fan=0.99")
                               for s in DRYRUN_KILL])
    assert g["verdict"] == "RESTART"


# =========================================================================== #
# 8. A diagnostic may not abort a verdict (fixed while re-gating v4.1)        #
# =========================================================================== #
def test_an_unmatchable_comparative_metric_does_not_kill_the_verdict(tmp_path):
    """v4.1's trainer logs ``plan_ade``/``wm``/``oracle_ade`` and has NO
    ``g_op_fwd_ade_m``, which the card names. ``matched_step_ratio`` raises on
    that — and before 2026-07-26 the raise propagated and no v4.1 verdict could
    be rendered at all. The comparative block is a DIAGNOSTIC; it is now
    recorded as unavailable and the gate still decides."""
    ref = tmp_path / "ref.jsonl"
    ref.write_text('{"step": 100, "step_s": 5, "g_op_fwd_ade_m": 0.5}\n'
                   '{"step": 200, "step_s": 5, "g_op_fwd_ade_m": 0.4}\n')
    p = tmp_path / "card.json"
    p.write_text(json.dumps(dict(
        LEGACY_CARD, secondary=[], reference_run="v1",
        reference_log=str(ref), compare_metric="g_op_fwd_ade_m")))
    (tmp_path / "eval.json").write_text(json.dumps(
        {"cluster_bootstrap": {"model": {"ade_0_2s": _cb(0.4271, 0.36, 0.49)}}}))
    (tmp_path / "log.jsonl").write_text(
        '{"step": 10000, "step_s": 1, "plan_ade": 0.7}\n')
    rg.main(["check", "--card", str(p), "--log", str(tmp_path / "log.jsonl"),
             "--eval-json", str(tmp_path / "eval.json"),
             "--json", str(tmp_path / "gate.json")])
    g = json.loads((tmp_path / "gate.json").read_text())
    assert g["matched_step_ratio"]["available"] is False
    assert "diagnostic" in g["matched_step_ratio"]["note"]
    assert g["verdict"] == "CONTINUE"           # the verdict was still rendered


# =========================================================================== #
# 9. The committed artifacts the docstrings cite still say what we quote      #
# =========================================================================== #
@pytest.mark.skipif(not _E1A.exists(), reason="E1a artifact absent")
def test_the_e1a_numbers_this_module_quotes_are_the_artifact_s():
    d = json.loads(_E1A.read_text(encoding="utf-8"))
    pcs = d["paired_common_start"]
    assert pcs["20"]["overall"]["corridor_departure_rate"]["mean"] == E1A_CDR_K20_OVERALL
    assert pcs["185"]["overall"]["corridor_departure_rate"]["mean"] == E1A_CDR_K185_OVERALL
    assert pcs["185"]["junction"]["corridor_departure_rate"]["mean"] == E1A_CDR_K185_JUNCTION
    dl = pcs["deltas_vs_K20"]["overall"]["185"]["d_corridor_departure_rate"]
    assert dl["delta"] == 0.5842 and dl["separated"] is True
    assert dl["estimator"] == rg.PAIRED_CLUSTER_BOOTSTRAP_ESTIMATOR
    # the same 43 windows record essentially NOTHING on ADE@2s
    ade = pcs["deltas_vs_K20"]["overall"]["185"]["d_closed_ade2s_m"]
    assert ade["separated"] is False
    # K=185 is admissible under our ceiling; K=200 is not
    assert rg.validate_horizon_K(max(d["horizons_K"]))["horizon_K"] == 185
    with pytest.raises(SystemExit):
        rg.validate_horizon_K(200)


@pytest.mark.skipif(not _REFC_DRIVING.exists(), reason="REF-C driving JSON absent")
def test_the_refc_ade_this_module_quotes_is_the_artifact_s():
    d = json.loads(_REFC_DRIVING.read_text(encoding="utf-8"))
    node = d["headline"]["ade_0_2s"]
    assert node["mean"] == REFC_ADE_2S
    assert node["estimator"] == rg.CLUSTER_BOOTSTRAP_ESTIMATOR
    val, prov = rg._read_eval_metric(d, "ade_0_2s")
    assert val == REFC_ADE_2S and "episode_cluster_bootstrap" in prov


# =========================================================================== #
# 10. The co-primary EMITTER (gate_emitters.py)                               #
# =========================================================================== #
import gate_emitters as ge                                         # noqa: E402


def test_emitter_carries_the_horizon_the_junction_and_the_estimator():
    r = ge.corridor_from_corridor_dict(_corridor_doc())
    assert r["value"] == E1A_CDR_K185_OVERALL
    assert r["horizon_K"] == 185 and r["horizon_s"] == 18.5
    assert r["n_windows"] == 43 and r["n_episodes"] == 43
    assert r["estimator"] == "episode_cluster_bootstrap"
    assert r["junction"]["value"] == E1A_CDR_K185_JUNCTION
    assert r["junction"]["n_windows"] == 6
    assert r["evidence_class"].startswith("MEASURED")


def test_emitter_flags_a_blind_horizon_rather_than_emitting_it_quietly():
    r = ge.corridor_from_corridor_dict(_corridor_doc(K=20))
    assert r["horizon_K"] == 20
    assert "REFUSE" in r["WARNING_blind_horizon"]


def test_emitter_reports_a_skipped_panel_as_NOT_MEASURED():
    r = ge.corridor_from_corridor_dict(
        {"skipped": "no dense path in this dump (pred_dense/gt_dense).",
         "dense_surface_available": False})
    assert r["value"] is None
    assert r["evidence_class"].startswith("NOT MEASURED")


def test_the_co_primary_is_never_routed_through_the_secondary_channel(tmp_path):
    """An off-card ``--secondary-value`` becomes REPORT-ONLY and adjudicates
    nothing (§9 split card). Emitting the co-primary there would silently
    disarm it, so ``gate_values`` keeps the two channels apart."""
    p = tmp_path / "corridor.json"
    p.write_text(json.dumps(_corridor_doc()))
    out = ge.gate_values(corridor_json=str(p))
    assert out["co_primary"]["value"] == E1A_CDR_K185_OVERALL
    assert out["co_primary_arg"].startswith("--corridor-json")
    assert out["secondary_value_args"] == []
    assert "corridor_departure_rate" not in ge.GATE_NAMES
