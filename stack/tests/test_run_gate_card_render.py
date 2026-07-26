"""``run_gate`` — the three defects the flagship-v4 30 k gate found in the RENDERER.

WHY THESE TESTS EXIST (all three MEASURED at the 30 k gate, 2026-07-26)
----------------------------------------------------------------------
The 30 k gate had to be **hand-adjudicated**, because ``run_gate.py check``
could not process its own registered card
(``Project Steering/Gates/flagship-v4-30k.card.json``). Three defects:

D1  ``GateCard(**json.loads(card))`` — the card carries **11 registered keys**
    the dataclass has no slot for, so it died with
    ``TypeError: GateCard.__init__() got an unexpected keyword argument
    'registered_before_checkpoint_exists'``; and its ``co_primary`` is a NESTED
    dict where the tool expected FLAT ``co_primary_*`` fields, so even after the
    TypeError is dodged ``has_co_primary`` reads False and the **demoted**
    ``ade_0_2s`` illegally re-enters the kill conjunction.

D2  ``role: REPORT_ONLY_THIS_GATE`` and ``secondary_void`` were **unimplemented**
    — ``run_gate.py`` contained no reference to either. The tool had no state
    matching this card's actual configuration (primary demoted **and** co-primary
    report-only ⇒ **secondaries alone adjudicate**).

D3  ⭐ ``cur < card.gate_step`` returned ``NOT_YET`` on ``29999 < 30000`` for a
    run that completed all 30,000 steps (trainers log ``step`` 0-indexed;
    ``metrics.json final_step 29999``). **This alone would refuse every completed
    gate.**

THE LOAD-BEARING TEST is :func:`test_reproduces_the_hand_adjudicated_30k_verdict`
— it renders the REAL registered card against the REAL committed artifacts and
asserts the hand adjudication, number for number:

  ``wm_canary_ade_2s`` 1.1409 vs ≤ 0.55 **FAIL** · ``miss_at_2m`` 0.2123 vs
  ≤ 0.10 **FAIL** · 3 PASS · 2 unbuilt ⇒ **NOT-CONTINUE / RESTART** (budget 0/2),
  with ``nonav_route_beats_majority`` printed **INSTRUMENT-FAIL**.

Pure-Python: the committed card + the committed corridor/eval JSONs + a synthetic
train log. No torch, no GPU.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_gate as rg                                              # noqa: E402

_REPO = Path(__file__).resolve().parents[2]
_CARD_30K = _REPO / "Project Steering" / "Gates" / "flagship-v4-30k.card.json"
_GATE_DIR = (_REPO / "TanitAD Research Hub" / "Benchmarks & Eval"
             / "Implementation" / "incoming" / "2026-07-26-v4-30k-gate")
_EVAL_ORACLE = _GATE_DIR / "raw" / "flagship-v4-fromscratch-30k-oracle.json"
_CORRIDOR = _GATE_DIR / "coprimary" / "corridor_v4_30k_K185.json"

# The hand adjudication (GATE_30K_RESULTS.md §4.3 / §4.4), named so a drift goes
# red rather than quietly redefining what "reproduces" means.
WM_CANARY = 1.1409059762954712
ORACLE_IN_FAN = 0.23301841780357274
MISS_AT_2M = 0.2123
SEAM_NORM = 0.1208
ENCODER_LEVERS = 2
CDR_K185_OVERALL = 0.6388
CDR_K185_JUNCTION = 0.8432
ADE_0_2S_ORACLE = 0.6423

_HAVE_ARTIFACTS = _CARD_30K.exists() and _EVAL_ORACLE.exists() and _CORRIDOR.exists()
_needs_artifacts = pytest.mark.skipif(
    not _HAVE_ARTIFACTS, reason="the committed 30 k gate artifacts are absent")


# =========================================================================== #
# helpers                                                                     #
# =========================================================================== #
class _NS:
    """A stand-in for the argparse namespace ``cmd_check`` consumes."""

    def __init__(self, **kw):
        self.card = self.log = self.reference_log = None
        self.eval_json = self.corridor_json = self.corridor_paired_json = None
        self.secondary_value = {}
        self.fit_metric = None
        self.fit_from, self.fit_to = 1500, 7500
        self.json = None
        self.__dict__.update(kw)


def _train_log(tmp_path, last_step, name="log.jsonl", metric="plan_ade"):
    """A minimal but SHAPE-FAITHFUL trainer log: 0-indexed steps, and NO
    ``g_op_fwd_ade_m`` (the v4 line does not emit it — which is why the
    comparative diagnostic must degrade rather than abort)."""
    p = tmp_path / name
    rows = [{"step": s, metric: 1.0 - s * 1e-6, "elapsed_s": float(s), "step_s": 1.0}
            for s in range(max(0, last_step - 4), last_step + 1)]
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return str(p)


def _minimal_card(**over):
    d = {"run": "t", "gate_step": 100, "primary_metric": "ade_0_2s",
         "primary_threshold": 0.6, "primary_direction": "<=",
         "primary_source": "held-out"}
    d.update(over)
    return d


# =========================================================================== #
# D3 — the off-by-one that refused every completed gate                       #
# =========================================================================== #
class TestStepIndexing:
    def test_29999_of_30000_has_reached_the_gate(self):
        """⭐ The single most consequential line: before this, a COMPLETE
        59-hour 30,000-step run rendered NOT_YET on `29999 < 30000`."""
        s = rg.step_reached(29999, 30000)
        assert s["reached"] is True
        assert s["steps_completed"] == 30000

    def test_it_names_the_convention_it_used(self):
        """A verdict that silently picks an indexing convention is the same
        class of defect as one that silently picks a horizon."""
        assert rg.step_reached(29999, 30000)["convention"] == rg.STEP_INDEXING_ZERO
        assert rg.step_reached(30000, 30000)["convention"] == rg.STEP_INDEXING_ONE
        assert rg.step_reached(30123, 30000)["convention"] == rg.STEP_INDEXING_ONE

    def test_a_genuinely_early_run_is_still_refused(self):
        """The fix must not become a licence to gate an unfinished run: only the
        exact 0-indexed boundary is accepted, never `gate_step - 2`."""
        s = rg.step_reached(29998, 30000)
        assert s["reached"] is False
        assert "EITHER indexing convention" in s["note"]

    @pytest.mark.parametrize("cur,gate,reached", [
        (0, 1, True), (9999, 10000, True), (9998, 10000, False),
        (4999, 5000, True), (5000, 5000, True), (1, 10000, False)])
    def test_boundaries(self, cur, gate, reached):
        assert rg.step_reached(cur, gate)["reached"] is reached

    def test_check_renders_a_verdict_at_29999(self, tmp_path):
        card = tmp_path / "c.json"
        card.write_text(json.dumps(_minimal_card(gate_step=30000)), encoding="utf-8")
        out_json = tmp_path / "o.json"
        a = _NS(card=str(card), log=_train_log(tmp_path, 29999),
                eval_json=None, json=str(out_json))
        rg.cmd_check(a)
        out = json.loads(out_json.read_text(encoding="utf-8"))
        assert out["verdict"] != "NOT_YET"
        assert out["step_indexing"]["convention"] == rg.STEP_INDEXING_ZERO


# =========================================================================== #
# D1 — the registered card must LOAD                                          #
# =========================================================================== #
class TestCardLoads:
    @_needs_artifacts
    def test_the_registered_30k_card_loads(self):
        """`GateCard(**json.loads(...))` raised TypeError on this exact file."""
        raw = json.loads(_CARD_30K.read_text(encoding="utf-8"))
        with pytest.raises(TypeError):
            rg.GateCard(**raw)                       # the DEFECT, pinned
        card = rg.GateCard.from_dict(raw)            # the FIX
        assert card.run == "flagship-v4-fromscratch"
        assert card.gate_step == 30000

    @_needs_artifacts
    def test_unknown_registered_keys_are_preserved_not_dropped(self):
        """A card is a PRE-REGISTRATION document. Dropping its fields silently is
        how a card's own text stops binding the tool that renders it."""
        raw = json.loads(_CARD_30K.read_text(encoding="utf-8"))
        card = rg.GateCard.from_dict(raw)
        for key in ("registered_before_checkpoint_exists", "registration_note",
                    "primary_role_note", "goal_provenance",
                    "goal_provenance_note", "reference_ade_0_2s",
                    "reference_note", "required_reporting", "preflight_checks"):
            assert key in card.card_extras, key
        assert card.card_extras["goal_provenance"] == "ORACLE"

    @_needs_artifacts
    def test_the_nested_co_primary_dict_maps_onto_the_flat_fields(self):
        """The second half of D1: with `co_primary` unmapped, `has_co_primary`
        is False and the DEMOTED ade_0_2s illegally re-enters the conjunction."""
        card = rg.GateCard.from_dict(
            json.loads(_CARD_30K.read_text(encoding="utf-8")))
        assert card.has_co_primary is True
        assert card.co_primary_metric == "corridor_departure_rate"
        assert card.co_primary_horizon_K == 185
        assert card.co_primary_corridor_m == 1.75
        assert card.co_primary_role == rg.CO_PRIMARY_ROLE_REPORT_ONLY
        # NOTHING is invented: the card registers no bar, so none appears.
        assert card.co_primary_threshold is None

    def test_an_unknown_co_primary_role_fails_loudly(self):
        """Silently treating an unimplemented role as 'kill' is precisely how the
        30 k gate mis-adjudicated."""
        with pytest.raises(SystemExit, match="unknown co-primary role"):
            rg.GateCard.from_dict(_minimal_card(
                co_primary={"metric": "corridor_departure_rate",
                            "horizon_K": 185, "role": "advisory"}))

    def test_a_pre_horizon_card_still_loads_unchanged(self):
        """Back-compat is bounded but real: a card written before 2026-07-26
        carries no co-primary and still renders its historical verdict."""
        card = rg.GateCard.from_dict(_minimal_card())
        assert card.has_co_primary is False
        assert card.co_primary_role == rg.CO_PRIMARY_ROLE_KILL
        assert card.secondary_void == []
        assert card.card_extras == {}


# =========================================================================== #
# D2a — REPORT_ONLY_THIS_GATE                                                 #
# =========================================================================== #
def _report_only_card(**over):
    d = _minimal_card(
        gate_step=100, secondary=["a<=1.0"],
        co_primary={"metric": "corridor_departure_rate", "horizon_K": 185,
                    "corridor_half_width_m": 1.75,
                    "role": rg.CO_PRIMARY_ROLE_REPORT_ONLY,
                    "report_only_rationale": "no bar has ever been agreed",
                    "becomes_kill_criterion_at": "the next v4-line gate"},
        primary_role="diagnostic")
    d.update(over)
    return d


class TestReportOnlyCoPrimary:
    def test_it_is_registered_measured_and_excluded(self):
        card = rg.GateCard.from_dict(_report_only_card())
        assert card.has_co_primary is True          # registered
        assert card.co_primary_is_report_only is True
        assert card.co_primary_adjudicates is False  # NOT in the conjunction

    def test_the_kill_conjunction_is_the_secondaries_alone(self, tmp_path):
        """Primary demoted AND co-primary report-only ⇒ secondaries adjudicate.
        The tool had no state for this; whichever way the card was projected it
        mis-adjudicated."""
        card = tmp_path / "c.json"
        card.write_text(json.dumps(_report_only_card()), encoding="utf-8")
        ev = tmp_path / "e.json"
        ev.write_text(json.dumps({"ade_0_2s": 99.0}), encoding="utf-8")
        out_json = tmp_path / "o.json"
        a = _NS(card=str(card), log=_train_log(tmp_path, 100),
                eval_json=str(ev), json=str(out_json),
                secondary_value={"a": 0.5})
        rg.cmd_check(a)
        out = json.loads(out_json.read_text(encoding="utf-8"))
        # ade_0_2s = 99.0 fails its 0.6 bar by 165x and STILL does not kill.
        assert out["primary"]["pass"] is False
        assert out["primary"]["role"] == "diagnostic"
        assert out["verdict"] == "CONTINUE"
        assert out["verdict_adjudicated_by"] == ["secondary(1)"]

    def test_a_report_only_co_primary_is_printed_in_full(self, tmp_path, capsys):
        """'MEASURED, REPORTED IN FULL' is the card's own requirement — the
        overall value, its interval, its n, and the junction stratum."""
        card = tmp_path / "c.json"
        card.write_text(json.dumps(_report_only_card()), encoding="utf-8")
        ev = tmp_path / "e.json"
        ev.write_text(json.dumps({"ade_0_2s": 0.5}), encoding="utf-8")
        a = _NS(card=str(card), log=_train_log(tmp_path, 100),
                eval_json=str(ev), corridor_json=str(_CORRIDOR),
                secondary_value={"a": 0.5})
        if not _CORRIDOR.exists():
            pytest.skip("committed corridor artifact absent")
        rg.cmd_check(a)
        txt = capsys.readouterr().out
        assert rg.CO_PRIMARY_ROLE_REPORT_ONLY in txt
        assert str(CDR_K185_OVERALL) in txt
        assert str(CDR_K185_JUNCTION) in txt
        assert "does not adjudicate" in txt or "EXCLUDED" in txt

    def test_no_threshold_never_becomes_a_pass_or_a_fail(self, tmp_path):
        """An unthresholded co-primary must read REPORTED — not PASS (which
        would launder it) and not FAIL (which would kill on an invented bar)."""
        if not _CORRIDOR.exists():
            pytest.skip("committed corridor artifact absent")
        card = rg.GateCard.from_dict(_report_only_card())
        a = _NS(corridor_json=str(_CORRIDOR))
        co = rg._co_primary_block(a, card)
        assert co["measured"] is True
        assert co["pass"] is None
        assert co["value"] == CDR_K185_OVERALL
        assert co["no_threshold_note"]

    def test_a_kill_co_primary_with_no_bar_is_refused(self, tmp_path):
        """Picking a bar now — after the number exists — is GATE_PROTOCOL 0.3's
        garden of forking paths. The tool refuses instead of guessing."""
        d = _report_only_card()
        d["co_primary"]["role"] = rg.CO_PRIMARY_ROLE_KILL
        card = tmp_path / "c.json"
        card.write_text(json.dumps(d), encoding="utf-8")
        ev = tmp_path / "e.json"
        ev.write_text(json.dumps({"ade_0_2s": 0.5}), encoding="utf-8")
        a = _NS(card=str(card), log=_train_log(tmp_path, 100),
                eval_json=str(ev), secondary_value={"a": 0.5})
        with pytest.raises(SystemExit, match="NO\n?\\s*threshold|no bar|NO "):
            rg.cmd_check(a)

    def test_a_kill_role_co_primary_still_adjudicates(self, tmp_path):
        """The report-only path must not weaken the normal one."""
        d = _report_only_card()
        d["co_primary"].update(role=rg.CO_PRIMARY_ROLE_KILL, threshold=0.35)
        card = rg.GateCard.from_dict(d)
        assert card.co_primary_adjudicates is True
        assert card.co_primary_is_report_only is False


# =========================================================================== #
# D2b — secondary_void (GATE_PROTOCOL 0.7)                                    #
# =========================================================================== #
_VOID_ENTRY = {
    "metric": "nonav_route_beats_majority", "original_threshold": ">=1",
    "status": "VOID_BY_CONSTRUCTION",
    "adjudication": "INSTRUMENT-FAIL, NEVER MODEL-FAIL",
    "authority": "GATE_PROTOCOL 0.7",
    "reason": "the target is a lookup of the input",
    "re_arms_when": "an arm trains with real route supervision"}


class TestSecondaryVoid:
    def test_it_is_excluded_from_the_conjunction(self, tmp_path):
        card = tmp_path / "c.json"
        card.write_text(json.dumps(_minimal_card(
            gate_step=100, secondary=["a<=1.0"],
            secondary_void=[_VOID_ENTRY])), encoding="utf-8")
        ev = tmp_path / "e.json"
        ev.write_text(json.dumps({"ade_0_2s": 0.5}), encoding="utf-8")
        out_json = tmp_path / "o.json"
        a = _NS(card=str(card), log=_train_log(tmp_path, 100),
                eval_json=str(ev), json=str(out_json),
                secondary_value={"a": 0.5})
        rg.cmd_check(a)
        out = json.loads(out_json.read_text(encoding="utf-8"))
        assert out["verdict"] == "CONTINUE"          # the void did NOT kill it
        assert out["kill_conjunction"]["n_void"] == 1
        assert "nonav_route_beats_majority" not in out["kill_conjunction"]["failed"]

    def test_it_is_PRINTED_with_its_adjudication_string(self, tmp_path, capsys):
        """GATE_PROTOCOL 0.7 step 3: **a suppressed criterion that is not printed
        is indistinguishable from one that passed.**"""
        card = tmp_path / "c.json"
        card.write_text(json.dumps(_minimal_card(
            gate_step=100, secondary_void=[_VOID_ENTRY])), encoding="utf-8")
        ev = tmp_path / "e.json"
        ev.write_text(json.dumps({"ade_0_2s": 0.5}), encoding="utf-8")
        out_json = tmp_path / "o.json"
        a = _NS(card=str(card), log=_train_log(tmp_path, 100),
                eval_json=str(ev), json=str(out_json))
        rg.cmd_check(a)
        txt = capsys.readouterr().out
        assert "nonav_route_beats_majority" in txt
        assert "INSTRUMENT-FAIL" in txt
        assert "VOID_BY_CONSTRUCTION" in txt
        assert "GATE_PROTOCOL 0.7" in txt
        assert "IN KILL SET  : NO" in txt
        out = json.loads(out_json.read_text(encoding="utf-8"))
        row = out["secondary_void"][0]
        assert row["in_kill_set"] is False and row["adjudicated"] is False
        assert row["adjudication"] == "INSTRUMENT-FAIL, NEVER MODEL-FAIL"

    def test_a_void_metric_listed_as_a_KILL_secondary_is_still_void(self, tmp_path):
        """`flagship-v4.card.json` lists it among the KILL secondaries. 0.7 is an
        adjudication, not a suggestion — without this a healthy arm dies on a
        label bug, 'the most expensive possible way to be wrong'."""
        card = tmp_path / "c.json"
        card.write_text(json.dumps(_minimal_card(
            gate_step=100, secondary=["nonav_route_beats_majority>=1"],
            secondary_void=[_VOID_ENTRY])), encoding="utf-8")
        ev = tmp_path / "e.json"
        ev.write_text(json.dumps({"ade_0_2s": 0.5}), encoding="utf-8")
        out_json = tmp_path / "o.json"
        a = _NS(card=str(card), log=_train_log(tmp_path, 100),
                eval_json=str(ev), json=str(out_json),
                secondary_value={"nonav_route_beats_majority": 0.0})
        rg.cmd_check(a)
        out = json.loads(out_json.read_text(encoding="utf-8"))
        assert out["secondary"][0]["voided"] is True
        assert out["secondary"][0]["pass"] is None
        assert out["verdict"] == "CONTINUE"

    def test_a_bare_string_void_entry_gets_the_protocol_default(self):
        """An unlabelled suppression is the thing 0.7 forbids, so a default
        adjudication string is always attached."""
        rows = rg._normalise_void(["x>=1"])
        assert rows[0]["metric"] == "x"
        assert rows[0]["adjudication"] == "INSTRUMENT-FAIL, NEVER MODEL-FAIL"
        assert rows[0]["authority"] == "GATE_PROTOCOL 0.7"

    def test_a_void_entry_with_no_metric_is_refused(self):
        with pytest.raises(SystemExit, match="no 'metric'"):
            rg._normalise_void([{"status": "VOID_BY_CONSTRUCTION"}])


# =========================================================================== #
# The unsatisfiable conjunction                                               #
# =========================================================================== #
class TestUnsatisfiableConjunction:
    def _run(self, tmp_path, supplied, secondary):
        card = tmp_path / "c.json"
        card.write_text(json.dumps(_minimal_card(
            gate_step=100, secondary=secondary, restart_cap=2,
            lever_family="joint-planner-wm")), encoding="utf-8")
        ev = tmp_path / "e.json"
        ev.write_text(json.dumps({"ade_0_2s": 0.5}), encoding="utf-8")
        out_json = tmp_path / "o.json"
        a = _NS(card=str(card), log=_train_log(tmp_path, 100),
                eval_json=str(ev), json=str(out_json), secondary_value=supplied)
        rg.cmd_check(a)
        return json.loads(out_json.read_text(encoding="utf-8"))

    def test_a_hard_fail_beats_an_unmeasured_secondary(self, tmp_path):
        """No future measurement turns a FAIL into a PASS, so an outstanding
        secondary cannot downgrade a determined NOT-CONTINUE to INCOMPLETE."""
        out = self._run(tmp_path, {"a": 9.0}, ["a<=1.0", "b<=1.0"])
        assert out["verdict"] == "RESTART"
        assert out["not_continue"] is True
        assert out["formally_incomplete"] is True
        assert out["kill_conjunction"]["unmeasured"] == ["b"]

    def test_INCOMPLETE_survives_when_nothing_has_failed(self, tmp_path):
        """The reservation is real: with no FAIL the conjunction is still
        satisfiable and no verdict is admissible."""
        out = self._run(tmp_path, {"a": 0.5}, ["a<=1.0", "b<=1.0"])
        assert out["verdict"] == "INCOMPLETE"
        assert out["kill_conjunction"]["unmeasured"] == ["b"]

    def test_an_exhausted_restart_budget_refutes_the_lever_family(self, tmp_path):
        card = tmp_path / "c.json"
        card.write_text(json.dumps(_minimal_card(
            gate_step=100, secondary=["a<=1.0"], restarts_used=2,
            restart_cap=2, lever_family="joint-planner-wm")), encoding="utf-8")
        ev = tmp_path / "e.json"
        ev.write_text(json.dumps({"ade_0_2s": 0.5}), encoding="utf-8")
        out_json = tmp_path / "o.json"
        a = _NS(card=str(card), log=_train_log(tmp_path, 100),
                eval_json=str(ev), json=str(out_json), secondary_value={"a": 9.0})
        rg.cmd_check(a)
        out = json.loads(out_json.read_text(encoding="utf-8"))
        assert out["verdict"] == "REFUTE_LEVER_FAMILY"


# =========================================================================== #
# The tripwire that must NOT be disarmed                                      #
# =========================================================================== #
class TestDeprecatedTripwireIntact:
    def test_deprecated_present_still_keys_on_the_literal_heldout_name(self):
        """`bench.py` deliberately retains `heldout` as a back-compat alias.
        Renaming that key silently disarms the gate's own refusal
        (GATE_PROTOCOL 0.5, 'Live tripwire, do not clean up')."""
        ev = {"heldout": {"model": {"ade_0_2s": {
            "mean": 1.0, "estimator": rg.DEPRECATED_ESTIMATOR}}}}
        assert rg._deprecated_present(ev, ("ade_0_2s",)) is True
        assert rg._deprecated_present({"held_out": ev["heldout"]},
                                      ("ade_0_2s",)) is False
        with pytest.raises(SystemExit, match="DEPRECATED"):
            rg._read_eval_metric(ev, "ade_0_2s")

    def test_the_literal_key_appears_in_the_source(self):
        src = (Path(rg.__file__)).read_text(encoding="utf-8")
        assert '("heldout", "model")' in src
        assert '("driving", "heldout", "model")' in src


# =========================================================================== #
# ⭐ THE REPRODUCTION — the fixed tool vs the hand adjudication                #
# =========================================================================== #
@_needs_artifacts
def test_reproduces_the_hand_adjudicated_30k_verdict(tmp_path):
    """The whole point of the repair, on the REAL registered card and the REAL
    committed artifacts.

    Hand adjudication (``GATE_30K_RESULTS.md`` §4.3/§4.4):
      * ``wm_canary_ade_2s`` 1.1409 vs ≤ 0.55 — FAIL (2.07x over)
      * ``miss_at_2m``       0.2123 vs ≤ 0.10 — FAIL (2.12x over)
      * 3 PASS · 2 NOT MEASURED (no emitter exists for either)
      * co-primary 0.6388 / junction 0.8432 — REPORT-ONLY, does not adjudicate
      * ``nonav_route_beats_majority`` — INSTRUMENT-FAIL, printed
      * ⇒ NOT-CONTINUE; ``restarts_used 0 / restart_cap 2`` ⇒ RESTART
    """
    out_json = tmp_path / "verdict.json"
    a = _NS(card=str(_CARD_30K), log=_train_log(tmp_path, 29999),
            eval_json=str(_EVAL_ORACLE), corridor_json=str(_CORRIDOR),
            json=str(out_json),
            secondary_value={"wm_canary_ade_2s": WM_CANARY,
                             "oracle_in_fan": ORACLE_IN_FAN,
                             "miss_at_2m": MISS_AT_2M,
                             "seam_norm_ratio_max": SEAM_NORM,
                             "encoder_touching_levers": float(ENCODER_LEVERS)})
    rg.cmd_check(a)
    out = json.loads(out_json.read_text(encoding="utf-8"))

    # -- the run reached its gate (D3) --------------------------------------- #
    assert out["current_step"] == 29999
    assert out["step_indexing"]["reached"] is True
    assert out["step_indexing"]["convention"] == rg.STEP_INDEXING_ZERO

    # -- the demoted primary is recorded and does NOT adjudicate ------------- #
    assert out["primary"]["value"] == ADE_0_2S_ORACLE
    assert out["primary"]["pass"] is False
    assert out["primary"]["role"] == "diagnostic"
    assert "co_primary.corridor_departure_rate" not in out["verdict_adjudicated_by"]
    assert out["verdict_adjudicated_by"] == ["secondary(7)"]

    # -- the co-primary: measured, reported in full, excluded (D2a) ----------- #
    co = out["co_primary"]
    assert co["role"] == rg.CO_PRIMARY_ROLE_REPORT_ONLY
    assert co["adjudicates"] is False and co["measured"] is True
    assert co["pass"] is None
    assert co["value"] == CDR_K185_OVERALL
    assert co["junction"]["value"] == CDR_K185_JUNCTION
    assert co["n_windows"] == 41 and co["n_episodes"] == 40
    assert co["horizon_K"] == 185
    assert co["estimator"] == rg.CLUSTER_BOOTSTRAP_ESTIMATOR

    # -- the secondaries: 2 FAIL, 3 PASS, 2 unbuilt --------------------------- #
    by = {s["metric"]: s for s in out["secondary"]}
    assert by["wm_canary_ade_2s"]["pass"] is False
    assert round(by["wm_canary_ade_2s"]["value"], 4) == 1.1409
    assert by["miss_at_2m"]["pass"] is False
    assert by["miss_at_2m"]["value"] == 0.2123
    assert by["oracle_in_fan"]["pass"] is True
    assert by["seam_norm_ratio_max"]["pass"] is True
    assert by["encoder_touching_levers"]["pass"] is True
    assert by["speed_benefit_recovered_frac"]["pass"] is None
    assert by["deploy_tick_p99_ms"]["pass"] is None
    kc = out["kill_conjunction"]
    assert (kc["n_pass"], kc["n_fail"], kc["n_unmeasured"]) == (3, 2, 2)

    # -- the void secondary: printed, INSTRUMENT-FAIL, never a model fail ----- #
    void = out["secondary_void"]
    assert len(void) == 1
    assert void[0]["metric"] == "nonav_route_beats_majority"
    assert void[0]["adjudication"] == "INSTRUMENT-FAIL, NEVER MODEL-FAIL"
    assert void[0]["in_kill_set"] is False

    # -- NOT-CONTINUE / RESTART, budget 0/2 ----------------------------------- #
    assert out["verdict"] == "RESTART"
    assert out["not_continue"] is True
    assert out["formally_incomplete"] is True
    assert out["restart_budget_ok"] is True

    # -- the card's own text survived the render (D1) ------------------------- #
    assert out["card_extras"]["goal_provenance"] == "ORACLE"
    assert "required_reporting" in out["card_extras"]


@_needs_artifacts
def test_the_old_code_path_would_have_refused_this_gate():
    """Pins the DEFECT itself, so a regression is loud rather than silent."""
    raw = json.loads(_CARD_30K.read_text(encoding="utf-8"))
    with pytest.raises(TypeError):
        rg.GateCard(**raw)                                   # D1
    card = rg.GateCard.from_dict(raw)
    assert 29999 < card.gate_step                            # D3, the comparison
    assert rg.step_reached(29999, card.gate_step)["reached"] # ...and the fix
