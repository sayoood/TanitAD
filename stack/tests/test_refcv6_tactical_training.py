"""refcv6 §4/§5 — THE TRAINER WIRING of the tactical behaviour decoder and the
4-value max-speed input.

⛔ THE DEFECT THIS MODULE EXISTS FOR, MEASURED at tip ``837c308``:
``refcv6_tactical.TacticalBehaviourDecoder`` was BUILT (``refc_v3.py:1128``),
FORWARD-RUN (``:1622``) and wrote ``cache["tacv6_*"]`` (``:1627``) — while
``scripts/refc_v3_train.py`` contained **ZERO** occurrences of the string
``tacv6``. 2,262,020 parameters (at ``d_bev`` 256) with no gradient path and no
flag that could switch them on. It is ``tac_goal_tok_head`` — 11,286 params at
``grad_abs_sum`` EXACTLY 0 for all 40,284 steps of refcv5-v2 — at 200x scale.

⛔ EVERY GUARD IS PROVEN BY MUTATION, NEVER BY INSPECTION. An AST census once
read "0 suspects" on BOTH the fixed and the broken trainer, and four checks in
one night were green forever because their expected value was an expression
over the code under test. So each test here either runs the real parser and the
real pin, or reintroduces the defect and asserts the check GOES RED.

⭐ AND EVERY REFUSAL CARRIES ITS GREEN CONTROL. MEASURED while writing the
companion ``refusals.py``: an argv missing ``--agent-join`` made all four green
controls read REFUSED while all fifteen refusals also read REFUSED — a table of
"every guard fires" with a BRICK underneath. A RED-only table cannot tell a
guard from a brick.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import refc_v3_train as T                                    # noqa: E402
from tanitad.refs import refc_v3 as v3                       # noqa: E402
from tanitad.refs import refcv6_max_speed as v6ms            # noqa: E402
from tanitad.refs import refcv6_tactical as v6tac            # noqa: E402

#: A minimal argv that reaches the pin. ⛔ `--agent-join` is a DUMMY path here:
#: the pin does not open it, and the goal is to exercise `_pin_refcv6_tactical`
#: without a fixture. `_args_ok` asserts the base actually passes, so a future
#: guard that starts opening the path fails LOUDLY instead of turning every
#: refusal below into a false positive.
BASE = ["--arm", "hier", "--size", "tiny", "--out", "X",
        "--v7-labels", "labels.jsonl.gz",
        "--agents", "head", "--w-agent", "1.0",
        "--agent-join", "join.jsonl.xz", "--agent-join-verify", "off"]
TAC = ["--tac-decoder-v6", "--w-tac-v6", "1.0"]


def _pin(argv):
    """Real parser + real pin. -> (ok, message)."""
    args = T.build_parser().parse_args(argv)
    cfg = v3.RefCV3Config(hier=(args.arm == "hier"))
    try:
        T._pin_trainer_cfg(cfg, args)
        return True, None, cfg
    except SystemExit as exc:
        return False, str(exc), None


def _drop(argv, flag, n_values=1):
    """⛔ Explicit removal, and it REFUSES to be a no-op. A mutation harness
    that silently fails to mutate makes every case pass vacuously."""
    out, i = [], 0
    while i < len(argv):
        if argv[i] == flag:
            i += 1 + n_values
            continue
        out.append(argv[i])
        i += 1
    assert len(out) != len(argv), f"drop({flag!r}) removed nothing"
    return out


def _set(argv, flag, value):
    out = list(argv)
    assert flag in out, f"{flag!r} absent from base argv"
    out[out.index(flag) + 1] = value
    return out


# =========================================================================== #
# 0. THE GREEN CONTROL — without it every refusal below is unreadable         #
# =========================================================================== #
def test_GREEN_the_valid_tactical_arm_passes():
    """⛔ IF THIS FAILS, EVERY `test_RED_*` BELOW IS MEANINGLESS — they would
    all be refusing for a reason that has nothing to do with the thing they
    name. This is the exact failure measured in `refusals.py`'s first run."""
    ok, msg, cfg = _pin(BASE + TAC)
    assert ok, f"the valid arm was refused: {msg}"
    assert cfg.tac_decoder_v6 is True
    assert cfg.tac_decoder_cfg.d_bev == 0          # agent-only: the live arm
    assert tuple(cfg.tac_decoder_cfg.sources) == ("agent",)


# =========================================================================== #
# 1. THE CORE REFUSAL — a built head with no live weight                      #
# =========================================================================== #
def test_RED_decoder_with_zero_weight_refuses():
    ok, msg, _ = _pin(BASE + ["--tac-decoder-v6", "--w-tac-v6", "0"])
    assert not ok
    # ⛔ The message must name the DEFECT, not merely the flag combination —
    # a reader who meets it should learn why 2.26 M dead parameters matter.
    assert "grad" in msg.lower() and "2,262,020" in msg


def test_RED_weight_without_decoder_refuses():
    ok, msg, _ = _pin(_drop(BASE + TAC, "--tac-decoder-v6", 0))
    assert not ok and "tacv6_goal_logits" in msg


@pytest.mark.parametrize("argv,needle", [
    (_set(BASE + TAC, "--arm", "flat"), "FLAT"),
    (_drop(BASE + TAC, "--v7-labels"), "kin3"),
    (BASE + TAC + ["--tac-goal-tok-head"], "two experiments"),
    # ⭐ PI RULING 2026-09-17 R2 lifted the STRUCTURAL block; what remains is a
    # PRECONDITION. The BEV tokens ARE the supervised map branch's features, so
    # a BEV width with no live map weight still refuses — for a different, and
    # now correct, reason. ⛔ The needle is `--w-map`, not `bev_tokens`: a test
    # that kept the old needle would go green again the day someone restored
    # the old blanket refusal, which is the regression this pins against.
    (BASE + TAC + ["--tac-decoder-d-bev", "96"], "--w-map"),
])
def test_RED_each_dead_configuration_refuses_for_its_own_reason(argv, needle):
    ok, msg, _ = _pin(argv)
    assert not ok, f"expected a refusal, got a pass: {argv}"
    assert needle in msg, f"refused for the wrong reason: {msg[:200]}"


def test_GREEN_d_bev_zero_is_the_buildable_arm():
    """The control for the `--tac-decoder-d-bev` refusals above: 0 must
    pass, or that guard is a brick rather than a gate."""
    ok, msg, _ = _pin(BASE + TAC + ["--tac-decoder-d-bev", "0"])
    assert ok, msg


def test_graft_behaviour_sel_reaches_the_BUILT_DECODER_not_only_the_config():
    """⛔ READ OFF THE BUILT OBJECT, NOT OFF `cfg`. The 2026-09-17 `--image-hw`
    post-mortem is exactly this: a check that diffs argv PASSED while the
    defect lived between argv and the model. The flag's real destination is
    ``AnchoredDiffusionDecoder.graft_behaviour_sel`` (``refc.py:1792``), four
    constructor hops from the Namespace.

    ⭐ Both arms are asserted, so this cannot pass on a decoder that hardcodes
    ``True`` — a single-arm assertion would."""
    seen = {}
    for extra, key in ((["--graft-behaviour-sel"], True), ([], False)):
        args = T.build_parser().parse_args(BASE + TAC + extra)
        cfg = v3.RefCV3Config(hier=True)
        T._pin_trainer_cfg(cfg, args)
        cfg.core.graft_target_latent = True      # an unrelated v3 precondition
        model = v3.RefCV3Model(cfg)
        seen[key] = bool(model.core.decoder.graft_behaviour_sel)
        # the decoder itself is built either way — only the GATE moves
        assert model.tac_decoder_v6 is not None
        assert model.tac_decoder_v6.n_params == 2228996   # agent-only, d 384
    assert seen == {True: True, False: False}, seen


def _map_argv(tmp_path) -> list:
    """The argv of a LIVE map arm: a real per-clip extrinsics table on disk
    (the pin OPENS it), a declared frame, and the timm trunk.

    ⛔ 256 x 1024 and not the PI's 408 x 1024 — deliberately, and it is not a
    geometry this test believes in: ``TimmTrunkConfig.__post_init__`` REFUSES
    any axis that 32 does not divide, and ``408 % 32 == 8``. That refusal is
    correct (at 408 the trunk emits a 26-row stride-16 map while
    ``timm_trunk.py:406`` would declare ``408 // 16 == 25``), and it is
    reported as a named blocker on ruling R1 rather than worked around here.
    Nothing in this file reads the numbers: they go in as argv.
    """
    p = tmp_path / "extr.json"
    p.write_text(json.dumps({
        "clip%02d" % i: {"qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0,
                         "tx": 2.05, "ty": 0.0, "tz": 1.30 + 0.01 * i}
        for i in range(4)}), encoding="utf-8")
    return ["--trunk", "timm", "--image-hw", "256", "1024",
            "--w-map", "1.0", "--map-gt-root", "m",
            "--agent-rig-camera", "extrinsics",
            "--agent-rig-extrinsics", str(p)]


def test_the_bev_seam_stamp_is_the_ARM_S_OWN_answer_not_a_constant(tmp_path):
    """⭐⭐ PI RULING 2026-09-17 R2/R3, pinned on BOTH arms.

    ⛔ Until the ruling this field was the literal ``False`` and the test
    asserted it. A one-arm assertion cannot tell a wired seam from a hardcoded
    constant — which is exactly what it was — so both arms are read here and
    the pair must DIFFER. The PI asked for behaviours learned from *"the agent
    and the map"*; an arm that ran agent-only must still say so in its own
    artifact, and an arm that ran with the map must not have to be taken on
    trust.
    """
    seen = {}
    for extra, tag in ((["--tac-decoder-d-bev", "96"], "bev"),
                       ([], "agent_only")):
        args = T.build_parser().parse_args(
            BASE + TAC + _map_argv(tmp_path) + extra)
        cfg = v3.RefCV3Config(hier=True)
        T._pin_trainer_cfg(cfg, args)
        seen[tag] = T._seam_stamp(cfg, args)["tac_decoder_v6"]

    assert seen["bev"]["bev_tokens_reach_decoder"] is True
    assert seen["bev"]["sources"] == ["agent", "bev"]
    # R3: attached is the RULING, so a default BEV arm trains the trunk.
    assert seen["bev"]["bev_grad_reaches_trunk"] is True
    assert seen["bev"]["bev_detached"] is False

    # ⛔ THE DISCRIMINATING CONTROL. Same everything, `d_bev 0`: the stamp must
    # FLIP. Without this the three assertions above pass on a constant `True`.
    assert seen["agent_only"]["bev_tokens_reach_decoder"] is False
    assert seen["agent_only"]["sources"] == ["agent"]
    assert seen["agent_only"]["bev_grad_reaches_trunk"] is False
    assert seen["bev"]["w"] == 1.0    # the fact `tac_goal_tok_head` lacked


def test_the_R3_detach_ablation_is_expressible_and_STAMPED(tmp_path):
    """⚠️ The PI ruled the tactical loss MAY shape the trunk. The ablation must
    be RUNNABLE — otherwise the ruling is untestable — and it must be VISIBLE in
    ``config.json``, or a detached arm and an attached arm are indistinguishable
    in the record. ⛔ And it must REFUSE when there is nothing to detach: a flag
    that is silently inert while the record stamps it on is the dead-flag class
    this trainer refuses five times over."""
    args = T.build_parser().parse_args(
        BASE + TAC + _map_argv(tmp_path)
        + ["--tac-decoder-d-bev", "96", "--tac-decoder-bev-detach"])
    cfg = v3.RefCV3Config(hier=True)
    T._pin_trainer_cfg(cfg, args)
    st = T._seam_stamp(cfg, args)["tac_decoder_v6"]
    assert cfg.tac_decoder_bev_detach is True
    assert st["bev_detached"] is True
    # ⭐ the seam is still WIRED — only the gradient is cut. Conflating the two
    # would let a detached arm read as an agent-only one.
    assert st["bev_tokens_reach_decoder"] is True
    assert st["bev_grad_reaches_trunk"] is False

    ok, msg, _ = _pin(BASE + TAC + ["--tac-decoder-bev-detach"])
    assert not ok, "a detach flag with no BEV path must refuse"
    assert "no BEV path to detach" in msg, msg[:200]


# =========================================================================== #
# 2. §5 — the 4-value set-speed                                               #
# =========================================================================== #
@pytest.mark.parametrize("argv,needle", [
    (BASE + TAC + ["--max-speed-input-v6"], "sidecar"),
    (BASE + TAC + ["--max-speed-input-v6", "--speed-max-sidecar-v6", "s.jsonl",
                   "--max-speed-input"], "Pick one"),
    (_drop(_drop(BASE + TAC, "--tac-decoder-v6", 0), "--w-tac-v6")
     + ["--max-speed-input-v6", "--speed-max-sidecar-v6", "s.jsonl"],
     "ONLY consumer"),
])
def test_RED_max_speed_v6_dead_configurations_refuse(argv, needle):
    ok, msg, _ = _pin(argv)
    assert not ok and needle in msg, msg[:200]


def test_GREEN_max_speed_v6_with_a_sidecar_passes():
    ok, msg, cfg = _pin(BASE + TAC + ["--max-speed-input-v6",
                                      "--speed-max-sidecar-v6", "s.jsonl"])
    assert ok, msg
    assert cfg.max_speed_onehot_v6 is True


# =========================================================================== #
# 3. THE SIDECAR READER — mutated copies of a real-shaped sidecar             #
# =========================================================================== #
def _sidecar(rows):
    d = tempfile.mkdtemp()
    p = Path(d) / "s.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return str(p)


def _row(sid, v=11.19, b=1, valid=1):
    oh = [0.0] * v6ms.N_SPEED_MAX_BINS_V6
    oh[b] = 1.0
    return {"sid": sid, "v_hi_ms": v, "bin": b, "one_hot": oh,
            "limit_kmh": v6ms.SPEED_MAX_STEPS_KMH_V6[b],
            "over_ceiling": False, "valid": valid}


def test_GREEN_a_well_formed_sidecar_reads():
    by_sid, rep = v6ms.read_speed_max_sidecar_v6(
        _sidecar([_row(1), _row(2)]))
    assert rep["n_valid"] == 2 and by_sid[1][1] == 1.0
    # ⛔ THE RAW m/s IS SHIPPED, NOT THE BIN. The ladder is applied ONCE, on
    # the model side — re-snapping a shipped bucket moved 2,631 of 4,572 clips
    # (57.5 %) one step up on the E16 channel.
    assert by_sid[1][0] == pytest.approx(11.19)


@pytest.mark.parametrize("rows,needle", [
    ([{k: v for k, v in _row(1).items() if k != "sid"}], "`sid`"),
    ([_row(1, b=3)], "ladder"),
    ([_row(1, valid=0)], "valid ceiling"),
    ([], "ZERO rows"),
])
def test_RED_the_reader_refuses_each_broken_sidecar(rows, needle):
    with pytest.raises(SystemExit) as e:
        v6ms.read_speed_max_sidecar_v6(_sidecar(rows))
    assert needle in str(e.value), str(e.value)[:200]


def test_a_sid_less_sidecar_names_the_CAUSE_not_the_symptom():
    """⛔⛔ THE ORDERING DEFECT, MEASURED 2026-09-17 while wiring this and
    fixed here. On a clip_id-keyed sidecar the reader used to report *"NOT ONE
    of the sidecar's 147 rows carries a valid ceiling"* — which was FALSE: all
    147 carried one, and were skipped for having no ``sid``. A true-sounding
    message naming a SYMPTOM as the CAUSE sends the reader to rebuild the
    labels instead of re-keying the sidecar.

    ⭐ THE DISCRIMINATING HALF is the second assertion: asserting only that
    the message mentions ``sid`` would pass on the OLD message too if it
    happened to mention it in passing. The symptom must be ABSENT."""
    rows = [{k: v for k, v in _row(i).items() if k != "sid"}
            for i in range(3)]
    with pytest.raises(SystemExit) as e:
        v6ms.read_speed_max_sidecar_v6(_sidecar(rows))
    msg = str(e.value)
    assert "`sid`" in msg
    assert "valid ceiling" not in msg


def test_RED_a_sidecar_built_over_a_different_blob_refuses():
    """Two quantizations of one corpus is two experiments, and the failure is
    SILENT: a mismatched sidecar joins perfectly and feeds wrong ceilings."""
    p = _sidecar([_row(1)])
    Path(p + ".meta.json").write_text(
        json.dumps({"source_md5": "0" * 32}), encoding="utf-8")
    with pytest.raises(SystemExit) as e:
        v6ms.read_speed_max_sidecar_v6(p, label_md5="f" * 32)
    assert "md5" in str(e.value)
    # ⭐ the GREEN control: the SAME file with the matching md5 reads.
    by_sid, _ = v6ms.read_speed_max_sidecar_v6(p, label_md5="0" * 32)
    assert by_sid[1][1] == 1.0


# =========================================================================== #
# 4. THE EXHAUSTIVE AUDIT SEES THE NEW WEIGHT                                 #
# =========================================================================== #
def test_the_new_weight_has_a_gate_row_and_the_gate_discriminates():
    """⛔ `REFC_WEIGHT_GATES` ENUMERATES DECLARED WEIGHTS and is therefore
    STRUCTURALLY BLIND to a head that has none — which is exactly how
    `tac_goal_tok_head` stayed invisible while 11,286 params learned nothing.
    Giving this head a real weight flag is part of the fix, not packaging."""
    assert "w_tac_v6" in T.REFC_WEIGHT_GATES
    g = T.REFC_WEIGHT_GATES["w_tac_v6"]["gate"]
    met, why = g(T.build_parser().parse_args(BASE + TAC))
    assert met is True, why
    # ⭐ THE MUTATION: remove each precondition in turn; the gate must close.
    for flag, n in (("--tac-decoder-v6", 0), ("--v7-labels", 1)):
        met2, why2 = g(T.build_parser().parse_args(
            _drop(BASE + TAC, flag, n)))
        assert met2 is False, f"gate stayed open without {flag}"
        assert why2


def test_effective_weight_row_reads_TRAINS_only_when_it_can_train():
    rows, _src = T.effective_weight_rows_v3(
        T.build_parser().parse_args(BASE + TAC))
    row = [r for r in rows if r.flag == "--w-tac-v6"][0]
    assert row.builds_graph is True
    # the mutation: same weight, no decoder -> must NOT build a graph
    rows2, _ = T.effective_weight_rows_v3(T.build_parser().parse_args(
        _drop(BASE + TAC, "--tac-decoder-v6", 0)))
    row2 = [r for r in rows2 if r.flag == "--w-tac-v6"][0]
    assert row2.builds_graph is False


# =========================================================================== #
# 5. THE DEFAULT PATH IS UNTOUCHED                                            #
# =========================================================================== #
def test_the_defaults_are_off_and_zero_as_LITERALS():
    """⛔ LITERALS, never an expression over the code under test — that is the
    green-forever defect. These four values are what make the no-flag path
    bit-identical to the pre-wiring trainer."""
    a = T.build_parser().parse_args(BASE)
    assert a.tac_decoder_v6 is False
    assert a.w_tac_v6 == 0.0
    assert a.max_speed_input_v6 is False
    assert a.tac_decoder_d_bev == 0
    assert a.graft_behaviour_sel is False


def test_no_flags_builds_no_decoder_and_stamps_nulls():
    args = T.build_parser().parse_args(BASE)
    cfg = v3.RefCV3Config(hier=True)
    T._pin_trainer_cfg(cfg, args)
    assert getattr(cfg, "tac_decoder_v6", False) is False
    assert getattr(cfg, "max_speed_onehot_v6", False) is False
    s = T._seam_stamp(cfg, args)["tac_decoder_v6"]
    assert s["requested"] is False and s["cfg"] is False
    assert s["decoder_cfg"] is None and s["sources"] is None


def test_the_loss_weights_stay_inside_the_MANEUVER_WEIGHT_budget():
    """⛔ *"inside the existing MANEUVER_WEIGHT budget (do not inflate the
    total)"*. Written as LITERALS: 0.05 + 0.025 + 0.025 == 0.10."""
    w = v6tac.TacticalLossWeights()
    assert w.goal_bce == 0.05 and w.lat_ce == 0.025 and w.lon_ce == 0.025
    assert w.total() == pytest.approx(0.10)
    w.assert_within_budget()
    # the mutation: inflate it, and the guard must go RED.
    with pytest.raises(ValueError):
        v6tac.TacticalLossWeights(goal_bce=0.5).assert_within_budget()
