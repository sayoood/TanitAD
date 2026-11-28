"""D-ROLL-1h — the `--tac-goal-tok-head` TRAINER FLAG: default-off, two-sided,
reached on every launch path, and stamped into the run record.

⭐ WHY THIS FILE EXISTS SEPARATELY FROM ``test_refc_v3_rollability.py``. The
sibling pinned the MODEL-side gate (``RefCV3Config.tac_goal_tok_head``, default
OFF) and, with it, the rollability of every banked checkpoint. That fix left the
head switchable only PROGRAMMATICALLY — no CLI flag — so ``D-TACGOAL-1``'s arm
could not be launched at all. This file pins the other half: the flag, the pin
in ``_pin_trainer_cfg``, and the ``config.json`` stamp.

⛔ THE DEFAULT IS LOAD-BEARING, NOT A STYLE CHOICE. MEASURED 2026-09-06: the
live refcv5 A40 run rebuilds through this same trainer on every supervisor
relaunch, so a default that moved would have made a training IN FLIGHT
unresumable. Every test below therefore asserts the OFF state as hard as the ON
state, and :func:`test_the_flag_is_two_sided_on_the_real_recorded_argv` reads
the ledger in BOTH positions — a flag that reads the same both ways has measured
nothing, which is the defect class this whole night was about.

⭐ CORRECTNESS AND WIRING ARE DIFFERENT CLAIMS, and only the first is usually
tested. MEASURED the same night: ``main`` runs ``preflight`` ONLY under
``--preflight`` and otherwise calls ``train()`` directly, so a lever wired into
one of them covers one launch path of two.
:func:`test_the_flag_reaches_every_launch_path` asserts BOTH paths structurally
(AST over ``main``'s dispatch and over each entry point's config production) and
empirically (a counter on the head's ``__init__``, with the same-breath control
that the counter can move at all).

⛔ THE STAMP IS NOT A SECOND COPY OF THE BUILD CONDITION. The root-cause class
named in the brief: ``refcv3_arm``'s ``a_star`` comment claimed it computed the
value *"exactly as the trainer computes it"* — true when written, never updated
when the trainer was corrected, and it cost a contaminated metric family. So the
record carries ``requested`` (argv), ``cfg`` (the pin) and ``built`` (read off
the CONSTRUCTED MODULE) as three separate facts, and never re-derives
``_vv != "kin3" and cfg.tac_goal_tok_head``.
:func:`test_the_stamp_never_re_derives_the_build_condition` is what fails if
anyone adds that second copy.
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import io
import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))          # <repo>/stack/tests
_STACK = os.path.dirname(_HERE)                             # <repo>/stack
if _STACK not in sys.path:
    sys.path.insert(0, _STACK)

from tanitad.refs import refc_v3 as v3                      # noqa: E402

_TRAINER_PY = os.path.join(_STACK, "scripts", "refc_v3_train.py")

#: The REAL recorded ``argv`` of the arm the regression made unrollable, copied
#: from ``C:\\Users\\Admin\\refcv4b_final\\config.json`` (banked by the sibling in
#: ``Research/2026-09-06-rollability/RESULT.md`` section 1.1). ⛔ It cannot
#: mention ``--tac-goal-tok-head`` — the flag did not exist when it was written,
#: which is precisely why a default-OFF switch is the only shape that lets this
#: record rebuild.
REFCV4B_ARGV = ["--arm", "hier", "--size", "base", "--v2-cache", "/x/tr",
                "--v7-labels", "/x/tr.jsonl.gz", "--image-hw", "256", "640",
                "--n-anchors", "117", "--anchor-v0-conditioned",
                "--anchor-control-units", "alat", "--sel-accel-max", "2.0",
                "--goal-str", "--ego-state-inject", "--ego-dropout", "0.5",
                "--nav-from-v7", "--out", "/x/o"]
REFCV4B_TOTAL = 107058488          # the recorded `param_breakdown['total']`


def _trainer():
    """``refc_v3_train.py`` by PATH — it is a SCRIPT, and ``build_parser`` +
    ``_pin_trainer_cfg`` are the only way a run's config is recoverable. This
    is ``refcv3_arm``'s own convention, reused rather than copied."""
    if not os.path.exists(_TRAINER_PY):
        pytest.skip(f"trainer not present at {_TRAINER_PY}")
    spec = importlib.util.spec_from_file_location("refc_v3_train_for_tacflag",
                                                  _TRAINER_PY)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["refc_v3_train_for_tacflag"] = mod
    spec.loader.exec_module(mod)
    return mod


def _src():
    """The trainer's source, with the READ asserted.

    ⛔ 0 hits from a file that could not be OPENED is a claim about the search,
    not about the content — the G: mount serves metadata while content reads
    fail. A non-empty read is the same-breath control."""
    src = io.open(_TRAINER_PY, encoding="utf-8").read()
    assert len(src) > 100_000, \
        f"trainer source read back only {len(src)} chars -- INCONCLUSIVE, not " \
        f"a negative result"
    return src


def _cfg(tr, argv, size_cfg=True):
    args = tr.build_parser().parse_args(list(argv))
    base = (v3.refc_v3_sized_config(args.size, hier=(args.arm == "hier"))
            if size_cfg else v3.refc_v3_smoke_config(args.arm == "hier"))
    return tr._pin_trainer_cfg(base, args), args


# ==========================================================================
# 1 — THE FLAG EXISTS, AND ITS DEFAULT IS THE ONE THE LIVE RUN NEEDS
# ==========================================================================
def test_the_flag_exists_and_defaults_off():
    """⛔ Without this flag ``D-TACGOAL-1``'s arm cannot be launched at all; with
    the wrong default the live refcv5 run cannot resume. Both halves here."""
    tr = _trainer()
    ap = tr.build_parser()
    act = [a for a in ap._actions if "--tac-goal-tok-head" in a.option_strings]
    assert len(act) == 1, \
        f"expected exactly one --tac-goal-tok-head action, got {len(act)}"
    a = act[0]
    assert a.dest == "tac_goal_tok_head", a.dest
    assert a.nargs == 0, "must be a switch, not a value-taking option"
    assert a.default is False, \
        f"--tac-goal-tok-head default is {a.default!r}; a default that is not " \
        f"False makes every banked checkpoint unrollable (D-ROLL-1)"
    # the same-breath control: the parser really is the trainer's, and it still
    # knows a flag that predates this one.
    assert any("--v7-labels" in x.option_strings for x in ap._actions)


def test_a_pre_flag_recorded_argv_parses_to_off():
    """⭐ THE RECORD CANNOT MENTION A FLAG THAT DID NOT EXIST. The real
    refcv4b/refcv5 ``argv`` is the input this default has to survive."""
    tr = _trainer()
    args = tr.build_parser().parse_args(list(REFCV4B_ARGV))
    assert args.tac_goal_tok_head is False
    assert "--tac-goal-tok-head" not in REFCV4B_ARGV      # control on the input


# ==========================================================================
# 2 — TWO-SIDED: THE PIN, THE MODULE AND THE LEDGER MOVE TOGETHER
# ==========================================================================
def test_the_flag_is_two_sided_on_the_real_recorded_argv():
    """⛔⛔ THE MUTATION. The SAME recorded argv, once as recorded and once with
    the flag appended — the ledger line must APPEAR and DISAPPEAR, and the
    delta must be exactly the head's 11,286 parameters.

    ⚠ A flag that reads the same in both positions has measured nothing. Both
    directions are asserted here, in one test, so neither can be skipped."""
    tr = _trainer()

    cfg_off, _ = _cfg(tr, REFCV4B_ARGV)
    assert cfg_off.tac_vocab_version == "v7.0", "fixture must exercise v7"
    assert cfg_off.d_tac == 512, cfg_off.d_tac        # pins the 11,286 below
    assert cfg_off.tac_goal_tok_head is False
    m_off = v3.RefCV3Model(cfg_off)
    try:
        assert m_off.tac_goal_tok_head is None, \
            "the head was built without being asked for -- D-ROLL-1 exactly"
        bd_off = {k: int(v) for k, v in v3.param_breakdown_v3(m_off).items()}
    finally:
        del m_off
    assert "tac_goal_tok_head" not in bd_off
    assert bd_off["total"] == REFCV4B_TOTAL, (
        f"the DEFAULT rebuild no longer reproduces refcv4b's recorded total: "
        f"{bd_off['total']} != {REFCV4B_TOTAL}. That is an unrollable "
        f"checkpoint, measured without the 1.3 GB weights.")

    cfg_on, _ = _cfg(tr, list(REFCV4B_ARGV) + ["--tac-goal-tok-head"])
    assert cfg_on.tac_goal_tok_head is True, \
        "the flag did not reach the config -- the pin is not wired"
    m_on = v3.RefCV3Model(cfg_on)
    try:
        assert m_on.tac_goal_tok_head is not None, \
            "the flag is stamped but the head is NOT BUILT -- an inert lever"
        bd_on = {k: int(v) for k, v in v3.param_breakdown_v3(m_on).items()}
    finally:
        del m_on
    want = 22 * (cfg_on.d_tac + 1)                    # DERIVED, never typed
    assert want == 11286, want
    assert bd_on.get("tac_goal_tok_head") == want, bd_on.get("tac_goal_tok_head")
    assert bd_on["total"] - bd_off["total"] == want, (
        bd_on["total"], bd_off["total"])
    # the ledger still sums exactly in BOTH states: an unaccounted module is a
    # capacity confound inside the very claim the arm exists to test.
    for bd in (bd_off, bd_on):
        assert sum(v for k, v in bd.items() if k != "total") == bd["total"]
    # ...and the ONLY difference between the two ledgers is that one line: the
    # flag must not move any other capacity, or the arm's cost is not
    # attributable to the head.
    assert set(bd_on) - set(bd_off) == {"tac_goal_tok_head"}
    assert set(bd_off) - set(bd_on) == set()
    rest_on = {k: v for k, v in bd_on.items()
               if k not in ("tac_goal_tok_head", "total")}
    rest_off = {k: v for k, v in bd_off.items() if k != "total"}
    assert rest_on == rest_off, (
        f"the flag moved capacity outside its own head: "
        f"{ {k: (rest_off.get(k), rest_on.get(k)) for k in set(rest_on) | set(rest_off) if rest_on.get(k) != rest_off.get(k)} }")


def test_exactly_one_head_is_built_when_the_flag_is_on():
    """⭐ 'Built' is not enough — built ONCE. A second construction site would
    double the capacity cost and make the arm's ledger unattributable."""
    tr = _trainer()
    from tanitad.refs import tac_goal_head as tg
    real_init, calls = tg.TacGoalTokenHead.__init__, []

    def counting(self, *a, **k):
        calls.append(1)
        return real_init(self, *a, **k)

    tg.TacGoalTokenHead.__init__ = counting
    try:
        cfg_off, _ = _cfg(tr, REFCV4B_ARGV, size_cfg=False)
        del_ = v3.RefCV3Model(cfg_off)
        n_off = len(calls)
        del del_
        cfg_on, _ = _cfg(tr, list(REFCV4B_ARGV) + ["--tac-goal-tok-head"],
                         size_cfg=False)
        del_ = v3.RefCV3Model(cfg_on)
        n_on = len(calls) - n_off
        del del_
    finally:
        tg.TacGoalTokenHead.__init__ = real_init
    assert n_off == 0, f"the flag was OFF and {n_off} head(s) were built"
    assert n_on == 1, f"the flag was ON and {n_on} head(s) were built, not 1"


def test_kin3_refuses_the_flag_rather_than_ignoring_it():
    """⛔ THE DEAD-FLAG CLASS. ``kin3`` is the 3x3 KINEMATIC derivation and has
    no tactical-goal token set, so ``refc_v3.py`` would build nothing and the
    operator would get an arm that silently did not carry the lever it asked
    for. This trainer already refuses four flags for exactly that reason.

    ⚠ The vocabulary stays NECESSARY — this refusal does not loosen it."""
    tr = _trainer()
    argv_kin3 = ["--arm", "hier", "--size", "base", "--out", "/x/o",
                 "--tac-goal-tok-head"]            # no --v7-labels => kin3
    with pytest.raises(SystemExit) as ex:
        _cfg(tr, argv_kin3)
    assert "kin3" in str(ex.value), str(ex.value)[:300]
    # SAME-BREATH CONTROL: the identical flag under the v7 vocabulary must NOT
    # refuse. A guard that refuses everything is not a guard.
    cfg, _ = _cfg(tr, list(REFCV4B_ARGV) + ["--tac-goal-tok-head"])
    assert cfg.tac_vocab_version == "v7.0" and cfg.tac_goal_tok_head is True
    # ...and the refusal message is ASCII: a SystemExit is written to stderr,
    # and non-ASCII is fatal on the cp1252 dev box, so a guard that crashes on
    # its own refusal text has refused nothing legible.
    str(ex.value).encode("ascii")


# ==========================================================================
# 3 — REACHABILITY: the flag is on EVERY launch path, not just one
# ==========================================================================
def test_the_flag_reaches_every_launch_path():
    """⭐⭐ WIRING, NOT CORRECTNESS. ``main`` dispatches to exactly two entry
    points and both must reach the pin; a lever wired into ``preflight`` alone
    covers one launch path of two, which is MEASURED to be how this trainer is
    shaped."""
    tr = _trainer()
    tree = ast.parse(_src())

    fns = {n.name: n for n in ast.walk(tree)
           if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    assert {"main", "preflight", "train", "_pin_trainer_cfg"} <= set(fns)

    # (a) main dispatches to preflight and train, and to no THIRD entry point.
    called = {getattr(n.func, "id", None)
              for n in ast.walk(fns["main"]) if isinstance(n, ast.Call)}
    assert {"preflight", "train"} <= called, sorted(x for x in called if x)
    # every other call in `main` must not itself produce a trainer config
    for name in called:
        if name in (None, "preflight", "train") or name not in fns:
            continue
        inner = {getattr(c.func, "id", None) or getattr(c.func, "attr", None)
                 for c in ast.walk(fns[name]) if isinstance(c, ast.Call)}
        assert "RefCV3Model" not in inner, (
            f"main() reaches {name}(), which builds a model outside the two "
            f"known launch paths -- a third path the flag would not cover")

    # (b) BOTH entry points pass the parsed `args` to the ONE config helper.
    for entry in ("preflight", "train"):
        pins = [c for c in ast.walk(fns[entry]) if isinstance(c, ast.Call)
                and getattr(c.func, "id", None) == "_pin_trainer_cfg"]
        assert pins, f"{entry}() never calls _pin_trainer_cfg"
        for c in pins:
            assert len(c.args) == 2 and getattr(c.args[1], "id", None) == "args", (
                f"{entry}() calls _pin_trainer_cfg without passing `args` -- "
                f"the flag could not reach the pin on this path")

    # (c) the pin reads the flag off `args`, inside _pin_trainer_cfg.
    pin_src = ast.get_source_segment(_src(), fns["_pin_trainer_cfg"]) or ""
    assert 'getattr(args, "tac_goal_tok_head", False)' in pin_src, \
        "the pin does not read the flag from args"
    assert "cfg.tac_goal_tok_head = True" in pin_src, \
        "the pin does not put the flag on the config"

    # (d) EMPIRICAL, on a config the helper actually produced from a real argv.
    from tanitad.refs import tac_goal_head as tg
    real_init, calls = tg.TacGoalTokenHead.__init__, []

    def counting(self, *a, **k):
        calls.append(1)
        return real_init(self, *a, **k)

    tg.TacGoalTokenHead.__init__ = counting
    try:
        cfg, _ = _cfg(tr, REFCV4B_ARGV, size_cfg=False)
        m = v3.RefCV3Model(cfg)
        n_default = len(calls)
        del m
        cfg_on, _ = _cfg(tr, list(REFCV4B_ARGV) + ["--tac-goal-tok-head"],
                         size_cfg=False)
        m = v3.RefCV3Model(cfg_on)
        n_flag = len(calls) - n_default
        del m
    finally:
        tg.TacGoalTokenHead.__init__ = real_init
    assert n_default == 0, "the default path built the head"
    assert n_flag == 1, (
        f"the FLAG is not reached: it built {n_flag} heads, not 1. A counter "
        f"that never moves would report 'not reached' for a gate that is "
        f"simply never exercised -- this half is that control.")


# ==========================================================================
# 4 — THE RUN RECORD: three separate facts, checked against the MODEL
# ==========================================================================
def _stamp(tr, argv):
    cfg, args = _cfg(tr, argv, size_cfg=False)
    return tr._seam_stamp(cfg, args), cfg, args


def test_the_stamp_carries_requested_cfg_and_built_separately():
    """⛔ A run that cannot state its own knobs is unauditable (M18). Here the
    knob is structural, so the record carries what argv ASKED, what the pin
    PUT ON THE CONFIG, and — filled from the constructed module — what the
    model HAS."""
    tr = _trainer()
    for argv, want_req in ((REFCV4B_ARGV, False),
                           (list(REFCV4B_ARGV) + ["--tac-goal-tok-head"], True)):
        st, cfg, _ = _stamp(tr, argv)
        blk = st.get("tac_goal_tok_head")
        assert isinstance(blk, dict), "no tac_goal_tok_head block in the stamp"
        assert set(blk) == {"requested", "cfg", "tac_vocab_version", "built"}
        assert blk["requested"] is want_req
        assert blk["cfg"] is want_req
        assert blk["tac_vocab_version"] == "v7.0"
        assert blk["built"] is None, \
            "`built` must be UNSET until the MODEL fills it -- a stamp that " \
            "guesses the build from the config is the stale-consumer defect"


def test_the_stamp_never_re_derives_the_build_condition():
    """⛔ THE STALE-CONSUMER GUARD. ``refcv3_arm``'s ``a_star`` comment claimed
    to compute a value 'exactly as the trainer computes it'; the trainer changed
    and the copy did not, and it cost a contaminated metric family. So the
    stamp must READ the built object, never re-derive
    ``_vv != "kin3" and cfg.tac_goal_tok_head``."""
    src = _src()
    tree = ast.parse(src)
    fns = {n.name: n for n in ast.walk(tree)
           if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    # ⚠ ASSERT ON THE CODE, NOT ON THE FILE. A raw-source grep also reads
    # the COMMENTS, and the comment that explains why the condition must not
    # be copied necessarily NAMES the condition -- so the first version of
    # this test failed on its own rationale. `ast.unparse` drops comments and
    # keeps executable code, which is the thing that can actually drift.
    seam_src = ast.get_source_segment(src, fns["_seam_stamp"]) or ""
    assert '"tac_goal_tok_head"' in seam_src, "the stamp lost its block"
    seam_code = ast.unparse(fns["_seam_stamp"])
    assert "kin3" not in seam_code, (
        "`_seam_stamp` re-derives the vocabulary half of the build condition. "
        "That is a SECOND copy of a build rule, which is exactly how the "
        "a_star comment drifted. Read the built object instead.")
    # the same-breath control: the discriminator IS present in the code that
    # is ALLOWED to hold it, so a kin3-free source tree cannot pass vacuously.
    assert "kin3" in ast.unparse(fns["_pin_trainer_cfg"]), \
        "no kin3 in the pin either -- this probe is not discriminating"
    # ...and `built` is assigned from the MODEL, in `train`.
    train_src = ast.get_source_segment(src, fns["train"]) or ""
    assert '_seams["tac_goal_tok_head"]["built"] = (' in train_src, \
        "`built` is never filled in from the constructed model"
    assert 'getattr(model, "tac_goal_tok_head", None) is not None' in train_src


def test_assert_seams_are_built_refuses_a_lying_record_both_ways():
    """⛔⛔ BIDIRECTIONAL, and it is the check whose ABSENCE was D-ROLL-1:
    11,286 params in the model that no record names makes a checkpoint
    unloadable, and 11,286 params in the record that the weights lack is the
    mirror image. A gate never shown to fail the defect certifies nothing."""
    tr = _trainer()

    # --- the honest pairs pass -------------------------------------------- #
    st_off, cfg_off, _ = _stamp(tr, REFCV4B_ARGV)
    m_off = v3.RefCV3Model(cfg_off)
    st_off["tac_goal_tok_head"]["built"] = m_off.tac_goal_tok_head is not None
    assert st_off["tac_goal_tok_head"]["built"] is False
    tr.assert_seams_are_built(m_off, st_off)                    # green first

    st_on, cfg_on, _ = _stamp(tr, list(REFCV4B_ARGV) + ["--tac-goal-tok-head"])
    m_on = v3.RefCV3Model(cfg_on)
    st_on["tac_goal_tok_head"]["built"] = m_on.tac_goal_tok_head is not None
    assert st_on["tac_goal_tok_head"]["built"] is True
    tr.assert_seams_are_built(m_on, st_on)                      # green first

    # --- (1) the record claims a head the weights do not contain ---------- #
    lying = dict(st_off)
    lying["tac_goal_tok_head"] = dict(st_off["tac_goal_tok_head"],
                                      built=True, cfg=True)
    with pytest.raises(SystemExit) as e1:
        tr.assert_seams_are_built(m_off, lying)
    assert "tac_goal_tok_head" in str(e1.value)

    # --- (2) a LIVE head absent from the record --------------------------- #
    silent = dict(st_on)
    silent["tac_goal_tok_head"] = dict(st_on["tac_goal_tok_head"],
                                       built=False, cfg=False)
    with pytest.raises(SystemExit) as e2:
        tr.assert_seams_are_built(m_on, silent)
    assert "tac_goal_tok_head" in str(e2.value)

    # --- (3) no block at all, with the head built ------------------------- #
    gone = {k: v for k, v in st_on.items() if k != "tac_goal_tok_head"}
    with pytest.raises(SystemExit) as e3:
        tr.assert_seams_are_built(m_on, gone)
    assert "tac_goal_tok_head" in str(e3.value)
    del m_off, m_on


def test_the_flag_is_outside_the_agent_knob_stamp_and_the_weight_audit():
    """⚠ SCOPE CONTROL. This lever is STRUCTURAL, not a weight: it must not
    enter ``agent_knob_dests`` (which is derived from ``--agent-*``/``--w-*``)
    and it must not appear in the effective-weight audit. ⛔ Five weights here
    default to 0.0 and are hard-guarded; this change touches none of them."""
    tr = _trainer()
    p = tr.build_parser()
    assert "tac_goal_tok_head" not in tr.agent_knob_dests(p)
    assert "tac_goal_tok_head" not in tr.REFC_WEIGHT_GATES
    # control: the derivation is alive and still names the knobs it should
    assert "w_agent" in tr.agent_knob_dests(p)
    # the weight audit is unchanged in both flag positions
    for extra in ([], ["--tac-goal-tok-head"]):
        args = p.parse_args(list(REFCV4B_ARGV) + extra)
        args._ew_parser, args._ew_argv = p, list(REFCV4B_ARGV) + extra
        rows, _ = tr.effective_weight_rows_v3(args)
        assert all(getattr(r, "dest", None) != "tac_goal_tok_head"
                   for r in rows)
        tr.check_effective_weights(args)          # must not refuse either way


# ==========================================================================
# 6 — --w-tac-goal: THE TARGET REACHES THE BATCH FROM THE REAL LOADER
# ==========================================================================
#: ⛔ The tokens the v7.2 blob emits from GEOMETRY, so an ABSENT one is a real
#: negative. Pinned as a LITERAL here exactly as ``load_v7_labels`` pins it
#: from the blob it read -- never imported from the code under test.
_GEOM = frozenset({"FOLLOW_LANE", "SPEED_BAND", "STOP_POINT", "TURN_L",
                   "TURN_R", "YIELD_FOR_TURN_L", "YIELD_FOR_TURN_R"})


def _labelled_ds(tr, monkeypatch, *, enabled: bool,
                 negatives: str = "measured"):
    """A real ``V3Dataset`` over synthetic episodes with a v7.2 join.

    ⭐ Episode 9000 CARRIES a record; episode 9001 deliberately does NOT. The
    second one is the point: a clip with no record must still get both keys, as
    an explicitly all-ignored row. A batch that sometimes carries the key and
    sometimes does not would make the loss-time refusal fire at random instead
    of at launch.
    """
    import torch
    from tanitad.data import v7_labels as v7l
    monkeypatch.setattr(v7l, "_MEASURED_GEOMETRY_TOKENS", _GEOM)
    cfg = v3.refc_v3_smoke_config(True)
    eps = tr._synth_episodes(2, cfg.core, seed=0)
    for i, e in enumerate(eps):
        e.episode_id = str(9000 + i)          # the loader does int(episode_id)
    ds = tr.V3Dataset(eps, window=cfg.core.window, max_horizon=20,
                      channels=cfg.core.encoder.in_channels)
    lab = v7l.V7Label(
        clip_id="clip_9000", tac_lat="LANE_KEEP", tac_lon="CRUISE",
        str_action="HOLD_MAIN_ROAD", str_goal="HOLD_MAIN_ROAD",
        tac_anchor=None, bands={"tactical_s": [0.0, 60.0]}, t0_s=0.3,
        horizon={}, tac_goals=frozenset({"FOLLOW_LANE", "SPEED_BAND"}),
        tac_goal_meta={"FOLLOW_LANE": {"provenance": "geometry"},
                       "SPEED_BAND": {"provenance": "geometry"}})
    ds.v7_by_sid = {9000: lab}                # 9001 is ABSENT on purpose
    ds.v7_dt = 0.1
    ds.tac_goal_targets = enabled
    ds.tac_goal_negatives = negatives
    first = {e_i: i for i, (e_i, _t) in reversed(list(enumerate(ds.index)))}
    return ds, first, v7l, torch


def test_the_dataset_emits_no_goal_target_when_the_channel_is_OFF(monkeypatch):
    """⛔ THE BIT-IDENTITY PRECONDITION AT THE LOADER. Default OFF must add no
    key at all -- an extra key would change the collated batch of a live
    recipe that never asked for the term."""
    tr = _trainer()
    ds, first, _v7l, _torch = _labelled_ds(tr, monkeypatch, enabled=False)
    item = ds[first[0]]
    assert "tac_goal_y" not in item and "tac_goal_w" not in item, (
        "the loader emitted a goal-set target on an arm that did not ask for "
        "it -- the OFF path is no longer bit-identical")
    # control, same breath: the join IS live, so the absence above is the
    # CHANNEL being off and not the label being missing
    assert "lat_v7" in item, "the v7 join is not live -- the test proves nothing"


def test_the_dataset_emits_the_goal_target_when_the_channel_is_ON(monkeypatch):
    """⭐ The real ``__getitem__`` path -- the one the gradient probes bypass."""
    tr = _trainer()
    ds, first, v7l, torch = _labelled_ds(tr, monkeypatch, enabled=True)
    item = ds[first[0]]
    assert "tac_goal_y" in item and "tac_goal_w" in item, (
        "the loader did NOT emit the goal-set target with the channel on -- "
        "`--w-tac-goal` would refuse at loss time on the real corpus")
    assert tuple(item["tac_goal_y"].shape) == (22,), item["tac_goal_y"].shape
    assert tuple(item["tac_goal_w"].shape) == (22,), item["tac_goal_w"].shape
    assert item["tac_goal_y"].dtype == torch.float32
    ix = {t: i for i, t in enumerate(v7l.TAC_GOAL_TOKENS)}
    # the two tokens this record carries are POSITIVE and supervised
    for tok in ("FOLLOW_LANE", "SPEED_BAND"):
        assert float(item["tac_goal_y"][ix[tok]]) == 1.0, tok
        assert float(item["tac_goal_w"][ix[tok]]) == 1.0, tok
    # a geometry token this record does NOT carry is a supervised NEGATIVE
    assert float(item["tac_goal_y"][ix["TURN_L"]]) == 0.0
    assert float(item["tac_goal_w"][ix["TURN_L"]]) == 1.0
    # ⛔ and a CoT-only token is IGNORED, never supervised as a negative:
    # 3,574 of 4,572 clips were never asked the traffic-light question.
    assert float(item["tac_goal_w"][ix["TRAFFIC_LIGHT_REACT_RED"]]) == 0.0


def test_a_clip_with_NO_record_still_gets_both_keys_all_ignored(monkeypatch):
    """⛔ NEVER A MISSING KEY. Episode 9001 has no v7.2 record; it must still
    carry both tensors, with every cell explicitly ignored."""
    tr = _trainer()
    ds, first, _v7l, _torch = _labelled_ds(tr, monkeypatch, enabled=True)
    item = ds[first[1]]
    assert "tac_goal_y" in item and "tac_goal_w" in item, (
        "a clip with no v7.2 record dropped the key entirely -- the batch "
        "would be ragged and the loss-time refusal would fire at random")
    assert float(item["tac_goal_w"].sum()) == 0.0, (
        "a clip with NO record supervised something -- an unmapped episode "
        "must train nothing, not train a default")
    assert float(item["tac_goal_y"].sum()) == 0.0


def test_the_negatives_policy_is_READ_and_not_hardcoded(monkeypatch):
    """⭐ THE DISCRIMINATING CONTROL for `--tac-goal-negatives`.

    A knob that is parsed and stamped but never reaches the derivation is the
    `--wp-index` defect (3 of 6 knobs inert). Under ``all`` the CoT-only
    traffic-light cell MUST become supervised; under ``measured`` it must not.
    If both read the same, the knob is inert.
    """
    tr = _trainer()
    ds_m, first, v7l, _t = _labelled_ds(tr, monkeypatch, enabled=True,
                                        negatives="measured")
    ds_a, first_a, _v, _t2 = _labelled_ds(tr, monkeypatch, enabled=True,
                                          negatives="all")
    ix = {t: i for i, t in enumerate(v7l.TAC_GOAL_TOKENS)}
    j = ix["TRAFFIC_LIGHT_REACT_RED"]
    w_measured = float(ds_m[first[0]]["tac_goal_w"][j])
    w_all = float(ds_a[first_a[0]]["tac_goal_w"][j])
    assert w_measured == 0.0, (
        f"--tac-goal-negatives=measured supervised a CoT-only cell "
        f"(w={w_measured}); 3,574 of 4,572 clips were never asked the "
        f"traffic-light question, so that is training on no evidence")
    assert w_all == 1.0, (
        f"--tac-goal-negatives=all left the CoT-only cell ignored (w={w_all}) "
        f"-- the knob is PARSED, STAMPED AND INERT, which is the --wp-index "
        f"defect verbatim")


# ==========================================================================
# 7 — THE SPLIT-FITTED CONSTANTS: every key train() reads, as a LITERAL
# ==========================================================================
def _three_labels(monkeypatch):
    from tanitad.data import v7_labels as v7l
    monkeypatch.setattr(v7l, "_MEASURED_GEOMETRY_TOKENS", _GEOM)

    def _lab(clip, goals):
        return v7l.V7Label(
            clip_id=clip, tac_lat="LANE_KEEP", tac_lon="CRUISE",
            str_action="HOLD_MAIN_ROAD", str_goal="HOLD_MAIN_ROAD",
            tac_anchor=None, bands={"tactical_s": [0.0, 60.0]}, t0_s=8.0,
            horizon={}, tac_goals=frozenset(goals),
            tac_goal_meta={k: {"provenance": "geometry"} for k in goals})

    return v7l, [_lab("c1", ("FOLLOW_LANE", "SPEED_BAND")),
                 _lab("c2", ("TURN_L", "SPEED_BAND")),
                 _lab("c3", ("FOLLOW_LANE",))]


def test_train_fits_pos_weight_and_mask_from_the_split_not_a_literal(
        monkeypatch):
    """⛔ EVERY KEY `train()` READS, ASSERTED AS A LITERAL.

    `mask_report` must return exactly these five keys and `goal_pos_weight` a
    22-tuple, because ``train()`` indexes them by name. ⭐ The expectation is
    written out, never derived by calling the same functions and comparing them
    to themselves — that would measure determinism, not correctness.
    """
    import torch
    from tanitad.refs import tac_goal_head as tgh
    v7l, labels = _three_labels(monkeypatch)

    census = v7l.goal_supervision_census(labels)
    mask = tgh.mask_report(census)
    pw = v7l.goal_pos_weight(labels)

    assert len(census) == 22
    assert set(mask) == {"mask", "masked_why", "n_total", "n_trainable",
                         "trainable"}, sorted(mask)
    assert int(mask["n_total"]) == 22
    assert len(pw) == 22
    # exactly the two conversions train() performs
    pw_t = torch.tensor(pw, dtype=torch.float32)
    mask_t = torch.tensor(mask["mask"], dtype=torch.float32)
    assert tuple(pw_t.shape) == (22,) and pw_t.dtype == torch.float32
    assert tuple(mask_t.shape) == (22,) and mask_t.dtype == torch.float32
    # and the stamp train() writes into config.json must be JSON-serialisable:
    # a run record that cannot be written is a run record that does not exist
    import json
    json.dumps({"negatives": "measured",
                "n_trainable": int(mask["n_trainable"]),
                "n_total": int(mask["n_total"]),
                "trainable": list(mask["trainable"]),
                "masked_why": mask["masked_why"],
                "pos_weight": [float(x) for x in pw],
                "census": census})


def test_the_loss_accepts_those_two_tensors_and_a_gradient_flows(monkeypatch):
    """⭐ AN ANALYTIC TARGET, not a recorded number.

    With every logit at 0 and every target 0, BCE-with-logits is exactly
    ``-log(1 - sigmoid(0)) = log 2 = 0.6931471805599453`` per supervised cell,
    and the weighted mean over supervised cells is that same value whatever the
    mask is. ⛔ An analytic expectation is the strongest cross-check available
    here, because it is derived from the definition of the loss rather than from
    a previous run of this code.
    """
    import math
    import torch
    from tanitad.refs import tac_goal_head as tgh
    v7l, labels = _three_labels(monkeypatch)
    mask = tgh.mask_report(v7l.goal_supervision_census(labels))

    K = 22
    logits = torch.zeros(2, K, requires_grad=True)
    y = torch.zeros(2, K)
    w = torch.ones(2, K)
    loss, n_sup = tgh.tac_goal_loss(
        logits, y, w,
        pos_weight=torch.tensor(v7l.goal_pos_weight(labels),
                                dtype=torch.float32),
        class_mask=torch.tensor(mask["mask"], dtype=torch.float32))
    assert abs(float(loss) - math.log(2.0)) < 1e-6, (
        f"BCE at logit 0 with target 0 must be exactly log 2 "
        f"({math.log(2.0)}); got {float(loss)}")
    # n_supervised = (rows) x (classes the mask leaves live) -- a literal
    assert n_sup == 2 * int(mask["n_trainable"]), (
        f"n_supervised {n_sup} != 2 rows x {int(mask['n_trainable'])} "
        f"trainable classes -- the class_mask is not being applied")
    loss.backward()
    assert logits.grad is not None
    assert float(logits.grad.abs().sum()) > 0.0, (
        "the loss produced no gradient on its own logits")
