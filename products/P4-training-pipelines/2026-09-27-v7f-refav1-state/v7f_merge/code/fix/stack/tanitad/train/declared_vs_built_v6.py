"""G-DVB for the v6-STAGED trainer (v7F) — argv levers vs the BUILT ``V6Stack`` (and the loss weights in force).

⭐ UNION (TrainingFlyWheel merge, 2026-09-27). Two streams each wrote a NEW module at this path: the R1/R4 stream
(``--tac-op-cond``, ``--max-speed-input-v6``, ``--plan-vmax-cap``, ``--speed-max-sidecar-v6``) and the R3 stream
(``--w-tac-label-all``, ``--tac-goal-negatives``, ``--cot-negative-sidecar``) — same registry name, different reader
signatures, so a naive union raises ``TypeError`` in both directions. This file is the union:

* ONE registry (``REGISTRY_V6``), ONE reader signature ``(stack, args, weights_in_force)``; the R1/R4 readers take no
  weights and are wrapped;
* ``check`` / ``refuse_on_mismatch`` — the R1/R4 API, called at BUILD time (``build_stack_from_args``, before any
  loss weights exist) — run every lever that does NOT need the loss weights;
* ``check_v6`` / ``refuse_on_mismatch_v6`` — the R3 API, called where the stage's ``V6LossWeights`` are known —
  run EVERY lever (a superset), plus the parser-coverage check when a parser is given.

⛔⛔ WHAT THIS IS, AND WHAT IT IS NOT. ``tanitad.train.declared_vs_built`` is the binding G-DVB check of SPEC_REFCV7
§2, keyed on ``refc_v3_train.py``'s argparse dests and reading ``RefCV3Model`` internals, and its ``check()`` runs
EVERY registered lever against the model it is handed — so a v6 lever registered there would be run against a
RefCV3Model (``max_speed_input_v6``, ``tac_goal_negatives`` and ``cot_negative_sidecar`` already name refcv3 entries).
The launch gate has NO v6/v7F profile at ``c36b6ddd``. This registry is therefore SEPARATE and ADDITIVE, with the
SAME ``Lever`` / ``Mismatch`` types and the SAME rules:

* ``built`` / ``loss`` levers READ THE BUILT stack (and, for ``loss``, the weights ``v6_loss_step`` is handed) and
  compare with a LITERAL expectation taken from argv by a rule stated here — never by calling the trainer function
  under test (a cross-check derived from the code it checks measures determinism, not correctness);
* ``data`` / ``runtime`` / ``record`` levers carry a reason a reader can verify.

⚠️ SCOPE (updated by the launch-gate agent, 2026-09-27): the seven dests the two changes add run in the TRAINER
(``check`` at build, ``check_v6`` on the R3 path); every OTHER trainer dest -- the tip's 231 + R6's ``strategic_off``
-- has a GATE-TIME entry (``register_gate``, the ``GATE_TIME`` set) that only ``check_all`` runs: the v7f launch
gate calls it on the FULLY built and frozen stack, so ``uncovered_v6(parser) == []`` on the merged trainer and the
trainer's own behaviour is unchanged.

THE API::

    from tanitad.train import declared_vs_built_v6 as dvb6
    dvb6.check(stack, args)                     # build time: -> list[Mismatch]; [] == clean
    dvb6.refuse_on_mismatch(stack, args)        # SystemExit naming every mismatch
    dvb6.coverage(dests)                        # -> the given dests with no entry
    dvb6.check_v6(stack, args, weights_in_force=w_stage, parser=None)   # every lever
    dvb6.refuse_on_mismatch_v6(stack, args, weights_in_force=w_stage, parser=None, where="train")
    dvb6.coverage_v6(parser)                    # -> NEW_V6_DESTS missing from the parser or the registry
    dvb6.uncovered_v6(parser)                   # -> every parser dest with no v6 entry (the gate's gap)
    dvb6.check_all(stack, args, weights_in_force=w_stage, parser=parser)   # EVERY entry (the launch gate)
"""
from __future__ import annotations

from typing import Any, Callable

from tanitad.train.declared_vs_built import KINDS, Lever, Mismatch

__all__ = ["REGISTRY_V6", "R1R4_DESTS", "NEW_V6_DESTS", "ALL_V6_DESTS", "NEEDS_WEIGHTS",
           "register", "register_v6", "check", "refuse_on_mismatch", "coverage",
           "check_v6", "refuse_on_mismatch_v6", "coverage_v6", "uncovered_v6",
           "V7_GOAL_WIDTH", "V7_LAT_WIDTH", "V7_LON_WIDTH", "SPEED_BAND_ARG_SLOTS", "Mismatch",
           "GATE_TIME", "register_gate", "check_all", "kinds_census", "MODEL_CFG_LEVERS",
           "STAGE_LOSS_TERMS"]

REGISTRY_V6: dict[str, Lever] = {}
#: dests whose reader needs the loss weights in force -- not checkable at build time
NEEDS_WEIGHTS: set[str] = set()
#: dests checked only at GATE time (``check_all``), never inside the trainer -- see ``register_gate``
GATE_TIME: set[str] = set()

#: the dests the R1/R4 change adds (pinned by tests/test_v7f_r1r4.py through the real parser)
R1R4_DESTS: tuple[str, ...] = ("tac_op_cond", "max_speed_input_v6", "plan_vmax_cap", "speed_max_sidecar_v6")
#: the dests the R3 change adds (pinned by tests/test_tactical_label_reach_v6.py)
NEW_V6_DESTS: tuple[str, ...] = ("w_tac_label_all", "tac_goal_negatives", "cot_negative_sidecar")
ALL_V6_DESTS: tuple[str, ...] = R1R4_DESTS + NEW_V6_DESTS

#: LITERALS, written here -- never read from the module under test.
#: R1/R4: the refcv6 ladder has FOUR bins ({30, 50, 100, 120} km/h, PI 2026-09-16); the R4 port reads the
#: LAT x LON behaviour pair (2 x d_goal_embed wide) and writes one goal embedding.
_N_VMAX_BINS = 4
_TAC_OP_MODES = ("off", "detached", "e2e")
#: R3: the widths of the FROZEN v7 vocabularies (vocab_v7 TACTICAL_GOAL_TOKENS_V7 / TACTICAL_LAT_ACTIONS_V7 /
#: TACTICAL_LON_ACTIONS_V7). `_frozen_v7()` cross-checks them, so a vocabulary edit fails HERE by name.
V7_GOAL_WIDTH = 22
V7_LAT_WIDTH = 8
V7_LON_WIDTH = 8
#: the two g_tac arg slots R3's PROPOSED mapping gives SPEED_BAND (v_lo_ms, v_hi_ms)
SPEED_BAND_ARG_SLOTS: tuple[int, ...] = (0, 1)


def _register(dest: str, kind: str, check: Callable | None, reason: str, tag: str) -> None:
    if kind not in KINDS or kind == "drivort":
        raise ValueError(f"G-DVB(v6) kind {kind!r} not in {tuple(k for k in KINDS if k != 'drivort')}")
    if kind in ("built", "loss") and check is None:
        raise ValueError(f"G-DVB(v6) `{dest}` is a {kind} lever and has no check")
    if kind not in ("built", "loss") and not reason:
        raise ValueError(f"G-DVB(v6) `{dest}` ({kind}) needs a reason a reader can verify")
    REGISTRY_V6[dest] = Lever(dest, kind, check, reason)


def register(dest: str, kind: str, check: Callable | None = None, reason: str = "") -> None:
    """R1/R4 API: ``check(stack, args) -> list[Mismatch]`` (no loss weights). Wrapped to the uniform signature."""
    wrapped = None if check is None else (lambda m, a, w=None, _c=check: _c(m, a))
    _register(dest, kind, wrapped, reason, "r1r4")
    NEEDS_WEIGHTS.discard(dest)


def register_v6(dest: str, kind: str, check: Callable | None = None, reason: str = "") -> None:
    """R3 API: ``check(stack, args, weights_in_force) -> list[Mismatch]``; a ``loss`` lever needs the weights."""
    _register(dest, kind, check, reason, "r3")
    if kind == "loss":
        NEEDS_WEIGHTS.add(dest)
    else:
        NEEDS_WEIGHTS.discard(dest)


def _flag(dest: str) -> str:
    return "--" + dest.replace("_", "-")


def _eq(dest, declared, built, where, why="") -> list:
    return [] if declared == built else [Mismatch(_flag(dest), declared, built, where, why)]


def _a(args, name, default=None) -> Any:
    return getattr(args, name, default)


# ---------------------------------------------------------------------------
# R1 / R4 readers -- the BUILT side
# ---------------------------------------------------------------------------

def _c_tac_op_cond(m, a) -> list:
    """R4: argv mode == the mode the forward branches on (``stack.cfg.tac_op_cond``), the port is built iff the
    mode is not ``off``, and it has the LITERAL shape and group."""
    want = str(_a(a, "tac_op_cond", "off") or "off")
    if want not in _TAC_OP_MODES:
        return [Mismatch("--tac-op-cond", f"one of {_TAC_OP_MODES}", want, "argv")]
    out = _eq("tac_op_cond", want, str(getattr(m.cfg, "tac_op_cond", "<absent>")),
              "stack.cfg.tac_op_cond (read by V6Stack.forward)")
    port = getattr(m, "tac_op_port", None)
    out += _eq("tac_op_cond", want != "off", port is not None, "stack.tac_op_port is not None",
               "an R4 arm whose behaviours never reach the plan, or a control carrying the port")
    if port is not None:
        d = int(m.cfg.d_goal_embed)
        out += _eq("tac_op_cond", (2 * d, d), (int(port.in_features), int(port.out_features)),
                   "stack.tac_op_port (in, out)", "reads e_a_tac [2*d_goal_embed], writes e_g_tac")
        out += _eq("tac_op_cond", "planner", m.group_of("tac_op_port.weight"),
                   "stack.group_of('tac_op_port.weight')",
                   "the consuming side of the seam trains with the planner (S-T)")
    return out


def _c_max_speed_input_v6(m, a) -> list:
    """R1a: the port is built iff argv asks, with 4 input bins and the tactical width."""
    want = bool(_a(a, "max_speed_input_v6", False))
    out = _eq("max_speed_input_v6", want, bool(getattr(m.cfg, "max_speed_input_v6", False)),
              "stack.cfg.max_speed_input_v6")
    port = getattr(m, "vmax_tac", None)
    out += _eq("max_speed_input_v6", want, port is not None, "stack.vmax_tac is not None",
               "a max-speed arm whose tactical layer never sees the limit (or the mirror)")
    if port is not None:
        out += _eq("max_speed_input_v6", (_N_VMAX_BINS, int(m.cfg.d_tac)),
                   (int(port.in_features), int(port.out_features)),
                   "stack.vmax_tac (in, out)", "the refcv6 4-bin one-hot into z_tac_p [d_tac]")
        out += _eq("max_speed_input_v6", "layer_tac", m.group_of("vmax_tac.weight"),
                   "stack.group_of('vmax_tac.weight')")
    return out


def _c_plan_vmax_cap(m, a) -> list:
    """R1b: holds no parameters, so the BUILT fact is the config field ``emit`` reads for its eval-mode refusal --
    argv and the model must agree, or an eval harness reading the checkpoint's config would not know the arm
    requires the cap."""
    want = bool(_a(a, "plan_vmax_cap", False))
    return _eq("plan_vmax_cap", want, bool(getattr(m.cfg, "plan_vmax_cap", False)),
               "stack.cfg.plan_vmax_cap (read by V6Stack.emit)")


# ---------------------------------------------------------------------------
# R3 reader -- the BUILT side AND the loss weights in force
# ---------------------------------------------------------------------------

def _frozen_v7() -> tuple[tuple, tuple, tuple]:
    from tanitad.models import vocab_v7 as V  # noqa: PLC0415
    goal, lat, lon = (tuple(V.TACTICAL_GOAL_TOKENS_V7), tuple(V.TACTICAL_LAT_ACTIONS_V7),
                      tuple(V.TACTICAL_LON_ACTIONS_V7))
    if (len(goal), len(lat), len(lon)) != (V7_GOAL_WIDTH, V7_LAT_WIDTH, V7_LON_WIDTH):
        raise RuntimeError(
            f"G-DVB(v6): the frozen v7 vocabulary widths moved ({len(goal)}/{len(lat)}/"
            f"{len(lon)} vs the literals {V7_GOAL_WIDTH}/{V7_LAT_WIDTH}/{V7_LON_WIDTH}) -- a "
            f"vocabulary change is a versioned change (vocab_v7.assert_frozen); update both.")
    return goal, lat, lon


def _width(head) -> int | None:
    th = getattr(head, "type_head", None)
    return None if th is None else int(getattr(th, "out_features", -1))


def _c_w_tac_label_all(stack, args, weights_in_force) -> list[Mismatch]:
    """R3's weight: in force as declared, and the heads it supervises built as the loss needs.

    OFF (argv 0.0): the weight the loss reads must be 0.0 too (a hand-built weights object that switches the term on
    behind argv's back is the D-REFCV6-CONFIG-BUILD class). R3 builds NO module, so there is nothing else to be
    built-or-not. ON: every check below reads the BUILT stack; every expectation is a literal."""
    w = float(getattr(args, "w_tac_label_all", 0.0) or 0.0)
    got = float(getattr(weights_in_force, "w_tac_label_all", float("nan")))
    out: list[Mismatch] = []
    if abs(got - w) > 1e-12:
        out.append(Mismatch(
            "--w-tac-label-all", w, got, "weights_in_force.w_tac_label_all (read by v6_loss_step)",
            "the weight the LOSS reads is not the declared one -- V6LossWeights.for_stage zeroes it "
            "where layer_tac is frozen (S-W/S-S): R3 is an S-T/S-J lever"))
    if w <= 0.0:
        return out
    goal, lat, lon = _frozen_v7()
    for attr, head_attr, want_toks, want_w in (
            ("vocab_tac", "goal_head_tac", goal, V7_GOAL_WIDTH),
            ("vocab_a_lat", "act_head_lat", lat, V7_LAT_WIDTH),
            ("vocab_a_lon", "act_head_lon", lon, V7_LON_WIDTH)):
        have = tuple(getattr(getattr(stack, attr, None), "tokens", ()) or ())
        if have != want_toks:
            out.append(Mismatch("--w-tac-label-all", f"stack.{attr}.tokens == the frozen v7 tuple "
                                f"({len(want_toks)})", f"{len(have)} tokens", f"stack.{attr}.tokens",
                                "the v7.2 ids are minted in the v7.0 vocabulary"))
        width = _width(getattr(stack, head_attr, None))
        if width != want_w:
            out.append(Mismatch("--w-tac-label-all", want_w, width,
                                f"stack.{head_attr}.type_head.out_features",
                                "the supervised head must be as wide as the label vocabulary"))
    gh = getattr(stack, "goal_head_tac", None)
    if not bool(getattr(gh, "multilabel", False)):
        out.append(Mismatch("--w-tac-label-all", "stack.goal_head_tac.multilabel True",
                            getattr(gh, "multilabel", None), "stack.goal_head_tac.multilabel",
                            "without the gates, e_g_tac (the operative planner's goal) reads "
                            "softmax(logits), not the BCE-supervised sigmoid gates (R3)"))
    for fa in ("goal_head_tac_lat", "goal_head_tac_lon"):
        if getattr(stack, fa, None) is not None:
            out.append(Mismatch("--w-tac-label-all", f"stack.{fa} is None", "built", f"stack.{fa}",
                                "a factored stack builds e_g_tac from the factored pair, so the "
                                "supervised mixed g_tac would condition nothing"))
    iso = getattr(getattr(stack, "cfg", None), "isolate_planner_from_encoder", None)
    if iso is not True:
        out.append(Mismatch("--w-tac-label-all", True, iso,
                            "stack.cfg.isolate_planner_from_encoder",
                            "the forward's planner cut reads it; without the cut the tactical "
                            "LABEL loss is a TRUNK loss"))
    n_args = int(getattr(getattr(gh, "arg_head", None), "out_features", -1))
    if n_args <= max(SPEED_BAND_ARG_SLOTS):
        out.append(Mismatch("--w-tac-label-all", f"> {max(SPEED_BAND_ARG_SLOTS)} arg slots", n_args,
                            "stack.goal_head_tac.arg_head.out_features",
                            "SPEED_BAND's (v_lo, v_hi) have no slot to land in"))
    return out


# ============================================================================
# THE REGISTRY -- one entry per NEW dest (coverage is tested)
# ============================================================================
register("tac_op_cond", "built", _c_tac_op_cond)
register("max_speed_input_v6", "built", _c_max_speed_input_v6)
register("plan_vmax_cap", "built", _c_plan_vmax_cap)
register("speed_max_sidecar_v6", "data", reason=(
    "the refcv6 max-speed sidecar (RAW SPEED_BAND.v_hi_ms per clip, keyed by stable id): "
    "read by train_v6_staged.build_vmax_join; refused by _preflight_r1_r4 when no lever "
    "consumes it or when a lever needs it and it is absent"))
register_v6("w_tac_label_all", "loss", _c_w_tac_label_all)
register_v6("tac_goal_negatives", "data", reason=(
    "the label NEGATIVE policy (v7_labels.tactical_goal_targets); computed on the LOADED split by "
    "train_v6_staged.tac_label_policy and recorded as config.json['tac_label_all']['negatives']; "
    "refused as inert without --w-tac-label-all (preflight)"))
register_v6("cot_negative_sidecar", "data", reason=(
    "the PI 2026-09-16 absence-as-negative permission token; v7_labels.load_cot_negative_sidecar "
    "refuses one built over another blob md5, and its stamp is recorded as "
    "config.json['tac_label_all']['cot_absence_negative']"))


# ============================================================================
# ⭐ THE REST OF THE TRAINER'S FLAGS -- GATE-TIME entries (launch-gate agent, 2026-09-27)
# ============================================================================
# Every pre-existing `train_v6_staged` dest (the tip's 231: 232 options, `--require-parity` /
# `--no-require-parity` share one) plus R6's `strategic_off`, so `uncovered_v6(parser) == []` on the
# merged trainer (239 dests). ⛔ GATE-TIME ONLY: these readers run in `check_all` (the v7f launch
# gate, on the FULLY built and frozen stack), NEVER inside `build_stack_from_args` -- whose call
# site runs BEFORE the O5 teacher, the O14 head, the anchor table and the fallback calibration are
# installed (a reader there would refuse every build that uses them), and never in the R3 train
# path (`check_v6`), so the trainer stays bit-identical to the merge. Wiring the full registry into
# the trainer is an INTEGRATION decision (on the first merge it would, e.g., have refused every
# `--nav-cond` build, because nav was declared but never built -- `_c_nav_cond`; fixed since).
_ABSENT = "<absent>"


def register_gate(dest: str, kind: str, check: Callable | None = None, reason: str = "") -> None:
    """GATE-TIME entry: ``check(stack, args) -> list[Mismatch]`` on the FULLY built stack (wrapped to
    the uniform signature); run by ``check_all`` only."""
    wrapped = None if check is None else (lambda m, a, w=None, _c=check: _c(m, a))
    _register(dest, kind, wrapped, reason, "gate")
    NEEDS_WEIGHTS.discard(dest)
    GATE_TIME.add(dest)


def _near(dest, declared, built, where, why="", tol=1e-9) -> list:
    try:
        ok = abs(float(declared) - float(built)) <= tol * max(1.0, abs(float(declared)))
    except (TypeError, ValueError):
        ok = declared == built
    return [] if ok else [Mismatch(_flag(dest), declared, built, where, why)]


def _get(obj, path: str):
    for part in path.split("."):
        obj = getattr(obj, part, _ABSENT)
        if obj is _ABSENT:
            return _ABSENT
    return obj


def _present(stack, attr: str) -> bool:
    return getattr(stack, attr, None) is not None


def _c_cfg(dest: str, path: str, want_fn: Callable, why: str = "", tol=None) -> Callable:
    """``stack.cfg.<path>`` (the config the forward branches on) against a LITERAL from argv."""
    def chk(m, a):
        built = _get(m.cfg, path)
        want = want_fn(a)
        if tol is not None and built is not _ABSENT:
            return _near(dest, want, built, f"stack.cfg.{path}", why, tol)
        if isinstance(built, list):
            built = tuple(built)
        return _eq(dest, want, built, f"stack.cfg.{path}", why)
    return chk


def _c_module(dest: str, attr: str, want_fn: Callable, why: str = "") -> Callable:
    """a module is BUILT iff argv asks (presence on the stack)."""
    def chk(m, a):
        return _eq(dest, bool(want_fn(a)), _present(m, attr), f"stack.{attr} is not None", why)
    return chk


def _both(*fns: Callable) -> Callable:
    def chk(m, a):
        out: list = []
        for f in fns:
            out += list(f(m, a) or [])
        return out
    return chk


def _c_encoder_geometry(dest: str) -> Callable:
    """The ENCODER as built: ``patch`` conv [d_model, in_ch, P, P], the token grid, ``len(blocks)`` --
    structural, so a config that is right while the module is wrong is caught."""
    def chk(m, a):
        enc = getattr(m, "encoder", None)
        w = getattr(getattr(enc, "patch", None), "weight", None)
        out = []
        if dest == "enc_depth":
            blocks = getattr(enc, "blocks", None)
            if blocks is not None:
                out += _eq(dest, int(_a(a, "enc_depth")), len(blocks), "len(stack.encoder.blocks)")
            return out + _c_cfg(dest, "encoder.depth", lambda a: int(_a(a, "enc_depth")))(m, a)
        if dest == "enc_heads":
            return _c_cfg(dest, "encoder.n_heads", lambda a: int(_a(a, "enc_heads")))(m, a)
        if w is None:
            return [Mismatch(_flag(dest), "a ViT patch conv", type(enc).__name__,
                             "stack.encoder.patch.weight", "encoder geometry unreadable")]
        if dest == "enc_dim":
            out += _eq(dest, int(_a(a, "enc_dim")), int(w.shape[0]), "stack.encoder.patch.weight[0]")
        elif dest == "in_channels":
            out += _eq(dest, int(_a(a, "in_channels")), int(w.shape[1]),
                       "stack.encoder.patch.weight[1]")
        elif dest == "patch":
            out += _eq(dest, int(_a(a, "patch")), int(w.shape[-1]), "stack.encoder.patch.weight[-1]")
        elif dest in ("frame_h", "frame_w"):
            p = int(w.shape[-1])
            want = int(_a(a, dest))
            got = int(getattr(enc, "grid_h" if dest == "frame_h" else "grid_w", -1)) * p
            out += _eq(dest, want, got, "stack.encoder.grid_{h,w} x patch",
                       "the encoder is sized for the frame the data will deliver")
        return out
    return chk


def _c_nav_cond(m, a) -> list:
    """R2 (PI 2026-08-30 / R1-R6 2026-09-27): the nav conditioner is BUILT iff argv declares it.
    ⛔ MEASURED at c36b6ddd and on the first v7F merge (the launch-gate agent, 2026-09-27):
    `build_stack_from_args` never mapped `--nav-cond` into V6Config, so every `--nav-cond` launch built
    NO NavConditioner while its preflight REQUIRED the flag. FIXED in the merge the same day (the
    mapping line, pinned through the real launch path by test_v7f_r2_nav_fixes.py::test_REAL_LAUNCH_*).
    This reader stays as the guard: the gate's arm `v6_nav_not_mapped` re-installs the unmapped build
    and it must FAIL."""
    want = bool(_a(a, "nav_cond", False))
    out = _eq("nav_cond", want, bool(getattr(m.cfg, "nav_cond", False)),
              "stack.cfg.nav_cond (read by V6Stack.__init__/forward)",
              "a nav-declared arm whose model never sees the nav command")
    out += _eq("nav_cond", want, _present(m, "nav"), "stack.nav is not None",
               "the NavConditioner that feeds all three layers")
    return out


def _c_n_candidates(m, a) -> list:
    want = int(_a(a, "n_candidates", 8))
    out = _c_cfg("n_candidates", "n_candidates", lambda a: int(_a(a, "n_candidates", 8)))(m, a)
    cq = getattr(m, "cand_queries", None)
    if cq is not None and getattr(cq, "weight", None) is not None:
        out += _eq("n_candidates", want, int(cq.weight.shape[0]), "stack.cand_queries.weight[0]",
                   "the emitted fan width")
    return out


def _c_selector(m, a) -> list:
    want = str(_a(a, "selector", "none"))
    out = _c_cfg("selector", "selector", lambda a: str(_a(a, "selector", "none")))(m, a)
    return out + _eq("selector", want != "none", _present(m, "cand_score"),
                     "stack.cand_score is not None", "a scorer is built iff a selector is declared")


def _c_proposals(m, a) -> list:
    want = str(_a(a, "proposals", "query"))
    out = _c_cfg("proposals", "proposals", lambda a: str(_a(a, "proposals", "query")))(m, a)
    return out + _eq("proposals", want == "diffusion", _present(m, "prop_diffusion"),
                     "stack.prop_diffusion is not None")


def _c_o5_target(m, a) -> list:
    """live = no teacher; ema = the EMA pair; frozen = the fixed pair (installed AFTER V6Stack by
    build_stack_from_args -- a gate-time reader)."""
    want = str(_a(a, "o5_target", "live"))
    has_ema = hasattr(m, "ema_o5_enc") and hasattr(m, "ema_o5_ro")
    has_frz = hasattr(m, "frozen_o5_enc") and hasattr(m, "frozen_o5_ro")
    built = "ema" if has_ema and not has_frz else ("frozen" if has_frz and not has_ema else
                                                   ("live" if not (has_ema or has_frz) else "both"))
    return _eq("o5_target", want, built, "stack.{ema,frozen}_o5_{enc,ro} presence")


def _c_uplink(m, a) -> list:
    want = str(_a(a, "uplink", "stopgrad"))
    out = _c_cfg("uplink", "uplink", lambda a: str(_a(a, "uplink", "stopgrad")))(m, a)
    return out + _eq("uplink", want == "ema", _present(m, "ema_adapter_tac"),
                     "stack.ema_adapter_tac is not None", "the EMA-slow uplink target")


def _c_per_layer(m, a) -> list:
    want = not bool(_a(a, "per_layer_encoders", False))
    out = _c_cfg("per_layer_encoders", "shared_encoder",
                 lambda a: not bool(_a(a, "per_layer_encoders", False)))(m, a)
    return out + _eq("per_layer_encoders", not want, _present(m, "encoder_tac"),
                     "stack.encoder_tac is not None", "E-ENC arm (b): one encoder per layer")


def _c_vit5(m, a) -> list:
    want = bool(_a(a, "vit5_encoder", False))
    out = _c_cfg("vit5_encoder", "vit5_encoder", lambda a: bool(_a(a, "vit5_encoder", False)))(m, a)
    return out + _eq("vit5_encoder", "ViT5Encoder" if want else "ViTEncoder",
                     type(getattr(m, "encoder", None)).__name__, "type(stack.encoder).__name__")


def _c_w_o14(m, a) -> list:
    return _eq("w_o14", float(_a(a, "w_o14", 0.0) or 0.0) > 0, hasattr(m, "o14_head"),
               "hasattr(stack, 'o14_head')", "the O14 head is built iff --w-o14 > 0")


def _c_anchor_goal(m, a) -> list:
    want = str(_a(a, "anchor_goal", "none"))
    out = _c_cfg("anchor_goal", "anchor_goal", lambda a: str(_a(a, "anchor_goal", "none")))(m, a)
    return out + _eq("anchor_goal", want != "none", _present(m, "anchor_head"),
                     "stack.anchor_head is not None")


def _c_newest_frame_only(m, a) -> list:
    if not bool(_a(a, "newest_frame_only", False)):
        return []
    return _eq("newest_frame_only", 3, int(getattr(m.cfg.encoder, "in_channels", -1)),
               "stack.cfg.encoder.in_channels",
               "--newest-frame-only feeds 3-channel frames (the loader side is the provider's)")


def _c_enc_gc(m, a) -> list:
    v = _a(a, "enc_grad_checkpoint", "auto")
    want = bool(_a(a, "grad_checkpoint", False)) if v in (None, "auto") else v == "on"
    return _eq("enc_grad_checkpoint", want, bool(getattr(m.cfg.encoder, "grad_checkpoint", False)),
               "stack.cfg.encoder.grad_checkpoint", "resolved against --grad-checkpoint (auto)")


def _c_tac_vocab(m, a) -> list:
    # a RECORDED-args namespace without the key means v6.0 (the round trip build_stack_from_args
    # documents) -- a LITERAL restatement of that rule
    want = str(_a(a, "tac_vocab_version", "v6.0"))
    return _eq("tac_vocab_version", want, str(getattr(m.cfg, "tac_vocab_version", _ABSENT)),
               "stack.cfg.tac_vocab_version")


def _c_anchor_table(m, a) -> list:
    if not _a(a, "anchor_table", None):
        return []
    tab = getattr(getattr(m, "anchor_head", None), "anchors", None)
    loaded = tab is not None and bool(tab.numel()) and bool((tab.detach().abs().sum() > 0).item())
    return _eq("anchor_table", True, loaded, "stack.anchor_head.anchors (non-zero)",
               "the anchor table must be INSTALLED at build time")


def _c_fallback_calibration(m, a) -> list:
    if not _a(a, "fallback_calibration", None):
        return []
    fb = getattr(m, "fallback", None)
    cal = getattr(fb, "calibrated", None) if fb is not None else None
    if callable(cal):
        cal = cal()
    return _eq("fallback_calibration", True, bool(cal) if cal is not None else fb is not None,
               "stack.fallback (calibrated)")


def _c_strategic_off(m, a) -> list:
    """R6 (merge): the field the tactical goal heads branch on."""
    return _c_cfg("strategic_off", "strategic_off", lambda a: bool(_a(a, "strategic_off", False)),
                  "R6: the tactical goal heads read a ZERO strategic conditioning iff declared")(m, a)


def _loss_head(dest: str, head_fn: Callable, where: str) -> Callable:
    """a LOSS weight > 0 needs its head BUILT; the term's presence/finiteness/reach are G-LIVE's."""
    def chk(m, a):
        w = float(_a(a, dest, 0.0) or 0.0)
        if w <= 0:
            return []
        return _eq(dest, True, bool(head_fn(m)), where, f"{_flag(dest)} {w} needs its head")
    return chk


def _i(d, dflt=None):
    return lambda a, _d=d, _v=dflt: None if _a(a, _d, _v) is None else int(_a(a, _d, _v))


def _fl(d, dflt=None):
    return lambda a, _d=d, _v=dflt: None if _a(a, _d, _v) is None else float(_a(a, _d, _v))


def _s(d, dflt=None):
    return lambda a, _d=d, _v=dflt: None if _a(a, _d, _v) is None else str(_a(a, _d, _v))


def _bo(d):
    return lambda a, _d=d: bool(_a(a, _d, False))


def _gb(dest, fn):
    register_gate(dest, "built", fn)


#: dest -> (cfg path, literal expectation) for the plain V6Config levers (build_stack_from_args maps
#: each; the expectation is restated here, never imported)
MODEL_CFG_LEVERS: dict[str, tuple[str, Callable]] = {
    "readout_grid": ("readout.grid", _i("readout_grid")),
    "readout_dim": ("readout.d_readout", _i("readout_dim")),
    "readout_grid_w": ("readout.grid_w", _i("readout_grid_w")),
    "pred_dim": ("predictor.d_model", _i("pred_dim")),
    "pred_depth": ("predictor.depth", _i("pred_depth")),
    "pred_heads": ("predictor.n_heads", _i("pred_heads")),
    "window": ("predictor.window", _i("window")),
    "horizons": ("predictor.horizons", lambda a: tuple(int(x) for x in _a(a, "horizons"))),
    "pred_modern": ("predictor.modern", _bo("pred_modern")),
    "d_tac": ("d_tac", _i("d_tac")), "d_str": ("d_str", _i("d_str")),
    "d_goal_embed": ("d_goal_embed", _i("d_goal_embed")),
    "adapter_hidden": ("adapter_hidden", _i("adapter_hidden")),
    "plan_steps": ("plan_steps", _i("plan_steps")), "dt": ("dt", _fl("dt")),
    "a_max": ("a_max", _fl("a_max")), "kappa_max": ("kappa_max", _fl("kappa_max")),
    "no_isolate_planner": ("isolate_planner_from_encoder",
                           lambda a: not bool(_a(a, "no_isolate_planner", False))),
    "no_isolate_uplink": ("isolate_uplink", lambda a: not bool(_a(a, "no_isolate_uplink", False))),
    "no_isolate_interp": ("isolate_interp_from_encoder",
                          lambda a: not bool(_a(a, "no_isolate_interp", False))),
    "ema_decay": ("ema_decay", _fl("ema_decay")),
    "sigreg_slices": ("sigreg_slices", _i("sigreg_slices")),
    "sigreg_subspaces": ("sigreg_subspaces", _i("sigreg_subspaces", 1)),
    "sigreg_free_dims": ("sigreg_free_dims", _i("sigreg_free_dims")),
    "param_budget": ("param_budget", _i("param_budget")),
    "f_hidden_tac": ("f_hidden_tac", _i("f_hidden_tac")),
    "f_hidden_str": ("f_hidden_str", _i("f_hidden_str")), "f_blocks": ("f_blocks", _i("f_blocks")),
    "selector_tau_m": ("selector_tau_m", _fl("selector_tau_m")),
    "selector_mlp_hidden": ("selector_mlp_hidden", _i("selector_mlp_hidden", 256)),
    "plan_wta_eps": ("plan_wta_eps", _fl("plan_wta_eps")),
    "goal_factored": ("goal_factored", _bo("goal_factored")),
    "goal_multilabel": ("goal_multilabel", _bo("goal_multilabel")),
    "goal_cat_args": ("goal_cat_args", _bo("goal_cat_args")),
    "tac_goal_cond": ("tac_goal_cond", _bo("tac_goal_cond")),
    "t2_contrastive": ("t2_contrastive", _bo("t2_contrastive")),
    "d_t2_proj": ("d_t2_proj", _i("d_t2_proj", 128)), "d_t2_hidden": ("d_t2_hidden", _i("d_t2_hidden", 256)),
    "t2_tau": ("t2_tau", _fl("t2_tau", 0.1)),
    "n_anchors": ("n_anchors", _i("n_anchors", 256)), "n_lat_bins": ("n_lat_bins", _i("n_lat_bins", 16)),
    "n_agent_slots": ("n_agent_slots", _i("n_agent_slots", 8)),
    "diffusion_steps": ("diffusion_steps", _i("diffusion_steps", 4)),
    "diffusion_noise_rho": ("diffusion_noise_rho", _fl("diffusion_noise_rho", 0.9)),
    "diffusion_hidden": ("diffusion_hidden", _i("diffusion_hidden", 256)),
    "diffusion_sigma_a": ("diffusion_sigma_a", _fl("diffusion_sigma_a", 2.0)),
    "diffusion_sigma_k": ("diffusion_sigma_k", _fl("diffusion_sigma_k", 0.1)),
    "mpc_refine": ("mpc_refine", _bo("mpc_refine")), "mpc_topk": ("mpc_topk", _i("mpc_topk", 2)),
    "mpc_steps": ("mpc_steps", _i("mpc_steps", 3)), "mpc_lr": ("mpc_lr", _fl("mpc_lr", 0.05)),
    "mpc_roll_k": ("mpc_roll_k", _i("mpc_roll_k", 0)),
    "mpc_w_goal": ("mpc_w_goal", _fl("mpc_w_goal", 1.0)), "mpc_w_kin": ("mpc_w_kin", _fl("mpc_w_kin", 0.1)),
    "mpc_w_consist": ("mpc_w_consist", _fl("mpc_w_consist", 0.0)),
    "fallback_trigger": ("fallback_trigger", _bo("fallback_trigger")),
    "fallback_roll_k": ("fallback_roll_k", _i("fallback_roll_k", 10)),
    "agent_slots": ("agent_slots", _bo("agent_slots")),
    "n_slot_queries": ("n_slot_queries", _i("n_slot_queries")),
    "slot_hidden": ("slot_hidden", _i("slot_hidden", 256)), "slot_depth": ("slot_depth", _i("slot_depth", 3)),
    "slot_heads": ("slot_heads", _i("slot_heads", 8)), "slot_src": ("slot_src", _s("slot_src", "cells")),
    "n_registers": ("n_registers", _i("n_registers")),
}
for _d, (_p, _fn) in MODEL_CFG_LEVERS.items():
    _gb(_d, _c_cfg(_d, _p, _fn))
# structural presence for the levers that BUILD a module (the cfg field alone could be right while the
# module is absent -- the D-REFCV6-CONFIG-BUILD class)
for _d, _attr in (("tac_goal_cond", "cond_tac_dyn"), ("t2_contrastive", "t2_head"),
                  ("fallback_trigger", "fallback"), ("agent_slots", "agent_slots"),
                  ("goal_factored", "goal_head_tac_lat")):
    _gb(_d, _both(_c_cfg(_d, MODEL_CFG_LEVERS[_d][0], MODEL_CFG_LEVERS[_d][1]),
                  _c_module(_d, _attr, _bo(_d))))
for _d in ("enc_dim", "enc_depth", "enc_heads", "in_channels", "patch", "frame_h", "frame_w"):
    _gb(_d, _c_encoder_geometry(_d))
for _d, _fn in (("nav_cond", _c_nav_cond), ("n_candidates", _c_n_candidates), ("selector", _c_selector),
                ("proposals", _c_proposals), ("o5_target", _c_o5_target), ("uplink", _c_uplink),
                ("per_layer_encoders", _c_per_layer), ("vit5_encoder", _c_vit5),
                ("anchor_goal", _c_anchor_goal), ("newest_frame_only", _c_newest_frame_only),
                ("enc_grad_checkpoint", _c_enc_gc), ("tac_vocab_version", _c_tac_vocab),
                ("anchor_table", _c_anchor_table), ("fallback_calibration", _c_fallback_calibration),
                ("strategic_off", _c_strategic_off)):
    _gb(_d, _fn)
register_gate("grad_checkpoint", "elsewhere", reason=(
    "the MASTER switch: resolved into --enc-grad-checkpoint (built: stack.cfg.encoder.grad_checkpoint) "
    "and --rollout-grad-checkpoint (a v6_loss_step argument, the gate's G-LIVE loss-knob capture)"))
register_gate("o5_target_crop", "elsewhere", reason=(
    "refused outside (0, 1) by build_stack_from_args; applied to the O5 TARGET frames in train() "
    "(azimuthal_target_crop) -- the target side only, no module"))
for _d in ("ema_decay_ramp", "ema_decay_start", "ema_decay_end"):
    register_gate(_d, "runtime", reason=(
        "the O5-EMA teacher's per-step tau schedule (ema_tau_at, validated at build); stamped in the "
        "log rows by _ema_tau_record -- a schedule, not a module"))
# -- loss weights: presence/finiteness/reach are G-LIVE's; the head dependency is read here --------
register_gate("w_o14", "loss", _c_w_o14)
register_gate("w_select", "loss", _loss_head("w_select", lambda m: _present(m, "cand_score"),
                                             "stack.cand_score is not None"))
register_gate("w_anchor", "loss", _loss_head("w_anchor", lambda m: _present(m, "anchor_head"),
                                             "stack.anchor_head is not None"))
register_gate("w_t2_contrast", "loss", _loss_head("w_t2_contrast", lambda m: _present(m, "t2_head"),
                                                  "stack.t2_head is not None"))
for _d, _term in (("w_o1_ctrl", "o1"), ("w_o1_fact", "o1"), ("w_o1_scene", "o1"), ("w_o2", "o2"),
                  ("w_o3", "o3"), ("w_o5", "o5"), ("w_o6", "o6"), ("w_o11_cf", "o11"),
                  ("w_o13_ego", "o13"), ("w_t1", "t1"), ("w_s1", "s1"), ("w_s1_multi", "s1_multi"),
                  ("w_s2_goal", "s2"), ("w_t5_consist", "t5"), ("lambda_plan", "plan")):
    register_gate(_d, "elsewhere", reason=(
        f"a V6LossWeights field (term `{_term}` of v6_loss_step): its head is ALWAYS built, so the "
        f"declared-vs-live question is G-LIVE's -- the term must be present, finite and reach a "
        f"trainable parameter on every training step iff the trainer's effective-weight registry "
        f"says it builds a graph at this stage"))
for _d, _mod in (("w_o7_distill", "O7Distill"), ("w_o8_pixel", "O8Pixel"), ("w_o9_ema", "O9EmaMasked"),
                 ("w_o10_psg", "O10PSG"), ("w_trunk_anchor", "TrunkAnchor")):
    register_gate(_d, "elsewhere", reason=(
        f"built INSIDE train() ({_mod}, not a V6Stack module) only when the weight is > 0 and added to "
        f"L['loss'] after v6_loss_step: its log keys are G-LIVE's term check"))
for _d in ("o1_detach_encoder", "o1_stopgrad_factual"):
    register_gate(_d, "elsewhere", reason="a V6LossWeights switch of the O1 term (G-LIVE: O1's log row "
                                          "echoes it as o1_detach_encoder / o1_stopgrad_factual)")
for _d in ("o1_k", "o5_k", "o5_mode", "o5_form", "o3_mode", "o3_blocks", "o3_block_h", "o3_block_w",
           "o3_band_rows", "o2_tau_s", "dkappa", "daccel", "o11_k", "o11_tau", "o11_negs", "o13_k",
           "o13_seed", "cond_param", "anchor_objective", "anchor_axis_w", "t2_positive", "t2_negative",
           "t5_w_kappa", "bptt_truncate", "rollout_grad_checkpoint", "o6_innovation",
           "o6_innovation_shuffle", "sigreg_accum"):
    register_gate(_d, "elsewhere", reason=(
        "a v6_loss_step ARGUMENT (no module): the launch gate's G-LIVE records the arguments of every "
        "training call and compares them with argv (the 'dry-run passed a knob the train path did "
        "not' class)"))
for _d in ("o14_mode", "o14_k", "o14_shuffle_targets"):
    register_gate(_d, "elsewhere", reason="the O14 TARGET construction at the batch-build site "
                                          "(_o14_pixel_target); G-LIVE checks the o14 term when "
                                          "--w-o14 > 0")
for _d in ("rand_dkappa_max", "rand_daccel_max"):
    register_gate(_d, "elsewhere", reason="the O1 random-counterfactual draw (sample_random_deltas) at "
                                          "the batch-build site; O1 is G-LIVE's term")
register_gate("s1_multi_k", "elsewhere", reason="the F-11 roll depth (z_str_multi_target, a batch key); "
                                                "refused as unreachable by train(); its term is G-LIVE's")
register_gate("t5_lag", "elsewhere", reason="the T5 partner offset (batch['t5_lag']); refused without "
                                            "--t5-pairs by preflight")
register_gate("t5_pairs", "elsewhere", reason="the consecutive-window pair sampler (batch['t5_pairs']); "
                                              "T5's term is G-LIVE's")
for _d in ("o7_model", "o9_mask_frac", "o9_momentum", "o9_neighbour_k", "psg_enc_only",
           "psg_eval_every"):
    register_gate(_d, "elsewhere", reason="a knob of an O7/O9/O10 module built INSIDE train() only "
                                          "when its weight is > 0 (G-LIVE's term check covers it)")
for _d, _grp in (("freeze_encoder", "encoder"), ("freeze_readout", "readout")):
    register_gate(_d, "elsewhere", reason=(
        f"an EXTRA freeze of the `{_grp}` group applied inside train() AFTER the corpus build, i.e. "
        f"after the point a build-only capture sees: G-LIVE's frozen-group check holds every "
        f"`{_grp}` parameter bit-identical over the smoke when set"))
register_gate("stage", "elsewhere", reason=(
    "the STAGE: the per-group trainability after the trainer's freeze is checked against the gate's "
    "LITERAL stage table (the v7f G-DVB probe), and the loss terms in force by G-LIVE"))
for _d, _why in (("lr", "the AdamW learning rate (build_trunk_optimizer)"),
                 ("wd", "AdamW weight decay (build_trunk_optimizer)"),
                 ("clip", "the grad-norm clip (clip_grad_norm_ in the loop)"),
                 ("trunk_lr_scale", "the trunk param group's LR factor (build_trunk_optimizer; "
                                    "recorded as trunk_optimizer in config.json)"),
                 ("trunk_lr_warmup_steps", "the trunk LR hard gate (trunk_lr_factor)"),
                 ("steps", "the run length (and the LR schedule's horizon)"),
                 ("batch", "the batch size"), ("eps_per_batch", "episodes per batch (the sampler)"),
                 ("seed", "the RNG seed: torch/random/the sampler generator are seeded from it at EVERY "
                          "launch -- the launch gate's G-CKPT measures what that does to a resume"),
                 ("device", "the device"), ("no_amp", "autocast off (resolve_amp_dtype, recorded)"),
                 ("log_every", "logging cadence"), ("save_every", "checkpoint cadence"),
                 ("no_step_ckpts", "keep/skip ckpt_step<N>.pt copies"),
                 ("resume", "auto/off resume policy (resume_guard)"),
                 ("force_rerun", "the done-marker override (resume_guard)"),
                 ("max_horizon", "the WINDOWING horizon (refused below what the stage needs)"),
                 ("spectrum_every", "O6 spectrum monitor cadence"),
                 ("spectrum_accum", "O6 spectrum pooling (a monitor)"),
                 ("spectrum_ci_reps", "O6 spectrum bootstrap CI (a monitor)"),
                 ("x4_spectrum_layers", "the X4 per-layer spectrum monitor"),
                 ("obs_monitor_every", "the Observer-Effect monitor cadence (an instrument)"),
                 ("obs_monitor_window", "the Observer-Effect monitor buffer"),
                 ("obs_monitor_dims", "the Observer-Effect projection width"),
                 ("dry_run", "the --dry-run mode switch"), ("dry_steps", "dry-run length"),
                 ("dry_batch", "dry-run batch"), ("dry_k", "dry-run future depth"),
                 ("print_launch", "print the launch line and exit"),
                 ("refuse_unreached", "promote the grad-reach census to a refusal"),
                 ("out", "the run directory"), ("o4_alpha", "O4 interaction-weighted SAMPLING"),
                 ("o4_floor", "O4 sampling floor"),
                 ("t3_alpha_start", "T3 curriculum schedule (sampling)"),
                 ("t3_alpha_end", "T3 curriculum schedule (sampling)"),
                 ("t3_warmup_frac", "T3 curriculum schedule (sampling)"),
                 ("t3_floor", "T3 curriculum floor (sampling)"),
                 ("domain_tau", "F-10 domain mix temperature (sampling)"),
                 ("domain_max_amp", "F-10 domain mix cap (sampling)"),
                 ("domain_min_stratum", "F-10 domain mix minimum stratum (sampling)"),
                 ("nav_semantics", "NavEmitter's arg semantics (t0_constant/decremented) -- the nav "
                                   "INPUT's time rule; its CLOCK is G-CLOCK's")):
    register_gate(_d, "runtime", reason=_why)
for _d, _why in (("v2_cache", "the corpus (parity-guarded by build_train_episodes)"),
                 ("v2_lru", "a cache size"), ("require_parity", "the corpus-parity refusal"),
                 ("v2_subframe", "the model sub-frame of the cache (the E2 frame lock in train() "
                                 "compares it with the encoder)"),
                 ("frame_hfov", "the cache frame's HFOV (resolve_v2_frames / the parity geometry "
                                "binding)"),
                 ("projection", "the cache frame's projection (resolve_v2_frames / the parity "
                                "geometry binding)"),
                 ("v2_val_cache", "⛔ REFUSED by train() (P4-5): not wired in this trainer"),
                 ("s2_labels", "the S2/v7.2 label blob (the S2 join; R3's tactical labels)"),
                 ("nav_labels", "the nav command source (the NavEmitter join)"),
                 ("psg_labels", "the O10 PSG targets"), ("t3_scores", "T3 per-window scores"),
                 ("domain_strata", "F-10 strata"),
                 ("exclude_eval_clips", "the v7.2 EVAL-split exclusion source"),
                 ("allow_eval_clips_in_train", "an eval-exclusion override (stamped)"),
                 ("allow_any_labels", "a label-version override (stamped)"),
                 ("init_from", "the predecessor checkpoint (load_stage_init: strict, "
                               "STAGE_MAY_INTRODUCE-adjudicated, trunk md5 stamped)"),
                 ("init_encoder_from", "the encoder-only seed (apply_encoder_seed: provenance stamp + "
                                       "geometry refusal)"),
                 ("prev_gate", "the predecessor's stage_gate.json (assert_stage_precondition)"),
                 ("gate_probes", "externally run battery probes folded into the stage gate"),
                 ("trunk_anchor_model", "cross-checked against the seed's provenance stamp "
                                        "(build_trunk_anchor)"),
                 ("dump_seam_plan", "an output path (the X2 seam dump)")):
    register_gate(_d, "data", reason=_why)
for _d, _why in (("allow_inconclusive_gate", "an operator override of an INCONCLUSIVE predecessor "
                                             "gate, recorded with --gate-off-reason"),
                 ("gate_off_reason", "the recorded reason for the override"),
                 ("predates_nav", "an ACKNOWLEDGEMENT (reproduce a pre-nav arm), recorded; the v7f "
                                  "launch-gate profile REFUSES it (R2)"),
                 ("control_arm_ack", "an ACKNOWLEDGEMENT (the isolation control arm); the v7f profile "
                                     "REFUSES it"),
                 ("allow_discarded_weights", "an ACKNOWLEDGEMENT of weights the effective-weight audit "
                                             "reports as discarded (stamped)"),
                 ("allow_unreached", "the on-the-record list of modules a run knowingly leaves at "
                                     "init (grad-reach census; the gate's G-LIVE honours it)"),
                 ("expect_n_trainable", "an optional runbook assertion on the trainable budget "
                                        "(_declared_freeze_preflight)"),
                 ("dump_seam_plan_degenerate", "allow banking a degenerate seam dump (recorded)")):
    register_gate(_d, "record", reason=_why)

#: v6_loss_step term keys a v7F stage must never put in force (R6 at the loss level; the gate reads it)
STAGE_LOSS_TERMS = {"strategic": ("s1", "s1_multi", "s2")}


# ============================================================================
# the API
# ============================================================================

def _run(stack, args, weights_in_force, *, include_weight_levers: bool,
         include_gate_levers: bool = False) -> list[Mismatch]:
    out: list[Mismatch] = []
    for lever in list(REGISTRY_V6.values()):
        if lever.kind not in ("built", "loss") or lever.check is None:
            continue
        if lever.dest in NEEDS_WEIGHTS and not include_weight_levers:
            continue
        if lever.dest in GATE_TIME and not include_gate_levers:
            continue
        try:
            out += list(lever.check(stack, args, weights_in_force) or [])
        except Exception as e:                           # noqa: BLE001 -- a reader that cannot read is a FAIL
            out.append(Mismatch(_flag(lever.dest), "readable", f"{type(e).__name__}: {e}",
                                "G-DVB(v6) reader", "the built value could not be read"))
    return out


def check(stack, args) -> list[Mismatch]:
    """BUILD-time check (R1/R4 API): every lever that does not need the loss weights; ``[]`` == clean."""
    return _run(stack, args, None, include_weight_levers=False)


def refuse_on_mismatch(stack, args, where: str = "train_v6_staged") -> None:
    bad = check(stack, args)
    if bad:
        raise SystemExit(
            f"[G-DVB-v6] ⛔⛔ {where}: {len(bad)} lever(s) DECLARED in argv are NOT what "
            f"the stack BUILT -- refusing before the first step:\n  - "
            + "\n  - ".join(str(b) for b in bad))


def coverage(dests) -> list[str]:
    """The given dests with NO entry here (``[]`` == all covered)."""
    return [d for d in dests if d not in REGISTRY_V6]


def _dests(parser) -> set[str]:
    return {x.dest for x in getattr(parser, "_actions", []) if x.dest != "help"}


def coverage_v6(parser, dests: tuple[str, ...] = NEW_V6_DESTS) -> list[str]:
    """The ``dests`` absent from ``parser`` OR from REGISTRY_V6 (``[]`` == covered). Both directions, because both
    are holes: a registered lever the parser dropped is a check that guards nothing, and a parser flag with no entry
    is the SPEC_REFCV7 §2 refusal."""
    have = _dests(parser)
    return sorted(d for d in dests if d not in have or d not in REGISTRY_V6)


def uncovered_v6(parser) -> list[str]:
    """Every dest of the v6 parser with NO v6 entry -- the size of the gap a v7F launch-gate profile must close
    (MEASURED, never assumed)."""
    return sorted(d for d in _dests(parser) if d not in REGISTRY_V6)


def check_v6(stack, args, *, weights_in_force, parser=None) -> list[Mismatch]:
    """-> every declared-vs-built mismatch for ALL v6 levers (the R1/R4 ones included); ``[]`` == clean.
    ``weights_in_force`` is the ``V6LossWeights`` object ``v6_loss_step`` will be handed."""
    out: list[Mismatch] = []
    if parser is not None:
        for d in coverage_v6(parser, ALL_V6_DESTS):
            out.append(Mismatch(_flag(d), "a trainer flag WITH a G-DVB(v6) entry",
                                "missing from the parser or the registry",
                                "tanitad.train.declared_vs_built_v6.REGISTRY_V6",
                                "a new v6 lever without a declared-vs-built entry is refused"))
    out += _run(stack, args, weights_in_force, include_weight_levers=True)
    return out


def refuse_on_mismatch_v6(stack, args, *, weights_in_force, parser=None, where: str = "train") -> None:
    """``SystemExit`` naming EVERY mismatch, or ``None`` when the build is clean."""
    bad = check_v6(stack, args, weights_in_force=weights_in_force, parser=parser)
    if bad:
        raise SystemExit(
            f"[G-DVB v6] ⛔⛔ {where}: {len(bad)} lever(s) DECLARED in argv are NOT what the stack "
            f"BUILT / the loss reads. Refusing before the first step:\n  - "
            + "\n  - ".join(str(b) for b in bad))


def check_all(stack, args, *, weights_in_force=None, parser=None) -> list[Mismatch]:
    """-> EVERY registered lever on the FULLY built stack (build-time, R3 and gate-time entries), plus
    the parser coverage of EVERY dest (``uncovered_v6``) and G-HYG's walk of the config tree the stack
    holds; ``[]`` == clean. The launch gate's v7f G-DVB. With ``weights_in_force=None`` the levers that
    need the loss weights are skipped (say so: pass the stage's weights)."""
    from tanitad.train.config_hygiene import undeclared_attributes  # noqa: PLC0415
    out: list[Mismatch] = []
    if parser is not None:
        for d in uncovered_v6(parser):
            out.append(Mismatch(_flag(d), "a G-DVB(v6) registry entry", "none",
                                "tanitad.train.declared_vs_built_v6.REGISTRY_V6",
                                "a trainer flag with no declared-vs-built entry is refused "
                                "(SPEC_REFCV7 §2): register it with its built reader"))
    out += _run(stack, args, weights_in_force, include_weight_levers=weights_in_force is not None,
                include_gate_levers=True)
    cfg = getattr(stack, "cfg", None)
    if cfg is not None:
        for path, cls, attr in undeclared_attributes(cfg, "stack.cfg"):
            out.append(Mismatch(f"{path}.{attr}", "a declared field", "an ad-hoc attribute",
                                f"{path} ({cls})", "G-HYG: dropped by the next rebuild"))
    return out


def kinds_census(parser) -> dict[str, int]:
    """{kind: n} over the parser's dests (``UNCOVERED`` for a dest with no entry) -- the gate's table."""
    out: dict[str, int] = {}
    for d in sorted(_dests(parser)):
        k = REGISTRY_V6[d].kind if d in REGISTRY_V6 else "UNCOVERED"
        out[k] = out.get(k, 0) + 1
    return out
