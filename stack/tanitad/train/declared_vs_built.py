"""G-DVB — every argv lever equals the BUILT model's value, or the run refuses (SPEC_REFCV7 §2).

⛔⛔ WHY A THIRD GUARD. The trainer already had two, and both passed on refcv6-r101-s0:

* the ARGV guards in ``_pin_trainer_cfg`` read argv — they cannot see a value lost AFTER them;
* ``assert_seams_are_built`` compares the seam STAMP with the model — but the stamp is built
  from the CONFIG, so a lever lost between argv and the config is lost from the stamp too, and
  the two agree about the wrong value. A check that shares the defect it checks for is green
  forever.

MEASURED on refcv6-r101-s0, three levers were declared and not built: ``--equalize-bottom-rows 43``
(the trunk built 0: D-REFCV6-EQUALIZE-DROPPED), F3's per-stage outputs (never passed through:
D-REFCV6-F3-WHITELIST), and three of the refcv6 selection mechanisms its ``config.json`` declared
(D-REFCV6-CONFIG-BUILD). Each was found by hand, weeks late. This module compares ARGV with the
BUILT MODULES directly, skipping the config in between.

THE API (stable — the launch gate imports it)::

    from tanitad.train import declared_vs_built as dvb
    mismatches = dvb.check(model, args, parser=None)   # -> list[dvb.Mismatch]; [] == clean
    dvb.refuse_on_mismatch(model, args, parser=None, where="train",
                           forbid_kinds=())   # SystemExit on any; a refcv7 launch
                                              # passes forbid_kinds=("drivort",)
    dvb.coverage(parser)      # -> dests of `parser` that have NO registry entry
    dvb.register(dest, kind, check=None, reason="")   # a NEW lever's entry (see below)

* ``model`` — the BUILT ``RefCV3Model`` (perception branch attached, withheld bank applied), i.e.
  the object the first step will run. ``args`` — the parsed Namespace AFTER the trainer's pins.
* ``parser`` — the trainer's ``build_parser()``. When given, every ``dest`` must have an entry:
  ⭐ **a trainer flag without a G-DVB entry is refused** (SPEC_REFCV7 §2), so a new lever cannot
  land without saying how its built counterpart is read.

Each entry has a ``kind``:

* ``built``  — a reader on the built modules and a LITERAL expectation from argv;
* ``loss``   — a loss weight: the head it supervises must be built iff the weight is > 0, and
  the model must carry the weight the loss actually reads;
* ``elsewhere`` — verified on a built object by a NAMED existing guard (the reason names it);
* ``data`` / ``runtime`` / ``record`` — not a model lever (paths, schedules, logging, stamps).
* ``drivort`` — a DrivoR-T flag (SPEC_REFCV7 §6.1: NOT refcv7, pending a rename): no internals
  are read; ``forbid_kinds=("drivort",)`` refuses any set away from its literal default.

⛔ EXPECTATIONS ARE LITERALS. Every expected value is written here from ``args`` by a rule stated
in this file — never by calling ``_pin_trainer_cfg``, ``refcv6_flags_from_args`` or any other
function under test, and never by reading the config object that built the model. A cross-check
derived from the code it checks measures determinism, not correctness.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Callable

from tanitad.train.config_hygiene import undeclared_attributes

__all__ = ["Mismatch", "Lever", "REGISTRY", "check", "refuse_on_mismatch", "coverage",
           "register", "KINDS", "DRIVORT_DEFAULTS", "drivort_levers_set",
           "REFCV7_REQUIRED_ON", "check_refcv7_required"]

KINDS = ("built", "loss", "elsewhere", "data", "runtime", "record", "drivort")


@dataclass(frozen=True)
class Mismatch:
    """One declared-vs-built disagreement. ``declared`` is what argv says, ``built`` what the
    model holds, ``read_from`` the attribute path the built value was read from."""
    lever: str
    declared: Any
    built: Any
    read_from: str
    why: str = ""

    def __str__(self) -> str:
        return (f"{self.lever}: declared {self.declared!r} but BUILT {self.built!r} "
                f"(read from {self.read_from})" + (f" -- {self.why}" if self.why else ""))


@dataclass(frozen=True)
class Lever:
    dest: str
    kind: str
    check: Callable[[Any, Any], list] | None = None
    reason: str = ""
    #: ``drivort`` entries only: the argparse default, written as a LITERAL -- the value the
    #: flag must hold on a refcv7 launch (SPEC_REFCV7 §6.1).
    default: Any = None


REGISTRY: dict[str, Lever] = {}


def register(dest: str, kind: str, check: Callable | None = None, reason: str = "") -> None:
    """Add (or replace) the entry for one argparse ``dest``.

    ⛔ ``built`` / ``loss`` entries need a ``check(model, args) -> list[Mismatch]``; the other
    kinds need a ``reason`` a reader can verify. A future arm (e.g. refcv7's residual prior)
    registers its levers here, in its own module, before the trainer calls :func:`check`.
    """
    if kind not in KINDS:
        raise ValueError(f"G-DVB kind {kind!r} not in {KINDS}")
    if kind == "drivort":
        raise ValueError("`drivort` entries are fixed by DRIVORT_DEFAULTS, not registered")
    if kind in ("built", "loss") and check is None:
        raise ValueError(f"G-DVB `{dest}` is a {kind} lever and has no check")
    if kind not in ("built", "loss") and not reason:
        raise ValueError(f"G-DVB `{dest}` ({kind}) needs a reason a reader can verify")
    REGISTRY[dest] = Lever(dest, kind, check, reason)


# ============================================================================
# readers — the BUILT side. Each reads a module attribute, never a config input.
# ============================================================================

def _core(m):
    return getattr(m, "core", m)


def _dec(m):
    return _core(m).decoder


def _enc(m):
    return _core(m).encoder


def _is_timm(m) -> bool:
    return type(_enc(m)).__name__ == "TimmResNetTrunk"


def _a(args, name, default=None):
    return getattr(args, name, default)


def _flag(o) -> str:
    return "--" + o.replace("_", "-")


def _eq(dest, declared, built, where, why="") -> list:
    return [] if declared == built else [Mismatch(_flag(dest), declared, built, where, why)]


def _near(dest, declared, built, where, why="", tol=1e-9) -> list:
    try:
        ok = abs(float(declared) - float(built)) <= tol
    except (TypeError, ValueError):
        ok = False
    return [] if ok else [Mismatch(_flag(dest), declared, built, where, why)]


# ============================================================================
# the TRUNK (FIX-3 lives here)
# ============================================================================

_TRUNK_CLASS = {"timm": "TimmResNetTrunk", "refc": "ResNetEncoder"}


def _c_trunk(m, a):
    want = _TRUNK_CLASS[str(_a(a, "trunk", "refc"))]
    return _eq("trunk", want, type(_enc(m)).__name__, "core.encoder (class)",
               "the record would claim a trunk the weights do not have")


def _timm_cfg(dest, attr, want_fn, why=""):
    def chk(m, a):
        if not _is_timm(m):
            return []
        return _eq(dest, want_fn(a), getattr(_enc(m).cfg, attr, "<absent>"),
                   f"core.encoder.cfg.{attr}", why)
    return chk


def _c_equalize(m, a):
    """FIX-3 (D-REFCV6-EQUALIZE-DROPPED): the rows the TRUNK zeroes and the rows the LIFT marks
    unobserved are BOTH the argv value — the lever is one decision with two consumers."""
    n = int(_a(a, "equalize_bottom_rows", 0) or 0)
    out = []
    if _is_timm(m):
        out += _eq("equalize_bottom_rows", n,
                   int(getattr(_enc(m).cfg, "equalize_bottom_rows", -1)),
                   "core.encoder.cfg.equalize_bottom_rows",
                   "the trunk does not zero the rows argv names (D-REFCV6-EQUALIZE-DROPPED)")
    elif n > 0:
        out.append(Mismatch("--equalize-bottom-rows", n, 0, f"core.encoder ({type(_enc(m)).__name__})",
                            "the in-repo `refc` trunk implements no equalisation"))
    bank = getattr(m, "_lift_bank", None)
    if bank is not None:
        out += _eq("equalize_bottom_rows", n, int(getattr(bank, "equalize_bottom_rows", -1)),
                   "model._lift_bank.equalize_bottom_rows", "the BEV lift mask disagrees")
    return out


def _c_image_hw(m, a):
    hw = _a(a, "image_hw")
    if not hw:
        return []
    want = (int(hw[0]), int(hw[1]))
    if _is_timm(m):
        return _eq("image_hw", want, tuple(int(v) for v in _enc(m).cfg.image_hw),
                   "core.encoder.cfg.image_hw")
    got = tuple(int(v) for v in _core(m).cfg.encoder.image_hw())
    return _eq("image_hw", want, got, "core.cfg.encoder.image_hw() (the model-held config)")


def _c_trunk_in_channels(m, a):
    n = int(_a(a, "trunk_in_channels", 0) or 0)
    if n <= 0:
        return []
    if _is_timm(m):
        return _eq("trunk_in_channels", n, int(_enc(m).cfg.in_channels),
                   "core.encoder.cfg.in_channels")
    import torch.nn as nn
    stem = next((x for x in _enc(m).modules() if isinstance(x, nn.Conv2d)), None)
    return _eq("trunk_in_channels", n, None if stem is None else int(stem.in_channels),
               "core.encoder first Conv2d .in_channels")


def _levers(m) -> dict:
    return dict(getattr(_enc(m), "memory_levers", {}) or {})


def _c_trunk_lever(dest, key, want_fn, is_count=False):
    def chk(m, a):
        want = want_fn(a)
        if not _is_timm(m):
            return ([] if not want else
                    [Mismatch(_flag(dest), want, None, "core.encoder.memory_levers",
                              "the in-repo `refc` trunk has no such lever")])
        got = _levers(m).get(key, 0 if is_count else False)
        if is_count:                                  # a COUNT > 0 is the fact that it happened
            got = bool(int(got or 0) > 0)
        return _eq(dest, want, got, f"core.encoder.memory_levers[{key!r}]")
    return chk


def _c_trunk_compile(m, a):
    want = bool(_a(a, "trunk_compile", False))
    if not _is_timm(m):
        return [] if not want else [Mismatch("--trunk-compile", True, None, "core.encoder",
                                             "the in-repo `refc` trunk cannot be compiled")]
    return _eq("trunk_compile", want, getattr(_enc(m), "_net_fn", None) is not None,
               "core.encoder._net_fn is not None")


def _c_trunk_bn_recalib(m, a):
    # recalibration itself runs after config.json (`_bn_recalib_start`); what must be BUILT is
    # its precondition -- the BN frozen, or the recalibrated statistics are overwritten.
    if int(_a(a, "trunk_bn_recalib", 0) or 0) <= 0:
        return []
    return _eq("trunk_bn_recalib", True, bool(_levers(m).get("frozen_bn", False)),
               "core.encoder.memory_levers['frozen_bn']",
               "recalibration without frozen BN is overwritten by BN momentum")


# ============================================================================
# the DECODER — sampler, refcv6 F1..F9, anchors, selection
# ============================================================================

def _c_sampler(m, a):
    d = _dec(m)
    want = str(_a(a, "sampler", "none"))
    built = {k: getattr(d, k, None) is not None for k in ("control_head", "time_mlp", "sched")}
    out = _eq("sampler", want, str(getattr(d.cfg, "sampler", "none")), "core.decoder.cfg.sampler")
    live = want != "none"
    for k, on in built.items():
        if on != live:
            out.append(Mismatch("--sampler", want, f"decoder.{k} {'BUILT' if on else 'None'}",
                                f"core.decoder.{k}", "a denoiser the record does not describe"))
    return out


def _dec_cfg(dest, attr, want_fn):
    def chk(m, a):
        return _eq(dest, want_fn(a), getattr(_dec(m).cfg, attr, "<absent>"),
                   f"core.decoder.cfg.{attr}")
    return chk


#: argparse dest -> DiffusionFlags field. ⛔ Written out, NOT read from
#: `refcv6_flags_from_args`: that function is the thing under test.
_RV6_FIELD = {
    "f1_random_t": "f1_random_t", "f1_t_max": "f1_t_max", "f2_dd_step": "f2_dd_step",
    "f3_per_layer": "f3_per_layer", "f4_adaln": "f4_adaln", "f4_zero_init": "f4_zero_init",
    "f5_emitting_conf": "f5_emitting_conf", "f5_refuse_blind_rank": "f5_refuse_blind_rank",
    "f5_focal": "f5_focal", "f6_w_u0_zero": "f6_w_u0_zero",
    "f7_samples_per_anchor": "f7_samples_per_anchor", "f7_ack_eval_join": "f7_ack_eval_join",
    "f8_flat_noise": "f8_flat_waypoint_noise", "f9_assert_vocab": "f9_assert_vocab",
}
#: the argv DEFAULT of each F-dest (argparse's), literal
_RV6_DEFAULT = {"f1_t_max": 50, "f7_samples_per_anchor": 1}


def _c_rv6(dest):
    field = _RV6_FIELD[dest]

    def chk(m, a):
        want = _a(a, dest, _RV6_DEFAULT.get(dest, False))
        want = type(_RV6_DEFAULT.get(dest, False))(want)
        rv6 = getattr(_dec(m), "rv6", None)
        return _eq(dest, want, getattr(rv6, field, "<no rv6>"), f"core.decoder.rv6.{field}")
    return chk


def _c_f3_structure(m, a):
    """F3's per-stage outputs PRESENT: the cascade built with one head per layer, AND the model's
    forward passes `layer_u0_hat` / `layer_logits` through (D-REFCV6-F3-WHITELIST)."""
    if not bool(_a(a, "f3_per_layer", False)):
        return _eq("f3_per_layer", False, getattr(_dec(m), "cascade", None) is not None,
                   "core.decoder.cascade is not None")
    d = _dec(m)
    out = []
    casc = getattr(d, "cascade", None)
    if casc is None:
        return [Mismatch("--f3-per-layer", True, None, "core.decoder.cascade",
                         "the per-layer heads were never built")]
    n_layers = len(getattr(d, "layers", []))
    heads = getattr(casc, "control_heads", None)
    if heads is not None:
        out += _eq("f3_per_layer", n_layers, len(heads),
                   "len(core.decoder.cascade.control_heads) vs len(core.decoder.layers)",
                   "one cascade stage per decoder layer")
    passthrough = tuple(getattr(type(_core(m)), "DECODER_PASSTHROUGH", ()))
    for key in ("layer_u0_hat", "layer_logits"):          # literal: the loss reads these two
        if key not in passthrough:
            out.append(Mismatch("--f3-per-layer", f"{key} in the forward output", "absent",
                                f"{type(_core(m)).__name__}.DECODER_PASSTHROUGH",
                                "the per-stage loss would be skipped (D-REFCV6-F3-WHITELIST)"))
    return out


def _c_f4_structure(m, a):
    want = bool(_a(a, "f4_adaln", False))
    d = _dec(m)
    ad = getattr(d, "adaln", None)
    out = _eq("f4_adaln", want, ad is not None, "core.decoder.adaln is not None")
    if want and ad is not None:
        out += _eq("f4_adaln", len(getattr(d, "layers", [])), len(ad),
                   "len(core.decoder.adaln) vs len(core.decoder.layers)")
    return out


def _c_n_anchors(m, a):
    n = _a(a, "n_anchors")
    if not n:
        return []
    return _eq("n_anchors", int(n), int(_dec(m).anchors.shape[0]), "core.decoder.anchors.shape[0]")


def _c_anchor_v0(m, a):
    d = _dec(m)
    want = bool(_a(a, "anchor_v0_conditioned", False))
    out = _eq("anchor_v0_conditioned", want, bool(getattr(d, "anchor_v0_cond", False)),
              "core.decoder.anchor_v0_cond")
    if want:
        out += _near("anchor_ref_speed", float(_a(a, "anchor_ref_speed", 10.0)),
                     getattr(d, "anchor_ref_speed", None), "core.decoder.anchor_ref_speed")
        units = _a(a, "anchor_control_units") or "kappa"
        out += _eq("anchor_control_units", str(units), getattr(d, "anchor_control_units", None),
                   "core.decoder.anchor_control_units")
    return out


def _c_sel(dest, attr, want_fn):
    def chk(m, a):
        want = want_fn(a)
        if want is None:
            return []
        got = getattr(getattr(_dec(m), "sel", None), attr, "<absent>")
        if isinstance(want, float):
            return _near(dest, want, got, f"core.decoder.sel.{attr}")
        return _eq(dest, want, got, f"core.decoder.sel.{attr}")
    return chk


def _c_sel_emitted_t(m, a):
    if not bool(_a(a, "sel_score_emitted", False)):
        return []
    return _eq("sel_score_emitted_t", int(_a(a, "sel_score_emitted_t", -1)),
               getattr(getattr(_dec(m), "sel", None), "score_emitted_t", "<absent>"),
               "core.decoder.sel.score_emitted_t")


# ---- FIX-4: the refcv6 SELECTION mechanisms ------------------------------------------------ #
#: the keys `RefCV3Model.provenance_roles()["selection_mechanisms_built"]` must carry, and the
#: argv flag that asks for each (literal). `None` = no flag: present whenever v6 is on.
SELECTION_KEYS = {"tac8_prior": "graft_tac8_prior", "behaviour_set": "graft_behaviour_sel",
                  "nav_compliance": "graft_nav_compliance", "speed_ceiling": "speed_ceiling_filter"}


def _sel_built(m) -> dict:
    """G-DVB's OWN reading of the decoder — independent of `refc_v3.refcv6_selection_built`."""
    d = _dec(m)
    return {
        "tac8_prior": getattr(d, "tac8_lat_to_anchor", None) is not None,
        "behaviour_set": (bool(getattr(d, "graft_behaviour_sel", False))
                          and getattr(m, "tac_behaviour_gate_v6", None) is not None),
        "nav_compliance": getattr(d, "navc_gate", None) is not None,
        "speed_ceiling": (bool(getattr(d, "speed_ceiling_filter", False))
                          and getattr(m, "max_speed_1h_v6", None) is not None),
    }


def _c_selection(dest):
    key = {v: k for k, v in SELECTION_KEYS.items()}[dest]

    def chk(m, a):
        want = bool(_a(a, dest, False))
        out = _eq(dest, want, _sel_built(m)[key], f"selection mechanism `{key}` (built)",
                  "a selection mechanism the ranked score does not have (D-REFCV6-CONFIG-BUILD)"
                  if want else "a live selection mechanism no flag asked for")
        if dest == "speed_ceiling_filter":
            # PI 2026-09-26 A2: INFERENCE-ONLY. The class switch that restores the pre-ruling
            # in-training mask exists for the regression test alone; a built decoder carrying it
            # is refused.
            out += _eq(dest, False, bool(getattr(_dec(m), "speed_ceiling_in_training", False)),
                       "core.decoder.speed_ceiling_in_training",
                       "the ceiling would mask the TRAINING selection (the LAW input)")
        return out
    return chk


def _c_navc_tau_file(m, a):
    """SPEC_REFCV7 §7: the BUILT tolerance against the BANKED file's τ -- a literal read from the
    file, never from the argv float the file exists to verify."""
    p = _a(a, "nav_compliance_tau_file", None)
    if not p:
        return []
    import json as _json
    try:
        with open(p, encoding="utf-8") as fh:
            want = float(_json.load(fh)["tau"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return [Mismatch(_flag("nav_compliance_tau_file"), p, None, "the banked tau file",
                         f"unreadable ({type(exc).__name__})")]
    return _near("nav_compliance_tau_file", want, getattr(_dec(m), "navc_tau_rad", None),
                 "core.decoder.navc_tau_rad", "the BUILT tolerance must be the banked one",
                 tol=1e-12)


def _c_navc_tau(m, a):
    if not bool(_a(a, "graft_nav_compliance", False)):
        return []
    return _near("nav_compliance_tau_rad", float(_a(a, "nav_compliance_tau_rad", 0.0) or 0.0),
                 getattr(_dec(m), "navc_tau_rad", None), "core.decoder.navc_tau_rad")


def _c_selection_declaration(m, a) -> list:
    """The DECLARATION (`provenance_roles`, written to config.json) must name exactly the
    mechanisms that are built — the static-list defect (D-REFCV6-CONFIG-BUILD)."""
    if getattr(m, "tac_decoder_v6", None) is None or not hasattr(m, "provenance_roles"):
        return []
    roles = m.provenance_roles()
    declared = roles.get("selection_mechanisms_built")
    if not isinstance(declared, dict):
        return [Mismatch("provenance_roles", "selection_mechanisms_built {key: bool}",
                         type(declared).__name__, "model.provenance_roles()",
                         "config.json cannot say which selection mechanisms were built")]
    out = []
    for key, built in _sel_built(m).items():
        if bool(declared.get(key, not built)) != built:
            out.append(Mismatch(f"provenance_roles[{key}]", declared.get(key), built,
                                "model.provenance_roles()['selection_mechanisms_built']",
                                "config.json would declare a selection mechanism the decoder "
                                "does not have (or hide one it has)"))
    return out


# ============================================================================
# the hierarchy / v3 model levers
# ============================================================================

def _c_arm(m, a):
    return _eq("arm", str(_a(a, "arm")) == "hier", bool(getattr(m.cfg, "hier", None)),
               "model.cfg.hier")


def _c_mod(dest, attr, want_fn, owner=lambda m: m, where=None):
    """``want_fn(args)`` -> is the module ``owner(model).attr`` expected to be BUILT?"""
    def chk(m, a):
        want = bool(want_fn(a))
        return _eq(dest, want, getattr(owner(m), attr, None) is not None,
                   where or f"model.{attr} is not None")
    return chk


def _c_attr(dest, attr, want_fn, owner=lambda m: m, where=None, tol=None):
    def chk(m, a):
        want = want_fn(a)
        if want is None:
            return []
        got = getattr(owner(m), attr, "<absent>")
        if tol is not None:
            return _near(dest, want, got, where or f"model.{attr}", tol=tol)
        return _eq(dest, want, got, where or f"model.{attr}")
    return chk


def _c_ego_history(m, a):
    want = bool(_a(a, "ego_history", False))
    eh = getattr(_core(m), "ego_hist", None)
    out = _eq("ego_history", want, eh is not None, "core.ego_hist is not None")
    if want and eh is not None:
        c = getattr(eh, "cfg", None)
        out += _eq("ego_history_kind", str(_a(a, "ego_history_kind", "gru")),
                   getattr(c, "kind", None), "core.ego_hist.cfg.kind")
        out += _eq("ego_history_hidden", int(_a(a, "ego_history_hidden", 64)),
                   getattr(c, "hidden", None), "core.ego_hist.cfg.hidden")
        out += _eq("ego_history_out", int(_a(a, "ego_history_out", 32)),
                   getattr(c, "out_dim", None), "core.ego_hist.cfg.out_dim")
    return out


def _c_ego_valid_channel(m, a):
    want = bool(_a(a, "ego_valid_channel", False) or _a(a, "ego_state_inject", False))
    if not want:
        return []                           # a size config may carry it on by itself
    return _eq("ego_valid_channel", True, bool(getattr(_core(m).cfg, "ego_valid_channel", False)),
               "core.cfg.ego_valid_channel")


def _c_tac_v6(m, a):
    want = bool(_a(a, "tac_decoder_v6", False))
    td = getattr(m, "tac_decoder_v6", None)
    out = _eq("tac_decoder_v6", want, td is not None, "model.tac_decoder_v6 is not None")
    if want and td is not None:
        out += _eq("tac_decoder_d_bev", int(_a(a, "tac_decoder_d_bev", 0) or 0),
                   int(getattr(td.cfg, "d_bev", -1)), "model.tac_decoder_v6.cfg.d_bev")
        out += _eq("tac_decoder_bev_detach", bool(_a(a, "tac_decoder_bev_detach", False)),
                   bool(getattr(m.cfg, "tac_decoder_bev_detach", None)),
                   "model.cfg.tac_decoder_bev_detach (read by the forward)")
        out += _near("tac_decoder_valid_threshold",
                     float(_a(a, "tac_decoder_valid_threshold", 0.5)),
                     getattr(m.cfg, "tac_decoder_valid_threshold", None),
                     "model.cfg.tac_decoder_valid_threshold (read by the forward)")
    return out


def _c_goal_point(m, a):
    inj = bool(_a(a, "goal_point_inject", False))
    out = _eq("goal_point_inject", inj, getattr(m, "gp_head", None) is not None,
              "model.gp_head is not None")
    geo = inj and bool(_a(a, "goal_point_geo_prior", False))
    out += _eq("goal_point_geo_prior", geo, getattr(_dec(m), "gp_point_gate", None) is not None,
               "core.decoder.gp_point_gate is not None")
    if inj:
        out += _near("goal_point_t", float(_a(a, "goal_point_t", 4.0)),
                     getattr(getattr(m.cfg, "goal_point_cfg", None), "t_goal_s", None),
                     "model.cfg.goal_point_cfg.t_goal_s")
    return out


def _c_perception(m, a):
    w_map = float(_a(a, "w_map", 0.0) or 0.0)
    w_b3d = float(_a(a, "w_box3d", 0.0) or 0.0)
    br = getattr(m, "_perception", None)
    out = _eq("w_map", (w_map > 0.0 or w_b3d > 0.0), br is not None,
              "model._perception is not None", "a perception head with no live weight, or a "
              "live weight with no head")
    out += _near("w_map", w_map, getattr(m, "_w_map", None), "model._w_map (read by the loss)")
    out += _near("w_box3d", w_b3d, getattr(m, "_w_box3d", None),
                 "model._w_box3d (read by the loss)")
    out += _eq("w_map", w_map > 0.0, getattr(m, "_lift_bank", None) is not None,
               "model._lift_bank is not None", "the BEV map is lifted only behind a live map weight")
    return out


def _c_agents(m, a):
    mode = str(_a(a, "agents", "off"))
    core = _core(m)
    out = []
    for attr in ("agent_head", "agent_embed"):
        out += _eq("agents", mode != "off", getattr(core, attr, None) is not None,
                   f"core.{attr} is not None")
    lay = [getattr(ly, "cross_agent", None) is not None for ly in getattr(_dec(m), "layers", [])]
    if lay:
        out += _eq("agents", mode != "off", all(lay),
                   "every core.decoder.layers[i].cross_agent is not None",
                   "agent tokens reach no decoder layer" if mode != "off" else
                   "agent cross-attention no flag asked for")
    return out


def _c_loss_weight(dest, model_attr, head_fn=None, head_where=""):
    """heads <-> loss weights: the weight the LOSS reads, and the head it supervises."""
    def chk(m, a):
        w = float(_a(a, dest, 0.0) or 0.0)
        out = _near(dest, w, getattr(m, model_attr, None), f"model.{model_attr} (read by the loss)")
        if head_fn is not None:
            out += _eq(dest, w > 0.0, bool(head_fn(m)), head_where,
                       "a built head with no live weight gets NO gradient (the tac_goal_tok_head "
                       "defect), and a live weight with no head supervises nothing")
        return out
    return chk


def _c_w_u0(m, a):
    w = float(_a(a, "w_u0", 0.0) or 0.0)
    if w <= 0.0:
        return []
    return _eq("w_u0", True, getattr(_dec(m), "control_head", None) is not None,
               "core.decoder.control_head is not None",
               "an x0 loss with no control head to supervise")


def _c_withheld(m, a):
    d = _dec(m)
    # ⚠️ Before step 1 the bank is FIXED whenever a warm-up is set: `_apply_withheld_bank`
    # writes "fixed" and the loop switches the mode on at the warm-up step.
    mode = str(_a(a, "withheld_bank", "fixed"))
    want = "fixed" if int(_a(a, "withheld_bank_warmup", 0) or 0) > 0 else mode
    out = _eq("withheld_bank", want,
              str(getattr(d, "anchor_withheld_bank", None)), "core.decoder.anchor_withheld_bank")
    out += _near("withheld_speed_max", float(_a(a, "withheld_speed_max", 35.0)),
                 getattr(d, "anchor_withheld_speed_max", None),
                 "core.decoder.anchor_withheld_speed_max")
    return out


def _c_wp_index(m, a):
    on = str(_a(a, "wp_index", "off")) != "off"
    mods = [getattr(ly, "wp_index", None) for ly in getattr(_dec(m), "layers", [])]
    built = [x is not None for x in mods]
    out = _eq("wp_index", on, bool(built) and all(built), "core.decoder.layers[i].wp_index")
    if on and all(built) and mods:
        c = getattr(mods[0], "cfg", None)
        out += _eq("wp_index_mode", str(_a(a, "wp_index_mode", "geom")), getattr(c, "mode", None),
                   "core.decoder.layers[0].wp_index.cfg.mode")
        out += _eq("wp_index_detach", bool(_a(a, "wp_index_detach", False)),
                   getattr(c, "detach", None), "core.decoder.layers[0].wp_index.cfg.detach")
        out += _near("wp_index_radius_m", float(_a(a, "wp_index_radius_m", 0.0)),
                     getattr(c, "radius_m", None), "core.decoder.layers[0].wp_index.cfg.radius_m")
        out += _eq("wp_index_hidden", int(_a(a, "wp_index_hidden", 32)),
                   getattr(c, "hidden", None), "core.decoder.layers[0].wp_index.cfg.hidden")
        out += _near("wp_index_scale_m", float(_a(a, "wp_index_scale_m", 10.0)),
                     getattr(c, "scale_m", None), "core.decoder.layers[0].wp_index.cfg.scale_m")
    return out


def _c_bev_coupling(m, a):
    on = bool(_a(a, "bev_coupling", False))
    fn = getattr(_dec(m), "bev_coupling_provenance", None)
    prov = fn() if callable(fn) else {"enabled": False}
    out = _eq("bev_coupling", on, bool(prov.get("enabled", False)),
              "core.decoder.bev_coupling_provenance()['enabled']")
    if on and prov.get("enabled"):
        out += _eq("bev_coupling_learned_offsets", bool(_a(a, "bev_coupling_learned_offsets", False)),
                   prov.get("learned_offsets"), "bev_coupling_provenance()['learned_offsets']")
        out += _near("bev_coupling_offset_max_m", float(_a(a, "bev_coupling_offset_max_m", 2.0)),
                     prov.get("offset_max_m"), "bev_coupling_provenance()['offset_max_m']")
    return out


def _c_bev_aux(m, a):
    on = str(_a(a, "bev_aux", "off")) != "off"
    return _eq("bev_aux", on, getattr(_core(m), "bev_aux_head", None) is not None,
               "core.bev_aux_head is not None")


def _c_graft_lan(m, a):
    on = bool(_a(a, "graft_lan", False))
    return _eq("graft_lan", on, getattr(_dec(m), "lan_gate", None) is not None,
               "core.decoder.lan_gate is not None")


def _c_residual_prior(m, a):
    """refcv7 NEW-1 (SPEC_REFCV7 section 1). The mode is read from the BUILT decoder
    (``AnchoredDiffusionDecoder.residual_prior``, set at construction and the attribute every
    forward branches on) AND from its config; a residual build must also carry the three
    mechanisms the prior composes with. Expectation: the LITERAL argv value."""
    want = str(_a(a, "residual_prior", "off"))
    d = _dec(m)
    out = _eq("residual_prior", want, str(getattr(d, "residual_prior", "<absent>")),
              "core.decoder.residual_prior")
    out += _eq("residual_prior", want, str(getattr(d.cfg, "residual_prior", "<absent>")),
               "core.decoder.cfg.residual_prior")
    if want != "off":
        for ok, where, why in (
                (bool(getattr(d, "anchor_v0_cond", False)), "core.decoder.anchor_v0_cond",
                 "the residual vocabulary is rolled per window from v0"),
                (getattr(d, "time_mlp", None) is not None, "core.decoder.time_mlp",
                 "the residual IS the DDIM sampler's state"),
                (getattr(_core(m), "ego_hist", None) is not None, "core.ego_hist",
                 "the prior reads the observed pose window")):
            if not ok:
                out.append(Mismatch("--residual-prior", want, f"{where} missing", where, why))
        passthrough = tuple(getattr(type(_core(m)), "DECODER_PASSTHROUGH", ()))
        # literal: the cascade re-roll reads the first two, G-LIVE the third
        for key in ("residual_prior_ctrl", "residual_prior_v", "residual_prior_path"):
            if key not in passthrough:
                out.append(Mismatch("--residual-prior", f"{key} in the forward output",
                                    "absent", f"{type(_core(m)).__name__}.DECODER_PASSTHROUGH",
                                    "the F3 re-roll could not compose on P (the A16 class)"))
    return out


# ============================================================================
# THE REGISTRY — every trainer dest, one entry each (coverage is tested)
# ============================================================================

def _b(dest, fn):
    register(dest, "built", fn)


# -- trunk --
_b("trunk", _c_trunk)
_b("trunk_name", _timm_cfg("trunk_name", "model_name", lambda a: str(_a(a, "trunk_name"))))
_b("trunk_mode", _timm_cfg("trunk_mode", "mode", lambda a: str(_a(a, "trunk_mode", "shared"))))
_b("trunk_fuse", _timm_cfg("trunk_fuse", "fuse", lambda a: str(_a(a, "trunk_fuse", "concat1x1"))))
_b("trunk_fuse_plain_init", _timm_cfg("trunk_fuse_plain_init", "fuse_identity_init",
                                      lambda a: not bool(_a(a, "trunk_fuse_plain_init", False)),
                                      "the fusion init is the CONTROL arm's lever"))
_b("trunk_pretrained", _timm_cfg("trunk_pretrained", "pretrained",
                                 lambda a: True if _a(a, "trunk_pretrained") is None
                                 else bool(_a(a, "trunk_pretrained")),
                                 "ImageNet vs random init is the knockout arm"))
_b("trunk_in_channels", _c_trunk_in_channels)
_b("equalize_bottom_rows", _c_equalize)
_b("image_hw", _c_image_hw)
_b("trunk_chunk_ckpt", lambda m, a: (
    [] if not _is_timm(m) and not int(_a(a, "trunk_chunk_ckpt", 0) or 0) else
    _eq("trunk_chunk_ckpt", int(_a(a, "trunk_chunk_ckpt", 0) or 0),
        int(_levers(m).get("chunk_ckpt", 0) or 0), "core.encoder.memory_levers['chunk_ckpt']")))
_b("trunk_frozen_bn", _c_trunk_lever("trunk_frozen_bn", "frozen_bn",
                                     lambda a: bool(_a(a, "trunk_frozen_bn", False))))
_b("trunk_bf16", _c_trunk_lever("trunk_bf16", "bf16", lambda a: bool(_a(a, "trunk_bf16", False))))
_b("trunk_channels_last", _c_trunk_lever("trunk_channels_last", "channels_last",
                                         lambda a: bool(_a(a, "trunk_channels_last", False))))
_b("trunk_fold_bn", _c_trunk_lever("trunk_fold_bn", "bn_folded",
                                   lambda a: bool(_a(a, "trunk_fold_bn", False)), is_count=True))
_b("trunk_dedup_frames", _c_trunk_lever("trunk_dedup_frames", "dedup_frames",
                                        lambda a: bool(_a(a, "trunk_dedup_frames", False))))
_b("trunk_compile", _c_trunk_compile)
_b("trunk_bn_recalib", _c_trunk_bn_recalib)

# -- decoder / sampler / refcv6 F-flags --
_b("sampler", _c_sampler)
_b("sampler_space", _dec_cfg("sampler_space", "sampler_space",
                             lambda a: str(_a(a, "sampler_space", "control"))))
_b("sampler_train_t_max", _dec_cfg("sampler_train_t_max", "sampler_train_t_max",
                                   lambda a: int(_a(a, "sampler_train_t_max", 50))))
_b("sampler_infer_t", _dec_cfg("sampler_infer_t", "sampler_infer_t",
                               lambda a: int(_a(a, "sampler_infer_t", 8))))
_b("sampler_steps", _dec_cfg("sampler_steps", "sampler_steps",
                             lambda a: int(_a(a, "sampler_steps", 2))))
_b("sampler_groups", _dec_cfg("sampler_groups", "sampler_groups",
                              lambda a: int(_a(a, "sampler_groups", 1))))
for _d in _RV6_FIELD:
    _b(_d, _c_rv6(_d))
_f3_flag = REGISTRY["f3_per_layer"].check
_b("f3_per_layer", lambda m, a, _f=_f3_flag: _f(m, a) + _c_f3_structure(m, a))
_f4_flag = REGISTRY["f4_adaln"].check
_b("f4_adaln", lambda m, a, _f=_f4_flag: _f(m, a) + _c_f4_structure(m, a))
_b("n_anchors", _c_n_anchors)
_b("anchor_v0_conditioned", _c_anchor_v0)
register("anchor_control_units", "elsewhere", reason=(
    "checked with --anchor-v0-conditioned against core.decoder.anchor_control_units (G-DVB), "
    "and against the artifact's own declaration by `_read_anchor_artifact`"))
register("anchor_ref_speed", "elsewhere", reason=(
    "checked with --anchor-v0-conditioned against core.decoder.anchor_ref_speed (G-DVB)"))
register("anchors", "elsewhere", reason=(
    "the vocabulary FILE: `_check_anchor_artifact_against_cfg` refuses a file/config mismatch and "
    "`_anchor_stamp` records the BUILT buffer by sha256 (`model.core.decoder.anchors`)"))
_b("sel_refined", _c_sel("sel_refined", "refined", lambda a: bool(_a(a, "sel_refined", False))))
_b("sel_score_emitted", _c_sel("sel_score_emitted", "score_emitted",
                               lambda a: bool(_a(a, "sel_score_emitted", False))))
_b("sel_score_emitted_t", _c_sel_emitted_t)
_b("sel_accel_max", _c_sel("sel_accel_max", "accel_max",
                           lambda a: None if _a(a, "sel_accel_max") is None
                           else float(_a(a, "sel_accel_max"))))
for _d, _attr, _dflt in (("feasible_decode", "feasible_decode", False),
                         ("feasible_entry", "feasible_entry", False),
                         ("feasible_mu", "feasible_mu", 0.7),
                         ("feasible_a_max", "feasible_a_max", 4.0),
                         ("feasible_kappa_max", "feasible_kappa_max", 0.2),
                         ("feasible_prefix_slots", "feasible_prefix_slots", 4)):
    _b(_d, _dec_cfg(_d, _attr, lambda a, _d=_d, _t=type(_dflt), _v=_dflt: _t(_a(a, _d, _v))))
for _d in SELECTION_KEYS.values():
    _b(_d, _c_selection(_d))
_b("nav_compliance_tau_rad", _c_navc_tau)
_b("nav_compliance_tau_file", _c_navc_tau_file)
_b("bev_coupling", _c_bev_coupling)
register("bev_coupling_learned_offsets", "elsewhere", reason="checked with --bev-coupling (G-DVB)")
register("bev_coupling_offset_max_m", "elsewhere", reason="checked with --bev-coupling (G-DVB)")
_b("wp_index", _c_wp_index)
for _d in ("wp_index_mode", "wp_index_detach", "wp_index_radius_m", "wp_index_hidden",
           "wp_index_scale_m"):
    register(_d, "elsewhere", reason="checked with --wp-index against the built layer's cfg "
                                     "(G-DVB); refused as a dead knob with --wp-index off")
register("wp_index_const_xy", "elsewhere", reason=(
    "the `const` CONTROL's address, refused as a dead knob off that mode (`_pin_refcv5_seams`) "
    "and stamped in `seams.wp_index`; checked by `assert_seams_are_built` (mode)"))
_b("withheld_bank", _c_withheld)
register("withheld_speed_max", "elsewhere", reason="checked with --withheld-bank (G-DVB)")
register("withheld_bank_warmup", "runtime", reason=(
    "a STEP threshold applied inside the loop (the bank goes live after N steps)"))
register("w_u0", "loss", _c_w_u0)
register("ack_ddim_no_u0", "record", reason=(
    "an OPERATOR acknowledgement; stamped as seams.u0_absent_under_ddim"))
register("mode", "runtime", reason=(
    "the in-training eval's decode mode (classifier vs diffusion steps) -- a forward ARGUMENT, "
    "not a module"))

# -- hierarchy / v3 --
_b("arm", _c_arm)
register("size", "elsewhere", reason=(
    "the RUNG: a whole-model choice; the built ledger `param_breakdown_v3` is stamped and the "
    "hier/flat pair is pinned by `REGISTERED_DELTA_KEYS` (C122)"))
register("smoke", "elsewhere", reason="the CI rung; same ledger as --size")
_b("no_strategic", _c_attr("no_strategic", "no_strategic",
                           lambda a: bool(_a(a, "no_strategic", False)),
                           owner=lambda m: _core(m).cfg,
                           where="core.cfg.no_strategic (the forward gate reads it)"))
_b("ego_history", _c_ego_history)
# refcv7 NEW-1: the plan is a residual on a causal kinematic prior
_b("residual_prior", _c_residual_prior)
for _d in ("ego_history_kind", "ego_history_hidden", "ego_history_out"):
    register(_d, "elsewhere", reason="checked with --ego-history against core.ego_hist.cfg (G-DVB)")
_b("ego_state_inject", _c_mod("ego_state_inject", "ego_inj",
                              lambda a: bool(_a(a, "ego_state_inject", False))))
_b("echo_base", _c_attr("echo_base", "echo_base", lambda a: bool(_a(a, "echo_base", False)),
                        owner=lambda m: m.cfg, where="model.cfg.echo_base (read by the forward)"))
_b("ego_valid_channel", _c_ego_valid_channel)
_b("ego_dropout", _c_attr("ego_dropout", "ego_dropout",
                          lambda a: None if _a(a, "ego_dropout") is None
                          else float(_a(a, "ego_dropout")),
                          owner=lambda m: _core(m).cfg, where="core.cfg.ego_dropout", tol=1e-9))
_b("nav_args", _c_mod("nav_args", "nav_arg_proj", lambda a: bool(_a(a, "nav_args", False))))
_b("max_speed_input", _c_mod("max_speed_input", "max_speed_cond",
                             lambda a: bool(_a(a, "max_speed_input", False))))
register("max_speed_mode", "elsewhere", reason=(
    "refused as INERT without --max-speed-input (`_pin_trainer_cfg`); the conditioner carries it"))
_b("max_speed_input_v6", _c_mod("max_speed_input_v6", "max_speed_1h_v6",
                                lambda a: bool(_a(a, "max_speed_input_v6", False))))
_b("goal_point_inject", _c_goal_point)
register("goal_point_geo_prior", "elsewhere", reason="checked with --goal-point-inject (G-DVB)")
register("goal_point_t", "elsewhere", reason="checked with --goal-point-inject (G-DVB)")
register("goal_point_w", "loss", _c_loss_weight("goal_point_w", "_w_goal_point",
                                  lambda m: getattr(m, "gp_head", None) is not None,
                                  "model.gp_head is not None"))
_b("tac_goal_tok_head", _c_mod("tac_goal_tok_head", "tac_goal_tok_head",
                               lambda a: bool(_a(a, "tac_goal_tok_head", False))))
register("w_tac_goal", "loss", _c_loss_weight("w_tac_goal", "_w_tac_goal"))
_b("tac_decoder_v6", _c_tac_v6)
for _d in ("tac_decoder_d_bev", "tac_decoder_bev_detach", "tac_decoder_valid_threshold"):
    register(_d, "elsewhere", reason="checked with --tac-decoder-v6 against the built decoder (G-DVB)")
register("w_tac_v6", "loss", _c_loss_weight("w_tac_v6", "_w_tac_v6",
                              lambda m: getattr(m, "tac_decoder_v6", None) is not None,
                              "model.tac_decoder_v6 is not None"))
# ⛔⛔ DrivoR-T (SPEC_REFCV7 §6.1, PI 2026-09-26): `--refcv7` / `--w-r7-*` / `--r7-*` are the
# 2026-09-19 DrivoR-T draft (`Project Steering/SPEC_DRIVORT.md`), NOT refcv7, and are pending a
# mechanical rename to `drivort_*`. G-DVB reads NONE of their internals; it lists them with the
# literal default each must hold on a refcv7 launch. `check(..., forbid_kinds=("drivort",))`
# refuses any that is set; `train()` itself keeps DrivoR-T's own model-vs-weight refusal.
DRIVORT_DEFAULTS: dict[str, Any] = {
    "refcv7": False, "w_r7_wta": 0.0, "w_r7_scorer": 0.0, "r7_queries": 64, "r7_d_model": 256,
    "r7_layers": 4, "r7_heads": 8, "r7_img_pool": 2, "r7_no_select": False,
    "r7_scorer_self_attn": False, "r7_nav_tau_rad": 0.0, "r7_n_perturb": 32, "r7_toad": False,
    "r7_toad_iters": 5, "r7_toad_samples": 64, "r7_toad_seed": 0, "r7_w_ttc": 5.0,
    "r7_w_ep": 5.0, "r7_w_comf": 2.0,
}
for _d, _v in DRIVORT_DEFAULTS.items():
    REGISTRY[_d] = Lever(_d, "drivort", None, (
        "DrivoR-T (SPEC_DRIVORT.md), not refcv7: OFF on a refcv7 launch (SPEC_REFCV7 §6.1); "
        "pending the refcv7_* -> drivort_* rename"), _v)
register("goal_str", "elsewhere", reason=(
    "a LOSS switch on the strategic LAN goal; stamped as seams.goal_str_loss_applied and "
    "refused against --no-strategic (`route_loss_respects_strategic_bypass`)"))
_b("graft_lan", _c_graft_lan)
register("lan_arclengths", "elsewhere", reason=(
    "the LAN label ladder; pinned onto core.lan (`LanConfig(k=len(...))`) when LAN is on"))
register("lan_min_lead_m", "data", reason="a LAN LABEL derivation parameter (dataset side)")

# -- agents / perception / bev aux --
_b("agents", _c_agents)
register("w_agent", "loss", _c_loss_weight("w_agent", "_w_agent"))
_b("agent_cls_weight", lambda m, a: _eq(
    "agent_cls_weight", str(_a(a, "agent_cls_weight", "off")) != "off",
    getattr(m, "_cls_class_weight", None) is not None, "model._cls_class_weight is not None"))
for _d in ("agent_queries", "agent_w_project", "agent_w_ground", "agent_rig_camera",
           "agent_cam_height", "agent_sigma_range", "agent_miss_rate", "agent_pad",
           "agent_presence_hard"):
    register(_d, "elsewhere", reason=(
        "an agent-seam knob: every value must be recoverable from `seams.agent_knobs` "
        "(`assert_knobs_stamped`), and the seam itself is built-checked by --agents (G-DVB) and "
        "`assert_seams_are_built`"))
_b("w_map", _c_perception)
register("w_box3d", "elsewhere", reason="checked with --w-map against model._perception (G-DVB)")
_b("map_lift_valid_mask", _c_attr("map_lift_valid_mask", "_map_lift_valid_mask",
                                  lambda a: bool(_a(a, "map_lift_valid_mask", True))))
_b("box3d_visible_filter", _c_attr("box3d_visible_filter", "_box3d_visible_filter",
                                   lambda a: bool(_a(a, "box3d_visible_filter", True))))
_b("bev_aux", _c_bev_aux)
register("w_bev_aux", "loss", _c_loss_weight("w_bev_aux", "_w_bev_aux"))
_b("bev_aux_shuffle", _c_attr("bev_aux_shuffle", "_bev_shuffle",
                              lambda a: bool(_a(a, "bev_aux_shuffle", False))))
for _d in ("bev_aux_occlusion", "bev_aux_detach", "bev_aux_rng", "bev_aux_rmax",
           "bev_aux_pos_weight", "bev_aux_hidden", "bev_aux_dtok"):
    register(_d, "elsewhere", reason=(
        "a BEV-aux knob pinned onto core.bev_aux (BEVAuxConfig) and stamped in seams.bev_aux; the "
        "head itself is built-checked by --bev-aux (G-DVB)"))
register("conflict_detector", "runtime", reason=(
    "an INSTRUMENT built after config.json (`GradientConflictDetector.for_model`); refused when "
    "`on` with no perception term, and its config is stamped as `conflict_detector`"))
register("conflict_mode", "runtime", reason="the conflict detector's mode (stamped)")
register("conflict_every", "runtime", reason="the conflict detector's cadence (stamped)")
register("grad_probe_modules", "runtime", reason="logging only (per-module grad_abs_sum rows)")
register("ablate_frames", "runtime", reason=(
    "a gate CONTROL applied to the BATCH (frames zeroed), stamped as `ablate_frames`"))

# -- optimiser / schedule --
for _d in ("opt", "weight_decay", "encoder_lr_mult", "lr"):
    register(_d, "elsewhere", reason=(
        "the OPTIMIZER that will run is checked against the recipe by "
        "`assert_optimizer_matches_recipe` after `build_optimizer`"))
register("warmup", "runtime", reason="the lr schedule (applied per step by `apply_lr_schedule`)")
register("steps", "runtime", reason="the run length")
register("batch", "runtime", reason="the batch size (a data-loader argument)")
register("seed", "runtime", reason="the RNG seed (recorded; data order is ResumableEpochSampler)")
register("cudnn_benchmark", "runtime", reason="a backend setting, stamped as READ BACK")

# -- data / labels / sidecars / IO --
for _d, _why in (
        ("data_root", "the corpus path"), ("v2_cache", "the corpus path"),
        ("prefetch_factor", "a data-loader argument"), ("u8_batches", "the in-flight dtype"),
        ("eval_cache", "the eval corpus path"), ("eval_labels", "the eval label blob"),
        ("clip_clock_sidecar", "the label CLOCK source; G3 checks the clock it produces"),
        ("label_clock_max_unverified", "G3's coverage cap (an operator decision, stamped)"),
        ("v7_labels", "the train label blob"), ("cot_negative_sidecar", "a label sidecar"),
        ("tac_goal_negatives", "the label NEGATIVE policy"), ("v2_lru", "a cache size"),
        ("allow_eval_clips_in_train", "a corpus-split override (stamped)"),
        ("require_parity", "a corpus-parity refusal"), ("synth_episodes", "the CI corpus"),
        ("episodes", "a corpus subset size"), ("map_gt_root", "the SAM3 map GT path"),
        ("map_min_coverage", "a map-GT coverage filter"), ("map_lru", "a cache size"),
        ("join3d", "the 3-D box join path"), ("agent_join", "the agent label join path"),
        ("agent_join_no_rates", "a join reporting switch"),
        ("agent_join_verify", "the join digest check"),
        ("agent_join_allow_legacy_ids", "a join id override"),
        ("agent_rig_extrinsics", "the per-clip camera table (lift + projection geometry)"),
        ("agent_rig_extrinsics_allow_partial", "an extrinsics coverage override"),
        ("speed_max_sidecar_v6", "the max-speed INPUT sidecar (train)"),
        ("speed_max_sidecar_v6_eval", "the max-speed INPUT sidecar (eval)"),
        ("nav_from_v7", "the nav command SOURCE (labels)"),
        ("trunk_bn_recalib_seed", "which windows recalibrate BN (data)")):
    register(_d, "data", reason=_why)
for _d, _why in (
        ("eval_every", "in-training eval cadence"), ("eval_batches", "in-training eval size"),
        ("eval_window_dump", "an eval output path"), ("out", "the run directory"),
        ("workers", "data-loader workers"), ("log_every", "logging cadence"),
        ("save_every", "checkpoint cadence"), ("device", "the device"),
        ("preflight", "the preflight mode switch")):
    register(_d, "runtime", reason=_why)


# ============================================================================
# the API
# ============================================================================

def coverage(parser) -> list[str]:
    """Every ``dest`` of ``parser`` with no registry entry (``[]`` == every flag is covered)."""
    dests = sorted({a.dest for a in parser._actions if a.dest not in ("help",)})
    return [d for d in dests if d not in REGISTRY]


#: ⛔⛔ PI RULING 2026-09-26, verbatim: *"go with your recommendation for E1, assure that the three
#: selection mechanism are on"* (SPEC_REFCV7 §7 / A2, GOALS D-REFCV7-E1). The trainer's defaults
#: stay OFF (other arms bit-identical); a refcv7 launch must carry all three, built.
REFCV7_REQUIRED_ON: tuple[str, ...] = ("graft_tac8_prior", "graft_nav_compliance",
                                       "speed_ceiling_filter")


def check_refcv7_required(model, args, *, tau_file: str | None = None) -> list[Mismatch]:
    """-> a Mismatch for every REFCV7_REQUIRED_ON lever that is OFF in argv or UNBUILT in the
    model ([] == all three on and built). With ``tau_file`` (the banked
    `raw/nav_compliance_tau_train.json`), argv's `--nav-compliance-tau-rad` must equal its `tau`.
    """
    out: list[Mismatch] = []
    built = _sel_built(model)
    for dest in REFCV7_REQUIRED_ON:
        key = {v: k for k, v in SELECTION_KEYS.items()}[dest]
        if not bool(_a(args, dest, False)):
            out.append(Mismatch(_flag(dest), "ON (PI 2026-09-26, D-REFCV7-E1)", "OFF in argv",
                                "argv", "refcv7 requires all three selection mechanisms"))
        if not built[key]:
            out.append(Mismatch(_flag(dest), "BUILT", "not built",
                                f"selection mechanism `{key}` (built)",
                                "refcv7 requires all three selection mechanisms"))
    if tau_file is not None:
        import json as _json
        want = float(_json.load(open(tau_file, encoding="utf-8"))["tau"])
        got = float(_a(args, "nav_compliance_tau_rad", 0.0) or 0.0)
        if abs(got - want) > 1e-12:
            out.append(Mismatch("--nav-compliance-tau-rad", want, got, f"argv vs {tau_file}",
                                "tau must be the value derived on the TRAIN split"))
    return out


def drivort_levers_set(args) -> list[str]:
    """The DrivoR-T flags ``args`` sets away from their literal defaults (SPEC_REFCV7 §6.1)."""
    out = []
    for d, v in DRIVORT_DEFAULTS.items():
        got = getattr(args, d, v)
        if (abs(float(got) - float(v)) > 1e-12 if isinstance(v, float) else got != v):
            out.append(_flag(d))
    return out


def check(model, args, parser=None, *, forbid_kinds: tuple = ()) -> list[Mismatch]:
    """-> every declared-vs-built mismatch on the BUILT ``model``; ``[]`` == clean.

    ``forbid_kinds=("drivort",)`` -- what a refcv7 launch passes -- also refuses every DrivoR-T
    flag set away from its default (SPEC_REFCV7 §6.1). ``train()`` does not forbid them, so a
    DrivoR-T arm still trains under its own refusals until it is renamed.

    ⛔ Order of the report: unregistered flags first (a flag G-DVB cannot see is a hole in the
    guard itself), then every lever's mismatches in registry order, then the selection
    DECLARATION, then G-HYG on the config tree the model holds.
    """
    out: list[Mismatch] = []
    if parser is not None:
        for d in coverage(parser):
            out.append(Mismatch(_flag(d), "a G-DVB registry entry", "none",
                                "tanitad.train.declared_vs_built.REGISTRY",
                                "a trainer flag with no declared-vs-built entry is refused "
                                "(SPEC_REFCV7 §2): register it with its built reader"))
    for lever in list(REGISTRY.values()):
        if lever.kind in ("built", "loss") and lever.check is not None:
            try:
                out += list(lever.check(model, args) or [])
            except Exception as e:                  # a reader that cannot read is a FAIL
                out.append(Mismatch(_flag(lever.dest), "readable", f"{type(e).__name__}: {e}",
                                    "G-DVB reader", "the built value could not be read"))
    if "drivort" in tuple(forbid_kinds):
        for f in drivort_levers_set(args):
            d = f[2:].replace("-", "_")
            out.append(Mismatch(f, DRIVORT_DEFAULTS[d], getattr(args, d, None), "argv",
                                "a DrivoR-T lever on a refcv7 launch (SPEC_REFCV7 §6.1): "
                                "refcv7 passes none of them"))
    out += _c_selection_declaration(model, args)
    cfg = getattr(model, "cfg", None)
    if cfg is not None:
        for path, cls, attr in undeclared_attributes(cfg, "model.cfg"):
            out.append(Mismatch(f"{path}.{attr}", "a declared field", "an ad-hoc attribute",
                                f"{path} ({cls})", "G-HYG: dropped by the next rebuild"))
    return out


def refuse_on_mismatch(model, args, parser=None, where: str = "train", *,
                       forbid_kinds: tuple = ()) -> None:
    """``SystemExit`` naming EVERY mismatch, or return ``None`` when the build is clean."""
    bad = check(model, args, parser, forbid_kinds=forbid_kinds)
    if bad:
        raise SystemExit(
            f"[G-DVB] ⛔⛔ {where}: {len(bad)} lever(s) DECLARED in argv are NOT what the model "
            f"BUILT. Refusing before the first step -- a run that trains the other arm while "
            f"its record names this one is FALSE PROVENANCE (D-REFCV6-EQUALIZE-DROPPED, "
            f"D-REFCV6-F3-WHITELIST, D-REFCV6-CONFIG-BUILD):\n  - "
            + "\n  - ".join(str(b) for b in bad))


# ============================================================================
# D3 (2026-09-26): declared vs LOGGED -- the same defect class, one step later
# ============================================================================

def check_logged_rows(config: dict, rows) -> list[Mismatch]:
    """-> a Mismatch per key the run DECLARED it logs (config.json ``grad_reach_logging``) that a
    regular metrics row (one carrying ``loss``) does NOT carry, and per ``ga_*`` key in any row of
    a run that declared none ([] == the rows carry exactly what config.json declared).

    ⛔⛔ An instrument that is BUILT and DECLARED but never WRITES is the declared-vs-built defect
    one step later. MEASURED 2026-09-26 by the map-signal audit: refcv6-r101-s0 reports per-head
    gradient reach, and **0 of its 4,621 metrics rows carry a `ga_*` key** -- the row was filled
    on the pre-increment step and written on the post-increment one. Run this on a launch
    smoke's ``metrics.jsonl``; the trainer runs it itself on its FIRST log row and refuses.
    ⚠️ A config.json with NO declaration (a pre-D3 trainer) is itself a Mismatch: nothing in it
    says what should have been logged, so nothing can be shown to have been.
    """
    dec = (config or {}).get("grad_reach_logging")
    if not isinstance(dec, dict):
        return [Mismatch("grad_reach_logging", "a declaration in config.json", None,
                         "config.json", "pre-D3 trainer: nothing says what the run logs")]
    keys = [str(k) for k in (dec.get("keys") or [])]
    regular = [r for r in rows if isinstance(r, dict) and "loss" in r]
    out: list[Mismatch] = []
    if dec.get("declared"):
        if not keys:
            out.append(Mismatch("grad_reach_logging.keys", "the ga_* keys of the built heads",
                                [], "config.json", "declared ON with an empty key list"))
        if not regular:
            out.append(Mismatch("grad_reach_logging", "at least one metrics row", 0,
                                "metrics.jsonl",
                                "no regular (loss-carrying) row to hold the declaration against"))
        for r in regular:
            miss = [k for k in keys if k not in r]
            if miss:
                out.append(Mismatch("grad_reach_logging", keys, [k for k in keys if k in r],
                                    f"metrics.jsonl row step={r.get('step')}",
                                    f"declared ga_* keys MISSING: {miss}"))
    else:
        for r in rows:
            extra = sorted(k for k in (r if isinstance(r, dict) else {})
                           if str(k).startswith("ga_"))
            if extra:
                out.append(Mismatch("grad_reach_logging", "no ga_* key (declared OFF)", extra,
                                    f"metrics.jsonl row step={r.get('step')}",
                                    "a key the run did not declare: an OFF arm's schema must "
                                    "equal the pre-instrument trainer's"))
    return out
