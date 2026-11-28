"""G-DVB for ``refa_v1_train.py`` -- every argv lever equals the BUILT ``RefAV1``, or the run
refuses (PI directive 2026-09-27, item 4; the refav1 counterpart of
``tanitad.train.declared_vs_built``, which is written for ``refc_v3_train.py`` / ``RefCV3Model``).

⭐ WHY A SEPARATE REGISTRY AND NOT ENTRIES IN THE SHARED ONE. ``declared_vs_built.check`` runs
EVERY ``built`` / ``loss`` lever of its global ``REGISTRY`` on the model it is handed. A refav1
lever registered there would run its reader on a ``RefCV3Model`` at every refcv7 launch and FAIL
it ("the built value could not be read") -- a refav1 entry would break the binding refcv7 gate.
So this module keeps its OWN registry with the SAME contract (``Lever`` / ``Mismatch`` are
imported, not re-declared) and the SAME API names::

    from tanitad.train import declared_vs_built_refav1 as dvb1
    dvb1.check(model, args, parser=None)          # -> list[Mismatch]; [] == clean
    dvb1.refuse_on_mismatch(model, args, parser)  # SystemExit naming EVERY mismatch
    dvb1.coverage(parser)                         # dests with NO entry ([] == covered)

⛔ A TRAINER FLAG WITHOUT AN ENTRY IS REFUSED (the SPEC_REFCV7 §2 rule, applied to refav1):
``refa_v1_train.main`` calls :func:`refuse_on_mismatch` with its own parser before the first
step, so a new flag cannot land without saying how its built counterpart is read.

⛔ EXPECTATIONS ARE LITERALS FROM ``args``. Every expected value is written here from argv by a
rule stated in this file -- never by calling ``refa_v1_train.build_model`` /
``r1r6_cfg_kwargs`` (the functions under test). The BUILT side is read off modules first
(``model.traj_head is not None``, a Linear's ``in_features``); ``model.cfg`` is read only for
weights the loss itself reads from the config (``kind='loss'``), which IS the built value.

⚠️ SCOPE, STATED: this is the G-DVB entry set for refav1, not a refav1 PROFILE of the binding
launch gate (``stack/scripts/launch_gate.py``). The gate's G-LIVE / G-CKPT / G-EVAL / G-CLOCK
machinery drives ``refc_v3_train.py``'s internals (``trainer_capture`` patches that trainer's
functions), so a refav1 profile is a separate build -- delivered as a PROPOSAL with this module.
"""
from __future__ import annotations

from typing import Any, Callable

from tanitad.train.config_hygiene import undeclared_attributes
from tanitad.train.declared_vs_built import KINDS, Lever, Mismatch

__all__ = ["REGISTRY", "register", "check", "refuse_on_mismatch", "coverage", "NEW_2026_09_27"]

REGISTRY: dict[str, Lever] = {}

#: the flags the PI 2026-09-27 R1/R3/R5/R6 build added (listed so a test can require that every
#: one of them is a checked lever, not merely a covered name)
NEW_2026_09_27: tuple[str, ...] = ("strategic_off", "vmax_input", "w_goal", "goal_negatives",
                                   "cot_negative_sidecar", "w_speed_band", "w_traj", "r5_prior",
                                   "r5_kappa_source")


def register(dest: str, kind: str, check: Callable | None = None, reason: str = "") -> None:
    if kind not in KINDS or kind == "drivort":
        raise ValueError(f"refav1 G-DVB kind {kind!r} not in {KINDS} (drivort is refc-only)")
    if kind in ("built", "loss") and check is None:
        raise ValueError(f"refav1 G-DVB `{dest}` is a {kind} lever and has no check")
    if kind not in ("built", "loss") and not reason:
        raise ValueError(f"refav1 G-DVB `{dest}` ({kind}) needs a reason a reader can verify")
    REGISTRY[dest] = Lever(dest, kind, check, reason)


def _a(args, name, default=None):
    return getattr(args, name, default)


def _flag(dest: str) -> str:
    return "--" + dest.replace("_", "-")


def _eq(dest, declared, built, where, why="") -> list:
    return [] if declared == built else [Mismatch(_flag(dest), declared, built, where, why)]


def _near(dest, declared, built, where, why="", tol=1e-12) -> list:
    try:
        ok = abs(float(declared) - float(built)) <= tol
    except (TypeError, ValueError):
        ok = False
    return [] if ok else [Mismatch(_flag(dest), declared, built, where, why)]


def _hier(args) -> bool:
    return not bool(_a(args, "no_hierarchy", False))


# ============================================================================
# readers -- the BUILT side
# ============================================================================

def _c_no_hierarchy(m, a):
    want = _hier(a)
    return _eq("no_hierarchy", want, m.tactical_policy is not None,
               "model.tactical_policy is not None",
               "the hierarchy argv asks for is not the one built")


def _c_strategic_off(m, a):
    off = bool(_a(a, "strategic_off", False))
    out = []
    if not _hier(a):
        return out                            # refused by the trainer before this
    out += _eq("strategic_off", not off, m.strategic_policy is not None,
               "model.strategic_policy is not None",
               "R6: the strategic POLICY must be absent iff --strategic-off")
    out += _eq("strategic_off", not off, m.strategic is not None,
               "model.strategic is not None",
               "R6: the strategic SUBSPACE predictor must be absent iff --strategic-off")
    if off:
        out += _eq("strategic_off", True, m.tactical_policy is not None,
                   "model.tactical_policy is not None",
                   "R6 KEEPS the tactical layer (that is what separates it from --no-hierarchy)")
        out += _eq("strategic_off", True, getattr(m, "nav_inj_emb", None) is not None,
                   "model.nav_inj_emb is not None", "R6 KEEPS nav")
        for w in ("w_feat_str", "w_feat_str_ext", "w_str_label"):
            out += _near("strategic_off", 0.0, getattr(m.cfg, w, None), f"model.cfg.{w}",
                         "a strategic weight on a term that no longer exists")
    return out


def _c_vmax_input(m, a):
    want = bool(_a(a, "vmax_input", False))
    ctx, itn = getattr(m, "vmax_to_ctx", None), getattr(m, "vmax_to_intent", None)
    out = _eq("vmax_input", want, ctx is not None and itn is not None,
              "model.vmax_to_ctx / model.vmax_to_intent are not None",
              "R1: the max-speed input path argv declares is not built")
    if want and ctx is not None:
        out += _eq("vmax_input", 4, int(ctx.in_features), "model.vmax_to_ctx.in_features",
                   "the refcv6 4-way ladder")
        out += _eq("vmax_input", 4, int(itn.in_features), "model.vmax_to_intent.in_features",
                   "the refcv6 4-way ladder")
    traj = getattr(m, "traj_head", None)
    if want and traj is not None:
        # the R5 condition must carry the 4 max-speed slots (R1 reaches the trajectory too)
        out += _eq("vmax_input", int(m.traj_cond_dim()), int(traj.d_cond),
                   "model.traj_head.d_cond", "R1 must reach the R5 head")
    return out


def _c_loss_head(dest, attr, width_fn=None):
    def chk(m, a):
        w = float(_a(a, dest, 0.0) or 0.0)
        head = getattr(m, attr, None)
        out = _eq(dest, w > 0.0, head is not None, f"model.{attr} is not None",
                  "a loss weight > 0 with no head is inert; a head with weight 0 is untrained")
        out += _near(dest, w, getattr(m.cfg, dest, None), f"model.cfg.{dest} (read by the loss)")
        if head is not None and width_fn is not None:
            want_w, got_w, where = width_fn(m, head)
            out += _eq(dest, want_w, got_w, where)
        return out
    return chk


def _goal_width(m, head):
    from tanitad.data.v7_labels import TAC_GOAL_TOKENS
    return len(TAC_GOAL_TOKENS), int(head[-1].out_features), "model.goal_head[-1].out_features"


def _sb_width(m, head):
    return 2, int(head[-1].out_features), "model.speed_band_head[-1].out_features"


def _traj_width(m, head):
    return int(m.cfg.traj_steps) * 2, int(head.out.out_features), "model.traj_head.out.out_features"


def _c_r5_prior(m, a):
    want = str(_a(a, "r5_prior", "kdx"))
    if getattr(m, "traj_head", None) is None:
        # the trainer refuses a non-default prior without --w-traj; nothing is built to read
        return _eq("r5_prior", "kdx", want, "argv (no R5 head built)",
                   "a prior stamped with no head to sit under")
    return _eq("r5_prior", want, str(m.cfg.traj_prior), "model.cfg.traj_prior (read by "
               "RefAV1.traj_readout on every forward)")


def _c_r5_kappa_source(m, a):
    want = str(_a(a, "r5_kappa_source", "steer_t0"))
    if getattr(m, "traj_head", None) is None:
        return _eq("r5_kappa_source", "steer_t0", want, "argv (no R5 head built)",
                   "a kappa0 source stamped with no head to feed")
    return _eq("r5_kappa_source", want, str(m.cfg.traj_kappa_source),
               "model.cfg.traj_kappa_source (the loader is built from it: "
               "`r1r6_loader_kwargs`)")


def _c_cfg(dest, attr, want_fn, tol=None):
    def chk(m, a):
        want = want_fn(a)
        got = getattr(m.cfg, attr, "<absent>")
        if tol is not None:
            return _near(dest, want, got, f"model.cfg.{attr}", tol=tol)
        return _eq(dest, want, got, f"model.cfg.{attr}")
    return chk


def _c_motion_inject(m, a):
    return _eq("motion_inject", bool(_a(a, "motion_inject", False)),
               getattr(m, "motion_in", None) is not None, "model.motion_in is not None")


def _c_detach_aux(m, a):
    d = _a(a, "detach_aux", None)
    return _eq("detach_aux", True if d is None else bool(d),
               bool(m.cfg.detach_aux_targets), "model.cfg.detach_aux_targets")


def _c_bptt(m, a):
    want = int(_a(a, "bptt_truncate", 15))
    out = _eq("bptt_truncate", want, int(getattr(m.operative, "bptt_truncate", -1)),
              "model.operative.bptt_truncate")
    out += _eq("bptt_truncate", want, int(getattr(m.tactical, "bptt_truncate", -1)),
               "model.tactical.bptt_truncate")
    return out


def _c_speed_channel(m, a):
    want = 2 + (1 if bool(_a(a, "speed_channel", False)) else 0)
    return _eq("speed_channel", want, int(m.operative.act[0].in_features),
               "model.operative.act[0].in_features", "(a, kappa) [+ v/30]")


def _c_tmix(m, a):
    g = _a(a, "tmix_groups", None)
    want = int(m.cfg.d_state) if g is None else int(g)
    return _eq("tmix_groups", want, int(m.adapter.tmix.groups), "model.adapter.tmix.groups")


def _c_ema(m, a):
    return _eq("ema_targets", bool(_a(a, "ema_targets", False)),
               getattr(m, "ema", None) is not None, "model.ema is not None")


def _c_target_space(m, a):
    ts = str(_a(a, "target_space", "frozen"))
    out = _eq("target_space", ts, str(m.cfg.target_space), "model.cfg.target_space")
    out += _eq("target_space", ts == "frozen", getattr(m, "to_enc", None) is not None,
               "model.to_enc is not None", "the frozen-target readout")
    return out


def _c_proposal_k(m, a):
    k = int(_a(a, "proposal_k", 1))
    c = m.cfg
    out = _eq("proposal_k", k * int(c.plan_steps) * int(c.a_dim),
              int(m.proposal[-1].out_features), "model.proposal[-1].out_features")
    out += _eq("proposal_k", k > 1, getattr(m, "proposal_score", None) is not None,
               "model.proposal_score is not None")
    return out


# ============================================================================
# the registry -- one entry per `refa_v1_train.build_parser()` dest
# ============================================================================
register("no_hierarchy", "built", _c_no_hierarchy)
register("strategic_off", "built", _c_strategic_off)
register("vmax_input", "built", _c_vmax_input)
register("w_goal", "loss", _c_loss_head("w_goal", "goal_head", _goal_width))
register("w_speed_band", "loss", _c_loss_head("w_speed_band", "speed_band_head", _sb_width))
register("w_traj", "loss", _c_loss_head("w_traj", "traj_head", _traj_width))
register("r5_prior", "built", _c_r5_prior)
register("r5_kappa_source", "built", _c_r5_kappa_source)
register("goal_negatives", "elsewhere", reason=(
    "a LOADER policy (`RefAV1Windows(goal_negatives=)` -> `v7_labels.tactical_goal_targets`), "
    "not a model lever: the trainer refuses a non-default policy without --w-goal, fits the "
    "head's pos_weight / class mask under THIS policy (`set_goal_supervision`, buffers in the "
    "checkpoint) and stamps it as config.json goal_head.negatives"))
register("cot_negative_sidecar", "data", reason=(
    "the PI 2026-09-16 ruling as a file: `v7_labels.load_cot_negative_sidecar` refuses a sidecar "
    "built over another blob md5, and the trainer requires it iff --goal-negatives "
    "cot-absence-negative"))
register("w_cf", "loss", _c_cfg("w_cf", "w_cf", lambda a: float(_a(a, "w_cf", 0.0)), tol=1e-12))
register("cf_negs", "built", _c_cfg("cf_negs", "cf_negs", lambda a: int(_a(a, "cf_negs", 3))))
register("cf_at_step", "built", _c_cfg("cf_at_step", "cf_at_step",
                                       lambda a: int(_a(a, "cf_at_step", 4))))
register("motion_inject", "built", _c_motion_inject)
register("detach_aux", "built", _c_detach_aux)
register("w_sigreg", "loss", _c_cfg("w_sigreg", "w_sigreg",
                                    lambda a: float(_a(a, "w_sigreg", 0.0)), tol=1e-12))
register("var_floor", "loss", _c_cfg("var_floor", "var_floor",
                                     lambda a: float(_a(a, "var_floor", 0.0)), tol=1e-12))
register("min_participation", "built", _c_cfg("min_participation", "min_participation",
                                              lambda a: float(_a(a, "min_participation", 0.0)),
                                              tol=1e-12))
register("bptt_truncate", "built", _c_bptt)
register("speed_channel", "built", _c_speed_channel)
register("tmix_groups", "built", _c_tmix)
register("ema_targets", "built", _c_ema)
register("ema_decay", "built", _c_cfg("ema_decay", "ema_decay",
                                      lambda a: float(_a(a, "ema_decay", 0.996)), tol=1e-12))
register("ema_decay_end", "built", _c_cfg("ema_decay_end", "ema_decay_end",
                                          lambda a: float(_a(a, "ema_decay_end", 0.999)),
                                          tol=1e-12))
register("target_space", "built", _c_target_space)
register("w_aux_head", "loss", _c_cfg("w_aux_head", "w_aux_head",
                                      lambda a: float(_a(a, "w_aux_head", 0.0)), tol=1e-12))
register("proposal_k", "built", _c_proposal_k)
for _d, _why in (
        ("cache", "the stage-1 feature cache; its geometry is refused at load (`verify_cache`)"),
        ("episodes", "the v2ep kinematics dir; the loader refuses a mis-gridded episode"),
        ("labels", "the v7.2 label blob; its md5 is stamped by `load_v7_labels` (join_report)"),
        ("nav", "the nav source; its md5 + the oracle stamp are in join_report"),
        ("allow_unlabelled", "a REFUSAL switch (real cache without --labels), not a lever")):
    register(_d, "data", reason=_why)
for _d, _why in (
        ("lru", "episodes held in RAM by the loader (a memory setting)"),
        ("out", "the run directory"), ("steps", "the run length"),
        ("bs", "the batch size (a loader argument)"),
        ("lr", "the AdamW base lr (param group 1), stamped in config.json args"),
        ("adapter_lr_mult", "the adapter param group's lr multiplier (optimizer)"),
        ("log_every", "logging cadence"), ("save_every", "checkpoint cadence"),
        ("seed", "the RNG seed (recorded)"), ("resume", "continue from <out>/ckpt.pt "
                                                        "(strict load, `load_resume_state`)"),
        ("smoke", "the CPU SmokeData rung (tiny dims, synthetic tensors) -- never a registered arm"),
        ("device", "the torch device"),
        ("precision", "the forward+loss autocast (stamped READ BACK by apply_precision_flags)"),
        ("tf32", "a backend switch (stamped READ BACK)"),
        ("clip", "the grad-norm clip (optimizer side)"),
        ("skip_nonfinite", "an optimizer-step guard (counted in every log row)")):
    register(_d, "runtime", reason=_why)


# ============================================================================
# the API
# ============================================================================

def coverage(parser) -> list[str]:
    """Every ``dest`` of ``parser`` with no registry entry (``[]`` == every flag is covered)."""
    dests = sorted({a.dest for a in parser._actions if a.dest not in ("help",)})
    return [d for d in dests if d not in REGISTRY]


def check(model, args, parser=None) -> list[Mismatch]:
    """-> every declared-vs-built mismatch on the BUILT ``RefAV1``; ``[]`` == clean.

    Order: unregistered flags first (a flag this guard cannot see is a hole in the guard), then
    each lever in registry order, then G-HYG on the model's config tree."""
    out: list[Mismatch] = []
    if parser is not None:
        for d in coverage(parser):
            out.append(Mismatch(_flag(d), "a refav1 G-DVB registry entry", "none",
                                "tanitad.train.declared_vs_built_refav1.REGISTRY",
                                "a trainer flag with no declared-vs-built entry is refused: "
                                "register it with its built reader"))
    for lever in list(REGISTRY.values()):
        if lever.kind in ("built", "loss") and lever.check is not None:
            try:
                out += list(lever.check(model, args) or [])
            except Exception as e:                  # a reader that cannot read is a FAIL
                out.append(Mismatch(_flag(lever.dest), "readable", f"{type(e).__name__}: {e}",
                                    "refav1 G-DVB reader", "the built value could not be read"))
    cfg = getattr(model, "cfg", None)
    if cfg is not None:
        for path, cls, attr in undeclared_attributes(cfg, "model.cfg"):
            out.append(Mismatch(f"{path}.{attr}", "a declared field", "an ad-hoc attribute",
                                f"{path} ({cls})", "G-HYG: dropped by the next rebuild"))
    return out


def refuse_on_mismatch(model, args, parser=None, where: str = "refa_v1_train") -> None:
    """``SystemExit`` naming EVERY mismatch, or ``None`` when the build is clean."""
    bad = check(model, args, parser)
    if bad:
        raise SystemExit(
            f"[G-DVB refav1] ⛔⛔ {where}: {len(bad)} lever(s) DECLARED in argv are NOT what the "
            f"model BUILT. Refusing before the first step -- a run that trains another arm "
            f"while its record names this one is FALSE PROVENANCE:\n  - "
            + "\n  - ".join(str(b) for b in bad))
