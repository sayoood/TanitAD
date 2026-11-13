"""K2DC (PREREG_K2DC_2026-11-12): a map-dose SHARE CONTROLLER for ``refc_v3_train.py``.

K2M1N20 trained at ``--w-map-hires 25``; the map term then took 0.39 / 0.81 / 0.77 of the trunk gradient at steps
1k / 3k / 5k against design 2.2's 0.05-0.15 band (MEASURED, banked 28913706): the map gradient per unit weight grows
~5x, so no constant weight holds the band. This controller holds it: every ``--map-hires-share-every`` steps it reads
the trainer's own gradient-share statistic (``grad_share.measure``, on its OWN schedule and its OWN ``dcp_`` prefix, so
the ``gs_*`` rows are untouched) and sets the map weight to the k whose predicted share equals the target.

The maths (pure, no torch). With ``a`` the map term's gradient on the group at the CURRENT weight ``w``, ``b`` the
rest, ``g = a + b``, the probe reports ``|a|``, ``|g|`` and ``proj = <a, g> / |g|^2``:

    A2 = |a|^2,  T2 = |g|^2,  AB = <a, b> = proj * T2 - A2,  B2 = |b|^2 = T2 - A2 - 2 AB
    per unit weight  a1 = a / w:  |a1|^2 = A2 / w^2,  <a1, b> = AB / w
    s(k) = (k^2 |a1|^2 + k <a1, b>) / (k^2 |a1|^2 + 2 k <a1, b> + |b|^2)        (the share at weight k)

``solve`` bisects s(k) = target on [LO, HI] and clamps to the end whose share is nearest when no root is inside.
"""
from __future__ import annotations

import math
from typing import Optional

__all__ = ["FLAGS", "DEFAULTS", "GROUPS", "PROBE_PREFIX", "STATE_KEY", "recover", "share_at", "solve",
           "parse_bounds", "is_on", "pin_args", "MapDoseController", "controller_for", "ckpt_extra", "resume_guard",
           "register_dvb_levers", "dvb_check", "add_arguments", "groups_for", "probe_before_forward",
           "read_after_losses", "apply_weight", "dc_reading_rows"]

FLAGS = ("map_hires_share_target", "map_hires_share_every", "map_hires_share_group", "map_hires_w_bounds")
DEFAULTS = {"map_hires_share_target": None, "map_hires_share_every": 250, "map_hires_share_group": "trunk",
            "map_hires_w_bounds": "1,25"}
GROUPS = ("trunk", "layer2")
#: the probe's metrics-row prefix: NOT ``gs_`` -- the gs instrument's rows must stay byte-for-byte what they are
PROBE_PREFIX = "dcp_"
STATE_KEY = "map_dose_ctl"


def add_arguments(g) -> None:
    """The four flags, added to an argparse GROUP (the trainer passes its gradient-share group ``g8``)."""
    g.add_argument("--map-hires-share-target", type=float, default=None,
                   help="K2DC: hold the 10 cm map term's share of the trunk gradient at this value (0 < t < 1) by "
                        "re-solving --w-map-hires every --map-hires-share-every steps. Default None = OFF (the tip's "
                        "constant weight, nothing built). --w-map-hires is then the INITIAL weight (steps before the "
                        "first firing run at it); needs --map-hires on and map_hires in --grad-share-terms.")
    g.add_argument("--map-hires-share-every", type=int, default=250,
                   help="K2DC: the controller fires on rows that are multiples of this many steps (>= 1, a multiple of "
                        "--log-every, <= --steps), on its OWN schedule, independent of --grad-share-at.")
    g.add_argument("--map-hires-share-group", type=str, default="trunk", choices=GROUPS,
                   help="K2DC: the parameter group the share is read on (grad_share.GROUP_PREFIXES): trunk | layer2.")
    g.add_argument("--map-hires-w-bounds", type=str, default="1,25",
                   help="K2DC: LO,HI -- the solved weight is clamped to [LO, HI]; 0 < LO <= HI; the initial "
                        "--w-map-hires must lie inside.")


# ---------------------------------------------------------------------------------------------------------------- #
# the maths                                                                                                         #
# ---------------------------------------------------------------------------------------------------------------- #
def recover(norm_a: float, norm_total: float, proj_a: float, w: float) -> dict:
    """``(|a|, |g|, proj)`` read at weight ``w`` -> ``{A2, T2, AB, B2, a1sq, a1b}`` (exact algebra; see the module doc)."""
    A2 = float(norm_a) ** 2
    T2 = float(norm_total) ** 2
    AB = float(proj_a) * T2 - A2
    B2 = T2 - A2 - 2.0 * AB
    w = float(w)
    return {"A2": A2, "T2": T2, "AB": AB, "B2": B2, "a1sq": A2 / (w * w), "a1b": AB / w}


def share_at(k: float, a1sq: float, a1b: float, B2: float) -> float:
    """s(k): the map term's projection share of the gradient at weight ``k``."""
    k = float(k)
    num = k * k * a1sq + k * a1b
    den = k * k * a1sq + 2.0 * k * a1b + B2
    return num / den if den > 0.0 else float("nan")


def solve(target: float, a1sq: float, a1b: float, B2: float, lo: float, hi: float, iters: int = 200) -> tuple:
    """-> ``(k, clamped)``: the k in [lo, hi] with ``share_at(k) == target`` by bisection; when no sign change brackets
    the target, the end whose share is nearest (``clamped`` True). A non-finite share at either end -> ``(nan, True)``."""
    f_lo = share_at(lo, a1sq, a1b, B2) - target
    f_hi = share_at(hi, a1sq, a1b, B2) - target
    if not (math.isfinite(f_lo) and math.isfinite(f_hi)):
        return float("nan"), True
    if f_lo == 0.0:
        return float(lo), False
    if f_hi == 0.0:
        return float(hi), False
    if (f_lo < 0.0) == (f_hi < 0.0):                       # no root inside: the end nearest the target
        return (float(lo) if abs(f_lo) <= abs(f_hi) else float(hi)), True
    a, b, fa = float(lo), float(hi), f_lo
    for _ in range(int(iters)):
        m = 0.5 * (a + b)
        fm = share_at(m, a1sq, a1b, B2) - target
        if not math.isfinite(fm):
            return float("nan"), True
        if fm == 0.0:
            return m, False
        if (fm < 0.0) == (fa < 0.0):
            a, fa = m, fm
        else:
            b = m
        if b - a <= 1e-15 * max(1.0, abs(b)):
            break
    return 0.5 * (a + b), False


def parse_bounds(spec) -> tuple:
    """``"LO,HI"`` -> ``(lo, hi)`` floats; ValueError on anything but 0 < LO <= HI, both finite."""
    parts = [p.strip() for p in str(spec).split(",")]
    if len(parts) != 2:
        raise ValueError(f"{spec!r} is not LO,HI")
    lo, hi = float(parts[0]), float(parts[1])
    if not (math.isfinite(lo) and math.isfinite(hi) and 0.0 < lo <= hi):
        raise ValueError(f"{spec!r}: need 0 < LO <= HI, both finite")
    return lo, hi


# ---------------------------------------------------------------------------------------------------------------- #
# argv refusals                                                                                                     #
# ---------------------------------------------------------------------------------------------------------------- #
def is_on(args) -> bool:
    return getattr(args, "map_hires_share_target", None) is not None


def pin_args(args) -> None:
    """``SystemExit`` unless the controller's declaration is coherent. OFF (target None) with any other K2DC flag set away
    from its default is refused too: a declared controller that is not built is the defect class this gate exists for."""
    if not is_on(args):
        stray = [f for f in FLAGS[1:] if getattr(args, f, DEFAULTS[f]) != DEFAULTS[f]]
        if stray:
            raise SystemExit("[k2dc] ⛔ --%s set but --map-hires-share-target is unset: the controller would be declared "
                             "in argv and never built." % "/--".join(s.replace("_", "-") for s in stray))
        return
    t = float(args.map_hires_share_target)
    if not (math.isfinite(t) and 0.0 < t < 1.0):
        raise SystemExit(f"[k2dc] ⛔ --map-hires-share-target {t!r} must satisfy 0 < target < 1.")
    if str(getattr(args, "map_hires", "off")) != "on":
        raise SystemExit("[k2dc] ⛔ --map-hires-share-target needs --map-hires on (there is no map term to dose).")
    w = float(getattr(args, "w_map_hires", 0.0) or 0.0)
    if not (w > 0.0):
        raise SystemExit("[k2dc] ⛔ --map-hires-share-target needs --w-map-hires > 0 (the INITIAL weight).")
    every = int(args.map_hires_share_every)
    if every < 1:
        raise SystemExit(f"[k2dc] ⛔ --map-hires-share-every {every} must be >= 1.")
    log_every = max(1, int(getattr(args, "log_every", 50) or 50))
    if every % log_every != 0:
        raise SystemExit(f"[k2dc] ⛔ --map-hires-share-every {every} must be a multiple of --log-every {log_every}: the "
                         f"firing's dc_* row would be written on a step the loop does not log.")
    steps = int(getattr(args, "steps", 0) or 0)
    if steps and every > steps:
        raise SystemExit(f"[k2dc] ⛔ --map-hires-share-every {every} > --steps {steps}: the controller would never fire.")
    if str(args.map_hires_share_group) not in GROUPS:
        raise SystemExit(f"[k2dc] ⛔ --map-hires-share-group {args.map_hires_share_group!r} not in {GROUPS}.")
    try:
        lo, hi = parse_bounds(args.map_hires_w_bounds)
    except ValueError as e:
        raise SystemExit(f"[k2dc] ⛔ --map-hires-w-bounds: {e}")
    if not (lo <= w <= hi):
        raise SystemExit(f"[k2dc] ⛔ --w-map-hires {w:g} (the initial weight) is outside --map-hires-w-bounds [{lo:g}, {hi:g}].")
    from tanitad.train import grad_share as _gs
    try:
        keep = _gs.parse_keep(getattr(args, "grad_share_terms", None))
    except ValueError as e:
        raise SystemExit(f"[k2dc] ⛔ --grad-share-terms: {e}")
    if keep is not None and "map_hires" not in keep:
        raise SystemExit("[k2dc] ⛔ --map-hires-share-target needs `map_hires` in --grad-share-terms (the share it holds is the "
                         "map term's; the launch gate's SHARE-ALARM reads the same term).")


# ---------------------------------------------------------------------------------------------------------------- #
# the controller                                                                                                    #
# ---------------------------------------------------------------------------------------------------------------- #
class MapDoseController:
    """Holds ``w_eff`` (the weight the map term runs at). ``fire(probe, row_step)`` reads a ``dcp_`` probe row, solves
    for the new weight, stores it in ``w_eff`` and returns the ``dc_*`` log row. The TRAINER applies ``w_eff`` to
    ``model._w_map_hires`` after the step's backward, so the new weight acts from the next step."""

    def __init__(self, target: float, every: int, group: str, lo: float, hi: float, w_init: float):
        self.target, self.every, self.group = float(target), int(every), str(group)
        self.lo, self.hi = float(lo), float(hi)
        self.w_init = float(w_init)
        self.w_eff = float(w_init)
        self.n_fired = 0

    def due(self, row_step: int) -> bool:
        return int(row_step) % self.every == 0

    def fire(self, probe: dict, row_step: int) -> dict:
        p = PROBE_PREFIX + self.group
        try:
            na, nt, pr = (float(probe[f"{p}_norm_map_hires"]), float(probe[f"{p}_norm_total"]),
                          float(probe[f"{p}_proj_map_hires"]))
        except KeyError as e:
            raise SystemExit(f"[k2dc] ⛔ the controller's probe row carries no {e.args[0]!r} (keys: {sorted(probe)[:8]}...): "
                             f"the group {self.group!r} or the map term was not read -- refusing to hold a dose on nothing.")
        w0 = self.w_eff
        rec = recover(na, nt, pr, w0)
        k, clamped = float("nan"), True
        pred = float("nan")
        if all(math.isfinite(v) for v in (na, nt, pr)) and rec["a1sq"] > 0.0 and rec["B2"] >= 0.0:
            k, clamped = solve(self.target, rec["a1sq"], rec["a1b"], rec["B2"], self.lo, self.hi)
            if math.isfinite(k):
                pred = share_at(k, rec["a1sq"], rec["a1b"], rec["B2"])
        if math.isfinite(k):
            self.w_eff = float(k)
        self.n_fired += 1
        return {"dc_w_before": float(w0), "dc_w_eff": float(self.w_eff), "dc_share_read": float(pr),
                "dc_share_pred": float(pred), "dc_clamped": float(1.0 if clamped else 0.0),
                "dc_target": self.target}

    # -- checkpoint ------------------------------------------------------------------------------------------- #
    def config(self) -> dict:
        return {"target": self.target, "every": self.every, "group": self.group, "lo": self.lo, "hi": self.hi,
                "w_init": self.w_init}

    def state_dict(self) -> dict:
        return dict(self.config(), w_eff=float(self.w_eff), n_fired=int(self.n_fired))

    def load_state_dict(self, st: dict) -> None:
        for k, v in self.config().items():
            if k not in st or (abs(float(st[k]) - v) > 1e-12 if k not in ("group",) else str(st[k]) != v):
                raise SystemExit(f"[k2dc] ⛔ resume REFUSED: ckpt.pt's controller {k} is {st.get(k)!r}, this launch's is {v!r}.")
        w = float(st["w_eff"])
        if not (math.isfinite(w) and self.lo <= w <= self.hi):
            raise SystemExit(f"[k2dc] ⛔ resume REFUSED: ckpt.pt's w_eff {w!r} is outside [{self.lo:g}, {self.hi:g}].")
        self.w_eff = w
        self.n_fired = int(st.get("n_fired", 0))


def controller_for(args) -> Optional[MapDoseController]:
    """The controller the argv declares, or ``None`` (OFF). ``pin_args`` has already refused an incoherent argv."""
    if not is_on(args):
        return None
    lo, hi = parse_bounds(args.map_hires_w_bounds)
    return MapDoseController(args.map_hires_share_target, args.map_hires_share_every, args.map_hires_share_group,
                             lo, hi, float(args.w_map_hires))


def ckpt_extra(model) -> dict:
    """The key every ``ckpt.pt`` writer adds ({} when OFF: the tip's checkpoint, key for key)."""
    c = getattr(model, "_map_dose_ctl", None)
    return {} if c is None else {STATE_KEY: c.state_dict()}


def resume_guard(state: dict, ctl: Optional[MapDoseController]) -> None:
    """A resume continues the SAME experiment. Controller ON + a checkpoint without the state REFUSES; OFF + a
    checkpoint carrying it REFUSES; ON + ON restores ``w_eff`` (refusing a changed target / bounds / group / every)."""
    have = state.get(STATE_KEY)
    if ctl is None:
        if have is not None:
            raise SystemExit("[k2dc] ⛔ resume REFUSED: ckpt.pt carries a map-dose controller state but this launch has "
                             "no --map-hires-share-target -- the resumed run would be a different experiment.")
        return
    if have is None:
        raise SystemExit(f"[k2dc] ⛔ resume REFUSED: --map-hires-share-target is set but ckpt.pt carries no "
                         f"{STATE_KEY!r} (its keys: {sorted(state)[:12]}): the weight the run had reached is unknown.")
    ctl.load_state_dict(have)


# ---------------------------------------------------------------------------------------------------------------- #
# the three hops the trainer's step calls (and the tests' tiny loop calls, THE SAME functions)                       #
# ---------------------------------------------------------------------------------------------------------------- #
def groups_for(model, ctl: MapDoseController) -> dict:
    """``{group: [params]}`` for the controller's group; REFUSES (at build time, not at the first firing) when the group is
    empty on this model (``layer2`` on a trunk with no ``net.layer2``)."""
    from tanitad.train import grad_share as _gs
    g = _gs.param_groups(model, {ctl.group: _gs.GROUP_PREFIXES[ctl.group]})
    if not g.get(ctl.group):
        raise SystemExit(f"[k2dc] ⛔ --map-hires-share-group {ctl.group}: no trainable parameter of the built model lies in it "
                         f"({_gs.GROUP_PREFIXES[ctl.group]}); the share cannot be read.")
    return g


def probe_before_forward(ctl, model, row_step: int, groups: dict, run_losses, traj_weight: float):
    """HOP 1 (before the training forward). Under a bf16 trunk the reading is taken on an fp32 REPLAY of this very step
    (``grad_share.fp32_replay``: RNG / generators / buffers restored, ``.grad`` untouched); otherwise ``None`` and
    :func:`read_after_losses` measures the step's own graph."""
    if ctl is None or not ctl.due(row_step):
        return None
    from tanitad.train import grad_share as _gs
    if _gs.bf16_levers(model):
        return _gs.fp32_replay(model, run_losses, groups, _gs.weights_of(model, traj_weight), prefix=PROBE_PREFIX,
                               keep=("map_hires",))
    return None


def read_after_losses(ctl, model, losses: dict, row_step: int, groups: dict, traj_weight: float, replay=None) -> dict:
    """HOP 2 (after the losses, BEFORE the backward). Takes the reading (or uses the replay), fires the controller (which
    updates ``ctl.w_eff``) and returns the metrics-row keys (``dcp_*`` probe + ``dc_*`` decision); ``{}`` when not due."""
    if ctl is None or not ctl.due(row_step):
        return {}
    if abs(float(getattr(model, "_w_map_hires", float("nan"))) - ctl.w_eff) > 1e-12:
        raise SystemExit(f"[k2dc] ⛔ model._w_map_hires {getattr(model, '_w_map_hires', None)!r} != the controller's w_eff {ctl.w_eff!r}: the "
                         f"probe would be read at one weight and solved for another.")
    from tanitad.train import grad_share as _gs
    probe = replay
    if probe is None:
        probe = _gs.measure(_gs.term_tensors(losses, _gs.weights_of(model, traj_weight), ("map_hires",)), losses["loss"],
                            groups, prefix=PROBE_PREFIX)
    return {**probe, **ctl.fire(probe, row_step)}


def apply_weight(ctl, model, row: dict) -> None:
    """HOP 3 (AFTER the step's backward): the solved weight becomes the one ``_map_hires_loss`` multiplies by from the NEXT
    step. ``row`` empty (no firing) = no-op, so between firings the weight is exactly what the last firing left."""
    if ctl is not None and row:
        model._w_map_hires = float(ctl.w_eff)


def dc_reading_rows(argv_every: int, steps: int) -> list:
    """The metrics rows on which the controller fires (for the launch gate's cost accounting)."""
    return list(range(int(argv_every), int(steps) + 1, int(argv_every))) if int(argv_every) > 0 else []


# ---------------------------------------------------------------------------------------------------------------- #
# G-DVB                                                                                                             #
# ---------------------------------------------------------------------------------------------------------------- #
def dvb_check(model, args) -> list:
    """Built controller state == argv, for all four flags. OFF: no controller on the model."""
    from tanitad.train.declared_vs_built import Mismatch
    c = getattr(model, "_map_dose_ctl", None)
    out = []
    if not is_on(args):
        if c is not None:
            out.append(Mismatch("--map-hires-share-target", None, c.target, "model._map_dose_ctl",
                                "a controller is built but argv declares none"))
        return out
    if c is None:
        return [Mismatch("--map-hires-share-target", float(args.map_hires_share_target), None, "model._map_dose_ctl",
                         "the controller declared in argv was never built (the 11,286-parameter defect class)")]
    lo, hi = parse_bounds(args.map_hires_w_bounds)
    for flag, want, got in (("--map-hires-share-target", float(args.map_hires_share_target), c.target),
                            ("--map-hires-share-every", int(args.map_hires_share_every), c.every),
                            ("--map-hires-share-group", str(args.map_hires_share_group), c.group),
                            ("--map-hires-w-bounds", (lo, hi), (c.lo, c.hi))):
        if want != got:
            out.append(Mismatch(flag, want, got, "model._map_dose_ctl"))
    return out


def _check_for(dest):
    flag = "--" + dest.replace("_", "-")

    def chk(model, args) -> list:
        return [m for m in dvb_check(model, args) if m.lever == flag]
    return chk


def register_dvb_levers(register) -> None:
    """The four flags enter the G-DVB registry as ``built`` levers (the controller object on the model); each entry
    reports only its own flag's mismatch."""
    for dest in FLAGS:
        register(dest, "built", _check_for(dest))
