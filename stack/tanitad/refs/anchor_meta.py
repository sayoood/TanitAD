"""Self-describing anchor-vocabulary artifacts: the UNITS live IN the file.

⛔ WHY THIS MODULE EXISTS — MEASURED 2026-09-04 on the LIVE refcv4b vocabulary
(`/workspace/experiments/refcv4b-b1-v72-40k/anchors.pt`, sha256 `e86cf507…e8fb`):
the file is a dict with exactly two keys, ``anchors`` [117, 8, 2] and ``controls``
[117, 2]. Column 1 of ``controls`` is LATERAL ACCELERATION in m/s², and nothing in
the file says so. Read as CURVATURE (1/m) the SAME BYTES give
``a_lat = v² · κ = 36² × 3.0 = 3,888 m/s² = 396 g`` at 36 m/s, with 104 of 117
anchors over a μ = 0.7 friction circle; read correctly they give ``3.0 m/s² =
0.31 g`` and 0 of 117. Both tables look like answers, and the 396 g one was
produced first, from the shipped file. **A correct formula applied under the
wrong units reads exactly like an answer** — the `df` / `free` / `step_s` scope
trap in a units costume.

The RUN was never ambiguous: ``config.json['argv']`` carries
``--anchor-control-units alat`` and ``config.json['anchors']['control_units']``
reads ``alat``. Only the standalone ``.pt``, opened in isolation, was.

The rule this module enforces:

* **builders WRITE** ``control_units``, ``horizon_s``, ``dt``, ``ref_speed_ms``,
  ``kappa_cap``, ``alat_v_floor`` and a provenance stamp INTO the ``.pt``
  (:func:`build_anchor_artifact`);
* **consumers REQUIRE** ``control_units`` on any file that carries ``controls``
  (:func:`read_anchor_artifact`), and a legacy file that declares nothing loads
  ONLY through an explicit override, which is recorded as
  ``control_units_source = "cli-override-legacy-file"`` so the run record says
  the units came from the operator, not from the artifact.

⛔ The live ``anchors.pt`` is NOT modified by anything here; it is exactly the
legacy case, and it keeps loading through ``--anchor-control-units alat``.
"""

from __future__ import annotations

import hashlib
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import torch
from torch import Tensor

#: schema tag written into every artifact this module builds.
SCHEMA = "tanitad.anchor_artifact/1"

#: what column 1 of ``controls`` MEANS. ``kappa`` = curvature 1/m, integrated as
#: supplied; ``alat`` = lateral acceleration m/s², curvature DERIVED per window
#: as ``a_lat / max(v0, alat_v_floor)²`` and clamped to ``kappa_cap``
#: (``tanitad.refs.refc.AnchorDecoder.roll_bank``).
CONTROL_UNITS = ("kappa", "alat")

#: ``control_units`` of a FIXED-PATH artifact (no ``controls``): the ``anchors``
#: are ego-frame metres, x along-track / y lateral, and nothing is re-rolled.
PATHS_ONLY = "paths"

#: ⭐ WHAT THE OPTIONAL THIRD COLUMN OF ``controls`` MEANS (2026-09-06).
#: ``constant`` — ``controls`` is [N, 2] and one (a_lon, a_lat|kappa) pair is
#: held for the whole horizon; this is the incumbent and the DEFAULT.
#: ``two_segment_alat_flip`` — ``controls`` is [N, 3] and column 2 is
#: ``t_split_s``, the time (SECONDS) at which the LATERAL channel's sign flips;
#: ``t_split_s >= horizon_s`` means it never flips, i.e. exactly ``constant``
#: (``tanitad.refs.anchor_twoseg.roll_bank``, bit-identical in that limit).
#: ``three_segment_alat_pulse`` — ``controls`` is [N, 4], column 2 is ``t1_s``
#: (the ``+ -> -`` flip) and column 3 is ``t2_s``, the time the LATERAL channel
#: goes to **ZERO** and the candidate runs straight (2026-09-06). ``t2_s >=
#: horizon_s`` means the third segment is empty, i.e. exactly
#: ``two_segment_alat_flip`` with ``t_split_s = t1_s``; ``t1_s >= horizon_s``
#: means exactly ``constant``. ⭐ THE NESTING IS EXACT AND IS WHAT MAKES EACH
#: FAMILY'S OFF PATH PROVABLE BY COMPARISON rather than asserted.
#: ⛔ A [N, 3] or [N, 4] file that declares NO schedule is REFUSED for the same
#: reason a units-less [N, 2] file is: an undeclared column read as the wrong
#: quantity produces a table that looks exactly like an answer
#: (:data:`INCIDENT`). ⛔ AND THERE IS NO CLI OVERRIDE FOR A SCHEDULE, unlike
#: units: the units override exists because a real LEGACY file (refcv4b's live
#: ``anchors.pt``) must keep loading, and **no legacy 3- or 4-column file
#: exists anywhere** — so a permitted guess could not rescue a real artifact,
#: only INVENT one.
CONSTANT_SCHEDULE = "constant"
TWO_SEGMENT_SCHEDULE = "two_segment_alat_flip"
THREE_SEGMENT_SCHEDULE = "three_segment_alat_pulse"
CONTROL_SCHEDULES = (CONSTANT_SCHEDULE, TWO_SEGMENT_SCHEDULE,
                     THREE_SEGMENT_SCHEDULE)

#: how many ``controls`` columns each schedule has.
SCHEDULE_NCOL = {CONSTANT_SCHEDULE: 2, TWO_SEGMENT_SCHEDULE: 3,
                 THREE_SEGMENT_SCHEDULE: 4}

#: the extra column names each schedule appends to the base two.
SCHEDULE_EXTRA_COLUMNS = {
    CONSTANT_SCHEDULE: [],
    TWO_SEGMENT_SCHEDULE: ["t_split_s"],
    THREE_SEGMENT_SCHEDULE: ["t1_s", "t2_s"],
}

#: the fields every artifact carries — present-but-``None`` where a field does
#: not apply (a fixed-path file has no reference speed), never absent.
REQUIRED_META = ("control_units", "horizon_s", "dt", "ref_speed_ms",
                 "kappa_cap", "alat_v_floor")

#: one sentence a refusal can quote so the reader knows what the rule costs.
INCIDENT = (
    "MEASURED 2026-09-04 on refcv4b's live anchors.pt: `controls[:, 1]` is "
    "LATERAL ACCELERATION (m/s^2) but the file did not say so; read as "
    "CURVATURE (1/m) the same bytes give a_lat = v^2*kappa = 36^2 x 3.0 = "
    "3,888 m/s^2 = 396 g at 36 m/s with 104/117 anchors over a mu = 0.7 "
    "friction circle, read correctly 3.0 m/s^2 = 0.31 g and 0/117 -- both "
    "tables look like answers.")


class AnchorUnitsError(ValueError):
    """Base class: the artifact's control units cannot be established."""


class AnchorUnitsMissing(AnchorUnitsError):
    """A ``controls``-carrying artifact declares no ``control_units`` and no
    explicit override was given."""


class AnchorUnitsConflict(AnchorUnitsError):
    """The artifact declares one unit and the operator passed another."""


class AnchorScheduleMissing(AnchorUnitsError):
    """A ``controls`` tensor with a THIRD column that declares no
    ``control_schedule``. Same family as :class:`AnchorUnitsMissing`, one column
    to the right: nothing in the file says the column is a split TIME, and read
    as anything else it silently emits a different vocabulary."""


class AnchorScheduleConflict(AnchorUnitsError):
    """The declared ``control_schedule`` does not match the number of
    ``controls`` columns the file actually carries."""


@dataclass
class AnchorArtifact:
    """What :func:`read_anchor_artifact` hands back.

    ``control_units`` is the RESOLVED value (file, override, or ``"paths"``);
    ``control_units_source`` says where it came from — ``"file"``,
    ``"file+cli"`` (both given, equal), ``"cli-override-legacy-file"`` (the
    live refcv4b case), or ``"n/a-fixed-paths"`` (no ``controls``).
    ``meta`` is every non-tensor entry of the file (``{}`` for a legacy file or
    a bare tensor)."""
    anchors: Tensor
    controls: Tensor | None
    control_units: str
    control_units_source: str
    meta: dict[str, Any] = field(default_factory=dict)
    path: str | None = None
    #: ⭐ RESOLVED control schedule (2026-09-06). A 2-column file that declares
    #: nothing is ``constant`` — that is the incumbent and is not a guess, it is
    #: the only thing a 2-column tensor CAN be. A 3-column file that declares
    #: nothing is REFUSED (:class:`AnchorScheduleMissing`).
    control_schedule: str = CONSTANT_SCHEDULE

    @property
    def declared(self) -> dict[str, Any]:
        """The :data:`REQUIRED_META` fields as the FILE declares them
        (``None`` where the file is silent) — what a run record should stamp."""
        return {k: self.meta.get(k) for k in REQUIRED_META}


# ----------------------------------------------------------------- helpers ---
def sha256_of_tensor(t: Tensor) -> str:
    return hashlib.sha256(
        t.detach().to("cpu", torch.float32).contiguous().numpy().tobytes()
    ).hexdigest()


def _sha256_of_file(path: str | os.PathLike | None) -> str | None:
    if not path:
        return None
    try:
        with open(path, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()
    except OSError:
        return None


def provenance_stamp(builder: str | os.PathLike | None,
                     extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Who built it, from what, when — by CONTENT where possible.

    ``builder`` is the builder script's path (``__file__``); its sha256 is
    recorded when the file is readable, so the artifact names the bytes that
    produced it rather than a filename that may have been edited since."""
    b = os.fspath(builder) if builder else None
    stamp: dict[str, Any] = {
        "builder": os.path.basename(b) if b else None,
        "builder_path": os.path.abspath(b) if b else None,
        "builder_sha256": _sha256_of_file(b),
        "argv": list(sys.argv[1:]),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "torch_version": str(torch.__version__),   # TorchVersion is not weights_only-safe
        "schema": SCHEMA,
    }
    if extra:
        stamp.update(extra)
    return stamp


# ----------------------------------------------------------------- build -----
def build_anchor_artifact(anchors: Tensor, controls: Tensor | None = None, *,
                          control_units: str, horizons, dt: float = 0.1,
                          ref_speed_ms: float | None = None,
                          kappa_cap: float | None = None,
                          alat_v_floor: float | None = None,
                          control_schedule: str = CONSTANT_SCHEDULE,
                          builder: str | os.PathLike | None,
                          extra: dict[str, Any] | None = None
                          ) -> dict[str, Any]:
    """The dict a builder ``torch.save``s — tensors PLUS the fields that make
    them readable in isolation.

    * ``controls`` given ⇒ ``control_units`` must be one of
      :data:`CONTROL_UNITS`, and ``ref_speed_ms`` / ``kappa_cap`` /
      ``alat_v_floor`` must all be given (the decoder re-rolls the bank under
      exactly these; a file that omits one cannot be re-rolled faithfully).
    * ``controls`` absent ⇒ ``control_units`` must be :data:`PATHS_ONLY`; the
      three constants are recorded as ``None`` (present, not applicable).
    * ``horizon_s`` is DERIVED as ``max(horizons) * dt`` and the slot count is
      asserted against ``anchors.shape[1]``, so a file cannot claim a horizon
      its tensors do not have.
    """
    a = anchors.detach().to("cpu", torch.float32).contiguous()
    if a.ndim != 3 or a.shape[-1] != 2:
        raise ValueError(f"anchors must be [N, S, 2]; got {tuple(a.shape)}")
    hz = tuple(int(h) for h in horizons)
    if len(hz) != a.shape[1]:
        raise ValueError(f"{len(hz)} horizons declared for anchors with "
                         f"{a.shape[1]} slots")
    if control_schedule not in CONTROL_SCHEDULES:
        raise ValueError(f"control_schedule {control_schedule!r} must be one of "
                         f"{CONTROL_SCHEDULES}")
    ncol = SCHEDULE_NCOL[control_schedule]
    c = None
    if controls is not None:
        c = controls.detach().to("cpu", torch.float32).contiguous()
        if tuple(c.shape) != (a.shape[0], ncol):
            raise ValueError(f"controls must be [{a.shape[0]}, {ncol}] for "
                             f"control_schedule={control_schedule!r}; got "
                             f"{tuple(c.shape)}")
        if control_units not in CONTROL_UNITS:
            raise ValueError(f"control_units {control_units!r} must be one of "
                             f"{CONTROL_UNITS} for a controls-carrying artifact")
        missing = [k for k, v in (("ref_speed_ms", ref_speed_ms),
                                  ("kappa_cap", kappa_cap),
                                  ("alat_v_floor", alat_v_floor)) if v is None]
        if missing:
            raise ValueError(f"a controls-carrying artifact must declare "
                             f"{missing} -- the decoder re-rolls the bank "
                             f"under them")
    elif control_units != PATHS_ONLY:
        raise ValueError(f"control_units {control_units!r} given but no "
                         f"controls; a fixed-path artifact declares "
                         f"{PATHS_ONLY!r}")
    art: dict[str, Any] = {
        "anchors": a,
        "schema": SCHEMA,
        "control_units": control_units,
        "horizon_s": float(max(hz) * float(dt)),
        "dt": float(dt),
        "horizons_steps": list(hz),
        "n_anchors": int(a.shape[0]),
        "ref_speed_ms": None if ref_speed_ms is None else float(ref_speed_ms),
        "kappa_cap": None if kappa_cap is None else float(kappa_cap),
        "alat_v_floor": None if alat_v_floor is None else float(alat_v_floor),
        "anchors_sha256": sha256_of_tensor(a),
        # ⛔ present-but-``None`` where it does not apply, like the three re-roll
        # constants above: a FIXED-PATH file has no ``controls``, so declaring a
        # schedule for them would be a statement about a tensor the file does not
        # carry — the exact shape of error this module exists to prevent.
        "control_schedule": control_schedule if controls is not None else None,
        "provenance": provenance_stamp(builder),
    }
    if c is not None:
        cols = (["a_lon_ms2", "a_lat_ms2"] if control_units == "alat"
                else ["a_lon_ms2", "kappa_inv_m"])
        cols = cols + list(SCHEDULE_EXTRA_COLUMNS[control_schedule])
        art["controls"] = c
        art["controls_columns"] = cols
        art["controls_sha256"] = sha256_of_tensor(c)
        art["straight_ahead_control_present"] = bool(
            ((c[:, 0] == 0) & (c[:, 1] == 0)).any())
        hs = float(art["horizon_s"])
        if control_schedule == TWO_SEGMENT_SCHEDULE:
            # ⭐ A row whose split is at or beyond the horizon NEVER flips and is
            # therefore a `constant` candidate living in a 3-column tensor. The
            # counts are declared so a reader does not have to re-derive which
            # rows are which -- the same reason the units are declared.
            flips = c[:, 2] < hs - 1e-9
            art["n_two_segment"] = int(flips.sum())
            art["n_constant"] = int((~flips).sum())
            art["t_split_s_values"] = sorted(
                {round(float(x), 6) for x in c[flips, 2]})
        elif control_schedule == THREE_SEGMENT_SCHEDULE:
            # ⭐ Same declaration, one nesting level deeper. A 4-column tensor
            # holds THREE populations and a reader must not have to re-derive
            # which row is which: `t1 >= horizon` never flips (constant),
            # `t2 >= horizon` flips but never returns to straight (two-segment),
            # and only `t2 < horizon` is a genuine three-segment pulse.
            flips = c[:, 2] < hs - 1e-9
            returns = flips & (c[:, 3] < hs - 1e-9)
            art["n_three_segment"] = int(returns.sum())
            art["n_two_segment"] = int((flips & ~returns).sum())
            art["n_constant"] = int((~flips).sum())
            art["t1_s_values"] = sorted({round(float(x), 6) for x in c[flips, 2]})
            art["t2_s_values"] = sorted(
                {round(float(x), 6) for x in c[returns, 3]})
            # ⛔ An ORDERING assertion, not a comment. `t2 < t1` would emit a
            # negative-duration middle segment, which the integrator silently
            # renders as "never counter-steer" -- a candidate that is not the one
            # the row declares. Refused at BUILD time so no such file exists.
            bad = torch.nonzero(returns & (c[:, 3] < c[:, 2] - 1e-9)).reshape(-1)
            if bad.numel():
                raise ValueError(
                    f"{bad.numel()} three-segment row(s) declare t2 < t1 "
                    f"(first at index {int(bad[0])}: t1={float(c[bad[0], 2])}, "
                    f"t2={float(c[bad[0], 3])}). The middle segment would have "
                    f"negative duration and the integrator would emit a "
                    f"candidate that is not the one the row declares.")
    else:
        art["path_units"] = "m, ego frame at t0: x along-track, y lateral"
    if extra:
        for k, v in extra.items():
            if k in ("anchors", "controls"):
                raise ValueError(f"extra may not override {k!r}")
            art.setdefault(k, v)
    return art


# ------------------------------------------------------------------ read -----
def _resolve_schedule(controls: Tensor, meta: dict[str, Any],
                      where: str) -> str:
    """The RESOLVED ``control_schedule`` of a ``controls``-carrying artifact.

    ⛔ There is no override here on purpose. Units were overridable because a
    LEGACY file existed whose units were known from its run record; **no legacy
    3-column or 4-column file exists anywhere**, so an undeclared wide tensor is
    a file nobody can read, and guessing is exactly the failure this module was
    written after — a permitted guess could not rescue a real artifact, only
    INVENT one. A 2-column tensor is ``constant`` by construction — that is a
    fact about the shape, not a default.
    """
    ncol = int(controls.shape[-1])
    declared = meta.get("control_schedule")
    if declared is not None and declared not in CONTROL_SCHEDULES:
        raise AnchorScheduleConflict(
            f"{where} declares control_schedule={declared!r}, not one of "
            f"{CONTROL_SCHEDULES}")
    if declared is None:
        if ncol == 2:
            return CONSTANT_SCHEDULE
        # name the schedule this WIDTH would be, so the refusal is actionable
        # rather than merely correct.
        fits = [s for s, k in SCHEDULE_NCOL.items() if k == ncol]
        hint = (f"Rebuild it with build_anchor_artifact("
                f"control_schedule='{fits[0]}')." if fits else
                f"No schedule in {CONTROL_SCHEDULES} is [N, {ncol}].")
        extra = ", ".join(SCHEDULE_EXTRA_COLUMNS.get(fits[0], [])) if fits \
            else "the extra column(s)"
        raise AnchorScheduleMissing(
            f"{where} carries `controls` {tuple(controls.shape)} with {ncol} "
            f"columns but declares NO `control_schedule`, so the extra "
            f"column(s) ({extra}) cannot be told apart from anything else "
            f"those numbers could mean. {hint} There is NO override for this "
            f"and there will not be one: no legacy wide file exists, so a "
            f"permitted guess would invent an artifact rather than rescue one. "
            f"This is the same failure as the units one, one column to the "
            f"right: {INCIDENT}")
    want = SCHEDULE_NCOL[declared]
    if ncol != want:
        raise AnchorScheduleConflict(
            f"{where} declares control_schedule={declared!r}, which is "
            f"[N, {want}], but carries controls {tuple(controls.shape)}. A "
            f"schedule that does not match the tensor is a vocabulary nobody "
            f"can re-roll faithfully.")
    return declared


def read_anchor_artifact(src, *, cli_control_units: str | None = None,
                         map_location="cpu") -> AnchorArtifact:
    """Load an anchor artifact and RESOLVE its control units.

    ``src`` is a path (loaded with ``weights_only=True``) or an already-loaded
    object (a bare tensor or a dict). Resolution, for a ``controls``-carrying
    file, with ``F`` = the file's ``control_units`` and ``C`` = the override:

    ============  ============  ======================================
    F             C             result
    ============  ============  ======================================
    absent        absent        ⛔ :class:`AnchorUnitsMissing`
    absent        given         C, source ``cli-override-legacy-file``
    given         absent        F, source ``file``
    given         == C          F, source ``file+cli``
    given         != C          ⛔ :class:`AnchorUnitsConflict`
    ============  ============  ======================================

    A file without ``controls`` needs no units (``paths``, source
    ``n/a-fixed-paths``); an override passed for such a file is ignored, since
    there is nothing it could apply to.
    """
    path = None
    obj = src
    if isinstance(src, (str, os.PathLike)):
        path = os.fspath(src)
        obj = torch.load(path, map_location=map_location, weights_only=True)
    if isinstance(obj, Tensor):
        obj = {"anchors": obj}
    if not isinstance(obj, dict) or "anchors" not in obj:
        raise ValueError(f"anchor artifact {path or type(obj)} is neither a "
                         f"tensor nor a dict with an `anchors` entry")
    anchors = obj["anchors"]
    controls = obj.get("controls")
    meta = {k: v for k, v in obj.items() if k not in ("anchors", "controls")}
    where = path or "<in-memory anchor artifact>"
    if cli_control_units is not None and cli_control_units not in CONTROL_UNITS:
        raise ValueError(f"--anchor-control-units {cli_control_units!r} not in "
                         f"{CONTROL_UNITS}")
    if controls is None:
        declared = meta.get("control_units")
        if declared not in (None, PATHS_ONLY):
            raise AnchorUnitsConflict(
                f"{where} declares control_units={declared!r} but carries no "
                f"`controls` -- a fixed-path artifact declares {PATHS_ONLY!r}")
        if meta.get("control_schedule") is not None:
            raise AnchorScheduleConflict(
                f"{where} declares control_schedule="
                f"{meta['control_schedule']!r} but carries no `controls`. A "
                f"schedule describes how a control is applied over the horizon; "
                f"a fixed-path artifact has no control to apply, so the field "
                f"is a statement about a tensor the file does not hold.")
        return AnchorArtifact(anchors, None, PATHS_ONLY, "n/a-fixed-paths",
                              meta, path)
    sched = _resolve_schedule(controls, meta, where)
    declared = meta.get("control_units")
    if declared is not None and declared not in CONTROL_UNITS:
        raise AnchorUnitsConflict(
            f"{where} declares control_units={declared!r}, not one of "
            f"{CONTROL_UNITS}")
    if declared is None and cli_control_units is None:
        raise AnchorUnitsMissing(
            f"{where} carries `controls` {tuple(controls.shape)} but declares "
            f"NO `control_units`, so column 1 cannot be told apart from a "
            f"curvature: {INCIDENT} Rebuild it with "
            f"tanitad.refs.anchor_meta.build_anchor_artifact, or -- for a "
            f"legacy file whose units you KNOW -- pass "
            f"--anchor-control-units {{kappa|alat}} explicitly; the run record "
            f"will then say the units came from the operator "
            f"(control_units_source='cli-override-legacy-file').")
    if declared is None:
        return AnchorArtifact(anchors, controls, cli_control_units,
                              "cli-override-legacy-file", meta, path,
                              control_schedule=sched)
    if cli_control_units is None:
        return AnchorArtifact(anchors, controls, declared, "file", meta, path,
                              control_schedule=sched)
    if cli_control_units != declared:
        raise AnchorUnitsConflict(
            f"{where} declares control_units={declared!r} but "
            f"--anchor-control-units {cli_control_units!r} was passed. The "
            f"artifact is the authority on what its own column means; drop the "
            f"flag, or rebuild the artifact if it is the file that is wrong. "
            f"({INCIDENT})")
    return AnchorArtifact(anchors, controls, declared, "file+cli", meta, path,
                          control_schedule=sched)


def mismatches(art: AnchorArtifact, *, horizon_s: float | None = None,
               dt: float | None = None, ref_speed_ms: float | None = None,
               kappa_cap: float | None = None,
               alat_v_floor: float | None = None,
               tol: float = 1e-6) -> list[str]:
    """Where the file's DECLARED constants disagree with the values the consumer
    will use. Silent (undeclared) fields never mismatch; a declared field that
    differs is one sentence per field, so the caller can refuse with all of
    them at once. A consumer that re-rolls the bank under different constants
    than the file was built with would hold two vocabularies under one name."""
    out = []
    for name, want in (("horizon_s", horizon_s), ("dt", dt),
                       ("ref_speed_ms", ref_speed_ms),
                       ("kappa_cap", kappa_cap),
                       ("alat_v_floor", alat_v_floor)):
        have = art.meta.get(name)
        if want is None or have is None:
            continue
        if abs(float(have) - float(want)) > tol:
            out.append(f"{name}: the artifact declares {float(have)!r}, the "
                       f"consumer would use {float(want)!r}")
    return out


def describe(art: AnchorArtifact) -> str:
    """One line for a launch log."""
    d = art.declared
    return (f"anchors {tuple(art.anchors.shape)}"
            + (f", controls {tuple(art.controls.shape)}"
               if art.controls is not None else "")
            + f", units={art.control_units} ({art.control_units_source})"
            + (f", schedule={art.control_schedule}"
               if art.controls is not None else "")
            + f", horizon_s={d['horizon_s']}, dt={d['dt']}, "
              f"ref_speed_ms={d['ref_speed_ms']}, kappa_cap={d['kappa_cap']}, "
              f"alat_v_floor={d['alat_v_floor']}, "
              f"schema={art.meta.get('schema')}")
