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
    c = None
    if controls is not None:
        c = controls.detach().to("cpu", torch.float32).contiguous()
        if tuple(c.shape) != (a.shape[0], 2):
            raise ValueError(f"controls must be [{a.shape[0]}, 2]; got "
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
        "provenance": provenance_stamp(builder),
    }
    if c is not None:
        art["controls"] = c
        art["controls_columns"] = (["a_lon_ms2", "a_lat_ms2"]
                                   if control_units == "alat"
                                   else ["a_lon_ms2", "kappa_inv_m"])
        art["controls_sha256"] = sha256_of_tensor(c)
        art["straight_ahead_control_present"] = bool(
            ((c[:, 0] == 0) & (c[:, 1] == 0)).any())
    else:
        art["path_units"] = "m, ego frame at t0: x along-track, y lateral"
    if extra:
        for k, v in extra.items():
            if k in ("anchors", "controls"):
                raise ValueError(f"extra may not override {k!r}")
            art.setdefault(k, v)
    return art


# ------------------------------------------------------------------ read -----
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
        return AnchorArtifact(anchors, None, PATHS_ONLY, "n/a-fixed-paths",
                              meta, path)
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
                              "cli-override-legacy-file", meta, path)
    if cli_control_units is None:
        return AnchorArtifact(anchors, controls, declared, "file", meta, path)
    if cli_control_units != declared:
        raise AnchorUnitsConflict(
            f"{where} declares control_units={declared!r} but "
            f"--anchor-control-units {cli_control_units!r} was passed. The "
            f"artifact is the authority on what its own column means; drop the "
            f"flag, or rebuild the artifact if it is the file that is wrong. "
            f"({INCIDENT})")
    return AnchorArtifact(anchors, controls, declared, "file+cli", meta, path)


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
            + f", horizon_s={d['horizon_s']}, dt={d['dt']}, "
              f"ref_speed_ms={d['ref_speed_ms']}, kappa_cap={d['kappa_cap']}, "
              f"alat_v_floor={d['alat_v_floor']}, "
              f"schema={art.meta.get('schema')}")
