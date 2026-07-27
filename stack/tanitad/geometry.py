"""THE geometry seam: one :class:`CanonicalFrame` flowing config -> cache build
-> trainer -> encoder, and a guard that makes a PARTIAL flip impossible.

Why this module exists
----------------------
Before 2026-07-27 the input geometry was two module constants (``F_REF``, the
literal ``256``) threaded as DEFAULT ARGUMENTS through ~10 ``calib`` functions,
re-declared as literals in three ``CORPUS_META`` dicts, re-declared again in
every cache-build ``params`` dict, and read a fourth time off
``cfg.encoder.image_size`` by the trainers. Changing the geometry meant editing
every one of those, and **missing one was silent**.

That is not hypothetical. It is the failure class that made every committed v4
number unreproducible this week: a changed ``vision_rank`` default reached some
call sites and not others, and the resulting checkpoints failed a STRICT load
with no earlier symptom.

So this module offers exactly two operations and one assertion:

* :func:`frame_of`      — the ONE way to read a config's geometry.
* :func:`apply_frame`   — the ONE way to change it. It writes the frame AND the
  encoder's ``image_size`` / ``image_width`` together, so they cannot diverge.
* :func:`assert_geometry_consistent` — called by the trainers and the cache
  builder. It REFUSES a config whose declared frame and encoder input geometry
  disagree, i.e. a half-applied change.

⛔ **This module chooses nothing.** ``CANONICAL_256`` remains the default
everywhere; which frame v5 trains on is decided by the FOV audit
(``…/incoming/2026-07-27-fov-crop-audit/``) and the PI, and lands as a separate,
declared change.
"""

from __future__ import annotations

import dataclasses

from tanitad.config import StackConfig
from tanitad.data.calib import (CANONICAL_256, CanonicalFrame, geometry_params)

__all__ = ["frame_of", "apply_frame", "assert_geometry_consistent",
           "build_params", "GeometryMismatch", "geometry_report",
           "geometry_literals", "assert_geometry_literals_consistent",
           "add_geometry_args", "frame_from_args", "apply_geometry_args"]


class GeometryMismatch(ValueError):
    """A config whose declared frame and encoder input geometry disagree."""


def frame_of(cfg: StackConfig) -> CanonicalFrame:
    """The canonical frame this config declares. ``None`` == ``CANONICAL_256``."""
    g = getattr(cfg, "geometry", None)
    return CANONICAL_256 if not g else CanonicalFrame.from_dict(g)


def apply_frame(cfg: StackConfig, frame: CanonicalFrame) -> StackConfig:
    """Return ``cfg`` with ``frame`` applied to EVERY geometry-bearing field.

    Writes, atomically as far as the caller is concerned:
      * ``cfg.geometry``            — the declared frame (what the cache key and
                                      the data card read);
      * ``cfg.encoder.image_size``  — output HEIGHT;
      * ``cfg.encoder.image_width`` — output WIDTH (``None`` when square, so a
                                      square frame leaves the config byte-
                                      identical to a pre-2026-07-27 one).

    This is the single command that changes the geometry. Everything else in the
    stack reads it back through :func:`frame_of` / :func:`assert_geometry_consistent`.
    """
    if frame.height % cfg.encoder.patch_size or frame.width % cfg.encoder.patch_size:
        raise GeometryMismatch(
            f"frame {frame.height}x{frame.width} is not divisible by patch "
            f"{cfg.encoder.patch_size}; the ViT cannot tokenize it. Pick a "
            f"multiple of the patch size (or change patch_size).")
    cfg.encoder = dataclasses.replace(
        cfg.encoder, image_size=frame.height,
        image_width=None if frame.is_square else frame.width)
    cfg.geometry = None if frame.is_canonical else frame.to_dict()
    return cfg


def assert_geometry_consistent(cfg: StackConfig, *, label: str = "config"
                               ) -> CanonicalFrame:
    """⭐ THE STALE-DEFAULT GUARD. Refuse a half-applied geometry change.

    Returns the frame when consistent; raises :class:`GeometryMismatch` when the
    declared frame and ``cfg.encoder``'s input geometry disagree — which is
    exactly what a call site that kept the old default produces. Call it at the
    top of any trainer, cache build or evaluator, BEFORE any GPU work.
    """
    frame = frame_of(cfg)
    enc_hw = cfg.encoder.image_hw()
    if enc_hw != frame.hw:
        raise GeometryMismatch(
            f"[{label}] GEOMETRY IS HALF-APPLIED.\n"
            f"  declared frame     : {frame.height}x{frame.width} "
            f"(f_ref {frame.f_ref}, {frame.projection})\n"
            f"  encoder input      : {enc_hw[0]}x{enc_hw[1]} "
            f"(image_size={cfg.encoder.image_size}, "
            f"image_width={cfg.encoder.image_width})\n"
            f"  One of these was changed and the other kept its old default. "
            f"That is the exact failure that made every committed v4 number "
            f"unreproducible. Change geometry ONLY via "
            f"tanitad.geometry.apply_frame(cfg, frame).")
    tiling_report(cfg, warn=True)
    return frame


def tiling_report(cfg: StackConfig, *, warn: bool = False) -> dict:
    """Does the token grid tile the readout grid evenly?

    THE RULE, re-derived (the encoder-tokenization stream reported it as
    "width must be a multiple of 64"): the readout pools a ``token_h x token_w``
    grid onto ``grid x grid_w`` cells, so it tiles exactly iff
    ``(H/patch) % grid == 0`` and ``(W/patch) % grid_w == 0``. At the deployed
    ``patch=16, grid=4`` that is ``H % 64 == 0`` and ``W % 64 == 0`` — the
    multiple-of-64 rule, which is a CONSEQUENCE of patch x grid, not a constant.

    ⚠️ TWO CORRECTIONS, both MEASURED:
      * ``448`` DOES satisfy it (448 = 7 x 64 -> 28 token cols, 28 % 4 == 0).
        A report that 448 fails is arithmetic, not measurement.
      * ``state_dim`` does NOT break when it fails. Since the pooling routes were
        converged (2026-07-27) the readout falls back to ``AdaptiveAvgPool2d``,
        which still yields ``grid * grid_w * d_readout``. What a non-tiling width
        costs is UNEVEN pooling bins (e.g. 42 cols -> bins of 11/10/11/10), not a
        shape failure. Worth avoiding, not fatal.
    """
    gh, gw = cfg.encoder.token_grid()
    r, rw = cfg.readout.grid, (cfg.readout.grid_w or cfg.readout.grid)
    tiles_h, tiles_w = gh % r == 0, gw % rw == 0
    h, w = cfg.encoder.image_hw()
    rep = {
        "token_grid": [gh, gw], "readout_grid": [r, rw],
        "tiles_exactly": bool(tiles_h and tiles_w),
        "height_multiple_of": cfg.encoder.patch_size * r,
        "width_multiple_of": cfg.encoder.patch_size * rw,
        "height_ok": bool(h % (cfg.encoder.patch_size * r) == 0),
        "width_ok": bool(w % (cfg.encoder.patch_size * rw) == 0),
        "state_dim": r * rw * cfg.readout.d_readout,
        "state_dim_is_geometry_invariant": True,
    }
    if warn and not rep["tiles_exactly"]:
        print(f"[geometry] WARNING: token grid {gh}x{gw} does not tile the "
              f"readout grid {r}x{rw}, so pooling falls back to ADAPTIVE "
              f"(uneven bins). state_dim is still {rep['state_dim']} — this is "
              f"a quality note, not a failure. To tile exactly, make height a "
              f"multiple of {rep['height_multiple_of']} and width a multiple of "
              f"{rep['width_multiple_of']}.", flush=True)
    return rep


def build_params(cfg: StackConfig, base: dict) -> dict:
    """Cache-build ``params`` for ``cfg``: ``base`` plus the geometry fragment.

    ⚠️ PARITY-CRITICAL. For the CANONICAL frame :func:`geometry_params` returns
    ``{}``, so the params dict hashes to exactly what it hashes today and
    ``physicalai-train-e438721ae894`` keeps meaning precisely what it means.
    Only a non-canonical frame adds a key, which is what makes a re-cropped
    cache structurally unable to collide with the parity cache.
    """
    frame = assert_geometry_consistent(cfg, label="cache build")
    return {**base, **geometry_params(frame)}


# --------------------------------------------------------------------------- #
# ⭐ THE DEFAULT-FLIP CHECKLIST, EXECUTABLE                                      #
# --------------------------------------------------------------------------- #
# The geometry is ALSO written as literals in places no config reaches: four
# `CORPUS_META` dicts (the I7 task-identity fingerprint) and the lake episode
# schema. They are the most likely place for a stale default to survive a flip,
# because nothing imports them from the frame — so "remember to update them" is
# a memory test, and this program has lost that test before.
#
# `assert_geometry_literals_consistent()` turns it into an assertion: flip
# `CANONICAL_256` (or a corpus's own frame) without moving these, and it FAILS
# and NAMES every file that still has to move. Same shape as
# `assert_geometry_consistent`, one level out.

#: Every geometry literal that no config reaches, and the frame it must match.
#: (module path, dict/attr name, key, what it must equal)
GEOMETRY_LITERAL_SITES: tuple[tuple[str, str, str, str], ...] = (
    ("tanitad.data.physicalai", "CORPUS_META", "image_size", "width"),
    ("tanitad.data.physicalai", "CORPUS_META", "f_eff_px", "f_ref"),
    ("tanitad.data.comma2k19", "CORPUS_META", "image_size", "width"),
    ("tanitad.data.comma2k19", "CORPUS_META", "f_eff_px", "f_ref"),
    ("tanitad.data.cosmos_drive", "CORPUS_META", "image_size", "width"),
    ("tanitad.data.cosmos_drive", "CORPUS_META", "f_eff_px", "f_ref"),
    ("tanitad.data.l2d", "CORPUS_META", "image_size", "width"),
    ("tanitad.data.l2d", "CORPUS_META", "f_eff_px", "f_ref"),
)


def geometry_literals(frame: CanonicalFrame | None = None) -> list[dict]:
    """Observed vs required value at every geometry literal site."""
    import importlib
    f = frame or CANONICAL_256
    want = {"width": float(f.width), "height": float(f.height),
            "f_ref": float(f.f_ref)}
    out = []
    for mod_name, obj_name, key, field in GEOMETRY_LITERAL_SITES:
        try:
            mod = importlib.import_module(mod_name)
            got = float(getattr(mod, obj_name)[key])
            err = None
        except Exception as e:                                # noqa: BLE001
            got, err = None, f"{type(e).__name__}: {e}"
        out.append({
            "site": f"{mod_name}.{obj_name}[{key!r}]",
            "must_equal": f"frame.{field}", "expected": want[field],
            "observed": got, "ok": (err is None and got == want[field]),
            "error": err,
        })
    # the lake episode schema default (a scalar; it has no width/height split —
    # see GEOMETRY_CONFIGURABLE.md section 3.2)
    try:
        from tanitad.lake import schema as _schema
        import dataclasses as _dc
        default = next(fl.default for fl in _dc.fields(_schema.LakeRecord)
                       if fl.name == "image_size")
        out.append({
            "site": "tanitad.lake.schema.LakeRecord.image_size (default)",
            "must_equal": "frame.width", "expected": want["width"],
            "observed": float(default), "ok": float(default) == want["width"],
            "error": None,
            "note": "scalar-only: this schema cannot express a NON-SQUARE frame "
                    "at all; widening needs an image_h/image_w schema bump.",
        })
    except Exception as e:                                    # noqa: BLE001
        out.append({"site": "tanitad.lake.schema.LakeRecord.image_size",
                    "ok": False, "error": f"{type(e).__name__}: {e}"})
    return out


def assert_geometry_literals_consistent(frame: CanonicalFrame | None = None
                                        ) -> list[dict]:
    """⭐ FAIL if the canonical frame moved but a geometry LITERAL did not.

    Returns the site report when consistent; raises :class:`GeometryMismatch`
    naming every file that still has to move otherwise. This is the flip
    checklist, executed rather than remembered.
    """
    rows = geometry_literals(frame)
    bad = [r for r in rows if not r["ok"]]
    if bad:
        f = frame or CANONICAL_256
        raise GeometryMismatch(
            "\n".join([
                "GEOMETRY LITERALS ARE STALE.",
                f"  canonical frame : {f.height}x{f.width}, f_ref {f.f_ref}, "
                f"{f.projection}",
                "  The frame moved but these hardcoded values did not. They are",
                "  reached by NO config, so nothing else will catch them:",
                *[f"    - {r['site']}: expected {r.get('expected')}, "
                  f"observed {r.get('observed')}"
                  + (f"  [{r['error']}]" if r.get("error") else "")
                  for r in bad],
                "",
                "  CORPUS_META is the I7 task-identity fingerprint: changing it",
                "  re-keys probe compatibility ACROSS corpora, so update it as a",
                "  DECLARED change alongside the default flip, never silently.",
            ]))
    return rows


# --------------------------------------------------------------------------- #
# CLI — the ONE flag set that changes the geometry                              #
# --------------------------------------------------------------------------- #
def add_geometry_args(ap) -> None:
    """Attach the geometry flags to an ``argparse`` parser.

    Every default reproduces the deployed frame EXACTLY, so adding these flags
    to a script changes nothing until one is passed. Shared by the cache builder
    and the trainers so a geometry can never be spelled two different ways.
    """
    g = ap.add_argument_group(
        "input geometry (2026-07-27)",
        "ALL defaults == the deployed 256x256 / f_ref 266 / pinhole frame. "
        "Pass --frame-hfov OR --f-ref, never both.")
    g.add_argument("--frame-h", type=int, default=None,
                   help="output HEIGHT px (default: keep the config's)")
    g.add_argument("--frame-w", type=int, default=None,
                   help="output WIDTH px (default: square)")
    g.add_argument("--frame-hfov", type=float, default=None,
                   help="retained horizontal FOV in degrees; solves f_ref for "
                        "the given width and projection")
    g.add_argument("--f-ref", type=float, default=None,
                   help="canonical focal in output px (alternative to "
                        "--frame-hfov)")
    g.add_argument("--projection", choices=["pinhole", "cylindrical"],
                   default=None,
                   help="output parametrisation (default: pinhole == deployed)")


def frame_from_args(args, cfg: StackConfig) -> CanonicalFrame:
    """Resolve the CLI flags into a frame, defaulting to ``cfg``'s own.

    Refuses --frame-hfov together with --f-ref: two ways to say the same thing
    is exactly the ambiguity this whole module exists to remove.
    """
    cur = frame_of(cfg)
    if getattr(args, "frame_hfov", None) is not None and \
            getattr(args, "f_ref", None) is not None:
        raise GeometryMismatch(
            "pass --frame-hfov OR --f-ref, not both — they are two spellings of "
            "the same quantity and having both is how a geometry drifts.")
    h = args.frame_h if getattr(args, "frame_h", None) else cur.height
    w = args.frame_w if getattr(args, "frame_w", None) else (
        args.frame_h if getattr(args, "frame_h", None) else cur.width)
    proj = getattr(args, "projection", None) or cur.projection
    if getattr(args, "frame_hfov", None) is not None:
        return CanonicalFrame.from_hfov(args.frame_hfov, h, w, proj)
    f = args.f_ref if getattr(args, "f_ref", None) else cur.f_ref
    return CanonicalFrame(height=h, width=w, f_ref=f, projection=proj)


def apply_geometry_args(args, cfg: StackConfig, *, label: str = "cli"
                        ) -> CanonicalFrame:
    """CLI -> frame -> config, in one call, with the consistency guard.

    Prints the resolved geometry (including whether it is the deployed one) so
    a run's log always says what field it trained on.
    """
    frame = frame_from_args(args, cfg)
    apply_frame(cfg, frame)
    rep = geometry_report(cfg)
    tag = "DEPLOYED (unchanged)" if frame.is_canonical else "NON-DEFAULT"
    print(f"[geometry] {label}: {tag} - {frame.height}x{frame.width}px, "
          f"f_ref {frame.f_ref:.2f}, {frame.projection}, "
          f"HFOV {rep['hfov_deg']:.2f}deg / VFOV {rep['vfov_deg']:.2f}deg, "
          f"tokens {rep['n_tokens']} ({rep['token_grid'][0]}x"
          f"{rep['token_grid'][1]} @ patch {rep['patch_size']}), "
          f"state_dim {rep['state_dim']}, cache fragment "
          f"{rep['cache_key_fragment'] or '{} (parity-preserving)'}", flush=True)
    if rep["exceeds_comma2k19_field"]:
        print(f"[geometry] WARNING: this frame ({rep['hfov_deg']:.2f}deg) EXCEEDS "
              f"comma2k19's entire field ({rep['comma2k19_hfov_ceiling_deg']}"
              f"deg). comma2k19 cannot supply it at any resolution — it must be "
              f"letterboxed (geometry_mode='rectify', explicit unobserved mask), "
              f"given its own frame, or dropped from the mix. That is a PI "
              f"decision, not a default.", flush=True)
    return frame


def geometry_report(cfg: StackConfig) -> dict:
    """Data-card rows: the frame, the token grid it produces, the state dim."""
    frame = assert_geometry_consistent(cfg, label="report")
    gh, gw = cfg.encoder.token_grid()
    rg = cfg.readout.grid
    rgw = rg if cfg.readout.grid_w is None else cfg.readout.grid_w
    return {
        **frame.report(),
        "patch_size": cfg.encoder.patch_size,
        "token_grid": [gh, gw],
        "n_tokens": gh * gw,
        "readout_grid": [rg, rgw],
        "state_dim": rg * rgw * cfg.readout.d_readout,
        "cache_key_fragment": geometry_params(frame),
    }
