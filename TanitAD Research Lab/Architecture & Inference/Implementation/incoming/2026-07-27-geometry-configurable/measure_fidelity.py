"""BOTH-DIRECTIONS validation of the configurable-geometry refactor.

Direction 1 (fidelity): the NEW code path, run at the OLD geometry, must
reproduce the pre-refactor output. Reference = ``git show <base>:stack/tanitad/
data/calib.py``, imported as a standalone module (calib.py depends only on
math/warnings/dataclasses/torch, so it loads clean out of tree).
Tolerance claimed: **BIT-EXACT** (``torch.equal``) on the uint8 output and exact
equality on every integer crop box — justified because the canonical branches
evaluate the *same arithmetic expressions*, not merely equivalent ones. Anything
short of bit-exact here would mean the "no default changed" claim is false.

Direction 2 (deliberate failure): inputs that MUST be rejected. A refactor that
only proves the happy path proves nothing about the guard rails.

Usage:  python measure_fidelity.py --ref <path to HEAD calib.py> --out <json>
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import torch

STACK = Path(__file__).resolve().parents[5] / "stack"
sys.path.insert(0, str(STACK))


def load_ref(path: Path):
    spec = importlib.util.spec_from_file_location("calib_ref", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["calib_ref"] = mod          # @dataclass resolves via sys.modules
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="pre-refactor calib.py")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ref = load_ref(Path(args.ref))
    from tanitad.data import calib as new

    rows: list[dict] = []
    ok = True

    def check(name: str, cond: bool, detail=None):
        nonlocal ok
        ok = ok and bool(cond)
        rows.append({"check": name, "pass": bool(cond), "detail": detail})

    # ---------------------------------------------------------------- setup #
    g = torch.Generator().manual_seed(20260727)
    # Both PhysicalAI rigs. cy 543 = rig A, cy 755 = rig B (the ~215 px split).
    rigs = {
        "rigA_per_clip": dict(cx=958.0, cy=543.0, per_clip=True),
        "rigB_per_clip": dict(cx=961.0, cy=755.0, per_clip=True),
        "median_fallback": dict(cx=958.0, cy=753.18, per_clip=False),
    }
    poly = (0.0, 927.5032, 23.1353, -58.5012, 16.5067)

    # ------------------------------------------------- 1. constants intact #
    check("F_REF unchanged", new.F_REF == ref.F_REF == 266.0,
          {"new": new.F_REF, "ref": ref.F_REF})
    check("COMMA2K19_FOCAL_PX unchanged",
          new.COMMA2K19_FOCAL_PX == ref.COMMA2K19_FOCAL_PX)
    check("PHYSICALAI fallback intrinsics unchanged",
          (new.PHYSICALAI_FRONT_WIDE_FTHETA.poly
           == ref.PHYSICALAI_FRONT_WIDE_FTHETA.poly
           and new.PHYSICALAI_FRONT_WIDE_FTHETA.cy
           == ref.PHYSICALAI_FRONT_WIDE_FTHETA.cy))
    check("canonical_halfangle_rad unchanged",
          new.canonical_halfangle_rad() == ref.canonical_halfangle_rad(),
          {"rad": new.canonical_halfangle_rad(),
           "deg": math.degrees(2 * new.canonical_halfangle_rad())})
    check("CANONICAL_256 is the deployed frame",
          (new.CANONICAL_256.height == 256 and new.CANONICAL_256.width == 256
           and new.CANONICAL_256.f_ref == 266.0
           and new.CANONICAL_256.projection == "pinhole"
           and new.CANONICAL_256.is_canonical),
          new.CANONICAL_256.report())

    # ------------------------------------- 2. f-theta crop, BIT-EXACT, both rigs #
    for rig, kw in rigs.items():
        i_new = new.FThetaIntrinsics(poly=poly, width=1920, height=1080, **kw)
        i_ref = ref.FThetaIntrinsics(poly=poly, width=1920, height=1080, **kw)
        for size in (256, 224, 128):
            check(f"ftheta_crop_size[{rig},{size}]",
                  new.ftheta_crop_size(i_new, size) == ref.ftheta_crop_size(i_ref, size),
                  {"c": new.ftheta_crop_size(i_new, size)})
            for center in ("principal", "geometric"):
                a = new.ftheta_crop_box(i_new, 1080, 1920, size, center=center)
                b = ref.ftheta_crop_box(i_ref, 1080, 1920, size, center=center)
                check(f"ftheta_crop_box[{rig},{size},{center}]", a == b,
                      {"c_top_left": list(a)})
        # decoded at a NON-native resolution too (the sx/sy scaling branch)
        for (h, w) in ((1080, 1920), (540, 960)):
            vid = torch.randint(0, 256, (3, 3, h, w), generator=g,
                                dtype=torch.uint8)
            a = new.ftheta_crop_resize(vid, i_new, 256)
            fa, fay = new.ftheta_crop_resize.last_f_eff, None
            b = ref.ftheta_crop_resize(vid, i_ref, 256)
            fb = ref.ftheta_crop_resize.last_f_eff
            check(f"ftheta_crop_resize BIT-EXACT[{rig},{h}x{w}]",
                  torch.equal(a, b) and a.shape == b.shape,
                  {"shape": list(a.shape),
                   "maxabsdiff": int((a.int() - b.int()).abs().max())})
            check(f"ftheta_crop_resize.last_f_eff[{rig},{h}x{w}]", fa == fb,
                  {"f_eff": fa})
            # explicit-frame spelling of the SAME geometry must also be bit-exact
            c = new.ftheta_crop_resize(vid, i_new, frame=new.CANONICAL_256)
            check(f"explicit CANONICAL_256 frame BIT-EXACT[{rig},{h}x{w}]",
                  torch.equal(a, c),
                  {"maxabsdiff": int((a.int() - c.int()).abs().max())})
        # horizon row + feff report
        for center in ("principal", "geometric"):
            check(f"ftheta_horizon_row[{rig},{center}]",
                  new.ftheta_horizon_row(i_new, center=center)
                  == ref.ftheta_horizon_row(i_ref, center=center),
                  {"row": round(new.ftheta_horizon_row(i_new, center=center), 4)})
        check(f"ftheta_feff_report[{rig}]",
              new.ftheta_feff_report(i_new) == ref.ftheta_feff_report(i_ref),
              new.ftheta_feff_report(i_new))

    # ------------------------------------------- 3. pinhole paths, BIT-EXACT #
    for nm, i_new, i_ref in (
            ("comma2k19", new.COMMA2K19_INTR, ref.COMMA2K19_INTR),
            ("pandaset", new.PANDASET_FRONT_INTR, ref.PANDASET_FRONT_INTR)):
        h, w = i_new.height, i_new.width
        vid = torch.randint(0, 256, (2, 3, h, w), generator=g, dtype=torch.uint8)
        a = new.pinhole_rectify(vid, i_new)
        oa = new.pinhole_rectify.last_observed_frac
        b = ref.pinhole_rectify(vid, i_ref)
        ob = ref.pinhole_rectify.last_observed_frac
        check(f"pinhole_rectify BIT-EXACT[{nm}]", torch.equal(a, b),
              {"maxabsdiff": int((a.int() - b.int()).abs().max()),
               "observed_frac": round(oa, 6)})
        check(f"pinhole_rectify observed_frac[{nm}]", oa == ob)
        check(f"pinhole_geometry_report[{nm}]",
              new.pinhole_geometry_report(i_new)
              == ref.pinhole_geometry_report(i_ref),
              new.pinhole_geometry_report(i_new))
        c = new.pinhole_rectify(vid, i_new, frame=new.CANONICAL_256)
        check(f"pinhole explicit-frame BIT-EXACT[{nm}]", torch.equal(a, c))

    vid = torch.randint(0, 256, (2, 3, 874, 1164), generator=g, dtype=torch.uint8)
    a = new.focal_crop_resize(vid, new.COMMA2K19_FOCAL_PX, 256)
    b = ref.focal_crop_resize(vid, ref.COMMA2K19_FOCAL_PX, 256)
    check("focal_crop_resize BIT-EXACT[comma2k19]", torch.equal(a, b),
          {"f_eff": new.focal_crop_resize.last_f_eff,
           "maxabsdiff": int((a.int() - b.int()).abs().max())})
    c = new.focal_crop_resize(vid, new.COMMA2K19_FOCAL_PX, 256,
                              frame=new.CANONICAL_256)
    check("focal_crop_resize explicit-frame BIT-EXACT", torch.equal(a, c))
    check("ftheta_undistort BIT-EXACT",
          torch.equal(
              new.ftheta_undistort(vid[:, :, :540, :960],
                                   new.PHYSICALAI_FRONT_WIDE_FTHETA),
              ref.ftheta_undistort(vid[:, :, :540, :960],
                                   ref.PHYSICALAI_FRONT_WIDE_FTHETA)))

    # ---------------------------- 4. the rectangular path degenerates to square #
    i_new = new.FThetaIntrinsics(poly=poly, cx=958.0, cy=543.0, width=1920,
                                 height=1080, per_clip=True)
    for size in (256, 128):
        sq = new.CanonicalFrame(height=size, width=size)
        ch, cw = new.ftheta_crop_size_hw(i_new, sq)
        check(f"rect crop == square crop[{size}]",
              ch == cw == new.ftheta_crop_size(i_new, size),
              {"c_h": ch, "c_w": cw})

    # --------------------------------------- 5. cache-key parity by construction #
    from tanitad.data.epcache import cache_key
    from tanitad.data.physicalai import geometry_build_params
    legacy = {"size": 256, "n_stack": 3, "hz": 10, "calib": "ftheta_v2"}
    ids = [{"clip_id": f"c{i:04d}"} for i in range(64)]
    k_legacy = cache_key(ids, legacy)
    check("geometry_params(canonical) is EMPTY", new.geometry_params() == {},
          new.geometry_params())
    check("geometry_build_params(canonical) is EMPTY",
          geometry_build_params() == {})
    check("canonical geometry leaves the cache key IDENTICAL",
          cache_key(ids, {**legacy, **geometry_build_params()}) == k_legacy,
          {"key": k_legacy})
    wide = new.CanonicalFrame(height=256, width=640,
                              f_ref=new.CanonicalFrame.from_hfov(
                                  100.0, 256, 640).f_ref)
    k_wide = cache_key(ids, {**legacy, **geometry_build_params(wide)})
    check("a NON-canonical geometry mints a DIFFERENT key", k_wide != k_legacy,
          {"legacy": k_legacy, "wide": k_wide,
           "fragment": geometry_build_params(wide)})
    k_cyl = cache_key(ids, {**legacy,
                            **geometry_build_params(wide, "cylindrical")})
    check("projection alone mints a DIFFERENT key",
          len({k_legacy, k_wide, k_cyl}) == 3,
          {"cyl": k_cyl})
    check("the SAME frame is key-STABLE (re-derivable, not random)",
          cache_key(ids, {**legacy, **geometry_build_params(wide)}) == k_wide)
    # ⭐ the pre-existing hole: f_ref alone used to leave the key untouched
    only_f = new.CanonicalFrame(height=256, width=256, f_ref=400.0)
    check("changing ONLY f_ref now moves the key (it did NOT before)",
          cache_key(ids, {**legacy, **geometry_build_params(only_f)}) != k_legacy,
          {"note": "pre-refactor params carried `size` but never `f_ref`, so a "
                   "changed F_REF produced different pixels under the SAME key"})

    # ---------------------------------------- 6. DELIBERATE FAILURES (direction 2) #
    def must_raise(name, fn, exc=Exception):
        try:
            fn()
        except exc as e:
            check(f"REFUSES {name}", True, f"{type(e).__name__}: {str(e)[:120]}")
        else:
            check(f"REFUSES {name}", False, "no exception raised")

    from tanitad.config import EncoderConfig, flagship4b_smoke_config
    from tanitad.geometry import GeometryMismatch, apply_frame
    from tanitad.models.encoder import ViTEncoder

    must_raise("a frame AND non-default legacy scalars together",
               lambda: new.as_frame(new.CanonicalFrame(128, 128), 256, 999.0))
    must_raise("an unknown projection",
               lambda: new.CanonicalFrame(projection="fisheye"))
    must_raise("a degenerate frame", lambda: new.CanonicalFrame(4, 4))
    must_raise("a non-positive f_ref",
               lambda: new.CanonicalFrame(256, 256, f_ref=0.0))
    must_raise("frame.size on a NON-SQUARE frame",
               lambda: new.CanonicalFrame(256, 640).size)
    must_raise("cylindrical_rectify on the corpus-median (rig-B) intrinsic",
               lambda: new.cylindrical_rectify(
                   torch.zeros(1, 3, 108, 192),
                   new.PHYSICALAI_FRONT_WIDE_FTHETA,
                   new.CanonicalFrame(64, 64)))
    must_raise("an encoder input that does not match the declared geometry",
               lambda: ViTEncoder(EncoderConfig(in_channels=3, image_size=32,
                                                image_width=64, patch_size=8,
                                                d_model=16, depth=1, n_heads=2)
                                  )(torch.zeros(1, 3, 32, 32)))
    must_raise("grid_hw read on a NON-SQUARE token grid",
               lambda: ViTEncoder(EncoderConfig(in_channels=3, image_size=32,
                                                image_width=64, patch_size=8,
                                                d_model=16, depth=1, n_heads=2)
                                  ).grid_hw)
    must_raise("a frame not divisible by the patch size",
               lambda: apply_frame(flagship4b_smoke_config(),
                                   new.CanonicalFrame(60, 60)),
               GeometryMismatch)

    def half_applied():
        cfg = flagship4b_smoke_config()
        cfg.geometry = new.CanonicalFrame(64, 128).to_dict()   # frame moved...
        from tanitad.geometry import assert_geometry_consistent
        assert_geometry_consistent(cfg)                        # ...encoder didn't
    must_raise("a HALF-APPLIED geometry (frame moved, encoder stale)",
               half_applied, GeometryMismatch)

    def half_applied_other_way():
        cfg = flagship4b_smoke_config()
        import dataclasses
        cfg.encoder = dataclasses.replace(cfg.encoder, image_width=128)
        from tanitad.geometry import assert_geometry_consistent
        assert_geometry_consistent(cfg)
    must_raise("a HALF-APPLIED geometry (encoder moved, frame stale)",
               half_applied_other_way, GeometryMismatch)

    out = {
        "artifact": "geometry-configurable fidelity + deliberate-failure suite",
        "date": "2026-07-27",
        "tolerance": "BIT-EXACT (torch.equal) on uint8 output; exact equality "
                     "on integer crop boxes and float f_eff",
        "reference": str(args.ref),
        "checks": rows,
        "n_checks": len(rows),
        "n_failed": sum(1 for r in rows if not r["pass"]),
        "all_pass": ok,
    }
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "checks"}, indent=2))
    for r in rows:
        if not r["pass"]:
            print("FAIL:", r)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
