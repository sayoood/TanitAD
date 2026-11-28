"""Verify the WIDE VAL cache's geometry on REAL DECODED FRAMES — not on the
config that was passed to the builder.

WHY THIS EXISTS
---------------
``parity.py`` §9 and ``V5_TRAINER.md`` §7 both say the same thing plainly:
**nothing hashes pixels.** Membership proves WHICH CLIPS; ``_geometry.json``
and the manifest entry prove WHAT THE BUILDER RECORDED; the payload's
``image_h``/``image_w`` prove the RASTER SIZE. None of them proves the
RESAMPLER produced the field it claims — *"a cache recorded as 120 deg whose
resampler produced 90 deg would pass"*. This script is the only thing in the
chain that looks at the stored bytes.

FOUR PASSES, in increasing strength
-----------------------------------
A. **DECLARATION, every built clip.** ``image_h``/``image_w``/``frame``/
   ``codec``/``projection_mode`` read off each payload (mmap — the PNG buffer
   is never faulted in). Catches a MIXED cache, which a single-clip probe
   cannot. Still only a record.

B. **REAL DECODE.** The stored PNGs are decoded and the tensor's own shape is
   asserted. This is a property of the bytes.

C. ⭐ **THE FOV DISCRIMINATOR — the pass that closes the stated gap.**
   At 120 deg cylindrical on this corpus the requested frame is WIDER than the
   rig-B sensor can observe, so the periphery is *masked* — genuinely zero
   pixels — at a rate the ray map predicts exactly from the per-clip
   intrinsics. At 90 deg the same sensor observes the whole frame and the rate
   is ~0. So the ZERO-PIXEL FRACTION OF THE DECODED PNGs separates 120 deg from
   90 deg **without trusting any record**: we predict both counterfactuals from
   the intrinsics and check which one the pixels match.
   ⚠️ This is a discriminator, not a hash: it is diagnostic wherever the
   predicted rates differ (rig B: ~8.9 % vs ~0 %). On rig A both predictions are
   ~0 %, so rig A contributes shape evidence only — said here rather than
   implied.

D. **GEOMETRY CENSUS over every val clip id**, mirroring the train build's
   published check: achieved HFOV, shortfall vs requested, and ``f_eff``
   (mean/stdev/min/max), computed from each clip's own intrinsics through the
   same ``cylindrical_rectify`` the builder used. n = the whole split.

🔒 Clip ids are gated-confidential: this writes COUNTS, RATES and DIGESTS only.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
import time
from pathlib import Path

V2_SUFFIX = ".v2ep.pt"


def _agg(vals):
    v = [float(x) for x in vals if x is not None]
    if not v:
        return None
    return {"n": len(v), "mean": round(st.mean(v), 6),
            "median": round(st.median(v), 6), "min": round(min(v), 6),
            "max": round(max(v), 6),
            "stdev": round(st.stdev(v), 8) if len(v) > 1 else 0.0}


def scan_declarations(paths, expect_h: int, expect_w: int) -> dict:
    """PASS A — what every payload SAYS it is, read off the bytes on disk.

    Split out so it can be exercised without a pod (``--self-test``): this is
    the pass that runs over the whole split, so a bug here is a bug in the only
    check that can see a MIXED cache.

    ``mmap=True`` keeps the ~40 MB PNG buffer out of memory — only the metadata
    tensors are faulted in — with a plain load as the fallback."""
    import torch
    t0 = time.time()
    shapes: dict = {}
    codecs: dict = {}
    projs: dict = {}
    frefs: dict = {}
    nframes: list = []
    bad: list = []
    for p in paths:
        try:
            try:
                d = torch.load(p, map_location="cpu", weights_only=False,
                               mmap=True)
            except Exception:
                d = torch.load(p, map_location="cpu", weights_only=False)
            hw = (int(d.get("image_h", d.get("image_size", -1))),
                  int(d.get("image_w", d.get("image_size", -1))))
            shapes[hw] = shapes.get(hw, 0) + 1
            codecs[str(d.get("codec"))] = codecs.get(str(d.get("codec")), 0) + 1
            projs[str(d.get("projection_mode"))] = \
                projs.get(str(d.get("projection_mode")), 0) + 1
            fr = d.get("frame") or {}
            key = (round(float(fr.get("f_ref", float("nan"))), 6),
                   str(fr.get("projection")))
            frefs[key] = frefs.get(key, 0) + 1
            nframes.append(int(len(d["jpeg_len"])))
        except Exception as e:                                # noqa: BLE001
            bad.append({"i": len(bad), "error": f"{type(e).__name__}: {e}"})
    return {
        "n": len(list(paths)),
        "distinct_image_hw": {f"{h}x{w}": n for (h, w), n in shapes.items()},
        "distinct_codec": codecs,
        "distinct_projection_mode": projs,
        "distinct_frame_f_ref_projection": {f"{f}|{pr}": n
                                            for (f, pr), n in frefs.items()},
        "frames_per_clip": _agg(nframes),
        "unreadable_payloads": len(bad), "errors": bad[:5],
        "uniform": (len(shapes) == 1 and len(codecs) == 1 and len(projs) == 1
                    and len(frefs) == 1 and not bad),
        "matches_request": (list(shapes) == [(expect_h, expect_w)]),
        "seconds": round(time.time() - t0, 1),
    }


def _hfov_of(frame_w: int, f_eff: float, projection: str) -> float:
    """The builder's own formula (``_assert_geometry_deliverable``), reused so
    the two numbers are comparable rather than merely similar."""
    half = (frame_w / 2) / f_eff
    return math.degrees(2 * (math.atan(half) if projection == "pinhole"
                             else half))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="the built v2 val cache dir")
    ap.add_argument("--root", required=True, help="physicalai_phase0 root")
    ap.add_argument("--expect-h", type=int, default=256)
    ap.add_argument("--expect-w", type=int, default=640)
    ap.add_argument("--expect-hfov", type=float, default=120.0)
    ap.add_argument("--counterfactual-hfov", type=float, default=90.0,
                    help="the field this cache must be shown NOT to be")
    ap.add_argument("--projection-mode", default="cylindrical")
    ap.add_argument("--decode-n", type=int, default=24,
                    help="clips to really decode, balanced across rigs")
    ap.add_argument("--limit", type=int, default=0,
                    help="SMOKE ONLY: cap the clips scanned (0 = all). A capped "
                         "run is NOT a census and says so in the JSON.")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import torch
    from tanitad.data.calib import CanonicalFrame, cylindrical_rectify
    from tanitad.data.physicalai import intrinsics_for_clip

    proj = "cylindrical" if a.projection_mode == "cylindrical" else "pinhole"
    frame = CanonicalFrame.from_hfov(a.expect_hfov, a.expect_h, a.expect_w, proj)
    cf = CanonicalFrame.from_hfov(a.counterfactual_hfov, a.expect_h,
                                  a.expect_w, proj)

    cache = Path(a.cache)
    all_paths = sorted(cache.glob("*" + V2_SUFFIX))
    paths = all_paths[:a.limit] if a.limit else all_paths
    cids = [p.name[:-len(V2_SUFFIX)] for p in paths]
    out: dict = {
        "cache": str(cache), "clips_in_cache": len(all_paths),
        "clips_scanned": len(paths),
        "is_census": (len(paths) == len(all_paths)),
        "smoke_limit": a.limit or None,
        "request": {"height": a.expect_h, "width": a.expect_w,
                    "hfov_deg": a.expect_hfov, "projection_mode": proj},
        "declared_frame": {**frame.to_dict(), "tag": frame.tag(),
                           "hfov_deg": float(frame.hfov_deg)},
        "counterfactual_frame": {**cf.to_dict(), "tag": cf.tag(),
                                 "hfov_deg": float(cf.hfov_deg)},
        "pass_A_declaration": {}, "pass_B_real_decode": [],
        "pass_C_discriminator": {}, "pass_D_geometry_census": {},
    }

    out["pass_A_declaration"] = scan_declarations(paths, a.expect_h, a.expect_w)

    # ---- PASS D (first, it feeds C): geometry census over every val clip --- #
    t0 = time.time()
    probe_cache: dict = {}
    census: list = []
    for cid in cids:
        try:
            intr = intrinsics_for_clip(cid, a.root)
        except Exception as e:                                # noqa: BLE001
            census.append({"error": f"{type(e).__name__}: {e}"})
            continue
        cx = float(getattr(intr, "cx", float("nan")))
        cy = float(getattr(intr, "cy", float("nan")))
        if math.isnan(cy):
            census.append({"error": "intrinsics carry no cy"})
            continue
        k = (round(cx, 3), round(cy, 3), int(intr.height), int(intr.width),
             bool(intr.per_clip))
        if k not in probe_cache:
            probe = torch.zeros(1, 3, intr.height, intr.width, dtype=torch.uint8)
            cylindrical_rectify(probe, intr, frame,
                                require_per_clip=intr.per_clip)
            f_eff = float(cylindrical_rectify.last_f_eff)
            obs = float(cylindrical_rectify.last_observed_frac)
            cylindrical_rectify(probe, intr, cf, require_per_clip=intr.per_clip)
            cf_obs = float(cylindrical_rectify.last_observed_frac)
            cf_feff = float(cylindrical_rectify.last_f_eff)
            probe_cache[k] = (f_eff, obs, cf_feff, cf_obs)
        f_eff, obs, cf_feff, cf_obs = probe_cache[k]
        census.append({
            "cx": cx, "cy": cy, "per_clip": bool(intr.per_clip),
            "sensor_hw": [int(intr.height), int(intr.width)],
            "f_eff": f_eff, "achieved_hfov_deg": _hfov_of(a.expect_w, f_eff,
                                                          proj),
            "masked_frac": 1.0 - obs,
            "cf_f_eff": cf_feff, "cf_masked_frac": 1.0 - cf_obs,
        })
    ok = [r for r in census if "cy" in r]
    cys = sorted(r["cy"] for r in ok)
    boundary = None
    if len(cys) > 1:
        gaps = [(cys[i + 1] - cys[i], i) for i in range(len(cys) - 1)]
        gmax, gi = max(gaps)
        boundary = (cys[gi] + cys[gi + 1]) / 2.0
        for r in ok:
            r["rig"] = "A" if r["cy"] < boundary else "B"
        out["rig_boundary"] = {"cy_min": cys[0], "cy_max": cys[-1],
                               "largest_gap": round(gmax, 2),
                               "boundary_cy": round(boundary, 2),
                               "bimodal": bool(gmax > 50.0)}
    ach = [r["achieved_hfov_deg"] for r in ok]
    out["pass_D_geometry_census"] = {
        "n": len(ok), "intrinsics_failures": len(census) - len(ok),
        "distinct_sensor_geometries": len(probe_cache),
        "achieved_hfov_deg": _agg(ach),
        "shortfall_deg": _agg([a.expect_hfov - x for x in ach]),
        "f_eff": _agg([r["f_eff"] for r in ok]),
        "by_rig": {
            rg: {"n": sum(1 for r in ok if r.get("rig") == rg),
                 "share": round(sum(1 for r in ok if r.get("rig") == rg)
                                / max(len(ok), 1), 6),
                 "cy": _agg([r["cy"] for r in ok if r.get("rig") == rg]),
                 "masked_frac_predicted_at_request": _agg(
                     [r["masked_frac"] for r in ok if r.get("rig") == rg]),
                 "masked_frac_predicted_at_counterfactual": _agg(
                     [r["cf_masked_frac"] for r in ok if r.get("rig") == rg]),
                 } for rg in ("A", "B")},
        "seconds": round(time.time() - t0, 1),
    }

    # ---- PASS B + C: REAL DECODE of the stored PNGs, balanced across rigs -- #
    byrig = {"A": [], "B": []}
    for i, r in enumerate(ok):
        byrig.get(r.get("rig"), []).append(i)
    pick: list[int] = []
    for rg in ("A", "B"):
        lst = byrig[rg]
        want = min(a.decode_n // 2, len(lst))
        if want:
            pick += [lst[round(j * (len(lst) - 1) / max(want - 1, 1))]
                     for j in range(want)]
    pick = sorted(set(pick))
    # census[] is built by iterating `paths` in order, so the n-th surviving
    # census row and the n-th surviving path are the same clip.
    ok_to_path = [paths[n] for n, r in enumerate(census) if "cy" in r]
    assert len(ok_to_path) == len(ok)
    import torchvision.io as tvio
    t0 = time.time()
    for i in pick:
        rec = {"rig": ok[i].get("rig"),
               "predicted_masked_frac_at_request": ok[i]["masked_frac"],
               "predicted_masked_frac_at_counterfactual":
                   ok[i]["cf_masked_frac"]}
        try:
            d = torch.load(ok_to_path[i], map_location="cpu",
                           weights_only=False)
            lens = d["jpeg_len"]
            offs = torch.cat([torch.zeros(1, dtype=torch.int64),
                              torch.cumsum(lens, 0)])
            buf = d["jpeg_buf"]
            dec = (tvio.decode_png if d.get("codec") == "png"
                   else tvio.decode_jpeg)
            n = int(len(lens))
            take = list(range(0, n, max(n // 12, 1)))[:12]
            imgs = [dec(buf[int(offs[j]):int(offs[j + 1])],
                        mode=tvio.ImageReadMode.RGB) for j in take]
            vid = torch.stack(imgs)                       # [k,3,H,W] REAL PIXELS
            rec["decoded_shape"] = list(vid.shape[1:])
            rec["frames_total"] = n
            rec["frames_decoded"] = int(vid.shape[0])
            allzero = (vid == 0).all(dim=1)
            rec["observed_zero_pixel_frac"] = round(
                float(allzero.float().mean()), 6)
            # column profile: which COLUMNS are masked (the periphery signature)
            colzero = allzero.float().mean(dim=(0, 1))    # [W]
            rec["masked_cols_gt_half"] = int((colzero > 0.5).sum())
            rec["leftmost_col_unmasked"] = int(
                (colzero <= 0.5).nonzero()[0]) if (colzero <= 0.5).any() else -1
            rec["rightmost_col_unmasked"] = int(
                (colzero <= 0.5).nonzero()[-1]) if (colzero <= 0.5).any() else -1
        except Exception as e:                                # noqa: BLE001
            rec["error"] = f"{type(e).__name__}: {e}"
        out["pass_B_real_decode"].append(rec)
        print("[decode] " + json.dumps(rec), flush=True)
    out["pass_B_seconds"] = round(time.time() - t0, 1)

    dec = [r for r in out["pass_B_real_decode"]
           if "observed_zero_pixel_frac" in r]
    shapes_seen = {tuple(r["decoded_shape"][-2:]) for r in dec}
    disc = {}
    for rg in ("A", "B"):
        rows = [r for r in dec if r["rig"] == rg]
        if not rows:
            continue
        obs = _agg([r["observed_zero_pixel_frac"] for r in rows])
        pr = _agg([r["predicted_masked_frac_at_request"] for r in rows])
        pc = _agg([r["predicted_masked_frac_at_counterfactual"] for r in rows])
        d_req = abs(obs["mean"] - pr["mean"])
        d_cf = abs(obs["mean"] - pc["mean"])
        disc[rg] = {
            "n": len(rows), "observed_zero_pixel_frac": obs,
            "predicted_at_request_%.0fdeg" % a.expect_hfov: pr,
            "predicted_at_counterfactual_%.0fdeg" % a.counterfactual_hfov: pc,
            "abs_err_vs_request": round(d_req, 6),
            "abs_err_vs_counterfactual": round(d_cf, 6),
            "separation": round(abs(pr["mean"] - pc["mean"]), 6),
            "diagnostic": bool(abs(pr["mean"] - pc["mean"]) > 0.01),
            "pixels_match": ("request" if d_req < d_cf else "counterfactual"),
        }
    out["pass_C_discriminator"] = {
        "by_rig": disc,
        "decoded_shapes_seen": [list(s) for s in shapes_seen],
        "decoded_shape_matches_request": (
            shapes_seen == {(a.expect_h, a.expect_w)}),
        "verdict": (
            "PIXELS MATCH THE DECLARED %.0f deg FIELD" % a.expect_hfov
            if all(v["pixels_match"] == "request"
                   for v in disc.values() if v["diagnostic"])
            and any(v["diagnostic"] for v in disc.values())
            and shapes_seen == {(a.expect_h, a.expect_w)}
            else "NOT PROVEN — inspect by_rig"),
    }

    Path(a.out).write_text(json.dumps(out, indent=1))
    print("VAL_GEOMETRY_CHECK " + json.dumps({
        "clips_in_cache": out["clips_in_cache"],
        "clips_scanned": out["clips_scanned"], "is_census": out["is_census"],
        "A_uniform": out["pass_A_declaration"]["uniform"],
        "A_hw": out["pass_A_declaration"]["distinct_image_hw"],
        "A_codec": out["pass_A_declaration"]["distinct_codec"],
        "D_achieved_hfov": out["pass_D_geometry_census"]["achieved_hfov_deg"],
        "D_shortfall": out["pass_D_geometry_census"]["shortfall_deg"],
        "D_f_eff": out["pass_D_geometry_census"]["f_eff"],
        "D_rig_n": {rg: out["pass_D_geometry_census"]["by_rig"][rg]["n"]
                    for rg in ("A", "B")},
        "C_verdict": out["pass_C_discriminator"]["verdict"],
        "C_by_rig": {rg: {"obs": v["observed_zero_pixel_frac"]["mean"],
                          "match": v["pixels_match"],
                          "diagnostic": v["diagnostic"]}
                     for rg, v in disc.items()},
    }), flush=True)


if __name__ == "__main__":
    main()
