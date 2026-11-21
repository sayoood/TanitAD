"""Finalise the 416x1024 cache: merged MANIFEST, ``_geometry.json``, sidecar.

Three artifacts, each answering a question a consumer will actually ask:

``MANIFEST.json``      one row per clip (sha12, n_frames, bytes, sha256, the
                       payload's OWN geometry), merged from the per-shard
                       records the driver archived.
``_geometry.json``     the cache-level record: the canonical frame, the PARITY
                       INGEST GATE re-run over the whole 4,713 (not a union of
                       per-shard gates -- a union answers a different question),
                       the codec/n_stack/projection invariants, the build's
                       provenance, and the BLACK-BOTTOM-ROW census so a consumer
                       can ask "does this episode carry the strip?" of the
                       artifact instead of re-deriving it.
``_digest_scope.json`` what the per-clip ``sha256`` in MANIFEST is TAKEN OVER,
                       declared with ``tanitad.data.join_meta`` -- the bytes of
                       the ``.v2ep.pt`` as it sits on disk (scope
                       ``compressed``). MEASURED 2026-09-05, quoted by that
                       module: two sidecars in this programme recorded digests
                       over DIFFERENT artifacts and the filename could not be
                       used to guess which, so a checker inherited from one
                       REFUSED the other's perfectly good join.

⛔ Geometry is DERIVED here too, never re-typed: ``f_ref`` comes from
``CanonicalFrame.from_hfov`` and is asserted bitwise against the analytic
``(W/2)/radians(HFOV/2)``. HFOV is computed with the CYLINDRICAL relation
(azimuth linear in column); the pinhole formula reads 92.6414 deg on this frame
and would look entirely plausible.
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, statistics, sys, time

HFOV_DEG, N_STACK, CODEC, PROJ = 120.0, 3, "png", "cylindrical"


def sha12(c):
    return hashlib.sha256(c.encode()).hexdigest()[:12]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--root", default="/home/nvidia/data/_b1stage416")
    p.add_argument("--height", type=int, default=416)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--stack", default="/home/nvidia/TanitAD/stack")
    a = p.parse_args()
    sys.path.insert(0, a.stack)

    from tanitad.data.calib import CanonicalFrame
    fr = CanonicalFrame.from_hfov(HFOV_DEG, a.height, a.width, PROJ)
    want = (a.width / 2.0) / math.radians(HFOV_DEG / 2.0)
    assert fr.f_ref == want, f"f_ref {fr.f_ref!r} != analytic {want!r}"
    hfov = math.degrees(2.0 * (a.width / 2.0) / fr.f_ref)
    vfov = math.degrees(2.0 * math.atan((a.height / 2.0) / fr.f_ref))
    assert abs(hfov - HFOV_DEG) < 1e-9, f"HFOV {hfov}"
    ref = CanonicalFrame.from_hfov(HFOV_DEG, 256, 640, PROJ)

    # ---- merge the shard records --------------------------------------- #
    rows, seen = [], set()
    shd = os.path.join(a.cache, "_shards")
    for f in sorted(os.listdir(shd)):
        if not (f.startswith("shard_") and f.endswith(".json")):
            continue
        for r in json.load(open(os.path.join(shd, f))).get("clips", []):
            if r["clip_sha12"] in seen:
                continue
            seen.add(r["clip_sha12"])
            rows.append(r)
    rows.sort(key=lambda r: r["clip_sha12"])
    # ⛔ the record must match the DISK, positively, per path.
    on_disk = {f[: -len(".v2ep.pt")] for f in os.listdir(a.cache)
               if f.endswith(".v2ep.pt")}
    disk12 = {sha12(c) for c in on_disk}
    recorded_not_on_disk = sorted(seen - disk12)
    on_disk_not_recorded = sorted(disk12 - seen)

    ids = [l.strip() for l in open(os.path.join(a.root, "clips_4713.txt"))
           if l.strip()]

    # ---- the parity ingest gate, over the WHOLE set --------------------- #
    from tanitad.data import parity
    parity.require_ingest_gate("finalize_cache")
    kept, gate = parity.guard_corpus_build(
        ids, label=f"finalize_cache -> {a.cache}", role="train", mode="refuse")
    try:
        leak = sorted(sha12(c) for c in parity.clips_in_parity_train(ids))
        vleak = sorted(sha12(c) for c in parity.clips_in_deployed_val(ids))
    except Exception as e:                                       # noqa: BLE001
        leak, vleak = [f"UNAVAILABLE: {type(e).__name__}"], []
    gate = dict(gate, overlap_sha12_parity_train=leak,
                overlap_sha12_deployed_val=vleak,
                clean_subset_n=len(ids) - len(leak))

    by = [r["bytes"] for r in rows]
    cen_p = os.path.join(a.cache, "_black_rows_census.json")
    cen = json.load(open(cen_p)) if os.path.exists(cen_p) else None
    blk = None
    if cen:
        blk = {k: cen[k] for k in (
            "definition", "frames_sampled_per_clip", "n_clips",
            "n_carrying_strip", "n_clean", "frac_carrying_strip",
            "strip_rows_min", "strip_rows_max", "strip_rows_mean",
            "strip_rows_median", "strip_frac_of_height_mean",
            "n_not_constant_across_frames", "histogram_rows_to_count",
            "bimodal_check", "rig_correlation") if k in cen}
        blk["per_clip_sha12"] = {k: v["black_bottom_rows"]
                                 for k, v in cen["per_clip"].items()}
        blk["provenance"] = (
            "census_black_rows.py over THIS cache; retraction class C26 "
            "(calib.py:1031-1036) -- a rig-correlated BLACK region is still a "
            "rig-correlated signal. RECORDED, NOT MITIGATED: the geometry is "
            "the PI's 416x1024 ruling and is unchanged; the mitigation is a "
            "loader/model-side mask owned by another stream.")

    geo = {
        "cache": os.path.basename(a.cache.rstrip("/")),
        "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "builder": "TanitAD Research Lab/Data Engineering/Research/"
                   "2026-09-16-256x1024-cache/code/build_v2ep_wide.py "
                   "(UNMODIFIED), driven by 2026-09-23-b1-416x1024-cache/code/"
                   "run_b1_416_build.py",
        "resampler": "v2_compressed.build_compressed -> "
                     "tanitad.data.calib.cylindrical_rectify (UNMODIFIED)",
        "source_repo": "Sayood/tanitad-v7-training-corpus",
        "source_revision": "a0cf20dfb4eafa29b0ac1f3c05337f002bc33ca0",
        "image_h": a.height, "image_w": a.width,
        "projection_mode": PROJ, "codec": CODEC, "n_stack": N_STACK,
        "quality": None,
        "selection_parquet": os.path.join(a.root, "r0", "r0_selection.parquet"),
        "clips_requested": len(ids),
        "frame": {"height": a.height, "width": a.width, "f_ref": float(fr.f_ref),
                  "projection": PROJ, "hfov_deg": hfov, "vfov_deg": vfov,
                  "deg_per_col": hfov / a.width,
                  "centre_col": (a.width - 1) / 2.0, "tag": fr.tag()},
        "geometry_check": {
            "requested_hfov_deg": HFOV_DEG, "achieved_hfov_deg": round(hfov, 10),
            "f_ref": float(fr.f_ref),
            "f_ref_analytic_512_over_radians60": want,
            "f_ref_agrees_bitwise": fr.f_ref == want,
            "vfov_deg": round(vfov, 10),
            "reference_256x640_vfov_deg": round(
                math.degrees(2 * math.atan((256 / 2) / ref.f_ref)), 10),
            "vertical_field_vs_256x640_deg": round(
                vfov - math.degrees(2 * math.atan((256 / 2) / ref.f_ref)), 10),
            "hfov_pinhole_formula_WRONG_here": round(
                math.degrees(2 * math.atan((a.width / 2) / fr.f_ref)), 4),
            "hfov_from_rounded_f_ref_488_92": round(
                math.degrees(2.0 * (a.width / 2.0) / 488.92), 4),
            "note": "CYLINDRICAL: azimuth is LINEAR in column, "
                    "az_max = (W/2)/f_ref. The pinhole formula is wrong here.",
            "height_stride32_rows": a.height / 32,
            "height_divisible_by_32": a.height % 32 == 0},
        "parity_ingest_gate": gate,
        "n_clips_built": len(rows), "n_clips_expected": len(ids),
        "n_missing": len(ids) - len(rows),
        "recorded_not_on_disk_sha12": recorded_not_on_disk[:50],
        "on_disk_not_recorded_sha12": on_disk_not_recorded[:50],
        "record_matches_disk": (not recorded_not_on_disk
                                and not on_disk_not_recorded),
        "total_bytes": sum(by),
        "total_GB_decimal": round(sum(by) / 1e9, 2),
        "total_GiB": round(sum(by) / 1024 ** 3, 2),
        "MB_per_ep_decimal": round(statistics.mean(by) / 1e6, 3) if by else None,
        "MiB_per_ep": round(statistics.mean(by) / 1048576, 3) if by else None,
        "bytes_min": min(by) if by else None, "bytes_max": max(by) if by else None,
        "black_bottom_rows": blk,
    }
    man = {k: geo[k] for k in ("cache", "built_utc", "builder", "resampler",
                               "source_repo", "source_revision", "frame",
                               "codec", "n_stack", "projection_mode",
                               "parity_ingest_gate", "total_bytes")}
    man.update(n_clips=len(rows), n_failed=len(ids) - len(rows),
               build_s_per_clip_mean=round(statistics.mean(
                   [r["build_s"] for r in rows if r.get("build_s", 0) > 0]), 2)
               if rows else None,
               clips=rows)
    json.dump(man, open(os.path.join(a.cache, "MANIFEST.json"), "w"), indent=1)
    json.dump(geo, open(os.path.join(a.cache, "_geometry.json"), "w"), indent=1)

    # ---- the digest-scope sidecar -------------------------------------- #
    from tanitad.data import join_meta
    side = join_meta.attach(
        {"about": "per-clip sha256 in MANIFEST.json of this v2ep cache",
         "cache": geo["cache"], "n_clips": len(rows),
         "algo": "sha256", "per_clip_key": "clips[].sha256"},
        rows[0]["sha256"] if rows else "0" * 64,
        scope="compressed",
        filename=(f"{sorted(on_disk)[0]}.v2ep.pt" if on_disk
                  else "example.v2ep.pt"),
        algo="sha256", declared_by="finalize_cache.py",
        note="Each MANIFEST row's sha256 is taken over the BYTES OF THAT "
             "CLIP'S .v2ep.pt AS IT SITS ON DISK -- the container, not any "
             "stream inside it. The example digest/filename names one row so "
             "the declaration can be checked against a real file.")
    json.dump(side, open(os.path.join(a.cache, "_digest_scope.json"), "w"),
              indent=1)

    brief = {k: v for k, v in geo.items()
             if k not in ("black_bottom_rows", "parity_ingest_gate")}
    brief["parity_ingest_gate"] = {k: gate[k] for k in gate
                                   if not k.startswith("overlap_sha12")}
    if blk:
        brief["black_bottom_rows"] = {k: v for k, v in blk.items()
                                      if k != "per_clip_sha12"}
    print(json.dumps(brief, indent=1), flush=True)
    print(f"ZZFINAL-{len(rows)}-{len(ids)}-{geo['total_GB_decimal']}ZZ",
          flush=True)
    return 0 if geo["record_matches_disk"] and len(rows) == len(ids) else 1


if __name__ == "__main__":
    sys.exit(main())
