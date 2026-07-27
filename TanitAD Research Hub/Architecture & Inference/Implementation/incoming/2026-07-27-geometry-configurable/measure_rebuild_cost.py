"""(A) SELECTION-vs-CACHE verdict and (B) MEASURED rebuild cost.

(A) is the question the coordinator flagged as most consequential: does changing
the crop/resolution invalidate only the FEATURE CACHE, or also the EPISODE
SELECTION? It is answered here by EXECUTING the selection chain, not by reading
it: the ordered source list is built at several geometries and compared, the
cache keys are recomputed, and the skip-marker behaviour is exercised with a
deliberately failing builder.

(B) times the geometry-dependent work on REAL PhysicalAI clips. ⚠️ The local
clip set is the NON-PARITY selection (cache key ``14231cd29c74``); it is used
here ONLY to measure per-clip cost, never to produce a training corpus. No clip
UUID is written to any artifact (PhysicalAI-AV is gated-confidential): clips are
reported as an index and a sha1 prefix of the id.

Cost is decomposed into DECODE (geometry-independent — the same mp4 bytes) and
REMAP (geometry-dependent — crop/rectify + resize), so the pod extrapolation can
substitute the pod's own decode throughput instead of inheriting this box's.

Usage:
  python measure_rebuild_cost.py --root <physicalai root> --clips 6 --out <json>
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import sys
import time
from pathlib import Path

import torch

STACK = Path(__file__).resolve().parents[5] / "stack"
sys.path.insert(0, str(STACK))

from tanitad.data import calib as C                                # noqa: E402
from tanitad.data.epcache import build_episodes_cached, cache_key  # noqa: E402
from tanitad.data.physicalai import (DEFAULT_PROJECTION_MODE,      # noqa: E402
                                     discover_r0_clips,
                                     geometry_build_params,
                                     intrinsics_for_clip, label_params,
                                     split_clips)

# The candidate SPACE the PI named (100-120 deg) plus the deployed frame.
# ⛔ THIS IS NOT A RECOMMENDATION. Choosing among these is the FOV audit's
# deliverable (…/incoming/2026-07-27-fov-crop-audit/). Costed here so the choice
# has numbers under it.
def candidates() -> list[tuple[str, C.CanonicalFrame, str]]:
    F = C.CanonicalFrame
    return [
        ("deployed_256sq_51deg", C.CANONICAL_256, "ftheta_crop"),
        ("100deg_256x256_pin", F.from_hfov(100.0, 256, 256), "ftheta_crop"),
        ("100deg_256x640_pin", F.from_hfov(100.0, 256, 640), "ftheta_crop"),
        ("100deg_256x640_cyl", F.from_hfov(100.0, 256, 640, "cylindrical"),
         "cylindrical"),
        ("120deg_256x640_pin", F.from_hfov(120.0, 256, 640), "ftheta_crop"),
        ("120deg_256x640_cyl", F.from_hfov(120.0, 256, 640, "cylindrical"),
         "cylindrical"),
        ("120deg_384x960_cyl", F.from_hfov(120.0, 384, 960, "cylindrical"),
         "cylindrical"),
        ("120deg_384x384_cyl", F.from_hfov(120.0, 384, 384, "cylindrical"),
         "cylindrical"),
    ]


def _rss_mb() -> float | None:
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 1e6
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# (A) SELECTION vs CACHE                                                        #
# --------------------------------------------------------------------------- #
def selection_verdict(root: str) -> dict:
    out: dict = {}

    # A1. The selection chain's SIGNATURES carry no geometry at all.
    out["selection_signatures"] = {
        "discover_r0_clips": str(inspect.signature(discover_r0_clips)),
        "split_clips": str(inspect.signature(split_clips)),
    }
    out["selection_takes_no_geometry_argument"] = not any(
        p in str(inspect.signature(f))
        for f in (discover_r0_clips, split_clips)
        for p in ("size", "frame", "f_ref", "projection"))

    # A2. EXECUTE it. The ordered id list must be identical at every geometry
    #     (it is produced before any geometry is consulted).
    clips = discover_r0_clips(root)
    tr, va = split_clips(clips, val_frac=0.2, seed=0)
    ids_tr = [c["clip_id"] for c in tr]
    ids_va = [c["clip_id"] for c in va]
    digest = lambda ids: hashlib.sha256("\n".join(ids).encode()).hexdigest()
    out["local_selection"] = {
        "note": "NON-PARITY local selection, used only to exercise the chain",
        "n_clips": len(clips), "n_train": len(tr), "n_val": len(va),
        "ordered_train_id_sha256": digest(ids_tr),
        "ordered_val_id_sha256": digest(ids_va),
    }

    # A3. Cache keys at every candidate. `ids` is the SAME list every time; only
    #     `params` moves. This is the whole mechanism in one table.
    base = {"size": 256, "n_stack": 3, "hz": 10, "calib": "ftheta_v2",
            **label_params()}
    rows = []
    for name, frame, mode in candidates():
        frag = geometry_build_params(frame, mode)
        p = {**base, **frag}
        rows.append({
            "candidate": name, "frame": frame.to_dict(), "tag": frame.tag(),
            "projection_mode": mode, "params_fragment": frag,
            "hfov_deg": round(frame.hfov_deg, 3),
            "train_cache_key": cache_key(tr, p),
            "params_identical_to_legacy": p == base,
        })
    out["cache_keys"] = rows
    out["canonical_key_unchanged"] = rows[0]["params_identical_to_legacy"]
    out["all_candidate_keys_distinct"] = (
        len({r["train_cache_key"] for r in rows}) == len(rows))

    # A4. Episode IDENTITY under a change of geometry: same sources -> same
    #     ep_%05d.pt uids and the SAME skip indices, because the index is the
    #     position in the source list and the skip is written by the builder's
    #     exception, which no geometry can reach. Exercised with a builder that
    #     fails on fixed indices, at two geometries.
    import tempfile

    from tanitad.data.toy_driving import ToyEpisode
    fail_at = {3, 7}
    runs = {}
    for label, hw in (("canonical_256sq", (256, 256)), ("wide_256x640", (256, 640))):
        with tempfile.TemporaryDirectory() as td:
            srcs = [{"clip_id": f"c{i:03d}"} for i in range(12)]

            def build_one(s, hw=hw):
                i = int(s["clip_id"][1:])
                if i in fail_at:
                    raise RuntimeError("synthetic corrupt clip (decode failure)")
                return ToyEpisode(
                    frames=torch.zeros(5, 9, hw[0], hw[1], dtype=torch.uint8),
                    actions=torch.zeros(5, 2), poses=torch.zeros(5, 4),
                    episode_id=i)

            eps = build_episodes_cached(srcs, build_one, td, "geomtest",
                                        {"geom": label})
            d = next(Path(td).glob("geomtest-*"))
            runs[label] = {
                "episode_uids": sorted(p.name for p in d.glob("ep_*.pt")),
                "skip_indices": sorted(int(p.name.split("_")[1])
                                       for p in d.glob("skip_*")),
                "n_episodes": len(eps),
                "T_per_episode": int(eps[0].frames.shape[0]),
                "frame_hw": list(eps[0].frames.shape[-2:]),
            }
    a, b = runs["canonical_256sq"], runs["wide_256x640"]
    out["identity_under_geometry_change"] = {
        "runs": runs,
        "episode_uid_sets_identical": a["episode_uids"] == b["episode_uids"],
        "skip_indices_identical": a["skip_indices"] == b["skip_indices"],
        "episode_counts_identical": a["n_episodes"] == b["n_episodes"],
        "T_identical": a["T_per_episode"] == b["T_per_episode"],
        "only_the_pixels_differ": a["frame_hw"] != b["frame_hw"],
        "uid_sha256_canonical": digest(a["episode_uids"]),
        "uid_sha256_wide": digest(b["episode_uids"]),
    }

    # A5. Does the PARITY GUARD accept a re-cropped cache? (The operational
    #     consequence: the guard keys on the DIRECTORY NAME.)
    from tanitad.data import parity
    newname = f"physicalai-train-{rows[2]['train_cache_key']}"
    out["parity_guard"] = {
        "parity_train_key": parity.PARITY_TRAIN_KEY,
        "corpus_key_of_recropped_dir": parity.corpus_key_of(
            Path(tempfile.gettempdir()) / newname),
        "recropped_dir_is_recognised": parity.corpus_key_of(
            Path(tempfile.gettempdir()) / newname) is not None,
        "note": "corpus_key_of() substring-matches REGISTERED keys. A re-cropped "
                "cache has a new key, so it reads as NON-PARITY and any caller "
                "with require=True (train_flagship_v4) REFUSES it until the new "
                "key is registered in parity_manifest.json.",
    }
    ent = parity.manifest_entry(parity.PARITY_TRAIN_KEY)
    out["parity_manifest"] = {
        "episode_count": ent["episode_count"],
        "skip_count": ent["skip_count"],
        "episode_uid_sha256": ent["episode_uid_sha256"],
        "uid_kind": ent["uid_kind"],
        "note": "uid_kind == epcache_basename: identity is ep_%05d.pt, i.e. the "
                "POSITION in the ordered source list. Geometry cannot move it.",
    }
    return out


# --------------------------------------------------------------------------- #
# (B) REBUILD COST on real clips                                                #
# --------------------------------------------------------------------------- #
def decode_raw(mp4: Path, max_frames: int) -> torch.Tensor:
    """Decode-only: geometry-independent, timed separately."""
    import av
    out = []
    with av.open(str(mp4)) as c:
        st = c.streams.video[0]
        st.thread_type = "AUTO"
        st.thread_count = 4
        for fr in c.decode(st):
            out.append(torch.from_numpy(fr.to_ndarray(format="rgb24")
                                        ).permute(2, 0, 1))
            if len(out) >= max_frames:
                break
    return torch.stack(out)


def cost(root: str, n_clips: int, max_frames: int, n_stack: int) -> dict:
    clips = discover_r0_clips(root)
    sample = clips[:n_clips]
    cands = candidates()
    per_cand: dict[str, dict] = {n: {"remap_s": [], "out_bytes_per_frame": None,
                                     "observed_frac": [], "f_eff": [],
                                     "crop_hw": []}
                                 for n, _, _ in cands}
    decode_s, native_hw, n_frames_seen, clip_rows = [], None, [], []

    for i, clip in enumerate(sample):
        cid = clip["clip_id"]
        t0 = time.perf_counter()
        vid = decode_raw(Path(clip["mp4"]), max_frames)
        decode_s.append(time.perf_counter() - t0)
        native_hw = list(vid.shape[-2:])
        n_frames_seen.append(int(vid.shape[0]))
        intr = intrinsics_for_clip(cid, root)
        clip_rows.append({"index": i,
                          "clip_sha1_8": hashlib.sha1(cid.encode()).hexdigest()[:8],
                          "per_clip_calib": bool(intr.per_clip),
                          "rig": "B" if intr.cy > 650 else "A",
                          "frames_decoded": int(vid.shape[0])})
        for name, frame, mode in cands:
            rss0 = _rss_mb()
            t0 = time.perf_counter()
            if mode == "cylindrical":
                out = C.cylindrical_rectify(vid, intr, frame,
                                            require_per_clip=intr.per_clip)
                obs = C.cylindrical_rectify.last_observed_frac
                feff = C.cylindrical_rectify.last_f_eff
                ch, cw = frame.height, frame.width
            else:
                out = C.ftheta_crop_resize(vid, intr, frame=frame)
                obs = 1.0
                feff = C.ftheta_crop_resize.last_f_eff
                ch, cw = C.ftheta_crop_size_hw(intr, frame)
            dt = time.perf_counter() - t0
            rss1 = _rss_mb()
            r = per_cand[name]
            r["remap_s"].append(dt / max(1, int(vid.shape[0])))
            r["out_bytes_per_frame"] = int(out.shape[-1] * out.shape[-2] * 3)
            r["observed_frac"].append(float(obs))
            r["f_eff"].append(float(feff))
            r["crop_hw"].append([int(ch), int(cw)])
            if rss0 is not None:
                r.setdefault("rss_delta_mb", []).append(round(rss1 - rss0, 1))
            del out

    mean = lambda xs: sum(xs) / max(1, len(xs))
    rows = []
    for name, frame, mode in cands:
        r = per_cand[name]
        # storage: an episode is [T, 3*n_stack, H, W] uint8, T ~ frames at 10 Hz
        bpf = r["out_bytes_per_frame"] * n_stack
        rows.append({
            "candidate": name, "frame": frame.to_dict(), "tag": frame.tag(),
            "projection_mode": mode,
            "hfov_deg": round(frame.hfov_deg, 3),
            "vfov_deg": round(frame.vfov_deg, 3),
            "n_tokens_patch16": (frame.height // 16) * (frame.width // 16),
            "native_crop_hw_mean": [round(mean([c[0] for c in r["crop_hw"]]), 1),
                                    round(mean([c[1] for c in r["crop_hw"]]), 1)],
            "requested_f_ref": round(frame.f_ref, 3),
            "achieved_f_eff_mean": round(mean(r["f_eff"]), 3),
            # ⭐ A crop that CLAMPS at the sensor edge does not widen the field —
            # it ZOOMS, silently. This column is the field actually delivered.
            "achieved_hfov_deg": round(
                2 * math.degrees(
                    math.atan((frame.width / 2.0) / mean(r["f_eff"]))
                    if frame.projection == "pinhole"
                    else (frame.width / 2.0) / mean(r["f_eff"])), 3),
            "field_shortfall_deg": round(
                frame.hfov_deg - 2 * math.degrees(
                    math.atan((frame.width / 2.0) / mean(r["f_eff"]))
                    if frame.projection == "pinhole"
                    else (frame.width / 2.0) / mean(r["f_eff"])), 3),
            "observed_frac_mean": round(mean(r["observed_frac"]), 5),
            "remap_s_per_frame": round(mean(r["remap_s"]), 5),
            "remap_rel_to_deployed": None,
            "bytes_per_stacked_frame": bpf,
            # Peak float32 intermediate inside one decode batch (PAI_DECODE_BATCH
            # =24). COMPUTED from the measured crop, not sampled: the crop path
            # upcasts only the crop; the cylindrical path grid_samples the FULL
            # native frame, so its intermediate is the whole 1080x1920.
            # ⚠️ The sampled RSS delta is reported but NOT quotable — the torch
            # caching allocator makes it non-monotone (values from 7 to 182 MB
            # for arithmetically similar work). Use the computed column.
            "peak_float_intermediate_mb_batch24": round(
                (native_hw[0] * native_hw[1] if mode == "cylindrical"
                 else mean([c[0] for c in r["crop_hw"]])
                 * mean([c[1] for c in r["crop_hw"]]))
                * 3 * 24 * 4 / 1e6, 1),
            "rss_delta_mb_mean_NOT_QUOTABLE": (round(mean(r["rss_delta_mb"]), 1)
                                               if r.get("rss_delta_mb") else None),
            **C.projection_density_report(frame),
        })
    base_remap = rows[0]["remap_s_per_frame"]
    base_bytes = rows[0]["bytes_per_stacked_frame"]
    for r in rows:
        r["remap_rel_to_deployed"] = round(r["remap_s_per_frame"] / base_remap, 3)
        r["bytes_rel_to_deployed"] = round(r["bytes_per_stacked_frame"]
                                           / base_bytes, 3)
    return {
        "machine": "dev box (Windows 11, CPU decode+remap; NO GPU involved, so "
                   "the WDDM host-RAM spill artefact does not apply to any "
                   "number here)",
        "n_clips_sampled": len(sample),
        "max_frames_per_clip": max_frames,
        "frames_decoded_mean": round(mean(n_frames_seen), 1),
        "native_frame_hw": native_hw,
        "decode_s_per_frame": round(mean(decode_s) / max(1, mean(n_frames_seen)), 5),
        "decode_is_geometry_independent": True,
        "clips": clip_rows,
        "candidates": rows,
    }


def extrapolate(costs: dict, episodes_train: int, episodes_val: int,
                frames_per_episode: float, decoded_frames_per_episode: float
                ) -> list[dict]:
    """Whole-corpus projection. STATED EXTRAPOLATION, and the two frame counts
    are DIFFERENT — conflating them understates the build by ~3x:

      work    per episode = (decode s/frame + remap s/frame)
                            * DECODED_frames_per_episode   [605: the clip is
                            decoded in full, then resampled]
      storage per episode = bytes_per_stacked_frame * STORED_frames_per_episode
                            [199: 10 Hz over ~20.2 s, minus n_stack-1]

    Whole corpus = per-episode * (train + val), divided by the number of
    parallel build workers. Decode dominates and is geometry-INDEPENDENT, so the
    geometry delta is the remap column alone.

    ⚠️ Wall-clock is from THIS dev box (CPU decode, 4 decoder threads/clip). A
    pod's own decode throughput should be substituted; the remap RATIO is the
    portable number. No GPU is involved anywhere here, so the WDDM
    host-RAM-spill artefact cannot inflate any figure on this page.
    """
    n_ep = episodes_train + episodes_val
    d = costs["decode_s_per_frame"]
    out = []
    for r in costs["candidates"]:
        per_ep_s = (d + r["remap_s_per_frame"]) * decoded_frames_per_episode
        gb = r["bytes_per_stacked_frame"] * frames_per_episode * n_ep / 1e9
        out.append({
            "candidate": r["candidate"],
            "hfov_deg": r["hfov_deg"],
            "seconds_per_episode_1worker": round(per_ep_s, 2),
            "hours_full_corpus_1worker": round(per_ep_s * n_ep / 3600.0, 2),
            "hours_full_corpus_8workers": round(per_ep_s * n_ep / 3600.0 / 8, 2),
            "hours_full_corpus_16workers": round(per_ep_s * n_ep / 3600.0 / 16, 2),
            "cache_size_gb": round(gb, 1),
            "cache_size_rel_to_deployed": r["bytes_rel_to_deployed"],
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--clips", type=int, default=6)
    ap.add_argument("--max-frames", type=int, default=48)
    ap.add_argument("--n-stack", type=int, default=3)
    ap.add_argument("--episodes-train", type=int, default=2376)
    ap.add_argument("--episodes-val", type=int, default=600)
    ap.add_argument("--frames-per-episode", type=float, default=199.0,
                    help="STORED frames: 10 Hz over ~20.2 s, minus n_stack-1. "
                         "MEASURED: parity manifest T_out=199, and a real local "
                         "episode is 117.384 MB == 199*9*256*256 bytes exactly.")
    ap.add_argument("--decoded-frames-per-episode", type=float, default=605.0,
                    help="DECODED frames: the whole clip is decoded, then "
                         "resampled. MEASURED on 5 real clips: 604-605 frames "
                         "(20.13-20.17 s at 30.0 fps).")
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-cost", action="store_true")
    args = ap.parse_args()

    res = {"artifact": "selection-vs-cache verdict + measured rebuild cost",
           "date": "2026-07-27",
           "confidentiality": "PhysicalAI-AV is gated; no clip UUID appears here "
                              "(clips are index + sha1 prefix only)."}
    res["selection"] = selection_verdict(args.root)
    if not args.skip_cost:
        res["cost"] = cost(args.root, args.clips, args.max_frames, args.n_stack)
        res["extrapolation"] = {
            "method": extrapolate.__doc__.strip(),
            "episodes_train": args.episodes_train,
            "episodes_val": args.episodes_val,
            "stored_frames_per_episode": args.frames_per_episode,
            "decoded_frames_per_episode": args.decoded_frames_per_episode,
            "rows": extrapolate(res["cost"], args.episodes_train,
                                args.episodes_val, args.frames_per_episode,
                                args.decoded_frames_per_episode),
        }
    Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    s = res["selection"]
    print("=== SELECTION VERDICT ===")
    print("selection takes no geometry arg :",
          s["selection_takes_no_geometry_argument"])
    print("canonical key UNCHANGED         :", s["canonical_key_unchanged"])
    print("candidate keys all distinct     :", s["all_candidate_keys_distinct"])
    i = s["identity_under_geometry_change"]
    print("episode uids identical          :", i["episode_uid_sets_identical"])
    print("skip indices identical          :", i["skip_indices_identical"])
    print("only the pixels differ          :", i["only_the_pixels_differ"])
    print("re-cropped dir seen as parity   :",
          s["parity_guard"]["recropped_dir_is_recognised"])
    if not args.skip_cost:
        print("\n=== COST ===")
        for r in res["extrapolation"]["rows"]:
            print(f"  {r['candidate']:24s} {r['hfov_deg']:6.1f}deg  "
                  f"{r['hours_full_corpus_16workers']:6.2f} h @16w  "
                  f"{r['cache_size_gb']:8.1f} GB  "
                  f"(x{r['cache_size_rel_to_deployed']})")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
