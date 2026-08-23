"""P1/P2 — DECISION-GRADE non-CC scale-up of the YouTube-IDM harvest.

This is the pilot `harvest.py` extended per the 2026-07-25 scale-up brief. Sayed
committed to non-CC licensing (2026-07-25), which REMOVES the CC-license filter that
capped the pilot at 80 clips (from ~339 CC candidates). The forward-dashcam pool is
now abundant, so this harvester broadens discovery to general (non-CC) forward-facing
dashcam driving video and targets ~500-1000 clips.

WHAT CHANGED vs the pilot harvest.py (and, deliberately, WHAT DID NOT):
  CHANGED  (licensing/yield only):
    * CC gate is OPT-OUT via --allow-noncc (default ON here). The per-video `license`
      field is STILL recorded in every pointer (is_cc + license string) for full
      auditability; we simply no longer REJECT non-CC.
    * Discovery broadened: `ytsearchN:` over general forward-dashcam queries (no CC
      search filter) + optional channel-uploads enumeration (long continuous drives
      = high clean-yield per video). Time-manipulation, duration and shot-cut filters
      are UNCHANGED (we still want clean, continuous, forward-facing footage).
    * Yield caps raised (per-video-clips, max-frames-per-video) for long drive videos.
    * Optional GeoCalib per-video intrinsics (--geocalib-json): if present, use the
      per-video HFOV; else the fixed-HFOV fallback (recorded per pointer -> re-runnable
      with GeoCalib later by re-decoding from the pointers).
  UNCHANGED  (privacy is MANDATORY and preserved verbatim from the pilot):
    * face + license-plate + body Haar blur applied to the FULL-RES frame BEFORE the
      256 downscale (yt_pilot_common.Anonymizer, imported unmodified).
    * source mp4 DELETED immediately after decode; clip frames are transient (deleted
      by pseudo_label after they are encoded to latents). NO raw video / full-res
      frame is ever persisted. Persistent artifacts = latents (non-imagery) +
      pseudo-labels (numbers) + URL/timestamp pointers ("ship pointers, never bytes").
    * if the privacy detector cannot load, Anonymizer RAISES -> harvest refuses to
      store footage (STOP + escalate), exactly as the pilot.

Footprint is bounded by the run_scaleup.sh DRIVER (batched harvest -> pseudo_label
encode+delete -> repeat), so on-disk imagery never exceeds ~one batch of clips.

SIMPLE-TOKEN CLI (drives cleanly over native OpenSSH; all inputs are FILES).
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
from pathlib import Path

import numpy as np
import torch
import yt_dlp

# reuse the EXACT pilot privacy + geometry + pointer code (unmodified)
sys.path.insert(0, "/workspace/tmp/yt_pilot/scripts")
sys.path.insert(0, "/workspace/tmp/yt_scaleup/scripts")
sys.path.insert(0, "/workspace/TanitAD/stack")
import yt_pilot_common as C                                          # noqa: E402

# GeoCalib per-video intrinsics (drop-in for decode_canonical). The GeoCalib agent
# (2026-07-25) MEASURED that YouTube dashcams sit at median ~66.6 deg HFOV (range
# 32-77), so the fixed 100 deg over-crops ~1.4x and inflates pseudo-speed on most
# clips -> running the decision-grade lift on fixed-HFOV bakes a systematic error
# into the headline. decode_canonical_geocalib estimates each video's focal (median
# vFoV over N frames + MAD outlier rejection + confidence gate) and FALLS BACK to
# fixed-HFOV when GeoCalib is low-confidence -> never worse than the pilot. It
# decodes thread_type="NONE" (a threaded PyAV decoder torn down with a live CUDA
# context DEADLOCKS — MEASURED by the GeoCalib agent).
_GEO_OK = False
_GEO_ERR = ""
try:
    from geocalib_intrinsics import GeoCalibEstimator, decode_canonical_geocalib
    _GEO_OK = True
except Exception as _e:                                             # pragma: no cover
    _GEO_ERR = f"{type(_e).__name__}: {_e}"


def geocalib_estimate_frames(mp4_path, anonymizer, n=24, skip_head=15, window=1200):
    """Decode + anonymize ~n frames SPREAD over the first ``window`` frames
    (thread_type NONE) for a per-video GeoCalib estimate.

    Camera intrinsics are constant per video, but GeoCalib's per-FRAME estimate
    quality depends on scene content, so a diverse sample gives a more confident
    aggregate than n CONSECUTIVE frames (MEASURED: consecutive-24 tripped the
    low-confidence fallback on a highway clip that a spread sample called high-conf).
    Spreading over a BOUNDED window (~40 s) keeps the estimate diverse yet AVOIDS
    `estimate_from_video` decoding the ENTIRE video single-thread just to reach
    spread indices (the scale-killer: a 25-min video = ~45k frames/estimate).
    Blur is applied full-res before GeoCalib sees any pixel (privacy)."""
    import av
    frames = []
    stride = max(1, (window - skip_head) // max(1, n))
    try:
        with av.open(str(mp4_path)) as c:
            st = c.streams.video[0]
            st.thread_type = "NONE"
            i = 0
            for fr in c.decode(st):
                if i >= skip_head and (i - skip_head) % stride == 0:
                    rgb = fr.to_ndarray(format="rgb24")
                    frames.append(np.ascontiguousarray(
                        anonymizer(np.ascontiguousarray(rgb))))
                    if len(frames) >= n:
                        break
                i += 1
                if i >= window:
                    break
    except Exception:
        return []
    return frames

# time-manipulation reject list (unchanged from pilot) — we still drop these
BAD_TITLE = ("5x", "10x", "4x speed", "2x speed", "fast forward", "fast-forward",
             "timelapse", "time lapse", "time-lapse", "hyperlapse", "sped up",
             "sped-up", "speed up", "speeded")


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def read_lines(path):
    if not path or not os.path.exists(path):
        return []
    out = []
    for ln in Path(path).read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if ln and not ln.startswith("#"):
            out.append(ln)
    return out


def as_watch_url(token: str) -> str:
    if token.startswith("http"):
        return token
    return f"https://www.youtube.com/watch?v={token}"


def discover_search(queries, per_query, want_total):
    """General (non-CC) search discovery via `ytsearchN:` — more robust and higher
    yield than the CC results-page scrape the pilot used. Per-video license is still
    recorded downstream; discovery no longer restricts to CC."""
    ids, seen = [], set()
    flat = {"quiet": True, "no_warnings": True, "skip_download": True,
            "extract_flat": "in_playlist"}
    for q in queries:
        try:
            with yt_dlp.YoutubeDL(flat) as ydl:
                pl = ydl.extract_info(f"ytsearch{per_query}:{q}", download=False)
        except Exception as e:
            log(f"  search failed [{q}]: {type(e).__name__}: {e}")
            continue
        for e in (pl.get("entries") or []):
            vid = e and e.get("id")
            if vid and vid not in seen:
                seen.add(vid); ids.append(vid)
        log(f"  query [{q}] -> {len(ids)} cumulative candidates")
        if len(ids) >= want_total:
            break
    return ids


def discover_channels(channel_urls, per_channel, want_total):
    """Enumerate a channel's uploads (long continuous forward-dashcam drives yield
    many clean clips per video). Channel URLs are hand-verified forward-facing
    dashcam channels (channels.txt)."""
    ids, seen = [], set()
    flat = {"quiet": True, "no_warnings": True, "skip_download": True,
            "extract_flat": "in_playlist", "playlistend": per_channel}
    for cu in channel_urls:
        url = cu if cu.rstrip("/").endswith(("/videos", "/streams")) else cu.rstrip("/") + "/videos"
        try:
            with yt_dlp.YoutubeDL(flat) as ydl:
                pl = ydl.extract_info(url, download=False)
        except Exception as e:
            log(f"  channel failed [{cu}]: {type(e).__name__}: {e}")
            continue
        for e in (pl.get("entries") or []):
            vid = e and e.get("id")
            if vid and vid not in seen:
                seen.add(vid); ids.append(vid)
        log(f"  channel [{cu}] -> {len(ids)} cumulative channel candidates")
        if len(ids) >= want_total:
            break
    return ids


def bad_title(info) -> bool:
    t = (info.get("title") or "").lower()
    return any(b in t for b in BAD_TITLE)


def per_video_hfov(vid, info, geo_map, default_hfov):
    """GeoCalib per-video HFOV if available, else fixed-HFOV fallback. Returns
    (hfov_deg, source) where source is 'geocalib' or 'fixed'."""
    if geo_map and vid in geo_map:
        rec = geo_map[vid]
        if isinstance(rec, dict) and rec.get("hfov_deg"):
            return float(rec["hfov_deg"]), "geocalib"
        if isinstance(rec, dict) and rec.get("focal_px") and info.get("width"):
            f = float(rec["focal_px"]); w = float(info["width"])
            return math.degrees(2.0 * math.atan(w / (2.0 * f))), "geocalib"
        if isinstance(rec, (int, float)):
            return float(rec), "geocalib"
    return default_hfov, "fixed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/workspace/tmp/yt_scaleup")
    ap.add_argument("--queries-file", default="")
    ap.add_argument("--channels-file", default="")
    ap.add_argument("--seed-file", default="")
    ap.add_argument("--geocalib-json", default="",
                    help="(legacy) precomputed {video_id: {hfov_deg|focal_px}}; superseded by inline GeoCalib")
    ap.add_argument("--no-geocalib", action="store_true",
                    help="force the fixed-HFOV fallback even if geocalib is installed")
    ap.add_argument("--geocalib-est-frames", type=int, default=24,
                    help="frames sampled from the video START for the per-video estimate "
                         "(constant intrinsics -> first-n == spread-n; avoids full-video decode)")
    ap.add_argument("--max-clips", type=int, default=800)       # TOTAL target (resumes via state)
    ap.add_argument("--per-video-clips", type=int, default=30)  # long drives -> many clips
    ap.add_argument("--clip-frames", type=int, default=250)     # 25 s @ 10 Hz (pilot parity)
    ap.add_argument("--max-videos", type=int, default=9999)     # per-invocation cap (driver bounds total)
    ap.add_argument("--per-query", type=int, default=40)
    ap.add_argument("--per-channel", type=int, default=60)
    ap.add_argument("--allow-noncc", action="store_true", default=True)
    ap.add_argument("--cc-only", dest="allow_noncc", action="store_false",
                    help="revert to pilot CC-only behavior")
    ap.add_argument("--hfov-deg", type=float, default=C.DEFAULT_HFOV_DEG)
    ap.add_argument("--cut-thresh", type=float, default=9.0)
    ap.add_argument("--min-duration", type=float, default=60)     # >=1 min continuous
    ap.add_argument("--max-duration", type=float, default=1500)   # <=25 min: reject long drives
    #   at the METADATA gate (zero wasted download bytes; a 480p >25min file can blow the 400MB cap).
    #   We only decode the first max-frames anyway, and one <=25min video already saturates
    #   per-video-clips, so capping here costs no clips/video and favors more-videos = more diversity.
    ap.add_argument("--max-frames-per-video", type=int, default=3000)  # 5 min @10Hz (RAM-bounded; diversity)
    ap.add_argument("--sleep", type=float, default=0.0,
                    help="yt-dlp per-request sleep + between-video sleep (seconds). Set >0 for a "
                         "GENTLE re-run after a YouTube bot-block (high-volume bursts trip it).")
    args = ap.parse_args()

    work = Path(args.work)
    clips_dir = work / "clips"; clips_dir.mkdir(parents=True, exist_ok=True)
    dl_dir = work / "dl"; dl_dir.mkdir(parents=True, exist_ok=True)
    ptr_path = work / "pointers.jsonl"
    state_path = work / "harvest_state.json"
    manifest_path = work / "manifest.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else \
        {"done_videos": [], "n_clips": 0, "geocalib_hits": 0}
    done = set(state["done_videos"])
    clip_id = state["n_clips"]

    anon = C.Anonymizer()   # RAISES if privacy cascades fail to load (refuse-to-store)
    log(f"anonymizer ready: face={len(anon.face)} plate={len(anon.plate)} "
        f"body={len(anon.body)} cascades; allow_noncc={args.allow_noncc}")

    # GeoCalib per-video estimator (one CUDA model per worker, reused across clips).
    use_geo = _GEO_OK and not args.no_geocalib
    est = None
    if use_geo:
        est = GeoCalibEstimator(hfov_fallback_deg=args.hfov_deg)     # distorted, cuda
        log(f"GEOMETRY = GeoCalib per-video (distorted/cuda; fixed-HFOV {args.hfov_deg} deg fallback)")
    else:
        why = "forced --no-geocalib" if args.no_geocalib else f"geocalib import failed [{_GEO_ERR}]"
        log(f"GEOMETRY = fixed-HFOV {args.hfov_deg} deg  ({why})")

    seeds = [as_watch_url(s) for s in read_lines(args.seed_file)]
    queries = read_lines(args.queries_file)
    channels = read_lines(args.channels_file)
    want = max(args.max_videos, (args.max_clips // 2) + 50) * 2
    disc_ch = discover_channels(channels, args.per_channel, want) if channels else []
    disc_q = discover_search(queries, args.per_query, want) if queries else []
    # channels first (higher clean-yield/video), then search, then seeds
    seen = set()
    candidates = []
    for v in ([as_watch_url(x) for x in disc_ch] + [as_watch_url(x) for x in disc_q] + seeds):
        if v not in seen:
            seen.add(v); candidates.append(v)
    log(f"candidates: {len(disc_ch)} channel + {len(disc_q)} search + {len(seeds)} seed "
        f"-> {len(candidates)} unique")

    meta_opts = {"quiet": True, "no_warnings": True, "skip_download": True,
                 "noplaylist": True}
    dl_opts = {"quiet": True, "no_warnings": True, "noplaylist": True,
               "noprogress": True,
               "format": "bv*[height<=480]/b[height<=480]/bv*/b",
               "max_filesize": 400 * 1024 * 1024,
               "outtmpl": str(dl_dir / "%(id)s.%(ext)s")}
    if args.sleep > 0:                     # gentle re-run: throttle yt-dlp requests
        for o in (meta_opts, dl_opts):
            o["sleep_interval"] = args.sleep
            o["max_sleep_interval"] = args.sleep * 2
            o["sleep_interval_requests"] = 1

    rejects = {"not_cc_kept": 0, "bad_title": 0, "duration": 0, "dl_fail": 0,
               "decode_fail": 0, "no_license_field": 0, "cc": 0}
    lic_counts = {}
    geo_conf_counts = {}          # GeoCalib confidence distribution (high/medium/low/None)
    accepted_videos = 0
    n_videos_tried = 0
    geocalib_hits = state.get("geocalib_hits", 0)

    for url in candidates:
        if clip_id >= args.max_clips or accepted_videos >= args.max_videos:
            break
        try:
            with yt_dlp.YoutubeDL(meta_opts) as ydl:
                info = ydl.extract_info(url, download=False)
        except Exception as e:
            log(f"  meta fail {url}: {type(e).__name__}: {e}"); continue
        vid = info.get("id")
        if not vid or vid in done:
            continue
        n_videos_tried += 1
        lic = info.get("license")
        lic_counts[str(lic)] = lic_counts.get(str(lic), 0) + 1
        is_cc = C.is_creative_commons(info)
        # ---- GATE 1: license ----
        if info.get("license") is None:
            rejects["no_license_field"] += 1
        if not args.allow_noncc and not is_cc:
            rejects["cc"] += 1
            log(f"  REJECT non-CC [{vid}] license={lic!r}"); done.add(vid); continue
        if not is_cc:
            rejects["not_cc_kept"] += 1        # kept, but recorded as non-CC
        # ---- GATE 2: not time-manipulated ----
        if bad_title(info):
            rejects["bad_title"] += 1
            log(f"  REJECT time-manipulated [{vid}] {str(info.get('title'))[:50]!r}")
            done.add(vid); continue
        # ---- GATE 3: duration ----
        dur = info.get("duration") or 0
        if dur < args.min_duration or dur > args.max_duration:
            rejects["duration"] += 1; done.add(vid); continue

        # ---- download (bounded) ----
        try:
            with yt_dlp.YoutubeDL(dl_opts) as ydl:
                ydl.download([url])
        except Exception as e:
            rejects["dl_fail"] += 1
            log(f"  dl fail [{vid}]: {type(e).__name__}: {e}"); done.add(vid); continue
        mp4s = list(dl_dir.glob(f"{vid}.*"))
        if not mp4s:
            rejects["dl_fail"] += 1; done.add(vid); continue
        mp4 = str(mp4s[0])

        # ---- decode + anonymize + canonical crop (GeoCalib per-video, or fixed HFOV) ----
        conf = None; intr = {}
        try:
            anon.reset()
            if use_geo:
                # FAST per-video estimate from the first ~N frames (constant intrinsics),
                # then thread_type=NONE decode+crop at that focal. estimate_from_frames
                # keeps the module's robust median-vFoV + MAD reject + confidence gate +
                # fixed-HFOV fallback; passing `estimated=` skips estimate_from_video's
                # full-video decode. Blur full-res before GeoCalib/crop (privacy).
                ef = geocalib_estimate_frames(mp4, anon, n=args.geocalib_est_frames)
                estimated = est.estimate_from_frames(ef) if ef else None
                anon.reset()   # clear carry-forward from the estimate pass
                vid_u8, meta = decode_canonical_geocalib(
                    mp4, anon, estimator=est, estimated=estimated,
                    max_frames=args.max_frames_per_video,
                    hfov_fallback_deg=args.hfov_deg)
                intr = meta.get("intrinsics", {}) or {}
                hfov = meta.get("hfov_used_deg", args.hfov_deg)
                conf = intr.get("confidence")
                geosrc = "geocalib_fallback" if meta.get("geocalib_fallback_used") else "geocalib"
                if geosrc == "geocalib":
                    geocalib_hits += 1
                geo_conf_counts[str(conf)] = geo_conf_counts.get(str(conf), 0) + 1
            else:
                vid_u8, meta = C.decode_canonical(
                    mp4, anon, hfov_deg=args.hfov_deg,
                    max_frames=args.max_frames_per_video)
                hfov = args.hfov_deg; geosrc = "fixed"
        except Exception as e:
            rejects["decode_fail"] += 1
            log(f"  decode fail [{vid}]: {type(e).__name__}: {e}")
            try: os.remove(mp4)
            except OSError: pass
            done.add(vid); continue
        finally:
            for m in dl_dir.glob(f"{vid}.*"):        # DELETE source video ALWAYS
                try: os.remove(m)
                except OSError: pass

        # ---- segment -> clips ----
        T = vid_u8.shape[0]; cf = args.clip_frames; made = 0
        with open(ptr_path, "a", encoding="utf-8") as pf:
            for start in range(0, T - cf + 1, cf):
                if clip_id >= args.max_clips or made >= args.per_video_clips:
                    break
                seg = vid_u8[start:start + cf]
                cut = C.shotcut_score(seg)
                if cut > args.cut_thresh:
                    continue                          # drop spliced / scene-cut clip
                stacked = C.stack_frames(seg, C.N_STACK)
                n = stacked.shape[0]
                clip_path = clips_dir / f"clip_{clip_id:05d}.pt"
                torch.save({"frames_u8": stacked,
                            "poses": torch.zeros(n, 4),
                            "actions": torch.zeros(n, 2),
                            "video_id": vid, "clip_id": clip_id}, clip_path)
                ptr = C.clip_pointer(info, clip_id, start, n, C.TARGET_HZ, meta,
                                     extra={"shotcut_score": round(cut, 2),
                                            "clip_path": str(clip_path),
                                            "geometry_source": geosrc,
                                            "hfov_used_deg": round(float(hfov), 2),
                                            "geocalib_vfov_deg": intr.get("vfov_deg"),
                                            "geocalib_confidence": conf,
                                            "geocalib_fallback_used": intr.get("fallback_used"),
                                            "geocalib_vfov_mad_deg": intr.get("vfov_mad_deg"),
                                            "achieved_f_eff": meta.get("achieved_f_eff"),
                                            "fully_canonical": meta.get("fully_canonical"),
                                            "is_cc": is_cc, "license": lic})
                pf.write(json.dumps(ptr) + "\n")
                clip_id += 1; made += 1
        del vid_u8
        accepted_videos += 1 if made else 0
        done.add(vid)
        state.update(done_videos=sorted(done), n_clips=clip_id, geocalib_hits=geocalib_hits)
        state_path.write_text(json.dumps(state))
        an = meta.get("anon") or {}
        log(f"  [{vid}] {str(info.get('title',''))[:40]!r} dur={dur}s cc={is_cc} "
            f"geo={geosrc}/{conf} hfov={float(hfov):.0f} -> {made} clips (total {clip_id}) "
            f"anon f/p/b={an.get('faces')}/{an.get('plates')}/{an.get('bodies')}")

    manifest = {
        "experiment": "youtube_idm_scaleup_harvest",
        "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_clips": clip_id, "accepted_videos": accepted_videos,
        "videos_tried": n_videos_tried, "rejects": rejects,
        "license_distribution": lic_counts,
        "allow_noncc": args.allow_noncc,
        "clip_frames": args.clip_frames,
        "geometry": ("geocalib_per_video" if use_geo else f"fixed_hfov_{args.hfov_deg:g}"),
        "geocalib_enabled": bool(use_geo),
        "geocalib_confident_hits": geocalib_hits,
        "geocalib_confidence_distribution": geo_conf_counts,
        "hfov_fallback_deg": args.hfov_deg, "cut_thresh": args.cut_thresh,
        "privacy": "faces+plates+bodies Haar-blurred at full-res before 256 downscale; "
                   "no raw video / full-res frames persisted; clip frames are transient "
                   "(deleted after encode); pointers+latents+pseudo-labels only.",
        "license_gate": ("none (non-CC allowed; license recorded per pointer)"
                         if args.allow_noncc else C.CC_LICENSE),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    log(f"HARVEST DONE: {clip_id} clips from {accepted_videos} videos this call; "
        f"rejects={rejects}; licenses={lic_counts}")
    log(f"WROTE {manifest_path}")
    log("YT_SCALEUP_HARVEST_DONE")


if __name__ == "__main__":
    main()
