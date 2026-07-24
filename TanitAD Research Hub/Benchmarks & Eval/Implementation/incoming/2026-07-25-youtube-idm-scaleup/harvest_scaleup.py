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

import torch
import yt_dlp

# reuse the EXACT pilot privacy + geometry + pointer code (unmodified)
sys.path.insert(0, "/workspace/tmp/yt_pilot/scripts")
sys.path.insert(0, "/workspace/tmp/yt_scaleup/scripts")
sys.path.insert(0, "/workspace/TanitAD/stack")
import yt_pilot_common as C                                          # noqa: E402

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
                    help="optional {video_id: {hfov_deg|focal_px}} from the GeoCalib agent")
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
    ap.add_argument("--max-duration", type=float, default=7200)   # up to 2 h drives
    ap.add_argument("--max-frames-per-video", type=int, default=6000)  # 10 min @10Hz (RAM-bounded)
    args = ap.parse_args()

    work = Path(args.work)
    clips_dir = work / "clips"; clips_dir.mkdir(parents=True, exist_ok=True)
    dl_dir = work / "dl"; dl_dir.mkdir(parents=True, exist_ok=True)
    ptr_path = work / "pointers.jsonl"
    state_path = work / "harvest_state.json"
    manifest_path = work / "manifest.json"
    geo_map = {}
    if args.geocalib_json and os.path.exists(args.geocalib_json):
        geo_map = json.loads(Path(args.geocalib_json).read_text())
        log(f"GeoCalib intrinsics loaded: {len(geo_map)} videos")
    else:
        log(f"GeoCalib intrinsics NOT supplied -> fixed-HFOV fallback ({args.hfov_deg} deg); "
            f"re-runnable from pointers when GeoCalib lands")

    state = json.loads(state_path.read_text()) if state_path.exists() else \
        {"done_videos": [], "n_clips": 0, "geocalib_hits": 0}
    done = set(state["done_videos"])
    clip_id = state["n_clips"]

    anon = C.Anonymizer()   # RAISES if privacy cascades fail to load (refuse-to-store)
    log(f"anonymizer ready: face={len(anon.face)} plate={len(anon.plate)} "
        f"body={len(anon.body)} cascades; allow_noncc={args.allow_noncc}")

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

    rejects = {"not_cc_kept": 0, "bad_title": 0, "duration": 0, "dl_fail": 0,
               "decode_fail": 0, "no_license_field": 0, "cc": 0}
    lic_counts = {}
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

        # ---- decode + anonymize + canonical crop (GeoCalib or fixed HFOV) ----
        hfov, geosrc = per_video_hfov(vid, info, geo_map, args.hfov_deg)
        if geosrc == "geocalib":
            geocalib_hits += 1
        try:
            anon.reset()
            vid_u8, meta = C.decode_canonical(
                mp4, anon, hfov_deg=hfov, max_frames=args.max_frames_per_video)
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
                                            "hfov_used_deg": round(hfov, 2),
                                            "is_cc": is_cc, "license": lic})
                pf.write(json.dumps(ptr) + "\n")
                clip_id += 1; made += 1
        del vid_u8
        accepted_videos += 1 if made else 0
        done.add(vid)
        state.update(done_videos=sorted(done), n_clips=clip_id, geocalib_hits=geocalib_hits)
        state_path.write_text(json.dumps(state))
        log(f"  [{vid}] {str(info.get('title',''))[:45]!r} dur={dur}s cc={is_cc} "
            f"geo={geosrc} -> {made} clips (total {clip_id}) "
            f"anon f/p/b={meta['anon']['faces']}/{meta['anon']['plates']}/{meta['anon']['bodies']}")

    manifest = {
        "experiment": "youtube_idm_scaleup_harvest",
        "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_clips": clip_id, "accepted_videos": accepted_videos,
        "videos_tried": n_videos_tried, "rejects": rejects,
        "license_distribution": lic_counts,
        "allow_noncc": args.allow_noncc,
        "clip_frames": args.clip_frames,
        "geometry": ("geocalib_per_video" if geo_map else f"fixed_hfov_{args.hfov_deg:g}"),
        "geocalib_hits": geocalib_hits,
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
