"""P1/P2 — privacy-safe CC-licensed YouTube forward-dashcam harvest.

Pipeline per source video:
  yt-dlp metadata -> VERIFY Creative-Commons license (reject otherwise) ->
  reject time-manipulated (5x / timelapse / hyperlapse) & out-of-range duration ->
  download <=480p (no audio) -> decode+resample 10Hz + FACE/PLATE BLUR (full-res)
  + canonical focal crop to 256 -> DELETE source mp4 -> segment into fixed clips ->
  shot-cut filter (drop compilation splices) -> 3-frame 9-channel stack ->
  save clip frames_u8 [T,9,256,256] (+ zero poses/actions placeholders) + pointer.

Persistent output (footprint-bounded): clips/clip_XXXXX.pt (transient imagery,
deleted after pseudo_label encodes it), pointers.jsonl, manifest.json. Raw video
and full-res frames are NEVER kept.

SIMPLE-TOKEN CLI (no quotes/parens needed over ssh): inputs are FILES.
  --queries-file  one CC search query per line (harvested via YT CC filter)
  --seed-file     one URL or 11-char video id per line (hand-picked CC sources)
  --max-clips N --per-video-clips M --clip-frames F --max-videos V
  --hfov-deg 100 --cut-thresh 9.0 --min-duration 30 --max-duration 1500
"""
from __future__ import annotations
import argparse, json, os, sys, time, traceback
from pathlib import Path

import torch
import yt_dlp

sys.path.insert(0, "/workspace/TanitAD/stack")
sys.path.insert(0, "/workspace/tmp/yt_pilot/scripts")
import yt_pilot_common as C                                          # noqa: E402

CC_SEARCH_SUFFIX = "&sp=EgIwAQ%3D%3D"        # YouTube 'Creative Commons' filter
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


def discover(queries, per_query, want_total):
    """CC-filtered search -> ordered list of candidate video ids (delegated CC
    verification happens later per-video)."""
    ids, seen = [], set()
    flat = {"quiet": True, "no_warnings": True, "skip_download": True,
            "extract_flat": "in_playlist", "playlistend": per_query}
    for q in queries:
        url = ("https://www.youtube.com/results?search_query="
               + q.replace(" ", "+") + CC_SEARCH_SUFFIX)
        try:
            with yt_dlp.YoutubeDL(flat) as ydl:
                pl = ydl.extract_info(url, download=False)
        except Exception as e:
            log(f"  search failed [{q}]: {type(e).__name__}: {e}")
            continue
        for e in (pl.get("entries") or []):
            vid = e and e.get("id")
            if vid and vid not in seen:
                seen.add(vid)
                ids.append(vid)
        log(f"  query [{q}] -> {len(ids)} cumulative candidates")
        if len(ids) >= want_total:
            break
    return ids


def bad_title(info) -> bool:
    t = (info.get("title") or "").lower()
    return any(b in t for b in BAD_TITLE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/workspace/tmp/yt_pilot")
    ap.add_argument("--queries-file", default="")
    ap.add_argument("--seed-file", default="")
    ap.add_argument("--max-clips", type=int, default=120)
    ap.add_argument("--per-video-clips", type=int, default=8)
    ap.add_argument("--clip-frames", type=int, default=250)      # 25 s @ 10 Hz
    ap.add_argument("--max-videos", type=int, default=60)
    ap.add_argument("--per-query", type=int, default=25)
    ap.add_argument("--hfov-deg", type=float, default=C.DEFAULT_HFOV_DEG)
    ap.add_argument("--cut-thresh", type=float, default=9.0)
    ap.add_argument("--min-duration", type=float, default=30)
    ap.add_argument("--max-duration", type=float, default=1500)
    ap.add_argument("--max-frames-per-video", type=int, default=3000)  # 5 min @10Hz
    args = ap.parse_args()

    work = Path(args.work)
    clips_dir = work / "clips"; clips_dir.mkdir(parents=True, exist_ok=True)
    dl_dir = work / "dl"; dl_dir.mkdir(parents=True, exist_ok=True)
    ptr_path = work / "pointers.jsonl"
    state_path = work / "harvest_state.json"
    manifest_path = work / "manifest.json"

    state = json.loads(state_path.read_text()) if state_path.exists() else \
        {"done_videos": [], "n_clips": 0}
    done = set(state["done_videos"])
    clip_id = state["n_clips"]

    anon = C.Anonymizer()
    log(f"anonymizer ready: face={len(anon.face)} plate={len(anon.plate)} "
        f"body={len(anon.body)} cascades")

    seeds = [as_watch_url(s) for s in read_lines(args.seed_file)]
    queries = read_lines(args.queries_file)
    discovered = discover(queries, args.per_query, args.max_videos * 3) if queries else []
    candidates = seeds + [as_watch_url(v) for v in discovered]
    log(f"candidates: {len(seeds)} seed + {len(discovered)} discovered "
        f"= {len(candidates)}")

    meta_opts = {"quiet": True, "no_warnings": True, "skip_download": True,
                 "noplaylist": True}
    dl_opts = {"quiet": True, "no_warnings": True, "noplaylist": True,
               "noprogress": True,
               "format": "bv*[height<=480]/b[height<=480]/bv*/b",
               "max_filesize": 250 * 1024 * 1024,
               "outtmpl": str(dl_dir / "%(id)s.%(ext)s")}

    rejects = {"not_cc": 0, "bad_title": 0, "duration": 0, "dl_fail": 0,
               "decode_fail": 0, "no_license_field": 0}
    accepted_videos = 0
    n_videos_tried = 0

    for url in candidates:
        if clip_id >= args.max_clips or accepted_videos >= args.max_videos:
            break
        try:
            with yt_dlp.YoutubeDL(meta_opts) as ydl:
                info = ydl.extract_info(url, download=False)
        except Exception as e:
            log(f"  meta fail {url}: {type(e).__name__}: {e}"); continue
        vid = info.get("id")
        if vid in done:
            continue
        n_videos_tried += 1
        # ---- GATE 1: license MUST be Creative Commons ----
        if info.get("license") is None:
            rejects["no_license_field"] += 1
        if not C.is_creative_commons(info):
            rejects["not_cc"] += 1
            log(f"  REJECT non-CC [{vid}] license={info.get('license')!r}")
            done.add(vid); continue
        # ---- GATE 2: not time-manipulated ----
        if bad_title(info):
            rejects["bad_title"] += 1
            log(f"  REJECT time-manipulated [{vid}] {info.get('title')[:50]!r}")
            done.add(vid); continue
        # ---- GATE 3: duration ----
        dur = info.get("duration") or 0
        if dur < args.min_duration or dur > args.max_duration:
            rejects["duration"] += 1
            done.add(vid); continue

        # ---- download (bounded) ----
        try:
            with yt_dlp.YoutubeDL(dl_opts) as ydl:
                ydl.download([url])
        except Exception as e:
            rejects["dl_fail"] += 1
            log(f"  dl fail [{vid}]: {type(e).__name__}: {e}")
            done.add(vid); continue
        mp4s = list(dl_dir.glob(f"{vid}.*"))
        if not mp4s:
            rejects["dl_fail"] += 1; done.add(vid); continue
        mp4 = str(mp4s[0])

        # ---- decode + anonymize + canonical crop ----
        try:
            anon.reset()
            vid_u8, meta = C.decode_canonical(
                mp4, anon, hfov_deg=args.hfov_deg,
                max_frames=args.max_frames_per_video)
        except Exception as e:
            rejects["decode_fail"] += 1
            log(f"  decode fail [{vid}]: {type(e).__name__}: {e}")
            os.remove(mp4); done.add(vid); continue
        finally:
            for m in dl_dir.glob(f"{vid}.*"):        # DELETE source video always
                try: os.remove(m)
                except OSError: pass

        # ---- segment -> clips ----
        T = vid_u8.shape[0]
        cf = args.clip_frames
        made = 0
        with open(ptr_path, "a", encoding="utf-8") as pf:
            for start in range(0, T - cf + 1, cf):
                if clip_id >= args.max_clips or made >= args.per_video_clips:
                    break
                seg = vid_u8[start:start + cf]                 # [cf,3,256,256]
                cut = C.shotcut_score(seg)
                if cut > args.cut_thresh:
                    continue                                  # drop spliced clip
                stacked = C.stack_frames(seg, C.N_STACK)       # [cf-2,9,256,256]
                n = stacked.shape[0]
                clip_path = clips_dir / f"clip_{clip_id:05d}.pt"
                torch.save({"frames_u8": stacked,
                            "poses": torch.zeros(n, 4),
                            "actions": torch.zeros(n, 2),
                            "video_id": vid, "clip_id": clip_id}, clip_path)
                ptr = C.clip_pointer(info, clip_id, start, n, C.TARGET_HZ, meta,
                                     extra={"shotcut_score": round(cut, 2),
                                            "clip_path": str(clip_path)})
                pf.write(json.dumps(ptr) + "\n")
                clip_id += 1; made += 1
        accepted_videos += 1 if made else 0
        done.add(vid)
        state["done_videos"] = sorted(done); state["n_clips"] = clip_id
        state_path.write_text(json.dumps(state))
        log(f"  [{vid}] {info.get('title','')[:45]!r} dur={dur}s -> {made} clips "
            f"(total {clip_id}) anon f/p/b={meta['anon']['faces']}/"
            f"{meta['anon']['plates']}/{meta['anon']['bodies']}")

    manifest = {
        "experiment": "youtube_idm_pilot_harvest",
        "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_clips": clip_id, "accepted_videos": accepted_videos,
        "videos_tried": n_videos_tried, "rejects": rejects,
        "clip_frames": args.clip_frames, "hfov_assumed_deg": args.hfov_deg,
        "cut_thresh": args.cut_thresh,
        "privacy": "faces+plates+bodies Haar-blurred at full-res before 256 "
                   "downscale; no raw video / full-res frames persisted; clip "
                   "frames are transient (deleted after encode).",
        "license_gate": C.CC_LICENSE,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    log(f"HARVEST DONE: {clip_id} clips from {accepted_videos} videos; "
        f"rejects={rejects}")
    log(f"WROTE {manifest_path}")
    log("YT_HARVEST_DONE")


if __name__ == "__main__":
    main()
