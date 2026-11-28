"""Test whether yt-dlp can extract from YouTube on this pod's IP (datacenter IPs
are frequently bot-blocked with 'Sign in to confirm you're not a bot'), and
whether the Creative-Commons license filter surfaces forward-dashcam candidates.
Metadata only (download=False) — no video bytes fetched. Prints a JSON report.
"""
from __future__ import annotations
import json, sys

report = {"extract_single": None, "cc_search": None, "errors": []}

try:
    import yt_dlp
    report["yt_dlp_version"] = yt_dlp.version.__version__
except Exception as e:
    print(json.dumps({"fatal": f"import yt_dlp failed: {e}"}))
    sys.exit(1)

CC = "Creative Commons Attribution license (reuse allowed)"

# ---- 1) single known-stable video: proves extraction works at all ----
opts = {"quiet": True, "no_warnings": True, "skip_download": True,
        "noplaylist": True}
try:
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info("https://www.youtube.com/watch?v=jNQXAC9IVRw",
                                download=False)
    report["extract_single"] = {
        "id": info.get("id"), "license": info.get("license"),
        "title": (info.get("title") or "")[:60],
        "duration_s": info.get("duration")}
except Exception as e:
    report["errors"].append(f"single_extract: {type(e).__name__}: {e}")

# ---- 2) CC-filtered search for forward dashcam driving ----
# sp=EgIwAQ%3D%3D is YouTube's 'Creative Commons' search filter param.
search_url = ("https://www.youtube.com/results?search_query="
              "forward+dashcam+highway+driving&sp=EgIwAQ%3D%3D")
try:
    sopts = {"quiet": True, "no_warnings": True, "skip_download": True,
             "extract_flat": "in_playlist", "playlistend": 10}
    with yt_dlp.YoutubeDL(sopts) as ydl:
        pl = ydl.extract_info(search_url, download=False)
    entries = [e for e in (pl.get("entries") or []) if e]
    ids = [e.get("id") for e in entries][:6]
    # per-video full extract to read the license field (flat entries omit it)
    detail = []
    with yt_dlp.YoutubeDL(opts) as ydl:
        for vid in ids:
            try:
                i = ydl.extract_info(
                    f"https://www.youtube.com/watch?v={vid}", download=False)
                detail.append({
                    "id": i.get("id"),
                    "license": i.get("license"),
                    "is_cc": i.get("license") == CC,
                    "duration_s": i.get("duration"),
                    "title": (i.get("title") or "")[:60]})
            except Exception as e:
                detail.append({"id": vid, "error": f"{type(e).__name__}: {e}"})
    report["cc_search"] = {"n_results": len(entries), "detail": detail}
except Exception as e:
    report["errors"].append(f"cc_search: {type(e).__name__}: {e}")

print("YTDLP_TEST_JSON_START")
print(json.dumps(report, indent=2))
print("YTDLP_TEST_JSON_END")
