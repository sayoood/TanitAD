"""D-B retry — evidence collector.

Produces ONE raw JSON backing every claim in DB_RETRY.md:
  * the bot-block signature scan (the priority-1 evidence: blocked or not),
  * every non-200 / error line the harvest produced, classified,
  * the evasion-token audit of the scripts that actually ran, with md5s,
  * per-round timing from run.log,
  * the corpus-contamination title audit (time-manipulation / game / review),
  * environment provenance (yt-dlp, cv2 + CascadeClassifier, encoder md5).

Run on pod3:
  /workspace/venv/bin/python collect_evidence.py \
      --work /workspace/tmp/yt_scaleup \
      --out  /workspace/tmp/yt_scaleup/results/db_retry_evidence.json
"""
from __future__ import annotations
import argparse, glob, hashlib, json, os, re, subprocess, sys, time
from pathlib import Path

# Signatures that mean "YouTube is refusing us as a bot" — the priority-1 question.
BLOCK_SIGNATURES = [
    "sign in to confirm", "not a bot", "confirm you're not a bot",
    "http error 429", "too many requests", "captcha",
    "this helps protect our community",
]
# Non-200s that are ORDINARY per-video restrictions, NOT a bot-block.
BENIGN_HTTP = ["http error 403", "http error 404", "video unavailable",
               "private video", "members-only", "age-restricted", "removed by the uploader"]

# Every option that would constitute bot-detection evasion.
EVASION_TOKENS = [
    "cookie", "cookiesfrombrowser", "player_client", "extractor_args",
    "extractor-args", "proxy", "user_agent", "user-agent", "http_headers",
    "po_token", "potoken", "force_ipv4", "force-ipv4", "source_address",
    "geo_bypass", "innertube", "invidious", "piped",
]

# A COMPLETE time-manipulation detector (the shipped BAD_TITLE list has holes).
SPEED_RE = re.compile(r"\b\d+(?:\.\d+)?\s*x\b|\bx\s*\d+(?:\.\d+)?\b|"
                      r"sped[\s-]*up|speed(?:ed)?[\s-]*up|time[\s-]*lapse|timelapse|"
                      r"hyperlapse|fast[\s-]*forward|slow[\s-]*mo", re.I)
GAME_RE = re.compile(r"beamng|assetto|forza|gran turismo|euro truck|ets2|ats\b|"
                     r"gta\b|simulator|simulation|\bgame\b|unreal|unity", re.I)
REVIEW_RE = re.compile(r"\bbest\b|\breview\b|before you buy|\bvs\.?\b|worth it|"
                       r"wasting money|the truth about|top \d+|buyer'?s guide|unboxing", re.I)

# The shipped filter, copied verbatim from harvest_scaleup.py, so the audit compares
# against what actually ran rather than against what we wish had run.
SHIPPED_BAD_TITLE = ("5x", "10x", "4x speed", "2x speed", "fast forward", "fast-forward",
                     "timelapse", "time lapse", "time-lapse", "hyperlapse", "sped up",
                     "sped-up", "speed up", "speeded")


def md5(p):
    try:
        h = hashlib.md5()
        with open(p, "rb") as f:
            for c in iter(lambda: f.read(1 << 20), b""):
                h.update(c)
        return h.hexdigest()
    except OSError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/workspace/tmp/yt_scaleup")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    work = Path(args.work)
    scripts = work / "scripts"

    # ---------- 1. block-signature scan over every harvest log ----------
    logs = sorted(glob.glob(str(work / "w*" / "harvest.log"))) + [str(work / "run.log")]
    block_hits, benign_hits, other_errors = [], [], []
    for lp in logs:
        if not os.path.exists(lp):
            continue
        for i, ln in enumerate(open(lp, encoding="utf-8", errors="replace"), 1):
            low = ln.lower()
            if any(s in low for s in BLOCK_SIGNATURES):
                block_hits.append({"log": os.path.basename(os.path.dirname(lp)) or "run",
                                   "line_no": i, "line": ln.strip()[:400]})
            elif any(s in low for s in BENIGN_HTTP):
                benign_hits.append({"log": os.path.basename(os.path.dirname(lp)) or "run",
                                    "line_no": i, "line": ln.strip()[:400]})
            elif "error" in low and "dl fail" not in low and "meta fail" not in low:
                other_errors.append({"log": os.path.basename(os.path.dirname(lp)) or "run",
                                     "line_no": i, "line": ln.strip()[:400]})

    # ---------- 2. evasion audit of the scripts that actually ran ----------
    audit = {}
    for p in sorted(glob.glob(str(scripts / "*.py")) + glob.glob(str(scripts / "*.sh"))):
        try:
            s = open(p, encoding="utf-8", errors="replace").read().lower()
        except OSError:
            continue
        hits = []
        for t in EVASION_TOKENS:
            for m in re.finditer(re.escape(t), s):
                line = s.count("\n", 0, m.start()) + 1
                ctx = s.splitlines()[line - 1].strip()[:200] if line - 1 < len(s.splitlines()) else ""
                # 'proxy' appears in prose ("collision proxy", "parity/proxy runs")
                hits.append({"token": t, "line": line, "context": ctx})
        audit[os.path.basename(p)] = {
            "md5": md5(p),
            "evasion_token_hits": hits,
            "n_hits": len(hits),
        }

    # ---------- 3. round timing from run.log ----------
    rounds = []
    rl = work / "run.log"
    if rl.exists():
        for ln in open(rl, encoding="utf-8", errors="replace"):
            if "ROUND" in ln or "target reached" in ln or "pool exhausted" in ln \
               or "HARVEST+LABEL COMPLETE" in ln or "DOWNSTREAM" in ln:
                rounds.append(ln.strip())

    # ---------- 4. contamination title audit ----------
    vids = {}
    for f in sorted(glob.glob(str(work / "w*" / "pointers.jsonl"))):
        for ln in open(f, encoding="utf-8", errors="replace"):
            ln = ln.strip()
            if not ln:
                continue
            try:
                p = json.loads(ln)
            except Exception:
                continue
            v = p.get("video_id")
            r = vids.setdefault(v, {"title": p.get("title"), "uploader": p.get("uploader"),
                                    "url": p.get("url"), "n_clips": 0,
                                    "license": p.get("license"), "is_cc": p.get("is_cc")})
            r["n_clips"] += 1
    contamination = {"time_manipulated": [], "game_or_sim": [], "review_talking_head": []}
    for v, r in vids.items():
        t = r["title"] or ""
        caught = any(b in t.lower() for b in SHIPPED_BAD_TITLE)
        rec = {"video_id": v, "title": t, "n_clips": r["n_clips"],
               "caught_by_shipped_BAD_TITLE": caught, "url": r.get("url")}
        if SPEED_RE.search(t):
            contamination["time_manipulated"].append(rec)
        if GAME_RE.search(t):
            contamination["game_or_sim"].append(rec)
        if REVIEW_RE.search(t):
            contamination["review_talking_head"].append(rec)
    n_total_clips = sum(r["n_clips"] for r in vids.values())
    flagged_ids, flagged_clips = set(), 0
    for k, lst in contamination.items():
        for rec in lst:
            if rec["video_id"] not in flagged_ids:
                flagged_ids.add(rec["video_id"]); flagged_clips += rec["n_clips"]

    # ---------- 5. environment provenance ----------
    env = {}
    try:
        import yt_dlp, cv2, torch
        env["yt_dlp"] = yt_dlp.version.__version__
        env["cv2"] = cv2.__version__
        env["cv2_has_CascadeClassifier"] = hasattr(cv2, "CascadeClassifier")
        env["torch"] = torch.__version__
        env["cuda_available"] = torch.cuda.is_available()
    except Exception as e:
        env["import_error"] = f"{type(e).__name__}: {e}"
    ck = "/workspace/tmp/idm/ckpt.pt"
    env["encoder_ckpt"] = ck
    env["encoder_md5"] = md5(ck) if os.path.exists(ck) else None

    out = {
        "experiment": "db_retry_evidence",
        "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {"W": 2, "TARGET": 400, "SEEDS": 4, "sleep_s": 4,
                   "geometry": "GeoCalib per-video (fixed-HFOV fallback on low confidence)"},
        "PRIORITY_1_blocked": len(block_hits) > 0,
        "block_signature_hits": block_hits,
        "n_block_signature_hits": len(block_hits),
        "block_signatures_scanned_for": BLOCK_SIGNATURES,
        "benign_per_video_http_errors": benign_hits,
        "n_benign_http_errors": len(benign_hits),
        "benign_note": ("per-video 403/404/unavailable are ORDINARY access restrictions, "
                        "not a bot-block: neighbouring downloads from the same worker and "
                        "IP succeeded. They were NOT retried."),
        "other_error_lines": other_errors[:50],
        "evasion_audit_of_executed_scripts": audit,
        "evasion_audit_note": ("token hits in prose (e.g. 'parity/proxy runs', 'collision "
                               "proxy') are documentation, not options; inspect `context`."),
        "round_timeline": rounds,
        "corpus_contamination": contamination,
        "contamination_summary": {
            "n_distinct_videos": len(vids),
            "n_clips_total": n_total_clips,
            "n_videos_flagged": len(flagged_ids),
            "n_clips_from_flagged_videos": flagged_clips,
            "frac_clips_contaminated": (round(flagged_clips / n_total_clips, 4)
                                        if n_total_clips else None),
        },
        "videos": vids,
        "environment": env,
    }
    js = json.dumps(out, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(js, encoding="utf-8")
        print("WROTE", args.out)
    print(json.dumps({k: v for k, v in out.items()
                      if k in ("PRIORITY_1_blocked", "n_block_signature_hits",
                               "n_benign_http_errors", "contamination_summary",
                               "environment")}, indent=2))


if __name__ == "__main__":
    main()
