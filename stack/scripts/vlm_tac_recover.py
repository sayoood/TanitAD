"""Rebuild the extractor's JSONL from the OPERATOR'S LOG, and merge runs.

⛔ THE GAP THIS CLOSES. ``vlm_tac_extract.py`` resumes from its output file —
but on Colab that file lives on the VM, so when the VM goes the resume state
goes with it and the next session restarts from zero. MEASURED 2026-08-19: four
of five sessions were reclaimed, one of them after ~15 min of generation, and a
run that had banked work locally would have lost none of it.

The frames streamed to stdout are the ONE store that cannot be reclaimed,
because they land on the operator's disk as they are produced. This turns them
back into the extractor's own format, so the loop is:

    launch -> session dies -> recover -> re-upload -> relaunch -> resumes

and the same JSONL is what Thor or a pod would continue from.

⚠️ Every frame carries its own length, so a short or dropped frame is DETECTED
and reported rather than silently decoding to garbage. Merging is by
(clip_id, kind), last-writer-wins, so re-running a clip after a fix supersedes
the older record instead of duplicating it.

Usage:
    python stack/scripts/vlm_tac_recover.py --out raw.jsonl <log> [<log> ...]
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import sys
from pathlib import Path

FRAME = re.compile(r"@@G(\d+)@([A-Za-z0-9+/=\s]+?)@@")


def frames_from(text: str) -> tuple[list[dict], int]:
    """-> (records, n_short). Whitespace inside a frame is stripped: a log
    viewer or PTY may wrap a long line, which is harmless, whereas a LENGTH
    mismatch means real loss and is counted."""
    recs, short = [], 0
    for declared, blob in FRAME.findall(text):
        blob = re.sub(r"\s+", "", blob)
        if len(blob) != int(declared):
            short += 1
            continue
        try:
            recs.append(json.loads(base64.b64decode(blob)))
        except (ValueError, json.JSONDecodeError):
            short += 1
    return recs, short


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--merge", action="store_true", default=True,
                    help="fold in any records already in --out (default)")
    args = ap.parse_args(argv)

    merged: dict[tuple[str, str], dict] = {}
    out = Path(args.out)
    if args.merge and out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "clip_id" in r and "kind" in r:
                    merged[(r["clip_id"], r["kind"])] = r
        print(f"existing {out.name}: {len(merged)} records")

    total_short = 0
    for lg in args.logs:
        p = Path(lg)
        if not p.exists():
            print(f"  {lg}: MISSING")
            continue
        # logs may carry NULs and a BOM from PowerShell redirection
        text = p.read_bytes().replace(b"\x00", b"").decode("utf-8", "replace")
        recs, short = frames_from(text)
        total_short += short
        new = sum(1 for r in recs if (r.get("clip_id"), r.get("kind")) not in merged)
        for r in recs:
            if "clip_id" in r and "kind" in r:
                merged[(r["clip_id"], r["kind"])] = r
        print(f"  {p.name}: {len(recs)} frames ({new} new)"
              + (f"  ⚠️ {short} SHORT/CORRUPT" if short else ""))

    with out.open("w", encoding="utf-8") as f:
        for (cid, kind), r in sorted(merged.items()):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    ok = sum(1 for r in merged.values() if "raw" in r)
    err = sum(1 for r in merged.values() if "error" in r)
    cap = sum(1 for r in merged.values() if r.get("hit_cap"))
    print(f"-> {out}  {len(merged)} records (ok {ok}, error {err}, cap-hits {cap})")
    if total_short:
        print(f"⚠️ {total_short} frame(s) were short or unparseable and were "
              f"DROPPED — those (clip, kind) pairs will simply be re-run on "
              f"the next resume, which is the safe outcome.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
