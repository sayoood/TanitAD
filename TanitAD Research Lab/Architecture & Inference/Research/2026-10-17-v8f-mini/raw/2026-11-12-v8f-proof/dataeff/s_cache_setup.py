"""s_cache_setup.py -- build the restricted cache root of the S-1/4 arm (DATAEFF_DEFS.md R9).  stdlib only (runs on the pod).

  python s_cache_setup.py --src-cache <full cache root> --list lists/S_quarter.txt --dst-cache <new root>

Writes <dst>/train/manifest.jsonl (the listed clips only, in the source manifest's file order) and a link <dst>/train/clips -> <src>/train/clips.
Refuses when a listed clip is not in the source manifest or its clip file is absent.  Reads no clip content.
"""
import argparse
import json
import os
import sys


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-cache", required=True)
    ap.add_argument("--list", required=True)
    ap.add_argument("--dst-cache", required=True)
    a = ap.parse_args(argv)
    with open(a.list, encoding="utf-8") as f:
        want = [ln.strip() for ln in f if ln.strip()]
    ws = set(want)
    src_man = os.path.join(a.src_cache, "train", "manifest.jsonl")
    keep, seen = [], set()
    with open(src_man, encoding="utf-8") as f:
        for ln in f:
            if ln.strip():
                c = json.loads(ln)["clip"]
                if c in ws:
                    keep.append(ln.rstrip("\n"))
                    seen.add(c)
    miss = ws - seen
    if miss:
        raise SystemExit(f"REFUSED: {len(miss)} listed clips are not in {src_man}, e.g. {sorted(miss)[:3]}")
    clips_dir = os.path.join(a.src_cache, "train", "clips")
    absent = [c for c in want if not os.path.exists(os.path.join(clips_dir, c + ".pt"))]
    if absent:
        raise SystemExit(f"REFUSED: {len(absent)} clip files absent under {clips_dir}, e.g. {absent[:3]}")
    dst = os.path.join(a.dst_cache, "train")
    os.makedirs(dst, exist_ok=True)
    with open(os.path.join(dst, "manifest.jsonl"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(keep) + "\n")
    link = os.path.join(dst, "clips")
    if not os.path.lexists(link):
        os.symlink(os.path.abspath(clips_dir), link, target_is_directory=True)
    print(f"s_cache_setup: {len(keep)} clips -> {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
