"""make_a6cache_thor.py -- build the Thor-side TRAIN139 ("A6") episode directory as 139 SYMLINKS, and verify it against the dev-box copy.

STDLIB ONLY (it runs on Thor before the roll; no torch, no GPU). The raw clip ids never leave Thor and never reach a banked file: the
selection is `sha12(clip_id) in the banked TRAIN139_WINDOWS.json`, the ids come from Thor's own pool list, and the only things printed are COUNTS.

  python3 make_a6cache_thor.py --windows TRAIN139_WINDOWS.json --sizes a6_sizes_md5_sha12.json --out /home/nvidia/r8ns/l7cap/a6cache

Checks (each must hold or the script exits 3 and prints ZZL7CA6BADZZ):
  * exactly 139 clips resolved, each file present under SRC (default /home/nvidia/data/refcv6-b1-416x1024-train -- the cache the dev-box A6 files came from)
  * every file's byte size == the dev-box A6 file's size (139/139)
  * the 3 md5 probes (first / middle / last sha12) equal the dev-box md5s
"""
import argparse
import hashlib
import json
import os
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", required=True)
    ap.add_argument("--sizes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pool", default="/home/nvidia/data/refcv6_train_clips.txt")
    ap.add_argument("--src", default="/home/nvidia/data/refcv6-b1-416x1024-train")
    ap.add_argument("--link", choices=("symlink", "none"), default="symlink", help="'none' = verify only (used by the dev-box test, which has no symlink privilege)")
    a = ap.parse_args()
    want = {k[len("clip-sha12-"):] for k in json.load(open(a.windows))["clips"]}
    sz = json.load(open(a.sizes))
    ids = {}
    for ln in open(a.pool):
        ln = ln.strip()
        if ln:
            ids[hashlib.sha256(ln.encode()).hexdigest()[:12]] = ln
    miss = sorted(want - set(ids))
    os.makedirs(a.out, exist_ok=True)
    n_ok = n_size = 0
    for k in sorted(want & set(ids)):
        src = os.path.join(a.src, ids[k] + ".v2ep.pt")
        dst = os.path.join(a.out, ids[k] + ".v2ep.pt")
        if os.path.exists(src):
            n_ok += 1
            n_size += int(os.stat(src).st_size == int(sz["sizes_by_sha12"][k]))
            if a.link == "symlink":
                if os.path.lexists(dst):
                    os.remove(dst)
                os.symlink(src, dst)
    md5_bad = 0
    for k, want_md5 in sz["md5_probes_by_sha12"].items():
        h = hashlib.md5()
        with open(os.path.join(a.src, ids[k] + ".v2ep.pt"), "rb") as fh:
            for b in iter(lambda: fh.read(1 << 22), b""):
                h.update(b)
        md5_bad += int(h.hexdigest() != want_md5)
    n_links = sum(1 for f in os.listdir(a.out) if f.endswith(".v2ep.pt")) if a.link == "symlink" else 139
    ok = (len(want) == 139 and not miss and n_ok == 139 and n_size == 139 and md5_bad == 0 and n_links == 139)
    print("%s ids_missing=%d files_present=%d size_equal=%d md5_probe_bad=%d links=%d" % (
        "ZZL7CA6OKZZ" if ok else "ZZL7CA6BADZZ", len(miss), n_ok, n_size, md5_bad, n_links), flush=True)
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())
