"""clipid_scan.py -- the clip-id rule, asserted on CONTENT: no raw PhysicalAI clip id, nor an 8+-hex prefix of one, in any file of the hold.

Universe: the clip ids keyed in the refcv8 extrinsics json (train4572 + eval139).  Every text file is READ (unreadable files are COUNTED and fail the scan --
"0 hits" is a claim about the content only when every file was read).  A same-run CONTROL must hit: a planted line carrying one real prefix.
    python clipid_scan.py <dir> [--out <json>]
"""
import argparse, json, re, sys
from pathlib import Path
UNIVERSE = "E:/refcv8_eval_kit/data/refcv8_train4572_eval139_extrinsics.json"
TEXT = {".py", ".sh", ".md", ".json", ".txt", ".log", ".jsonl", ".csv", ".yaml", ".yml"}
TOK = re.compile(r"(?<![0-9a-f])([0-9a-f]{8,})", re.I)

def universe():
    j = json.load(open(UNIVERSE, encoding="utf-8"))
    ids = list(j.keys()) if isinstance(j, dict) else []
    if isinstance(j, dict) and len(ids) < 1000:
        for v in j.values():
            if isinstance(v, dict) and len(v) > 1000:
                ids = list(v.keys()); break
    pre = {i.replace("-", "").lower()[:8] for i in ids if len(i.replace("-", "")) >= 8}
    return ids, pre

def scan_text(txt, pre):
    return [m.group(1) for m in TOK.finditer(txt) if m.group(1).lower()[:8] in pre]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("--out"); a = ap.parse_args()
    ids, pre = universe()
    assert len(pre) > 4000, f"universe too small ({len(pre)}): refusing to report a clean scan"
    ctl = scan_text(f"control {ids[0]} end", pre)
    hits, unread, n = [], [], 0
    for p in sorted(Path(a.root).rglob("*")):
        if not p.is_file() or p.suffix.lower() not in TEXT:
            continue
        try:
            t = p.read_text(encoding="utf-8", errors="replace"); n += 1
        except Exception as e:
            unread.append(f"{p}: {type(e).__name__}"); continue
        for h in scan_text(t, pre):
            hits.append({"file": str(p), "token_prefix8": h[:8]})
    rec = {"root": a.root, "n_universe": len(ids), "n_files_read": n, "n_unreadable": len(unread), "unreadable": unread[:20], "n_hits": len(hits),
           "hits": hits[:50], "control_hit": bool(ctl), "status": "CLEAN" if (not hits and not unread and ctl) else ("HITS" if hits else "INCONCLUSIVE")}
    if a.out:
        Path(a.out).write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(json.dumps({k: rec[k] for k in ("n_universe", "n_files_read", "n_unreadable", "n_hits", "control_hit", "status")}))
    return 0 if rec["status"] == "CLEAN" else 3

if __name__ == "__main__":
    sys.exit(main())
