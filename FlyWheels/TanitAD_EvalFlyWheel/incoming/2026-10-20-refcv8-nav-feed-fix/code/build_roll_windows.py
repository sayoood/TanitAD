"""build_roll_windows.py -- turn the sealed, clip-id-free window selection into the {clip_id: [ws,..]} json the roll reads.  RUN-TIME work file.

The sealed selection (raw/windows_navfeed.json) is keyed by episode_index + sha12 (the clip-id rule: no raw clip id, no clip-id prefix may sit in
a deliverable).  The roll needs the clip ids, so this script resolves them from the ARCHIVED battery dump's manifest at launch time, VERIFIES
sha256(clip_id)[:12] == the sealed sha12 for every episode (a mismatch refuses), and writes the result ONLY under the run's out dir (never under
the hold).  It prints counts only -- never an id.

    python build_roll_windows.py --windows <hold>/raw/windows_navfeed.json --archive-dump <archive>/dump_s0 --out <out>/windows_roll.json \
                                 --smoke-out <out>/windows_smoke.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

SMOKE_PER_KIND = 3


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", required=True)
    ap.add_argument("--archive-dump", default="C:/Users/Admin/refcv8_eval_archive/battery/step40000ema_unc/dump_s0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke-out", required=True)
    a = ap.parse_args(argv)
    win = json.load(open(a.windows, encoding="utf-8"))
    man = json.load(open(Path(a.archive_dump) / "manifest.json", encoding="utf-8"))
    cid = {int(e["episode_index"]): str(e["clip_id"]) for e in man["episodes"]}
    roll, smoke_ep, n = {}, None, 0
    for e in win["episodes"]:
        ei = int(e["episode_index"])
        c = cid.get(ei)
        if c is None or hashlib.sha256(c.encode()).hexdigest()[:12] != e["sha12"]:
            raise SystemExit(f"[build_roll_windows] episode_index {ei}: the archive manifest does not resolve to the sealed sha12")
        ws = sorted(set(e["mismatch_ws"]) | set(e["control_ws"]))
        if len(ws) != len(e["mismatch_ws"]) + len(e["control_ws"]):
            raise SystemExit(f"[build_roll_windows] episode_index {ei}: a window is in both sets")
        roll[c] = ws
        n += len(ws)
        if smoke_ep is None and len(e["mismatch_ws"]) >= SMOKE_PER_KIND and len(e["control_ws"]) >= SMOKE_PER_KIND:
            smoke_ep = (c, ei, sorted(e["mismatch_ws"])[:SMOKE_PER_KIND], sorted(e["control_ws"])[:SMOKE_PER_KIND])
    want = win["counts"]["n_mismatch"] + win["counts"]["n_control"]
    if n != want:
        raise SystemExit(f"[build_roll_windows] {n} windows resolved, the sealed selection holds {want}")
    if smoke_ep is None:
        raise SystemExit("[build_roll_windows] no episode with 3 mismatch and 3 control windows for the smoke")
    json.dump(roll, open(a.out, "w", encoding="utf-8"))
    json.dump({smoke_ep[0]: sorted(smoke_ep[2] + smoke_ep[3])}, open(a.smoke_out, "w", encoding="utf-8"))
    print(json.dumps({"n_windows": n, "n_episodes": len(roll), "smoke_episode_index": smoke_ep[1],
                      "smoke_mismatch_ws": smoke_ep[2], "smoke_control_ws": smoke_ep[3]}))


if __name__ == "__main__":
    sys.exit(main())
