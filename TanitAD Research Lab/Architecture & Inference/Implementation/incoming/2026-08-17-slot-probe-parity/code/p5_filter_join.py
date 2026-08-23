"""P5 — the SAME join, restricted to the declared clips. No re-derivation.

sp1 loads every join record into a dict; the full 433,040-record file would be
gigabytes of Python objects for 130 clips' worth of use. This writes the SAME
LINES, byte-for-byte, for the selected clips only — it parses `clip_id` and
copies; it never re-computes a coordinate. Line count and box count are
cross-checked against the census so a silent under-copy is impossible.
"""
import json, lzma, sys
from collections import defaultdict
from pathlib import Path

sel = json.loads(Path(sys.argv[1]).read_text("utf-8"))
cen = {d["clip_id"]: d for d in json.loads(Path(sys.argv[2]).read_text("utf-8"))["per_clip"]}
src, dst = Path(sys.argv[3]), Path(sys.argv[4])
keep = set(sel["eval_clips"]) | set(sel["train_clips"])
n_lines = defaultdict(int); n_boxes = defaultdict(int); wrote = 0
with lzma.open(str(src), "rt", encoding="utf-8") as f, \
     open(dst, "w", encoding="utf-8", newline="\n") as g:
    for line in f:
        cid = line.split('"clip_id":', 1)[1].split('"', 2)[1]
        if cid in keep:
            g.write(line if line.endswith("\n") else line + "\n")
            wrote += 1; n_lines[cid] += 1
            n_boxes[cid] += line.count('"track_id"')
bad = [c for c in keep if n_lines[c] != cen[c]["n_labelled_frames"]]
box_bad = [c for c in keep if n_boxes[c] != cen[c]["n_boxes"]]
rec = {"_evidence_class": "MEASURED (ours; byte-copy of the md5-verified join)",
       "src": str(src), "dst": str(dst), "n_clips": len(keep), "n_lines": wrote,
       "n_boxes": int(sum(n_boxes.values())),
       "expected_lines": int(sum(cen[c]["n_labelled_frames"] for c in keep)),
       "expected_boxes": int(sum(cen[c]["n_boxes"] for c in keep)),
       "clips_with_line_mismatch": bad, "clips_with_box_mismatch": box_bad,
       "missing_clips": sorted(c for c in keep if c not in n_lines)}
rec["ok"] = (not bad and not box_bad and not rec["missing_clips"]
             and wrote == rec["expected_lines"])
Path(str(dst) + ".meta.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
print(json.dumps(rec, indent=1))
if not rec["ok"]:
    raise SystemExit("[p5] REFUSING: filtered join does not reproduce the census counts")
