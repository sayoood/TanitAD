"""Did the OLD ci.py's display defect actually reach a PUBLISHED record?

HEAD's `_render_bounds` fixes a RENDERING defect: `separated` is decided on the
UNROUNDED bounds, but `lo`/`hi`/`delta` were rounded to 4 dp, so a record could
read `{"delta": 0.0, "lo": 0.0, "hi": 0.0, "separated": true}`.

This walks every committed JSON in the repo and counts records that carry the
signature.  It is the blast-radius measurement for the DISPLAY defect — separate
from, and much smaller than, a statistical defect would have been.
"""
import json, sys
from pathlib import Path

ROOT = Path(sys.argv[1])
hits, scanned, files = [], 0, 0


def walk(o, path, fp):
    global scanned
    if isinstance(o, dict):
        if "separated" in o and "lo" in o and "hi" in o:
            scanned += 1
            try:
                lo, hi, sep = float(o["lo"]), float(o["hi"]), bool(o["separated"])
            except Exception:
                return
            contradicts = sep and not (lo > 0 or hi < 0)
            zerowidth = sep and lo == 0.0 and hi == 0.0
            if contradicts or zerowidth:
                hits.append({"file": str(fp.relative_to(ROOT)).replace("\\", "/"),
                             "json_path": path,
                             "record": {k: o.get(k) for k in
                                        ("delta", "lo", "hi", "ci95", "separated",
                                         "p_delta_gt0", "estimator", "n_windows",
                                         "n_episodes", "display_dp", "degenerate")},
                             "printed_interval_contradicts_separated": contradicts,
                             "zero_width_and_separated": zerowidth})
        for k, v in o.items():
            walk(v, f"{path}.{k}", fp)
    elif isinstance(o, list):
        for i, v in enumerate(o):
            walk(v, f"{path}[{i}]", fp)


for fp in ROOT.rglob("*.json"):
    if ".git" in fp.parts:
        continue
    try:
        o = json.loads(fp.read_text(encoding="utf-8"))
    except Exception:
        continue
    files += 1
    walk(o, "$", fp)

out = {
    "what": "published records where the OLD ci.py's 4 dp rendering PRINTS an "
            "interval that contradicts its own `separated` verdict",
    "evidence_class": "MEASURED (ours; scan of every committed JSON at HEAD)",
    "root": str(ROOT),
    "n_json_files_parsed": files,
    "n_interval_records_scanned": scanned,
    "n_records_with_the_signature": len(hits),
    "note": "This is the DISPLAY defect's real blast radius. The estimator is "
            "byte-identical between the two ci.py files (see ci_equivalence.json), "
            "so no VALUE changes; what changes is whether a reader or an "
            "automated gate could be misled by the printed record.",
    "hits": hits[:200],
}
print(json.dumps({k: out[k] for k in out if k != "hits"}, indent=1))
for h in hits[:20]:
    print(json.dumps(h, indent=1))
Path(sys.argv[2]).write_text(json.dumps(out, indent=1), encoding="utf-8")
