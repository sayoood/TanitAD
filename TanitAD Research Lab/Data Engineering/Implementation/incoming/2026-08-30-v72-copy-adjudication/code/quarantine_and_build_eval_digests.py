"""Two jobs the Master Mind asked for, both consequences of the same finding.

JOB 1 - DISARM THE `_v72_verify` TRAP.
Adjudicated by content (`adjudicate_v72_copies.py`): same 4,572 / 147 clips,
IDENTICAL payloads, differing ONLY in `schema_version` - the field the fix
changed. They are the pre-fix blobs. Superseded.

⭐ I QUARANTINE RATHER THAN DELETE, AND THE RENAME IS THE POINT. The defect was
`glob.glob("C:/Users/Admin/**/<name>")[0]` - a match on FILENAME. Moving the
files while keeping their names would leave the trap fully armed. Renaming them
breaks the glob, which is the actual fix; keeping the bytes costs 2 MB and keeps
the finding reproducible. If the Master Mind would rather they were gone, the
quarantine directory is one `rm -rf` away and nothing references it.

JOB 2 - BANK THE 147 EVAL CLIP DIGESTS.
`sha256(clip_id.encode("utf-8")).hexdigest()`, one per line, sorted, so the
EvalFlyWheel can make the deployed-val40 overlap a LIVE GATE instead of a
documented caveat. No clip ids leave Thor - a digest is one-way, which is also
why this sidesteps the `confidentiality` field on the val40 digest file.

⚠️ THE GATE IS A DRIFT DETECTOR, NOT A DISCOVERY INSTRUMENT. The answer is 6 BY
CONSTRUCTION - `build_v72.py` asserted it. Its value is the day the eval set
changes and nobody re-checks. The written file says so in its own header, because
a future reader finding "6" must not read it as a fresh measurement.
"""
import hashlib
import json
import shutil
from pathlib import Path

REL = Path("C:/Users/Admin/tanitad-wt/_s2build/release")
QUAR = REL / "_superseded_pre_schema_fix"

# ── JOB 1 ────────────────────────────────────────────────────────────────────
MOVES = [
    (REL / "_v72_verify" / "labels" / "s2_labels_v7.2_train.jsonl.gz",
     "s2_labels_v72_train.PRE-SCHEMA-FIX.DO-NOT-USE.jsonl.gz"),
    (REL / "_v72_verify" / "labels" / "s2_labels_v7.2_eval.jsonl.gz",
     "s2_labels_v72_eval.PRE-SCHEMA-FIX.DO-NOT-USE.jsonl.gz"),
]
QUAR.mkdir(exist_ok=True)
moved = []
for src, newname in MOVES:
    if not src.exists():
        print("  already gone: %s" % src)
        continue
    md5 = hashlib.md5(src.read_bytes()).hexdigest()
    dst = QUAR / newname
    shutil.move(str(src), str(dst))
    assert hashlib.md5(dst.read_bytes()).hexdigest() == md5, "bytes changed in the move"
    moved.append((src.name, dst.name, md5))
    print("  quarantined %s -> %s  (md5 %s, unchanged)" % (src.name, dst.name, md5))

(QUAR / "README.md").write_text(
    "# SUPERSEDED - pre-schema-fix v7.2 blobs. DO NOT USE.\n\n"
    "These are the v7.2 label blobs as first written, when `schema_version` was\n"
    "wrongly set to `s2_labels_v7.2`. That field identifies the FORMAT CONTRACT,\n"
    "not the release, so `tanitad.data.v7_labels` (EXPECTED_SCHEMA `s2-geom-v7`)\n"
    "refused them. The fix restored the contract value and moved the release into\n"
    "its own `release` key.\n\n"
    "ADJUDICATED BY CONTENT, not by belief: same clips (4,572 / 147), byte-identical\n"
    "payloads, differing ONLY in `schema_version` (and these predate `release`\n"
    "entirely, which reads `None`).\n\n"
    "NOTE they PARSE FINE as JSON - the failure was the loader's schema check, not\n"
    "JSON validity. Reading it as a parse failure points at the wrong fix.\n\n"
    "THEY ARE RENAMED ON PURPOSE. `intrain_eval.py` pinned its md5 guard to these\n"
    "copies via `glob.glob('C:/Users/Admin/**/<name>')[0]` - a match on FILENAME -\n"
    "so the guard would have REFUSED the canonical artifact and ACCEPTED these.\n"
    "Moving them while keeping their names would have left that trap armed.\n\n"
    "CANONICAL: `release/v72/` and HF `Sayood/tanitad-v7-training-corpus` (private).\n"
    "  train  md5 0ff902130ce76886b8a925eceed9e3a5\n"
    "  eval   md5 aa12c948f062181c3297265b51526ec5\n",
    encoding="utf-8")

leftovers = sorted(p.name for p in (REL / "_v72_verify").rglob("*") if p.is_file())
print("  _v72_verify now holds: %s" % (leftovers or "nothing"))

# ── JOB 2 ────────────────────────────────────────────────────────────────────
# ⚠️ THE LOCAL NAME IS NOT THE HF NAME. On disk this is `clip_index_eval.json`;
# it was renamed to `clip_index_v7.2_eval.json` on upload. Reaching for the HF
# name here raised FileNotFoundError - the same location-vs-content confusion
# that caused the trap above, caught this time by the file simply not existing.
idx = json.loads((REL / "v72" / "clip_index_eval.json").read_text(encoding="utf-8"))
clips = sorted(idx["clips"])
assert len(clips) == 147, "expected 147 eval clips, got %d" % len(clips)

digests = sorted(hashlib.sha256(c.encode("utf-8")).hexdigest() for c in clips)
assert len(set(digests)) == 147, "digest collision - impossible, but assert it anyway"

# the overlap, recomputed here so the banked file states a CHECKED number
v40 = set(json.loads((REL / "val40_digests.json").read_text(encoding="utf-8"))["clip_id_digests"])
overlap = sorted(set(digests) & v40)
print("\n  eval clips %d | val40 overlap %d (expected 6 BY CONSTRUCTION)" % (len(clips), len(overlap)))
assert len(overlap) == 6, "overlap is %d, not the 6 build_v72.py asserted" % len(overlap)

out = REL / "v72" / "eval_v72_clip_digests.txt"
out.write_text(
    "# sha256(clip_id.encode('utf-8')).hexdigest() for the 147 clips of the v7.2\n"
    "# EVAL split. One per line, sorted. Source: index/clip_index_v7.2_eval.json.\n"
    "#\n"
    "# PURPOSE: make the deployed-val40 overlap a LIVE GATE rather than a caveat in\n"
    "# a manifest. Intersect this with the val40 digests; the answer must be 6.\n"
    "#\n"
    "# THIS IS A DRIFT DETECTOR, NOT A DISCOVERY INSTRUMENT. The 6 is true BY\n"
    "# CONSTRUCTION - build_v72.py asserted it before writing the split. A gate\n"
    "# reading 6 today has confirmed nothing new; its value is the day the eval set\n"
    "# changes and nobody re-checks. Do not quote this as a finding.\n"
    "#\n"
    "# Digests only - no clip id is recoverable, so nothing confidential travels.\n"
    "# n_eval=147  n_val40_overlap=6  built_from_md5=aa12c948f062181c3297265b51526ec5\n"
    + "\n".join(digests) + "\n", encoding="utf-8")

# read it back and re-assert ON THE WRITTEN FILE
back = [ln.strip() for ln in out.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.startswith("#")]
assert back == digests, "written file does not match the intended digests"
assert len(set(back) & v40) == 6, "written file does not reproduce the overlap"
print("  wrote %s  (%d digests, re-asserted on the written file)" % (out.name, len(back)))
print("  sha256(file) = %s" % hashlib.sha256(out.read_bytes()).hexdigest()[:16])
