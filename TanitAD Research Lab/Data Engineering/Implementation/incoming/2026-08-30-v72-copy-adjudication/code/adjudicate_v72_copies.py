"""Which of the two v7.2 copies is which — settled by CONTENT, not by belief.

The Master Mind found two copies of each v7.2 name with DIFFERENT md5s, and
`intrain_eval.py` had pinned its hash guard to the `_v72_verify` pair — so the
guard would have refused the canonical artifact and accepted the other one.

Their hypothesis is that `_v72_verify/` is the PRE-schema-fix download (the blob
I called unloadable). That is checkable rather than arguable: the fix changed
exactly one field, `schema_version`, from "s2_labels_v7.2" back to the contract
value "s2-geom-v7". So:

  * if `_v72_verify` carries schema_version "s2_labels_v7.2"  -> stale debris,
    superseded, safe to delete;
  * if it carries "s2-geom-v7" and STILL differs               -> a real content
    divergence between two same-named releases, and nothing may be deleted until
    the difference is explained.

⛔ It also reports the RECORD-LEVEL diff, because equal record counts with equal
clip ids and a byte difference would mean the payloads drifted — a third reading
neither of us proposed, and the only one that would be alarming.
"""
import gzip
import hashlib
import json
from pathlib import Path

REL = Path("C:/Users/Admin/tanitad-wt/_s2build/release")
PAIRS = {
    "train": (REL / "_v72_verify" / "labels" / "s2_labels_v7.2_train.jsonl.gz",
              REL / "v72" / "s2_labels_v7.2_train.jsonl.gz"),
    "eval": (REL / "_v72_verify" / "labels" / "s2_labels_v7.2_eval.jsonl.gz",
             REL / "v72" / "s2_labels_v7.2_eval.jsonl.gz"),
}


def load(p):
    """Read a copy. A copy that cannot be parsed is a FINDING, not a crash."""
    b = p.read_bytes()
    info = {"path": str(p), "bytes": len(b), "md5": hashlib.md5(b).hexdigest()}
    try:
        rows = [json.loads(x) for x in gzip.open(p, "rt", encoding="utf-8") if x.strip()]
    except Exception as e:                                          # noqa: BLE001
        info["parse"] = "FAILED: %s: %s" % (type(e).__name__, e)
        return info, None
    info["parse"] = "ok"
    info["n"] = len(rows)
    info["schema_version"] = sorted({r.get("schema_version") for r in rows})
    info["release"] = sorted({r.get("release") for r in rows})
    info["split"] = sorted({r.get("split") for r in rows})
    return info, rows


verdicts = {}
for side, (verify_p, canon_p) in PAIRS.items():
    print("=" * 74)
    print(side.upper())
    print("=" * 74)
    vi, vr = load(verify_p)
    ci, cr = load(canon_p)
    for tag, i in (("_v72_verify", vi), ("v72 (canonical)", ci)):
        print("  %-18s %10d B  md5 %s" % (tag, i["bytes"], i["md5"]))
        print("  %-18s parse=%s  n=%s  schema_version=%s  release=%s  split=%s"
              % ("", i["parse"], i.get("n"), i.get("schema_version"),
                 i.get("release"), i.get("split")))

    if vr is None:
        verdicts[side] = "UNPARSEABLE -> superseded debris"
        print("\n  VERDICT: the _v72_verify copy does not parse at all. It cannot be a")
        print("           valid release under any reading. Superseded debris.")
        continue

    v_ids = {r["clip_id"] for r in vr}
    c_ids = {r["clip_id"] for r in cr}
    print("\n  clip ids: identical=%s  only_in_verify=%d  only_in_canonical=%d"
          % (v_ids == c_ids, len(v_ids - c_ids), len(c_ids - v_ids)))

    # the decisive field: what did the schema fix change?
    v_sv, c_sv = set(vi["schema_version"]), set(ci["schema_version"])
    if v_sv != c_sv:
        print("  schema_version DIFFERS: verify=%s  canonical=%s" % (sorted(v_sv), sorted(c_sv)))

    # ⛔ the alarming reading: same ids, same schema, but payloads drifted
    v_by = {r["clip_id"]: r for r in vr}
    payload_diff = []
    for cid in sorted(v_ids & c_ids):
        a = dict(v_by[cid]); b = dict({r["clip_id"]: r for r in cr}[cid])
        for k in ("schema_version", "release"):
            a.pop(k, None); b.pop(k, None)
        if json.dumps(a, sort_keys=True) != json.dumps(b, sort_keys=True):
            payload_diff.append(cid)

    if v_ids == c_ids and not payload_diff and v_sv != c_sv:
        verdicts[side] = "SUPERSEDED (schema_version only)"
        print("\n  VERDICT: same clips, IDENTICAL payloads, differing only in the field")
        print("           the fix changed. -> the _v72_verify copy is the PRE-FIX blob.")
        print("           Stale debris. Safe to delete.")
    elif payload_diff:
        verdicts[side] = "⛔ PAYLOAD DIVERGENCE -- DELETE NOTHING"
        print("\n  ⛔ VERDICT: %d records differ in fields the fix did NOT touch."
              % len(payload_diff))
        print("     e.g. %s" % [c[:8] for c in payload_diff[:5]])
        print("     This is NOT stale debris. Delete nothing; explain the difference.")
    else:
        verdicts[side] = "⛔ CLIP-SET DIVERGENCE -- DELETE NOTHING"
        print("\n  ⛔ VERDICT: the two copies do not even cover the same clips.")

print("\n" + "=" * 74)
for k, v in verdicts.items():
    print("  %-6s %s" % (k, v))
print("=" * 74)
