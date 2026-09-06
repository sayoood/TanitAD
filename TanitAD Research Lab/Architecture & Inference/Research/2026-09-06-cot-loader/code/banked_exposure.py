"""PR-4b: could a BANKED label artifact have been produced through the silent `{}`?

⛔ CHECKED POSITIVELY. The question is not "does the file look complete" -- an
artifact built on an empty Alpamayo dict looks entirely complete, which is the
whole defect. Each artifact is opened and required to carry a POSITIVE marker
that only a non-empty augmentation could have produced (a `cot` string, a box
label, a `meta_action`, or an `alpamayo` layer with real content).

A file that cannot be READ is reported UNREADABLE, never "absent" and never
"clean" -- a 0 from an unreadable file is indistinguishable from a real zero.

ASCII-only stdout.
usage: banked_exposure.py <out.json>
"""
import gzip
import json
import os
import sys

ROOT = "G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD"
TARGETS = [
    "TanitAD Research Lab/Data Engineering/Implementation/incoming/"
    "2026-08-24-label-extraction-overnight/raw/s2_labels_v7.jsonl.gz",
    "TanitAD Research Lab/Data Engineering/Implementation/incoming/"
    "2026-09-04-v72-label-release/raw/s2_labels_v7.2_train.jsonl.gz",
    "TanitAD Research Lab/Data Engineering/Implementation/incoming/"
    "2026-09-04-v72-label-release/raw/s2_labels_v7.2_eval.jsonl.gz",
    "products/P2-data-pipelines/2026-08-23-label-validation-sample/raw/"
    "labels_v7/s2_labels_v7.jsonl",
    "TanitAD Research Lab/Data Engineering/Implementation/incoming/"
    "2026-08-16-s2-v1-labels/labels/s2_labels_aug120.jsonl",
    "TanitAD Research Lab/Data Engineering/Implementation/incoming/"
    "2026-08-23-s2-abstain-v3/labels_v3/s2_labels_aug120.jsonl",
]


def openit(p):
    if p.endswith(".gz"):
        return gzip.open(p, "rt", encoding="utf-8")
    return open(p, "rt", encoding="utf-8")


def marker_of(rec):
    """Positive evidence that a NON-EMPTY Alpamayo augmentation was present."""
    alp = rec.get("alpamayo")
    if isinstance(alp, dict):
        if alp.get("boxes"):
            return "alpamayo.boxes"
        if alp.get("cot") or alp.get("chain_of_causation"):
            return "alpamayo.cot"
        if alp.get("meta_action"):
            return "alpamayo.meta_action"
        if alp.get("lateral", {}).get("alpamayo_side") not in (None, ""):
            return "alpamayo.lateral"
    for k in ("cot", "cot_tokens", "chain_of_causation", "meta_action"):
        v = rec.get(k)
        if v:
            return k
    src = rec.get("sources") or rec.get("provenance")
    if isinstance(src, (str, list)) and "alpamayo" in json.dumps(src):
        return "provenance:alpamayo"
    return None


out = {}
for rel in TARGETS:
    p = os.path.join(ROOT, rel)
    rec = {"path": rel}
    if not os.path.exists(p):
        rec["verdict"] = "NOT_ON_DISK"
        out[rel] = rec
        print("  %-58s NOT_ON_DISK" % rel[-58:])
        continue
    rec["bytes"] = os.path.getsize(p)
    try:
        n = 0
        markers = {}
        first_keys = []
        with openit(p) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                n += 1
                if n > 4000:
                    break
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if not first_keys:
                    first_keys = sorted(r.keys())
                m = marker_of(r)
                markers[m or "NONE"] = markers.get(m or "NONE", 0) + 1
        rec["records_scanned"] = n
        rec["markers"] = markers
        rec["first_keys"] = first_keys
        pos = sum(v for k, v in markers.items() if k != "NONE")
        rec["with_alpamayo_evidence"] = pos
        # ⛔ A file with NO alpamayo-derived key was never at risk -- the
        # geometry-only s2 v1/v3 emitter does not read the augmentation at all.
        # Distinguish that from a file that HAS the key and finds it empty.
        dep = any(k in (rec.get("first_keys") or []) for k in
                  ("alpamayo", "cot_tokens", "cot", "semantics", "meta_action"))
        rec["depends_on_alpamayo"] = bool(dep)
        if not dep:
            rec["verdict"] = "NOT_ALPAMAYO_DEPENDENT (geometry-only schema)"
        elif pos:
            rec["verdict"] = "CLEAN (positive alpamayo content present)"
        else:
            rec["verdict"] = "EXPOSED -- alpamayo-dependent schema, zero content"
        print("  %-58s n=%-5d alpamayo_evidence=%-5d %s"
              % (rel[-58:], n, pos, rec["verdict"]))
    except Exception as exc:
        rec["verdict"] = "UNREADABLE"
        rec["error"] = "%s: %s" % (type(exc).__name__, exc)
        print("  %-58s UNREADABLE %s" % (rel[-58:], exc))
    out[rel] = rec

json.dump(out, open(sys.argv[1], "w", encoding="utf-8"), indent=1, sort_keys=True)
print("WROTE %s" % sys.argv[1])
