#!/usr/bin/env python
"""dscale_evalview.py -- the evaluation view of card PREREG_V8F_PROOF_ADD45 (note N3). Standard library only, CPU, seconds.

Control d2e asks that the evaluation manifest the kit reads holds the study's evaluation clips in the study's manifest order (file STUDY_EVAL_MD5.tsv of the kit).
A pod whose own evaluation cache lists the same clips in another order gets a VIEW folder, not a rewritten cache:

    <view>/eval/manifest.jsonl    the lines of the source manifest, byte for byte, in the order of STUDY_EVAL_MD5.tsv
    <view>/eval/<entry>           a link to every other entry of the source evaluation folder (clips, meta.json, ...); names that start with a dot are left out

Nothing is copied and no manifest line is edited. The kit reads the evaluation set from the view at every stage, and d2e is checked on the view.

Refused (exit 2): the id sets differ, an id occurs twice on either side, a count is not --expect-eval, the view folder is the source folder, an existing view
manifest differs from the one this call would write, or an existing view entry is not the link this call would make.

Exit codes: 0 the view is in place (written now, or found equal), 2 refused, 1 an input is missing.
"""
import argparse
import hashlib
import json
import os
import sys


def find_manifest(cache, split):
    for p in (os.path.join(cache, split, "manifest.jsonl"), os.path.join(cache, "manifest.jsonl")):
        if os.path.exists(p):
            return p
    return None


def read_lines(path):
    """[(clip id, the line's bytes without its line end)] in file order; empty lines are left out."""
    rows = []
    with open(path, "rb") as fh:
        for raw in fh.read().split(b"\n"):
            raw = raw.rstrip(b"\r")
            if raw.strip():
                rows.append((str(json.loads(raw.decode("utf-8"))["clip"]), raw))
    return rows


def read_tsv_ids(path):
    ids = []
    for ln in open(path, encoding="utf-8"):
        ln = ln.rstrip("\n").rstrip("\r")
        if ln and not ln.startswith("#"):
            ids.append(ln.split("\t")[0])
    return ids


def sha_ids(ids):                                                                    # the same digest as dscale_tokens_same.py
    return hashlib.sha256(("\n".join(ids) + "\n").encode("utf-8")).hexdigest()


def build(eval_cache, study_tsv, view, expect):
    """Returns (record, exit code)."""
    rec = {"tool": "dscale_evalview.py", "pass": False, "eval_cache": eval_cache, "view": view, "expect_eval": expect}
    src_manifest = find_manifest(eval_cache, "eval")
    if src_manifest is None or not os.path.exists(study_tsv):
        rec["missing"] = [p for p in (os.path.join(eval_cache, "eval", "manifest.jsonl"), study_tsv) if not os.path.exists(p)]
        return rec, 1
    src_dir = os.path.dirname(os.path.abspath(src_manifest))
    vdir = os.path.join(os.path.abspath(view), "eval")
    vman = os.path.join(vdir, "manifest.jsonl")
    rows = read_lines(src_manifest)
    ids = [c for c, _ in rows]
    want = read_tsv_ids(study_tsv)
    rec.update({"source_manifest": src_manifest, "source_manifest_sha256": hashlib.sha256(open(src_manifest, "rb").read()).hexdigest(), "n_here": len(ids),
                "n_study": len(want), "sha256_ids_source": sha_ids(ids), "sha256_ids_study": sha_ids(want), "source_in_study_order": ids == want,
                "positions_that_differ": sum(1 for a, b in zip(ids, want) if a != b) + abs(len(ids) - len(want))})
    refused = []
    if os.path.realpath(src_dir) == os.path.realpath(vdir):
        refused.append("the view folder is the source folder")
    if len(set(ids)) != len(ids):
        refused.append("an id occurs twice in the source manifest")
    if len(set(want)) != len(want):
        refused.append("an id occurs twice in the study list")
    if set(ids) != set(want):
        refused.append("the id sets differ: %d only here, %d only in the study list" % (len(set(ids) - set(want)), len(set(want) - set(ids))))
    if len(ids) != expect or len(want) != expect:
        refused.append("counts: %d here, %d in the study list, %d expected" % (len(ids), len(want), expect))
    if refused:
        rec["refused"] = refused
        return rec, 2
    by = dict(rows)
    body = b"".join(by[c] + b"\n" for c in want)
    os.makedirs(vdir, exist_ok=True)
    written = False
    if os.path.exists(vman):
        if open(vman, "rb").read() != body:
            rec["refused"] = ["an existing view manifest differs from the one this call would write: " + vman]
            return rec, 2
    else:
        with open(vman, "wb") as fh:
            fh.write(body)
        written = True
    links = []
    for name in sorted(os.listdir(src_dir)):
        if name == "manifest.jsonl" or name.startswith("."):
            continue
        tgt, lnk = os.path.join(src_dir, name), os.path.join(vdir, name)
        if os.path.lexists(lnk):
            if not (os.path.islink(lnk) and os.path.realpath(lnk) == os.path.realpath(tgt)):
                rec["refused"] = ["an existing view entry is not the link to the source entry: " + lnk]
                return rec, 2
        else:
            os.symlink(tgt, lnk)
        links.append(name)
    back = read_lines(vman)
    back_ids = [c for c, _ in back]
    rec.update({"view_manifest": vman, "view_manifest_sha256": hashlib.sha256(open(vman, "rb").read()).hexdigest(), "manifest_written_by_this_call": written,
                "links": links, "sha256_ids_view": sha_ids(back_ids), "view_in_study_order": back_ids == want,
                "lines_unchanged": sorted(r for _, r in back) == sorted(r for _, r in rows)})
    rec["pass"] = bool(rec["view_in_study_order"] and rec["lines_unchanged"])
    return rec, 0 if rec["pass"] else 2


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-cache", required=True, help="the pod's evaluation cache (holds eval/manifest.jsonl, or manifest.jsonl at its root)")
    ap.add_argument("--study-eval-tsv", required=True)
    ap.add_argument("--view", required=True, help="the view folder; the kit then uses it as its evaluation cache")
    ap.add_argument("--expect-eval", type=int, default=141)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rec, rc = build(a.eval_cache, a.study_eval_tsv, a.view, a.expect_eval)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(rec, fh, indent=1)
    print("evalview: rc %d pass %s source in study order %s, positions that differ %s, written now %s%s -> %s" % (
        rc, rec["pass"], rec.get("source_in_study_order"), rec.get("positions_that_differ"), rec.get("manifest_written_by_this_call"),
        (" REFUSED: " + "; ".join(rec["refused"])) if rec.get("refused") else (" MISSING: " + "; ".join(rec["missing"])) if rec.get("missing") else "", a.out))
    return rc


if __name__ == "__main__":
    sys.exit(main())
