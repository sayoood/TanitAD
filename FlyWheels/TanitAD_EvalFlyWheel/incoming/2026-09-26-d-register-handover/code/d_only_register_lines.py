#!/usr/bin/env python3
"""Which lines of D:'s copy of a steering file are NOT on the branch tip — and which of those may be handed over.

WHY (2026-09-26, Master Mind request). D:'s register copies are ~3,700 lines behind the tip, and streams working
in D: edited them and listed the WHOLE FILES for landing — which the lander correctly refuses, because a whole-file
land would revert the tip's newer lines. So those edits are invisible on the branch. The safe handover is APPEND
BLOCKS, and the hard part is deciding which lines they are. "Not on the tip" is NOT enough:

  * HISTORICAL        the line exists in some branch version of the file (HEAD..tip) — the tip since changed it.
                      Re-adding it would resurrect superseded text (e.g. a retracted number). NEVER hand over.
  * NEVER_ON_BRANCH   written on D: only. Sub-classified against the closest tip line:
      - SHA12_REDACTION     every changed token is a clip handle rewritten to ``sha12:`` — an edit of an EXISTING
                            tip row, not content. The PI ruled 2026-09-17 not to redact banked evidence
                            (``tools/clipid_scan.py`` docstring). NOT handed over.
      - DESCENDANT          the tip row survives CHARACTER-COMPLETE inside the D: row, which adds to it — a safe
                            REPLACEMENT (it cannot revert anything). Handed over as REPLACE.
      - OTHER_EDIT          any other edit of an existing tip row — reported with its position, never auto-applied.
      - NEW                 no tip line within similarity 0.5 — handed over as APPEND, verbatim.

⛔ Only NEW and DESCENDANT lines are written out as TEXT. Everything else is recorded by line number, class, row id
and sha256 — so this tool never copies a clip handle into a new file (the clip-id guard stops the count GROWING).

usage:  python d_only_register_lines.py --tip <sha> --out <json> [--emit-dir <dir>] -- <file> [<file> ...]
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
import subprocess
from pathlib import Path

SEP = re.compile(r"\|?[\s:|-]+\|?")
ROW_ID = re.compile(r"\|\s*[^\w|]*\**\s*([A-Z][A-Z0-9]*\d[A-Z0-9]*(?:-[A-Za-z0-9]+)*|[A-Z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)+)")
TOK = re.compile(r"\S+")


def _lines(text: str) -> list:
    return text.replace("\r\n", "\n").split("\n")


def blob_lines(ref: str, path: str) -> list:
    r = subprocess.run(["git", "show", f"{ref}:{path}"], capture_output=True)
    if r.returncode != 0:
        raise SystemExit(f"cannot read {ref}:{path} -- refusing to classify against an unread tip")
    return _lines(r.stdout.decode("utf-8"))


def branch_history(tip: str, path: str) -> tuple:
    revs = subprocess.run(["git", "rev-list", f"HEAD..{tip}", "--", path], capture_output=True, text=True,
                          check=True).stdout.split()
    hist = set(blob_lines("HEAD", path))
    for c in revs:
        hist |= set(blob_lines(c, path))
    return hist, len(revs)


def subclass(line: str, tip_lines: list) -> dict:
    c = difflib.get_close_matches(line, tip_lines, n=1, cutoff=0.5)
    if not c:
        return {"sub": "NEW", "similarity": 0.0}
    t = c[0]
    sim = round(difflib.SequenceMatcher(None, t, line, autojunk=False).ratio(), 3)
    a, b = TOK.findall(t), TOK.findall(line)
    ops = [o for o in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes() if o[0] != "equal"]
    changed = [(" ".join(a[i1:i2]), " ".join(b[j1:j2])) for _, i1, i2, j1, j2 in ops]
    # ⛔ SAME-ROW EVIDENCE. Two DIFFERENT items written from one template ("## ITEM (<date> ~<time> Berlin) — <title>")
    # clear a 0.5 similarity cut. MEASURED 2026-09-26 on PI_DECISION_QUEUE.md: a REFe item's heading was paired with
    # a refcv7 item's and read as an EDIT. An edit keeps most of the row: if more than half of the tip line's tokens
    # changed, it is a different row, i.e. NEW.
    tip_changed = sum(i2 - i1 for _, i1, i2, _, _ in ops)
    if a and tip_changed / len(a) > 0.5:
        return {"sub": "NEW", "similarity": sim, "nearest_tip_line_rejected": "over half its tokens changed"}
    if changed and all("sha12:" in new and "sha12:" not in old for old, new in changed):
        return {"sub": "SHA12_REDACTION", "similarity": sim, "n_changes": len(changed)}
    cops = difflib.SequenceMatcher(None, t, line, autojunk=False).get_opcodes()
    lost = sum(i2 - i1 for op, i1, i2, _, _ in cops if op in ("delete", "replace"))
    if lost == 0 and len(line) > len(t):
        return {"sub": "DESCENDANT", "similarity": sim, "tip_row_sha256": hashlib.sha256(t.encode()).hexdigest(),
                "tip_line": tip_lines.index(t) + 1, "tip_chars": len(t), "added_chars": len(line) - len(t)}
    pos = next((i1 for op, i1, *_ in cops if op != "equal"), None)
    return {"sub": "OTHER_EDIT", "similarity": sim, "n_changes": len(changed), "first_diff_char": pos,
            "chars_changed_on_tip_side": lost}


def classify(tip: str, path: str) -> dict:
    W = _lines(Path(path).read_text(encoding="utf-8"))
    T = blob_lines(tip, path)
    Ts = set(T)
    hist, n_versions = branch_history(tip, path)
    rows = []
    for n, line in enumerate(W, 1):
        s = line.strip()
        if not s or line in Ts or SEP.fullmatch(s):
            continue
        m = ROW_ID.match(line)
        rec = {"n": n, "sha256": hashlib.sha256(line.encode()).hexdigest(), "row_id": m.group(1) if m else None}
        if line in hist:
            rec["class"] = "HISTORICAL"
        else:
            rec["class"] = "NEVER_ON_BRANCH"
            rec.update(subclass(line, T))
            if rec["sub"] == "NEW" and rec["row_id"]:
                rec["row_id_on_tip"] = any(rec["row_id"] in x[:120] for x in T if x.startswith("|"))
        rows.append(rec)
    counts = {}
    for r in rows:
        k = r["class"] if r["class"] == "HISTORICAL" else r["sub"]
        counts[k] = counts.get(k, 0) + 1
    return {"file": path, "tip": tip, "lines_worktree": len(W), "lines_tip": len(T),
            "branch_versions_checked": n_versions + 1, "d_only_lines": len(rows), "counts": counts, "rows": rows,
            "_W": W, "_T": T}


def _row_label(r: dict) -> str:
    """A replaced row by its id, or -- for a field row such as `| **Status** |`, which has none -- by the tip line."""
    return r.get("row_id") or f"at tip line {r.get('tip_line')}"


def anchor(rep: dict, a: int) -> str:
    """Where a region goes: the nearest PRECEDING D: line that is on the tip verbatim, cited by TIP LINE NUMBER and
    hash -- never by copied text, which could carry a clip handle into a new file -- or a replaced row, by id."""
    W, T = rep["_W"], rep.get("_T") or []
    tip_index = {}
    for i, x in enumerate(T, 1):
        tip_index.setdefault(x, i)
    desc = {r["n"]: _row_label(r) for r in rep["rows"] if r.get("sub") == "DESCENDANT"}
    for k in range(a - 1, 0, -1):
        if k in desc:
            return f"insert after the row {desc[k]} (it is replaced -- see REPLACE file)"
        x = W[k - 1]
        if x.strip() and x in tip_index:
            return (f"insert after tip line {tip_index[x]} (sha256 {hashlib.sha256(x.encode()).hexdigest()[:12]}, "
                    f"= D: line {k})")
    return "insert at the top of the file"


def emit(rep: dict, out_dir: Path) -> list:
    """APPEND = contiguous NEW regions verbatim (blank lines inside a region kept); REPLACE = DESCENDANT rows."""
    W, stem, written = rep["_W"], Path(rep["file"]).stem, []
    new = [r["n"] for r in rep["rows"] if r.get("sub") == "NEW"]
    d_only = {r["n"] for r in rep["rows"]}
    regions, cur = [], None
    for n in new:
        # a gap of <= 3 lines that are blank or ALREADY ON THE TIP stays inside the region (a block may repeat a
        # common line such as a separator); a gap holding any OTHER D:-only line ends it
        if cur and n - cur[1] <= 4 and all(not W[k - 1].strip() or k not in d_only for k in range(cur[1] + 1, n)):
            cur[1] = n
        else:
            cur = [n, n]
            regions.append(cur)
    if regions:
        p = out_dir / f"APPEND_{stem}.md"
        body = [f"<!-- APPEND to `Project Steering/{Path(rep['file']).name}` on the tip {rep['tip'][:8]} -- "
                f"{len(regions)} region(s), copied verbatim from D:'s worktree lines "
                + ", ".join(f"{a}-{b}" for a, b in regions) + ". Land as a MOD on the tip blob; never as a file. -->", ""]
        for a, b in regions:
            body += [f"<!-- region D: {a}-{b}: {anchor(rep, a)} -->"] + W[a - 1:b] + [""]
        p.write_text("\n".join(body), encoding="utf-8")
        written.append(str(p))
    desc = [r for r in rep["rows"] if r.get("sub") == "DESCENDANT"]
    if desc:
        p = out_dir / f"REPLACE_{stem}.md"
        body = [f"<!-- REPLACE rows in `Project Steering/{Path(rep['file']).name}` on the tip {rep['tip'][:8]}: each line "
                f"below replaces the tip row with the same id. Each is a DESCENDANT: the tip row survives in it "
                f"character-complete (0 characters lost), so the replacement cannot revert anything. -->", ""]
        for r in desc:
            body += [f"<!-- row {_row_label(r)}: D: line {r['n']}; tip row sha256 {r['tip_row_sha256'][:16]}..., "
                     f"{r['tip_chars']} chars kept + {r['added_chars']} added -->", W[r["n"] - 1], ""]
        p.write_text("\n".join(body), encoding="utf-8")
        written.append(str(p))
    return written


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tip", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--emit-dir", default=None)
    ap.add_argument("files", nargs="+")
    a = ap.parse_args(argv)
    reps, written = [], []
    for f in a.files:
        rep = classify(a.tip, f)
        if a.emit_dir:
            written += emit(rep, Path(a.emit_dir))
        rep.pop("_W"); rep.pop("_T", None)
        reps.append(rep)
        print(f"{f}: D:-only {rep['d_only_lines']} over {rep['branch_versions_checked']} branch versions -> {rep['counts']}")
    Path(a.out).write_text(json.dumps({"schema": "d-only-register-lines/1", "reports": reps, "emitted": written},
                                      indent=1, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
