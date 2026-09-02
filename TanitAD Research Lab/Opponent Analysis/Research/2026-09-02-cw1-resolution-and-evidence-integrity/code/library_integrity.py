"""E-OPP-CW1-1 (part B) — is the banked evidence base actually READABLE?

⛔ WHY THIS EXISTS. Resolving correction-watch **CW-1** (DriveFuture's headline
navhard EPDMS 55.5 vs its own ablation rows 30.9-34.6) required reading the banked
primary `2605.09701`. It is **TRUNCATED** -- no `%%EOF` marker anywhere in
1,373,148 bytes, and pypdf dies with `Stream has ended unexpectedly`.

⭐ THE PART THAT GENERALISES, and it is why this is a survey and not a bug report:
`kb_add.py --verify` re-hashes each file against the hash recorded **at bank
time**. If the download was already truncated when it was hashed, the hash MATCHES
and the file "verifies" -- forever. ⇒ **`--verify` proves a file has not CHANGED;
it cannot prove the file was ever COMPLETE.** A corrupt primary therefore sits in
the Library indefinitely, indistinguishable from a good one, until somebody tries
to read it.

⚠️ This is the failure mode `LAB_BACKLOG.md` row 36 predicted from a single case in
run 001 (*"truncated 5 MiB PDF without %%EOF"*). This script measures how many
there actually are.

The check, per file, cheapest-first:
  1. **magic** -- does it start with `%PDF`?
  2. **EOF marker** -- does `%%EOF` appear in the last 4 KiB? (a well-formed PDF
     ends with `startxref` / `%%EOF`)
  3. **whole-file EOF** -- if not in the tail, is it anywhere at all? (distinguishes
     "appended garbage" from "truncated")
  4. **parse** -- pypdf page count, and text extraction from page 1.

⛔ CONTENT, NOT PRESENCE. `os.path.exists` and a size are not evidence a paper is
readable -- that is exactly the class the memmap-of-zeros retraction named. A file
that exists at the right size and cannot be opened is worse than a missing one,
because the index reports it as an asset.

Usage:
    python library_integrity.py --out ../raw/library_integrity.json
"""
from __future__ import annotations

import argparse
import io
import json
import os
from datetime import datetime, timezone

REPO = r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD"
LIB = os.path.join(REPO, "TanitAD Research Lab", "Library", "library.json")
TAIL = 4096


def check(path: str) -> dict:
    rec: dict = {"exists": os.path.exists(path)}
    if not rec["exists"]:
        rec["verdict"] = "MISSING"
        return rec
    rec["size"] = os.path.getsize(path)
    try:
        with open(path, "rb") as fh:
            head = fh.read(8)
            fh.seek(max(0, rec["size"] - TAIL))
            tail = fh.read(TAIL)
    except Exception as e:                                    # noqa: BLE001
        rec["verdict"] = "UNREADABLE_BYTES"
        rec["error"] = f"{type(e).__name__}: {str(e)[:120]}"
        return rec

    rec["is_pdf_magic"] = head.startswith(b"%PDF")
    rec["eof_in_tail"] = b"%%EOF" in tail
    if not rec["eof_in_tail"]:
        # only now pay for a whole-file scan
        with open(path, "rb") as fh:
            rec["eof_anywhere"] = b"%%EOF" in fh.read()
    else:
        rec["eof_anywhere"] = True

    if not rec["is_pdf_magic"]:
        rec["verdict"] = "NOT_A_PDF"
        return rec
    if not rec["eof_anywhere"]:
        rec["verdict"] = "TRUNCATED_NO_EOF"
        return rec

    try:
        from pypdf import PdfReader
        r = PdfReader(path)
        rec["n_pages"] = len(r.pages)
        t = r.pages[0].extract_text() or ""
        rec["page1_chars"] = len(t)
        rec["verdict"] = "OK" if rec["n_pages"] > 0 else "ZERO_PAGES"
    except Exception as e:                                    # noqa: BLE001
        rec["verdict"] = "PARSE_FAILED"
        rec["error"] = f"{type(e).__name__}: {str(e)[:120]}"
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    lib = json.load(io.open(LIB, encoding="utf-8"))
    entries = lib["entries"]
    keys = sorted(entries)
    if a.limit:
        keys = keys[:a.limit]

    out, tally = {}, {}
    for k in keys:
        e = entries[k]
        rel = e.get("path") or ""
        full = rel if os.path.isabs(rel) else os.path.join(REPO, rel)
        rec = check(full)
        rec["title"] = (e.get("title") or "")[:110]
        rec["tags"] = e.get("tags")
        rec["path"] = rel
        out[k] = rec
        tally[rec["verdict"]] = tally.get(rec["verdict"], 0) + 1
        print(f"{rec['verdict']:18s} {k}  {rec['title'][:70]}", flush=True)

    bad = {k: v for k, v in out.items() if v["verdict"] != "OK"}
    res = {
        "_experiment": "E-OPP-CW1-1 (part B)",
        "_date": "2026-09-02",
        "_generated_utc": datetime.now(timezone.utc).isoformat(),
        "_evidence_class": "MEASURED (ours; byte-level checks over the banked Library)",
        "_tier": "N/A -- an artifact-integrity measurement",
        "n_entries_checked": len(keys),
        "tally": tally,
        "unreadable": bad,
        "all": out,
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with io.open(a.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print(f"\nTALLY {tally}")
    print(f"UNREADABLE {len(bad)} of {len(keys)}")
    for k, v in bad.items():
        print(f"  {v['verdict']:18s} {k}  {v['title'][:60]}")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
