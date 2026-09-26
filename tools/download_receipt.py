#!/usr/bin/env python3
"""APPEND-ONLY receipts for authorised dataset downloads.

⛔ WHY (MEASURED 2026-09-26). The nuScenes fetch script wrote its receipt with ``printf … > RECEIPT``.
Re-running it to add S3-ETag verification TRUNCATED the file, and the receipt's ``utc`` moved from the
real first fetch (13:10:51Z) to the re-verification (13:47:12Z). The receipt had forgotten WHEN the
Terms were accepted and the bytes arrived — the one fact it exists to keep. (Caught by the Master Mind;
the original survives in git at ``55aa747``.)

THE CONTRACT
* ``first_fetch_utc`` and ``accepted_terms_by`` are written ONCE and never changed.
* Every run is APPENDED to ``runs``; every verification is APPENDED to its file's ``verifications``.
* ⛔ If a re-fetch sees DIFFERENT bytes or md5 for a key already recorded, that is REFUSED (exit 1),
  never absorbed: a different object was served, and a receipt that silently updates has stopped being
  evidence of what was fetched.
* Writes are atomic (temp file + ``os.replace``), so an interrupted run cannot leave a torn receipt.

    download_receipt.py open   <receipt> --tier T --who W --source URL [--first-fetch-utc ISO --restored-from TEXT]
    download_receipt.py record <receipt> --key K --bytes N --md5 H --etag E --verdict V
    download_receipt.py close  <receipt> [--note KEY=VALUE ...]
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _save(path: str, doc: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)


def _migrate(doc: dict) -> dict:
    """A pre-2026-09-26 receipt carried a single ``utc`` and no history. Keep every byte of meaning:
    ``utc`` becomes ``first_fetch_utc`` ONLY if nothing better is known (``open`` can restore it)."""
    if "first_fetch_utc" not in doc and "utc" in doc:
        doc["first_fetch_utc"] = doc.pop("utc")
        doc.setdefault("migrated_from_single_utc", True)
    doc.setdefault("runs", [])
    for f in doc.get("files", []):
        if "verifications" not in f:
            v = {k: f.pop(k) for k in ("s3_etag", "etag_verdict") if k in f}
            f["verifications"] = [dict(v, verified_utc=None)] if v else []
        f.setdefault("first_fetch_utc", doc.get("first_fetch_utc"))
    return doc


def cmd_open(a) -> int:
    doc = _load(a.receipt)
    now = _now()
    if doc is None:
        doc = {"tier": a.tier, "source": a.source, "accepted_terms_by": a.who,
               "first_fetch_utc": a.first_fetch_utc or now, "files": [], "runs": []}
        action = "fetch"
    else:
        doc = _migrate(doc)
        for k, v in (("tier", a.tier), ("source", a.source)):
            if doc.get(k) != v:
                print(f"REFUSED: receipt is for {k}={doc.get(k)!r}, this run is {v!r} — "
                      f"never merge receipts across {k}s", file=sys.stderr)
                return 1
        action = "re-verify"
    if a.first_fetch_utc:
        # one-time REPAIR of a receipt that was overwritten before this tool existed -- recorded, never silent
        if doc.get("first_fetch_utc") != a.first_fetch_utc:
            doc.setdefault("repairs", []).append(
                {"utc": now, "field": "first_fetch_utc", "was": doc.get("first_fetch_utc"),
                 "now": a.first_fetch_utc, "restored_from": a.restored_from or "unstated"})
            doc["first_fetch_utc"] = a.first_fetch_utc
            for f in doc.get("files", []):
                f["first_fetch_utc"] = a.first_fetch_utc
    doc["runs"].append({"utc": now, "who": a.who, "action": action})
    _save(a.receipt, doc)
    print(f"receipt {action}: first_fetch_utc={doc['first_fetch_utc']} runs={len(doc['runs'])}")
    return 0


def cmd_record(a) -> int:
    doc = _load(a.receipt)
    if doc is None:
        print("REFUSED: no receipt — run `open` first", file=sys.stderr)
        return 1
    doc = _migrate(doc)
    now = _now()
    v = {"verified_utc": now, "s3_etag": a.etag, "etag_verdict": a.verdict}
    entry = next((f for f in doc["files"] if f.get("key") == a.key), None)
    if entry is None:
        doc["files"].append({"key": a.key, "bytes": int(a.bytes), "md5": a.md5,
                             "first_fetch_utc": now, "verifications": [v]})
    else:
        if int(entry["bytes"]) != int(a.bytes) or entry["md5"] != a.md5:
            print(f"REFUSED: {a.key} now has {a.bytes} B / md5 {a.md5}, but the receipt recorded "
                  f"{entry['bytes']} B / md5 {entry['md5']} at first fetch "
                  f"({entry.get('first_fetch_utc')}). A DIFFERENT object was served; the receipt will "
                  f"not absorb it.", file=sys.stderr)
            return 1
        entry["verifications"].append(v)
    _save(a.receipt, doc)
    return 0


def cmd_close(a) -> int:
    doc = _load(a.receipt)
    if doc is None:
        print("REFUSED: no receipt", file=sys.stderr)
        return 1
    doc = _migrate(doc)
    for kv in a.note or []:
        k, _, val = kv.partition("=")
        doc[k] = val                      # idempotent: the same note written again is the same value
    _save(a.receipt, doc)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("open")
    o.add_argument("receipt"); o.add_argument("--tier", required=True)
    o.add_argument("--who", required=True); o.add_argument("--source", required=True)
    o.add_argument("--first-fetch-utc"); o.add_argument("--restored-from")
    r = sub.add_parser("record")
    r.add_argument("receipt"); r.add_argument("--key", required=True)
    r.add_argument("--bytes", required=True); r.add_argument("--md5", required=True)
    r.add_argument("--etag", required=True); r.add_argument("--verdict", required=True)
    c = sub.add_parser("close")
    c.add_argument("receipt"); c.add_argument("--note", action="append")
    a = ap.parse_args(argv)
    return {"open": cmd_open, "record": cmd_record, "close": cmd_close}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
