# -*- coding: utf-8 -*-
"""Re-open every banked PDF named in published_rows.py and check that each 'auto' row's numbers appear, in print order,
on the stated PDF page. Verifies by CONTENT: a row transcribed wrongly, or a page number that is off, fails.

Usage (from the repo root, CPU only; needs `pdftotext` from poppler on PATH):
    python "TanitAD Research Lab/Architecture & Inference/Research/2026-10-18-ep-rl-lit/raw/verify_published_rows.py" [--papers DIR] [--out FILE]
Exit code 0 only if every 'auto' row matches. 'manual' rows are listed, not checked.
"""
import argparse
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from published_rows import R  # noqa: E402


def find_pdf(papers, key):
    for f in sorted(os.listdir(papers)):
        if f.startswith(key + "_") or f == key + ".pdf":
            return os.path.join(papers, f)
    return None


def page_text(pdf, page, layout=True):
    cmd = ["pdftotext"] + (["-layout"] if layout else []) + ["-f", str(page), "-l", str(page), pdf, "-"]
    out = subprocess.run(cmd, capture_output=True)
    return out.stdout.decode("utf-8", "replace")


def tokens_in_order(text, tokens):
    t = re.sub(r"\s+", " ", text)
    pat = r".{0,80}?".join(r"(?<![0-9.])" + re.escape(x) + r"(?![0-9])" for x in tokens)
    return re.search(pat, t) is not None


def main():
    ap = argparse.ArgumentParser()
    root = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))   # repo root: raw -> pkg -> Research -> Architecture & Inference -> TanitAD Research Lab -> repo
    ap.add_argument("--papers", default=os.path.join(root, "TanitAD Research Lab", "Library", "papers"))
    ap.add_argument("--out", default=os.path.join(HERE, "verify_published_rows.json"))
    a = ap.parse_args()
    res = []
    bad = 0
    for r in R:
        pdf = find_pdf(a.papers, r["key"])
        if pdf is None:
            res.append(dict(id=r["id"], status="NO_PDF"))
            bad += 1
            continue
        if r["check"] == "manual":
            res.append(dict(id=r["id"], status="MANUAL", key=r["key"], page=r["pdf_page"], note=r["note"]))
            continue
        ok = False
        for layout in (True, False):
            if tokens_in_order(page_text(pdf, r["pdf_page"], layout), r["tokens"]):
                ok = True
                break
        res.append(dict(id=r["id"], status="PASS" if ok else "FAIL", key=r["key"], page=r["pdf_page"], tokens=r["tokens"]))
        bad += 0 if ok else 1
    n = {s: sum(1 for x in res if x["status"] == s) for s in ("PASS", "FAIL", "MANUAL", "NO_PDF")}
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(dict(counts=n, rows=res), f, indent=1)
    print(n)
    for x in res:
        if x["status"] in ("FAIL", "NO_PDF"):
            print("  ", x["status"], x["id"], x.get("key"), x.get("page"), x.get("tokens"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
