#!/usr/bin/env python3
"""Verify every tabled number against the banked PDF.

For each row in rows_def.ROWS the script opens the banked PDF (TanitAD Research Lab/Library/papers/<key>_*.pdf), clusters the
words of every page into visual lines, and requires the row's printed numbers to appear as an ORDERED SUBSEQUENCE of one visual
line. It records every PDF page (1-based physical page) where that holds. A row PASSes if the page set is non-empty; the
`page` column is the first such page. A row whose numbers are found nowhere is FAIL and is not quotable.

Usage:  python verify_published_rows.py <library papers dir> <out csv> <out json>
Content check, not presence check: the PDF is parsed, not grepped for the arXiv id.
"""
import csv, json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pdfrows import find_pdf, pages_with
from rows_def import ROWS


# a table whose rows also occur on an earlier page (a different table) is pinned to its own page
PIN = {("2506.06659", "Tab.5"): 7, ("2602.18887", "Tab.6"): 7, ("2602.18887", "Tab.5"): 7, ("2602.18887", "Tab.4"): 7, ("2505.15111", "Tab.4"): 8, ("2510.24108", "Tab.8"): 17}


def main(lib, out_csv, out_json):
    res = []
    for key, tab, group, label, cols, vals in ROWS:
        pdf = find_pdf(lib, key)
        pages = pages_with(pdf, vals)
        pin = PIN.get((key, tab))
        if pin is not None:
            pages = [pin] if pin in pages else []
        res.append(dict(key=key, table=tab, group=group, label=label, cols=cols, values=vals,
                        pages=pages, page=(pages[0] if pages else None),
                        status="PASS" if pages else "FAIL"))
    with open(out_csv, "w", encoding="utf-8", newline="\n") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["arxiv", "table", "pdf_page", "group", "row", "columns", "printed_values", "status", "all_pages_matching"])
        for r in res:
            w.writerow([r["key"], r["table"], r["page"], r["group"], r["label"], " ".join(r["cols"]),
                        " ".join(r["values"]), r["status"], ";".join(map(str, r["pages"]))])
    with open(out_json, "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1)
    n = len(res)
    p = sum(1 for r in res if r["status"] == "PASS")
    print(f"{p}/{n} rows PASS")
    for r in res:
        if r["status"] != "PASS":
            print("FAIL", r["key"], r["table"], r["label"], r["values"])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
