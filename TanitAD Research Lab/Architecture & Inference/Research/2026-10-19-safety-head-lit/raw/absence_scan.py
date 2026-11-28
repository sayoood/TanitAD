#!/usr/bin/env python3
"""Scoped absence scan: which banked papers print a per-head accuracy / calibration / ranking-quality number for a
safety sub-score head (NC, DAC, TTC) of a NAVSIM scorer?

Method (reproducible, deliberately loose so that it over-reports): extract each banked PDF with PyMuPDF, collapse
whitespace, and count regex hits of metric words (AUC, AUROC, ECE, calibration, reliability, concordance, pairwise
ranking accuracy, precision, recall, F1) within 160 characters of a safety-head word (NC, DAC, TTC, no at-fault collision,
drivable area compliance, time to collision). Every hit is printed with its page so a reader can open it. A paper with zero
hits is NOT proof of absence in the paper's appendix figures; it is a statement about its text layer.
Usage: python absence_scan.py <library papers dir> <arxiv ids file> <out json>
"""
import glob, json, os, re, sys
import fitz

METRIC = re.compile(r"\b(auc|auroc|roc|ece|expected calibration|calibrat\w*|reliability|concordance|pairwise ranking accuracy|precision|recall|f1)\b", re.I)
HEAD = re.compile(r"\b(nc|dac|ttc|no at-fault collision\w*|drivable area compliance|time[- ]to[- ]collision)\b", re.I)


def main(lib, ids_file, out):
    ids = [l.strip() for l in open(ids_file, encoding="utf-8") if l.strip()]
    res = {}
    for i in ids:
        fs = glob.glob(os.path.join(lib, i + "_*.pdf")) or glob.glob(os.path.join(lib, i + "v*_*.pdf"))
        if not fs:
            res[i] = {"status": "NOT_IN_LIBRARY"}
            continue
        doc = fitz.open(fs[0])
        hits = []
        chars = 0
        any_metric = 0   # POSITIVE CONTROL: metric words anywhere on the page, no head-word requirement
        for pn, pg in enumerate(doc, 1):
            t = re.sub(r"\s+", " ", pg.get_text("text"))
            chars += len(t)
            any_metric += len(METRIC.findall(t))
            for m in METRIC.finditer(t):
                lo, hi = max(0, m.start() - 160), min(len(t), m.end() + 160)
                if HEAD.search(t[lo:hi]):
                    hits.append({"page": pn, "metric": m.group(0), "ctx": t[lo:hi]})
        res[i] = {"status": "SCANNED", "pdf": os.path.basename(fs[0]), "chars": chars, "metric_words_anywhere": any_metric, "n_hits": len(hits), "hits": hits[:12]}
    json.dump(res, open(out, "w", encoding="utf-8", newline="\n"), indent=1, ensure_ascii=False)
    sc = [r for r in res.values() if r["status"] == "SCANNED"]
    print(f"{len(sc)} papers scanned; min text-layer chars {min(r['chars'] for r in sc)}; "
          f"{sum(r['metric_words_anywhere'] > 0 for r in sc)} have metric words somewhere (control); "
          f"{sum(r['n_hits'] > 0 for r in sc)} have a metric word within 160 chars of a safety-head word")
    for i, r in res.items():
        if r.get("n_hits"):
            print("HIT", i, [(h["page"], h["metric"]) for h in r["hits"]])


if __name__ == "__main__":
    main(*sys.argv[1:4])
