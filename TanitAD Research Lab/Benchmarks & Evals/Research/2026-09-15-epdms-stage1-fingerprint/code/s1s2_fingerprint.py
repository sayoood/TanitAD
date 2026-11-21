"""E-BE-S1S2-1 — Stage-1 / Stage-2 substrate fingerprint of external navhard EPDMS rows (0 GPU).

For each banked primary: extract text (pypdf), find every PDM-Closed mention, and count which
fingerprint tokens occur in a +-700-char window. Stage 1 of PDM-Closed is deterministic and identical
on both known substrates; Stage 2 separates the 08/2025 (v2) and 03/2026 (v3/TOAD) snapshots.
Reference values: repo:TanitAD Research Lab/Benchmarks & Evals/Research/2026-09-13-pdmc-stage2-provenance/raw/stage_subscores.md
"""
import glob, json, re, sys
from pypdf import PdfReader
LIB = "D:/Projects/TanitAD/TanitAD Research Lab/Library/papers/"
S1 = ["94.4", "98.8", "99.5", "93.5", "99.3", "87.7", "36.0"]            # NC DAC TLC TTC LK HC EC (DDC/EP = 100 omitted: uninformative)
V2 = ["51.3", "88.1", "83.1", "25.4", "73.7", "91.5", "96.3", "98.5"]    # 08/2025 snapshot Stage-2 (+ headline)
V3 = ["56.6", "90.5", "86.6", "29.7", "74.2", "91.9", "95.4", "98.4"]    # 03/2026 snapshot Stage-2 (+ headline)
PAPERS = {"2605.09701": "DriveFuture 55.5", "2601.05083": "DrivoR 54.6/56.3", "2605.15120": "CLOVER 48.3",
          "2510.04333": "RAP-DINO 36.9", "2606.07170": "TOAD (POSITIVE CONTROL: must match v3)",
          "2506.04218": "Pseudo-Sim v3 (POSITIVE CONTROL: must match v3)", "2511.18729": "GuideFlow 23.1-45.0"}
out = {"reference": {"S1": S1, "V2": V2, "V3": V3}, "papers": {}}
for pid, label in PAPERS.items():
    f = glob.glob(LIB + pid + "*.pdf")
    if not f:
        out["papers"][pid] = {"label": label, "status": "NOT BANKED / file absent"}; continue
    r = PdfReader(f[0]); txt = "\n".join((p.extract_text() or "") for p in r.pages)
    ver = re.search(r"arXiv:\s*%s(v\d+)" % re.escape(pid), txt)
    hits = [m.start() for m in re.finditer(r"PDM[- ]?Closed|PDM-C\b", txt)]
    best = {"S1": 0, "V2": 0, "V3": 0}; ctx = []
    for h in hits:
        w = txt[max(0, h - 700): h + 700]
        nums = set(re.findall(r"\d{2,3}\.\d", w))
        sc = {k: sum(t in nums for t in v) for k, v in (("S1", S1), ("V2", V2), ("V3", V3))}
        if sum(sc.values()) > sum(best.values()):
            best = sc; ctx = [w.replace("\n", " ")]
    out["papers"][pid] = {"label": label, "file": f[0].split("/")[-1], "chars": len(txt), "arxiv_version": ver.group(1) if ver else None,
                          "pdm_closed_mentions": len(hits), "navhard_mentions": len(re.findall(r"navhard", txt, re.I)),
                          "best_window_token_hits": best, "best_window": ctx[:1]}
json.dump(out, open(sys.argv[1], "w", encoding="utf-8"), indent=1, ensure_ascii=False)
for pid, v in out["papers"].items():
    print(pid, v.get("label"), v.get("arxiv_version"), v.get("pdm_closed_mentions"), v.get("navhard_mentions"), v.get("best_window_token_hits", v.get("status")))
