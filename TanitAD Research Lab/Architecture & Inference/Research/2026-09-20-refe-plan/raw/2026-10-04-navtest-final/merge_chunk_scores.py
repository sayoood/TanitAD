"""Merge an arm's per-chunk harness CSVs (scored chunk by chunk: a 3,069-token list exceeds Windows' 32 KB command line)
into the single CSV a9_analyze.py reads. Refuses unless every chunk PASSED its own harness guards (C1 / C2 / C3) and the
merged token set equals the arm's registered token set exactly, each token once.
    python merge_chunk_scores.py pdm_route 6
"""
import csv
import json
import os
import sys

DRV = os.environ.get("REFE_DRIVE", "D:")
SC = f"{DRV}/Projects/TanitAD/data/refe_navtest/score"
A9 = f"{DRV}/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-01-goal-trigger/a9"
SP = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/a085b71c-4fb3-409b-ae6b-65f605ed575e/scratchpad"


def main():
    arm, n = sys.argv[1], int(sys.argv[2])
    want = json.load(open(f"{A9}/tokens_{arm}.json", encoding="utf-8"))["tokens"]
    rows, header, statuses = [], None, {}
    for c in range(n):
        lab = f"refe_a9_{arm}_c{c}"
        st = None
        for line in reversed(open(f"{SP}/score_a9_{arm}_c{c}.log", encoding="utf-8").read().splitlines()):
            if line.startswith("{") and '"status"' in line:
                st = json.loads(line); break
        statuses[lab] = {k: st.get(k) for k in ("status", "log_successful", "log_failed", "csv_valid_rows", "C1_max_abs_delta")} if st else None
        if not st or st.get("status") != "PASS":
            raise SystemExit(f"REFUSED: chunk {lab} did not PASS: {st}")
        with open(f"{SC}/{lab}/{lab}.csv", encoding="utf-8") as f:
            r = csv.DictReader(f)
            header = header or r.fieldnames
            rows += [x for x in r if x["token"] != "average"]
    toks = [x["token"] for x in rows]
    if len(toks) != len(set(toks)) or set(toks) != set(want):
        raise SystemExit(f"REFUSED: merged {len(toks)} rows / {len(set(toks))} unique vs {len(want)} registered; "
                         f"missing {len(set(want) - set(toks))}, extra {len(set(toks) - set(want))}")
    out = f"{SC}/refe_a9_{arm}"
    os.makedirs(out, exist_ok=True)
    num = [k for k in header if k not in ("token", "valid")]
    avg = {"token": "average", "valid": "True"}
    for k in num:
        try:
            avg[k] = sum(float(x[k]) for x in rows) / len(rows)
        except ValueError:
            avg[k] = ""
    with open(f"{out}/refe_a9_{arm}.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader(); w.writerows(rows); w.writerow(avg)
    json.dump({"merged_from": statuses, "rows": len(rows), "pdms_x100": 100 * avg["score"]},
              open(f"{out}/refe_a9_{arm}.status.json", "w", encoding="utf-8"), indent=1)
    print(f"ZZMERGE {arm} rows {len(rows)} PDMS {100 * avg['score']:.4f} chunks {list(statuses)}")


if __name__ == "__main__":
    main()
