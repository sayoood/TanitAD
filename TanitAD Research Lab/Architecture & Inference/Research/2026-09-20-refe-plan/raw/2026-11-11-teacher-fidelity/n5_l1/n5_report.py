"""n5_report.py - summarise an n5_run.py output: per arm counts, certificate violations, statuses, per-token cause table. Pure python (no navsim).
usage: python n5_report.py OUT.jsonl [--title T] [--json SUMMARY.json] [--table]"""
import argparse
import collections
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--title", default="")
    ap.add_argument("--json", default=None)
    ap.add_argument("--table", action="store_true")
    a = ap.parse_args()
    R = [json.loads(x) for x in open(a.out, encoding="utf-8") if x.strip().endswith("}")]
    err = [r for r in R if "error" in r]
    R = [r for r in R if "error" not in r]
    fail0 = [r for r in R if r.get("base_dac0")]
    arms = sorted({k for r in fail0 for k in r["arms"]})
    S = {"title": a.title, "rows": len(R), "error_rows": len(err), "base_dac0": len(fail0)}
    for arm in arms:
        rs = [r["arms"][arm] for r in fail0 if arm in r["arms"]]
        S[arm] = {"repaired": sum(1 for x in rs if x["applied"]), "n": len(rs), "cert_violations": sum(1 for x in rs if x.get("cert_violation")),
                  "statuses": dict(collections.Counter(x["status"] for x in rs)),
                  "mean_labels": round(sum(x["n_labels"] for x in rs) / max(len(rs), 1), 2),
                  "infeasible_flagged": sum(1 for x in rs if x["status"] == "INFEASIBLE"),
                  "relaxed_margin_used": sum(1 for x in rs if any(e.get("relaxed_steps") for e in x.get("iters", [])))}
    if "A1" in S and "L1" in S:
        both = [(r["arms"]["A1"]["applied"], r["arms"]["L1"]["applied"]) for r in fail0]
        S["l1_only"] = sum(1 for p, q in both if q and not p)
        S["a1_only"] = sum(1 for p, q in both if p and not q)
        S["both"] = sum(1 for p, q in both if p and q)
        S["neither"] = sum(1 for p, q in both if not p and not q)
    print(json.dumps(S))
    if a.json:
        json.dump(S, open(a.json, "w", encoding="utf-8"), indent=1)
    if a.table:
        print("| id | A1 | A1 extent seq (m) | A1 lat flips | L1 | L1 status | L1 iters | L1 flips | tight width (m) | infeasible steps |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for r in fail0:
            x, y = r["arms"].get("A1", {}), r["arms"].get("L1", {})
            it = y.get("iters", [])
            print("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                r["id"], "yes" if x.get("applied") else "NO", ",".join("%.3f" % v for v in x.get("extent_seq_m", [])), x.get("lat_sign_flips", ""),
                "yes" if y.get("applied") else "NO", y.get("status", ""), len(it), y.get("n_flips", ""),
                ("%.2f" % min(e["interval_width_min_m"] for e in it if "interval_width_min_m" in e)) if any("interval_width_min_m" in e for e in it) else "-",
                max((e.get("n_infeasible_steps", 0) for e in it), default=0)))


if __name__ == "__main__":
    main()
