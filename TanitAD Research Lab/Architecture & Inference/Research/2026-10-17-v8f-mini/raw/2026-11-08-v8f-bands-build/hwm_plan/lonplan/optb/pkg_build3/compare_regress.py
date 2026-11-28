"""Per-test base-vs-new comparison from regress_pertest.json. A regression = fails (or errors) on new and passes on base. Writes regress_base.tsv and regress_compare.json; prints the verdict."""
import json
O = "F:/TanitAD-scratch/actrsp/optb"
d = json.load(open(O + "/regress_pertest.json"))
files = sorted({k.split("|", 1)[1] for k in d})
rows, reg, summary = [], [], {}
for f in files:
    n, b = d.get("new|" + f), d.get("base|" + f)
    if n is None or b is None:
        rows.append(f"{f}\tINCOMPLETE"); continue
    nt, bt = n["t"], b["t"]
    cnt = lambda t: {s: sum(1 for v in t.values() if v[0] == s) for s in ("pass", "fail", "skip")}  # noqa: E731
    r = sorted(k for k, v in nt.items() if v[0] == "fail" and bt.get(k, ["missing"])[0] == "pass")
    only_new_fail = sorted(k for k, v in nt.items() if v[0] == "fail" and k not in bt)
    both_fail_diff = sorted(set(k for k, v in nt.items() if v[0] == "fail") ^ set(k for k, v in bt.items() if v[0] == "fail"))
    rows.append(f"{f}\tnew rc={n['rc']} {cnt(nt)}\tbase rc={b['rc']} {cnt(bt)}\tregressions={len(r)}\tnew-only-ids-failing={len(only_new_fail)}\tfail-set-differs={len(both_fail_diff)}")
    summary[f] = {"new": cnt(nt), "base": cnt(bt), "regressions": r, "new_ids_failing": only_new_fail, "fail_set_xor": both_fail_diff,
                  "msgs": {k: nt[k][1] for k in r + only_new_fail}}
    reg += [f + "::" + k for k in r + only_new_fail]
open(O + "/regress_base.tsv", "w", encoding="utf-8").write("\n".join(rows) + "\n")
json.dump({"regressions": reg, "files": summary}, open(O + "/regress_compare.json", "w"), indent=1)
print("files", len(files), "regressions (fail on new, pass or absent on base):", len(reg))
for x in reg[:20]:
    print(" ", x)
