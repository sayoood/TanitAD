"""finalize.py LEVER OUTJSON SPEC DEST -- turn the driver output into MUTATIONS_<LEVER>.json (counts first, then control runs, then one record per mutation)."""
import json
import sys

lever, outp, specp, dest = sys.argv[1:5]
o = json.load(open(outp))
spec = json.load(open(specp))
ms = o["mutations"]
order = {m["id"]: i for i, m in enumerate(spec["mutations"])}
ms.sort(key=lambda m: order.get(m["id"], 99))
n_red = sum(1 for m in ms if m["observed"] == "RED")
n_green = sum(1 for m in ms if m["observed"] == "STILL-GREEN")
n_not_run = sum(1 for m in ms if m["observed"] not in ("RED", "STILL-GREEN"))
ctl = o.get("control", {})
cta = o.get("control_after", {})
res = {
    "lever": lever,
    "counts": {"n_mutations": len(ms), "n_red": n_red, "n_still_green": n_green, "n_other": n_not_run,
               "control_green": bool(ctl.get("green")), "control_after_green": bool(cta.get("green")) if cta else None},
    "counts_line": "n %d | red %d | still-green %d | other %d | control green %s | control after green %s" % (
        len(ms), n_red, n_green, n_not_run, "yes" if ctl.get("green") else "no", ("yes" if cta.get("green") else "no") if cta else "not run"),
    "checks": spec["test_files"], "written_utc": o.get("written_utc"), "control": ctl, "control_after": cta,
    "mutations": [{"id": m["id"], "file": m["file"], "original_line": m["old"], "mutated_line": m["new"], "expected_red": m["expect"], "observed": m["observed"],
                   "rc": m.get("rc"), "failed": m.get("failed", []), "expected_all_red": m.get("expected_all_red"), "why": m.get("why", "")} for m in ms],
}
json.dump(res, open(dest, "w"), indent=1)
print(res["counts_line"])
for m in res["mutations"]:
    if m["observed"] != "RED" or not m["expected_all_red"]:
        print("CHECK", m["id"], m["observed"], m["expected_all_red"])
