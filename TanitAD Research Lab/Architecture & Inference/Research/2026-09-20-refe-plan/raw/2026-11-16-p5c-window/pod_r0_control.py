"""MM 2026-11-16: A3 section 7 pod R0 control read. Per log: pod R0 traj vs bank traj (dxy, bar 1e-6 m), CUDA-built bank rows
(teacher_device None) split out. Outcome A = every CPU-class row exact on the pod; B = some logs diverge. Neither blocks C1 (A3 s7)."""
import collections, glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rj5_bank_join as J
R = "F:/tanitad_scratch/rj5nt"
OUT = sys.argv[1]
bank, _, _ = J.load_bank(R + "/bank_p7_rank0_slim.jsonl.gz")
dev = {(r["log_name"], r["token"]): str(r.get("teacher_device")) for r in J.read_jsonl(R + "/bank_p7_rank0_slim.jsonl.gz")}
devbox = {d["log"]: "diverged" for d in json.load(open("E:/Projects/TanitAD-artifacts/scorer_gym_20261107/rj5path/control_by_log_bulk.json"))["diverged"]}
for d in json.load(open("E:/Projects/TanitAD-artifacts/scorer_gym_20261107/rj5path/control_by_log_bulk.json"))["exact"]:
    devbox.setdefault(d["log"], "exact")
res = {}
for name, d in (("control36", R + "/pod_c/run_control36/rows"), ("chainC_inline", R + "/pod_c/run_fetch_C/rows")):
    per = collections.defaultdict(lambda: {"n": 0, "over": 0, "max_dxy_m": 0.0, "not_in_bank": 0})
    files = sorted(glob.glob(d + "/*.jsonl"))
    for fp in files:
        for r in J.read_jsonl(fp):
            if r.get("arm") != "R0" or "traj" not in r:
                continue
            k = (r["log_name"], r["token"])
            if k not in bank:
                per[(r["log_name"], "NOT_IN_BANK")]["not_in_bank"] += 1
                continue
            cls = "cuda_built" if dev[k] == "None" else "cpu_built"
            dx = J.dxy(J.flat(r["traj"]), bank[k][0])
            p = per[(r["log_name"], cls)]
            p["n"] += 1; p["over"] += dx > 1e-6; p["max_dxy_m"] = max(p["max_dxy_m"], dx)
    rows = [{"log": k[0], "class": k[1], **v, "devbox_bulk_control": devbox.get(k[0], "not_in_devbox_control")} for k, v in sorted(per.items())]
    cpu = [x for x in rows if x["class"] == "cpu_built"]
    res[name] = {"files": len(files), "n_logs": len({x["log"] for x in rows}),
                 "cpu_built": {"logs": len(cpu), "logs_exact": sum(1 for x in cpu if x["over"] == 0), "logs_diverged": sum(1 for x in cpu if x["over"] > 0),
                               "rows": sum(x["n"] for x in cpu), "rows_over_1e-6": sum(x["over"] for x in cpu), "max_dxy_m": max([x["max_dxy_m"] for x in cpu] or [0.0])},
                 "cuda_built": {"logs": sum(1 for x in rows if x["class"] == "cuda_built"), "rows": sum(x["n"] for x in rows if x["class"] == "cuda_built"),
                                "rows_over_1e-6": sum(x["over"] for x in rows if x["class"] == "cuda_built")},
                 "not_in_bank_rows": sum(x["not_in_bank"] for x in rows),
                 "devbox_diverged_logs_exact_on_pod": sum(1 for x in cpu if x["over"] == 0 and x["devbox_bulk_control"] == "diverged"),
                 "devbox_diverged_logs_diverged_on_pod": sum(1 for x in cpu if x["over"] > 0 and x["devbox_bulk_control"] == "diverged"),
                 "per_log": rows}
c = res["control36"]["cpu_built"]
res["outcome"] = "A (all CPU-class rows exact on the pod)" if c["rows"] > 0 and c["rows_over_1e-6"] == 0 else "B (some logs diverge on the pod too)"
json.dump(res, open(OUT, "w"), indent=1)
for name in ("control36", "chainC_inline"):
    print(name, {k: v for k, v in res[name].items() if k != "per_log"})
print("OUTCOME", res["outcome"])
