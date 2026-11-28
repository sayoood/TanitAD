"""A3.3 proof comparison, run on the pod: the new gates / tb0r outputs in /workspace/data/p5c_a33_check against run 3's (read-only).  Prints a compact verdict."""
import json
import os
import sys

R3 = "/workspace/data/p5c_label_a32"
CHK = "/workspace/data/p5c_a33_check"
L = lambda p: json.load(open(p, encoding="utf-8"))


def scrub(o, drop=("merged_jsonl", "inputs", "bank_files")):
    return {k: v for k, v in o.items() if k not in drop}


def cmp_gates(tag, new, old, only_ids=None):
    n, o = L(new), L(old)
    out = []
    for gid in sorted(set(n["gates"]) | set(o["gates"])):
        a, b = n["gates"].get(gid), o["gates"].get(gid)
        same = json.dumps(a, sort_keys=True, default=str) == json.dumps(b, sort_keys=True, default=str)
        out.append((gid, a["status"] if a else None, b["status"] if b else None, "IDENTICAL" if same else "DIFFERS"))
    rp = json.dumps(n["rank_policy"], sort_keys=True, default=str) == json.dumps(o["rank_policy"], sort_keys=True, default=str)
    print(f"== {tag}: new {new.replace(CHK + '/', '')} vs run 3 {old.replace(R3 + '/', '')}")
    for r in out:
        print("   %-5s new=%-8s run3=%-8s %s" % r)
    print("   rank_policy", "IDENTICAL" if rp else "DIFFERS", "| failing new:", n["failing"], "run3:", o["failing"], "| label_records", n["label_records"], o["label_records"])
    return n, o


def dd(a, b, path=""):
    """leaf-level differences of two JSON values"""
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a:
                yield (path + "/" + k, "<absent>", b[k])
            elif k not in b:
                yield (path + "/" + k, a[k], "<absent>")
            else:
                yield from dd(a[k], b[k], path + "/" + k)
    elif a != b:
        yield (path, a, b)


step = sys.argv[1]
if step == "control":
    old_run3 = R3 + "/gates/control/gates.json"
    n, o = cmp_gates("control NEW (tb0-tol 1e-8, ranks all)", CHK + "/gates_control_new/gates.json", old_run3)
    t0 = n["gates"]["TB-0"]
    d = t0["detail"]["tc20s_warm_start_vs_dev_build"]
    print("   TB-0 NEW: status", t0["status"], "| tc20s part", d["status"], "value", d["value"], "n", d["n"], "tol", d["tol"], "| bar:", t0["bar"][-60:])
    n2, o2 = cmp_gates("control DEFAULT-ARGS (tol 1e-9, ranks 0) -> must reproduce run 3", CHK + "/gates_control_old/gates.json", old_run3)
    print("   defaults identical to run 3 (every gate + rank_policy + failing):", all(g[3] == "IDENTICAL" for g in [(k, 0, 0, "IDENTICAL" if json.dumps(n2["gates"][k], sort_keys=True, default=str) == json.dumps(o2["gates"][k], sort_keys=True, default=str) else "D") for k in n2["gates"]]))
elif step == "main":
    old_run3 = R3 + "/gates/main/gates.json"
    n, o = cmp_gates("main NEW (fallback + allow-list)", CHK + "/gates_main_new/gates.json", old_run3)
    t5 = n["gates"]["TB-5"]
    print("   TB-5 NEW: status", t5["status"], "value", t5["value"], "n", t5["n"], "| detail.label_fallback:", json.dumps(t5["detail"].get("label_fallback"), default=str)[:500])
    print("   TB-5 run3: status", o["gates"]["TB-5"]["status"], "value", o["gates"]["TB-5"]["value"], "| S1_S2_S5_not_true", o["gates"]["TB-5"]["detail"]["S1_S2_S5_not_true"])
    for gid in ("TB-2", "TB-6", "TB-7", "TB-4"):
        diffs = list(dd(n["gates"][gid], o["gates"][gid]))
        print("   %s leaf differences new vs run3: %d" % (gid, len(diffs)), json.dumps(diffs[:6], default=str)[:700])
    n2, o2 = cmp_gates("main DEFAULT-ARGS (no fallback, original rekeyed labels) -> must reproduce run 3", CHK + "/gates_main_old/gates.json", old_run3)
    print("   main defaults identical to run 3 (all 8 gates):", all(json.dumps(n2["gates"][k], sort_keys=True, default=str) == json.dumps(o2["gates"][k], sort_keys=True, default=str) for k in n2["gates"]),
          "| rank_policy identical:", json.dumps(n2["rank_policy"], sort_keys=True, default=str) == json.dumps(o2["rank_policy"], sort_keys=True, default=str))
    m_new, m_old, m_r3 = L(CHK + "/gates_main_new/table_meta_fields.json"), L(CHK + "/gates_main_old/table_meta_fields.json"), L(R3 + "/gates/main/table_meta_fields.json")
    print("   meta keys new-only:", sorted(set(m_new) - set(m_r3)), "| meta(old-args) vs run3 differing leaves:", [d[0] for d in dd(m_old, m_r3)][:8])
    print("   meta p5r_label_fallback:", json.dumps(m_new.get("p5r_label_fallback"), default=str)[:400])
    # the merged table file: only the ruled row's label may differ
    def rows(p):
        return {(r["log_name"], r["token"], r["rank"]): r for r in (json.loads(x) for x in open(p, encoding="utf-8") if x.strip())}
    a, b = rows(CHK + "/gates_main_new/merged_labels_rank0only.jsonl"), rows(R3 + "/gates/main/merged_labels_rank0only.jsonl")
    ch = [k for k in a if a[k] != b[k]]
    print("   merged_labels_rank0only: rows", len(a), len(b), "| rows that differ new vs run3:", len(ch), [(k[1], k[2], a[k]["source"]) for k in ch][:3])
    a_old = rows(CHK + "/gates_main_old/merged_labels_rank0only.jsonl") if os.path.exists(CHK + "/gates_main_old/merged_labels_rank0only.jsonl") else None
    print("   merged (defaults) written:", a_old is not None, "| identical to run 3:", a_old == b if a_old is not None else "n/a (TB-5 FAIL gates still write? see file)")
elif step == "tb0r":
    n, o = L(CHK + "/tb0r_new.json"), L(R3 + "/tb0r.json")
    d = L(CHK + "/tb0r_old.json")
    print("== TB-0r NEW (tol 1e-8):", n["status"], "| tc20s status", n["tc20s"]["status"], "max", n["tc20s"]["value"], "n", n["tc20s"]["n"], "| p5r ok", n["p5r"]["ok"], "| coverage_ok", n["coverage_ok"])
    print("== TB-0r DEFAULT (tol 1e-9):", d["status"], "max", d["tc20s"]["value"], "| run 3:", o["status"], "max", o["tc20s"]["value"])
    ign = {"bar", "gates_module_sha256"}
    print("   defaults identical to run 3 apart from bar/gates-module sha:", {k: v for k, v in d.items() if k not in ign} == {k: v for k, v in o.items() if k not in ign})
    print("   NEW vs run 3 differing leaves:", [x[0] for x in dd({k: v for k, v in n.items() if k not in ign}, {k: v for k, v in o.items() if k not in ign})][:12])
