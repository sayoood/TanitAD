"""Per-suite table pristine vs dagger from the junit XMLs, every test that differs, and the tests that were red on the partial di/tree (di/out_run2/p3_*.out).

  python compare.py [tag]
"""
import json
import re
import sys
import xml.etree.ElementTree as ET

RT = "E:/Projects/TanitAD-artifacts/redtriage"
OLD = "E:/Projects/TanitAD-artifacts/di/out_run2"
SUITES = ["test_v8f_oacc", "test_v8f_lora", "test_v8f_lora_ar2", "test_v8f_speedavg", "test_v8f_kindx_s1", "test_v8f_kscore"]
tag = sys.argv[1] if len(sys.argv) > 1 else ""


def load(tree, suite):
    p = f"{RT}/out/{tag}{tree}__{suite}.xml"
    try:
        root = ET.parse(p).getroot()
    except (OSError, ET.ParseError):
        return None
    res = {}
    for tc in root.iter("testcase"):
        key = f"{tc.get('classname', '').split('.')[-1]}::{tc.get('name')}"
        st, msg = "passed", ""
        for ch in tc:
            if ch.tag in ("failure", "error", "skipped"):
                st, msg = {"failure": "failed", "error": "error", "skipped": "skipped"}[ch.tag], (ch.get("message") or "")[:160]
                break
        res[key] = (st, msg)
    return res


def old_failed(suite):
    try:
        txt = open(f"{OLD}/p3_{suite}.out", encoding="utf-8", errors="replace").read()
    except OSError:
        return set()
    return {re.sub(r"^(FAILED|ERROR) stack\\tests\\[^:]+\.py::", "", ln.split(" - ")[0]).strip() for ln in txt.splitlines() if ln.startswith(("FAILED", "ERROR"))}


def cnt(r):
    c = {"passed": 0, "failed": 0, "error": 0, "skipped": 0}
    for st, _ in r.values():
        c[st] += 1
    return c


rows, diffs = [], []
for s in SUITES:
    p, d = load("pristine", s), load("dagger", s)
    rows.append((s, p and cnt(p), d and cnt(d)))
    if p is None or d is None:
        continue
    for k in sorted(set(p) | set(d)):
        a, b = p.get(k, ("(absent)", "")), d.get(k, ("(absent)", ""))
        if a[0] != b[0]:
            diffs.append((s, k, a, b))
        elif a[0] in ("failed", "error") and a[1] != b[1]:
            diffs.append((s, k + "  [same status, DIFFERENT message]", a, b))
fmt = lambda c: "n/a" if c is None else f"{c['passed']} / {c['failed']} / {c['error']}" + (f" (+{c['skipped']} skipped)" if c["skipped"] else "")   # noqa: E731
print(f"{'suite':22s} {'pristine p/f/e':28s} {'dagger p/f/e':28s}")
for s, a, b in rows:
    print(f"{s:22s} {fmt(a):28s} {fmt(b):28s}")
print()
print("tests whose outcome differs pristine -> dagger:", len(diffs))
for s, k, a, b in diffs:
    print(f"  {s}::{k}\n      pristine {a[0]}: {a[1]}\n      dagger   {b[0]}: {b[1]}")
print()
for s in SUITES:
    p, d = load("pristine", s), load("dagger", s)
    if p is None:
        continue
    redp = {k.split("::", 1)[1] for k, (st, _) in p.items() if st in ("failed", "error")}
    o = old_failed(s)
    print(f"{s}: red on pristine {len(redp)}; red on the partial di/tree (old p3) {len(o)}; red on partial tree AND pristine {len(o & redp)}")
