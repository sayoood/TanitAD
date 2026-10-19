"""Build the deliverable of C2-GATE-SIBLING-LEAK from the evidence of the matrix: RESULT.md, the diff vs the tip, the blob table, out/ (the staged files at their REAL repo paths) and out/paths.json.
    python -B -I finalize.py <work dir (c2leak)>        stdlib only; safe to re-run (it rewrites out/ from scratch).  Missing matrix cells are printed as PENDING and the headline says so.
"""
import difflib
import hashlib
import json
import os
import shutil
import sys

work = os.path.abspath(sys.argv[1])
ev = os.path.join(work, "evidence")
out = os.path.join(work, "out")
RAW = "TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw"
CODE = f"{RAW}/2026-10-18-sel-rank/code"
PKG = f"{RAW}/2026-10-19-c2-sibling-leak"
TIP = "d2f4a062783f3fd9757ef10cc3266a13b42d8815"
EDITED = ("srttc_c2_gate.py", "test_srttc_c2_gate.py")
GATE_TEST, R1_TEST = "test_srttc_c2_gate.py", "test_srttc_r1.py"


def rb(p):
    with open(p, "rb") as f:
        return f.read()


def blob_sha1(data):
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def lf(data):
    return data.replace(b"\r\r\n", b"\n").replace(b"\r\n", b"\n")


# ----------------------------------------------------------------------------------------------------------------------------------------- matrix
res = {}
rp = os.path.join(ev, "matrix_results.jsonl")
if os.path.exists(rp):
    for ln in lf(rb(rp)).decode("utf-8").splitlines():
        if ln.strip():
            r = json.loads(ln)
            res[(r["variant"], r["file"], r["runner"], r["order"])] = r


def cell(variant, fname, runner, order):
    r = res.get((variant, fname, runner, order))
    if r is None:
        return "PENDING", None
    if "ran" not in r or r.get("rc") == "TIMEOUT":
        return f"NO RESULT (rc {r.get('rc')})", r
    bad = r["failed"] + r["errors"]
    s = f"{r['passed']}/{r['ran']} pass" + (f", {bad} FAIL" if bad else "") + (f", {r['skipped']} skipped" if r["skipped"] else "")
    return s, r


ORDERS_U = ("default", "first", "last")
ORDERS_P = ("default", "first", "last", "reversed")
rows, problems = [], []
for runner, orders in (("unittest", ORDERS_U), ("pytest", ORDERS_P)):
    for order in orders:
        b, rb_ = cell("base", GATE_TEST, runner, order)
        m, rm = cell("mine", GATE_TEST, runner, order)
        rows.append(f"| `{GATE_TEST}` | {runner} | {order} | {b}; real-run test: {(rb_ or {}).get('real_run_test', '-')} | {m}; real-run test: {(rm or {}).get('real_run_test', '-')} |")
        if rm is None or "ran" not in (rm or {}):
            problems.append(f"mine {runner} {order}: not measured")
        elif rm["failed"] or rm["errors"] or rm["ran"] != rm["listed"]:
            problems.append(f"mine {runner} {order}: {rm['not_ok']}")
        if rb_ is not None and "ran" in rb_ and (rb_["failed"] or rb_["errors"]):
            problems.append(f"BASE {runner} {order}: {rb_['not_ok']}")
for runner in ("unittest", "pytest"):
    b, rb_ = cell("base", R1_TEST, runner, "default")
    m, rm = cell("mine", R1_TEST, runner, "default")
    rows.append(f"| `{R1_TEST}` | {runner} | default | {b} | {m} |")
    if rm is None or "ran" not in (rm or {}) or rm["failed"] or rm["errors"]:
        problems.append(f"mine {runner} {R1_TEST}: not green or not measured")
nl, rnl = cell("nl", GATE_TEST, "unittest", "default")
nlb, rnlb = cell("nlbase", GATE_TEST, "unittest", "default")
nlb_txt = nlb + (f"; not ok: {rnlb['not_ok']}" if rnlb and rnlb.get("not_ok") else "")
nl_txt = nl + (f"; not ok: {rnl['not_ok']}" if rnl and rnl.get("not_ok") else "")
mine_counts = {(runner, order): cell("mine", GATE_TEST, runner, order)[0] for runner, orders in (("unittest", ORDERS_U), ("pytest", ORDERS_P)) for order in orders}
base_counts = {(runner, order): cell("base", GATE_TEST, runner, order)[0] for runner, orders in (("unittest", ORDERS_U), ("pytest", ORDERS_P)) for order in orders}
matrix_table = "| test file | runner | order | BASE (tip files) | MINE (the fix) |\n|---|---|---|---|---|\n" + "\n".join(rows)


def short(d):
    return "; ".join(f"{k[0]} {k[1]}: {v.replace(' pass', '')}" for k, v in d.items())


arm_mine = rb(os.path.join(ev, "arm_table_MINE.md")).decode("utf-8").strip() if os.path.exists(os.path.join(ev, "arm_table_MINE.md")) else "PENDING"
arm_base = rb(os.path.join(ev, "arm_table_BASEGATE.md")).decode("utf-8").strip() if os.path.exists(os.path.join(ev, "arm_table_BASEGATE.md")) else "PENDING"
tl_mine = rb(os.path.join(ev, "time_load_MINE.txt")).decode("utf-8").strip()
tl_base = rb(os.path.join(ev, "time_load_BASE.txt")).decode("utf-8").strip()
repro_base = rb(os.path.join(ev, "repro_BASE.txt")).decode("utf-8").strip()
repro_mine = rb(os.path.join(ev, "repro_MINE.txt")).decode("utf-8").strip()
trace = lf(rb(os.path.join(ev, "loader_trace_BASE.txt"))).decode("utf-8").strip()

# ----------------------------------------------------------------------------------------------------------------------------------------- blobs, pins, diff
base_dir, mine_dir = os.path.join(work, "base"), os.path.join(work, "mine")
blobs, diffs = {}, []
for f in EDITED:
    b, m = rb(os.path.join(base_dir, f)), rb(os.path.join(mine_dir, f))
    assert b"\r" not in m, f"{f}: CR in the new file"
    blobs[f] = {"before": {"git_blob": blob_sha1(b), "sha256": sha256(b)}, "after": {"git_blob": blob_sha1(m), "sha256": sha256(m)}}
    diffs.append("".join(difflib.unified_diff(b.decode("utf-8").splitlines(True), m.decode("utf-8").splitlines(True), fromfile=f"a/{CODE}/{f} (tip {TIP[:8]})", tofile=f"b/{CODE}/{f} (fix)", n=3)))
diff_text = "\n".join(diffs)
untouched = {f: sha256(rb(os.path.join(base_dir, f))) == sha256(rb(os.path.join(mine_dir, f))) for f in ("srttc_c2.py", "srttc_r1.py", "srttc_r1_gate.py", "srttc_c2_run.py", "test_srttc_r1.py", "test_srttc_r1_gate.py")}
assert all(untouched.values()), untouched
blob_md = ("# EDITED_BLOBS -- tip `%s` blob, then new blob (git blob sha1; sha256 in brackets). Pure LF both sides.\n\n| repo path (under `%s/`) | tip blob | new blob |\n|---|---|---|\n" % (TIP[:8], RAW)
           + "\n".join(f"| `2026-10-18-sel-rank/code/{f}` | `{v['before']['git_blob']}` [`{v['before']['sha256'][:16]}`] | `{v['after']['git_blob']}` [`{v['after']['sha256'][:16]}`] |" for f, v in blobs.items())
           + "\n\nNOT edited, byte-identical to the tip (sha256 checked): `srttc_c2.py`, `srttc_r1.py`, `srttc_r1_gate.py`, `srttc_c2_run.py`, `test_srttc_r1.py`, `test_srttc_r1_gate.py`.\n"
           "NEW files: every file under `%s/2026-10-19-c2-sibling-leak/`.\n" % RAW)
pins = {"what": "C2-GATE-SIBLING-LEAK: git blob and sha256 of the two edited files, before (tip %s) and after." % TIP[:8], "tip": TIP,
        "note": "CODE_SHA256_C2.json and raw/r1/f10/c2_gate_token.json describe the BEFORE bytes and are deliberately NOT rewritten (they record what ran).  The C2 token binds srttc_c2_gate.py (BOUND_FILES = LANDED_FILES + NEW_FILES): a fresh mint is required before any reuse.  The 'after' hashes of the earlier NOT-landed attempt in ../2026-10-19-epr1ub-gate-cache/ describe other bytes.",
        "files": {f"{CODE}/{f}": v for f, v in blobs.items()}}

# ----------------------------------------------------------------------------------------------------------------------------------------- RESULT.md
unexpected = ""
if problems or "PENDING" in matrix_table:
    unexpected = ("> **UNEXPECTED / INCOMPLETE -- read before trusting this file:**\n" + "".join(f"> * {p}\n" for p in problems) + ("> * some matrix cells are PENDING (the job had not finished)\n" if "PENDING" in matrix_table else "") + "\n")
mine_all_green = not problems and "PENDING" not in matrix_table
n_base = (res.get(("base", GATE_TEST, "unittest", "default")) or {}).get("ran", "?")
n_mine = (res.get(("mine", GATE_TEST, "unittest", "default")) or {}).get("ran", "?")
headline1 = (f"**Poison arm GREEN on the fix / RED on the tip gate; deleted-code_dir arm GREEN / RED; mutation: each of the 3 restored lookups turns its arm RED (table in section 4).  "
             f"`{GATE_TEST}`: tip {n_base} tests, fix {n_mine} tests (+6 new); passed/ran per runner and order -- tip: {short(base_counts)} | fix: {short(mine_counts)}.**")

tmpl = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "RESULT_TEMPLATE.md"), "rb").read().decode("utf-8")
text = (tmpl.replace("@@UNEXPECTED@@", unexpected).replace("@@HEADLINE1@@", headline1).replace("@@MATRIX@@", matrix_table).replace("@@NL@@", nl_txt).replace("@@NLBASE@@", nlb_txt).replace("@@ARM_MINE@@", arm_mine)
        .replace("@@ARM_BASE@@", arm_base).replace("@@TIME_MINE@@", tl_mine).replace("@@TIME_BASE@@", tl_base).replace("@@REPRO_BASE@@", repro_base).replace("@@REPRO_MINE@@", repro_mine)
        .replace("@@TRACE@@", trace).replace("@@TIP@@", TIP).replace("@@BLOBS@@", "\n".join(f"* `{f}`: tip blob `{v['before']['git_blob']}` -> new `{v['after']['git_blob']}` (sha256 `{v['after']['sha256']}`)" for f, v in blobs.items())))
open(os.path.join(ev, "RESULT.md"), "wb").write(text.encode("utf-8"))
open(os.path.join(ev, "TIP_SHA.txt"), "wb").write((TIP + "\n").encode("utf-8"))
open(os.path.join(ev, "EDITED_BLOBS.md"), "wb").write(blob_md.encode("utf-8"))
open(os.path.join(ev, "CODE_SHA256_SIBLINGFIX.json"), "wb").write((json.dumps(pins, indent=1) + "\n").encode("utf-8"))
open(os.path.join(ev, "c2_gate_sibling_fix.diff"), "wb").write(diff_text.encode("utf-8"))

# ----------------------------------------------------------------------------------------------------------------------------------------- out/ at the REAL repo paths
if os.path.isdir(out):
    shutil.rmtree(out)
paths = []


def put(repo_path, data):
    dst = os.path.join(out, *repo_path.split("/"))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "wb") as f:
        f.write(data)
    paths.append([repo_path, dst.replace("\\", "/"), "raw"])


for f in EDITED:
    put(f"{CODE}/{f}", rb(os.path.join(mine_dir, f)))
for name in ("RESULT.md", "TIP_SHA.txt", "EDITED_BLOBS.md", "CODE_SHA256_SIBLINGFIX.json", "c2_gate_sibling_fix.diff"):
    put(f"{PKG}/{name}", rb(os.path.join(ev, name)))
codedir = os.path.join(work, "code")
for name in ("run_order.py", "run_matrix.py", "finalize.py", "RESULT_TEMPLATE.md", "arm_table.py", "repro_leak.py", "time_load.py"):
    put(f"{PKG}/code/{name}", lf(rb(os.path.join(codedir, name))))
for name in sorted(os.listdir(ev)):
    p = os.path.join(ev, name)
    if name in ("RESULT.md", "TIP_SHA.txt", "EDITED_BLOBS.md", "CODE_SHA256_SIBLINGFIX.json", "c2_gate_sibling_fix.diff", "matrix_cmd_pid.txt") or name.endswith(".err") and os.path.getsize(p) == 0:
        continue
    if os.path.isfile(p):
        put(f"{PKG}/raw/{name}", lf(rb(p)))
    elif name == "logs":
        for ln in sorted(os.listdir(p)):
            put(f"{PKG}/raw/logs/{ln}", lf(rb(os.path.join(p, ln))))
with open(os.path.join(out, "paths.json"), "w") as f:
    json.dump(paths, f, indent=1)
total = sum(os.path.getsize(p[1]) for p in paths)
print(f"out/: {len(paths)} files, {total/1e6:.2f} MB; problems: {problems or 'none'}; PENDING cells: {'yes' if 'PENDING' in matrix_table else 'no'}")
