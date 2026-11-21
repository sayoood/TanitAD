#!/usr/bin/env python3
"""post_queue.py : wait for the S4R-1 final test queue (C:/so/s4r_logs/qfinal/QUEUE_DONE), re-run every FAILING test once on the BASELINE tree (the pristine tip + the SO-A16 test fix only, C:/so/s4r) with the same
runner, then write TEST_TABLE.md, fill S4R_1_RESULT.md between its markers, copy the logs into the record directory and regenerate C:/Users/Admin/s4r_pkg/paths.json from everything under s4r_pkg/files.
Verdicts: PASS (rc 0, a PASS token, 0 skips) | PRE-EXISTING (fails on the baseline tree too) | REGRESSION (passes on the baseline, fails patched) | NEW-TEST-FAILS (no baseline counterpart) | ABSENT.
Runs nothing else; kills nothing."""
import json, os, re, shutil, subprocess, sys, time

LOGS = "C:/so/s4r_logs"
Q, QB = LOGS + "/qfinal", LOGS + "/qbase"
PKR = "TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
BASE_PK = "C:/so/s4r/" + PKR
PKG = "C:/Users/Admin/s4r_pkg"
REC = f"{PKG}/files/{PKR}/raw/2026-10-17-s4r-integration"
PY = "C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe"

while not os.path.exists(Q + "/QUEUE_DONE"):
    time.sleep(30)


def parse(summary):
    rows = {}
    if os.path.exists(summary):
        for ln in open(summary, encoding="utf-8", errors="replace"):
            m = re.match(r"(\S+)\s+rc=(\S+) secs=(\d+) ram_wait=(\d+)s skips=(\d+) tok=\[(.*?)\] last=\[(.*)\]$", ln.rstrip("\n"))
            if m:
                rows[m.group(1)] = dict(rc=m.group(2), secs=int(m.group(3)), skips=int(m.group(5)), tok=m.group(6), last=m.group(7))
    return rows


def good(r):
    return r["rc"] == "0" and "PASS" in r["tok"] and r["skips"] == 0


jobs = json.load(open("C:/so/s4r_tools/queue_final.json", encoding="utf-8"))
final = parse(Q + "/SUMMARY.txt")
bad = [j for j in jobs if j["id"] not in final or not good(final[j["id"]])]
bjobs = []
for j in bad:
    script = j["cmd"][0]
    if os.path.isfile(f"{BASE_PK}/{script}"):
        bjobs.append(dict(j, tree=BASE_PK))
if bjobs:
    json.dump(bjobs, open("C:/so/s4r_tools/queue_base.json", "w"), indent=1)
    os.makedirs(QB, exist_ok=True)
    subprocess.run([PY, "-X", "utf8", "C:/so/s4r_tools/queue.py", "C:/so/s4r_tools/queue_base.json", QB])
base = parse(QB + "/SUMMARY.txt")

rows = ["| test (queue id) | patched: rc / token / s | baseline (pristine tip + SO-A16), if it failed | verdict |", "|---|---|---|---|"]
nbad = 0
for j in jobs:
    i = j["id"]
    r = final.get(i)
    if r is None:
        rows.append(f"| {i} | NOT RUN | | ABSENT |")
        nbad += 1
        continue
    if good(r):
        v, b = "**PASS**", ""
    else:
        br = base.get(i)
        if br is None:
            v, b = "**NEW-TEST-FAILS (no baseline counterpart)**", ""
        elif good(br):
            v, b = "**REGRESSION**", f"rc {br['rc']} / `{br['tok'][:40]}` / {br['secs']} s"
        else:
            v, b = "**PRE-EXISTING (fails on the baseline too)**", f"rc {br['rc']} / {br['last'][:90]}"
        nbad += 1
    rows.append(f"| {i} | rc {r['rc']} / `{r['tok'][:46]}` / {r['secs']} s / skips {r['skips']} | {b} | {v} |")
table = "\n".join(rows)
hdr = (f"**Generated {time.strftime('%Y-%m-%d %H:%M')} by `tools/post_queue.py`: {len(jobs)} runs, {nbad} not clean.** Raw: `logs/qfinal/SUMMARY.txt` (patched, one line per run), `logs/qbase/SUMMARY.txt` (baseline re-run of each failure). "
       "Each test ran ALONE, at below-normal priority, CUDA not hidden, no bash shim except the row named `_shim`. Tree = the 2026-10-17 final set on a `git archive` of tip `3b8c39f`.\n\n")
open(f"{REC}/TEST_TABLE.md", "w", encoding="utf-8", newline="\n").write(hdr + table + "\n")
res = open(f"{REC}/S4R_1_RESULT.md", encoding="utf-8").read()
res = re.sub(r"<!--TEST_TABLE_BEGIN-->.*?<!--TEST_TABLE_END-->", lambda m: "<!--TEST_TABLE_BEGIN-->\n" + hdr + table + "\n<!--TEST_TABLE_END-->", res, flags=re.S)
open(f"{REC}/S4R_1_RESULT.md", "w", encoding="utf-8", newline="\n").write(res)

# the after-fix depth-2 reproduction (12 s), now that nothing else is running: the SAME child that failed rc 1 before the fix must pass
os.makedirs(f"{REC}/logs/repro", exist_ok=True)
rr = subprocess.run([PY, "-X", "utf8", "C:/so/s4r_tools/emul_child.py", f"C:/so/s4r_p/{PKR}/refe", "test_sft4r_gate.py", "--outer"], capture_output=True, text=True, encoding="utf-8", errors="replace")
open(f"{REC}/logs/repro/emul_depth2_after_fix.log", "w", encoding="utf-8", newline="\n").write(rr.stdout + rr.stderr)
for src, dst in ((LOGS + "/emul2.log", "emul2_depth2_before_fix.log"), (LOGS + "/emul1.log", "emul1_depth1.log"), (LOGS + "/sg_alone.log", "sg_alone_bare_bash.log")):
    if os.path.isfile(src):
        shutil.copyfile(src, f"{REC}/logs/repro/{dst}")
os.makedirs(f"{REC}/logs/e2e", exist_ok=True)
for src, dst in ((LOGS + "/e2e_clean.log", "suite_prerun_clean.log"), (LOGS + "/e2e_outer.log", "suite_prerun_outer_depth1.log"), (LOGS + "/e2e1.log", "suite_prerun_shim_rebased_pre_fix.log"),
                 ("C:/so/s4r_work/lg_noshim.log", "test_launch_gate_full_noshim_tip_plus_SO-A16.log")):
    if os.path.isfile(src):
        shutil.copyfile(src, f"{REC}/logs/e2e/{dst}")
shutil.copyfile("C:/so/s4r_tools/post_queue.py", f"{REC}/tools/post_queue.py")

# logs into the record
for src, dst in ((Q, "qfinal"), (QB, "qbase"), (LOGS + "/qfinal_attempt1", "qfinal_attempt1")):
    if os.path.isdir(src):
        os.makedirs(f"{REC}/logs/{dst}", exist_ok=True)
        for f in os.listdir(src):
            p = f"{src}/{f}"
            if os.path.isfile(p) and f != "QUEUE_DONE" and os.path.getsize(p) < 2_000_000:
                shutil.copyfile(p, f"{REC}/logs/{dst}/{f}")

# paths.json from everything under files/
base_dir = f"{PKG}/files/"
items = []
for root, _d, fs in os.walk(base_dir):
    for f in sorted(fs):
        a = os.path.join(root, f).replace(chr(92), "/")
        items.append([a[len(base_dir):], a, "raw"])
items.sort()
json.dump(items, open(f"{PKG}/paths.json", "w", encoding="utf-8"), indent=1)
open(f"{PKG}/POST_QUEUE_DONE", "w").write(f"{time.strftime('%Y-%m-%d %H:%M')} runs={len(jobs)} not_clean={nbad} files={len(items)}\n")
