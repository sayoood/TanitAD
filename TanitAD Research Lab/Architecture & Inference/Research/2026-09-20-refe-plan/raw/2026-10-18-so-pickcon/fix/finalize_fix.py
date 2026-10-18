#!/usr/bin/env python3
"""SO-PICKCON regression-fix finalizer (run ONCE when BOTH fix lanes are done; the watcher does it).  Idempotent: it rewrites only its own blocks.
Reads  base_summary2.json + patched_summary.json  (the first pass)  and  base_summary_fix.json + patched_summary_fix.json  (the fix pass),
merges them (a test re-run in the fix pass REPLACES its first-pass row), writes the "Regression fix" table into TEST_RESULTS.md and the <!--FIX--> block of RESULT.md,
copies the fix-pass logs, the three PATCH diffs of the fixed files and the helper scripts under out/, updates BASE_BLOBS.json / test_results.json / paths.json.
It does NOT re-run finalize.py (that would overwrite TEST_RESULTS.md with the first-pass table).  Never commits, never pushes."""
import difflib, hashlib, json, os, re, shutil, subprocess, sys, time

ROOT = "C:/Users/Admin/pickcon"
PKREL = "TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
OUT = f"{ROOT}/out"
OUTPK = f"{OUT}/{PKREL}/raw/2026-10-18-so-pickcon"
FIXED = [f"{PKREL}/eval/safeobj_outcome.py", f"{PKREL}/eval/test_safeobj_outcome.py", f"{PKREL}/refe/test_dump_scores.py"]
MIRROR, TIP = "C:/Users/Admin/tanitad-push", "0a5958a"
EXPECT_TOKEN = {"test_safeobj_outcome.py": "ZZTEST_PASS 136", "test_rl_pickcon.py": "ZZTEST_PASS 20"}      # outcome: 133 (baseline) + 3 new RED mutations; pickcon: the committed 20
RERUN_FIX = ("test_dump_scores.py", "test_rl_pickcon.py", "test_launch_gate.py_static_no-banked", "test_rl_safeobj.py", "test_safeobj_outcome.py")


def sh(*a):
    return subprocess.run(a, capture_output=True, text=True, encoding="utf-8").stdout.strip()


def blob_of(path):
    b = open(path, "rb").read()
    return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def load(p):
    return json.load(open(f"{ROOT}/{p}")) if os.path.exists(f"{ROOT}/{p}") else {}


def tok(r):
    if not r:
        return "(not run)"
    return r["tokens"][-1] if r["tokens"] else (r["tail"][-1][:90] if r["tail"] else "(no token)")


def norm_tail(r):
    return re.sub(r"\d+(\.\d+)?", "#", " ".join(r["tail"][-1:])) if r else ""


def main():
    ob, op, fb, fp = (load(x) for x in ("base_summary2.json", "patched_summary.json", "base_summary_fix.json", "patched_summary_fix.json"))
    done = bool(fb.get("_done")) and bool(fp.get("_done"))
    rows, regress, same_fail, fixed, tokbad = [], [], [], [], []
    for name in sorted((set(ob) | set(op) | set(fb) | set(fp)) - {"_done"}):
        b = fb.get(name, ob.get(name))
        p = fp.get(name, op.get(name))
        p0 = op.get(name)
        rerun = name in fp
        if b and p and b["rc"] == 0 and p["rc"] != 0:
            v = "REGRESSION"
            regress.append(name)
        elif b and p and b["rc"] != 0 and p["rc"] != 0:
            v = "FAILS AT BASELINE TOO (environmental)" if (norm_tail(b) == norm_tail(p)) else "FAILS AT BASELINE AND PATCHED, DIFFERENT TAIL"
            same_fail.append(name)
        elif b and p and b["rc"] == 0 and p["rc"] == 0:
            v = "PASS = PASS"
            if p0 and p0["rc"] != 0:
                v = "FIXED (first pass REGRESSION) -> PASS = PASS"
                fixed.append(name)
        elif p and not b:
            v = "NEW TEST" if p["rc"] == 0 else "NEW TEST FAILS"
            if p["rc"] != 0:
                regress.append(name)
        else:
            v = "incomplete"
        if p and p["rc"] == 0 and rerun:
            want = EXPECT_TOKEN.get(name) or (tok(b).split("  ")[0] if b else None)
            if want and tok(p).split("  ")[0] != want:
                tokbad.append((name, tok(p), want))
        rows.append((name, b["rc"] if b else None, tok(b), p0["rc"] if p0 else None, p["rc"] if p else None, tok(p), p["secs"] if p else None, rerun, v))
    head = (f"Fix pass complete: {done}. Regressions after the fix: {regress or 'none'}. Fixed: {fixed or 'none'}. Failing at baseline too (environmental): {same_fail or 'none'}. "
            f"Token mismatches on re-run files: {tokbad or 'none'}.")
    md = ["| test file (full form) | re-run in fix pass | baseline rc | baseline token | patched rc (first pass) | patched rc (fix pass / final) | patched token (final) | patched s | verdict |", "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| `{r[0]}` | {'yes' if r[7] else 'no (first-pass row stands)'} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} | {r[6]} | **{r[8]}** |")
    table = "\n".join(md)
    # ---- TEST_RESULTS.md: replace/append the fix section
    tp = f"{OUTPK}/TEST_RESULTS.md"
    t = open(tp, encoding="utf-8").read()
    t = re.sub(r"\n<!--FIXPASS-->.*?<!--/FIXPASS-->\n?", "\n", t, flags=re.S)
    t += f"\n<!--FIXPASS-->\n## Regression fix, second pass ({time.strftime('%Y-%m-%d')}) -- supersedes the two REGRESSION rows above\n\n{head}\n\n{table}\n<!--/FIXPASS-->\n"
    open(tp, "w", encoding="utf-8", newline="\n").write(t)
    # ---- the tested copies of the fixed files must be byte-identical to what the lane ran (patched_full)
    blobs = json.load(open(f"{OUTPK}/BASE_BLOBS.json"))
    same = {}
    for rel in FIXED:
        a, b_ = f"{OUT}/{rel}", f"{ROOT}/patched_full/{rel}"
        same[rel] = sha256(a) == sha256(b_)
        base = sh("git", "-C", MIRROR, "rev-parse", f"{TIP}:{rel}")
        blobs[rel] = {"base_blob_at_0a5958a": base if len(base) == 40 else "(new file)", "new_blob": blob_of(a), "sha256_staged": sha256(a)}
        old = sh("git", "-C", MIRROR, "-c", "core.autocrlf=false", "show", f"{TIP}:{rel}").splitlines()
        new = open(a, encoding="utf-8", newline="").read().splitlines()
        open(f"{OUTPK}/PATCH_{os.path.basename(rel)}.diff", "w", encoding="utf-8", newline="\n").write("\n".join(difflib.unified_diff(old, new, f"a/{rel}", f"b/{rel}", lineterm="", n=2)) + "\n")
    json.dump(blobs, open(f"{OUTPK}/BASE_BLOBS.json", "w"), indent=1)
    # ---- logs + helpers
    for lane, d in (("baseline_fix", f"{ROOT}/base_logs_fix"), ("patched_fix", f"{ROOT}/patched_logs_fix")):
        if os.path.isdir(d):
            os.makedirs(f"{OUTPK}/logs/tests_{lane}", exist_ok=True)
            for f in os.listdir(d):
                shutil.copyfile(f"{d}/{f}", f"{OUTPK}/logs/tests_{lane}/{f}")
    os.makedirs(f"{OUTPK}/fix", exist_ok=True)
    for f in ("fix/lane_fix_patched.cmd", "fix/lane_fix_base.cmd", "fix/quick_outcome.py", "finalize_fix.py"):
        shutil.copyfile(f"{ROOT}/{f}", f"{OUTPK}/fix/{os.path.basename(f)}")
    # ---- results json
    rj = json.load(open(f"{OUTPK}/test_results.json"))
    rj["fix_pass"] = {"done": done, "regressions": regress, "fixed": fixed, "same_fail": same_fail, "token_mismatch": tokbad, "tested_equals_staged": same, "rows": rows}
    json.dump(rj, open(f"{OUTPK}/test_results.json", "w"), indent=1)
    # ---- RESULT.md <!--FIX--> block
    rp = f"{OUTPK}/RESULT.md"
    r = open(rp, encoding="utf-8").read()
    a, b_ = r.find("<!--FIX-->"), r.find("<!--/FIX-->")
    if a >= 0 and b_ > a:
        blk = ("<!--FIX-->\n" + head + "\n\nTested == staged (sha256 of `out/` file equals the `patched_full` file the lane ran): " + ", ".join(f"`{os.path.basename(k)}` {v}" for k, v in same.items()) + "\n\n" + table + "\n")
        r = r[:a] + blk + r[b_:]
        open(rp, "w", encoding="utf-8", newline="\n").write(r)
    # ---- paths.json
    paths = []
    for dp, dn, fn in os.walk(OUT):
        for f in sorted(fn):
            if f == "paths.json":
                continue
            ab = os.path.join(dp, f).replace("\\", "/")
            paths.append([os.path.relpath(ab, OUT).replace("\\", "/"), ab, "raw"])
    json.dump(sorted(paths), open(f"{OUT}/paths.json", "w"), indent=1)
    print(head)
    print(table)
    print("tested_equals_staged:", same)
    print("paths.json entries:", len(paths))
    return 0 if (done and not regress and not tokbad and all(same.values())) else 1


if __name__ == "__main__":
    sys.exit(main())
