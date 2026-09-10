# -*- coding: utf-8 -*-
"""WP-D step 23 -- copy the A3 deliverables into the repo and STAGE them.

⛔ STAGE ONLY. No commit, no push (AGENT_OPERATING_STANDARD rule 1).
⛔ The G: mount is degraded, so:
   * every copy retries and is verified by md5 on BOTH sides;
   * a comparison whose operands are not the right length reports INCONCLUSIVE,
     never MATCH (CLAUDE.md's empty-string blob-comparison hole);
   * `git add` is retried, and staging is verified by a BLOB COMPARISON with
     both operands asserted to be 40 characters.
Written in Python, not shell: two heredoc-quoting bugs tonight (a lost `\\$1`
and a lost line-continuation) came from exactly that layer.
"""
import hashlib
import io
import os
import subprocess
import sys
import time

REPO = r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD"
PKG = r"TanitAD Research Lab\Architecture & Inference\Research\2026-09-07-wpd-bev-aux"
SRC = r"C:\Users\Admin\wpd-probe"


def md5(p):
    try:
        h = hashlib.md5()
        with open(p, "rb") as fh:
            for b in iter(lambda: fh.read(1 << 20), b""):
                h.update(b)
        return h.hexdigest()
    except Exception:
        return ""


def copy(src, rel, tries=12):
    dst = os.path.join(REPO, rel)
    a = md5(src)
    if len(a) != 32:
        return ("SRC_UNREADABLE", rel)
    for i in range(tries):
        try:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(src, "rb") as f:
                data = f.read()
            with open(dst, "wb") as f:
                f.write(data)
            b = md5(dst)
            if len(b) != 32:
                time.sleep(4)
                continue                      # INCONCLUSIVE, never "match"
            if a == b:
                return ("OK", rel)
            return ("MISMATCH", rel)
        except Exception:
            time.sleep(4)
    return ("INCONCLUSIVE", rel)


CODE = ["b1_bank_wpd.py", "b2_probe_wpd.py", "b5_fast_boot.py",
        "b10_run_a3.sh", "b10_wait_then_run.sh", "b11_a3_verdict.py",
        "b12_determinism_control.py", "b15_instrument_floor.sh",
        "b17b_planner_seed_floor.py", "b18_chain_tail.sh",
        "b19_reconstruct_cart_meta.py", "b20_close_a3_in_result.py",
        "b21_write_5b_closure.py", "b22_update_register.py", "b23_stage.py",
        "b24_planner_floor_writeup.py", "b25_correct_headline.py",
        "b26_register_f2.py"]

RAWF = ["panel_rep_cart.json", "panel_rep_pol.json",
        "boot_rep_cart.json", "boot_rep_pol.json",
        "panel_instrument_cart.json", "boot_instrument_cart.json",
        "a3_verdict_cart.json", "a3_verdict_pol.json",
        "bank_determinism_control.json", "D0c_argv_audit.json",
        "probe_rep_cart.log", "probe_rep_pol.log",
        "boot_rep_cart.log", "boot_rep_pol.log",
        "probe_instrument_cart.log", "boot_instrument_cart.log",
        "paired_floor_wpdD0b_vs_D0.json", "paired_floor_wpdD0c_vs_D0.json",
        "planner_wpdD0b.log", "planner_wpdD0c.log", "planner_seed_floor.log"]

plan = []
for f in CODE:
    p = os.path.join(SRC, "code", f)
    if os.path.exists(p):
        plan.append((p, os.path.join(PKG, "raw" if False else "code", f)))
for f in RAWF:
    p = os.path.join(SRC, "raw", f)
    if os.path.exists(p):
        plan.append((p, os.path.join(PKG, "raw", f)))
if os.path.exists(os.path.join(SRC, "bank_rep.log")):
    plan.append((os.path.join(SRC, "bank_rep.log"), os.path.join(PKG, "raw", "bank_rep.log")))
if os.path.exists(os.path.join(SRC, "bank_rep", "idx.json")):
    plan.append((os.path.join(SRC, "bank_rep", "idx.json"),
                 os.path.join(PKG, "raw", "bank_rep_idx.json")))
plan.append((os.path.join(SRC, "PANEL_RESULT.md"), os.path.join(PKG, "PANEL_RESULT.md")))
plan.append((os.path.join(SRC, "GOALS_AND_CLAIMS.new.md"),
             os.path.join("Project Steering", "GOALS_AND_CLAIMS.md")))

results, copied = [], []
for s, rel in plan:
    st, r = copy(s, rel)
    results.append((st, r))
    if st == "OK":
        copied.append(r)
    print(f"[copy] {st:12s} {r}")
bad = [r for st, r in results if st != "OK"]
print(f"\n[copy] {len(copied)} OK, {len(bad)} not OK")
for r in bad:
    print(f"  ⚠️ NOT COPIED: {r}")

if not copied:
    print("NOTHING COPIED -- refusing to stage")
    sys.exit(3)

# ---------------------------------------------------------------- stage ------
def git(*args, tries=6):
    for i in range(tries):
        p = subprocess.run(("git",) + args, cwd=REPO, capture_output=True, text=True)
        if p.returncode == 0:
            return p.stdout.strip()
        if i == tries - 1:
            return None
        time.sleep(5)
    return None


print("\n[stage] git add, explicit paths, in batches")
for i in range(0, len(copied), 6):
    batch = copied[i:i + 6]
    for t in range(8):
        p = subprocess.run(["git", "add", "--"] + batch, cwd=REPO,
                           capture_output=True, text=True)
        if p.returncode == 0:
            break
        time.sleep(5)
    print(f"  add batch {i//6}: rc={p.returncode} {p.stderr.strip()[:90]}")

# ---- VERIFY by BLOB COMPARISON, both operands asserted 40 chars -------------
print("\n[verify] index blob vs worktree blob (INCONCLUSIVE unless both are 40 chars)")
ok = incon = mism = 0
for rel in copied:
    stage = git("ls-files", "--stage", "--", rel)
    idx = stage.split()[1] if stage and len(stage.split()) > 2 else ""
    wt = git("hash-object", "--", os.path.join(REPO, rel)) or ""
    if len(idx) != 40 or len(wt) != 40:
        print(f"  INCONCLUSIVE  {rel}  (idx {len(idx)}c, wt {len(wt)}c)")
        incon += 1
    elif idx == wt:
        print(f"  VERIFIED {idx[:12]}  {rel}")
        ok += 1
    else:
        print(f"  MISMATCH      {rel}  idx {idx[:12]} wt {wt[:12]}")
        mism += 1
print(f"\n[verify] VERIFIED {ok} · INCONCLUSIVE {incon} · MISMATCH {mism} "
      f"· of {len(copied)} copied")
print("STAGED_NOT_COMMITTED (no git commit, no git push was run)")
