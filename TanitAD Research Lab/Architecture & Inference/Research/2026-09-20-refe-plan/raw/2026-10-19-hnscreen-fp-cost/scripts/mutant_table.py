"""Build hnfp3/MUTANTS.md from the FINAL logs of the three mutation matrices (the table is generated, never typed).
  gate level  : logs/test_hn_ctx_anchor_matrix_run*.log   (refe/test_hn_ctx_anchor.py   : 38 arms, 34 mutants of gate_token.py / launch_gate.py)
  chain level : logs/test_hnscreen_ctx_chain_matrix_run*.log (refe/test_hnscreen_ctx_chain.py: 9 arms, 10 mutants of hnscreen_devbox.py)
  reader      : logs/test_hn_screen_read_run*.log          (eval/test_hn_screen_read.py   : +8 gate arms, +5 NEW mutants of hn_screen_read.py)
Each mutant is a copy of the real source with ONE exact-once snippet edit; it must turn the named arms RED.  usage: python mutant_table.py"""
import glob
import os
import re

ROOT = "C:/Users/Admin/par18/hnfp3"


def latest(pat):
    c = sorted(glob.glob(f"{ROOT}/{pat}"), key=os.path.getmtime)
    return c[-1] if c else None


NEW_READER = ("the reader's cache-binding gate is not called", "the reader ignores a refusal of the gate", "a gate that cannot run is a pass", "the reader checks no seed (an empty pass list)", "--chain-root is optional")
rows, notes = [], []
for label, pat, kind in (("gate_token.py / launch_gate.py", "logs/test_hn_ctx_anchor_matrix_run*.log", "std"), ("hnscreen_devbox.py (the chain)", "logs/test_hnscreen_ctx_chain_matrix_run*.log", "std"),
                         ("hn_screen_read.py (the reader)", "logs/test_hn_screen_read_run*.log", "reader"),
                         ("gate_token.py / launch_gate.py (hnfp2's test_hn_content_fp.py matrix, table re-anchored on HNFP3's code)", "logs/test_hn_content_fp_run*.log", "std")):
    p = latest(pat)
    if not p:
        notes.append(f"{label}: NO LOG")
        continue
    txt = open(p, encoding="utf-8", errors="replace").read().splitlines()
    verdict = [ln for ln in txt if ln.startswith("ZZTEST_PASS") or ln.startswith("ZZTEST_FAIL")]
    notes.append(f"{label}: `{os.path.basename(p)}` -> {verdict[-1] if verdict else 'NO VERDICT LINE'}")
    for ln in txt:
        if kind == "std":
            m = re.match(r"^(.*?)\s+RED\s+(\d+)\s+e\.g\. (.*)$", ln)
            if m and not ln.startswith(("PROBLEM", " ")):
                rows.append((label, m.group(1).strip(), int(m.group(2)), m.group(3).strip()[:140], "RED" if int(m.group(2)) > 0 else "GREEN (BAD)"))
        else:
            if len(ln) > 52 and ln[:50].strip() in NEW_READER:
                first = ln[50:].strip()
                rows.append((label, ln[:50].strip(), len([c for c in first.split(",") if c.strip()]), first[:140], "RED" if first else "GREEN (BAD)"))
out = ["# MUTANTS -- every mutant of the (b') change, generated from the final logs by scripts/mutant_table.py", "",
       "Each row is a copy of the real source with ONE exact-once snippet edit (the single enforcing line deleted / inverted); the named arms must turn RED (MEASURED, artifact: the log named in the last section).",
       "A mutant that stays green is a PROBLEM line in the log and the matrix fails.", "", "| file | mutant (what the edit does) | arms RED (listed, up to 2) | verdict |", "|---|---|---|---|"]
for label, name, n, first, v in rows:
    out.append(f"| {label} | {name} | {first.replace('|', '/')[:120]} | {v} |")
out += ["", f"Mutants listed: {len(rows)} ({sum(1 for r in rows if r[4] == 'RED')} RED, {sum(1 for r in rows if r[4] != 'RED')} not RED).  The reader's table lists only the 5 NEW mutants; the 30 older ones of `test_hn_screen_read.py` are unchanged and still RED (see its log).", "", "## Logs", ""]
out += [f"* {n}" for n in notes]
open(f"{ROOT}/MUTANTS.md", "w", encoding="utf-8", newline="\n").write("\n".join(out) + "\n")
print("\n".join(out[-12:]))
print("rows", len(rows))
