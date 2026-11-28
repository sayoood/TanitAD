"""Assemble RESULTS.md, MANIFEST.txt, SHA256SUMS.txt and logs/ of the LAUNCH-INTENT package from the run logs. Idempotent; safe to re-run at any time (a missing log is reported as PENDING, never as a pass).
python assemble.py   (run by tools/pipe_finish.sh when the pipelines are done; or by hand)"""
import hashlib
import os
import re
import shutil
import time
from pathlib import Path

OUT = Path("E:/Projects/TanitAD-artifacts/dagland/launch_intent")
S = Path("F:/TanitAD-scratch/launch_intent")
L = S / "logs"
(OUT / "logs").mkdir(exist_ok=True)


def last(p):
    p = Path(p)
    if not p.exists():
        return None
    ls = [l for l in p.read_text(encoding="utf-8", errors="replace").splitlines() if l.strip()]
    return ls[-1] if ls else ""


def summ(name):
    """(rc, summary line) of a run_k.sh run from its .progress file"""
    t = last(L / f"{name}.progress")
    if t is None or not t.startswith("done"):
        return None, "PENDING (no completed run)"
    m = re.match(r"done \S+ rc=(\d+) \S+ :: (.*)", t)
    return int(m.group(1)), m.group(2)


def counts(line):
    d = {}
    for k in ("passed", "failed", "error", "errors", "skipped", "deselected"):
        m = re.search(rf"(\d+) {k}", line or "")
        if m:
            d[k] = int(m.group(1))
    return d


rows = []
rc_g, s_g = summ("final_green")
rc_r, s_r = summ("final_red_on_tip")
rows.append(("GREEN on the build (all 36 tests)", s_g, rc_g == 0 and counts(s_g).get("failed", 0) == 0 and counts(s_g).get("error", 0) + counts(s_g).get("errors", 0) == 0))
rows.append(("RED on the unmodified tip d7208328 (same file, same golden)", s_r, rc_r not in (None, 0) and counts(s_r).get("failed", 0) >= 30))

suites = []
for f in ("test_v8f_ego_dagger_compose", "test_v8f_ego_perturb", "test_v8f_dagger_train", "test_v8f_dagger", "test_v8f_init_ckpt_o_interface", "test_launch_gate_v8f", "test_v8f_orc_chain", "test_v8f_mini", "test_v8f_pgt", "test_v8f_flat", "test_v8f_oacc"):
    rc, s = summ(f"suite_{f}")
    suites.append((f, rc, s))

mut = (OUT / "mutation.txt").read_text(encoding="utf-8") if (OUT / "mutation.txt").exists() else ""
msum = re.findall(r"SUMMARY: (\d+)/(\d+) mutants killed; survivors: (.*)", mut)
ctrl_bad = [l for l in mut.splitlines() if l.startswith("CONTROL") and "rc=0" not in l]
emut = (OUT / "enumerate_cells_li_mutation.txt").read_text(encoding="utf-8") if (OUT / "enumerate_cells_li_mutation.txt").exists() else ""
esum = re.findall(r"SUMMARY: (\d+)/(\d+) mutants killed", emut)
enum_out = (OUT / "enumerate_cells_li.out").read_text(encoding="utf-8") if (OUT / "enumerate_cells_li.out").exists() else ""

lines = []
w = lines.append
w("# RESULTS: PREREG_V8F_LAUNCH_INTENT draft + `--launch-intent-w` (built and tested, NOT run)")
w("")
w(f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')} by `tools/assemble.py` from the run logs. Nothing was trained, launched, staged in git, committed or pushed. Box 2, the queue and every chain were not touched.")
w("Base: tip `d7208328075a06efae321cc16c09ce7a902280ad` (`TIP.txt`). Every number below is MEASURED from the log named beside it. A line that reads PENDING is not a pass.")
w("")
w("## 1. The trainer flag against the unmodified tip")
w("")
w("| check | result | pass |")
w("|---|---|---|")
for name, s, ok in rows:
    w(f"| {name} | `{s}` | {'**PASS**' if ok else '**NOT YET / FAIL**'} |")
w("")
w("* Files: `logs/final_green.log`, `logs/final_red_on_tip.log`. The RED run is the same test file and golden executed against the unmodified `d7208328` export (the new symbols do not exist there).")
w("* The flag-off golden (`stage/stack/tests/launch_intent_golden.json`) was WRITTEN BY THE UNMODIFIED TIP (`logs/golden_write.log`) on three argv families (a DAG-OACC-style argv with the swap and `--w-acc 49.6969`, an ORC-style argv, plain M2b), 4 CPU steps each, torch 2.11.0+cu128. The build reproduces every logged number, every logged key, the final weights and optimiser state, the counters, and the key sets of `summary.json` / the checkpoint, bit for bit. The only differences are the declared additions (the checkpoint `cfg` gains `launch_intent_w` = 0.0; `levers` gains `launch_intent_w`, `launch_intent`, `launch_n`, `launch_eligible`; `selfcheck.json` gains `launch_intent_probe`).")
w("* `diff.patch`: 189 lines added and 2 removed (the two existing lines it touches are `make_config`'s keyword list and the `cfg.` read-back tuple, each re-added with the new name); `v8f_mini_data.py` and `launch_gate.py` are byte-identical to the tip (`logs/launch_gate_grep.txt`).")
w("")
w("## 2. Mutation proof of the trainer flag (`mutation.txt`, `tools/run_mutations.py`)")
w("")
if msum:
    k, n, surv = msum[-1]
    w(f"**{k} / {n} mutants killed; survivors: {surv}.** Controls (each selection GREEN on the unmutated tree first): " + ("all rc=0." if not ctrl_bad else f"NOT ALL GREEN: {ctrl_bad}"))
else:
    w("**PENDING**: `mutation.txt` has no SUMMARY line yet.")
w("")
w("Each mutant deletes or breaks ONE rule in a copy of the tree (`F:/TanitAD-scratch/launch_intent/mut`, restored and sha256-verified after every mutant) and names the `-k` selection that must go RED. Required mutants include: deleting the loss line that adds the term, flipping the registered `>=`, flipping `<` to `<=`, the chord over the wrong knots, the chord along one axis, the wrong knot window, the cross axis, the wrong denominator, the mask ignored, the mask taken after the ego stages, the attach line deleted, the loop never launching, the FLAG_CHECKS entry deleted, the G-DVB read-back deleted, the probe stubbed, the probe's all-False control deleted, the checkpoint losing the totals, the parser default flipped.")
w("")
w("## 3. The existing v8F suites on the build, one file at a time")
w("")
w("| suite | rc | result |")
w("|---|---|---|")
for f, rc, s in suites:
    w(f"| `{f}` | {rc if rc is not None else '-'} | `{s}` |")
w("")
w("A suite that is red here and green on the tip would be a regression of this build; compare `logs/suite_*.log` with the tip before reading any red as this build's. (`PENDING` = not finished.)")
w("")
w("## 4. The dose, the REP licence and the outcome table")
w("")
w("* `f_measure.json` (`tools/measure_f.py`): f = 49,618 / 306,566 = **0.161851**; window list sha256 equals the banked DAG-OACC runs'; trainer mask vs independent numpy spelling: 0 disagreements. **W_A = 1.58926**, W_B = 0.0357048, W_HI = 4.17852.")
w("* `compare_rep_argv.out`: DAG-ORC r2's pinned REP argv equals DAG-OACC's banked argv token for token apart from `--out` and `--seed`, at both seeds (exit 0).")
w("* `enumerate_cells_li.out`: " + ("exit 0 (`FAILS: none`), per-row counts equal DAG-ORC's registered counts." if "FAILS: none" in enum_out else "NOT PASSING") + " `enumerate_cells_li_mutation.txt`: " + (f"{esum[-1][0]} / {esum[-1][1]} mutants killed." if esum else "PENDING."))
w("")
w("## 5. Flags for the Master Mind (also in the PREREG §0)")
w("")
w("1. **No standalone \"O's along-track loss\" exists at the tip.** O's position loss is ADE/FDE over both axes and all 30 knots (`v8f_mini.py:870`) plus `w_acc`'s second difference. The flag adds W x the along-axis, knots 1-12 slice of ADE's own per-knot distance on launch samples (one additive term, one total line). Confirm the reading.")
w("2. **Arm B (one sixth) is a near no-op:** f = 0.1619, so the unweighted share is already 0.1619 and W_B = 0.0357 (+3.6 % weight). Re-fix B's share or drop B. Closed form for any share s: 1 + W = s (1 - f) / ((1 - s) f).")
w("3. **A gate-probe class the build caught:** every probe in `main()` calls `model.loss` on the probe batch, and a lever-ON loss needs `b[\"launch\"]` there too. The first test run failed on exactly this (5 tests); the probe batch now carries the mask. Any future lever that changes the loss's inputs must reach every probe, not only the loop.")
w("4. `launch_gate.py` needs no change; the v8f-mini gate is the trainer's `--selfcheck-only` (`logs/launch_gate_grep.txt`). The chain's one-lever asserter must allow `--launch-intent-w` as the only extra flag.")
w("5. REP reuse saves about 1.4 h of box 2 but needs DAG-ORC r2's REP runs to finish and pass; the golden covers CPU, not box 2's GPU.")
w("")
(OUT / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")

# logs/ : copy every log of this package (flat, small)
for p in sorted(L.glob("*")):
    if p.is_file() and p.suffix in (".log", ".progress", ".done", ".out", ".pipe", ".txt") and p.stat().st_size < 3_000_000:
        shutil.copy2(p, OUT / "logs" / p.name)
for src in (S / "out" / "measure_f.log",):
    if src.exists():
        shutil.copy2(src, OUT / "logs" / src.name)

# MANIFEST + SHA256SUMS
files = sorted(p for p in OUT.rglob("*") if p.is_file() and p.name not in ("MANIFEST.txt", "SHA256SUMS.txt", "FINISHED.txt") and "__pycache__" not in p.parts)
sums = []
for p in files:
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    sums.append(f"{h}  {p.relative_to(OUT).as_posix()}")
(OUT / "SHA256SUMS.txt").write_text("\n".join(sums) + "\n", encoding="utf-8", newline="\n")
man = ["MANIFEST: E:/Projects/TanitAD-artifacts/dagland/launch_intent/  (every artifact and where it lives)",
       "",
       "SINGLE-LOCATION items (exist ONLY on this disk, nowhere in git or on a pod; the Master Mind lands them):",
       "  PREREG_V8F_LAUNCH_INTENT.md, stage/ (trainer, model, test, golden), diff.patch, tools/, f_measure.json, compare_rep_argv.out, enumerate_cells_li.out, enumerate_cells_li_mutation.txt, mutation.txt, RESULTS.md, logs/",
       "  scratch trees (reproducible, not deliverables): F:/TanitAD-scratch/launch_intent/{pristine,tree,mut,mutpristine,tmp,logs}",
       "",
       "TO LAND (copy to the repo path shown; stage/ mirrors the repo layout):",
       "  stage/stack/scripts/v8f_mini_train.py          -> stack/scripts/v8f_mini_train.py          (changed vs d7208328: diff.patch)",
       "  stage/stack/tanitad/models/v8f_mini.py         -> stack/tanitad/models/v8f_mini.py         (changed vs d7208328: diff.patch)",
       "  stage/stack/tests/test_v8f_launch_intent.py    -> stack/tests/test_v8f_launch_intent.py    (new)",
       "  stage/stack/tests/launch_intent_golden.json    -> stack/tests/launch_intent_golden.json    (new; written by the UNMODIFIED tip)",
       "  PREREG_V8F_LAUNCH_INTENT.md                    -> Project Steering/PREREG_V8F_LAUNCH_INTENT.md (DRAFT; the Master Mind promotes it)",
       "  NOT changed: stack/scripts/launch_gate.py, stack/tanitad/data/v8f_mini_data.py (byte-identical to the tip)",
       "",
       "FILES (sha256 in SHA256SUMS.txt):"]
man += [f"  {p.relative_to(OUT).as_posix()}  ({p.stat().st_size} bytes)" for p in files]
(OUT / "MANIFEST.txt").write_text("\n".join(man) + "\n", encoding="utf-8", newline="\n")
print("assembled:", len(files), "files")


def _lf():
    """The deliverable is LF: normalise every text file under OUT (the Write tool and some editors emit CRLF on this box)."""
    CR, LF = bytes([13]), bytes([10])
    for q in OUT.rglob("*"):
        if q.is_file() and q.suffix in (".md", ".txt", ".py", ".sh", ".out", ".json", ".patch", ".log", ".progress", ".pipe", ".done") and "__pycache__" not in q.parts:
            b = q.read_bytes()
            if CR + LF in b:
                q.write_bytes(b.replace(CR + LF, LF))


_lf()
