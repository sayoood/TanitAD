#!/usr/bin/env python3
"""Declare the DAG-OACC / FLAT-DAG queue labels in the SIX repo tests that pin the queue's label set (Master Mind's addenda). Base = the Master Mind's patched versions (C:/Users/Admin/ofx/stack/tests, never edited);
every edit is an exact-count replace.
    python apply_test_edits.py <patched tests dir> <out dir>      (the staged result is oaccflatq/stage/stack/tests/)
  test_v8f_orc_chain.py : LATER_LABELS (label -> PREREG) and LATER_HEADER (usage-line prefixes)
  test_v8f_flat.py, test_v8f_oacc.py, test_v8f_kind.py, test_v8f_mixo.py : DAG_LABELS |= {...}  (their `set(blocks) - ... - DAG_LABELS` assertions)
  test_v8f_lora_ar2.py : dag |= {...}  (the local set subtracted as `- orc - dag`)
"""
import sys
from pathlib import Path

NL = chr(10)
src, out = Path(sys.argv[1]), Path(sys.argv[2])
OACC_PREREG = "PREREG_V8F_DAG_OACC"
FLAT_PREREG = "PREREG_V8F_FLATDAG"
DAGOACC_LABELS = ["dagoaccarm", "dagoaccarm_dry", "dagoaccarm_have", "dagoaccwacc", "dagoaccwacc_dry", "dagoaccwacc_have", "waccdag"]
FLATDAG_LABELS = ["flatdagarm", "flatdagarm_dry", "flatdagarm_have", "evalflatdag", "evalflatdag_1", "evalflatdag_2", "bankflat"]


def rep(t, old, new, n=1):
    assert t.count(old) == n, (old[:80], t.count(old))
    return t.replace(old, new)


def lit(labels):
    return "{" + ", ".join(f'"{l}"' for l in labels) + "}"


def dict_line(labels, prereg):
    return "    " + ", ".join(f'"{l}": "{prereg}"' for l in labels) + ","


def after_line(t, startswith, add_lines):
    """Insert ``add_lines`` after the single line of ``t`` that starts with ``startswith``."""
    lines = t.split(NL)
    k = [i for i, x in enumerate(lines) if x.startswith(startswith)]
    assert len(k) == 1, (startswith, k)
    return NL.join(lines[:k[0] + 1] + add_lines + lines[k[0] + 1:])


# ---- test_v8f_orc_chain.py
t = (src / "test_v8f_orc_chain.py").read_text(encoding="utf-8")
old_tail = '    "dag2arm": "PREREG_V8F_DAGGER2", "dag2arm_dry": "PREREG_V8F_DAGGER2", "dag2arm_have": "PREREG_V8F_DAGGER2", "bankdag2": "PREREG_V8F_DAGGER2",' + NL + "}"
t = rep(t, old_tail, old_tail[:-1] + dict_line(DAGOACC_LABELS, OACC_PREREG) + NL + dict_line(FLATDAG_LABELS, FLAT_PREREG) + NL + "}")
t = rep(t, '    + ("rem   gatedag / traindag / evaldag also take ARM dag2 ",)',
        '    + ("rem   gatedag / traindag / evaldag also take ARM dag2 ", "rem   gatedag / traindag / evaldag also take ARM dagoacc ", "rem   gatedag / traindag / evaldag also take ARM flatdag ",' + NL
        + '       "rem   v8f_mini_box2_queue.cmd waccdag ", "rem   v8f_mini_box2_queue.cmd bankflat ")')
t = rep(t, "(raw/2026-10-26-dagger-queue/, raw/2026-10-27-dagger2-queue/)", "(raw/2026-10-26-dagger-queue/, raw/2026-10-27-dagger2-queue/, raw/2026-10-27-oaccflat-queue/)")
(out / "test_v8f_orc_chain.py").write_text(t, encoding="utf-8", newline=NL)

# ---- the five that subtract a DAG set
for name, start, indent, var in (("test_v8f_flat.py", "DAG_LABELS = {", "", "DAG_LABELS"), ("test_v8f_oacc.py", "DAG_LABELS = {", "", "DAG_LABELS"), ("test_v8f_kind.py", "DAG_LABELS = {", "", "DAG_LABELS"),
                                 ("test_v8f_mixo.py", "DAG_LABELS = {", "", "DAG_LABELS"), ("test_v8f_lora_ar2.py", "    dag = {", "    ", "dag")):
    t = (src / name).read_text(encoding="utf-8")
    k = t.split(NL)
    # the flat / oacc definitions span two physical lines (the set literal continues): find the line that CLOSES the assignment
    i = next(i for i, x in enumerate(k) if x.startswith(start))
    j = i
    while not k[j].split("#")[0].rstrip().endswith("}"):
        j += 1
    add = [f"{indent}{var} |= {lit(DAGOACC_LABELS)}      # {OACC_PREREG}, after DAGGER2's", f"{indent}{var} |= {lit(FLATDAG_LABELS)}      # {FLAT_PREREG}, after DAG-OACC's"]
    t = NL.join(k[:j + 1] + add + k[j + 1:])
    (out / name).write_text(t, encoding="utf-8", newline=NL)
print("staged", sorted(p.name for p in out.glob("test_v8f_*.py")))
