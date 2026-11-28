#!/usr/bin/env python3
"""Declare the DAG-ORC (DAG-OACC base) queue label in the SIX repo tests that pin the queue's label set. Base = the tip 6cd741b6's versions (never edited in place); every edit is an exact-count replace.
    python apply_test_edits.py <tests dir of the tip> <out dir>      (the staged result is chain/stage/stack/tests/)
  test_v8f_orc_chain.py : LATER_LABELS (label -> PREREG) and LATER_HEADER (usage-line prefixes)
  test_v8f_flat.py, test_v8f_oacc.py, test_v8f_kind.py, test_v8f_mixo.py : DAG_LABELS |= {...}  (their `set(blocks) - ... - DAG_LABELS` assertions)
  test_v8f_lora_ar2.py : dag |= {...}  (the local set subtracted as `- orc - dag`)
The edit adds exactly ONE label, dagoaccorcarm: the two arms ride the existing gatedag / traindag / evaldag stages, so no dispatched stage and no dry-run variant label exist.
"""
import sys
from pathlib import Path

NL = chr(10)
src, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
PREREG = "PREREG_V8F_DAG_ORC"
LABELS = ["dagoaccorcarm"]


def rep(t, old, new, n=1):
    assert t.count(old) == n, (old[:80], t.count(old))
    return t.replace(old, new)


def lit(labels):
    return "{" + ", ".join(f'"{l}"' for l in labels) + "}"


# ---- test_v8f_orc_chain.py
t = (src / "test_v8f_orc_chain.py").read_bytes().decode("utf-8")
k = t.split(NL)
i = [n for n, x in enumerate(k) if x.startswith('    "dagorcarm": "PREREG_V8F_DAG_ORC"')]
assert len(i) == 1, i
k.insert(i[0] + 1, "    " + ", ".join(f'"{l}": "{PREREG}"' for l in LABELS) + ",")
t = NL.join(k)
t = rep(t, '"rem   gatedag / traindag / evaldag also take ARM dagorc ")', '"rem   gatedag / traindag / evaldag also take ARM dagorc ",' + NL + '       "rem   gatedag / traindag / evaldag also take ARM dagoaccorc ")')
t = rep(t, "raw/2026-10-27-dag-orc-queue/)", "raw/2026-10-27-dag-orc-queue/, raw/2026-10-27-dagorc-oacc-queue/)")
(out / "test_v8f_orc_chain.py").write_bytes(t.encode("utf-8"))

# ---- the five that subtract a DAG set: the new line goes after the LAST `<var> |=` line of the assignment block
for name, var, indent in (("test_v8f_flat.py", "DAG_LABELS", ""), ("test_v8f_oacc.py", "DAG_LABELS", ""), ("test_v8f_kind.py", "DAG_LABELS", ""), ("test_v8f_mixo.py", "DAG_LABELS", ""), ("test_v8f_lora_ar2.py", "dag", "    ")):
    t = (src / name).read_bytes().decode("utf-8")
    k = t.split(NL)
    last = [n for n, x in enumerate(k) if x.startswith(f"{indent}{var} |= ")]
    assert last, name
    j = last[-1]
    assert k[j].split("#")[0].rstrip().endswith("}"), (name, k[j][:100])
    add = [f"{indent}{var} |= {lit(LABELS)}      # {PREREG} Amendment 1 (DAG-OACC base), after DAG-ORC's"]
    t = NL.join(k[:j + 1] + add + k[j + 1:])
    (out / name).write_bytes(t.encode("utf-8"))
print("staged", sorted(p.name for p in out.glob("test_v8f_*.py")))
