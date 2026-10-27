#!/usr/bin/env python3
"""finalize_dag2q.py -- fill NOTES.md's mutation section from mutation.json and write MANIFEST.txt (sha256 of every deliverable). Idempotent; run after mutate_dag2_queue.py."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def mutation_section() -> str:
    mj = HERE / "mutation.json"
    if not mj.is_file():
        return "(mutation.json is missing: the mutation proof did not finish; see HANDOFF.md)"
    d = json.loads(mj.read_text(encoding="utf-8"))
    lines = [f"**MEASURED: {d['killed']}/{d['n_mutants']} mutants killed, {d['survived']} survived, {d['anchor_errors']} anchor errors; the unmutated control (all sections, full sweep) is green** "
             f"(`mutate_dag2_queue.py`; each mutant is ONE defect in a COPY of the queue / helper / chain; the self-test sections that cover that file are re-run on the copy and the mutant is killed iff they go RED; "
             f"a mutant whose anchor text is not found exactly once is an ANCHOR ERROR and never counts as killed).", "",
             "| id | kind | the defect | result | red checks |", "|---|---|---|---|---|"]
    kinds = {"Q": "queue", "K": "bank judge / DAG-read predicate", "G": "gate asserter", "C": "chain"}
    for r in d["results"]:
        lines.append(f"| {r['id']} | {kinds.get(r['kind'], r['kind'])} | {r['desc']}{(' -- ' + r['note']) if r.get('note') else ''} | {r['status']} | {r.get('n_red', '')} |")
    if d.get("history"):
        lines.insert(2, "History: " + d["history"])
        lines.insert(3, "")
    surv = [r for r in d["results"] if r["status"] != "KILLED"]
    lines.append("")
    lines.append("All killed." if not surv else "NOT killed: " + ", ".join(f"{r['id']} ({r['status']})" for r in surv))
    return "\n".join(lines)


def main() -> int:
    notes = HERE / "NOTES.md"
    s = notes.read_text(encoding="utf-8")
    a, b = s.index("<!--MUTATION-->"), s.index("<!--/MUTATION-->")
    s = s[: a + len("<!--MUTATION-->")] + "\n" + mutation_section() + "\n" + s[b:]
    notes.write_bytes(s.encode("utf-8"))
    rows = []
    for p in sorted(HERE.rglob("*")):
        if p.is_file() and p.name != "MANIFEST.txt" and "__pycache__" not in p.parts and not p.name.endswith(".pyc"):
            rows.append((p.relative_to(HERE).as_posix(), p.stat().st_size, sha(p)))
    where = {"stage/": "dag2q/stage (ONE PLACE, the dev box E: only; the Master Mind lands it at its repo path)",
             "box2_ship/": "dag2q/box2_ship (ONE PLACE; installed by copying to C:/Users/Admin/box2_ship/)"}
    out = ["MANIFEST -- E:/Projects/TanitAD-artifacts/dag2q/ (every file lives ONLY here on the dev box E: unless stated; nothing is committed, pushed or on box 2)", ""]
    for rel, size, h in rows:
        tag = next((v for k, v in where.items() if rel.startswith(k)), "dag2q (ONE PLACE)")
        out.append(f"{h}  {size:>9}  {rel}   [{tag}]")
    out.append("")
    out.append("scratch (disposable, not deliverables): C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/dag2q_work/{tree,bt,mut,tmp,shim,sweep,fakebox,chk_*,reach_syn_work}")
    (HERE / "MANIFEST.txt").write_bytes(("\n".join(out) + "\n").encode("utf-8"))
    print(f"MANIFEST.txt: {len(rows)} files; NOTES.md mutation section filled")
    return 0


if __name__ == "__main__":
    sys.exit(main())
