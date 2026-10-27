#!/usr/bin/env python3
"""Write MANIFEST.txt: every deliverable of dagq/, its sha256, its role, and where it lands. Run last (after the proofs)."""
from __future__ import annotations

import hashlib
from pathlib import Path

H = Path(__file__).resolve().parent
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
RAW = PK + "/raw/2026-10-26-dagger-queue"
ROWS = [
    # (path under dagq, role, lands at)
    (f"stage/{PK}/code/v8f_mini_box2_queue.cmd", "the box-2 queue: tip 2a78f059 + stages bankdag gatedag traindag evaldag (+ helpers dagname prepdag dagarm); additive (queue.diff)", f"REPO {PK}/code/v8f_mini_box2_queue.cmd (replaces the tip's)"),
    ("stage/stack/scripts/v8f_dagger_queue_tool.py", "the shard fan-out (`bank`) and `sha` helper the queue stages call", "REPO stack/scripts/v8f_dagger_queue_tool.py (NEW)"),
    ("queue.diff", "unified diff of the queue against the tip (insertions only)", f"REPO {RAW}/queue.diff"),
    ("queue_additions/doc.txt", "source of the queue's documentation lines", f"REPO {RAW}/queue_additions/"),
    ("queue_additions/dispatch.txt", "source of the queue's dispatch lines", f"REPO {RAW}/queue_additions/"),
    ("queue_additions/stages.txt", "source of the queue's new stage bodies", f"REPO {RAW}/queue_additions/"),
    ("build_queue.py", "rebuilds the staged queue from the tip + additions (additive check)", f"REPO {RAW}/"),
    ("tip_v8f_mini_box2_queue.cmd", "the tip's queue (git show 2a78f059), the base of the diff", f"REPO {RAW}/ (reference copy)"),
    ("box2_ship/chain_v8f_dag.sh", "the dev-box chain (generated): ORC predicate, banks, K-B0..K-B5 controls + COVERAGE gate, BANK PUBLICATION (shard npz + 11-name SHA256SUMS.txt per seed, after the judge), 4 gates, 4 runs, standard T0 eval", "DEV BOX C:/Users/Admin/box2_ship/chain_v8f_dag.sh (NEW; not repo)"),
    ("box2_ship/chain_v8f_dag.sh.in", "the chain's template (the banked checkpoint hashes are copied in by make_chain.py)", f"REPO {RAW}/ (with make_chain.py)"),
    ("make_chain.py", "generates chain_v8f_dag.sh / v8f_guard_dag.sh / run_chain_v8f_dag.cmd / wake_v8fdag.sh from the originals", f"REPO {RAW}/"),
    ("box2_ship/gate_dag_assert.py", "the gate assertions + token writer, run on a pulled selfcheck.json", "DEV BOX C:/Users/Admin/box2_ship/gate_dag_assert.py (NEW)"),
    ("box2_ship/v8f_dag_chain_checks.py", "the ORC predicate + bank judge (controls incl. K-B4 / K-B5, COVERAGE) + argv helpers + the bank-publication checks (bank-files / bank-verify / bank-plan)", "DEV BOX C:/Users/Admin/box2_ship/v8f_dag_chain_checks.py (NEW)"),
    ("box2_ship/v8f_guard_dag.sh", "COPY of v8f_guard.sh that also stops the bank generators", "DEV BOX C:/Users/Admin/box2_ship/v8f_guard_dag.sh (NEW)"),
    ("box2_ship/run_chain_v8f_dag.cmd", "detached launcher (NOT RUN)", "DEV BOX C:/Users/Admin/box2_ship/run_chain_v8f_dag.cmd (NEW)"),
    ("box2_ship/wake_v8fdag.sh", "the waker", "DEV BOX C:/Users/Admin/box2_ship/wake_v8fdag.sh (NEW)"),
    ("dagq_lib.py", "shared helpers: runs the real queue through cmd.exe and parses what it prints", f"REPO {RAW}/"),
    ("selftest_dag_queue.py", "THE PROOF (sections A-H): dry-run render, fake-box run, chain decisions, chain script, real selfcheck assertions, shard tool, bank publication checks, the chain's offline publish flow", f"REPO {RAW}/"),
    ("selftest_output.txt", "output of the selftest on the final bytes", f"REPO {RAW}/"),
    ("real_gate_proof.py", "the REAL trainer's --selfcheck-only on the queue's argv tail (CPU, synthetic world)", f"REPO {RAW}/"),
    ("real_gate_proof.log", "its output", f"REPO {RAW}/"),
    ("rg/out/RESULT.json", "its result record", f"REPO {RAW}/real_gate/"),
    ("rg/out/dag_s0/selfcheck.json", "REAL gate selfcheck.json, DAG argv", f"REPO {RAW}/real_gate/"),
    ("rg/out/dag0_s0/selfcheck.json", "REAL gate selfcheck.json, DAG-0 argv", f"REPO {RAW}/real_gate/"),
    ("kb_shape_probe.py", "the generator's own K-B4 / K-B5 records (di/tree, read-only import) run through the judge's rules", f"REPO {RAW}/"),
    ("kb_shape_probe.txt", "its output", f"REPO {RAW}/"),
    ("rg/COPY_PROVENANCE.txt", "when/what rg/tree was copied from di/tree", f"REPO {RAW}/"),
    ("mutate_dag_queue.py", "mutation proof: 51 mutants (20 queue stages, 9 bank judge, 12 bank-publication checks, 10 chain)", f"REPO {RAW}/"),
    ("mutation_output.txt", "its output (51/51 killed)", f"REPO {RAW}/"),
    ("bankpub.diff", "unified diff of the bank-publication revision (checks, chain, chain template, selftest, mutator, NOTES) against the bytes before it", f"REPO {RAW}/bankpub.diff"),
    ("chain_dry_run.txt", "chain --dry-run against the staged queue (ORC read HARMS -> p 0.1)", f"REPO {RAW}/"),
    ("make_manifest.py", "this script", f"REPO {RAW}/"),
    ("NOTES.md", "what was built, the guarantees, UNVERIFIED, open decisions", f"REPO {RAW}/NOTES.md"),
]


def main() -> int:
    out = ["# dagq MANIFEST (sha256  role  ->  where it lands). `rg/tree`, `rg/work`, `rg/out` (bulk), `bt` are DISPOSABLE (a python-only copy of di/tree, scratch, test worlds).", ""]
    miss = 0
    for rel, role, lands in ROWS:
        p = H / rel
        if not p.exists():
            out.append(f"MISSING  {rel}")
            miss += 1
            continue
        out.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {rel}\n    {role}\n    -> {lands}")
    (H / "MANIFEST.txt").write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"wrote MANIFEST.txt ({len(ROWS) - miss} files, {miss} missing)")
    return 1 if miss else 0


if __name__ == "__main__":
    raise SystemExit(main())
