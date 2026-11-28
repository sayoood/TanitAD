"""mk_manifest.py -- (re)writes MANIFEST.txt of this directory: path, bytes, sha256, role.   python mk_manifest.py"""
import hashlib
import os

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
ROLE = {
    "dag2_t1_chain.sh": "THE CHAIN (derived from dagt1/dag_t1_chain.sh by mk_dag2_helpers.py)",
    "dag2_t1_chain.cmd": "detached WMI launcher for the chain",
    "dag2_read.py": "THE READER (composes dag_read.py + Read A; writes DAG2_READ.json)",
    "dag2t1_manifest.py": "manifest / initsha / dagbase / treetip / ids (imports dagt1_manifest.py + dagt1_pullcheck.py unchanged)",
    "dag2t1_pullcheck.py": "content check of one pulled run / bank (derived)",
    "mk_dag2_helpers.py": "derivation script (exact-count substitutions) for the chain and the pull check",
    "mk_manifest.py": "writes this file",
    "chain.diff": "diff: chain vs dagt1/dag_t1_chain.sh",
    "pullcheck.diff": "diff: pull check vs dagt1/dagt1_pullcheck.py",
    "dag2t1_selftest.py": "self-test + mutation proof (plain script, no pytest)",
    "selftest_output.txt": "output of the last full self-test run",
    "mutation.json": "per-mutant result of that run",
    "dry_run_real_config.txt": "PRE-AMENDMENT evidence (PREREG pin 87d6c77d, dagland path), NOT regenerated in the A1 re-proof: a real-configuration dry run stream-hashes the pulled eval_M2b-dag2* files (blinding); bash dag2_t1_chain.sh --dry-run, DAG2T1_TEST_CFREE_MB=3000",
    "AMEND1.md": "what changed for PREREG_V8F_DAGGER2 Amendment 1 (A1-2026-10-27, 1446333d), counts before/after, what was not run (blinding)",
    "inputs/PREREG_V8F_DAGGER2.md": "sha-verified copy (c4239e45...) of the 1446333d blob of Project Steering/PREREG_V8F_DAGGER2.md (as amended); the reader and chain pin and read this copy",
    "inputs/enumerate_cells.py": "the PREREG Amendment 1 enumeration (blob of 1446333d, sha 3a437ea8...), executed by the self-test (checks E1-E6)",
    "NOTES.md": "install, launch, MEASURED vs UNVERIFIED, PREREG issues",
    "HANDOFF.md": "state and next steps",
    "readA_code/rr_lib.py": "SINGLE-COPY besides dagreread/code/ (NOT in the repo at 1d4f4fbb): Read A driver code, sha-verified copy",
    "readA_code/readA.py": "SINGLE-COPY besides dagreread/code/ (NOT in the repo at 1d4f4fbb): Read A driver code, sha-verified copy",
    "readA_code/build_report.py": "SINGLE-COPY besides dagreread/code/ (NOT in the repo at 1d4f4fbb): Read A driver code, sha-verified copy",
    "inputs/b1_eval_lead_block_k30.npz": "sha-verified copy (7a879baa...) of repo blob 1d4f4fbb:<v8f-mini>/raw/2026-10-22-dk/b1_eval_lead_block_k30.npz",
}
TOP = {"logs": "run log of this build", "preflight": "preflight evidence (bank + run-record controls on the pulled iteration-2 runs; ids/ = clip-id lists)",
       "repro": "RC-DAG evidence: layer A reproduces DAG's banked Read A table with zero differences"}
SKIP_DIRS = ("__pycache__", "_mut", "_tmp", "_st")
SKIP_FILES = ("MANIFEST.txt", "_mutant_chain.sh", "_scen_driver.py")
rows = []
for d, ds, fs in os.walk("."):
    ds[:] = [x for x in ds if x not in SKIP_DIRS]
    for f in fs:
        p = os.path.join(d, f).replace(chr(92), "/")[2:]
        if f in SKIP_FILES:
            continue
        b = open(p, "rb").read()
        rows.append((p, len(b), hashlib.sha256(b).hexdigest()))
rows.sort()
tot = sum(n for _, n, _ in rows)
out = [
    "# MANIFEST of E:/Projects/TanitAD-artifacts/dag2t1/ (2026-10-27). Every file: sha256, bytes, path, role. Nothing here is in git or on any other disk unless the role says so.",
    f"# {len(rows)} files, {tot} bytes apparent (limit 50 MB); ~75 MB ALLOCATED on exFAT. Scratch NOT listed (re-creatable by dag2t1_selftest.py): _mut/ _st/ _tmp/ __pycache__/ _mutant_chain.sh _scen_driver.py.",
    "# SINGLE-COPY / INTEGRATE: this whole directory lives on E: only (external exFAT). Read A's driver code (readA_code/) is not in the repo. Land dag2_read.py, dag2t1_manifest.py, dag2t1_pullcheck.py, dag2_t1_chain.sh, mk_dag2_helpers.py, dag2t1_selftest.py, inputs/PREREG_V8F_DAGGER2.md, inputs/enumerate_cells.py, AMEND1.md (the PREREG's Amendment 1 placed the 4 cells; see AMEND1.md).",
    "",
]
for p, n, h in rows:
    out.append(f"{h}  {n:>9}  {p}   [{ROLE.get(p) or TOP.get(p.split('/')[0], '')}]")
out += ["", "# READ IN PLACE, NOT OWNED (pinned by sha256 in dag2_read.py PINS / the chain): C:/Users/Admin/dagread/dag_read.py (sha 1cf56c7c..., byte-identical to the repo blob 1d4f4fbb:<v8f-mini>/raw/2026-10-26-dagger-read/dag_read.py); E:/Projects/TanitAD-artifacts/{dagreread/reader_src, dagt1_tree_b43c7a9a, dagt1, dagland/PREREG_V8F_DAGGER2.md, v8f-dagger, v8f-dagger-t1, v8f-dagger2, sfresh_inputs}."]
open("MANIFEST.txt", "w", encoding="utf-8", newline="\n").write("\n".join(out) + "\n")
print(len(rows), "files,", tot, "bytes")
