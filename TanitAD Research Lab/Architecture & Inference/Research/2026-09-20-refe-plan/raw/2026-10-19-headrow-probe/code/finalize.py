"""Package the HEADROW-PROBE run into the staging tree (stage, never push): copy code + the run's small artifacts, hash everything, write CODE_SHA256_HEADROW_PROBE.json, MANIFEST.md and paths.json.
Large per-cell OOF tables (oof/, c2/, cpos/ npz) stay on disk (listed with their sha256 as ONE-PLACE artifacts); the export npz that carries the 16 main OOF tables IS staged.
usage: python finalize.py"""
import hashlib
import json
import os
import shutil

ROOT = "C:/Users/Admin/par18/probe"
REL = "TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-19-headrow-probe"
OUT = f"{ROOT}/out"
STAGE = f"{OUT}/{REL}"
RUN = f"{ROOT}/run/f10"
CODE = f"{ROOT}/code"
PARENT_SHA = "a751c63eb8e38343a7ec1286114a158046ce4ec13390b2eea0a4b1d2237c38d8"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def cp(src, dst_rel):
    dst = f"{STAGE}/{dst_rel}"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.abspath(src) != os.path.abspath(dst):
        shutil.copyfile(src, dst)
    return dst


entries = []     # (repo_rel, abs_path, class, note)
for n in ("headrow_probe.py", "test_headrow_probe.py", "export_oof.py", "gen_tables.py", "launch_probe.py", "launch_chain.py", "chain.sh", "wait_pid.py", "finalize.py", "c2x.py", "c2y.py", "c2z.py", "slotprior.py", "run_c2_cells.py", "launch_b.sh"):
    entries.append((f"code/{n}", cp(f"{CODE}/{n}", f"code/{n}"), "code", ""))
code_sha = {n: sha(f"{STAGE}/code/{n}") for n in sorted(os.listdir(f"{STAGE}/code")) if os.path.isfile(f"{STAGE}/code/{n}") and n != "CODE_SHA256_HEADROW_PROBE.json"}
with open(f"{STAGE}/code/CODE_SHA256_HEADROW_PROBE.json", "w", encoding="utf-8", newline="\n") as f:
    json.dump({"files": code_sha, "landed_parent_imported_unchanged": {"epr1.py": PARENT_SHA, "tip": "1afc54f6fa5ee865f2966e3c0a7b3141f24ef0c1",
               "repo_path": "TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-18-ep-r1/code/epr1.py", "note": "deployed byte-exact next to headrow_probe.py at run time; NOT re-staged"},
               "note": "the gate token binds headrow_probe.py and test_headrow_probe.py by sha256 (their values here must equal gate_token.json bindings.code_sha256)"}, f, indent=1, sort_keys=True)
    f.write("\n")
entries.append(("code/CODE_SHA256_HEADROW_PROBE.json", f"{STAGE}/code/CODE_SHA256_HEADROW_PROBE.json", "code", ""))

raw = [("gate_token.json", "the minted launch-gate token (5/5 PASS)"), ("cid.json", "C-ID on the real bank"), ("probe_result.json", "every statistic, the per-row calls, the controls, the outcome"),
       ("nullrate.json", "the C2 rule's measured false-RED rate under its null"), ("noise.json", "C-NOISE grid"), ("ares.json", "A-RES"), ("tables.md", "rendered tables of RESULT.md"),
       ("cpos/planted_info.json", "C-POS planted-label info"), ("gate.log", ""), ("cid.log", ""), ("main.log", ""), ("c2.log", ""), ("cpos.log", ""), ("noise.log", ""), ("ares.log", ""),
       ("nullrate.log", ""), ("metrics.log", ""), ("export.log", ""), ("chain.log", ""), ("test_headrow_probe.log", "synthetic test run (39 tests)"), ("c2x.json", "Amendment 2 control (POST-HOC)"), ("c2y.json", "Amendment 3 control (POST-HOC)"), ("index_diag.json", "Amendment 3 diagnostic (POST-HOC)"), ("c2z.json", "Amendment 4 control (POST-HOC)"), ("slot_prior.json", "Amendment 5 diagnostic (POST-HOC)"), ("slot_prior.log", ""), ("c2z_nc.log", ""), ("c2z_ep.log", ""), ("c2x_nc.log", ""), ("c2x_ep.log", ""), ("c2y_nc.log", ""), ("c2y_ep.log", ""), ("c2b.log", ""), ("metrics_premature.log", "PREMATURE metrics run log (c2 incomplete); see RESULT section 8"), ("export_verify_premature.json", "verify of the premature run")]
for n, note in raw:
    p = f"{RUN}/{n}"
    if os.path.exists(p):
        entries.append((f"raw/f10/{n}", cp(p, f"raw/f10/{n}"), "raw", note))
for n in ("oof_proposals_f10.npz", "oof_proposals_f10.meta.json", "export_verify.json"):
    p = f"{STAGE}/raw/f10/{n}"
    if os.path.exists(p):
        entries.append((f"raw/f10/{n}", p, "raw", "Master Mind addendum export (Amendment 1)"))
for n in ("PREREG_HEADROW_PROBE.md", "PREREG_HEADROW_PROBE.sha256", "PREREG_HEADROW_PROBE_AMENDMENT1.md", "PREREG_HEADROW_PROBE_AMENDMENT1.sha256", "PREREG_HEADROW_PROBE_AMENDMENT2.md", "PREREG_HEADROW_PROBE_AMENDMENT2.sha256", "PREREG_HEADROW_PROBE_AMENDMENT3.md", "PREREG_HEADROW_PROBE_AMENDMENT3.sha256", "PREREG_HEADROW_PROBE_AMENDMENT4.md", "PREREG_HEADROW_PROBE_AMENDMENT4.sha256", "PREREG_HEADROW_PROBE_AMENDMENT5.md", "PREREG_HEADROW_PROBE_AMENDMENT5.sha256", "RESULT.md"):
    p = f"{STAGE}/{n}"
    if os.path.exists(p):
        entries.append((n, p, "doc", ""))

one_place = []
for sub in ("oof", "c2", "cpos", "c2x", "c2y", "c2z"):
    d = f"{RUN}/{sub}"
    if os.path.isdir(d):
        for n in sorted(os.listdir(d)):
            one_place.append((f"{d}/{n}", sha(f"{d}/{n}"), os.path.getsize(f"{d}/{n}")))

lines = ["# MANIFEST: HEADROW-PROBE (stage, never push)\n",
         f"Repo target root: `{REL}/`. Staging root: `{OUT}/` (the path column of `paths.json`). Nothing is committed or pushed.\n",
         "| repo path (under the target root) | staged at | sha256 | bytes | class | note |", "|---|---|---|---|---|---|"]
paths = []
for rel, ab, cls, note in entries:
    s = sha(ab)
    lines.append(f"| `{rel}` | `{ab}` | `{s}` | {os.path.getsize(ab)} | {cls} | {note} |")
    paths.append([f"{REL}/{rel}", ab, cls])
lines.append("\n## Single-place artifacts (on disk only; NOT staged; recompute-able from the staged export or by rerunning the stages)\n")
lines.append("| path | sha256 | bytes |")
lines.append("|---|---|---|")
for p, s, b in one_place:
    lines.append(f"| `{p}` | `{s}` | {b} |")
lines.append("\nThe 16 main OOF tables are ALSO inside `raw/f10/oof_proposals_f10.npz` (staged). The C2, C-POS and post-hoc control tables (`c2/`, `cpos/`, `c2x/`, `c2y/`, `c2z/`; about 360 MB) exist ONLY on this disk; their statistics are in `probe_result.json`, `c2x.json`, `c2y.json`, `c2z.json`.")
with open(f"{STAGE}/MANIFEST.md", "w", encoding="utf-8", newline="\n") as f:
    f.write("\n".join(lines) + "\n")
paths.append([f"{REL}/MANIFEST.md", f"{STAGE}/MANIFEST.md", "doc"])
with open(f"{OUT}/paths.json", "w", encoding="utf-8", newline="\n") as f:
    json.dump(paths, f, indent=1)
    f.write("\n")
print("entries", len(entries), "one-place", len(one_place), "paths", len(paths))
