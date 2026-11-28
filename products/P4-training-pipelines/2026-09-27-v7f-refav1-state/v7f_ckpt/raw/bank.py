"""Bank the F3 package into the D: repo (copy files, write the apply proof), then `git add` each file by EXACT path and
verify each by blob comparison (index blob == hash-object, both 40 chars, else INCONCLUSIVE). Never commits."""
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path("D:/Projects/TanitAD")
PKG_REL = "products/P4-training-pipelines/2026-09-27-v7f-refav1-state/v7f_ckpt"
PKG = REPO / PKG_REL
W = Path("C:/Users/Admin/v7f_ckpt")
G = Path("C:/Users/Admin/v7f_ckpt_gate")


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def cp(src, rel):
    dst = PKG / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    return rel


files: list[str] = []
# --- code ------------------------------------------------------------------------------------------------------
files.append(cp(W / "pkg/code/apply_f3_exact_resume.py", "code/apply_f3_exact_resume.py"))
for rel in ("scripts/train_v6_staged.py", "scripts/launch_gate.py", "tests/test_launch_gate_v7f.py",
            "tests/test_v6_exact_resume.py"):
    files.append(cp(W / "stack" / rel, f"code/fix/stack/{rel}"))
# unified diffs vs the pre-F3 files (for review; the script is the deliverable)
for rel, base in (("scripts/train_v6_staged.py", "train_v6_staged.py"), ("scripts/launch_gate.py", "launch_gate.py"),
                  ("tests/test_launch_gate_v7f.py", "test_launch_gate_v7f.py")):
    d = subprocess.run(["git", "-c", "core.autocrlf=false", "diff", "--no-index", "--text",
                        str(W / "pristine" / base), str(W / "stack" / rel)], capture_output=True)
    out = PKG / f"code/diffs_vs_pre_f3/{Path(rel).name}.diff"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(d.stdout)
    files.append(f"code/diffs_vs_pre_f3/{Path(rel).name}.diff")
# --- raw -------------------------------------------------------------------------------------------------------
raw_src = {
    "raw/probe_before_after.json": W / "work/probe_before_after.json",
    "raw/probe_before_after.py": W / "work/probe_before_after.py",
    "raw/discovery_F3b_rng_fix_only.txt": W / "work/discovery_F3b_rng_fix_only.txt",
    "raw/mutations/mutation_record.json": W / "work/mutations/mutation_record.json",
    "raw/mutations/mutate_source.py": W / "work/mutate_source.py",
    "raw/run_suites.py": W / "work/run_suites.py",
    "raw/run_suites_before.log": W / "work/run_suites_v1.log",
    "raw/run_suites_after_final.log": W / "work/run_suites_after_final.log",
    "raw/apply_proof.py": W / "work/apply_proof.py",
    "raw/bank.py": W / "work/bank.py",
}
for name in ("M1_restore_call_deleted", "M2_lr_reset_deleted", "M3_capture_not_saved"):
    raw_src[f"raw/mutations/{name}.pytest.summary.txt"] = W / f"work/mutations/{name}.pytest.summary.txt"
for p in sorted((W / "work/suites").glob("*.txt")):
    raw_src[f"raw/pytest/{p.name}"] = p
for tag, d in (("gate_after_sw", "after_sw"), ("gate_after_st", "after_st"),
               ("gate_oldprotocol_sw", "probe_oldprotocol_sw"), ("gate_oldprotocol_st", "probe_oldprotocol_st")):
    raw_src[f"raw/{tag}/cli.log"] = G / f"runs/{d}.cli.log"
    for ev in sorted((G / f"runs/{d}/evidence").glob("G-*.json")):
        raw_src[f"raw/{tag}/evidence/{ev.name}"] = ev
    tok = next((G / f"runs/{d}").glob("INCOMPLETE_*.json"), None)
    if tok is not None:
        t = json.loads(tok.read_text(encoding="utf-8"))
        summ = {"_read": "a SUMMARY of a dev-box REHEARSAL token (the token itself is NOT banked). Signed with a "
                         "throwaway test key outside the tree, fake commit 7f7f..., never verifiable.",
                "verdict": t.get("verdict"), "checks_in_token": {k: v.get("status") if isinstance(v, dict) else v
                                                                 for k, v in (t.get("checks") or {}).items()},
                "token_reasons": t.get("reasons"), "rehearsal": t.get("rehearsal")}
        out = PKG / f"raw/{tag}/rehearsal_verdicts.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summ, indent=1), encoding="utf-8")
        files.append(f"raw/{tag}/rehearsal_verdicts.json")
for rel, src in raw_src.items():
    files.append(cp(src, rel))
# gate argv files used by the CLI runs (the gate agent's launch argv)
files.append(cp(G / "argv_st.json", "raw/gate_launch_argv_st.json"))
files.append(cp(G / "argv_sw.json", "raw/gate_launch_argv_sw.json"))
apply_proof = json.loads((W / "work/apply_proof.json").read_text(encoding="utf-8"))
(PKG / "raw/apply_proof.json").write_text(json.dumps(apply_proof, indent=1), encoding="utf-8")
files.append("raw/apply_proof.json")
files.append(cp(W / "pkg/RESULT.md", "RESULT.md"))
files = sorted(set(files))

# --- stage by EXACT path, verify by blob comparison ------------------------------------------------------------
ver = {}
for rel in files:
    path = f"{PKG_REL}/{rel}"
    r = subprocess.run(["git", "-C", str(REPO), "add", "--", path], capture_output=True, text=True)
    st = subprocess.run(["git", "-C", str(REPO), "ls-files", "--stage", "--", path], capture_output=True, text=True)
    ho = subprocess.run(["git", "-C", str(REPO), "hash-object", "--", path], capture_output=True, text=True)
    idx = (st.stdout.split() + ["", ""])[1] if st.stdout.strip() else ""
    wt = ho.stdout.strip()
    if len(idx) != 40 or len(wt) != 40:
        v = "INCONCLUSIVE"
    else:
        v = "VERIFIED" if idx == wt else "MISMATCH"
    ver[path] = {"verdict": v, "index_blob": idx, "worktree_blob": wt, "add_rc": r.returncode,
                 "sha256": sha(REPO / path)}
    print(v, path, flush=True)
(W / "work/staging_verification.json").write_text(json.dumps(ver, indent=1), encoding="utf-8")
bad = [p for p, v in ver.items() if v["verdict"] != "VERIFIED"]
print("NOT VERIFIED:", bad)
sys.exit(1 if bad else 0)
