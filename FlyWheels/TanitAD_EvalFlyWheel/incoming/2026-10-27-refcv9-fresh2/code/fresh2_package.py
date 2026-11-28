"""fresh2_package.py -- stage the FRESH2 deliverables for the Master Mind's lander (never commits, never pushes, never lands).
  -> C:/Users/Admin/qland/hold_fresh2_read/code/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-27-refcv9-fresh2/{code/*, result/*, select/*_sha12.json, READING_FRESH2.*, DEVIATIONS.md, LH_G_FROZEN.*, MANIFEST.sha256}
     + expect.tsv (<repo path>\tNEW) + msg.txt.  Logs are NOT staged (upstream builders print clip-id prefixes).  NEW file (FRESH2 agent, 2026-10-27).
"""
import hashlib
import shutil
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent
HOLD = Path("C:/Users/Admin/qland/hold_fresh2_read")
REL = "FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-27-refcv9-fresh2"
DST = HOLD / "code" / REL
CODE = ["fresh2_make_root.py", "fresh2_build_inputs.py", "l1c_run.py", "fresh2_refusals.py", "fresh2_split.py", "chain_build_rest.sh", "fetch_fresh2_obstacle.py",
        "chain_label_pipeline_fresh2.sh", "fresh2_ztac_roll.py", "remap_fresh2.json", "chain_roll.sh", "fresh2_derive.py", "fresh2_lhg_fit.py", "fresh2_stage1.py",
        "fresh2_content_check.py", "fresh2_episode_check.py", "chain_after_build.sh", "fresh2_package.py"]
TOP = ["READING_FRESH2.md", "READING_FRESH2.sha256", "DEVIATIONS.md"]
LHG = [Path("F:/l1c/fresh2/lhg/LH_G_FROZEN.pt"), Path("F:/l1c/fresh2/lhg/LH_G_FROZEN.sha256")]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    if HOLD.exists():
        shutil.rmtree(HOLD)
    files = []
    for n in CODE:
        files.append((SRC / "code" / n, f"code/{n}"))
    for p in sorted((SRC / "result").glob("*")):
        if p.suffix in (".json", ".md"):
            files.append((p, f"result/{p.name}"))
    for p in sorted((SRC / "select").glob("*_sha12.json")):
        files.append((p, f"select/{p.name}"))
    for n in TOP:
        files.append((SRC / n, n))
    for p in LHG:
        files.append((p, p.name))
    for s, rel in files:
        d = DST / rel
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(s, d)
        assert sha(s) == sha(d), rel
    man = "".join(f"{sha(DST / rel)} *{rel}\n" for _, rel in sorted(files, key=lambda x: x[1]))
    (DST / "MANIFEST.sha256").write_text(man, encoding="utf-8", newline="\n")
    rels = sorted([rel for _, rel in files] + ["MANIFEST.sha256"])
    (HOLD / "expect.tsv").write_text("".join(f"{REL}/{r}\tNEW\n" for r in rels), encoding="utf-8", newline="\n")
    msg = Path(sys.argv[1]).read_text(encoding="utf-8") if len(sys.argv) > 1 else ""
    assert msg.rstrip().endswith("Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"), "msg.txt must end with the attribution line"
    (HOLD / "msg.txt").write_text(msg, encoding="utf-8", newline="\n")
    print(f"ZZPKGZZ {len(rels)} files -> {DST}")


if __name__ == "__main__":
    main()
