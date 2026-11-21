"""Stage the P6-HN-S landing package from the edit tree into C:/Users/Admin/hn34/out/<repo path>, write paths.json and the blob table.  Re-runnable (overwrites).  NO git."""
import hashlib, json, os, shutil, sys
from pathlib import Path

TREE = Path("C:/Users/Admin/hn34/tree")
BASE = Path("C:/Users/Admin/hn34/base")
OUT = Path("C:/Users/Admin/hn34/out")
WORK = Path("C:/Users/Admin/hn34/work")
LOGS = Path("C:/Users/Admin/hn34/logs")
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
IMPL = f"{PK}/raw/2026-10-18-p6hn-prereg/impl3"
EDITED = [f"{PK}/refe/{n}" for n in ("scorer_finetune.py", "carvefeed_layer.py", "gate_checks.py", "launch_gate.py", "test_launch_gate.py", "test_dump_scores.py", "test_carve_feed.py")]
NEW = [f"{PK}/refe/{n}" for n in ("hnscreen_layer.py", "test_hn_screen_wiring.py")] + [f"{PK}/refe/fixtures/{n}" for n in ("scorer_finetune_pre_hn.py.txt", "hn_wiring_evidence.json")]


def blob(b: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()


rows, paths = [], []


def put(repo_path: str, src: Path, kind: str, base_blob: str = ""):
    dst = OUT / repo_path
    dst.parent.mkdir(parents=True, exist_ok=True)
    b = src.read_bytes()
    assert b"\r" not in b, ("CRLF in " + repo_path)
    dst.write_bytes(b)
    assert dst.read_bytes() == b
    paths.append([repo_path, str(dst).replace("\\", "/"), "raw"])
    rows.append((repo_path, kind, base_blob, blob(b), len(b)))


for rp in EDITED:
    bb = blob((BASE / rp).read_bytes())
    put(rp, TREE / rp, "EDIT", bb)
for rp in NEW:
    put(rp, TREE / rp, "NEW")
# provenance: generators (the layer's docstring cites impl3/gen_layer.py), the blocks, the patch scripts, the batch drivers, the logs
GEN = [("gen_layer.py", WORK / "apply_sf.py"), ("RESULT.md", OUT / "RESULT.md")]
if (WORK / "regression_base_vs_new.log").exists():
    GEN.append(("regression_base_vs_new.log", WORK / "regression_base_vs_new.log"))
for p in sorted((WORK / "blocks").glob("*.py")):
    GEN.append((f"gen_blocks/{p.name}", p))
for n in ("patch_top1.py", "patch_gate_checks.py", "patch_launch_gate.py", "patch_tlg.py", "patch_tlg2.py", "patch_tlg3.py", "patch_fix1.py", "hn_static_block.py", "test_tail.py",
          "run_batch.py", "run_both.sh", "pass2.sh", "tests_common.txt", "stage_out.py", "compare.py"):
    if (WORK / n).exists():
        GEN.append((f"gen/{n}", WORK / n))
for name, src in GEN:
    put(f"{IMPL}/{name}", src, "PROV")
LOGSTAGE = []
for d in ("batch_new_pass1", "batch_base", "batch_new", "batch_base_long", "batch_new_env", "batch_base_env"):
    for p in sorted((LOGS / d).glob("*.log")) if (LOGS / d).exists() else []:
        LOGSTAGE.append((f"logs/{d}/{p.name}", p))
    if (LOGS / d / "summary.json").exists():
        LOGSTAGE.append((f"logs/{d}/summary.json", LOGS / d / "summary.json"))
for n in ("wiring_new_run3.log", "lg_new_static5.log"):
    if (LOGS / n).exists():
        LOGSTAGE.append((f"logs/{n}", LOGS / n))
for name, src in LOGSTAGE:
    t = src.read_bytes().replace(b"\r\n", b"\n")
    tmp = OUT / "_tmp_stage"
    tmp.write_bytes(t)
    put(f"{IMPL}/{name}", tmp, "LOG")
(OUT / "_tmp_stage").unlink()
(OUT / "paths.json").write_text(json.dumps(paths, indent=1), encoding="utf-8", newline="\n")
md = ["| repo path | kind | tip (base) blob | new blob | bytes |", "|---|---|---|---|---|"]
for rp, kind, bb, nb, n in rows:
    if kind in ("EDIT", "NEW"):
        md.append(f"| `{rp.split(PK + '/')[-1]}` | {kind} | `{bb or '-'}` | `{nb}` | {n} |")
(WORK / "blob_table.md").write_text("\n".join(md) + "\n", encoding="utf-8", newline="\n")
print("\n".join(md))
print("staged", len(paths), "files; paths.json written")
