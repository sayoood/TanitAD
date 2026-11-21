"""Bank the final files into the deliverable dir (LF), write the diffs (vs the tip blobs and vs the v7F merge
package), and PROVE each diff: `git -c core.autocrlf=false apply` onto a scratch copy of its base reproduces the
banked file byte-for-byte; plus the diffs vs the coordinator's CURRENT merge copy (C:/Users/Admin/v7f_merge,
LF-normalised). Writes raw/bank_proof.json."""
import difflib
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

REPO = Path(r"D:/Projects/TanitAD")
TIP = "c36b6dddc18ef8d43debc610591b58432748e104"
TM = Path(r"C:/Users/Admin/v7f_gate/tree_m")
DEL = Path(r"D:/Projects/TanitAD/products/P4-training-pipelines/2026-09-27-v7f-refav1-state/v7f_gate")
MERGE_PKG = Path(r"D:/Projects/TanitAD/products/P4-training-pipelines/2026-09-27-v7f-refav1-state/v7f_merge/"
                 r"code/fix")
#: the coordinator's CURRENT working merge copy (holds my previously banked files + the nav fix)
MERGE_COPY = Path(r"C:/Users/Admin/v7f_merge")
FILES = ["stack/scripts/launch_gate.py", "stack/tanitad/train/declared_vs_built_v6.py",
         "stack/tests/test_launch_gate_v7f.py"]


def tip_blob(rel: str) -> bytes | None:
    p = subprocess.run(["git", "-c", "core.autocrlf=false", "cat-file", "-p", f"{TIP}:{rel}"],
                       cwd=REPO, capture_output=True)
    return p.stdout if p.returncode == 0 else None


def udiff(rel: str, old: bytes | None, new: bytes) -> str:
    a = old.decode("utf-8").splitlines(keepends=True) if old is not None else []
    b = new.decode("utf-8").splitlines(keepends=True)
    head = f"diff --git a/{rel} b/{rel}\n"
    if old is None:
        head += "new file mode 100644\n"
    body = "".join(difflib.unified_diff(a, b, fromfile=f"a/{rel}" if old is not None else "/dev/null",
                                        tofile=f"b/{rel}", n=3))
    return head + body


def prove(rel: str, base: bytes | None, diff: str, want: bytes) -> dict:
    with tempfile.TemporaryDirectory() as td:
        t = Path(td)
        if base is not None:
            (t / rel).parent.mkdir(parents=True, exist_ok=True)
            (t / rel).write_bytes(base)
        (t / "x.diff").write_bytes(diff.encode("utf-8"))
        p = subprocess.run(["git", "-c", "core.autocrlf=false", "apply", "--whitespace=nowarn",
                            "x.diff"], cwd=t, capture_output=True, text=True)
        got = (t / rel).read_bytes() if (t / rel).is_file() else b""
        return {"apply_rc": p.returncode, "stderr": p.stderr[-300:],
                "reproduces_banked_file_byte_for_byte": got == want,
                "sha256": hashlib.sha256(want).hexdigest()}


proof = {"tip": TIP, "files": {}}
for rel in FILES:
    new = (TM / rel).read_bytes().replace(b"\r\n", b"\n")          # LF: the tip blobs' convention
    dst = DEL / "code" / "fix" / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(new)
    rec = {"banked": str(dst), "lf_only": b"\r\n" not in new, "bytes": len(new)}
    base = tip_blob(rel)
    d = udiff(rel, base, new)
    dp = DEL / "code" / "diffs_vs_c36b6ddd" / (rel.replace("/", "_") + ".diff")
    dp.parent.mkdir(parents=True, exist_ok=True)
    dp.write_bytes(d.encode("utf-8"))
    rec["vs_tip"] = {"diff": str(dp), "base": "absent at the tip (new file)" if base is None else
                     "tip blob", **prove(rel, base, d, new)}
    mbase = (MERGE_PKG / rel).read_bytes() if (MERGE_PKG / rel).is_file() else None
    if mbase is not None:
        d2 = udiff(rel, mbase, new)
        dp2 = DEL / "code" / "diffs_vs_v7f_merge" / (rel.replace("/", "_") + ".diff")
        dp2.parent.mkdir(parents=True, exist_ok=True)
        dp2.write_bytes(d2.encode("utf-8"))
        rec["vs_merge"] = {"diff": str(dp2), "base": "the v7f_merge package's banked file",
                           **prove(rel, mbase, d2, new)}
    else:
        rec["vs_merge"] = "the merge does not touch this file: the vs-tip diff applies to the merge as-is"
    cbase_raw = (MERGE_COPY / rel).read_bytes() if (MERGE_COPY / rel).is_file() else None
    cbase = None if cbase_raw is None else cbase_raw.replace(b"\r\n", b"\n")
    dp3 = DEL / "code" / "diffs_vs_merge_copy_current" / (rel.replace("/", "_") + ".diff")
    dp3.parent.mkdir(parents=True, exist_ok=True)
    common = {"base": str(MERGE_COPY / rel),
              "base_sha256_lf": None if cbase is None else hashlib.sha256(cbase).hexdigest(),
              "base_had_crlf": None if cbase_raw is None else (b"\r\n" in cbase_raw)}
    if cbase == new:
        # IDENTICAL to the current merge copy: no diff file (an empty patch is not a patch)
        if dp3.is_file():
            dp3.unlink()
        rec["vs_merge_copy_current"] = {"diff": None, "identical": True, **common,
                                        "reproduces_banked_file_byte_for_byte": True,
                                        "sha256": hashlib.sha256(new).hexdigest()}
    else:
        d3 = udiff(rel, cbase, new)
        dp3.write_bytes(d3.encode("utf-8"))
        rec["vs_merge_copy_current"] = {"diff": str(dp3), "identical": False, **common,
                                        **prove(rel, cbase, d3, new)}
    proof["files"][rel] = rec
(DEL / "raw").mkdir(parents=True, exist_ok=True)
(DEL / "raw" / "bank_proof.json").write_text(json.dumps(proof, indent=1), encoding="utf-8")
print(json.dumps({k: {"tip": v["vs_tip"]["reproduces_banked_file_byte_for_byte"],
                      "merge": (v["vs_merge"]["reproduces_banked_file_byte_for_byte"]
                                if isinstance(v["vs_merge"], dict) else "n/a"),
                      "merge_copy_current": v["vs_merge_copy_current"]["reproduces_banked_file_byte_for_byte"]}
                  for k, v in proof["files"].items()}, indent=1))
