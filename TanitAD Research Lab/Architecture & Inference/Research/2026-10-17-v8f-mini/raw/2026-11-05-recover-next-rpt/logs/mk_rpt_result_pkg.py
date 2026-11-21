"""Assemble the RPT result package (addendum 4) after the read: RESULTS.md, the reader JSON, the run records and the logs, laid out
as the RAPR-P01 result package. Lands into the folder the freeze package already holds (raw/2026-11-05-recover-next-rpt/), so the
new files go to stage/ and the folder's MANIFEST.txt goes to mods/ (append-only: the landed rows first, byte for byte, then the new ones).
usage: python -I mk_rpt_result_pkg.py <tip_sha>   (needs RESULTS.md written by gen_rpt_results.py and logs/box2_rnext_chain_rpt.log)"""
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

NX = Path("E:/Projects/TanitAD-artifacts/dagland/recover_next")
HERE = NX / "rpt_result"
PKG = HERE / "pkg"
RUNS = Path("E:/Projects/TanitAD-artifacts/v8f-recover")
MRK = RUNS / "rnext_markers"
OUT = Path("C:/Users/Admin/rr_scratch/rnext/out_RPT")
GIT = "C:/Users/Admin/tanitad-push"
REL = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/2026-11-05-recover-next-rpt"
HOLD = Path("C:/Users/Admin/qland/hold_rnext_rpt_result")


def sha(b):
    return hashlib.sha256(b).hexdigest()


def put(src, rel):
    dst = PKG / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


tip = sys.argv[1]
assert len(tip) == 40
assert not PKG.exists(), f"{PKG} exists"
assert not HOLD.exists(), f"{HOLD} exists"
landed = subprocess.run(["git", "-C", GIT, "ls-tree", "-r", "--name-only", tip, REL + "/"], capture_output=True, text=True, check=True).stdout.split("\n")
landed = {p[len(REL) + 1:] for p in landed if p}
assert "PREREG_RECOVER_NEXT_ADDENDUM4_RPT.md" in landed and "MANIFEST.txt" in landed, sorted(landed)[:5]
put(HERE / "RESULTS.md", "RESULTS.md")
put(NX / "read/raw/RECOVER_READ_RPT.json", "RECOVER_READ_RPT.json")
logs = {"logs/launch_rpt.out": NX / "launch_rpt.out", "logs/wake_pull_rpt.out": NX / "wake_pull_rpt.out", "logs/pull_rpt.out": NX / "pull_rpt.out",
        "logs/precheck_RPT.out": NX / "read/logs/precheck_RPT.out", "logs/chain_RPT.out": NX / "read/logs/chain_RPT.out",
        "logs/box2_rnext_chain_rpt.log": HERE / "logs/box2_rnext_chain_rpt.log", "logs/wake_pull_rpt.sh": NX / "wake_pull_rpt.sh",
        "logs/mk_rpt_result_pkg.py": HERE / "mk_rpt_result_pkg.py", "NOTE_HISTORY_WINDOW.md": HERE / "NOTE_HISTORY_WINDOW.md"}
for rel, src in logs.items():
    put(src, rel)
lines = ["# measured at package build: sha256 of the pulled ckpt_last.pt vs the box-2 RUN_<tag>.done marker's evidence.ckpt_sha256"]
import json
for s in (0, 1):
    run = f"M2b_s{s}_recft_RPT"
    for f in ("SHA256SUMS.txt", "argv.json", "gate_selfcheck.json", "selfcheck.json", "summary.json"):
        put(RUNS / run / f, f"runs/{run}/{f}")
    put(MRK / f"RUN_RPT_{s}.done", f"runs/RUN_RPT_{s}.done")
    put(MRK / f"gate_token_RPT_{s}.json", f"runs/gate_token_RPT_{s}.json")
    want = json.load(open(MRK / f"RUN_RPT_{s}.done", encoding="utf-8"))["evidence"]["ckpt_sha256"]
    got = sha((RUNS / run / "ckpt_last.pt").read_bytes())
    lines.append(f"RPT_{s} marker {want} pulled {got} {'MATCH' if want == got else 'MISMATCH'}")
    assert want == got
(PKG / "logs/ckpt_sha_vs_markers.txt").write_bytes(("\n".join(lines) + "\n").encode())
done = sorted(OUT.glob("done_*.txt"))
assert done, f"no rollout done markers in {OUT}"
(PKG / "logs/rollout_done_markers_sha256.txt").write_bytes(("".join(f"{sha(p.read_bytes())} *{p.name}\n" for p in done)).encode())
new = sorted(p.relative_to(PKG).as_posix() for p in PKG.rglob("*") if p.is_file())
clash = [p for p in new if p in landed]
assert not clash, f"paths already on the tip: {clash}"
for p in new:
    b = (PKG / p).read_bytes()
    if p.endswith((".py", ".md", ".sh")):
        assert b"\r\n" not in b, f"CRLF in {p}"
# stage: the new files; mods: the folder MANIFEST, append-only
for p in new:
    d = HOLD / "stage" / REL / p
    d.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(PKG / p, d)
old = subprocess.run(["git", "-C", GIT, "cat-file", "blob", f"{tip}:{REL}/MANIFEST.txt"], capture_output=True, check=True).stdout
assert old.endswith(b"\n")
rows = "".join(f"./{p} | {(PKG / p).stat().st_size} | {sha((PKG / p).read_bytes())}\n" for p in new)
m = HOLD / "mods" / REL / "MANIFEST.txt"
m.parent.mkdir(parents=True, exist_ok=True)
m.write_bytes(old + rows.encode())
print("package files", len(new), "-> hold", HOLD)
