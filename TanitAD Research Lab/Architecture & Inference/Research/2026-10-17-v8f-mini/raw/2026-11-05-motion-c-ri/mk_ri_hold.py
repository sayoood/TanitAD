"""Build qland hold_motion_c_ri: the (c) design note, the RI tool package and the RI-C0 read, plus the GOALS_AND_CLAIMS row.
usage: python -I mk_ri_hold.py <tip_sha>"""
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

MC = Path("E:/Projects/TanitAD-artifacts/dagland/motion_c")
GIT = "C:/Users/Admin/tanitad-push"
REL = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/2026-11-05-motion-c-ri"
GAC = "Project Steering/GOALS_AND_CLAIMS.md"
HOLD = Path("C:/Users/Admin/qland/hold_motion_c_ri")
ROW = ("| **H-RI-C0-1** (2026-11-05, RI-C0 read, option (c) instrument) | Does the C0 base (`M2b_s{0,1}_recft_C0`) depend on the logged "
       "camera clock of T1-kin? New `--scene reindexed` (tool `v8f_t1_eval_ri.py` sha256 0417c838..., the FRAMEFREEZE tool + a patch; "
       "12 tests, 3 mutations RED): after t0 each history row shows the logged frame where the human was at the model's OWN along-track "
       "position, never a later frame (longitudinal-only approximate re-perception, not T2). Card PREREG_RI_C0.md (sha256 942b6391..., "
       "frozen before any RI rollout). Motivation (DESIGN_C.md): a motion-reading arm would read the HUMAN's speed off logged frames, so "
       "every (c) / (a) arm is decided on `reindexed`. | **MEASURED 2026-11-05 12:18: CLOCK-INSENSITIVE** at both seeds (T1-kin, paired "
       "episode-cluster bootstrap B = 2,000, seed 0, 141 episodes / 2,399 windows): M1 crawl share RIC0 − C0 +0.0001 [−0.0005, +0.0007] / "
       "+0.0002 [−0.0005, +0.0008], M1 excess vs P10 reindexed / logged +0.0533 / +0.0532 and +0.0596 / +0.0594 (ratio 1.002 / 1.003); "
       "ADE6 RIC0 − C0 +0.0018 / +0.0012 m; composite RIC0C vs P10 NOT DOMINANT exactly as logged (D1, D3 true; D2 fails on M1 alone, "
       "+0.0591 / +0.0619). Non-vacuous: 7,573 to 8,028 rows reindexed per rollout; past rows bit-identical; the floor's kinematic rows "
       "bit-identical between modes (its S-POS rows read S's frames and differ). | SUPPORTED (instrument read): logged-frame T1-kin reads of "
       "frozen-feature arms stand; `reindexed` decides the (c) and (a) arms. | "
       f"`{REL}/` (ri_read/RESULT_RI_C0.md, ri_read/raw/RI_C0_READ.json, ri/, DESIGN_C.md) |\n")


def sha(b):
    return hashlib.sha256(b).hexdigest()


tip = sys.argv[1]
assert len(tip) == 40 and not HOLD.exists()
have = subprocess.run(["git", "-C", GIT, "ls-tree", "-r", "--name-only", tip, REL + "/"], capture_output=True, text=True, check=True).stdout.strip()
assert not have, f"{REL} already on the tip"
files = [MC / "DESIGN_C.md", MC / "BUILD_BRIEF_CA.md", MC / "mk_ri_hold.py"]
for sub in ("ri", "ri_read"):
    files += sorted(p for p in (MC / sub).rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc")
rels = [p.relative_to(MC).as_posix() for p in files]
for p, r in zip(files, rels):
    b = p.read_bytes()
    if r.endswith((".py", ".md", ".sh")):
        assert b"\r\n" not in b, f"CRLF in {r}"
    d = HOLD / "stage" / REL / r
    d.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(p, d)
man = "".join(f"./{r} | {p.stat().st_size} | {sha(p.read_bytes())}\n" for p, r in zip(files, rels))
(HOLD / "stage" / REL / "MANIFEST.txt").write_bytes(man.encode())
old = subprocess.run(["git", "-C", GIT, "cat-file", "blob", f"{tip}:{GAC}"], capture_output=True, check=True).stdout
assert old.endswith(b"\n") and b"\r\n" not in old and b"H-RI-C0-1" not in old
g = HOLD / "mods" / GAC
g.parent.mkdir(parents=True, exist_ok=True)
g.write_bytes(old + ROW.encode())
print("staged", len(files) + 1, "files; GOALS row appended; hold", HOLD)
