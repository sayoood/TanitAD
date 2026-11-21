"""Write LANDING_READY_X10.txt from the files as they are on disk (md5s are computed, never typed).
    python make_landing.py
"""
import hashlib
import os
import subprocess

PK = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).replace("\\", "/")
PFX = "FlyWheels/TanitAD_DataFlyWheel/incoming/2026-10-04-x10-pose-timing/"
GD = "C:/Users/Admin/tanitad-push/.git"


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def blob(p):
    return subprocess.run(["git", "hash-object", "--no-filters", p], capture_output=True, text=True, check=True).stdout.strip()


def tip(path):
    r = subprocess.run(["git", "--git-dir", GD, "rev-parse", f"agent/arch-inf-20260803:{path}"], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else "(not at tip: NEW file)"


L = []
L.append("## X10")
L.append("# Work item X10 (refcv8): pose-to-image timing. Repo root = D:/Projects/TanitAD. Package prefix: " + PFX)
L.append("# format: <md5>  <repo-relative path>  -- <what>")
L.append("# Nothing here is staged or committed (Master Mind is the sole committer). Built on lander tip agent/arch-inf-20260803 @ acc6180;")
L.append("# the three base blobs of the MODIFIED files (and the 7 source files DESIGN.md cites) were re-verified UNCHANGED at the tip c51d1cb. Line endings follow each file's TIP BLOB: refc_v3_train.py is CRLF,")
L.append("# declared_vs_built.py, test_declared_vs_built.py and the three new files are LF. D:'s working tree is OLDER than the tip for refc_v3_train.py (and v2_dataset.py) -- land onto the tip, not D:.")
L.append("")
L.append("## A. REPO FILES TO LAND  (source on disk: <package>/code/fix/<same path>)")
repo = [
    ("stack/tanitad/data/pose_sync.py", "NEW. The correction: grid replica, closed form, fractional-row re-sampling, sidecar reader (8 refusals), window view."),
    ("stack/scripts/build_pose_sync_sidecar.py", "NEW. Builds the sidecar from camera timestamps.parquet (K1-K4 controls; refuses a foreign timeline)."),
    ("stack/tests/test_pose_sync.py", "NEW. 27 tests: literal known values, zero-offset bit-identity, flipped-sign + per-row deliberate regressions, refusals, builder."),
    ("stack/scripts/refc_v3_train.py", "MODIFIED (+109/-0, CRLF kept): V3Dataset.enable_pose_sync/_pose_sync_apply, one `if self.pose_sync is not None` hook, --pose-sync-sidecar (default off), train+eval enable, config.json `pose_sync`."),
    ("stack/tanitad/train/declared_vs_built.py", "MODIFIED (+1/-0): registers the new dest `pose_sync_sidecar`."),
    ("stack/tests/test_declared_vs_built.py", "MODIFIED (+2/-0): the pinned registry size 256 -> 257 (the only regression the neighbour run found)."),
]
for rel, what in repo:
    p = f"{PK}/code/fix/{rel}"
    L.append(f"{md5(p)}  {rel}  -- {what}")
    L.append(f"#   fix blob (git hash-object --no-filters) {blob(p)} ; tip base blob {tip(rel)}")
L.append("")
L.append("## B. ARTIFACT TO SHIP  (regenerable: code/fix/stack/scripts/build_pose_sync_sidecar.py + the dev-box camera timestamps)")
for rel, what in [("raw/refcv6_pose_sync_sidecar.jsonl", "4,508/4,508 corpus clips (4,369 train + 139 eval), no clip ids, 5.4 MB. Copy to Thor /home/nvidia/data/refcv6_pose_sync_sidecar.jsonl and D:/refcv6_eval_kit/data/ ; launch flag --pose-sync-sidecar <path>."),
                  ("raw/refcv6_pose_sync_sidecar.jsonl.meta.json", "its provenance (input manifest md5s, K1 self-test, K4 worst |dt| 6.2e-9 s, delta stats).")]:
    L.append(f"{md5(PK + '/' + rel)}  {PFX}{rel}  -- {what}")
L.append("")
L.append("## C. PACKAGE (research record; lands under the prefix above)")
for rel, what in [
    ("DESIGN.md", "trace from source with file:line, what the correction changes, rejected options, warm-start statement"),
    ("RESULT.md", "measured offset, target shift, POSE-SYNC row, the decision for the Master Mind"),
    ("code/x10_measure.py", "the measurement (offset over 4,508 clips; target shift on eval139 + seeded 400-clip train sample; controls K1-K3)"),
    ("code/x10_dataset_e2e.py", "the REAL V3Dataset hook OFF vs ON over all 23,772 eval windows; POSE-SYNC vs the 100 Hz log"),
    ("code/x10_pixel_check.py", "pixel-level check of the premise (cached row == camera frame frame_idx[k], not the nearest one) on 4 rows / 2 clips"),
    ("code/summarize_suite.py", "compares the affected-test run on the overlay tree with the pure tip tree"),
    ("raw/x10_pixel_check.json", "its result: 4/4 rows best-match offset 0"),
    ("code/make_tables.py", "renders the two raw JSONs into raw/RESULT_tables_generated.md"),
    ("code/mutation_check.py", "7 re-introduced defects against test_pose_sync.py"),
    ("code/make_trainer_patch.py", "regenerates the three modified files from the tip blobs, byte-identically (provenance of the fix tree)"),
    ("code/overlay_test_tree.sh", "builds the clean tip+fix test tree (git archive of the tip; never touches D: or G:)"),
    ("code/make_landing.py", "writes this file"),
    ("raw/x10_measure.json", "MEASURED: offset + target-shift statistics, controls, cross-checks (per-clip rows carry sha12 only)"),
    ("raw/x10_measure.log", "its log"),
    ("raw/x10_dataset_e2e.json", "MEASURED: hook OFF vs ON, POSE-SYNC, label flips, encoder-channel shifts"),
    ("raw/RESULT_tables_generated.md", "generated tables (T1-T4)"),
    ("raw/mutation_check.json", "7/7 mutations caught"),
    ("raw/suite_regression.json", "affected-test regression: overlay tree vs pure tip tree (0 regressions; 33 identical pre-existing ids)"),
    ("raw/suite_overlay_final.log", "pytest log, tip + fix: 14 failed / 1776 passed / 10 skipped / 19 errors"),
    ("raw/suite_base.log", "pytest log, pure tip: the identical counts"),
]:
    p = PK + "/" + rel
    if os.path.exists(p):
        L.append(f"{md5(p)}  {PFX}{rel}  -- {what}")
    else:
        L.append(f"MISSING  {PFX}{rel}  -- {what}")
L.append("")
L.append("## D. NOT FOR GIT / single copies")
L.append("# scratchpad test trees under C:/Users/Admin/AppData/Local/Temp/claude/.../scratchpad/x10/{tree,tree2,tree3,tree_base,mutscr}: disposable (rebuild with code/overlay_test_tree.sh).")
L.append("# D3's train_v2manifest.pt copy (scratchpad/d3/, 24 MB, regenerable) was READ, not modified.")
L.append("# No Thor, no GPU, no G: access. Every python / pytest process started by this stream has exited (checked by command line before this file was written).")
L.append("# Other agents' pytest processes were running on the dev box at that moment; they are not this stream's and were not touched.")
open(f"{PK}/LANDING_READY_X10.txt", "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")
print("wrote LANDING_READY_X10.txt")
