"""Copy WP-C's changed / new tracked files from the working extraction into ``code/fix/<repo path>`` (tip-blob EOL = LF) and write the manifest.

* MODIFIED files: the delivered bytes are the working file with CRLF -> LF (``git archive`` hands out CRLF; the tip blobs are LF), and the
  script ASSERTS that the delivered file differs from the tip blob ONLY by the intended hunks (every tip line survives: a descendant check).
* NEW files: copied as they are (LF); binary ``.npz`` fixtures byte for byte.
* ``LANDING_READY_WPC.txt``: kind, eol, tip blob, delivered git blob, md5, path -- what. Nothing is staged or committed.

    python assemble_wpc_package.py --work C:/Users/Admin/r8_wpc --pkg <package dir> --git-dir C:/Users/Admin/tanitad-push/.git --tip 50efa52
"""
import argparse
import difflib
import hashlib
import json
import subprocess
from pathlib import Path

MODIFIED = [
    ("stack/scripts/train_p8_occupancy.py", "fix 3 + 4: JoinFileReader(.., defect_masks=None) - OPT-IN ego-row strip + track-id-switch rate mask; default read bit-identical (golden digest)"),
    ("stack/tanitad/eval/detection_metrics.py", "fix 5: window_packs(.., with_zh_range=False) - OPT-IN pair_zh_range / pair_h_err; default pack byte-identical (golden digest)"),
]
NEW = [
    ("stack/tanitad/eval/detection_nms.py", "F4b: centre-distance NMS + TRAIN fit + census keys eval_<head>_nms_* (read-only; planner never imports it)"),
    ("stack/tanitad/eval/detection_zh.py", "fix 5: z/h error by range, beyond-30-m bins low-trust, near-field headline"),
    ("stack/tanitad/data/join_label_hygiene.py", "fix 3 + 4: JoinDefectMasks (sha12-keyed frame lists + the two read-time operations)"),
    ("stack/scripts/mine_join_label_defects.py", "fix 3 + 4: mines the lists from a 3-D join; --check-d3 asserts D3's aggregates EXACTLY"),
    ("stack/scripts/refit_perception_thresholds.py", "item 2: TRAIN re-fit of F1 / F4 / F4b for any checkpoint; every output re-read by the stack's loader"),
    ("stack/tanitad/configs/refcv7_det_nms_train.json", "F4b shipped config (refcv7-r101-s0 @ 50,400): box3d 2.5 m / 0.2145, agent 3.0 m / 0.1809 (full-precision gates + TRAIN grid)"),
    ("stack/tanitad/configs/refcv8_join_label_defects.json", "fix 3 + 4 shipped lists: 693 ego frames / 18 clips; 6,227 track-jump events (pose-glitch frames excluded)"),
    ("stack/tanitad/configs/refcv8_box_zh_range_trust.json", "fix 5 shipped trust table (D3 base_face_by_range; near field derived = 30 m)"),
    ("stack/tests/test_refcv8_det_nms.py", "F4b unit tests (25)"),
    ("stack/tests/test_refcv8_det_nms_acceptance_full.py", "F4b full-set acceptance against the banked route packs (6; skipped without D:/refcv7_route_bin; TRAIN fit needs REFCV8_FULL_FIT=1)"),
    ("stack/tests/test_refcv8_det_zh_range.py", "fix 5 tests (12)"),
    ("stack/tests/test_refcv8_join_label_hygiene.py", "fix 3 + 4 tests (16)"),
    ("stack/tests/test_refcv8_refit_perception_thresholds.py", "item 2 tests (17; 3 full-set skipped without the banked TRAIN outputs)"),
    ("stack/tests/test_refcv8_eval_map_mask.py", "fix 6 tests (10)"),
    ("stack/tests/fixtures/refcv8_perception_fixes/refcv8_box_packs_train_every9th.npz", "fixture: every 9th TRAIN pack (120 windows x 2 heads), ids removed"),
    ("stack/tests/fixtures/refcv8_perception_fixes/refcv8_map_hist_train_pooled.npz", "fixture: the TRAIN score histogram [8,2,4000] + class weights"),
]


def git(gd, *a, stdin=None):
    return subprocess.run(["git", "--git-dir", gd, *a], input=stdin, capture_output=True, check=True).stdout


def blob_id(gd, data: bytes) -> str:
    return git(gd, "hash-object", "--no-filters", "--stdin", stdin=data).decode().strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--pkg", required=True)
    ap.add_argument("--git-dir", required=True)
    ap.add_argument("--tip", required=True)
    a = ap.parse_args()
    work, pkg = Path(a.work), Path(a.pkg)
    rows, md5s = [], []
    for kind, items in (("MODIFIED", MODIFIED), ("NEW", NEW)):
        for rel, what in items:
            src = work / rel
            data = src.read_bytes()
            textual = rel.endswith((".py", ".json", ".md", ".txt"))
            if textual:
                data = data.replace(b"\r\n", b"\n")
            tip_blob = "-"
            if kind == "MODIFIED":
                tip_blob = git(a.git_dir, "rev-parse", f"{a.tip}:{rel}").decode().strip()
                tip = git(a.git_dir, "cat-file", "blob", tip_blob)
                assert b"\r" not in tip, f"{rel}: the tip blob is not LF"
                tl, nl = tip.decode("utf-8").split("\n"), data.decode("utf-8").split("\n")
                lost = [ln for ln in difflib.unified_diff(tl, nl, lineterm="", n=0) if ln.startswith("-") and not ln.startswith("---")]
                # a tip line may change ONLY where this fix extends a signature / a docstring end / a call-site continuation (<= 6 lines)
                assert len(lost) <= 6, (rel, lost)
                print(f"[assemble] {rel}: {len(lost)} tip line(s) altered (signature / call-site continuation lines), all other tip lines kept:")
                for ln in lost:
                    print("    ", ln[:150])
            dst = pkg / "code" / "fix" / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(data)
            md5 = hashlib.md5(data).hexdigest()
            rows.append((kind, "LF" if textual else "bin", tip_blob[:12], blob_id(a.git_dir, data)[:12], md5, rel, what, len(data)))
            md5s.append(f"{md5}  code/fix/{rel}")
    out = []
    out.append("LANDING_READY_WPC -- refcv8 WP-C perception fixes (Architecture & Inference, 2026-10-04)")
    out.append("Hand-over: NOTHING staged or committed by this agent (no git add / commit / push). The Master Mind lands these paths BY PATH.")
    out.append("Prefix every `code/fix/...` path with: TanitAD Research Lab/Architecture & Inference/Research/2026-10-04-refcv8-perception-fixes/")
    out.append(f"BASE = tip {a.tip}. Delivered text files are LF (= the tip blobs' EOL); `git archive` output on this box is CRLF - do not diff against a worktree.")
    out.append("Landing path of each file = the path after `code/fix/`. MODIFIED files carry the tip blob they were built from; take them whole.")
    out.append("NOT TOUCHED (WP-B owns): stack/scripts/refc_v3_train.py, the refc_v3 model/decoder modules, stack/tanitad/train/declared_vs_built.py. See INTEGRATION_WPC.md.")
    out.append("")
    out.append("kind      eol  tip_blob     new_blob     md5                                bytes    path  --  what")
    for k, e, tb, nb, md5, rel, what, n in rows:
        out.append(f"{k:9s} {e:4s} {tb:12s} {nb:12s} {md5}  {n:8d} {rel}  --  {what}")
    out.append("")
    out.append("PACKAGE-LOCAL (land with the package, not into stack/): INTEGRATION_WPC.md, LANDING_READY_WPC.txt, raw/*.json (evidence), code/*.py (drivers, mutation campaign, acceptance scripts)")
    out.append("NOT FOR GIT: the working extractions C:/Users/Admin/r8_wpc, r8_wpc_tip, r8_wpc_mut; the evidence bins D:/refcv7_diag_bin, D:/refcv7_route_bin, D:/refcv6_eval_kit; the 409 MB join; scratch copies of the TIP reader / detection_metrics.")
    out.append("THOR-ONLY: INTEGRATION_WPC.md I4 step 1 (the TRAIN-DIAG forward pass of the refcv8 checkpoint through run_diag.py) needs the Thor GPU; every other step here is CPU.")
    (pkg / "LANDING_READY_WPC.txt").write_text("\n".join(out) + "\n", encoding="utf-8")
    (pkg / "raw" / "MD5SUMS_code_fix.txt").write_text("\n".join(md5s) + "\n", encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
