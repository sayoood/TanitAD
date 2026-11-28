"""WP-C: a REAL-DEFECT mutation campaign - reintroduce each fix's historical defect into the PRODUCTION source and demand the fix's test goes RED.

CLAUDE.md "guards need mutation, not inspection": a test that only passes on correct code proves nothing until the defect it guards has
been put back and the test has been seen to fail. For every fix this applies one-line source mutations to a PRIVATE COPY of the stack
(``--tree``; never the repo, never the extraction the tests were developed in), runs the named test file, and records RED (>= 1 test failed -
the guard works), SURVIVED (exit 0 - the guard is blind to this defect) or ERROR (the mutant broke an import / collection: red, but trivially).
A CONTROL entry (no mutation) must read GREEN: a harness that reports red for everything is as useless as one that never does.

    python mutation_campaign_wpc.py --tree C:/Users/Admin/r8_wpc_mut --out raw/mutation_campaign.json
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

PY = sys.executable
S = "stack"

# (id, fix, relative file under the tree, old (LF), new (LF), test file, extra pytest args)
MUTATIONS = [
    ("CONTROL", "-", None, None, None, "tests/test_refcv8_det_zh_range.py", []),
    # ---- F4b: the NMS -------------------------------------------------------------------------------------------------------------
    ("N1_exclusive_radius", "F4b", f"{S}/tanitad/eval/detection_nms.py",
     "xi - float(xy[j, 0]), yi - float(xy[j, 1])) <= r:", "xi - float(xy[j, 0]), yi - float(xy[j, 1])) < r:",
     "tests/test_refcv8_det_nms.py", ["-k", "test_A or test_B or test_D or test_E or test_F"]),
    ("N2_ascending_score_order", "F4b", f"{S}/tanitad/eval/detection_nms.py",
     'order = cand[np.argsort(-p[cand], kind="mergesort")]', 'order = cand[np.argsort(p[cand], kind="mergesort")]',
     "tests/test_refcv8_det_nms.py", ["-k", "test_A or test_B or test_D or test_E or test_F"]),
    ("N3_floor_ignored", "F4b", f"{S}/tanitad/eval/detection_nms.py",
     "hi = p >= float(p_floor)", "hi = p >= 0.0", "tests/test_refcv8_det_nms.py", ["-k", "test_A or test_B or test_D or test_E or test_F"]),
    ("N4_tie_prefers_larger_radius", "F4b", f"{S}/tanitad/eval/detection_nms.py",
     "best = max(cands, key=lambda t: (t[0], t[1]))[2]", "best = max(cands, key=lambda t: (t[0], -t[1]))[2]",
     "tests/test_refcv8_det_nms.py", ["-k", "test_C_a_tie"]),
    ("N5_input_pack_mutated", "F4b", f"{S}/tanitad/eval/detection_nms.py",
     "    q = dict(pk)\n    for k in PER_SLOT_KEYS:", "    q = pk\n    for k in PER_SLOT_KEYS:",
     "tests/test_refcv8_det_nms.py", ["-k", "test_D"]),
    ("N6_gate_unvalidated", "F4b", f"{S}/tanitad/eval/detection_nms.py",
     "if not (math.isfinite(g) and 0.0 < g < 1.0):", "if False:", "tests/test_refcv8_det_nms.py", ["-k", "test_F"]),
    ("N7_census_read_at_the_declared_gate", "F4b", f"{S}/tanitad/eval/detection_nms.py",
     'c = _census(q, e["gate"])', "c = _census(q, 0.5)", "tests/test_refcv8_det_nms.py", ["-k", "test_D"]),
    ("N8_planner_imports_nms (scanner)", "F4b", f"{S}/tanitad/models/slot_presence.py",
     "from __future__ import annotations", "from __future__ import annotations\nfrom tanitad.eval import detection_nms  # MUTANT",
     "tests/test_refcv8_det_nms.py", ["-k", "test_E_no_planner"]),
    # ---- F1 / F4 / F4b refit tool ---------------------------------------------------------------------------------------------------
    ("T1_last_edge_on_ties", "refit", f"{S}/scripts/refit_perception_thresholds.py",
     "i = int(np.argmax(j))", "i = int(len(j) - 1 - np.argmax(j[::-1]))", "tests/test_refcv8_refit_perception_thresholds.py", ["-k", "test_A or test_B"]),
    ("T2_gate_is_the_declared_half", "refit", f"{S}/scripts/refit_perception_thresholds.py",
     'gates[hd] = float(g["gate"])', "gates[hd] = 0.5", "tests/test_refcv8_refit_perception_thresholds.py", ["-k", "test_C"]),
    ("T3_empty_class_not_refused", "refit", f"{S}/scripts/refit_perception_thresholds.py",
     "if float(h[c, 1].sum()) <= 0:", "if False:", "tests/test_refcv8_refit_perception_thresholds.py", ["-k", "test_E_a_class"]),
    ("T4_first_band_only", "refit", f"{S}/scripts/refit_perception_thresholds.py",
     "        h = h.sum(axis=1)\n", "        h = h[:, 0]\n", "tests/test_refcv8_refit_perception_thresholds.py", ["-k", "test_B"]),
    # ---- fix 3: the ego box ----------------------------------------------------------------------------------------------------------
    ("J1_ego_list_ignored", "3", f"{S}/tanitad/data/join_label_hygiene.py",
     "return bool(s) and int(frame_idx) in s", "return True", "tests/test_refcv8_join_label_hygiene.py", []),
    ("J2_footprint_too_wide", "3", f"{S}/tanitad/data/join_label_hygiene.py",
     "EGO_X_RANGE_M: tuple[float, float] = (-1.0, 4.0)", "EGO_X_RANGE_M: tuple[float, float] = (-5.0, 4.0)",
     "tests/test_refcv8_join_label_hygiene.py", []),
    ("R1_reader_discards_the_stripped_agents", "3", f"{S}/scripts/train_p8_occupancy.py",
     'rec["agents"], _ = self.defect_masks.strip_ego_boxes(', "_discard, _ = self.defect_masks.strip_ego_boxes(",
     "tests/test_refcv8_join_label_hygiene.py", []),
    ("J3_load_accepts_raw_clip_keys", "3", f"{S}/tanitad/data/join_label_hygiene.py",
     "if len(k) != 12 or any(ch not in \"0123456789abcdef\" for ch in k):", "if False:", "tests/test_refcv8_join_label_hygiene.py", []),
    # ---- fix 4: the id switch ----------------------------------------------------------------------------------------------------------
    ("J4_only_record_f_masked", "4", f"{S}/tanitad/data/join_label_hygiene.py",
     "for r in (f - 1, f):", "for r in (f,):", "tests/test_refcv8_join_label_hygiene.py", []),
    ("J5_masked_row_not_zeroed", "4", f"{S}/tanitad/data/join_label_hygiene.py", "r2[bad] = 0.0", "r2[bad] = r2[bad]",
     "tests/test_refcv8_join_label_hygiene.py", []),
    ("J6_mask_flag_not_cleared", "4", f"{S}/tanitad/data/join_label_hygiene.py", "m2[bad] = False", "m2[bad] = True",
     "tests/test_refcv8_join_label_hygiene.py", []),
    ("R2_reader_discards_the_rate_mask", "4", f"{S}/scripts/train_p8_occupancy.py",
     "rt, rm = self.defect_masks.apply_rate_mask(", "_rt, _rm = self.defect_masks.apply_rate_mask(",
     "tests/test_refcv8_join_label_hygiene.py", []),
    # ---- fix 5: z / h by range -----------------------------------------------------------------------------------------------------------
    ("D1_range_is_x_only", "5", f"{S}/tanitad/eval/detection_metrics.py",
     "z_rng = gt_box[b][cc][:, :2].norm(dim=-1)[zm]", "z_rng = gt_box[b][cc][:, 0].abs()[zm]", "tests/test_refcv8_det_zh_range.py", []),
    ("D2_h_error_reads_z", "5", f"{S}/tanitad/eval/detection_metrics.py",
     'dh = (pred["h"].detach().float()[b][rr] - tgt_real["h"].float()[b][cc]).abs()',
     'dh = (pred["cz"].detach().float()[b][rr] - tgt_real["cz"].float()[b][cc]).abs()', "tests/test_refcv8_det_zh_range.py", []),
    ("D3_flag_on_by_default", "5", f"{S}/tanitad/eval/detection_metrics.py",
     "with_match: bool = True, with_zh_range: bool = False)", "with_match: bool = True, with_zh_range: bool = True)",
     "tests/test_refcv8_det_zh_range.py", []),
    ("Z1_near_field_is_the_highest_low_edge", "5", f"{S}/tanitad/eval/detection_zh.py",
     "near = min(low) if low else math.inf", "near = max(low) if low else math.inf", "tests/test_refcv8_det_zh_range.py", []),
    ("Z2_far_bin_trusted", "5", f"{S}/tanitad/eval/detection_zh.py",
     "return float(hi) <= float(near_field_m)", "return float(hi) <= float(near_field_m) + 40.0", "tests/test_refcv8_det_zh_range.py", []),
    ("Z3_headline_pools_every_range", "5", f"{S}/tanitad/eval/detection_zh.py",
     "s = _stats(z[trusted_mask], h[trusted_mask])", "s = _stats(z, h)", "tests/test_refcv8_det_zh_range.py", []),
    # ---- fix 6: the eval clips without map GT --------------------------------------------------------------------------------------------
    ("X1_not_seen_cells_counted_as_supervised", "6", f"{S}/tanitad/models/map_head_hires.py",
     "    sup = tgt != NOT_SEEN_CODE\n    n = int(sup.sum())", "    sup = tgt >= 0\n    n = int(sup.sum())",
     "tests/test_refcv8_eval_map_mask.py", []),
    ("X2_missing_file_falls_back_to_zeros", "6", f"{S}/scripts/refc_v3_train.py",
     "                                           _sem_fine.NOT_SEEN_CODE,\n                                           dtype=torch.uint8),\n                    \"map_fine_label\": torch.tensor(False)}",
     "                                           0,\n                                           dtype=torch.uint8),\n                    \"map_fine_label\": torch.tensor(False)}",
     "tests/test_refcv8_eval_map_mask.py", []),
    ("X3_missing_file_flagged_labelled", "6", f"{S}/scripts/refc_v3_train.py",
     "                                           _sem_fine.NOT_SEEN_CODE,\n                                           dtype=torch.uint8),\n                    \"map_fine_label\": torch.tensor(False)}",
     "                                           _sem_fine.NOT_SEEN_CODE,\n                                           dtype=torch.uint8),\n                    \"map_fine_label\": torch.tensor(True)}",
     "tests/test_refcv8_eval_map_mask.py", []),
]


def run_tests(tree: Path, test: str, extra: list) -> tuple:
    env = dict(os.environ, PYTHONUTF8="1", OMP_NUM_THREADS="2", CUDA_VISIBLE_DEVICES="",
               PYTHONPATH=f"{tree}/stack;{tree}/stack/scripts;{tree}/taniteval")
    t0 = time.time()
    p = subprocess.run([PY, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", test, *extra], cwd=str(tree / "stack"), env=env,
                       capture_output=True, text=True, timeout=1800)
    out = (p.stdout or "") + (p.stderr or "")
    tail = [ln for ln in out.strip().splitlines() if ln.strip()][-1:] or [""]
    failed = len(re.findall(r"^FAILED ", out, flags=re.M))
    errors = len(re.findall(r"^ERROR ", out, flags=re.M))
    return p.returncode, failed, errors, tail[0], round(time.time() - t0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True, help="a PRIVATE copy of the stack (the mutants are applied here and restored)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default=None, help="comma-separated mutation ids")
    a = ap.parse_args()
    tree = Path(a.tree)
    results = []
    for (mid, fix, rel, old, new, test, extra) in MUTATIONS:
        if a.only and mid.split(" ")[0] not in a.only.split(","):
            continue
        orig = None
        path = tree / rel if rel else None
        try:
            if path is not None:
                raw = path.read_bytes()
                orig = raw
                txt = raw.decode("utf-8")
                crlf = "\r\n" in txt
                lf = txt.replace("\r\n", "\n")
                if lf.count(old) != 1:
                    results.append({"id": mid, "fix": fix, "verdict": "BAD_ANCHOR", "anchor_count": lf.count(old)})
                    print(mid, "BAD_ANCHOR", lf.count(old), flush=True)
                    continue
                mut = lf.replace(old, new)
                path.write_bytes((mut.replace("\n", "\r\n") if crlf else mut).encode("utf-8"))
            rc, failed, errors, tail, secs = run_tests(tree, test, extra)
        finally:
            if orig is not None:
                path.write_bytes(orig)
        if mid == "CONTROL":
            verdict = "GREEN" if rc == 0 else "HARNESS_BROKEN"
        elif rc == 0:
            verdict = "SURVIVED"
        elif failed:
            verdict = "RED"
        else:
            verdict = "ERROR"
        results.append({"id": mid, "fix": fix, "file": rel, "test": test, "verdict": verdict, "failed": failed, "errors": errors,
                        "pytest_tail": tail, "secs": secs})
        print(f"{mid:45s} {verdict:10s} failed={failed} errors={errors} {secs}s  {tail}", flush=True)
    Path(a.out).write_text(json.dumps(results, indent=1), encoding="utf-8")
    bad = [r for r in results if r["verdict"] in ("SURVIVED", "ERROR", "BAD_ANCHOR", "HARNESS_BROKEN")]
    print(f"[campaign] {len(results)} entries, {len(bad)} not red/green-as-expected: {[r['id'] for r in bad]}", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
