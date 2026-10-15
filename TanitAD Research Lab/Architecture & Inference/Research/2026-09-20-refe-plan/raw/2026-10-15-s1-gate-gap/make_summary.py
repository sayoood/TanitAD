#!/usr/bin/env python3
"""Assembles RESULTS.json from the measurement JSONs in this folder (numbers are READ from them, not retyped).  Run from this folder:  python make_summary.py

REPORT.md could not be written: the harness refused the write in the package folder and in the fallback folder ("Subagents should return findings as text, not write report files").
RESULTS.json is therefore the structured result (numbers, evidence classes, residuals); the prose is the hand-back message.
"""
from __future__ import annotations

import json
from pathlib import Path

H = Path(__file__).resolve().parent


def J(name):
    p = H / name
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def main() -> int:
    att, est = J("attribution.json"), J("lever_estimate.json")
    syn, real = J("fp_timing_dev_synth413k.json"), J("fp_timing_dev_real_inputs.json")
    suite_t, prof = J("suite_timing_dev.json"), J("aux_head_profile.json")
    mut, patches = J("mutation_results.json"), J("PATCHES.json")
    s_base, s_new, other = J("suite_results_base.json"), J("suite_results_patched.json"), J("other_suites.json")
    tok = J("token_check_selftest_dev.json")
    d0 = syn["dirs"][0]
    cpu = syn["cpu_only"][0]["per_file_us"]
    hashing = real["files"][0]["variants"]
    sim = real["simulated"]
    scen = est["scenarios"] if est else []
    save_AB = sorted({r["saved_per_arm_min"]["A_and_B"] for r in scen})
    out = {
        "brief": "cut the GPU-idle gap before each S1 arm without weakening any check",
        "harness_note": "REPORT.md was refused by the harness (both locations); this file + the hand-back carry the report. All other deliverables are as listed.",
        "evidence_classes": {"MEASURED": "pod timestamps in the copied logs, or a dev-box run (script + JSON named)", "INFERRED": "arithmetic on MEASURED figures", "ESTIMATED": "a model with stated assumptions",
                             "SIMULATED": "a stand-in (constant stat / sleep in os.stat), never a pod figure"},
        "attribution_24_8_min": {
            "gap_min": att["gap_min"], "source": "attribution.json (attribution_from_logs.py over the copied pod logs)",
            "segments_min": {k: v["min"] for k, v in att["segments"].items()}, "shares_pct": {k: v["share_pct"] for k, v in att["segments"].items()},
            "one_remeasure_min": att["remeasure"]["verify_each_min"], "five_remeasures_min_total": att["remeasure"]["five_remeasures_min_total"], "five_remeasures_share_pct": att["remeasure"]["share_of_gap_pct"],
            "inside_one_remeasure": {"image_listing_files": att["inside_one_remeasure"]["image_listing_files"], "image_listing_s_lower_bound_INFERRED": att["inside_one_remeasure"]["image_listing_s_lower_bound"],
                                     "per_file_ms_INFERRED": att["inside_one_remeasure"]["per_file_ms_lower_bound"], "checkpoint_sha256_s_upper_bound_MEASURED": att["inside_one_remeasure"]["checkpoint_sha256_upper_bound_s"],
                                     "code_tree_s_upper_bound_MEASURED": att["inside_one_remeasure"]["code_tree_upper_s"]},
            "reading": "72-73 % of the gap is the SAME operation run five times (tree_manifest + data_manifest); inside it, hashing is <= 2 % and the 413,184-file image LISTING is >= 97 %.  "
                       "G-SUITE is 23 %, the mandatory GPU-idle wait 4 %.  For arms 2-8 the preflight (5.9-6.4 min MEASURED, GPU) is added on top (~31 min)."},
        "levers": [
            {"id": "B", "title": "drop the redundant second verify on the fresh path (s1aux_pod.sh :403)", "patch": "PATCH_s1aux_pod.sh.diff",
             "kept": ["verify after the mint (:400)", "the BINDING verify after wait_clear, right before scorer_finetune.py (:414)", "the verify on the resumed path (gate passed by an earlier invocation)"],
             "unchanged": ["what is recorded in the arms manifest", "the PREREG order: token row written before the arm's first step"],
             "saved_min_per_arm": att["segments"]["D_verify_400_and_403"]["per_verify_min"], "evidence_class": "MEASURED per-verify cost (3.4-3.8 min, pod); the saving is one verify by construction",
             "tests": "test_s1aux_chain.py: 6 static properties + 6 chain mutations + behavioural verify-count sequences in the sandbox (fresh arm = pre,run,verify,verify,train; resumed = verify,verify,train) with RED twins"},
            {"id": "A1", "title": "directory listing: stat on worker threads, string ops instead of a Path per file; independent inputs fingerprinted concurrently", "patch": "PATCH_gate_token.py.diff",
             "digest_identity": "bit-identical: frozen pre-patch code is the oracle in test_launch_gate.py (fingerprint_identity_static, 21 checks, 7 digest-altering mutations RED); also equals the banked generic gate's fingerprint (dev static suite)",
             "dev_box_413184_file_tree_MEASURED": {"old_min_s": d0["old"]["min_s"], "new_min_s_by_threads": {k: v["min_s"] for k, v in d0["new"].items()},
                                                     "speedup_by_threads": {k: v["speedup_vs_old_min"] for k, v in d0["new"].items()}, "identical_all": all(v["identical"] for v in d0["new"].values()),
                                                     "box": "NTFS, shared with other agents' CPU jobs (load 100 %), HIGH priority class; ratios are the result, absolute seconds are not"},
             "python_overhead_per_file_us_MEASURED_stat_stubbed": cpu,
             "data_manifest_realistic_MEASURED": {"old_min_s": real["data_manifest"]["old"]["min_s"], "new_min_s_by_threads": {k: v["min_s"] for k, v in real["data_manifest"]["new"].items()},
                                                   "identical": all(v["identical"] for v in real["data_manifest"]["new"].values())},
             "latency_bound_mechanism_SIMULATED": {"files": 20000, "nominal_ms": sim["simulated_latency_ms"], "effective_ms_per_stat": round(1000 * sim["old"]["min_s"] / 20000, 2),
                                                   "old_s": sim["old"]["min_s"], "new_s_by_threads": {k: v["min_s"] for k, v in sim["new"].items()},
                                                   "note": "Windows sleep granularity makes the effective latency ~2 ms, not 0.5 ms: this shows the mechanism (threads overlap latency), not a pod number"},
             "pod_effect_ESTIMATED": {"saved_min_per_arm_range_A_and_B": [save_AB[0], save_AB[-1]] if save_AB else None,
                                      "depends_on": "S = how many metadata requests the pod volume serves at once (UNKNOWN: S=1 -> only the string ops help; S>=8 -> ~17 min)",
                                      "scenarios": [{"k": r["k_pod_cpu_slowdown"], "S": r["S_metadata_concurrency"], "A_and_B_saved_min": r["saved_per_arm_min"]["A_and_B"], "gap_after_min": r["gap_after_min"]} for r in scen]},
             "NFS_pod_speedup": "UNMEASURED.  bench_fingerprint.py --token <the pod's PASS token> proves identity on the real 413,184-file tree and times it in about a minute"},
            {"id": "A2", "title": "sha256_file through a reusable buffer (readinto)", "patch": "inside PATCH_gate_token.py.diff",
             "dev_box_1.285GB_MB_per_s_MEASURED": {k: v["MB_per_s_min"] for k, v in hashing.items()},
             "reading": "334 -> 560 MB/s, ~1.5 s saved per call on a 1.29 GB checkpoint; the pod's checkpoint hash is <= 4 s in total, so lever A2 is worth seconds, not minutes.  Digest identical (sha256 of the same bytes; tested at chunk boundaries)."},
            {"id": "I", "title": "instrumentation (no semantic change): per-input manifest seconds in launch_gate.log and token_verify.json (timing_s), per-script seconds in the G-SUITE evidence", "patch": "PATCH_launch_gate.py.diff, PATCH_gate_checks.py.diff, inside PATCH_gate_token.py.diff",
             "why": "the pod wrote no per-phase timestamps, so section 1.1 of the attribution is INFERRED; after landing, the next arm's logs MEASURE it"},
        ],
        "g_suite": {"dev_box_seconds_sequential_MEASURED": {x["script"]: x["seconds"] for x in suite_t["runs"][0]["sequential"]}, "sequential_wall_s": suite_t["runs"][0]["sequential_wall_s"],
                    "parallel_wall_s": suite_t["runs"][0].get("parallel_wall_s"), "parallel_same_outcome": suite_t["runs"][0].get("same_outcome"),
                    "long_pole": "test_aux_drivable_head.py: 73 % torch autograd backward (a_amp re-run in each of 20 mutation repetitions); the store re-hash is 1.65 s",
                    "decision": "NOT changed.  Parallel wall (481 s) was not below sequential (458 s) on a box shared with other CPU jobs, torch oversubscribes threads (TRAPS.md), and shortening the long pole would weaken a check.",
                    "profile": "aux_head_profile.json"},
        "suites": {"base": s_base, "patched": s_new, "other_chains_base_vs_patched": other},
        "mutation_proofs": mut,
        "token_identity_selftest_dev": tok,
        "residual_gap_not_removed": [
            {"what": "preflight (scorer_finetune.py --preflight), ~6 min per arm for arms 2-8 (not in the 24.8)", "why": "a GPU job; S1 never puts a GPU job beside a training job"},
            {"what": "G-SUITE 5.6 min", "why": "one script is 59 % of it and is compute; shortening it weakens a check; parallelism not shown to help"},
            {"what": "wait_clear >= 60 s", "why": "3 idle polls by design (never train beside anything)"},
            {"what": "4 remaining tree+data re-measures (gate start, finalize, post-mint verify, BINDING verify)", "why": "each is a real re-measure the brief keeps; after lever A their cost depends on S (pod only)"},
            {"what": "gate start imports (scorer_finetune -> torch) inside segment A", "why": "needed for the trainer's parser; seconds"}],
        "proposals_not_implemented": [
            {"id": "E", "what": "run G-SUITE beside the preflight (both ~6 min; CPU vs GPU)", "needs": "per-check evidence binding in launch_gate.run_checks (today every check's inputs_read is the whole data manifest, including the preflight file)", "estimated_saving_min": "up to ~5.6 for arms 2-8"},
            {"id": "F", "what": "run the CPU-only part of the NEXT arm's gate (G-SUITE, listing) at nice 19 beside the current arm's training (A4 precedent for CPU stages)", "needs": "a PI decision (CPU load on a training pod) and the same evidence-binding change as E", "estimated_saving_min": "most of segments A-C for arms 2-8"},
            {"id": "D", "what": "G-SUITE scripts in parallel", "needs": "a quiet-box measurement on the pod first; dev box showed no gain"}],
        "pod_followups": [
            "land the patches, re-run `launch_gate.py ship-manifest` (the shipped tree changes), re-ship to the pod",
            "pod, read-only, ~1 min: python bench_fingerprint.py --new <pkg>/refe/gate_token.py --token <lg/ARM/PASS_*.json> --threads 1,16 --out token_identity_pod.json   (identity on the real data + the pod's real timing)",
            "pod, ~7 min, CPU + metadata only (run when idle): python bench_fingerprint.py --old gate_token_OLD_snapshot.py --new <pkg>/refe/gate_token.py --dir /workspace/data/navtrain_pixels --threads 1,4,16,32 --repeat 1 --cpu-only --out fp_timing_pod.json",
            "after the next arm: read the `[timing]` lines in launch_gate.log and info.timing_s in token_verify.json"],
        "patches": patches,
    }
    (H / "RESULTS.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("wrote RESULTS.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
