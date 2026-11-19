#!/usr/bin/env python
"""run_mutations_ip1.py -- card PREREG_V8F_PROOF_ADD44 tool checks: every mutation of a source line must turn its named tests red, and the unmutated control must be green before and after.

  python run_mutations_ip1.py --out MUTATIONS_INTEGPRE.json [--only NAME]

One reused work folder work/mut/work (files overwritten, never added): the sources are copied there, the mutation is applied (each old text must occur exactly once), pytest runs the named tests with IP_SRC pointing at the
work folder, so the tests import the mutated copy. The unmutated control runs the whole suite from the same folder, once before the mutations and once after. Nothing outside integpre/ is read or written.
The three mutations the card names are: queue_wrong_checkpoint_for_a_head, queue_head_of_another_base, queue_rule_form_swapped_*.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(HERE, "work", "mut", "work")
BASETEMP = os.path.join(HERE, "work", "bt", "m", "n")
SOURCES = ("read_ip1.py", "read_ip1.sh", "chain_ip1.sh", "pod_integpre_queue.sh", "integpre_arms.txt", "ip_testlib.py")
TESTS = ("test_read_ip1.py", "test_queue_ip.py")
RD, QU = "read_ip1.py", "pod_integpre_queue.sh"
TQ = "test_queue_ip.py::"
TR = "test_read_ip1.py::"
FC = TR + "test_a_failed_control_withholds_every_name_and_difference"

M = [
    ("read_cut_needs_only_two_bases", RD, "nneg >= MIN_BASES", "nneg >= 2", [TR + "test_cut_name_both_directions"]),
    ("read_cut_sign_reversed", RD, 'sum(1 for d in d1["d_s"] if d < 0)', 'sum(1 for d in d1["d_s"] if d > 0)', [TR + "test_cut_name_both_directions"]),
    ("read_cut_interval_not_required", RD, 'd1["A_excludes_0"] and d1["A_hi"] < 0', "True", [TR + "test_cut_name_both_directions"]),
    ("read_share_threshold_lowered", RD, "s >= 0.5", "s >= 0.1", [TR + "test_reaches_label_rule_both_directions"]),
    ("read_share_is_inverted", RD, "float(a) / float(b)", "float(b) / float(a)", [TR + "test_share_is_d2_over_label_minus_off_not_the_inverse"]),
    ("read_reaches_ignores_interval", RD, 'd2["A_hi"] < 0', "True", [TR + "test_reaches_label_rule_both_directions"]),
    ("read_levers_interfere_on_upper_end", RD, 'd4["A_lo"] > 0', 'd4["A_hi"] > 0', [TR + "test_levers_both_directions"]),
    ("read_price_always_no_cost", RD, '"INTEG-PRE-NO-COST" if not cost else', '"INTEG-PRE-NO-COST" if True else', [TR + "test_price_names"]),
    ("read_missing_base_rule_one_base", RD, "if len(bases) < MIN_BASES:", "if len(bases) < 1:", [TR + "test_fewer_than_three_bases_names_nothing"]),
    ("read_controls_do_not_withhold", RD, 'allp = all(v.get("PASS") for v in res["controls"].values())', "allp = True", [FC]),
    ("read_plan_ignores_the_coverage_band", RD, [('if os.path.exists(os.path.join(small, "run_MISSING_%s.txt" % b)):', "if False:"), ("elif rec.get(H5) is not True:", "elif False:")],
     [TR + "test_missing_base_is_reported_and_three_bases_still_name"]),
    ("read_c1_wrong_checkpoint_accepted", RD, '(rec.get("states_record") or {}).get("ckpt_sha256") == ck.group(1)', "True", [FC]),
    ("read_c1_head_of_another_base_accepted", RD, '"head_in_run_equals_head_file": run_sha == hs', '"head_in_run_equals_head_file": True', [FC]),
    ("read_c1_fit_form_ignored", RD, 'bool(tool == FIT_TOOL_SHA and rec.get("clip_sets") == CLIP_SETS)', "True", [FC]),
    ("read_c2_sign_per_base_ignored", RD, 'u["A_excludes_0"] and all(d < 0 for d in u["d_s"])', 'u["A_excludes_0"]', [FC]),
    ("read_c3_differing_files_ignored", RD, "int(d) == 0 and int(n) > 0", "True", [FC]),
    ("read_c4_lead_window_count_ignored", RD, 'cnt.get("n_lead_windows") == N_LEAD and', "True and", [FC]),
    ("read_c5_reader_controls_ignored", RD, '"PASS": bool(hwj and q3j and hwj["controls"].get("ALL_PASS") and q3j["controls"].get("ALL_PASS"))}', '"PASS": True}', [FC]),
    ("read_c6_tolerance_ignored", RD, 'all(abs(x) <= C6_TOL for x in u["d_s"])', "True", [FC]),
    ("read_strategic_line_dropped", RD, 'STRATEGIC, "", "Bases read', '"", "Bases read', [TR + "test_baseline_world_names_and_fixed_lines"]),
    ("read_a_pair_of_the_card_dropped", RD, ', ("IR0_" + k, "IPox_" + k)]', "]", [TR + "test_pairs_cover_the_card"]),
    # the three mutations the card names, on the queue
    ("queue_wrong_checkpoint_for_a_head", QU, 'CK="$W/runs/$Q/$BASE/ckpt_last.pt"', 'CK="$W/runs/actcons_c0/RLAm_s0/ckpt_last.pt"', [TQ + "test_each_checkpoint_uses_its_own_files"]),
    ("queue_head_of_another_base", QU, 'qhead "$P" --qhead "$HEADS/$BASE.qhead.npz"', 'qhead "$P" --qhead "$HEADS/$CTRL_BASE.qhead.npz"', [TQ + "test_each_checkpoint_uses_its_own_files"]),
    ("queue_rule_form_swapped_label_for_off", QU, 'roll "IRL_$BASE" "$Q" "$BASE" label 0', 'roll "IRL_$BASE" "$Q" "$BASE" off 0', [TQ + "test_rule_forms_are_the_cards"]),
    ("queue_rule_form_swapped_off_for_label", QU, 'roll "IR0_$BASE" "$Q" "$BASE" off 0', 'roll "IR0_$BASE" "$Q" "$BASE" label 0', [TQ + "test_rule_forms_are_the_cards"]),
    ("queue_rule_form_swapped_identity_for_head", QU, 'qhead "$P" --qhead "$HEADS/$BASE.qhead.npz"', 'qhead "$P" --q-equals-head', [TQ + "test_each_checkpoint_uses_its_own_files"]),
    # further queue mutations
    ("queue_fit_reads_states_of_another_base", QU, '--states "$S" --pack "$PACK"', '--states "$ST/${CTRL_BASE}_train.npz" --pack "$PACK"', [TQ + "test_each_checkpoint_uses_its_own_files"]),
    ("queue_no_slot_wait_before_rollout", QU, '  if [ "$DRY" = 1 ]; then\n    slot_wait "rollout $NEW"\n', '  if [ "$DRY" = 1 ]; then\n', [TQ + "test_order_and_slot_waits"]),
    ("queue_control_c3_on_another_base", QU, "CTRL_BASE=RLAm_s0", "CTRL_BASE=RLAm_s1", [TQ + "test_control_c3_is_on_candidate_base0_only"]),
    ("queue_second_start_allowed", QU, [("for m in INTEGPRE_DONE.txt INTEGPRE_STOP.txt INTEGPRE_STARTED.txt; do", "for m in INTEGPRE_DONE.txt INTEGPRE_STOP.txt; do"), ("set -o noclobber; ", "")],
     [TQ + "test_second_start_and_missing_inputs_are_refused"]),
    ("queue_volume_check_absent", QU, '[ "$VOLFREE" -ge $(( EXPECT + VOL_MARGIN_GB * 1073741824 )) ] ||', "true ||", [TQ + "test_volume_refusal"]),
    ("queue_one_arm_dropped_from_the_file", "integpre_arms.txt", "IRQ_RLFT_s3 actcons_e3 RLFT_s3 3 0.5\n", "", [TQ + "test_arms_file_has_the_eight_checkpoints_of_the_card"]),
]


def prepare():
    os.makedirs(WORK, exist_ok=True)
    for f in SOURCES:
        shutil.copyfile(os.path.join(HERE, f), os.path.join(WORK, f))


def pytest(nodes, extra):
    env = dict(os.environ, IP_SRC=WORK, CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="2", PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1")
    cmd = [sys.executable, "-m", "pytest", "-q", "--basetemp", BASETEMP, "-p", "no:cacheprovider"] + nodes + extra
    r = subprocess.run(cmd, cwd=HERE, env=env, capture_output=True, encoding="utf-8", errors="replace", timeout=1500)
    return r.returncode, r.stdout


def control(label, extra):
    prepare()
    rc, out = pytest(list(TESTS), extra)
    summ = out.strip().splitlines()[-1] if out.strip() else ""
    print("CONTROL unmutated suite %s: rc %d %s" % (label, rc, summ), flush=True)
    return {"rc": rc, "green": rc == 0, "summary": summ}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="")
    a, extra = ap.parse_known_args(argv)
    os.makedirs(os.path.dirname(BASETEMP), exist_ok=True)
    res = {"card": "PREREG_V8F_PROOF_ADD44", "mutations": [], "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if not a.only:
        res["control_unmutated_before"] = control("before", extra)
    for ent in M:
        name, fn, old = ent[:3]
        new, tests = (ent[3], ent[4]) if len(ent) == 5 else (None, ent[3])
        if a.only and a.only != name:
            continue
        prepare()
        p = os.path.join(WORK, fn)
        s = open(p, encoding="utf-8", newline="").read()
        pairs = old if isinstance(old, list) else [(old, new)]
        if any(s.count(o) != 1 for o, _ in pairs):
            res["mutations"].append({"name": name, "file": fn, "applied": False, "why": "old text occurs %s times" % [s.count(o) for o, _ in pairs], "red": False})
            print("MUTATION %-46s NOT APPLIED" % name, flush=True)
            continue
        for o, nw in pairs:
            s = s.replace(o, nw)
        open(p, "w", encoding="utf-8", newline="").write(s)
        rc, out = pytest(tests, extra)
        failed = [l.split(" ")[1].split(" - ")[0] for l in out.splitlines() if l.startswith("FAILED ")]
        red = bool(rc != 0 and all(any(t.split("::")[1] in f for f in failed) for t in tests))
        res["mutations"].append({"name": name, "file": fn, "applied": True, "tests": tests, "red": red, "failed": failed[:6], "rc": rc})
        print("MUTATION %-46s %s (%s)" % (name, "RED" if red else "NOT RED", ", ".join(f.split("::")[-1] for f in failed)[:120]), flush=True)
    if not a.only:
        res["control_unmutated_after"] = control("after", extra)
    res["n_mutations"] = len([m for m in res["mutations"] if m.get("applied")])
    res["n_red"] = len([m for m in res["mutations"] if m.get("red")])
    res["all_ok"] = bool(not a.only and res["n_mutations"] == len(M) and res["n_red"] == len(M) and res["control_unmutated_before"]["green"] and res["control_unmutated_after"]["green"])
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1)
    print("n_mutations %d of %d, red %d, all_ok %s" % (res["n_mutations"], len(M), res["n_red"], res["all_ok"]), flush=True)
    return 0 if res["all_ok"] else 2


if __name__ == "__main__":
    sys.exit(main())
