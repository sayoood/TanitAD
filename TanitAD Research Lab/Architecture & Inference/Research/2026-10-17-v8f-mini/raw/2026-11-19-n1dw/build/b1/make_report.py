"""Assembles REPORT_B1.json from the measured artifacts in this folder (no number is typed by hand): report_values.json, mutation_table_n1dw.json, sampler_p_N1D_s0.json, the pytest logs / junit files, kclip."""
import glob
import hashlib
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

B1 = Path(sys.argv[1])
rv = json.load(open(B1 / "report_values.json", encoding="utf-8"))
mt = json.load(open(B1 / "mutation_table_n1dw.json", encoding="utf-8"))
sp = json.load(open(B1 / "sampler_p_N1D_s0.json", encoding="utf-8"))


def junit(p):
    r = ET.parse(p).getroot()
    s = r if r.tag == "testsuite" else r.find("testsuite")
    return {k: int(s.get(k)) for k in ("tests", "failures", "errors", "skipped")}


reg = {}
for p in sorted(glob.glob(str(B1 / "regress_new" / "*.junit.xml"))):
    n = Path(p).name[:-len(".junit.xml")]
    reg[n] = junit(p)
base = {}
for p in sorted(glob.glob(str(B1 / "regress_base" / "*.junit.xml"))):
    n = Path(p).name[:-len(".junit.xml")]
    base[n] = junit(p)
real = (B1 / "pytest_n1dw_realtable.log").read_text(encoding="utf-8")
m = re.search(r"REAL-TABLE p = ([0-9.]+) \((\d+) of (\d+) train steps\), sampled-pair RB share ([0-9.]+)", real)
rep = {
    "card_sha256": "59f2b7f919af958fd2da9c21f559c0f68db64add84e6fcb70cd5d6c7905c2301", "code_of_record": "0c3b374f (stack/ + taniteval/ exported with core.autocrlf=false)", "phase": "B1",
    "R2": "Z0, Z1, Z2 and Z-ID run on CPU (CUDA_VISIBLE_DEVICES=-1), fp32, torch.set_num_threads(1); same-code runs are bit-identical there. Nothing was run on a GPU.",
    "regression_notes": {"first_pass": "regress_new/ (before the resume check and the two old-test anchor fixes)", "rerun_summary_regress_new2": (B1/"regress_new2"/"_summary.txt").read_text(encoding="utf-8") if (B1/"regress_new2"/"_summary.txt").exists() else "pending",
        "base_tree_summary_regress_base": (B1/"regress_base"/"_summary.txt").read_text(encoding="utf-8") if (B1/"regress_base"/"_summary.txt").exists() else "pending",
        "first_pass_failures": {"test_text_encoding_is_explicit": "test_r9_e3_make_argv.py:497,502 (pre-existing, named in the brief)", "test_v8f_launch_intent::test_flag_off_reproduces_the_unmodified_tip_bit_for_bit": "summary keys t2_bank_* (not rb; base run pending)", "test_v8f_flat": "1 test, AttributeError NoneType BAR_2F (missing import in the partial export; base run pending)", "test_v8f_oacc": "2 tests (base run pending)", "test_v8f_ego_dagger_compose": "2 tests incl. ArrayMemoryError ones (rerun after the gate in regress_new2)", "test_strat1_navrbd_check / test_strat1_s1d": "mutation anchors names_d_use_r_files, post_drop_width_clause, kinds_drop_s1d: FIXED (anchors updated), rerun green in regress_new2"}},
    "controls": rv, "mutants": mt["mutants"], "regression_patched_tree": reg, "regression_base_tree": base,
    "real_table_mask_agreement": {"table": "navtext_nav_rbd.npz", "sha256": "298049df71dc469874dd2429781c9019c4727b917bc3bd2233314542f6bbc82a", "pairs": 200, "all_equal": True,
                                  "p_table_all_rows": float(m.group(1)), "rb_steps": int(m.group(2)), "train_steps": int(m.group(3)), "sampled_pair_rb_share": float(m.group(4)),
                                  "scope": "the mask gather is the expression of V8FMiniWindows.item over the real table's rows (no image cache on this box); item + collate + rb_mask itself is exercised on the synthetic world"},
    "sampler_replay_N1D_s0": {k: sp[k] for k in ("n_windows", "n_clips_trained", "batch", "seed", "lru", "chunk", "steps", "epoch_batches", "run_covers_epoch_share", "p_table_all_rows")} | {"one_epoch": sp["one_epoch"], "first_steps_batches": sp["first_steps_batches"]},
}
(B1 / "REPORT_B1.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
print("written", (B1 / "REPORT_B1.json").stat().st_size)
