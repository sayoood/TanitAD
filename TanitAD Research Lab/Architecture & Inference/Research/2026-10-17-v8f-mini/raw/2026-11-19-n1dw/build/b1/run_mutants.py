"""K-MUT driver (PREREG_V8F_N1DW section 6): for each mutant, edit ONE source line of the build tree, run the NAMED test (it must go RED), restore the file, verify its sha256 equals the pre-run value.
Writes mutation_table_n1dw.json. One pytest at a time, CPU only. usage: run_mutants.py <tree> <outjson> <tmpdir> [mutant ids...]"""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

TREE, OUT, TMP = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
ONLY = set(sys.argv[4:])
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
T = "tests/test_v8f_n1dw.py::"
MODEL, TRAIN, DATA = "stack/tanitad/models/v8f_mini.py", "stack/scripts/v8f_mini_train.py", "stack/tanitad/data/v8f_mini_data.py"

MUTANTS = [
    ("a_delete_extra_term_line", MODEL,
     '''        total = total + cfg.w_traj * (cfg.rb_window_w - 1.0) * T["rb_traj"] if cfg.rb_window_w > 1.0 else total''',
     '''        total = total''',
     [T + "test_z1_w4_vs_w1_every_existing_term_and_output_bit_equal_total_moves_by_exactly_the_term", T + "test_z2_probe_term_positive_head_gradient_and_exact_zero_controls"]),
    ("b_mask_all_ones", TRAIN,
     '''    return batch["nav_rb"][:, -1, 0] > 0.5''',
     '''    return torch.ones_like(batch["nav_rb"][:, -1, 0]) > 0.5''',
     [T + "test_mask_is_c0_of_the_nav_rb_row_at_now_a_bool_per_sample", T + "test_z2_probe_term_positive_head_gradient_and_exact_zero_controls"]),
    ("c_mask_read_at_now_plus_5", DATA,
     '''            it["nav_rb"] = J["nav_rb"][[self.row(ci, int(j)) for j in idc]]''',
     '''            it["nav_rb"] = J["nav_rb"][[self.row(ci, min(int(j) + 5, T - 1)) for j in idc]]''',
     [T + "test_mask_no_future_row_garbage_after_now_leaves_it_unchanged", T + "test_mask_is_c0_of_the_nav_rb_row_at_now_a_bool_per_sample"]),
    ("d_divide_by_rb_count", MODEL,
     '''    g = gok * rb.to(gok.dtype)
    n = gok.sum().clamp_min(1.0)''',
     '''    g = gok * rb.to(gok.dtype)
    n = g.sum().clamp_min(1.0)''',
     [T + "test_z1_w4_vs_w1_every_existing_term_and_output_bit_equal_total_moves_by_exactly_the_term", T + "test_zid_per_sample_trajectory_gradient_scales_by_exactly_w_on_rb_and_not_on_others"]),
    ("e_flag_without_gdvb_entry", TRAIN,
     '''    "rb_window_w": "cfg:rb_window_w",
''',
     '''''',
     [T + "test_a_flag_without_a_gdvb_entry_is_refused_by_the_gate_itself", T + "test_flag_default_is_one_has_a_flag_check_and_a_wm_only_default"]),
    ("f_flag_not_applied_in_make_config", TRAIN,
     '''                           rb_window_w=a.rb_window_w,                                                                   # PREREG_V8F_N1DW: the line that applies --rb-window-w (a mutation deletes it)
''',
     '''''',
     [T + "test_gdvb_passes_an_honoured_argv_and_names_the_flag_in_built_levers"]),
    ("g_mask_not_attached_to_the_batch", TRAIN,
     '''                    b["rb"] = rmask
''',
     '''                    pass
''',
     [T + "test_end_to_end_n1dw_arm_probe_levers_counts_and_checkpoint"]),
]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


rows = []
for mid, rel, old, new, tests in MUTANTS:
    if ONLY and mid not in ONLY:
        continue
    path = TREE / rel
    raw = path.read_bytes()
    before = hashlib.sha256(raw).hexdigest()
    text = raw.decode("utf-8")
    assert text.count(old) == 1, (mid, text.count(old))
    path.write_bytes(text.replace(old, new).encode("utf-8"))
    res = []
    try:
        for tid in tests:
            run_tmp = TMP / f"mut_{mid}"
            run_tmp.mkdir(parents=True, exist_ok=True)
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "PYTHONUTF8": "1", "TMP": str(run_tmp), "TEMP": str(run_tmp), "PYTHONPATH": str(TREE / "stack")}
            r = subprocess.run([PY, "-m", "pytest", tid, "-q", "-x", "-p", "no:cacheprovider", f"--basetemp={run_tmp / 'bt'}"], cwd=str(TREE / "stack"), env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=900)
            tail = [ln for ln in r.stdout.splitlines() if ln.startswith(("FAILED", "ERROR")) or " failed" in ln or " passed" in ln][-3:]
            res.append({"test": tid.split("::")[-1], "rc": r.returncode, "red": r.returncode != 0, "tail": tail})
    finally:
        path.write_bytes(raw)
    after = sha(path)
    rows.append({"mutant": mid, "file": rel, "edit": f"{len(old)} -> {len(new)} chars", "tests": res, "named_test_red": all(x["red"] for x in res), "any_named_test_red": any(x["red"] for x in res),
                 "file_sha256_before": before, "file_sha256_after_restore": after, "restored": before == after})
    print(mid, rows[-1]["named_test_red"], rows[-1]["restored"], flush=True)
OUT.write_text(json.dumps({"card_sha256": "59f2b7f919af958fd2da9c21f559c0f68db64add84e6fcb70cd5d6c7905c2301", "mutants": rows}, indent=1), encoding="utf-8")
