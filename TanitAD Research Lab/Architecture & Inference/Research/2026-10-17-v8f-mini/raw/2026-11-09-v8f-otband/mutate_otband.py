"""OTBAND mutations: each copies the tree, applies ONE deliberate regression (every target text asserted to occur exactly once), runs the named tests, records red/green. RED = at least one test fails or errors.
Usage: python mutate_otband.py [name ...]   (no name = all). Result: logs/mutations.json next to this file. Pattern: theory/ladder/mutate.py."""
import json
import os
import shutil
import subprocess
import sys
import time

SRC = "F:/TanitAD-scratch/hwm_bands/b/tree/stack"
MUT = "F:/TanitAD-scratch/hwm_bands/b/mut"
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = HERE + "/logs/mutations.json"
M, T, Dd = "tanitad/models/v8f_mini.py", "scripts/v8f_mini_train.py", "tanitad/data/v8f_mini_data.py"
# (name, [(file, old, new), ...], -k expression)
MUTS = [
    ("a_code_detach_removed", [(M, "        if cfg.detach_upper:\n            code = code.detach()\n", "        if False:\n            code = code.detach()\n")], "with_detach_upper or probe_is_green"),
    ("b_checkpoint_detach_removed", [(M, "        if cfg.detach_upper:\n            ck_src = ck_src.detach()\n", "        if False:\n            ck_src = ck_src.detach()\n")], "with_detach_upper or probe_is_green"),
    ("c_op_total_line_deleted", [(M, "        total = total + cfg.w_op * T[\"op\"] if cfg.w_op > 0.0 else total", "        total = total")], "moves_the_total_by_exactly_its_weight or probe_is_green"),
    ("d_tac_total_line_deleted", [(M, "        total = total + cfg.w_tac_traj * T[\"tac\"] if cfg.w_tac_traj > 0.0 else total\n", "")], "moves_the_total_by_exactly_its_weight or probe_is_green"),
    ("e_code_wire_cut", [(M, "self.e_code(code), self.e_age(ages)]", "self.e_code(code * 0.0), self.e_age(ages)]")], "changing_t_code or zeroing_the_code or probe_is_green or stok_reaches"),
    ("f_stok_token_not_passed_to_t", [(M, "tout = self.run_t(b, adapt, last_t, ck_for_t, ck_ok, stok_tok=stok_tok)\n", "tout = self.run_t(b, adapt, last_t, ck_for_t, ck_ok)\n")], "stok_reaches or probe_is_green or sample_and_hold"),
    ("g_stok_read_at_now_not_s_tick", [(M, 'v, ok = self._at(b["stok"], last_s).detach().float(), self._at(b["stok_ok"], last_s).detach()', 'v, ok = b["stok"][:, -1].detach().float(), b["stok_ok"][:, -1].detach()')], "sample_and_hold"),
    ("h_stok_shuffle_is_identity", [(Dd, "self.source = {order[i]: order[(i + 1) % len(order)] for i in range(len(order))}", "self.source = {c: c for c in order}")], "shuffle_is_a_seeded_derangement or a_shuffle_declared or stok_reaches"),
    ("i_gate_never_refuses_oband", [(T, "    bad, info = [], data_info or {}\n    for nm in (\"w_op\"", "    return []\n    bad, info = [], data_info or {}\n    for nm in (\"w_op\"")], "the_gate_refuses or launch_refuses or v10_release"),
    ("j_make_config_drops_w_op", [(T, "w_op=a.w_op, w_tac_traj=a.w_tac_traj,", "w_tac_traj=a.w_tac_traj,")], "log_carries or launch_refuses or unbanded"),
    ("k_make_config_drops_stok_dim", [(T, "stok_dim=(D.stok_dim_of(a.t_stok_dir) if a.t_stok_dir else 0))", "stok_dim=0)")], "log_carries or unbanded or launch_refuses"),
    ("l_band_terms_computed_by_default", [(M, "        if cfg.w_op > 0.0 or cfg.w_tac_traj > 0.0:                    # PREREG_V8F_OTBAND: the per-band terms", "        if True:                    # PREREG_V8F_OTBAND: the per-band terms")], "off_is_the_tip or flag_off_records or logged_and_without"),
    ("m_band_edge_moved_one_knot", [(M, "KNOT_BAND = 10", "KNOT_BAND = 11")], "band_metrics_average or reaches_only_their_knots"),
    ("n_topk_ignores_rank", [(M, "c[:, a:b] = F.one_hot(idx[:, k] // N_LON, N_LAT)", "c[:, a:b] = F.one_hot(idx[:, 0] // N_LON, N_LAT)")], "topk"),
    ("o_trainer_ignores_freeze", [(T, "    if a.freeze and not a.wm_only:\n        freeze_groups(model, a.freeze)", "    if False:\n        freeze_groups(model, a.freeze)")], "frozen_t_and_s or launch_refuses or unbanded"),
]
only = sys.argv[1:]
os.makedirs(os.path.dirname(OUT), exist_ok=True)
res = json.load(open(OUT)) if os.path.exists(OUT) else {}
for name, edits, k in MUTS:
    if only and name not in only:
        continue
    d = f"{MUT}/{name}"
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d + "/stack/tests")
    for sub in ("tanitad", "scripts"):
        shutil.copytree(f"{SRC}/{sub}", f"{d}/stack/{sub}", ignore=shutil.ignore_patterns("__pycache__"))
    for t in ("conftest.py", "test_v8f_otband.py", "v8f_mini_tip_snapshot_otband.txt"):
        shutil.copy(f"{SRC}/tests/{t}", f"{d}/stack/tests/{t}")
    shutil.copytree(f"{SRC}/../taniteval", f"{d}/taniteval", ignore=shutil.ignore_patterns("__pycache__"))
    ok = True
    for f, old, new in edits:
        p = f"{d}/stack/{f}"
        s = open(p, encoding="utf-8", newline="").read()
        n = s.count(old)
        if n != 1:                       # files are LF (checked); a CRLF file would count 0 here and be reported, not hidden
            res[name] = {"status": "MUTATION_NOT_APPLIED", "occurrences": n, "target": old[:60]}
            ok = False
            break
        open(p, "w", encoding="utf-8", newline="").write(s.replace(old, new))
    if not ok:
        json.dump(res, open(OUT, "w"), indent=1)
        print(name, res[name]["status"], flush=True)
        continue
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8", PYTHONPATH=";".join([f"{d}/stack".replace("/", "\\"), f"{d}/stack/scripts".replace("/", "\\"), f"{d}/taniteval".replace("/", "\\")]))
    bt = f"{MUT}/bt_{name}"
    os.makedirs(bt, exist_ok=True)
    t0 = time.time()
    r = subprocess.run([PY, "-u", "-m", "pytest", "-q", "-p", "no:cacheprovider", f"--basetemp={bt}/b", "tests/test_v8f_otband.py", "-k", k], cwd=f"{d}/stack", env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    tail = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""
    failed = sorted({ln.split("::")[1].split(" ")[0] for ln in r.stdout.splitlines() if ln.startswith(("FAILED", "ERROR")) and "::" in ln})
    res[name] = {"status": "RED" if r.returncode != 0 else "GREEN", "rc": r.returncode, "k": k, "summary": tail, "tests_red": failed[:6], "seconds": round(time.time() - t0)}
    shutil.rmtree(d, ignore_errors=True)
    shutil.rmtree(bt, ignore_errors=True)
    json.dump(res, open(OUT, "w"), indent=1)
    print(name, res[name]["status"], tail, flush=True)
json.dump(res, open(OUT, "w"), indent=1)
