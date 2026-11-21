"""Mutation pass for the PREREG_V8F_SPEEDSUB additions (CPU only). Each mutant breaks ONE value-emitting line of the new code in a copy of the staged instrument; the staged
test file (test_v8f_speedsub.py) is run against the copy through SPEEDREAD_MODULE and the mutant is KILLED when pytest fails. A mutant whose replacement does not apply exactly once is
reported NOT APPLIED (a failure of this script, not a kill). Baseline (the unmutated instrument) must be green before and after.
    python mutate_speedsub.py <stage/stack> <scratch dir> <out json>
"""
import json
import os
import subprocess
import sys

STACK, SCR, OUT = (os.path.abspath(a) for a in sys.argv[1:4])
SRC = os.path.join(STACK, "scripts", "v8f_mini_eval_speedread.py")
TEST = os.path.join(STACK, "tests", "test_v8f_speedsub.py")
PY = sys.executable

MUT = [
    ("M01_ego_slice_last_frame_off_by_one", 'ok_all = _np(batch["frame_ok"]).astype(bool)\n    B, Fn = ok_all.shape[0], ok_all.shape[1]\n    a_t = _np(batch["a_t"]).astype(np.int64).reshape(B)\n    last = (Fn - 1) - a_t',
     'ok_all = _np(batch["frame_ok"]).astype(bool)\n    B, Fn = ok_all.shape[0], ok_all.shape[1]\n    a_t = _np(batch["a_t"]).astype(np.int64).reshape(B)\n    last = (Fn - 2) - a_t'),
    ("M02_ego_slice_nav_read_at_frame_0", '"nav": _np(batch["nav"])[b, last].astype(np.float32),\n            "limit": _np(batch["limit"]).astype(np.float32), "limit_ok": _np(batch["limit_ok"]).astype(np.float32).reshape(B, 1)}\n\n\ndef ego_vectors',
     '"nav": _np(batch["nav"])[b, 0].astype(np.float32),\n            "limit": _np(batch["limit"]).astype(np.float32), "limit_ok": _np(batch["limit_ok"]).astype(np.float32).reshape(B, 1)}\n\n\ndef ego_vectors'),
    ("M03_ego_slice_frame_ok_reversed", 'return {"ok": ok_all[b[:, None], pos], "p10f"', 'return {"ok": ok_all[b[:, None], pos][:, ::-1], "p10f"'),
    ("M04_ego_read_keys_admit_feats", 'EGO_READ_KEYS = ("p10f", "nav", "limit", "limit_ok", "frame_ok", "a_t")', 'EGO_READ_KEYS = ("p10f", "nav", "limit", "limit_ok", "frame_ok", "a_t", "feats")'),
    ("M05_mode_ego_reads_lin_ego", 'MODE_READOUT = {"ego": "mlp_ego", "egoshuf": "mlp_ego"}', 'MODE_READOUT = {"ego": "lin_ego", "egoshuf": "lin_ego"}'),
    ("M06_ego_branch_reads_the_scene_path", 'return phys_to_std(readout.predict_ego_batch(batch), mu, sd)\n    return phys_to_std(readout.predict_batch(batch), mu, sd)',
     'return phys_to_std(readout.predict_batch(batch), mu, sd)\n    return phys_to_std(readout.predict_batch(batch), mu, sd)'),
    ("M07_ego_cells_not_standardised", 'return phys_to_std(readout.predict_ego_batch(batch), mu, sd)', 'return np.asarray(readout.predict_ego_batch(batch), np.float64)'),
    ("M08_pass_bar_is_F_not_F_over_2", 'r["delta"] <= -F_pass for r in ego):', 'r["delta"] <= -2.0 * F_pass for r in ego):'),
    ("M09_harms_needs_both_seeds", 'if any(separated_worse(r) for r in ego):', 'if all(separated_worse(r) for r in ego):'),
    ("M10_regression_arm_refusal_dropped", 'if not all(separated_worse(rows[k]["egoshuf"]) for k in s):', 'if False:'),
    ("M11_separated_worse_counts_lo_equal_zero", 'return r["lo"] > 0.0\n\n\ndef controls_sub', 'return r["lo"] >= 0.0\n\n\ndef controls_sub'),
    ("M12_shuffle_seed_moved", 'return np.random.default_rng(SHUF_SEED).permutation(int(n))', 'return np.random.default_rng(SHUF_SEED + 1).permutation(int(n))'),
    ("M13_shuffle_does_not_permute", 'return rows[perm[i0:i0 + B]]', 'return rows[i0:i0 + B]'),
    ("M14_shuffle_counter_never_advances", 'self.n_seen += B\n            return rows[perm', 'self.n_seen += 0\n            return rows[perm'),
    ("M15_shuffle_alignment_not_measured", 'self.shuf_dev = max(self.shuf_dev, float(np.max(np.abs(live - rows[i0:i0 + B]))))', 'self.shuf_dev = 0.0'),
    ("M16_identity_control_lenient", 'c["S1_identity_eq_own"] = float(np.max(np.abs(a["identity"]["ade6"] - a["own"]["ade6"]))) == 0.0', 'c["S1_identity_eq_own"] = float(np.max(np.abs(a["identity"]["ade6"] - a["own"]["ade6"]))) <= 1.0'),
    ("M17_ego_path_control_lenient", 'c["S7_ego_path_eq_identity_pred"] = float(np.max(np.abs(ce["used_std"] - want))) <= 1e-5', 'c["S7_ego_path_eq_identity_pred"] = float(np.max(np.abs(ce["used_std"] - want))) <= 1e9'),
    ("M18_shuffle_alignment_control_lenient", 'sh["alignment_max_abs_dev"] <= 1e-6', 'sh["alignment_max_abs_dev"] <= 1e9'),
    ("M19_selection_rule_control_wrong_arm", 'min(sel, key=sel.get) == "mlp_ego"', 'min(sel, key=sel.get) == "lin_ego"'),
    ("M20_subevals_runs_the_speedread_flow", 'a.flow = "sub"\n    return cmd_evals(a)', 'a.flow = "read"\n    return cmd_evals(a)'),
    ("M21_pass_constant_is_F", 'F_SUB_PASS = F_M / 2.0 ', 'F_SUB_PASS = F_M * 1.0 '),
    ("M22_ego_mirror_check_dropped", 'self.mode not in EGO_MODES or bool(np.array_equal(t_tick_ego(batch)["ok"], ok))', 'True'),
    ("M23_sub_identity_extras_are_the_full_set", 'keys = SUB_IDENTITY_EXTRAS if sub else', 'keys = SUB_IDENTITY_EXTRAS if False else'),
    ("M24_read_json_control_dropped", 'c["S6b_gt_delta_eq_speedread_read_json"] = {s:', 'c["S6b_dropped"] = {s:'),
    ("M25_shuffle_source_reads_the_wrong_cells", 'rows = np.load(p)["pred_mlp_ego"].astype(np.float64)', 'rows = np.load(p)["pred_const"].astype(np.float64)'),
    ("M26_sub_dump_block_drops_the_readout_key", '"readout_key": MODE_READOUT.get(mode, mode), ', ''),
    ("M27_gt_reproduction_control_lenient", 'c["S6_gt_eq_speedread_gt"] = b is not None and float(np.max(np.abs(a["gt"]["ade6"] - b["gt"]))) <= tol_gt', 'c["S6_gt_eq_speedread_gt"] = True'),
    ("M28_null_called_when_one_seed_harms", 'if any(separated_worse(r) for r in ego):\n        return "HARMS"', 'if any(separated_worse(r) for r in ego):\n        return "NULL"'),
]


def run(module, label, tmp):
    env = dict(os.environ, SPEEDREAD_MODULE=module, CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="4")
    bt = os.path.join(SCR, "bt_" + label)
    r = subprocess.run([PY, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", f"--basetemp={bt}", TEST], env=env, capture_output=True, text=True, cwd=os.path.dirname(TEST))
    tail = [ln for ln in (r.stdout + r.stderr).splitlines() if ln.strip()][-1:]
    return r.returncode, tail[0] if tail else ""


os.makedirs(SCR, exist_ok=True)
src = open(SRC, encoding="utf-8", newline="").read()
res = {"baseline_before": None, "mutants": [], "baseline_after": None, "source_sha256": None}
import hashlib  # noqa: E402
res["source_sha256"] = hashlib.sha256(src.encode("utf-8")).hexdigest()
res["baseline_before"] = run(SRC, "base0", SCR)
print("baseline", res["baseline_before"], flush=True)
if res["baseline_before"][0] != 0:                                    # a harness that cannot run the tests must not report kills
    sys.exit("REFUSED: the unmutated instrument is not green; no mutant is run")
for mid, old, new in MUT:
    n = src.count(old)
    if n != 1:
        res["mutants"].append({"id": mid, "status": "NOT APPLIED", "occurrences": n})
        print(mid, "NOT APPLIED", n, flush=True)
        continue
    p = os.path.join(SCR, mid + ".py")
    open(p, "w", encoding="utf-8", newline="").write(src.replace(old, new))
    rc, tail = run(p, mid, SCR)
    st = {0: "SURVIVED", 1: "KILLED"}.get(rc, "ERROR")                  # only a test FAILURE (rc 1) is a kill; a usage / collection error is not
    res["mutants"].append({"id": mid, "status": st, "pytest_rc": rc, "tail": tail})
    print(mid, st, "|", tail, flush=True)
res["baseline_after"] = run(SRC, "base1", SCR)
res["summary"] = {k: sum(1 for m in res["mutants"] if m["status"] == k) for k in ("KILLED", "SURVIVED", "NOT APPLIED", "ERROR")}
res["summary"]["n"] = len(res["mutants"])
json.dump(res, open(OUT, "w", encoding="utf-8", newline="\n"), indent=1)
print("summary", res["summary"], "baseline_after", res["baseline_after"], flush=True)
