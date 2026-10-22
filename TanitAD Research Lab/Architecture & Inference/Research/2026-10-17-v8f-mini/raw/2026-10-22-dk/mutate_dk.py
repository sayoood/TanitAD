"""Mutation check for tests/test_v8f_dk.py: every mutant must turn the suite RED. Sandboxed copies; the originals are never edited.

  python mutate_dk.py  ->  raw/mutation_dk.json
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
TREE = HERE / "tree"
READER = HERE / "code" / "v8f_dk_read.py"
LM = TREE / "taniteval" / "taniteval" / "lead_metrics.py"
SB = HERE / "mut_sandbox"

MUTANTS = [
    # (name, kind, target-file, old, new)
    ("gap_line_deleted (per_step_gap)", "lm", LM, "    gap = along - float(lead_len) / 2.0\n", ""),
    ("gap_ignores_half_length (per_step_gap)", "lm", LM, "    gap = along - float(lead_len) / 2.0\n", "    gap = along\n"),
    ("headway_takes_max_not_min (distance_keeping)", "lm", LM, "        headway[i] = np.nanmin(g)", "        headway[i] = np.nanmax(g)"),
    ("lead_assignment_deleted (join_block)", "reader", READER,
     '    leads[is_lead] = np.asarray(blk["leads"], np.float64)[rows[is_lead]][:, :int(k)]\n', ""),
    ("speed_proof_removed (join_block)", "reader", READER, "    if not dmax <= speed_tol:", "    if False:"),
    ("contract_check_removed", "reader", READER, "    if not med <= CONTRACT_MEDIAN_TOL_M:", "    if False:"),
    ("no_label_scored_as_lead (join_block)", "reader", READER, '    is_lead = state == "LEAD"', '    is_lead = state != "NO_LABEL"'),
    ("join_by_position_not_key (join_block)", "reader", READER,
     "    rows = np.array([blk[\"_idx\"].get((c, int(f)), -1) for c, f in zip(clip.tolist(), frame.tolist())], np.int64)",
     "    rows = np.arange(clip.size, dtype=np.int64)"),
    ("human_reference_uses_the_arm_not_gt", "reader", READER,
     "    hdk = lm.distance_keeping(gt[:, :k], J[\"leads\"], J[\"lead_lens\"], J[\"speeds\"], dt)",
     "    hdk = lm.distance_keeping(np.asarray(next(iter(preds.values())), np.float64)[:, :k], J[\"leads\"], J[\"lead_lens\"], J[\"speeds\"], dt)"),
    ("paired_delta_sign_flipped", "reader", READER,
     'lm.paired_distance_keeping(dkS[a_name], dkS[b_name], eid, names=(a_name, b_name), n_boot=n_boot, seed=seed)["metrics"],',
     'lm.paired_distance_keeping(dkS[b_name], dkS[a_name], eid, names=(a_name, b_name), n_boot=n_boot, seed=seed)["metrics"],'),
    ("dt_doubled_in_arm_scoring", "reader", READER,
     "        dk = lm.distance_keeping(pk, J[\"leads\"], J[\"lead_lens\"], J[\"speeds\"], dt)\n        dkS[nm] = dk",
     "        dk = lm.distance_keeping(pk, J[\"leads\"], J[\"lead_lens\"], J[\"speeds\"], 2 * dt)\n        dkS[nm] = dk"),
    ("identity_control_removed", "reader", READER,
     '        if r.get("status") == "OK" and not (r["delta"] == 0.0 and r["lo"] == 0.0 and r["hi"] == 0.0):',
     "        if False:"),
    ("lead_viol_uses_arm_as_human (arm_block)", "reader", READER,
     "    lv = lm.lead_viol(dk, hdk, eid,", "    lv = lm.lead_viol(dk, dk, eid,"),
    ("ade_guard_removed (check_pred)", "reader", READER, "        if not dev <= TOL_ADE:\n            refuse(f\"{name}: ||pred", "        if False:\n            refuse(f\"{name}: ||pred"),
]


def run(env_extra, pytest_args):
    env = dict(os.environ)
    env.update({"CUDA_VISIBLE_DEVICES": "-1", "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"})
    env.update(env_extra)
    p = subprocess.run([PY, "-m", "pytest", str(HERE / "tests" / "test_v8f_dk.py"), "-q", "-p", "no:cacheprovider", "--tb=line", *pytest_args],
                       cwd=str(HERE), env=env, capture_output=True, text=True)
    tail = [l for l in p.stdout.splitlines() if l.strip()][-1] if p.stdout.strip() else p.stderr[-300:]
    return p.returncode, tail, p.stdout


def main():
    base_pp = f"{TREE / 'stack'};{TREE / 'taniteval'}"
    rc, tail, _ = run({"PYTHONPATH": base_pp}, [])
    out = {"baseline": {"rc": rc, "summary": tail}, "mutants": []}
    print("baseline:", rc, tail)
    if SB.exists():
        shutil.rmtree(SB)
    SB.mkdir()
    pkg_copy = SB / "taniteval_pkg"
    for i, (name, kind, target, old, new) in enumerate(MUTANTS):
        src = target.read_text(encoding="utf-8")
        assert src.count(old) == 1, f"{name}: target text found {src.count(old)} times"
        mut = src.replace(old, new)
        d = SB / f"m{i:02d}"
        d.mkdir()
        env = {}
        if kind == "lm":
            shutil.copytree(TREE / "taniteval" / "taniteval", d / "taniteval")
            (d / "taniteval" / "lead_metrics.py").write_text(mut, encoding="utf-8")
            env["PYTHONPATH"] = f"{d};{TREE / 'stack'};{TREE / 'taniteval'}"
        else:
            rp = d / "v8f_dk_read.py"
            rp.write_text(mut, encoding="utf-8")
            env["PYTHONPATH"] = base_pp
            env["V8F_DK_READER"] = str(rp)
        rc, tail, so = run(env, [])
        failed = [l.split("::")[-1].split(" ")[0] for l in so.splitlines() if l.startswith("FAILED") or l.startswith("ERROR")]
        out["mutants"].append({"mutant": name, "killed": rc != 0, "summary": tail, "red_tests": failed[:12]})
        print(f"{'KILLED  ' if rc != 0 else 'SURVIVED'} {name}: {tail}")
    out["n_mutants"] = len(MUTANTS)
    out["n_killed"] = sum(m["killed"] for m in out["mutants"])
    (HERE / "raw").mkdir(exist_ok=True)
    (HERE / "raw" / "mutation_dk.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    shutil.rmtree(SB)
    print(f"{out['n_killed']}/{out['n_mutants']} killed")
    return 0 if out["n_killed"] == out["n_mutants"] else 1


if __name__ == "__main__":
    sys.exit(main())
