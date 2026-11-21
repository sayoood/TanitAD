"""Source mutations of scripts/v6_chain.py that must turn tests/test_v6_chain_v7f.py RED.
Each mutation: exact-once string replace on the working file (EOL preserved) -> pytest -> record -> RESTORE
(sha256 re-verified). usage: python mutation_run.py <out_json>"""
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

STACK = Path("<V7F_CHAIN>/stack")
F = STACK / "scripts" / "v6_chain.py"
PY = sys.executable
orig = F.read_bytes()
orig_sha = hashlib.sha256(orig).hexdigest()
CRLF = b"\r\n"

MUTATIONS = {
    "M1_delete_the_emission_guard_call": (
        "        assert_v7f_argv(step, argv)\n    return argv",
        "        pass\n    return argv"),
    "M2_v7f_S_T_emits_tac_goal_cond": (
        '            init_from_key="S-W", prev_gate_key="S-W", max_horizon=60,\n'
        '            tac_goal_cond=False, extra=st_extra,',
        '            init_from_key="S-W", prev_gate_key="S-W", max_horizon=60,\n'
        '            tac_goal_cond=True, extra=st_extra,'),
    "M3_S_T_drops_plan_vmax_cap": (
        '                   "--max-speed-input-v6", "--plan-vmax-cap",\n',
        '                   "--max-speed-input-v6",\n'),
    "M4_S_W_fan_back_to_the_trainer_default": (
        '        cfg.n_candidates = 1              # R5 — overrides the dry fan of 3',
        '        pass                              # R5 — MUTATED: the fan is left at the v6 default'),
}

env = dict(os.environ)
env.update(PYTHONPATH=f"{STACK};{STACK.parent}", CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="4",
           PYTHONUTF8="1", MSYS_NO_PATHCONV="1")
res = {"file": str(F), "orig_sha256": orig_sha, "mutations": {}}
try:
    for name, (old, new) in MUTATIONS.items():
        text = orig.decode("utf-8").replace("\r\n", "\n")
        n = text.count(old)
        if n != 1:
            res["mutations"][name] = {"applied": False, "anchor_count": n}
            print(f"{name}: anchor count {n} -- NOT applied")
            continue
        F.write_bytes(text.replace(old, new).replace("\n", "\r\n").encode("utf-8"))
        p = subprocess.run([PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-rf",
                            "tests/test_v6_chain_v7f.py"], cwd=str(STACK), env=env,
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        F.write_bytes(orig)
        assert hashlib.sha256(F.read_bytes()).hexdigest() == orig_sha, "RESTORE FAILED"
        failed = sorted(set(re.findall(r"^FAILED (\S+)", p.stdout, re.M)))
        summary = [ln for ln in p.stdout.strip().splitlines() if " passed" in ln or " failed" in ln
                   or " error" in ln][-1:]
        res["mutations"][name] = {"applied": True, "rc": p.returncode, "summary": summary,
                                  "n_failed": len(failed), "failed": failed,
                                  "old": old, "new": new}
        print(f"{name}: rc={p.returncode} {summary} ({len(failed)} failed)")
        for f in failed:
            print("   RED", f)
finally:
    F.write_bytes(orig)
res["restored_sha256"] = hashlib.sha256(F.read_bytes()).hexdigest()
res["restored_ok"] = res["restored_sha256"] == orig_sha
Path(sys.argv[1]).write_text(json.dumps(res, indent=1), encoding="utf-8")
print("restored:", res["restored_ok"])
