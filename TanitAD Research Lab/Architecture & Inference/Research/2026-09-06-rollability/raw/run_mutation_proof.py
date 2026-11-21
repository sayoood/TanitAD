"""D-ROLL-1 — TWO-SIDED MUTATION PROOF, at the SOURCE, on a REAL checkpoint.

A fix that cannot be shown to fail is not evidenced, and a guard that cannot
fail measures nothing. So this harness:

  DEFECT   re-introduces the exact pre-fix construction
           (``if _vv != "kin3":``) and asserts the real instrument
           ``refcv3_arm.load_model`` REFUSES the real ``refcv4b@40284``
           checkpoint, AND that ``stack/tests/test_refc_v3_rollability.py``
           goes RED;
  FIXED    restores the gate and asserts both PASS.

⛔ IT MUTATES THE OFF-DRIVE MIRROR ONLY, never the repo, and restores in a
``finally`` with an md5 comparison whose operands are SHAPE-ASSERTED — two
empty strings compare equal, and a shell that reports MATCH on a failed read is
how a commit was once reported as landed when it had not.

ASCII-only output (cp1252 console).
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

WT = r"C:\Users\Admin\tanitad-wt"
TARGET = os.path.join(WT, "stack", "tanitad", "refs", "refc_v3.py")
PY = r"C:\Users\Admin\venvs\tanitad\Scripts\python.exe"
CKPT = r"C:\Users\Admin\refcv4b_final\ckpt_40284_FINAL.pt"

FIXED_GATE = ('        if _vv != "kin3" and bool(getattr(cfg, '
              '"tac_goal_tok_head", False)):\n')
DEFECT_GATE = '        if _vv != "kin3":\n'

PROBE = r"""
import importlib.util, json, os, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
WT = r"C:\Users\Admin\tanitad-wt"
for p in (os.path.join(WT, "stack"), WT):
    if p not in sys.path:
        sys.path.insert(0, p)
_p = os.path.join(WT, "taniteval", "tools", "refcv3_arm.py")
_s = importlib.util.spec_from_file_location("refcv3_arm_mut", _p)
arm = importlib.util.module_from_spec(_s); sys.modules["refcv3_arm_mut"] = arm
_s.loader.exec_module(arm)
try:
    model, cfg, targs, prov = arm.load_model(sys.argv[1], device="cpu")
except SystemExit as ex:
    print(json.dumps({"verdict": "REFUSED", "msg": str(ex)[:900]})); raise SystemExit(0)
sd = prov["state_dict_load"]
print(json.dumps({"verdict": "LOADED",
                  "n_missing": len(sd["missing_keys"]),
                  "n_unexpected": len(sd["unexpected_keys"]),
                  "inert": sd["tolerated_inert_buffers"],
                  "total": prov["param_breakdown"]["total"],
                  "has_tacgoal_line":
                      "tac_goal_tok_head" in prov["param_breakdown"],
                  "control_n_params": sum(p.numel() for p in model.parameters())}))
"""


def md5(path: str) -> str:
    with io.open(path, "rb") as fh:
        return hashlib.md5(fh.read()).hexdigest()


def write(path: str, text: str) -> None:
    with io.open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)


def probe_roll() -> dict:
    probe_py = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "_probe_roll.py")
    write(probe_py, PROBE)
    env = dict(os.environ, OMP_NUM_THREADS="6", PYTHONIOENCODING="utf-8")
    r = subprocess.run([PY, probe_py, CKPT], capture_output=True, text=True,
                       env=env, encoding="utf-8", errors="replace")
    last = [ln for ln in (r.stdout or "").splitlines() if ln.startswith("{")]
    if not last:
        return {"verdict": "PROBE-ERROR",
                "stderr": (r.stderr or "")[-700:], "rc": r.returncode}
    return json.loads(last[-1])


def probe_suite() -> dict:
    env = dict(os.environ, OMP_NUM_THREADS="6", PYTHONIOENCODING="utf-8",
               PYTHONPATH=os.path.join(WT, "stack") + ";" + os.path.join(WT, "taniteval"))
    r = subprocess.run(
        [PY, "-m", "pytest", "stack/tests/test_refc_v3_rollability.py", "-q",
         "--no-header", "-x"],
        cwd=WT, capture_output=True, text=True, env=env, encoding="utf-8",
        errors="replace")
    tail = [ln for ln in (r.stdout or "").splitlines() if ln.strip()][-1:]
    return {"rc": r.returncode, "tail": tail[0] if tail else "",
            "verdict": "GREEN" if r.returncode == 0 else "RED"}


def main() -> int:
    original = io.open(TARGET, encoding="utf-8", newline="").read()
    h0 = md5(TARGET)
    assert FIXED_GATE in original, "the fixed gate is not in the mirror"
    assert DEFECT_GATE not in original, "the defect gate is already present"
    out = {"target": TARGET, "md5_fixed": h0, "ckpt": CKPT}

    try:
        # ---------------------------------------------------------- FIXED ---
        out["FIXED"] = {"roll": probe_roll(), "suite": probe_suite()}
        print("FIXED   roll=%-8s suite=%-5s :: %s"
              % (out["FIXED"]["roll"]["verdict"],
                 out["FIXED"]["suite"]["verdict"],
                 out["FIXED"]["suite"]["tail"]))
        sys.stdout.flush()

        # --------------------------------------------------------- DEFECT ---
        write(TARGET, original.replace(FIXED_GATE, DEFECT_GATE))
        assert DEFECT_GATE in io.open(TARGET, encoding="utf-8").read()
        out["md5_defect"] = md5(TARGET)
        out["DEFECT"] = {"roll": probe_roll(), "suite": probe_suite()}
        print("DEFECT  roll=%-8s suite=%-5s :: %s"
              % (out["DEFECT"]["roll"]["verdict"],
                 out["DEFECT"]["suite"]["verdict"],
                 out["DEFECT"]["suite"]["tail"]))
        sys.stdout.flush()
    finally:
        write(TARGET, original)
        h1 = md5(TARGET)
        # SHAPE-ASSERT BOTH OPERANDS. Two failed reads both return "" and a
        # naive comparison then prints MATCH.
        if len(h0) != 32 or len(h1) != 32:
            print("RESTORE INCONCLUSIVE h0=%r h1=%r" % (h0, h1))
        elif h0 == h1:
            print("RESTORE VERIFIED", h1)
        else:
            print("RESTORE MISMATCH", h0, h1)

    ok = (out["FIXED"]["roll"]["verdict"] == "LOADED"
          and out["FIXED"]["suite"]["verdict"] == "GREEN"
          and out["DEFECT"]["roll"]["verdict"] == "REFUSED"
          and out["DEFECT"]["suite"]["verdict"] == "RED")
    out["TWO_SIDED_PROOF"] = "PASS" if ok else "FAIL"
    dst = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "mutation_proof.json")
    with io.open(dst, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print("TWO-SIDED PROOF:", out["TWO_SIDED_PROOF"], "->", dst)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
