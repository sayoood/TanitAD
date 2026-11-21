"""RED/GREEN — every guard shipped with C2 is removed one at a time and its test
must go RED, then the file is restored BIT-EXACT. A guard nobody has seen fail is
not a guard."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(r"G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD")
PY = r"C:/Users/Admin/venvs/tanitad/Scripts/python.exe"

MOD = REPO / "stack/tanitad/models/wm_reference_select.py"
EVAL = REPO / "stack/scripts/eval_flagship_v4.py"

CASES = [
    # (file, old, new, cwd, test target, what it proves)
    (MOD, "WM_REFERENCE_SELECT_DEFAULT = False", "WM_REFERENCE_SELECT_DEFAULT = True",
     "stack", "tests/test_wm_reference_select.py",
     "the default cannot be flipped ON while a measured arm is separated-WORSE"),
    (MOD, "return (fan - r[:, None].to(fan.dtype)).norm(dim=-1).mean(dim=-1)",
     "return (fan - r[:, None].to(fan.dtype)).norm(dim=-1).sum(dim=-1)",
     "stack", "tests/test_wm_reference_select.py",
     "the cost formula is pinned to the published cost_C2_ref"),
    (MOD, "return (fan - r[:, None].to(fan.dtype)).norm(dim=-1).mean(dim=-1)",
     "return (fan - r[:, None].to(fan.dtype)).norm(dim=-1).sum(dim=-1)",
     "taniteval", "tests/test_c2_published_policy.py",
     "a changed cost formula breaks fidelity to the 881-window picks"),
    (MOD, "wp, _ = rollout_decode(predictor, states, actions, None, step_readout, k)",
     "wp, _ = rollout_decode(predictor, states, actions, actions, step_readout, k)",
     "stack", "tests/test_wm_reference_select.py",
     "the reference roll cannot be handed future actions (the expert_future trap)"),
    (MOD, '    if n_candidates > 1:\n', '    if False:\n',
     None, None, "SKIP"),  # placeholder, replaced below
    (EVAL, 'default="as-trained",', 'default="c2-wm-ref",',
     "stack", "tests/test_eval_flagship_v4_select_rule.py",
     "the CLI default cannot silently become the new rule"),
    (EVAL, '    if select_rule == "c2-wm-ref" and c2 is None:',
     '    if False:',
     "stack", "tests/test_eval_flagship_v4_select_rule.py",
     "c2-wm-ref cannot run without a named scorer"),
    (MOD, '    if scorer is None:\n        raise ValueError(',
     '    if False:\n        raise ValueError(',
     "stack", "tests/test_wm_reference_select.py",
     "self-scoring cannot be reached by omission"),
]
CASES = [c for c in CASES if c[3] is not None]

out = {"_read": "each row REMOVES one guard and requires its test to FAIL",
       "cases": []}
for path, old, new, cwd, target, why in CASES:
    src = path.read_text(encoding="utf-8")
    h0 = hashlib.sha256(src.encode()).hexdigest()
    assert src.count(old) == 1, (path.name, old[:60], src.count(old))
    try:
        path.write_text(src.replace(old, new), encoding="utf-8")
        r = subprocess.run([PY, "-m", "pytest", target, "-q", "--no-header", "-x"],
                           cwd=REPO / cwd, capture_output=True, text=True)
        red = r.returncode != 0
        tail = [ln for ln in r.stdout.strip().splitlines() if ln.strip()][-1:]
    finally:
        path.write_text(src, encoding="utf-8")
    h1 = hashlib.sha256(path.read_text(encoding="utf-8").encode()).hexdigest()
    out["cases"].append({
        "file": str(path.relative_to(REPO)).replace("\\", "/"),
        "guard_removed": old.strip().splitlines()[0][:90],
        "suite": cwd, "test": target, "proves": why,
        "WENT_RED": red, "pytest_tail": tail,
        "restored_bit_exact": h0 == h1,
    })
    print(("RED   " if red else "!!GREEN") + f" {cwd}/{target}  <- {why}")

out["ALL_PASS"] = all(c["WENT_RED"] and c["restored_bit_exact"] for c in out["cases"])
Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
print("ALL_PASS:", out["ALL_PASS"])
