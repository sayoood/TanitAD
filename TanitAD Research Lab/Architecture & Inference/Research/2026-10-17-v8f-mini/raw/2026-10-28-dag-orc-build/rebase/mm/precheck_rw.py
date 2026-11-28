"""MM 2026-10-28: pre-check the lander's superset + demotion guards on every rewritten file of the DAG-ORC build hold, against the live tip.
python precheck_rw.py <tip>"""
import subprocess
import sys

TIP = sys.argv[1]
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
Q = "C:/Users/Admin/qland/"
CS = "E:/Projects/TanitAD-artifacts/dagland/ego_dagger_compose/stage/"
RS = "E:/Projects/TanitAD-artifacts/dagorc_rebase/chain/stage/"
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/"
FILES = [(CS, "stack/scripts/v8f_mini_train.py"), (CS, "stack/tanitad/data/v8f_mini_data.py"), (RS, PK + "code/v8f_mini_box2_queue.cmd")]
FILES += [(RS, f"stack/tests/test_v8f_{t}.py") for t in ("flat", "oacc", "kind", "mixo", "lora_ar2", "orc_chain")]
env = {"GIT_DIR": "C:/Users/Admin/tanitad-push/.git"}
import os
env = dict(os.environ, **env)
bad = 0
for base, rel in FILES:
    p = base + rel
    if not os.path.isfile(p):
        print("MISSING", rel); bad += 1; continue
    on_tip = subprocess.run(["git", "-c", "safe.directory=*", "cat-file", "-e", f"{TIP}:{rel}"], env=env).returncode == 0
    s = subprocess.run([PY, Q + "superset_check.py", f"{TIP}:{rel}", p, "none"], env=env, capture_output=True, text=True).stdout.strip().splitlines()
    d = subprocess.run([PY, Q + "demote_check.py", f"{TIP}:{rel}", p, "none"], env=env, capture_output=True, text=True).stdout.strip().splitlines()
    cr = open(p, "rb").read().count(b"\r")
    ok = on_tip and s and s[-1] == "ZZSUPERSET-OKZZ" and d and d[-1] == "ZZDEMOTE-OKZZ" and (cr == 0 or rel.endswith(".cmd"))
    bad += not ok
    print(f"{'OK ' if ok else 'BAD'} {rel.split('/')[-1]}: on_tip={on_tip} {s[-1] if s else '?'} {d[-1] if d else '?'} CR={cr}")
    if not ok:
        print("   ", "\n    ".join(s[-6:]))
print("ALL OK" if not bad else f"{bad} BAD")
