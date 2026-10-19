"""Seal PREREG_SH5_DAC_PROX.md: fill the code-pin table, write CODE_PINS.json, record the prereg's sha256 and the time.  Run ONCE, after the design calculations, the simulations and the CPU tests; a second run
refuses (a sealed prereg changes only by a written Amendment).

    python seal.py --draft PREREG_SH5_DAC_PROX.draft.md --root C:/Users/Admin/par18/sh5
The draft must contain the line  {{CODE_PINS}}  (replaced by the table) and may contain {{SEALED_AT}}.
"""
import argparse
import datetime
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sh5_guard as G  # noqa: E402

PINNED = ["code/sh5_lib.py", "code/sh5_branch.py", "code/sh5_train_eval.py", "code/sh5_ctx_build.py", "code/sh5_gpulock.py", "code/sh5_admit.py", "code/sh5_gates.py", "code/sh5_synth.py",
          "code/sh5_design_calc.py", "code/sh5_cost.py", "code/sh5_sim_falsefire.py", "code/test_sh5_lib.py", "code/test_sh5_branch.py", "code/test_sh5_ctx.py", "code/test_sh5_pipeline.py", "code/run_tests.py"]
CHECKERS = ["code/sh5_guard.py", "code/seal.py"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", required=True)
    ap.add_argument("--root", required=True)
    a = ap.parse_args()
    root = a.root
    if os.path.exists(os.path.join(root, "PREREG_SH5_DAC_PROX.sha256")):
        raise SystemExit("already sealed: a sealed prereg changes only by a written Amendment")
    files = {n: G.sha256_file(os.path.join(root, n)) for n in PINNED}
    table = "| file | sha256 |\n|---|---|\n" + "\n".join(f"| `{n}` | `{h}` |" for n, h in files.items())
    now = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    txt = open(a.draft, encoding="utf-8").read()
    assert "{{CODE_PINS}}" in txt
    txt = txt.replace("{{CODE_PINS}}", table).replace("{{SEALED_AT}}", now)
    out = os.path.join(root, "PREREG_SH5_DAC_PROX.md")
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write(txt)
    with open(os.path.join(root, "CODE_PINS.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump({"sealed_at": now, "files": files}, f, indent=1)
    h = G.sha256_file(out)
    with open(os.path.join(root, "PREREG_SH5_DAC_PROX.sha256"), "w", encoding="utf-8", newline="\n") as f:
        f.write(f"{h}  PREREG_SH5_DAC_PROX.md\n")
    chk = {n: G.sha256_file(os.path.join(root, n)) for n in CHECKERS}
    with open(os.path.join(root, "SEALED_AT.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write(f"{now}\nPREREG_SH5_DAC_PROX.md sealed (see .sha256); {len(files)} code files pinned in CODE_PINS.json; no SH5 branch trained, no ctx built, no label array of this arm opened before this line\n")
        for n, c in chk.items():
            f.write(f"checker {n} sha256 {c}\n")
    print(json.dumps({"sealed_at": now, "prereg_sha256": h, "pinned": len(files)}))


if __name__ == "__main__":
    main()
