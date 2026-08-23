#!/usr/bin/env bash
# Thor drift probe — READ-ONLY, and NEVER `git fetch`.
#
# ⛔ WHY NOT `git log`. Thor's `git rev-parse HEAD` reports 30d6d60 while its
# working tree is a FILE-SHIPPED PATCHWORK: `train_v6_staged.py` is commit
# be3b89b (2026-08-14) and `v6.py` is 30d6d60 (2026-08-15), in the same
# checkout. And `git fetch` HANGS on Thor (no credentials), while a
# `checkout -B` after a failed fetch would RESET the tree to that ancient HEAD
# and destroy every shipped file. So: md5s, then a REAL IMPORT.
#
# ⚠️ CPU ONLY. `CUDA_VISIBLE_DEVICES=""` is not a nicety — Thor is training and
# nothing here may touch its GPU.
#
# Usage:  bash thor_import_probe.sh [host]      (default tanitad-thor-wifi)
# Re-run after any file-ship: every ⛔ row below must flip.
set -u
HOST="${1:-tanitad-thor-wifi}"
SSH="${SSH:-ssh}"

# ⛔ `ssh -n` is mandatory inside a script: a nested ssh EATS THE REST OF THE
# SCRIPT'S STDIN and the tail silently never runs (measured, cost two rounds).
"$SSH" -n -o ConnectTimeout=20 -o BatchMode=yes "$HOST" 'cd /home/nvidia/TanitAD && \
echo "=== MD5 ==="; md5sum \
  stack/scripts/train_v6_staged.py \
  stack/scripts/v6_chain.py \
  stack/tanitad/models/v6.py \
  stack/tanitad/models/tactical.py \
  stack/scripts/train_flagship_v4.py \
  stack/scripts/eval_flagship_v4.py \
  stack/tanitad/data/v2_dataset.py \
  stack/tanitad/data/parity.py \
  stack/tanitad/data/calib.py \
  stack/tanitad/geometry.py \
  taniteval/taniteval/seam.py \
  taniteval/taniteval/seam_dump.py \
  taniteval/tools/seam_probe.py 2>&1'

# The import probe. PYTHONPATH is deliberately set to EXACTLY what the launch
# line sets, so an import that only works under a wider path shows up as the
# failure it will be at run time.
"$SSH" -n -o ConnectTimeout=20 -o BatchMode=yes "$HOST" 'cd /home/nvidia/TanitAD/stack && \
PYTHONPATH=/home/nvidia/TanitAD/stack OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="" \
/home/nvidia/venvs/tanitad-train/bin/python - <<"PY"
import importlib, json, sys
sys.path.insert(0, "/home/nvidia/TanitAD/stack/scripts")
res = {}
def probe(name, fn):
    try:
        res[name] = {"ok": True, "value": fn()}
    except BaseException as e:
        res[name] = {"ok": False, "err": f"{type(e).__name__}: {str(e)[:160]}"}

probe("torch", lambda: importlib.import_module("torch").__version__)
def trainer():
    T = importlib.import_module("train_v6_staged")
    src = open("scripts/train_v6_staged.py").read()
    return {"STAGE_MAY_INTRODUCE": hasattr(T, "STAGE_MAY_INTRODUCE"),
            "RESUME_CONTRACT": hasattr(T, "RESUME_CONTRACT"),
            "STAGE_GATE_SPEC": hasattr(T, "STAGE_GATE_SPEC"),
            "cli_selector": "--selector" in src,
            "cli_tac_goal_cond": "--tac-goal-cond" in src,
            "cli_dump_seam_plan": "--dump-seam-plan" in src}
probe("train_v6_staged", trainer)
probe("v6", lambda: {"LADDER_UNTRAINED_GROUPS":
                     hasattr(importlib.import_module("tanitad.models.v6"),
                             "LADDER_UNTRAINED_GROUPS")})
probe("v6_probe_trunk", lambda: hasattr(
    importlib.import_module("tanitad.eval.v6_probe_trunk"), "is_v6_checkpoint"))
probe("v6_chain", lambda: importlib.import_module("v6_chain").__file__)
# ⭐ these three are the seam path, and they are the reason PYTHONPATH above is
# NOT widened: `taniteval` is a SIBLING of stack/, so the launch line as written
# cannot import it and --dump-seam-plan banks nothing while logging a failure.
for m in ("taniteval", "taniteval.ci", "taniteval.seam", "taniteval.seam_dump"):
    probe(m, (lambda mm: (lambda: importlib.import_module(mm).__file__))(m))
# ⛔ opaque marker: the emitted token must be DISJOINT from anything a
# client-side filter searches for (the self-matching-monitor trap).
print("ZZJSONZZ" + json.dumps(res) + "ZZENDZZ")
PY'
