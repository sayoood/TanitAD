#!/usr/bin/env bash
# P-RC21 v2 -- the >=GT TRUNCATION ARM, plus its reproduction control.
# Adapted from run_rc21.sh (2026-09-10). EVERY GATE IS KEPT, deliberately:
# the GPU-busy gate correctly refused a launch while a sibling held the card.
#
# THREE STAGES, in priority order so a kill at any point still yields value:
#   A) P1-REPRO  bar OFF, seed 0, 2000 steps -- MUST reproduce the banked
#                p1-grpo-2k bit-for-bit. The control for "default OFF is
#                unchanged"; without it the ON arm proves nothing.
#   B) P1-GTBAR  the SAME flags + --gt-bar. The pair differs by exactly one key.
#   C) P2-SEP    the hackable reward again, to bank the reward audit's new
#                SEPARATION verdict from the real pipeline rather than a unit test.
set -u

CK="C:/Users/Admin/tanitad-data/models/refc-base-30k/ckpt.pt"
WANT_MD5="8f10d6f934f4199e11ddc7352e074939"
WANT_BYTES="1250838325"
OUT="C:/Users/Admin/tanitad-data/rl-pilot"
REPO="C:/Users/Admin/tanitad-rlrun"
PY="C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1 OMP_NUM_THREADS=6
cd "$REPO" || exit 1
export PYTHONPATH="$REPO/stack"

TR_EP="C:/Users/Admin/tanitad-data/physicalai/_epcache/physicalai-train-14231cd29c74"
VA_EP="C:/Users/Admin/tanitad-data/physicalai/_epcache/physicalai-val-bb543bdf7836"

# --- PRE-LAUNCH GATE: assert on ARTIFACTS, never on a status code -------------
GOT_B=$(stat -c %s "$CK" 2>/dev/null)
if [ "$GOT_B" != "$WANT_BYTES" ]; then echo "ZZGATE-CKPT-BYTES-${GOT_B:-none}ZZ"; exit 2; fi
GOT=$(md5sum "$CK" | awk '{print $1}')
echo "[chain] ckpt md5 $GOT (want $WANT_MD5)"
if [ "$GOT" != "$WANT_MD5" ]; then echo "ZZCHAIN-MD5-REFUSEDZZ"; exit 3; fi

RESOLVED=$("$PY" -c "import tanitad;print(tanitad.__file__)" 2>&1 | tr -d '\r')
case "$RESOLVED" in
  *tanitad-rlrun*stack*tanitad*) echo "[chain] tanitad resolves to the run tree: $RESOLVED" ;;
  *) echo "ZZGATE-WRONG-TREE-${RESOLVED}ZZ"; exit 5 ;;
esac

# ⭐ THE FLAG MUST EXIST IN *THIS* TREE, not in the repo I edited. A shipped file
# that did not land is the stale-pod-checkout class; the marker is disjoint from
# the searched token so a PTY echo cannot match it.
HASFLAG=$("$PY" - <<'PYGATE'
import importlib.util, os
p = os.path.join("stack", "scripts", "rl_pilot_refc21.py")
s = importlib.util.spec_from_file_location("pilot_gate", p)
m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
ns = m.build_argparser().parse_args(["--ckpt","x","--train-epdir","x",
     "--train-agents","x","--val-epdir","x","--val-agents","x","--out","x"])
print("YES" if hasattr(ns, "gt_bar") and ns.gt_bar is False else "NO")
PYGATE
)
HASFLAG=$(echo "$HASFLAG" | tr -d '\r' | tail -1)
if [ "$HASFLAG" != "YES" ]; then echo "ZZGATE-NO-BAR-FLAG-${HASFLAG}ZZ"; exit 6; fi
echo "[chain] --gt-bar present in the run tree and defaults OFF"

# GPU must be free of python compute. The emitted marker is disjoint from the
# searched token on purpose.
BUSY=$(nvidia-smi --query-compute-apps=process_name --format=csv,noheader 2>/dev/null | grep -ci python)
if [ "${BUSY:-0}" -gt 0 ]; then echo "ZZCHAIN-GPU-BUSY-REFUSEDZZ"; exit 4; fi

mkdir -p "$OUT"

run_arm () {          # $1 = out-name  $2 = steps  $3 = reward  $4... = extra flags
  local name="$1" steps="$2" reward="$3"; shift 3
  echo "[chain] $name launch (steps=$steps reward=$reward extra='$*') $(date -u +%FT%TZ)"
  timeout 7200 "$PY" stack/scripts/rl_pilot_refc21.py \
    --ckpt "$CK" --train-epdir "$TR_EP" --train-agents "$OUT/pilot_train_agents.jsonl" \
    --val-epdir "$VA_EP" --val-agents "$OUT/pilot_val_agents.jsonl" \
    --out "$OUT/$name" --steps "$steps" --batch 2 --reward "$reward" --seed 0 "$@" \
    > "$OUT/$name.log" 2>&1
  local rc=$?
  # the ARTIFACT is the evidence, never $? -- a timed-out, output-less stage has
  # reported exit 0 in this programme before.
  if [ -s "$OUT/$name/pilot_summary.json" ]; then
    echo "ZZ${name}-OK-${rc}ZZ"
  else
    echo "ZZ${name}-NOSUMMARY-${rc}ZZ"
  fi
}

run_arm p1-grpo-2k-repro  2000 default
run_arm p1-grpo-2k-gtbar  2000 default --gt-bar
run_arm p2-reg-2k-sep      300 hackable
echo "ZZCHAIN-GTBAR-DONEZZ"
