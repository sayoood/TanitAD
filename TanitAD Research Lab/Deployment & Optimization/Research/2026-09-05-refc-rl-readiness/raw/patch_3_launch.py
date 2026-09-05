"""Patch 3: launch_refcv3_rl_min.sh — humanflag gate in STAGE 0, --lead-mode on every
driver call, the real SPEC path, the work dir pinned. Applied to BOTH the run copy
(C:\\Users\\Admin\\rl_refcv3_min) and the WP copy."""
import os
import sys

SRC = sys.argv[1]           # the predecessor's script (scratchpad copy)
DSTS = sys.argv[2:]         # where to write the patched script

s = open(SRC, encoding="utf-8").read()


def rep(old, new, count=1):
    global s
    n = s.count(old)
    assert n == count, f"expected {count}, found {n}: {old[:60]!r}"
    s = s.replace(old, new)


rep("# SPEC:  TanitAD Research Lab/Deployment & Optimization/Research/2026-09-05-refc-rl-readiness/SPEC.md\n",
    "# SPEC:  TanitAD Research Lab/Deployment & Optimization/Research/2026-09-05-refc-rl-readiness/SPEC.md\n"
    "# RESUMED 2026-09-05 (Production/Deploy FlyWheel, second agent): STAGE 0 gained the\n"
    "# `humanflag` gate (H-RL-THRESH-1 class: the reward must not flag the human driver's\n"
    "# own future) and every driver call carries --lead-mode (default `track`: the lead's\n"
    "# own obstacle.offline track, time-aligned; `static` is the legacy comparison).\n")
rep('W="${RLMIN_WORK:-/c/Users/Admin/rl_refcv3_min}"             # work dir (local disk, never G:)\n',
    'W="/c/Users/Admin/rl_refcv3_min"                              # work dir (local disk, never G:)\n')
rep("BATCH=2; GROUP=4; NOISE=0.1; SEED=0; STRIDE=5; NBOOT=2000\n",
    "BATCH=2; GROUP=4; NOISE=0.1; SEED=0; STRIDE=5; NBOOT=2000\n"
    "LEAD_MODE=\"${LEAD_MODE:-track}\"; HUMAN_FLAG_MAX=0.15      # SPEC §3 / §4 (the humanflag gate)\n")
# STAGE 0: humanflag BEFORE the preflight
rep('PF_EPS="${PF_EPS:-$W/fit8}"; PF_LEAD="${PF_LEAD:-$W/fit8_lead_block.npz}"\nneed "$PF_LEAD"\n',
    'PF_EPS="${PF_EPS:-$W/fit8}"; PF_LEAD="${PF_LEAD:-$W/fit8_lead_block.npz}"\nneed "$PF_LEAD"\n'
    '# ⛔ the H-RL-THRESH-1 gate (0 GPU): the reward\'s own scene context must not flag the HUMAN future\n'
    '"$PY" "$DRV" --mode humanflag --lead-mode "$LEAD_MODE" --ckpt "$CKPT" --config "$CONFIG" --expect-step 40284 \\\n'
    '   --episodes "$PF_EPS" --labels "$TRAIN_LABELS" --lead-block "$PF_LEAD" --lru 8 \\\n'
    '   --human-flag-max "$HUMAN_FLAG_MAX" --out "$W/humanflag_fit8.json"\n'
    'need "$W/humanflag_fit8.json"; jsonok "$W/humanflag_fit8.json"\n'
    '"$PY" -c "import json,sys; r=json.load(open(sys.argv[1],encoding=\'utf-8\')); assert r[\'PASS\'], \'humanflag FAILED under \'+r[\'active_lead_mode\']" "$W/humanflag_fit8.json"\n')
rep('"$PY" "$DRV" --mode preflight --ckpt "$CKPT" --config "$CONFIG" --expect-step 40284 \\\n',
    '"$PY" "$DRV" --mode preflight --lead-mode "$LEAD_MODE" --ckpt "$CKPT" --config "$CONFIG" --expect-step 40284 \\\n')
rep('  "$PY" "$DRV" --mode arm --arm "$arm" --ckpt "$CKPT" --config "$CONFIG" --expect-step 40284 \\\n',
    '  "$PY" "$DRV" --mode arm --lead-mode "$LEAD_MODE" --arm "$arm" --ckpt "$CKPT" --config "$CONFIG" --expect-step 40284 \\\n')
rep('stage 0 "preflight: base md5 · imports · surface · no-future-leak · audit · anchor==0 · lr=0 timing"\n',
    'stage 0 "preflight: base md5 · imports · fitlist · HUMANFLAG gate · surface · no-future-leak · audit · anchor==0 · lr=0 timing"\n')
rep('echo "RLMIN_CHAIN_DONE  ->  $W/arms/verdict.json  (bank: verdict.json, paired_*.json/.md, eval/*.json, arms/*/arm_summary.json, preflight.json)"',
    'echo "RLMIN_CHAIN_DONE  ->  $W/arms/verdict.json  (bank: verdict.json, paired_*.json/.md, eval/*.json, arms/*/arm_summary.json, preflight.json, humanflag_fit8.json)"')
for d in DSTS:
    os.makedirs(os.path.dirname(d), exist_ok=True)
    open(d, "w", encoding="utf-8", newline="\n").write(s)
    print("wrote", d)
