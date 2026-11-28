#!/usr/bin/env bash
# pod_smoke_q.sh : ON THE POD, one CPU process in the foreground (timeout 840 s), note PREREG_V8F_PROOF_ADD40_N7 smoke: source cand-cvq on LS_s0 with its banked head, IP_MAX_WINDOWS windows (default 8),
# CUDA_VISIBLE_DEVICES=-1. Output goes into /workspace/v8proof/imagplan_q/smoke/ (a new folder of this build; nothing under runs/ or chains/ is touched). Refuses under 26 GB of free container memory
# and an existing output folder. Same command as `IP_DRY=1 pod_imagplan_run_q.sh ... cand-cvq`, with the output paths moved.
set -u
W=/workspace/v8proof; D=$W/imagplan_q; NEW=${1:-IPcqSmoke_LS_s0}; N=${IP_MAX_WINDOWS:-8}
O=$D/smoke/$NEW; TAG=${NEW}_m10_causal_heads; TREE=$W/tree_leadsup
M=$(cat /sys/fs/cgroup/memory.max); A=$(grep '^anon ' /sys/fs/cgroup/memory.stat | cut -d' ' -f2); F=$(( (M - A) / 1073741824 ))
[ "$F" -ge 26 ] || { echo "REFUSED: free container memory $F GB < 26 GB"; exit 2; }
[ -e "$O" ] && { echo "REFUSED: $O exists"; exit 2; }
mkdir -p "$O/ticks"
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
export IP_QHEAD_FILE=${IP_QHEAD_FILE:-$W/qhead_dose_ls/heads/LS_s0.qhead.npz}
[ -z "${SMOKE_CONST:-}" ] || export IP_QHEAD_CONST=$SMOKE_CONST
echo "free $F GB; windows $N; head $(sha256sum "$IP_QHEAD_FILE" | cut -c1-16); started $(date -u +%Y-%m-%dT%H:%M:%SZ)"
PYTHONPATH="$TREE/stack:$D:$W/distrule:$W/probe" timeout 840 /workspace/venv/bin/python $D/v8f_t1_eval_imagplan_q.py --tree $TREE --imag-plan cand-cvq \
  --imag-log "$O/ticks/imag_log_$NEW.npz" --floor card --fallback zero --cand-family decel run --ckpt $W/runs/leadsup_a/LS_s0/ckpt_last.pt --tag "$TAG" \
  --tracks $W/data/inputs/native_tracks.pt --eval-cache $W/data/cache_pod --eval-grid $W/hierrep/pins/eval_grid_windows.json --eval-join $W/hierrep/pins/join_eval141.npz \
  --out-dir "$O" --replan-steps 10 --nav deadreckon --scene logged --schedule periodic --limit causal --dump-ticks "$O/ticks" --dump-heads --device cpu --max-windows "$N" > "$O/t1.log" 2>&1
echo "rc $? finished $(date -u +%Y-%m-%dT%H:%M:%SZ)"
tail -n 4 "$O/t1.log" | cut -c1-240
ls "$O/ticks" | head -5
