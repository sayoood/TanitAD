#!/bin/bash
# rh_refs_chain.sh -- ROLLHIST stage-A references under the form of record (Amendment 2 item 5). Dev-box CPU, 2 threads, one rollout at a time, each gated on commit >= 5.5 GB and F: >= 6 GB.
# 7 rollouts: UE_s0.A and UE_s1.A x R0G50Hb20, R0G0Hb20, R0C10Hb20; UE_s1.A x R0C10XHb20.  Resumable (out/steps_refs/*.ok and T1_DONE).
# Last log line: "ZZ ROLLHIST-REFS DONE ZZ" or "ZZ ROLLHIST-REFS FAIL ZZ <why>". Every line goes to the log with >> "$LOG", never | tee.
export PATH="/usr/bin:/bin:$PATH"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8 HF_HUB_DISABLE_XET=1
E=F:/TanitAD-scratch/rollhist/read
PY=${RH_PY:-C:/Users/Admin/venvs/tanitad/Scripts/python.exe}
O=$E/out; STEPS=$O/steps_refs; LOG=$O/rh_refs_chain.log
export TMP=$E/tmp TEMP=$E/tmp
WC=F:/TanitAD-scratch/strat1/read/seeds/wait_commit.py
cd "$E" || exit 1
mkdir -p "$O" "$STEPS" "$E/tmp" "$E/ckpts/UE_s1.A"
fail() { echo "ZZ ROLLHIST-REFS FAIL ZZ $1" >> "$LOG"; echo "FAIL $1" > "$O/ROLLHIST_REFS.done"; exit 1; }
gate() { $PY $WC 5.5 60 >> "$LOG" 2>&1 || fail "commit headroom stayed below 5.5 GB for 60 min before: $1"
         fk=$(df -k /f | awk 'NR==2{print int($4/1048576)}'); [ "${fk:-0}" -ge 6 ] || fail "F: free ${fk} GB < 6 GB before: $1"; }
step() { local n=$1 d=$2; shift 2; if [ -f "$STEPS/$n.ok" ]; then echo "[skip] $n" >> "$LOG"; return 0; fi
  echo "[step] $n: $d $(date)" >> "$LOG"; gate "$n"; "$@" >> "$LOG" 2>&1 || fail "$n ($d): see the log above"; echo ok > "$STEPS/$n.ok"; }
echo "[rollhist refs chain] start $(date) tree $E/tree (43319227)" >> "$LOG"
pull_ue1() {
  local want=f13df8ce026eed31e0d9d1df72de3e7b7977fefa6da3def2b411b7d74dbd4ebc
  if [ -f "$E/ckpts/UE_s1.A/ckpt_last.pt" ] && [ "$(sha256sum "$E/ckpts/UE_s1.A/ckpt_last.pt" | cut -d' ' -f1)" = "$want" ]; then echo "UE_s1.A present, sha ok"; return 0; fi
  export HF_TOKEN="$(grep -oE 'hf_[A-Za-z0-9]+' E:/Projects/TanitAD/Keys.txt | head -1)"
  $PY -c "
import os,shutil
from huggingface_hub import hf_hub_download
p=hf_hub_download('Sayood/tanitad-v8f-dscale10k-runs','UE_s1/A/ckpt_last.pt',repo_type='dataset',local_dir=os.environ['TMP']+'/hfdl')
shutil.move(p,'$E/ckpts/UE_s1.A/ckpt_last.pt')
" || return 1
  [ "$(sha256sum "$E/ckpts/UE_s1.A/ckpt_last.pt" | cut -d' ' -f1)" = "$want" ] || { echo "UE_s1.A sha mismatch"; return 1; }
  echo "UE_s1.A pulled, sha ok"
}
step pull_ue1 "pull UE_s1.A from HF, check sha256" pull_ue1
for s in 0 1; do for f in R0G50Hb20 R0G0Hb20 R0C10Hb20; do
  step "ref_UEA_${f}_s$s" "rollout UEA $f seed $s" $PY -u $E/rh_roll.py UEA $f $s
done; done
step "ref_UEA_R0C10XHb20_s1" "rollout UEA R0C10XHb20 seed 1" $PY -u $E/rh_roll.py UEA R0C10XHb20 1
echo "DONE" > "$O/ROLLHIST_REFS.done"
echo "ZZ ROLLHIST-REFS DONE ZZ" >> "$LOG"
