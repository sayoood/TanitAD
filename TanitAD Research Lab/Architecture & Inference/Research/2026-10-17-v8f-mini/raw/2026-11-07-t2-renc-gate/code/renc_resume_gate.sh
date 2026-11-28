#!/usr/bin/env bash
# RENC resume (Master Mind, 2026-11-07): the train,gate chain (logs/chain_train_gate.out) died silently after "ZZ RENC train LAUNCHED" at 07:29:20,
# while training finished on box 2 (RENC_TRAIN.done 09:48:42). This (1) moves aside any busy marker left by the dead chain (content checked),
# (2) runs the gate mode with RENC_RESUME=1 into logs/chain_gate_resume3.out, (3) pulls train's small files into the SAME pull dir and checks sizes,
# (4) appends ONLY the success line "ZZ RENC PULLED modes=train,gate -> <dir> ZZ" to chain_train_gate.out. Notes go to logs/renc_resume_gate.notes,
# never into chain_train_gate.out, because watchers grep that file for "ZZ RENC-" failure lines.
set -u
R=/e/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f/renc
O=$R/logs/chain_train_gate.out
L=$R/logs/chain_gate_resume3.out
N=$R/logs/renc_resume_gate.notes
B=/c/Users/Admin/box2_ship/BOX2_BUSY.txt
SSH="ssh -o BatchMode=yes -o ConnectTimeout=20 tanitad-box2"
F='post-quantum|store now|upgraded|pq.html'
TS=$(date +%Y%m%d_%H%M%S)
note() { echo "[resume $(date +%T)] $*" >> "$N"; }
if [ -f "$B" ]; then
  [ "$(head -1 "$B" | tr -d '\r')" = "owner=T2V8F renc train,gate" ] || { note "dev-box marker belongs to someone else: $(head -1 "$B"); stop"; exit 3; }
  mkdir -p /c/Users/Admin/box2_ship/busy_aside && mv "$B" "/c/Users/Admin/box2_ship/busy_aside/BOX2_BUSY.t2v8f_renc.deadchain.$TS.txt"
fi
b2=$($SSH "powershell -NoProfile -Command \"if (Test-Path D:/t2v8f/BOX2_BUSY.txt) { Get-Content D:/t2v8f/BOX2_BUSY.txt } else { 'ABSENT' }\"" 2>&1 | grep -vE "$F" | tr -d '\r' | head -1)
case "$b2" in
  ABSENT) ;;
  "owner=T2V8F renc") $SSH "powershell -NoProfile -Command \"New-Item -ItemType Directory -Force D:/t2v8f/busy_aside | Out-Null; Move-Item D:/t2v8f/BOX2_BUSY.txt D:/t2v8f/busy_aside/BOX2_BUSY.renc.deadchain.$TS.txt\"" >/dev/null 2>&1 ;;
  *) note "box-2 marker belongs to someone else: '$b2'; stop"; exit 3 ;;
esac
note "markers checked ($TS); gate mode resumes, log $L"
RENC_MODES=gate RENC_RESUME=1 RENC_BUILD=E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f/renc/build2 bash $R/renc_chain.sh < /dev/null > "$L" 2>&1
line=$(tr -d '\r' < "$L" | grep -E '^ZZ RENC PULLED modes=gate -> ' | tail -1)
[ -n "$line" ] || { note "gate resume did not pull: $(tr -d '\r' < "$L" | grep -E 'ZZ RENC' | tail -1)"; exit 8; }
PT=$(echo "$line" | sed -E 's/^ZZ RENC PULLED modes=gate -> (.*) ZZ$/\1/')
sftp -b - tanitad-box2 <<EOS 2>&1 | grep -vE "$F" | tail -3 >> "$N"
-get /D:/t2v8f/renc/out/TRAIN_RESULT_A.json $PT/TRAIN_RESULT_A.json
-get /D:/t2v8f/renc/out/TRAIN_RESULT_B.json $PT/TRAIN_RESULT_B.json
-get /D:/t2v8f/renc/out/RENC_TRAIN.done $PT/RENC_TRAIN.done
-get /D:/t2v8f/renc/out/E_A/E_A_lora_blob.pt $PT/E_A_lora_blob.pt
-get /D:/t2v8f/renc/out/E_B/E_B_lora_blob.pt $PT/E_B_lora_blob.pt
EOS
for f in TRAIN_RESULT_A.json:4540 TRAIN_RESULT_B.json:4548 RENC_TRAIN.done:10 E_A_lora_blob.pt:12616423 E_B_lora_blob.pt:12616423; do
  n=${f%%:*}; s=${f##*:}
  [ "$(stat -c %s "$PT/$n" 2>/dev/null)" = "$s" ] || { note "train pull size mismatch: $n"; exit 6; }
done
note "train small files pulled by hand into $PT (sizes match box 2)"
echo "ZZ RENC PULLED modes=train,gate -> $PT ZZ" >> "$O"
