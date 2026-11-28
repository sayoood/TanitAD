#!/usr/bin/env bash
# MM 2026-11-25: when P's post is OK on the pod (post_SFTP.status ZZSFTPPOST_OK), pull P's read inputs and write read_p/pod/PULLED.txt LAST.
# config / summary / metrics come from HF scorerft/p (so the HF bank is verified by the same act) and must equal the pod's sha256;
# HF's LFS sha256 of model_final.pt must equal the pod's model sha; the weight rule is pulled framed from the pod and must equal the census pin.
# Any mismatch => no PULLED.txt (sft_read_p.sh keeps waiting) and the reason is in the log. Polls by script, 10 min, give-up 36 h.
set -u
W=E:/Projects/TanitAD-artifacts/scorerft_read; D=E:/Projects/TanitAD-artifacts/scorer_gym_20261107/scorerft_read
OP=$W/read_p/pod; LOG=$W/pull_p.log; TMP=$W/tmp_pullp
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
PF=$D/pull_framed.py
RUN=/workspace/data/refe_runs/scorerft_p; ROOT=/workspace/data/refe_p5c
say() { echo "[$(date -u +%FT%TZ)] PULLP $*" >> "$LOG"; }
pd() { bash C:/Users/Admin/par18/pd.sh 60 "$1" 2>/dev/null | grep -v hf_ | tr -d '\r'; }
[ -s "$OP/PULLED.txt" ] && { say "PULLED.txt already present: nothing to do"; exit 0; }
say "start pid $$: waiting for the pod's post_SFTP.status"
n=0
while :; do
  st=$(pd "tail -n 3 $ROOT/post_SFTP.status 2>/dev/null" | grep -E "^ZZSFTPPOST_(OK|FAIL)" | tail -n 1)
  case "$st" in ZZSFTPPOST_OK*) say "post OK: ${st:0:80}"; break ;; ZZSFTPPOST_FAIL*) say "the pod post did not pass: ${st:0:200}"; exit 2 ;; esac
  n=$((n+1)); [ $n -gt 216 ] && { say "gave up after 216 polls (36 h)"; exit 3; }
  sleep 600
done
mkdir -p "$TMP"
ok=0
for i in 1 2 3; do
  pd "sha256sum $RUN/config.json $RUN/summary.json $RUN/metrics.jsonl; echo \"MODEL \$(cut -c1-64 $ROOT/model_final_sha256_SFTP.txt)\"" \
    | grep -E "^([0-9a-f]{64}  /workspace/|MODEL [0-9a-f]{64}$)" > "$TMP/pod_sha.txt"
  [ "$(wc -l < "$TMP/pod_sha.txt")" = 4 ] && { ok=1; break; }
  say "pod sha read attempt $i gave $(wc -l < "$TMP/pod_sha.txt") of 4 lines"; sleep 30
done
[ $ok = 1 ] || { say "could not read the pod's sha256 lines"; exit 4; }
"$PY" "$D/hf_get.py" Sayood/tanitad-refe-p5c "$TMP/hf" scorerft/p/config.json scorerft/p/summary.json scorerft/p/metrics.jsonl --lfs scorerft/p/model_final.pt \
  2>"$TMP/hf_get.err" | grep -v hf_ > "$TMP/hf_get.out" || { say "HF download failed: $(tail -n 1 "$TMP/hf_get.err" | grep -v hf_ | cut -c1-200)"; exit 5; }
grep "^LFS " "$TMP/hf_get.out" >> "$TMP/pod_sha.txt"
say "HF: $(tr '\n' ' ' < "$TMP/hf_get.out" | cut -c1-400)"
(MSYS_NO_PATHCONV=1 "$PY" "$PF" "$ROOT/scorerft/weight_rule_rank_p.json" "$TMP/weight_rule_rank_p.json") > "$TMP/pf.log" 2>&1 \
  || { say "framed pull of the weight rule failed: $(tail -n 1 "$TMP/pf.log" | cut -c1-200)"; exit 6; }
"$PY" "$D/build_p_inputs.py" "$TMP/hf" "$TMP/pod_sha.txt" "$TMP/weight_rule_rank_p.json" "$OP" > "$TMP/build.log" 2>&1 \
  || { say "$(tail -n 1 "$TMP/build.log" | cut -c1-300)"; exit 7; }
say "PULLED.txt written: $(head -n 1 "$OP/PULLED.txt")"
s=$(du -sm "$TMP" | cut -f1); rm -rf "E:/Projects/TanitAD-artifacts/scorerft_read/tmp_pullp"; say "deleted E:/Projects/TanitAD-artifacts/scorerft_read/tmp_pullp (${s} MB; HF and pod copies remain)"
say "done"
