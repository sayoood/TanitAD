#!/bin/bash
# REPO COPY: the pod token-file path is redacted as <POD_TOKEN_FILE> (public repo); the live dev-box copy differs on that argument only.
# MM 2026-11-20: start a SCORER-FT arm's navtest pre-work the moment the pod posts it (= auto_field_seam.sh for the SCORER-FT arms).
#   poll the pod every 10 min: post_<ARM>.status (sft_post.sh rewrites it from START on each run) -> ZZ<ARM>POST_OK | ZZ<ARM>POST_FAIL
#   -> push model_final_sha256_<ARM>.txt to HF <prefix> with the project's egress tool (sft_post.sh writes it AFTER its push, so HF lacks it
#   and hf_pull.py needs it) -> guards (export done, E: >= 4 GB, C: >= 4 GB: the model goes to C:) -> detached WMI launch of
#   run_sft_seam.cmd <ARM> <prefix> <name> -> heartbeat. Prints exactly one ZZAUTOS_* line per terminal state.
# usage: auto_sft_seam.sh <ARM: SFTC|SFTT|SFTP|SFTCs|SFTTs|SFTPs> <HF prefix, e.g. scorerft/c> <seam name, e.g. scorerft_c>
set -u
A="${1:?arm}"; PFX="${2:?prefix}"; NAME="${3:?name}"
case "$A" in SFTC|SFTT|SFTP|SFTCs|SFTTs|SFTPs) ;; *) echo "ZZAUTOS_REFUSED bad arm $A"; exit 0;; esac
TAG="arm_$(echo "$PFX" | tr '/' '_')_shafile"
PD() { bash C:/Users/Admin/par18/pd.sh "$1" "$2" 2>/dev/null | sed -E 's/[0-9]{1,3}(\.[0-9]{1,3}){3}/<IP>/g; /hf_[A-Za-z0-9]{8,}/d'; }
ST="S=/workspace/data/refe_p5c/post_$A.status; [ -f \$S ] && grep -E 'ZZ${A}POST_OK|ZZ${A}POST_FAIL' \$S | tail -n 1"
for i in $(seq 1 432); do
  s="$(PD 60 "$ST")"
  case "$s" in
    *"ZZ${A}POST_FAIL"*) echo "ZZAUTOS_POST_FAIL $A $(date -u +%FT%H:%MZ) $s"; exit 0 ;;
    *"ZZ${A}POST_OK"*) break ;;
  esac
  sleep 600
done
case "$s" in *"ZZ${A}POST_OK"*) ;; *) echo "ZZAUTOS_TIMEOUT $A $(date -u +%FT%H:%MZ)"; exit 0 ;; esac
echo "$A posted: $s"
e="$(PD 300 "R=/workspace/data/refe_p5c; RUN=\$R/code_10848403/refe; PY=/workspace/venv-teacher/bin/python; L=\$R/chain.log; n0=\$(wc -l < \$L); [ -s \$R/model_final_sha256_$A.txt ] || { echo NO_SHAFILE; exit 0; }; cd \$RUN && env -u HF_TOKEN PYTHONPATH=\$RUN \$PY hf_egress.py push --repo Sayood/tanitad-refe-p5c --prefix $PFX --max-push-bytes 1e6 --tag $TAG --chain-log \$L --egress-log \$R/egress.jsonl --token-file <POD_TOKEN_FILE> --files \$R/model_final_sha256_$A.txt > \$R/egress_${A}_shafile.log 2>&1; tail -n +\$((n0 + 1)) \$L | grep -q ' ZZEGRESS_OK $TAG ' && echo SHAFILE_PUSHED")"
case "$e" in *SHAFILE_PUSHED*) echo "$A sha file on HF" ;; *) echo "ZZAUTOS_SHAFILE_FAIL $A $(date -u +%FT%H:%MZ) $e"; exit 0 ;; esac
[ -f F:/TanitAD_p5cread_tip/EXPORT_DONE.txt ] || { echo "ZZAUTOS_BLOCKED_EXPORT $A $(date -u +%FT%H:%MZ)"; exit 0; }
gE=$(df -k /e | tail -n 1 | awk '{print int($4/1048576)}'); gC=$(df -k /c | tail -n 1 | awk '{print int($4/1048576)}')
[ "$gE" -ge 4 ] && [ "$gC" -ge 4 ] || { echo "ZZAUTOS_BLOCKED_DISK $A $(date -u +%FT%H:%MZ) E: ${gE} GB C: ${gC} GB (need 4 each); $A is posted, sha file on HF"; exit 0; }
W=E:/Projects/TanitAD-artifacts/scorerft_read; mkdir -p "$W"
n0=$(cat $W/CHAIN.log 2>/dev/null | wc -l)
powershell -NoProfile -Command "\$hold = 'E:\Projects\TanitAD-artifacts\scorer_gym_20261107\scorerft_read'; \$cmd = 'cmd.exe /c \"\"' + \$hold + '\run_sft_seam.cmd\" $A $PFX $NAME\"'; \$si = (New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ CreateFlags = [uint32]0x08000200 }); \$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = \$cmd; CurrentDirectory = \$hold; ProcessStartupInformation = \$si }; 'launched: ReturnValue=' + \$r.ReturnValue + ' ProcessId=' + \$r.ProcessId"
sleep 90
n1=$(cat $W/CHAIN.log 2>/dev/null | wc -l)
if [ "$n1" -gt "$n0" ]; then echo "ZZAUTOS_LAUNCHED $A $(date -u +%FT%H:%MZ) E:${gE}GB C:${gC}GB heartbeat: $(tail -n 1 $W/CHAIN.log | cut -c1-200)"; else echo "ZZAUTOS_NO_HEARTBEAT $A $(date -u +%FT%H:%MZ) CHAIN.log did not grow in 90 s"; fi
