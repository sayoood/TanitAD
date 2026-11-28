#!/bin/bash
# deliver_reps2.sh -- DSCALE10K REPS2 DELIVERY (dev box, git-bash). NOT RUN by the builder. usage: bash deliver_reps2.sh UE_s2 [UE_s3]    (run it when the chain log reads `ZZ ARM <arm> DONE ZZ`; one arm at a time is fine)
# What it does, per arm, in this order, and refuses (rc 1, nothing pushed) at the first mismatch:
#   1 LoRA pod (read-only): runs/<arm>/ARM_DONE exists; sha256 of runs/<arm>/A/ckpt_last.pt = the first token of runs/<arm>/A/ARM_A_DONE; the step is 61568 (summary_check.json); the gate token's commit, argv sha256,
#     body sha256 and file sha256 are read; bank_ok/<arm>.A.sha256 (the HF private-repo copy, read back by the banker) is reported and, if present, must carry the SAME sha256.
#   2 pull ckpt_last.pt (87 MB) to F:/TanitAD-scratch/dscale10k/reps2/stage/<arm>/ with lorapod/pcat.sh (binary-clean); sha256 at the dev box = the pod's, else REFUSED.   [SHA CHECK 1: pod == dev box]
#   3 push it to the v8proof fork pod as /workspace/v8proof/integ10k/ckpt/<arm>/A/ckpt_last.pt.part, sha256 on the fork pod == the dev-box sha, then rename.     [SHA CHECK 2: dev box == fork pod]
#     An existing destination is never overwritten: same sha = skip the push, different sha = REFUSED.
#   4 write the ARM RECORD (arm_record.json: arm, commit, argv sha256 + argv, gate token path + file sha256 + body sha256, ckpt sha256 and step, summary seconds, selfcheck verdict, post gate, hub marker line, delivery time,
#     destination path) and a copy of the gate token beside it on the fork pod; sha256 of both read back on the fork pod.
#   5 delete the staged dev-box copy (re-creatable: the pod copy and the fork-pod copy both exist and were just verified) by literal absolute path; logged in DELETIONS.log (path, size).
# THE ROUTE (INHERITED, LOOP_STATE_2026-11-02.md line 3373, 2026-11-19 11:13Z): "Sent: UE_s0/A (2fd4b747) and UE_s1/A (f13df8ce) on <account>/tanitad-v8f-dscale10k-runs" = the banker's private HF copy (sha256 = the hub LFS oid),
#   which the coordinator fetched onto its pod. REPS2 keeps that hub copy (the chain's banker writes it and bank_ok) and ADDS the direct relay below, because the fork pod's HF credentials were not verified by the builder.
# Addresses are read from the helper files and never printed. Needs: C:/Users/Admin/lorapod/{podenv.sh,pcat.sh} (LoRA pod) and C:/Users/Admin/v8pod/{pod_direct.txt,pssh.sh} (fork pod).
set -u
HERE=F:/TanitAD-scratch/dscale10k/reps2
STAGE=$HERE/stage
DEST_ROOT=/workspace/v8proof/integ10k/ckpt
D=/root/dscale10k
source /c/Users/Admin/lorapod/podenv.sh
F8=/c/Users/Admin/v8pod/pod_direct.txt
H8=$(grep -oP '^host=\K.*' "$F8" | tr -d '\r'); P8=$(grep -oP '^port=\K.*' "$F8" | tr -d '\r')
SSH8="ssh -i $HOME/.ssh/tanitad_pod -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=30 -o StrictHostKeyChecking=accept-new -o LogLevel=ERROR -p $P8 root@$H8"
lp() { timeout 120 bash -c 'source /c/Users/Admin/lorapod/podenv.sh; pssh "$1"' _ "$1" < /dev/null; }       # LoRA pod, one command
fp() { timeout 300 $SSH8 "$1" < /dev/null; }                                                                  # fork pod, one command
die() { echo "DELIVER-REFUSED: $1"; exit 1; }
[ $# -ge 1 ] || die "usage: deliver_reps2.sh UE_s2 [UE_s3]"
mkdir -p "$STAGE"
for ARM in "$@"; do
  case $ARM in UE_s2|UE_s3) ;; *) die "arm $ARM is not UE_s2 / UE_s3";; esac
  echo "== $ARM"
  lp "test -e $D/runs/$ARM/ARM_DONE" || die "$ARM: no ARM_DONE on the LoRA pod (the chain has not finished it)"
  PSHA=$(lp "sha256sum $D/runs/$ARM/A/ckpt_last.pt" | awk '{print $1}')
  ASHA=$(lp "cut -d' ' -f1 $D/runs/$ARM/A/ARM_A_DONE" | tr -d '\r')
  [ -n "$PSHA" ] && [ "$PSHA" = "$ASHA" ] || die "$ARM: pod ckpt sha256 [$PSHA] != ARM_A_DONE [$ASHA]"
  SUMC=$(lp "cat $D/runs/$ARM/A/summary_check.json" | tr -d '\r')
  echo "$SUMC" | grep -q '"ckpt_step": 61568' || die "$ARM: summary_check.json does not read step 61568: $SUMC"
  HUB=$(lp "cat $D/bank_ok/$ARM.A.sha256 2>/dev/null" | tr -d '\r')
  if [ -n "$HUB" ]; then [ "$(echo $HUB | cut -d' ' -f1)" = "$PSHA" ] || die "$ARM: bank_ok sha [$(echo $HUB | cut -d' ' -f1)] != pod ckpt sha [$PSHA]"; else echo "NOTE $ARM: no bank_ok marker (HF copy not confirmed by the banker); the relay below is the only verified copy"; fi
  # the arm record's facts, read on the pod (one python call, nothing printed but JSON)
  lp "/root/venv/bin/python - <<'E'
import hashlib, json
D = '$D'; A = '$ARM'
tp = f'{D}/gate/tokens/{A}.A/gate_token.json'
t = json.load(open(tp))
s = json.load(open(f'{D}/runs/{A}/A/summary.json'))
rec = {'arm': A, 'stage': 'A', 'commit': t['commit'], 'argv_sha256': t['argv_sha256'], 'argv': json.load(open(f'{D}/argv/{A}.A.json'))['argv'],
       'gate_token': {'pod_path': tp, 'file_sha256': hashlib.sha256(open(tp, 'rb').read()).hexdigest(), 'body_sha256': t['body_sha256'], 'minted_at': t['at'], 'n_required_checks_pass': sum(1 for v in t['checks'].values() if v.get('pass'))},
       'ckpt_sha256': hashlib.sha256(open(f'{D}/runs/{A}/A/ckpt_last.pt', 'rb').read()).hexdigest(), 'ckpt_step': s['steps'], 'summary_seconds': s['seconds'], 'summary_seed': s['seed'], 'summary_argv_sha': s['argv_sha'],
       'selfcheck_declared_vs_built': json.load(open(f'{D}/runs/{A}/A/selfcheck.json')).get('declared_vs_built'), 'post_gate': json.load(open(f'{D}/runs/{A}/A/post.json'))}
json.dump(rec, open('/dev/stdout', 'w'), indent=1, sort_keys=True)
E" > "$STAGE/$ARM.record.json.raw" || die "$ARM: could not read the arm facts on the LoRA pod"
  grep -q '"commit"' "$STAGE/$ARM.record.json.raw" || die "$ARM: the arm facts are empty"
  mkdir -p "$STAGE/$ARM"
  # 2 pull
  bash /c/Users/Admin/lorapod/pcat.sh "$D/runs/$ARM/A/ckpt_last.pt" "$STAGE/$ARM/ckpt_last.pt" || die "$ARM: pull failed"
  LSHA=$(sha256sum "$STAGE/$ARM/ckpt_last.pt" | awk '{print $1}')
  [ "$LSHA" = "$PSHA" ] || die "$ARM: SHA CHECK 1 failed: dev box [$LSHA] != pod [$PSHA]"
  echo "$ARM SHA CHECK 1 ok (pod == dev box) $LSHA"
  # 3 push
  DD=$DEST_ROOT/$ARM/A
  EX=$(fp "sha256sum $DD/ckpt_last.pt 2>/dev/null" | awk '{print $1}')
  if [ -n "$EX" ]; then
    [ "$EX" = "$LSHA" ] || die "$ARM: $DD/ckpt_last.pt exists on the fork pod with a DIFFERENT sha [$EX]; never overwritten"
    echo "$ARM: already on the fork pod with the same sha (push skipped)"
  else
    fp "mkdir -p $DD" || die "$ARM: mkdir on the fork pod failed"
    timeout 900 $SSH8 "cat > $DD/ckpt_last.pt.part" < "$STAGE/$ARM/ckpt_last.pt" || die "$ARM: push failed"
    RSHA=$(fp "sha256sum $DD/ckpt_last.pt.part" | awk '{print $1}')
    [ "$RSHA" = "$LSHA" ] || die "$ARM: SHA CHECK 2 failed: fork pod [$RSHA] != dev box [$LSHA] (the .part file is left for inspection)"
    fp "mv $DD/ckpt_last.pt.part $DD/ckpt_last.pt" || die "$ARM: rename failed"
  fi
  echo "$ARM SHA CHECK 2 ok (dev box == fork pod) $LSHA"
  # 4 record + token copy
  C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I - "$STAGE/$ARM.record.json.raw" "$STAGE/$ARM/arm_record.json" "$DD/ckpt_last.pt" "$HUB" <<'E'
import json, sys, time
r = json.load(open(sys.argv[1])); r["delivered_to"] = sys.argv[3]; r["hub_marker"] = sys.argv[4] or None; r["delivered_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
r["route"] = "LoRA pod -> dev box (pcat, sha256) -> fork pod (ssh cat, sha256 read back); HF private-repo copy written by the chain banker (hub_marker)"
json.dump(r, open(sys.argv[2], "w"), indent=1, sort_keys=True)
E
  [ -s "$STAGE/$ARM/arm_record.json" ] || die "$ARM: arm record not written"
  lp "cat $D/gate/tokens/$ARM.A/gate_token.json" > "$STAGE/$ARM/gate_token_copy.json" || die "$ARM: token copy failed"
  for f in arm_record.json gate_token_copy.json; do
    L=$(sha256sum "$STAGE/$ARM/$f" | awk '{print $1}')
    timeout 120 $SSH8 "cat > $DD/$f" < "$STAGE/$ARM/$f" || die "$ARM: push of $f failed"
    R=$(fp "sha256sum $DD/$f" | awk '{print $1}')
    [ "$R" = "$L" ] || die "$ARM: sha of $f differs on the fork pod"
  done
  echo "$ARM record + token copy delivered and verified"
  # 5 delete the staged copy (re-creatable and verified at both ends), by literal absolute path
  SZ=$(stat -c %s "$STAGE/$ARM/ckpt_last.pt")
  rm -f "$HERE/stage/$ARM/ckpt_last.pt" && echo "$(date -u +%FT%TZ)	$HERE/stage/$ARM/ckpt_last.pt	$SZ	staged copy; pod and fork-pod copies sha256-verified" >> "$HERE/DELETIONS.log"
done
echo "ZZ DELIVER-REPS2 DONE ZZ"
