# pod_leadscr_setup.sh : runs ON THE POD (bash -s). Card PREREG_V8F_PROOF_ADD39. Writes chain2/arms_leadscr_a.txt (two arms LCS trained from the start:
# the line of LS with the tree of card ADD35 and --lead-cond on), prints control s2 (argv of LCS against argv of LS), runs ONE dry gate (LCS_s0). Trains nothing.
set -u
W=/workspace/v8proof; cd $W/chain2 || exit 1
A=arms_leadscr_a.txt; CARD=PREREG_V8F_PROOF_ADD39
[ -e $A ] && { echo "SETUP-NOT-RUN: $A exists"; exit 1; }
n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .); [ "${n:-0}" -lt 3 ] || { echo "SETUP-NOT-RUN: $n GPU jobs running, a dry gate would be the fourth"; exit 1; }
a=$(grep -E '^anon ' /sys/fs/cgroup/memory.stat | cut -d' ' -f2); m=$(cat /sys/fs/cgroup/memory.max); [ $((m - a)) -gt 28000000000 ] || { echo 'SETUP-NOT-RUN: less than 28 GB of the memory limit free'; exit 1; }
echo "SETUP-RAN $(date -u +%Y-%m-%dT%H:%M:%SZ) with $n GPU jobs before, free $(( (m - a) / 1000000000 )) GB"
{
  echo "# LEADSCR ($CARD): arm LS of card ADD24 plus the lead read as an input of the plan heads (--lead-cond on), trained from the start. Control = LS_s0, LS_s1."
  grep '^PIN|' arms_leadcond_a.txt
  grep '^ARM|LS_s[01]|' arms_leadsup_a.txt | tr -d '\r' | sed -e 's/^ARM|LS_s/ARM|LCS_s/' -e 's#|/workspace/v8proof/tree_leadsup|#|/workspace/v8proof/tree_leadcond|#' \
    -e 's/--w-lead-sup 1.0|/--w-lead-sup 1.0 --lead-cond on|/' -e 's/levers.lead_sup_detach=off /levers.lead_sup_detach=off levers.lead_cond=on /'
} > $A
np=$(grep -c '^PIN|' $A); na=$(grep -c '^ARM|LCS_s[01]|/workspace/v8proof/tree_leadcond|HIER|[01]|' $A); nf=$(grep -c -- '--lead-cond on|' $A); ne=$(grep -c 'levers.lead_cond=on ' $A)
echo "arms file: PIN $np (12), ARM $na (2), flag $nf (2), expectation $ne (2); sha256 $(sha256sum $A | cut -c1-64)"
[ "$np" = 12 ] && [ "$na" = 2 ] && [ "$nf" = 2 ] && [ "$ne" = 2 ] || { echo "SETUP-STOP: the arms file is not as intended"; mv -n $A $A.bad; exit 1; }
grep '^ARM|' $A | cut -c1-700
# control s2: the argv the chain would use for LCS_s0 against the argv it used for LS_s0
T1=$(C2_CARD=PREREG_V8F_PROOF_ADD24 bash chain_arms.sh dryrun_LS_s0 arms_leadsup_a.txt --dry-run 2>&1 | grep '^TRAIN LS_s0: ' | sed 's|^TRAIN LS_s0: ||')
T2=$(C2_CARD=$CARD bash chain_arms.sh dryrun_LCS_s0 $A --dry-run 2>&1 | grep '^TRAIN LCS_s0: ' | sed 's|^TRAIN LCS_s0: ||')
/workspace/venv/bin/python - "$T1" "$T2" <<'PYEOF'
import sys
a, b = sys.argv[1].split(), sys.argv[2].split()
print("S2 argv tokens: LS_s0 %d, LCS_s0 %d" % (len(a), len(b)))
if not a or not b:
    print("S2 NOT READ: a TRAIN line is empty"); sys.exit(0)
i = 0; diffs = []
bb = list(b)
# remove the flag pair from the LCS argv, then compare position by position
if "--lead-cond" in bb:
    k = bb.index("--lead-cond"); flag = bb[k:k + 2]; del bb[k:k + 2]
else:
    flag = None
for x, y in zip(a, bb):
    if x != y: diffs.append((x[-70:], y[-70:]))
print("S2 flag pair in LCS argv:", flag, "| length equal after removing it:", len(a) == len(bb), "| differing tokens:", len(diffs))
for d in diffs: print("   ", d)
PYEOF
bash dry_gate.sh $A LCS_s0 $CARD > $W/chains/leadscr_dry_LCS_s0.out 2>&1; echo "dry gate LCS_s0 rc $?"; tail -n 2 $W/chains/leadscr_dry_LCS_s0.out | cut -c1-300
/workspace/venv/bin/python - <<'PYEOF'
import json
for a in ('LCS_s0',):
    try:
        s = json.load(open('/workspace/v8proof/runs/motion_dry/%s/out/selfcheck.json' % a)); lv = s.get('levers', {})
        print('SELFCHECK', a, 'dvb', s.get('declared_vs_built'), 'live', (s.get('live') or {}).get('pass'), 'steps_total', s.get('steps_total'), 'init', json.dumps(lv.get('init_ckpt'))[:120], 'lead_sup', lv.get('lead_sup'), 'lead_cond', lv.get('lead_cond'))
        print('   lead_cond probe', json.dumps(s.get('lead_cond_probe'))[:900])
        print('   live', json.dumps(s.get('live'))[:500])
    except Exception as e:
        print('SELFCHECK', a, 'ERR', e)
PYEOF
echo "--- how the chain makes its token"; grep -n 'gate_token' chain_arms.sh | cut -c1-330 | head -n 6
echo "SETUP-END $(date -u +%Y-%m-%dT%H:%M:%SZ)"
