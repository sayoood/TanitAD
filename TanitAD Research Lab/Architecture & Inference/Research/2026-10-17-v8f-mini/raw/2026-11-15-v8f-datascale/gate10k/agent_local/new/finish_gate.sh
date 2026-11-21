#!/bin/bash
# finish_gate.sh -- everything after the GPU half of the gate (pod). Detached; waits for gpu_gate.sh. Steps:
#   1 B_rep.A: the trainer's own --selfcheck-only on the stage-A argv (GPU, only while idle)
#   2 stageb_items.py (C-BANK / G-DAGGER on the real pilot bank, the mutated swap, the U smoke)       -> gate/stageb.json
#   3 assemble_evidence.py B_s0 B_rep U_s0                                                             -> gate/evidence_<arm>.json (refuses on any failed item or any regression arm not RED)
#   4 mint stage A for B_s0, U_s0, B_rep (chk_arm.sh mint: the checker writes a token only when every required check reads PASS)
#   5 verify_token.py (the runner's refusal logic on each token + 4 tampered variants that must be refused)
#   6 a DRY stage-B mint of B_s0 on the REAL pilot bank + pilot checkpoint into gate/tokens_dry/ (the positive path of the in-chain mint; this token binds the pilot checkpoint, it can never start a run)
# R_s0 is NOT minted. Ends with `ZZ FINISH-GATE DONE ZZ` or `ZZ FINISH-GATE FAIL ZZ <why>` in gate/finish_gate.log.
D=/root/dscale10k; G=$D/gate; PY=/root/venv/bin/python; TREE=$D/tree
export PYTHONPATH="$TREE/stack:$TREE/stack/scripts:$TREE:$D/study" HF_HUB_DISABLE_XET=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
fail() { echo "ZZ FINISH-GATE FAIL ZZ $1"; exit 1; }
gpu_idle() { u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '); [ "${u:-99999}" -lt 3500 ]; }
n=0; until grep -qE "ZZ GPU-GATE (DONE|FAIL) ZZ" $G/gpu_gate.log 2>/dev/null; do n=$((n+1)); [ $n -gt 1440 ] && fail "gpu_gate did not end in 4 h"; sleep 10; done
grep -q "ZZ GPU-GATE DONE ZZ" $G/gpu_gate.log || fail "gpu_gate: $(tail -n 1 $G/gpu_gate.log | cut -c1-300)"
echo "$(date -u +%T) gpu gate done"
# 1
n=0; until gpu_idle; do n=$((n+1)); [ $n -gt 360 ] && fail "GPU not idle before the B_rep selfcheck"; sleep 10; done
$PY - <<'E' || fail "B_rep selfcheck argv"
import json
a = [x for x in json.load(open("/root/dscale10k/argv/B_rep.A.json"))["argv"] if x != "--fit-join"]
i = a.index("--out"); a[i + 1] = "/root/dscale10k/gate/selfcheck_B_rep.A"
json.dump({"argv": a + ["--selfcheck-only"]}, open("/root/dscale10k/gate/selfcheck_B_rep.A.argv.json", "w"))
E
$PY $TREE/stack/scripts/v8f_mini_train.py $($PY -c "import json;print(' '.join(json.load(open('$G/selfcheck_B_rep.A.argv.json'))['argv']))") > $G/selfcheck_B_rep.A.log 2>&1 || fail "B_rep.A selfcheck rc: $(tail -c 300 $G/selfcheck_B_rep.A.log)"
echo "$(date -u +%T) B_rep.A selfcheck done"
# 2, 3
$PY $G/stageb_items.py || fail "stageb_items"
$PY $G/assemble_evidence.py B_s0 B_rep U_s0 || fail "assemble_evidence"
# 4, 5
for a in B_s0 B_rep U_s0; do
  bash $D/chk_arm.sh mint $a A > $G/mint_$a.A.log 2>&1 || fail "mint $a A refused: $(tail -c 600 $G/mint_$a.A.log)"
  tail -n 1 $G/mint_$a.A.log
  $PY $G/verify_token.py $a A || fail "verify_token $a A"
done
# 6: dry stage-B mint
mkdir -p $G/tokens_dry
$PY $D/study/dscale10k_check.py mint --arm B_s0 --stage B --argv $G/sb/B_s0.B.argv.json --root $TREE --commit cdbff971fa6dad4d20b937fe8abda6cb9f622adb --gate-dir $G/tokens_dry/B_s0.B --py $PY --gpu 0 \
   --base-argv $D/argv/B_s0.B.tmpl.json --evidence $G/evidence_B_s0.json --dry $D/dry/B_s0.A.dry.json --selfcheck $G/sb/B_s0.B/selfcheck.json --eval-ids /root/ops/clips141.txt \
   --holdout-manifest /root/data/cache_pod/train/manifest.jsonl --digest-json $(ls $TREE/stack/tanitad/data/*_clip_digests.json | tr '\n' ' ') > $G/mint_dry_B_s0.B.log 2>&1
echo "$(date -u +%T) dry stage-B mint of B_s0 (pilot bank + pilot checkpoint): $(tail -n 1 $G/mint_dry_B_s0.B.log | cut -c1-400)"
echo "ZZ FINISH-GATE DONE ZZ"
