#!/bin/bash
# R1 arms chain on Thor (SPEC_R1 REGISTERED sha256 e9873e12..., 2026-10-04T11:15:41Z). Runs ONLY after the capture
# chain's artifacts exist AND the identity gates pass. Every GPU pass is its own flock acquisition (<= ~40 min each,
# the shared-lock courtesy towards WP-RL); the ARTIFACT, never the exit code, says a pass ran.
# Order (MM 2026-10-04): timing smoke -> H0 eval (+ I-D) -> H1, H2, H2s, H3, H4, H5, H5c (train, eval) -> H0n eval -> score.
set -u
R=/home/nvidia/refcv8_r1
PY=/home/nvidia/venvs/tanitad-train/bin/python
LOCK=/home/nvidia/refcv7_post/thor_gpu.lock
export REFCV6_REPO=/home/nvidia/refcv7_run/fec3a0dccf REFCV6_KIT=/home/nvidia
export PYTHONPATH=/home/nvidia/refcv7_run/fec3a0dccf/stack:/home/nvidia/refcv7_run/fec3a0dccf/taniteval
export OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
A=$R/arms
mkdir -p $A $R/logs $R/status
cd $R/code
free_gb () { df --output=avail -BG / | tail -1 | tr -dc 0-9; }
gpu () {   # tag artifact attempts args...
  local tag=$1 art=$2 n=$3; shift 3
  local i
  for i in $(seq 1 $n); do
    if [ -s "$art" ]; then echo "[arms] $tag banked"; return 0; fi
    if [ "$(free_gb)" -lt 20 ]; then echo "[arms] $tag REFUSED: $(free_gb) GB free"; return 1; fi
    echo "[arms] $(date -u +%H:%M:%S) start $tag attempt $i"
    flock $LOCK $PY r1_arms.py "$@" >> $R/logs/arm_$tag.log 2>&1 < /dev/null
    echo $? > $R/status/arm_$tag.exit
    echo "[arms] $(date -u +%H:%M:%S) end $tag rc=$(cat $R/status/arm_$tag.exit) artifact=$( [ -s "$art" ] && echo yes || echo NO )"
    sleep 20
  done
  [ -s "$art" ]
}
# ---- gate: the capture artifacts and the identity controls I-0 / I-X ------------------------------------------- #
for f in $R/cache/eval/record.json $R/cache/train/record.json $R/cache/eval/identity_report.json; do
  [ -s $f ] || { echo "[arms] MISSING $f -- no arm runs"; exit 1; }
done
$PY - <<'EOF' || { echo "[arms] identity gate FAILED -- no arm runs"; exit 1; }
import json, sys
r = json.load(open("/home/nvidia/refcv8_r1/cache/eval/identity_report.json"))
ok = r["I0_capture_vs_bank_eval_s0"]["PASS"] and r["IX_exact_tensors"]["PASS"]
print("[arms] I-0", r["I0_capture_vs_bank_eval_s0"]["PASS"], "I-X", r["IX_exact_tensors"]["PASS"],
      "I-S", r["IS_stored_tensors"]["PASS"])
sys.exit(0 if ok else 1)
EOF
gpu timing $A/timing_smoke.json 2 --phase timing --timing-steps 30 || exit 1
cp $A/timing_smoke.json $R/raw/timing_smoke.json
gpu H0_eval $A/H0/eval_s1.pt 3 --phase eval --arm H0 --eval-seeds 0,1 || exit 1
$PY - <<'EOF' || { echo "[arms] I-D FAILED -- no trained arm runs"; exit 1; }
import glob, sys, torch
h0 = torch.load("/home/nvidia/refcv8_r1/arms/H0/eval_s0.pt", map_location="cpu", weights_only=False)
sh = sorted(glob.glob("/home/nvidia/refcv8_r1/cache/eval/shard_*.pt"))
st = {k: [] for k in ("store_sel", "store_selb", "store_s_e9", "b_r1_i")}
for s in sh:
    d = torch.load(s, map_location="cpu", weights_only=False)
    for k in st:
        st[k].append(d[k])
st = {k: torch.cat(v) for k, v in st.items()}
assert torch.equal(st["b_r1_i"], h0["b_r1_i"]), "window order"
n_sel = int((h0["sel"] != st["store_sel"]).sum()); n_selb = int((h0["selb"] != st["store_selb"]).sum())
d = float((h0["s_e9"] - st["store_s_e9"]).abs().max())
print(f"[arms] I-D: sel differ {n_sel}, selb differ {n_selb}, max|ds_e9| {d} over {len(h0['sel'])} windows")
open("/home/nvidia/refcv8_r1/arms/I_D.json", "w").write(
    '{"n": %d, "n_sel_differ": %d, "n_selb_differ": %d, "maxabs_s_e9": %r, "PASS": %s}'
    % (len(h0["sel"]), n_sel, n_selb, d, "true" if (n_sel == 0 and n_selb == 0 and d == 0.0) else "false"))
sys.exit(0 if (n_sel == 0 and n_selb == 0 and d == 0.0) else 1)
EOF
for arm in H1 H2 H2s H3 H4; do
  gpu ${arm}_train $A/$arm/train_record.json 3 --phase train --arm $arm || exit 1
  gpu ${arm}_eval $A/$arm/eval_s1.pt 3 --phase eval --arm $arm --eval-seeds 0,1 || exit 1
done
for arm in H5 H5c; do
  gpu ${arm}_train $A/$arm/train_record.json 3 --phase train --arm $arm || exit 1
  gpu ${arm}_eval0 $A/$arm/eval_s0.pt 3 --phase eval --arm $arm --eval-seeds 0 --ctrl || exit 1
  gpu ${arm}_eval1 $A/$arm/eval_s1.pt 3 --phase eval --arm $arm --eval-seeds 1 --ctrl || exit 1
done
gpu H0n_eval $A/H0n/eval_s1.pt 3 --phase eval --arm H0n --eval-seeds 0,1 || exit 1
$PY r1_score.py --cache $R/cache --arms $A > $R/logs/score.log 2>&1 < /dev/null
echo "[arms] score rc=$? ALLDONE $(date -u +%H:%M:%S)"
