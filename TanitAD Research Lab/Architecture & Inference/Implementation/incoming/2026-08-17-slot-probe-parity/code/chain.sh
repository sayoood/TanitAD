#!/usr/bin/env bash
# Master chain. Each stage BANKS its own artifacts, so a kill at any point keeps
# everything already produced. Ordered by the brief's priority:
#   1 primary cells + C-SHUF at the LIVE checkpoint
#   2 the tokens arm (the D1-vs-D2 discriminator) on matched windows
#   3 the trajectory points
set -u
S="C:/Users/Admin/AppData/Local/Temp/claude/G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD/8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2/scratchpad/sp2"
SP="C:/Users/Admin/AppData/Local/Temp/claude/G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD/8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2/scratchpad/slotprobe"
PY="C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
export PYTHONUTF8=1 OMP_NUM_THREADS=6

say(){ echo "=== [chain] $* $(date +%H:%M:%S)"; }

# ---- STAGE 0: pipeline NULL CONTROL — needs the join only, no episodes -------
if [ ! -f "$S/out_null/slot_probe_results.json" ]; then
  say "STAGE 0 null control"
  "$PY" "$S/spX_fake_cache.py" "$S/lead130_agents.jsonl" "$S/cache_null" 4
  "$PY" "$S/sp2_probe.py" --cache "$S/cache_null/latents.pt" --out "$S/out_null" \
      --split-json "$S/p3_selection.json" --arms cells --n-queries 74 \
      --steps 3000 --batch 32 --lr 1e-3 --seed 0
  cp "$S/out_null/slot_probe_results.json" "$S/raw/results_NULLCONTROL.json"
fi

# ---- wait for the declared corpus and the live checkpoint --------------------
say "waiting for 130 episodes"
until [ "$(ls "$S/cache/slotprobe-lead130-w120-256x640cyl/"*.v2ep.pt 2>/dev/null | wc -l)" -ge 130 ]; do sleep 30; done
say "episodes ready"

# ---- STAGE 1: PRIMARY — cells at the LIVE checkpoint -------------------------
say "waiting for the live checkpoint"
until [ -f "$S/ck/v6F_sw_step011250.fp16.pt.local.md5" ]; do sleep 30; done
say "STAGE 1 primary cells @11250"
bash "$S/run_point.sh" s11250 "$S/ck/v6F_sw_step011250.fp16.pt" 4 0 74

# ---- STAGE 2: the tokens arm, matched windows -------------------------------
say "STAGE 2 tokens+cells @11250"
bash "$S/run_point.sh" tok11250 "$S/ck/v6F_sw_step011250.fp16.pt" 8 1 74

# ---- STAGE 3: the trajectory ------------------------------------------------
say "STAGE 3 trajectory"
bash "$S/run_point.sh" s09000 "$SP/w9000/weights_fp16_s9000.pt"        4 0 74
bash "$S/run_point.sh" s10000 "$S/ck/v6F_sw_step010000.fp16.pt"        4 0 74
bash "$S/run_point.sh" s09250 "$S/ck/v6F_sw_step009250.fp16.pt"        4 0 74
bash "$S/run_point.sh" s02000 "$SP/w/weights_fp16.pt"                  4 0 74

# ---- STAGE 4: n_queries sensitivity on the PRIMARY cache --------------------
say "STAGE 4 n_queries sensitivity (74 fitted vs 32 inherited)"
"$PY" "$S/sp2_probe.py" --cache "$S/cache_s11250/latents.pt" --out "$S/out_s11250_nq32" \
    --split-json "$S/p3_selection.json" --arms cells --n-queries 32 \
    --steps 3000 --batch 32 --lr 1e-3 --seed 0
cp "$S/out_s11250_nq32/slot_probe_results.json" "$S/raw/results_s11250_nq32.json"
say "CHAINDONE"
