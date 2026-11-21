#!/usr/bin/env bash
# CHAIN — E-R1-0, the pooling-ratio ladder (POOLING_BOTTLENECK_R1R2.md §7.1).
#
# ⛔ ZERO TRAINING. ZERO THOR. ZERO POD. One frozen banked checkpoint's token
# cache + the dev-box RTX 4060 for the projections; the ridge is CPU.
#
# ⭐ THE ORDER IS NOT ARBITRARY. `gate` runs FIRST because it is the only stage
# that can prove the harness reproduces the PRODUCER's committed output
# (`ll_cells_tokwin.json`) rather than agreeing with itself (C94). `pc*` (the
# planted positive controls) run before the reading of any negative, because a
# negative without a positive control taught this programme nothing (C79/D1).
set -u
SP="C:/Users/Admin/AppData/Local/Temp/claude/G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD/8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2/scratchpad"
REPO="G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD"
D="$REPO/TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-08-18-pooling-ladder-ER10"
PY="C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
CACHE="${ER10_CACHE:-$SP/sp2/cache_tok11250/latents.pt}"
EPS="$SP/sp2/cache/slotprobe-lead130-w120-256x640cyl"
JOIN="$SP/sp2/lead130_agents.jsonl"
SPLIT="$SP/sp2/p3_selection.json"
BANKED="$REPO/TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-08-17-latent-linear-ladder/raw/ll_cells_tokwin.json"
# ⛔ torch spawns ~113 threads PER PROCESS and concurrent arms then make NO
# progress (CLAUDE.md, MEASURED 2026-07-27). Capped here, not remembered.
export OMP_NUM_THREADS=6 PYTHONUTF8=1
mkdir -p "$D/raw"

run () {  # <tag> <extra...>
  local tag="$1"; shift
  "$PY" "$D/code/er10_pool_ladder.py" --cache "$CACHE" --split-json "$SPLIT" \
    --episodes-dir "$EPS" --join-file "$JOIN" --out "$D/raw/er10_$tag.json" \
    "$@" > "$D/raw/log_er10_$tag.txt" 2>&1
  echo "ER10 $tag rc=$?"
}

case "${1:-all}" in
  gate)
    # ⛔ J5 — reproduce the BANKED ladder through THIS harness on the INCUMBENT
    # biased solve. If this fails, nothing below is readable.
    run gate --label "REPRO-GATE cells vs ll_cells_tokwin (BIASED solve)" \
      --arms cells --proj-seeds 0 --legacy-penalised-intercept \
      --alphas 0.01 0.1 1 10 100 1000 10000 100000 --gate-json "$BANKED"
    ;;
  pc)
    # ⭐ the three PLANTED positive controls, on the four planted rungs
    for k in dist local local2; do
      run "pc_$k" --label "PC-${k}@11250" --arms p40 p10 p4 p1 \
        --targets ego_v0 lead_gap lead_closing ego_yawrate \
        --proj-seeds 0 1 2 --oracle "$k" --oracle-amp 1.0
    done
    ;;
  pcsweep)
    # ⭐ THE CALIBRATION THAT MAKES A FLAT LADDER READABLE. At amp 1.0x the
    # token sd a LOCALISED plant is recovered at r2 = 1.0000 by EVERY arm, so
    # it proves the harness reads but says nothing about SLOPE. Sweeping the
    # amplitude down finds the SNR band where pooling actually costs something
    # — i.e. how much ladder slope a genuinely localised signal WOULD make
    # under this exact random projection.
    for amp in 0.3 0.1 0.03 0.01 0.003; do
      run "pcloc_a$amp" --label "PC-LOCAL amp${amp}@11250" \
        --arms p40 p10 p4 p1 --targets lead_closing --proj-seeds 0 1 \
        --oracle local --oracle-amp "$amp" --n-boot 1000
    done
    ;;
  full)
    # ⛔ the NO-PROJECTION supplement: exact dual ridge on ALL features
    "$PY" "$D/code/er10_full_ridge.py" --cache "$CACHE" --split-json "$SPLIT" \
      --episodes-dir "$EPS" --join-file "$JOIN" \
      --out "$D/raw/er10_full.json" --label "v6F-SW-30k@11250 FULL-FEATURE" \
      --arms p40 p10 p4 p1 > "$D/raw/log_er10_full.txt" 2>&1
    echo "ER10 full rc=$?"
    ;;
  main)
    run main --label "v6F-SW-30k@11250 POOLING LADDER" \
      --arms p40 p10 p4 p1 cells --proj-seeds 0 1 2 3 4
    ;;
  null)
    run null --label "MATCHED-RANDOM NULL (per-arm mu/sd)" \
      --arms p40 p10 p4 p1 cells --proj-seeds 0 1 2 \
      --randomise-features 20260818
    ;;
  proxy)
    run proxyv0 --label "C-V0 TRIVIAL-PROXY (ego speed scalar alone)" \
      --arms p40 --proj-seeds 0 --proxy-v0
    ;;
  dino)
    # Job 2 — the corpus-narrowness discriminator: the SAME ladder, the SAME
    # windows, a FOREIGN frozen encoder. Point ER10_CACHE at its token cache.
    run dino --label "${DINO_LABEL:-FROZEN-EXTERNAL} POOLING LADDER" \
      --arms p40 p10 p4 p1 --proj-seeds 0 1 2
    ;;
esac
echo "CHAIN_DONE_${1:-all}"
