#!/usr/bin/env bash
# CHAIN — the latent linear ladder. One invocation per ARM; every arm runs the
# IDENTICAL target ladder, seeds and estimator, so the columns are comparable.
#
# ⭐ THE POSITIVE CONTROL RUNS FIRST AND IS NOT OPTIONAL. `cache_orcdir` is the
# banked GT-oracle memory on which the precedent measured a ridge at 1.016 m /
# r +0.979. Running it through THIS script proves the readout can measure at
# this n/p (2231 train windows, 2049 features) — which is the only thing that
# licenses reading a NEGATIVE row as a fact about the latent rather than about
# the instrument. That is the lesson the D1 withdrawal cost.
set -u
W="C:/Users/Admin/AppData/Local/Temp/claude/G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD/8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2/scratchpad/ll"
S="C:/Users/Admin/AppData/Local/Temp/claude/G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD/8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2/scratchpad/sp2"
P="C:/Users/Admin/AppData/Local/Temp/claude/G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD/8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2/scratchpad/pc"
PY="C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
EPS="$S/cache/slotprobe-lead130-w120-256x640cyl"
JOIN="$S/lead130_agents.jsonl"
export PYTHONUTF8=1 OMP_NUM_THREADS=4

run () {  # <tag> <cache> <label> [extra...]
  local tag="$1" cache="$2" label="$3"; shift 3
  "$PY" "$W/ll1_ladder.py" --cache "$cache" --split-json "$S/p3_selection.json" \
    --episodes-dir "$EPS" --join-file "$JOIN" --label "$label" \
    --out "$W/raw/ll_$tag.json" --seeds 0 1 2 "$@" \
    > "$W/log_$tag.txt" 2>&1
  echo "LL $tag rc=$?"
}

case "${1:-all}" in
  ctrl)
    # positive control (readout can measure) + the trivial-proxy control
    run orcdir  "$P/cache_orcdir/latents.pt"        "GT-ORACLE-DIRECT@11250"
    run proxyv0 "$S/cache_s11250/latents.pt"        "C-V0-PROXY@11250" --proxy-v0
    ;;
  main)
    run s11250     "$S/cache_s11250/latents.pt"     "v6F-SW-30k@11250"
    run nullmatched "$S/cache_nullmatched/latents.pt" "RANDOM-LATENT-NULL-MATCHED@11250"
    ;;
  ckpt)
    run s09000 "$S/cache_s09000/latents.pt" "v6F-SW-30k@9000"
    run s09250 "$S/cache_s09250/latents.pt" "v6F-SW-30k@9250"
    run s10000 "$S/cache_s10000/latents.pt" "v6F-SW-30k@10000"
    run s02000 "$S/cache_s02000/latents.pt" "v6F-SW-30k@2000"
    ;;
  egoorc)
    # ⭐ the SNR sweep of the readout's own positive control for the ANCHOR
    for nr in 0.1 1 3 10; do
      run "egoorc_n$nr" "$W/cache_egoorc_n$nr/latents.pt" "EGO-ORACLE-n$nr@11250"
    done
    ;;
  tokens)
    # ⭐ LOCALISATION: is ego motion absent from the ENCODER, or present in its
    # tokens and discarded by the 16x128 readout? Same ridge, same split, same
    # estimator; the feature set changes and so does n, both stated. The null
    # is the MATCHED-RANDOM feature set, so this extension carries its own
    # floor rather than borrowing the cells run's.
    run tok11250     "$S/cache_tok11250/latents.pt" "v6F-SW-30k@11250 TOKENS-MEAN" --features tokens_mean
    run tok11250null "$S/cache_tok11250/latents.pt" "TOKENS-MEAN MATCHED-RANDOM NULL" --features tokens_mean --randomise-features 20260817
    run cells_tokwin "$S/cache_tok11250/latents.pt" "v6F-SW-30k@11250 CELLS on the TOKENS window set"
    ;;
  repair)
    # the same ladder with the INTERCEPT REPAIR (see ll1_ladder._solve), on the
    # arm, the null and both controls — a repair that is not run on the null
    # has not been controlled.
    A="--fit-mode centred --alphas 0.01 0.1 1 10 100 1000 10000 100000 1000000 10000000"
    run rep_s11250     "$S/cache_s11250/latents.pt"        "v6F-SW-30k@11250 REPAIRED" $A
    run rep_nullmatched "$S/cache_nullmatched/latents.pt"  "RANDOM-LATENT-NULL REPAIRED" $A
    run rep_orcdir     "$P/cache_orcdir/latents.pt"        "GT-ORACLE-DIRECT REPAIRED" $A
    run rep_proxyv0    "$S/cache_s11250/latents.pt"        "C-V0-PROXY REPAIRED" --proxy-v0 $A
    ;;
esac
echo "CHAIN_DONE_${1:-all}"
