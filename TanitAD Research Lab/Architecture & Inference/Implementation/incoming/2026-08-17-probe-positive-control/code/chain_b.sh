#!/usr/bin/env bash
# CHAIN B — TASK 2 stratification, run on THREE arms so the pattern can be
# attributed: the live checkpoint, an earlier one, and the window-matched
# RANDOM-LATENT NULL. If the null shows the SAME stratum profile, the profile is
# a property of the LABEL/BASELINE, not of the latent — and saying so requires
# having run it.
set -u
W="C:/Users/Admin/AppData/Local/Temp/claude/G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD/8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2/scratchpad/pc"
S="C:/Users/Admin/AppData/Local/Temp/claude/G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD/8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2/scratchpad/sp2"
PY="C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
export PYTHONUTF8=1 OMP_NUM_THREADS=4
run () {  # <tag> <cachedir> <outdir> <expect> <label>
  "$PY" "$W/pc3_stratify.py" --cache "$S/$2/latents.pt" \
    --head "$S/$3/head_cells.pt" --split-json "$S/p3_selection.json" \
    --expect-json "$S/raw/$4" --out "$W/raw/pc3_strata_$1.json" \
    --label "$5" > "$W/log_pc3_$1.txt" 2>&1
  echo "PC3 $1 rc=$?"
}
run s11250     cache_s11250     out_s11250     results_s11250.json     "v6F-SW-30k@11250"
run s09000     cache_s09000     out_s09000     results_s09000.json     "v6F-SW-30k@9000"
run nullmatched cache_nullmatched out_nullmatched results_NULLMATCHED.json "RANDOM-LATENT-NULL-MATCHED@11250"
echo "CHAIN_B_DONE"
