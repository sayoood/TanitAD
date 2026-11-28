p = "optb_chain.sh"
s = open(p, encoding="utf-8", newline="").read()


def sub(old, new, count=1):
    global s
    assert s.count(old) == count, (s.count(old), old[:90])
    s = s.replace(old, new)


sub("# optb_chain.sh -- the O/T band-split retrain on box 2, in the card's order (theory/PREREG_V8F_OPTB.md):",
    "# optb_chain.sh -- ACTRSP Option B (frozen spec AMEND_OPTB.md): ten world-model fine-tunes of the R35 checkpoints on box 2, chassis 0 first (MM_RULINGS_OPTB.md items 2 and 5):")
sub("-> ship (own tree) -> PILOT (200 steps of RSP_s0, projected total, 72 h rule)",
    "-> ship (own tree) -> inputs (the two R35 checkpoints and two probes, sha256 checked on arrival) -> PILOT (200 steps of FTRSP_s0, projected total, 72 h rule)")
sub("#   -> RSP_s0, RSP_s2, NULL_s4, RSP_s1, RSP_s3, NULL_s5 (card section 3 order).",
    "#   -> FTRSP_s0, FTNULL_s0, FTRSP_s2, FTNULL_s2, FTNULL_s4 (chassis 0), then FTRSP_s1, FTNULL_s1, FTRSP_s3, FTNULL_s3, FTNULL_s5 (chassis 1) in the checker's RUN_ORDER.\n#   A refused or failed chassis-0 arm stops the chain BEFORE chassis 1 starts (STOP.txt reason chassis0-failed). After each run is banked on the dev box it is also banked PRIVATE to HF (stages/optb_hfbank.sh).")
sub("for f in optb_check.py optb_ship_tree.py optb_refresh.py frozen/argvs_optb.json frozen/export_list_optb.txt stages/optb_common.sh stages/optb_ship.sh stages/optb_run.sh jobs/optb_stage_job.cmd jobs/optb_ship_job.cmd; do",
    "for f in optb_check.py optb_ship_tree.py optb_refresh.py optb_hfbank.py frozen/argvs_optb.json frozen/export_list_optb.txt stages/optb_common.sh stages/optb_ship.sh stages/optb_inputs.sh stages/optb_run.sh stages/optb_hfbank.sh jobs/optb_stage_job.cmd jobs/optb_ship_job.cmd; do")
sub('case "$1" in ship) bash "$STAGES_DIR/optb_ship.sh";;', 'case "$1" in ship) bash "$STAGES_DIR/optb_ship.sh";; inputs) bash "$STAGES_DIR/optb_inputs.sh";;')
sub("run_stage ship\nrun_stage pilot\n", "run_stage ship\nrun_stage inputs\nrun_stage pilot\n")
# chassis table from the checker (init of each tag)
sub('''REFUSED=""
for t in $RUN_ORDER; do
  if [ -e "$STATE/done/$t.done" ]; then log "skip $t (done)"; continue; fi
''', '''REFUSED=""; HFPEND=""
chassis_of() { "$PYV" -I -c "import sys; sys.path.insert(0, sys.argv[1]); import optb_check as C; print(C.TABLE[sys.argv[2]]['init'])" "$HERE" "$1"; }
for t in $RUN_ORDER; do
  if [ -e "$STATE/done/$t.done" ]; then log "skip $t (done)"; continue; fi
  ch="$(chassis_of "$t")"; case "$ch" in 0|1) ;; *) stop "$t" "chassis-unreadable" 5;; esac
  if [ "$ch" = 1 ] && [ -n "$REFUSED" ]; then   # MM ruling 2: chassis 0 completes first; a chassis-0 failure stops the chain before chassis 1 spends box-2 time
    echo "ZZ OPTB-STOP chassis0-failed arms-refused:$REFUSED rc=4" > "$STATE/STOP.txt"; log "chassis 0 had refused arms:$REFUSED; chassis 1 is not started"; release_box; exit 4
  fi
''')
sub('''  if [ "$rc" -eq 0 ]; then echo "$t $(date '+%F %T')" > "$STATE/done/$t.done"; log "done $t"
  elif [ "$rc" -eq 11 ]; then''', '''  if [ "$rc" -eq 0 ]; then echo "$t $(date '+%F %T')" > "$STATE/done/$t.done"; log "done $t"
    if [ ! -e "$STATE/done/hf_$t.done" ]; then
      if bash "$STAGES_DIR/optb_hfbank.sh" "$t"; then echo "$t $(date '+%F %T')" > "$STATE/done/hf_$t.done"; log "banked $t to HF (private)"
      else HFPEND="$HFPEND $t"; echo "hf bank of $t failed: the dev-box bank is intact, upload again with stages/optb_hfbank.sh $t" > "$STATE/HF_BANK_PENDING_$t.txt"; log "HF bank of $t failed (pending)"; fi
    fi
  elif [ "$rc" -eq 11 ]; then''')
sub('''if [ -n "$REFUSED" ]; then echo "ZZ OPTB-STOP runs arms-refused:$REFUSED rc=4" > "$STATE/STOP.txt"; log "finished with refused arms:$REFUSED"; release_box; exit 4; fi
''', '''if [ -n "$REFUSED" ]; then echo "ZZ OPTB-STOP runs arms-refused:$REFUSED rc=4" > "$STATE/STOP.txt"; log "finished with refused arms:$REFUSED"; release_box; exit 4; fi
if [ -n "$HFPEND" ]; then echo "ZZ OPTB-STOP hfbank hf-bank-pending:$HFPEND rc=5" > "$STATE/STOP.txt"; log "all runs trained and banked on the dev box; HF bank pending:$HFPEND"; release_box; exit 5; fi
''')
open(p, "w", encoding="utf-8", newline="").write(s)
print("chain patched")
