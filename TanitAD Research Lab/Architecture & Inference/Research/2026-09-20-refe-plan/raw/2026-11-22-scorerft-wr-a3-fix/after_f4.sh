#!/bin/bash
# MM 2026-11-22: after_f4.sh = v3's helpers (after_f3.sh SOURCED with AF3_NO_MAIN=1, never edited; sha256 checked below) + sft_stage1_v4 + main_v4 (mode sft_fs).
# Why v4: v3 (adopt_cs) posted Cs 21:38:30Z, then the A2 weight rule crashed at its SECOND autograd.grad on one --compile graph (in-place version 18 vs 0; donated buffers were
# already off). v3 fell through to Fs; the MM stopped v3 (pid 210085) and the Fs launch gate (222757, 222682) BEFORE Fs launched. Package A3 (weight rule: one forward + one
# backward per gradient, no retained graph) replaces A2 via $SF/PKG_A.txt + expect_A.sha256.
# sft_stage1_v4 = v3's sft_stage1 with TWO caller defects fixed (MEASURED 2026-11-22 against the tool's REAL JSON, wr_probe.json):
#   (1) v3 read d['status'] == 'OK' and d['rank_w']; the tool writes neither at top level: the decision is d['rule'] = {status: SELECT | K2 | K2b | ERROR, rank_w}. So v3 would have
#       logged "JSON unreadable" and skipped SCORER-FT on a SUCCESSFUL rule. v4 reads d['rule'], requires SELECT, and refuses a synthetic-label (NOT_A_DECISION) result.
#   (2) v3's WR_BANKED check counted chain.log AFTER the rule and grepped only later lines, so it could never match (no egress ran in between). v4 logs the JSON's sha256; the
#       MM banks it (pull + land in the repo).
# Order: weight rule (A3) -> SFTC -> package B wait (6 h) -> SFTT -> Fs -> handoff (after_af.sh = C1r).   usage: after_f4.sh <field PKG = code_10848403 dir>
set -u
F3=/workspace/data/refe_p5c/after_f3.sh
F3_SHA=e5e9751bf055d697
[ "$(sha256sum $F3 | cut -c1-16)" = "$F3_SHA" ] || { echo "after_f3.sh is not the v3 the MM read ($F3_SHA)"; exit 2; }
AF3_NO_MAIN=1 source "$F3" "$1"

sft_stage1_v4() {
  local PA PB W wr
  PA="$(cat $SF/PKG_A.txt 2>/dev/null)"
  [ -n "$PA" ] && pkg_ok "$PA" "$SF/expect_A.sha256" || { log "ZZAF_SFT_SKIP package A missing or not the shipped bytes ($PA)"; return 1; }
  grep -q "^GRAD_PASSES = 3 " "$PA/refe/weight_rule_rank.py" || { log "ZZAF_SFT_SKIP package A's weight rule is not A3 (no GRAD_PASSES = 3)"; return 1; }
  for a in SFTC WR; do [ -s "$SF/ARGS_$a.txt" ] || { log "ZZAF_SFT_SKIP no $SF/ARGS_$a.txt"; return 1; }; done
  cp "$SF/ARGS_SFTC.txt" "$ROOT/ARGS_SFTC.txt"
  for o in scorerft_c scorerft_t; do [ -e "$RUNS/$o" ] && { log "ZZAF_SFT_SKIP $RUNS/$o already exists"; return 1; }; done
  [ -e "$SF/weight_rule_rank.log" ] && mv -n "$SF/weight_rule_rank.log" "$SF/weight_rule_rank.CRASH_20261122T2139Z.log"
  [ -e "$SF/weight_rule_rank.log" ] && { log "ZZAF_SFT_SKIP the A2 crash log could not be moved aside"; return 1; }
  # ---- 2. the weight rule (GPU), on T's argv, from package A3
  log "WR_START v4 package $PA"
  (cd "$PA/refe" && env -u HF_TOKEN PYTHONUNBUFFERED=1 PYTHONPATH=$PA/refe REFE_BACKBONE_ROOT=$D/backbones PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4 \
     NUPLAN_MAPS_ROOT=$D/nuplan-maps/nuplan-maps-v1.0 "$PY" weight_rule_rank.py --args "$(cat $SF/ARGS_WR.txt)" --refe "$PA/refe" --out "$SF/weight_rule_rank.json" > "$SF/weight_rule_rank.log" 2>&1)
  wr=$?
  log "WR_RC $wr $(tail -n 1 $SF/weight_rule_rank.log | cut -c1-200)"
  [ -s "$SF/weight_rule_rank.json" ] && log "WR_JSON sha256 $(sha256sum $SF/weight_rule_rank.json | cut -c1-16) (the MM banks it)"
  [ "$wr" = 0 ] || { log "ZZAF_SFT_SKIP weight rule exit $wr (5 = K2/K2b: T is not launched, prereg 3; 4 = could not measure): the card goes back to the MM"; return 1; }
  W="$("$PY" -c "import json,sys; d=json.load(open(sys.argv[1])); r=d['rule']; assert r['status']=='SELECT', r['status']; assert d['NOT_A_DECISION'] is False, 'synthetic labels'; print(repr(float(r['rank_w'])))" "$SF/weight_rule_rank.json" 2>>"$S")" \
    || { log "ZZAF_SFT_SKIP weight rule JSON unreadable, rule not SELECT, or synthetic"; return 1; }
  log "WR_W $W"
  # ---- 3. C (package A3)
  run_arm SFTC refe_scorerft_c "$PA" "$RUNS/scorerft_c" || { log "ZZAF_SFT_SKIP C did not launch"; return 1; }
  wait_arm SFTC "$RUNS/scorerft_c" 1000 || { log "ZZAF_SFT_SKIP C did not finish"; return 1; }
  post_sft SFTC "$PA" "$RUNS/scorerft_c" scorerft/c || { log "ZZAF_SFT_SKIP C post failed"; return 1; }
  # ---- 4. T (package B: A3 + the registered weight); wait up to 6 h for the MM's land + ship
  local i=0
  while :; do
    PB="$(cat $SF/PKG_B.txt 2>/dev/null)"
    if [ -n "$PB" ] && pkg_ok "$PB" "$SF/expect_B.sha256" && [ -s "$SF/ARGS_SFTT.txt" ]; then break; fi
    i=$((i+1)); [ $i -gt 72 ] && { log "ZZAF_SFT_SKIP package B not shipped 6 h after C posted: T not launched (C is banked; T runs by hand later)"; return 1; }
    [ $i = 1 ] && log "WAIT package B (SCORERFT_RANK_W = $W) + ARGS_SFTT.txt"
    sleep 300
  done
  for f in refe/train.py refe/rank_loss.py refe/online_labeller.py; do
    cmp -s "$PA/$f" "$PB/$f" || { log "ZZAF_SFT_SKIP package B's $f differs from package A's: T and C would not share a trainer"; return 1; }
  done
  "$PY" -c "import sys; sys.path.insert(0, sys.argv[1]); import launch_gate as L; w=float(sys.argv[2]); assert L.SCORERFT_RANK_W is not None and float(L.SCORERFT_RANK_W)==w, (L.SCORERFT_RANK_W, w)" "$PB/refe" "$W" 2>>"$S" \
    || { log "ZZAF_SFT_SKIP package B's SCORERFT_RANK_W != the rule's $W"; return 1; }
  grep -q -- "--rank-w $W " "$SF/ARGS_SFTT.txt" || { log "ZZAF_SFT_SKIP ARGS_SFTT.txt does not carry --rank-w $W"; return 1; }
  cp "$SF/ARGS_SFTT.txt" "$ROOT/ARGS_SFTT.txt"
  run_arm SFTT refe_scorerft_t "$PB" "$RUNS/scorerft_t" || { log "ZZAF_SFT_SKIP T did not launch"; return 1; }
  wait_arm SFTT "$RUNS/scorerft_t" 1000 || { log "ZZAF_SFT_SKIP T did not finish"; return 1; }
  post_sft SFTT "$PB" "$RUNS/scorerft_t" scorerft/t "$RUNS/scorerft_c" || { log "ZZAF_SFT_SKIP T post failed"; return 1; }
  log "ZZAF_SFT_STAGE1_DONE (the read is the MM's, on the dev box; Stage 2 is conditional on it)"
  return 0
}

main_v4() {
  [ -d "$PKG/refe" ] || { echo "no package $PKG"; exit 2; }
  log "START pid $$ v4 mode sft_fs field PKG $PKG (after_f3.sh $F3_SHA sourced; FIELD_REP_DECISION: $(head -n 1 $ROOT/FIELD_REP_DECISION.txt 2>/dev/null | cut -c1-120))"
  if gpu_free; then sft_stage1_v4 || true; else log "ZZAF_SFT_SKIP the GPU was not free for the weight rule"; fi
  run_arm Fs refe_field_p7_s1 "$PKG" "$RUNS/p7field_fs" || fail run_Fs "see RUN_FAIL above"
  wait_arm Fs "$RUNS/p7field_fs" 2000 || fail wait_Fs "see WAIT_FAIL above"
  post_field Fs "$PKG" "$RUNS/p7field_fs" p7field/fs "$RUNS/p7field_f"
  log "ZZAF_DONE"
  handoff
}
[ "${AF4_NO_MAIN:-0}" = 1 ] || main_v4
