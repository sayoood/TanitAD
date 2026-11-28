#!/usr/bin/env bash
# PREREG_V8F_GRID on box 2: the stack's scene input at 8x20 tokens per frame instead of 4x10 (G = B + --grid pool8x20, ONE lever). ARMED ONLY AFTER PREREG_V8F_PGT's chain ends.
# One job at a time under SAFE MODE (mk_v8f_job.py). Derived from chain_v8f_pgt.sh; markers ZZV8FMINI-DONE / -STOP in ITS OWN log (chain_v8f_grid.out), which v8f_guard.sh's
# anchored end test already knows.
#   chain_v8f_grid.sh --dry-run   PRINTS the exact command every stage runs, for BOTH bases (the base is only decided at run time), from the tip's queue or $V8F_QUEUE, and exits;
#                                 nothing is shipped, launched or waited for. The line printed IS the line that runs.
#   0. wait for the predecessor (default: PREREG_V8F_PGT's chain, chain_v8f_pgt.out, ZZV8FMINI-DONE; stop on its ZZV8FMINI-STOP)
#   1. refresh box 2's code from the REMOTE tip; sha-check 8 key files, md5-check both v9 releases; extract the reader HERE from the same archive
#   2. THE BASE (fixed in the PREREG before PGT had data): `v8f_grid_read.py --decide-base` reads PGT's own files. B = M2b-pgt iff PGT's FINAL read says VERDICT == SUCCESS,
#      else B = M2b (any FAIL, a seed-0 STOP). A contradiction between the PGT files, or no read at all, STOPS the chain.
#   3. the B side is staged (M2b: the banked runs / evals; M2b-pgt: PGT's own pulled runs / evals) and `v8f_grid_read.py --check-inputs` runs on it BEFORE any training: runs, evals,
#      the PREREG's known values (B = M2b), the tie to the PGT read, the identity control (+0 [0, 0], M-LAT included)
#   4. THE CACHE: box 2's train AND eval manifests are pulled and `check_grid_cache.py` requires EVERY effective row to list 8x20 with shapes.pool8x20 = [n_frames, 160, 1024], the train
#      manifest to hold 4,572 clips and the eval manifest every clip of the eval-141 grid -- while box 2 is still idle
#   5. the in-process launch gate for G (`gategrid BASE`), pulled and asserted HERE: 4,512 clips, the RA-1 holdout withheld, B's clip list and 19,164 steps, B's levers and weights, B's
#      p_gt schedule (the queue's PGTS / PGTE for M2b-pgt; the trainer's 1.0 -> 0.2 for M2b) and G's parameter count = B's + (160 - 40) x 256 EXACTLY, adapter group only
#   6. STAGE 1: G s0 (`traingrid 0 BASE`) -> its four evals (`evalgrid 0 BASE`); v8f_grid_read.py --stage s0 against B's runs. STOP -> DONE with the seed-0 verdict (committed)
#   7. STAGE 2 (on CONTINUE): G s1 -> evalgrid; v8f_grid_read.py --stage final -> grid_read.json
# Stops on a reboot (waitjob 2), unreachable (3), timeout (4), any nonzero rc, or a guard stop. Script polling, not model turns.
# TIMEOUTS: the training wait is 1,320 polls of 30 s = 11 h (was 10 h = the guard cap; raised 2026-10-22 so the guard binds first): G is ESTIMATED at 1.5-2x B's 4.4 h = 6.6-8.8 h, and v8f_guard.sh stops any run it PROJECTS above V8F_MAX_RUN_H (default 10 h), so
# no run the guard lets live can outlast the wait. Set V8F_MAX_RUN_H before arming to move that cap; the PREREG fixes steps, not wall-clock.
PK="TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
if [ "${1:-}" = "--dry-run" ]; then
  QF=${V8F_QUEUE:-}
  if [ -z "$QF" ]; then
    MIRROR=${MIRROR:-/c/Users/Admin/tanitad-push}; BR=${BR:-agent/arch-inf-20260803}
    T=$(git -C $MIRROR ls-remote origin $BR | cut -f1); QF=${TMPDIR:-/tmp}/v8fgrid_queue_dry_$T.cmd
    git -C $MIRROR show "$T:$PK/code/v8f_mini_box2_queue.cmd" > "$QF" || { echo "ZZV8FMINI-STOP no queue at the tip"; exit 1; }
  fi
  # cmd.exe cannot be handed `"<path with an ampersand>" args` through MSYS quoting (it strips the quotes it needs): a one-line wrapper batch file can
  W=${TMPDIR:-/tmp}/v8fgrid_dry_$$.cmd
  for st in "gategrid M2b" "gategrid M2b-pgt" "traingrid 0 M2b" "traingrid 1 M2b" "traingrid 0 M2b-pgt" "traingrid 1 M2b-pgt" "evalgrid 0 M2b" "evalgrid 1 M2b" "evalgrid 0 M2b-pgt" "evalgrid 1 M2b-pgt"; do
    printf '@echo off\r\nset V8F_DRYRUN=1\r\ncall "%s" %s\r\n' "$(cygpath -w "$QF")" "$st" > "$W"
    echo "== $st"; MSYS_NO_PATHCONV=1 cmd /c "$(cygpath -w "$W")" </dev/null | tr -d '\r' || exit 1
  done
  exit 0
fi
B=/c/Users/Admin/box2_ship; L=$B/logs; E=$L/v8f_grid; PGTD=${PGTD:-$L/v8f_pgt}; mkdir -p $E/runs $E/eval
MIRROR=/c/Users/Admin/tanitad-push; BR=agent/arch-inf-20260803
HJ="TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-refav1-hj-mini"
B1="TanitAD Research Lab/Benchmarks & Evals/Research/2026-09-02-b1-eval-lead-block/raw/b1_eval_lead_block.npz"
SSH="ssh -o BatchMode=yes -o ConnectTimeout=20 tanitad-box2"
SCP="scp -q -o BatchMode=yes -o ConnectTimeout=20"
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
R=D:/tanitad/runs/v8f_mini_full
CACHE=D:/tanitad/cache_full
HOLDOUT_SHA=f6a1dee02af0b96f815416bfef9bc11a04554702525135aa75a740942d7750f9
M2B_STEPS=19164
TRAIN_POLLS=1320; EVAL_POLLS=600   # 11 h: one hour above the guard cap, so the GUARD (a committed rule), never the wait, ends a slow run
TAG=$(date '+%m%d%H%M')
CUR=$L/v8fgrid_current.txt
AFTER_OUT=${AFTER_OUT:-$L/chain_v8f_pgt.out}; AFTER_DONE=${AFTER_DONE:-ZZV8FMINI-DONE}; AFTER_STOP=${AFTER_STOP:-ZZV8FMINI-STOP}
boot() { $SSH "powershell -NoProfile -Command \"(Get-CimInstance Win32_OperatingSystem).LastBootUpTime.ToString('yyyy-MM-dd HH:mm:ss')\"" 2>/dev/null | tr -d '\r' | grep -E '^[0-9]{4}-' | head -1; }
launch() { for t in 1 2 3; do o=$($SSH "powershell -NoProfile -ExecutionPolicy Bypass -File D:\tanitad\launch.ps1 -Job D:\tanitad\jobs\\$1.cmd -Log D:\tanitad\logs\\$1.log" 2>/dev/null | tr -d '\r') && { echo "$(date '+%m-%d %H:%M') $o"; return 0; }; sleep 20; done; return 1; }
clean() { tr '\r' '\n' < "$L/$1.log"; }
stop() { echo "ZZV8FMINI-STOP $*"; echo "" > $CUR; exit 1; }
waitrc() {   # waitrc <job> <polls of 30 s>
  bash $B/waitjob.sh $1 $2 "$BOOT0"; local w=$?
  [ $w = 0 ] || stop "$1 wait=$w (2 = CRASH/reboot, 3 = unreachable, 4 = timeout)"
  local rc=$(clean $1 | grep -a 'ZZJOB-END' | tail -1 | sed -n 's/.* rc=\([-0-9]*\).*/\1/p')
  echo "$(date '+%m-%d %H:%M') $1 rc=$rc"
  [ "$rc" = 0 ] || { clean $1 | grep -a -i -E 'FAIL|refus|error|Traceback' | tail -8; grep ZZGUARD-STOP $L/v8fgrid_guard.out 2>/dev/null; stop "$1 rc=$rc"; }
}
queue() {    # queue <job> <polls> <queue args...>: every job name carries this chain's TAG, so an old log's ZZJOB-END can never satisfy a new wait
  local j="v8fgrid${TAG}_$1" m=$2; shift 2
  $PY "$(cygpath -m $B)/mk_v8f_job.py" $j "$@" >/dev/null && $SCP $B/$j.cmd tanitad-box2:D:/tanitad/jobs/ || stop "ship $j"
  rm -f $L/$j.log
  launch $j || stop "launch $j"
  waitrc $j $m
}
rsha() { $SSH "powershell -NoProfile -Command \"(Get-FileHash -Algorithm $2 -LiteralPath '$1').Hash.ToLower()\"" 2>/dev/null | tr -d '\r' | grep -E '^[0-9a-f]{32,64}$'; }
pull() {     # pull <box-2 path under R> <local dir>
  mkdir -p "$2"; $SCP "tanitad-box2:$R/$1" "$2/" || stop "pull $R/$1"
}
finish() {   # finish <summary>: the committed end of the chain (also after a stage-1 STOP)
  $SSH "powershell -NoProfile -Command \"New-Item -ItemType File -Force D:/tanitad/logs/telemetry.stop | Out-Null\"" 2>/dev/null
  $SCP tanitad-box2:D:/tanitad/logs/telemetry_full.csv $E/telemetry_v8fgrid.csv 2>/dev/null
  echo "" > $CUR
  rm -f $B/BOX2_BUSY.txt
  echo "ZZV8FMINI-DONE $(date '+%m-%d %H:%M') tip $T: $1 (boot unchanged $BOOT0)"
  exit 0
}
runone() {   # runone <seed>: train G, eval (+ absmask + GT oracle + all-latcons oracle), pull
  local s=$1 g="${BASE}_s${1}_8x20" tg="${BASE}-8x20_s${1}"
  echo "$g" > $CUR
  queue "t_$g" $TRAIN_POLLS traingrid $s $BASE
  echo "" > $CUR
  queue "e_$g" $EVAL_POLLS evalgrid $s $BASE
  for f in selfcheck.json summary.json argv.json train_log.jsonl train.log ckpt_last.pt; do pull "$g/$f" "$E/runs/$g"; done
  for f in "" -tacabs -tacorc -tacorc-all-latcons; do for x in json npz log; do pull "eval/eval_$tg$f.$x" "$E/eval"; done; done
  grep ZZGUARD-STOP $L/v8fgrid_guard.out 2>/dev/null && stop "guard stop"
}
gridread() {   # gridread <stage> <out>
  PYTHONIOENCODING=utf-8 $PY "$RD/v8f_grid_read.py" --stage $1 --base $BASE --pgt-dir "$(cygpath -m $PGTD)" --eval-dir "$(cygpath -m $E/eval)" --runs-dir "$(cygpath -m $E/runs)" \
    --out "$(cygpath -m $E/$2)" || stop "grid read $1"
}
jget() { $PY -c "import json,sys;print(json.load(open(sys.argv[1],encoding='utf-8'))[sys.argv[2]])" "$(cygpath -m $E/$1)" $2; }
echo "$(date '+%m-%d %H:%M') waiting for the predecessor chain ($AFTER_OUT)"
until grep -q "^$AFTER_DONE" "$AFTER_OUT" 2>/dev/null; do
  grep -q "^$AFTER_STOP" "$AFTER_OUT" 2>/dev/null && { echo "ZZV8FMINI-STOP the predecessor stopped: $(grep "^$AFTER_STOP" "$AFTER_OUT" | tail -1)"; exit 1; }
  sleep 300
done
[ -f $B/BOX2_BUSY.txt ] && { echo "ZZV8FMINI-STOP busy marker present: $(cat $B/BOX2_BUSY.txt)"; exit 1; }
echo "v8F-mini PREREG_V8F_GRID (G = B + --grid pool8x20, B decided by PREREG_V8F_PGT's reads; seed 0, then seed 1 on CONTINUE) since $(date '+%m-%d %H:%M'), ~7-9 h per seed. Do NOT launch box 2 jobs while this file exists." > $B/BOX2_BUSY.txt
BOOT0=$(boot); [ -n "$BOOT0" ] || stop "box 2 unreachable"
echo "$(date '+%m-%d %H:%M') boot0=$BOOT0"
# ---- 1. code refresh from the remote tip
git -C $MIRROR fetch -q origin $BR; T=$(git -C $MIRROR ls-remote origin $BR | cut -f1)
git -C $MIRROR cat-file -e "$T^{commit}" || stop "remote tip $T is not in the mirror"
QT=$B/v8fgrid_queue_$TAG.cmd; git -C $MIRROR show "$T:$PK/code/v8f_mini_box2_queue.cmd" | tr -d '\r' > $QT
for st in gategrid traingrid evalgrid; do grep -q "^:$st\$" $QT || stop "the tip's queue has no $st stage"; done
PGTS=$(sed -n 's/^set "PGTS=\(.*\)"$/\1/p' $QT); PGTE=$(sed -n 's/^set "PGTE=\(.*\)"$/\1/p' $QT)
[ -n "$PGTS" ] && [ -n "$PGTE" ] || stop "the tip's queue defines no PGTS / PGTE (the M2b-pgt dose lives once, in the queue)"
[ "$(grep -c '^set "PGTS=' $QT)" = 1 ] && [ "$(grep -c '^set "PGTE=' $QT)" = 1 ] || stop "the tip's queue defines the dose more than once"
for f in v8f_grid_read.py v8f_pgt_read.py v8f_dk_read.py v8f_wc_read.py v8f_tac_groups_read.py v8f_mini_full_report.py check_grid_cache.py; do
  git -C $MIRROR cat-file -e "$T:$PK/code/$f" || stop "the tip has no $f"
done
git -C $MIRROR show "$T:stack/tanitad/models/v8f_mini.py" | grep -q '"p_gt_start": cfg.p_gt_start' || stop "the tip's built_levers does not record p_gt_start"
git -C $MIRROR show "$T:stack/tanitad/models/v8f_mini.py" | grep -q '"p_gt_end": cfg.p_gt_end' || stop "the tip's built_levers does not record p_gt_end"
git -C $MIRROR cat-file -e "$T:$B1" || stop "the tip has no B1 lead block"
git -C $MIRROR -c core.autocrlf=false archive --format=tar -o $B/repo_v8fgrid_tip.tar $T -- stack taniteval "$HJ" "$PK" "$B1" || stop "archive"
$SCP $B/repo_v8fgrid_tip.tar tanitad-box2:D:/tanitad/repo_v8fgrid_tip.tar || stop "ship the code archive"
printf '@echo off\r\necho ===== refresh %s %%DATE%% %%TIME%%\r\ntar -xf D:\\tanitad\\repo_v8fgrid_tip.tar -C D:\\tanitad\\repo\r\nset RC=%%ERRORLEVEL%%\r\necho %s> D:\\tanitad\\repo\\TIP_V8FGRID.txt\r\necho ===== ZZJOB-END %%DATE%% %%TIME%% rc=%%RC%%\r\n' "$T" "$T" > $B/v8fgrid${TAG}_refresh.cmd
$SCP $B/v8fgrid${TAG}_refresh.cmd tanitad-box2:D:/tanitad/jobs/ || stop "ship refresh job"
launch v8fgrid${TAG}_refresh || stop "launch refresh"
waitrc v8fgrid${TAG}_refresh 40
for f in stack/scripts/v8f_mini_train.py stack/scripts/v8f_mini_eval.py stack/tanitad/models/v8f_mini.py stack/tanitad/data/v8f_mini_data.py \
         stack/tanitad/models/hjepa.py "$PK/code/v8f_mini_box2_queue.cmd" "$HJ/raw/p10_yaw4512/p10_prior.pt" "$HJ/raw/p10_yaw4512/p10_eval_pred6.npy"; do
  l=$(git -C $MIRROR show "$T:$f" | sha256sum | cut -c1-64)
  [ "$l" = "$(rsha "D:/tanitad/repo/$f" SHA256)" ] || stop "box 2 $f != tip $T"
done
for kv in v9_labels_train.npz=83e30bd3791f0767707eade5f4671759 v9_labels_eval139.npz=6b5c7f207cffc3b7eb3cd527fd433599; do
  [ "${kv#*=}" = "$(rsha "D:/tanitad/data/v9/${kv%%=*}" MD5)" ] || stop "box 2 v9 ${kv%%=*} md5 != ${kv#*=}"
done
echo "$(date '+%m-%d %H:%M') box 2 code = tip $T (8 key files sha-checked; both v9 releases md5-checked)"
# the reader runs HERE at the same tip, from the archive box 2 got (a fresh directory, never rm -rf)
X=$B/v8fgrid_tip_$(date +%Y%m%d%H%M%S); mkdir -p $X && tar -xf $B/repo_v8fgrid_tip.tar -C $X || stop "extract the reader"
RD="$(cygpath -m "$X/$PK/code")"
# ---- 2. the base B, decided by PREREG_V8F_PGT's own files (the rule is in the reader, where it is tested)
PYTHONIOENCODING=utf-8 $PY "$RD/v8f_grid_read.py" --decide-base --pgt-dir "$(cygpath -m $PGTD)" --out "$(cygpath -m $E/base.json)" || stop "the base B cannot be decided from the PGT reads in $PGTD"
BASE=$(jget base.json base)
case "$BASE" in M2b|M2b-pgt) ;; *) stop "the decided base '$BASE' is neither M2b nor M2b-pgt";; esac
echo "$(date '+%m-%d %H:%M') base B = $BASE: $(jget base.json why)"
# ---- 3. the B side, staged into this chain's dirs BEFORE any training (a missing file stops the chain while box 2 is still idle)
for s in 0 1; do
  mkdir -p $E/runs/${BASE}_s$s
  if [ "$BASE" = M2b-pgt ]; then
    for f in summary.json argv.json selfcheck.json train_log.jsonl; do cp "$PGTD/runs/M2b-pgt_s$s/$f" $E/runs/M2b-pgt_s$s/ || stop "stage PGT's M2b-pgt_s$s/$f"; done
    for suf in "" -tacabs -tacorc -tacorc-all-latcons; do for x in json npz; do cp "$PGTD/eval/eval_M2b-pgt_s$s$suf.$x" $E/eval/ || stop "stage PGT's eval_M2b-pgt_s$s$suf.$x"; done; done
  else
    for f in summary.json argv.json selfcheck.json; do cp "$L/v8f_mini/runs/M2b_s$s/$f" $E/runs/M2b_s$s/ || stop "stage the banked M2b_s$s/$f"; done
    cp "$L/v8f_mini/runs/M2b_s$s/train_log.jsonl" $E/runs/M2b_s$s/ 2>/dev/null || echo "$(date '+%m-%d %H:%M') NOTE: no banked M2b_s$s/train_log.jsonl: its p_gt schedule is NOT-CHECKED (by construction 1.0 -> 0.2)"
    cp $L/v8f_mini/eval/eval_M2b_s$s.npz $L/v8f_mini/eval/eval_M2b_s$s.json $E/eval/ || stop "stage eval_M2b_s$s"
    for f in eval_M2b_s$s-tacabs.npz eval_M2b_s$s-tacabs.json eval_M2b_s$s-tacorc.npz; do cp "$L/v8f_cz/eval/$f" $E/eval/ || stop "stage $f"; done
    # M2b's tacorc predates tac_code (WC Amendment 1): its GT constraints come from the all-goal oracle; M-LEAK needs the all-latcons oracle (same dir)
    for f in eval_M2b_s$s-tacorc-all-goal.npz eval_M2b_s$s-tacorc-all-latcons.npz; do cp "$L/v8f_tacgrp2/eval/$f" $E/eval/ || stop "stage $f"; done
    cp "$L/v8f_tacgrp2/eval/eval_M2b_s$s-tacorc-all-latcons.json" $E/eval/ 2>/dev/null
  fi
done
cp "$L/v8f_mini/eval/floors_perwindow.npz" $E/eval/ || stop "stage floors_perwindow.npz (the P10 reproduction guard)"
# the reader must load the REAL B side BEFORE any training: runs, evals, the PREREG's known values, the PGT tie, the identity control
PYTHONIOENCODING=utf-8 $PY "$RD/v8f_grid_read.py" --check-inputs --base $BASE --pgt-dir "$(cygpath -m $PGTD)" --eval-dir "$(cygpath -m $E/eval)" --runs-dir "$(cygpath -m $E/runs)" \
  --out "$(cygpath -m $E/check_inputs.json)" || stop "the reader cannot use the $BASE side (check-inputs)"
# ---- 4. the cache: EVERY manifest row of box 2's train and eval caches lists 8x20 (box 2 is idle; a clip without it would crash a trainer hours in)
for sp in train eval; do $SCP "tanitad-box2:$CACHE/$sp/manifest.jsonl" "$E/manifest_$sp.jsonl" || stop "pull box 2's $sp manifest"; done
PYTHONIOENCODING=utf-8 $PY "$RD/check_grid_cache.py" --train "$(cygpath -m $E/manifest_train.jsonl)" --eval "$(cygpath -m $E/manifest_eval.jsonl)" \
  --eval-grid "$RD/../raw/eval_grid_windows.json" --n-train 4572 --out "$(cygpath -m $E/cache_check.json)" || stop "box 2's cache does not carry 8x20 for every clip (check_grid_cache)"
$SSH "powershell -NoProfile -Command \"if (Test-Path D:/tanitad/logs/telemetry.stop) { Remove-Item D:/tanitad/logs/telemetry.stop }\"" 2>/dev/null
launch telemetry_long || echo "$(date '+%m-%d %H:%M') telemetry did not launch"
tl() { $SSH "powershell -NoProfile -Command \"Get-Content D:/tanitad/logs/telemetry_full.csv -Tail 1\"" 2>/dev/null | tr -d '\r'; }
sleep 45; t1=$(tl); sleep 35; t2=$(tl)
[[ $t2 == 20* && $t2 != "$t1" ]] || stop "no fresh telemetry ($t1 / $t2): the temperature guard would be blind"
grep -qF "'^(ZZV8FMINI|" $B/v8f_guard.sh || stop "v8f_guard.sh does not know this chain's markers (anchored): it would never exit"
echo "" > $CUR
V8F_CHAIN_OUT=$L/chain_v8f_grid.out V8F_CURRENT=$CUR bash $B/v8f_guard.sh > $L/v8fgrid_guard.out 2>&1 &
# ---- 5. the launch gate for G, asserted here as well as in-process
queue gategrid 120 gategrid $BASE
pull "gate_${BASE}-8x20/selfcheck.json" "$E/gate_${BASE}-8x20"
$PY - "$(cygpath -m $E/gate_${BASE}-8x20/selfcheck.json)" $HOLDOUT_SHA $M2B_STEPS "$(cygpath -m $E/runs/${BASE}_s0/summary.json)" $PGTS $PGTE $BASE <<'EOF' || stop "gate G assertions"
import json, sys
sc, hold, steps, b, p0, p1, base = json.load(open(sys.argv[1])), sys.argv[2], int(sys.argv[3]), json.load(open(sys.argv[4])), float(sys.argv[5]), float(sys.argv[6]), sys.argv[7]
i, lv, bl = sc["info"], sc["levers"], b["levers"]
want = (p0, p1) if base == "M2b-pgt" else (1.0, 0.2)
bad = []
if sc.get("declared_vs_built") != "PASS": bad.append("declared_vs_built")
if not sc["live"].get("pass") or sc["live"].get("control_prior_grad_sum") != 0.0: bad.append(f"live {sc['live'].get('pass')}")
if (i["n_clips"], i["n_excluded"], i["n_excluded_declared"], i["excluded_in_train"]) != (4512, 60, 60, 0): bad.append("clips / holdout counts")
if i["excluded_sha256"] != hold: bad.append("the excluded set is not the RA-1 holdout")
if i["train_sha256"] != b["train_sha256"]: bad.append(f"TRAIN list {i['train_sha256'][:12]} != B's {b['train_sha256'][:12]}")
if i["kin_mode"] != "yaw": bad.append("kin_mode")
if (lv["wm"], lv["navonly"], lv["consistency"], lv.get("cons_absent")) != ("b", False, True, "mask"): bad.append(f"levers {lv}")
# B's schedule is G's: the queue's dose for M2b-pgt, the trainer's default for M2b -- G moves the GRID and nothing else
if (lv.get("p_gt_start"), lv.get("p_gt_end")) != want: bad.append(f"built p_gt {lv.get('p_gt_start')} -> {lv.get('p_gt_end')}, expected {want[0]} -> {want[1]}")
w = lv.get("w") or {}
for k in ("w_pen", "w_pen_self", "w_wm"):
    if w.get(k) != bl["w"][k]: bad.append(f"{k} {w.get(k)} != B's {bl['w'][k]}")
if (w.get("w_tac"), w.get("w_cons")) != (1.0, 0.5): bad.append(f"w_tac / w_cons {w.get('w_tac')} / {w.get('w_cons')} != B's 1.0 / 0.5 (a second lever moved)")
if sc["steps_total"] != steps: bad.append(f"steps {sc['steps_total']} != B's {steps}")
# THE LEVER, from the BUILT model: the position table is the only per-token parameter, so G has (160 - 40) x 256 = 30,720 more parameters, all in the adapter group, and nothing else moved
D = (160 - 40) * 256
pg, pb = lv["params"], bl["params"]
if lv["n_params_trainable"] != bl["n_params_trainable"] + D: bad.append(f"n_params_trainable {lv['n_params_trainable']} != B's {bl['n_params_trainable']} + {D}")
for k in sorted(set(pg) | set(pb)):
    if pg.get(k, 0) - pb.get(k, 0) != (D if k == "adapter" else 0): bad.append(f"parameter group {k}: {pg.get(k)} vs B's {pb.get(k)}")
if not i.get("lru"): bad.append("info.lru is not recorded in the selfcheck (the pipeline sizing at 160 tokens per frame must be on the record)")
print("G gate:", "PASS" if not bad else bad, "| clips", i["n_clips"], "steps", sc["steps_total"], "| built p_gt", lv.get("p_gt_start"), "->", lv.get("p_gt_end"),
      "| params", lv["n_params_trainable"], "(B", bl["n_params_trainable"], "+", D, ") | lru", i.get("lru"))
sys.exit(1 if bad else 0)
EOF
# ---- 6. STAGE 1: seed 0, read against B
runone 0
gridread s0 grid_read_s0.json
G1=$(jget grid_read_s0.json reading); echo "$(date '+%m-%d %H:%M') $G1"
[ "$(jget grid_read_s0.json DECISION)" = CONTINUE ] || finish "$G1 -- seed 1 not trained (committed)"
# ---- 7. STAGE 2: seed 1, the final reading
runone 1
gridread final grid_read.json
finish "$G1 | $(jget grid_read.json reading)"
