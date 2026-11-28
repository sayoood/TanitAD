#!/usr/bin/env bash
# PREREG_V8F_HIST on box 2: the stack's scene readers over 10 history frames instead of 4 (H = M2b + --hist 10 --window-start 3, ONE lever set by two flags: Amendment 1). ARMED ONLY AFTER PREREG_V8F_GRID's chain ends, and only on the PI's go.
# One job at a time under SAFE MODE (mk_v8f_job.py). Derived from chain_v8f_grid.sh; markers ZZV8FMINI-DONE / -STOP in ITS OWN log (chain_v8f_hist.out), which v8f_guard.sh's
# anchored end test already knows.
#   chain_v8f_hist.sh --dry-run   PRINTS the exact command every stage runs, from the tip's queue or $V8F_QUEUE, and exits; nothing is shipped, launched or waited for.
#                                 The line printed IS the line that runs.
#   0. wait for the predecessor (default: PREREG_V8F_GRID's chain, chain_v8f_grid.out, ZZV8FMINI-DONE; stop on its ZZV8FMINI-STOP)
#   1. refresh box 2's code from the REMOTE tip; sha-check 8 key files, md5-check both v9 releases; extract the reader HERE from the same archive
#   2. THE BASE is fixed in the PREREG: B = M2b (PREREG_V8F_PGT and PREREG_V8F_GRID both FAILED, so neither is a base). There is no --decide-base.
#   3. the B side is staged from the banked M2b files (exactly as chain_v8f_grid.sh's M2b branch) and `v8f_hist_read.py --check-inputs` runs on it BEFORE any training: runs, evals,
#      the PREREG's known values, the committed window counts (2,399 / 2,117 / 141 / 141), the identity control (+0 [0, 0], M-LAT / the t >= 11 subset / the per-slot sets included)
#   4. the in-process launch gate for H (`gatehist`), pulled and asserted HERE: 4,512 clips, the RA-1 holdout withheld, B's clip list and WINDOW SET (306,566 windows, B's own n_windows) and
#      19,164 steps, the BUILT hist 10 / hmax 16 / window_start 3 read from selfcheck.json's levers (and the window list's first t and the dataset's clamp origin from its info), B's levers and
#      weights (w_tac / w_cons 1.0 / 0.5), B's p_gt schedule (1.0 -> 0.2), the queue's own printed line (`--hist 10 --window-start 3` once each, no `--hmax`: hmax is the arm's auto value,
#      B's), and H's parameter count = B's + (10 - 4) x 256 EXACTLY, adapter group only (which is what a built hist of 10 weighs)
#   5. STAGE 1: H s0 (`trainhist 0`) -> its four evals (`evalhist 0`); v8f_hist_read.py --stage s0 against B's runs. STOP -> DONE with the seed-0 verdict (committed)
#   6. STAGE 2 (on CONTINUE): H s1 -> evalhist; v8f_hist_read.py --stage final -> hist_read.json
# Stops on a reboot (waitjob 2), unreachable (3), timeout (4), any nonzero rc, or a guard stop. Script polling, not model turns.
# TIMEOUTS: the training wait is 1,320 polls of 30 s = 11 h, one hour above the guard's 10 h cap (V8F_MAX_RUN_H): H is ESTIMATED at 4.5-5 h per seed (400 scene tokens per reader against 160),
# and v8f_guard.sh stops any run it PROJECTS above the cap, so no run the guard lets live can outlast the wait. The PREREG fixes steps, not wall-clock.
PK="TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
dry_stage() {   # dry_stage <queue file> <stage args...>: the line(s) the queue prints for a stage under V8F_DRYRUN. cmd.exe cannot be handed `"<path with an ampersand>" args`
                # through MSYS quoting (it strips the quotes it needs): a one-line wrapper batch file can
  local qf=$1; shift
  local w=${TMPDIR:-/tmp}/v8fhist_dry_$$.cmd
  printf '@echo off\r\nset V8F_DRYRUN=1\r\ncall "%s" %s\r\n' "$(cygpath -w "$qf")" "$*" > "$w"
  MSYS_NO_PATHCONV=1 cmd /c "$(cygpath -w "$w")" </dev/null | tr -d '\r'
}
if [ "${1:-}" = "--dry-run" ]; then
  QF=${V8F_QUEUE:-}
  if [ -z "$QF" ]; then
    MIRROR=${MIRROR:-/c/Users/Admin/tanitad-push}; BR=${BR:-agent/arch-inf-20260803}
    T=$(git -C $MIRROR ls-remote origin $BR | cut -f1); QF=${TMPDIR:-/tmp}/v8fhist_queue_dry_$T.cmd
    git -C $MIRROR show "$T:$PK/code/v8f_mini_box2_queue.cmd" > "$QF" || { echo "ZZV8FMINI-STOP no queue at the tip"; exit 1; }
  fi
  for st in "gatehist" "trainhist 0" "trainhist 1" "evalhist 0" "evalhist 1"; do
    echo "== $st"; dry_stage "$QF" $st || exit 1
  done
  exit 0
fi
B=/c/Users/Admin/box2_ship; L=$B/logs; E=$L/v8f_hist; mkdir -p $E/runs $E/eval
MIRROR=/c/Users/Admin/tanitad-push; BR=agent/arch-inf-20260803
HJ="TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-refav1-hj-mini"
B1="TanitAD Research Lab/Benchmarks & Evals/Research/2026-09-02-b1-eval-lead-block/raw/b1_eval_lead_block.npz"
SSH="ssh -o BatchMode=yes -o ConnectTimeout=20 tanitad-box2"
SCP="scp -q -o BatchMode=yes -o ConnectTimeout=20"
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
R=D:/tanitad/runs/v8f_mini_full
HOLDOUT_SHA=f6a1dee02af0b96f815416bfef9bc11a04554702525135aa75a740942d7750f9
M2B_STEPS=19164
M2B_WINDOWS=306566   # B's n_windows_all_clips (banked summary.json); H must train on B's window LIST, so the same number (Amendment 1)
TRAIN_POLLS=1320; EVAL_POLLS=600   # 11 h: one hour above the guard cap, so the GUARD (a committed rule), never the wait, ends a slow run
TAG=$(date '+%m%d%H%M')
CUR=$L/v8fhist_current.txt
AFTER_OUT=${AFTER_OUT:-$L/chain_v8f_grid.out}; AFTER_DONE=${AFTER_DONE:-ZZV8FMINI-DONE}; AFTER_STOP=${AFTER_STOP:-ZZV8FMINI-STOP}
boot() { $SSH "powershell -NoProfile -Command \"(Get-CimInstance Win32_OperatingSystem).LastBootUpTime.ToString('yyyy-MM-dd HH:mm:ss')\"" 2>/dev/null | tr -d '\r' | grep -E '^[0-9]{4}-' | head -1; }
launch() { for t in 1 2 3; do o=$($SSH "powershell -NoProfile -ExecutionPolicy Bypass -File D:\tanitad\launch.ps1 -Job D:\tanitad\jobs\\$1.cmd -Log D:\tanitad\logs\\$1.log" 2>/dev/null | tr -d '\r') && { echo "$(date '+%m-%d %H:%M') $o"; return 0; }; sleep 20; done; return 1; }
clean() { tr '\r' '\n' < "$L/$1.log"; }
stop() { echo "ZZV8FMINI-STOP $*"; echo "" > $CUR; exit 1; }
waitrc() {   # waitrc <job> <polls of 30 s>
  bash $B/waitjob.sh $1 $2 "$BOOT0"; local w=$?
  [ $w = 0 ] || stop "$1 wait=$w (2 = CRASH/reboot, 3 = unreachable, 4 = timeout)"
  local rc=$(clean $1 | grep -a 'ZZJOB-END' | tail -1 | sed -n 's/.* rc=\([-0-9]*\).*/\1/p')
  echo "$(date '+%m-%d %H:%M') $1 rc=$rc"
  [ "$rc" = 0 ] || { clean $1 | grep -a -i -E 'FAIL|refus|error|Traceback' | tail -8; grep ZZGUARD-STOP $L/v8fhist_guard.out 2>/dev/null; stop "$1 rc=$rc"; }
}
queue() {    # queue <job> <polls> <queue args...>: every job name carries this chain's TAG, so an old log's ZZJOB-END can never satisfy a new wait
  local j="v8fhist${TAG}_$1" m=$2; shift 2
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
  $SCP tanitad-box2:D:/tanitad/logs/telemetry_full.csv $E/telemetry_v8fhist.csv 2>/dev/null
  echo "" > $CUR
  rm -f $B/BOX2_BUSY.txt
  echo "ZZV8FMINI-DONE $(date '+%m-%d %H:%M') tip $T: $1 (boot unchanged $BOOT0)"
  exit 0
}
runone() {   # runone <seed>: train H, eval (+ absmask + GT oracle + all-latcons oracle), pull
  local s=$1 g="M2b_s${1}_h10" tg="M2b-h10_s${1}"
  echo "$g" > $CUR
  queue "t_$g" $TRAIN_POLLS trainhist $s
  echo "" > $CUR
  queue "e_$g" $EVAL_POLLS evalhist $s
  for f in selfcheck.json summary.json argv.json train_log.jsonl train.log ckpt_last.pt; do pull "$g/$f" "$E/runs/$g"; done
  for f in "" -tacabs -tacorc -tacorc-all-latcons; do for x in json npz log; do pull "eval/eval_$tg$f.$x" "$E/eval"; done; done
  grep ZZGUARD-STOP $L/v8fhist_guard.out 2>/dev/null && stop "guard stop"
}
histread() {   # histread <stage> <out>
  PYTHONIOENCODING=utf-8 $PY "$RD/v8f_hist_read.py" --stage $1 --eval-dir "$(cygpath -m $E/eval)" --runs-dir "$(cygpath -m $E/runs)" \
    --out "$(cygpath -m $E/$2)" || stop "hist read $1"
}
jget() { $PY -c "import json,sys;print(json.load(open(sys.argv[1],encoding='utf-8'))[sys.argv[2]])" "$(cygpath -m $E/$1)" $2; }
echo "$(date '+%m-%d %H:%M') waiting for the predecessor chain ($AFTER_OUT)"
until grep -q "^$AFTER_DONE" "$AFTER_OUT" 2>/dev/null; do
  grep -q "^$AFTER_STOP" "$AFTER_OUT" 2>/dev/null && { echo "ZZV8FMINI-STOP the predecessor stopped: $(grep "^$AFTER_STOP" "$AFTER_OUT" | tail -1)"; exit 1; }
  sleep 300
done
[ -f $B/BOX2_BUSY.txt ] && { echo "ZZV8FMINI-STOP busy marker present: $(cat $B/BOX2_BUSY.txt)"; exit 1; }
echo "v8F-mini PREREG_V8F_HIST (H = M2b + --hist 10, B = M2b; seed 0, then seed 1 on CONTINUE) since $(date '+%m-%d %H:%M'), ~4.5-5 h per seed. Do NOT launch box 2 jobs while this file exists." > $B/BOX2_BUSY.txt
BOOT0=$(boot); [ -n "$BOOT0" ] || stop "box 2 unreachable"
echo "$(date '+%m-%d %H:%M') boot0=$BOOT0"
# ---- 1. code refresh from the remote tip
git -C $MIRROR fetch -q origin $BR; T=$(git -C $MIRROR ls-remote origin $BR | cut -f1)
git -C $MIRROR cat-file -e "$T^{commit}" || stop "remote tip $T is not in the mirror"
QT=$B/v8fhist_queue_$TAG.cmd; git -C $MIRROR show "$T:$PK/code/v8f_mini_box2_queue.cmd" | tr -d '\r' > $QT
QC=$B/v8fhist_queue_$TAG.crlf.cmd; git -C $MIRROR show "$T:$PK/code/v8f_mini_box2_queue.cmd" > $QC      # the committed bytes (CRLF): what cmd.exe can run for the dry-run line
for st in gatehist trainhist evalhist; do grep -q "^:$st\$" $QT || stop "the tip's queue has no $st stage"; done
for f in v8f_hist_read.py v8f_grid_read.py v8f_pgt_read.py v8f_dk_read.py v8f_wc_read.py v8f_tac_groups_read.py v8f_mini_full_report.py; do
  git -C $MIRROR cat-file -e "$T:$PK/code/$f" || stop "the tip has no $f"
done
git -C $MIRROR show "$T:stack/tanitad/models/v8f_mini.py" | grep -q '"p_gt_start": cfg.p_gt_start' || stop "the tip's built_levers does not record p_gt_start"
git -C $MIRROR show "$T:stack/tanitad/models/v8f_mini.py" | grep -q '"p_gt_end": cfg.p_gt_end' || stop "the tip's built_levers does not record p_gt_end"
git -C $MIRROR show "$T:stack/tanitad/models/v8f_mini.py" | grep -q '"window_start": cfg.window_start_effective' || stop "the tip's built_levers does not record window_start (Amendment 1)"
git -C $MIRROR show "$T:stack/scripts/v8f_mini_train.py" | grep -q -- '"--window-start"' || stop "the tip's trainer has no --window-start flag (Amendment 1)"
git -C $MIRROR cat-file -e "$T:$B1" || stop "the tip has no B1 lead block"
git -C $MIRROR -c core.autocrlf=false archive --format=tar -o $B/repo_v8fhist_tip.tar $T -- stack taniteval "$HJ" "$PK" "$B1" || stop "archive"
$SCP $B/repo_v8fhist_tip.tar tanitad-box2:D:/tanitad/repo_v8fhist_tip.tar || stop "ship the code archive"
printf '@echo off\r\necho ===== refresh %s %%DATE%% %%TIME%%\r\ntar -xf D:\\tanitad\\repo_v8fhist_tip.tar -C D:\\tanitad\\repo\r\nset RC=%%ERRORLEVEL%%\r\necho %s> D:\\tanitad\\repo\\TIP_V8FHIST.txt\r\necho ===== ZZJOB-END %%DATE%% %%TIME%% rc=%%RC%%\r\n' "$T" "$T" > $B/v8fhist${TAG}_refresh.cmd
$SCP $B/v8fhist${TAG}_refresh.cmd tanitad-box2:D:/tanitad/jobs/ || stop "ship refresh job"
launch v8fhist${TAG}_refresh || stop "launch refresh"
waitrc v8fhist${TAG}_refresh 40
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
X=$B/v8fhist_tip_$(date +%Y%m%d%H%M%S); mkdir -p $X && tar -xf $B/repo_v8fhist_tip.tar -C $X || stop "extract the reader"
RD="$(cygpath -m "$X/$PK/code")"
# ---- 2. the base is fixed (B = M2b): there is nothing to decide
# ---- 3. the B side, staged into this chain's dirs BEFORE any training (a missing file stops the chain while box 2 is still idle)
for s in 0 1; do
  mkdir -p $E/runs/M2b_s$s
  for f in summary.json argv.json selfcheck.json; do cp "$L/v8f_mini/runs/M2b_s$s/$f" $E/runs/M2b_s$s/ || stop "stage the banked M2b_s$s/$f"; done
  cp "$L/v8f_mini/runs/M2b_s$s/train_log.jsonl" $E/runs/M2b_s$s/ 2>/dev/null || echo "$(date '+%m-%d %H:%M') NOTE: no banked M2b_s$s/train_log.jsonl: its p_gt schedule is NOT-CHECKED (by construction 1.0 -> 0.2)"
  cp $L/v8f_mini/eval/eval_M2b_s$s.npz $L/v8f_mini/eval/eval_M2b_s$s.json $E/eval/ || stop "stage eval_M2b_s$s"
  for f in eval_M2b_s$s-tacabs.npz eval_M2b_s$s-tacabs.json eval_M2b_s$s-tacorc.npz; do cp "$L/v8f_cz/eval/$f" $E/eval/ || stop "stage $f"; done
  # M2b's tacorc predates tac_code (WC Amendment 1): its GT constraints come from the all-goal oracle; M-LEAK needs the all-latcons oracle (same dir)
  for f in eval_M2b_s$s-tacorc-all-goal.npz eval_M2b_s$s-tacorc-all-latcons.npz; do cp "$L/v8f_tacgrp2/eval/$f" $E/eval/ || stop "stage $f"; done
  cp "$L/v8f_tacgrp2/eval/eval_M2b_s$s-tacorc-all-latcons.json" $E/eval/ 2>/dev/null
done
cp "$L/v8f_mini/eval/floors_perwindow.npz" $E/eval/ || stop "stage floors_perwindow.npz (the P10 reproduction guard)"
# the reader must load the REAL B side BEFORE any training: runs, evals, the PREREG's known values, the committed window counts, the identity control
PYTHONIOENCODING=utf-8 $PY "$RD/v8f_hist_read.py" --check-inputs --eval-dir "$(cygpath -m $E/eval)" --runs-dir "$(cygpath -m $E/runs)" \
  --out "$(cygpath -m $E/check_inputs.json)" || stop "the reader cannot use the M2b side (check-inputs)"
$SSH "powershell -NoProfile -Command \"if (Test-Path D:/tanitad/logs/telemetry.stop) { Remove-Item D:/tanitad/logs/telemetry.stop }\"" 2>/dev/null
launch telemetry_long || echo "$(date '+%m-%d %H:%M') telemetry did not launch"
tl() { $SSH "powershell -NoProfile -Command \"Get-Content D:/tanitad/logs/telemetry_full.csv -Tail 1\"" 2>/dev/null | tr -d '\r'; }
sleep 45; t1=$(tl); sleep 35; t2=$(tl)
[[ $t2 == 20* && $t2 != "$t1" ]] || stop "no fresh telemetry ($t1 / $t2): the temperature guard would be blind"
grep -qF "'^(ZZV8FMINI|" $B/v8f_guard.sh || stop "v8f_guard.sh does not know this chain's markers (anchored): it would never exit"
echo "" > $CUR
V8F_CHAIN_OUT=$L/chain_v8f_hist.out V8F_CURRENT=$CUR bash $B/v8f_guard.sh > $L/v8fhist_guard.out 2>&1 &
# ---- 4. the launch gate for H, asserted here as well as in-process. GLINE is the line the queue PRINTS for the stage: a static test pins it to the line the stage RUNS.
GLINE=$(dry_stage "$QC" gatehist | grep -a 'v8f_mini_train.py'); [ -n "$GLINE" ] || stop "the tip's queue prints no gatehist line"
queue gatehist 120 gatehist
pull "gate_M2b-h10/selfcheck.json" "$E/gate_M2b-h10"
$PY - "$(cygpath -m $E/gate_M2b-h10/selfcheck.json)" $HOLDOUT_SHA $M2B_STEPS "$(cygpath -m $E/runs/M2b_s0/summary.json)" "$GLINE" $M2B_WINDOWS <<'EOF' || stop "gate H assertions"
import json, sys
sc, hold, steps, b, line, windows = json.load(open(sys.argv[1])), sys.argv[2], int(sys.argv[3]), json.load(open(sys.argv[4])), sys.argv[5], int(sys.argv[6])
i, lv, bl = sc["info"], sc["levers"], b["levers"]
bad = []
if sc.get("declared_vs_built") != "PASS": bad.append("declared_vs_built")
if not sc["live"].get("pass") or sc["live"].get("control_prior_grad_sum") != 0.0: bad.append(f"live {sc['live'].get('pass')}")
if (i["n_clips"], i["n_excluded"], i["n_excluded_declared"], i["excluded_in_train"]) != (4512, 60, 60, 0): bad.append("clips / holdout counts")
if i["excluded_sha256"] != hold: bad.append("the excluded set is not the RA-1 holdout")
if i["train_sha256"] != b["train_sha256"]: bad.append(f"TRAIN list {i['train_sha256'][:12]} != B's {b['train_sha256'][:12]}")
if i["kin_mode"] != "yaw": bad.append("kin_mode")
if (lv["wm"], lv["navonly"], lv["consistency"], lv.get("cons_absent")) != ("b", False, True, "mask"): bad.append(f"levers {lv}")
# B's schedule is H's: M2b's is the trainer's default 1.0 -> 0.2; H moves the HISTORY and nothing else
if (lv.get("p_gt_start"), lv.get("p_gt_end")) != (1.0, 0.2): bad.append(f"built p_gt {lv.get('p_gt_start')} -> {lv.get('p_gt_end')}, expected 1.0 -> 0.2")
w = lv.get("w") or {}
for k in ("w_pen", "w_pen_self", "w_wm"):
    if w.get(k) != bl["w"][k]: bad.append(f"{k} {w.get(k)} != B's {bl['w'][k]}")
if (w.get("w_tac"), w.get("w_cons")) != (1.0, 0.5): bad.append(f"w_tac / w_cons {w.get('w_tac')} / {w.get('w_cons')} != B's 1.0 / 0.5 (a second lever moved)")
# THE PREREG'S CLAIMS ABOUT THE BUDGET: "19,164 steps" and "the window set ... [is] B's" -- true again under Amendment 1. The trainer sizes S from train_windows(hist=cfg.hist, start=window_start),
# whose DEFAULT first window is t = hist - 1: at --hist 10 ALONE the t = 3..8 windows of every clip are gone, so n_windows, S and steps all FALL (BUILD.md F-1: 279,494 windows, 17,472 steps).
# --window-start 3 restores B's 306,566 windows and 19,164 steps; asserted as the PREREG states them, so a trainer in which the flag does not reach train_windows stops the chain HERE.
hint = f" -- n_windows_all_clips {i.get('n_windows_all_clips')} vs B's {b.get('n_windows')}: the trainer's window start follows --hist (train_windows, t >= hist - 1) unless --window-start reaches it, BUILD.md F-1 / Amendment 1"
if i.get("n_windows_all_clips") != b.get("n_windows"): bad.append(f"the window set is not B's: n_windows_all_clips {i.get('n_windows_all_clips')} != B's {b.get('n_windows')}" + hint)
if i.get("n_windows_all_clips") != windows: bad.append(f"n_windows_all_clips {i.get('n_windows_all_clips')} != the PREREG's {windows}" + hint)
if sc["steps_total"] != steps: bad.append(f"steps {sc['steps_total']} != B's {steps}" + hint)
# THE BUILT HISTORY AND WINDOW START, from the run's own record (BUILD.md F-3): the model's levers (hist = the time table's rows) and the data's read-backs (the window list's first t, the dataset's
# held-age clamp origin) must all say what the argv says. A flag the library supports and the caller never passed shows here as a 9, not a 3.
if lv.get("hist") != 10: bad.append(f"built hist {lv.get('hist')} != 10")
if lv.get("hmax") != 16: bad.append(f"built hmax {lv.get('hmax')} != 16 (the arm's auto value, B's)")
if lv.get("window_start") != 3: bad.append(f"built window_start {lv.get('window_start')} != 3 (hist - 1 = 9 means the flag did not reach the model's levers)")
if (i.get("window_start"), i.get("window_t_min"), i.get("ds_window_start")) != (3, 3, 3): bad.append(f"window start (asked / first t of the built window list / dataset clamp origin) {(i.get('window_start'), i.get('window_t_min'), i.get('ds_window_start'))} != (3, 3, 3)")
# THE LEVER, from the BUILT model: the per-frame time table is the only per-history parameter, so H has (10 - 4) x 256 = 1,536 more parameters, all in the adapter group, and nothing else moved
D = (10 - 4) * 256
pg, pb = lv["params"], bl["params"]
if lv["n_params_trainable"] != bl["n_params_trainable"] + D: bad.append(f"n_params_trainable {lv['n_params_trainable']} != B's {bl['n_params_trainable']} + {D}")
for k in sorted(set(pg) | set(pb)):
    if pg.get(k, 0) - pb.get(k, 0) != (D if k == "adapter" else 0): bad.append(f"parameter group {k}: {pg.get(k)} vs B's {pb.get(k)}")
# THE HISTORY, from the queue's own printed line and the built table: --hist 10 --window-start 3 once each; no --hmax (hmax is the arm's auto value 16 -- B's, so the batch content is B's);
# the built time table has 4 + 1,536 / 256 = 10 rows
toks = line.split()
if toks.count("--hist") != 1 or toks[toks.index("--hist") + 1] != "10": bad.append("the queue's gatehist line does not carry --hist 10 exactly once")
if toks.count("--window-start") != 1 or toks[toks.index("--window-start") + 1] != "3": bad.append("the queue's gatehist line does not carry --window-start 3 exactly once (Amendment 1)")
if "--hmax" in toks: bad.append("the queue's gatehist line sets --hmax: hmax is not the arm's auto value (B's)")
if lv["wm"] != bl.get("wm"): bad.append(f"wm {lv['wm']} != B's {bl.get('wm')}: the auto hmax (16 with a world-model auxiliary) would differ from B's")
if pb.get("adapter") is None or pg.get("adapter") is None or 4 + (pg["adapter"] - pb["adapter"]) // 256 != 10 or (pg["adapter"] - pb["adapter"]) % 256: bad.append("the built time table does not have 10 rows")
if not i.get("lru"): bad.append("info.lru is not recorded in the selfcheck (the pipeline sizing must be on the record)")
print("H gate:", "PASS" if not bad else bad, "| clips", i["n_clips"], "steps", sc["steps_total"], "windows", i.get("n_windows_all_clips"), "| built hist/hmax/window_start", lv.get("hist"), lv.get("hmax"), lv.get("window_start"), "| built p_gt", lv.get("p_gt_start"), "->", lv.get("p_gt_end"),
      "| params", lv["n_params_trainable"], "(B", bl["n_params_trainable"], "+", D, ") | lru", i.get("lru"))
sys.exit(1 if bad else 0)
EOF
# ---- 5. STAGE 1: seed 0, read against B
runone 0
histread s0 hist_read_s0.json
H1=$(jget hist_read_s0.json reading); echo "$(date '+%m-%d %H:%M') $H1"
[ "$(jget hist_read_s0.json DECISION)" = CONTINUE ] || finish "$H1 -- seed 1 not trained (committed)"
# ---- 6. STAGE 2: seed 1, the final reading
runone 1
histread final hist_read.json
finish "$H1 | $(jget hist_read.json reading)"
