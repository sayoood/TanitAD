#!/usr/bin/env bash
# PREREG_V8F_DSCALE on box 2: is v8F-mini's T data-limited? Q = M2b + --data-frac 0.5 (ONE lever: the first round(0.5 x N_B) clips of B's list in sha12 order, a nested half) at B's 19,164 steps. SEED 0 ONLY.
# ARMED ONLY AFTER PREREG_V8F_HIST's chain ends, and only on the PI's go. One job at a time under SAFE MODE (mk_v8f_job.py). Derived from chain_v8f_hist.sh; markers ZZV8FMINI-DONE / -STOP in ITS OWN log
# (chain_v8f_dscale.out), which v8f_guard.sh's anchored end test already knows.
#   chain_v8f_dscale.sh --dry-run   PRINTS the exact command every stage runs, from the tip's queue or $V8F_QUEUE, and exits; nothing is shipped, launched or waited for.
#                                   The line printed IS the line that runs.
#   0. wait for the predecessor (default: PREREG_V8F_HIST's chain, chain_v8f_hist.out, ZZV8FMINI-DONE; stop on its ZZV8FMINI-STOP)
#   1. refresh box 2's code from the REMOTE tip; sha-check 8 key files, md5-check both v9 releases; extract the reader HERE from the same archive; pull box 2's cache train manifest (the list B trained on)
#   2. THE BASE is fixed in the PREREG: B = M2b. There is no --decide-base.
#   3. the B side is staged from the banked M2b files (exactly as chain_v8f_hist.sh) and `v8f_dscale_read.py --check-inputs` runs on it BEFORE any training: runs, evals, the PREREG's known values,
#      the manifest reproducing B's recorded train_sha256 (so it IS B's list) and giving the half Q must train on, the committed window counts, the identity control (+0 [0, 0], D-T included)
#   4. the in-process launch gate for Q (`gatedscale`), pulled and asserted HERE: the BUILT data fraction (n_clips / n_clips_all) is exactly 0.5, 2,256 of 4,512 clips, the clips ARE the first half of B's
#      list in sha12 order (train_sha256 recomputed from the manifest), B's holdout withheld, 19,164 steps and B's S (the budget counts ALL clips: 306,566 windows) with the half's own window count
#      from the manifest's frame counts, B's levers / weights / p_gt schedule / built hist 4, hmax 16, window_start 3, B's parameter count EXACTLY (data moves, the model does not), and the queue's own
#      printed line (`--data-frac 0.5` once, nothing else of the lever family: no --steps, --max-clips, --hist, --window-start, --hmax, --w-*, --p-gt)
#   5. Q s0 (`traindscale 0`) -> its four evals (`evaldscale 0`); v8f_dscale_read.py -> dscale_read.json (DATA-LIMITED / NOT DATA-LIMITED / EPOCH-LIMITED, committed) -> DONE. No seed-1 stage.
# Stops on a reboot (waitjob 2), unreachable (3), timeout (4), any nonzero rc, or a guard stop. Script polling, not model turns.
# TIMEOUTS: the training wait is 1,320 polls of 30 s = 11 h, one hour above the guard's 10 h cap (V8F_MAX_RUN_H): Q is ESTIMATED at B's ~4.4 h (the same steps and scene tokens; half the clips only
# improves the clip cache's hit rate), and v8f_guard.sh stops any run it PROJECTS above the cap, so no run the guard lets live can outlast the wait. The PREREG fixes steps, not wall-clock.
PK="TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
dry_stage() {   # dry_stage <queue file> <stage args...>: the line(s) the queue prints for a stage under V8F_DRYRUN. cmd.exe cannot be handed `"<path with an ampersand>" args`
                # through MSYS quoting (it strips the quotes it needs): a one-line wrapper batch file can
  local qf=$1; shift
  local w=${TMPDIR:-/tmp}/v8fdscale_dry_$$.cmd
  printf '@echo off\r\nset V8F_DRYRUN=1\r\ncall "%s" %s\r\n' "$(cygpath -w "$qf")" "$*" > "$w"
  MSYS_NO_PATHCONV=1 cmd /c "$(cygpath -w "$w")" </dev/null | tr -d '\r'
}
if [ "${1:-}" = "--dry-run" ]; then
  QF=${V8F_QUEUE:-}
  if [ -z "$QF" ]; then
    MIRROR=${MIRROR:-/c/Users/Admin/tanitad-push}; BR=${BR:-agent/arch-inf-20260803}
    T=$(git -C $MIRROR ls-remote origin $BR | cut -f1); QF=${TMPDIR:-/tmp}/v8fdscale_queue_dry_$T.cmd
    git -C $MIRROR show "$T:$PK/code/v8f_mini_box2_queue.cmd" > "$QF" || { echo "ZZV8FMINI-STOP no queue at the tip"; exit 1; }
  fi
  for st in "gatedscale" "traindscale 0" "evaldscale 0"; do
    echo "== $st"; dry_stage "$QF" $st || exit 1
  done
  exit 0
fi
B=/c/Users/Admin/box2_ship; L=$B/logs; E=$L/v8f_dscale; mkdir -p $E/runs $E/eval
MIRROR=/c/Users/Admin/tanitad-push; BR=agent/arch-inf-20260803
HJ="TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-refav1-hj-mini"
B1="TanitAD Research Lab/Benchmarks & Evals/Research/2026-09-02-b1-eval-lead-block/raw/b1_eval_lead_block.npz"
SSH="ssh -o BatchMode=yes -o ConnectTimeout=20 tanitad-box2"
SCP="scp -q -o BatchMode=yes -o ConnectTimeout=20"
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
R=D:/tanitad/runs/v8f_mini_full
HOLDOUT_SHA=f6a1dee02af0b96f815416bfef9bc11a04554702525135aa75a740942d7750f9
M2B_STEPS=19164
M2B_WINDOWS=306566   # B's n_windows_all_clips (banked summary.json): Q's step budget counts ALL clips, so Q's S and steps are B's whatever the fraction
TRAIN_POLLS=1320; EVAL_POLLS=600   # 11 h: one hour above the guard cap, so the GUARD (a committed rule), never the wait, ends a slow run
TAG=$(date '+%m%d%H%M')
CUR=$L/v8fdscale_current.txt
AFTER_OUT=${AFTER_OUT:-$L/chain_v8f_hist.out}; AFTER_DONE=${AFTER_DONE:-ZZV8FMINI-DONE}; AFTER_STOP=${AFTER_STOP:-ZZV8FMINI-STOP}
boot() { $SSH "powershell -NoProfile -Command \"(Get-CimInstance Win32_OperatingSystem).LastBootUpTime.ToString('yyyy-MM-dd HH:mm:ss')\"" 2>/dev/null | tr -d '\r' | grep -E '^[0-9]{4}-' | head -1; }
launch() { for t in 1 2 3; do o=$($SSH "powershell -NoProfile -ExecutionPolicy Bypass -File D:\tanitad\launch.ps1 -Job D:\tanitad\jobs\\$1.cmd -Log D:\tanitad\logs\\$1.log" 2>/dev/null | tr -d '\r') && { echo "$(date '+%m-%d %H:%M') $o"; return 0; }; sleep 20; done; return 1; }
clean() { tr '\r' '\n' < "$L/$1.log"; }
stop() { echo "ZZV8FMINI-STOP $*"; echo "" > $CUR; exit 1; }
waitrc() {   # waitrc <job> <polls of 30 s>
  bash $B/waitjob.sh $1 $2 "$BOOT0"; local w=$?
  [ $w = 0 ] || stop "$1 wait=$w (2 = CRASH/reboot, 3 = unreachable, 4 = timeout)"
  local rc=$(clean $1 | grep -a 'ZZJOB-END' | tail -1 | sed -n 's/.* rc=\([-0-9]*\).*/\1/p')
  echo "$(date '+%m-%d %H:%M') $1 rc=$rc"
  [ "$rc" = 0 ] || { clean $1 | grep -a -i -E 'FAIL|refus|error|Traceback' | tail -8; grep ZZGUARD-STOP $L/v8fdscale_guard.out 2>/dev/null; stop "$1 rc=$rc"; }
}
queue() {    # queue <job> <polls> <queue args...>: every job name carries this chain's TAG, so an old log's ZZJOB-END can never satisfy a new wait
  local j="v8fdscale${TAG}_$1" m=$2; shift 2
  $PY "$(cygpath -m $B)/mk_v8f_job.py" $j "$@" >/dev/null && $SCP $B/$j.cmd tanitad-box2:D:/tanitad/jobs/ || stop "ship $j"
  rm -f $L/$j.log
  launch $j || stop "launch $j"
  waitrc $j $m
}
rsha() { $SSH "powershell -NoProfile -Command \"(Get-FileHash -Algorithm $2 -LiteralPath '$1').Hash.ToLower()\"" 2>/dev/null | tr -d '\r' | grep -E '^[0-9a-f]{32,64}$'; }
pull() {     # pull <box-2 path under R> <local dir>
  mkdir -p "$2"; $SCP "tanitad-box2:$R/$1" "$2/" || stop "pull $R/$1"
}
finish() {   # finish <summary>: the committed end of the chain
  $SSH "powershell -NoProfile -Command \"New-Item -ItemType File -Force D:/tanitad/logs/telemetry.stop | Out-Null\"" 2>/dev/null
  $SCP tanitad-box2:D:/tanitad/logs/telemetry_full.csv $E/telemetry_v8fdscale.csv 2>/dev/null
  echo "" > $CUR
  rm -f $B/BOX2_BUSY.txt
  echo "ZZV8FMINI-DONE $(date '+%m-%d %H:%M') tip $T: $1 (boot unchanged $BOOT0)"
  exit 0
}
runone() {   # runone <seed>: train Q, eval (+ absmask + GT oracle + all-latcons oracle), pull
  local s=$1 g="M2b_s${1}_df50" tg="M2b-df50_s${1}"
  echo "$g" > $CUR
  queue "t_$g" $TRAIN_POLLS traindscale $s
  echo "" > $CUR
  queue "e_$g" $EVAL_POLLS evaldscale $s
  for f in selfcheck.json summary.json argv.json train_log.jsonl train.log ckpt_last.pt; do pull "$g/$f" "$E/runs/$g"; done
  for f in "" -tacabs -tacorc -tacorc-all-latcons; do for x in json npz log; do pull "eval/eval_$tg$f.$x" "$E/eval"; done; done
  grep ZZGUARD-STOP $L/v8fdscale_guard.out 2>/dev/null && stop "guard stop"
}
dscaleread() {   # dscaleread <out>
  PYTHONIOENCODING=utf-8 $PY "$RD/v8f_dscale_read.py" --eval-dir "$(cygpath -m $E/eval)" --runs-dir "$(cygpath -m $E/runs)" --train-manifest "$(cygpath -m $E/manifest_train.jsonl)" \
    --out "$(cygpath -m $E/$1)" || stop "dscale read"
}
jget() { $PY -c "import json,sys;print(json.load(open(sys.argv[1],encoding='utf-8'))[sys.argv[2]])" "$(cygpath -m $E/$1)" $2; }
echo "$(date '+%m-%d %H:%M') waiting for the predecessor chain ($AFTER_OUT)"
until grep -q "^$AFTER_DONE" "$AFTER_OUT" 2>/dev/null; do
  grep -q "^$AFTER_STOP" "$AFTER_OUT" 2>/dev/null && { echo "ZZV8FMINI-STOP the predecessor stopped: $(grep "^$AFTER_STOP" "$AFTER_OUT" | tail -1)"; exit 1; }
  sleep 300
done
[ -f $B/BOX2_BUSY.txt ] && { echo "ZZV8FMINI-STOP busy marker present: $(cat $B/BOX2_BUSY.txt)"; exit 1; }
echo "v8F-mini PREREG_V8F_DSCALE (Q = M2b + --data-frac 0.5, B = M2b; seed 0 only) since $(date '+%m-%d %H:%M'), ~4.5 h. Do NOT launch box 2 jobs while this file exists." > $B/BOX2_BUSY.txt
BOOT0=$(boot); [ -n "$BOOT0" ] || stop "box 2 unreachable"
echo "$(date '+%m-%d %H:%M') boot0=$BOOT0"
# ---- 1. code refresh from the remote tip
git -C $MIRROR fetch -q origin $BR; T=$(git -C $MIRROR ls-remote origin $BR | cut -f1)
git -C $MIRROR cat-file -e "$T^{commit}" || stop "remote tip $T is not in the mirror"
QT=$B/v8fdscale_queue_$TAG.cmd; git -C $MIRROR show "$T:$PK/code/v8f_mini_box2_queue.cmd" | tr -d '\r' > $QT
QC=$B/v8fdscale_queue_$TAG.crlf.cmd; git -C $MIRROR show "$T:$PK/code/v8f_mini_box2_queue.cmd" > $QC      # the committed bytes (CRLF): what cmd.exe can run for the dry-run line
for st in gatedscale traindscale evaldscale; do grep -q "^:$st\$" $QT || stop "the tip's queue has no $st stage"; done
for f in v8f_dscale_read.py v8f_hist_read.py v8f_grid_read.py v8f_pgt_read.py v8f_dk_read.py v8f_wc_read.py v8f_tac_groups_read.py v8f_mini_full_report.py; do
  git -C $MIRROR cat-file -e "$T:$PK/code/$f" || stop "the tip has no $f"
done
git -C $MIRROR show "$T:stack/tanitad/models/v8f_mini.py" | grep -q '"p_gt_start": cfg.p_gt_start' || stop "the tip's built_levers does not record p_gt_start"
git -C $MIRROR show "$T:stack/tanitad/models/v8f_mini.py" | grep -q '"p_gt_end": cfg.p_gt_end' || stop "the tip's built_levers does not record p_gt_end"
git -C $MIRROR show "$T:stack/tanitad/models/v8f_mini.py" | grep -q '"window_start": cfg.window_start_effective' || stop "the tip's built_levers does not record window_start (the reader requires it of Q)"
git -C $MIRROR show "$T:stack/scripts/v8f_mini_train.py" | grep -q -- '"--data-frac"' || stop "the tip's trainer has no --data-frac flag"
git -C $MIRROR show "$T:stack/scripts/v8f_mini_train.py" | grep -q '"data_frac": "data:n_clips"' || stop "the tip's trainer has no G-DVB entry for data_frac"
git -C $MIRROR show "$T:stack/scripts/v8f_mini_train.py" | grep -q 'select_train_clips(clips_all, a.data_frac)' || stop "the tip's trainer does not hand --data-frac to select_train_clips"
git -C $MIRROR cat-file -e "$T:$B1" || stop "the tip has no B1 lead block"
git -C $MIRROR -c core.autocrlf=false archive --format=tar -o $B/repo_v8fdscale_tip.tar $T -- stack taniteval "$HJ" "$PK" "$B1" || stop "archive"
$SCP $B/repo_v8fdscale_tip.tar tanitad-box2:D:/tanitad/repo_v8fdscale_tip.tar || stop "ship the code archive"
printf '@echo off\r\necho ===== refresh %s %%DATE%% %%TIME%%\r\ntar -xf D:\\tanitad\\repo_v8fdscale_tip.tar -C D:\\tanitad\\repo\r\nset RC=%%ERRORLEVEL%%\r\necho %s> D:\\tanitad\\repo\\TIP_V8FDSCALE.txt\r\necho ===== ZZJOB-END %%DATE%% %%TIME%% rc=%%RC%%\r\n' "$T" "$T" > $B/v8fdscale${TAG}_refresh.cmd
$SCP $B/v8fdscale${TAG}_refresh.cmd tanitad-box2:D:/tanitad/jobs/ || stop "ship refresh job"
launch v8fdscale${TAG}_refresh || stop "launch refresh"
waitrc v8fdscale${TAG}_refresh 40
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
X=$B/v8fdscale_tip_$(date +%Y%m%d%H%M%S); mkdir -p $X && tar -xf $B/repo_v8fdscale_tip.tar -C $X || stop "extract the reader"
RD="$(cygpath -m "$X/$PK/code")"
# the list B trained on: box 2's cache train manifest (clip, n_frames), the source of truth the half is derived from; the reader REFUSES it unless its B list reproduces B's recorded train_sha256
$SCP tanitad-box2:D:/tanitad/cache_full/train/manifest.jsonl $E/manifest_train.jsonl || stop "pull box 2's cache train manifest"
# ---- 2. the base is fixed (B = M2b): there is nothing to decide
# ---- 3. the B side, staged into this chain's dirs BEFORE any training (a missing file stops the chain while box 2 is still idle)
for s in 0 1; do
  mkdir -p $E/runs/M2b_s$s
  for f in summary.json argv.json selfcheck.json; do cp "$L/v8f_mini/runs/M2b_s$s/$f" $E/runs/M2b_s$s/ || stop "stage the banked M2b_s$s/$f"; done
  cp "$L/v8f_mini/runs/M2b_s$s/train_log.jsonl" $E/runs/M2b_s$s/ 2>/dev/null || echo "$(date '+%m-%d %H:%M') NOTE: no banked M2b_s$s/train_log.jsonl: its p_gt schedule and the training-loss beside read are NOT-CHECKED"
  cp $L/v8f_mini/eval/eval_M2b_s$s.npz $L/v8f_mini/eval/eval_M2b_s$s.json $E/eval/ || stop "stage eval_M2b_s$s"
  for f in eval_M2b_s$s-tacabs.npz eval_M2b_s$s-tacabs.json eval_M2b_s$s-tacorc.npz; do cp "$L/v8f_cz/eval/$f" $E/eval/ || stop "stage $f"; done
  # M2b's tacorc predates tac_code (WC Amendment 1): its GT constraints come from the all-goal oracle; M-LEAK needs the all-latcons oracle (same dir)
  for f in eval_M2b_s$s-tacorc-all-goal.npz eval_M2b_s$s-tacorc-all-latcons.npz; do cp "$L/v8f_tacgrp2/eval/$f" $E/eval/ || stop "stage $f"; done
  cp "$L/v8f_tacgrp2/eval/eval_M2b_s$s-tacorc-all-latcons.json" $E/eval/ 2>/dev/null
done
cp "$L/v8f_mini/eval/floors_perwindow.npz" $E/eval/ || stop "stage floors_perwindow.npz (the P10 reproduction guard)"
# the reader must load the REAL B side BEFORE any training: runs, evals, the PREREG's known values, the manifest, the committed window counts, the identity control
PYTHONIOENCODING=utf-8 $PY "$RD/v8f_dscale_read.py" --check-inputs --eval-dir "$(cygpath -m $E/eval)" --runs-dir "$(cygpath -m $E/runs)" --train-manifest "$(cygpath -m $E/manifest_train.jsonl)" \
  --out "$(cygpath -m $E/check_inputs.json)" || stop "the reader cannot use the M2b side (check-inputs)"
$SSH "powershell -NoProfile -Command \"if (Test-Path D:/tanitad/logs/telemetry.stop) { Remove-Item D:/tanitad/logs/telemetry.stop }\"" 2>/dev/null
launch telemetry_long || echo "$(date '+%m-%d %H:%M') telemetry did not launch"
tl() { $SSH "powershell -NoProfile -Command \"Get-Content D:/tanitad/logs/telemetry_full.csv -Tail 1\"" 2>/dev/null | tr -d '\r'; }
sleep 45; t1=$(tl); sleep 35; t2=$(tl)
[[ $t2 == 20* && $t2 != "$t1" ]] || stop "no fresh telemetry ($t1 / $t2): the temperature guard would be blind"
grep -qF "'^(ZZV8FMINI|" $B/v8f_guard.sh || stop "v8f_guard.sh does not know this chain's markers (anchored): it would never exit"
echo "" > $CUR
V8F_CHAIN_OUT=$L/chain_v8f_dscale.out V8F_CURRENT=$CUR bash $B/v8f_guard.sh > $L/v8fdscale_guard.out 2>&1 &
# ---- 4. the launch gate for Q, asserted here as well as in-process. GLINE is the line the queue PRINTS for the stage: a static test pins it to the line the stage RUNS.
GLINE=$(dry_stage "$QC" gatedscale | grep -a 'v8f_mini_train.py'); [ -n "$GLINE" ] || stop "the tip's queue prints no gatedscale line"
queue gatedscale 120 gatedscale
pull "gate_M2b-df50/selfcheck.json" "$E/gate_M2b-df50"
$PY - "$(cygpath -m $E/gate_M2b-df50/selfcheck.json)" $HOLDOUT_SHA $M2B_STEPS "$(cygpath -m $E/runs/M2b_s0/summary.json)" "$GLINE" $M2B_WINDOWS "$(cygpath -m $E/manifest_train.jsonl)" <<'EOF' || stop "gate Q assertions"
import hashlib, json, sys
sc, hold, steps, b, line, windows, man = json.load(open(sys.argv[1])), sys.argv[2], int(sys.argv[3]), json.load(open(sys.argv[4])), sys.argv[5], int(sys.argv[6]), sys.argv[7]
i, lv, bl = sc["info"], sc["levers"], b["levers"]
bad = []
# THE LIST, from box 2's own cache manifest: B's list is the manifest minus --exclude 540:600 of its sha12 order; Q's is the FIRST round(0.5 x N_B) of it (select_train_clips). Both are recomputed here.
sha12 = lambda c: hashlib.sha256(c.encode("utf-8")).hexdigest()[:12]
shal = lambda cs: hashlib.sha256(chr(10).join(cs).encode("utf-8")).hexdigest()
ents = {}
for ln in open(man, encoding="utf-8"):
    if ln.strip():
        d = json.loads(ln)
        ents[d["clip"]] = int(d["n_frames"])
allc = list(ents)
excl = sorted(allc, key=sha12)[540:600]
order = sorted([c for c in allc if c not in set(excl)], key=sha12)
n_b, n_h = len(order), max(1, int(round(0.5 * len(order))))
half = order[:n_h]
wins = lambda cs: sum(len(range(3, ents[c] - 30)) for c in cs)       # train_windows at B's hist 4: t in range(3, n_frames - HORIZON 30)
if shal(order) != b["train_sha256"]: bad.append(f"the manifest's B list {shal(order)[:12]} != B's recorded train_sha256 {b['train_sha256'][:12]}: the manifest is not the list B trained on")
if shal(excl) != hold: bad.append("the manifest's excluded clips are not the RA-1 holdout")
if wins(order) != b["n_windows"]: bad.append(f"the manifest's frame counts give {wins(order)} windows for B, B's record says {b['n_windows']}")
if sc.get("declared_vs_built") != "PASS": bad.append("declared_vs_built")
if not sc["live"].get("pass") or sc["live"].get("control_prior_grad_sum") != 0.0: bad.append(f"live {sc['live'].get('pass')}")
# THE CLIPS: round(0.5 x N_B) of N_B, the built fraction read back from the data (not the flag), the first half of B's list in sha12 order, and the holdout still withheld
if (i["n_clips"], i["n_clips_expected"], i["n_clips_all"]) != (n_h, n_h, n_b) or n_b != b["n_clips"] or n_h != 2256: bad.append(f"clips built / declared / of {(i['n_clips'], i['n_clips_expected'], i['n_clips_all'])} != ({n_h}, {n_h}, {n_b}) = round(0.5 x N_B), N_B = B's {b['n_clips']}")
if i["n_clips"] / i["n_clips_all"] != 0.5: bad.append(f"built data fraction {i['n_clips']} / {i['n_clips_all']} != 0.5")
if (i["n_excluded"], i["n_excluded_declared"], i["excluded_in_train"]) != (60, 60, 0): bad.append("holdout counts")
if i["excluded_sha256"] != hold: bad.append("the excluded set is not the RA-1 holdout")
if i["train_sha256"] != shal(half): bad.append(f"the TRAIN list {i['train_sha256'][:12]} is not the first {n_h} of B's list in sha12 order ({shal(half)[:12]})")
if i["train_sha256"] == b["train_sha256"]: bad.append("the TRAIN list is B's whole list: --data-frac did not reach select_train_clips")
if i["kin_mode"] != "yaw": bad.append("kin_mode")
if (lv["wm"], lv["navonly"], lv["consistency"], lv.get("cons_absent")) != ("b", False, True, "mask"): bad.append(f"levers {lv}")
# B's schedule is Q's: M2b's is the trainer's default 1.0 -> 0.2; Q moves the DATA and nothing else
if (lv.get("p_gt_start"), lv.get("p_gt_end")) != (1.0, 0.2): bad.append(f"built p_gt {lv.get('p_gt_start')} -> {lv.get('p_gt_end')}, expected 1.0 -> 0.2")
w = lv.get("w") or {}
for k in ("w_pen", "w_pen_self", "w_wm"):
    if w.get(k) != bl["w"][k]: bad.append(f"{k} {w.get(k)} != B's {bl['w'][k]}")
if (w.get("w_tac"), w.get("w_cons")) != (1.0, 0.5): bad.append(f"w_tac / w_cons {w.get('w_tac')} / {w.get('w_cons')} != B's 1.0 / 0.5 (a second lever moved)")
# THE BUDGET: S = ceil(n_windows_all_clips / batch) counts ALL clips, so Q's steps are B's 19,164 and its S is B's; Q TRAINS on the half's windows (the manifest's frame counts give the exact number)
hint = f" -- n_windows_all_clips {i.get('n_windows_all_clips')} vs B's {b.get('n_windows')}: the trainer sizes S from the windows of ALL clips (build_data's `allw`), never from the half"
if i.get("n_windows_all_clips") != b.get("n_windows"): bad.append(f"the budget is not sized from B's window set: n_windows_all_clips {i.get('n_windows_all_clips')} != B's {b.get('n_windows')}" + hint)
if i.get("n_windows_all_clips") != windows: bad.append(f"n_windows_all_clips {i.get('n_windows_all_clips')} != the PREREG's {windows}" + hint)
if sc["steps_total"] != steps: bad.append(f"steps {sc['steps_total']} != B's {steps}" + hint)
if sc.get("S_steps_per_epoch_100pct") != b.get("S"): bad.append(f"S {sc.get('S_steps_per_epoch_100pct')} != B's {b.get('S')}" + hint)
if i.get("n_windows") != wins(half): bad.append(f"the half trains {i.get('n_windows')} windows, its clips' frame counts give {wins(half)}")
if not 0 < i.get("n_windows", 0) < i["n_windows_all_clips"]: bad.append("the half's window count is not below the all-clips count")
# THE MODEL IS B's: built hist 4, hmax 16, window_start 3 (hist - 1), and B's parameter count EXACTLY, group by group -- data moves, the model does not
if lv.get("hist") != 4: bad.append(f"built hist {lv.get('hist')} != 4")
if lv.get("hmax") != 16: bad.append(f"built hmax {lv.get('hmax')} != 16 (the arm's auto value, B's)")
if lv.get("window_start") != 3: bad.append(f"built window_start {lv.get('window_start')} != 3 (B's, hist - 1)")
pg, pb = lv["params"], bl["params"]
if lv["n_params_trainable"] != bl["n_params_trainable"]: bad.append(f"n_params_trainable {lv['n_params_trainable']} != B's {bl['n_params_trainable']}")
for k in sorted(set(pg) | set(pb)):
    if pg.get(k) != pb.get(k): bad.append(f"parameter group {k}: {pg.get(k)} vs B's {pb.get(k)}")
if lv["wm"] != bl.get("wm"): bad.append(f"wm {lv['wm']} != B's {bl.get('wm')}")
# THE LEVER, from the queue's own printed line: --data-frac 0.5 once, and nothing else of the lever family
toks = line.split()
if toks.count("--data-frac") != 1 or toks[toks.index("--data-frac") + 1] != "0.5": bad.append("the queue's gatedscale line does not carry --data-frac 0.5 exactly once")
for flag in ("--steps", "--max-clips", "--hist", "--window-start", "--hmax", "--w-tac", "--w-cons", "--w-pen", "--w-wm", "--p-gt-start", "--p-gt-end", "--grid", "--seed"):
    if flag in toks: bad.append(f"the queue's gatedscale line sets {flag}: not one lever")
if "--selfcheck-only" not in toks: bad.append("the queue's gatedscale line is not a --selfcheck-only run")
if not i.get("lru"): bad.append("info.lru is not recorded in the selfcheck (the pipeline sizing must be on the record)")
print("Q gate:", "PASS" if not bad else bad, "| clips", i["n_clips"], "of", i["n_clips_all"], "(built fraction", i["n_clips"] / i["n_clips_all"], ") first/last sha12", sha12(half[0]), sha12(half[-1]), "| steps", sc["steps_total"],
      "S", sc.get("S_steps_per_epoch_100pct"), "| windows trained", i.get("n_windows"), "of", i.get("n_windows_all_clips"), "| built hist/hmax/window_start", lv.get("hist"), lv.get("hmax"), lv.get("window_start"),
      "| built p_gt", lv.get("p_gt_start"), "->", lv.get("p_gt_end"), "| params", lv["n_params_trainable"], "(B", bl["n_params_trainable"], ") | lru", i.get("lru"))
sys.exit(1 if bad else 0)
EOF
# ---- 5. seed 0, read against B (SEED 0 ONLY: the deciding read is on T, whose seed floor is 0.2 %; there is no seed-1 stage)
runone 0
dscaleread dscale_read.json
D1=$(jget dscale_read.json reading); echo "$(date '+%m-%d %H:%M') $D1"
finish "$D1 | $(jget dscale_read.json next)"
