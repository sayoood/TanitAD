#!/usr/bin/env bash
# PREREG_V8F_ORECOVER on box 2 (Amendments 1 and 2): does training on perturbed ego states make the hierarchy's decisions survive the model's own motion? FOUR runs, ALWAYS ALL FOUR, in the PREREG's order (section 5):
# ORC s0, ORC-0 s0, ORC s1, ORC-0 s1. ORC = the BANKED M2b_s<seed> checkpoint fine-tuned 3,000 steps with the ego perturbation ON (--ego-perturb-p 0.3); ORC-0 = the same fine-tune with --ego-perturb-p 0 and NONE of the five law flags (PREREG Amendment 3: the trainer's gate
# refuses a law flag set while p is 0; they are inert at p = 0 and vmax's default 8.0 is ORC's) -- the control: it separates the perturbation from "3,000 more steps". Nothing here is conditional on a read. QUEUED AFTER THE CHAIN THAT HOLDS BOX 2 NOW and ARMED ONLY ON THE PI's go (given 2026-10-26: "queue O-RECOVER on box 2 after OACC and MIXO, through the launch gate,
# both seeds"). One job at a time under SAFE MODE (mk_v8f_job.py). Derived from chain_v8f_oacc.sh (refresh, key-file sha, banked-checkpoint sha, gate pulled and asserted, token written by the chain, train, eval, pull) and chain_v8f_mixo.sh
# (the predecessor wait). Markers ZZV8FMINI-DONE / -STOP in ITS OWN log (chain_v8f_orc.out), which v8f_guard.sh's anchored end test already knows.
#   chain_v8f_orc.sh --dry-run    PRINTS the exact command every stage runs against box 2 (the queue's own `gateorc` / `trainorc` / `evalorc` lines, from the tip's queue or $V8F_QUEUE, with box 2's paths pinned), the checkpoint sha256
#                                 values it asserts, the sha256 of each run's TRAINING argv (what its launch-gate token binds) and the token schema, and exits; nothing is shipped, launched or waited for. The line printed IS the line that runs.
#   AFTER_OUT=<log> [AFTER_DONE=.. AFTER_STOP=..] chain_v8f_orc.sh
#                                 AFTER_OUT is REQUIRED and has NO DEFAULT: a default is a path, and a path can already hold an OLD log whose end marker would satisfy the wait on the first poll (MIXO's has ended; OACC holds box 2 now). Unset -> this
#                                 chain refuses to start (ZZV8FMINI-STOP, nothing touched). Set it to the log of the chain that holds box 2.
#   0. wait for the predecessor: ZZV8FMINI-DONE in AFTER_OUT -> go; ZZV8FMINI-STOP there -> this chain stops (fail closed); the busy marker is then taken ATOMICALLY (noclobber)
#   1. refresh box 2's code from the REMOTE tip; sha-check 8 key files, md5-check both v9 releases; the tip must carry the lever (flags, G-DVB entries, probe call, apply line, run totals) and the three queue stages
#   2. the base is fixed by the PREREG: box 2's two M2b checkpoints ARE the banked runs' (sha256, raw/full/MANIFEST.md); a different file stops the chain while box 2 is idle. Asserted again after the last run (the fine-tunes only READ them)
#   3. ALL FOUR GATES FIRST (`gateorc`, the trainer's own in-process gate, v8f_mini_train.py --selfcheck-only: G-DVB incl. the perturbation, --init-ckpt, --warmup; G-LIVE incl. ego_perturb_probe on a PERTURBED batch), each pulled and
#      asserted HERE: a gate that fails for the fourth run is found before the first GPU-hour is spent. The gate's argv is the run's training argv with --out and --selfcheck-only changed and nothing else (asserted). The assertions: the PASS
#      (declared_vs_built, G-LIVE with a frozen-prior control of exactly 0.0 and no dead weight), the ego probe (ORC: the perturbed batch moves the loss and the gradient, every control exactly 0.0; ORC-0: p = 0 reads `off`), the BUILT
#      levers, the init checkpoint's sha256 = the banked one and read back with 0 mismatches, 4,512 clips, the RA-1 holdout withheld, B's clip list, window set (306,566) and parameter count, 3,000 steps, lr 1e-4 / warmup 100 / p_gt 0.2 -> 0.2.
#      The LAUNCH-GATE TOKEN of each run (launch_gate_token.json: kind launch-gate-token, verdict PASS, binding {commit, argv_sha256 of the TRAINING argv, data_sha256}) is written by the chain, never by hand, only after those
#      assertions pass. No PASS => no training (the trainer's own in-process gate refuses as well, ZZDVB-REFUSEDZZ)
#   4. the four runs, in order: `trainorc` -> checkpoint sha256 measured on box 2 -> `evalorc` (the standard T0 eval, --project --families, tag M2b-orc_s<S> / M2b-orc0_s<S>) -> the sha256 again (one value) -> pull to
#      E:/Projects/TanitAD-artifacts/v8f-orecover/<run>/ (checkpoint, argv / summary / selfcheck / train_log / train.log, eval json / npz / log, the token; sha256 printed and written to SHA256SUMS.txt) -> the run record is asserted
#      (argv.json = the planned training argv, so the token binds what ran; done; 3,000 steps; init sha256 banked) -> the PERTURBATION-REACHED control is PRINTED (ORC: ego_perturb_n / eligible in 0.3 +- 0.02; ORC-0: exactly 0)
#      but does not stop the chain: the reader (orc_read.py) judges it and a failed control there is REFUSED, "fix and re-run". Pulled files on E: are never deleted by the chain
#   NOT HERE: the T1-kin rollouts (plain, --limit causal, plus one --limit logged run per arm and seed for K4) and the reader run on the dev box CPU (PREREG section 2 and Amendment 2).
# Stops on a reboot (waitjob 2), unreachable (3), timeout (4), any nonzero rc, or a guard stop. Script polling, not model turns.
# TIMEOUTS: a run is 3,000 steps of M2b's graph (B: 19,164 steps in about 4.4 h, so ESTIMATED about 0.7 h); the training wait is 1,320 polls of 30 s = 11 h, one hour above the guard's 10 h cap (V8F_MAX_RUN_H), so the GUARD (a committed
# rule), never the wait, ends a slow run. The eval is ~2-4 min (wait 600 polls = 5 h); a gate is minutes (wait 120 polls = 1 h).
PK="TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
R=D:/tanitad/runs/v8f_mini_full
# the banked checkpoints (raw/full/MANIFEST.md); orc_read.py pins the same two values (BANKED_M2B_SHA) and a test pins that they agree
CKPT_SHA=(601cb435267317862539c2ba8bcce78b04eb0c31fe08a183ecb1bafc71dcc6f8 6104fb15dc21d524b6ed53be632b8d53ec5413526ac587734c22684c7e50c2e6)
# the four runs "<arm> <seed>", in the PREREG's order: ORC s0, ORC-0 s0, ORC s1, ORC-0 s1. No run is conditional on another's read.
RUNS=("orc 0" "orc0 0" "orc 1" "orc0 1")
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
ART=${ORC_ART:-/e/Projects/TanitAD-artifacts/v8f-orecover}
ORC_STEPS=3000
rn() { echo "M2b_s${2}_${1}"; }       # rn <arm> <seed>: the run's directory under $R (M2b_s0_orc, M2b_s0_orc0, ...)
rt() { echo "M2b-${1}_s${2}"; }       # rt <arm> <seed>: the run's eval tag
rg() { echo "gate_M2b-${1}_s${2}"; }  # rg <arm> <seed>: the run's gate directory under $R
pof() { if [ "$1" = orc ]; then echo 0.3; else echo 0; fi; }     # pof <arm>: the ONE value of --ego-perturb-p that differs between the arms (the queue decides it; the assertions compare with this)
dry_stage() {   # dry_stage <queue file> <stage args...>: the line(s) the queue prints for a stage under V8F_DRYRUN, with BOX 2's paths pinned (the dev box's environment must not decide what box 2 will run). cmd.exe cannot be handed
                # `"<path with an ampersand>" args` through MSYS quoting (it strips the quotes it needs): a one-line wrapper batch file can
  local qf=$1; shift
  local w=${TMPDIR:-/tmp}/v8forc_dry_$$.cmd
  printf '@echo off\r\nset V8F_DRYRUN=1\r\nset "REPO=%s"\r\nset "PY=%s"\r\nset "CACHEFULL=%s"\r\nset "V9=%s"\r\nset "RUNSM=%s"\r\ncall "%s" %s\r\n' \
    'D:\tanitad\repo' 'D:\tanitad\venv\Scripts\python.exe' 'D:\tanitad\cache_full' 'D:\tanitad\data\v9' 'D:\tanitad\runs\v8f_mini_full' "$(cygpath -w "$qf")" "$*" > "$w"
  MSYS_NO_PATHCONV=1 cmd /c "$(cygpath -w "$w")" </dev/null | tr -d '\r'
  rm -f "$w"
}
plan() {   # plan <queue file> <arm> <seed>: GL / TL / EL = the lines the queue prints for the run's gate / train / eval stages. rc 1 if a stage prints no trainer / eval line
  GL=$(dry_stage "$1" gateorc $2 $3 | grep -a 'v8f_mini_train.py'); TL=$(dry_stage "$1" trainorc $2 $3 | grep -a 'v8f_mini_train.py'); EL=$(dry_stage "$1" evalorc $2 $3 | grep -a 'v8f_mini_eval.py')
  [ -n "$GL" ] && [ -n "$TL" ] && [ -n "$EL" ]
}
argvtool() {   # argvtool <json|sha|same> <queue line>: the training argv of a printed line (everything after v8f_mini_train.py; double quotes group and are removed, nothing else is interpreted) as a json array, or its launch-gate sha256
               # (launch_gate.argv_sha256: sha256 of the ORDERED json array, compact separators), or `same` = the gate/train identity (the line given is the gate line; $3 is the train line)
  local f=${TMPDIR:-/tmp}/v8forc_line_$$.txt g=${TMPDIR:-/tmp}/v8forc_line2_$$.txt rc
  printf '%s' "$2" > "$f"; printf '%s' "${3:-}" > "$g"
  $PY - "$1" "$(cygpath -m "$f")" "$(cygpath -m "$g")" <<'EOF'
import hashlib, json, re, sys
mode, p1, p2 = sys.argv[1:4]


def argv_of(path):
    t = [x.strip('"') for x in re.findall(r'"[^"]*"|\S+', open(path, encoding="utf-8").read().strip())]
    return t[next(k for k, x in enumerate(t) if x.lower().endswith("v8f_mini_train.py")) + 1:]


a = argv_of(p1)
if mode == "json":
    print(json.dumps(a))
elif mode == "sha":
    print(hashlib.sha256(json.dumps(a, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest())
elif mode == "same":                                   # gate argv minus (--out <v>, --selfcheck-only) == train argv minus (--out <v>)
    def drop(x, bare):
        o, k = [], 0
        while k < len(x):
            if x[k] == "--out":
                k += 2
            elif x[k] in bare:
                k += 1
            else:
                o.append(x[k])
                k += 1
        return o
    g, t = drop(a, {"--selfcheck-only"}), drop(argv_of(p2), set())
    print("SAME" if g == t and a.count("--selfcheck-only") == 1 else "DIFFERENT")
EOF
  rc=$?; rm -f "$f" "$g"; return $rc
}
if [ "${1:-}" = "--dry-run" ]; then
  QF=${V8F_QUEUE:-}
  if [ -z "$QF" ]; then
    MIRROR=${MIRROR:-/c/Users/Admin/tanitad-push}; BR=${BR:-agent/arch-inf-20260803}
    T=$(git -C $MIRROR ls-remote origin $BR | cut -f1); QF=${TMPDIR:-/tmp}/v8forc_queue_dry_$T.cmd
    git -C $MIRROR show "$T:$PK/code/v8f_mini_box2_queue.cmd" > "$QF" || { echo "ZZV8FMINI-STOP no queue at the tip"; exit 1; }
  fi
  echo "== predecessor"; echo "wait for ${AFTER_OUT:-<UNSET: a real run REFUSES to start without AFTER_OUT>}: ^${AFTER_DONE:-ZZV8FMINI-DONE} (stop on ^${AFTER_STOP:-ZZV8FMINI-STOP})"
  echo "== checkpoint sha256 (asserted on box 2 before the gates and after the last run; each run's measured after training and after the evals)"
  for s in 0 1; do echo "M2b_s$s $R/M2b_s$s/ckpt_last.pt ${CKPT_SHA[$s]}"; done
  echo "== launch-gate token (written by the chain, one per run, after the gate's assertions pass; never by hand)"
  echo '{"kind": "launch-gate-token", "verdict": "PASS", "binding": {"commit": "<the remote tip, 40 hex>", "argv_sha256": "<launch_gate.argv_sha256 of THE TRAINING ARGV (the run line below, after v8f_mini_train.py), not the gate'"'"'s>", "data_sha256": "<the gate selfcheck'"'"'s info.train_sha256, 64 hex>"}, ...}'
  for run in "${RUNS[@]}"; do
    set -- $run; arm=$1; s=$2
    plan "$QF" $arm $s || { echo "ZZV8FMINI-STOP the queue prints no gate / train / eval line for $(rn $arm $s)"; exit 1; }
    echo "== $(rn $arm $s): --ego-perturb-p $(pof $arm), init $R/M2b_s$s/ckpt_last.pt (${CKPT_SHA[$s]}), tag $(rt $arm $s)"
    echo "-- gateorc $arm $s"; echo "$GL"
    echo "-- trainorc $arm $s"; echo "$TL"
    echo "-- evalorc $arm $s"; echo "$EL"
    echo "-- token argv_sha256 (the training argv): $(argvtool sha "$TL")"
    same=$(argvtool same "$GL" "$TL"); echo "-- gate argv = training argv with --out and --selfcheck-only changed: $same"
    [ "$same" = SAME ] || { echo "ZZV8FMINI-STOP the gate line is not the training line with --out and --selfcheck-only changed ($(rn $arm $s))"; exit 1; }
  done
  echo "== not run here: the T1-kin rollouts (dev box CPU) and orc_read.py"
  exit 0
fi
# ---- the predecessor is NAMED, never defaulted (a default log can already hold an end marker): refuse before anything is touched
if [ -z "${AFTER_OUT:-}" ]; then
  echo "ZZV8FMINI-STOP AFTER_OUT is not set: this chain has no default predecessor log (a default can already be satisfied by an OLD log). Set AFTER_OUT to the log of the chain that holds box 2 now."
  exit 1
fi
AFTER_DONE=${AFTER_DONE:-ZZV8FMINI-DONE}; AFTER_STOP=${AFTER_STOP:-ZZV8FMINI-STOP}
[ -d "$(dirname "$ART")" ] || { echo "ZZV8FMINI-STOP $(dirname "$ART") does not exist: the artifacts have nowhere to go"; exit 1; }
B=/c/Users/Admin/box2_ship; L=$B/logs; E=$L/v8f_orc
CK=$E/ckpt_sha256.json
MIRROR=/c/Users/Admin/tanitad-push; BR=agent/arch-inf-20260803
HJ="TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-refav1-hj-mini"
B1="TanitAD Research Lab/Benchmarks & Evals/Research/2026-09-02-b1-eval-lead-block/raw/b1_eval_lead_block.npz"
SSH="ssh -o BatchMode=yes -o ConnectTimeout=20 tanitad-box2"
SCP="scp -q -o BatchMode=yes -o ConnectTimeout=20"
HOLDOUT_SHA=f6a1dee02af0b96f815416bfef9bc11a04554702525135aa75a740942d7750f9
N_CLIPS=4512
M2B_WINDOWS=306566   # B's n_windows_all_clips (banked summary.json); the fine-tune trains on B's window LIST (steps and LR are the fine-tune's, the data is B's)
GATE_POLLS=120; TRAIN_POLLS=1320; EVAL_POLLS=600   # 11 h: one hour above the guard cap, so the GUARD (a committed rule), never the wait, ends a slow run
TAG=$(date '+%m%d%H%M')
CUR=$L/v8forc_current.txt
boot() { $SSH "powershell -NoProfile -Command \"(Get-CimInstance Win32_OperatingSystem).LastBootUpTime.ToString('yyyy-MM-dd HH:mm:ss')\"" 2>/dev/null | tr -d '\r' | grep -E '^[0-9]{4}-' | head -1; }
launch() { for t in 1 2 3; do o=$($SSH "powershell -NoProfile -ExecutionPolicy Bypass -File D:\tanitad\launch.ps1 -Job D:\tanitad\jobs\\$1.cmd -Log D:\tanitad\logs\\$1.log" 2>/dev/null | tr -d '\r') && { echo "$(date '+%m-%d %H:%M') $o"; return 0; }; sleep 20; done; return 1; }
clean() { tr '\r' '\n' < "$L/$1.log"; }
stop() { echo "ZZV8FMINI-STOP $*"; echo "" > $CUR; rm -f $B/BOX2_BUSY.txt; exit 1; }
waitrc() {   # waitrc <job> <polls of 30 s>
  bash $B/waitjob.sh $1 $2 "$BOOT0"; local w=$?
  [ $w = 0 ] || stop "$1 wait=$w (2 = CRASH/reboot, 3 = unreachable, 4 = timeout)"
  local rc=$(clean $1 | grep -a 'ZZJOB-END' | tail -1 | sed -n 's/.* rc=\([-0-9]*\).*/\1/p')
  echo "$(date '+%m-%d %H:%M') $1 rc=$rc"
  [ "$rc" = 0 ] || { clean $1 | grep -a -i -E 'FAIL|refus|error|Traceback' | tail -8; grep ZZGUARD-STOP $L/v8forc_guard.out 2>/dev/null
                     [ -z "${GATE_LOG:-}" ] || { echo "--- the gate's own log on box 2 (tail): $GATE_LOG"; $SSH "powershell -NoProfile -Command \"Get-Content '$GATE_LOG' -Tail 6\"" 2>/dev/null | tr -d '\r'; }
                     stop "$1 rc=$rc"; }
}
queue() {    # queue <job> <polls> <queue args...>: every job name carries this chain's TAG, so an old log's ZZJOB-END can never satisfy a new wait
  local j="v8forc${TAG}_$1" m=$2; shift 2
  $PY "$(cygpath -m $B)/mk_v8f_job.py" $j "$@" >/dev/null && $SCP $B/$j.cmd tanitad-box2:D:/tanitad/jobs/ || stop "ship $j"
  rm -f $L/$j.log
  launch $j || stop "launch $j"
  waitrc $j $m
}
rsha() { $SSH "powershell -NoProfile -Command \"(Get-FileHash -Algorithm $2 -LiteralPath '$1').Hash.ToLower()\"" 2>/dev/null | tr -d '\r' | grep -E '^[0-9a-f]{32,64}$'; }
pull() {     # pull <box-2 path under R> <local dir>
  mkdir -p "$2"; $SCP "tanitad-box2:$R/$1" "$2/" || stop "pull $R/$1"
}
ckset() {    # ckset <key> <sha256>: one measured value into the json
  $PY -c "import json,os,sys;p=sys.argv[1];d=json.load(open(p,encoding='utf-8')) if os.path.exists(p) else {};d[sys.argv[2]]=sys.argv[3];json.dump(d,open(p,'w',encoding='utf-8'),indent=1)" "$(cygpath -m $CK)" "$1" "$2"
}
ckget() {    # ckget <key>
  $PY -c "import json,sys;print(json.load(open(sys.argv[1],encoding='utf-8'))[sys.argv[2]])" "$(cygpath -m $CK)" "$1"
}
ckptsha_b() {   # ckptsha_b <phase>: box 2's two M2b checkpoints ARE the banked runs' (raw/full/MANIFEST.md); the fine-tunes only read them
  local s g
  for s in 0 1; do
    g=$(rsha "$R/M2b_s$s/ckpt_last.pt" SHA256); [ -n "$g" ] || stop "$1: cannot hash box 2's M2b_s$s checkpoint"
    [ "$g" = "${CKPT_SHA[$s]}" ] || stop "$1: box 2's M2b_s$s checkpoint sha256 $g != the banked run's ${CKPT_SHA[$s]}"
    ckset "M2b_s$s" "$g"
  done
  echo "$(date '+%m-%d %H:%M') $1: both M2b checkpoints are the banked runs' (sha256 asserted)"
}
ckptsha_x() {   # ckptsha_x <arm> <seed> <trained|after_evals>: the fine-tuned checkpoint, measured on box 2; after the evals it must be the value measured after training (the scored checkpoint IS the trained one)
  local g h h0; g=$(rn $1 $2)
  h=$(rsha "$R/$g/ckpt_last.pt" SHA256); [ -n "$h" ] || stop "cannot hash box 2's $g checkpoint ($3)"
  if [ "$3" = after_evals ]; then
    h0=$(ckget "$(rt $1 $2)_trained"); [ "$h" = "$h0" ] || stop "$g's checkpoint changed during the evals: $h != $h0"
  fi
  ckset "$(rt $1 $2)_$3" "$h"
  echo "$(date '+%m-%d %H:%M') $g checkpoint sha256 ($3) ${h:0:16}..."
}
finish() {   # finish <summary>: the committed end of the chain
  $SSH "powershell -NoProfile -Command \"New-Item -ItemType File -Force D:/tanitad/logs/telemetry.stop | Out-Null\"" 2>/dev/null
  $SCP tanitad-box2:D:/tanitad/logs/telemetry_full.csv $E/telemetry_v8forc.csv 2>/dev/null
  echo "" > $CUR
  rm -f $B/BOX2_BUSY.txt
  echo "ZZV8FMINI-DONE $(date '+%m-%d %H:%M') tip $T: $1 (boot unchanged $BOOT0)"
  exit 0
}
gate_one() {   # gate_one <arm> <seed>: the run's in-process launch gate on box 2, pulled and asserted HERE, then its launch-gate token (written by this function, never by hand)
  local arm=$1 s=$2 g gd P SC
  g=$(rn $arm $s); gd=$(rg $arm $s); P=$(pof $arm); SC="$ART/$g/gate/selfcheck.json"
  plan "$QC" $arm $s || stop "the tip's queue prints no gate / train / eval line for $g"
  mkdir -p "$ART/$g" "$E/lines"
  printf '%s\n' "$GL" > "$E/lines/$g.gate.txt"; printf '%s\n' "$TL" > "$E/lines/$g.train.txt"; printf '%s\n' "$EL" > "$E/lines/$g.eval.txt"
  GATE_LOG="$R/$gd.log"; queue "gate_$g" $GATE_POLLS gateorc $arm $s; GATE_LOG=      # a refused gate says WHY in the chain's own log (its stdout / stderr went to that file on box 2)
  pull "$gd/selfcheck.json" "$ART/$g/gate"
  $SCP "tanitad-box2:$R/$gd.log" "$ART/$g/gate/" 2>/dev/null
  $PY - "$(cygpath -m $SC)" $HOLDOUT_SHA "$(cygpath -m $E/runs/M2b_s$s/summary.json)" "$(cygpath -m $E/lines/$g.gate.txt)" "$(cygpath -m $E/lines/$g.train.txt)" $arm $s $P ${CKPT_SHA[$s]} $ORC_STEPS $M2B_WINDOWS $N_CLIPS <<'EOF' || stop "gate ORC assertions $g"
import json, re, sys
scp, hold, bsum, glf, tlf, arm, seed, p, banked, steps, windows, nclips = sys.argv[1:13]
seed, p, steps, windows, nclips = int(seed), float(p), int(steps), int(windows), int(nclips)
sc, b = json.load(open(scp, encoding="utf-8")), json.load(open(bsum, encoding="utf-8"))
i, lv, bl = sc["info"], sc["levers"], b["levers"]
bad = []


def argv_of(path):
    t = [x.strip('"') for x in re.findall(r'"[^"]*"|\S+', open(path, encoding="utf-8").read().strip())]
    return t[next(k for k, x in enumerate(t) if x.lower().endswith("v8f_mini_train.py")) + 1:]


def flags(a):
    o, k = {}, 0
    while k < len(a):
        if a[k].startswith("--"):
            j = k + 1
            while j < len(a) and not a[j].startswith("--"):
                j += 1
            o.setdefault(a[k], []).append(a[k + 1:j])
            k = j
        else:
            k += 1
    return o


# THE PASS TOKEN's ingredients: the trainer's in-process gate wrote these; without all of them there is no token
if sc.get("declared_vs_built") != "PASS": bad.append("declared_vs_built")
live = sc["live"]
if not live.get("pass") or live.get("control_prior_grad_sum") != 0.0: bad.append(f"live {live.get('pass')} / control {live.get('control_prior_grad_sum')}")
if live.get("dead_params") != [] or live.get("declared_dead") != []: bad.append(f"G-LIVE dead set {live.get('dead_params')} (declared {live.get('declared_dead')}) != []: the fine-tune kills no weight")
# G-LIVE ON A PERTURBED BATCH (ORC) / the identity (ORC-0): the perturbation REACHES the loss the trainer backpropagates, and every control reads EXACTLY 0.0
pr = sc.get("ego_perturb_probe")
if not isinstance(pr, dict) or pr.get("pass") is not True:
    bad.append(f"ego_perturb_probe {pr}")
elif p > 0.0:
    if pr.get("ego_perturb") == "off" or pr.get("p_declared") != p: bad.append(f"the ego probe ran for p {pr.get('p_declared')!r} ({pr.get('ego_perturb')!r}), the run's is {p}")
    if not (isinstance(pr.get("n_selected_at_p1"), int) and pr["n_selected_at_p1"] > 0): bad.append(f"the ego probe saw no eligible window ({pr.get('n_selected_at_p1')})")
    if not (pr.get("loss_diff", 0.0) != 0.0 and pr.get("grad_diff_all", 0.0) > 0.0): bad.append(f"the perturbed batch moves the loss by {pr.get('loss_diff')} and the gradient by {pr.get('grad_diff_all')}: the perturbation does not reach the loss")
    for k in ("control_repeat_loss_diff", "control_repeat_grad_diff", "control_p0_loss_diff", "control_p0_grad_diff", "control_prior_grad_diff"):
        if pr.get(k) != 0.0: bad.append(f"ego probe control {k} = {pr.get(k)!r} (must be exactly 0.0)")
    if pr.get("control_p0_same_object") is not True: bad.append("ego probe: p = 0 is not the identity (the batch object changed)")
    if pr.get("control_wm_grad_diff") not in (None, 0.0): bad.append("the world-model auxiliary's gradient moved with the perturbation")
    lp = pr.get("live_perturbed") or {}
    if lp.get("pass") is not True or lp.get("control_prior_grad_sum") != 0.0: bad.append(f"G-LIVE on the PERTURBED batch: pass {lp.get('pass')} / prior control {lp.get('control_prior_grad_sum')}")
elif pr.get("ego_perturb") != "off":
    bad.append(f"ORC-0 (p = 0): the ego probe should read 'off' (nothing is applied), it read {pr.get('ego_perturb')!r}")
# THE LEVER, from the BUILT perturber: every value the PREREG fixes
ep = lv.get("ego_perturb") or {}
# PREREG Amendment 3: ORC names the five law flags; ORC-0 names NONE of them (the trainer's gate refuses a law flag set while p is 0), so its BUILT values are the trainer's defaults. vmax's default 8.0 is ORC's: the eligible denominator is the same.
LAW_ORC = {"ego_perturb_vscale": [0.5, 1.0], "ego_perturb_vmax": 8.0, "ego_perturb_adrop": 1.5, "ego_perturb_yawrate_std": 2.0, "ego_perturb_seed": seed}
LAW_DEFAULT = {"ego_perturb_vscale": [0.5, 1.0], "ego_perturb_vmax": 8.0, "ego_perturb_adrop": 1.5, "ego_perturb_yawrate_std": 0.0, "ego_perturb_seed": 0}
for k, w in {"ego_perturb_p": p, **(LAW_ORC if p > 0.0 else LAW_DEFAULT), "kin_mode": "yaw"}.items():
    if ep.get(k) != w: bad.append(f"built ego_perturb {k} {ep.get(k)!r} != {w!r}")
# THE START STATE: the BANKED M2b checkpoint of this seed, hashed and read back (0 tensors differ after the strict load)
ic = lv.get("init_ckpt") or {}
if ic.get("sha256") != banked: bad.append(f"the fine-tune does not start from the banked M2b_s{seed} checkpoint: init sha256 {str(ic.get('sha256'))[:16]} != {banked[:16]}")
if i.get("init_ckpt") != ic: bad.append("info.init_ckpt and levers.init_ckpt disagree")
if ic.get("applied") is not True or ic.get("n_mismatch_after_load") != 0 or not ic.get("n_tensors"): bad.append(f"init_ckpt not applied / read back: applied {ic.get('applied')} mismatches {ic.get('n_mismatch_after_load')} tensors {ic.get('n_tensors')}")
if not str(ic.get("path", "")).replace("\\", "/").endswith(f"/M2b_s{seed}/ckpt_last.pt"): bad.append(f"init_ckpt path {ic.get('path')!r} is not .../M2b_s{seed}/ckpt_last.pt")
# THE FINE-TUNE'S SCHEDULE (Amendment 1): 3,000 steps, lr 1e-4 with warmup 100, teacher forcing held at M2b's final 0.2
if sc.get("steps_total") != steps: bad.append(f"steps {sc.get('steps_total')} != the PREREG's {steps}")
sch = lv.get("lr_schedule") or {}
if (sch.get("lr"), sch.get("warmup"), sch.get("total")) != (1e-4, 100, steps): bad.append(f"built lr schedule {sch} != lr 1e-4 / warmup 100 / total {steps}")
if (lv.get("p_gt_start"), lv.get("p_gt_end")) != (0.2, 0.2): bad.append(f"built p_gt {lv.get('p_gt_start')} -> {lv.get('p_gt_end')}, expected 0.2 -> 0.2")
# B's DATA AND MODEL are the fine-tune's: clip list, holdout, window set, levers, weights, parameter count
if (i["n_clips"], i["n_excluded"], i["n_excluded_declared"], i["excluded_in_train"]) != (nclips, 60, 60, 0): bad.append("clips / holdout counts")
if i["excluded_sha256"] != hold: bad.append("the excluded set is not the RA-1 holdout")
if i["train_sha256"] != b["train_sha256"]: bad.append(f"TRAIN list {i['train_sha256'][:12]} != B's {b['train_sha256'][:12]}")
if i["kin_mode"] != "yaw": bad.append("kin_mode")
if (lv["wm"], lv["navonly"], lv["consistency"], lv.get("cons_absent")) != ("b", False, True, "mask"): bad.append(f"levers wm / navonly / consistency / cons_absent {(lv['wm'], lv['navonly'], lv['consistency'], lv.get('cons_absent'))}")
if lv.get("o_interface", "full") != "full": bad.append(f"built o_interface {lv.get('o_interface')!r} != 'full' (B's)")
if lv.get("code_mix", "whole") != "whole": bad.append(f"built code_mix {lv.get('code_mix')!r} != 'whole' (B's)")
if lv.get("w_acc", 0.0) != 0.0: bad.append(f"built w_acc {lv.get('w_acc')!r} != 0.0 (B's)")
wd = lv.get("w") or {}
for k in ("w_pen", "w_pen_self", "w_wm"):
    if wd.get(k) != bl["w"][k]: bad.append(f"{k} {wd.get(k)} != B's {bl['w'][k]}")
if (wd.get("w_tac"), wd.get("w_cons")) != (1.0, 0.5): bad.append(f"w_tac / w_cons {wd.get('w_tac')} / {wd.get('w_cons')} != B's 1.0 / 0.5 (a second lever moved)")
if i.get("n_windows_all_clips") != b.get("n_windows"): bad.append(f"the window set is not B's: n_windows_all_clips {i.get('n_windows_all_clips')} != B's {b.get('n_windows')}")
if i.get("n_windows_all_clips") != windows: bad.append(f"n_windows_all_clips {i.get('n_windows_all_clips')} != the PREREG's {windows}")
if lv.get("hist") != 4: bad.append(f"built hist {lv.get('hist')} != 4 (B's)")
if lv.get("hmax") != 16: bad.append(f"built hmax {lv.get('hmax')} != 16 (the arm's auto value, B's)")
if lv.get("window_start") != 3: bad.append(f"built window_start {lv.get('window_start')} != 3 (B's effective first window, hist - 1)")
if (i.get("window_start"), i.get("window_t_min"), i.get("ds_window_start")) != (3, 3, 3): bad.append(f"window start (asked / first t of the built window list / dataset clamp origin) {(i.get('window_start'), i.get('window_t_min'), i.get('ds_window_start'))} != (3, 3, 3)")
pg, pb = lv["params"], bl["params"]
if lv["n_params_trainable"] != bl["n_params_trainable"]: bad.append(f"n_params_trainable {lv['n_params_trainable']} != B's {bl['n_params_trainable']}")
for k in sorted(set(pg) | set(pb)):
    if pg.get(k, 0) != pb.get(k, 0): bad.append(f"parameter group {k}: {pg.get(k)} vs B's {pb.get(k)}")
# THE QUEUE'S OWN PRINTED LINES: the registered fine-tune, every lever NAMED exactly once with its registered value (the flags' defaults disable the lever, so a default is not a test of it), and nothing else moved
gl, tl = argv_of(glf), argv_of(tlf)
want = {"--arm": ["M2b"], "--seed": [str(seed)], "--grid": ["pool4x10"], "--steps": [str(steps)], "--lr": ["1e-4"], "--warmup": ["100"], "--p-gt-start": ["0.2"], "--p-gt-end": ["0.2"], "--ego-perturb-p": [sys.argv[8]]}
LAWF = {"--ego-perturb-vscale": ["0.5", "1.0"], "--ego-perturb-vmax": ["8.0"], "--ego-perturb-adrop": ["1.5"], "--ego-perturb-yawrate-std": ["2.0"], "--ego-perturb-seed": [str(seed)]}
if p > 0.0: want.update(LAWF)                                  # ORC names all five law flags; ORC-0 (Amendment 3) names none
for name, line in (("train", tl), ("gate", gl)):
    fl = flags(line)
    if p == 0.0:
        for k in LAWF:
            if k in fl: bad.append(f"the queue's {name} line names {k} on ORC-0: the trainer's gate refuses a law flag set while p is 0 (PREREG Amendment 3: ORC-0 names ONLY --ego-perturb-p 0)")
    for k, w in want.items():
        if fl.get(k) != [w]: bad.append(f"the queue's {name} line does not carry {k} {' '.join(w)} exactly once (it has {fl.get(k)})")
    ik = fl.get("--init-ckpt") or []
    if len(ik) != 1 or len(ik[0]) != 1 or not ik[0][0].replace("\\", "/").endswith(f"/M2b_s{seed}/ckpt_last.pt"): bad.append(f"the queue's {name} line --init-ckpt {ik} is not .../M2b_s{seed}/ckpt_last.pt exactly once")
    for fx in ("--hist", "--hmax", "--window-start", "--w-tac", "--w-cons", "--w-traj", "--w-pen", "--w-pen-self", "--w-wm", "--w-acc", "--code-mix", "--o-interface", "--data-frac", "--max-clips", "--cons-absent", "--wm", "--navonly",
               "--no-consistency", "--detach-upper", "--no-resume"):
        if fx in fl: bad.append(f"the queue's {name} line sets {fx}: a second lever beside the fine-tune")
if "--selfcheck-only" not in gl or gl.count("--selfcheck-only") != 1: bad.append("the gate line does not carry --selfcheck-only exactly once")
if "--selfcheck-only" in tl: bad.append("the TRAIN line carries --selfcheck-only")
g2, t2 = [x for k, x in enumerate(gl) if x != "--selfcheck-only"], tl
def drop_out(x):
    o, k = [], 0
    while k < len(x):
        if x[k] == "--out": k += 2
        else: o.append(x[k]); k += 1
    return o
if drop_out(g2) != drop_out(t2): bad.append("the gate argv is not the training argv with --out and --selfcheck-only changed")
if not i.get("lru"): bad.append("info.lru is not recorded in the selfcheck (the pipeline sizing must be on the record)")
print("ORC gate:", "PASS" if not bad else bad, "|", arm, "s%d" % seed, "| p", p, "| clips", i["n_clips"], "steps", sc.get("steps_total"), "windows", i.get("n_windows_all_clips"), "| init", str(ic.get("sha256"))[:12], "| probe grad diff", (pr or {}).get("grad_diff_all"),
      "| params", lv["n_params_trainable"], "(B", bl["n_params_trainable"], ") | lru", i.get("lru"))
sys.exit(1 if bad else 0)
EOF
  # the token record: the exact commit, the sha256 of the TRAINING argv (what argv.json will record: the reader compares it), the data manifest the gate read, the sha256 of the queue and of the gate's selfcheck, the key files box 2 was checked
  # against (written by the chain, never by hand). It exists only after the assertions above passed.
  $PY - "$(cygpath -m $SC)" "$(cygpath -m $ART/$g/launch_gate_token.json)" "$T" "$(sha256sum $QT | cut -c1-64)" "{${KEYSHA%, }}" "$(cygpath -m $E/lines/$g.gate.txt)" "$(cygpath -m $E/lines/$g.train.txt)" $g <<'EOF' || stop "write the gate token $g"
import hashlib, json, re, sys
scp, out, commit, qsha, keys, glf, tlf, run = sys.argv[1:9]
sc = json.load(open(scp, encoding="utf-8"))


def argv_of(path):
    t = [x.strip('"') for x in re.findall(r'"[^"]*"|\S+', open(path, encoding="utf-8").read().strip())]
    return t[next(k for k, x in enumerate(t) if x.lower().endswith("v8f_mini_train.py")) + 1:]


def gate_argv_sha256(argv):                    # launch_gate.argv_sha256: sha256 of the ORDERED json array (orc_read.gate_argv_sha256 is the same function; a test pins the equality)
    return hashlib.sha256(json.dumps(list(argv), ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


g_argv, t_argv = argv_of(glf), argv_of(tlf)
bad = []
if sc.get("declared_vs_built") != "PASS": bad.append("the gate's selfcheck is not a PASS")
if sc.get("argv_sha") != hashlib.sha256(" ".join(g_argv).encode()).hexdigest()[:16]: bad.append("the gate ran an argv other than the queue's printed gate line (the selfcheck's argv_sha differs)")
if not re.fullmatch(r"[0-9a-f]{40}", commit): bad.append(f"commit {commit!r} is not 40-hex")
data = sc["info"]["train_sha256"]
if not re.fullmatch(r"[0-9a-f]{64}", data): bad.append("data_sha256 is not 64-hex")
if "--selfcheck-only" in t_argv: bad.append("the TRAINING argv carries --selfcheck-only")
if bad:
    print("no token:", bad)
    sys.exit(1)
tok = {"kind": "launch-gate-token", "verdict": "PASS", "prereg": "PREREG_V8F_ORECOVER", "run": run,
       "binding": {"commit": commit, "argv_sha256": gate_argv_sha256(t_argv), "data_sha256": data},
       "binding_note": "argv_sha256 = launch_gate.argv_sha256 of the TRAINING argv (v8f_mini_train.py's sys.argv[1:], which argv.json will record); the in-process gate ran the same argv with --out and --selfcheck-only changed",
       "gate": {"kind": "v8f_mini_train.py --selfcheck-only (in-process: G-DVB FLAG_CHECKS + G-LIVE + ego_perturb_probe)", "argv_sha256": gate_argv_sha256(g_argv), "argv_sha16": sc.get("argv_sha"),
                "selfcheck_sha256": hashlib.sha256(open(scp, "rb").read()).hexdigest()},
       "queue_sha256": qsha, "key_files_sha256": json.loads(keys), "excluded_sha256": sc["info"]["excluded_sha256"], "init_ckpt_sha256": sc["levers"]["init_ckpt"]["sha256"],
       "v9_md5": {"v9_labels_train.npz": "83e30bd3791f0767707eade5f4671759", "v9_labels_eval139.npz": "6b5c7f207cffc3b7eb3cd527fd433599"},
       "hmac": "none: written by the box-2 chain, not by launch_gate.py (its key lives on the launch host); orc_read.py checks the binding and does not verify an HMAC"}
json.dump(tok, open(out, "w", encoding="utf-8"), indent=1)
print("gate token written for", run, "commit", commit, "argv_sha256", tok["binding"]["argv_sha256"][:16])
EOF
}
runone() {   # runone <arm> <seed>: train, eval, pull, record and print the checkpoint sha256, assert the run's record; the perturbation-reached control is PRINTED, not enforced
  local arm=$1 s=$2 g tg f x l
  g=$(rn $arm $s); tg=$(rt $arm $s)
  echo "$g" > $CUR
  queue "t_$g" $TRAIN_POLLS trainorc $arm $s
  echo "" > $CUR
  ckptsha_x $arm $s trained
  queue "e_$g" $EVAL_POLLS evalorc $arm $s
  ckptsha_x $arm $s after_evals
  for f in selfcheck.json summary.json argv.json train_log.jsonl train.log ckpt_last.pt; do pull "$g/$f" "$ART/$g"; done
  for x in json npz log; do pull "eval/eval_$tg.$x" "$ART/$g"; done
  l=$(sha256sum "$ART/$g/ckpt_last.pt" | cut -c1-64); [ "$l" = "$(ckget "${tg}_after_evals")" ] || stop "$g: the pulled checkpoint's sha256 $l != the value measured on box 2"
  ( cd "$ART/$g" && sha256sum ckpt_last.pt "eval_$tg.json" "eval_$tg.npz" selfcheck.json summary.json argv.json train_log.jsonl launch_gate_token.json | tee SHA256SUMS.txt )
  echo "$(date '+%m-%d %H:%M') $g pulled to $ART/$g (checkpoint sha256 $l)"
  $PY - "$(cygpath -m $ART/$g)" "$(cygpath -m $E/lines/$g.train.txt)" $arm $s $(pof $arm) $ORC_STEPS ${CKPT_SHA[$s]} <<'EOF' || stop "run ORC record $g"
import hashlib, json, os, re, sys
d, tlf, arm, seed, p, steps, banked = sys.argv[1:8]
seed, p, steps = int(seed), float(p), int(steps)
J = lambda f: json.load(open(os.path.join(d, f), encoding="utf-8"))
t = [x.strip('"') for x in re.findall(r'"[^"]*"|\S+', open(tlf, encoding="utf-8").read().strip())]
planned = t[next(k for k, x in enumerate(t) if x.lower().endswith("v8f_mini_train.py")) + 1:]
aj, sm, sc = J("argv.json"), J("summary.json"), J("selfcheck.json")
bad = []
if aj.get("argv") != planned:
    got = aj.get("argv") or []
    k = next((k for k in range(min(len(got), len(planned))) if got[k] != planned[k]), min(len(got), len(planned)))
    bad.append(f"argv.json is not the planned training argv (first difference at token {k}: {got[k:k + 2]} vs {planned[k:k + 2]}): the token binds an argv that did not run")
sha16 = hashlib.sha256(" ".join(planned).encode()).hexdigest()[:16]
for name, v in (("argv.json sha", aj.get("sha")), ("summary.json argv_sha", sm.get("argv_sha")), ("selfcheck.json argv_sha", sc.get("argv_sha"))):
    if v != sha16: bad.append(f"{name} {v} != the planned argv's {sha16}")
if sm.get("done") is not True: bad.append("summary.json done is not true")
if sm.get("steps") != steps: bad.append(f"summary.json steps {sm.get('steps')} != {steps}")
if (sm.get("arm"), sm.get("seed")) != ("M2b", seed): bad.append(f"summary.json arm / seed {(sm.get('arm'), sm.get('seed'))} != ('M2b', {seed})")
if sc.get("declared_vs_built") != "PASS": bad.append("the run's own selfcheck.json is not a PASS")
lv = sm.get("levers") or {}
if (lv.get("init_ckpt") or {}).get("sha256") != banked: bad.append(f"summary.json levers.init_ckpt.sha256 {(lv.get('init_ckpt') or {}).get('sha256')} != the banked M2b_s{seed} checkpoint's")
if (lv.get("ego_perturb") or {}).get("ego_perturb_p") != p: bad.append(f"summary.json built ego_perturb_p {(lv.get('ego_perturb') or {}).get('ego_perturb_p')} != {p}")
print("ORC run record:", "PASS" if not bad else bad, "|", arm, "s%d" % seed, "| argv sha16", sha16, "| steps", sm.get("steps"))
sys.exit(1 if bad else 0)
EOF
  $PY - "$(cygpath -m $ART/$g/summary.json)" $arm $s <<'EOF' || echo "ZZORC-CONTROL-FAIL $g: the perturbation did not reach training as registered (the reader will REFUSE this run; the chain goes on)"
import json, sys
sm, arm, seed = json.load(open(sys.argv[1], encoding="utf-8")), sys.argv[2], sys.argv[3]
n, e = sm.get("ego_perturb_n"), sm.get("ego_perturb_eligible")
if arm == "orc0":                                              # ORC-0: exactly 0 perturbed
    ok = isinstance(n, int) and n == 0
    print("ORC perturbation-reached:", "PASS" if ok else "FAIL", f"| orc0 s{seed} | ego_perturb_n {n} (must be exactly 0) | eligible {e}")
else:                                                          # ORC: n / eligible in 0.3 +- 0.02 (PREREG section 3)
    ok = isinstance(n, int) and isinstance(e, int) and e > 0 and 0 <= n <= e and 0.3 - 0.02 - 1e-12 <= n / e <= 0.3 + 0.02 + 1e-12
    print("ORC perturbation-reached:", "PASS" if ok else "FAIL", f"| orc s{seed} | ego_perturb_n {n} / eligible {e} = {n / e if ok else 'n/a'} (must lie in 0.3 +- 0.02)")
sys.exit(0 if ok else 1)
EOF
  grep ZZGUARD-STOP $L/v8forc_guard.out 2>/dev/null && stop "guard stop"
}
echo "$(date '+%m-%d %H:%M') waiting for the predecessor chain ($AFTER_OUT: end marker ^$AFTER_DONE, stop marker ^$AFTER_STOP; log present: $([ -f "$AFTER_OUT" ] && echo yes || echo NO))"
until grep -q "^$AFTER_DONE" "$AFTER_OUT" 2>/dev/null; do
  grep -q "^$AFTER_STOP" "$AFTER_OUT" 2>/dev/null && { echo "ZZV8FMINI-STOP the predecessor stopped: $(grep "^$AFTER_STOP" "$AFTER_OUT" | tail -1)"; exit 1; }
  sleep 300
done
[ -f $B/BOX2_BUSY.txt ] && { echo "ZZV8FMINI-STOP busy marker present: $(cat $B/BOX2_BUSY.txt)"; exit 1; }
# the marker is taken ATOMICALLY (noclobber): two chains that wait on the same predecessor cannot both pass the test above and both write it
( set -C; echo "v8F-mini PREREG_V8F_ORECOVER (ORC / ORC-0 = M2b fine-tuned 3,000 steps with / without the ego perturbation; ORC s0, ORC-0 s0, ORC s1, ORC-0 s1; four gates first) since $(date '+%m-%d %H:%M'), ~0.7 h per run plus eval. Do NOT launch box 2 jobs while this file exists." > $B/BOX2_BUSY.txt ) 2>/dev/null \
  || { echo "ZZV8FMINI-STOP busy marker taken while arming: $(cat $B/BOX2_BUSY.txt 2>/dev/null)"; exit 1; }
BOOT0=$(boot); [ -n "$BOOT0" ] || stop "box 2 unreachable"
echo "$(date '+%m-%d %H:%M') boot0=$BOOT0"
mkdir -p $E/runs $E/lines      # only now: a chain that refused or stopped while waiting has touched nothing
# ---- 1. code refresh from the remote tip
git -C $MIRROR fetch -q origin $BR; T=$(git -C $MIRROR ls-remote origin $BR | cut -f1)
git -C $MIRROR cat-file -e "$T^{commit}" || stop "remote tip $T is not in the mirror"
QT=$B/v8forc_queue_$TAG.cmd; git -C $MIRROR show "$T:$PK/code/v8f_mini_box2_queue.cmd" | tr -d '\r' > $QT
QC=$B/v8forc_queue_$TAG.crlf.cmd; git -C $MIRROR show "$T:$PK/code/v8f_mini_box2_queue.cmd" > $QC      # the committed bytes (CRLF): what cmd.exe can run for the dry-run lines
for st in gateorc trainorc evalorc orcarm; do grep -q "^:$st\$" $QT || stop "the tip's queue has no $st stage"; done
git -C $MIRROR cat-file -e "$T:$PK/raw/2026-10-26-orcread/orc_read.py" || stop "the tip has no orc_read.py (the reader of this run's outputs)"
git -C $MIRROR show "$T:stack/scripts/v8f_mini_train.py" | grep -q -- '"--init-ckpt"' || stop "the tip's trainer has no --init-ckpt flag"
git -C $MIRROR show "$T:stack/scripts/v8f_mini_train.py" | grep -q -- '"--ego-perturb-p"' || stop "the tip's trainer has no --ego-perturb-p flag"
git -C $MIRROR show "$T:stack/scripts/v8f_mini_train.py" | grep -q '"ego_perturb_p": "ego:ego_perturb_p"' || stop "the tip's trainer has no G-DVB entry (FLAG_CHECKS) for --ego-perturb-p"
git -C $MIRROR show "$T:stack/scripts/v8f_mini_train.py" | grep -q '"init_ckpt": "data:init_ckpt"' || stop "the tip's trainer has no G-DVB entry (FLAG_CHECKS) for --init-ckpt"
git -C $MIRROR show "$T:stack/scripts/v8f_mini_train.py" | grep -q 'ego_perturb_probe(model, probe, cfg, ego)' || stop "the tip's trainer does not run the ego perturbation probe in its in-process gate"
git -C $MIRROR show "$T:stack/scripts/v8f_mini_train.py" | grep -q 'b = ego.apply(b, step)' || stop "the tip's trainer does not apply the perturbation in its loop"
git -C $MIRROR show "$T:stack/scripts/v8f_mini_train.py" | grep -q '"ego_perturb_n": ego_tot\["n"\]' || stop "the tip's trainer does not write the run totals ego_perturb_n / ego_perturb_eligible"
git -C $MIRROR show "$T:stack/scripts/v8f_mini_train.py" | grep -q '^class LRSchedule' || stop "the tip's trainer has no LRSchedule (--warmup is not read back by the gate)"
git -C $MIRROR show "$T:stack/tanitad/data/v8f_mini_data.py" | grep -q '^class EgoPerturb' || stop "the tip's data module has no EgoPerturb"
git -C $MIRROR cat-file -e "$T:$B1" || stop "the tip has no B1 lead block"
git -C $MIRROR -c core.autocrlf=false archive --format=tar -o $B/repo_v8forc_tip.tar $T -- stack taniteval "$HJ" "$PK" "$B1" || stop "archive"
$SCP $B/repo_v8forc_tip.tar tanitad-box2:D:/tanitad/repo_v8forc_tip.tar || stop "ship the code archive"
printf '@echo off\r\necho ===== refresh %s %%DATE%% %%TIME%%\r\ntar -xf D:\\tanitad\\repo_v8forc_tip.tar -C D:\\tanitad\\repo\r\nset RC=%%ERRORLEVEL%%\r\necho %s> D:\\tanitad\\repo\\TIP_V8FORC.txt\r\necho ===== ZZJOB-END %%DATE%% %%TIME%% rc=%%RC%%\r\n' "$T" "$T" > $B/v8forc${TAG}_refresh.cmd
$SCP $B/v8forc${TAG}_refresh.cmd tanitad-box2:D:/tanitad/jobs/ || stop "ship refresh job"
launch v8forc${TAG}_refresh || stop "launch refresh"
waitrc v8forc${TAG}_refresh 40
KEYSHA=""
for f in stack/scripts/v8f_mini_train.py stack/scripts/v8f_mini_eval.py stack/tanitad/models/v8f_mini.py stack/tanitad/data/v8f_mini_data.py \
         stack/tanitad/models/hjepa.py "$PK/code/v8f_mini_box2_queue.cmd" "$HJ/raw/p10_yaw4512/p10_prior.pt" "$HJ/raw/p10_yaw4512/p10_eval_pred6.npy"; do
  l=$(git -C $MIRROR show "$T:$f" | sha256sum | cut -c1-64)
  [ "$l" = "$(rsha "D:/tanitad/repo/$f" SHA256)" ] || stop "box 2 $f != tip $T"
  KEYSHA="$KEYSHA\"$f\": \"$l\", "
done
for kv in v9_labels_train.npz=83e30bd3791f0767707eade5f4671759 v9_labels_eval139.npz=6b5c7f207cffc3b7eb3cd527fd433599; do
  [ "${kv#*=}" = "$(rsha "D:/tanitad/data/v9/${kv%%=*}" MD5)" ] || stop "box 2 v9 ${kv%%=*} md5 != ${kv#*=}"
done
echo "$(date '+%m-%d %H:%M') box 2 code = tip $T (8 key files sha-checked; both v9 releases md5-checked)"
# ---- 2. the base is fixed (the PREREG): the fine-tunes start from the banked checkpoints (box 2 is idle: a different file costs nothing)
rm -f $CK
ckptsha_b "before the gates"
# ---- 3. the B side (summary.json of the banked M2b runs: clip list, window set, parameter count), staged BEFORE any gate (a missing file stops the chain while box 2 is still idle)
for s in 0 1; do
  mkdir -p $E/runs/M2b_s$s
  cp "$L/v8f_mini/runs/M2b_s$s/summary.json" $E/runs/M2b_s$s/ || stop "stage the banked M2b_s$s/summary.json"
done
$SSH "powershell -NoProfile -Command \"if (Test-Path D:/tanitad/logs/telemetry.stop) { Remove-Item D:/tanitad/logs/telemetry.stop }\"" 2>/dev/null
launch telemetry_long || echo "$(date '+%m-%d %H:%M') telemetry did not launch"
tl() { $SSH "powershell -NoProfile -Command \"Get-Content D:/tanitad/logs/telemetry_full.csv -Tail 1\"" 2>/dev/null | tr -d '\r'; }
sleep 45; t1=$(tl); sleep 35; t2=$(tl)
[[ $t2 == 20* && $t2 != "$t1" ]] || stop "no fresh telemetry ($t1 / $t2): the temperature guard would be blind"
grep -qF "'^(ZZV8FMINI|" $B/v8f_guard.sh || stop "v8f_guard.sh does not know this chain's markers (anchored): it would never exit"
echo "" > $CUR
V8F_CHAIN_OUT=$L/chain_v8f_orc.out V8F_CURRENT=$CUR bash $B/v8f_guard.sh > $L/v8forc_guard.out 2>&1 &
# ---- 4. ALL FOUR GATES, each asserted and tokened, before ANY training: a gate that fails for the fourth run is found before the first GPU-hour is spent
for run in "${RUNS[@]}"; do set -- $run; gate_one $1 $2; done
# ---- 5. the four runs, in the PREREG's order; nothing is conditional on a read
for run in "${RUNS[@]}"; do set -- $run; runone $1 $2; done
ckptsha_b "after the last run"
echo "$(date '+%m-%d %H:%M') summary: $(for run in "${RUNS[@]}"; do set -- $run; echo -n "$(rn $1 $2)=$(ckget "$(rt $1 $2)_after_evals" | cut -c1-12) "; done)"
finish "PREREG_V8F_ORECOVER: ORC / ORC-0 x seeds 0 and 1 trained (3,000 steps each), T0-evaluated and pulled to $ART (sha256 in each SHA256SUMS.txt); the T1-kin rollouts and orc_read.py run on the dev box (NOT here)"
