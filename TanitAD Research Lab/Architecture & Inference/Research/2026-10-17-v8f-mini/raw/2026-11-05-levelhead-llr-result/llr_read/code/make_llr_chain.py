#!/usr/bin/env python3
"""make_llr_chain.py -- builds llr_read_chain.sh and llr_read_chain.cmd from the CLR chain (clr_read/clr_read_chain.sh, sha256 pinned): CLR -> LLR and clr_ -> llr_ renamed everywhere, then exact-once edits
(a replacement that does not match exactly once aborts).  The differences: TEN rollouts (no attribution-base rollout: the base is the banked CLR arrays), the LLR read tree, KA-NEW2, the LLR pre-check, the 16 reused
arrays, three pre-flights of LLR checkpoints, and the token-wiring O-sensitivity read-out.   usage: python -I make_llr_chain.py [--check]"""
import hashlib
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
PK = os.path.dirname(HERE)
SRC = "E:/Projects/TanitAD-artifacts/dagland/motion_c/clr_read/clr_read_chain.sh"
SRC_CMD = "E:/Projects/TanitAD-artifacts/dagland/motion_c/clr_read/clr_read_chain.cmd"
SRC_SHA = open(f"{HERE}/chain_src.sha", encoding="utf-8").read().split()[0]
CMD_SHA = open(f"{HERE}/chain_src.sha", encoding="utf-8").read().split()[1]

HEADER = '''# LLR READ chain (PREREG_LEVELHEAD_A2.md, Reads and Outcomes + A2_RULING_1.md + FROZEN_LLR.txt): the two LLR checkpoints (CLR's recipe + the lead head, token wiring, trained on box 2) over ALL 2,399 eval windows,
# then the reader, then the O-sensitivity read-out adapted to the token wiring.  Dev-box CPU, ZERO training, no ssh, ONE rollout at a time.  Cloned from clr_read/clr_read_chain.sh.
# NOT launched by the package that wrote it: the Master Mind starts it AFTER the LLR checkpoints are pulled into E:/Projects/TanitAD-artifacts/v8f-recover/ (M2b_s{0,1}_recft_LLR/, llr_markers/).
#   usage:  llr_read_chain.sh --dry-run     lists the 10 rollouts, the gates and what is present; runs nothing, writes no marker
#           llr_read_chain.sh               (normal; start through llr_read_chain.cmd = WMI Win32_Process.Create so it survives the caller)
# READ TREE: F:/TanitAD-scratch/motion_c/llr_read/tree = the CL read tree with the 83c9ee0 lead-head model / data / trainer (build_llr_read_tree.py); the CL tree's model cannot build an LLR checkpoint.  KA-NEW2 (raw/KA_NEW2.json)
# proves the tree reproduces the banked C0 / RIC0 / RICLR rollouts bit for bit before any LLR read.
# Rollouts in PRIORITY order (a killed chain still yields the deciding contrast first), reader flags (--replan-steps 10 --nav deadreckon --schedule periodic --limit causal --dump-ticks <out>/ticks --dump-heads --device cpu --smooth sg72):
#   RILLR_s0 RILLR_s1 (--scene reindexed, DECIDING)  RILLRC_s0 RILLRC_s1 (reindexed + --feasible kbox, D)  LLR_s0 LLR_s1 (logged)  LLRC_s0 LLRC_s1 (logged + kbox)  FZLLR_s0 FZLLR_s1 (hold_now)
#   NO attribution-base rollout: the card's attribution LLR - CLR reads CLR's banked arrays (clr_read/raw/arrays: CLR logged, RICLR reindexed; sha256-pinned in the reader).
#   Threads: logged / hold_now arms LLR_TH_LOGGED (default 16 = the banked C0 / C0C / FZC0 / C0REP / CLR rollouts' count), reindexed arms (RILLR, RILLRC) LLR_TH_RI (default 12 = RIC0 / RIC0C / RICLR's).  MEASURED (KA-NEW2): the same tool at
#   the base's own thread count reproduces the banked rollout bit for bit; at another count a 1e-5 m rounding grows to 0.6 m on one window.
# Gates (REFUSE = raw/LLRRO.fail with the reason, exit code):
#   0  PINS.txt: the reader, pre-check, read-out, tests, this script, the LLR tree's tool / model / trainer / data and the CLR package's evidence match their sha256; the unit-test logs are green; the mutation log says ALL RED;
#      KA-NEW2 and the reader known answers (ka-null, ka-ri, ka-clr) say PASS                                                                                                     (exit 2)
#   1  code/llr_precheck.py: both LLR checkpoints present (absent = REFUSED), sha256 == SHA256SUMS.txt == the box-2 marker llr_markers/RUN_LLR_{s}.done evidence, argv = CLR's + `--lead-head on --w-lead W_LEAD --aux-cond token`,
#      cfg lead_head on / w_lead / aux_cond, e_aux keys, gate token, CLR's own checkpoint = the one the banked CLR arrays came from                                                (exit 5 absent, 8 failed)
#   2  the reused arrays' sha256 (16: C0 family, RIC0 family, CLR, RICLR), the CL card's rules sha, the A2 card sha = FROZEN_SHA.txt                                                (exit 6)
#   3  no other tool rollout / extractor runs (CPU shared; waits at most 3 h)
#   4  a 6-window PRE-FLIGHT per scene family: s0 LLR --scene reindexed --feasible kbox, s1 LLR --scene hold_now, s0 LLR --scene logged: each must exit 0 and write its ticks file                (exit 7)
# Each rollout writes done_<tag>.txt only on rc 0; a finished run is skipped on re-launch (RESUME = launch llr_read_chain.cmd again); an incomplete out dir is MOVED to _incomplete_*, never deleted.
# Markers: ZZLLR-* lines in logs/chain.out; raw/LLRRO.done (all ten rollouts) / raw/LLRRO.fail (reason inside); then the reader (code/llr_read.py --mode production): raw/LLREAD.done (raw/LLR_READ.json + RESULTS.md) /
#   raw/LLREAD.fail; then the token-wiring read-out: raw/LLRREADOUT.done (readout/LLR_READOUT.json, SENSITIVITY.md) / raw/LLRREADOUT.fail.  A REFUSED control of the reader is a result.
# WRITES: this package (logs, raw, readout, RESULTS.md, SENSITIVITY.md) and F:/TanitAD-scratch/motion_c/llr_read/{out,smoke,work,readout}.  The shared CL tree is not touched; the LLR tree is READ ONLY here.
'''
TABLE = '''table() {   # tag seed scene feasible run   (PRIORITY order; tab-separated)
  local r
  for r in "RILLR_s0 0 reindexed off LLR" "RILLR_s1 1 reindexed off LLR" "RILLRC_s0 0 reindexed kbox LLR" "RILLRC_s1 1 reindexed kbox LLR" \\
           "LLR_s0 0 logged off LLR" "LLR_s1 1 logged off LLR" "LLRC_s0 0 logged kbox LLR" "LLRC_s1 1 logged kbox LLR" "FZLLR_s0 0 hold_now off LLR" "FZLLR_s1 1 hold_now off LLR"; do
    set -- $r
    printf '%s\\t%s\\t%s\\t%s\\t%s\\n' "$1" "$2" "$3" "$4" "$5"
  done
}
'''
DRY = '''if [ $DRY -eq 1 ]; then
  echo "LLR read chain, DRY RUN (nothing runs, no marker is written)"
  echo "rollouts in priority order (tag seed scene feasible run; threads: reindexed $TH_RI, others $TH_LOGGED):"; table
  for s in 0 1; do
    r=M2b_s${s}_recft_LLR; b=M2b_s${s}_recft_CLR
    echo "checkpoint $RDB/$r/ckpt_last.pt: $([ -f "$RDB/$r/ckpt_last.pt" ] && echo PRESENT || echo ABSENT)   marker $RDB/llr_markers/RUN_LLR_$s.done: $([ -f "$RDB/llr_markers/RUN_LLR_$s.done" ] && echo PRESENT || echo ABSENT)   gate token: $([ -f "$RDB/llr_markers/gate_token_LLR_$s.json" ] && echo PRESENT || echo ABSENT)   base (CLR ckpt) $RDB/$b/ckpt_last.pt: $([ -f "$RDB/$b/ckpt_last.pt" ] && echo PRESENT || echo ABSENT)   banked CLR arrays: $([ -f "$ART/dagland/motion_c/clr_read/raw/arrays/CLR_s$s.npz" ] && [ -f "$ART/dagland/motion_c/clr_read/raw/arrays/RICLR_s$s.npz" ] && echo PRESENT || echo ABSENT)"
  done
  echo "FROZEN_LLR.txt + A2 card + FROZEN_SHA: $([ -f "$(dirname "$CARD")/FROZEN_LLR.txt" ] && [ -f "$CARD" ] && [ -f "$(dirname "$CARD")/FROZEN_SHA.txt" ] && echo PRESENT || echo ABSENT)   landed_tip.txt: $([ -f "$(dirname "$CARD")/../llr_launch/landed_tip.txt" ] && echo PRESENT || echo ABSENT)"
  echo "tree tool: $([ -f "$TOOL" ] && sha256sum "$TOOL" | cut -c1-16 || echo ABSENT)   tree model: $([ -f "$TREE/stack/tanitad/models/v8f_mini.py" ] && sha256sum "$TREE/stack/tanitad/models/v8f_mini.py" | cut -c1-16 || echo ABSENT)   PINS.txt: $([ -f "$PK/PINS.txt" ] && echo PRESENT || echo ABSENT)   KA_NEW2.json: $([ -f "$PK/raw/KA_NEW2.json" ] && echo PRESENT || echo ABSENT)"
  exit 0
fi

'''
GATE0 = '''for need in code/llr_read.py code/llr_precheck.py code/llr_readout.py code/llr_edits.py code/ka_new2.py tests/test_llr_read.py tests/test_llr_chain_gates.py tests/test_llr_readout.py llr_read_chain.sh; do
  [ -n "$(pin "$need")" ] || fail "PINS.txt carries no pin for $need" 2
done
for kv in "stack/scripts/v8f_t1_eval_ri.py:tool" "stack/tanitad/models/v8f_mini.py:model" "stack/scripts/v8f_mini_train.py:trainer" "stack/tanitad/data/v8f_mini_data.py:data"; do
  f=${kv%%:*}; k=${kv##*:}; want=$(pin "$k"); got=$(sha256sum "$TREE/$f" | cut -c1-64)
  [ -n "$want" ] && [ "$want" = "$got" ] || fail "LLR tree $k ($f) sha $got != PINS.txt ($want)" 2
done
for ev in "dagland/motion_c/clr_read/raw/CLR_READ.json:clr_read_json" "dagland/motion_c/clr_read/raw/CLR_PRECHECK.json:clr_precheck_json"; do
  f=${ev%%:*}; k=${ev##*:}; want=$(pin "$k"); got=$(sha256sum "/e/Projects/TanitAD-artifacts/$f" | cut -c1-64)
  [ -n "$want" ] && [ "$want" = "$got" ] || fail "CLR package evidence $k ($f) sha $got != PINS.txt ($want)" 2
done
for lg in pytest_llr_read pytest_llr_chain_gates pytest_llr_readout; do
  last=$(grep -v '^[[:space:]]*$' "$PK/logs/$lg.log" | tail -n 1)
  case "$last" in *" passed"*) case "$last" in *failed*|*error*) fail "unit tests not green ($lg): $last" 2;; esac;; *) fail "unit tests not green ($lg): $last" 2;; esac
done
grep -q "^MUTATIONS ALL RED" "$PK/logs/mutations_llr.log" || fail "the mutation log does not carry MUTATIONS ALL RED" 2
"$PY" -I -c "
import json,sys
pk=sys.argv[1]
for p,k in ((pk+'/raw/KA_NEW2.json','PASS'),(pk+'/raw/KA_READER_ka-null.json','KA_PASS'),(pk+'/raw/KA_READER_ka-ri.json','KA_PASS'),(pk+'/raw/KA_READER_ka-clr.json','KA_PASS')):
    d=json.load(open(p,encoding='utf-8'))
    if d.get(k) is not True: print('NOT PASS:',p); sys.exit(1)
" "$PKW" || fail "a known answer is not PASS (KA_NEW2, KA_READER ka-null / ka-ri / ka-clr)" 2
'''
GATE1_LOG = '''log "ZZLLR-GATE1 LLR checkpoints present, sha256 == SHA256SUMS.txt == box-2 marker, argv = CLR's + the three lead tokens (W_LEAD, token wiring), cfg lead_head on, e_aux keys, gate token, CLR base checkpoint = the banked arrays' checkpoint"

'''
READOUT = '''# ---- the O-sensitivity read-out adapted to the token wiring (the token's level input +-3 m/s, the lead presence input 0 <-> 1); a failure here does not touch LLREAD.done
mkdir -p "$PK/readout"
ro_rc=0
for s in 0 1; do
  OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 \\
    "$PY" -I "$PKW/code/llr_readout.py" --seed $s --threads 12 --batch 32 > "$PK/logs/readout_s$s.out" 2>&1 < /dev/null || ro_rc=$?
  log "ZZLLR-READOUT s$s rc=$ro_rc"
done
if [ $ro_rc -eq 0 ] && "$PY" -I "$PKW/code/llr_readout.py" --merge > "$PK/logs/readout_merge.out" 2>&1; then
  echo "$(date +%FT%T) readout/LLR_READOUT.json and SENSITIVITY.md written" > "$PK/raw/LLRREADOUT.done"
  tail -n 1 "$PK/logs/readout_merge.out" | tee -a "$PK/logs/chain.out"
  log "ZZLLR-CHAIN-DONE"
else
  echo "$(date +%FT%T) read-out rc=$ro_rc (see logs/readout_s*.out, logs/readout_merge.out)" > "$PK/raw/LLRREADOUT.fail"
  log "ZZLLR-READOUT-FAIL"
  exit 4
fi
'''


def rename(t):
    return t.replace("CLR", "LLR").replace("clr_", "llr_").replace("motion_c/cl_read/tree", "motion_c/llr_read/tree")


def span(t, name, start, end, new):
    if t.count(start) != 1:
        raise SystemExit(f"span {name!r}: start matches {t.count(start)} times (must be 1)")
    i = t.index(start)
    if t.count(end, i) != 1:
        raise SystemExit(f"span {name!r}: end matches {t.count(end, i)} times after start (must be 1)")
    return t[:i] + new + t[t.index(end, i):]


def edit(t, name, old, new):
    if t.count(old) != 1:
        raise SystemExit(f"edit {name!r}: old text matches {t.count(old)} times (must be 1)")
    return t.replace(old, new)


def build():
    raw = open(SRC, "rb").read()
    assert hashlib.sha256(raw).hexdigest() == SRC_SHA, "clr_read_chain.sh is not the pinned file"
    t = rename(raw.decode("utf-8"))
    t = span(t, "header", "# LLR READ chain (PREREG", "set -u\n", HEADER)
    t = span(t, "table", "table() {", "pin() {", TABLE)
    t = span(t, "dry", "if [ $DRY -eq 1 ]; then", 'if [ -f "$PK/raw/LLRRO.running" ]; then', DRY)
    t = span(t, "gate0", "for need in code/llr_read.py", 'log "ZZLLR-GATE0', GATE0)
    t = span(t, "gate1_log", 'log "ZZLLR-GATE1', "# ---- gate 2", GATE1_LOG)
    t = edit(t, "gate2_log", "reused arrays (14)", "reused arrays (16, the banked CLR base included)")
    t = edit(t, "preflight", 'pre PRE_RAPRP01_s1_ri 1 RAPRP01 "$TH_RI" --smooth sg72 --scene reindexed', 'pre PRE_LLR_s0_logged 0 LLR "$TH_LOGGED" --smooth sg72 --scene logged')
    i = t.index("# ---- A1's O-sensitivity read-out")
    t = t[:i] + READOUT
    t = edit(t, "twelve", "twelve LLR-read rollouts done", "ten LLR-read rollouts done")
    cmd = open(SRC_CMD, "rb").read()
    assert hashlib.sha256(cmd).hexdigest() == CMD_SHA, "clr_read_chain.cmd is not the pinned file"
    return t, rename(cmd.decode("utf-8")).replace("LLR read chain launcher", "LLR read chain launcher")


if __name__ == "__main__":
    sh, cmd = build()
    outs = {f"{PK}/llr_read_chain.sh": sh, f"{PK}/llr_read_chain.cmd": cmd.replace("\r\n", "\n")}
    ok = True
    for p, v in outs.items():
        if "--check" in sys.argv:
            same = os.path.isfile(p) and open(p, encoding="utf-8", newline="").read() == v
            print(os.path.basename(p), "== generated" if same else "DIFFERS from generated")
            ok &= same
        else:
            open(p, "w", encoding="utf-8", newline="\n").write(v)
            print("wrote", p, len(v))
    sys.exit(0 if ok else 1)
