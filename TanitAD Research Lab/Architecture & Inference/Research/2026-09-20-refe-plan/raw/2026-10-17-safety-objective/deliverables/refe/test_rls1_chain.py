#!/usr/bin/env python3
"""Tests for the RL-S1 pod chain: refe/rls1_pod.sh and refe/rls1_checks.py (eval/RLS1_POD_RUNBOOK.md).  No pod, no GPU, no network.

  1. `check_chain(text)`: static properties the chain must hold (P2 over, pinned checkpoints, partition, teacher lane, coverage + remedy, the join gate, the per-arm
     order preflight -> gate -> token -> train, the token VERIFIED not grepped, egress, kill monitor).  Every property is broken in a copy: `check_chain` must say so.
  2. BEHAVIOUR, extracted verbatim from the chain and run under bash: p2_over / gpu_busy (lock held, GPU busy, P2 not done), cgroup_cores (v1 and v2), arm_args
     (the five argv compared TOKEN FOR TOKEN with the gate section of eval/HANDOFF_RL_S1.md, then judged by the REAL launch gate: CLEAN, and a C arm carrying
     --epo / an R arm without it / R0 with K,L ... REFUSED), token_ok against tokens minted by the gate's own mint() for refe_rls1.
  3. The REAL rls1_pod.sh executed in a sandbox up to its first refusals (lock held, GPU busy, P2 unfinished, no HF_TOKEN, missing or wrong model_final).
  4. rls1_checks.py in both directions (pins, partition, teacher lane, the join gate below 100 % / with a G-REWARD-ID break, the kill rules, monitor, replay).
  5. TL-1 (2026-10-12): the chain PRODUCES the teacher-lane labels (smoke -> per-shard labeller -> EXACT gate), the partition constants the chain carries equal the pinned
     ones, the planned update count equals the pre-registered budget; `rls1_checks.tlane_exact` / `updates_check` are driven in both directions and by source mutations.

  6. The FAN-EQUIVALENCE GATE (2026-10-12, stage `fancmp`): does this pod's dumper reproduce the fan the main run banked at step 10,075?  `check_fancmp` holds its static
     properties (the pinned slice, the argv that reaches rl_fan_dump.py, the verdict that is read from the dumper's exit AND BOTH markers at the bound each echoes, the TWO
     bounds -- the catastrophe bound `FANCMP_ATOL` and the mean bound `FANCMP_MEAN_ATOL` -- being constants of the chain equal to the ones the measurement chose in
     raw/rl_fan/atol_calibration/round2/verdict_round2.json (`catastrophe_atol_kept`, `CHOSEN_MEAN_ATOL`, state PASS), `RLS1_FANCMP_ATOL` / `RLS1_FANCMP_MEAN_ATOL` being guards that
     must EQUAL the pins and never inputs, the done-marker bound to the bounds it passed under, the banked-sets path equal to the P2 chain's); `fancmp_behaviour` runs the
     stage's own text under bash against a stub dumper and asserts the argv the caller really passes, every refusal, every env value that must be refused, and a stale marker.

  7. THE QUEUE and FAIL-CLOSED supervision (2026-10-14): `check_queue` holds their static properties here; the interleavings (S1, the P2 seed replicate before / after its wait, P6, the
     settle race, the GPU), the monitor-dies-while-training case that must end in K-MON, the checker's memory and the mutations that must go RED run in refe/test_rls1_queue.py.

  8. RL-S1c / R0c (2026-10-15; eval/rls1c_arms.json, profile refe_rls1c, raw/2026-10-15-rls1-adam-dose/PREREG_R0C_DRAFT.md): `check_rls1c` holds the static properties (registered, its profile, its argv
     line, its control rule, the loud second-control marker, the generic kill rule, the suite files the gate runs); `r0c_behaviour` RUNS the chain's own text under bash (the argv token for token against
     eval/rls1c_arms.json and R0's banked sha256, the profile each arm gets, the control-read block with and without C0's banked dumps, the kill refusals, the precheck refusal of misspelled arms);
     `r0c_mutations` breaks each property in a copy and the matching check must say so; the REAL gate and a REAL minted token are driven for the chain's own R0c argv and profile.
     `write_sh` gives every shim the exec bit (the pod runs them directly: without +x each one fails with rc 126; Git Bash hides it).

Run: python refe/test_rls1_chain.py   -> ZZTEST_PASS <n>   (needs bash on PATH; Git Bash on the dev box).   Any skipped check prints ZZTEST_SKIP and is NOT counted.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import io
import contextlib
import shutil
import subprocess
import types
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKGROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PKGROOT / "eval"))
import gate_token as GT  # noqa: E402
import rls1_checks as RC  # noqa: E402
from test_p2r_chain import drop_line, logical_commands, tar_owner_violations  # noqa: E402

CHAIN = HERE / "rls1_pod.sh"
ARMS = ("C0", "C1", "R0", "R1", "R2")
CHILD_RX = re.compile(r'"\$PY" +(scorer_finetune\.py|rls1_checks\.py|rl_fan_dump\.py|rl_partition\.py|rl_pool_available\.py|onpolicy_relabel_pdm\.py|rl_terms_label\.py|rl_tlane_label\.py|'
                      r'pdm_label_complete\.py|launch_gate\.py|hf_egress\.py)|"\$PY" +"\$PKG/refe/launch_gate\.py"|bash +"\$PKG/code/|^sleep |\bsleep +\d+ |\bdd +if=')
ORDER = ["p2_over)\" || refuse", "; stop_after queue", "source \"$WORK/teacher_env.sh\"", "--pin model_final", "--pin u400", "verify-partition", "rl_fan_dump.py --plan-only",
         "--compare-atol \"$FANCMP_ATOL\"", "STAGES=dbs", "PT._setup()", "dump_one val 0 1", "rl_tlane_label.py --sets \"$SETS_VA\"", "rls1_checks.py tlane-exact", "rls1_checks.py join-gate", "--tag rlfan_data",
         "rls1_checks.py updates-check", "scorer_finetune.py $ARGS --preflight",
         "launch_gate.py run --profile \"$PROFILE\"", "scorer_finetune.py $ARGS >>", "rls1_checks.py replay", "--rl-val-scenes $VAL_TSV", "--tag \"arm_$ARM\""]
PRESENT = ["onpolicy_relabel_pdm.py --sets", "rl_terms_label.py --sets", "rl_tlane_label.py --sets", "dump_one()", "label_one()"]
TOKEN_OK_RX = re.compile(r"^token_ok\(\) \{\n.*?^\}\n", re.S | re.M)


def fn_body(text, name):
    m = re.search(rf"^{name}\(\) \{{\n.*?^\}}\n", text, re.S | re.M)
    return m.group(0) if m else ""


def fn_any(text, name):
    """a function body whether it is written over several lines or on one."""
    b = fn_body(text, name)
    if b:
        return b
    m = re.search(rf"^{name}\(\) .*$", text, re.M)
    return m.group(0) if m else ""


def check_queue(text):
    """Static properties of the QUEUE and of the FAIL-CLOSED supervision (2026-10-14).  [] = all hold.  Behaviour is driven in refe/test_rls1_queue.py."""
    v = []
    qw, ah, sa, st, gi = (fn_any(text, n) for n in ("queue_wait", "ahead", "supervise_arm", "stop_trainer", "gpu_idle_wait"))
    # ---- lock discipline: chain.queue (fd 199) is the single-instance guard for the whole life; chain.lock (fd 200) is taken ONLY inside queue_wait, and given back while anything is ahead
    if 'exec 199>"$R/chain.queue"' not in text or "flock -n 199" not in text:
        v.append("QUEUE_NO_INSTANCE_GUARD")
    if len(re.findall(r'^\s*exec 200>"\$R/chain\.lock"', text, re.M)) != 1 or 'exec 200>"$R/chain.lock"' not in qw:
        v.append("CHAIN_LOCK_TAKEN_OUTSIDE_QUEUE")
    if "exec 200>&-" not in qw:
        v.append("QUEUE_NEVER_YIELDS_THE_LOCK")
    if "flock -n 200 || refuse" not in qw:
        v.append("QUEUE_LOCK_NOT_CHECKED")
    # ---- what is AHEAD: P2 and S1 by their lock; the replicate and P6 only once PAST their wait (a waiting chain waits on us)
    for need in ('lock_held "$P2"', 'lock_held "$S1"', 'chain_past "$P2REP" "ZZP2REP_START" "ZZP2REP_WAIT_OK"', 'chain_past "$P6D"', "ZZP6_WAIT_OK", "p6_pod.sh"):
        if need not in ah:
            v.append("AHEAD_LACKS:" + need)
    if "lock_held \"$P2REP\"" in ah.replace("chain_past", ""):
        v.append("AHEAD_WAITS_ON_A_WAITING_REPLICATE")
    cp = fn_any(text, "chain_past")
    if "lock_held" not in cp or "index($0, s) { ok = 0 }" not in cp or "index($0, k) { ok = 1 }" not in cp:
        v.append("CHAIN_PAST_NOT_PER_ATTEMPT")
    if not fn_any(text, "lock_held").count("9<>") or re.search(r"9<(?!>)", fn_any(text, "lock_held")):
        v.append("LOCK_PROBE_NOT_READ_WRITE")
    # ---- the GPU: 3 idle polls before the first load (queue) and before every trainer; an unanswerable nvidia-smi is busy (gpu_busy, checked above)
    if "gpu_busy" not in qw or "RLS1_IDLE_POLLS" not in qw or "RLS1_SETTLE_S" not in qw or '[ "$idle" -ge "$need" ] && return 0' not in qw:
        v.append("QUEUE_NO_GPU_POLLS")
    if "gpu_busy" not in gi or "RLS1_IDLE_POLLS" not in gi or '[ "$idle" -ge "$need" ] && return 0' not in gi:
        v.append("NO_GPU_IDLE_WAIT_FN")
    # ---- order: the queue clears before the disk test, the checkpoints, the dump (GPU) and every arm; the arm list is validated before the queue
    qpos = text.find("\nqueue_wait\n")
    if qpos < 0:
        v.append("QUEUE_NOT_CALLED")
    else:
        for later in ('dd if=/dev/zero', 'source "$WORK/teacher_env.sh"', '"$PY" rl_fan_dump.py', 'scorer_finetune.py $ARGS'):
            if text.find(later) < qpos:                                  # absent (-1) or before the queue clears
                v.append("ORDER:queue_after:" + later)
        arm_chk = text.find('arm_registered "$ARM" || refuse "unknown arm \'$ARM\' in RLS1_ARMS')
        if arm_chk < 0 or arm_chk > qpos:
            v.append("ARMS_NOT_VALIDATED_EARLY")
    tp = text.find("scorer_finetune.py $ARGS >>")
    if "gpu_idle_wait || refuse" not in text[max(0, tp - 700):tp]:
        v.append("NO_GPU_IDLE_BEFORE_TRAINER")
    wp = text.find('wait "$TPID"; TRC=$?')
    if 'supervise_arm "$ARM" "$TPID" "$MPID"' not in text[tp:wp] or wp < 0:
        v.append("NO_SUPERVISOR")
    # ---- fail closed (K-MON)
    if "pid_alive \"$mp\"" not in sa or "wait \"$mp\"" not in sa or 'if pid_alive "$tp"; then why="kill rules unenforceable: monitor died rc $MON_RC"; fi' not in sa:
        v.append("KMON_NO_DEAD_MONITOR_BRANCH")
    if '[ $((SECONDS - t0)) -ge "$ready_s" ]' not in sa or "monitor not READY within" not in sa or '^ZZRLS1_MONITOR_READY' not in sa:
        v.append("KMON_NO_READY_TIMEOUT")
    if 'stop_trainer "$tp"' not in sa:
        v.append("KMON_TRAINER_NOT_STOPPED")
    if '"code": "K-MON"' not in sa or '> "$ad/KILLED.json"' not in sa:
        v.append("KMON_NO_KILLED_JSON")
    if '[ -f "$ad/KILLED.json" ] || printf' not in sa:
        v.append("KMON_OVERWRITES_A_KILL")
    if 'kill -TERM -- "-$pg"' not in st or 'kill -KILL -- "-$pg"' not in st or '[ "$pg" -gt 1 ]' not in st or '"$pg" != "$$"' not in st:
        v.append("STOP_NOT_BY_EXPLICIT_PGID")
    if not re.search(r'if \[ -n "\$\{MON_RC:-\}" \]; then MRC=\$MON_RC; else wait "\$MPID"; MRC=\$\?; fi', text):
        v.append("MON_RC_NOT_CARRIED")
    if '"$ARMDIR"/kill_checks.jsonl "$ARMDIR"/monitor.log' not in text:
        v.append("MONITOR_LOG_NOT_BANKED")
    if '&& [ ! -f "$S/arm_${ARM}_killed" ]; then' not in text:
        v.append("KILLED_ARM_RETRAINS")
    if 'rm -f "$S/read_$ARM.ok" "$S/egress_$ARM.ok"' not in text:
        v.append("STALE_MARKERS_KEPT")
    # ---- the RL-S1b arms are somebody else's: three named insertion points, none filled
    for k in (1, 2, 3):
        if f"RLS1B-INSERTION-POINT-{k}" not in text:
            v.append(f"NO_INSERTION_POINT_{k}")
    known = {"C0", "C1", "R0", "R1", "R2", "R0b", "R1b", "R0c", "R0d", "R1d", "R1o", "*"}                  # R0c: 2026-10-15, eval/rls1c_arms.json (checked in check_rls1c)
    if set(re.findall(r"^    (\w+|\*)\)", fn_body(text, "arm_args"), re.M)) - known:
        v.append("INVENTED_RLS1B_ARGV")                       # R2b and anything else RL-S1b names needs a PI decision / its owner's argv: never invented here
    mreg = re.search(r'^REGISTERED_ARMS="([^"]*)"', text, re.M)
    if not mreg or not {"R0b", "R1b"} <= set(mreg.group(1).split()):
        v.append("RLS1B_ARMS_NOT_REGISTERED")
    if 'R0b|R1b) echo refe_rls1b' not in fn_body(text, "arm_profile"):
        v.append("NO_RLS1B_PROFILE")
    if 'PROFILE="$(arm_profile "$ARM")"' not in text or 'refuse "RL-S1b arm $ARM is read against $RD' not in text:
        v.append("RLS1B_CONTROL_NOT_ENFORCED")
    if text.count('arm_registered "$ARM" || refuse') != 2:
        v.append("ARM_NAMES_NOT_REFUSED_TWICE")
    return v


def arm_block(text, arm):
    m = re.search(rf"^    {arm}\) .*$", fn_body(text, "arm_args"), re.M)
    return m.group(0) if m else ""


R0_ARGV_SHA256 = "24451b621d515066bf8febefaf66bea2c7a345b48bf7a89c9443e2d848d8eef6"      # R0's banked token argv (HF Sayood/tanitad-refe-rls1-r0 final/PASS_a120418a870f.json)
R0C_ARGV_SHA256 = "f588a6cbdf8319bc30e8da12407f5eeb65a5e0a82d392320f2676ca3f94ed534"      # R0's argv with /arms/R0 -> /arms/R0c plus `--epo-ref armA` (raw/2026-10-15-r0c-build/R0c_ARGV.txt)
R0C_EPO_TOKENS = {"--epo", "--epo-terms-train", "$TERMS_TR", "--epo-terms-heldout", "$TERMS_VA", "--epo-ref", "armA"}      # R0c = R0 + --epo-ref armA: no other lever may appear in its line
R0C_SUITE_FILES = ("refe/test_rl_s1b.py", "refe/test_rl_r0c.py")                          # the suite the refe_rls1c gate runs on the pod (launch_gate.RLS1C_SUITE): absent = a refusal AFTER the queue wait and the preflight
SAFEOBJ_FILES = ("eval/safeobj_read.py", "eval/test_safeobj_read.py", "eval/test_safeobj_gate.py")      # the gate loads the first BY PATH (rls1_checks._safeobj_reader), the launch gate's suite runs the other two: absent = a refusal AFTER the queue wait
KILL_HEAD_RX = re.compile(r'^\s*\[ -f "\$S/arm_\$\{ARM\}_killed" \](?: && stage_done "egress_\$ARM")?\s*$')           # a kill rule keyed on the arm VARIABLE, never on an arm's name


def argv_sha256(argv):
    return hashlib.sha256(json.dumps(argv, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def control_block(text):
    """The arm loop's control-read lines (stage 9, from the KIND / RD case to just before 9a): the text r0c_behaviour runs under bash."""
    m = re.search(r'^  case "\$ARM" in C\*\) KIND=A;.*?\n(?=  # ---- 9a\.)', text, re.S | re.M)
    return m.group(0) if m else ""


def check_rls1c(text):
    """Static properties of the R0c wiring (RL-S1c, 2026-10-15).  [] = all hold.  The behaviour behind each is driven by r0c_behaviour and the arm-loop scenarios of refe/test_rls1_queue.py."""
    v = []
    mreg = re.search(r'^REGISTERED_ARMS="([^"]*)"', text, re.M)
    if not mreg or "R0c" not in mreg.group(1).split():
        v.append("R0C_NOT_REGISTERED")
    if "R0c) echo refe_rls1c" not in fn_body(text, "arm_profile"):
        v.append("R0C_NO_PROFILE")
    blk = arm_block(text, "R0c")
    if not blk:
        v.append("R0C_NO_ARGV")
    else:
        if "seed=0;" not in blk:
            v.append("R0C_WRONG_SEED")
        m = re.search(r'epo="([^"]*)"', blk)
        toks = m.group(1).split() if m else []
        if toks.count("--epo-ref") != 1 or toks[toks.index("--epo-ref") + 1:toks.index("--epo-ref") + 2] != ["armA"]:
            v.append("R0C_NO_EPO_REF_ARMA")
        v += ["R0C_EXTRA_LEVER:" + t for t in toks if t not in R0C_EPO_TOKENS]
    cb = control_block(text)
    if not cb:
        v.append("R0C_NO_CONTROL_BLOCK")
    else:
        if '[ "$ARM" = R0c ] && RD=C0' not in cb:
            v.append("R0C_SECOND_CONTROL_NOT_NAMED")
        i_generic, i_own = cb.find('if [ -z "$RD" ] ||'), cb.find('[ "$ARM" = R0c ] && REFDIR="$ARMDIR/scores"')
        if i_own < 0 or i_generic < 0 or i_own < i_generic:
            v.append("R0C_NOT_READ_AGAINST_ITS_OWN_A")
    if 'elif [ "$ARM" = R0c ]; then\n        log "ZZRLS1_R0C_NO_SECOND_CONTROL' not in text:
        v.append("R0C_SILENT_WITHOUT_SECOND_CONTROL")
    kills = [ln for ln in text.split("\n") if "arm $ARM was KILLED" in ln and not ln.lstrip().startswith("#")]
    if len(kills) != 2:
        v.append(f"KILL_RULE_LINES:{len(kills)}")
    v += ["KILL_RULE_ARM_GATED:" + ln.strip()[:50] for ln in kills if not KILL_HEAD_RX.match(ln.split("&& refuse", 1)[0])]
    fl = re.search(r'for f in (.*?); do\n  \[ -f "\$PKG/\$f" \] \|\| refuse', text, re.S)
    req = fl.group(1).replace("\\\n", " ").split() if fl else []
    v += ["R0C_SUITE_FILE_NOT_REQUIRED:" + f for f in R0C_SUITE_FILES if f not in req]
    if 'if [ "$PROFILE" = refe_rls1b ] && [ "$KIND" = B ] && [ "$REFDIR" = "$ARMDIR/scores" ]; then' not in cb:
        v.append("RLS1B_GUARD_EDITED")                     # the RL-S1b rule is untouched, and R0c is not under it (its own arm A IS its registered control)
    return v


def token_line_ok(ln):
    s = ln.strip()
    return (s.startswith("#") or "env -u HF_TOKEN" in s or s == '[ -n "${HF_TOKEN:-}" ] || refuse "HF_TOKEN is not set in the environment"'
            or re.match(r'if \[ -n "\$\{PY:-\}" \] && \[ -d "\$PKG/refe" \] && \[ -n "\$\{HF_TOKEN:-\}" \]; then', s) is not None)


FANCMP_BLOCK_RX = re.compile(r"^if ! stage_done fancmp; then\n(.*?)^fi\n", re.S | re.M)
CAL_DIR = PKGROOT / "raw" / "rl_fan" / "atol_calibration"


def check_fancmp(text):
    """Static properties of stage `fancmp` (the fan-equivalence gate).  [] = all hold."""
    import hashlib
    v = []
    m = FANCMP_BLOCK_RX.search(text)
    if not m:
        return ["FANCMP_STAGE_MISSING"]
    body = m.group(1)
    cmds = [c for c in logical_commands(body) if "rl_fan_dump.py" in c and '"$PY"' in c]
    if len(cmds) != 1:
        return [f"FANCMP_DUMP_CALLS:{len(cmds)}"]
    c = cmds[0]
    for need, tag in (('--ckpt "$FINAL"', "FANCMP_NOT_FROM_MODEL_FINAL"), ('--expect-ckpt-sha256 "$FINAL_SHA256"', "FANCMP_SHA_NOT_PINNED"), ('--keys "$FANCMP_KEYS"', "FANCMP_KEYS_NOT_THE_SLICE"),
                      ('--targets "$TARGETS"', "FANCMP_WRONG_BANK"), ('--images "$IMAGES"', "FANCMP_WRONG_IMAGES"), ('--calib "$CALIB"', "FANCMP_WRONG_CALIB"),
                      ('--compare-sets "$BANKED"', "FANCMP_NO_COMPARE_SETS"), ('--compare-atol "$FANCMP_ATOL"', "FANCMP_NO_ATOL"),
                      ('--compare-mean-atol "$FANCMP_MEAN_ATOL"', "FANCMP_NO_MEAN_ATOL"),
                      ('--compare-min-matched "$NK"', "FANCMP_NO_MIN_MATCHED"), ("200>&-", "FANCMP_FD200"), ("env -u HF_TOKEN", "FANCMP_TOKEN_NOT_STRIPPED"),
                      ("OMP_NUM_THREADS=1", "FANCMP_THREADS")):
        if need not in c:
            v.append(tag)
    for bad in ("--allow-missing", "--limit", "--plan-only", "--shard"):
        if bad in c:
            v.append("FANCMP_NARROWED:" + bad)
    # the verdict: the dumper's exit AND the max-bound marker AND the mean-bound marker, each with the slice's key count and ECHOING the pinned bound it applied
    verdict_line = ('{ [ "$frc" = 0 ] && grep -qE "^ZZRLFAN_ATOL_PASS $NK [^ ]+ ${FANCMP_ATOL}\\$" "$R/fancmp.log" '
                    '&& grep -qE "^ZZRLFAN_MEAN_ATOL_PASS $NK [^ ]+ ${FANCMP_MEAN_ATOL}\\$" "$R/fancmp.log"; } || refuse')
    if verdict_line not in body:
        v.append("FANCMP_VERDICT_UNCHECKED")
    if not re.search(r'sha256sum "\$FANCMP_KEYS".*= "\$FANCMP_KEYS_SHA256" \] \|\| refuse', body):
        v.append("FANCMP_KEYS_UNPINNED")
    if not re.search(r'ls "\$BANKED"/onpolicy_\*\.jsonl > /dev/null 2>&1 \|\| refuse', body):
        v.append("FANCMP_NO_BANKED_CHECK")
    if 'NK=$(( $(wc -l < "$FANCMP_KEYS") - 1 ))' not in body:
        v.append("FANCMP_NK_NOT_THE_SLICE_SIZE")
    if "stop_after fancmp" not in text:
        v.append("FANCMP_NO_STOP_AFTER")
    if 'log "ZZRLS1_FANCMP_OK' not in text:
        v.append("FANCMP_NO_OK_MARKER")
    if text.find("stage_done fancmp") > text.find("dump_one val 0 1"):
        v.append("ORDER:fancmp_after_the_first_dump")
    if not re.search(r"^\s+splits/rl_fancmp_keys\.tsv( \\)?$", text, re.M):
        v.append("FANCMP_KEYS_NOT_REQUIRED")
    # ---- drift against the artifacts the constants must equal
    kf = PKGROOT / "splits" / "rl_fancmp_keys.tsv"
    msha = re.search(r"^FANCMP_KEYS_SHA256=(\w+)$", text, re.M)
    if not kf.is_file():
        v.append("FANCMP_KEYS_FILE_ABSENT")
    else:
        raw = kf.read_bytes()
        if not msha or msha.group(1) != hashlib.sha256(raw).hexdigest():
            v.append("FANCMP_KEYS_PIN_DRIFT")
        rows = raw.decode("utf-8").split("\n")
        ho = set((PKGROOT / "splits" / "sft_heldout_logs.txt").read_text(encoding="utf-8").split())
        body_rows = [r.split("\t") for r in rows[1:] if r]
        if rows[0] != "token\tlog" or b"\r" in raw or len(body_rows) != 512 or len({r[0] for r in body_rows}) != 512 or any(len(r) != 2 for r in body_rows):
            v.append("FANCMP_KEYS_MALFORMED")
        elif any(r[1] in ho for r in body_rows):
            v.append("FANCMP_KEYS_IN_HELDOUT_LOGS")
    # The BOUNDS: constants of the chain, NOT inputs, equal to what the measurement chose.  Round 2 (raw/rl_fan/atol_calibration/round2/verdict_round2.json, state PASS) chose the
    # MEAN bound (CHOSEN_MEAN_ATOL) and kept the catastrophe bound (catastrophe_atol_kept); a number the pre-registered rule did not accept is an assumption.
    vj = CAL_DIR / "round2" / "verdict_round2.json"
    verdict = json.loads(vj.read_text(encoding="utf-8")) if vj.is_file() else None
    if verdict is None:
        v.append("FANCMP_BOUNDS_UNMEASURED")
    elif verdict.get("state") != "PASS":
        v.append(f"FANCMP_ROUND2_NOT_PASS:{verdict.get('state')}")
    for var, key in (("FANCMP_ATOL", "catastrophe_atol_kept"), ("FANCMP_MEAN_ATOL", "CHOSEN_MEAN_ATOL")):
        if len(re.findall(rf"^{var}=", text, re.M)) != 1:
            v.append(f"{var}_NOT_ASSIGNED_ONCE")
        mc = re.search(rf"^{var}=([0-9]*\.?[0-9]+)$", text, re.M)
        if not mc:
            v.append(f"{var}_NOT_PINNED")
        elif verdict is not None:
            want = verdict.get(key)
            if want is None or abs(float(mc.group(1)) - float(want)) > 1e-12:
                v.append(f"{var}_DRIFT:chain {mc.group(1)!r} measured {want!r} ({key})")
    # the env vars are GUARDS (set -> must equal the pin, else refuse), read nowhere else
    for env, var in (("RLS1_FANCMP_ATOL", "FANCMP_ATOL"), ("RLS1_FANCMP_MEAN_ATOL", "FANCMP_MEAN_ATOL")):
        gl = next((ln for ln in body.split("\n") if ln.strip().startswith(f'[ -z "${{{env}:-}}" ] ||')), "")
        if not (gl and f'awk -v a="${env}" -v p="${var}"' in gl and "a + 0 == p + 0" in gl and gl.rstrip().count("|| refuse") == 1 and "a ~ /^[0-9]*[.]?[0-9]+$/" in gl):
            v.append(f"{env}_NOT_GUARDED")
    for ln in text.split("\n"):
        if re.search(r"(?<![A-Z0-9])RLS1_FANCMP_(MEAN_)?ATOL", ln) and not ln.lstrip().startswith("#") and not ln.lstrip().startswith('[ -z "${RLS1_FANCMP_'):
            v.append("FANCMP_BOUND_READ_FROM_ENV:" + ln.strip()[:60])
    # the done-marker is bound to the bounds it passed under: a marker from another pair (or none) is discarded and the stage runs again
    if not re.search(r'^if stage_done fancmp && ! grep -qx "atol=\$FANCMP_ATOL mean_atol=\$FANCMP_MEAN_ATOL" "\$S/fancmp\.ok"; then\n.*\n  rm -f "\$S/fancmp\.ok"\nfi$', text, re.M):
        v.append("FANCMP_MARKER_NOT_BOUND_TO_BOUNDS")
    if not re.search(r'  stage_mark fancmp\n  echo "atol=\$FANCMP_ATOL mean_atol=\$FANCMP_MEAN_ATOL" >> "\$S/fancmp\.ok"\n', body):
        v.append("FANCMP_MARKER_BOUNDS_NOT_RECORDED")
    if "grep '^ZZRLFAN_MEAN_ATOL_PASS'" not in text:
        v.append("FANCMP_OK_MARKER_NO_MEAN_VERDICT")
    # the banked-sets default is P2's own SETS path (refe/p2r_pod.sh), not a guess
    mb = re.search(r'^BANKED="\$\{RLS1_BANKED_SETS:-(\S+?)\}"$', text, re.M)
    p2 = (HERE / "p2r_pod.sh").read_text(encoding="utf-8")
    ms = re.search(r"^SETS=(\$DATA/\S+)$", p2, re.M)
    if not mb or not ms or mb.group(1) != ms.group(1):
        v.append(f"FANCMP_BANKED_DRIFT:{mb.group(1) if mb else None} vs p2r_pod.sh {ms.group(1) if ms else None}")
    return v


def check_chain(text):
    v = []
    if "\r" in text:
        v.append("CRLF")
    if re.search(r"^\s*set\s+-[a-wyz]*x|\bbash\s+-x\b", text, re.M):
        v.append("XTRACE")
    for ln in text.split("\n"):
        if "HF_TOKEN" in ln and not token_line_ok(ln):
            v.append("HF_TOKEN_USE:" + ln.strip()[:60])
    if '[ -n "${HF_TOKEN:-}" ] || refuse' not in text:
        v.append("NO_TOKEN_ASSERT")
    for c in logical_commands(text):
        if CHILD_RX.search(c) and "200>&-" not in c:
            v.append("FD200:" + c[:70])
    pos = [text.find(k) for k in ORDER]
    v += ["MISSING:" + k for k, p in zip(ORDER, pos) if p < 0]
    v += ["MISSING:" + k for k in PRESENT if k not in text]
    if all(p >= 0 for p in pos) and pos != sorted(pos):
        v.append("ORDER:" + ",".join(ORDER[i] for i in range(1, len(ORDER)) if pos[i] < pos[i - 1]))
    # ---- P2 is over
    po, gb = fn_body(text, "p2_over"), fn_body(text, "gpu_busy")
    if "flock -n 9" not in po or '"$P2/chain.lock"' not in po:
        v.append("NO_P2_LOCK_CHECK")
    if "gpu_busy" in po:
        v.append("P2_OVER_JUDGES_THE_GPU")                  # a chain queued behind S1 must be startable while S1 trains
    if "gpu_busy" not in fn_body(text, "queue_wait") or "--query-compute-apps=pid" not in gb or "|| return 0" not in gb:
        v.append("NO_GPU_CHECK")
    if 'grep -q "ZZP2R_DONE" "$P2/chain.log"' not in po:
        v.append("NO_P2_DONE_CHECK")
    if 'why="$(p2_over)" || refuse "$why"' not in text:
        v.append("P2_OVER_NOT_CALLED")
    elif text.find('why="$(p2_over)"') > text.find("dd if=/dev/zero"):
        v.append("ORDER:p2_over_after_dd")
    if re.search(r"^[^#\n]*\b(pgrep|pkill)\b", text, re.M):
        v.append("PGREP_SELF_MATCH")
    if not re.search(r"(^|; )PKG=\$WORK/refe-plan-rls1(;|$)", text, re.M):
        v.append("PKG_IS_NOT_ITS_OWN_DIR")
    # ---- checkpoints, partition, teacher lane, coverage
    if f"FINAL_SHA256={RC.PINS['model_final']['sha256']}" not in text:
        v.append("PIN_MISMATCH:FINAL_SHA256")
    for pin, lg in (("model_final", "ckpt_final.log"), ("u400", "ckpt_u400.log")):
        if f"rls1_checks.py verify-file --path" not in text or f"--pin {pin}" not in text:
            v.append("CKPT_NOT_VERIFIED:" + pin)
        elif not re.search(rf'grep -q "\^ZZRLS1_FILE_OK" "\$R/{lg}" \|\| refuse', text):
            v.append("CKPT_RESULT_UNCHECKED:" + pin)
    if not re.search(r'grep -q "\^ZZRLS1_PARTITION_OK" "\$R/partition_verify\.log" \|\| refuse', text):
        v.append("PARTITION_UNCHECKED")
    # ---- TL-1: the chain PRODUCES the teacher-lane labels (smoke, per-shard labeller, exact gate); nothing is supplied from outside
    if re.search(r"RLS1_TLANE_(TRAIN|HELDOUT|MIN)", text):
        v.append("TLANE_OVERRIDE_PRESENT")
    if not re.search(r'grep -q "\^ZZTLANE_DONE 3 0 0\$" "\$R/tlane_smoke\.log" \|\| refuse', text):
        v.append("TLANE_SMOKE_UNCHECKED")
    if not re.search(r'grep -q "\^ZZRLS1_TLANE_OK" "\$R/tlane_smoke_check\.log" \|\| refuse', text):
        v.append("TLANE_SMOKE_ROWS_UNCHECKED")
    if (not re.search(r'grep -q "\^ZZRLS1_TLANE_OK" "\$R/tlane_exact_\$side\.log" \|\| refuse', text)
            or 'tlane-exact --dir "$tl" --sets "$sd" --tsv "$tsv" --expect "$ex"' not in text):
        v.append("TLANE_EXACT_UNCHECKED")
    if not re.search(r'grep -q "\^ZZTLANE_DONE \[0-9\]\* \[0-9\]\* 0\$" "\$R/\$m\.tlane\.log" \|\| \{ log "REFUSED', text):
        v.append("TLANE_LABELS_UNAUDITED")
    tl_cmds = [c for c in logical_commands(text) if "rl_tlane_label.py" in c and '"$PY"' in c]
    if len(tl_cmds) != 2:
        v.append(f"TLANE_LABELLER_CALLS:{len(tl_cmds)}")
    for c in tl_cmds:
        if "--repair-last-heading" not in c:
            v.append("TLANE_NO_REPAIR")
        if "REFE_SIM_HZ=10" not in c:
            v.append("TLANE_NO_SIM_HZ")
        if '--bank "$TARGETS"' not in c:
            v.append("TLANE_BANK_NOT_THE_TRAINING_BANK")
    lab = [c for c in tl_cmds if "$4" in c]
    if len(lab) != 1 or '--keys "$8"' not in lab[0] or '--shard "$2/$3"' not in lab[0]:
        v.append("TLANE_KEYS_OR_SHARD_NOT_PINNED")
    if "for d in sets_train sets_val pdm_train pdm_val terms_train terms_val tlane_train tlane_heldout; do" not in text:
        v.append("BANK_LACKS_TLANE")
    if not re.search(r'grep -q "\^ZZRLS1_UPDATES_OK" "\$ARMDIR/updates\.log" \|\| refuse', text) or 'updates-check --args "$ARGS" --n-sets "$NTRAIN"' not in text:
        v.append("UPDATES_UNCHECKED")
    # the partition constants the chain carries are the pinned ones (a re-cut without a re-pin, or the reverse, is a silent drift)
    mex = re.search(r"tsv=\$TRAIN_TSV; ex=(\d+); else .*? tsv=\$VAL_TSV; ex=(\d+); fi", text)
    if not mex or (int(mex.group(1)), int(mex.group(2))) != (RC.PARTITION["n_train"], RC.PARTITION["n_val"]):
        v.append("PARTITION_DRIFT:join_gate_counts")
    mrm = re.search(r"--n-train (\d+) --n-val (\d+) --min-val-logs (\d+) --seed 20261012", text)
    if not mrm or tuple(int(x) for x in mrm.groups()) != (RC.PARTITION["n_train"], RC.PARTITION["n_val"], RC.PARTITION["min_val_logs"]):
        v.append("PARTITION_DRIFT:remedy_params")
    mrid = re.search(r"ZZRLS1_REMEDY old partition_id (\w+) new", text)
    if not mrid or not RC.PARTITION["partition_id"].startswith(mrid.group(1)):
        v.append("PARTITION_DRIFT:remedy_old_id")
    mpn = re.search(r"^PART=.*?; PNAME=(\S+)$", text, re.M)
    if not mpn:
        v.append("PARTITION_DRIFT:no_PNAME")
    else:
        mf_ = PKGROOT / "raw" / "rl_pool" / "partition_runs" / (mpn.group(1) + ".manifest.json")
        if not mf_.is_file() or RC.sha256_file(str(mf_)) != RC.PARTITION["manifest_sha256"]:
            v.append("PARTITION_DRIFT:manifest_sha256")
        if f"raw/rl_pool/partition_runs/{mpn.group(1)}.manifest.json" not in text:
            v.append("PARTITION_DRIFT:required_files")
    mep = re.search(r"--epochs (\S+)", fn_body(text, "arm_args"))
    if not mep or not RC.updates_check(f"--max-updates 1000 --epochs {mep.group(1)}", RC.PARTITION["n_train"])[0]:
        v.append("UPDATES_BUDGET:" + (mep.group(1) if mep else "none"))
    if "--plan-only" not in text or not re.search(r'grep -q "\^ZZRLFAN_DONE 0 0 0\$" "\$R/plan\.log" \|\| refuse', text):
        v.append("NO_PLAN_ONLY")
    if '"$prc" = 3' not in text or "rl_pool_available.py" not in text or "ZZRLS1_REMEDY" not in text:
        v.append("NO_REMEDY")
    elif not re.search(r'ZZRLS1_REMEDY.*?refuse "image-coverage gap', text, re.S):
        v.append("REMEDY_DOES_NOT_STOP")
    # ---- the dump is model_final's frozen fan; labels on every proposal
    if 'rl_fan_dump.py --ckpt "$FINAL"' not in text:
        v.append("DUMP_NOT_FROM_MODEL_FINAL")
    if '--expect-ckpt-sha256 "$FINAL_SHA256"' not in text:
        v.append("DUMP_SHA_NOT_PINNED")
    if re.search(r"^[^#\n]*--served-only", text, re.M):
        v.append("LABELS_SERVED_ONLY")
    if not re.search(r'pdm_label_complete\.py.*?--all-versions', text, re.S) or not re.search(r'grep -q "\^ZZPDM_COMPLETE" .*? \|\| \{ log "REFUSED', text):
        v.append("PDM_COMPLETE_UNCHECKED")
    if not re.search(r'grep -q "\^ZZRELABEL_RLT_DONE \[0-9\]\* 0 0\$"', text):
        v.append("TERMS_DONE_UNCHECKED")
    if not re.search(r'grep -q "\^ZZPDM_SETUP_OK" "\$R/pdm_setup\.log" \|\| refuse', text):
        v.append("PDM_SETUP_NOT_CHECKED")
    # ---- the join gate
    jg = len(re.findall(r'rls1_checks\.py join-gate --side "\$side"', text))
    if jg != 1 or '"$side" = train' not in text:
        v.append("NO_JOIN_GATE")
    if not re.search(r'grep -q "\^ZZRLS1_JOIN_OK" "\$R/join_\$side\.log" \|\| refuse', text):
        v.append("JOIN_UNCHECKED")
    if text.find("rls1_checks.py join-gate") > text.find("--tag rlfan_data"):
        v.append("ORDER:join_after_bank")
    # ---- banking
    if " ZZEGRESS_OK rlfan_data " not in text:
        v.append("BANK_UNCHECKED")
    if " ZZEGRESS_OK arm_$ARM " not in text:
        v.append("ARM_EGRESS_UNCHECKED")
    if not re.search(r"hf_egress\.py watch .*?--halt-pid", text, re.S):
        v.append("NO_WATCHER")
    if 'rls1_checks.py monitor --arm "$ARM"' not in text:
        v.append("NO_KILL_MONITOR")
    if 'rls1_checks.py replay --arm "$ARM"' not in text or "ZZRLS1_REPLAY_CLEAN" not in text:
        v.append("NO_KILL_REPLAY")
    if "ZZRLS1_ARM_KILLED" not in text or 'arm_${ARM}_killed' not in text or text.count('"$S/arm_${ARM}_killed" ] && ') < 1:
        v.append("KILLED_ARM_NOT_HELD")
    if "ZZRLS1_DONE" not in text:
        v.append("NO_DONE_MARKER")
    if not re.search(r"hf_egress\.py push --repo \"\$OUT_REPO\" --prefix logs --tag chain_tick", text):
        v.append("NO_TICKER")
    # ---- arms
    if not re.search(r'\$\{RLS1_ARMS:-C0 R0 C1 R1 R2\}', text):
        v.append("CONTROL_NOT_BEFORE_R")
    ab = fn_body(text, "arm_args")
    for a in ("C0", "C1"):
        if "epo" in arm_block(text, a):
            v.append("C_ARM_HAS_EPO:" + a)
    for a in ("R0", "R1", "R2"):
        if "--epo --epo-terms-train $TERMS_TR --epo-terms-heldout $TERMS_VA" not in arm_block(text, a):
            v.append("R_ARM_LACKS_EPO:" + a)
    if "--epo-k --epo-l" not in arm_block(text, "R2") or any("--epo-k" in arm_block(text, a) or "--epo-l" in arm_block(text, a) for a in ("R0", "R1")):
        v.append("ARM_K_L_WRONG")
    if "--eval-every 250" not in ab:
        v.append("ARGV_NO_EVAL_EVERY")
    if "--dump-scores $ad/scores" not in ab:
        v.append("ARGV_NO_DUMP_SCORES")
    if '--ckpt $BASE ' not in ab:
        v.append("ARGV_NOT_FROM_U400")
    if len(re.findall(r'^  ARMDIR=\$R/arms/\$ARM; LG=\$R/lg/\$ARM; ARGS="\$\(arm_args "\$ARM"\)"', text, re.M)) != 1 or text.count("ARGS=") != 1:
        v.append("ARGS_DEFINED_TWICE_OR_NOT_ONCE")
    if '--arm-name "$ARM"' not in text:
        v.append("NO_ARM_NAME")
    m = TOKEN_OK_RX.search(text)
    if not m:
        v.append("NO_TOKEN_OK_FUNCTION")
    else:
        body = m.group(0)
        for needle in ('launch_gate.py" verify', '--profile "$PROFILE"', '--args "$ARGS"', '--pkg "$PKG"', '--token "$tok"'):
            if needle not in body:
                v.append("TOKEN_OK_NOT_VERIFYING:" + needle)
        if re.search(r"\bgrep\b|^\s*return 0|\btrue\b", body, re.M):
            v.append("TOKEN_OK_NOT_VERIFYING:grep_or_constant")
    lp = text.find("scorer_finetune.py $ARGS >>")
    if "token_ok || refuse" not in text[max(0, lp - 260):lp]:
        v.append("TRAIN_WITHOUT_TOKEN")
    if text.count("token_ok ||") < 3:
        v.append("TOKEN_CHECKS_MISSING")
    for ln in text.split("\n"):
        if not ln.strip().startswith("#") and "grep" in ln and ("ZZGATE-" in ln or "launch_gate.log" in ln):
            v.append("TOKEN_GREPPED:" + ln.strip()[:60])
    if "set -m" not in text or '[ "$PG" != "$TPID" ]' not in text:
        v.append("TRAINER_NOT_OWN_PROCESS_GROUP")
    if not re.search(r'^export TANITAD_GATE_KEY_FILE="\$R/', text, re.M):
        v.append("KEY_NOT_OUTSIDE_TREE")
    if tar_owner_violations(text):
        v.append("TAR_OWNER")
    fl = re.search(r'for f in (.*?); do\n  \[ -f "\$PKG/\$f" \] \|\| refuse', text, re.S)
    if not fl:
        v.append("REQUIRED_FILES_LOOP_MISSING")
    else:
        for f in fl.group(1).replace("\\\n", " ").split():
            if not (PKGROOT / f).is_file():
                v.append("REQUIRED_FILE_ABSENT:" + f)
        if "SHIP_MANIFEST.json" not in fl.group(1).split():
            v.append("SHIP_MANIFEST_NOT_REQUIRED")
        v += ["SAFEOBJ_FILE_NOT_REQUIRED:" + f for f in SAFEOBJ_FILES if f not in fl.group(1).split()]
        gate_loads = sorted(set(re.findall(r'EVAL_DIR / "([A-Za-z0-9_]+\.py)"', (HERE / "rls1_checks.py").read_text(encoding="utf-8"))))          # every file the chain's helper loads from eval/ BY PATH
        v += ["GATE_LOADS_A_FILE_THE_PRECHECK_DOES_NOT_REQUIRE:eval/" + f for f in gate_loads if "eval/" + f not in fl.group(1).split()]
    v += check_fancmp(text)
    v += check_queue(text)
    v += check_rls1c(text)
    return v


def mutations(text):
    sw = lambda a, b: text.replace(a, b)  # noqa: E731
    out = [
        ("no P2 lock check", drop_line(text, '9<> "$P2/chain.lock"'), "NO_P2_LOCK_CHECK"),
        ("no GPU check in the queue", sw("if gpu_busy; then idle=0; why=", "if false; then idle=0; why="), "NO_GPU_CHECK"),
        ("p2_over judges the GPU again", sw("  if ! grep -q \"ZZP2R_DONE\" \"$P2/chain.log\"", "  if gpu_busy; then echo busy; return 1; fi\n  if ! grep -q \"ZZP2R_DONE\" \"$P2/chain.log\""), "P2_OVER_JUDGES_THE_GPU"),
        ("GPU unknown counts as idle", sw("|| return 0\n  [ -n", "|| return 1\n  [ -n"), "NO_GPU_CHECK"),
        ("no P2 finished check", sw('grep -q "ZZP2R_DONE" "$P2/chain.log"', 'grep -q "ZZNOTHING" "$P2/chain.log"'), "NO_P2_DONE_CHECK"),
        ("p2_over never called", sw('why="$(p2_over)" || refuse "$why"', "true"), "P2_OVER_NOT_CALLED"),
        ("pgrep -f", sw("gpu_busy() {", "gpu_busy() {\n  pgrep -f scorer_finetune > /dev/null && return 0"), "PGREP_SELF_MATCH"),
        ("PKG is P2's directory", sw("PKG=$WORK/refe-plan-rls1", "PKG=$WORK/refe-plan"), "PKG_IS_NOT_ITS_OWN_DIR"),
        ("model_final not verified", drop_line(text, "--pin model_final"), "CKPT_NOT_VERIFIED:model_final"),
        ("u400 not verified", drop_line(text, "--pin u400"), "CKPT_NOT_VERIFIED:u400"),
        ("model_final verdict unchecked", drop_line(text, 'grep -q "^ZZRLS1_FILE_OK" "$R/ckpt_final.log"'), "CKPT_RESULT_UNCHECKED:model_final"),
        ("u400 verdict unchecked", sw('grep -q "^ZZRLS1_FILE_OK" "$R/ckpt_u400.log" || refuse', 'grep -q "^ZZRLS1_FILE_OK" "$R/ckpt_u400.log" || true'),
         "CKPT_RESULT_UNCHECKED:u400"),
        ("model_final sha256 pin edited", sw("FINAL_SHA256=4246a320", "FINAL_SHA256=4246a321"), "PIN_MISMATCH"),
        ("partition verdict unchecked", drop_line(text, 'grep -q "^ZZRLS1_PARTITION_OK"'), "PARTITION_UNCHECKED"),
        ("a label override is back", sw("TLANE_TR=$R/tlane_train;", 'TLANE_TR="${RLS1_TLANE_TRAIN:-$R/tlane_train}";'), "TLANE_OVERRIDE_PRESENT"),
        ("no labeller smoke", re.sub(r"(?s)# ---- 6a\..*?stop_after tlane\n", "", text), 'MISSING:rl_tlane_label.py --sets "$SETS_VA"'),
        ("smoke verdict unchecked", drop_line(text, 'grep -q "^ZZTLANE_DONE 3 0 0$"'), "TLANE_SMOKE_UNCHECKED"),
        ("smoke rows unchecked", drop_line(text, 'grep -q "^ZZRLS1_TLANE_OK" "$R/tlane_smoke_check.log"'), "TLANE_SMOKE_ROWS_UNCHECKED"),
        ("exact verdict unchecked", drop_line(text, 'grep -q "^ZZRLS1_TLANE_OK" "$R/tlane_exact_$side.log"'), "TLANE_EXACT_UNCHECKED"),
        ("exact gate without the partition's count", sw(' --expect "$ex" --json "$R/tlane_$side.json"', ' --json "$R/tlane_$side.json"'), "TLANE_EXACT_UNCHECKED"),
        ("exact gate deleted", sw("rls1_checks.py tlane-exact", "rls1_checks.py tlane-skipped"), "MISSING:rls1_checks.py tlane-exact"),
        ("tlane labels unaudited", drop_line(text, 'grep -q "^ZZTLANE_DONE [0-9]* [0-9]* 0$"'), "TLANE_LABELS_UNAUDITED"),
        ("tlane labeller without the repair", sw('--shard "$2/$3" \\\n      --repair-last-heading --workers "$LW"', '--shard "$2/$3" \\\n      --workers "$LW"'), "TLANE_NO_REPAIR"),
        ("tlane labeller reads another bank", sw('--sets "$4" --bank "$TARGETS" --out "$7"', '--sets "$4" --bank "$IMAGES" --out "$7"'), "TLANE_BANK_NOT_THE_TRAINING_BANK"),
        ("tlane labeller without REFE_SIM_HZ", sw("OMP_NUM_THREADS=1 REFE_SIM_HZ=10 nice", "OMP_NUM_THREADS=1 nice"), "TLANE_NO_SIM_HZ"),
        ("tlane labeller not tied to the partition", sw(' --keys "$8" --shard', " --shard"), "TLANE_KEYS_OR_SHARD_NOT_PINNED"),
        ("tlane labeller keeps fd 200", sw('--workers "$LW" ) >> "$R/$m.tlane.log" 2>&1 200>&-', '--workers "$LW" ) >> "$R/$m.tlane.log" 2>&1'), "FD200"),
        ("the lane labels are not banked", sw(" terms_val tlane_train tlane_heldout; do", " terms_val; do"), "BANK_LACKS_TLANE"),
        ("the update budget is unchecked", drop_line(text, 'grep -q "^ZZRLS1_UPDATES_OK"'), "UPDATES_UNCHECKED"),
        ("--epochs back to 2 on 16,000 scenes", sw("--max-updates 1000 --epochs 4 --workers 3", "--max-updates 1000 --epochs 2 --workers 3"), "UPDATES_BUDGET:2"),
        ("the join gate still expects 48,000", sw("tsv=$TRAIN_TSV; ex=16000", "tsv=$TRAIN_TSV; ex=48000"), "PARTITION_DRIFT:join_gate_counts"),
        ("the remedy re-cuts at 48,000", sw("--n-train 16000 --n-val 3000", "--n-train 48000 --n-val 3000"), "PARTITION_DRIFT:remedy_params"),
        ("the remedy logs the old id", sw("old partition_id 9995a39aeeac", "old partition_id 272cfd357b81"), "PARTITION_DRIFT:remedy_old_id"),
        ("the chain reads the 48,000 partition", sw("PNAME=A_no_trained_list_seed20261012_n16000\n", "PNAME=A_no_trained_list_seed20261012\n"), "PARTITION_DRIFT:manifest_sha256"),
        ("no coverage plan", drop_line(text, "rl_fan_dump.py --plan-only"), "MISSING:rl_fan_dump.py --plan-only"),
        ("plan verdict unchecked", drop_line(text, 'grep -q "^ZZRLFAN_DONE 0 0 0$"'), "NO_PLAN_ONLY"),
        ("no remedy", sw('"$prc" = 3', '"$prc" = 99'), "NO_REMEDY"),
        ("the remedy continues instead of stopping", re.sub(r'    refuse "image-coverage gap on this pod[^\n]*\n', "", text), "REMEDY_DOES_NOT_STOP"),
        ("dump from u400", sw('rl_fan_dump.py --ckpt "$FINAL"', 'rl_fan_dump.py --ckpt "$BASE"'), "DUMP_NOT_FROM_MODEL_FINAL"),
        ("dump checkpoint not hash-pinned", sw(' --expect-ckpt-sha256 "$FINAL_SHA256"', ""), "DUMP_SHA_NOT_PINNED"),
        ("labels with --served-only", sw('--files "$f" --workers "$LW" ) >> "$R/$m.pdm.log"', '--files "$f" --served-only --workers "$LW" ) >> "$R/$m.pdm.log"'),
         "LABELS_SERVED_ONLY"),
        ("PDM completeness unaudited", drop_line(text, 'grep -q "^ZZPDM_COMPLETE"'), "PDM_COMPLETE_UNCHECKED"),
        ("rl_terms completeness unaudited", drop_line(text, "ZZRELABEL_RLT_DONE"), "TERMS_DONE_UNCHECKED"),
        ("NAVSIM setup unchecked", drop_line(text, 'grep -q "^ZZPDM_SETUP_OK"'), "PDM_SETUP_NOT_CHECKED"),
        ("join gate deleted", re.sub(r"(?s)# ---- 7\. THE JOIN GATE.*?stop_after join\n", "", text), "NO_JOIN_GATE"),
        ("join verdict unchecked", drop_line(text, 'grep -q "^ZZRLS1_JOIN_OK"'), "JOIN_UNCHECKED"),
        ("join gate after the bank", (lambda b: text.replace(b, "") + "\n" + b)(re.search(r"# ---- 7\. THE JOIN GATE.*?stop_after join\n", text, re.S).group(0)),
         "ORDER"),
        ("bank unchecked", drop_line(text, "ZZEGRESS_OK rlfan_data"), "BANK_UNCHECKED"),
        ("arm evidence unchecked", drop_line(text, "ZZEGRESS_OK arm_$ARM"), "ARM_EGRESS_UNCHECKED"),
        ("no checkpoint watcher", re.sub(r'    "\$PY" hf_egress\.py watch .*?200>&- &\n', "", text, flags=re.S), "NO_WATCHER"),
        ("no kill monitor", drop_line(text, 'rls1_checks.py monitor --arm "$ARM"'), "NO_KILL_MONITOR"),
        ("no kill replay", drop_line(text, 'rls1_checks.py replay --arm "$ARM"'), "NO_KILL_REPLAY"),
        ("killed arms not held", sw("ZZRLS1_ARM_KILLED", "ZZRLS1_ARM_LOGGED"), "KILLED_ARM_NOT_HELD"),
        ("no ticker", drop_line(text, "--tag chain_tick"), "NO_TICKER"),
        ("no done marker", sw("ZZRLS1_DONE", "ZZRLS1_OVER"), "NO_DONE_MARKER"),
        ("an R arm before its control", sw("${RLS1_ARMS:-C0 R0 C1 R1 R2}", "${RLS1_ARMS:-R0 C0 R1 C1 R2}"), "CONTROL_NOT_BEFORE_R"),
        ("C0 carries --epo", sw("C0) seed=0 ;;", 'C0) seed=0; epo="--epo" ;;'), "C_ARM_HAS_EPO:C0"),
        ("C1 carries an EPO lever at its default", sw("C1) seed=1 ;;", 'C1) seed=1; epo="--epo-temp 0.1" ;;'), "C_ARM_HAS_EPO:C1"),
        ("R0 loses --epo", sw('R0) seed=0; epo="--epo --epo-terms-train $TERMS_TR --epo-terms-heldout $TERMS_VA" ;;', "R0) seed=0 ;;"), "R_ARM_LACKS_EPO:R0"),
        ("R0 carries K and L", sw('R0) seed=0; epo="--epo --epo-terms-train $TERMS_TR --epo-terms-heldout $TERMS_VA" ;;',
                                  'R0) seed=0; epo="--epo --epo-terms-train $TERMS_TR --epo-terms-heldout $TERMS_VA --epo-k --epo-l" ;;'), "ARM_K_L_WRONG"),
        ("R2 loses K and L", sw(" --epo-k --epo-l\" ;;", '" ;;'), "ARM_K_L_WRONG"),
        ("no 250-update checks", sw(" --eval-every 250", ""), "ARGV_NO_EVAL_EVERY"),
        ("no score dump", sw(" --dump-scores $ad/scores", ""), "ARGV_NO_DUMP_SCORES"),
        ("arms train from model_final", sw("--ckpt $BASE --backbone", "--ckpt $FINAL --backbone"), "ARGV_NOT_FROM_U400"),
        ("ARGS defined twice", sw('  mkdir -p "$ARMDIR" "$LG"\n', '  mkdir -p "$ARMDIR" "$LG"\n  ARGS="$(arm_args "$ARM")"\n'), "ARGS_DEFINED_TWICE_OR_NOT_ONCE"),
        ("the gate is minted without an arm", sw(' --arm-name "$ARM"', ""), "NO_ARM_NAME"),
        ("token_ok always true", re.sub(r'(?s)(token_ok\(\) \{\n).*?(\n\}\n)', r'\1  return 0\2', text, count=1), "TOKEN_OK_NOT_VERIFYING"),
        ("token_ok greps the gate log", re.sub(r'(?s)(token_ok\(\) \{\n).*?(\n\}\n)', r'\1  grep -q ZZGATE-PASS "$ARMDIR/launch_gate.log"\2', text, count=1),
         "TOKEN_OK_NOT_VERIFYING"),
        ("token_ok verifies another argv", sw('--args "$ARGS" --pkg "$PKG" \\', '--args "--lr 1" --pkg "$PKG" \\'), "TOKEN_OK_NOT_VERIFYING"),
        ("token_ok forgets the profile", sw('verify --profile "$PROFILE" --token', "verify --token"), "TOKEN_OK_NOT_VERIFYING"),
        ("the trainer starts without the token check", drop_line(text, "scorer_finetune.py is NOT started"), "TRAIN_WITHOUT_TOKEN"),
        ("the gate log is grepped for PASS", sw('    token_ok || refuse "the launch gate did not mint', '    grep -q "^ZZGATE-PASS-" "$ARMDIR/launch_gate.log" || refuse "x"\n    token_ok || refuse "the launch gate did not mint'),
         "TOKEN_GREPPED"),
        ("the trainer keeps fd 200", sw('"$ARMDIR/sft.log" 2>&1 ) 200>&- &', '"$ARMDIR/sft.log" 2>&1 ) &'), "FD200"),
        ("the gate child keeps fd 200", sw('> "$ARMDIR/launch_gate.log" 2>&1 200>&-', '> "$ARMDIR/launch_gate.log" 2>&1'), "FD200"),
        ("the trainer is not its own process group", sw("set -m\n", ""), "TRAINER_NOT_OWN_PROCESS_GROUP"),
        ("the manifest is not required", sw(" SHIP_MANIFEST.json \\", " \\"), "SHIP_MANIFEST_NOT_REQUIRED"),
        ("the S-reader is not required by the precheck", sw(" eval/safeobj_read.py", ""), "SAFEOBJ_FILE_NOT_REQUIRED:eval/safeobj_read.py"),
        ("the S-reader's own test is not required by the precheck", sw(" eval/test_safeobj_read.py", ""), "SAFEOBJ_FILE_NOT_REQUIRED:eval/test_safeobj_read.py"),
        ("the S-gate's test is not required by the precheck", sw(" eval/test_safeobj_gate.py", ""), "SAFEOBJ_FILE_NOT_REQUIRED:eval/test_safeobj_gate.py"),
        ("the helper loads eval/safeobj_read.py by path and the precheck does not require it", sw(" eval/safeobj_read.py", ""), "GATE_LOADS_A_FILE_THE_PRECHECK_DOES_NOT_REQUIRE:eval/safeobj_read.py"),
        ("the key lives in the shipped tree", sw('export TANITAD_GATE_KEY_FILE="$R/launch_gate.key"', 'export TANITAD_GATE_KEY_FILE="$PKG/launch_gate.key"'), "KEY_NOT_OUTSIDE_TREE"),
        ("echo the token", sw('log "env ok', 'log "env ok token=$HF_TOKEN'), "HF_TOKEN_USE"),
        ("xtrace on", sw("set -u\n", "set -ux\n", ), "XTRACE"),
        ("the 2026-10-11 mangled continuation", sw("code/pod_teacher_env.sh code/pod_dataprep.sh", "code/pod_teacher_env.sh \\n code/pod_dataprep.sh"), "REQUIRED_FILE_ABSENT:\\n"),
        ("a required file the package does not ship", sw("refe/rl_partition.py ", "refe/rl_partition_gone.py "), "REQUIRED_FILE_ABSENT:refe/rl_partition_gone.py"),
        ("preflight after the gate", (lambda a, b: text.replace(a + b, b + a))(*[re.search(p, text, re.S).group(0) for p in
            (r"  # ---- 9a\..*?(?=  # ---- 9b)", r"  # ---- 9b\..*?(?=  # ---- 9c)")]), "ORDER"),
        ("tar without --no-same-owner", sw("log \"env ok", "tar -xf x.tar -C /tmp\nlog \"env ok"), "TAR_OWNER"),
    ]
    out += [
        # ---- the queue and the fail-closed supervision (2026-10-14)
        ("the queue is never called", sw("\nqueue_wait\nlog", "\ntrue\nlog"), "QUEUE_NOT_CALLED"),
        ("the queue clears AFTER the disk test", (lambda b: text.replace(b, "").replace('log "dd write test ok (${DDMB} MB)"\n', 'log "dd write test ok (${DDMB} MB)"\n' + b))(
            re.search(r"queue_wait\nlog \"ZZRLS1_QUEUE_CLEAR[^\n]*\n", text).group(0)), "ORDER:queue_after"),
        ("chain.lock is taken at the top (held while queued)", sw('exec 199>"$R/chain.queue"', 'exec 200>"$R/chain.lock"\nexec 199>"$R/chain.queue"'), "CHAIN_LOCK_TAKEN_OUTSIDE_QUEUE"),
        ("the queue never gives chain.lock back", sw("exec 200>&-; held=0;", "held=0;"), "QUEUE_NEVER_YIELDS_THE_LOCK"),
        ("the queue does not check its own lock", sw('flock -n 200 || refuse "another process holds', 'flock -n 200 || true "another process holds'), "QUEUE_LOCK_NOT_CHECKED"),
        ("no single-instance guard", drop_line(text, "flock -n 199 ||"), "QUEUE_NO_INSTANCE_GUARD"),
        ("S1 is not ahead", drop_line(text, '  if lock_held "$S1"; then echo'), "AHEAD_LACKS:lock_held \"$S1\""),
        ("P2 is not ahead", drop_line(text, '  if lock_held "$P2"; then echo "the P2 chain holds'), "AHEAD_LACKS:lock_held \"$P2\""),
        ("a WAITING replicate is ahead (deadlock with it)", sw('chain_past "$P2REP" "ZZP2REP_START" "ZZP2REP_WAIT_OK"', 'lock_held "$P2REP"'), "AHEAD_LACKS:chain_past \"$P2REP\""),
        ("P6 is not ahead once past its wait", drop_line(text, '  if chain_past "$P6D"'), "AHEAD_LACKS:chain_past \"$P6D\""),
        ("past-wait ignores the attempt boundary", sw("index($0, s) { ok = 0 } ", ""), "CHAIN_PAST_NOT_PER_ATTEMPT"),
        ("the lock probe is read-only", sw('( flock -n 9 ) 9<> "$1/chain.lock"; }', '( flock -n 9 ) 9< "$1/chain.lock"; }'), "LOCK_PROBE_NOT_READ_WRITE"),
        ("the queue skips the GPU polls", sw('[ "$idle" -ge "$need" ] && return 0; fi\n    fi', 'return 0; fi\n    fi'), "QUEUE_NO_GPU_POLLS"),
        ("no settle window", sw('"${RLS1_SETTLE_S:-20}"', '0'), "QUEUE_NO_GPU_POLLS"),
        ("the arm list is not validated before the queue", drop_line(text, "for ARM in ${RLS1_ARMS:-C0 R0 C1 R1 R2}; do arm_registered"), "ARMS_NOT_VALIDATED_EARLY"),
        ("no GPU idle wait before a trainer", drop_line(text, "gpu_idle_wait || refuse"), "NO_GPU_IDLE_BEFORE_TRAINER"),
        ("the trainer is not supervised", drop_line(text, 'supervise_arm "$ARM" "$TPID" "$MPID"'), "NO_SUPERVISOR"),
        ("the fail-closed branch is removed", sw('if pid_alive "$tp"; then why="kill rules unenforceable: monitor died rc $MON_RC"; fi', ":"), "KMON_NO_DEAD_MONITOR_BRANCH"),
        ("the READY timeout is removed", sw('[ $((SECONDS - t0)) -ge "$ready_s" ]', "false"), "KMON_NO_READY_TIMEOUT"),
        ("the trainer is not stopped", sw('  stop_trainer "$tp"\n', ""), "KMON_TRAINER_NOT_STOPPED"),
        ("K-MON writes no KILLED.json", sw('"code": "K-MON"', '"code": "K-X"'), "KMON_NO_KILLED_JSON"),
        ("K-MON overwrites a real kill", sw('[ -f "$ad/KILLED.json" ] || printf', "printf"), "KMON_OVERWRITES_A_KILL"),
        ("the trainer is stopped by name", sw('kill -TERM -- "-$pg"', "pkill -f scorer_finetune"), "STOP_NOT_BY_EXPLICIT_PGID"),
        ("the stop may signal the chain's own group", sw('{ [ "$pg" -gt 1 ] && [ "$pg" != "$$" ]; } || return 1', "true"), "STOP_NOT_BY_EXPLICIT_PGID"),
        ("the reaped monitor rc is lost", sw('if [ -n "${MON_RC:-}" ]; then MRC=$MON_RC; else wait "$MPID"; MRC=$?; fi', 'wait "$MPID"; MRC=$?'), "MON_RC_NOT_CARRIED"),
        ("monitor.log is not banked", sw('"$ARMDIR"/kill_checks.jsonl "$ARMDIR"/monitor.log', '"$ARMDIR"/kill_checks.jsonl'), "MONITOR_LOG_NOT_BANKED"),
        ("a killed arm re-trains", sw(' && [ ! -f "$S/arm_${ARM}_killed" ]; then', "; then"), "KILLED_ARM_RETRAINS"),
        ("a re-run keeps the stale read / egress markers", drop_line(text, 'rm -f "$S/read_$ARM.ok" "$S/egress_$ARM.ok"'), "STALE_MARKERS_KEPT"),
        ("an insertion point is gone", sw("RLS1B-INSERTION-POINT-2", "NOTHING"), "NO_INSERTION_POINT_2"),
        ("an RL-S1b arm is invented", sw("    # RLS1B-INSERTION-POINT-2", "    R2b) seed=0 ;;\n    # RLS1B-INSERTION-POINT-2"), "INVENTED_RLS1B_ARGV"),
        ("R0b / R1b are not registered", sw('REGISTERED_ARMS="C0 C1 R0 R1 R2 R0b R1b R0c R0d R1d R1o"', 'REGISTERED_ARMS="C0 C1 R0 R1 R2 R0c R0d R1d R1o"'), "RLS1B_ARMS_NOT_REGISTERED"),
        ("R0b is judged under the original profile", sw("R0b|R1b) echo refe_rls1b", "R0b|R1b) echo refe_rls1"), "NO_RLS1B_PROFILE"),
        ("a missing control is silently replaced by the arm's own A", sw('refuse "RL-S1b arm $ARM is read against $RD', 'refuse "x $ARM $RD'), "RLS1B_CONTROL_NOT_ENFORCED"),
        ("the in-loop arm check is gone", sw('  arm_registered "$ARM" || refuse "unknown arm \'$ARM\'"\n', ""), "ARM_NAMES_NOT_REFUSED_TWICE"),
    ]
    atol_line = re.search(r"^FANCMP_ATOL=.*$", text, re.M).group(0)
    out += [
        # ---- the fan-equivalence gate (stage fancmp)
        ("no fan-equivalence stage", re.sub(r"(?s)# ---- 4b\..*?stop_after fancmp\n", "", text), "FANCMP_STAGE_MISSING"),
        ("the tolerance flag is dropped", sw(' --compare-atol "$FANCMP_ATOL"', ""), "FANCMP_NO_ATOL"),
        ("the comparison itself is dropped", sw('--compare-sets "$BANKED" ', ""), "FANCMP_NO_COMPARE_SETS"),
        ("the matched-key floor is dropped", sw(' --compare-min-matched "$NK"', ""), "FANCMP_NO_MIN_MATCHED"),
        ("the verdict is not read", drop_line(text, "ZZRLFAN_ATOL_PASS $NK"), "FANCMP_VERDICT_UNCHECKED"),
        ("the verdict reads the marker but not the exit", sw('{ [ "$frc" = 0 ] && grep -q', '{ [ "$frc" != 99 ] && grep -q'), "FANCMP_VERDICT_UNCHECKED"),
        ("the gate tolerates missing images", sw("--tag fancmp --batch 4", "--tag fancmp --batch 4 --allow-missing"), "FANCMP_NARROWED:--allow-missing"),
        ("the gate dumps only a smoke", sw("--tag fancmp --batch 4", "--tag fancmp --batch 4 --limit 8"), "FANCMP_NARROWED:--limit"),
        ("the gate dumps from u400", sw('rl_fan_dump.py --ckpt "$FINAL" --expect-ckpt-sha256 "$FINAL_SHA256" --keys "$FANCMP_KEYS"',
                                         'rl_fan_dump.py --ckpt "$BASE" --expect-ckpt-sha256 "$FINAL_SHA256" --keys "$FANCMP_KEYS"'), "FANCMP_NOT_FROM_MODEL_FINAL"),
        ("the gate's checkpoint is not hash-pinned", sw(' --expect-ckpt-sha256 "$FINAL_SHA256" --keys "$FANCMP_KEYS"', ' --keys "$FANCMP_KEYS"'), "FANCMP_SHA_NOT_PINNED"),
        ("the gate child keeps fd 200", sw('--compare-min-matched "$NK" >> "$R/fancmp.log" 2>&1 200>&-', '--compare-min-matched "$NK" >> "$R/fancmp.log" 2>&1'), "FANCMP_FD200"),
        ("the gate child inherits the HF token", sw('env -u HF_TOKEN OMP_NUM_THREADS=1 "$PY" rl_fan_dump.py --ckpt "$FINAL" --expect-ckpt-sha256 "$FINAL_SHA256" --keys "$FANCMP_KEYS"',
                                                     'OMP_NUM_THREADS=1 "$PY" rl_fan_dump.py --ckpt "$FINAL" --expect-ckpt-sha256 "$FINAL_SHA256" --keys "$FANCMP_KEYS"'),
         "FANCMP_TOKEN_NOT_STRIPPED"),
        ("the gate compares the RL partition instead of its slice", sw('--keys "$FANCMP_KEYS" --targets', '--keys "$TRAIN_TSV" --targets'), "FANCMP_KEYS_NOT_THE_SLICE"),
        ("the gate reads another bank", sw('--targets "$TARGETS" \\\n    --images "$IMAGES" --calib "$CALIB" --out "$R/fancmp"', '--targets "$R" \\\n    --images "$IMAGES" --calib "$CALIB" --out "$R/fancmp"'),
         "FANCMP_WRONG_BANK"),
        ("the slice pin is edited", sw("FANCMP_KEYS_SHA256=" + "4c613ff0", "FANCMP_KEYS_SHA256=" + "4c613ff1"), "FANCMP_KEYS_PIN_DRIFT"),
        ("the slice pin is not enforced", drop_line(text, 'sha256sum "$FANCMP_KEYS"'), "FANCMP_KEYS_UNPINNED"),
        ("the slice size is a constant", sw('NK=$(( $(wc -l < "$FANCMP_KEYS") - 1 ))', "NK=1"), "FANCMP_NK_NOT_THE_SLICE_SIZE"),
        # ---- the TWO bounds are pinned constants (round 2 PASS), the env vars are guards, the done-marker is bound to the bounds
        ("the mean flag is dropped", sw(' --compare-mean-atol "$FANCMP_MEAN_ATOL"', ""), "FANCMP_NO_MEAN_ATOL"),
        ("the mean verdict is not read", sw(' && grep -qE "^ZZRLFAN_MEAN_ATOL_PASS $NK [^ ]+ ${FANCMP_MEAN_ATOL}\\$" "$R/fancmp.log"', ""), "FANCMP_VERDICT_UNCHECKED"),
        ("the max verdict is not read", sw(' && grep -qE "^ZZRLFAN_ATOL_PASS $NK [^ ]+ ${FANCMP_ATOL}\\$" "$R/fancmp.log"', ""), "FANCMP_VERDICT_UNCHECKED"),
        ("a PASS marker at ANOTHER max bound is accepted", sw('[^ ]+ ${FANCMP_ATOL}\\$"', '[^ ]+ [0-9.]+\\$"'), "FANCMP_VERDICT_UNCHECKED"),
        ("a PASS marker at ANOTHER mean bound is accepted", sw('[^ ]+ ${FANCMP_MEAN_ATOL}\\$"', '[^ ]+ [0-9.]+\\$"'), "FANCMP_VERDICT_UNCHECKED"),
        ("the mean marker's key count is not the slice's", sw('"^ZZRLFAN_MEAN_ATOL_PASS $NK [^ ]+', '"^ZZRLFAN_MEAN_ATOL_PASS [0-9]+ [^ ]+'), "FANCMP_VERDICT_UNCHECKED"),
        ("the catastrophe bound is edited", sw("FANCMP_ATOL=0.5\n", "FANCMP_ATOL=0.9\n"), "FANCMP_ATOL_DRIFT"),
        ("the catastrophe bound is loosened past the measured one", sw("FANCMP_ATOL=0.5\n", "FANCMP_ATOL=1.0\n"), "FANCMP_ATOL_DRIFT"),
        ("the mean bound is loosened", sw("FANCMP_MEAN_ATOL=0.025\n", "FANCMP_MEAN_ATOL=0.05\n"), "FANCMP_MEAN_ATOL_DRIFT"),
        ("the mean bound is tightened without a verdict", sw("FANCMP_MEAN_ATOL=0.025\n", "FANCMP_MEAN_ATOL=0.0125\n"), "FANCMP_MEAN_ATOL_DRIFT"),
        ("the catastrophe bound is an input again", sw("FANCMP_ATOL=0.5\n", 'FANCMP_ATOL="${RLS1_FANCMP_ATOL:-0.5}"\n'), "FANCMP_ATOL_NOT_PINNED"),
        ("the mean bound is an input", sw("FANCMP_MEAN_ATOL=0.025\n", 'FANCMP_MEAN_ATOL="${RLS1_FANCMP_MEAN_ATOL:-0.025}"\n'), "FANCMP_MEAN_ATOL_NOT_PINNED"),
        ("the catastrophe bound is re-assigned from the env", sw('BANKED="${RLS1_BANKED_SETS', 'FANCMP_ATOL="${RLS1_FANCMP_ATOL}"\nBANKED="${RLS1_BANKED_SETS'), "FANCMP_ATOL_NOT_ASSIGNED_ONCE"),
        ("the argv reads the env, not the pin", sw('--compare-atol "$FANCMP_ATOL"', '--compare-atol "${RLS1_FANCMP_ATOL:-0.9}"'), "FANCMP_NO_ATOL"),
        ("a stale max-bound export is not refused", drop_line(text, '[ -z "${RLS1_FANCMP_ATOL:-}" ]'), "RLS1_FANCMP_ATOL_NOT_GUARDED"),
        ("a stale mean-bound export is not refused", drop_line(text, '[ -z "${RLS1_FANCMP_MEAN_ATOL:-}" ]'), "RLS1_FANCMP_MEAN_ATOL_NOT_GUARDED"),
        ("the export guard accepts any number", sw("a + 0 == p + 0", "a + 0 > 0"), "RLS1_FANCMP_ATOL_NOT_GUARDED"),
        ("the export guard is advisory", sw("|| refuse \"RLS1_FANCMP_MEAN_ATOL=", "|| true \"RLS1_FANCMP_MEAN_ATOL="), "RLS1_FANCMP_MEAN_ATOL_NOT_GUARDED"),
        ("the done-marker is not bound to the bounds", sw(re.search(r'if stage_done fancmp && ! grep -qx .*?\nfi\n', text, re.S).group(0), ""), "FANCMP_MARKER_NOT_BOUND_TO_BOUNDS"),
        ("the done-marker does not record the bounds", drop_line(text, 'echo "atol=$FANCMP_ATOL mean_atol=$FANCMP_MEAN_ATOL" >> "$S/fancmp.ok"'), "FANCMP_MARKER_BOUNDS_NOT_RECORDED"),
        ("the pass line omits the mean verdict", sw("$(grep '^ZZRLFAN_MEAN_ATOL_PASS' \"$R/fancmp.log\" | tail -n 1) ", ""), "FANCMP_OK_MARKER_NO_MEAN_VERDICT"),
        ("the banked sets are looked for elsewhere", sw('onpolicy/sets}"', 'onpolicy/sets_full}"'), "FANCMP_BANKED_DRIFT"),
        ("an absent banked dir is not a refusal", drop_line(text, 'ls "$BANKED"/onpolicy_*.jsonl'), "FANCMP_NO_BANKED_CHECK"),
        ("the gate runs after the first dump", (lambda b: text.replace(b, "").replace('stage_done dump_val_s0of1 || refuse "dump of RL-val did not finish"\n',
                                                                                       'stage_done dump_val_s0of1 || refuse "dump of RL-val did not finish"\n' + b))(
            re.search(r"# ---- 4b\..*?stop_after fancmp\n\n", text, re.S).group(0)), "ORDER"),
        ("the stage cannot be stopped after", sw("; stop_after fancmp", ""), "FANCMP_NO_STOP_AFTER"),
        ("the pass is not logged", sw('log "ZZRLS1_FANCMP_OK', 'log "ZZRLS1_FANCMP_SKIPPED'), "FANCMP_NO_OK_MARKER"),
        ("the slice file is not shipped", sw("         splits/rl_fancmp_keys.tsv \\\n", ""), "FANCMP_KEYS_NOT_REQUIRED"),
    ]
    return out


# ------------------------------------------------------------------------------------------------------------- bash helpers
def bash_run(script, env=None, timeout=120):
    e = dict(os.environ, **(env or {}))
    r = subprocess.run([shutil.which("bash"), "-c", script], capture_output=True, text=True, env=e, timeout=timeout)
    return r.returncode, r.stdout.strip(), r.stderr.strip()


def posix(p):
    return str(p).replace("\\", "/")


def write_sh(path, text):
    Path(path).write_text(text, encoding="utf-8", newline="\n")
    # +x (2026-10-15; the same fix refe/test_p2rep_chain.py took): the chain runs its shims DIRECTLY ("$nv", `flock` looked up on PATH).  Git Bash on the dev box needs no exec bit;
    # Linux does, so on the POD every shim here failed with rc 126 (gpu_busy read "busy" forever) and a non-executable `flock` on PATH is skipped for the real /usr/bin/flock.
    os.chmod(path, 0o755)


def write_sh_violations(fn):
    """[] iff `fn(path, text)` leaves `path` executable.  Windows records no exec bit, so the os.chmod CALL is recorded (and, off Windows, the file is really run directly)."""
    calls, real = [], os.chmod
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "shim"

        def spy(path, mode, *a, **k):
            calls.append((Path(path), mode))
            return real(path, mode, *a, **k)
        os.chmod = spy
        try:
            fn(p, "#!/bin/bash\nexit 0\n")
        finally:
            os.chmod = real
        v = [] if any(c[0] == p and c[1] & 0o111 == 0o111 for c in calls) else ["NO_CHMOD_X"]
        if os.name != "nt":
            if not os.access(p, os.X_OK):
                v.append("NOT_EXECUTABLE")
            elif subprocess.run([str(p)], capture_output=True).returncode != 0:
                v.append("DIRECT_EXEC_FAILS")
    return v


def handoff_expected_tokens():
    """The gate section of eval/HANDOFF_RL_S1.md: the first fenced block after 'The exact argv per arm' -> placeholder tokens."""
    t = (PKGROOT / "eval" / "HANDOFF_RL_S1.md").read_text(encoding="utf-8")
    sec = t[t.index("## The exact argv per arm that passes the static gate"):]
    block = sec.split("```")[1]
    toks = []
    for ln in block.splitlines():
        toks += re.findall(r"<[^>]+>|\S+", ln.split("#")[0])
    return toks


def expected_argv(arm, P):
    """The gate section's tokens, in its order, placeholders resolved from the chain's paths; plus exactly two run-kind additions after --seed and the arm's EPO tail."""
    seed = {"C0": 0, "C1": 1, "R0": 0, "R1": 1, "R2": 0}[arm]
    fixed = {"<base.pt>": P["BASE"], "<train targets>": P["TARGETS"], "<pix>": P["IMAGES"], "<calib.json>": P["CALIB"], "<RL-train sets>": P["SETS_TR"],
             "<RL-val sets>": P["SETS_VA"], "<RL-val logs.txt>": P["VALLOGS"], "<arm dir>": f"{P['R']}/arms/{arm}", "<S>": str(seed)}
    byflag = {"--pdm-labels-train": P["PDM_TR"], "--pdm-labels-heldout": P["PDM_VA"], "--teacher-lane-train": P["TLANE_TR"], "--teacher-lane-heldout": P["TLANE_VA"]}
    out, prev = [], None
    for t in handoff_expected_tokens():
        if t == "<dir>":
            t = byflag[prev]
        else:
            t = fixed.get(t, t)
        out.append(t)
        prev = t if t.startswith("--") else prev
        if prev == "--seed" and not t.startswith("--"):
            out += ["--eval-every", "250", "--dump-scores", f"{P['R']}/arms/{arm}/scores"]
    tail = {"C": [], "R": ["--epo", "--epo-terms-train", P["TERMS_TR"], "--epo-terms-heldout", P["TERMS_VA"]]}[arm[0]]
    return out + tail + (["--epo-k", "--epo-l"] if arm == "R2" else [])


def chain_paths(text):
    """The chain's own path variables, resolved by bash itself (the header assignments only)."""
    hdr = "\n".join(l for l in text.split("\n") if re.match(r"^(WORK|PKG|DATA|P2|R|FINAL|BASE|TARGETS|IMAGES|CALIB|PART|PNAME|MANIFEST|TRAIN_TSV|VAL_TSV|SETS_TR|PDM_TR|TERMS_TR|TLANE_TR|VALLOGS)=", l)
                    or l.startswith("SETS_TR="))
    keys = ("R", "BASE", "TARGETS", "IMAGES", "CALIB", "SETS_TR", "SETS_VA", "PDM_TR", "PDM_VA", "TERMS_TR", "TERMS_VA", "TLANE_TR", "TLANE_VA", "VALLOGS")
    rc, out, err = bash_run(hdr + "\nTLANE_TR=${TLANE_TR:-x}\n" + "".join(f'echo "{k}=${k}"\n' for k in keys))
    assert rc == 0, err
    return dict(l.split("=", 1) for l in out.splitlines())


def arm_args_runner(text, tmp, arm):
    body = fn_body(text, "arm_args")
    hdr = "\n".join(l for l in text.split("\n") if re.match(r"^[A-Z0-9_]+=", l) and not l.startswith("export") and "$(" not in l and "FINAL_SHA256" not in l
                    and not l.startswith("NSH"))
    sc = Path(tmp) / "aa.sh"
    write_sh(sc, hdr + "\n" + body + f'\narm_args "$1"\n')
    rc, out, err = bash_run(f'bash "{posix(sc)}" {arm}')
    assert rc == 0, (arm, err)
    return out


# ------------------------------------------------------------------------------------------------------------- RL-S1c / R0c: the chain's own text, RUN under bash
def control_runner(text, tmp, arm, profile, c0=False, c1=False):
    """The arm loop's control-read block extracted verbatim and run for `arm` under `profile`, with C0's / C1's banked dumps present or not.  -> (rc, stdout)."""
    d = Path(tmp) / f"ctl_{arm}_{profile}_{int(c0)}{int(c1)}"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    for name, have in (("C0", c0), ("C1", c1)):
        if have:
            (d / "arms" / name / "scores").mkdir(parents=True)
            (d / "arms" / name / "scores" / "scores_update1000_A.npz").write_text("x")
    sc = d / "ctl.sh"
    write_sh(sc, f'set -u\nR="{posix(d)}"; ARM="{arm}"; PROFILE="{profile}"; ARMDIR="$R/arms/$ARM"\nrefuse() {{ echo "REFUSED: $*"; exit 1; }}\n{control_block(text)}\n'
                 'echo "KIND=$KIND RD=$RD REFDIR=${REFDIR#$R/}"\n')
    rc, out, _ = bash_run(f'bash "{posix(sc)}"')
    return rc, out


R0C_CONTROL_CASES = (       # (arm, profile, C0 dumps on the pod, C1 dumps on the pod) -> KIND, RD, the dir the monitor and the replay read against (relative to $R), or REFUSED
    ("R0c", "refe_rls1c", True, False, "B", "C0", "arms/R0c/scores"),      # R0c: its OWN arm A, even with C0's dumps there (PREREG_R0C_DRAFT.md section 2) ...
    ("R0c", "refe_rls1c", False, False, "B", "C0", "arms/R0c/scores"),     # ... and without them (no refusal: C0 is the SECOND control, and a log line says when it is missing)
    ("R0c", "refe_rls1c", True, True, "B", "C0", "arms/R0c/scores"),
    ("R0", "refe_rls1", True, False, "B", "C0", "arms/C0/scores"),         # every arm before R0c is read exactly as it was
    ("R0", "refe_rls1", False, False, "B", "C0", "arms/R0/scores"),
    ("R1", "refe_rls1", False, True, "B", "C1", "arms/C1/scores"),
    ("R2", "refe_rls1", True, False, "B", "C0", "arms/C0/scores"),
    ("R0b", "refe_rls1b", True, False, "B", "C0", "arms/C0/scores"),
    ("R0b", "refe_rls1b", False, False, "B", "C0", "REFUSED"),
    ("R1b", "refe_rls1b", False, True, "B", "C1", "arms/C1/scores"),
    ("C0", "refe_rls1", True, False, "A", "", "arms/C0/scores"),
)
R0C_UNKNOWN_ARMS = ("R0C", "r0c", "R0cc", "R0c2", "R0-c", "RC0", "Roc", "R0c,", "R0dd", "c", "0c", "R", "R0 c", "R0b R0C", "R0c R1c")      # every one must be refused at the precheck
R0C_KNOWN_LISTS = ("R0c", "C0 R0c", "R0 R0c", "R0b C1 R1b R0c")


def r0c_behaviour(text, tmp, P, groups=None):
    """RUN the chain's own text under bash for R0c (everything extracted verbatim).  -> ([violations], probes run); [] = all hold.
    groups (None = all): argv (the argv token for token against eval/rls1c_arms.json and R0's banked sha256, then the REAL gate), profile, control (the block that picks the
    reference the monitor and the replay read against), kill (the two refusals that stop the chain after a kill), arms (the precheck refusal of an unknown or misspelled arm)."""
    import launch_gate as LGM
    tmp, v, k = Path(tmp), [], 0                                                                  # k = the number of probes run (each one a counted check)
    want = lambda g: groups is None or g in groups                                               # noqa: E731
    bash = shutil.which("bash")
    if want("argv"):
        k += 6
        try:
            got_s = arm_args_runner(text, tmp, "R0c")
            got = shlex.split(got_s)
        except AssertionError:
            got_s, got = None, None
        if got is None:
            v.append("R0C_NO_ARGV")
        else:
            js = json.loads((PKGROOT / "eval" / "rls1c_arms.json").read_text(encoding="utf-8"))["arms"]["R0c"]
            if got != js["argv"]:
                v.append("R0C_ARGV_NOT_THE_JSON")
            if argv_sha256(got) != R0C_ARGV_SHA256 or js["argv_sha256"] != R0C_ARGV_SHA256:
                v.append("R0C_ARGV_SHA256_DRIFT")
            r0 = shlex.split(arm_args_runner(text, tmp, "R0"))
            if argv_sha256(r0) != R0_ARGV_SHA256:
                v.append("R0_ARGV_NOT_THE_BANKED_TOKEN")
            ad = f"{P['R']}/arms"
            rebuilt = [t.replace(f"{ad}/R0c", f"{ad}/R0") for t in got[:-2]]                      # R0c minus the lever, with the arm directory swapped back, IS R0
            if ([t for t in got if t not in r0] != [f"{ad}/R0c", f"{ad}/R0c/scores", "--epo-ref", "armA"] or [t for t in r0 if t not in got] != [f"{ad}/R0", f"{ad}/R0/scores"]
                    or len(got) != len(r0) + 2 or got[-2:] != ["--epo-ref", "armA"] or rebuilt != r0):
                v.append("R0C_NOT_R0_PLUS_ONE_FLAG")
            if LGM.profile_refusals({**LGM.PROFILES["refe_rls1c"], "declared_arm": "R0c"}, LGM.parse_args_string(got_s)):
                v.append("R0C_REFUSED_BY_THE_REAL_GATE")
    if want("profile"):
        k += 9
        pr = tmp / "ap.sh"
        write_sh(pr, fn_any(text, "arm_profile") + '\narm_profile "$1"\n')
        for arm, arms, want_p in (("R0c", "", "refe_rls1c"), ("R0c", "R0c", "refe_rls1c"), ("R0c", "C0 R0c", "refe_rls1c"), ("R0c", "R0b C1 R1b R0c", "refe_rls1c"),
                                  ("C1", "R0c", "refe_rls1"), ("C0", "R0c", "refe_rls1"), ("R0", "R0c", "refe_rls1"), ("R0b", "R0c", "refe_rls1b"), ("C1", "R0b C1 R1b", "refe_rls1b")):
            got_p = subprocess.run([bash, posix(pr), arm], capture_output=True, text=True, env={**os.environ, "RLS1_ARMS": arms}).stdout.strip()
            if got_p != want_p:
                v.append(f"R0C_WRONG_PROFILE:{arm}[{arms}]->{got_p}")
    if want("control"):
        k += len(R0C_CONTROL_CASES)
        for arm, prof, c0, c1, kind, rd, ref in R0C_CONTROL_CASES:
            rc, out = control_runner(text, tmp, arm, prof, c0, c1)
            ok = (rc != 0 and f"RL-S1b arm {arm} is read against {rd}" in out) if ref == "REFUSED" else (rc == 0 and out == f"KIND={kind} RD={rd} REFDIR={ref}")
            if not ok:
                v.append(f"{'R0C_CONTROL' if arm == 'R0c' else 'CONTROL_DRIFT'}:{arm}:C0={c0}:C1={c1}:{out[:120]}")
    if want("kill"):
        k += 6
        heads = [ln for ln in text.split("\n") if "arm $ARM was KILLED" in ln and not ln.lstrip().startswith("#")]
        sdone = re.search(r"^stage_done\(\) .*$", text, re.M)
        if len(heads) != 2 or not sdone:
            v.append("R0C_KILL_LINES_MISSING")
        else:
            for arm in ("R0c", "R0b"):
                for marker, egress, case in ((True, True, "head"), (True, False, "tail"), (False, False, "none")):
                    d = tmp / f"kill_{arm}_{case}"
                    shutil.rmtree(d, ignore_errors=True)
                    (d / "stage").mkdir(parents=True)
                    if marker:
                        (d / "stage" / f"arm_{arm}_killed").write_text("K3 g5 breached")
                    if egress:
                        (d / "stage" / f"egress_{arm}.ok").write_text("x")
                    sc = d / "kill.sh"
                    write_sh(sc, f'set -u\nS="{posix(d)}/stage"; ARM="{arm}"\nrefuse() {{ echo "REFUSED: $*"; exit 1; }}\n{sdone.group(0)}\n{heads[0]}\necho PASSED_HEAD\n{heads[1]}\necho PASSED_TAIL\n')
                    rc, out, _ = bash_run(f'bash "{posix(sc)}"')
                    ok = {"head": rc == 1 and f"REFUSED: arm {arm} was KILLED" in out and "PASSED_HEAD" not in out and "delete" in out,
                          "tail": rc == 1 and "PASSED_HEAD" in out and f"REFUSED: arm {arm} was KILLED" in out and "before any further arm" in out and "PASSED_TAIL" not in out,
                          "none": rc == 0 and "PASSED_HEAD" in out and "PASSED_TAIL" in out and "REFUSED" not in out}[case]
                    if not ok:
                        v.append(f"R0C_KILL_DOES_NOT_REFUSE:{arm}:{case}:{out[:100]}")
    if want("arms"):
        k += len(R0C_KNOWN_LISTS) + len(R0C_UNKNOWN_ARMS)
        loop = re.search(r'^for ARM in \$\{RLS1_ARMS:-C0 R0 C1 R1 R2\}; do arm_registered "\$ARM" \|\| refuse "unknown arm.*?; done$', text, re.M)
        regs = re.search(r'^REGISTERED_ARMS="[^"]*"$', text, re.M)
        if not loop or not regs:
            v.append("R0C_NO_PRECHECK")
        else:
            d = tmp / "arms_empty_cwd"
            d.mkdir(exist_ok=True)
            sc = tmp / "precheck.sh"
            write_sh(sc, f'set -u\ncd "{posix(d)}"\nrefuse() {{ echo "REFUSED: $*"; exit 1; }}\n{regs.group(0)}\n{fn_any(text, "arm_registered")}\n{loop.group(0)}\necho ACCEPTED\n')
            for arms in R0C_KNOWN_LISTS:
                rc, out, _ = bash_run(f'bash "{posix(sc)}"', {"RLS1_ARMS": arms})
                if not (rc == 0 and out == "ACCEPTED"):
                    v.append(f"R0C_REFUSED_AT_PRECHECK:[{arms}]:{out[:100]}")
            for arms in R0C_UNKNOWN_ARMS:
                rc, out, _ = bash_run(f'bash "{posix(sc)}"', {"RLS1_ARMS": arms})
                if not (rc == 1 and "REFUSED: unknown arm" in out and "ACCEPTED" not in out):
                    v.append(f"MISSPELLED_ARM_ACCEPTED:[{arms}]:{out[:100]}")
    return v, k


def r0c_mutations(text):
    """(name, mutated text, static violation prefix check_chain must raise or None, (r0c_behaviour group, violation prefix) or None).  Every mutated copy must differ from `text`."""
    sw = lambda a, b: text.replace(a, b)                                                         # noqa: E731
    regs = 'REGISTERED_ARMS="C0 C1 R0 R1 R2 R0b R1b R0c R0d R1d R1o"'
    precheck = re.search(r'^for ARM in \$\{RLS1_ARMS:-C0 R0 C1 R1 R2\}; do arm_registered "\$ARM" \|\| refuse "unknown arm.*?; done\n', text, re.M).group(0)
    own = re.search(r'^  \[ "\$ARM" = R0c \] && REFDIR="\$ARMDIR/scores"\n', text, re.M).group(0)
    generic = '  if [ -z "$RD" ] || [ ! -f "$R/arms/$RD/scores/scores_update1000_A.npz" ]; then REFDIR="$ARMDIR/scores"; else REFDIR="$R/arms/$RD/scores"; fi\n'
    r0c_line = re.search(r"^    R0c\) seed=0;.*\n", text, re.M).group(0)
    return [
        ("the R0c argv line is deleted", text.replace(r0c_line, ""), "R0C_NO_ARGV", ("argv", "R0C_NO_ARGV")),
        ("R0c is not registered", sw(regs, regs.replace(" R0c", "")), "R0C_NOT_REGISTERED", ("arms", "R0C_REFUSED_AT_PRECHECK")),
        ("R0c is registered under a typo (R0C)", sw(regs, regs.replace(" R0c", " R0C")), "R0C_NOT_REGISTERED", ("arms", "MISSPELLED_ARM_ACCEPTED")),
        ("R0c is judged under the original profile", sw("R0c) echo refe_rls1c", "R0c) echo refe_rls1"), "R0C_NO_PROFILE", ("profile", "R0C_WRONG_PROFILE")),
        ("R0c is judged under the RL-S1b profile", sw("R0c) echo refe_rls1c ;;", "R0c) echo refe_rls1b ;;"), "R0C_NO_PROFILE", ("profile", "R0C_WRONG_PROFILE")),
        ("R0c has no profile line (falls through to refe_rls1)", drop_line(text, "R0c) echo refe_rls1c"), "R0C_NO_PROFILE", ("profile", "R0C_WRONG_PROFILE")),
        ("C1 is judged under refe_rls1b whenever R0c is listed", sw('*" R0b "*|*" R1b "*) echo refe_rls1b', '*" R0b "*|*" R1b "*|*" R0c "*) echo refe_rls1b'), None, ("profile", "R0C_WRONG_PROFILE")),
        ("--epo-ref armA is dropped from R0c's argv", sw(' --epo-ref armA" ;;', '" ;;'), "R0C_NO_EPO_REF_ARMA", ("argv", "R0C_ARGV_NOT_THE_JSON")),
        ("--epo-ref armA is dropped (the real gate says so too)", sw(' --epo-ref armA" ;;', '" ;;'), None, ("argv", "R0C_REFUSED_BY_THE_REAL_GATE")),
        ("--epo-ref base is named instead of armA", sw("--epo-ref armA\" ;;", "--epo-ref base\" ;;"), "R0C_NO_EPO_REF_ARMA", ("argv", "R0C_ARGV_NOT_THE_JSON")),
        ("a second lever rides along (--epo-lam 0.25)", sw('--epo-ref armA" ;;', '--epo-ref armA --epo-lam 0.25" ;;'), "R0C_EXTRA_LEVER:--epo-lam", ("argv", "R0C_NOT_R0_PLUS_ONE_FLAG")),
        ("R0c runs with seed 1", sw("R0c) seed=0;", "R0c) seed=1;"), "R0C_WRONG_SEED", ("argv", "R0C_ARGV_NOT_THE_JSON")),
        ("R0 is no longer the banked token's argv", sw("    R0) seed=0;", "    R0) seed=7;"), None, ("argv", "R0_ARGV_NOT_THE_BANKED_TOKEN")),
        ("R0c is read against C0's dumps when they are there (the own-A override is deleted)", drop_line(text, '[ "$ARM" = R0c ] && REFDIR='), "R0C_NOT_READ_AGAINST_ITS_OWN_A", ("control", "R0C_CONTROL")),
        ("the own-A override runs BEFORE the generic assignment and is overwritten", text.replace(own, "").replace(generic, own + generic), "R0C_NOT_READ_AGAINST_ITS_OWN_A",
         ("control", "R0C_CONTROL")),
        ("C0 is not named as R0c's second control (static only: ${ARM:1:1} happens to give C0 as well)", drop_line(text, '[ "$ARM" = R0c ] && RD=C0'), "R0C_SECOND_CONTROL_NOT_NAMED", None),
        ("R0c falls under the RL-S1b control guard (refused against its own arm A)",
         sw('if [ "$PROFILE" = refe_rls1b ] && [ "$KIND" = B ]', 'if { [ "$PROFILE" = refe_rls1b ] || [ "$PROFILE" = refe_rls1c ]; } && [ "$KIND" = B ]'), "RLS1B_GUARD_EDITED", ("control", "R0C_CONTROL")),
        ("R0b / R1b lose their guard (a missing control is silently replaced by the arm's own A)", sw('if [ "$PROFILE" = refe_rls1b ] && [ "$KIND" = B ]', 'if false && [ "$KIND" = B ]'), "RLS1B_GUARD_EDITED",
         ("control", "CONTROL_DRIFT")),
        ("no loud marker when C0's banked dumps are missing", re.sub(r'      elif \[ "\$ARM" = R0c \]; then\n        log "ZZRLS1_R0C_NO_SECOND_CONTROL[^\n]*\n', "", text), "R0C_SILENT_WITHOUT_SECOND_CONTROL", None),
        ("the end-of-loop kill refusal skips R0c", sw('  [ -f "$S/arm_${ARM}_killed" ] && refuse "arm $ARM was KILLED ($(cat', '  [ "$ARM" != R0c ] && [ -f "$S/arm_${ARM}_killed" ] && refuse "arm $ARM was KILLED ($(cat'),
         "KILL_RULE_ARM_GATED", ("kill", "R0C_KILL_DOES_NOT_REFUSE")),
        ("the restart-after-kill refusal skips R0c", sw('  [ -f "$S/arm_${ARM}_killed" ] && stage_done "egress_$ARM" && refuse', '  [ "$ARM" != R0c ] && [ -f "$S/arm_${ARM}_killed" ] && stage_done "egress_$ARM" && refuse'),
         "KILL_RULE_ARM_GATED", ("kill", "R0C_KILL_DOES_NOT_REFUSE")),
        ("the gate's suite files are not required before the queue wait", sw(" refe/test_rl_s1b.py refe/test_rl_r0c.py refe/test_rl_safeobj.py \\", " \\"), "R0C_SUITE_FILE_NOT_REQUIRED", None),
        ("arm names are matched as substrings", sw('*" $1 "*) return 0', '*"$1"*) return 0'), None, ("arms", "MISSPELLED_ARM_ACCEPTED")),
        ("the precheck no longer validates the arm names", sw(precheck, ""), "ARM_NAMES_NOT_REFUSED_TWICE", ("arms", "R0C_NO_PRECHECK")),
    ]


# ------------------------------------------------------------------------------------------------------------- fixtures for the helper
N = 4


def mk_dump(tmp, scenes, ckpt=RC.CKPT_STEP, name="onpolicy_rlfan_fan_s0of1.jsonl"):
    d = Path(tmp) / "sets"
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text("".join(json.dumps({"kind": "onpolicy_set", "log_name": l, "token": t, "step": 0, "rank": 0, "ckpt_step": ckpt}) + "\n" for t, l in scenes),
                          encoding="utf-8")
    return d


def mk_labels(tmp, scenes, ckpt=RC.CKPT_STEP, skip_pdm=(), skip_terms=(), bad_id=(), fires=False):
    pd, td = Path(tmp) / "pdm", Path(tmp) / "terms"
    pd.mkdir(parents=True, exist_ok=True)
    td.mkdir(parents=True, exist_ok=True)
    rows_p, rows_t = [], []
    for t, l in scenes:
        key = [l, t, 0, 0]
        dac = [1.0, 0.0, 1.0, 1.0]
        if t not in skip_pdm:
            rows_p.append({"kind": "pdm_targets", "key": key, "ckpt_step": ckpt, "nc": [1.0] * N, "dac": dac, "ep": [0.5] * N, "ttc": [1.0] * N, "c": [1.0] * N,
                           "ddc": [1.0] * N, "pdms": [0.0] * N})
        if t not in skip_terms:
            g = [0.0 if x == 1.0 else 0.4 for x in dac]
            if t in bad_id:
                g[0] = 0.3                        # g > 0 on a DAC = 1 proposal: the identity breaks
            rows_t.append({"kind": "rl_terms", "key": key, "ckpt_step": ckpt, "d": [0.0, 0.5, 0.0, 0.0], "gmax": g, "k": [0.0] * N, "lc_executed": [0, 0, 0, 0],
                           "dither": [0, 0, 0, 0], "label_fires": fires})
    (pd / "pdm_a.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows_p), encoding="utf-8")
    (td / "rl_terms_a.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows_t), encoding="utf-8")
    return pd, td


def main() -> int:
    n = 0
    text = CHAIN.read_text(encoding="utf-8")
    for f in ("rls1_pod.sh", "rls1_checks.py", "test_rls1_chain.py"):
        assert b"\r" not in (HERE / f).read_bytes(), f + " must stay LF"
        n += 1
    rc, _, err = bash_run(f'bash -n "{posix(CHAIN)}"')
    assert rc == 0, err; n += 1                                                              # bash can parse it
    # ---- 0b. write_sh gives every shim the exec bit (the pod runs them directly); the helper without os.chmod is caught
    assert write_sh_violations(write_sh) == [], write_sh_violations(write_sh); n += 1
    assert "NO_CHMOD_X" in write_sh_violations(lambda p, t: Path(p).write_text(t, encoding="utf-8", newline="\n")); n += 1
    # ---- 1. the real chain satisfies every property; every property goes RED when broken
    assert check_chain(text) == [], check_chain(text); n += 1
    for name, mut, want in mutations(text):
        assert mut != text, f"mutation '{name}' changed nothing"
        got = check_chain(mut)
        assert any(g.startswith(want) for g in got), f"mutation '{name}' did not raise {want}: {got}"
        n += 1
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        P = chain_paths(text)
        # ---- 1b. RL-S1c / R0c: the chain's own text RUN under bash (every probe counted), then each property broken in a copy -- the static check and/or the behaviour probe must say so
        bad_beh, probes = r0c_behaviour(text, tmp, P)
        assert bad_beh == [], bad_beh; n += probes
        for name, mut, swant, beh in r0c_mutations(text):
            assert mut != text, f"R0c mutation '{name}' changed nothing"
            assert swant or beh, f"R0c mutation '{name}' has no catcher"
            if swant:
                got = check_chain(mut)
                assert any(g.startswith(swant) for g in got), f"R0c mutation '{name}' did not raise {swant}: {got}"; n += 1
            if beh:
                got, _ = r0c_behaviour(mut, tmp, P, groups={beh[0]})
                assert any(g.startswith(beh[1]) for g in got), f"R0c mutation '{name}' did not raise {beh[1]} in group {beh[0]}: {got}"; n += 1
        # ---- 2a. p2_over / gpu_busy: extracted verbatim, run under bash with a shimmed flock and a fake nvidia-smi
        p2 = tmp / "p2"
        p2.mkdir()
        (p2 / "chain.lock").write_text("")
        (p2 / "chain.log").write_text("ZZP2R_DONE\n")
        nv_idle, nv_busy, nv_broken = tmp / "nv_idle.sh", tmp / "nv_busy.sh", tmp / "nv_broken.sh"
        write_sh(nv_idle, "#!/bin/bash\nexit 0\n")
        write_sh(nv_busy, "#!/bin/bash\necho 4242\n")
        write_sh(nv_broken, "#!/bin/bash\nexit 9\n")
        scr = tmp / "po.sh"
        write_sh(scr, f'P2="{posix(p2)}"\nflock() {{ return "${{FAKE_FLOCK_RC:-0}}"; }}\n{fn_body(text, "gpu_busy")}\n{fn_body(text, "p2_over")}\np2_over\n')

        def po(**env):
            return bash_run(f'bash "{posix(scr)}"', {"RLS1_NVSMI": posix(nv_idle), **env})
        rc, out, _ = po(); assert rc == 0 and out == ""; n += 1                                  # everything clear
        rc, out, _ = po(FAKE_FLOCK_RC="1"); assert rc == 1 and "still holds" in out; n += 1       # P2's chain lock is held
        rc, out, _ = po(RLS1_NVSMI=posix(nv_busy)); assert rc == 0 and out == ""; n += 1          # a process on the GPU is the QUEUE's business now (a chain queued behind S1 must start)
        gscr = tmp / "gb.sh"
        write_sh(gscr, fn_body(text, "gpu_busy") + "\ngpu_busy\n")
        for nv, want in ((nv_idle, 1), (nv_busy, 0), (nv_broken, 0)):                              # idle -> rc 1 (not busy); a process -> busy; nvidia-smi cannot say -> busy
            assert bash_run(f'bash "{posix(gscr)}"', {"RLS1_NVSMI": posix(nv)})[0] == want; n += 1
        (p2 / "chain.log").write_text("ZZP2_FAILED\n")
        rc, out, _ = po(); assert rc == 1 and "ZZP2R_DONE" in out; n += 1                         # P2 never finished
        rc, out, _ = po(RLS1_P2_ABANDONED="1"); assert rc == 0; n += 1                            # ... unless the PI said so
        (p2 / "chain.log").write_text("ZZP2R_DONE\n")
        (p2 / "chain.lock").unlink()
        rc, out, _ = po(FAKE_FLOCK_RC="1"); assert rc == 0; n += 1                                # no lock file: nothing to hold
        # ---- 2b. cgroup_cores: v2, v1, unlimited
        cg = tmp / "cg"
        (cg / "cpu").mkdir(parents=True)
        cs = tmp / "cg.sh"
        write_sh(cs, fn_body(text, "cgroup_cores") + "\ncgroup_cores\n")
        nproc = bash_run("nproc")[1]
        (cg / "cpu" / "cpu.cfs_quota_us").write_bytes(b"1105000\n"); (cg / "cpu" / "cpu.cfs_period_us").write_bytes(b"100000\n")   # LF, as on the pod
        assert bash_run(f'bash "{posix(cs)}"', {"RLS1_CG_ROOT": posix(cg)})[1] == "11"; n += 1     # MEASURED on the P2 pod: v1, 11.05 cores, nproc said 96
        (cg / "cpu.max").write_bytes(b"400000 100000\n")
        assert bash_run(f'bash "{posix(cs)}"', {"RLS1_CG_ROOT": posix(cg)})[1] == "4"; n += 1      # v2 wins when present
        (cg / "cpu.max").write_bytes(b"max 100000\n"); (cg / "cpu" / "cpu.cfs_quota_us").write_bytes(b"-1\n")
        assert bash_run(f'bash "{posix(cs)}"', {"RLS1_CG_ROOT": posix(cg)})[1] == nproc; n += 1     # unlimited -> nproc
        # ---- 2c. arm_args: the argv each arm runs equals the gate section's, token for token; then the REAL launch gate judges it
        import launch_gate as LGM
        argv_of = {}
        for arm in ARMS:
            got = arm_args_runner(text, tmp, arm)
            assert shlex.split(got) == expected_argv(arm, P), (arm, shlex.split(got), expected_argv(arm, P)); n += 1
            argv_of[arm] = got
            prof = {**LGM.PROFILES["refe_rls1"], "declared_arm": arm}
            assert LGM.profile_refusals(prof, LGM.parse_args_string(got)) == [], (arm, LGM.profile_refusals(prof, LGM.parse_args_string(got))); n += 1
        assert "--epo" not in shlex.split(argv_of["C0"]) + shlex.split(argv_of["C1"]); n += 1
        bad = {
            "C0 + --epo": ("C0", argv_of["C0"] + " --epo --epo-terms-train x"),
            "C1 + an EPO lever at its default": ("C1", argv_of["C1"] + " --epo-temp 0.1"),
            "R0 without --epo": ("R0", argv_of["C0"].replace("--seed 0", "--seed 0")),
            "R0 + K and L": ("R0", argv_of["R2"]),
            "R2 without K and L": ("R2", argv_of["R0"]),
            "R0 with C0's seed 1": ("R0", argv_of["R0"].replace("--seed 0", "--seed 1")),
            "R1 with 750 updates": ("R1", argv_of["R1"].replace("--max-updates 1000", "--max-updates 750")),
            "C0 without the lane trio": ("C0", argv_of["C0"].replace("--lane-head ", "")),
        }
        for name, (arm, argv) in bad.items():
            prof = {**LGM.PROFILES["refe_rls1"], "declared_arm": arm}
            assert LGM.profile_refusals(prof, LGM.parse_args_string(argv)), f"the real gate accepted: {name}"; n += 1
        # ---- 2d. token_ok, extracted verbatim, against tokens minted by the gate's own mint() for refe_rls1
        m3 = TOKEN_OK_RX.search(text)
        h3 = tmp / "h3.sh"
        write_sh(h3, m3.group(0))
        key = GT.load_key(tmp / "k" / "gate.key", forbid_under=[PKGROOT])
        lgdir, armdir = tmp / "lg", tmp / "armdir"
        armdir.mkdir()

        def mint(arm, checks):
            shutil.rmtree(lgdir, ignore_errors=True)
            av = GT.argv_from_string(argv_of[arm])
            return GT.mint(profile="refe_rls1", commit="a" * 40, tree_man=GT.tree_manifest(PKGROOT), argv=av, trainer="refe/scorer_finetune.py",
                           data=GT.data_manifest(av, LGM.RLS1_DATA), checks=checks, reasons=[], key=key, out_dir=lgdir)

        def tok(args, keyfile=None):
            env = {"PY": posix(sys.executable), "PKG": posix(PKGROOT), "PROFILE": "refe_rls1", "LG": posix(lgdir), "ARMDIR": posix(armdir), "ARGS": args, "PYTHONIOENCODING": "utf-8",
                   "TANITAD_GATE_KEY_FILE": posix(keyfile or (tmp / "k" / "gate.key"))}
            return bash_run(f'bash -c \'. "{posix(h3)}"; token_ok\'', env)[0]
        mint("R0", {"G-HYG": {"status": "PASS"}})
        assert tok(argv_of["R0"]) == 0; n += 1                                                  # the matching argv: the trainer may start
        assert tok(argv_of["C0"]) != 0; n += 1                                                  # another arm's argv: refused
        assert tok(argv_of["R0"].replace("--lr 1e-4", "--lr 2e-4")) != 0; n += 1                # a changed lever
        assert tok(argv_of["R0"] + " --resume") != 0; n += 1                                    # an extra flag
        assert tok(argv_of["R0"], keyfile=tmp / "k" / "other.key") != 0; n += 1                 # another host's key
        tokp = next(lgdir.glob("PASS_*.json"))
        tj = json.loads(tokp.read_text(encoding="utf-8"))
        tj["binding"]["argv_sha256"] = "0" * 64
        tokp.write_text(json.dumps(tj), encoding="utf-8")
        assert tok(argv_of["R0"]) != 0; n += 1                                                  # an edited token
        mint("R0", {"G-HYG": {"status": "FAIL"}})
        assert not list(lgdir.glob("PASS_*.json")) and tok(argv_of["R0"]) != 0; n += 1          # a FAIL token is no PASS token
        shutil.rmtree(lgdir, ignore_errors=True)
        assert tok(argv_of["R0"]) != 0; n += 1                                                  # NO token at all
        # ---- 2e. R0c: the chain's OWN argv and the profile the chain's own arm_profile picks, judged by the REAL gate (the CLI the arm loop calls) and by a REAL minted token
        pr_sc = tmp / "ap_c.sh"
        write_sh(pr_sc, fn_any(text, "arm_profile") + '\narm_profile "$1"\n')
        prof_c = subprocess.run([shutil.which("bash"), posix(pr_sc), "R0c"], capture_output=True, text=True, env={**os.environ, "RLS1_ARMS": "R0c"}).stdout.strip()
        assert prof_c == "refe_rls1c", prof_c; n += 1
        argv_c, argv_r0 = arm_args_runner(text, tmp, "R0c"), argv_of["R0"]

        def lg_cli(profile, args, arm="R0c"):
            r = subprocess.run([sys.executable, str(HERE / "launch_gate.py"), "refusals", "--profile", profile, "--arm-name", arm, "--args", args], capture_output=True, text=True, timeout=180)
            return r.returncode, r.stdout + r.stderr
        rc, out = lg_cli(prof_c, argv_c); assert rc == 0 and "ZZGATE-CLEANZZ" in out, out; n += 1                    # `launch_gate.py run --profile "$PROFILE" --arm-name "$ARM"` accepts what the chain builds
        rc, out = lg_cli(prof_c, argv_r0); assert rc == 5 and "--epo-ref" in out, out; n += 1                       # R0's argv under R0c's name: the lever is missing and the gate says so
        rc, out = lg_cli("refe_rls1", argv_c, arm="R0"); assert rc == 5 and "--epo-ref" in out, out; n += 1         # the original profile refuses R0c's argv even under R0's name
        rc, out = lg_cli("refe_rls1", argv_c); assert rc != 0 and "'R0c' is not an arm of refe_rls1" in out, out; n += 1   # ... and R0c is not one of its arms

        def mint_c(profile, args):
            shutil.rmtree(lgdir, ignore_errors=True)
            av = GT.argv_from_string(args)
            return GT.mint(profile=profile, commit="a" * 40, tree_man=GT.tree_manifest(PKGROOT), argv=av, trainer="refe/scorer_finetune.py",
                           data=GT.data_manifest(av, LGM.RLS1_DATA), checks={"G-HYG": {"status": "PASS"}}, reasons=[], key=key, out_dir=lgdir)

        def tok_c(args, profile):
            env = {"PY": posix(sys.executable), "PKG": posix(PKGROOT), "PROFILE": profile, "LG": posix(lgdir), "ARMDIR": posix(armdir), "ARGS": args, "PYTHONIOENCODING": "utf-8",
                   "TANITAD_GATE_KEY_FILE": posix(tmp / "k" / "gate.key")}
            return bash_run(f'bash -c \'. "{posix(h3)}"; token_ok\'', env)[0]
        mint_c(prof_c, argv_c)
        assert tok_c(argv_c, prof_c) == 0; n += 1                                               # the chain's argv under the chain's profile: the trainer may start
        assert tok_c(argv_r0, prof_c) != 0; n += 1                                              # R0's argv (no --epo-ref armA) is not R0c's token
        assert tok_c(argv_c.replace("--epo-ref armA", "--epo-ref base"), prof_c) != 0; n += 1   # the lever flipped back
        assert tok_c(argv_c.replace("/arms/R0c", "/arms/R0"), prof_c) != 0; n += 1              # R0c's lever, R0's directory
        assert tok_c(argv_c, "refe_rls1") != 0; n += 1                                          # the token is bound to its profile
        mint_c("refe_rls1", argv_r0)
        assert tok_c(argv_c, prof_c) != 0; n += 1                                               # R0's token does not start R0c
        shutil.rmtree(lgdir, ignore_errors=True)
        # ---- 2f. the hops between the chain's argv and the trainer: the planned-update check, the trainer's OWN parser and its EPO refusals take R0c's argv and read `armA` (R0's reads `base`)
        ok, why = RC.updates_check(argv_c, 16000); assert ok, why; n += 1
        import scorer_finetune as SF
        sf_ap = SF.build_parser()
        ns_c, ns_0 = sf_ap.parse_args(shlex.split(argv_c)), sf_ap.parse_args(shlex.split(argv_r0))
        assert ns_c.epo and ns_c.epo_ref == "armA" and SF.epo_refusals(ns_c, sf_ap) == [], (ns_c.epo_ref, SF.epo_refusals(ns_c, sf_ap)); n += 1
        assert ns_0.epo and ns_0.epo_ref == "base" and SF.epo_refusals(ns_0, sf_ap) == []; n += 1
        # ---- 3. the REAL chain, executed in a sandbox up to its first refusals
        n += sandbox_runs(text, tmp)
        # ---- 3b. the fan-equivalence stage, run from the chain's own text against a stub dumper
        n += fancmp_behaviour(text, tmp)
        # ---- 4. rls1_checks.py
        n += helper_checks(tmp)
    print(f"ZZTEST_PASS {n}")
    return 0


def sandbox_runs(text, tmp):
    """Run rls1_pod.sh itself with WORK pointed at a temp dir, a fake flock/nvidia-smi on PATH, and no network."""
    n = 0
    if os.name == "nt" and subprocess.run([shutil.which("bash"), "-c", "exec 200>/dev/null"], capture_output=True).returncode != 0:
        print("ZZTEST_SKIP sandbox_runs: this bash cannot open fd 200")
        return 0
    w = tmp / "work"
    shutil.rmtree(w, ignore_errors=True)
    (w / "data" / "refe_p2").mkdir(parents=True)
    pk = w / "refe-plan-rls1"
    (pk / "refe").mkdir(parents=True)
    bindir = tmp / "bin"
    bindir.mkdir(exist_ok=True)
    write_sh(bindir / "flock", '#!/bin/bash\nif [ "$2" = 200 ] || [ "$2" = 199 ]; then exit 0; fi\nexit "${FAKE_FLOCK_RC:-0}"\n')
    write_sh(bindir / "nvidia-smi", '#!/bin/bash\nif [ -n "${FAKE_GPU_PID:-}" ]; then echo "$FAKE_GPU_PID"; fi\nexit 0\n')
    script = tmp / "rls1_sandbox.sh"
    write_sh(script, text.replace("WORK=/workspace", f"WORK={posix(w)}").replace("sleep 1800 200>&-", "sleep 1 200>&-"))
    R = w / "data" / "rls1"
    (w / "data" / "refe_p2" / "chain.lock").write_text("")
    (w / "data" / "refe_p2" / "chain.log").write_text("ZZP2R_DONE\n")
    base_env = {"PATH": posix(bindir) + os.pathsep + os.environ["PATH"], "HF_TOKEN": "hf_notarealtoken", "HF_HUB_OFFLINE": "1", "RLS1_DD_MB": "8",
                "RLS1_QUEUE_POLL_S": "1", "RLS1_IDLE_POLL_S": "1", "RLS1_SETTLE_S": "0", "RLS1_WAIT_MAX_S": "10"}

    def run(env_extra=None, drop=()):
        shutil.rmtree(R, ignore_errors=True)
        env = {k: v for k, v in {**base_env, **(env_extra or {})}.items() if k not in drop}
        e = {k: v for k, v in os.environ.items() if k not in drop}
        e.update(env)
        r = subprocess.run([shutil.which("bash"), posix(script)], capture_output=True, text=True, env=e, timeout=180)
        log = (R / "chain.log").read_text(encoding="utf-8") if (R / "chain.log").exists() else ""
        return r.returncode, r.stdout + log
    rc, out = run({"FAKE_FLOCK_RC": "1"}); assert rc == 1 and "REFUSED: the P2 chain still holds" in out, out; n += 1          # P2's lock is held
    (w / "data" / "refe_p2" / "chain.log").write_text("ZZP2_FAILED\n")
    rc, out = run(); assert rc == 1 and "REFUSED: P2 never logged ZZP2R_DONE" in out, out; n += 1                                # P2 unfinished
    (w / "data" / "refe_p2" / "chain.log").write_text("ZZP2R_DONE\n")
    rc, out = run(drop=("HF_TOKEN",)); assert rc == 1 and "REFUSED: HF_TOKEN is not set" in out, out; n += 1                    # no token
    reg = re.search(r'^REGISTERED_ARMS="([^"]*)"', text, re.M).group(1)
    for bad_arm in ("R0C", "r0c", "R0 R0cc", "R0c R1c"):                                                                           # a misspelled / unknown arm is refused at the PRECHECK, before the package or the queue
        rc, out = run({"RLS1_ARMS": bad_arm}); assert rc == 1 and re.search(r"REFUSED: unknown arm '(R0C|r0c|R0cc|R1c)' in RLS1_ARMS \(registered: " + re.escape(reg) + r"\)", out), (bad_arm, out); n += 1
    rc, out = run({"RLS1_ARMS": "R0c"}); assert rc == 1 and "REFUSED:" in out and "unknown arm" not in out and "missing: ship" in out, out; n += 1   # R0c passes the arm check; the (empty) package is what stops it
    rc, out = run(); assert rc == 1 and "REFUSED:" in out and "missing: ship" in out, out; n += 1                                 # past the P2 check: the package is empty
    # the S-reader and the two tests that pin the S gate are REQUIRED by the precheck (SAFEOBJ round 4): with every other listed file present, a package lacking any ONE of them is refused by the chain
    # itself, in seconds, naming it -- before the queue, the disk test and the GPU
    listed = re.search(r'for f in (.*?); do\n  \[ -f', text, re.S).group(1).replace("\\\n", " ").split()
    for skip in SAFEOBJ_FILES:
        assert skip in listed, skip; n += 1
        for f in listed:
            if f != skip:
                (pk / f).parent.mkdir(parents=True, exist_ok=True)
                (pk / f).write_text("")
        if (pk / skip).exists():
            (pk / skip).unlink()
        rc, out = run(); assert rc == 1 and f"/{skip} missing: ship" in out and "ZZRLS1_QUEUE_CLEAR" not in out, (skip, out); n += 1
    # a package with every required file, then the env stage, then model_final: absent -> refused; a wrong file -> refused
    for f in re.search(r'for f in (.*?); do\n  \[ -f', text, re.S).group(1).replace("\\\n", " ").split():
        (pk / f).parent.mkdir(parents=True, exist_ok=True)
        (pk / f).write_text("")
    shutil.copy(HERE / "rls1_checks.py", pk / "refe" / "rls1_checks.py")
    (w / "teacher_env.sh").write_text(f'export DRIVERL_EVAL_PYTHON="{posix(sys.executable)}"\n')
    (w / "data" / "backbones").mkdir(parents=True, exist_ok=True)
    env = {"RLS1_TICK": "0", "RLS1_DD_MB": "8"}
    rc, out = run({"RLS1_TICK": "0", "FAKE_GPU_PID": "777"}); assert rc == 1 and "REFUSED: waited" in out and "a process is on the GPU" in out, out; n += 1   # a trainer on the GPU: the QUEUE waits (bounded here), never trains beside it
    rc, out = run(env); assert rc == 1 and "REFUSED: model_final.pt is not the pinned file" in out and "missing:" in out, out; n += 1
    fin = w / "data" / "refe_runs" / "vitl16_navtrain10_grow_tau0.3" / "model_final.pt"
    fin.parent.mkdir(parents=True)
    fin.write_bytes(b"not the checkpoint")
    rc, out = run(env); assert rc == 1 and "REFUSED: model_final.pt is not the pinned file" in out and "size_mismatch" in out, out; n += 1
    return n


def fancmp_behaviour(text, tmp):
    """Stage `fancmp`, extracted VERBATIM from the chain and run under bash.  `$PY rl_fan_dump.py` resolves to a stub in the cwd that records its argv and replays a canned
    output and exit code, so what is asserted is the argv the CALLER really passes (the flag-to-caller trace) and what the stage does with every outcome."""
    n = 0
    helpers = "\n".join(re.findall(r"^(?:log|refuse|stage_done|stage_mark|stop_after)\(\) .*$", text, re.M))
    assert helpers.count("refuse()") == 1 and helpers.count("stop_after()") == 1, helpers
    blk = re.search(r"(?ms)^FANCMP_KEYS=.*?stop_after fancmp\n", text)
    assert blk, "stage fancmp not found"
    w = Path(tmp) / "fc"
    shutil.rmtree(w, ignore_errors=True)
    stub, pkg = w / "stub", w / "pkg"
    (pkg / "splits").mkdir(parents=True)
    stub.mkdir(parents=True)
    keys_src = PKGROOT / "splits" / "rl_fancmp_keys.tsv"
    shutil.copy(keys_src, pkg / "splits" / "rl_fancmp_keys.tsv")
    (stub / "rl_fan_dump.py").write_text("import json, os, sys\nopen(os.environ['FAKE_ARGV'], 'w').write(json.dumps(sys.argv[1:]))\n"
                                         "sys.stdout.write(open(os.environ['FAKE_OUT'], encoding='utf-8').read())\nsys.exit(int(os.environ.get('FAKE_RC', '0')))\n", encoding="utf-8")
    banked = w / "banked"
    banked.mkdir()
    (banked / "onpolicy_r0_w0.jsonl").write_text("{}\n", encoding="utf-8")
    argvf, outf = w / "argv.json", w / "out.txt"
    data = w / "data"
    script = w / "stage.sh"
    write_sh(script, f'set -u\nPKG={posix(pkg)}; DATA={posix(data)}; R=$DATA/rls1; S=$R/stage; L=$R/chain.log\nmkdir -p "$R" "$S"\n{helpers}\n'
                     f'PY={posix(sys.executable)}; FINAL=POD_FINAL.pt; FINAL_SHA256=abc123; TARGETS=POD_TARGETS; IMAGES=POD_PIXELS; CALIB=POD_CALIB.json\n'
                     f'cd {posix(stub)}\n{blk.group(0)}echo STAGE_PASSED\n')
    nk = len(keys_src.read_text(encoding="utf-8").split("\n")) - 2          # header and the final newline
    assert nk == 512, nk
    atol = re.search(r"^FANCMP_ATOL=(\S+)$", text, re.M).group(1)            # the chain's own PINNED bounds: check_fancmp ties them to round2/verdict_round2.json
    matol = re.search(r"^FANCMP_MEAN_ATOL=(\S+)$", text, re.M).group(1)
    v2 = json.loads((CAL_DIR / "round2" / "verdict_round2.json").read_text(encoding="utf-8"))
    assert atol == f"{v2['catastrophe_atol_kept']:g}" and matol == f"{v2['CHOSEN_MEAN_ATOL']:g}", (atol, matol); n += 1
    marker_bounds = f"atol={atol} mean_atol={matol}"
    stage_ok = data / "rls1" / "stage" / "fancmp.ok"
    BASE_ENV = {"RLS1_FANCMP_ATOL": "", "RLS1_FANCMP_MEAN_ATOL": ""}         # hermetic: an export in the developer's shell must not leak into the cases

    def go(rc, out, env=None, clean=True):
        if argvf.exists():
            argvf.unlink()
        if clean:
            shutil.rmtree(data, ignore_errors=True)
        outf.write_text(out, encoding="utf-8")
        r = bash_run(f'bash "{posix(script)}"', {"FAKE_ARGV": posix(argvf), "FAKE_OUT": posix(outf), "FAKE_RC": str(rc), "RLS1_BANKED_SETS": posix(banked), **BASE_ENV, **(env or {})})
        lg = (data / "rls1" / "chain.log")
        return r[0], r[1], (lg.read_text(encoding="utf-8") if lg.exists() else ""), stage_ok.exists()

    def passes(a=None, m=None, na=None, nm=None):
        """What a good dumper prints.  a / m = the bound each marker echoes; na / nm = the matched count on each marker."""
        return (f"ZZRLFAN_PLAN {nk}\nZZRLFAN_ATOL_PASS {nk if na is None else na} 0.25 {atol if a is None else a}\n"
                f"ZZRLFAN_MEAN_ATOL_PASS {nk if nm is None else nm} 0.0104167 {matol if m is None else m}\nZZRLFAN_DONE {nk} 0 0\n")
    PASS = passes()
    # ---- the good case, and the argv the CALLER passes
    rc, so, lg, ok = go(0, PASS)
    assert rc == 0 and "STAGE_PASSED" in so and ok and "ZZRLS1_FANCMP_OK ZZRLFAN_ATOL_PASS" in lg, (rc, so, lg); n += 1
    assert f"ZZRLFAN_MEAN_ATOL_PASS {nk} 0.0104167 {matol} (atol {atol}, mean_atol {matol}," in lg; n += 1                                   # the OK line carries BOTH verdicts and BOTH bounds
    assert stage_ok.read_text(encoding="utf-8").split("\n")[1] == marker_bounds; n += 1                                                       # the marker records the bounds it passed under
    argv = json.loads(argvf.read_text(encoding="utf-8"))

    def after(flag):
        return argv[argv.index(flag) + 1]
    assert argv[0] == "--ckpt" and after("--ckpt") == "POD_FINAL.pt" and after("--expect-ckpt-sha256") == "abc123"; n += 1             # model_final, hash-pinned
    assert after("--keys") == posix(pkg / "splits" / "rl_fancmp_keys.tsv") and argv.count("--keys") == 1; n += 1                              # the pinned slice, nothing else
    assert after("--compare-sets") == posix(banked) and after("--compare-min-matched") == str(nk); n += 1                                     # the gate flags REACH the dumper
    assert after("--compare-atol") == atol and after("--compare-mean-atol") == matol and argv.count("--compare-atol") == 1 and argv.count("--compare-mean-atol") == 1; n += 1   # BOTH pinned bounds, once each
    assert (after("--targets"), after("--images"), after("--calib")) == ("POD_TARGETS", "POD_PIXELS", "POD_CALIB.json"); n += 1
    assert after("--out").endswith("/rls1/fancmp") and after("--tag") == "fancmp" and after("--batch") == "4"; n += 1
    assert not ({"--allow-missing", "--limit", "--plan-only", "--shard", "--ranks", "--workers"} & set(argv)); n += 1                          # nothing narrows or reshapes the gate
    # ---- a finished stage is not re-run
    rc, so, lg, ok = go(0, PASS, clean=False)
    assert rc == 0 and not argvf.exists() and "STAGE_PASSED" in so; n += 1
    # ---- a marker written under OTHER bounds (or none, or a different spelling) is not trusted: the gate runs again and rewrites it
    for stale in ("2026-01-01T00:00:00\n", f"2026-01-01T00:00:00\natol=0.9 mean_atol={matol}\n", f"2026-01-01T00:00:00\natol={atol} mean_atol=0.05\n", f"2026-01-01T00:00:00\n{marker_bounds} \n"):
        go(0, PASS)
        stage_ok.write_text(stale, encoding="utf-8")
        rc, so, lg, ok = go(0, PASS, clean=False)
        assert rc == 0 and argvf.exists() and "was not written under the pinned bounds" in lg and stage_ok.read_text(encoding="utf-8").split("\n")[1] == marker_bounds, (stale, rc, lg); n += 1
    go(0, PASS)
    stage_ok.write_text(f"2026-01-01T00:00:00\n{marker_bounds}\n", encoding="utf-8")
    rc, so, lg, ok = go(0, PASS, clean=False)
    assert rc == 0 and not argvf.exists() and "was not written under" not in lg; n += 1                                                       # a marker under the pinned bounds IS trusted
    # ---- every refusal names its cause and leaves NO stage marker
    for name, args, want in (
            ("dumper exit 5 (a set beyond the catastrophe bound)", (5, f"ZZRLFAN_ATOL_FAIL {nk} 3 3.0 {atol}\nZZRLFAN_MEAN_ATOL_PASS {nk} 0.01 {matol}\nZZRLFAN_DONE {nk} 0 0\n"), "ZZRLFAN_ATOL_FAIL"),
            ("dumper exit 5 (a set beyond the MEAN bound only)", (5, f"ZZRLFAN_ATOL_PASS {nk} 0.3 {atol}\nZZRLFAN_MEAN_ATOL_FAIL {nk} 2 0.05 {matol}\nZZRLFAN_DONE {nk} 0 0\n"), "ZZRLFAN_MEAN_ATOL_FAIL"),
            ("exit 0 but no PASS marker", (0, f"ZZRLFAN_DONE {nk} 0 0\n"), "want 0 and ZZRLFAN_ATOL_PASS"),
            ("exit 0 with the max-bound PASS only (the mean verdict missing)", (0, f"ZZRLFAN_ATOL_PASS {nk} 0.25 {atol}\nZZRLFAN_DONE {nk} 0 0\n"), "want 0 and ZZRLFAN_ATOL_PASS"),
            ("exit 0 with the mean-bound PASS only (the max verdict missing)", (0, f"ZZRLFAN_MEAN_ATOL_PASS {nk} 0.01 {matol}\nZZRLFAN_DONE {nk} 0 0\n"), "want 0 and ZZRLFAN_ATOL_PASS"),
            ("PASS markers that echo ANOTHER max bound (the dumper ran with a different number)", (0, passes(a="0.9")), "want 0 and ZZRLFAN_ATOL_PASS"),
            ("PASS markers that echo ANOTHER mean bound", (0, passes(m="0.05")), "want 0 and ZZRLFAN_ATOL_PASS"),
            ("max-bound PASS with fewer matched keys than the slice", (0, passes(na=nk - 1)), "want 0 and ZZRLFAN_ATOL_PASS"),
            ("mean-bound PASS with fewer matched keys than the slice", (0, passes(nm=nk - 1)), "want 0 and ZZRLFAN_ATOL_PASS"),
            ("exit 4 (the dumper refused: nothing matched)", (4, "REFUSED: --compare-sets matched NO key -- a gate that compares nothing proves nothing\n"), "REFUSED: --compare-sets matched NO key"),
            ("exit 4 (the dumper refused the mean flag)", (4, "REFUSED: --compare-mean-atol needs --compare-atol (x)\n"), "REFUSED: --compare-mean-atol needs --compare-atol"),
            ("exit 3 (images missing on this pod)", (3, passes(na=nk - 2, nm=nk - 2).replace(f"ZZRLFAN_PLAN {nk}", f"ZZRLFAN_PLAN {nk - 2}")), "want 0 and ZZRLFAN_ATOL_PASS"),
            ("PASS markers printed AFTER a failing exit", (5, PASS), "exit 5")):
        rc, so, lg, ok = go(*args)
        assert rc == 1 and "STAGE_PASSED" not in so and not ok and "REFUSED: the fan-equivalence gate FAILED" in lg and "fancmp.log" in lg and want in lg, (name, rc, lg); n += 1
    # ---- the bounds are PINNED, not inputs: an export that equals the pin (or is empty) is harmless; any other value is a refusal BEFORE the dumper runs, and the argv carries the PIN
    for env, pin, flag, what in (("RLS1_FANCMP_ATOL", atol, "--compare-atol", "catastrophe"), ("RLS1_FANCMP_MEAN_ATOL", matol, "--compare-mean-atol", "mean")):
        equal = ("", pin, f"{float(pin):.3f}", f"{float(pin):.5f}", ("." + pin.split(".")[1]) if pin.startswith("0.") else pin)
        for val in equal:
            rc, so, lg, ok = go(0, PASS, {env: val})
            a = json.loads(argvf.read_text(encoding="utf-8"))
            assert rc == 0 and ok and a[a.index(flag) + 1] == pin, (env, val, rc, lg); n += 1                                                   # accepted, and the dumper still gets the pinned spelling
        for val in (f"{float(pin) * 1.8:g}", f"{float(pin) / 2:g}", f"{float(pin) * 1.0001:.6g}", f"{float(pin) * 0.9999:.6g}", "0", "-" + pin, "abc", pin + "x", f"{float(pin):e}", " " + pin, "1", "10"):
            rc, so, lg, ok = go(0, PASS, {env: val})
            assert rc == 1 and not argvf.exists() and not ok and f"{env}='{val}' is not the pinned {what} bound" in lg, (env, val, rc, lg); n += 1
    rc, so, lg, ok = go(0, PASS, {"RLS1_FANCMP_ATOL": atol, "RLS1_FANCMP_MEAN_ATOL": matol})
    assert rc == 0 and ok; n += 1                                                                                                             # both exported at the pins: fine
    rc, so, lg, ok = go(0, PASS, {"RLS1_FANCMP_ATOL": atol, "RLS1_FANCMP_MEAN_ATOL": "0.5"})
    assert rc == 1 and not argvf.exists() and "RLS1_FANCMP_MEAN_ATOL='0.5' is not the pinned mean bound" in lg; n += 1                         # one right export does not excuse the other
    rc, so, lg, ok = go(0, PASS, {"RLS1_FANCMP_ATOL": "0.025", "RLS1_FANCMP_MEAN_ATOL": matol})
    assert rc == 1 and not argvf.exists() and "RLS1_FANCMP_ATOL='0.025' is not the pinned catastrophe bound" in lg; n += 1                    # the two numbers are not interchangeable
    # ---- refusals BEFORE the dumper runs
    rc, so, lg, ok = go(0, PASS, {"RLS1_BANKED_SETS": posix(w / "nowhere")})
    assert rc == 1 and not argvf.exists() and "no banked step-10,075 sets" in lg and not ok; n += 1                                            # absent banked dir
    empty = w / "empty"
    empty.mkdir(exist_ok=True)
    rc, so, lg, ok = go(0, PASS, {"RLS1_BANKED_SETS": posix(empty)})
    assert rc == 1 and not argvf.exists() and "no banked step-10,075 sets" in lg; n += 1                                                       # present but empty
    ks = pkg / "splits" / "rl_fancmp_keys.tsv"
    good = ks.read_bytes()
    ks.write_bytes(good + b"deadbeefdeadbeef\t2021.01.01.00.00.00_veh-00_00000_00000\n")
    rc, so, lg, ok = go(0, PASS)
    assert rc == 1 and not argvf.exists() and "not the pinned fan-equivalence slice" in lg; n += 1                                             # a different slice is a different gate
    ks.write_bytes(good)
    rc, so, lg, ok = go(0, PASS, {"RLS1_BANKED_SETS": posix(banked) + "/."})
    assert rc == 0 and after_flag(json.loads(argvf.read_text(encoding="utf-8")), "--compare-sets").endswith("banked/."); n += 1                  # RLS1_BANKED_SETS reaches the dumper
    # ---- the default banked dir is P2's SETS path under $DATA
    rc, so, lg, ok = go(0, PASS, {"RLS1_BANKED_SETS": ""})
    assert rc == 1 and "no banked step-10,075 sets at " in lg, lg; n += 1                                                                      # an EMPTY override falls back to the default path, which is absent in this sandbox
    # ---- the dry-run stop
    rc, so, lg, ok = go(0, PASS, {"RLS1_STOP_AFTER": "fancmp"})
    assert rc == 0 and "STAGE_PASSED" not in so and "stopping after fancmp" in lg and ok; n += 1
    return n


def after_flag(argv, flag):
    return argv[argv.index(flag) + 1]


def helper_checks(tmp):
    n = 0
    # ---- pins against their sources
    regs = [p / "Project Steering" / "MODEL_REGISTRY.md" for start in (Path.cwd(), HERE, PKGROOT) for p in (start, *start.parents)]
    reg_file = next((p for p in regs if p.is_file()), None)       # the file itself, not a directory of the same name (a junction on this dev box points elsewhere)
    assert reg_file is not None, "MODEL_REGISTRY.md not found above the package"
    reg = reg_file.read_text(encoding="utf-8")
    assert RC.PINS["model_final"]["sha256"] in reg and RC.PINS["model_final"]["md5"] in reg; n += 1
    assert RC.PINS["u400"]["md5"].startswith("8e382eeb") and str(RC.PINS["u400"]["size"]) in (PKGROOT / "STOP_CONSEQUENCES_2026-10-11.md").read_text(encoding="utf-8").replace(",", ""); n += 1
    real = {"model_final": Path("E:/Projects/TanitAD/data/refe_runs_eval/model_final.pt"), "u400": Path("E:/Projects/TanitAD/data/refe_runs_eval/sft4_peek/model_sft4_B_u400.pt")}
    for pin, f in real.items():
        if f.is_file():
            ok, why = RC.verify_file(str(f), pin); assert ok, why; n += 1
        else:
            print(f"ZZTEST_SKIP real {pin} file absent")
    # ---- verify_file in both directions, on small files with their own pins
    f = tmp / "ck.bin"
    f.write_bytes(b"abcdef")
    import hashlib
    good = {"x": {"size": 6, "md5": hashlib.md5(b"abcdef").hexdigest(), "sha256": hashlib.sha256(b"abcdef").hexdigest()}}
    assert RC.verify_file(str(f), "x", good)[0]; n += 1
    assert RC.verify_file(str(f), "x", {"x": {**good["x"], "size": 7}})[1].startswith("size_mismatch"); n += 1
    f2 = tmp / "ck2.bin"; f2.write_bytes(b"abcdeX")                                              # same size, one byte different
    assert RC.verify_file(str(f2), "x", good)[1].startswith("md5_mismatch"); n += 1
    assert RC.verify_file(str(f), "x", {"x": {**good["x"], "sha256": "0" * 64}})[1].startswith("sha256_mismatch"); n += 1
    assert RC.verify_file(str(tmp / "nope.bin"), "x", good)[1].startswith("missing"); n += 1
    assert RC.verify_file(str(f), "model_final")[1].startswith("size_mismatch"); n += 1           # the real pin refuses a wrong file
    assert RC.verify_file(str(f), "nope", good)[1].startswith("unknown_pin"); n += 1
    # ---- the frozen partition: the real files pass; every tamper is refused
    pr = PKGROOT / "raw" / "rl_pool" / "partition_runs"
    PN = re.search(r"^PART=.*?; PNAME=(\S+)$", CHAIN.read_text(encoding="utf-8"), re.M).group(1)
    mf, tr, va = pr / f"{PN}.manifest.json", pr / PN / "rl_train_scenes.tsv", pr / PN / "rl_val_scenes.tsv"
    ok, why = RC.verify_partition(str(mf), str(tr), str(va)); assert ok, why; n += 1
    t2 = tmp / "rl_train_scenes.tsv"; t2.write_bytes(tr.read_bytes()[:-1] + b"X")
    assert RC.verify_partition(str(mf), str(t2), str(va))[1].startswith("train_tsv_sha256_mismatch"); n += 1
    m2 = tmp / "m.json"; m2.write_bytes(mf.read_bytes().replace(b"20261012", b"20261013", 1))
    assert RC.verify_partition(str(m2), str(tr), str(va))[1].startswith("manifest_sha256_mismatch"); n += 1
    assert RC.verify_partition(str(mf), str(tr), str(va), {**RC.PARTITION, "partition_id": "0" * 64})[1].startswith("partition_id_mismatch"); n += 1
    assert RC.verify_partition(str(mf), str(tr), str(va), {**RC.PARTITION, "n_val": 2999})[1].startswith("counts"); n += 1
    old = pr / "A_no_trained_list_seed20261012.manifest.json"                                      # the 48,000-scene partition is no longer the pinned one
    assert old.is_file() and RC.verify_partition(str(old), str(pr / "A_no_trained_list_seed20261012" / "rl_train_scenes.tsv"), str(va))[1].startswith("manifest_sha256_mismatch"); n += 1
    assert RC.PARTITION["partition_id"] != RC.OLD_PARTITION_ID and RC.PARTITION["n_train"] == 16000 and RC.PARTITION["n_val"] == 3000; n += 1
    ltr_ = {l for _, l in RC.read_scenes(str(tr))}
    old_tr = RC.read_scenes(str(pr / "A_no_trained_list_seed20261012" / "rl_train_scenes.tsv"))
    assert RC.read_scenes(str(tr)) == old_tr[:16000] and len(ltr_) == 166; n += 1                      # the re-cut is an exact PREFIX of the frozen train order (same tool, same seed)
    assert RC.read_scenes(str(va)) == RC.read_scenes(str(pr / "A_no_trained_list_seed20261012" / "rl_val_scenes.tsv")); n += 1   # RL-val is byte-identical to the frozen one
    mini = tmp / "mini"; mini.mkdir()
    for nm, rows in (("rl_train_scenes.tsv", [("t1", "L1"), ("t2", "L2")]), ("rl_val_scenes.tsv", [("t3", "L3"), ("t4", "L2")])):   # L2 is on BOTH sides
        (mini / nm).write_bytes(("token\tlog\n" + "".join(f"{t}\t{l}\n" for t, l in rows)).encode())
    sh = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731
    mm = mini / "m.json"
    mm.write_text(json.dumps({"verify": "clean", "partition_id": "pid", "train": {"file": "rl_train_scenes.tsv", "sha256": sh(mini / "rl_train_scenes.tsv")},
                              "val": {"file": "rl_val_scenes.tsv", "sha256": sh(mini / "rl_val_scenes.tsv")}}))
    pins = {"partition_id": "pid", "manifest_sha256": sh(mm), "n_train": 2, "n_val": 2, "min_val_logs": 1}
    assert RC.verify_partition(str(mm), str(mini / "rl_train_scenes.tsv"), str(mini / "rl_val_scenes.tsv"), pins)[1].startswith("logs_on_both_sides"); n += 1
    assert RC.write_val_logs(str(va), str(tmp / "vl.txt")) == 75 and (tmp / "vl.txt").read_text().count("\n") == 75; n += 1
    # ---- the teacher-lane labels that reach the 7th output
    scenes = [("t1", "L1"), ("t2", "L1"), ("t3", "L2")]
    tsv = tmp / "x.tsv"; tsv.write_text("token\tlog\n" + "".join(f"{t}\t{l}\n" for t, l in scenes))
    td = tmp / "tl"; td.mkdir()
    prop = lambda v: [{"center_line.CenterLine.info": v}] * 64  # noqa: E731
    (td / "teacher_lane_r0.jsonl").write_text("".join(json.dumps(r) + "\n" for r in (
        {"key": ["L1", "t1", 0, 0], "ckpt_step": RC.CKPT_STEP, "props": prop(0.5)},
        {"key": ["L1", "t2", 0, 0], "ckpt_step": 5, "props": prop(0.5)},               # another checkpoint's proposals: stale, the trainer ignores it
        {"key": ["L2", "t3", 0, 0], "ckpt_step": RC.CKPT_STEP, "props": prop(None)},   # no distance: masked
        {"key": ["L9", "t9", 0, 0], "ckpt_step": RC.CKPT_STEP, "props": prop(0.5)})), encoding="utf-8")   # not a scene of this side
    ok, why, good_n = RC.tlane_check(str(td), str(tsv), RC.CKPT_STEP, 1); assert ok and good_n == 1, why; n += 1
    assert not RC.tlane_check(str(td), str(tsv), RC.CKPT_STEP, 2)[0]; n += 1
    assert not RC.tlane_check(str(tmp / "empty_dir"), str(tsv), RC.CKPT_STEP, 1)[0]; n += 1
    n += tlane_exact_checks(tmp)
    # ---- THE JOIN GATE: ok, then every way below 100 % / broken identity / a dump that is not the partition side
    jg = lambda sd, pd, td_: RC.join_gate("train", str(sd), str(pd), str(td_), str(tsv), RC.CKPT_STEP, n_prop=N)  # noqa: E731
    d1 = mk_dump(tmp / "j1", scenes); p1, t1 = mk_labels(tmp / "j1", scenes)
    ok, why, rep = jg(d1, p1, t1); assert ok, why; n += 1
    assert rep["pdm_labelled"] == rep["terms_labelled"] == 3; n += 1
    d2 = mk_dump(tmp / "j2", scenes); p2_, t2_ = mk_labels(tmp / "j2", scenes, skip_pdm={"t3"})
    assert jg(d2, p2_, t2_)[1].startswith("labels_below_100_percent:pdm 2/3 terms 3/3"); n += 1                  # a PDM label missing (2 of 3 sets)
    d3 = mk_dump(tmp / "j3", scenes); p3, t3 = mk_labels(tmp / "j3", scenes, skip_terms={"t1"})
    assert jg(d3, p3, t3)[1].startswith("labels_below_100_percent:pdm 3/3 terms 2/3"); n += 1                  # an rl_terms record missing
    d4 = mk_dump(tmp / "j4", scenes); p4, t4 = mk_labels(tmp / "j4", scenes, ckpt=RC.CKPT_STEP - 1)
    assert jg(d4, p4, t4)[1].startswith("labels_below_100_percent"); n += 1                                     # labels of ANOTHER checkpoint's proposals do not count
    d5 = mk_dump(tmp / "j5", scenes); p5, t5 = mk_labels(tmp / "j5", scenes, bad_id={"t2"})
    assert jg(d5, p5, t5)[1].startswith("G_REWARD_ID_broken"); n += 1                                           # g > 0 on a DAC = 1 proposal
    d6 = mk_dump(tmp / "j6", scenes[:2]); p6, t6 = mk_labels(tmp / "j6", scenes[:2])
    assert jg(d6, p6, t6)[1].startswith("dump_is_not_the_partition_side"); n += 1                              # a scene was never dumped
    d7 = mk_dump(tmp / "j7", scenes + [("t8", "L8")]); p7, t7 = mk_labels(tmp / "j7", scenes + [("t8", "L8")])
    assert jg(d7, p7, t7)[1].startswith("dump_is_not_the_partition_side"); n += 1                              # a foreign scene was dumped
    d8 = mk_dump(tmp / "j8", scenes, ckpt=7); p8, t8 = mk_labels(tmp / "j8", scenes, ckpt=7)
    assert jg(d8, p8, t8)[1].startswith("ckpt_step"); n += 1                                                    # not model_final's fan
    d9 = mk_dump(tmp / "j9", scenes); mk_dump(tmp / "j9", scenes, name="onpolicy_r0_w0.jsonl"); p9, t9 = mk_labels(tmp / "j9", scenes)
    assert jg(d9, p9, t9)[1].startswith("foreign_or_duplicate_sets"); n += 1                                    # a teacher-labelled file beside the lean dump
    dA = mk_dump(tmp / "jA", scenes); pA, tA = mk_labels(tmp / "jA", scenes, fires=1)
    assert jg(dA, pA, tA)[1].startswith("record_shape"); n += 1                                                 # label_fires is not a boolean
    assert not RC.join_gate("train", str(tmp / "nodump"), str(p1), str(t1), str(tsv), RC.CKPT_STEP, n_prop=N)[0]; n += 1
    assert RC.join_gate("train", str(d1), str(p1), str(t1), str(tsv), RC.CKPT_STEP, n_prop=N, expect=4)[1].startswith("partition_side_size"); n += 1
    # the CLI prints the marker the chain greps
    r = subprocess.run([sys.executable, "-I", str(HERE / "rls1_checks.py"), "join-gate", "--side", "train", "--sets", str(d1), "--pdm", str(p1), "--terms", str(t1),
                        "--tsv", str(tsv), "--n-prop", str(N)], capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.startswith("ZZRLS1_JOIN_OK"), r.stdout + r.stderr; n += 1
    r = subprocess.run([sys.executable, "-I", str(HERE / "rls1_checks.py"), "join-gate", "--side", "train", "--sets", str(d2), "--pdm", str(p2_), "--terms", str(t2_),
                        "--tsv", str(tsv), "--n-prop", str(N)], capture_output=True, text=True)
    assert r.returncode == 1 and r.stdout.startswith("ZZRLS1_JOIN_REFUSED"), r.stdout; n += 1
    # ---- the kill rules
    rec = lambda tag, g3=True, g5=True, k4=False, k5=False: {"tag": tag, "g3_ok": g3, "g5_ok": g5, "k4_breach": k4, "k5_breach": k5}  # noqa: E731
    assert RC.judge("R0", rec("u500", g3=False), [rec("u250", g3=True)])[0] is None; n += 1                    # one breach is a flag, not the stop
    assert RC.judge("R0", rec("u500", g3=False), [rec("u250", g3=False)])[0] == "K3"; n += 1
    assert RC.judge("R0", rec("u500", g5=False), [rec("u250", g3=False)])[0] == "K3"; n += 1                   # g3 then g5: both are K3 telemetry
    assert RC.judge("R0", rec("u500", g3=False), [rec("u250", g3=False)], k3_enabled=False)[0] is None; n += 1
    assert RC.judge("R0", rec("u500", k4=True), [rec("u250", k4=True)])[0] == "K4"; n += 1
    assert RC.judge("R0", rec("u500", k4=True), [rec("u250", k4=False)])[0] is None; n += 1
    assert RC.judge("R2", rec("u250", k5=True), [])[0] == "K5"; n += 1                                          # K5 needs no second check
    assert RC.judge("R0", rec("u250", k5=True), [])[0] is None; n += 1                                          # ... and is R2's alone
    st = RC.lane_change_stats([True, False, False, False], [True, True, False, False])
    assert (st["rate"], st["base_rate"], st["precision"], st["recall"], st["breach"]) == (0.25, 0.5, 1.0, 0.5, False); n += 1
    assert RC.lane_change_stats([True, True, True, False], [False, False, False, True])["breach"]; n += 1        # 0.75 > 2 x 0.25
    assert not RC.lane_change_stats([True, True, False, False], [False, False, False, True])["breach"]; n += 1   # exactly 2 x is not ABOVE 2 x
    assert RC.lane_change_stats([True, False], [False, False])["breach"]; n += 1                                 # base rate 0: any lane change is above 2 x 0
    # ---- monitor / replay with a fake trainer
    cnt = [0]

    def run_mon(arm, events, recs, now=None, alive=(True, False), boom=()):
        cnt[0] += 1
        od = tmp / f"mon_{arm}_{cnt[0]}"
        od.mkdir(parents=True)
        (od / "sft.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")
        killed, seq = [], list(alive)

        def chk(tag, files):
            if tag in boom:
                raise RuntimeError("boom")
            return recs[tag]
        clock = iter(now) if now else None
        rc = RC.monitor(arm, str(od), chk, 99, lambda p: seq.pop(0) if len(seq) > 1 else seq[0], killed.append, sleep_fn=lambda s: None,
                        now_fn=(lambda: next(clock)) if clock else (lambda: 0.0), start=0.0)
        return rc, killed, od
    dump = lambda tag: {"event": "score_dump", "tag": tag, "files": {"B": "b", "A": "a", "base": "z"}}  # noqa: E731
    ev = [dump("step0"), dump("update250"), dump("update500")]
    ok_recs = {"step0": rec("step0"), "update250": rec("update250"), "update500": rec("update500")}
    rc, killed, od = run_mon("R0", ev, ok_recs); assert rc == 0 and not killed and not (od / "KILLED.json").exists(); n += 1
    assert len((od / "kill_checks.jsonl").read_text().splitlines()) == 2; n += 1
    k4 = {**ok_recs, "update250": rec("update250", k4=True), "update500": rec("update500", k4=True)}
    rc, killed, od = run_mon("R0", ev, k4); assert rc == 2 and killed == [99] and json.loads((od / "KILLED.json").read_text())["code"] == "K4"; n += 1
    k3 = {**ok_recs, "update250": rec("update250", g3=False), "update500": rec("update500", g3=False)}
    rc, killed, od = run_mon("R1", ev, k3); assert rc == 2 and json.loads((od / "KILLED.json").read_text())["code"] == "K3"; n += 1
    k3b = {**k3, "step0": rec("step0", g3=False)}                                                               # the BASE already breaches g3 on RL-val: K3 is off, loudly
    rc, killed, od = run_mon("R1", ev, k3b); assert rc == 0 and not killed; n += 1
    rc, killed, od = run_mon("R2", ev, {**ok_recs, "update250": rec("update250", k5=True)}); assert rc == 2 and json.loads((od / "KILLED.json").read_text())["code"] == "K5"; n += 1
    rc, killed, od = run_mon("R0", ev, {**ok_recs, "update250": rec("update250", k5=True)}); assert rc == 0 and not killed; n += 1
    rc, killed, od = run_mon("R0", ev, ok_recs, boom=("update250",)); assert rc == 0 and not killed; n += 1     # a check that raises does not kill (replay re-judges it)
    nan = [{"event": "train", "update": 25, "bceA": 0.3, "lossB": float("nan")}]
    rc, killed, od = run_mon("C0", nan, {}); assert rc == 2 and json.loads((od / "KILLED.json").read_text())["code"] == "K2"; n += 1
    rc, killed, od = run_mon("C0", ev, {}); assert rc == 0 and not killed; n += 1                               # a control is not judged on K3 / K4 / K5
    rc, killed, od = run_mon("R0", [dump("step0")], ok_recs, now=[7 * 3600.0] * 4); assert rc == 2 and json.loads((od / "KILLED.json").read_text())["code"] == "K6"; n += 1
    od2 = tmp / "rep"; od2.mkdir(); (od2 / "sft.jsonl").write_text("".join(json.dumps(e) + "\n" for e in ev), encoding="utf-8")
    assert RC.replay("R0", str(od2), lambda tag, f: k4[tag])[0] == "K4"; n += 1                                 # the replay finds what a flaky monitor missed
    assert RC.replay("R0", str(od2), lambda tag, f: ok_recs[tag])[0] is None; n += 1
    assert RC.replay("C0", str(od2), lambda tag, f: 1 / 0)[0] is None; n += 1
    # ---- R0c is an R arm (RL-S1c): judged on K3 / K4 / K6 and the replay like R0, never as a control, and K5 stays R2's alone
    rc, killed, od = run_mon("R0c", ev, k4); assert rc == 2 and killed == [99] and json.loads((od / "KILLED.json").read_text())["code"] == "K4"; n += 1
    rc, killed, od = run_mon("R0c", ev, k3); assert rc == 2 and json.loads((od / "KILLED.json").read_text())["code"] == "K3"; n += 1
    rc, killed, od = run_mon("R0c", ev, ok_recs); assert rc == 0 and not killed and len((od / "kill_checks.jsonl").read_text().splitlines()) == 2; n += 1
    rc, killed, od = run_mon("R0c", ev, {**ok_recs, "update250": rec("update250", k5=True)}); assert rc == 0 and not killed; n += 1
    rc, killed, od = run_mon("R0c", [dump("step0")], ok_recs, now=[7 * 3600.0] * 4); assert rc == 2 and json.loads((od / "KILLED.json").read_text())["code"] == "K6"; n += 1
    assert RC.replay("R0c", str(od2), lambda tag, f: k4[tag])[0] == "K4"; n += 1
    assert RC.replay("R0c", str(od2), lambda tag, f: ok_recs[tag])[0] is None; n += 1
    # ---- the default checker (in-process rl_val_eval on a tiny RL-val) so the monitor's real path is exercised
    n += checker_smoke(tmp)
    return n


def tlane_exact_checks(tmp, RCm=None, tag="te"):
    """TL-1: `tlane_exact` and `updates_check` in both directions.  RCm = the module under test (the real one, or a source mutation of it)."""
    RCm = RCm or RC
    n = 0
    CK = RC.CKPT_STEP
    scenes = [("t1", "L1"), ("t2", "L1"), ("t3", "L2")]
    base = Path(tmp) / tag
    shutil.rmtree(base, ignore_errors=True)
    base.mkdir(parents=True)
    tsv = base / "side.tsv"
    tsv.write_text("token\tlog\n" + "".join(f"{t}\t{l}\n" for t, l in scenes), encoding="utf-8")
    sd = mk_dump(base, scenes)

    def row(l, t, **kw):
        d = {"kind": "teacher_lane_labels", "key": [l, t, 0, 0], "ckpt_step": CK, "repair": "last_heading_hold", "teacher": {}, "props": [{"center_line.CenterLine.info": 0.3}] * 64}
        d.update(kw)
        return d

    def write(name, rows, tail=""):
        d = base / name
        d.mkdir(parents=True)
        (d / "teacher_lane_r0_s0of1.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows) + tail, encoding="utf-8")
        return d
    full = [row(l, t) for t, l in scenes]
    ok, why, rep = RCm.tlane_exact(str(write("ok", full)), str(sd), str(tsv), CK, 64, 3); assert ok and rep["accepted"] == 3 and rep["torn"] == 0, why; n += 1
    ok, why, rep = RCm.tlane_exact(str(write("torn", full, tail='{"kind": "teacher_lane_la')), str(sd), str(tsv), CK, 64, 3); assert ok and rep["torn"] == 1, why; n += 1   # a kill's fragment
    bad = {
        "missing": (full[:2], "missing:1"), "foreign": (full + [row("L9", "t9")], "foreign:1"), "doubled": (full + [full[0]], "doubled:1"),
        "other_ckpt": ([row("L1", "t1", ckpt_step=5)] + full[1:], "rejected_rows:1"), "no_repair": ([row("L1", "t1", repair=None)] + full[1:], "rejected_rows:1"),
        "null_distance": ([row("L1", "t1", props=[{"center_line.CenterLine.info": None}] * 64)] + full[1:], "rejected_rows:1"),
        "short_fan": ([row("L1", "t1", props=[{"center_line.CenterLine.info": 0.3}] * 63)] + full[1:], "rejected_rows:1"),
        "rank_one": ([row("L1", "t1", key=["L1", "t1", 0, 1])] + full[1:], "rejected_rows:1"),
        "wrong_kind": ([row("L1", "t1", kind="onpolicy_set")] + full[1:], "rejected_rows:1"),
        "empty": ([], "missing:3"),
    }
    for name, (rows, want) in bad.items():
        ok, why, _ = RCm.tlane_exact(str(write("bad_" + name, rows)), str(sd), str(tsv), CK, 64, 3)
        assert not ok and want in why, (name, ok, why); n += 1
    ok, why, _ = RCm.tlane_exact(str(base / "ok"), str(sd), str(tsv), CK, 64, 4); assert not ok and "dumped_count:3!=4" in why, why; n += 1          # the partition's count is not 3
    t2 = base / "side4.tsv"
    t2.write_text("token\tlog\n" + "".join(f"{t}\t{l}\n" for t, l in scenes + [("t4", "L2")]), encoding="utf-8")
    ok, why, _ = RCm.tlane_exact(str(base / "ok"), str(sd), str(t2), CK, 64, None); assert not ok and "dump_is_not_the_side" in why, why; n += 1    # a scene that was never dumped
    t3 = base / "side_swapped.tsv"
    t3.write_text("token\tlog\nt1\tL2\nt2\tL1\nt3\tL2\n", encoding="utf-8")
    ok, why, _ = RCm.tlane_exact(str(base / "ok"), str(sd), str(t3), CK, 64, None); assert not ok and "dump_is_not_the_side" in why, why; n += 1    # token under another log
    sd_dup = mk_dump(base / "dupdump", scenes + [("t1", "L1")])
    ok, why, _ = RCm.tlane_exact(str(base / "ok"), str(sd_dup), str(tsv), CK, 64, None); assert not ok and "dump_not_clean" in why, why; n += 1   # a set dumped twice
    # updates_check: the pre-registered budget is 1,000 updates (batch 8 x accum 8)
    uc = RCm.updates_check
    assert uc("--max-updates 1000 --epochs 2", 48000)[0]; n += 1                                   # the original pool: 1,500 capped to 1,000
    ok, why = uc("--max-updates 1000 --epochs 2", 16000); assert not ok and "planned_updates:500!=max_updates:1000" in why, why; n += 1   # TL-1's trap: 2 epochs of 16,000 = 500
    assert uc("--max-updates 1000 --epochs 4", 16000) == (True, "1000 updates over 16000 sets"); n += 1
    assert not uc("--epochs 4", 16000)[0] and not uc("--epochs 4", 0)[0]; n += 1                  # no --max-updates: nothing pins the budget (and 0 sets would "equal" 0 updates)
    assert not uc("--max-updates 1000 --epochs 4", 15000)[0]; n += 1                                 # a pool that lost scenes: 938 updates
    assert RCm.planned_updates("--epochs 1.0 --max-updates 1000", 48000)[0] == 750; n += 1          # the handoff's documented silent trap
    assert RCm.planned_updates("--max-updates 1000 --epochs 4 --batch 4 --accum 8", 16000)[0] == 1000; n += 1
    # the CLI markers
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc1 = RCm.main(["tlane-exact", "--dir", str(base / "ok"), "--sets", str(sd), "--tsv", str(tsv), "--expect", "3", "--json", str(base / "te.json")])
        rc2 = RCm.main(["tlane-exact", "--dir", str(base / "bad_missing"), "--sets", str(sd), "--tsv", str(tsv)])
        rc3 = RCm.main(["updates-check", "--args", "--max-updates 1000 --epochs 4", "--n-sets", "16000"])
        rc4 = RCm.main(["updates-check", "--args", "--max-updates 1000 --epochs 2", "--n-sets", "16000"])
    out = buf.getvalue()
    assert (rc1, rc2, rc3, rc4) == (0, 1, 0, 1) and out.count("ZZRLS1_TLANE_OK") == 1 and out.count("ZZRLS1_TLANE_REFUSED") == 1, out
    assert out.count("ZZRLS1_UPDATES_OK") == 1 and out.count("ZZRLS1_UPDATES_REFUSED") == 1 and json.loads((base / "te.json").read_text())["ok"] is True; n += 1
    if RCm is RC:                                                                                      # the real module: now every source mutation of it must turn this red
        src = (HERE / "rls1_checks.py").read_text(encoding="utf-8")
        muts = [
            ("exact: missing rows ignored", 'if rep["missing"]:', "if False:"),
            ("exact: foreign rows ignored", 'if rep["foreign"]:', "if False:"),
            ("exact: doubled rows ignored", 'if rep["doubled"]:', "if False:"),
            ("exact: rejected rows ignored", 'if rep["rejected"]:', "if False:"),
            ("exact: the partition's count ignored", "if expect is not None and len(dumped) != expect:", "if False:"),
            ("exact: dump need not be the side", "if len(scenes) != len(dumped) or any(scenes.get(k[1]) != k[0] for k in dumped):", "if False:"),
            ("exact: a doubled dump accepted", "if dup_dump or stray:", "if False:"),
            ("exact: the recipe marker not required", 'd.get("repair") == "last_heading_hold"', "True"),
            ("exact: rank, checkpoint and fan size not required", 'int(k[3]) == 0 and int(d["ckpt_step"]) == ckpt_step and len(props) == n_prop and d.get("repair")',
             'True and True and True and d.get("repair")'),
            ("exact: any kind accepted", 'and d.get("kind") == "teacher_lane_labels"', "and True"),
            ("updates: no budget needed", "if mx <= 0:", "if False:"),
            ("updates: a shortfall accepted", "if total != mx:", "if False:"),
            ("updates: accum ignored", "total = int(math.ceil(n_sets / batch * epochs / accum))", "total = int(math.ceil(n_sets / batch * epochs))"),
            ("updates: the cap ignored", "return (min(total, mx) if mx else total), mx", "return total, mx"),
        ]
        for name, old, new in muts:
            assert src.count(old) == 1, f"mutation {name}: snippet occurs {src.count(old)} times"
            m = types.ModuleType("rls1_checks_mut")
            m.__file__ = str(HERE / "rls1_checks.py")
            sys.modules["rls1_checks_mut"] = m
            exec(compile(src.replace(old, new), m.__file__, "exec"), m.__dict__)
            try:
                tlane_exact_checks(tmp, m, tag="mut")
            except Exception:                                                                          # noqa: BLE001  (any failure IS the suite seeing the mutation)
                n += 1
                continue
            raise AssertionError(f"rls1_checks mutation '{name}' stayed GREEN")
    return n


def checker_smoke(tmp):
    import numpy as np
    import rl_val_eval as RV
    M, T = 8, 6
    d = tmp / "cs"
    for sub in ("pdm", "terms", "sets", "scores_c", "scores_r"):
        (d / sub).mkdir(parents=True)
    keys = [("LA", "ta", 0, 0), ("LA", "tb", 0, 0), ("LB", "tc", 0, 0), ("LB", "td", 0, 0)]
    ep = [0.9, 0.8, 0.7, 0.6, 0.5, 0.95, 0.85, 0.75]
    pdms = [(5 * e + 5 + 2) / 12 for e in ep]
    (d / "scenes.tsv").write_text("token\tlog\n" + "".join(f"{k[1]}\t{k[0]}\n" for k in keys))
    with open(d / "pdm" / "pdm_a.jsonl", "w") as f, open(d / "terms" / "rl_terms_a.jsonl", "w") as g, open(d / "sets" / "onpolicy_rlfan_fan_s0of1.jsonl", "w") as h:
        for k in keys:
            f.write(json.dumps({"kind": "pdm_targets", "key": list(k), "ckpt_step": 5, "nc": [1.0] * M, "dac": [1.0] * M, "ep": ep, "ttc": [1.0] * M, "c": [1.0] * M,
                                "ddc": [1.0] * M, "pdms": pdms}) + "\n")
            g.write(json.dumps({"kind": "rl_terms", "key": list(k), "ckpt_step": 5, "d": [0.0] * M, "k": [0.0] * M, "lc_executed": [0] * M, "dither": [0] * M,
                                "label_fires": False}) + "\n")
            h.write(json.dumps({"kind": "onpolicy_set", "log_name": k[0], "token": k[1], "step": 0, "rank": 0, "ckpt_step": 5,
                                "traj": [[[0.0, 0.0]] + [[float(i + 1) * (j + 1), 0.0] for i in range(T - 1)] for j in range(M)]}) + "\n")

    def scores(pick):
        s = np.full((len(keys), M, 7), -8.0)
        s[:, pick, :6] = 8.0
        s[:, :, 6] = 8.0
        return s
    for tag, pick_c, pick_r in (("update250", 0, 0), ("update500", 0, 1)):
        RV.write_scores_npz(str(d / "scores_c" / f"scores_{tag}_A.npz"), keys, [5] * len(keys), scores(pick_c))
        RV.write_scores_npz(str(d / "scores_r" / f"scores_{tag}_B.npz"), keys, [5] * len(keys), scores(pick_r))
    chk = RC.RlValChecker("R0", str(d / "scores_c"), str(d / "pdm"), str(d / "terms"), str(d / "sets"), str(d / "scenes.tsv"), n_boot=50)
    same = chk("update250", {"B": str(d / "scores_r" / "scores_update250_B.npz")})                             # R picks what C picks
    assert same["k4_breach"] is False and same["pdms_delta_x100"] == 0.0 and same["n_sets"] == 4 and same["g3"] is not None; n = 2
    worse = chk("update500", {"B": str(d / "scores_r" / "scores_update500_B.npz")})                           # R picks proposal 1 (EP 0.8): below C by 100 * 0.1 * 5/12
    assert abs(worse["pdms_delta_x100"] + 100 * 0.1 * 5 / 12) < 1e-9 and worse["k4_breach"] is True and worse["k5_breach"] is False; n += 1
    return n


if __name__ == "__main__":
    sys.exit(main())
