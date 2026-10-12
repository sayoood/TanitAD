#!/usr/bin/env python3
"""Tests for the RL-S1 pod chain: refe/rls1_pod.sh and refe/rls1_checks.py (eval/RLS1_POD_RUNBOOK.md).  No pod, no GPU, no network.

  1. `check_chain(text)`: static properties the chain must hold (P2 over, pinned checkpoints, partition, teacher lane, coverage + remedy, the join gate, the per-arm
     order preflight -> gate -> token -> train, the token VERIFIED not grepped, egress, kill monitor).  Every property is broken in a copy: `check_chain` must say so.
  2. BEHAVIOUR, extracted verbatim from the chain and run under bash: p2_over / gpu_busy (lock held, GPU busy, P2 not done), cgroup_cores (v1 and v2), arm_args
     (the five argv compared TOKEN FOR TOKEN with the gate section of eval/HANDOFF_RL_S1.md, then judged by the REAL launch gate: CLEAN, and a C arm carrying
     --epo / an R arm without it / R0 with K,L ... REFUSED), token_ok against tokens minted by the gate's own mint() for refe_rls1.
  3. The REAL rls1_pod.sh executed in a sandbox up to its first refusals (lock held, GPU busy, P2 unfinished, no HF_TOKEN, missing or wrong model_final).
  4. rls1_checks.py in both directions (pins, partition, teacher lane, the join gate below 100 % / with a G-REWARD-ID break, the kill rules, monitor, replay).

Run: python refe/test_rls1_chain.py   -> ZZTEST_PASS <n>   (needs bash on PATH; Git Bash on the dev box).   Any skipped check prints ZZTEST_SKIP and is NOT counted.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
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
CHILD_RX = re.compile(r'"\$PY" +(scorer_finetune\.py|rls1_checks\.py|rl_fan_dump\.py|rl_partition\.py|rl_pool_available\.py|onpolicy_relabel_pdm\.py|rl_terms_label\.py|'
                      r'pdm_label_complete\.py|launch_gate\.py|hf_egress\.py)|"\$PY" +"\$PKG/refe/launch_gate\.py"|bash +"\$PKG/code/|^sleep |\bsleep +\d+ |\bdd +if=')
ORDER = ["p2_over)\" || refuse", "source \"$WORK/teacher_env.sh\"", "--pin model_final", "--pin u400", "verify-partition", "tlane-check", "rl_fan_dump.py --plan-only",
         "STAGES=dbs", "PT._setup()", "dump_one val 0 1", "rls1_checks.py join-gate", "--tag rlfan_data", "scorer_finetune.py $ARGS --preflight",
         "launch_gate.py run --profile refe_rls1", "scorer_finetune.py $ARGS >>", "rls1_checks.py replay", "--rl-val-scenes $VAL_TSV", "--tag \"arm_$ARM\""]
PRESENT = ["onpolicy_relabel_pdm.py --sets", "rl_terms_label.py --sets", "dump_one()", "label_one()"]
TOKEN_OK_RX = re.compile(r"^token_ok\(\) \{\n.*?^\}\n", re.S | re.M)


def fn_body(text, name):
    m = re.search(rf"^{name}\(\) \{{\n.*?^\}}\n", text, re.S | re.M)
    return m.group(0) if m else ""


def arm_block(text, arm):
    m = re.search(rf"^    {arm}\) .*$", fn_body(text, "arm_args"), re.M)
    return m.group(0) if m else ""


def token_line_ok(ln):
    s = ln.strip()
    return (s.startswith("#") or "env -u HF_TOKEN" in s or s == '[ -n "${HF_TOKEN:-}" ] || refuse "HF_TOKEN is not set in the environment"'
            or re.match(r'if \[ -n "\$\{PY:-\}" \] && \[ -d "\$PKG/refe" \] && \[ -n "\$\{HF_TOKEN:-\}" \]; then', s) is not None)


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
    if "gpu_busy" not in po or "--query-compute-apps=pid" not in gb or "|| return 0" not in gb:
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
    if len(re.findall(r'grep -q "\^ZZRLS1_TLANE_OK" "\$R/tlane_(?:train|heldout)\.log" \|\| refuse', text)) != 2:
        v.append("TLANE_UNCHECKED")
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
        for needle in ('launch_gate.py" verify', '--profile refe_rls1', '--args "$ARGS"', '--pkg "$PKG"', '--token "$tok"'):
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
    return v


def mutations(text):
    sw = lambda a, b: text.replace(a, b)  # noqa: E731
    out = [
        ("no P2 lock check", drop_line(text, '9< "$P2/chain.lock"'), "NO_P2_LOCK_CHECK"),
        ("no GPU check", drop_line(text, "if gpu_busy; then"), "NO_GPU_CHECK"),
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
        ("teacher-lane verdict unchecked", drop_line(text, 'grep -q "^ZZRLS1_TLANE_OK" "$R/tlane_heldout.log"'), "TLANE_UNCHECKED"),
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
        ("token_ok forgets the profile", sw("verify --profile refe_rls1 --token", "verify --token"), "TOKEN_OK_NOT_VERIFYING"),
        ("the trainer starts without the token check", drop_line(text, "scorer_finetune.py is NOT started"), "TRAIN_WITHOUT_TOKEN"),
        ("the gate log is grepped for PASS", sw('    token_ok || refuse "the launch gate did not mint', '    grep -q "^ZZGATE-PASS-" "$ARMDIR/launch_gate.log" || refuse "x"\n    token_ok || refuse "the launch gate did not mint'),
         "TOKEN_GREPPED"),
        ("the trainer keeps fd 200", sw('"$ARMDIR/sft.log" 2>&1 ) 200>&- &', '"$ARMDIR/sft.log" 2>&1 ) &'), "FD200"),
        ("the gate child keeps fd 200", sw('> "$ARMDIR/launch_gate.log" 2>&1 200>&-', '> "$ARMDIR/launch_gate.log" 2>&1'), "FD200"),
        ("the trainer is not its own process group", sw("set -m\n", ""), "TRAINER_NOT_OWN_PROCESS_GROUP"),
        ("the manifest is not required", sw(" SHIP_MANIFEST.json \\", " \\"), "SHIP_MANIFEST_NOT_REQUIRED"),
        ("the key lives in the shipped tree", sw('export TANITAD_GATE_KEY_FILE="$R/launch_gate.key"', 'export TANITAD_GATE_KEY_FILE="$PKG/launch_gate.key"'), "KEY_NOT_OUTSIDE_TREE"),
        ("echo the token", sw('log "env ok', 'log "env ok token=$HF_TOKEN'), "HF_TOKEN_USE"),
        ("xtrace on", sw("set -u\n", "set -ux\n", ), "XTRACE"),
        ("the 2026-10-11 mangled continuation", sw("code/pod_teacher_env.sh code/pod_dataprep.sh", "code/pod_teacher_env.sh \\n code/pod_dataprep.sh"), "REQUIRED_FILE_ABSENT:\\n"),
        ("a required file the package does not ship", sw("refe/rl_partition.py ", "refe/rl_partition_gone.py "), "REQUIRED_FILE_ABSENT:refe/rl_partition_gone.py"),
        ("preflight after the gate", (lambda a, b: text.replace(a + b, b + a))(*[re.search(p, text, re.S).group(0) for p in
            (r"  # ---- 9a\..*?(?=  # ---- 9b)", r"  # ---- 9b\..*?(?=  # ---- 9c)")]), "ORDER"),
        ("tar without --no-same-owner", sw("log \"env ok", "tar -xf x.tar -C /tmp\nlog \"env ok"), "TAR_OWNER"),
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
        rc, out, _ = po(RLS1_NVSMI=posix(nv_busy)); assert rc == 1 and "GPU" in out; n += 1       # a process on the GPU
        rc, out, _ = po(RLS1_NVSMI=posix(nv_broken)); assert rc == 1 and "GPU" in out; n += 1     # nvidia-smi cannot say: not provably idle
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
            env = {"PY": posix(sys.executable), "PKG": posix(PKGROOT), "LG": posix(lgdir), "ARMDIR": posix(armdir), "ARGS": args, "PYTHONIOENCODING": "utf-8",
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
        # ---- 3. the REAL chain, executed in a sandbox up to its first refusals
        n += sandbox_runs(text, tmp)
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
    write_sh(bindir / "flock", '#!/bin/bash\nif [ "$2" = 200 ]; then exit 0; fi\nexit "${FAKE_FLOCK_RC:-0}"\n')
    write_sh(bindir / "nvidia-smi", '#!/bin/bash\nif [ -n "${FAKE_GPU_PID:-}" ]; then echo "$FAKE_GPU_PID"; fi\nexit 0\n')
    script = tmp / "rls1_sandbox.sh"
    write_sh(script, text.replace("WORK=/workspace", f"WORK={posix(w)}").replace("sleep 1800 200>&-", "sleep 1 200>&-"))
    R = w / "data" / "rls1"
    (w / "data" / "refe_p2" / "chain.lock").write_text("")
    (w / "data" / "refe_p2" / "chain.log").write_text("ZZP2R_DONE\n")
    base_env = {"PATH": posix(bindir) + os.pathsep + os.environ["PATH"], "HF_TOKEN": "hf_notarealtoken", "HF_HUB_OFFLINE": "1", "RLS1_DD_MB": "8"}

    def run(env_extra=None, drop=()):
        shutil.rmtree(R, ignore_errors=True)
        env = {k: v for k, v in {**base_env, **(env_extra or {})}.items() if k not in drop}
        e = {k: v for k, v in os.environ.items() if k not in drop}
        e.update(env)
        r = subprocess.run([shutil.which("bash"), posix(script)], capture_output=True, text=True, env=e, timeout=180)
        log = (R / "chain.log").read_text(encoding="utf-8") if (R / "chain.log").exists() else ""
        return r.returncode, r.stdout + log
    rc, out = run({"FAKE_FLOCK_RC": "1"}); assert rc == 1 and "REFUSED: the P2 chain still holds" in out, out; n += 1          # P2's lock is held
    rc, out = run({"FAKE_GPU_PID": "777"}); assert rc == 1 and "REFUSED: a process is on the GPU" in out, out; n += 1             # a trainer on the GPU
    (w / "data" / "refe_p2" / "chain.log").write_text("ZZP2_FAILED\n")
    rc, out = run(); assert rc == 1 and "REFUSED: P2 never logged ZZP2R_DONE" in out, out; n += 1                                # P2 unfinished
    (w / "data" / "refe_p2" / "chain.log").write_text("ZZP2R_DONE\n")
    rc, out = run(drop=("HF_TOKEN",)); assert rc == 1 and "REFUSED: HF_TOKEN is not set" in out, out; n += 1                    # no token
    rc, out = run(); assert rc == 1 and "REFUSED:" in out and "missing: ship" in out, out; n += 1                                  # past the P2 check: the package is empty
    # a package with every required file, then the env stage, then model_final: absent -> refused; a wrong file -> refused
    for f in re.search(r'for f in (.*?); do\n  \[ -f', text, re.S).group(1).replace("\\\n", " ").split():
        (pk / f).parent.mkdir(parents=True, exist_ok=True)
        (pk / f).write_text("")
    shutil.copy(HERE / "rls1_checks.py", pk / "refe" / "rls1_checks.py")
    (w / "teacher_env.sh").write_text(f'export DRIVERL_EVAL_PYTHON="{posix(sys.executable)}"\n')
    (w / "data" / "backbones").mkdir(parents=True, exist_ok=True)
    env = {"RLS1_TICK": "0", "RLS1_DD_MB": "8"}
    rc, out = run(env); assert rc == 1 and "REFUSED: model_final.pt is not the pinned file" in out and "missing:" in out, out; n += 1
    fin = w / "data" / "refe_runs" / "vitl16_navtrain10_grow_tau0.3" / "model_final.pt"
    fin.parent.mkdir(parents=True)
    fin.write_bytes(b"not the checkpoint")
    rc, out = run(env); assert rc == 1 and "REFUSED: model_final.pt is not the pinned file" in out and "size_mismatch" in out, out; n += 1
    return n


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
    mf, tr, va = pr / "A_no_trained_list_seed20261012.manifest.json", pr / "A_no_trained_list_seed20261012" / "rl_train_scenes.tsv", pr / "A_no_trained_list_seed20261012" / "rl_val_scenes.tsv"
    ok, why = RC.verify_partition(str(mf), str(tr), str(va)); assert ok, why; n += 1
    t2 = tmp / "rl_train_scenes.tsv"; t2.write_bytes(tr.read_bytes()[:-1] + b"X")
    assert RC.verify_partition(str(mf), str(t2), str(va))[1].startswith("train_tsv_sha256_mismatch"); n += 1
    m2 = tmp / "m.json"; m2.write_bytes(mf.read_bytes().replace(b"20261012", b"20261013", 1))
    assert RC.verify_partition(str(m2), str(tr), str(va))[1].startswith("manifest_sha256_mismatch"); n += 1
    assert RC.verify_partition(str(mf), str(tr), str(va), {**RC.PARTITION, "partition_id": "0" * 64})[1].startswith("partition_id_mismatch"); n += 1
    assert RC.verify_partition(str(mf), str(tr), str(va), {**RC.PARTITION, "n_val": 2999})[1].startswith("counts"); n += 1
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
    # ---- the default checker (in-process rl_val_eval on a tiny RL-val) so the monitor's real path is exercised
    n += checker_smoke(tmp)
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
