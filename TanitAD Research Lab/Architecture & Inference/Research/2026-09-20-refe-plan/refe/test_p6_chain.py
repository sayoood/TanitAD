#!/usr/bin/env python3
"""Tests for the P6 pod chain: refe/p6_pod.sh, refe/p6_prep.py, gate_fixture.carve_p6_gate_bank, hf_egress.gate --ckpt, gate_checks.check_egr (SPEC_P6 D10).

No pod, no GPU, no network, no NAVSIM stack. Four layers:
  1. `check_chain(text)` -- static properties the chain must hold (the P2 chain's, plus P6's: the open yaw decision, P2's data reused and never re-fetched, the
     metric cache, D6 / D7 refusing, the three pod-run instruments, a NEW private repo, P2's chain not running, the shipped tree not P2's).
  2. MUTATIONS of the chain text: each property is broken in a copy and `check_chain` must say so. A check that cannot go RED is not a check.
  3. The helpers, each in both directions (p6_prep, the P6 smoke-bank carve, the G-EGR first-checkpoint size, the expanded ARGS against the refe_p6 profile).
  4. The REAL p6_pod.sh run under bash in a sandbox with stubbed tools: stage order, every stage skipped when done and resumed after an interruption, an argv change voiding
     what depends on it, the token verified before the trainer, the yaw refusal, D6 / D7 refusals, and NO token in any file. Behavioural mutants (echo the token, re-fetch) are caught too.

Run: python refe/test_p6_chain.py   -> ZZP6CHAIN_PASS <n>   (needs bash on PATH; Git Bash on the dev box)
Never prints ZZTEST_SKIP (the launch gate's G-SUITE treats a skip as a failure); an optional real-data check that is absent prints a plain ZZINFO line.
"""
from __future__ import annotations

import json
import lzma
import os
import pickle
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from string import Template

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import test_p2r_chain as T2  # noqa: E402   (logical_commands, drop_line, token_line_ok)
import p6_prep as PP  # noqa: E402

CHAIN = HERE / "p6_pod.sh"
ARGS_RX = re.compile(r'^ARGS="(.*)"$', re.M)
# the training arguments, pinned. Variables stay symbolic here; the expanded string is checked against the refe_p6 profile in layer 3.
ARGS_REF = ("--backbone vitl16 --targets $CBANK --images $PIX --calib $BANK/calib_table.json --metric-cache $MC --scorer-sees-trunk --yaw-wrap $YAW_WRAP "
            "--nav-mode command --labeller online --label-workers $LW --label-chunks $LC --epochs 25 --epoch-unit scenes --batch 4 --accum 64 --lr 2e-4 --weight-decay 0.01 "
            "--amp bf16 --score-w 0.1 --tf32 --compile --workers 2 --log-every 1 --ckpt-every-min 20 --out $RUN --resume")
CHILD_RX = re.compile(r'"\$PY" +(train\.py|p2r_sync\.py|hf_egress\.py|p6_prep\.py|build_metric_cache\.py|annotate_commands\.py|diag_p6_labeller_identity\.py|launch_gate\.py|-c)'
                      r'|"\$PY" +"\$PKG/refe/launch_gate\.py"|bash +"\$PKG/code/|apt-get|uv pip|\bcurl\b|\btar +-x|\bsleep +["\d$]|\bdd +if=|\bfind .* -exec')
FETCH_RX = re.compile(r'bash +"\$PKG/code/pod_dataprep\.sh"|p2r_sync\.py restore|\bcurl +-|apt-get install|uv pip install')
TOKEN_OK_RX = re.compile(r"^token_ok\(\) \{\n.*?^\}\n", re.S | re.M)
ORDER = ['source "$WORK/teacher_env.sh"', '"$PY" p2r_sync.py watch', "STAGES=weights", "p2r_sync.py recover", "p6_prep.py command-table", "annotate_commands.py --bank",
         "ZZD9_PASS", "p6_prep.py lane-ranks", "\nARGS=", "\nlab_stage() {", "\nwait_clear\n", "train.py $MEM_ARGS", "hf_egress.py gate",
         "launch_gate.py run --profile refe_p6", "train.py $ARGS --preflight", "train.py $ARGS >>", "p2r_sync.py final"]
STAGE_DONE = ("weights", "recover", "commands", "cache", "lanes", "launch_gate")
KEY_DONE = ("lab", "mem", "egr")
STOPS = ("env", "weights", "recover", "inputs", "commands", "cache", "lanes", "lab", "wait", "earlylab", "mem", "egr", "launch_gate")
TOKEN = "hf_" + "Q" * 24                      # built at runtime: no token-shaped literal in the repo


def code_of(text):
    return "\n".join(ln for ln in text.split("\n") if not ln.strip().startswith("#"))


def refetch_violations(text):
    code = code_of(text)
    spans = list(re.finditer(r"^(fetch_\w+)\(\) \{\n.*?^\}\n", code, re.S | re.M))
    outside = code
    for m in spans:
        outside = outside.replace(m.group(0), "")
    v = []
    if FETCH_RX.search(outside):
        v.append("FETCH_OUTSIDE_FETCH_FUNCTION")
    names = sorted(m.group(1) for m in spans)
    for need in ("fetch_bank", "fetch_dbs", "fetch_pix", "fetch_navlogs"):
        if need not in names:
            v.append("FETCH_FUNCTION_MISSING:" + need)
    for nm in names:
        x = nm[len("fetch_"):]
        calls = list(re.finditer(r"(?<![\w])" + nm + r"(?![\w(])", outside))
        if not calls:
            v.append("FETCH_NEVER_CALLED:" + x)
        for m in calls:
            before = outside[max(0, m.start() - 40):m.start()]
            if not re.search(r"have_" + x + r" \|\| $", before):
                v.append("REFETCH_UNGUARDED:" + x)
    return v


def check_chain(text, ref_args=ARGS_REF):
    v = []
    if "\r" in text:
        v.append("CRLF")
    am = ARGS_RX.findall(text)
    if len(am) != 1:
        v.append("ARGS_DEFINED_TWICE")
    elif am[0] != ref_args:
        v.append("ARGS_CHANGED")
    # the stage-0 write test is sized for P6's disk need (P6_ON_11VCPU section 5): a default far below it proves nothing about the volume
    dd = re.findall(r'DDGB="\$\{P6_DD_GB:-\$\(\[ "\$\{P6_FREE_NAVLOGS:-0\}" = 1 \] && echo (\d+) \|\| echo (\d+)\)\}"', text)
    if len(dd) != 1 or int(dd[0][0]) < 72 or int(dd[0][1]) < 96:
        v.append("DD_TEST_UNDERSIZED")
    # the NAVSIM metadata is pinned to a REVISION with its size and sha256, and a mismatch refuses (MEASURED 2026-10-12)
    if "/resolve/main/" in text or not re.search(r"/resolve/[0-9a-f]{40}/openscene-v1\.1/openscene_metadata_trainval\.tgz", text):
        v.append("NAVLOGS_NOT_PINNED")
    if not re.search(r'sha256sum "\$DATA/navsim_logs_dl/meta\.tgz".*= "\$NAVLOGS_SHA256" \] \|\| \{', text) \
            or not re.search(r'= "\$NAVLOGS_BYTES" \] \|\| \{', text):
        v.append("NAVLOGS_NOT_VERIFIED")
    for ln in text.split("\n"):
        if "HF_TOKEN" in ln and not T2.token_line_ok(ln):
            v.append("HF_TOKEN_USE:" + ln.strip()[:60])
    if re.search(r"^\s*set\s+-[a-wyz]*x|\bbash\s+-x\b", text, re.M):
        v.append("XTRACE")
    if '[ -n "${HF_TOKEN:-}" ] || refuse' not in text:
        v.append("NO_TOKEN_ASSERT")
    for c in T2.logical_commands(text):
        if CHILD_RX.search(c) and "200>&-" not in c:
            v.append("FD200:" + c[:70])
    code = code_of(text)
    pos = [code.find(k) for k in ORDER]
    for k, p in zip(ORDER, pos):
        if p < 0:
            v.append("MISSING:" + k.strip())
    if all(p >= 0 for p in pos) and pos != sorted(pos):
        v.append("ORDER:" + ",".join(ORDER[i].strip() for i in range(1, len(ORDER)) if pos[i] < pos[i - 1]))
    # ---- the open PI decision P6-Y: ONE variable, validated, never defaulted, never a literal in ARGS
    # PI ruling P6-Y (2026-10-11): unwrapped yaw => the default is `off`; `-` and not `:-`, so a BLANKED variable is not silently refilled
    if len(re.findall(r'^YAW_WRAP="\$\{P6_YAW_WRAP-off\}"$', text, re.M)) != 1 or len(re.findall(r"^\s*YAW_WRAP=", text, re.M)) != 1:
        v.append("YAW_NOT_ONE_VARIABLE")
    if re.search(r"--yaw-wrap +(on|off)\b", code):
        v.append("YAW_LITERAL")
    if not re.search(r"yaw_check\(\) \{\n  case \"\$YAW_WRAP\" in\n    off\) return 0 ;;\n  esac\n  refuse ", text):
        v.append("YAW_CHECK_NOT_STRICT")
    if code.count("\nyaw_check\n") < 2:
        v.append("YAW_CHECK_NOT_CALLED")                   # at the top (cheap, before any CPU is spent) and again right before ARGS
    elif code.find("\nyaw_check\n", code.find("\nyaw_check\n") + 1) > code.find("\nARGS="):
        v.append("YAW_CHECKED_AFTER_ARGS")
    # ---- P2 is over, and P6 is not shipped over P2's tree
    # the lock probe is READ-WRITE (`9<>`; EBADF on nfs4 otherwise), over P2, RL-S1 and S1, each REFUSED; only the P2 seed replicate may run beside P6's CPU-only stages
    if not re.search(r'^lock_held\(\) \{ \[ -e "\$1/chain\.lock" \] && ! \( flock -n 9 \) 9<> "\$1/chain\.lock"; \}$', text, re.M) \
            or not re.search(r'for u in "\$P2DIR" "\$DATA/rls1" "\$DATA/s1aux"; do\n  if lock_held "\$u"; then refuse ', text):
        v.append("P2_GUARD_MISSING")
    if not re.search(r'if lock_held "\$P2REPD"; then\n  \[ "\$\{P6_OVERLAP:-1\}" = 1 \] \|\| refuse .*\n  grep -q "ZZP2REP_TRAIN_START" "\$P2REPD/chain\.log" 2> /dev/null \\\n    \|\| refuse ', text):
        v.append("OVERLAP_NOT_GATED_ON_TRAINING")
    # ---- OVERLAP: hidden CUDA until the wait, restored after it, before the first GPU stage; the wait names the replicate's DONE marker; pools capped, niced, guarded
    code = code_of(text)
    i_hide, i_wait, i_restore, i_mem = (code.find('export CUDA_VISIBLE_DEVICES=""'), code.find("\nwait_clear\n"), code.find('export CUDA_VISIBLE_DEVICES="$CVD_SAVED"'),
                                        code.find("train.py $MEM_ARGS"))
    if i_hide < 0:
        v.append("CUDA_NOT_HIDDEN_BESIDE_TRAINER")
    if i_restore < 0 or not (i_wait < i_restore < i_mem):
        v.append("CUDA_NOT_RESTORED_BEFORE_GPU_STAGE")
    if i_wait < 0:
        v.append("WAIT_BEFORE_GPU_STAGE_MISSING")
    elif not (0 <= i_hide < i_wait < i_mem):
        v.append("WAIT_ORDER")
    wb = re.search(r"^wait_clear\(\) \{\n.*?^\}\n", code, re.S | re.M)
    if not wb or 'grep -q "ZZP2REP_DONE"' not in wb.group(0) or "gpu_busy" not in wb.group(0) or 'lock_held "$u"' not in wb.group(0) or '"$P2REPD"' not in wb.group(0):
        v.append("WAIT_CONDITIONS_INCOMPLETE")
    if len(re.findall(r'--workers "\$CWX"', code)) != 3 or '--workers "$CW"' in code:
        v.append("POOLS_NOT_CAPPED_BESIDE_TRAINER")
    if len(re.findall(r'nice -n "\$NICEX"', code)) != 2 or "nice -n 10" in code:
        v.append("NOT_NICED_BESIDE_TRAINER")
    if len(re.findall(r"guard_watch \$!", code)) != 3:
        v.append("NO_STEP_TIME_GUARD")
    if 'REFE_PAUSE_FILE="$PAUSE_FILE"' not in code:
        v.append("PAUSE_FILE_NOT_PASSED")
    if not re.search(r'^PKG="\$\{P6_PKG:-\$WORK/refe-plan-p6\}"$', text, re.M) or re.search(r"refe-plan(/|\")", code):
        v.append("P6_PKG_IS_P2_DIR")
    # ---- a NEW private repo, on every HF call; the watcher on the 30-minute cadence
    if not re.search(r'^REPO="\$\{P6_REPO:-Sayood/tanitad-refe-p6-[a-z-]+\}"', text, re.M) or "tanitad-refe-p2-joint-pdm" in text:
        v.append("NOT_A_NEW_REPO")
    for c in T2.logical_commands(text):
        m = re.search(r'p2r_sync\.py (watch|recover|final|tick)\b', c)
        if m and '--repo "$REPO"' not in c:
            v.append("NO_NEW_REPO:" + m.group(1))
    w = re.search(r'"\$PY" p2r_sync\.py watch.*?200>&- &', text, re.S)
    if not w or '--interval-min "${P6_SYNC_MIN:-30}"' not in w.group(0) or '--heavy-min "${P6_HEAVY_MIN:-30}"' not in w.group(0):
        v.append("NO_PERIODIC_EGRESS")
    if "p2r_sync.py final" not in text or "ZZP2R_FINAL_OK" not in text:
        v.append("NO_FINAL_EGRESS")
    elif not text.rstrip().endswith("exit 1"):
        v.append("FINAL_FAILURE_NOT_PROPAGATED")
    # ---- P2's data reused in place: a fetch exists only in a fetch_* function and is reached only as `have_X || fetch_X`
    v += refetch_violations(text)
    # ---- workers from the cgroup quota, never nproc
    if "/sys/fs/cgroup/cpu.max" not in text or not re.search(r"^LW=.*\bCORES\b", text, re.M) or not re.search(r"^CW=.*\bCORES\b", text, re.M):
        v.append("WORKERS_NOT_FROM_CGROUP")
    if re.search(r"nproc", re.sub(r"^cgroup_cores\(\) \{\n.*?^\}\n", "", code, flags=re.S | re.M)):
        v.append("WORKERS_FROM_NPROC")
    if '"^PREFLIGHT_OK"' not in text:
        v.append("NO_PREFLIGHT_CHECK")
    # ---- the labeller instrument measures the LAUNCH's chunk count (2026-10-18): identity, the stale refusal and throughput are properties of the chunked path
    if not re.search(r'^LC="\$\{P6_LABEL_CHUNKS:-[0-9]+\}"', text, re.M) or code.count('--chunks "$LC"') != 2:
        v.append("LAB_NOT_AT_LAUNCH_CHUNKS")
    if 'key_done lab "w$LW-c$LC"' not in text or 'key_mark lab "w$LW-c$LC"' not in text:
        v.append("LAB_KEY_IGNORES_CHUNKS")
    # ---- every stage: skippable when done, marked, dry-runnable; a re-measured thing voids what depends on it
    for s in STAGE_DONE:
        if f"stage_done {s}" not in text or f"stage_mark {s}" not in text:
            v.append("STAGE_NOT_SKIPPABLE_OR_MARKED:" + s)
    for s in KEY_DONE:
        if f"key_done {s} " not in text or f"key_mark {s} " not in text:
            v.append("STAGE_NOT_SKIPPABLE_OR_MARKED:" + s)
    for s in STOPS:
        if f"stop_after {s}" not in text:
            v.append("NO_STOP_AFTER:" + s)
    for needle in ("invalidate cache lanes lab mem egr launch_gate", "invalidate lanes lab mem egr launch_gate", "invalidate lab mem egr launch_gate",
                   "invalidate egr launch_gate"):
        if needle not in text:
            v.append("INVALIDATION_MISSING:" + needle)
    if text.count("invalidate launch_gate") < 2:
        v.append("INVALIDATION_MISSING:invalidate launch_gate")
    # ---- D6 / D7 are explicit and refuse
    for needle in ('refuse "D6: no NAVSIM metadata', 'refuse "D6: the command table', 'refuse "D7: the DriveRL route patch', 'refuse "D7: the rank-$k twin build failed',
                   'refuse "D7: rank-$k twins are NOT'):
        if needle not in text:
            v.append("D6_D7_NOT_ENFORCED:" + needle[8:30])
    if 'DRIVERL_EVAL_ROUTE_LANE_RANK="$k"' not in text or 'exec env -u HF_TOKEN -u DRIVERL_EVAL_ROUTE_LANE_RANK' not in text:
        v.append("ROUTE_ENV_NOT_SET_FOR_TWINS_ONLY")
    if 'P6_D7_FALLBACK:-0}" = 1' not in text:
        v.append("D7_FALLBACK_NOT_EXPLICIT")
    # ---- the instruments feed the gate, the gate gets all three
    gl = re.search(r'launch_gate\.py run --profile refe_p6.*?200>&-', text, re.S)
    if not gl:
        v.append("NO_LAUNCH_GATE_STAGE")
    else:
        for needle in ("--gate-bank", "--gate-lab-json", "--gate-mem-metrics", "--gate-egr-json", '--args "$ARGS"', '--pkg "$PKG"'):
            if needle not in gl.group(0):
                v.append("GATE_INPUT_MISSING:" + needle)
    if '--ckpt "$P6/mem_run/ckpt_last.pt"' not in text:
        v.append("EGR_WITHOUT_CKPT_SIZE")
    if 'p6_prep.py mem-argv --args "$ARGS"' not in text or text.count("train.py $MEM_ARGS") != 1:
        v.append("MEM_RUN_NOT_FROM_ARGS")
    # ---- the BINDING launch gate: the token is verified against the exact argv before the preflight and before every launch
    v += check_launch_gate(text)
    return v


def prev_code_line(text, needle):
    """The last non-blank, non-comment line BEFORE the line that carries `needle` ('' if none): a token check must sit directly in front of the launch it guards."""
    lines = text.split("\n")
    for i, ln in enumerate(lines):
        if needle in ln and not ln.strip().startswith("#"):
            j = i - 1
            while j >= 0 and (not lines[j].strip() or lines[j].strip().startswith("#")):
                j -= 1
            return lines[j].strip() if j >= 0 else ""
    return ""


def check_launch_gate(text):
    v = []
    code = code_of(text)
    run_pos = code.find("launch_gate.py run --profile refe_p6")
    if "stage_mark launch_gate" not in text:
        v.append("LAUNCH_GATE_STAGE_NOT_MARKED")
    am = ARGS_RX.search(text)
    if am and run_pos >= 0 and am.start() > text.find("launch_gate.py run --profile refe_p6"):
        v.append("ARGS_AFTER_GATE")
    m = TOKEN_OK_RX.search(text)
    if not m:
        v.append("NO_TOKEN_OK_FUNCTION")
    else:
        body = m.group(0)
        for needle in ('launch_gate.py" verify', "--profile refe_p6", '--args "$ARGS"', '--pkg "$PKG"', '--token "$tok"'):
            if needle not in body:
                v.append("TOKEN_OK_NOT_VERIFYING:" + needle)
        if re.search(r"\bgrep\b|^\s*return 0|\btrue\b", body, re.M):
            v.append("TOKEN_OK_NOT_VERIFYING:grep_or_constant")
    if not prev_code_line(text, "train.py $ARGS --preflight").startswith("token_ok || refuse"):
        v.append("PREFLIGHT_WITHOUT_TOKEN")
    if not prev_code_line(text, "train.py $ARGS >>").strip().startswith("token_ok || refuse"):
        v.append("TRAIN_WITHOUT_TOKEN")
    if text.count("token_ok ||") < 4:
        v.append("TOKEN_CHECKS_MISSING")
    if run_pos >= 0 and 0 <= code.find("train.py $ARGS") < run_pos:
        v.append("TRAINER_BEFORE_GATE")
    for ln in text.split("\n"):
        if not ln.strip().startswith("#") and "grep" in ln and ("ZZGATE-" in ln or "launch_gate.log" in ln):
            v.append("TOKEN_GREPPED:" + ln.strip()[:60])
    if "SHIP_MANIFEST.json" not in text:
        v.append("SHIP_MANIFEST_NOT_REQUIRED")
    if not re.search(r'^export TANITAD_GATE_KEY_FILE="\$P6/', text, re.M):
        v.append("KEY_NOT_OUTSIDE_TREE")
    return v


def mutations(text):
    """(name, mutated text, violation prefix that must appear)."""
    d = T2.drop_line
    out = [
        ("delete the final push", d(text, "p2r_sync.py final"), "NO_FINAL_EGRESS"),
        ("metadata from main", text.replace("/resolve/a76f840b65e972bc45e56c2adced897498e9a026/", "/resolve/main/"), "NAVLOGS_NOT_PINNED"),
        ("dd test back to 8 GB", text.replace("echo 96", "echo 8"), "DD_TEST_UNDERSIZED"),
        ("dd test undersized when the metadata is freed", text.replace("echo 72", "echo 8"), "DD_TEST_UNDERSIZED"),
        ("metadata sha256 unchecked", d(text, '= "$NAVLOGS_SHA256" ] ||'), "NAVLOGS_NOT_VERIFIED"),
        ("metadata size unchecked", d(text, '= "$NAVLOGS_BYTES" ] ||'), "NAVLOGS_NOT_VERIFIED"),
        ("delete the periodic watcher", re.sub(r'"\$PY" p2r_sync\.py watch.*?200>&- &\n', "", text, flags=re.S), "NO_PERIODIC_EGRESS"),
        ("hourly egress", text.replace("P6_SYNC_MIN:-30", "P6_SYNC_MIN:-360"), "NO_PERIODIC_EGRESS"),
        ("watcher pushes to P2's repo", text.replace('--repo "$REPO" --run "$RUN" --logs "$P6" --chain-log "$L" --outbox "$P6/outbox" --stop-file', '--run "$RUN" --logs "$P6" --chain-log "$L" --outbox "$P6/outbox" --stop-file'),
         "NO_NEW_REPO:watch"),
        ("P2's repo name", text.replace("tanitad-refe-p6-paper-faithful", "tanitad-refe-p2-joint-pdm"), "NOT_A_NEW_REPO"),
        ("delete the preflight check", d(text, '"^PREFLIGHT_OK"'), "NO_PREFLIGHT_CHECK"),
        ("train child keeps fd 200", text.replace('"$PY" train.py $ARGS >> "$P6/train.log" 2>&1 200>&-', '"$PY" train.py $ARGS >> "$P6/train.log" 2>&1'), "FD200"),
        ("echo the token", text.replace('log "env ok"', 'log "env ok token=$HF_TOKEN"'), "HF_TOKEN_USE"),
        ("xtrace on", text.replace("set -u\n", "set -ux\n", 1), "XTRACE"),
        ("change a training argument", text.replace("--lr 2e-4 --weight-decay", "--lr 1e-4 --weight-decay"), "ARGS_CHANGED"),
        ("change the label-worker argument", text.replace("--label-workers $LW", "--label-workers 12"), "ARGS_CHANGED"),
        ("hard-code the yaw value", text.replace("--yaw-wrap $YAW_WRAP", "--yaw-wrap off"), "YAW_LITERAL"),
        ("default the yaw value to on", text.replace('YAW_WRAP="${P6_YAW_WRAP-off}"', 'YAW_WRAP="${P6_YAW_WRAP-on}"'), "YAW_NOT_ONE_VARIABLE"),
        ("refill a blanked yaw value", text.replace('YAW_WRAP="${P6_YAW_WRAP-off}"', 'YAW_WRAP="${P6_YAW_WRAP:-off}"'), "YAW_NOT_ONE_VARIABLE"),
        ("the yaw check accepts anything", text.replace("    off) return 0 ;;", "    *) return 0 ;;"), "YAW_CHECK_NOT_STRICT"),
        ("the yaw check accepts on", text.replace("    off) return 0 ;;", "    on|off) return 0 ;;"), "YAW_CHECK_NOT_STRICT"),
        ("the yaw check is not called at the top", text.replace("blanked or different value stops the chain before hours of CPU are spent\nyaw_check\n", "blanked or different value stops the chain before hours of CPU are spent\n"),
         "YAW_CHECK_NOT_CALLED"),
        ("drop the token assertion", d(text, '[ -n "${HF_TOKEN:-}" ] || refuse'), "NO_TOKEN_ASSERT"),
        ("final failure swallowed", text.rstrip()[: -len("exit 1")] + "exit 0\n", "FINAL_FAILURE_NOT_PROPAGATED"),
        ("train.py without the token check", d(text, "train.py is NOT started"), "TRAIN_WITHOUT_TOKEN"),
        ("preflight without the token check", d(text, "the preflight and the trainer do not start"), "PREFLIGHT_WITHOUT_TOKEN"),
        ("token_ok always true", re.sub(r"(?s)(token_ok\(\) \{\n).*?(\n\}\n)", r"\1  return 0\2", text, count=1), "TOKEN_OK_NOT_VERIFYING"),
        ("token_ok greps the gate log", re.sub(r"(?s)(token_ok\(\) \{\n).*?(\n\}\n)", r'\1  grep -q ZZGATE-PASS "$P6/launch_gate.log"\2', text, count=1), "TOKEN_OK_NOT_VERIFYING"),
        ("token_ok verifies another argv", text.replace('--args "$ARGS" --pkg "$PKG" \\\n    --json', '--args "--lr 1" --pkg "$PKG" \\\n    --json'), "TOKEN_OK_NOT_VERIFYING"),
        ("token_ok forgets the profile", text.replace("verify --profile refe_p6 --token", "verify --token"), "TOKEN_OK_NOT_VERIFYING"),
        ("ARGS defined twice", text.replace(ARGS_RX.search(text).group(0) + "\n", ARGS_RX.search(text).group(0) + "\n" + ARGS_RX.search(text).group(0) + "\n", 1), "ARGS_DEFINED_TWICE"),
        ("the stage is never marked", d(text, "stage_mark launch_gate"), "LAUNCH_GATE_STAGE_NOT_MARKED"),
        ("the gate is not given the egress result", text.replace("--gate-egr-json", "--gate-egr-x"), "GATE_INPUT_MISSING"),
        ("G-EGR without the checkpoint size", text.replace(' --ckpt "$P6/mem_run/ckpt_last.pt"', ""), "EGR_WITHOUT_CKPT_SIZE"),
        ("the memory run is not derived from ARGS", text.replace('p6_prep.py mem-argv --args "$ARGS"', 'p6_prep.py mem-argv --args "$ARGS_X"'), "MEM_RUN_NOT_FROM_ARGS"),
        ("reinstate a re-fetch (DBs)", text.replace("have_dbs || fetch_dbs ||", "fetch_dbs ||"), "REFETCH_UNGUARDED:dbs"),
        ("reinstate a re-fetch (pixels)", text.replace("have_pix || fetch_pix ||", "fetch_pix ||"), "REFETCH_UNGUARDED:pix"),
        ("reinstate a re-fetch (bank)", text.replace("have_bank || fetch_bank ||", "fetch_bank ||"), "REFETCH_UNGUARDED:bank"),
        ("a download in the main body", text.replace('log "inputs ok', 'curl -sSL -o /dev/null http://x 200>&-\nlog "inputs ok', 1), "FETCH_OUTSIDE_FETCH_FUNCTION"),
        ("no P2 guard", text.replace("! ( flock -n 9 )", "! ( true )"), "P2_GUARD_MISSING"),
        ("read-only lock probe (EBADF on nfs4)", text.replace('9<> "$1/chain.lock"', '9< "$1/chain.lock"'), "P2_GUARD_MISSING"),
        ("RL-S1 and S1 not guarded", text.replace('for u in "$P2DIR" "$DATA/rls1" "$DATA/s1aux"; do', 'for u in "$P2DIR"; do'), "P2_GUARD_MISSING"),
        ("beside a replicate that is not training yet", d(text, 'grep -q "ZZP2REP_TRAIN_START"'), "OVERLAP_NOT_GATED_ON_TRAINING"),
        ("CUDA not hidden", d(text, '  export CUDA_VISIBLE_DEVICES=""'), "CUDA_NOT_HIDDEN_BESIDE_TRAINER"),
        ("CUDA never restored", d(text, 'export CUDA_VISIBLE_DEVICES="$CVD_SAVED"'), "CUDA_NOT_RESTORED_BEFORE_GPU_STAGE"),
        ("no wait before the first GPU stage", text.replace("\nwait_clear\nif [", "\n: no wait\nif [", 1), "WAIT_BEFORE_GPU_STAGE_MISSING"),
        ("the wait forgets the replicate's DONE marker", text.replace('! grep -q "ZZP2REP_DONE" "$P2REPD/chain.log"', "false"), "WAIT_CONDITIONS_INCOMPLETE"),
        ("the wait forgets the GPU", text.replace('if [ "$rc" = 0 ] && gpu_busy; then', 'if [ "$rc" = 0 ] && false; then'), "WAIT_CONDITIONS_INCOMPLETE"),
        ("the pools are not capped", text.replace('--workers "$CWX"', '--workers "$CW"'), "POOLS_NOT_CAPPED_BESIDE_TRAINER"),
        ("not niced beside the trainer", text.replace('nice -n "$NICEX"', "nice -n 10"), "NOT_NICED_BESIDE_TRAINER"),
        ("no step-time guard", text.replace("guard_watch $!", "wait $!"), "NO_STEP_TIME_GUARD"),
        ("the pause file is not passed to the builder", text.replace(' REFE_PAUSE_FILE="$PAUSE_FILE"', ""), "PAUSE_FILE_NOT_PASSED"),
        ("shipped over P2's tree", text.replace("refe-plan-p6", "refe-plan"), "P6_PKG_IS_P2_DIR"),
        ("workers from nproc", re.sub(r"^LW=.*$", 'LW="$(nproc)"', text, flags=re.M), "WORKERS_NOT_FROM_CGROUP"),
        ("the lab instrument ignores the chunk count", text.replace('--workers "$LW" --chunks "$LC" --n-sets "${P6_LAB_SETS:-256}" --out "$P6/lab.json"',
                                                                    '--workers "$LW" --n-sets "${P6_LAB_SETS:-256}" --out "$P6/lab.json"'), "LAB_NOT_AT_LAUNCH_CHUNKS"),
        ("a lab reading at another chunk count is reused", text.replace('key_done lab "w$LW-c$LC"', 'key_done lab "w$LW"'), "LAB_KEY_IGNORES_CHUNKS"),
        ("D7 never refuses", text.replace('refuse "D7: rank-$k twins are NOT', 'log "D7: rank-$k twins are NOT'), "D6_D7_NOT_ENFORCED"),
        ("D6 never refuses", text.replace('refuse "D6: no NAVSIM metadata', 'log "D6: no NAVSIM metadata'), "D6_D7_NOT_ENFORCED"),
        ("the twin build without the route env", text.replace('DRIVERL_EVAL_ROUTE_LANE_RANK="$k" ', ""), "ROUTE_ENV_NOT_SET_FOR_TWINS_ONLY"),
        ("the D7 fallback is automatic", text.replace('[ "${P6_D7_FALLBACK:-0}" = 1 ]', "true"), "D7_FALLBACK_NOT_EXPLICIT"),
        ("a stage that cannot be skipped", text.replace("if ! { stage_done cache &&", "if ! { false &&"), "STAGE_NOT_SKIPPABLE_OR_MARKED:cache"),
        ("a stage that is never marked", d(text, "stage_mark cache"), "STAGE_NOT_SKIPPABLE_OR_MARKED:cache"),
        ("an instrument that cannot be skipped", text.replace("key_done mem ", "false mem "), "STAGE_NOT_SKIPPABLE_OR_MARKED:mem"),
        ("no dry run after lanes", text.replace('stop_after lanes', ': lanes'), "NO_STOP_AFTER:lanes"),
        ("a re-measured memory run does not void the gate", text.replace("  invalidate egr launch_gate\n", ""), "INVALIDATION_MISSING"),
        ("a rebuilt cache does not void the instruments", text.replace("invalidate lanes lab mem egr launch_gate", "invalidate lanes"), "INVALIDATION_MISSING"),
        ("the trainer before the gate", text.replace("# ---- 8. G-LIVE-LAB's instrument", 'env -u HF_TOKEN "$PY" train.py $ARGS > /dev/null 2>&1 200>&-\n# ---- 8. G-LIVE-LAB\'s instrument', 1), "TRAINER_BEFORE_GATE"),
    ]
    return out


# ------------------------------------------------------------------------------------------------ the sandbox (layer 4)
FAKEPY = r'''
import json, lzma, os, pickle, runpy, sys
from pathlib import Path
import numpy as np
REAL = Path(os.environ["STUB_REAL_REFE"])
argv = sys.argv[1:]
script = os.path.basename(argv[0]) if argv else ""
rest = argv[1:]
with open(os.environ["STUB_CALLS"], "a", encoding="utf-8") as f:
    f.write(script + " " + " ".join(rest) + "\n")
with open(os.environ["STUB_CALLS"] + ".cvd", "a", encoding="utf-8") as f:
    f.write(script + " CVD=" + repr(os.environ.get("CUDA_VISIBLE_DEVICES")) + "\n")
def opt(name, default=None):
    return rest[rest.index(name) + 1] if name in rest else default
def rows(bank):
    for fp in sorted(Path(bank).glob("targets_rank*.jsonl")):
        for ln in open(fp, encoding="utf-8"):
            if ln.strip():
                yield json.loads(ln)
def put(path, centerline):
    path.parent.mkdir(parents=True, exist_ok=True)
    parts = (np.zeros((3, 3)), "ego", "obs", centerline, ["lane"], "dam")
    path.write_bytes(lzma.compress(pickle.dumps({"format": "tanitad.p6parts/1", "parts": parts}, protocol=4), preset=1))
if script in ("p6_prep.py", "annotate_commands.py", "p6_overlap.py"):
    sys.path.insert(0, str(REAL))
    sys.argv = [str(REAL / script)] + rest
    runpy.run_path(str(REAL / script), run_name="__main__")
elif script == "-c":
    sys.exit(1 if os.environ.get("STUB_PATCH_MISSING") else 0)
elif script == "p2r_sync.py":
    sub = rest[0]
    if sub == "recover":
        print("ZZRECOVER_OK labels=no ckpt=no")
    elif sub == "restore":
        print("ZZRESTORE_OK ok")
    elif sub == "tick":
        print("ZZSYNC_TICK_OK nothing_changed")
    elif sub == "final":
        run = Path(opt("--run"))
        ok = (run / "summary.json").is_file() and json.loads((run / "summary.json").read_text()).get("done") is True
        print("ZZP2R_FINAL_OK stub" if ok else "ZZP2R_FINAL_FAIL summary_not_done")
        sys.exit(0 if ok else 1)
elif script == "build_metric_cache.py":
    if rest[0] == "verify":
        print("ZZD9_PASS")
    else:
        if os.environ.get("STUB_SLOW_TRAINER") and not os.path.exists(os.environ["STUB_CALLS"] + ".slowed"):    # the build 'slows' the replicate: slow step rows, then it waits for the chain's pause
            open(os.environ["STUB_CALLS"] + ".slowed", "w").write("1")
            mf = Path(os.environ["STUB_METRICS"]); el = max(json.loads(x)["elapsed_s"] for x in mf.read_text().splitlines() if '"traj_L1"' in x)
            st = max(json.loads(x)["step"] for x in mf.read_text().splitlines() if '"traj_L1"' in x)
            import time as _t
            with open(mf, "a") as fh:
                for i in range(8):
                    el += 72.0; st += 1
                    fh.write(json.dumps({"step": st, "traj_L1": 1.0, "elapsed_s": el}) + "\n")
            for _ in range(40):
                if os.path.exists(os.environ["REFE_PAUSE_FILE"]):
                    open(os.environ["STUB_CALLS"], "a").write("PAUSE_SEEN\n")
                    break
                _t.sleep(0.5)
            with open(mf, "a") as fh:
                for i in range(8):
                    el += 60.0; st += 1
                    fh.write(json.dumps({"step": st, "traj_L1": 1.0, "elapsed_s": el}) + "\n")
            for _ in range(40):                                    # the chain removes the pause when its pause is over
                if not os.path.exists(os.environ["REFE_PAUSE_FILE"]):
                    break
                _t.sleep(0.5)
        k = opt("--lane-rank"); out = Path(opt("--out")); n = 0; seen = set(); logs = set()
        for r in rows(opt("--targets")):
            aug = r.get("aug"); lane = isinstance(aug, dict) and aug.get("kind") == "lane_rank"
            if k is None and lane:
                continue
            if k is not None and not (lane and int(aug["value"]) == int(k)):
                continue
            key = (r["log_name"], r["token"], k)
            if key in seen:
                continue
            seen.add(key); logs.add(r["log_name"])
            if k is None:
                put(out / r["log_name"] / (r["token"] + ".pkl.xz"), np.array([[0.0, 0.0], [1.0, 0.0]]))
            else:
                same = os.environ.get("STUB_TWIN_SAME")
                put(out / r["log_name"] / (r["token"] + "__lane_rank_" + str(k) + ".pkl.xz"), np.array([[0.0, 0.0], [1.0, 0.0 if same else 1.0]]))
            n += 1
        print("ZZMETRIC_CACHE_DONE built=%d skipped_5s_guard=0 failed=0 logs=%d 1s" % (n, len(logs)))
elif script == "diag_p6_labeller_identity.py":
    out = Path(opt("--out"))
    thr = float(os.environ.get("STUB_LAB_THR", "9.9"))
    bad = bool(os.environ.get("STUB_LAB_IDENTITY_FAIL"))
    out.write_text(json.dumps({"identity_ok": not bad, "n_sets": 64, "identity_equal_sets": 63 if bad else 64, "stale_refused": True, "throughput_sets_s": thr,
                               "workers": int(opt("--workers")), "chunks": int(opt("--chunks", "1"))}), encoding="utf-8")
    print("ZZLAB_IDENTITY_FAIL" if bad else "ZZLAB_IDENTITY_PASS"); print("ZZLAB_THROUGHPUT " + str(thr))
elif script == "hf_egress.py":
    out = Path(opt("--out")); out.mkdir(parents=True, exist_ok=True)
    sz = Path(opt("--ckpt")).stat().st_size
    (out / "g_egr_result.json").write_text(json.dumps({"pass": True, "main": {"ok": True}, "regression": {"refused_by_server": True},
                                                       "ckpt": {"local_size": sz, "remote_size": sz, "push_ok": True}}), encoding="utf-8")
    print("ZZEGR_PASS")
elif script == "launch_gate.py":
    sub = rest[0]
    if sub == "run":
        od = Path(opt("--out-dir")); od.mkdir(parents=True, exist_ok=True)
        (od / "evidence_G-HYG.json").write_text("{}")
        (od / "PASS_stub.json").write_text(json.dumps({"args": opt("--args"), "profile": opt("--profile")}))
        print("ZZGATE-PASS-stubZZ")
    else:
        tok = json.loads(Path(opt("--token")).read_text())
        bad = os.path.exists(os.environ["STUB_FORCE_TOKEN_FAIL"]) or tok["args"] != opt("--args") or tok["profile"] != opt("--profile")
        print("ZZGATE-REFUSEDZZ" if bad else "ZZGATE-MATCHZZ")
        sys.exit(5 if bad else 0)
elif script == "train.py":
    out = Path(opt("--out") or ".")
    if "--preflight" in rest:
        print("PREFLIGHT_OK")
    elif "--steps" in rest:                                          # the memory run
        out.mkdir(parents=True, exist_ok=True)
        (out / "metrics.jsonl").write_text(json.dumps({"traj_L1": 1.0, "cuda_max_mem_gb": 31.0, "samples": 800, "batch": 4}) + "\n")
        (out / "ckpt_last.pt").write_bytes(b"x" * 2000)
        print("TRAIN_DONE")
    else:
        cnt = Path(os.environ["STUB_TRAIN_COUNT"]); n = int(cnt.read_text()) if cnt.exists() else 0; cnt.write_text(str(n + 1))
        if os.environ.get("STUB_TRAIN_CRASH_FIRST") and n == 0:
            sys.exit(1)
        out.mkdir(parents=True, exist_ok=True)
        (out / "model_final.pt").write_bytes(b"m"); (out / "summary.json").write_text(json.dumps({"done": True}))
        print("TRAIN_DONE")
else:
    print("fakepy: unknown stub " + script); sys.exit(2)
'''
POD_TRAIN = '#!/bin/bash\necho "pod_train STAGES=$STAGES" >> "$STUB_CALLS"\necho "ZZOK weights ZZ"\n'
POD_DATAPREP = r'''#!/bin/bash
echo "pod_dataprep STAGES=$STAGES" >> "$STUB_CALLS"
case "$STAGES" in
  dbs) mkdir -p "$DATA/navtrain_dbs"; for i in $(seq 1 "$P6_EXPECT_DBS"); do : > "$DATA/navtrain_dbs/l$i.db"; done; echo "ZZOK dbs $P6_EXPECT_DBS files ZZ" ;;
  pix) mkdir -p "$DATA/navtrain_pixels"; for i in $(seq 1 "$P6_EXPECT_PIX_SHARDS"); do : > "$DATA/navtrain_pixels/.done_$i"; done
       echo "ZZOK pix $P6_EXPECT_PIX_SHARDS/$P6_EXPECT_PIX_SHARDS shards ZZ" ;;
esac
'''
REQUIRED = ["refe/train.py", "refe/p2r_sync.py", "refe/hf_egress.py", "refe/p6_prep.py", "refe/p6_overlap.py", "refe/p6_cpu_probe.py", "refe/build_metric_cache.py", "refe/annotate_commands.py",
            "refe/diag_p6_labeller_identity.py", "refe/launch_gate.py", "refe/gate_token.py", "refe/gate_checks.py", "refe/gate_fixture.py", "refe/test_launch_gate.py",
            "refe/test_p2r_chain.py", "refe/test_p6_levers.py", "refe/test_p6_chain.py", "SHIP_MANIFEST.json", "code/pod_teacher_env.sh", "code/pod_train.sh",
            "code/pod_dataprep.sh", "splits/navtrain.yaml"]
LOGS3 = ("L0", "L1", "L2")
TOKS2 = ("t0", "t1")


def posix(p):
    return str(p).replace("\\", "/")


# ⛔ G-SUITE runs this file INSIDE the chain's own environment (p6_pod.sh -> launch_gate.py run -> check_suite passes os.environ on), so the sandboxed chain must not
# inherit the real chain's knobs.  MEASURED 2026-10-18 (the 3-C gate dry run, G-SUITE FAIL): the hand-kept pop list that used to stand in Sandbox.run missed P6_CBANK
# (added a5a4ee8); the leaked value named the REAL 3-C bank, and the sandboxed chain REFUSED in its commands stage -- the stage whose rebuild ends `rm -rf "$CBANK"` and
# which only a5a4ee8's guard stopped.  So the chain's whole namespace is dropped by PREFIX (a new knob cannot be missing from it), plus the variables the chain EXPORTS
# to its children (paths to real data and to the gate key).  Everything else is kept: PATH, the temp dirs, and the gate's own nesting guard (REFE_GATE_NESTED).
SANDBOX_DROP_PREFIXES = ("P6_", "STUB_", "NUPLAN_", "DRIVERL_", "TANITAD_GATE_")
SANDBOX_DROP_NAMES = ("REFE_BACKBONE_ROOT", "REFE_NUPLAN_DB_ROOT", "REFE_NAVTRAIN_YAML", "REFE_PAUSE_FILE", "REFE_PAUSE_POLL_S", "CUDA_VISIBLE_DEVICES",
                      "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_HUB_TOKEN")


def bash_exe() -> str:
    """The bash the sandboxed chain runs under.  Git for Windows' LAUNCHER `<Git>/bin/bash.exe` prepends its own /usr/bin to PATH, which shadows the sandbox's stubs
    (`nice`): MEASURED 2026-10-18, a dev-box runner that put <Git>/bin first on PATH failed 4c's `nice` assertion (`type -p nice` -> /usr/bin/nice) while the same blob
    passed from an interactive Git bash, whose PATH resolves `bash` to <Git>/usr/bin/bash.exe.  That MSYS bash honours PATH as given, so it is used whenever the
    launcher is what PATH found.  Anywhere else (the pod: Linux) this is plain shutil.which("bash")."""
    b = shutil.which("bash")
    if b and os.name == "nt":
        p = Path(b)
        real = p.parent.parent / "usr" / "bin" / "bash.exe"
        if p.parent.name.lower() == "bin" and p.parent.parent.name.lower() != "usr" and real.is_file():
            return str(real)
    return b


def sandbox_env(base) -> dict:
    """The environment the sandboxed chain starts from: `base` minus the chain's namespace (SANDBOX_DROP_PREFIXES / SANDBOX_DROP_NAMES)."""
    return {k: v for k, v in base.items() if not k.upper().startswith(SANDBOX_DROP_PREFIXES) and k.upper() not in SANDBOX_DROP_NAMES}


class Sandbox:
    """A fake /workspace in which the REAL p6_pod.sh runs under bash. Every tool it calls is a stub that records the call; p6_prep.py and annotate_commands.py are the REAL files."""

    def __init__(self, root, script_text=None, navlogs=True, dbs=True, pix=True):
        self.root = Path(root)
        self.ws = self.root / "ws"
        self.data = self.ws / "data"
        self.pkg = self.ws / "refe-plan-p6"
        self.stub = self.root / "stub"
        self.calls = self.root / "calls.log"
        self.calls.write_text("", encoding="utf-8")
        (self.pkg / "refe").mkdir(parents=True)
        (self.pkg / "code").mkdir()
        (self.pkg / "splits").mkdir()
        self.stub.mkdir()
        self.script = self.pkg / "refe" / "p6_pod.sh"
        self.script.write_text(script_text if script_text is not None else CHAIN.read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
        for f in REQUIRED:
            if not (self.pkg / f).exists():
                (self.pkg / f).write_text("", encoding="utf-8")
        for name, body in (("pod_train.sh", POD_TRAIN), ("pod_dataprep.sh", POD_DATAPREP)):
            (self.pkg / "code" / name).write_text(body, encoding="utf-8", newline="\n")
        (self.stub / "fakepy.py").write_text(FAKEPY, encoding="utf-8")
        py = posix(sys.executable)
        (self.stub / "fakepy").write_text(f'#!/bin/bash\nexec "{py}" "{posix(self.stub / "fakepy.py")}" "$@"\n', encoding="utf-8", newline="\n")
        # a path-aware flock: a lock is HELD when its directory name is in STUB_HELD (or in the file STUB_HELD_FILE names, re-read on every probe so a test can release it mid-run)
        (self.stub / "bash_env.sh").write_text(
            'flock() {\n  local fd="${@: -1}" tgt who held\n  tgt="$(readlink /dev/fd/$fd 2> /dev/null)"\n  who="$(basename "$(dirname "$tgt")")"\n'
            '  held="${STUB_HELD:-}"; [ -n "${STUB_HELD_FILE:-}" ] && [ -f "$STUB_HELD_FILE" ] && held="$held $(cat "$STUB_HELD_FILE")"\n'
            '  case " $held " in *" $who "*) return 1 ;; esac\n  return 0\n}\n', encoding="utf-8", newline="\n")
        (self.stub / "nice").write_text('#!/bin/bash\necho "NICE $2" >> "$STUB_CALLS.nice"\nshift 2\nexec "$@"\n', encoding="utf-8", newline="\n")
        (self.stub / "nvidia-smi").write_text('#!/bin/bash\n[ -f "$STUB_GPU_BUSY_FILE" ] && echo 4242\nexit 0\n', encoding="utf-8", newline="\n")
        subprocess.run([bash_exe(), "-c", f"chmod +x '{posix(self.stub / 'nice')}' '{posix(self.stub / 'nvidia-smi')}'"], check=True)
        subprocess.run([bash_exe(), "-c", f"chmod +x '{posix(self.stub / 'fakepy')}'"], check=True)
        maps = self.data / "nuplan-maps" / "v"
        for c in ("a", "b", "c", "d"):
            (maps / c).mkdir(parents=True)
        (self.ws / "teacher_env.sh").write_text(f'export DRIVERL_EVAL_PYTHON="{posix(self.stub / "fakepy")}" NUPLAN_MAPS_ROOT="{posix(maps)}"\n', encoding="utf-8", newline="\n")
        bank = self.data / "refe_navtrain" / "train_grow"
        bank.mkdir(parents=True)
        (bank / "calib_table.json").write_text("{}", encoding="utf-8")
        r0 = [{"log_name": lg, "token": t, "step": 0, "rank": 0} for lg in LOGS3 for t in TOKS2]
        r1 = [{"log_name": "L0", "token": "t0", "step": 0, "rank": 1, "aug": {"kind": "goal_horizon", "value": 16}},
              {"log_name": "L0", "token": "t0", "step": 0, "rank": 1, "aug": {"kind": "lane_rank", "value": 1}},
              {"log_name": "L1", "token": "t1", "step": 0, "rank": 1, "aug": {"kind": "lane_rank", "value": 1}}]
        for nm, rs in (("targets_rank0.jsonl", r0), ("targets_rank1.jsonl", r1)):
            (bank / nm).write_text("".join(json.dumps(r) + "\n" for r in rs), encoding="utf-8", newline="\n")
        if dbs:
            (self.data / "navtrain_dbs").mkdir(parents=True)
            for i in range(3):
                (self.data / "navtrain_dbs" / f"l{i}.db").write_text("", encoding="utf-8")
        if pix:
            (self.data / "navtrain_pixels").mkdir(parents=True)
            for i in range(2):
                (self.data / "navtrain_pixels" / f".done_{i}").write_text("", encoding="utf-8")
        if navlogs:
            self.add_navlogs()

    def add_navlogs(self):
        d = self.data / "navsim_logs" / "trainval"
        d.mkdir(parents=True, exist_ok=True)
        for lg in LOGS3:
            with open(d / f"{lg}.pkl", "wb") as fh:
                pickle.dump([{"token": t, "log_name": lg, "driving_command": np.array([0, 1, 0, 0])} for t in TOKS2] + [{"token": "other", "driving_command": np.array([1, 0, 0, 0])}], fh)

    def run(self, extra_env=None, timeout=300):
        env = sandbox_env(os.environ)                       # the chain's namespace is NOT inherited (see SANDBOX_DROP_PREFIXES): G-SUITE runs this inside the real chain
        env.update({"WORK": posix(self.ws), "HF_TOKEN": TOKEN, "P6_DD_GB": "0", "P6_EXPECT_DBS": "3", "P6_EXPECT_PIX_SHARDS": "2", "P6_ATTEMPT_SLEEP": "0",
                    "STUB_CALLS": posix(self.calls), "STUB_REAL_REFE": posix(HERE), "STUB_TRAIN_COUNT": posix(self.root / "train_count"),
                    "STUB_FORCE_TOKEN_FAIL": posix(self.root / "force_token_fail"), "BASH_ENV": posix(self.stub / "bash_env.sh"), "P6_CACHE_WORKERS": "2",
                    "P6_NVSMI": posix(self.stub / "nvidia-smi"), "STUB_GPU_BUSY_FILE": posix(self.root / "gpu_busy"), "P6_POLL_S": "0", "P6_OVERLAP_DD_PAUSE_S": "0",
                    "P6_OVERLAP_BASELINE_TRIES": "1", "P6_OVERLAP_BASELINE_SLEEP_S": "0", "P6_OVERLAP_WORKERS": "2", "P6_GUARD_TICK_S": "1", "P6_GUARD_POLL_S": "2",
                    "P6_GUARD_PAUSE_S": "2", "STUB_METRICS": posix(self.data / "refe_runs" / "p2_joint_pdm_s1" / "metrics.jsonl"),
                    "P6_LABEL_WORKERS": "2", "PYTHONIOENCODING": "utf-8"})
        env["PATH"] = str(self.stub) + os.pathsep + env.get("PATH", "")
        env.update(extra_env or {})
        r = subprocess.run([bash_exe(), posix(self.script)], capture_output=True, text=True, env=env, timeout=timeout, encoding="utf-8", errors="replace")
        return r.returncode, r.stdout + r.stderr

    def calls_list(self):
        return [ln for ln in self.calls.read_text(encoding="utf-8").splitlines() if ln.strip()]

    def chain_log(self):
        p = self.data / "refe_p6" / "chain.log"
        return p.read_text(encoding="utf-8") if p.exists() else ""

    def stage(self, name):
        return (self.data / "refe_p6" / "stage" / f"{name}.ok").exists()

    def files_containing(self, needle: bytes):
        hits = []
        for p in self.root.rglob("*"):
            if p.is_file() and p.stat().st_size < 50_000_000:
                try:
                    if needle in p.read_bytes():
                        hits.append(str(p))
                except OSError:
                    pass
        return hits


def n_calls(calls, prefix, contains=None):
    return sum(1 for c in calls if c.startswith(prefix) and (contains is None or contains in c))


# ------------------------------------------------------------------------------------------------ layer 3 helpers
def make_parts_tree(root, logs_tokens, twin_k=None, twin_same=False):
    for lg, tk in logs_tokens:
        for suffix, cl in (("", np.array([[0.0, 0.0], [1.0, 0.0]])), (f"__lane_rank_{twin_k}" if twin_k else None, np.array([[0.0, 0.0], [1.0, 0.0 if twin_same else 1.0]]))):
            if suffix is None:
                continue
            p = Path(root) / lg / f"{tk}{suffix}.pkl.xz"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(lzma.compress(pickle.dumps({"format": "tanitad.p6parts/1", "parts": (np.zeros(3), "e", "o", cl, ["lane"], "d")}, protocol=4), preset=1))


def write_bank(d, r0, r1=()):
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    (d / "targets_rank0.jsonl").write_text("".join(json.dumps(r) + "\n" for r in r0), encoding="utf-8", newline="\n")
    if r1:
        (d / "targets_rank1.jsonl").write_text("".join(json.dumps(r) + "\n" for r in r1), encoding="utf-8", newline="\n")


def main() -> int:
    n = 0
    text = CHAIN.read_text(encoding="utf-8")
    for f in ("p6_pod.sh", "p6_prep.py", "test_p6_chain.py"):
        assert b"\r" not in (HERE / f).read_bytes(), f + " must stay LF"
        n += 1
    # ---- 1. the real chain satisfies every property; the pinned arguments are the ones in the script
    assert check_chain(text) == [], check_chain(text); n += 1
    assert ARGS_RX.search(text).group(1) == ARGS_REF; n += 1
    # ---- 2. every property goes RED when broken
    for name, mut, want in mutations(text):
        assert mut != text, f"mutation '{name}' changed nothing"
        got = check_chain(mut)
        assert any(g.startswith(want) for g in got), f"mutation '{name}' did not raise {want}: {got}"
        n += 1
    blk = re.search(r"# ---- 7\. D7.*?\nlog \"lanes ok\"; stop_after lanes\n", text, re.S)
    assert blk, "the D7 block moved"
    assert any(g.startswith("MISSING") or g.startswith("STAGE_NOT") for g in check_chain(text.replace(blk.group(0), ""))); n += 1       # D7 cannot be deleted silently
    late = text.replace(blk.group(0), "") + "\n" + blk.group(0)
    assert any(g.startswith("ORDER") for g in check_chain(late)); n += 1                                                              # nor reordered
    # ---- 3a. the expanded ARGS against the refe_p6 profile (the launch gate's own rule), both yaw values; and against the trainer's parser
    import launch_gate as L
    import train as TR
    env = {"CBANK": "/w/cmd", "PIX": "/w/pix", "BANK": "/w/bank", "MC": "/w/mc", "LW": "16", "LC": "2", "RUN": "/w/run"}
    prof = L.PROFILES["refe_p6"]
    assert prof["required_flags"]["--yaw-wrap"] == "off"; n += 1                                                                       # PI ruling P6-Y (2026-10-11): the profile asserts the VALUE
    for yaw in ("on", "off"):
        argv = Template(ARGS_REF).substitute(env, YAW_WRAP=yaw).split()
        rs = L.profile_refusals(prof, argv)
        if yaw == "off":
            assert rs == [], rs                                                                                                  # off passes
        else:
            assert rs and any("requires --yaw-wrap off" in r and "has --yaw-wrap on" in r for r in rs), rs                        # on is refused: a value mismatch, not an absence
        n += 1
        TR.build_parser().parse_args(argv); n += 1                                                                               # every flag is one the trainer builds
        assert argv[argv.index("--yaw-wrap") + 1] == yaw; n += 1
    for yaw, want_rc, want_marker in (("on", 5, "ZZGATE-REFUSEDZZ"), ("off", 0, "ZZGATE-CLEANZZ")):                                  # and through the CLI the chain's gate uses
        r = subprocess.run([sys.executable, str(HERE / "launch_gate.py"), "refusals", "--profile", "refe_p6", "--args", Template(ARGS_REF).substitute(env, YAW_WRAP=yaw)],
                           capture_output=True, text=True)
        assert r.returncode == want_rc and want_marker in r.stdout, r.stdout; n += 1
    argv = Template(ARGS_REF).substitute(env, YAW_WRAP="off").split()
    bad = [t for t in argv if t not in ("--yaw-wrap", "off")]
    assert any("--yaw-wrap" in r for r in L.profile_refusals(prof, bad)); n += 1                                                      # no yaw flag -> the profile refuses
    assert any("FORBIDS" in r for r in L.profile_refusals(prof, argv + ["--label-backend", "synthetic"])); n += 1
    assert any("--label-workers" in " ".join(x) for x in [["--label-workers"]]) and "--label-workers" in argv; n += 1
    assert {"suite": 1} and any(e.get("script") == "test_p6_chain.py" for e in prof["suite"]); n += 1                                   # the gate's G-SUITE runs this test on the pod
    # ---- 3b. p6_prep
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        bank = tmp / "bank"
        r0 = [{"log_name": lg, "token": t, "step": s, "rank": 0} for lg in LOGS3 for t in TOKS2 for s in (0, 1)]
        r1 = [{"log_name": "L0", "token": "t0", "step": 0, "rank": 1, "aug": {"kind": "goal_horizon", "value": 16}},
              {"log_name": "L0", "token": "t0", "step": 0, "rank": 1, "aug": {"kind": "lane_rank", "value": 2}},
              {"log_name": "L1", "token": "t1", "step": 0, "rank": 1, "aug": {"kind": "lane_rank", "value": 1}}]
        write_bank(bank, r0, r1)
        assert PP.lane_ranks(bank) == [1, 2]; n += 1
        assert PP.rank0_keys(bank) == {(lg, t) for lg in LOGS3 for t in TOKS2}; n += 1                                                # twins are not annotated rows
        nav = tmp / "nav"
        nav.mkdir()
        for lg in LOGS3:
            with open(nav / f"{lg}.pkl", "wb") as fh:
                pickle.dump([{"token": t, "driving_command": np.array([0, 0, 1, 0])} for t in TOKS2] + [{"token": "zz", "driving_command": np.array([1, 0, 0, 0])}], fh)
        table, st = PP.build_command_table(nav, bank)
        assert st["ok"] and st["matched"] == 6 and st["missing"] == 0 and table["L0/t0"] == [0, 0, 1, 0] and "L0/zz" not in table; n += 4
        (nav / "L2.pkl").unlink()
        table, st = PP.build_command_table(nav, bank)
        assert not st["ok"] and st["missing"] == 2 and st["logs_missing"] == 1; n += 1                                                  # a table that misses 2/6 of the bank FAILS
        table, st = PP.build_command_table(nav, bank, max_missing_frac=0.5)
        assert st["ok"]; n += 1
        real_logs = Path("D:/Archive/devbox-C/navsim/data/openscene/navsim_logs/test")
        if real_logs.is_dir():                                                                   # MEASURED on real NAVSIM metadata (navtest pickles), dev box only
            pk = sorted(real_logs.glob("*.pkl"))[0]
            frames = pickle.load(open(pk, "rb"))
            rb = tmp / "realbank"
            write_bank(rb, [{"log_name": pk.stem, "token": fr["token"], "step": 0, "rank": 0} for fr in frames[:5]])
            table, st = PP.build_command_table(real_logs, rb)
            assert st["ok"] and st["matched"] == 5 and all(len(v) == 4 and sum(v) == 1 for v in table.values()), st; n += 1
            print("ZZINFO real NAVSIM metadata pickle read:", pk.name, st["matched"], "frames")
        else:
            print("ZZINFO real NAVSIM metadata absent on this box; the synthetic pickles above cover the format")
        # D7: the twin's route must differ from the logged one
        mc = tmp / "mc"
        make_parts_tree(mc, [("L0", "t0"), ("L1", "t1")], twin_k=1)
        d = PP.lane_distinct(bank, mc, 1)
        assert d["ok"] and d["sampled"] == 1 and d["differ"] == 1, d; n += 2                                                          # (L1,t1) is the rank-1 twin in the bank; L0/t0 is rank 2
        mc2 = tmp / "mc2"
        make_parts_tree(mc2, [("L1", "t1")], twin_k=1, twin_same=True)
        d2 = PP.lane_distinct(bank, mc2, 1)
        assert not d2["ok"] and d2["differ"] == 0 and d2["sampled"] == 1; n += 1                                                      # the patch not honoured: twin == logged -> FAIL
        mc3 = tmp / "mc3"
        make_parts_tree(mc3, [("L1", "t1")])
        assert PP.lane_distinct(bank, mc3, 1)["sampled"] == 0 and not PP.lane_distinct(bank, mc3, 1)["ok"]; n += 1                    # twins that were never built are not a pass
        a = pickle.load(open(mc / "L1" / "t1.pkl.xz", "rb")) if False else None
        p_l, p_t = mc / "L1" / "t1.pkl.xz", mc / "L1" / "t1__lane_rank_1.pkl.xz"
        import online_labeller as OL
        assert PP.route_digest(OL.load_parts(p_l)) == PP.route_digest(OL.load_parts(p_l)); n += 1                                      # control: the same file twice is equal
        assert PP.route_digest(OL.load_parts(p_l)) != PP.route_digest(OL.load_parts(p_t)); n += 1
        assert PP.quarantine(mc, 1, tmp / "q") == 2 and not list(mc.glob("*/*__lane_rank_1.pkl.xz")) and (tmp / "q" / "L1" / "t1__lane_rank_1.pkl.xz").is_file(); n += 2
        assert PP.canon({"a": np.arange(3), "b": {1, 2}}) == PP.canon({"b": {2, 1}, "a": np.arange(3)}); n += 1                          # canonical form: order- and address-free
        # sample-tokens never returns a twin
        toks = PP.sample_tokens(mc2, 8)
        assert toks and all("__" not in t for t in toks), toks; n += 1
        make_parts_tree(tmp / "mc4", [(f"G{i}", f"x{j}") for i in range(4) for j in range(5)], twin_k=2)
        st4 = PP.sample_tokens(tmp / "mc4", 5)
        assert len(st4) == 5 and len({t.split(":")[0] for t in st4}) > 1 and all("__" not in t for t in st4); n += 1
        # mem-argv
        argv = Template(ARGS_REF).substitute(env, YAW_WRAP="off")
        m = PP.mem_argv(argv, "/w/mem")
        assert "--epochs" not in m and "--resume" not in m and m[m.index("--steps") + 1] == "4" and m[m.index("--accum") + 1] == "64", m; n += 3     # the LAUNCH accum (R42)
        assert m[m.index("--out") + 1] == "/w/mem" and m[m.index("--batch") + 1] == "4" and m[m.index("--yaw-wrap") + 1] == "off" and "--compile" in m and "--tf32" in m; n += 2
        assert m[m.index("--ckpt-every-min") + 1] == "999" and m[m.index("--label-workers") + 1] == "16"; n += 1
        # egr-ok
        good = {"pass": True, "ckpt": {"push_ok": True, "local_size": 228234301, "remote_size": 228234301}}
        for k, (payload, want) in enumerate([(good, True), ({**good, "pass": False}, False), ({"pass": True}, False),
                                              ({**good, "ckpt": {**good["ckpt"], "remote_size": 228234302}}, False),
                                              ({**good, "ckpt": {**good["ckpt"], "push_ok": False}}, False)]):
            jp = tmp / f"egr{k}.json"
            jp.write_text(json.dumps(payload), encoding="utf-8")
            assert PP.egr_check(jp)[0] is want, (k, PP.egr_check(jp)); n += 1
        # CLI markers
        for argvs, marker in ((["lane-ranks", "--bank", str(bank)], "ZZLANE_RANKS 1 2"), (["mem-argv", "--args", argv, "--out", "/w/mem"], "ZZMEM_ARGV --backbone vitl16")):
            r = subprocess.run([sys.executable, str(HERE / "p6_prep.py")] + argvs, capture_output=True, text=True)
            assert r.returncode == 0 and r.stdout.startswith(marker), r.stdout[:100]; n += 1
        r = subprocess.run([sys.executable, str(HERE / "p6_prep.py"), "command-table", "--logs", str(nav), "--bank", str(bank), "--out", str(tmp / "t.json")], capture_output=True, text=True)
        assert r.returncode == 1 and r.stdout.startswith("ZZCMD_TABLE_BAD"), r.stdout; n += 1                                          # a failing table prints the BAD marker, not the OK one
        # ---- 3c. carve_p6_gate_bank: only rows whose parts exist; the twin keyed by its OWN parts file
        import gate_fixture as F
        cb = tmp / "cbank"
        rows0 = [{"log_name": "A", "token": f"t{i}", "step": 0, "rank": 0} for i in range(6)]
        rows1 = [{"log_name": "A", "token": "t0", "step": 0, "rank": 1, "aug": {"kind": "lane_rank", "value": 1}},
                 {"log_name": "A", "token": "t1", "step": 0, "rank": 1, "aug": {"kind": "lane_rank", "value": 1}},
                 {"log_name": "A", "token": "t2", "step": 0, "rank": 1, "aug": {"kind": "goal_horizon", "value": 8}}]
        write_bank(cb, rows0, rows1)
        pm = tmp / "pm"
        make_parts_tree(pm, [("A", "t1"), ("A", "t2"), ("A", "t4"), ("A", "t5")], twin_k=1)                                         # t0, t3 have no parts; the twin (t1) and the horizon twin (t2) do
        out = F.carve_p6_gate_bank(str(cb), str(pm), str(tmp / "smoke"), 3)
        got0 = [json.loads(x)["token"] for x in (tmp / "smoke" / "targets_rank0.jsonl").read_text().splitlines()]
        got1 = [(json.loads(x)["token"], json.loads(x)["aug"]["kind"]) for x in (tmp / "smoke" / "targets_rank1.jsonl").read_text().splitlines()]
        assert got0 == ["t1", "t2", "t4"] and out["rows"]["targets_rank0.jsonl"] == 3 and out["no_parts"]["targets_rank0.jsonl"] == 2; n += 3            # capped at n; t0 and t3 (no parts) skipped
        assert got1 == [("t1", "lane_rank"), ("t2", "goal_horizon")], got1; n += 1                                                       # twin t0 has no twin file -> skipped
        out2 = F.carve_p6_gate_bank(str(cb), str(pm), str(tmp / "smoke2"), 100)
        assert out2["rows"]["targets_rank0.jsonl"] == 4; n += 1
        try:
            F.carve_p6_gate_bank(str(cb), str(tmp / "empty_cache"), str(tmp / "smoke3"), 5)
            raise AssertionError("an empty cache must refuse")
        except RuntimeError:
            n += 1
        # ---- 3c'. REACHABILITY of the carve from the launch gate's CLI: `launch_gate.py run` with the online labeller must carve the P6 smoke bank (and a P2-style call must not)
        import types
        import gate_token as GT
        p6argv = Template(ARGS_REF).substitute(env | {"CBANK": str(cb), "MC": str(pm)}, YAW_WRAP="off")
        s_argv = L.derived_argvs(prof, p6argv.split(), tmp / "work", gate_bank=str(tmp / "smoke_x"))[1]
        assert s_argv[s_argv.index("--targets") + 1] == str(tmp / "smoke_x") and "--scorer-targets" not in s_argv and "--epochs" not in s_argv; n += 3     # smoke: carved bank, no scorer file
        assert s_argv[s_argv.index("--steps") + 1] == "2" and s_argv[s_argv.index("--metric-cache") + 1] == str(pm); n += 1                              # ... and the cache that covers it
        calls_seen = {}
        real = (L.run_checks, L.finalize, GT.load_key)
        L.run_checks = lambda *a, **k: calls_seen.setdefault("run_checks", k.get("gate_bank")) or {}
        L.finalize = lambda *a, **k: ("FAIL", Path("x"), {"reasons": [], "binding": {"commit": None, "tree_sha256": "0" * 12}})
        GT.load_key = lambda *a, **k: b"k"
        try:
            rc = L.main(["run", "--profile", "refe_p6", "--args", p6argv, "--pkg", str(tmp / "pkg_empty"), "--out-dir", str(tmp / "lg_out"), "--gate-bank", str(tmp / "smoke_cli"),
                         "--gate-bank-n", "2"])
        finally:
            L.run_checks, L.finalize, GT.load_key = real
        carved = [json.loads(x)["token"] for x in (tmp / "smoke_cli" / "targets_rank0.jsonl").read_text().splitlines()]
        assert rc == 1 and carved == ["t1", "t2"] and calls_seen["run_checks"] == str(tmp / "smoke_cli"), (rc, carved, calls_seen); n += 2                  # --gate-bank-n honoured, parts-less t0 skipped
        # ---- 3d. G-EGR: the first-checkpoint size, from the same json the hf_egress gate wrote
        import gate_checks as C
        import hf_egress as E
        import test_hf_egress as TH
        TH._patch(E)
        ck = tmp / "ckpt_last.pt"
        ck.write_bytes(b"c" * 1234)
        api = TH.FakeApi(deny={TH.REG_REPO})
        rc = E.gate(api, "Sayood/tanitad-refe-p6-paper-faithful", "model", tmp / "g1", chain_log=tmp / "chain.log", ckpt=str(ck))
        res = json.loads((tmp / "g1" / "g_egr_result.json").read_text(encoding="utf-8"))
        assert rc == 0 and res["pass"] and res["ckpt"] == {**res["ckpt"], "local_size": 1234, "remote_size": 1234, "push_ok": True, "path": "gate/ckpt_last_probe.pt"}, res; n += 2
        rc = E.gate(TH.FakeApi(deny={TH.REG_REPO}), "Sayood/tanitad-refe-p6-paper-faithful", "model", tmp / "g2", chain_log=tmp / "chain.log")
        res2 = json.loads((tmp / "g2" / "g_egr_result.json").read_text(encoding="utf-8"))
        assert rc == 0 and "ckpt" not in res2; n += 1                                                                                    # P2-style call unchanged
        rc = E.gate(TH.FakeApi(deny={TH.REG_REPO, "Sayood/tanitad-refe-p6-paper-faithful"}), "Sayood/tanitad-refe-p6-paper-faithful", "model", tmp / "g3", ckpt=str(ck))
        assert rc == 1; n += 1                                                                                                           # a refused push fails the gate
        big = {"pass": True, "main": {"ok": True}, "regression": {"refused_by_server": True},
               "ckpt": {"path": "gate/ckpt_last_probe.pt", "local_size": 228234301, "remote_size": 228234301, "push_ok": True, "reason": "1 files"}}

        def egr_status(payload, arm=None, name="e.json", **kw):
            (tmp / name).write_text(json.dumps(payload), encoding="utf-8")
            return C.check_egr(str(tmp / name), arm=arm, **kw)
        assert egr_status(big)["status"] == "PASS"; n += 1
        assert egr_status(big, arm="egr_wrong_size")["status"] == "FAIL"; n += 1                                                         # the regression arm still bites on the new path
        assert egr_status({k: v for k, v in big.items() if k != "ckpt"})["status"] == "FAIL"; n += 1                                     # no size check made -> not a pass
        assert egr_status({k: v for k, v in big.items() if k != "ckpt"}, require_ckpt=False)["status"] == "PASS"; n += 1
        assert egr_status({**big, "ckpt": {**big["ckpt"], "remote_size": 228234300}})["status"] == "FAIL"; n += 1                         # listing size != local size
        assert egr_status({**big, "ckpt": {**big["ckpt"], "local_size": 1234, "remote_size": 1234}})["status"] == "FAIL"; n += 1          # a 1 KB 'checkpoint' proves nothing
        assert egr_status({**big, "ckpt": {**big["ckpt"], "push_ok": False}})["status"] == "FAIL"; n += 1
        assert egr_status(big, ckpt_sizes=(228234301, 228234301), name="e2.json")["status"] == "PASS"; n += 1                            # the explicit form still works
        assert not C.judge_egr(big, 228234301, 228234301) and C.judge_egr(big, 228234301, 228234302); n += 1
    # ---- 4. the REAL chain in a sandbox
    py_env = {"P6_NAVSIM_LOGS_URL": ""}
    with tempfile.TemporaryDirectory() as tmp:
        sb = Sandbox(tmp)
        calls = sb.calls_list
        # 4a. PI ruling P6-Y (2026-10-11): off is the default; a BLANKED value, `on`, or anything else REFUSES before any stage runs
        for bad_yaw in ("", "on", "maybe"):
            rc, out = sb.run({**py_env, "P6_YAW_WRAP": bad_yaw})
            assert rc != 0 and "REFUSED: PI ruling P6-Y" in sb.chain_log(), (bad_yaw, rc, sb.chain_log()[-400:]); n += 2
            assert n_calls(calls(), "build_metric_cache.py") == 0 and n_calls(calls(), "train.py") == 0 and n_calls(calls(), "p2r_sync.py watch") == 0; n += 1
        # 4b. env -> lanes (yaw UNSET = the ruled default): P2's data REUSED (nothing fetched), commands annotated, caches built, D7 checked
        rc, out = sb.run({**py_env, "P6_STOP_AFTER": "lanes"})
        assert rc == 0, out[-600:]
        c = calls()
        assert n_calls(c, "pod_dataprep") == 0 and n_calls(c, "p2r_sync.py restore") == 0, c; n += 2                                      # P2's DBs / pixels / bank are reused, not re-fetched
        assert n_calls(c, "annotate_commands.py") == 1 and n_calls(c, "build_metric_cache.py build") == 2 and n_calls(c, "build_metric_cache.py verify") == 1; n += 3
        assert n_calls(c, "build_metric_cache.py build", "--lane-rank 1") == 1 and n_calls(c, "p2r_sync.py recover", "--repo Sayood/tanitad-refe-p6-paper-faithful") == 1; n += 2
        assert n_calls(c, "train.py") == 0 and all(sb.stage(s) for s in ("env", "weights", "recover", "commands", "cache", "lanes")) and not sb.stage("launch_gate"); n += 2
        cb = sb.data / "refe_navtrain" / "train_grow_cmd"
        rows = [json.loads(x) for fp in sorted(cb.glob("targets_rank*.jsonl")) for x in fp.read_text().splitlines()]
        assert all(r.get("command") == 1 for r in rows if r["rank"] == 0) and not any("command" in r for r in rows if r["rank"] == 1), rows; n += 2   # NAVSIM's STRAIGHT on rank 0 only
        assert (sb.data / "refe_navtrain" / "train_grow" / "targets_rank0.jsonl").read_text() != (cb / "targets_rank0.jsonl").read_text(); n += 1   # P2's bank untouched
        # 4c. the full run, the yaw value left at its ruled default
        base = len(calls())
        rc, out = sb.run(py_env)
        assert rc == 0 and "ZZP6_DONE" in sb.chain_log(), (rc, out[-500:], sb.chain_log()[-500:]); n += 2
        c = calls()
        new = c[base:]
        assert n_calls(c, "build_metric_cache.py build") == 2 and n_calls(c, "annotate_commands.py") == 1 and n_calls(c, "pod_dataprep") == 0; n += 3     # nothing rebuilt, nothing fetched
        gate_i = next(i for i, x in enumerate(c) if x.startswith("launch_gate.py run"))
        mem_i = next(i for i, x in enumerate(c) if x.startswith("train.py") and "--steps 4" in x)
        pre_i = next(i for i, x in enumerate(c) if x.startswith("train.py") and "--preflight" in x)
        full_i = next(i for i, x in enumerate(c) if x.startswith("train.py") and "--steps" not in x and "--preflight" not in x)
        egr_i = next(i for i, x in enumerate(c) if x.startswith("hf_egress.py gate"))
        lab_i = next(i for i, x in enumerate(c) if x.startswith("diag_p6_labeller_identity.py"))
        ver_i = [i for i, x in enumerate(c) if x.startswith("launch_gate.py verify")]
        assert lab_i < mem_i < egr_i < gate_i < pre_i < full_i; n += 1
        assert any(gate_i < i < pre_i for i in ver_i) and any(pre_i < i < full_i for i in ver_i), ver_i; n += 1                         # the token is verified before the preflight AND before the launch
        g, f, mm = c[gate_i], c[full_i], c[mem_i]
        assert "--profile refe_p6" in g and all(k in g for k in ("--gate-lab-json", "--gate-mem-metrics", "--gate-egr-json", "--gate-bank")); n += 2
        assert "--yaw-wrap off" in f and "--resume" in f and "--labeller online" in f and "--nav-mode command" in f and "--label-workers 2" in f, f; n += 1
        assert "--resume" not in mm and "--epochs" not in mm and "--accum 64" in mm and "refe_p6/mem_run" in mm; n += 1
        assert "--ckpt" in c[egr_i] and "--repo Sayood/tanitad-refe-p6-paper-faithful" in c[egr_i] and "--workers 2" in c[lab_i]; n += 1
        assert any(x.startswith("p2r_sync.py final") and "--repo Sayood/tanitad-refe-p6-paper-faithful" in x for x in c); n += 1
        assert (sb.data / "refe_p6" / "gate" / "PASS_stub.json").is_file(); n += 1
        # 4d. NO token in any file anywhere in the sandbox, after a full run
        assert sb.files_containing(TOKEN.encode()) == [], sb.files_containing(TOKEN.encode()); n += 1
        # 4e. everything done: a second start executes NO stage command (the preflight is the only trainer call)
        base = len(calls())
        rc, out = sb.run({**py_env, "P6_YAW_WRAP": "off"})
        new = calls()[base:]
        assert rc == 0 and "ZZP6_DONE" in sb.chain_log(), out[-400:]; n += 1
        forbidden = ("build_metric_cache.py", "annotate_commands.py", "diag_p6_labeller_identity.py", "hf_egress.py gate", "launch_gate.py run", "pod_dataprep", "pod_train",
                     "p2r_sync.py restore", "p6_prep.py command-table", "p6_prep.py mem-argv")
        assert not [x for x in new if x.startswith(forbidden)], [x for x in new if x.startswith(forbidden)]; n += 1
        assert any(x.startswith("launch_gate.py verify") for x in new) and any(x.startswith("p2r_sync.py final") for x in new); n += 1          # what it DOES run: the token check and the final push
        assert not [x for x in new if x.startswith("train.py") and "--preflight" not in x], new; n += 1
        # 4f. one instrument re-measured -> exactly what depends on it re-runs: the memory run voids G-EGR and the gate, not the labeller instrument or the cache
        (sb.data / "refe_p6" / "stage" / "mem.ok").unlink()
        base = len(calls())
        rc, out = sb.run({**py_env, "P6_YAW_WRAP": "off"})
        new = calls()[base:]
        assert rc == 0 and any(x.startswith("train.py") and "--steps 4" in x for x in new), out[-300:]; n += 2
        assert any(x.startswith("hf_egress.py gate") for x in new) and any(x.startswith("launch_gate.py run") for x in new); n += 2
        assert not any(x.startswith(("diag_p6_labeller_identity.py", "build_metric_cache.py", "annotate_commands.py")) for x in new); n += 1
        # 4g. the argv changed (another pod: 3 label workers, not 2) -> the labeller instrument (measured AT the launch worker count), the memory run, G-EGR and the gate depend on
        #     it and re-run; the token minted for the old argv is not accepted; nothing upstream (cache, commands) is rebuilt
        base = len(calls())
        rc, out = sb.run({**py_env, "P6_LABEL_WORKERS": "3"})
        new = calls()[base:]
        assert rc == 0 and any("--label-workers 3" in x for x in new if x.startswith("launch_gate.py run")), out[-300:]; n += 2
        assert any(x.startswith("train.py") and "--steps 4" in x for x in new) and any(x.startswith("diag_p6_labeller_identity.py") and "--workers 3" in x for x in new); n += 2
        assert not any(x.startswith(("build_metric_cache.py", "annotate_commands.py")) for x in new); n += 1
        # 4h. a token that does not verify: the trainer is NOT started (the run is made unfinished first)
        (sb.root / "force_token_fail").write_text("1")
        (sb.data / "refe_runs" / "p6_paper_faithful" / "summary.json").unlink()
        base = len(calls())
        rc, out = sb.run({**py_env, "P6_LABEL_WORKERS": "3"})
        new = calls()[base:]
        assert rc != 0 and "REFUSED:" in sb.chain_log().splitlines()[-1] + sb.chain_log()[-300:], sb.chain_log()[-300:]; n += 2
        assert not [x for x in new if x.startswith("train.py") and "--preflight" not in x and "--steps" not in x], new; n += 1
        # 4i. an interrupted run resumes: attempt 1 dies, attempt 2 relaunches with --resume and finishes
        (sb.root / "force_token_fail").unlink()
        (sb.root / "train_count").unlink(missing_ok=True)
        base = len(calls())
        rc, out = sb.run({**py_env, "P6_LABEL_WORKERS": "3", "STUB_TRAIN_CRASH_FIRST": "1"})
        new = [x for x in calls()[base:] if x.startswith("train.py") and "--preflight" not in x and "--steps" not in x]
        assert rc == 0 and len(new) == 2 and all("--resume" in x for x in new), (rc, new); n += 2
        assert "attempt 1 exited rc 1" in sb.chain_log() and "attempt 2 exited rc 0" in sb.chain_log(); n += 1
        assert sb.files_containing(TOKEN.encode()) == []; n += 1
    # ---- 4i'. HERMETIC under the gate (2026-10-18): G-SUITE runs this file INSIDE the real chain's environment.  The chain's namespace is dropped by prefix ...
    keep = {"PATH": "x", "REFE_GATE_NESTED": "[\"test_p6_chain.py\"]", "TEMP": "t", "HF_HOME": "h"}
    leak = {"P6_CBANK": "/real/bank", "P6_SOME_FUTURE_KNOB": "1", "STUB_HELD": "refe_p2rep", "TANITAD_GATE_KEY_FILE": "/real/key", "TANITAD_GATE_ARM": "x",
            "NUPLAN_DATA_ROOT": "/real", "DRIVERL_EVAL_ROUTE_LANE_RANK": "1", "REFE_NUPLAN_DB_ROOT": "/real/dbs", "REFE_PAUSE_FILE": "/real/p", "HF_TOKEN": "real",
            "CUDA_VISIBLE_DEVICES": "0", "p6_lower_case": "1"}
    se = sandbox_env({**keep, **leak})
    assert se == keep, se; n += 1
    # ... and BEHAVIOURALLY: a P6_CBANK leaked into os.environ (the 3-C gate dry run's exact state) neither redirects the sandboxed chain nor reaches the bank it names
    with tempfile.TemporaryDirectory() as tmp:
        canary = Path(tmp) / "real_refe_navtrain" / "train_grow_cmd_3c"
        canary.mkdir(parents=True)
        (canary / "targets_rank0.jsonl").write_text('{"canary": 1}\n', encoding="utf-8")
        saved_cb = os.environ.get("P6_CBANK")
        os.environ["P6_CBANK"] = posix(canary)
        try:
            (Path(tmp) / "sb").mkdir(); (Path(tmp) / "sb2").mkdir()
            sb = Sandbox(Path(tmp) / "sb")
            rc, out = sb.run({**py_env, "P6_STOP_AFTER": "lanes"})
            assert rc == 0 and "lanes ok" in sb.chain_log() and "REFUSED" not in sb.chain_log(), (rc, sb.chain_log()[-500:]); n += 2
            assert not [x for x in sb.calls_list() if "real_refe_navtrain" in x], [x for x in sb.calls_list() if "real_refe_navtrain" in x][:3]; n += 1
            # the contrast arm (this probe CAN see a leak): the same value handed to the chain, as the un-scrubbed environment did, is REFUSED by a5a4ee8's
            # commands-stage guard -- the only thing that stood between the leaked path and the stage's `rm -rf "$CBANK"`
            sb2 = Sandbox(Path(tmp) / "sb2")
            rc2, _ = sb2.run({**py_env, "P6_STOP_AFTER": "lanes", "P6_CBANK": posix(canary)})
            assert rc2 != 0 and "REFUSED: P6_CBANK is set" in sb2.chain_log(), (rc2, sb2.chain_log()[-400:]); n += 2
        finally:
            if saved_cb is None:
                os.environ.pop("P6_CBANK", None)
            else:
                os.environ["P6_CBANK"] = saved_cb
        assert sorted(p.name for p in canary.parent.iterdir()) == ["train_grow_cmd_3c"] and (canary / "targets_rank0.jsonl").read_text(encoding="utf-8") == '{"canary": 1}\n'; n += 2
    # ---- 4j. D7 refuses: the twins came out on the LOGGED route (the patch is not honoured) -> REFUSED; the fallback is explicit and quarantines them
    with tempfile.TemporaryDirectory() as tmp:
        sb = Sandbox(tmp)
        rc, out = sb.run({**py_env, "P6_STOP_AFTER": "lanes", "STUB_TWIN_SAME": "1"})
        assert rc != 0 and "REFUSED: D7: rank-1 twins are NOT on their own route" in sb.chain_log() and not sb.stage("lanes"), sb.chain_log()[-400:]; n += 2
        rc, out = sb.run({**py_env, "P6_STOP_AFTER": "lanes", "STUB_TWIN_SAME": "1", "P6_D7_FALLBACK": "1"})
        assert rc == 0 and "D7 FALLBACK" in sb.chain_log() and sb.stage("lanes"), sb.chain_log()[-400:]; n += 2
        assert not list((sb.data / "refe_navtrain" / "metric_cache").glob("*/*__lane_rank_1.pkl.xz")) and list((sb.data / "refe_navtrain" / "metric_cache_twins_rejected").glob("*/*")); n += 2
    with tempfile.TemporaryDirectory() as tmp:
        sb = Sandbox(tmp)
        rc, out = sb.run({**py_env, "P6_STOP_AFTER": "lanes", "STUB_PATCH_MISSING": "1"})
        assert rc != 0 and "REFUSED: D7: the DriveRL route patch is not importable" in sb.chain_log() and not sb.stage("lanes"); n += 2   # D7's inputs missing -> REFUSED, not skipped
    # ---- 4k. D6 refuses without the NAVSIM metadata (and with the download forbidden); with it, the stage passes
    with tempfile.TemporaryDirectory() as tmp:
        sb = Sandbox(tmp, navlogs=False)
        rc, out = sb.run({**py_env, "P6_STOP_AFTER": "commands"})
        assert rc != 0 and "REFUSED: D6: no NAVSIM metadata" in sb.chain_log() and not sb.stage("commands"); n += 2
        sb.add_navlogs()
        rc, out = sb.run({**py_env, "P6_STOP_AFTER": "commands"})
        assert rc == 0 and sb.stage("commands"); n += 2
    # ---- 4k2. P6_FREE_NAVLOGS (P6_ON_11VCPU section 5): off by default; on, the default metadata tree + the download dir go AFTER the command table exists; a custom P6_NAVSIM_LOGS is never touched
    with tempfile.TemporaryDirectory() as tmp:
        sb = Sandbox(tmp)
        (sb.data / "navsim_logs_dl").mkdir()
        (sb.data / "navsim_logs_dl" / "junk.bin").write_bytes(b"x")
        rc, out = sb.run({**py_env, "P6_STOP_AFTER": "commands"})
        assert rc == 0 and sb.stage("commands") and (sb.data / "navsim_logs" / "trainval" / "L0.pkl").exists() and (sb.data / "navsim_logs_dl" / "junk.bin").exists(); n += 3   # default: nothing deleted
    with tempfile.TemporaryDirectory() as tmp:
        sb = Sandbox(tmp)
        (sb.data / "navsim_logs_dl").mkdir()
        (sb.data / "navsim_logs_dl" / "junk.bin").write_bytes(b"x")
        rc, out = sb.run({**py_env, "P6_STOP_AFTER": "commands", "P6_FREE_NAVLOGS": "1"})
        assert rc == 0 and sb.stage("commands") and not (sb.data / "navsim_logs").exists() and not (sb.data / "navsim_logs_dl").exists(), sb.chain_log()[-400:]; n += 3
        assert "NAVSIM metadata deleted" in sb.chain_log() and (sb.data / "refe_p6" / "commands.json").exists() and list((sb.data / "refe_navtrain" / "train_grow_cmd").glob("targets_rank*.jsonl")); n += 3
        rc, out = sb.run({**py_env, "P6_STOP_AFTER": "lanes", "P6_FREE_NAVLOGS": "1"})                     # the rest of the chain does not need the metadata again
        assert rc == 0 and sb.stage("lanes") and not (sb.data / "navsim_logs").exists(); n += 2
    with tempfile.TemporaryDirectory() as tmp:
        sb = Sandbox(tmp, navlogs=False)
        custom = Path(tmp) / "shared_meta"
        custom.mkdir()
        for lg in LOGS3:
            with open(custom / f"{lg}.pkl", "wb") as fh:
                pickle.dump([{"token": t, "log_name": lg, "driving_command": np.array([0, 1, 0, 0])} for t in TOKS2], fh)
        rc, out = sb.run({**py_env, "P6_STOP_AFTER": "commands", "P6_FREE_NAVLOGS": "1", "P6_NAVSIM_LOGS": posix(custom)})
        assert rc == 0 and sb.stage("commands") and (custom / "L0.pkl").exists(); n += 2                      # somebody else's directory is never deleted
    with tempfile.TemporaryDirectory() as tmp:
        sb = Sandbox(tmp, navlogs=False)
        rc, out = sb.run({**py_env, "P6_STOP_AFTER": "commands", "P6_FREE_NAVLOGS": "1"})                 # a refused commands stage deletes nothing and marks nothing
        assert rc != 0 and not sb.stage("commands"); n += 2
    # ---- 4n. OVERLAP (eval/P6_ON_11VCPU.md section 9): P6's CPU-only stages beside the P2 seed replicate's trainer, and the wait before the first GPU stage
    def overlap_fixture(sb, train_start=True, done=False, metrics=True, steps=12, dt=60.0):
        d = sb.data / "refe_p2rep"
        d.mkdir(parents=True, exist_ok=True)
        (d / "chain.lock").write_text("")
        lines = ["2026-10-14T05:00:00 ZZP2REP_START chain"] + (["2026-10-14T05:20:00 ZZP2REP_TRAIN_START (train.py preflight OK; seed 1; launching)"] if train_start else []) \
            + (["2026-10-14T23:00:00 ZZP2REP_DONE"] if done else [])
        (d / "chain.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
        mp = sb.data / "refe_runs" / "p2_joint_pdm_s1"
        mp.mkdir(parents=True, exist_ok=True)
        if metrics:
            el = 1000.0
            rows = [json.dumps({"event": "start"})]
            for i in range(1, steps + 1):
                el += dt
                rows.append(json.dumps({"step": i, "traj_L1": 1.0, "elapsed_s": el}))
            (mp / "metrics.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")

    def calls_file(sb, suffix):
        fp = Path(str(sb.calls) + suffix)
        return fp.read_text(encoding="utf-8").splitlines() if fp.exists() else []

    # 4n-a. P2, RL-S1 and S1 must be OVER (each refused on its own lock); the replicate may run only once it TRAINS, and P6_OVERLAP=0 restores the refusal
    for held_dir, rel in (("refe_p2", "refe_p2"), ("rls1", "rls1"), ("s1aux", "s1aux")):
        with tempfile.TemporaryDirectory() as tmp:
            sb = Sandbox(tmp)
            (sb.data / rel).mkdir(parents=True, exist_ok=True)
            (sb.data / rel / "chain.lock").write_text("")
            rc, out = sb.run({**py_env, "STUB_HELD": held_dir, "P6_STOP_AFTER": "env"})
            assert rc != 0 and f"REFUSED: the chain at {sb.data / rel}".replace("\\", "/") in sb.chain_log().replace("\\", "/") and not sb.stage("env"), (held_dir, sb.chain_log()[-300:]); n += 2
            assert sb.calls_list() == [], sb.calls_list(); n += 1
    with tempfile.TemporaryDirectory() as tmp:
        sb = Sandbox(tmp)
        overlap_fixture(sb, train_start=False)
        rc, out = sb.run({**py_env, "STUB_HELD": "refe_p2rep", "P6_STOP_AFTER": "env"})
        assert rc != 0 and "is not TRAINING yet" in sb.chain_log() and not sb.stage("env"); n += 2                  # its own gates measure this pod: P6 starts beside it only once it trains
        overlap_fixture(sb)
        rc, out = sb.run({**py_env, "STUB_HELD": "refe_p2rep", "P6_OVERLAP": "0", "P6_STOP_AFTER": "env"})
        assert rc != 0 and "P6_OVERLAP=0" in sb.chain_log() and not sb.stage("env"); n += 2
    # 4n-b. beside a training replicate: the CPU-only stages up to the lab, with CUDA hidden, nice 19, pools capped, a trainer baseline; and the contrast run without it
    with tempfile.TemporaryDirectory() as tmp:
        sb = Sandbox(tmp)
        overlap_fixture(sb)
        rc, out = sb.run({**py_env, "STUB_HELD": "refe_p2rep", "P6_CACHE_WORKERS": "5", "P6_STOP_AFTER": "lab"})
        log_ = sb.chain_log()
        assert rc == 0 and sb.stage("lanes") and "ZZP6_OVERLAP on" in log_ and "ZZP6_OVERLAP_BASELINE trainer step 60.0" in log_, (rc, log_[-500:]); n += 3
        c = sb.calls_list()
        assert n_calls(c, "build_metric_cache.py build") == 3 and n_calls(c, "build_metric_cache.py build", "--workers 2") == 3 and n_calls(c, "build_metric_cache.py build", "--workers 5") == 0; n += 3   # 5 capped to 2
        assert n_calls(c, "build_metric_cache.py build", "--limit 3") == 1 and "ZZP6_EARLY_LAB ZZLAB_BAR ok thr=9.9" in log_ and (sb.data / "refe_p6" / "lab_early.json").exists() and sb.stage("earlylab"); n += 4
        assert n_calls(c, "diag_p6_labeller_identity.py") == 2 and n_calls(c, "diag_p6_labeller_identity.py", "lab_early.json") == 1 and n_calls(c, "diag_p6_labeller_identity.py", "lab.json") == 1; n += 3

        nice = calls_file(sb, ".nice")
        assert len(nice) >= 5 and set(nice) == {"NICE 19"}, nice; n += 2                                                    # command table, annotate, early slice, both builds: nice 19, never 10
        cvd = calls_file(sb, ".cvd")
        assert cvd and all(x.endswith("CVD=''") for x in cvd if x.startswith(("build_metric_cache.py", "annotate_commands.py", "p6_prep.py", "p6_overlap.py", "diag_p6"))), cvd; n += 1
        assert any(x.startswith("diag_p6_labeller_identity.py") and x.endswith("CVD=''") for x in cvd) and n_calls(c, "diag_p6_labeller_identity.py") == 2; n += 2
        assert (sb.data / "refe_p6" / "lab_beside_trainer").exists() and "BESIDE the P2 seed replicate" in log_; n += 2
        assert not (sb.data / "refe_p6" / "overlap.pause").exists(); n += 1
    with tempfile.TemporaryDirectory() as tmp:                                                                           # the contrast: nothing beside it -> unconstrained, nice 10, CUDA untouched
        sb = Sandbox(tmp)
        rc, out = sb.run({**py_env, "P6_CACHE_WORKERS": "5", "P6_STOP_AFTER": "lab"})
        c = sb.calls_list()
        assert rc == 0 and n_calls(c, "build_metric_cache.py build", "--workers 5") == 2 and "ZZP6_OVERLAP" not in sb.chain_log(); n += 3
        assert set(calls_file(sb, ".nice")) == {"NICE 10"} and all(x.endswith("CVD=None") for x in calls_file(sb, ".cvd")); n += 2
        assert not (sb.data / "refe_p6" / "lab_beside_trainer").exists() and n_calls(c, "diag_p6_labeller_identity.py") == 1 and not sb.stage("earlylab") and "ZZP6_EARLY_LAB" not in sb.chain_log(); n += 3
    with tempfile.TemporaryDirectory() as tmp:                                                                           # P6_EARLY_LAB=0 skips the early reading, and an identity failure in it REFUSES
        sb = Sandbox(tmp)
        overlap_fixture(sb)
        rc, out = sb.run({**py_env, "STUB_HELD": "refe_p2rep", "P6_EARLY_LAB": "0", "P6_STOP_AFTER": "earlylab"})
        assert rc == 0 and not sb.stage("earlylab") and n_calls(sb.calls_list(), "build_metric_cache.py build") == 0; n += 2
        rc, out = sb.run({**py_env, "STUB_HELD": "refe_p2rep", "STUB_LAB_IDENTITY_FAIL": "1", "P6_STOP_AFTER": "earlylab"})
        assert rc != 0 and "REFUSED: EARLY G-LIVE-LAB: identity" in sb.chain_log() and not sb.stage("earlylab"); n += 2
    # 4n-c. no readable trainer baseline -> REFUSED (the guard could not protect the replicate); the unguarded run is the Master Mind's word
    with tempfile.TemporaryDirectory() as tmp:
        sb = Sandbox(tmp)
        overlap_fixture(sb, metrics=False)
        rc, out = sb.run({**py_env, "STUB_HELD": "refe_p2rep", "P6_STOP_AFTER": "env"})
        assert rc != 0 and "no trainer baseline readable" in sb.chain_log() and not sb.stage("env"); n += 2
        rc, out = sb.run({**py_env, "STUB_HELD": "refe_p2rep", "P6_STOP_AFTER": "env", "P6_OVERLAP_UNGUARDED": "1"})
        assert rc == 0 and "ZZP6_OVERLAP_UNGUARDED" in sb.chain_log() and sb.stage("env"); n += 2
    # 4n-d. the step-time guard: the build slows the trainer by 20 % -> two slow reads -> the stage is PAUSED (the builder sees the pause file) -> it resumes; a quiet trainer is never paused
    with tempfile.TemporaryDirectory() as tmp:
        sb = Sandbox(tmp)
        overlap_fixture(sb)
        rc, out = sb.run({**py_env, "STUB_HELD": "refe_p2rep", "STUB_SLOW_TRAINER": "1", "P6_STOP_AFTER": "cache"}, timeout=120)
        log_ = sb.chain_log()
        assert rc == 0 and sb.stage("cache"), (rc, log_[-600:]); n += 2
        assert "PAUSE_SEEN" in sb.calls_list() and log_.count("ZZP6_OVERLAP_SLOW") >= 2 and "ZZP6_OVERLAP_BACKOFF" in log_, log_[-800:]; n += 3
        assert not (sb.data / "refe_p6" / "overlap.pause").exists(); n += 1
    with tempfile.TemporaryDirectory() as tmp:
        sb = Sandbox(tmp)
        overlap_fixture(sb)
        rc, out = sb.run({**py_env, "STUB_HELD": "refe_p2rep", "P6_STOP_AFTER": "cache"})
        assert rc == 0 and "ZZP6_OVERLAP_BACKOFF" not in sb.chain_log() and "PAUSE_SEEN" not in sb.calls_list(); n += 2
    # 4n-e. the WAIT before the first GPU stage, end to end: the replicate holds its lock and the GPU; P6 does its CPU stages, then WAITS; the lock is released but the GPU is busy -> still waiting;
    #       the GPU goes idle -> the GPU stages start, with CUDA visible again.  mem must not have started before.
    with tempfile.TemporaryDirectory() as tmp:
        import threading
        import time as _time
        sb = Sandbox(tmp)
        overlap_fixture(sb)
        held, gpu = sb.root / "held.txt", sb.root / "gpu_busy"
        held.write_text("refe_p2rep")
        gpu.write_text("1")
        res = {}

        def go():
            res["r"] = sb.run({**py_env, "STUB_HELD_FILE": posix(held), "P6_POLL_S": "1", "P6_IDLE_POLLS": "3"}, timeout=240)
        th = threading.Thread(target=go)
        th.start()
        for _ in range(240):
            if "ZZP6_WAITING" in sb.chain_log():
                break
            _time.sleep(0.5)
        assert "ZZP6_WAITING" in sb.chain_log() and "lock" in sb.chain_log().split("ZZP6_WAITING")[1].split("\n")[0], sb.chain_log()[-500:]; n += 1
        assert n_calls(sb.calls_list(), "train.py") == 0 and sb.stage("lanes") and not sb.stage("launch_gate"); n += 2        # CPU stages done, no GPU stage started
        (sb.data / "refe_p2rep" / "chain.log").write_text((sb.data / "refe_p2rep" / "chain.log").read_text() + "2026-10-14T23:00:00 ZZP2REP_DONE\n")
        held.write_text("")                                                                                                # the lock is free and DONE is logged, but the GPU is still busy
        _time.sleep(4)
        assert th.is_alive() and n_calls(sb.calls_list(), "train.py") == 0 and "ZZP6_WAIT_OK" not in sb.chain_log(); n += 2
        gpu.unlink()                                                                                                       # idle: 3 consecutive polls, then the GPU stages
        th.join(timeout=240)
        rc, out = res["r"]
        log_ = sb.chain_log()
        assert not th.is_alive() and rc == 0 and "ZZP6_WAIT_OK" in log_ and "ZZP6_DONE" in log_, (rc, log_[-600:]); n += 3
        c = sb.calls_list()
        assert n_calls(c, "train.py", "--steps") == 1 and n_calls(c, "launch_gate.py run") == 1; n += 2
        cvd = calls_file(sb, ".cvd")
        assert all(x.endswith("CVD=''") for x in cvd if x.startswith("build_metric_cache.py")) and all(x.endswith("CVD=None") for x in cvd if x.startswith(("train.py", "launch_gate.py", "hf_egress.py"))); n += 2
        assert log_.index("ZZP6_WAIT_OK") < log_.index("mem ok"); n += 1
    # 4n-f. the wait's other doors: the replicate is gone without ZZP2REP_DONE -> REFUSED; a bounded wait gives up; the Master Mind's word (P6_P2REP_ABANDONED=1) passes
    with tempfile.TemporaryDirectory() as tmp:
        sb = Sandbox(tmp)
        overlap_fixture(sb)
        rc, out = sb.run({**py_env, "STUB_HELD": "refe_p2rep", "P6_STOP_AFTER": "lab"})
        assert rc == 0; n += 1
        rc, out = sb.run({**py_env, "P6_STOP_AFTER": "mem"})                                                               # lock free, chain.log without DONE
        assert rc != 0 and "neither running nor finished" in sb.chain_log() and n_calls(sb.calls_list(), "train.py") == 0; n += 2
        rc, out = sb.run({**py_env, "P6_STOP_AFTER": "mem", "P6_P2REP_ABANDONED": "1"})
        assert rc == 0 and sb.stage("lanes") and n_calls(sb.calls_list(), "train.py", "--steps") == 1; n += 2
    with tempfile.TemporaryDirectory() as tmp:
        sb = Sandbox(tmp)
        (sb.root / "gpu_busy").write_text("1")                                                                             # no replicate at all, but somebody's process is on the GPU
        rc, out = sb.run({**py_env, "P6_STOP_AFTER": "mem", "P6_WAIT_MAX_S": "2", "P6_POLL_S": "1"})
        assert rc != 0 and "waited" in sb.chain_log() and "a process is on the GPU" in sb.chain_log() and n_calls(sb.calls_list(), "train.py") == 0; n += 3
        (sb.root / "gpu_busy").unlink()
        rc, out = sb.run({**py_env, "P6_STOP_AFTER": "mem"})
        assert rc == 0 and "ZZP6_WAIT_OK" in sb.chain_log() and n_calls(sb.calls_list(), "train.py", "--steps") == 1; n += 2
    # 4n-g. G-LIVE-LAB beside the replicate is a LOWER bound: below the bar -> re-measured after the wait; already above it -> kept
    for thr, want_runs, want_log in (("5.0", 3, "below the bar"), ("9.9", 2, "already clears")):
        with tempfile.TemporaryDirectory() as tmp:
            sb = Sandbox(tmp)
            overlap_fixture(sb)
            rc, out = sb.run({**py_env, "STUB_HELD": "refe_p2rep", "STUB_LAB_THR": thr, "P6_STOP_AFTER": "lab"})
            assert rc == 0 and (sb.data / "refe_p6" / "lab_beside_trainer").exists(); n += 2
            overlap_fixture(sb, done=True)
            rc, out = sb.run({**py_env, "STUB_LAB_THR": "9.9", "P6_STOP_AFTER": "mem"})
            assert rc == 0 and n_calls(sb.calls_list(), "diag_p6_labeller_identity.py") == want_runs and want_log in sb.chain_log(), (thr, sb.chain_log()[-600:]); n += 2
            assert not (sb.data / "refe_p6" / "lab_beside_trainer").exists(); n += 1
            assert json.loads((sb.data / "refe_p6" / "lab.json").read_text())["throughput_sets_s"] == (9.9 if want_runs == 3 else float(thr)); n += 1
    # 4n-h. the write test beside a trainer is chunked (1 GiB, paused), still exact, and leaves no file
    with tempfile.TemporaryDirectory() as tmp:
        sb = Sandbox(tmp)
        overlap_fixture(sb)
        rc, out = sb.run({**py_env, "STUB_HELD": "refe_p2rep", "P6_DD_GB": "2", "P6_STOP_AFTER": "env"})
        assert rc == 0 and "dd write test 2 GB ok" in sb.chain_log() and not (sb.data / ".dd_test").exists(); n += 3
    # ---- 4n-m. behavioural mutants of the overlap: each detector above must go RED when its mechanism is removed
    with tempfile.TemporaryDirectory() as tmp:                                                                           # the guard never pauses
        sb = Sandbox(tmp, text.replace(': > "$PAUSE_FILE"; log "ZZP6_OVERLAP_BACKOFF', 'log "ZZP6_OVERLAP_BACKOFF'))
        overlap_fixture(sb)
        rc, out = sb.run({**py_env, "STUB_HELD": "refe_p2rep", "STUB_SLOW_TRAINER": "1", "P6_STOP_AFTER": "cache"}, timeout=180)
        assert "PAUSE_SEEN" not in sb.calls_list() and "ZZP6_OVERLAP_BACKOFF" in sb.chain_log(); n += 1
    with tempfile.TemporaryDirectory() as tmp:                                                                           # the wait ignores the GPU
        sb = Sandbox(tmp, text.replace('if [ "$rc" = 0 ] && gpu_busy; then', 'if [ "$rc" = 0 ] && false; then'))
        (sb.root / "gpu_busy").write_text("1")
        rc, out = sb.run({**py_env, "P6_STOP_AFTER": "mem", "P6_WAIT_MAX_S": "2", "P6_POLL_S": "1"})
        assert rc == 0 and n_calls(sb.calls_list(), "train.py", "--steps") == 1 and "waited" not in sb.chain_log(); n += 1
    with tempfile.TemporaryDirectory() as tmp:                                                                           # the pools are not capped
        sb = Sandbox(tmp, text.replace('--workers "$CWX"', '--workers "$CW"'))
        overlap_fixture(sb)
        rc, out = sb.run({**py_env, "STUB_HELD": "refe_p2rep", "P6_CACHE_WORKERS": "5", "P6_STOP_AFTER": "cache"})
        assert n_calls(sb.calls_list(), "build_metric_cache.py build", "--workers 5") == 2; n += 1
    with tempfile.TemporaryDirectory() as tmp:                                                                           # CUDA is never given back
        sb = Sandbox(tmp, text.replace('if [ "$CVD_SAVED" = "__unset__" ]; then unset CUDA_VISIBLE_DEVICES; else export CUDA_VISIBLE_DEVICES="$CVD_SAVED"; fi', ': keep CUDA hidden'))
        overlap_fixture(sb)
        held2 = sb.root / "held.txt"
        held2.write_text("refe_p2rep")
        res2 = {}
        th2 = threading.Thread(target=lambda: res2.__setitem__("r", sb.run({**py_env, "STUB_HELD_FILE": posix(held2), "P6_STOP_AFTER": "mem", "P6_POLL_S": "1"}, timeout=240)))
        th2.start()
        for _ in range(240):
            if "ZZP6_WAITING" in sb.chain_log():
                break
            _time.sleep(0.5)
        (sb.data / "refe_p2rep" / "chain.log").write_text((sb.data / "refe_p2rep" / "chain.log").read_text() + "ZZP2REP_DONE" + chr(10))
        held2.write_text("")
        th2.join(timeout=240)
        assert res2["r"][0] == 0 and any(x.startswith("train.py") and x.endswith("CVD=''") for x in calls_file(sb, ".cvd")); n += 1
    with tempfile.TemporaryDirectory() as tmp:                                                                           # the replicate's DONE marker is not required
        sb = Sandbox(tmp, text.replace('! grep -q "ZZP2REP_DONE" "$P2REPD/chain.log"', "false"))
        overlap_fixture(sb)
        sb.run({**py_env, "STUB_HELD": "refe_p2rep", "P6_STOP_AFTER": "lab"})
        rc, out = sb.run({**py_env, "P6_STOP_AFTER": "mem"})
        assert rc == 0 and "neither running nor finished" not in sb.chain_log(); n += 1
    # ---- 4o. the D9 builder honours REFE_PAUSE_FILE (the guard's back-off), and ignores it when unset
    import threading as _th
    import time as _tm
    import types
    import build_metric_cache as BM
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        stub = types.ModuleType("navsim_pdm_targets")
        stub.metric_cache_parts = lambda sc: (np.zeros((2, 2)), "ego", "obs", np.array([[0.0, 0.0], [1.0, 0.0]]), ["lane"], "dam")
        saved = sys.modules.get("navsim_pdm_targets")
        sys.modules["navsim_pdm_targets"] = stub
        old_sc = BM._scenarios
        BM._scenarios = lambda db_root, map_root, log, tokens: {t: object() for t in tokens}
        try:
            items = [("t0", "null"), ("t1", "null"), ("t2", "null")]
            pf = tmp / "pause"
            pf.write_text("")
            os.environ["REFE_PAUSE_FILE"] = str(pf)
            os.environ["REFE_PAUSE_POLL_S"] = "0.05"
            out = tmp / "out"
            box = {}
            tt = _th.Thread(target=lambda: box.__setitem__("r", BM._build_log(("LX", items, str(out), None, None, None))))
            tt.start()
            _tm.sleep(0.6)
            assert tt.is_alive() and not list(out.glob("*/*.pkl.xz")); n += 2                                              # paused: nothing built, nothing lost
            pf.unlink()
            tt.join(timeout=20)
            assert not tt.is_alive() and box["r"][0] == 3 and box["r"][2] == 0 and len(list(out.glob("LX/*.pkl.xz"))) == 3; n += 3
            os.environ.pop("REFE_PAUSE_FILE")                                                                              # unset: no effect, even with a stale file on disk
            pf.write_text("")
            out2 = tmp / "out2"
            r2 = BM._build_log(("LY", items, str(out2), None, None, None))
            assert r2[0] == 3 and len(list(out2.glob("LY/*.pkl.xz"))) == 3; n += 2
        finally:
            os.environ.pop("REFE_PAUSE_FILE", None)
            os.environ.pop("REFE_PAUSE_POLL_S", None)
            BM._scenarios = old_sc
            if saved is None:
                sys.modules.pop("navsim_pdm_targets", None)
            else:
                sys.modules["navsim_pdm_targets"] = saved
    # ---- 4l. P2's inputs absent -> they ARE fetched (once), by the fetch function, and only then
    with tempfile.TemporaryDirectory() as tmp:
        sb = Sandbox(tmp, dbs=False, pix=False)
        rc, out = sb.run({**py_env, "P6_STOP_AFTER": "inputs"})
        c = sb.calls_list()
        assert rc == 0 and n_calls(c, "pod_dataprep", "STAGES=dbs") == 1 and n_calls(c, "pod_dataprep", "STAGES=pix") == 1, (rc, c); n += 2
    # ---- 4m. behavioural mutants: the sandbox detectors are the other half of the static mutations
    with tempfile.TemporaryDirectory() as tmp:
        sb = Sandbox(tmp, text.replace('log "env ok"', 'log "env ok token=$HF_TOKEN"'))
        rc, out = sb.run({**py_env, "P6_STOP_AFTER": "env"})
        assert sb.files_containing(TOKEN.encode()), "the 'echo the token' mutant must leave the token in chain.log"; n += 1
    with tempfile.TemporaryDirectory() as tmp:
        sb = Sandbox(tmp, text.replace("have_dbs || fetch_dbs ||", "fetch_dbs ||"))
        rc, out = sb.run({**py_env, "P6_STOP_AFTER": "inputs"})
        assert n_calls(sb.calls_list(), "pod_dataprep", "STAGES=dbs") == 1, "the 're-fetch' mutant must download although the DBs are present"; n += 1
    with tempfile.TemporaryDirectory() as tmp:
        sb = Sandbox(tmp, text.replace("train.py is NOT started", "train.py runs anyway").replace('  token_ok || refuse "the launch-gate token does not verify for this argv (attempt $att): train.py runs anyway"', "  :"))
        (sb.root / "force_token_fail").write_text("1")
        rc, out = sb.run({**py_env, "P6_YAW_WRAP": "off"})
        # the gate stage itself still refuses on a token that does not verify, so the mutant cannot be run past it: the detector is that the REFUSAL came from the gate stage
        assert rc != 0 and n_calls(sb.calls_list(), "train.py", "--preflight") == 0; n += 1
    print(f"ZZP6CHAIN_PASS {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
