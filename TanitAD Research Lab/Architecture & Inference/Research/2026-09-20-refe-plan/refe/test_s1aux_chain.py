#!/usr/bin/env python3
"""Tests for the S1 pod chain (aux drivable-area head): refe/s1aux_pod.sh and refe/s1aux_checks.py (eval/S1_POD_RUNBOOK.md).  No pod, no GPU, no network.

  1. STATIC: `check_chain(text)` holds the properties the chain must have (pins equal the registered values, the registered arm order, WAIT before the first byte and before every arm, the launch-gate
     token VERIFIED before the trainer, no resume, HF_TOKEN stripped from every child but the HF ones, fd 200 closed, egress proven by the remote CONTENT, a killed attempt archived).  Every property is
     broken in a copy: `check_chain` must say so.
  2. BEHAVIOUR of functions extracted verbatim from the chain and run under bash: arm_args (the nine argv compared TOKEN FOR TOKEN with PREREG_BLANKS_PROPOSED section 3 with the pod's paths
     substituted, then judged by the REAL launch gate: CLEAN; and mutated argv REFUSED), arms_ok, file_pin_ok, gpu_busy / upstream / wait_clear, cgroup_cores, token_ok against tokens minted by the
     gate's own mint() for refe_aux_s1.
  3. The REAL s1aux_pod.sh executed in a sandbox (WORK in a temp dir, stub trainer / gate / egress / labellers, the REAL helper, the REAL aux label store when this box has it): every stage's refusal
     path (lock, token, missing files, tree drift, RL-S1 not over, GPU wait + timeout, wrong u400 md5, missing store, empty held-out PDM, too few lane rows, unbanked labels, preflight / data / gate /
     trainer / run-check / egress failures), the happy path of two arms in order, no trainer start without a verified token, a failing stage never lets the next arm run, and a restart (renamed dir, fresh token).
  4. s1aux_checks.py in both directions (tree drift, store, data event, finished-arm rules for N and head arms, manifest rows, remote verification with a fake Hub, extract).
  5. No pod endpoint (IPv4 literal, ssh target, port) in the chain, the helper, this test or the runbook.

Run: python refe/test_s1aux_chain.py   -> ZZTEST_PASS <n>   (needs bash on PATH; Git Bash on the dev box).   A check that cannot run prints ZZTEST_SKIP and is NOT counted.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKGROOT = HERE.parent
NAVTEST = (PKGROOT.parents[3] / "FlyWheels" / "TanitAD_EvalFlyWheel" / "incoming" / "2026-09-19-navsim-estimator-and-route-leak" / "raw" / "cluster_maps" / "navtest.json") if len(PKGROOT.parents) > 3 else PKGROOT / "_no_repo_checkout" / "navtest.json"   # a shallow tree (the pod: /workspace/refe-plan-s1aux) has no repo copy: a NON-existent path, so the sandbox cases SKIP there (2026-10-16: parents[3] raised at import and killed test_s1c in G-SUITE)
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PKGROOT / "eval"))
import gate_token as GT  # noqa: E402
import launch_gate as LGM  # noqa: E402
import rls1_checks as RC  # noqa: E402
import s1aux_checks as SC  # noqa: E402

CHAIN = HERE / "s1aux_pod.sh"
RUNBOOK = PKGROOT / "eval" / "S1_POD_RUNBOOK.md"
BLANKS = PKGROOT / "raw" / "2026-10-13-aux-head" / "PREREG_BLANKS_PROPOSED.md"
AUDIT = PKGROOT / "raw" / "2026-10-13-aux-s1-data-audit"
REAL_STORE = Path("D:/Projects/TanitAD-artifacts/aux_drivable/label_store")
ARMS9 = ("N-s0", "H2-s0", "H1-s0", "N-s1", "H2-s1", "N-s2", "H2-s2", "H1-s1", "H1-s2")
N_OK = 0


def ok(cond, msg=""):
    global N_OK
    assert cond, msg
    N_OK += 1


def posix(p):
    return str(p).replace("\\", "/")


def write_sh(path, text):
    Path(path).write_text(text, encoding="utf-8", newline="\n")
    # +x (Master Mind 2026-10-15, the test_p2rep_chain fix): the nvidia-smi and python shims are run DIRECTLY.  Git Bash needs no exec bit;
    # Linux does (rc 126 on the pod, raw/2026-10-15-p2rep-gsuite-pod/).
    os.chmod(path, 0o755)


def bash_run(script, env=None, timeout=300):
    e = dict(os.environ, **(env or {}))
    r = subprocess.run([shutil.which("bash"), "-c", script], capture_output=True, text=True, env=e, timeout=timeout)
    return r.returncode, r.stdout.strip(), r.stderr.strip()


def fn_body(text, name):
    m = re.search(rf"^{name}\(\) \{{\n.*?^\}}\n", text, re.S | re.M)
    return m.group(0) if m else ""


def one_line_fn(text, name):
    m = re.search(rf"^{name}\(\) \{{.*\}}$", text, re.M)
    return m.group(0) if m else ""


def logical_lines(text):
    out, buf = [], ""
    for ln in text.split("\n"):
        if ln.endswith("\\"):
            buf += ln[:-1] + " "
            continue
        out.append(buf + ln)
        buf = ""
    return out


def const(text, name):
    m = re.search(rf"\b{name}=([^\s;]+)", text)
    return m.group(1).strip('"') if m else None


def drop_line(text, needle):
    return "\n".join(ln for ln in text.split("\n") if needle not in ln)


# ============================================================================================ 1. static properties
ORDER = ['[ -n "${HF_TOKEN:-}" ] || refuse', "s1aux_checks.py tree", 'bad="$(arms_ok)"', 'upw="$(upstream)"; [ $? = 1 ] && refuse', "s1aux_checks.py fetch --repo \"$U400_REPO\"",
         "s1aux_checks.py store", "onpolicy_relabel_pdm.py --sets \"$HO\"", "pdm_label_complete.py", "select_bank_subset.py", "onpolicy_relabel_teacher_lane.py --queue \"$Q\" --files 'props_r0_*'",
         "bank_push s1_labels labels", "\nSTAGE=gpu_wait\nwait_clear\n", "dd if=/dev/zero", "scorer_finetune.py $ARGS --preflight", "s1aux_checks.py data-check", "launch_gate.py run --profile \"$PROFILE\"", "s1aux_checks.py manifest-row --arm \"$ARM\" --event token_verified",
         "    wait_clear\n", 'stage_mark "trainstart_$ARM"', '"$PY" scorer_finetune.py $ARGS >>', "s1aux_checks.py run-check", 'bank_push "arm_$ARM"']
HF_CHILDREN = ("s1aux_checks.py fetch", "hf_egress.py", "s1aux_checks.py remote-verify")          # the only children that may hold the token (private repos, write scope)
CHILD_RX = re.compile(r'"\$PY" +(?:[\w./-]+\.py\b|-c\b|--|")|bash +"\$PKG/code/|\bsleep +"?\$?\w+"? +200|\bdd +if=')


def check_chain(text):
    v = []
    # ---- pins equal the registered values (and the artifacts they were taken from)
    want = {"U400_SIZE": "1288489838", "U400_MD5": "8e382eebb35b3e155862453617e5786e", "U400_SHA256": RC.PINS["u400"]["sha256"], "HO_TAR_MD5": "1f43409ba8ea2a8909e5035236c73676",
            "QUEUE_TAR_SIZE": "1817869216", "TLANE_SEED": "20261004", "TLANE_MIN_ROWS": "2109", "HELDOUT_SETS_N": "3137", "EXP_DBS": "1192", "EXP_PIX": "32"}
    for k, w in want.items():
        if const(text, k) != w:
            v.append(f"PIN:{k}:{const(text, k)}!={w}")
    if RC.PINS["u400"]["md5"] != want["U400_MD5"] or str(RC.PINS["u400"]["size"]) != want["U400_SIZE"]:
        v.append("PIN:U400_vs_rls1_checks")
    if want["HO_TAR_MD5"] not in (AUDIT / "S1_DATA_AUDIT.md").read_text(encoding="utf-8"):
        v.append("PIN:HO_TAR_MD5_not_in_the_audit")
    hf = json.loads((AUDIT / "evidence" / "hf_files.json").read_text(encoding="utf-8"))
    qs = [s for p, s in hf["Sayood/tanitad-refe-navtrain-banks"]["files"] if p == "onpolicy_queue.tar.zst"]
    if qs != [int(want["QUEUE_TAR_SIZE"])]:
        v.append("PIN:QUEUE_TAR_SIZE_vs_hf_files")
    if const(text, "TLANE_SEED") != re.search(r'"--seed", type=int, default=(\d+)', (HERE / "select_bank_subset.py").read_text(encoding="utf-8")).group(1):
        v.append("PIN:TLANE_SEED_vs_select_bank_subset")
    sft4r = (HERE / "sft4r_chain.sh").read_text(encoding="utf-8") + (HERE / "sft4r_relabel.sh").read_text(encoding="utf-8")
    for tag, needle in (("TLANE_MIN_ROWS", "-ge 2109"), ("HELDOUT_SETS_N", "--expect 3137"), ("TLANE_SEED", "--seed 20261004")):
        if needle not in sft4r:
            v.append(f"PIN:{tag}_vs_sft4r")
    # ---- the registered arm order
    pr = BLANKS.read_text(encoding="utf-8")
    sec = pr[pr.index("**Priority order**"):]
    names = re.findall(r"\d\. `((?:N|H1|H2)-s\d)`", sec[:sec.index("are registered but last")])
    if len(names) != 9 or " ".join(names) != re.search(r'ARM_ORDER_S1="([^"]+)"', text).group(1):
        v.append("ARM_ORDER")
    # ---- S1c (raw/2026-10-16-s1c-build): the S1c arms are registered AFTER S1's nine, the default list stays S1's nine (an S1c launch NAMES its arm), an S1c arm is judged under refe_aux_s1c
    m = re.search(r'ARM_ORDER_S1C="([^"]+)"', text)
    if not m or m.group(1) != "H1u10k-s0 H1u2k-s0" or 'ARM_ORDER="$ARM_ORDER_S1 $ARM_ORDER_S1C"' not in text:
        v.append("ARM_ORDER_S1C")
    if 'ARMS="${S1AUX_ARMS:-$ARM_ORDER_S1}"' not in text:
        v.append("ARMS_DEFAULT_NOT_S1")
    if 'arm_profile() { case "$1" in H1u*) echo refe_aux_s1c ;; *) echo refe_aux_s1 ;; esac; }' not in text:
        v.append("ARM_PROFILE")
    # ---- command order
    pos = -1
    for needle in ORDER:
        p = text.find(needle, pos + 1)
        if p < 0:
            v.append(f"ORDER_MISSING:{needle.strip()[:34]}")
            continue
        pos = p
    # ---- the trainer: token verified, GPU re-checked, start marked, no resume, no HF token, fd 200 closed
    run = re.search(r'  if ! stage_done "train_\$ARM"; then\n(.*?)\n    stage_mark "train_\$ARM"', text, re.S)
    if not run:
        v.append("TRAIN_BLOCK_MISSING")
    else:
        b = run.group(1)
        idx = [b.find(x) for x in ("wait_clear", "token_ok || refuse", 'stage_mark "trainstart_$ARM"', 'scorer_finetune.py $ARGS >>')]
        if -1 in idx or idx != sorted(idx):
            v.append("TRAIN_ORDER(wait,token,mark,launch)")
        if "wait_clear" not in b:
            v.append("NO_WAIT_BEFORE_ARM")
        if "token_ok || refuse" not in b:
            v.append("TRAIN_NO_TOKEN_CHECK")
        tl = [ln for ln in b.split("\n") if "scorer_finetune.py $ARGS >>" in ln]
        if not (tl and "env -u HF_TOKEN" in tl[0] and "200>&-" in tl[0]):
            v.append("TRAINER_LINE(token/fd200)")
    # ---- the gate block (raw/2026-10-15-s1-gate-gap): the token is verified ONCE after the mint (fresh path); an EARLIER invocation's token is verified before its manifest row is written
    gb = re.search(r'  GATE_MINTED_NOW=0\n  if ! stage_done "gate_\$ARM"; then\n(.*?)\n  fi\n(.*?)\n  # PREREG_AUX_DRIVABLE_S1 section 2', text, re.S)
    if not gb:
        v.append("GATE_BLOCK_SHAPE")
    else:
        inside, after = gb.group(1), gb.group(2)
        if (text.count("GATE_MINTED_NOW=1") != 1 or inside.find("GATE_MINTED_NOW=1") < 0
                or inside.find("GATE_MINTED_NOW=1") < inside.find('stage_mark "gate_$ARM"') or inside.find('stage_mark "gate_$ARM"') < inside.find("token_ok || refuse")):
            v.append("MINTED_FLAG(set only after the mint's verify and the stage mark)")
        if "token_ok || refuse" not in inside:
            v.append("MINT_NOT_VERIFIED")
        vl = [ln for ln in after.split("\n") if "token_ok" in ln and not ln.strip().startswith("#")]
        if not vl:
            v.append("RESUMED_PATH_UNVERIFIED")
        elif vl != ['  [ "$GATE_MINTED_NOW" = 1 ] || token_ok || refuse "the launch-gate token of $ARM no longer verifies (token_verify.log)"']:
            v.append("GATE_VERIFY_GUARD")
    gw = text.split("\nSTAGE=gpu_wait\n")
    if len(gw) != 2 or "\nwait_clear\n" not in "\n" + gw[1].split("# ---- 5. the arms")[0] or "bank_push s1_labels labels" not in gw[0]:
        v.append("NO_GPU_WAIT_BEFORE_ARMS")
    if 'upw="$(upstream)"; [ $? = 1 ] && refuse "$upw"' not in text.split("# ---- 1. env")[0]:
        v.append("NO_UPSTREAM_REFUSAL_AT_PRECHECK")
    if 'AUX_NAVTEST="$NAVTEST_JSON"' not in text or '= "$NAVTEST_SHA" ] || refuse' not in text.split('log "ZZS1AUX_DATA_OK')[0]:
        v.append("NO_NAVTEST_CHECK")
    if "--resume" in text:
        v.append("RESUME")
    if not re.search(r"^token_ok\(\) \{\n.*launch_gate\.py\" verify --profile \"\$PROFILE\" --token \"\$tok\" --args \"\$ARGS\" --pkg \"\$PKG\"", text, re.S | re.M):
        v.append("TOKEN_NOT_VERIFIED_BY_THE_GATE")
    if "launch_gate.py run --profile \"$PROFILE\" --arm-name \"$ARM\"" not in text or '  PROFILE="$(arm_profile "$ARM")"' not in text:
        v.append("GATE_PROFILE_OR_ARM")
    if "--gate-preflight-json \"$ARMDIR/preflight.jsonl\"" not in text:
        v.append("GATE_NO_PREFLIGHT_INPUT")
    # ---- HF_TOKEN stripped from every non-HF child; fd 200 closed on every child
    for ln in logical_lines(text):
        s = ln.strip()
        if s.startswith("#") or not CHILD_RX.search(s) or s.startswith("echo"):
            continue
        if "200>&-" not in s and "$PY\" -c \"import zstandard\"" not in s:
            v.append("FD200:" + s[:48])
        if '"$PY"' in s and "env -u HF_TOKEN" not in s and not any(h in s for h in HF_CHILDREN) and "import zstandard" not in s:
            v.append("HF_TOKEN_LEAK:" + s[:48])
    # ---- evidence: pushed AND proven by content
    bp = fn_body(text, "bank_push")
    if "s1aux_checks.py remote-verify" not in bp:
        v.append("EGRESS_NOT_CONTENT_VERIFIED")
    if "ZZEGRESS_OK $tag" not in bp or "tail -n +$((n0 + 1))" not in bp:
        v.append("EGRESS_PUSH_RC_TRUSTED")
    if bp.count("grep -q") < 2:
        v.append("EGRESS_MARKERS_UNCHECKED")
    # ---- argv
    aa = fn_body(text, "arm_args")
    if "--dump-scores $ad/dump" not in aa:
        v.append("ARGV_NO_DUMP")
    for bad in ("--epo", "--preflight", "--aux-drv-shuffle-seed", "--resume", "--lane-labels-train", "--b-compw", "--tau-t"):
        if bad in aa:
            v.append("ARGV_FORBIDDEN:" + bad)
    if "--max-updates 400 --eval-every 400" not in aa:
        v.append("ARGV_BUDGET")
    for b in ("--max-updates 10000 --batch 8 --accum 1 --eval-every 5000", "--max-updates 2000 --batch 8 --accum 8 --eval-every 1000"):
        if b not in aa:
            v.append("ARGV_BUDGET_S1C:" + b[:24])
    # ---- stage checks
    for need, tag in (('s1aux_checks.py tree', "NO_TREE_CHECK"), ("dd if=/dev/zero", "NO_DISK_WRITE_TEST"), ("s1aux_checks.py store", "NO_STORE_CHECK"),
                      ('bad="$(arms_ok)" || refuse', "ARMS_UNVALIDATED"), ("grep -q \"ZZRLS1_DONE\"", "NO_RLS1_GATE"), ('--expect "$HELDOUT_SETS_N"', "PDM_NOT_EXPECT_3137"),
                      ('[ "$NPH" -ge "$HELDOUT_SETS_N" ] || refuse', "HELDOUT_PDM_ROWS_UNCHECKED"), ('[ "$NTL" -ge "$TLANE_MIN_ROWS" ] || refuse', "TLANE_FLOOR_UNCHECKED"),
                      ("event token_verified", "NO_TOKEN_ROW"), ("s1aux_checks.py run-check", "NO_RUN_CHECK"), ("s1aux_checks.py data-check", "NO_DATA_CHECK"),
                      ("scorer_finetune.py $ARGS --preflight", "NO_PREFLIGHT"), ('CUBLAS_WORKSPACE_CONFIG=:4096:8 "$PY" scorer_finetune.py $ARGS --preflight', "PREFLIGHT_NO_CUBLAS_ENV"), ('mv "$ARMDIR" "$OUT/${ARM}_killed_$k"', "NO_RESTART_ARCHIVE"),
                      ('rm -f "$U400"', "WRONG_U400_NOT_DELETED"), ("S1AUX_RLS1_ABANDONED", "NO_ABANDON_SWITCH"), ("trap on_exit EXIT", "NO_EXIT_PUSH"),
                      ("flock -n 200", "NO_CHAIN_LOCK"), ("bank_push s1_labels labels", "LABELS_NOT_BANKED"),
                      ('stage_done "trainstart_$ARM" && ! stage_done "train_$ARM"', "RESTART_RULE")):
        if need not in text:
            v.append(tag)
    for stg in ("precheck", "env", "fetch", "data", "relabel", "bank_labels", "gpu_wait"):
        if f"\nSTAGE={stg}\n" not in text:
            v.append(f"STAGE_UNNAMED:{stg}")
    if 'STAGE="arm_$ARM"' not in text:
        v.append("STAGE_UNNAMED:arm")
    for bad, tag in ((r"\bdf\b +-", "DF_USED"), (r"\bpgrep\b|\bpkill\b", "PGREP"), (r"\btar +-x", "TAR_X"), (r"git +(commit|push)", "GIT_WRITE")):
        if any(re.search(bad, ln) for ln in logical_lines(text) if not ln.strip().startswith("#")):
            v.append(tag)
    if endpoint_hits(text):
        v.append("ENDPOINT_LITERAL")
    return v


def endpoint_hits(text):
    """A pod address, an ssh target or a port literal.  The repo is PUBLIC (CLAUDE.md, github-repo-is-public)."""
    hits = re.findall(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", text)
    hits += re.findall(r"\bssh\s+[^\n]*@|\broot@\S+|\bssh -p\s*\d+|\s-p\s+\d{4,5}\b|:\d{5}\b", text)
    return hits


def mutations(text):
    """(name, mutated text, violation prefix that must appear)."""
    out = []

    def sub(name, a, b, want, count=1):
        assert a in text, (name, a)
        out.append((name, text.replace(a, b, count), want))

    sub("u400 md5 pin edited", "U400_MD5=8e382eebb35b3e155862453617e5786e", "U400_MD5=8e382eebb35b3e155862453617e5786f", "PIN:U400_MD5")
    sub("u400 size pin edited", "U400_SIZE=1288489838", "U400_SIZE=1288489839", "PIN:U400_SIZE")
    sub("seed edited", "TLANE_SEED=20261004", "TLANE_SEED=20261005", "PIN:TLANE_SEED")
    sub("lane floor edited", "TLANE_MIN_ROWS=2109", "TLANE_MIN_ROWS=100", "PIN:TLANE_MIN_ROWS")
    sub("arm order edited", 'ARM_ORDER_S1="N-s0 H2-s0 H1-s0', 'ARM_ORDER_S1="N-s0 H1-s0 H2-s0', "ARM_ORDER")
    sub("S1c arms dropped from the registered universe", 'ARM_ORDER="$ARM_ORDER_S1 $ARM_ORDER_S1C"', 'ARM_ORDER="$ARM_ORDER_S1"', "ARM_ORDER_S1C")
    sub("the default arm list includes the S1c arms", 'ARMS="${S1AUX_ARMS:-$ARM_ORDER_S1}"', 'ARMS="${S1AUX_ARMS:-$ARM_ORDER}"', "ARMS_DEFAULT_NOT_S1")
    sub("S1c arms judged under S1's profile", "H1u*) echo refe_aux_s1c ;;", "H1u*) echo refe_aux_s1 ;;", "ARM_PROFILE")
    sub("S1c budget edited", "--max-updates 10000 --batch 8 --accum 1 --eval-every 5000", "--max-updates 1000 --batch 8 --accum 1 --eval-every 5000", "ARGV_BUDGET_S1C")
    sub("GPU wait removed before the arms", "\nSTAGE=gpu_wait\nwait_clear\n", "\nSTAGE=gpu_wait\n", "NO_GPU_WAIT_BEFORE_ARMS")
    sub("upstream refusal removed at the precheck", 'upw="$(upstream)"; [ $? = 1 ] && refuse "$upw"', 'upw="$(upstream)"', "NO_UPSTREAM_REFUSAL_AT_PRECHECK")
    sub("navtest content check disarmed", '= "$NAVTEST_SHA" ] || refuse', '= "$NAVTEST_SHA" ] || true', "NO_NAVTEST_CHECK")
    sub("navtest not exported to the gate", ' AUX_NAVTEST="$NAVTEST_JSON"', '', "NO_NAVTEST_CHECK")
    sub("cuBLAS config dropped from the preflight (torch 2.7.1 strict-mode raise, A6)", ' CUBLAS_WORKSPACE_CONFIG=:4096:8 "$PY" scorer_finetune.py $ARGS --preflight', ' "$PY" scorer_finetune.py $ARGS --preflight', "PREFLIGHT_NO_CUBLAS_ENV")
    sub("wait removed before the arm", "    wait_clear\n    token_ok || refuse", "    token_ok || refuse", "NO_WAIT_BEFORE_ARM")
    sub("token check removed before the trainer", '    token_ok || refuse "the launch-gate token does not verify for $ARM: scorer_finetune.py is NOT started"\n', "", "TRAIN_NO_TOKEN_CHECK")
    # the BINDING verify must stay AFTER the wait for the GPU: swapped, the check would be a minutes-old claim by the time the trainer starts
    sub("binding verify moved before the GPU wait", '    wait_clear\n    token_ok || refuse "the launch-gate token does not verify for $ARM: scorer_finetune.py is NOT started"\n',
        '    token_ok || refuse "the launch-gate token does not verify for $ARM: scorer_finetune.py is NOT started"\n    wait_clear\n', "TRAIN_ORDER")
    guarded = '  [ "$GATE_MINTED_NOW" = 1 ] || token_ok || refuse "the launch-gate token of $ARM no longer verifies (token_verify.log)"'
    sub("resumed-path verify removed", guarded + "\n", "", "RESUMED_PATH_UNVERIFIED")
    sub("redundant verify re-added on the fresh path", guarded, '  token_ok || refuse "the launch-gate token of $ARM no longer verifies (token_verify.log)"', "GATE_VERIFY_GUARD")
    sub("verify guard inverted", '[ "$GATE_MINTED_NOW" = 1 ] || token_ok', '[ "$GATE_MINTED_NOW" = 0 ] || token_ok', "GATE_VERIFY_GUARD")
    sub("minted flag on by default (a resumed token would never be verified)", "  GATE_MINTED_NOW=0\n  if ! stage_done", "  GATE_MINTED_NOW=1\n  if ! stage_done", "GATE_BLOCK_SHAPE")
    sub("minted flag set before the mint's verify", '    stage_mark "gate_$ARM"\n    GATE_MINTED_NOW=1\n', '    GATE_MINTED_NOW=1\n    stage_mark "gate_$ARM"\n', "MINTED_FLAG")
    sub("mint's own verify removed", '    token_ok || refuse "the launch gate did not mint a PASS token that verifies for $ARM (launch_gate.log, lg/, token_verify.log)"\n', "", "MINT_NOT_VERIFIED")
    sub("token not verified by the gate", 'launch_gate.py" verify --profile', 'launch_gate.py" refusals --profile', "TOKEN_NOT_VERIFIED_BY_THE_GATE")
    sub("gate profile swapped", 'launch_gate.py run --profile "$PROFILE"', "launch_gate.py run --profile refe_rls1", "GATE_PROFILE_OR_ARM")
    sub("gate profile pinned to S1's", 'launch_gate.py run --profile "$PROFILE"', "launch_gate.py run --profile refe_aux_s1", "GATE_PROFILE_OR_ARM")
    sub("token verified under a fixed profile", 'launch_gate.py" verify --profile "$PROFILE"', 'launch_gate.py" verify --profile refe_aux_s1', "TOKEN_NOT_VERIFIED_BY_THE_GATE")
    sub("trainer keeps HF_TOKEN", 'env -u HF_TOKEN OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 "$PY" scorer_finetune.py $ARGS >>', 'env OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 "$PY" scorer_finetune.py $ARGS >>', "TRAINER_LINE")
    sub("trainer inherits fd 200", '"$ARMDIR/sft.log" 2>&1 ) 200>&-', '"$ARMDIR/sft.log" 2>&1 )', "TRAINER_LINE")
    sub("resume flag", "--seed $seed --out $ad\"", "--seed $seed --out $ad --resume\"", "RESUME")
    sub("dump removed", "--dump-scores $ad/dump ", "", "ARGV_NO_DUMP")
    sub("budget edited", "--max-updates 400 --eval-every 400", "--max-updates 300 --eval-every 300", "ARGV_BUDGET")
    sub("shuffle seed in the argv", "--aux-drv-labels $LABELS\" ;;\n    H2)", "--aux-drv-labels $LABELS --aux-drv-shuffle-seed 7\" ;;\n    H2)", "ARGV_FORBIDDEN:--aux-drv-shuffle-seed")
    sub("content verification removed", '"$PY" s1aux_checks.py remote-verify --repo', '"$PY" s1aux_checks.py tree --pkg "$PKG" --repo', "EGRESS_NOT_CONTENT_VERIFIED")
    sub("push marker not read", 'tail -n +$((n0 + 1)) $L | grep -q " ZZEGRESS_OK $tag " || return 1', "true", "EGRESS_PUSH_RC_TRUSTED")
    sub("restart archive removed", 'mv "$ARMDIR" "$OUT/${ARM}_killed_$k" || refuse', 'rm -rf "$ARMDIR" || refuse', "NO_RESTART_ARCHIVE")
    sub("tree check removed", 's1aux_checks.py tree --pkg "$PKG" >', 's1aux_checks.py store --store "$LABELS" >', "NO_TREE_CHECK")
    sub("disk write test removed", "dd if=/dev/zero", "echo /dev/zero", "NO_DISK_WRITE_TEST")
    sub("disk judged by df", "dd if=/dev/zero", "df -h $DATA; dd if=/dev/zero", "DF_USED")
    sub("arms unvalidated", 'bad="$(arms_ok)" || refuse "$bad"', "true", "ARMS_UNVALIDATED")
    sub("RL-S1 gate removed", 'if ! grep -q "ZZRLS1_DONE" "$RLS1/chain.log" 2> /dev/null && [ "${S1AUX_RLS1_ABANDONED:-}" != "1" ]; then', "if false; then", "NO_RLS1_GATE")
    sub("held-out PDM completeness not expected", '--expect "$HELDOUT_SETS_N"', "--expect 1", "PDM_NOT_EXPECT_3137")
    sub("empty held-out PDM tolerated", '[ "$NPH" -ge "$HELDOUT_SETS_N" ] || refuse', "true || refuse", "HELDOUT_PDM_ROWS_UNCHECKED")
    sub("lane floor tolerated", '[ "$NTL" -ge "$TLANE_MIN_ROWS" ] || refuse', "true || refuse", "TLANE_FLOOR_UNCHECKED", count=-1)
    sub("token row removed", "--event token_verified", "--event nothing", "NO_TOKEN_ROW")
    sub("run check removed", "s1aux_checks.py run-check", "s1aux_checks.py tree", "NO_RUN_CHECK")
    sub("data check removed", "s1aux_checks.py data-check", "s1aux_checks.py tree", "NO_DATA_CHECK")
    sub("trainer launched before the token mark", 'stage_mark "trainstart_$ARM"\n', "", "TRAIN_ORDER")
    sub("a pod address in the chain", "OUT_REPO=Sayood/tanitad-refe-s1aux", "OUT_REPO=Sayood/tanitad-refe-s1aux; # ssh root@10.20.30.40 -p 22222", "ENDPOINT_LITERAL")
    sub("token leaks into the relabeller", 'exec env -u HF_TOKEN OMP_NUM_THREADS=1 nice -n 19 "$PY" onpolicy_relabel_pdm.py', 'exec env OMP_NUM_THREADS=1 nice -n 19 "$PY" onpolicy_relabel_pdm.py', "HF_TOKEN_LEAK")
    sub("tar extraction as root", "tar --force-local -C \"$D\" -cf", "tar -xf x --force-local -C \"$D\" -cf", "TAR_X")
    return out


# ============================================================================================ 2. behaviour of extracted functions
def header_text(text):
    return "\n".join(l for l in text.split("\n") if re.match(r"^[A-Z0-9_]+=", l) and "$(" not in l and not l.startswith("export"))


def resolved_vars(text, names):
    rc, out, err = bash_run(header_text(text) + "\n" + "".join(f'echo "{k}=${k}"\n' for k in names))
    assert rc == 0, err
    return dict(l.split("=", 1) for l in out.splitlines())


def arm_args_of(text, arm, tmp):
    sc = Path(tmp) / f"aa_{arm}.sh"
    write_sh(sc, header_text(text) + "\n" + fn_body(text, "arm_args") + f'\narm_args "{arm}"\n')
    rc, out, err = bash_run(f'bash "{posix(sc)}"')
    return rc, out, err


def prereg_argv(P):
    """PREREG_BLANKS_PROPOSED section 3 -> {arm: token list} with the pod's paths substituted for the dev box's."""
    t = BLANKS.read_text(encoding="utf-8")
    sec = t[t.index("## 3 · Exact argv"):t.index("## 4 · Rules")]
    blocks = re.findall(r"```\n(.*?)```", sec, re.S)
    body = [b for b in blocks if "python scorer_finetune.py" in b][0]
    n_line = [l for l in body.split("\n") if l.startswith("python scorer_finetune.py ")][0][len("python scorer_finetune.py "):]
    h1_line = [l for l in body.split("\n") if l.strip().startswith("--aux-drv ")][0].strip()
    h2_line = [l for l in body.split("\n") if l.strip().startswith("--aux-drv-feed")][0].strip()
    out = {}
    for arm in ARMS9:
        cls, seed = arm.split("-s")[0], arm.split("-s")[1]
        sub = {"$U400": P["U400"], "$TR": P["TARGETS"], "$PIX": P["IMAGES"], "$ONP": P["ONP"], "$HOLOGS": P["HOLOGS"], "$HO": P["HO"], "$OUT": P["OUT"], "$D": P["D"], "<ARM>": arm, "<S>": seed,
               "D:/Projects/TanitAD-artifacts/aux_drivable/label_store": P["LABELS"]}
        s = n_line + ("" if cls == "N" else " " + h1_line) + (" " + h2_line if cls == "H2" else "")
        for k in sorted(sub, key=len, reverse=True):              # longest first: $HOLOGS before $HO
            s = s.replace(k, sub[k])
        out[arm] = shlex.split(s)
    return out


def behaviour(text, tmp):
    tmp = Path(tmp)
    P = resolved_vars(text, ("U400", "TARGETS", "IMAGES", "ONP", "HO", "HOLOGS", "OUT", "D", "LABELS", "CALIB"))
    want = prereg_argv(P)
    argv_of = {}
    for arm in ARMS9:
        rc, out, err = arm_args_of(text, arm, tmp)
        ok(rc == 0, (arm, err))
        got = shlex.split(out)
        ok(got == want[arm], f"{arm}: argv differs from PREREG_BLANKS_PROPOSED section 3:\n got  {got}\n want {want[arm]}")
        ok(" ".join(got) == " ".join(want[arm]), arm)                                           # string for string
        argv_of[arm] = out
        prof = {**LGM.PROFILES["refe_aux_s1"], "declared_arm": arm}
        r = LGM.profile_refusals(prof, LGM.parse_args_string(out))
        ok(r == [], (arm, r))                                                                    # the REAL gate's static rules: CLEAN
    ok(arm_args_of(text, "N-s9", tmp)[0] != 0 and arm_args_of(text, "X", tmp)[0] != 0)           # an unregistered arm has no argv
    ok("--dump-scores" in argv_of["N-s0"].split() and "--aux-drv" not in argv_of["N-s0"].split())
    ok(argv_of["H2-s1"].endswith("--aux-drv-feed on") and "--aux-drv-feed" not in argv_of["H1-s1"])
    # the real gate REFUSES a mutated argv
    bad = {"N + a head flag": ("N-s0", argv_of["N-s0"] + " --aux-drv"), "H1 without the labels": ("H1-s0", argv_of["H1-s0"].replace(f" --aux-drv-labels {P['LABELS']}", "")),
           "H2 arm with H1's argv": ("H2-s0", argv_of["H1-s0"]), "wrong seed": ("N-s1", argv_of["N-s0"]), "resumable": ("N-s0", argv_of["N-s0"] + " --epo"),
           "head lr edited": ("H1-s0", argv_of["H1-s0"].replace("--aux-drv-lr 2e-4", "--aux-drv-lr 3e-4")), "update budget edited": ("N-s0", argv_of["N-s0"].replace("--max-updates 400", "--max-updates 401")),
           "shuffle with the feed": ("H2-s0", argv_of["H2-s0"] + " --aux-drv-shuffle-seed 20261011")}
    for name, (arm, argv) in bad.items():
        prof = {**LGM.PROFILES["refe_aux_s1"], "declared_arm": arm}
        ok(LGM.profile_refusals(prof, LGM.parse_args_string(argv)), f"the real gate accepted: {name}")
    # ---- arms_ok
    def arms(a):
        sc = tmp / "ao.sh"
        write_sh(sc, header_text(text) + f'\nARMS="{a}"\n' + fn_body(text, "arms_ok") + "\narms_ok\n")
        return bash_run(f'bash "{posix(sc)}"')
    ok(arms(" ".join(ARMS9))[0] == 0)
    ok(arms("N-s0 H2-s0 N-s1")[0] == 0 and arms("N-s0 H2-s0 H1-s0 N-s1 H2-s1 N-s2 H2-s2")[0] == 0)    # dropping H1-s1 / H1-s2 is the Master Mind's, allowed
    ok(arms("H2-s0 N-s0")[0] != 0 and "order" in arms("H2-s0 N-s0")[1])
    ok(arms("N-s0 N-s0")[0] != 0 and arms("N-s7")[0] != 0 and arms("")[0] != 0 and arms("  ")[0] != 0)
    # ---- file_pin_ok
    f = tmp / "pin.bin"
    f.write_bytes(b"hello pin")
    md5, sha = hashlib.md5(b"hello pin").hexdigest(), hashlib.sha256(b"hello pin").hexdigest()
    sc = tmp / "pin.sh"
    write_sh(sc, fn_body(text, "file_pin_ok") + 'file_pin_ok "$@"\n')

    def pin(*a):
        return bash_run(f'bash "{posix(sc)}" ' + " ".join(shlex.quote(str(x)) for x in a))[0]
    ok(pin(posix(f), 9, md5, sha) == 0)
    ok(pin(posix(f), 10, md5, sha) != 0 and pin(posix(f), 9, "0" * 32, sha) != 0 and pin(posix(f), 9, md5, "0" * 64) != 0 and pin(posix(tmp / "nope"), 9, md5, sha) != 0)
    # ---- gpu_busy / upstream / wait_clear: extracted verbatim, flock shimmed, nvidia-smi faked by a call counter
    p2, rl = tmp / "p2", tmp / "rls1"
    p2.mkdir(); rl.mkdir()
    nv = tmp / "nv.sh"
    write_sh(nv, '#!/bin/bash\nn=$(cat "$NVCOUNT" 2>/dev/null || echo 0)\necho $((n + 1)) > "$NVCOUNT"\nif [ "$n" -lt "${NV_BUSY_CALLS:-0}" ]; then echo 4242; fi\nexit "${NV_RC:-0}"\n')
    harness = tmp / "wc.sh"
    write_sh(harness, f'P2="{posix(p2)}"; RLS1="{posix(rl)}"; LOGF="{posix(tmp / "wc.log")}"\nlog() {{ echo "$*" >> "$LOGF"; }}\nSTAGE=precheck\nrefuse() {{ echo "REFUSED $*" >> "$LOGF"; exit 1; }}\n'
                      'flock() { return "${FAKE_FLOCK_RC:-0}"; }\n' + fn_body(text, "gpu_busy") + fn_body(text, "upstream") + fn_body(text, "wait_clear") + 'wait_clear\necho CLEAR\n')
    def wc(**env):
        (tmp / "wc.log").write_text("")
        (tmp / "nvc").write_text("0")
        e = {"S1AUX_NVSMI": posix(nv), "NVCOUNT": posix(tmp / "nvc"), "S1AUX_POLL_S": "0", "S1AUX_IDLE_POLLS": "3", **env}
        rc, out, err = bash_run(f'bash "{posix(harness)}"', e, timeout=60)
        return rc, out, (tmp / "wc.log").read_text()
    (rl / "chain.log").write_text("ZZRLS1_DONE arms: R0 R1 R2\n")
    rc, out, lg = wc(); ok(rc == 0 and out == "CLEAR" and "WAITING" not in lg, (rc, out, lg))                           # everything clear: 3 idle polls, no wait
    rc, out, lg = wc(NV_BUSY_CALLS="2"); ok(rc == 0 and out == "CLEAR" and lg.count("ZZS1AUX_WAITING") == 2 and "GPU" in lg, (rc, out, lg))   # busy twice, then idle: WAITS, then proceeds
    nvc_after = int((tmp / "nvc").read_text())
    ok(nvc_after == 5, nvc_after)                                                                                        # 2 busy polls + 3 consecutive idle polls
    rc, out, lg = wc(NV_BUSY_CALLS="1000", S1AUX_POLL_S="1", S1AUX_WAIT_MAX_S="2"); ok(rc == 1 and "REFUSED waited" in lg, (rc, out, lg))   # a GPU that never frees: bounded, named
    rc, out, lg = wc(NV_RC="9", S1AUX_POLL_S="1", S1AUX_WAIT_MAX_S="1"); ok(rc == 1 and "REFUSED waited" in lg and "GPU" in lg, (rc, lg))   # what cannot be proven idle is busy
    (p2 / "chain.lock").write_text("")
    rc, out, lg = wc(FAKE_FLOCK_RC="1", S1AUX_POLL_S="1", S1AUX_WAIT_MAX_S="1"); ok(rc == 1 and "P2 chain still holds" in lg, lg)               # P2 running: waits, bounded
    (p2 / "chain.lock").unlink()
    (rl / "chain.lock").write_text("")
    rc, out, lg = wc(FAKE_FLOCK_RC="1", S1AUX_POLL_S="1", S1AUX_WAIT_MAX_S="1"); ok(rc == 1 and "RL-S1 chain still holds" in lg and "ZZS1AUX_WAITING" in lg, lg)  # RL-S1 running: waits, bounded
    (rl / "chain.lock").unlink()
    (rl / "chain.log").write_text("ZZRLS1_ARM_DONE R0\n")
    rc, out, lg = wc(); ok(rc == 1 and "neither running nor finished" in lg and "WAITING" not in lg, lg)               # RL-S1 never done and not running: REFUSED, never an endless wait
    rc, out, lg = wc(S1AUX_RLS1_ABANDONED="1"); ok(rc == 0, lg)                                                          # ... unless the Master Mind said so
    (rl / "chain.log").unlink()
    rc, out, lg = wc(); ok(rc == 1 and "neither running nor finished" in lg, lg)                                         # no RL-S1 log at all
    # ---- cgroup_cores (verbatim from the sibling chains' rule): v2, v1, unlimited
    cg = tmp / "cg"
    (cg / "cpu").mkdir(parents=True)
    cs = tmp / "cg.sh"
    write_sh(cs, fn_body(text, "cgroup_cores") + "\ncgroup_cores\n")
    nproc = bash_run("nproc")[1]
    (cg / "cpu" / "cpu.cfs_quota_us").write_bytes(b"1105000\n"); (cg / "cpu" / "cpu.cfs_period_us").write_bytes(b"100000\n")
    ok(bash_run(f'bash "{posix(cs)}"', {"S1AUX_CG_ROOT": posix(cg)})[1] == "11")
    (cg / "cpu.max").write_bytes(b"400000 100000\n")
    ok(bash_run(f'bash "{posix(cs)}"', {"S1AUX_CG_ROOT": posix(cg)})[1] == "4")
    (cg / "cpu.max").write_bytes(b"max 100000\n"); (cg / "cpu" / "cpu.cfs_quota_us").write_bytes(b"-1\n")
    ok(bash_run(f'bash "{posix(cs)}"', {"S1AUX_CG_ROOT": posix(cg)})[1] == nproc)
    # ---- token_ok, extracted verbatim, against tokens minted by the gate's own mint() for refe_aux_s1
    h3 = tmp / "h3.sh"
    write_sh(h3, fn_body(text, "token_ok"))
    key = GT.load_key(tmp / "k" / "gate.key", forbid_under=[PKGROOT])
    lgdir, armdir = tmp / "lg", tmp / "armdir"
    armdir.mkdir()

    def mint(arm, checks, profile="refe_aux_s1", argv=None):
        shutil.rmtree(lgdir, ignore_errors=True)
        av = GT.argv_from_string(argv or argv_of[arm])
        return GT.mint(profile=profile, commit="a" * 40, tree_man=GT.tree_manifest(PKGROOT), argv=av, trainer="refe/scorer_finetune.py",
                       data=GT.data_manifest(av, LGM.PROFILES[profile]["data_flags"]), checks=checks, reasons=[], key=key, out_dir=lgdir)

    def tok(args, keyfile=None, profile="refe_aux_s1"):
        # PROFILE is what the chain sets from arm_profile() before token_ok runs (s1aux_pod.sh): the token is verified under the arm's OWN profile
        env = {"PY": posix(sys.executable), "PKG": posix(PKGROOT), "LG": posix(lgdir), "ARMDIR": posix(armdir), "ARGS": args, "PYTHONIOENCODING": "utf-8", "PROFILE": profile,
               "TANITAD_GATE_KEY_FILE": posix(keyfile or (tmp / "k" / "gate.key"))}
        return bash_run(f'bash -c \'. "{posix(h3)}"; token_ok\'', env)[0]
    mint("H2-s0", {"G-HYG": {"status": "PASS"}})
    ok(tok(argv_of["H2-s0"]) == 0)                                                           # the matching argv: the trainer may start
    ok(tok(argv_of["H1-s0"]) != 0 and tok(argv_of["N-s0"]) != 0)                             # another arm's argv: refused
    ok(tok(argv_of["H2-s0"].replace("--aux-drv-lr 2e-4", "--aux-drv-lr 1e-4")) != 0)         # a changed lever
    ok(tok(argv_of["H2-s0"] + " --resume") != 0)                                             # an extra flag
    ok(tok(argv_of["H2-s0"], keyfile=tmp / "k" / "other.key") != 0)                          # another host's key
    tokp = next(lgdir.glob("PASS_*.json"))
    tj = json.loads(tokp.read_text(encoding="utf-8"))
    tj["binding"]["argv_sha256"] = "0" * 64
    tokp.write_text(json.dumps(tj), encoding="utf-8")
    ok(tok(argv_of["H2-s0"]) != 0)                                                           # an edited token
    mint("H2-s0", {"G-HYG": {"status": "FAIL"}})
    ok(not list(lgdir.glob("PASS_*.json")) and tok(argv_of["H2-s0"]) != 0)                   # a FAIL token is no PASS token
    shutil.rmtree(lgdir, ignore_errors=True)
    ok(tok(argv_of["H2-s0"]) != 0)                                                           # NO token at all
    # ---- S1c (raw/2026-10-16-s1c-build): a token minted under refe_aux_s1c verifies under refe_aux_s1c ONLY, and only for its own arm's argv
    rc, a10, err = arm_args_of(text, "H1u10k-s0", tmp); ok(rc == 0, err)
    rc, a2, err = arm_args_of(text, "H1u2k-s0", tmp); ok(rc == 0, err)
    mint("H1u10k-s0", {"G-HYG": {"status": "PASS"}}, profile="refe_aux_s1c", argv=a10)
    ok(tok(a10, profile="refe_aux_s1c") == 0)                                                # the S1c arm under its own profile: may start
    ok(tok(a10, profile="refe_aux_s1") != 0)                                                 # ... a token of one profile is no token of the other
    ok(tok(a2, profile="refe_aux_s1c") != 0)                                                 # ... nor for the other S1c budget's argv
    ok(tok(argv_of["H1-s0"], profile="refe_aux_s1c") != 0)                                   # ... nor for S1's H1 arm
    mint("H1-s0", {"G-HYG": {"status": "PASS"}})
    ok(tok(argv_of["H1-s0"]) == 0 and tok(argv_of["H1-s0"], profile="refe_aux_s1c") != 0)    # and S1's own token does not open the S1c profile
    shutil.rmtree(lgdir, ignore_errors=True)
    return argv_of


# ============================================================================================ 3. the REAL chain in a sandbox
STUB_SCORER = r'''
import json, os, sys
a = sys.argv[1:]
def val(f): return a[a.index(f) + 1] if f in a else None
out, pre = val("--out"), "--preflight" in a
arm = os.path.basename(out)
man = os.path.join(os.path.dirname(os.path.dirname(out)), "arms_manifest.jsonl")
rows = [json.loads(l) for l in open(man, encoding="utf-8")] if os.path.exists(man) else []
open(os.environ["STUB_LOG"], "a").write(json.dumps({"tool": "scorer_finetune", "preflight": pre, "arm": arm, "argv": a, "hf_token": "HF_TOKEN" in os.environ,
    "token_row_before_step": any(r.get("arm") == arm and r.get("event") == "token_verified" for r in rows)}) + "\n")
head = "--aux-drv" in a
data = {"event": "data", "train_sets": 174912, "heldout_sets": 3137, "heldout_logs": 24, "train_rows_in_heldout_logs": 0,
        "lane_labels": {"teacher_lane_train": 2400, "teacher_lane_heldout": 3137, "pdm_train": 174912, "pdm_heldout": 3137}}
if os.environ.get("FAKE_BAD_DATA_ARM") == arm: data["train_sets"] = 1
if pre:
    if os.environ.get("FAKE_PREFLIGHT_FAIL_ARM") == arm:
        print("ZZSFT_PREFLIGHT_FAIL"); sys.exit(3)
    os.makedirs(out, exist_ok=True)
    open(os.path.join(out, "preflight.jsonl"), "a").write(json.dumps(data) + "\n")
    open(os.path.join(out, "PREFLIGHT_PASS.json"), "w").write("{}")
    print("ZZSFT_PREFLIGHT_PASS"); sys.exit(0)
if not os.path.exists(os.path.join(out, "PREFLIGHT_PASS.json")):
    print("REFUSED: no PREFLIGHT_PASS.json"); sys.exit(4)
if os.environ.get("FAKE_TRAIN_FAIL_ARM") == arm:
    open(os.path.join(out, "sft.jsonl"), "a").write(json.dumps({"event": "start", "updates": 400}) + "\n"); sys.exit(1)
ups = int(os.environ.get("FAKE_UPDATES", "400"))
L = [data, {"event": "start", "updates": ups}]
for u in (50, 100):
    r = {"event": "train", "update": u, "bceA": 0.5, "lossB": 0.4}
    if head: r["aux_bce"] = 0.69
    L.append(r)
for tag in ("step0", "update400"):
    L.append({"event": "eval", "tag": tag, "n": 3137})
    if head: L.append({"event": "aux_eval", "tag": tag, "iou": 0.5})
L.append({"event": "done", "updates": ups})
with open(os.path.join(out, "sft.jsonl"), "a") as f:
    for r in L: f.write(json.dumps(r) + "\n")
open(os.path.join(out, "model_sft_B.pt"), "wb").write(b"B" * 4096)
os.makedirs(os.path.join(out, "dump"), exist_ok=True)
for tag in ("step0", "update400"):
    for k in ("base", "A", "B"): open(os.path.join(out, "dump", "scores_%s_%s.npz" % (tag, k)), "wb").write(b"s")
    if head: open(os.path.join(out, "dump", "head_pred_%s.npz" % tag), "wb").write(b"h")
if head: open(os.path.join(out, "aux_drv.pt"), "wb").write(b"aux")
open(os.path.join(out, "scorer_states_last.pt"), "wb").write(b"states")
print("ZZSFT_DONE")
'''
STUB_GATE = r'''
import hashlib, json, os, sys
a = sys.argv[1:]
def val(f): return a[a.index(f) + 1] if f in a else None
cmd = a[0]
open(os.environ["STUB_LOG"], "a").write(json.dumps({"tool": "launch_gate", "cmd": cmd, "arm": val("--arm-name"), "profile": val("--profile"), "args": val("--args"), "hf_token": "HF_TOKEN" in os.environ}) + "\n")
if cmd == "run":
    arm = val("--arm-name")
    if os.environ.get("FAKE_GATE_REFUSE_ARM") == arm:
        print("ZZGATE-REFUSEDZZ"); sys.exit(1)
    if not os.path.isfile(val("--gate-preflight-json")): print("no preflight"); sys.exit(1)
    os.makedirs(val("--out-dir"), exist_ok=True)
    tok = {"kind": "launch-gate-token", "verdict": "PASS", "key_id": "kk", "created_utc": "now", "argv": val("--args"),
           "binding": {"profile": val("--profile"), "commit": "c" * 40, "tree_sha256": "t" * 64, "argv_sha256": hashlib.sha256(val("--args").encode()).hexdigest()}}
    json.dump(tok, open(os.path.join(val("--out-dir"), "PASS_%s.json" % ("%x" % abs(hash(arm)))[:8]), "w"))
    print("ZZGATE-PASS-ccccccccccccZZ"); sys.exit(0)
tok = json.load(open(val("--token")))
good = tok.get("argv") == val("--args") and val("--profile") == "refe_aux_s1"
if val("--json"): json.dump({"verdict": "MATCH" if good else "REFUSED"}, open(val("--json"), "w"))
print("ZZGATE-MATCHZZ" if good else "ZZGATE-REFUSEDZZ"); sys.exit(0 if good else 5)
'''
STUB_EGRESS = r'''
import json, os, shutil, sys, time
a = sys.argv[1:]
def val(f): return a[a.index(f) + 1] if f in a else None
def vals(f):
    i = a.index(f) + 1; out = []
    while i < len(a) and not a[i].startswith("--"): out.append(a[i]); i += 1
    return out
tag, repo, prefix, cl = val("--tag"), val("--repo"), val("--prefix") or "", val("--chain-log")
files = vals("--files"); names = vals("--names") if "--names" in a else [os.path.basename(f) for f in files]
open(os.environ["STUB_LOG"], "a").write(json.dumps({"tool": "hf_egress", "tag": tag, "repo": repo, "prefix": prefix, "names": names, "hf_token": "HF_TOKEN" in os.environ}) + "\n")
ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
if "HF_TOKEN" not in os.environ or os.environ.get("FAKE_PUSH_FAIL_TAG") == tag:
    open(cl, "a").write("%s ZZEGRESS_FAIL %s no_token_or_forced\n" % (ts, tag)); sys.exit(1)
hub = os.path.join(os.environ["FAKEHUB"], repo.replace("/", "_"), prefix.strip("/"))
for f, n in zip(files, names):
    dst = os.path.join(hub, n); os.makedirs(os.path.dirname(dst), exist_ok=True); shutil.copyfile(f, dst)
if os.environ.get("FAKE_PUSH_CORRUPT_TAG") == tag:
    with open(os.path.join(hub, names[0]), "ab") as fh: fh.write(b"x")
open(cl, "a").write("%s ZZEGRESS_OK %s %d files 0 B\n" % (ts, tag, len(files)))
sys.exit(0)
'''
STUB_PDM = r'''
import json, os, sys
a = sys.argv[1:]
def val(f): return a[a.index(f) + 1] if f in a else None
open(os.environ["STUB_LOG"], "a").write(json.dumps({"tool": "relabel_pdm", "sets": val("--sets"), "out": val("--out"), "served_only": "--served-only" in a, "hf_token": "HF_TOKEN" in os.environ}) + "\n")
os.makedirs(val("--out"), exist_ok=True)
n = int(os.environ.get("FAKE_PDM_ROWS", "3137"))
with open(os.path.join(val("--out"), "pdm_r0.jsonl"), "a") as f:
    for i in range(n): f.write('{"k": %d}\n' % i)
print("ZZPDM_DONE")
'''
STUB_PDM_COMPLETE = r'''
import glob, os, sys
a = sys.argv[1:]
def val(f): return a[a.index(f) + 1] if f in a else None
if os.environ.get("FAKE_PDM_COMPLETE_LIES") == "1":
    print("ZZPDM_COMPLETE 0"); sys.exit(0)
n = sum(sum(1 for _ in open(p)) for p in glob.glob(os.path.join(val("--out"), "*.jsonl")))
exp = int(val("--expect") or 0)
if val("--json"): open(val("--json"), "w").write('{"rows": %d, "complete": %s}' % (n, "true" if n >= exp else "false"))
if n >= exp: print("ZZPDM_COMPLETE %d" % n); sys.exit(0)
print("ZZPDM_INCOMPLETE %d < %d" % (n, exp)); sys.exit(1)
'''
STUB_SUBSET = r'''
import json, os, sys
a = sys.argv[1:]
def val(f): return a[a.index(f) + 1] if f in a else None
open(os.environ["STUB_LOG"], "a").write(json.dumps({"tool": "select_bank_subset", "queue": val("--queue"), "n": val("--n"), "seed": val("--seed")}) + "\n")
out = val("--out"); os.makedirs(out, exist_ok=True)
for r in (0, 1): open(os.path.join(out, "props_r%d_subset.jsonl" % r), "w").write('{"teacher": 1, "props": 1}\n')
json.dump({"keys": []}, open(os.path.join(out, "subset_keys.json"), "w"))
n = int(os.environ.get("FAKE_FOUND", val("--n")))
print("ZZSUBSET served 100000 picked %s queue_rows_found %d" % (val("--n"), n))
'''
STUB_TLANE = r'''
import json, os, sys
a = sys.argv[1:]
def val(f): return a[a.index(f) + 1] if f in a else None
rk = int(val("--rank"))
open(os.environ["STUB_LOG"], "a").write(json.dumps({"tool": "relabel_tlane", "rank": rk, "workers": val("--workers"), "repair": "--repair-last-heading" in a, "hf_token": "HF_TOKEN" in os.environ}) + "\n")
os.makedirs(val("--out"), exist_ok=True)
n = int(os.environ.get("FAKE_TL_ROWS", "1200"))
with open(os.path.join(val("--out"), "teacher_lane_r%d.jsonl" % rk), "a") as f:
    for i in range(n): f.write('{"kind": "teacher_lane_labels", "i": %d}\n' % i)
print("rank %d: %d samples" % (rk, n))
if os.environ.get("FAKE_TL_NO_DONE") != "1": print("ZZTEACHER_LANE_DONE %d 0" % n)
'''
SHIM = r'''
import hashlib, os, subprocess, sys
a = sys.argv[1:]
def val(f): return a[a.index(f) + 1] if f in a else None
def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
if a[:2] == ["s1aux_checks.py", "fetch"]:
    fx = os.path.join(os.environ["FIXDIR"], os.path.basename(val("--path")))
    if not os.path.isfile(fx): print("ZZS1AUX_FETCH_REFUSED no fixture"); sys.exit(1)
    dst = os.path.join(val("--local-dir"), val("--path")); os.makedirs(os.path.dirname(dst), exist_ok=True)
    open(dst, "wb").write(open(fx, "rb").read()); print("ZZS1AUX_FETCH_OK", dst); sys.exit(0)
if a[:2] == ["s1aux_checks.py", "remote-verify"]:
    i = a.index("--files") + 1; files = []
    while i < len(a) and not a[i].startswith("--"): files.append(a[i]); i += 1
    names = []
    if "--names" in a:
        i = a.index("--names") + 1
        while i < len(a) and not a[i].startswith("--"): names.append(a[i]); i += 1
    names = names or [os.path.basename(f) for f in files]
    hub = os.path.join(os.environ["FAKEHUB"], val("--repo").replace("/", "_"), (val("--prefix") or "").strip("/"))
    bad = [n for f, n in zip(files, names) if not os.path.isfile(os.path.join(hub, n)) or sha(os.path.join(hub, n)) != sha(f)]
    print("ZZS1AUX_REMOTE_OK %d files" % len(files) if not bad else "ZZS1AUX_REMOTE_REFUSED " + ",".join(bad)); sys.exit(1 if bad else 0)
if a[:1] == ["-c"] and "navsim_pdm_targets" in a[1]:
    if os.environ.get("FAKE_PDM_SETUP_FAIL") == "1": print("ModuleNotFoundError: navsim"); sys.exit(1)
    print("ZZPDM_SETUP_OK"); sys.exit(0)
sys.exit(subprocess.call([sys.executable] + a))
'''


class Sandbox:
    """A temp WORK with the pod's inputs in miniature, the REAL helper, the REAL launch-gate-free stubs for everything heavy, and the REAL chain text (WORK and the fixture pins substituted)."""

    def __init__(self, tmp, text, real_store=True):
        self.tmp, self.W = Path(tmp), Path(tmp) / "work"
        W = self.W
        shutil.rmtree(W, ignore_errors=True)
        self.R = W / "data" / "s1aux"
        self.pk = W / "refe-plan-s1aux"
        self.fix = self.tmp / "fix"
        self.hub = self.tmp / "hub"
        self.stub_log = self.tmp / "stub.jsonl"
        for d in (W / "data" / "refe_p2", W / "data" / "backbones", self.pk / "refe", self.pk / "code", self.pk / "splits", self.fix, self.hub, self.tmp / "bin"):
            d.mkdir(parents=True, exist_ok=True)
        self.u400 = b"fake u400 checkpoint " * 8
        self.bad_u400 = b"the WRONG u400 checkpoint " * 8
        (self.fix / "model_sft4_B_u400.pt").write_bytes(self.u400)
        t = text.replace("WORK=/workspace", f"WORK={posix(W)}", 1).replace("sleep 1800 200>&-", "sleep 1 200>&-")
        a = t
        t = re.sub(r"U400_SIZE=\d+; U400_MD5=[0-9a-f]+; U400_SHA256=[0-9a-f]+",
                   f"U400_SIZE={len(self.u400)}; U400_MD5={hashlib.md5(self.u400).hexdigest()}; U400_SHA256={hashlib.sha256(self.u400).hexdigest()}", t)
        assert t != a
        self.store = REAL_STORE if (real_store and (REAL_STORE / "manifest.json").is_file()) else W / "data" / "aux_drivable" / "label_store"
        b = t
        t = t.replace("LABELS=$DATA/aux_drivable/label_store", f"LABELS={posix(self.store)}", 1)
        assert t != b
        self.text = t
        script = self.tmp / "s1aux_sandbox.sh"
        write_sh(script, t)
        self.script = script
        # the package: every file the chain requires; the REAL helper (and what it imports) + stubs for the heavy children; then the manifest over the whole tree
        files = re.search(r'for f in (.*?); do\n  \[ -f "\$PKG/\$f" \]', t, re.S).group(1).replace("\\\n", " ").split()
        self.shiplist = files
        for f in files:
            (self.pk / f).parent.mkdir(parents=True, exist_ok=True)
            if not (self.pk / f).exists():
                (self.pk / f).write_text("", encoding="utf-8")
        for real in ("aux_drivable_head.py", "aux_label_store.py", "field_loss.py", "model.py", "gate_token.py", "p2r_sync.py"):
            shutil.copy(HERE / real, self.pk / "refe" / real)
        h = (HERE / "s1aux_checks.py").read_text(encoding="utf-8")
        h2 = h.replace("MODEL_B_MIN_BYTES = 1_200_000_000", "MODEL_B_MIN_BYTES = 1000")                  # the sandbox's model files are 4 KB, not 1.29 GB (the real constant is asserted elsewhere)
        assert h2 != h
        (self.pk / "refe" / "s1aux_checks.py").write_text(h2, encoding="utf-8", newline="\n")
        for name, src in (("scorer_finetune.py", STUB_SCORER), ("launch_gate.py", STUB_GATE), ("hf_egress.py", STUB_EGRESS), ("onpolicy_relabel_pdm.py", STUB_PDM),
                          ("pdm_label_complete.py", STUB_PDM_COMPLETE), ("select_bank_subset.py", STUB_SUBSET), ("onpolicy_relabel_teacher_lane.py", STUB_TLANE)):
            (self.pk / "refe" / name).write_text(src, encoding="utf-8", newline="\n")
        for xz in ("heldout_navsim_lane_labels.jsonl.xz", "heldout_teacher_lane_labels.jsonl.xz"):
            shutil.copy(PKGROOT / "raw" / "2026-10-04-lane-discipline" / xz, self.pk / "raw" / "2026-10-04-lane-discipline" / xz)
        man = GT.ship_manifest(self.pk)
        GT.write_json(self.pk / "SHIP_MANIFEST.json", man)
        # pod state
        (W / "data" / "rls1").mkdir(parents=True, exist_ok=True)
        (W / "data" / "rls1" / "chain.log").write_text("2026-10-14T00:00:00 ZZRLS1_DONE arms: R0 R1 R2\n")
        fakepy = self.tmp / "bin" / "fakepy.sh"
        shim = self.tmp / "bin" / "shim.py"
        shim.write_text(SHIM, encoding="utf-8", newline="\n")
        write_sh(fakepy, f'#!/bin/bash\nexec "{posix(sys.executable)}" "{posix(shim)}" "$@"\n')
        (W / "teacher_env.sh").write_text(f'export DRIVERL_EVAL_PYTHON="{posix(fakepy)}"\n')
        write_sh(self.tmp / "bin" / "flock", '#!/bin/bash\nif [ "$2" = 200 ]; then exit "${FAKE_FLOCK200_RC:-0}"; fi\nexit "${FAKE_FLOCK_RC:-0}"\n')
        write_sh(self.tmp / "bin" / "nvidia-smi", '#!/bin/bash\nn=$(cat "$NVCOUNT" 2>/dev/null || echo 0)\necho $((n + 1)) > "$NVCOUNT"\nif [ "$n" -lt "${NV_BUSY_CALLS:-0}" ]; then echo 4242; fi\nexit 0\n')
        d = W / "data"
        (d / "aux_drivable").mkdir(parents=True, exist_ok=True)
        shutil.copy(NAVTEST, d / "aux_drivable" / "navtest.json")
        (d / "navtrain_dbs").mkdir(parents=True, exist_ok=True)
        for i in range(1192):
            (d / "navtrain_dbs" / f"{i}.db").write_bytes(b"")
        (d / "refe_navtrain" / "train_grow").mkdir(parents=True, exist_ok=True)
        (d / "refe_navtrain" / "train_grow" / "targets_rank0.jsonl").write_text("{}\n")
        (d / "refe_navtrain" / "train_grow" / "calib_table.json").write_text("{}")
        (d / "refe_navtrain" / "onpolicy" / "sets").mkdir(parents=True, exist_ok=True)
        (d / "refe_navtrain" / "onpolicy" / "sets" / "onpolicy_r0.jsonl").write_text("{}\n")
        for i in range(32):
            (d / "navtrain_pixels" / f"navtrain_current_{i:02d}").mkdir(parents=True, exist_ok=True)
        (d / "refe_sft2" / "train_pdm").mkdir(parents=True, exist_ok=True)
        (d / "refe_sft2" / "train_pdm" / "pdm_r0.jsonl").write_text("{}\n")
        (d / "refe_heldout" / "sets").mkdir(parents=True, exist_ok=True)
        for i in range(7):
            (d / "refe_heldout" / "sets" / f"onpolicy_r{i}.jsonl").write_text("{}\n")
        (d / "refe_heldout" / "heldout_logs.txt").write_text(" ".join(f"log{i}" for i in range(24)) + "\n")
        self.qsrc = self.tmp / "queue_src"
        self.qsrc.mkdir(exist_ok=True)
        (self.qsrc / "props_r0_a.jsonl").write_text("{}\n")
        self.base_env = {"PATH": posix(self.tmp / "bin") + os.pathsep + os.environ["PATH"], "HF_TOKEN": "hf_notarealtoken", "HF_HUB_OFFLINE": "1", "S1AUX_DD_MB": "4", "S1AUX_TICK": "0",
                         "S1AUX_POLL_S": "0", "S1AUX_LABEL_WORKERS": "2", "STUB_LOG": posix(self.stub_log), "FAKEHUB": posix(self.hub), "FIXDIR": posix(self.fix), "NVCOUNT": posix(self.tmp / "nvc"),
                         "LANE_QUEUE_SRC": posix(self.qsrc), "PYTHONIOENCODING": "utf-8"}

    def reset(self):
        """a pod that has not run the chain: its work dir, the Hub, the stub log, and the labels the chain regenerates / restores (the P2 inputs stay)"""
        shutil.rmtree(self.R, ignore_errors=True)
        shutil.rmtree(self.hub, ignore_errors=True)
        self.hub.mkdir()
        for d in ("heldout_pdm", "train_teacher_lane", "train_subset_queue", "heldout_lane", "heldout_teacher_lane"):
            shutil.rmtree(self.W / "data" / "refe_sft2" / d, ignore_errors=True)
        if self.stub_log.exists():
            self.stub_log.unlink()

    def run(self, env_extra=None, drop=(), fresh=False, timeout=600):
        if fresh:
            self.reset()
        (self.tmp / "nvc").write_text("0")
        e = {k: v for k, v in os.environ.items() if k not in drop}
        e.update({k: v for k, v in {**self.base_env, **(env_extra or {})}.items() if k not in drop})
        r = subprocess.run([shutil.which("bash"), posix(self.script)], capture_output=True, text=True, env=e, timeout=timeout)
        return r.returncode, r.stdout + r.stderr, self.log()

    def log(self):
        p = self.R / "chain.log"
        return p.read_text(encoding="utf-8") if p.exists() else ""

    def calls(self, tool=None):
        if not self.stub_log.exists():
            return []
        rows = [json.loads(l) for l in self.stub_log.read_text(encoding="utf-8").splitlines() if l.strip()]
        return [r for r in rows if tool is None or r["tool"] == tool]

    def marks(self):
        d = self.R / "stage"
        return sorted(p.stem for p in d.glob("*.ok")) if d.exists() else []


def sandbox_runs(text, tmp):
    if os.name == "nt" and subprocess.run([shutil.which("bash"), "-c", "exec 200>/dev/null"], capture_output=True).returncode != 0:
        print("ZZTEST_SKIP sandbox_runs: this bash cannot open fd 200")
        return
    if not NAVTEST.is_file():           # TEST-RED-1: a tree exported WITHOUT the repo's FlyWheels/ (or a shallow pod tree) has no navtest list; Sandbox() copies it unconditionally -> FileNotFoundError at line 733
        print("ZZTEST_SKIP sandbox_runs: the repo's navtest list is absent on this machine (this tree has no FlyWheels/TanitAD_EvalFlyWheel/...); the sandbox copies it, so no sandbox case is run")
        return
    sb = Sandbox(tmp, text)
    have_store = sb.store == REAL_STORE
    if not have_store:
        print("ZZTEST_SKIP sandbox_runs: the real aux label store is absent on this box; the stages past `data` are not run")

    def refused(out, lg, stage, frag=""):
        ok(f"ZZS1AUX_REFUSED stage={stage}:" in lg and frag in lg, (stage, frag, lg[-700:]))
    # ---- precheck refusals
    rc, out, lg = sb.run(fresh=True, env_extra={"FAKE_FLOCK200_RC": "1"}); ok(rc == 1 and "ZZS1AUX_REFUSED another s1aux_pod.sh holds" in out, out)         # a second chain
    rc, out, lg = sb.run(fresh=True, drop=("HF_TOKEN",)); ok(rc == 1, out); refused(out, lg, "precheck", "HF_TOKEN is not set")
    gone = sb.pk / "refe" / "test_launch_gate.py"
    gone.rename(gone.with_suffix(".bak"))
    rc, out, lg = sb.run(fresh=True); ok(rc == 1, out); refused(out, lg, "precheck", "test_launch_gate.py missing")
    gone.with_suffix(".bak").rename(gone)
    te = sb.W / "teacher_env.sh"
    te.rename(te.with_suffix(".bak"))
    rc, out, lg = sb.run(fresh=True); ok(rc == 1, out); refused(out, lg, "precheck", "teacher_env.sh is missing")
    te.with_suffix(".bak").rename(te)
    # a drifted file: the tree is no longer its manifest
    ft = sb.pk / "refe" / "route_rank.py"
    ft.write_text("# edited after the manifest\n")
    rc, out, lg = sb.run(fresh=True); ok(rc == 1, out); refused(out, lg, "precheck", "not its SHIP_MANIFEST.json")
    ft.write_text("")
    # RL-S1 not over
    rl = sb.W / "data" / "rls1"
    (rl / "chain.log").write_text("ZZRLS1_ARM_DONE R0\n")
    rc, out, lg = sb.run(fresh=True); ok(rc == 1, out); refused(out, lg, "precheck", "neither running nor finished")
    rc, out, lg = sb.run(fresh=True, env_extra={"S1AUX_RLS1_ABANDONED": "1", "S1AUX_STOP_AFTER": "precheck"}); ok(rc == 0 and "ZZS1AUX_PRECHECK_OK" in lg, lg)
    (rl / "chain.log").write_text("ZZRLS1_DONE arms: R0 R1 R2\n")
    rc, out, lg = sb.run(fresh=True, env_extra={"S1AUX_ARMS": "H2-s0 N-s0"}); ok(rc == 1, out); refused(out, lg, "precheck", "out of the registered priority order")
    # RL-S1 still running: the CPU stages proceed beside it (A4): precheck passes WITHOUT waiting and WITHOUT polling the GPU, and names the running upstream
    (rl / "chain.lock").write_text("")
    (sb.tmp / "nvc").write_text("0")
    rc, out, lg = sb.run(fresh=True, env_extra={"FAKE_FLOCK_RC": "1", "S1AUX_STOP_AFTER": "precheck"})
    ok(rc == 0 and "ZZS1AUX_PRECHECK_OK" in lg and "upstream: the RL-S1 chain still holds" in lg and "ZZS1AUX_WAITING" not in lg, lg)
    ok(int((sb.tmp / "nvc").read_text() or 0) == 0, "precheck must not poll the GPU")
    (rl / "chain.lock").unlink()
    # ---- fetch: a WRONG u400 stops the chain at fetch and is deleted; the right one passes
    (sb.fix / "model_sft4_B_u400.pt").write_bytes(sb.bad_u400)
    rc, out, lg = sb.run(fresh=True, env_extra={"S1AUX_STOP_AFTER": "fetch"}); ok(rc == 1, out); refused(out, lg, "fetch", "is not the pinned file")
    ok(not (sb.R / "base" / "model_sft4_B_u400.pt").exists(), "a wrong u400 must not stay at the path the argv points to")
    ok("data" not in sb.marks() and not sb.calls("relabel_pdm"))
    (sb.fix / "model_sft4_B_u400.pt").unlink()
    rc, out, lg = sb.run(fresh=True, env_extra={"S1AUX_STOP_AFTER": "fetch"}); ok(rc == 1, out); refused(out, lg, "fetch", "is not the pinned file")           # no fixture: the fetch fails, same refusal
    (sb.fix / "model_sft4_B_u400.pt").write_bytes(sb.u400)
    rc, out, lg = sb.run(fresh=True, env_extra={"S1AUX_STOP_AFTER": "fetch"}); ok(rc == 0 and "ZZS1AUX_FETCH_OK u400" in lg, lg)
    (sb.fix / "model_sft4_B_u400.pt").unlink()                                                                         # re-run: the verified file is reused (no fetch needed)
    rc, out, lg = sb.run(env_extra={"S1AUX_STOP_AFTER": "fetch"}); ok(rc == 0, lg)
    (sb.R / "base" / "model_sft4_B_u400.pt").write_bytes(b"corrupted on the pod")                                      # a corrupted copy is noticed on the next start
    rc, out, lg = sb.run(env_extra={"S1AUX_STOP_AFTER": "fetch"}); ok(rc == 1, lg); refused(out, lg, "fetch")
    (sb.fix / "model_sft4_B_u400.pt").write_bytes(sb.u400)
    # ---- data: the store
    if have_store:
        nt = sb.W / "data" / "aux_drivable" / "navtest.json"
        nt.unlink()
        rc, out, lg = sb.run(fresh=True, env_extra={"S1AUX_STOP_AFTER": "data"}); ok(rc == 1, out); refused(out, lg, "data", "navtest list")
        nt.write_text('{"token_to_log_name": {}}')                                                                  # present, but not the pinned content
        rc, out, lg = sb.run(fresh=True, env_extra={"S1AUX_STOP_AFTER": "data"}); ok(rc == 1, out); refused(out, lg, "data", "navtest list")
        shutil.copy(NAVTEST, nt)
        rc, out, lg = sb.run(fresh=True, env_extra={"S1AUX_STOP_AFTER": "data"}); ok(rc == 0 and "ZZS1AUX_DATA_OK" in lg and "held-out lane + teacher lane 3137 rows each" in lg, lg)
        lane = sb.W / "data" / "refe_sft2" / "heldout_teacher_lane"
        ok(len(list(lane.glob("*.jsonl"))) == 1)                                                                       # restored from the repo's .xz: 3,137 rows each (the labels SFT-4 trained against)
    # a store that is not the pinned one: the manifest alone, no shards
    bogus = sb.tmp / "bogus_store"
    bogus.mkdir(exist_ok=True)
    if have_store:
        shutil.copy(REAL_STORE / "manifest.json", bogus / "manifest.json")
        text_b = sb.text.replace(f"LABELS={posix(REAL_STORE)}", f"LABELS={posix(bogus)}")
        script_b = sb.script
        sb.script = sb.tmp / "s1aux_sandbox_bogus.sh"
        write_sh(sb.script, text_b)
        rc, out, lg = sb.run(fresh=True, env_extra={"S1AUX_STOP_AFTER": "data"}); ok(rc == 1, out); refused(out, lg, "data", "fails its own sha check")
        (bogus / "manifest.json").write_text("{}")
        rc, out, lg = sb.run(fresh=True, env_extra={"S1AUX_STOP_AFTER": "data"}); ok(rc == 1, out); refused(out, lg, "data", "fails its own sha check")
        shutil.rmtree(bogus)
        rc, out, lg = sb.run(fresh=True, env_extra={"S1AUX_STOP_AFTER": "data"}); ok(rc == 1, out); refused(out, lg, "data", "aux label store is absent")
        sb.script = script_b
        # held-out lane labels with the wrong number of rows
        sb.reset()
        rc, out, lg = sb.run(env_extra={"S1AUX_STOP_AFTER": "data"}); ok(rc == 0, lg)                                  # restores the lane labels from the repo
        lane = sb.W / "data" / "refe_sft2" / "heldout_lane"
        next(lane.glob("*.jsonl")).write_text("{}\n")
        rc, out, lg = sb.run(env_extra={"S1AUX_STOP_AFTER": "data"}); ok(rc == 1, out); refused(out, lg, "data", "want 3137")
        # the held-out bank missing and not fetchable
        ho = sb.W / "data" / "refe_heldout"
        hold = sb.tmp / "ho_saved"
        shutil.move(str(ho), str(hold))
        rc, out, lg = sb.run(fresh=True, env_extra={"S1AUX_STOP_AFTER": "data"}); ok(rc == 1, out); refused(out, lg, "data", "is not the pinned refe_heldout_v5.tar.zst")
        shutil.move(str(hold), str(ho))
        # P2's inputs gone
        px = sb.W / "data" / "navtrain_pixels"
        hold = sb.tmp / "px_saved"
        shutil.move(str(px), str(hold))
        rc, out, lg = sb.run(fresh=True, env_extra={"S1AUX_STOP_AFTER": "data"}); ok(rc == 1, out); refused(out, lg, "data", "navtrain_pixels is missing")
        shutil.move(str(hold), str(px))
    if not have_store:
        return
    # ---- relabel: an empty held-out PDM stops the chain at relabel, and so does every other labelling failure
    E = {"S1AUX_STOP_AFTER": "relabel"}
    rc, out, lg = sb.run(fresh=True, env_extra={**E, "FAKE_PDM_ROWS": "0"}); ok(rc == 1, out); refused(out, lg, "relabel", "incomplete after 3 attempts")
    ok(len(sb.calls("relabel_pdm")) == 3 and "pdm_heldout" not in sb.marks() and not sb.calls("relabel_tlane"), sb.marks())                  # 3 attempts, no marker, the teacher-lane labeller never started
    rc, out, lg = sb.run(fresh=True, env_extra={**E, "FAKE_PDM_ROWS": "0", "FAKE_PDM_COMPLETE_LIES": "1"}); ok(rc == 1, out); refused(out, lg, "relabel", "heldout_pdm holds 0 rows")   # a completeness stub that lies: the chain counts rows itself
    rc, out, lg = sb.run(fresh=True, env_extra={**E, "FAKE_PDM_SETUP_FAIL": "1"}); ok(rc == 1, out); refused(out, lg, "relabel", "the labellers cannot start")
    rc, out, lg = sb.run(fresh=True, env_extra={**E, "FAKE_FOUND": "100"}); ok(rc == 1, out); refused(out, lg, "relabel", "covers 100 of the picked subset")
    ok(not (sb.W / "data" / "refe_sft2" / "train_subset_queue").exists(), "a too-small subset must not stay as the queue")
    rc, out, lg = sb.run(fresh=True, env_extra={**E, "FAKE_TL_ROWS": "100"}); ok(rc == 1, out); refused(out, lg, "relabel", "train teacher-lane rows 200 < 2109")
    rc, out, lg = sb.run(fresh=True, env_extra={**E, "FAKE_TL_NO_DONE": "1"}); ok(rc == 1, out); refused(out, lg, "relabel", "did not run to its end")
    rc, out, lg = sb.run(fresh=True, env_extra={**E, "LANE_QUEUE_SRC": ""}); ok(rc == 1, out); refused(out, lg, "relabel")                               # no queue source and nothing to fetch it from
    rc, out, lg = sb.run(fresh=True, env_extra=E); ok(rc == 0 and "ZZS1AUX_RELABEL_OK" in lg and "train teacher lane 2400 rows" in lg, lg)
    sub = sb.calls("select_bank_subset")
    ok(len(sub) == 1 and sub[0]["n"] == "15000" and sub[0]["seed"] == "20261004" and sub[0]["queue"] == posix(sb.qsrc), sub)                        # SFT-4R's subset, from the named queue
    ok(sorted(c["rank"] for c in sb.calls("relabel_tlane")) == [0, 1])
    ok(all(c["repair"] for c in sb.calls("relabel_tlane")) and all(c["served_only"] for c in sb.calls("relabel_pdm")))
    ok(not any(c["hf_token"] for c in sb.calls() if c["tool"] in ("relabel_pdm", "relabel_tlane")), "the labellers must not see HF_TOKEN")
    # ---- bank_labels: the regenerated labels must reach the Hub, proven by content
    rc, out, lg = sb.run(env_extra={"S1AUX_STOP_AFTER": "bank_labels", "FAKE_PUSH_FAIL_TAG": "s1_labels"}); ok(rc == 1, out); refused(out, lg, "bank_labels", "were not banked")
    rc, out, lg = sb.run(env_extra={"S1AUX_STOP_AFTER": "bank_labels", "FAKE_PUSH_CORRUPT_TAG": "s1_labels"}); ok(rc == 1, out); refused(out, lg, "bank_labels", "were not banked")   # pushed, but the content differs
    rc, out, lg = sb.run(env_extra={"S1AUX_STOP_AFTER": "bank_labels"}); ok(rc == 0 and "ZZS1AUX_LABELS_BANKED" in lg, lg)
    seg = lg[lg.rfind("ZZS1AUX_START"):]
    ok("ZZS1AUX_GPU_CLEAR" not in seg and "ZZS1AUX_WAITING" not in seg, "the GPU wait and the dd test must come AFTER the CPU stages")
    # ---- gpu_wait: RL-S1 still running blocks the FIRST GPU use (bounded here); a busy GPU waits, then proceeds after 3 consecutive idle polls
    (rl / "chain.lock").write_text("")
    rc, out, lg = sb.run(env_extra={"FAKE_FLOCK_RC": "1", "S1AUX_POLL_S": "1", "S1AUX_WAIT_MAX_S": "2", "S1AUX_STOP_AFTER": "gpu_wait"})
    ok(rc == 1 and "ZZS1AUX_WAITING the RL-S1 chain still holds" in lg, lg); refused(out, lg, "gpu_wait", "waited")
    (rl / "chain.lock").unlink()
    (sb.tmp / "nvc").write_text("0")
    rc, out, lg = sb.run(env_extra={"NV_BUSY_CALLS": "3", "S1AUX_STOP_AFTER": "gpu_wait"})
    seg = lg[lg.rfind("ZZS1AUX_START"):]                                                                              # THIS run only: chain.log accumulates across runs
    ok(rc == 0 and seg.count("ZZS1AUX_WAITING") == 3 and "ZZS1AUX_GPU_CLEAR" in seg, seg)
    ok(int((sb.tmp / "nvc").read_text()) == 6)                                                                      # 3 busy + 3 consecutive idle polls
    hub = sb.hub / "Sayood_tanitad-refe-s1aux" / "labels"
    ok(all((hub / n).is_file() for n in ("s1_heldout_pdm.tar", "s1_train_teacher_lane.tar", "s1_heldout_lane.tar", "s1_heldout_teacher_lane.tar", "MANIFEST_s1_labels_sha256.txt", "pdm_complete_heldout.json", "subset_keys.json")))
    # ---- arms.  A failing preflight / data check / gate leaves the trainer UNSTARTED and the next arm UNTOUCHED
    E2 = {"S1AUX_ARMS": "N-s0 H2-s0 H1-s0"}
    rc, out, lg = sb.run(env_extra={**E2, "FAKE_PREFLIGHT_FAIL_ARM": "N-s0"}); ok(rc == 1, out); refused(out, lg, "arm_N-s0", "--preflight did not pass")
    ok(not [c for c in sb.calls("scorer_finetune") if not c["preflight"]] and not (sb.R / "runs" / "H2-s0").exists() and not sb.calls("launch_gate"))
    rc, out, lg = sb.run(env_extra={**E2, "FAKE_BAD_DATA_ARM": "N-s0"}); ok(rc == 1, out); refused(out, lg, "arm_N-s0", "is not the audited data")
    ok(not [c for c in sb.calls("scorer_finetune") if not c["preflight"]] and not sb.calls("launch_gate"))
    rc, out, lg = sb.run(env_extra={**E2, "FAKE_GATE_REFUSE_ARM": "N-s0"}); ok(rc == 1, out); refused(out, lg, "arm_N-s0", "did not mint a PASS token")
    ok(not [c for c in sb.calls("scorer_finetune") if not c["preflight"]], "an arm without a PASS token must not train")                      # the trainer is never started
    ok(not (sb.R / "arms_manifest.jsonl").exists() or "token_verified" not in (sb.R / "arms_manifest.jsonl").read_text())
    ok(not (sb.R / "runs" / "H2-s0").exists(), "the next arm must not be touched")
    # ---- the happy path of two arms, in order
    sb.run(fresh=True, env_extra={"S1AUX_STOP_AFTER": "bank_labels"})
    rc, out, lg = sb.run(env_extra={"S1AUX_ARMS": "N-s0 H2-s0"}); ok(rc == 0 and "ZZS1AUX_DONE arms: N-s0 H2-s0" in lg, (rc, lg[-900:], out[-400:]))
    i_done = [lg.index(f"ZZS1AUX_ARM_DONE {a}") for a in ("N-s0", "H2-s0")]
    ok(i_done == sorted(i_done) and lg.index("ZZS1AUX_RELABEL_OK") < lg.index("ZZS1AUX_ARM_GATE_OK N-s0") < lg.index("ZZS1AUX_ARM_TRAIN_START N-s0") < lg.index("ZZS1AUX_ARM_DONE N-s0")
       < lg.index("ZZS1AUX_ARM_GATE_OK H2-s0"), lg)
    P = resolved_vars(sb.text, ("OUT",))
    sc = [c for c in sb.calls("scorer_finetune")]
    for arm in ("N-s0", "H2-s0"):
        want = shlex.split(arm_args_of(sb.text, arm, sb.tmp)[1])
        pre = [c for c in sc if c["arm"] == arm and c["preflight"]]
        trn = [c for c in sc if c["arm"] == arm and not c["preflight"]]
        gate_run = [c for c in sb.calls("launch_gate") if c["cmd"] == "run" and c["arm"] == arm]
        ok(len(pre) == 1 and len(trn) == 1 and len(gate_run) == 1, (arm, len(pre), len(trn), len(gate_run)))
        ok(pre[0]["argv"] == want + ["--preflight"], (arm, pre[0]["argv"]))                                                  # the preflight runs the SAME argv plus --preflight
        ok(trn[0]["argv"] == want, (arm, trn[0]["argv"]))                                                                    # the trainer runs the registered argv, token for token
        ok(shlex.split(gate_run[0]["args"]) == want and gate_run[0]["profile"] == "refe_aux_s1", arm)                       # ... which is the argv the gate was minted for
        ok(not pre[0]["hf_token"] and not trn[0]["hf_token"] and not gate_run[0]["hf_token"], "HF_TOKEN must not reach the trainer, the preflight or the gate")
        ok(trn[0]["token_row_before_step"], "the token id must be in the arms manifest BEFORE the arm's first step")
        ok(any(c["cmd"] == "verify" and c["arm"] is None for c in sb.calls("launch_gate")))
        arm_files = sb.hub / "Sayood_tanitad-refe-s1aux" / "arms" / arm
        names = {p.relative_to(arm_files).as_posix() for p in arm_files.rglob("*") if p.is_file()}
        base = {"model_sft_B.pt", "sft.jsonl", "sft.log", "preflight.jsonl", "PREFLIGHT_PASS.json", "launch_gate.log", "run_check.json", "arms_manifest.jsonl", "scorer_states_last.pt",
                "dump/scores_step0_B.npz", "dump/scores_update400_B.npz"}
        ok(base <= names, (arm, sorted(base - names)))
        ok(any(n.startswith("PASS_") for n in names))
        ok(("aux_drv.pt" in names and "dump/head_pred_update400.npz" in names) == arm.startswith("H"), (arm, sorted(names)))      # the head files go up iff it is a head arm
        ok("model_sft_A.pt" not in names)
    man = [json.loads(l) for l in (sb.R / "arms_manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    ok([m["arm"] for m in man if m["event"] == "token_verified"] == ["N-s0", "H2-s0"] and all(m["argv_sha256"] and m["commit"] for m in man if m["event"] == "token_verified"))
    ok(sb.calls("hf_egress")[-1]["tag"] == "chain_exit", "the exit trap pushes chain.log last")
    seq_all = [("pre" if c["preflight"] else "train") if c["tool"] == "scorer_finetune" else c["cmd"] for c in sb.calls() if c["tool"] in ("scorer_finetune", "launch_gate")]
    ok(seq_all == ["pre", "run", "verify", "verify", "train"] * 2, seq_all)                  # raw/2026-10-15-s1-gate-gap: each fresh arm of the happy path is verified twice (after the mint, BINDING before the trainer), not three times
    # a re-run of a finished chain redoes nothing
    n_before = len(sb.calls("scorer_finetune"))
    rc, out, lg = sb.run(env_extra={"S1AUX_ARMS": "N-s0 H2-s0"}); ok(rc == 0 and len(sb.calls("scorer_finetune")) == n_before and "arm N-s0 already done" in lg, lg[-300:])
    # ---- the VERIFY COUNT (raw/2026-10-15-s1-gate-gap): a freshly minted token is re-measured ONCE after the mint and ONCE (BINDING) right before the trainer -- not three times (each
    # verify re-measures argv, code tree and the 413,184-file image listing: 3.4-3.8 min of GPU idle on the pod).  A token minted by an EARLIER invocation is verified before its manifest row
    # is written and again right before the trainer.  The call sequence of one arm is read off the stub log; the same runs on a chain with a verify deleted / re-added must differ (RED).
    def arm_seq(start):
        return [("pre" if c["preflight"] else "train") if c["tool"] == "scorer_finetune" else c["cmd"] for c in sb.calls()[start:] if c["tool"] in ("scorer_finetune", "launch_gate")]

    def reset_arm(arm="N-s0"):                       # the arm's own state only: the labels the chain regenerated (stage bank_labels) stay, so a variant costs one arm run, not a whole chain
        for d in (sb.R / "runs" / arm, sb.R / "lg" / arm):
            shutil.rmtree(d, ignore_errors=True)
        for mk in (sb.R / "stage").glob(f"*_{arm}.ok"):
            mk.unlink()
        if (sb.R / "arms_manifest.jsonl").exists():
            (sb.R / "arms_manifest.jsonl").unlink()

    def variant(script_text, stop=None):
        reset_arm()
        n = len(sb.calls())
        write_sh(sb.script, script_text)
        try:
            res = sb.run(env_extra={"S1AUX_ARMS": "N-s0", **({"S1AUX_STOP_AFTER": stop} if stop else {})})
        finally:
            write_sh(sb.script, sb.text)
        return res, n
    FRESH = ["pre", "run", "verify", "verify", "train"]
    binding = '    token_ok || refuse "the launch-gate token does not verify for $ARM: scorer_finetune.py is NOT started"\n'
    guarded = '  [ "$GATE_MINTED_NOW" = 1 ] || token_ok || refuse "the launch-gate token of $ARM no longer verifies (token_verify.log)"'
    ok(binding in sb.text and guarded in sb.text)
    sb.run(fresh=True, env_extra={"S1AUX_STOP_AFTER": "bank_labels"})
    (rc, out, lg), n = variant(sb.text, stop="gate_N-s0")
    ok(rc == 0 and "ZZS1AUX_ARM_GATE_OK N-s0" in lg and arm_seq(n) == ["pre", "run", "verify"], arm_seq(n))      # a fresh arm up to its manifest row: preflight, the gate mints, ONE verify
    n = len(sb.calls())
    rc, out, lg = sb.run(env_extra={"S1AUX_ARMS": "N-s0"})                                                      # ... the same arm resumed by a later invocation (its gate was passed before)
    ok(rc == 0 and "ZZS1AUX_ARM_DONE N-s0" in lg and arm_seq(n) == ["verify", "verify", "train"], arm_seq(n))   # old token: verified before its row, again (BINDING) before the trainer
    (rc, out, lg), n = variant(sb.text)
    ok(rc == 0 and arm_seq(n) == FRESH, arm_seq(n))                                                             # a fresh arm end to end: pre, run, verify, verify, train
    # RED: the same runs on chains with a verify deleted / re-added differ from the registered sequences
    (rc, out, lg), n = variant(sb.text.replace(binding, ""))
    ok(arm_seq(n) != FRESH and arm_seq(n) == ["pre", "run", "verify", "train"], arm_seq(n))                    # the BINDING verify deleted: the trainer starts after ONE verify
    (rc, out, lg), n = variant(sb.text.replace(guarded, '  token_ok || refuse "the launch-gate token of $ARM no longer verifies (token_verify.log)"'))
    ok(arm_seq(n) != FRESH and arm_seq(n).count("verify") == 3, arm_seq(n))                                     # the redundant verify re-added: three
    no_resumed = sb.text.replace(guarded + "\n", "")
    variant(no_resumed, stop="gate_N-s0")
    n = len(sb.calls())
    write_sh(sb.script, no_resumed)
    try:
        sb.run(env_extra={"S1AUX_ARMS": "N-s0"})
    finally:
        write_sh(sb.script, sb.text)
    ok(arm_seq(n) != ["verify", "verify", "train"] and arm_seq(n) == ["verify", "train"], arm_seq(n))           # the resumed-path verify deleted: a stale token reaches its row unverified (RED)
    # ---- a stage that fails never lets the next run: N-s0 ok, then H2-s0's trainer dies
    rc, out, lg = sb.run(fresh=True, env_extra={"S1AUX_STOP_AFTER": "bank_labels"}); ok(rc == 0, lg)
    rc, out, lg = sb.run(env_extra={**E2, "FAKE_TRAIN_FAIL_ARM": "H2-s0"}); ok(rc == 1, out); refused(out, lg, "arm_H2-s0", "the trainer did not finish")
    ok("ZZS1AUX_ARM_DONE N-s0" in lg and "ZZS1AUX_ARM_DONE H2-s0" not in lg and not (sb.R / "runs" / "H1-s0").exists(), lg[-500:])
    ok(len([c for c in sb.calls("scorer_finetune") if not c["preflight"]]) == 2)                                                   # N-s0 and the failed H2-s0; H1-s0 never started
    ok("trainstart_H2-s0" in sb.marks() and "train_H2-s0" not in sb.marks() and "egress_H2-s0" not in sb.marks())
    n_hub = len(list((sb.hub / "Sayood_tanitad-refe-s1aux" / "arms").glob("H2-s0*")))
    # ---- RESTART: the killed attempt is archived (never stitched), a fresh preflight and a FRESH token follow, the arm then completes
    gates_before = len([c for c in sb.calls("launch_gate") if c["cmd"] == "run" and c["arm"] == "H2-s0"])
    rc, out, lg = sb.run(env_extra=E2); ok(rc == 0 and "ZZS1AUX_ARM_RESTART H2-s0 k=1" in lg and "ZZS1AUX_DONE" in lg, lg[-700:])
    ok((sb.R / "runs" / "H2-s0_killed_1" / "sft.jsonl").is_file() and (sb.R / "runs" / "H2-s0" / "model_sft_B.pt").is_file())
    ok(len([c for c in sb.calls("launch_gate") if c["cmd"] == "run" and c["arm"] == "H2-s0"]) == gates_before + 1, "a fresh token must be minted for the restart")
    kl = (sb.R / "runs" / "H2-s0_killed_1" / "sft.jsonl").read_text(encoding="utf-8")
    nl = (sb.R / "runs" / "H2-s0" / "sft.jsonl").read_text(encoding="utf-8")
    ok(kl.count('"event": "start"') == 1 and nl.count('"event": "start"') == 1 and '"event": "done"' not in kl, "the restarted arm starts from update 0 in a fresh directory (no stitching)")
    ok((sb.R / "lg" / "H2-s0_killed_1").is_dir())
    man = [json.loads(l) for l in (sb.R / "arms_manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    ok(any(m["event"] == "restart" and m["arm"] == "H2-s0" and m["restart_k"] == 1 for m in man))
    ok(len([m for m in man if m["event"] == "token_verified" and m["arm"] == "H2-s0"]) == 2)
    # ---- the run check: a run that did not read updates == 400 is no S1 arm, and its evidence is NOT banked as one
    rc, out, lg = sb.run(fresh=True, env_extra={"S1AUX_STOP_AFTER": "bank_labels"}); ok(rc == 0, lg)
    rc, out, lg = sb.run(env_extra={"S1AUX_ARMS": "N-s0", "FAKE_UPDATES": "300"}); ok(rc == 1, out); refused(out, lg, "arm_N-s0", "is not a valid S1 arm")
    ok(not (sb.hub / "Sayood_tanitad-refe-s1aux" / "arms" / "N-s0").exists(), "an invalid arm must not be banked")
    # ---- egress: pushed but different on the Hub / not pushed -> refused, the arm is NOT marked done
    rc, out, lg = sb.run(fresh=True, env_extra={"S1AUX_STOP_AFTER": "bank_labels"}); ok(rc == 0, lg)
    rc, out, lg = sb.run(env_extra={"S1AUX_ARMS": "N-s0", "FAKE_PUSH_CORRUPT_TAG": "arm_N-s0"}); ok(rc == 1, out); refused(out, lg, "arm_N-s0", "was not banked to HF")
    ok("arm_N-s0" not in sb.marks() and "egress_N-s0" not in sb.marks() and "train_N-s0" in sb.marks())
    rc, out, lg = sb.run(env_extra={"S1AUX_ARMS": "N-s0"}); ok(rc == 0 and "ZZS1AUX_DONE" in lg, lg[-400:])                          # a re-run banks the SAME arm (no retraining)
    ok(len([c for c in sb.calls("scorer_finetune") if not c["preflight"] and c["arm"] == "N-s0"]) == 1)
    # ---- the exit trap: a REFUSED line reaches HF
    rc, out, lg = sb.run(fresh=True, env_extra={"S1AUX_ARMS": "N-s0", "FAKE_PREFLIGHT_FAIL_ARM": "N-s0"}); ok(rc == 1, out)
    pushed = (sb.hub / "Sayood_tanitad-refe-s1aux" / "logs" / "chain.log").read_text(encoding="utf-8")
    ok("ZZS1AUX_REFUSED stage=arm_N-s0" in pushed, pushed[-300:])


# ============================================================================================ 4. the helper
class FakeItem:
    def __init__(self, path, sha=None, blob=None):
        self.path, self.blob_id = path, blob
        self.lfs = {"sha256": sha, "size": 1} if sha else None


class FakeApi:
    def __init__(self, items):
        self.items = items

    def list_repo_tree(self, repo, recursive=True, repo_type="model"):
        return list(self.items)


def helper_checks(tmp):
    tmp = Path(tmp)
    # ---- constants vs the registered values
    ok(SC.UPDATES == 400 and SC.TRAIN_SETS == 174912 and SC.HELDOUT_SETS == 3137 and SC.HELDOUT_LOGS == 24 and SC.TLANE_MIN_ROWS == 2109 and SC.STORE_KEYS == 103139 and SC.MODEL_B_MIN_BYTES == 1_200_000_000)
    import aux_drivable_head as AUX
    ok(AUX.PINNED_STORE_KEYS == SC.STORE_KEYS)
    # ---- tree
    pk = tmp / "pk"
    (pk / "refe").mkdir(parents=True)
    (pk / "refe" / "a.py").write_text("print(1)\n")
    (pk / "code").mkdir()
    (pk / "code" / "b.sh").write_text("echo 1\n")
    GT.write_json(pk / "SHIP_MANIFEST.json", GT.ship_manifest(pk))
    good, why = SC.tree_check(pk); ok(good, why)
    (pk / "refe" / "a.py").write_text("print(2)\n")
    good, why = SC.tree_check(pk); ok(not good and "refe/a.py" in why, why)                                  # changed
    (pk / "refe" / "a.py").write_text("print(1)\n"); (pk / "refe" / "c.py").write_text("x\n")
    good, why = SC.tree_check(pk); ok(not good and "refe/c.py" in why, why)                                  # extra
    (pk / "refe" / "c.py").unlink(); (pk / "code" / "b.sh").unlink()
    good, why = SC.tree_check(pk); ok(not good and "code/b.sh" in why, why)                                  # missing
    (pk / "SHIP_MANIFEST.json").unlink()
    good, why = SC.tree_check(pk); ok(not good and "missing" in why, why)
    # ---- store
    if (REAL_STORE / "manifest.json").is_file():
        good, why = SC.store_check(REAL_STORE); ok(good and "103139 keys" in why and "4a17db673b154" in why, why)
        bg = tmp / "bg"; bg.mkdir(); shutil.copy(REAL_STORE / "manifest.json", bg / "manifest.json")
        good, why = SC.store_check(bg); ok(not good, why)                                                    # shards absent
        m = json.loads((REAL_STORE / "manifest.json").read_text(encoding="utf-8")); m["store_sha256"] = "0" * 64
        (bg / "manifest.json").write_text(json.dumps(m))
        good, why = SC.store_check(bg); ok(not good and "pinned" in why, why)                                # another store
    else:
        print("ZZTEST_SKIP helper_checks: the real aux label store is absent")
    good, why = SC.store_check(tmp / "nowhere"); ok(not good, why)
    # ---- data event
    ev = {"event": "data", "train_sets": 174912, "heldout_sets": 3137, "heldout_logs": 24, "train_rows_in_heldout_logs": 0,
          "lane_labels": {"teacher_lane_train": 2109, "teacher_lane_heldout": 3137, "pdm_train": 174912, "pdm_heldout": 3137}}
    ok(SC.data_problems(ev) == [])
    for k, bad in (("train_sets", 146185), ("heldout_sets", 0), ("heldout_logs", 23), ("train_rows_in_heldout_logs", 5)):
        e = dict(ev, **{k: bad}); ok(SC.data_problems(e), k)
    for k, bad in (("teacher_lane_train", 0), ("teacher_lane_train", 2108), ("teacher_lane_heldout", 0), ("pdm_heldout", 0), ("pdm_train", 0)):
        e = json.loads(json.dumps(ev)); e["lane_labels"][k] = bad; ok(SC.data_problems(e), k)               # the SFT-4 defect class: a named dir that loads nothing
    ok(SC.data_problems(None))
    jl = tmp / "pre.jsonl"
    jl.write_text(json.dumps(ev) + "\n", encoding="utf-8")
    good, why = SC.data_check(jl); ok(good, why)
    jl.write_text(json.dumps(ev) + "\n" + json.dumps(ev) + "\n", encoding="utf-8")
    good, why = SC.data_check(jl); ok(not good, why)                                                         # two data events: not one run
    # ---- finished arms
    def mk(arm, head, updates=400, b_size=2000, drop=(), extra=()):
        d = tmp / f"arm_{arm}_{head}_{updates}_{b_size}_{'-'.join(drop)}"
        shutil.rmtree(d, ignore_errors=True); (d / "dump").mkdir(parents=True)
        rows = [ev, {"event": "start", "updates": updates}, {"event": "train", "update": 50, "bceA": 0.5, "lossB": 0.4, **({"aux_bce": 0.7} if head else {})}]
        for tag in ("step0", "update400"):
            rows.append({"event": "eval", "tag": tag})
            if head: rows.append({"event": "aux_eval", "tag": tag})
        rows.append({"event": "done", "updates": updates})
        rows = [r for r in rows if r.get("event") not in drop] + list(extra)
        (d / "sft.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        (d / "model_sft_B.pt").write_bytes(b"B" * b_size)
        for tag in ("step0", "update400"):
            (d / "dump" / f"scores_{tag}_B.npz").write_bytes(b"s")
            if head: (d / "dump" / f"head_pred_{tag}.npz").write_bytes(b"h")
        if head: (d / "aux_drv.pt").write_bytes(b"a")
        return d
    SC_min = SC.MODEL_B_MIN_BYTES
    SC.MODEL_B_MIN_BYTES = 1000                                                                              # the fixtures are small; the real constant was asserted above and is exercised below
    try:
        for arm, head in (("N-s0", False), ("H1-s0", True), ("H2-s1", True)):
            good, why = SC.run_check(mk(arm, head), arm); ok(good, why)
        ok(not SC.run_check(mk("N-s0", False, updates=300), "N-s0")[0])                                       # not an S1 arm
        ok(not SC.run_check(mk("N-s0", False, drop=("done",)), "N-s0")[0])                                    # never finished
        ok(not SC.run_check(mk("N-s0", False, drop=("eval",)), "N-s0")[0])
        ok(not SC.run_check(mk("N-s0", False, drop=("data",)), "N-s0")[0])
        ok(not SC.run_check(mk("N-s0", False, extra=({"event": "start", "updates": 400},)), "N-s0")[0])       # two starts = stitched
        ok(not SC.run_check(mk("N-s0", False, extra=({"event": "train", "bceA": float("nan"), "lossB": 1.0},)), "N-s0")[0])   # K2
        ok(not SC.run_check(mk("H1-s0", False), "H1-s0")[0])                                                  # a head arm without head artifacts
        ok(not SC.run_check(mk("N-s0", True), "N-s0")[0])                                                     # an N arm WITH head artifacts: not "the head alone differs"
        ok(not SC.run_check(mk("H2-s0", True, drop=("aux_eval",)), "H2-s0")[0])
        d = mk("H2-s0", True); (d / "aux_drv.pt").unlink(); ok(not SC.run_check(d, "H2-s0")[0])
        d = mk("H2-s0", True); (d / "dump" / "head_pred_update400.npz").unlink(); ok(not SC.run_check(d, "H2-s0")[0])
        d = mk("N-s0", False); (d / "dump" / "scores_update400_B.npz").unlink(); ok(not SC.run_check(d, "N-s0")[0])
    finally:
        SC.MODEL_B_MIN_BYTES = SC_min
    ok(not SC.run_check(mk("N-s0", False, b_size=2000), "N-s0")[0])                                           # with the REAL constant a 2 KB model_sft_B.pt is no checkpoint
    # ---- manifest row
    tk = tmp / "PASS_x.json"
    tk.write_text(json.dumps({"verdict": "PASS", "key_id": "kk", "created_utc": "t", "binding": {"profile": "refe_aux_s1", "commit": "c" * 40, "tree_sha256": "t" * 64, "argv_sha256": "a" * 64}}), encoding="utf-8")
    mo = tmp / "man.jsonl"
    r = SC.manifest_row("N-s0", "token_verified", mo, token=tk)
    ok(r["argv_sha256"] == "a" * 64 and r["commit"] == "c" * 40 and r["token_sha256"] == hashlib.sha256(tk.read_bytes()).hexdigest())
    SC.manifest_row("N-s0", "restart", mo, k=2, note="x")
    rows = [json.loads(l) for l in mo.read_text().splitlines()]
    ok(len(rows) == 2 and rows[1]["restart_k"] == 2)
    # ---- remote verification with a fake Hub
    f1, f2 = tmp / "big.pt", tmp / "small.txt"
    f1.write_bytes(b"BIG" * 100); f2.write_bytes(b"small file")
    items = [FakeItem("p/big.pt", sha=hashlib.sha256(f1.read_bytes()).hexdigest()), FakeItem("p/small.txt", blob=SC.git_blob_id(f2))]
    pairs = [("p/big.pt", f1), ("p/small.txt", f2)]
    ok(SC.verify_remote(FakeApi(items), "r", "model", pairs) == [])
    ok(SC.git_blob_id(f2) == subprocess.run(["git", "hash-object", str(f2)], capture_output=True, text=True).stdout.strip() or shutil.which("git") is None)   # the blob id is git's
    ok(SC.verify_remote(FakeApi([FakeItem("p/big.pt", sha="0" * 64), items[1]]), "r", "model", pairs) == ["sha256_mismatch:p/big.pt"])           # different LFS content
    ok(SC.verify_remote(FakeApi([items[0], FakeItem("p/small.txt", blob="1" * 40)]), "r", "model", pairs) == ["blob_id_mismatch:p/small.txt"])  # different small content
    ok(SC.verify_remote(FakeApi([items[0]]), "r", "model", pairs) == ["missing_remote:p/small.txt"])                                          # not there
    # ---- extract: members validated BEFORE they are written
    src = tmp / "src"; src.mkdir(); (src / "props_r0.jsonl").write_text("{}\n")
    tar = tmp / "q.tar"
    with tarfile.open(tar, "w") as t:
        t.add(src / "props_r0.jsonl", arcname="queue/props_r0.jsonl")
    import p2r_sync as S
    dest = tmp / "dest"
    n, nb = S.safe_extract(tar, dest, set()); ok(n == 1 and (dest / "queue" / "props_r0.jsonl").is_file())
    evil = tmp / "evil.tar"
    with tarfile.open(evil, "w") as t:
        info = tarfile.TarInfo("../escape.txt"); info.size = 2
        import io
        t.addfile(info, io.BytesIO(b"hi"))
    try:
        S.safe_extract(evil, tmp / "dest2", set()); ok(False, "an unsafe member was extracted")
    except ValueError as e:
        ok("unsafe_path" in str(e))
    ok(not (tmp / "escape.txt").exists())
    r = subprocess.run([sys.executable, str(HERE / "s1aux_checks.py"), "extract", "--archive", str(evil), "--dest", str(tmp / "dest3")], capture_output=True, text=True)
    ok(r.returncode == 1 and "ZZS1AUX_EXTRACT_REFUSED" in r.stdout, r.stdout)
    r = subprocess.run([sys.executable, str(HERE / "s1aux_checks.py"), "extract", "--archive", str(tar), "--dest", str(tmp / "dest4")], capture_output=True, text=True)
    ok(r.returncode == 0 and "ZZS1AUX_EXTRACT_OK" in r.stdout, r.stdout)
    # ---- the CLI prints ONE marker and exits 1 on a refusal
    r = subprocess.run([sys.executable, str(HERE / "s1aux_checks.py"), "tree", "--pkg", str(tmp / "pk")], capture_output=True, text=True)
    ok(r.returncode == 1 and r.stdout.startswith("ZZS1AUX_TREE_REFUSED"), r.stdout)


# ============================================================================================ 4b. the status script
def status_checks(tmp):
    tmp = Path(tmp)
    st = HERE / "s1aux_status.sh"
    bn = tmp / "bin"
    bn.mkdir(exist_ok=True)
    write_sh(bn / "flock", '#!/bin/bash\nexit "${FAKE_FLOCK_RC:-0}"\n')
    R = tmp / "s1"
    R.mkdir(exist_ok=True)

    def status(lines, held=False):
        (R / "chain.log").write_text("".join(l + "\n" for l in lines), encoding="utf-8")
        if held:
            (R / "chain.lock").write_text("")
        elif (R / "chain.lock").exists():
            (R / "chain.lock").unlink()
        r = subprocess.run([shutil.which("bash"), posix(st), posix(R)], capture_output=True, text=True,
                           env={**os.environ, "PATH": posix(bn) + os.pathsep + os.environ["PATH"], "FAKE_FLOCK_RC": "1" if held else "0"})
        return r.returncode, r.stdout.strip()
    T = "2026-10-14T01:02:03 "
    S0 = T + "ZZS1AUX_START chain (arms: N-s0 H2-s0)"
    rc, out = subprocess.run([shutil.which("bash"), posix(st), posix(tmp / "nowhere")], capture_output=True, text=True).returncode, ""
    ok(rc == 13)
    rc, out = status([S0, T + "ZZS1AUX_PRECHECK_OK x", T + "ZZS1AUX_ARM_GATE_OK N-s0 (x)", T + "ZZS1AUX_ARM_TRAIN_START N-s0"], held=True)
    ok(rc == 0 and out.startswith("ZZS1AUX_STATUS running") and "now=N-s0:TRAIN_START" in out and "arms_done=-" in out, (rc, out))
    rc, out = status([S0, T + "ZZS1AUX_WAITING the RL-S1 chain still holds x (waited 0s)"], held=True)
    ok(rc == 9 and "waiting" in out, (rc, out))
    rc, out = status([S0, T + "ZZS1AUX_ARM_DONE N-s0", T + "ZZS1AUX_ARM_DONE N-s0", T + "ZZS1AUX_ARM_TRAIN_START H2-s0"], held=False)
    ok(rc == 12 and "dead" in out and "arms_done=N-s0 " in out, (rc, out))                                   # a start with no terminal marker and nobody holding the lock
    rc, out = status([S0, T + "ZZS1AUX_ARM_DONE N-s0", T + "ZZS1AUX_ARM_DONE H2-s0", T + "ZZS1AUX_DONE arms: N-s0 H2-s0"], held=False)
    ok(rc == 10 and out.startswith("ZZS1AUX_STATUS done arms_done=N-s0,H2-s0"), (rc, out))
    rc, out = status([S0, T + "ZZS1AUX_REFUSED stage=fetch: u400 is not the pinned file"], held=False)
    ok(rc == 11 and "refused" in out and "stage=fetch" in out, (rc, out))
    rc, out = status([S0, T + "ZZS1AUX_REFUSED stage=fetch: x", T + "ZZS1AUX_START chain (re-run)"], held=True)      # a re-run after a refusal: the LAST start decides
    ok(rc == 0 and "running" in out, (rc, out))
    rc, out = status(["ZZS1AUX_DONE typed by hand, not a chain line", T + "ZZS1AUX_START chain"], held=True)             # the markers are anchored on the timestamp
    ok(rc == 0, (rc, out))


# ============================================================================================ 5. hygiene
def hygiene():
    for f in ("s1aux_pod.sh", "s1aux_checks.py", "s1aux_status.sh", "test_s1aux_chain.py"):
        ok(b"\r" not in (HERE / f).read_bytes(), f + " must stay LF")
    ok(endpoint_hits((HERE / "s1aux_pod.sh").read_text(encoding="utf-8")) == [], "chain")
    ok(endpoint_hits((HERE / "s1aux_checks.py").read_text(encoding="utf-8")) == [], "helper")
    ok(endpoint_hits((HERE / "s1aux_status.sh").read_text(encoding="utf-8")) == [], "status")
    rcb = subprocess.run([shutil.which("bash"), "-n", posix(HERE / "s1aux_status.sh")], capture_output=True, text=True)
    ok(rcb.returncode == 0, rcb.stderr)
    me = (HERE / "test_s1aux_chain.py").read_text(encoding="utf-8")
    if RUNBOOK.is_file():
        ok(endpoint_hits(RUNBOOK.read_text(encoding="utf-8")) == [], "runbook")
    # the test's own text: the only dotted quad is the planted mutation (a documentation address), never a real pod's
    quads = re.findall(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", me)
    ok(set(quads) <= {"10.20.30.40"}, quads)


def main() -> int:
    text = CHAIN.read_text(encoding="utf-8")
    rc, _, err = bash_run(f'bash -n "{posix(CHAIN)}"')
    ok(rc == 0, err)                                                                                       # bash can parse it
    ok(check_chain(text) == [], check_chain(text))
    for name, mut, want in mutations(text):
        ok(mut != text, f"mutation '{name}' changed nothing")
        got = check_chain(mut)
        ok(any(g.startswith(want) for g in got), f"mutation '{name}' did not raise {want}: {got}")
    hygiene()
    with tempfile.TemporaryDirectory() as tmp:
        behaviour(text, _mk(tmp, "b"))
        helper_checks(_mk(tmp, "h"))
        status_checks(_mk(tmp, "st"))
        sandbox_runs(text, _mk(tmp, "s"))
    print(f"ZZTEST_PASS {N_OK}")
    return 0


def _mk(tmp, name):
    p = Path(tmp) / name
    p.mkdir(exist_ok=True)
    return p


if __name__ == "__main__":
    sys.exit(main())
