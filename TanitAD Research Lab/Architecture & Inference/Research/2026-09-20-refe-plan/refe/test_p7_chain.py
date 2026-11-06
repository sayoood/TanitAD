#!/usr/bin/env python3
"""Tests for the P7 pod chain: refe/p7_pod.sh, refe/p7_prep.py (SPEC_P7; BACKLOG P7-LAUNCH-READY (2) + PKG-SHIP-COMPLETE).  Same stub-sandbox style as test_p6_chain.py.

No pod, no GPU, no network, no NAVSIM stack.  Four layers:
  1. `check_p7(text)`: static properties of the chain, each with a MUTATION of the chain text that `check_p7` must report (a check that cannot go RED is not a check).
  2. the helpers `p7_prep.py` (twins plan/apply, noparts, local-check) in both directions, and the expanded ARGS against the REAL refe_p7 profile (`launch_gate.py refusals`).
  3. the REAL p7_pod.sh under bash in a sandbox with stubbed tools: the happy path (stage order, call arguments), then every refusal path with its own arm:
       (a) twins LINKED then `lane-distinct` OK at every rank BEFORE the gate; a missing / conflicting / reserve-log twin, non-distinct twins and a rank set other than the registered one REFUSE;
       (b) P7_D7_FALLBACK defaults to 0 and any other value REFUSES; no quarantine, no twin build ever runs;
       (c) the `no_parts` guard stops the trainer by pid and refuses; a resumed attempt judges its own segment;
       (d) --label-workers is pinned by P7_LABEL_WORKERS (unset / non-integer REFUSES; never derived from the cgroup quota);
       (e) the run dir and metric cache on a network/FUSE filesystem REFUSE;
       p7_pins.py --check-run after the first checkpoint: a mismatch REFUSES and STOPS the trainer (also with the REAL p7_pins.py against the REGISTERED pins);
       its own run dir / HF repo / package / chain dir, never P6's; P6 over (P7_WAIT_P6); an incomplete package refused before any download.
  4. behavioural MUTANTS of the chain (delete the kill, flip the guard, re-add the fallback): each must make a layer-3 assertion fail.
Run: python refe/test_p7_chain.py   -> ZZP7CHAIN_PASS <n>   (needs bash on PATH; Git Bash on the dev box).  Never prints ZZTEST_SKIP.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import test_p2r_chain as T2  # noqa: E402
import test_p6_chain as T6  # noqa: E402
import p7_prep as PP7  # noqa: E402

CHAIN = HERE / "p7_pod.sh"
NPASS = 0
FAILS: list = []


def ok(cond, name):
    global NPASS
    if cond:
        NPASS += 1
    else:
        FAILS.append(name)
        print("  FAIL:", name, flush=True)


def code_of(text):
    return "\n".join(ln for ln in text.split("\n") if not ln.strip().startswith("#"))


ARGS_REF7 = ("--backbone vitl16 --targets $CBANK --images $PIX --calib $BANK/calib_table.json --metric-cache $MC --scorer-sees-trunk --yaw-wrap $YAW_WRAP --nav-mode command "
             "--labeller online --label-workers $LW --label-chunks $LC --epochs 25 --epoch-unit scenes --batch 4 --accum 64 --lr 2e-4 --weight-decay 0.01 --amp bf16 --score-w 0.1 "
             "--aux-drv --w-aux-drv 0.1 --aux-drv-tap visual --aux-drv-depth 2 --aux-drv-feed off --aux-drv-labels $P7_LABEL_STORE --w-field 0.0 "
             "--tf32 --compile --workers 2 --log-every 1 --ckpt-every-min 20 --out $RUN --resume")
ORDER7 = ['"$PY0" "$PKG/refe/p7_ship_list.py" check', 'p7_prep.py" local-check', 'source "$WORK/teacher_env.sh"', '"$PY" p2r_sync.py watch', "STAGES=weights", "p2r_sync.py recover",
          "p6_prep.py reserve-verify", "p6_prep.py reserve-filter", "build_metric_cache.py build --targets", "p6_prep.py lane-ranks", "p7_prep.py twins plan", "p7_prep.py twins apply",
          "p6_prep.py lane-distinct", "\nARGS=", '"$PY" diag_p6_labeller_identity.py --cache', "\nwait_clear\n", "train.py $MEM_ARGS", "hf_egress.py gate", "launch_gate.py run --profile refe_p7",
          "train.py $ARGS --preflight", "train.py $ARGS >>", "p2r_sync.py final"]


def check_p7(text):
    v = []
    code = code_of(text)
    if "\r" in text:
        v.append("CRLF")
    am = T6.ARGS_RX.findall(text)
    if len(am) != 1 or am[0] != ARGS_REF7:
        v.append("ARGS_CHANGED_OR_TWICE")
    for ln in text.split("\n"):
        if "HF_TOKEN" in ln and not T2.token_line_ok(ln):
            v.append("HF_TOKEN_USE:" + ln.strip()[:50])
    for c in T2.logical_commands(text):
        if T6.CHILD_RX.search(c) and "200>&-" not in c:
            v.append("FD200:" + c[:60])
    pos = [code.find(k) for k in ORDER7]
    for k, p in zip(ORDER7, pos):
        if p < 0:
            v.append("MISSING:" + k.strip()[:50])
    if all(p >= 0 for p in pos) and pos != sorted(pos):
        v.append("ORDER:" + ",".join(ORDER7[i].strip()[:30] for i in range(1, len(ORDER7)) if pos[i] < pos[i - 1]))
    if "--profile refe_p6" in code:
        v.append("P6_PROFILE_NAMED")
    if '--gate-reserve-json "$P7_RESERVE"' not in code or "verify --profile refe_p7" not in code:
        v.append("GATE_NOT_P7_RESERVE")
    if not re.search(r'if \[ "\$pins_done" = 0 \] && \[ -f "\$RUN/ckpt_last\.pt" \]; then\n\s+if "\$PY" p7_pins\.py --check-run "\$RUN" .*; then\n.*\n\s+else\n\s+: > "\$P7/pins\.refused";.*kill_pid "\$pid"; return 0', code):
        v.append("PINS_CHECK_NOT_AFTER_FIRST_CKPT_WITH_KILL")
    if '[ ! -f "$P7/pins.refused" ] || refuse' not in code:
        v.append("PINS_REFUSAL_NOT_FINAL")
    if "kill_pid()" not in code or "kill -9" not in code or re.search(r"\bpkill\b|\bkillall\b", code):
        v.append("KILL_NOT_BY_PID")
    if not re.search(r'p7_prep\.py noparts --run "\$RUN" --skip "\$skip"', code) or '[ ! -f "$P7/noparts.refused" ] || refuse' not in code:
        v.append("NOPARTS_GUARD_MISSING")
    if '--count-only' not in code:
        v.append("NOPARTS_NOT_PER_ATTEMPT")
    for need in ('[ "$RUN" != "$P6RUN" ]', '[ "$REPO" != "$P6REPO" ] || refuse', '[ "$PKG" != "$WORK/refe-plan-p6" ] || refuse', '[ "$P7" != "$P6DIR" ] || refuse'):
        if need not in code:
            v.append("ISOLATION_MISSING:" + need[:22])
    if re.search(r"p6_prep\.py quarantine|build_metric_cache\.py build [^\n]*--lane-rank|FALLBACK=1|_D7_FALLBACK:-1", code) or not re.search(r'case "\$\{P7_D7_FALLBACK:-0\}" in\n  0\) ;;\n  \*\) refuse', code):
        v.append("D7_FALLBACK_OR_BUILD_PRESENT")
    if not re.search(r'case "\$\{P7_LABEL_WORKERS:-\}" in\n  \'\'\|\*\[!0-9\]\*\) refuse', code) or re.search(r"LW=.*(CORES|nproc|cgroup)", code):
        v.append("LABEL_WORKERS_NOT_PINNED")
    if 'for _d in "$RUN" "$MC"; do' not in code or "local-check --path" not in code:
        v.append("LOCAL_DISK_CHECK_MISSING")
    if re.search(r'if \[ -n "\$\{P7_RESERVE:-\}" \]', code) or not (code.find('p6_prep.py reserve-filter') < code.find("\nARGS=")):
        v.append("RESERVE_STAGE_DORMANT")
    if not re.search(r'\[ "\$RANKS" = "\$EXPECT_RANKS" \] \|\| refuse', code):
        v.append("RANKS_NOT_EXACT")
    if not re.search(r'twins plan .*--reserve "\$P7_RESERVE"', code) or not re.search(r'twins apply .*--reserve "\$P7_RESERVE"', code):
        v.append("TWINS_NOT_RESERVE_AWARE")
    return v


def mutations(text):
    def sub(a, b, cnt=1):
        return lambda t: (t.replace(a, b, cnt) if a in t else (_ for _ in ()).throw(AssertionError("anchor missing: " + a[:40])))
    return [
        ("P6_PROFILE_NAMED", sub("--profile refe_p7 --pkg", "--profile refe_p6 --pkg")),
        ("GATE_NOT_P7_RESERVE", sub(' --gate-reserve-json "$P7_RESERVE"', "")),
        ("PINS_CHECK_NOT_AFTER_FIRST_CKPT_WITH_KILL", sub('kill_pid "$pid"; return 0\n      fi\n    fi\n    if [ "$np_done"', 'return 0\n      fi\n    fi\n    if [ "$np_done"')),
        ("PINS_REFUSAL_NOT_FINAL", sub('[ ! -f "$P7/pins.refused" ] || refuse', '[ ! -f "$P7/pins.refused" ] || log')),
        ("KILL_NOT_BY_PID", sub("  kill -9 \"$p\" 2> /dev/null\n", "  pkill -f train.py\n")),
        ("NOPARTS_GUARD_MISSING", sub('[ ! -f "$P7/noparts.refused" ] || refuse', '[ ! -f "$P7/noparts.refused" ] || log')),
        ("NOPARTS_NOT_PER_ATTEMPT", sub(" --count-only", "")),
        ("ISOLATION_MISSING", sub('[ "$REPO" != "$P6REPO" ] || refuse', ': || refuse')),
        ("D7_FALLBACK_OR_BUILD_PRESENT", sub("  0) ;;\n  *) refuse \"P7_D7_FALLBACK", "  0|1) ;;\n  *) refuse \"P7_D7_FALLBACK")),
        ("LABEL_WORKERS_NOT_PINNED", sub('LW="$P7_LABEL_WORKERS"; ', 'LW="$(( CORES - 2 ))"; ')),
        ("LOCAL_DISK_CHECK_MISSING", sub('for _d in "$RUN" "$MC"; do', 'for _d in "$RUN"; do')),
        ("RESERVE_STAGE_DORMANT", sub('P7OUT="$("$PY" p6_prep.py reserve-verify', 'if [ -n "${P7_RESERVE:-}" ]; then :; fi\nP7OUT="$("$PY" p6_prep.py reserve-verify')),
        ("RANKS_NOT_EXACT", sub('[ "$RANKS" = "$EXPECT_RANKS" ] || refuse', '[ -n "$RANKS" ] || refuse')),
        ("TWINS_NOT_RESERVE_AWARE", sub('twins apply --bank "$CBANK" --src "$TWIN_SRC" --dst "$MC" --reserve "$P7_RESERVE"', 'twins apply --bank "$CBANK" --src "$TWIN_SRC" --dst "$MC"')),
        ("ARGS_CHANGED_OR_TWICE", sub("--w-aux-drv 0.1 ", "")),
        ("MISSING", sub('"$PY0" "$PKG/refe/p7_ship_list.py" check', "true")),
    ]


# ------------------------------------------------------------------------------------------------ the sandbox
_i = T6.FAKEPY.index('elif script == "train.py":')
_head, _tail = T6.FAKEPY[:_i], T6.FAKEPY[T6.FAKEPY.index("else:\n    print(\"fakepy: unknown stub"):]
_route = 'if script in ("p6_prep.py", "annotate_commands.py", "p6_overlap.py"):'
assert _route in _head
_head = _head.replace(_route, 'if script in ("p6_prep.py", "annotate_commands.py", "p6_overlap.py", "p7_prep.py") and not (script == "p6_prep.py" and rest[0].startswith("reserve-") and rest[0] != "reserve-verify"):', 1)
STUB7 = r'''
elif script == "p6_prep.py":                     # reserve-* (the REAL ones are test_session_reserve.py's / test_p7_integration.py's)
    sub = rest[0]
    if sub == "reserve-filter":
        out = Path(opt("--out")); out.mkdir(parents=True, exist_ok=True)
        for fp in sorted(Path(opt("--bank")).glob("targets_rank*.jsonl")):
            (out / fp.name).write_bytes(fp.read_bytes())
        print("ZZRESERVE_FILTER_OK rows_in=1 rows_out=1 excluded=0 unresolved=0 targets_sha256=stub")
    elif sub == "reserve-calib":
        Path(opt("--out")).write_text("{}"); print("ZZRESERVE_CALIB_OK logs_in=3 logs_out=3 excluded=0")
    elif sub == "reserve-move":
        print("ZZRESERVE_MOVE_OK kind=stub moved=0 already=0")
    elif sub == "reserve-labelstore":
        print("ZZRESERVE_LABELSTORE_OK moved=0")
    elif sub == "reserve-audit":
        if os.environ.get("STUB_AUDIT_LEAK"):
            print("ZZRESERVE_AUDIT_LEAK stub"); sys.exit(1)
        Path(opt("--out")).write_text("{}"); print("ZZRESERVE_AUDIT_OK inputs=5")
elif script == "aux_label_store.py":
    print("ZZ finalize stub")
elif script == "p7_ship_list.py":
    if os.environ.get("STUB_SHIP_BAD"):
        print("ZZP7SHIP_BAD missing=1 first=['eval/rls1b_arms.json']"); sys.exit(1)
    print("ZZP7SHIP_OK files=1")
elif script == "p7_pins.py":
    if os.environ.get("STUB_REAL_PINS"):
        sys.path.insert(0, str(REAL)); sys.argv = [str(REAL / script)] + rest
        runpy.run_path(str(REAL / script), run_name="__main__")
    elif os.environ.get("STUB_PINS_BAD"):
        print("ZZP7RUN_REFUSED stub: targets_sha256 differs", file=sys.stderr); sys.exit(1)
    else:
        print("ZZP7RUN_OK bank=stub")
elif script == "train.py":
    out = Path(opt("--out") or ".")
    if "--preflight" in rest:
        print("PREFLIGHT_OK")
    elif "--steps" in rest:
        out.mkdir(parents=True, exist_ok=True)
        (out / "metrics.jsonl").write_text(json.dumps({"traj_L1": 1.0, "cuda_max_mem_gb": 31.0, "samples": 800, "batch": 4}) + "\n")
        (out / "ckpt_last.pt").write_bytes(b"x" * 2000)
        print("TRAIN_DONE")
    else:
        import time
        cnt = Path(os.environ["STUB_TRAIN_COUNT"]); n = int(cnt.read_text()) if cnt.exists() else 0; cnt.write_text(str(n + 1))
        out.mkdir(parents=True, exist_ok=True)
        (out / "config.json").write_text(json.dumps({"args": {"targets": opt("--targets")}, "identity": {"targets_sha256": os.environ.get("STUB_ID_SHA", "x"), "n_tuples": int(os.environ.get("STUB_ID_N", "0")), "n_scenes": 0}}))
        (out / "ckpt_last.pt").write_bytes(b"x" * 2000)
        with open(out / "metrics.jsonl", "a") as fh:
            fh.write("\x00\x00\x00 not json\n")                                     # a NUL-padded line (PREREG_P6 Amendment (d) item 4)
            for i in range(int(os.environ.get("STUB_EVENTS", "5"))):
                fh.write(json.dumps({"step": i, "event": "labeller", "sets": 256, "labelled": 256, "no_parts": int(os.environ.get("STUB_NOPARTS", "0")), "failed": 0}) + "\n")
        time.sleep(float(os.environ.get("STUB_TRAIN_SLEEP", "4")))
        (out / "model_final.pt").write_bytes(b"m"); (out / "summary.json").write_text(json.dumps({"done": True}))
        print("TRAIN_DONE")
'''
FAKEPY7 = _head + STUB7 + _tail
REQUIRED7 = ["refe/" + n for n in ("train.py p2r_sync.py hf_egress.py p6_prep.py p7_prep.py p7_pins.py session_reserve.py aux_label_store.py aux_drivable_head.py online_labeller.py build_metric_cache.py "
             "diag_p6_labeller_identity.py launch_gate.py gate_token.py gate_checks.py gate_fixture.py test_launch_gate.py test_p2r_chain.py test_p6_levers.py test_p6_chain.py "
             "test_p7_head_wiring.py test_session_reserve.py test_p7_integration.py test_p7_chain.py p7_ship_list.py p7_ship_list.json").split()] + \
    ["SHIP_MANIFEST.json", "code/pod_teacher_env.sh", "code/pod_train.sh", "code/pod_dataprep.sh", "splits/navtrain.yaml"]
P6RUNDIR, P6REPOID = "refe_runs/p6_paper_faithful", "Sayood/tanitad-refe-p6-paper-faithful"


class Sandbox7(T6.Sandbox):
    """A fake /workspace in which the REAL p7_pod.sh runs under bash; every tool is a stub that records its call.  p6_prep.py (except reserve-*), p7_prep.py are the REAL files."""

    def __init__(self, root, script_text=None):
        self.root = Path(root); self.ws = self.root / "ws"; self.data = self.ws / "data"; self.pkg = self.ws / "refe-plan-p7"; self.stub = self.root / "stub"
        self.calls = self.root / "calls.log"; self.calls.write_text("", encoding="utf-8")
        for d in (self.pkg / "refe" / "fixtures", self.pkg / "code", self.pkg / "splits", self.stub):
            d.mkdir(parents=True)
        self.script = self.pkg / "refe" / "p7_pod.sh"
        self.script.write_text(script_text if script_text is not None else CHAIN.read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
        for f in REQUIRED7:
            (self.pkg / f).write_text("", encoding="utf-8")
        shutil.copyfile(HERE / "fixtures" / "p7_reserve_sessions.json", self.pkg / "refe" / "fixtures" / "p7_reserve_sessions.json")      # the REAL, sha256-pinned reserve file (reserve-verify is the real one)
        for name, body in (("pod_train.sh", T6.POD_TRAIN), ("pod_dataprep.sh", T6.POD_DATAPREP.replace("P6_EXPECT", "P7_EXPECT"))):
            (self.pkg / "code" / name).write_text(body, encoding="utf-8", newline="\n")
        (self.stub / "fakepy.py").write_text(FAKEPY7, encoding="utf-8")
        py = T6.posix(sys.executable)
        (self.stub / "fakepy").write_text(f'#!/bin/bash\nexec "{py}" "{T6.posix(self.stub / "fakepy.py")}" "$@"\n', encoding="utf-8", newline="\n")
        (self.stub / "bash_env.sh").write_text(
            'flock() {\n  local fd="${@: -1}" tgt who held\n  tgt="$(readlink /dev/fd/$fd 2> /dev/null)"\n  who="$(basename "$(dirname "$tgt")")"\n'
            '  held="${STUB_HELD:-}"; [ -n "${STUB_HELD_FILE:-}" ] && [ -f "$STUB_HELD_FILE" ] && held="$held $(cat "$STUB_HELD_FILE")"\n'
            '  case " $held " in *" $who "*) return 1 ;; esac\n  return 0\n}\n', encoding="utf-8", newline="\n")
        (self.stub / "nice").write_text('#!/bin/bash\necho "NICE $2" >> "$STUB_CALLS.nice"\nshift 2\nexec "$@"\n', encoding="utf-8", newline="\n")
        (self.stub / "nvidia-smi").write_text('#!/bin/bash\n[ -f "$STUB_GPU_BUSY_FILE" ] && echo 4242\nexit 0\n', encoding="utf-8", newline="\n")
        subprocess.run([T6.bash_exe(), "-c", f"chmod +x '{T6.posix(self.stub / 'nice')}' '{T6.posix(self.stub / 'nvidia-smi')}' '{T6.posix(self.stub / 'fakepy')}'"], check=True)
        maps = self.data / "nuplan-maps" / "v"
        for c in ("a", "b", "c", "d"):
            (maps / c).mkdir(parents=True)
        (self.ws / "teacher_env.sh").write_text(f'export DRIVERL_EVAL_PYTHON="{T6.posix(self.stub / "fakepy")}" NUPLAN_MAPS_ROOT="{T6.posix(maps)}"\n', encoding="utf-8", newline="\n")
        nav = self.data / "refe_navtrain"
        for bname in ("train_grow", "train_grow_cmd_3c"):
            (nav / bname).mkdir(parents=True)
            (nav / bname / "calib_table.json").write_text("{}", encoding="utf-8")
            r0 = [{"log_name": lg, "token": t, "step": 0, "rank": 0} for lg in T6.LOGS3 for t in T6.TOKS2]
            r1 = [{"log_name": "L0", "token": "t0", "step": 0, "rank": 1, "aug": {"kind": "goal_horizon", "value": 16}},
                  {"log_name": "L0", "token": "t0", "step": 0, "rank": 1, "aug": {"kind": "lane_rank", "value": 1}},
                  {"log_name": "L1", "token": "t1", "step": 0, "rank": 1, "aug": {"kind": "lane_rank", "value": 1}}]
            for nm, rs in (("targets_rank0.jsonl", r0), ("targets_rank1.jsonl", r1)):
                (nav / bname / nm).write_text("".join(json.dumps(r) + "\n" for r in rs), encoding="utf-8", newline="\n")
        self.twin_src = nav / "metric_cache_twins_3c"
        T6.make_parts_tree(self.twin_src, [("L0", "t0"), ("L1", "t1")], twin_k=1)
        (nav / "aux_label_store_p7").mkdir(); (nav / "aux_label_store_p7" / "manifest.json").write_text("{}", encoding="utf-8")
        (self.data / "navtrain_dbs").mkdir(parents=True)
        for i in range(3):
            (self.data / "navtrain_dbs" / f"l{i}.db").write_text("", encoding="utf-8")
        (self.data / "navtrain_pixels").mkdir(parents=True)
        for i in range(2):
            (self.data / "navtrain_pixels" / f".done_{i}").write_text("", encoding="utf-8")
        (self.root / "mounts").write_text("overlay / overlay rw 0 0\n", encoding="utf-8")

    def run(self, extra_env=None, timeout=300):
        env = T6.sandbox_env(os.environ)
        env.update({"WORK": T6.posix(self.ws), "HF_TOKEN": T6.TOKEN, "P7_DD_GB": "0", "P7_EXPECT_DBS": "3", "P7_EXPECT_PIX_SHARDS": "2", "P7_ATTEMPT_SLEEP": "0", "P7_POLL_S": "0",
                    "STUB_CALLS": T6.posix(self.calls), "STUB_REAL_REFE": T6.posix(HERE), "STUB_TRAIN_COUNT": T6.posix(self.root / "train_count"),
                    "STUB_FORCE_TOKEN_FAIL": T6.posix(self.root / "force_token_fail"), "BASH_ENV": T6.posix(self.stub / "bash_env.sh"), "P7_CACHE_WORKERS": "2",
                    "P7_NVSMI": T6.posix(self.stub / "nvidia-smi"), "STUB_GPU_BUSY_FILE": T6.posix(self.root / "gpu_busy"), "P7_LABEL_WORKERS": "6", "PYTHONIOENCODING": "utf-8",
                    "P7_BOOT_PYTHON": T6.posix(self.stub / "fakepy"), "P7_MOUNTS": T6.posix(self.root / "mounts"), "P7_EXPECT_RANKS": "1", "P7_GUARD_POLL_S": "1", "P7_KILL_WAIT_S": "5",
                    "P7_NOPARTS_EVENTS": "3", "P7_TWIN_HOW": "copy"})
        env["PATH"] = str(self.stub) + os.pathsep + env.get("PATH", "")
        env.update({k: str(v).replace("__WS__", T6.posix(self.ws)) for k, v in (extra_env or {}).items()})
        r = subprocess.run([T6.bash_exe(), T6.posix(self.script)], capture_output=True, text=True, env=env, timeout=timeout, encoding="utf-8", errors="replace")
        return r.returncode, r.stdout + r.stderr

    def chain_log(self):
        p = self.data / "refe_p7" / "chain.log"
        return p.read_text(encoding="utf-8") if p.exists() else ""

    def stage(self, name):
        return (self.data / "refe_p7" / "stage" / f"{name}.ok").exists()

    @property
    def run_dir(self):
        return self.data / "refe_runs" / "p7_reserve_head"


def first_idx(calls, pre):
    return next((i for i, c in enumerate(calls) if c.startswith(pre)), -1)


def main() -> int:
    text = CHAIN.read_text(encoding="utf-8")
    # ============================================================ layer 1
    v0 = check_p7(text)
    ok(v0 == [], f"check_p7 on the real chain: {v0[:4]}")
    for name, fn in mutations(text):
        try:
            mv = check_p7(fn(text))
        except AssertionError as e:
            ok(False, f"mutation {name}: {e}")
            continue
        ok(any(x.startswith(name) for x in mv), f"MUTANT {name} not reported (got {mv[:3]})")
    ok("--profile refe_p6" not in code_of(text), "no code line gates on refe_p6")

    # ============================================================ layer 2: the helpers
    tmp = Path(tempfile.mkdtemp(prefix="p7chain_"))
    try:
        # --- local-check, both directions, on injected mounts
        mf = tmp / "m1"; mf.write_text("overlay / overlay rw 0 0\nmfs#x /workspace/data fuse.mfs rw 0 0\ntmpfs /dev tmpfs rw 0 0\n")
        ok(PP7.local_check("/workspace/data/refe_runs/p7", mf)[0] is False, "local-check: a path under a fuse.mfs mount is refused")
        ok(PP7.local_check("/workspace/other", mf)[0] is True and PP7.local_check("/root/tad/mc", mf)[0] is True, "local-check: overlay paths pass (longest mount prefix)")
        ok(PP7.local_check("/workspace/data", tmp / "absent")[0] is False, "local-check: an unreadable mounts file is never 'local'")
        ok(all(PP7.is_network_fs(t) for t in ("nfs4", "fuse", "fuseblk", "fuse.mfs", "cifs", "9p")) and not any(PP7.is_network_fs(t) for t in ("overlay", "ext4", "xfs", "tmpfs", "btrfs")), "is_network_fs table")
        # --- noparts
        run = tmp / "run"; run.mkdir()

        def ev(n, nop):
            return "".join(json.dumps({"step": i, "event": "labeller", "sets": 256, "no_parts": nop}) + "\n" for i in range(n))
        (run / "metrics.jsonl").write_text("\x00\x00\n" + ev(3, 0) + "{broken\n", encoding="utf-8")
        a = lambda *x: subprocess.run([sys.executable, str(HERE / "p7_prep.py"), "noparts", "--run", str(run), *x], capture_output=True, text=True)
        ok(a("--events", "3").returncode == 0, "noparts: 3 clean events pass (NUL / broken lines skipped)")
        ok(a("--events", "5").returncode == 2, "noparts: fewer events than asked -> WAIT (rc 2)")
        (run / "metrics.jsonl").write_text(ev(3, 29), encoding="utf-8")
        r = a("--events", "3")
        ok(r.returncode == 1 and "ZZP7NOPARTS_BAD" in r.stdout, "noparts: R67's 29 of 256 sets REFUSES (rc 1)")
        (run / "metrics.jsonl").write_text(ev(3, 29) + ev(3, 0), encoding="utf-8")
        ok(a("--events", "3", "--skip", "3").returncode == 0 and a("--events", "3").returncode == 1, "noparts: --skip judges the resumed segment, the first events judge themselves")
        ok("ZZP7NOPARTS_COUNT 6" in a("--count-only").stdout, "noparts: --count-only")
        (run / "metrics.jsonl").write_text(ev(3, 0).replace('"no_parts": 0', '"no_parts": 0, "sets": 0', 1).replace('"sets": 256, "no_parts": 0, "sets": 0', '"sets": 0, "no_parts": 0'), encoding="utf-8")
        # --- twins: the happy path, then each refusal
        bank, src, dst, rsv = tmp / "bank", tmp / "src", tmp / "mc", HERE / "fixtures" / "p7_reserve_sessions.json"
        bank.mkdir()
        (bank / "targets_rank1.jsonl").write_text("".join(json.dumps(r) + "\n" for r in (
            {"log_name": "L0", "token": "t0", "rank": 1, "aug": {"kind": "lane_rank", "value": 1}}, {"log_name": "L1", "token": "t1", "rank": 1, "aug": {"kind": "lane_rank", "value": 2}},
            {"log_name": "L1", "token": "t1", "rank": 1, "aug": {"kind": "goal_horizon", "value": 16}})))
        T6.make_parts_tree(src, [("L0", "t0")], twin_k=1); T6.make_parts_tree(src, [("L1", "t1")], twin_k=2)
        T6.make_parts_tree(dst, [("L0", "t0"), ("L1", "t1")], twin_k=None)
        tw = lambda mode, *x: subprocess.run([sys.executable, str(HERE / "p7_prep.py"), "twins", mode, "--bank", str(bank), "--src", str(src), "--dst", str(dst), *x], capture_output=True, text=True)
        p = tw("plan", "--reserve", str(rsv))
        ok(p.returncode == 0 and "to_place=2" in p.stdout and not (dst / "L0" / "t0__lane_rank_1.pkl.xz").exists(), f"twins plan: placeable, writes nothing ({p.stdout.strip()[:100]})")
        p = tw("apply", "--reserve", str(rsv), "--how", "copy")
        ok(p.returncode == 0 and (dst / "L0" / "t0__lane_rank_1.pkl.xz").is_file() and (dst / "L1" / "t1__lane_rank_2.pkl.xz").is_file() and "dst_missing_after=0" in p.stdout, f"twins apply: both twins placed, 100 % ({p.stdout.strip()[:120]})")
        ok(tw("apply", "--reserve", str(rsv)).returncode == 0 and "to_place=0" in tw("plan", "--reserve", str(rsv)).stdout, "twins apply is idempotent (present, to_place=0)")
        (dst / "L0" / "t0__lane_rank_1.pkl.xz").unlink()
        os.remove(src / "L0" / "t0__lane_rank_1.pkl.xz")
        p = tw("apply", "--reserve", str(rsv))
        ok(p.returncode == 1 and "missing_src=1" in p.stdout and not (dst / "L0" / "t0__lane_rank_1.pkl.xz").exists(), "twins: a twin missing in the source REFUSES before writing")
        T6.make_parts_tree(src, [("L0", "t0")], twin_k=1)
        (dst / "L1" / "t1__lane_rank_2.pkl.xz").write_bytes(b"different")
        p = tw("apply", "--reserve", str(rsv))
        ok(p.returncode == 1 and "conflict=1" in p.stdout, "twins: a different file at the destination REFUSES (conflict)")
        shutil.rmtree(dst / "L1")
        p = tw("plan", "--reserve", str(rsv))
        ok(p.returncode == 1 and "missing_logged=1" in p.stdout, "twins: a missing LOGGED part REFUSES (the twin's partner)")
        T6.make_parts_tree(dst, [("L1", "t1")], twin_k=None)
        import session_reserve as SR
        rl = sorted(SR.reserve_logs(rsv))[0]
        (bank / "targets_rank1.jsonl").write_text(json.dumps({"log_name": rl, "token": "tz", "rank": 1, "aug": {"kind": "lane_rank", "value": 1}}) + "\n")
        p = tw("plan", "--reserve", str(rsv))
        ok(p.returncode == 1 and "reserve_rows=1" in p.stdout, "twins: a bank row whose log is in the RESERVE refuses (it would put a reserve key back into the cache)")
        (bank / "targets_rank1.jsonl").write_text(json.dumps({"log_name": "L0", "token": "t0", "rank": 1}) + "\n")
        ok(tw("plan").returncode == 1 and "no_lane_rank_rows" in tw("plan").stdout, "twins: a bank without a single lane_rank twin REFUSES")

        # --- the expanded ARGS against the REAL refe_p7 profile
        import launch_gate as LG
        expanded = ARGS_REF7
        for k, val in (("$CBANK", "/w/train_grow_cmd_3c_p7"), ("$PIX", "/w/px"), ("$BANK", "/w/calib"), ("$MC", "/w/mc"), ("$YAW_WRAP", "off"), ("$LW", "6"), ("$LC", "2"),
                       ("$P7_LABEL_STORE", "/w/store"), ("$RUN", "/w/run")):
            expanded = expanded.replace(k, val)
        argv = LG.parse_args_string(expanded)
        prof = LG.PROFILES["refe_p7"]
        ok(LG.p7_refusals(prof, argv) == [], "the chain's ARGS pass p7_refusals (pins registered)")
        ok(not LG.profile_refusals(prof, argv), f"the chain's ARGS pass the refe_p7 profile's argv refusals: {LG.profile_refusals(prof, argv)[:2]}")
        bad = LG.parse_args_string(expanded.replace("--w-aux-drv 0.1 ", ""))
        ok(bool(LG.profile_refusals(prof, bad)), "control: the same ARGS WITHOUT --w-aux-drv are refused (default-disables class)")
        ok(bool(LG.p7_refusals(prof, LG.parse_args_string(expanded.replace("train_grow_cmd_3c_p7", "train_grow_cmd_3c")))), "control: the UNFILTERED 3-C bank name is refused")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # ============================================================ layer 3: the real chain in the sandbox
    def sb_run(extra=None, mutate=None, prep=None, timeout=300):
        d = Path(tempfile.mkdtemp(prefix="p7sb_"))
        s = Sandbox7(d, mutate(text) if mutate else None)
        if prep:
            prep(s)
        rc, out = s.run(extra)
        return s, rc, out, d

    def done(s, rc, out):
        return rc == 0 and "ZZP7_DONE" in s.chain_log()

    dirs = []
    try:
        s, rc, out, d = sb_run(); dirs.append(d)
        calls = s.calls_list(); lg = s.chain_log()
        ok(done(s, rc, out), f"HAPPY PATH reaches ZZP7_DONE (rc {rc}; tail {lg[-300:]!r})")
        ok(first_idx(calls, "p7_ship_list.py check") == 0, "the ship-list check is the FIRST tool call (before any download)")
        ok(first_idx(calls, "p7_prep.py twins apply") != -1 and first_idx(calls, "p7_prep.py twins apply") < first_idx(calls, "p6_prep.py lane-distinct") < first_idx(calls, "diag_p6_labeller_identity.py")
           < first_idx(calls, "launch_gate.py run"), "twins applied, THEN lane-distinct, THEN lab, THEN the gate")
        gate = next((c for c in calls if c.startswith("launch_gate.py run")), "")
        ok("--profile refe_p7" in gate and "--gate-reserve-json" in gate and "p7_reserve_sessions.json" in gate, "the gate runs profile refe_p7 with --gate-reserve-json")
        ok(not any("--lane-rank" in c for c in calls) and not any("quarantine" in c for c in calls), "no twin is ever BUILT or quarantined")
        ok(any(c.startswith("p7_pins.py --check-run") and "p7_reserve_head" in c for c in calls), "p7_pins.py --check-run runs on P7's own run dir")
        tr = next((c for c in calls if c.startswith("train.py") and "--preflight" not in c and "--steps" not in c), "")
        ok("--label-workers 6" in tr and "--aux-drv" in tr and "--w-aux-drv 0.1" in tr and "--w-field 0.0" in tr and "p7_reserve_head" in tr and "p6_paper_faithful" not in tr and "refe_p6" not in tr, "the trainer argv: pinned workers, the head's levers named, P7's run dir")
        ok(all(not (s.data / "refe_runs" / "p6_paper_faithful").exists() for _ in [0]) and s.files_containing(T6.TOKEN.encode()) == [], "P6's run dir untouched; no token written to any file")

        def refused(name, extra=None, mutate=None, prep=None, why="", no_call=None, expect_rc=1):
            s2, rc2, out2, d2 = sb_run(extra, mutate, prep); dirs.append(d2)
            lg2 = s2.chain_log()
            good = rc2 == expect_rc and "REFUSED" in lg2 and (not why or why in lg2 + out2)
            if no_call:
                good = good and not any(c.startswith(no_call) for c in s2.calls_list())
            ok(good, f"REFUSAL {name} (rc {rc2}; why {why!r} in log: {why in lg2 + out2}; tail {lg2[-200:]!r})")
            return s2, rc2, out2

        # ---- (d) label workers pinned
        refused("label workers unset", {"P7_LABEL_WORKERS": ""}, why="P7_LABEL_WORKERS", no_call="pod_train")
        refused("label workers non-integer", {"P7_LABEL_WORKERS": "six"}, why="P7_LABEL_WORKERS", no_call="pod_train")
        refused("label workers outside 1..16", {"P7_LABEL_WORKERS": "13x"}, why="P7_LABEL_WORKERS", no_call="pod_train")
        # ---- (b) fallback
        refused("D7 fallback 1", {"P7_D7_FALLBACK": "1"}, why="does not exist in the P7 chain", no_call="pod_train")
        # ---- (a) twins
        s2, _, _ = refused("twin missing in the source", prep=lambda sb: os.remove(sb.twin_src / "L1" / "t1__lane_rank_1.pkl.xz"), why="100 % placeable", no_call="diag_p6_labeller")
        ok(not s2.stage("lanes") and not any("launch_gate.py run" in c for c in s2.calls_list()), "  ... and the gate never ran")
        refused("twins NOT distinct (built on the logged route)", {"STUB_TWIN_SAME": "1"}, prep=lambda sb: T6.make_parts_tree(sb.twin_src, [("L0", "t0"), ("L1", "t1")], twin_k=1, twin_same=True),
                why="NOT on their own route", no_call="launch_gate.py run")
        refused("rank set is not the registered one", {"P7_EXPECT_RANKS": "1 2 3"}, why="registered set", no_call="p7_prep.py twins")
        refused("twin source absent", prep=lambda sb: shutil.rmtree(sb.twin_src), why="twin-parts source", no_call="p6_prep.py reserve-verify")
        # ---- (e) local disk
        refused("metric cache / run dir on a FUSE volume", prep=lambda sb: (sb.root / "mounts").write_text("mfs#x / fuse.mfs rw 0 0\n"), why="ZZP7LOCAL_BAD", no_call="pod_train")
        # ---- the package
        refused("package incomplete (ship list)", {"STUB_SHIP_BAD": "1"}, why="incomplete or stale", no_call="pod_train")
        refused("a required package file missing", prep=lambda sb: os.remove(sb.pkg / "refe" / "test_p7_integration.py"), why="missing", no_call="pod_train")
        refused("the aux label store is not on the pod", prep=lambda sb: os.remove(sb.data / "refe_navtrain" / "aux_label_store_p7" / "manifest.json"), why="NOT ON THE POD", no_call="pod_train")
        # ---- isolation from P6
        refused("run dir == P6's", {"P7_RUN": "__WS__/data/refe_runs/p6_paper_faithful"}, why="P6's run directory", no_call="pod_train")
        refused("run dir INSIDE P6's", {"P7_RUN": "__WS__/data/refe_runs/p6_paper_faithful/x"}, why="P6's run directory", no_call="pod_train")
        refused("HF repo == P6's", {"P7_REPO": P6REPOID}, why="P6's HF repo", no_call="pod_train")
        refused("package == P6's", {"P7_PKG": "__WS__/refe-plan-p6"}, why="P6's package", no_call="pod_train")
        def p6_lock(sb):
            (sb.data / "refe_p6").mkdir(parents=True); (sb.data / "refe_p6" / "chain.lock").write_text("")      # the probe is `[ -e chain.lock ] && ! flock`: the lock FILE must exist
        refused("P6 chain still running", {"STUB_HELD": "refe_p6"}, prep=p6_lock, why="still holds", no_call="pod_train")
        refused("P7_WAIT_P6: P6 lock free but NO ZZP6_DONE", {"P7_WAIT_P6": "1"}, why="no ZZP6_DONE", no_call="pod_train")

        def p6_done(sb):
            (sb.data / "refe_p6").mkdir(parents=True); (sb.data / "refe_p6" / "chain.log").write_text("2026-11-20T00:00:00 ZZP6_DONE\n")
        s4, rc4, out4, d4 = sb_run({"P7_WAIT_P6": "1"}, prep=p6_done); dirs.append(d4)
        ok(done(s4, rc4, out4), f"P7_WAIT_P6=1 with ZZP6_DONE and P6's lock free proceeds (rc {rc4})")
        # ---- the reserve audit
        refused("reserve audit leak", {"STUB_AUDIT_LEAK": "1"}, why="audit FAILED", no_call="launch_gate.py run")
        # ---- the guards
        s5, rc5, out5 = refused("pins mismatch AFTER the first checkpoint: refuse and STOP the trainer", {"STUB_PINS_BAD": "1", "STUB_TRAIN_SLEEP": "25"}, why="ZZP7_PINS_REFUSED")
        ok(not (s5.run_dir / "summary.json").exists() and (s5.run_dir / "ckpt_last.pt").exists(), "  ... the trainer was stopped before it finished (no summary.json) and the refusal is final (one attempt)")
        ok(len([c for c in s5.calls_list() if c.startswith("train.py") and "--preflight" not in c and "--steps" not in c]) == 1, "  ... exactly one trainer launch (no retry after a pins refusal)")
        s6, rc6, out6 = refused("no_parts > 0 on the first labeller events: refuse and STOP the trainer", {"STUB_NOPARTS": "29", "STUB_TRAIN_SLEEP": "25"}, why="ZZP7_NOPARTS_REFUSED")
        ok(not (s6.run_dir / "summary.json").exists(), "  ... the trainer was stopped (no summary.json)")
        s7, rc7, out7, d7 = sb_run({"STUB_REAL_PINS": "1", "STUB_ID_SHA": "92af8d663ecdd04a1fe7a2ed5ec0995499ca87d48a01052da4868503a92d385f", "STUB_ID_N": "150603"}); dirs.append(d7)
        # the REAL p7_pins.py reads the REGISTERED pins of the bank named by config.json args.targets: the chain's bank is train_grow_cmd_3c_p7
        ok(done(s7, rc7, out7) and "ZZP7_PINS_OK" in s7.chain_log(), f"REAL p7_pins.py --check-run: the REGISTERED pins accepted (rc {rc7}; {s7.chain_log()[-250:]!r})")
        s8, rc8, out8 = refused("REAL p7_pins.py: a bank with another digest is refused", {"STUB_REAL_PINS": "1", "STUB_ID_SHA": "0" * 64, "STUB_ID_N": "150603", "STUB_TRAIN_SLEEP": "25"}, why="ZZP7_PINS_REFUSED")
        ok(not (s8.run_dir / "summary.json").exists(), "  ... (real pins) the trainer was stopped")

        # ============================================================ layer 4: behavioural mutants of the chain
        def mutant(name, fn, extra, expect):
            s9, rc9, out9, d9 = sb_run(extra, mutate=lambda t: fn(t)); dirs.append(d9)
            ok(expect(s9, rc9, out9), f"MUTANT {name} is caught by a layer-3 assertion (rc {rc9})")
        no_kill = lambda t: t.replace('kill_pid "$pid"; return 0\n      fi\n    fi\n    if [ "$np_done"', 'return 0\n      fi\n    fi\n    if [ "$np_done"', 1)
        mutant("pins refusal without the kill", no_kill, {"STUB_PINS_BAD": "1", "STUB_TRAIN_SLEEP": "6"}, lambda s9, r, o: (s9.run_dir / "summary.json").exists())
        mutant("fallback accepted", lambda t: t.replace("  0) ;;\n  *) refuse \"P7_D7_FALLBACK", "  0|1) ;;\n  *) refuse \"P7_D7_FALLBACK", 1), {"P7_D7_FALLBACK": "1"}, lambda s9, r, o: "does not exist" not in s9.chain_log())
        mutant("link stage dropped", lambda t: t.replace('"$PY" p7_prep.py twins apply', 'true', 1), {}, lambda s9, r, o: not any(c.startswith("p7_prep.py twins apply") for c in s9.calls_list()))
        mutant("label workers derived from the quota", lambda t: t.replace('LW="$P7_LABEL_WORKERS"; ', 'LW="$(( CORES - 2 ))"; ', 1), {"P7_LABEL_WORKERS": "6"},
               lambda s9, r, o: "--label-workers 6" not in " ".join(c for c in s9.calls_list() if c.startswith("train.py")))
        mutant("no_parts guard flipped to pass", lambda t: t.replace('*) : > "$P7/noparts.refused"', '*) : > "$P7/noparts.ignored"', 1), {"STUB_NOPARTS": "29", "STUB_TRAIN_SLEEP": "6", "P7_MAX_ATTEMPTS": "2"},
               lambda s9, r, o: "REFUSED: ZZP7_NOPARTS_REFUSED" not in s9.chain_log())
    finally:
        for dd in dirs:
            shutil.rmtree(dd, ignore_errors=True)

    if FAILS:
        print(f"ZZP7CHAIN_FAIL {len(FAILS)} failing, {NPASS} passing")
        return 1
    print(f"ZZP7CHAIN_PASS {NPASS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
