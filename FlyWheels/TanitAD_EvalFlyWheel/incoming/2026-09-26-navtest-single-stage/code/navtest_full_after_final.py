#!/usr/bin/env python3
"""AGENT-FREE WAITER: the FULL-split navtest one-stage floors, run only when the shared box is QUIET.

⛔ REWIRED 2026-09-26 (~18:50Z) — the refcv6 FINAL dependency is DROPPED. The PI STOPPED refcv6 at 20:40
Berlin (*"go with b, stop refcv6"*; D:/refcv6_eval_kit/ckpt_final/STOPPED_BY_PI.json: ckpt 38000, md5
5a2e7222…, and NO summary.json will be written). The FINAL chain therefore never runs and never writes
``ZZFINALV2ENDZZ`` — a waiter gated on it would wait until --max-wait-h and GIVE UP. Replaced, at the
Master Mind's request, by the box-quiet GPU gate: tonight's GPU queue is navhard@30k -> the PI's
agent-box video -> the battery's A6 -> these floors, and the gate holds the floors until that queue drains.

    cd D:/Projects/TanitAD/taniteval
    nohup C:/Users/Admin/venvs/tanitad/Scripts/python.exe \
        ../FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-26-navtest-single-stage/code/navtest_full_after_final.py \
        > ../FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-26-navtest-single-stage/raw/full_waiter.out 2>&1 &

W8, 2026-09-26. The brief: *"Tonight ~19:30 Berlin the refcv6 FINAL battery runs on this box and requires
the GPU and >= 8 GB free host RAM … It has priority … schedule the full-split floors to run AFTER the refcv6
FINAL completes."* This waiter enforces exactly that, then runs the pre-registered validation:

  1. WAIT until ALL hold AT ONCE (polled every --poll-s, and ALL re-read before EVERY run attempt):
       * the 12,146-token v2 cache is DONE (its CACHE_DONE.json certifies the yaml token set);
       * the GPU gate PASSES — the battery's own ``gpu_gate.py`` (the PI's box policy: GPU used < 4300 MiB,
         NO other python compute on the GPU, >= 8 GB free host RAM), so ONE definition decides "box quiet";
         an unreadable gate is a WAIT, never a pass (fail closed);
       * NOTHING QUEUED AHEAD is alive (``QUEUE_AHEAD``): ``run_battery``; the ``ev6_battery`` chain (tonight's
         A6 runs as a6_chain.sh -> a6_roll.py and NEVER names run_battery.py, so the old check could not see
         it); navhard@30k's ``run_navsim_refcv6``; any ``navsim_win.py`` scorer. An unreadable process table
         is a WAIT (its positive control: this process must appear in it);
       * RAM: >= --start-avail-mb (11500 = the 9000 floor + the scorer's own peak) available, SUSTAINED
         (5 samples, the perf counter); the run itself then aborts below --min-avail-mb (9000);
       (was: the FINAL chain had ENDED — dropped 2026-09-26, see above; ``final_ended()`` is kept for the record)
  2. ⛔ Why every gate is re-read together: FINAL-ended was a LATCH, so checking it once and then waiting only
     on RAM was sound. The GPU gate and the queue are NOT latches — a RAM wait that never looked back could
     start the floors hours later on a box that had filled up again (and it used to run even when its 24 h
     RAM wait had FAILED).
  3. ``python -m taniteval.bench navsim_v2 --ckpt none --split navtest_single_stage --arms CV,STOP,HUMAN
     --ram-floor-mb 9000`` — a RAM-guard abort is retried after the next RAM window (max --runs);
  4. on COMPLETE: E-T0 on the full cache, then ``code/cross_protocol_check.py --full`` (PREREG.md §3).

Every state change is a line in raw/full_waiter.log with an opaque ZZW8…ZZ marker (never the words a
monitor would grep for — CLAUDE.md, the self-matching monitor trap). The final record is
raw/full_waiter.json. ⛔ Assert on those ARTIFACTS, not on this process's exit status.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
REPO = PKG.parents[3]
TANITEVAL = REPO / "taniteval"
sys.path.insert(0, str(TANITEVAL))
from taniteval.bench.navsim import cache_build as CB          # noqa: E402  (the RAM gate)
from taniteval.bench.navsim import profiles as P              # noqa: E402

FINAL_LOG = Path("C:/Users/Admin/ev6_battery/raw/final_v2.log")      # no longer gates (refcv6 stopped by the PI)
#: the battery's gate, used by path so the "box quiet" rule has exactly one definition
GPU_GATE = (PKG.parents[3] / "FlyWheels" / "TanitAD_EvalFlyWheel" / "incoming" /
            "2026-09-23-refcv6-standard-tests" / "battery" / "code" / "gpu_gate.py")
TPY = Path("C:/Users/Admin/venvs/tanitad/Scripts/python.exe")
LOG = PKG / "raw" / "full_waiter.log"


def note(tag: str, **kw):
    line = json.dumps({"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "marker": f"ZZW8{tag}ZZ", **kw})
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def cache_done(prof) -> dict:
    mk = Path(prof.cache) / "CACHE_DONE.json"
    if not mk.exists():
        return {"ok": False, "why": f"{mk} absent"}
    m = json.loads(mk.read_text(encoding="utf-8"))
    return {"ok": bool(m.get("token_sets_equal_yaml") and m.get("scene_types_match_yaml")),
            "why": {k: m.get(k) for k in ("token_set", "n_cached", "token_sets_equal_yaml", "scene_types_match_yaml")}}


def final_ended() -> dict:
    if not FINAL_LOG.exists():
        return {"ok": False, "why": f"{FINAL_LOG} absent"}
    lines = FINAL_LOG.read_text(encoding="utf-8", errors="replace").splitlines()
    starts = [i for i, ln in enumerate(lines) if ln.startswith("ZZFINALV2STARTZZ")]
    ends = [i for i, ln in enumerate(lines) if ln.startswith("ZZFINALV2ENDZZ")]
    if not starts:
        return {"ok": False, "why": "no ZZFINALV2STARTZZ line yet"}
    ok = bool(ends and ends[-1] > starts[-1])
    return {"ok": ok, "why": (lines[ends[-1]] if ok else f"started {lines[starts[-1]]}; no END after it")}


def gpu_gate() -> dict:
    """The PI's box-quiet gate, by the battery's own script: exit 0 = PASS, 3 = WAIT.

    ⛔ FAIL CLOSED: a missing script, a timeout, or any other exit code is a WAIT, never a pass — a gate
    that cannot be read must not be the reason the floors start on a busy box."""
    if not GPU_GATE.exists():
        return {"ok": False, "why": f"gate script absent: {GPU_GATE}"}
    try:
        r = subprocess.run([str(TPY), str(GPU_GATE)], capture_output=True, text=True, timeout=180)
    except Exception as e:                                               # noqa: BLE001
        return {"ok": False, "why": f"gate unreadable: {type(e).__name__}: {e}"}
    last = (r.stdout.strip().splitlines() or [f"rc {r.returncode}, no output"])[-1][:400]
    return {"ok": r.returncode == 0, "rc": r.returncode, "why": last}


#: what must NOT be alive when the floors start, each labelled with the job it holds them for. Matched against
#: PYTHON processes only, so the PowerShell query that lists them can never match itself.
QUEUE_AHEAD = (
    ("battery (run_battery)", re.compile(r"run_battery")),
    # ⛔ tonight's battery never names run_battery.py: a6_chain.sh -> a6_roll.py / a6_fit_score.py / bank_tag.py,
    # and the gate wait between its rolls is ``python -c "import run_battery as RB"``. All carry the battery ROOT.
    ("battery (ev6_battery chain)", re.compile(r"ev6_battery", re.I)),
    ("navhard@30k (run_navsim_refcv6)", re.compile(r"run_navsim_refcv6")),
    # any scorer our NavSim harness launches: RAM-heavy, and the E1 RAM guard aborted navtest@30k's official
    # scorer SIX times (2026-09-26 14:12-14:46Z) while other jobs held the box's RAM
    ("NavSim scorer (navsim_win.py)", re.compile(r"navsim_win\.py")),
)


def classify_queue_ahead(procs, self_pids=()) -> list:
    """PURE: [(pid, cmdline)] -> the processes the floors must wait for, each labelled with its job."""
    out = []
    for pid, cmd in procs:
        if pid in self_pids or not cmd:
            continue
        hits = [label for label, rx in QUEUE_AHEAD if rx.search(cmd)]
        if hits:
            out.append({"pid": pid, "what": hits[0], "cmd": cmd[:160]})
    return out


def python_processes() -> list:
    """[(pid, cmdline)] for every live python process. Raises when the table cannot be read."""
    ps = ("[Console]::OutputEncoding = [Text.Encoding]::UTF8; "
          "Get-CimInstance Win32_Process | Where-Object { $_.Name -like 'python*' } | "
          "Select-Object ProcessId, CommandLine | ConvertTo-Json -Compress")
    r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
                       capture_output=True, timeout=120)
    if r.returncode != 0:
        raise RuntimeError(f"process query rc {r.returncode}")
    data = json.loads(r.stdout.decode("utf-8", errors="replace").strip() or "[]")
    data = [data] if isinstance(data, dict) else data
    return [(int(d["ProcessId"]), d.get("CommandLine") or "") for d in data]


def queue_ahead() -> list:
    """The live processes queued ahead of the floors. ⛔ FAIL CLOSED, with a POSITIVE control: this waiter is a
    python process, so a table that does not list its own pid was not read — an empty answer from a failed
    query must never read as "nothing ahead"."""
    try:
        procs = python_processes()
    except Exception as e:                                               # noqa: BLE001
        return [{"pid": -1, "what": f"process table unreadable: {type(e).__name__}: {e}"[:200], "cmd": ""}]
    if not any(pid == os.getpid() for pid, _ in procs):
        return [{"pid": -1, "what": f"process table unreadable: own pid absent ({len(procs)} rows)", "cmd": ""}]
    return classify_queue_ahead(procs, self_pids={os.getpid(), os.getppid()})


def gates(prof) -> dict:
    c, g, q = cache_done(prof), gpu_gate(), queue_ahead()
    return {"cache_done": c, "gpu_gate": g, "queue_ahead": q, "ok": bool(c["ok"] and g["ok"] and not q)}


def wait_until_quiet(a, prof, t0: float):
    """Block until the cache, the GPU gate, the queue AND a sustained RAM window all hold together.
    -> the gate record, or None when --max-wait-h elapsed first."""
    last = None
    while True:
        s, ram = gates(prof), None
        if s["ok"]:
            ram = CB.wait_for_ram(a.start_avail_mb, 5, 6.0, a.poll_s, log=lambda m: None)
            if ram["ok"]:
                s = gates(prof)                  # the RAM window took up to --poll-s: re-read everything after it
                if s["ok"]:
                    note("STATE", cache_done=s["cache_done"], gpu_gate=s["gpu_gate"], queue_ahead=s["queue_ahead"],
                         ram=ram, go=True)
                    return {**s, "ram": ram}
        key = (s["cache_done"]["ok"], s["gpu_gate"]["ok"], tuple(sorted(x["pid"] for x in s["queue_ahead"])),
               None if ram is None else ram["ok"])
        if key != last:
            note("STATE", cache_done=s["cache_done"], gpu_gate=s["gpu_gate"], queue_ahead=s["queue_ahead"], ram=ram)
            last = key
        if time.time() - t0 > a.max_wait_h * 3600:
            return None
        if ram is None:                          # a RAM wait already spent the poll interval
            time.sleep(a.poll_s)


def run_cli(a, reuse_from: str | None = None) -> dict:
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    cmd = [str(TPY), "-m", "taniteval.bench", "navsim_v2", "--ckpt", "none", "--split", "navtest_single_stage",
           "--arms", "CV,STOP,HUMAN", "--ram-floor-mb", str(a.min_avail_mb)]
    if reuse_from:                       # arms that PASSED in the previous attempt are adopted, identity-checked
        cmd += ["--reuse-scored-arms", reuse_from]
    lp = PKG / "raw" / f"full_cli_{time.strftime('%Y%m%dT%H%M%S')}.log"
    with open(lp, "w", encoding="utf-8") as fh:
        rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=str(TANITEVAL)).returncode
    text = lp.read_text(encoding="utf-8", errors="replace")
    rd = re.findall(r"BENCH_RUN_DIR=(\S+)", text)
    st = re.findall(r"BENCH_STATUS=(\S+)", text)
    rt = re.findall(r"BENCH_RETRYABLE=(\d)", text)
    return {"rc": rc, "log": str(lp), "run_dir": rd[-1] if rd else None, "status": st[-1] if st else None,
            "retryable": bool(rt and rt[-1] == "1"), "cmd": cmd}


V1_TREE = "D:/Archive/devbox-C/navsim/navsim-3e8291bfa89ff247231e0227778840cd0a036896"
V1_CACHE = "D:/Archive/devbox-C/navsim/exp/w3_navtest_v1/metric_cache_navtest"


def _navsim(args: list, *, v1: bool, log_name: str) -> dict:
    """Run a package script in the NavSim venv (v1 tree on PYTHONPATH when v1=True). Asserts on the artifact."""
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "OMP_NUM_THREADS": "2",
           "PYTHONPATH": (V1_TREE if v1 else str(P.DEVKIT_SIDE).replace(os.sep, "/")),
           "NUPLAN_MAP_VERSION": "nuplan-maps-v1.0",
           "NUPLAN_MAPS_ROOT": ("D:/Archive/devbox-C/navsim/data/maps" if v1 else "C:/Users/Admin/navsim-crun/data/maps")}
    lp = PKG / "raw" / f"{log_name}.log"
    with open(lp, "w", encoding="utf-8") as fh:
        rc = subprocess.run([str(P.PY)] + [str(x) for x in args], stdout=fh, stderr=subprocess.STDOUT, env=env,
                            cwd=str(PKG)).returncode
    return {"rc": rc, "log": str(lp)}


def post_process(run_dir: str, cache, *, tag: str = "full", tokens_file=None, full: bool = True) -> dict:
    """PREREG §3 + AMENDMENT A2. Each step writes its own JSON (suffix ``tag``); the verdict reads them.
    ``tokens_file`` restricts the digest/classification (the pipeline test on a smoke); ``full`` = the
    full-split bars (H-HIGH, K-NEG >= 1000)."""
    raw, code, out = PKG / "raw", HERE, {}
    tf = ["--tokens-file", tokens_file] if tokens_file else []
    out["e_t0"] = _navsim([code / "e_t0_from_cache.py", "--cache", cache, "--out", raw / f"e_t0_{tag}.json"],
                          v1=False, log_name=f"e_t0_{tag}")
    note("ET0", **out["e_t0"], exists=(raw / f"e_t0_{tag}.json").exists())
    out["digest_v1"] = _navsim([code / "e_obs_digest.py", "--cache", V1_CACHE, "--side", "v1",
                                "--out", raw / f"e_obs_v1_{tag}.json"] + tf, v1=True, log_name=f"e_obs_v1_{tag}")
    note("DIGESTV1", **out["digest_v1"])
    out["classify"] = _navsim([code / "e_obs_classify.py", "--v1-cache", V1_CACHE, "--v2-cache", cache,
                               "--control-v1-digest", raw / f"e_obs_v1_{tag}.json", "--out", raw / f"e_obs_class_{tag}.json"] + tf,
                              v1=False, log_name=f"e_obs_class_{tag}")
    note("CLASSIFY", **out["classify"])
    xc = raw / f"cross_protocol_{tag}.json"
    r = subprocess.run([str(TPY), str(code / "cross_protocol_check.py"), "--run", run_dir, "--e-t0", str(raw / f"e_t0_{tag}.json"),
                        "--e-obs-class", str(raw / f"e_obs_class_{tag}.json"), "--out", str(xc)] + (["--full"] if full else []),
                       capture_output=True, text=True, env={**os.environ, "PYTHONUTF8": "1"})
    out["xcheck"] = {"rc": r.returncode, "exists": xc.exists(), "stdout": r.stdout[-1500:]}
    note("XCHECK", rc=r.returncode, exists=xc.exists())
    for arm in ("CV", "STOP", "HUMAN"):
        o = raw / f"counterfactual_{tag}_{arm}.json"
        out[f"cf_{arm}"] = _navsim([code / "counterfactual_v2fn_v1inputs.py", "--run", run_dir, "--arm", arm,
                                    "--v1-cache", V1_CACHE, "--v2-cache", cache, "--tokens-json", xc, "--out", o],
                                   v1=False, log_name=f"counterfactual_{tag}_{arm}")
        note("CF", arm=arm, **out[f"cf_{arm}"], exists=o.exists())
    # ---- the A2 verdict, read from the artifacts (never from exit codes)
    v = {}
    try:
        x = json.loads(xc.read_text(encoding="utf-8"))
        v.update(x.get("verdict", {}))
        for arm in ("CV", "STOP", "HUMAN"):
            c = json.loads((raw / f"counterfactual_{tag}_{arm}.json").read_text(encoding="utf-8"))
            cf, ctl = c["CF_v2_function_on_v1_inputs_vs_v1_csv"], c["CONTROL_reproduces_run_csv"]
            v[f"C-NC-FN[{arm}]"] = bool(c["n_tokens"] > 0 and c["n_failures"] == 0 and cf["NC_mismatch"] == 0
                                        and cf["DAC_mismatch"] == 0 and cf["TTC_below_v1"] == 0
                                        and ctl["NC_mismatch"] == 0 and ctl["DAC_mismatch"] == 0)
    except Exception as e:                                               # noqa: BLE001
        v["VERDICT_ASSEMBLY"] = False
        out["verdict_error"] = f"{type(e).__name__}: {e}"
    a2_required = [k for k in v if not (k.startswith("C-NC[") or k.startswith("C-TTC["))]
    out["verdict"] = v
    out["validated_under_A2"] = bool(v) and all(v[k] for k in a2_required)
    out["original_C_NC_C_TTC"] = {k: v[k] for k in v if k.startswith("C-NC[") or k.startswith("C-TTC[")}
    out["tag"], out["full_split_bars"] = tag, full
    (raw / f"verdict_{tag}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    note("VERDICT", validated_under_A2=out["validated_under_A2"])
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--poll-s", type=float, default=300.0)
    ap.add_argument("--max-wait-h", type=float, default=72.0)
    ap.add_argument("--min-avail-mb", type=float, default=9000.0)
    ap.add_argument("--runs", type=int, default=6)
    ap.add_argument("--start-avail-mb", type=float, default=11500.0,
                    help="pre-start gate for the full CLI run: the floor + the scorer's own ESTIMATED peak")
    ap.add_argument("--post-only", default=None, help="run ONLY the post-processing on this run dir (pipeline test)")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--tokens-file", default=None)
    ap.add_argument("--tag", default="full")
    ap.add_argument("--gate-once", action="store_true",
                    help="read every start gate ONCE, print it, run nothing and write no log (verification)")
    a = ap.parse_args(argv)
    prof = P.SPLITS["navtest_single_stage"]
    if a.gate_once:
        print(json.dumps({"pid": os.getpid(), **gates(prof)}, indent=1))
        return 0
    if a.post_only:
        global LOG
        LOG = PKG / "raw" / f"waiter_{a.tag}.log"          # a pipeline test never writes the real waiter's log
        post = post_process(a.post_only, a.cache or prof.cache, tag=a.tag, tokens_file=a.tokens_file,
                            full=(a.tag == "full"))
        print(json.dumps({"validated_under_A2": post.get("validated_under_A2"), "verdict": post.get("verdict")}, indent=1))
        return 0
    rec = {"schema": "w8-full-waiter/1", "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "pid": os.getpid(),
           "rule": ("cache DONE AND the GPU gate PASSES AND nothing queued ahead is alive (run_battery, the "
                    "ev6_battery chain, navhard@30k run_navsim_refcv6, any navsim_win.py scorer) AND a sustained "
                    "RAM window -- all re-read together before EVERY run attempt"),
           "rewired": ("2026-09-26: the refcv6 FINAL dependency was DROPPED -- the PI stopped refcv6 at 20:40 "
                       "Berlin and no summary.json will be written, so the FINAL chain can never end")}
    note("START", pid=os.getpid(), rule=rec["rule"])
    t0 = time.time()
    runs = []
    for i in range(a.runs):
        # ⚠ the START gate is the floor PLUS the scorer's own peak (ESTIMATED ~2.5 GB for the full split; W3's v1
        # full-split analog MEASURED 2.2 GB): the wrapper's guard counts OUR footprint too, so starting at exactly
        # the floor would make the run yield on itself. The wrapper's abort floor stays at --min-avail-mb.
        g = wait_until_quiet(a, prof, t0)
        if g is None:
            rec.update(status=(runs[-1].get("status") if runs else None) or "NOT_RUN", runs=runs,
                       reason=f"max wait elapsed before {'the first run' if not runs else f'retry {i + 1}'}")
            (PKG / "raw" / "full_waiter.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
            note("GAVEUP")
            return 2
        note("RAMOK", gate=g["ram"])
        prev = next((x["run_dir"] for x in reversed(runs) if x.get("run_dir")), None)
        r = run_cli(a, reuse_from=prev)
        runs.append(r)
        note("RUN", **{k: r[k] for k in ("rc", "status", "run_dir", "retryable")})
        if r["status"] == "COMPLETE" or not r["retryable"]:
            break
    rec["runs"] = runs
    last_run = runs[-1] if runs else {}
    if last_run.get("status") == "COMPLETE" and last_run.get("run_dir"):
        post = post_process(last_run["run_dir"], prof.cache, tag="full", full=True)
        rec["post"] = post
    rec["ended"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    rec["status"] = last_run.get("status") or "NOT_RUN"
    (PKG / "raw" / "full_waiter.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    note("END", status=rec["status"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
