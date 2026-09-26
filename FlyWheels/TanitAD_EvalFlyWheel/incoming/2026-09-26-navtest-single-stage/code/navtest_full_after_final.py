#!/usr/bin/env python3
"""AGENT-FREE WAITER: the FULL-split navtest one-stage floors, run only AFTER the refcv6 FINAL battery.

    cd D:/Projects/TanitAD/taniteval
    nohup C:/Users/Admin/venvs/tanitad/Scripts/python.exe \
        ../FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-26-navtest-single-stage/code/navtest_full_after_final.py \
        > ../FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-26-navtest-single-stage/raw/full_waiter.out 2>&1 &

W8, 2026-09-26. The brief: *"Tonight ~19:30 Berlin the refcv6 FINAL battery runs on this box and requires
the GPU and >= 8 GB free host RAM … It has priority … schedule the full-split floors to run AFTER the refcv6
FINAL completes."* This waiter enforces exactly that, then runs the pre-registered validation:

  1. WAIT until ALL hold (polled every --poll-s):
       * the 12,146-token v2 cache is DONE (its CACHE_DONE.json certifies the yaml token set);
       * the FINAL chain has ENDED: ``ZZFINALV2ENDZZ`` appears in C:/Users/Admin/ev6_battery/raw/final_v2.log
         AFTER the latest ``ZZFINALV2STARTZZ`` (chain_final_v2.sh writes it last, whether or not the final ran);
       * NO ``run_battery.py`` process is alive (any tag — the box is shared);
  2. RAM gate: >= --min-avail-mb (9000) available, sustained (5 samples, the perf counter);
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

FINAL_LOG = Path("C:/Users/Admin/ev6_battery/raw/final_v2.log")
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


def battery_alive() -> list:
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command",
                            "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'run_battery\\.py' } | "
                            "ForEach-Object { $_.ProcessId }"], capture_output=True, text=True, timeout=120)
        return [int(x) for x in r.stdout.split() if x.strip().isdigit()]
    except Exception as e:                                               # noqa: BLE001
        return [-1]                                                      # unreadable -> treat as alive (fail closed)


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
    a = ap.parse_args(argv)
    prof = P.SPLITS["navtest_single_stage"]
    if a.post_only:
        global LOG
        LOG = PKG / "raw" / f"waiter_{a.tag}.log"          # a pipeline test never writes the real waiter's log
        post = post_process(a.post_only, a.cache or prof.cache, tag=a.tag, tokens_file=a.tokens_file,
                            full=(a.tag == "full"))
        print(json.dumps({"validated_under_A2": post.get("validated_under_A2"), "verdict": post.get("verdict")}, indent=1))
        return 0
    rec = {"schema": "w8-full-waiter/1", "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "pid": os.getpid(),
           "rule": "cache DONE AND FINAL chain ended AND no run_battery.py alive, then a sustained RAM window"}
    note("START", pid=os.getpid())
    t0 = time.time()
    last = None
    while True:
        c, f, b = cache_done(prof), final_ended(), battery_alive()
        state = (c["ok"], f["ok"], not b)
        if state != last:
            note("STATE", cache_done=c, final_ended=f, battery_pids=b)
            last = state
        if all(state):
            break
        if time.time() - t0 > a.max_wait_h * 3600:
            rec.update(status="NOT_RUN", reason="max wait elapsed", cache=c, final=f, battery=b)
            (PKG / "raw" / "full_waiter.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
            note("GAVEUP")
            return 2
        time.sleep(a.poll_s)
    runs = []
    for i in range(a.runs):
        # ⚠ the START gate is the floor PLUS the scorer's own peak (ESTIMATED ~2.5 GB for the full split; W3's v1
        # full-split analog MEASURED 2.2 GB): the wrapper's guard counts OUR footprint too, so starting at exactly
        # the floor would make the run yield on itself. The wrapper's abort floor stays at --min-avail-mb.
        g = CB.wait_for_ram(a.start_avail_mb, 5, 6.0, 24 * 3600, log=lambda m: note("RAMWAIT", detail=m))
        note("RAMOK", gate=g)
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
