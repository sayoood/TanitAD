"""Build the NavSim **v2** METRIC CACHE of a SINGLE-STAGE split, one LOG GROUP per process, RAM-gated.

    cd taniteval
    python -m taniteval.bench.navsim.cache_build --split navtest_single_stage --out <raw dir>
    python -m taniteval.bench.navsim.cache_build --split navtest_single_stage --out <raw dir> \
        --tokens-file <json> --cache C:/Users/Admin/navsim-crun/exp/metric_cache_navtest_v2_smoke200

W8 (EvalFlyWheel, 2026-09-26). PROMOTED — restructured, same mechanism — from W3's
``2026-09-19-navsim-v1-navtest/code/run_v1.py::cmd_cache`` (per-log groups, RAM wait, resume) and
``code/write_cache_metadata.py`` (the metadata CSV written by the DEVKIT's own writer from a
directory scan), re-pointed from the v1.1 tree at the v2 devkit (0a380a9) through the suite's
promoted E1 wrapper (``devkit_side/navsim_win.py --script metric_caching``). The devkit is not edited.

⛔ WHY PER LOG GROUP (W3, MEASURED 2026-09-20 on the same 136 navtest logs): ``cache_data`` builds ONE
SceneLoader over EVERY log before it caches anything (2.1 GB RSS, ~2 min), and a sibling's memory spike
then killed the whole pass. One small group per process needs ~0.3-0.8 GB, is RESUMABLE (a finished log
is skipped by counting its pickles), and a spike costs ONE group. Only ``scene_filter.log_names`` is
overridden per group.

⛔ WHY THE METADATA CSV IS WRITTEN AT THE END: every devkit run ends with ``save_cache_metadata`` into
the SAME file name (``<cache>/metadata/<cache.name>_metadata_node_0.csv``), so after N group runs the CSV
names only the LAST group. It is re-written from the pickles on disk by the devkit's own writer and
verified on CONTENT (``devkit_side/write_cache_metadata.py``), which also writes ``CACHE_DONE.json`` and
the per-file sha256 ``CACHE_MANIFEST.json``. Until then the suite's preflight REFUSES the cache (its
row count is wrong) — fail closed, never a partial cache read as a whole one.

⛔ THE BOX IS SHARED (brief 2026-09-26: the refcv6 FINAL battery needs >= 8 GB free host RAM and has
priority). A group is started only after ``--sustain`` consecutive samples of
``Win32_PerfFormattedData_PerfOS_Memory.AvailableMBytes`` all read >= ``--min-avail-mb`` (default 9000),
and the wrapper is launched with ``--ram-floor-mb`` = the same floor, so a running group YIELDS
(exit 3) after 2 min sustained below it; the group is retried after the next RAM window.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from . import profiles as P

WRITE_META = P.DEVKIT_SIDE / "write_cache_metadata.py"


# --------------------------------------------------------------------------- #
# RAM gate                                                                     #
# --------------------------------------------------------------------------- #
def avail_mb() -> tuple:
    """-> (MB or None, source). The Windows perf counter the brief names; psutil only as a labelled
    fallback (the wrapper's own guard reads psutil — GlobalMemoryStatusEx ullAvailPhys, same pages)."""
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command",
                            "(Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory).AvailableMBytes"],
                           capture_output=True, text=True, timeout=90)
        s = r.stdout.strip()
        if re.fullmatch(r"\d+", s):
            return float(s), "Win32_PerfFormattedData_PerfOS_Memory.AvailableMBytes"
    except Exception:                                                    # noqa: BLE001
        pass
    try:
        import psutil
        return psutil.virtual_memory().available / 2**20, "psutil.virtual_memory().available (FALLBACK)"
    except Exception:                                                    # noqa: BLE001
        return None, "UNREADABLE"


def wait_for_ram(min_mb: float, sustain: int, every_s: float, max_wait_s: float, log=print) -> dict:
    """Block until ``sustain`` CONSECUTIVE samples read >= ``min_mb``. An unreadable sample counts as a
    FAILED sample (fail closed). -> record; ``ok`` False when ``max_wait_s`` elapsed first."""
    t0, run, last, n = time.time(), 0, None, 0
    samples = []
    while True:
        v, src = avail_mb()
        n += 1
        samples.append(None if v is None else round(v))
        samples = samples[-sustain * 3:]
        run = run + 1 if (v is not None and v >= min_mb) else 0
        if run >= sustain:
            return {"ok": True, "waited_s": round(time.time() - t0, 1), "samples_tail": samples[-sustain:],
                    "source": src, "min_mb": min_mb, "sustain": sustain}
        if time.time() - t0 > max_wait_s:
            return {"ok": False, "waited_s": round(time.time() - t0, 1), "samples_tail": samples,
                    "source": src, "min_mb": min_mb, "sustain": sustain}
        if v is not None and v < min_mb and (last is None or time.time() - last > 600):
            log(json.dumps({"event": "RAM_WAIT", "avail_mb": round(v), "floor_mb": min_mb, "source": src}))
            last = time.time()
        time.sleep(every_s)


# --------------------------------------------------------------------------- #
def cached_tokens_by_log(cache: Path) -> dict:
    """{log_name: {token, ...}} of the ``metric_cache.pkl`` files on disk (<cache>/<log>/<type>/<token>/)."""
    out = collections.defaultdict(set)
    if not cache.is_dir():
        return {}
    for p in cache.glob("*/*/*/metric_cache.pkl"):
        out[p.parts[-4]].add(p.parts[-2])
    return dict(out)


def _hydra_list(xs) -> str:
    return "[" + ",".join(f"'{x}'" for x in xs) + "]"


def plan(prof: "P.SplitProfile", tokens: list | None, cache: Path, logs_per_process: int) -> dict:
    y = P.read_split_yaml(prof.name)
    t2l = P.token_to_log(prof)
    toks = list(tokens) if tokens is not None else list(y["stage_one"])
    foreign = sorted(set(toks) - set(y["stage_one"]))
    if foreign:
        raise P.Refusal(f"{len(foreign)} token(s) are not {prof.tts} tokens (first {foreign[0]})")
    per_log = collections.defaultdict(list)
    for t in toks:
        per_log[t2l[t]].append(t)
    logs = [ln for ln in y["log_names"] if ln in per_log]              # the devkit yaml's order
    missing = [ln for ln in logs if not (prof.logs_dir / f"{ln}.pkl").exists()]
    if missing:
        raise P.Refusal(f"{len(missing)}/{len(logs)} log pickles missing under {prof.logs_dir} (first {missing[0]}) — "
                        "the devkit would SILENTLY skip them (dataloader.py:34-36)")
    groups = [logs[i:i + logs_per_process] for i in range(0, len(logs), logs_per_process)]
    return {"tokens": toks, "per_log": {k: sorted(v) for k, v in per_log.items()}, "logs": logs, "groups": groups,
            "subset": tokens is not None, "cache": cache}


def group_cmd(prof, cache: Path, grp: list, label: str, out_dir: Path, min_avail_mb: float,
              subset_tokens: list | None) -> list:
    ov = [f"train_test_split={prof.tts}", f"metric_cache_path={str(cache).replace(os.sep, '/')}",
          f"navsim_log_path={str(prof.logs_dir).replace(os.sep, '/')}", "worker=sequential",
          f"train_test_split.scene_filter.log_names={_hydra_list(grp)}"]
    if subset_tokens is not None:
        ov.append(f"train_test_split.scene_filter.tokens={_hydra_list(subset_tokens)}")
    return [str(P.PY), str(P.WRAPPER), "--script", "metric_caching", "--label", label, "--out-dir", str(out_dir),
            "--ram-floor-mb", str(min_avail_mb), "--"] + ov


def run_groups(prof, pl: dict, out: Path, label: str, a, log=print) -> list:
    cache: Path = pl["cache"]
    gdir = out / "groups"
    gdir.mkdir(parents=True, exist_ok=True)
    reps = []
    prog = out / "cache_build_progress.jsonl"
    for i, grp in enumerate(pl["groups"]):
        want = {ln: set(pl["per_log"][ln]) for ln in grp}
        have = cached_tokens_by_log(cache)
        if all(want[ln] <= have.get(ln, set()) for ln in grp):
            rec = {"group": i, "logs": grp, "status": "SKIPPED_COMPLETE", "n_expected": sum(len(v) for v in want.values())}
            reps.append(rec)
            continue
        sub = sorted(t for ln in grp for t in want[ln]) if pl["subset"] else None
        tries, yields, rc, gate, wall = 0, 0, None, None, 0.0
        # ⭐ a RAM-guard YIELD (rc 3) is the box being busy, not the group failing: it does not consume a
        # try (MEASURED 2026-09-26: other sessions held the box at ~7 GB available and group 0 yielded
        # twice in 10 min). Only non-yield failures count toward --tries; yields are capped separately.
        while tries < a.tries and yields <= a.max_yields:
            gate = wait_for_ram(a.min_avail_mb, a.sustain, a.sample_every_s, a.max_wait_min * 60, log=log)
            if not gate["ok"]:
                rc = "NO_RAM_WINDOW"
                break
            attempt = tries + yields + 1
            lab = f"{label}_g{i:03d}" + (f"_try{attempt}" if attempt > 1 else "")
            cmd = group_cmd(prof, cache, grp, lab, gdir, a.min_avail_mb, sub)
            lp = gdir / f"{lab}.log"
            t0 = time.time()
            with open(lp, "w", encoding="utf-8") as fh:
                fh.write("CMD: " + json.dumps(cmd) + "\n")
                fh.flush()
                rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=P.scorer_env(P.EXP_ROOT),
                                    cwd=str(gdir)).returncode
            wall = time.time() - t0
            if rc != 3:                                  # 3 = the wrapper's RAM guard: YIELDED -> wait, retry
                tries += 1
                if rc == 0:
                    break
                log(json.dumps({"event": "GROUP_FAILED", "group": i, "rc": rc, "try": tries}))
                continue
            yields += 1
            log(json.dumps({"event": "YIELDED_RAM_GUARD", "group": i, "yield": yields}))
        have = cached_tokens_by_log(cache)
        n_have = sum(len(want[ln] & have.get(ln, set())) for ln in grp)
        n_want = sum(len(v) for v in want.values())
        rec = {"group": i, "logs": grp, "rc": rc, "tries": tries, "yields": yields, "wall_s": round(wall, 1),
               "n_expected": n_want, "n_cached_after": n_have, "ram_gate": gate,
               "status": "OK" if (rc == 0 and n_have == n_want) else "INCOMPLETE",
               "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        reps.append(rec)
        with open(prog, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        log(json.dumps({k: rec[k] for k in ("group", "status", "rc", "tries", "wall_s", "n_expected", "n_cached_after")}))
        if rc == "NO_RAM_WINDOW":
            log("⛔ no RAM window within --max-wait-min; stopping (resumable: re-run the same command)")
            break
    return reps


def finalize(prof, pl: dict, out: Path, a, log=print) -> dict:
    """Write the metadata CSV (devkit's writer) + CACHE_DONE.json + CACHE_MANIFEST.json, verified on content,
    in the NavSim venv."""
    cache: Path = pl["cache"]
    tok_file = out / "expected_tokens.json"
    tok_file.write_text(json.dumps({"split": prof.tts, "subset": pl["subset"], "tokens": sorted(pl["tokens"])}),
                        encoding="utf-8")
    cmd = [str(P.PY), str(WRITE_META), "--cache", str(cache), "--expected-tokens", str(tok_file),
           "--split", prof.tts, "--logs-dir", str(prof.logs_dir), "--out-json", str(out / "finalize.json"),
           "--load", a.load]
    lp = out / "finalize.log"
    t0 = time.time()
    with open(lp, "w", encoding="utf-8") as fh:
        fh.write("CMD: " + json.dumps(cmd) + "\n")
        fh.flush()
        rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=P.scorer_env(P.EXP_ROOT),
                            cwd=str(out)).returncode
    res = {"rc": rc, "wall_s": round(time.time() - t0, 1), "log": str(lp)}
    fj = out / "finalize.json"
    if fj.exists():
        res["finalize"] = json.loads(fj.read_text(encoding="utf-8"))
    log(json.dumps({"event": "FINALIZE", "rc": rc, "PASS": (res.get("finalize") or {}).get("PASS")}))
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True, type=Path, help="where the build records go (repo package raw/)")
    ap.add_argument("--tokens-file", default=None, help="a SUBSET of the split (needs --cache)")
    ap.add_argument("--cache", default=None, type=Path, help="cache dir (default: the profile's)")
    ap.add_argument("--label", default="cache")
    ap.add_argument("--logs-per-process", type=int, default=4)
    ap.add_argument("--min-avail-mb", type=float, default=9000.0)
    ap.add_argument("--sustain", type=int, default=5, help="consecutive perf-counter samples >= the floor")
    ap.add_argument("--sample-every-s", type=float, default=6.0)
    ap.add_argument("--max-wait-min", type=float, default=720.0)
    ap.add_argument("--tries", type=int, default=4, help="non-yield failures per group")
    ap.add_argument("--max-yields", type=int, default=200, help="RAM-guard yields per group before giving up")
    ap.add_argument("--load", choices=("all", "sample", "none"), default="all",
                    help="finalize: unpickle every entry (all), every 25th (sample), or none")
    ap.add_argument("--finalize-only", action="store_true")
    a = ap.parse_args(argv)
    prof = P.SPLITS.get(a.split)
    if prof is None or prof.stages != 1:
        print(f"⛔ REFUSED: {a.split!r} is not a single-stage split of the suite "
              f"({sorted(k for k, v in P.SPLITS.items() if v.stages == 1)}); two-stage caches are E1's recipe")
        return 2
    tokens = P.read_tokens_file(a.tokens_file) if a.tokens_file else None
    cache = Path(a.cache) if a.cache else prof.cache
    if tokens is not None and cache.resolve() == Path(prof.cache).resolve():
        print(f"⛔ REFUSED: a token SUBSET may not be written into the split's canonical cache {prof.cache} "
              "(pass --cache <another dir>)")
        return 2
    a.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    pl = plan(prof, tokens, cache, a.logs_per_process)
    rec = {"schema": "w8-cache-build/1", "split": prof.name, "devkit_split": prof.tts, "cache": str(cache).replace(os.sep, "/"),
           "logs_dir": str(prof.logs_dir).replace(os.sep, "/"), "subset": pl["subset"], "n_tokens": len(pl["tokens"]),
           "n_logs": len(pl["logs"]), "n_groups": len(pl["groups"]), "logs_per_process": a.logs_per_process,
           "ram_gate": {"min_avail_mb": a.min_avail_mb, "sustain": a.sustain, "every_s": a.sample_every_s},
           "devkit_sha_pin": P.DEVKIT_SHA, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "argv": sys.argv}
    (a.out / "cache_build_plan.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(json.dumps({k: rec[k] for k in ("split", "cache", "n_tokens", "n_logs", "n_groups")}), flush=True)
    reps = [] if a.finalize_only else run_groups(prof, pl, a.out, a.label, a, log=lambda m: print(m, flush=True))
    have = cached_tokens_by_log(cache)
    n_have = sum(len(set(v) & have.get(ln, set())) for ln, v in pl["per_log"].items())
    rec.update({"groups": reps, "n_cached": n_have, "complete": n_have == len(pl["tokens"])})
    if rec["complete"]:
        rec["finalize"] = finalize(prof, pl, a.out, a, log=lambda m: print(m, flush=True))
    rec["wall_s"] = round(time.time() - t0, 1)
    rec["ended_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    ok = bool(rec["complete"] and ((rec.get("finalize") or {}).get("finalize") or {}).get("PASS"))
    rec["status"] = "PASS" if ok else ("INCOMPLETE" if not rec["complete"] else "FINALIZE_FAILED")
    (a.out / "cache_build.json").write_text(json.dumps(rec, indent=1, default=str), encoding="utf-8")
    print(f"CACHE_BUILD_STATUS={rec['status']} cached={n_have}/{len(pl['tokens'])} wall_s={rec['wall_s']}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
