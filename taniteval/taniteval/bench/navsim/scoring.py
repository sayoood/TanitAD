"""Score ONE arm with the OFFICIAL, UNMODIFIED NavSim two-stage runner — PROMOTED from E2
``code/score_arm.py`` (EvalFlyWheel 2026-09-19), which itself drove E1's wrapper.

The runner ``navsim.planning.script.run_pdm_score`` runs in the NavSim venv as its own process
with Hydra OVERRIDES only (no devkit file is edited), THROUGH the promoted E1 wrapper
``devkit_side/navsim_win.py --patch-loader`` (Windows loader patch + observation-only hooks + RAM
guard). A run that did no work is REFUSED (E2's guards, unchanged):

* the log must report ``successful == expected`` and ``failed == 0``;
* the CSV must hold exactly ``expected`` valid token rows + the 3 summary rows;
* a seam arm's agent call log must show one call per token with the declared seam/stand-in split.

"Success over an empty set" is a FAILURE.
"""
from __future__ import annotations

import csv
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from . import profiles as P


def overrides_for(prof: "P.SplitProfile", *, exp_name: str, worker: str, tokens: list | None = None,
                  token_log: dict | None = None) -> list:
    """The Hydra overrides that pin the split, cache, logs and (single-stage) traffic policy.

    ⭐ W8 2026-09-26 — a SINGLE-STAGE split: no synthetic_* paths (it has none), the log directory named
    EXPLICITLY (the C: default holds only navhard's 76 logs and the devkit skips an absent log in
    silence, dataloader.py:34-36), and ``traffic_agents`` passed EXPLICITLY although it is the runner's
    default (default_common.yaml:22) — a default is a claim about the config file, the argv is a claim
    about the run. A token SUBSET adds log_names/tokens (never for a two-stage split)."""
    if prof.stages == 1:
        if not prof.traffic_agents:
            raise P.Refusal(f"{prof.name}: a single-stage profile must name its traffic_agents policy")
        ov = [f"train_test_split={prof.tts}", f"experiment_name={exp_name}",
              f"metric_cache_path={_fwd(prof.cache)}", f"navsim_log_path={_fwd(prof.logs_dir)}",
              f"traffic_agents={prof.traffic_agents}", f"worker={worker}"]
        if tokens is not None:
            if not token_log:
                raise P.Refusal("a token subset needs its token -> log_name map")
            logs = sorted({token_log[t] for t in tokens})
            ov.append("train_test_split.scene_filter.log_names=[" + ",".join(f"'{x}'" for x in logs) + "]")
            ov.append("train_test_split.scene_filter.tokens=[" + ",".join(f"'{x}'" for x in sorted(tokens)) + "]")
        return ov
    if tokens is not None:
        raise P.Refusal(f"{prof.name}: a token subset is only defined on a single-stage split")
    return [f"train_test_split={prof.name}", f"experiment_name={exp_name}",
            f"metric_cache_path={_fwd(prof.cache)}", f"synthetic_sensor_path={_fwd(prof.syn_sensors)}",
            f"synthetic_scenes_path={_fwd(prof.syn_scenes)}", f"worker={worker}"]


def score_arm(*, arm: str, prof: "P.SplitProfile", raw_dir: Path, exp_dir: Path, seam: Path | None = None,
              official_agent: str | None = None, worker: str = "sequential", ram_floor_mb: float = 3000.0,
              python: Path = P.PY, log=print, baseline_blobs: dict | None = None,
              tokens: list | None = None, token_log: dict | None = None) -> dict:
    """-> the counts/guard record (E2's ``score_<arm>.counts.json`` shape) with ``status`` PASS|FAIL
    and ``csv`` = the devkit CSV (copied UNMODIFIED into ``raw_dir``).

    The runner is the one the split's stage count REQUIRES (``profiles.runner_for``; W8 2026-09-26:
    ``pdm_score_one_stage`` for a single-stage split). ``tokens`` = a single-stage SUBSET."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(exp_dir) / arm
    call_log = raw_dir / f"{arm}.calls.jsonl"
    if call_log.exists():
        call_log.unlink()
    exp_name = f"tanitad_bench_{prof.name}_{arm}"
    script = P.check_runner(prof, P.runner_for(prof))
    ov = overrides_for(prof, exp_name=exp_name, worker=worker, tokens=tokens, token_log=token_log)
    if official_agent:
        ov.append(f"agent={official_agent}")
    else:
        if seam is None or not Path(seam).exists():
            raise P.Refusal(f"{arm}: seam file missing: {seam!r}")
        ov += ["agent=constant_velocity_agent",
               "agent._target_=tanitad_seam_agent.TanitADSeamAgent",
               f"+agent.seam_file={_fwd(Path(seam).resolve())}",
               f"+agent.call_log={_fwd(call_log.resolve())}"]
    ov.append(f"output_dir={_fwd(out_dir)}")
    preagg = raw_dir / f"{arm}_preaggregation.csv"
    cmd = [str(python), str(P.WRAPPER), "--script", script, "--label", arm, "--out-dir", str(raw_dir),
           "--patch-loader", "--ram-floor-mb", str(ram_floor_mb), "--dump-final-scores",
           str(raw_dir / f"{arm}_final_scores_frame.csv"),
           "--dump-preaggregation", str(preagg), "--"] + ov
    wrap_sha = hashlib.sha256(P.WRAPPER.read_bytes()).hexdigest()
    # ⭐ PER-ARM PROVENANCE (orchestrator ruling, 2026-09-20). bench_run.json's `suite_code_blobs`
    # is a RUN-START snapshot, but every devkit-side file is loaded by a FRESH SUBPROCESS PER ARM, so
    # a file changed mid-run is recorded truthfully for arm 1 and FALSELY for arm 2. These blobs are
    # read HERE, microseconds before the launch, and in the SAME hash space (git blob over RAW bytes)
    # so they are directly comparable to the snapshot. ⚠ `wrapper_sha256` below is a SHA-256 and is
    # NOT comparable to it -- it is kept unchanged because it is E2's banked record shape.
    launch_blobs = _devkit_side_blobs()
    drift = devkit_side_drift(launch_blobs, baseline_blobs)
    if drift:
        log(f"⚠ [navsim] {arm}: devkit-side code CHANGED since this run started: {', '.join(drift)} "
            f"— recorded per arm in {arm}.counts.json (devkit_side_drift)")
    log_path = raw_dir / f"{arm}.score.log"
    Path(exp_dir).mkdir(parents=True, exist_ok=True)
    env = P.scorer_env(Path(exp_dir))
    log(f"[navsim] scoring {arm} ({'official ' + official_agent if official_agent else 'seam'}) on {prof.name} …")
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as fh:
        fh.write("CMD: " + " ".join(cmd) + "\n")
        fh.flush()
        rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=str(exp_dir)).returncode
    wall = time.time() - t0
    text = log_path.read_text(encoding="utf-8", errors="replace")
    m_ok = re.findall(r"Number of successful scenarios:\s*(\d+)", text)
    m_bad = re.findall(r"Number of failed scenarios:\s*(\d+)", text)
    m_res = re.findall(r"Results are stored in:\s*(\S+\.csv)", text)
    expected = len(tokens) if tokens is not None else prof.n_stage1 + prof.n_stage2
    shape = P.summary_row_shape(prof.protocol)              # the PROTOCOL declares its summary rows
    fails = []
    n_ok = int(m_ok[-1]) if m_ok else None
    n_bad = int(m_bad[-1]) if m_bad else None
    if n_ok is None:
        fails.append("summary line 'Number of successful scenarios' ABSENT — cannot confirm work")
    elif n_ok != expected:
        fails.append(f"successful {n_ok} != expected {expected}")
    if n_bad not in (0,):
        fails.append(f"failed scenarios = {n_bad}")
    csv_src = m_res[-1].rstrip(".") if m_res else None
    if csv_src is None:
        cands = sorted(glob.glob(str(out_dir / "*.csv")), key=os.path.getmtime)
        csv_src = cands[-1] if cands else None
    n_rows = n_valid = None
    csv_dst = raw_dir / f"{arm}.devkit.csv"
    if csv_src and os.path.exists(csv_src):
        shutil.copyfile(csv_src, csv_dst)
        with open(csv_dst, encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        # ⭐ W8: a token row is any row that is not one of the protocol's DECLARED summary rows (and not a
        # two-stage summary row either — a one-stage CSV carrying one is a wrong-runner CSV, caught below)
        tok_rows = [r for r in rows if r["token"] not in shape["rows"]
                    and not r["token"].startswith("extended_pdm_score") and r["token"] != "average_all_frames"]
        n_rows = len(tok_rows)
        n_valid = sum(1 for r in tok_rows if r["valid"] in ("True", "true", "1"))
        if n_rows != expected or n_valid != expected:
            fails.append(f"CSV token rows {n_rows} / valid {n_valid} != expected {expected}")
        summ = [r["token"] for r in rows if r["token"] in shape["rows"]]
        if len(summ) != shape["n"]:
            fails.append(f"CSV summary rows {summ} != the {prof.protocol} shape {list(shape['rows'])}")
        foreign = sorted({r["token"] for r in rows if (r["token"].startswith("extended_pdm_score")
                                                       or r["token"] == "average_all_frames")
                          and r["token"] not in shape["rows"]})
        if foreign:
            fails.append(f"CSV carries summary rows of ANOTHER protocol {foreign} — wrong runner for {prof.protocol}")
        if tokens is not None:
            got = {r["token"] for r in tok_rows}
            if got != set(tokens):
                fails.append(f"CSV token set != the requested subset ({len(got - set(tokens))} extra, "
                             f"{len(set(tokens) - got)} missing)")
    else:
        fails.append("no result CSV found")
    calls = None
    if not official_agent:
        calls = {"seam": 0, "cv_standin": 0, "UNKNOWN_TOKEN": 0}
        if call_log.exists():
            for ln in open(call_log, encoding="utf-8"):
                r = json.loads(ln)
                if r.get("event") == "call":
                    calls[r["source"]] = calls.get(r["source"], 0) + 1
        z = np.load(seam, allow_pickle=False)
        src = [str(x) for x in z["source"]]
        want = {"seam": sum(1 for s in src if s != "cv_standin"),
                "cv_standin": sum(1 for s in src if s == "cv_standin")}
        bad = {k: v for k, v in calls.items() if k not in ("seam", "cv_standin") and v}
        if bad or calls["seam"] != want["seam"] or calls["cv_standin"] != want["cv_standin"]:
            fails.append(f"agent calls {calls} != seam declaration {want}")
    for extra in sorted(glob.glob(str(out_dir / "*.log"))):          # the runner's own log
        shutil.copyfile(extra, raw_dir / f"{arm}.{Path(extra).name}")
    # ⛔ "the expensive part finished and the AGGREGATION died" is NOT a failed run (MEASURED
    # 2026-09-19: navhard CV scored all 5,912 scenarios in 68 min, then create_scene_aggregators ->
    # calculate_individual_mapping_scores raised on NaN; exit 1, NO CSV). The per-token rows the
    # parent held are banked by the wrapper's pre-aggregation dump, so the run is classified
    # AGGREGATION_FAILED with its evidence, and can be re-aggregated (reaggregate.py).
    preagg_rows = None
    if preagg.exists():
        # ⛔ W8 2026-09-26, MEASURED on the navtest sub200 smoke: the dump carries `ego_simulated_states`
        # (numpy arrays whose repr spans lines inside quoted cells), so a PHYSICAL line count read 24,600
        # for 200 records, the AGGREGATION_FAILED classification never fired, and a fully scored arm was
        # reported as a plain FAIL. Count CSV RECORDS, never lines.
        csv.field_size_limit(1 << 30)
        with open(preagg, encoding="utf-8", newline="") as fh:
            preagg_rows = sum(1 for _ in csv.DictReader(fh))
    # rc 3 = the wrapper's RAM guard (available memory below the floor). The run did NOT fail on its
    # own terms — it was never allowed to finish — so it is RETRYABLE, and the waiter retries it
    # rather than a human reading a red "FAIL" as a defect in the harness.
    ram_guard_abort = (rc == 3)
    aggregation_failed = bool(fails and csv_dst.exists() is False and preagg_rows == expected
                              and not ram_guard_abort)
    if aggregation_failed:
        fails = [f for f in fails if not f.startswith("no result CSV")]
        fails.append(f"AGGREGATION FAILED after all {expected} scenarios were scored — the per-token rows are "
                     f"banked in {preagg.name} ({preagg_rows} rows); re-aggregate with "
                     f"taniteval/taniteval/bench/navsim/devkit_side/reaggregate.py")
    rep = {"arm": arm, "split": prof.name, "cmd": cmd, "rc": rc, "wall_s": round(wall, 1),
           "runner_script": script, "stages": prof.stages, "traffic_agents": prof.traffic_agents,
           "logs_dir": _fwd(prof.logs_dir), "n_tokens_subset": (len(tokens) if tokens is not None else None),
           "preaggregation_csv": str(preagg) if preagg.exists() else None, "preaggregation_rows": preagg_rows,
           "expected_tokens": expected, "log_successful": n_ok, "log_failed": n_bad,
           "csv_src": csv_src, "csv": str(csv_dst) if csv_dst.exists() else None,
           "csv_token_rows": n_rows, "csv_valid_rows": n_valid, "agent_calls": calls,
           "cache": _fwd(prof.cache), "worker": worker, "wrapper": _fwd(P.WRAPPER), "wrapper_sha256": wrap_sha,
           "devkit_side_blobs_at_launch": launch_blobs,
           "devkit_side_drift": drift,
           "devkit_side_blob_basis": ("git blob over RAW bytes, read at THIS arm's launch; comparable to "
                                      "bench_run.json.git.suite_code_blobs. An empty drift list means the "
                                      "devkit-side code this arm ran IS the code the run recorded at start."),
           "runtime": _fwd(P.CR), "env_fixed": P.scorer_env_fixed(Path(exp_dir)),
           "status": ("RAM_GUARD_ABORT" if ram_guard_abort else
                      ("AGGREGATION_FAILED" if aggregation_failed else ("FAIL" if fails else "PASS"))),
           "retryable": bool(ram_guard_abort),
           "scoring_complete": bool(aggregation_failed or not fails), "failures": fails,
           "_promoted_from": "FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-navsim-refcv4b-bridge/code/score_arm.py"}
    (raw_dir / f"{arm}.counts.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    log(f"[navsim] {arm}: {rep['status']} successful={n_ok} failed={n_bad} rows={n_valid} wall={wall:.0f}s"
        + (f" failures={fails}" if fails else ""))
    return rep


def devkit_side_drift(launch_blobs: dict, baseline_blobs: dict | None) -> list:
    """Files whose blob at THIS arm's launch differs from the run-start snapshot.

    ⛔ A path ABSENT from the baseline is NOT drift. ``suite_code_blobs`` globs ``*.py`` only, so
    PROVENANCE.json is legitimately missing from it, and reporting that as drift would cry wolf on
    every single arm -- which is how a real warning gets ignored.
    """
    base = baseline_blobs or {}
    return sorted(k for k, v in launch_blobs.items() if base.get(k) not in (None, v))


def _devkit_side_blobs() -> dict:
    """git blob id (RAW bytes, no eol conversion) of every devkit-side file, keyed by repo-relative
    path so it lines up with ``suite_code_blobs``.

    ⚠ RAW BYTES ON PURPOSE. ``git hash-object`` applies the text/eol conversion, so on a CRLF file
    it returns a DIFFERENT id for the same bytes -- MEASURED 2026-09-20 on the IDM patch, where the
    normalising hash read 214ef5ee… and the raw-byte hash b95bcc7f…. The normalising one is also
    BLIND to a pure line-ending change, which is exactly the drift this function exists to catch.
    """
    from ..contract import code_blobs
    return code_blobs(sorted(p for p in P.DEVKIT_SIDE.rglob("*")
                             if p.is_file() and "__pycache__" not in p.parts))


def _fwd(p) -> str:
    return str(p).replace(os.sep, "/")
