"""``navsim_v1`` — NAVSIM **v1.1** PDMS on ``navtest`` (W3, EvalFlyWheel, 2026-09-19/20).

    python -m taniteval.bench navsim_v1 --ckpt none --split navtest --arms CV,STOP,HUMAN

⛔ PROTOCOL ``PDMS_v1_navtest`` — the v1 metric from the **v1.1 code**
(``autonomousvision/navsim`` v1.1 branch @ ``3e8291bfa89ff247231e0227778840cd0a036896``,
unpacked at ``D:/Archive/devbox-C/navsim/navsim-3e8291b…``). ``main`` (0a380a9) is v2 and computes
EPDMS; a "v1 number" from it is a different metric (``NAVSIM_PROTOCOL.md`` §1 branch discipline).
PDMS = ``NC·DAC·(5·EP + 5·TTC + 2·C)/12``; DDC has weight 0; the headline column is ``score``
(v1.1 has no ``pdm_score`` column at all). navtest is **SINGLE-STAGE**: 12,146 tokens over 136
logs, so ``per_stage`` carries one block and says so.

WHAT THIS PLUGIN DOES — AND WHAT IT REUSES
------------------------------------------
Every devkit step runs through W3's package
``FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-navsim-v1-navtest``: ``code/run_v1.py``
(driver + count guards C1/C2/C3/C9) around ``code/navsim_v1_win.py`` (the unmodified v1.1
entrypoint imported by PYTHONPATH, E1's loader patch, the RAM guard, the per-token JSONL stream
that survives an aggregation crash). Floors: ``STOP`` = ``code/w3_agents_v1.py::StopAgent``
(all-zero plan), ``CV`` = the devkit's own ``ConstantVelocityAgent``, ``HUMAN`` = the devkit's
own privileged ``HumanAgent``. The interval is **W2's registered log-cluster bootstrap**
(``taniteval/adapters/navsim_ci.py``, ``PDMS_v1_navtest`` → single-stage nanmean over tokens,
clusters = the 136 OpenScene logs ≥ the RG-14 floor of 8), paired deltas likewise.

⚠️ THE CACHE IS THE EXPENSIVE PART. The metric cache (12,146 scenes, D: only — PI decision) is
built once, ~4–5 h at one worker, and REUSED by every arm. If it is absent this plugin refuses
with the exact command rather than starting a 5 h job inside a CLI call.

⚠️ MODEL ARMS (refcv4b …) need the navtest FRAME BANK and a GPU gap; they are REFUSED here with
the queue command until both exist (PI 2026-09-19: our-model inference waits for a GPU gap).
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
W3 = REPO / "FlyWheels" / "TanitAD_EvalFlyWheel" / "incoming" / "2026-09-19-navsim-v1-navtest"
V11_TREE = "D:/Archive/devbox-C/navsim/navsim-3e8291bfa89ff247231e0227778840cd0a036896"
V11_SHA = "3e8291bfa89ff247231e0227778840cd0a036896"
EXP = Path("D:/Archive/devbox-C/navsim/exp/w3_navtest_v1")
CACHE_NAME = "navtest"
CLUSTER_MAP = (REPO / "FlyWheels/TanitAD_EvalFlyWheel/incoming/"
               "2026-09-19-navsim-estimator-and-route-leak/raw/cluster_maps/navtest.json")
NV_PY = "C:/Users/Admin/navsim-crun/venv/Scripts/python.exe"
TIE = 1e-12
#: handed to W1's gate so its refusal text carries the way to queue this run
QUEUE_HINT = ("code/run_bridge_navtest.py --device auto --gpu-gap  then  code/run_v1.py score "
              "--arm SEAM:<npz> --cache-name navtest --patch-loader  (W3 package)")

#: SPEC §4 (pre-registered): arXiv 2406.15349v2 Tab. 1 (p. 7) and Tab. 3 (p. 9). ×100.
PUBLISHED = {
    "_source": {"library_key": "2406.15349", "version": "arXiv v2 (31 Oct 2024)",
                "sha256": "d3bc66d321cfceccc4f431113dea3f0f481d74d29dd283a33e204a4326f0403c",
                "tables": "Table 1 (navtest benchmark, p. 7); Table 3 (NAVSIM 1.1 leaderboard, p. 9)",
                "evidence_class": "PUBLISHED (banked primary, re-read 2026-09-19)"},
    "CV": {"NC": 68.0, "DAC": 57.8, "TTC": 50.0, "C": 100.0, "EP": 19.4, "PDMS": 20.6},
    "HUMAN": {"NC": 100.0, "DAC": 100.0, "TTC": 100.0, "C": 99.9, "EP": 87.5, "PDMS": 94.8},
    "EGO_MLP": {"NC": 93.0, "DAC": 77.3, "TTC": 83.6, "C": 100.0, "EP": 62.8, "PDMS": 65.6,
                "PDMS_table3_3seeds": "66.4 ± 0.9"},
    "_leaderboard_INHERITED": {"CV": 20.6517, "EGO_MLP": "66.3989 ± 0.9406 (3 seeds)",
                               "source": "NAVSIM_PROTOCOL.md §6.1, HF navtest LB retrieved 2026-08-23"},
}
ARMS = {
    "CV": {"kind": "floor", "runner": "CV", "declared_inputs": ["ego_velocity[t0]"],
           "what": "the devkit's ConstantVelocityAgent — the published lower bound (20.6)"},
    "STOP": {"kind": "floor", "runner": "STOP", "declared_inputs": [],
             "what": "w3_agents_v1.StopAgent: 8 x (0,0,0). Reads NOTHING. The do-nothing floor"},
    "HUMAN": {"kind": "reference", "runner": "HUMAN",
              "declared_inputs": ["logged future trajectory (PRIVILEGED, label-side only)"],
              "what": "the devkit's HumanAgent — the published ceiling (94.8)"},
}
#: the devkit class each arm MUST have been scored with — half of the reuse key
#: (orchestrator ruling 2026-09-20 (a): an arm can never inherit another's CSV)
AGENT_CLASS = {"CV": "navsim.agents.constant_velocity_agent.ConstantVelocityAgent",
               "HUMAN": "navsim.agents.human_agent.HumanAgent",
               "STOP": "w3_agents_v1.StopAgent"}
TERM_COL = {"NC": "no_at_fault_collisions", "DAC": "drivable_area_compliance",
            "EP": "ego_progress", "TTC": "time_to_collision_within_bound", "C": "comfort",
            "DDC": "driving_direction_compliance"}


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _rows(csv_path: Path) -> tuple:
    import csv as _csv
    rows = list(_csv.DictReader(open(csv_path, encoding="utf-8")))
    tok = {r["token"]: r for r in rows if r["token"] != "average" and r["valid"] in ("True", "true", "1")}
    avg = next((r for r in rows if r["token"] == "average"), None)
    return tok, avg


def _mean(tok: dict, col: str) -> float:
    return sum(float(r[col]) for r in tok.values()) / len(tok)


def _round_half_up(x: float, dp: int) -> float:
    from decimal import ROUND_HALF_UP, Decimal
    return float(Decimal(repr(x)).quantize(Decimal(1).scaleb(-dp), rounding=ROUND_HALF_UP))


def _trunc(x: float, dp: int) -> float:
    import math
    f = 10 ** dp
    return math.trunc(x * f) / f


def _verdict(got_x100: float, want_x100: float, dp: int) -> dict:
    """SPEC §4 + **AMENDMENT A1 (2026-09-20, before any navtest CSV existed)**.

    ⛔ The published table's printed-precision convention is UNSETTLED for this paper
    (`raw/print_convention_probe.json`): its one internal test implies ROUNDING, the INHERITED
    leaderboard cells split 3 ROUNDING / 1 TRUNCATION, and E1 MEASURED that the sibling paper
    [N2] TRUNCATES. So the measured value is read BOTH ways and a cell is REPRODUCED only when
    BOTH readings give the published digits; one-sided agreement is reported as
    ``REPRODUCED_UNDER_ROUNDING`` / ``REPRODUCED_UNDER_TRUNCATION`` with the other reading printed
    beside it — never as a bare REPRODUCED, and never chosen for being the flattering one."""
    d = got_x100 - want_x100
    r, t = _round_half_up(got_x100, dp), _trunc(got_x100, dp)
    pub = _round_half_up(want_x100, dp)
    if r == pub and t == pub:
        v = "REPRODUCED"
    elif r == pub:
        v = "REPRODUCED_UNDER_ROUNDING"
    elif t == pub:
        v = "REPRODUCED_UNDER_TRUNCATION"
    elif abs(d) <= 0.5:
        v = "CLOSE"
    else:
        v = "NOT REPRODUCED"
    return {"got_x100": round(got_x100, 4), "published_x100": want_x100, "delta": round(d, 4),
            "printed_dp": dp, "reading_rounded": r, "reading_truncated": t, "verdict": v,
            "_convention": ("UNSETTLED for arXiv 2406.15349v2 — both readings are reported "
                            "(SPEC AMENDMENT A1; raw/print_convention_probe.json)")}


def _paired(a_tok: dict, b_tok: dict, clusters: dict, name_a: str, name_b: str) -> dict:
    common = sorted(set(a_tok) & set(b_tok))
    if not common:
        return {"status": "UNAVAILABLE", "reason": "no common tokens", "n": 0}
    d = [float(a_tok[t]["score"]) - float(b_tok[t]["score"]) for t in common]
    out = {"status": "OK", "headline_delta": sum(d) / len(d), "n_common": len(common),
           "headline_delta_x100": round(100 * sum(d) / len(d), 4),
           "wins": sum(x > TIE for x in d), "ties": sum(abs(x) <= TIE for x in d),
           "losses": sum(x < -TIE for x in d),
           "submetric_mean_deltas_x100": {
               k: round(100 * (_mean({t: a_tok[t] for t in common}, c)
                               - _mean({t: b_tok[t] for t in common}, c)), 4)
               for k, c in TERM_COL.items()}}
    try:
        sys.path.insert(0, str(REPO / "taniteval"))
        from adapters import navsim_ci as CI
        ca = CI.single_stage_contributions([float(a_tok[t]["score"]) for t in common])
        cb = CI.single_stage_contributions([float(b_tok[t]["score"]) for t in common])
        out["interval"] = CI.paired_log_cluster_bootstrap(
            ca, cb, [clusters[t] for t in common], aggregation=CI.AGG_SINGLE_STAGE)
    except Exception as e:                                            # noqa: BLE001
        out["interval"] = {"status": "UNAVAILABLE", "n": len(common),
                           "reason": f"paired log-cluster bootstrap failed: {type(e).__name__}: {e}"[:300]}
    return out


def device_decision(args, log=print) -> dict:
    """The suite-wide device gate for MODEL inference (orchestrator arbitration 2026-09-20).

    * ``--device auto|cuda``  -> the GPU-gap launcher decides/waits (W1's ``GpuGapLauncher``);
    * ``--device cpu``        -> ALLOWED when no training process is alive;
    * ``--device cpu`` with a training process alive -> **REFUSED** unless the operator passes
      ``--accept-training-box-load`` (or, until W1's CLI carries that flag,
      ``W3_ACCEPT_TRAINING_BOX_LOAD=1``); the flag AND the training process it accepted are
      recorded here, and the core folds them into ``bench_run.json: device.gpu_gap``.

    The probe is W1's, never a second implementation: if ``taniteval.bench.gpu_gap`` exports a
    named shared gate it is CALLED; otherwise the decision is composed from that module's own
    ``probe_training_pids``, and ``gate_source`` says which."""
    from taniteval.bench import gpu_gap as GG
    req = getattr(args, "device", "auto")
    flagged = bool(getattr(args, "accept_training_box_load", False))
    accept = flagged or os.environ.get("W3_ACCEPT_TRAINING_BOX_LOAD", "") == "1"
    for nm in ("device_gate", "cpu_gate", "check_cpu_box", "acquire_cpu", "cpu_box_gate"):
        fn = getattr(GG, nm, None)
        if not callable(fn):
            continue
        out = fn(req, accept_training_box_load=accept, queue_hint=QUEUE_HINT)
        if not isinstance(out, dict):
            continue
        # W1's vocabulary is {CUDA, CPU, REFUSE} + `record` (what bench_run.json must carry);
        # W3 adds only the two keys its own callers read, and never overwrites W1's.
        out.setdefault("gate_source", "taniteval.bench.gpu_gap." + nm)
        out.setdefault("allowed", str(out.get("decision", "")).upper() in ("CPU", "CUDA"))
        out.setdefault("requested", req)
        out.setdefault("accept_training_box_load", accept)
        out.setdefault("accept_via", ("--accept-training-box-load" if flagged else
                                      ("env W3_ACCEPT_TRAINING_BOX_LOAD=1" if accept else None)))
        log("[navsim_v1] device gate (" + out["gate_source"] + "): " + str(out.get("decision")))
        return out
    rec = {"requested": req,
           "gate_source": ("composed from taniteval.bench.gpu_gap.probe_training_pids "
                           "(no named shared gate exported yet)"),
           "rule": ("auto|cuda -> WAIT for a GPU gap; cpu -> allowed only when no training "
                    "process is alive, else REFUSED unless --accept-training-box-load names "
                    "the override"),
           "accept_training_box_load": accept,
           "accept_via": ("--accept-training-box-load" if flagged
                          else ("env W3_ACCEPT_TRAINING_BOX_LOAD=1" if accept else None))}
    if req in ("auto", "cuda"):
        rec.update(decision="GPU_GAP", allowed=True,
                   note="the GpuGapLauncher waits for: no training process AND GPU memory < 1 GB")
        return rec
    pids = list(GG.probe_training_pids())
    procs = []
    try:
        import psutil
        for pid in pids:
            try:
                procs.append({"pid": pid, "cmdline": " ".join(psutil.Process(pid).cmdline())[:200]})
            except Exception:                                          # noqa: BLE001
                procs.append({"pid": pid, "cmdline": "UNREADABLE"})
    except Exception:                                                  # noqa: BLE001
        procs = [{"pid": p} for p in pids]
    rec["training_processes"] = procs
    if not pids:
        rec.update(decision="CPU_ALLOWED", allowed=True, note="no training process alive")
    elif accept:
        rec.update(decision="CPU_ACCEPTED_TRAINING_BOX_LOAD", allowed=True,
                   note=("the operator accepted CPU inference beside a live training process; "
                         "the process is recorded above"))
    else:
        rec.update(decision="REFUSED_TRAINING_ALIVE", allowed=False,
                   reason=("--device cpu while training is alive: pids "
                           + str([p["pid"] for p in procs])
                           + ". Pass --accept-training-box-load (or W3_ACCEPT_TRAINING_BOX_LOAD=1) "
                             "to accept the load, or --device auto to wait for a GPU gap."))
    log("[navsim_v1] device gate: " + str(rec.get("decision")))
    return rec


def recheck_csv(csv_path, expected_tokens: set) -> dict:
    """Re-run the count guards ON THIS FILE — a passed run is evidence about the moment it ran,
    not about the file now (orchestrator ruling 2026-09-20 (b)). v1.1 CSVs carry exactly ONE
    summary row (``average``): the three ``extended_pdm_score_*`` rows are the v2 shape."""
    import csv as _csv
    import hashlib
    rows = list(_csv.DictReader(open(csv_path, encoding="utf-8")))
    tok = [r for r in rows if r["token"] != "average"]
    summ = [r for r in rows if r["token"] == "average"]
    valid = [r for r in tok if r["valid"] in ("True", "true", "1")]
    got = {r["token"] for r in tok}
    dmax = 0.0
    for r in valid:
        f = (float(r["no_at_fault_collisions"]) * float(r["drivable_area_compliance"])
             * (5 * float(r["ego_progress"]) + 5 * float(r["time_to_collision_within_bound"])
                + 2 * float(r["comfort"])) / 12.0)
        dmax = max(dmax, abs(f - float(r["score"])))
    out = {"file": str(csv_path).replace(os.sep, "/"),
           "sha256": hashlib.sha256(open(csv_path, "rb").read()).hexdigest(),
           "n_token_rows": len(tok), "n_valid": len(valid), "n_summary_rows": len(summ),
           "n_expected": len(expected_tokens), "n_missing": len(expected_tokens - got),
           "n_unexpected": len(got - expected_tokens),
           "C1_pdms_formula_max_abs_delta": dmax,
           "_summary_row_shape": ("PDMS_v1: exactly 1 `average` row; v2's 3 extended_pdm_score_* "
                                  "rows do not exist in v1.1")}
    out["ok"] = bool(out["n_token_rows"] == out["n_valid"] == len(expected_tokens)
                     and out["n_summary_rows"] == 1 and not out["n_missing"]
                     and not out["n_unexpected"] and dmax <= 1e-12)
    return out


def run_benchmark(ctx) -> None:                                        # noqa: C901
    a = ctx.args
    run, log = ctx.run, ctx.log
    if a.split != "navtest":
        raise SystemExit(f"⛔ navsim_v1 supports only split 'navtest' (got {a.split!r}); "
                         f"PDMS_v1 is defined on navtest (SPEC §2)")
    run_v1 = _load("w3_run_v1", W3 / "code" / "run_v1.py")
    logs, toks = run_v1.navtest_tokens()
    clusters = json.load(open(CLUSTER_MAP, encoding="utf-8"))["token_to_log_name"]

    ctx.set_protocol("PDMS_v1_navtest")
    ctx.set_split("navtest", len(toks), len(logs),
                  scene_filter=str(run_v1.NAVTEST_YAML),
                  single_stage=True,
                  note="navtest is single-stage: every token is an ORIGINAL log frame")
    ctx.set_devkit("autonomousvision/navsim", V11_SHA, patches=[
        {"name": "MetricCacheLoader._load_metric_cache_paths: separator-agnostic token "
                 "(E1's function, imported; navsim/common/dataloader.py:180 splits on '/' and "
                 "nuPlan writes str(WindowsPath))", "kind": "in-process",
         "control": "W3 C7: the unpatched loader raises IndexError on the real v1.1 cache"}],
        branch="v1.1", tree=V11_TREE, runtime_venv="C:/Users/Admin/navsim-crun/venv (PYTHONPATH -> v1.1)",
        nuplan_devkit="ce3c323af01c0d7ec5672f7832ef53f9c679aab0 (tag nuplan-devkit-v1.2)",
        wrapper="FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-navsim-v1-navtest/code/navsim_v1_win.py")
    ctx.set_stamps("T1-family", {
        "single_stage": "OPEN (one query, the plan is then fixed and propagated by LQR + kinematic "
                        "bicycle at 10 Hz over 4 s, no environmental feedback)",
        "background_traffic": "NON-REACTIVE: logged actors replay their recorded futures "
                              "(v1.1 docs/metrics.md) — unlike v2, which is IDM-reactive"},
        "MEASURED", closed_loop=False)
    ctx.set_claim_bearing(True)

    requested = [x for x in (a.arms or []) if x]
    for f in ("STOP", "CV"):
        if f not in requested:
            requested.append(f)
    known = [x for x in requested if x in ARMS]
    model_arms = [x for x in requested if x not in ARMS]
    for name in known:
        ctx.add_arm(name, ARMS[name]["kind"], ARMS[name]["declared_inputs"],
                    added_by_rule=name not in (a.arms or []), status="FAILED",
                    what=ARMS[name]["what"])
    for name in model_arms:
        ctx.add_arm(name, "model", ["frames(CAM_F0/L0/R0 stitch)", "ego t0", "nav command"],
                    status="REFUSED",
                    refusal=("model arms on navtest need the navtest FRAME BANK "
                             "(code/build_navtest_frames.py, 32 shards); the DEVICE is then "
                             "decided by the suite gate (auto|cuda wait for a GPU gap; cpu only "
                             "with no training alive, or with --accept-training-box-load) — see "
                             "raw/device_gate.json. Queue: code/run_bridge_navtest.py then "
                             "run_v1.py score --arm SEAM:<npz>"))

    gate = device_decision(a, log)
    ctx.rec["device_detail"] = gate
    if model_arms:
        dec = str(gate.get("decision", "")).upper()
        ctx.rec["device_used_override"] = ("cuda" if dec == "CUDA" else
                                           ("cpu" if dec.startswith("CPU") else None))
    run.write_json("raw/device_gate.json", gate)
    cache = EXP / f"metric_cache_{CACHE_NAME}"
    # ⛔ the METADATA CSV is what MetricCacheLoader reads — a metadata DIRECTORY proves nothing
    # (MEASURED 2026-09-20: an aborted pass leaves the hydra output dir with no CSV at all).
    meta_csvs = sorted((cache / "metadata").glob("*.csv")) if (cache / "metadata").is_dir() else []
    meta_rows = 0
    if len(meta_csvs) == 1:
        meta_rows = max(0, sum(1 for _ in open(meta_csvs[0], encoding="utf-8")) - 1)
    # the pickle walk costs ~60 s at 12k token dirs on exFAT (MEASURED), and it only tells you
    # something the CSV does not while the cache is still being built.
    if meta_rows == len(toks):
        n_cached, n_cached_source = meta_rows, "metadata csv (complete — the walk was skipped)"
    else:
        n_cached = sum(1 for _, _, fs in os.walk(cache) for f in fs if f == "metric_cache.pkl")
        n_cached_source = "os.walk over the cache (the metadata csv is incomplete)"
    banked = {n: (W3 / "raw" / f"{n}_navtest" / f"{n}_navtest.counts.json").exists() for n in known}
    if getattr(a, "dry_run", False):
        plan = {"arms": known, "model_arms_refused": model_arms, "n_scenes": len(toks),
                "n_logs": len(logs), "metric_cache": str(cache).replace(os.sep, "/"),
                "metric_cache_metadata_csvs": [str(x) for x in meta_csvs],
                "metric_cache_metadata_rows": meta_rows,
                "n_metric_cache_pkl": n_cached, "n_metric_cache_pkl_source": n_cached_source,
                "banked_w3_runs": banked,
                "cluster_map": str(CLUSTER_MAP), "cluster_map_tokens": len(clusters),
                "devkit": V11_TREE, "protocol": "PDMS_v1_navtest"}
        run.write_json("raw/plan.json", plan)
        for n in known:
            ctx.arm_rec(n)["status"] = "SKIPPED"
        ctx.rec["status"] = "REFUSED"
        ctx.rec["dry_run"] = True
        ctx.rec["refusal"] = ("DRY RUN: preflight only, nothing scored — "
                              f"{n_cached}/{len(toks)} scenes in the metric cache")
        log("[navsim_v1] " + ctx.rec["refusal"])
        return
    if len(meta_csvs) != 1 or meta_rows != len(toks):
        ctx.rec["status"] = "REFUSED"
        ctx.rec["refusal"] = (
            f"the navtest metric cache at {cache} is not complete: {len(meta_csvs)} metadata csv(s), "
            f"{meta_rows} rows, {n_cached} pickles, expected {len(toks)} (D: only — PI). "
            f"Build it once (~4-5 h, ONE "
            f"worker): PYTHONPATH={V11_TREE} {NV_PY} code/run_v1.py cache --label cache_navtest "
            f"--cache-name navtest   [cwd {W3}]")
        run.write_json("raw/refusal.json", {"reason": ctx.rec["refusal"]})
        log("[navsim_v1] " + ctx.rec["refusal"])
        return

    # ---- score every arm (reusing a PASSed W3 run of the same cache when present) ------ #
    reuse = os.environ.get("W3_NAVSIM_V1_REUSE", "1") != "0"
    scored, counts = {}, {}
    for name in known:
        label = f"{name}_navtest"
        prev = W3 / "raw" / label / f"{label}.counts.json"
        rep = None
        if reuse and prev.exists():
            r = json.load(open(prev, encoding="utf-8"))
            csvp = W3 / r.get("csv", "") if r.get("csv") else None
            same_arm = (r.get("arm") == ARMS[name]["runner"]
                        and r.get("agent_class") == AGENT_CLASS[name])
            if r.get("status") == "PASS" and same_arm \
                    and r.get("cache", "").endswith(f"metric_cache_{CACHE_NAME}") \
                    and csvp and csvp.exists():
                rc = recheck_csv(csvp, set(toks))
                if rc["ok"]:
                    rep = dict(r, reused_from=str(csvp.relative_to(REPO)).replace(os.sep, "/"),
                               reuse_recheck=rc,
                               reuse_key={"arm": r.get("arm"), "agent_class": r.get("agent_class"),
                                          "cache": r.get("cache")})
                    log(f"[navsim_v1] {name}: reusing the banked W3 run — RE-CHECKED NOW "
                        f"({rc['n_valid']}/{rc['n_expected']} valid, {rc['n_summary_rows']} summary "
                        f"row, C1 max|d| {rc['C1_pdms_formula_max_abs_delta']:.1e}, "
                        f"sha256 {rc['sha256'][:16]}…)")
                else:
                    log(f"[navsim_v1] {name}: the banked run was REJECTED by the re-check "
                        f"({rc}) — re-scoring")
        if rep is None:
            out_dir = run.p(f"raw/{name}")
            out_dir.mkdir(parents=True, exist_ok=True)
            cmd = [NV_PY, str(W3 / "code" / "run_v1.py"), "score", "--label", label,
                   "--arm", ARMS[name]["runner"], "--cache-name", CACHE_NAME, "--patch-loader"]
            env = dict(os.environ, PYTHONPATH=V11_TREE, PYTHONIOENCODING="utf-8", PYTHONUTF8="1",
                       CUDA_VISIBLE_DEVICES="-1")
            log(f"[navsim_v1] scoring {name} on {len(toks)} tokens (CPU, one worker)…")
            t0 = time.time()
            r = subprocess.run(cmd, cwd=str(W3), env=env, capture_output=True, text=True,
                               encoding="utf-8", errors="replace")
            (out_dir / "driver_stdout.txt").write_text(r.stdout + "\n[stderr]\n" + r.stderr,
                                                       encoding="utf-8")
            log(f"[navsim_v1] {name}: driver rc={r.returncode} in {time.time() - t0:.0f} s")
            p = W3 / "raw" / label / f"{label}.counts.json"
            rep = json.load(open(p, encoding="utf-8")) if p.exists() else {
                "status": "FAIL", "failures": [f"driver rc={r.returncode}, no counts.json"]}
        if rep.get("status") == "PASS" and "reuse_recheck" not in rep:
            rep = dict(rep, recheck=recheck_csv(W3 / rep["csv"], set(toks)))
            if not rep["recheck"]["ok"]:
                rep["status"] = "FAIL"
                rep.setdefault("failures", []).append(
                    "post-run re-check failed: " + str(rep["recheck"]))
        counts[name] = rep
        if rep.get("status") == "PASS":
            src = W3 / rep["csv"]
            ctx.write_scores(name, src)
            scored[name] = run.p(f"scores/{name}.csv")
            ctx.arm_rec(name)["status"] = "OK"
    run.write_json("raw/counts.json", counts)
    if not scored:
        ctx.rec["status"] = "FAILED"
        ctx.rec["refusal"] = f"no arm produced a valid CSV: {counts}"
        return

    # ---- summary ---------------------------------------------------------------------- #
    tok_rows = {n: _rows(scored[n])[0] for n in scored}
    avg_rows = {n: _rows(scored[n])[1] for n in scored}
    summary = assemble_summary(run_id=ctx.run_id, stamps=ctx.rec["stamps"], known=known,
                               model_arms=model_arms, tok_rows=tok_rows, avg_rows=avg_rows,
                               counts=counts, clusters=clusters, cache=cache,
                               model_refusals={n: ctx.arm_rec(n).get("refusal") for n in model_arms})
    ctx.write_summary(summary)
    for name in scored:
        ctx.arm_rec(name)["status"] = "OK"
    ctx.rec["status"] = "COMPLETE" if len(scored) == len(known) and not model_arms else "PARTIAL"


def assemble_summary(*, run_id: str, stamps: dict, known: list, model_arms: list, tok_rows: dict,
                     avg_rows: dict, counts: dict, clusters: dict, cache, model_refusals=None) -> dict:
    """The ``summary.json`` of one navsim_v1 run — a pure function of the scored CSVs, so the
    contract can be tested without a 12,146-token run (``tests/test_bench_navsim_v1_summary.py``)."""
    sys.path.insert(0, str(REPO / "taniteval"))
    from adapters import navsim_ci as CI
    from taniteval.bench.navsim import summarize as S

    scored = dict(tok_rows)
    floors = [f for f in ("STOP", "CV") if f in scored]
    model_refusals = model_refusals or {}
    arms_block = {}
    for name in known:
        if name not in scored:
            arms_block[name] = {
                "kind": ARMS[name]["kind"], "status": "FAILED",
                "declared_inputs": ARMS[name]["declared_inputs"],
                "headline": {"status": "UNAVAILABLE", "n": 0,
                             "reason": f"scoring FAILED: {counts.get(name, {}).get('failures')}"},
                "per_stage": {}, "submetrics": {}, "per_log": {},
                "paired": {f: ({"status": "SELF"} if f == name else
                               {"status": "UNAVAILABLE", "reason": "arm not scored", "n": 0})
                           for f in floors},
                "interval": {"status": "UNAVAILABLE", "reason": "arm not scored", "n": 0},
                "families": S.families_refused("arm not scored"),
                "files": {"counts": "raw/counts.json"}}
            continue
        tk = tok_rows[name]
        n = len(tk)
        head = _mean(tk, "score")
        sub = {k: {"value": _mean(tk, c), "x100": round(100 * _mean(tk, c), 4),
                   "role": ("multiplier" if k in ("NC", "DAC") else
                            ("weighted w=5" if k in ("EP", "TTC") else
                             ("weighted w=2" if k == "C" else "weighted w=0 (contributes NOTHING)")))}
               for k, c in TERM_COL.items()}
        per_log = {}
        for t, r in tk.items():
            per_log.setdefault(clusters.get(t, "?"), []).append(float(r["score"]))
        interval = CI.interval_from_run(protocol="PDMS_v1_navtest",
                                        clusters_by_unit=clusters,
                                        scores={t: float(r["score"]) for t, r in tk.items()},
                                        official_value=head)
        blk = {
            "kind": ARMS[name]["kind"], "status": "OK",
            "declared_inputs": ARMS[name]["declared_inputs"],
            "headline": {"value": head, "column": "score", "n": n, "x100": round(100 * head, 4),
                         "statistic": ("plain mean of the official per-token `score` over the valid "
                                       "rows = the devkit's own `average` row "
                                       "(run_pdm_score.py:144-147)")},
            "per_stage": {"single_stage": {"score": head, "n": n, "column": "score"},
                          "_note": "PDMS_v1 navtest is SINGLE-STAGE; there is no stage-1/stage-2 split"},
            "submetrics": {"variant": "PDMS_v1", "denominator": 12,
                           "formula": "NC·DAC·(5·EP + 5·TTC + 2·C)/12", **sub},
            "per_log": {"_statistic": "plain mean of the per-token `score` by OpenScene log",
                        **{k: {"n": len(v), "score_mean": sum(v) / len(v)} for k, v in sorted(per_log.items())}},
            "paired": {f: ({"status": "SELF"} if f == name else
                           _paired(tk, tok_rows[f], clusters, name, f)) for f in floors},
            "interval": interval,
            "families": S.families_refused(
                "the four families are computed by W3's artifact builder "
                "(code/artifacts_navtest.py, taniteval/adapters/navsim.py::build_artifact) and "
                "attached to artifacts/<arm>.json; this summary carries NavSim's own metric", n),
            "files": {"scores": f"scores/{name}.csv", "counts": "raw/counts.json"},
            "controls": {"C1_pdms_formula_max_abs_delta": counts[name].get("C1_max_abs_delta"),
                         "C2_log_successful": counts[name].get("log_successful"),
                         "C2_log_failed": counts[name].get("log_failed"),
                         "C3_missing_tokens": counts[name].get("C3_missing"),
                         "C3_unexpected_tokens": counts[name].get("C3_unexpected"),
                         "devkit_average_row_score": (float(avg_rows[name]["score"])
                                                      if avg_rows.get(name) else None),
                         "csv_sha256": counts[name].get("csv_sha256"),
                         "reused_from": counts[name].get("reused_from")},
        }
        if name in PUBLISHED:
            blk["vs_published"] = {k: _verdict(100 * _mean(tk, TERM_COL.get(k, "score")), v, 1)
                                   for k, v in PUBLISHED[name].items() if isinstance(v, (int, float))}
            if name == "CV":
                blk["vs_published"]["PDMS_leaderboard_INHERITED"] = _verdict(
                    100 * head, PUBLISHED["_leaderboard_INHERITED"]["CV"], 4)
        arms_block[name] = blk
    for name in model_arms:
        arms_block[name] = {
            "kind": "model", "status": "REFUSED",
            "declared_inputs": ["frames(CAM_F0/L0/R0 stitch)", "ego t0", "nav command"],
            "headline": {"status": "UNAVAILABLE", "n": 0,
                         "reason": model_refusals.get(name) or "model arm not runnable here"},
            "per_stage": {}, "submetrics": {}, "per_log": {},
            "paired": {f: {"status": "UNAVAILABLE", "reason": "arm not scored", "n": 0} for f in floors},
            "interval": {"status": "UNAVAILABLE", "reason": "arm not scored", "n": 0},
            "families": S.families_refused("arm not scored"), "files": {}}

    return {
        "schema": "taniteval.bench.summary/1", "run_id": run_id, "benchmark": "navsim_v1",
        "protocol": "PDMS_v1_navtest", "split": "navtest", "claim_bearing": True,
        "evidence_class": "MEASURED",
        "stamps": stamps,
        "headline_metric": {"name": "PDMS (NavSim v1)", "column": "score", "higher_is_better": True,
                            "statistic": "mean of the per-token `score` over the valid rows, x100 for reporting"},
        "floors": floors, "arms": arms_block,
        "controls": {"C7_loader_patch": "raw/C7_loader_*.json (W3 package): the unpatched v1.1 "
                                        "loader raises IndexError on this cache; the patched one "
                                        "maps every token to its own path",
                     "C4_determinism": "W3 raw/CV_smoke20 vs CV_smoke20_hs2: per-token rows "
                                       "bit-identical under a different PYTHONHASHSEED",
                     "KX_export_equals_scorer": "raw/KX_export_vs_scorer_smoke20.json (max |Δ| 0.0)"},
        "provenance": {
            "published_reference": PUBLISHED,
            "w3_package": "FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-navsim-v1-navtest",
            "metric_cache": str(cache).replace(os.sep, "/"),
            "cluster_map": str(CLUSTER_MAP.relative_to(REPO)).replace(os.sep, "/"),
            "estimator_module": "taniteval/adapters/navsim_ci.py (W2, PDMS_v1_navtest registered)",
            "data_placement": "D: only (PI 2026-09-19)"},
    }
