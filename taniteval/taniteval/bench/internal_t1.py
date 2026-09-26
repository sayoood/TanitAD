"""``python -m taniteval.bench internal_t1 …`` — the REF-C T1 harness in the run-dir contract.

Wraps ``taniteval/tools/refcv3_arm.py`` (read ``taniteval/tools/REFCV3_ARM.md`` §2 first: the
deployed arm is ``os`` — ONE forward at the window origin — and NEVER ``cl``). The tool is run
as a SUBPROCESS, unmodified; this module only (a) chooses the device through the GPU-gap launcher,
(b) backs the rollout off to CPU (terminate + re-run on CPU) the moment the gap closes, and
(c) maps the tool's record into ``summary.json``.

    arms (the tool's own):  os (model, stamped T1* — ruling OPEN) · ha0 (constant velocity at the
                            measured v0: the FLOOR, bit-comparable across refav1/refcv3) · ha
                            (hold-action control) · ha0_ext (echo) · os_navshuf / os_navzero
                            (nav controls, BACKLOG R39)
    headline:               ADE over the dense grid (m, LOWER is better), episode-cluster bootstrap
                            interval — admissible only with >= 8 episodes (RG-14), else UNAVAILABLE.

``--analyze-only <dump_dir>`` re-analyses a banked dump with ZERO model compute.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from .contract import REPO, code_blobs, ckpt_display

TOOL = REPO / "taniteval" / "tools" / "refcv3_arm.py"
PROTOCOL = "TanitAD_T1_refc_physicalai"
MIN_EPISODES = 8                                     # RG-14 EPISODE_CLUSTER_FLOOR
ARM_KIND = {"os": "model", "ha0": "floor", "ha": "reference", "ha0_ext": "floor",
            "os_navshuf": "reference", "os_navzero": "reference", "os_navflip": "reference",
            "os_navpred": "reference", "oracle_sel": "reference"}
ARM_INPUTS = {
    "os": ["frames (window)", "nav_cmd (v7.2 token, per --nav-source)", "v0[t0] (measured)"],
    "ha0": ["v0[t0] (measured)"], "ha": ["frames <= t0 (the closing action)", "v0[t0]"],
    "ha0_ext": ["v0[t0]", "a0[t0]", "kappa0[t0]"],
    "os_navshuf": ["frames", "nav_cmd PERMUTED across windows", "v0[t0]"],
    "os_navzero": ["frames", "nav WITHHELD (nav_cmd=None)", "v0[t0]"],
}
FLOOR = "ha0"


def build_cmd(a, run_dir: Path, device: str, dump_dir: Path) -> list:
    out = run_dir / "raw" / "refcv3_arm.json"
    cmd = [sys.executable, str(TOOL), "--out", str(out), "--n-boot", str(a.n_boot), "--seed", str(a.seed)]
    if a.analyze_only:
        cmd += ["--analyze-only", str(a.analyze_only)]
        if a.labels:
            cmd += ["--labels", a.labels]
    else:
        cmd += ["--ckpt", a.ckpt, "--episodes", a.episodes, "--labels", a.labels, "--dump-dir", str(dump_dir),
                "--device", device, "--grid", a.grid]
        if a.config:
            cmd += ["--config", a.config]
        if a.episodes_n:
            cmd += ["--episodes-n", str(a.episodes_n)]
        if a.window_stride:
            cmd += ["--window-stride", str(a.window_stride)]
    for x in (a.t1_args or "").split():
        cmd.append(x)
    return cmd


def _run_tool(cmd: list, log_path: Path, gpu=None, poll_s: float = 5.0) -> dict:
    """Popen + poll; on a gap back-off terminate the tool (it holds the GPU) and report it."""
    t0 = time.time()
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write("CMD: " + " ".join(cmd) + "\n")
        fh.flush()
        env = dict(os.environ)
        p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=str(REPO))
        while True:
            try:
                rc = p.wait(timeout=poll_s)
                return {"rc": rc, "wall_s": round(time.time() - t0, 1), "backed_off": False}
            except subprocess.TimeoutExpired:
                if gpu is not None and gpu.should_back_off():
                    _kill_tree(p.pid)
                    p.wait(timeout=60)
                    return {"rc": None, "wall_s": round(time.time() - t0, 1), "backed_off": True}


def _kill_tree(pid: int):
    try:
        import psutil
        par = psutil.Process(pid)
        for c in par.children(recursive=True):
            c.kill()
        par.kill()
    except Exception:                                                  # noqa: BLE001
        pass


def _ade(blk: dict) -> dict:
    return ((blk.get("intervals") or {}).get("metrics") or {}).get("ade_dense_m") or {}


def ckpt_path_from_record(rec: dict):
    """The checkpoint refcv3_arm.py ACTUALLY loaded.

    ⛔ NOT ``rec["ckpt"]``: that top-level field is None on every run of this tool, while the real
    path sits at ``refcv3.manifest.model.ckpt``. MEASURED 2026-09-20 — the suite published an
    internal_t1 run with a MODEL arm (``os``) and a null checkpoint, i.e. an anonymous model row,
    while the path was recorded one file down in raw/refcv3_arm.json. Pure: reads the record only.
    """
    man = ((rec.get("refcv3") or {}).get("manifest") or {}).get("model") or {}
    return man.get("ckpt") or rec.get("ckpt") or None


def ckpt_triple(rec: dict, *, sha256=None, registry_key=None, registry_key_status=None) -> dict:
    """The {path, sha256, registry_key} triple summary.json and bench_run.json BOTH carry.
    ⚠️ ``sha256`` is passed IN rather than computed here, so this stays a pure function of the
    record; the caller does the file IO (and says so when the file is gone)."""
    path = ckpt_path_from_record(rec)
    out = {"path": path, "sha256": sha256, "registry_key": registry_key,
           "source": "refcv3.manifest.model.ckpt (the checkpoint the tool loaded)",
           "config_json": (((rec.get("refcv3") or {}).get("manifest") or {}).get("model") or {}).get("config_json")}
    out["registry_key_display"] = ckpt_display(out)
    if not registry_key:
        out["registry_key_status"] = registry_key_status or (
            "NOT NAMED by the operator (--registry-key) — a leaderboard row needs it; the checkpoint "
            "is still uniquely identified by its sha256")
    return out


def summarize_record(rec: dict, run_id: str, split: str, *, ckpt: dict | None = None) -> dict:
    """refcv3_arm record -> summary.json (schema taniteval.bench.summary/1). Pure function."""
    from .navsim.summarize import families_from_artifact
    n_ep = int(rec.get("n_episodes") or 0)
    n_win = int(rec.get("n_windows") or 0)
    paired_src = ((rec.get("refcv3") or {}).get("families_paired") or {})
    arms = {}
    for name, blk in (rec.get("arms") or {}).items():
        ade = _ade(blk)
        if isinstance(ade.get("mean"), (int, float)):
            head = {"value": float(ade["mean"]), "column": "ade_dense_m", "n": n_win,
                    "statistic": "mean ADE over the dense grid (full_set point estimate), refcv3_arm.py"}
        else:
            head = {"status": "UNAVAILABLE", "reason": "no ade_dense_m in the tool record", "n": n_win}
        if ade.get("estimator") == "episode_cluster_bootstrap" and n_ep >= MIN_EPISODES:
            interval = {"status": "OK", "estimator": "episode_cluster_bootstrap", "cluster_unit": "episode",
                        "lo": ade["lo"], "hi": ade["hi"], "n_clusters": n_ep, "n_boot": ade.get("n_boot"),
                        "question_answered": ("would another draw of EPISODES say this? — blind to training and "
                                              "inference variance (H-ESTIM-SEED-1)")}
        else:
            interval = {"status": "UNAVAILABLE", "n": n_ep,
                        "reason": (f"{n_ep} episodes < the RG-14 floor of {MIN_EPISODES}: the tool's interval "
                                   f"[{ade.get('lo')}, {ade.get('hi')}] is not admissible" if n_ep < MIN_EPISODES
                                   else f"estimator {ade.get('estimator')!r} is not the episode-cluster bootstrap")}
        paired = {}
        if name == FLOOR:
            paired[FLOOR] = {"status": "SELF"}
        else:
            src = paired_src.get(f"paired_{name}_minus_{FLOOR}")
            d = (((src or {}).get("families") or {}).get("ADE") or {}).get("ade_m") or {}
            if isinstance(d.get("delta"), (int, float)):
                paired[FLOOR] = {"status": "OK", "headline_delta": float(d["delta"]), "n_common": int(d.get("n_windows", n_win)),
                                 "direction": f"{name} - {FLOOR} (ADE: NEGATIVE is better)",
                                 "interval": {"lo": d.get("lo"), "hi": d.get("hi"), "separated": d.get("separated"),
                                              "estimator": d.get("estimator"), "n_episodes": d.get("n_episodes"),
                                              "admissible": bool(n_ep >= MIN_EPISODES)}}
            elif FLOOR in (rec.get("arms") or {}) and name in (rec.get("arms") or {}) and head.get("value") is not None:
                fa = _ade(rec["arms"][FLOOR]).get("mean")
                paired[FLOOR] = {"status": "OK", "headline_delta": (None if fa is None else head["value"] - float(fa)),
                                 "n_common": n_win, "direction": f"{name} - {FLOOR} (difference of means; no paired CI in the record)"}
            else:
                paired[FLOOR] = {"status": "UNAVAILABLE", "reason": f"no {FLOOR} arm / paired block in the record", "n": 0}
        arms[name] = {"kind": ARM_KIND.get(name, "reference"), "status": "OK",
                      "declared_inputs": ARM_INPUTS.get(name, []), "tier": blk.get("tier"),
                      "headline": head, "paired": paired, "interval": interval,
                      "families": families_from_artifact({"four_families": blk.get("four_families") or {}},
                                                         f"refcv3_arm.py on {n_win} windows / {n_ep} episodes"),
                      "statistics": {k: v for k, v in ((blk.get("intervals") or {}).get("metrics") or {}).items()},
                      "files": {"record": "raw/refcv3_arm.json"}}
    tiers = {a: b.get("tier") for a, b in (rec.get("arms") or {}).items()}
    return {
        "schema": "taniteval.bench.summary/1", "run_id": run_id, "benchmark": "internal_t1", "protocol": PROTOCOL,
        "split": split, "claim_bearing": True, "evidence_class": "MEASURED",
        "stamps": {"tier": "T1*", "loop": {"os": "one-shot plan, NO action input (T1* — the ruling is OPEN)",
                                           "ha/ha0": "T1 (integrated controls)"}, "closed_loop": False,
                   "per_arm_tier": tiers},
        "headline_metric": {"name": "ADE", "column": "ade_dense_m", "higher_is_better": False,
                            "statistic": "mean ADE over the dense grid; the admissible cross-model claim is the MARGIN over ha0"},
        "floors": [FLOOR] if FLOOR in arms else [], "arms": arms,
        "provenance": {"tool": "taniteval/tools/refcv3_arm.py", "tool_provenance": rec.get("_provenance"),
                       "n_windows": n_win, "n_episodes": n_ep, "mode": rec.get("mode"),
                       "ckpt": ckpt if ckpt is not None else ckpt_triple(rec)},
    }


def _sha256_of(path):
    """sha256 of a file, or None. ⚠️ Returns None rather than raising: a missing checkpoint must
    surface as a CONTRACT refusal on the summary (which names the run), not as a crash here."""
    import hashlib
    if not path:
        return None
    f = Path(path)
    if not f.is_file():
        return None
    h = hashlib.sha256()
    with open(f, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_benchmark(ctx) -> None:
    a = ctx.args
    if not a.analyze_only:
        miss = [n for n, v in (("--ckpt", a.ckpt), ("--episodes", a.episodes), ("--labels", a.labels)) if not v]
        if miss:
            raise ValueError(f"internal_t1 rollout needs {miss} (or --analyze-only <dump_dir>)")
    ctx.set_protocol(PROTOCOL)
    ctx.set_devkit("TanitAD taniteval/tools/refcv3_arm.py", "NOT_APPLICABLE", [
        {"name": "none — the tool runs unmodified at the repo's git_head", "kind": "none",
         "tool_blob": code_blobs([TOOL])}])
    ctx.set_stamps("T1*", {"os": "one-shot, no action input (T1* ruling OPEN)", "ha/ha0": "T1"}, "MEASURED",
                   closed_loop=False)
    ctx.set_claim_bearing(True)
    run = ctx.run
    dump_dir = run.offrepo / "dump"
    device = "cpu"
    if not a.analyze_only:
        device = ctx.gpu_device(queue_hint=(f"python -m taniteval.bench internal_t1 --ckpt {a.ckpt} "
                                            f"--split {a.split} --device cpu --accept-training-box-load"))
    log_path = run.p("raw/refcv3_arm.log")
    res = _run_tool(build_cmd(a, run.path, device, dump_dir), log_path, gpu=ctx.gpu if device == "cuda" else None)
    if res["backed_off"]:
        ctx.gpu.backed_off()
        dump_dir = run.offrepo / "dump_cpu"
        res = _run_tool(build_cmd(a, run.path, "cpu", dump_dir), log_path)
        res["after_backoff"] = True
    run.write_json("raw/refcv3_arm.run.json", {**res, "device": device, "dump_dir": str(dump_dir)})
    out = run.p("raw/refcv3_arm.json")
    if res["rc"] != 0 or not out.exists():
        ctx.rec["status"] = "FAILED"
        raise RuntimeError(f"refcv3_arm.py failed rc={res['rc']} — see raw/refcv3_arm.log")
    rec = json.loads(out.read_text(encoding="utf-8"))
    split = a.split
    ctx.set_split(split, int(rec.get("n_windows") or 0), None, n_episodes=int(rec.get("n_episodes") or 0),
                  unit="windows (n_scenes) over PhysicalAI episodes; n_logs not applicable")
    ck = ckpt_triple(rec, sha256=_sha256_of(ckpt_path_from_record(rec)),
                     registry_key=getattr(a, "registry_key", None))
    # ⭐ bench_run.json and summary.json carry the SAME object, so the two cannot disagree about
    # which checkpoint produced the row (W4, 2026-09-20). cli.py filled it from --ckpt, which is
    # `none` for this benchmark -- the real path comes from the TOOL's own manifest.
    ctx.rec["ckpt"] = {**(ctx.rec.get("ckpt") or {}), **ck}
    summ = summarize_record(rec, ctx.run_id, split, ckpt=ck)
    for name, blk in summ["arms"].items():
        ctx.add_arm(name, blk["kind"], blk["declared_inputs"], status="OK", tier=blk.get("tier"))
    ctx.write_summary(summ)
    ctx.rec["status"] = "COMPLETE"
