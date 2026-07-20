"""Open-loop replay app: 3-arm comparison harness + rerun visualization.

Replays cached episodes (ep_*.pt dirs) through any subset of the three
architecture arms — main world model, REF-A frozen-DINO, REF-B E2E — on
IDENTICAL windows, then:

  --mode test   writes stats.json (ADE/FDE, action MAE, maneuver accuracy,
                imag_rel, latency p50/p95, worst-K windows). With --baseline
                it prints a delta table against a previous stats.json and
                EXITS 1 on any out-of-tolerance regression (CI-hookable).
  --mode viz    additionally streams every window into rerun: camera +
                trajectory fans, BEV comparison, action/error/monitor time
                series, scrubbed on a global step timeline. Sink is an .rrd
                artifact (--rrd) and/or a live web viewer (--serve PORT).

Arm specs (Windows drive letters are safe — only the FIRST colon splits, and
a trailing :pool/:grid is recognized for REF-A):

    --arms main:D:/ckpts/main.pt refa:D:/ckpts/refa.pt:grid refb:D:/ckpts/refb.pt

Examples
--------
Regression gate (pod or CI):
  python scripts/replay_app.py --mode test \
      --arms main:/workspace/exp/ckpt.pt refb:/workspace/refb/ckpt.pt \
      --data-root /opt/comma_epcache --episodes 24 --stride 8 \
      --out /workspace/replay --baseline /workspace/replay_baseline/stats.json

Visualization artifact:
  python scripts/replay_app.py --mode viz --arms main:... refa:...:grid refb:... \
      --data-root /opt/comma_epcache --episodes 8 --out /workspace/replay \
      --rrd /workspace/replay/replay.rrd

Live viewer on a RunPod pod (see tanitad/replay/README.md for proxy detail):
  python scripts/replay_app.py --mode viz ... --serve 9090 \
      --connect-url rerun+http://<pod-id>-9876.proxy.runpod.net/proxy

Smoke/demo (synthetic 1-channel episodes, smoke-sized models,
deterministic toy tokenizer instead of online DINO):
  python scripts/replay_app.py --mode test --smoke --refa-tokenizer toy \
      --arms main:main.pt refa:refa.pt:pool refb:refb.pt \
      --data-root <episodes> --out <out> --rrd <out>/replay.rrd
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Bind the script to ITS OWN checkout's `tanitad` (worktrees/pods may carry
# an editable install pointing at a different checkout — silent version skew
# in a regression gate would be poison). Add the scripts dir too so the shared
# formal-gate code in compare_arms.py imports (one home, no divergent copies).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

import compare_arms as gates
from tanitad.instruments.numerics import strict_numerics
from tanitad.replay import stats as replay_stats
from tanitad.replay.arms import MainArm, RefAArm, RefBArm, ToyTokenizer
from tanitad.replay.engine import (ReplayEngine, load_corpora,
                                   split_fit_replay)

ARM_NAMES = ("main", "refa", "refb")


def parse_arm_spec(spec: str) -> tuple[str, str, str | None]:
    """``name:ckpt[:pool|grid]`` -> (name, ckpt_path, option).

    Only the first colon separates the arm name, so Windows paths
    (``main:C:\\ckpts\\ckpt.pt``) survive; a trailing ``:pool``/``:grid``
    is the REF-A adapter kind.
    """
    name, sep, rest = spec.partition(":")
    name = name.lower()
    if not sep or name not in ARM_NAMES:
        raise SystemExit(
            f"bad --arms spec {spec!r}: expected one of "
            f"{'/'.join(ARM_NAMES)} + ':<ckpt path>' (+ ':pool'/':grid' "
            f"for refa)")
    opt = None
    head, sep2, tail = rest.rpartition(":")
    if sep2 and tail.lower() in ("pool", "grid") and head:
        rest, opt = head, tail.lower()
    if opt is not None and name != "refa":
        raise SystemExit(f"arm option {opt!r} is only valid for refa: {spec}")
    if not Path(rest).is_file():
        raise SystemExit(f"arm checkpoint not found: {rest} (from {spec!r})")
    return name, rest, opt


def _main_cfg(main_config: str | None, smoke: bool):
    """Config for the ``main`` arm. ``--main-config`` lets the main-track arm be
    the flagship 4-brain (so TanitResim can visualize + gate the real flagship),
    else the historical default (smoke/base250cam)."""
    from tanitad.config import (base250cam_config, flagship4b_config,
                                flagship4b_reduced_config,
                                flagship4b_smoke_config, smoke_config)
    table = {"smoke": smoke_config, "base250cam": base250cam_config,
             "flagship4b": flagship4b_config,
             "flagship4b_reduced": flagship4b_reduced_config,
             "flagship4b_smoke": flagship4b_smoke_config}
    if main_config:
        return table[main_config]()
    return smoke_config() if smoke else base250cam_config()


def build_arms(specs: list[str], smoke: bool, device: str,
               refa_tokenizer: str, imag_rel: bool,
               main_config: str | None = None) -> list:
    """Instantiate arm adapters from CLI specs (fail-loud on duplicates)."""
    parsed = [parse_arm_spec(s) for s in specs]
    if len({p[0] for p in parsed}) != len(parsed):
        raise SystemExit(f"duplicate arm names in --arms: "
                         f"{[p[0] for p in parsed]}")
    arms = []
    for name, ckpt, opt in parsed:
        if name == "main":
            cfg = _main_cfg(main_config, smoke)
            arms.append(MainArm(ckpt, cfg=cfg, device=device,
                                compute_imag_rel=imag_rel))
        elif name == "refa":
            pred_cfg = None
            if smoke:
                from tanitad.replay.arms import _script_module
                pred_cfg = _script_module("refa_train").smoke_pred_config()
            tokenizer = None
            if refa_tokenizer == "toy":
                print("[replay] WARNING: REF-A uses the deterministic TOY "
                      "tokenizer (tests/demos only — NOT DINO features)",
                      flush=True)
                tokenizer = ToyTokenizer()
            arms.append(RefAArm(ckpt, adapter=opt or "grid",
                                pred_cfg=pred_cfg, device=device,
                                tokenizer=tokenizer,
                                compute_imag_rel=imag_rel))
        else:
            from tanitad.refs.refb import refb_config, refb_smoke_config
            cfg = refb_smoke_config() if smoke else refb_config()
            arms.append(RefBArm(ckpt, cfg=cfg, device=device))
    return arms


def parse_tolerances(pairs: list[str]) -> dict[str, float]:
    """``metric=rel_tol`` pairs (substring match against metric leaves)."""
    tol: dict[str, float] = {}
    for p in pairs:
        key, sep, val = p.partition("=")
        if not sep:
            raise SystemExit(f"bad --tol {p!r}: expected metric=rel_tol")
        tol[key] = float(val)
    return tol


def _run_export(args, engine, arms, fit_reps, replay_reps, corpora, out) -> int:
    """--mode export: fit probes, stream the replay into a TanitResim bundle.

    Writes ``<out>/session.json`` + ``<out>/frames/*.jpg`` (portable). Serve
    the PARENT of ``<out>`` with ``scripts/resim_app.py --sessions-root``.
    """
    from tanitad.resim.export import export_bundle

    session_name = args.session_name or Path(out).name
    try:
        from tanitad.refs.refb import MANEUVER_CLASSES
        maneuver_classes = MANEUVER_CLASSES
    except Exception:                                  # refb optional at export
        maneuver_classes = None

    # Ground-truth ego poses per episode -> arm-independent kinematic maneuver
    # labels in the bundle (the engine iterates replay_reps in order, so a
    # record's ep_index indexes this list). Every bundle gets a maneuver strip,
    # even a main-only run with no REF-B maneuver head.
    ego_poses = {i: rep.episode.poses for i, rep in enumerate(replay_reps)}

    with strict_numerics():
        engine.prepare(fit_reps)
        for arm in arms:
            if getattr(arm, "fit_report", None):
                print(f"[resim] {arm.name} probes: {arm.fit_report}",
                      flush=True)
        # Formal gates (shared code) BEFORE streaming export so the bundle
        # carries the D1-D3 verdict alongside the overlays. Gate over ALL val
        # episodes (fit + replay) so numbers reconcile with compare_arms.py.
        _report, gate_blocks = compute_gate_report(
            arms, list(fit_reps) + list(replay_reps), engine.device, args)
        t0 = time.perf_counter()
        session = export_bundle(
            engine.run(replay_reps), out, session_name,
            corpora=corpora,
            arm_ckpts={a.name: getattr(a, "ckpt", "") for a in arms},
            maneuver_classes=maneuver_classes,
            ego_poses=ego_poses,
            jpeg_quality=args.jpeg_quality,
            arm_gates=(gate_blocks["arms"] if gate_blocks else None),
            gates_summary=({k: gate_blocks[k] for k in
                            ("baselines", "verdict", "n_val_episodes",
                             "n_windows", "camera_ade_max_m",
                             "oracle_ceiling_target_m")}
                           if gate_blocks else None))
    wall_s = time.perf_counter() - t0

    n_steps = sum(len(ep["steps"]) for ep in session["episodes"])
    print(f"[resim] bundle {session_name!r}: {len(session['episodes'])} "
          f"episode(s), {n_steps} step(s) in {wall_s:.1f}s -> {out}",
          flush=True)
    for a in session["meta"]["arms"]:
        print(f"  {a['name']}: ADE {a['ade']} m | FDE {a['fde']} m | "
              f"latency p50 {a['latency_p50']} ms | color {a['color']}",
              flush=True)
    print(f"[resim] serve with: python scripts/resim_app.py --port 8888 "
          f"--sessions-root {Path(out).parent}", flush=True)
    return 0


def compute_gate_report(arms, replay_reps, device, args):
    """Formal D1-D3 gates + baselines + Phase-0 GO verdict over the replay arms,
    via the SHARED compare_arms gate code (one metric path — reconciles with a
    compare_arms.py run on the same episodes). Returns (full_report,
    compact_blocks) or (None, None) on failure/opt-out (never breaks replay)."""
    if args.no_gates:
        return None, None
    try:
        with strict_numerics():
            report = gates.compute_arm_gates(
                arms, replay_reps, device,
                n_splits=args.gate_splits, val_frac=args.gate_val_frac,
                seed=args.seed, mlp_epochs=args.gate_mlp_epochs,
                batch=args.batch, stride=args.stride,
                oracle_target=args.oracle_target,
                behavior_epochs=0 if args.no_behavior else args.behavior_epochs)
        return report, gates.compact_gate_blocks(report)
    except Exception as e:                              # gates must never break replay
        print(f"[replay] WARNING: formal gate computation failed ({e!r}) — "
              f"stats/overlays still written; gates omitted", flush=True)
        return None, None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="TanitAD open-loop replay: 3-arm test & viz harness")
    ap.add_argument("--mode", choices=("test", "viz", "export"), required=True)
    ap.add_argument("--arms", nargs="+", required=True,
                    metavar="name:ckpt[:pool|grid]")
    ap.add_argument("--data-root", required=True,
                    help="cache dir with ep_*.pt, or a parent of cache dirs "
                         "(corpus tag = dir name)")
    ap.add_argument("--corpus-glob", default="*",
                    help="subdir filter when --data-root is a parent "
                         "(e.g. '*val*')")
    ap.add_argument("--episodes", type=int, default=0,
                    help="max episodes per corpus (0 = all)")
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--fit-stride", type=int, default=None,
                    help="window stride during probe fitting (default: "
                         "max(1, stride // 4) — denser than replay)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--fit-frac", type=float, default=0.5,
                    help="leading fraction of each corpus used to fit "
                         "probes (episode-level split)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--session-name", default=None,
                    help="export mode: bundle title (default: --out dir name)")
    ap.add_argument("--jpeg-quality", type=int, default=80,
                    help="export mode: camera-frame JPEG quality (default 80)")
    ap.add_argument("--baseline", default=None,
                    help="baseline stats.json for regression compare "
                         "(test mode)")
    ap.add_argument("--tol", nargs="*", default=[],
                    metavar="metric=rel_tol",
                    help="tolerance overrides, e.g. ade=0.10 latency=2.0")
    ap.add_argument("--rrd", default=None,
                    help="write a rerun .rrd artifact (any mode)")
    ap.add_argument("--serve", type=int, default=None,
                    help="viz mode: serve the rerun web viewer on this port")
    ap.add_argument("--connect-url", default=None,
                    help="override the viewer's data URL (HTTP-proxy setups)")
    ap.add_argument("--grpc-only", action="store_true",
                    help="serve ONLY the data stream on the --serve port (no "
                         "local web viewer) — for single-proxied-port pods; "
                         "open app.rerun.io/?url=rerun+https://<proxied>/proxy")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--half", action="store_true",
                    help="fp16 autocast on CUDA")
    ap.add_argument("--smoke", action="store_true",
                    help="smoke-sized model configs (CI/demo)")
    ap.add_argument("--main-config", default=None,
                    choices=("smoke", "base250cam", "flagship4b",
                             "flagship4b_reduced", "flagship4b_smoke"),
                    help="config for the 'main' arm (default: base250cam, or "
                         "smoke with --smoke). Set flagship4b to gate/visualize "
                         "the real 4-brain flagship.")
    ap.add_argument("--no-gates", action="store_true",
                    help="skip the formal D1-D3 gate suite (overlays + ADE/FDE "
                         "stats only)")
    ap.add_argument("--gate-splits", type=int, default=8,
                    help="route-resampled episode splits for the D1 gate")
    ap.add_argument("--gate-val-frac", type=float, default=0.2)
    ap.add_argument("--gate-mlp-epochs", type=int, default=60)
    ap.add_argument("--behavior-epochs", type=int, default=40,
                    help="probe epochs for the behavior block (tactical maneuver "
                         "+ strategic route decodability)")
    ap.add_argument("--no-behavior", action="store_true",
                    help="skip the behavior block in the gate suite")
    ap.add_argument("--oracle-target", type=float, default=1.65,
                    help="grounded-ADE maturity reference (m); repo oracle "
                         "ceiling is 1.52-1.65m (see phase0_go_criteria.md)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--refa-tokenizer", choices=("dino", "toy"),
                    default="dino")
    ap.add_argument("--no-imag-rel", action="store_true",
                    help="skip imag_rel diagnostics (saves future-frame "
                         "encodes)")
    args = ap.parse_args(argv)

    device = ("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else args.device
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    arms = build_arms(args.arms, args.smoke, device, args.refa_tokenizer,
                      imag_rel=not args.no_imag_rel,
                      main_config=args.main_config)
    reps = load_corpora(args.data_root, episodes=args.episodes,
                        pattern=args.corpus_glob)
    needs_fit = any(a.requires_fit for a in arms)
    fit_reps, replay_reps = split_fit_replay(
        reps, args.fit_frac if needs_fit else 0.0)
    corpora = sorted({r.corpus for r in reps})
    print(f"[replay] {len(reps)} episodes over {len(corpora)} corpora "
          f"{corpora}; fit={len(fit_reps)} replay={len(replay_reps)}; "
          f"device={device} arms={[a.name for a in arms]}", flush=True)

    fit_stride = args.fit_stride if args.fit_stride is not None \
        else max(1, args.stride // 4)
    engine = ReplayEngine(arms, device=device, batch_size=args.batch,
                          stride=args.stride, half=args.half,
                          fit_stride=fit_stride,
                          emit_frames=args.mode in ("viz", "export")
                          or bool(args.rrd))

    if args.mode == "export":
        return _run_export(args, engine, arms, fit_reps, replay_reps,
                           corpora, out)

    logger = None
    if args.mode == "viz" or args.rrd:
        from tanitad.replay.rr_log import RerunLogger
        rrd = args.rrd
        if args.mode == "viz" and rrd is None and args.serve is None:
            rrd = str(out / "replay.rrd")     # viz without sink = artifact
        logger = RerunLogger(rrd=rrd, serve=args.serve,
                             connect_url=args.connect_url,
                             grpc_only=args.grpc_only)
        if rrd:
            print(f"[replay] rerun artifact -> {rrd}", flush=True)
        if logger.serve_url:
            print(f"[replay] live viewer on http://localhost:{args.serve} "
                  f"(data stream {logger.serve_url})", flush=True)

    t0 = time.perf_counter()
    with strict_numerics():
        engine.prepare(fit_reps)
        for arm in arms:
            if getattr(arm, "fit_report", None):
                print(f"[replay] {arm.name} probes: {arm.fit_report}",
                      flush=True)
        records = []
        for rec in engine.run(replay_reps):
            if logger is not None:
                logger.log_record(rec)
            rec.frame = None                  # keep memory flat
            records.append(rec)
    wall_s = time.perf_counter() - t0

    meta = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": args.mode, "device": device, "half": args.half,
        "smoke": args.smoke, "data_root": str(args.data_root),
        "corpora": corpora, "episodes_fit": len(fit_reps),
        "episodes_replay": len(replay_reps), "stride": args.stride,
        "wall_s": round(wall_s, 2),
        "arms": {a.name: a.describe() for a in arms},
    }
    stats = replay_stats.aggregate(records, meta=meta)

    # Formal D1-D3 gates + baselines + Phase-0 GO verdict (shared gate code).
    # Gate over ALL loaded val episodes (the gate does its own internal
    # route-resampled splits) so the numbers reconcile with a compare_arms.py
    # run on the same cache — NOT the resim fit/replay overlay split.
    report, gate_blocks = compute_gate_report(arms, reps, device, args)
    if gate_blocks is not None:
        for name, block in gate_blocks["arms"].items():
            if name in stats["arms"]:
                stats["arms"][name]["gates"] = block
        stats["gates"] = {k: gate_blocks[k] for k in
                          ("baselines", "verdict", "n_val_episodes",
                           "n_windows", "camera_ade_max_m",
                           "oracle_ceiling_target_m")}

    stats_path = out / "stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"[replay] {len(records)} windows in {wall_s:.1f}s -> {stats_path}",
          flush=True)
    for name, m in stats["arms"].items():
        g = m.get("gates", {})
        gate_txt = (f" | D1 {g['D1']} (ade {g['d1_ade_0_2s']:.3f}) "
                    f"D2 {g['D2']} D3 {g['D3']}" if g else "")
        print(f"  {name}: ADE {m['ade']:.3f} m | "
              f"ADE@20 {m['ade@20']:.3f} m | "
              f"steer MAE {m.get('steer_mae', float('nan')):.4f} | "
              f"latency p50 {m['latency_p50_ms']:.1f} ms{gate_txt}", flush=True)

    exit_code = 0
    if args.baseline:
        baseline = replay_stats.load_stats(args.baseline)
        regressions, rows = replay_stats.compare(
            stats, baseline, parse_tolerances(args.tol))
        table = replay_stats.format_table(rows)
        print("\n[replay] regression compare vs "
              f"{args.baseline}:\n{table}", flush=True)
        (out / "regression.json").write_text(
            json.dumps({"regressions": regressions, "rows": rows}, indent=2),
            encoding="utf-8")
        if regressions:
            print(f"[replay] REGRESSION: {len(regressions)} metric(s) out of "
                  f"tolerance — failing", flush=True)
            exit_code = 1
        else:
            print("[replay] regression gate PASS", flush=True)

    if logger is not None:
        logger.close()
        if args.serve is not None:
            print("[replay] serving — Ctrl-C to stop", flush=True)
            try:
                while True:
                    time.sleep(3600)
            except KeyboardInterrupt:
                pass
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
