"""Checkpoint-watch: auto-run the Phase-0 DEFINED BENCHMARK gate suite on VAL as
each training checkpoint lands, appending to a per-run gate log.

WHAT IT RUNS (the defined benchmarks, on the val split — NOT train)
-------------------------------------------------------------------
For every new ``ckpt.pt`` it runs the SAME machinery as ``compare_arms.py`` for
ONE arm (no metric is reinvented — it reuses ``compare_arms.compare``):
  - D1/D2/D3 decode gates with instrument-doctrine PASS/FAIL/BLOCKED
    (``tanitad.eval.gates``);
  - grounded-decode ADE toward the oracle ceiling
    (``metric_dynamics.rollout_decode``, the eval_grounded_rollout_4b method);
  - trivial baselines + in-distribution oracle ceiling
    (``driving_diagnostic`` helpers);
  - the Phase-0 §4 verdict (D1 camera <1.0m; D2 dir-acc >0.7; D3 <=1.5x oracle;
    grounded beats the CV floor).
The closed-loop gates D4-D6 remain the arbiters and are computed in sim, not here.

GATE LOG
--------
Appends one JSON line per checkpoint to ``--gate-log`` (default
``<exp-dir>/gate_log.jsonl``): {ts, step, ckpt, summary{D1,D2,D3},
d1_ade_0_2s, grounded_ade_0_2s, cv_ade_0_2s, beats_cv, verdict}. The full
per-checkpoint report is also written to ``<out>/gates_step<STEP>.json`` +
``.md`` so a milestone is fully reproducible.

DEPLOYMENT (dev-box vs pod — the GPU-contention tradeoff)
---------------------------------------------------------
RECOMMENDED: run on the DEV-BOX 4060, pulling each new ``ckpt.pt`` + a fixed
val SUBSET (100-200 episodes) off the pod. Rationale: the gate suite competes
for GPU; running it on the pod steals cycles from the training it is measuring
(a 30k run is already GPU-bound). The 4060 evaluates a 150-episode val subset in
a few minutes and never touches the training GPU. The only cost is copying the
checkpoint (~0.5-1 GB) + the val subset once.
ALTERNATIVE: ``--once`` on the pod at a checkpoint boundary, when the trainer is
between saves — acceptable for a single milestone, not for a tight poll loop.

Usage (dev-box poll loop, flagship):
  python scripts/watch_gates.py --arm flagship --flagship-config flagship4b \
      --exp-dir /local/pull/flagship-30k \
      --frame-cache-dirs /local/pull/physicalai_val \
      --episodes 150 --interval-s 300 --out /local/pull/flagship-30k/gates

  # single pass (e.g. after copying a milestone ckpt):
  python scripts/watch_gates.py --arm flagship --flagship-config flagship4b \
      --exp-dir <dir> --frame-cache-dirs <val> --episodes 150 --once
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import compare_arms as ca  # noqa: E402


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------- #
# Optional pod pull (dev-box deploy) — refresh the local ckpt + a val subset   #
# from a training pod over ssh/scp. READ-ONLY on the pod (never writes there); #
# scp of a checkpoint file is non-disruptive to the running trainer.           #
# --------------------------------------------------------------------------- #
def _run(cmd: list, dry: bool):
    """Run (or, with ``dry``, just print) a shell-FREE command (list argv).
    Returns the CompletedProcess (or None in dry-run)."""
    print(f"[pull{':dry-run' if dry else ''}] {' '.join(cmd)}", flush=True)
    if dry:
        return None
    return subprocess.run(cmd, capture_output=True, text=True)


def _ssh_mtime(host: str, remote: str, dry: bool):
    """Remote file mtime (epoch int) via ssh stat; None if unavailable."""
    cp = _run(["ssh", host, "stat", "-c", "%Y", remote], dry)
    if cp is None or cp.returncode != 0:
        return None
    try:
        return int(cp.stdout.strip())
    except ValueError:
        return None


def pull_val_subset(host: str, remote_val: str, local_val: Path, n: int,
                    dry: bool) -> None:
    """One-time: pull the FIRST ``n`` ``ep_*.pt`` of a pod val cache into a
    local ``*val*`` dir. Shell-FREE + cross-platform: (1) list the subset ON
    THE POD via one ssh (so the glob/sort/head run remotely), (2) one scp of the
    explicit remote files. Idempotent (skips if already pulled)."""
    dest = local_val / "physicalai-val"
    lst_cmd = ["ssh", host,
               f"cd {remote_val} && ls ep_*.pt | sort | head -{n}"]
    if dry:                                   # show commands only; touch nothing
        _run(lst_cmd, dry=True)
        print(f"[pull:dry-run] scp -q {host}:{remote_val}/ep_XXXX.pt ... "
              f"(first {n}) {dest}/", flush=True)
        return
    dest.mkdir(parents=True, exist_ok=True)
    if any(dest.glob("ep_*.pt")):
        return
    # (1) list the first n episode files on the pod (remote glob/sort/head)
    lst = _run(lst_cmd, dry=False)
    if lst is None or lst.returncode != 0:
        print(f"[pull] WARNING val list failed: {getattr(lst,'stderr','')!r}",
              flush=True)
        return
    files = [f for f in lst.stdout.split() if f.endswith(".pt")]
    srcs = [f"{host}:{remote_val}/{f}" for f in files]
    if srcs:
        _run(["scp", "-q", *srcs, str(dest) + "/"], dry=False)


def pull_ckpt(host: str, remote_ckpt: str, local_ckpt: Path,
              last_mtime, dry: bool):
    """scp the pod checkpoint locally only when its remote mtime changed
    (the trainer overwrites ckpt.pt every save) — avoids re-pulling ~1 GB each
    poll. Returns (new_mtime, pulled_bool)."""
    mtime = _ssh_mtime(host, remote_ckpt, dry)
    if not dry and mtime is not None and mtime == last_mtime:
        return last_mtime, False
    if not dry:
        local_ckpt.parent.mkdir(parents=True, exist_ok=True)
    cp = _run(["scp", "-q", f"{host}:{remote_ckpt}", str(local_ckpt)], dry)
    ok = dry or (cp is not None and cp.returncode == 0)
    return (mtime if mtime is not None else last_mtime), ok


def _find_ckpt(exp_dir: Path, pattern: str) -> Path | None:
    cands = sorted(exp_dir.glob(pattern), key=lambda p: p.stat().st_mtime)
    return cands[-1] if cands else None


def _ckpt_step(path: Path, device) -> int:
    try:
        ck = torch.load(path, map_location=device, weights_only=True)
        return int(ck.get("step", -1)) if isinstance(ck, dict) else -1
    except Exception:
        return -1


def build_arm(args, ckpt, device):
    if args.arm == "flagship":
        return ca.build_flagship(str(ckpt), args.flagship_config, device)
    if args.arm == "refa":
        return ca.build_refa(str(ckpt), args.refa_adapter, args.refa_smoke, device)
    return ca.build_refb(str(ckpt), args.refb_smoke, device)


def run_suite(args, ckpt, device, frame_val, feat_by_id) -> dict:
    arm = build_arm(args, ckpt, device)
    report = ca.compare([arm], frame_val, feat_by_id, device,
                        n_splits=args.n_splits, val_frac=args.val_frac,
                        seed=args.seed, mlp_epochs=args.mlp_epochs,
                        batch=args.batch, stride=args.stride,
                        git_hash=args.git_hash, oracle_target=args.oracle_target,
                        behavior_epochs=0 if args.no_behavior else args.behavior_epochs,
                        behavior_turn_deg=args.behavior_turn_deg)
    report["ckpt"] = str(ckpt)
    report["watch_arm"] = args.arm
    return report


def gate_log_row(report: dict, arm_name: str) -> dict:
    r = report["arms"][arm_name]
    g = r.get("grounded") or {}
    im = r.get("imagination") or {}
    cv = report["baselines"]["constant_velocity"]["ade_0_2s"]
    return {
        "ts": _now(), "step": r["step"], "ckpt": report.get("ckpt"),
        "arm": arm_name,
        "summary": {"D1": r["decode"]["d1_status"],
                    "D2": im.get("d2_status", "N/A"),
                    "D3": im.get("d3_status", "N/A")},
        "d1_ade_0_2s": r["decode"]["d1_ade_0_2s"],
        "oracle_ceiling_ade_0_2s": r["decode"]["oracle_ceiling_ade_0_2s"],
        "grounded_ade_0_2s": g.get("ade_0_2s"),
        "cv_ade_0_2s": cv,
        "grounded_beats_cv": g.get("beats_cv_overall"),
        "d2_dir_acc": im.get("d2_dir_acc"),
        "d3_ratio": im.get("d3_ratio"),
        "n_val_episodes": report["val"]["n_common_episodes"],
        "n_windows": report["val"]["n_windows"],
    }


def one_pass(args, device, frame_val, feat_by_id, seen: set, gate_log: Path,
             out_dir: Path) -> int | None:
    """Evaluate the newest checkpoint if unseen. Returns the step evaluated."""
    exp_dir = Path(args.exp_dir)
    ckpt = _find_ckpt(exp_dir, args.ckpt_glob)
    if ckpt is None:
        print(f"[watch] no {args.ckpt_glob} under {exp_dir} yet", flush=True)
        return None
    step = _ckpt_step(ckpt, device)
    key = (ckpt.name, step)
    if key in seen:
        return None
    print(f"[watch] new checkpoint {ckpt.name} step {step} -> running gate suite",
          flush=True)
    report = run_suite(args, ckpt, device, frame_val, feat_by_id)
    seen.add(key)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"gates_step{step}.json").write_text(
        json.dumps(report, indent=2, default=str))
    (out_dir / f"gates_step{step}.md").write_text(ca.render_markdown(report))
    row = gate_log_row(report, args.arm)
    with gate_log.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, default=str) + "\n")
    print(f"[watch] step {step}: D1={row['summary']['D1']} "
          f"D2={row['summary']['D2']} D3={row['summary']['D3']} | "
          f"d1_ade={row['d1_ade_0_2s']:.3f} "
          f"grounded_ade={row['grounded_ade_0_2s']} "
          f"cv={row['cv_ade_0_2s']:.3f} -> log {gate_log}", flush=True)
    return step


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=["flagship", "refa", "refb"],
                    default="flagship")
    ap.add_argument("--flagship-config", default="flagship4b",
                    choices=["flagship4b", "flagship4b_reduced", "smoke"])
    ap.add_argument("--refa-adapter", default="grid", choices=["grid", "pool"])
    ap.add_argument("--refa-smoke", action="store_true")
    ap.add_argument("--refb-smoke", action="store_true")
    ap.add_argument("--exp-dir", required=True,
                    help="dir the trainer writes ckpt.pt into")
    ap.add_argument("--ckpt-glob", default="ckpt.pt",
                    help="checkpoint filename/glob to watch (e.g. 'ckpt*.pt')")
    ap.add_argument("--frame-cache-dirs", nargs="+", default=None,
                    help="val frame caches (flagship/refb) — <root>/*val*/ep_*.pt "
                         "(auto-set to the pulled val dir when --pull-host is used)")
    ap.add_argument("--refa-feat-dir", default=None,
                    help="val DINO features (refa arm)")
    # -- pod pull (dev-box deploy): auto-refresh ckpt + val subset over ssh --
    ap.add_argument("--pull-host", default=None,
                    help="ssh alias of the training pod (e.g. tanitad-pod2). "
                         "Enables auto-pull of ckpt + a val subset to the 4060, "
                         "non-disruptive to training (read-only scp).")
    ap.add_argument("--pull-ckpt", default=None,
                    help="remote checkpoint path (scp'd to <exp-dir>/<ckpt-glob> "
                         "each poll when its mtime changes)")
    ap.add_argument("--pull-val", default=None,
                    help="remote val cache dir (first N ep_*.pt streamed once "
                         "to <exp-dir>/val_subset/physicalai-val)")
    ap.add_argument("--pull-val-episodes", type=int, default=None,
                    help="val subset size to pull (default: --episodes)")
    ap.add_argument("--dry-run-pull", action="store_true",
                    help="print the exact scp/ssh pull commands and exit "
                         "(shows the standing command; runs nothing)")
    ap.add_argument("--out", default=None, help="report dir (default exp-dir/gates)")
    ap.add_argument("--gate-log", default=None,
                    help="JSONL gate log (default <out>/gate_log.jsonl)")
    ap.add_argument("--episodes", type=int, default=150)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--n-splits", type=int, default=8)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mlp-epochs", type=int, default=60)
    ap.add_argument("--behavior-epochs", type=int, default=40,
                    help="probe epochs for the behavior block (0 with --no-behavior)")
    ap.add_argument("--no-behavior", action="store_true",
                    help="skip the behavior block (decode + grounded gates only)")
    ap.add_argument("--behavior-turn-deg", type=float, default=45.0)
    ap.add_argument("--pose-tol", type=float, default=1e-2)
    ap.add_argument("--oracle-target", type=float, default=1.65)
    ap.add_argument("--git-hash", default="unknown")
    ap.add_argument("--once", action="store_true",
                    help="single pass then exit (no poll loop)")
    ap.add_argument("--interval-s", type=int, default=300,
                    help="poll interval when not --once")
    ap.add_argument("--max-iters", type=int, default=0,
                    help="stop after N polls (0 = unbounded)")
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available() and torch.cuda.device_count() > 0
              else "cpu")
    n_val_pull = args.pull_val_episodes or args.episodes
    ckpt_local = Path(args.exp_dir) / args.ckpt_glob
    val_local = Path(args.exp_dir) / "val_subset"

    # -- pod pull: --dry-run-pull just SHOWS the exact standing commands --------
    if args.dry_run_pull:
        if not (args.pull_host and args.pull_ckpt):
            raise SystemExit("--dry-run-pull needs --pull-host + --pull-ckpt")
        if args.pull_val:
            pull_val_subset(args.pull_host, args.pull_val, val_local,
                            n_val_pull, dry=True)
        pull_ckpt(args.pull_host, args.pull_ckpt, ckpt_local, None, dry=True)
        print("WATCH_GATES_DRY_RUN_DONE", flush=True)
        return

    # -- pod pull: fetch the val subset ONCE, point the watcher at it ----------
    if args.pull_host and args.pull_val:
        pull_val_subset(args.pull_host, args.pull_val, val_local, n_val_pull,
                        dry=False)
        if not args.frame_cache_dirs:
            args.frame_cache_dirs = [str(val_local)]
    if not args.frame_cache_dirs:
        raise SystemExit("--frame-cache-dirs is required (or use --pull-host "
                         "+ --pull-val to auto-provision it)")

    need_feature = args.arm == "refa"
    out_dir = Path(args.out) if args.out else Path(args.exp_dir) / "gates"
    gate_log = Path(args.gate_log) if args.gate_log else out_dir / "gate_log.jsonl"
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_val = ca.load_frame_val(args.frame_cache_dirs, args.episodes)
    assert frame_val, "no frame val episodes loaded"
    feat_eps = (ca.load_feature_val(args.refa_feat_dir, args.episodes)
                if need_feature else [])
    frame_val, feat_by_id, common_ids = ca.load_common_val(
        frame_val, feat_eps, need_feature, args.pose_tol)
    assert frame_val, "no common val episodes"
    print(f"[watch] arm={args.arm} device={device} "
          f"val_episodes={len(frame_val)} gate_log={gate_log}", flush=True)

    seen: set = set()
    last_mtime = None
    iters = 0
    while True:
        if args.pull_host and args.pull_ckpt:      # refresh ckpt when it changed
            last_mtime, _ = pull_ckpt(args.pull_host, args.pull_ckpt,
                                      ckpt_local, last_mtime, dry=False)
        one_pass(args, device, frame_val, feat_by_id, seen, gate_log, out_dir)
        iters += 1
        if args.once or (args.max_iters and iters >= args.max_iters):
            break
        time.sleep(args.interval_s)
    print("WATCH_GATES_DONE", flush=True)


if __name__ == "__main__":
    main()
