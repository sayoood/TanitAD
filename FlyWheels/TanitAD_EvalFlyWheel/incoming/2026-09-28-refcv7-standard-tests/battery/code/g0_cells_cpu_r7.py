"""G0 CELL-LEVEL DIAGNOSIS ON CPU (2026-10-04) -- WHY does step 30,000's `eval_tacv6_goal_conf_bce`
sit 1.5 % off the in-run value on all 24 seeds (sd 0)?

    CUDA_VISIBLE_DEVICES=-1 python g0_cells_cpu_r7.py --ckpt D:/refcv7_eval_kit/ckpt/ckpt_30000.pt \
        --config D:/refcv7_eval_kit/ckpt/config.json \
        --metrics D:/refcv7_eval_kit/thor_reads/metrics_final_50400.jsonl \
        --out D:/refcv7_eval_kit/battery/g0cells_step30000_cpu/cells.json

NOT a gate and NOT a battery number. It replays G0's in-run eval (the SAME 128 windows in the SAME 8
batches, G0's own functions imported, never copied) with ONE lever changed -- the trunk runs in fp32
NCHW on the CPU instead of eager bf16 NHWC on the RTX 4060 -- i.e. a NUMERICS-ONLY perturbation: same
weights, same flags, same windows, same trainer code. It never touches the GPU (CUDA hidden).

It also CAPTURES, per batch, the exact tensors `refcv6_tactical.tactical_behaviour_losses` received
(validity logits, confidence logits, goal_y, goal_w, the class mask). `tac_goal_conf_bce`'s TARGET is
`(sigmoid(goal_logits) >= 0.5) == goal_y` (refcv6_tactical.py:823-825): a HARD THRESHOLD on the
model's own validity logit, so a numerics-level change of a near-zero logit flips a target and moves
the batch loss by c_i * w_i / sum(w) in ONE step (c_i = that cell's confidence logit). The per-cell
record lets the analysis (`g0_cells_analyse.py`) decompose any bf16-vs-fp32 or replay-vs-in-run
difference into such flips plus a continuous remainder.

Memory: the shared dev box is COMMIT-bound (MEASURED 2026-10-04 07:20: 4.3 GB free commit). The job
waits for >= --min-commit-gb free commit before it allocates anything, collates ONE batch at a time
(no disk batch cache -- D: has < 10 GB free), and a watchdog thread EXITS THIS PROCESS (os._exit 3,
nothing else is ever touched) if free commit falls below --abort-commit-gb, so it can never be the
allocation that kills another stream's job. Assert on the JSON (--out), never on the exit code.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")      # never the GPU (the lock is someone else's)
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from wait_commit import free_commit_gb  # noqa: E402


def _now():
    return time.strftime("%FT%T%z")


CHAIN_ACTIVE = "C:/Users/Admin/qland/work/refcv7/devbox_gpu.chain_active"
GPU_LOCK = "C:/Users/Admin/qland/work/refcv7/devbox_gpu.lock"


def _refcv7_gpu_job_holds_lock() -> bool:
    """True while one of THIS battery's GPU jobs (job name `refcv7-milestone-*`, `refcv7-g0*`) holds the
    dev-box lock: its G0 needs the host memory this CPU job would take."""
    try:
        with open(GPU_LOCK, encoding="utf-8") as fh:
            job = str(json.load(fh).get("job", ""))
    except (OSError, ValueError):
        return False
    return job.startswith("refcv7-milestone-") or job.startswith("refcv7-g0")


def wait_for_commit(min_gb: float, max_wait_s: int, poll_s: int = 30) -> dict:
    """Free commit >= min_gb on 2 consecutive polls AND no milestone chain active (its marker file) AND
    no battery GPU job holding the lock: a CPU diagnosis must never start inside a milestone chain or a
    G0 and take the memory it waits for. ⚠️ MEASURED 2026-10-04 08:09-08:11: free commit swung 1.57 ->
    5.86 GB within two polls on the shared box (another stream's ~4.3 GB burst), so the bar for a ~3 GB
    CPU job is 10 GB, not the 6 GB G0 itself waits for."""
    t0, hist, ok_run = time.time(), [], 0
    while True:
        try:
            fc, tc, fp = free_commit_gb()
        except Exception as exc:                       # noqa: BLE001 -- recorded, retried
            fc, tc, fp = -1.0, -1.0, -1.0
            hist.append({"t": _now(), "error": str(exc)[:200]})
        chain = os.path.exists(CHAIN_ACTIVE) or _refcv7_gpu_job_holds_lock()
        hist = (hist + [{"t": _now(), "free_commit_gb": fc, "commit_limit_gb": tc,
                         "free_phys_gb": fp, "chain_or_g0_active": chain}])[-20:]
        ok_run = ok_run + 1 if (fc >= min_gb and not chain) else 0
        if ok_run >= 2:
            return {"met": True, "waited_s": round(time.time() - t0, 1), "last": hist[-1]}
        if time.time() - t0 > max_wait_s:
            return {"met": False, "waited_s": round(time.time() - t0, 1), "history_tail": hist}
        time.sleep(poll_s)


class CommitWatchdog(threading.Thread):
    """Exits THIS process (os._exit(3)) when free commit < abort_gb on two consecutive polls; the
    reason is written to `<out>.ABORTED.json` first. Never signals any other process."""

    def __init__(self, abort_gb: float, out_p: Path, poll_s: int = 15):
        super().__init__(daemon=True)
        self.abort_gb, self.out_p, self.poll_s = abort_gb, out_p, poll_s
        self.low = 0
        self.min_seen = None

    def run(self):
        while True:
            try:
                fc = free_commit_gb()[0]
                self.min_seen = fc if self.min_seen is None else min(self.min_seen, fc)
                self.low = self.low + 1 if 0 <= fc < self.abort_gb else 0
                if self.low >= 2:
                    json.dump({"aborted": True, "t": _now(), "free_commit_gb": fc,
                               "abort_gb": self.abort_gb,
                               "why": "free commit below the abort bar on 2 consecutive polls; this "
                                      "process exited so it cannot starve another stream"},
                              open(str(self.out_p) + ".ABORTED.json", "w", encoding="utf-8"))
                    os._exit(3)
            except Exception:                           # noqa: BLE001 -- a failed probe is not low
                pass
            time.sleep(self.poll_s)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--micro", default="2,2,3,3,3,3")
    ap.add_argument("--batches", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--min-commit-gb", type=float, default=10.0)
    ap.add_argument("--abort-commit-gb", type=float, default=0.8)
    ap.add_argument("--max-wait-s", type=int, default=18000)
    a = ap.parse_args()
    out_p = Path(a.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    rec = {"tool": "g0_cells_cpu_r7.py", "what": "G0 cell-level numerics diagnosis on CPU; NOT a gate, "
           "NOT a result", "started": _now(), "ckpt": a.ckpt, "config": a.config,
           "lever": "trunk fp32 NCHW on CPU (vs G0: eager bf16 NHWC on the RTX 4060); nothing else",
           "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}

    def bank():
        tmp = Path(str(out_p) + ".part")
        json.dump(rec, open(tmp, "w", encoding="utf-8"), indent=1, default=str)
        tmp.replace(out_p)

    rec["commit_wait"] = wait_for_commit(a.min_commit_gb, a.max_wait_s)
    bank()
    if not rec["commit_wait"]["met"]:
        rec["finished"] = None
        rec["refused"] = f"free commit never reached {a.min_commit_gb} GB in {a.max_wait_s} s"
        bank()
        return 3
    wd = CommitWatchdog(a.abort_commit_gb, out_p)
    wd.start()

    import g0_refcv7 as G                               # bootstraps the loader (sys.path, tanitad)
    from g0_refcv7 import L
    import torch
    from microbatch import MicroBatchForward
    torch.set_num_threads(int(a.threads))
    rec["torch"] = torch.__version__
    rec["cuda_available"] = bool(torch.cuda.is_available())
    if rec["cuda_available"]:
        raise SystemExit("[cells] CUDA is visible -- this diagnosis must never touch the GPU")
    rec["tanitad_file"] = __import__("tanitad").__file__
    rec["ckpt_md5"] = L.md5_file(a.ckpt)
    rec["config_md5"] = L.md5_file(a.config)
    tr = L.trainer()
    t0 = time.time()
    config = L.load_config(a.config)
    model, cfg, args, mrec = L.build_model(config, a.ckpt, "cpu")
    step = int(mrec["state_dict"]["step"])
    rec["step"] = step
    rec["model"] = {"strict_missing": len(mrec["state_dict"]["missing"]),
                    "strict_unexpected": len(mrec["state_dict"]["unexpected"]),
                    "param_breakdown_equal": mrec["param_breakdown"]["equal"],
                    "trunk_memory_levers_built": mrec.get("trunk_memory_levers_built"),
                    "departures": mrec.get("departures"), "build_s": round(time.time() - t0, 1)}
    # requires_grad AS TRAINED (G0 does this; the flag state changes slot-head outputs)
    tf = G.training_flags(model)
    G.set_flags(model, tf["flags"])
    from tanitad.models.timm_trunk import TimmResNetTrunk
    trunk = [m for m in model.modules() if isinstance(m, TimmResNetTrunk)][0]
    lev0 = dict(trunk.memory_levers)
    trunk.memory_levers["bf16"] = False                 # on CPU an enabled lever WOULD autocast bf16
    trunk.memory_levers["channels_last"] = False
    trunk.net.to(memory_format=torch.contiguous_format)
    rec["trunk_levers"] = {"as_built": lev0, "this_arm": dict(trunk.memory_levers)}
    rows = [json.loads(ln) for ln in open(a.metrics, encoding="utf-8") if ln.strip()]
    ev = [r for r in rows if r.get("step") == step and "eval_loss" in r]
    if len(ev) != 1:
        raise SystemExit(f"[cells] {len(ev)} eval rows at step {step}")
    inrun = ev[0]
    rec["inrun_tac"] = {k: v for k, v in inrun.items() if "tac" in k or "goal" in k}
    law_idx = int(tr.LAW_AHEAD) - 1
    t_ds = time.time()
    e_ds, _e_eps, drec = L.build_eval_dataset(model, cfg, args, config, with_perception_targets=True,
                                              dataset_cls=G.make_g0_dataset_cls(tr, int(tr.LAW_AHEAD)))
    B = int(args.batch)
    perm = L.inrun_eval_perm(e_ds, int(args.eval_batches), B)
    rec["perm_sha256"] = hashlib.sha256(json.dumps(perm).encode()).hexdigest()
    rec["dataset"] = {"n_episodes": drec.get("n_episodes"), "n_windows": drec.get("n_windows"),
                      "build_s": round(time.time() - t_ds, 1)}
    print(f"[cells] built model + dataset; step {step}; perm {rec['perm_sha256'][:12]}", flush=True)
    bank()
    # ---- the capture: exactly what the tactical loss saw -------------------------------- #
    cap = []
    orig_tbl = tr.v6tac.tactical_behaviour_losses

    def tbl(out, **kw):
        total, tele = orig_tbl(out, **kw)
        cm = kw.get("goal_class_mask")
        cap.append({"goal_logits": out["goal_logits"].detach().float().cpu().tolist(),
                    "goal_conf": out["goal_conf"].detach().float().cpu().tolist(),
                    "goal_y": kw["goal_y"].detach().float().cpu().tolist(),
                    "goal_w": kw["goal_w"].detach().float().cpu().tolist(),
                    "class_mask": None if cm is None else
                    (cm.detach().float().cpu().tolist() if torch.is_tensor(cm) else list(cm)),
                    "pos_weight_sha": None if kw.get("goal_pos_weight") is None else hashlib.sha256(
                        torch.as_tensor(kw["goal_pos_weight"]).float().cpu().numpy().tobytes()
                    ).hexdigest()[:16],
                    "tac_goal_conf_bce": float(tele["tac_goal_conf_bce"]),
                    "tac_goal_bce": float(tele["tac_goal_bce"])})
        return total, tele
    tr.v6tac.tactical_behaviour_losses = tbl
    G.patch_frames_to_device(tr)
    mode = getattr(args, "mode", "diffusion")
    abl = bool(getattr(args, "ablate_frames", False))
    mb = MicroBatchForward(model, [int(x) for x in a.micro.split(",")]).install()
    which = [int(x) for x in a.batches.split(",") if x.strip()]
    torch.manual_seed(0)
    acc, nb, per_batch, cells = {}, 0, {}, {}
    rec["per_batch"], rec["cells"] = per_batch, cells
    try:
        with torch.no_grad():
            for bi in which:
                t_b = time.time()
                eb = G.collate(e_ds, perm[bi * B:(bi + 1) * B], law_idx)
                n_cap = len(cap)
                el = tr.compute_losses_v3(model, eb, "cpu", mode=mode, ablate_frames=abl)
                row = {}
                for k, v in el.items():
                    if torch.is_tensor(v) and v.ndim == 0:
                        acc[k] = acc.get(k, 0.0) + float(v.detach())
                        row[k] = float(v.detach())
                    elif isinstance(v, (int, float, bool)):
                        acc[k] = acc.get(k, 0.0) + float(v)
                        row[k] = float(v)
                if len(cap) != n_cap + 1:
                    raise SystemExit(f"[cells] batch {bi}: the tactical loss was called "
                                     f"{len(cap) - n_cap} times, expected 1")
                per_batch[str(bi)] = row
                cells[str(bi)] = cap[-1]
                nb += 1
                del el, eb
                rec["wall_by_batch_s"] = {**(rec.get("wall_by_batch_s") or {}),
                                          str(bi): round(time.time() - t_b, 1)}
                rec["min_free_commit_gb_seen"] = wd.min_seen
                print(f"[cells] batch {bi}: goal_conf_bce {row.get('tacv6_goal_conf_bce'):.6f} "
                      f"goal_bce {row.get('tacv6_goal_bce'):.6f} ({time.time() - t_b:.0f}s)", flush=True)
                bank()
    finally:
        mb.remove()
        tr.v6tac.tactical_behaviour_losses = orig_tbl
    if nb == int(args.eval_batches):
        erow = tr._eval_row_from_acc(acc, nb, model)
        rec["row"] = erow
    rec["n_batches_done"] = nb
    rec["finished"] = _now()
    rec["wall_s"] = round(time.time() - t0, 1)
    bank()
    print(f"[cells] wrote {out_p} ({rec['wall_s']} s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
