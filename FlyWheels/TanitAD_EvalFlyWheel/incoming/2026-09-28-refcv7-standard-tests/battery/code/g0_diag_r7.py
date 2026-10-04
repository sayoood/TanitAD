"""G0 DIAGNOSTIC (2026-10-04, EvalFlyWheel) -- WHY did step 5,000's `eval_traj` fall outside its band?

    python g0_diag_r7.py --ckpt D:/refcv7_eval_kit/ckpt/ckpt_5000.pt \
        --config D:/refcv7_eval_kit/ckpt/config.json --metrics <metrics copy> \
        --arms s0,eps0,fp32_s0,fp32_eps0,loaderflags_s0 --out <diag.json>

NOT a gate and NOT a battery number. It replays the in-run eval on the SAME 128 windows in the SAME 8
batches exactly as `g0_refcv7.py` does (its own functions are imported, not copied) and changes ONE
thing per arm, so the arm that moves `eval_traj` onto the in-run value names the mechanism:

  s0              G0's own seed-0 replay (control: must reproduce g0.json's seed 0 bit-for-bit-ish)
  eps0            the DDIM draw ZEROED (`torch.randn_like` -> zeros): is the in-run value what a
                  noise-free decode gives?
  fp32_s0         trunk fp32 + NCHW, cuDNN TF32 off (P3 settings): trunk numerics
  fp32_eps0       both
  loaderflags_s0  every parameter requires_grad=False (the loader's state) instead of the training flags
  micro_alt_s0    micro-batch sizes 3,3,3,3,4 instead of 2,2,3,3,3,3 (batch-composition numerics)
  seed<K>         G0's own arm at inference seed K; K >= 8 EXTENDS G0's 8 seeds, so the seed spread of the
                  128-window mean (the quantity the STOCHASTIC tolerance is built on) is re-measured

Every arm records the full eval row; the STOCHASTIC/decoder terms are summarised against the in-run
row. Assert on the JSON (`--out`), never on the exit code.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import g0_refcv7 as G  # noqa: E402  (bootstraps the loader)
from g0_refcv7 import L  # noqa: E402
import torch  # noqa: E402
from microbatch import MicroBatchForward  # noqa: E402
import wrapper_probe_r7 as W  # noqa: E402

KEYS = ("eval_traj", "eval_cascade", "eval_sel_v3", "eval_loss", "eval_goal_score_absmean",
        "eval_cls", "eval_anchor_acc", "eval_lat", "eval_lon_tac", "eval_goal_tac", "eval_law")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--arms", default="s0,eps0,fp32_s0,fp32_eps0,loaderflags_s0,micro_alt_s0")
    ap.add_argument("--micro", default="2,2,3,3,3,3")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-cache", default=None, help="dir for the 8 collated batches (mmap)")
    a = ap.parse_args()
    t_all = time.time()
    tr = L.trainer()
    device = "cuda"
    spec = HERE.parent / "SPEC.md"
    rec = {"tool": "g0_diag_r7.py", "what": "G0 eval_traj diagnosis; NOT a gate, NOT a result",
           "spec_sha256": hashlib.sha256(spec.read_bytes()).hexdigest(),
           "ckpt": a.ckpt, "ckpt_md5": L.md5_file(a.ckpt), "config_md5": L.md5_file(a.config),
           "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
           "tanitad_file": __import__("tanitad").__file__, "started": time.strftime("%FT%T%z"),
           "arms": {}}
    out_p = Path(a.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    def bank():
        json.dump(rec, open(out_p, "w", encoding="utf-8"), indent=1, default=str)

    config = L.load_config(a.config)
    model, cfg, args, mrec = L.build_model(config, a.ckpt, device)
    step = int(mrec["state_dict"]["step"])
    rec["step"] = step
    rows = [json.loads(ln) for ln in open(a.metrics, encoding="utf-8") if ln.strip()]
    ev = [r for r in rows if r.get("step") == step and "eval_loss" in r]
    if len(ev) != 1:
        raise SystemExit(f"[diag] {len(ev)} eval rows at step {step}")
    inrun = ev[0]
    rec["inrun"] = {k: inrun.get(k) for k in KEYS}
    law_idx = int(tr.LAW_AHEAD) - 1
    e_ds, e_eps, drec = L.build_eval_dataset(model, cfg, args, config, with_perception_targets=True,
                                             dataset_cls=G.make_g0_dataset_cls(tr, int(tr.LAW_AHEAD)))
    B = int(args.batch)
    perm = L.inrun_eval_perm(e_ds, int(args.eval_batches), B)
    rec["perm_sha256"] = hashlib.sha256(json.dumps(perm).encode()).hexdigest()
    # ⚠️ HOST COMMIT IS THE BINDING RESOURCE on the shared dev box (MEASURED 2026-10-04 00:33: 2.0 GB
    # free commit of 50 GB; the first launch died in `md5_file` with MemoryError). The 8 collated
    # batches (~0.5 GB uint8 each) are therefore written to disk ONCE and re-read through a
    # file-backed mmap per arm (no commit charge); the content is identical to G0's `cached` list.
    cached = G.MmapBatches(Path(a.batch_cache or (str(out_p.parent / "batch_cache"))),
                           [lambda i=i: G.collate(e_ds, perm[i * B:(i + 1) * B], law_idx)
                            for i in range(int(args.eval_batches))])
    rec["batch_cache"] = str(cached.root)

    def batches():
        return iter(cached)
    G.patch_frames_to_device(tr)
    mode = getattr(args, "mode", "diffusion")
    abl = bool(getattr(args, "ablate_frames", False))
    tf = G.training_flags(model)
    loader_flags = {n: bool(p.requires_grad) for n, p in model.named_parameters()}
    from tanitad.models.timm_trunk import TimmResNetTrunk
    trunk = [m for m in model.modules() if isinstance(m, TimmResNetTrunk)][0]
    lev0 = dict(trunk.memory_levers)
    bk0 = (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32,
           torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
    rec["backends_at_start"] = bk0
    orig_randn_like = torch.randn_like
    print(f"[diag] built; step {step}; perm {rec['perm_sha256'][:12]}; arms {a.arms}", flush=True)
    for arm in [x.strip() for x in a.arms.split(",") if x.strip()]:
        t0 = time.time()
        sizes = [int(x) for x in (("3,3,3,3,4" if arm == "micro_alt_s0" else a.micro).split(","))]
        G.set_flags(model, loader_flags if arm == "loaderflags_s0" else tf["flags"])
        if arm.startswith("fp32"):
            W._set_condition(W.CONDITIONS["P3_fp32_det"], trunk)
        if "eps0" in arm:
            torch.randn_like = lambda x, *aa, **kk: torch.zeros_like(x)
        # `seed<K>`: G0's own arm at inference seed K (K >= 8 extends G0's 8 seeds: the seed spread of
        # the 128-window mean is the quantity the STOCHASTIC tolerance is built on)
        seed = int(arm[4:]) if arm.startswith("seed") else 0
        mb = MicroBatchForward(model, sizes).install()
        try:
            erow, pb, _ = G.run_eval(tr, model, batches, device, mode, abl, seed, B)
        except torch.cuda.OutOfMemoryError as exc:
            erow, pb = {"OOM": str(exc)[:300]}, []
            torch.cuda.empty_cache()
        finally:
            mb.remove()
            torch.randn_like = orig_randn_like
            trunk.memory_levers.update(lev0)
            (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32,
             torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark) = bk0
            G.set_flags(model, tf["flags"])
        summ = {}
        for k in KEYS:
            v, i = erow.get(k), inrun.get(k)
            summ[k] = {"arm": v, "inrun": i,
                       "rel_vs_inrun": (None if v is None or i is None
                                        else (v - i) / max(abs(i), 1e-12))}
        rec["arms"][arm] = {"micro": sizes, "seed": seed, "summary": summ,
                            "per_batch_traj": [r.get("traj") for r in pb],
                            "per_batch": pb,
                            "row": erow, "wall_s": round(time.time() - t0, 1)}
        print(f"[diag] {arm}: eval_traj {erow.get('eval_traj')} (in-run {inrun.get('eval_traj')}) "
              f"cascade {erow.get('eval_cascade')} sel_v3 {erow.get('eval_sel_v3')} "
              f"lat {erow.get('eval_lat')} ({time.time() - t0:.0f} s)", flush=True)
        bank()
    rec["wall_s"] = round(time.time() - t_all, 1)
    rec["finished"] = time.strftime("%FT%T%z")
    bank()
    print(f"[diag] wrote {out_p} ({rec['wall_s']} s)", flush=True)


if __name__ == "__main__":
    main()
