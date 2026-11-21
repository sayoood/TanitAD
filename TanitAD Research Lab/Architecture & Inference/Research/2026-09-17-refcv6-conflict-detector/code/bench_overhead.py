"""refcv6 §6 -- MEASURED overhead of the gradient-conflict detector.

Three trunk sizes x four modes (off / probe / reuse / subtract), on the trainer's
own ``RefCV3Model`` + the WP-D BEV auxiliary head, CPU only, synthetic corpus.
⛔ The prereg budgets the detector at "one extra backward over the trunk -- no
extra arm, no extra GPU-day"; this is the file that says what it actually costs.

    python bench_overhead.py --out <dir>/overhead.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, *([".."] * 5)))
STACK = os.environ.get("TANITAD_STACK", os.path.join(REPO, "stack"))
if not os.path.isdir(os.path.join(STACK, "tanitad")):
    raise SystemExit(
        f"no `tanitad` package under {STACK!r}. Run from the repo, or set "
        f"TANITAD_STACK to the worktree's `stack/`.")
for _p in (STACK, os.path.join(STACK, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

spec = importlib.util.spec_from_file_location(
    "refc_v3_train_bench", os.path.join(STACK, "scripts", "refc_v3_train.py"))
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)

from tanitad.refs import refc_bev_aux as RB          # noqa: E402
from tanitad.train import grad_conflict as gc        # noqa: E402

torch.set_num_threads(4)



def build(base_width, image_size, blocks, w_bev=0.2, n_rng=6):
    cfg = T.v3.refc_v3_smoke_config(True)
    cfg.core.encoder.base_width = base_width
    cfg.core.encoder.image_size = image_size
    cfg.core.encoder.blocks = blocks
    gs = cfg.core.encoder.grid_shape
    gh, gw = (gs() if callable(gs) else gs)
    cfg.core.bev_aux = RB.BEVAuxConfig(enable=True, kind="col", n_rng=n_rng,
                                       n_az=gw, d_tok=8, hidden=16, w=w_bev,
                                       pos_weight=30.0)
    torch.manual_seed(0)
    m = T.v3.RefCV3Model(cfg)
    m._w_bev_aux = w_bev
    m._w_goal_point = 0.0
    m._w_tac_goal = 0.0
    m._tac_goal_pos_weight = None
    m._tac_goal_class_mask = None
    m.train()
    return cfg, m, (gh, gw, n_rng)


def make_batch(cfg, shp, seed=0):
    gh, gw, n_rng = shp
    eps = T._synth_episodes(2, cfg.core, seed=seed)
    ds = T.V3Dataset(eps, window=cfg.core.window, max_horizon=20,
                     channels=cfg.core.encoder.in_channels)
    b = torch.utils.data.default_collate([ds[0], ds[1]])
    v7l = T.v7l
    b["lat_v7"] = torch.tensor([0, v7l.IGNORE_ID], dtype=torch.long)
    b["lon_v7"] = torch.tensor([len(v7l.HEADS["tac_lon"]) - 1, v7l.IGNORE_ID],
                               dtype=torch.long)
    b["nav_cmd"] = torch.tensor([1, 2], dtype=torch.long)
    b["nav_valid"] = torch.tensor([True, True])
    K = len(v7l.TAC_GOAL_TOKENS)
    b["tac_goal_y"] = torch.zeros(2, K)
    b["tac_goal_w"] = torch.zeros(2, K)
    g = torch.Generator().manual_seed(7 + seed)
    b["bev_occ"] = (torch.rand(2, n_rng, gw, generator=g) < 0.1).float()
    b["bev_mask"] = torch.ones(2, n_rng, gw, dtype=torch.bool)
    return b


def bench(cfg, model, shp, mode, n=15, warm=4):
    det = None
    if mode != "off":
        det = gc.GradientConflictDetector.for_model(
            model, gc.ConflictConfig(enabled=True, mode=mode,
                                     trunk_prefixes=gc.resolve_trunk_prefixes(model)))
    batches = [make_batch(cfg, shp, seed=i % 2) for i in range(2)]
    ts, probe_ts = [], []
    for i in range(n + warm):
        b = batches[i % 2]
        t0 = time.perf_counter()
        ls = T.compute_losses_v3(model, b, "cpu", mode="diffusion")
        model.zero_grad(set_to_none=True)
        if det is not None:
            a, c = T._conflict_terms(model, ls)
            if mode == gc.MODE_SUBTRACT:
                ls["loss"].backward(retain_graph=True)
                p0 = time.perf_counter()
                det.measure_after_backward(c, step=i)
                p1 = time.perf_counter()
            else:
                p0 = time.perf_counter()
                det.measure(a, c, step=i, retain_graph=(mode != gc.MODE_REUSE))
                p1 = time.perf_counter()
                if mode == gc.MODE_REUSE:
                    det.accumulate_()
                else:
                    ls["loss"].backward()
        else:
            p0 = p1 = 0.0
            ls["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        dt = time.perf_counter() - t0
        if i >= warm:
            ts.append(dt)
            probe_ts.append(p1 - p0)
    return statistics.median(ts), statistics.median(probe_ts)


#: ⚠️ REPLICATED AND INTERLEAVED. The dev box is shared; a single timing pass
#: per mode drifts with whatever else is running, and three sequential passes
#: measured the SAME configuration at +80 %, +93 % and +84 % on three runs. The
#: modes are therefore cycled inside each replicate and the per-mode figure is
#: the MEDIAN over replicates, with the min/max kept so the spread is visible
#: rather than averaged away.
REPLICATES = 3
MODES = ("off", gc.MODE_PROBE, gc.MODE_REUSE, gc.MODE_SUBTRACT)
SIZES = ((8, 64, (1, 1, 1, 1)), (24, 64, (2, 2, 2, 2)), (48, 96, (2, 2, 3, 2)))


def measure():
    rows = []
    for bw, imsz, blocks in SIZES:
        cfg, model, shp = build(bw, imsz, blocks)
        n_trunk = sum(p.numel() for n, p in model.named_parameters()
                      if n.startswith("core.encoder."))
        n_all = sum(p.numel() for p in model.parameters())
        per_mode = {m: [] for m in MODES}
        for _rep in range(REPLICATES):
            for m in MODES:
                t, _inner = bench(cfg, model, shp, m)
                per_mode[m].append(t)
        med = {m: statistics.median(v) for m, v in per_mode.items()}
        off = med["off"]
        row = {"base_width": bw, "image": imsz, "trunk_params": n_trunk,
               "total_params": n_all, "replicates": REPLICATES,
               "step_s_median": med,
               "step_s_all": per_mode,
               "pct_vs_off": {m: 100.0 * (med[m] - off) / off for m in MODES},
               "pct_vs_off_spread": {
                   m: [100.0 * (min(per_mode[m]) - off) / off,
                       100.0 * (max(per_mode[m]) - off) / off] for m in MODES}}
        rows.append(row)
        print("bw=%-3d trunk=%9d | off %.4fs | probe %+6.1f%% | reuse %+6.1f%% "
              "| subtract %+6.1f%%   (off spread %.4f-%.4f s)"
              % (bw, n_trunk, off, row["pct_vs_off"][gc.MODE_PROBE],
                 row["pct_vs_off"][gc.MODE_REUSE],
                 row["pct_vs_off"][gc.MODE_SUBTRACT],
                 min(per_mode["off"]), max(per_mode["off"])), flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = measure()
    sys.stdout.reconfigure(encoding="utf-8")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({
            "what": "gradient-conflict detector overhead, CPU, batch 2, real "
                    "RefCV3Model + WP-D BEV aux",
            "method": "median of 15 steps after 4 warm-up; step = "
                      "compute_losses_v3 + zero_grad + [detector] + backward "
                      "+ clip_grad_norm_",
            "torch": torch.__version__, "threads": 4, "rows": rows}, f, indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
