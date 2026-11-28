"""refcv6 §6 -- the conflict detector's WORKED EXAMPLE, on a tiny two-head smoke.

Builds the trainer's own ``RefCV3Model`` (``refc_v3_smoke_config``) with the
WP-D BEV auxiliary head attached to the shared trunk, runs a short supervised
loop at 1x and at the 30x mutation, and writes ONE ``metrics.jsonl`` per arm in
exactly the schema ``refc_v3_train.py`` writes -- so the rows in ``RESULT.md``
are rows a real run would produce, not a rendering of them.

⛔ CPU only, no data, no download: the corpus is ``_synth_episodes``.

    python worked_example.py --out <dir>

Writes ``<dir>/{1x,30x}/metrics.jsonl``, ``<dir>/summary.json``.
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
# code -> <slug> -> Research -> "Architecture & Inference" -> Lab -> <repo>
REPO = os.path.abspath(os.path.join(HERE, *([".."] * 5)))
STACK = os.environ.get("TANITAD_STACK", os.path.join(REPO, "stack"))
if not os.path.isdir(os.path.join(STACK, "tanitad")):
    raise SystemExit(
        f"no `tanitad` package under {STACK!r}. Run from the repo, or set "
        f"TANITAD_STACK to the worktree's `stack/` -- guessing a path is how "
        f"an import lands on a different checkout than the one under test.")
for p in (STACK, os.path.join(STACK, "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from tanitad.train import grad_conflict as gc          # noqa: E402

W_BEV = 0.2
N_RNG = 6
SEED = 20260917


def trainer():
    path = os.path.join(STACK, "scripts", "refc_v3_train.py")
    spec = importlib.util.spec_from_file_location("refc_v3_train_worked", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build(T, w_bev=W_BEV):
    from tanitad.refs import refc_bev_aux as RB
    cfg = T.v3.refc_v3_smoke_config(True)
    gs = cfg.core.encoder.grid_shape
    gh, gw = (gs() if callable(gs) else gs)
    cfg.core.bev_aux = RB.BEVAuxConfig(enable=True, kind="col", n_rng=N_RNG,
                                       n_az=gw, d_tok=8, hidden=16, w=w_bev,
                                       pos_weight=30.0)
    torch.manual_seed(0)
    m = T.v3.RefCV3Model(cfg)
    m._w_bev_aux = float(w_bev)
    m._w_goal_point = 0.0
    m._w_tac_goal = 0.0
    m._tac_goal_pos_weight = None
    m._tac_goal_class_mask = None
    return cfg, m, gw


def batch(T, cfg, gw, seed=0):
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
    k = len(v7l.TAC_GOAL_TOKENS)
    b["tac_goal_y"] = torch.zeros(2, k)
    b["tac_goal_w"] = torch.zeros(2, k)
    g = torch.Generator().manual_seed(7 + seed)
    b["bev_occ"] = (torch.rand(2, N_RNG, gw, generator=g) < 0.1).float()
    b["bev_mask"] = torch.ones(2, N_RNG, gw, dtype=torch.bool)
    return b


def run_arm(T, out_dir, scale, steps=40):
    """One arm. ``scale`` multiplies the aux loss: 1.0, or the 30x MUTATION."""
    os.makedirs(out_dir, exist_ok=True)
    cfg, model, gw = build(T)
    model.train()
    det = gc.GradientConflictDetector.for_model(model, gc.ConflictConfig(
        enabled=True, trunk_prefixes=gc.resolve_trunk_prefixes(model)))
    opt = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    batches = [batch(T, cfg, gw, seed=i) for i in range(4)]
    log = open(os.path.join(out_dir, "metrics.jsonl"), "w", encoding="utf-8")
    controls = None
    rows = []
    for step in range(steps):
        b = batches[step % len(batches)]
        losses = T.compute_losses_v3(model, b, "cpu", mode="diffusion")
        lt, la0 = T._conflict_terms(model, losses)   # la0 = w_bev * bev
        # ⛔ THE MUTATION, in BOTH places it has to be: the detector's aux side
        # AND the loss the arm actually trains on. Scaling only the detector's
        # copy would measure a defect the run does not have.
        la = la0 * float(scale)
        total = losses["loss"] + (float(scale) - 1.0) * la0
        opt.zero_grad(set_to_none=True)
        if controls is None:
            c = det.self_check(lt, la)
            controls = c.as_dict()
            log.write(json.dumps({"step": step, "conflict_controls": controls,
                                  "conflict_provenance": det.provenance()}) + "\n")
        r = det.measure(lt, la, step=step)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        row = {k: round(float(v.detach()), 5) for k, v in losses.items()
               if torch.is_tensor(v) and v.ndim == 0}
        row.update(step=step + 1, aux_scale=float(scale))
        row.update(r.row())
        log.write(json.dumps(row) + "\n")
        rows.append(row)
    log.close()
    return rows, controls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=40)
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    torch.set_num_threads(4)
    T = trainer()
    out = {}
    for tag, scale in (("1x", 1.0), ("30x", 30.0)):
        t0 = time.time()
        rows, controls = run_arm(T, os.path.join(a.out, tag), scale, a.steps)
        out[tag] = {
            "controls": controls,
            "step0": {k: rows[0][k] for k in rows[0] if k.startswith("cd_")},
            "median_cos": statistics.median(r["cd_cos"] for r in rows),
            "median_ratio": statistics.median(r["cd_ratio"] for r in rows),
            "median_proj": statistics.median(r["cd_proj"] for r in rows),
            "frac_cos_negative": sum(1 for r in rows if r["cd_cos"] < 0) / len(rows),
            "frac_proj_below_minus_1": sum(
                1 for r in rows if r["cd_proj"] < -1.0) / len(rows),
            "elapsed_s": round(time.time() - t0, 2),
            "steps": len(rows),
        }
        print("[%s] median cos %+.6f  ratio %.4f  proj %+.6f  (%d steps)"
              % (tag, out[tag]["median_cos"], out[tag]["median_ratio"],
                 out[tag]["median_proj"], len(rows)), flush=True)
    with open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
