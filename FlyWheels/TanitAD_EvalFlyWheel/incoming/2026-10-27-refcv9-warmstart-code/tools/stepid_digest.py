"""stepid_digest.py -- digest a REAL training step on the refcv8 CPU rig: compute_losses_v3 (the trainer's own loss
function, which runs r8_before_forward -> the forward -> r8_losses) on a CI-synthetic batch, backward, ONE AdamW step.
Digests every loss/telemetry tensor + scalar, every gradient and every post-step state tensor, for a pristine-vs-modified
bit-identity read (refcv9 --r9-alloc-force-schedule, flag ABSENT; or attached with --spec on the modified tree).

  CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=<tree>/stack python -X utf8 stepid_digest.py --tree <tree> --out <json> [--spec S]
  python -X utf8 stepid_digest.py --compare a.json b.json --report r.json [--ignore-prefix r9_pool]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def _digest(t) -> str:
    import torch
    t = t.detach().cpu().reshape(-1)
    d = torch.empty(t.shape, dtype=t.dtype)
    d.copy_(t)
    h = hashlib.sha256()
    h.update(str(t.dtype).encode() + str(tuple(t.shape)).encode())
    h.update(d.view(torch.uint8).numpy().tobytes() if d.numel() else b"")
    return h.hexdigest()


def run(tree: Path, spec: str | None) -> dict:
    stack = tree / "stack"
    for p in (str(stack), str(stack / "scripts"), str(stack / "tests")):
        sys.path.insert(0, p)
    import torch
    import tanitad
    import _refcv8_rig as R
    torch.set_num_threads(1)
    T = R.trainer()
    res = {"tanitad_file": tanitad.__file__, "torch": torch.__version__, "spec": spec, "arms": {}}
    for name, r8 in (("refcv7_step", False), ("refcv8_step", True)):
        kw = dict(n_alloc=8, w_alloc_l1=1.0, w_sat=0.1) if r8 else {}
        cfg, m = R.build(T, r8, seed=0, **kw)
        if spec is not None and r8:
            from types import SimpleNamespace
            from tanitad.train import refcv9_alloc_force as AF
            AF.attach(m, SimpleNamespace(r9_alloc_force_schedule=spec, r9_alloc_force_seed=AF.DEFAULT_SEED,
                                         refcv8=True, r8_n_alloc=8, r9_calib_stage=False, steps=10))
            AF.apply_step(m, 7)
        eps = T._synth_episodes(2, cfg.core, seed=0, min_frames=120)
        ds = T.V3Dataset(eps, window=cfg.core.window, max_horizon=60, channels=cfg.core.encoder.in_channels)
        ds.ego_history = True
        n = len(ds)
        batch = torch.utils.data.default_collate([ds[i] for i in range(0, n, max(1, n // 3))][:3])
        v7l = T.v7l
        batch["lat_v7"] = torch.tensor([0, v7l.IGNORE_ID, 0], dtype=torch.long)
        batch["lon_v7"] = torch.tensor([0, v7l.IGNORE_ID, 0], dtype=torch.long)
        batch["tac_goal_y"] = torch.zeros(3, len(v7l.TAC_GOAL_TOKENS))
        batch["tac_goal_w"] = torch.zeros(3, len(v7l.TAC_GOAL_TOKENS))
        m._w_tac_v6 = 1.0
        m.train()
        opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-3)
        opt.zero_grad(set_to_none=True)
        torch.manual_seed(5)
        L = T.compute_losses_v3(m, batch, "cpu", mode="diffusion")
        L["loss"].backward()
        acc = {}
        for k in sorted(L, key=str):
            v = L[k]
            if torch.is_tensor(v):
                acc[f"loss.{k}"] = _digest(v)
            elif isinstance(v, (int, float, bool, str)) or v is None:
                acc[f"loss.{k}"] = repr(v)
        for k, p in m.named_parameters():
            if p.grad is not None:
                acc[f"grad.{k}"] = _digest(p.grad)
        opt.step()
        st = {k: _digest(v) for k, v in m.state_dict().items()}
        res["arms"][name] = {"out": acc, "state": st, "n_out": len(acc), "n_state": len(st),
                             "loss": float(L["loss"].detach()),
                             "r8_alloc_gt_rows": (None if "r8_alloc_gt_rows" not in L else int(L["r8_alloc_gt_rows"]))}
    return res


def compare(a: dict, b: dict, ignore: tuple[str, ...]) -> dict:
    rep = {}

    def keep(k):
        return not any(k.startswith(f"loss.{p}") for p in ignore)
    for arm in a["arms"]:
        A, Bm = a["arms"][arm], b["arms"][arm]
        d_out = sorted(k for k in A["out"] if keep(k) and A["out"][k] != Bm["out"].get(k))
        x_out = sorted(k for k in set(Bm["out"]) - set(A["out"]) if keep(k))
        ign = sorted(k for k in set(Bm["out"]) | set(A["out"]) if not keep(k))
        d_st = sorted(k for k in A["state"] if A["state"][k] != Bm["state"].get(k))
        x_st = sorted(set(Bm["state"]) - set(A["state"]))
        rep[arm] = {"n_out": A["n_out"], "out_differ": d_out, "out_extra_in_b": x_out, "ignored_keys": ign,
                    "n_state": A["n_state"], "state_differ": d_st, "state_extra_in_b": x_st,
                    "BIT_IDENTICAL": not (d_out or x_out or d_st or x_st)}
    rep["ALL_BIT_IDENTICAL"] = all(v["BIT_IDENTICAL"] for v in rep.values() if isinstance(v, dict))
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree")
    ap.add_argument("--out")
    ap.add_argument("--spec", default=None)
    ap.add_argument("--compare", nargs=2)
    ap.add_argument("--report")
    ap.add_argument("--ignore-prefix", nargs="*", default=[])
    a = ap.parse_args()
    if a.compare:
        r = compare(*(json.loads(Path(p).read_text(encoding="utf-8")) for p in a.compare), tuple(a.ignore_prefix))
        Path(a.report).write_text(json.dumps(r, indent=1), encoding="utf-8")
        print("ALL_BIT_IDENTICAL", r["ALL_BIT_IDENTICAL"],
              {k: (len(v["out_differ"]), len(v["state_differ"])) for k, v in r.items() if isinstance(v, dict)})
        sys.exit(0 if r["ALL_BIT_IDENTICAL"] else 3)
    r = run(Path(a.tree), a.spec)
    Path(a.out).write_text(json.dumps(r, indent=1), encoding="utf-8")
    print("tanitad from", r["tanitad_file"], {k: (v["n_out"], v["n_state"], v["loss"], v["r8_alloc_gt_rows"])
                                               for k, v in r["arms"].items()})
