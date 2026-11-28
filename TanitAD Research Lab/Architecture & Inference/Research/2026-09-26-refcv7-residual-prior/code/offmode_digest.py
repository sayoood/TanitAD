#!/usr/bin/env python3
"""refcv7 NEW-1, test (c) across TREES: is ``--residual-prior off`` bit-identical to refcv6?

Builds the SAME refcv6-shaped smoke arm (sampler ddim, v0-conditioned alat anchors that include
(0, 0), F1-F6, ego history, ego-dropout 0.5) through the trainer's OWN ``build_parser`` +
``_pin_trainer_cfg`` + ``RefCV3Model`` + ``compute_losses_v3``, with NO residual-prior flag, on a
fixed seed, and prints ONE sha256 over every tensor that leaves the model and the trainer:

  * TRAIN mode: the trainer's own loss dict (every scalar), and every parameter gradient after
    one backward (sorted by name);
  * EVAL mode: the forward's planner outputs (anchor_bank, anchor_traj, traj, sel_idx,
    anchor_logits, refined_logits, sel_score, u0_hat, layer_u0_hat, layer_logits).

Run it on the TIP tree and on the candidate tree: the two digests must be EQUAL. The pytest
``test_residual_prior.py::test_c_*`` pins the tip's digest as a literal (platform-guarded).

usage: python offmode_digest.py <repo>      (prints JSON)
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys


def build_and_digest(repo: str, extra_argv: list | None = None) -> dict:
    stack = os.path.join(repo, "stack")
    for p in (os.path.join(stack, "scripts"), stack):
        if p not in sys.path:
            sys.path.insert(0, p)
    import torch
    import tanitad
    # ONE intra-op thread, set HERE: CPU reductions are summed in a thread-count-dependent
    # order, so the same code gives a different digest at 1 / 2 / 4 / 8 threads (MEASURED
    # 2026-09-26: four different digests on the tip). The digest must depend on the CODE only.
    _nt = torch.get_num_threads()
    torch.set_num_threads(1)
    spec = importlib.util.spec_from_file_location(
        "rv3t_digest", os.path.join(stack, "scripts", "refc_v3_train.py"))
    T = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(T)
    f1_f6 = ["--f1-random-t", "--f2-dd-step", "--f3-per-layer", "--f4-adaln", "--f5-focal",
             "--f5-emitting-conf", "--f6-w-u0-zero"]
    argv = ["--arm", "hier", "--sampler", "ddim", "--anchor-v0-conditioned",
            "--anchor-control-units", "alat", "--n-anchors", "20", "--ego-history",
            "--out", os.path.join(repo, "_digest_unused_out")] + f1_f6 + list(extra_argv or [])
    args = T.build_parser().parse_args(argv)
    cfg = T._pin_trainer_cfg(T.v3.refc_v3_smoke_config(True), args)
    torch.manual_seed(0)
    model = T.v3.RefCV3Model(cfg)
    ac = model.core.decoder.anchor_controls
    a_lon = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    a_lat = torch.tensor([-1.5, -0.5, 0.0, 0.5])
    with torch.no_grad():
        ac.copy_(torch.cartesian_prod(a_lon, a_lat).to(ac.dtype))
    eps = T._synth_episodes(2, cfg.core, seed=0)
    ds = T.V3Dataset(eps, window=cfg.core.window, max_horizon=20,
                     channels=cfg.core.encoder.in_channels)
    ds.ego_history = True
    batch = torch.utils.data.default_collate([ds[0], ds[7], ds[23], ds[31]])
    h = hashlib.sha256()

    def feed(name: str, t) -> None:
        h.update(name.encode())
        if t is None:
            h.update(b"<None>")
            return
        if isinstance(t, (list, tuple)):
            for i, x in enumerate(t):
                feed(f"{name}[{i}]", x)
            return
        x = t.detach().cpu().contiguous()
        h.update(str(tuple(x.shape)).encode() + str(x.dtype).encode())
        h.update(x.numpy().tobytes())

    model.train()
    torch.manual_seed(1)
    losses = T.compute_losses_v3(model, batch, "cpu", mode="diffusion")
    for k in sorted(losses):
        v = losses[k]
        if torch.is_tensor(v) and v.dim() == 0:
            feed(f"loss:{k}", v)
    losses["loss"].backward()
    n_grad = 0
    for name, p in sorted(model.named_parameters()):
        if p.grad is not None:
            feed(f"grad:{name}", p.grad)
            n_grad += 1
    model.zero_grad(set_to_none=True)
    model.eval()
    torch.manual_seed(2)
    with torch.no_grad():
        ph = batch["pose_hist"]
        model.core.set_ego_window(ph, int(ph.shape[1]))
        out = model(T.frames_to_device(batch["frames"], "cpu"), nav_cmd=batch["nav_cmd"],
                    v0=batch["pose_last"][:, 3], steps=cfg.core.decoder.diffusion_steps)
    keys = ("anchor_bank", "anchor_traj", "traj", "sel_idx", "anchor_logits",
            "refined_logits", "sel_score", "u0_hat", "layer_u0_hat", "layer_logits")
    for k in keys:
        feed(f"out:{k}", out.get(k))
    torch.set_num_threads(_nt)
    return {"repo": repo, "tanitad": tanitad.__file__, "torch": torch.__version__,
            "threads": 1,
            "n_param_grads": n_grad, "n_params": sum(p.numel() for p in model.parameters()),
            "state_dict_keys_sha256": hashlib.sha256(
                "\n".join(sorted(model.state_dict().keys())).encode()).hexdigest(),
            "residual_keys_in_out": sorted(k for k in out if k.startswith("residual_prior")),
            "digest": h.hexdigest()}


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    print(json.dumps(build_and_digest(os.path.abspath(sys.argv[1]), sys.argv[2:]), indent=1))
