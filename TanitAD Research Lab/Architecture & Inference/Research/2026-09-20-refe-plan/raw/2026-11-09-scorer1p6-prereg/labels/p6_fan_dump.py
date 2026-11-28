#!/usr/bin/env python3
"""P6 fan dump = refe/rl_fan_dump.py with the nav_mode defect closed and, for FIT sets, P6's logits saved.  Wraps the pod's rl_fan_dump.py; edits nothing.

THE DEFECT (rl_fan_dump.py:561-563).  `run()` builds the dataset config as `REFeConfig.for_backbone(a.backbone)` -- nav_mode "goal" -- and hands it to
`train.TargetBank`.  Only the MODEL goes through `ckpt_io.config_for_checkpoint` (load_model :254), so for P6 (meta nav_mode=command) the model expects the
one-hot command in the `goal` slot while `TargetBank.__getitem__` (train.py:962-965) feeds `r["goal"][:2*n_goal_points]` = 4 goal-point floats.  Nothing raises:
2 * n_goal_points == n_commands == 4 (model.py:170, :192), so the shape check at model.py:768 passes.  Every P6 proposal from the unpatched dumper is conditioned
on garbage.  Here the dataset config comes from the same `config_for_checkpoint`, and the job refuses unless the result is nav_mode "command".
The upstream fix is `rl_fan_dump_nav_mode.patch` (same change inside run()); this wrapper works on any pod copy of rl_fan_dump.py meanwhile.

LOGITS (--save-logits).  `forward_batch` discards the score output; for FIT sets it is saved, per set, to `<out>/fit_logits_<tag>_s<i>of<n>.jsonl`
({"key": [log, token, step, rank], "logits": [M][K]}), written (fsync) BEFORE the dump's own chunk write, so every dumped set has its logits; a resumed batch
may appear twice and the later line wins (p6_labels_tool.py pack).  The forward pass is rl_fan_dump.forward_batch's, line for line.
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def import_rl_fan_dump(refe_dir):
    if refe_dir not in sys.path:
        sys.path.insert(0, refe_dir)
    import rl_fan_dump as RD
    return RD


def dataset_cfg(backbone, ckpt, require_command=True):
    """The config the DATASET must be built with: the checkpoint's own (nav_mode, n_score_components, ...)."""
    import ckpt_io
    from model import REFeConfig
    cfg = ckpt_io.config_for_checkpoint(REFeConfig.for_backbone(backbone), ckpt)
    if require_command and cfg.nav_mode != "command":
        raise SystemExit(f"P6FAN_REFUSED {ckpt} resolves to nav_mode {cfg.nav_mode!r}; the P6 checkpoint is a command-mode model")
    return cfg


class RowIndexed:
    """Delegates to the dataset and appends the row index to every item, so the forward pass can name the row (forward_batch reads it[0], it[1], it[2], it[8] only)."""

    def __init__(self, ds):
        self._ds = ds

    def __getitem__(self, i):
        return tuple(self._ds[i]) + (int(i),)

    def __getattr__(self, name):
        return getattr(self._ds, name)


class LogitSaver:
    def __init__(self, path, rows):
        self.path, self.rows = path, rows
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.fh = open(path, "a", encoding="utf-8", newline="\n")
        self.n = 0

    def forward_batch(self, model, items, device, amp=True):
        """rl_fan_dump.forward_batch (onpolicy_dump.py lines 128-136) with the score output kept."""
        import torch
        img = torch.stack([it[0] for it in items]).to(device, non_blocking=True)
        ego = torch.stack([it[1] for it in items]).to(device, non_blocking=True)
        goal = torch.stack([it[2] for it in items]).to(device, non_blocking=True)
        cal = torch.stack([it[8] for it in items]) if items[0][8].numel() else None
        if amp:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                traj, score = model(img, ego, goal, calib=cal)
        else:
            with torch.no_grad():
                traj, score = model(img, ego, goal, calib=cal)
        sc = score.float().cpu().numpy()
        if sc.ndim != 3 or sc.shape[0] != len(items):
            raise RuntimeError(f"score output has shape {sc.shape}, need [B, M, K]")
        for j, it in enumerate(items):
            row = self.rows[it[-1]]
            key = [row["log_name"], row["token"], int(row.get("step", 0)), int(row.get("rank", 0))]
            self.fh.write(json.dumps({"key": key, "logits": [[round(float(x), 6) for x in m] for m in sc[j]]}) + "\n")
            self.n += 1
        self.fh.flush()
        os.fsync(self.fh.fileno())
        return traj.float().cpu().numpy()


def main(argv=None, *, build=None, loader=None, ds_inner=None):
    """`ds_inner(a, cfg)` is the test seam for the dataset (default train.TargetBank)."""
    argv = list(sys.argv[1:] if argv is None else argv)
    refe = os.environ.get("REFE_DIR", "")
    if "--refe-dir" in argv:
        i = argv.index("--refe-dir")
        refe = argv[i + 1]
        del argv[i:i + 2]
    if not refe:
        print("P6FAN_REFUSED set --refe-dir (or REFE_DIR) to the pod's refe/ directory")
        return 4
    save = "--save-logits" in argv
    if save:
        argv.remove("--save-logits")
    RD = import_rl_fan_dump(refe)
    a = RD.build_parser().parse_args(argv)

    def factory(a_):
        cfg = dataset_cfg(a_.backbone, a_.ckpt)
        if ds_inner is not None:
            ds = ds_inner(a_, cfg)
        else:
            import train as T
            try:
                ds = T.TargetBank(a_.targets, a_.images, cfg, calib=a_.calib, max_missing_frac=1.0)
            except RuntimeError as e:
                print(f"REFUSED: {e}")
                raise SystemExit(RD.EXIT_REFUSED)
        if save:
            ds = RowIndexed(ds)
        return ds, cfg.n_proposals, cfg.horizon_steps

    saver = None
    if save and not a.plan_only:
        sn = f"{a.tag}_s{RD.parse_shard(a.shard)[0]}of{RD.parse_shard(a.shard)[1]}"
        # rows are known only after the dataset exists; the saver binds them lazily
        class _Lazy:
            def __init__(self):
                self.rows = None
        lazy = _Lazy()
        orig_factory = factory

        def factory(a_):                                          # noqa: F811
            ds, n_prop, horizon = orig_factory(a_)
            lazy.rows = ds.rows
            return ds, n_prop, horizon
        os.makedirs(a.out, exist_ok=True)
        saver = LogitSaver(os.path.join(a.out, f"fit_logits_{sn}.jsonl"), None)
        orig_fwd = RD.forward_batch

        def fwd(model, items, device, amp=True):
            saver.rows = lazy.rows
            return saver.forward_batch(model, items, device, amp)
        RD.forward_batch = fwd
    try:
        rc = RD.run(a, build=build, loader=loader, ds_factory=factory)
    finally:
        if saver is not None:
            RD.forward_batch = orig_fwd
            saver.fh.close()
    if rc == 0 and not a.plan_only:
        with open(os.path.join(a.out, "p6_dump_nav_mode.json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump({"nav_mode": "command", "dataset_cfg_from": "ckpt_io.config_for_checkpoint", "logits_saved": bool(save)}, f)
    return rc


if __name__ == "__main__":
    sys.exit(main())
