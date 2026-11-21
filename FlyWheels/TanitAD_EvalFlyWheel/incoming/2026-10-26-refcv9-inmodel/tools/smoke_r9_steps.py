"""smoke_r9_steps.py -- a CPU smoke of the TRAINER's calibration-stage step with --r9-progress-head --r9-calib-stage ON:
argv -> the trainer's own parser + `_pin_trainer_cfg` (incl. the R9 pin) -> RefCV3Model -> `_r9p.apply_calib_freeze` ->
G-DVB `check` -> N x [`compute_losses_v3` (the trainer's REAL loss function, incl. the stage's eval switch) -> backward
-> AdamW over the trainable set]. Data: the trainer's CI-synthetic corpus through `V3Dataset` with ego history ON and
max_horizon 60 (so the 5.0 s / 6.0 s decision slots are VALID), exactly as tests/test_grad_unreachable_declared builds it.

Why not `train()` end to end: on `--synth-episodes` the real loop's batches carry no `pose_hist` (MEASURED: the
trainer refuses "--ego-history but the batch carries no `pose_hist`"), and the head needs the observed pose window.

Asserts (artifact SMOKE_R9_STEPS.json): S1 finite loss every step and loss == the R9 loss; S2 all 6 head tensors get a
NON-ZERO gradient every step; S3 every non-R9 tensor bit-equal to the build after N steps, the head moved; S4 G-DVB
reports no mismatch on the R9 levers / the freeze; S5 decision rows present (the CE term is live, not masked out).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=10)
    a = ap.parse_args()
    stack = Path(a.tree) / "stack"
    for p in (str(stack), str(stack / "scripts"), str(stack / "tests")):
        sys.path.insert(0, p)
    import torch
    import tanitad
    import refc_v3_train as T
    from tanitad.refs import refcv9_progress as r9p
    from tanitad.train import declared_vs_built as dvb
    import _refcv8_rig as R
    torch.set_num_threads(4)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    argv = ["--arm", "hier", "--sampler", "ddim", "--anchor-v0-conditioned", "--anchor-control-units", "alat",
            "--n-anchors", "20", "--ego-history", "--residual-prior", "ha0_ext_pose"] + R.F1_F6 + \
           ["--r9-progress-head", "--r9-calib-stage", "--w-r9-prog", "1.0", "--w-r9-regime", "1.0",
            "--out", str(out / "unused")]
    parser = T.build_parser()
    args = parser.parse_args(argv)
    cfg = T._pin_trainer_cfg(T.v3.refc_v3_smoke_config(True), args)
    torch.manual_seed(0)
    model = T.v3.RefCV3Model(cfg)
    with torch.no_grad():
        model.core.decoder.anchor_controls.copy_(R.grid().to(model.core.decoder.anchor_controls.dtype))
    frz = r9p.apply_calib_freeze(model)
    mism = [str(m) for m in dvb.check(model, args, parser)]
    r9_mism = [m for m in mism if "r9" in m or "frozen parameters" in m]
    eps = T._synth_episodes(2, cfg.core, seed=0, min_frames=120)
    ds = T.V3Dataset(eps, window=cfg.core.window, max_horizon=60, channels=cfg.core.encoder.in_channels)
    ds.ego_history = True
    n = len(ds)
    batch = torch.utils.data.default_collate([ds[i] for i in range(0, n, max(1, n // 4))][:4])
    snap = {k: v.detach().clone() for k, v in model.state_dict().items()}
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    rows = []
    model.train()
    for step in range(a.steps):
        opt.zero_grad(set_to_none=True)
        torch.manual_seed(100 + step)
        L = T.compute_losses_v3(model, batch, "cpu", mode="diffusion")
        loss = L["loss"]
        loss.backward()
        g = {n: float(p.grad.abs().sum()) for n, p in model.named_parameters() if r9p.is_refcv9_key(n) and p.grad is not None}
        rows.append({"step": step, "loss": float(loss.detach()), "r9_total": float(L["r9_total"]),
                     "loss_is_r9": bool(float(loss.detach()) == float(L["r9_total"])),
                     "l_prog": float(L["r9_l_prog"]), "l_regime": float(L["r9_l_regime"]),
                     "decision_rows": float(L["r9_decel_rows"]), "calib_rows": float(L["r9_calib_rows"]),
                     "regime_acc": float(L["r9_regime_acc"]), "head_grad_abs_sum": g,
                     "frozen_grad_leak": r9p.frozen_grad_leak(model), "model_training_after": bool(model.training)})
        opt.step()
    sd = model.state_dict()
    frozen_bad = [k for k, v in snap.items() if not r9p.is_refcv9_key(k) and not torch.equal(sd[k], v)]
    moved = sorted(k for k, v in snap.items() if r9p.is_refcv9_key(k) and not torch.equal(sd[k], v))
    rec = {"tanitad_file": tanitad.__file__, "torch": torch.__version__, "argv": argv, "freeze": frz,
           "dvb_mismatches_total": len(mism), "dvb_mismatches_r9_or_freeze": r9_mism, "dvb_all": mism,
           "n_windows": n, "batch": 4, "steps": rows,
           "S1_finite_and_loss_is_r9": all(math.isfinite(r["loss"]) and r["loss_is_r9"] for r in rows),
           "S2_all_6_head_tensors_nonzero_grad_every_step": all(len(r["head_grad_abs_sum"]) == 6
                                                                and min(r["head_grad_abs_sum"].values()) > 0 for r in rows),
           "S2b_zero_leak_every_step": all(r["frozen_grad_leak"] == 0.0 for r in rows),
           "S3_frozen_bit_equal": frozen_bad == [], "frozen_mismatch_examples": frozen_bad[:5], "S3_head_moved": moved,
           "S4_dvb_clean_on_r9": r9_mism == [],
           "S5_regime_ce_live": all(r["calib_rows"] > 0 for r in rows) and any(r["l_regime"] > 0 for r in rows),
           "loss_first_last": [rows[0]["loss"], rows[-1]["loss"]]}
    rec["ALL_PASS"] = bool(rec["S1_finite_and_loss_is_r9"] and rec["S2_all_6_head_tensors_nonzero_grad_every_step"]
                           and rec["S2b_zero_leak_every_step"] and rec["S3_frozen_bit_equal"] and len(moved) >= 6
                           and rec["S4_dvb_clean_on_r9"] and rec["S5_regime_ce_live"])
    (out / "SMOKE_R9_STEPS.json").write_text(json.dumps(rec, indent=1, default=str), encoding="utf-8")
    print("SMOKE_STEPS ALL_PASS", rec["ALL_PASS"], "loss", [round(r["loss"], 4) for r in rows],
          "dvb", len(mism), r9_mism[:2])
    return 0 if rec["ALL_PASS"] else 5


if __name__ == "__main__":
    sys.exit(main())
