"""Q3b / Q6 -- does refcv6's F3 per-layer (cascade) loss REACH the trainer on the PRODUCTION
forward?  Measured on the real `train()` (synthetic rig, CPU), not on the decoder in isolation.

WHY THIS INSTRUMENT EXISTS. The live run's metrics.jsonl carries NO `cascade` key in 4,024 rows
(read-only on Thor, 2026-09-26), and the checkpoint's `core.decoder.cascade.control_heads.{0,1,2}`
/ `conf_heads.{0,1,2}` / `adaln.{0,1,2}` are bit-identical at steps 1,000 / 5,000 / 30,000
(raw/q2_positional_tensors.json). Reading the code:
  * the decoder exports `layer_u0_hat` / `layer_logits`            (refc.py:3390-3393)
  * RefCModel.forward builds its `out` from a WHITELIST of decoder keys and never copies those
    two                                                               (refc.py:4466-4534)
  * the loss runs only `if _rv6f.f3_per_layer and "layer_u0_hat" in out`
                                                                     (refc_v3_train.py:3964)
The 2026-09-22 review proved F3 live on `AnchoredDiffusionDecoder.forward` directly
(diag_f1_f9_liveness.py builds the DECODER) -- i.e. on the component, never on the consumer.

ARMS (each is the REAL train(), 3 steps, synthetic episodes, the run's diffusion flags):
  A  as shipped                               -> the prediction is: NO `cascade` key (RED for the
                                                 guard "F3 on => the per-layer loss runs")
  B  RefCModel.forward patched to pass the two decoder keys through (the one-line fix)
                                              -> `cascade` key present and finite (the loss block is
                                                 otherwise alive: the defect is the whitelist alone)
  C  as shipped, F3 OFF                       -> no `cascade` key (the key's absence is not an
                                                 artefact of the rig: it is also absent where it must be)
Each arm also reports whether stage-0's cascade head received ANY gradient on its first step.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

C.bootstrap()
import torch  # noqa: E402

SCR = C.SCRATCH.parent / "q3b_runs"
ANCH = str(C.KIT / "data/anchors/refc_anchors_6s_v0cond_alat_117.pt")
BASE = ["--arm", "hier", "--smoke", "--synth-episodes", "2", "--steps", "3", "--batch", "2",
        "--device", "cpu", "--save-every", "100", "--log-every", "1",
        "--sampler", "ddim", "--anchors", ANCH, "--anchor-v0-conditioned", "--n-anchors", "117"]
F_ON = ["--f1-random-t", "--f2-dd-step", "--f3-per-layer", "--f4-adaln", "--f5-focal",
        "--f5-emitting-conf", "--f6-w-u0-zero"]
F_OFF = ["--f1-random-t", "--f2-dd-step", "--f4-adaln", "--f5-focal", "--f5-emitting-conf",
         "--f6-w-u0-zero"]


def run_arm(name: str, argv: list, patch_passthrough: bool) -> dict:
    T = C.trainer_module()
    from tanitad.refs import refc
    out_dir = SCR / name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    orig_fwd = refc.RefCModel.forward
    grad_probe = {}

    def fwd_passthrough(self, *a, **k):
        o = orig_fwd(self, *a, **k)
        dec = self.decoder
        if getattr(dec, "cascade", None) is not None and getattr(dec, "_rv6_layer_u0", None):
            o["layer_u0_hat"] = dec._rv6_layer_u0
            o["layer_logits"] = dec._rv6_layer_conf
        return o

    orig_step = torch.optim.Optimizer.step
    model_ref = {}

    def fwd_record(self, *a, **k):
        model_ref["m"] = self
        dec = self.decoder
        if "hooked" not in model_ref and getattr(dec, "cascade", None) is not None:
            model_ref["hooked"] = True
            probes = {"control_head0": dec.cascade.control_heads[0].weight,
                      "conf_head0": dec.cascade.conf_heads[0].weight,
                      "control_headL": dec.cascade.control_heads[-1].weight}
            if getattr(dec, "adaln", None) is not None:
                probes["adaln0"] = dec.adaln[0].scale_shift_mlp[1].weight
                probes["adalnL"] = dec.adaln[-1].scale_shift_mlp[1].weight
            for nm, prm in probes.items():
                grad_probe.setdefault("grad_abs_sum", {})[nm] = 0.0
                grad_probe.setdefault("n_grad_events", {})[nm] = 0

                def _h(g, _nm=nm):
                    grad_probe["grad_abs_sum"][_nm] += float(g.detach().abs().sum())
                    grad_probe["n_grad_events"][_nm] += 1
                    return g
                prm.register_hook(_h)
        return (fwd_passthrough if patch_passthrough else orig_fwd)(self, *a, **k)

    def step_probe(self, *a, **k):
        m = model_ref.get("m")
        if m is not None and "stage0" not in grad_probe and \
                getattr(m.decoder, "cascade", None) is not None:
            h = m.decoder.cascade.control_heads[0].weight
            c0 = m.decoder.cascade.conf_heads[0].weight
            a0 = m.decoder.adaln[0].scale_shift_mlp[1].weight if m.decoder.adaln is not None else None
            grad_probe["stage0"] = {
                "control_head0_grad": None if h.grad is None else float(h.grad.abs().sum()),
                "conf_head0_grad": None if c0.grad is None else float(c0.grad.abs().sum()),
                "adaln0_grad": (None if a0 is None or a0.grad is None
                                else float(a0.grad.abs().sum())),
                "control_headL_grad": (lambda g: None if g is None else float(g.abs().sum()))(
                    m.decoder.cascade.control_heads[-1].weight.grad)}
        return orig_step(self, *a, **k)

    refc.RefCModel.forward = fwd_record
    torch.optim.Optimizer.step = step_probe
    try:
        T.train(T.build_parser().parse_args(argv + ["--out", str(out_dir)]))
        err = None
    except SystemExit as e:
        err = f"SystemExit: {e}"
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"
    finally:
        refc.RefCModel.forward = orig_fwd
        torch.optim.Optimizer.step = orig_step
    rows = []
    mp = out_dir / "metrics.jsonl"
    if mp.exists():
        rows = [json.loads(x) for x in mp.read_text(encoding="utf-8").splitlines() if x.strip()]
    loss_rows = [r for r in rows if "loss" in r]
    return {"arm": name, "error": err, "n_rows": len(rows), "n_loss_rows": len(loss_rows),
            "cascade_in_rows": sum(1 for r in loss_rows if "cascade" in r),
            "cascade_values": [r.get("cascade") for r in loss_rows],
            "grad_probe_all_steps": {k: v for k, v in grad_probe.items() if k != "stage0"},
            "loss_keys_first_row": sorted(loss_rows[0].keys()) if loss_rows else []}


if __name__ == "__main__":
    C.ram_guard("q3b_f3_cascade_reach (light job; the brief's 8 GB floor applies to every job)")
    SCR.mkdir(parents=True, exist_ok=True)
    res = {"what": "F3 per-layer loss reach on the PRODUCTION forward (real train(), synthetic rig)",
           "evidence_class": "MEASURED (ours, CPU)",
           "arms": [run_arm("A_as_shipped", BASE + F_ON, False),
                    run_arm("B_passthrough_fix", BASE + F_ON, True),
                    run_arm("C_f3_off", BASE + F_OFF, False)]}
    a, b, c = res["arms"]
    res["verdict"] = {
        "A_no_cascade_as_shipped": a["error"] is None and a["n_loss_rows"] > 0 and a["cascade_in_rows"] == 0,
        "B_cascade_present_with_fix": b["error"] is None and b["cascade_in_rows"] == b["n_loss_rows"] > 0,
        "C_no_cascade_with_f3_off": c["error"] is None and c["cascade_in_rows"] == 0}
    for r in res["arms"]:
        print(json.dumps({k: v for k, v in r.items() if k != "loss_keys_first_row"}))
    print("VERDICT", res["verdict"])
    C.write_json("q3b_f3_cascade_reach.json", res)
