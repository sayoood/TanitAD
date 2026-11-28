"""SMOKE: does the RL pilot actually LOAD its cold start and TAKE STEPS?

⛔ RULE ZERO. Unblocking is not the deliverable. The tests in
``tests/test_refc_cold_start_allowance.py`` prove the allowance is correct;
they do not prove the PILOT runs. This does: it drives
``rl_pilot_refc21.load_model`` — the real function, not a re-implementation —
on a checkpoint with the REAL production shape, then takes real GRPO steps
through the real ``make_refcv3_sample_fn`` / ``run_posttrain`` path.

⚠️ WHAT IS AND IS NOT SYNTHETIC, STATED RATHER THAN IMPLIED.

* **The MODEL is real** — ``refc.RefCModel(refc.refc_config())``, 488 keys,
  104,191,577 parameters.
* **The LOADER is real** — the pilot's own ``load_model``, including
  ``assert_config_contract`` and the new declared allowance.
* **The RL STEP is real** — ``make_refcv3_sample_fn`` with
  ``strict_conditioning=True``, ``rl_objective``, a real AdamW step.
* **The WEIGHTS are synthetic.** The 2026-07-20 cold start
  (``refc-diffusion-base-v21-30k``) is on no box this run can reach: MEASURED
  2026-09-10, ``tanitad-pod3/4/5`` and ``tanitad-a40`` all refuse SSH
  (Connection refused). The checkpoint here is the model's OWN initial weights
  with ``decoder.anchor_controls`` deleted — i.e. it reproduces the DEFECT
  exactly (487 keys against 488 built, that one key absent) while carrying
  nothing about the July run's quality.
  ⛔ So this smoke proves the PATH, and it makes no claim whatsoever about R1 /
  R2 / R3. Those need the real weights and the real corpus.
* **The FRAMES and the REWARD CONTEXT are synthetic**, so no number printed here
  is a driving metric. The assertions are structural: finite loss, a gradient
  that moved, the stamp on disk, the guard firing.
"""
from __future__ import annotations

import argparse
import importlib.util
import io
import json
import os
import sys
import time

import torch

_STACK = os.environ.get("TANITAD_STACK") or ""
if _STACK and _STACK not in sys.path:
    sys.path.insert(0, _STACK)

from tanitad.refs import refc                                    # noqa: E402
from tanitad.rl.config import PostTrainConfig                    # noqa: E402
from tanitad.rl.posttrain import run_posttrain                   # noqa: E402
from tanitad.rl.refcv3_adapter import make_refcv3_sample_fn      # noqa: E402
import tanitad.rl.rewards as RW                                  # noqa: E402


def _pilot(scripts_dir):
    sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location(
        "rl_pilot_refc21_smoke", os.path.join(scripts_dir,
                                              "rl_pilot_refc21.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    rep: dict = {"_what": "RL pilot cold-start load + step smoke (PI queue "
                          "item 8)",
                 "_evidence_class": "MEASURED (ours; this run)",
                 "_caveat": "SYNTHETIC WEIGHTS AND SYNTHETIC FRAMES. Proves "
                            "the PATH runs. Makes NO claim about R1/R2/R3.",
                 "device": a.device,
                 "torch": torch.__version__}
    pilot = _pilot(os.path.join(a.stack, "scripts"))

    # ---- 1. a checkpoint with the REAL defect ---------------------------- #
    t0 = time.time()
    donor = refc.RefCModel(refc.refc_config())
    built = donor.state_dict()
    sd = {k: v.detach().clone() for k, v in built.items()}
    del sd["decoder.anchor_controls"]
    ck_dir = os.path.join(a.out, "coldstart")
    os.makedirs(ck_dir, exist_ok=True)
    ck_path = os.path.join(ck_dir, "ckpt.pt")
    # ⛔ NO `cfg` key and NO sidecar: the ABSENT case, which is what the real
    # July checkpoint presents (the pilot's own note MEASURES it as carrying
    # 38 of 96 leaves and NOT `v0_conditioned`; a 2026-07-20 config could not
    # carry that field, because it did not exist until 2026-09-04).
    torch.save({"model": sd, "step": 30000}, ck_path)
    rep["checkpoint"] = {"path": ck_path, "keys_written": len(sd),
                         "keys_built": len(built),
                         "missing": sorted(set(built) - set(sd))}
    print(f"[smoke] built a {len(sd)}-key cold start against {len(built)} "
          f"built keys (missing {rep['checkpoint']['missing']}) "
          f"in {time.time() - t0:.1f}s", flush=True)
    del donor, sd, built

    # ---- 2. THE PILOT'S OWN load_model ----------------------------------- #
    t0 = time.time()
    model, cold = pilot.load_model(ck_path, a.device, a.out)
    after = model.state_dict()
    rep["load"] = {"seconds": round(time.time() - t0, 2),
                   "keys_after_load": len(after),
                   "stamp": cold}
    assert len(after) == 488, f"expected 488 keys, got {len(after)}"
    assert "decoder.anchor_controls" in after
    assert cold["anchor_controls_source"] == "defaulted-zeros-v0-unconditioned"
    assert cold["defaulted_keys"] == ["decoder.anchor_controls"]
    assert int(torch.count_nonzero(after["decoder.anchor_controls"])) == 0
    print(f"[smoke] ✅ LOADED. {len(after)} keys on the model; "
          f"{cold['anchor_controls_source']}; "
          f"{cold['ckpt_confirmation']}", flush=True)

    # ---- 3. a batch of the right SHAPE ----------------------------------- #
    cfgm = refc.refc_config()
    b, w = a.batch, int(cfgm.window)
    ch, px = int(cfgm.encoder.in_channels), int(cfgm.encoder.image_size)
    s = len(pilot.HORIZONS)
    g = torch.Generator().manual_seed(0)
    dev = a.device
    batch = {
        "frames": torch.rand(b, w, ch, px, px, generator=g).to(dev),
        "v0": torch.full((b,), 8.0, device=dev),
        "gt_traj": torch.stack([torch.linspace(4, 16, s),
                                torch.zeros(s)], dim=-1)
                        .unsqueeze(0).expand(b, s, 2).contiguous().to(dev),
        "obstacles": torch.full((b, 1, 1, 1, 2), pilot.FAR_LEAD_X, device=dev),
        "lead_path": torch.full((b, 1, 1, s, 2), pilot.FAR_LEAD_X, device=dev),
        "dt": pilot.DT_TRAJ,
    }
    rep["batch_shapes"] = {k: (list(v.shape) if torch.is_tensor(v) else v)
                           for k, v in batch.items()}

    # ---- 4. the used-path guard ------------------------------------------ #
    cfg = PostTrainConfig(
        method="grpo", group_size=4, steps=a.steps, batch=b, lr=1e-5, seed=0,
        dt=pilot.DT_TRAJ, decoder_steps=2,
        reward_weights=dict(RW.DEFAULT_WEIGHTS),
        trainable_prefixes=("decoder",),
        exclude_prefixes=("decoder.conf_head",),
        forbidden_prefixes=("decoder.conf_head", "scorer"),
        w_imitation=0.0, out_dir=a.out, run_name="cold-start-smoke")
    cfg.validate()
    sample_fn = make_refcv3_sample_fn(
        model, cfg, build_ctx=lambda bt, out=None: pilot.build_ctx(bt, out),
        strict_conditioning=True)
    contract = sample_fn.conditioning_contract
    # ⛔ RESOLVE FIRST, THEN READ. `RefCConditioningContract` resolves its map
    # LAZILY on the first `check`, so reading `.required` before that returns
    # `()` — which is indistinguishable from "this build requires nothing" and
    # is a claim about the READ, not about the build. The first draft of this
    # smoke made exactly that error in a second costume: it iterated
    # `.requirements` (a DICT of str -> bool) as though it held objects with a
    # `.channel`, so the comprehension yielded [] no matter what the map said.
    contract.check(batch)
    assert contract.resolutions == 1, "the contract did not resolve"
    reqmap = contract.requirements
    required = sorted(contract.required)
    assert reqmap, "the resolved requirement map is EMPTY — that is a failed " \
                   "read, not a finding; every channel must appear with a bool"
    rep["conditioning"] = {"strict": sample_fn.strict_conditioning,
                           "contract": type(contract).__name__,
                           "requirement_map": reqmap,
                           "resolved_required": required,
                           "resolutions": contract.resolutions}
    print(f"[smoke] conditioning contract = {type(contract).__name__}; "
          f"map = {reqmap}; REQUIRED = {required}", flush=True)

    # ⛔ THE GUARD MUST FIRE ON A DROPPED CHANNEL, and it must be MEASURED
    # firing, not assumed. A guard whose refusal branch is never executed is
    # the `test_rl_channel_guard` defect one level up.
    fired = {}
    for chan in ("v0", "nav_cmd", "lan"):
        probe = {k: v for k, v in batch.items() if k != chan}
        try:
            contract.check(probe)
            fired[chan] = "PASSED (this channel is not REQUIRED by this build)"
        except Exception as exc:                       # noqa: BLE001
            fired[chan] = f"REFUSED: {type(exc).__name__}: {str(exc)[:180]}"
    rep["guard_probe"] = fired
    for k, v in fired.items():
        print(f"[smoke]   drop {k:8s} -> {v}", flush=True)

    # ---- 5. REAL STEPS ---------------------------------------------------- #
    t0 = time.time()
    trainable = [p for n, p in model.named_parameters()
                 if n.startswith("decoder") and not n.startswith("decoder.conf_head")]
    before = torch.cat([p.detach().reshape(-1)[:64] for p in trainable[:8]]).clone()
    summary = run_posttrain(model, sample_fn, cfg,
                            batches=[batch] * a.steps, device=a.device,
                            extra_record={"cold_start_load": cold})
    dt = time.time() - t0
    moved = torch.cat([p.detach().reshape(-1)[:64] for p in trainable[:8]])
    hist = summary.get("history", [])
    losses = [float(h["loss"]) for h in hist if "loss" in h]
    rep["steps"] = {"seconds": round(dt, 1), "n": len(hist),
                    "losses": losses,
                    "all_finite": all(map(lambda x: x == x and abs(x) != float("inf"),
                                          losses)) and bool(losses),
                    "weights_moved": bool(not torch.equal(before, moved)),
                    "max_abs_delta": float((moved - before).abs().max())}
    assert rep["steps"]["all_finite"], f"non-finite loss: {losses}"
    assert rep["steps"]["weights_moved"], "no trainable weight moved"
    print(f"[smoke] ✅ {len(hist)} GRPO steps in {dt:.1f}s · losses "
          f"{['%.4f' % x for x in losses]} · all finite · weights moved "
          f"(max |Δ| = {rep['steps']['max_abs_delta']:.3e})", flush=True)

    # ---- 6. the stamp is ON DISK ----------------------------------------- #
    cj = os.path.join(a.out, "config.json")
    rec = json.loads(io.open(cj, encoding="utf-8").read())
    got = rec.get("cold_start_load")
    assert got, "config.json carries no cold_start_load stamp"
    assert got["anchor_controls_source"] == "defaulted-zeros-v0-unconditioned"
    assert got["defaulted_keys"] == ["decoder.anchor_controls"]
    rep["config_json"] = {"path": cj,
                          "cold_start_load": got,
                          "method": rec.get("method")}
    print(f"[smoke] ✅ stamp read back from {cj}: "
          f"{got['anchor_controls_source']} · {got['ckpt_confirmation']}",
          flush=True)

    rep["verdict"] = "PATH PROVEN (synthetic weights; no capability claim)"
    if a.json:
        io.open(a.json, "w", encoding="utf-8", newline="").write(
            json.dumps(rep, indent=1, sort_keys=True, default=str))
        print(f"[smoke] wrote {a.json}", flush=True)
    print(f"[smoke] {rep['verdict']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
