#!/usr/bin/env python3
"""PREFLIGHT SMOKE for the wired control-space sampler. ⛔ NOT AN ARM LAUNCH.

No optimizer is constructed, no `optimizer.step()` is taken, no checkpoint is
written and no result is produced. It answers one question the mutation proofs
cannot: does the CONTROL-SPACE path, wired into the real refcv3 model on a real
batch, produce (a) the right shapes, (b) FLYABLE candidates, and (c) a `logp`
whose gradient actually reaches the decoder? A guard that refuses the wrong path
is worth nothing if the right path does not run.

Also enumerates every arm and builds + validates its `PostTrainConfig`, so a
config error surfaces here rather than after minutes of paid start-up.
"""
import importlib.util
import json
import os
import sys
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ["TANITAD_REPO"]
spec = importlib.util.spec_from_file_location(
    "rlmin", os.path.join(REPO, "stack", "scripts", "rl_refcv3_min.py"))
M = importlib.util.module_from_spec(spec)
sys.modules["rlmin"] = M
spec.loader.exec_module(M)
torch = M.torch

W = sys.argv[1]
LABELS = sys.argv[2]

a = SimpleNamespace(
    ckpt=os.path.join(W, "base", "ckpt_step40284_frozen.pt"),
    config=os.path.join(W, "base", "config.json"), expect_step=40284,
    episodes=os.path.join(W, "fit120"), labels=LABELS,
    lead_block=os.path.join(W, "fit120_lead_block.npz"),
    lru=4, batch=2, group=4, noise=0.04, seed=0, steps=2,
    out_dir=os.path.join(W, "_cs_smoke"), arm="rl", device="cuda",
    lead_mode="track", readout_windows=8)

print("== EVERY ARM BUILDS AND VALIDATES ITS CONFIG (0 GPU)")
prov_stub = {"decoder_steps": 0, "step": 40284}
bad = []
for arm in M.ARMS:
    try:
        sp, hyp = M.assert_sample_space(arm)
        cfg = M.make_cfg(arm, a, prov_stub, a.out_dir)
        cfg.validate()
        assert cfg.to_dict()["sample_space"] == sp
        print("   %-14s space=%-8s hyp=%-5s noise_mode=%-11s lr=%-7s steps=%-5d veto=%s"
              % (arm, sp, hyp, cfg.noise_mode, cfg.lr, cfg.steps, cfg.veto_enabled))
    except SystemExit as e:
        print("   %-14s REFUSED: %s" % (arm, str(e)[:90])); bad.append(arm)
    except Exception as e:
        print("   %-14s ERROR  : %s: %s" % (arm, type(e).__name__, str(e)[:90]))
        bad.append(arm)
if bad:
    raise SystemExit("[cs-smoke] arms failed to configure: %s" % bad)

device = "cuda" if torch.cuda.is_available() else "cpu"
model, cfg_model, _t, prov = M.load(a, device)
corp = M.open_corpus(a.episodes, a.labels, cfg_model, prov, a.lru)
lead, _lm, _li = M.load_lead_block(a.lead_block)
wis = M.scoreable_windows(corp, lead)[:2]
batch = M.build_batch(corp, lead, wis, device, with_gt=True)

out = {}
for arm in ("rl", "reg_metre"):
    pcfg = M.make_cfg(arm, a, prov, a.out_dir)
    pcfg.validate()
    # select_trainable SETS requires_grad on the model and returns a REPORT
    rep = M.select_trainable(model, pcfg)
    train = [p for p in model.parameters() if p.requires_grad]
    n_tr = len(train)
    assert n_tr == rep["n_trainable_tensors"], (n_tr, rep["n_trainable_tensors"])
    fn = M.make_sample_fn(model, None, echo_reward=False, gt_bar=bool(pcfg.use_gt_bar))
    traj2, logp, ctx, extras = fn(batch, pcfg)
    spec_r = M.RewardSpec(weights=dict(pcfg.reward_weights), dt=pcfg.dt)
    r = spec_r(traj2, ctx)
    adv = (r - r.mean(dim=2, keepdim=True)).detach()
    loss = -(logp * adv).sum()
    loss.backward()
    g = sum(float(p.grad.abs().sum()) for p in train if p.grad is not None)
    ngrad = sum(1 for p in train if p.grad is not None and float(p.grad.abs().sum()) > 0)
    # flyability of what was actually explored
    ctl = M.CS.controls_from_path(traj2[..., 1:, :], dt=M.DT_REWARD_S)
    viol = float(M.CS.envelope_violation(ctl).max())
    print("\n== ARM %s (space=%s)" % (arm, pcfg.sample_space))
    print("   traj2 %s  logp %s  reward %s" % (tuple(traj2.shape), tuple(logp.shape),
                                               tuple(r.shape)))
    print("   logp.requires_grad %s  grad_fn %s" % (logp.requires_grad,
                                                    type(logp.grad_fn).__name__))
    print("   trainable params %d   params WITH a non-zero grad %d   |grad| sum %.6e"
          % (n_tr, ngrad, g))
    print("   envelope_violation of the EXPLORED fan: %.6f" % viol)
    out[arm] = {"sample_space": pcfg.sample_space,
                "traj_shape": list(traj2.shape), "logp_shape": list(logp.shape),
                "logp_requires_grad": bool(logp.requires_grad),
                "n_trainable_tensors": int(n_tr),
                "n_tensors_with_nonzero_grad": int(ngrad),
                "grad_abs_sum": g, "envelope_violation_max": viol}
    model.zero_grad(set_to_none=True)

print("\n== VERDICT")
ok = (out["rl"]["n_tensors_with_nonzero_grad"] > 0
      and out["rl"]["grad_abs_sum"] > 0.0
      and out["rl"]["envelope_violation_max"] == 0.0)
print("   control-space `rl`: gradient reaches the decoder = %s ; explored fan FLYABLE = %s"
      % (out["rl"]["grad_abs_sum"] > 0.0, out["rl"]["envelope_violation_max"] == 0.0))
print("   metre-space `reg_metre` envelope_violation = %.6f (the regression need not be 0)"
      % out["reg_metre"]["envelope_violation_max"])
print("   => %s" % ("PREFLIGHT PASS" if ok else "PREFLIGHT FAIL"))
out["_verdict"] = "PASS" if ok else "FAIL"
out["_what"] = ("preflight smoke of the wired control-space sampler on the real model; "
                "NOT an arm launch -- no optimizer, no step, no checkpoint, no result")
out["_tier"] = "T0 preflight"
out["_evidence_class"] = "MEASURED (ours)"
with open(sys.argv[3], "w", encoding="utf-8") as fh:
    json.dump(out, fh, indent=1)
print("wrote %s" % sys.argv[3])
raise SystemExit(0 if ok else 1)
