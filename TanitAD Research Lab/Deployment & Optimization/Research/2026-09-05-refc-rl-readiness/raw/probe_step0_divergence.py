"""probe_step0_divergence.py — WHY does the RL preflight's step-5 gate fail on refcv3?

The gate asserts trajectory_divergence(live.anchor_traj, reference.anchor_traj) == 0
exactly, where `reference` is a frozen deepcopy of the live model on the SAME batch
(the TRAIN-C5 mode-mismatch guard that caught 2.774 m² on the v2.1 pilot). On refcv3
@ 40,284 on the dev-box 4060 it fired. Three candidates, separated here:
  (1) CUDA non-determinism (live vs live-REPEAT also != 0)        -> tolerance, not a bug
  (2) a real mode mismatch (live == live-repeat, live != reference) -> investigate state
  (3) determinism restored by torch.use_deterministic_algorithms   -> set it in the driver
Evidence class MEASURED (ours). Dev box only; the training pod is never touched.
"""
import json
import os
import random
import sys
from types import SimpleNamespace

os.environ.setdefault("TANITAD_REPO", r"C:\Users\Admin\refcv4b_repo")
sys.path.insert(0, os.path.join(os.environ["TANITAD_REPO"], "stack", "scripts"))
import rl_refcv3_min as M   # noqa: E402
import torch                # noqa: E402
from tanitad.rl.anchor import ReferencePolicy, trajectory_divergence  # noqa: E402

W = r"C:\Users\Admin\rl_refcv3_min"
a = SimpleNamespace(ckpt=os.path.join(W, "base", "ckpt_step40284_frozen.pt"),
                    config=os.path.join(W, "base", "config.json"), expect_step=40284,
                    episodes=os.path.join(W, "fit8"),
                    labels=r"C:\Users\Admin\tanitad-wt\_s2build\release\v72\s2_labels_v7.2_train.jsonl.gz",
                    lead_block=os.path.join(W, "fit8_lead_block.npz"), lru=8, batch=2, seed=0)
out = {"_what": "step-0 divergence decomposition, refcv3 @ 40284, dev-box 4060",
       "_evidence_class": "MEASURED (ours)", "torch": torch.__version__}


def run(tag, deterministic: bool):
    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg, targs, prov = M.load(a, device)
    corp = M.open_corpus(a.episodes, a.labels, cfg, prov, a.lru)
    lead, _, _ = M.load_lead_block(a.lead_block)
    wis = M.scoreable_windows(corp, lead)
    rng = random.Random(0)
    b = M.build_batch(corp, lead, sorted(rng.sample(wis, 2)), device, with_gt=False)
    steps = int(prov["decoder_steps"])
    ref = ReferencePolicy(model).to(device)
    with torch.no_grad():
        l1 = model(b["frames"], nav_cmd=b["nav_cmd"], v0=b["v0"], steps=steps)
        l2 = model(b["frames"], nav_cmd=b["nav_cmd"], v0=b["v0"], steps=steps)
        r1 = ref(b["frames"], nav_cmd=b["nav_cmd"], v0=b["v0"], steps=steps)
        r2 = ref(b["frames"], nav_cmd=b["nav_cmd"], v0=b["v0"], steps=steps)
    d = lambda x, y: float(trajectory_divergence(x["anchor_traj"], y["anchor_traj"]).abs().max())
    rec = {"deterministic_algos": deterministic, "device": device,
           "model_training_flag": bool(model.training), "ref_training_flag": bool(ref.module.training),
           "live_vs_live_repeat_m2": d(l1, l2), "ref_vs_ref_repeat_m2": d(r1, r2),
           "live_vs_ref_m2": d(l1, r1), "live2_vs_ref2_m2": d(l2, r2),
           "logits_live_vs_ref_maxabs": float((l1["anchor_logits"] - r1["anchor_logits"]).abs().max()),
           "sel_idx_agree": float((l1["sel_idx"] == r1["sel_idx"]).float().mean()),
           "any_submodule_in_train_mode": sorted({n for n, m in model.named_modules() if m.training})[:10]}
    # which submodules differ in STATE between live and reference? (buffers / params)
    diff = []
    for (n, p), (_, q) in zip(model.state_dict().items(), ref.module.state_dict().items()):
        if not torch.equal(p, q):
            diff.append(n)
    rec["state_dict_keys_differing"] = diff[:20]
    rec["n_state_dict_keys_differing"] = len(diff)
    out[tag] = rec
    print(tag, json.dumps(rec, indent=1), flush=True)
    del model, ref, corp
    torch.cuda.empty_cache()


run("default", deterministic=False)
run("deterministic", deterministic=True)
with open(os.path.join(W, "probe_step0_divergence.json"), "w", encoding="utf-8") as fh:
    json.dump(out, fh, indent=1)
print("->", os.path.join(W, "probe_step0_divergence.json"))
