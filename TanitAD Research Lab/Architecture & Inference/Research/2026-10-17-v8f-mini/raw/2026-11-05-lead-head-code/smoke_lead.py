"""Real-C0 smoke for the lead head (dev box, CPU, no training): load the real C0 init checkpoint into lead-head models (add; token with both heads), report the fresh keys, the parameter counts of every lever group,
whether the step-0 forward equals the C0 model's (add: EXACTLY; token: differs by construction), and run the G-LIVE lead probe (and the level probe under token) on the real-weight model with a synthetic probe batch.
usage: python -I smoke_lead.py <patched tree root, e.g. F:/TanitAD-scratch/motion_c/lead_build/new>   (writes to stdout)
The synthetic batch makes the printed W-for-10-percent number meaningless: that number comes from box 2 (--selfcheck-only on the real train corpus)."""
import sys
import tempfile
from pathlib import Path

import torch

TREE = Path(sys.argv[1])
for p in (TREE / "stack", TREE / "taniteval", TREE / "stack" / "scripts", TREE / "stack" / "tests"):
    sys.path.insert(0, str(p))
import test_v8f_lead_head as T  # noqa: E402  (the test module's helpers: synthetic world, lead file, batch, labels)
from tanitad.data import v8f_mini_data as D  # noqa: E402
from tanitad.models import v8f_mini as M  # noqa: E402
import v8f_mini_train as TR  # noqa: E402

C0 = T.C0_INIT
print("C0 init", C0, "exists", C0.exists())
ck = torch.load(C0, map_location="cpu", weights_only=False)
cfg_d = {**ck["cfg"]}
with tempfile.TemporaryDirectory() as td:
    root = Path(td)
    w = D.make_synthetic_world(root, n_train=6, n_eval=2, T=60, seed=0)
    npz = T.write_lead_npz(root / "lead_train.npz", w["names"]["train"])
    world = {"root": root, "npz": npz, **w}
    b, J = T.make_batch(world, hmax=cfg_d["hmax"])
    LL = T.LLof(world, J)
    mu, sd = M.cons_stats(J)

    def mk(**k):
        torch.manual_seed(0)
        return M.V8FMini(M.V8FMiniConfig(**{**cfg_d, **k}), M.P10Prior.synthetic(0), mu, sd)
    off = mk()
    off.load_state_dict(ck["model"], strict=True)
    off.eval()
    bb = T.with_both(b, LL)
    for tag, kw in (("add (lead only)", dict(lead_head="on", w_lead=1.0)),
                    ("add (level + lead)", dict(level_head="on", w_level=1.0, lead_head="on", w_lead=1.0)),
                    ("token (level + lead)", dict(level_head="on", w_level=1.0, lead_head="on", w_lead=1.0, aux_cond="token")),
                    ("token (lead only)", dict(lead_head="on", w_lead=1.0, aux_cond="token"))):
        on = mk(**kw)
        rec = TR.load_init_ckpt(str(C0), on, on.cfg)
        sd_on = on.state_dict()
        identical = all(torch.equal(sd_on[k], v) for k, v in ck["model"].items())
        on.eval()
        with torch.no_grad():
            a, c = off.forward_infer(bb), on.forward_infer(bb)
        same = all(torch.equal(a[k], c[k]) for k in a if torch.is_tensor(a[k]))
        lv = on.built_levers()
        print(f"[{tag}] tensors {rec['n_tensors']} mismatch {rec['n_mismatch_after_load']} fresh {len(rec['fresh_keys'])} {sorted(rec['fresh_keys'])}")
        print(f"   old keys bit-identical: {identical} | step-0 forward equals C0's exactly: {same} | max |d traj60| {float((a['traj60'] - c['traj60']).abs().max()):.6g}")
        print(f"   lever params {{k: v for k, v in lv['params'].items() if k in ('LEVEL', 'LEAD', 'AUX')}} -> " + str({k: v for k, v in lv["params"].items() if k in ("LEVEL", "LEAD", "AUX")}) + f" | aux_cond {lv.get('aux_cond', 'add')}")
        on.train()
        pr = TR.lead_head_probe(on, bb, on.cfg)
        print(f"   lead probe pass {pr['pass']} head_grad_diff {pr['head_grad_diff']:.4g} presence_row {pr['head_grad_diff_presence_row']:.4g} gap_row {pr['head_grad_diff_gap_row']:.4g} "
              f"detach_control {pr['head_grad_at_w0_detach_control']} allfalse_term {pr['control_allfalse_term']} plan_moves {pr['plan_moves_with_token']:.4g}")
        print(f"   step0 (synthetic batch, meaningless) lead_share {pr['step0_lead_share']:.4f} w_lead_for_10pct {pr['w_lead_for_10pct_share']:.4f}")
        if on.cfg.level_head == "on":
            lp = TR.level_head_probe(on, bb, on.cfg)
            print(f"   level probe pass {lp['pass']} keys {sorted(k for k in lp if k.startswith('e_'))} plan_moves {lp['plan_moves_with_token']:.4g}")
print("SMOKE-DONE")
