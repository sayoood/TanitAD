"""c2 zero-init identity (LEADCOND). (a) tree_patch with --lead-cond on, loaded from an LS-shape checkpoint written by tree_base (only lead_cond.weight / lead_cond.bias missing), before any step:
every tensor output of forward (p_gt 0) and forward_infer equals the output of tree_base's model with the same checkpoint: max abs diff 0.0.
(b) from-scratch build, same seed, lever on vs off in tree_patch: the state dicts are equal apart from the two new keys; the next random draw after the build is the same (no other draw moved).
Controls: a build that does NOT restore the RNG state moves the next draw (reads non-zero); a non-zero lead_cond.weight moves the outputs (reads non-zero). Usage: c2_zero_init.py"""
import json
import os
import subprocess
import sys

import torch

LC = "F:/TanitAD-scratch/v8proof/leadcond"
CK = f"{LC}/checks"
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
w = lambda *p: "\\".join(p).replace("/", "\\")


def env_for(tree):
    root = f"{LC}/work/tree_{tree}"
    return dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONIOENCODING="utf-8", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1",
                TMP=w(LC, "work", "tmp"), TEMP=w(LC, "work", "tmp"), PYTHONPATH=";".join([w(root, "stack"), w(root, "taniteval"), w(root, "stack", "scripts"), w(CK)]))


import lc_common as C  # noqa: E402

res = {}
outs = {}
for tree, lever in (("base", "off"), ("patch", "on"), ("patch", "off")):
    o = f"{LC}/work/tmp/c2_{tree}_{lever}.pt"
    r = subprocess.run([PY, f"{CK}/lc_dump.py", tree, C.BASE_CKPT, o, lever], env=env_for(tree), capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 0, r.stderr[-3000:]
    outs[(tree, lever)] = torch.load(o, weights_only=False)
A, B, P0 = outs[("base", "off")], outs[("patch", "on")], outs[("patch", "off")]
diffs = {}
for part in ("forward", "infer"):
    ks = sorted(A[part])
    diffs[part] = {"n_keys": len(ks), "keys_equal": sorted(B[part]) == ks and sorted(P0[part]) == ks,
                   "max_abs_diff_on_vs_base": max(float((B[part][k].float() - A[part][k].float()).abs().max()) for k in ks if A[part][k].numel()),
                   "max_abs_diff_off_vs_base": max(float((P0[part][k].float() - A[part][k].float()).abs().max()) for k in ks if A[part][k].numel())}
res["a_outputs"] = {"diffs": diffs, "missing_keys_on": B["missing"], "n_state_keys": [A["n_keys_state"], B["n_keys_state"]]}
# (b) from-scratch build
C.use_tree("patch")
from tanitad.models import v8f_mini as M  # noqa: E402

batch, J = C.world_batch("c2b")
torch.manual_seed(0)
_, _, m_off = C.build_from_args(["--lead-sup", "on", "--w-lead-sup", "1.0"], J, seed=7)
draw_off = torch.rand(3).tolist()
_, _, m_on = C.build_from_args(["--lead-sup", "on", "--w-lead-sup", "1.0", "--lead-cond", "on"], J, seed=7)
draw_on = torch.rand(3).tolist()
so, sn = m_off.state_dict(), m_on.state_dict()
new_keys = sorted(set(sn) - set(so))
same = all(torch.equal(so[k], sn[k]) for k in so)
# control: a build that does not restore the RNG state moves the next draw
import torch.nn as nn  # noqa: E402

_, _, m_ctl = C.build_from_args(["--lead-sup", "on", "--w-lead-sup", "1.0"], J, seed=7)
nn.Linear(2, 32)                                                        # an unrestored default init: draws
draw_ctl = torch.rand(3).tolist()
# control: non-zero weight moves the outputs
m_on.eval()
bt = {k: v for k, v in batch.items() if k in M.INFERENCE_INPUT_KEYS}
with torch.no_grad():
    m_on.head_o[2].weight.normal_(0, 0.1)                                # wake O's zero-init head so the plan can respond to its conditioning
    o0 = m_on.forward_infer(bt)
    m_on.lead_cond.weight.fill_(0.3)
    o1 = m_on.forward_infer(bt)
mv = {k: float((o1[k] - o0[k]).abs().max()) for k in ("traj60", "goal_logits", "lat_logits")}
res["b_scratch"] = {"new_keys": new_keys, "other_keys_equal": bool(same), "n_keys_off_on": [len(so), len(sn)], "next_draw_equal": draw_off == draw_on, "next_draw_off": draw_off, "next_draw_on": draw_on,
                    "control_unrestored_init_moves_next_draw": draw_ctl != draw_off, "control_nonzero_weight_moves_outputs_max_abs": mv}
ok = (all(d["keys_equal"] and d["max_abs_diff_on_vs_base"] == 0.0 and d["max_abs_diff_off_vs_base"] == 0.0 for d in diffs.values()) and sorted(res["a_outputs"]["missing_keys_on"]) == ["lead_cond.bias", "lead_cond.weight"]
      and new_keys == ["lead_cond.bias", "lead_cond.weight"] and same and draw_off == draw_on and res["b_scratch"]["control_unrestored_init_moves_next_draw"] and all(v > 0.0 for v in mv.values()))
res["PASS"] = bool(ok)
C.dump(f"{CK}/c2_zero_init.json", res)
print(json.dumps(res)[:1500])
print("C2", "PASS" if ok else "FAIL")
