"""Integration check of ``integration.diff`` on the REAL RefCV3Model code path (a tiny refcv8 build from the stack's own
test rig ``stack/tests/_refcv8_rig.py``, CPU).  Run against a scratch tree = the frozen launch tree + the diff applied:

    L7_TREE=<scratch>/tree python tests/integ_refcv8_l7.py     # PYTHONPATH must NOT contain another stack

Checks (expectations are literals or test-own boolean logic, never the code under test):
  I1  no scorer attached: every emitted key is bit-identical to a second unpatched-path forward (the block is skipped)
  I2  scorer attached, select_live=False: the emitted plan keys are bit-identical to I1, and sel_idx_l7 is present
  I3  a CONSTANT scorer (all head weights and biases 0), select_live=True: sel_idx == the lowest index allowed by
      reach_keep AND ceil_keep AND r8_emit_keep (the test's own boolean scan), and traj == anchor_traj[sel_idx]
  I4  mode S attached: the scene sources reach it (r7_*_tokens captured although refcv7_wta is None) and its pick is in range
  I5  deliberate regression: a scorer fed a forbidden key through the integration's assemble path REFUSES
"""
from __future__ import annotations

import os
import sys
import tempfile

TREE = os.environ["L7_TREE"]
sys.path.insert(0, os.path.join(TREE, "stack", "tests"))
sys.path.insert(0, os.path.join(TREE, "stack"))
import torch  # noqa: E402

import _refcv8_rig as R  # noqa: E402
from tanitad.refs import refc_scorer as RS  # noqa: E402

KEYS = ("traj", "sel_idx", "anchor_traj", "sel_score", "sel_score_v3", "sel_idx_base")


def save(model: RS.L7Scorer, d: str, name: str) -> str:
    p = os.path.join(d, name)
    torch.save({"cfg": vars(model.cfg), "state_dict": model.state_dict(), "epoch": 0}, p)
    return p


def main() -> int:
    import tanitad
    assert os.path.normcase(os.path.abspath(tanitad.__file__)).startswith(os.path.normcase(os.path.abspath(TREE))), \
        f"tanitad imported from {tanitad.__file__}, not the patched tree"
    fails = []
    T = R.trainer()
    cfg, m = R.build(T, True, n_alloc=4, vmax=False)
    bt = R.batch(cfg)
    o_a = R.forward(cfg, m, bt)
    o_b = R.forward(cfg, m, bt)
    if not all(torch.equal(o_a[k], o_b[k]) for k in KEYS):
        fails.append("I1: two scorer-free forwards differ (rig non-determinism; nothing else is readable)")
    if "sel_idx_l7" in o_a:
        fails.append("I1: sel_idx_l7 emitted with no scorer attached")
    tmp = tempfile.mkdtemp(prefix="l7integ_")
    torch.manual_seed(0)
    pf = save(RS.L7Scorer(RS.L7Config(mode="F")), tmp, "f.pt")
    RS.attach(m, pf, select_live=False)
    o2 = R.forward(cfg, m, bt)
    bad = [k for k in KEYS if not torch.equal(o_a[k], o2[k])]
    if bad:
        fails.append(f"I2: shadow scoring changed the emitted keys {bad}")
    if "sel_idx_l7" not in o2:
        fails.append("I2: sel_idx_l7 missing with a scorer attached")
    # I3 -- constant scorer
    c = RS.L7Scorer(RS.L7Config(mode="F"))
    with torch.no_grad():
        for h in c.heads.values():
            h.weight.zero_()
            h.bias.zero_()
    pc = save(c, tmp, "const.pt")
    RS.attach(m, pc, select_live=True)
    o3 = R.forward(cfg, m, bt)
    B, N = o3["anchor_traj"].shape[:2]
    allow = torch.ones(B, N, dtype=torch.bool)
    for k in ("reach_keep", "ceil_keep", "r8_emit_keep"):
        if o3.get(k) is not None:
            allow &= o3[k].to(torch.bool)
    want = []
    for i in range(B):
        idx = [j for j in range(N) if bool(allow[i, j])]
        want.append(idx[0] if idx else 0)
    got = o3["sel_idx"].tolist()
    if got != want:
        fails.append(f"I3: constant scorer emitted {got}, lowest allowed index is {want}")
    if not torch.equal(o3["traj"], o3["anchor_traj"][torch.arange(B), o3["sel_idx"]]):
        fails.append("I3: traj is not anchor_traj[sel_idx]")
    # I4 -- mode S
    dims = {}
    if o3.get("r7_bev_tokens") is not None:
        dims["bev"] = int(o3["r7_bev_tokens"].shape[-1])
    if o3.get("r7_agent_tokens") is not None:
        dims["agent"] = int(o3["r7_agent_tokens"].shape[-1])
    if not dims:
        fails.append("I4: no r7_*_tokens captured with an L7 scorer attached (the hook condition did not widen)")
    else:
        ps = save(RS.L7Scorer(RS.L7Config(mode="S", source_dims=dims)), tmp, "s.pt")
        RS.attach(m, ps, select_live=True)
        o4 = R.forward(cfg, m, bt)
        s4 = o4["sel_idx"]
        if not (bool((s4 >= 0).all()) and bool((s4 < N).all())):
            fails.append(f"I4: mode S pick out of range {s4.tolist()}")
    # I5 -- the guard on the integration's input path
    try:
        RS.assemble_inputs({"anchor_traj": o3["anchor_traj"], "v0": bt["v0"], "v_max_ms": bt["v0"]}, "E")
        fails.append("I5: v_max_ms accepted")
    except RS.InadmissibleInput:
        pass
    print("ZZL7INTEG " + ("PASS" if not fails else "FAIL " + " | ".join(fails)),
          {"B": B, "N": N, "sel_I3": got, "dims_S": dims})
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
