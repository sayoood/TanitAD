"""Read-only diagnosis: WHICH selection-score term carries the aarch64 ulp in
test_refcv8_warm_start::test_allocation_and_prior_free_group_leave_the_emitted_plan_and_the_base_fan_unchanged.
Records every `_apply_grafts(base, terms, ...)` call for the refcv7 and the refcv8 build on the test's own rig and
batch, then compares base and each term on the BASE candidate slice. Run from <tree>/stack/tests with the gate env."""
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path.cwd()))
import _refcv8_rig as R  # noqa: E402
from tanitad.refs import refc as RC  # noqa: E402

rec: list = []
_orig = RC.AnchoredDiffusionDecoder._apply_grafts


def _spy(self, base, terms, state, surface, patience):
    out = _orig(self, base, terms, state, surface, patience)
    rec.append((surface, base.detach().clone(), [t.detach().clone() for t in terms], out[0].detach().clone()))
    return out


RC.AnchoredDiffusionDecoder._apply_grafts = _spy
T = R.trainer()
cfg7, m7 = R.build(T, False)
bt = R.batch(cfg7)
rec.clear()
o7 = R.forward(cfg7, m7, bt)
r7 = list(rec)
cfg8, m8 = R.build(T, True, n_alloc=8, prior_free_group=True)
R.copy_into(m7, m8)
rec.clear()
o8 = R.forward(cfg8, m8, bt)
r8 = list(rec)
nb = int(o8["r8_n_base"])
print("torch", torch.__version__, "threads", torch.get_num_threads(), "nb", nb, "N8", o8["anchor_traj"].shape[1])
print("calls7", [(s, len(t)) for s, _, t, _ in r7], "calls8", [(s, len(t)) for s, _, t, _ in r8])
for k in ("traj", "sel_idx", "anchor_traj", "u0_hat", "sel_score", "sel_score_v3"):
    a, b = o7[k], o8[k]
    if k in ("anchor_traj", "u0_hat", "sel_score", "sel_score_v3"):
        b = b[:, :nb]
    print(f"OUT {k:14s} equal={torch.equal(a, b)} maxabs={(a.float() - b.float()).abs().max().item():.3e}")
for i, ((s7, b7, t7, y7), (s8, b8, t8, y8)) in enumerate(zip(r7, r8)):
    print(f"CALL {i} {s7}/{s8} base equal={torch.equal(b7, b8[:, :nb])} "
          f"maxabs={(b7 - b8[:, :nb]).abs().max().item():.3e} | out equal={torch.equal(y7, y8[:, :nb])}")
    for j in range(max(len(t7), len(t8))):
        if j < len(t7) and j < len(t8):
            print(f"   term {j}: equal={torch.equal(t7[j], t8[j][:, :nb])} "
                  f"maxabs={(t7[j] - t8[j][:, :nb]).abs().max().item():.3e} absmax7={t7[j].abs().max().item():.3e}")
        else:
            x = (t8[j][:, :nb] if j < len(t8) else None)
            print(f"   term {j}: only in {'r8' if j >= len(t7) else 'r7'};"
                  f" base-slice absmax={'-' if x is None else f'{x.abs().max().item():.3e}'}"
                  f" exact-zero={'-' if x is None else bool((x == 0).all())}")
print("ZZULPDONEZZ")
