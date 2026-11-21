"""Read-only: first point of divergence inside the decoder's BASE pass, refcv7 vs refcv8 (test rig)."""
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path.cwd()))
import _refcv8_rig as R  # noqa: E402

def spy(m):
    dec = m.core.decoder
    log = []
    def mk(name):
        def h(mod, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            i0 = next((t for t in inp if torch.is_tensor(t)), None)
            log.append((name, tuple(o.shape), o.detach().clone(),
                        None if i0 is None else (i0.is_contiguous(), tuple(i0.stride()))))
        return h
    dec.traj_proj.register_forward_hook(mk("traj_proj"))
    for i, L in enumerate(dec.layers):
        L.register_forward_hook(mk(f"layer{i}"))
    dec.conf_head.register_forward_hook(mk("conf_head"))
    if getattr(dec, "r8_mod", None) is not None:
        dec.r8_mod.register_forward_hook(mk("r8_mod"))
    return log

T = R.trainer()
cfg7, m7 = R.build(T, False); bt = R.batch(cfg7)
cfg8, m8 = R.build(T, True, n_alloc=8, prior_free_group=True); R.copy_into(m7, m8)
l7, l8 = spy(m7), spy(m8)
R.forward(cfg7, m7, bt); o8 = R.forward(cfg8, m8, bt)
nb = int(o8["r8_n_base"])
b8 = [e for e in l8 if e[1][1] == nb and e[0] != "r8_mod"]
m8mods = [e for e in l8 if e[0] == "r8_mod" and e[1][1] == nb]
print("calls7", len(l7), "calls8 base", len(b8), "r8_mod base calls", len(m8mods))
first = None
for k, (a, b) in enumerate(zip(l7, b8)):
    eq = a[0] == b[0] and torch.equal(a[2], b[2])
    if k < 12 or not eq:
        print(k, a[0], b[0], a[1], b[1], "equal" if eq else f"DIFF {(a[2]-b[2]).abs().max().item():.3e}", "in7", a[3], "in8", b[3])
    if not eq and first is None:
        first = k
    if first is not None and k > first + 3:
        break
for j, e in enumerate(m8mods[:4]):
    print("r8_mod out==in? (cannot see input here) shape", e[1])
print("ZZLAYERSDONEZZ")
