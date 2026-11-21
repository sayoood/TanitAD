"""MM 2026-11-21: reproduce the pod's weight-rule crash on the dev box (CPU) and show the fix.
A compiled module + three torch.autograd.grad calls with retain_graph=True on the same graph (what weight_rule_rank.measure does).
  mode red  : donated_buffer default (True) -> expect the RuntimeError 'compiled with non-empty donated buffers'
  mode green: torch._functorch.config.donated_buffer = False before the first compiled forward -> no error, and the three gradients equal
              the EAGER model's gradients (same weights, same input)."""
import sys
import torch
import torch._functorch.config as fc
mode, backend = sys.argv[1], sys.argv[2]
if mode == "green":
    fc.donated_buffer = False
torch.manual_seed(0)
class M(torch.nn.Module):
    def __init__(s):
        super().__init__()
        s.trunk = torch.nn.Sequential(torch.nn.Linear(16, 64), torch.nn.GELU(), torch.nn.Linear(64, 64), torch.nn.GELU())
        s.head_a = torch.nn.Linear(64, 8)
        s.head_b = torch.nn.Linear(64, 8)
    def forward(s, x):
        h = s.trunk(x)
        return s.head_a(h), s.head_b(h)
eager = M()
comp = M(); comp.load_state_dict(eager.state_dict())
comp.trunk = torch.compile(comp.trunk, backend=backend)       # the trainer compiles the trunk / backbone in place
x = torch.randn(32, 16)
def grads(m):
    a, b = m(x)
    l1, l2, l3 = (a ** 2).mean(), (b.sigmoid()).mean(), (a * b).mean()
    ps = list(m.parameters())
    g1 = torch.autograd.grad(l1, ps, retain_graph=True, allow_unused=True)
    g2 = torch.autograd.grad(l2, ps, retain_graph=True, allow_unused=True)
    g3 = torch.autograd.grad(l3, ps, retain_graph=True, allow_unused=True)
    return [torch.cat([(g if g is not None else torch.zeros_like(p)).flatten() for g, p in zip(gs, ps)]) for gs in (g1, g2, g3)]
ge = grads(eager)
try:
    gc = grads(comp)
except RuntimeError as e:
    print("ZZREPRO_ERROR", mode, backend, str(e)[:110]); sys.exit(3 if mode == "red" else 1)
d = max(float((a - b).abs().max()) for a, b in zip(ge, gc))
print("ZZREPRO_OK", mode, backend, "torch", torch.__version__, "max |grad_compiled - grad_eager|", d)
sys.exit(0 if mode == "green" else 2)
