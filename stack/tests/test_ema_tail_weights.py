"""ema_tail_weights.py (E2-TAIL): the tail average out of two WeightEMA snapshots. CPU, tiny tensors, no scipy.

* IDENTITY: the REAL ``WeightEMA`` runs N steps on a toy model whose raw weights at every step are known; the shadow is saved at
  steps a and b exactly as ``refc_v3_train._save_ema_model`` saves it; ``W_tail`` must equal
  ``sum_{s=a+1..b} d^(b-s) (1 - d) theta_s / (1 - d^n)`` to 1e-6.
* A frozen tensor and an int buffer are copied bit for bit; every refusal fires; the output has the input's structure.
* MUTATION: the source with ``/ (1 - d^n)`` dropped, or the ``d^n *`` term dropped, is rebuilt in memory and the identity check
  must go RED on it (the check is the same function the real test uses).
"""
from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest
import torch

_STACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_STACK))
sys.path.insert(0, str(_STACK / "scripts"))
import ema_tail_weights as T                                   # noqa: E402
from tanitad.train.weight_ema import WeightEMA                 # noqa: E402

D, A_STEP, B_STEP = 0.9, 5, 15


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(3, 4)
        self.bn = torch.nn.BatchNorm1d(4)                      # float buffers + an int64 num_batches_tracked
        self.frozen = torch.nn.Linear(2, 2)
        for p in self.frozen.parameters():
            p.requires_grad_(False)


def _payload(ema, model, step):
    """refc_v3_train._save_ema_model's dict (refc_v3_train.py:5707-5708)."""
    return {"model": {k: v.clone() for k, v in ema.model_state_dict(model).items()}, "step": int(step), "ema_decay": ema.decay,
            "num_updates": int(ema.num_updates), "kind": "ema_model_only"}


@pytest.fixture(scope="module")
def run():
    g = torch.Generator().manual_seed(7)
    m = Toy()
    ema = WeightEMA(m, D)
    thetas, snaps = {}, {}
    for s in range(1, B_STEP + 1):
        with torch.no_grad():
            for n, t in m.state_dict(keep_vars=True).items():
                if n.startswith("frozen") or not t.is_floating_point():
                    continue
                t.copy_(torch.randn(t.shape, generator=g) * 2.0)
        ema.update(m)
        thetas[s] = {n: t.detach().clone() for n, t in m.state_dict(keep_vars=True).items()}
        if s in (A_STEP, B_STEP):
            snaps[s] = _payload(ema, m, s)
    return m, ema, thetas, snaps


def _exact_tail(thetas, name, d=D, a=A_STEP, b=B_STEP):
    n = b - a
    acc = torch.zeros_like(thetas[b][name], dtype=torch.float64)
    for s in range(a + 1, b + 1):
        acc += (d ** (b - s)) * (1 - d) * thetas[s][name].double()
    return acc / (1 - d ** n)


def _identity_gap(tail_fn, run) -> float:
    """max |W_tail - exact weighted mean| over every shadowed tensor, for a tail implementation ``tail_fn(ck_a, ck_b)``."""
    m, ema, thetas, snaps = run
    out = tail_fn(snaps[A_STEP], snaps[B_STEP])
    gap = 0.0
    for name in ema.shadow:
        gap = max(gap, float((out[name].double() - _exact_tail(thetas, name)).abs().max()))
    return gap


def _real_tail(ck_a, ck_b):
    return T.tail_state(ck_a, ck_b)[0]


def test_identity_against_the_exact_weighted_mean(run):
    m, ema, thetas, snaps = run
    assert len(ema.shadow) >= 6                                # weight, bias, bn weight/bias, running mean/var
    assert snaps[A_STEP]["num_updates"] == A_STEP and snaps[B_STEP]["num_updates"] == B_STEP
    gap = _identity_gap(_real_tail, run)
    assert gap < 1e-6, gap


def _mutant(old: str, new: str):
    src = (Path(T.__file__)).read_text(encoding="utf-8")
    assert src.count(old) == 1, f"the mutation target {old!r} must occur exactly once"
    mod = types.ModuleType("ema_tail_mutant")
    exec(compile(src.replace(old, new), "ema_tail_mutant", "exec"), mod.__dict__)
    return lambda a, b: mod.tail_state(a, b)[0]


@pytest.mark.parametrize("old,new", [
    ("(tb.double() - dn * ta.double()) / denom", "(tb.double() - dn * ta.double())"),             # drop / (1 - d^n)
    ("(tb.double() - dn * ta.double()) / denom", "tb.double() / denom"),                           # drop the d^n * term
])
def test_mutants_go_red(run, old, new):
    assert _identity_gap(_mutant(old, new), run) > 1e-3        # the identity bar (1e-6) is far below what a mutant shows


def test_frozen_and_int_tensors_are_copied_bit_for_bit(run):
    m, ema, thetas, snaps = run
    out, info = T.tail_state(snaps[A_STEP], snaps[B_STEP])
    for k in ("frozen.weight", "frozen.bias", "bn.num_batches_tracked"):
        assert k not in ema.shadow or k.startswith("bn")
        assert out[k].dtype == snaps[B_STEP]["model"][k].dtype
        assert torch.equal(out[k], snaps[B_STEP]["model"][k])
        assert out[k].data_ptr() != snaps[B_STEP]["model"][k].data_ptr()      # a copy, not an alias
    assert info["n_copied"] == 3 and info["n"] == B_STEP - A_STEP
    assert all(out[k].dtype == snaps[B_STEP]["model"][k].dtype for k in out)


def test_output_has_the_structure_of_the_input(run):
    m, ema, thetas, snaps = run
    out, _ = T.tail_weights(snaps[A_STEP], snaps[B_STEP])
    assert list(out) == list(snaps[B_STEP]) and out["kind"] == "ema_model_only"
    assert list(out["model"]) == list(snaps[B_STEP]["model"])
    assert out["step"] == B_STEP and out["num_updates"] == B_STEP and out["ema_decay"] == D


def _fresh(run):
    a = {k: (v if k != "model" else {n: t.clone() for n, t in v.items()}) for k, v in run[3][A_STEP].items()}
    b = {k: (v if k != "model" else {n: t.clone() for n, t in v.items()}) for k, v in run[3][B_STEP].items()}
    return a, b


def test_refusals(run):
    def refuses(msg, mut):
        a, b = _fresh(run)
        mut(a, b)
        with pytest.raises(T.TailError, match=msg):
            T.tail_state(a, b)

    refuses("decays differ", lambda a, b: a.__setitem__("ema_decay", 0.95))
    refuses("must be > 0", lambda a, b: a.__setitem__("num_updates", B_STEP))                     # n == 0
    refuses("must be > 0", lambda a, b: a.__setitem__("num_updates", B_STEP + 3))                 # n < 0
    refuses("no 'num_updates'", lambda a, b: b.pop("num_updates"))
    refuses("no 'num_updates'", lambda a, b: a.pop("num_updates"))
    refuses("not an int", lambda a, b: b.__setitem__("num_updates", 15.0))
    refuses("kind", lambda a, b: b.__setitem__("kind", "full_checkpoint"))
    refuses("tensor sets differ", lambda a, b: b["model"].pop("lin.bias"))
    refuses("shape", lambda a, b: b["model"].__setitem__("lin.bias", torch.zeros(5)))
    refuses("dtype", lambda a, b: b["model"].__setitem__("lin.bias", b["model"]["lin.bias"].double()))
    refuses("non-floating tensor", lambda a, b: b["model"].__setitem__("bn.num_batches_tracked", torch.tensor(99)))
    refuses("non-finite", lambda a, b: b["model"].__setitem__("lin.bias", torch.full((4,), float("nan"))))
    refuses("non-finite", lambda a, b: b["model"].__setitem__("lin.bias", torch.full((4,), 3.0e38)))   # fp32 overflow after the cast
    refuses("non-finite", lambda a, b: (a["model"].__setitem__("frozen.bias", torch.full((2,), float("inf"))),
                                        b["model"].__setitem__("frozen.bias", torch.full((2,), float("inf")))))   # a copied inf


def test_cli_writes_a_loadable_file_and_refuses_overwrite(run, tmp_path):
    pa, pb, po = tmp_path / "a.pt", tmp_path / "b.pt", tmp_path / "out.pt"
    torch.save(run[3][A_STEP], pa)
    torch.save(run[3][B_STEP], pb)
    assert T.main(["--ema-a", str(pa), "--ema-b", str(pb), "--out", str(po)]) == 0
    ck = torch.load(po, map_location="cpu", weights_only=False)
    assert sorted(ck) == sorted(run[3][B_STEP]) and ck["kind"] == "ema_model_only"
    prov = json.loads(Path(str(po) + ".provenance.json").read_text(encoding="utf-8"))
    assert prov["n"] == 10 and prov["top_level_keys"] == sorted(ck)
    assert T.main(["--ema-a", str(pa), "--ema-b", str(pb), "--out", str(po)]) == 2        # exists
    assert T.main(["--ema-a", str(pb), "--ema-b", str(pa), "--out", str(tmp_path / "x.pt")]) == 2   # reversed: n < 0
    assert T.main(["--ema-a", str(pa), "--ema-b", str(pb), "--out", str(pb)]) == 2        # out == input
    assert not (tmp_path / "x.pt").exists() and not list(tmp_path.glob("*.tmp"))


def test_works_with_scipy_blocked(run, monkeypatch):
    """The Thor venv has no scipy: a None entry makes the import raise, and the tool must still import and run."""
    import importlib
    monkeypatch.setitem(sys.modules, "scipy", None)
    mod = importlib.reload(T)
    out, _ = mod.tail_state(*_fresh(run))
    assert len(out) == len(run[3][B_STEP]["model"])
    assert _identity_gap(lambda a, b: mod.tail_state(a, b)[0], run) < 1e-6
