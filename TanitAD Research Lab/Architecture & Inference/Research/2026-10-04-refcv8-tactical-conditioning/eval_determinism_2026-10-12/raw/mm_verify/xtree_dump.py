"""Master Mind's own cross-tree check of the B4 fix: run ONE tree through train -> eval x3 -> train x2 on the refcv8 rig
and dump every output / grad / loss / r8 generator state. Run once per tree (d014ff2-based and fix), then compare.
usage: python xtree_dump.py <tree> <out.pt> <mode>   mode: off | on | sched (fresh build, apply_emit_schedule per step)"""
import sys
from pathlib import Path

tree, out_path, mode = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
sys.path[:0] = [str(tree / "stack"), str(tree / "stack" / "tests"), str(tree / "stack" / "scripts")]
import torch  # noqa: E402

import _refcv8_rig as R  # noqa: E402
import tanitad  # noqa: E402
from tanitad.train import refcv8_train as RT  # noqa: E402

assert str(tree) in tanitad.__file__, tanitad.__file__
A2 = dict(n_alloc=32, alloc_top_k=4, alloc_min_each=2, alloc_emit=True, emit_start=1 if mode == "sched" else 1000,
          lat_prior_dropout=0.3, cond_dropout=0.15, w_cons=0.05, w_sat=0.1, w_alloc_l1=1.0, w_listwise=1.0,
          w_subscore=0.5, speed_input="label", v9_cons=True, w_v9_cons=0.05)
VMAX = (13.8889, 27.7778, 33.3333)
T = R.trainer()
cfg, m = R.build(T, True, seed=0, vmax=True, **A2)
dec = m.core.decoder
born = bool(dec.r8_emit_live)
if mode in ("on", "off"):
    dec.r8_emit_live = mode == "on"
bt = R.batch(cfg, b=3)


def flat(o, p="out"):
    r = {}
    if torch.is_tensor(o):
        r[p] = o.detach().clone()
    elif isinstance(o, dict):
        for k in o:
            r.update(flat(o[k], f"{p}.{k}"))
    elif isinstance(o, (list, tuple)):
        for i, v in enumerate(o):
            r.update(flat(v, f"{p}[{i}]"))
    return r


def fwd(seed, train):
    m.train(train)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        ph = bt["pose_hist"]
        b = int(ph.shape[0])
        with (torch.enable_grad() if train else torch.no_grad()):
            m.core.set_ego_window(ph, int(ph.shape[1]))
            return m(bt["frames"], nav_cmd=bt["nav_cmd"], v0=bt["v0"], steps=cfg.core.decoder.diffusion_steps,
                     v_max_ms=torch.tensor(VMAX[:b]), v_max_valid=torch.ones(b))


res = {"born_emit_live": born, "phases": []}


def train_step(tag, step):
    if mode == "sched":
        RT.apply_emit_schedule(m, step)
    m.zero_grad(set_to_none=True)
    out = fwd(2 + step, True)
    fl = flat(out)
    live = list(_live(out))
    loss = sum(v.float().mean() for v in live)
    loss.backward()
    res["phases"].append({"tag": tag, "emit_live": bool(dec.r8_emit_live), "out": fl, "loss": loss.detach().clone(),
                          "grads": {n: p.grad.detach().clone() for n, p in m.named_parameters() if p.grad is not None},
                          "gen": dec.r8_gen.get("cpu").get_state().clone(), "global": torch.get_rng_state().clone()})


def _live(o):
    if torch.is_tensor(o):
        if o.requires_grad and o.is_floating_point():
            yield o
    elif isinstance(o, dict):
        for k in o:
            yield from _live(o[k])
    elif isinstance(o, (list, tuple)):
        for v in o:
            yield from _live(v)


torch.manual_seed(0)
train_step("train0", 0)
for i, s in enumerate((1234, 777, 1234)):
    res["phases"].append({"tag": f"eval{i}_s{s}", "out": flat(fwd(s, False)),
                          "gen": dec.r8_gen.get("cpu").get_state().clone()})
train_step("train1", 1)
train_step("train2", 2)
torch.save(res, out_path)
print("ZZDUMP-OK", mode, "born", born, "phases", len(res["phases"]))
