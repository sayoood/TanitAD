"""Dev-box repro helpers for the refcv8 eval-determinism defect (CPU only)."""
import copy, json, sys
TREE = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].endswith("tree") else None
import os
TREE = os.environ.get("R8DET_TREE", r"C:/Users/Admin/r8det/tree")
for p in (TREE + "/stack/tests", TREE + "/stack/scripts", TREE + "/stack"):
    if p not in sys.path:
        sys.path.insert(0, p)
import torch
import _refcv8_rig as R

T = R.trainer()
PROV = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "argv_PROVISIONAL.json")))["argv"]


def r8_from_prov():
    """cfg.refcv8 as the TRAINER's own `_pin_refcv8` sets it from the provisional A2 argv."""
    args = T.build_parser().parse_args(PROV)
    cfg = T.v3.refc_v3_smoke_config(True)
    T._pin_refcv8(cfg, args)
    return cfg.refcv8.to_dict()


def build(kind, seed=0):
    if kind == "r7":
        return R.build(T, False, seed=seed, vmax=True)
    kw = r8_from_prov()
    kw.pop("enable", None)
    assert kw.pop("seed") == 20261004          # the pinned r8 seed IS the dataclass default: dropping it is exact
    return R.build(T, True, seed=seed, vmax=True, **kw)   # vmax: the argv carries --max-speed-input-v6


def fwd_forked(cfg, m, bt, seed=1234, train=False):
    """The model forward under a FORKED global RNG + manual_seed, exactly the gate's control discipline."""
    m.train(train)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        ph = bt["pose_hist"]
        with (torch.enable_grad() if train else torch.no_grad()):
            m.core.set_ego_window(ph, int(ph.shape[1]))
            b = int(bt["v0"].shape[0])
            vmax = torch.tensor([13.8889, 27.7778, 33.3333, 8.3333, 13.8889][:b])
            return m(bt["frames"], nav_cmd=bt["nav_cmd"], v0=bt["v0"], steps=cfg.core.decoder.diffusion_steps,
                     v_max_ms=vmax, v_max_valid=torch.ones(b))


def digests(out):
    import importlib
    G = importlib.import_module("launch_gate")
    return G.output_digests(out)


def differ(a, b):
    return sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
