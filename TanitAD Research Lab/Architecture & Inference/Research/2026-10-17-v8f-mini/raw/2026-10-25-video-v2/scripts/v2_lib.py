"""v2_lib.py -- model, eval-world, nav-injection chain and frame access for the v8F-mini evidence video v2 (CPU only, read-only on every input).

Model code: the tip 9a6551d5 export under OUT/code (never the stale editable `tanitad` on G:).
Chain: the model's OWN run_s / run_t / run_o, exactly as `V8FMini._forward` calls them with use_labels = False (no re-implementation of the cond
assembly).  A nav command replaces `b['nav']` (all history positions) for the layer(s) named; every other input is the window's own.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

SC = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad"
OUT = f"{SC}/viz2"
V1 = f"{SC}/viz"
WORK = f"{V1}/work"                      # v1 inputs (read-only): ckpt/, inputs/, cache_viz/ (the 4 clips), banked/
CODE = f"{OUT}/code"
SFRESH_CACHE = f"{SC}/sfresh/cache"      # the full 141-clip eval cache (read-only)
REBUILD = "C:/Users/Admin/tanitad-data/refav1-eval141-256rb"
for _p in (f"{CODE}/taniteval", f"{CODE}/stack"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

torch.set_num_threads(int(os.environ.get("V2_THREADS", "6")))
from tanitad.data import v8f_mini_data as D  # noqa: E402
from tanitad.models import v8f_mini as M  # noqa: E402

if not os.path.abspath(M.__file__).replace("\\", "/").startswith(CODE):
    raise ImportError(f"tanitad imported from {M.__file__}, not from the tip export {CODE}")

SEL = json.load(open(f"{V1}/out/selection.json"))
KEYS = ["JUNCTION", "CURVE", "NO-TURN", "FAILURE(worst)"]
WINDOWS = {k: SEL[k] for k in KEYS}
SEEDS = ("M2b_s0", "M2b_s1")
CKPT = {s: f"{WORK}/ckpt/{s}.pt" for s in SEEDS}
INPUTS = f"{WORK}/inputs"
LAT5 = ("LANE_KEEP", "TURN_L", "TURN_R", "LANE_CHG_L", "LANE_CHG_R")
LON7 = ("HOLD", "CREEP", "STOP", "FOLLOW", "DECEL", "ACCEL", "KEEP")
CMDS = ("FOLLOW", "TURN_L", "TURN_R")
D_NEXT_M = 30.0
CMD_VEC = {"FOLLOW": [1.0, 0.0, 0.0, D_NEXT_M / 100.0, 1.0], "TURN_L": [0.0, 1.0, 0.0, D_NEXT_M / 100.0, 1.0],
           "TURN_R": [0.0, 0.0, 1.0, D_NEXT_M / 100.0, 1.0]}
CONFIGS = {"a_all": ("S", "T", "O"), "b_O": ("O",), "c_T": ("T",), "d_S": ("S",)}


def load_model(name: str):
    ck = torch.load(CKPT[name], map_location="cpu", weights_only=False)
    cfg = M.V8FMiniConfig(**ck["cfg"])
    model = M.V8FMini(cfg, M.P10Prior(torch.zeros(8), torch.ones(8), torch.zeros(2 * M.K30)))
    model.load_state_dict(ck["model"])
    return model.eval(), cfg


class EvalWorld:
    """The eval's own loader (v8f_mini_eval.cmd_run): same grid, join, X, ClipStore and window dataset; window i of the 2,399 is
    (ci, t) with `--ages cycle`: a_t = i % 3, a_s = a_t + (i // 3) % 5 (then clamped by `item`)."""

    def __init__(self, cfg, lru: int = 4, cache: str | None = None):
        cache = cache or f"{WORK}/cache_viz"
        self.grid = json.load(open(f"{INPUTS}/eval_grid_windows.json"))
        self.J = D.load_join(f"{INPUTS}/join_eval141.npz")
        self.X = D.eval_context(self.grid, f"{INPUTS}/eval141_kin.npz", self.J, kin_mode=cfg.kin_mode)
        self.tracks = D.load_native_tracks(f"{INPUTS}/native_tracks.pt")
        self.store = D.ClipStore(cache, "eval", lru=lru, grid="pool4x10", kin_mode=cfg.kin_mode, tracks=self.tracks)
        self.wins = D.eval_grid_windows(f"{INPUTS}/eval_grid_windows.json", self.J)
        self.ds = D.V8FMiniWindows(self.store, self.J, self.wins, hist=cfg.hist, hmax=cfg.hmax, max_age_t=cfg.max_age_t,
                                   max_age_s=cfg.max_age_s, window_start=cfg.window_start)
        self.clips = list(self.J["clip"].tolist())
        self.N = len(self.wins)

    def ages(self, i: int):
        a_t = i % 3
        return a_t, a_t + (i // 3) % 5

    def item(self, i: int, fresh: bool = False):
        ci, t = self.wins[i][0], self.wins[i][1]
        a_t, a_s = (0, 0) if fresh else self.ages(i)
        return self.ds.item(ci, t, a_t, a_s)

    def batch(self, idx, fresh: bool = False):
        return D.collate([self.item(int(i), fresh) for i in idx])


# ------------------------------------------------------------------------------------------------ the chain with a nav command injected
def cmd_tensor(b: dict, cmd) -> torch.Tensor:
    """[B, F, 5] constant command over the whole history; `cmd` is a name or a 5-vector."""
    v = torch.as_tensor(CMD_VEC[cmd] if isinstance(cmd, str) else cmd, dtype=b["nav"].dtype)
    return v[None, None].expand_as(b["nav"]).clone()


@torch.no_grad()
def chain(model, b: dict, nav=None, where=()):
    """S -> T -> O with `nav` ([B, F, 5] tensor) replacing the nav input of the layers named in `where`; None / () = the window's own
    (exactly `forward_infer`)."""
    B, Fn = b["feats"].shape[0], b["feats"].shape[1]
    adapt = model._encode(b)
    last_now = torch.full((B,), Fn - 1, dtype=torch.long)
    last_t, last_s = last_now - b["a_t"], last_now - b["a_s"]

    def bb(layer):
        if nav is not None and layer in where:
            d = dict(b)
            d["nav"] = nav
            return d
        return b

    ck_s = model.run_s(bb("S"), adapt, last_s)
    ck_ok = torch.ones(B, 3)
    tout = model.run_t(bb("T"), adapt, last_t, M.transform_ckpt(ck_s, b["pose_s"], b["pose_t"]), ck_ok)
    code = model._code_pred(tout)
    ck_now = M.transform_ckpt(ck_s, b["pose_s"], b["pose_now"])
    traj60, p30 = model.run_o(bb("O"), adapt, last_now, ck_now, ck_ok, code)
    return {"ckpt_s": ck_s, "ckpt_now": ck_now, "lat_logits": tout["lat_logits"], "lon_logits": tout["lon_logits"], "goal_prob": tout["goal_prob"],
            "traj60": traj60, "traj30": M.subsample_02(traj60), "p10": p30, "code": code}


def own_forward(model, b: dict) -> dict:
    with torch.no_grad():
        return model.forward_infer(b)


# ------------------------------------------------------------------------------------------------ frames (v1's access, unchanged)
_FR: dict = {}


def v2ep_frame(clip: str, j: int) -> np.ndarray:
    """cache step j = native frame 2j of the v2ep episode (v1 `render_video.load_clip`); [256, 640, 3] uint8."""
    import torchvision
    if clip not in _FR:
        o = torch.load(f"{REBUILD}/{clip}.v2ep.pt", map_location="cpu", weights_only=False)
        buf, lens = o["jpeg_buf"], o["jpeg_len"]
        offs = torch.cat([torch.zeros(1, dtype=lens.dtype), lens.cumsum(0)])
        _FR[clip] = (buf, offs)
    buf, offs = _FR[clip]
    return torchvision.io.decode_png(buf[offs[2 * j]:offs[2 * j + 1]].clone()).permute(1, 2, 0).numpy()
