"""Shared helpers of ralign/tests/ (RALIGN prep agent): paths, synthetic GAP48-shaped token files (written by THIS test code in a tmp dir; no real GAP48 token file is ever opened),
and the CPU scenario that drives the real C0 checkpoint behind the stub encoder.

Run from t2v8f/:  CUDA_VISIBLE_DEVICES=-1 C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I -m pytest ralign/tests -q --basetemp F:/TanitAD-scratch/ralign_pytest
"""
from __future__ import annotations

import hashlib
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
RA = os.path.dirname(HERE)
T2 = os.path.dirname(RA)
CODE = f"{T2}/code"
for _p in (RA, CODE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import numpy as np  # noqa: E402
import torch  # noqa: E402

import ralign_lib as L  # noqa: E402

ART = "E:/Projects/TanitAD-artifacts"
CKPT = {s: f"{ART}/v8f-recover/M2b_s{s}_recft_C0/ckpt_last.pt" for s in (0, 1)}
GAP48_DIR = os.path.realpath(f"{T2}/gap48").replace("\\", "/")


def assert_not_gap48(path):
    """The tests only ever read files they wrote themselves: a path under t2v8f/gap48/ is a bug in the test."""
    assert not os.path.realpath(str(path)).replace("\\", "/").startswith(GAP48_DIR), f"test reads under gap48/: {path}"


def scene_ids(n_a, n_b, digit=0, prefix="sc"):
    """Synthetic 36-character scene ids with exactly n_a in fold A and n_b in fold B (by the card's rule)."""
    out, i = {"A": [], "B": []}, 0
    while len(out["A"]) < n_a or len(out["B"]) < n_b:
        sid = hashlib.sha256(f"{prefix}{i}".encode()).hexdigest()
        sid = f"{sid[:8]}-{sid[8:12]}-{sid[12:16]}-{sid[16:20]}-{sid[20:32]}"
        f = L.fold_of(sid, digit)
        if len(out[f]) < (n_a if f == "A" else n_b):
            out[f].append(sid)
        i += 1
    return out["A"] + out["B"]


def fp8_like(x: torch.Tensor) -> torch.Tensor:
    """fp32 -> fp8 e4m3fn -> fp16 (the exact image of what GAP48 banks)."""
    return x.clamp(-400, 400).to(torch.float8_e4m3fn).to(torch.float16)


def synth_pairs(sids, n_ticks, shape=L.SHAPE, a_true=0.6, b_true=0.3, noise=0.05, seed=0, scene_offset=0.0):
    """{scene id: (render fp16, real fp16)}: real = a_true * render + b_true + noise (per cell), both through the fp8 grid. A render-to-real map that OLS can recover."""
    g = torch.Generator().manual_seed(seed)
    out = {}
    for sid in sids:
        r = torch.randn((n_ticks, *shape), generator=g) * 2.0
        x = a_true * r + b_true + noise * torch.randn((n_ticks, *shape), generator=g) + scene_offset
        out[sid] = (fp8_like(r), fp8_like(x))
    return out


def write_token_dir(pairs: dict, tdir, scene_key="scene", render_key="render", real_key="real"):
    """GAP48 layout: tokens/<scene8>.pt, a torch dict (scene, ticks, cam_frames, render, real, ... as t2v8f_gap48.py:302 builds it)."""
    assert_not_gap48(tdir)
    os.makedirs(tdir, exist_ok=True)
    for sid, (r, x) in pairs.items():
        n = r.shape[0]
        torch.save({scene_key: sid, "ticks": torch.arange(n) * 10, "cam_frames": torch.arange(n) * 10, render_key: r, real_key: x, "token_dtype": "float16 (exact image of the fp8 e4m3fn tokens)"},
                   f"{tdir}/{sid[:8]}.pt")
    return tdir


# ------------------------------------------------------------------------------------------------------------------------------ the CPU scenario (real C0 checkpoint, stub encoder)
def build_model_policy(seed=0, arm="model", composite=True):
    import t2a_v8f_encoder as E
    import t2a_v8f_policy as P
    import t2v8f_c0c as C
    p = P.V8FMiniT2Policy(CKPT[seed], encoder=E.StubEncoder(), arm=arm, schedule="periodic", device="cpu")
    return C.CompositePolicy(p) if (composite and arm == "model") else p


def base_of(pol):
    import ralign_policy as RP
    return RP.base_policy(pol)


def run_scenario(pol, ticks=3, capture=True, hook=None):
    """Seeds a straight 10 m/s track, then plans `ticks` times on the 0.2 s grid, advancing the track by two rows between plans. -> list of (traj, extra, batch dict or None).
    `hook(pol, tick)` runs before each plan (the mutants use it)."""
    rng = np.random.default_rng(0)
    frames_all = [rng.integers(0, 255, (8, 8, 3), dtype=np.uint8) for _ in range(60 + 2 * (ticks - 1))]
    rows = np.stack([[float(i), 0.0, 0.0, 10.0] for i in range(61)])
    nav = np.tile(np.array([1, 0, 0, 0, 0], np.float32), (31, 1))
    pol.seed_track(rows, nav_rows=nav)
    pol.set_nav(row=nav[0])
    b = base_of(pol)
    caught = []
    orig = b.forward_batch
    if capture:
        def rec(batch):
            caught.append({k: v.clone() for k, v in batch.items()})
            return orig(batch)
        b.forward_batch = rec
    out = []
    for t in range(ticks):
        if t:
            nr = len(b.track)
            new = np.stack([[float(nr + j), 0.0, 0.0, 10.0] for j in range(2)])
            h = np.vstack([b.track, new])[-16:]
            pol.set_history(h, n_new=2)
        if hook is not None:
            hook(pol, t)
        traj, extra = pol.plan(frames_all[:60 + 2 * t], None, 10.0, None)
        out.append((traj, extra, caught[-1] if capture and caught else None))
    if capture:
        b.forward_batch = orig
    return out


def strip(extra, keys=("ralign",)):
    return {k: v for k, v in extra.items() if k not in keys}


def bit_equal_runs(a, b):
    """True iff every tick's plan is byte-identical and every `extra` (minus the wrapper's key) is equal."""
    return len(a) == len(b) and all(x[0].tobytes() == y[0].tobytes() and strip(x[1]) == strip(y[1]) for x, y in zip(a, b))


def rand_map(seed=1, scale=0.3, shape=L.SHAPE):
    g = torch.Generator().manual_seed(seed)
    return L.TokenMap(1.0 + scale * torch.randn(shape, generator=g), 0.2 * torch.randn(shape, generator=g), {"fold": "B", "fold_scenes": []})


def identity_map():
    a, b = L.identity_pair()
    return L.TokenMap(a, b, {"fold": "B", "fold_scenes": []})
