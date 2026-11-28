# -*- coding: utf-8 -*-
"""WP-D step 1 -- bank the THREE trunks' feature maps on ONE decode pass.

For every labelled eval row we store, at REF-C's real 8x20 stride-32 grid:

  tok_D0.npy  [N, 160, 704] f16   D0's ResNet map   (aux OFF, the control)
  tok_D1.npy  [N, 160, 704] f16   D1's ResNet map   (aux ON,  THE LEVER)
  tok_D2.npy  [N, 160, 704] f16   D2's ResNet map   (shuffled target)
  pix.npy     [N, 160,   9] f16   per-cell mean of the RAW input -> the A2 floor
  y_cart.npy  [N, 7680]     u8    WP-A's Cartesian GRID_DEFAULT raster (PRIMARY,
                                  so section 5A's reference values are comparable)
  y_pol.npy   [N, 480]      u8    the POLAR 24x20 target the aux head trained on
  m_pol.npy   [N, 480]      u8    its occlusion IGNORE mask (occlusion="mask")

ONE decode, THREE encodes: the trunks differ only in weights, so decoding the
episode once and running all three on the SAME tensor removes decode cost AND
removes any chance that the arms saw different pixels.

CONTENT ASSERTIONS, never exit codes -- a memmap a failed decode left as zeros
looks exactly like a finished bank (CLAUDE.md, the E-DETECT-1 all-zero floor).
"""
import argparse
import hashlib
import json
import lzma
import os
import sys
import time

import numpy as np
import torch

SNAP = r"C:\Users\Admin\wpd-probe\snap"
sys.path.insert(0, SNAP)

from tanitad.data._contract import to_float_frames
from tanitad.data.v2_dataset import decode_full_episode
from tanitad.data.bev_raster import GRID_DEFAULT, agents_to_array, rasterize
from tanitad.data import bev_aux as BA
from tanitad.refs.refc import CNNEncoderConfig, ResNetEncoder

ap = argparse.ArgumentParser()
ap.add_argument("--trunks", default=r"C:\Users\Admin\wpd-probe\trunks")
ap.add_argument("--out", default=r"C:\Users\Admin\wpd-probe\bank")
ap.add_argument("--stride", type=int, default=2)   # == WP-A's, so plans align
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--arms", default="D0,D1,D2")
a = ap.parse_args()

EPS = r"C:\Users\Admin\tanitad-data\refav1-eval141\eps"
JOIN = r"C:\Users\Admin\tanitad-caches\b1-agent-join-20260906\b1eval_agents.jsonl.xz"
WPA_IDX = r"C:\Users\Admin\wpa-readout\bank\k8r1\idx.json"
WPA_Y = r"C:\Users\Admin\wpa-readout\bank\k8r1\y.npy"
ARMS = a.arms.split(",")
os.makedirs(a.out, exist_ok=True)
print(f"[env] tanitad from {sys.modules['tanitad'].__file__}", flush=True)

# ---- 1. join -> {clip: {row: agents}} -------------------------------------
t0 = time.time()
per_clip, labelled, n_lines = {}, {}, 0
with lzma.open(JOIN, "rt", encoding="utf-8") as fh:
    for line in fh:
        r = json.loads(line)
        n_lines += 1
        assert r["frame_idx"] == r["frame"] - 2, (r["frame"], r["frame_idx"])
        per_clip.setdefault(r["clip_id"], {})[int(r["frame_idx"])] = r["agents"]
print(f"[join] {n_lines} lines, {len(per_clip)} clips, {time.time()-t0:.1f}s",
      flush=True)

have = {f.split(".v2ep")[0] for f in os.listdir(EPS) if f.endswith(".v2ep.pt")}
clips = sorted(set(per_clip) & have)
plan = [(ci, r) for ci, c in enumerate(clips) for r in sorted(per_clip[c])[::a.stride]]
N = len(plan)
print(f"[corpus] join {len(per_clip)} x eps {len(have)} -> {len(clips)} clips, "
      f"N={N} rows (stride {a.stride})", flush=True)

# ⭐ ROW ALIGNMENT with WP-A, asserted rather than assumed: quoting WP-A's
# reference values only means anything if the two banks index the same frames.
wpa = json.load(open(WPA_IDX, encoding="utf-8"))
align = {"wpa_clips": len(wpa["clips"]), "wpa_N": int(wpa["N"]),
         "clips_equal": wpa["clips"] == clips,
         "plan_equal": [list(p) for p in plan] == [list(p) for p in wpa["plan"]]}
print("[align]", json.dumps(align), flush=True)

# ---- 2. the three trunks (identical architecture, three weight sets) -------
dev = "cuda" if torch.cuda.is_available() else "cpu"
cfg = CNNEncoderConfig(in_channels=9, image_size=256, image_width=640,
                       base_width=88, blocks=(3, 6, 16, 6))
GH, GW = cfg.grid_shape
F = cfg.feat_dim
assert (GH, GW) == (8, 20) and F == 704, (GH, GW, F)
enc, tinfo = {}, {}
for arm in ARMS:
    p = os.path.join(a.trunks, f"trunk_{arm}.pt")
    blob = torch.load(p, map_location="cpu", weights_only=False)
    m = ResNetEncoder(cfg)
    missing, unexpected = m.load_state_dict(blob["encoder"], strict=True), None
    m.eval().to(dev)
    for q in m.parameters():
        q.requires_grad_(False)
    enc[arm] = m
    tinfo[arm] = {"step": int(blob["step"]), "n_enc_params": int(blob["n_enc_params"]),
                  "bev_aux_keys": blob.get("bev_aux_keys", []),
                  "md5": hashlib.md5(open(p, "rb").read()).hexdigest()}
    print(f"[trunk {arm}] step {blob['step']} params {blob['n_enc_params']} "
          f"bev_aux_keys {len(blob.get('bev_aux_keys', []))} md5 {tinfo[arm]['md5'][:12]}",
          flush=True)

# ⛔ THE ARMS MUST NOT BE THE SAME WEIGHTS. Two trunks that are bit-identical
# would make every "difference" a tie-break, and nothing downstream would say so.
w_sig = {arm: float(torch.cat([p.reshape(-1)[:4096] for p in enc[arm].parameters()][:8]).abs().sum())
         for arm in ARMS}
print("[trunk-distinct]", json.dumps(w_sig), flush=True)
assert len(set(f"{v:.9f}" for v in w_sig.values())) == len(ARMS), \
    "TWO TRUNKS ARE IDENTICAL -- the panel would be comparing an arm with itself"

# ---- 3. memmaps -----------------------------------------------------------
NT = GH * GW
POL = BA.POLAR_DEFAULT
NCELL = GRID_DEFAULT.shape[0] * GRID_DEFAULT.shape[1]
NPOL = POL.n_rng * POL.n_az
mm = {arm: np.lib.format.open_memmap(os.path.join(a.out, f"tok_{arm}.npy"),
                                     mode="w+", dtype=np.float16, shape=(N, NT, F))
      for arm in ARMS}
pix_mm = np.lib.format.open_memmap(os.path.join(a.out, "pix.npy"), mode="w+",
                                   dtype=np.float16, shape=(N, NT, 9))
yc_mm = np.lib.format.open_memmap(os.path.join(a.out, "y_cart.npy"), mode="w+",
                                  dtype=np.uint8, shape=(N, NCELL))
yp_mm = np.lib.format.open_memmap(os.path.join(a.out, "y_pol.npy"), mode="w+",
                                  dtype=np.uint8, shape=(N, NPOL))
mp_mm = np.lib.format.open_memmap(os.path.join(a.out, "m_pol.npy"), mode="w+",
                                  dtype=np.uint8, shape=(N, NPOL))

# ---- 4. one decode, three encodes ----------------------------------------
w = 0
t0 = time.time()
rows_by_clip = {}
for ci, r in plan:
    rows_by_clip.setdefault(ci, []).append(r)
for ci, c in enumerate(clips):
    ep = decode_full_episode(os.path.join(EPS, c + ".v2ep.pt"))
    fr = ep.frames
    assert fr.shape[1] == 9 and tuple(fr.shape[2:]) == (256, 640), fr.shape
    rows = [r for r in rows_by_clip.get(ci, []) if r < fr.shape[0]]
    for b0 in range(0, len(rows), a.batch):
        sel = rows[b0:b0 + a.batch]
        f = to_float_frames(fr[sel]).to(dev)               # [b, 9, 256, 640]
        n = len(sel)
        with torch.no_grad():
            for arm in ARMS:
                fm, _pooled = enc[arm](f)                  # [b, 704, 8, 20]
                mm[arm][w:w + n] = (fm.flatten(2).transpose(1, 2)
                                    .to(torch.float16).cpu().numpy())
            pm = torch.nn.functional.avg_pool2d(f, (32, 32))   # [b, 9, 8, 20]
            pix_mm[w:w + n] = (pm.flatten(2).transpose(1, 2)
                               .to(torch.float16).cpu().numpy())
        for j, r in enumerate(sel):
            ags = per_clip[c][r]
            arr = agents_to_array(ags)
            yc_mm[w + j] = rasterize(arr, GRID_DEFAULT).astype(np.uint8).reshape(-1)
            occ, msk = BA.build_target(ags, labelled=True, spec=POL,
                                       occlusion="mask")
            yp_mm[w + j] = (occ > 0.5).astype(np.uint8).reshape(-1)
            mp_mm[w + j] = msk.astype(np.uint8).reshape(-1)
        w += n
    if ci % 20 == 0:
        el = time.time() - t0
        print(f"[bank] clip {ci}/{len(clips)} rows {w}/{N} {el:.0f}s "
              f"eta {el / max(w, 1) * (N - w):.0f}s", flush=True)
print(f"[bank] wrote {w} rows in {time.time()-t0:.0f}s", flush=True)
for arr in list(mm.values()) + [pix_mm, yc_mm, yp_mm, mp_mm]:
    arr.flush()

# ---- 5. CONTENT assertions ------------------------------------------------
step = max(1, w // 200)
chk = {}
for arm in ARMS:
    chk[f"mean_abs_tok_{arm}"] = float(np.abs(np.asarray(mm[arm][:w:step],
                                                         dtype=np.float32)).mean())
chk["mean_abs_pix"] = float(np.abs(np.asarray(pix_mm[:w:step], dtype=np.float32)).mean())
yc = np.asarray(yc_mm[:w]); yp = np.asarray(yp_mm[:w]); mp = np.asarray(mp_mm[:w]).astype(bool)
chk["cart_occ_rate_allcells"] = float(yc.mean())
chk["polar_base_rate_supervised"] = float(yp[mp].mean())
chk["polar_masked_frac"] = float(1.0 - mp.mean())
chk["rows_with_any_agent_cart"] = int((yc.sum(1) > 0).sum())
chk["rows_with_any_agent_polar"] = int((yp.sum(1) > 0).sum())
print("[assert]", json.dumps(chk, indent=1), flush=True)
assert all(chk[f"mean_abs_tok_{arm}"] > 1e-4 for arm in ARMS), "A TRUNK BANK IS ZERO"
assert chk["mean_abs_pix"] > 1e-4 and chk["cart_occ_rate_allcells"] > 0

# ⭐ the three trunks must produce DIFFERENT features on the same pixels
import itertools as _it
_dif = {}
for _a, _b in _it.combinations(ARMS, 2):
    _d = float(np.abs(np.asarray(mm[_a][:w:step], np.float32)
                      - np.asarray(mm[_b][:w:step], np.float32)).mean())
    _dif[f"mean_abs_diff_{_a}_{_b}"] = _d
    print(f"[distinct-features] |{_a}-{_b}| {_d:.6f}", flush=True)
chk.update(_dif)
assert all(v > 1e-4 for v in _dif.values()), "TWO TRUNKS PRODUCE IDENTICAL FEATURES"

# ⭐ CROSS-CHECK against WP-A's banked Cartesian target, if the plans aligned.
if align["plan_equal"]:
    ywpa = np.asarray(np.load(WPA_Y, mmap_mode="r"))[:w]
    align["y_cart_equals_wpa"] = bool(np.array_equal(yc, ywpa))
    align["y_cart_disagree_cells"] = int((yc != ywpa).sum())
    print("[align] y_cart vs WP-A:", align["y_cart_equals_wpa"],
          align["y_cart_disagree_cells"], flush=True)

json.dump({"clips": clips, "plan": plan[:w], "N": w, "stride": a.stride,
           "token_grid": [GH, GW], "feat_dim": F, "in_channels": 9,
           "grid": list(GRID_DEFAULT.shape), "cell_m": GRID_DEFAULT.cell_m,
           "x_fwd_m": GRID_DEFAULT.x_fwd_m, "y_half_m": GRID_DEFAULT.y_half_m,
           "polar_spec": POL.to_dict(), "arms": ARMS, "trunks": tinfo,
           "align_with_wpa": align, "assert": chk,
           "_evidence_class": "MEASURED (ours; artifact = this file + this bank)"},
          open(os.path.join(a.out, "idx.json"), "w", encoding="utf-8"), indent=1)
print("DONE", a.out, f"{time.time()-t0:.0f}s")
