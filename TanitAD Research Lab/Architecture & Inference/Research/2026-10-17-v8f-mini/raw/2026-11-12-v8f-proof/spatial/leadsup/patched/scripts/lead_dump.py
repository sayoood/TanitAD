"""lead_dump.py: the READ HOOK of the LEADSUP head (new file, outside the pinned set; nothing of the driving path imports it).

Loads a checkpoint trained with --lead-sup on, takes frames in the layout of the eval windows (frames 0 .. 4 of a window: the S0 = 4 history of spatial/objslot) and writes the head's outputs per window:
  presence probability (sigmoid of the logit), gap [m] (output x 50), closing speed [m/s] (output x 10; the slope of the gap, positive = the gap opens), at the END frame of the supplied frames.

Input (exactly one):
  --feats  FILE  .npy / .npz[key feats]   cached DINOv3 tokens [n, F, n_tok, d_feat] (float16 ok): the path adapter -> + pos -> level-1 projection -> head, exactly as V8FMini.loss feeds the head
  --states FILE  .npy / .npz[key Z1]      level-1 grid states [n, F, n_tok * wm_cell_dim] (the Z1 arrays of the latents files): straight into the head
F must be >= the head's history (5 by default); with F = 5 the single end frame is frame 4 (window frames 0 .. 4). With F > hist the end frames are hist-1 .. F-1 and --end-frame picks one (default: the last).
Output: --out NPZ with presence_prob, gap_m, closing_ms [n], logits [n, 3], end_frame, hist, plus the sha256 of the checkpoint and of the input file and the lead fields of the checkpoint config.
Tier T0 (a world-model diagnostic read on logged frames): never driving performance. CPU or cuda; chunks of --chunk windows.
Usage: python lead_dump.py --ckpt CKPT --feats FILE|--states FILE --out OUT.npz [--end-frame N] [--chunk 256]"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.join(HERE, ".."), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from tanitad.models import v8f_mini as M  # noqa: E402


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def load_array(path: str, key: str) -> np.ndarray:
    if path.endswith(".npy"):
        return np.load(path)
    z = np.load(path)
    if key not in z.files:
        raise SystemExit(f"REFUSED: {path} has no array {key!r} (it has {z.files})")
    return z[key]


def load_model(ckpt_path: str):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = M.V8FMiniConfig(**ck["cfg"])                                   # the same rebuild v8f_t1_eval.py does: the cfg carries the lead fields, so the head is built and the load is strict
    if cfg.lead_sup != "on":
        raise SystemExit(f"REFUSED: {ckpt_path} was trained with lead_sup {cfg.lead_sup!r}: it has no lead head to read")
    model = M.V8FMini(cfg, M.P10Prior(torch.zeros(8), torch.ones(8), torch.zeros(2 * M.K30)))
    model.load_state_dict(ck["model"])
    return cfg, model.eval()


@torch.no_grad()
def head_outputs(model, cfg, x: torch.Tensor, kind: str) -> torch.Tensor:
    """x: tokens [B, F, n_tok, d_feat] (kind feats) or states [B, F, D] (kind states) -> head output [B, F - hist + 1, 3] (presence logit, gap / 50, closing / 10)."""
    if kind == "feats":
        adapt = model.adapter(x.float()) + model.pos[None, None]          # as V8FMini.loss: [B, F, n_tok, d]
        z1 = model.wm.z1(adapt)
    else:
        z1 = x.float()
    return model.wm.lead_head(z1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--feats")
    g.add_argument("--states")
    ap.add_argument("--out", required=True)
    ap.add_argument("--end-frame", type=int, default=-1, help="the end frame to report (default: the last supplied frame)")
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)
    cfg, model = load_model(a.ckpt)
    kind, path = ("feats", a.feats) if a.feats else ("states", a.states)
    X = load_array(path, "feats" if kind == "feats" else "Z1")
    hist = cfg.lead_hist
    F_ = X.shape[1]
    if F_ < hist:
        raise SystemExit(f"REFUSED: the input has {F_} frames per window, the head reads {hist}")
    end = (F_ - 1) if a.end_frame < 0 else a.end_frame
    if not hist - 1 <= end <= F_ - 1:
        raise SystemExit(f"REFUSED: end frame {end} outside [{hist - 1}, {F_ - 1}]")
    n = X.shape[0]
    outs = []
    for i in range(0, n, a.chunk):
        xb = torch.as_tensor(np.asarray(X[i:i + a.chunk]))
        outs.append(head_outputs(model, cfg, xb, kind)[:, end - (hist - 1)].float())
    o = torch.cat(outs).numpy() if outs else np.zeros((0, 3), np.float32)
    res = dict(presence_prob=(1.0 / (1.0 + np.exp(-o[:, 0]))).astype(np.float32), gap_m=(o[:, 1] * M.LEAD_SUP_GAP_SCALE_M).astype(np.float32),
               closing_ms=(o[:, 2] * M.LEAD_SUP_CLOS_SCALE_MS).astype(np.float32), logits=o.astype(np.float32), end_frame=np.int64(end), hist=np.int64(hist))
    np.savez_compressed(a.out, **res, ckpt_sha256=sha256_file(a.ckpt), input_sha256=sha256_file(path), input_kind=kind,
                        cfg_lead=json.dumps({k: getattr(cfg, k) for k in ("lead_sup", "w_lead_sup", "lead_hist", "lead_sup_detach")}))
    print(f"lead_dump: {n} windows, end frame {end}, hist {hist}, input {kind} {tuple(X.shape)} -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
