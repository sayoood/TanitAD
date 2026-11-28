"""pod_lead_apply.py: applies the trained LEADSUP head of ONE checkpoint to the level-1 states of a state file (card PREREG_V8F_PROOF_ADD24, section 3a).
Runs ON THE POD (Linux, /workspace/venv/bin/python), CPU only, torch available, the patched tree at --tree (default /workspace/v8proof/tree_leadsup).
Tier T0 (a world-model diagnostic on logged frames); never driving performance.

  python pod_lead_apply.py --tree /workspace/v8proof/tree_leadsup --ckpt .../ckpt_last.pt --states <ARM>_eval.npz --out <ARM>_eval_lead.npz

State file (written by extract_states.py): Z [n, 640] float16, clip_idx [n], t [n], clips [n_clips]. For every row that has 4 predecessors which are
consecutive frames (t-1 .. t-4) of the SAME clip, the head reads the 5 states (oldest first, end frame = that row) and writes its three outputs;
every other row gets NaN. Output (row-aligned to the state file):
  <out>                 presence_logit, gap_m (= output x 50), closing_ms (= output x 10; positive = the gap opens), has_history (bool), hist
  <out>.head.npz        the six head tensors (frame_w, frame_b, mlp0_w, mlp0_b, mlp2_w, mlp2_b) as float32
  <out>.json            sha256 of the checkpoint, of the state file, of the head tensors and of the outputs, and the counts
The model is rebuilt from the checkpoint cfg exactly as scripts/lead_dump.py and v8f_t1_eval.py do (V8FMiniConfig(**ck["cfg"]), strict load).
A checkpoint trained without --lead-sup on is refused. No GPU: map_location cpu, CUDA hidden.
"""
import argparse
import hashlib
import json
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import numpy as np  # noqa: E402

GAP_SCALE_M = 50.0                   # = v8f_mini.LEAD_SUP_GAP_SCALE_M (asserted against the tree below)
CLOS_SCALE_MS = 10.0                 # = v8f_mini.LEAD_SUP_CLOS_SCALE_MS
HEAD_KEYS = {"frame_w": "wm.lead_head.frame.weight", "frame_b": "wm.lead_head.frame.bias", "mlp0_w": "wm.lead_head.mlp.0.weight",
             "mlp0_b": "wm.lead_head.mlp.0.bias", "mlp2_w": "wm.lead_head.mlp.2.weight", "mlp2_b": "wm.lead_head.mlp.2.bias"}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def history_rows(clip_idx, t, hist):
    """idx [n, hist] = row index of the frames t-hist+1 .. t of the same clip (oldest first), -1 where any of them is absent. Pure numpy."""
    clip_idx = np.asarray(clip_idx, np.int64)
    t = np.asarray(t, np.int64)
    key = {(int(c), int(g)): i for i, (c, g) in enumerate(zip(clip_idx, t))}
    idx = np.full((len(t), hist), -1, np.int64)
    for j in range(hist):
        off = hist - 1 - j
        for i, (c, g) in enumerate(zip(clip_idx, t)):
            idx[i, j] = key.get((int(c), int(g) - off), -1)
    ok = (idx >= 0).all(1)
    idx[~ok] = -1
    return idx, ok


def load_model(tree, ckpt_path):
    sp = os.path.join(tree, "stack")
    for p in (sp, os.path.join(sp, "scripts")):
        if p not in sys.path:
            sys.path.insert(0, p)
    import torch
    from tanitad.models import v8f_mini as M
    tf = os.path.abspath(sys.modules["tanitad"].__file__).replace(os.sep, "/")
    if not tf.startswith(os.path.abspath(tree).replace(os.sep, "/")):
        raise SystemExit("REFUSED: tanitad is imported from %s, not from the tree %s" % (tf, tree))
    assert M.LEAD_SUP_GAP_SCALE_M == GAP_SCALE_M and M.LEAD_SUP_CLOS_SCALE_MS == CLOS_SCALE_MS
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = M.V8FMiniConfig(**ck["cfg"])
    if cfg.lead_sup != "on":
        raise SystemExit("REFUSED: %s was trained with lead_sup %r: it has no lead head to read" % (ckpt_path, cfg.lead_sup))
    model = M.V8FMini(cfg, M.P10Prior(torch.zeros(8), torch.ones(8), torch.zeros(2 * M.K30)))
    model.load_state_dict(ck["model"])
    if getattr(model.wm, "lead_head", None) is None:
        raise SystemExit("REFUSED: the rebuilt model has no wm.lead_head")
    return cfg, model.eval(), ck


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", default="/workspace/v8proof/tree_leadsup")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--states", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--plumbing", action="store_true", help="dev-box plumbing check only: do not refuse a state file extracted from another checkpoint (recorded in the json)")
    a = ap.parse_args(argv)
    import torch
    torch.set_num_threads(a.threads)
    t0 = time.time()
    cfg, model, ck = load_model(a.tree, a.ckpt)
    head = model.wm.lead_head
    hist = int(head.hist)
    z = np.load(a.states, allow_pickle=False)
    for k in ("Z", "clip_idx", "t", "clips"):
        if k not in z.files:
            raise SystemExit("REFUSED: state file lacks %s" % k)
    Z, ci, tt = z["Z"], z["clip_idx"], z["t"]
    if Z.ndim != 2 or Z.shape[1] != head.frame.in_features:
        raise SystemExit("REFUSED: state width %s, the head reads %d" % (Z.shape, head.frame.in_features))
    rec_path = a.states + ".json"
    rec = json.load(open(rec_path, encoding="utf-8")) if os.path.exists(rec_path) else None
    ck_sha = sha256_file(a.ckpt)
    if rec is not None and rec.get("ckpt_sha256") not in (None, ck_sha) and not a.plumbing:
        raise SystemExit("REFUSED: the state file was extracted from checkpoint %s, this checkpoint is %s" % (rec.get("ckpt_sha256"), ck_sha))
    idx, ok = history_rows(ci, tt, hist)
    n = len(Z)
    rows = np.flatnonzero(ok)
    raw = np.full((n, 3), np.nan, np.float32)
    with torch.no_grad():
        for i in range(0, len(rows), a.chunk):
            r = rows[i:i + a.chunk]
            x = torch.from_numpy(np.asarray(Z[idx[r].reshape(-1)], np.float32)).reshape(len(r), hist, -1)
            o = head(x)                                          # [b, 1, 3]: the end frame is the last of the `hist` frames
            assert o.shape[1] == 1
            raw[r] = o[:, 0].float().numpy()
    out = {"presence_logit": raw[:, 0].copy(), "gap_m": (raw[:, 1] * GAP_SCALE_M).astype(np.float32), "closing_ms": (raw[:, 2] * CLOS_SCALE_MS).astype(np.float32),
           "has_history": ok, "hist": np.int64(hist)}
    with open(a.out, "wb") as fh:
        np.savez(fh, **out)
    sd = {k: v for k, v in ck["model"].items()}
    tens = {nm: sd[key].detach().cpu().numpy().astype(np.float32) for nm, key in HEAD_KEYS.items()}
    hp = a.out + ".head.npz"
    with open(hp, "wb") as fh:
        np.savez(fh, **tens, hist=np.int64(hist))
    record = {"ckpt": a.ckpt, "ckpt_sha256": ck_sha, "states": a.states, "states_sha256": sha256_file(a.states), "states_record_ckpt_sha256": (rec or {}).get("ckpt_sha256"),
              "out": a.out, "out_sha256": sha256_file(a.out), "head": hp, "head_sha256": sha256_file(hp), "n_rows": int(n), "n_applied": int(len(rows)), "n_nan": int(n - len(rows)),
              "hist": hist, "n_clips": int(len(z["clips"])), "cfg_lead": {k: getattr(cfg, k) for k in ("lead_sup", "w_lead_sup", "lead_hist", "lead_sup_detach")},
              "step": ck.get("step"), "plumbing": bool(a.plumbing), "seconds": round(time.time() - t0, 1), "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with open(a.out + ".json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(record, fh, indent=1)
    print("pod_lead_apply: %d rows, %d with a %d-frame history, ckpt %s -> %s" % (n, len(rows), hist, ck_sha[:12], a.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
