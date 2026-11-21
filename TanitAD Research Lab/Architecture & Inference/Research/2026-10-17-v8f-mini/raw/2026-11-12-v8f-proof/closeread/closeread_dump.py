#!/usr/bin/env python
"""closeread_dump.py -- card PREREG_V8F_PROOF_ADD77 (CLOSEREAD), section 2: the per-frame reads of one token reader seed on cached tokens, for two splits.
  fit   the clips held out of the reader's training by the rule int(sha256(clip id)[:8], 16) % 10 == 0 (tok_probe_lr.py:317), recomputed from the manifest of the train cache after the
        trainer's --exclude slice (v8f_mini_train.py:2720) and checked against n_heldout, n_trained and heldout_sha256 of the reader's train_meta.json (the hash recipe is
        sha256 of the sorted held-out ids joined by a line feed: tok_probe_lr.py:317); a mismatch is a refusal.
  score all clips of the evaluation cache.
Per (clip, frame) it writes: clip index (row of the join), clip id index, frame, presence p, gap read (before any map), closing output times 10 m/s (reader_closing of
v8f_t1_eval_reader_rule_rzgfhqc.py:218, CLOS_SCALE_MS 10), validity of the five-frame window (frames f-4..f all exist in the clip) and the labels present / gap / closing / run length
of the label source of that split (LabelSource of v8f_t1_eval_rule.py:79, rewritten here for whole clips). It prints counts and paths only, never a comparison of a read with a label.
The output folder must not exist. Tier T0.

  python closeread_dump.py --reader CKPT --reader-seed K --train-cache DIR --eval-cache DIR --join-train NPZ --join-eval NPZ --train-meta JSON --build-dir DIR --out-dir NEWDIR [--device cuda]
Writes reader_s<K>_fit.npz, reader_s<K>_score.npz, reader_s<K>_lists.json, reader_s<K>_meta.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GAP_MAX_M, STEP_MAX_M, RUN_MAX, RUN_MIN = 60.0, 3.0, 5, 3      # v8f_t1_eval_rule.py:37-39
DT_S = 0.2                                                      # dist_rule.py:25 DT; v8f_t1_eval_rule.py:110 (LabelSource time axis)
CLOS_SCALE_MS, GAP_SCALE_M = 10.0, 50.0                         # v8f_t1_eval_reader_rule_rzgfhqc.py:161; tok_reader.py:33
PARITY_TOL = 1e-3


def heldout_rule(clip_id):
    return int(hashlib.sha256(str(clip_id).encode("utf-8")).hexdigest()[:8], 16) % 10 == 0


def sha12(clip_id):
    return hashlib.sha256(str(clip_id).encode("utf-8")).hexdigest()[:12]


def list_sha256(ids):
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def parse_exclude(meta):
    """'--exclude A:B' of the trainer's argv in train_meta.json (a list of tokens or one string) -> (A, B) or None."""
    argv = meta.get("trainer_argv", "")
    s = " ".join(map(str, argv)) if isinstance(argv, (list, tuple)) else str(argv)
    m = re.search(r"--exclude[ =](\d+):(\d+)", s)
    return (int(m.group(1)), int(m.group(2))) if m else None


def reader_lists(manifest_ids, meta):
    """-> dict of the reader's lists; raises SystemExit when the rule recomputed from the manifest does not give the count and the hash of train_meta.json."""
    ho = meta["heldout"]
    ex = parse_exclude(meta)
    excluded = set(sorted(manifest_ids, key=sha12)[ex[0]:ex[1]]) if ex else set()
    held = sorted(c for c in manifest_ids if c not in excluded and heldout_rule(c))
    train = sorted(c for c in manifest_ids if c not in excluded and not heldout_rule(c))
    sha = list_sha256(held)
    bad = []
    if len(held) != int(ho["n_heldout"]):
        bad.append("held-out count %d, train_meta %s" % (len(held), ho["n_heldout"]))
    if len(train) != int(ho["n_trained"]):
        bad.append("trained count %d, train_meta %s" % (len(train), ho["n_trained"]))
    if ho.get("heldout_sha256") and sha != ho["heldout_sha256"]:
        bad.append("held-out list hash differs from train_meta")
    if bad:
        raise SystemExit("REFUSED: " + "; ".join(bad))
    return {"rule": ho["rule"], "manifest_ids": sorted(manifest_ids), "exclude": list(ex) if ex else None, "excluded_ids": sorted(excluded), "heldout_ids": held, "train_ids": train,
            "meta_n_heldout": int(ho["n_heldout"]), "meta_n_trained": int(ho["n_trained"]), "meta_heldout_sha256": ho.get("heldout_sha256", ""), "recomputed_heldout_sha256": sha,
            "hash_recipe": "sha256 of the sorted held-out clip ids joined by a line feed (tok_probe_lr.py:317)"}


# ---------------------------------------------------------------- labels (LabelSource.read of v8f_t1_eval_rule.py:92-112 for every frame of a clip)
def label_clip(join, ci, T, dt=DT_S):
    """-> present [T] bool, gap [T] (NaN where not present), closing [T] (0.0 where the run is shorter than RUN_MIN, as the label source returns it), run [T] int (frames in the trailing run, 0 where not present)."""
    off, n = int(join["off"][ci]), int(join["n"][ci])
    gap, val = np.asarray(join["lead_gap"], np.float64), np.asarray(join["lead_valid"])
    ok = lambda r: bool(val[r] == 1 and np.isfinite(gap[r]) and 0.0 <= gap[r] <= GAP_MAX_M)
    pres, g_o, clo, run = np.zeros(T, bool), np.full(T, np.nan), np.zeros(T), np.zeros(T, np.int64)
    for j in range(T):
        if j >= n:
            continue
        r = off + j
        if not ok(r):
            continue
        idx = [r]
        f = r - 1
        while f >= off and len(idx) < RUN_MAX and ok(f) and abs(gap[f + 1] - gap[f]) <= STEP_MAX_M:
            idx.append(f)
            f -= 1
        pres[j], g_o[j], run[j] = True, float(gap[r]), len(idx)
        if len(idx) >= RUN_MIN:
            idx = idx[::-1]
            G = gap[idx]
            t = np.arange(len(idx), dtype=np.float64) * dt
            tb = t.mean()
            clo[j] = float(((t - tb) * (G - G.mean())).sum() / ((t - tb) ** 2).sum())
    return pres, g_o, clo, run


# ---------------------------------------------------------------- the reader (torch forward of the weights of tok_reader.TokReader)
class TorchReader:
    def __init__(self, rd, device):
        import torch
        self.torch, self.rd, self.dev = torch, rd, torch.device(device)
        self.w = {k: torch.from_numpy(np.ascontiguousarray(v)).to(self.dev) for k, v in rd.w.items()}
        self.hist, self.n_tok, self.W = rd.hist, rd.n_tok, rd.W

    def clip(self, tokens, chunk=64):
        """tokens [T, n_tok, 1024] -> p, gap_m, closing_ms (float64 [T], NaN for frames before the first full window), valid [T]."""
        t, w = self.torch, self.w
        T = int(tokens.shape[0])
        E = []
        with t.no_grad():
            for i in range(0, T, chunk):
                x = t.from_numpy(np.ascontiguousarray(tokens[i:i + chunk], np.float32)).to(self.dev)
                x = t.nn.functional.layer_norm(x, (x.shape[-1],), eps=1e-5)
                h = t.nn.functional.gelu(x @ w["tok.0.weight"].T + w["tok.0.bias"])
                z = h @ w["tok.2.weight"].T + w["tok.2.bias"]
                z = z.reshape(z.shape[0], self.n_tok * self.W)
                E.append(z @ w["head.frame.weight"].T + w["head.frame.bias"])
            E = t.cat(E)
            p, g, c = (np.full(T, np.nan) for _ in range(3))
            valid = np.zeros(T, bool)
            if T >= self.hist:
                idx = t.arange(self.hist - 1, T, device=self.dev)
                win = t.stack([E[idx - (self.hist - 1 - k)] for k in range(self.hist)], 1).reshape(len(idx), -1)
                h = t.nn.functional.gelu(win @ w["head.mlp.0.weight"].T + w["head.mlp.0.bias"])
                o = (h @ w["head.mlp.2.weight"].T + w["head.mlp.2.bias"]).double().cpu().numpy()
                p[self.hist - 1:], g[self.hist - 1:], c[self.hist - 1:] = 1.0 / (1.0 + np.exp(-o[:, 0])), o[:, 1] * GAP_SCALE_M, o[:, 2] * CLOS_SCALE_MS
                valid[self.hist - 1:] = True
        return p, g, c, valid


def numpy_reference(rd, tokens, f):
    """the reader on the five frames ending at f, written the way reader_closing and TokReader.head_from_embeds write it: (p, gap_m, closing_ms)."""
    import tok_reader as TR
    E = rd.frame_embed(np.asarray(tokens[f - rd.hist + 1:f + 1], np.float32)[None])
    x = E.reshape(1, -1)
    h = TR._gelu(x @ rd.w["head.mlp.0.weight"].T + rd.w["head.mlp.0.bias"])
    o = h @ rd.w["head.mlp.2.weight"].T + rd.w["head.mlp.2.bias"]
    return 1.0 / (1.0 + np.exp(-float(o[0, 0]))), float(o[0, 1]) * GAP_SCALE_M, float(o[0, 2]) * CLOS_SCALE_MS


def read_tokens(path):
    import torch
    d = torch.load(path, map_location="cpu")
    x = d["pool4x10"]
    return x.to(torch.float16).float().numpy()                   # the path of check_midlook_closing.py:342


def dump_split(trd, rd, clips, cache_split_dir, join, join_clips, split, seed, dt, parity_log):
    row = {c: i for i, c in enumerate(join_clips)}
    cols = {k: [] for k in ("cid", "clip_index", "frame", "p", "gap", "clos", "valid", "lab_present", "lab_gap", "lab_closing", "lab_run")}
    for n, c in enumerate(clips):
        if c not in row:
            raise SystemExit("REFUSED: clip %s is not in the join of the %s split" % (c, split))
        x = read_tokens(os.path.join(cache_split_dir, "clips", c + ".pt"))
        if x.ndim != 3 or x.shape[1] != rd.n_tok or x.shape[2] != 1024:
            raise SystemExit("REFUSED: tokens of %s have shape %s, expected [T, %d, 1024]" % (c, list(x.shape), rd.n_tok))
        T = int(x.shape[0])
        p, g, cl, valid = trd.clip(x)
        if n == 0 or n == len(clips) // 2:
            for f in sorted({rd.hist - 1, T // 2, T - 1}):
                if f >= rd.hist - 1:
                    ref = np.array(numpy_reference(rd, x, f))
                    got = np.array([p[f], g[f], cl[f]])
                    parity_log.append({"split": split, "clip": c, "frame": int(f), "max_abs_diff": float(np.abs(ref - got).max())})
        pr, lg, lc, lr = label_clip(join, row[c], T, dt)
        for k, v in (("cid", np.full(T, n)), ("clip_index", np.full(T, row[c])), ("frame", np.arange(T)), ("p", p), ("gap", g), ("clos", cl), ("valid", valid),
                     ("lab_present", pr), ("lab_gap", lg), ("lab_closing", lc), ("lab_run", lr)):
            cols[k].append(v)
    out = {k: np.concatenate(v) for k, v in cols.items()}
    out["clips"] = np.asarray(clips)
    out["dt_s"] = np.float64(dt)
    out["seed"] = np.int64(seed)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(allow_abbrev=False)
    for a in ("reader", "train-cache", "eval-cache", "join-train", "join-eval", "train-meta", "build-dir", "out-dir"):
        ap.add_argument("--" + a, required=True)
    ap.add_argument("--reader-seed", type=int, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dt", type=float, default=DT_S)
    a = ap.parse_args(argv)
    if os.path.exists(a.out_dir):
        print("REFUSED: output folder exists: " + a.out_dir)
        return 2
    sys.path.insert(0, a.build_dir)
    import torch
    import tok_reader as TR
    dev = a.device if (a.device != "cuda" or torch.cuda.is_available()) else "cpu"
    dt_note = "argument (default 0.2: dist_rule.py:25 DT)"
    try:
        import dist_rule as DR
        if abs(float(DR.DT) - a.dt) > 1e-12:
            print("REFUSED: dist_rule.DT = %r differs from --dt %r" % (DR.DT, a.dt))
            return 2
        dt_note = "equals dist_rule.DT of the build folder"
    except SystemExit:
        raise
    except Exception:                                           # noqa: BLE001
        dt_note += "; dist_rule could not be imported here, not compared"
    rd = TR.TokReader.from_ckpt(a.reader)
    if rd.meta.get("seed") is not None and int(rd.meta["seed"]) != a.reader_seed:
        print("REFUSED: the checkpoint records seed %s, --reader-seed is %d" % (rd.meta["seed"], a.reader_seed))
        return 2
    meta = json.load(open(a.train_meta, encoding="utf-8"))
    man = [json.loads(l)["clip"] for l in open(os.path.join(a.train_cache, "train", "manifest.jsonl"), encoding="utf-8") if l.strip()]
    lists = reader_lists(man, meta)
    eman = [json.loads(l)["clip"] for l in open(os.path.join(a.eval_cache, "eval", "manifest.jsonl"), encoding="utf-8") if l.strip()]
    Jt, Je = np.load(a.join_train, allow_pickle=True), np.load(a.join_eval, allow_pickle=True)
    jt, je = [str(c) for c in Jt["clip"]], [str(c) for c in Je["clip"]]
    score_clips = sorted(eman, key=lambda c: je.index(c) if c in je else 10 ** 9)
    os.makedirs(a.out_dir)
    trd = TorchReader(rd, dev)
    parity = []
    Jt_d = {k: Jt[k] for k in ("off", "n", "lead_gap", "lead_valid")}
    Je_d = {k: Je[k] for k in ("off", "n", "lead_gap", "lead_valid")}
    fit = dump_split(trd, rd, lists["heldout_ids"], os.path.join(a.train_cache, "train"), Jt_d, jt, "fit", a.reader_seed, a.dt, parity)
    score = dump_split(trd, rd, score_clips, os.path.join(a.eval_cache, "eval"), Je_d, je, "score", a.reader_seed, a.dt, parity)
    worst = max(x["max_abs_diff"] for x in parity)
    if worst > PARITY_TOL:
        print("REFUSED: the torch forward differs from the numpy reader by %.3g (tolerance %.0e); nothing is written" % (worst, PARITY_TOL))
        return 2
    paths = {}
    for nm, D in (("fit", fit), ("score", score)):
        p = os.path.join(a.out_dir, "reader_s%d_%s.npz" % (a.reader_seed, nm))
        with open(p, "wb") as fh:
            np.savez(fh, **D)
        paths[nm] = p
    json.dump(lists, open(os.path.join(a.out_dir, "reader_s%d_lists.json" % a.reader_seed), "w", encoding="utf-8", newline="\n"))
    info = {"card": "PREREG_V8F_PROOF_ADD77", "tier": "T0", "reader": a.reader, "reader_sha256": sha256_file(a.reader), "reader_meta": {k: (v if isinstance(v, (int, float, str)) or v is None else str(v)) for k, v in rd.meta.items()},
            "reader_seed": a.reader_seed, "device": dev, "dt_s": a.dt, "dt_note": dt_note, "n_fit_clips": len(lists["heldout_ids"]), "n_score_clips": len(score_clips), "n_fit_rows": int(len(fit["frame"])), "n_score_rows": int(len(score["frame"])),
            "heldout_check": {"n_heldout": lists["meta_n_heldout"], "n_trained": lists["meta_n_trained"], "sha_recomputed": lists["recomputed_heldout_sha256"], "sha_train_meta": lists["meta_heldout_sha256"], "exclude": lists["exclude"]},
            "torch_vs_numpy_parity_max_abs_diff": worst, "parity_rows": parity, "files": {k: {"path": v, "sha256": sha256_file(v)} for k, v in paths.items()}, "join_train": a.join_train, "join_eval": a.join_eval,
            "train_meta": a.train_meta}
    json.dump(info, open(os.path.join(a.out_dir, "reader_s%d_meta.json" % a.reader_seed), "w", encoding="utf-8", newline="\n"), indent=1)
    print("dump seed %d: fit clips %d rows %d, score clips %d rows %d, parity %.2e, out %s" % (a.reader_seed, info["n_fit_clips"], info["n_fit_rows"], info["n_score_clips"], info["n_score_rows"], worst, a.out_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
