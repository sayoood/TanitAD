#!/usr/bin/env python
"""tokclose_fit.py -- card PREREG_V8F_PROOF_ADD81 (TOKCLOSE), tier T0: does a temporal head on the token features of the last H frames read the closing speed better than the estimate
from the reader's three outputs (cells N_H10 and L_H25 of card ADD78)?  Definitions and every fixed hyperparameter: TOKCLOSE_DEFS.md.

  fit    one arm (E10, E25, T10, T25) and seed; writes tokclose_<TAG>.pt (head), tokclose_<TAG>_log.json, tokclose_<TAG>_table.npz (estimate per (clip index, frame) of the evaluation clips)
  score  controls t0..t4 first; then names, TOKCLOSE_RESULT.json and TOKCLOSE_RESULT.txt (nothing about an arm is printed or written when a control fails)
  --selftest   CPU, synthetic clips with a planted closing signal in the tokens

Imported unchanged: closeread_dump (label function, held-out lists), closeread_fit (rows, statistics), closest (saved estimators), tok_reader (frame embedding of the reader).
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import copy
import hashlib
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, os.path.join(HERE, "..", "closeread"), os.path.join(HERE, "..", "integ10k")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
import closeread_dump as CD  # noqa: E402
import closeread_fit as CF  # noqa: E402
import closest as CL  # noqa: E402

CARD = "PREREG_V8F_PROOF_ADD81"
READER_HIST = 5
RUN_MIN = CD.RUN_MIN
ARMS = ("E10", "E25", "T10", "T25")
HS = (10, 25)
CELL_OF_H = {10: ("N10", "N_H10"), 25: ("L25", "L_H25")}
BANNED = ["vo" + "id", "super" + "seded", "re" + "tracted", "for" + "merly", "cor" + "rection", "with" + "drawn", "no lo" + "nger", "clo" + "sed loop", "driving " + "performance"]
HP = {"E": {"batch": 8, "lr": 1e-3, "clip_norm": 1.0, "max_evals": 100, "patience": 10},
      "T": {"batch": 8, "lr": 1e-3, "clip_norm": 1.0, "max_evals": 8, "patience": 2},
      "huber_delta": 1.0, "stop_frac": 0.2, "split_seed": 0, "shuffle_seed": 20261120, "order_seed_base": 1000}
DIMS = {"W": 16, "emb": 64, "hid": 128, "mid": 64, "tok_hidden": 128}
EXPECT = {"train": 4080, "heldout": 432, "eval": 141}
T3_TOL = 1e-5
T1_BAND = 0.10
T0_TOL_R = 0.005
T0_TOL_LABEL = 1e-6


def parse_arm(name):
    if len(name) < 3 or name[0] not in "ET" or not name[1:].isdigit() or int(name[1:]) not in HS:
        raise SystemExit("REFUSED: arm %r is not one of %s" % (name, ", ".join(ARMS)))
    return name[0], int(name[1:])


def tag_of(arm, seed, shuffled=False):
    mode, H = parse_arm(arm)
    return "%s%s_s%d" % (arm, "shuf" if shuffled else "", seed)


def sha_ids(ids):
    return hashlib.sha256("\n".join(sorted(map(str, ids))).encode("utf-8")).hexdigest()


def sha16(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


# ============================================================================================ frames and labels
def eligible_frames(pres, run, clo, T, H):
    """frames f of a clip that are rows of the card: the H reads on f-H+1..f exist (a read at j needs frames j-4..j, so f >= H+3), a label with a lead and a run of at least 3 frames, finite closing speed.
    The same rule for fit and score; test_rows_equal_build_rows proves it equals closeread_fit.build_rows."""
    f = np.arange(T)
    ok = (f >= H + READER_HIST - 2) & np.asarray(pres, bool) & (np.asarray(run, np.int64) >= RUN_MIN) & np.isfinite(np.asarray(clo, np.float64))
    return f[ok]


def build_clip_rows(join, join_clips, clip_ids, H, dt=CD.DT_S):
    """clip id -> (frames int64, closing speed float32); labels by the one label function (closeread_dump.label_clip) on the join of the split."""
    row = {c: i for i, c in enumerate(join_clips)}
    j = {k: np.asarray(join[k]) for k in ("off", "n", "lead_gap", "lead_valid")}
    out = {}
    for c in clip_ids:
        if c not in row:
            raise SystemExit("REFUSED: clip %s is not in the join" % c)
        ci = row[c]
        T = int(j["n"][ci])
        pres, _, clo, run = CD.label_clip(j, ci, T, dt)
        f = eligible_frames(pres, run, clo, T, H)
        out[c] = (f.astype(np.int64), clo[f].astype(np.float32))
    return out


def permute_labels(rows, ids, seed):
    """control t1: the finite closing labels permuted among the labelled rows of the given clips (the mask, hence the counts, stay)."""
    ids = [c for c in ids if len(rows[c][1])]
    if not ids:
        return rows
    ys = np.concatenate([rows[c][1] for c in ids])
    ys = ys[np.random.default_rng(seed).permutation(len(ys))]
    out, p = dict(rows), 0
    for c in ids:
        n = len(rows[c][1])
        out[c] = (rows[c][0], ys[p:p + n].copy())
        p += n
    return out


# ============================================================================================ model
class TokClose(nn.Module):
    """E: input is the frozen frame embedding [T, emb]; T: input is tokens [T, n_tok, d_feat], embedded by a trainable projection of the form of the reader's per-frame part.
    Head: LayerNorm(emb), one GRU layer (hid units) over the H embeddings that end at the frame, Linear(hid, mid), ReLU, Linear(mid, 1). Output at frame f reads frames f-H+1..f only.
    ``shift`` is a test hook (a head that looks that many frames ahead); it is 0 in every fit."""

    def __init__(self, mode, H, d_feat=1024, n_tok=40, W=16, emb=64, hid=128, mid=64, tok_hidden=128):
        super().__init__()
        self.mode, self.H, self.shift = mode, int(H), 0
        self.dims = {"d_feat": int(d_feat), "n_tok": int(n_tok), "W": int(W), "emb": int(emb), "hid": int(hid), "mid": int(mid), "tok_hidden": int(tok_hidden)}
        if mode == "T":
            self.tok = nn.Sequential(nn.Linear(d_feat, tok_hidden), nn.GELU(), nn.Linear(tok_hidden, W))
            self.frame = nn.Linear(n_tok * W, emb)
        self.norm = nn.LayerNorm(emb)
        self.gru = nn.GRU(emb, hid, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(hid, mid)
        self.fc2 = nn.Linear(mid, 1)

    def embed(self, tokens):
        x = F.layer_norm(tokens.float(), (tokens.shape[-1],))
        return self.frame(self.tok(x).flatten(-2))

    def head(self, win):
        o, _ = self.gru(self.norm(win))
        return self.fc2(F.relu(self.fc1(o[:, -1])))[:, 0]

    def rows(self, E, frames):
        """E [T, emb] per-frame embeddings of one clip, frames [N] -> [N] estimates; the window of frame f is the H embeddings f-H+1..f."""
        off = torch.arange(-self.H + 1, 1, device=E.device) + self.shift
        idx = (frames[:, None] + off[None, :]).clamp(0, E.shape[0] - 1)
        return self.head(E[idx])


def make_model(mode, H, dims):
    return TokClose(mode, H, **dims)


def reader_embed(rd, tokens_f32, chunk=64):
    return np.concatenate([rd.frame_embed(tokens_f32[i:i + chunk]) for i in range(0, len(tokens_f32), chunk)]).astype(np.float32)


def load_tokens(path):
    d = torch.load(path, map_location="cpu")
    x = d["pool4x10"]
    if x.ndim != 3:
        raise SystemExit("REFUSED: tokens of %s have shape %s" % (path, list(x.shape)))
    return x.to(torch.float16)                                  # the path of closeread_dump.read_tokens


def clip_input(mode, tokens, rd):
    """what the model takes for one clip: tokens (T arms) or the frozen frame embedding of the reader (E arms)."""
    if mode == "T":
        return tokens
    return torch.from_numpy(reader_embed(rd, tokens.float().numpy()))


@torch.no_grad()
def predict_clip(model, x, device):
    """x as clip_input gives it -> estimate [T] float64, NaN before frame H-1 (no full window of embeddings)."""
    model.eval()
    x = x.to(device)
    E = model.embed(x) if model.mode == "T" else x
    T = int(E.shape[0])
    out = np.full(T, np.nan)
    if T >= model.H:
        fr = torch.arange(model.H - 1, T, device=device)
        out[model.H - 1:] = model.rows(E, fr).double().cpu().numpy()
    return out


# ============================================================================================ training
def _prefetch(batches, load, depth=2, workers=3):
    """yield (ids, [load(id) ...]) in order, the loads of the next batches running in threads (at most depth batches ahead resident)."""
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        q = collections.deque()
        it = iter(batches)

        def push():
            try:
                b = next(it)
            except StopIteration:
                return False
            q.append((b, [pool.submit(load, c) for c in b]))
            return True
        for _ in range(depth):
            if not push():
                break
        while q:
            b, futs = q.popleft()
            data = [f.result() for f in futs]
            push()
            yield b, data


def _batch_loss(model, ids, data, rows, device, delta, reduction):
    outs, ys = [], []
    for c, x in zip(ids, data):
        f, y = rows[c]
        x = x.to(device)
        E = model.embed(x) if model.mode == "T" else x
        outs.append(model.rows(E, torch.from_numpy(f).to(device)))
        ys.append(torch.from_numpy(y).to(device))
    o, y = torch.cat(outs), torch.cat(ys)
    return F.huber_loss(o, y, delta=delta, reduction=reduction), int(len(y))


@torch.no_grad()
def eval_loss(model, ids, load, rows, device, hp):
    model.eval()
    ids = [c for c in ids if len(rows[c][1])]
    tot, n = 0.0, 0
    for b, data in _prefetch([ids[i:i + hp["batch"]] for i in range(0, len(ids), hp["batch"])], load):
        l, m = _batch_loss(model, b, data, rows, device, HP["huber_delta"], "sum")
        tot += float(l)
        n += m
    return tot / max(n, 1)


def fit_head(mode, H, seed, train_ids, stop_ids, load, rows_train, rows_stop, dims, device, hp=None, log=print):
    """train one head; the stopping set picks the state. -> model (best state loaded), record. ``load(id)`` gives clip_input of a clip (tokens or embedding); rows_*: id -> (frames, y)."""
    hp = dict(hp or HP[mode])
    torch.manual_seed(seed)
    model = make_model(mode, H, dims).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"])
    rows = dict(rows_train)
    rows.update(rows_stop)
    tr = [c for c in train_ids if len(rows[c][1])]
    st = [c for c in stop_ids if len(rows[c][1])]
    if not tr or not st:
        raise SystemExit("REFUSED: %d training clips and %d stopping clips carry labelled frames" % (len(tr), len(st)))
    gen = np.random.default_rng(HP["order_seed_base"] + seed)
    steps_per_eval = int(math.ceil(len(tr) / hp["batch"]))
    t0 = time.time()
    loss_before = eval_loss(model, st, load, rows, device, hp)
    curve = [{"step": 0, "loss": loss_before}]
    best, best_state, best_step, bad, step = float("inf"), None, 0, 0, 0
    stopped = "max_evals"
    for ev in range(hp["max_evals"]):
        model.train()
        order = [tr[i] for i in gen.permutation(len(tr))]
        for b, data in _prefetch([order[i:i + hp["batch"]] for i in range(0, len(order), hp["batch"])], load):
            opt.zero_grad()
            loss, _ = _batch_loss(model, b, data, rows, device, HP["huber_delta"], "mean")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), hp["clip_norm"])
            opt.step()
            step += 1
        v = eval_loss(model, st, load, rows, device, hp)
        curve.append({"step": step, "loss": v})
        log("  eval %d step %d stopping-set loss %.4f (before the first step %.4f)" % (ev + 1, step, v, loss_before))
        if v < best - 1e-9:
            best, best_step, bad = v, step, 0
            best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= hp["patience"]:
                stopped = "patience"
                break
    if best_state is None:
        best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
        best, best_step = float("nan"), step
    model.load_state_dict(best_state)
    model.eval()
    rec = {"loss_before": loss_before, "loss_at_stop": best, "best_step": best_step, "steps": step, "steps_per_eval": steps_per_eval, "stopped_by": stopped, "curve": curve,
           "seconds": round(time.time() - t0, 1), "n_train_clips_with_rows": len(tr), "n_stop_clips_with_rows": len(st), "n_train_rows": int(sum(len(rows[c][1]) for c in tr)),
           "n_stop_rows": int(sum(len(rows[c][1]) for c in st)), "hp": hp}
    return model, rec


# ============================================================================================ the fit subcommand
def _peak():
    out = {}
    try:
        import resource
        out["peak_rss_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20
    except Exception:                                            # noqa: BLE001
        out["peak_rss_gb"] = None
    out["peak_gpu_gb"] = torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else None
    return out


def _read_manifest(path):
    return [json.loads(l)["clip"] for l in open(path, encoding="utf-8") if l.strip()]


def run_fit(a, log=print):
    mode, H = parse_arm(a.arm)
    sys.path.insert(0, a.build_dir)
    import tok_reader as TR
    dev = a.device if (a.device != "cuda" or torch.cuda.is_available()) else "cpu"
    tag = tag_of(a.arm, a.seed, a.shuffle_closing_labels)
    os.makedirs(a.out_dir, exist_ok=True)
    paths = {k: os.path.join(a.out_dir, "tokclose_%s%s" % (tag, s)) for k, s in (("head", ".pt"), ("log", "_log.json"), ("table", "_table.npz"))}
    if any(os.path.exists(p) for p in paths.values()):
        print("REFUSED: output files of %s exist in %s" % (tag, a.out_dir))
        return 2
    hp = dict(HP[mode])
    override = json.loads(os.environ.get("TOKCLOSE_HP_OVERRIDE", "") or "{}")
    hp.update(override)
    meta = json.load(open(a.train_meta, encoding="utf-8"))
    man = _read_manifest(os.path.join(a.train_cache, "train", "manifest.jsonl"))
    lists = CD.reader_lists(man, meta)
    if a.lists_json:
        L = json.load(open(a.lists_json, encoding="utf-8"))
        if L["heldout_ids"] != lists["heldout_ids"] or L["train_ids"] != lists["train_ids"]:
            print("REFUSED: the lists of --lists-json differ from the lists recomputed from the manifest")
            return 2
    eman = _read_manifest(os.path.join(a.eval_cache, "eval", "manifest.jsonl"))
    Jt, Je = np.load(a.join_train, allow_pickle=True), np.load(a.join_eval, allow_pickle=True)
    jt, je = [str(c) for c in Jt["clip"]], [str(c) for c in Je["clip"]]
    Jt_d = {k: Jt[k] for k in ("off", "n", "lead_gap", "lead_valid")}
    Je_d = {k: Je[k] for k in ("off", "n", "lead_gap", "lead_valid")}
    eval_ids = sorted(eman, key=lambda c: je.index(c) if c in je else 10 ** 9)
    held, train = lists["heldout_ids"], lists["train_ids"]
    if set(held) & set(eval_ids) or set(train) & set(eval_ids):
        print("REFUSED: a fit or training clip is an evaluation clip")
        return 2
    if mode == "E":
        sp = CF.split_by_clip(held, HP["stop_frac"], HP["split_seed"])
        fit_ids = [c for c, s in zip(held, sp) if not s]
        stop_ids = [c for c, s in zip(held, sp) if s]
    else:
        fit_ids, stop_ids = train, held
    rows_fit = build_clip_rows(Jt_d, jt, fit_ids, H)
    rows_stop = build_clip_rows(Jt_d, jt, stop_ids, H)
    if a.shuffle_closing_labels:
        rows_fit = permute_labels(rows_fit, fit_ids, HP["shuffle_seed"])
        rows_stop = permute_labels(rows_stop, stop_ids, HP["shuffle_seed"] + 1)
    rd = None
    if mode == "E":
        if not a.reader:
            print("REFUSED: an E arm needs --reader")
            return 2
        rd = TR.TokReader.from_ckpt(a.reader)
        if rd.meta.get("seed") is not None and int(rd.meta["seed"]) != a.seed:
            print("REFUSED: the checkpoint records seed %s, --seed is %d" % (rd.meta["seed"], a.seed))
            return 2
    cache = os.path.join(a.train_cache, "train", "clips")
    pre = {}
    if mode == "E":
        for c in fit_ids + stop_ids:                              # 432 clips: embeddings are small, tokens are not kept
            x = load_tokens(os.path.join(cache, c + ".pt"))
            _check_frames(c, x, rows_fit.get(c) or rows_stop.get(c))
            pre[c] = clip_input("E", x, rd)
        load = lambda c: pre[c]
        d_feat, n_tok, emb = int(x.shape[2]), int(x.shape[1]), int(rd.w["head.frame.weight"].shape[0])
    else:
        first = load_tokens(os.path.join(cache, fit_ids[0] + ".pt"))
        d_feat, n_tok, emb = int(first.shape[2]), int(first.shape[1]), DIMS["emb"]

        def load(c):
            x = load_tokens(os.path.join(cache, c + ".pt"))
            _check_frames(c, x, rows_fit.get(c) or rows_stop.get(c))
            return x
    dims = dict(DIMS, d_feat=d_feat, n_tok=n_tok, emb=emb)
    log("fit %s: %d fit clips, %d stopping clips, device %s" % (tag, len(fit_ids), len(stop_ids), dev))
    model, rec = fit_head(mode, H, a.seed, fit_ids, stop_ids, load, rows_fit, rows_stop, dims, dev, hp, log)
    # evaluation clips: table of the estimate per (clip index, frame) for a later rollout card
    ci_l, fr_l, es_l, cid_l = [], [], [], []
    ecache = os.path.join(a.eval_cache, "eval", "clips")
    jrow = {c: i for i, c in enumerate(je)}
    for n, c in enumerate(eval_ids):
        x = load_tokens(os.path.join(ecache, c + ".pt"))
        est = predict_clip(model, clip_input(mode, x, rd), dev)
        T = len(est)
        ci_l.append(np.full(T, jrow[c])), fr_l.append(np.arange(T)), es_l.append(est), cid_l.append(np.full(T, n))
    np.savez(paths["table"], clip_index=np.concatenate(ci_l), frame=np.concatenate(fr_l), est=np.concatenate(es_l), cid=np.concatenate(cid_l), clips=np.asarray(eval_ids), H=np.int64(H),
             seed=np.int64(a.seed))
    torch.save({"arm": a.arm, "mode": mode, "H": H, "seed": a.seed, "shuffled": bool(a.shuffle_closing_labels), "dims": model.dims, "state_dict": model.state_dict(), "reader_path": a.reader or ""},
               paths["head"])
    ev_set = set(eval_ids)
    full = dict(rec, card=CARD, tag=tag, arm=a.arm, mode=mode, H=H, seed=a.seed, shuffled=bool(a.shuffle_closing_labels), dims=model.dims, device=dev, hp_override=bool(override),
                n_train_clips=len(fit_ids), n_stop_clips=len(stop_ids), n_eval_clips=len(eval_ids), train_ids_sha256=sha_ids(fit_ids), stop_ids_sha256=sha_ids(stop_ids),
                overlap_fit_eval=len(set(fit_ids) & ev_set), overlap_stop_eval=len(set(stop_ids) & ev_set), overlap_fit_stop=len(set(fit_ids) & set(stop_ids)),
                heldout_sha256_recomputed=lists["recomputed_heldout_sha256"], reader=a.reader or "", reader_sha256=sha16(a.reader) if a.reader else "",
                head_sha256_16=sha16(paths["head"]), table_sha256_16=sha16(paths["table"]), **_peak())
    json.dump(full, open(paths["log"], "w", encoding="utf-8", newline="\n"), indent=1, default=float)
    log("fit %s done: loss before %.4f, at stop %.4f (step %d of %d), %d training clips (%d rows), %d stopping clips (%d rows)" %
        (tag, rec["loss_before"], rec["loss_at_stop"], rec["best_step"], rec["steps"], len(fit_ids), rec["n_train_rows"], len(stop_ids), rec["n_stop_rows"]))
    return 0


def _check_frames(c, x, rows):
    f = rows[0] if rows is not None else np.zeros(0, np.int64)
    if len(f) and int(f.max()) >= int(x.shape[0]):
        raise SystemExit("REFUSED: clip %s has %d token frames, the label reaches frame %d" % (c, x.shape[0], int(f.max())))


# ============================================================================================ the score subcommand: statistics
def pooled_r(est, y):
    return CF.pearson(est, y)


def pair_vs_cell(ests, cells, rows_list, B, seed):
    """paired difference of the absolute error, X minus cell, on the same frames; two-level bootstrap of card ADD78 (pair_summary)."""
    allep = sorted(set(np.concatenate([r["ep"] for r in rows_list]).tolist()))
    emap = {e: i for i, e in enumerate(allep)}
    E = len(allep)
    stats = []
    for est, cell, r in zip(ests, cells, rows_list):
        epi = np.asarray([emap[e] for e in r["ep"]], np.int64)
        sx, sc = CF.episode_sums(est, r["y"], epi, E), CF.episode_sums(cell, r["y"], epi, E)
        st = dict(sx)
        st["sae"], st["say"] = sx["sae"], sc["sae"]
        stats.append(st)
    return CF.pair_summary(stats, B, seed)


def summarise_arm(ests, cells, rows_list, B, seed):
    s = CF.cell_summary(ests, rows_list, B, seed)
    c = CF.cell_summary(cells, rows_list, B, seed)
    d = pair_vs_cell(ests, cells, rows_list, B, seed)
    return s, c, d


def name_arm(arm, s, c, d):
    """names of the card: READABLE when the pooled r interval is above 0; BETTER-THAN-OUTPUTS when the pooled paired interval is below 0 and r of X is above r of the cell in every seed."""
    readable = bool(s["r_ci"][0] > 0)
    better = bool(d[2] < 0 and all(rx > rc for rx, rc in zip(s["r_per_seed"], c["r_per_seed"])))
    return ("TOKCLOSE-%s-%s" % (arm, "READABLE" if readable else "NOT-READABLE"), "TOKCLOSE-%s-%s" % (arm, "BETTER-THAN-OUTPUTS" if better else "NOT-BETTER")), readable, better


def verdict_name(betters):
    n = int(sum(betters))
    return ("TOKCLOSE-A-TOKEN-HEAD-READS-MORE" if n >= 1 else "TOKCLOSE-NO-GAIN-FROM-TOKENS"), n


def load_table(path):
    z = np.load(path, allow_pickle=False)
    return {k: np.asarray(z[k]) for k in ("clip_index", "frame", "est")}


def table_lookup(tbl, ci, fr):
    key = np.asarray(ci, np.int64) * 1_000_000 + np.asarray(fr, np.int64)
    tk = tbl["clip_index"].astype(np.int64) * 1_000_000 + tbl["frame"].astype(np.int64)
    o = np.argsort(tk)
    pos = np.minimum(np.searchsorted(tk[o], key), len(tk) - 1)
    hit = tk[o][pos] == key
    return np.where(hit, tbl["est"][o][pos], np.nan)


def row_clip_index(D, rows):
    first = {}
    cid = np.asarray(D["cid"], np.int64)
    for i, c in enumerate(D["clips"]):
        w = np.flatnonzero(cid == i)
        first[str(c)] = int(np.asarray(D["clip_index"])[w[0]]) if len(w) else -1
    return np.asarray([first[str(c)] for c in rows["clip"]], np.int64)


# ============================================================================================ controls
def control_t0(dumps, Je_join, ep_map, est_dir, add78, ks):
    """the frames and the label are those of card ADD78: labels of this tool's function on the join equal the dump rows (1e-6 m/s), the frame sets equal, the saved estimators reproduce r per seed (0.005)."""
    out = {"pass": True, "label": {}, "frames": {}, "estimators": {}}
    j = {k: np.asarray(Je_join[k]) for k in ("off", "n", "lead_gap", "lead_valid")}
    for k in ks:
        D = dumps[k]
        cid, fr = np.asarray(D["cid"], np.int64), np.asarray(D["frame"], np.int64)
        worst, same_mask = 0.0, True
        tool_sets = {H: set() for H in HS}
        for i, c in enumerate(D["clips"]):
            w = np.flatnonzero(cid == i)
            T = len(w)
            ci = int(np.asarray(D["clip_index"])[w[0]])
            pres, _, clo, run = CD.label_clip(j, ci, T, float(D.get("dt_s", CD.DT_S)))
            dp, dr = np.asarray(D["lab_present"], bool)[w], np.asarray(D["lab_run"], np.int64)[w]
            dc = np.asarray(D["lab_closing"], np.float64)[w]
            same_mask = same_mask and bool(np.array_equal(pres, dp) and np.array_equal(run, dr))
            use = pres & (run >= RUN_MIN)
            if use.any():
                worst = max(worst, float(np.abs(clo[use] - dc[use]).max()))
            for H in HS:
                tool_sets[H] |= {(str(c), int(f)) for f in eligible_frames(pres, run, clo, T, H)}
        out["label"]["seed%d" % k] = {"max_abs_diff_ms": worst, "mask_equal": same_mask, "pass": bool(same_mask and worst <= T0_TOL_LABEL)}
        for H in HS:
            rows = CF.build_rows(D, H, ep_map)
            dump_set = {(str(c), int(f)) for c, f in zip(rows["clip"], rows["frame"])}
            ok = tool_sets[H] == dump_set
            out["frames"]["H%d_seed%d" % (H, k)] = {"n_tool": len(tool_sets[H]), "n_dump": len(dump_set), "pass": bool(ok)}
        for H in HS:
            tag, cell = CELL_OF_H[H]
            rows = CF.build_rows(D, H, ep_map)
            E = CL.ClosingEstimator.load(os.path.join(est_dir, "closest_%s_s%d.npz" % (tag, k)))
            r = CF.pearson(E.apply(rows["X"]), rows["y"])
            want = float(add78["cells"][cell]["r_per_seed"][ks.index(k)])
            n_want = add78["cells"][cell].get("n_frames_per_seed")
            n_ok = True if n_want is None else bool(int(n_want[ks.index(k)]) == len(rows["y"]))
            out["estimators"]["%s_seed%d" % (tag, k)] = {"r_tool": r, "r_add78": want, "n_frames": len(rows["y"]), "n_frames_equal_add78": n_ok,
                                                       "pass": bool(np.isfinite(r) and abs(r - want) <= T0_TOL_R and n_ok)}
    out["pass"] = bool(all(v["pass"] for g in ("label", "frames", "estimators") for v in out[g].values()))
    return out


def control_t1(dump0, ep_map, table_path, lo=-T1_BAND, hi=T1_BAND, H=10):
    rows = CF.build_rows(dump0, H, ep_map)
    est = table_lookup(load_table(table_path), row_clip_index(dump0, rows), rows["frame"])
    if not np.isfinite(est).all():
        return {"pass": False, "reason": "the planted table lacks scored rows"}
    r = CF.pearson(est, rows["y"])
    return {"r": r, "band": [lo, hi], "n_frames": int(len(est)), "pass": bool(np.isfinite(r) and lo <= r <= hi)}


def control_t2(logs, lists, ids_eval, expect, allow_override=False, judge_planted_loss=False):
    """logs: tag -> fit log; lists: reader seed -> lists json. Every fit: held-out loss at the stopping point below the loss before the first step; counts equal the split; no evaluation clip; hyperparameters are the fixed ones."""
    out = {"pass": True, "fits": {}}
    for tag, L in logs.items():
        k = int(L["seed"])
        lst = lists[k]
        e = {"loss_before": L["loss_before"], "loss_at_stop": L["loss_at_stop"], "n_train_clips": L["n_train_clips"], "n_stop_clips": L["n_stop_clips"], "n_train_rows": L["n_train_rows"],
             "n_stop_rows": L["n_stop_rows"]}
        loss_ok = bool(np.isfinite(L["loss_at_stop"]) and L["loss_at_stop"] < L["loss_before"])
        held, train = sorted(lst["heldout_ids"]), sorted(lst["train_ids"])
        if L["mode"] == "T":
            cnt = L["n_train_clips"] == expect["train"] and L["n_stop_clips"] == expect["heldout"] and L["train_ids_sha256"] == sha_ids(train) and L["stop_ids_sha256"] == sha_ids(held)
        else:
            cnt = L["n_train_clips"] + L["n_stop_clips"] == expect["heldout"] and L["overlap_fit_stop"] == 0
        cnt = bool(cnt and L["overlap_fit_eval"] == 0 and L["overlap_stop_eval"] == 0 and L["n_eval_clips"] == expect["eval"] and L["n_train_rows"] > 0 and L["n_stop_rows"] > 0)
        hp_ok = bool(allow_override or (not L.get("hp_override") and L["hp"] == HP[L["mode"]]))
        judged = bool(loss_ok or (L.get("shuffled") and not judge_planted_loss))     # the planted fit: its loss pair is reported; judging it is an option (see TOKCLOSE_DEFS.md section 6)
        e.update({"loss_ok": loss_ok, "loss_judged": bool(not (L.get("shuffled") and not judge_planted_loss)), "counts_ok": cnt, "hp_fixed": hp_ok, "pass": bool(judged and cnt and hp_ok)})
        out["fits"][tag] = e
    out["pass"] = bool(out["fits"] and all(v["pass"] for v in out["fits"].values()))
    return out


def t3_max_diff(model, tokens_list, rd, device, n_frames=3, seed=0):
    """largest absolute change of the output at frames up to t when every frame after t is replaced by noise (m/s)."""
    rng = np.random.default_rng(seed)
    worst = 0.0
    for x in tokens_list:
        T = int(x.shape[0])
        ref = predict_clip(model, clip_input(model.mode, x, rd), device)
        lo = model.H + READER_HIST - 1
        for t in sorted(set(np.linspace(lo, T - 2, n_frames).astype(int).tolist())):
            xn = x.clone()
            sd = float(x.float().std())
            xn[t + 1:] = torch.from_numpy((rng.standard_normal(tuple(xn[t + 1:].shape)) * sd).astype(np.float32)).to(torch.float16)
            got = predict_clip(model, clip_input(model.mode, xn, rd), device)
            d = np.abs(got[:t + 1] - ref[:t + 1])
            d = d[np.isfinite(d)]
            if len(d):
                worst = max(worst, float(d.max()))
    return worst


def load_head(path, reader_override=""):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = make_model(ck["mode"], ck["H"], ck["dims"])
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m, ck


def control_t3(head_files, eval_cache, eval_ids, build_dir, n_clips=6, device="cpu"):
    sys.path.insert(0, build_dir)
    out = {"pass": True, "heads": {}, "tol": T3_TOL}
    toks = [load_tokens(os.path.join(eval_cache, "eval", "clips", c + ".pt")) for c in eval_ids[:n_clips]]
    for tag, p in head_files.items():
        m, ck = load_head(p)
        rd = None
        if m.mode == "E":
            import tok_reader as TR
            rd = TR.TokReader.from_ckpt(ck["reader_path"])
        d = t3_max_diff(m, toks, rd, device)
        out["heads"][tag] = {"max_abs_diff_ms": d, "pass": bool(d <= T3_TOL)}
    out["pass"] = bool(out["heads"] and all(v["pass"] for v in out["heads"].values()))
    return out


def control_t4(B=300, seed=11):
    """planted estimates with a known gain and with none: the name rule gives BETTER-THAN-OUTPUTS and NOT-BETTER as planted (and READABLE for both)."""
    rng = np.random.default_rng(seed)
    n_ep, per = 60, 20
    ep = np.asarray(["e%03d" % e for e in np.repeat(np.arange(n_ep), per)])
    y = rng.normal(0, 2.0, n_ep * per)
    rows = [{"y": y, "ep": ep}, {"y": y, "ep": ep}]
    cells = [0.3 * y + rng.normal(0, 1.5, len(y)) for _ in range(2)]
    gain = [0.8 * y + rng.normal(0, 1.0, len(y)) for _ in range(2)]
    none = [0.3 * y + rng.normal(0, 1.5, len(y)) for _ in range(2)]
    res = {}
    for key, est in (("gain", gain), ("none", none)):
        s, c, d = summarise_arm(est, cells, rows, B, 1010)
        (_, nb), rd_, bt = name_arm("E10", s, c, d)
        res[key] = {"name": nb, "better": bt, "readable": rd_, "diff_ci": [d[1], d[2]]}
    ok = bool(res["gain"]["better"] and res["gain"]["name"].endswith("-BETTER-THAN-OUTPUTS") and not res["none"]["better"] and res["none"]["name"].endswith("-NOT-BETTER") and res["gain"]["readable"] and res["none"]["readable"])
    # a gain in one seed only is not enough: seed 1 reads exactly what the cell reads (r not above the cell's)
    s, c, d = summarise_arm([gain[0], cells[1]], cells, rows, B, 1010)
    res["one_seed_only"] = {"better": name_arm("E10", s, c, d)[2]}
    ok = bool(ok and not res["one_seed_only"]["better"])
    res["pass"] = ok
    return res


# ============================================================================================ the score subcommand
def run_score(a, log=print):
    if os.path.exists(a.out_dir):
        print("REFUSED: output folder exists: " + a.out_dir)
        return 2
    ks = [int(x) for x in a.seeds.split(",")]
    ep_map = json.load(open(a.episode_map, encoding="utf-8")) if a.episode_map else None
    dumps = {k: CF.load_dump(os.path.join(a.dump_dir, "reader_s%d_score.npz" % k)) for k in ks}
    lists = {k: json.load(open(os.path.join(a.dump_dir, "reader_s%d_lists.json" % k), encoding="utf-8")) for k in ks}
    add78 = json.load(open(a.add78_result, encoding="utf-8"))
    Je = np.load(a.join_eval, allow_pickle=True)
    Je_d = {k: Je[k] for k in ("off", "n", "lead_gap", "lead_valid")}
    je = [str(c) for c in Je["clip"]]
    eman = _read_manifest(os.path.join(a.eval_cache, "eval", "manifest.jsonl"))
    eval_ids = sorted(eman, key=lambda c: je.index(c) if c in je else 10 ** 9)
    fits = {}
    for arm in ARMS:
        for k in ks:
            fits[(arm, k)] = os.path.join(a.fit_dir, "tokclose_%s" % tag_of(arm, k))
    plant = os.path.join(a.fit_dir, "tokclose_%s" % tag_of("T10", 0, True))
    need = [p + s for p in list(fits.values()) + [plant] for s in (".pt", "_log.json", "_table.npz")]
    miss = [p for p in need if not os.path.exists(p)]
    if miss:
        print("REFUSED: missing fit files: " + ", ".join(os.path.basename(p) for p in miss[:6]))
        return 2
    os.makedirs(a.out_dir)
    controls = {}
    log("== controls (before any arm value) ==")
    controls["t0"] = control_t0(dumps, Je_d, ep_map, a.est_dir, add78, ks)
    controls["t1"] = control_t1(dumps[ks[0]], ep_map, plant + "_table.npz")
    logs = {os.path.basename(p)[len("tokclose_"):]: json.load(open(p + "_log.json", encoding="utf-8")) for p in list(fits.values()) + [plant]}
    expect = {"train": a.expect_train, "heldout": a.expect_heldout, "eval": a.expect_eval}
    controls["t2"] = control_t2(logs, lists, eval_ids, expect, a.allow_hp_override, a.judge_planted_loss)
    heads = {os.path.basename(p)[len("tokclose_"):]: p + ".pt" for p in list(fits.values()) + [plant]}
    controls["t3"] = control_t3(heads, a.eval_cache, eval_ids, a.build_dir, a.t3_clips, a.t3_device)
    controls["t4"] = control_t4()
    allp = all(c["pass"] for c in controls.values())
    for nm, txt in (("t0", "the frames and the label are those of card ADD78; saved estimators reproduce its r per seed"), ("t1", "a head cannot read what is not there (planted arm)"),
                    ("t2", "the fits ran (loss at the stopping point below the loss before, counts equal the split)"), ("t3", "no leak of the future"), ("t4", "planted name rule")):
        log("%s %s: %s" % (nm, txt, "PASS" if controls[nm]["pass"] else "FAIL"))
    json.dump({"controls": controls, "controls_all_pass": bool(allp)}, open(os.path.join(a.out_dir, "TOKCLOSE_CONTROLS.json"), "w", encoding="utf-8", newline="\n"), indent=1, default=float)
    if not allp:
        log("a control did not pass: no name and no arm value is written or printed; the failure is reported with the card and no control is changed inside it.")
        return 1
    res = {"card": CARD, "tier": "T0", "reader_seeds": ks, "bootstrap": {"B": a.boot, "levels": "reader seeds, then evaluation episodes", "interval": "percentile 2.5 / 97.5"},
           "episode_source": "map file" if ep_map else "the clip is its own episode", "arms": {}, "names": {}, "controls_all_pass": True}
    betters, readables = [], []
    for arm in ARMS:
        mode, H = parse_arm(arm)
        rows_list = [CF.build_rows(dumps[k], H, ep_map) for k in ks]
        ests = []
        for k, rows in zip(ks, rows_list):
            e = table_lookup(load_table(fits[(arm, k)] + "_table.npz"), row_clip_index(dumps[k], rows), rows["frame"])
            if not np.isfinite(e).all():
                print("REFUSED: the table of %s seed %d lacks scored rows" % (arm, k))
                return 2
            ests.append(e)
        ctag = CELL_OF_H[H][0]
        cells = [CL.ClosingEstimator.load(os.path.join(a.est_dir, "closest_%s_s%d.npz" % (ctag, k))).apply(rows["X"]) for k, rows in zip(ks, rows_list)]
        s, c, d = summarise_arm(ests, cells, rows_list, a.boot, 1000 + H)
        (n_read, n_better), rd_, bt = name_arm(arm, s, c, d)
        ymean = float(np.mean(np.concatenate([r["y"] for r in rows_list])))
        const = float(np.mean(np.abs(np.concatenate([r["y"] for r in rows_list]) - ymean)))
        res["arms"][arm] = {"H": H, "cell": ctag, "r_pooled": s["r_pooled"], "r_ci": s["r_ci"], "r_per_seed": s["r_per_seed"], "mae": s["mae_est"], "mae_constant_zero": s["mae_zero"],
                            "mae_constant_mean": const, "cell_r_pooled": c["r_pooled"], "cell_r_ci": c["r_ci"], "cell_r_per_seed": c["r_per_seed"], "cell_mae": c["mae_est"],
                            "paired_diff_abs_err_x_minus_cell": d[0], "paired_diff_ci": [d[1], d[2]], "n_frames": s["n_frames"], "n_episodes": s["n_episodes"], "names": [n_read, n_better]}
        res["names"][arm] = [n_read, n_better]
        betters.append(bt), readables.append(rd_)
    top, nb = verdict_name(betters)
    res["verdict"], res["n_better_of_four"], res["n_readable_of_four"] = top, nb, int(sum(readables))
    res["families"] = {"LONGITUDINAL": "not measured by this read (T0 diagnostic of what the features carry)", "LATERAL": "not measured by this read", "TACTICAL": "not measured by this read",
                       "STRATEGIC": "NOT COMPUTED (n = 0; a token probe reads no strategic quantity)"}
    json.dump(res, open(os.path.join(a.out_dir, "TOKCLOSE_RESULT.json"), "w", encoding="utf-8", newline="\n"), indent=1, default=float)
    lines = ["# TOKCLOSE (card ADD81, tier T0: reads on logged frames, no rollout)", "", "All controls t0 to t4 pass. Episodes: %s." % res["episode_source"], "",
             "| arm | H | r pooled | r 95 % | r per seed | MAE | cell | cell r pooled | cell r per seed | cell MAE | paired diff X - cell 95 % | names |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for arm, v in res["arms"].items():
        lines.append("| %s | %d | %+.3f | [%+.3f, %+.3f] | %s | %.3f | %s | %+.3f | %s | %.3f | %+.3f [%+.3f, %+.3f] | %s, %s |" % (
            arm, v["H"], v["r_pooled"], v["r_ci"][0], v["r_ci"][1], ", ".join("%+.3f" % r for r in v["r_per_seed"]), v["mae"], v["cell"], v["cell_r_pooled"],
            ", ".join("%+.3f" % r for r in v["cell_r_per_seed"]), v["cell_mae"], v["paired_diff_abs_err_x_minus_cell"], v["paired_diff_ci"][0], v["paired_diff_ci"][1], v["names"][0], v["names"][1]))
    lines += ["", "Mean absolute error of a constant read (0 m/s / the label mean): " + "; ".join("%s %.3f / %.3f" % (k, v["mae_constant_zero"], v["mae_constant_mean"]) for k, v in res["arms"].items()), "",
              "**%s** (%d of four arms BETTER-THAN-OUTPUTS; %d of four READABLE)." % (top, nb, res["n_readable_of_four"]), "",
              "Families: this read measures none of them. LONGITUDINAL, LATERAL, TACTICAL: not measured by this read. STRATEGIC: NOT COMPUTED (n = 0)."]
    open(os.path.join(a.out_dir, "TOKCLOSE_RESULT.txt"), "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")
    log("== arms ==")
    for arm, v in res["arms"].items():
        log("%s r %+.3f [%+.3f, %+.3f] diff vs %s %+.3f [%+.3f, %+.3f] -> %s %s" % (arm, v["r_pooled"], v["r_ci"][0], v["r_ci"][1], v["cell"], v["paired_diff_abs_err_x_minus_cell"],
                                                                                    v["paired_diff_ci"][0], v["paired_diff_ci"][1], v["names"][0], v["names"][1]))
    log(top)
    return 0


# ============================================================================================ synthetic world (selftest and tests)
def make_world(root, n_train=40, n_held=24, n_eval=20, T=48, d_feat=1024, n_tok=4, W=4, signal=True, seed=0, exclude=(0, 2), reader_seeds=(0, 1)):
    """a tiny cache on disk with the layout of the pod: <cache>/train and <cache>/eval (manifest.jsonl, clips/<id>.pt key pool4x10), joins, train_meta.json, two tiny readers, the score and fit dumps of
    closeread_dump, and the saved estimators of card ADD78. The lead's closing speed is planted in the tokens: channel 0 carries the gap, channel 1 the rate of the gap, both plus noise."""
    rng = np.random.default_rng(seed)
    os.makedirs(root, exist_ok=True)

    def new_ids(n, held, prefix):
        out, i = [], 0
        while len(out) < n:
            c = "%s-%06d" % (prefix, i)
            i += 1
            if CD.heldout_rule(c) == held:
                out.append(c)
        return out
    man = new_ids(n_held, True, "wtr") + new_ids(n_train + 2, False, "wtr")
    eids = new_ids(n_eval, False, "wev")

    def build(ids, cache_dir, split):
        os.makedirs(os.path.join(cache_dir, split, "clips"), exist_ok=True)
        gaps, valids, offs, ns = [], [], [], []
        off = 0
        with open(os.path.join(cache_dir, split, "manifest.jsonl"), "w", encoding="utf-8", newline="\n") as fh:
            for c in ids:
                amp, per, ph, g0 = rng.uniform(1.0, 3.0), rng.uniform(15, 30), rng.uniform(0, 6.28), rng.uniform(20, 40)
                t = np.arange(T)
                rate = amp * np.sin(2 * np.pi * t / per + ph)
                g = np.clip(g0 + np.cumsum(rate) * CD.DT_S, 3.0, 58.0)
                inc = np.gradient(g, CD.DT_S)
                tok = rng.normal(0, 0.3, (T, n_tok, d_feat)).astype(np.float32)
                if signal:
                    tok[:, 0, 0] += (g - 30.0) / 10.0 * 2.0
                    tok[:, 0, 1] += inc / 2.0 * 2.0 + rng.normal(0, 0.3, T)
                torch.save({"pool4x10": torch.from_numpy(tok).to(torch.float16)}, os.path.join(cache_dir, split, "clips", c + ".pt"))
                fh.write(json.dumps({"clip": c}) + "\n")
                gaps.append(g)
                valids.append(np.ones(T))
                offs.append(off)
                ns.append(T)
                off += T
        np.savez(os.path.join(root, "join_%s.npz" % split), clip=np.asarray(ids), off=np.asarray(offs), n=np.asarray(ns), lead_gap=np.concatenate(gaps), lead_valid=np.concatenate(valids))
    build(man, os.path.join(root, "cache"), "train")
    build(eids, os.path.join(root, "evalcache"), "eval")
    ex = set(sorted(man, key=CD.sha12)[exclude[0]:exclude[1]])
    held = sorted(c for c in man if c not in ex and CD.heldout_rule(c))
    train = sorted(c for c in man if c not in ex and not CD.heldout_rule(c))
    meta = {"trainer_argv": "x --exclude %d:%d --batch 64" % exclude, "heldout": {"rule": "int(sha256(id)[:8],16)%10==0", "n_heldout": len(held), "n_trained": len(train), "heldout_sha256": CD.list_sha256(held)}}
    json.dump(meta, open(os.path.join(root, "train_meta.json"), "w", encoding="utf-8"))
    sys.path.insert(0, default_build_dir())
    import tok_reader as TR
    dumpdir, estdir = os.path.join(root, "dump"), os.path.join(root, "est")
    os.makedirs(dumpdir, exist_ok=True)
    os.makedirs(estdir, exist_ok=True)
    Jt, Je = np.load(os.path.join(root, "join_train.npz")), np.load(os.path.join(root, "join_eval.npz"))
    jt, je = [str(c) for c in Jt["clip"]], [str(c) for c in Je["clip"]]
    Jt_d = {k: Jt[k] for k in ("off", "n", "lead_gap", "lead_valid")}
    Je_d = {k: Je[k] for k in ("off", "n", "lead_gap", "lead_valid")}
    cells_r = {"N_H10": [], "L_H25": []}
    ep_map = {c: "ep%03d" % (i // 2) for i, c in enumerate(eids)}
    for k in reader_seeds:
        g = np.random.default_rng(100 + k)
        w = {"tok.0.weight": g.normal(0, 0.3, (128, d_feat)), "tok.0.bias": g.normal(0, 0.1, 128), "tok.2.weight": g.normal(0, 0.3, (W, 128)), "tok.2.bias": g.normal(0, 0.1, W),
             "head.frame.weight": g.normal(0, 0.3, (64, n_tok * W)), "head.frame.bias": g.normal(0, 0.1, 64), "head.mlp.0.weight": g.normal(0, 0.1, (128, 5 * 64)),
             "head.mlp.0.bias": g.normal(0, 0.1, 128), "head.mlp.2.weight": g.normal(0, 0.3, (3, 128)), "head.mlp.2.bias": g.normal(0, 0.1, 3)}
        os.makedirs(os.path.join(root, "reader%d" % k), exist_ok=True)
        ckp = os.path.join(root, "reader%d" % k, "ckpt_last.pt")
        torch.save({"kind": "tokprobe", "input_norm": "ln", "grid": "pool4x10", "hist": 5, "n_tok": n_tok, "arm": "TK%d" % W, "step": 1, "W": W, "seed": k,
                    "state_dict": {a: torch.from_numpy(b.astype(np.float32)) for a, b in w.items()}}, ckp)
        rd = TR.TokReader.from_ckpt(ckp)
        trd = CD.TorchReader(rd, "cpu")
        par = []
        fitd = CD.dump_split(trd, rd, held, os.path.join(root, "cache", "train"), Jt_d, jt, "fit", k, CD.DT_S, par)
        scd = CD.dump_split(trd, rd, eids, os.path.join(root, "evalcache", "eval"), Je_d, je, "score", k, CD.DT_S, par)
        for nm, D in (("fit", fitd), ("score", scd)):
            np.savez(os.path.join(dumpdir, "reader_s%d_%s.npz" % (k, nm)), **D)
        json.dump({"heldout_ids": held, "train_ids": train}, open(os.path.join(dumpdir, "reader_s%d_lists.json" % k), "w", encoding="utf-8"))
        loaded = CF.load_dump(os.path.join(dumpdir, "reader_s%d_score.npz" % k))
        gg = np.random.default_rng(200 + k)
        for H, kind in ((10, "N"), (25, "L")):
            n = 3 * H
            mu, sd = gg.normal(0, 1, n), gg.uniform(0.5, 2, n)
            if kind == "L":
                CL.save_linear(os.path.join(estdir, "closest_L25_s%d.npz" % k), H, mu, sd, gg.normal(0, 0.05, n), 0.1, seed=k)
            else:
                nets = [(gg.normal(0, 0.1, (64, n)), gg.normal(0, 0.1, 64), gg.normal(0, 0.1, (64, 64)), gg.normal(0, 0.1, 64), gg.normal(0, 0.1, (1, 64)), gg.normal(0, 0.1, 1)) for _ in range(3)]
                CL.save_nets(os.path.join(estdir, "closest_N10_s%d.npz" % k), H, mu, sd, nets, seed=k)
            rows = CF.build_rows(loaded, H, ep_map)
            est = CL.ClosingEstimator.load(os.path.join(estdir, "closest_%s%d_s%d.npz" % (kind, H, k))).apply(rows["X"])
            cells_r["N_H10" if H == 10 else "L_H25"].append((CF.pearson(est, rows["y"]), len(rows["y"])))
    add78 = {"cells": {nm: {"r_per_seed": [x[0] for x in v], "n_frames_per_seed": [x[1] for x in v]} for nm, v in cells_r.items()}}
    json.dump(add78, open(os.path.join(root, "add78_result.json"), "w", encoding="utf-8"))
    json.dump(ep_map, open(os.path.join(root, "episode_map.json"), "w", encoding="utf-8"))
    return {"root": root, "held": held, "train": train, "eval": eids, "dims": {"d_feat": d_feat, "n_tok": n_tok, "W": W}, "T": T}


def default_build_dir():
    return os.environ.get("TOKCLOSE_BUILD_DIR") or os.path.join(HERE, "..", "integ10k")


def fit_args(world, arm, seed, shuffle=False, out="fits"):
    r = world["root"]
    return argparse.Namespace(arm=arm, seed=seed, reader=os.path.join(r, "reader%d" % seed, "ckpt_last.pt") if arm[0] == "E" else "", train_cache=os.path.join(r, "cache"),
                              eval_cache=os.path.join(r, "evalcache"), join_train=os.path.join(r, "join_train.npz"), join_eval=os.path.join(r, "join_eval.npz"),
                              train_meta=os.path.join(r, "train_meta.json"), build_dir=default_build_dir(), out_dir=os.path.join(r, out), device="cpu",
                              shuffle_closing_labels=shuffle, lists_json="")


def score_args(world, out="score", fits="fits", **kw):
    r = world["root"]
    d = dict(fit_dir=os.path.join(r, fits), dump_dir=os.path.join(r, "dump"), est_dir=os.path.join(r, "est"), add78_result=os.path.join(r, "add78_result.json"),
             episode_map=os.path.join(r, "episode_map.json"), eval_cache=os.path.join(r, "evalcache"), join_eval=os.path.join(r, "join_eval.npz"),
             build_dir=default_build_dir(), out_dir=os.path.join(r, out), seeds="0,1", boot=200, expect_train=len(world["train"]), expect_heldout=len(world["held"]),
             expect_eval=len(world["eval"]), allow_hp_override=True, judge_planted_loss=False, t3_clips=3, t3_device="cpu")
    d.update(kw)
    return argparse.Namespace(**d)


TEST_HP = {"batch": 8, "lr": 1e-3, "clip_norm": 1.0, "max_evals": 30, "patience": 30}


def run_world_fits(world, hp=None, quiet=True):
    """all eight fits and the planted fit on a synthetic world (the hyperparameter override is the small test one; the log records it)."""
    prev = os.environ.get("TOKCLOSE_HP_OVERRIDE")
    os.environ["TOKCLOSE_HP_OVERRIDE"] = json.dumps(hp or TEST_HP)
    log = (lambda *a: None) if quiet else print
    try:
        for arm in ARMS:
            for k in (0, 1):
                rc = run_fit(fit_args(world, arm, k), log)
                if rc:
                    return rc
        return run_fit(fit_args(world, "T10", 0, True), log)
    finally:
        if prev is None:
            os.environ.pop("TOKCLOSE_HP_OVERRIDE", None)
        else:
            os.environ["TOKCLOSE_HP_OVERRIDE"] = prev


# ============================================================================================ selftest
def selftest():
    import tempfile
    fails = []

    def chk(name, cond):
        print(("PASS " if cond else "FAIL ") + name, flush=True)
        if not cond:
            fails.append(name)
    srcs = [os.path.join(HERE, f) for f in sorted(os.listdir(HERE)) if f.endswith((".py", ".md", ".sh", ".json")) and not f.startswith("tmp")]
    bad = [(os.path.basename(p), x) for p in srcs for x in BANNED if x in open(p, encoding="utf-8", errors="replace").read().lower()]
    chk("no banned word in the %d files of the folder" % len(srcs), not bad)
    root = tempfile.mkdtemp(prefix="tmp_selftest_", dir=HERE)
    world = make_world(root)
    chk("the synthetic world is built (%d held out, %d trained, %d evaluation clips)" % (len(world["held"]), len(world["train"]), len(world["eval"])), len(world["held"]) > 4)
    chk("all nine fits ran", run_world_fits(world) == 0)
    L = {t: json.load(open(os.path.join(root, "fits", "tokclose_%s_log.json" % t), encoding="utf-8")) for t in [tag_of(x, k) for x in ARMS for k in (0, 1)] + [tag_of("T10", 0, True)]}
    real = {t: v for t, v in L.items() if not v["shuffled"]}
    chk("every real fit has a stopping loss below the loss before the first step (%d fits)" % len(real), all(v["loss_at_stop"] < v["loss_before"] for v in real.values()))
    pl = L[tag_of("T10", 0, True)]
    print("INFO planted fit: loss before %.4f, at stop %.4f (reported, not judged)" % (pl["loss_before"], pl["loss_at_stop"]))
    # planted signal read by a T arm; planted arm reads nothing
    Je = np.load(os.path.join(root, "join_eval.npz"))
    D0 = CF.load_dump(os.path.join(root, "dump", "reader_s0_score.npz"))
    em = json.load(open(os.path.join(root, "episode_map.json")))
    rows = CF.build_rows(D0, 10, em)
    ci = row_clip_index(D0, rows)
    r_t = CF.pearson(table_lookup(load_table(os.path.join(root, "fits", "tokclose_T10_s0_table.npz")), ci, rows["frame"]), rows["y"])
    r_sh = CF.pearson(table_lookup(load_table(os.path.join(root, "fits", "tokclose_T10shuf_s0_table.npz")), ci, rows["frame"]), rows["y"])
    chk("a T arm learns the planted closing signal (r %+.2f)" % r_t, r_t > 0.5)
    chk("the arm with shuffled closing labels reads r near 0 (r %+.2f)" % r_sh, abs(r_sh) < 0.2)
    # t3 catches a head that looks one frame ahead
    m, ck = load_head(os.path.join(root, "fits", "tokclose_T10_s0.pt"))
    toks = [load_tokens(os.path.join(root, "evalcache", "eval", "clips", c + ".pt")) for c in world["eval"][:3]]
    d_ok = t3_max_diff(m, toks, None, "cpu")
    m.shift = 1
    d_bad = t3_max_diff(m, toks, None, "cpu")
    chk("t3 reads %.1e on the causal head and %.1e on a head that looks one frame ahead" % (d_ok, d_bad), d_ok <= T3_TOL and d_bad > T3_TOL)
    # t0 catches a shifted label
    dumps = {k: CF.load_dump(os.path.join(root, "dump", "reader_s%d_score.npz" % k)) for k in (0, 1)}
    Je_d = {k: Je[k] for k in ("off", "n", "lead_gap", "lead_valid")}
    add78 = json.load(open(os.path.join(root, "add78_result.json")))
    est_dir = os.path.join(root, "est")
    chk("t0 passes on the dumps of the world", control_t0(dumps, Je_d, em, est_dir, add78, [0, 1])["pass"])
    sh = {k: dict(v) for k, v in dumps.items()}
    sh[0]["lab_closing"] = np.roll(dumps[0]["lab_closing"], 1)
    chk("t0 fails on a label shifted by one row", not control_t0(sh, Je_d, em, est_dir, add78, [0, 1])["pass"])
    # t4 and the end to end score
    chk("t4: the name rule gives the planted names", control_t4()["pass"])
    buf = []
    rc = run_score(score_args(world), log=buf.append)
    res = json.load(open(os.path.join(root, "score", "TOKCLOSE_RESULT.json"))) if rc == 0 else {}
    chk("score: controls pass and names are written (%s)" % res.get("verdict"), rc == 0 and res.get("controls_all_pass") is True and res.get("verdict") in ("TOKCLOSE-A-TOKEN-HEAD-READS-MORE", "TOKCLOSE-NO-GAIN-FROM-TOKENS"))
    chk("T10 is READABLE on the planted signal", res.get("names", {}).get("T10", ["", ""])[0] == "TOKCLOSE-T10-READABLE")
    print("ZZ TOKCLOSE-SELFTEST rc %d (files in %s)" % (0 if not fails else 1, root))
    return 0 if not fails else 1


def main(argv=None):
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--selftest", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    f = sub.add_parser("fit")
    f.add_argument("--arm", required=True, choices=ARMS)
    f.add_argument("--seed", type=int, required=True)
    for n in ("train-cache", "eval-cache", "join-train", "join-eval", "train-meta", "build-dir", "out-dir"):
        f.add_argument("--" + n, required=True)
    f.add_argument("--reader", default="")
    f.add_argument("--lists-json", default="")
    f.add_argument("--device", default="cuda")
    f.add_argument("--shuffle-closing-labels", action="store_true")
    s = sub.add_parser("score")
    for n in ("fit-dir", "dump-dir", "est-dir", "add78-result", "episode-map", "eval-cache", "join-eval", "build-dir", "out-dir"):
        s.add_argument("--" + n, required=True)
    s.add_argument("--seeds", default="0,1")
    s.add_argument("--boot", type=int, default=2000)
    s.add_argument("--expect-train", type=int, default=EXPECT["train"])
    s.add_argument("--expect-heldout", type=int, default=EXPECT["heldout"])
    s.add_argument("--expect-eval", type=int, default=EXPECT["eval"])
    s.add_argument("--allow-hp-override", action="store_true")
    s.add_argument("--judge-planted-loss", action="store_true")
    s.add_argument("--t3-clips", type=int, default=6)
    s.add_argument("--t3-device", default="cpu")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.cmd == "fit":
        return run_fit(a)
    if a.cmd == "score":
        return run_score(a)
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
