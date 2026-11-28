#!/usr/bin/env python3
"""L2 SAFE-HEAD: the cache contract, the sealed index, the head file, and the ONE selection arithmetic (PREREG_SAFEHEAD.md, PK/raw/2026-11-27-safehead/).

Shared by refe/safehead_train.py (the head-only trainer), refe/safehead_gate.py (its launch-gate checks) and raw/2026-11-27-safehead/safehead_read.py (the navtest table read).
CPU only, numpy (+ torch only where a caller needs it).  Fail loud: every contract breach raises ContractError, never a warning.

THE CACHE CONTRACT (the data agent writes it; F:/safehead_cache/ + an index CSV):
  index CSV columns  token, log, split, file [, row] [, sha256]      split in {train, heldout} (navtrain) | {navtest} (eval only, no labels)
                     the data agent's layout (DATA_CONTRACT.md v1): one npz per LOG, arrays stacked over its tokens, rows found by the file's `token` array
                     `file` is relative to the index's directory (or absolute); `row` (optional) indexes a stacked file whose arrays carry a leading token axis.
  per token          s float16/32 [64,256] (the exact input of score_head), logits float32 [64,6] (C1's own head on s), pick int (C1's planner pick),
                     props float32 [64,T,D], and for train/heldout: tr_nc tr_dac tr_ddc tr_ttc tr_ep tr_c tr_pdms float32 [64] (NAVSIM v1.1 PDM, LQR-tracked)
THE SEALED INDEX: the index plus a sha256 per referenced file, written by `seal` into the RUN directory (never into the cache).  The trainer refuses an unsealed index and
re-hashes every file it reads against it, so the launch gate's sha256 of the sealed index binds the CONTENT of the cache.

    python safehead_data.py check --index F:/safehead_cache/index.csv [--heldout-logs ../splits/sft_heldout_logs.txt]
    python safehead_data.py seal  --index F:/safehead_cache/index.csv --out <run>/index_sealed.csv
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

SCHEMA_HEAD = "safehead.head/1"
N_PROP, D_S, N_OUT = 64, 256, 6
# Component order of the six outputs: planner.py:547-549 / train.py:269 ScorerBank.COMPONENTS = (collision, drivable_area, progress, ttc, comfort, ddc)
ROW_NAMES = ("nc", "dac", "ep", "ttc", "c", "ddc")
SAFETY_ROWS = (0, 1, 3, 5)                       # NC, DAC, TTC, DDC (the card's arm SH trains exactly these)
FROZEN_ROWS = (2, 4)                             # EP, C
LABELS = ("tr_nc", "tr_dac", "tr_ddc", "tr_ttc", "tr_ep", "tr_c", "tr_pdms")
ROW_LABEL = {0: "tr_nc", 1: "tr_dac", 3: "tr_ttc", 5: "tr_ddc"}
# L2-NEXT (PREREG_EPHEAD): row 2 (EP) may be a TRAINED row, with the SOFT label tr_ep (the cache's EP is a rebuilt metric cache; see PREREG_EPHEAD 2.3).  ROW_LABEL above is
# unchanged (the SAFE-HEAD card and its read use it); ROW_LABEL_ALL names every output's label (also used by the gate's "a frozen row enters the loss" arm).
ROW_LABEL_ALL = {0: "tr_nc", 1: "tr_dac", 2: "tr_ep", 3: "tr_ttc", 4: "tr_c", 5: "tr_ddc"}
TRAINABLE_ROWS = (0, 1, 2, 3, 5)                 # SAFETY_ROWS + EP; row 4 (C) is never a parameter in any arm
SCHEMA_HEAD_MLP = "safehead.head/2"              # a head whose trained rows carry a residual MLP (SH-MLP, card 8.1); the linear W, b are C1's, bit for bit
MLP_IN = 256
LABEL_VALUES = {"tr_nc": (0.0, 0.5, 1.0), "tr_dac": (0.0, 1.0), "tr_ttc": (0.0, 1.0), "tr_ddc": (0.0, 0.5, 1.0), "tr_c": (0.0, 1.0)}
SPLITS_NAVTRAIN = ("train", "heldout")
PDMS_TOL = 1e-6                                  # tr_pdms == NC*DAC*(5EP+5TTC+2C)/12 (NAVSIM v1.1; DDC does not enter: MEASURED on the C1 table, max diff 0)


class ContractError(RuntimeError):
    pass


def sha256_file(p, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# ------------------------------------------------------------------------------------------ the selection arithmetic (ONE definition)
def agg32(L) -> np.ndarray:
    """The planner's navsim_v1 aggregate (planner.py:558-561: NC x DAC x (5 EP + 5 TTC + 2 C) / 12, DDC does not enter) in FLOAT32 numpy, in the exact form
    PREREG_P5C_NEXT AMENDMENT_1 validated against C1's dump (build_rule_seams.py:17-20 agg32(L, 1.0, 5.0): 36,438 / 36,438 tokens; float64 flips near-ties)."""
    L = np.asarray(L, np.float32)
    p = 1.0 / (1.0 + np.exp(-L))
    return (p[..., 0] * p[..., 1]) ** 1.0 * (5 * p[..., 2] + 5.0 * p[..., 3] + 2 * p[..., 4]) / (7 + 5.0)


def select(L) -> np.ndarray:
    return agg32(L).argmax(-1)


def delta_logits(base_logits, s, W, b, W0, b0, rows=SAFETY_ROWS) -> np.ndarray:
    """The read's logits of a trained head (PREREG_SAFEHEAD 4.1): C1's OWN banked float32 logits plus the head's CHANGE applied to the cached s,
    base + s @ (W - W0)[rows].T + (b - b0)[rows], on the trained rows only.  The untrained head (W == W0) returns `base` BIT FOR BIT (CTRL-ID by construction);
    rows outside `rows` are never touched.  The deployed head s @ W.T + b differs from this by float rounding only (reported, M5)."""
    base = np.asarray(base_logits, np.float32)
    s = np.asarray(s, np.float32)
    rows = list(rows)
    dW = (np.asarray(W, np.float32) - np.asarray(W0, np.float32))[rows]
    db = (np.asarray(b, np.float32) - np.asarray(b0, np.float32))[rows]
    out = base.copy()
    out[..., rows] = base[..., rows] + (s @ dW.T + db)
    return out


def direct_logits(s, W, b) -> np.ndarray:
    return np.asarray(s, np.float32) @ np.asarray(W, np.float32).T + np.asarray(b, np.float32)


def rank_auc(score, label) -> float:
    """AUC of `score` for label == 1 vs label == 0 (0.5 labels dropped); NaN when one class is empty.  Average ranks on ties."""
    score, label = np.asarray(score, np.float64).ravel(), np.asarray(label, np.float64).ravel()
    m = (label == 0.0) | (label == 1.0)
    score, label = score[m], label[m]
    n1, n0 = int((label == 1).sum()), int((label == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score))
    ss = score[order]
    i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[label == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


# ------------------------------------------------------------------------------------------ the index
def read_index(path) -> list:
    path = Path(path)
    if not path.is_file():
        raise ContractError(f"index {path} does not exist")
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ContractError(f"index {path} has no rows")
    need = {"token", "log", "split", "file"}
    miss = need - set(rows[0])
    if miss:
        raise ContractError(f"index {path} lacks columns {sorted(miss)} (contract: token, log, split, file [, row] [, sha256])")
    toks = [r["token"] for r in rows]
    if len(set(toks)) != len(toks):
        raise ContractError(f"index {path}: duplicate tokens")
    for r in rows:
        r["_path"] = str((path.parent / r["file"]).resolve()) if not os.path.isabs(r["file"]) else r["file"]
    return rows


def seal_index(index, out) -> dict:
    """Write the SEALED index (same rows + a sha256 per referenced file) to `out`.  Refuses to write inside the cache directory."""
    index, out = Path(index), Path(out)
    cache_dir = Path(os.path.normcase(str(index.resolve().parent)))
    out_dir = Path(os.path.normcase(str(out.resolve().parent)))
    if out_dir == cache_dir or cache_dir in out_dir.parents:
        raise ContractError(f"refusing to write the sealed index inside the cache directory ({out}): the cache is the data agent's")
    rows = read_index(index)
    shas: dict = {}
    for r in rows:
        if r["_path"] not in shas:
            if not os.path.isfile(r["_path"]):
                raise ContractError(f"index row {r['token']}: file {r['_path']} missing")
            shas[r["_path"]] = sha256_file(r["_path"])
        if r.get("sha256") and r["sha256"] != shas[r["_path"]]:
            raise ContractError(f"index row {r['token']}: its own sha256 column disagrees with the file")
    cols = ["token", "log", "split", "file"] + (["row"] if "row" in rows[0] else []) + ["sha256"]
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(cols)
        for r in rows:
            w.writerow([r["_path"] if c == "file" else (shas[r["_path"]] if c == "sha256" else r[c]) for c in cols])
    return {"rows": len(rows), "files": len(shas), "sealed_index": str(out), "sealed_sha256": sha256_file(out), "source_index_sha256": sha256_file(index)}


# ------------------------------------------------------------------------------------------ loading
_KEY_ALIASES = {"s": ("s", "score_feat", "s_feat"), "logits": ("logits", "score", "lg"), "pick": ("pick", "k_ship"), "props": ("props", "proposals", "traj")}


def _get(z, key):
    for k in _KEY_ALIASES.get(key, (key,)):
        if k in z.files:
            return z[k]
    raise ContractError(f"file lacks {key!r} (tried {_KEY_ALIASES.get(key, (key,))}; has {sorted(z.files)})")


def load_cache(index, *, splits, require_sealed: bool = True, labels: bool = True, verify_sha: bool = True) -> dict:
    """-> dict of arrays over the rows of `index` whose split is in `splits`, in index order.  Validates the contract (fail loud).
    `require_sealed`: every row carries a sha256 and (verify_sha) every file is re-hashed against it."""
    rows = [r for r in read_index(index) if r["split"] in splits]
    if not rows:
        raise ContractError(f"index {index}: no rows with split in {splits}")
    bad = sorted({r["split"] for r in read_index(index)} - set(SPLITS_NAVTRAIN) - {"navtest"})
    if bad:
        raise ContractError(f"index {index}: unknown split value(s) {bad}")
    if require_sealed and any(not r.get("sha256") for r in rows):
        raise ContractError(f"index {index} is NOT SEALED (a row has no sha256): run `safehead_data.py seal` into the run directory")
    files: dict = {}
    memo: dict = {"p": None, "a": {}}
    hashed: dict = {}
    tokrow: dict = {}
    S, LG, PK, PR = [], [], [], []
    Y = {k: [] for k in LABELS} if labels else {}
    for r in rows:
        p = r["_path"]
        if p not in files:
            if not os.path.isfile(p):
                raise ContractError(f"token {r['token']}: file {p} missing")
            if require_sealed and verify_sha:
                hashed[p] = sha256_file(p)
            files[p] = np.load(p, allow_pickle=False)
        if require_sealed and verify_sha and hashed[p] != r["sha256"]:
            raise ContractError(f"token {r['token']}: {p} sha256 {hashed[p][:12]} != sealed {r['sha256'][:12]} (the cache changed after sealing)")
        z = files[p]
        if r.get("row") not in (None, ""):
            i = int(r["row"])
        elif "token" in z.files and np.asarray(z["token"]).ndim == 1:            # the data agent's layout: one npz per LOG, rows found by its `token` array
            if p not in tokrow:
                tokrow[p] = {str(t): j for j, t in enumerate(np.asarray(z["token"]).astype(str))}
                for key, want in (("log", r["log"]), ("split", r["split"])):
                    if key in z.files and np.asarray(z[key]).ndim == 0 and str(z[key]) != want:
                        raise ContractError(f"{p}: file {key} {str(z[key])!r} != index {want!r}")
            if r["token"] not in tokrow[p]:
                raise ContractError(f"token {r['token']} is not in {p}'s token array")
            i = tokrow[p][r["token"]]
        else:
            i = None

        def take(key):
            # MEMORY FIX (run1 launch agent): read each array of a file ONCE (not once per row) and COPY the row, so a row never pins
            # its whole-file base array (the unpatched view kept ~N_rows x file_size alive: tens of GB on the real cache).
            if memo["p"] != p:
                memo["p"], memo["a"] = p, {}
            if key not in memo["a"]:
                memo["a"][key] = _get(z, key)
            v = memo["a"][key]
            return np.array(v[i]) if i is not None else v
        one = {"s": np.asarray(take("s"), np.float32), "logits": np.asarray(take("logits"), np.float32), "props": np.asarray(take("props"), np.float32)}
        for k in Y:
            one[k] = np.asarray(take(k), np.float32)
        want = {"s": (N_PROP, D_S), "logits": (N_PROP, N_OUT), **{k: (N_PROP,) for k in Y}}
        for k, shp in want.items():
            if one[k].shape != shp:
                raise ContractError(f"token {r['token']}: {k} has shape {one[k].shape}, contract {shp}")
        if one["props"].ndim != 3 or one["props"].shape[0] != N_PROP or (PR and one["props"].shape != PR[0].shape):
            raise ContractError(f"token {r['token']}: props has shape {one['props'].shape}, contract [{N_PROP},T,D] (one T,D for the cache)")
        S.append(one["s"])
        LG.append(one["logits"])
        PK.append(int(np.asarray(take("pick"))))
        PR.append(one["props"])
        for k in Y:
            Y[k].append(one[k])
    out = {"token": np.array([r["token"] for r in rows]), "log": np.array([r["log"] for r in rows]), "split": np.array([r["split"] for r in rows]),
           "s": np.stack(S), "logits": np.stack(LG), "pick": np.array(PK, np.int64), "props": np.stack(PR),
           "files_sha256": {p: hashed.get(p) for p in files}}
    for k in Y:
        out[k] = np.stack(Y[k])
    validate(out, labels=labels)
    return out


def validate(c: dict, labels: bool = True) -> None:
    n = len(c["token"])
    if c["s"].shape != (n, N_PROP, D_S):
        raise ContractError(f"s has shape {c['s'].shape}, contract [N,{N_PROP},{D_S}]")
    if c["logits"].shape != (n, N_PROP, N_OUT):
        raise ContractError(f"logits has shape {c['logits'].shape}, contract [N,{N_PROP},{N_OUT}]")
    if c["props"].ndim != 4 or c["props"].shape[:2] != (n, N_PROP):
        raise ContractError(f"props has shape {c['props'].shape}, contract [N,{N_PROP},T,D]")
    for k in ("s", "logits", "props"):
        if not np.isfinite(c[k]).all():
            raise ContractError(f"{k} carries non-finite values")
    if ((c["pick"] < 0) | (c["pick"] >= N_PROP)).any():
        raise ContractError("pick out of range")
    sel = select(c["logits"])
    if not (sel == c["pick"]).all():
        raise ContractError(f"CTRL-S (float32): argmax agg32(logits) != pick on {int((sel != c['pick']).sum())} / {n} tokens: the cached pick is not C1's planner rule on the cached logits")
    if not labels:
        return
    for k in LABELS:
        if c[k].shape != (n, N_PROP):
            raise ContractError(f"{k} has shape {c[k].shape}, contract [N,{N_PROP}]")
        if not np.isfinite(c[k]).all():
            raise ContractError(f"{k} carries non-finite values")
    for k, vals in LABEL_VALUES.items():
        u = np.unique(c[k])
        if not np.isin(u, vals).all():
            raise ContractError(f"{k} carries values {u[~np.isin(u, vals)][:5]} outside {vals}")
    # EP / PDMS of the cache are a REBUILT metric cache (DATA_CONTRACT.md caveats: EP differs from the harness on 6/57 navtest picks, max 0.0108): the card uses
    # NO cache EP / PDMS label for any fitted or selected quantity (PREREG_SAFEHEAD 2.3, MM 2026-11-26).  The PDMS identity is therefore REPORTED, never a refusal.
    pd = c["tr_nc"] * c["tr_dac"] * (5 * c["tr_ep"] + 5 * c["tr_ttc"] + 2 * c["tr_c"]) / 12.0
    c["diag"] = {"pdms_identity_max_abs": float(np.abs(pd - c["tr_pdms"]).max()), "pdms_identity_tol": PDMS_TOL}


def read_log_list(path) -> list:
    return [ln.strip() for ln in open(path, encoding="utf-8") if ln.strip() and not ln.startswith("#")]


def leak_problems(c: dict, heldout_logs, navtest_logs, navtest_tokens=()) -> list:
    """CTRL-LEAK (card 6): train and held-out disjoint by log; every held-out log in splits/sft_heldout_logs.txt; no train/held-out log is a navtest log;
    no train/held-out token is a navtest token; no train log is in the SFT held-out list."""
    out = []
    tr = set(c["log"][c["split"] == "train"])
    ho = set(c["log"][c["split"] == "heldout"])
    H, NT = set(heldout_logs), set(navtest_logs)
    if tr & ho:
        out.append(f"{len(tr & ho)} log(s) in BOTH train and heldout (first {sorted(tr & ho)[0]})")
    if ho - H:
        out.append(f"{len(ho - H)} held-out log(s) not in sft_heldout_logs (first {sorted(ho - H)[0]})")
    if tr & H:
        out.append(f"{len(tr & H)} train log(s) are SFT held-out logs (first {sorted(tr & H)[0]})")
    if (tr | ho) & NT:
        out.append(f"{len((tr | ho) & NT)} navtrain log(s) are NAVTEST logs (first {sorted((tr | ho) & NT)[0]})")
    if not NT:
        out.append("the navtest log list is EMPTY: the leak check cannot run (absence at one location is not absence)")
    nt_tok = set(navtest_tokens)
    if nt_tok & set(c["token"]):
        out.append(f"{len(nt_tok & set(c['token']))} cached navtrain token(s) are navtest tokens")
    return out


# ------------------------------------------------------------------------------------------ the head file (npz, no pickle)
MLP_KEYS = ("mlp_w1", "mlp_b1", "mlp_w2", "mlp_b2")


def save_head(path, *, W, b, W0, b0, meta: dict, mlp=None) -> str:
    """`mlp` (SH-MLP only): {mlp_w1 [R,H,256], mlp_b1 [R,H], mlp_w2 [R,H], mlp_b2 [R]} over the trained rows in `meta["rows"]` order; the file is then schema /2.
    Without it the file is schema /1 and byte-identical to what SAFE-HEAD wrote."""
    path = Path(path)
    tmp = path.with_suffix(".tmp.npz")
    extra = {} if mlp is None else {k: np.asarray(mlp[k], np.float32) for k in MLP_KEYS}
    schema = SCHEMA_HEAD if mlp is None else SCHEMA_HEAD_MLP
    np.savez(tmp, weight=np.asarray(W, np.float32), bias=np.asarray(b, np.float32), init_weight=np.asarray(W0, np.float32),
             init_bias=np.asarray(b0, np.float32), meta=np.array(json.dumps({"schema": schema, **meta}, sort_keys=True)), **extra)
    os.replace(tmp, path)
    return sha256_file(path)


def load_head(path) -> dict:
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    if meta.get("schema") not in (SCHEMA_HEAD, SCHEMA_HEAD_MLP):
        raise ContractError(f"{path}: not a {SCHEMA_HEAD} / {SCHEMA_HEAD_MLP} file")
    h = {"W": z["weight"], "b": z["bias"], "W0": z["init_weight"], "b0": z["init_bias"], "meta": meta, "mlp": None}
    if h["W"].shape != (N_OUT, D_S) or h["b"].shape != (N_OUT,):
        raise ContractError(f"{path}: head shape {h['W'].shape}/{h['b'].shape}")
    rows = list(meta.get("rows", SAFETY_ROWS))
    if meta["schema"] == SCHEMA_HEAD_MLP:
        h["mlp"] = {k: z[k] for k in MLP_KEYS}
        R, H = len(rows), int(meta["mlp_hidden"])
        want = {"mlp_w1": (R, H, MLP_IN), "mlp_b1": (R, H), "mlp_w2": (R, H), "mlp_b2": (R,)}
        bad = {k: h["mlp"][k].shape for k, v in want.items() if h["mlp"][k].shape != v}
        if bad:
            raise ContractError(f"{path}: mlp array shape(s) {bad}, expected {want}")
        if not (np.array_equal(h["W"], h["W0"]) and np.array_equal(h["b"], h["b0"])):
            raise ContractError(f"{path}: an SH-MLP head's LINEAR part must be C1's, bit for bit (the residual MLP is the only trained part)")
        return h
    frozen = [r for r in range(N_OUT) if r not in rows]
    if not (np.array_equal(h["W"][frozen], h["W0"][frozen]) and np.array_equal(h["b"][frozen], h["b0"][frozen])):
        raise ContractError(f"{path}: an untrained row ({[ROW_NAMES[r] for r in frozen]}) differs from C1's init")
    return h


def mlp_forward(s, mlp) -> np.ndarray:
    """The residual MLPs on s: z_r = W2_r . gelu(W1_r s + b1_r) + b2_r for each trained row r (torch exact-erf GELU, float32, CPU).  [..., 256] -> [..., R].
    ONE definition, shared by the read and the gate (the trainer's own forward is the same arithmetic in torch)."""
    import torch
    s = torch.from_numpy(np.ascontiguousarray(np.asarray(s, np.float32)))
    out = []
    with torch.no_grad():
        for i in range(mlp["mlp_w1"].shape[0]):
            h = torch.nn.functional.gelu(s @ torch.from_numpy(np.asarray(mlp["mlp_w1"][i], np.float32)).T + torch.from_numpy(np.asarray(mlp["mlp_b1"][i], np.float32)))
            out.append(h @ torch.from_numpy(np.asarray(mlp["mlp_w2"][i], np.float32)) + torch.from_numpy(np.asarray(mlp["mlp_b2"][i:i + 1], np.float32)))
    return torch.stack(out, -1).numpy()


def head_forward(s, h) -> np.ndarray:
    """The deployed head on s: the linear part, plus the residual MLPs on the trained rows (schema /2).  [..., 256] -> [..., 6]."""
    out = direct_logits(s, h["W"], h["b"])
    if h.get("mlp") is not None:
        rows = list(h["meta"]["rows"])
        out = out.copy()
        out[..., rows] = out[..., rows] + mlp_forward(s, h["mlp"])
    return out


def head_logits(base_logits, s, h, rows=None) -> np.ndarray:
    """The READ's logits of ANY saved head (card 4.1, generalised): C1's banked float32 logits plus the head's CHANGE on the trained rows.  Schema /1: delta_logits
    (the SAFE-HEAD form, unchanged).  Schema /2: base + the residual MLP on s (zero at step 0 by construction: the last layer is zero-initialised, so the untrained
    head returns `base` BIT FOR BIT).  `rows` overrides the trained rows (the gate's row-shift arm only)."""
    rows = list(h["meta"]["rows"]) if rows is None else list(rows)
    if h.get("mlp") is None:
        return delta_logits(base_logits, s, h["W"], h["b"], h["W0"], h["b0"], rows)
    base = np.asarray(base_logits, np.float32)
    out = base.copy()
    out[..., rows] = base[..., rows] + mlp_forward(s, h["mlp"])
    return out


def cv_folds(logs, k: int = 5) -> dict:
    """The fixed 5-fold log-level split of PREREG_EPHEAD 4.3: logs sorted by NAME, fold = index mod k.  -> {log: fold}."""
    u = sorted(set(map(str, logs)))
    return {lg: i % k for i, lg in enumerate(u)}


def head_arrays_sha256(h: dict) -> str:
    """sha256 of the head's arrays.  Schema /1: W, b (unchanged).  Schema /2 also hashes the residual MLP arrays (W, b are C1's there: they alone would be blind)."""
    parts = [np.ascontiguousarray(h[k], np.float32).tobytes() for k in ("W", "b")]
    if h.get("mlp") is not None:
        parts += [np.ascontiguousarray(h["mlp"][k], np.float32).tobytes() for k in MLP_KEYS]
    return sha256_bytes(b"".join(parts))


def c1_head_from_checkpoint(path) -> tuple:
    """(W [6,256], b [6]) float32 from C1's model_final.pt ({"model": state_dict, "meta"} or a bare state dict; ckpt_io.py:11, :166)."""
    import torch
    try:
        sd = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except Exception:                                                       # noqa: BLE001 -- an old-format file cannot be mmapped
        sd = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    for k in ("score_head.weight", "score_head.bias"):
        if k not in sd:
            raise ContractError(f"{path}: no {k} in the state dict")
    W = sd["score_head.weight"].detach().float().cpu().numpy().copy()
    b = sd["score_head.bias"].detach().float().cpu().numpy().copy()
    if W.shape != (N_OUT, D_S):
        raise ContractError(f"{path}: score_head.weight {W.shape} (a 7-output SFT-4 head is not C1)")
    return W, b


def export_full(c1_ckpt, head_path, out) -> dict:
    """ON ADOPT ONLY (card 5): C1's model_final.pt with score_head replaced by the trained head; asserts every OTHER tensor is bit-identical to C1."""
    import torch
    h = load_head(head_path)
    sd = torch.load(c1_ckpt, map_location="cpu", weights_only=False)
    model = sd["model"] if isinstance(sd, dict) and "model" in sd else sd
    before = {k: v.clone() for k, v in model.items() if k not in ("score_head.weight", "score_head.bias")}
    model["score_head.weight"] = torch.from_numpy(h["W"]).to(model["score_head.weight"].dtype)
    model["score_head.bias"] = torch.from_numpy(h["b"]).to(model["score_head.bias"].dtype)
    moved = [k for k, v in before.items() if not torch.equal(v, model[k])]
    if moved:
        raise ContractError(f"export: non-head tensors moved: {moved[:5]}")
    if isinstance(sd, dict) and "model" in sd:
        sd.setdefault("meta", {})["safehead"] = {"head": str(head_path), "head_sha256": sha256_file(head_path), "rows": h["meta"].get("rows")}
    torch.save(sd, out)
    return {"out": str(out), "sha256": sha256_file(out), "n_other_tensors_identical": len(before)}


# ------------------------------------------------------------------------------------------ CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("--index", required=True)
    c.add_argument("--heldout-logs", default="")
    c.add_argument("--navtest-logs", default="")
    s = sub.add_parser("seal")
    s.add_argument("--index", required=True)
    s.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "seal":
        print(json.dumps(seal_index(a.index, a.out), indent=1))
        print("ZZSH_SEAL_OK")
        return 0
    rows = read_index(a.index)
    sp = sorted({r["split"] for r in rows})
    rep = {"index": a.index, "rows": len(rows), "splits": {k: sum(r["split"] == k for r in rows) for k in sp},
           "logs": {k: len({r["log"] for r in rows if r["split"] == k}) for k in sp}, "sealed": all(r.get("sha256") for r in rows)}
    nav = [k for k in sp if k in SPLITS_NAVTRAIN]
    if nav:
        cc = load_cache(a.index, splits=tuple(nav), require_sealed=False)
        rep["validate_navtrain"] = "PASS"
        if a.heldout_logs:
            rep["leak"] = leak_problems(cc, read_log_list(a.heldout_logs), read_log_list(a.navtest_logs) if a.navtest_logs else [])
    if "navtest" in sp:
        load_cache(a.index, splits=("navtest",), require_sealed=False, labels=False)
        rep["validate_navtest"] = "PASS"
    print(json.dumps(rep, indent=1))
    print("ZZSH_CHECK_OK" if not rep.get("leak") else "ZZSH_CHECK_LEAK")
    return 0 if not rep.get("leak") else 3


if __name__ == "__main__":
    sys.exit(main())
