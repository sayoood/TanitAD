"""q5_c2_planted.py -- AMENDMENT 7 control C2 (planted positive).  CPU only.  s0p_fit.py is called, never edited; nothing is written into run/assembled or run/work.

  python q5_c2_planted.py make --base B [--samples S1,S2]     planted COPIES under B/c2/assembled/ (only v40 differs; every other file is an os.link hardlink: zero extra bytes)
  python q5_c2_planted.py prep --base B [--samples S1,S2]     PCA + projections of the PLANTED v40 only (s0p_fit.fit_pca / project, the same files `s0p_fit.py prep` writes for v40)
  python q5_c2_planted.py fit  --base B --sample S1 --model linear|mlp     s0p_fit.py `fit --arm V40+E` on the planted dirs -> B/c2/work/<S>_416/fit_V40+E_<model>.{json,npy}
  python q5_c2_planted.py read --base B                       -> B/c2/RESULT_C2.json {"c2_auroc", "c2_outcome", detail}

CHOICES THE AMENDMENT LEAVES OPEN (decided here, written before any planted score exists; see README_C2.md):
  cell      4x10 grid cell (row 2, col 5), row-major token index 25 (CELL_ROW/CELL_COL).
  arrays    v40 ONLY.  v160 and mean are untouched (the control tests V40@416+E).
  norm      median over ALL tokens of S1's TRAIN v40@416 (decoded fp8 -> fp32) of the L2 norm; ONE number used for S1, S2 and eval.
  direction numpy RandomState(5).standard_normal(1024), unit length, times the norm; added in fp32, re-quantised to fp8 e4m3fn.
  positives y == 1 (train and eval).
DISK: F: must keep >= FLOOR_GB (25) free; make/prep/fit refuse below it.  No full copy is ever made: if os.link fails, `make` raises.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD = os.path.dirname(HERE)
sys.path.insert(0, BUILD)
S0P = os.environ.get("Q5_S0P", "E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/stopperc/s0p")
PY_DEFAULT = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
SEED, CELL_ROW, CELL_COL, GRID_W = 5, 2, 5, 10
FLOOR_GB = float(os.environ.get("C2_FLOOR_GB", "25"))
TRAIN_FILES = ("E", "y", "clip", "ws", "eid", "v160", "mean")      # hardlinked
SMALL_LINKS = ("sample.json", "MANIFEST.sha256")                     # hardlinked if present (MANIFEST is stale for v40 and says so in meta.json)


def sha256_file(p, chunk=1 << 24):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                return h.hexdigest()
            h.update(b)


def free_gb(path):
    p = path
    while not os.path.exists(p):
        p = os.path.dirname(p) or "."
    return shutil.disk_usage(p).free / 1e9


def need_free(path, extra_bytes, floor_gb=None):
    floor = FLOOR_GB if floor_gb is None else floor_gb
    f = free_gb(path)
    if f - extra_bytes / 1e9 < floor:
        raise RuntimeError(f"disk: {f:.1f} GB free at {path}, this step writes ~{extra_bytes / 1e9:.2f} GB, floor {floor} GB")


def _fp8():
    sys.path.insert(0, S0P)
    import s0p_lib as L
    import q5_now_encode as Q
    return L.decode_fp8, Q.to_fp8


def direction(dim=1024, seed=SEED):
    d = np.random.RandomState(seed).standard_normal(dim)
    return d / np.linalg.norm(d)


def median_token_norm(v40, chunk=128):
    dec, _ = _fp8()
    out = []
    for i in range(0, len(v40), chunk):
        x = dec(np.ascontiguousarray(v40[i:i + chunk])).astype(np.float64)
        out.append(np.linalg.norm(x, axis=-1).ravel())
    return float(np.median(np.concatenate(out)))


def plant_array(v40, y, vec, cell, plant_on="pos", chunk=256):
    """Yield (start, planted chunk uint8) with `vec` added (fp32, then fp8) to token `cell` of the selected windows.  plant_on 'pos' (the control) or 'all' (the deliberate regression)."""
    dec, to8 = _fp8()
    import torch
    sel_all = np.asarray(y).astype(bool) if plant_on == "pos" else np.ones(len(y), bool)
    for i in range(0, len(v40), chunk):
        a = np.array(v40[i:i + chunk])                     # a writable copy
        s = np.nonzero(sel_all[i:i + chunk])[0]
        if len(s):
            x = dec(a[s, cell]) + vec.astype(np.float32)
            a[s, cell] = to8(torch.from_numpy(x.astype(np.float32)))
        yield i, a, int(len(s))


def write_npy_stream(path, shape, dtype, chunks):
    with open(path, "wb") as f:
        np.lib.format.write_array_header_1_0(f, {"descr": np.lib.format.dtype_to_descr(np.dtype(dtype)), "fortran_order": False, "shape": tuple(shape)})
        for a in chunks:
            f.write(np.ascontiguousarray(a).tobytes())


def link(src, dst):
    if os.path.exists(dst):
        os.remove(dst)
    os.link(src, dst)                                       # raises on failure: NEVER a copy fallback


def make_train(src, dst, vec, cell, plant_on="pos"):
    os.makedirs(dst, exist_ok=True)
    for k in TRAIN_FILES:
        link(os.path.join(src, k + ".npy"), os.path.join(dst, k + ".npy"))
    for k in SMALL_LINKS:
        if os.path.exists(os.path.join(src, k)):
            link(os.path.join(src, k), os.path.join(dst, k))
    v40 = np.load(os.path.join(src, "v40.npy"), mmap_mode="r")
    y = np.load(os.path.join(src, "y.npy"))
    counts = []

    def gen():
        for _, a, n in plant_array(v40, y, vec, cell, plant_on):
            counts.append(n)
            yield a

    write_npy_stream(os.path.join(dst, "v40.npy"), v40.shape, v40.dtype, gen())
    meta = json.load(open(os.path.join(src, "meta.json"), encoding="utf-8"))
    info = {"touched_windows": int(sum(counts)), "n_windows": int(len(y)), "n_positive": int(np.asarray(y).astype(bool).sum()), "plant_on": plant_on}
    meta["c2"] = dict(info, replaced=["v40"], hardlinked=list(TRAIN_FILES) + list(SMALL_LINKS), note="MANIFEST.sha256 is the source's and does not cover the planted v40")
    with open(os.path.join(dst, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1, sort_keys=True)
    return info


def make_eval(src_npz, dst_npz, vec, cell, plant_on="pos"):
    z = np.load(src_npz, allow_pickle=False)
    d = {k: z[k] for k in z.files}
    y = d["y"]
    parts, n = [], 0
    for _, a, k in plant_array(d["v40"], y, vec, cell, plant_on):
        parts.append(a)
        n += k
    d["v40"] = np.concatenate(parts)
    meta = json.loads(str(d["meta"]))
    meta["c2"] = {"touched_windows": n, "n_windows": int(len(y)), "n_positive": int(y.astype(bool).sum()), "plant_on": plant_on, "replaced": ["v40"]}
    d["meta"] = np.array(json.dumps(meta, sort_keys=True))
    np.savez(dst_npz, **d)
    return {"touched_windows": n, "n_windows": int(len(y)), "n_positive": int(y.astype(bool).sum()), "plant_on": plant_on}


def make(base, samples, amp=1.0, plant_on="pos", row=CELL_ROW, col=CELL_COL, seed=SEED, norm_from=None, log=print):
    out = os.path.join(base, "c2")
    cell = row * GRID_W + col
    est = sum(os.path.getsize(os.path.join(base, "assembled", f"{s}_416", "v40.npy")) for s in samples) + os.path.getsize(os.path.join(base, "assembled", "eval_416.npz"))
    need_free(out, est)
    src0 = os.path.join(base, "assembled", f"{(norm_from or samples[0])}_416")
    v0 = np.load(os.path.join(src0, "v40.npy"), mmap_mode="r")
    norm = median_token_norm(v0)
    u = direction(v0.shape[-1], seed)
    vec = (u * norm * amp).astype(np.float32)
    prov = {"seed": seed, "cell_row": row, "cell_col": col, "cell_index": cell, "median_token_norm": norm, "norm_from": src0, "amp": amp, "plant_on": plant_on,
            "direction_unit_sha256_float64": hashlib.sha256(u.tobytes()).hexdigest(), "vector_sha256_float32": hashlib.sha256(vec.tobytes()).hexdigest(),
            "vector_l2": float(np.linalg.norm(vec.astype(np.float64))), "expected_extra_bytes": est, "samples": {}}
    for s in samples:
        dst = os.path.join(out, "assembled", f"{s}_416")
        info = make_train(os.path.join(base, "assembled", f"{s}_416"), dst, vec, cell, plant_on)
        info["v40_sha256"] = sha256_file(os.path.join(dst, "v40.npy"))
        prov["samples"][s] = info
        log(f"planted {dst}: {info}")
    ev = os.path.join(out, "assembled", "eval_416.npz")
    prov["eval"] = make_eval(os.path.join(base, "assembled", "eval_416.npz"), ev, vec, cell, plant_on)
    prov["eval"]["npz_sha256"] = sha256_file(ev)
    with open(os.path.join(out, "c2_make.json"), "w", encoding="utf-8") as f:
        json.dump(prov, f, indent=1)
    log(f"planted eval {ev}: {prov['eval']}")
    return prov


# ------------------------------------------------------------------------------------------------------------ prep (v40 only) and fit
def prep_v40(work, tr, ev, log=print):
    sys.path.insert(0, S0P)
    import s0p_fit as F
    os.makedirs(work, exist_ok=True)
    if os.path.exists(os.path.join(work, "proj_eval_v40.npy")):
        log("  skip prep (banked)")
        return
    pca = F.fit_pca(tr["v40"])
    np.savez(os.path.join(work, "pca_v40.npz"), mean=pca["mean"], comps=pca["comps"], eig=pca["eig"], n_tokens=pca["n_tokens"])
    np.save(os.path.join(work, "proj_train_v40.npy"), F.project(tr["v40"], pca))
    np.save(os.path.join(work, "proj_eval_v40.npy"), F.project(ev["v40"], pca))
    log(f"  pca v40: {pca['n_tokens']} tokens")


def paths(base, s):
    c = os.path.join(base, "c2")
    return os.path.join(c, "assembled", f"{s}_416"), os.path.join(c, "assembled", "eval_416.npz"), os.path.join(c, "work", f"{s}_416")


def cmd_prep(base, samples):
    sys.path.insert(0, S0P)
    import s0p_fit as F
    for s in samples:
        tr_dir, ev_p, work = paths(base, s)
        need_free(work, 0.4e9)
        prep_v40(work, F.load_any(tr_dir), F.load_any(ev_p))


def cmd_fit(base, s, model, py=PY_DEFAULT):
    tr_dir, ev_p, work = paths(base, s)
    need_free(work, 0.05e9)
    r = subprocess.run([py, f"{S0P}/s0p_fit.py", "fit", "--train", tr_dir, "--eval", ev_p, "--work", work, "--arm", "V40+E", "--model", model], encoding="utf-8", errors="replace")
    if r.returncode:
        raise RuntimeError(f"s0p_fit fit {s} {model} rc {r.returncode}")


# ------------------------------------------------------------------------------------------------------------ read
def decide_from_scores(per_sample, y, eid, ci_mod, n_boot=None):
    """per_sample[s] = {"E|f", "V40@416+E|f", "V40@256p+E|f"} eval score arrays (f in linear, mlp) -> (code, detail, cells, aurocs)."""
    import q5_read as R
    nb = n_boot or R.NB
    inp, aur = {}, {}
    for s, sc in per_sample.items():
        _, cells = R.bootstrap_sample(sc, y, eid, ci_mod, n_boot=nb)
        inp[s] = cells
        aur[s] = {f: R._auroc(sc[f"V40@416+E|{f}"], y) for f in R.FAMILIES if f"V40@416+E|{f}" in sc}
    code, det = R.decide(inp)
    return code, det, inp, aur


def read(base, samples=("S1", "S2"), n_boot=None, tree=None):
    import q5_read as R
    cfg = json.load(open(os.path.join(base, "read_cfg.json"), encoding="utf-8"))
    tree = tree or cfg["tree"]
    sys.path.insert(0, tree)
    sys.path.insert(0, tree + "/taniteval")
    from taniteval import ci as CI
    ev = np.load(cfg["eval"], allow_pickle=False)
    y, eid = ev["y"], ev["eid"]
    per = {}
    for s in samples:
        w416, w256 = os.path.join(base, "work", f"{s}_416"), os.path.join(base, "work", f"{s}_256p")
        wc = paths(base, s)[2]
        sc = {}
        for f in R.FAMILIES:
            sc[f"E|{f}"] = np.load(os.path.join(w416, f"fit_E_{f}.npy"))                              # REAL
            sc[f"V40@256p+E|{f}"] = np.load(os.path.join(w256, f"fit_V40+E_{f}.npy"))                 # REAL
            sc[f"V40@416+E|{f}"] = np.load(os.path.join(wc, f"fit_V40+E_{f}.npy"))                    # PLANTED
        per[s] = sc
    code, det, inp, aur = decide_from_scores(per, y, eid, CI, n_boot)
    lin = min(a["linear"] for a in aur.values())
    allm = min(v for a in aur.values() for v in a.values())
    res = {"c2_auroc": allm, "c2_outcome": code,
           "c2_auroc_definition": "min over samples AND over both families (linear, mlp) of the V40@416+E eval AUROC, planted; the amendment names neither, this is the conservative reading",
           "c2_auroc_linear_min": lin, "c2_auroc_mlp_min": min(a["mlp"] for a in aur.values()), "auroc_by_sample": aur, "outcome_detail": det, "cells_g40": {s: {f: inp[s][40].get(f) for f in R.FAMILIES} for s in inp},
           "bar": {"auroc_ge": 0.99, "outcome": "RES416-ADDS-40"}, "scores_planted": "V40@416+E from run/c2/work; E and V40@256p+E are the REAL run/work fits",
           "make": json.load(open(os.path.join(base, "c2", "c2_make.json"), encoding="utf-8")) if os.path.exists(os.path.join(base, "c2", "c2_make.json")) else None}
    with open(os.path.join(base, "c2", "RESULT_C2.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, default=str)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("make", "prep", "fit", "read"))
    ap.add_argument("--base", required=True)
    ap.add_argument("--samples", default="S1,S2")
    ap.add_argument("--sample")
    ap.add_argument("--model", choices=("linear", "mlp"))
    ap.add_argument("--python", default=PY_DEFAULT)
    ap.add_argument("--amp", type=float, default=1.0)
    ap.add_argument("--plant-on", choices=("pos", "all"), default="pos")
    a = ap.parse_args(argv)
    samples = [s for s in a.samples.split(",") if s]
    if a.cmd == "make":
        make(a.base, samples, a.amp, a.plant_on)
    elif a.cmd == "prep":
        cmd_prep(a.base, samples)
    elif a.cmd == "fit":
        cmd_fit(a.base, a.sample, a.model, a.python)
    else:
        r = read(a.base, tuple(samples))
        print(json.dumps({k: r[k] for k in ("c2_auroc", "c2_outcome", "c2_auroc_linear_min", "c2_auroc_mlp_min")}, indent=1))


if __name__ == "__main__":
    main()
