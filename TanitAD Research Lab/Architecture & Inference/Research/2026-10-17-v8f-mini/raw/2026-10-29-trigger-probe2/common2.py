"""common2.py -- shared layer of trigger probe 2 (PREREG_FROZEN.md).  Reads the previous package (trigger_probe/) READ-ONLY; writes only under trigger_probe2/.
EVIDENCE CLASS of everything computed with it: MEASURED (artifacts under raw/).  ESTIMATOR: clip-cluster bootstrap B = 2000 seed 0 (answers "another draw of EPISODES", blind to training variance, H-ESTIM-SEED-1).
PRIMARY METRIC: the mean of per-fold AUCs ("fold"); the pooled out-of-fold AUC is reported beside ("pooled")."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.dont_write_bytecode = True
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
RAW, SCR = f"{HERE}/raw", f"{HERE}/scratch"
OLD = "E:/Projects/TanitAD-artifacts/dagland/trigger_probe"
OLDRAW, OLDSCR = f"{OLD}/raw", f"{OLD}/scratch"
TREE = "E:/Projects/TanitAD-artifacts/dagt1_tree_b43c7a9a"
EVAL_JOIN = f"{TREE}/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/join_eval141.npz"
CKPT = "E:/Projects/TanitAD-artifacts/v8f-dagoacc/M2b_s0_dagoacc/ckpt_last.pt"
CKPT_SHA = "67265ec1666fae3a6d51e924ee06dfb68b2cd0630afc62e42f3a46182b8bc45f"
CACHE_ROOT = "F:/TanitAD-scratch/speedread/cache"
CACHE_CLIPS = f"{CACHE_ROOT}/train/clips"
MANIFEST_TRAIN = "C:/Users/Admin/box2_ship/logs/v8f_dscale/manifest_train.jsonl"
TRACKS = "E:/Projects/TanitAD-artifacts/sfresh_inputs/inputs/native_tracks.pt"
EVAL_CACHE = "E:/Projects/TanitAD-artifacts/sfresh_inputs/cache"
sys.path.insert(0, HERE)
import tp_lib as L  # noqa: E402  (byte-identical copy of trigger_probe/tp_lib.py)

POPS = ("pstop", "pmove")


def free_mb() -> int:
    r = subprocess.run(["powershell", "-NoProfile", "-Command", "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1024)"], capture_output=True, text=True)
    return int(r.stdout.strip() or 0)


def require_ram(min_mb=3000, what="heavy load"):
    fm = free_mb()
    if fm < min_mb:
        raise SystemExit(f"RAM GUARD: free {fm} MB < {min_mb} MB before {what}: write HANDOFF.md and end (a SOUP job has priority)")
    return fm


def sha256_file(p, blk=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(blk), b""):
            h.update(b)
    return h.hexdigest()


def load_windows():
    return np.load(f"{OLDRAW}/windows_train.npz", allow_pickle=False)


def load_pop(pop):
    """Identical to run_probe.load_pop of the previous package (same rows, same order, same folds)."""
    A = load_windows()
    idx = np.load(f"{OLDRAW}/sample_{pop}.npz")["idx"]
    clip = A["clip_i"][idx].astype(np.int64)
    sha = A["episode_sha12"][clip]
    d = {"idx": idx, "y": A["go"][idx].astype(int), "y24": A["go24"][idx].astype(int), "clip": clip, "sha": sha, "outer": L.outer_fold(sha), "inner": L.inner_fold(sha),
         "E": A["E"][idx].astype(np.float64), "v0": A["v0"][idx], "t": A["t"][idx]}
    for k in ("TL_ANY", "TL_RED", "TL_GREEN", "TL_YELLOW", "NO_TL", "lead_now", "no_lead", "lead_depart"):
        d[k] = A[k][idx]
    return d


def banked(pop, fam):
    """Old banked out-of-fold scores (read-only): dict name -> scores, with y / clip for the row-order assert."""
    ov = dict(np.load(f"{OLDRAW}/oof_{pop}_V_{fam}.npz"))
    oz = dict(np.load(f"{OLDRAW}/oof_{pop}_Z_{fam}.npz"))
    out = {"E": ov["E"], "V": ov["V"], "V+E": ov["V+E"], "Z": oz["Z"], "Z+E": oz["Z+E"]}
    if fam == "linear":
        out["E_H"] = ov["E_H"]
        out["y"], out["clip"] = ov["y"], ov["clip"]
    return out


def join_nav(J, clips, clip_i, t):
    """nav5 / limit5 at NOW for windows (clip_i indexes `clips`, the windows_train name table).  J = the join npz as a dict of arrays."""
    ix = {c: i for i, c in enumerate(J["clip"].tolist())}
    ci = np.array([ix[str(clips[c])] for c in clip_i])
    row = J["off"][ci] + np.asarray(t)
    nav = J["nav"][row].astype(np.float64)
    lim = np.concatenate([J["limit"][ci], J["limit_ok"][ci][:, None].astype(np.float32)], 1).astype(np.float64)
    return nav, lim, row, ci


def load_join_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


# ------------------------------------------------------------------------------------------------ resumable arm runner
def rec(o, y, clip, outer, counts):
    return {"pooled": L.auc_ci(o, y, clip, counts=counts), "fold": L.auc_ci_fold(o, y, clip, outer, counts=counts)}


def paired(a, b, y, clip, outer, counts):
    return {"fold": L.dauc_ci_fold(a, b, y, clip, outer, counts=counts), "pooled": L.dauc_ci(a, b, y, clip, counts=counts)}


def run_arms(tag, d, fam, arms, budget_s=170.0):
    """arms: ordered dict name -> (feat_fn(f) -> [N, d] float64, labels or None).  Banks raw/oof2_<tag>_<fam>.npz and raw/res2_<tag>_<fam>.json after EVERY arm; resumable; stops after ``budget_s`` (returns False)."""
    y, clip, outer, inner = d["y"], d["clip"], d["outer"], d["inner"]
    counts = L.boot_counts(clip)
    resf, oofp = f"{RAW}/res2_{tag}_{fam}.json", f"{RAW}/oof2_{tag}_{fam}.npz"
    res = json.load(open(resf)) if os.path.exists(resf) else {"tag": tag, "family": fam, "n": int(len(y)), "n_go": int(y.sum()), "n_clips": int(len(np.unique(clip))), "arms": {}}
    oof = dict(np.load(oofp)) if os.path.exists(oofp) else {}
    t0 = time.time()
    for name, (fn, lab) in arms.items():
        key = name.replace(":", "__")
        if key in oof:
            continue
        labels = y if lab is None else lab
        o, hps = L.cv_oof(fn, labels, outer, inner, fam)
        oof[key] = o
        r = rec(o, labels, clip, outer, counts)
        res["arms"][name] = {**r, "hp_per_fold": hps}
        np.savez(oofp, **oof)
        json.dump(res, open(resf, "w"), indent=1)
        print(f"{tag} {fam} {name}: fold-mean {r['fold']['auc']:.4f} [{r['fold']['lo']:.4f},{r['fold']['hi']:.4f}]  pooled {r['pooled']['auc']:.4f} [{r['pooled']['lo']:.4f},{r['pooled']['hi']:.4f}]  {time.time() - t0:.0f}s", flush=True)
        if time.time() - t0 > budget_s:
            print("time box: banked, re-invoke to continue", flush=True)
            return False
    return True


# ------------------------------------------------------------------------------------------------ donor draws (ARM 2), pure numpy
def draw_donors(own_clip, clip_all, seed):
    """Donor window index (into the canonical windows) per item, never from the item's own clip; deterministic given (own_clip order, seed)."""
    rng = np.random.default_rng(seed)
    n_all = len(clip_all)
    d = rng.integers(0, n_all, len(own_clip))
    bad = clip_all[d] == own_clip
    while bad.any():
        d[bad] = rng.integers(0, n_all, int(bad.sum()))
        bad = clip_all[d] == own_clip
    return d


def nav_hist(J, jclip, td, hmax):
    """The nav rows of the donor window exactly as V8FMiniWindows.item builds them (idc = clip(arange(t - hmax + 1, t + 1), 0, T - 1))."""
    T = int(J["n"][jclip])
    idc = np.clip(np.arange(td - hmax + 1, td + 1), 0, T - 1)
    return J["nav"][int(J["off"][jclip]) + idc]
