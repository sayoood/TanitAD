"""PREREG_V8F_DECN. Usage: python run_decn.py a   (prepare TUNE inputs, control 1, stream 1,000 train clips -> latents)
                    python run_decn.py b   (readers for N53, N194, N1053 -> runs/<rung>_<tag>_<N>.json)
Exit codes of stage a: 0 done; 10 network stop (stage b still useful); 2 control 1 failed; 3 md5 mismatch (control 6) - stop everything.
Resumable: present shards / unit files are skipped."""
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import torch

import decn_lib as DN
from decn_lib import DL

W = DL.W
D_ = W.D
SFTP = "C:/Program Files/Git/usr/bin/sftp.exe"
REMOTE = "/D:/tanitad/cache_full/train/clips"
STREAM = f"{DN.SCRATCH}/stream"
DEST_REL = "cache/train/clips/_current.pt"
N_START = 17


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def jdump(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


# -------------------------------------------------------------------------------------------------------- stage a
class ClipWins:
    """The feats / act / frame_ok part of V8FMiniWindows.item for one streamed clip (same expressions as wmact/collect_latents_train.ClipWins)."""

    def __init__(self, d, hmax=16):
        self.d, self.hmax = d, hmax

    def item(self, t):
        d = self.d
        T = len(d["pose"])
        idx = np.arange(t - self.hmax + 1, t + 1)
        okf = idx >= 0
        idc = np.clip(idx, 0, T - 1)
        okf &= np.isfinite(d["p10f"][idc]).all(1) | (idx < 1)
        return {"feats": d["feats"][idc], "frame_ok": okf, "act": np.stack([np.nan_to_num(d["a"][idc]), d["ch1"][idc]], -1).astype(np.float32), "idx": idx}


def window_starts(clip, n_frames, cfg):
    J = {"clip": np.array([clip]), "n": np.array([n_frames])}
    valid = np.array([w[1] for w in D_.train_windows(None, J, [clip], hist=cfg.hist, start=cfg.window_start)])
    return np.unique(np.round(np.linspace(valid[0], valid[-1], N_START)).astype(int))


def pull(clip, md5):
    os.makedirs(os.path.join(STREAM, "cache/train/clips"), exist_ok=True)
    batch = os.path.join(STREAM, "_get.txt")
    with open(batch, "w", newline="\n") as fh:
        fh.write(f"get {REMOTE}/{clip}.pt {DEST_REL}\nbye\n")
    last = None
    for attempt in range(3):
        try:
            subprocess.run([SFTP, "-q", "-b", "_get.txt", "tanitad-box2"], cwd=STREAM, check=True, timeout=300, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as e:
            last = ("net", repr(e)); time.sleep(20); continue
        h = hashlib.md5()
        with open(os.path.join(STREAM, DEST_REL), "rb") as fh:
            for blk in iter(lambda: fh.read(1 << 20), b""):
                h.update(blk)
        if h.hexdigest() == md5:
            return
        last = ("md5", h.hexdigest()); log("  md5 mismatch", clip, attempt)
    if last[0] == "md5":
        raise RuntimeError(f"MD5:{clip}: never matched the manifest")
    raise ConnectionError(f"NET:{clip}: {last[1]}")


def models_all():
    ms = {}
    for tag in ("M2b_s0", "M2b_s1"):
        ms[tag] = DL.load_any(DL.M2B[tag])[0]
    for tag, p in DL.ladder_models().items():
        ms[tag] = DL.load_any(p)[0]
    return ms


def prepare():
    """TUNE inputs for every tag, control 1, ladder tags. Resumable by file."""
    os.makedirs(DN.LAT, exist_ok=True)
    c1p = f"{DN.SCRATCH}/control1.json"
    if os.path.exists(DN.TUNE_NPZ) and os.path.exists(c1p):
        return json.load(open(c1p, encoding="utf-8"))
    DL.wait_ram()
    D, sp = DL.load_tune()
    assert D.n == 982 and D.nf == 744 and D.fit_eps == 53 and D.held_eps == 17
    L = torch.load(DL.EVAL, weights_only=False); eid_all = np.array(L["eid"]); tt = np.asarray(L["t"])
    # control 1: the eval clips' own cache `v` (the logged-speed source used for train clips) against the eval target
    a, b, vs = [], [], {}
    for j, di in enumerate(D.ds_idx):
        e = str(eid_all[di]); g = int(tt[di]) - 15
        if e not in vs:
            vs[e] = torch.load(f"{W.SF}/cache/eval/clips/{e}.pt", weights_only=True)["v"].numpy().astype(np.float64)
        for k in range(16):
            if np.isfinite(D.vk[j, k]) and 0 <= g + k < len(vs[e]):
                a.append(D.vk[j, k]); b.append(vs[e][g + k])
    a, b = np.array(a), np.array(b)
    r2 = float(1 - ((a - b) ** 2).sum() / ((a - a.mean()) ** 2).sum()); mad = float(np.abs(a - b).mean())
    c1 = dict(n_values=int(len(a)), n_target_values=int(np.isfinite(D.vk).sum()), n_tune_windows=int(D.n), r2=r2, mean_abs_diff=mad, passed=bool(r2 >= 0.99 and mad <= 0.10 and len(a) == int(np.isfinite(D.vk).sum())))
    log("control 1", c1)
    if not c1["passed"]:
        jdump(c1, c1p)
        open(os.path.join(HERE, "CONTROL1_FAILED.txt"), "w").write(json.dumps(c1))
        raise SystemExit(2)
    ms = models_all()
    tags = [t for t in ms if t not in ("M2b_s0", "M2b_s1")]
    out = dict(eid=D.eid.astype("U36"), vk=D.vk, yk=D.yk, nf=np.array(D.nf), es=D.es, fit_eps=np.array(D.fit_eps), held_eps=np.array(D.held_eps), pos=D.pos)
    meta = {}
    for tag, model in ms.items():
        DL.wait_ram()
        z1, pooled = DL.vector_inputs(model, D.feats)
        if tag in D.z1_ref:
            dif = float((z1 - D.z1_ref[tag]).abs().max()); meta[tag] = dict(z1_vs_stored_latents_max_abs_diff=dif)
            assert dif < 1e-4, (tag, dif)
        out[f"z1_{tag}"] = z1.numpy().astype(np.float32)
        if tag.startswith("M2b"):
            out[f"pooled_{tag}"] = pooled.numpy().astype(np.float32)
        log("tune inputs", tag)
    np.savez(DN.TUNE_NPZ, **out)
    jdump(dict(ladder_tags=tags, meta=meta), f"{DN.SCRATCH}/ladder_tags.json")
    jdump(c1, c1p)
    del D
    return c1


def stage_a():
    os.makedirs(STREAM, exist_ok=True)
    c1 = prepare()
    if not c1["passed"]:
        return 2
    c141, ce = DN.clip_lists()
    order = list(c141["clips"]) + list(ce["clips"])
    md5 = dict(c141["md5"]); md5.update(ce["md5"]); nfr = dict(c141["n_frames"]); nfr.update(ce["n_frames"])
    assert len(order) == 1000 and len(set(order)) == 1000
    T = DN.load_tune_npz()
    tune_eids = set(map(str, T["eid"]))
    assert not (set(order) & tune_eids), "a train addition is a TUNE episode"
    ev_clips = {f[:-3] for f in os.listdir(f"{W.SF}/cache/eval/clips") if f.endswith(".pt")}
    DN.split_guard_n(set(map(str, T["eid"][:int(T["nf"])])), set(map(str, T["eid"][int(T["nf"]):])), DL.es_ids_of(set(map(str, T["eid"][:int(T["nf"])]))), ev_clips, order)
    tags_meta = json.load(open(f"{DN.SCRATCH}/ladder_tags.json", encoding="utf-8"))
    ms = models_all()
    assert [t for t in ms if t not in ("M2b_s0", "M2b_s1")] == tags_meta["ladder_tags"], "ladder checkpoints changed since the TUNE inputs were built"
    _, cfg, _ = W.load_model(W.CKPT["M2b_s0"])
    assert cfg.hmax == 16 and cfg.hist == 4 and cfg.window_start_effective == 3
    tracks_all = D_.load_native_tracks(f"{W.SF}/inputs/native_tracks.pt")
    store = D_.ClipStore(STREAM + "/cache", "train", lru=1, grid="pool4x10", kin_mode="yaw", tracks={"_current": None})
    t0 = time.time(); done = skipped = 0
    vdiff = {}
    for i, c in enumerate(order):
        sp_ = f"{DN.LAT}/{c}.npz"
        if os.path.exists(sp_):
            skipped += 1; continue
        DL.wait_ram()
        try:
            pull(c, md5[c])
        except RuntimeError as e:
            open(os.path.join(HERE, "STREAM_STOPPED.txt"), "w").write(f"md5 mismatch (control 6): {e}\n"); log("STOP", e); return 3
        except ConnectionError as e:
            open(os.path.join(HERE, "STREAM_STOPPED.txt"), "w").write(f"network stop after retries at clip {i}: {e}\n"); log("STOP", e); return 10
        store.tracks.clear(); store.tracks["_current"] = tracks_all[c]; store._c.clear()
        d = store.get("_current")
        assert len(d["pose"]) == nfr[c], (c, len(d["pose"]), nfr[c])
        cw = ClipWins(d)
        ts = window_starts(c, nfr[c], cfg)
        keep, items = [], []
        for t in ts:
            it = cw.item(int(t))
            if bool(it["frame_ok"].all()) and it["idx"].max() < len(d["v"]):
                keep.append(int(t)); items.append(it)
        arrs = {"ts": np.array(keep)}
        if items:
            feats = np.stack([it["feats"] for it in items])
            for tag, model in ms.items():
                z1, pooled = DL.vector_inputs(model, feats)
                arrs[f"z1_{tag}"] = z1.numpy().astype(np.float32)
                if tag.startswith("M2b"):
                    arrs[f"pooled_{tag}"] = pooled.numpy().astype(np.float32)
            arrs["v"] = np.stack([d["v"][it["idx"]].astype(np.float64) for it in items])
            arrs["yaw"] = np.stack([it["act"][:, 1].astype(np.float64) for it in items])
            tr = np.stack([tracks_all[c][2 * it["idx"], 3] for it in items])
            vdiff[c] = float(np.nanmax(np.abs(tr - arrs["v"]))) if np.isfinite(tr).any() else None
        else:
            for tag in ms:
                arrs[f"z1_{tag}"] = np.zeros((0, 16, 128), np.float32)
                if tag.startswith("M2b"):
                    arrs[f"pooled_{tag}"] = np.zeros((0, 16, 256), np.float32)
            arrs["v"] = np.zeros((0, 16)); arrs["yaw"] = np.zeros((0, 16))
        tmp = sp_ + ".tmp"
        with open(tmp, "wb") as fh:
            np.savez(fh, **arrs)
        os.replace(tmp, sp_)
        done += 1
        if done % 10 == 0 or done == 1:
            log(f"clip {i + 1}/{len(order)} {c[:8]} kept {len(keep)} windows, streamed this run {done}, {time.time() - t0:.0f}s")
        if done % 50 == 0:
            jdump(vdiff, f"{DN.SCRATCH}/train_v_vs_track_part{os.getpid()}.json")
    p = f"{DN.SCRATCH}/train_v_vs_track.json"
    prev = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}
    prev.update(vdiff); jdump(prev, p)
    try:
        os.remove(os.path.join(STREAM, DEST_REL))                    # own scratch clip, source on box 2
    except OSError:
        pass
    log("stage a complete", done, "streamed,", skipped, "already present")
    return 0


# -------------------------------------------------------------------------------------------------------- stage b
def unit_path(rung, tag, N):
    return os.path.join(DN.RUNS, f"{rung}_{tag}_{N}.json")


def train_targets(order, k):
    vs, ys = [], []
    for c in order[:k]:
        z = np.load(f"{DN.LAT}/{c}.npz")
        vs.append(z["v"]); ys.append(z["yaw"])
    return np.concatenate(vs), np.concatenate(ys)


def stage_b():
    os.makedirs(DN.RUNS, exist_ok=True)
    c141, ce = DN.clip_lists()
    order = list(c141["clips"]) + list(ce["clips"])
    k_av = DN.available_clips(order)
    log("clips available", k_av, "of", len(order))
    T = DN.load_tune_npz()
    tags_meta = json.load(open(f"{DN.SCRATCH}/ladder_tags.json", encoding="utf-8"))
    c1 = json.load(open(f"{DN.SCRATCH}/control1.json", encoding="utf-8"))
    Ns = [("N53", 0)]
    if k_av >= 141:
        Ns.append(("N194", 141))
    if k_av >= 1000:
        Ns.append(("N1053", 1000))
    elif k_av > 141:
        Ns.append((f"NP{53 + k_av}", k_av))
    units = []
    for tag in ("M2b_s0", "M2b_s1"):
        units += [("R0", tag, "z1"), ("R2", tag, "pooled")]
    for tag in tags_meta["ladder_tags"]:
        units.append(("R1", tag, "z1"))
    ev_clips = {f[:-3] for f in os.listdir(f"{W.SF}/cache/eval/clips") if f.endswith(".pt")}
    # the N loop is outermost per tag so that a stopped run still has the deciding inputs complete
    for rung, tag, kind in units:
        key = f"{kind}_{tag}"
        todo = [(N, k) for N, k in Ns if not os.path.exists(unit_path(rung, tag, N))]
        if not todo:
            continue
        DL.wait_ram()
        kmax = max(k for _, k in todo)
        if kmax:
            X_tr, ci_tr = DN.load_train_arrays(order, kmax, key)
            tr_t = train_targets(order, kmax)
            assert len(tr_t[0]) == len(X_tr)
        else:
            X_tr, ci_tr, tr_t = np.zeros((0, 16, T[key].shape[2]), np.float32), np.zeros(0, int), (np.zeros((0, 16)), np.zeros((0, 16)))
        for N, k in todo:
            DL.wait_ram()
            D, X, ids_tr = DN.assemble(T, key, X_tr, ci_tr, order, k, tr_t)
            DN.split_guard_n(set(map(str, D.eid[:D.nf])), set(map(str, D.eid[D.nf:])), DL.es_ids_of(set(map(str, D.eid[:D.nf]))), ev_clips, ids_tr)
            assert not (set(map(str, D.eid[:D.nf])) & set(map(str, D.eid[D.nf:]))), "fit and held-out episodes overlap"
            t0 = time.time()
            log("unit", rung, tag, N, "fit windows", D.nf, "episodes", D.n_fit_episodes, "held windows", D.n - D.nf)
            rows, planted = DN.read_unit(D, X, rung, tag, N, log)
            meta = dict(added_after_freeze=bool(rung == "R1" and tag not in DL.FROZEN_LADDER), held_digest=D.held_digest, n_clips=k, n_fit_windows=int(D.nf), n_held_windows=int(D.n - D.nf),
                        n_fit_episodes=D.n_fit_episodes, n_held_episodes=D.n_held_episodes, wall_s=time.time() - t0, control1=c1)
            if N == "N53" and rung == "R0":
                rep = {}
                for tgt in ("speed", "yaw"):
                    mine = [r["r2"] for r in rows if r["reader"] == "ridge" and r["m"] == 1 and r["target"] == tgt][0]
                    pre = [r["r2"] for r in json.load(open(f"{DL.HERE}/runs/R0_{tag}.json", encoding="utf-8"))["rows"] if r["reader"] == "ridge" and r["m"] == 1 and r["target"] == tgt][0]
                    rep[tgt] = dict(mine=mine, decinfo=pre, abs_diff=abs(mine - pre), passed=bool(abs(mine - pre) <= 0.01))
                meta["control2_reproduction"] = rep
            jdump(dict(rung=rung, tag=tag, N=N, tier="T0", rows=rows, planted=planted, meta=meta), unit_path(rung, tag, N))
            log("saved", unit_path(rung, tag, N), f"{time.time() - t0:.0f}s")
            del D, X
        del X_tr
    log("stage b complete")
    return 0


if __name__ == "__main__":
    st = sys.argv[1]
    try:
        rc = stage_a() if st == "a" else stage_b()
    except SystemExit as e:
        rc = int(e.code)
    except Exception:
        log("FAILED", traceback.format_exc()); rc = 1
    sys.exit(rc)
