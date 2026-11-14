"""PREREG_V8F_WMACT TRAIN read, step 1: STREAMED pull + level-1 latents of 141 TRAIN clips (CPU).

For each clip of train_stream/clips141.json: sftp (read only) the clip into ONE scratch path (cache/train/clips/_current.pt, overwritten each time),
verify its md5 against box 2's manifest, open it with D.ClipStore, take 17 window starts, run W.collect_latents (both M2b seeds), keep only latents + raw actions.
Writes latents_train.pt with the layout of latents.pt (+ clips / clips_sha256 / ctrl4 / starts).  Resumable via latents_train_R35_partial.pt.
Command: see HANDOFF_WMACT_TRAIN.md.
"""
import hashlib, json, os, subprocess, sys, time
import numpy as np, torch
import wmact_lib as W

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
HERE = "F:/TanitAD-scratch/hwm_plan"
TS = os.path.join(HERE, "ts")
SFTP = "C:/Program Files/Git/usr/bin/sftp.exe"
REMOTE = "/D:/tanitad/cache_full/train/clips"
N_START = 17
D = W.D


class ClipWins:
    """Duck-typed dataset for ``W.collect_latents``: one clip's windows. ``item`` is the feats / act / frame_ok / t part of ``V8FMiniWindows.item``
    (v8f_mini_data.py:573-597, same expressions); test_train_item_shim.py compares it array for array with the real ``item`` on eval windows."""

    def __init__(self, d: dict, ts, hmax: int):
        self.d, self.hmax = d, int(hmax)
        self.windows = [(0, int(t), 0, 0) for t in ts]

    def item(self, ci: int, t: int, a_t: int = 0, a_s: int = 0) -> dict:
        d = self.d
        T = len(d["pose"])
        idx = np.arange(t - self.hmax + 1, t + 1)
        okf = idx >= 0
        idc = np.clip(idx, 0, T - 1)
        okf &= np.isfinite(d["p10f"][idc]).all(1) | (idx < 1)
        return {"feats": d["feats"][idc], "frame_ok": okf, "act": np.stack([np.nan_to_num(d["a"][idc]), d["ch1"][idc]], -1).astype(np.float32), "t": t}


def window_starts(clip: str, n_frames: int, cfg):
    """17 starts evenly spaced over the TRAINER's valid starts: ``D.train_windows`` (the function the trainer calls, v8f_mini_train.py:2411) on a two-key join."""
    J = {"clip": np.array([clip]), "n": np.array([n_frames])}
    valid = np.array([w[1] for w in D.train_windows(None, J, [clip], hist=cfg.hist, start=cfg.window_start)])
    return np.unique(np.round(np.linspace(valid[0], valid[-1], N_START)).astype(int)), valid


def pull(clip: str, md5: str, dest_rel: str) -> float:
    batch = os.path.join(TS, "_get.txt")
    with open(batch, "w", newline="\n") as fh:
        fh.write(f"get {REMOTE}/{clip}.pt {dest_rel}\nbye\n")
    for attempt in range(3):
        t0 = time.time()
        subprocess.run([SFTP, "-q", "-b", "_get.txt", "tanitad-box2"], cwd=TS, check=True, timeout=300, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        dt = time.time() - t0
        h = hashlib.md5()
        with open(os.path.join(TS, dest_rel), "rb") as fh:
            for blk in iter(lambda: fh.read(1 << 20), b""):
                h.update(blk)
        if h.hexdigest() == md5:
            return dt
        print(f"  md5 mismatch on {clip}, attempt {attempt}", flush=True)
    raise RuntimeError(f"{clip}: md5 never matched box 2's manifest")


def control4(models: dict, feats: torch.Tensor, act: torch.Tensor) -> dict:
    """Control 4 as run_wmact.py runs it (same expressions), on the first 64 TRAIN windows."""
    out = {}
    for tag, m in models.items():
        with torch.no_grad():
            adapt = m.adapter(feats.float()) + m.pos[None, None]
            torch.manual_seed(0); direct = m.wm(adapt, act)
            torch.manual_seed(0); via = m.wm.stack.loss(m.wm.z1(adapt), m.wm.scale_actions(act)[:, :-1])
        out[tag] = {"max_abs_diff": max(abs(float(direct[k]) - float(via[k])) for k in direct), "terms": {k: float(v) for k, v in direct.items()}, "n_windows": int(feats.shape[0])}
    return out


def main():
    t0 = time.time()
    meta = json.load(open(os.path.join(TS, "clips141.json")))
    clips = meta["clips"]
    assert hashlib.sha256("\n".join(clips).encode()).hexdigest() == meta["clips_sha256"]
    models, cfgs = {}, {}
    LAD = {"R35_s0": "C:/Users/Admin/t2v8f_pulls/ladder_R35_s0/ckpt_last.pt", "R35_s1": "C:/Users/Admin/t2v8f_pulls/ladder_R35_s1/ckpt_last.pt"}
    for tag, path in LAD.items():
        models[tag], cfgs[tag], _ = W.load_model(path)
    cfg = cfgs["R35_s0"]
    assert all(c.hmax == 91 and c.wm == "b" and c.kin_mode == "yaw" and c.hist == 4 and c.window_start_effective == 3 for c in cfgs.values())
    tracks_all = D.load_native_tracks(f"{W.SF}/inputs/native_tracks.pt")
    cache = os.path.join(TS, "cache")
    store = D.ClipStore(cache, "train", lru=1, grid="pool4x10", kin_mode="yaw", tracks={"_current": None})
    dest_rel = "cache/train/clips/_current.pt"
    part = os.path.join(HERE, "latents_train_R35_partial.pt")
    S = {"Z": {k: [] for k in models}, "A": [], "OK": [], "T": [], "eid": [], "done": 0, "pull_s": [], "ctrl4": None, "starts": {}, "c4buf": None}
    if os.path.exists(part):
        S = torch.load(part, weights_only=False)
        print("resumed at clip", S["done"], flush=True)
    for i in range(S["done"], len(clips)):
        c = clips[i]
        pdt = pull(c, meta["md5"][c], dest_rel)
        store.tracks.clear(); store.tracks["_current"] = tracks_all[c]
        store._c.clear()
        d = store.get("_current")      # ClipStore.get: the cache file, yaw-mode kinematics from the native track (checked against pose_grid)
        assert len(d["pose"]) == meta["n_frames"][c], (c, len(d["pose"]), meta["n_frames"][c])
        ts, valid = window_starts(c, meta["n_frames"][c], cfg)
        cw = ClipWins(d, ts, 16)
        Z, A, OK, T = W.collect_latents(models, cw, len(ts))
        for k in models:
            S["Z"][k].append(Z[k])
        S["A"].append(A); S["OK"].append(OK); S["T"].append(T); S["eid"] += [c] * len(ts); S["pull_s"].append(pdt); S["starts"][c] = [int(x) for x in ts]
        if S["ctrl4"] is None:         # the first 64 windows' feats / act are kept until control 4 has run once
            f = torch.from_numpy(np.stack([cw.item(0, int(t))["feats"] for t in ts]))
            S["c4buf"] = (f, A.clone()) if S["c4buf"] is None else (torch.cat([S["c4buf"][0], f]), torch.cat([S["c4buf"][1], A]))
            if S["c4buf"][0].shape[0] >= 64:
                S["ctrl4"] = control4(models, S["c4buf"][0][:64], S["c4buf"][1][:64]); S["c4buf"] = None
                print("control 4:", {k: v["max_abs_diff"] for k, v in S["ctrl4"].items()}, flush=True)
        S["done"] = i + 1
        print(f"clip {i + 1:3d}/{len(clips)} {c[:8]} T={len(d['pose'])} valid={len(valid)} win={len(ts)} (all16 real={int(OK.all(1).sum())}) pull={pdt:.1f}s total={time.time() - t0:.0f}s", flush=True)
        if (i + 1) % 20 == 0 and i + 1 < len(clips):
            torch.save(S, part)
    out = {"Z1": {k: torch.cat(v) for k, v in S["Z"].items()}, "A_raw": torch.cat(S["A"]), "ok": torch.cat(S["OK"]), "t": np.concatenate(S["T"]), "eid": S["eid"],
           "N": len(S["eid"]), "tanitad_file": W.tanitad.__file__, "clips": clips, "clips_sha256": meta["clips_sha256"], "ctrl4": S["ctrl4"], "starts": S["starts"],
           "pull_s": S["pull_s"], "wall_s": time.time() - t0}
    assert out["ctrl4"] is not None
    for k in models:
        o = dict(out); o["Z1"] = {k: out["Z1"][k]}; o["ctrl4"] = {k: out["ctrl4"][k]}
        torch.save(o, os.path.join(HERE, f"latents_train_{k}.pt"))
    print("saved", {k: tuple(v.shape) for k, v in out["Z1"].items()}, tuple(out["A_raw"].shape), f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
