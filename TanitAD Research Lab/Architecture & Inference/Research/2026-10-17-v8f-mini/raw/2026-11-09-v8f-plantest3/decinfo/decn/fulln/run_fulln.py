"""DECN full-n probe (the card's RISES trigger): the deciding inputs only (R0, R2 at M2b s0 and s1), every trained clip, streamed read-only from box 2 one clip at a time as DECN (no GPU or RAM load on box 2), CPU on the dev box.
Usage: python run_fulln.py a   (stream the remaining clips -> shards)   |   python run_fulln.py b   (readers at N4565 -> runs/<rung>_<tag>_N4565.json). Exit codes of stage a as DECN: 0, 10 network stop, 3 md5 mismatch. Resumable."""
import json, os, sys, time, traceback
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
DECN = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, DECN)
import numpy as np
import run_decn as RD

DN, DL, W, D_ = RD.DN, RD.DL, RD.W, RD.D_
LAT2 = "F:/TanitAD-scratch/decn_scratch/fulln/lat"
STREAM = "F:/TanitAD-scratch/decn_scratch/fulln/stream"
RUNS = os.path.join(HERE, "runs")
RD.STREAM = STREAM
NAME = "N4565"
TAGS = ("M2b_s0", "M2b_s1")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def lists():
    c141, ce = DN.clip_lists()
    cf = json.load(open(os.path.join(HERE, "clips_full.json"), encoding="utf-8"))
    return list(c141["clips"]) + list(ce["clips"]), cf


def shard(c):
    p = f"{DN.LAT}/{c}.npz"
    return p if os.path.exists(p) else f"{LAT2}/{c}.npz"


def stage_a():
    os.makedirs(LAT2, exist_ok=True); os.makedirs(STREAM, exist_ok=True)
    first, cf = lists()
    assert len(first) == 1000 and all(os.path.exists(f"{DN.LAT}/{c}.npz") for c in first), "DECN's 1,000 shards are required"
    ev_clips = {f[:-3] for f in os.listdir(f"{W.SF}/cache/eval/clips") if f.endswith(".pt")}
    assert not (set(cf["clips"]) & ev_clips) and not (set(cf["clips"]) & set(first))
    ms = {t: DL.load_any(DL.M2B[t])[0] for t in TAGS}
    _, cfg, _ = W.load_model(W.CKPT["M2b_s0"])
    assert cfg.hmax == 16 and cfg.hist == 4 and cfg.window_start_effective == 3
    tracks_all = D_.load_native_tracks(f"{W.SF}/inputs/native_tracks.pt")
    store = D_.ClipStore(STREAM + "/cache", "train", lru=1, grid="pool4x10", kin_mode="yaw", tracks={"_current": None})
    t0, done = time.time(), 0
    for i, c in enumerate(cf["clips"]):
        sp_ = f"{LAT2}/{c}.npz"
        if os.path.exists(sp_):
            continue
        DL.wait_ram()
        try:
            RD.pull(c, cf["md5"][c])
        except RuntimeError as e:
            open(os.path.join(HERE, "STREAM_STOPPED.txt"), "w").write(f"md5 mismatch: {e}\n"); log("STOP", e); return 3
        except ConnectionError as e:
            open(os.path.join(HERE, "STREAM_STOPPED.txt"), "w").write(f"network stop at clip {i}: {e}\n"); log("STOP", e); return 10
        store.tracks.clear(); store.tracks["_current"] = tracks_all[c]; store._c.clear()
        d = store.get("_current")
        assert len(d["pose"]) == cf["n_frames"][c]
        cw = RD.ClipWins(d)
        items = [it for it in (cw.item(int(t)) for t in RD.window_starts(c, cf["n_frames"][c], cfg)) if bool(it["frame_ok"].all()) and it["idx"].max() < len(d["v"])]
        arrs = {}
        if items:
            feats = np.stack([it["feats"] for it in items])
            for tag, model in ms.items():
                z1, pooled = DL.vector_inputs(model, feats)
                arrs[f"z1_{tag}"], arrs[f"pooled_{tag}"] = z1.numpy().astype(np.float32), pooled.numpy().astype(np.float32)
            arrs["v"] = np.stack([d["v"][it["idx"]].astype(np.float64) for it in items])
            arrs["yaw"] = np.stack([it["act"][:, 1].astype(np.float64) for it in items])
        else:
            for tag in ms:
                arrs[f"z1_{tag}"], arrs[f"pooled_{tag}"] = np.zeros((0, 16, 128), np.float32), np.zeros((0, 16, 256), np.float32)
            arrs["v"], arrs["yaw"] = np.zeros((0, 16)), np.zeros((0, 16))
        with open(sp_ + ".tmp", "wb") as fh:
            np.savez(fh, **arrs)
        os.replace(sp_ + ".tmp", sp_)
        done += 1
        if done % 50 == 1:
            log(f"clip {i + 1}/{len(cf['clips'])} streamed this run {done}, {time.time() - t0:.0f}s")
    try:
        os.remove(os.path.join(STREAM, RD.DEST_REL))             # own scratch clip, source on box 2
    except OSError:
        pass
    log("stage a complete", done)
    return 0


def stage_b():
    os.makedirs(RUNS, exist_ok=True)
    first, cf = lists()
    order = first + list(cf["clips"])
    assert len(order) == 4512 and len(set(order)) == 4512 and all(os.path.exists(shard(c)) for c in order), "shards incomplete"
    T = DN.load_tune_npz()
    ev_clips = {f[:-3] for f in os.listdir(f"{W.SF}/cache/eval/clips") if f.endswith(".pt")}
    for rung, tag, kind in [(r, t, k) for t in TAGS for r, k in (("R0", "z1"), ("R2", "pooled"))]:
        out = os.path.join(RUNS, f"{rung}_{tag}_{NAME}.json")
        if os.path.exists(out):
            continue
        DL.wait_ram()
        key, t0 = f"{kind}_{tag}", time.time()
        xs, ci, vs, ys = [], [], [], []
        for i, c in enumerate(order):
            z = np.load(shard(c))
            a = z[key]
            xs.append(a); ci += [i] * len(a); vs.append(z["v"]); ys.append(z["yaw"])
        X_tr, ci_tr = np.concatenate(xs), np.array(ci)
        del xs
        D, X, ids_tr = DN.assemble(T, key, X_tr, ci_tr, order, len(order), (np.concatenate(vs), np.concatenate(ys)))
        del X_tr
        DN.split_guard_n(set(map(str, D.eid[:D.nf])), set(map(str, D.eid[D.nf:])), DL.es_ids_of(set(map(str, D.eid[:D.nf]))), ev_clips, ids_tr)
        log("unit", rung, tag, NAME, "fit windows", D.nf, "episodes", D.n_fit_episodes, "held windows", D.n - D.nf)
        rows, planted = DN.read_unit(D, X, rung, tag, NAME, log)
        RD.jdump(dict(rung=rung, tag=tag, N=NAME, tier="T0", rows=rows, planted=planted, meta=dict(held_digest=D.held_digest, n_clips=len(order), n_fit_windows=int(D.nf), n_held_windows=int(D.n - D.nf), n_fit_episodes=D.n_fit_episodes, n_held_episodes=D.n_held_episodes, wall_s=time.time() - t0)), out)
        log("saved", out, f"{time.time() - t0:.0f}s")
        del D, X
    log("stage b complete")
    return 0


if __name__ == "__main__":
    try:
        rc = stage_a() if sys.argv[1] == "a" else stage_b()
    except SystemExit as e:
        rc = int(e.code)
    except Exception:
        log("FAILED", traceback.format_exc()); rc = 1
    sys.exit(rc)
