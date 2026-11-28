"""SLOWWM beside B1: per-frame inputs of the 141 WMACT TRAIN clips, in the formats the read package already uses for eval clips. NOT RUN in the build (the clips are on box 2; this session does no box-2 work).
  python decod_build_train.py --out-dir DIR --adapter-run 0=<R35H_s0 run dir> [--adapter-run 1=<R35H_s1 run dir>] [--basis slow=<basis_slow.npz>] [--basis pca=<basis_pca.npz>]
  -> DIR/frames_train_s{seed}.npz   (clips, T, zmean, act, ok, adapter_sha: ``lr_lib.build_eval_frames``' format, zmean = ``lr_lib.zmean_of``)
     DIR/ubasis_train_{kind}.npz    (clips, T, ubasis, kind, basis_sha: the hook's ``build_eval_basis_frames`` format, u = ``slowwm_hook.ubasis_of_feats``)
The stream is ``wmact/collect_latents_train.py`` unchanged in its pull (sftp, md5 against box 2's manifest, one scratch path overwritten per clip) and its ClipStore read; only what is kept differs. ``clip_inputs`` is
the shared per-clip function and is tested against the banked eval frames (tests/test_decod_wmact.py). Run it only once box 2 may be read; each clip is pulled once for all adapters and bases."""
import json
import os
import sys
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import decod_lib as DL  # noqa: E402
import numpy as np  # noqa: E402

LR, SH = DL.LR, DL.SH


def clip_inputs(d, adapters, bases):
    """One clip (a ``ClipStore.get`` dict) -> (zmean {seed: [T, d]}, ubasis {kind: [T, 256]}, act [T, 2], ok [T]); act and ok exactly as ``lr_lib.build_eval_frames``."""
    z = {s: LR.zmean_of(m, d["feats"]).numpy() for s, m in adapters.items()}
    u = {k: SH.ubasis_of_feats(B, d["feats"]) for k, B in bases.items()}
    act = np.stack([np.nan_to_num(d["a"]), d["ch1"]], -1).astype(np.float32)
    ok = np.isfinite(d["p10f"]).all(1)
    ok[0] = True
    return z, u, act, ok


def main(argv):
    import torch
    import collect_latents_train as CT          # the box-2 stream's pull / ClipStore constants (imports wmact_lib)
    D = CT.D
    opts = {"--adapter-run": [], "--basis": []}
    it = iter(argv)
    for a in it:
        if a in opts:
            opts[a].append(next(it))
        elif a == "--out-dir":
            out_dir = next(it)
    adapters, bases, sha = {}, {}, {}
    for s in opts["--adapter-run"]:
        seed, run = s.split("=", 1)
        m, _, _, _ = LR.load_arm(run)
        adapters[int(seed)], sha[int(seed)] = m, LR.adapter_sha(m)
    for s in opts["--basis"]:
        kind, path = s.split("=", 1)
        bases[kind] = DL.S.SH.SB.load_basis(path)
        assert bases[kind]["kind"] == kind, (kind, bases[kind]["kind"])
    os.makedirs(out_dir, exist_ok=True)
    meta = json.load(open(os.path.join(CT.TS, "clips141.json")))
    clips = meta["clips"]
    tracks_all = D.load_native_tracks(f"{CT.W.SF}/inputs/native_tracks.pt")
    store = D.ClipStore(os.path.join(CT.TS, "cache"), "train", lru=1, grid="pool4x10", kin_mode="yaw", tracks={"_current": None})
    dest_rel = "cache/train/clips/_current.pt"
    Zs, Us, As, Os, Ts = {s: [] for s in adapters}, {k: [] for k in bases}, [], [], []
    t0 = time.time()
    for i, c in enumerate(clips):
        CT.pull(c, meta["md5"][c], dest_rel)
        store.tracks.clear()
        store.tracks["_current"] = tracks_all[c]
        store._c.clear()
        d = store.get("_current")
        assert len(d["pose"]) == meta["n_frames"][c], (c, len(d["pose"]), meta["n_frames"][c])
        z, u, act, ok = clip_inputs(d, adapters, bases)
        for s in z:
            Zs[s].append(z[s])
        for k in u:
            Us[k].append(u[k])
        As.append(act), Os.append(ok), Ts.append(len(act))
        print(f"clip {i + 1:3d}/{len(clips)} {c[:8]} T={len(act)} total={time.time() - t0:.0f}s", flush=True)
    for s in adapters:
        part = os.path.join(out_dir, f"frames_train_s{s}.part.npz")
        np.savez(part, clips=np.array(clips), T=np.array(Ts), zmean=np.concatenate(Zs[s]), act=np.concatenate(As), ok=np.concatenate(Os), adapter_sha=sha[s])
        os.replace(part, os.path.join(out_dir, f"frames_train_s{s}.npz"))
    for k, B in bases.items():
        part = os.path.join(out_dir, f"ubasis_train_{k}.part.npz")
        np.savez(part, clips=np.array(clips), T=np.array(Ts), ubasis=np.concatenate(Us[k]), kind=np.array(B["kind"]), basis_sha=np.array(B["sha"]))
        os.replace(part, os.path.join(out_dir, f"ubasis_train_{k}.npz"))
    print("done", f"{time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
