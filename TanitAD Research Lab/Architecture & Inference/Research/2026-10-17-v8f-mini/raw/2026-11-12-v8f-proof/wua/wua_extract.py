"""wua_extract.py -- card ADD155 (World Understanding Audit) model side, ONE RH checkpoint, eval-141, CPU, zero training.

  python -I wua_extract.py --seed 0 --out C:/Users/Admin/wua_work/ext

Per clip (ascending clip id of the eval cache manifest): adapter tokens (model._encode) -> TOK-GRID [T, 40*256] fp16, TOK-MEAN [T, 256], L1 = wm.z1, L2 = encoders[0](L1),
L3 = encoders[1](L2) (pointwise, w = 1). Rollouts (context 1) from anchors t = 10, 12, .. under the logged actions and under offsets (LON +-1 m/s^2, LAT +-0.1 rad/s, and 0 for the
identity control): L1 k in (5, 10), L2 k in (5, 15), L3 k in (3, 5, 15) where the clip holds the frames (else NaN). Writes per clip nothing; one npz per seed + json meta.
"""
import argparse, hashlib, json, os, sys, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
sys.dont_write_bytecode = True
import numpy as np
import torch

TREE = "F:/TanitAD-scratch/v8proof/rhacc10k/tree"
CK = {k: f"F:/TanitAD-scratch/v8proof/rhacc10k/inputs/RH_s{k}/ckpt_last.pt" for k in (0, 1, 2)}
CACHE = "E:/Projects/TanitAD-artifacts/sfresh_inputs/cache"
INPUTS = "F:/TanitAD-scratch/v8proof/rhacc10k/inputs/INPUTS.json"
KS = {1: (5, 10), 2: (5, 15), 3: (3, 5, 15)}
STRIDE = {1: 1, 2: 2, 3: 4}           # frames per level step
OFFS = {"base": (0.0, 0.0), "zero": (0.0, 0.0), "lonp": (1.0, 0.0), "lonm": (-1.0, 0.0), "latp": (0.0, 0.1), "latm": (0.0, -0.1)}
ANCHOR0, ANCHOR_STEP = 10, 2


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def load(seed):
    sys.path.insert(0, os.path.join(TREE, "stack"))
    from tanitad.models import v8f_mini as M
    want = json.load(open(INPUTS))[f"inputs/RH_s{seed}/ckpt_last.pt"]["sha256"]
    got = sha(CK[seed])
    assert got == want, ("checkpoint sha mismatch", got, want)
    ck = torch.load(CK[seed], map_location="cpu", weights_only=False)
    cfg = M.V8FMiniConfig(**ck["cfg"])
    m = M.V8FMini(cfg, M.P10Prior(torch.zeros(8), torch.ones(8), torch.zeros(2 * M.K30)))
    r = m.load_state_dict(ck["model"], strict=True)
    m.eval()
    assert m.wm is not None and getattr(cfg, "wm_state", "mean") == "mean" and int(cfg.wm_k) == 1
    return m, got


@torch.no_grad()
def rollouts(wm, z1, act_raw, t):
    """z1 [T, D] of one clip, act_raw [T, 2]. Returns {(cond, level, k): [D] or None}."""
    st = wm.stack
    T = z1.shape[0]
    out = {}
    hi = min(T - 1, t + 60)
    for cond, (da, dy) in OFFS.items():
        a = act_raw[t:hi].clone()
        a[:, 0] += da
        a[:, 1] += dy
        a_sc = wm.scale_actions(a)                                 # [n, 2], action of transition j -> j+1
        zseq = z1[t:hi + 1][None]
        enc = st.encode(zseq, a_sc[None]) if hi - t >= 4 else None
        for lv, ks in KS.items():
            kmax = max(k for k in ks if t + k * STRIDE[lv] <= T - 1) if any(t + k * STRIDE[lv] <= T - 1 for k in ks) else 0
            if kmax == 0 or (lv > 1 and enc is None):
                for k in ks:
                    out[(cond, lv, k)] = None
                continue
            if lv == 1:
                acts = a_sc[:kmax][None]
                z0 = z1[t][None]
            else:
                acts = enc.act[lv - 1][:, :kmax]
                z0 = enc.z[lv - 1][:, 0]
                if acts.shape[1] < kmax:
                    kmax = acts.shape[1]
                    acts = acts[:, :kmax]
            roll = st.rollout(lv, z0, acts, context=1)[0]          # [kmax, D]
            for k in ks:
                out[(cond, lv, k)] = roll[k - 1].float().numpy() if k <= kmax else None
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)
    torch.set_num_threads(4)
    m, cksha = load(a.seed)
    wm = m.wm
    man = [json.loads(l) for l in open(f"{CACHE}/eval/manifest.jsonl")]
    clips = sorted(r["clip"] for r in man)
    if a.limit:
        clips = clips[:a.limit]
    G, TM, Z = [], [], {1: [], 2: [], 3: []}
    CI, TT, V, A, Y = [], [], [], [], []
    R = {}
    AC, AT = [], []
    t0 = time.time()
    with torch.no_grad():
        for j, c in enumerate(clips):
            d = torch.load(f"{CACHE}/eval/clips/{c}.pt", map_location="cpu", weights_only=False)
            x = d["pool4x10"].float()[None]
            adapt = m._encode({"feats": x})[0]                     # [T, 40, 256]
            T = adapt.shape[0]
            z1 = wm.z1(adapt[None])[0]
            z2 = wm.stack.encoders[0](z1)
            z3 = wm.stack.encoders[1](z2)
            G.append(adapt.reshape(T, -1).to(torch.float16).numpy())
            TM.append(adapt.mean(1).float().numpy())
            for lv, z in ((1, z1), (2, z2), (3, z3)):
                Z[lv].append(z.float().numpy())
            CI.append(np.full(T, j, np.int32)); TT.append(np.arange(T, dtype=np.int16))
            v = d["v"].float(); acc = d["a"].float(); yr = d["yaw_rate"].float()
            V.append(v.numpy()); A.append(acc.numpy()); Y.append(yr.numpy())
            act_raw = torch.stack([acc, yr], -1)
            for t in range(ANCHOR0, T - 5, ANCHOR_STEP):
                ro = rollouts(wm, z1, act_raw, t)
                for key, val in ro.items():
                    R.setdefault(key, []).append(val if val is not None else np.full(z1.shape[1], np.nan, np.float32))
                AC.append(j); AT.append(t)
            del d, x, adapt
            if (j + 1) % 20 == 0:
                print(f"[wua s{a.seed}] {j + 1}/{len(clips)} {time.time() - t0:.0f}s", flush=True)
    os.makedirs(a.out, exist_ok=True)
    base = os.path.join(a.out, f"WUA_RH_s{a.seed}")
    np.save(base + "_tokgrid.npy", np.concatenate(G))
    rk = sorted(R)
    np.savez(base + ".npz", tokmean=np.concatenate(TM).astype(np.float32), L1=np.concatenate(Z[1]), L2=np.concatenate(Z[2]), L3=np.concatenate(Z[3]),
             clip_idx=np.concatenate(CI), t=np.concatenate(TT), v=np.concatenate(V), a=np.concatenate(A), yaw_rate=np.concatenate(Y), clips=np.array(clips),
             anc_clip=np.array(AC, np.int32), anc_t=np.array(AT, np.int16),
             roll_keys=np.array(["%s|%d|%d" % k for k in rk]), roll=np.stack([np.stack(R[k]) for k in rk]).astype(np.float32))
    meta = {"card": "ADD155", "seed": a.seed, "ckpt_sha256": cksha, "n_clips": len(clips), "n_frames": int(sum(len(x) for x in V)), "n_anchors": len(AC),
            "roll_keys": ["%s|%d|%d" % k for k in rk], "seconds": round(time.time() - t0, 1), "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "tool_sha256": sha(os.path.abspath(__file__))}
    meta["npz_sha256"] = sha(base + ".npz")
    json.dump(meta, open(base + ".json", "w"), indent=1)
    print(json.dumps({k: v for k, v in meta.items() if k != "roll_keys"}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
