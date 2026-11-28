"""LAT-TOK stage 1 (ZERO training, CPU): the frozen R35 adapter's per-frame tokens for the 141 TRAIN clips, three feature sets per frame.

Pre-registration: AMENDMENT_1_BCAL_DRAFT.md sha256 4e122aa5...f53834, section 2.2 item 1 (LAT-TOK).  Nothing here is fitted.
Frames: exactly the (clip, frame) cells of the banked latents_train_R35_s{k}.pt (141 clips x 17 window starts x 16 frames, `ok` frames, first window wins),
rebuilt with the collector's own window construction (collect_latents_train_R35.py ClipWins.item, hmax 16) from the pod's train cache.
Per frame and chassis: adapt = adapter(feats) + pos  [40, 256] (token order row-major r*10+c, 4 rows x 10 columns: pull_encode_pooled.py pool_field)
  (a) = adapt.mean(tokens)           256   (the present z1 input; z1 = proj(a))
  (b) = adapt.view(4,10,256).mean(rows) -> 10 column means, 2560
  (c) = adapt, all 40 tokens, 10240
Stored float32 [U, 13056] = a | b | c per chassis.
Controls (written to the log and ctrl json): clip md5 vs the manifest; recomputed act / frame_ok == banked A_raw / ok exactly; z1_from_mean(a) == wm.z1(adapt) bit-equal;
wm.z1 vs the banked Z1 (max abs diff; bit-equal count); duplicate-frame agreement.
Usage: python lattok_feats.py <outdir> <n_clips or 0 for all> [smoke]
"""
import dataclasses, hashlib, json, os, sys, time
import numpy as np
import torch

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
ROOT = "/root/lattok"
sys.path.insert(0, f"{ROOT}/stack")
from tanitad.models import v8f_mini as M  # noqa: E402
from tanitad.data import v8f_mini_data as D  # noqa: E402

OUT, NCL = sys.argv[1], int(sys.argv[2])
SMOKE = len(sys.argv) > 3
T0 = time.time()
CACHE = "/root/data/cache_pod"
os.makedirs(OUT, exist_ok=True)


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def load_model(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    names = {f.name for f in dataclasses.fields(M.V8FMiniConfig)}
    cfg = M.V8FMiniConfig(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in ck["cfg"].items() if k in names})
    model = M.V8FMini(cfg, M.P10Prior(torch.zeros(8), torch.ones(8), torch.zeros(2 * M.K30)))
    r = model.load_state_dict(ck["model"], strict=False)
    assert not r.missing_keys and set(r.unexpected_keys) <= {"hacc_w"}, (r.missing_keys, r.unexpected_keys)
    assert cfg.n_tok == 40 and model.wm.enc1 is None, (cfg.n_tok,)
    return model.eval()


meta = json.load(open(f"{ROOT}/data/clips141.json"))
clips_all = meta["clips"]
assert hashlib.sha256("\n".join(clips_all).encode()).hexdigest() == meta["clips_sha256"]
models, Ls = {}, {}
for k in (0, 1):
    models[k] = load_model(f"{ROOT}/data/ckpt_R35_s{k}.pt")
    for p in models[k].parameters():
        p.requires_grad_(False)
    Ls[k] = torch.load(f"{ROOT}/data/latents_train_R35_s{k}.pt", map_location="cpu", weights_only=False)
L0 = Ls[0]
assert list(Ls[1]["eid"]) == list(L0["eid"]) and np.array_equal(Ls[1]["t"], L0["t"]) and torch.equal(Ls[1]["ok"], L0["ok"]), "chassis files disagree on windows"
eid, tw, okw, Araw = list(L0["eid"]), np.asarray(L0["t"]), L0["ok"].bool(), L0["A_raw"]
W = okw.shape[1]
assert W == 16
clips = clips_all if NCL == 0 else clips_all[:NCL]
tracks_all = D.load_native_tracks(f"{CACHE}/native_tracks.pt")
store = D.ClipStore(CACHE, "train", lru=1, grid="pool4x10", kin_mode="yaw", tracks={c: tracks_all[c] for c in clips})
# rows: first appearance over (window, j) for the clips in scope, the same setdefault order amend1_kstep.py uses
cix = {c: i for i, c in enumerate(sorted(set(eid)))}
rows, row_of = [], {}
win_of_clip = {}
for w, e in enumerate(eid):
    win_of_clip.setdefault(e, []).append(w)
    if e not in clips:
        continue
    for j in range(W):
        if bool(okw[w, j]):
            key = (cix[e], int(tw[w]) - (W - 1) + j)
            if key not in row_of:
                row_of[key] = len(rows); rows.append(key)
U = len(rows)
DF = 256 + 2560 + 10240
log("clips", len(clips), "unique frames", U, "feature width", DF)
feat = {k: np.lib.format.open_memmap(f"{OUT}/feat_s{k}.npy", mode="w+", dtype=np.float32, shape=(U, DF)) for k in (0, 1)}
ctrl = {"n_clips": len(clips), "U": U, "md5_ok": 0, "act_ok_equal": True, "okmask_equal": True, "z1_from_mean_bit_equal": True,
        "z1_vs_banked_max_abs": {0: 0.0, 1: 0.0}, "z1_vs_banked_bit_equal_frac": {0: [0, 0], 1: [0, 0]}, "dup_max_abs": {0: 0.0, 1: 0.0}, "dup_checked": 0, "z1_rel_l2_vs_banked_sums": {0: [0.0, 0.0], 1: [0.0, 0.0]}}


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def item(d, t, hmax=16):
    T = len(d["pose"])
    idx = np.arange(t - hmax + 1, t + 1); okf = idx >= 0; idc = np.clip(idx, 0, T - 1)
    okf &= np.isfinite(d["p10f"][idc]).all(1) | (idx < 1)
    return {"feats": d["feats"][idc], "frame_ok": okf, "act": np.stack([np.nan_to_num(d["a"][idc]), d["ch1"][idc]], -1).astype(np.float32)}


with torch.no_grad():
    for i, c in enumerate(clips):
        p = f"{CACHE}/train/clips/{c}.pt"
        ctrl["md5_ok"] += int(md5(p) == meta["md5"][c])       # NOT fatal: the pod cache is a RE-ENCODE (compare_box2.jsonl: pool4x10 rel_l2 ~0.024 vs box 2), so 0 matches are expected
        d = store.get(c); store._c.clear()
        assert len(d["pose"]) == meta["n_frames"][c], c
        ws = win_of_clip[c]
        its = [item(d, int(tw[w])) for w in ws]
        feats = torch.from_numpy(np.stack([it["feats"] for it in its]))
        act = torch.from_numpy(np.stack([it["act"] for it in its])); okf = torch.from_numpy(np.stack([it["frame_ok"] for it in its]))
        ctrl["okmask_equal"] &= bool(torch.equal(okf, okw[ws]))
        fin = okw[ws]
        ctrl["act_ok_equal"] &= bool(np.array_equal(act[fin].numpy(), Araw[ws][fin].numpy(), equal_nan=True))
        for k in (0, 1):
            m = models[k]
            adapt = m.adapter(feats.float()) + m.pos[None, None]                       # [nw, 16, 40, 256]
            z1 = m.wm.z1(adapt)
            am = adapt.mean(2)
            ctrl["z1_from_mean_bit_equal"] &= bool(torch.equal(m.wm.z1_from_mean(am), z1))
            zb = Ls[k]["Z1"][f"R35_s{k}"][ws].float()
            df = (z1 - zb).abs()[fin]
            ctrl["z1_rel_l2_vs_banked_sums"][k][0] += float(((z1 - zb)[fin] ** 2).sum()); ctrl["z1_rel_l2_vs_banked_sums"][k][1] += float((zb[fin] ** 2).sum())
            ctrl["z1_vs_banked_max_abs"][k] = max(ctrl["z1_vs_banked_max_abs"][k], float(df.max()))
            ctrl["z1_vs_banked_bit_equal_frac"][k][0] += int((df == 0).all(-1).sum()) if df.ndim > 1 else 0
            ctrl["z1_vs_banked_bit_equal_frac"][k][1] += int(fin.sum())
            col = adapt.view(adapt.shape[0], W, 4, 10, -1).mean(2).flatten(2)             # 10 column means
            full = torch.cat([am, col, adapt.flatten(2)], -1).numpy()                      # [nw, 16, 13056]
            seen = {}
            for wi, w in enumerate(ws):
                for j in range(W):
                    if not bool(okw[w, j]):
                        continue
                    key = (cix[c], int(tw[w]) - (W - 1) + j)
                    r = row_of[key]
                    if key in seen:
                        ctrl["dup_max_abs"][k] = max(ctrl["dup_max_abs"][k], float(np.abs(full[wi, j] - seen[key]).max())); ctrl["dup_checked"] += 1
                    else:
                        feat[k][r] = full[wi, j]; seen[key] = full[wi, j]
        if (i + 1) % 10 == 0 or SMOKE or i + 1 == len(clips):
            log(f"clip {i + 1}/{len(clips)}", {k: (v if not isinstance(v, dict) else v) for k, v in ctrl.items() if k in ('md5_ok', 'okmask_equal', 'act_ok_equal', 'z1_from_mean_bit_equal', 'z1_vs_banked_max_abs')})
for k in (0, 1):
    feat[k].flush()
ctrl["z1_rel_l2_vs_banked"] = {k: (v[0] / v[1]) ** 0.5 for k, v in ctrl["z1_rel_l2_vs_banked_sums"].items()}
ctrl["z1_vs_banked_bit_equal_frac"] = {k: v[0] / max(v[1], 1) for k, v in ctrl["z1_vs_banked_bit_equal_frac"].items()}
np.save(f"{OUT}/rows.npy", np.array(rows, dtype=np.int64))
json.dump(ctrl, open(f"{OUT}/ctrl_feats.json", "w"), indent=1)
log("ctrl", json.dumps(ctrl))
assert ctrl["okmask_equal"] and ctrl["act_ok_equal"], "recomputed frame_ok / act differ from the banked windows"
log("ZZ FEATS DONE ZZ")
