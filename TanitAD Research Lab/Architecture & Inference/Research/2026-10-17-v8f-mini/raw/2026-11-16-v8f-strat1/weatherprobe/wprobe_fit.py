"""wprobe_fit.py -- PREREG_WEATHERPROBE (pod side).  Frozen S1 adapter -> attention-pooling probe for weather; fit on TRAIN clips, scored (logits dumped) on the eval grid once.
Runs on the LoRA pod GPU (inference-only extraction + 0.27 M-parameter probe; GPU memory < 2 GB).  Writes wprobe_out.npz with eval logits per arm.  Not a scorer: scoring is numpy on the dev box."""
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn as nn

W = "/root/wprobe"
TRAIN_CACHE, EVAL_CACHE = "/root/data/cache_pod/train/clips", "/root/data/cache_pod/eval/clips"
DEV = "cuda"
SEEDS = (0, 1)
EPOCH_CHOICES = (2, 4, 8)
NW = 8                      # NOW frames per train clip
SMOKE = len(sys.argv) > 1 and sys.argv[1] == "smoke"
LOG = open(f"{W}/wprobe_log{'_smoke' if SMOKE else ''}.txt", "a")


def log(*a):
    s = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.write(s + "\n")
    LOG.flush()


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


tab = dict(np.load(f"{W}/strat_labels_table.npz"))
assert sha(f"{W}/strat_labels_table.npz") == "8055a321df6db1725ea17abde2999d48bcb68d43f1a480a69ed42dd0cf184eff"
grid = json.load(open(f"{W}/eval_grid_windows.json"))
assert sha(f"{W}/eval_grid_windows.json") == "659c3b65e364ea5de87a4e17d076eba5645df95cdcf39d8abc08caab1b2d594c"
join = dict(np.load(f"{W}/join_eval141.npz"))
eval_clips = [str(c) for c in join["clip"]]
assert eval_clips == list(grid["order"]) == [str(c) for c in tab["eval_clip"]]
train_clips = [str(c) for c in tab["train_clip"]]

# ---------------------------------------------------------------- disjointness (must read 0)
disj = {"table_train_vs_eval": len(set(train_clips) & set(eval_clips)),
        "train_cache_files_vs_eval": len({f[:-3] for f in os.listdir(TRAIN_CACHE)} & set(eval_clips)),
        "train_cache_files_vs_grid_order": len({f[:-3] for f in os.listdir(TRAIN_CACHE)} & set(grid["order"]))}
log("DISJOINTNESS", disj)
assert all(v == 0 for v in disj.values()), "train/eval overlap: FAIL"

# eval cache identity vs the dev box copy the read used
loc = {j["clip"]: j["content_md5"] for j in map(json.loads, open(f"{W}/eval_manifest_local.jsonl"))}
pod = {j["clip"]: j["content_md5"] for j in map(json.loads, open("/root/data/cache_pod/eval/manifest.jsonl"))}
eval_cache_same = sum(loc[c] == pod.get(c) for c in eval_clips)
log("eval cache content_md5 equal to the dev-box copy for", eval_cache_same, "of", len(eval_clips), "(the pod cache was re-encoded on the pod: bytes differ)")


def load_clip(path):
    d = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    return d["pool4x10"]


# ---------------------------------------------------------------- windows
def window_frames(t):
    return np.arange(t - 3, t + 1)


def build_train(clips_idx):
    """-> feats uint8-viewed fp8 [N,4,40,1024], y [N], clip_of [N]; skips mismatching clips"""
    sel, skipped = [], 0
    for c in clips_idx:
        n = int(tab["train_n"][c])
        ts = np.unique(np.round(np.linspace(3, n - 1, NW)).astype(int))
        o = int(tab["train_off"][c])
        for t in ts:
            r = o + int(t)
            if tab["train_flag"][r] == 0 and tab["train_weather_valid"][r]:
                sel.append((c, int(t), int(tab["train_weather_id"][r])))
    log("train windows", len(sel), "from", len(clips_idx), "clips")
    by = {}
    for i, (c, t, y) in enumerate(sel):
        by.setdefault(c, []).append(i)
    N = len(sel)
    feats = torch.empty((N, 4, 40, 1024), dtype=torch.float8_e4m3fn)
    ok = np.ones(N, bool)

    def work(c):
        p = f"{TRAIN_CACHE}/{train_clips[c]}.pt"
        x = load_clip(p)
        if x.shape[0] != int(tab["train_n"][c]):
            return c, False
        for i in by[c]:
            t = sel[i][1]
            feats[i] = x[t - 3:t + 1]
        return c, True
    t0 = time.time()
    with ThreadPoolExecutor(16) as ex:
        for k, (c, good) in enumerate(ex.map(work, list(by))):
            if not good:
                skipped += 1
                for i in by[c]:
                    ok[i] = False
            if k % 500 == 0:
                log("  read", k, "of", len(by), f"{time.time() - t0:.0f}s")
    y = np.array([s[2] for s in sel], np.int64)
    cl = np.array([s[0] for s in sel], np.int64)
    log("skipped clips (frame count != train_n):", skipped)
    return feats[torch.from_numpy(ok)], y[ok], cl[ok], skipped


def build_eval():
    ci, ts = [], []
    for i, key in enumerate(grid["order"]):
        for t in grid["windows"][key]:
            ci.append(i)
            ts.append(int(t))
    ci, ts = np.array(ci), np.array(ts)
    feats = torch.empty((len(ci), 4, 40, 1024), dtype=torch.float8_e4m3fn)
    cache = {}
    for k in range(len(ci)):
        c = int(ci[k])
        if c not in cache:
            cache = {c: load_clip(f"{EVAL_CACHE}/{eval_clips[c]}.pt")}
        feats[k] = cache[c][ts[k] - 3:ts[k] + 1]
    return feats, ci, ts


# ---------------------------------------------------------------- frozen adapter and probe
class Adapter(nn.Module):
    def __init__(self, sd=None, rand_seed=None):
        super().__init__()
        self.ln, self.lin = nn.LayerNorm(1024), nn.Linear(1024, 256)
        self.pos, self.time = nn.Parameter(torch.zeros(40, 256)), nn.Parameter(torch.zeros(4, 256))
        if sd is not None:
            self.ln.weight.data, self.ln.bias.data = sd["adapter.0.weight"].float(), sd["adapter.0.bias"].float()
            self.lin.weight.data, self.lin.bias.data = sd["adapter.1.weight"].float(), sd["adapter.1.bias"].float()
            self.pos.data, self.time.data = sd["pos"].float(), sd["time"].float()
        else:
            g = torch.Generator().manual_seed(rand_seed)
            torch.manual_seed(rand_seed)
            self.ln, self.lin = nn.LayerNorm(1024), nn.Linear(1024, 256)
            self.pos.data, self.time.data = torch.randn(40, 256, generator=g) * 0.02, torch.randn(4, 256, generator=g) * 0.02
        for p in self.parameters():
            p.requires_grad_(False)

    def tokens(self, f8):                      # [B,4,40,1024] fp8 -> [B,4,40,256] (adapter + pos + time)
        f = f8.float()
        return self.lin(self.ln(f)) + self.pos[None, None] + self.time[None, :, None]

    def tokens_nopos_time(self, f8):           # = _encode: adapter + pos, no time
        return self.lin(self.ln(f8.float())) + self.pos[None]


class Probe(nn.Module):
    def __init__(self, seed):
        super().__init__()
        torch.manual_seed(seed)
        self.n1 = nn.LayerNorm(256)
        self.q = nn.Parameter(torch.randn(1, 1, 256) * 0.02)
        self.att = nn.MultiheadAttention(256, 4, batch_first=True)
        self.n2 = nn.LayerNorm(256)
        self.out = nn.Linear(256, 4)

    def forward(self, tok):                    # [B,160,256]
        x = self.n1(tok)
        o, _ = self.att(self.q.expand(x.shape[0], -1, -1), x, x, need_weights=False)
        return self.out(self.n2(o[:, 0]))


def fit(adapter, feats, y, epochs, seed=0, bs=256, lr=1e-3, wd=0.01, warm=100):
    adapter = adapter.to(DEV)
    probe = Probe(seed).to(DEV)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=wd)
    N = len(y)
    steps = epochs * ((N + bs - 1) // bs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + np.cos(np.pi * min(1.0, s / max(1, steps - 1)))))
    rng = np.random.RandomState(seed)
    yt = torch.from_numpy(y)
    for ep in range(epochs):
        perm = rng.permutation(N)
        tot = 0.0
        for i in range(0, N, bs):
            ix = torch.from_numpy(perm[i:i + bs])
            with torch.no_grad():
                tok = adapter.tokens(feats[ix].to(DEV)).flatten(1, 2)
            loss = nn.functional.cross_entropy(probe(tok), yt[ix].to(DEV))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss) * len(ix)
        log(f"    ep {ep + 1}/{epochs} loss {tot / N:.4f}")
    return probe


@torch.no_grad()
def predict(adapter, probe, feats, bs=256):
    probe.eval()
    out = []
    for i in range(0, len(feats), bs):
        tok = adapter.tokens(feats[i:i + bs].to(DEV)).flatten(1, 2)
        out.append(probe(tok).float().cpu().numpy())
    return np.concatenate(out)


def margin(logits, y):
    p = logits.argmax(1)
    return float((p == y).mean() - (y == 0).mean())      # constant = clear (class 0) = majority of train and eval


def main():
    t0 = time.time()
    clips_idx = list(range(len(train_clips)))
    if SMOKE:
        clips_idx = clips_idx[:300]
    feats, y, cl, skipped = build_train(clips_idx)
    log("train feats", tuple(feats.shape), "class counts", np.bincount(y, minlength=4).tolist(), f"{time.time() - t0:.0f}s")
    ef_pod, eci, ets = build_eval()
    L = torch.load(f"{W}/eval_local_feats_u8.pt", weights_only=False)
    assert np.array_equal(L["ci"], eci) and np.array_equal(L["t"], ets)
    ef = L["feats"].view(torch.float8_e4m3fn)            # PRIMARY: the dev-box eval cache the T0 read used; ef_pod (pod-encoded) is scored beside
    meta_diff = float((ef.float() - ef_pod.float()).abs().mean()), float(ef.float().abs().mean())
    log("eval feats", tuple(ef.shape), "mean|local-pod|, mean|local|", meta_diff, f"{time.time() - t0:.0f}s")
    out = {"eval_ci": eci, "eval_t": ets}
    meta = {"eval_feat_diff_mean_abs_and_mean_abs": list(meta_diff), "disjointness": disj, "eval_cache_md5_equal": int(eval_cache_same), "skipped_clips": int(skipped), "n_train_windows": int(len(y)),
            "train_class_counts": np.bincount(y, minlength=4).tolist(), "smoke": SMOKE, "eval_feat_mean_abs_diff_local_vs_pod": None, "script_sha256": sha(__file__), "chosen_epochs": {}, "holdout": {}}
    # the train-only 80/20 clip split
    uc = np.unique(cl)
    perm = np.random.RandomState(0).permutation(len(uc))
    held = set(uc[perm[: len(uc) // 5]].tolist())
    is_held = np.array([c in held for c in cl])
    ckpts = {}
    for s in SEEDS:
        sd = torch.load(f"{W}/S1_s{s}_ckpt_last.pt", map_location="cpu", weights_only=False)["model"]
        ckpts[s] = sd
    # instrument identity: the stored ctx_head on this script's adapter reproduces the T0 weather argmax
    for s in SEEDS:
        sd = ckpts[s]
        ad = Adapter(sd).to(DEV)
        z = np.load(f"{W}/eval_STRAT1-S1_s{s}_real.npz", allow_pickle=True)
        assert np.array_equal(z["win_ci"], eci) and np.array_equal(z["win_t"], ets)
        with torch.no_grad():
            lg = []
            for i in range(0, len(ef), 256):
                tok = ad.tokens_nopos_time(ef[i:i + 256, 3].to(DEV))        # NOW frame, adapter + pos (no time), mean over 40 tokens
                h = tok.mean(1)
                h = torch.nn.functional.gelu(h @ sd["ctx_head.0.weight"].float().to(DEV).T + sd["ctx_head.0.bias"].float().to(DEV))
                lg.append((h @ sd["ctx_head.2.weight"].float().to(DEV).T + sd["ctx_head.2.bias"].float().to(DEV)).cpu().numpy())
            lg = np.concatenate(lg)
        a = lg[:, 25:29].argmax(1)
        b = z["ctx_logits"][:, 25:29].argmax(1)
        meta[f"identity_s{s}_weather_argmax_agree"] = float((a == b).mean())
        meta[f"identity_s{s}_max_abs_logit_diff"] = float(np.abs(lg - z["ctx_logits"]).max())
        log(f"IDENTITY s{s}: weather argmax agreement {meta[f'identity_s{s}_weather_argmax_agree']:.4f}, max |dlogit| {meta[f'identity_s{s}_max_abs_logit_diff']:.4g}")
        del ad
    # shuffled labels (R1): permute labels between train clips, seed 0
    rs = np.random.RandomState(0)
    pc = rs.permutation(len(uc))
    cmap = {int(uc[i]): int(uc[pc[i]]) for i in range(len(uc))}
    rows_of = {int(c): np.where(cl == c)[0] for c in uc}
    y_shuf = y.copy()
    for c in uc:
        dst = rows_of[int(c)]
        src = rows_of[cmap[int(c)]]
        y_shuf[dst] = y[src[np.arange(len(dst)) % len(src)]]
    for s in SEEDS:
        ad = Adapter(ckpts[s])
        # choose E on the train-only split
        scores = {}
        fi, hi = torch.from_numpy(np.where(~is_held)[0]), torch.from_numpy(np.where(is_held)[0])
        for E in EPOCH_CHOICES:
            log(f"S1_s{s}: holdout fit E={E} on {len(fi)} windows")
            pr = fit(ad, feats[fi], y[fi.numpy()], E)
            lgh = predict(ad, pr, feats[hi])
            scores[E] = margin(lgh, y[hi.numpy()])
            log(f"S1_s{s}: E={E} held-out accuracy-minus-constant {scores[E]:+.4f}")
            del pr
        E = max(EPOCH_CHOICES, key=lambda e: (scores[e], -e))
        meta["chosen_epochs"][f"S1_s{s}"], meta["holdout"][f"S1_s{s}"] = E, {str(k): v for k, v in scores.items()}
        log(f"S1_s{s}: E chosen {E}")
        pr = fit(ad, feats, y, E)
        out[f"S1_s{s}_real"] = predict(ad, pr, ef)
        out[f"S1_s{s}_real_podenc"] = predict(ad, pr, ef_pod)
        log(f"S1_s{s}: final fit done; eval logits stored (scored on the dev box)")
        del pr
        pr = fit(ad, feats, y_shuf, E)
        out[f"S1_s{s}_R1_shuffle"] = predict(ad, pr, ef)
        del pr
        if s == 0:
            ad2 = Adapter(None, rand_seed=0)
            pr = fit(ad2, feats, y, meta["chosen_epochs"]["S1_s0"])
            out["S1_s0_R2_randadapter"] = predict(ad2, pr, ef)
            del pr
        np.savez_compressed(f"{W}/wprobe_out{'_smoke' if SMOKE else ''}.npz", **out)
        json.dump(meta, open(f"{W}/wprobe_meta{'_smoke' if SMOKE else ''}.json", "w"), indent=1)
    log("ZZ WPROBE DONE ZZ", f"{time.time() - t0:.0f}s", "peak GPU MiB", torch.cuda.max_memory_allocated() / 2**20)


if __name__ == "__main__":
    main()
