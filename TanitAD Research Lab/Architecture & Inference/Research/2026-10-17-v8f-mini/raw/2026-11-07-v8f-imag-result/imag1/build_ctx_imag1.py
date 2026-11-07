"""IMAG1 model side: turns the two M2b models + the cached level-1 latents into the arrays of a Ctx (imag1_lib). CPU, no training, no probe.
Per dataset (eval141 / train): frame bank (all 141 clips), labels (labels_imag1_*.npz), per tag and level the per-frame encoded latents BZ, the window start state Z0, the rollout PRED under the
logged actions, the encoded true future ORA, the +0.2 s offset latent OFF (control 2), the rollouts under the four action contrasts CON, the IWM action-bank ranks, and (eval141) the random-init
bank latents. Usage: python build_ctx_imag1.py eval141|train   (WMACT_CODE_ROOT set). Writes F:/TanitAD-scratch/imag1_ctx_<ds>.pkl and ctx_<ds>.report.json."""
import hashlib, json, os, pickle, sys, time
sys.dont_write_bytecode = True
import numpy as np
import torch
import imag1_model as IM
import labels_imag1 as LB
import imag1_lib as IL
W, LBb = IM.W, LB.LBb

F = 16
CONTRAST = {"lonp": (0, +1.0), "lonm": (0, -1.0), "latp": (1, +0.25), "latm": (1, -0.25)}
N_CAND = 16


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


@torch.no_grad()
def pack(wm, Z1, A_sc):
    """per level: Z0 [n, D], ORA[k], PRED[k] for k = 1..3 (HJEPAStack.encode / rollout, context 1: the planning form, as WMACT / LATSEM2)."""
    st = wm.stack
    enc = st.encode(Z1, A_sc)
    z0, ora, pred = {}, {}, {}
    for l in IL.LEVELS:
        acts = A_sc[:, :3] if l == 1 else enc.act[l - 1][:, :3]
        roll = st.rollout(l, enc.z[l - 1][:, 0], acts)
        z0[l] = enc.z[l - 1][:, 0]
        for k in IL.KS:
            ora[(l, k)] = enc.z[l - 1][:, k]
            pred[(l, k)] = roll[:, k - 1]
    return z0, ora, pred


def main(ds):
    t0 = time.time()
    path = LB.LATENTS[ds]
    B = LBb.build_bank_train(path)
    lab = np.load(f"labels_imag1_{ds}.npz")
    assert (lab["beid"] == B["beid"].astype("U36")).all() and (lab["bg"] == B["bg"]).all(), "labels do not match the bank"
    L, eid, ok, A_raw, G, key2b, bw, bf, beid = B["L"], B["eid"], B["ok"], B["A_raw"], B["G"], B["key2b"], B["bw"], B["bf"], B["beid"]
    sel = B["wkeep"][ok[B["wkeep"]].all(1)]
    nw = len(sel)
    bidx = np.array([[key2b[(eid[i], int(G[i, f]))] for f in range(F)] for i in sel])
    Y = {q: np.asarray(lab[q], np.float64) for q in IL.TARGETS}
    Yw = {q: Y[q][bidx] for q in IL.TARGETS}
    fold = np.array([int(hashlib.sha256(e.encode()).hexdigest()[:8], 16) % 2 for e in beid])
    TR = {k: np.asarray(v, np.float64) for k, v in torch.load(LB.TRACKS, weights_only=False)["tracks"].items()}
    v_prev = np.array([TR[str(eid[i])][2 * (int(G[i, 0]) - 2), 3] if int(G[i, 0]) >= 2 and 2 * (int(G[i, 0]) - 2) < len(TR[str(eid[i])]) else np.nan for i in sel])
    c = type("Ctx", (), {})()
    c.name, c.beid, c.fold, c.Y, c.qe, c.Yw, c.tags = ds, beid.astype("U36"), fold, Y, eid[sel].astype("U36"), Yw, list(IL.TAGS)
    c.v_prev = v_prev
    c.BZ, c.Z0, c.PRED, c.ORA, c.OFF, c.CON, c.bankrank, c.actmean = {}, {}, {}, {}, {}, {}, {}, {}
    Araw_sel = A_raw[sel].float()
    c.win_mean_yaw = Araw_sel[:, :, 1].mean(1).numpy().astype(np.float64)
    c.win_mean_acc = Araw_sel[:, :, 0].mean(1).numpy().astype(np.float64)
    rng = np.random.default_rng(0)
    cand = np.zeros((nw, N_CAND), int)
    cand[:, 0] = np.arange(nw)
    for i in range(nw):
        cand[i, 1:] = rng.choice(np.delete(np.arange(nw), i), size=N_CAND - 1, replace=False)
    for tag in IL.TAGS:
        m, cfg, ck = W.load_model(W.CKPT[tag])
        wm, st = m.wm, m.wm.stack
        Z1 = L["Z1"][tag]
        with torch.no_grad():
            zs = st.initial_latents(Z1[bw, bf])
        c.BZ[tag] = {l: zs[l - 1].float().numpy() for l in IL.LEVELS}
        A_sc = wm.scale_actions(Araw_sel)[:, :-1]
        z0, ora, pred = pack(wm, Z1[sel], A_sc)
        c.Z0[tag] = {l: z0[l].float().numpy() for l in IL.LEVELS}
        c.PRED[tag] = {k: v.float().numpy() for k, v in pred.items()}
        c.ORA[tag] = {k: v.float().numpy() for k, v in ora.items()}
        c.OFF[tag] = {}
        for l in IL.LEVELS:
            for k in IL.KS:
                f = k * IL.STRIDE[l] + 1
                b = np.array([key2b[(eid[i], int(G[i, f]))] for i in sel])
                c.OFF[tag][(l, k)] = c.BZ[tag][l][b]
        c.CON[tag] = {}
        for cn, (ch, d) in CONTRAST.items():
            Ac = Araw_sel.clone()
            Ac[..., ch] += d
            _, _, pc = pack(wm, Z1[sel], wm.scale_actions(Ac)[:, :-1])
            for (l, k), v in pc.items():
                c.CON[tag][(cn, l, k)] = v.float().numpy()
        # the IWM action bank: the logged action's sequence against 15 sequences of other windows, ranked by distance to the encoded true future
        dist = {key: [] for key in c.ORA[tag]}
        for j in range(N_CAND):
            _, _, pj = pack(wm, Z1[sel], A_sc[torch.as_tensor(cand[:, j])])
            for key, v in pj.items():
                dist[key].append(((v.float().numpy() - c.ORA[tag][key]) ** 2).sum(-1))
        c.bankrank[tag] = {}
        for key, dl in dist.items():
            d = np.stack(dl)
            c.bankrank[tag][key] = 1 + (d[1:] < d[0][None]).sum(0) + 0.5 * (d[1:] == d[0][None]).sum(0)
        print("tag", ds, tag, round(time.time() - t0, 1), "s", flush=True)
    c.chance_bank = float((1.0 / np.arange(1, N_CAND + 1)).sum() / N_CAND)
    for k in IL.KS:
        for l in IL.LEVELS:
            n_ = k * IL.STRIDE[l]
            c.actmean[(l, k)] = Araw_sel[:, :n_].mean(1).numpy().astype(np.float64)
    c.rand_tags = []
    if ds == "eval141" and os.path.exists("latents_rand.pt"):
        R = torch.load("latents_rand.pt", weights_only=False)
        _, cfg, _ = W.load_model(W.CKPT["M2b_s0"])
        for r in (0, 1):
            tg = f"rand{r}"
            mr = IM.make_random_model(r, cfg)
            with torch.no_grad():
                zs = mr.wm.stack.initial_latents(R["Z1"][tg][bw, bf])
            c.BZ[tg] = {l: zs[l - 1].float().numpy() for l in IL.LEVELS}
            c.rand_tags.append(tg)
    c = IL.finish_ctx(c)
    uq, W0 = W.boot_weights(c.beid)
    assert (np.asarray(uq).astype(str) == c.uniq).all() and (W0 == c.W).all(), "bootstrap weights differ from wmact_lib.boot_weights"
    out = f"F:/TanitAD-scratch/imag1_ctx_{ds}.pkl"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    keep = {k: v for k, v in c.__dict__.items()}
    with open(out, "wb") as fh:
        pickle.dump(keep, fh, protocol=4)
    rep = {"dataset": ds, "n_bank_frames": int(len(beid)), "n_episodes": int(c.n_ep), "n_windows": int(nw), "n_windows_episodes": int(len(np.unique(c.qe))),
           "n_per_target": {q: int(c.val[q].sum()) for q in IL.TARGETS}, "n_windows_per_target_at_start": {q: int(np.isfinite(Yw[q][:, 0]).sum()) for q in IL.TARGETS},
           "inputs_sha256": {"latents": sha(path), "labels": sha(f"labels_imag1_{ds}.npz"), "hacc_s0": sha(W.CKPT["M2b_s0"]), "hacc_s1": sha(W.CKPT["M2b_s1"])},
           "rand_tags": c.rand_tags, "tanitad_file": W.tanitad.__file__, "wall_s": round(time.time() - t0, 1), "ctx_bytes": os.path.getsize(out),
           "bank_chance_mrr": c.chance_bank}
    json.dump(rep, open(f"ctx_{ds}.report.json", "w", encoding="utf-8"), indent=1)
    print(json.dumps(rep)[:1500], flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
