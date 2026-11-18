"""AMEND_HWMPLAN_7 section 2.2 / 3.3 / 3.7: the pair table of the 1,000 clips, with the two R35 rollouts per pair and the GAIN7 anchor contrasts. CPU only.
  python pairs7.py --chassis s0|s1      writes amend7/pairs/pairs7_<k>.npz + PAIRS7_<k>.json (or reuses the npz when its sha256 equals the record)
Columns (rows in fit_ra6.pairs' order = clips sorted by id): z0, z10 (float32), y (mean scaled acceleration over the 10 transitions), ep (clip index in the sorted order), oc (index in the FIT-CURVE order = clips141 + clips6[:859]),
fold (RandomState(0).permutation(1000) % 5 [oc]), frame, fr0 (first table frame of the clip), part (swap partner), act (10 scaled actions per pair), F_true / F_alt (R35 rollouts, card 3.3),
anc / zp / zm (anchors every 10 frames and the +/- delta rollouts, card 3.7).
Asserts C-PAIRS7 (equal to fit_ra6.pairs bit for bit, 58,934 pairs in 1,000 clips, every chunk's sha256) and C-ISO7 on build. Tier T0."""
import amend7_env  # noqa: F401  (first)
import argparse, hashlib, json, os, sys, time
sys.dont_write_bytecode = True
import numpy as np
import torch
from amend7_env import ROOT, ST, A6, R
import fit_ra6 as F
import mlp7 as M

DELTA = 0.42                         # card 3.7: RSP_DELTA[0] (v8f_mini.py:168), WM units
N_PAIRS, N_CLIPS, N_NEW = 58934, 1000, 859
C141 = "F:/TanitAD-scratch/hwm_plan/ts/clips141.json"
LAT_DIR = A6 + "/lat6"
CLIPS6 = A6 + "/clips6.json"
PAIRS_DIR = ROOT + "/pairs"


def pairs_ext(LT, scale_fn, k=10):
    """fit_ra6.pairs with the extra columns kept. Same frame table (first window wins), same 11-consecutive-finite-frames rule, same float32 means."""
    tag = list(LT["Z1"])[0]
    Zw, Aw, okw = LT["Z1"][tag].float(), scale_fn(LT["A_raw"].float()), LT["ok"].bool()
    tw, eidw = np.asarray(LT["t"]), list(LT["eid"])
    clips = sorted(set(eidw)); cix = {c: i for i, c in enumerate(clips)}
    tab = {}
    for w in range(Zw.shape[0]):
        for j in range(Zw.shape[1]):
            if bool(okw[w, j]):
                tab.setdefault((cix[eidw[w]], int(tw[w]) - (Zw.shape[1] - 1) + j), (Zw[w, j], Aw[w, j]))
    byc = {}
    for (c, f) in tab:
        byc.setdefault(c, []).append(f)
    EP, ZT, ZK, AB, FR, FR0, ACT = [], [], [], [], [], [], []
    for ci in range(len(clips)):
        fr = sorted(byc.get(ci, []))
        pos = {int(f): i for i, f in enumerate(fr)}
        if not fr:
            continue
        Z = torch.stack([tab[(ci, f)][0] for f in fr]); A = torch.stack([tab[(ci, f)][1] for f in fr])
        for i, f in enumerate(fr):
            f = int(f)
            if all((f + m) in pos for m in range(k + 1)):
                js = [pos[f + m] for m in range(k)]
                if torch.isfinite(A[js]).all():
                    EP.append(ci); ZT.append(Z[i]); ZK.append(Z[pos[f + k]]); AB.append(A[js].mean(0)); FR.append(f); FR0.append(int(fr[0])); ACT.append(A[js])
    zt, zk = torch.stack(ZT).numpy(), torch.stack(ZK).numpy()                       # float32
    ab = torch.stack(AB).numpy().astype(np.float64)
    return dict(z0=zt, z10=zk, D=zk.astype(np.float64) - zt.astype(np.float64), y=ab[:, 0], ep=np.array(EP), frame=np.array(FR), fr0=np.array(FR0), act=torch.stack(ACT).numpy()), clips


def rollouts(st, z0, act, abar=M.ABAR, bs=1024):
    """F_true: from z0 under the stretch's 10 logged scaled actions (both channels); F_alt: the same with the acceleration channel set to abar and the logged yaw kept. Through ra_l1.roll (context 1, read after 10 actions)."""
    z0t, at = torch.from_numpy(z0), torch.from_numpy(act)
    Ft, Fa = np.zeros(z0.shape, np.float32), np.zeros(z0.shape, np.float32)
    with torch.no_grad():
        for b in range(0, len(z0), bs):
            sl = slice(b, min(len(z0), b + bs))
            a_t = at[sl].unsqueeze(1)
            a_a = a_t.clone(); a_a[..., 0] = abar
            Ft[sl] = R.roll(st, z0t[sl], a_t)[:, 0].numpy()
            Fa[sl] = R.roll(st, z0t[sl], a_a)[:, 0].numpy()
    return Ft, Fa


def anchor_contrast(st, z0, act, anc, delta=DELTA, bs=1024):
    """Card 3.7: contrast +/- delta on the acceleration channel of the logged sequence at the anchors; the end states after 10 actions."""
    z0t, at = torch.from_numpy(z0[anc]), torch.from_numpy(act[anc])
    zp, zm = np.zeros((len(anc), z0.shape[1]), np.float32), np.zeros((len(anc), z0.shape[1]), np.float32)
    with torch.no_grad():
        for b in range(0, len(anc), bs):
            sl = slice(b, min(len(anc), b + bs))
            a2 = torch.stack([at[sl].clone(), at[sl].clone()], 1)
            a2[:, 0, :, 0] += delta; a2[:, 1, :, 0] -= delta
            e = R.roll(st, z0t[sl], a2)
            zp[sl], zm[sl] = e[:, 0].numpy(), e[:, 1].numpy()
    return zp, zm


def fold_of(oc, n_clips=N_CLIPS):
    return (np.random.RandomState(0).permutation(n_clips) % 5)[oc]


def build_table(LT, scale_fn, st, c141, new_clips, score_eids, n_pairs=None, n_clips=None, delta=DELTA, abar=M.ABAR):
    """-> table dict. LT: the merged latents. Raises AssertionError when C-PAIRS7 or C-ISO7 reads RED."""
    ext, clips = pairs_ext(LT, scale_fn)
    Dr, Yf, EP, clips_ref = F.pairs(LT, scale_fn)
    c7 = M.check_pairs(ext, dict(Dr=Dr, Yf=Yf, EP=EP, clips=clips_ref), n_pairs, n_clips)
    assert c7["ok"], f"C-PAIRS7 RED: {c7}"
    order = list(c141) + list(new_clips)
    ci = M.check_iso(clips, c141, list(new_clips), score_eids, n_new=len(new_clips))
    assert ci["ok"], f"C-ISO7 RED: {ci}"
    remap = {c: i for i, c in enumerate(order)}
    oc = np.array([remap[clips[i]] for i in ext["ep"]])
    t = dict(ext)
    t.pop("D")
    t["oc"] = oc
    t["fold"] = fold_of(oc, len(order))
    t["part"] = M.swap_partner(ext["ep"], ext["frame"])
    t["F_true"], t["F_alt"] = rollouts(st, ext["z0"], ext["act"], abar)
    anc = np.where((ext["frame"] - ext["fr0"]) % 10 == 0)[0]
    t["anc"] = anc
    t["zp"], t["zm"] = anchor_contrast(st, ext["z0"], ext["act"], anc, delta)
    return t, dict(C_PAIRS7=c7, C_ISO7=ci, clip_order_sha256=hashlib.sha256(json.dumps(order).encode()).hexdigest(), n_clips=len(order))


def load_table(path, rec_path=None):
    """Reads pairs7_<k>.npz; with its record, the sha256 must match (a stale or edited table refuses)."""
    if rec_path is not None:
        want = json.load(open(rec_path, encoding="utf-8"))["sha256"]
        got = hashlib.sha256(open(path, "rb").read()).hexdigest()
        if got != want:
            raise RuntimeError(f"{path} sha256 {got[:12]} != the record {want[:12]}")
    with np.load(path, allow_pickle=False) as z:
        return {x: z[x] for x in z.files}


def _model(k):
    sys.path.insert(0, ST.LONPLAN_BUILD)
    m, cfg, _ = R.D.load_model(ST.CH[k]["ck"])
    wm = m.wm
    for p in wm.parameters():
        p.requires_grad_(False)
    wm.eval(); wm.stack.eval()
    return wm


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--chassis", required=True); a = ap.parse_args()
    k = a.chassis
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    t0 = time.time()
    os.makedirs(PAIRS_DIR, exist_ok=True)
    out, rec_p = f"{PAIRS_DIR}/pairs7_{k}.npz", f"{ROOT}/PAIRS7_{k}.json"
    if os.path.exists(out) and os.path.exists(rec_p):
        try:
            load_table(out, rec_p)
            print("PAIRS7", k, "reused (sha256 equals the record)"); return
        except RuntimeError as e:
            print("PAIRS7", k, "rebuilding:", e)
    wm = _model(k)
    tag = ST.CH[k]["tag"]
    c141 = list(json.load(open(C141, encoding="utf-8"))["clips"])
    c6 = json.load(open(CLIPS6, encoding="utf-8"))["clips"][:N_NEW]
    J = json.load(open(LAT_DIR + "/CHUNKS.json", encoding="utf-8"))
    import hashlib as _h
    sha_chunks = _h.sha256(open(LAT_DIR + "/CHUNKS.json", "rb").read()).hexdigest()
    LT141 = torch.load(ST.CH[k]["lat"], map_location="cpu", weights_only=False)
    M_ = F.merge_latents([LT141] + F.chunk_parts(tag, N_NEW, LAT_DIR + "/CHUNKS.json", LAT_DIR))        # checks every chunk's sha256 against CHUNKS.json
    del LT141
    score_eids = sorted(set(np.load(ST.CH[k]["win"], allow_pickle=False)["eid"].tolist()))
    t, ck = build_table(M_, wm.scale_actions, wm.stack, c141, c6, score_eids, N_PAIRS, N_CLIPS)
    np.savez(out, **t)
    sha = hashlib.sha256(open(out, "rb").read()).hexdigest()
    rec = dict(chassis=k, file=out, sha256=sha, n=int(len(t["y"])), n_clips=ck["n_clips"], n_anchors=int(len(t["anc"])), clip_order_sha256=ck["clip_order_sha256"], abar=M.ABAR, delta=DELTA,
               chunks_json_sha256=sha_chunks, n_chunks=len(J["chunks"]), C_PAIRS7=ck["C_PAIRS7"], C_ISO7=ck["C_ISO7"], wall_s=round(time.time() - t0, 1), tier="T0", evidence="MEASURED at build",
               finished=time.strftime("%Y-%m-%dT%H:%M:%S"))
    json.dump(rec, open(rec_p, "w", encoding="utf-8", newline="\n"), indent=1, default=float)
    print("PAIRS7", k, "n", rec["n"], "clips", rec["n_clips"], "anchors", rec["n_anchors"], "sha", sha[:12], "wall", rec["wall_s"])


if __name__ == "__main__":
    main()
