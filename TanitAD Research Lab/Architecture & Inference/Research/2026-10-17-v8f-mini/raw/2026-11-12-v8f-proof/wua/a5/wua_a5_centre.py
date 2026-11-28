"""wua_a5_centre.py -- card ADD155 Amendment 2 (C-CENTRE): the A5 LEAD masks applied to speed-matched NO-LEAD windows. Imports wua_a5 (no copy).

  python -I wua_a5_centre.py --seed 0      -> C:/Users/Admin/wua_work/a5/RESULT_A5_CENTRE_s<seed>.json
  python -I wua_a5_centre.py --combine     -> RESULT_A5_CENTRE.json (name across seeds)
"""
import argparse, hashlib, json, lzma, os, pickle, sys, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
sys.dont_write_bytecode = True
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import wua_a5 as A5   # noqa: E402

OUT = A5.OUT
K_MATCH, DV_TOL = 3, 1.0


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def select(J, wins, store):
    sel = pickle.load(open(f"{OUT}/SEL.pkl", "rb"))["kept"]
    AG = {}
    with lzma.open(A5.AGENTS, "rt") as fh:
        for line in fh:
            r = json.loads(line)
            AG[(r["clip_id"], int(r["frame"]))] = r["agents"]
    vcache = {}

    def v_at(clip, t):
        if clip not in vcache:
            vcache[clip] = np.asarray(store.get(clip)["v"], np.float64) if "v" in store.get(clip) else None
        return float(vcache[clip][t])
    pool = []
    for ci, t, _, _ in wins:
        clip = str(J["clip"][ci]); row = int(J["off"][ci]) + t
        if J["lead_valid"][row] == 1:
            continue
        ag = AG.get((clip, 2 * t))
        if ag is None or any(0 < a["cx"] <= 50.0 and abs(a["cy"]) <= 2.0 for a in ag):
            continue
        pool.append((clip, int(ci), int(t), v_at(clip, t)))
    pv = np.array([p[3] for p in pool])
    match = []
    for k in sel:
        vk = v_at(k["clip"], k["t"])
        cand = [i for i in np.argsort(np.abs(pv - vk), kind="stable") if pool[i][0] != k["clip"] and abs(pv[i] - vk) <= DV_TOL]
        cand = sorted(cand[:50], key=lambda i: (abs(pv[i] - vk), pool[i][0], pool[i][2]))[:K_MATCH]
        match.append(cand)
    return sel, pool, match


def run(seed, threads=4, batch=32):
    import torch
    torch.set_num_threads(threads)
    sys.path.insert(0, A5.WUA)
    import wua_extract as X
    D, J, store, wins = A5.open_data()
    model, cksha = X.load(seed)
    cfg = model.cfg
    ds = D.V8FMiniWindows(store, J, [], hist=cfg.hist, hmax=cfg.hmax, max_age_t=cfg.max_age_t, max_age_s=cfg.max_age_s, window_start=cfg.window_start)
    keys = sorted(set(A5.model_keys()))
    fill = torch.from_numpy(np.load(f"{OUT}/cellmean.npy"))
    sel, pool, match = select(J, wins, store)
    pairs = [(w, i) for w, m in enumerate(match) for i in m]
    uniq = sorted({i for _, i in pairs})
    t0 = time.time()

    def fwd(idx, masks):
        items = [ds.item(pool[i][1], pool[i][2], 0, 0) for i in idx]
        full = D.collate(items)
        b = {k: full[k] for k in keys}
        if masks is not None:
            b["feats"] = A5.apply_mask(b["feats"], torch.from_numpy(np.stack(masks)), fill)
        with torch.no_grad():
            return model.forward_infer(b)["traj30"].float().cpu()
    vbase = {}
    empty_max, empty_bit = 0.0, True
    for b0 in range(0, len(uniq), batch):
        idx = uniq[b0:b0 + batch]
        tr = fwd(idx, None)
        te = fwd(idx, [np.zeros((A5.HMAX, A5.NR * A5.NC), bool) for _ in idx])
        empty_bit &= bool(torch.equal(tr, te))
        sp, se = A5.speed2s(tr), A5.speed2s(te)
        empty_max = max(empty_max, float(np.abs(se - sp).max()))
        for i, v in zip(idx, sp):
            vbase[i] = float(v)
    dvm = np.full(len(pairs), np.nan)
    for b0 in range(0, len(pairs), batch):
        pb = pairs[b0:b0 + batch]
        tr = fwd([i for _, i in pb], [sel[w]["ml"] for w, _ in pb])
        sp = A5.speed2s(tr)
        for j, (w, i) in enumerate(pb):
            dvm[b0 + j] = sp[j] - vbase[i]
        print(f"[centre s{seed}] {b0 + len(pb)}/{len(pairs)} {time.time() - t0:.0f}s", flush=True)
    arr = np.load(f"{OUT}/RESULT_A5_s{seed}_arrays.npz")
    assert len(arr["v2_base"]) == len(sel)
    dl = arr["v2_lead_mean"] - arr["v2_base"]
    pw = np.array([w for w, _ in pairs])
    absc = np.array([np.abs(dvm[pw == w]).mean() if (pw == w).any() else np.nan for w in range(len(sel))])
    sgnc = np.array([dvm[pw == w].mean() if (pw == w).any() else np.nan for w in range(len(sel))])
    ok = np.isfinite(absc)
    cid = arr["cid"]
    W = A5.boot_weights(int(cid.max()) + 1)
    res = {"card": "ADD155", "amendment": 2, "seed": seed, "ckpt_sha256": cksha, "n_lead_windows": len(sel), "n_matched": int(ok.sum()),
           "n_nolead_pool": len(pool), "n_pairs": len(pairs), "n_unique_nolead": len(uniq),
           "metric_absdv_lead_minus_centre": A5.boot_mean(np.abs(dl) - absc, cid, W, ok),
           "absdv_lead": A5.boot_mean(np.abs(dl), cid, W, ok), "absdv_centre": A5.boot_mean(absc, cid, W, ok),
           "signed_dv_lead": A5.boot_mean(dl, cid, W, ok), "signed_dv_centre": A5.boot_mean(sgnc, cid, W, ok),
           "control_empty": {"max_abs_dv": empty_max, "traj30_bit_identical": empty_bit, "PASS": bool(empty_max == 0.0 and empty_bit)},
           "am2_sha256": sha("F:/TanitAD-scratch/v8proof/PREREG_V8F_PROOF_ADD155_AM2.md"), "tool_sha256": sha(os.path.abspath(__file__)),
           "seconds": round(time.time() - t0, 1)}
    json.dump(res, open(f"{OUT}/RESULT_A5_CENTRE_s{seed}.json", "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("n_matched", "metric_absdv_lead_minus_centre", "signed_dv_centre", "control_empty")}))


def combine():
    R = [json.load(open(f"{OUT}/RESULT_A5_CENTRE_s{s}.json")) for s in (0, 1, 2)]
    name = "A5-LEAD-SPECIFIC" if all(r["metric_absdv_lead_minus_centre"]["lo"] > 0 for r in R) and all(r["control_empty"]["PASS"] for r in R) else "A5-CENTRE-NOT-EXCLUDED"
    out = {"card": "ADD155", "amendment": 2, "name": name, "per_seed": R}
    json.dump(out, open(f"{OUT}/RESULT_A5_CENTRE.json", "w"), indent=1)
    print(name)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int)
    ap.add_argument("--combine", action="store_true")
    a = ap.parse_args()
    combine() if a.combine else run(a.seed)
