"""lh_fit.py -- PREREG_LONHEAD_C s1: fit the lon-decision head LH (and its DER control) on the FRESH split ONLY, by 5-fold episode-grouped CV,
freeze exactly one configuration each.  CPU only; reads no eval-side model output.  NEW file (refcv9 LONHEAD agent, 2026-10-26).

  input   refcv8 EMA-48,852's frozen z_tac of every fresh window (F:/l1c/bank/ztac_fresh.npz, the L1C os-only fresh roll) + v0/10
          (v0 = the value the roll FED, bank `v0_in`; cross-checked == the fresh dump's extras `v0`)
  target  the per-window GT LON6 class in refcv8's v7 space (0,1,3,4,5,7) by the TRAINER'S function (lh_klabel.trainer_lon_targets:
          refc_v3_train.py:4646-4674 -> R8LabelJoin.item -> window_targets(ids='v7')['lon']) on a fresh v9 release that CARRIES the lead pass
          (K-LABEL: the fresh kit as built has none -> REFUSED unless every clip has_lead; `--allow-no-lead` is a mechanics test that writes only
          under <out>/_mechanics_only and never the frozen names)
  rows    every fresh window whose label is a LON6 class (an IGNORE / masked-class window carries no target)
  models  multinomial logistic, lambda in {1e-3, 1e-2, 1e-1, 1, 10, 100}  (objective sum CE + lambda/2 ||W||^2, i.e. sklearn C = 1/lambda)
          one-hidden-layer MLP (256, ReLU), AdamW lr 1e-3, weight decay in {1e-4, 1e-3, 1e-2}, batch 256, <= 200 epochs, early stop (patience 10)
          on an INNER episode-grouped 20 % split of each outer-train fold; the frozen MLP is refit on all fresh rows for the median CV epoch
  CV      5 folds over FRESH EPISODES (clips), numpy default_rng(0) permutation; criterion = mean cross-entropy over all out-of-fold rows
  DER     the same recipe on targets DERANGED across fresh episodes (seed 0): each episode's label sequence is replaced by another episode's
          (no fixed point), tiled / truncated to its own length -- keeps the label stream's within-episode structure and rate, breaks z -> label
  controls (beside, never chosen): CONSTANT (outer-train class prior) and V0-ONLY logistic (the raw ego-scalar floor) under the same folds; n, d.
Freeze: a torch .pt of plain tensors + primitives (no timestamps -> the sha256 is a property of the content); `predict_logits6(blob, z, v0)`.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import lh_klabel as KL  # noqa: E402

LON6_V7 = KL.LON6_V7
LAMBDAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
WDS = (1e-4, 1e-3, 1e-2)
HIDDEN, LR, BATCH, MAX_EP, PATIENCE = 256, 1e-3, 256, 200, 10
SCHEMA = "tanitad.refcv9_lonhead.lh/1"
BANK = Path("F:/l1c/bank/ztac_fresh.npz")
FRESH_LEAD_RELEASE = Path("F:/refcv9_lonhead/kit_fresh_lead/v9/v9_labels_eval.npz")


def sha256_file(p) -> str:
    return KL.sha256_file(p)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# --------------------------------------------------------------------------------------------------------------- data
def load_fresh(release: Path, allow_no_lead: bool, lead_clips_only: bool = False) -> dict:
    z_rel = np.load(release, allow_pickle=False)
    n_clip, n_lead = int(len(z_rel["clip__has_lead"])), int(z_rel["clip__has_lead"].sum())
    nolead_s12 = {str(s) for s, h in zip(z_rel["clip__sha12"], z_rel["clip__has_lead"]) if not bool(h)}
    if lead_clips_only:                     # PREREG_LONHEAD_C_AMENDMENT1 (715b9df1): the 8 clips with NO obstacle.offline are excluded whole
        assert len(nolead_s12) == 8 and n_clip == 300 and n_lead == 292, (len(nolead_s12), n_clip, n_lead)
    if n_lead != n_clip and not allow_no_lead and not lead_clips_only:
        raise SystemExit(f"[lh_fit] REFUSING: the fresh release carries the lead pass on {n_lead}/{n_clip} clips -- its targets are NOT the "
                         f"battery's function (K-LABEL B); rebuild it with code/chain_label_pipeline.sh fresh")
    r = KL.trainer_lon_targets(KL.FRESH, 0, release=str(release), md5=KL.md5_file(release))
    sys.path.insert(0, "C:/Users/Admin/refcv8_pf/refcv9_l1/code")
    bk = np.load(BANK, allow_pickle=False)
    meta = json.loads(str(bk["meta_json"]))
    W = int(meta["window_W"])
    ws_b = bk["t"].astype(np.int64) + W - 1
    idx = {(int(c), int(w)): j for j, (c, w) in enumerate(zip(bk["e_i"], ws_b))}
    j = np.array([idx.get((int(c), int(w)), -1) for c, w in zip(r["clip_index"], r["ws"])])
    if (j < 0).any() or len(idx) != len(ws_b):
        raise SystemExit(f"[lh_fit] {int((j < 0).sum())} fresh windows have no z_tac bank row (or duplicate bank keys)")
    ex = np.load(Path(KL.FRESH["dumps"][0]) / "refcv6_extras.npz")
    v0 = bk["v0_in"].astype(np.float64)[j]
    dv0 = float(np.abs(v0 - ex["v0"].astype(np.float64)).max())
    lab = r["lon"]
    keep = np.isin(lab, LON6_V7)
    if lead_clips_only:
        jx = json.load(open(Path(KL.FRESH["dumps"][0]) / "refcv6_extras.npz.json", encoding="utf-8"))
        i2s = {int(k): str(v) for k, v in jx["clip_sha12_by_index"].items()}
        s12 = np.array([i2s[int(c)] for c in r["clip_index"]])
        lead_ok = ~np.isin(s12, sorted(nolead_s12))
        assert len(set(s12[~lead_ok])) == 8, len(set(s12[~lead_ok]))
        keep = keep & lead_ok
    y = np.array([LON6_V7.index(int(v)) if v in LON6_V7 else -1 for v in lab])
    return {"Z": bk["z_tac"].astype(np.float64)[j][keep], "v0": v0[keep], "y": y[keep], "eid": r["clip_index"][keep], "ws": r["ws"][keep],
            "n_windows": int(len(lab)), "n_rows": int(keep.sum()), "n_episodes": int(len(np.unique(r["clip_index"][keep]))),
            "class_counts": {f"{LON6_V7[k]} {KL.LON7[LON6_V7[k]]}": int((y[keep] == k).sum()) for k in range(6)},
            "checks": {**r["checks"], "bank_v0_in_vs_extras_v0_maxabs": dv0, "bank_final": bool(meta.get("final")),
                       "bank_n_windows": int(meta.get("n_windows", -1))},
            "sources": {"release": str(release), "release_md5": KL.md5_file(release), "release_has_lead_clips": n_lead, "release_n_clips": n_clip, "lead_clips_only": bool(lead_clips_only),
                        "excluded_nolead_clip_sha12": sorted(nolead_s12) if lead_clips_only else [],
                        "bank": str(BANK), "bank_sha256": sha256_file(BANK)}}


def derange_by_episode(y: np.ndarray, eid: np.ndarray, seed: int = 0) -> tuple:
    ue = np.unique(eid)
    rng = np.random.default_rng(seed)
    while True:
        p = rng.permutation(len(ue))
        if not (p == np.arange(len(ue))).any():
            break
    seq = {int(e): y[eid == e] for e in ue}
    out = np.empty_like(y)
    for i, e in enumerate(ue):
        src = seq[int(ue[p[i]])]
        m = eid == e
        n = int(m.sum())
        out[m] = np.resize(src, n)                    # tile / truncate the partner episode's sequence
    return out, {"seed": seed, "n_episodes": int(len(ue)), "fixed_points": int((p == np.arange(len(ue))).sum()),
                 "rows_label_unchanged_frac": float((out == y).mean())}


def folds_of(eid: np.ndarray, k: int = 5, seed: int = 0) -> np.ndarray:
    ue = np.unique(eid)
    perm = np.random.default_rng(seed).permutation(len(ue))
    f = {int(e): int(r % k) for e, r in zip(ue[perm], range(len(ue)))}
    return np.array([f[int(e)] for e in eid])


def feats(Z, v0, st):
    return np.column_stack([(Z - st["zmu"]) / st["zsd"], v0 / 10.0])


def std_of(Z):
    return {"zmu": Z.mean(0), "zsd": Z.std(0) + 1e-9}


def ce_rows(logp6: np.ndarray, y: np.ndarray) -> np.ndarray:
    return -np.maximum(logp6[np.arange(len(y)), y], np.log(1e-12))


def log_softmax(a):
    a = a - a.max(1, keepdims=True)
    return a - np.log(np.exp(a).sum(1, keepdims=True))


# --------------------------------------------------------------------------------------------------------------- models
def fit_logistic(X, y, lam):
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression(C=1.0 / lam, solver="lbfgs", max_iter=5000, tol=1e-8)
    m.fit(X, y)
    W = np.full((6, X.shape[1]), 0.0)
    b = np.full(6, -1e4)                                    # a class absent from the fit cannot be predicted
    cls = [int(c) for c in m.classes_]
    if len(cls) == 2:                                       # sklearn binary: one row = logit(class 1) - logit(class 0)
        W[cls[0]], b[cls[0]] = 0.0, 0.0
        W[cls[1]], b[cls[1]] = m.coef_[0], m.intercept_[0]
    else:
        for i, c in enumerate(cls):
            W[c], b[c] = m.coef_[i], m.intercept_[i]
    return {"kind": "logistic", "W": W, "b": b, "n_iter": int(np.max(m.n_iter_))}


def logits_logistic(m, X):
    return X @ m["W"].T + m["b"]


def _mlp(d_in):
    import torch
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(d_in, HIDDEN), torch.nn.ReLU(), torch.nn.Linear(HIDDEN, 6)).double()


def _train_mlp(X, y, wd, epochs=None, Xv=None, yv=None):
    """epochs=None: early stop on (Xv, yv), patience PATIENCE, returns the best-epoch state and that epoch; else exactly `epochs`."""
    import torch
    torch.manual_seed(0)
    net = _mlp(X.shape[1])
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=wd)
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y.astype(np.int64))
    g = torch.Generator().manual_seed(0)
    best, best_ep, best_state, bad = np.inf, 0, None, 0
    n_ep = MAX_EP if epochs is None else int(epochs)
    for ep in range(1, n_ep + 1):
        net.train()
        perm = torch.randperm(len(Xt), generator=g)
        for i in range(0, len(Xt), BATCH):
            ix = perm[i:i + BATCH]
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(net(Xt[ix]), yt[ix])
            loss.backward()
            opt.step()
        if epochs is None:
            net.eval()
            with torch.no_grad():
                v = float(torch.nn.functional.cross_entropy(net(torch.from_numpy(Xv)), torch.from_numpy(yv.astype(np.int64))))
            if v < best - 1e-9:
                best, best_ep, bad = v, ep, 0
                best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
            else:
                bad += 1
                if bad >= PATIENCE:
                    break
    if epochs is None:
        net.load_state_dict(best_state)
    return net, (best_ep if epochs is None else n_ep)


def logits_mlp(net, X):
    import torch
    net.eval()
    with torch.no_grad():
        return net(torch.from_numpy(X)).numpy()


# --------------------------------------------------------------------------------------------------------------- CV
def cv(D: dict, y: np.ndarray, fold: np.ndarray) -> dict:
    """Out-of-fold mean CE for every config + the two controls.  Standardisation stats are fitted on the outer-train rows only."""
    configs = [("logistic", lam) for lam in LAMBDAS] + [("mlp", wd) for wd in WDS]
    oof = {c: np.full(len(y), np.nan) for c in configs + [("constant", 0), ("v0_only", 1.0)]}
    mlp_ep = {("mlp", wd): [] for wd in WDS}
    for k in range(5):
        tr, te = fold != k, fold == k
        st = std_of(D["Z"][tr])
        Xtr, Xte = feats(D["Z"][tr], D["v0"][tr], st), feats(D["Z"][te], D["v0"][te], st)
        pri = np.bincount(y[tr], minlength=6) / tr.sum()
        oof[("constant", 0)][te] = ce_rows(np.log(np.maximum(np.tile(pri, (te.sum(), 1)), 1e-12)), y[te])
        mv = fit_logistic((D["v0"][tr] / 10.0)[:, None], y[tr], 1.0)
        oof[("v0_only", 1.0)][te] = ce_rows(log_softmax(logits_logistic(mv, (D["v0"][te] / 10.0)[:, None])), y[te])
        for lam in LAMBDAS:
            m = fit_logistic(Xtr, y[tr], lam)
            oof[("logistic", lam)][te] = ce_rows(log_softmax(logits_logistic(m, Xte)), y[te])
        # inner episode-grouped 20 % split of the outer-train fold for the MLP's early stop
        etr = np.unique(D["eid"][tr])
        rin = np.random.default_rng(100 + k).permutation(len(etr))
        inner_val = set(etr[rin[: max(1, int(round(0.2 * len(etr))))]].tolist())
        iv = np.array([int(e) in inner_val for e in D["eid"][tr]])
        for wd in WDS:
            net, ep = _train_mlp(Xtr[~iv], y[tr][~iv], wd, None, Xtr[iv], y[tr][iv])
            mlp_ep[("mlp", wd)].append(int(ep))
            oof[("mlp", wd)][te] = ce_rows(log_softmax(logits_mlp(net, Xte)), y[te])
        print(f"[lh_fit] fold {k}: n_tr {int(tr.sum())} n_te {int(te.sum())}", flush=True)
    table = {}
    for c, v in oof.items():
        fold_ce = [float(np.mean(v[fold == k])) for k in range(5)]
        table[f"{c[0]}:{c[1]:g}"] = {"variant": c[0], "hp": float(c[1]), "mean_ce_oof": float(np.mean(v)), "fold_ce": fold_ce,
                                     **({"early_stop_epochs": mlp_ep[c]} if c in mlp_ep else {})}
    cand = {k: v for k, v in table.items() if v["variant"] in ("logistic", "mlp")}
    best = min(cand, key=lambda k: (cand[k]["mean_ce_oof"], k))
    return {"table": table, "chosen": best, "chosen_cfg": table[best]}


def freeze(D: dict, y: np.ndarray, chosen: dict, tag: str, prov: dict) -> bytes:
    import torch
    st = std_of(D["Z"])
    X = feats(D["Z"], D["v0"], st)
    blob = {"schema": SCHEMA, "tag": tag, "variant": chosen["variant"], "hp": chosen["hp"], "class_order_v7": list(LON6_V7),
            "zmu": torch.from_numpy(st["zmu"].copy()), "zsd": torch.from_numpy(st["zsd"].copy()), "v0_scale": 10.0, "d_z": int(D["Z"].shape[1]),
            "n_rows": int(len(y)), "provenance": prov,
            "use": "logits6 = f(concat[(z_tac - zmu)/zsd, v0/10]) float64 over LON6 v7 ids (0,1,3,4,5,7); argmax -> the v7 id to force"}
    if chosen["variant"] == "logistic":
        m = fit_logistic(X, y, chosen["hp"])
        blob.update(W=torch.from_numpy(m["W"].copy()), b=torch.from_numpy(m["b"].copy()), n_iter=m["n_iter"])
    else:
        ep = int(np.median(chosen["early_stop_epochs"]))
        net, _ = _train_mlp(X, y, chosen["hp"], ep)
        blob.update(state_dict={k: v.detach().clone().contiguous() for k, v in net.state_dict().items()}, hidden=HIDDEN, epochs=ep)
    buf = io.BytesIO()
    torch.save(blob, buf)
    return buf.getvalue()


def load_frozen(path: Path, registered_sha: str | None = None) -> dict:
    import torch
    got = sha256_file(path)
    if registered_sha is not None:
        reg = "".join(ch for ch in str(registered_sha).split()[0] if ch in "0123456789abcdef")
        if len(got) != 64 or len(reg) != 64 or got != reg:
            raise SystemExit(f"[lh] {path} sha256 {got} != registered {reg!r} -- refusing")
    blob = torch.load(path, map_location="cpu", weights_only=True)
    if blob.get("schema") != SCHEMA:
        raise SystemExit(f"[lh] schema {blob.get('schema')!r}")
    blob["_sha256"] = got
    return blob


def predict_logits6(blob: dict, Z: np.ndarray, v0: np.ndarray) -> np.ndarray:
    st = {"zmu": blob["zmu"].numpy(), "zsd": blob["zsd"].numpy()}
    X = feats(np.asarray(Z, np.float64), np.asarray(v0, np.float64), st)
    if blob["variant"] == "logistic":
        return X @ blob["W"].numpy().T + blob["b"].numpy()
    net = _mlp(X.shape[1])
    net.load_state_dict(blob["state_dict"])
    return logits_mlp(net, X)


# --------------------------------------------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", default=str(FRESH_LEAD_RELEASE))
    ap.add_argument("--out", default=str(ROOT))
    ap.add_argument("--lead-clips-only", action="store_true", help="PREREG_LONHEAD_C_AMENDMENT1: fit only the clips the lead pass covers")
    ap.add_argument("--allow-no-lead", action="store_true", help="MECHANICS TEST ONLY: writes under <out>/_mechanics_only, never the frozen names")
    a = ap.parse_args()
    t0 = time.time()
    out = Path(a.out) / ("_mechanics_only" if a.allow_no_lead else "")
    out.mkdir(parents=True, exist_ok=True)
    D = load_fresh(Path(a.release), a.allow_no_lead, a.lead_clips_only)
    missing = [k for k, n in D["class_counts"].items() if n < 10]
    if missing and not a.allow_no_lead:
        raise SystemExit(f"[lh_fit] REFUSING: LON6 classes with < 10 fresh rows: {missing}")
    fold = folds_of(D["eid"])
    prov = {"script_sha256": sha256_file(__file__), "klabel_sha256": sha256_file(HERE / "lh_klabel.py"), **D["sources"],
            "prereg_sha256": sha256_file(ROOT / "PREREG_LONHEAD_C.md"), "label_function": "lh_klabel.trainer_lon_targets (refc_v3_train.py:4646-4674)"}
    info = {"tool": "lh_fit.py", "prereg_sha256": prov["prereg_sha256"], "script_sha256": prov["script_sha256"],
            "population": {k: D[k] for k in ("n_windows", "n_rows", "n_episodes", "class_counts")}, "d_features": int(D["Z"].shape[1] + 1),
            "checks": D["checks"], "sources": D["sources"], "folds": {"k": 5, "seed": 0, "by": "fresh episode (clip)"},
            "mechanics_only": bool(a.allow_no_lead)}
    res_lh = cv(D, D["y"], fold)
    yd, drec = derange_by_episode(D["y"], D["eid"], 0)
    res_der = cv(D, yd, fold)
    info["LH"] = res_lh
    info["DER"] = {**res_der, "derangement": drec}
    lt = res_lh["table"]
    info["controls"] = {"constant_mean_ce": lt["constant:0"]["mean_ce_oof"], "v0_only_mean_ce": lt["v0_only:1"]["mean_ce_oof"],
                        "LH_chosen_mean_ce": res_lh["chosen_cfg"]["mean_ce_oof"], "DER_chosen_mean_ce": res_der["chosen_cfg"]["mean_ce_oof"],
                        "LH_beats_constant": bool(res_lh["chosen_cfg"]["mean_ce_oof"] < lt["constant:0"]["mean_ce_oof"]),
                        "LH_beats_v0_only": bool(res_lh["chosen_cfg"]["mean_ce_oof"] < lt["v0_only:1"]["mean_ce_oof"]),
                        "DER_not_better_than_its_constant": bool(res_der["chosen_cfg"]["mean_ce_oof"] >= res_der["table"]["constant:0"]["mean_ce_oof"] - 0.01)}
    for tag, res, yy in (("LH", res_lh, D["y"]), ("DER", res_der, yd)):
        b = freeze(D, yy, res["chosen_cfg"], tag, {**prov, **({"derangement": drec} if tag == "DER" else {})})
        b2 = freeze(D, yy, res["chosen_cfg"], tag, {**prov, **({"derangement": drec} if tag == "DER" else {})})
        p = out / f"{tag}_FROZEN.pt"
        tmp = p.with_suffix(".pt.tmp")
        tmp.write_bytes(b)
        os.replace(tmp, p)
        h = sha256_file(p)
        (out / f"{tag}_FROZEN.sha256").write_text(f"{h}  {p.name}\n", encoding="utf-8")
        info[f"{tag}_frozen"] = {"file": str(p), "sha256": h, "config": res["chosen"], "refreeze_bytes_identical": bool(b == b2),
                                 "known_value_reload": None}
        bl = load_frozen(p, h)                                  # the frozen file reproduces the in-memory refit (known value)
        lg = predict_logits6(bl, D["Z"], D["v0"])
        info[f"{tag}_frozen"]["known_value_reload"] = {"train_rows_mean_ce": float(ce_rows(log_softmax(lg), yy).mean()),
                                                      "train_rows_argmax_acc": float((lg.argmax(1) == yy).mean())}
    info["wall_s"] = round(time.time() - t0, 1)
    tmp = out / "LH_FIT_INFO.json.tmp"
    tmp.write_text(json.dumps(info, indent=1, default=float), encoding="utf-8")
    os.replace(tmp, out / "LH_FIT_INFO.json")
    print(f"ZZLHFITZZ LH {res_lh['chosen']} ce {res_lh['chosen_cfg']['mean_ce_oof']:.4f} | DER {res_der['chosen']} "
          f"ce {res_der['chosen_cfg']['mean_ce_oof']:.4f} | const {lt['constant:0']['mean_ce_oof']:.4f} v0 {lt['v0_only:1']['mean_ce_oof']:.4f} -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
