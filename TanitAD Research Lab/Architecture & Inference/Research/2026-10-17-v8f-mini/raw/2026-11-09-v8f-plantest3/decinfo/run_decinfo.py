"""PREREG_V8F_DECINFO: the real-window run. Units write runs/<rung>_<tag>.json; a present unit file is skipped (resumable).
Usage: python run_decinfo.py [only_rungs, e.g. R0,R2]   Order: R0+R2 (M2b), R1 (ladder), R3+R4 (M2b s0), R5+R6, R3+R4 (M2b s1)."""
import json
import os
import sys
import time
import traceback

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import torch

import decinfo_lib as DL

RUNS = os.path.join(HERE, "runs")
os.makedirs(RUNS, exist_ok=True)
TARGETS = ("speed", "yaw")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def unit_path(rung, tag):
    return os.path.join(RUNS, f"{rung}_{tag}.json")


def base_row(D, rung, tag, reader, m, tgt):
    return dict(rung=rung, tag=tag, reader=reader, m=m, target=tgt, n_fit_episodes=D.fit_eps, n_held_episodes=D.held_eps)


def shuffled(D, y, seed):
    ys = y.copy()
    ys[:D.nf] = DL.derange_tracks(y[:D.nf], D.eid[:D.nf], seed)
    return ys


def read_rung(D, rung, tag, X, kind, ms=(1, 4), readers=None, extra=None):
    """X: (n, 16, C) (vector) or (n, 16, C, 40) (grid), fit windows first. Returns rows + planted rows. kind 'vec' -> ridge + mlp, 'grid' -> ridge + conv."""
    rows, planted = [], []
    ys_all = DL.targets(D)
    for m in ms:
        Xs = DL.stack_ch(X, m)
        Xf = Xs[:D.nf].reshape(D.nf * Xs.shape[1], -1); Xh = Xs[D.nf:].reshape((D.n - D.nf) * Xs.shape[1], -1)
        a, z = DL.planted_check(Xf, Xh)
        planted.append(dict(rung=rung, tag=tag, m=m, r2_signal=a, r2_zeroed=z, passed=bool(a >= 0.80 and z <= 0.02), n_features=int(Xf.shape[1])))
        log(rung, tag, f"m={m} planted signal {a:.3f} zeroed {z:.3f}")
        for ti, tgt in enumerate(TARGETS):
            y = ys_all[tgt]
            ysh = shuffled(D, y, seed=100 + ti)
            yS, ySh = y[:, m - 1:], ysh[:, m - 1:]
            r, nf_, nh_ = DL.ridge_read(D, Xs, yS, yS)
            rs, _, _ = DL.ridge_read(D, Xs, yS, ySh)
            row = base_row(D, rung, tag, "ridge", m, tgt); row.update(r2=r, r2_shuffled=rs, n_fit=nf_, n_held=nh_)
            rows.append(row); log(rung, tag, "ridge", m, tgt, f"{r:+.3f} shuf {rs:+.3f}")
            if kind == "vec":
                r, e = DL.mlp_read(D, Xs, yS, yS)
                rs, _ = DL.mlp_read(D, Xs, yS, ySh)
                row = base_row(D, rung, tag, "mlp", m, tgt); row.update(r2=r, r2_shuffled=rs, n_fit=e["n_train"] + e["n_es"], n_held=e["n_held"], detail=e)
            else:
                rr, es_ = [], []
                for sd in (0, 1, 2):
                    r, e = DL.conv_read(D, Xs, yS, yS, sd); rr.append(r); es_.append(e)
                rs, _ = DL.conv_read(D, Xs, yS, ySh, 0)
                row = base_row(D, rung, tag, "conv", m, tgt); row.update(r2=float(np.mean(rr)), r2_seeds=rr, r2_shuffled=rs, n_fit=es_[0]["n_train"] + es_[0]["n_es"], n_held=es_[0]["n_held"], detail=es_)
            rows.append(row); log(rung, tag, row["reader"], m, tgt, f"{row['r2']:+.3f} shuf {row['r2_shuffled']:+.3f}")
    return rows, planted


def save(rung, tag, rows, planted, meta):
    with open(unit_path(rung, tag), "w", encoding="utf-8") as f:
        json.dump(dict(rung=rung, tag=tag, tier="T0", rows=rows, planted=planted, meta=meta), f, indent=1)


def tag_meta(tag):
    return dict(added_after_freeze=bool(tag not in DL.FROZEN_LADDER and not tag.startswith("M2b")))


def vector_units(D, tag, path, rungs_wanted, kind_tags):
    wait_needed = [r for r in kind_tags if not os.path.exists(unit_path(r, tag))]
    if not wait_needed:
        return
    DL.wait_ram()
    model, info = DL.load_any(path)
    z1, pooled = DL.vector_inputs(model, D.feats)
    meta = dict(tag_meta(tag), ckpt=info)
    if tag in D.z1_ref:
        dif = float((z1 - D.z1_ref[tag]).abs().max())
        meta["z1_vs_stored_latents_max_abs_diff"] = dif
        assert dif < 1e-4, (tag, dif)
    for rung in wait_needed:
        if rung not in rungs_wanted:
            continue
        X = (z1 if rung in ("R0", "R1") else pooled).numpy().astype(np.float32)
        rows, planted = read_rung(D, rung, tag, X, "vec")
        m2 = dict(meta)
        if rung == "R0" and tag.startswith("M2b"):                                   # control 4
            pre = json.load(open(DL.PRE.format(tag=tag), encoding="utf-8"))["enc_L1"]
            rep = {}
            for tgt, key in (("speed", "r2_speed"), ("yaw", "r2_yaw")):
                mine = [r["r2"] for r in rows if r["reader"] == "ridge" and r["m"] == 1 and r["target"] == tgt][0]
                rep[tgt] = dict(mine=mine, plantest3=pre[key], abs_diff=abs(mine - pre[key]), passed=bool(abs(mine - pre[key]) <= 0.01))
            m2["control4_reproduction"] = rep
        if rung == "R2" and tag.startswith("M2b"):                                   # control 5
            sub = np.arange(0, 96, 6)
            assert len(sub) == 16 and sub.max() < D.nf
            with torch.no_grad():
                adapt = model.adapter(torch.from_numpy(D.feats[sub]).float()) + model.pos[None, None]
                pool_sub = adapt.mean(2)
                eq = bool(torch.equal(model.wm.z1(adapt), model.wm.proj(pool_sub)))
                eq_stored = bool(torch.equal(model.wm.proj(torch.from_numpy(pooled.numpy()[sub])), model.wm.proj(pool_sub)))
            m2["control5_chain_identity"] = dict(n_windows=16, bit_for_bit=eq, stored_pooled_equals_fresh=eq_stored, passed=eq)
        save(rung, tag, rows, planted, m2)
        log("saved", unit_path(rung, tag))


def grid_units(D, tag, rungs, build):
    todo = [(r, m) for r, m in rungs if not os.path.exists(unit_path(r, tag))]
    if not todo:
        return
    DL.wait_ram()
    grid, (mu, comps) = build()
    meta = dict(tag_meta(tag), pca_k=DL.PCA_K, pca_fit_windows=D.nf)
    for rung, m in todo:
        rows, planted = read_rung(D, rung, tag, grid, "grid", ms=(m,))
        save(rung, tag, rows, planted, meta)
        log("saved", unit_path(rung, tag))
    del grid


def main():
    only = set(sys.argv[1].split(",")) if len(sys.argv) > 1 else None
    want = lambda *r: only is None or bool(only & set(r))
    log("free RAM GB", round(DL.free_gb(), 1), "tanitad", DL.W.tanitad.__file__)
    D, sp = DL.load_tune()
    log("TUNE windows", D.n, "fit", D.nf, "held", D.n - D.nf, "episodes", D.fit_eps, D.held_eps, "feats GB", round(D.feats.nbytes / 2 ** 30, 2))
    assert D.n == 982 and D.nf == 744
    for tag in ("M2b_s0", "M2b_s1"):
        if want("R0", "R2"):
            vector_units(D, tag, DL.M2B[tag], {"R0", "R2"}, ("R0", "R2"))
    if want("R1"):
        for tag, p in DL.ladder_models().items():
            try:
                vector_units(D, tag, p, {"R1"}, ("R1",))
            except Exception:
                log("R1", tag, "FAILED", traceback.format_exc())
    def adapter_unit(tag):
        model, _ = DL.load_any(DL.M2B[tag])
        grid_units(D, tag, (("R3", 1), ("R4", 4)), lambda: DL.adapter_grid(model, D))
    if want("R3", "R4"):
        adapter_unit("M2b_s0")
    if want("R5", "R6"):
        grid_units(D, "encoder", (("R5", 1), ("R6", 4)), lambda: DL.encoder_grid(D))
    if want("R3", "R4"):
        adapter_unit("M2b_s1")
    log("run complete")


if __name__ == "__main__":
    sys.exit(main() or 0)
