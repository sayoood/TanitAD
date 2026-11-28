"""preflight_l7s.py -- the L7-S PREFLIGHT on the REAL train139 banks.  ZERO optimizer steps, no optimizer object, no checkpoint.

    python run_isolated.py --stack <stack_min> preflight_l7s.py --fan-npz <s0>/fan_extras.npz <s1>/fan_extras.npz \
        --targets-npz <t_s0.npz> <t_s1.npz> --scene-cache <dir> --out <preflight_l7s.json>

Gates (each is asserted; the script exits non-zero and says which):
  G1  the split is the E / F split (n_train 8730, n_val 980, 125 / 14 episodes at seed 20261021)
  G2  scene join: every bank's (clip_index, ws) is a bijection onto its fan; the gathered rows equal an INDEPENDENT read (numpy memmap of the
      cache file for random rows, and a streaming inflate of the original zip member for the first 64 rows)
  G3  forward + loss on real batches: every loss finite, every head output finite
  G4  EVERY parameter tensor of the S model receives a finite gradient on >= 1 batch, and the scene branch (memory.proj.*, memory.type_emb.*,
      trunk cross-attention) receives a NON-ZERO one -- the scene is wired into the loss, not decorative
  G5  the weights are bit-identical before and after (no optimizer exists; the grads are the only thing that moved)
Measures per-batch time (gather / forward+loss / backward) and the evaluate() time, and extrapolates the wall clock of <= 30 epochs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import zipfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import refc_scorer as RS  # noqa: E402
import scene_bank as SB  # noqa: E402
import train_l7 as TL  # noqa: E402


def sd_hash(m) -> str:
    h = hashlib.sha256()
    for k, v in sorted(m.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fan-npz", nargs="+", required=True)
    ap.add_argument("--targets-npz", nargs="+", required=True)
    ap.add_argument("--scene-cache", required=True)
    ap.add_argument("--batches", type=int, default=6)
    ap.add_argument("--eval-windows", type=int, default=128)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rep: dict = {"gates": {}, "batches": a.batches}
    fails: list[str] = []

    def gate(name: str, ok: bool, detail=None):
        rep["gates"][name] = {"pass": bool(ok), "detail": detail}
        print(f"ZZGATE {name} {'PASS' if ok else 'FAIL'} {json.dumps(detail)[:300]}", flush=True)
        if not ok:
            fails.append(name)

    t0 = time.time()
    fanz, tg = TL.load_bank(a.fan_npz, a.targets_npz)
    fanz["scene"] = SB.attach(fanz, list(a.fan_npz), a.scene_cache)
    eid = np.asarray(tg["eid"]).astype(str)
    tr, va = TL.episode_split(eid, 0.1, 20261021)
    rep["setup_s"] = round(time.time() - t0, 2)
    gate("G1_split_is_the_EF_split", (len(tr), len(va), len(set(eid[tr])), len(set(eid[va]))) == (8730, 980, 125, 14),
         {"n_train": len(tr), "n_val": len(va), "ep_train": len(set(eid[tr])), "ep_val": len(set(eid[va]))})

    # G2 -- independent reads -------------------------------------------------------------------------------------------------
    sb = fanz["scene"]
    rng = np.random.default_rng(7)
    ok2, det2 = True, {"identical_order": sb.identical_order, "dims": sb.dims}
    for bi, st in enumerate(sb.stores):
        rows = np.sort(rng.choice(len(st.clip_index), 24, replace=False))
        g = st.gather(rows)
        for key in SB.SCENE_MEMBERS:
            mm = np.load(st.cache_files[key], mmap_mode="r")
            same = bool(np.array_equal(g[key], np.asarray(mm[rows])))
            ok2 &= same
            det2[f"bank{bi}_{key}_vs_memmap_24_random_rows"] = same
        with zipfile.ZipFile(st.npz_path) as z:
            for key, member in SB.SCENE_MEMBERS.items():
                with z.open(member + ".npy") as f:
                    shape, dtype, _ = SB.npy_header(f)
                    n = 64
                    first = np.frombuffer(f.read(n * int(np.prod(shape[1:])) * dtype.itemsize), dtype).reshape((n,) + tuple(shape[1:]))
                same = bool(np.array_equal(st.gather(np.arange(n))[key], first))
                ok2 &= same
                det2[f"bank{bi}_{key}_vs_zip_stream_first64"] = same
        ci_back = st.clip_index[sb.scene_row[sb.bank_of_row == bi]]
        ok2 &= bool(np.array_equal(ci_back, fanz["clip_index"][sb.bank_of_row == bi]))
    gate("G2_scene_join_and_independent_reads", ok2, det2)

    # model ---------------------------------------------------------------------------------------------------------------------
    torch.manual_seed(0)
    cfg = RS.L7Config(mode="S", source_dims=dict(sb.dims))
    model = RS.L7Scorer(cfg)
    rep["params"] = RS.n_params(model)
    before = sd_hash(model)
    rng = np.random.default_rng(0)
    perm = rng.permutation(tr)
    names = [n for n, _ in model.named_parameters()]
    seen = {n: False for n in names}
    nonzero = {n: False for n in names}
    finite = True
    t_g, t_f, t_b, losses = [], [], [], []
    model.train()
    for k in range(a.batches):
        jj = perm[k * 32:(k + 1) * 32]
        t = time.perf_counter()
        inp, t_, keep = TL.to_batch(fanz, tg, jj, "S")
        t_g.append(time.perf_counter() - t)
        t = time.perf_counter()
        out = model(**inp)
        loss, lg = RS.pai_loss(out, t_, keep, 0.5)
        t_f.append(time.perf_counter() - t)
        finite &= bool(torch.isfinite(loss)) and all(bool(torch.isfinite(v).all()) for v in out.values())
        t = time.perf_counter()
        model.zero_grad(set_to_none=True)
        loss.backward()
        t_b.append(time.perf_counter() - t)
        losses.append(float(loss.detach()))
        for n, p in model.named_parameters():
            if p.grad is not None:
                seen[n] = True
                finite &= bool(torch.isfinite(p.grad).all())
                nonzero[n] |= bool(p.grad.abs().max() > 0)
    model.zero_grad(set_to_none=True)
    gate("G3_loss_and_outputs_finite", finite and all(np.isfinite(losses)), {"losses": [round(x, 4) for x in losses]})
    missing = [n for n in names if not seen[n]]
    scene_names = [n for n in names if n.startswith("memory.") or ".ca." in n]
    zero_scene = [n for n in scene_names if not nonzero[n]]
    gate("G4_every_parameter_gets_a_finite_gradient", not missing and not zero_scene,
         {"n_param_tensors": len(names), "no_grad_at_all": missing, "scene_branch_tensors": len(scene_names), "scene_zero_grad": zero_scene,
          "param_tensors_with_all_zero_grad": [n for n in names if not nonzero[n]]})
    gate("G5_zero_updates", sd_hash(model) == before, {"state_dict_sha256_before_eq_after": sd_hash(model) == before})

    # G6 -- the scene-shuffle control's map is cross-episode on the real validation slice, and reads (not trains) -----------------------
    smap = TL.cross_episode_map(eid, va, len(eid))
    same_ep = int((eid[smap[va]] == eid[va]).sum())
    sh = TL.evaluate(model, fanz, tg, va[:128], "S", temperature=0.5, scene_map=smap)
    tr_ = TL.evaluate(model, fanz, tg, va[:128], "S", temperature=0.5)
    gate("G6_scene_shuffle_map_is_cross_episode_and_changes_the_read", same_ep == 0 and sh["pick_cost_m"] != tr_["pick_cost_m"],
         {"val_rows_mapped_to_the_same_episode": same_ep, "untrained_model_pick_true_vs_shuffled_first128_val": [tr_["pick_cost_m"], sh["pick_cost_m"]]})

    # timing + evaluate ---------------------------------------------------------------------------------------------------------
    med = lambda x: float(np.median(x[1:] if len(x) > 2 else x))        # drop the first (allocator / thread-pool warm-up)
    per_batch = med(t_g) + med(t_f) + med(t_b)
    t = time.perf_counter()
    TL.evaluate(model, fanz, tg, va[:a.eval_windows], "S", temperature=0.5)
    t_eval = (time.perf_counter() - t) / a.eval_windows
    n_batches = int(np.ceil(len(tr) / 32))
    s_epoch_train = n_batches * per_batch * 1.15           # x1.15: the optimizer step + python overhead is NOT measured (no optimizer exists here)
    s_epoch_eval = t_eval * (2 * len(va) + 512)       # val + the scene-shuffled val (control C-S1) + the 512-window train subset
    rep["timing"] = {"threads": os.environ.get("OMP_NUM_THREADS"), "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
                     "gather_s_per_batch32_median": round(med(t_g), 4), "fwd_loss_s_per_batch32_median": round(med(t_f), 4),
                     "bwd_s_per_batch32_median": round(med(t_b), 4), "per_batch_total_s": round(per_batch, 4),
                     "eval_s_per_window": round(t_eval, 5), "n_batches_per_epoch": n_batches,
                     "est_s_per_epoch_train_x1.15": round(s_epoch_train, 1), "est_s_per_epoch_eval": round(s_epoch_eval, 1),
                     "est_s_per_epoch": round(s_epoch_train + s_epoch_eval, 1),
                     "est_min_30_epochs": round(30 * (s_epoch_train + s_epoch_eval) / 60, 1),
                     "est_min_2_runs_S_ts0_ts1": round(2 * 30 * (s_epoch_train + s_epoch_eval) / 60, 1)}
    rep["verdict"] = "PASS" if not fails else "FAIL: " + ",".join(fails)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=1)
    print("ZZL7SPREFLIGHT " + rep["verdict"] + " " + json.dumps(rep["timing"]), flush=True)
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
