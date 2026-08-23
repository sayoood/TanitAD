"""E-DETECT-1T — is v6's encoder IMPROVING, or is it flat?

⛔ THE DECISION THIS SERVES. `v6F-SW-30k` is ~36 h from finishing 30,000 steps.
E-DETECT-1 probed step **20,000** and found the encoder's 640-token field
statistically indistinguishable from RAW PIXELS (0.0923 [0.0815, 0.1034] vs
0.0912 [0.0814, 0.1014]) while DINOv3 on the identical instrument scored 0.1884.
Whether the remaining 10,000 steps are worth anything depends on a quantity
nobody has measured: **the TRAJECTORY of that number.**

Flat from 11,250 to 20,000 => more steps of the same objective will not fix it,
and that is knowable now rather than in 36 h.
Rising => extrapolate, and the run is earning its GPU.

⚠️⚠️ THE TWO CACHES ARE NOT ON THE SAME FRAME SET, AND COMPARING THEM RAW WOULD
COMPARE TWO DIFFERENT EXPERIMENTS. `cache_tok11250` was built at **stride 8**
(2,809 frames); `cache_tok20000_s4` at **stride 4** (5,617). This is exactly the
derived-constant trap in `CLAUDE.md`: a cache parameter changed and the row set
changed with it. So this script INTERSECTS the two key sets and scores every arm
on the matched rows only, with folds and the `prior` recomputed on that subset.

⚠️ The matched subset is ~half the rows, so ABSOLUTE AP will be lower than
E-DETECT-1's — the head is already data-limited (see `e_detect_capacity.py`).
Every arm takes that handicap identically, so the COMPARISON is valid and the
absolute numbers are NOT comparable to the 5,617-row table. Do not quote them
into it.

Arms on the matched subset:
  prior        closed form, no features — the floor, recomputed on the subset
  pixel        raw 16x16x3 patches — the reference line E-DETECT-1 established
  v6@11250     the encoder 8,750 steps earlier
  v6@20000     the encoder E-DETECT-1 probed

TIER: T0-DIAGNOSTIC. Dev-box only; Thor untouched.
"""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

SP = Path(r"C:\Users\Admin\AppData\Local\Temp\claude"
          r"\G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD"
          r"\8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2\scratchpad")
sys.path.insert(0, str(SP))
import e_detect as E        # noqa: E402
import e_detect_prep as P   # noqa: E402
import e_trunk2_probe as T  # noqa: E402

OLD_CACHE = SP / "sp2/cache_tok11250/latents.pt"
OUT = P.OUT
N_TOK, D_TOK = 640, 768


def export_old() -> list[tuple[str, int]]:
    """Bank step-11,250 tokens + their keys. Streams row-by-row into a memmap so
    the 2.78 GB cache never coexists with a second full copy in RAM."""
    kp, tp = OUT / "v6_keys_11250.json", OUT / "v6_tokens_11250.npy"
    if kp.exists() and tp.exists():
        keys = [tuple(k) for k in json.loads(kp.read_text(encoding="utf-8"))]
        print(f"  [cached] step-11250 bank: {len(keys)} rows")
        return keys
    t0 = time.time()
    obj = torch.load(OLD_CACHE, map_location="cpu", weights_only=False)
    rows = [r for r in obj["rows"] if r.get("tokens") is not None]
    keys = [(r["clip_id"], int(r["frame_idx"])) for r in rows]
    m = np.lib.format.open_memmap(tp, mode="w+", dtype=np.float16,
                                  shape=(len(rows), N_TOK * D_TOK))
    for i, r in enumerate(rows):
        m[i] = r["tokens"].reshape(-1).to(torch.float16).numpy()
    m.flush()
    # ⛔ CONTENT CHECK. A pre-allocated memmap that never got written is a
    # full-size file of zeros that still passes `ls` — and here it would be the
    # arm the whole trajectory claim rests on.
    s = np.asarray(m[::200])
    if (np.abs(s).reshape(len(s), -1).max(1) == 0).any():
        raise SystemExit("[FATAL] step-11250 bank has all-zero rows")
    kp.write_text(json.dumps(keys), encoding="utf-8")
    del obj, rows, m
    gc.collect()
    print(f"  exported {len(keys)} rows in {time.time() - t0:.0f}s "
          f"(mean |x| {float(np.abs(s).mean()):.4f})")
    return keys


def main() -> None:
    new_keys = [tuple(k) for k in
                json.loads((P.FEAT / "keys.json").read_text(encoding="utf-8"))]
    old_keys = export_old()

    new_ix = {k: i for i, k in enumerate(new_keys)}
    old_ix = {k: i for i, k in enumerate(old_keys)}
    matched = [k for k in old_keys if k in new_ix]
    print(f"\n  step-11250 rows {len(old_keys):,} | step-20000 rows "
          f"{len(new_keys):,} | MATCHED {len(matched):,} "
          f"({100 * len(matched) / len(old_keys):.1f}% of the older set)")
    if len(matched) < 0.9 * len(old_keys):
        raise SystemExit("[FATAL] the stride-8 keys are NOT a subset of the "
                         "stride-4 keys — the two caches sample different "
                         "frames and no matched comparison is possible")

    sel_new = np.array([new_ix[k] for k in matched])
    sel_old = np.array([old_ix[k] for k in matched])
    ep = [k[0] for k in matched]
    occ = np.load(OUT / "occ.npy")[sel_new]
    folds = T.episode_folds(ep)
    tmp: dict[str, list[int]] = {}
    for i, e in enumerate(ep):
        tmp.setdefault(e, []).append(i)
    rows_by_ep = {k: np.array(v) for k, v in tmp.items()}
    base = float(occ.mean())
    pos_w = (1 - base) / base
    print(f"  matched subset: {len(matched):,} rows, {len(rows_by_ep)} episodes,"
          f" base rate {base:.4f}, {occ.sum(1).mean():.3f} occupied cells/frame")

    out = {"_evidence_class": "MEASURED (ours; dev-box RTX 4060)",
           "eval_tier": "T0-DIAGNOSTIC",
           "question": "is v6's encoder IMPROVING between step 11,250 and "
                       "20,000 on the validated detection probe?",
           "warning": "MATCHED SUBSET of 2,809-ish rows (stride-8 keys). "
                      "Absolute AP is NOT comparable to E-DETECT-1's 5,617-row "
                      "table — the head is data-limited. Only the WITHIN-TABLE "
                      "comparison is valid.",
           "n_matched": len(matched), "n_episodes": len(rows_by_ep),
           "base_rate": round(base, 6), "arms": {}}

    # ---- prior, recomputed on the subset ---------------------------------- #
    pr = np.zeros_like(occ, np.float32)
    for k, te in enumerate(folds):
        tr = np.concatenate([folds[j] for j in range(len(folds)) if j != k])
        pr[te] = occ[tr].mean(0)[None, :]
    preds = {"prior": pr}
    out["arms"]["prior"] = {"n_tok": 0, "d": 0,
                            **E.cluster_boot(pr, occ, rows_by_ep,
                                             np.random.default_rng(E.SEED)),
                            "topk_loc_err_m": round(E.topk_loc_err(pr, occ), 3)}
    a = out["arms"]["prior"]
    print(f"\n  {'prior':<14} AP {a['ap']:.4f} {a['ap_ci95']}  "
          f"AUC {a['auc']:.4f}", flush=True)

    banks = [
        ("pixel", np.load(SP / "sp2/detect/pixels.npy", mmap_mode="r")[sel_new],
         N_TOK, 768),
        ("v6@11250", np.load(OUT / "v6_tokens_11250.npy",
                             mmap_mode="r")[sel_old].reshape(-1, N_TOK, D_TOK),
         N_TOK, D_TOK),
        ("v6@20000", np.load(P.FEAT / "v6_tokens.npy",
                             mmap_mode="r")[sel_new].reshape(-1, N_TOK, D_TOK),
         N_TOK, D_TOK),
    ]
    for name, X, n_tok, d_in in banks:
        t0 = time.time()
        pred = np.zeros_like(occ, np.float32)
        taps = []
        for k, te in enumerate(folds):
            tr = np.concatenate([folds[j] for j in range(len(folds)) if j != k])
            pred[te], tap = E.run_fold(X, occ, tr, te, n_tok, d_in, pos_w)
            taps.append(tap)
            print(f"    {name} fold {k + 1}/{len(folds)} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        preds[name] = pred
        out["arms"][name] = {
            "n_tok": n_tok, "d": d_in,
            **E.cluster_boot(pred, occ, rows_by_ep,
                             np.random.default_rng(E.SEED)),
            "topk_loc_err_m": round(E.topk_loc_err(pred, occ), 3),
            "train_ap_mean": round(float(np.mean(taps)), 4),
            "train_s": round(time.time() - t0, 1)}
        r = out["arms"][name]
        print(f"  {name:<14} AP {r['ap']:.4f} {r['ap_ci95']}  "
              f"AUC {r['auc']:.4f}  loc {r['topk_loc_err_m']} m", flush=True)
        (SP / "e_detect_traj.json").write_text(json.dumps(out, indent=1),
                                               encoding="utf-8")
        del X
        gc.collect()

    # ---- the trajectory delta, PAIRED on the same resamples --------------- #
    out["paired"] = {}
    for a_, b_ in (("v6@20000", "v6@11250"), ("v6@20000", "pixel"),
                   ("v6@11250", "pixel"), ("v6@20000", "prior")):
        import e_detect_paired as PA
        rng = np.random.default_rng(E.SEED)
        d = PA.paired(preds[a_], preds[b_], occ, rows_by_ep, rng)
        out["paired"][f"{a_}_vs_{b_}"] = d
        sig = ("ABOVE" if d["d_ap_ci95"][0] > 0 else
               "BELOW" if d["d_ap_ci95"][1] < 0 else "indistinguishable")
        print(f"  {a_:<10} vs {b_:<10} dAP {d['d_ap']:+.4f} {d['d_ap_ci95']}"
              f"  -> {sig}", flush=True)
    (SP / "e_detect_traj.json").write_text(json.dumps(out, indent=1),
                                           encoding="utf-8")
    print(f"\n-> {SP / 'e_detect_traj.json'}")


if __name__ == "__main__":
    main()
