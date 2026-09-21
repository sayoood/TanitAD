"""Is the head's SIZE output worse than a zero-parameter floor? The number the SPEC needs.

`c920f15` established that the size TARGET is predictable — `l` and `w` are 75-78 % predictable from
target-side geometry — and concluded the head is not using available signal. ⛔ BUT THAT IS A
STATEMENT ABOUT THE TARGET, NOT ABOUT THE HEAD. It bounds what could be known; it never asked what
the head's own `l`/`w` are actually worth. A head-change SPEC cannot commit a success criterion
without that number, and `occ_calib.py` just showed today how far apart those two questions can be:
a field can beat its base rate handsomely and still lose to a free read.

⭐ THE FLOORS ARE THE POINT, AND THEY COST ZERO PARAMETERS EACH.

  F_GLOBAL  the median size over the FIT episodes. One number per component. The no-information
            control: a head that cannot beat it has added nothing at all.
  F_CLSHAT  the median size PER PREDICTED CLASS, using the head's OWN `cls_logits` argmax. This is
            the one that matters, because it is IMPLEMENTABLE — it uses only what the head already
            emits, exactly like `occ_calib.py`'s H2. Cars have typical widths; if a lookup table on
            the head's own class label beats its size regression, the regression is worse than free.
  F_CLSGT   the same table keyed on the GT class. NOT implementable — reported only to separate
            "the size head is bad" from "the class head is bad", because F_CLSHAT confounds them.
  O         the head's own `l`/`w`.

⛔ Every median is computed on the FIT episodes only and applied to the disjoint SCORED half.
⛔ Per-component, never pooled: `l` and `w` have different scales (MAE floors 1.03 vs 0.33 in
`quiet_fields.json`) and a pooled size error would be dominated by `l` and hide `w` entirely — the
same defect that hid a completely dead velocity component until `4a7166a` split it.
⚠️ Matched pairs come from `match_slots`, whose cost uses centre/cls/presence but NOT size
(`agent_slots.py:475-477`), so scoring size on these pairs is NOT circular — the same argument that
made velocity the clean place to test alignment in `align_vs_skill.py`.
⛔ CPU only, no GPU. Rows cached to `size_rows.npz`.
"""
from __future__ import annotations

import collections
import json
import pathlib
import random
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = pathlib.Path("D:/Projects/TanitAD")
sys.path.insert(0, str(REPO / "taniteval" / "tools"))
sys.path.insert(0, str(REPO / "stack"))

CACHE = pathlib.Path("size_rows.npz")
A8 = pathlib.Path("C:/Users/Admin/tanitad-caches/a8-occupancy-5k-20260919/run")
EP = "D:/Projects/TanitAD-artifacts/v2ep-eval124clean-416x1024cyl-halfB"
LAB = ("C:/Users/Admin/tanitad-caches/a7-imagenet-knockout-20260919/inputs/"
       "s2_labels_v8_eval.jsonl.gz")
AG = "D:/Projects/TanitAD-artifacts/b1-agent-join-3d-20260917/b1eval_agents_3d.jsonl.xz"
N_WIN = 60
B = 4000
SEED = 20260921


def build():
    import torch
    import s1_pass as SP                                  # noqa: E402
    from tanitad.models.agent_slots import match_slots    # noqa: E402
    corp = SP.Corpus(str(A8 / "config.json"), EP, LAB, AG, None)
    SP.attach_agent_gt(corp, corp.targs)
    mp = SP.ModelPass(corp, str(A8 / "ckpt_5000.pt"), "cpu")
    wis = [w for w in SP.trainer_windows(corp.ds, 1500)
           if corp.eligibility(w) is None][:N_WIN]
    R = []
    for wi in wis:
        item = corp.ds[wi]
        if not bool(item.get("agent_label", False)):
            continue
        e_i, _ = corp.ds.index[wi]
        out = mp.forward(item, str(corp.clip_ids[e_i]))
        sl = {k: v[0].float().cpu() for k, v in out["perception"]["box_slots"].items()
              if torch.is_tensor(v) and v.dim() >= 1}
        m = match_slots({k: v[None] for k, v in sl.items()},
                        {"box": item["agent_box"][None].float(),
                         "valid": item["agent_valid"][None], "cls": item["agent_cls"][None]})
        r, c = m["rows"][0].tolist(), m["cols"][0].tolist()
        tb = item["agent_box"].float().numpy()
        tc = item["agent_cls"].numpy()
        pb = sl["box"].numpy()
        pc = sl["cls_logits"].argmax(-1).numpy()
        for i, j in zip(r, c):
            R.append([float(e_i), float(tb[j, 2]), float(tb[j, 3]),
                      float(pb[i, 2]), float(pb[i, 3]),
                      float(tc[j]), float(pc[i])])
    A = np.asarray(R, dtype=np.float64)
    np.savez_compressed(CACHE, rows=A)
    print(f"built {A.shape[0]} matched pairs")
    return A


def table(keys_f, yf, keys_s, glob):
    """Median of yf per key, fit-side; unseen keys fall back to the global fit median."""
    d = collections.defaultdict(list)
    for k, v in zip(keys_f, yf):
        d[int(k)].append(v)
    med = {k: float(np.median(v)) for k, v in d.items()}
    return np.array([med.get(int(k), glob) for k in keys_s])


def main() -> int:
    A = np.load(CACHE)["rows"] if CACHE.exists() else build()
    epi, l_t, w_t, l_p, w_p, c_t, c_p = (A[:, k] for k in range(7))
    eps_ = sorted(set(epi.tolist()))
    half = set(eps_[: len(eps_) // 2])
    fit = np.array([e in half for e in epi])
    sc = ~fit
    assert fit.sum() > 20 and sc.sum() > 20, "ZZABORT a split side is too small"

    rgen = random.Random(SEED)
    es = epi[sc]
    keys = sorted(set(es.tolist()))
    idx = {k: np.where(es == k)[0] for k in keys}

    def ci(d):
        ms = sorted(float(np.concatenate([d[idx[k]] for k in rgen.choices(keys, k=len(keys))]).mean())
                    for _ in range(B))
        return [round(ms[int(0.025 * B)], 5), round(ms[int(0.975 * B)], 5)]

    res = {"_what": "is the head's SIZE output worse than a zero-parameter floor?",
           "_evidence_class": "MEASURED (ours), CPU, A8 ckpt_5000",
           "_non_circular": "match_slots' cost uses centre/cls/presence, NOT size "
                            "(agent_slots.py:475-477)",
           "_per_component": "l and w are never pooled — different scales, and pooling hid a dead "
                             "component once already (4a7166a)",
           "n_pairs": int(A.shape[0]), "n_episodes": len(eps_),
           "n_scored_pairs": int(sc.sum()), "n_scored_episodes": len(keys),
           "components": {}}

    for name, t, p in (("l", l_t, l_p), ("w", w_t, w_p)):
        glob = float(np.median(t[fit]))
        arms = {
            "O_head": np.abs(p[sc] - t[sc]),
            "F_GLOBAL_median": np.abs(np.full(int(sc.sum()), glob) - t[sc]),
            "F_CLSHAT_median_on_head_own_class": np.abs(
                table(c_p[fit], t[fit], c_p[sc], glob) - t[sc]),
            "F_CLSGT_median_on_GT_class": np.abs(
                table(c_t[fit], t[fit], c_t[sc], glob) - t[sc]),
        }
        row = {"fit_global_median_m": round(glob, 4),
               "MAE": {k: round(float(v.mean()), 5) for k, v in arms.items()}}
        for f in ("F_GLOBAL_median", "F_CLSHAT_median_on_head_own_class",
                  "F_CLSGT_median_on_GT_class"):
            d = arms[f] - arms["O_head"]            # >0 means the HEAD wins
            c = ci(d)
            row.setdefault("head_minus_floor", {})[f] = {
                "gain_floor_minus_head": round(float(d.mean()), 5), "CI95": c,
                "head_beats_floor": bool(c[0] > 0),
                "floor_beats_head": bool(c[1] < 0)}
        h = row["head_minus_floor"]["F_CLSHAT_median_on_head_own_class"]
        row["reading"] = ("⛔ WORSE THAN FREE — a median lookup on the head's OWN predicted class "
                          "beats its size regression" if h["floor_beats_head"] else
                          "⭐ the head beats the implementable zero-parameter floor"
                          if h["head_beats_floor"] else
                          "⚠️ indistinguishable from a zero-parameter class-median lookup")
        res["components"][name] = row

    rl = res["components"]["l"]["reading"]
    rw = res["components"]["w"]["reading"]
    res["_VERDICT"] = (f"l: {rl}\nw: {rw}")
    print(json.dumps(res, indent=1, ensure_ascii=False))
    print("\n" + res["_VERDICT"])
    pathlib.Path("size_floor.json").write_text(json.dumps(res, indent=1, ensure_ascii=False),
                                               encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
