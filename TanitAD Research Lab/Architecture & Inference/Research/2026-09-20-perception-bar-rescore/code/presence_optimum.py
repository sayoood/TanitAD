"""Is the presence head sitting at the ANALYTIC CONSTANT OPTIMUM of its own loss?

`286e0d3` established the head has no per-slot signal and named three unseparated candidates. One
(under-training, via an earlier checkpoint) was ruled untestable — both banked checkpoints are
step 5,000. This tests the ∅-WEIGHT candidate WITHOUT retraining, by derivation.

THE DERIVATION. `agent_slots.py:556-565` is a weighted BCE: matched slots carry target 1 at weight
1.0, unmatched carry target 0 at weight `NO_OBJECT_W` (= 0.1 from source, :232). For a head with
NO per-slot information the best it can do is one constant σ per window. With a matched fraction
p, the weighted loss is

    L(σ) = p·1·(−log σ) + (1−p)·w·(−log(1−σ))
    dL/dσ = −p/σ + (1−p)w/(1−σ) = 0
    ⇒  σ* = p / (w + p(1−w))

⭐ THE TEST THAT CAN FAIL, and it is per-window rather than aggregate: p varies window to window
with the target count, so σ* varies too. If the head has learned the loss's constant solution, its
observed mean presence must TRACK σ* ACROSS WINDOWS. If the observed mean is flat while σ* moves,
the head has not even learned the per-window rate and the ∅-weight story is wrong.

⛔ CONTROLS: the flat predictor (one global σ* from the pooled p) is scored beside the per-window
one — if per-window σ* does not beat it, the tracking claim is not supported.
⛔ CPU only, matcher-free.
"""
from __future__ import annotations

import json
import pathlib
import statistics as st
import sys

import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = pathlib.Path("D:/Projects/TanitAD")
sys.path.insert(0, str(REPO / "taniteval" / "tools"))
sys.path.insert(0, str(REPO / "stack"))
import s1_pass as SP                                      # noqa: E402
from tanitad.models.agent_slots import NO_OBJECT_W        # noqa: E402  canonical, not retyped

A8 = pathlib.Path("C:/Users/Admin/tanitad-caches/a8-occupancy-5k-20260919/run")
EP = "D:/Projects/TanitAD-artifacts/v2ep-eval124clean-416x1024cyl-halfB"
LAB = ("C:/Users/Admin/tanitad-caches/a7-imagenet-knockout-20260919/inputs/"
       "s2_labels_v8_eval.jsonl.gz")
AG = "D:/Projects/TanitAD-artifacts/b1-agent-join-3d-20260917/b1eval_agents_3d.jsonl.xz"
N_WIN = 40


def sigma_star(p: float, w: float) -> float:
    return p / (w + p * (1.0 - w)) if (w + p * (1.0 - w)) else float("nan")


def main() -> int:
    w = float(NO_OBJECT_W)
    print(f"NO_OBJECT_W from source = {w}")
    corp = SP.Corpus(str(A8 / "config.json"), EP, LAB, AG, None)
    SP.attach_agent_gt(corp, corp.targs)
    mp = SP.ModelPass(corp, str(A8 / "ckpt_5000.pt"), "cpu")
    wis = [w_ for w_ in SP.trainer_windows(corp.ds, 1000)
           if corp.eligibility(w_) is None][:N_WIN]

    rows = []
    for wi in wis:
        item = corp.ds[wi]
        if not bool(item.get("agent_label", False)):
            continue
        e_i, _ = corp.ds.index[wi]
        out = mp.forward(item, str(corp.clip_ids[e_i]))
        pr = torch.sigmoid(out["perception"]["box_slots"]["presence_logit"][0].float().cpu())
        n_q = int(pr.numel())
        n_t = int(item["agent_valid"].numpy().astype(bool).sum())
        rows.append({"p": n_t / n_q, "obs": float(pr.mean()),
                     "pred": sigma_star(n_t / n_q, w), "n_t": n_t})
    assert rows, "control: zero windows"

    obs = [r["obs"] for r in rows]
    pred = [r["pred"] for r in rows]
    p_pool = st.mean(r["p"] for r in rows)
    flat = sigma_star(p_pool, w)                    # the CONTROL: one global sigma*

    mae_pred = st.mean(abs(a - b) for a, b in zip(pred, obs))
    mae_flat = st.mean(abs(flat - b) for b in obs)
    # Pearson r between predicted and observed
    mo, mp_ = st.mean(obs), st.mean(pred)
    num = sum((a - mp_) * (b - mo) for a, b in zip(pred, obs))
    den = (sum((a - mp_) ** 2 for a in pred) * sum((b - mo) ** 2 for b in obs)) ** 0.5
    r = num / den if den else float("nan")

    res = {"_what": "is presence sitting at the analytic constant optimum of its own weighted BCE?",
           "_evidence_class": "MEASURED (ours) + DERIVED, CPU, A8 ckpt_5000, matcher-free",
           "_formula": "sigma* = p / (w + p(1-w)),  w = NO_OBJECT_W",
           "NO_OBJECT_W": w, "n_windows": len(rows),
           "pooled_matched_fraction_p": round(p_pool, 4),
           "predicted_sigma_star_pooled": round(flat, 4),
           "observed_mean_presence": round(st.mean(obs), 4),
           "abs_gap_pooled": round(abs(flat - st.mean(obs)), 4),
           "per_window_MAE_vs_sigma_star": round(mae_pred, 4),
           "per_window_MAE_vs_FLAT_control": round(mae_flat, 4),
           "pearson_r_pred_vs_obs": round(r, 4),
           "tracks_per_window": bool(mae_pred < mae_flat)}
    res["_VERDICT"] = (
        "⭐ the head sits at the loss's CONSTANT OPTIMUM and TRACKS it per window ⇒ it has learned "
        "the marginal rate and nothing per-slot; the ∅ weight sets where that constant lands"
        if res["tracks_per_window"] and res["abs_gap_pooled"] < 0.05 else
        "the head sits near a constant but does NOT track sigma* per window ⇒ the ∅-weight account "
        "is NOT supported by this test")
    # what weight would put the degenerate constant at 0.5, i.e. below a usable threshold
    res["w_for_sigma_star_0.5"] = round(p_pool / (1 - p_pool), 4)
    print(json.dumps(res, indent=1))
    print("\n" + res["_VERDICT"])
    pathlib.Path("presence_optimum.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
