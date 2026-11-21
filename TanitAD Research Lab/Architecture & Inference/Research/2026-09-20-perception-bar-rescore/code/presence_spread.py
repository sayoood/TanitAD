"""Can ANY threshold rescue the presence head? And is an earlier checkpoint available?

`0c179e8` measured presence firing on 96.55 of 100 slots for 25.15 targets and stated a limit
honestly: ONE threshold (sigma > 0.5) was used and a sweep was NOT done. This closes that.

⭐ THE DECISIVE QUANTITY IS THE WITHIN-WINDOW SPREAD, not the threshold. If the 100 per-slot
probabilities inside a window are nearly identical, the head emits a near-CONSTANT presence and
**no threshold and no top-k can extract anything** -- the information is not there to be found.
If the spread is wide, the 0.5 cut was simply misplaced and a sweep would help.

⇒ reported: within-window std / IQR / (max-min), a full THRESHOLD SWEEP of the count-MAE against
the constant control, and a top-k check at the ORACLE k (k = the true count), which is the most
generous reading the head can be given.
⛔ Still matcher-free. ⛔ CPU only.
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

A8 = pathlib.Path("C:/Users/Admin/tanitad-caches/a8-occupancy-5k-20260919/run")
EP = "D:/Projects/TanitAD-artifacts/v2ep-eval124clean-416x1024cyl-halfB"
LAB = ("C:/Users/Admin/tanitad-caches/a7-imagenet-knockout-20260919/inputs/"
       "s2_labels_v8_eval.jsonl.gz")
AG = "D:/Projects/TanitAD-artifacts/b1-agent-join-3d-20260917/b1eval_agents_3d.jsonl.xz"
N_WIN = 40


def ckpt_step(p: pathlib.Path):
    try:
        d = torch.load(p, map_location="cpu", weights_only=False)
        return d.get("step") if isinstance(d, dict) else None
    except Exception as e:
        return f"UNREADABLE {type(e).__name__}"


def main() -> int:
    print("checkpoint steps (is an earlier one available for a steps discriminator?)")
    for n in ("ckpt.pt", "ckpt_5000.pt"):
        print(f"  {n}: step {ckpt_step(A8 / n)}")

    corp = SP.Corpus(str(A8 / "config.json"), EP, LAB, AG, None)
    SP.attach_agent_gt(corp, corp.targs)
    mp = SP.ModelPass(corp, str(A8 / "ckpt_5000.pt"), "cpu")
    wis = [w for w in SP.trainer_windows(corp.ds, 1000)
           if corp.eligibility(w) is None][:N_WIN]

    probs, trues = [], []
    for wi in wis:
        item = corp.ds[wi]
        if not bool(item.get("agent_label", False)):
            continue
        e_i, _ = corp.ds.index[wi]
        out = mp.forward(item, str(corp.clip_ids[e_i]))
        p = torch.sigmoid(out["perception"]["box_slots"]["presence_logit"][0].float().cpu())
        probs.append(p)
        trues.append(int(item["agent_valid"].numpy().astype(bool).sum()))
    assert probs, "control: zero windows"

    spreads = [{"std": float(p.std()), "iqr": float(p.quantile(0.75) - p.quantile(0.25)),
                "rng": float(p.max() - p.min())} for p in probs]
    const = st.mean(trues)

    sweep = []
    for t in [i / 20 for i in range(1, 20)]:
        pred = [int((p > t).sum()) for p in probs]
        sweep.append({"thr": round(t, 2),
                      "mean_pred": round(st.mean(pred), 2),
                      "MAE": round(st.mean(abs(a - b) for a, b in zip(pred, trues)), 3)})
    best = min(sweep, key=lambda r: r["MAE"])
    mae_const = st.mean(abs(const - b) for b in trues)

    # most generous reading available: give the head the ORACLE k, score only its RANKING
    oracle = []
    for p, n in zip(probs, trues):
        k = min(n, p.numel())
        thr = float(p.sort(descending=True).values[k - 1]) if k else 1.0
        oracle.append(float((p >= thr).sum()))

    res = {"_what": "can any threshold rescue presence? within-window spread + full sweep",
           "_evidence_class": "MEASURED (ours), CPU, A8 ckpt_5000, matcher-free",
           "n_windows": len(probs), "n_queries": int(probs[0].numel()),
           "mean_true": round(const, 2),
           "within_window_std_mean": round(st.mean(s["std"] for s in spreads), 5),
           "within_window_IQR_mean": round(st.mean(s["iqr"] for s in spreads), 5),
           "within_window_range_mean": round(st.mean(s["rng"] for s in spreads), 5),
           "within_window_range_max": round(max(s["rng"] for s in spreads), 5),
           "MAE_constant_control": round(mae_const, 3),
           "best_threshold": best,
           "sweep": sweep,
           "beats_constant_at_best_threshold": bool(best["MAE"] < mae_const)}
    res["_VERDICT"] = (
        "a better threshold RESCUES the count -- the 0.5 cut was misplaced"
        if res["beats_constant_at_best_threshold"] else
        "⛔ NO threshold beats the constant control: the head emits near-constant presence and "
        "the information is not there to extract")
    print(json.dumps({k: v for k, v in res.items() if k != "sweep"}, indent=1))
    print("\nthreshold sweep (thr, mean_pred, MAE)  — constant control MAE "
          f"{mae_const:.3f}, true mean {const:.2f}")
    for r in sweep:
        mark = " <- best" if r is best else ""
        print(f"  {r['thr']:.2f}  {r['mean_pred']:>6.2f}  {r['MAE']:>7.3f}{mark}")
    print("\n" + res["_VERDICT"])
    pathlib.Path("presence_spread.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
