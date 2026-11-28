#!/usr/bin/env python3
"""FIX 4 — re-adjudicate the three situation targets with the REPAIRED firewall.

`…/2026-07-26-situation-classifier/artifacts/sc_results.json` records
`verdict: CIRCULAR` on all three situation targets while its own
`context_leaks = false` refutes that verdict on every one. Two degenerate routes:

  1. `blind >= 1 - DETERMINISTIC_EPS` fires on the MAJORITY CLASS itself at a
     positive rate of 0.0030 (`roundabout`);
  2. `vision_buys_nothing` compares ACCURACIES, and a recall-seeking rare-event
     model must lose that comparison to "always predict negative"
     (`intersection`: real 0.8194 vs majority 0.9743).

⚠️ The situation-classifier author FOUND route 1 and wrote it into the record's
own `MDE_AUDIT` ("DEGENERATE — the CIRCULAR branch cannot fail on this target").
The record still says `verdict: CIRCULAR`, and the packaged module still shipped
the defect. That is the "escalate integration, don't write it into a doc"
failure class: a correct diagnosis living next to the wrong verdict.

This script re-runs the PACKAGED firewall — repaired — on the same held-out
frames, the same context construction and the same subsample rule as
`sc_eval.py:249-268`, and reports the verdict before and after.

Usage:
  python fix4_readjudicate_situations.py --sc <2026-07-26-situation-classifier> \
      --out <…/2026-07-27-confirmed-fixes/raw>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

BLIND_MAX_ROWS = 40_000            # sc_eval.py:45, verbatim


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sc", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n-boot", type=int, default=400)   # sc_eval.py's own value
    a = p.parse_args()
    sc_dir, outd = Path(a.sc).resolve(), Path(a.out).resolve()
    outd.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path(__file__).resolve().parents[6] / "taniteval"))
    from taniteval.blind_baseline import blind_conditioning_baseline

    art = sc_dir / "artifacts"
    published = json.loads((art / "sc_results.json").read_text(encoding="utf-8"))
    D = np.load(art / "heldout_frames.npz", allow_pickle=True)
    sits = [str(s) for s in D["situations"]]
    out = {"block": "confirmed_fixes/FIX4_rare_event_firewall",
           "source": "…/2026-07-26-situation-classifier/artifacts/"
                     "heldout_frames.npz + sc_results.json",
           "context_construction": "sc_eval.py:251-254, verbatim",
           "n_boot": a.n_boot, "situations": {}}

    for i, sit in enumerate(sits):
        m = D["valid"][:, i].astype(bool)
        y = D["y"][m, i].astype(np.int64)
        eid = D["clip_cluster"][m]
        ee = D["ego"][m]
        real = (D["head_img_ego"][m, i] > 0.5).astype(np.int64)
        ctx = {"v_bin": np.digitize(ee[:, 0], np.arange(0, 30, 2.0)).astype(np.int64),
               "alon_bin": np.digitize(ee[:, 1], np.arange(-4, 4, 0.5)).astype(np.int64),
               "omega_bin": np.digitize(ee[:, 2], np.arange(-0.6, 0.6, 0.05)).astype(np.int64)}
        keep = np.ones(len(y), bool)
        cl = np.unique(eid)
        if len(y) > BLIND_MAX_ROWS:
            rg = np.random.default_rng(0)
            take = set(rg.choice(cl, size=max(40, int(len(cl) * BLIND_MAX_ROWS / len(y))),
                                 replace=False).tolist())
            keep = np.array([e in take for e in eid])
        fw = blind_conditioning_baseline(
            {k: v[keep] for k, v in ctx.items()}, y[keep], eid[keep],
            real_pred=real[keep], problem=f"situation:{sit}", n_boot=a.n_boot)

        pub = published["situations"][sit]["C_BLIND"]
        rec = {
            "published": {k: pub.get(k) for k in
                          ("verdict", "blind_skill_over_majority",
                           "target_is_deterministic_in_context",
                           "vision_buys_nothing", "context_leaks", "summary")},
            "published_MDE_AUDIT_already_said": pub.get("MDE_AUDIT", {}).get("reading"),
            "repaired": {k: fw.get(k) for k in
                         ("verdict", "statistic", "admissible",
                          "blind_skill_over_majority",
                          "target_is_deterministic_in_context",
                          "vision_buys_nothing", "context_leaks", "summary")},
            "degeneracy_audit": fw["degeneracy_audit"],
            "reproduction": {
                "published_blind_accuracy": pub.get("blind_accuracy", {}).get("mean"),
                "repaired_blind_accuracy": fw["blind_accuracy"]["mean"],
                "published_majority": pub.get("majority_base_rate", {}).get("mean"),
                "repaired_majority": fw["majority_base_rate"]["mean"],
                "published_real": pub.get("real_model_accuracy", {}).get("mean"),
                "repaired_real": fw["real_model_accuracy"]["mean"],
                "n_windows_published": pub.get("n_windows"),
                "n_windows_repaired": fw["n_windows"]},
            "VERDICT_CHANGED": bool(pub.get("verdict") != fw["verdict"]),
        }
        out["situations"][sit] = rec
        print(f"[{sit:13s}] {pub.get('verdict')} -> {fw['verdict']} "
              f"(on {fw['statistic']}) | blind {fw['blind_accuracy']['mean']:.4f} "
              f"maj {fw['majority_base_rate']['mean']:.4f} "
              f"real {fw['real_model_accuracy']['mean']:.4f} "
              f"| bal_blind {fw['degeneracy_audit']['balanced_accuracy_blind']} "
              f"bal_real {fw['degeneracy_audit']['balanced_accuracy_real']}",
              flush=True)

    ch = [s for s, r in out["situations"].items() if r["VERDICT_CHANGED"]]
    out["headline"] = {
        "changed": ch,
        "_read": ("every situation target was published CIRCULAR — i.e. "
                  "INADMISSIBLE, its scores 'measuring the lookup, not the "
                  "model' — on a firewall whose own `context_leaks = false` "
                  "said the context carries nothing. The repaired firewall "
                  "decides on balanced accuracy, whose floor is 1/n_class at "
                  "any imbalance."),
        "what_this_does_NOT_say": (
            "nothing here says the situation heads WORK. It says the "
            "circularity firewall was not the instrument that retired them. "
            "The pre-registered primary comparison (the AP-based vs_head_ego "
            "contrast) is untouched."),
    }
    (outd / "fix4_situation_readjudication.json").write_text(
        json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"[write] {outd / 'fix4_situation_readjudication.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
