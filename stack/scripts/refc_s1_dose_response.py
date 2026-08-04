"""⭐ THE DENOISE-DEPTH DOSE-RESPONSE — POST-HOC, and labelled as such.

It decides NO registered branch and moved NO threshold. It exists because E-S1-0's
S1b row came back separated in the ADVERSE direction (prereg §4.3's `S1b-ADVERSE`),
and the natural question — *why does scoring the object you actually emit make the
ranking worse?* — has a cheap, sharp answer available from the SAME weights.

THE MECHANISM UNDER TEST. `loss_cls` supervises REF-C's `conf_head` on exactly ONE
distribution: the RAW ANCHOR VOCABULARY at timestep token t=0. Every other readout
of that same head is off-distribution, and the further the input is from the
anchors the further off it should be. That predicts a MONOTONE DOSE-RESPONSE in
denoise depth, which is exactly what this measures — on one axis, one arm, one set
of weights:

    conf(anchors, t=0)   `logits`            SUPERVISED         <- the shipped ranker
    conf(X_1,     t=2)   `prefinal_logits`   1 denoise step in  <- what S1 ranks today
    conf(X_2,     t=2)   `emitted_logits`    2 steps in, EMITTED<- S1b
    conf(X_2,     t=0)   `emitted_t0_logits` 2 steps in, but the SUPERVISED token

⭐ THE FOURTH ROW IS THE DISCRIMINATOR. If the degradation is about the TRAJECTORY
being off-distribution, pinning t=0 changes little. If a large part of it is the
TIME TOKEN — a token `loss_cls` never trains — then the fourth row recovers, and
the fix is free (`--sel-score-emitted-t 0`, still 0 parameters).

⛔ Both E-SEL and S3_DEPLOYABLE left "`conf_head` is off its training distribution"
as an explicitly UNTESTED HYPOTHESIS. This is the test, with a dose axis.

Estimator: paired episode-cluster bootstrap, unit = episode, n_boot = 2000, on the
canonical 881 windows. Restricted to the S2-reachable survivors FIRST, and the
headline is SELECTION ADE — never rho (S3_DEPLOYABLE §3).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("OMP_NUM_THREADS", "6")

_HERE = Path(__file__).resolve()
for _p in (Path.home() / "TanitAD" / "taniteval", Path.home() / "TanitAD" / "stack",
           Path.home() / "TanitAD" / "stack" / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import refc_sel_probe as P                                        # noqa: E402
import refc_s1_climbout_probe as C                                # noqa: E402

READOUTS = [
    ("logits", "conf(anchors, t=0)", "SUPERVISED by loss_cls", 0),
    ("prefinal_logits", "conf(X_1, t=2)", "unsupervised, 1 denoise step in", 1),
    ("emitted_logits", "conf(X_2, t=2)", "unsupervised, 2 steps in — EMITTED", 2),
    ("emitted_t0_logits", "conf(X_2, t=0)",
     "unsupervised trajectory, SUPERVISED token", 2),
]


def run(bank: str, emitted: str, arm: str, out_dir: str) -> dict:
    t0 = time.time()
    d = P.load_fan(bank)
    em = torch.load(emitted, map_location="cpu", weights_only=False)
    eid = list(d["eid"])
    de_all = P.candidate_ade(d["fan"], d["gt"])
    de_or = de_all.min(1).values
    keep, reach = C.survivor_mask(d)

    ctl = {
        "fan_bit_identical_to_esel": bool(torch.equal(d["fan"], em["fan"])),
        "gt_bit_identical": bool(torch.equal(d["gt"], em["gt"])),
        "eid_match": eid == list(em["eid"]),
        "prefinal_reproduces_esel_refined": bool(
            torch.equal(d["refined_logits"], em["prefinal_logits"])),
        "logits_bit_identical": bool(torch.equal(d["logits"], em["logits"])),
        "t0_differs_from_t2": not bool(torch.equal(em["emitted_t0_logits"],
                                                   em["emitted_logits"])),
        "can_fire": True,
    }
    src = {"logits": d["logits"], "prefinal_logits": em["prefinal_logits"],
           "emitted_logits": em["emitted_logits"],
           "emitted_t0_logits": em["emitted_t0_logits"]}

    rows, per = {}, {}
    for key, what, note, depth in READOUTS:
        idx = C.argmax_over_survivors(src[key], keep)
        blk = P.ranker_block(de_all, de_or, idx, eid, tag=key)
        per[key] = blk["_per_window_ade"]
        rows[key] = {
            "scores": what, "supervision": note, "denoise_depth": depth,
            **{k: v for k, v in blk.items() if not k.startswith("_")},
            "rank_acc_x_chance": round(float(blk["rank_acc"]["mean"]
                                             * d["logits"].shape[1]), 2),
            "corr_with_shipped": round(float(np.corrcoef(
                d["logits"].flatten().numpy(),
                src[key].flatten().numpy())[0, 1]), 4),
        }
    paired = {}
    for a, b in (("prefinal_logits", "logits"),
                 ("emitted_logits", "prefinal_logits"),
                 ("emitted_logits", "logits"),
                 ("emitted_t0_logits", "emitted_logits"),
                 ("emitted_t0_logits", "logits"),
                 ("emitted_t0_logits", "prefinal_logits")):
        paired[f"{a}__minus__{b}"] = P._paired(per[a].numpy(), per[b].numpy(), eid)

    # the four families for the readout that could recover — per family, paired
    fams = {}
    for key in ("emitted_logits", "emitted_t0_logits"):
        idx = C.argmax_over_survivors(src[key], keep)
        fams[key] = P.family_paired(d, idx, d["sel"], eid,
                                    tag=f"{key}-minus-shipped")

    ade = [rows[k]["ade_0_2s"]["mean"] for k, *_ in READOUTS[:3]]
    res = {
        "experiment": ("denoise-depth dose-response of REF-C's conf_head — "
                       "POST-HOC diagnostic, decides no registered branch"),
        "arm": arm, "n_windows": int(d["fan"].shape[0]),
        "n_episodes": len(set(eid)), "n_anchors": int(d["logits"].shape[1]),
        "chance_rank_acc": round(1.0 / d["logits"].shape[1], 6),
        "status": ("POST-HOC. Added after E-S1-0 returned S1b-ADVERSE. It moved "
                   "no threshold and adjudicated no branch — the registered "
                   "verdict stands on its own artifact."),
        "controls": ctl,
        "reachability": reach,
        "readouts": rows,
        "paired": paired,
        "monotone_in_denoise_depth": bool(ade[0] < ade[1] < ade[2]),
        "families_paired_vs_shipped": fams,
        "prereg_pin": C._pin(),
        "estimator": C.PREREG_THRESHOLDS["estimator"],
        "wall_s": round(time.time() - t0, 1),
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"s1_dose_response_{arm}.json").write_text(
        json.dumps(P._clean(res), indent=2), encoding="utf-8")
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--emitted", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    r = run(a.bank, a.emitted, a.arm, a.out)
    print(json.dumps({"controls": r["controls"],
                      "monotone": r["monotone_in_denoise_depth"],
                      "readouts": {k: {"ade": v["ade_0_2s"]["mean"],
                                       "rank_acc": v["rank_acc"]["mean"],
                                       "x_chance": v["rank_acc_x_chance"],
                                       "f2x": v["frac_sel_2x_worse"]["mean"]}
                                   for k, v in r["readouts"].items()},
                      "paired": {k: [v["delta"], v["lo"], v["hi"],
                                     v["separated"]]
                                 for k, v in r["paired"].items()}}, indent=1))
    return 0


if __name__ == "__main__":                                     # pragma: no cover
    raise SystemExit(main())
