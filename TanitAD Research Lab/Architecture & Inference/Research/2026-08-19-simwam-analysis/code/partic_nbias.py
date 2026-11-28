"""H-RANK-21 — is the participation ratio N-BIASED, and does that alone explain
the 8.56-vs-40.77 floor contradiction?

THE STAKE.  `O6_PARTICIPATION_FLOOR = 8.56` is the gate every collapse verdict in
this programme is measured against — champ30k's FAIL (6.499 < 8.56) among them.
E-TRUNK-3 reports **40.77** for the SAME frozen DINOv3.  H-RANK-16 established
they are different instruments but not WHY they differ by 4.76x.

The one difference visible in both provenance strings is the SAMPLE SIZE:
    8.56  measured at n = 1440   (12 val clips x 120 frames)
    40.77 measured at n = 5617   (130 episodes)
and DINOv3's ambient d = 2048.  At n = 1440 the sample covariance is RANK
DEFICIENT (n < d): its eigenspectrum cannot be the population's.

⚠️ THIS IS A CONTROL EXPERIMENT, NOT A MEASUREMENT OF OUR MODEL.  It runs on a
SYNTHETIC population whose TRUE participation is known in closed form, so the
estimator's bias can be read off directly — the CLAUDE.md rule that a probe panel
must contain something that reads a known value.  No GPU, no model, no data.

PRE-REGISTERED PREDICTION (committed before running):
  * if participation is n-biased in this regime, P_hat(n) rises steeply with n
    while the TRUE value is constant, and the ratio P_hat(5617)/P_hat(1440)
    should land near the observed 40.77/8.56 = 4.76.
  * if P_hat is flat in n, sample size is NOT the explanation and the floor
    contradiction is a genuine instrument/representation difference.

Either way the actionable consequence is the same and is the point of the run:
a participation number is meaningless without its n, and two arms may only be
compared at MATCHED n.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

SP = Path(__file__).resolve().parent
OUT = SP / "h_rank21_partic_nbias.json"
RNG = np.random.default_rng(0)


def participation(eigs: np.ndarray) -> float:
    """p ∝ σ²: (Σλ)² / Σλ².  The statistic `spectrum_report` uses."""
    s = float(eigs.sum())
    return float(s * s / max(float((eigs ** 2).sum()), 1e-30))


def empirical_participation(X: np.ndarray) -> float:
    """Participation of the sample covariance of X [n, d] — centred, as in
    `spectrum_report` (which centres then takes svdvals, eig = sv**2)."""
    Xc = X - X.mean(0, keepdims=True)
    sv = np.linalg.svd(Xc, compute_uv=False)
    return participation(sv ** 2)


def main() -> int:
    D = 2048                      # DINOv3 ambient dim — the case the floor was set on
    NS = [180, 360, 720, 1440, 2880, 5617, 11234, 22468]
    REPS = 12

    # ---- a population with a KNOWN true participation ------------------------
    # power-law spectrum, the shape real representations show
    rep = {"_evidence_class": "MEASURED (ours; synthetic control, closed-form truth)",
           "hypothesis": "H-RANK-21 (pre-registered prediction committed before running)",
           "d": D, "reps_per_n": REPS, "populations": {}}
    print(f"\n  H-RANK-21 · participation-ratio finite-n bias · d={D}, {REPS} draws per n")
    print("  synthetic populations with a CLOSED-FORM true participation "
          "(the control that must read a known value)\n")

    for alpha, tag in ((1.0, "power-law a=1.0"), (2.0, "power-law a=2.0"), (0.0, "isotropic")):
        idx = np.arange(1, D + 1)
        lam = np.ones(D) if alpha == 0.0 else idx.astype(np.float64) ** (-alpha)
        lam = lam / lam.sum() * D
        true_p = participation(lam)
        sd = np.sqrt(lam)
        print(f"  {tag:<18} TRUE participation = {true_p:8.3f}")
        print(f"     {'n':>8}{'n/d':>7}{'P_hat mean':>12}{'sd':>8}{'P_hat/TRUE':>12}")
        rows = {}
        for n in NS:
            vals = []
            for _ in range(REPS):
                X = RNG.standard_normal((n, D)) * sd[None, :]
                vals.append(empirical_participation(X))
            m, s = float(np.mean(vals)), float(np.std(vals))
            rows[str(n)] = {"p_hat_mean": round(m, 3), "p_hat_sd": round(s, 3),
                            "ratio_to_true": round(m / true_p, 4)}
            print(f"     {n:>8}{n / D:>7.2f}{m:>12.3f}{s:>8.3f}{m / true_p:>12.4f}")
        r1440 = rows["1440"]["p_hat_mean"]
        r5617 = rows["5617"]["p_hat_mean"]
        ratio = r5617 / max(r1440, 1e-9)
        rows["_observed_gap_check"] = {
            "p_hat_1440": r1440, "p_hat_5617": r5617,
            "ratio_5617_over_1440": round(ratio, 3),
            "programme_observed_ratio_40p77_over_8p56": 4.763,
            "explains_gap": bool(0.6 * 4.763 <= ratio <= 1.6 * 4.763)}
        print(f"     -> P_hat(5617)/P_hat(1440) = {ratio:.3f}   "
              f"(programme's 40.77/8.56 = 4.763)  "
              f"{'CONSISTENT' if rows['_observed_gap_check']['explains_gap'] else 'not consistent'}\n")
        rep["populations"][tag] = {"true_participation": round(true_p, 3), "by_n": rows}

    # ---- the verdict --------------------------------------------------------
    cons = [t for t, v in rep["populations"].items()
            if v["by_n"]["_observed_gap_check"]["explains_gap"]]
    biased = [t for t, v in rep["populations"].items()
              if v["by_n"]["1440"]["ratio_to_true"] < 0.85]
    if biased:
        rep["verdict"] = (
            f"participation is STRONGLY N-BIASED at n<d — at n=1440 (n/d={1440 / D:.2f}) it reads "
            f"{', '.join(f'{t}: {rep['populations'][t]['by_n']['1440']['ratio_to_true']:.2f}x truth' for t in biased)}. "
            + (f"The 4.76x floor gap IS reproduced by sample size alone on: {cons}. "
               if cons else "The gap is NOT fully reproduced by sample size alone. ")
            + "⇒ a participation number is inadmissible without its n, and arms may only be "
              "compared at MATCHED n.")
    else:
        rep["verdict"] = ("participation is NOT materially n-biased in this regime ⇒ sample size "
                          "does NOT explain the 8.56 vs 40.77 contradiction; it is a genuine "
                          "instrument or representation difference.")
    print(f"  VERDICT: {rep['verdict']}")
    OUT.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(f"\n-> {OUT}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
