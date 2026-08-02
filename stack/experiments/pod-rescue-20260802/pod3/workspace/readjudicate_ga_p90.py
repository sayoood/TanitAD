"""TASK #36 — PI chose OPTION C: Ga gates on the LATERAL p90, not the open-loop mean.

Re-adjudicate every E1 arm/step under Ga-p90 and report whether ANY point now satisfies
BOTH the closed-loop primary AND the guardrail. If one does, it is the programme's first
D-A deliverable; if none does, closed-loop FT closes as a characterised negative even under
the relaxed-in-kind guardrail, which is a stronger statement than the old Ga could make.

⛔ POST-HOC BY CONSTRUCTION, AND LABELLED AS SUCH. These four arms were run and their p90s
were already visible when the PI chose option C. Nothing here may be read as a
pre-registered pass: a criterion applied after seeing the data cannot certify an arm. What
it CAN legitimately do is answer "does changing the statistic change the verdict?" — which
is a question about the CRITERION, not a claim about the arm. Any arm that looks like a
pass here must be re-run against a pre-registered Ga-p90 before it counts.

Estimator throughout: paired episode-cluster bootstrap already stored in the frontier
(967 windows / 44 episodes, B=2000). overlapping_holdout_se is not read.
"""
from __future__ import annotations

import glob
import json

FRONTIERS = {
    "E1c": "/workspace/e1c/e1c_frontier.json",
    "E1e-A": "/workspace/e1e/e1e_A_frontier.json",
    "E1e-B": "/workspace/e1e/e1e_B_frontier.json",
    "E1f": "/workspace/e1f/e1f_frontier.json",
}


def dig(o, *path):
    for p in path:
        if not isinstance(o, dict) or p not in o:
            return None
        o = o[p]
    return o


def fmt(d):
    if not isinstance(d, dict):
        return "        --        "
    return (f"{d.get('delta', float('nan')):+7.4f} "
            f"[{d.get('lo', float('nan')):+6.3f},{d.get('hi', float('nan')):+6.3f}] "
            f"{'SEP' if d.get('separated') else ' - '}")


def main() -> None:
    print("Ga-p90 RE-ADJUDICATION (POST-HOC — see docstring)\n")
    hits = []
    for arm, path in FRONTIERS.items():
        if not glob.glob(path):
            print(f"--- {arm}: frontier ABSENT at {path}")
            continue
        d = json.load(open(path))
        pts = d.get("points", {})
        print(f"--- {arm}   ({len(pts)} points)")
        print(f"    {'step':>5s}  {'P1 dep_overall':^26s}  {'Ga-p90 cross_p90@2s':^26s}  "
              f"{'old Ga ade_0_2s':^26s}  verdict")
        for step in sorted(pts, key=lambda s: int(s)):
            p = pts[step]
            p1 = (dig(p, "M0_corridor", "paired_delta_ft_minus_base", "departure_rate_overall")
                  or dig(p, "M0_corridor", "paired_delta_ft_minus_base", "dep_overall"))
            gp = dig(p, "M1_lateral_split", "openloop", "ego",
                     "paired_delta_ft_minus_base", "cross_p90@2s")
            ga = (dig(p, "M2_openloop", "paired_delta_ft_minus_base", "ade_0_2s")
                  or dig(p, "M1_lateral_split", "openloop", "ego",
                         "paired_delta_ft_minus_base", "ade_0_2s"))
            # P1 must be separated AND better (negative departure delta);
            # Ga-p90 must NOT be separated-worse.
            p1_ok = bool(p1 and p1.get("separated") and p1.get("delta", 0) < 0)
            gp_ok = bool(gp and not gp.get("separated"))
            verdict = "**BOTH**" if (p1_ok and gp_ok) else ("P1 only" if p1_ok else
                                                            ("Ga-p90 only" if gp_ok else "-"))
            if p1_ok and gp_ok:
                hits.append((arm, step, p1, gp))
            print(f"    {step:>5s}  {fmt(p1)}  {fmt(gp)}  {fmt(ga)}  {verdict}")
        print()

    print("=" * 100)
    if hits:
        print(f"⭐ {len(hits)} POINT(S) SATISFY BOTH the closed-loop primary AND Ga-p90:")
        for arm, step, p1, gp in hits:
            print(f"    {arm} @ step {step}: dep_overall {p1['delta']:+.4f} "
                  f"[{p1['lo']:+.4f},{p1['hi']:+.4f}] SEPARATED-better; "
                  f"cross_p90 {gp['delta']:+.4f} [{gp['lo']:+.4f},{gp['hi']:+.4f}] not separated")
        print("\n⛔ THIS IS NOT A PASS. The criterion was chosen after these numbers were "
              "visible. It must be RE-RUN against a pre-registered Ga-p90 to count.")
    else:
        print("NO point satisfies both. Switching Ga to the lateral p90 changes NO verdict, "
              "and closed-loop FT closes as a characterised negative under BOTH guardrails.")


if __name__ == "__main__":
    main()
