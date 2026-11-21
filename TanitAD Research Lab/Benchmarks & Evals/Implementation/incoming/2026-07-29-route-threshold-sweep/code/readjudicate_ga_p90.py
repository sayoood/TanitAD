"""TASK #36 (PI chose OPTION C) — re-adjudicate every E1 point under Ga = LATERAL p90.

POST-HOC AND LABELLED AS SUCH: these arms ran and their p90s were visible before option C
was chosen. This answers "does changing the STATISTIC change the VERDICT?" — a question
about the criterion. It cannot certify an arm; that needs a pre-registered re-run.

Keys verified against the real frontier structure (my first attempt used wrong paths and
silently produced an all-empty P1 column, which would have read as a false negative).
"""
import glob, json

FRONTIERS = {
    "E1c":   ["/workspace/e1c/e1c_frontier_result.json", "/workspace/e1c/e1c_frontier.json"],
    "E1e-A": ["/workspace/e1e/e1e_A_frontier.json"],
    "E1e-B": ["/workspace/e1e/e1e_B_frontier.json"],
    "E1f":   ["/workspace/e1f/e1f_frontier.json", "/workspace/e1f/e1f_frontier_result.json"],
}

def dig(o, *path):
    for p in path:
        if not isinstance(o, dict) or p not in o: return None
        o = o[p]
    return o

print("Ga-p90 RE-ADJUDICATION  (POST-HOC — cannot certify, only tests the criterion)\n")
hits = []
for arm, paths in FRONTIERS.items():
    path = next((p for p in paths if glob.glob(p)), None)
    if not path:
        print(f"--- {arm}: frontier ABSENT ({paths})"); continue
    pts = json.load(open(path)).get("points", {})
    print(f"--- {arm}  [{path.split('/')[-1]}]  {len(pts)} points")
    print(f"    {'step':>5s} {'P1 dep_overall':>16s} {'P1sep':>6s} | "
          f"{'Ga-p90 delta':>13s} {'CI95':>18s} {'sep':>4s} | {'oldGa':>6s} | verdict")
    for step in sorted(pts, key=lambda s: int(s)):
        p = pts[step]; E = p.get("EVAL", {})
        d1 = E.get("_dep_overall_delta")
        s1 = E.get("P1_dep_overall_separated_lower")
        ga_old = E.get("Ga_openloop_ade2s_ok")
        gp = dig(p, "M1_lateral_split", "openloop", "ego",
                 "paired_delta_ft_minus_base", "cross_p90@2s") or {}
        gp_sep = gp.get("separated")
        p1_ok = bool(s1) and (d1 is not None and d1 < 0)
        gp_ok = (gp_sep is False)
        v = "**BOTH**" if (p1_ok and gp_ok) else ("P1 only" if p1_ok else ("Ga-p90 only" if gp_ok else "-"))
        if p1_ok and gp_ok: hits.append((arm, step, d1, gp))
        print(f"    {step:>5s} {(d1 if d1 is not None else float('nan')):+16.4f} {str(s1):>6s} | "
              f"{gp.get('delta', float('nan')):+13.4f} "
              f"[{gp.get('lo', float('nan')):+.3f},{gp.get('hi', float('nan')):+.3f}] "
              f"{str(gp_sep):>4s} | {str(ga_old):>6s} | {v}")
    print()

print("=" * 104)
if hits:
    print(f"{len(hits)} POINT(S) satisfy BOTH the closed-loop primary AND Ga-p90:")
    for arm, step, d1, gp in hits:
        print(f"   {arm} @ {step}: dep_overall {d1:+.4f} SEPARATED-better ; "
              f"cross_p90 {gp['delta']:+.4f} [{gp['lo']:+.4f},{gp['hi']:+.4f}] NOT separated")
    print("\n⛔ NOT A PASS — criterion chosen after these numbers were visible. Requires a "
          "pre-registered re-run to count.")
else:
    print("NO point satisfies both => switching Ga to the lateral p90 changes NO verdict.")
