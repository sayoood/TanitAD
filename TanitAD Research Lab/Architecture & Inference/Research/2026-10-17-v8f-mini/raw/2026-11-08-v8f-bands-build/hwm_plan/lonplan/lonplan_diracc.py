"""LON-PLAN SCORE-side DIR-ACC table (AMEND_HWMPLAN_3 section 5). Beside, no bar.
Per arm: Pearson r of x = mean over knots 1-10 of (plan acc - a0) and y = the same for the logged acc, over the SCORE windows; CI from the episode-cluster bootstrap (B 2,000, seed 0) through lonplan_select.dir_acc (the one implementation the TRAIN selection uses).
usage: python lonplan_diracc.py --windows DIR/windows.npz --arm A0_s0=<plans.npz> --arm A1_s0=... --arm R_s0=... --arm S_s0=... --out OUT.json
Writes OUT.json and OUT.md (a table). The estimator answers another draw of episodes only. T0 diagnostic."""
import argparse, sys
sys.dont_write_bytecode = True
import lonplan_common as C
import numpy as np
import lonplan_select as S


def diracc_table(windows_path, arms, n_boot=2000, seed=0):
    with np.load(windows_path, allow_pickle=False) as w:
        A = w["A"].astype(np.float64); eid = w["eid"]
    rows = {}
    for name, path in arms:
        with np.load(path, allow_pickle=False) as z:
            plan = z["plan_raw"].astype(np.float64)
        assert plan.shape[0] == len(A), (name, plan.shape, len(A))
        rows[name] = dict(S.dir_acc(plan, A, eid, n_boot, seed), plans=path, plans_sha256=C.sha256_file(path))
    return dict(kind="LONPLAN_DIRACC", windows=windows_path, windows_sha256=C.sha256_file(windows_path), estimator="episode_cluster_bootstrap", B=int(n_boot), seed=int(seed), tier="T0", arms=rows,
                note="DIR-ACC has no bar. n/a when x or y is constant.")


def markdown(tab):
    f = lambda v: "n/a" if v != v else f"{v:+.4f}"
    L = ["| arm | DIR-ACC r | 95% CI | n windows | n episodes |", "|---|---|---|---|---|"]
    for k, r in tab["arms"].items():
        L.append(f"| {k} | {f(r['r'])} | [{f(r['lo'])}, {f(r['hi'])}] | {r['n_windows']} | {r.get('n_episodes', 'n/a')} |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", required=True); ap.add_argument("--arm", action="append", required=True, help="NAME=plans.npz"); ap.add_argument("--out", required=True); ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    arms = [tuple(x.split("=", 1)) for x in a.arm]
    tab = diracc_table(a.windows, arms, a.n_boot)
    C.dump_json(tab, a.out)
    open(a.out.rsplit(".", 1)[0] + ".md", "w", encoding="utf-8", newline="\n").write(markdown(tab))
    print(markdown(tab))


if __name__ == "__main__":
    main()
