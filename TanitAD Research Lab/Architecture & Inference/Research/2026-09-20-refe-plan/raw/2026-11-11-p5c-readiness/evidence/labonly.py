"""P5-C readiness probe (scratch): Tc20s ONLY, timed, compared with the banked P5-S npz. CPU only."""
import sys, os, time, json, glob
import numpy as np
sys.argv = [sys.argv[0]]
import p5copy as P
import navsim_dac as ND
import navtrain_scenarios as NS
PK = P.PKG
Z = np.load(PK + "/raw/2026-10-11-p5s-smoothed/p5s_cubic.npz", allow_pickle=True)
bank_lab = {(str(l), str(t)): Z["labels20_t20_tc20_tw"][i] for i, (l, t) in enumerate(zip(Z["logs"], Z["tokens"]))}
held = {}
for line in open(P.HELD, encoding="utf-8"):
    r = json.loads(line)
    if int(r.get("rank", 0)) == 0 and int(r.get("step", 0)) == 0 and "teacher" in r:
        held[(r["log"], r["token"])] = r
bank = {}
for fp in glob.glob(f"{P.BANK}/**/targets_rank0.jsonl", recursive=True):
    for line in open(fp, encoding="utf-8"):
        r = json.loads(line)
        k = (r.get("log_name"), r.get("token"))
        if k in held and int(r.get("sim_hz", 0)) == 10 and int(r.get("step", 0)) == 0:
            bank[k] = r["traj"]
keys = sorted(bank)[:int(os.environ.get("N", "20"))]
by = {}
for lg, tok in keys: by.setdefault(lg, []).append(tok)
dbs = NS.index_dbs(db_root=os.path.dirname(P.TRAINVAL))
t_all = time.time(); t_lab = 0.0; n = 0; maxdiff = 0.0; ident = 0
for lg, toks in sorted(by.items()):
    t0 = time.time()
    scs = list(NS.build_scenarios_for_log(dbs[lg], toks))
    t_scn = time.time() - t0
    for sc in scs:
        tok = sc._initial_lidar_token
        ego = sc.get_ego_state_at_iteration(0)
        t20 = np.asarray(bank[(lg, tok)], np.float64)
        T = ND.to_navsim(t20)
        ctl = {"knot_exact": True, "knot_checks": 0}
        t1 = time.time()
        tc20, _, _, _ = P.iterate20(ND, t20, T, ego, ctl, None, "cubic")
        t_lab += time.time() - t1
        ref = bank_lab[(lg, tok)][1]
        d = float(np.abs(tc20 - ref).max()); maxdiff = max(maxdiff, d); ident += int(d == 0.0); n += 1
    print(lg[-22:], "scenario build", round(t_scn, 2), "s for", len(scs), flush=True)
print(f"ZZLAB n={n} iterate20_total_s={t_lab:.2f} per_row_s={t_lab/n:.3f} wall_s={time.time()-t_all:.1f} max_abs_diff_vs_banked={maxdiff:.3e} bit_identical={ident}/{n}")
