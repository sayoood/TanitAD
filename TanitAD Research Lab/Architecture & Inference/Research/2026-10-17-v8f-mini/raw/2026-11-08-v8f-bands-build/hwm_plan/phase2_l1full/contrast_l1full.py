"""Phase 2 contrast reader (AMEND_HWMPLAN_2). Imports the code of record unchanged (run_hwm.load_ws, hwm_data.lead_for, hwm_eval.band_metrics / contrast / tactical_blocks / FAMILIES / BANDS) and
reads banked plans: per band and family, paired episode-cluster bootstrap A - B, A - HOLD (and B - HOLD), and the inference-seed replicate of A (and of B). B = 2000, seed 0 (inside E.contrast).
Tier T0 (oracle goal): a world-model diagnostic, never driving performance. STRATEGIC: PAUSED (PI item 25).
usage: python contrast_l1full.py --windows DIR --a PLANS_A_s0.npz --b PLANS_B_s0.npz [--rep-a PLANS_A_s1.npz] [--rep-b PLANS_B_s1.npz] --out OUT.json [--check RESULT_HWMPLAN.json --check-a hwm_oracle --check-b flat_oracle]
--a / --b also accept the word 'hold'."""
import argparse, json, os, sys
sys.dont_write_bytecode = True
HERE = "E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/hwm_plan"
sys.path.insert(0, HERE)
import numpy as np
import hwm_plan as H
import hwm_data as D
import hwm_eval as E
import run_hwm as R

p = argparse.ArgumentParser()
p.add_argument("--windows", required=True); p.add_argument("--a", required=True); p.add_argument("--b", required=True)
p.add_argument("--rep-a"); p.add_argument("--rep-b"); p.add_argument("--out", required=True)
p.add_argument("--n-boot", type=int, default=2000)
p.add_argument("--check"); p.add_argument("--check-a"); p.add_argument("--check-b")
a = p.parse_args()

ws = R.load_ws(a.windows)
ws.lead = D.lead_for(ws.eid, ws.now)
ws.lead6 = D.lead_for(ws.eid, ws.now, D.LEAD_BLOCK6)


def met(spec):
    plan = H.hold_plan(ws.A[:, 0], H.N_TRAJ) if spec == "hold" else np.load(spec)["plan_raw"]
    return E.band_metrics(plan, ws.A, ws.v0, ws.lead, ws.lead6)[0]


MA, MB, MH = met(a.a), met(a.b), met("hold")
rows = []
for fam, mets in E.FAMILIES.items():
    for b in E.BANDS:
        for m in mets + (E.BESIDE if fam == "LATERAL" else ()):
            for x, y, nx, ny, M1, M2 in (("A", "B", a.a, a.b, MA, MB), ("A", "HOLD", a.a, "hold", MA, MH), ("B", "HOLD", a.b, "hold", MB, MH)):
                rows.append(dict(kind="contrast", family=fam, band=b, metric=m, a=x, b=y, a_src=nx, b_src=ny, tier="T0", evidence="MEASURED", **E.contrast(M1[b][m], M2[b][m], ws.eid, a.n_boot)))
rep = []
for tag, f0, f1 in (("A", a.a, a.rep_a), ("B", a.b, a.rep_b)):
    if f1:
        M0, M1 = met(f0), met(f1)
        for fam, mets in E.FAMILIES.items():
            for b in E.BANDS:
                for m in mets:
                    rep.append(dict(kind="inference_seed_replicate", arm=tag, family=fam, band=b, metric=m, tier="T0", evidence="MEASURED", **E.contrast(M1[b][m], M0[b][m], ws.eid, a.n_boot)))
tac = E.tactical_blocks({"A": MA, "B": MB, "HOLD": MH})
tac_k = {r: {b: {k: tac[r][b][k]["kappa"] for k in ("lateral", "longitudinal", "maneuver_5way_collapsed")} for b in E.BANDS} for r in tac}
out = dict(rows=rows, replicate=rep, tactical_kappa=tac_k, tactical_note=E.TACTICAL_SCOPE, strategic=E.STRATEGIC, n_windows=int(ws.n), n_episodes=int(len(set(ws.eid))), a=a.a, b=a.b, rep_a=a.rep_a, rep_b=a.rep_b,
           note="positive delta = worse for A (lower is better). Estimator: paired episode-cluster bootstrap (answers 'another draw of episodes'); the replicate rows are the inference-seed floor. T0 diagnostic, never driving performance.")
json.dump(out, open(a.out, "w"), indent=1, default=float)

if a.check:
    ref = json.load(open(a.check))
    bad, n = 0, 0
    for r in rows:
        if r["a"] != "A" or r["b"] != "B":
            continue
        m = [q for q in ref["rows"] if q["kind"] == "contrast" and q["family"] == r["family"] and q["band"] == r["band"] and q["metric"] == r["metric"] and q["a"] == a.check_a and q["b"] == a.check_b]
        if len(m) != 1:
            print("NO REF ROW", r["family"], r["band"], r["metric"]); bad += 1; continue
        q = m[0]; n += 1
        for k in ("delta", "lo", "hi"):
            x, y = r[k], q[k]
            if not (x == y or (x != x and y != y)):
                bad += 1; print("DIFF", r["family"], r["band"], r["metric"], k, x, y)
    if a.rep_a:
        for r in rep:
            if r["arm"] != "A":
                continue
            m = [q for q in ref["replicate"] if q["arm"] == a.check_a and q["family"] == r["family"] and q["band"] == r["band"] and q["metric"] == r["metric"]]
            q = m[0]; n += 1
            for k in ("delta", "lo", "hi"):
                x, y = r[k], q[k]
                if not (x == y or (x != x and y != y)):
                    bad += 1; print("REPDIFF", r["family"], r["band"], r["metric"], k, x, y)
    print(f"CHECK rows compared {n}, mismatches {bad}", "REPRODUCED EXACTLY" if bad == 0 and n > 0 else "NOT REPRODUCED")
else:
    for r in rows + rep:
        if r["band"] in ("2-6s", "0-2s") and r["metric"] in ("acc_mae", "yaw_mae"):
            print(r["kind"], r["band"], r["metric"], r.get("a"), r.get("b"), r.get("arm"), round(r["delta"], 4), round(r["lo"], 4), round(r["hi"], 4), r["label"])
