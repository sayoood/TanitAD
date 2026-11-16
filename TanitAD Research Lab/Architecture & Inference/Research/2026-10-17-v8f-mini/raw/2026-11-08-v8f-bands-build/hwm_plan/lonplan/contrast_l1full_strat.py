"""contrast_l1full_strat.py: phase2_l1full/contrast_l1full.py with `hwm_eval` resolved to the strategic-audit reader (raw/2026-11-16-v8f-strategic-audit/reader/src/, landed 7a571fc7), so the STRATEGIC family is COMPUTED, not PAUSED.
The one script that imports the reader for both ACTRSP section 4.2 and LON-PLAN (MM freeze line, AMEND_HWMPLAN_3 Log 2026-11-16 ~14:35Z).
Changes against the code of record, all stated:
  1. module resolution: --eval-src (default LONPLAN_READER_SRC, else ./reader_src) is put FIRST on sys.path, so `hwm_eval`, `hwm_plan`, `hwm_data`, `strategic_cells`, `strategic_text` come from the reader; run_hwm (load_ws only) comes from the code of record and sees the same cached modules.
  2. the navigation-text row is attached to the windows (hwm_eval.attach_text) and passed to band_metrics; STRATEGIC rows are contrasted with the reader's direction (S_ADH higher is better, S_FALSE neutral).
  3. `strategic` in the output is the reader's strategic_block (state, tier stamp, n per arm, logged-path rate, not-computed entries with reason). An eval module without strategic_block (the code of record) writes its STRATEGIC constant instead, which reads PAUSED: that is the C-STRAT regression arm.
  4. HERE is not hard-coded. Everything else (row order, estimator, B 2000 seed 0 inside E.contrast, replicate rows, tactical kappa, --check) is the original's code.
Tier T0 (oracle goal): a world-model diagnostic, never driving performance. Estimator: paired episode-cluster bootstrap (answers 'another draw of episodes', nothing else).
usage: python contrast_l1full_strat.py --windows DIR --a PLANS_A_s0.npz --b PLANS_B_s0.npz [--rep-a PLANS_A_s1.npz] [--rep-b PLANS_B_s1.npz] --out OUT.json [--n-boot N] [--eval-src DIR] [--check RESULT.json --check-a hwm_oracle --check-b flat_oracle]
--a / --b also accept the word 'hold'."""
import argparse, json, os, sys
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = os.environ.get("LONPLAN_CODE_ROOT", "E:/Projects/TanitAD-artifacts/dagland/motion_c/theory")
RECORD = os.path.join(CODE_ROOT, "hwm_plan")
READER = os.environ.get("LONPLAN_READER_SRC", os.path.join(HERE, "reader_src"))

p = argparse.ArgumentParser()
p.add_argument("--windows", required=True); p.add_argument("--a", required=True); p.add_argument("--b", required=True)
p.add_argument("--rep-a"); p.add_argument("--rep-b"); p.add_argument("--out", required=True)
p.add_argument("--n-boot", type=int, default=2000)
p.add_argument("--check"); p.add_argument("--check-a"); p.add_argument("--check-b")
p.add_argument("--eval-src", default=READER, help="folder whose hwm_eval / hwm_plan / hwm_data are imported first (default: the strategic-audit reader; the code-of-record folder is the C-STRAT regression arm)")
a = p.parse_args()

for d in (RECORD, os.path.join(CODE_ROOT, "wmact"), os.path.join(CODE_ROOT, "plantest2"), os.path.join(CODE_ROOT, "plantest"), a.eval_src):    # last inserted = first on the path
    sys.path.insert(0, d)
import numpy as np
import hwm_plan as H
import hwm_data as D
import hwm_eval as E
import run_hwm as R

assert os.path.dirname(os.path.abspath(E.__file__)) == os.path.abspath(a.eval_src), (E.__file__, a.eval_src)

ws = R.load_ws(a.windows)
ws.lead = D.lead_for(ws.eid, ws.now)
ws.lead6 = D.lead_for(ws.eid, ws.now, D.LEAD_BLOCK6)
HAS_STRAT = hasattr(E, "strategic_block") and "STRATEGIC" in E.FAMILIES
text = E.attach_text(ws) if HAS_STRAT else None                    # the reader's navigation-text row of every window's NOW (sha-checked table); without the reader there is none
ws.text = text


def met(spec):
    plan = H.hold_plan(ws.A[:, 0], H.N_TRAJ) if spec == "hold" else np.load(spec)["plan_raw"]
    return E.band_metrics(plan, ws.A, ws.v0, ws.lead, ws.lead6, text)[0] if HAS_STRAT else E.band_metrics(plan, ws.A, ws.v0, ws.lead, ws.lead6)[0]


def con(x, y, fam, m):
    if fam == "STRATEGIC":
        return E.contrast(x, y, ws.eid, a.n_boot, E.DIRECTION.get(m, "lower"))
    return E.contrast(x, y, ws.eid, a.n_boot)


MA, MB, MH = met(a.a), met(a.b), met("hold")
rows = []
for fam, mets in E.FAMILIES.items():
    for b in E.BANDS:
        for m in mets + (E.BESIDE if fam == "LATERAL" else ()):
            for x, y, nx, ny, M1, M2 in (("A", "B", a.a, a.b, MA, MB), ("A", "HOLD", a.a, "hold", MA, MH), ("B", "HOLD", a.b, "hold", MB, MH)):
                rows.append(dict(kind="contrast", family=fam, band=b, metric=m, a=x, b=y, a_src=nx, b_src=ny, tier="T0", evidence="MEASURED", **con(M1[b][m], M2[b][m], fam, m)))
rep = []
for tag, f0, f1 in (("A", a.a, a.rep_a), ("B", a.b, a.rep_b)):
    if f1:
        M0, M1 = met(f0), met(f1)
        for fam, mets in E.FAMILIES.items():
            for b in E.BANDS:
                for m in mets:
                    rep.append(dict(kind="inference_seed_replicate", arm=tag, family=fam, band=b, metric=m, tier="T0", evidence="MEASURED", **con(M1[b][m], M0[b][m], fam, m)))
tac = E.tactical_blocks({"A": MA, "B": MB, "HOLD": MH})
tac_k = {r: {b: {k: tac[r][b][k]["kappa"] for k in ("lateral", "longitudinal", "maneuver_5way_collapsed")} for b in E.BANDS} for r in tac}
strategic = E.strategic_block({"A": MA, "B": MB, "HOLD": MH}, ws) if HAS_STRAT else E.STRATEGIC
out = dict(rows=rows, replicate=rep, tactical_kappa=tac_k, tactical_note=E.TACTICAL_SCOPE, strategic=strategic, n_windows=int(ws.n), n_episodes=int(len(set(ws.eid))), a=a.a, b=a.b, rep_a=a.rep_a, rep_b=a.rep_b,
           eval_module=os.path.abspath(E.__file__), strategic_computed=bool(HAS_STRAT),
           note="positive delta = worse for A (lower is better) except S_ADH (higher is better) and S_FALSE (neutral, plan minus logged). Estimator: paired episode-cluster bootstrap (answers 'another draw of episodes'); the replicate rows are the inference-seed floor. T0 diagnostic, never driving performance.")
json.dump(out, open(a.out, "w"), indent=1, default=float)

if a.check:
    ref = json.load(open(a.check))
    bad, n = 0, 0
    for r in rows:
        if r["a"] != "A" or r["b"] != "B" or r["family"] == "STRATEGIC":
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
            if r["arm"] != "A" or r["family"] == "STRATEGIC":
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
    if HAS_STRAT:
        for r in rows:
            if r["family"] == "STRATEGIC" and r["band"] == "full6s" and r["a"] == "A" and r["b"] == "B":
                print("STRATEGIC", r["metric"], round(r["delta"], 4) if r["delta"] == r["delta"] else "n/a", r["label"], "n_windows", r["n_windows"], "n_episodes", r["n_episodes"], r.get("reason", ""))
    else:
        print("STRATEGIC", strategic)
