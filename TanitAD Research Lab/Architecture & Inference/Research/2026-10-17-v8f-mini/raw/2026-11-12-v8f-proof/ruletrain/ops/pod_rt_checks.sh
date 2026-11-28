cd /workspace/v8proof || exit 1
# pod_rt_checks.sh : card PREREG_V8F_PROOF_ADD32 (RULETRAIN), checks read ON THE POD from finished files. CPU only, starts nothing else.
#   chain log: gate and token of each of the six arms, no STOP marker
#   t4: selfcheck of each arm (declared flag equals built value, relabel count of the live step, init checkpoint applied) and the relabel count in the train log
#   first tick: share of windows whose plan the label-fed rule changes at the first tick, from the rule logs of GFTRL and GRTRL; lift 0 in those logs
echo "== chain"; grep -a -c "ARM DONE" chains/ruletrain_a/chain.log; grep -a "token\|STOP\|ZZ CHAIN" chains/ruletrain_a/chain.log | cut -c1-170 | tail -n 9; ls chains/ruletrain_a | tr "\n" " "; echo
echo "== files of one arm"; ls runs/ruletrain_a/GRT_s0 | tr "\n" " "; echo; ls runs/ruletrain_a/GRT_s0/out 2>/dev/null | tr "\n" " "; echo
CUDA_VISIBLE_DEVICES=-1 nice -n 19 /workspace/venv/bin/python - <<'PY'
import json, os, glob
import numpy as np
R = "runs/ruletrain_a"
for a in ("GFT_s0", "GRT_s0", "GFT_s1", "GRT_s1", "GFTb_s0", "GFTb_s1"):
    sc = None
    for p in (os.path.join(R, a, "out", "selfcheck.json"), os.path.join(R, a, "selfcheck.json")):
        if os.path.exists(p):
            sc = json.load(open(p)); break
    if sc is None:
        hits = glob.glob(os.path.join(R, a, "**", "selfcheck*.json"), recursive=True)
        sc = json.load(open(hits[0])) if hits else None
    if sc is None:
        print("T4 %-8s NO SELFCHECK" % a); continue
    lv, info = sc.get("levers", {}), sc.get("info", {})
    print("T4 %-8s dvb=%s live=%s steps_total=%s rule_relabel=%s n=%s changed=%s init_applied=%s seed=%s" % (
        a, sc.get("declared_vs_built"), (sc.get("live") or {}).get("pass"), sc.get("steps_total"), lv.get("rule_relabel"), info.get("rule_relabel_n"),
        info.get("rule_relabel_changed"), (lv.get("init_ckpt") or {}).get("applied"), sc.get("seed", lv.get("seed"))))
    logs = glob.glob(os.path.join(R, a, "**", "*.jsonl"), recursive=True)
    for lp in logs[:3]:
        rows = []
        for ln in open(lp, errors="replace"):
            try:
                rows.append(json.loads(ln))
            except Exception:
                pass
        if not rows:
            continue
        keys = sorted(k for k in rows[-1] if "rule" in k or "relabel" in k)
        last = rows[-1]
        tot = {k: float(np.nansum([r.get(k, 0) or 0 for r in rows if isinstance(r.get(k, 0), (int, float))])) for k in keys}
        print("   log %s rows=%d last step=%s rule keys=%s sums=%s loss keys(last)=%s" % (os.path.basename(lp), len(rows), last.get("step"), keys, {k: round(v, 2) for k, v in tot.items()},
              [k for k in last if "traj" in k or k in ("loss", "ade", "l_traj")][:6]))
root = "runs/distrule/t1"
tot_lift = 0
for a in ("GFTRL_s0", "GRTRL_s0", "GFTRL_s1", "GRTRL_s1", "GRL_s0", "GRL_s1", "LSRL_s0", "LSRL_s1"):
    p = os.path.join(root, a, "ticks", "rule_log_%s.npz" % a)
    if not os.path.exists(p):
        print("TICK %-9s NO RULE LOG" % a); continue
    z = np.load(p, allow_pickle=True)
    m0 = z["tick"] == 0
    pres = z["present"].astype(bool)
    ch = z["changed"].astype(bool)
    lift = int(z["lift"].sum())
    if a.startswith("G") and "T" in a:
        tot_lift += lift
    n0 = int(m0.sum()); nl0 = int((m0 & pres).sum())
    print("TICK %-9s ticks=%d lift_sum=%d | first tick: windows=%d with lead=%d changed of all=%.4f changed of lead=%.4f | all ticks: present=%.4f changed=%.4f changed of lead=%.4f sat_exec_sum=%d" % (
        a, z["ci"].size, lift, n0, nl0, ch[m0].mean() if n0 else float("nan"), ch[m0 & pres].mean() if nl0 else float("nan"),
        pres.mean(), ch.mean(), ch[pres].mean() if pres.any() else float("nan"), int(z["sat_exec"].sum())))
print("TICK lift over the four logs of the card: %d (%s)" % (tot_lift, "PASS" if tot_lift == 0 else "FAIL"))
PY
