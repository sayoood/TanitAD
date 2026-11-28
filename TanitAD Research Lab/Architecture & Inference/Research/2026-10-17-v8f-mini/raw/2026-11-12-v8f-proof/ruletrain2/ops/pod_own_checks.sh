cd /workspace/v8proof || exit 1
# pod_own_checks.sh : card PREREG_V8F_PROOF_ADD34 (arms GOW, GOW4; chain ruleown_a) and stage 2 of note ADD32_N1 (arms GRW6; chain ruledose2_a).
# Checks read ON THE POD from finished files. CPU only, starts nothing else.
#   chain logs: every arm done, no STOP marker; o4 / w4: selfcheck of each arm and the keys of the train log; first tick from the rule logs; lift 0.
for c in ruleown_a ruledose2_a; do echo "== chain $c"; grep -a -c "ARM DONE" chains/$c/chain.log; grep -a "STOP\|ZZ CHAIN" chains/$c/chain.log | cut -c1-150 | tail -n 3; ls chains/$c | tr "\n" " "; echo; done
tail -n 1 distrule/rtrule_OWN_queue.log | cut -c1-160; tail -n 1 distrule/rtrule_DOSE2_queue.log | cut -c1-160
CUDA_VISIBLE_DEVICES=-1 nice -n 19 /workspace/venv/bin/python - <<'PY'
import json, os
import numpy as np
for a, q in (("GOW_s0", "ruleown_a"), ("GOW_s1", "ruleown_a"), ("GOW4_s0", "ruleown_a"), ("GOW4_s1", "ruleown_a"), ("GRW6_s0", "ruledose2_a"), ("GRW6_s1", "ruledose2_a")):
    d = os.path.join("runs", q, a)
    try:
        sc = json.load(open(os.path.join(d, "selfcheck.json")))
    except Exception as e:
        print("O4 %-8s NO SELFCHECK %s" % (a, e)); continue
    lv, info = sc.get("levers", {}), sc.get("info", {})
    su = {}
    try:
        su = json.load(open(os.path.join(d, "summary.json")))
    except Exception:
        pass
    pr = sc.get("rule_own_probe") or {}
    print("O4 %-8s dvb=%s live=%s steps_total=%s w_rule_own=%s rule_relabel=%s weight=%s init_applied=%s | summary done=%s steps=%s | probe term=%s n_lead=%s n_active=%s pass=%s" % (
        a, sc.get("declared_vs_built"), (sc.get("live") or {}).get("pass"), sc.get("steps_total"), lv.get("w_rule_own"), lv.get("rule_relabel"), lv.get("rule_relabel_weight"),
        (lv.get("init_ckpt") or {}).get("applied"), su.get("done"), su.get("steps"), pr.get("term"), pr.get("n_lead"), pr.get("n_active"), pr.get("pass")))
    rows = []
    for ln in open(os.path.join(d, "train_log.jsonl"), errors="replace"):
        try:
            rows.append(json.loads(ln))
        except Exception:
            pass
    if not rows:
        print("   NO LOG ROWS"); continue
    keys = sorted(k for k in rows[-1] if "rule" in k or "own" in k)
    def col(k):
        return np.array([float(r.get(k, np.nan)) if isinstance(r.get(k), (int, float)) else np.nan for r in rows])
    out = {}
    for k in keys:
        v = col(k)
        out[k] = {"sum": round(float(np.nansum(v)), 3), "first5_mean": round(float(np.nanmean(v[:5])), 4), "last5_mean": round(float(np.nanmean(v[-5:])), 4)}
    print("   log rows=%d last step=%s keys=%s" % (len(rows), rows[-1].get("step"), json.dumps(out)[:900]))
root = "runs/distrule/t1"; tot_lift = 0
for a in ("GOWRL_s0", "GOWRL_s1", "GOW4RL_s0", "GOW4RL_s1", "GRW6RL_s0", "GRW6RL_s1", "GFTRL_s0", "GFTRL_s1", "GFT6RL_s0", "GFT6RL_s1"):
    p = os.path.join(root, a, "ticks", "rule_log_%s.npz" % a)
    if not os.path.exists(p):
        print("TICK %-10s NO RULE LOG" % a); continue
    z = np.load(p, allow_pickle=True)
    m0 = z["tick"] == 0; pres = z["present"].astype(bool); ch = z["changed"].astype(bool); lift = int(z["lift"].sum())
    if a[:3] in ("GOW", "GRW"):
        tot_lift += lift
    print("TICK %-10s ticks=%d lift_sum=%d | first tick: with lead=%d changed of lead=%.4f | all ticks: changed of lead=%.4f sat_exec_sum=%d" % (
        a, z["ci"].size, lift, int((m0 & pres).sum()), ch[m0 & pres].mean(), ch[pres].mean(), int(z["sat_exec"].sum())))
print("TICK lift over the logs of the new arms: %d (%s)" % (tot_lift, "PASS" if tot_lift == 0 else "FAIL"))
PY
