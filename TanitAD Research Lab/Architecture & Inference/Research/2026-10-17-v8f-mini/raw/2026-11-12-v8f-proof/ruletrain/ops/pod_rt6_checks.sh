cd /workspace/v8proof || exit 1
# pod_rt6_checks.sh : note PREREG_V8F_PROOF_ADD32_N1 stage 1 (6,000-step fine-tunes), checks read ON THE POD from finished files. CPU only, starts nothing else.
#   chain logs: every arm done, no STOP marker; t4: selfcheck of each arm and the relabel count in the train log; first tick from the rule logs; lift 0.
for c in a b; do echo "== chain ruledose_$c"; grep -a -c "ARM DONE" chains/ruledose_$c/chain.log; grep -a "STOP\|ZZ CHAIN" chains/ruledose_$c/chain.log | cut -c1-150 | tail -n 3; ls chains/ruledose_$c | tr "\n" " "; echo; done
tail -n 2 distrule/rtrule6_queue.log | cut -c1-160
CUDA_VISIBLE_DEVICES=-1 nice -n 19 /workspace/venv/bin/python - <<'PY'
import json, os
import numpy as np
for a, q in (("GFT6_s0", "ruledose_a"), ("GRT6_s0", "ruledose_a"), ("GFT6b_s0", "ruledose_a"), ("GFT6_s1", "ruledose_b"), ("GRT6_s1", "ruledose_b"), ("GFT6b_s1", "ruledose_b")):
    d = os.path.join("runs", q, a)
    try:
        sc = json.load(open(os.path.join(d, "selfcheck.json")))
    except Exception as e:
        print("T4 %-9s NO SELFCHECK %s" % (a, e)); continue
    lv, info = sc.get("levers", {}), sc.get("info", {})
    su = {}
    try:
        su = json.load(open(os.path.join(d, "summary.json")))
    except Exception:
        pass
    print("T4 %-9s dvb=%s live=%s steps_total=%s rule_relabel=%s n=%s changed=%s init_applied=%s | summary done=%s steps=%s" % (
        a, sc.get("declared_vs_built"), (sc.get("live") or {}).get("pass"), sc.get("steps_total"), lv.get("rule_relabel"), info.get("rule_relabel_n"),
        info.get("rule_relabel_changed"), (lv.get("init_ckpt") or {}).get("applied"), su.get("done"), su.get("steps")))
    rows = []
    for ln in open(os.path.join(d, "train_log.jsonl"), errors="replace"):
        try:
            rows.append(json.loads(ln))
        except Exception:
            pass
    keys = sorted(k for k in rows[-1] if "rule" in k) if rows else []
    tot = {k: float(np.nansum([r.get(k, 0) or 0 for r in rows])) for k in keys}
    print("   log rows=%d last step=%s sums=%s ade first/last=%s/%s" % (len(rows), rows[-1].get("step") if rows else None, {k: round(v) for k, v in tot.items()},
          rows[0].get("ade") if rows else None, rows[-1].get("ade") if rows else None))
root = "runs/distrule/t1"; tot_lift = 0
for a in ("GFT6RL_s0", "GRT6RL_s0", "GFT6RL_s1", "GRT6RL_s1"):
    p = os.path.join(root, a, "ticks", "rule_log_%s.npz" % a)
    if not os.path.exists(p):
        print("TICK %-10s NO RULE LOG" % a); continue
    z = np.load(p, allow_pickle=True)
    m0 = z["tick"] == 0; pres = z["present"].astype(bool); ch = z["changed"].astype(bool); lift = int(z["lift"].sum()); tot_lift += lift
    print("TICK %-10s ticks=%d lift_sum=%d | first tick: windows=%d with lead=%d changed of all=%.4f changed of lead=%.4f | all ticks: changed=%.4f changed of lead=%.4f sat_exec_sum=%d" % (
        a, z["ci"].size, lift, int(m0.sum()), int((m0 & pres).sum()), ch[m0].mean(), ch[m0 & pres].mean(), ch.mean(), ch[pres].mean(), int(z["sat_exec"].sum())))
print("TICK lift over the four logs: %d (%s)" % (tot_lift, "PASS" if tot_lift == 0 else "FAIL"))
PY
