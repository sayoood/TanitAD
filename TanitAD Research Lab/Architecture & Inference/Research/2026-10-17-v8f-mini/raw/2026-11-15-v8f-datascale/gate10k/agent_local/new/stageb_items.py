"""Stage-B gate items and the B_rep stage-A selfcheck record (pod, CPU). usage: stageb_items.py
Reads what gpu_gate.sh left (gate/sb/*, gate/bank_B_pilot, gate/probe2/*) and writes /root/dscale10k/gate/stageb.json:
  B_s0.B (REAL stage-B argv, real complete 4-shard bank built from the pilot checkpoint, real --init-ckpt = that checkpoint): C-BANK and G-DAGGER on the trainer's own --selfcheck-only record
  regression arms (each must read RED): the same selfcheck with the swap's apply mutated (the trainer must have refused: no passing record); the pilot's PARTIAL 1-clip-cap bank; a selfcheck
    whose dagger controls are tampered; a bank whose shard count is 1
  U_s0.B smoke (--max-clips 200 bank + selfcheck on the union argv): C-BANK (n_shards 4) and G-DAGGER
  B_rep.A: the trainer's declared_vs_built / live record of the stage-A argv (the pilot ran B_s0 and U_s0 only)
"""
import copy
import json
import os
import sys

D = "/root/dscale10k"
G = D + "/gate"
TREE = D + "/tree"
for p in (TREE + "/stack", TREE + "/stack/scripts", TREE, D + "/study"):
    if p not in sys.path:
        sys.path.insert(0, p)
import dscale10k_check as C  # noqa: E402

L = lambda p: json.load(open(p))
out = {"item": "stageb", "record": {}, "regression": [], "problems": []}


def red(name, problems):
    out["regression"].append({"arm": name, "red": bool(problems), "problems": list(problems)[:2]})


# ---- B_s0.B, real
argv = L(f"{G}/sb/B_s0.B.argv.json")["argv"]
dry = L(f"{D}/dry/B_s0.A.dry.json")
res = L(f"{G}/sb/B_s0.B.result.json")
bank_p = C.bank_problems(f"{G}/bank_B_pilot", f"{D}/pilot/pB/ckpt_last.pt", dry)
if res.get("selfcheck") and os.path.isfile(res["selfcheck"]):
    sc = L(res["selfcheck"])
    dag_p = C.dagger_problems(sc, argv)
else:
    sc = None
    dag_p = [f"no selfcheck.json from the real stage-B argv: {res.get('message')}"]
bj = L(f"{G}/bank_B_pilot/bank.json")
out["record"]["B_s0.B"] = {"C-BANK": {"problems": bank_p, "n_shards": bj.get("n_shards"), "partial": bj.get("partial"), "coverage": bj.get("coverage"), "n_records": bj.get("n_records")},
                           "G-DAGGER": {"problems": dag_p, "result": res,
                                        "dagger_probe": {k: (sc or {}).get("dagger_probe", {}).get(k) for k in ("pass", "n_rows", "n_eligible", "n_selected_at_p1", "loss_diff", "grad_diff_all", "control_repeat_loss_diff",
                                                                                                                 "control_repeat_grad_diff", "control_p0_same_object", "control_p0_loss_diff",
                                                                                                                 "control_p0_grad_diff", "control_prior_grad_diff", "control_wm_grad_diff", "rows_changed")},
                                        "acc_probe": {k: (sc or {}).get("acc_probe", {}).get(k) for k in ("pass", "w_acc_declared", "head_o_grad_diff", "declared_zero_total_reproduces_exactly")}}}
out["problems"] += [f"B_s0.B C-BANK: {x}" for x in bank_p] + [f"B_s0.B G-DAGGER: {x}" for x in dag_p]
# ---- regression arms
mut = L(f"{G}/sb/B_s0.B.mut.json")
mut_sc = L(mut["selfcheck"]) if mut.get("selfcheck") and os.path.isfile(mut["selfcheck"]) else None
mp = []
if mut.get("refused"):
    mp.append("the trainer refused the mutated swap: " + str(mut.get("message"))[:160])
if mut_sc is None:
    mp.append("no passing selfcheck.json under the mutation")
else:
    mp += C.dagger_problems(mut_sc, argv)
red("dagger_apply_mutated (the swapped tensors discarded)", mp)
out["record"]["mutation"] = {k: mut.get(k) for k in ("rc", "refused", "message", "selfcheck")}
red("partial_bank", C.bank_problems(f"{D}/pilot/bank200_4sh", f"{D}/pilot/pB/ckpt_last.pt", dry))
if sc:
    t = copy.deepcopy(sc)
    t["dagger_probe"]["control_p0_grad_diff"] = 1e-9
    red("p0_control_not_exactly_zero", C.dagger_problems(t, argv))
    t = copy.deepcopy(sc)
    t["dagger_probe"]["grad_diff_all"] = 0.0
    red("swap_does_not_reach_the_loss", C.dagger_problems(t, argv))
    t = copy.deepcopy(sc)
    t["info"]["dagger"]["coverage"]["fraction"] = 0.3
    red("coverage_below_0.5", C.dagger_problems(t, argv))
# ---- U smoke
ru = L(f"{G}/sb/U_s0.B.result.json")
ua = L(f"{G}/sb/U_s0.B.argv.json")["argv"]
ub = L(f"{G}/sb/U_smoke_bank/bank.json")
tr = (ub.get("common") or {}).get("train") or {}
udry = {"n_clips": 200, "n_windows": tr.get("n_windows"), "train_sha256": tr.get("train_sha256")}
ubp = C.bank_problems(f"{G}/sb/U_smoke_bank", f"{D}/pilot/pU/ckpt_last.pt", udry)
if ru.get("selfcheck") and os.path.isfile(ru["selfcheck"]):
    usc = L(ru["selfcheck"])
    udp = C.dagger_problems(usc, ua)
else:
    usc, udp = None, [f"no selfcheck.json from the U smoke stage-B argv: {ru.get('message')}"]
out["record"]["U_s0.B_smoke"] = {"scope": "--max-clips 200 bank (4 shards, parallel) and selfcheck on the UNION stage-B argv; a full union bank is built in the chain (about 2.3 h)", "C-BANK": {"problems": ubp, "n_shards": ub.get("n_shards"),
                                 "coverage": ub.get("coverage")}, "G-DAGGER": {"problems": udp, "dagger_probe_pass": ((usc or {}).get("dagger_probe") or {}).get("pass"),
                                 "grad_diff_all": ((usc or {}).get("dagger_probe") or {}).get("grad_diff_all"), "loss_diff": ((usc or {}).get("dagger_probe") or {}).get("loss_diff")}}
out["problems"] += [f"U_s0.B smoke C-BANK: {x}" for x in ubp] + [f"U_s0.B smoke G-DAGGER: {x}" for x in udp]
# ---- B_rep.A selfcheck record
sp = f"{G}/selfcheck_B_rep.A/selfcheck.json"
if os.path.isfile(sp):
    s = L(sp)
    lv = s.get("live") or {}
    out["record"]["B_rep.A_selfcheck"] = {"declared_vs_built": s.get("declared_vs_built"), "live_pass": lv.get("pass"), "n_trainable_tensors": lv.get("n_trainable")}
    if s.get("declared_vs_built") != "PASS" or not lv.get("pass"):
        out["problems"].append("B_rep.A selfcheck does not read PASS")
else:
    out["problems"].append(f"{sp} is missing")
out["pass"] = not out["problems"] and all(r["red"] for r in out["regression"])
json.dump(out, open(f"{G}/stageb.json", "w"), indent=1, default=str)
print("stageb", "PASS" if out["pass"] else "FAIL", out["problems"][:4], [r["red"] for r in out["regression"]])
