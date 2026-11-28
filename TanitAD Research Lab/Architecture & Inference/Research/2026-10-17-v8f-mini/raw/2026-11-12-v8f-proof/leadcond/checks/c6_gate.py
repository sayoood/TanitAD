"""c6 gate (LEADCOND), trainer --selfcheck-only on the tiny synthetic world, LS-shape fine-tune argv (the arm shapes LFT / LC / LO / LCO of ARMS_LEADCOND.txt with the card steps, lr, warmup):
 off (LFT)   tree_patch selfcheck has NO new key anywhere (flattened key set equal to tree_base's on the same argv; the text has no `lead_cond`)
 on  (LC)    carries levers.lead_cond=on, levers.lead_cond_zero_init, the lead_cond_probe (grad on the weight > 0, n_rows, mean / std of p and g, control exactly 0, detach 0.0), init_ckpt read back: fresh keys exactly
             lead_cond.weight / lead_cond.bias, 0 mismatching tensors; declared_vs_built PASS
 LO / LCO    the same with --w-rule-own 4.0 --rule-own-form l1 (LCO has both levers)
 gate_extra  (the pod chain gate) accepts the right expectation (rc 0) and refuses a wrong one (rc != 0): levers.lead_cond=on on the off selfcheck, levers.lead_cond=off on the on selfcheck
 refusals    --lead-cond on without --lead-sup on exits rc 1; an unknown value exits rc 2
Usage: c6_gate.py"""
import json
import os
import subprocess

import lc_common as C

LC = C.LC
CK = f"{LC}/checks"
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
GATE = "F:/TanitAD-scratch/v8proof/pod/chain2/gate_extra.py"
w = lambda *p: "\\".join(p).replace("/", "\\")
BASE = C.BASE_CKPT
G = ["--arm", "M2b", "--grid", "pool4x10", "--kin-mode", "yaw", "--wm-state", "grid", "--wm-cell-dim", "16", "--exclude", "1:3"]
TINY = ["--synthetic", "--batch", "8", "--d", "32", "--n-q", "2", "--threads", "1"]
LSF = ["--lead-sup", "on", "--w-lead-sup", "1.0"]
FT = ["--init-ckpt", BASE, "--steps", "6000", "--lr", "1e-4", "--warmup", "100"]
os.makedirs(f"{CK}/selfchecks", exist_ok=True)


def env_for(tree):
    root = f"{LC}/work/tree_{tree}"
    return dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONIOENCODING="utf-8", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1",
                TMP=w(LC, "work", "tmp"), TEMP=w(LC, "work", "tmp"), PYTHONPATH=";".join([w(root, "stack"), w(root, "taniteval"), w(root, "stack", "scripts")]))


def run(tree, tag, extra, out):
    r = subprocess.run([PY, f"{LC}/work/tree_{tree}/stack/scripts/v8f_mini_train.py", *TINY, *G, *extra, "--selfcheck-only", "--out", out], env=env_for(tree), capture_output=True, text=True, encoding="utf-8")
    with open(f"{CK}/selfchecks/c6_{tag}_{tree}.log", "w", encoding="utf-8", newline="\n") as fh:
        fh.write((r.stdout or "")[-1500:] + "\nSTDERR\n" + (r.stderr or "")[-2500:] + f"\nRC {r.returncode}\n")
    return r


def gate(sc_path, exp):
    return subprocess.run([PY, GATE, "check", "--selfcheck", sc_path, "--card", "LEADCOND_SELFTEST", "--out", f"{LC}/work/tmp/c6_extra_token.json", "--expect", *exp], capture_output=True, text=True, encoding="utf-8")


def flat(d, pre=""):
    out = set()
    if isinstance(d, dict):
        for k, v in d.items():
            out |= {pre + k} | flat(v, pre + k + ".")
    return out


SH = {"LFT": [], "LC": ["--lead-cond", "on"], "LO": ["--w-rule-own", "4.0", "--rule-own-form", "l1"], "LCO": ["--lead-cond", "on", "--w-rule-own", "4.0", "--rule-own-form", "l1"]}
res = {"shapes": {}}
for tag, extra in SH.items():
    sc = {}
    for tree in ("base", "patch"):
        if tree == "base" and "--lead-cond" in extra:
            continue
        out = f"{LC}/work/tmp/c6_{tag}_{tree}"
        r = run(tree, tag, [*LSF, *FT, *extra], out)
        sc[tree] = (r.returncode, json.load(open(f"{out}/selfcheck.json", encoding="utf-8")) if r.returncode == 0 else {})
    rc, s = sc["patch"]
    with open(f"{CK}/selfchecks/c6_selfcheck_{tag}.json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(s, fh, indent=1)
    lv = s.get("levers", {})
    ic = lv.get("init_ckpt") or {}
    rec = {"rc": rc, "declared_vs_built": s.get("declared_vs_built"), "live_pass": (s.get("live") or {}).get("pass"), "levers_lead_cond": lv.get("lead_cond", "ABSENT"), "levers_lead_cond_zero_init": lv.get("lead_cond_zero_init", "ABSENT"),
           "init_ckpt_fresh_keys": ic.get("fresh_keys"), "init_ckpt_n_mismatch": ic.get("n_mismatch_after_load"), "steps_total": s.get("steps_total"), "probe": s.get("lead_cond_probe", "ABSENT"),
           "text_has_lead_cond": "lead_cond" in json.dumps(s)}
    if "--lead-cond" not in extra:
        b = sc["base"][1]
        rec["key_set_equal_to_base_tree"] = flat(s) == flat(b) or sorted(flat(s) ^ flat(b))
        rec["key_set_symmetric_difference"] = sorted(flat(s) ^ flat(b))
        rec["gate_extra_wrong"] = {"levers.lead_cond=on": gate(f"{LC}/work/tmp/c6_{tag}_patch/selfcheck.json", ["levers.lead_cond=on"]).returncode}
        rec["PASS"] = bool(rc == 0 and rec["declared_vs_built"] == "PASS" and not rec["text_has_lead_cond"] and not rec["key_set_symmetric_difference"] and all(v != 0 for v in rec["gate_extra_wrong"].values())
                           and rec["init_ckpt_n_mismatch"] == 0)
    else:
        p = rec["probe"]
        exp = ["levers.wm_state=grid", "levers.lead_sup=on", "levers.w_lead_sup=1", "levers.lead_hist=5", "levers.lead_cond=on"] + (["levers.w_rule_own=4", "levers.rule_own_form=l1"] if "--w-rule-own" in extra else [])
        g = gate(f"{LC}/work/tmp/c6_{tag}_patch/selfcheck.json", exp)
        rec["gate_extra_right"] = {"rc": g.returncode, "out": g.stdout.strip()[:160], "expect": exp}
        rec["gate_extra_wrong"] = {e: gate(f"{LC}/work/tmp/c6_{tag}_patch/selfcheck.json", [e]).returncode for e in ("levers.lead_cond=off", "levers.lead_cond=on_wrong")}
        rec["PASS"] = bool(rc == 0 and rec["declared_vs_built"] == "PASS" and rec["levers_lead_cond"] == "on" and sorted(rec["init_ckpt_fresh_keys"] or []) == ["lead_cond.bias", "lead_cond.weight"] and rec["init_ckpt_n_mismatch"] == 0
                           and isinstance(p, dict) and p.get("pass") and p["grad_norm_weight"] > 0.0 and p["control_zero_u_grad_norm_weight"] == 0.0 and p["n_rows"] > 0 and p["grad_lead_head_through_u_max_abs_diff"] == 0.0
                           and g.returncode == 0 and all(v != 0 for v in rec["gate_extra_wrong"].values()))
    res["shapes"][tag] = rec
    print(tag, json.dumps({k: v for k, v in rec.items() if k != "probe"})[:900], "probe:", json.dumps(rec["probe"])[:700])
ref = {}
for tag, extra in (("on_without_lead_sup", ["--lead-cond", "on"]), ("unknown_value", ["--lead-sup", "on", "--w-lead-sup", "1.0", "--lead-cond", "maybe"])):
    r = run("patch", "refuse_" + tag, extra, f"{LC}/work/tmp/c6_refuse_{tag}")
    ref[tag] = {"rc": r.returncode, "tail": ((r.stdout or "") + (r.stderr or ""))[-230:].replace("\n", " ")}
    print("refusal", tag, r.returncode)
res["refusals"] = ref
res["refusals_ok"] = bool(ref["on_without_lead_sup"]["rc"] == 1 and ref["unknown_value"]["rc"] == 2)
res["PASS"] = bool(all(v["PASS"] for v in res["shapes"].values()) and res["refusals_ok"])
C.dump(f"{CK}/c6_gate.json", res)
print("C6", res["PASS"])
