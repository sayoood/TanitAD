"""o4 (the gate) and o5 (a checkpoint written with the linear form loads like its base) of RULEOWNL1. CPU, tiny synthetic world.
o4  trainer --selfcheck-only: default G (no rule_own key anywhere), GOW shape (--w-rule-own 1.0: levers.w_rule_own=1, no levers.rule_own_form key, probe form sq), GOL (--w-rule-own 1.0 --rule-own-form l1: levers.rule_own_form=l1,
    info.rule_own.form l1, rule_own_probe form l1 with finite term, n_lead > 0, counts, gradient norms, declared_vs_built PASS), GOL4 (4.0). gate_extra (the pod chain gate) accepts the right expectation (rc 0) and refuses a wrong one
    (levers.rule_own_form=sq on an l1 selfcheck, levers.rule_own_form=l1 on a default selfcheck, levers.w_rule_own=2). Refusals (rc != 0): --rule-own-form l1 without --w-rule-own, with --w-rule-own 0.0, with --ego-perturb-p.
    The train log of l1 and sq runs carries the same keys (rule_own_term, rule_own_n_lead, rule_own_n_active), none when off.
o5  the cfg of a checkpoint written with the linear form loads in the eval builder exactly as v8f_t1_eval.cmd_run builds it (strict load_state_dict); cfg keys and values and state-dict keys equal its base's.
Usage: o4_o5_l1.py"""
import hashlib
import json
import os
import subprocess
import sys

import torch

S = "F:/TanitAD-scratch/v8proof/ruletrain3/work"
CK = "F:/TanitAD-scratch/v8proof/ruletrain3/checks"
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
GATE = "F:/TanitAD-scratch/v8proof/pod/chain2/gate_extra.py"
BASE = f"{S}/tmp/ft/base/ckpt_last.pt"
w = lambda *p: "\\".join(p).replace("/", "\\")
root = f"{S}/tree_patch"
env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONIOENCODING="utf-8", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1", TMP=w(S, "tmp"), TEMP=w(S, "tmp"),
           PYTHONPATH=";".join([w(root, "stack"), w(root, "taniteval"), w(root, "stack", "scripts")]))
G = ["--arm", "M2b", "--grid", "pool4x10", "--kin-mode", "yaw", "--wm-state", "grid", "--wm-cell-dim", "16"]
TINY = ["--batch", "8", "--d", "32", "--n-q", "2"]
TR = f"{root}/stack/scripts/v8f_mini_train.py"
os.makedirs(f"{CK}/selfchecks", exist_ok=True)


def run(tag, extra, out, tiny=False):
    r = subprocess.run([PY, TR, "--synthetic", *G, *(TINY if tiny else []), *extra, "--exclude", "1:3", "--threads", "2" if not tiny else "1", "--out", out], env=env, capture_output=True, text=True, encoding="utf-8")
    with open(f"{CK}/selfchecks/o4_{tag}.log", "w", encoding="utf-8", newline="\n") as fh:
        fh.write((r.stdout or "")[-2500:] + "\nSTDERR\n" + (r.stderr or "")[-2500:] + f"\nRC {r.returncode}\n")
    return r


def gate(sc_path, exp):
    return subprocess.run([PY, GATE, "check", "--selfcheck", sc_path, "--card", "RULEOWNL1_SELFTEST", "--out", f"{S}/tmp/r3o4_extra_token.json", "--expect", *exp], capture_output=True, text=True, encoding="utf-8")


res = {"o4": {}}
cases = (("default", [], None, []), ("GOW", ["--w-rule-own", "1.0"], ["levers.wm_state=grid", "levers.w_rule_own=1"], ["levers.rule_own_form=l1", "levers.w_rule_own=2"]),
         ("GOL", ["--w-rule-own", "1.0", "--rule-own-form", "l1"], ["levers.wm_state=grid", "levers.w_rule_own=1", "levers.rule_own_form=l1"], ["levers.rule_own_form=sq", "levers.w_rule_own=2"]),
         ("GOL4", ["--w-rule-own", "4.0", "--rule-own-form", "l1"], ["levers.w_rule_own=4", "levers.rule_own_form=l1"], ["levers.rule_own_form=sq", "levers.w_rule_own=1"]))
for tag, extra, exp, wrong in cases:
    out = f"{S}/tmp/r3o4_{tag}"
    r = run(tag, [*extra, "--selfcheck-only"], out)
    sc = json.load(open(f"{out}/selfcheck.json", encoding="utf-8"))
    with open(f"{CK}/selfchecks/o4_selfcheck_{tag}.json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(sc, fh, indent=1)
    lv = sc["levers"]
    keys = [k for k in list(lv) + list(sc["info"]) + list(sc) if "rule_own" in k or k == "w_rule_own"]
    p = sc.get("rule_own_probe")
    rec = {"rc": r.returncode, "declared_vs_built": sc.get("declared_vs_built"), "live_pass": sc.get("live", {}).get("pass"), "levers_w_rule_own": lv.get("w_rule_own"), "levers_rule_own_form": lv.get("rule_own_form", "ABSENT"),
           "info_rule_own_form": (sc["info"].get("rule_own") or {}).get("form", "ABSENT" if sc["info"].get("rule_own") else None), "own_keys_present": keys, "probe": p}
    if exp:
        g = gate(f"{out}/selfcheck.json", exp)
        rec["gate_extra_right"] = {"rc": g.returncode, "out": g.stdout.strip()[:160], "expect": exp}
        rec["gate_extra_wrong"] = {e: {"rc": gate(f"{out}/selfcheck.json", [e]).returncode} for e in wrong}
    if tag == "default":
        rec["gate_extra_wrong"] = {"levers.rule_own_form=l1": {"rc": gate(f"{out}/selfcheck.json", ["levers.rule_own_form=l1"]).returncode}}
        rec["PASS"] = bool(r.returncode == 0 and rec["declared_vs_built"] == "PASS" and not keys and all(v["rc"] != 0 for v in rec["gate_extra_wrong"].values()))
    else:
        want_form = "sq" if tag == "GOW" else "l1"
        rec["PASS"] = bool(r.returncode == 0 and rec["declared_vs_built"] == "PASS" and rec["live_pass"] is not False and p and p["pass"] and p["form"] == want_form
                           and all(torch.isfinite(torch.tensor(float(p[k]))) for k in ("term", "grad_norm_term_on_traj30", "grad_norm_traj_loss_on_traj30")) and p["n_lead"] > 0 and "n_active" in p and p["control_all_false_term"] == 0.0
                           and (rec["levers_rule_own_form"] == ("ABSENT" if tag == "GOW" else "l1")) and rec["gate_extra_right"]["rc"] == 0 and all(v["rc"] != 0 for v in rec["gate_extra_wrong"].values()))
    res["o4"][tag] = rec
    print(tag, json.dumps({k: v for k, v in rec.items() if k != "probe"}), "probe:", json.dumps(p))
res["o4"]["refusals"] = {}
for tag, extra in (("l1_without_weight", ["--rule-own-form", "l1"]), ("l1_with_weight_zero", ["--w-rule-own", "0.0", "--rule-own-form", "l1"]), ("l1_with_ego_perturb", ["--w-rule-own", "1.0", "--rule-own-form", "l1", "--ego-perturb-p", "0.5"]),
                   ("unknown_form", ["--w-rule-own", "1.0", "--rule-own-form", "l2"])):
    r = run("refuse_" + tag, [*extra, "--selfcheck-only"], f"{S}/tmp/r3o4_refuse_{tag}")
    res["o4"]["refusals"][tag] = {"rc": r.returncode, "refused": r.returncode != 0, "tail": ((r.stdout or "") + (r.stderr or ""))[-260:].replace("\n", " ")}
    print("refusal", tag, r.returncode)
runs = {}
for tag, extra in (("off", []), ("sq", ["--w-rule-own", "1.0"]), ("l1", ["--w-rule-own", "1.0", "--rule-own-form", "l1"]), ("l1_4", ["--w-rule-own", "4.0", "--rule-own-form", "l1"])):
    out = f"{S}/tmp/r3o4_run_{tag}"
    r = run("run_" + tag, ["--init-ckpt", BASE, "--lr", "1e-4", "--warmup", "2", "--steps", "4", "--log-every", "1", "--ckpt-every", "100", *extra], out, tiny=True)
    rows = [json.loads(line) for line in open(f"{out}/train_log.jsonl", encoding="utf-8")]
    runs[tag] = {"rc": r.returncode, "rows": rows, "ck": torch.load(f"{out}/ckpt_last.pt", map_location="cpu", weights_only=False)}
lg = {}
for tag in runs:
    lg[tag] = {"n_lines": len(runs[tag]["rows"]), "rule_own_keys": sorted({k for x in runs[tag]["rows"] for k in x if "rule_own" in k}), "rc": runs[tag]["rc"]}
for tag in ("sq", "l1", "l1_4"):
    lg[tag]["term_per_line"] = [x["rule_own_term"] for x in runs[tag]["rows"]]
    lg[tag]["n_lead_per_line"] = [x["rule_own_n_lead"] for x in runs[tag]["rows"]]
    lg[tag]["n_active_per_line"] = [x["rule_own_n_active"] for x in runs[tag]["rows"]]
    lg[tag]["term_finite_all"] = all(torch.isfinite(torch.tensor(t)) for t in lg[tag]["term_per_line"])
lg["PASS"] = bool(not lg["off"]["rule_own_keys"] and lg["sq"]["rule_own_keys"] == lg["l1"]["rule_own_keys"] == lg["l1_4"]["rule_own_keys"] == ["rule_own_n_active", "rule_own_n_lead", "rule_own_term"]
                  and all(lg[t]["term_finite_all"] and min(lg[t]["n_lead_per_line"]) > 0 for t in ("sq", "l1", "l1_4")) and all(lg[t]["rc"] == 0 for t in runs))
res["o4"]["train_log"] = lg
res["o4"]["PASS"] = bool(all(res["o4"][t]["PASS"] for t in ("default", "GOW", "GOL", "GOL4")) and all(v["refused"] for v in res["o4"]["refusals"].values()) and lg["PASS"])
sys.path.insert(0, f"{root}/stack")
import tanitad  # noqa: E402

assert os.path.abspath(tanitad.__file__).lower().startswith(os.path.abspath(root).lower()), tanitad.__file__
from tanitad.models import v8f_mini as M  # noqa: E402

base = torch.load(BASE, map_location="cpu", weights_only=False)
eval_sha = hashlib.sha256(open(f"{root}/stack/scripts/v8f_t1_eval.py", "rb").read()).hexdigest()
res["o5"] = {"eval_tool_sha256": eval_sha, "eval_tool_sha256_is_pinned": eval_sha == "61e140edb9da01acbbba4eef81e74bbb68b57e2ff31842e05958759446d97397", "ckpts": {}}
for tag in ("l1", "l1_4"):
    ck = runs[tag]["ck"]
    cfg = M.V8FMiniConfig(**ck["cfg"])                                                                    # exactly what v8f_t1_eval.cmd_run does
    model = M.V8FMini(cfg, M.P10Prior(torch.zeros(8), torch.ones(8), torch.zeros(2 * M.K30)))
    r_load = model.load_state_dict(ck["model"])                                                             # strict
    res["o5"]["ckpts"][tag] = {"strict_load": str(r_load), "cfg_keys_equal_base": sorted(ck["cfg"]) == sorted(base["cfg"]), "cfg_values_equal_base": ck["cfg"] == base["cfg"], "state_keys_equal_base": list(ck["model"]) == list(base["model"]),
                               "cfg_equal_sq_run": ck["cfg"] == runs["sq"]["ck"]["cfg"], "extra_top_level_keys_vs_base": sorted(set(ck) - set(base)), "init_ckpt_n_mismatch": (ck.get("init_ckpt") or {}).get("n_mismatch_after_load")}
res["o5"]["PASS"] = bool(res["o5"]["eval_tool_sha256_is_pinned"] and all("all keys matched" in v["strict_load"].lower() and v["cfg_keys_equal_base"] and v["cfg_values_equal_base"] and v["state_keys_equal_base"] and v["cfg_equal_sq_run"]
                                                                       for v in res["o5"]["ckpts"].values()))
res["PASS"] = bool(res["o4"]["PASS"] and res["o5"]["PASS"])
json.dump(res, open(f"{CK}/o4_o5.json", "w", encoding="utf-8", newline="\n"), indent=1)
print("O4", res["o4"]["PASS"], "O5", res["o5"]["PASS"], json.dumps(res["o5"]))
