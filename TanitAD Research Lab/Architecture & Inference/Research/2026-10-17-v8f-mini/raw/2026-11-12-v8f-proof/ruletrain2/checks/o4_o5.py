"""o4 (the gate), w4 (selfcheck part) and o5 (a checkpoint written with a flag loads like its base) of RULEOWN. CPU, tiny synthetic world.
o4  trainer --selfcheck-only: default G (no rule_own / weight key anywhere), `--w-rule-own 1.0` (levers.w_rule_own, info.rule_own, rule_own_probe finite with n_lead > 0, n_active, gradient norms, declared_vs_built PASS), `--w-rule-own 4.0
    --rule-relabel on`, `--rule-relabel on --rule-relabel-weight 4` (levers.rule_relabel_weight). gate_extra (the pod chain gate) accepts the right expectation (rc 0) and refuses a wrong one (rc 1). Refusals (rc != 0): own with
    --ego-perturb-p, --recover-p, --dagger-p; the weight without --rule-relabel on; own on a join with no lead rows (in-process, the synthetic lead filler patched out). The train log carries the three keys at every logged step
    when on (term finite) and none when off (4-step runs).
o5  the cfg of a checkpoint written with the flag loads in the eval builder exactly as v8f_t1_eval.cmd_run builds it (V8FMiniConfig(**ck["cfg"]), V8FMini, strict load_state_dict); cfg keys and values and state-dict keys equal its base's;
    the eval tool file has the pinned sha256. The checkpoints are the 4-step runs above, started from the tiny base checkpoint with --init-ckpt (the fine-tune shape).
Usage: o4_o5.py"""
import hashlib
import json
import os
import subprocess
import sys

import torch

S = "F:/TanitAD-scratch/v8proof/ruletrain2/work"
CK = "F:/TanitAD-scratch/v8proof/ruletrain2/checks"
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
    return subprocess.run([PY, GATE, "check", "--selfcheck", sc_path, "--card", "RULEOWN_SELFTEST", "--out", f"{S}/tmp/r3o4_extra_token.json", "--expect", *exp], capture_output=True, text=True, encoding="utf-8")


res = {"o4": {}}
cases = (("default", [], None), ("GOW", ["--w-rule-own", "1.0"], ["levers.wm_state=grid", "levers.w_rule_own=1"]), ("GOW4_RT", ["--w-rule-own", "4.0", "--rule-relabel", "on"], ["levers.w_rule_own=4", "levers.rule_relabel=on"]),
         ("GRW", ["--rule-relabel", "on", "--rule-relabel-weight", "4"], ["levers.rule_relabel=on", "levers.rule_relabel_weight=4"]))
for tag, extra, exp in cases:
    out = f"{S}/tmp/r3o4_{tag}"
    r = run(tag, [*extra, "--selfcheck-only"], out)
    sc = json.load(open(f"{out}/selfcheck.json", encoding="utf-8"))
    with open(f"{CK}/selfchecks/o4_selfcheck_{tag}.json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(sc, fh, indent=1)
    lv = sc["levers"]
    keys = [k for k in list(lv) + list(sc["info"]) + list(sc) if "rule_own" in k or "rule_relabel_weight" in k or k == "w_rule_own"]
    p = sc.get("rule_own_probe")
    rec = {"rc": r.returncode, "declared_vs_built": sc.get("declared_vs_built"), "live_pass": sc.get("live", {}).get("pass"), "levers_w_rule_own": lv.get("w_rule_own"), "levers_rule_relabel_weight": lv.get("rule_relabel_weight"),
           "own_keys_present": keys, "probe": p, "info_rule_own": (sc["info"].get("rule_own") or {}).get("w_rule_own")}
    if exp:
        g = gate(f"{out}/selfcheck.json", exp)
        rec["gate_extra_right"] = {"rc": g.returncode, "out": g.stdout.strip()[:160], "expect": exp}
        wrong = {"levers.w_rule_own=2": "levers.w_rule_own=2", "levers.w_rule_own=0": "levers.w_rule_own=0"} if "w_rule_own" in lv else {"levers.rule_relabel_weight=2": "levers.rule_relabel_weight=2"}
        rec["gate_extra_wrong"] = {k: {"rc": gate(f"{out}/selfcheck.json", [e]).returncode} for k, e in wrong.items()}
    if tag == "default":
        rec["PASS"] = bool(r.returncode == 0 and rec["declared_vs_built"] == "PASS" and not keys)
    elif tag in ("GOW", "GOW4_RT"):
        rec["PASS"] = bool(r.returncode == 0 and rec["declared_vs_built"] == "PASS" and rec["live_pass"] is not False and p and p["pass"] and all(torch.isfinite(torch.tensor(float(p[k]))) for k in ("term", "grad_norm_term_on_traj30", "grad_norm_traj_loss_on_traj30"))
                           and p["n_lead"] > 0 and "n_active" in p and p["control_all_false_term"] == 0.0 and rec["gate_extra_right"]["rc"] == 0 and all(v["rc"] != 0 for v in rec["gate_extra_wrong"].values()))
    else:
        rec["PASS"] = bool(r.returncode == 0 and rec["declared_vs_built"] == "PASS" and lv.get("rule_relabel_weight") == 4.0 and "rule_own_probe" not in sc and rec["gate_extra_right"]["rc"] == 0
                           and all(v["rc"] != 0 for v in rec["gate_extra_wrong"].values()))
    res["o4"][tag] = rec
    print(tag, json.dumps({k: v for k, v in rec.items() if k != "probe"}), "probe:", json.dumps(p))
res["o4"]["refusals"] = {}
for tag, extra in (("own_with_ego_perturb", ["--w-rule-own", "1.0", "--ego-perturb-p", "0.5"]), ("own_with_recover", ["--w-rule-own", "1.0", "--recover-p", "0.5"]), ("own_with_dagger", ["--w-rule-own", "1.0", "--dagger-p", "0.5"]),
                   ("weight_without_relabel", ["--rule-relabel-weight", "4"]), ("negative_own", ["--w-rule-own", "-1.0"])):
    r = run("refuse_" + tag, [*extra, "--selfcheck-only"], f"{S}/tmp/r3o4_refuse_{tag}")
    res["o4"]["refusals"][tag] = {"rc": r.returncode, "refused": r.returncode != 0, "tail": ((r.stdout or "") + (r.stderr or ""))[-260:].replace("\n", " ")}
    print("refusal", tag, r.returncode)
code = ('import sys,os;sys.path.insert(0,r"%s/stack/scripts");import v8f_mini_train as TR;from tanitad.data import v8f_mini_data as D;D.synthetic_lead_labels=lambda J,seed=0:J;'
        'sys.exit(TR.main(["--synthetic","--arm","M2b","--grid","pool4x10","--kin-mode","yaw","--wm-state","grid","--wm-cell-dim","16","--w-rule-own","1.0","--exclude","1:3","--threads","2","--selfcheck-only","--out",r"%s/tmp/r3o4_nolead"]))' % (root, S))
r = subprocess.run([PY, "-c", code], env=env, capture_output=True, text=True, encoding="utf-8")
res["o4"]["refusals"]["own_on_a_join_with_no_lead_rows"] = {"rc": r.returncode, "refused": r.returncode != 0, "tail": ((r.stdout or "") + (r.stderr or ""))[-260:].replace("\n", " ")}
print("refusal no lead", r.returncode)
# train logs and checkpoints: 4-step fine-tune shape from the tiny base, flags on and off
runs = {}
for tag, extra in (("off", []), ("own", ["--w-rule-own", "1.0"]), ("own4", ["--w-rule-own", "4.0"]), ("wt", ["--rule-relabel", "on", "--rule-relabel-weight", "4"]), ("rt", ["--rule-relabel", "on"])):
    out = f"{S}/tmp/r3o4_run_{tag}"
    r = run("run_" + tag, ["--init-ckpt", BASE, "--lr", "1e-4", "--warmup", "2", "--steps", "4", "--log-every", "1", "--ckpt-every", "100", *extra], out, tiny=True)
    rows = [json.loads(line) for line in open(f"{out}/train_log.jsonl", encoding="utf-8")]
    runs[tag] = {"rc": r.returncode, "rows": rows, "ck": torch.load(f"{out}/ckpt_last.pt", map_location="cpu", weights_only=False)}
lg = {}
for tag in runs:
    keys = sorted({k for x in runs[tag]["rows"] for k in x if "rule_own" in k or "weighted" in k})
    lg[tag] = {"n_lines": len(runs[tag]["rows"]), "rule_own_or_weighted_keys": keys}
lg["own"]["term_finite_all"] = all(torch.isfinite(torch.tensor(x["rule_own_term"])) for x in runs["own"]["rows"])
lg["own"]["n_lead_per_line"] = [x["rule_own_n_lead"] for x in runs["own"]["rows"]]
lg["own"]["n_active_per_line"] = [x["rule_own_n_active"] for x in runs["own"]["rows"]]
lg["own"]["term_per_line"] = [x["rule_own_term"] for x in runs["own"]["rows"]]
lg["wt"]["weighted_n_per_line"] = [x["rule_relabel_weighted_n"] for x in runs["wt"]["rows"]]
lg["own_total_minus_off_total"] = [a["total"] - b["total"] for a, b in zip(runs["own"]["rows"], runs["off"]["rows"])]
lg["wt_total_minus_rt_total"] = [a["total"] - b["total"] for a, b in zip(runs["wt"]["rows"], runs["rt"]["rows"])]
res["o4"]["train_log"] = lg
res["o4"]["train_log"]["PASS"] = bool(not lg["off"]["rule_own_or_weighted_keys"] and not lg["rt"]["rule_own_or_weighted_keys"] and lg["own"]["rule_own_or_weighted_keys"] == ["rule_own_n_active", "rule_own_n_lead", "rule_own_term"]
                                      and lg["own"]["term_finite_all"] and min(lg["own"]["n_lead_per_line"]) > 0 and lg["wt"]["rule_own_or_weighted_keys"] == ["rule_relabel_weighted_n"] and lg["own4"]["rule_own_or_weighted_keys"] == ["rule_own_n_active", "rule_own_n_lead", "rule_own_term"])
res["o4"]["PASS"] = bool(all(res["o4"][t]["PASS"] for t in ("default", "GOW", "GOW4_RT", "GRW")) and all(v["refused"] for v in res["o4"]["refusals"].values()) and res["o4"]["train_log"]["PASS"])
# ---- o5
sys.path.insert(0, f"{root}/stack")
import tanitad  # noqa: E402

assert os.path.abspath(tanitad.__file__).lower().startswith(os.path.abspath(root).lower()), tanitad.__file__
from tanitad.models import v8f_mini as M  # noqa: E402

base = torch.load(BASE, map_location="cpu", weights_only=False)
eval_sha = hashlib.sha256(open(f"{root}/stack/scripts/v8f_t1_eval.py", "rb").read()).hexdigest()
res["o5"] = {"eval_tool_sha256": eval_sha, "eval_tool_sha256_is_pinned": eval_sha == "61e140edb9da01acbbba4eef81e74bbb68b57e2ff31842e05958759446d97397", "ckpts": {}}
for tag in ("own", "own4", "wt"):
    ck = runs[tag]["ck"]
    cfg = M.V8FMiniConfig(**ck["cfg"])                                                                    # exactly what v8f_t1_eval.cmd_run does
    model = M.V8FMini(cfg, M.P10Prior(torch.zeros(8), torch.ones(8), torch.zeros(2 * M.K30)))
    r_load = model.load_state_dict(ck["model"])                                                             # strict
    res["o5"]["ckpts"][tag] = {"strict_load": str(r_load), "cfg_keys_equal_base": sorted(ck["cfg"]) == sorted(base["cfg"]), "cfg_values_equal_base": ck["cfg"] == base["cfg"], "state_keys_equal_base": list(ck["model"]) == list(base["model"]),
                               "extra_top_level_keys_vs_base": sorted(set(ck) - set(base)), "init_ckpt_n_mismatch": (ck.get("init_ckpt") or {}).get("n_mismatch_after_load")}
res["o5"]["PASS"] = bool(res["o5"]["eval_tool_sha256_is_pinned"] and all("all keys matched" in v["strict_load"].lower() and v["cfg_keys_equal_base"] and v["cfg_values_equal_base"] and v["state_keys_equal_base"]
                                                                       for v in res["o5"]["ckpts"].values()))
res["PASS"] = bool(res["o4"]["PASS"] and res["o5"]["PASS"])
json.dump(res, open(f"{CK}/o4_o5.json", "w", encoding="utf-8", newline="\n"), indent=1)
print("O4", res["o4"]["PASS"], "O5", res["o5"]["PASS"], json.dumps(res["o5"]))
