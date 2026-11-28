"""t4 (the gate) and t5 (the eval tool loads a relabel checkpoint like its base) of RULETRAIN. CPU, tiny synthetic world.
t4  trainer --selfcheck-only: default G (no rule_relabel key in levers / info), `--rule-relabel on` (levers.rule_relabel=on, info.rule_relabel_n > 0, declared_vs_built PASS, rule_relabel_probe pass, G-LIVE pass); gate_extra
    (the pod's chain gate) accepts the right expectations (rc 0) and refuses a wrong one (levers.rule_relabel=off, info.rule_relabel_n + 1; rc 1); refusals (rc != 0): on with --ego-perturb-p 0.5, on with --recover-p 0.5, on with
    a join that has no lead rows (in-process, the synthetic lead filler patched out). The train log carries rule_relabel_n at every logged step (from t_ft.json's run, 6 of 6 lines > 0) and a flag-off log carries none.
t5  the cfg of a relabel checkpoint loads in the eval builder exactly as v8f_t1_eval.cmd_run builds it (V8FMiniConfig(**ck["cfg"]), V8FMini, strict load_state_dict), its cfg key set and state-dict key set equal its base's, and the pinned
    eval tool file has the sha256 of the LEADSUP tree's (61e140ed...).
Usage: t4_t5.py"""
import hashlib
import json
import os
import subprocess
import sys

import torch

S = "C:/Users/Admin/v8proof_c/ruletrain"
CK = "F:/TanitAD-scratch/v8proof/ruletrain/checks"
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
GATE = "F:/TanitAD-scratch/v8proof/pod/chain2/gate_extra.py"
w = lambda *p: "\\".join(p).replace("/", "\\")
root = f"{S}/tree_patch"
env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONIOENCODING="utf-8", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1", TMP=w(S, "tmp"), TEMP=w(S, "tmp"),
           PYTHONPATH=";".join([w(root, "stack"), w(root, "taniteval"), w(root, "stack", "scripts")]))
G = ["--arm", "M2b", "--grid", "pool4x10", "--kin-mode", "yaw", "--wm-state", "grid", "--wm-cell-dim", "16"]
os.makedirs(f"{CK}/selfchecks", exist_ok=True)


def sc_run(tag, extra):
    out = f"{S}/tmp/sc4_{tag}"
    r = subprocess.run([PY, f"{root}/stack/scripts/v8f_mini_train.py", "--synthetic", *G, *extra, "--exclude", "1:3", "--threads", "2", "--selfcheck-only", "--out", out], env=env, capture_output=True, text=True, encoding="utf-8")
    with open(f"{CK}/selfchecks/selfcheck_{tag}.log", "w", encoding="utf-8", newline="\n") as fh:
        fh.write((r.stdout or "")[-3000:] + "\nSTDERR\n" + (r.stderr or "")[-2500:] + f"\nRC {r.returncode}\n")
    return r, out


res = {"t4": {}}
for tag, extra in (("default", []), ("RT", ["--rule-relabel", "on"])):
    r, out = sc_run(tag, extra)
    sc = json.load(open(f"{out}/selfcheck.json", encoding="utf-8"))
    with open(f"{CK}/selfchecks/selfcheck_{tag}.json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(sc, fh, indent=1)
    lv, info = sc["levers"], sc["info"]
    rec = {"rc": r.returncode, "declared_vs_built": sc.get("declared_vs_built"), "live_pass": sc.get("live", {}).get("pass"), "levers_rule_relabel": lv.get("rule_relabel"),
           "info_rule_relabel_n": info.get("rule_relabel_n"), "info_rule_relabel_changed": info.get("rule_relabel_changed"), "probe": sc.get("rule_relabel_probe"),
           "extra_keys_when_off": [k for k in list(lv) + list(info) + list(sc) if "rule_relabel" in k] if tag == "default" else None}
    if tag == "RT":
        n = info["rule_relabel_n"]
        exp = ["levers.wm_state=grid", "levers.rule_relabel=on", f"info.rule_relabel_n={n}", "info.n_clips=4"]
        g = subprocess.run([PY, GATE, "check", "--selfcheck", f"{out}/selfcheck.json", "--card", "RULETRAIN_SELFTEST", "--out", f"{out}/extra_token.json", "--expect", *exp], capture_output=True, text=True, encoding="utf-8")
        rec["gate_extra_right"] = {"rc": g.returncode, "out": g.stdout.strip()[:200], "expect": exp}
        rec["gate_extra_wrong"] = {}
        for k, e in (("levers.rule_relabel", "levers.rule_relabel=off"), ("info.rule_relabel_n", f"info.rule_relabel_n={n + 1}")):
            b = subprocess.run([PY, GATE, "check", "--selfcheck", f"{out}/selfcheck.json", "--card", "x", "--out", f"{out}/never.json", "--expect", e], capture_output=True, text=True, encoding="utf-8")
            rec["gate_extra_wrong"][k] = {"expected": e, "rc": b.returncode, "out": b.stdout.strip()[:160]}
        rec["PASS"] = bool(r.returncode == 0 and rec["declared_vs_built"] == "PASS" and rec["live_pass"] is not False and lv.get("rule_relabel") == "on" and n > 0 and rec["probe"]["pass"] and g.returncode == 0
                           and all(v["rc"] != 0 for v in rec["gate_extra_wrong"].values()))
    else:
        rec["PASS"] = bool(r.returncode == 0 and rec["declared_vs_built"] == "PASS" and not rec["extra_keys_when_off"])
    res["t4"][tag] = rec
    print(tag, json.dumps({k: v for k, v in rec.items() if k != "probe"}))
res["t4"]["refusals"] = {}
for tag, extra in (("on_with_ego_perturb", ["--rule-relabel", "on", "--ego-perturb-p", "0.5"]), ("on_with_recover", ["--rule-relabel", "on", "--recover-p", "0.5"])):
    r, out = sc_run("refuse_" + tag, extra)
    res["t4"]["refusals"][tag] = {"rc": r.returncode, "refused": r.returncode != 0, "tail": ((r.stdout or "") + (r.stderr or ""))[-280:].replace("\n", " ")}
    print("refusal", tag, r.returncode)
# a join with no lead rows: the synthetic lead filler is patched out in-process; the gate must refuse (labels n_pres 0)
code = ('import sys,os;sys.path.insert(0,r"%s/stack/scripts");import v8f_mini_train as TR;from tanitad.data import v8f_mini_data as D;D.synthetic_lead_labels=lambda J,seed=0:J;'
        'sys.exit(TR.main(["--synthetic","--arm","M2b","--grid","pool4x10","--kin-mode","yaw","--wm-state","grid","--wm-cell-dim","16","--rule-relabel","on","--exclude","1:3","--threads","2","--selfcheck-only","--out",r"%s/tmp/sc4_nolead"]))' % (root, S))
r = subprocess.run([PY, "-c", code], env=env, capture_output=True, text=True, encoding="utf-8")
res["t4"]["refusals"]["on_with_no_lead_rows"] = {"rc": r.returncode, "refused": r.returncode != 0, "tail": ((r.stdout or "") + (r.stderr or ""))[-280:].replace("\n", " ")}
print("refusal no lead", r.returncode)
lines = [json.loads(l) for l in open(f"{S}/tmp/ft_v3/run_s0_on/train_log.jsonl", encoding="utf-8")]
lines_off = [json.loads(l) for l in open(f"{S}/tmp/ft_v3/run_s0_off/train_log.jsonl", encoding="utf-8")]
res["t4"]["train_log"] = {"on_lines": len(lines), "on_lines_with_count_gt0": sum(1 for x in lines if x.get("rule_relabel_n", 0) > 0), "off_lines_with_any_rule_key": sum(1 for x in lines_off if any("rule_relabel" in k for k in x))}
res["t4"]["PASS"] = bool(res["t4"]["default"]["PASS"] and res["t4"]["RT"]["PASS"] and all(v["refused"] for v in res["t4"]["refusals"].values())
                         and res["t4"]["train_log"]["on_lines"] > 0 and res["t4"]["train_log"]["on_lines"] == res["t4"]["train_log"]["on_lines_with_count_gt0"] and res["t4"]["train_log"]["off_lines_with_any_rule_key"] == 0)

# ---- t5
sys.path.insert(0, f"{root}/stack")
os.environ["PYTHONPATH"] = env["PYTHONPATH"]
import tanitad  # noqa: E402

assert os.path.abspath(tanitad.__file__).lower().startswith(os.path.abspath(root).lower()), tanitad.__file__
from tanitad.models import v8f_mini as M  # noqa: E402

base = torch.load(f"{S}/tmp/ft_v3/base/ckpt_last.pt", map_location="cpu", weights_only=False)
rt = torch.load(f"{S}/tmp/ft_v3/run_s0_on/ckpt_last.pt", map_location="cpu", weights_only=False)
cfg = M.V8FMiniConfig(**rt["cfg"])                                                                   # exactly what v8f_t1_eval.cmd_run does
model = M.V8FMini(cfg, M.P10Prior(torch.zeros(8), torch.ones(8), torch.zeros(2 * M.K30)))
r_load = model.load_state_dict(rt["model"])                                                            # strict
eval_sha = hashlib.sha256(open(f"{root}/stack/scripts/v8f_t1_eval.py", "rb").read()).hexdigest()
res["t5"] = {"strict_load": str(r_load), "cfg_keys_equal_base": sorted(rt["cfg"]) == sorted(base["cfg"]), "cfg_values_equal_base": rt["cfg"] == base["cfg"], "state_keys_equal_base": list(rt["model"]) == list(base["model"]),
             "extra_top_level_keys_vs_base": sorted(set(rt) - set(base)), "eval_tool_sha256": eval_sha, "eval_tool_sha256_is_pinned": eval_sha == "61e140edb9da01acbbba4eef81e74bbb68b57e2ff31842e05958759446d97397"}
res["t5"]["PASS"] = bool("all keys matched" in res["t5"]["strict_load"].lower() and res["t5"]["cfg_keys_equal_base"] and res["t5"]["cfg_values_equal_base"] and res["t5"]["state_keys_equal_base"] and res["t5"]["eval_tool_sha256_is_pinned"])
res["PASS"] = bool(res["t4"]["PASS"] and res["t5"]["PASS"])
json.dump(res, open(f"{CK}/t4_t5.json", "w", encoding="utf-8", newline="\n"), indent=1)
print("T4", res["t4"]["PASS"], "T5", res["t5"]["PASS"], json.dumps(res["t5"]))
