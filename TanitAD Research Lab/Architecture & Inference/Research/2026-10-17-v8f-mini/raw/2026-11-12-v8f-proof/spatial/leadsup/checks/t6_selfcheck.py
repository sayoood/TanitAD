"""t6: the trainer selfcheck (--selfcheck-only, synthetic world with the opt-in synthetic lead track, --exclude 1:3 as the previous levers did) of the LEADSUP tree_patch for
  default   grid state, flat predictor (arm G): the built levers read off / 5 / off / 0
  LS        --lead-sup on --w-lead-sup 1.0
  LSD       LS + --lead-sup-detach on
  LS3       LS + --lead-hist 3
It reads the BUILT levers out of selfcheck.json, runs the gate_extra expectations (the right ones PASS, a deliberately wrong one per lever is REFUSED, rc 1), checks that the trainer's declared_vs_built
list is PASS, the G-LIVE terms (lead_sup and its three parts finite in the log; no dead parameter), and then the REFUSAL cases: each declaration that does nothing / cannot be built must make the selfcheck exit non-zero.
Output: checks/t6_selfcheck.json and checks/selfchecks/*.json|log.  Usage: t6_selfcheck.py"""
import json
import os
import subprocess

S = "C:/Users/Admin/v8proof_c/leadsup"
CK = "F:/TanitAD-scratch/v8proof/spatial/leadsup/checks"
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
GATE = "F:/TanitAD-scratch/v8proof/pod/chain2/gate_extra.py"
w = lambda *p: "\\".join(p).replace("/", "\\")
root = f"{S}/tree_patch"
env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONIOENCODING="utf-8", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1",
           TMP=w(S, "tmp"), TEMP=w(S, "tmp"), PYTHONPATH=";".join([w(root, "stack"), w(root, "taniteval"), w(root, "stack", "scripts")]))
GRIDARM = ["--arm", "M2b", "--grid", "pool4x10", "--kin-mode", "yaw", "--wm-state", "grid", "--wm-cell-dim", "16"]
LS = ["--lead-sup", "on", "--w-lead-sup", "1.0"]
ARMS = {
    "default": ([], {"lead_sup": "off", "lead_hist": "5", "lead_sup_detach": "off", "w_lead_sup": "0"}),
    "LS": (LS, {"lead_sup": "on", "lead_hist": "5", "lead_sup_detach": "off", "w_lead_sup": "1"}),
    "LSD": (LS + ["--lead-sup-detach", "on"], {"lead_sup": "on", "lead_hist": "5", "lead_sup_detach": "on", "w_lead_sup": "1"}),
    "LS3": (LS + ["--lead-hist", "3"], {"lead_sup": "on", "lead_hist": "3", "lead_sup_detach": "off", "w_lead_sup": "1"}),
}
WRONG = {"lead_sup": {"off": "on", "on": "off"}, "lead_hist": {"5": "4", "3": "5"}, "lead_sup_detach": {"off": "on", "on": "off"}, "w_lead_sup": {"0": "1", "1": "0"}}
REFUSALS = {
    "on_with_weight_0": LS[:2] + ["--w-lead-sup", "0.0"],
    "weight_with_head_off": ["--w-lead-sup", "1.0"],
    "detach_with_head_off": ["--lead-sup-detach", "on"],
    "hist_with_head_off": ["--lead-hist", "3"],
    "head_on_with_mean_state": LS + ["--wm-state", "mean"],
    "hist_longer_than_the_window": LS + ["--lead-hist", "17"],
}
res = {}
os.makedirs(f"{CK}/selfchecks", exist_ok=True)


def run(tag, extra, grid=GRIDARM):
    out = f"{S}/tmp/sc_{tag}"
    r = subprocess.run([PY, f"{root}/stack/scripts/v8f_mini_train.py", "--synthetic", *grid, *extra, "--exclude", "1:3", "--threads", "2", "--selfcheck-only", "--out", out],
                       env=env, capture_output=True, text=True, encoding="utf-8")
    with open(f"{CK}/selfchecks/selfcheck_{tag}.log", "w", encoding="utf-8", newline="\n") as fh:
        fh.write((r.stdout or "")[-3000:] + "\nSTDERR\n" + (r.stderr or "")[-2500:] + f"\nRC {r.returncode}\n")
    return r, out


for tag, (extra, want) in ARMS.items():
    r, out = run(tag, extra)
    sc = json.load(open(f"{out}/selfcheck.json", encoding="utf-8"))
    with open(f"{CK}/selfchecks/selfcheck_{tag}.json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(sc, fh, indent=1)
    lv, info = sc["levers"], sc["info"]
    terms = sc.get("live", {}).get("terms", {})
    rec = {"rc": r.returncode, "declared_vs_built": sc.get("declared_vs_built"), "live_pass": sc.get("live", {}).get("pass"), "built": {k: lv.get(k) for k in want},
           "live_terms": {k: terms.get(k) for k in ("lead_sup", "lead_sup_bce", "lead_sup_gap", "lead_sup_clos", "wm", "total")},
           "live_missing_terms": sc.get("live", {}).get("missing_terms"), "live_unexpected_dead": sc.get("live", {}).get("unexpected_dead"),
           "info_lead": {k: v for k, v in info.items() if k.startswith("lead_sup") or k == "ds_lead_sup"}}
    exp = ["levers.wm_state=grid", "info.n_clips=4"] + [f"levers.{k}={v}" for k, v in want.items()]
    if want["lead_sup"] == "on":
        exp.append(f"info.lead_sup_n_labelled={info['lead_sup_n_labelled']}")
    g = subprocess.run([PY, GATE, "check", "--selfcheck", f"{out}/selfcheck.json", "--card", "LEADSUP_SELFTEST", "--out", f"{out}/extra_token.json", "--expect", *exp],
                       capture_output=True, text=True, encoding="utf-8")
    rec["gate_extra_right_expectations"] = {"rc": g.returncode, "out": g.stdout.strip()[:240], "expect": exp}
    rec["gate_extra_wrong_expectation_refused"] = {}
    for k, flip in WRONG.items():
        bad = subprocess.run([PY, GATE, "check", "--selfcheck", f"{out}/selfcheck.json", "--card", "x", "--out", f"{out}/never.json", "--expect", f"levers.{k}={flip[want[k]]}"],
                             capture_output=True, text=True, encoding="utf-8")
        rec["gate_extra_wrong_expectation_refused"][k] = {"expected": flip[want[k]], "rc": bad.returncode, "out": bad.stdout.strip()[:160]}
    if want["lead_sup"] == "on":
        bad = subprocess.run([PY, GATE, "check", "--selfcheck", f"{out}/selfcheck.json", "--card", "x", "--out", f"{out}/never.json", "--expect", f"info.lead_sup_n_labelled={info['lead_sup_n_labelled'] + 1}"],
                             capture_output=True, text=True, encoding="utf-8")
        rec["gate_extra_wrong_expectation_refused"]["info.lead_sup_n_labelled"] = {"rc": bad.returncode}
    on = want["lead_sup"] == "on"
    rec["PASS"] = bool(r.returncode == 0 and rec["declared_vs_built"] in ([], None, "PASS") and rec["live_pass"] is not False and g.returncode == 0
                       and all(v["rc"] != 0 for v in rec["gate_extra_wrong_expectation_refused"].values())
                       and all(str(lv.get(k)) == v or (v == "0" and float(lv.get(k)) == 0.0) or (k == "w_lead_sup" and float(lv.get(k)) == float(v)) for k, v in want.items())
                       and (not on or (all(rec["live_terms"][k] is not None and rec["live_terms"][k] == rec["live_terms"][k] for k in ("lead_sup", "lead_sup_bce", "lead_sup_gap", "lead_sup_clos"))
                                       and rec["live_missing_terms"] == [] and rec["live_unexpected_dead"] == [] and info["lead_sup_n_labelled"] > 0 and info["ds_lead_sup"] is True))
                       and (on or ("lead_sup" not in terms and "lead_sup_n_labelled" not in info)))
    res[tag] = rec
    print(tag, json.dumps(rec))
res["refusals"] = {}
for tag, extra in REFUSALS.items():
    grid = [x for x in GRIDARM]
    if "--wm-state" in extra:                                # the arm's own --wm-state is replaced (mean has no cell dim)
        grid = ["--arm", "M2b", "--grid", "pool4x10", "--kin-mode", "yaw"]
    r, out = run("refuse_" + tag, extra, grid)
    msg = ((r.stdout or "") + (r.stderr or ""))[-700:].replace("\n", " ")
    res["refusals"][tag] = {"rc": r.returncode, "refused": r.returncode != 0, "message_tail": msg[-300:]}
    print("refusal", tag, r.returncode)
res["refusals"]["PASS"] = all(v["refused"] for k, v in res["refusals"].items())
res["PASS"] = all(v["PASS"] for k, v in res.items() if k != "refusals") and res["refusals"]["PASS"]
json.dump(res, open(f"{CK}/t6_selfcheck.json", "w", encoding="utf-8", newline="\n"), indent=1)
print("T6", "PASS" if res["PASS"] else "FAIL")
