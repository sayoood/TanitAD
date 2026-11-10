"""Unit tests of stopbal_check.py (pure Python; numpy only for the moved identity; no torch, no GPU, no network).
Run:  C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I tests/test_stopbal_check.py <scratch dir>     -> prints  UNIT ok=<n> bad=<m>  (exit 1 on any bad)
Every gate assertion has a planted-bad selfcheck that must be refused, for every tag; every clean selfcheck must pass."""
import copy
import hashlib
import json
import os
import pickle
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, PKG)
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import stopbal_check as CK  # noqa: E402
import plant_trainer as PT  # noqa: E402

N_OK, N_BAD = 0, 0
DAGOACC = "E:/Projects/TanitAD-artifacts/v8f-dagoacc"
PUSH = "C:/Users/Admin/tanitad-push"
COMMIT_REC = "2bca1c52abbc2346d7380d33ae78b324cbda9df7"
COMMIT_PARENT = "5514a6840b57ad115dcadeb7e264b10f5c1d0081"
OBD_ARGVS = os.path.join(PKG, "..", "..", "obd", "argvs_obd.json")


def check(name, cond):
    global N_OK, N_BAD
    if cond:
        N_OK += 1
    else:
        N_BAD += 1
        print(f"UNIT-FAIL {name}")


def sc_of(tag):
    toks = CK.tokens(tag)
    gt = CK.gate_argv(tag, toks)
    os.environ["STOPBAL_FAKE_INIT_SHA_FILE"] = os.path.join(TMP, "init_sha.json")
    os.environ["STOPBAL_FAKE_BANK_SHA_FILE"] = os.path.join(TMP, "bank_sha.json")
    return toks, PT.good_selfcheck(gt)


def put(p, data=b""):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "wb").write(data)


def main(tmp):
    global TMP
    TMP = tmp
    os.makedirs(tmp, exist_ok=True)
    json.dump({"0": CK.INIT_SHA[0], "1": CK.INIT_SHA[1]}, open(os.path.join(tmp, "init_sha.json"), "w"))
    json.dump({"0": CK.BANK_SHA[0], "1": CK.BANK_SHA[1]}, open(os.path.join(tmp, "bank_sha.json"), "w"))
    runnable = CK.RUN_ORDER + [CK.PILOT_TAG]
    # ---- the card's order and the table
    check("run order is the card's: NULL_s0, NULL_rep, G_s0, L_s0, NULL_s1, G_s1, L_s1", CK.RUN_ORDER == ["BAL-NULL_s0", "BAL-NULL_rep", "BAL-G_s0", "BAL-L_s0", "BAL-NULL_s1", "BAL-G_s1", "BAL-L_s1"])
    check("seven runs, no conditional arm in the table", len(CK.TABLE) == 7 and not any(v.get("conditional") for v in CK.TABLE.values()) and not hasattr(CK, "CONDITIONAL_TAG"))
    check("steps: 3,000 per run, pilot 200, limit 12 h", CK.RUN_STEPS == 3000 and CK.PILOT_STEPS == 200 and CK.LIMIT_H == 12.0)
    check("the parity run is BAL-NULL_s0, the pilot is BAL-G_s0's, the pilot also gates G, L and NULL", CK.PARITY_TAG == "BAL-NULL_s0" and CK.PILOT_BASE == "BAL-G_s0" and set(CK.GATE_AT_PILOT) == {"BAL-G_s0", "BAL-L_s0", "BAL-NULL_s0"})
    check("the card's weights: goal 4.5616, lon 4.6991", CK.W_GOAL == 4.5616 and CK.W_LON == 4.6991 and CK.WEIGHTS == {"null": (1.0, 1.0), "g": (4.5616, 1.0), "l": (1.0, 4.6991)})
    check("the OBD-only machinery is gone (ACC_PROBE_REQUIRED, purity_keys, W_OP)", not any(hasattr(CK, n) for n in ("ACC_PROBE_REQUIRED", "purity_keys", "W_OP", "W_TAC")))
    for t, v in CK.TABLE.items():
        a = CK.tokens(t)
        check(f"{t}: table seed / init equal the pinned argv's --seed / init-ckpt / --dagger-seed", CK.flag_value(a, "--seed") == str(v["seed"]) and f"M2b_s{v['init']}" in CK.flag_value(a, "--init-ckpt") and CK.flag_value(a, "--dagger-seed") == str(v["init"]))
        wg, wl = CK.WEIGHTS[v["kind"]]
        check(f"{t}: the argv carries exactly its own lever ({wg} / {wl})", CK.flag_value(a, "--goal-pos-weight-stop") == (None if wg == 1.0 else repr(wg)) and CK.flag_value(a, "--lon-class-weight-stop") == (None if wl == 1.0 else repr(wl)))
    # ---- argv pin and composition (each arm = DAG-OACC's argv + only the declared changes)
    for t in CK.TABLE:
        check(f"frozen argv of {t} is accepted", CK.audit_line(t, " ".join(CK.tokens(t))) == [])
    check("pilot argv = BAL-G_s0 argv with --out moved and --steps 200 appended", CK.tokens(CK.PILOT_TAG)[:-2] == [CK.out_decl(CK.PILOT_TAG) if x == CK.out_decl("BAL-G_s0") else x for x in CK.tokens("BAL-G_s0")] and CK.tokens(CK.PILOT_TAG)[-2:] == ["--steps", "200"])
    check("pilot out dir differs from the run out dir", CK.out_decl(CK.PILOT_TAG) != CK.out_decl("BAL-G_s0"))
    for t in CK.TABLE:
        check(f"{t}: --out is D:\\tanitad\\runs\\v8f_stopbal\\<tag>", CK.out_decl(t) == "D:\\tanitad\\runs\\v8f_stopbal\\" + t)
    extra_of = {"null": [], "g": ["--goal-pos-weight-stop", "4.5616"], "l": ["--lon-class-weight-stop", "4.6991"]}
    if os.path.isfile(f"{DAGOACC}/M2b_s0_dagoacc/argv.json"):
        base = {k: json.load(open(f"{DAGOACC}/M2b_s{k}_dagoacc/argv.json", encoding="utf-8"))["argv"] for k in (0, 1)}
        for t, v in CK.TABLE.items():
            want = list(base[v["init"]])
            want[want.index("--out") + 1] = CK.out_decl(t)
            if t.endswith("_rep"):
                want[want.index("--seed") + 1] = "2"
            check(f"{t}: equals DAG-OACC s{v['init']}'s argv.json with only --out{', --seed 2' if t.endswith('_rep') else ''}{' and ' + ' '.join(extra_of[v['kind']]) if extra_of[v['kind']] else ''} changed", CK.tokens(t) == want + extra_of[v["kind"]])
        check("DAG-OACC s1's bank sha256 in the checker is the full 64 hex of its argv.json", CK.BANK_SHA[1] == base[1][base[1].index("--dagger-bank-sha256") + 1] and CK.HEX64.match(CK.BANK_SHA[1]) is not None)
        check("DAG-OACC s0's bank sha256 in the checker is its argv.json's", CK.BANK_SHA[0] == base[0][base[0].index("--dagger-bank-sha256") + 1])
        check("DAG-OACC's argv has no detach and no w-traj (the recipe the checker asserts)", all("--detach-upper" not in b and "--w-traj" not in b for b in base.values()))
        for k in (0, 1):
            d = json.load(open(f"{DAGOACC}/M2b_s{k}_dagoacc/selfcheck.json", encoding="utf-8"))
            lvd = d["levers"]
            check(f"DAG-OACC s{k} selfcheck.json agrees with the checker's constants (params, steps, init, bank, dagger seed, w_acc, p_gt, data)",
                  lvd["params"] == CK.PARAMS and lvd["n_params_trainable"] == CK.N_PARAMS_TRAINABLE and d["steps_total"] == CK.RUN_STEPS and d["S_steps_per_epoch_100pct"] == CK.S_PER_EPOCH
                  and d["info"]["init_ckpt"]["sha256"] == CK.INIT_SHA[k] and lvd["dagger_bank_sha256"] == CK.BANK_SHA[k] and lvd["dagger"]["dagger_seed"] == k and lvd["w_acc"] == CK.W_ACC
                  and lvd["p_gt_start"] == CK.P_GT and lvd["dagger_p"] == CK.DAGGER_P and all(d["info"][x] == v for x, v in CK.DATA.items()))
            check(f"DAG-OACC s{k} selfcheck.json carries the live keys the new assertions read (dead_params list, goal and lon terms > 0)", isinstance(d["live"]["dead_params"], list) and d["live"]["terms"]["goal"] > 0 and d["live"]["terms"]["lon"] > 0)
        sums = open(f"{DAGOACC}/M2b_s0_dagoacc/SHA256SUMS.txt", encoding="utf-8").read()
        check("the pinned DAG-OACC s0 ckpt sha256 is the SHA256SUMS.txt line", CK.sums_sha(f"{DAGOACC}/M2b_s0_dagoacc/SHA256SUMS.txt") == CK.DAGOACC_S0_CKPT_SHA and CK.DAGOACC_S0_CKPT_SHA in sums)
    if os.path.isfile(OBD_ARGVS):
        obd = json.load(open(OBD_ARGVS, encoding="utf-8"))
        for t, o in (("BAL-NULL_s0", "OBD-NULL_s0"), ("BAL-NULL_rep", "OBD-NULL_rep"), ("BAL-NULL_s1", "OBD-NULL_s1")):
            a, b = CK.tokens(t), obd[o]["argv"]
            d = [i for i in range(len(a)) if a[i] != b[i]]
            check(f"{t} is {o} of the OBD package with only --out moved", len(a) == len(b) and d == [a.index("--out") + 1])
    good = CK.tokens("BAL-G_s0")
    i = good.index("--goal-pos-weight-stop") + 1
    bad = good[:i] + ["1.0"] + good[i + 1:]
    r = CK.audit_line("BAL-G_s0", " ".join(bad))
    check("a changed goal weight is refused and the first differing token is named", len(r) == 1 and f"token {i} is '1.0'" in r[0] and "'4.5616'" in r[0])
    check("a changed seed is refused", CK.audit_line("BAL-G_s0", " ".join(good).replace("--seed 0", "--seed 3")) != [])
    check("a half-rounded lon weight is refused", CK.audit_line("BAL-L_s0", " ".join(CK.tokens("BAL-L_s0")).replace("4.6991", "4.7")) != [])
    check("the lever dropped from G is refused (the arm would test nothing)", CK.audit_line("BAL-G_s0", " ".join(CK.tokens("BAL-G_s0")).replace(" --goal-pos-weight-stop 4.5616", "")) != [])
    check("a lever added to NULL is refused", CK.audit_line("BAL-NULL_s0", " ".join(CK.tokens("BAL-NULL_s0")) + " --goal-pos-weight-stop 4.5616") != [])
    check("the lon lever added to G is refused (the levers are separate arms)", CK.audit_line("BAL-G_s0", " ".join(CK.tokens("BAL-G_s0")) + " --lon-class-weight-stop 4.6991") != [])
    check("the G argv shipped for the L tag is refused", CK.audit_line("BAL-L_s1", " ".join(CK.tokens("BAL-G_s1"))) != [])
    check("a detach added is refused", CK.audit_line("BAL-NULL_s0", " ".join(CK.tokens("BAL-NULL_s0")) + " --detach-upper") != [])
    check("--w-acc changed is refused", CK.audit_line("BAL-G_s0", " ".join(CK.tokens("BAL-G_s0")).replace("--w-acc 49.6969", "--w-acc 0")) != [])
    check("--dagger-p changed is refused", CK.audit_line("BAL-G_s0", " ".join(CK.tokens("BAL-G_s0")).replace("--dagger-p 0.1", "--dagger-p 0")) != [])
    check("a truncated bank sha256 is refused", CK.audit_line("BAL-G_s1", " ".join(CK.tokens("BAL-G_s1")).replace(CK.BANK_SHA[1], CK.BANK_SHA[1][:60])) != [])
    check("--steps changed is refused", CK.audit_line("BAL-G_s0", " ".join(CK.tokens("BAL-G_s0")).replace("--steps 3000", "--steps 2000")) != [])
    check("--freeze added is refused (nothing is frozen)", CK.audit_line("BAL-G_s0", " ".join(CK.tokens("BAL-G_s0")) + " --freeze S") != [])
    check("an extra token is refused", CK.audit_line("BAL-L_s0", " ".join(CK.tokens("BAL-L_s0")) + " --steps 5") != [])
    check("a missing token is refused", CK.audit_line("BAL-L_s0", " ".join(CK.tokens("BAL-L_s0")[:-2])) != [])
    check("the s0 argv shipped for an s1 tag is refused", CK.audit_line("BAL-G_s1", " ".join(CK.tokens("BAL-G_s0"))) != [])
    check("NULL_rep is BAL-NULL_s0 apart from the seed and --out", [(a, b) for a, b in zip(CK.tokens("BAL-NULL_rep"), CK.tokens("BAL-NULL_s0")) if a != b] == [("2", "0"), (CK.out_decl("BAL-NULL_rep"), CK.out_decl("BAL-NULL_s0"))])
    check("the frozen argv file hashes to the pin", CK.sha_file(CK.ARGVS_FILE) == CK.ARGVS_SHA256)
    meta = os.path.join(PKG, "argvs_stopbal.meta.json")
    if os.path.exists(meta):
        mj = json.load(open(meta))
        check("the argv meta names the code of record, its trainer sha256 and the pinned argv file hash", mj["commit"] == COMMIT_REC and mj["argvs_json_sha256"] == CK.ARGVS_SHA256 and len(mj["trainer_sha256_at_commit"]) == 64)
    check("the CLI argv command prints the NULL arm's line", CK.main(["argv", "--tag", "BAL-NULL_s0", "--out-txt", os.path.join(tmp, "a.txt")]) == 0 and open(os.path.join(tmp, "a.txt")).read().strip() == " ".join(CK.tokens("BAL-NULL_s0")))
    # ---- a clean selfcheck passes, per run tag
    for t in runnable:
        toks, sc = sc_of(t)
        check(f"clean selfcheck passes {t}", CK.gate_problems(t, toks, sc) == [])
    # ---- planted-bad selfchecks: every assertion refuses, for every tag
    muts = {
        "declared_vs_built": lambda s: s.__setitem__("declared_vs_built", "FAIL"),
        "argv_sha": lambda s: s.__setitem__("argv_sha", "0" * 16),
        "steps_total": lambda s: s.__setitem__("steps_total", 12345),
        "S per epoch": lambda s: s.__setitem__("S_steps_per_epoch_100pct", 4000),
        "data train_sha256": lambda s: s["info"].__setitem__("train_sha256", "0" * 64),
        "data n_clips": lambda s: s["info"].__setitem__("n_clips", 4000),
        "data excluded_in_train": lambda s: s["info"].__setitem__("excluded_in_train", 2),
        "md5_overridden": lambda s: s["info"].__setitem__("md5_overridden", True),
        "batch": lambda s: s["info"].__setitem__("batch", 32),
        "init sha256": lambda s: s["info"]["init_ckpt"].__setitem__("sha256", "0" * 64),
        "init applied": lambda s: s["info"]["init_ckpt"].__setitem__("applied", False),
        "init mismatch": lambda s: s["info"]["init_ckpt"].__setitem__("n_mismatch_after_load", 3),
        "init source seed": lambda s: s["info"]["init_ckpt"].__setitem__("source_seed", 5),
        "init path": lambda s: s["info"]["init_ckpt"].__setitem__("path", r"D:\tanitad\runs\other\ckpt_last.pt"),
        "init record absent": lambda s: s["info"].pop("init_ckpt"),
        "new keys declared": lambda s: s["info"]["init_ckpt"].__setitem__("declared_new_keys", ["e_x.weight"]),
        "S frozen": lambda s: s["levers"]["params"].__setitem__("S", 0),
        "adapter frozen": lambda s: s["levers"]["params"].__setitem__("adapter", 0),
        "cond frozen": lambda s: s["levers"]["params"].__setitem__("cond", 0),
        "T frozen": lambda s: s["levers"]["params"].__setitem__("T", 0),
        "O frozen": lambda s: s["levers"]["params"].__setitem__("O", 0),
        "WM frozen": lambda s: s["levers"]["params"].__setitem__("WM", 0),
        "T count changed": lambda s: s["levers"]["params"].__setitem__("T", 1463869),
        "an extra group": lambda s: s["levers"]["params"].__setitem__("STOK", 9),
        "n_params_trainable": lambda s: s["levers"].__setitem__("n_params_trainable", 7214494),
        "params record absent": lambda s: s["levers"].pop("params"),
        "w_acc built": lambda s: s["levers"].__setitem__("w_acc", 0.0),
        "p_gt_start": lambda s: s["levers"].__setitem__("p_gt_start", 0.5),
        "dagger_p built 0.5": lambda s: s["levers"].__setitem__("dagger_p", 0.5),
        "dagger_p built 0": lambda s: s["levers"].__setitem__("dagger_p", 0.0),
        "dagger bank sha256 built": lambda s: s["levers"].__setitem__("dagger_bank_sha256", "0" * 64),
        "dagger bank sha256 declared": lambda s: s["levers"].__setitem__("dagger_bank_sha256_declared", "0" * 64),
        "dagger bank sha256 absent": lambda s: s["levers"].pop("dagger_bank_sha256"),
        "dagger.dagger_seed": lambda s: s["levers"]["dagger"].__setitem__("dagger_seed", 7),
        "dagger.dagger_p": lambda s: s["levers"]["dagger"].__setitem__("dagger_p", 0.2),
        "dagger.dagger_bank path": lambda s: s["levers"]["dagger"].__setitem__("dagger_bank", r"D:\tanitad\runs\v8f_mini_full\dagger_bank\s9\bank.json"),
        "dagger record absent": lambda s: s["levers"].pop("dagger"),
        "info.dagger bank sha256": lambda s: s["info"]["dagger"].__setitem__("bank_sha256", "0" * 64),
        "dagger coverage blocked": lambda s: s["info"]["dagger"]["coverage"].__setitem__("blocked", True),
        "dagger_probe absent": lambda s: s.pop("dagger_probe"),
        "dagger_probe pass": lambda s: s["dagger_probe"].__setitem__("pass", False),
        "dagger_probe no swapped row": lambda s: s["dagger_probe"].__setitem__("n_selected_at_p1", 0),
        "dagger_probe p0 control": lambda s: s["dagger_probe"].__setitem__("control_p0_loss_diff", 1e-9),
        "dagger_probe repeat control": lambda s: s["dagger_probe"].__setitem__("control_repeat_grad_diff", 1e-9),
        "dagger_probe prior control": lambda s: s["dagger_probe"].__setitem__("control_prior_grad_diff", 1e-9),
        "acc_probe absent": lambda s: s.pop("acc_probe"),
        "acc_probe pass": lambda s: s["acc_probe"].__setitem__("pass", False),
        "acc_probe w_acc_declared": lambda s: s["acc_probe"].__setitem__("w_acc_declared", 1.0),
        "acc_probe acc zero": lambda s: s["acc_probe"].__setitem__("acc", 0.0),
        "acc_probe weight does not reach the total": lambda s: s["acc_probe"].__setitem__("contribution", 0.0),
        "acc_probe head_o gradient": lambda s: s["acc_probe"].__setitem__("head_o_grad_diff", 0.0),
        "acc_probe repeat control": lambda s: s["acc_probe"].__setitem__("control_repeat_total_diff", 1e-9),
        "acc_probe prior control": lambda s: s["acc_probe"].__setitem__("control_prior_grad_diff", 1e-9),
        "ego_perturb on": lambda s: s["levers"]["ego_perturb"].__setitem__("ego_perturb_p", 0.3),
        "recover on": lambda s: s["levers"]["recover"].__setitem__("recover_p", 0.3),
        "recover absent": lambda s: s["levers"].pop("recover"),
        "ego+dagger composed": lambda s: s["levers"].__setitem__("ego_perturb_with_dagger", True),
        "launch intent on": lambda s: s["levers"].__setitem__("launch_intent", {"launch_intent_w": 0.5}),
        "wm_only built": lambda s: s["levers"].__setitem__("wm_only", {"wm_only": True}),
        "level head built": lambda s: s["levers"].__setitem__("level_head", "on"),
        "lead head built": lambda s: s["levers"].__setitem__("lead_head", "on"),
        "stok_dim built": lambda s: s["levers"].__setitem__("stok_dim", 8),
        "a band lever built (w_op)": lambda s: s["levers"].__setitem__("w_op", 0.3333333),
        "an otband probe present (a detach / band arm)": lambda s: s.__setitem__("otband_probe", {"otband": "detach-only", "detach_upper": True, "pass": True}),
        "kin_mode": lambda s: s["levers"].__setitem__("kin_mode", "steer"),
        # the STOPBAL levers, every tag: the BUILT value must equal the argv's and the card's
        "goal weight not built (reads 1.0 for its own lever / any other value for NULL)": lambda s: s["levers"].__setitem__("goal_pos_weight_stop", 1.0 if s["levers"]["goal_pos_weight_stop"] != 1.0 else 4.5616),
        "lon weight not built (reads 1.0 for its own lever / any other value for NULL)": lambda s: s["levers"].__setitem__("lon_class_weight_stop", 1.0 if s["levers"]["lon_class_weight_stop"] != 1.0 else 4.6991),
        "goal weight off by one digit": lambda s: s["levers"].__setitem__("goal_pos_weight_stop", s["levers"]["goal_pos_weight_stop"] + 0.0001),
        "lon weight off by one digit": lambda s: s["levers"].__setitem__("lon_class_weight_stop", s["levers"]["lon_class_weight_stop"] + 0.0001),
        "goal weight absent from the levers": lambda s: s["levers"].pop("goal_pos_weight_stop"),
        "lon weight absent from the levers": lambda s: s["levers"].pop("lon_class_weight_stop"),
        "goal weight a string": lambda s: s["levers"].__setitem__("goal_pos_weight_stop", str(s["levers"]["goal_pos_weight_stop"])),
        "goal weight True (== 1)": lambda s: s["levers"].__setitem__("goal_pos_weight_stop", True),
        "both weights built (a stacked arm)": lambda s: s["levers"].update(goal_pos_weight_stop=4.5616, lon_class_weight_stop=4.6991),
        # G-LIVE
        "live pass": lambda s: s["live"].__setitem__("pass", False),
        "live missing terms": lambda s: s["live"].__setitem__("missing_terms", ["tac"]),
        "live unexpected dead": lambda s: s["live"].__setitem__("unexpected_dead", ["e_code.weight"]),
        "live control": lambda s: s["live"].__setitem__("control_prior_grad_sum", 1e-9),
        "live head_o": lambda s: s["live"].__setitem__("head_o_grad_step0", 0.0),
        "live non-finite term": lambda s: s["live"]["terms"].__setitem__("ade", float("nan")),
        "live ade term absent": lambda s: s["live"]["terms"].pop("ade"),
        "live acc term zero": lambda s: s["live"]["terms"].__setitem__("acc", 0.0),
        "live goal term absent": lambda s: s["live"]["terms"].pop("goal"),
        "live goal term not finite": lambda s: s["live"]["terms"].__setitem__("goal", float("inf")),
        "live goal term zero": lambda s: s["live"]["terms"].__setitem__("goal", 0.0),
        "live lon term absent": lambda s: s["live"]["terms"].pop("lon"),
        "live lon term nan": lambda s: s["live"]["terms"].__setitem__("lon", float("nan")),
        "live lon term zero": lambda s: s["live"]["terms"].__setitem__("lon", 0.0),
        "T's goal head gets no gradient (dead_params only)": lambda s: s["live"].__setitem__("dead_params", ["h_goal.0.weight"]),
        "T's lon head gets no gradient (dead_params only)": lambda s: s["live"].__setitem__("dead_params", ["h_lon.2.bias"]),
        "T's goal head gets no gradient (dead + unexpected_dead)": lambda s: s["live"].update(dead_params=["h_goal.0.weight"], unexpected_dead=["h_goal.0.weight"]),
        "dead_params absent": lambda s: s["live"].pop("dead_params"),
        "dead_params not a list": lambda s: s["live"].__setitem__("dead_params", "none"),
    }
    for g in ("adapter", "S", "T", "O", "head_s", "head_o", "wm"):
        muts[f"live group {g} has no gradient (a frozen group)"] = (lambda g_: lambda s: s["live"]["groups_grad"].__setitem__(g_, 0.0))(g)
    for t in runnable:
        toks, sc = sc_of(t)
        for name, f in muts.items():
            s2 = copy.deepcopy(sc)
            try:
                f(s2)
            except KeyError:
                continue
            check(f"{t}: planted '{name}' is refused", len(CK.gate_problems(t, toks, s2)) >= 1)
    # per-kind plants that only make sense for one kind
    for t in runnable:
        toks, sc = sc_of(t)
        kind = CK.TABLE[CK.base_tag(t)]["kind"]
        if kind != "null":
            s2 = copy.deepcopy(sc)
            s2["levers"].update(goal_pos_weight_stop=1.0, lon_class_weight_stop=1.0)
            check(f"{t}: both weights read back 1.0 (the lever was not built) is refused", len(CK.gate_problems(t, toks, s2)) >= 1)
        other = {"g": "l", "l": "g", "null": "g"}[kind]
        s3 = copy.deepcopy(sc)
        s3["levers"].update(goal_pos_weight_stop=CK.WEIGHTS[other][0], lon_class_weight_stop=CK.WEIGHTS[other][1])
        check(f"{t}: the other kind's weights built is refused", len(CK.gate_problems(t, toks, s3)) >= 1)
    # the dead_params-only plants must be refused by the NEW assertion alone (unexpected_dead stays empty), and say so
    for t in ("BAL-G_s0", "BAL-L_s1"):
        toks, sc = sc_of(t)
        sc["live"]["dead_params"] = ["h_lon.0.weight"]
        pr = CK.gate_problems(t, toks, sc)
        check(f"{t}: a dead T lon-head parameter is named by the goal / lon head assertion", any("goal / lon head" in x and "h_lon.0.weight" in x for x in pr))
        toks, sc = sc_of(t)
        sc["live"]["dead_params"] = ["wm.idm_heads.0.weight"]
        check(f"{t}: a dead parameter outside T's goal / lon heads is not this assertion's business (clean when unexpected_dead is empty)", CK.gate_problems(t, toks, sc) == [])
    # the lever-read-back stop line names the lever, the value read and the one wanted
    toks, sc = sc_of("BAL-G_s0")
    sc["levers"]["goal_pos_weight_stop"] = 1.0
    pr = CK.gate_problems("BAL-G_s0", toks, sc)
    check("BAL-G_s0 reading 1.0 for its own lever: the refusal names the lever, the read value and the wanted one", any("goal_pos_weight_stop" in x and "1.0" in x and "4.5616" in x for x in pr))
    # the pinned argv is itself read: a changed argv handed to the gate is refused even when the selfcheck is clean
    for t, mut in (("BAL-G_s0", lambda a: [("3" if x == "0" and a[i - 1] == "--seed" else x) for i, x in enumerate(a)]),
                   ("BAL-L_s1", lambda a: [x for x in a if x not in ("--lon-class-weight-stop", "4.6991")]), ("BAL-NULL_s0", lambda a: a + ["--detach-upper"]),
                   ("BAL-NULL_s1", lambda a: a + ["--goal-pos-weight-stop", "4.5616"]), ("BAL-G_s1", lambda a: [("4.5617" if x == "4.5616" else x) for x in a])):
        toks, sc = sc_of(t)
        check(f"{t}: a clean selfcheck does not rescue a changed pinned argv", len(CK.gate_problems(t, mut(list(toks)), sc)) >= 1)
    # ---- the needles against the real commits (source read; skipped when the mirror is absent)
    check("the trainer needles name both new flags", '"--goal-pos-weight-stop"' in CK.NEEDLES[CK.TRAINER_REL] and '"--lon-class-weight-stop"' in CK.NEEDLES[CK.TRAINER_REL])
    if os.path.isdir(PUSH):
        def show(commit, path):
            r = subprocess.run(["git", "-c", "safe.directory=*", "-C", PUSH, "show", f"{commit}:{path}"], capture_output=True)
            return r.stdout.decode("utf-8", "replace") if r.returncode == 0 else None
        for rel, ns in CK.NEEDLES.items():
            src = show(COMMIT_REC, rel)
            if src is None:
                check(f"{rel} is readable at {COMMIT_REC[:8]}", False)
                continue
            check(f"at {COMMIT_REC[:8]} {rel} carries every needle", [n for n in ns if n not in src] == [])
        src_p = show(COMMIT_PARENT, CK.TRAINER_REL) or ""
        check("at the parent 5514a684 the trainer LACKS both new flags (a parent shipped by mistake is refused)", '"--goal-pos-weight-stop"' not in src_p and '"--lon-class-weight-stop"' not in src_p and src_p != "")
        mod_p = show(COMMIT_PARENT, CK.MODEL_REL) or ""
        check("at the parent the model lacks the two loss lines", "if cfg.goal_pos_weight_stop != 1.0" not in mod_p and mod_p != "")
    # ---- moved identity (pure)
    a = PT.model_state(0)
    b = {k: v.copy() for k, v in a.items()}
    for k in b:
        if k.startswith(("read_s", "head_s", "read_t", "trunk_t", "h_goal", "head_o", "read_o")):
            b[k] = b[k] + 1.0
    ev, pr = CK.moved_identity(a, b)
    check("moved identity: S, T and O moved -> clean", pr == [] and all(ev[g]["n_moved"] >= 1 for g in "STO"))
    for g, pre in CK.MOVED_PREFIXES.items():
        b2 = {k: (a[k].copy() if k.startswith(pre) else v.copy()) for k, v in b.items()}
        check(f"moved identity: group {g} not moved is refused", CK.moved_identity(a, b2)[1] != [])
    b3 = dict(b)
    b3.pop("head_s.w")
    check("moved identity: a missing S tensor is refused", CK.moved_identity(a, b3)[1] != [])
    check("moved identity: nothing trained is refused", CK.moved_identity(a, {k: v.copy() for k, v in a.items()})[1] != [])
    check("moved identity: an init without S / T / O keys is refused", CK.moved_identity({"x": np.zeros(2)}, {"x": np.zeros(2)})[1] != [])
    # ---- token binding, post check and the cleanup guard on a fake box folder
    root = os.path.join(tmp, "box", "D", "tanitad")
    os.environ.update(STOPBAL_ROOT=root, STOPBAL_COMMIT="a" * 40, STOPBAL_CKPT_LOADER="pickle", STOPBAL_FLOOR_BYTES="1000")
    for rel in ("cache_full/native_tracks.pt", "data/v9/lead_train4572.npz", "data/v9/refcv6_speed_max_v8_train.jsonl", "runs/v8f_mini_full/join_train.npz", "runs/v8f_mini_full/p10/p10_prior.pt"):
        put(os.path.join(root, *rel.split("/")), rel.encode())
    put(os.path.join(root, "data", "v9", "v9_labels_train.npz"), b"v9 labels")
    os.environ["STOPBAL_FAKE_V9_MD5"] = hashlib.md5(b"v9 labels").hexdigest()
    shas, bshas = {}, {}
    for s in (0, 1):
        p = os.path.join(root, "runs", "v8f_mini_full", f"M2b_s{s}", "ckpt_last.pt")
        put(p, pickle.dumps({"model": PT.model_state(s)}))
        shas[str(s)] = hashlib.sha256(open(p, "rb").read()).hexdigest()
        bp = os.path.join(root, "runs", "v8f_mini_full", "dagger_bank", f"s{s}", "bank.json")
        put(bp, f"bank {s}".encode())
        bshas[str(s)] = hashlib.sha256(open(bp, "rb").read()).hexdigest()
    json.dump(shas, open(os.path.join(tmp, "init_sha.json"), "w"))
    json.dump(bshas, open(os.path.join(tmp, "bank_sha.json"), "w"))
    put(os.path.join(root, "stopbal", "repo", "stack", "scripts", "v8f_mini_train.py"), b"trainer v1")
    put(os.path.join(root, "stopbal", "tree_manifest.json"), b"{}")
    tag = "BAL-G_s0"
    line = " ".join(CK.tokens(tag))
    b_, bad_ = CK.binding(tag, line)
    check("binding reads the six data files and the DAgger bank", bad_ == [] and len(b_["data_md5"]) == 6 and b_["dagger_bank_sha256"] == bshas["0"])
    check("bank_problems is clean for both seeds", CK.bank_problems(0) == [] and CK.bank_problems(1) == [])
    sc_p = os.path.join(CK.run_dir(tag), "gate", "selfcheck.json")
    put(sc_p, b"{}")
    tok = {"schema": "stopbal-gate-token-v1", "written_by": "stopbal_check.py gate", "selfcheck_sha256": CK.sha_file(sc_p), **b_}
    put(CK.token_path(tag), json.dumps(tok).encode())
    check("a token bound to the current state is valid", CK.token_ok(tag, line) == [])
    os.environ["STOPBAL_COMMIT"] = "b" * 40
    check("a token for another commit is refused", CK.token_ok(tag, line) != [])
    os.environ["STOPBAL_COMMIT"] = "a" * 40
    check("a token for another argv is refused", CK.token_ok(tag, line + " --x") != [])
    check("a token for the same argv with the lever weight changed is refused", CK.token_ok(tag, line.replace("4.5616", "4.5617")) != [])
    put(os.path.join(root, "stopbal", "repo", "stack", "scripts", "v8f_mini_train.py"), b"trainer v2")
    check("a token after the trainer bytes changed is refused", CK.token_ok(tag, line) != [])
    put(os.path.join(root, "stopbal", "repo", "stack", "scripts", "v8f_mini_train.py"), b"trainer v1")
    put(os.path.join(root, "data", "v9", "lead_train4572.npz"), b"changed")
    check("a token after a data file changed is refused", CK.token_ok(tag, line) != [])
    put(os.path.join(root, "data", "v9", "lead_train4572.npz"), b"data/v9/lead_train4572.npz")
    bp0 = os.path.join(root, "runs", "v8f_mini_full", "dagger_bank", "s0", "bank.json")
    put(bp0, b"another bank")
    check("a token after the DAgger bank changed is refused (and the bank problem is named)", CK.token_ok(tag, line) != [] and CK.bank_problems(0) != [])
    put(bp0, b"bank 0")
    os.remove(bp0)
    check("a missing DAgger bank is refused", CK.bank_problems(0) != [])
    put(bp0, b"bank 0")
    put(os.path.join(root, "stopbal", "tree_manifest.json"), b'{"x": 1}')
    check("a token after the shipped tree manifest changed is refused", CK.token_ok(tag, line) != [])
    put(os.path.join(root, "stopbal", "tree_manifest.json"), b"{}")
    put(sc_p, b'{"x":1}')
    check("a token after selfcheck.json changed is refused", CK.token_ok(tag, line) != [])
    put(sc_p, b"{}")
    put(CK.token_path(tag), json.dumps(dict(tok, written_by="hand")).encode())
    check("a hand-written token is refused", CK.token_ok(tag, line) != [])
    put(CK.token_path(tag), json.dumps(tok).encode())
    arg = os.path.join(tmp, "arg.txt")
    open(arg, "w").write(line + "\n")
    tp = os.path.join(tmp, "tok.json")
    put(tp, json.dumps(dict(tok, tag=tag, argv_sha256=CK.tokens_sha(line.split(" ")), commit="a" * 40)).encode())
    vt = lambda tg, cm, f=arg: CK.cmd_verify_token({"token": tp, "argv-file": f, "tag": tg, "commit": cm})      # noqa: E731
    check("verify-token accepts the matching token", vt(tag, "a" * 40) == 0)
    check("verify-token refuses a wrong tag", vt("BAL-L_s0", "a" * 40) == 1)
    check("verify-token refuses a wrong commit", vt(tag, "c" * 40) == 1)
    open(os.path.join(tmp, "arg2.txt"), "w").write(line.replace("--seed 0", "--seed 9") + "\n")
    check("verify-token refuses an argv that is not the frozen one", vt(tag, "a" * 40, os.path.join(tmp, "arg2.txt")) == 1)

    # post check, per tag, on an out dir written by the stand-in's own format
    def post_case(tg):
        toks = CK.tokens(tg)
        spec = CK.TABLE[tg]
        kind, init, seed = spec["kind"], spec["init"], spec["seed"]
        wg, wl = CK.WEIGHTS[kind]
        out = CK.loc(CK.out_decl(tg))
        st0 = PT.model_state(init)
        moved = {k: (v + 1.0 if k.startswith(("read_s", "head_s", "read_t", "trunk_t", "h_goal", "head_o", "read_o")) else v) for k, v in st0.items()}
        ck_ok = {"model": moved, "step": CK.RUN_STEPS, "seed": seed, "arm": "M2b", "argv_sha": CK.argv_sha16(toks),
                 "cfg": {"detach_upper": False, "w_traj": 1.0, "w_acc": CK.W_ACC, "goal_pos_weight_stop": wg, "lon_class_weight_stop": wl},
                 "dagger": {"dagger_p": 0.1, "dagger_seed": init, "dagger_bank_sha256": CK.bank_sha(init)}, "init_ckpt": {"sha256": shas[str(init)]}}
        summ = {"done": True, "steps": CK.RUN_STEPS, "argv_sha": CK.argv_sha16(toks), "seed": seed}

        def write_out(ck, sm):
            put(os.path.join(out, "ckpt_last.pt"), pickle.dumps(ck))
            put(os.path.join(out, "summary.json"), json.dumps(sm).encode())
        write_out(ck_ok, summ)
        check(f"post check {tg}: a correct finished run passes", CK.post_problems(tg, toks, out)[1] == [])
        cfg_with = lambda **kw: dict(ck_ok, cfg=dict(ck_ok["cfg"], **kw))             # noqa: E731
        cases = [("steps short", ck_ok, dict(summ, steps=100)), ("not done", ck_ok, dict(summ, done=False)), ("summary argv_sha", ck_ok, dict(summ, argv_sha="0" * 16)),
                 ("summary seed", ck_ok, dict(summ, seed=seed + 5)), ("ckpt step", dict(ck_ok, step=5), summ), ("ckpt seed", dict(ck_ok, seed=seed + 5), summ), ("ckpt arm", dict(ck_ok, arm="other"), summ),
                 ("ckpt argv_sha", dict(ck_ok, argv_sha="0" * 16), summ),
                 ("cfg detach_upper on", cfg_with(detach_upper=True), summ),
                 ("cfg w_traj", cfg_with(w_traj=0.0), summ), ("cfg w_acc", cfg_with(w_acc=0.0), summ),
                 ("cfg goal weight changed", cfg_with(goal_pos_weight_stop=wg + 1.0), summ), ("cfg lon weight changed", cfg_with(lon_class_weight_stop=wl + 1.0), summ),
                 ("cfg both weights at the other kind's values", cfg_with(goal_pos_weight_stop=4.5616 if wg == 1.0 else 1.0, lon_class_weight_stop=4.6991 if wl == 1.0 else 1.0), summ),
                 ("cfg weights absent", dict(ck_ok, cfg={k: v for k, v in ck_ok["cfg"].items() if k not in ("goal_pos_weight_stop", "lon_class_weight_stop")}), summ),
                 ("dagger p", dict(ck_ok, dagger=dict(ck_ok["dagger"], dagger_p=0.0)), summ), ("dagger seed", dict(ck_ok, dagger=dict(ck_ok["dagger"], dagger_seed=7)), summ),
                 ("dagger bank sha256", dict(ck_ok, dagger=dict(ck_ok["dagger"], dagger_bank_sha256="0" * 64)), summ), ("dagger record absent", {k: v for k, v in ck_ok.items() if k != "dagger"}, summ),
                 ("init sha256", dict(ck_ok, init_ckpt={"sha256": "0" * 64}), summ),
                 ("S not moved", dict(ck_ok, model={**moved, "head_s.w": st0["head_s.w"], "read_s.w": st0["read_s.w"]}), summ),
                 ("T not moved", dict(ck_ok, model={**moved, "read_t.w": st0["read_t.w"], "trunk_t.w": st0["trunk_t.w"], "h_goal.w": st0["h_goal.w"]}), summ),
                 ("O not moved", dict(ck_ok, model={**moved, "read_o.w": st0["read_o.w"], "head_o.w": st0["head_o.w"]}), summ), ("nothing trained", dict(ck_ok, model=st0), summ)]
        for name, ck_m, sm_m in cases:
            write_out(ck_m, sm_m)
            check(f"post check {tg}: planted '{name}' is refused", CK.post_problems(tg, toks, out)[1] != [])
        os.remove(os.path.join(out, "ckpt_last.pt"))
        check(f"post check {tg}: no checkpoint is refused", CK.post_problems(tg, toks, out)[1] != [])
        return out
    for tg in CK.RUN_ORDER:
        out = post_case(tg)
    # cleanup guard
    tag = "BAL-L_s1"
    gout = CK.loc(CK.gate_out_decl(tag))
    put(os.path.join(gout, "selfcheck.json"), b"x")
    put(os.path.join(out, "ckpt_last.pt"), b"ckpt")
    check("cleanup without bank proof refuses and deletes nothing", CK.cmd_cleanup(tag, "G1") == 6 and os.path.exists(gout))
    put(os.path.join(CK.run_dir(tag), "BANK_VERIFIED_G1"), b"ok")
    check("cleanup with bank proof deletes the gate dir, logs it, and keeps the checkpoint", CK.cmd_cleanup(tag, "G1") == 0 and not os.path.exists(gout) and os.path.exists(os.path.join(out, "ckpt_last.pt"))
          and json.load(open(os.path.join(CK.run_dir(tag), "CLEANUP.json")))["deleted_own_temp"][0]["path"] == gout)
    check("rm_own_temp refuses a dir whose parent is not the named one", CK.rm_own_temp(out, "gate_out") is None and os.path.isdir(out))
    # ---- pilot timing and the 12 h rule
    pl = os.path.join(tmp, "pilot")
    os.makedirs(pl, exist_ok=True)
    with open(os.path.join(pl, "train_log.jsonl"), "w") as f:
        for s in (50, 100, 150, 200):
            f.write(json.dumps({"step": s, "sec": 30 + 2.0 * s}) + "\n")
    tm = CK.time_pilot(os.path.join(pl, "train_log.jsonl"))
    check("pilot timing excludes start-up (2.0 s/step)", abs(tm["mean_s_per_step"] - 2.0) < 1e-9)
    open(os.path.join(pl, "short.jsonl"), "w").write(json.dumps({"step": 50, "sec": 1.0}) + "\n" + json.dumps({"step": 200, "sec": 3.0}) + "\n")
    try:
        CK.time_pilot(os.path.join(pl, "short.jsonl"))
        check("a pilot log without 4 records is refused", False)
    except ValueError:
        check("a pilot log without 4 records is refused", True)
    pr1 = CK.project(1.0)
    check("1.0 s/step: 7 runs x 3,000 steps = 5.83 h", abs(pr1["projected_total_h_planned"] - 21000 / 3600) < 1e-9 and pr1["within_limit_planned"] and pr1["n_runs_planned"] == 7 and "projected_total_h_with_null_s1" not in pr1)
    check("2.0 s/step is 11.67 h (within 12 h); 2.1 s/step is 12.25 h (over)", CK.project(2.0)["within_limit_planned"] and not CK.project(2.1)["within_limit_planned"])
    check("the card's measured 0.759 s/step (2,276 s / 3,000 steps) projects 4.4 h", abs(CK.project(2276 / 3000)["projected_total_h_planned"] - 4.4) < 0.05)

    def pilot_json(mean, commit="a" * 40):
        pj = {"commit": commit, "projection": CK.project(mean)}
        p = os.path.join(tmp, f"pj_{mean}_{commit[:1]}.json")
        open(p, "w").write(json.dumps(pj))
        return p
    check("permit: within 12 h passes", CK.cmd_permit({"pilot": pilot_json(1.0), "commit": "a" * 40}) == 0)
    p6 = pilot_json(3.0)
    check("permit: over 12 h without an amendment is refused (3)", CK.cmd_permit({"pilot": p6, "commit": "a" * 40}) == 3)
    check("permit: a pilot of another commit is refused", CK.cmd_permit({"pilot": pilot_json(1.0, "b" * 40), "commit": "a" * 40}) == 3)
    am = os.path.join(tmp, "am.txt")
    open(am, "w").write(f"pilot_sha256={'0' * 64} commit={'a' * 40}\n")
    check("permit: an amendment naming another pilot is refused", CK.cmd_permit({"pilot": p6, "commit": "a" * 40, "amendment": am}) == 3)
    open(am, "w").write(f"pilot_sha256={CK.sha_file(p6)} commit={'a' * 40}\n")
    check("permit: an amendment binding this pilot and commit is accepted", CK.cmd_permit({"pilot": p6, "commit": "a" * 40, "amendment": am}) == 0)
    # ---- parity of BAL-NULL_s0 against DAG-OACC s0 (dev box): computed and reported, never a stop
    cks = os.path.join(tmp, "par")
    os.makedirs(cks, exist_ok=True)
    st = PT.model_state(0)
    mine = os.path.join(cks, "mine.pt")
    ref = os.path.join(cks, "ref.pt")
    pickle.dump({"model": st, "argv_sha": "aaaa", "step": 3000}, open(mine, "wb"))
    pickle.dump({"model": st, "argv_sha": "bbbb", "step": 3000}, open(ref, "wb"))            # same weights, other metadata: the whole-file sha256 cannot match (the checkpoint embeds argv_sha, which holds --out)
    sums = os.path.join(cks, "SHA256SUMS.txt")
    open(sums, "w").write(f"{CK.sha_file(ref)} *ckpt_last.pt\n{'1' * 64} *summary.json\n")
    os.environ["STOPBAL_TEST_DAGOACC_SHA"] = CK.sha_file(ref)
    outp = os.path.join(cks, "o.json")
    rd = lambda: json.load(open(outp))      # noqa: E731
    check("sums_sha reads the ckpt_last.pt line", CK.sums_sha(sums) == CK.sha_file(ref))
    check("parity (tensors, the default): the same weights -> EQUAL although the whole-file sha256 differs", CK.cmd_parity({"bank-ckpt": mine, "sums": sums, "out": outp, "dagoacc-ckpt": ref}) == 0
          and rd()["null_parity"] == "EQUAL" and rd()["tensor_equal"] is True and rd()["file_sha_equal"] is False and rd()["basis"] == "tensors" and rd()["status"] == "computed" and "append_stopbal_null_s1" not in rd())
    check("parity (file basis): other metadata -> DIFFERS, reported, rc 0", CK.cmd_parity({"bank-ckpt": mine, "sums": sums, "out": outp, "dagoacc-ckpt": ref, "basis": "file"}) == 0 and rd()["null_parity"] == "DIFFERS")
    same = os.path.join(cks, "same.pt")
    open(same, "wb").write(open(ref, "rb").read())
    check("parity (file basis): an identical file -> EQUAL", CK.cmd_parity({"bank-ckpt": same, "sums": sums, "out": outp, "basis": "file"}) == 0 and rd()["null_parity"] == "EQUAL" and rd()["file_sha_equal"] is True)
    st2 = dict(st)
    st2["head_o.w"] = st["head_o.w"] + 1e-7
    other = os.path.join(cks, "other.pt")
    pickle.dump({"model": st2, "argv_sha": "bbbb", "step": 3000}, open(other, "wb"))
    check("parity (tensors): one changed tensor -> DIFFERS, one tensor named, rc 0 (a difference is a report, not a stop)", CK.cmd_parity({"bank-ckpt": other, "sums": sums, "out": outp, "dagoacc-ckpt": ref}) == 0 and rd()["null_parity"] == "DIFFERS" and rd()["tensor_detail"]["n_tensors_differ"] == 1)
    check("parity: tensors without the DAG-OACC checkpoint -> not-computed (rc 2), the file says so and nothing is written as EQUAL", CK.cmd_parity({"bank-ckpt": mine, "sums": sums, "out": outp}) == 2 and rd()["status"] == "not-computed" and rd()["null_parity"] == "UNKNOWN")
    check("parity: a basis other than file / tensors is refused (2)", CK.cmd_parity({"bank-ckpt": mine, "sums": sums, "out": outp, "basis": "x"}) == 2)
    check("parity: a missing bank checkpoint -> not-computed (2)", CK.cmd_parity({"bank-ckpt": os.path.join(cks, "none.pt"), "sums": sums, "out": outp, "dagoacc-ckpt": ref}) == 2 and rd()["status"] == "not-computed")
    check("parity: a SHA256SUMS without a ckpt_last.pt line -> not-computed (2)", CK.cmd_parity({"bank-ckpt": mine, "sums": os.path.join(cks, "empty.txt"), "out": outp}) == 2 and rd()["status"] == "not-computed")
    open(os.path.join(cks, "sums2.txt"), "w").write(f"{'2' * 64} *ckpt_last.pt\n")
    check("parity: a SHA256SUMS whose value is not the pinned DAG-OACC one -> not-computed (2): the reference moved", CK.cmd_parity({"bank-ckpt": mine, "sums": os.path.join(cks, "sums2.txt"), "out": outp, "dagoacc-ckpt": ref}) == 2 and rd()["status"] == "not-computed")
    check("parity: a DAG-OACC checkpoint that does not hash to the SUMS value -> not-computed (2)", CK.cmd_parity({"bank-ckpt": mine, "sums": sums, "out": outp, "dagoacc-ckpt": other}) == 2 and rd()["status"] == "not-computed")
    corrupt = os.path.join(cks, "corrupt.pt")
    open(corrupt, "wb").write(b"not a pickle")
    os.environ["STOPBAL_TEST_DAGOACC_SHA"] = CK.sha_file(corrupt)
    open(os.path.join(cks, "sums3.txt"), "w").write(f"{CK.sha_file(corrupt)} *ckpt_last.pt\n")
    os.environ["STOPBAL_CKPT_LOADER"] = "pickle"
    check("parity: an unloadable checkpoint is not-computed (2), never an exception", CK.cmd_parity({"bank-ckpt": mine, "sums": os.path.join(cks, "sums3.txt"), "out": outp, "dagoacc-ckpt": corrupt}) == 2 and rd()["status"] == "not-computed")
    os.environ.pop("STOPBAL_TEST_DAGOACC_SHA")
    # ---- hygiene of the package
    bad_files = []
    for dp, _, fs in os.walk(PKG):
        for f in fs:
            p = os.path.join(dp, f)
            if f.endswith((".pyc", ".out", ".log")) or "__pycache__" in dp:
                continue
            raw = open(p, "rb").read()
            if f.endswith(".sh") and b"\r" in raw:
                bad_files.append(f"CRLF in {f}")
            if re.search(rb"\btext\s*=\s*True", raw) and not f.startswith("test_stopbal_check"):
                bad_files.append(f"bare text=True in {f}")
            if re.search(rb"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", raw) and not f.endswith(".json"):
                bad_files.append(f"IP-like string in {f}")
            if re.match(r"(?i)^(aux|con|prn|nul)(\.|$)", f):
                bad_files.append(f"reserved name {f}")
            if f in ("stopbal_check.py", "stopbal_chain.sh", "stopbal_run.sh", "stopbal_ship.sh", "stopbal_common.sh") and re.search(rb"BAL-XX|OBD-(NULL|CTL|BAND)", raw):
                bad_files.append(f"a leftover OBD tag in {f}")
    check("package hygiene (LF scripts, no bare text=True, no IP, no reserved names, no leftover OBD tags): " + "; ".join(bad_files), bad_files == [])
    print(f"UNIT ok={N_OK} bad={N_BAD}")
    return 0 if N_BAD == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
