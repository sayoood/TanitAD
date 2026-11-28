"""Unit tests of obd_check.py (pure Python; numpy only for the moved identity; no torch, no GPU, no network).
Run:  C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I tests/test_obd_check.py <scratch dir>     -> prints  UNIT ok=<n> bad=<m>  (exit 1 on any bad)
Every gate assertion has a planted-bad selfcheck that must be refused; every clean selfcheck must pass."""
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
import obd_check as CK  # noqa: E402
import plant_trainer as PT  # noqa: E402

N_OK, N_BAD = 0, 0
DAGOACC = "E:/Projects/TanitAD-artifacts/v8f-dagoacc"
PUSH = "C:/Users/Admin/tanitad-push"
COMMIT_REC = "202849da013b5d74c0b0086dd9865fcda6836018"


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
    os.environ["OBD_FAKE_INIT_SHA_FILE"] = os.path.join(TMP, "init_sha.json")
    os.environ["OBD_FAKE_BANK_SHA_FILE"] = os.path.join(TMP, "bank_sha.json")
    return toks, PT.good_selfcheck(gt)


def put(p, data=b""):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "wb").write(data)


def diff_idx(a, b):
    return [i for i, (x, y) in enumerate(zip(a, b)) if x != y]


def main(tmp):
    global TMP
    TMP = tmp
    os.makedirs(tmp, exist_ok=True)
    json.dump({"0": CK.INIT_SHA[0], "1": CK.INIT_SHA[1]}, open(os.path.join(tmp, "init_sha.json"), "w"))
    json.dump({"0": CK.BANK_SHA[0], "1": CK.BANK_SHA[1]}, open(os.path.join(tmp, "bank_sha.json"), "w"))
    runnable = CK.RUN_ORDER + [CK.PILOT_TAG, CK.CONDITIONAL_TAG]
    # ---- the card's order and the table
    check("run order is NULL_s0, NULL_rep, CTL_s0, BAND_s0, CTL_s1, BAND_s1, BAND_rep", CK.RUN_ORDER == ["OBD-NULL_s0", "OBD-NULL_rep", "OBD-CTL_s0", "OBD-BAND_s0", "OBD-CTL_s1", "OBD-BAND_s1", "OBD-BAND_rep"])
    check("OBD-NULL_s1 is in the table, conditional, and not in the planned order", CK.TABLE["OBD-NULL_s1"].get("conditional") and "OBD-NULL_s1" not in CK.RUN_ORDER and CK.CONDITIONAL_TAG == "OBD-NULL_s1")
    check("steps: 3,000 per run, pilot 200, limit 12 h", CK.RUN_STEPS == 3000 and CK.PILOT_STEPS == 200 and CK.LIMIT_H == 12.0)
    check("the parity run is OBD-NULL_s0 and the pilot is OBD-BAND_s0's", CK.PARITY_TAG == "OBD-NULL_s0" and CK.PILOT_BASE == "OBD-BAND_s0" and set(CK.GATE_AT_PILOT) == {"OBD-BAND_s0", "OBD-CTL_s0", "OBD-NULL_s0"})
    for t, v in CK.TABLE.items():
        a = CK.tokens(t)
        check(f"{t}: table seed / init equal the pinned argv's --seed / init-ckpt", CK.flag_value(a, "--seed") == str(v["seed"]) and f"M2b_s{v['init']}" in CK.flag_value(a, "--init-ckpt") and CK.flag_value(a, "--dagger-seed") == str(v["init"]))
    # ---- argv pin and composition (each arm = DAG-OACC's argv + only the declared changes)
    for t in CK.TABLE:
        check(f"frozen argv of {t} is accepted", CK.audit_line(t, " ".join(CK.tokens(t))) == [])
    check("pilot argv = OBD-BAND_s0 argv with --out moved and --steps 200 appended", CK.tokens(CK.PILOT_TAG)[:-2] == [CK.out_decl(CK.PILOT_TAG) if x == CK.out_decl("OBD-BAND_s0") else x for x in CK.tokens("OBD-BAND_s0")] and CK.tokens(CK.PILOT_TAG)[-2:] == ["--steps", "200"])
    check("pilot out dir differs from the run out dir", CK.out_decl(CK.PILOT_TAG) != CK.out_decl("OBD-BAND_s0"))
    for t in CK.TABLE:
        check(f"{t}: --out is D:\\tanitad\\runs\\v8f_obd\\<tag>", CK.out_decl(t) == "D:\\tanitad\\runs\\v8f_obd\\" + t)
    if os.path.isfile(f"{DAGOACC}/M2b_s0_dagoacc/argv.json"):
        base = {k: json.load(open(f"{DAGOACC}/M2b_s{k}_dagoacc/argv.json", encoding="utf-8"))["argv"] for k in (0, 1)}
        decl = {"OBD-NULL_s0": (0, [], None), "OBD-NULL_rep": (0, [], "2"), "OBD-CTL_s0": (0, ["--detach-upper"], None), "OBD-CTL_s1": (1, ["--detach-upper"], None),
                "OBD-BAND_s0": (0, ["--detach-upper", "--w-traj", "0", "--w-op", "0.3333333", "--w-tac-traj", "0.6666667"], None),
                "OBD-BAND_s1": (1, ["--detach-upper", "--w-traj", "0", "--w-op", "0.3333333", "--w-tac-traj", "0.6666667"], None),
                "OBD-BAND_rep": (0, ["--detach-upper", "--w-traj", "0", "--w-op", "0.3333333", "--w-tac-traj", "0.6666667"], "2"), "OBD-NULL_s1": (1, [], None)}
        for t, (k, extra, sd) in decl.items():
            want = list(base[k])
            want[want.index("--out") + 1] = CK.out_decl(t)
            if sd:
                want[want.index("--seed") + 1] = sd
            check(f"{t}: equals DAG-OACC s{k}'s argv.json with only --out{', --seed ' + sd if sd else ''}{' and ' + ' '.join(extra) if extra else ''} changed", CK.tokens(t) == want + extra)
        check("DAG-OACC s1's bank sha256 in the checker is the full 64 hex of its argv.json", CK.BANK_SHA[1] == base[1][base[1].index("--dagger-bank-sha256") + 1] and CK.HEX64.match(CK.BANK_SHA[1]) is not None)
        check("DAG-OACC s0's bank sha256 in the checker is its argv.json's", CK.BANK_SHA[0] == base[0][base[0].index("--dagger-bank-sha256") + 1])
        for k in (0, 1):
            d = json.load(open(f"{DAGOACC}/M2b_s{k}_dagoacc/selfcheck.json", encoding="utf-8"))
            lvd = d["levers"]
            check(f"DAG-OACC s{k} selfcheck.json agrees with the checker's constants (params, steps, init, bank, dagger seed, w_acc, p_gt)",
                  lvd["params"] == CK.PARAMS and lvd["n_params_trainable"] == CK.N_PARAMS_TRAINABLE and d["steps_total"] == CK.RUN_STEPS and d["S_steps_per_epoch_100pct"] == CK.S_PER_EPOCH
                  and d["info"]["init_ckpt"]["sha256"] == CK.INIT_SHA[k] and lvd["dagger_bank_sha256"] == CK.BANK_SHA[k] and lvd["dagger"]["dagger_seed"] == k and lvd["w_acc"] == CK.W_ACC
                  and lvd["p_gt_start"] == CK.P_GT and lvd["dagger_p"] == CK.DAGGER_P and all(d["info"][x] == v for x, v in CK.DATA.items()))
        sums = open(f"{DAGOACC}/M2b_s0_dagoacc/SHA256SUMS.txt", encoding="utf-8").read()
        check("the pinned DAG-OACC s0 ckpt sha256 is the SHA256SUMS.txt line", CK.sums_sha(f"{DAGOACC}/M2b_s0_dagoacc/SHA256SUMS.txt") == CK.DAGOACC_S0_CKPT_SHA and CK.DAGOACC_S0_CKPT_SHA in sums)
    good = CK.tokens("OBD-BAND_s0")
    i = good.index("--w-op") + 1
    bad = good[:i] + ["0.5"] + good[i + 1:]
    r = CK.audit_line("OBD-BAND_s0", " ".join(bad))
    check("a changed --w-op is refused and the first differing token is named", len(r) == 1 and f"token {i} is '0.5'" in r[0] and "'0.3333333'" in r[0])
    check("a changed seed is refused", CK.audit_line("OBD-BAND_s0", " ".join(good).replace("--seed 0", "--seed 3")) != [])
    check("the detach dropped (CTL) is refused", CK.audit_line("OBD-CTL_s0", " ".join(CK.tokens("OBD-CTL_s0")).replace(" --detach-upper", "")) != [])
    check("a detach added to NULL is refused", CK.audit_line("OBD-NULL_s0", " ".join(CK.tokens("OBD-NULL_s0")) + " --detach-upper") != [])
    check("--w-acc changed is refused", CK.audit_line("OBD-CTL_s0", " ".join(CK.tokens("OBD-CTL_s0")).replace("--w-acc 49.6969", "--w-acc 0")) != [])
    check("--dagger-p changed is refused", CK.audit_line("OBD-CTL_s0", " ".join(CK.tokens("OBD-CTL_s0")).replace("--dagger-p 0.1", "--dagger-p 0")) != [])
    check("a truncated bank sha256 is refused", CK.audit_line("OBD-CTL_s1", " ".join(CK.tokens("OBD-CTL_s1")).replace(CK.BANK_SHA[1], CK.BANK_SHA[1][:60])) != [])
    check("--steps changed is refused", CK.audit_line("OBD-CTL_s0", " ".join(CK.tokens("OBD-CTL_s0")).replace("--steps 3000", "--steps 2000")) != [])
    check("--freeze added is refused (nothing is frozen)", CK.audit_line("OBD-CTL_s0", " ".join(CK.tokens("OBD-CTL_s0")) + " --freeze S") != [])
    check("an extra token is refused", CK.audit_line("OBD-CTL_s0", " ".join(CK.tokens("OBD-CTL_s0")) + " --steps 5") != [])
    check("a missing token is refused", CK.audit_line("OBD-CTL_s0", " ".join(CK.tokens("OBD-CTL_s0")[:-2])) != [])
    check("the CTL argv shipped for BAND is refused", CK.audit_line("OBD-BAND_s1", " ".join(CK.tokens("OBD-CTL_s1"))) != [])
    check("the s0 argv shipped for an s1 tag is refused", CK.audit_line("OBD-CTL_s1", " ".join(CK.tokens("OBD-CTL_s0"))) != [])
    check("BAND_rep is OBD-BAND_s0 apart from the seed and --out", [(a, b) for a, b in zip(CK.tokens("OBD-BAND_rep"), CK.tokens("OBD-BAND_s0")) if a != b] == [("2", "0"), (CK.out_decl("OBD-BAND_rep"), CK.out_decl("OBD-BAND_s0"))])
    check("NULL_rep is OBD-NULL_s0 apart from the seed and --out", [(a, b) for a, b in zip(CK.tokens("OBD-NULL_rep"), CK.tokens("OBD-NULL_s0")) if a != b] == [("2", "0"), (CK.out_decl("OBD-NULL_rep"), CK.out_decl("OBD-NULL_s0"))])
    check("the frozen argv file hashes to the pin", CK.sha_file(CK.ARGVS_FILE) == CK.ARGVS_SHA256)
    orig = os.path.join(PKG, "..", "argvs_obd.json")
    if os.path.exists(orig):
        check("the packaged argvs equal theory/obd/argvs_obd.json", CK.sha_file(orig) == CK.ARGVS_SHA256)
    check("the CLI argv command prints the NULL arm's line", CK.main(["argv", "--tag", "OBD-NULL_s0", "--out-txt", os.path.join(tmp, "a.txt")]) == 0 and open(os.path.join(tmp, "a.txt")).read().strip() == " ".join(CK.tokens("OBD-NULL_s0")))
    # ---- a clean selfcheck passes, per run tag
    for t in runnable:
        toks, sc = sc_of(t)
        check(f"clean selfcheck passes {t}", CK.gate_problems(t, toks, sc) == [])
    # ---- planted-bad selfchecks: every assertion refuses
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
        "S count changed": lambda s: s["levers"]["params"].__setitem__("S", 1321738),
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
        "stok_dim built": lambda s: s["levers"].__setitem__("stok_dim", 8),
        "kin_mode": lambda s: s["levers"].__setitem__("kin_mode", "steer"),
        "live pass": lambda s: s["live"].__setitem__("pass", False),
        "live missing terms": lambda s: s["live"].__setitem__("missing_terms", ["tac"]),
        "live unexpected dead": lambda s: s["live"].__setitem__("unexpected_dead", ["e_code.weight"]),
        "live control": lambda s: s["live"].__setitem__("control_prior_grad_sum", 1e-9),
        "live head_o": lambda s: s["live"].__setitem__("head_o_grad_step0", 0.0),
        "live non-finite term": lambda s: s["live"]["terms"].__setitem__("ade", float("nan")),
        "live ade term absent": lambda s: s["live"]["terms"].pop("ade"),
        "live acc term zero": lambda s: s["live"]["terms"].__setitem__("acc", 0.0),
    }
    for g in ("adapter", "S", "T", "O", "head_s", "head_o", "wm"):
        muts[f"live group {g} has no gradient (a frozen group)"] = (lambda g_: lambda s: s["live"]["groups_grad"].__setitem__(g_, 0.0))(g)
    band_muts = {
        "w_op built value": lambda s: s["levers"].__setitem__("w_op", 0.5),
        "w_tac_traj built value": lambda s: s["levers"].__setitem__("w_tac_traj", 1.0),
        "w_op absent": lambda s: s["levers"].pop("w_op"),
        "probe absent": lambda s: s.pop("otband_probe"),
        "probe pass": lambda s: s["otband_probe"].__setitem__("pass", False),
        "probe detach_upper off": lambda s: s["otband_probe"].__setitem__("detach_upper", False),
        "probe says detach-only": lambda s: s["otband_probe"].__setitem__("otband", "detach-only"),
        "probe weight w_op": lambda s: s["otband_probe"]["contribution"]["w_op"].__setitem__("weight", 0.5),
        "probe weight w_tac": lambda s: s["otband_probe"]["contribution"]["w_tac_traj"].__setitem__("weight", 0.1),
        "probe band does not reach the total (w_op)": lambda s: s["otband_probe"]["contribution"]["w_op"].__setitem__("moved_total_by", 0.0),
        "probe band does not reach the total (w_tac)": lambda s: s["otband_probe"]["contribution"]["w_tac_traj"].__setitem__("moved_total_by", 0.0),
        "probe band term zero": lambda s: s["otband_probe"]["contribution"]["w_op"].__setitem__("term", 0.0),
        "purity upper gradient": lambda s: s["otband_probe"]["purity"].__setitem__("upper_grad_abs_sum", 1e-12),
        "purity absent": lambda s: s["otband_probe"].pop("purity"),
        "purity head_o zero": lambda s: s["otband_probe"]["purity"].__setitem__("head_o", 0.0),
        "code edge cut": lambda s: s["otband_probe"].__setitem__("code_edge_max_dknot", 0.0),
        "purity op term on S and T": lambda s: s["otband_probe"]["purity"]["by_term"]["op"].__setitem__("upper_grad_abs_sum", 1e-12),
        "purity tac term on S and T": lambda s: s["otband_probe"]["purity"]["by_term"]["tac"].__setitem__("upper_grad_abs_sum", 1e-12),
        "purity tac term head_o zero": lambda s: s["otband_probe"]["purity"]["by_term"]["tac"].__setitem__("head_o", 0.0),
        "purity tac term absent": lambda s: s["otband_probe"]["purity"]["by_term"].pop("tac"),
        "purity acc term absent (the held trainer's probe)": lambda s: s["otband_probe"]["purity"]["by_term"].pop("acc"),
        "purity acc term on S and T": lambda s: s["otband_probe"]["purity"]["by_term"]["acc"].__setitem__("upper_grad_abs_sum", 1e-12),
        "purity acc term head_o zero": lambda s: s["otband_probe"]["purity"]["by_term"]["acc"].__setitem__("head_o", 0.0),
        "purity ade (an unexpected term) present": lambda s: s["otband_probe"]["purity"]["by_term"].__setitem__("ade", {"upper_grad_abs_sum": 0.0, "head_o": 1.0}),
        "band term missing in G-LIVE": lambda s: s["live"]["terms"].pop("fde_tac"),
    }
    ctl_muts = {
        "band lever built in a control": lambda s: s["levers"].__setitem__("w_op", CK.W_OP),
        "probe absent in a control": lambda s: s.pop("otband_probe"),
        "control probe reads a band lever": lambda s: s["otband_probe"].__setitem__("otband", "on"),
        "control detach off": lambda s: s["otband_probe"].__setitem__("detach_upper", False),
        "control probe failed": lambda s: s["otband_probe"].__setitem__("pass", False),
        "control ade on S and T": lambda s: s["otband_probe"]["purity"]["by_term"]["ade"].__setitem__("upper_grad_abs_sum", 1e-12),
        "control fde head_o zero": lambda s: s["otband_probe"]["purity"]["by_term"]["fde"].__setitem__("head_o", 0.0),
        "control fde absent": lambda s: s["otband_probe"]["purity"]["by_term"].pop("fde"),
        "control purity acc term absent (the held trainer's probe)": lambda s: s["otband_probe"]["purity"]["by_term"].pop("acc"),
        "control purity acc term on S and T": lambda s: s["otband_probe"]["purity"]["by_term"]["acc"].__setitem__("upper_grad_abs_sum", 1e-12),
        "control purity acc head_o zero": lambda s: s["otband_probe"]["purity"]["by_term"]["acc"].__setitem__("head_o", 0.0),
        "band terms in a control": lambda s: s["live"]["terms"].__setitem__("op", 0.5),
        "control purity sum": lambda s: s["otband_probe"]["purity"].__setitem__("upper_grad_abs_sum", 1e-12),
    }
    null_muts = {
        "band lever built in NULL": lambda s: s["levers"].__setitem__("w_op", CK.W_OP),
        "NULL carries a detach probe": lambda s: s.__setitem__("otband_probe", {"otband": "detach-only", "detach_upper": True, "pass": True}),
        "NULL carries a band probe": lambda s: s.__setitem__("otband_probe", {"otband": "on", "detach_upper": False, "pass": True}),
        "band terms in NULL": lambda s: s["live"]["terms"].__setitem__("op", 0.5),
    }
    for t in runnable:
        toks, sc = sc_of(t)
        kind = CK.TABLE[CK.base_tag(t)]["kind"]
        ms = dict(muts)
        ms.update({"band": band_muts, "ctl": ctl_muts, "null": null_muts}[kind])
        for name, f in ms.items():
            s2 = copy.deepcopy(sc)
            try:
                f(s2)
            except KeyError:
                continue
            check(f"{t}: planted '{name}' is refused", len(CK.gate_problems(t, toks, s2)) >= 1)
    # the pinned argv is itself read: a changed seed / dagger seed / w-acc / detach in the line handed to the gate is refused even when the selfcheck is clean
    for t, mut in (("OBD-BAND_s0", lambda a: [("3" if x == "0" and a[i - 1] == "--seed" else x) for i, x in enumerate(a)]),
                   ("OBD-CTL_s1", lambda a: [x for x in a if x != "--detach-upper"]), ("OBD-NULL_s0", lambda a: a + ["--detach-upper"])):
        toks, sc = sc_of(t)
        check(f"{t}: a clean selfcheck does not rescue a changed pinned argv", len(CK.gate_problems(t, mut(list(toks)), sc)) >= 1)
    # the acceleration blocker is a constant, tested both ways
    toks, sc = sc_of("OBD-CTL_s0")
    sc_noacc = copy.deepcopy(sc)
    sc_noacc["otband_probe"]["purity"]["by_term"].pop("acc")
    check("BLOCKER: the held trainer's selfcheck (purity without acc) is refused for CTL", len(CK.gate_problems("OBD-CTL_s0", toks, sc_noacc)) >= 1 and "acc" in " ".join(CK.gate_problems("OBD-CTL_s0", toks, sc_noacc)))
    toks_b, sc_b = sc_of("OBD-BAND_s0")
    sc_b["otband_probe"]["purity"]["by_term"].pop("acc")
    check("BLOCKER: the held trainer's selfcheck (purity without acc) is refused for BAND", len(CK.gate_problems("OBD-BAND_s0", toks_b, sc_b)) >= 1)
    toks_n, sc_n = sc_of("OBD-NULL_s0")
    check("OBD-NULL needs no probe: its clean selfcheck has none and passes", "otband_probe" not in sc_n and CK.gate_problems("OBD-NULL_s0", toks_n, sc_n) == [])
    CK.ACC_PROBE_REQUIRED = False
    check("with ACC_PROBE_REQUIRED off (a test-only switch) the held trainer's selfcheck would pass: the refusal above is the constant's doing", CK.gate_problems("OBD-CTL_s0", toks, sc_noacc) == [])
    CK.ACC_PROBE_REQUIRED = True
    check("the trainer needles include the probe change", 'keys = (*keys, "acc")' in CK.NEEDLES[CK.TRAINER_REL])
    # ---- the needles against the real commit (source read; skipped when the mirror is absent)
    if os.path.isdir(PUSH):
        def show(path):
            r = subprocess.run(["git", "-c", "safe.directory=*", "-C", PUSH, "show", f"{COMMIT_REC}:{path}"], capture_output=True)
            return r.stdout.decode("utf-8", "replace") if r.returncode == 0 else None
        for rel, ns in CK.NEEDLES.items():
            src = show(rel)
            if src is None:
                check(f"{rel} is readable at {COMMIT_REC[:8]}", False)
                continue
            lacking = [n for n in ns if n not in src]
            check(f"at {COMMIT_REC[:8]} {rel} carries every needle except the probe change", lacking == [] or (rel == CK.TRAINER_REL and lacking == ['keys = (*keys, "acc")']))
        check("at 202849da the purity probe change is ABSENT (the blocker)", 'keys = (*keys, "acc")' not in (show(CK.TRAINER_REL) or ""))
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
    os.environ.update(OBD_ROOT=root, OBD_COMMIT="a" * 40, OBD_CKPT_LOADER="pickle", OBD_FLOOR_BYTES="1000")
    for rel in ("cache_full/native_tracks.pt", "data/v9/lead_train4572.npz", "data/v9/refcv6_speed_max_v8_train.jsonl", "runs/v8f_mini_full/join_train.npz", "runs/v8f_mini_full/p10/p10_prior.pt"):
        put(os.path.join(root, *rel.split("/")), rel.encode())
    put(os.path.join(root, "data", "v9", "v9_labels_train.npz"), b"v9 labels")
    os.environ["OBD_FAKE_V9_MD5"] = hashlib.md5(b"v9 labels").hexdigest()
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
    put(os.path.join(root, "obd", "repo", "stack", "scripts", "v8f_mini_train.py"), b"trainer v1")
    put(os.path.join(root, "obd", "tree_manifest.json"), b"{}")
    tag = "OBD-BAND_s0"
    line = " ".join(CK.tokens(tag))
    b_, bad_ = CK.binding(tag, line)
    check("binding reads the six data files and the DAgger bank", bad_ == [] and len(b_["data_md5"]) == 6 and b_["dagger_bank_sha256"] == bshas["0"])
    check("bank_problems is clean for both seeds", CK.bank_problems(0) == [] and CK.bank_problems(1) == [])
    sc_p = os.path.join(CK.run_dir(tag), "gate", "selfcheck.json")
    put(sc_p, b"{}")
    tok = {"schema": "obd-gate-token-v1", "written_by": "obd_check.py gate", "selfcheck_sha256": CK.sha_file(sc_p), **b_}
    put(CK.token_path(tag), json.dumps(tok).encode())
    check("a token bound to the current state is valid", CK.token_ok(tag, line) == [])
    os.environ["OBD_COMMIT"] = "b" * 40
    check("a token for another commit is refused", CK.token_ok(tag, line) != [])
    os.environ["OBD_COMMIT"] = "a" * 40
    check("a token for another argv is refused", CK.token_ok(tag, line + " --x") != [])
    put(os.path.join(root, "obd", "repo", "stack", "scripts", "v8f_mini_train.py"), b"trainer v2")
    check("a token after the trainer bytes changed is refused", CK.token_ok(tag, line) != [])
    put(os.path.join(root, "obd", "repo", "stack", "scripts", "v8f_mini_train.py"), b"trainer v1")
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
    put(os.path.join(root, "obd", "tree_manifest.json"), b'{"x": 1}')
    check("a token after the shipped tree manifest changed is refused", CK.token_ok(tag, line) != [])
    put(os.path.join(root, "obd", "tree_manifest.json"), b"{}")
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
    check("verify-token refuses a wrong tag", vt("OBD-CTL_s0", "a" * 40) == 1)
    check("verify-token refuses a wrong commit", vt(tag, "c" * 40) == 1)
    open(os.path.join(tmp, "arg2.txt"), "w").write(line.replace("--seed 0", "--seed 9") + "\n")
    check("verify-token refuses an argv that is not the frozen one", vt(tag, "a" * 40, os.path.join(tmp, "arg2.txt")) == 1)

    # post check, per kind, on an out dir written by the stand-in's own format
    def post_case(tg):
        toks = CK.tokens(tg)
        spec = CK.TABLE[tg]
        kind, init, seed = spec["kind"], spec["init"], spec["seed"]
        out = CK.loc(CK.out_decl(tg))
        st0 = PT.model_state(init)
        moved = {k: (v + 1.0 if k.startswith(("read_s", "head_s", "read_t", "trunk_t", "h_goal", "head_o", "read_o")) else v) for k, v in st0.items()}
        ck_ok = {"model": moved, "step": CK.RUN_STEPS, "seed": seed, "arm": "M2b", "argv_sha": CK.argv_sha16(toks),
                 "cfg": {"detach_upper": kind != "null", "w_traj": 0.0 if kind == "band" else 1.0, "w_op": CK.W_OP if kind == "band" else 0.0, "w_tac_traj": CK.W_TAC if kind == "band" else 0.0, "w_acc": CK.W_ACC},
                 "dagger": {"dagger_p": 0.1, "dagger_seed": init, "dagger_bank_sha256": CK.bank_sha(init)}, "init_ckpt": {"sha256": shas[str(init)]}}
        summ = {"done": True, "steps": CK.RUN_STEPS, "argv_sha": CK.argv_sha16(toks), "seed": seed}

        def write_out(ck, sm):
            put(os.path.join(out, "ckpt_last.pt"), pickle.dumps(ck))
            put(os.path.join(out, "summary.json"), json.dumps(sm).encode())
        write_out(ck_ok, summ)
        check(f"post check {tg}: a correct finished run passes", CK.post_problems(tg, toks, out)[1] == [])
        cases = [("steps short", ck_ok, dict(summ, steps=100)), ("not done", ck_ok, dict(summ, done=False)), ("summary argv_sha", ck_ok, dict(summ, argv_sha="0" * 16)),
                 ("summary seed", ck_ok, dict(summ, seed=seed + 5)), ("ckpt step", dict(ck_ok, step=5), summ), ("ckpt seed", dict(ck_ok, seed=seed + 5), summ), ("ckpt arm", dict(ck_ok, arm="other"), summ),
                 ("ckpt argv_sha", dict(ck_ok, argv_sha="0" * 16), summ),
                 ("cfg detach_upper flipped", dict(ck_ok, cfg=dict(ck_ok["cfg"], detach_upper=not ck_ok["cfg"]["detach_upper"])), summ),
                 ("cfg w_traj flipped", dict(ck_ok, cfg=dict(ck_ok["cfg"], w_traj=1.0 - ck_ok["cfg"]["w_traj"])), summ), ("cfg w_op", dict(ck_ok, cfg=dict(ck_ok["cfg"], w_op=0.1)), summ),
                 ("cfg w_tac_traj", dict(ck_ok, cfg=dict(ck_ok["cfg"], w_tac_traj=0.1)), summ), ("cfg w_acc", dict(ck_ok, cfg=dict(ck_ok["cfg"], w_acc=0.0)), summ),
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
    for tg in ("OBD-NULL_s0", "OBD-CTL_s1", "OBD-BAND_rep"):
        out = post_case(tg)
    # cleanup guard
    tag = "OBD-BAND_rep"
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
    check("1.0 s/step: 7 runs x 3,000 steps = 5.83 h; with OBD-NULL_s1 6.67 h", abs(pr1["projected_total_h_planned"] - 21000 / 3600) < 1e-9 and abs(pr1["projected_total_h_with_null_s1"] - 24000 / 3600) < 1e-9 and pr1["within_limit_planned"] and pr1["n_runs_planned"] == 7)
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
    # ---- NULL parity (dev box)
    cks = os.path.join(tmp, "par")
    os.makedirs(cks, exist_ok=True)
    st = PT.model_state(0)
    mine = os.path.join(cks, "mine.pt")
    ref = os.path.join(cks, "ref.pt")
    pickle.dump({"model": st, "argv_sha": "aaaa", "step": 3000}, open(mine, "wb"))
    pickle.dump({"model": st, "argv_sha": "bbbb", "step": 3000}, open(ref, "wb"))            # same weights, other metadata: the whole-file sha256 cannot match (the checkpoint embeds argv_sha, which holds --out)
    sums = os.path.join(cks, "SHA256SUMS.txt")
    open(sums, "w").write(f"{CK.sha_file(ref)} *ckpt_last.pt\n{'1' * 64} *summary.json\n")
    os.environ["OBD_TEST_DAGOACC_SHA"] = CK.sha_file(ref)
    outp = os.path.join(cks, "o.json")
    check("sums_sha reads the ckpt_last.pt line", CK.sums_sha(sums) == CK.sha_file(ref))
    check("parity (file basis): same weights but other metadata -> file sha differs -> OBD-NULL_s1 is appended", CK.cmd_parity({"bank-ckpt": mine, "sums": sums, "out": outp, "dagoacc-ckpt": ref}) == 0
          and json.load(open(outp))["append_obd_null_s1"] is True and json.load(open(outp))["file_sha_equal"] is False and json.load(open(outp))["tensor_equal"] is True)
    check("parity (tensors basis): the same weights -> not appended", CK.cmd_parity({"bank-ckpt": mine, "sums": sums, "out": outp, "dagoacc-ckpt": ref, "basis": "tensors"}) == 0 and json.load(open(outp))["append_obd_null_s1"] is False)
    shutil_copy = lambda a, b: open(b, "wb").write(open(a, "rb").read())      # noqa: E731
    same = os.path.join(cks, "same.pt")
    shutil_copy(ref, same)
    check("parity (file basis): an identical file -> not appended", CK.cmd_parity({"bank-ckpt": same, "sums": sums, "out": outp}) == 0 and json.load(open(outp))["append_obd_null_s1"] is False and json.load(open(outp))["file_sha_equal"] is True)
    st2 = dict(st)
    st2["head_o.w"] = st["head_o.w"] + 1e-7
    other = os.path.join(cks, "other.pt")
    pickle.dump({"model": st2, "argv_sha": "bbbb", "step": 3000}, open(other, "wb"))
    check("parity (tensors basis): one changed tensor -> appended", CK.cmd_parity({"bank-ckpt": other, "sums": sums, "out": outp, "dagoacc-ckpt": ref, "basis": "tensors"}) == 0 and json.load(open(outp))["append_obd_null_s1"] is True
          and json.load(open(outp))["tensor_detail"]["n_tensors_differ"] == 1)
    check("parity: basis tensors without the DAG-OACC checkpoint is refused (2)", CK.cmd_parity({"bank-ckpt": mine, "sums": sums, "out": outp, "basis": "tensors"}) == 2)
    check("parity: a basis other than file / tensors is refused (2)", CK.cmd_parity({"bank-ckpt": mine, "sums": sums, "out": outp, "basis": "x"}) == 2)
    check("parity: a missing bank checkpoint is refused (2)", CK.cmd_parity({"bank-ckpt": os.path.join(cks, "none.pt"), "sums": sums, "out": outp}) == 2)
    check("parity: a SHA256SUMS without a ckpt_last.pt line is refused (2)", CK.cmd_parity({"bank-ckpt": mine, "sums": os.path.join(cks, "empty.txt"), "out": outp}) == 2)
    open(os.path.join(cks, "sums2.txt"), "w").write(f"{'2' * 64} *ckpt_last.pt\n")
    check("parity: a SHA256SUMS whose value is not the pinned DAG-OACC one is refused (2): the reference moved", CK.cmd_parity({"bank-ckpt": mine, "sums": os.path.join(cks, "sums2.txt"), "out": outp}) == 2)
    check("parity: a DAG-OACC checkpoint that does not hash to the SUMS value is refused (2)", CK.cmd_parity({"bank-ckpt": mine, "sums": sums, "out": outp, "dagoacc-ckpt": other}) == 2)
    os.environ.pop("OBD_TEST_DAGOACC_SHA")
    # ---- hygiene of the package
    bad_files = []
    for dp, _, fs in os.walk(PKG):
        for f in fs:
            p = os.path.join(dp, f)
            if f.endswith((".pyc", ".out")) or "__pycache__" in dp:
                continue
            raw = open(p, "rb").read()
            if f.endswith(".sh") and b"\r" in raw:
                bad_files.append(f"CRLF in {f}")
            if re.search(rb"\btext\s*=\s*True", raw) and not f.startswith("test_obd_check"):
                bad_files.append(f"bare text=True in {f}")
            if re.search(rb"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", raw) and not f.endswith(".json"):
                bad_files.append(f"IP-like string in {f}")
            if re.match(r"(?i)^(aux|con|prn|nul)(\.|$)", f):
                bad_files.append(f"reserved name {f}")
    check("package hygiene (LF scripts, no bare text=True, no IP, no reserved names): " + "; ".join(bad_files), bad_files == [])
    print(f"UNIT ok={N_OK} bad={N_BAD}")
    return 0 if N_BAD == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
