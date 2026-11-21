"""Unit tests of actsens_check.py (pure Python; numpy only for the weights identity; no torch, no GPU, no network).
Run:  C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I tests/test_actsens_check.py <scratch dir>     -> prints  UNIT ok=<n> bad=<m>  (exit 1 on any bad)
Every gate assertion has a planted-bad selfcheck that must be refused; every clean selfcheck must pass."""
import copy
import hashlib
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, PKG)
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import actsens_check as CK  # noqa: E402
import plant_trainer as PT  # noqa: E402

N_OK, N_BAD = 0, 0


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
    os.environ["ACTSENS_FAKE_INIT_SHA_FILE"] = os.path.join(TMP, "init_sha.json")
    return toks, PT.good_selfcheck(gt)


def GP(t, toks, sc, ref=None):
    """gate_problems with the reference gate's selfcheck supplied the way do_gate does it: an IDM tag is read against REF_s{k}'s (the clean one unless `ref` is given); REF / DIV / NULL need none"""
    if CK.TABLE[CK.base_tag(t)]["lever"] == "idm" and ref is None:
        ref = sc_of(CK.ref_tag(CK.TABLE[t]["init"]))[1]
    return CK.gate_problems(t, toks, sc, ref)


def only_idm(s):
    d = s["levers"].get("wm_idm_gamma_readback")
    if not d or d.get("declared", [0])[0] != 1000.0:
        raise KeyError("not an IDM arm")


def only_ref(s):
    d = s["levers"].get("wm_idm_gamma_readback")
    if not d or d.get("declared") != [100.0, 100.0, 100.0]:
        raise KeyError("not a REF arm")


def put(p, data=b""):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "wb").write(data)


def main(tmp):
    global TMP
    TMP = tmp
    os.makedirs(tmp, exist_ok=True)
    json.dump({"0": CK.INIT_SHA[0], "1": CK.INIT_SHA[1]}, open(os.path.join(tmp, "init_sha.json"), "w"))
    runnable = CK.RUN_ORDER + [CK.PILOT_TAG, "REF_s0", "REF_s1"]
    # ---- the card's order
    check("run order is IDM_s0, NULL_s2, DIV_s0, IDM_s1, DIV_s1, NULL_s3", CK.RUN_ORDER == ["IDM_s0", "NULL_s2", "DIV_s0", "IDM_s1", "DIV_s1", "NULL_s3"])
    check("no blocked tag", all(not v["blocked"] for v in CK.TABLE.values()))
    check("AMEND_ACTSENS_1: REF_s0 / REF_s1 are gate-only tags: lever ref, order 0, not in RUN_ORDER", [t for t, v in CK.TABLE.items() if v["lever"] == "ref"] == ["REF_s0", "REF_s1"] and all(CK.TABLE[t]["order"] == 0 and t not in CK.RUN_ORDER for t in ("REF_s0", "REF_s1")))
    check("AMEND_ACTSENS_1: the pilot job gates REF_s0, IDM_s0, DIV_s0 in that order", CK.GATE_AT_PILOT == ["REF_s0", "IDM_s0", "DIV_s0"])
    check("AMEND_ACTSENS_1: the trainer needles carry live_first", '"live_first"' in CK.NEEDLES[CK.TRAINER_REL])
    for k in (0, 1):
        rt, it, nt = CK.tokens(f"REF_s{k}"), CK.tokens(f"NULL_s{k + 2}"), CK.tokens(f"IDM_s{k}")
        check(f"AMEND_ACTSENS_1: REF_s{k} = R35_s{k}'s argv + --wm-idm-gamma 100 100 100, --out = the REF gate out", rt[-4:] == ["--wm-idm-gamma", "100", "100", "100"] and rt[rt.index("--out") + 1] == CK.gate_out_decl(f"REF_s{k}")
              and [x for x in rt[:-4] if x != rt[rt.index("--out") + 1]] == [x for x in nt[:-4] if x != nt[nt.index("--out") + 1]] and rt[rt.index("--seed") + 1] == str(k))

    check("steps: 19,164 full; pilot 200", CK.RUN_STEPS == 19164 and CK.PILOT_STEPS == 200)
    # ---- argv pin
    for t in CK.TABLE:
        check(f"frozen argv of {t} is accepted", CK.audit_line(t, " ".join(CK.tokens(t))) == [])
        tk = CK.tokens(t)
        g = lambda f: tk[tk.index(f) + 1]                                                  # noqa: E731
        lev = CK.TABLE[t]["lever"]
        if lev == "ref":
            continue
        check(f"{t}: lever {lev}: the card's one change from R35, --seed {CK.TABLE[t]['seed']}, init M2b_s{CK.TABLE[t]['init']}, zcache of the same seed as the init",
              (("--wm-idm-gamma" in tk and tk[tk.index("--wm-idm-gamma") + 1:tk.index("--wm-idm-gamma") + 4] == ["1000", "100", "100"]) == (lev == "idm")) and (("--wm-sampler" in tk and g("--wm-sampler") == "divrs") == (lev == "div"))
              and g("--seed") == str(CK.TABLE[t]["seed"]) and f"M2b_s{CK.TABLE[t]['init']}" in g("--init-ckpt") and f"M2b_s{CK.TABLE[t]['init']}" in g("--wm-zcache") and g("--wm-k") == "1")
        check(f"{t}: the run names --wm-zcache and --wm-only", "--wm-zcache" in tk and "--wm-only" in tk)
    check("pilot argv = DIV_s0 argv with --out moved and --steps 200 appended", CK.tokens(CK.PILOT_TAG)[:-2] == [CK.out_decl(CK.PILOT_TAG) if x == CK.out_decl("DIV_s0") else x for x in CK.tokens("DIV_s0")] and CK.tokens(CK.PILOT_TAG)[-2:] == ["--steps", "200"])
    check("pilot out dir differs from the run out dir", CK.out_decl(CK.PILOT_TAG) != CK.out_decl("DIV_s0"))
    good = CK.tokens("DIV_s0")
    i = good.index("--wm-sampler") + 1
    bad = good[:i] + ["uniform"] + good[i + 1:]
    r = CK.audit_line("DIV_s0", " ".join(bad))
    check("a changed --wm-sampler is refused and the first differing token is named", len(r) == 1 and f"token {i} is 'uniform'" in r[0])
    check("a changed seed is refused", CK.audit_line("DIV_s0", " ".join(good).replace("--seed 0", "--seed 3")) != [])
    check("the IDM argv shipped for DIV is refused", CK.audit_line("DIV_s1", " ".join(CK.tokens("IDM_s1"))) != [])
    check("an extra token is refused", CK.audit_line("DIV_s0", " ".join(good) + " --steps 5") != [])
    check("a missing token is refused", CK.audit_line("DIV_s0", " ".join(good[:-2])) != [])
    check("NULL_s2 is R35_s0 apart from the seed and --out; NULL_s3 is R35_s1 apart from the seed and --out", [(a, b) for a, b in zip(CK.tokens("NULL_s2"), CK.tokens("IDM_s0")) if a != b][:2] == [("2", "0"), (CK.out_decl("NULL_s2"), CK.out_decl("IDM_s0"))]
          and len(CK.tokens("NULL_s2")) + 4 == len(CK.tokens("IDM_s0")) and len(CK.tokens("NULL_s3")) + 4 == len(CK.tokens("IDM_s1")))
    check("the frozen argv file hashes to the pin", CK.sha_file(CK.ARGVS_FILE) == CK.ARGVS_SHA256)
    orig = os.path.join(PKG, "..", "argvs_actsens_amend1.txt")
    if os.path.exists(orig):
        rows = {l.split("\t")[0]: l.rstrip("\n").split("\t")[1] for l in open(orig, encoding="utf-8") if l.strip() and not l.startswith("#")}
        check("the packaged argvs equal theory/actsens/argvs_actsens.txt token for token", all(" ".join(CK.tokens(t)) == rows[t] for t in CK.TABLE) and set(rows) == set(CK.TABLE))
    # ---- a clean selfcheck passes, per run tag
    for t in runnable:
        toks, sc = sc_of(t)
        check(f"clean selfcheck passes {t}", GP(t, toks, sc) == [])
        check(f"{t}: live_first is present exactly when --wm-idm-gamma is named (IDM and REF)", ("live_first" in sc) == (t.startswith(("IDM", "REF"))))
    # ---- planted-bad selfchecks: every assertion refuses
    muts = {
        "declared_vs_built": lambda s: s.__setitem__("declared_vs_built", "FAIL"),
        "argv_sha": lambda s: s.__setitem__("argv_sha", "0" * 16),
        "steps_total": lambda s: s.__setitem__("steps_total", 12345),
        "data train_sha256": lambda s: s["info"].__setitem__("train_sha256", "0" * 64),
        "data n_clips": lambda s: s["info"].__setitem__("n_clips", 4000),
        "data excluded_in_train": lambda s: s["info"].__setitem__("excluded_in_train", 2),
        "md5_overridden": lambda s: s["info"].__setitem__("md5_overridden", True),
        "batch": lambda s: s["info"].__setitem__("batch", 32),
        "wm_only sha256": lambda s: s["info"]["wm_only"].__setitem__("sha256", "0" * 64),
        "wm_only applied": lambda s: s["info"]["wm_only"].__setitem__("applied", False),
        "wm_only mismatch": lambda s: s["info"]["wm_only"].__setitem__("n_mismatch_after_load", 3),
        "wm_only source seed": lambda s: s["info"]["wm_only"].__setitem__("source_seed", 5),
        "wm_only path": lambda s: s["info"]["wm_only"].__setitem__("path", r"D:\tanitad\runs\other\ckpt_last.pt"),
        "wm_only record absent": lambda s: s["info"].pop("wm_only"),
        "wm re-init seed": lambda s: s["info"]["wm_only"].__setitem__("wm_reinit_seed", 99),
        "idm readback absent": lambda s: s["levers"].pop("wm_idm_gamma_readback"),
        "idm readback built at the old weights": lambda s: (only_idm(s), s["levers"]["wm_idm_gamma_readback"]["built"].__setitem__(0, 100.0)),
        "ref readback built at the dose": lambda s: (only_ref(s), s["levers"]["wm_idm_gamma_readback"]["built"].__setitem__(0, 1000.0)),
        "ref readback declared at the dose": lambda s: (only_ref(s), s["levers"]["wm_idm_gamma_readback"].__setitem__("declared", [1000.0, 100.0, 100.0])),
        "idm readback built level 2 moved": lambda s: s["levers"]["wm_idm_gamma_readback"]["built"].__setitem__(1, 1000.0),
        "idm readback declared": lambda s: (only_idm(s), s["levers"]["wm_idm_gamma_readback"].__setitem__("declared", [100.0, 100.0, 100.0])),
        "idm readback on a non-idm arm": lambda s: s["levers"].__setitem__("wm_idm_gamma_readback", {"built": [1000.0, 100.0, 100.0]}) if "wm_idm_gamma_readback" not in s["levers"] else (_ for _ in ()).throw(KeyError()),
        "sampler record absent on a div arm": lambda s: s["levers"].pop("wm_sampler"),
        "sampler built uniform on a div arm": lambda s: s["levers"]["wm_sampler"].__setitem__("kind", "uniform"),
        "sampler edge drifted": lambda s: s["levers"]["wm_sampler"]["yaw_edges"].__setitem__(3, 0.25),
        "sampler cap drifted": lambda s: s["levers"]["wm_sampler"].__setitem__("cap_mult", 30.0),
        "sampler cell counts do not sum": lambda s: s["levers"]["wm_sampler"].__setitem__("cell_counts", [1] * 25),
        "sampler control 1 failed": lambda s: s["wm_sampler_probe"].__setitem__("pass", False),
        "sampler probe absent on a div arm": lambda s: s.pop("wm_sampler_probe"),
        "sampler probe on a non-div arm": lambda s: s.__setitem__("wm_sampler_probe", {"pass": True}) if "wm_sampler_probe" not in s else (_ for _ in ()).throw(KeyError()),
        "WMAux.gamma moved": lambda s: s["levers"].__setitem__("wm_idm_gamma", 1000.0),
        "n_windows moved": lambda s: s["info"].__setitem__("n_windows", 40000),
        "window list sha": lambda s: s["info"].__setitem__("train_window_list_sha256", "0" * 64),
        "C3 live_first: an unweighted term moved": lambda s: (only_idm(s), s["live_first"].__setitem__("wm/pred/L2", s["live_first"]["wm/pred/L2"] * 1.0000001)),
        "C3 live_first: level/L2 moved": lambda s: (only_idm(s), s["live_first"].__setitem__("wm/level/L2", s["live_first"]["wm/level/L2"] * 1.0000001)),
        "C3 live_first: level/L1 not REF + 900 x idm": lambda s: (only_idm(s), s["live_first"].__setitem__("wm/level/L1", s["live_first"]["wm/level/L1"] + 1.0)),
        "C3 live_first: a term missing": lambda s: (only_idm(s), s["live_first"].pop("wm/sigreg/L3")),
        "C3 live_first absent on an IDM arm": lambda s: (only_idm(s), s.pop("live_first")),
        "C3 live_first equal to live.terms (the last step)": lambda s: (only_idm(s), s.__setitem__("live_first", dict(s["live"]["terms"]))),
        "REF live.terms: one term changed": lambda s: (only_ref(s), s["live"]["terms"].__setitem__("wm/sigreg/L3", s["live"]["terms"]["wm/sigreg/L3"] * (1 + 1e-12))),
        "REF live.terms: level/L1 changed": lambda s: (only_ref(s), s["live"]["terms"].__setitem__("wm/level/L1", s["live"]["terms"]["wm/level/L1"] + 1e-9)),
        "REF live.terms: total changed": lambda s: (only_ref(s), s["live"]["terms"].__setitem__("total", s["live"]["terms"].get("total", 1.0) + 1e-9)) if "total" in s["live"]["terms"] else (_ for _ in ()).throw(KeyError()),
        "REF live.terms: an extra key": lambda s: (only_ref(s), s["live"]["terms"].__setitem__("wm/extra", 1.0)),
        "REF live.terms: a key missing": lambda s: (only_ref(s), s["live"]["terms"].pop("wm/rollout_masked/L3")),
        "REF live_first absent": lambda s: (only_ref(s), s.pop("live_first")),
        "REF live_first holds a non-finite term": lambda s: (only_ref(s), s["live_first"].__setitem__("wm/pred/L1", float("nan"))),
        "live_first on a non-IDM, non-REF arm": lambda s: s.__setitem__("live_first", {"wm/pred/L1": 0.1}) if "live_first" not in s else (_ for _ in ()).throw(KeyError()),
        "ladder readback absent": lambda s: s["levers"].pop("wm_ladder_readback"),
        "rollout k": lambda s: s["levers"]["wm_ladder_readback"]["built"].__setitem__("k", [10, 10, 1]),
        "context": lambda s: s["levers"]["wm_ladder_readback"]["built"].__setitem__("ctx_built", [3, 3, 2]),
        "cum stride": lambda s: s["levers"]["wm_ladder_readback"]["built"].__setitem__("cum_stride", [1, 2, 4]),
        "k-step weight": lambda s: s["levers"]["wm_ladder_readback"]["built"].__setitem__("w_in_loss", [1.0, 0.5, 1.0]),
        "windows kind": lambda s: s["levers"]["wm_ladder_readback"]["windows_built"].__setitem__("kind", "dense"),
        "windows not pooled": lambda s: s["levers"]["wm_ladder_readback"]["windows_built"].__setitem__("pooled", False),
        "zcache of the other seed": lambda s: s["levers"]["wm_ladder_readback"]["windows_built"].__setitem__("zcache", r"D:\tanitad\zcache\M2b_s7\train"),
        "zcache missing clips": lambda s: s["levers"]["wm_ladder_readback"]["windows_built"].__setitem__("zcache_missing", 4),
        "no windows": lambda s: s["levers"]["wm_ladder_readback"]["windows_built"].__setitem__("n_windows", 0),
        "history K": lambda s: s["levers"].__setitem__("wm_k", 3),
        "strides": lambda s: s["levers"].__setitem__("wm_strides", [2, 2]),
        "wm_only not built": lambda s: s["levers"].__setitem__("wm_only", {"wm_only": False}),
        "WM parameter count": lambda s: s["levers"]["wm_only"].__setitem__("n_wm", 2143298 + 96),
        "trainable parameter count": lambda s: s["levers"].__setitem__("n_params_trainable", 9),
        "non-WM group trainable": lambda s: s["levers"]["params"].__setitem__("S", 5),
        "params record absent": lambda s: s["levers"].pop("params"),
        "live pass": lambda s: s["live"].__setitem__("pass", False),
        "live missing terms": lambda s: s["live"].__setitem__("missing_terms", ["wm/rollout"]),
        "live dead wm group": lambda s: s["live"].__setitem__("dead_wm_groups", ["L2"]),
        "live control adapter": lambda s: s["live"].__setitem__("control_adapter_grad_sum", 1e-9),
        "live control non-WM": lambda s: s["live"].__setitem__("control_nonwm_grad_sum", 1e-9),
        "live WM group zero": lambda s: s["live"]["wm_groups_grad"].__setitem__("L3", 0.0),
        "live non-finite term": lambda s: s["live"]["terms"].__setitem__("wm", float("nan")),
        "planner identity": lambda s: s["planner_identity_probe"].__setitem__("n_equal", 39),
        "dagger on": lambda s: s["levers"].__setitem__("dagger_p", 0.5),
        "recover on": lambda s: s["levers"]["recover"].__setitem__("recover_p", 0.3),
        "kin_mode": lambda s: s["levers"].__setitem__("kin_mode", "steer"),
    }
    n_ref = 0
    for t in runnable:
        for name, f in muts.items():
            toks, sc = sc_of(t)
            s2 = copy.deepcopy(sc)
            try:
                f(s2)
            except (KeyError, TypeError):
                continue
            n_ref += 1
            check(f"{t}: planted '{name}' is refused", len(GP(t, toks, s2)) >= 1)
    check("the planted mutations ran on every run tag (>= 40 each)", n_ref >= 40 * len(runnable))
    # ---- AMEND_ACTSENS_1 A2: the three deliberate regressions, each must make the gate refuse (stand-in modes) and the clean pair must pass
    for k in (0, 1):
        it, rt = f"IDM_s{k}", f"REF_s{k}"
        itoks = CK.tokens(it)
        gi, gr = CK.gate_argv(it, itoks), CK.gate_argv(rt, CK.tokens(rt))
        clean_i, clean_r = PT.good_selfcheck(gi), PT.good_selfcheck(gr)
        check(f"{it}: the clean IDM gate passes against the clean {rt}", CK.gate_problems(it, itoks, clean_i, clean_r) == [])
        r35 = CK.r35_terms(k)
        check(f"{rt}: the stand-in REF writes live.terms equal to R35_s{k}'s on every key, bit for bit", clean_r["live"]["terms"] == r35)
        check(f"{it}: its live.terms are NOT R35's (the last of three AdamW steps has moved under the dose: the 02:38 case), so a C3 that read live.terms would refuse it",
              clean_i["live"]["terms"] != r35 and any(clean_i["live"]["terms"][u] != r35[u] for u in CK.UNWEIGHTED))
        check(f"{it}: regression 1, C3 reading live.terms instead of live_first (live_first := the last step): refused", any("C3" in x for x in CK.gate_problems(it, itoks, PT.good_selfcheck(gi, "drifted_first"), clean_r)))
        check(f"{it}: regression 3, live_first from another batch: refused and the unweighted term is named", any("C3" in x and "unweighted" in x for x in CK.gate_problems(it, itoks, PT.good_selfcheck(gi, "other_batch_first"), clean_r)))
        check(f"{it}: live_first absent: refused", any("live_first" in x for x in CK.gate_problems(it, itoks, PT.good_selfcheck(gi, "no_first"), clean_r)))
        check(f"{it}: the REF's live_first is from another batch than the IDM's: refused", any("C3" in x for x in CK.gate_problems(it, itoks, clean_i, PT.good_selfcheck(gr, "other_batch_first"))))
        r_bad = PT.good_selfcheck(gr, "ref_term_changed")
        check(f"{rt}: regression 2, a REF record with one term changed: refused, the term named", any("REF" in x and "differ" in x for x in CK.gate_problems(rt, CK.tokens(rt), r_bad)))
        check(f"{it}: no REF gate output at all: the IDM arm is refused", any("no " + rt in x for x in CK.gate_problems(it, itoks, clean_i, None)))
        check(f"{it}: a REF whose live_first is absent refuses the IDM arm", any("live_first" in x for x in CK.gate_problems(it, itoks, clean_i, PT.good_selfcheck(gr, "no_first"))))
        check(f"C3 identity at the numbers: level/L1 difference = 900 x idm/L1 on the stand-in pair", abs((clean_i["live_first"]["wm/level/L1"] - clean_r["live_first"]["wm/level/L1"]) - 900.0 * clean_r["live_first"]["wm/idm/L1"]) <= 1e-5 * 900.0 * clean_r["live_first"]["wm/idm/L1"])
    # ---- weights identity (pure)
    a = PT.model_state(0)
    b = {**{k: v.copy() for k, v in a.items()}, **{k: v + 1.0 for k, v in a.items() if k.startswith("wm.")}}
    ev, pr = CK.weights_identity(a, b)
    check("weights identity: non-WM equal, WM moved -> clean", pr == [] and ev["n_nonwm_equal"] == ev["n_nonwm"] and ev["n_wm_tensors"] > 0)
    for k in [k for k in a if not k.startswith("wm.")]:
        b2 = {kk: vv.copy() for kk, vv in b.items()}
        b2[k] = b2[k] + 1e-6
        check(f"weights identity: one changed value in {k} is refused", CK.weights_identity(a, b2)[1] != [])
    b3 = {k: v for k, v in b.items() if k != "read_s.w"}
    check("weights identity: a missing non-WM tensor is refused", CK.weights_identity(a, b3)[1] != [])
    check("weights identity: a non-WM tensor of another dtype is refused", CK.weights_identity(a, {**b, "e_ego.w": b["e_ego.w"].astype(np.float64)})[1] != [])
    check("weights identity: no WM tensor at all is refused", CK.weights_identity(a, {k: v for k, v in b.items() if not k.startswith("wm.")})[1] != [])
    check("weights identity: an init without non-WM keys is refused", CK.weights_identity({"wm.x": np.zeros(2)}, {"wm.x": np.zeros(2)})[1] != [])
    # ---- C2 (gamma from the log) and C5 (Adam state), both with a positive and a negative control
    def rows(g):
        out = []
        for st_ in (50, 100):
            r = {}
            for L in (1, 2, 3):
                r.update({f"wm/pred/L{L}": 0.2, f"wm/sigreg/L{L}": 3.0, f"wm/rollout/L{L}": 0.25, f"wm/idm/L{L}": 0.11 * L})
                if L >= 2:
                    r[f"wm/act_sigreg/L{L}"] = 8.0
                r[f"wm/level/L{L}"] = 0.2 + 0.25 + 0.24 + g[L - 1] * 0.11 * L + (0.04 if L >= 2 else 0.0)
            out.append(r)
        return out
    for tag_, g, okk in (("IDM_s0", [1000.0, 100.0, 100.0], True), ("IDM_s0", [100.0, 100.0, 100.0], False), ("NULL_s2", [100.0, 100.0, 100.0], True), ("DIV_s0", [1000.0, 100.0, 100.0], False), ("NULL_s3", [100.0, 100.0, 300.0], False)):
        lp = os.path.join(tmp, "gam.jsonl")
        open(lp, "w", encoding="utf-8").write("\n".join(json.dumps(r) for r in rows(g)) + "\n")
        check(f"C2 gamma-from-log {tag_} built {g} -> {'clean' if okk else 'refused'}", (CK.gamma_log_problems(tag_, lp)[1] == []) == okk)
    open(os.path.join(tmp, "gam0.jsonl"), "w").write("{}\n")
    check("C2 gamma-from-log: a log without loss terms is refused", CK.gamma_log_problems("IDM_s0", os.path.join(tmp, "gam0.jsonl"))[1] != [])
    good_opt = {"opt": {"param_groups": [{"params": [0, 1, 2]}], "state": {0: {}, 1: {}, 2: {}}}}
    check("C5 Adam state: every parameter has state -> clean", CK.adam_state_problems(good_opt)[1] == [])
    check("C5 Adam state: a parameter without state is refused", CK.adam_state_problems({"opt": {"param_groups": [{"params": [0, 1, 2]}], "state": {0: {}, 1: {}}}})[1] != [])
    check("C5 Adam state: no optimiser groups is refused", CK.adam_state_problems({"opt": {}})[1] != [])
    # ---- tree needles
    check("NEEDLES name all four patched files", set(CK.NEEDLES) == {CK.TRAINER_REL, CK.MODEL_REL, CK.DATA_REL, CK.WINDOWS_REL})
    # ---- token binding, post check and the cleanup guard on a fake box folder
    root = os.path.join(tmp, "box", "D", "tanitad")
    os.environ.update(ACTSENS_ROOT=root, ACTSENS_COMMIT="a" * 40, ACTSENS_CKPT_LOADER="pickle", ACTSENS_FLOOR_BYTES="1000")
    for rel in ("cache_full/native_tracks.pt", "data/v9/lead_train4572.npz", "data/v9/refcv6_speed_max_v8_train.jsonl", "runs/v8f_mini_full/join_train.npz", "runs/v8f_mini_full/p10/p10_prior.pt"):
        put(os.path.join(root, *rel.split("/")), rel.encode())
    put(os.path.join(root, "data", "v9", "v9_labels_train.npz"), b"v9 labels")
    os.environ["ACTSENS_FAKE_V9_MD5"] = hashlib.md5(b"v9 labels").hexdigest()
    import pickle
    shas = {}
    for s in (0, 1):
        p = os.path.join(root, "runs", "v8f_mini_full", f"M2b_s{s}", "ckpt_last.pt")
        put(p, pickle.dumps({"model": PT.model_state(s)}))
        shas[str(s)] = hashlib.sha256(open(p, "rb").read()).hexdigest()
    json.dump(shas, open(os.path.join(tmp, "init_sha.json"), "w"))
    put(os.path.join(root, "actsens", "repo", "stack", "scripts", "v8f_mini_train.py"), b"trainer v1")
    put(os.path.join(root, "actsens", "tree_manifest.json"), b"{}")
    tag = "DIV_s0"
    line = " ".join(CK.tokens(tag))
    b_, bad_ = CK.binding(tag, line)
    check("binding reads the six data files", bad_ == [] and len(b_["data_md5"]) == 6)
    sc_p = os.path.join(CK.run_dir(tag), "gate", "selfcheck.json")
    put(sc_p, b"{}")
    tok = {"schema": "actsens-gate-token-v1", "written_by": "actsens_check.py gate", "selfcheck_sha256": CK.sha_file(sc_p), **b_}
    put(CK.token_path(tag), json.dumps(tok).encode())
    check("a token bound to the current state is valid", CK.token_ok(tag, line) == [])
    os.environ["ACTSENS_COMMIT"] = "b" * 40
    check("a token for another commit is refused", CK.token_ok(tag, line) != [])
    os.environ["ACTSENS_COMMIT"] = "a" * 40
    check("a token for another argv is refused", CK.token_ok(tag, line + " --x") != [])
    put(os.path.join(root, "actsens", "repo", "stack", "scripts", "v8f_mini_train.py"), b"trainer v2")
    check("a token after the trainer bytes changed is refused", CK.token_ok(tag, line) != [])
    put(os.path.join(root, "actsens", "repo", "stack", "scripts", "v8f_mini_train.py"), b"trainer v1")
    put(os.path.join(root, "data", "v9", "lead_train4572.npz"), b"changed")
    check("a token after a data file changed is refused", CK.token_ok(tag, line) != [])
    put(os.path.join(root, "data", "v9", "lead_train4572.npz"), b"data/v9/lead_train4572.npz")
    put(os.path.join(root, "actsens", "tree_manifest.json"), b'{"x": 1}')
    check("a token after the shipped tree manifest changed is refused", CK.token_ok(tag, line) != [])
    put(os.path.join(root, "actsens", "tree_manifest.json"), b"{}")
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
    check("verify-token refuses a wrong tag", vt("IDM_s0", "a" * 40) == 1)
    check("verify-token refuses a wrong commit", vt(tag, "c" * 40) == 1)
    open(os.path.join(tmp, "arg2.txt"), "w").write(line.replace("--seed 0", "--seed 9") + "\n")
    check("verify-token refuses an argv that is not the frozen one", vt(tag, "a" * 40, os.path.join(tmp, "arg2.txt")) == 1)
    # post check on an out dir written by the stand-in's own format
    out = CK.loc(CK.out_decl(tag))
    toks = line.split(" ")
    m0 = PT.model_state(0)
    fin = {**m0, **{k: v + 1.0 for k, v in m0.items() if k.startswith("wm.")}}
    good_opt3 = {"param_groups": [{"params": [0, 1, 2]}], "state": {0: {}, 1: {}, 2: {}}}
    ck_ok = {"model": fin, "step": CK.RUN_STEPS, "opt": good_opt3}
    summ = {"done": True, "steps": CK.RUN_STEPS, "argv_sha": CK.argv_sha16(toks)}

    def write_out(ck, sm):
        put(os.path.join(out, "ckpt_last.pt"), pickle.dumps(ck))
        put(os.path.join(out, "summary.json"), json.dumps(sm).encode())
        put(os.path.join(out, "train_log.jsonl"), ("\n".join(json.dumps(r) for r in rows([100.0, 100.0, 100.0])) + "\n").encode())
    write_out(ck_ok, summ)
    check("post check: a correct finished run passes", CK.post_problems(tag, toks, out)[1] == [])
    for name, ck_m, sm_m in (("steps short", ck_ok, dict(summ, steps=100)), ("summary argv_sha", ck_ok, dict(summ, argv_sha="0" * 16)),
                             ("ckpt step", dict(ck_ok, step=5), summ),
                             ("frozen tensor moved", dict(ck_ok, model={**fin, "e_ego.w": fin["e_ego.w"] + 1}), summ),
                             ("an optimiser parameter without Adam state", dict(ck_ok, opt={"param_groups": [{"params": [0, 1, 2]}], "state": {0: {}, 1: {}}}), summ),
                             ("no WM tensor", dict(ck_ok, model={k: v for k, v in fin.items() if not k.startswith("wm.")}), summ)):
        write_out(ck_m, sm_m)
        check(f"post check: planted '{name}' is refused", CK.post_problems(tag, toks, out)[1] != [])
    write_out(ck_ok, summ)
    put(os.path.join(out, "train_log.jsonl"), ("\n".join(json.dumps(r) for r in rows([1000.0, 100.0, 100.0])) + "\n").encode())
    check("post check: a DIV run whose log shows gamma 1000 at level 1 is refused (gamma-from-log)", any("gamma-from-log" in x for x in CK.post_problems(tag, toks, out)[1]))
    write_out(ck_ok, summ)
    ptoks = CK.tokens(CK.PILOT_TAG)
    pout = CK.loc(CK.out_decl(CK.PILOT_TAG))
    put(os.path.join(pout, "summary.json"), json.dumps({"done": True, "steps": CK.PILOT_STEPS, "argv_sha": CK.argv_sha16(ptoks)}).encode())
    check("post check: a correct pilot summary passes (no checkpoint is read)", CK.post_problems(CK.PILOT_TAG, ptoks, pout)[1] == [])
    os.remove(os.path.join(out, "ckpt_last.pt"))
    check("post check: no checkpoint is refused", CK.post_problems(tag, toks, out)[1] != [])
    # cleanup guard
    gout = CK.loc(CK.gate_out_decl(tag))
    put(os.path.join(gout, "selfcheck.json"), b"x")
    put(os.path.join(out, "ckpt_last.pt"), b"ckpt")
    check("cleanup without bank proof refuses and deletes nothing", CK.cmd_cleanup(tag, "G1") == 6 and os.path.exists(gout))
    put(os.path.join(CK.run_dir(tag), "BANK_VERIFIED_G1"), b"ok")
    check("cleanup with bank proof deletes the gate dir, logs it, and keeps the checkpoint", CK.cmd_cleanup(tag, "G1") == 0 and not os.path.exists(gout) and os.path.exists(os.path.join(out, "ckpt_last.pt"))
          and json.load(open(os.path.join(CK.run_dir(tag), "CLEANUP.json")))["deleted_own_temp"][0]["path"] == gout)
    check("rm_own_temp refuses a dir whose parent is not the named one", CK.rm_own_temp(out, "gate_out") is None and os.path.isdir(out))
    # ---- pilot timing and the 72 h rule
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
    check("1.0 s/step: 6 runs x 19,164 steps = 31.9 h", abs(pr1["projected_total_h_planned"] - 114984 / 3600) < 1e-9 and pr1["within_limit_planned"] and "projected_total_h_with_token_arms" not in pr1)
    check("3.5 s/step is over 72 h", not CK.project(3.5)["within_limit_planned"])

    def pilot_json(mean, commit="a" * 40):
        pj = {"commit": commit, "projection": CK.project(mean)}
        p = os.path.join(tmp, f"pj_{mean}_{commit[:1]}.json")
        open(p, "w").write(json.dumps(pj))
        return p
    check("permit: within 72 h passes", CK.cmd_permit({"pilot": pilot_json(1.0), "commit": "a" * 40}) == 0)
    p6 = pilot_json(3.5)
    check("permit: over 72 h without an amendment is refused (3)", CK.cmd_permit({"pilot": p6, "commit": "a" * 40}) == 3)
    check("permit: a pilot of another commit is refused", CK.cmd_permit({"pilot": pilot_json(1.0, "b" * 40), "commit": "a" * 40}) == 3)
    am = os.path.join(tmp, "am.txt")
    open(am, "w").write(f"pilot_sha256={'0' * 64} commit={'a' * 40}\n")
    check("permit: an amendment naming another pilot is refused", CK.cmd_permit({"pilot": p6, "commit": "a" * 40, "amendment": am}) == 3)
    open(am, "w").write(f"pilot_sha256={CK.sha_file(p6)} commit={'a' * 40}\n")
    check("permit: an amendment binding this pilot and commit is accepted", CK.cmd_permit({"pilot": p6, "commit": "a" * 40, "amendment": am}) == 0)
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
            if re.search(rb"\btext\s*=\s*True", raw) and not f.startswith("test_actsens_check"):
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
