"""Unit tests of wmpool_check.py (pure Python; numpy only for the weights identity; no torch, no GPU, no network, no model, no checkpoint).
Run:  C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I tests/test_wmpool_check.py <scratch dir>     -> prints  UNIT ok=<n> bad=<m>  (exit 1 on any bad)
Every gate assertion has a planted-bad selfcheck that must be refused; every clean selfcheck must pass; every refusal of the cells stage has a planted-bad input."""
import ast
import copy
import hashlib
import json
import os
import pickle
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, PKG)
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import wmpool_check as CK  # noqa: E402
import plant_trainer as PT  # noqa: E402

N_OK, N_BAD = 0, 0
BS = chr(92)


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
    os.environ["WMPOOL_FAKE_INIT_SHA_FILE"] = os.path.join(TMP, "init_sha.json")
    return toks, PT.good_selfcheck(gt)


def put(p, data=b""):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "wb").write(data)


def main(tmp):
    global TMP
    TMP = tmp
    os.makedirs(tmp, exist_ok=True)
    json.dump({"0": CK.INIT_SHA[0], "1": CK.INIT_SHA[1]}, open(os.path.join(tmp, "init_sha.json"), "w"))
    runnable = CK.RUN_ORDER + [CK.PILOT_TAG]
    # ---- the card's order
    check("run order is P-ATT_s0, P-ATT_s1, P-ATT_rep_s2", CK.RUN_ORDER == ["P-ATT_s0", "P-ATT_s1", "P-ATT_rep_s2"])
    check("no blocked tag", all(not v["blocked"] for v in CK.TABLE.values()))
    check("steps: 19,164 full; pilot 200", CK.RUN_STEPS == 19164 and CK.PILOT_STEPS == 200)
    check("the pilot gates all three full-length argvs", CK.GATE_AT_PILOT == CK.RUN_ORDER)
    check("pilot tag is P-ATT_s0's", CK.PILOT_TAG == "PILOT_P-ATT_s0" and CK.PILOT_BASE == "P-ATT_s0")
    # ---- argv pin
    for t in CK.TABLE:
        check(f"frozen argv of {t} is accepted", CK.audit_line(t, " ".join(CK.tokens(t))) == [])
        tk = CK.tokens(t)
        g = lambda f: tk[tk.index(f) + 1]                                                  # noqa: E731
        init = CK.TABLE[t]["init"]
        check(f"{t}: --wm-pool att25 is named, --seed {CK.TABLE[t]['seed']}, init and CELL cache of M2b_s{init}",
              g("--wm-pool") == "att25" and g("--seed") == str(CK.TABLE[t]["seed"]) and f"M2b_s{init}" in g("--init-ckpt") and g("--wm-zcache") == CK.ZC_DECL + BS + f"M2b_s{init}" + BS + "train")
        check(f"{t}: the run names --wm-zcache, --wm-only and no --wm-speed", "--wm-zcache" in tk and "--wm-only" in tk and "--wm-speed" not in tk)
        check(f"{t}: --out under D:{BS}tanitad{BS}runs{BS}v8f_wmpool", g("--out").lower().startswith(CK.DECL_D.lower() + BS + "runs" + BS + "v8f_wmpool" + BS))
    pt = CK.tokens(CK.PILOT_TAG)
    check("pilot argv = P-ATT_s0 argv with --out moved and --steps 200 appended", pt[:-2] == [CK.out_decl(CK.PILOT_TAG) if x == CK.out_decl("P-ATT_s0") else x for x in CK.tokens("P-ATT_s0")] and pt[-2:] == ["--steps", "200"])
    check("pilot out dir differs from the run out dir", CK.out_decl(CK.PILOT_TAG) != CK.out_decl("P-ATT_s0"))
    good = CK.tokens("P-ATT_s0")
    i = good.index("--wm-pool") + 1
    bad = good[:i] + ["mean"] + good[i + 1:]
    r = CK.audit_line("P-ATT_s0", " ".join(bad))
    check("a changed --wm-pool is refused and the first differing token is named", len(r) == 1 and f"token {i} is 'mean'" in r[0])
    check("--wm-pool dropped from the argv is refused", CK.audit_line("P-ATT_s0", " ".join(good[:-2])) != [])
    check("a changed seed is refused", CK.audit_line("P-ATT_s0", " ".join(good).replace("--seed 0", "--seed 3")) != [])
    check("the s1 argv shipped for s0 is refused", CK.audit_line("P-ATT_s0", " ".join(CK.tokens("P-ATT_s1"))) != [])
    check("the token-mean cache path in place of the cell cache is refused", CK.audit_line("P-ATT_s0", " ".join(good).replace("zcells", "zcache")) != [])
    check("an extra token is refused", CK.audit_line("P-ATT_s0", " ".join(good) + " --steps 5") != [])
    check("rep_s2 is s0 apart from the seed and --out", [(a, b) for a, b in zip(CK.tokens("P-ATT_rep_s2"), CK.tokens("P-ATT_s0")) if a != b] == [("2", "0"), (CK.out_decl("P-ATT_rep_s2"), CK.out_decl("P-ATT_s0"))])
    check("the frozen argv file hashes to the pin", CK.sha_file(CK.ARGVS_FILE) == CK.ARGVS_SHA256)
    orig = os.path.join(PKG, "..", "argvs_wmpool.txt")
    if os.path.exists(orig):
        rows = {l.split("\t")[0]: l.rstrip("\r\n").split("\t")[1] for l in open(orig, encoding="utf-8") if l.strip() and not l.startswith("#")}
        check("the packaged argvs equal theory/wmpool/argvs_wmpool.txt token for token", set(rows) == set(CK.TABLE) and all(" ".join(CK.tokens(t)) == rows[t] for t in CK.TABLE))
    # ---- a clean selfcheck passes, per run tag
    for t in runnable:
        toks, sc = sc_of(t)
        check(f"clean selfcheck passes {t}", CK.gate_problems(t, toks, sc) == [])
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
        # the pool: declared vs BUILT (levers.wm_pool, levers.wm_pool_readback)
        "pool lever absent": lambda s: s["levers"].pop("wm_pool"),
        "pool lever mean": lambda s: s["levers"].__setitem__("wm_pool", "mean"),
        "pool readback absent": lambda s: s["levers"].pop("wm_pool_readback"),
        "pool readback declared": lambda s: s["levers"]["wm_pool_readback"].__setitem__("declared", "mean"),
        "pool built mode is mean": lambda s: s["levers"]["wm_pool_readback"]["built"].__setitem__("mode", "mean"),
        "pool built n_params": lambda s: s["levers"]["wm_pool_readback"]["built"].__setitem__("n_params", 512),
        "pool built q_shape": lambda s: s["levers"]["wm_pool_readback"]["built"].__setitem__("q_shape", [10]),
        "pool query not zero at build": lambda s: s["levers"]["wm_pool_readback"]["built"].__setitem__("q_zero_at_build", False),
        "pool n_cells": lambda s: s["levers"]["wm_pool_readback"]["built"].__setitem__("n_cells", 40),
        "pool cell map": lambda s: s["levers"]["wm_pool_readback"]["built"].__setitem__("cell_map_ok", False),
        "pool weights not uniform at build": lambda s: s["levers"]["wm_pool_readback"]["built"].__setitem__("uniform_at_build", False),
        "pool does not attend under a probe query": lambda s: s["levers"]["wm_pool_readback"]["built"].__setitem__("attends_when_set", False),
        "pool built record absent": lambda s: s["levers"]["wm_pool_readback"].pop("built"),
        "pool dataset serves the mean": lambda s: s["levers"]["wm_pool_readback"].__setitem__("dataset_pool", "mean"),
        "pool dataset cell shape": lambda s: s["levers"]["wm_pool_readback"].__setitem__("dataset_zcell_shape", [91, 40, 256]),
        "pool cell shape absent": lambda s: s["levers"]["wm_pool_readback"].pop("dataset_zcell_shape"),
        "pool cache problem (adapter sha)": lambda s: s["levers"]["wm_pool_readback"].__setitem__("cache_problems", ["adapter sha differs"]),
        "pool cache problems absent": lambda s: s["levers"]["wm_pool_readback"].pop("cache_problems"),
        "pool readback names the token-mean cache": lambda s: s["levers"]["wm_pool_readback"].__setitem__("zcache", r"D:\tanitad\zcache\M2b_s0\train"),
        "ladder readback absent": lambda s: s["levers"].pop("wm_ladder_readback"),
        "rollout k": lambda s: s["levers"]["wm_ladder_readback"]["built"].__setitem__("k", [10, 10, 1]),
        "context": lambda s: s["levers"]["wm_ladder_readback"]["built"].__setitem__("ctx_built", [3, 3, 2]),
        "cum stride": lambda s: s["levers"]["wm_ladder_readback"]["built"].__setitem__("cum_stride", [1, 2, 4]),
        "k-step weight": lambda s: s["levers"]["wm_ladder_readback"]["built"].__setitem__("w_in_loss", [1.0, 0.5, 1.0]),
        "windows kind": lambda s: s["levers"]["wm_ladder_readback"]["windows_built"].__setitem__("kind", "dense"),
        "windows not pooled": lambda s: s["levers"]["wm_ladder_readback"]["windows_built"].__setitem__("pooled", False),
        "cache of the other seed": lambda s: s["levers"]["wm_ladder_readback"]["windows_built"].__setitem__("zcache", r"D:\tanitad\zcells\M2b_s7\train"),
        "token-mean cache on the built dataset": lambda s: s["levers"]["wm_ladder_readback"]["windows_built"].__setitem__("zcache", r"D:\tanitad\zcache\M2b_s0\train"),
        "cache missing clips": lambda s: s["levers"]["wm_ladder_readback"]["windows_built"].__setitem__("zcache_missing", 4),
        "no windows": lambda s: s["levers"]["wm_ladder_readback"]["windows_built"].__setitem__("n_windows", 0),
        "history K": lambda s: s["levers"].__setitem__("wm_k", 1),
        "strides": lambda s: s["levers"].__setitem__("wm_strides", [2, 2]),
        "wm_only not built": lambda s: s["levers"].__setitem__("wm_only", {"wm_only": False}),
        "WM parameter count without the query": lambda s: s["levers"]["wm_only"].__setitem__("n_wm", 2143298),
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
            check(f"{t}: planted '{name}' is refused", len(CK.gate_problems(t, toks, s2)) >= 1)
    check("the planted mutations ran on every run tag (>= 55 each)", n_ref >= 55 * len(runnable))
    toks, sc = sc_of("P-ATT_s0")
    sc["levers"]["wm_pool_readback"]["built"]["mode"] = "mean"
    check("the read-back names the pool it found (levers.wm_pool_readback.built.mode) in the refusal", any("wm_pool_readback.built.mode" in x for x in CK.gate_problems("P-ATT_s0", toks, sc)))
    # ---- weights identity (pure)
    a = PT.model_state(0)
    b = {**{k: v.copy() for k, v in a.items()}, **{k: v + 1.0 for k, v in a.items() if k.startswith("wm.")}, "wm.pool_q": np.full(4, 0.01, dtype=np.float32)}
    ev, pr = CK.weights_identity(a, b)
    check("weights identity: non-WM equal, WM moved, the pool query non-zero -> clean", pr == [] and ev["n_nonwm_equal"] == ev["n_nonwm"] and ev["n_pool_q_nonzero"] == 1)
    for k in [k for k in a if not k.startswith("wm.")]:
        b2 = {kk: vv.copy() for kk, vv in b.items()}
        b2[k] = b2[k] + 1e-6
        check(f"weights identity: one changed value in {k} is refused", CK.weights_identity(a, b2)[1] != [])
    check("weights identity: a missing non-WM tensor is refused", CK.weights_identity(a, {k: v for k, v in b.items() if k != "read_s.w"})[1] != [])
    check("weights identity: a non-WM tensor of another dtype is refused", CK.weights_identity(a, {**b, "e_ego.w": b["e_ego.w"].astype(np.float64)})[1] != [])
    check("weights identity: the pool query still zero is refused", CK.weights_identity(a, {**b, "wm.pool_q": np.zeros(4, dtype=np.float32)})[1] != [])
    check("weights identity: no pool query in the checkpoint is refused", CK.weights_identity(a, {k: v for k, v in b.items() if k != "wm.pool_q"})[1] != [])
    check("weights identity: a second pool query tensor is refused", CK.weights_identity(a, {**b, "wm.x.pool_q": np.ones(4, dtype=np.float32)})[1] != [])
    check("weights identity: no WM tensor at all is refused", CK.weights_identity(a, {k: v for k, v in b.items() if not k.startswith("wm.")})[1] != [])
    check("weights identity: an init without non-WM keys is refused", CK.weights_identity({"wm.x": np.zeros(2)}, {"wm.x": np.zeros(2)})[1] != [])
    # ---- tree needles
    check("NEEDLES name the trainer, model and windows files", set(CK.NEEDLES) == {CK.TRAINER_REL, CK.MODEL_REL, CK.WINDOWS_REL})
    # ---- token binding, post check and the cleanup guard on a fake box folder
    root = os.path.join(tmp, "box", "D", "tanitad")
    os.environ.update(WMPOOL_ROOT=root, WMPOOL_COMMIT="a" * 40, WMPOOL_CKPT_LOADER="pickle", WMPOOL_FLOOR_BYTES="1000")
    for rel in ("cache_full/native_tracks.pt", "data/v9/lead_train4572.npz", "data/v9/refcv6_speed_max_v8_train.jsonl", "runs/v8f_mini_full/join_train.npz", "runs/v8f_mini_full/p10/p10_prior.pt"):
        put(os.path.join(root, *rel.split("/")), rel.encode())
    put(os.path.join(root, "data", "v9", "v9_labels_train.npz"), b"v9 labels")
    os.environ["WMPOOL_FAKE_V9_MD5"] = hashlib.md5(b"v9 labels").hexdigest()
    shas = {}
    for s in (0, 1):
        p = os.path.join(root, "runs", "v8f_mini_full", f"M2b_s{s}", "ckpt_last.pt")
        put(p, pickle.dumps({"model": PT.model_state(s)}))
        shas[str(s)] = hashlib.sha256(open(p, "rb").read()).hexdigest()
    json.dump(shas, open(os.path.join(tmp, "init_sha.json"), "w"))
    put(os.path.join(root, "wmpool", "repo", "stack", "scripts", "v8f_mini_train.py"), b"trainer v1")
    put(os.path.join(root, "wmpool", "tree_manifest.json"), b"{}")
    tag = "P-ATT_s0"
    line = " ".join(CK.tokens(tag))
    b_, bad_ = CK.binding(tag, line)
    check("binding reads the six data files", bad_ == [] and len(b_["data_md5"]) == 6)
    sc_p = os.path.join(CK.run_dir(tag), "gate", "selfcheck.json")
    put(sc_p, b"{}")
    tok = {"schema": "wmpool-gate-token-v1", "written_by": "wmpool_check.py gate", "selfcheck_sha256": CK.sha_file(sc_p), **b_}
    put(CK.token_path(tag), json.dumps(tok).encode())
    check("a token bound to the current state is valid", CK.token_ok(tag, line) == [])
    os.environ["WMPOOL_COMMIT"] = "b" * 40
    check("a token for another commit is refused", CK.token_ok(tag, line) != [])
    os.environ["WMPOOL_COMMIT"] = "a" * 40
    check("a token for another argv is refused", CK.token_ok(tag, line + " --x") != [])
    put(os.path.join(root, "wmpool", "repo", "stack", "scripts", "v8f_mini_train.py"), b"trainer v2")
    check("a token after the trainer bytes changed is refused", CK.token_ok(tag, line) != [])
    put(os.path.join(root, "wmpool", "repo", "stack", "scripts", "v8f_mini_train.py"), b"trainer v1")
    put(os.path.join(root, "data", "v9", "lead_train4572.npz"), b"changed")
    check("a token after a data file changed is refused", CK.token_ok(tag, line) != [])
    put(os.path.join(root, "data", "v9", "lead_train4572.npz"), b"data/v9/lead_train4572.npz")
    put(os.path.join(root, "wmpool", "tree_manifest.json"), b'{"x": 1}')
    check("a token after the shipped tree manifest changed is refused", CK.token_ok(tag, line) != [])
    put(os.path.join(root, "wmpool", "tree_manifest.json"), b"{}")
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
    check("verify-token refuses a wrong tag", vt("P-ATT_s1", "a" * 40) == 1)
    check("verify-token refuses a wrong commit", vt(tag, "c" * 40) == 1)
    open(os.path.join(tmp, "arg2.txt"), "w").write(line.replace("--seed 0", "--seed 9") + "\n")
    check("verify-token refuses an argv that is not the frozen one", vt(tag, "a" * 40, os.path.join(tmp, "arg2.txt")) == 1)
    # post check on an out dir written by the stand-in's own format
    out = CK.loc(CK.out_decl(tag))
    toks = line.split(" ")
    m0 = PT.model_state(0)
    fin = {**m0, **{k: v + 1.0 for k, v in m0.items() if k.startswith("wm.")}, "wm.pool_q": np.full(4, 0.01, dtype=np.float32)}
    ck_ok = {"model": fin, "step": CK.RUN_STEPS}
    summ = {"done": True, "steps": CK.RUN_STEPS, "argv_sha": CK.argv_sha16(toks)}

    def write_out(ck, sm):
        put(os.path.join(out, "ckpt_last.pt"), pickle.dumps(ck))
        put(os.path.join(out, "summary.json"), json.dumps(sm).encode())
    write_out(ck_ok, summ)
    check("post check: a correct finished run passes", CK.post_problems(tag, toks, out)[1] == [])
    for name, ck_m, sm_m in (("steps short", ck_ok, dict(summ, steps=100)), ("summary argv_sha", ck_ok, dict(summ, argv_sha="0" * 16)),
                             ("ckpt step", dict(ck_ok, step=5), summ),
                             ("frozen tensor moved", dict(ck_ok, model={**fin, "e_ego.w": fin["e_ego.w"] + 1}), summ),
                             ("pool query still zero", dict(ck_ok, model={**fin, "wm.pool_q": np.zeros(4, dtype=np.float32)}), summ),
                             ("no pool query", dict(ck_ok, model={k: v for k, v in fin.items() if k != "wm.pool_q"}), summ),
                             ("no WM tensor", dict(ck_ok, model={k: v for k, v in fin.items() if not k.startswith("wm.")}), summ)):
        write_out(ck_m, sm_m)
        check(f"post check: planted '{name}' is refused", CK.post_problems(tag, toks, out)[1] != [])
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
    check("the cell cache dir is never a cleanup target", all("zcells" not in x["path"] for x in json.load(open(os.path.join(CK.run_dir(tag), "CLEANUP.json")))["deleted_own_temp"]))
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
    check("1.0 s/step: 3 runs x 19,164 steps = 16.0 h", pr1["n_runs_planned"] == 3 and pr1["steps_total"] == 57492 and abs(pr1["projected_total_h_planned"] - 57492 / 3600) < 1e-9 and pr1["within_limit_planned"])
    check("4.7 s/step is over 72 h (3 runs)", not CK.project(4.7)["within_limit_planned"] and CK.project(4.0)["within_limit_planned"])

    def pilot_json(mean, commit="a" * 40):
        pj = {"commit": commit, "projection": CK.project(mean)}
        p = os.path.join(tmp, f"pj_{mean}_{commit[:1]}.json")
        open(p, "w").write(json.dumps(pj))
        return p
    check("permit: within 72 h passes", CK.cmd_permit({"pilot": pilot_json(1.0), "commit": "a" * 40}) == 0)
    p6 = pilot_json(4.7)
    check("permit: over 72 h without an amendment is refused (3)", CK.cmd_permit({"pilot": p6, "commit": "a" * 40}) == 3)
    check("permit: a pilot of another commit is refused", CK.cmd_permit({"pilot": pilot_json(1.0, "b" * 40), "commit": "a" * 40}) == 3)
    am = os.path.join(tmp, "am.txt")
    open(am, "w").write(f"pilot_sha256={'0' * 64} commit={'a' * 40}\n")
    check("permit: an amendment naming another pilot is refused", CK.cmd_permit({"pilot": p6, "commit": "a" * 40, "amendment": am}) == 3)
    open(am, "w").write(f"pilot_sha256={CK.sha_file(p6)} commit={'a' * 40}\n")
    check("permit: an amendment binding this pilot and commit is accepted", CK.cmd_permit({"pilot": p6, "commit": "a" * 40, "amendment": am}) == 0)

    # ---- the cells stage: check3 (card control 3), counts, disk, verify-cells, the pins
    good3 = {"tol_identity_fp32": 1e-5, "clips": {f"eval/c{j}/s{k}": {} for j in range(8) for k in (0, 1)}, "max_identity_fp32": 4e-7, "max_stored_vs_cached": 1.2e-4, "min_wrongseed_diff": 0.3, "identity_pass": True, "wrongseed_fails_identity": True}
    check("check3: a clean result passes", CK.cells_check3_problems(good3) == [])
    for name, f in {"identity failed": lambda r: r.update(identity_pass=False, max_identity_fp32=0.5), "wrong-seed adapter passes the identity": lambda r: r.update(wrongseed_fails_identity=False),
                    "stored cells far from the cached token mean": lambda r: r.update(max_stored_vs_cached=0.2), "stored-vs-cached not a number": lambda r: r.update(max_stored_vs_cached=None),
                    "fewer than 8 rows": lambda r: r.update(clips={"a": {}}), "identity key absent": lambda r: r.pop("identity_pass")}.items():
        r2 = copy.deepcopy(good3)
        f(r2)
        check(f"check3: planted '{name}' is refused", CK.cells_check3_problems(r2) != [])
    check("check3: a non-dict result is refused", CK.cells_check3_problems(None) != [])
    check("check3: wrongseed None (one seed only) is tolerated", CK.cells_check3_problems(dict(good3, wrongseed_fails_identity=None)) == [])
    need = CK.cells_need_gb()
    check("cells need cache + 1 GB: 2 x 2.36 + 1 = 5.72 GB and the dev-box stage asks 6", abs(need - 5.72) < 1e-9 and re.search(r"NEED_GB_CELLS=6\b", open(os.path.join(PKG, "stages", "wmpool_common.sh")).read()) is not None)
    check("a run asks 1 GB in both files", CK.NEED_GB_RUN == 1 and re.search(r"NEED_GB_RUN=1\b", open(os.path.join(PKG, "stages", "wmpool_common.sh")).read()) is not None)
    os.environ["WMPOOL_DRY"] = "1"
    os.environ["WMPOOL_FAKE_FREE_GB"] = "5.5"
    check("cells disk: 5.5 GB free (< cache + 1 GB) is refused", len(CK.cells_disk_problems()) == 1)
    os.environ["WMPOOL_FAKE_FREE_GB"] = "5.8"
    check("cells disk: 5.8 GB free passes", CK.cells_disk_problems() == [])
    del os.environ["WMPOOL_DRY"]
    del os.environ["WMPOOL_FAKE_FREE_GB"]
    os.environ["WMPOOL_TEST_CELL_COUNTS"] = "train=3,eval=2"
    check("the count hook is ignored outside a dry run (4572 / 141)", CK.cell_counts() == {"train": 4572, "eval": 141})
    os.environ["WMPOOL_DRY"] = "1"
    check("the count hook works in a dry run", CK.cell_counts() == {"train": 3, "eval": 2})
    croot = os.path.join(tmp, "cells_root")

    def build_cells(short=None, tmpfile=False, noman=False):
        import shutil
        shutil.rmtree(croot, ignore_errors=True)
        for k in (0, 1):
            for sp, n in {"train": 3, "eval": 2}.items():
                d = os.path.join(croot, f"M2b_s{k}", sp)
                for j in range(n - (1 if short == (k, sp) else 0)):
                    put(os.path.join(d, f"c{j}.npz"), b"x")
                if tmpfile and (k, sp) == (0, "eval"):
                    put(os.path.join(d, "c9.npz.tmp"), b"x")
                if not (noman and (k, sp) == (1, "eval")):
                    put(os.path.join(d, "CELL_MANIFEST.json"), json.dumps({"adapter_state_sha256": hashlib.sha256(f"a{k}".encode()).hexdigest(), "n_files": n}).encode())
    build_cells()
    f_, p_ = CK.cells_count_problems(croot)
    check("cells counts: a complete cache passes", p_ == [] and f_ == {"s0/train": 3, "s0/eval": 2, "s1/train": 3, "s1/eval": 2})
    for name, kw in (("one clip file short", {"short": (1, "train")}), ("a stray .tmp file", {"tmpfile": True}), ("a missing CELL_MANIFEST.json", {"noman": True})):
        build_cells(**kw)
        check(f"cells counts: planted '{name}' is refused", CK.cells_count_problems(croot)[1] != [])
    build_cells()
    del os.environ["WMPOOL_TEST_CELL_COUNTS"]
    check("cells counts: with the real counts (4572 / 141) a 3-file cache is refused", CK.cells_count_problems(croot)[1] != [])
    # verify-cells on the dev box over a pulled record
    os.environ["WMPOOL_TEST_CELL_COUNTS"] = "train=3,eval=2"
    mans = {f"s{k}/{sp}": {"adapter_state_sha256": hashlib.sha256(f"a{k}".encode()).hexdigest(), "n_files": n} for k in (0, 1) for sp, n in {"train": 3, "eval": 2}.items()}
    c3p = os.path.join(tmp, "c3.json")
    open(c3p, "w").write(json.dumps(good3))
    rec = {"commit": "a" * 40, "counts": {"s0/train": 3, "s0/eval": 2, "s1/train": 3, "s1/eval": 2}, "expected_counts": {"train": 3, "eval": 2}, "manifests": mans, "check3_sha256": CK.sha_file(c3p)}
    cjp = os.path.join(tmp, "cells.json")

    def vc(r, c3=c3p, commit="a" * 40):
        open(cjp, "w").write(json.dumps(r))
        return CK.cmd_verify_cells({"cells-json": cjp, "check3-json": c3, "commit": commit})
    check("verify-cells: a correct record passes", vc(rec) == 0)
    check("verify-cells: another commit is refused", vc(rec, commit="b" * 40) == 11)
    check("verify-cells: a check3 file that differs from the recorded sha is refused", vc(dict(rec, check3_sha256="0" * 64)) == 11)
    check("verify-cells: counts below the expected are refused", vc(dict(rec, counts=dict(rec["counts"], **{"s1/train": 2}))) == 11)
    check("verify-cells: the two seeds with the same adapter sha (a wrong-seed cache) are refused", vc(dict(rec, manifests={k: dict(v, adapter_state_sha256=hashlib.sha256(b"a0").hexdigest()) for k, v in mans.items()})) == 11)
    check("verify-cells: train and eval of one seed with different adapter shas are refused", vc(dict(rec, manifests=dict(mans, **{"s0/eval": dict(mans["s0/eval"], adapter_state_sha256="f" * 64)}))) == 11)
    bad3 = os.path.join(tmp, "c3bad.json")
    open(bad3, "w").write(json.dumps(dict(good3, identity_pass=False)))
    check("verify-cells: a failed check3 is refused even when the record is clean", vc(dict(rec, check3_sha256=CK.sha_file(bad3)), c3=bad3) == 11)
    check("verify-cells: an unreadable record is refused", CK.cmd_verify_cells({"cells-json": os.path.join(tmp, "nope.json"), "check3-json": c3p, "commit": "a" * 40}) == 11)
    del os.environ["WMPOOL_DRY"], os.environ["WMPOOL_TEST_CELL_COUNTS"]
    # the shipped cell code is pinned and its CLI is what the cells job calls (read by ast: torch is never imported here)
    for f, want in CK.CELLCODE_SHA256.items():
        check(f"CELLCODE_SHA256 pins the shipped frozen/{f}", CK.sha_file(os.path.join(PKG, "frozen", f)) == want)
    src_zc = os.path.join(PKG, "..", "zcells.py")
    if os.path.exists(src_zc):
        check("frozen/zcells.py is byte-identical to theory/wmpool/zcells.py", CK.sha_file(src_zc) == CK.sha_file(os.path.join(PKG, "frozen", "zcells.py")))
    tree = ast.parse(open(os.path.join(PKG, "frozen", "zcells.py"), encoding="utf-8").read())
    flags, modes = set(), set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "add_argument" and n.args and isinstance(n.args[0], ast.Constant):
            if str(n.args[0].value).startswith("--"):
                flags.add(n.args[0].value)
            else:
                for kw in n.keywords:
                    if kw.arg == "choices":
                        modes = {e.value for e in kw.value.elts}
    check("zcells.py offers build, manifest and check3", modes == {"build", "manifest", "check3"})
    check("every flag the cells job passes exists in zcells.py", {"--ckpt", "--out", "--commit", "--clips-file", "--zmean-dir", "--out-json"} <= flags)
    jb = open(os.path.join(PKG, "wmpool_check.py"), encoding="utf-8").read()
    used = set(re.findall(r'"(--[a-z0-9-]+)"', jb[jb.index("def cmd_cells"):jb.index("def cmd_cells") + 3500])) & {"--ckpt", "--out", "--commit", "--clips-file", "--zmean-dir", "--out-json", "--splits", "--root", "--tracks", "--limit"}
    check("the flags cmd_cells uses are all known to zcells.py", used <= flags and len(used) >= 5)
    check("check3 reads the 8-clip list of the ladder (8 clips)", sum(len(v) for v in json.load(open(os.path.join(PKG, "frozen", "clips8.json"))).values()) == 8)
    check("the cell cache path in the argvs is the one zcells.py writes (M2b_s{k}/train under ZC_DECL)", all(CK.tokens(t)[CK.tokens(t).index("--wm-zcache") + 1] == CK.ZC_DECL + BS + f"M2b_s{CK.TABLE[t]['init']}" + BS + "train" for t in CK.TABLE))
    check("the GPU idle override defaults to 3500 MiB", "WMPOOL_GPU_IDLE_MIB:-3500" in open(os.path.join(PKG, "stages", "wmpool_common.sh")).read())
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
            if re.search(rb"\btext\s*=\s*True", raw) and not f.startswith("test_wmpool_check"):
                bad_files.append(f"bare text=True in {f}")
            if re.search(rb"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", raw) and not f.endswith(".json") and f not in ("PACKAGE_MANIFEST.sha256",):
                bad_files.append(f"IP-like string in {f}")
            if re.match(r"(?i)^(aux|con|prn|nul)(\.|$)", f):
                bad_files.append(f"reserved name {f}")
    check("package hygiene (LF scripts, no bare text=True, no IP, no reserved names): " + "; ".join(bad_files), bad_files == [])
    print(f"UNIT ok={N_OK} bad={N_BAD}")
    return 0 if N_BAD == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
