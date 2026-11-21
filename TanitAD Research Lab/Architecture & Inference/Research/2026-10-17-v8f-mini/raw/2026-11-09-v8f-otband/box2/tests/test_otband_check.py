"""Unit tests of otband_check.py (pure Python; numpy only for the freeze identity; no torch, no GPU, no network).
Run:  C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I tests/test_otband_check.py <scratch dir>     -> prints  UNIT ok=<n> bad=<m>  (exit 1 on any bad)
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
import otband_check as CK  # noqa: E402
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
    os.environ["OTBAND_FAKE_INIT_SHA_FILE"] = os.path.join(TMP, "init_sha.json")
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
    # ---- the card's order and the data-change property of the table
    check("run order is CTL_s0, BAND_s0, CTL_s1, BAND_s1, BAND_rep", CK.RUN_ORDER == ["OB-CTL_s0", "OB-BAND_s0", "OB-CTL_s1", "OB-BAND_s1", "OB-BAND_rep"])
    check("the token arms are in the table, blocked, and not in the run order", all(CK.TABLE[t]["blocked"] and t not in CK.RUN_ORDER for t in ("OB-BAND-TOK_s0", "OB-BAND-TOK_s1")))
    check("steps: 4791 x 2 = 9,582; pilot 200", CK.RUN_STEPS == 9582 and CK.PILOT_STEPS == 200)
    # ---- argv pin
    for t in CK.TABLE:
        check(f"frozen argv of {t} is accepted", CK.audit_line(t, " ".join(CK.tokens(t))) == [])
    check("pilot argv = OB-BAND_s0 argv with --out moved and --steps 200 appended", CK.tokens(CK.PILOT_TAG)[:-2] == [CK.out_decl(CK.PILOT_TAG) if x == CK.out_decl("OB-BAND_s0") else x for x in CK.tokens("OB-BAND_s0")] and CK.tokens(CK.PILOT_TAG)[-2:] == ["--steps", "200"])
    check("pilot out dir differs from the run out dir", CK.out_decl(CK.PILOT_TAG) != CK.out_decl("OB-BAND_s0"))
    good = CK.tokens("OB-BAND_s0")
    i = good.index("--w-op") + 1
    bad = good[:i] + ["0.5"] + good[i + 1:]
    r = CK.audit_line("OB-BAND_s0", " ".join(bad))
    check("a changed --w-op is refused and the first differing token is named", len(r) == 1 and f"token {i} is '0.5'" in r[0] and "'0.3333333'" in r[0])
    check("a changed seed is refused", CK.audit_line("OB-BAND_s0", " ".join(good).replace("--seed 0", "--seed 3")) != [])
    check("the detach dropped is refused", CK.audit_line("OB-CTL_s0", " ".join(CK.tokens("OB-CTL_s0")).replace(" --detach-upper", "")) != [])
    check("the freeze set changed is refused", CK.audit_line("OB-CTL_s0", " ".join(CK.tokens("OB-CTL_s0")).replace("--freeze S adapter cond", "--freeze S adapter")) != [])
    check("an extra token is refused", CK.audit_line("OB-CTL_s0", " ".join(CK.tokens("OB-CTL_s0")) + " --steps 5") != [])
    check("a missing token is refused", CK.audit_line("OB-CTL_s0", " ".join(CK.tokens("OB-CTL_s0").__getitem__(slice(0, -2)))) != [])
    check("the CTL argv shipped for BAND is refused", CK.audit_line("OB-BAND_s1", " ".join(CK.tokens("OB-CTL_s1"))) != [])
    check("BAND_rep is OB-BAND_s0 apart from the seed and --out", [(a, b) for a, b in zip(CK.tokens("OB-BAND_rep"), CK.tokens("OB-BAND_s0")) if a != b] == [("2", "0"), (CK.out_decl("OB-BAND_rep"), CK.out_decl("OB-BAND_s0"))])
    for t in ("OB-BAND-TOK_s0", "OB-BAND-TOK_s1"):
        check(f"{t}: the CLI refuses a blocked tag (rc 23)", CK.main(["argv", "--tag", t]) == 23)
        check(f"{t}: its argv still carries the placeholder token dir", "UNVERIFIED" in " ".join(CK.tokens(t)))
    check("the frozen argv file hashes to the pin", CK.sha_file(CK.ARGVS_FILE) == CK.ARGVS_SHA256)
    orig = os.path.join(PKG, "..", "argvs_otband.json")
    if os.path.exists(orig):
        check("the packaged argvs equal theory/otband/argvs_otband.json", CK.sha_file(orig) == CK.ARGVS_SHA256)
    # ---- a clean selfcheck passes, per run tag
    for t in runnable:
        toks, sc = sc_of(t)
        check(f"clean selfcheck passes {t}", CK.gate_problems(t, toks, sc) == [])
    for t in ("OB-BAND-TOK_s0", "OB-BAND-TOK_s1"):
        toks, sc = sc_of(t)
        pr = CK.gate_problems(t, toks, sc)
        check(f"clean token selfcheck of {t} is refused only for the blocker", len(pr) == 1 and "blocked" in pr[0])
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
        "frozen S trainable": lambda s: s["levers"]["params"].__setitem__("S", 5),
        "frozen adapter trainable": lambda s: s["levers"]["params"].__setitem__("adapter", 5),
        "frozen cond trainable": lambda s: s["levers"]["params"].__setitem__("cond", 5),
        "T frozen by mistake": lambda s: s["levers"]["params"].__setitem__("T", 0),
        "O frozen by mistake": lambda s: s["levers"]["params"].__setitem__("O", 0),
        "params record absent": lambda s: s["levers"].pop("params"),
        "live pass": lambda s: s["live"].__setitem__("pass", False),
        "live missing terms": lambda s: s["live"].__setitem__("missing_terms", ["tac"]),
        "live unexpected dead": lambda s: s["live"].__setitem__("unexpected_dead", ["e_code.weight"]),
        "live control": lambda s: s["live"].__setitem__("control_prior_grad_sum", 1e-9),
        "live head_o": lambda s: s["live"].__setitem__("head_o_grad_step0", 0.0),
        "live frozen S gradient": lambda s: s["live"]["groups_grad"].__setitem__("S", 0.5),
        "live T has no gradient": lambda s: s["live"]["groups_grad"].__setitem__("T", 0.0),
        "live non-finite term": lambda s: s["live"]["terms"].__setitem__("ade", float("nan")),
        "dagger on": lambda s: s["levers"].__setitem__("dagger_p", 0.5),
        "recover on": lambda s: s["levers"]["recover"].__setitem__("recover_p", 0.3),
        "wm_only built": lambda s: s["levers"].__setitem__("wm_only", {"wm_only": True}),
        "kin_mode": lambda s: s["levers"].__setitem__("kin_mode", "steer"),
    }
    band_muts = {
        "w_op built value": lambda s: s["levers"].__setitem__("w_op", 0.5),
        "w_tac_traj built value": lambda s: s["levers"].__setitem__("w_tac_traj", 1.0),
        "w_op absent": lambda s: s["levers"].pop("w_op"),
        "probe absent": lambda s: s.pop("otband_probe"),
        "probe pass": lambda s: s["otband_probe"].__setitem__("pass", False),
        "probe detach_upper off": lambda s: s["otband_probe"].__setitem__("detach_upper", False),
        "probe weight w_op": lambda s: s["otband_probe"]["contribution"]["w_op"].__setitem__("weight", 0.5),
        "probe weight w_tac": lambda s: s["otband_probe"]["contribution"]["w_tac_traj"].__setitem__("weight", 0.1),
        "probe band does not reach the total (w_op)": lambda s: s["otband_probe"]["contribution"]["w_op"].__setitem__("moved_total_by", 0.0),
        "probe band does not reach the total (w_tac)": lambda s: s["otband_probe"]["contribution"]["w_tac_traj"].__setitem__("moved_total_by", 0.0),
        "probe band term zero": lambda s: s["otband_probe"]["contribution"]["w_op"].__setitem__("term", 0.0),
        "purity upper gradient": lambda s: s["otband_probe"]["purity"].__setitem__("upper_grad_abs_sum", 1e-12),
        "purity absent": lambda s: s["otband_probe"].pop("purity"),
        "purity head_o zero": lambda s: s["otband_probe"]["purity"].__setitem__("head_o", 0.0),
        "code edge cut": lambda s: s["otband_probe"].__setitem__("code_edge_max_dknot", 0.0),
        "purity tac term on S and T": lambda s: s["otband_probe"]["purity"]["by_term"]["tac"].__setitem__("upper_grad_abs_sum", 1e-12),
        "purity tac term head_o zero": lambda s: s["otband_probe"]["purity"]["by_term"]["tac"].__setitem__("head_o", 0.0),
        "purity tac term absent": lambda s: s["otband_probe"]["purity"]["by_term"].pop("tac"),
        "band term missing in G-LIVE": lambda s: s["live"]["terms"].pop("fde_tac"),
        "token embedding in a band arm": lambda s: s["levers"]["params"].__setitem__("STOK", 5),
    }
    ctl_muts = {
        "band lever built in a control": lambda s: s["levers"].__setitem__("w_op", W),
        "stok built in a control": lambda s: s["levers"].__setitem__("stok_dim", 8),
        "probe absent in a control": lambda s: s.pop("otband_probe"),
        "control probe reads a band lever": lambda s: s["otband_probe"].__setitem__("otband", "on"),
        "control detach off": lambda s: s["otband_probe"].__setitem__("detach_upper", False),
        "control probe failed": lambda s: s["otband_probe"].__setitem__("pass", False),
        "control ade on S and T": lambda s: s["otband_probe"]["purity"]["by_term"]["ade"].__setitem__("upper_grad_abs_sum", 1e-12),
        "control fde head_o zero": lambda s: s["otband_probe"]["purity"]["by_term"]["fde"].__setitem__("head_o", 0.0),
        "control fde absent": lambda s: s["otband_probe"]["purity"]["by_term"].pop("fde"),
        "band terms in a control": lambda s: s["live"]["terms"].__setitem__("op", 0.5),
    }
    W = CK.W_OP
    for t in runnable:
        toks, sc = sc_of(t)
        kind = CK.TABLE[CK.base_tag(t)]["kind"]
        ms = dict(muts)
        ms.update(band_muts if kind != "ctl" else ctl_muts)
        for name, f in ms.items():
            s2 = copy.deepcopy(sc)
            try:
                f(s2)
            except KeyError:
                continue
            check(f"{t}: planted '{name}' is refused", len(CK.gate_problems(t, toks, s2)) >= 1)
    # the token arm's own assertions (blocked, but the assertion code must work when it is unblocked)
    saved = CK.TABLE["OB-BAND-TOK_s0"]["blocked"]
    CK.TABLE["OB-BAND-TOK_s0"]["blocked"] = None
    toks, sc = sc_of("OB-BAND-TOK_s0")
    check("unblocked token arm: clean selfcheck passes", CK.gate_problems("OB-BAND-TOK_s0", toks, sc) == [])
    for name, f in {"token edge cut": lambda s: s["otband_probe"]["token_edge"].__setitem__("max_dlogit", 0.0), "shuffle declared": lambda s: s["stok"].__setitem__("stok_shuffle", 3),
                    "stok dim differs": lambda s: s["stok"].__setitem__("stok_dim", 4), "no new keys declared": lambda s: s["info"]["init_ckpt"].pop("declared_new_keys"),
                    "STOK params zero": lambda s: s["levers"]["params"].__setitem__("STOK", 0), "stok_dim missing": lambda s: s["levers"].pop("stok_dim")}.items():
        s2 = copy.deepcopy(sc)
        f(s2)
        check(f"token arm: planted '{name}' is refused", len(CK.gate_problems("OB-BAND-TOK_s0", toks, s2)) >= 1)
    CK.TABLE["OB-BAND-TOK_s0"]["blocked"] = saved
    # ---- freeze identity (pure)
    a = PT.model_state(0)
    b = {k: v.copy() for k, v in a.items()}
    for k in b:
        if k.startswith(("head_o", "read_o", "trunk_t", "h_goal")):
            b[k] = b[k] + 1.0
    ev, pr = CK.freeze_identity(a, b)
    check("freeze identity: S, adapter, cond equal, T and O moved -> clean", pr == [] and ev["n_frozen_tensors"] == 11 and ev["n_frozen_equal"] == 11)
    for k in ("read_s.w", "head_s.w", "adapter.0.w", "pos", "time", "e_ego.w", "e_nav.w", "e_lim.w", "e_age.w", "e_ck.w", "e_code.w"):
        b2 = {kk: v.copy() for kk, v in b.items()}
        b2[k] = b2[k] + np.float32(1e-7)
        check(f"freeze identity: one changed bit in {k} is refused", CK.freeze_identity(a, b2)[1] != [])
    b3 = dict(b)
    b3.pop("pos")
    check("freeze identity: a missing frozen tensor is refused", CK.freeze_identity(a, b3)[1] != [])
    check("freeze identity: nothing trained is refused", CK.freeze_identity(a, {k: v.copy() for k, v in a.items()})[1] != [])
    check("freeze identity: a frozen tensor of another dtype is refused", CK.freeze_identity(a, {**b, "e_ego.w": b["e_ego.w"].astype(np.float64)})[1] != [])
    check("freeze identity: an init without S / adapter / cond keys is refused", CK.freeze_identity({"x": np.zeros(2)}, {"x": np.zeros(2)})[1] != [])
    # ---- token binding, post check and the cleanup guard on a fake box folder
    root = os.path.join(tmp, "box", "D", "tanitad")
    os.environ.update(OTBAND_ROOT=root, OTBAND_COMMIT="a" * 40, OTBAND_CKPT_LOADER="pickle", OTBAND_FLOOR_BYTES="1000")
    for rel in ("cache_full/native_tracks.pt", "data/v9/lead_train4572.npz", "data/v9/refcv6_speed_max_v8_train.jsonl", "runs/v8f_mini_full/join_train.npz", "runs/v8f_mini_full/p10/p10_prior.pt"):
        put(os.path.join(root, *rel.split("/")), rel.encode())
    put(os.path.join(root, "data", "v9", "v9_labels_train.npz"), b"v9 labels")
    os.environ["OTBAND_FAKE_V9_MD5"] = hashlib.md5(b"v9 labels").hexdigest()
    import pickle
    shas = {}
    for s in (0, 1):
        p = os.path.join(root, "runs", "v8f_mini_full", f"M2b_s{s}", "ckpt_last.pt")
        put(p, pickle.dumps({"model": PT.model_state(s)}))
        shas[str(s)] = hashlib.sha256(open(p, "rb").read()).hexdigest()
    json.dump(shas, open(os.path.join(tmp, "init_sha.json"), "w"))
    put(os.path.join(root, "otband", "repo", "stack", "scripts", "v8f_mini_train.py"), b"trainer v1")
    put(os.path.join(root, "otband", "tree_manifest.json"), b"{}")
    tag = "OB-BAND_s0"
    line = " ".join(CK.tokens(tag))
    b_, bad_ = CK.binding(tag, line)
    check("binding reads the six data files", bad_ == [] and len(b_["data_md5"]) == 6)
    sc_p = os.path.join(CK.run_dir(tag), "gate", "selfcheck.json")
    put(sc_p, b"{}")
    tok = {"schema": "otband-gate-token-v1", "written_by": "otband_check.py gate", "selfcheck_sha256": CK.sha_file(sc_p), **b_}
    put(CK.token_path(tag), json.dumps(tok).encode())
    check("a token bound to the current state is valid", CK.token_ok(tag, line) == [])
    os.environ["OTBAND_COMMIT"] = "b" * 40
    check("a token for another commit is refused", CK.token_ok(tag, line) != [])
    os.environ["OTBAND_COMMIT"] = "a" * 40
    check("a token for another argv is refused", CK.token_ok(tag, line + " --x") != [])
    put(os.path.join(root, "otband", "repo", "stack", "scripts", "v8f_mini_train.py"), b"trainer v2")
    check("a token after the trainer bytes changed is refused", CK.token_ok(tag, line) != [])
    put(os.path.join(root, "otband", "repo", "stack", "scripts", "v8f_mini_train.py"), b"trainer v1")
    put(os.path.join(root, "data", "v9", "lead_train4572.npz"), b"changed")
    check("a token after a data file changed is refused", CK.token_ok(tag, line) != [])
    put(os.path.join(root, "data", "v9", "lead_train4572.npz"), b"data/v9/lead_train4572.npz")
    put(os.path.join(root, "otband", "tree_manifest.json"), b'{"x": 1}')
    check("a token after the shipped tree manifest changed is refused", CK.token_ok(tag, line) != [])
    put(os.path.join(root, "otband", "tree_manifest.json"), b"{}")
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
    check("verify-token refuses a wrong tag", vt("OB-CTL_s0", "a" * 40) == 1)
    check("verify-token refuses a wrong commit", vt(tag, "c" * 40) == 1)
    open(os.path.join(tmp, "arg2.txt"), "w").write(line.replace("--seed 0", "--seed 9") + "\n")
    check("verify-token refuses an argv that is not the frozen one", vt(tag, "a" * 40, os.path.join(tmp, "arg2.txt")) == 1)
    # post check on an out dir written by the stand-in's own format
    out = CK.loc(CK.out_decl(tag))
    toks = line.split(" ")
    ck_ok = {"model": {**PT.model_state(0), **{k: v + 1.0 for k, v in PT.model_state(0).items() if k.startswith(("head_o", "read_o", "trunk_t", "h_goal"))}},
             "cfg": {"detach_upper": True, "w_traj": 0.0, "w_op": CK.W_OP, "w_tac_traj": CK.W_TAC, "stok_dim": 0}, "step": CK.RUN_STEPS}
    summ = {"done": True, "steps": CK.RUN_STEPS, "argv_sha": CK.argv_sha16(toks)}

    def write_out(ck, sm):
        put(os.path.join(out, "ckpt_last.pt"), pickle.dumps(ck))
        put(os.path.join(out, "summary.json"), json.dumps(sm).encode())
    write_out(ck_ok, summ)
    check("post check: a correct finished run passes", CK.post_problems(tag, toks, out)[1] == [])
    for name, ck_m, sm_m in (("steps short", ck_ok, dict(summ, steps=100)), ("not done", ck_ok, dict(summ, done=False)), ("summary argv_sha", ck_ok, dict(summ, argv_sha="0" * 16)),
                             ("ckpt step", dict(ck_ok, step=5), summ), ("cfg detach_upper off", dict(ck_ok, cfg=dict(ck_ok["cfg"], detach_upper=False)), summ),
                             ("cfg w_traj on", dict(ck_ok, cfg=dict(ck_ok["cfg"], w_traj=1.0)), summ), ("cfg w_op", dict(ck_ok, cfg=dict(ck_ok["cfg"], w_op=0.1)), summ),
                             ("frozen tensor moved", dict(ck_ok, model={**ck_ok["model"], "e_code.w": ck_ok["model"]["e_code.w"] + 1}), summ),
                             ("nothing trained", dict(ck_ok, model=PT.model_state(0)), summ)):
        write_out(ck_m, sm_m)
        check(f"post check: planted '{name}' is refused", CK.post_problems(tag, toks, out)[1] != [])
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
    check("1.0 s/step: 5 runs x 9,582 steps = 13.3 h; with token arms 18.6 h", abs(pr1["projected_total_h_planned"] - 47910 / 3600) < 1e-9 and abs(pr1["projected_total_h_with_token_arms"] - 67074 / 3600) < 1e-9 and pr1["within_limit_planned"])
    check("6.0 s/step is over 72 h", not CK.project(6.0)["within_limit_planned"])

    def pilot_json(mean, commit="a" * 40):
        pj = {"commit": commit, "projection": CK.project(mean)}
        p = os.path.join(tmp, f"pj_{mean}_{commit[:1]}.json")
        open(p, "w").write(json.dumps(pj))
        return p
    check("permit: within 72 h passes", CK.cmd_permit({"pilot": pilot_json(1.0), "commit": "a" * 40}) == 0)
    p6 = pilot_json(6.0)
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
            if re.search(rb"\btext\s*=\s*True", raw) and not f.startswith("test_otband_check"):
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
