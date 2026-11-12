"""STOPPERC stage-1 launch-gate checker (PREREG_V8F_STOPPERC section 5, Launch; stage1_A/RESULTS.md "Gate list"; AMENDMENT_3_A_READINGS). Runs on box 2 and on the dev box.
CONSTRUCTION: this is the STOPBAL checker (theory/stopbal/box2/stopbal_check.py, sha256 pinned below, shipped UNCHANGED beside this file) re-pointed at the STOPPERC tags, paths and argv file, with the arm-A clauses added.
It patches the base module's globals (TABLE, RUN_ORDER, HOME_DECL, ARGVS loader, gate_problems, post_problems, binding, data_paths) and then calls the base `main`; every base command (argv, render, preflight, stage, cleanup, verify-token, permit, parity)
therefore keeps its behaviour. New commands:  wsmoke --gid G   (box 2: selfcheck-only run of A_s0 at W 1.0, copies selfcheck.json, refuses probe_n_pos 0)   |   reuse --ckpt F --bal-ckpt F --out F  (dev box: NULL reuse rule, tensor basis + file sha256).
Code of record: 3e264bbf3fa2f6f17bea448de09aa972faaf7725.  Env: STOPBAL_COMMIT (the base reads this name), STOPPERC_W (the final W, 4 decimals), STOPPERC_TABLE_SHA (test hook only).
Arms: NULL_s0, NULL_rep2 (seed 3), A_s0, A_s1; conditional NULL_s1, NULL_rep (blocked unless STOPPERC_NULL_REUSE_FAILED=1)."""
import copy
import hashlib
import json
import ntpath
import os
import re
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
for _d in (HERE, os.environ.get("STOPPERC_BASE_DIR", ""), os.path.join(HERE, "..", "..", "..", "stopbal", "box2")):
    if _d and os.path.isfile(os.path.join(_d, "stopbal_check.py")):
        sys.path.insert(0, _d)
        break
import stopbal_check as B  # noqa: E402

BASE_SHA256 = "70e260a697a6e3e23b33a37e6781e34ef4d482bd80b5e70d0e3964c19a8e5212"            # stopbal_check.py as banked (MEASURED 2026-11-12); a different base is refused
ARGVS_SHA256 = "b005ea0b5193e7c0d70579274b0a09fdacd030a6302e0443c45627d35b3ef4f7"            # argvs_stopperc.txt (make_argvs_stopperc.py; every argv parses with the trainer at 3e264bbf)
COMMIT = "3e264bbf3fa2f6f17bea448de09aa972faaf7725"
TABLE_SHA256 = "77b6e8dc63d07d90548d09182fd5b8073e81079e224e8ce672dda11d2c3ae361"          # stopaux_extr_join4572.json (4,572 / 4,572 clips)
TABLE_B2 = r"D:\tanitad\stopperc\data\stopaux_extr_join4572.json"
BAL_NULL_S0_CKPT_SHA256 = "5169ea9554bf74004f801b2d594358355363ba47fe07aa8cac052d1da9aa2fdf"   # C:/Users/Admin/t2v8f_pulls/stopbal_BAL-NULL_s0/BANK_MANIFEST.json (files.ckpt_last.pt, 86,971,285 bytes)
STOPAUX_PARAMS, N_TRAIN_CLIPS, MIN_COVERAGE = 132098, 4512, 0.99
STOP_KEYS = sorted(f"{h}.{i}.{p}" for h in ("stop_head", "stop_pres_head") for i in (0, 2) for p in ("weight", "bias"))
WPH, W_RE = "W_PLACEHOLDER", re.compile(r"^[0-9]+\.[0-9]{4}$")
ARGVS_FILE = os.path.join(HERE, "argvs_stopperc.txt")
HOME_DECL = B.DECL_D + r"\stopperc"
TABLE = {"NULL_s0": {"kind": "null", "arm": "NULL", "seed": 0, "init": 0, "order": 1, "blocked": None},
         "NULL_rep2": {"kind": "null", "arm": "NULL", "seed": 3, "init": 0, "order": 2, "blocked": None},
         "A_s0": {"kind": "null", "arm": "A", "seed": 0, "init": 0, "order": 3, "blocked": None},
         "A_s1": {"kind": "null", "arm": "A", "seed": 1, "init": 1, "order": 4, "blocked": None}}
_COND = "conditional: the card's NULL reuse rule failed (NULL_s0's model tensors differ from BAL-NULL_s0's); set STOPPERC_NULL_REUSE_FAILED=1 after `reuse` says DIFFERS"
TABLE["NULL_s1"] = {"kind": "null", "arm": "NULL", "seed": 1, "init": 1, "order": 5, "blocked": None if os.environ.get("STOPPERC_NULL_REUSE_FAILED") == "1" else _COND}
TABLE["NULL_rep"] = {"kind": "null", "arm": "NULL", "seed": 2, "init": 0, "order": 6, "blocked": None if os.environ.get("STOPPERC_NULL_REUSE_FAILED") == "1" else _COND}
RUN_ORDER = sorted((t for t, v in TABLE.items() if not v["blocked"]), key=lambda t: TABLE[t]["order"])
_base_gate, _base_post = B.gate_problems, B.post_problems


def base_ok():
    got = B.sha_file(B.__file__)
    if got != BASE_SHA256:
        raise SystemExit(f"REFUSED: {B.__file__} sha256 {got} is not the pinned STOPBAL checker {BASE_SHA256}")


def _w_final():
    w = os.environ.get("STOPPERC_W", "")
    if w and (not W_RE.match(w) or float(w) <= 0.0):
        raise SystemExit(f"REFUSED: STOPPERC_W {w!r} is not a positive 4-decimal number")
    return w


def frozen_argvs():
    got = B.sha_file(ARGVS_FILE)
    if got != ARGVS_SHA256 and not os.environ.get("STOPBAL_TEST_ARGVS"):
        raise SystemExit(f"REFUSED: {ARGVS_FILE} sha256 {got} != the pinned {ARGVS_SHA256} (the frozen argv file changed)")
    out, w = {}, _w_final()
    for ln in open(ARGVS_FILE, encoding="utf-8"):
        if ln.strip():
            tag, rest = ln.rstrip("\n").split(": ", 1)
            toks = rest.split(" ")
            if w and WPH in toks:
                toks[toks.index(WPH)] = w
            out[tag] = toks
    return out


def stop_keys_anywhere(x, path=""):
    """every dict key of a nested record that is a stop-aux key (`stop_*`, `STOPAUX`): the OFF arms must carry none"""
    hits = []
    if isinstance(x, dict):
        for k, v in x.items():
            if re.match(r"^(stop_|STOPAUX)", str(k)):
                hits.append(f"{path}/{k}")
            hits += stop_keys_anywhere(v, f"{path}/{k}")
    return hits


def argv_problems(tag, toks):
    """clause 6 (structure): arm A = the template argv + exactly --stop-aux on --w-stop-aux <W> --stop-aux-extr <table>; the NULL arms carry no stop flag"""
    p, has = [], [t for t in toks if t.startswith("--stop-aux") or t == "--w-stop-aux"]
    if TABLE[B.base_tag(tag)]["arm"] == "NULL":
        return [f"{tag}: NULL argv carries {has}"] if has else []
    if toks[-6:] != ["--stop-aux", "on", "--w-stop-aux", toks[-3], "--stop-aux-extr", TABLE_B2] or sorted(has) != ["--stop-aux", "--stop-aux-extr", "--w-stop-aux"]:
        p.append(f"{tag}: the argv does not end in exactly '--stop-aux on --w-stop-aux <W> --stop-aux-extr {TABLE_B2}' (flags seen: {has})")
    if not W_RE.match(toks[-3]):
        p.append(f"{tag}: --w-stop-aux {toks[-3]!r} is not a filled 4-decimal W (placeholder or malformed: run the W smoke first)")
    if B.flag_value(toks, "--grid") != "pool4x10":
        p.append(f"{tag}: --grid is not pool4x10")
    return p


def a_problems(tag, toks, sc):
    """G-DVB 1-5 and G-LIVE 1-8 for arm A, read from selfcheck.json keys the trainer at 3e264bbf writes"""
    p, lv = argv_problems(tag, toks), (sc.get("levers") if isinstance(sc.get("levers"), dict) else {})
    info, pr = (sc.get("info") if isinstance(sc.get("info"), dict) else {}), (lv.get("params") if isinstance(lv.get("params"), dict) else {})
    ex, pb = sc.get("stop_aux_extrinsics"), sc.get("stop_aux_probe")
    ex, pb = (ex if isinstance(ex, dict) else {}), (pb if isinstance(pb, dict) else {})
    w = toks[-3] if W_RE.match(toks[-3]) else None

    def eq(n, got, want):
        if got != want:
            p.append(f"{n} = {got!r}, wanted {want!r}")

    def pos(n, v):
        if not (B.finite(v) and v > 0.0):
            p.append(f"{n} = {v!r}, wanted finite > 0")
    # G-DVB 1: the flag and W read back from the BUILT model
    eq("G-DVB1 levers.stop_aux (built)", lv.get("stop_aux"), "on")
    if w is None or not (B.finite(lv.get("w_stop_aux")) and abs(lv["w_stop_aux"] - float(w)) <= 1e-9):
        p.append(f"G-DVB1 levers.w_stop_aux = {lv.get('w_stop_aux')!r}, the argv's W is {w!r}")
    # G-DVB 2: the 8 head tensors exist and are counted
    eq("G-DVB2 levers.params.STOPAUX", pr.get("STOPAUX"), STOPAUX_PARAMS)
    # G-DVB 3: vision only
    eq("G-DVB3 levers.stop_vision_only_source", lv.get("stop_vision_only_source"), True)
    vo = pb.get("vision_only") if isinstance(pb.get("vision_only"), dict) else {}
    eq("G-DVB3 stop_aux_probe.vision_only.source_problems", vo.get("source_problems"), [])
    eq("G-DVB3 stop_aux_probe.vision_only.keys_read", vo.get("keys_read"), [])
    # G-DVB 4: the camera table
    tsha = os.environ.get("STOPPERC_TABLE_SHA", TABLE_SHA256)
    if ntpath.normpath(str(ex.get("path", "")).replace("/", "\\")).lower() != ntpath.normpath(TABLE_B2).lower():
        p.append(f"G-DVB4 stop_aux_extrinsics.path {ex.get('path')!r} is not the declared table {TABLE_B2}")
    if not (isinstance(ex.get("sha256"), str) and B.HEX64.match(ex["sha256"]) and ex["sha256"] == tsha):
        p.append(f"G-DVB4 stop_aux_extrinsics.sha256 {ex.get('sha256')!r} is not the shipped table's {tsha}")
    if not (B.finite(ex.get("coverage_train")) and ex["coverage_train"] >= MIN_COVERAGE):
        p.append(f"G-DVB4 stop_aux_extrinsics.coverage_train = {ex.get('coverage_train')!r}, wanted >= {MIN_COVERAGE}")
    eq("G-DVB4 stop_aux_extrinsics.n_train_clips", ex.get("n_train_clips"), N_TRAIN_CLIPS)
    # G-DVB 5: M2b init with exactly the 8 stop keys fresh
    ic = info.get("init_ckpt") if isinstance(info.get("init_ckpt"), dict) else {}
    eq("G-DVB5 info.init_ckpt.fresh_keys (exactly the 8 stop keys)", sorted(ic.get("fresh_keys") or []), STOP_KEYS)
    eq("G-DVB5 info.init_ckpt.declared_new_keys (= fresh_keys ∩ stop keys)", sorted(set(ic.get("declared_new_keys") or []) & set(STOP_KEYS)), STOP_KEYS)
    # G-LIVE 1: finite term
    pos("G-LIVE1 stop_aux_probe.term", pb.get("term"))
    for k in ("step0_stop_cell", "step0_stop_pres"):
        if not B.finite(pb.get(k)):
            p.append(f"G-LIVE1 stop_aux_probe.{k} = {pb.get(k)!r}, wanted finite")
    # G-LIVE 2: non-zero gradient on head, presence head, adapter
    for k in ("head_grad_diff", "pres_head_grad_diff", "adapter_grad_diff"):
        pos(f"G-LIVE2 stop_aux_probe.{k}", pb.get(k))
    # G-LIVE 3: exactly 0.0 at w = 0 (the head's gradient), the off-arm control lives in the NULL records (no head to read)
    eq("G-LIVE3 head_grad_at_w0_zero_control (exactly 0.0)", pb.get("head_grad_at_w0_zero_control"), 0.0)
    # G-LIVE 4: mutation detector -- the declared weight reaches the total
    c, wx = pb.get("contribution"), pb.get("w_x_term")
    if not (B.finite(c) and B.finite(wx) and wx > 0.0 and abs(c - wx) <= 1e-3):
        p.append(f"G-LIVE4 contribution = {c!r}, w_x_term = {wx!r}: not equal within 1e-3 (the term does not reach the total)")
    # G-LIVE 5: controls and the probe's content (AMENDMENT_3: stop_has = positives + moving negatives; positives >= 1)
    for k in ("control_repeat_total_diff", "control_repeat_grad_diff", "control_allfalse_term", "control_allfalse_total_diff", "control_allfalse_head_grad_diff", "control_prior_grad_diff"):
        eq(f"G-LIVE5 stop_aux_probe.{k} (control: exactly 0.0)", pb.get(k), 0.0)
    if not (isinstance(pb.get("probe_n_pos"), int) and pb["probe_n_pos"] >= 1):
        p.append(f"G-LIVE5 probe_n_pos = {pb.get('probe_n_pos')!r}, wanted >= 1 (the probe cannot see the cell term)")
    if not (isinstance(pb.get("probe_n_has"), int) and isinstance(pb.get("probe_n_pos"), int) and pb["probe_n_has"] >= pb["probe_n_pos"]):
        p.append(f"G-LIVE5 probe_n_has = {pb.get('probe_n_has')!r} < probe_n_pos = {pb.get('probe_n_pos')!r} (stop_has must contain stop_pos: AMENDMENT_3 section 1)")
    # G-LIVE 6: W equals the argv's to the W script's rounding
    wf = pb.get("w_stop_aux_for_goal_equal")
    if w is None or not (B.finite(wf) and abs(round(wf, 4) - float(w)) <= 1e-9):
        p.append(f"G-LIVE6 w_stop_aux_for_goal_equal = {wf!r} rounds to {round(wf, 4) if B.finite(wf) else None!r}, the argv's W is {w!r}")
    # G-LIVE 7: live.pass with the stop terms present and nothing unexpectedly dead
    lp = sc.get("live") if isinstance(sc.get("live"), dict) else {}
    terms = lp.get("terms") if isinstance(lp.get("terms"), dict) else {}
    miss = [k for k in ("stop_aux", "stop_cell", "stop_pres") if k not in terms]
    if miss:
        p.append(f"G-LIVE7 live.terms lacks {miss}")
    eq("G-LIVE7 live.unexpected_dead", lp.get("unexpected_dead"), [])
    return p


def null_problems(tag, toks, sc):
    """an OFF arm: no stop flag in the argv, no stop_* key anywhere in selfcheck.json, no STOPAUX group, no fresh key (clause 5, and the card's 'new flags default to the tip, bit for bit')"""
    p = argv_problems(tag, toks)
    hits = stop_keys_anywhere(sc)
    if hits:
        p.append(f"G-DVB5 an --stop-aux off arm carries stop keys in selfcheck.json: {hits[:4]}")
    ic = ((sc.get("info") or {}).get("init_ckpt") or {}) if isinstance(sc.get("info"), dict) else {}
    if any(str(k).startswith("stop") for k in (ic.get("fresh_keys") or [])):
        p.append("G-DVB5 an off arm has fresh stop keys")
    return p


def gate_problems(tag, toks, sc):
    spec = TABLE[B.base_tag(tag)]
    if spec["arm"] == "NULL":
        return _base_gate(tag, toks, sc) + null_problems(tag, toks, sc)
    s2, extra = copy.deepcopy(sc), []                    # the base asserts DAG-OACC's architecture: show it the architecture WITHOUT the declared head, after a_problems pinned the head itself
    lv = s2.get("levers") if isinstance(s2.get("levers"), dict) else {}
    pr = lv.get("params") if isinstance(lv.get("params"), dict) else {}
    if pr.get("STOPAUX") == STOPAUX_PARAMS:
        pr.pop("STOPAUX")
        if B.num(lv.get("n_params_trainable")):
            lv["n_params_trainable"] -= STOPAUX_PARAMS
    ic = ((s2.get("info") or {}).get("init_ckpt")) if isinstance(s2.get("info"), dict) else None
    if isinstance(ic, dict) and sorted(ic.get("declared_new_keys") or []) == STOP_KEYS:
        ic["declared_new_keys"] = []
    return _base_gate(tag, toks, s2) + a_problems(tag, toks, sc) + extra


def post_problems(tag, toks, out):
    ev, p = _base_post(tag, toks, out)
    spec = TABLE[B.base_tag(tag)]
    ck_p = os.path.join(out, "ckpt_last.pt")
    if os.path.exists(ck_p) and not (B.fake() and os.environ.get("STOPBAL_SKIP_WEIGHTS") == "1"):
        b = B.load_ckpt(ck_p)
        cfg = b.get("cfg") if isinstance(b.get("cfg"), dict) else {}
        if spec["arm"] == "A":
            w = toks[-3]
            if cfg.get("stop_aux") != "on" or not (B.finite(cfg.get("w_stop_aux")) and abs(cfg["w_stop_aux"] - float(w)) <= 1e-9):
                p.append(f"checkpoint cfg stop_aux/w_stop_aux = {cfg.get('stop_aux')!r}/{cfg.get('w_stop_aux')!r}, wanted on/{w}")
            miss = [k for k in STOP_KEYS if k not in b.get("model", {})]
            if miss:
                p.append(f"checkpoint lacks head tensors {miss[:3]}")
            elif not os.environ.get("STOPPERC_SKIP_EVALBUILD"):
                p += eval_build_problems(b)
        else:
            bad = [k for k in cfg if str(k).startswith(("stop_aux", "w_stop_aux"))] + [k for k in b.get("model", {}) if str(k).startswith("stop_")]
            if bad:
                p.append(f"an off arm's checkpoint carries stop-aux entries {bad[:3]}")
    return ev, p


def eval_build_problems(ck):
    """G-LIVE 8 (build half): the eval loader's route -- V8FMiniConfig(**ck['cfg']) + strict load -- builds the ON checkpoint. (forward_infer's independence of the head is pinned by the tree's test_inference_never_reads_the_head, not re-run here.)"""
    try:
        sp = os.path.join(B.repo(), "stack")
        sys.path.insert(0, sp)
        from tanitad.models import v8f_mini as M
        m = M.V8FMini(M.V8FMiniConfig(**ck["cfg"]))
        m.load_state_dict(ck["model"], strict=True)
        return []
    except Exception as e:                                                      # noqa: BLE001  any failure is a problem, never a pass
        return [f"G-LIVE8 the eval loader cannot build the ON checkpoint: {type(e).__name__}: {str(e)[:200]}"]


def binding(tag, line):
    b, bad = _base_binding(tag, line)
    arm = TABLE[B.base_tag(tag)]["arm"]
    b["stopperc_checker_sha256"] = B.sha_file(os.path.abspath(__file__))
    b["w_stop_aux"] = line.split(" ")[-3] if arm == "A" else None
    b["stop_aux_table_sha256"] = os.environ.get("STOPPERC_TABLE_SHA", TABLE_SHA256) if arm == "A" else None
    if arm == "A":
        tp = B.loc(TABLE_B2)
        if not os.path.isfile(tp):
            bad = list(bad) + [f"camera table missing on this machine: {TABLE_B2}"]
        elif B.sha_file(tp) != b["stop_aux_table_sha256"]:
            bad = list(bad) + [f"camera table sha256 {B.sha_file(tp)} != {b['stop_aux_table_sha256']}"]
    return b, bad


_base_binding = B.binding


def data_paths():
    t = B.tokens("NULL_s0")
    g = lambda f: t[t.index(f) + 1]                                  # noqa: E731
    return {"v9_labels_train": g("--v9-train"), "lead_train": g("--lead-train"), "speed_train": g("--speed-train"), "join_train": g("--join-train"), "p10_prior": g("--p10"), "native_tracks": g("--tracks")}


def cmd_reuse(a):
    """the card's NULL reuse rule on the dev box: NULL_s0's model tensors against BAL-NULL_s0's (the checkpoint embeds argv_sha, which holds --out, so a whole-file sha256 cannot match by construction; both are recorded).
    EQUAL -> BAL-NULL_s1 / BAL-NULL_rep are reused and the conditional arms stay blocked. DIFFERS -> set STOPPERC_NULL_REUSE_FAILED=1 for the conditional NULL_s1 / NULL_rep. Always writes --out."""
    ck, bal, out = a.get("ckpt"), a.get("bal-ckpt"), a.get("out")
    if not (ck and bal and out):
        print("reuse refused: need --ckpt, --bal-ckpt, --out")
        return 2
    res = {"schema": "stopperc-null-reuse-v1", "at": B.now(), "ckpt": ck, "bal_ckpt": bal, "bal_pinned_sha256": BAL_NULL_S0_CKPT_SHA256}
    if not (os.path.isfile(ck) and os.path.isfile(bal)):
        res.update(status="not-computed", null_reuse="UNKNOWN", reason="a checkpoint is missing")
    elif B.sha_file(bal) != BAL_NULL_S0_CKPT_SHA256:
        res.update(status="not-computed", null_reuse="UNKNOWN", reason=f"{bal} does not hash to the pinned BAL-NULL_s0 checkpoint {BAL_NULL_S0_CKPT_SHA256}")
    else:
        eq_, det = B.tensor_parity(ck, bal)
        res.update(status="computed", ckpt_sha256=B.sha_file(ck), file_sha_equal=B.sha_file(ck) == BAL_NULL_S0_CKPT_SHA256, tensor_equal=eq_, tensor_detail=det, null_reuse="EQUAL" if eq_ else "DIFFERS")
    B.write_text(out, json.dumps(res, indent=1))
    print(f"NULL-REUSE {res['null_reuse']} ({res['status']})")
    return 0 if res["status"] == "computed" else 2


def wsmoke_line():
    w1 = list(B.tokens("A_s0"))
    return " ".join(w1[:-3] + ["1.0"] + w1[-2:]) if w1[-3] == WPH else None


def cmd_wsmoke(a):
    """box 2: the W smoke. Selfcheck-only run of A_s0 with --w-stop-aux 1.0 (any W > 0; the probe's W is goal / stop_aux and does not depend on it). Copies selfcheck.json to runs/W_SMOKE and refuses, naming the cause, when probe_n_pos is 0."""
    gid = a["gid"]
    rd = os.path.join(B.home(), "runs", "W_SMOKE")
    os.makedirs(rd, exist_ok=True)

    def fail(kind, why):
        B.write_text(os.path.join(rd, f"STAGE_{gid}.fail"), f"{kind}: {why}"[:900] + "\n")
        print(f"STOPPERC-FAIL W_SMOKE {kind}: {why}"[:900])
        return 3
    bad = B.disk_floor_problems() + B.tree_problems()
    if bad:
        return fail("preflight-fail", " | ".join(bad)[:700])
    toks = list(B.tokens("A_s0"))
    toks[toks.index(WPH)] = "1.0"
    toks[toks.index("--out") + 1] = B.HOME_DECL + r"\gate_out\W_SMOKE"
    gout = B.loc(toks[toks.index("--out") + 1])
    B.rm_own_temp(gout, "gate_out")
    rc, _, _ = B.run_trainer(toks + ["--selfcheck-only"], os.path.join(rd, "logs", "wsmoke.log"))
    scp = os.path.join(gout, "selfcheck.json")
    if rc != 0 or not os.path.exists(scp):
        return fail("gate-fail", f"the trainer's --selfcheck-only exited rc={rc}; selfcheck.json {'written' if os.path.exists(scp) else 'missing'} (log {rd}\\logs\\wsmoke.log). If the log names 'cannot see the cell term', probe_n_pos is 0: STOP, the sampler is NOT changed here")
    sc = json.load(open(scp, encoding="utf-8"))
    pb = sc.get("stop_aux_probe") if isinstance(sc.get("stop_aux_probe"), dict) else {}
    if not (isinstance(pb.get("probe_n_pos"), int) and pb["probe_n_pos"] >= 1):
        return fail("gate-fail", f"probe_n_pos = {pb.get('probe_n_pos')!r}: the probe batch has no in-view rest point, so the cell term is unreadable. STOP. A STOP-stratified probe batch is a code change landed before the launch; this stage does not change the sampler")
    import shutil
    shutil.copy2(scp, os.path.join(rd, "selfcheck.json"))
    B.write_text(os.path.join(rd, f"STAGE_{gid}.done"), "done\n")
    print(f"STOPPERC-OK W_SMOKE probe_n_pos={pb['probe_n_pos']} w_for_goal_equal={pb.get('w_stop_aux_for_goal_equal')!r}")
    return 0


def install():
    base_ok()
    B.TABLE, B.RUN_ORDER, B.HOME_DECL = TABLE, RUN_ORDER, HOME_DECL
    B.REPO_DECL, B.PILOT_OUT_DECL = HOME_DECL + r"\repo", HOME_DECL + r"\pilot_out"
    B.PILOT_TAG, B.PILOT_BASE, B.PARITY_TAG, B.GATE_AT_PILOT = "PILOT_NONE", "NULL_s0", "NULL_s0", []
    B.frozen_argvs, B.gate_problems, B.post_problems, B.binding, B.data_paths = frozen_argvs, gate_problems, post_problems, binding, data_paths
    B.__doc__ = __doc__


def main(argv):
    install()
    if argv and argv[0] == "reuse":
        return cmd_reuse(B.parse(argv[1:]))
    if argv and argv[0] == "wsmoke":
        return cmd_wsmoke(B.parse(argv[1:]))
    if argv and argv[0] == "argv" and "--tag" in argv:
        tag = argv[argv.index("--tag") + 1]
        if tag in TABLE and TABLE[tag]["arm"] == "A" and not _w_final():
            print("REFUSED: arm A's W is not filled (set STOPPERC_W from the W smoke)")
            return 23
    return B.main(argv)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
