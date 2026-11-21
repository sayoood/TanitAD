"""LEVELHEAD arm-CLR checker (runs on box 2 with the venv python; stdlib only, no torch, no network, no subprocess). Every marker and every pin is written HERE, one code path. GENERATED into box2_files/clr_check.py by mk_clr_package.py (the pins below are filled in).
Adapted from the arm-CL checker (cl_check.py), which was adapted from the RECOVER-NEXT checker (rnext_check.py): same init, same data, same gate protocol; the runs are CLR_0 CLR_1 (card PREREG_LEVELHEAD_A2.md, Arms): RAPR-P01's argv + `--level-head on --w-level W` (W = CL's frozen 0.2027; no probe of its own).
  preflight chain         job start: disk floor, the four shipped stack files' sha256, the argv audit of both argv files (W frozen, FROZEN_W.txt = the pinned W), init ckpts
  preflight run <tag>     before each run: disk floor (exit 9 = STOP the chain)
  pin <tag>               argv audit (a changed lever is REFUSED, rc 2), pin the argv (a re-pin that differs is REFUSED), derive the gate argv (the training argv with --out and --selfcheck-only changed, asserted), make the dirs
  gate-cached <tag>       0 iff a PASS token for this tag is still bound to this commit / argv / selfcheck / shipped bytes / card / W
  gate <tag> <rc>         read the gate's selfcheck.json + log, assert the registered controls, the BUILT level-head lever AND the BUILT recovery lever (RAPR-P01's: p 0.1, a-priori law, rejoin target, 10 steps, seed s, no table, no bank), every other lever as C0's;
                          write the per-run launch-gate token, FAIL => RUN_<tag>.FAILED, exit 1
  post <tag> <rc>         after training: summary / argv / ckpt checks (recover_n > 0: the recovery really perturbed samples); PASS => RUN_<tag>.done, else RUN_<tag>.FAILED; exit 0 always
  fail <tag> <reason>     write RUN_<tag>.FAILED
  stopped <reason>        write CHAIN_CLR.STOPPED
  finish                  write CHAIN_CLR.done
Paths come from CLR_HOME / CLR_RUNSM / CLR_REPO / CLR_DISK / CLR_FLOOR_BYTES (defaults are box 2's) so the same file is exercised on the dev box."""
import hashlib, json, os, shutil, sys, time

HOME = os.environ.get("CLR_HOME", r"D:\tanitad\clr")
RUNSM = os.environ.get("CLR_RUNSM", r"D:\tanitad\runs\v8f_mini_full")
REPO = os.environ.get("CLR_REPO", r"D:\tanitad\repo")
DISK = os.environ.get("CLR_DISK", "D:\\")
FLOOR = int(float(os.environ.get("CLR_FLOOR_BYTES", "8e9")))
RC = os.path.join(RUNSM, "clr"); GATE = os.path.join(RUNSM, "gate_clr"); PIN = os.path.join(RC, "pin"); LOGS = os.path.join(RC, "logs")
DECL = r"D:\tanitad\runs\v8f_mini_full"
PINS = json.loads(r'''{
 "card_file": "PREREG_LEVELHEAD_A2.md",
 "card_frozen_sha_file": "FROZEN_SHA.txt",
 "card_sha256": "9a117b201d74db13aa7ccb43b0407de7634d319141203550047929bda5b66a31",
 "frozen_w_sha256": "526c29358053e4fc03aed3382faa636a49f8a823bd57e6a782b08e800335124d",
 "rapr_argv_sha256": {
  "0": "8b79404537bb89ecacdbabcd975cee3e7fc199b171b3153863a3c78e4ed0d01b",
  "1": "a22705e44430758c7bde095178b3fd236f42645150b6f4aae2865f868243a962"
 },
 "stack": [
  {
   "origin": "tip",
   "rel": "stack/scripts/v8f_mini_train.py",
   "sha256": "9ffdf1b4788358d3eb2698f11ca4008bffb042c5ee3887995bac4f01adb390ae",
   "src": "stack_ship/stack/scripts/v8f_mini_train.py"
  },
  {
   "origin": "tip",
   "rel": "stack/tanitad/models/v8f_mini.py",
   "sha256": "a363427677c881cb7c82cf20fcfad317e864a9cde58cf58306e9fed24ef3658e",
   "src": "stack_ship/stack/tanitad/models/v8f_mini.py"
  },
  {
   "origin": "tip",
   "rel": "stack/tanitad/data/v8f_recover.py",
   "sha256": "0257ccb93251827fe5f1b3d7195999b5836145e8615cfd0d0a599262b8834b4c",
   "src": "stack_ship/stack/tanitad/data/v8f_recover.py"
  },
  {
   "origin": "tip",
   "rel": "stack/tanitad/data/v8f_mini_data.py",
   "sha256": "209da7a60765ba81268ee794aba187e21b0a6c73a49b9e98c86e2d876581ece6",
   "src": "stack_ship/stack/tanitad/data/v8f_mini_data.py"
  }
 ],
 "tip_commit": "219f6c76a01d3e7e4632ac517b71689e89c97e1a",
 "w": "0.2027",
 "w_source": "cl_prereg/FROZEN_W.txt (CL's frozen W; no probe of arm CLR's own)"
}''')
SHIPPED = {x["rel"]: x["sha256"] for x in PINS["stack"]}          # the four files over box 2's repo (launch_clr.sh ships them); mk_clr_package.py recomputes this from stack_ship/
TRAINER, MODEL = "stack/scripts/v8f_mini_train.py", "stack/tanitad/models/v8f_mini.py"
TIP = PINS["tip_commit"]                                         # the git tip the two unchanged files were staged against
W = PINS["w"]                                                    # CL's frozen weight string, exactly as it is in the argv (cl_prereg/FROZEN_W.txt); there is no placeholder state
CARD_SHA = PINS["card_sha256"]
PLACEHOLDER = "__W__"                                              # never valid: the generator refuses to run without a frozen W; kept so a stray placeholder in an argv is still named
RP = {"p": 0.1, "law": "apriori", "target": "rejoin", "rejoin": 10}   # RAPR-P01's recovery lever (rnext_check.py KINDS["RAPRP01"]; the seed is the run's seed)
SHARE_LO, SHARE_HI = 0.08, 0.12                                  # the card's window for the step-0 level share at seed 0
INFO = {"CLR_0": ("CLR", 0), "CLR_1": ("CLR", 1)}
INIT_SHA = {0: "67265ec1666fae3a6d51e924ee06dfb68b2cd0630afc62e42f3a46182b8bc45f", 1: "90b01474819082a6d32a9000d63a4954e0074916140abdbbcb67b353f7ac3333"}   # DAG-OACC ckpt_last.pt (MEASURED, as rnext_check.py)
KP1 = {"train_sha256": "37f4f9ee468e0872134074a82d63c2915458596fdbcf3d0fcf5b09f1fe0d8cc1",
       "train_window_list_sha256": "e144adcd9b6273f2b44ad7dfa162931b90e159f7b3911e0893b25777b2ae8168"}
STEPS = 3000
LEVEL_PREFIXES = ("lvl_head.", "e_level.")
BOOL_FLAGS = ("--selfcheck-only",)


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def tokens_sha(toks):
    return hashlib.sha256(json.dumps(list(toks), separators=(",", ":")).encode()).hexdigest()


def argv_file(tag):
    return os.path.join(HOME, "argv", tag + ".train.txt")


def shipped_line(tag):
    with open(argv_file(tag), encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    assert len(lines) == 1, f"{tag}: the shipped argv file must hold exactly one line, has {len(lines)}"
    return lines[0]


def loc(p):                                                 # identity on box 2; maps the declared run root onto CLR_RUNSM when the dev-box test hook points it elsewhere
    return RUNSM + p[len(DECL):] if (RUNSM != DECL and p.startswith(DECL)) else p


def out_of(toks):
    return toks[toks.index("--out") + 1]


def write_text(p, s):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="") as f:
        f.write(s)


def write_marker_file(p, text):
    if os.path.exists(p):                                   # never overwrite, never delete: move aside
        os.replace(p, p + ".prev_" + time.strftime("%Y%m%d_%H%M%S"))
    write_text(p, text)


def mname(tag):
    return f"RUN_{tag}"


def marker(tag, kind, payload):
    p = os.path.join(RC, f"{mname(tag)}.{kind}")
    if os.path.exists(p):                                   # never overwrite, never delete: move aside
        os.replace(p, p + ".prev_" + time.strftime("%Y%m%d_%H%M%S"))
    write_text(p, json.dumps({"tag": tag, "at": now(), "tip": TIP, **payload}, indent=1, default=str))
    print(f"ZZCLR-{kind.upper()} {tag} {payload.get('reason', '')}")


def fail(tag, reason, **kw):
    marker(tag, "FAILED", {"reason": reason, **kw})


def w_ok(w):
    try:
        return w != PLACEHOLDER and float(w) > 0.0 and float(w) < float("inf")
    except ValueError:
        return False


# ---- the frozen argv: RAPR-P01's argv (recover_next/argv/RAPRP01_s.train.txt = v8f-recover/M2b_s{s}_recft_RAPRP01/argv.json) with the card's tokens, written once, compared flag by flag (a named reason per difference) ------------------------
def expected_tokens(tag):
    """RAPR-P01's argv with --out replaced and the two level tokens appended: `--level-head on --w-level W`. Every other token, in order, is RAPR-P01's (its recovery tokens last, as its argv file is built)."""
    kind, s = INFO[tag]
    d, r = r"D:\tanitad", DECL
    base = [("--arm", "M2b"), ("--seed", str(s)), ("--grid", "pool4x10"), ("--out", r + rf"\M2b_s{s}_recft_CLR"), ("--cache-dir", d + r"\cache_full"),
            ("--v9-train", d + r"\data\v9\v9_labels_train.npz"), ("--v9-train-md5", "83e30bd3791f0767707eade5f4671759"), ("--lead-train", d + r"\data\v9\lead_train4572.npz"),
            ("--speed-train", d + r"\data\v9\refcv6_speed_max_v8_train.jsonl"), ("--join-train", r + r"\join_train.npz"), ("--p10", r + r"\p10\p10_prior.pt"), ("--kin-mode", "yaw"),
            ("--tracks", d + r"\cache_full\native_tracks.pt"), ("--exclude", "540:600"), ("--batch", "64"), ("--steps-mult", "4"), ("--device", "cuda"),
            ("--init-ckpt", r + rf"\M2b_s{s}_dagoacc\ckpt_last.pt"), ("--steps", str(STEPS)), ("--lr", "1e-4"), ("--warmup", "100"), ("--p-gt-start", "0.2"), ("--p-gt-end", "0.2"),
            ("--ego-perturb-p", "0"), ("--w-acc", "49.6969")]
    rec = [("--recover-p", str(RP["p"])), ("--recover-target", RP["target"]), ("--recover-rejoin-steps", str(RP["rejoin"])), ("--recover-seed", str(s)), ("--recover-dev-law", RP["law"])]
    toks = [t for kv in base + rec for t in kv]
    return toks + ["--level-head", "on", "--w-level", W]


def parse_flags(toks):
    d, bad, i = {}, [], 0
    while i < len(toks):
        t = toks[i]
        if not t.startswith("--"):
            bad.append(f"stray token {t!r} where a --flag was expected"); i += 1; continue
        if t in BOOL_FLAGS:
            d.setdefault(t, []).append(""); i += 1; continue
        if i + 1 >= len(toks) or toks[i + 1].startswith("--"):
            bad.append(f"flag {t} has no value"); d.setdefault(t, []).append(None); i += 1; continue
        d.setdefault(t, []).append(toks[i + 1]); i += 2
    return d, bad


def audit_argv(tag, toks):
    """[] iff `toks` is the frozen argv of `tag`. Every difference is named: the flag, what the argv has, what the card's argv has."""
    want_toks = expected_tokens(tag)
    got, bad = parse_flags(list(toks)); want, _ = parse_flags(want_toks)
    if not w_ok(W):
        bad.append(f"the frozen W is {W!r}: no positive number is pinned (cl_prereg/FROZEN_W.txt, then mk_clr_package.py)")
    if PLACEHOLDER in toks:
        bad.append(f"the argv still carries the placeholder {PLACEHOLDER}")
    for f, vs in got.items():
        if len(vs) > 1:
            bad.append(f"{f} is given {len(vs)} times {vs} (argparse keeps the last: the others are dead values)")
    for f in want:
        if f not in got:
            bad.append(f"missing {f} (the frozen {tag} argv has {want[f][0]!r})")
    for f in got:
        if f not in want:
            bad.append(f"unexpected flag {f} {got[f]}: not in the frozen {tag} argv (arm CLR = RAPR-P01's argv + the two level-head tokens, nothing else)")
        elif got[f] != want[f] and len(got[f]) == 1:
            bad.append(f"{f} = {got[f][0]!r} but the frozen {tag} argv has {want[f][0]!r}")
    if not bad and list(toks) != want_toks:
        bad.append("the flags and values are the frozen ones but the token order differs (the gate token binds the token list)")
    return bad


def shipped_runs(kind=None):
    return [t for t, (k, _) in INFO.items() if kind is None or k == kind]


def stack_problems():
    bad = []
    for rel, want in SHIPPED.items():
        p = os.path.join(REPO, *rel.split("/"))
        got = sha_file(p) if os.path.exists(p) else None
        if got != want:
            bad.append(f"{rel}: sha256 {got} != shipped {want} (box 2's repo does not hold the shipped bytes)")
    return bad


def cmd_preflight(args):
    free = shutil.disk_usage(DISK).free
    if args and args[0] == "run":
        if free < FLOOR:
            print(f"ZZCLR-STOP disk floor: {free} bytes free on {DISK} < {FLOOR}"); return 9
        return 0
    if args != ["chain"]:
        print("preflight needs chain | run <tag>"); return 2
    bad = []
    if free < FLOOR:
        bad.append(f"disk floor: {free} < {FLOOR}")
    bad += stack_problems()
    tags = list(INFO)
    if not w_ok(W):
        bad.append(f"W is {W!r}: the chain needs CL's frozen positive W")
    fw = os.path.join(HOME, "FROZEN_W.txt")
    if not os.path.exists(fw):
        bad.append("FROZEN_W.txt is not shipped")
    elif PINS.get("frozen_w_sha256") != sha_file(fw):
        bad.append("FROZEN_W.txt differs from the pinned bytes")
    for tag in tags:
        try:
            toks = shipped_line(tag).split(" ")
            bad += [f"{tag}: argv audit: {m}" for m in audit_argv(tag, toks)]
            ic = loc(toks[toks.index("--init-ckpt") + 1])
            if not os.path.exists(ic):
                bad.append(f"{tag}: init ckpt {ic} missing")
        except Exception as e:                              # noqa: BLE001
            bad.append(f"{tag}: {type(e).__name__}: {e}")
    if bad:
        print("ZZCLR-PREFLIGHT-FAIL " + " | ".join(bad)); return 3
    print(f"preflight OK (chain): disk free {free}, four shipped stack files match their pins, argv audited ({', '.join(tags)}), init ckpts, W {W}")
    return 0


def cmd_pin(tag):
    try:
        line = shipped_line(tag); toks = line.split(" ")
        bad = audit_argv(tag, toks)
    except (AssertionError, OSError, ValueError) as e:
        print(f"REFUSED: {tag}: the shipped argv file is unreadable or malformed: {type(e).__name__}: {e}"); return 2
    if bad:
        print(f"REFUSED: {tag}: argv audit against the frozen {tag} argv: " + " | ".join(bad)); return 2
    gout = os.path.join(GATE, os.path.basename(out_of(toks)))
    gt = list(toks); gt[gt.index("--out") + 1] = gout; gt.append("--selfcheck-only")
    back = [t for t in gt if t != "--selfcheck-only"]; back[back.index("--out") + 1] = out_of(toks)
    assert back == toks, "the gate argv is not the training argv with --out and --selfcheck-only changed"
    for name, text in (("train", line), ("gate", " ".join(gt))):
        p = os.path.join(PIN, f"{tag}.{name}.txt")
        if os.path.exists(p) and open(p, encoding="utf-8").read().strip() != text:
            print(f"REFUSED: {p} pins another {name} argv than the shipped one (move it aside only if the levers were deliberately changed)"); return 2
        write_text(p, text + "\r\n")
    for d in (loc(out_of(toks)), loc(gout), LOGS):
        os.makedirs(d, exist_ok=True)
    print(f"pinned {tag}: sha256(tokens) {tokens_sha(toks)}")
    return 0


def num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and x == x and abs(x) != float("inf")


def level_problems(lp, w):
    """The independent read of selfcheck.json -> level_head_probe (the trainer asserts each of these; this is a second reading of the written record). `w` = the weight the argv declared."""
    p = []
    if not isinstance(lp, dict):
        return ["level_head_probe is absent from selfcheck.json (the lever-ON key)"]
    if lp.get("pass") is not True:
        p.append(f"level_head_probe pass = {lp.get('pass')!r}")
    if lp.get("w_level_declared") != float(w):
        p.append(f"level_head_probe.w_level_declared = {lp.get('w_level_declared')!r}, the argv declares {float(w)!r}")
    for k in ("term", "step0_level_term", "step0_total", "head_grad_diff", "e_level_grad_at_w0", "plan_moves_with_token", "w_level_for_10pct_share"):
        v = lp.get(k)
        if not (num(v) and v > 0.0):
            p.append(f"level_head_probe.{k} = {v!r} (a finite number > 0 is required)")
    for k in ("head_grad_at_w0_detach_control", "control_repeat_total_diff", "control_repeat_grad_diff", "control_allfalse_term", "control_allfalse_total_diff", "control_allfalse_head_grad_diff", "control_prior_grad_diff"):
        if lp.get(k) != 0.0:
            p.append(f"level_head_probe.{k} = {lp.get(k)!r}: the control must read EXACTLY 0.0")
    try:
        c, wx, t0, tw = lp["contribution"], lp["w_level_x_term"], lp["total_0"], lp["total_w"]
        if abs(c - wx) > 2e-3 * abs(wx) + 1e-5 * max(abs(t0), abs(tw), 1.0):
            p.append(f"the total moved by {c} when the weight went 0 -> {w}, w_level x term is {wx}: the term does not reach the loss with its weight")
    except (KeyError, TypeError):
        p.append("level_head_probe lacks contribution / w_level_x_term / total_0 / total_w")
    try:                                                    # the number the card's W comes from, recomputed from the step-0 record
        want10 = (0.1 / 0.9) * lp["step0_total_without_level"] / lp["step0_level_term"]
        if abs(want10 - lp["w_level_for_10pct_share"]) > 1e-6 * max(abs(want10), 1.0):
            p.append(f"w_level_for_10pct_share {lp['w_level_for_10pct_share']} != (0.1/0.9) x step0_total_without_level / step0_level_term = {want10}")
    except (KeyError, TypeError, ZeroDivisionError):
        p.append("level_head_probe lacks the step-0 record (step0_total_without_level / step0_level_term)")
    return p


def recover_problems(lvs, rp, info, s):
    """RAPR-P01's own gate (rnext_check.py cmd_gate, kind RAPRP01), read the same way: the BUILT perturber's levers (levers.recover), the probe, and the absence of any deviation table or bank under the a-priori law."""
    p = []
    if rp.get("pass") is not True:
        p.append(f"recover_probe pass = {rp.get('pass')!r}")
    lv = lvs.get("recover") or {}
    want = {"recover_p": RP["p"], "recover_dev_law": RP["law"], "recover_target": RP["target"], "recover_rejoin_steps": RP["rejoin"], "recover_seed": s}
    for k, w in want.items():
        if lv.get(k) != w:
            p.append(f"built recover lever {k} = {lv.get(k)!r}, registered {w!r} (RAPR-P01's recovery recipe)")
    if info.get("recover"):
        p.append("a deviation table was built (selfcheck info.recover is present) under the a-priori law")
    if lvs.get("recover_table") is not None:
        p.append(f"levers.recover_table is not None ({str(lvs.get('recover_table'))[:80]})")
    if lv.get("recover_dev_bank") is not None or lv.get("recover_dev_bank_sha256") is not None:
        p.append(f"a bank is built under the a-priori law (recover_dev_bank {lv.get('recover_dev_bank')!r}, sha256 {lv.get('recover_dev_bank_sha256')!r})")
    return p


def common_gate_problems(tag, rc):
    """(problems, sc, info, gdir, toks, log): the process, the log, declared_vs_built, the init ckpt (carried whole, EXACTLY the six declared keys fresh), the data, the steps, the built level lever, the built recovery lever, every other lever as C0's."""
    kind, s = INFO[tag]
    toks = shipped_line(tag).split(" ")
    gdir = os.path.join(GATE, os.path.basename(out_of(toks)))
    p = []
    if str(rc) != "0":
        p.append(f"the gate process exited rc {rc}")
    try:
        sc = json.load(open(os.path.join(gdir, "selfcheck.json"), encoding="utf-8"))
    except Exception as e:                                  # noqa: BLE001
        sc = {}; p.append(f"no readable selfcheck.json: {type(e).__name__}")
    log = ""
    try:
        log = open(os.path.join(LOGS, f"{tag}.gate.log"), encoding="utf-8", errors="replace").read()
    except OSError:
        p.append("no gate log")
    if "SELFCHECK PASS" not in log:
        p.append("'SELFCHECK PASS' not in the gate log")
    if sc.get("declared_vs_built") != "PASS":
        p.append(f"declared_vs_built = {sc.get('declared_vs_built')}")
    info = sc.get("info") or {}
    ic = info.get("init_ckpt") or {}
    if ic.get("sha256") != INIT_SHA[s]:
        p.append(f"init ckpt sha256 {ic.get('sha256')} != DAG-OACC s{s}'s {INIT_SHA[s][:12]}")
    fresh, declared = ic.get("fresh_keys"), ic.get("declared_new_keys")
    if not (ic.get("applied") is True and ic.get("n_mismatch_after_load") == 0 and isinstance(fresh, list) and fresh == declared and len(fresh) == 6 and all(str(k).startswith(LEVEL_PREFIXES) for k in fresh)):
        p.append(f"init ckpt: applied {ic.get('applied')!r}, mismatch after load {ic.get('n_mismatch_after_load')!r}, fresh {fresh!r}, declared new {declared!r} "
                 f"(want applied, mismatch 0 and exactly the six lvl_head. / e_level. keys fresh, equal to the declared ones)")
    if info.get("train_sha256") != KP1["train_sha256"]:
        p.append(f"gate train_sha256 {info.get('train_sha256')} != DAG-OACC's (KP1)")
    if sc.get("steps_total") != STEPS:
        p.append(f"steps_total {sc.get('steps_total')} != {STEPS}")
    lvs = sc.get("levers") or {}
    if lvs.get("level_head") != "on" or lvs.get("w_level") != float(W):
        p.append(f"built level_head = {lvs.get('level_head')!r}, w_level = {lvs.get('w_level')!r} (registered on, {float(W)!r})")
    if lvs.get("level_token_zero_init") is not True or lvs.get("level_vision_only_source") is not True:
        p.append(f"built level_token_zero_init = {lvs.get('level_token_zero_init')!r}, level_vision_only_source = {lvs.get('level_vision_only_source')!r} (both must be True)")
    p += recover_problems(lvs, sc.get("recover_probe") or {}, info, s)
    other = {"dagger_p": lvs.get("dagger_p"), "launch_intent_w": lvs.get("launch_intent_w"), "ego_perturb_p": (lvs.get("ego_perturb") or {}).get("ego_perturb_p")}
    if any(v != 0.0 for v in other.values()) or (lvs.get("dagger") or {}).get("dagger_bank") is not None:
        p.append(f"another lever is on ({other}, dagger_bank {(lvs.get('dagger') or {}).get('dagger_bank')!r}): arm CLR is RAPR-P01 + the head, every other lever as C0's")
    c0 = {"w_acc": 49.6969, "p_gt_start": 0.2, "p_gt_end": 0.2, "kin_mode": "yaw"}
    for k, w in c0.items():
        if lvs.get(k) != w:
            p.append(f"built lever {k} = {lvs.get(k)!r}, C0's is {w!r}")
    p += level_problems(sc.get("level_head_probe"), W)
    return p, sc, info, gdir, toks, log


def level_share(lp, w):
    t, base = lp.get("step0_level_term"), lp.get("step0_total_without_level")
    return (float(w) * t / (base + float(w) * t)) if num(t) and num(base) and (base + float(w) * t) else None


def cmd_gate(tag, rc):
    kind, s = INFO[tag]
    p, sc, info, gdir, toks, log = common_gate_problems(tag, rc)
    lp = sc.get("level_head_probe") or {}
    share = level_share(lp, W)
    if share is None or not 0.0 < share < 0.5:
        p.append(f"level share at step 0 = {share!r} (W {W} puts the level term near a tenth of the total)")
    elif s == 0 and not SHARE_LO <= share <= SHARE_HI:       # seed 0 = CL's probe seed and batch: the frozen W must reproduce the tenth within the card's window; seed 1 reads a different batch: recorded only
        p.append(f"level share at step 0 = {share:.4f} on seed 0, the card gates [{SHARE_LO}, {SHARE_HI}] (W {W})")
    if p:
        fail(tag, "gate: " + "; ".join(p), gate_rc=rc); return 1
    tok = {"kind": "launch-gate-token", "verdict": "PASS", "prereg": PINS["card_file"], "prereg_sha256": CARD_SHA, "run": tag, "written_by": "clr_check.py gate (never by hand)", "at": now(),
           "binding": {"commit": TIP, "argv_sha256": tokens_sha(toks), "argv_sha256_method": "sha256 of json.dumps(token list, separators=(',',':'))", "w_level": W,
                       "data_sha256": info.get("train_sha256"), "init_ckpt_sha256": INIT_SHA[s], "selfcheck_sha256": sha_file(os.path.join(gdir, "selfcheck.json")), "gate_argv_sha_trainer": sc.get("argv_sha"),
                       "shipped_sha256": SHIPPED, "level_share_at_step0": share, "n_params_trainable": (sc.get("levers") or {}).get("n_params_trainable")}}
    write_text(os.path.join(RC, f"gate_token_{tag}.json"), json.dumps(tok, indent=1))
    print(f"ZZCLR-GATE-PASS {tag} level_share_at_step0 {share:.4f}")
    return 0


def cmd_post(tag, rc):
    kind, s = INFO[tag]
    toks = shipped_line(tag).split(" "); out = loc(out_of(toks))
    p = []
    if str(rc) != "0":
        p.append(f"the trainer exited rc {rc}")
    for f in ("summary.json", "argv.json", "ckpt_last.pt"):
        if not os.path.exists(os.path.join(out, f)):
            p.append(f"{f} missing")
    ev = {}
    try:
        sm = json.load(open(os.path.join(out, "summary.json"), encoding="utf-8"))
        ev["steps"] = sm.get("steps")
        if sm.get("steps") != STEPS:
            p.append(f"summary steps {sm.get('steps')} != {STEPS}")
        for k, v in KP1.items():
            if sm.get(k) != v:
                p.append(f"KP1: summary {k} {sm.get(k)} != DAG-OACC's")
        ev["level_head"], ev["w_level"], ev["recover_n"], ev["recover_share"] = sm.get("level_head"), sm.get("w_level"), sm.get("recover_n"), sm.get("recover_share")
        if sm.get("level_head") != "on" or sm.get("w_level") != float(W):
            p.append(f"summary level_head {sm.get('level_head')!r} / w_level {sm.get('w_level')!r}, registered on / {float(W)!r} (the run record of a level-head-ON arm)")
        rn = sm.get("recover_n")
        if not (isinstance(rn, (int, float)) and not isinstance(rn, bool) and rn > 0):   # KR1, as RAPR-P01's post: the recovery really perturbed samples
            p.append(f"KR1: arm CLR run perturbed no sample (recover_n {rn!r}; RAPR-P01's recipe has p 0.1)")
    except Exception as e:                                  # noqa: BLE001
        p.append(f"summary.json unreadable: {type(e).__name__}")
    try:
        av = json.load(open(os.path.join(out, "argv.json"), encoding="utf-8"))
        ev["argv_match_pin"] = (av.get("argv") == toks)
        if av.get("argv") != toks:
            p.append("argv.json != the pinned training argv")
    except Exception as e:                                  # noqa: BLE001
        p.append(f"argv.json unreadable: {type(e).__name__}")
    ck = os.path.join(out, "ckpt_last.pt")
    if os.path.exists(ck):
        ev["ckpt_bytes"] = os.path.getsize(ck)
    if p:
        fail(tag, "train: " + "; ".join(p), train_rc=rc, evidence=ev)
    else:
        ev["ckpt_sha256"] = sha_file(ck)
        marker(tag, "done", {"evidence": ev, "out": out})
    return 0


def cmd_gate_cached(tag):
    """0 only if a launch-gate token written by `gate` for THIS tag is still bound to exactly this commit, this pinned argv, this selfcheck.json, these shipped bytes, this card and this W; any difference means 'not cached'."""
    kind, s = INFO[tag]
    try:
        toks = shipped_line(tag).split(" ")
        tok = json.load(open(os.path.join(RC, f"gate_token_{tag}.json"), encoding="utf-8")); b = tok["binding"]
        gdir = os.path.join(GATE, os.path.basename(out_of(toks)))
        ok = (tok.get("verdict") == "PASS" and b["commit"] == TIP and b["argv_sha256"] == tokens_sha(toks) and b["w_level"] == W and tok.get("prereg_sha256") == CARD_SHA
              and b["selfcheck_sha256"] == sha_file(os.path.join(gdir, "selfcheck.json")) and b["init_ckpt_sha256"] == INIT_SHA[s]
              and b["shipped_sha256"] == SHIPPED and all(sha_file(os.path.join(REPO, *rel.split("/"))) == h for rel, h in SHIPPED.items()))
    except Exception:                                       # noqa: BLE001
        ok = False
    print(f"gate-cached {tag}: {'YES' if ok else 'no'}")
    return 0 if ok else 1


def cmd_finish():
    st = {}
    for tag in shipped_runs("CLR"):
        st[tag] = "done" if os.path.exists(os.path.join(RC, f"RUN_{tag}.done")) else ("FAILED" if os.path.exists(os.path.join(RC, f"RUN_{tag}.FAILED")) else "none")
    write_marker_file(os.path.join(RC, "CHAIN_CLR.done"), json.dumps({"at": now(), "tip": TIP, "runs": st}, indent=1))
    print("ZZCLR-CHAIN-DONE " + json.dumps(st))
    return 0


def main(a):
    c = a[0]
    if c == "preflight": return cmd_preflight(a[1:])
    if c == "pin": return cmd_pin(a[1])
    if c == "gate": return cmd_gate(a[1], a[2])
    if c == "post": return cmd_post(a[1], a[2])
    if c == "fail": fail(a[1], " ".join(a[2:])); return 0
    if c == "stopped": write_marker_file(os.path.join(RC, "CHAIN_CLR.STOPPED"), json.dumps({"at": now(), "reason": " ".join(a[1:])})); print("ZZCLR-STOPPED " + " ".join(a[1:])); return 0
    if c == "finish": return cmd_finish()
    if c == "gate-cached": return cmd_gate_cached(a[1])
    print("unknown command", c); return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
