"""LEVELHEAD arm-LLR checker (runs on box 2 with the venv python; stdlib only, no torch, no network, no subprocess). Every marker and every pin is written HERE, one code path. GENERATED into box2_files/llr_check.py by mk_llr_package.py (the pins below are filled in).
Adapted from the arm-CLR checker (clr_check.py): same init, same data, same gate protocol; the runs are LLR_0 LLR_1 (card PREREG_LEVELHEAD_A2.md, Arms + A2_RULING_1.md): CLR's argv + `--lead-head on --w-lead WL --aux-cond WIRING` (WL from the W_lead probe, WIRING from CLR's O-sensitivity: both in a2_prereg/FROZEN_LLR.txt),
and the W_lead probe WLPROBE_0 (a --selfcheck-only job: LLR_0's argv with `--w-lead 1`, its own --out M2b_s0_wleadprobe_LLR, the frozen wiring).
  preflight wprobe|chain  job start: disk floor, the four shipped stack files' sha256, the argv audit of the job's argv files, FROZEN_LLR.txt = the pinned bytes (wprobe: the wiring frozen; chain: the wiring AND W_lead frozen), FROZEN_W.txt = the pinned W, init ckpts
  preflight run <tag>     before each run: disk floor (exit 9 = STOP the chain)
  pin <tag>               argv audit (a changed lever or a placeholder is REFUSED, rc 2), pin the argv (a re-pin that differs is REFUSED), derive the gate argv (the training argv with --out and --selfcheck-only changed, asserted), make the dirs
  gate-cached <tag>       0 iff a PASS token for this tag is still bound to this commit / argv / selfcheck / shipped bytes / card / W / W_lead / wiring
  gate <tag> <rc>         read the gate's selfcheck.json + log, assert the registered controls, the BUILT level lever, the BUILT lead lever (W_lead, wiring read back: add = e_level and e_lead zero-init, token = e_aux built and both additive paths absent),
                          the BUILT recovery lever (RAPR-P01's), every other lever as C0's; write the per-run launch-gate token, FAIL => RUN_<tag>.FAILED, exit 1. WLPROBE_0: write WLPROBE.done (carrying lead_head_probe.w_lead_for_10pct_share) or WLPROBE.FAILED
  post <tag> <rc>         after training: summary / argv / ckpt checks (recover_n > 0; lead_head on, w_lead, aux_cond; lead label file = the gate's); PASS => RUN_<tag>.done, else RUN_<tag>.FAILED; exit 0 always
  fail <tag> <reason>     write RUN_<tag>.FAILED (WLPROBE_0: WLPROBE.FAILED)
  stopped <reason>        write CHAIN_LLR.STOPPED;  wprobe-stopped <reason>  write WLPROBE.STOPPED
  finish                  write CHAIN_LLR.done
Paths come from LLR_HOME / LLR_RUNSM / LLR_REPO / LLR_DISK / LLR_FLOOR_BYTES (defaults are box 2's) so the same file is exercised on the dev box."""
import hashlib, json, os, shutil, sys, time

HOME = os.environ.get("LLR_HOME", r"D:\tanitad\llr")
RUNSM = os.environ.get("LLR_RUNSM", r"D:\tanitad\runs\v8f_mini_full")
REPO = os.environ.get("LLR_REPO", r"D:\tanitad\repo")
DISK = os.environ.get("LLR_DISK", "D:\\")
FLOOR = int(float(os.environ.get("LLR_FLOOR_BYTES", "8e9")))
RC = os.path.join(RUNSM, "llr"); GATE = os.path.join(RUNSM, "gate_llr"); PIN = os.path.join(RC, "pin"); LOGS = os.path.join(RC, "logs")
DECL = r"D:\tanitad\runs\v8f_mini_full"
PINS = json.loads(r'''__PINS_JSON__''')
SHIPPED = {x["rel"]: x["sha256"] for x in PINS["stack"]}          # the four files over box 2's repo (launch_llr.sh ships them); mk_llr_package.py recomputes this from stack_ship/
TRAINER, MODEL = "stack/scripts/v8f_mini_train.py", "stack/tanitad/models/v8f_mini.py"
TIP = PINS["tip_commit"]                                         # the git tip the two unchanged files were staged against
WL, WIRING = PINS["w_lead"], PINS["wiring"]                           # W_lead string and wiring exactly as in the argv; a placeholder (__WL__ / __WIRING__) while the package is not frozen
W = PINS["w"]                                                    # CL's frozen weight string, exactly as it is in the argv (cl_prereg/FROZEN_W.txt); there is no placeholder state
CARD_SHA = PINS["card_sha256"]
PLACEHOLDER = "__W__"                                              # never valid: the generator refuses to run without a frozen W; kept so a stray placeholder in an argv is still named
RP = {"p": 0.1, "law": "apriori", "target": "rejoin", "rejoin": 10}   # RAPR-P01's recovery lever (rnext_check.py KINDS["RAPRP01"]; the seed is the run's seed)
SHARE_LO, SHARE_HI = 0.08, 0.12                                  # the card's window for the step-0 level share at seed 0
INFO = {"LLR_0": ("LLR", 0), "LLR_1": ("LLR", 1), "WLPROBE_0": ("WLPROBE", 0)}
INIT_SHA = {0: "67265ec1666fae3a6d51e924ee06dfb68b2cd0630afc62e42f3a46182b8bc45f", 1: "90b01474819082a6d32a9000d63a4954e0074916140abdbbcb67b353f7ac3333"}   # DAG-OACC ckpt_last.pt (MEASURED, as rnext_check.py)
KP1 = {"train_sha256": "37f4f9ee468e0872134074a82d63c2915458596fdbcf3d0fcf5b09f1fe0d8cc1",
       "train_window_list_sha256": "e144adcd9b6273f2b44ad7dfa162931b90e159f7b3911e0893b25777b2ae8168"}
STEPS = 3000
LEVEL_PREFIXES = ("lvl_head.", "e_level.")
_LV4 = ["lvl_head.0.weight", "lvl_head.0.bias", "lvl_head.2.weight", "lvl_head.2.bias"]; _LD4 = ["lead_head.0.weight", "lead_head.0.bias", "lead_head.2.weight", "lead_head.2.bias"]
FRESH = {"add": sorted(_LV4 + ["e_level.weight", "e_level.bias"] + _LD4 + ["e_lead.weight", "e_lead.bias"]),      # the keys the declared levers add to DAG-OACC's checkpoint, per wiring (smoke on the real C0 init: 12 / 10)
         "token": sorted(_LV4 + _LD4 + ["e_aux.weight", "e_aux.bias"])}
PH_WL, PH_WIRING = "__WL__", "__WIRING__"                          # never valid: a stray placeholder in an argv or a pin is named and refused
AUX_TOKEN_DIM = 3
LEAD_MIN_CLIP_COVERAGE = 0.5                                      # A2_RULING_1: the gate refuses lead labels covering < 50 % of the TRAIN clips
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
    return os.path.join(HOME, "argv", tag + (".gate.txt" if tag == "WLPROBE_0" else ".train.txt"))


def shipped_line(tag):
    with open(argv_file(tag), encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    assert len(lines) == 1, f"{tag}: the shipped argv file must hold exactly one line, has {len(lines)}"
    return lines[0]


def loc(p):                                                 # identity on box 2; maps the declared run root onto LLR_RUNSM when the dev-box test hook points it elsewhere
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
    return "WLPROBE" if tag == "WLPROBE_0" else f"RUN_{tag}"


def marker(tag, kind, payload):
    p = os.path.join(RC, f"{mname(tag)}.{kind}")
    if os.path.exists(p):                                   # never overwrite, never delete: move aside
        os.replace(p, p + ".prev_" + time.strftime("%Y%m%d_%H%M%S"))
    write_text(p, json.dumps({"tag": tag, "at": now(), "tip": TIP, **payload}, indent=1, default=str))
    print(f"ZZLLR-{kind.upper()} {tag} {payload.get('reason', '')}")


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
    base = [("--arm", "M2b"), ("--seed", str(s)), ("--grid", "pool4x10"), ("--out", r + (rf"\M2b_s{s}_recft_LLR" if kind == "LLR" else rf"\M2b_s{s}_wleadprobe_LLR")), ("--cache-dir", d + r"\cache_full"),
            ("--v9-train", d + r"\data\v9\v9_labels_train.npz"), ("--v9-train-md5", "83e30bd3791f0767707eade5f4671759"), ("--lead-train", d + r"\data\v9\lead_train4572.npz"),
            ("--speed-train", d + r"\data\v9\refcv6_speed_max_v8_train.jsonl"), ("--join-train", r + r"\join_train.npz"), ("--p10", r + r"\p10\p10_prior.pt"), ("--kin-mode", "yaw"),
            ("--tracks", d + r"\cache_full\native_tracks.pt"), ("--exclude", "540:600"), ("--batch", "64"), ("--steps-mult", "4"), ("--device", "cuda"),
            ("--init-ckpt", r + rf"\M2b_s{s}_dagoacc\ckpt_last.pt"), ("--steps", str(STEPS)), ("--lr", "1e-4"), ("--warmup", "100"), ("--p-gt-start", "0.2"), ("--p-gt-end", "0.2"),
            ("--ego-perturb-p", "0"), ("--w-acc", "49.6969")]
    rec = [("--recover-p", str(RP["p"])), ("--recover-target", RP["target"]), ("--recover-rejoin-steps", str(RP["rejoin"])), ("--recover-seed", str(s)), ("--recover-dev-law", RP["law"])]
    toks = [t for kv in base + rec for t in kv]
    toks = toks + ["--level-head", "on", "--w-level", W, "--lead-head", "on", "--w-lead", (WL if kind == "LLR" else "1"), "--aux-cond", WIRING]
    return toks + ["--selfcheck-only"] if kind == "WLPROBE" else toks


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
        bad.append(f"the frozen W is {W!r}: no positive number is pinned (cl_prereg/FROZEN_W.txt, then mk_llr_package.py)")
    for ph in (PLACEHOLDER, PH_WL, PH_WIRING):
        if ph in toks:
            bad.append(f"the argv still carries the placeholder {ph}")
    if WIRING not in ("add", "token"):
        bad.append(f"the frozen wiring is {WIRING!r}: it must be add or token (a2_prereg/FROZEN_LLR.txt from CLR's O-sensitivity: freeze_llr.py wiring --write, then mk_llr_package.py)")
    if INFO[tag][0] == "LLR" and not w_ok(WL):
        bad.append(f"the frozen W_lead is {WL!r}: no positive number is pinned (W_lead probe -> freeze_llr.py wlead --write -> mk_llr_package.py)")
    for f, vs in got.items():
        if len(vs) > 1:
            bad.append(f"{f} is given {len(vs)} times {vs} (argparse keeps the last: the others are dead values)")
    for f in want:
        if f not in got:
            bad.append(f"missing {f} (the frozen {tag} argv has {want[f][0]!r})")
    for f in got:
        if f not in want:
            bad.append(f"unexpected flag {f} {got[f]}: not in the frozen {tag} argv (arm LLR = CLR's argv + the three lead-head tokens, nothing else)")
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
            print(f"ZZLLR-STOP disk floor: {free} bytes free on {DISK} < {FLOOR}"); return 9
        return 0
    if not args or args[0] not in ("wprobe", "chain"):
        print("preflight needs wprobe | chain | run <tag>"); return 2
    mode = args[0]
    bad = []
    if free < FLOOR:
        bad.append(f"disk floor: {free} < {FLOOR}")
    bad += stack_problems()
    tags = ["WLPROBE_0"] if mode == "wprobe" else ["LLR_0", "LLR_1"]
    if not w_ok(W):
        bad.append(f"W is {W!r}: the job needs CL's frozen positive W")
    fw = os.path.join(HOME, "FROZEN_W.txt")
    if not os.path.exists(fw):
        bad.append("FROZEN_W.txt is not shipped")
    elif PINS.get("frozen_w_sha256") != sha_file(fw):
        bad.append("FROZEN_W.txt differs from the pinned bytes")
    if WIRING not in ("add", "token"):
        bad.append(f"the wiring is not frozen ({WIRING!r})")
    if mode == "chain" and not w_ok(WL):
        bad.append(f"W_lead is not frozen ({WL!r}): the chain needs the probe's frozen positive W_lead")
    fl = os.path.join(HOME, "FROZEN_LLR.txt")
    if not os.path.exists(fl):
        bad.append("FROZEN_LLR.txt is not shipped")
    elif PINS.get("frozen_llr_sha256") != sha_file(fl):
        bad.append("FROZEN_LLR.txt differs from the pinned bytes")
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
        print("ZZLLR-PREFLIGHT-FAIL " + " | ".join(bad)); return 3
    print(f"preflight OK ({mode}): disk free {free}, four shipped stack files match their pins, argv audited ({', '.join(tags)}), init ckpts, W {W}, W_lead {WL}, wiring {WIRING}")
    return 0


def cmd_pin(tag):
    try:
        line = shipped_line(tag); toks = line.split(" ")
        bad = audit_argv(tag, toks)
    except (AssertionError, OSError, ValueError) as e:
        print(f"REFUSED: {tag}: the shipped argv file is unreadable or malformed: {type(e).__name__}: {e}"); return 2
    if bad:
        print(f"REFUSED: {tag}: argv audit against the frozen {tag} argv: " + " | ".join(bad)); return 2
    if tag == "WLPROBE_0":                                  # the file IS the gate argv (--selfcheck-only already last); its --out is its own folder
        pairs, gout = (("gate", line),), out_of(toks)
    else:
        gout = os.path.join(GATE, os.path.basename(out_of(toks)))
        gt = list(toks); gt[gt.index("--out") + 1] = gout; gt.append("--selfcheck-only")
        back = [t for t in gt if t != "--selfcheck-only"]; back[back.index("--out") + 1] = out_of(toks)
        assert back == toks, "the gate argv is not the training argv with --out and --selfcheck-only changed"
        pairs = (("train", line), ("gate", " ".join(gt)))
    for name, text in pairs:
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


def level_problems(lp, w, wiring):
    """The independent read of selfcheck.json -> level_head_probe (the trainer asserts each of these; this is a second reading of the written record). `w` = the weight the argv declared."""
    p = []
    if not isinstance(lp, dict):
        return ["level_head_probe is absent from selfcheck.json (the lever-ON key)"]
    if lp.get("pass") is not True:
        p.append(f"level_head_probe pass = {lp.get('pass')!r}")
    if lp.get("w_level_declared") != float(w):
        p.append(f"level_head_probe.w_level_declared = {lp.get('w_level_declared')!r}, the argv declares {float(w)!r}")
    for k in ("term", "step0_level_term", "step0_total", "head_grad_diff", ("e_level_grad_at_w0" if wiring == "add" else "e_aux_level_col_grad_at_w0"), "plan_moves_with_token", "w_level_for_10pct_share"):
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


def lead_problems(lp, wl, wiring):
    """The independent read of selfcheck.json -> lead_head_probe (the trainer asserts each of these; this is a second reading of the written record). `wl` = the W_lead the argv declared (1.0 for the probe)."""
    p = []
    if not isinstance(lp, dict) or lp.get("lead") == "off":
        return ["lead_head_probe is absent from selfcheck.json (the lever-ON key)"]
    if lp.get("pass") is not True:
        p.append(f"lead_head_probe pass = {lp.get('pass')!r}")
    if lp.get("w_lead_declared") != float(wl):
        p.append(f"lead_head_probe.w_lead_declared = {lp.get('w_lead_declared')!r}, the argv declares {float(wl)!r}")
    if lp.get("aux_cond") != wiring:
        p.append(f"lead_head_probe.aux_cond = {lp.get('aux_cond')!r}, the frozen wiring is {wiring!r}")
    reach = "e_lead_grad_at_w0" if wiring == "add" else "e_aux_lead_cols_grad_at_w0"
    for k in ("term", "step0_lead_term", "step0_total", "head_grad_diff", "head_grad_diff_presence_row", "head_grad_diff_gap_row", reach, "plan_moves_with_token", "w_lead_for_10pct_share", "probe_n_has", "probe_n_near"):
        v = lp.get(k)
        if not (num(v) and v > 0.0):
            p.append(f"lead_head_probe.{k} = {v!r} (a finite number > 0 is required)")
    for k in ("head_grad_at_w0_detach_control", "control_repeat_total_diff", "control_repeat_grad_diff", "control_allfalse_term", "control_allfalse_total_diff", "control_allfalse_head_grad_diff", "control_prior_grad_diff"):
        if lp.get(k) != 0.0:
            p.append(f"lead_head_probe.{k} = {lp.get(k)!r}: the control must read EXACTLY 0.0")
    vo = lp.get("vision_only")
    if not (isinstance(vo, dict) and vo.get("source_problems") == [] and vo.get("keys_read") == ["frame_ok"]):
        p.append(f"lead_head_probe.vision_only = {str(vo)[:120]} (the head must be proven vision-only: keys_read exactly ['frame_ok'], no source problems)")
    try:
        c, wx, t0, tw = lp["contribution"], lp["w_lead_x_term"], lp["total_0"], lp["total_w"]
        if abs(c - wx) > 2e-3 * abs(wx) + 1e-5 * max(abs(t0), abs(tw), 1.0):
            p.append(f"the total moved by {c} when the weight went 0 -> {wl}, w_lead x term is {wx}: the term does not reach the loss with its weight")
    except (KeyError, TypeError):
        p.append("lead_head_probe lacks contribution / w_lead_x_term / total_0 / total_w")
    try:                                                    # the number W_lead comes from, recomputed from the step-0 record
        want10 = (0.1 / 0.9) * lp["step0_total_without_lead"] / lp["step0_lead_term"]
        if abs(want10 - lp["w_lead_for_10pct_share"]) > 1e-6 * max(abs(want10), 1.0):
            p.append(f"w_lead_for_10pct_share {lp['w_lead_for_10pct_share']} != (0.1/0.9) x step0_total_without_lead / step0_lead_term = {want10}")
        sh = lead_share(lp, wl)
        if sh is None or not num(lp.get("step0_lead_share")) or abs(sh - lp["step0_lead_share"]) > 1e-6:
            p.append(f"step0_lead_share {lp.get('step0_lead_share')!r} != W x term / (step0_total_without_lead + W x term) = {sh!r}")
    except (KeyError, TypeError, ZeroDivisionError):
        p.append("lead_head_probe lacks the step-0 record (step0_total_without_lead / step0_lead_term)")
    return p


def lead_lever_problems(lvs, info, sc, toks, wl, wiring):
    """The BUILT lead / aux levers (levers.*), the wiring read back (absent keys are the build's own statement of what it did not build) and the lead label coverage (A2_RULING_1: >= 50 % of the TRAIN clips)."""
    p = []
    if lvs.get("lead_head") != "on" or lvs.get("w_lead") != float(wl):
        p.append(f"built lead_head = {lvs.get('lead_head')!r}, w_lead = {lvs.get('w_lead')!r} (registered on, {float(wl)!r})")
    if lvs.get("lead_vision_only_source") is not True or lvs.get("aux_vision_only_source") is not True:
        p.append(f"built lead_vision_only_source = {lvs.get('lead_vision_only_source')!r}, aux_vision_only_source = {lvs.get('aux_vision_only_source')!r} (both must be True)")
    if lvs.get("aux_cond") != wiring:
        p.append(f"built aux_cond = {lvs.get('aux_cond')!r}, the frozen wiring is {wiring!r}")
    if wiring == "add":
        if lvs.get("lead_token_zero_init") is not True or lvs.get("level_token_zero_init") is not True:
            p.append(f"add wiring: built lead_token_zero_init = {lvs.get('lead_token_zero_init')!r}, level_token_zero_init = {lvs.get('level_token_zero_init')!r} (both must be True: the step-0 forward is the init's)")
        if "aux_token_dim" in lvs:
            p.append(f"add wiring but an aux token was built (aux_token_dim {lvs.get('aux_token_dim')!r})")
    elif wiring == "token":
        if lvs.get("aux_token_dim") != AUX_TOKEN_DIM:
            p.append(f"token wiring: built aux_token_dim = {lvs.get('aux_token_dim')!r}, want {AUX_TOKEN_DIM}")
        if "lead_token_zero_init" in lvs or "level_token_zero_init" in lvs:
            p.append("token wiring but an additive path (e_lead / e_level) was built: the token REPLACES both")
    ll, ll2 = info.get("lead_labels"), sc.get("lead_labels")
    if not isinstance(ll, dict) or ll != ll2:
        p.append(f"lead labels: info.lead_labels {str(ll)[:80]} / selfcheck lead_labels {str(ll2)[:80]} (both required, equal)")
    else:
        want_path = toks[toks.index("--lead-train") + 1]
        if os.path.normcase(os.path.abspath(str(ll.get("path")))) != os.path.normcase(os.path.abspath(want_path)):
            p.append(f"lead labels were built from {ll.get('path')!r}, the argv declares {want_path!r}")
        if not (isinstance(ll.get("sha256"), str) and len(ll["sha256"]) == 64):
            p.append(f"lead labels: sha256 not recorded ({ll.get('sha256')!r})")
        if not (num(ll.get("n_has")) and ll["n_has"] > 0 and num(ll.get("n_near")) and ll["n_near"] > 0):
            p.append(f"lead labels: {ll.get('n_has')!r} labelled frames, {ll.get('n_near')!r} with a lead within 40 m (both must be > 0)")
        if not (num(ll.get("n_clips")) and num(ll.get("n_clips_with_labels")) and ll["n_clips"] > 0 and ll["n_clips_with_labels"] >= LEAD_MIN_CLIP_COVERAGE * ll["n_clips"]):
            p.append(f"lead labels: {ll.get('n_clips_with_labels')!r} of {ll.get('n_clips')!r} train clips are labelled (the gate needs >= {LEAD_MIN_CLIP_COVERAGE:.0%})")
    return p


def lead_share(lp, w):
    t, base = lp.get("step0_lead_term"), lp.get("step0_total_without_lead")
    return (float(w) * t / (base + float(w) * t)) if num(t) and num(base) and (base + float(w) * t) else None


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
    gdir = loc(out_of(toks)) if tag == "WLPROBE_0" else os.path.join(GATE, os.path.basename(out_of(toks)))
    wdecl_lead = 1.0 if tag == "WLPROBE_0" else (float(WL) if w_ok(WL) else None)
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
    fresh_want = FRESH.get(WIRING)
    if not (ic.get("applied") is True and ic.get("n_mismatch_after_load") == 0 and isinstance(fresh, list) and isinstance(declared, list) and sorted(fresh) == sorted(declared) and fresh_want is not None and sorted(fresh) == fresh_want):
        p.append(f"init ckpt: applied {ic.get('applied')!r}, mismatch after load {ic.get('n_mismatch_after_load')!r}, fresh {fresh!r}, declared new {declared!r} "
                 f"(want applied, mismatch 0 and exactly the {len(fresh_want) if fresh_want else '?'} {WIRING}-wiring keys fresh, equal to the declared ones: {fresh_want})")
    if info.get("train_sha256") != KP1["train_sha256"]:
        p.append(f"gate train_sha256 {info.get('train_sha256')} != DAG-OACC's (KP1)")
    if sc.get("steps_total") != STEPS:
        p.append(f"steps_total {sc.get('steps_total')} != {STEPS}")
    lvs = sc.get("levers") or {}
    if lvs.get("level_head") != "on" or lvs.get("w_level") != float(W):
        p.append(f"built level_head = {lvs.get('level_head')!r}, w_level = {lvs.get('w_level')!r} (registered on, {float(W)!r})")
    if lvs.get("level_vision_only_source") is not True:
        p.append(f"built level_vision_only_source = {lvs.get('level_vision_only_source')!r} (must be True)")
    if wdecl_lead is None:
        p.append(f"W_lead is {WL!r}: no positive number is pinned")
    else:
        p += lead_lever_problems(lvs, info, sc, toks, wdecl_lead, WIRING)
    p += recover_problems(lvs, sc.get("recover_probe") or {}, info, s)
    other = {"dagger_p": lvs.get("dagger_p"), "launch_intent_w": lvs.get("launch_intent_w"), "ego_perturb_p": (lvs.get("ego_perturb") or {}).get("ego_perturb_p")}
    if any(v != 0.0 for v in other.values()) or (lvs.get("dagger") or {}).get("dagger_bank") is not None:
        p.append(f"another lever is on ({other}, dagger_bank {(lvs.get('dagger') or {}).get('dagger_bank')!r}): arm LLR is RAPR-P01 + the level head + the lead head, every other lever as C0's")
    c0 = {"w_acc": 49.6969, "p_gt_start": 0.2, "p_gt_end": 0.2, "kin_mode": "yaw"}
    for k, w in c0.items():
        if lvs.get(k) != w:
            p.append(f"built lever {k} = {lvs.get(k)!r}, C0's is {w!r}")
    p += level_problems(sc.get("level_head_probe"), W, WIRING)
    if wdecl_lead is not None:
        p += lead_problems(sc.get("lead_head_probe"), wdecl_lead, WIRING)
    return p, sc, info, gdir, toks, log


def level_share(lp, w):
    t, base = lp.get("step0_level_term"), lp.get("step0_total_without_level")
    return (float(w) * t / (base + float(w) * t)) if num(t) and num(base) and (base + float(w) * t) else None


def cmd_gate(tag, rc):
    kind, s = INFO[tag]
    p, sc, info, gdir, toks, log = common_gate_problems(tag, rc)
    lp, ldp = sc.get("level_head_probe") or {}, sc.get("lead_head_probe") or {}
    if tag == "WLPROBE_0":
        if p:
            fail(tag, "wleadprobe: " + "; ".join(p), gate_rc=rc); return 1
        marker(tag, "done", {"w_lead_for_10pct_share": ldp["w_lead_for_10pct_share"], "step0_total": ldp["step0_total"], "step0_lead_term": ldp["step0_lead_term"], "step0_total_without_lead": ldp["step0_total_without_lead"], "aux_cond": WIRING,
                             "evidence": {"selfcheck_sha256": sha_file(os.path.join(gdir, "selfcheck.json")), "argv_sha_trainer": sc.get("argv_sha"), "argv_sha256": tokens_sha(toks), "init_ckpt_sha256": INIT_SHA[s], "lead_labels": info.get("lead_labels")}, "out": gdir})
        return 0
    lsh, vsh = lead_share(ldp, WL), level_share(lp, W)
    if lsh is None or not 0.0 < lsh < 0.5:
        p.append(f"lead share at step 0 = {lsh!r} (W_lead {WL} puts the lead term near a tenth of the total)")
    elif s == 0 and WIRING == "add" and not SHARE_LO <= lsh <= SHARE_HI:   # seed 0 = the probe's seed and batch, add wiring (the step-0 forward is the init's): the frozen W_lead must reproduce the tenth within the card's window; token (step-0 plan differs by construction) and seed 1: recorded only
        p.append(f"lead share at step 0 = {lsh:.4f} on seed 0 (add wiring), the card gates [{SHARE_LO}, {SHARE_HI}] (W_lead {WL})")
    if vsh is None or not 0.0 < vsh < 0.5:
        p.append(f"level share at step 0 = {vsh!r} (must be a share in (0, 0.5); the window is not gated here: the lead term now shares the total)")
    if p:
        fail(tag, "gate: " + "; ".join(p), gate_rc=rc); return 1
    tok = {"kind": "launch-gate-token", "verdict": "PASS", "prereg": PINS["card_file"], "prereg_sha256": CARD_SHA, "run": tag, "written_by": "llr_check.py gate (never by hand)", "at": now(),
           "binding": {"commit": TIP, "argv_sha256": tokens_sha(toks), "argv_sha256_method": "sha256 of json.dumps(token list, separators=(',',':'))", "w_level": W, "w_lead": WL, "aux_cond": WIRING,
                       "data_sha256": info.get("train_sha256"), "init_ckpt_sha256": INIT_SHA[s], "selfcheck_sha256": sha_file(os.path.join(gdir, "selfcheck.json")), "gate_argv_sha_trainer": sc.get("argv_sha"),
                       "shipped_sha256": SHIPPED, "lead_share_at_step0": lsh, "level_share_at_step0": vsh, "lead_labels_sha256": (info.get("lead_labels") or {}).get("sha256"), "n_params_trainable": (sc.get("levers") or {}).get("n_params_trainable")}}
    write_text(os.path.join(RC, f"gate_token_{tag}.json"), json.dumps(tok, indent=1))
    print(f"ZZLLR-GATE-PASS {tag} lead_share_at_step0 {lsh:.4f} level_share_at_step0 {vsh:.4f}")
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
        ev["lead_head"], ev["w_lead"], ev["aux_cond"] = sm.get("lead_head"), sm.get("w_lead"), sm.get("aux_cond")
        if sm.get("lead_head") != "on" or sm.get("w_lead") != float(WL) or sm.get("aux_cond") != WIRING:
            p.append(f"summary lead_head {sm.get('lead_head')!r} / w_lead {sm.get('w_lead')!r} / aux_cond {sm.get('aux_cond')!r}, registered on / {float(WL)!r} / {WIRING!r} (the run record of a lead-head-ON arm)")
        try:
            gtok = json.load(open(os.path.join(RC, f"gate_token_{tag}.json"), encoding="utf-8"))["binding"]
        except Exception as e:                              # noqa: BLE001
            gtok = {}; p.append(f"no readable gate token for {tag}: {type(e).__name__} (the run's label file cannot be compared with the gate's)")
        sl = sm.get("lead_labels") or {}
        ev["lead_labels_sha256"] = sl.get("sha256")
        if gtok and (sl.get("sha256") != gtok.get("lead_labels_sha256") or not sl.get("sha256")):
            p.append(f"summary lead_labels.sha256 {sl.get('sha256')!r} != the gate's {gtok.get('lead_labels_sha256')!r}: the run trained on another label file than the gate saw")
        if sm.get("level_head") != "on" or sm.get("w_level") != float(W):
            p.append(f"summary level_head {sm.get('level_head')!r} / w_level {sm.get('w_level')!r}, registered on / {float(W)!r} (the run record of a level-head-ON arm)")
        rn = sm.get("recover_n")
        if not (isinstance(rn, (int, float)) and not isinstance(rn, bool) and rn > 0):   # KR1, as RAPR-P01's post: the recovery really perturbed samples
            p.append(f"KR1: arm LLR run perturbed no sample (recover_n {rn!r}; RAPR-P01's recipe has p 0.1)")
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
        ok = (tok.get("verdict") == "PASS" and b["commit"] == TIP and b["argv_sha256"] == tokens_sha(toks) and b["w_level"] == W and b.get("w_lead") == WL and b.get("aux_cond") == WIRING and tok.get("prereg_sha256") == CARD_SHA
              and b["selfcheck_sha256"] == sha_file(os.path.join(gdir, "selfcheck.json")) and b["init_ckpt_sha256"] == INIT_SHA[s]
              and b["shipped_sha256"] == SHIPPED and all(sha_file(os.path.join(REPO, *rel.split("/"))) == h for rel, h in SHIPPED.items()))
    except Exception:                                       # noqa: BLE001
        ok = False
    print(f"gate-cached {tag}: {'YES' if ok else 'no'}")
    return 0 if ok else 1


def cmd_finish():
    st = {}
    for tag in shipped_runs("LLR"):
        st[tag] = "done" if os.path.exists(os.path.join(RC, f"RUN_{tag}.done")) else ("FAILED" if os.path.exists(os.path.join(RC, f"RUN_{tag}.FAILED")) else "none")
    write_marker_file(os.path.join(RC, "CHAIN_LLR.done"), json.dumps({"at": now(), "tip": TIP, "runs": st}, indent=1))
    print("ZZLLR-CHAIN-DONE " + json.dumps(st))
    return 0


def main(a):
    c = a[0]
    if c == "preflight": return cmd_preflight(a[1:])
    if c == "pin": return cmd_pin(a[1])
    if c == "gate": return cmd_gate(a[1], a[2])
    if c == "post": return cmd_post(a[1], a[2])
    if c == "fail": fail(a[1], " ".join(a[2:])); return 0
    if c == "stopped": write_marker_file(os.path.join(RC, "CHAIN_LLR.STOPPED"), json.dumps({"at": now(), "reason": " ".join(a[1:])})); print("ZZLLR-STOPPED " + " ".join(a[1:])); return 0
    if c == "wprobe-stopped": write_marker_file(os.path.join(RC, "WLPROBE.STOPPED"), json.dumps({"at": now(), "reason": " ".join(a[1:])})); print("ZZLLR-STOPPED " + " ".join(a[1:])); return 0
    if c == "finish": return cmd_finish()
    if c == "gate-cached": return cmd_gate_cached(a[1])
    print("unknown command", c); return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
