"""RECOVER-NEXT chain checker (runs on box 2 with the venv python; stdlib only, no torch, no network, no subprocess). Every marker and every pin is written HERE, one code path.
Adapted from the RECOVER-FT chain's checker: same shipped trainer / data-library bytes, same init, same data, same gate protocol; the six runs are RAPR_0 RAPR_1 RTL_0 RTL_1 RP01_0 RP01_1 (card PREREG_RECOVER_NEXT_FROZEN.md).
  preflight              chain start: disk floor, the shipped trainer / data library sha256, RECOVER_DEFAULTS, the six argv files (argv audit), init ckpts, banks
  preflight run <tag>    before each run: disk floor (exit 9 = STOP the chain)
  pin <tag>              argv audit (a changed lever is REFUSED, rc 2), pin the training argv (a re-pin that differs is REFUSED), derive the gate argv (= training argv with --out and --selfcheck-only changed, asserted), make the dirs
  gate-cached <tag>      0 iff a PASS token for this tag is still bound to this commit / argv / selfcheck / shipped bytes (skips a ~40 min table re-build)
  gate <tag> <rc>        read the gate's selfcheck.json + log, assert the registered controls and the BUILT levers of the run's kind, write the per-run launch-gate token; FAIL => RUN_<tag>.FAILED, exit 1
  post <tag> <rc>        after training: summary / argv / ckpt / KP1 / KR1 checks; PASS => RUN_<tag>.done, else RUN_<tag>.FAILED; exit 0 always
  fail <tag> <reason>    write RUN_<tag>.FAILED
  stopped <reason>       write CHAIN_RNEXT.STOPPED
  finish                 write CHAIN_RNEXT.done (the chain reached its end; the per-run markers say which runs are done / FAILED)
Paths come from RNEXT_HOME / RNEXT_RUNSM / RNEXT_REPO / RNEXT_DISK / RNEXT_FLOOR_BYTES (defaults are box 2's) so the same file is exercised on the dev box."""
import ast, hashlib, json, os, shutil, sys, time

HOME = os.environ.get("RNEXT_HOME", r"D:\tanitad\rnext")
RUNSM = os.environ.get("RNEXT_RUNSM", r"D:\tanitad\runs\v8f_mini_full")
REPO = os.environ.get("RNEXT_REPO", r"D:\tanitad\repo")
DISK = os.environ.get("RNEXT_DISK", "D:\\")
FLOOR = int(float(os.environ.get("RNEXT_FLOOR_BYTES", "8e9")))
RC = os.path.join(RUNSM, "rnext"); GATE = os.path.join(RUNSM, "gate_rnext"); PIN = os.path.join(RC, "pin"); LOGS = os.path.join(RC, "logs")
TIP = "ad0bcb72718aa3c9e63ac4f589e65b94e8bf4ace"                       # the shipped tree on box 2 (no new tar): the trainer and the data library below are bound by sha256
PREREG_SHA = "01046fe694dea2eacd0df7cb31bdbc4b4580cfc741af40630e735d7d4cc46b1c"      # PREREG_RECOVER_NEXT_FROZEN.md (recover_next/FROZEN_SHA.txt)
RUNS = [("RAPR_0", "RAPR", 0), ("RAPR_1", "RAPR", 1), ("RTL_0", "RTL", 0), ("RTL_1", "RTL", 1), ("RP01_0", "RP01", 0), ("RP01_1", "RP01", 1)]
#: RAPR-P01 (PREREG_RECOVER_NEXT_ADDENDUM2_RAPR.md row E): RAPR's frozen argv with --recover-p 0.1. A tag outside RUNS is in scope only when its argv file is shipped to HOME/argv (shipped_runs), so the six-run chains and their tests are unchanged.
#: RPT (PREREG_RECOVER_NEXT_ADDENDUM4_RPT.md): R's frozen argv (empirical TRAIN-bank law, p 0.3, rejoin target, the seed's own bank) with --recover-dev-shape pertick. Same scoping rule as RAPR-P01.
EXTRA = [("RAPRP01_0", "RAPRP01", 0), ("RAPRP01_1", "RAPRP01", 1), ("RPT_0", "RPT", 0), ("RPT_1", "RPT", 1)]
INFO = {t: (k, s) for t, k, s in RUNS + EXTRA}
#: the three files the RPT launcher ships over the box-2 repo (launch_rpt.sh). `python mk_rpt_pins.py` recomputes these sha256 from their source files (pertick/stack_patch x 2, rpt_stack/ x 1) and rewrites this block.
SHIPPED = {"stack/scripts/v8f_mini_train.py": "bda468fe01b25e901df9115218fcc125ce4234225aa3ea43613e03df3742660e",
           "stack/tanitad/data/v8f_recover.py": "0257ccb93251827fe5f1b3d7195999b5836145e8615cfd0d0a599262b8834b4c",
           "stack/tanitad/data/v8f_mini_data.py": "209da7a60765ba81268ee794aba187e21b0a6c73a49b9e98c86e2d876581ece6"}
#: the trainer's own defaults, stack/tanitad/data/v8f_recover.py:28 (`RECOVER_DEFAULTS = {...}`; the file is bound by SHIPPED above and the line is re-read from the shipped bytes by `preflight`)
DEFAULTS = {"p": 0.0, "dev_law": "empirical", "target": "rejoin", "rejoin_steps": 10, "seed": 0, "dev_shape": "constant"}
DEFAULTS_LINE = 28
#: the registered lever of each kind (card section 2): what the BUILT perturber must report in the gate's selfcheck.json. RTL drops --recover-rejoin-steps (it does nothing under the logged target), so its BUILT value is the trainer's default.
KINDS = {"RAPR": {"p": 0.3, "law": "apriori",   "target": "rejoin", "rejoin": 10,                       "table": False},
         "RTL":  {"p": 0.3, "law": "empirical", "target": "logged", "rejoin": DEFAULTS["rejoin_steps"], "table": True},
         "RP01": {"p": 0.1, "law": "empirical", "target": "rejoin", "rejoin": 10,                       "table": True},
         "RAPRP01": {"p": 0.1, "law": "apriori", "target": "rejoin", "rejoin": 10,                     "table": False},
         "RPT": {"p": 0.3, "law": "empirical", "target": "rejoin", "rejoin": 10,                        "table": True, "shape": "pertick"}}     # RPT: the one kind with a "shape" key; the other kinds' gate reads no shape
INIT_SHA = {0: "67265ec1666fae3a6d51e924ee06dfb68b2cd0630afc62e42f3a46182b8bc45f", 1: "90b01474819082a6d32a9000d63a4954e0074916140abdbbcb67b353f7ac3333"}   # DAG-OACC ckpt_last.pt (MEASURED, E:/Projects/TanitAD-artifacts/v8f-dagoacc/*/ckpt_last.pt)
KP1 = {"train_sha256": "37f4f9ee468e0872134074a82d63c2915458596fdbcf3d0fcf5b09f1fe0d8cc1",
       "train_window_list_sha256": "e144adcd9b6273f2b44ad7dfa162931b90e159f7b3911e0893b25777b2ae8168"}                                       # DAG-OACC's summary.json, both seeds (MEASURED)
BANK_SHA = {0: "90c43c66f52f250d41ff23e837d5fb31d1592cf2aad14afacd7ded011734b589", 1: "a06c8289b20854a6933264e95dca201ea7d9e2723fd0b2248872f56361ea689a"}
STEPS = 3000


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


def shipped_line(tag):
    with open(os.path.join(HOME, "argv", tag + ".train.txt"), encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    assert len(lines) == 1, f"{tag}: the shipped argv file must hold exactly one line, has {len(lines)}"
    return lines[0]


DECL = r"D:\tanitad\runs\v8f_mini_full"


def loc(p):                                                 # identity on box 2; maps the declared run root onto RNEXT_RUNSM when the dev-box test hook points it elsewhere
    return RUNSM + p[len(DECL):] if (RUNSM != DECL and p.startswith(DECL)) else p


def out_of(toks):
    return toks[toks.index("--out") + 1]


def write_text(p, s):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="") as f:
        f.write(s)


def marker(tag, kind, payload):
    os.makedirs(RC, exist_ok=True)
    p = os.path.join(RC, f"RUN_{tag}.{kind}")
    if os.path.exists(p):                                   # never overwrite, never delete: move aside
        os.replace(p, p + ".prev_" + time.strftime("%Y%m%d_%H%M%S"))
    write_text(p, json.dumps({"tag": tag, "at": now(), "tip": TIP, **payload}, indent=1, default=str))
    print(f"ZZRNEXT-{kind.upper()} {tag} {payload.get('reason', '')}")


def fail(tag, reason, **kw):
    marker(tag, "FAILED", {"reason": reason, **kw})


# ---- the frozen argv: the card's levers, written once, compared flag by flag (a named reason per difference) --------------------------------------------------------------------------------------------------------------
def expected_tokens(tag):
    """The frozen training argv of `tag` (recover_next/argv/<tag>.train.txt): R_s's shipped argv (same init, data, steps, every other flag) with the card's one change per kind."""
    kind, s = INFO[tag]
    d, r = r"D:\tanitad", DECL
    base = [("--arm", "M2b"), ("--seed", str(s)), ("--grid", "pool4x10"), ("--out", r + rf"\M2b_s{s}_recft_{kind}"), ("--cache-dir", d + r"\cache_full"),
            ("--v9-train", d + r"\data\v9\v9_labels_train.npz"), ("--v9-train-md5", "83e30bd3791f0767707eade5f4671759"), ("--lead-train", d + r"\data\v9\lead_train4572.npz"),
            ("--speed-train", d + r"\data\v9\refcv6_speed_max_v8_train.jsonl"), ("--join-train", r + r"\join_train.npz"), ("--p10", r + r"\p10\p10_prior.pt"), ("--kin-mode", "yaw"),
            ("--tracks", d + r"\cache_full\native_tracks.pt"), ("--exclude", "540:600"), ("--batch", "64"), ("--steps-mult", "4"), ("--device", "cuda"),
            ("--init-ckpt", r + rf"\M2b_s{s}_dagoacc\ckpt_last.pt"), ("--steps", str(STEPS)), ("--lr", "1e-4"), ("--warmup", "100"), ("--p-gt-start", "0.2"), ("--p-gt-end", "0.2"),
            ("--ego-perturb-p", "0"), ("--w-acc", "49.6969")]
    bank = [("--recover-dev-bank", r + rf"\dagger_bank_dag2\s{s}"), ("--recover-dev-bank-sha256", BANK_SHA[s])]
    if kind == "RPT":       # R's argv (bank, rejoin target, p 0.3) plus the one new lever: the per-tick deviation shape (appended LAST, as the argv file is built)
        tail = [("--recover-p", "0.3")] + bank + [("--recover-target", "rejoin"), ("--recover-rejoin-steps", "10"), ("--recover-seed", str(s)), ("--recover-dev-shape", "pertick")]
    elif kind in ("RAPR", "RAPRP01"):      # the a-priori law: the bank flags are DROPPED (the trainer refuses a bank that would do nothing); RAPRP01 = RAPR with the dose --recover-p 0.1
        tail = [("--recover-p", "0.3" if kind == "RAPR" else "0.1"), ("--recover-target", "rejoin"), ("--recover-rejoin-steps", "10"), ("--recover-seed", str(s)), ("--recover-dev-law", "apriori")]
    elif kind == "RTL":     # the logged target: no rejoin blend, --recover-rejoin-steps dropped (it does nothing under the logged target)
        tail = [("--recover-p", "0.3")] + bank + [("--recover-target", "logged"), ("--recover-seed", str(s))]
    else:                   # RP01: the dose, --recover-p 0.1
        tail = [("--recover-p", "0.1")] + bank + [("--recover-target", "rejoin"), ("--recover-rejoin-steps", "10"), ("--recover-seed", str(s))]
    return [t for kv in base + tail for t in kv]


def parse_flags(toks):
    d, bad, i = {}, [], 0
    while i < len(toks):
        t = toks[i]
        if not t.startswith("--"):
            bad.append(f"stray token {t!r} where a --flag was expected"); i += 1; continue
        if i + 1 >= len(toks) or toks[i + 1].startswith("--"):
            bad.append(f"flag {t} has no value"); d.setdefault(t, []).append(None); i += 1; continue
        d.setdefault(t, []).append(toks[i + 1]); i += 2
    return d, bad


def audit_argv(tag, toks):
    """[] iff `toks` is the frozen argv of `tag`. Every difference is named: the flag, what the argv has, what the card's argv has."""
    kind = INFO[tag][0]
    want_toks = expected_tokens(tag)
    got, bad = parse_flags(list(toks)); want, _ = parse_flags(want_toks)
    for f, vs in got.items():
        if len(vs) > 1:
            bad.append(f"{f} is given {len(vs)} times {vs} (argparse keeps the last: the others are dead values)")
    for f in want:
        if f not in got:
            bad.append(f"missing {f} (the frozen {tag} argv has {want[f][0]!r})")
    for f in got:
        if f not in want:
            note = ""
            if kind in ("RAPR", "RAPRP01") and f in ("--recover-dev-bank", "--recover-dev-bank-sha256"):
                note = " (a bank flag with the a-priori law: the trainer refuses it, a lever that is declared and does nothing)"
            elif kind == "RTL" and f == "--recover-rejoin-steps":
                note = " (it does nothing under --recover-target logged: the arm drops it)"
            elif f == "--recover-dev-shape":
                note = " (the per-tick shape belongs to the RPT arm only: any other arm runs the constant shape and does not name the flag)"
            bad.append(f"unexpected flag {f} {got[f]}{note}: not in the frozen {tag} argv")
        elif got[f] != want[f] and len(got[f]) == 1:
            bad.append(f"{f} = {got[f][0]!r} but the frozen {tag} argv has {want[f][0]!r}")
    if not bad and list(toks) != want_toks:
        bad.append("the flags and values are the frozen ones but the token order differs (the gate token binds the token list)")
    return bad


def shipped_runs():
    """The six registered runs plus the EXTRA runs whose argv file is shipped to HOME/argv: a run is in scope for preflight and finish when its argv is shipped."""
    return RUNS + [r for r in EXTRA if os.path.exists(os.path.join(HOME, "argv", r[0] + ".train.txt"))]


def shipped_defaults():
    """(RECOVER_DEFAULTS, line number) read from the SHIPPED data library by ast (no import), or (None, None)."""
    try:
        tree = ast.parse(open(os.path.join(REPO, "stack", "tanitad", "data", "v8f_recover.py"), encoding="utf-8").read())
        for n in tree.body:
            if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "RECOVER_DEFAULTS" for t in n.targets):
                return ast.literal_eval(n.value), n.lineno
    except Exception:                                       # noqa: BLE001
        pass
    return None, None


def cmd_preflight(args):
    bad = []
    if args and args[0] == "run":
        free = shutil.disk_usage(DISK).free
        if free < FLOOR:
            print(f"ZZRNEXT-STOP disk floor: {free} bytes free on {DISK} < {FLOOR}"); return 9
        return 0
    free = shutil.disk_usage(DISK).free
    if free < FLOOR:
        bad.append(f"disk floor: {free} < {FLOOR}")
    for rel, want in SHIPPED.items():
        p = os.path.join(REPO, *rel.split("/"))
        got = sha_file(p) if os.path.exists(p) else None
        if got != want:
            bad.append(f"{rel}: sha256 {got} != shipped {want} (box 2's repo is not the landed tip {TIP[:8]})")
    sd, ln = shipped_defaults()
    if sd != DEFAULTS or ln != DEFAULTS_LINE:
        bad.append(f"RECOVER_DEFAULTS in the shipped v8f_recover.py (line {ln}) reads {sd}, the checker expects {DEFAULTS} at line {DEFAULTS_LINE}")
    runs = shipped_runs()
    for tag, kind, s in runs:
        try:
            line = shipped_line(tag); toks = line.split(" ")
            bad += [f"{tag}: argv audit: {m}" for m in audit_argv(tag, toks)]
            ic = loc(toks[toks.index("--init-ckpt") + 1])
            if not os.path.exists(ic):
                bad.append(f"{tag}: init ckpt {ic} missing")
            if KINDS[kind]["table"]:
                bj = os.path.join(loc(toks[toks.index("--recover-dev-bank") + 1]), "bank.json")
                if not os.path.exists(bj) or sha_file(bj) != BANK_SHA[s]:
                    bad.append(f"{tag}: {bj} missing or its sha256 != {BANK_SHA[s]}")
        except Exception as e:                              # noqa: BLE001
            bad.append(f"{tag}: {type(e).__name__}: {e}")
    if bad:
        print("ZZRNEXT-PREFLIGHT-FAIL " + " | ".join(bad)); return 3
    n_argv = "six" if len(runs) == 6 else str(len(runs))
    print(f"preflight OK: disk free {free}, shipped trainer / library sha256 match, RECOVER_DEFAULTS line {DEFAULTS_LINE}, {n_argv} argv files (audited), init ckpts, banks")
    return 0


def cmd_pin(tag):
    try:
        line = shipped_line(tag); toks = line.split(" ")
        bad = audit_argv(tag, toks)
    except (AssertionError, OSError, ValueError) as e:
        print(f"REFUSED: {tag}: the shipped argv file is unreadable or malformed: {type(e).__name__}: {e}"); return 2
    if bad:
        print(f"REFUSED: {tag}: argv audit against the frozen {INFO[tag][0]} argv: " + " | ".join(bad)); return 2
    gout = os.path.join(GATE, os.path.basename(out_of(toks)))
    gt = list(toks); gt[gt.index("--out") + 1] = gout; gt.append("--selfcheck-only")
    back = [t for t in gt if t != "--selfcheck-only"]; back[back.index("--out") + 1] = out_of(toks)
    assert back == toks, "the gate argv is not the training argv with --out and --selfcheck-only changed"
    for name, text in (("train", line), ("gate", " ".join(gt))):
        p = os.path.join(PIN, f"{tag}.{name}.txt")
        if os.path.exists(p) and open(p, encoding="utf-8").read().strip() != text:
            print(f"REFUSED: {p} pins another {name} argv than the shipped one (delete it only if the levers were deliberately changed)"); return 2
        write_text(p, text + "\r\n")
    for d in (loc(out_of(toks)), gout, LOGS):
        os.makedirs(d, exist_ok=True)
    print(f"pinned {tag}: train sha256(tokens) {tokens_sha(toks)}")
    return 0


def cmd_gate(tag, rc):
    kind, s = INFO[tag]; K = KINDS[kind]
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
    if (info.get("init_ckpt") or {}).get("sha256") != INIT_SHA[s]:
        p.append(f"init ckpt sha256 {(info.get('init_ckpt') or {}).get('sha256')} != DAG-OACC s{s}'s {INIT_SHA[s][:12]}")
    if info.get("train_sha256") != KP1["train_sha256"]:
        p.append(f"gate train_sha256 {info.get('train_sha256')} != DAG-OACC's (KP1)")
    if sc.get("steps_total") != STEPS:
        p.append(f"steps_total {sc.get('steps_total')} != {STEPS}")
    rp = sc.get("recover_probe") or {}
    if rp.get("pass") is not True:
        p.append(f"recover_probe pass = {rp.get('pass')}")
    lvs = sc.get("levers") or {}
    lv = lvs.get("recover") or {}
    want = {"recover_p": K["p"], "recover_dev_law": K["law"], "recover_target": K["target"], "recover_rejoin_steps": K["rejoin"], "recover_seed": s}
    for k, w in want.items():                               # the BUILT perturber's levers (selfcheck levers.recover = RecoverPerturb.built_levers()), not the declared ones
        if lv.get(k) != w:
            p.append(f"{kind}: built lever {k} = {lv.get(k)!r}, registered {w!r}")
    if K.get("shape"):                                      # RPT: the BUILT shape, the table's per-tick columns and the probe's profile read-back (the trainer asserts each; this is the independent read)
        if lv.get("recover_dev_shape") != K["shape"]:
            p.append(f"{kind}: built lever recover_dev_shape = {lv.get('recover_dev_shape')!r}, registered {K['shape']!r}")
        if (info.get("recover") or {}).get("dev_shape") != K["shape"]:
            p.append(f"{kind}: selfcheck info.recover.dev_shape = {(info.get('recover') or {}).get('dev_shape')!r}, registered {K['shape']!r}")
        pt = ((lvs.get("recover_table") or {}).get("per_tick")) or {}
        if not (isinstance(pt.get("ticks"), int) and pt.get("ticks") > 0 and all(isinstance(pt.get(k), list) and len(pt.get(k)) == pt.get("ticks") for k in ("mean_abs_dv_t", "mean_abs_da_t", "mean_abs_dom_t"))):
            p.append(f"{kind}: levers.recover_table.per_tick is absent or has no per-tick columns ({str(pt)[:80]})")
        ct = ((lvs.get("recover_table") or {}).get("counts")) or {}
        if not (isinstance(ct.get("dropped_per_tick_row_unreal"), int) and isinstance(ct.get("kept"), int) and ct.get("kept") > 0):
            p.append(f"{kind}: levers.recover_table.counts carries no per-tick drop count / kept count ({str(ct)[:80]})")
        pe = rp.get("pertick_v_max_abs_err")
        if not (isinstance(pe, (int, float)) and not isinstance(pe, bool) and pe <= 1e-4):
            p.append(f"{kind}: recover_probe.pertick_v_max_abs_err = {pe!r}: the drawn profile was not read back into the batch (want a number <= 1e-4)")
    rec = info.get("recover") or {}
    if K["table"]:                                          # RTL / RP01 / RPT: the deviation table is built from bank s (K-B3 read, the loaded bank's sha256 is the registered one)
        if not rec:
            p.append(f"{kind}: selfcheck info.recover (the table build record) is absent")
        else:
            if rec.get("bank_sha256") != BANK_SHA[s]:
                p.append(f"{kind}: built bank sha256 {rec.get('bank_sha256')} != {BANK_SHA[s][:12]}")
            if (rec.get("kb3") or {}).get("window_list_sha256") != KP1["train_window_list_sha256"]:
                p.append(f"{kind}: K-B3 window-list sha256 != DAG-OACC's")
        if lv.get("recover_dev_bank_sha256") != BANK_SHA[s]:
            p.append(f"{kind}: built lever recover_dev_bank_sha256 {lv.get('recover_dev_bank_sha256')} != {BANK_SHA[s][:12]}")
        tbl = lvs.get("recover_table") or {}
        if not (isinstance(tbl.get("n"), int) and tbl.get("n") > 0 and tbl.get("source_sha256") == BANK_SHA[s]):
            p.append(f"{kind}: levers.recover_table is empty or not from bank s{s} ({str(tbl)[:80]})")
    else:                                                   # RAPR: NO deviation table is built and no bank is read
        if rec:
            p.append(f"{kind}: a deviation table was built (selfcheck info.recover is present) under the a-priori law")
        if lvs.get("recover_table") is not None:
            p.append(f"{kind}: levers.recover_table is not None ({str(lvs.get('recover_table'))[:80]})")
        if lv.get("recover_dev_bank") is not None or lv.get("recover_dev_bank_sha256") is not None:
            p.append(f"{kind}: a bank is built under the a-priori law (recover_dev_bank {lv.get('recover_dev_bank')!r}, sha256 {lv.get('recover_dev_bank_sha256')!r})")
    if p:
        fail(tag, "gate: " + "; ".join(p), gate_rc=rc); return 1
    tok = {"kind": "launch-gate-token", "verdict": "PASS", "prereg": "PREREG_RECOVER_NEXT_FROZEN", "prereg_sha256": PREREG_SHA, "run": tag, "written_by": "rnext_check.py gate (never by hand)", "at": now(),
           "binding": {"commit": TIP, "argv_sha256": tokens_sha(toks), "argv_sha256_method": "sha256 of json.dumps(token list, separators=(',',':'))",
                       "data_sha256": info.get("train_sha256"), "init_ckpt_sha256": INIT_SHA[s], "selfcheck_sha256": sha_file(os.path.join(gdir, "selfcheck.json")),
                       "gate_argv_sha_trainer": sc.get("argv_sha"), "shipped_trainer_sha256": SHIPPED["stack/scripts/v8f_mini_train.py"], "shipped_recover_lib_sha256": SHIPPED["stack/tanitad/data/v8f_recover.py"]}}
    write_text(os.path.join(RC, f"gate_token_{tag}.json"), json.dumps(tok, indent=1))
    print(f"ZZRNEXT-GATE-PASS {tag}")
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
        rn = sm.get("recover_n")
        ev["recover_n"], ev["recover_share"] = rn, sm.get("recover_share")
        if not (isinstance(rn, (int, float)) and rn > 0):   # all three kinds run the perturbation (p > 0)
            p.append(f"KR1: {kind} run perturbed no sample (recover_n {rn})")
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
    """0 only if a launch-gate token written by `gate` for THIS tag is still bound to exactly this commit, this pinned argv, this selfcheck.json and these shipped trainer / library bytes. The RTL / RP01 gates cost one full table build
    (every TRAIN clip read once from the USB HDD, ~40 min), so a PASS from a gate-only run (or from an earlier start of this chain) is reused instead of re-built; any difference at all means 'not cached' and the chain gates again."""
    kind, s = INFO[tag]
    try:
        toks = shipped_line(tag).split(" ")
        tok = json.load(open(os.path.join(RC, f"gate_token_{tag}.json"), encoding="utf-8")); b = tok["binding"]
        gdir = os.path.join(GATE, os.path.basename(out_of(toks)))
        ok = (tok.get("verdict") == "PASS" and b["commit"] == TIP and b["argv_sha256"] == tokens_sha(toks)
              and b["selfcheck_sha256"] == sha_file(os.path.join(gdir, "selfcheck.json"))
              and b["shipped_trainer_sha256"] == sha_file(os.path.join(REPO, "stack", "scripts", "v8f_mini_train.py"))
              and b["shipped_recover_lib_sha256"] == sha_file(os.path.join(REPO, "stack", "tanitad", "data", "v8f_recover.py"))
              and b["init_ckpt_sha256"] == INIT_SHA[s])
    except Exception:                                       # noqa: BLE001
        ok = False
    print(f"gate-cached {tag}: {'YES' if ok else 'no'}")
    return 0 if ok else 1


def cmd_finish():
    st = {}
    for tag, _, _ in shipped_runs():
        st[tag] = "done" if os.path.exists(os.path.join(RC, f"RUN_{tag}.done")) else ("FAILED" if os.path.exists(os.path.join(RC, f"RUN_{tag}.FAILED")) else "none")
    write_text(os.path.join(RC, "CHAIN_RNEXT.done"), json.dumps({"at": now(), "tip": TIP, "runs": st}, indent=1))
    print("ZZRNEXT-CHAIN-DONE " + json.dumps(st))
    return 0


def main(a):
    c = a[0]
    if c == "preflight": return cmd_preflight(a[1:])
    if c == "pin": return cmd_pin(a[1])
    if c == "gate": return cmd_gate(a[1], a[2])
    if c == "post": return cmd_post(a[1], a[2])
    if c == "fail": fail(a[1], " ".join(a[2:])); return 0
    if c == "stopped": write_text(os.path.join(RC, "CHAIN_RNEXT.STOPPED"), json.dumps({"at": now(), "reason": " ".join(a[1:])})); print("ZZRNEXT-STOPPED " + " ".join(a[1:])); return 0
    if c == "finish": return cmd_finish()
    if c == "gate-cached": return cmd_gate_cached(a[1])
    print("unknown command", c); return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
