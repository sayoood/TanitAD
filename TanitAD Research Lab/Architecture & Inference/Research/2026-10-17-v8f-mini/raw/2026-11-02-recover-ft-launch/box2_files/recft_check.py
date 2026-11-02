"""RECOVER-FT chain checker (runs on box 2 with the venv python; stdlib only, no torch). Every marker and every pin is written HERE, one code path.
  preflight              chain start: disk floor, the shipped trainer / data library sha256, the six argv files, init ckpts, banks
  preflight run <tag>    before each run: disk floor (exit 9 = STOP the chain)
  pin <tag>              pin the training argv (a re-pin that differs is REFUSED), derive the gate argv (= training argv with --out and --selfcheck-only changed, asserted), make the dirs
  gate-cached <tag>      0 iff a PASS token for this tag is still bound to this commit / argv / selfcheck / shipped bytes (skips a ~40 min table re-build)
  gate <tag> <rc>        read the gate's selfcheck.json + log, assert the registered controls, write the per-run launch-gate token; FAIL => RUN_<tag>.FAILED, exit 1
  post <tag> <rc>        after training: summary / argv / ckpt / KP1 / KR1 checks; PASS => RUN_<tag>.done, else RUN_<tag>.FAILED; exit 0 always
  fail <tag> <reason>    write RUN_<tag>.FAILED
  stopped <reason>       write CHAIN_RECOVER.STOPPED
  finish                 write CHAIN_RECOVER.done (the chain reached its end; the per-run markers say which runs are done / FAILED)
Paths come from RECFT_HOME / RECFT_RUNSM (defaults are box 2's) so the same file is exercised on the dev box."""
import hashlib, json, os, shutil, sys, time

HOME = os.environ.get("RECFT_HOME", r"D:\tanitad\recft")
RUNSM = os.environ.get("RECFT_RUNSM", r"D:\tanitad\runs\v8f_mini_full")
REPO = os.environ.get("RECFT_REPO", r"D:\tanitad\repo")
DISK = os.environ.get("RECFT_DISK", "D:\\")
FLOOR = int(float(os.environ.get("RECFT_FLOOR_BYTES", "8e9")))
RC = os.path.join(RUNSM, "recft"); GATE = os.path.join(RUNSM, "gate_recft"); PIN = os.path.join(RC, "pin"); LOGS = os.path.join(RC, "logs")
TIP = "ad0bcb72718aa3c9e63ac4f589e65b94e8bf4ace"
RUNS = [("R_0", "R", 0), ("C0_0", "C0", 0), ("R_1", "R", 1), ("C0_1", "C0", 1), ("C0rep_0", "C0rep", 0), ("C0rep_1", "C0rep", 1)]
INFO = {t: (k, s) for t, k, s in RUNS}
SHIPPED = {"stack/scripts/v8f_mini_train.py": "1576a5783e6025630054ef30a6e8320d13df366fbeb89bbec98f2a155f71d9b0",
           "stack/tanitad/data/v8f_recover.py": "eff087dacfc30b04692952f8c828626a53384d3bc5caa74de7f3caf4435c6337"}
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


def loc(p):                                                 # identity on box 2; maps the declared run root onto RECFT_RUNSM when the dev-box test hook points it elsewhere
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
    print(f"ZZRECFT-{kind.upper()} {tag} {payload.get('reason', '')}")


def fail(tag, reason, **kw):
    marker(tag, "FAILED", {"reason": reason, **kw})


def cmd_preflight(args):
    bad = []
    if args and args[0] == "run":
        free = shutil.disk_usage(DISK).free
        if free < FLOOR:
            print(f"ZZRECFT-STOP disk floor: {free} bytes free on {DISK} < {FLOOR}"); return 9
        return 0
    free = shutil.disk_usage(DISK).free
    if free < FLOOR:
        bad.append(f"disk floor: {free} < {FLOOR}")
    for rel, want in SHIPPED.items():
        p = os.path.join(REPO, *rel.split("/"))
        got = sha_file(p) if os.path.exists(p) else None
        if got != want:
            bad.append(f"{rel}: sha256 {got} != shipped {want} (box 2's repo is not the landed tip {TIP[:8]})")
    for tag, kind, s in RUNS:
        try:
            toks = shipped_line(tag).split(" ")
            ic = toks[toks.index("--init-ckpt") + 1]
            if not os.path.exists(ic):
                bad.append(f"{tag}: init ckpt {ic} missing")
            if kind == "R":
                bank = toks[toks.index("--recover-dev-bank") + 1]
                bj = os.path.join(bank, "bank.json")
                if not os.path.exists(bj) or sha_file(bj) != BANK_SHA[s]:
                    bad.append(f"{tag}: {bj} missing or its sha256 != {BANK_SHA[s]}")
        except Exception as e:                              # noqa: BLE001
            bad.append(f"{tag}: {type(e).__name__}: {e}")
    if bad:
        print("ZZRECFT-PREFLIGHT-FAIL " + " | ".join(bad)); return 3
    print(f"preflight OK: disk free {free}, shipped trainer / library sha256 match, six argv files, init ckpts, banks")
    return 0


def cmd_pin(tag):
    line = shipped_line(tag); toks = line.split(" ")
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
    if (info.get("init_ckpt") or {}).get("sha256") != INIT_SHA[s]:
        p.append(f"init ckpt sha256 {(info.get('init_ckpt') or {}).get('sha256')} != DAG-OACC s{s}'s {INIT_SHA[s][:12]}")
    if info.get("train_sha256") != KP1["train_sha256"]:
        p.append(f"gate train_sha256 {info.get('train_sha256')} != DAG-OACC's (KP1)")
    if sc.get("steps_total") != STEPS:
        p.append(f"steps_total {sc.get('steps_total')} != {STEPS}")
    rp = sc.get("recover_probe") or {}
    if rp.get("pass") is not True:
        p.append(f"recover_probe pass = {rp.get('pass')}")
    lv = (sc.get("levers") or {}).get("recover") or {}
    if kind == "R":
        rec = info.get("recover") or {}
        if not rec:
            p.append("R: selfcheck info.recover (the table build record) is absent")
        else:
            if rec.get("bank_sha256") != BANK_SHA[s]:
                p.append(f"R: built bank sha256 {rec.get('bank_sha256')} != {BANK_SHA[s][:12]}")
            if (rec.get("kb3") or {}).get("window_list_sha256") != KP1["train_window_list_sha256"]:
                p.append("R: K-B3 window-list sha256 != DAG-OACC's")
        if lv.get("recover_p") != 0.3 or lv.get("recover_target") != "rejoin" or lv.get("recover_rejoin_steps") != 10 or lv.get("recover_seed") != s:
            p.append(f"R: built recover levers {lv}")
    else:
        if info.get("recover"):
            p.append("C0 / C0rep: a table was built at p = 0")
        if lv.get("recover_p") != 0.0:
            p.append(f"C0 / C0rep: built recover_p {lv.get('recover_p')} != 0.0")
    if kind == "C0rep" and f"--seed {10 + s} " not in " ".join(toks) + " ":
        p.append("C0rep: --seed is not 10 + s")
    if p:
        fail(tag, "gate: " + "; ".join(p), gate_rc=rc); return 1
    tok = {"kind": "launch-gate-token", "verdict": "PASS", "prereg": "PREREG_RECOVER_FT (+AMENDMENT_1)", "run": tag, "written_by": "recft_check.py gate (never by hand)", "at": now(),
           "binding": {"commit": TIP, "argv_sha256": tokens_sha(toks), "argv_sha256_method": "sha256 of json.dumps(token list, separators=(',',':'))",
                       "data_sha256": info.get("train_sha256"), "init_ckpt_sha256": INIT_SHA[s], "selfcheck_sha256": sha_file(os.path.join(gdir, "selfcheck.json")),
                       "gate_argv_sha_trainer": sc.get("argv_sha"), "shipped_trainer_sha256": SHIPPED["stack/scripts/v8f_mini_train.py"], "shipped_recover_lib_sha256": SHIPPED["stack/tanitad/data/v8f_recover.py"]}}
    write_text(os.path.join(RC, f"gate_token_{tag}.json"), json.dumps(tok, indent=1))
    print(f"ZZRECFT-GATE-PASS {tag}")
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
        if kind == "R" and not (isinstance(rn, (int, float)) and rn > 0):
            p.append(f"KR1: R run perturbed no sample (recover_n {rn})")
        if kind != "R" and rn not in (0, None):
            p.append(f"KR1: p = 0 run reads recover_n {rn} != 0")
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
    """0 only if a launch-gate token written by `gate` for THIS tag is still bound to exactly this commit, this pinned argv, this selfcheck.json and these shipped trainer / library bytes. The R gates cost one full table build
    (every TRAIN clip read once from the USB HDD, ~40 min), so the pre-chain gate's PASS is reused instead of re-built; any difference at all means 'not cached' and the chain gates again."""
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
    for tag, _, _ in RUNS:
        st[tag] = "done" if os.path.exists(os.path.join(RC, f"RUN_{tag}.done")) else ("FAILED" if os.path.exists(os.path.join(RC, f"RUN_{tag}.FAILED")) else "none")
    write_text(os.path.join(RC, "CHAIN_RECOVER.done"), json.dumps({"at": now(), "tip": TIP, "runs": st}, indent=1))
    print("ZZRECFT-CHAIN-DONE " + json.dumps(st))
    return 0


def main(a):
    c = a[0]
    if c == "preflight": return cmd_preflight(a[1:])
    if c == "pin": return cmd_pin(a[1])
    if c == "gate": return cmd_gate(a[1], a[2])
    if c == "post": return cmd_post(a[1], a[2])
    if c == "fail": fail(a[1], " ".join(a[2:])); return 0
    if c == "stopped": write_text(os.path.join(RC, "CHAIN_RECOVER.STOPPED"), json.dumps({"at": now(), "reason": " ".join(a[1:])})); print("ZZRECFT-STOPPED " + " ".join(a[1:])); return 0
    if c == "finish": return cmd_finish()
    if c == "gate-cached": return cmd_gate_cached(a[1])
    print("unknown command", c); return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
