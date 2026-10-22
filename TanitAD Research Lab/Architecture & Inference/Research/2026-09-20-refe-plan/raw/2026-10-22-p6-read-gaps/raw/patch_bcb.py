import re
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
s = open("C:/Users/Admin/par18/p6gaps/raw/build_candidate_bank.TIP.py", encoding="utf-8", newline="").read()
assert "\r\n" not in s


def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:70])
    s = s.replace(old, new)


m = re.search(r'    "srttc_s0_dump.py": "(1b41a9fbc21426f9[0-9a-f]+)",\n', s)
assert m
landed = m.group(1)
s = s.replace(m.group(0), '    "srttc_s0_dump.py": (S0_DUMP_LANDED_SHA256, S0_DUMP_G4_SHA256),        # goal-mode (P2): either; a command-mode candidate needs the G4 one (apply_command_source)\n')
rep('''# sha256 of the LANDED S0 code this wrapper runs (identical on the dev box and at the tip 807a29c, compared 2026-10-19)
S0_PINS = {''', '''# sha256 of the LANDED S0 code this wrapper runs (identical on the dev box and at the tip 807a29c, compared 2026-10-19)
# P6-READ-GAPS G4 (2026-10-22): srttc_s0_dump.py has TWO accepted forms. The landed one refuses every command-mode checkpoint; the G4 one (raw/2026-10-18-sel-rank/code) adds `--commands`
# (the P6 training command table) and is otherwise the landed script (a goal-mode run is unchanged). Both are CRLF files; the pins are raw sha256.
S0_DUMP_LANDED_SHA256 = "%s"
S0_DUMP_G4_SHA256 = "28e5e56bf2f2b15825a74b81549ea2f078bd7e86bd95c24047d0f1bbfea4441f"
S0_PINS = {''' % landed)
rep('''    bad = []
    for rel, want in S0_PINS.items():
        p = os.path.join(s0_code, rel)
        if not os.path.exists(p):
            bad.append(f"{rel}: missing under {s0_code}")
        elif sha256_file(p) != want:
            bad.append(f"{rel}: sha256 {sha256_file(p)[:16]}... != pinned {want[:16]}...")
    if bad:
        raise BuildError("the landed S0 code differs from the pinned copy: " + "; ".join(bad))
    return {rel: want for rel, want in S0_PINS.items()}''', '''    bad, got = [], {}
    for rel, want in S0_PINS.items():
        ok = (want,) if isinstance(want, str) else tuple(want)
        p = os.path.join(s0_code, rel)
        if not os.path.exists(p):
            bad.append(f"{rel}: missing under {s0_code}")
        elif sha256_file(p) not in ok:
            bad.append(f"{rel}: sha256 {sha256_file(p)[:16]}... != pinned {', '.join(w[:16] + '...' for w in ok)}")
        else:
            got[rel] = sha256_file(p)
    if bad:
        raise BuildError("the landed S0 code differs from the pinned copy: " + "; ".join(bad))
    return got''')
rep('''    cand = {"created": now(), "candidate_id": a.candidate_id,''', '''    cmd_rec = None
    if getattr(a, "commands", None):
        cmd_rec = check_command_source(a.commands, getattr(a, "commands_sha256", None), mp["tokens"])
    cand = {"created": now(), "candidate_id": a.candidate_id, "commands": cmd_rec,''')
rep('''def read_candidate(root):''', '''def ckpt_nav_mode(path) -> str:
    """the checkpoint nav mode as the planner loader reads it (meta.nav_mode, default 'goal'; refe/ckpt_io.py config_for_checkpoint)"""
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False, mmap=True)          # the loader's own call (ckpt_io.py:122): our checkpoint, sha256-checked by `prepare`
    m = ck.get("meta") if isinstance(ck, dict) and isinstance(ck.get("meta"), dict) else {}
    return str(m.get("nav_mode", "goal"))


def check_command_source(path, sha256, plan_tokens) -> dict:
    """G4: the P6 command table (HF logs/commands.json: {"<log>/<token>": one-hot [LEFT, STRAIGHT, RIGHT, UNKNOWN]}) must cover EVERY token of the S0 plan; refuses otherwise"""
    if not os.path.exists(path):
        raise BuildError(f"command source {path} does not exist")
    sha = sha256_file(path)
    if sha256 and sha != sha256.lower():
        raise BuildError(f"command source sha256 {sha} != the given {sha256}")
    tab = json.load(open(path, encoding="utf-8"))
    missing = [f"{lg}/{tk}" for tk, lg in plan_tokens.items() if f"{lg}/{tk}" not in tab]
    if missing:
        raise BuildError(f"the command source lacks {len(missing)} of the {len(plan_tokens)} plan tokens (e.g. {missing[:3]}): a command-mode checkpoint is never run on a default command")
    bad = [k for k in (f"{lg}/{tk}" for tk, lg in plan_tokens.items()) if sorted(int(x) for x in tab[k]) != [0, 0, 0, 1]]
    if bad:
        raise BuildError(f"the command source has non-one-hot rows for plan tokens (e.g. {bad[:3]})")
    return {"path": os.path.abspath(path).replace(chr(92), "/"), "sha256": sha, "rows": len(tab), "plan_tokens": len(plan_tokens), "plan_tokens_covered": len(plan_tokens)}


def apply_command_source(cand: dict, nav_mode: str, environ=None) -> dict:
    """G4: decide, BEFORE any GPU lock, how the dump gets its commands. command-mode + source: export SRTTC_COMMANDS_JSON / _SHA256 (the landed driver builds the dump argv and
    cannot forward a flag; its subprocess env is a copy of os.environ) and require the G4 dump; command-mode with NO source: REFUSE (as the landed dump did);
    goal-mode: the source is ignored and nothing is exported (a P2 bank is the landed run)."""
    environ = os.environ if environ is None else environ
    src = cand.get("commands")
    if nav_mode == "command":
        if not src:
            raise BuildError("a command-mode checkpoint needs a command source: `prepare --commands <P6 commands.json>` (the dump REFUSES a command-mode planner without one)")
        if sha256_file(src["path"]) != src["sha256"]:
            raise BuildError("the command source changed since `prepare` (sha256)")
        if cand["s0_code_pins"].get("srttc_s0_dump.py") != S0_DUMP_G4_SHA256:
            raise BuildError("a command-mode checkpoint needs the G4 srttc_s0_dump.py (--commands); the landed one refuses it: point `prepare --s0-code` at a copy of the S0 code dir with the G4 dump")
        environ["SRTTC_COMMANDS_JSON"], environ["SRTTC_COMMANDS_SHA256"] = src["path"], src["sha256"]
        return {"nav_mode": nav_mode, "commands_exported": True, "sha256": src["sha256"]}
    for k in ("SRTTC_COMMANDS_JSON", "SRTTC_COMMANDS_SHA256"):
        environ.pop(k, None)
    return {"nav_mode": nav_mode, "commands_exported": False, "note": "goal-mode checkpoint: a supplied command source is ignored" if src else None}


def read_candidate(root):''')
rep('''    D = load_s0_driver(cand)
    argv = ["--tag", "main",''', '''    cmdinfo = apply_command_source(cand, ckpt_nav_mode(cand["ckpt"]))
    print(f"[{now()}] command source: {cmdinfo}", flush=True)
    D = load_s0_driver(cand)
    argv = ["--tag", "main",''')
rep('''    p.add_argument("--candidate-id", default="p6")''', '''    p.add_argument("--candidate-id", default="p6")
    p.add_argument("--commands", default=None, help="G4: the P6 command table (HF logs/commands.json), required for a command-mode checkpoint; must cover every S0 plan token")
    p.add_argument("--commands-sha256", default=None)''')
rep('''    rep["checks"] = {"score_head_is_256_to_6":''', '''    rep["nav_mode"] = m_c.get("nav_mode", "goal")
    rep["needs_command_source"] = bool(rep["nav_mode"] == "command")
    rep["checks"] = {"score_head_is_256_to_6":''')
open("C:/Users/Admin/par18/p6gaps/code/%s/raw/2026-10-19-p6-read-prep/code/build_candidate_bank.py" % PK, "w", encoding="utf-8", newline="").write(s)
print("ok landed pin", landed[:16])
