import sys
PK="TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
s=open("C:/Users/Admin/par18/p6gaps/raw/srttc_s0_dump.TIP.py",encoding="utf-8",newline="").read()
assert s.count("\r\n")==229 and s.count("\n")==229
s=s.replace("\r\n","\n")
def rep(old,new):
    global s
    assert s.count(old)==1,(s.count(old),old[:70]); s=s.replace(old,new)
rep('''Prints ZZS0DUMP_OK <tokens> <sets> only when every requested token has a finite row and every fit set of those tokens a finite one.
''','''Prints ZZS0DUMP_OK <tokens> <sets> only when every requested token has a finite row and every fit set of those tokens a finite one.

P6-READ-GAPS G4 (2026-10-22): COMMAND-MODE checkpoints (P6, P6-N) are accepted when a COMMAND SOURCE is given: `--commands <commands.json>` (or env SRTTC_COMMANDS_JSON, because the landed
driver builds this script's argv and cannot forward a flag; `--commands-sha256` / SRTTC_COMMANDS_SHA256 pins the file). The source is P6's own training command table (HF logs/commands.json,
103,037 scenes): {"<log>/<token>": one-hot [LEFT, STRAIGHT, RIGHT, UNKNOWN] = NAVSIM's driving_command}. Each token's command is handed to the planner by the seam's OWN
`set_driving_command` (the function the navtest seam and the P6 dry run use). A token with no command, or a non-one-hot vector, RAISES (never a default). A command-mode (or sanitising)
checkpoint with NO source is still REFUSED, exactly as before; a goal-mode checkpoint (P2) ignores a supplied source and the run is the landed one.
''')
rep('''def wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi
''','''def wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def load_command_table(path, sha256=None) -> dict:
    """P6's command table: {"<log>/<token>": [4 ints]}; refuses a file whose sha256 differs from the pin, an empty table, and any non-one-hot row"""
    raw = open(path, "rb").read()
    got = hashlib.sha256(raw).hexdigest()
    if sha256 and got != sha256.lower():
        raise SystemExit(f"REFUSED: command table {path} sha256 {got} != the pinned {sha256}")
    tab = json.loads(raw.decode("utf-8"))
    if not isinstance(tab, dict) or not tab:
        raise SystemExit(f"REFUSED: command table {path} is empty or not a dict")
    for k, v in tab.items():
        if "/" not in k or len(v) != 4 or sorted(int(x) for x in v) != [0, 0, 0, 1]:
            raise SystemExit(f"REFUSED: command table {path}: row {k!r} = {v!r} is not '<log>/<token>' -> one-hot [LEFT, STRAIGHT, RIGHT, UNKNOWN]")
    return {"path": os.path.abspath(path), "sha256": got, "rows": tab}


def command_vec(table: dict, log: str, tok: str) -> list:
    """the token's one-hot command; RAISES when the table has none (a command-mode model fed a missing command would answer an unknown question)"""
    key = f"{log}/{tok}"
    if key not in table["rows"]:
        raise RuntimeError(f"token {tok} (log {log}): no command in {table['path']}: a command-mode checkpoint is never run on a default command")
    return [int(x) for x in table["rows"][key]]
''')
rep('''    ap.add_argument("--lean", action="store_true", help="do not store the token-level fan arrays (proposals, proposals20, fan features)")
''','''    ap.add_argument("--lean", action="store_true", help="do not store the token-level fan arrays (proposals, proposals20, fan features)")
    ap.add_argument("--commands", default=os.environ.get("SRTTC_COMMANDS_JSON") or None, help="G4: P6's command table (see the docstring); required for a command-mode checkpoint")
    ap.add_argument("--commands-sha256", default=os.environ.get("SRTTC_COMMANDS_SHA256") or None)
''')
rep('''    if nav_mode == "command" or planner.sanitize_goal:
        print("  REFUSED: a command-mode / sanitising planner needs NAVSIM's driving_command, which navtrain tokens do not carry here", flush=True)
        return 2
''','''    cmd_table = None
    if nav_mode == "command" or planner.sanitize_goal:
        if not a.commands:
            print("  REFUSED: a command-mode / sanitising planner needs NAVSIM's driving_command, which navtrain tokens do not carry here "
                  "(give --commands <P6 commands.json> or set SRTTC_COMMANDS_JSON)", flush=True)
            return 2
        cmd_table = load_command_table(a.commands, a.commands_sha256)
        print(f"  command source: {cmd_table['path']} sha256 {cmd_table['sha256'][:16]} rows {len(cmd_table['rows']):,}", flush=True)
    n_cmd_set = 0
''')
rep('''            feat_box.clear()
            sctx_box.clear()
            traj, _score, k = planner.infer(ego, img)''','''            feat_box.clear()
            sctx_box.clear()
            if cmd_table is not None:                                       # G4: NAVSIM's driving_command for this token, through the seam's own hop
                from refe_navtest_seam import set_driving_command
                n_cmd_set += int(set_driving_command(planner, tok, {"ego_statuses": [{"driving_command": command_vec(cmd_table, log, tok)}]}))
            traj, _score, k = planner.infer(ego, img)''')
rep('''"nav_mode": nav_mode, "sanitize_goal": planner.sanitize_goal,''','''"nav_mode": nav_mode, "sanitize_goal": planner.sanitize_goal,
           "commands": ({"path": cmd_table["path"], "sha256": cmd_table["sha256"], "rows": len(cmd_table["rows"]), "set": n_cmd_set} if cmd_table is not None else None),''')
out=s.replace("\n","\r\n")
open("C:/Users/Admin/par18/p6gaps/code/%s/raw/2026-10-18-sel-rank/code/srttc_s0_dump.py"%PK,"w",encoding="utf-8",newline="").write(out)
print("ok")
