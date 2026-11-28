"""Dump every argparse action of train_v6_staged (build_parser + main's extra flag) to JSON."""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, r"C:/Users/Admin/v7f_gate/tree_m/stack/scripts")
sys.path.insert(0, r"C:/Users/Admin/v7f_gate/tree_m/stack")
import train_v6_staged as T

ap = T.build_parser()
ap.add_argument("--i-know-this-is-the-control-arm", action="store_true",
                dest="control_arm_ack", help=argparse.SUPPRESS)
rows = []
for a in ap._actions:
    if a.dest == "help":
        continue
    rows.append({"dest": a.dest, "options": list(a.option_strings),
                 "default": a.default if isinstance(a.default, (int, float, str, bool, type(None), list, tuple)) else repr(a.default),
                 "type": getattr(a.type, "__name__", repr(a.type)) if a.type else None,
                 "choices": list(a.choices) if a.choices else None,
                 "nargs": a.nargs, "action": type(a).__name__, "required": bool(a.required),
                 "help": (a.help or "").split("\n")[0][:160] if a.help != argparse.SUPPRESS else "SUPPRESS"})
Path(sys.argv[1]).write_text(json.dumps(rows, indent=1, default=str), encoding="utf-8")
print(len(rows), "dests;", len({r["dest"] for r in rows}), "unique")
