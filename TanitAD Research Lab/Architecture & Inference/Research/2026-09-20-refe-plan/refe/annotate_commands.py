#!/usr/bin/env python3
"""P6 / D6 (pod or any box with the NAVSIM scene metadata): write NAVSIM's OWN `driving_command` into the bank rows as `row["command"]` (an int 0..3 =
LEFT/STRAIGHT/RIGHT/UNKNOWN), so the student is conditioned at TRAIN time on the same quantity NAVSIM hands it at TEST time. MEASURED (refe/measure_command_rule.py): the goal
rule agrees with NAVSIM on only 0.90 of navtest tokens (SPEC_P6 D6's bar is 0.95), so rank-0 rows should carry the real command; goal-augmented twins keep the goal rule
(their command must follow their OWN augmented goal).  Rows that already carry a `command` are left alone; the table is {"<log>/<token>": [l, s, r, u] or int}.
    python annotate_commands.py --bank <dir with targets_rank0.jsonl> --table commands.json [--out-dir <new dir>]    (never edits in place unless --in-place)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nav_command as NC  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--table", required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--in-place", action="store_true")
    a = ap.parse_args()
    table = json.load(open(a.table, encoding="utf-8"))
    src = Path(a.bank)
    dst = src if a.in_place else Path(a.out_dir or (str(src) + "_cmd"))
    dst.mkdir(parents=True, exist_ok=True)
    n = hit = 0
    for fp in sorted(src.glob("targets_rank*.jsonl")):
        out = []
        for ln in open(fp, encoding="utf-8"):
            r = json.loads(ln)
            n += 1
            if r.get("command") is None and int(r.get("rank", 0)) == 0 and not r.get("aug"):
                v = table.get(f"{r['log_name']}/{r['token']}")
                if v is not None:
                    r["command"] = NC.from_driving_command(v) if isinstance(v, list) else int(v)
                    hit += 1
            out.append(json.dumps(r) + "\n")
        (dst / fp.name).write_text("".join(out), encoding="utf-8", newline="\n")
    print(f"ZZANNOTATE rows={n} annotated={hit} -> {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
