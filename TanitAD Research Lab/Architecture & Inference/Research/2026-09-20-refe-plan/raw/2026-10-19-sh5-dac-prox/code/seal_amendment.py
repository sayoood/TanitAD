"""Seal a written Amendment k: fills {{PIN_TABLE}} / {{SEALED_AT}} in the amendment draft, writes PREREG_SH5_DAC_PROX_AMENDMENT<k>.md, its .sha256 and CODE_PINS_AMENDMENT<k>.json ({"from","to"}).
The guard (sh5_guard.require_sealed) then accepts the new hashes of exactly the listed files, starting from their CURRENT pins; anything else changed since the seal still refuses.

    python seal_amendment.py --k 1 --draft AMENDMENT1.draft.md --root C:/Users/Admin/par18/sh5 --files code/sh5_branch.py code/sh5_train_eval.py ...
"""
import argparse
import datetime
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sh5_guard as G  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--draft", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--files", nargs="+", required=True)
    a = ap.parse_args()
    r, k = a.root, a.k
    out_md = os.path.join(r, f"PREREG_SH5_DAC_PROX_AMENDMENT{k}.md")
    if os.path.exists(out_md):
        raise SystemExit(f"Amendment {k} is already sealed")
    merged = dict(json.load(open(os.path.join(r, "CODE_PINS.json"), encoding="utf-8"))["files"])
    for j in range(1, k):
        merged.update(json.load(open(os.path.join(r, f"CODE_PINS_AMENDMENT{j}.json"), encoding="utf-8"))["to"])
    frm, to = {}, {}
    for n in a.files:
        if n not in merged:
            raise SystemExit(f"{n} is not a pinned file")
        frm[n], to[n] = merged[n], G.sha256_file(os.path.join(r, n))
        if frm[n] == to[n]:
            raise SystemExit(f"{n} did not change: do not list it")
    now = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    table = "| file | pin before | pin after |\n|---|---|---|\n" + "\n".join(f"| `{n}` | `{frm[n]}` | `{to[n]}` |" for n in a.files)
    txt = open(a.draft, encoding="utf-8").read()
    assert "{{PIN_TABLE}}" in txt
    chk = ["code/sh5_guard.py", "code/seal.py", "code/seal_amendment.py", "code/test_sh5_guard.py"]
    ctable = "| checker (not pinned) | sha256 now |" + chr(10) + "|---|---|" + chr(10) + chr(10).join(f"| `{n}` | `{G.sha256_file(os.path.join(r, n))}` |" for n in chk)
    txt = txt.replace("{{PIN_TABLE}}", table).replace("{{CHECKER_TABLE}}", ctable).replace("{{SEALED_AT}}", now)
    with open(out_md, "w", encoding="utf-8", newline="\n") as f:
        f.write(txt)
    h = G.sha256_file(out_md)
    with open(os.path.join(r, f"PREREG_SH5_DAC_PROX_AMENDMENT{k}.sha256"), "w", encoding="utf-8", newline="\n") as f:
        f.write(f"{h}  PREREG_SH5_DAC_PROX_AMENDMENT{k}.md\n")
    with open(os.path.join(r, f"CODE_PINS_AMENDMENT{k}.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump({"sealed_at": now, "from": frm, "to": to}, f, indent=1)
    print(json.dumps({"sealed_at": now, "amendment_sha256": h, "files": a.files}))


if __name__ == "__main__":
    main()
