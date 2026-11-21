#!/usr/bin/env python3
"""finish.py table | logs | paths   (B8 carve feed; run after C:/Users/Admin/b8carve/full/MAIN_DONE.txt exists)
table : one markdown row per test: token line (or rc + last line) in base_full / mine_full / gate_full  -> stdout
logs  : copy full/{base,mine,gate}_logs/* to out/<plan>/raw/2026-10-18-r0e-carvefeed/logs/{base,mine,gate}_lane/
paths : rebuild out/paths.json from every file under out/ (repo-relative path, absolute source, "raw")"""
import json, os, shutil, sys
B = "C:/Users/Admin/b8carve"
F = B + "/full"
PLAN = "TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
OUT = f"{B}/out"
PK = f"{OUT}/{PLAN}/raw/2026-10-18-r0e-carvefeed"


def load(tag):
    p = f"{F}/{tag}_summary.json"
    return {k: v for k, v in json.load(open(p)).items() if k != "_done"} if os.path.exists(p) else {}


def cell(v):
    if v is None:
        return "not run"
    if v["rc"] == 0:
        return "rc 0: " + (v["tokens"][-1] if v["tokens"] else "(no ZZ line)")[:110]
    last = v["tail"][-1] if v["tail"] else ""
    return f"**rc {v['rc']}**: " + last.strip()[:150].replace("|", "/")


def table():
    S = {t: load(t) for t in ("base", "mine", "gate")}
    names = []
    for t in ("gate", "mine", "base"):
        for k in S[t]:
            if k not in names:
                names.append(k)
    print("| test | BASE (tip d90d033) | MINE (B8, tip gate) | GATE (B8 + G-DVB patch) |")
    print("|---|---|---|---|")
    for n in names:
        print(f"| `{n}` | " + " | ".join(cell(S[t].get(n)) for t in ("base", "mine", "gate")) + " |")


def logs():
    for tag in ("base", "mine", "gate"):
        src, dst = f"{F}/{tag}_logs", f"{PK}/logs/{tag}_lane"
        if not os.path.isdir(src):
            continue
        os.makedirs(dst, exist_ok=True)
        for f in os.listdir(src):
            shutil.copy2(f"{src}/{f}", f"{dst}/{f}")
        shutil.copy2(f"{F}/{tag}_summary.json", f"{dst}/_summary.json") if os.path.exists(f"{F}/{tag}_summary.json") else None
        print(tag, len(os.listdir(dst)), "files")


def paths():
    out = []
    for root, _, fs in os.walk(OUT):
        for f in fs:
            if f == "paths.json":
                continue
            a = os.path.join(root, f).replace("\\", "/")
            out.append([a[len(OUT) + 1:], a, "raw"])
    out.sort()
    json.dump(out, open(f"{OUT}/paths.json", "w"), indent=1)
    print(len(out), "entries")


{"table": table, "logs": logs, "paths": paths}[sys.argv[1]]()
