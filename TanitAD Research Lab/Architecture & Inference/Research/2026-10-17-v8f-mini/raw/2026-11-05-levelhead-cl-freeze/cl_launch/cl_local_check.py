"""Dev-box side of the arm-CL launchers (stdlib only; reads local files; NO network, NO subprocess, nothing on box 2). The launchers run it BEFORE they take the lock or make any remote call.
  python -I cl_local_check.py wprobe     the W probe may launch: PINS.json = the embedded pins of box2_files/cl_check.py, the four staged stack sources hash to their pins, WPROBE_0's argv passes the audit
  python -I cl_local_check.py chain      the chain may launch: all of the above for CL_0 / CL_1, plus W frozen:
                                           cl_prereg/FROZEN_W.txt exists and parses; it is the W in both argv files, in cl_check.py's pins and in the shipped copy (box2_files/FROZEN_W.txt, byte for byte);
                                           cl_prereg/W_PROBE.json exists and the frozen W is within 1 % of its level_head_probe.w_level_for_10pct_share (W is fixed by the card's formula, not chosen);
                                           the card on disk is the card the pins were made from (sha256), and it names the frozen W;
  python -I cl_local_check.py ship-files   the relative paths under box2_files/ the launchers ship (one per line)
  python -I cl_local_check.py stack-rows   `rel|source|sha256` per shipped stack file
Exit 0 = allowed; 6 = refused (every reason printed, prefixed ZZCL-REFUSED). Test hooks: CL_PREREG_DIR (the card folder), CL_PKG (the package folder)."""
import hashlib
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

H = Path(os.environ.get("CL_PKG") or Path(__file__).resolve().parent)
B2 = H / "box2_files"
PREREG = Path(os.environ.get("CL_PREREG_DIR") or H.parent / "cl_prereg")
W_TOL = 0.01


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_check():
    os.environ["CL_HOME"] = str(B2)                              # the checker reads its argv files from CL_HOME/argv
    sp = importlib.util.spec_from_file_location("cl_check_local", B2 / "cl_check.py")
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    return m


def ship_files():
    return sorted(str(p.relative_to(B2)).replace("\\", "/") for p in B2.rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc")


def problems(mode):
    bad = []
    if not (B2 / "cl_check.py").exists():
        return ["box2_files/cl_check.py is missing"]
    ck = load_check()
    pins = json.loads((H / "PINS.json").read_text(encoding="utf-8"))
    if pins != ck.PINS:
        bad.append("PINS.json differs from the pins embedded in box2_files/cl_check.py (run mk_cl_package.py)")
    for x in pins["stack"]:
        p = H / x["src"]
        if not p.exists() or sha(p) != x["sha256"]:
            bad.append(f"stack source {x['src']} is missing or its sha256 != the pin {x['sha256'][:12]}")
    tags = ["WPROBE_0"] if mode == "wprobe" else ["CL_0", "CL_1"]
    for t in tags:
        try:
            bad += [f"{t}: argv audit: {m}" for m in ck.audit_argv(t, ck.shipped_line(t).split(" "))]
        except Exception as e:                                   # noqa: BLE001
            bad.append(f"{t}: {type(e).__name__}: {e}")
    if mode != "chain":
        return bad
    w = pins["w"]
    fw = PREREG / "FROZEN_W.txt"
    if w == "__W__" or not ck.w_ok(w):
        bad.append(f"the pinned W is {w!r}: FROZEN_W.txt is not frozen into the package (write cl_prereg/FROZEN_W.txt, then python -I mk_cl_package.py)")
    if not fw.exists():
        bad.append(f"{fw} is absent: the card's W is not frozen")
    else:
        m = None
        for l in fw.read_text(encoding="utf-8").splitlines():
            if l.strip() and not l.lstrip().startswith("#"):
                m = re.fullmatch(r"\s*(?:W\s*=\s*)?([0-9]+(?:\.[0-9]+)?(?:[eE][-+]?[0-9]+)?)\s*", l)
                break
        if not m:
            bad.append(f"{fw}: the first non-comment line is not a positive number")
        elif m.group(1) != w:
            bad.append(f"FROZEN_W.txt says {m.group(1)!r} but the package pins {w!r} (regenerate: python -I mk_cl_package.py)")
        if (B2 / "FROZEN_W.txt").exists() and sha(B2 / "FROZEN_W.txt") != sha(fw):
            bad.append("box2_files/FROZEN_W.txt is not the cl_prereg/FROZEN_W.txt on disk (regenerate)")
        elif not (B2 / "FROZEN_W.txt").exists():
            bad.append("box2_files/FROZEN_W.txt is missing (regenerate)")
    for t in tags:
        toks = ck.shipped_line(t).split(" ")
        if "--w-level" not in toks or toks[toks.index("--w-level") + 1] != w:
            bad.append(f"{t}: the argv's --w-level is not the frozen W {w!r}")
    wp = PREREG / "W_PROBE.json"
    if not wp.exists():
        bad.append(f"{wp} is absent: W comes from the probe (launch_wprobe.sh, pull_wprobe.sh)")
    elif ck.w_ok(w):
        try:
            raw = float(json.loads(wp.read_text(encoding="utf-8"))["w_level_for_10pct_share"])
            if not raw > 0 or abs(float(w) - raw) > W_TOL * raw:
                bad.append(f"the frozen W {w} is not within {W_TOL:.0%} of the probe's w_level_for_10pct_share {raw}")
        except Exception as e:                                   # noqa: BLE001
            bad.append(f"{wp} unreadable or without w_level_for_10pct_share: {type(e).__name__}: {e}")
    card = PREREG / pins["card_file"]
    if not card.exists():
        bad.append(f"the card {card} is missing")
    else:
        if sha(card) != pins["card_sha256"]:
            bad.append("the card on disk is not the card the pins were made from (its sha256 changed: freeze it, then regenerate with mk_cl_package.py)")
        if w != "__W__" and w not in card.read_text(encoding="utf-8"):
            bad.append(f"the card does not name the frozen W {w} (paste the freeze line first)")
    return bad


def main(a):
    cmd = a[0] if a else ""
    if cmd == "ship-files":
        print("\n".join(ship_files())); return 0
    if cmd == "stack-rows":
        for x in json.loads((H / "PINS.json").read_text(encoding="utf-8"))["stack"]:
            src = (H / x["src"]).as_posix()                      # git-bash form (/e/...): scp reads `E:/...` as a host named E
            src = "/" + src[0].lower() + src[2:] if re.match(r"^[A-Za-z]:/", src) else src
            print(f"{x['rel']}|{src}|{x['sha256']}")
        return 0
    if cmd in ("wprobe", "chain"):
        bad = problems(cmd)
        if bad:
            for b in bad:
                print("ZZCL-REFUSED " + b)
            return 6
        print(f"local check OK ({cmd})"); return 0
    print(__doc__); return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
