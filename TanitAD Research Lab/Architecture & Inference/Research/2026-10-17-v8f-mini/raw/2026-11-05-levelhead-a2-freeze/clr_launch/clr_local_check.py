"""Dev-box side of the arm-CLR launchers (stdlib only; reads local files; NO network, NO subprocess, nothing on box 2). The launcher runs it BEFORE it takes the lock or makes any remote call.
  python -I clr_local_check.py chain       the chain may launch: PINS.json = the embedded pins of box2_files/clr_check.py, the four staged stack sources hash to their pins, both argv files pass the audit, RAPR-P01's argv copies are the pinned bytes, and W:
                                           cl_prereg/FROZEN_W.txt exists and parses; it is the W in both argv files, in clr_check.py's pins and in the shipped copy (box2_files/FROZEN_W.txt, byte for byte);
                                           cl_prereg/W_PROBE.json exists and the frozen W is within 1 % of its level_head_probe.w_level_for_10pct_share (CL's probe; arm CLR runs none);
                                           the A2 card on disk is the card the pins were made from (sha256), equals the hash its FROZEN_SHA.txt records, and names `--w-level <W>`.
  python -I clr_local_check.py ship-files  the relative paths under box2_files/ the launcher ships (one per line)
  python -I clr_local_check.py stack-rows  `rel|source|sha256` per shipped stack file
Exit 0 = allowed; 6 = refused (every reason printed, prefixed ZZCLR-REFUSED). Test hooks: CLR_PREREG_DIR (the A2 card folder), CLR_W_DIR (CL's W folder), CLR_PKG (the package folder)."""
import hashlib
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

H = Path(os.environ.get("CLR_PKG") or Path(__file__).resolve().parent)
B2 = H / "box2_files"
PREREG = Path(os.environ.get("CLR_PREREG_DIR") or H.parent / "a2_prereg")
W_DIR = Path(os.environ.get("CLR_W_DIR") or H.parent / "cl_prereg")
W_TOL = 0.01


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_check():
    os.environ["CLR_HOME"] = str(B2)                              # the checker reads its argv files from CLR_HOME/argv
    sp = importlib.util.spec_from_file_location("clr_check_local", B2 / "clr_check.py")
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    return m


def ship_files():
    return sorted(str(p.relative_to(B2)).replace("\\", "/") for p in B2.rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc")


def problems(mode):
    bad = []
    if not (B2 / "clr_check.py").exists():
        return ["box2_files/clr_check.py is missing"]
    ck = load_check()
    pins = json.loads((H / "PINS.json").read_text(encoding="utf-8"))
    if pins != ck.PINS:
        bad.append("PINS.json differs from the pins embedded in box2_files/clr_check.py (run mk_clr_package.py)")
    for x in pins["stack"]:
        p = H / x["src"]
        if not p.exists() or sha(p) != x["sha256"]:
            bad.append(f"stack source {x['src']} is missing or its sha256 != the pin {x['sha256'][:12]}")
    tags = ["CLR_0", "CLR_1"]
    for s_ in (0, 1):
        rp = H / "rapr_argv" / f"RAPRP01_s{s_}.argv.json"
        if not rp.exists() or sha(rp) != pins["rapr_argv_sha256"][str(s_)]:
            bad.append(f"rapr_argv/RAPRP01_s{s_}.argv.json is missing or is not the pinned RAPR-P01 argv.json")
    for t in tags:
        try:
            bad += [f"{t}: argv audit: {m}" for m in ck.audit_argv(t, ck.shipped_line(t).split(" "))]
        except Exception as e:                                   # noqa: BLE001
            bad.append(f"{t}: {type(e).__name__}: {e}")
    w = pins["w"]
    fw = W_DIR / "FROZEN_W.txt"
    if w == "__W__" or not ck.w_ok(w):
        bad.append(f"the pinned W is {w!r}: FROZEN_W.txt is not frozen into the package (cl_prereg/FROZEN_W.txt, then python -I mk_clr_package.py)")
    if not fw.exists():
        bad.append(f"{fw} is absent: CL's frozen W is not on disk")
    else:
        m = None
        for l in fw.read_text(encoding="utf-8").splitlines():
            if l.strip() and not l.lstrip().startswith("#"):
                m = re.fullmatch(r"\s*(?:W\s*=\s*)?([0-9]+(?:\.[0-9]+)?(?:[eE][-+]?[0-9]+)?)\s*", l)
                break
        if not m:
            bad.append(f"{fw}: the first non-comment line is not a positive number")
        elif m.group(1) != w:
            bad.append(f"FROZEN_W.txt says {m.group(1)!r} but the package pins {w!r} (regenerate: python -I mk_clr_package.py)")
        if (B2 / "FROZEN_W.txt").exists() and sha(B2 / "FROZEN_W.txt") != sha(fw):
            bad.append("box2_files/FROZEN_W.txt is not the cl_prereg/FROZEN_W.txt on disk (regenerate)")
        elif not (B2 / "FROZEN_W.txt").exists():
            bad.append("box2_files/FROZEN_W.txt is missing (regenerate)")
    for t in tags:
        toks = ck.shipped_line(t).split(" ")
        if "--w-level" not in toks or toks[toks.index("--w-level") + 1] != w:
            bad.append(f"{t}: the argv's --w-level is not the frozen W {w!r}")
    wp = W_DIR / "W_PROBE.json"
    if not wp.exists():
        bad.append(f"{wp} is absent: W comes from CL's probe (cl_launch/launch_wprobe.sh, pull_wprobe.sh)")
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
            bad.append("the card on disk is not the card the pins were made from (its sha256 changed: the A2 card is frozen; regenerate only for a deliberate new card)")
        if f"--w-level {w}" not in card.read_text(encoding="utf-8"):
            bad.append(f"the card does not name `--w-level {w}`")
        fz = PREREG / pins["card_frozen_sha_file"]
        if not fz.exists():
            bad.append(f"{fz} is absent: the card's frozen sha256 is not recorded")
        elif sha(card) not in fz.read_text(encoding="utf-8"):
            bad.append(f"the card's sha256 {sha(card)[:12]} is not the one {fz.name} records")
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
    if cmd == "chain":
        bad = problems(cmd)
        if bad:
            for b in bad:
                print("ZZCLR-REFUSED " + b)
            return 6
        print("local check OK (chain)"); return 0
    print(__doc__); return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
