"""Dev-box side of the arm-LLR launchers (stdlib only; reads local files; NO network, NO subprocess, nothing on box 2). The launcher runs it BEFORE it takes the lock or makes any remote call.
  python -I llr_local_check.py wprobe      the W_lead probe may launch: PINS.json = the pins embedded in box2_files/llr_check.py, the four staged stack sources hash to their pins AND are recorded as landed (landed_tip.txt = the pinned commit),
                                           the probe argv passes the audit, RAPR-P01's argv copies are the pinned bytes, CL's frozen W (cl_prereg/FROZEN_W.txt + W_PROBE.json), the A2 card AND ruling are the frozen ones (FROZEN_SHA.txt), and the WIRING is frozen:
                                           a2_prereg/FROZEN_LLR.txt parses, carries WIRING, equals the pinned one and the shipped copy, and agrees with CLR's O-sensitivity (clr_read/raw/CLR_WIRING.json: O-SENSITIVE -> add, O-INSENSITIVE -> token;
                                           a decision that names no wiring needs a RULING= line, and a RULING can never contradict a wiring the card names)
  python -I llr_local_check.py chain       the chain may launch: all of the above for both argv files, and W_LEAD frozen too: FROZEN_LLR.txt W_LEAD = the pinned W_lead = `--w-lead` in both argv files, and a2_prereg/WLEAD_PROBE.json (the W_lead probe's record)
                                           exists, was read under the same wiring, and the frozen W_lead is within 1 % of its w_lead_for_10pct_share
  python -I llr_local_check.py ship-files  the relative paths under box2_files/ the launcher ships (one per line)
  python -I llr_local_check.py stack-rows  `rel|source|sha256` per shipped stack file
Exit 0 = allowed; 6 = refused (every reason printed, prefixed ZZLLR-REFUSED). Test hooks: LLR_PREREG_DIR (the A2 card folder, holds FROZEN_LLR.txt and WLEAD_PROBE.json), LLR_W_DIR (CL's W folder), LLR_WIRING_DIR (CLR's wiring folder), LLR_PKG (the package folder)."""
import hashlib
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

H = Path(os.environ.get("LLR_PKG") or Path(__file__).resolve().parent)
B2 = H / "box2_files"
PREREG = Path(os.environ.get("LLR_PREREG_DIR") or H.parent / "a2_prereg")
W_DIR = Path(os.environ.get("LLR_W_DIR") or H.parent / "cl_prereg")
W_TOL = 0.01
WIRING_DIR = Path(os.environ.get("LLR_WIRING_DIR") or H.parent / "clr_read" / "raw")


def _fz():
    sp = importlib.util.spec_from_file_location("llr_frozen", H / "llr_frozen.py")
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m


FZ = _fz()


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_check():
    os.environ["LLR_HOME"] = str(B2)                              # the checker reads its argv files from LLR_HOME/argv
    sp = importlib.util.spec_from_file_location("llr_check_local", B2 / "llr_check.py")
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    return m


def ship_files():
    return sorted(str(p.relative_to(B2)).replace("\\", "/") for p in B2.rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc")


def problems(mode):
    bad = []
    if not (B2 / "llr_check.py").exists():
        return ["box2_files/llr_check.py is missing"]
    ck = load_check()
    pins = json.loads((H / "PINS.json").read_text(encoding="utf-8"))
    if pins != ck.PINS:
        bad.append("PINS.json differs from the pins embedded in box2_files/llr_check.py (run mk_llr_package.py)")
    for x in pins["stack"]:
        p = H / x["src"]
        if not p.exists() or sha(p) != x["sha256"]:
            bad.append(f"stack source {x['src']} is missing or its sha256 != the pin {x['sha256'][:12]}")
    lt = H / "landed_tip.txt"
    if not pins.get("landed") or not lt.exists() or lt.read_text(encoding="utf-8").strip() != pins["tip_commit"]:
        bad.append("the lead-head files are not recorded as landed (landed_tip.txt absent or not the pinned commit): the MM lands them, then `python -I mk_llr_package.py record-landed`, then regenerates")
    tags = ["WLPROBE_0"] if mode == "wprobe" else ["LLR_0", "LLR_1"]
    for s_ in (0, 1):
        rp = H / "rapr_argv" / f"RAPRP01_s{s_}.argv.json"
        if not rp.exists() or sha(rp) != pins["rapr_argv_sha256"][str(s_)]:
            bad.append(f"rapr_argv/RAPRP01_s{s_}.argv.json is missing or is not the pinned RAPR-P01 argv.json")
    for t_ in tags:
        try:
            bad += [f"{t_}: argv audit: {m}" for m in ck.audit_argv(t_, ck.shipped_line(t_).split(" "))]
        except Exception as e:                                   # noqa: BLE001
            bad.append(f"{t_}: {type(e).__name__}: {e}")
    # ---- CL's frozen W (CLR and LLR reuse it)
    w = pins["w"]
    fw = W_DIR / "FROZEN_W.txt"
    if not ck.w_ok(w):
        bad.append(f"the pinned W is {w!r}: FROZEN_W.txt is not frozen into the package (cl_prereg/FROZEN_W.txt, then python -I mk_llr_package.py)")
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
            bad.append(f"FROZEN_W.txt says {m.group(1)!r} but the package pins {w!r} (regenerate: python -I mk_llr_package.py)")
        if (B2 / "FROZEN_W.txt").exists() and sha(B2 / "FROZEN_W.txt") != sha(fw):
            bad.append("box2_files/FROZEN_W.txt is not the cl_prereg/FROZEN_W.txt on disk (regenerate)")
        elif not (B2 / "FROZEN_W.txt").exists():
            bad.append("box2_files/FROZEN_W.txt is missing (regenerate)")
    for t_ in tags:
        toks = ck.shipped_line(t_).split(" ")
        if "--w-level" not in toks or toks[toks.index("--w-level") + 1] != w:
            bad.append(f"{t_}: the argv's --w-level is not the frozen W {w!r}")
    wp = W_DIR / "W_PROBE.json"
    if not wp.exists():
        bad.append(f"{wp} is absent: W comes from CL's probe")
    elif ck.w_ok(w):
        try:
            raw = float(json.loads(wp.read_text(encoding="utf-8"))["w_level_for_10pct_share"])
            if not raw > 0 or abs(float(w) - raw) > W_TOL * raw:
                bad.append(f"the frozen W {w} is not within {W_TOL:.0%} of the probe's w_level_for_10pct_share {raw}")
        except Exception as e:                                   # noqa: BLE001
            bad.append(f"{wp} unreadable or without w_level_for_10pct_share: {type(e).__name__}: {e}")
    # ---- the LLR freeze file: wiring (both modes), W_lead (chain)
    fl = PREREG / "FROZEN_LLR.txt"
    fzd, fbad = FZ.parse_frozen(fl)
    bad += [f"FROZEN_LLR.txt: {b}" for b in fbad]
    if not fl.exists():
        bad.append(f"{fl} is absent: the wiring is not frozen (freeze_llr.py wiring --write, from CLR's O-sensitivity)")
    elif not fbad:
        wiring = fzd.get("WIRING")
        if wiring is None:
            bad.append(f"{fl.name} carries no WIRING")
        else:
            if pins["wiring"] != wiring:
                bad.append(f"FROZEN_LLR.txt says WIRING={wiring!r} but the package pins {pins['wiring']!r} (regenerate: python -I mk_llr_package.py)")
            if pins.get("frozen_llr_sha256") != sha(fl):
                bad.append("FROZEN_LLR.txt on disk is not the file the pins were made from (regenerate)")
            if not (B2 / "FROZEN_LLR.txt").exists() or sha(B2 / "FROZEN_LLR.txt") != sha(fl):
                bad.append("box2_files/FROZEN_LLR.txt is missing or is not the a2_prereg/FROZEN_LLR.txt on disk (regenerate)")
            dec, mapped, wb = FZ.clr_decision(WIRING_DIR / "CLR_WIRING.json")
            bad += wb
            if not wb:
                if mapped is None and "RULING" not in fzd:
                    bad.append(f"CLR's O-sensitivity decision is {dec!r}, which the card does not map to a wiring: FROZEN_LLR.txt needs the MM's RULING= line")
                if mapped is not None and mapped != wiring:
                    bad.append(f"CLR's decision {dec!r} fixes the wiring to {mapped!r} (card rule), FROZEN_LLR.txt says {wiring!r}: a RULING never contradicts a wiring the card names")
            for t_ in tags:
                toks = ck.shipped_line(t_).split(" ")
                if "--aux-cond" not in toks or toks[toks.index("--aux-cond") + 1] != wiring:
                    bad.append(f"{t_}: the argv's --aux-cond is not the frozen wiring {wiring!r}")
    if mode == "chain":
        wl = pins["w_lead"]
        if not ck.w_ok(wl):
            bad.append(f"the pinned W_lead is {wl!r}: not frozen (probe -> pull -> freeze_llr.py wlead --write -> python -I mk_llr_package.py)")
        if fzd.get("W_LEAD") is None:
            bad.append(f"{fl.name} carries no W_LEAD (the W_lead probe has not been frozen)")
        elif fzd["W_LEAD"] != wl:
            bad.append(f"FROZEN_LLR.txt says W_LEAD={fzd['W_LEAD']!r} but the package pins {wl!r} (regenerate: python -I mk_llr_package.py)")
        for t_ in tags:
            toks = ck.shipped_line(t_).split(" ")
            if "--w-lead" not in toks or toks[toks.index("--w-lead") + 1] != wl:
                bad.append(f"{t_}: the argv's --w-lead is not the frozen W_lead {wl!r}")
        wlp = PREREG / "WLEAD_PROBE.json"
        if not wlp.exists():
            bad.append(f"{wlp} is absent: W_lead comes from the W_lead probe (launch_wleadprobe.sh, pull_wleadprobe.sh)")
        elif ck.w_ok(wl):
            try:
                rec = json.loads(wlp.read_text(encoding="utf-8"))
                raw = float(rec["w_lead_for_10pct_share"])
                if not raw > 0 or abs(float(wl) - raw) > W_TOL * raw:
                    bad.append(f"the frozen W_lead {wl} is not within {W_TOL:.0%} of the probe's w_lead_for_10pct_share {raw}")
                if rec.get("aux_cond") != pins["wiring"]:
                    bad.append(f"the W_lead probe was read under wiring {rec.get('aux_cond')!r}, the package pins {pins['wiring']!r}: probe again under the frozen wiring")
            except Exception as e:                               # noqa: BLE001
                bad.append(f"{wlp} unreadable or without w_lead_for_10pct_share: {type(e).__name__}: {e}")
    # ---- the frozen A2 card and its ruling
    for key, fkey in (("card_file", "card_sha256"), ("ruling_file", "ruling_sha256")):
        f = PREREG / pins[key]
        if not f.exists():
            bad.append(f"{f} is missing")
            continue
        if sha(f) != pins[fkey]:
            bad.append(f"{pins[key]} on disk is not the file the pins were made from (its sha256 changed: the A2 card and ruling are frozen; regenerate only for a deliberate new card)")
        fz = PREREG / pins["card_frozen_sha_file"]
        if not fz.exists():
            bad.append(f"{fz} is absent: the frozen sha256 is not recorded")
        elif sha(f) not in fz.read_text(encoding="utf-8"):
            bad.append(f"{pins[key]}'s sha256 {sha(f)[:12]} is not the one {fz.name} records")
    card = PREREG / pins["card_file"]
    if card.exists() and f"--w-level {w}" not in card.read_text(encoding="utf-8"):
        bad.append(f"the card does not name `--w-level {w}`")
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
                print("ZZLLR-REFUSED " + b)
            return 6
        print(f"local check OK ({cmd})"); return 0
    print(__doc__); return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
