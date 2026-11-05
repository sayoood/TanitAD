"""The Master Mind's LLR-freezing helper (stdlib only, no network). TWO steps, in this order; each prints a proposal and writes only with --write, and never overwrites a different frozen value.
  python -I freeze_llr.py wiring [--write]    reads clr_read/raw/CLR_WIRING.json (decision O-SENSITIVE -> add, O-INSENSITIVE -> token) and proposes the line `WIRING=<w>` for a2_prereg/FROZEN_LLR.txt.
                                              When the decision names no wiring (WEAK / split seeds / REFUSED) it REFUSES unless  --wiring add|token --ruling "<the MM's words>"  is given (the RULING= line is written beside it).
                                              A --wiring that contradicts a decision the card maps is refused.
  python -I freeze_llr.py wlead [--write]     reads a2_prereg/WLEAD_PROBE.json (written by pull_wleadprobe.sh from the probe run under the frozen wiring) and proposes W_LEAD = w_lead_for_10pct_share to 4 significant figures;
                                              refuses unless the probe's aux_cond is the frozen wiring.
After --write: `python -I mk_llr_package.py` (regenerates argv, pins and the checker), then `python -I mk_llr_package.py check` and `python -I llr_local_check.py wprobe|chain`.
Test hooks: LLR_PREREG_DIR (a2_prereg), LLR_WIRING_DIR (clr_read/raw)."""
import importlib.util
import json
import os
import sys
from pathlib import Path

H = Path(__file__).resolve().parent
PRE = Path(os.environ.get("LLR_PREREG_DIR") or H.parent / "a2_prereg")
WDIR = Path(os.environ.get("LLR_WIRING_DIR") or H.parent / "clr_read" / "raw")


def _fz():
    sp = importlib.util.spec_from_file_location("llr_frozen", H / "llr_frozen.py")
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m


def append(fl, lines):
    old = fl.read_text(encoding="utf-8") if fl.exists() else ""
    fl.write_text(old + ("" if not old or old.endswith("\n") else "\n") + "\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def wiring(a, FZ):
    fl = PRE / "FROZEN_LLR.txt"
    cur, bad = FZ.parse_frozen(fl)
    if bad:
        print("FROZEN_LLR.txt: " + " | ".join(bad)); return 1
    dec, mapped, wb = FZ.clr_decision(WDIR / "CLR_WIRING.json")
    if wb:
        print("; ".join(wb)); return 1
    given = a[a.index("--wiring") + 1] if "--wiring" in a else None
    ruling = a[a.index("--ruling") + 1] if "--ruling" in a else None
    if mapped is not None:
        if given is not None and given != mapped:
            print(f"CLR's decision {dec!r} fixes the wiring to {mapped!r} (the card's rule); --wiring {given} contradicts it: refused"); return 1
        w, lines = mapped, [f"WIRING={mapped}", f"# from clr_read/raw/CLR_WIRING.json: decision {dec} (card PREREG_LEVELHEAD_A2.md, Arms)"]
    else:
        if given not in ("add", "token") or not ruling:
            print(f"CLR's decision is {dec!r}, which the card does not map to a wiring: pass --wiring add|token --ruling \"<the MM's words>\" to freeze a ruling"); return 1
        w, lines = given, [f"WIRING={given}", f"RULING={ruling}", f"# decision {dec}: the card names no wiring; the MM ruled"]
    print(f"proposal: {lines[0]}   (CLR decision {dec})")
    if "--write" in a:
        if "WIRING" in cur and cur["WIRING"] != w:
            print(f"{fl} already freezes WIRING={cur['WIRING']!r}: not overwritten (move it aside by hand to re-freeze)"); return 1
        if "WIRING" in cur:
            print("WIRING already frozen to the same value: nothing written"); return 0
        append(fl, lines); print(f"wrote {fl}")
    return 0


def wlead(a, FZ):
    fl = PRE / "FROZEN_LLR.txt"
    cur, bad = FZ.parse_frozen(fl)
    if bad or "WIRING" not in cur:
        print(f"{fl}: the wiring is not frozen ({'; '.join(bad) or 'no WIRING'}): freeze the wiring first, run the probe under it"); return 1
    wp = PRE / "WLEAD_PROBE.json"
    if not wp.exists():
        print(f"{wp} is absent: run the W_lead probe and pull it first"); return 1
    rec = json.loads(wp.read_text(encoding="utf-8"))
    if rec.get("aux_cond") != cur["WIRING"]:
        print(f"the probe was read under wiring {rec.get('aux_cond')!r}, the frozen wiring is {cur['WIRING']!r}: probe again"); return 1
    raw = float(rec["w_lead_for_10pct_share"])
    if not raw > 0:
        print(f"w_lead_for_10pct_share {raw} is not > 0"); return 1
    w = f"{raw:.4g}"
    print(f"proposal: W_LEAD={w}  (raw {raw!r}, selfcheck sha256 {rec['selfcheck_sha256']}, wiring {cur['WIRING']})")
    if "--write" in a:
        if "W_LEAD" in cur and cur["W_LEAD"] != w:
            print(f"{fl} already freezes W_LEAD={cur['W_LEAD']!r}: not overwritten (move it aside by hand to re-freeze)"); return 1
        if "W_LEAD" in cur:
            print("W_LEAD already frozen to the same value: nothing written"); return 0
        append(fl, [f"W_LEAD={w}", f"# from WLEAD_PROBE.json: w_lead_for_10pct_share {raw!r}, selfcheck sha256 {rec['selfcheck_sha256']}, wiring {cur['WIRING']}"]); print(f"wrote {fl}")
    return 0


def main(a):
    FZ = _fz()
    if a and a[0] == "wiring":
        return wiring(a, FZ)
    if a and a[0] == "wlead":
        return wlead(a, FZ)
    print(__doc__); return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
