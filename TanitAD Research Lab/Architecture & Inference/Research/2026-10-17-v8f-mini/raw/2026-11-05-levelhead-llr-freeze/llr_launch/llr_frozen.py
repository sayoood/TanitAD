"""The LLR freeze file and the CLR wiring reading, ONE code path for the generator, the local check, freeze_llr.py and the probe reader (stdlib only, no network). Loaded BY PATH (python -I does not put the script folder on sys.path).

`a2_prereg/FROZEN_LLR.txt` (written only by freeze_llr.py, never by hand), one `KEY=value` per line, `#` comments:
  WIRING=add|token         first (before the W_lead probe): CLR's O-sensitivity (clr_read/raw/CLR_WIRING.json decision O-SENSITIVE -> add, O-INSENSITIVE -> token), or an explicit MM ruling carried on a RULING= line
  RULING=<text>            only when the CLR decision names no wiring (WEAK / split seeds / REFUSED): the MM's words, verbatim
  W_LEAD=<positive number> second (after the probe): lead_head_probe.w_lead_for_10pct_share of the W_lead probe, 4 significant figures
"""
import json
import re
from pathlib import Path

NUM = re.compile(r"[0-9]+(?:\.[0-9]+)?(?:[eE][-+]?[0-9]+)?")
KEYS = ("WIRING", "RULING", "W_LEAD")
WIRING_OF = {"O-SENSITIVE": "add", "O-INSENSITIVE": "token"}      # the card's rule (PREREG_LEVELHEAD_A2.md, Arms)


def parse_frozen(path):
    """({KEY: value}, [problems]) of a freeze file; a missing file is ({}, [])."""
    p = Path(path)
    if not p.exists():
        return {}, []
    d, bad = {}, []
    for n, l in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        if not l.strip() or l.lstrip().startswith("#"):
            continue
        m = re.fullmatch(r"\s*([A-Z_]+)\s*=\s*(.*?)\s*", l)
        if not m or m.group(1) not in KEYS:
            bad.append(f"{p.name}:{n}: not a KEY=value line with a known key {KEYS}: {l!r}")
            continue
        k, v = m.groups()
        if k in d:
            bad.append(f"{p.name}:{n}: {k} is given twice")
        d[k] = v
    if "WIRING" in d and d["WIRING"] not in ("add", "token"):
        bad.append(f"{p.name}: WIRING={d['WIRING']!r} (must be add or token)")
    if "W_LEAD" in d and not (NUM.fullmatch(d["W_LEAD"]) and float(d["W_LEAD"]) > 0):
        bad.append(f"{p.name}: W_LEAD={d['W_LEAD']!r} is not a positive number")
    if "W_LEAD" in d and "WIRING" not in d:
        bad.append(f"{p.name}: W_LEAD is frozen but WIRING is not (the wiring is frozen first: the probe runs with it)")
    return d, bad


def clr_decision(path):
    """(decision, wiring or None, problems) read from clr_read/raw/CLR_WIRING.json."""
    p = Path(path)
    if not p.exists():
        return None, None, [f"{p} is absent: the CLR read has not written the wiring decision"]
    try:
        dec = json.loads(p.read_text(encoding="utf-8"))["WIRING"]["decision"]
    except Exception as e:                                       # noqa: BLE001
        return None, None, [f"{p} unreadable or without WIRING.decision: {type(e).__name__}: {e}"]
    return dec, WIRING_OF.get(dec), []
