"""patch_rhk_gate.py -- build rhk_gate_record.py from the pod's rh_gate_record.py (sha256 84701957...) by anchored edits (each anchor counted, else refuse). Usage: patch_rhk_gate.py SRC DST
Three edits, nothing else (the record's checks, its COMMIT, its binds and its verify are the RH arm's):
  1. the bank files are the readout-K bank: $R/bank_k/s<seed>/{bank.json,BANK_CHECK.json}  (the RH bank is never read for an RHK tag);
  2. the PILOT_RH.json readout clause: the pilot (a: defaults == unpatched, which does not depend on the readout) is the code-level pilot of the RH chain at the same commit and reads readout causal_g1x; the K bank's own
     readout is proved by the FULL-bank check (BANK_CHECK.json, readout causal_g0x, b to e on every record) which this record still requires for the declared readout;
  3. the tool name in the record (schema note)."""
import hashlib
import sys

SRC, DST = sys.argv[1], sys.argv[2]
s = open(SRC, encoding="utf-8").read()
assert hashlib.sha256(s.encode("utf-8")).hexdigest().startswith("847019579b8ae27135f0"), "source is not the pod's rh_gate_record.py"


def sub(old, new, n=1):
    global s
    assert s.count(old) == n, (old[:70], s.count(old), n)
    s = s.replace(old, new)


sub('        f["bank_json"] = f"{R}/bank/s{seed}/bank.json"\n        f["bank_check"] = f"{R}/bank/s{seed}/BANK_CHECK.json"\n',
    '        f["bank_json"] = f"{R}/bank_k/s{seed}/bank.json"                       # RHK: the readout-K bank, never the RH bank\n        f["bank_check"] = f"{R}/bank_k/s{seed}/BANK_CHECK.json"\n')
sub('    if pl.get("pass") is not True or pl.get("readout") != ro:\n        bad.append(f"PILOT_RH.json pass={pl.get(\'pass\')} readout={pl.get(\'readout\')!r} (want pass, {ro})")\n',
    '    if pl.get("pass") is not True or pl.get("readout") != PILOT_READOUT:      # RHK: the RH chain\'s code-level pilot (same commit); the K readout is proved by the full-bank check below\n        bad.append(f"PILOT_RH.json pass={pl.get(\'pass\')} readout={pl.get(\'readout\')!r} (want pass, {PILOT_READOUT})")\n')
sub('COMMIT = "97fe0a56767320dbc5f4b823d67fb07a3209a745"\n', 'COMMIT = "97fe0a56767320dbc5f4b823d67fb07a3209a745"\nPILOT_READOUT = "causal_g1x"\nK_TOKEN = "causal_g0x"\n')
sub('    tag, ro = a.tag, a.readout\n', '    tag, ro = a.tag, a.readout\n    if not tag.startswith("RHK_") or ro != K_TOKEN:\n        print("PROBLEM rhk_gate_record.py mints RHK_s<k> tags at readout", K_TOKEN, "only")\n        return 1\n')
sub('"note": "chain-local record; NOT a dscale10k_check / launch_gate token"', '"note": "chain-local record (RHK build of rh_gate_record.py); NOT a dscale10k_check / launch_gate token"')
open(DST, "w", encoding="utf-8", newline="").write(s)
print("OK", hashlib.sha256(s.encode("utf-8")).hexdigest())
