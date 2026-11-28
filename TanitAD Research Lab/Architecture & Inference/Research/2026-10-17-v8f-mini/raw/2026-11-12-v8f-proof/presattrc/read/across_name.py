#!/usr/bin/env python
"""across_name.py -- card PREREG_V8F_PROOF_ADD115 (PRESATTRC10K), section 4: the one name across settings, from the name on the unsafe rate at CONV (result file of this
card) and the one at K0 (result file of card ADD114). A result file exists only when the controls of its card passed (the names stage of each reader refuses otherwise),
and this tool refuses a file whose card name is not the expected one.

  python across_name.py --conv-result PRESATTRC10K_RESULT.json --k0-result PRESATTR10K_RESULT.json
"""
import argparse
import io
import json
import sys

KINDS = ("PRESENCE-CARRIES-UNSAFE", "GAP-CARRIES-UNSAFE", "BOTH-CARRY-UNSAFE", "UNSAFE-NOT-ATTRIBUTED")
NOT_ATTR = "UNSAFE-NOT-ATTRIBUTED"
PRE_CONV, PRE_K0 = "PRESATTRC10K-CONV-", "PRESATTR10K-K0-"
CARD_CONV, CARD_K0 = "PREREG_V8F_PROOF_ADD115", "PREREG_V8F_PROOF_ADD114"


def kind_of(name, prefix):
    if not isinstance(name, str) or not name.startswith(prefix) or name[len(prefix):] not in KINDS:
        raise SystemExit("REFUSED: %r is not a name on the unsafe rate with the prefix %s" % (name, prefix))
    return name[len(prefix):]


def across(conv_name, k0_name):
    c, k = kind_of(conv_name, PRE_CONV), kind_of(k0_name, PRE_K0)
    if c == NOT_ATTR and k == NOT_ATTR:
        return "PRESATTRC10K-NEITHER-SETTING-ATTRIBUTED"
    if c == NOT_ATTR or k == NOT_ATTR:
        return "PRESATTRC10K-ONE-SETTING-NOT-ATTRIBUTED"
    return "PRESATTRC10K-ATTRIBUTION-AS-AT-K0" if c == k else "PRESATTRC10K-ATTRIBUTION-NOT-AS-AT-K0"


def name_from(path, card, key):
    d = json.load(io.open(path, encoding="utf-8"))
    if d.get("card") != card:
        raise SystemExit("REFUSED: %s is not a result file of %s" % (path, card))
    nm = (d.get("names") or {}).get(key)
    if nm is None:
        raise SystemExit("REFUSED: %s has no name %s" % (path, key))
    return nm


def main(argv=None):
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--conv-result", required=True)
    ap.add_argument("--k0-result", required=True)
    a = ap.parse_args(argv)
    cn = name_from(a.conv_result, CARD_CONV, "CONV_OVERALL")
    kn = name_from(a.k0_result, CARD_K0, "K0_OVERALL")
    print("CONV %s" % cn)
    print("K0 %s" % kn)
    print("ZZ ACROSS-NAME %s" % across(cn, kn))
    return 0


if __name__ == "__main__":
    sys.exit(main())
