"""Extra gate for the arms of PREREG_V8F_PROOF_ADD12 (SPATIAL) and ADD13 (DATAEFF): the selfcheck of the exact argv must show the BUILT value of every lever the card moves.

  check --selfcheck <selfcheck.json> --card <name> --out <extra_token.json> --expect levers.wm_state=grid info.n_clips=1128 ...

Each expectation is <section>.<key>=<value>; section is `levers` or `info`; the comparison is on the string form (ints and floats that are whole are compared as ints).
Refuses (rc 1, prints REFUSED) on any mismatch or absent key and writes no token. stdlib only.
"""
import argparse
import hashlib
import json
import sys
import time


def norm(v):
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float) and v == int(v):
        return str(int(v))
    return str(v)


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("--selfcheck", required=True)
    c.add_argument("--card", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--expect", nargs="+", required=True)
    a = ap.parse_args(argv)
    with open(a.selfcheck, "rb") as fh:
        raw = fh.read()
    sc = json.loads(raw.decode("utf-8"))
    bad, seen = [], {}
    for e in a.expect:
        if "=" not in e or "." not in e.split("=", 1)[0]:
            bad.append(f"malformed expectation {e!r}")
            continue
        path, want = e.split("=", 1)
        sec, key = path.split(".", 1)
        if sec not in ("levers", "info"):
            bad.append(f"unknown section in {e!r}")
            continue
        d = sc.get(sec, {})
        if key not in d:
            bad.append(f"{path} absent from the selfcheck")
            continue
        got = norm(d[key])
        seen[path] = got
        if got != want:
            bad.append(f"{path}: built {got!r} != expected {want!r}")
    if bad:
        print("REFUSED: " + "; ".join(bad))
        return 1
    tok = {"card": a.card, "gate_extra": "PASS", "expect": seen, "selfcheck_sha256": hashlib.sha256(raw).hexdigest(),
           "selfcheck_argv_sha": sc.get("argv_sha"), "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(tok, fh, indent=1)
    print("extra token written", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
