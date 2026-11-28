"""⛔ PROVE THE TWO ARMS DIFFER BY EXACTLY ONE PARSED-NAMESPACE KEY.

⭐ WHY THE PARSED NAMESPACE AND NOT THE COMMAND LINE. Two argv lists that *look*
one token apart can parse to namespaces that differ in several keys — a flag with
a side effect, a default that moves because another flag was set, a `nargs="+"`
that swallows the next token. MEASURED precedent in this programme: the `--v2`
conflation, *"ten levers on two axes, result non-attributable"*. A single-lever
claim is a claim about the **parsed** state, so that is what is asserted.

⚠️ It also reports the keys that are EQUAL, and refuses a comparison where the
two namespaces share fewer than 20 keys — a diff over two nearly-empty objects
would report "one difference" and mean nothing. Same family as the `_same`
vacuity guard in ``test_max_speed_input.py`` (*"parity predicate compared only N
tensors -- vacuous"*).

Usage
-----
    python assert_single_lever.py --expect max_speed_input \\
        --base  "<the OFF line, without the program name>" \\
        --arm   "<the ON line, without the program name>"

or with ``--argv-file <json>`` holding ``{"base": [...], "arm": [...]}``.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import os
import shlex
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
TRAINER = REPO / "stack" / "scripts" / "refc_v3_train.py"

#: A diff over two tiny namespaces is not evidence of anything.
MIN_SHARED_KEYS = 20


def _trainer():
    sys.path.insert(0, str(REPO / "stack"))
    spec = importlib.util.spec_from_file_location("refc_v3_train_for_diff",
                                                  str(TRAINER))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["refc_v3_train_for_diff"] = mod
    spec.loader.exec_module(mod)
    return mod


def parse(tr, argv: list[str]) -> dict:
    """Parse one argv into a plain dict. argparse writes to stderr on error, so
    the failure is captured and re-raised NAMED rather than dumped."""
    buf = io.StringIO()
    try:
        with contextlib.redirect_stderr(buf):
            ns = tr.build_parser().parse_args(argv)
    except SystemExit as e:
        raise SystemExit(f"[single-lever] argv did not parse: {argv}\n"
                         f"{buf.getvalue()}") from e
    return dict(vars(ns))


def diff(base: dict, arm: dict) -> dict:
    keys = sorted(set(base) | set(arm))
    shared = [k for k in keys if k in base and k in arm]
    changed = {k: {"base": base.get(k), "arm": arm.get(k)}
               for k in keys if base.get(k) != arm.get(k)}
    return {"n_keys_compared": len(keys), "n_shared_keys": len(shared),
            "n_equal": len(shared) - len(changed), "changed": changed}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--expect", default="max_speed_input",
                    help="the ONE dest that may differ")
    ap.add_argument("--ignore", nargs="*", default=[],
                    help="RUN-IDENTITY dests exempted from the diff (e.g. "
                         "`out`, which MUST differ because two arms cannot "
                         "write to one directory). ⛔ The default is EMPTY so "
                         "an exemption is always deliberate, and every "
                         "exempted key is NAMED in the report — a silent "
                         "allowance is how a second lever hides.")
    ap.add_argument("--base"), ap.add_argument("--arm")
    ap.add_argument("--argv-file")
    ap.add_argument("--report")
    a = ap.parse_args(argv)

    if a.argv_file:
        blob = json.loads(Path(a.argv_file).read_text(encoding="utf-8"))
        base_argv, arm_argv = list(blob["base"]), list(blob["arm"])
    elif a.base and a.arm:
        base_argv, arm_argv = shlex.split(a.base), shlex.split(a.arm)
    else:
        raise SystemExit("[single-lever] pass --argv-file, or --base and --arm")

    tr = _trainer()
    d = diff(parse(tr, base_argv), parse(tr, arm_argv))

    ignored = {k: d["changed"].pop(k) for k in list(d["changed"])
               if k in set(a.ignore)}
    # ⛔ an --ignore entry that did NOT differ is a stale exemption: it hides
    # nothing today and will silently absolve a real lever tomorrow.
    stale = sorted(set(a.ignore) - set(ignored))

    ok_vacuity = d["n_shared_keys"] >= MIN_SHARED_KEYS
    only_expected = set(d["changed"]) == {a.expect}
    # ⭐ the POSITIVE half: the lever must actually have MOVED. "no differences"
    # would satisfy "no unexpected differences" and is the vacuous pass.
    ch = d["changed"].get(a.expect)
    moved = bool(ch) and ch["base"] != ch["arm"] and ch["arm"] is True

    rep = {"instrument": "assert_single_lever.py", "expect": a.expect,
           "base_argv": base_argv, "arm_argv": arm_argv, **d,
           "EXEMPTED_run_identity_keys": ignored,
           "STALE_exemptions_that_did_not_differ": stale,
           "CHECK_not_vacuous": ok_vacuity,
           "CHECK_only_the_expected_key_changed": only_expected,
           "CHECK_the_lever_actually_moved": moved,
           "CHECK_no_stale_exemptions": not stale,
           "VERDICT_single_lever": (ok_vacuity and only_expected and moved
                                    and not stale)}
    txt = json.dumps(rep, indent=1)
    print(txt)
    if a.report:
        Path(a.report).parent.mkdir(parents=True, exist_ok=True)
        Path(a.report).write_text(txt, encoding="utf-8")
    if not rep["VERDICT_single_lever"]:
        print(f"\nFAIL — changed keys: {sorted(d['changed'])}")
        return 1
    print(f"\nPASS — exactly one parsed key moved: {a.expect} "
          f"{ch['base']!r} -> {ch['arm']!r}  "
          f"({d['n_equal']} of {d['n_shared_keys']} keys identical)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
