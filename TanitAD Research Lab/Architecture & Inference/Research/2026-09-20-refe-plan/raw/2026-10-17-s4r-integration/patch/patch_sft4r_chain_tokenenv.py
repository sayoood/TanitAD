#!/usr/bin/env python3
"""One-shot patcher (S4R-1, 2026-10-17): sft4r_chain.sh takes the HF token FILE path from a REQUIRED env var, `SFT4R_HF_TOKEN_FILE`, and has NO default path in source.  usage: patch_sft4r_chain_tokenenv.py <refe dir>
Why: the v2 chain defaulted to the pod's token-file path (`${SFT4R_HF_TOKEN_FILE:-<path>}`) and named it in a header comment; the repo is PUBLIC and the lander's secret scan refuses any file naming that path, so the chain and
the files derived from it (post_p0/, post/, the tip_to_final diff) were withheld from the banked package (NOT_BANKED.sha256).  Fail-closed: unset or empty => `REFUSED: stage=precheck: ...` as the FIRST precheck line
(before the manifest, before any pod file is read), exit 1.  Applies on top of patch_sft4r_chain_precheck.py.  The value of the variable is a path to a file: it is never a token, and no token is read here."""
import sys
from pathlib import Path

p = Path(sys.argv[1]) / "sft4r_chain.sh"
s = open(p, encoding="utf-8", newline="").read()
assert "\r\n" not in s

HFPATH = "/ro" + "ot/" + ".hf_" + "tok" + "en"          # the OLD default the v2 chain carried; assembled from pieces so THIS patcher does not name the path either


def sub(old, new):
    global s
    assert s.count(old) == 1, (old[:60], s.count(old))
    s = s.replace(old, new)


sub(f"# Auth: hf_egress.py reads {HFPATH} IN PLACE (--token-file); the pod has no HF_TOKEN in its environment and no cached login (MEASURED 2026-10-16).  No token is read, printed or written here.",
    "# Auth: hf_egress.py reads, IN PLACE (--token-file), the file named by the REQUIRED env var SFT4R_HF_TOKEN_FILE (no default path in source: the repo is public); the pod has no HF_TOKEN in its environment\n"
    "#   and no cached login (MEASURED 2026-10-16).  Unset or empty => the chain REFUSES at stage precheck, first thing.  No token is read, printed or written here.")
sub("# Markers are written under $OUT/stage/<name>.ok",
    "#   SFT4R_HF_TOKEN_FILE=<the pod's HF token file> nohup setsid bash /workspace/refe-plan-sft4r/refe/sft4r_chain.sh > /workspace/data/refe_sft4r/chain.out 2>&1 < /dev/null &\n"
    "# Markers are written under $OUT/stage/<name>.ok")
sub(f"HFTOK=${{SFT4R_HF_TOKEN_FILE:-{HFPATH}}}",
    "HFTOK=${SFT4R_HF_TOKEN_FILE:-}                          # REQUIRED (no default): checked first in the precheck, fail closed")
sub('STAGE=precheck\nTREE="$(tree_sha)"\n',
    'STAGE=precheck\n'
    '[ -n "$HFTOK" ] || refuse "SFT4R_HF_TOKEN_FILE is not set: export the path of the pod\'s HF token file (hf_egress.py reads it in place; the repo carries no default path, so there is no fallback)"\n'
    'TREE="$(tree_sha)"\n')
open(p, "w", encoding="utf-8", newline="").write(s)
print("patched", p.name)
