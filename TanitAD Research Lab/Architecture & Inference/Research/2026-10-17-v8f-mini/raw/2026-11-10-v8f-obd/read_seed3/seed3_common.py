"""seed3_common.py -- the four seed-3 arms of AMEND_OBD_SEED3.md (sha256 a8a5848f22a3f164e4069bd7b302e7201cb05de5995494d05cf92dfde426e314) and the argv check shared by the dump driver and the reader.

Pure stdlib (no numpy, no torch).  Arms (argv = the banked arm's argv with ONLY --seed and --out changed):
  OBD-CTL_s10  <- OBD-CTL_s0  --seed 10  (init M2b s0)      OBD-BAND_s10 <- OBD-BAND_s0 --seed 10  (init M2b s0)
  OBD-CTL_s11  <- OBD-CTL_s1  --seed 11  (init M2b s1)      OBD-BAND_s11 <- OBD-BAND_s1 --seed 11  (init M2b s1)
"""
import os

AMEND_SHA256 = "a8a5848f22a3f164e4069bd7b302e7201cb05de5995494d05cf92dfde426e314"
# new tag: (base banked tag, --seed, init seed of M2b)
NEW = {
    "OBD-CTL_s10": ("OBD-CTL_s0", 10, 0),
    "OBD-BAND_s10": ("OBD-BAND_s0", 10, 0),
    "OBD-CTL_s11": ("OBD-CTL_s1", 11, 1),
    "OBD-BAND_s11": ("OBD-BAND_s1", 11, 1),
}
NEW_TAGS = tuple(NEW)                      # the order of the amendment's table
SEED_LABELS = {"s0": 0, "s1": 1, "s10": 0, "s11": 1}      # per-seed label -> init (M2b s0 / s1)


def init_of_tag(tag):
    """init seed (0 or 1) of an OBD tag: the new arms by table, the banked ones by their suffix (s1 -> 1; s0 and the replicates -> 0).  NOT endswith('_s1') alone: '_s11' is init 1 too."""
    if tag in NEW:
        return NEW[tag][2]
    return 1 if tag.endswith("_s1") else 0


def argv_tokens(text):
    return text.split()


def _drop(tokens, flags):
    out, i = [], 0
    while i < len(tokens):
        if tokens[i] in flags:
            i += 2
            continue
        out.append(tokens[i])
        i += 1
    return out


def argv_problem(pull, tag):
    """None when <pull>/obd_<tag>/argv.txt equals <pull>/obd_<base>/argv.txt except for --seed (== the amendment's value) and --out; else a reason string."""
    base, seed, _ = NEW[tag]
    pa, pb = f"{pull}/obd_{tag}/argv.txt", f"{pull}/obd_{base}/argv.txt"
    for p in (pa, pb):
        if not os.path.isfile(p):
            return f"argv.txt missing ({p})"
    a, b = argv_tokens(open(pa, encoding="utf-8").read()), argv_tokens(open(pb, encoding="utf-8").read())
    if "--seed" not in a or a[a.index("--seed") + 1] != str(seed):
        return f"{tag}: argv does not carry --seed {seed}"
    if _drop(a, ("--seed", "--out")) != _drop(b, ("--seed", "--out")):
        return f"{tag}: argv differs from {base}'s in more than --seed and --out"
    return None
