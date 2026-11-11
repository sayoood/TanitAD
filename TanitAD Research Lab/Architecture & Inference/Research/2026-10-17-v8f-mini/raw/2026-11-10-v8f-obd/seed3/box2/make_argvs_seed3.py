"""Derives the four OBD seed-3 argvs from the banked ones (theory/obd/argvs_obd.json), TOKEN FOR TOKEN: only `--seed` and `--out` change
(AMEND_OBD_SEED3.md: "argv = the banked arm's argv with ONLY --seed and --out changed"). Nothing else is touched, `--dagger-seed` included (it stays the init's index).

    OBD-CTL_s10  <- OBD-CTL_s0   --seed 10   init M2b s0
    OBD-BAND_s10 <- OBD-BAND_s0  --seed 10   init M2b s0
    OBD-CTL_s11  <- OBD-CTL_s1   --seed 11   init M2b s1
    OBD-BAND_s11 <- OBD-BAND_s1  --seed 11   init M2b s1

`--out` goes under the same runs root as the source (D:\\tanitad\\runs\\v8f_obd\\<tag>), with the new tag as the last component.
    python -I make_argvs_seed3.py [--out FILE]      (default FILE = frozen/argvs_obd_seed3.json next to this script)
Prints the sha256 of the written file; obd3_check.py pins it. The source file is pinned here too (SOURCE_SHA256)."""
import hashlib
import json
import ntpath
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.normpath(os.path.join(HERE, "..", "..", "argvs_obd.json"))
SOURCE_SHA256 = "346a48ed7b070fffacd22bf280e72c42831126fb906c231a7ac24301e624a424"
#: new tag -> (source tag, the new --seed)
DERIVE = {
    "OBD-CTL_s10": ("OBD-CTL_s0", "10"),
    "OBD-BAND_s10": ("OBD-BAND_s0", "10"),
    "OBD-CTL_s11": ("OBD-CTL_s1", "11"),
    "OBD-BAND_s11": ("OBD-BAND_s1", "11"),
}


def derive(source):
    """-> {new tag: {"what", "init_seed", "source_tag", "argv"}} in the order of DERIVE"""
    out = {}
    for tag, (src, seed) in DERIVE.items():
        rec = source[src]
        argv = list(rec["argv"])
        i, j = argv.index("--seed") + 1, argv.index("--out") + 1
        old_out = argv[j]
        head, last = ntpath.split(old_out)
        assert last == src, f"{src}: --out ends in {last!r}, not the source tag"
        argv[i], argv[j] = seed, ntpath.join(head, tag)
        out[tag] = {"what": f"{rec['what']}; seed {seed} (third-seed replicate of {src}: only --seed and --out differ)", "init_seed": rec["init_seed"], "source_tag": src, "argv": argv}
    return out


def render(derived):
    return json.dumps(derived, indent=1) + "\n"


def main(argv):
    out = os.path.join(HERE, "frozen", "argvs_obd_seed3.json")
    if "--out" in argv:
        out = argv[argv.index("--out") + 1]
    raw = open(SOURCE, "rb").read()
    got = hashlib.sha256(raw).hexdigest()
    if got != SOURCE_SHA256:
        print(f"REFUSED: {SOURCE} sha256 {got} != the pinned {SOURCE_SHA256}")
        return 2
    text = render(derive(json.loads(raw.decode("utf-8"))))
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print(hashlib.sha256(text.encode("utf-8")).hexdigest(), out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
