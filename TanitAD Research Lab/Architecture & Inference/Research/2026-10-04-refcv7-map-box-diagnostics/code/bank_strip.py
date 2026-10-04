"""Make bankable copies of the per-pass binaries: packs keep sha12 and drop the integer episode id `ep`
(a derived per-clip id; sha12 is the only clip key banked outputs carry). Accumulators are copied as they are
(they hold sha12 lists only). Usage: python bank_strip.py <out dir> <bank dir> <tag> [<tag> ...]"""
import pickle
import shutil
import sys
from pathlib import Path


def main(src, dst, *tags):
    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    for t in tags:
        p = src / f"{t}.packs.pkl"
        if p.exists():
            packs = pickle.load(open(p, "rb"))
            for h in packs:
                for pk in packs[h]:
                    pk.pop("ep", None)
            with open(dst / f"{t}.packs.pkl", "wb") as fh:
                pickle.dump(packs, fh, protocol=4)
        a = src / f"{t}.acc.pt"
        if a.exists():
            shutil.copy2(a, dst / a.name)
        print(f"[bank] {t}: packs {'yes' if p.exists() else 'no'}, acc {'yes' if a.exists() else 'no'}")


if __name__ == "__main__":
    main(*sys.argv[1:])
