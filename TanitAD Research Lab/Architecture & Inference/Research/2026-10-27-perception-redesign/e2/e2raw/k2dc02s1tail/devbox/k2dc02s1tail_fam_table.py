"""k2dc02s1tail_fam_table.py <E2_VERDICT.json> -- print the four-family K2 - K0 paired rows per seed as a markdown table (read-only)."""
import json
import sys


def main(argv):
    d = json.load(open(argv[1], encoding="utf-8"))
    ff = d["four_families"]
    print("| seed | family | row | K2DC02s1 - K0 [lo, hi] | better | separated worse | n (ep) |")
    print("|---|---|---|---|---|---|---|")
    for s in sorted(ff):
        for fam, b in ff[s].items():
            if not isinstance(b, dict) or "rows" not in b:
                print(f"| {s} | {fam} | - | {str(b.get('status') if isinstance(b, dict) else b)[:60]} | | | |")
                continue
            for r, v in b["rows"].items():
                if not isinstance(v, dict) or "delta" not in v:
                    print(f"| {s} | {fam} | {r} | {str(v)[:60]} | | | |")
                    continue
                print(f"| {s} | {fam} | {r} | {v['delta']:+.4f} [{v['lo']:+.4f}, {v['hi']:+.4f}] | {v.get('better', '')} | "
                      f"{'**YES**' if v.get('separated_worse') else 'no'} | {v.get('n')} ({v.get('n_episodes')}) |")
            print(f"| {s} | {fam} | family status | {b.get('status')} | | | |")


if __name__ == "__main__":
    main(sys.argv)
