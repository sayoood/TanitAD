"""n6_kv_compare.py A.json B.json -- known-value check of note N6 of card ADD45: the result file read through the wrapper of note N6 (B) equals the one read under note N5 (A) leaf for leaf,
apart from the fields the new wrapper adds (d4_note_n6; in the d4 details pass_under_note_n5, note_n6_other_clauses, note_n6_start_clause) and the run time. Prints counts only, no value of a result.
Exit code 0 when equal, 1 when not."""
import json
import sys

ADDED = {"d4_note_n6", "pass_under_note_n5", "note_n6_other_clauses", "note_n6_start_clause", "elapsed_s"}


def leaves(o, path=()):
    if isinstance(o, dict):
        for k in sorted(o):
            if k in ADDED:
                continue
            yield from leaves(o[k], path + (str(k),))
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from leaves(v, path + (str(i),))
    else:
        yield path, o


def main(a, b):
    A = dict(leaves(json.load(open(a, encoding="utf-8"))))
    B = dict(leaves(json.load(open(b, encoding="utf-8"))))
    only_a = sorted(set(A) - set(B))
    only_b = sorted(set(B) - set(A))
    diff = [p for p in A if p in B and A[p] != B[p] and not (isinstance(A[p], float) and isinstance(B[p], float) and A[p] != A[p] and B[p] != B[p])]
    same = not only_a and not only_b and not diff
    print("KV leaves A %d B %d only_in_A %d only_in_B %d different %d" % (len(A), len(B), len(only_a), len(only_b), len(diff)))
    for p in (only_a + only_b + diff)[:12]:
        print("   path %s" % "/".join(p))
    print("ZZ N6-KV %s" % ("SAME" if same else "DIFFERENT"))
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
