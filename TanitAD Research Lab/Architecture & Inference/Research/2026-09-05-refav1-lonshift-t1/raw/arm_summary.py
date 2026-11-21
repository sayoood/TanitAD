"""Per-arm FOUR-FAMILY absolute values + each arm's own `cl - ha0_ext` paired
delta, read straight out of the banked `rec_<arm>.json`.

⭐ WHY BOTH ROUTES. `refav1_paired_delta.py` recomputes the floor comparison
ACROSS dumps; each arm's own record already carries `paired_cl_minus_ha0ext`
computed INSIDE the arm run. They are independent computations of the same
quantity, so agreement is a genuine cross-check and disagreement is a defect
worth finding -- not two views of one number.

⛔ ADE alone is INCOMPLETE. Every family is printed separately, never pooled.

Usage:  arm_summary.py <outdir> <arm> [<arm> ...]
"""
import json
import os
import sys

FAM_ORDER = ["ADE", "longitudinal", "lateral", "tactical", "strategic"]


def fam_rows(block):
    """-> list of (family, metric, dict) in a stable order."""
    rows = []
    fams = block.get("families", block) or {}
    for fam in FAM_ORDER + [f for f in fams if f not in FAM_ORDER]:
        mets = fams.get(fam)
        if not isinstance(mets, dict):
            continue
        for mk, v in mets.items():
            rows.append((fam, mk, v))
    return rows


def main(argv):
    out = argv[0]
    arms = argv[1:]

    print("# Per-arm FOUR-FAMILY panel, read from each arm's own rec_<arm>.json")
    print()
    for a in arms:
        p = os.path.join(out, f"rec_{a}.json")
        if not os.path.exists(p) or os.path.getsize(p) == 0:
            print(f"## {a} -- RECORD ABSENT ({p})")
            print()
            continue
        d = json.load(open(p, encoding="utf-8"))
        r = d.get("refav1", {})
        print(f"## {a}")
        print(f"* arm id `{d.get('arm')}`  ckpt `{d.get('ckpt')}`")
        cost = d.get("cost", {})
        print(f"* cost metric `{cost.get('metric')}`  weights `{cost.get('weights')}`"
              f"  source `{cost.get('weights_source')}`")
        print(f"* n_windows {r.get('n_windows')}  n_episodes {r.get('n_episodes')}")
        pl = r.get("planner", {})
        if pl:
            print(f"* planner {json.dumps(pl)[:400]}")
        tp = r.get("trivial_profile")
        if tp:
            print(f"* trivial_profile {json.dumps(tp)[:400]}")
        dk = r.get("distance_keeping")
        if dk:
            s = json.dumps(dk)
            print(f"* distance_keeping {s[:260]}")
        st = r.get("strategic")
        if st:
            print(f"* strategic {json.dumps(st)[:220]}")
        print()

        # ---- absolute four-family values per sub-arm -------------------------
        for sub in ("cl", "ha", "ha0", "ha0_ext", "ol"):
            ff = ((d.get("arms", {}).get(sub) or {}).get("four_families") or {})
            rows = fam_rows(ff)
            if not rows:
                continue
            print(f"### {a}.{sub} — absolute four-family values")
            print("| family | metric | value |")
            print("|---|---|---|")
            for fam, mk, v in rows:
                if isinstance(v, dict):
                    val = v.get("value", v.get("mean", v.get("delta")))
                    if val is None and v.get("status"):
                        val = f"{v.get('status')}: {str(v.get('reason'))[:80]}"
                else:
                    val = v
                if isinstance(val, float):
                    print(f"| {fam} | `{mk}` | {val:.4f} |")
                else:
                    print(f"| {fam} | `{mk}` | {val} |")
            print()

        # ---- the arm's OWN paired deltas vs the floors -----------------------
        for pk, label in (("paired_cl_minus_ha0ext", "cl - ha0_ext  (THE FLOOR)"),
                          ("paired_cl_minus_ha0", "cl - ha0"),
                          ("paired_cl_minus_ha", "cl - ha"),
                          ("paired_closed_minus_open", "cl - ol  (T1 minus T0)")):
            blk = r.get("families_paired", {}).get(pk)
            if not blk:
                continue
            print(f"### {a}: {label}   [{blk.get('estimator')}]")
            print("| family | metric | delta | 95% CI | separated | n_win | n_ep |")
            print("|---|---|---|---|---|---|---|")
            for fam, mk, v in fam_rows(blk):
                if not isinstance(v, dict):
                    continue
                if v.get("status"):
                    print(f"| {fam} | `{mk}` | {v['status']} | {str(v.get('reason'))[:60]} | - | - | - |")
                    continue
                print(f"| {fam} | `{mk}` | {v.get('delta'):+.4f} "
                      f"| [{v.get('lo'):+.4f}, {v.get('hi'):+.4f}] "
                      f"| {'**yes**' if v.get('separated') else 'no'} "
                      f"| {v.get('n_windows')} | {v.get('n_episodes')} |")
            print()


if __name__ == "__main__":
    main(sys.argv[1:])
