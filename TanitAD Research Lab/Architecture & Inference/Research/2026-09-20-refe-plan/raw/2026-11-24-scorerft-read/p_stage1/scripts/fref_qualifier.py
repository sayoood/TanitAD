"""MM 2026-11-25: the F_REFe qualifier printed beside P's read (C1r card PREREG_C1R_READ.md section 4(b), sha256 af358db9...).

The C1r card commits: frozen cards (SCORER-FT T and P, field F) get F_REFe printed next to their result as a DESCRIPTIVE qualifier,
and their committed bars do not move. This script prints that line and decides nothing. P's bars (P1-P3, G1) are scorerft_read.py's.

Inputs:
  mirror   git mirror path
  tipsha   the read's tip commit (must contain the landed C1r read JSON)
  read_p   P's read JSON (read_p_stage1.json from scorerft_read.py --card pick)
  out      the text file to write
F_REFe comes from the LANDED C1r read JSON, checked by sha256, never from a typed number.
"""
import hashlib, json, subprocess, sys

PKR = "TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
C1R_JSON = PKR + "/raw/2026-11-25-c1r-read/read_c1r.json"
C1R_JSON_SHA256 = "69e9ffae6024c84ffe514c9acd0950900c8aeb005e3b392385fce3121033dafe"  # landed in e7834b36


def main(mirror, tipsha, read_p, out):
    blob = subprocess.run(["git", "-C", mirror, "show", f"{tipsha}:{C1R_JSON}"], capture_output=True, check=True).stdout
    sha = hashlib.sha256(blob).hexdigest()
    if sha != C1R_JSON_SHA256:
        raise SystemExit(f"C1r read JSON sha256 {sha[:12]} != the landed {C1R_JSON_SHA256[:12]}: qualifier not printed")
    c1r = json.loads(blob.decode("utf-8"))
    rd = c1r["reading"]
    if rd.get("reading") not in ("R-NULL", "R-SEP"):
        raise SystemExit(f"C1r reading {rd.get('reading')!r} is not R-NULL / R-SEP")
    f = float(rd["F_REFe"])
    p = json.load(open(read_p, encoding="utf-8"))
    t = p["P_minus_C"]["pdms"]["tracked"]
    pt, lo, hi = float(t["point"]), float(t["lo"]), float(t["hi"])
    a = abs(pt)
    stamp = p.get("STAMP", "")
    lines = [
        f"F_REFe QUALIFIER (descriptive; C1r card section 4(b); P's committed bars do not move): "
        f"F_REFe = {f:.4f} (C1r read {rd['reading']}, ONE draw, 1 df, C1 recipe seed 0 vs 1, NOT C's seed pair); "
        f"P - C tracked PDMS = {pt:+.4f} [{lo:+.4f}, {hi:+.4f}]; |P - C| {'>' if a > f else '<='} F_REFe; "
        f"{'>' if a > 2 * f else '<='} 2*F_REFe = {2 * f:.4f}. P3 (d > 2F, F = |Cs - C|) is unchanged: it needs Cs and Ps.",
        "Family cells in this read are single-seed: on REFe, seed alone separates 21 of 70 family components "
        "(C1r read, H-ESTIM-SEED-1-REFE), so a separated family cell here is not a lever effect without a replicate.",
        f"source: {C1R_JSON} sha256 {sha}; P read stamp: {stamp}",
    ]
    open(out, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print(lines[0])


if __name__ == "__main__":
    if len(sys.argv) != 5:
        raise SystemExit(__doc__)
    main(*sys.argv[1:])
