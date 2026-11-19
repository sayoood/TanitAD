"""dscale_lead_read_n4: thin wrapper around the unchanged dscale_lead_read.main with the k2 tolerance as a parameter.
  python dscale_lead_read_n4.py --k2-tol 1e-4 <every other argument of dscale_lead_read.py>
Default 1e-4 = the control as frozen. Sets TOL_K2 of the module in which control_k2 is defined (control_k2.__globals__; leadsup_read_n1 on the pod chain), runs the reader, then records the value used
(top-level field k2_tol_note_n4 in RESULT_DSCALE_LEAD.json and one line at the end of VERDICT_DSCALE_LEAD.md). Nothing else changes."""
import json
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dscale_lead_read as D  # noqa: E402


def _argv_pop(argv, name):
    for i, x in enumerate(argv):
        if x == name and i + 1 < len(argv):
            v = argv[i + 1]
            return v, argv[:i] + argv[i + 2:]
        if x.startswith(name + "="):
            return x.split("=", 1)[1], argv[:i] + argv[i + 1:]
    return None, argv


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    v, rest = _argv_pop(argv, "--k2-tol")
    tol = 1e-4 if v is None else float(v)
    mod = D.N.control_k2.__globals__
    assert "TOL_K2" in mod and "control_k2" in mod, "TOL_K2 not found in the module of control_k2"
    mod["TOL_K2"] = tol
    rc = D.main(rest)
    out_dir, _ = _argv_pop(rest, "--out-dir")
    if out_dir:
        jp, mp = out_dir + "/" + D.OUT_JSON, out_dir + "/" + D.OUT_MD
        if os.path.exists(jp):
            r = json.load(open(jp, encoding="utf-8"))
            r["k2_tol_note_n4"] = tol
            json.dump(r, open(jp, "w", encoding="utf-8"), indent=1)
        if os.path.exists(mp):
            with open(mp, "a", encoding="utf-8") as f:
                f.write("\nk2 tolerance used by the n4 wrapper: %r (frozen default 1e-4)\n" % tol)
    return rc


if __name__ == "__main__":
    sys.exit(main())
