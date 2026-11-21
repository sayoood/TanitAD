"""SLOWWM beside B2 (WMACT audit) runner, one spec per arm like ``read/ia0_slowwm.py``:
  python wmact_run.py [--smoke] [--force] [--limit N] [--no-ubasis-crosscheck] [--out DIR] "<arm>|<run_dir>|<basis_npz>|<ubasis_npz>" ...
  -> <out>/wmact_<arm>.json (resumable per arm; ``smoke_`` prefix with --smoke) and <out>/WMACT_B2.json (the L1 / L2 / L3 SKILL and ACTION-USE points of every arm read so far). Default <out> = $BESIDE_OUT.
A non-smoke row is written only when ``CONTROLS_B1B2.json`` says all_pass. Tier T0, BESIDE (no bar)."""
import json
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import wmact_arm as A  # noqa: E402
import torch  # noqa: E402

DL = A.DL


def main(argv):
    smoke, force, nocheck = "--smoke" in argv, "--force" in argv, "--no-ubasis-crosscheck" in argv
    opts, rest, it = {}, [], iter(argv)
    for a in it:
        if a in ("--limit", "--out"):
            opts[a] = next(it)
        elif not a.startswith("--"):
            rest.append(a)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    out_dir = opts.get("--out", DL.OUT_DIR)
    os.makedirs(out_dir, exist_ok=True)
    if not smoke and not DL.controls_passed():
        print("REFUSED: CONTROLS_B1B2.json does not say all_pass: no arm row is written", flush=True)
        return 2
    rc = 0
    for spec in rest:
        f = spec.split("|")
        name, run_dir = f[0], f[1]
        basis, ub = (f[2] if len(f) > 2 else ""), (f[3] if len(f) > 3 else "")
        out = os.path.join(out_dir, ("smoke_" if smoke else "") + f"wmact_{name}.json")
        if os.path.exists(out) and not force:
            print(f"[{name}] done already, skipped", flush=True)
            continue
        try:
            r = A.read_arm_wmact(name, run_dir, basis or None, ub or None, int(opts["--limit"]) if "--limit" in opts else None, not nocheck)
            r["smoke"] = bool(smoke)
            r["controls_file_all_pass"] = DL.controls_passed()
            json.dump(r, open(out + ".tmp", "w", encoding="utf-8"), default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o), indent=1)
            os.replace(out + ".tmp", out)
            print(f"[{name}] B2 read in {r['wall_s']} s -> {out}", flush=True)
        except BaseException as e:
            import traceback
            traceback.print_exc()
            open(os.path.join(out_dir, f"FAILED_wmact_{name}.txt"), "w", encoding="utf-8").write(f"{type(e).__name__}: {str(e)[:2000]}\n")
            rc = 1
    summ = {}
    for fn in sorted(os.listdir(out_dir)):
        if fn.startswith("wmact_") and fn.endswith(".json"):
            z = json.load(open(os.path.join(out_dir, fn), encoding="utf-8"))
            summ[z["name"]] = {"tier": "T0", "kind": z["route"]["kind"], "rows": {k: {s: v[s]["point"] for s in ("SKILL", "ACTION-USE", "ACTION-USE-MEAN")} for k, v in z["rows"].items()}}
    json.dump({"b2_summary": summ, "note": "BESIDE, no bar; each arm in its own z1 space"}, open(os.path.join(out_dir, "WMACT_B2.json"), "w", encoding="utf-8"), indent=1)
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
