"""Read the pulled W-probe selfcheck.json and write cl_prereg/W_PROBE.json (the number the card's W comes from). stdlib only, no network.
  python -I w_probe_read.py <pulled folder> [--out <W_PROBE.json>]
<pulled folder> holds selfcheck.json and WPROBE.done (both pulled from box 2 by pull_wprobe.sh). Refuses (exit 1, writes nothing) unless: WPROBE.done's selfcheck sha256 is the pulled file's, declared_vs_built PASS, the init ckpt is DAG-OACC s0's,
level_head_probe.pass is True, w_level_declared is 1.0, and level_head_probe.w_level_for_10pct_share is a finite number > 0 that equals (0.1/0.9) x step0_total_without_level / step0_level_term."""
import hashlib
import json
import math
import sys
import time
from pathlib import Path

H = Path(__file__).resolve().parent
INIT_S0 = "67265ec1666fae3a6d51e924ee06dfb68b2cd0630afc62e42f3a46182b8bc45f"
KEY = "w_level_for_10pct_share"


def read(folder):
    d = Path(folder)
    sc_p, mk_p = d / "selfcheck.json", d / "WPROBE.done"
    bad = []
    if not sc_p.exists():
        return None, ["selfcheck.json was not pulled"]
    if not mk_p.exists():
        return None, ["WPROBE.done was not pulled (the probe did not finish: see WPROBE.FAILED / logs)"]
    sc = json.loads(sc_p.read_text(encoding="utf-8")); mk = json.loads(mk_p.read_text(encoding="utf-8"))
    got = hashlib.sha256(sc_p.read_bytes()).hexdigest()
    if (mk.get("evidence") or {}).get("selfcheck_sha256") != got:
        bad.append(f"the pulled selfcheck.json sha256 {got[:12]} is not the one WPROBE.done recorded {str((mk.get('evidence') or {}).get('selfcheck_sha256'))[:12]}")
    if sc.get("declared_vs_built") != "PASS":
        bad.append(f"declared_vs_built = {sc.get('declared_vs_built')!r}")
    if ((sc.get("info") or {}).get("init_ckpt") or {}).get("sha256") != INIT_S0:
        bad.append("the init ckpt is not DAG-OACC s0's")
    lp = sc.get("level_head_probe")
    if not isinstance(lp, dict) or KEY not in lp:
        return None, bad + [f"selfcheck.json has no level_head_probe.{KEY}"]
    v = lp[KEY]
    if not (isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v > 0):
        bad.append(f"level_head_probe.{KEY} = {v!r} is not a finite number > 0")
    if lp.get("pass") is not True:
        bad.append(f"level_head_probe.pass = {lp.get('pass')!r}")
    if lp.get("w_level_declared") != 1.0:
        bad.append(f"the probe declared w_level {lp.get('w_level_declared')!r}, not 1.0")
    try:
        want = (0.1 / 0.9) * lp["step0_total_without_level"] / lp["step0_level_term"]
        if abs(want - v) > 1e-6 * max(abs(want), 1.0):
            bad.append(f"{KEY} {v} != (0.1/0.9) x step0_total_without_level / step0_level_term = {want}")
    except (KeyError, TypeError, ZeroDivisionError):
        bad.append("the step-0 record is incomplete")
    if bad:
        return None, bad
    rec = {KEY: v, "step0_total": lp.get("step0_total"), "step0_level_term": lp.get("step0_level_term"), "step0_total_without_level": lp.get("step0_total_without_level"),
           "formula": "W such that W x L_level = 0.1 x (L_without_level + W x L_level) at step 0 of the real-data smoke on the C0 init with the head on: W = (0.1/0.9) x L_without_level / L_level (the card's 'level term is a tenth of the total')",
           "probe_argv_sha": sc.get("argv_sha"), "selfcheck_sha256": got, "selfcheck_path": str(sc_p), "wprobe_done_sha256": hashlib.sha256(mk_p.read_bytes()).hexdigest(),
           "init_ckpt_sha256": INIT_S0, "read_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    return rec, []


def main(a):
    if not a:
        print(__doc__); return 2
    out = Path(a[a.index("--out") + 1]) if "--out" in a else H.parent / "cl_prereg" / "W_PROBE.json"
    rec, bad = read(a[0])
    if bad:
        print("ZZWPROBE-REFUSED " + " | ".join(bad)); return 1
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"W_PROBE.json written: {KEY} = {rec[KEY]} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
