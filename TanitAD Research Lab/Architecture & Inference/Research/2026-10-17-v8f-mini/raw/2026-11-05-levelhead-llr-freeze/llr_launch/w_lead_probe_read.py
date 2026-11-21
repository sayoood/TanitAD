"""Read the pulled W_lead-probe selfcheck.json and write a2_prereg/WLEAD_PROBE.json (the number W_lead comes from). stdlib only, no network.
  python -I w_lead_probe_read.py <pulled folder> [--out <WLEAD_PROBE.json>] [--frozen <FROZEN_LLR.txt>]
<pulled folder> holds selfcheck.json and WLPROBE.done (both pulled from box 2 by pull_wleadprobe.sh). Refuses (exit 1, writes nothing) unless: WLPROBE.done's selfcheck sha256 is the pulled file's, declared_vs_built PASS, the init ckpt is DAG-OACC s0's,
lead_head_probe.pass is True, w_lead_declared is 1.0, lead_head_probe.aux_cond is the FROZEN wiring (FROZEN_LLR.txt WIRING; the probe is only meaningful under the wiring the arm will use), the lead labels cover >= 50 % of the TRAIN clips, and
lead_head_probe.w_lead_for_10pct_share is a finite number > 0 that equals (0.1/0.9) x step0_total_without_lead / step0_lead_term."""
import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

H = Path(__file__).resolve().parent
INIT_S0 = "67265ec1666fae3a6d51e924ee06dfb68b2cd0630afc62e42f3a46182b8bc45f"
KEY = "w_lead_for_10pct_share"


def _fz():
    sp = importlib.util.spec_from_file_location("llr_frozen", H / "llr_frozen.py")
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m


def read(folder, frozen_path):
    d = Path(folder)
    sc_p, mk_p = d / "selfcheck.json", d / "WLPROBE.done"
    bad = []
    fz, fbad = _fz().parse_frozen(frozen_path)
    if fbad or "WIRING" not in fz:
        return None, [f"{frozen_path}: the wiring is not frozen ({'; '.join(fbad) or 'no WIRING line'}): the probe is read under the frozen wiring only"]
    if not sc_p.exists():
        return None, ["selfcheck.json was not pulled"]
    if not mk_p.exists():
        return None, ["WLPROBE.done was not pulled (the probe did not finish: see WLPROBE.FAILED / logs)"]
    sc = json.loads(sc_p.read_text(encoding="utf-8")); mk = json.loads(mk_p.read_text(encoding="utf-8"))
    got = hashlib.sha256(sc_p.read_bytes()).hexdigest()
    if (mk.get("evidence") or {}).get("selfcheck_sha256") != got:
        bad.append(f"the pulled selfcheck.json sha256 {got[:12]} is not the one WLPROBE.done recorded {str((mk.get('evidence') or {}).get('selfcheck_sha256'))[:12]}")
    if sc.get("declared_vs_built") != "PASS":
        bad.append(f"declared_vs_built = {sc.get('declared_vs_built')!r}")
    if ((sc.get("info") or {}).get("init_ckpt") or {}).get("sha256") != INIT_S0:
        bad.append("the init ckpt is not DAG-OACC s0's")
    lp = sc.get("lead_head_probe")
    if not isinstance(lp, dict) or KEY not in lp:
        return None, bad + [f"selfcheck.json has no lead_head_probe.{KEY}"]
    v = lp[KEY]
    if not (isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v > 0):
        bad.append(f"lead_head_probe.{KEY} = {v!r} is not a finite number > 0")
    if lp.get("pass") is not True:
        bad.append(f"lead_head_probe.pass = {lp.get('pass')!r}")
    if lp.get("w_lead_declared") != 1.0:
        bad.append(f"the probe declared w_lead {lp.get('w_lead_declared')!r}, not 1.0")
    if lp.get("aux_cond") != fz["WIRING"] or (sc.get("levers") or {}).get("aux_cond") != fz["WIRING"]:
        bad.append(f"the probe ran under wiring {lp.get('aux_cond')!r} / built {(sc.get('levers') or {}).get('aux_cond')!r}, the frozen wiring is {fz['WIRING']!r}")
    ll = (sc.get("info") or {}).get("lead_labels") or {}
    if not (ll.get("n_clips") and ll.get("n_clips_with_labels") is not None and ll["n_clips_with_labels"] >= 0.5 * ll["n_clips"]):
        bad.append(f"the lead labels cover {ll.get('n_clips_with_labels')!r} of {ll.get('n_clips')!r} train clips (< 50 %)")
    try:
        want = (0.1 / 0.9) * lp["step0_total_without_lead"] / lp["step0_lead_term"]
        if abs(want - v) > 1e-6 * max(abs(want), 1.0):
            bad.append(f"{KEY} {v} != (0.1/0.9) x step0_total_without_lead / step0_lead_term = {want}")
    except (KeyError, TypeError, ZeroDivisionError):
        bad.append("the step-0 record is incomplete")
    if bad:
        return None, bad
    rec = {KEY: v, "aux_cond": fz["WIRING"], "step0_total": lp.get("step0_total"), "step0_lead_term": lp.get("step0_lead_term"), "step0_total_without_lead": lp.get("step0_total_without_lead"),
           "step0_level_share": lp.get("step0_level_share"), "lead_labels": ll,
           "formula": "W_lead such that W_lead x L_lead = 0.1 x (L_without_lead + W_lead x L_lead) at step 0 of the real-data smoke on the C0 init with both heads on (the level term is part of L_without_lead): W_lead = (0.1/0.9) x L_without_lead / L_lead",
           "probe_argv_sha": sc.get("argv_sha"), "selfcheck_sha256": got, "selfcheck_path": str(sc_p), "wlprobe_done_sha256": hashlib.sha256(mk_p.read_bytes()).hexdigest(),
           "init_ckpt_sha256": INIT_S0, "read_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    return rec, []


def main(a):
    if not a:
        print(__doc__); return 2
    out = Path(a[a.index("--out") + 1]) if "--out" in a else H.parent / "a2_prereg" / "WLEAD_PROBE.json"
    frozen = Path(a[a.index("--frozen") + 1]) if "--frozen" in a else H.parent / "a2_prereg" / "FROZEN_LLR.txt"
    rec, bad = read(a[0], frozen)
    if bad:
        print("ZZWLPROBE-REFUSED " + " | ".join(bad)); return 1
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"WLEAD_PROBE.json written: {KEY} = {rec[KEY]} (wiring {rec['aux_cond']}) -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
