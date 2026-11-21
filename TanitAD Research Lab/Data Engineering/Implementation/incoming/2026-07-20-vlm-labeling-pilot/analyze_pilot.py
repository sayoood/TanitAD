#!/usr/bin/env python3
"""Post-process the VLM pilot sidecars.

1. Mint the AUTHORITATIVE VTARGET kinematically (85th-pct free-flow speed -> frozen band),
   per V3_GOAL_VOCABULARY_V1 label-minting ("VTARGET: kinematic + vlm cap").
2. Snap the VLM's free-form band back onto the grid to recover its intent (diagnostic only).
3. Score schema adherence per slot (strict vs effective) and scene tags vs the
   render-condition ground truth carried in the clip filename.
"""
import glob, json, os, re, sys
import numpy as np

OUT = sys.argv[1] if len(sys.argv) > 1 else "/root/vlm_pilot/out/reason1"

def vtarget_tokens():
    t = ["v_stop"] + ["v(%d-%d]" % (i, i + 1) for i in range(0, 10)]
    lo = 10.0
    while lo < 40.0:
        hi = lo + 2.5
        t.append("v(%g-%g]" % (lo, hi))
        lo = hi
    return t

VT = vtarget_tokens()

def band_for(v):
    """Snap a speed in m/s onto the frozen non-uniform VTARGET grid."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "unknown"
    if v <= 0.3:
        return "v_stop"
    if v <= 10.0:
        k = int(np.ceil(v)) - 1
        k = max(0, min(9, k))
        return "v(%d-%d]" % (k, k + 1)
    if v > 40.0:
        return "v(37.5-40]"
    lo = 10.0
    while lo < 40.0:
        hi = lo + 2.5
        if v <= hi:
            return "v(%g-%g]" % (lo, hi)
        lo = hi
    return "v(37.5-40]"

RANGE_RE = re.compile(r"v\(\s*([0-9.]+)\s*-\s*([0-9.]+)\s*[\])]")

def snap_emitted(s):
    """Recover the VLM's numeric intent from a fabricated band string."""
    if not s:
        return None, None
    m = RANGE_RE.search(str(s))
    if not m:
        return None, None
    lo, hi = float(m.group(1)), float(m.group(2))
    mid = 0.5 * (lo + hi)
    return band_for(mid), mid

# weather/time render-condition -> expected scene tag values
W_GT = {"Rainy": "rain", "Snowy": "snow", "Foggy": "fog"}
T_GT = {"Night": "night", "Morning": "day", "Sunny": "day",
        "Golden_hour": ("dusk", "dawn")}

def main():
    files = sorted(glob.glob(os.path.join(OUT, "*.json")))
    print("sidecars: %d  (%s)" % (len(files), OUT))
    if not files:
        return
    SLOTS = ["VTARGET", "VSOURCE", "LONMODE", "LATMANEUVER", "HEADWAY", "DYN",
             "TACPOINT", "LIGHTSTATE", "INTERACT", "RISK"]
    present = {s: 0 for s in SLOTS}
    strict = {s: 0 for s in SLOTS}
    viol = {s: 0 for s in SLOTS}
    unk = {s: 0 for s in SLOTS}
    parse_ok = {"a": 0, "b": 0}
    err = 0
    secs, tin, tout = [], [], []
    w_hit = w_tot = t_hit = t_tot = 0
    snap_ok = snap_tot = 0
    vt_agree = vt_tot = 0
    recs = []

    for f in files:
        d = json.load(open(f))
        if "error" in d:
            err += 1
        if d.get("pass_a_parse") in ("clean", "braces", "repaired"):
            parse_ok["a"] += 1
        if d.get("pass_b_parse") in ("clean", "braces", "repaired"):
            parse_ok["b"] += 1
        tm = d.get("timing", {})
        secs.append(tm.get("seconds", 0)); tin.append(tm.get("tokens_in", 0)); tout.append(tm.get("tokens_out", 0))

        vmap = {v["slot"]: v["emitted"] for v in d.get("violations", [])}
        nmap = {n["slot"] for n in d.get("normalized", [])}
        goal = d.get("goal_tactical", {}) or {}
        for s in SLOTS:
            if s in goal or s in vmap:
                present[s] += 1
            if s in vmap:
                viol[s] += 1
            elif s in goal:
                if goal[s] == "unknown":
                    unk[s] += 1
                elif s not in nmap:
                    strict[s] += 1

        # --- authoritative kinematic VTARGET
        kin = d.get("kinematics") or {}
        sp = kin.get("speed_mps")
        vt_kin = "unknown"
        if sp:
            v85 = float(np.percentile(np.array(sp, dtype=float), 85))
            vt_kin = band_for(v85)
            d["vtarget_kinematic"] = {"token": vt_kin, "v85_mps": round(v85, 2),
                                      "provenance": "kinematic"}
        # --- snap the VLM's emitted band
        emitted = vmap.get("VTARGET") or goal.get("VTARGET")
        snapped, mid = snap_emitted(emitted)
        if emitted and emitted != "unknown":
            snap_tot += 1
            if snapped:
                snap_ok += 1
                d["vtarget_vlm_snapped"] = {"token": snapped, "from": emitted,
                                            "midpoint_mps": round(mid, 2),
                                            "provenance": "vlm"}
        if snapped and vt_kin != "unknown":
            vt_tot += 1
            if snapped == vt_kin:
                vt_agree += 1

        # --- scene tag scoring vs render-condition GT
        gt = d.get("render_condition_gt")
        st = d.get("scene_tags", {}) or {}
        if gt in W_GT:
            w_tot += 1
            if st.get("weather") == W_GT[gt]:
                w_hit += 1
        if gt in T_GT:
            t_tot += 1
            exp = T_GT[gt]
            got = st.get("time_of_day")
            if (got == exp) if isinstance(exp, str) else (got in exp):
                t_hit += 1
        recs.append(d)
        with open(f, "w") as fh:
            json.dump(d, fh, indent=1, ensure_ascii=False)

    n = len(files)
    print("\n--- JSON parse ---")
    print("  pass A parsed: %d/%d (%.0f%%)   pass B parsed: %d/%d (%.0f%%)  errors: %d"
          % (parse_ok["a"], n, 100.0 * parse_ok["a"] / n, parse_ok["b"], n,
             100.0 * parse_ok["b"] / n, err))
    print("\n--- per-slot vocabulary adherence (of clips where slot emitted) ---")
    print("  %-12s %6s %8s %8s %8s" % ("slot", "n", "in-vocab", "unknown", "violation"))
    tot_p = tot_ok = tot_v = 0
    for s in SLOTS:
        p = present[s] or 1
        print("  %-12s %6d %7.0f%% %7.0f%% %7.0f%%" % (
            s, present[s], 100.0 * strict[s] / p, 100.0 * unk[s] / p, 100.0 * viol[s] / p))
        tot_p += present[s]; tot_ok += strict[s] + unk[s]; tot_v += viol[s]
    print("  %-12s %6d %7.0f%% in-vocab overall, %.0f%% violations"
          % ("ALL", tot_p, 100.0 * tot_ok / max(tot_p, 1), 100.0 * tot_v / max(tot_p, 1)))
    print("  ALL except VTARGET: %.1f%% in-vocab" % (
        100.0 * (tot_ok - strict["VTARGET"] - unk["VTARGET"]) /
        max(tot_p - present["VTARGET"], 1)))

    print("\n--- VTARGET recovery ---")
    print("  VLM free-form bands parseable back to grid: %d/%d (%.0f%%)"
          % (snap_ok, snap_tot, 100.0 * snap_ok / max(snap_tot, 1)))
    print("  snapped-VLM == kinematic 85th-pct band:     %d/%d (%.0f%%)"
          % (vt_agree, vt_tot, 100.0 * vt_agree / max(vt_tot, 1)))

    print("\n--- scene tags vs render-condition GT ---")
    print("  weather (Rainy/Snowy/Foggy): %d/%d (%.0f%%)" % (w_hit, w_tot, 100.0 * w_hit / max(w_tot, 1)))
    print("  time_of_day (Night/day/golden): %d/%d (%.0f%%)" % (t_hit, t_tot, 100.0 * t_hit / max(t_tot, 1)))

    print("\n--- throughput ---")
    tot_s = float(np.sum(secs))
    print("  mean %.1f s/clip (2 passes), total %.1f min -> %.0f clips/GPU-hr"
          % (np.mean(secs), tot_s / 60.0, 3600.0 / max(np.mean(secs), 1e-6)))
    print("  mean tokens in %.0f / out %.0f per clip" % (np.mean(tin), np.mean(tout)))

    import collections
    print("\n--- label distribution ---")
    for s in ["LONMODE", "LATMANEUVER", "VSOURCE", "RISK", "HEADWAY"]:
        c = collections.Counter((r.get("goal_tactical", {}) or {}).get(s, "-") for r in recs)
        print("  %-12s %s" % (s, dict(c.most_common(6))))
    lead = collections.Counter(bool((r.get("lead_state") or {}).get("present")) for r in recs)
    print("  lead_present %s" % dict(lead))
    nsign = sum(len(r.get("sign_reads") or []) for r in recs)
    print("  sign_reads total: %d across %d clips" % (nsign, n))


if __name__ == "__main__":
    main()
