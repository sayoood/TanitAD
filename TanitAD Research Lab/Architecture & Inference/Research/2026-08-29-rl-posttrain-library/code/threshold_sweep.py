"""⛔ DO OUR THRESHOLDS FLAG COMPETENT HUMAN DRIVING? A sweep over every constant.

P-RC21 Addendum 4 measured that `proximity_safe_m = 5.0` flags the human's own
future on 45.1% of windows. ⭐ Master Mind extension: that is not a reward detail,
it is a SHAPE OF ERROR, and our EVAL instruments carry constants of the same shape
-- round numbers, several of them self-labelled PROPOSED, none derived from this
corpus. If they flag the human at a material rate, then every "our arm violates X
on Y% of windows" number we have published inherits the miscalibration.

⚠️ THE MEASUREMENT IS DELIBERATELY NOT "is the constant right". It is: WHAT
FRACTION OF DEMONSTRATION WINDOWS DOES THIS CONSTANT FLAG? A threshold that flags
the demonstrations at a high rate is either (a) mis-calibrated, or (b) making a
claim that human driving is unsafe -- which we would then have to defend
explicitly. Either way it must be a decision, not an inherited default.

⭐ The human is the reference, NOT the ceiling: a threshold flagging ~5-15% of
demonstrations is doing its job (real driving contains real risk). One flagging
~50% is measuring the corpus, not the behaviour.

⚠️ SELECTION: every rate below is over ALL windows meeting the instrument's OWN
applicability gate (a lead exists / speed above its floor), NEVER over a subset
selected by the outcome being measured -- that is TRAIN-C8, caught tonight in this
same probe family. Each row prints its n.

Comfort/kinematic quantities are computed from the RAW 10 Hz poses over the same
2 s window, not from the 4-point downsample, so accel/jerk/yaw-rate are real.

Tier: T0. NON-PARITY corpus. Evidence class: MEASURED (ours).
"""
import importlib.util, sys, json, collections, io
import numpy as np

sys.path.insert(0, "stack")
_s = importlib.util.spec_from_file_location("pilot", "stack/scripts/rl_pilot_refc21.py")
P = importlib.util.module_from_spec(_s); _s.loader.exec_module(P)

O = r"C:/Users/Admin/tanitad-data/rl-pilot"
src = P.WindowSource(r"C:/Users/Admin/tanitad-data/physicalai/_epcache/physicalai-val-bb543bdf7836",
                     f"{O}/pilot_val_agents.jsonl", seed=0, lru=60)

raw = collections.defaultdict(dict)
for line in io.open(f"{O}/pilot_val_agents.jsonl", encoding="utf-8"):
    d = json.loads(line)
    raw[d["clip_id"]][d["frame_idx"]] = [(a["cx"], a["cy"], a.get("l", 4.5)) for a in d["agents"]]

# ---- the constants under test, each with its source ---------------------------
C = {
    "proximity_safe_m":       (5.0,  "rl/rewards.py:388"),
    "ttc_min_s":              (1.5,  "rl/rewards.py:329 (VETO — hard, not graded)"),
    "target_time_gap_s":      (2.0,  "rl/rewards.py:294 (T*, reward PEAK)"),
    "a_max_mps2":             (4.0,  "rl/rewards.py:75  (feasibility envelope)"),
    "kappa_max_1pm":          (0.2,  "rl/rewards.py:76"),
    "lat_acc_max_mps2":       (4.0,  "rl/rewards.py:77"),
    "jerk_max_mps3":          (8.0,  "rl/rewards.py:78 == pseudosim COMFORT_LIMITS"),
    "pdm_a_lon_max_mps2":     (3.0,  "pseudosim.py:803 COMFORT_LIMITS (PROPOSED)"),
    "pdm_a_lat_max_mps2":     (3.0,  "pseudosim.py:803 COMFORT_LIMITS (PROPOSED)"),
    "pdm_yaw_rate_max_radps": (0.95, "pseudosim.py:804 COMFORT_LIMITS (PROPOSED)"),
}
flag = {k: [] for k in C}
vals = {k: [] for k in C}
rng = np.random.default_rng(1234)
n_lead = n_all = 0

for stem in src.stems:
    ep = src._ep(stem); poses = ep["poses"].numpy(); T = int(poses.shape[0])
    lo, hi = P.WINDOW - 1, T - P.HORIZONS[-1] - 1
    if hi <= lo:
        continue
    key = next((k for k in raw if k in stem or stem in k), None)
    for t0 in sorted(rng.choice(np.arange(lo, hi), size=min(8, hi - lo), replace=False)):
        t0 = int(t0); H = P.HORIZONS[-1]
        seg = poses[t0:t0 + H + 1]                        # raw 10 Hz, 2 s
        if seg.shape[0] < 5:
            continue
        n_all += 1
        x0, y0, yaw0 = seg[0, 0], seg[0, 1], seg[0, 2]
        ex, ey = P.ego_frame_np(seg[:, 0], seg[:, 1], x0, y0, yaw0)
        dt = 0.1
        vx, vy = np.gradient(ex, dt), np.gradient(ey, dt)
        ax, ay = np.gradient(vx, dt), np.gradient(vy, dt)
        spd = np.hypot(vx, vy)
        hdg = np.unwrap(np.arctan2(vy, vx))
        yawr = np.gradient(hdg, dt)
        a_lon = np.gradient(spd, dt)
        a_lat = spd * yawr
        jerk = np.gradient(a_lon, dt)
        kappa = yawr / np.maximum(spd, 0.5)

        vals["a_max_mps2"].append(np.abs(a_lon).max())
        vals["pdm_a_lon_max_mps2"].append(np.abs(a_lon).max())
        vals["lat_acc_max_mps2"].append(np.abs(a_lat).max())
        vals["pdm_a_lat_max_mps2"].append(np.abs(a_lat).max())
        vals["jerk_max_mps3"].append(np.abs(jerk).max())
        vals["kappa_max_1pm"].append(np.abs(kappa).max())
        vals["pdm_yaw_rate_max_radps"].append(np.abs(yawr).max())
        for k in ("a_max_mps2", "pdm_a_lon_max_mps2", "lat_acc_max_mps2",
                  "pdm_a_lat_max_mps2", "jerk_max_mps3", "kappa_max_1pm",
                  "pdm_yaw_rate_max_radps"):
            flag[k].append(float(vals[k][-1] > C[k][0]))

        # --- lead-dependent rows: applicability gate = a lead exists -----------
        if key is None or t0 not in raw[key]:
            continue
        ag = [(cx, cy, ln) for cx, cy, ln in raw[key][t0]
              if 0.0 < cx < P.LEAD_MAX_GAP_M and abs(cy) <= 2.0]
        if not ag:
            continue
        n_lead += 1
        cx, cy, ln = min(ag, key=lambda a: a[0])
        gap = cx - ln / 2.0                                # bumper-to-bumper
        v0 = float(poses[t0, 3])
        clr = np.hypot(ex - cx, ey - cy).min() - 2.0
        vals["proximity_safe_m"].append(max(clr, 0.0))
        flag["proximity_safe_m"].append(float(clr < C["proximity_safe_m"][0]))
        if v0 >= 0.5:
            tg = gap / v0
            vals["target_time_gap_s"].append(tg)
            flag["target_time_gap_s"].append(float(tg < C["target_time_gap_s"][0]))
            ttc = gap / v0 if v0 > 0 else np.inf          # lead static ⇒ closing = v0
            vals["ttc_min_s"].append(ttc)
            flag["ttc_min_s"].append(float(ttc < C["ttc_min_s"][0]))

print(f"windows: {n_all} total · {n_lead} with an in-lane lead\n")
print(f"{'constant':<26}{'value':>7}{'n':>6}{'FLAGS HUMAN':>13}   {'human p50':>9}{'p95':>9}")
out = {}
for k, (v, srcline) in C.items():
    f, a = np.array(flag[k]), np.array(vals[k])
    if not f.size:
        print(f"{k:<26}{v:>7}{0:>6}{'no data':>13}"); continue
    out[k] = {"value": v, "n": int(f.size), "flag_frac": float(f.mean()),
              "human_p50": float(np.median(a)), "human_p95": float(np.percentile(a, 95)),
              "source": srcline}
    mark = " ⛔" if f.mean() > 0.25 else ("  ⚠️" if f.mean() > 0.10 else "")
    print(f"{k:<26}{v:>7}{f.size:>6}{100*f.mean():12.1f}%   "
          f"{np.median(a):9.3f}{np.percentile(a,95):9.3f}{mark}")
json.dump(out, open(f"{O}/threshold_sweep.json", "w"), indent=1)
print(f"\n-> {O}/threshold_sweep.json")
