"""Mutation-tested reference builder for the recommended refcv9 fed ceiling (L4 candidate C7).

C7 = the v9 past-only column `speed_n2_kmh` (max realised ego speed over [NOW-20 s, NOW], snapped UP to the road-law
8-step ladder, urban floor 50) fed through the refcv6 4-way CONTAINING window {30, 50, 100, 120}; a value above 120 keeps
its own value (refcv8 n2 rule: the ceiling is never below the fed value); -1 / missing = UNKNOWN (+inf, all-zero row).
Expectations are LITERALS (independently authored road-law steps), never expressions over the code under test.
Run: python -X utf8 test_l4_builder.py  (exit 0 iff production is GREEN and EVERY mutant is RED)."""
import math, sys
import numpy as np

LADDER4_KMH = (30, 50, 100, 120)
LADDER8_KMH = (20, 30, 50, 70, 80, 100, 120, 130)


# ---------------------------------------------------------------- production
def n2_from_log(v_ms, k, hz=10, past_s=20.0, floor_kmh=50.0):
    """PAST-ONLY: reads v_ms[k - past_s*hz : k] inclusive of NOW, never k+1.."""
    lo = max(0, k - int(round(past_s * hz)))
    vmax_kmh = max(float(np.nanmax(np.asarray(v_ms[lo:k + 1], np.float64))) * 3.6, floor_kmh)
    for s in LADDER8_KMH:                      # snap UP in km/h integers (no m/s rounding anywhere)
        if vmax_kmh <= s + 1e-9:
            return s
    return LADDER8_KMH[-1]


def c7_ceiling_kmh(n2_kmh):
    if n2_kmh is None or n2_kmh < 0:
        return math.inf                        # unknown -> all-zero one-hot -> +inf (X15)
    for s in LADDER4_KMH:
        if n2_kmh <= s + 1e-9:
            return float(s)
    return float(n2_kmh)                       # > 120: keep the fed value (refcv8 'never below the fed value')


# ---------------------------------------------------------------- the checks
LITERAL = {20: 30.0, 30: 30.0, 50: 50.0, 70: 100.0, 80: 100.0, 100: 100.0, 120: 120.0, 130: 130.0, -1: math.inf}


def checks(builder, n2fn):
    fails = []
    for n2, want in LITERAL.items():           # (1) literal table
        got = builder(n2)
        if not (got == want):
            fails.append(f"literal n2={n2}: got {got}, want {want}")
    rng = np.random.default_rng(0)             # (2) future invariance: scramble every sample after NOW
    for _ in range(200):
        # past slow (< 30 km/h, so the ceiling sits at the urban floor) and the scrambled future FAST: a check whose
        # data saturates the ladder (max of 200 draws up to 38 m/s always snaps to 130) is blind to a future read --
        # MEASURED: the first version of this check read M3 GREEN for exactly that reason.
        v = rng.uniform(0, 8, 300); k = int(rng.integers(0, 230))
        a = builder(n2fn(v, k)); v2 = v.copy(); v2[k + 1:] = rng.uniform(20, 38, 299 - k)
        if builder(n2fn(v2, k)) != a:
            fails.append("future samples changed the ceiling"); break
    for kmh in (49.0, 50.0, 50.0001, 69.9, 100.0, 120.0):   # (3) fixed point under an independent km/h consumer (rounding trap)
        c = builder(n2fn(np.array([kmh / 3.6]), 0))
        resnap = next((s for s in LADDER4_KMH if c <= s + 1e-9), c)
        if resnap != c:
            fails.append(f"ceiling {c} km/h is not a fixed point of a km/h-integer consumer (re-snaps to {resnap})")
    if builder(n2fn(np.array([50.0 / 3.6]), 0)) != 50.0:    # (4) analytic: exactly 50 km/h stays 50
        fails.append("exactly 50 km/h did not map to 50")
    return fails


# ---------------------------------------------------------------- mutants (each reintroduces a real defect class)
def m_snap_down(n2):                           # M1: nearest-lower step (the snap-DOWN)
    if n2 is None or n2 < 0:
        return math.inf
    return float(max([s for s in LADDER4_KMH if s <= n2] or [LADDER4_KMH[0]]))


def m_clamp_120(n2):                           # M2: clamp >120 to 120 (refcv6 label rule, the (i) cause)
    return min(c7_ceiling_kmh(n2), 120.0)


def m_future_window(v_ms, k, hz=10):           # M3: window reaches into the future [NOW-20 s, NOW+6 s]
    return n2_from_log(np.concatenate([v_ms, v_ms[-1:]]), min(k + 60, len(v_ms) - 1)) if k + 60 < len(v_ms) else n2_from_log(v_ms, len(v_ms) - 1)


def m_rounded_ladder(n2):                      # M4: the 4-dp m/s ladder, read back in km/h (the 57.5 % trap)
    if n2 is None or n2 < 0:
        return math.inf
    for s in LADDER4_KMH:
        if n2 / 3.6 <= round(s / 3.6, 4) + 1e-9:
            return round(s / 3.6, 4) * 3.6
    return float(n2)


def m_unknown_as_30(n2):                       # M5: unknown collapsed onto the slowest class (X15)
    return 30.0 if (n2 is None or n2 < 0) else c7_ceiling_kmh(n2)


if __name__ == "__main__":
    report = {"production": checks(c7_ceiling_kmh, n2_from_log)}
    for name, (b, f) in {"M1_snap_down": (m_snap_down, n2_from_log), "M2_clamp_120": (m_clamp_120, n2_from_log),
                         "M3_future_window": (c7_ceiling_kmh, m_future_window), "M4_rounded_ladder": (m_rounded_ladder, n2_from_log),
                         "M5_unknown_as_30": (m_unknown_as_30, n2_from_log)}.items():
        report[name] = checks(b, f)
    ok = (not report["production"]) and all(report[k] for k in report if k != "production")
    for k, v in report.items():
        print(f"{k:20s} {'GREEN' if not v else 'RED'}  {v[:2]}")
    print("SUITE", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
