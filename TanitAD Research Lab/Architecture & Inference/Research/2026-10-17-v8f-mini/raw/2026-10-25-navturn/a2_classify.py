"""a2_classify.py -- classify every one of the 339 nav/turn-label disagreement windows by CAUSE (EXPLORATORY, MEASURED eval-141).

Population (banked, T_ATTRIB s2c_routesilent): A = turn-label cells present at rowT & nav token FOLLOW at rowN (210);  B = nav token TURN at rowN & no turn-label cell at rowT (129).
Facts come from (1) the shipped join, (2) the pose track, (3) the mirror's own v9 builder re-run on the pose track (r1_repro.py; 99.99 % token / presence agreement on rows whose
lookahead the track covers). Evidence tier per window: REPRO (matched in the re-run turn table) / JOIN (decided from join cells + pose speed) / UNVERIFIED (event beyond the pose track).
Cause rules are ordered; every window gets exactly one primary cause (+ flags).
"""
import csv
import json
import numpy as np
from nt_common import *

J, K, T, grid = load_inputs()
R = dict(np.load(HERE + "repro_eval141.npz", allow_pickle=True))
TT = {int(k): v for k, v in json.load(open(HERE + "repro_turns.json")).items()}
P_all = clip_tracks(K)
ci, tt, a_t, rowT, hasw = T["ci"], T["tt"], T["a_t"], T["rowT"], T["has"]
N = len(ci)
rowN = J["off"][ci] + tt
mu, sd = T["mu"], T["sd"]
CONS = J["cons"].astype(np.float64)
tokN = J["nav"][rowN][:, :3].argmax(1)
tokT = J["nav"][rowT][:, :3].argmax(1)
G = T["gt_std"]
labT = np.isfinite(CONS[rowT, 1]) & hasw               # == banked `turn`
assert np.array_equal(labT, np.isfinite(G[:, 1]) & hasw)
A = labT & (tokN == 0)
Bm = hasw & (tokN != 0) & ~labT
print(f"A (turn label, nav FOLLOW) = {int(A.sum())}   B (nav TURN, no turn label) = {int(Bm.sum())}")

CLOCK = {c: clip_clock(J, c) for c in range(len(J["n"])) if J["has"][int(J["off"][c]):int(J["off"][c]) + int(J["n"][c])].any()}
E_NAMES = ["NONE"]


def seg_match(c, now_T, t_start, dyaw):
    """The re-run turn-table entry that the join's label cells describe (None if not in the reconstructed track)."""
    t0 = now_T + t_start
    best = None
    for u in TT.get(c, {}).get("turns", []):
        if abs(u["t0"] - t0) <= 0.25 and np.sign(u["dyaw"]) == np.sign(dyaw) and abs(u["dyaw"] - dyaw) <= 10.0:
            if best is None or abs(u["t0"] - t0) < abs(best["t0"] - t0):
                best = u
    return best


def vmin_inside(c, t0, t1):
    g0, dt = CLOCK[c]
    P = P_all[c]
    ts = g0 + np.arange(len(P)) * dt
    m = (ts >= t0 - 1e-9) & (ts <= t1 + 1e-9)
    return (float(P[m, 3].min()) if m.any() else np.nan), float(ts[-1])


rows = []
for w in np.where(A | Bm)[0]:
    c, rT, rN = int(ci[w]), int(rowT[w]), int(rowN[w])
    g0, dt = CLOCK[c]
    nowT, nowN = float(J["now_s"][rT]), float(J["now_s"][rN])
    v_N = float(P_all[c][2 * int(tt[w]), 3])
    d_ann_N = max(30.0, 6.0 * v_N)
    rec = {"w": int(w), "type": "A" if A[w] else "B", "clip": str(J["clip"][c]), "ci": c, "t": int(tt[w]), "a_T": int(a_t[w]), "rowT": rT, "rowN": rN,
           "tokN": int(tokN[w]), "tokT": int(tokT[w]), "navd_N": float(J["nav_d"][rN]) if np.isfinite(J["nav_d"][rN]) else np.nan, "v_N": round(v_N, 2),
           "d_ann_N": round(d_ann_N, 1), "lat_cls": int(J["lat_cls"][rT]), "dpsi6": float(T["dpsi6"][w]), "theta": float(CONS[rT, 0]),
           "ts": np.nan, "te": np.nan, "dstart": np.nan, "dyaw": np.nan, "Rarc": np.nan, "vmin": np.nan, "is_turn": "", "ev": "", "cause": "", "sub": "", "note": ""}
    if A[w]:
        ts_, te_, dst, dyaw, Rarc = (float(CONS[rT, q]) for q in (1, 2, 3, 5, 6))
        rec.update(ts=ts_, te=te_, dstart=dst, dyaw=dyaw, Rarc=Rarc)
        u = seg_match(c, nowT, ts_, dyaw)
        if u is not None:
            rec["ev"] = "REPRO"
            rec["vmin"] = round(u["vmin"], 2)
            rec["is_turn"] = bool(u["is_turn"])
            if not u["is_turn"]:
                why = []
                if u["R"] > 140.0:
                    why.append("R>140")
                if u["vmin"] > 8.0:
                    why.append("vmin>8")
                if abs(u["dyaw"]) < 15.0:
                    why.append("dyaw<15")
                rec.update(cause="CURVE", sub="+".join(why) if why else "other")
            else:
                to, tn = R["tok_off"][rN], R["tok_on"][rN]
                s_now_N = float(R["s_now"][rN])
                if u["s1"] <= s_now_N:
                    rec.update(cause="TIMING", sub="segment finished before the nav row")
                elif to != 0 and tn == 0:
                    rec.update(cause="SUPPRESSED", sub=f"A7 clip-level suppression (t0={u['t0']:.1f}s in [8,43])")
                elif to != 0:
                    rec.update(cause="UNEXPLAINED", sub="re-run announces, join does not, suppression does not explain")
                else:
                    dn = u["s0"] - s_now_N
                    rec.update(cause="HORIZON", sub="d_next>d_ann", note=f"d_next={dn:.1f}m d_ann={d_ann_N:.1f}m ratio={dn / d_ann_N:.2f} t_start={ts_:.1f}s")
                    rec["horizon_ratio"] = dn / d_ann_N
                    rec["d8"] = bool(dn <= max(30.0, 8.0 * v_N))
        else:
            vm, tlast = vmin_inside(c, nowT + ts_, nowT + te_)
            rec["vmin"] = round(vm, 2)
            if Rarc > 140.0:
                rec.update(ev="JOIN", cause="CURVE", sub="R>140 (segment beyond reconstructed track)", is_turn=False)
            elif not np.isfinite(vm):
                rec.update(ev="UNVERIFIED", cause="UNRESOLVED", sub="segment entirely beyond the pose track")
            elif vm > 8.0:
                rec.update(ev="UNVERIFIED", cause="UNRESOLVED", sub="vmin_inside>8 on a truncated segment")
            else:
                rec.update(ev="JOIN", is_turn=True)
                if dst > d_ann_N:
                    rec.update(cause="HORIZON", sub="d_start>d_ann (join-only)", note=f"d_start={dst:.1f}m d_ann={d_ann_N:.1f}m")
                    rec["horizon_ratio"] = dst / d_ann_N
                    rec["d8"] = bool(dst <= max(30.0, 8.0 * v_N))
                else:
                    rec.update(ev="UNVERIFIED", cause="UNRESOLVED", sub="is_turn by join but d_start<=d_ann and token FOLLOW")
    else:                                                  # type B
        navd = rec["navd_N"]
        if tokT[w] == 0:
            rec.update(ev="JOIN", cause="TIMING", sub="token at the T-tick row (what T reads) is FOLLOW: banked rowN pairing artefact")
        elif not np.isfinite(CONS[rT, 0]):
            rev = R["reversing"][rT]
            rec.update(ev="JOIN", cause="LABEL_MASKED", sub=("reversing" if rev == 1 else "h_obs<3 s or reversing (log end)"))
        else:
            # the announced turn the nav token describes: nearest announced (is_turn, s1 > s_now) in the re-run table
            s_now_N = float(R["s_now"][rN])
            tabs = TT.get(c, {}).get("turns", [])
            on = bool(R["tok_on"][rN] == tokN[w]) and not bool(R["tok_off"][rN] == tokN[w])
            cand = [u for u in tabs if u["is_turn"] and not (on and u["sup"]) and u["s1"] > s_now_N]
            u = min(cand, key=lambda q: q["s0"]) if cand else None
            if u is not None and np.isfinite(navd) and abs(max(0.0, u["s0"] - s_now_N) - navd) <= 1.5:
                rec["ev"] = "REPRO"
                a, b = u["t0"] - nowT, u["t1"] - nowT
                rec.update(ts=a, te=b, dyaw=u["dyaw"], Rarc=u["R"], vmin=round(u["vmin"], 2), is_turn=True)
                b1 = 8.0
                if b < 2.0:
                    rec.update(cause="BAND_START", sub="announced turn ends before NOW+2 s (label band is [NOW+2, NOW+8])", note=f"t0={a:.1f}s t1={b:.1f}s")
                elif a > b1:
                    rec.update(cause="HORIZON", sub="announced turn starts after NOW+8 s", note=f"t0={a:.1f}s d={navd:.1f}m d_ann={d_ann_N:.1f}m v={v_N:.1f}")
                else:
                    # overlap with the band but no dominant segment: the builder's sign / theta rule
                    th = CONS[rT, 0]
                    ov = min(b, 8.0) - max(a, 2.0)
                    if ov < 0.5:
                        rec.update(cause="BAND_EDGE", sub=f"announced turn grazes the label band [NOW+2, NOW+8] (overlap {ov:.2f} s)", note=f"t0={a:.1f}s t1={b:.1f}s theta={th:.1f} seg_dyaw={u['dyaw']:.1f}")
                    else:
                        rec.update(cause="SIGN_THETA", sub="overlaps band >=0.5 s, dominant-segment rule rejects it", note=f"t0={a:.1f}s t1={b:.1f}s overlap={ov:.2f}s theta={th:.1f} seg_dyaw={u['dyaw']:.1f}")
            elif np.isfinite(navd) and navd > float(R["s_left"][rN]) + 1.0 and float(R["t_left"][rT]) >= 8.0:
                # the announced turn starts beyond the END of the pose track, and the track still has >= 8 s after the label row => it starts after NOW+8 s
                rec.update(ev="TRACK", cause="HORIZON", sub="announced turn starts beyond the pose-track end, which is >= 8 s ahead", note=f"d={navd:.1f}m arc_left={float(R['s_left'][rN]):.1f}m t_left={float(R['t_left'][rT]):.1f}s v={v_N:.1f}")
            else:
                rec["ev"] = "UNVERIFIED"
                if False:
                    test = navd / v_N
                    if test > 8.0:
                        rec.update(ev="JOIN", cause="HORIZON", sub="nav distance / v_now > 8 s (ESTIMATED, segment beyond track)", note=f"d={navd:.1f}m v={v_N:.1f} -> {test:.1f}s")
                    else:
                        rec.update(cause="UNRESOLVED", sub="announced turn beyond the reconstructed track, time-to-start <= 8 s est.")
                else:
                    rec.update(cause="UNRESOLVED", sub="announced turn beyond the reconstructed track")
    rows.append(rec)

# ------------------------------------------------------------------ tables
def table(rs, typ, title):
    sel = [r for r in rs if r["type"] == typ]
    n = len(sel)
    print(f"\n{title}  (n = {n})")
    causes = {}
    for r in sel:
        causes.setdefault(r["cause"], []).append(r)
    print(f"  {'cause':14s} {'n':>4s} {'share':>6s}  evidence (REPRO/TRACK/JOIN/UNVERIFIED)   sub-reasons")
    for cz, lst in sorted(causes.items(), key=lambda kv: -len(kv[1])):
        ev = {e: sum(1 for r in lst if r["ev"] == e) for e in ("REPRO", "TRACK", "JOIN", "UNVERIFIED")}
        subs = {}
        for r in lst:
            subs[r["sub"].split(" (")[0]] = subs.get(r["sub"].split(" (")[0], 0) + 1
        neps = len({r["clip"] for r in lst})
        print(f"  {cz:14s} {len(lst):4d} {len(lst) / n:6.1%} eps={neps:3d}  {ev['REPRO']:4d}/{ev['TRACK']:4d}/{ev['JOIN']:4d}/{ev['UNVERIFIED']:4d}       {dict(sorted(subs.items(), key=lambda kv: -kv[1]))}")
    return causes


cA = table(rows, "A", "A: turn label present, nav FOLLOW (the 210)")
cB = table(rows, "B", "B: nav TURN, no turn label (the 129)")

# ------------------------------------------------------------------ examples
def ex(lst, k=4):
    seen, out = set(), []
    for r in sorted(lst, key=lambda r: (r["ev"] != "REPRO", r["clip"], r["t"])):
        if r["clip"] in seen:
            continue
        seen.add(r["clip"])
        out.append(r)
        if len(out) >= k:
            break
    return out


def fmt(r):
    f = lambda x, p=1: ("nan" if not np.isfinite(x) else f"{x:.{p}f}")
    return (f"ep {r['clip'][:8]} t={r['t']:3d} aT={r['a_T']} | nav tok={r['tokN']} d_next={f(r['navd_N'])}m d_ann={r['d_ann_N']}m v={r['v_N']}m/s | "
            f"label t_start={f(r['ts'])}s t_end={f(r['te'])}s d_start={f(r['dstart'])}m dyaw={f(r['dyaw'])} R={f(r['Rarc'])} vmin={f(r['vmin'], 2)} is_turn={r['is_turn']} | "
            f"theta={f(r['theta'])} dpsi6={f(r['dpsi6'])} lat_cls={LAT_NAMES.get(r['lat_cls'], r['lat_cls'])} | {r['sub']}{(' ' + r['note']) if r['note'] else ''}")


print("\n==== EXAMPLES ====")
for typ, cz in (("A", cA), ("B", cB)):
    for name, lst in cz.items():
        print(f"\n[{typ}] {name} (n={len(lst)})")
        for r in ex(lst):
            print("   " + fmt(r))

# ------------------------------------------------------------------ the HORIZON detail
print("\n==== HORIZON detail (A) ====")
hz = [r for r in rows if r["type"] == "A" and r["cause"] == "HORIZON"]
if hz:
    ratio = np.array([r["horizon_ratio"] for r in hz])
    print(f"n {len(hz)}; d_next/d_ann median {np.median(ratio):.2f} (p10 {np.percentile(ratio, 10):.2f}, p90 {np.percentile(ratio, 90):.2f}); within 1.25x of d_ann: {(ratio <= 1.25).sum()}; "
          f"would be announced by an 8 s x v_now lookahead: {sum(1 for r in hz if r.get('d8'))}")
    print(f"label t_start median {np.median([r['ts'] for r in hz]):.1f}s; v_now median {np.median([r['v_N'] for r in hz]):.1f} m/s")
hb = [r for r in rows if r["type"] == "B" and r["cause"] == "HORIZON"]
if hb:
    print(f"(B) HORIZON n {len(hb)}: v_now median {np.median([r['v_N'] for r in hb]):.1f} m/s, nav d_next median {np.median([r['navd_N'] for r in hb]):.1f} m, d_ann median {np.median([r['d_ann_N'] for r in hb]):.1f} m")

# ------------------------------------------------------------------ CURVE detail
print("\n==== CURVE detail (A) ====")
cv = [r for r in rows if r["type"] == "A" and r["cause"] == "CURVE"]
if cv:
    print(f"n {len(cv)}; lat_cls: {{{', '.join(f'{LAT_NAMES[k]}: {sum(1 for r in cv if r['lat_cls'] == k)}' for k in (-100, 0, 1, 2))}}}")
    print(f"|dyaw| median {np.median(np.abs([r['dyaw'] for r in cv])):.1f} deg; R_arc median {np.median([r['Rarc'] for r in cv]):.0f} m; vmin median {np.nanmedian([r['vmin'] for r in cv]):.1f} m/s")
    print(f"R>140: {sum(1 for r in cv if r['Rarc'] > 140)}  vmin>8: {sum(1 for r in cv if np.isfinite(r['vmin']) and r['vmin'] > 8)}  both: {sum(1 for r in cv if r['Rarc'] > 140 and np.isfinite(r['vmin']) and r['vmin'] > 8)}")
    big = sum(1 for r in cv if abs(r['theta']) >= 30)
    print(f"|theta|>=30 (the 'big' flag of the class rule): {big}")
    ep = len({r['clip'] for r in cv})
    print(f"episodes involved: {ep}")

# ------------------------------------------------------------------ the T-tick variant and the hidden disagreement
print("\n==== T-tick pairing variant (what T reads: nav and label both at rowT) ====")
tickA = labT & (tokT == 0)
tickB = hasw & (tokT != 0) & ~labT
print(f"A_tick {int(tickA.sum())}  B_tick {int(tickB.sum())}  total {int(tickA.sum() + tickB.sum())}  (banked 210 + 129 = 339; overlap A {int((A & tickA).sum())}, B {int((Bm & tickB).sum())})")

agree = labT & (tokT != 0)
navdT = J["nav_d"][rowT]
dstart = CONS[rowT, 3]
same = agree & np.isfinite(navdT) & (np.abs(dstart - navdT) <= 1.5)
print(f"agreeing windows (turn label & nav TURN at rowT): {int(agree.sum())}; nav distance equals the label's d_start (|diff|<=1.5 m, i.e. SAME event): {int(same.sum())} "
      f"({same.sum() / agree.sum():.1%}); different event {int((agree & ~same).sum())}")
sideL = np.sign(CONS[rowT, 5])
side_nav = np.where(tokT == 1, 1, np.where(tokT == 2, -1, 0))
print(f"side check on agreeing windows: nav side != sign(label dyaw) in {int((agree & (side_nav != sideL)).sum())} of {int(agree.sum())}")

with open(HERE + "navturn_windows.csv", "w", newline="", encoding="utf-8") as fh:
    keys = ["w", "type", "clip", "t", "a_T", "rowT", "rowN", "tokN", "tokT", "navd_N", "v_N", "d_ann_N", "lat_cls", "dpsi6", "theta", "ts", "te", "dstart", "dyaw", "Rarc", "vmin", "is_turn", "ev", "cause", "sub", "note"]
    wr = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
    wr.writeheader()
    for r in rows:
        wr.writerow(r)
print(f"\nwrote navturn_windows.csv ({len(rows)} rows)")
np.save(HERE + "navturn_cause.npy", np.array([(r["w"], r["type"], r["cause"]) for r in rows], dtype=object), allow_pickle=True)
