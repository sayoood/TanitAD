#!/usr/bin/env python3
"""ff_table.py -- print the FOUR BINDING METRIC FAMILIES from a refcv3_arm.py
result JSON, per arm, never pooled.

Usage:  python ff_table.py <result.json> [<label>]
        python ff_table.py <a.json> <b.json>   # two-file diff mode

Every row carries the arm's TIER. Families that are unavailable print their
status + reason + n rather than being dropped (clause 5 of the binding rule).
"""
import json
import sys


def g(d, *ks, default=None):
    for k in ks:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def fnum(v, nd=4):
    if v is None:
        return "  --  "
    try:
        return f"{float(v):.{nd}f}"
    except Exception:
        return str(v)


def load(p):
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def arm_rows(d):
    out = {}
    for arm, a in d.get("arms", {}).items():
        ff = a.get("four_families", {}) or {}
        lon = ff.get("longitudinal", {}) or {}
        lat = ff.get("lateral", {}) or {}
        tac = ff.get("tactical", {}) or {}
        st = ff.get("strategic", {}) or {}
        iv = a.get("intervals", {}) or {}
        # ADE: prefer the decision-grade interval block
        ade = None
        ade_lo = ade_hi = None
        mets = (iv.get("metrics") or {}) if isinstance(iv, dict) else {}
        for key in ("ade_dense_m", "ade_m", "ade"):
            if key in mets and isinstance(mets[key], dict):
                ade = mets[key].get("mean")
                ade_lo, ade_hi = mets[key].get("lo"), mets[key].get("hi")
                break
        fde = (mets.get("fde_last_m") or {}).get("mean")
        out[arm] = dict(
            tier=a.get("tier"), fde=fde, mets=mets,
            tier_note=(a.get("tier_note") or "")[:60],
            ade=ade, ade_lo=ade_lo, ade_hi=ade_hi,
            # LONGITUDINAL
            speed_mae=lon.get("speed_mae_mps"),
            speed_bias=lon.get("speed_bias_mps"),
            tgt_acc=lon.get("target_speed_acc"),
            along_mae=lon.get("along_mae_m"),
            dk=lon.get("distance_keeping"),
            # LATERAL
            head_mae=lat.get("heading_mae_deg"),
            yawr_mae=lat.get("yaw_rate_mae_degps"),
            curv_mae=lat.get("curvature_mae_1pm"),
            curv_bias=lat.get("curvature_bias_1pm"),
            cross_mae=lat.get("cross_mae_m"),
            n_curv=lat.get("n_steps_curvature"),
            # TACTICAL
            tac_status=tac.get("status"),
            lat_dec=tac.get("lateral_decision"),
            lon_dec=tac.get("longitudinal_decision"),
            goal=tac.get("goal_setting"),
            # STRATEGIC
            strat=st,
            ff_meta={k: v for k, v in ff.items() if k.startswith("_")},
        )
    return out


def dec_line(dd):
    if not isinstance(dd, dict):
        return "  --"
    acc = dd.get("accuracy", dd.get("acc"))
    kap = dd.get("kappa")
    n = dd.get("n")
    s = f"acc {fnum(acc)}  kappa {fnum(kap)}  n {n}"
    pc = dd.get("per_class") or {}
    if pc:
        bits = []
        for cls, v in pc.items():
            if not isinstance(v, dict):
                continue
            r = v.get("recall")
            nt = v.get("n_true", v.get("support"))
            if nt:
                bits.append(f"{cls} r={fnum(r,3)}(n{nt})")
        if bits:
            s += "\n            per-class: " + "  ".join(bits)
    return s


def show(path, label=None):
    d = load(path)
    label = label or path
    print("=" * 100)
    print(f"FOUR-FAMILY PANEL -- {label}")
    print(f"  n_windows={d.get('n_windows')}  n_episodes={d.get('n_episodes')}  "
          f"grid dt={d.get('dt_s')} horizon_steps={d.get('horizon_steps')}")
    print(f"  ckpt={d.get('ckpt')}")
    est = d.get("_estimator")
    print(f"  estimator={est if isinstance(est,str) else json.dumps(est)[:200]}")
    print("=" * 100)
    rows = arm_rows(d)
    for arm, r in rows.items():
        print(f"\n--- ARM {arm}   TIER {r['tier']}  {r['tier_note']}")
        print(f"  ADE(dense) {fnum(r['ade'])} m  CI [{fnum(r['ade_lo'])}, {fnum(r['ade_hi'])}]   FDE {fnum(r['fde'])} m")
        print(f"  LONGITUDINAL  speed_MAE {fnum(r['speed_mae'])} m/s  bias {fnum(r['speed_bias'])}"
              f"  target_speed_acc {fnum(r['tgt_acc'])}  along_MAE {fnum(r['along_mae'])} m")
        dk = r["dk"]
        if isinstance(dk, dict):
            print(f"                distance_keeping: status={dk.get('status')} n={dk.get('n')} "
                  f"headway_MAE={fnum(dk.get('headway_mae_m'))} m  ttc={fnum(dk.get('ttc_mae_s'))} s "
                  f"gap={fnum(dk.get('time_gap_mae_s'))} s")
        print(f"  LATERAL       heading_MAE {fnum(r['head_mae'])} deg  yaw_rate_MAE {fnum(r['yawr_mae'])} deg/s"
              f"  CURV_MAE {fnum(r['curv_mae'],6)} 1/m  curv_bias {fnum(r['curv_bias'],6)}"
              f"  cross_MAE {fnum(r['cross_mae'])} m  n_curv={r['n_curv']}")
        print(f"  TACTICAL      status={r['tac_status']}")
        print(f"     lateral_decision:      {dec_line(r['lat_dec'])}")
        print(f"     longitudinal_decision: {dec_line(r['lon_dec'])}")
        gs = r["goal"]
        if isinstance(gs, dict):
            print(f"     goal_setting: " + "  ".join(
                f"{k}={fnum(v) if isinstance(v,(int,float)) else v}"
                for k, v in list(gs.items())[:8]))
        st = r["strat"]
        if isinstance(st, dict):
            print(f"  STRATEGIC     status={st.get('status')} n={st.get('n')}")
            for k, v in st.items():
                if k in ("status", "n", "how_to_populate", "estimator"):
                    continue
                if isinstance(v, dict):
                    print(f"     {k}: " + "  ".join(
                        f"{kk}={fnum(vv) if isinstance(vv,(int,float)) else str(vv)[:40]}"
                        for kk, vv in list(v.items())[:8]))
                else:
                    print(f"     {k}: {str(v)[:160]}")
    print("\n" + "=" * 100)
    print("PAIRED MARGINS (decision-grade)")
    WANT = ("ade_m", "speed_mae_mps", "curvature_mae_1pm", "cross_mae_m",
            "heading_mae_deg", "along_mae_m", "yaw_rate_mae_degps")
    for k, v in (d.get("paired_decision_grade") or {}).items():
        if not isinstance(v, dict):
            continue
        print(f"  {k}   [{v.get('direction')}]  n_win={v.get('n')}")
        for m in WANT:
            b = v.get(m)
            if isinstance(b, dict):
                print(f"      {m:22s} delta={fnum(b.get('delta'),5)} "
                      f"[{fnum(b.get('lo'),5)}, {fnum(b.get('hi'),5)}] "
                      f"separated={b.get('separated')} p={b.get('p_delta_gt0')}")
    return rows, d


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[2].endswith(".json"):
        show(sys.argv[1], "A: " + sys.argv[1])
        show(sys.argv[2], "B: " + sys.argv[2])
    else:
        show(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
