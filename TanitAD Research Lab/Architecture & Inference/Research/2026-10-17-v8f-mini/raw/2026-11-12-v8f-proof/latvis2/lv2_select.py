"""lv2_select.py: the 18 windows of the sheets (card PREREG_V8F_PROOF_ADD38 section 2). LABELS AND LOGGED SIGNALS ONLY: the function takes the pack fields (eid, g0, V = ego speed per frame,
G15 = label lead gap per frame) and the earlier panel list; no model output (no dump, no state, no rollout) can enter. Frame 4 of a window = t, frame 14 = t + 2.0 s.

  python lv2_select.py --pack eval_pack.npz --panel-windows PANEL_WINDOWS.json --out WINDOWS_LV2.json

RULE_TEXT is written into the output. Ties are broken by the lower window index; "at most one per episode" holds inside the 6 lead windows; a window that is already one of the 12 panel
windows is not chosen again (the card asks for 18 windows)."""
import argparse
import json
import sys

import numpy as np

RULE_TEXT = ("12 windows of latvis/PANEL_WINDOWS.json (6 with the largest speed change over 2.0 s, 6 with the smallest lead gap; chosen before any decoder on 2026-11-13) + 6 lead windows: among the "
             "windows with a label gap at t (frame 4) between 5 and 50 m, an ego speed at t of at least 5 m/s and a label gap at t + 2.0 s (frame 14), at most one per episode, not already a panel window, "
             "the 3 with the most negative and the 3 with the most positive change of the label gap over 2.0 s; ties by the lower window index. Inputs: eval_pack fields eid, g0, V, G15 only.")
S0, T2 = 4, 14


def panel_windows(eid, g0, panel):
    out = []
    for w in panel["windows"]:
        i = int(w["window"])
        if str(eid[i]) != str(w["clip"]) or int(g0[i]) != int(w["g0"]) + S0:
            raise SystemExit(f"REFUSED: panel window {i} is not clip {w['clip']} at step {w['g0']} (+4) of the pack: pack has {eid[i]} / {g0[i]}")
        out.append({"window": i, "group": w["group"], "eid": str(eid[i]), "g0": int(g0[i])})
    return out


def lead_windows(eid, g0, V, G15, exclude=(), n_each=3):
    gap_t, gap_2 = np.asarray(G15, np.float64)[:, S0], np.asarray(G15, np.float64)[:, T2]
    ok = np.isfinite(gap_t) & np.isfinite(gap_2) & (gap_t >= 5.0) & (gap_t <= 50.0) & (np.asarray(V, np.float64)[:, S0] >= 5.0)
    ok[list(exclude)] = False
    chg = gap_2 - gap_t
    idx = np.flatnonzero(ok)
    used, out = set(), []
    for group, key in (("lead_gap_closing", lambda i: (chg[i], i)), ("lead_gap_opening", lambda i: (-chg[i], i))):
        n = 0
        for i in sorted(idx, key=key):
            if str(eid[i]) in used:
                continue
            used.add(str(eid[i]))
            out.append({"window": int(i), "group": group, "eid": str(eid[i]), "g0": int(g0[i]), "gap_t_m": float(gap_t[i]), "gap_t2_m": float(gap_2[i]), "change_m": float(chg[i]), "speed_t_mps": float(V[i, S0])})
            n += 1
            if n == n_each:
                break
    return out


def select_windows(eid, g0, V, G15, panel):
    pw = panel_windows(eid, g0, panel)
    lw = lead_windows(eid, g0, V, G15, exclude=[w["window"] for w in pw])
    return {"rule": RULE_TEXT, "windows": pw + lw, "window_indices": [w["window"] for w in pw + lw], "n_candidates_lead": int(len(lw))}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--panel-windows", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    P = np.load(a.pack, allow_pickle=True)
    with open(a.panel_windows, encoding="utf-8") as fh:
        panel = json.load(fh)
    r = select_windows(P["eid"].astype(str), P["g0"], P["V"], P["G15"], panel)
    with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(r, fh, indent=1)
    print(f"lv2_select: {len(r['windows'])} windows -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
