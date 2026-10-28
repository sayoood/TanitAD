"""H-ST-RECOVERYLABEL zero-training step on the real bank (deliverable 2). Frozen rules: FROZEN_CARDS.md L1-L8 + FROZEN_AMENDMENT_1.md A1.
Usage: python -I recovery_run.py s0|s1     -> ../raw/recovery_<seed>.json (+ raw/real_panel.npz from s0)
Labels every off_lat / off_head record with labeller.recovery_label; ok records keep the human label; end_edge / nonfinite stay cut.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "scripts"))
import common as C  # noqa: E402
import labeller as L  # noqa: E402

STATUS = ["emitted", "refused_infeasible", "no_logged_future", "nonfinite", "path_exhausted", "degenerate", "emitted_identity"]
SCODE = {s: i for i, s in enumerate(STATUS)}


def quant(x):
    x = np.asarray(x, np.float64)
    if len(x) == 0:
        return None
    q = np.percentile(x, [5, 25, 50, 75, 95, 99])
    return {"n": int(len(x)), "mean": float(x.mean()), "p5": float(q[0]), "p25": float(q[1]), "p50": float(q[2]), "p75": float(q[3]), "p95": float(q[4]), "p99": float(q[5]), "max": float(x.max())}


def main(seed):
    t_start = time.time()
    tab = C.attach_logged(C.load_table(seed), C.load_logged())
    lg = C.load_logged()
    G, V, offs, Ts = lg["G"], lg["V"], lg["offs"], lg["Ts"]
    reason, tick = tab["reason"], tab["tick"]
    cut = np.flatnonzero((reason == C.OFF_LAT) | (reason == C.OFF_HEAD))
    lim = int(__import__("os").environ.get("REC_LIMIT", "0"))
    if lim:
        cut = cut[np.random.default_rng(1).choice(len(cut), size=lim, replace=False)]      # SMOKE ONLY: never the final outputs
    n = len(cut)
    st = np.full(n, -1, np.int8)
    k_e = np.full(n, -1, np.int16)
    k_t = np.full(n, -1, np.int16)
    n_try = np.zeros(n, np.int8)
    dmean = np.full(n, np.nan)
    dmax = np.full(n, np.nan)
    dend = np.full(n, np.nan)
    viol = np.zeros(n, np.int8)
    rng = np.random.default_rng(0)
    keep_panel = set(rng.choice(n, size=min(400, n), replace=False).tolist()) if seed == "s0" else set()
    panel = []
    for ii, rec in enumerate(cut):
        c = int(tab["cidx"][rec])
        off, T = int(offs[c]), int(Ts[c])
        grid, vlog = G[off:off + T], V[off:off + T]
        r = int(tab["r"][rec])
        own = (float(tab["ox"][rec]), float(tab["oy"][rec]), float(tab["oyaw"][rec]))
        vown = float(tab["v_own"][rec])
        res = L.recovery_label(grid, vlog, r, own, vown)
        st[ii] = SCODE[res["status"]]
        n_try[ii] = res["n_rungs_tried"]
        if res["label_own"] is not None:
            k_e[ii] = res["k_rej"]
            k_t[ii] = res.get("k_target", -1)
            d = np.hypot(*(res["label_world"][1:] - grid[r + 1:r + 1 + L.H, :2]).T)
            dmean[ii], dmax[ii], dend[ii] = d.mean(), d.max(), d[-1]
            v = L.violations(res["label_own"], vown, res["k_rej"] + 1)             # independent re-check of the emitted label
            viol[ii] = 1 if v["any"] else 0
        if ii in keep_panel:
            panel.append((c, off, T, r, own, vown))
        if (ii + 1) % 20000 == 0:
            print(f"[{seed}] {ii + 1}/{n}  {time.time() - t_start:.0f} s", flush=True)
    out = {"seed": seed, "frozen_cards_sha256": C.FROZEN_SHA, "frozen_amendment_1_sha256": C.AMEND_SHA,
           "labeller_sha256": {"labeller.py": __import__("hashlib").sha256(open(HERE / "labeller.py", "rb").read()).hexdigest()},
           "tier": "T1-kin rollout states on TRAIN (DAgger bank); labels are offline (may use anything)", "evidence_class": "MEASURED",
           "n_cut_records_labelled": int(n), "seconds": round(time.time() - t_start, 1)}
    N = C.N_WINDOWS
    kept = {}
    for tk in (1, 2):
        m_tk = tick == tk
        n_ok = int(((reason == C.OK) & m_tk).sum())
        row = {"ok": n_ok, "baseline_kept_fraction": n_ok / N, "ceiling_if_all_offroute_recovered": int(((reason <= C.OFF_HEAD) & m_tk).sum()) / N}
        for rn, rc in (("off_lat", C.OFF_LAT), ("off_head", C.OFF_HEAD)):
            sel = (reason[cut] == rc) & (tick[cut] == tk)
            cnt = {STATUS[k]: int(((st == k) & sel).sum()) for k in range(len(STATUS))}
            row[rn] = {"n": int(sel.sum()), **cnt}
        rec = row["off_lat"]["emitted"] + row["off_head"]["emitted"]
        row["recovered_emitted"] = int(rec)
        row["kept_fraction_with_recovery"] = (n_ok + rec) / N
        row["end_edge_stays_cut"] = int(((reason == C.END_EDGE) & m_tk).sum())
        kept[f"tick{tk}"] = row
    out["kept"] = kept
    em = st == SCODE["emitted"]
    out["feasibility"] = {"n_emitted": int(em.sum()), "n_emitted_with_checker_violation": int(viol[em].sum()), "violation_rate_emitted": float(viol[em].mean()) if em.any() else None,
                          "n_refused_infeasible": int((st == SCODE["refused_infeasible"]).sum()),
                          "refusals_by_status": {STATUS[k]: int((st == k).sum()) for k in range(len(STATUS))}}
    out["deviation_m"] = {"mean_over_k": quant(dmean[em]), "max_over_k": quant(dmax[em]), "end_6s": quant(dend[em])}
    for rn, rc in (("off_lat", C.OFF_LAT), ("off_head", C.OFF_HEAD)):
        sel = em & (reason[cut] == rc)
        out["deviation_m"][f"{rn}_mean_over_k"] = quant(dmean[sel])
    out["target_row_k_j_hist"] = {int(k): int(((k_t == k) & em).sum()) for k in L.LADDER}
    out["arrival_step_k_e_quantiles"] = quant(k_e[em])
    # baseline: the HUMAN label's own control violations at 0.2 s resolution on a 4,000-record sample of ok records (from the human pose with v = logged speed)
    ok_idx = np.flatnonzero(reason == C.OK)
    smp = rng.choice(ok_idx, size=4000, replace=False)
    hv = 0
    nh = 0
    for rec in smp:
        c = int(tab["cidx"][rec])
        off, T = int(offs[c]), int(Ts[c])
        r = int(tab["r"][rec])
        if r + L.H > T - 1 or not np.isfinite(V[off + r]):
            continue
        nh += 1
        hv += L.violations(L.human_label(G[off:off + T], r), float(V[off + r]), 30)["any"]
    out["human_label_baseline_beside"] = {"n_sampled": int(nh), "n_with_violation_at_0.2s": int(hv), "rate": hv / max(nh, 1)}
    (C.RAW / (f"recovery_{seed}_SMOKE{lim}.json" if lim else f"recovery_{seed}.json")).write_text(json.dumps(out, indent=1))
    if seed == "s0" and not lim:
        # real panel for the pytest: 1,000 identity windows + the 400 off-route records (self-contained slices)
        ident = []
        for _ in range(1000):
            c = int(rng.integers(0, len(Ts)))
            T = int(Ts[c])
            r = int(rng.integers(1, T - 31))
            ident.append((c, int(offs[c]), T, r, (G[offs[c] + r, 0], G[offs[c] + r, 1], G[offs[c] + r, 2]), float(V[offs[c] + r])))
        recs = [(0,) + p for p in ident] + [(1,) + p for p in panel]
        starts, Gs, Vs, meta, owns = [], [], [], [], []
        pos = 0
        for kind, c, off, T, r, own, vown in recs:
            Gs.append(G[off + r:off + T])
            Vs.append(V[off + r:off + T])
            starts.append(pos)
            pos += T - r
            meta.append((kind, c, T, r))
            owns.append((own[0], own[1], own[2], vown))
        np.savez(C.RAW / "real_panel.npz", G=np.concatenate(Gs), V=np.concatenate(Vs), starts=np.array(starts), meta=np.array(meta), owns=np.array(owns))
    print(json.dumps({k: out[k] for k in ("kept", "feasibility")}, indent=1)[:3500])


if __name__ == "__main__":
    main(sys.argv[1])
