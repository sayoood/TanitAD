"""S0-C (agent check, successor of S0B-READOUT): a learned fusion q = f(features, P6 logits), fit out-of-fold, read on held-out.

MM 2026-11-14. Frozen by FREEZE_S0C.md BEFORE this script was run. CPU, navtrain fit pack only, nothing keyed.
  fit fold  = covered sets NOT in the SFT held-out logs (fit-extra); eval fold = covered held-out sets (the S0-B deciding column).
  label     = TRUE agent-ok per proposal: (nc > 0) & (ttc > 0)  (fit-pack sub-scores, NAVSIM's own).
  model     = sklearn LogisticRegression(C=1.0, max_iter=5000, lbfgs) on standardised features (scaler fit on the fit fold only).
  features  = F0: P6's six raw logits (no perception: a recalibration of P6's own heads, the cheapest 're-ranker')
              F-X: F0 + the S0-B readout flags (NC-proxy, TTC-proxy) of arm X, X in R-true / R-t0 / R-cv / C-SHUF.
  pick      = argmax over the 64 proposals of agg_torch(logits, lane=False) * p_ok (first index on ties). No tuned parameter.
  C-REG     = p_ok == 1 everywhere -> the pick IS P6's pick -> gain must be EXACTLY 0.0.
  gain      = 100 x (pdms[pick] - pdms[P6 pick]); capture = gain / O_NC+TTC (the S0-B oracle, same sets, same code path).
  estimator = the S0-B paired log-cluster bootstrap (10,000 draws, seed 20260927) over the held-out logs.
usage: python s0c_fusion.py --flags <S0-B run>/s0b_flags.npz --out <dir>
"""
import argparse, hashlib, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import s0b_readout as RD  # noqa: E402

PACK = "F:/tanitad_scratch/p6fit/fit_pack.npz"
PACK_SHA = "e58653c57641a659b5c200dc23a6c59791eb2f053c3a9806b2069b680a7d0a2a"
ARMS = ("F0", "F-R-true", "F-R-t0", "F-R-cv", "F-C-SHUF")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def unpack(F, key):
    return np.unpackbits(F[key], axis=1)[:, :64].astype(bool)


def features(logits, F, arm):
    X = [logits.reshape(-1, 6).astype(np.float64)]
    if arm != "F0":
        src = arm[2:]
        X.append(unpack(F, f"{src}__nc").reshape(-1, 1).astype(np.float64))
        X.append(unpack(F, f"{src}__ttc").reshape(-1, 1).astype(np.float64))
    return np.concatenate(X, axis=1)


def fused_pick(rule, p_ok):
    return np.argmax(rule * p_ok, axis=1)


def outcome(rep, creg_gain, n_ho):
    why = []
    if creg_gain != 0.0:
        why.append(f"C-REG gain {creg_gain!r} != 0.0")
    c0, ct0, csh = rep["F0"]["capture"], rep["F-R-t0"]["capture"], rep["F-C-SHUF"]["capture"]
    if c0 is None:
        return "NOT RUN", ["denominator is zero"]
    if csh - c0 >= 0.10:
        why.append(f"F-C-SHUF capture {csh:.4f} exceeds F0 {c0:.4f} by >= 0.10")
    if why:
        return "INVALID", why
    if ct0 >= 0.25 and rep["F-R-t0"]["gain"] > rep["F0"]["gain"]:
        return "S0C-FUSE-GO", why
    if c0 >= 0.25:
        return "S0C-RERANK", why
    return "S0C-NULL", why


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--flags", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pack", default=PACK)
    a = ap.parse_args(argv)
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    os.makedirs(a.out, exist_ok=True)
    assert sha256(a.pack) == PACK_SHA, "fit pack sha256 differs from the pin"
    sel = RD.load_sel_common(os.path.join(HERE, "ref"))
    F = np.load(a.flags, allow_pickle=False)
    rows, okm, logs = F["rows"], F["ok"], F["logs"]
    assert okm.all(), "S0-B status not ok on every covered set"
    P = np.load(a.pack, allow_pickle=False)
    assert list(P["pdm_names"]) == ["nc", "dac", "ep", "ttc", "c", "ddc", "pdms"]
    logits, pdm, ho = P["logits"][rows], P["pdm"][rows].astype(np.float64), P["is_heldout"][rows]
    assert (P["log"][rows] == logs).all(), "flags/log order mismatch"
    nc, ttc, pdms = pdm[..., 0], pdm[..., 3], pdm[..., 6]
    rule = sel.agg_torch(logits, lane=False)
    p6 = rule.argmax(axis=1)
    y = ((nc > 0) & (ttc > 0)).reshape(-1)
    fit, ev = ~ho, ho
    S = len(rows)
    a_ev = np.arange(int(ev.sum()))
    O_ev = RD.oracle_gains(rule[ev], pdms[ev], nc[ev], ttc[ev], p6[ev])["both"]
    gains, coefs = {}, {}
    for arm in ARMS:
        X = features(logits, F, arm).reshape(S, 64, -1)
        Xf, Xe = X[fit].reshape(-1, X.shape[-1]), X[ev].reshape(-1, X.shape[-1])
        sc = StandardScaler().fit(Xf)
        m = LogisticRegression(C=1.0, max_iter=5000).fit(sc.transform(Xf), y.reshape(S, 64)[fit].reshape(-1))
        p_ok = m.predict_proba(sc.transform(Xe))[:, 1].reshape(-1, 64)
        pk = fused_pick(rule[ev], p_ok)
        gains[arm] = 100.0 * (pdms[ev][a_ev, pk] - pdms[ev][a_ev, p6[ev]])
        coefs[arm] = {"coef_std": m.coef_[0].round(4).tolist(), "intercept": float(m.intercept_[0]),
                      "changed_pick_share": float((pk != p6[ev]).mean())}
    pk_reg = fused_pick(rule[ev], np.ones_like(rule[ev]))
    creg = float(np.max(np.abs(100.0 * (pdms[ev][a_ev, pk_reg] - pdms[ev][a_ev, p6[ev]]))))
    rep = RD.column_report(logs[ev], gains, O_ev)
    row, why = outcome(rep, creg, int(ev.sum()))
    res = {"pack_sha256": PACK_SHA, "flags_sha256": sha256(a.flags), "fit_sets": int(fit.sum()), "fit_logs": int(len(set(logs[fit].tolist()))),
           "eval_sets": int(ev.sum()), "eval_logs": int(len(set(logs[ev].tolist()))), "C-REG_max_abs_gain": creg,
           "heldout": rep, "models": coefs, "outcome": {"row": row, "reasons": why}}
    json.dump(res, open(os.path.join(a.out, "s0c_result.json"), "w", encoding="utf-8"), indent=1)
    lines = [f"# S0-C fusion result (MEASURED; fit {res['fit_sets']} sets / {res['fit_logs']} logs, read on held-out {res['eval_sets']} sets / {res['eval_logs']} logs)", "",
             f"oracle O_NC+TTC on held-out: {rep['O_NC+TTC']['gain']:+.4f} {rep['O_NC+TTC']['ci']}; C-REG max |gain| {creg!r}", "",
             "| arm | gain [95% CI] | capture [95% CI] | changed picks |", "|---|---|---|---|"]
    for arm in ARMS:
        r = rep[arm]
        lines.append(f"| {arm} | {r['gain']:+.4f} {[round(v, 4) for v in r['gain_ci']]} | {r['capture']:.4f} {[round(v, 4) for v in r['capture_ci']]} | {coefs[arm]['changed_pick_share']:.4f} |")
    lines += ["", f"**Outcome: {row}** {why}"]
    open(os.path.join(a.out, "S0C_RESULT.md"), "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
