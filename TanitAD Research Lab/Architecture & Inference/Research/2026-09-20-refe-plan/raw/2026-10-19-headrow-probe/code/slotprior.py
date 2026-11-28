"""SLOT-PRIOR diagnostic (Amendment 5; POST-HOC, UNREGISTERED, DESCRIPTIVE; zero training; decides nothing).  NEW FILE: headrow_probe.py (gate-bound) is imported unchanged.
For each row: score(set, slot j) = mean label of slot j over the VALID entries of the sets of the other four folds (leave-fold-out; EP: safe entries only, the registered convention).
No scene information at all.  Reported: its all-64 OOF concordance (registered estimator), next to the P2 reference and the L arm (seeds 0, 1), plus the paired differences.
  python slotprior.py   -> run/f10/slot_prior.json
"""
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import headrow_probe as H   # noqa: E402

E, np, torch = H.E, H.np, H.torch


def slot_prior_scores(pd, row):
    col = H.ROW_COL[row]
    y = pd.y[col].numpy().astype(np.float64)
    v = pd.valid[col].numpy().astype(np.float64)
    Z = np.empty((pd.n_sets, H.N_PROPOSALS), np.float32)
    spread = {}
    for k in range(E.N_FOLDS):
        tr = pd.fold != k
        num = (y[tr] * v[tr]).sum(0)
        den = v[tr].sum(0)
        pr = np.where(den > 0, num / np.maximum(den, 1.0), 0.0)
        Z[pd.fold == k] = pr.astype(np.float32)
        spread[k] = {"min": float(pr.min()), "max": float(pr.max()), "std": float(pr.std()), "n_slots_without_valid_train_entries": int((den == 0).sum())}
    return Z, spread


def main():
    a = H.build_parser().parse_args(["c2"])
    opts, guard, out, man_sha, ranked_sha, W6, b6, pd, tok = H._setup_stage(a)
    draws = E.boot_draws(pd.n_logs)
    res = {"schema": "headrow-slotprior-v1", "status": "POST-HOC, UNREGISTERED, DESCRIPTIVE (Amendment 5); no training; decides nothing", "rows": {}}
    for row in H.ROWS:
        col = H.ROW_COL[row]
        y, v = pd.y[col], pd.valid[col]
        Zs, spread = slot_prior_scores(pd, row)
        n_s, d_s = H.conc_per_set(Zs, y, v)
        ls, ds = H.per_log(pd, n_s), H.per_log(pd, d_s)
        ent = {"slot_prior": H.ratio_ci(ls, ds, draws), "prior_spread_by_fold": spread}
        n_p, d_p = H.conc_per_set(pd.z6[..., col].numpy(), y, v)
        lp = H.per_log(pd, n_p)
        ent["p2_reference"] = H.ratio_ci(lp, H.per_log(pd, d_p), draws)
        ent["p2_minus_slot"] = H.diff_ci(lp, ls, ds, draws)
        for s in (0, 1):
            z = np.load(io.BytesIO(E._io_read_bytes(guard, os.path.join(out, "oof", f"oof_{row}_L_s{s}.npz"))))["Z"]
            n_l, d_l = H.conc_per_set(z, y, v)
            assert np.array_equal(d_l, d_s)
            ll = H.per_log(pd, n_l)
            ent[f"L_s{s}"] = H.ratio_ci(ll, H.per_log(pd, d_l), draws)
            ent[f"L_s{s}_minus_slot"] = H.diff_ci(ll, ls, ds, draws)
        res["rows"][row] = ent
    nc, ttc = res["rows"]["nc"]["slot_prior"]["point"], res["rows"]["ttc"]["slot_prior"]["point"]
    res["committed_reading"] = {"rule": "slot prior >= 0.80 concordance on NC and on TTC => the C2 / C2x REDs are explained by the slot prior", "nc": nc, "ttc": ttc, "explained": bool(nc >= 0.80 and ttc >= 0.80)}
    E._io_write_bytes(guard, os.path.join(out, "slot_prior.json"), E.canon_json(res), exclusive=True)
    print(json.dumps({r: round(v["slot_prior"]["point"], 4) for r, v in res["rows"].items()}), json.dumps(res["committed_reading"]), flush=True)


if __name__ == "__main__":
    main()
