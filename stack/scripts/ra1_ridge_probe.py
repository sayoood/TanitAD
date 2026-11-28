"""RA-1 ridge probe (PREREG_RA1_RIDGE.md): is there ANY linearly readable signal in the pooled frozen DINOv3 features that improves
on the frozen prior P10-yaw-540 at 6 s, at RA-1's data scale (540 train clips)?

Why: the RA-1 budget sweep (PREREG_RA1_ES) found R0's MLP readout best at 250 steps, where it is still ~P10, and monotonically WORSE
with every further step. That says the MLP learns nothing that transfers; it does not say whether the features carry transferable
signal that an MLP of that capacity cannot extract from 540 clips without memorising. A ridge map with its penalty chosen on the
withheld holdout is the lowest-capacity readout there is: if it beats P10, the signal is there and the readout is the defect.

Everything is the RA-1 eval's own machinery, imported, never re-implemented: `ra1_pooled.ClipBank` / `split_train_holdout` /
`eval_windows`, the P10 prior via `load_p10(..., "yaw")`, the native yaw-rate tracks, `taniteval.ci` and `four_families`.

* Target: the RESIDUAL `gt - p10_traj` [30 x 2] at each window (a zero residual IS P10).
* Features (each fit separately, all standardised on the TRAIN windows):
    f8      the 8 standardised P10 features only          (CONTROL: no vision; the prior's own inputs)
    mean    the 4x10 pooled tokens averaged over the 40 cells = the 1024-d patch mean   (vision only)
    mean+f8 both
  plus `bias`: the mean train residual alone (a constant offset; the prior-correction control).
* Ridge: closed form on the centred train windows, intercept = the train residual mean. Penalty lambda * n_train on a log grid.
* SELECTION: per feature set, the lambda with the lowest holdout ADE@6s (60 clips, PREREG_HJ_STAGE0 Amendment 1). eval-141 scores
  each feature set ONCE at its selected lambda. lambda -> infinity is `bias`, so the holdout can always fall back to it.
* Controls that must hold: P10's own ADE@6s reproduces the banked 2.5627 (eval-141) and 3.0245 (holdout) to 1e-3 m; the holdout
  and train clip lists reproduce RA-1's sha256s; train/holdout/eval are pairwise disjoint.

Tier: T1-class open-loop trajectory (the readout sees the t0 observation and t0 kinematics only). Estimator: paired episode-cluster
bootstrap (episode variance only). The probe is deterministic given the data, so there is no training-seed or inference-seed floor.
⚠️ A negative from a linear probe is NOT a negative about learnability (TRAPS.md); it is read together with the ES sweep."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in (os.path.join(REPO, "stack"), os.path.join(REPO, "taniteval")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from tanitad.data import ra1_pooled as RD  # noqa: E402
from tanitad.data import v8f_mini_data as D  # noqa: E402
from tanitad.models import ra1_hjepa as R  # noqa: E402
from tanitad.models.v8f_mini import load_p10  # noqa: E402
from taniteval import ci  # noqa: E402
from taniteval import floors_kin as fk  # noqa: E402

FEATURE_SETS = ("f8", "mean", "mean+f8")
BANKED = dict(eval_p10_ade6=2.5627, hold_p10_ade6=3.0245,
              hold_sha="f6a1dee02af0b96f815416bfef9bc11a04554702525135aa75a740942d7750f9",
              train_sha="e386600fc37787bfb318ddd46568ab5a823d842bc55ec9d87c105af490c947e5")
PAUSE = {"status": "PAUSED", "ruling": "PI 2026-09-19, PI_DECISION_QUEUE.md ITEM 25: the strategic layer is switched off temporarily "
         "until tactical and operative drive with high quality. Reported as PAUSED -- not a pass, not omitted."}


def stat(x, eid, nb, sd):
    r = ci.episode_cluster_bootstrap(np.asarray(x, np.float64), eid, n_boot=nb, seed=sd)
    return {k: r[k] for k in ("mean", "lo", "hi", "ci95", "n_windows", "n_episodes", "estimator") if k in r}


def pair(a, b, eid, nb, sd):
    r = ci.paired_episode_cluster_bootstrap(np.asarray(a, np.float64), np.asarray(b, np.float64), eid, n_boot=nb, seed=sd)
    return {k: r[k] for k in ("delta", "lo", "hi", "separated", "n_windows", "n_episodes", "estimator") if k in r}


def gather(bank, wins):
    """windows -> (features dict, p10 [N,30,2], gt [N,30,2], episode ids)."""
    order = list(dict.fromkeys(c for c, _ in wins))
    cidx = {c: i for i, c in enumerate(order)}
    mean = torch.stack([bank.clips[c]["pool"][t].float().mean(0) for c, t in wins]).double().numpy()
    f8 = torch.stack([bank.clips[c]["f8n"][t] for c, t in wins]).double().numpy()
    p10 = torch.stack([bank.clips[c]["p10_traj"][t] for c, t in wins]).double().numpy()
    gt = torch.stack([bank.clips[c]["gt"][t] for c, t in wins]).double().numpy()
    feats = {"f8": f8, "mean": mean, "mean+f8": np.concatenate([mean, f8], 1)}
    return feats, p10, gt, np.array([cidx[c] for c, _ in wins])


class Ridge:
    """Closed-form ridge on standardised features; intercept = the train target mean; penalty = lam * n."""

    def __init__(self, X, Y):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-8
        Z = (X - self.mu) / self.sd
        self.ym = Y.mean(0)
        self.G, self.B, self.n = Z.T @ Z, Z.T @ (Y - self.ym), len(Z)

    def coef(self, lam):
        return np.linalg.solve(self.G + lam * self.n * np.eye(len(self.G)), self.B)

    def predict(self, X, W):
        return ((X - self.mu) / self.sd) @ W + self.ym


def run(a) -> dict:
    t0 = time.time()
    torch.set_num_threads(a.threads)
    prior = load_p10(a.p10_prior, "yaw")
    tracks = D.load_native_tracks(a.tracks)
    train_all = [c for c, _ in RD.list_clips(a.cache, "train")]
    tr, hold = RD.split_train_holdout(train_all, 0, a.holdout_n, RD.STAGE_A_CLIPS)
    ev_order = [c for c, _ in RD.list_clips(a.cache, "eval")]
    ctl = dict(hold_sha_ok=RD.clips_sha256(hold) == BANKED["hold_sha"], train_sha_ok=RD.clips_sha256(tr) == BANKED["train_sha"],
               train_hold_overlap=len(set(tr) & set(hold)), train_eval=D.assert_disjoint(tr, ev_order),
               hold_eval=D.assert_disjoint(hold, ev_order))
    bank = RD.ClipBank(a.cache, "train", prior, "4x10", "yaw", tracks)
    bank.load(tr + hold)
    Xtr, p10tr, gttr, _ = gather(bank, RD.eval_windows(bank, tr, None, a.train_stride))
    Xh, p10h, gth, eidh = gather(bank, RD.eval_windows(bank, hold, None, 1))
    bank.keep([])
    ebank = RD.ClipBank(a.cache, "eval", prior, "4x10", "yaw", tracks)
    ebank.load(ev_order)
    Xe, p10e, gte, eide = gather(ebank, RD.eval_windows(ebank, ev_order, a.grid_json, 1))
    Ytr = (gttr - p10tr).reshape(len(gttr), -1)
    hp6, ep6 = fk.ade(p10h, gth), fk.ade(p10e, gte)
    ctl.update(hold_p10_ade6=float(hp6.mean()), eval_p10_ade6=float(ep6.mean()),
               p10_reproduces=bool(abs(hp6.mean() - BANKED["hold_p10_ade6"]) < 1e-3 and abs(ep6.mean() - BANKED["eval_p10_ade6"]) < 1e-3))
    ctl["ok"] = bool(ctl["hold_sha_ok"] and ctl["train_sha_ok"] and ctl["train_hold_overlap"] == 0 and ctl["train_eval"]["overlap"] == 0
                     and ctl["hold_eval"]["overlap"] == 0 and ctl["p10_reproduces"])
    out = dict(schema="ra1-ridge-probe-v1", prereg="PREREG_RA1_RIDGE.md", tier="T1-class open-loop trajectory",
               estimator="paired episode-cluster bootstrap (EPISODES); the probe is deterministic (no seed floors)",
               n_train_windows=len(Ytr), n_train_clips=len(tr), n_hold_windows=len(gth), n_eval_windows=len(gte),
               n_eval_episodes=int(len(set(eide.tolist()))), controls=ctl, lambdas=a.lambdas, sets={})
    bias = Ytr.mean(0)
    nb, sd = a.n_boot, a.seed

    def score(res_h, res_e):
        ph, pe = p10h + res_h.reshape(p10h.shape), p10e + res_e.reshape(p10e.shape)
        return ph, pe, fk.ade(ph, gth), fk.ade(pe, gte)

    _, pe_b, h6_b, e6_b = score(np.broadcast_to(bias, (len(gth), bias.size)), np.broadcast_to(bias, (len(gte), bias.size)))
    out["sets"]["bias"] = dict(selected_lambda="inf", hold_ade6=float(h6_b.mean()), hold_minus_p10=pair(h6_b, hp6, eidh, nb, sd),
                               eval_ade6=stat(e6_b, eide, nb, sd), eval_minus_p10=pair(e6_b, ep6, eide, nb, sd))
    best_pred = {}
    for fs in FEATURE_SETS:
        rg = Ridge(Xtr[fs], Ytr)
        table = {}
        for lam in a.lambdas:
            W = rg.coef(lam)
            table[lam] = float(fk.ade(p10h + rg.predict(Xh[fs], W).reshape(p10h.shape), gth).mean())
        lam_s = min(table, key=table.get)
        W = rg.coef(lam_s)
        _, pe, h6, e6 = score(rg.predict(Xh[fs], W), rg.predict(Xe[fs], W))
        tr6 = fk.ade(p10tr + rg.predict(Xtr[fs], W).reshape(p10tr.shape), gttr)
        out["sets"][fs] = dict(d_in=int(Xtr[fs].shape[1]), hold_ade6_by_lambda=table, selected_lambda=lam_s,
                               selected_at_grid_edge=lam_s in (min(a.lambdas), max(a.lambdas)),
                               train_ade6=float(tr6.mean()), train_p10_ade6=float(fk.ade(p10tr, gttr).mean()),
                               hold_ade6=float(h6.mean()), hold_minus_p10=pair(h6, hp6, eidh, nb, sd),
                               hold_minus_bias=pair(h6, h6_b, eidh, nb, sd),
                               eval_ade6=stat(e6, eide, nb, sd), eval_minus_p10=pair(e6, ep6, eide, nb, sd),
                               eval_minus_bias=pair(e6, e6_b, eide, nb, sd),
                               eval_ade2_minus_p10=pair(fk.ade(pe[:, :10], gte[:, :10]), fk.ade(p10e[:, :10], gte[:, :10]), eide, nb, sd))
        best_pred[fs] = pe
    # vision adds over the no-vision control at the SAME selection rule (the leading question)
    out["vision_minus_f8_eval"] = pair(fk.ade(best_pred["mean+f8"], gte), fk.ade(best_pred["f8"], gte), eide, nb, sd)
    # four families for the vision probe and P10 on the eval windows (EVAL_DOCTRINE / the four-family rule)
    fam = {}
    try:
        from taniteval import four_families as FF
        for nm, path in (("ridge_mean+f8", best_pred["mean+f8"]), ("p10", p10e)):
            win = dict(pred=torch.from_numpy(path).float(), gt=torch.from_numpy(gte).float(), wp_steps=list(range(1, 31)), dt_s=R.DT,
                       eid=eide)
            f = FF.all_families(win, tactical_from_traj=True, tier="T1", n_boot=a.fam_boot, seed=sd)
            f["strategic"] = dict(PAUSE, underlying=f.get("strategic"))
            fam[nm] = f
    except Exception as e:                                                        # noqa: BLE001 -- reported, not hidden
        fam = dict(status="ERROR", error=repr(e)[:500], reason="four_families could not run; this is a WORK ITEM, not a pass")
    fam["lead_braking_stratum_and_distance_keeping"] = ("NOT COMPUTABLE from the pooled cache: it carries no lead-agent track "
                                                       "(needs the v9 join lead_gap); n = %d windows would have been scored" % len(gte))
    out["families"] = fam
    out["reading"] = reading(out)
    out["seconds"] = time.time() - t0
    return out


def reading(o) -> str:
    """PREREG_RA1_RIDGE's readings, in its order."""
    if not o["controls"]["ok"]:
        return "INSTRUMENT: a control failed (see `controls`); nothing is read"
    v = o["sets"]["mean+f8"]
    d, dv = v["eval_minus_p10"], o["vision_minus_f8_eval"]
    if d["separated"] and d["delta"] < 0 and dv["separated"] and dv["delta"] < 0:
        return ("SIGNAL: the ridge readout of the pooled features beats P10 at 6 s, separated (%+.4f [%+.4f, %+.4f]), and vision adds over "
                "the no-vision control (%+.4f [%+.4f, %+.4f]). The features carry transferable signal at 540 clips; the MLP readout is "
                "the defect. Next: constrain R0's readout (low-rank / ridge-initialised / strong weight decay), then the WM arms."
                % (d["delta"], d["lo"], d["hi"], dv["delta"], dv["lo"], dv["hi"]))
    if d["separated"] and d["delta"] < 0:
        return ("PRIOR CORRECTION ONLY: the ridge readout beats P10 (%+.4f [%+.4f, %+.4f]) but vision does not add over the no-vision "
                "control (%+.4f [%+.4f, %+.4f]). The gain is a correction of P10 from its own inputs, not vision."
                % (d["delta"], d["lo"], d["hi"], dv["delta"], dv["lo"], dv["hi"]))
    return ("NO LINEAR SIGNAL at 540 clips: the holdout-selected ridge readout does not beat P10 (%+.4f [%+.4f, %+.4f]). Together "
            "with the ES sweep (no budget beats P10), the frozen pooled features add nothing a 540-clip readout can use. Next lever: "
            "data scale (PREREG_HJ_STAGE0: all 4,572 clips). A linear negative is not a learnability negative (TRAPS.md)."
            % (d["delta"], d["lo"], d["hi"]))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--p10-prior", required=True)
    ap.add_argument("--grid-json", required=True)
    ap.add_argument("--holdout-n", type=int, required=True)
    ap.add_argument("--train-stride", type=int, default=1)
    ap.add_argument("--lambdas", type=float, nargs="+", default=[1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0, 100.0])
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--fam-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    res = run(a)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, default=float)
    print(json.dumps(dict(controls=res["controls"], reading=res["reading"],
                          sets={k: dict(lam=v.get("selected_lambda"), hold=v.get("hold_ade6"), eval_minus_p10=v.get("eval_minus_p10"))
                                for k, v in res["sets"].items()}), indent=1, default=float))
    return 0 if res["controls"]["ok"] else 3


if __name__ == "__main__":
    sys.exit(main())
