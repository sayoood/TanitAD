"""Four-metric-family evaluation (CLAUDE.md BINDING): longitudinal, lateral, tactical, strategic -- per family, never pooled.
Per-window metrics against the logged future; full-set means with episode-cluster bootstrap CIs (taniteval.ci), paired against the
hold-current (CTRA) control on the SAME windows. Families/metrics that cannot be computed are reported UNAVAILABLE with reason and n."""
from __future__ import annotations
import numpy as np
import torch
from . import vocab as V
from .decode import decide, CascadeCfg


def hold_current(v0, yaw):
    t = (np.arange(V.T_STEPS) + 1) * V.DT; th = yaw * t
    return np.stack([np.cumsum(v0 * np.cos(th) * V.DT), np.cumsum(v0 * np.sin(th) * V.DT)], 1).astype(np.float32)


def _heading(tr, i): d = tr[i] - tr[i - 1]; return np.arctan2(d[1], d[0])


def window_metrics(pred: np.ndarray, fut: np.ndarray, v0: float) -> dict:
    """pred, fut [60,2] ego-frame. 2 s = index 19, 4 s = 39."""
    i2, i4 = 19, 39
    err = np.linalg.norm(pred - fut, axis=1)
    sp = lambda tr, i: np.linalg.norm(tr[i] - tr[i - 5]) / (5 * V.DT)
    yr = lambda tr, i: float(np.angle(np.exp(1j * (_heading(tr, i) - _heading(tr, i - 5)))) / (5 * V.DT))
    cx = lambda tr: tr[i2, 1]
    return dict(
        ade_2s=float(err[:i2 + 1].mean()), fde_2s=float(err[i2]), ade_6s=float(err.mean()),
        speed_mae=abs(sp(pred, i2) - sp(fut, i2)), speed_ok_0p5=float(abs(sp(pred, i2) - sp(fut, i2)) <= 0.5),
        along_track_mae=abs(pred[i2, 0] - fut[i2, 0]),
        heading_err_deg=float(np.degrees(abs(np.angle(np.exp(1j * (_heading(pred, i2) - _heading(fut, i2))))))),
        yaw_rate_err=abs(yr(pred, i2) - yr(fut, i2)), cross_track=abs(cx(pred) - cx(fut)),
    )


FAMILY = dict(longitudinal=("speed_mae", "speed_ok_0p5", "along_track_mae"), lateral=("heading_err_deg", "yaw_rate_err", "cross_track"),
              trajectory=("ade_2s", "fde_2s", "ade_6s"))


def _ci(vals, eid, seed=0):
    try:
        from taniteval.ci import episode_cluster_bootstrap
        return episode_cluster_bootstrap(vals, eid, n_boot=2000, seed=seed)
    except Exception:
        vals = np.asarray(vals, float); u = np.unique(eid); rng = np.random.default_rng(seed)
        m = np.array([vals[np.isin(eid, rng.choice(u, len(u)))].mean() for _ in range(500)])
        return dict(mean=float(vals.mean()), lo=float(np.percentile(m, 2.5)), hi=float(np.percentile(m, 97.5)), estimator="episode_cluster_bootstrap(fallback B=500)")


def _paired(a, b, eid, seed=0):
    from taniteval.ci import paired_episode_cluster_bootstrap
    return paired_episode_cluster_bootstrap(a, b, eid, n_boot=2000, seed=seed)


@torch.no_grad()
def evaluate(model, Vt, vocab, ds, n_max: int | None = None, decider: str = "argmax", cfg: CascadeCfg = CascadeCfg(), device="cpu") -> dict:
    model.eval(); n = len(ds) if n_max is None else min(n_max, len(ds)); idx = np.linspace(0, len(ds) - 1, n).astype(int)
    rows = []; tact = []
    for i in idx:
        it = ds[int(i)]; b = {k: (v.unsqueeze(0).to(device) if torch.is_tensor(v) else v) for k, v in it.items()}
        v0 = float(it["v0"]); yaw = float(it["ego"][2]); fut = it["fut"].numpy()
        if decider == "cascade":
            out = decide(model, Vt, vocab, b, v0, float(it["ego"][1] / 0.2), yaw, cfg, nav=int(it["nav"]))
            pred, cell = out["traj"], out["cell"]
        else:
            from .losses import node_embeddings
            s, _, _, _ = model.encode(b); A = node_embeddings(model, Vt)
            p = int((s["s3P"][0] @ A["s3P"].t()).argmax()); q = int((s["s3V"][0] @ A["s3V"].t()).argmax())
            pred = V.compose(vocab.paths[p], vocab.profiles[q], v0); cell = (int(vocab.path_cls[p]), int(vocab.prof_cls[q]))
        m = window_metrics(pred, fut, v0); h = window_metrics(hold_current(v0, yaw), fut, v0)
        rows.append((m, h, int(it["eid"]), cell, int(it["lat_y"]), int(it["lon_y"])))
    eid = np.array([r[2] for r in rows]); out = {"n_windows": len(rows), "n_episodes": int(len(np.unique(eid))), "decider": decider, "families": {}}
    for fam, names in FAMILY.items():
        out["families"][fam] = {}
        for nm in names:
            a = np.array([r[0][nm] for r in rows]); b = np.array([r[1][nm] for r in rows])
            out["families"][fam][nm] = dict(model=_ci(a, eid), hold=_ci(b, eid), paired_model_minus_hold=_paired(a, b, eid))
    ok = [(r[3], r[4], r[5]) for r in rows if r[4] >= 0 and r[5] >= 0]
    out["families"]["tactical"] = dict(
        n_labelled=len(ok), lat_acc=float(np.mean([c[0] == l for c, l, _ in ok])) if ok else None,
        lon_acc=float(np.mean([c[1] == o for c, _, o in ok])) if ok else None,
        cell_acc=float(np.mean([c[0] == l and c[1] == o for c, l, o in ok])) if ok else None,
        note="decision-level accuracy of the path/profile classes; joint cell accuracy; paired CI to be added with the nav-zero rows")
    out["families"]["strategic"] = dict(status="UNAVAILABLE", n=0, reason="strategic labels (8-30 s) are not in the v9 release; five usable strategic actions only in the v7/v8.1 release")
    out["families"]["longitudinal"]["headway_ttc"] = dict(status="UNAVAILABLE", reason="no predicted-vs-logged lead-agent state in this harness; needs the agent tracks join (cost heads)")
    return out
