"""DEFINITIVE cheap downstream-benefit ablation for the YouTube-scale IDM path
(Sayed-greenlit). Settles the go/no-go the label-R2~0.63 proxy could not: do
pseudo-labels ACTUALLY help a (small) world-model, and what FRACTION of the
real-label-pretrain benefit do they capture?

Classic pretraining-benefit ablation in the LOW-real-label regime, on a proxy
domain where we ALSO have real labels (so downstream is measurable). Entirely on
CACHED v1 latents (no re-encode).

  substrate : v1 frozen encoder (the validated cheap substrate)
  model     : a SMALL dynamics model = IDMHead temporal trunk over v1 latent
              windows -> speed + 2s ego-trajectory (the WM's supervised dynamics
              readout; forward-prediction of the ego future). "small WM".
  pipeline  : v1 + IDMHead trained on {rigA+rigB} -> pseudo-label the proxy domain
              (held-out from the labeler) — the exact pseudo-labeling pipeline.

Three arms, PAIRED per seed, IDENTICAL downstream finetune on N_FT real clips:
  FLOOR   : random-init                       -> finetune(real) -> test   (the floor)
  PSEUDO  : pretrain(N_PT clips, PSEUDO labels)-> finetune(real) -> test
  CEILING : pretrain(N_PT clips, REAL  labels) -> finetune(real) -> test   (the ceiling)

KEY NUMBER: fraction_of_ceiling = (PSEUDO - FLOOR) / (CEILING - FLOOR), on speed R2
(and traj ADE), with per-seed CI. PRE-REG: pseudo beats floor (CI-separated) AND
captures >=50% of ceiling -> YouTube-IDM scale-up justified; pseudo ~= floor ->
too noisy, shelve; in-between -> report fraction + gap.

Proxy domains: comma (cross-CLASS YouTube proxy; speed+traj, yaw unreadable) as
PRIMARY; rig-B (richer, same-class) as SECONDARY. pod3, venv python, gpu_lock.
"""
from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path
import torch

sys.path.insert(0, "/workspace/TanitAD/stack")
sys.path.insert(0, "/workspace/TanitAD/stack/scripts")
import idm_head as ih                                              # noqa: E402
import run_idm_proof as R                                          # noqa: E402

V1_LAT = "/workspace/tmp/branchb_eval/lat_flagshipv1"


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def clip_windows(tags, k=4, stride=2):
    out = []
    for t in tags:
        d = torch.load(Path(V1_LAT) / f"{t}.pt", weights_only=False)
        zw, sc, tj = ih.build_windows(d["z"].float(), d["poses"].float(),
                                      d["actions"].float(), k=k, stride=stride)
        if zw.shape[0]: out.append((zw, sc, tj))
    return out


def cat(lst): return (torch.cat([x[0] for x in lst]), torch.cat([x[1] for x in lst]),
                      torch.cat([x[2] for x in lst]))


def train_head(train, device, *, epochs, batch=256, lr=3e-4, wd=0.01, seed=0,
               init_state=None):
    """Fit/continue an IDMHead on (Z, S_target, T_target). init_state warm-starts."""
    torch.manual_seed(seed)
    Z, S, T = train
    std = ih.Standardizer.fit(S)
    head = ih.IDMHead(state_dim=2048, horizons=ih.DEFAULT_HORIZONS).to(device)
    if init_state is not None:
        head.load_state_dict(init_state)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    n = Z.shape[0]
    Z, S, T = Z.to(device), S.to(device), T.to(device)   # full set on GPU once (small)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * max(1, n // batch))
    g = torch.Generator(device=device).manual_seed(seed + 1)
    for _ in range(epochs):
        head.train(); perm = torch.randperm(n, generator=g, device=device)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            ld = ih.idm_loss(head(Z[idx]), S[idx], T[idx], std)
            opt.zero_grad(set_to_none=True); ld["loss"].backward(); opt.step(); sched.step()
    head.eval()
    return head


@torch.no_grad()
def pseudo_targets(labeler, clips, device, b=2048):
    """Replace each clip's (scalars, traj) with the labeler head's predictions
    (raw units) -> the PSEUDO targets. Returns clips in the same per-clip shape."""
    out = []
    for zw, sc, tj in clips:
        ps, pt = [], []
        for i in range(0, zw.shape[0], b):
            o = labeler(zw[i:i + b].to(device)); ps.append(o["scalars"].cpu()); pt.append(o["traj"].cpu())
        out.append((zw, torch.cat(ps), torch.cat(pt)))
    return out


@torch.no_grad()
def evaluate(head, clips, device):
    m = ih.evaluate(head, *cat(clips), device=device)
    return {"speed_r2": m["r2"]["speed"], "yaw_r2": m["r2"]["yaw_rate"],
            "ade_2s": m["ade_2s"], "speed_mae": m["mae"]["speed"], "n": m["n"]}


def mean_std(xs):
    import statistics as st
    return (round(st.mean(xs), 4), round(st.pstdev(xs), 4))


def run_domain(name, labeler, pt_tags, ft_tags, te_tags, device, seeds,
               pt_epochs, ft_epochs):
    """One proxy domain: pretrain(pseudo/real) -> finetune(real) -> test, per seed."""
    pt_real = clip_windows(pt_tags)                    # pretrain clips, REAL targets
    pt_pseudo = pseudo_targets(labeler, pt_real, device)   # same clips, PSEUDO targets
    ft = clip_windows(ft_tags)                         # finetune clips, REAL
    te = clip_windows(te_tags)                         # test clips, REAL
    log(f"[{name}] pretrain {len(pt_real)} clips ({cat(pt_real)[0].shape[0]} win), "
        f"finetune {len(ft)} ({cat(ft)[0].shape[0]} win), test {len(te)} ({cat(te)[0].shape[0]} win)")
    arms = {"floor": [], "pseudo": [], "ceiling": []}
    per_seed = []
    for s in seeds:
        # FLOOR: random init -> finetune(real) only
        h_floor = train_head(cat(ft), device, epochs=ft_epochs, seed=s)
        # PSEUDO: pretrain(pseudo) -> finetune(real)
        h_pt_ps = train_head(cat(pt_pseudo), device, epochs=pt_epochs, seed=s)
        h_pseudo = train_head(cat(ft), device, epochs=ft_epochs, seed=s,
                              init_state={k: v.clone() for k, v in h_pt_ps.state_dict().items()})
        # CEILING: pretrain(real) -> finetune(real)
        h_pt_re = train_head(cat(pt_real), device, epochs=pt_epochs, seed=s)
        h_ceiling = train_head(cat(ft), device, epochs=ft_epochs, seed=s,
                               init_state={k: v.clone() for k, v in h_pt_re.state_dict().items()})
        e = {a: evaluate(h, te, device) for a, h in
             (("floor", h_floor), ("pseudo", h_pseudo), ("ceiling", h_ceiling))}
        for a in arms: arms[a].append(e[a]["speed_r2"])
        per_seed.append({"seed": s, **{a: {"speed_r2": e[a]["speed_r2"], "ade_2s": e[a]["ade_2s"]} for a in e}})
        log(f"  [{name}] seed {s}: floor speedR2 {e['floor']['speed_r2']:+.3f} | "
            f"pseudo {e['pseudo']['speed_r2']:+.3f} | ceiling {e['ceiling']['speed_r2']:+.3f}")
    # paired benefits per seed (speed R2)
    fl = arms["floor"]; ps = arms["pseudo"]; ce = arms["ceiling"]
    ps_ben = [ps[i] - fl[i] for i in range(len(seeds))]
    ce_ben = [ce[i] - fl[i] for i in range(len(seeds))]
    fracs = [ps_ben[i] / ce_ben[i] for i in range(len(seeds)) if abs(ce_ben[i]) > 1e-6]
    import statistics as st
    res = {"n_seeds": len(seeds),
           "speed_r2_mean_std": {a: mean_std(arms[a]) for a in arms},
           "pseudo_benefit_mean_std": mean_std(ps_ben),
           "ceiling_benefit_mean_std": mean_std(ce_ben),
           "pseudo_benefit_per_seed": [round(x, 4) for x in ps_ben],
           "ceiling_benefit_per_seed": [round(x, 4) for x in ce_ben],
           "fraction_of_ceiling_mean": round(st.mean(fracs), 3) if fracs else None,
           "fraction_of_ceiling_per_seed": [round(x, 3) for x in fracs],
           "pseudo_beats_floor_all_seeds": all(b > 0 for b in ps_ben),
           "ceiling_beats_floor_all_seeds": all(b > 0 for b in ce_ben),
           "per_seed": per_seed}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flagship-ckpt", default="/workspace/tmp/idm/ckpt.pt")
    ap.add_argument("--work", default="/workspace/tmp/idm_ablation")
    ap.add_argument("--out", default="/workspace/tmp/idm_ablation/results_idm_downstream_ablation.json")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--pt-epochs", type=int, default=40)
    ap.add_argument("--ft-epochs", type=int, default=60)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Path(args.work).mkdir(parents=True, exist_ok=True); t0 = time.time()
    fs_enc, fs_ro, _ = R.load_encoder(args.flagship_ckpt, device)  # only for md5/meta parity

    TR_A = [f"tr_a_{i:05d}" for i in range(100)]
    TR_B = [f"tr_b_{i:05d}" for i in range(120)]
    CM = [f"cm_{i:05d}" for i in range(80)]
    VA_B = [f"va_b_{i:05d}" for i in range(54)]
    seeds = list(range(args.seeds))

    res = {"meta": {"experiment": "idm_downstream_benefit_ablation",
                    "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "flagship_md5": R.md5_of(args.flagship_ckpt), "device": device,
                    "seeds": args.seeds, "pt_epochs": args.pt_epochs, "ft_epochs": args.ft_epochs,
                    "model": "IDMHead temporal trunk over frozen v1 latents (small dynamics WM readout)",
                    "metric": "downstream speed R2 (primary) + traj ADE, on held-out real-labeled test",
                    "pre_reg": "pseudo beats floor (all seeds) AND fraction_of_ceiling>=0.5 -> GO; "
                               "pseudo~=floor -> NO-GO; else report fraction+gap"},
           "domains": {}}

    # ---- PRIMARY: comma (cross-CLASS YouTube proxy). Labeler = v1 + {rigA+rigB} head ----
    log("labeler head {rigA+rigB} for comma pseudo-labels ...")
    lab_comma = train_head(cat(clip_windows(TR_A[:60] + TR_B[:60])), device, epochs=50, seed=0)
    res["domains"]["comma_crossCLASS"] = run_domain(
        "comma", lab_comma, CM[:50], CM[50:60], CM[60:80], device, seeds,
        args.pt_epochs, args.ft_epochs)

    # ---- SECONDARY: rig-B (richer, same-class). Labeler = v1 + {rigA+comma} head ----
    log("labeler head {rigA+comma} for rigB pseudo-labels ...")
    lab_rigb = train_head(cat(clip_windows(TR_A[:60] + CM[:40])), device, epochs=50, seed=0)
    # pretrain on rigB-train, finetune+test on rigB-val (episode-disjoint)
    res["domains"]["rigB_sameClass"] = run_domain(
        "rigB", lab_rigb, TR_B[:50], VA_B[:12], VA_B[12:54], device, seeds[:3],
        args.pt_epochs, args.ft_epochs)

    # ---- verdicts ----
    def verdict(d):
        frac = d["fraction_of_ceiling_mean"]
        sep = d["pseudo_beats_floor_all_seeds"]
        ceil_ok = d["ceiling_beats_floor_all_seeds"]
        if not ceil_ok:
            call = "INCONCLUSIVE (ceiling did not beat floor -> pretraining signal too weak to measure fraction)"
        elif sep and frac is not None and frac >= 0.5:
            call = "GO (pseudo beats floor all seeds AND captures >=50% of ceiling)"
        elif not sep:
            call = "NO-GO (pseudo does not beat floor -> labels too noisy)"
        else:
            call = f"IN-BETWEEN (pseudo beats floor; captures {frac:.0%} of ceiling)"
        return {"fraction_of_ceiling": frac, "pseudo_beats_floor_all_seeds": sep,
                "ceiling_beats_floor_all_seeds": ceil_ok, "call": call,
                "speed_r2": d["speed_r2_mean_std"]}
    res["verdicts"] = {k: verdict(v) for k, v in res["domains"].items()}

    Path(args.out).write_text(json.dumps(res, indent=2))
    for k, v in res["verdicts"].items():
        s = v["speed_r2"]
        log(f"VERDICT [{k}]: floor {s['floor']} pseudo {s['pseudo']} ceiling {s['ceiling']} | "
            f"frac_of_ceiling {v['fraction_of_ceiling']} | {v['call']}")
    log(f"WROTE {args.out}  ({time.time()-t0:.0f}s)")
    log("IDM_DOWNSTREAM_ABLATION_DONE")


if __name__ == "__main__":
    main()
