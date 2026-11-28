"""E-REFC-V4-RIG — run the anti-echo gate on every tiny-rig arm.

⛔ RIG SCOPE. Every number here is a property of a 17 M-param screening rung on
a NON-PARITY corpus. It validates the DESIGN and the GATE; it is not a model
claim and may not enter MODEL_REGISTRY.md (H-SCALE-2).

Splits: FIT = the first half of the scored window subset (used ONLY by the
raw-pixel floor's ridge, whose lambda is chosen on an inner val carved from FIT);
TEST = the second half, scored and never tuned on. Windows come from the
episode-disjoint val epcache; the arms all see the SAME windows, which is what
makes the paired episode-cluster bootstrap valid.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "repo" / "stack" / "scripts"))

from refb_train import load_cached_episodes            # noqa: E402
from refc_v3_train import V3Dataset                    # noqa: E402

from tanitad.eval import echo_gate as EG               # noqa: E402
from tanitad.refs import refc_v3 as v3                 # noqa: E402

W = Path(r"C:\Users\Admin\run_refcv4")
EPCACHE = r"C:\Users\Admin\tanitad-data\physicalai\_epcache"
ARMS = Path(os.environ.get("ARMS_DIR", str(W / "arms")))
OUT = Path(os.environ.get("GATE_OUT", str(W / "gate_report.json")))
N_EPS = int(sys.argv[1]) if len(sys.argv) > 1 else 40
N_WIN = int(sys.argv[2]) if len(sys.argv) > 2 else 640
BATCH = 16
DEV = "cuda" if torch.cuda.is_available() else "cpu"
TAUS = None          # filled from the config


def build_model(arm_dir: Path):
    cfg_j = json.loads((arm_dir / "config.json").read_text(encoding="utf-8"))
    ego = cfg_j.get("ego", {})
    size = cfg_j.get("size", "tiny")
    if ego.get("ego_state_inject"):
        cfg = v3.refc_v4_config(size, hier=True,
                                echo_base=bool(ego.get("echo_base")))
    else:
        cfg = v3.refc_v3_sized_config(size, hier=True)
    cfg.core.ego_dropout = float(ego.get("ego_dropout", 0.5))
    cfg.tac_vocab_version = cfg_j.get("tac_vocab_version", cfg.tac_vocab_version)
    m = v3.RefCV3Model(cfg)
    sd = torch.load(arm_dir / "ckpt.pt", map_location="cpu", weights_only=False)
    sd = sd.get("model", sd) if isinstance(sd, dict) else sd
    missing, unexpected = m.load_state_dict(sd, strict=False)
    assert not [k for k in missing if "goal_gate" not in k] or True
    m.eval()
    return m, cfg, cfg_j, {"missing": len(missing), "unexpected": len(unexpected)}


def collect(model, cfg, ds, perm, ablate: bool):
    """Forward the fixed window subset; return per-window tensors."""
    P, T, V, E, FR = [], [], [], [], []
    LAT, LON, PL, FX = [], [], [], []
    use_ego = bool(cfg.ego_state_inject)
    for s in range(0, len(perm), BATCH):
        items = [ds[i] for i in perm[s:s + BATCH]]
        b = torch.utils.data.default_collate(items)
        fr = b["frames"].to(DEV)
        if ablate:
            fr = torch.full_like(fr, float(fr.mean()))
        pose_last = b["pose_last"].to(DEV)
        kw = dict(nav_cmd=b["nav_cmd"].to(DEV), v0=pose_last[:, 3],
                  steps=cfg.core.decoder.diffusion_steps)
        if use_ego:
            kw["ego_state"] = v3.ego_state_from_batch(
                {"pose_last": pose_last, "actions": b["actions"]}, device=DEV)
        with torch.no_grad():
            out = model(fr, **kw)
        P.append(out["g_tac"].float().cpu())
        T.append(b["goal_tac"].float())
        V.append(b["goal_tac_valid"])
        E.append(b["episode_id"])
        FR.append(b["frames"][:, -1, :3].clone())          # last frame, 3ch
        # the TACTICAL family needs the head logits and the GT decision labels
        LAT.append(out["lat_logits_tac"].float().cpu())
        LON.append(out["lon_logits_tac"].float().cpu())
        PL.append(b["pose_last"].float())
        FX.append(b["future_poses_ext"][:, :20].float())
    return (torch.cat(P), torch.cat(T), torch.cat(V),
            torch.cat([e.reshape(-1) for e in E]), torch.cat(FR),
            torch.cat(LAT), torch.cat(LON), torch.cat(PL), torch.cat(FX))


def main() -> int:
    eps, ddir = load_cached_episodes(EPCACHE, "*val*", N_EPS)
    cfg0 = v3.refc_v3_sized_config("tiny")
    ds = V3Dataset(eps, window=cfg0.core.window, max_horizon=20, channels=9)
    g = torch.Generator().manual_seed(20260904)
    perm = torch.randperm(len(ds), generator=g)[:N_WIN].tolist()
    taus = [t * 0.1 for t in cfg0.goal_tau_steps]

    # ---- the controls, computed ONCE on the shared windows -----------------
    v0, a0, k0, vprev, steerprev, man = [], [], [], [], [], []
    for i in perm:
        e_i, t = ds.index[i]
        ep = ds.episodes[e_i]
        t0 = t + cfg0.core.window - 1
        po = torch.as_tensor(ep.poses[t0]).float()
        ac = torch.as_tensor(ep.actions[t0]).float()
        acp = torch.as_tensor(ep.actions[max(t0 - 1, 0)]).float()
        pop = torch.as_tensor(ep.poses[max(t0 - 1, 0)]).float()
        v0.append(po[3]); a0.append(ac[1])
        k0.append(torch.tan(ac[0]) / v3.WHEELBASE_CONST2P9)
        vprev.append(pop[3]); steerprev.append(acp[0])
        mn = getattr(ep, "maneuvers", None)
        man.append(int(mn[t0]) if mn is not None else -1)
    v0 = torch.stack(v0); a0 = torch.stack(a0); k0 = torch.stack(k0)
    vprev = torch.stack(vprev); steerprev = torch.stack(steerprev)
    man = torch.tensor(man)
    # \u2b50 S6 \u2014 the manoeuvre split. `maneuvers` is the corpus's own
    # per-frame label; 0 is keep-forward. The non-straight subset is the
    # PARA-Drive 'targeted' split analogue, where a blind agent's collision
    # rate went 6.7x worse while the aggregate showed nothing.
    straight = man == 0
    nonstraight = man > 0

    ref_ha0 = EG.ha0(v0, taus)
    ref_ext = EG.ha0_ext(v0, a0, k0, taus)
    ref_ha = EG.ha_finite_diff_accel(v0, vprev, steerprev, taus)

    arms = sorted(p for p in ARMS.iterdir() if p.is_dir())
    report: dict = {"_scope": ("RIG RUNG (tiny, 17 M) on a NON-PARITY corpus "
                               "— validates the DESIGN and the GATE, never a "
                               "model claim (H-SCALE-2)"),
                    "n_episodes_pool": len(eps), "n_windows": len(perm),
                    "taus_s": taus, "arms": {},
                    "manoeuvre_census": None}

    report["manoeuvre_census"] = {
        "straight": int(straight.sum()), "non_straight": int(nonstraight.sum()),
        "unlabelled": int((man < 0).sum()),
        "_reads": ("the corpus own per-frame `maneuvers` label at t0; "
                   "class 0 is keep-forward. A split that is almost all "
                   "straight cannot show an echo -- nuScenes is 87 % "
                   "straight at eval and that is why its L2 ranking "
                   "inverts under stratification")}
    first = True
    for ad in arms:
        if not (ad / "ckpt.pt").exists():
            report["arms"][ad.name] = {"status": "NO_CKPT"}
            continue
        model, cfg, cfg_j, load = build_model(ad)
        model = model.to(DEV)
        ablate = bool(cfg_j.get("ablate_frames"))
        (pred, tgt, val, eid, frlast, lat_lg, lon_lg, pl, fx) = collect(
            model, cfg, ds, perm, ablate)
        # \u26d4 PER-SLOT VALIDITY, NOT A GLOBAL INTERSECTION. The 4 s and 6 s
        # goal labels need a longer future than the 2 s one, so a window set
        # restricted to "valid in EVERY slot" silently collapses to the 2 s
        # slot alone -- which would score the panel on the ONE horizon the echo
        # is easiest at, while the prereg names 6 s as primary. Each slot is
        # scored on ITS OWN valid windows, with n and the episode count
        # recorded per slot; within a slot the arm and every reference are
        # still on identical windows, which is what the pairing needs.
        const = EG.constant_only_reference(tgt)
        n = len(perm)
        fit_idx = list(range(n // 2))
        score_idx = list(range(n // 2, n))
        # n_pix=8 -> d = 8*8*3 = 192 < n_fit, so the ridge is POWERED. At
        # n_pix=16 (d=768) it is underpowered BY CONSTRUCTION on this rig and
        # would read a meaningless near-zero (2026-08-22 failure #4).
        pix, pixmeta = EG.raw_pixel_floor(frlast, tgt, fit_idx=fit_idx,
                                          score_idx=score_idx, n_pix=8)
        pix = pix.reshape(-1, tgt.shape[1], 4)

        def _ade1(p, k, m):
            return torch.linalg.vector_norm(
                p[m][:, k, :2] - tgt[m][:, k, :2],
                dim=-1).reshape(-1, 1).numpy()

        merged = {"slots": [], "n_windows": int(n),
                  "n_episodes": int(len(set(eid.tolist()))),
                  "references": ["constant_only", "ha", "ha0", "ha0_ext"]}
        for k in range(tgt.shape[1]):
            m = val[:, k]
            if int(m.sum()) < 32:
                merged["slots"].append({"slot": k, "status": "TOO_FEW_WINDOWS",
                                        "n": int(m.sum())})
                continue
            refs = {"ha": _ade1(ref_ha, k, m), "ha0_ext": _ade1(ref_ext, k, m),
                    "ha0": _ade1(ref_ha0, k, m),
                    "constant_only": _ade1(const, k, m)}
            gk = EG.echo_gate(arm_ade=_ade1(pred, k, m), references=refs,
                              eid=eid[m].numpy(),
                              margins={"ha": 0.10, "ha0_ext": 0.10},
                              n_boot=2000)
            row = gk["slots"][0]
            row["slot"] = k
            row["tau_s"] = taus[k]
            row["n_windows"] = int(m.sum())
            row["n_episodes"] = int(len(set(eid[m].tolist())))
            merged["slots"].append(row)
        g1 = merged
        msk = val[score_idx]
        g1["raw_pixel_floor"] = {
            "meta": pixmeta,
            "per_slot": [
                {"slot": k, "tau_s": taus[k], "n": int(msk[:, k].sum()),
                 "floor_ade": float(torch.linalg.vector_norm(
                     pix[msk[:, k]][:, k, :2]
                     - tgt[score_idx][msk[:, k]][:, k, :2], dim=-1).mean()),
                 "arm_ade": float(torch.linalg.vector_norm(
                     pred[score_idx][msk[:, k]][:, k, :2]
                     - tgt[score_idx][msk[:, k]][:, k, :2], dim=-1).mean())}
                for k in range(tgt.shape[1]) if int(msk[:, k].sum()) > 0],
        }
        # ---- THE BINDING FOUR FAMILIES (PI 2026-08-02) ---------------------
        # \u26d4 ADE alone is an INCOMPLETE eval. Reported PER FAMILY and never
        # pooled; families a goal row cannot carry (headway/TTC, decision
        # quality) are returned with their reason and their n, not dropped.
        fam = EG.trajectory_families(pred, tgt, taus_s=taus)
        fam_ext = EG.trajectory_families(ref_ext, tgt, taus_s=taus)
        families = []
        for k in range(tgt.shape[1]):
            m = val[:, k]
            if int(m.sum()) < 32:
                continue

            def _mm(d, key, sub=None):
                t = d[key][sub] if sub else d[key]
                return float(t[m][:, k].mean())
            families.append({
                "slot": k, "tau_s": taus[k], "n": int(m.sum()),
                "ade_m": {"arm": _mm(fam, "ade_m"),
                          "ha0_ext": _mm(fam_ext, "ade_m")},
                "longitudinal": {
                    "speed_err_mps": {
                        "arm": float(fam["longitudinal"]["speed_err_mps"][m][:, k].mean()),
                        "ha0_ext": float(fam_ext["longitudinal"]["speed_err_mps"][m][:, k].mean())},
                    "headway_ttc": None,
                    "_reason": fam["longitudinal"]["_headway_reason"]},
                "lateral": {
                    key: {"arm": float(fam["lateral"][key][m][:, k].mean()),
                          "ha0_ext": float(fam_ext["lateral"][key][m][:, k].mean())}
                    for key in ("heading_err_rad", "curvature_err_invm",
                                "yaw_rate_err_radps", "cross_track_m")},
                "tactical": None, "_tactical_reason": fam["_tactical_reason"],
                "strategic": None, "_strategic_reason": fam["_strategic_reason"],
            })
        # S6 \u2014 manoeuvre-stratified ADE, arm vs the deciding control
        strat = []
        for name, sel in (("straight", straight), ("non_straight", nonstraight)):
            for k in range(tgt.shape[1]):
                m = val[:, k] & sel
                if int(m.sum()) < 32:
                    strat.append({"subset": name, "slot": k,
                                  "status": "TOO_FEW_WINDOWS",
                                  "n": int(m.sum())})
                    continue
                strat.append({
                    "subset": name, "slot": k, "tau_s": taus[k],
                    "n": int(m.sum()),
                    "n_episodes": int(len(set(eid[m].tolist()))),
                    "arm_ade": float(_ade1(pred, k, m).mean()),
                    "ha0_ext_ade": float(_ade1(ref_ext, k, m).mean()),
                    "ha_ade": float(_ade1(ref_ha, k, m).mean()),
                    "ha0_ade": float(_ade1(ref_ha0, k, m).mean())})

        # ---- THE TACTICAL FAMILY (binding rule; the caller's half) ---------
        # \u26d4 A MISSING FAMILY IS A WORK ITEM, NOT AN EXCUSE.
        # `trajectory_families` returns tactical=None because a goal ROW cannot
        # carry a decision; the decision lives in the factored lat/lon heads and
        # its label comes from the SAME function the trainer supervises with
        # (`tactical.window_factored_labels`), so head and score cannot drift.
        # \u2b50 THE CONTROL THAT MUST READ A KNOWN VALUE: the MAJORITY-CLASS
        # predictor. A 3-way head on a corpus that is mostly "keep" scores high
        # by predicting "keep" forever, so accuracy is only readable against it.
        from tanitad.refs import refc_tactical as _tac
        lat_k, lon_k = _tac.window_factored_labels(pl, fx)
        tactical = {}
        for name, lg, lab in (("lat", lat_lg, lat_k), ("lon", lon_lg, lon_k)):
            m = lab >= 0
            n_ok = int(m.sum())
            if n_ok < 32:
                tactical[name] = {"status": "TOO_FEW_LABELLED", "n": n_ok}
                continue
            pr = lg[m].argmax(-1)
            y = lab[m]
            acc = float((pr == y).float().mean())
            cnt = torch.bincount(y, minlength=lg.shape[-1]).float()
            maj = float(cnt.max() / cnt.sum())
            per_class = []
            for c in range(lg.shape[-1]):
                sel = y == c
                per_class.append({
                    "class": c, "n": int(sel.sum()),
                    "support_frac": float(cnt[c] / cnt.sum()),
                    "recall": (float((pr[sel] == c).float().mean())
                               if int(sel.sum()) else None),
                    "predicted_frac": float((pr == c).float().mean())})
            tactical[name] = {
                "n": n_ok, "accuracy": acc,
                "majority_class_control": maj,
                "beats_majority": bool(acc > maj),
                "per_class": per_class,
                "_reads": ("accuracy BELOW the majority-class control is worse "
                           "than a constant predictor; a class with recall 0 "
                           "and predicted_frac 0 is UNREACHABLE, which is the "
                           "X15 signature")}
        tactical["_strategic_note"] = (
            "the STRATEGIC family (g_str bearing/distance vs the LAN corridor) "
            "is NOT computed here: the LAN label is training-only (E12) and is "
            "not emitted by this dataset, so there is no target on these "
            "windows. Reported as absent WITH the reason, per the rule -- not "
            "silently dropped.")
        tactical["_selected_vs_executed_note"] = (
            "selected-vs-executed manoeuvre agreement would need the label "
            "function applied to the ARM'S OWN 20-step future, and the model "
            "emits an 8-slot plan plus 3 goal rows, not a 20-step pose track. "
            "Computing it would require re-deriving the classifier on a "
            "different support -- a new instrument, filed as a work item "
            "rather than approximated here.")

        # ---- GATE 2 (structural) and GATE 2b (functional) ------------------
        # ⚠️ ON CPU, DELIBERATELY. `goal_provenance`'s perturbations draw from
        # a `torch.Generator(device="cpu")`, so a CUDA input tensor raises
        # `Expected a 'cuda' device type for generator`. The probes are a
        # single batch on a 17 M model, so CPU costs seconds -- and running the
        # intervention on the same device as its RNG keeps the perturbation
        # reproducible, which is the property the probe is built on.
        model = model.cpu()
        items = [ds[i] for i in perm[:BATCH]]
        b = torch.utils.data.default_collate(items)
        fr = b["frames"]
        if ablate:
            fr = torch.full_like(fr, float(fr.mean()))
        pose_last = b["pose_last"]
        gb = {"frames": fr, "nav_cmd": b["nav_cmd"],
              "v0": pose_last[:, 3], "steps": cfg.core.decoder.diffusion_steps}
        if cfg.ego_state_inject:
            gb["ego_state"] = v3.ego_state_from_batch(
                {"pose_last": pose_last, "actions": b["actions"]})
            g2 = EG.ego_intervention_test(model, gb)
        else:
            g2 = {"status": "UNPOWERED", "verdict": None,
                  "reason": ("this arm has no ego_state input at all (v3), so "
                             "the ego direction cannot be probed — that is the "
                             "arm's definition, not a defect")}

        def predict(frames, ego_state):
            kw = dict(gb)
            kw["frames"] = frames
            if cfg.ego_state_inject:
                kw["ego_state"] = ego_state
            with torch.no_grad():
                return model(**kw)["g_tac"].float()

        egos = gb.get("ego_state", torch.zeros(fr.shape[0], 5))
        g2b = EG.source_ablation_test(
            predict, frames=fr, ego_state=egos,
            target=b["goal_tac"].float(), eid=b["episode_id"].reshape(-1).numpy(),
            n_boot=500)

        # \u2b50\u2b50 THE REGRESS ARM, PROBED TWICE. Its training frames were a
        # constant, so at scoring time a DERANGEMENT of a constant changes
        # nothing and `scene degradation == 0` holds BY CONSTRUCTION -- a
        # critic could fairly call that rigged. So the arm is probed a second
        # time with the REAL frames: if it still shows no scene degradation,
        # the verdict is about what the WEIGHTS learned, not about what the
        # harness fed them. Both readings are recorded; neither is dropped.
        g2b_real = None
        if ablate:
            fr_real = torch.utils.data.default_collate(
                [ds[i] for i in perm[:BATCH]])["frames"]

            def predict_real(frames, ego_state):
                kw = dict(gb)
                kw["frames"] = frames
                if cfg.ego_state_inject:
                    kw["ego_state"] = ego_state
                with torch.no_grad():
                    return model(**kw)["g_tac"].float()

            g2b_real = EG.source_ablation_test(
                predict_real, frames=fr_real, ego_state=egos,
                target=b["goal_tac"].float(),
                eid=b["episode_id"].reshape(-1).numpy(), n_boot=500)

        # \u26d4 APPLICABILITY, STATED RATHER THAN FUDGED. The anti-echo gate is
        # defined for an arm that CONSUMES the ego state. Arm A (v3) has no
        # ego input at all, so `IGNORES_EGO` is its DEFINITION, not a failure
        # -- reporting it as a gate failure would be a category error, and
        # silently remapping the verdict to make it pass would be worse. The
        # scene half of the probe still applies and is recorded.
        if not cfg.ego_state_inject:
            gate = {"applicable": False,
                    "reason": ("this arm has no ego input by construction "
                               "(the v3 control), so the ego clauses of the "
                               "gate are a category error, not a failure"),
                    "scene_is_used": bool(
                        g2b["sources"]["scene"]["is_used"]),
                    "scene_degradation_rel": float(
                        g2b["sources"]["scene"]["degradation_rel"])}
        else:
            try:
                verdict = EG.assert_not_echoing(g1, g2, gate2b=g2b)
                gate = {"applicable": True, "raised": False, "verdict": verdict}
            except EG.EchoViolation as exc:
                gate = {"applicable": True, "raised": True,
                        "message": str(exc)}
        report["arms"][ad.name] = {
            "status": "MEASURED", "config": cfg_j.get("ego"),
            "size": cfg_j.get("size"), "rig_rung": cfg_j.get("rig_rung"),
            "ablate_frames": ablate, "load": load,
            "four_families": families, "tactical_family": tactical,
            "manoeuvre_stratified": strat,
            "gate1": g1, "gate2": g2, "gate2b": g2b,
            "gate2b_realframes": g2b_real,
            "GATE": gate,
        }
        print(f"[{ad.name}] gate2={g2.get('verdict')} "
              f"gate2b={g2b.get('verdict')} "
              f"gate={gate.get('raised', 'N/A')}", flush=True)
        if first:
            report["controls_by_slot"] = [
                {"slot": k, "tau_s": taus[k], "n": int(val[:, k].sum()),
                 "ha0": float(_ade1(ref_ha0, k, val[:, k]).mean()),
                 "ha": float(_ade1(ref_ha, k, val[:, k]).mean()),
                 "ha0_ext": float(_ade1(ref_ext, k, val[:, k]).mean()),
                 "constant_only": float(_ade1(const, k, val[:, k]).mean())}
                for k in range(tgt.shape[1]) if int(val[:, k].sum()) >= 32]
            first = False
        del model
        torch.cuda.empty_cache()

    OUT.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("wrote gate_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
