"""TanitEval — tactical + strategic planning panel.

We have only scored the OPERATIVE brain (grounded rollout ADE). The flagship is
a 4-brain hierarchy; this panel measures the two planning brains against the
same pseudo-labels they were trained on (refb_labels derivations):

  STRATEGIC (route brain):
    route_acc_follow  route-heading (L/S/R over 15-25s) predicted from VISION
                      (nav=follow) — the genuine strategic-understanding metric
    route_acc_nav     with the true nav command given (upper bound)
    nav_reliance      route_acc_nav - route_acc_follow  (how much it leans on
                      the given command vs infers route from the scene)

  TACTICAL (maneuver + goal brain):
    maneuver_acc      5-way maneuver class (lane_keep/turn_L/turn_R/accel/brake)
    turn_recall       recall on {turn_left,turn_right} — the curve competence
    tactical_wp_ade   DIRECT 2s sub-waypoint ADE (a SEPARATE trajectory surface
                      from the operative rollout — compare to the 0.628)
    goal_latent_cos   cos(target_latent, true future latent @2s) — goal-imagination

Applies to WorldModel arms with trained tactical_policy + strategic_policy."""
from __future__ import annotations

import sys
from collections import Counter

import torch
import torch.nn.functional as F

sys.path.insert(0, "/root/TanitAD/stack")
sys.path.insert(0, "/root/TanitAD/stack/scripts")

import refb_labels as rl  # noqa: E402
from driving_diagnostic import WP_STEPS, gt_ego_waypoints  # noqa: E402
from tanitad.eval.gates import split_by_episode  # noqa: E402
from tanitad.refs.refb import MANEUVER_CLASSES, ROUTE_CLASSES  # noqa: E402

SPEED_SCALE = 10.0
WIN = 8
GOAL_H = max(WP_STEPS)                    # 20 (2 s) — maneuver + goal horizon
TURN = [MANEUVER_CLASSES.index(c) for c in ("turn_left", "turn_right")]
DECODE_MARGIN = 0.05                      # above-chance margin for the verdict
PROBE_SEEDS = (0, 1, 2)                   # episode-split resamples
PROBE_VAL_FRAC = 0.25
PROBE_EPOCHS = 200
LANE_KEEP = MANEUVER_CLASSES.index("lane_keep")   # 0 — majority baseline class


def _majority_rate(labels) -> float | None:
    """Majority-class fraction (the chance base rate for a raw-accuracy head)."""
    if labels is None or len(labels) == 0:
        return None
    c = Counter(int(x) for x in labels)
    return round(max(c.values()) / len(labels), 4)


def _behavior_probe(feats, labels, eid, n_classes):
    """Linear decodability of behavior (maneuver class) from the latent vs chance.

    REUSES eval_behavior's PRIMARY instrument verbatim (``fit_classifier`` — a
    z-scored, class-weighted linear logistic probe — + ``balanced_accuracy``),
    exactly as the superseded ``compare_arms._behavior_probe`` did, so this block
    reconciles with eval_behavior.py. Balanced accuracy (mean per-class recall) is
    the honest metric under class imbalance: raw accuracy is meaningless when
    lane-keep dominates, and chance for balanced accuracy is 1/n_classes.
    decodability_vs_chance <= 0 => the latent carries no linearly-readable
    behavior signal (the "behavior-decodability block")."""
    import eval_behavior as eb  # noqa: E402  (scripts on sys.path)
    if feats is None or labels is None or feats.shape[0] < 40:
        return {"skipped": "too few windows for a held-out probe"}
    eid_list = [int(e) for e in eid]
    if len(set(eid_list)) < 2:
        return {"skipped": "need >=2 episodes for an episode-disjoint split"}
    labels = labels.long()
    accs, bals, f1s = [], [], []
    for seed in PROBE_SEEDS:
        tr, va = split_by_episode(eid_list, PROBE_VAL_FRAC, seed)
        if len(tr) < 20 or len(va) < 10:
            continue
        with torch.enable_grad():        # run() holds no_grad; the probe trains
            pred, _ = eb.fit_classifier(feats[tr], labels[tr], feats[va],
                                        n_classes, kind="linear",
                                        epochs=PROBE_EPOCHS, seed=seed)
        cm = eb.confusion_matrix(labels[va], pred, n_classes)
        accs.append(eb.accuracy(cm))
        bals.append(eb.balanced_accuracy(cm))
        f1s.append(eb.macro_f1(cm))
    if not bals:
        return {"skipped": "no valid episode splits"}
    balacc = sum(bals) / len(bals)
    chance = 1.0 / n_classes
    return {
        "maneuver_balanced_accuracy": round(balacc, 4),
        "chance_balacc": round(chance, 4),
        "beats_chance": bool(balacc > chance),
        "decodability_vs_chance": round(balacc - chance, 4),
        "maneuver_accuracy_raw": round(sum(accs) / len(accs), 4),
        "maneuver_majority_acc": _majority_rate(labels.tolist()),
        "macro_f1": round(sum(f1s) / len(f1s), 4),
        "n_windows": int(feats.shape[0]), "n_seeds": len(bals),
        "classes": list(MANEUVER_CLASSES),
        "note": ("maneuver linearly decodable above chance (balanced-acc)"
                 if balacc - chance > DECODE_MARGIN else
                 "maneuver NOT decodable above chance from the latent"),
    }


@torch.no_grad()
def run(model, episodes, device, max_eps=20, stride=8):
    if getattr(model, "tactical_policy", None) is None:
        return {"skipped": "no trained tactical/strategic policy brains"}
    model.eval()
    man_ok = man_n = 0
    turn_hit = turn_tot = 0
    r_follow_ok = r_nav_ok = r_n = 0
    wp_ade, goal_cos = [], []
    man_conf = torch.zeros(len(MANEUVER_CLASSES), len(MANEUVER_CLASSES))
    lat_all, man_all, eid_all, route_valid = [], [], [], []   # behavior probe

    for ep in episodes[:max_eps]:
        fr, T = ep.feats, ep.feats.shape[0]
        for i in range(0, T - WIN - GOAL_H, stride * 8):
            ch = list(range(i, min(i + stride * 8, T - WIN - GOAL_H), stride))
            if not ch:
                continue
            last = torch.tensor([t + WIN - 1 for t in ch])
            fw = torch.stack([torch.as_tensor(fr[t:t + WIN]) for t in ch]
                             ).to(device).float()
            if ep.feats.dtype == torch.uint8:
                fw = fw.div_(255.0)
            fut = torch.stack([torch.as_tensor(ep.poses[t + WIN:t + WIN + GOAL_H])
                               for t in ch])                  # [b,GOAL_H,4]
            pl = ep.poses[last]                                # [b,4]
            states = model.encode_window(fw)
            lat_all.append(states[:, -1].detach().cpu().float())
            eid_all.extend([ep.episode_id] * len(ch))

            # --- labels (exactly the trainer's refb_labels derivations) ------
            man_tgt = rl.classify_maneuver(
                pl[:, 2], fut[:, GOAL_H - 1, 2], pl[:, 3],
                fut[:, GOAL_H - 1, 3]).long().to(device)
            man_all.append(man_tgt.detach().cpu())
            navs, rts, valids = [], [], []
            for t in ch:
                cmd, valid = rl.nav_command(ep.poses, t + WIN - 1)
                navs.append(cmd); valids.append(bool(valid))
                rts.append(rl.route_target(cmd))
            nav = torch.tensor(navs, device=device)
            rt = torch.tensor(rts, device=device)
            vmask = torch.tensor(valids, device=device)
            gtwp = gt_ego_waypoints(ep.poses, last).to(device)   # [b,4,2]

            # --- strategic: route from VISION (follow) vs with command -------
            follow = torch.zeros(len(ch), dtype=torch.long, device=device)
            sf = model.strategic_policy(states, follow)
            sn = model.strategic_policy(states, nav)
            if vmask.any():
                rf = sf["route_logits"].argmax(-1)[vmask]
                rn = sn["route_logits"].argmax(-1)[vmask]
                r_follow_ok += int((rf == rt[vmask]).sum())
                r_nav_ok += int((rn == rt[vmask]).sum())
                r_n += int(vmask.sum())
                route_valid.extend(rt[vmask].tolist())

            # --- tactical: maneuver + waypoints + goal latent (follow ctx) ---
            tac = model.tactical_policy(states, sf["ctx"])
            mp = tac["maneuver_logits"].argmax(-1)
            man_ok += int((mp == man_tgt).sum()); man_n += len(ch)
            for gt_c, pr_c in zip(man_tgt.tolist(), mp.tolist()):
                man_conf[gt_c, pr_c] += 1
            tm = torch.tensor([c in TURN for c in man_tgt.tolist()])
            turn_tot += int(tm.sum())
            turn_hit += int(((mp.cpu() == man_tgt.cpu()) & tm).sum())

            wp = torch.stack([tac["waypoints"][k] for k in WP_STEPS], 1)  # [b,4,2]
            wp_ade.append(torch.linalg.norm(wp - gtwp, dim=-1).mean(1).cpu())

            if "target_latent" in tac:
                zt = torch.as_tensor(fr[[t + WIN + GOAL_H - 1 for t in ch]]
                                     ).to(device).float()
                if ep.feats.dtype == torch.uint8:
                    zt = zt.div_(255.0)
                z_true = model.encode(zt)
                goal_cos.append(F.cosine_similarity(
                    tac["target_latent"], z_true, dim=-1).mean().cpu())

    n = man_n
    route_acc_follow = round(r_follow_ok / max(r_n, 1), 4)
    route_base = _majority_rate(route_valid)          # chance = majority route
    behavior = _behavior_probe(
        torch.cat(lat_all) if lat_all else None,
        torch.cat(man_all) if man_all else None,
        eid_all, len(MANEUVER_CLASSES))
    out = {
        "n_windows": n,
        "strategic": {
            "route_acc_follow": route_acc_follow,
            "route_acc_nav": round(r_nav_ok / max(r_n, 1), 4),
            "nav_reliance": round((r_nav_ok - r_follow_ok) / max(r_n, 1), 4),
            "majority_route_base_rate": route_base,
            "route_skill_vs_chance": (round(route_acc_follow - route_base, 4)
                                      if route_base is not None else None),
            "n_route_valid": r_n, "classes": list(ROUTE_CLASSES),
        },
        "behavior_decodability": behavior,
        "tactical": {
            "maneuver_acc": round(man_ok / max(n, 1), 4),
            "turn_recall": round(turn_hit / max(turn_tot, 1), 4),
            "n_turns": turn_tot,
            "tactical_wp_ade_2s": round(float(torch.cat(wp_ade).mean()), 4),
            "goal_latent_cos": (round(float(torch.stack(goal_cos).mean()), 4)
                                if goal_cos else None),
            "maneuver_classes": list(MANEUVER_CLASSES),
        },
    }
    out["verdict"] = _verdict(out)
    return out


def _verdict(o):
    s, t = o["strategic"], o["tactical"]
    b = o.get("behavior_decodability", {})
    bits = []
    if s.get("route_skill_vs_chance") is not None:
        bits.append("route beats chance +{:.2f}".format(s["route_skill_vs_chance"])
                    if s["route_skill_vs_chance"] > DECODE_MARGIN
                    else "route ~ chance (base {})".format(s["majority_route_base_rate"]))
    bits.append("route: infers from vision" if s["nav_reliance"] < 0.1
                else "route: leans on given command")
    bits.append("turns weak" if t["turn_recall"] < 0.4 else "turns ok")
    if b.get("decodability_vs_chance") is not None:
        bits.append("behavior decodable +{:.2f}".format(b["decodability_vs_chance"])
                    if b["decodability_vs_chance"] > DECODE_MARGIN
                    else "behavior ~ chance")
    return " · ".join(bits)


def run_and_save(key, device="cuda", max_eps=20,
                 out_dir="/root/taniteval/results"):
    """Standalone: load a policy arm, score the planning panel (now with route
    base-rate + behavior decodability), and write results/plan_<key>.json."""
    import json
    from pathlib import Path
    sys.path.insert(0, "/root/taniteval")
    from taniteval import data, loaders            # read-only use (adf3 owns)
    from taniteval.registry import MODELS
    entry = [m for m in MODELS if m["key"] == key][0]
    L = loaders.load(entry, device)
    if getattr(L["model"], "tactical_policy", None) is None:
        print(f"[plan] {key}: no trained tactical/strategic brains — skip",
              flush=True)
        return {"key": key, "skipped": "no policy brains"}
    files = data.list_val_episodes(
        "/root/valdata/physicalai-val-0c5f7dac3b11", 40)
    if entry.get("train_ids"):                      # replicate runner leak guard
        from tanitad.data.mixing import load_episode
        tid = set(Path(entry["train_ids"]).read_text().split())
        files = [f for f in files
                 if str(load_episode(str(f), mmap=True).episode_id) not in tid]
    eps = (data.load_frames(files) if L["feed"] == "frames"
           else data.load_features(files, L["feed"], device))
    res = run(L["model"], eps, device, max_eps=max_eps)
    res["model"] = {k: entry.get(k) for k in ("key", "name", "arch", "encoder")}
    res["ckpt_step"] = L["step"]
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / f"plan_{key}.json").write_text(
        json.dumps(res, indent=2, default=str))
    if res.get("skipped"):
        print(f"[plan] {key}: skipped ({res['skipped']})", flush=True)
        return res
    s, b = res["strategic"], res.get("behavior_decodability", {})
    print(f"[plan] {key}: route follow={s['route_acc_follow']} vs base="
          f"{s['majority_route_base_rate']} (skill {s['route_skill_vs_chance']}) "
          f"| maneuver bal-acc={b.get('maneuver_balanced_accuracy')} vs chance "
          f"{b.get('chance_balacc')} (beats={b.get('beats_chance')}) "
          f"-> plan_{key}.json", flush=True)
    return res


def main():
    import argparse
    sys.path.insert(0, "/root/taniteval")
    from taniteval.registry import MODELS
    ap = argparse.ArgumentParser("taniteval.planning")
    ap.add_argument("--model", help="registry key; omit with --all")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--max-eps", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    keys = ([m["key"] for m in MODELS] if a.all else [a.model])
    for key in keys:
        if not any(m["key"] == key for m in MODELS):
            print(f"[plan] unknown model {key}", flush=True)
            continue
        try:
            run_and_save(key, a.device, a.max_eps)
        except Exception as e:
            import traceback
            print(f"[plan] {key} FAILED: {type(e).__name__}: {str(e)[:160]}",
                  flush=True)
            traceback.print_exc()


if __name__ == "__main__":
    main()
