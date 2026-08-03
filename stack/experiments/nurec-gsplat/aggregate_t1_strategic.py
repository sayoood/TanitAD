#!/usr/bin/env python3
"""Turn the T1 nav-sweep ticks into the STRATEGIC family, with every degeneracy control.

Consumes ``results/t1_route_ticks.json`` (from :mod:`score_t1_strategic`) — per scene,
per admissible pose, each arm's route logits under **every** nav command — and emits the
family through :mod:`taniteval.strategic_optionset` with the episode-cluster bootstrap,
cluster = scene.

THE THREE CONTROLS, AND WHAT EACH ONE KILLS
-------------------------------------------
====================  ==================================================================
control               the false positive it makes unreachable
====================  ==================================================================
``BEST_CONSTANT``     a head that always answers one class. This is the defect that made
                      ``route_head_eq_logged = 1.0000`` meaningless. Fitted on the SAME
                      events, so beating it is conservative. (built into the family)
``NAV_ECHO``          a head that just repeats the nav command it was handed. The nav is
                      ``nav_command_v21(gt_future)`` — an ORACLE — so an arm that cannot
                      beat this has shown nothing about strategy under ``navORACLE``.
``navSHUFFLED``       pass-through with the marginal preserved: each event keeps a real
                      oracle nav, but ANOTHER event's. Permuted **globally over every
                      event in the run** — a within-scene permutation is the identity on
                      the 60 %+ of T1 scenes that carry exactly one decision.
====================  ==================================================================

``nav_passthrough_rate`` is the most direct of all: the fraction of scored poses at which
the arm's argmax MOVES when only the nav input changes and the pixels do not. A head at
1.0 is a relabelling of its own input.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
NAV_TO_ROUTE = {0: 1, 1: 0, 2: 2, 3: 1}     # follow/left/right/straight -> map class
NAV_NAMES = ("follow", "left", "right", "straight")
CONDS = ("navFOLLOW", "navORACLE", "navSHUFFLED", "navLEFT", "navRIGHT", "navSTRAIGHT")


def _taniteval():
    for cand in (Path("/home/nvidia/tv"), *HERE.parents):
        if (cand / "taniteval" / "ci.py").exists():
            sys.path.insert(0, str(cand))
            break
        if (cand / "taniteval" / "taniteval" / "ci.py").exists():
            sys.path.insert(0, str(cand / "taniteval"))
            break
    from taniteval import strategic_optionset as SO
    return SO


def nav_for(cond, row, shuffled):
    if cond == "navFOLLOW":
        return 0
    if cond == "navORACLE":
        return int(row["nav_oracle"])
    if cond == "navSHUFFLED":
        return int(shuffled[row["event_id"]])
    return {"navLEFT": 1, "navRIGHT": 2, "navSTRAIGHT": 3}[cond]


def build_shuffle(ticks, seed=1234):
    """event_id -> the ORACLE nav of a DIFFERENT event, permuted over the WHOLE run.

    ⛔ Global, not per scene. MEASURED on the first run: 60 %+ of T1 scenes carry exactly
    one decision event, so a within-scene permutation returned the identity and the
    control measured nothing while looking like it had run.
    """
    ev_nav = {}
    for rows in ticks.values():
        for r in rows:
            ev_nav.setdefault(r["event_id"], int(r["nav_oracle"]))
    ids = sorted(ev_nav)
    rng = np.random.default_rng(seed)
    perm = list(rng.permutation(len(ids)))
    # derange as far as the data allows, then report how far it got
    for i in range(len(ids)):
        if perm[i] == i and len(ids) > 1:
            j = (i + 1) % len(ids)
            perm[i], perm[j] = perm[j], perm[i]
    out = {ids[i]: ev_nav[ids[perm[i]]] for i in range(len(ids))}
    n_same_value = sum(1 for k in ids if out[k] == ev_nav[k])
    return out, {"n_events": len(ids), "n_fixed_points": sum(
        1 for i in range(len(ids)) if perm[i] == i),
        "n_events_whose_nav_VALUE_is_unchanged": n_same_value,
        "note": ("a 4-value vocabulary means a permutation often lands on the same VALUE; "
                 "that is counted, not hidden — it makes navSHUFFLED a CONSERVATIVE "
                 "control (it retains some genuine signal).")}


def preds_for(SO, ticks, reports, cond, shuffled, arm):
    """{event_id: pred} for one (arm, condition), joined per scene by pose."""
    out = {}
    for sid, rows in ticks.items():
        rep = reports.get(sid)
        if rep is None:
            continue
        tk = []
        for r in rows:
            nav = nav_for(cond, r, shuffled)
            if arm == "NAV_ECHO":
                v = NAV_TO_ROUTE.get(nav, 1)
            else:
                v = r["sweep"].get(arm, {}).get(str(nav), r["sweep"].get(arm, {}).get(nav))
            if v is None:
                continue
            tk.append({"i_gt": r["i_gt"], "route_head": v})
        p = SO.event_predictions_from_ticks(tk, rep)
        for eid, val in p.items():
            if eid != "_join":
                out[eid] = val
    return out


def sweeps_for(ticks, arm):
    """``[{nav: predicted_class}, ...]`` at FIXED observations — the family's echo guard.

    ``NAV_ECHO`` is the analytic echo, so its sweep is written out rather than measured:
    the baseline must trip the guard, otherwise the guard is not testing anything.
    """
    out = []
    for rows in ticks.values():
        for r in rows:
            if arm == "NAV_ECHO":
                out.append(dict(NAV_TO_ROUTE))
                continue
            sw = r["sweep"].get(arm)
            if not sw:
                continue
            d = {}
            for n in range(4):
                v = sw.get(str(n), sw.get(n))
                if v is not None:
                    d[n] = int(np.argmax(v))
            if d:
                out.append(d)
    return out


def passthrough(ticks, arm):
    """How often the argmax MOVES when only the nav input changes."""
    moved = same = 0
    per_nav = {n: {} for n in range(4)}
    for rows in ticks.values():
        for r in rows:
            sw = r["sweep"].get(arm)
            if not sw:
                continue
            am = {}
            for n in range(4):
                v = sw.get(str(n), sw.get(n))
                if v is not None:
                    am[n] = int(np.argmax(v))
            if len(am) < 2:
                continue
            if len(set(am.values())) > 1:
                moved += 1
            else:
                same += 1
            for n, c in am.items():
                per_nav[n][c] = per_nav[n].get(c, 0) + 1
    n = moved + same
    return {
        "n_poses": n,
        "nav_passthrough_rate": round(moved / n, 4) if n else None,
        "argmax_distribution_per_nav": {
            NAV_NAMES[k]: {("LEFT", "STRAIGHT", "RIGHT", "UTURN")[c]: v
                           for c, v in sorted(d.items())} for k, d in per_nav.items()},
        "reading": ("1.0 = the route head is a relabelling of its own nav input at EVERY "
                    "pose; 0.0 = the head ignores nav entirely and answers from pixels."),
    }


def logit_variance_decomposition(ticks, arm):
    """Where does the route logit's variance come from — the PIXELS or the nav input?

    ⚠️ Written because "flagship-v1 is a lookup table" would have been an OVERCLAIM and
    this is the number that keeps it honest. MEASURED: flagship-v1's logits **do** move
    with the image at a fixed nav (std 0.21-2.50), so the head is not a pure function of
    its input — yet the nav term still decides the argmax at **100 %** of poses. The
    correct statement is *nav-DOMINATED*, not *nav-only*.

    ``across_navs_at_fixed_pose`` is the decisive one: exactly 0.0 means the head cannot
    see the nav command at all (REF-C), and a value above the image term means the input
    outweighs the observation (flagship-v1).
    """
    per_nav, per_pose = {n: [] for n in range(4)}, []
    for rows in ticks.values():
        for r in rows:
            sw = r["sweep"].get(arm)
            if not sw:
                continue
            got = {}
            for n in range(4):
                v = sw.get(str(n), sw.get(n))
                if v is not None:
                    per_nav[n].append(v)
                    got[n] = v
            if len(got) >= 2:
                per_pose.append(np.asarray(list(got.values()), float).std(0).mean())
    img = [float(np.asarray(v, float).std(0).mean()) for v in per_nav.values() if len(v) > 1]
    nav = float(np.mean(per_pose)) if per_pose else None
    return {
        "n_poses": len(per_pose),
        "logit_std_across_POSES_at_fixed_nav_mean": round(float(np.mean(img)), 5) if img else None,
        "logit_std_across_POSES_at_fixed_nav_per_nav": {
            NAV_NAMES[k]: round(float(np.asarray(v, float).std(0).mean()), 5)
            for k, v in per_nav.items() if len(v) > 1},
        "logit_std_across_NAVS_at_fixed_pose_mean": round(nav, 5) if nav is not None else None,
        "nav_over_image_variance_ratio": (
            round(nav / max(float(np.mean(img)), 1e-9), 3) if (nav is not None and img)
            else None),
        "HEAD_IS_NAV_BLIND": bool(nav is not None and nav == 0.0),
        "reading": ("across_NAVS == 0.0 -> the head structurally cannot see its nav input. "
                    "ratio >> 1 -> the conditioning input outweighs the observation. "
                    "Neither is read alone: pair it with nav_passthrough_rate, which is "
                    "what actually decides the answer."),
    }


def slim(fam):
    keep = ("status", "reason", "n", "arm", "n_events_scoreable",
            "n_decision_instances_scored", "n_scenes_scored",
            "n_events_without_a_prediction", "n_events_gt_outside_head_vocabulary",
            "n_predictions_outside_the_option_set",
            "route_class_accuracy", "route_class_per_class",
            "route_class_confusion_gt_x_pred", "accuracy_by_branching_factor",
            "floors", "vs_best_constant", "beats_best_constant",
            "STRATEGIC_SKILL_ADMISSIBLE", "conditioning_echo_control", "⛔_ECHO",
            "label_degenerate", "prediction_degenerate",
            "decision_lead_distance_m", "branching_factor")
    o = {k: fam[k] for k in keep if k in fam}
    if "branching_factor" in o:
        o["branching_factor"] = {k: v for k, v in o["branching_factor"].items()
                                 if k != "values"}
    if "floors" in o:
        o["floors"] = {k: v for k, v in o["floors"].items() if not str(k).startswith("⛔")}
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticks", default=str(HERE / "results" / "t1_route_ticks.json"))
    ap.add_argument("--labels", default=str(HERE / "results" / "strategic_gt_t1"))
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--leakage", default=None,
                    help="leakage_check_t1.json; with --leak-free, DROP every scene whose "
                         "clip is in the PhysicalAI-AV train split")
    ap.add_argument("--leak-free", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    SO = _taniteval()
    tp = Path(a.ticks)
    if tp.is_dir():
        # ⛔ the per-scene checkpoint directory. Accepted directly so a run that was
        # interrupted (Thor rebooted once, mid-run) is still fully analysable from the
        # parts it did finish, without waiting for a combined file that was never written.
        ticks, scenes_prov = {}, {}
        for p in sorted(tp.glob("*.json")):
            d = json.loads(p.read_text())
            ticks[d["scene"]] = d["rows"]
            scenes_prov[d["scene"]] = d["prov"]
        prov = {"mode": "OPEN-LOOP on the LOGGED clipgt track, REAL 4K reference camera",
                "arms": {"flagship-v1": {}, "refc-base": {}},
                "scenes": scenes_prov, "source": "per-scene checkpoints"}
    else:
        blob = json.loads(tp.read_text())
        ticks, prov = blob["ticks"], blob["provenance"]
    all_reports = SO.load_label_reports(a.labels)

    leak = None
    if a.leakage:
        lk = json.loads(Path(a.leakage).read_text())
        in_train = set(lk["T1_with_a_scoreable_branch"]["in_train_split"])
        leak = {"n_scored_in_train_split": len(in_train & set(ticks)),
                "catalogue_train_split_size": lk["catalogue"]["n_train_valid"],
                "our_build_n_clips": 2400,
                "base_rate_of_selection": round(
                    2400 / max(lk["catalogue"]["n_train_valid"], 1), 5),
                "leak_free_subset_applied": bool(a.leak_free),
                "⛔_note": (
                    "'in the train SPLIT' is an UPPER BOUND, not leakage: the split holds "
                    f"{lk['catalogue']['n_train_valid']} valid clips and our corpus took "
                    "2400 of them. The assumption-free control is to re-score on the "
                    "COMPLEMENT (--leak-free), which is what this flag does.")}
        if a.leak_free:
            ticks = {k: v for k, v in ticks.items() if k not in in_train}

    # ⛔ score ONLY on scenes that actually produced ticks. A scene the runner could not
    # observe would otherwise be scored as "the arm answered nothing" = 0, which is a
    # RUNNER failure being charged to the model.
    reports = {s: all_reports[s] for s in ticks if s in all_reports}
    reports["_refused"] = all_reports.get("_refused", {})
    events, scenes = SO.scoreable_events(reports)
    shuffled, shuf_meta = build_shuffle(ticks)

    out = {
        "tool": "aggregate_t1_strategic.py",
        "evidence_class": "MEASURED",
        "mode": prov.get("mode"),
        "arms": prov.get("arms"),
        "TIER": "T1 (|turn| >= 60 deg, complete traversal) of the 1607-scene NuRec survey",
        "n_scenes_with_ticks": len(ticks),
        "n_scoreable_events": len(events),
        "n_bootstrap_clusters_scenes": len(set(scenes)),
        "n_decision_instances_scored": sum(
            len({r["event_id"] for r in rows}) for rows in ticks.values()),
        "n_poses_observed": sum(len(rows) for rows in ticks.values()),
        "SHUFFLE_CONTROL": shuf_meta,
        "LEAKAGE_CONTROL": leak,
        "NEGATIVE_CONTROL": SO.discrimination_control(reports, n_boot=a.n_boot),
        "NAV_PASSTHROUGH": {},
        "LOGIT_VARIANCE_DECOMPOSITION": {},
        "families": {},
        "PAIRED": {},
    }

    arms = [k for k in (prov.get("arms") or {})]
    for arm in arms:
        out["NAV_PASSTHROUGH"][arm] = passthrough(ticks, arm)
        out["LOGIT_VARIANCE_DECOMPOSITION"][arm] = logit_variance_decomposition(ticks, arm)

    vec = {}         # (arm, cond) -> per-event hit vector, aligned with `events`
    gt = [e.get("route_gt_class") for e in events]
    for arm in arms + ["NAV_ECHO"]:
        for cond in CONDS:
            if arm == "NAV_ECHO" and cond not in ("navORACLE", "navSHUFFLED", "navFOLLOW"):
                continue
            p = preds_for(SO, ticks, reports, cond, shuffled, arm)
            fam = SO.strategic_family(reports, p, arm=f"{arm}/{cond}", n_boot=a.n_boot,
                                      conditioning_sweeps=sweeps_for(ticks, arm))
            out["families"][f"{arm}/{cond}"] = slim(fam)
            vec[(arm, cond)] = [
                0.0 if (p.get(e["event_id"]) is None
                        or p[e["event_id"]].get("class") is None or g is None)
                else float(int(p[e["event_id"]]["class"]) == int(g))
                for e, g in zip(events, gt)]

    def pair(a_key, b_key, why):
        va, vb = vec.get(a_key), vec.get(b_key)
        if va is None or vb is None:
            return None
        r = SO._ci.paired_episode_cluster_bootstrap(va, vb, scenes,
                                                    n_boot=a.n_boot, seed=0)
        return {"A": "/".join(a_key), "B": "/".join(b_key), "why": why,
                **{k: v for k, v in r.items() if not str(k).startswith("⛔")}}

    P = out["PAIRED"]
    for arm in arms:
        P[f"{arm}: navORACLE - navSHUFFLED"] = pair(
            (arm, "navORACLE"), (arm, "navSHUFFLED"),
            "NAV PASS-THROUGH. >0 and separated => the score rides on the oracle nav input.")
        P[f"{arm}: navORACLE - NAV_ECHO"] = pair(
            (arm, "navORACLE"), ("NAV_ECHO", "navORACLE"),
            "does the arm add anything over repeating the oracle command it was handed?")
        P[f"{arm}: navFOLLOW - navORACLE"] = pair(
            (arm, "navFOLLOW"), (arm, "navORACLE"),
            "what the deployable (no-oracle) setting costs.")
    for cond in ("navFOLLOW", "navORACLE", "navSHUFFLED"):
        if len(arms) >= 2:
            P[f"{arms[0]} - {arms[1]} @ {cond}"] = pair(
                (arms[0], cond), (arms[1], cond),
                "THE head-to-head, paired on identical decision events.")

    Path(a.out).write_text(json.dumps(out, indent=1, default=str))
    print(f"wrote {a.out}")
    print(f"n_events={out['n_scoreable_events']} clusters={out['n_bootstrap_clusters_scenes']}")
    for k, v in out["NAV_PASSTHROUGH"].items():
        print(f"  passthrough {k:14s} {v['nav_passthrough_rate']}  (n={v['n_poses']})")
    for k, v in out["families"].items():
        acc = v.get("route_class_accuracy") or {}
        if "mean" in acc:
            bc = v["floors"]["BEST_CONSTANT"]
            print(f"  {k:28s} acc={acc['mean']:.4f} [{acc['lo']:.4f},{acc['hi']:.4f}] "
                  f"best_const={bc['class']}@{bc['accuracy']:.4f} "
                  f"beats={v.get('beats_best_constant')} "
                  f"SKILL_ADMISSIBLE={v.get('STRATEGIC_SKILL_ADMISSIBLE')}")
    for k, v in P.items():
        if v:
            print(f"  PAIRED {k:44s} {v['delta']:+.4f} [{v['lo']:+.4f},{v['hi']:+.4f}] "
                  f"sep={v['separated']}")


if __name__ == "__main__":
    main()
