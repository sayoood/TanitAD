"""PREREG_V8F_PLANTEST read-out: loads runs/*.npz, applies the card's committed table, writes RESULT_PLANTEST.json / .md. Tier T0. Estimator on every number:
EC-BOOT = paired episode-cluster bootstrap (141 episodes, B 2,000, seed 0, ratio of per-episode sums, 95 % percentile interval); interval-only separation (IMPL section 4)."""
import json, os, sys, hashlib
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import plantest_lib as P
import plantest_real as R

TAGS = ("M2b_s0", "M2b_s1")
FAM = {"LONGITUDINAL": "acc_mae", "LATERAL": "yaw_mae"}
UNIT = {"acc_mae": "m/s^2", "yaw_mae": "rad/s", "speed_err": "m/s", "heading_err": "rad", "ade": "m"}
BESIDE = ("speed_err", "heading_err", "ade")
PL = ("td3", "flat1")                                             # card log 2026-11-08 10:45: FLAT1 is the flat arm for outcomes and HIER-ADDS; FLAT did not pass control 1
ALL_K = ("td3", "flat", "flat1")


def load_run(tag, kind, goal, seed=0):
    p = f"{HERE}/runs/{tag}_{kind}_{goal}_s{seed}.npz"
    return dict(np.load(p)) if os.path.exists(p) else None


def metrics_for(ctx, plan, idx=None):
    A, v0 = (ctx.A, ctx.v0) if idx is None else (ctx.A[idx], ctx.v0[idx])
    return P.window_metrics(plan, A, v0)


def fmt(t):
    return f"{t[0]:.4f} [{t[1]:.4f}, {t[2]:.4f}]"


def main():
    out = {"estimator": "EC-BOOT: paired episode-cluster bootstrap, 141 episodes, B 2000, seed 0, ratio of per-episode sums, 95 % percentile", "tier": "T0", "controls": None}
    ctxs = {t: R.prepare(t) for t in TAGS}
    eids = ctxs["M2b_s0"].eid
    uniq, Wt, pos = P.boot_setup(eids)
    out["n_windows"], out["n_episodes"] = int(len(eids)), int(len(uniq))
    out["windows_dropped_no_2p4s_frame"] = 0
    M = {}                                                            # (tag, name) -> metrics dict
    for t in TAGS:
        c = ctxs[t]
        M[(t, "hold")] = metrics_for(c, P.hold_plan(c.A))
        M[(t, "prior")] = metrics_for(c, P.prior_plan(c.st, c.n, c.ch1))
        for k in ("td3", "flat", "flat1"):
            for g in P.GOALS:
                r = load_run(t, k, g)
                if r is not None:
                    M[(t, f"{k}_{g}")] = metrics_for(c, r["plan_raw"])
    res = {"per_seed": {}, "labels": {}, "hier_adds": {}, "beside": {}}
    # ----- rows: mean metrics
    means = {}
    for (t, name), m in M.items():
        for q in ("acc_mae", "yaw_mae") + BESIDE:
            means[f"{t}|{name}|{q}"] = P.boot_mean(m[q], pos, Wt)
    res["means"] = {k: list(v) for k, v in means.items()}
    # ----- deciding: paired differences and labels
    lab = {}
    for t in TAGS:
        for k in PL:
            if (t, f"{k}_true") not in M:
                continue
            for fam, q in FAM.items():
                tr = M[(t, f"{k}_true")][q]
                d_hold = P.boot_mean_diff(tr, M[(t, "hold")][q], pos, Wt)
                d_rand = P.boot_mean_diff(tr, M[(t, f"{k}_rand")][q], pos, Wt)
                d_stay = P.boot_mean_diff(tr, M[(t, f"{k}_stay")][q], pos, Wt)
                d_prior = P.boot_mean_diff(tr, M[(t, "prior")][q], pos, Wt)
                s_h, s_r = P.sep_label(*d_hold), P.sep_label(*d_rand)
                lab[(t, k, fam)] = dict(true_minus_hold=d_hold, true_minus_rand=d_rand, true_minus_stay=d_stay, true_minus_prior=d_prior, sep_hold=s_h, sep_rand=s_r,
                                        label=P.family_label(s_h, s_r), unit=UNIT[q])
    res["per_seed"] = {f"{t}|{k}|{fam}": v for (t, k, fam), v in lab.items()}
    for k in PL:
        for fam in FAM:
            ls = [lab[(t, k, fam)]["label"] for t in TAGS if (t, k, fam) in lab]
            res["labels"][f"{k}|{fam}"] = ls[0] if len(set(ls)) == 1 and len(ls) == 2 else "SEEDS-DISAGREE(" + "|".join(ls) + ")"
    # ----- HIER-ADDS: TD3 minus FLAT1 under G-TRUE (card log); TD3 minus the last-2-state FLAT is beside, no verdict
    for fam, q in FAM.items():
        row = {}
        for t in TAGS:
            if (t, "td3_true") in M and (t, "flat1_true") in M:
                d = P.boot_mean_diff(M[(t, "td3_true")][q], M[(t, "flat1_true")][q], pos, Wt)
                d0 = P.boot_mean_diff(M[(t, "td3_true")][q], M[(t, "flat_true")][q], pos, Wt) if (t, "flat_true") in M else None
                row[t] = dict(td3_minus_flat1=d, sep=P.sep_label(*d), td3_minus_flat_beside=d0, sep_flat=P.sep_label(*d0) if d0 else None)
        res["hier_adds"][fam] = dict(rows=row, HIER_ADDS=all(row[t]["sep"] == "better" for t in row) and len(row) == 2)
    # ----- beside metrics (speed, heading, ADE), monotonicity, fraction lowered, iterations, planner-seed floor
    beside = {}
    for t in TAGS:
        for k in ("td3", "flat", "flat1"):
            for g in P.GOALS:
                r = load_run(t, k, g)
                if r is None:
                    continue
                e = {}
                for lv in (1, 2, 3):
                    if f"mono{lv}" in r:
                        e[f"monotonicity_L{lv}"] = P.boot_mean(r[f"mono{lv}"], pos, Wt)
                        e[f"frac_cost_lowered_L{lv}"] = P.boot_mean((r[f"cost{lv}"] < r[f"init{lv}"]).astype(float), pos, Wt)
                        e[f"iters_L{lv}"] = float(r[f"iters{lv}"].mean())
                beside[f"{t}|{k}|{g}"] = e
    res["beside"]["cost_diagnostics"] = {k: {a: (list(b) if isinstance(b, tuple) else b) for a, b in v.items()} for k, v in beside.items()}
    floor = {}
    for t in TAGS:
        for k in ALL_K:
            r0, r1 = load_run(t, k, "true", 0), load_run(t, k, "true", 1)
            if r1 is None:
                continue
            m1 = metrics_for(ctxs[t], r1["plan_raw"])
            for fam, q in FAM.items():
                d = P.boot_mean_diff(m1[q], M[(t, f"{k}_true")][q], pos, Wt)
                floor[f"{t}|{k}|{fam}"] = dict(planner_seed1_minus_seed0=d, seed1_mean=P.boot_mean(m1[q], pos, Wt), sep=P.sep_label(*d))
    res["beside"]["planner_seed_replicate"] = floor
    # ----- untrained stacks (256 windows)
    un = {}
    idx = np.linspace(0, 1975, 256).astype(int)
    u_uniq, u_W, u_pos = P.boot_setup(eids[idx])
    c0 = ctxs["M2b_s0"]
    for r_ in (0, 1):
        tag = f"rand{r_}"
        if all(load_run(tag, k, "true") is None for k in ALL_K):
            continue
        for k in ALL_K:
            if load_run(tag, k, "true") is None or load_run(tag, k, "rand") is None:
                continue
            row = {}
            hold = metrics_for(c0, P.hold_plan(c0.A[idx]), idx) if False else P.window_metrics(P.hold_plan(c0.A[idx]), c0.A[idx], c0.v0[idx])
            for fam, q in FAM.items():
                tr = P.window_metrics(load_run(tag, k, "true")["plan_raw"], c0.A[idx], c0.v0[idx])[q]
                rd = P.window_metrics(load_run(tag, k, "rand")["plan_raw"], c0.A[idx], c0.v0[idx])[q]
                row[fam] = dict(true=P.boot_mean(tr, u_pos, u_W), rand=P.boot_mean(rd, u_pos, u_W), hold=P.boot_mean(hold[q], u_pos, u_W),
                                true_minus_rand=P.boot_mean_diff(tr, rd, u_pos, u_W), true_minus_hold=P.boot_mean_diff(tr, hold[q], u_pos, u_W))
                row[fam]["label_if_read_as_card"] = P.family_label(P.sep_label(*row[fam]["true_minus_hold"]), P.sep_label(*row[fam]["true_minus_rand"]))
            un[f"{tag}|{k}"] = row
    res["beside"]["untrained_256"] = un
    res["beside"]["untrained_n_windows"] = int(len(idx))
    out["result"] = res
    json.dump(out, open(f"{HERE}/RESULT_PLANTEST.json", "w", encoding="utf-8"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    write_md(out, means, lab, res, un, floor, beside)


def write_md(out, means, lab, res, un, floor, beside):
    L = []
    a = L.append
    a("# RESULT_PLANTEST: is today's hierarchical latent space plannable? (T0, zero training; agent, 2026-11-08)")
    a("Card `../PREREG_V8F_PLANTEST.md` (prefix sha256 `d5dbcc63...7783`), IMPL `IMPL_CHOICES_PLANTEST.md` (sha256 in `IMPL_CHOICES_PLANTEST.sha256`). **Tier T0: a world-model diagnostic with an oracle (true-future) goal, never a driving result.** Evidence class: MEASURED (this run, `runs/*.npz`, `RESULT_PLANTEST.json`) unless marked.")
    a(f"Estimator on every interval: EC-BOOT = paired episode-cluster bootstrap over {out['n_episodes']} episodes, B 2,000, seed 0, 95 % percentile; windows n = {out['n_windows']} (dropped for lacking the 2.4 s frame: 0). Separated-better = interval wholly below 0 (MAE, lower is better). The interval answers 'another draw of episodes'; it is blind to training-seed and planner-seed variance, hence both training seeds are required and the planner-seed floor is printed beside.")
    a("")
    a("**Controls: all four PASS for TD3 and for FLAT1 (`RESULT_PLANTEST_controls.md`). The last-2-state FLAT does not pass control 1 (card log, 2026-11-08 10:45), so FLAT1 is the flat arm for every outcome and for HIER-ADDS, and FLAT is reported beside with no verdict.**")
    a("")
    a("## Verdict (the card's committed table, per planner and family; both training seeds required)")
    a("| planner | family | M2b_s0 | M2b_s1 | decision label |")
    a("|---|---|---|---|---|")
    for k in PL:
        for fam in FAM:
            a(f"| {k.upper().replace('TD3', 'TD3')} | {fam} | {lab[('M2b_s0', k, fam)]['label']} | {lab[('M2b_s1', k, fam)]['label']} | **{res['labels'][f'{k}|{fam}']}** |")
    a("TACTICAL: not computed (n = 0: this read has no decision output). STRATEGIC: PAUSED (PI item 25, `PI_DECISION_QUEUE.md`).")
    a("")
    a("HIER-ADDS (TD3 minus FLAT1 under G-TRUE, separated-better at both seeds): " + "; ".join(f"{fam}: {'YES' if v['HIER_ADDS'] else 'NO'}" for fam, v in res["hier_adds"].items()))
    a("")
    a("## Deciding rows: MAE against the 12 logged actions (EC-BOOT mean [95 % interval])")
    for fam, q in FAM.items():
        a(f"### {fam} ({q}, {UNIT[q]})")
        a("| model | HOLD | PRIOR | TD3 true | TD3 rand | TD3 stay | FLAT1 true | FLAT1 rand | FLAT1 stay | FLAT true (beside, no verdict) |")
        a("|---|---|---|---|---|---|---|---|---|---|")
        for t in TAGS:
            g = lambda n: fmt(means[f"{t}|{n}|{q}"]) if f"{t}|{n}|{q}" in means else "-"
            a(f"| {t} | {g('hold')} | {g('prior')} | {g('td3_true')} | {g('td3_rand')} | {g('td3_stay')} | {g('flat1_true')} | {g('flat1_rand')} | {g('flat1_stay')} | {g('flat_true')} |")
        a("")
        a("| model | planner | TRUE - HOLD | sep | TRUE - RAND | sep | TRUE - STAY (beside) | TRUE - PRIOR (beside) | label |")
        a("|---|---|---|---|---|---|---|---|---|")
        for t in TAGS:
            for k in PL:
                v = lab[(t, k, fam)]
                a(f"| {t} | {k} | {fmt(v['true_minus_hold'])} | {v['sep_hold']} | {fmt(v['true_minus_rand'])} | {v['sep_rand']} | {fmt(v['true_minus_stay'])} | {fmt(v['true_minus_prior'])} | {v['label']} |")
        a("")
    a("## HIER-ADDS rows: TD3 minus FLAT1 under G-TRUE (negative = TD3 better)")
    a("| family | model | TD3 - FLAT1 | sep | TD3 - FLAT (beside, no verdict) | sep |")
    a("|---|---|---|---|---|---|")
    for fam, v in res["hier_adds"].items():
        for t, r in v["rows"].items():
            a(f"| {fam} | {t} | {fmt(r['td3_minus_flat1'])} | {r['sep']} | {fmt(r['td3_minus_flat_beside']) if r['td3_minus_flat_beside'] else '-'} | {r['sep_flat']} |")
    a("")
    a("## Beside, no bar: speed error, heading error and ADE of the unicycle integration, plan vs logged actions (G-TRUE; EC-BOOT mean [interval])")
    a("| model | planner/ref | speed err at 2.4 s [m/s] | heading err at 2.4 s [rad] | ADE [m] |")
    a("|---|---|---|---|---|")
    for t in TAGS:
        for n in ("hold", "prior", "td3_true", "td3_rand", "flat1_true", "flat1_rand", "flat_true", "flat_rand"):
            if f"{t}|{n}|ade" in means:
                a(f"| {t} | {n} | {fmt(means[f'{t}|{n}|speed_err'])} | {fmt(means[f'{t}|{n}|heading_err'])} | {fmt(means[f'{t}|{n}|ade'])} |")
    a("")
    a("## Beside, no bar: cost monotonicity per level (Tab. 10 statistic: -Spearman(goal distance, time); 1.0 = perfectly decreasing) and fraction of windows where planning lowered the cost below the best initial candidate")
    a("| run | level | monotonicity | frac lowered | mean iterations |")
    a("|---|---|---|---|---|")
    for key, e in beside.items():
        if key.split("|")[2] != "true":
            continue
        for lv in (1, 2, 3):
            if f"monotonicity_L{lv}" in e:
                a(f"| {key} | L{lv} | {fmt(e[f'monotonicity_L{lv}'])} | {fmt(e[f'frac_cost_lowered_L{lv}'])} | {e[f'iters_L{lv}']:.1f} |")
    a("")
    a("## Beside: planner-seed floor (planner seed 1 minus seed 0, G-TRUE, same checkpoint, EC-BOOT; a difference of this size is inference noise, not an effect)")
    for k, v in floor.items():
        a(f"* {k}: {fmt(v['planner_seed1_minus_seed0'])} ({v['sep']})")
    a("")
    a(f"## Beside: two untrained stacks (seeds 5000 + r) on M2b_s0's level-1 latents, {res['beside'].get('untrained_n_windows')} windows (the card's no-learned-dynamics reading; EC-BOOT on the subset; the label applies the card's rule for display only)")
    for k, row in un.items():
        for fam, v in row.items():
            a(f"* {k} {fam}: TRUE {fmt(v['true'])}, RAND {fmt(v['rand'])}, HOLD {fmt(v['hold'])}; TRUE-RAND {fmt(v['true_minus_rand'])}; TRUE-HOLD {fmt(v['true_minus_hold'])}; rule reads {v['label_if_read_as_card']}")
    a("")
    open(f"{HERE}/RESULT_PLANTEST_body.md", "w", encoding="utf-8").write("\n".join(L))


if __name__ == "__main__":
    main()
