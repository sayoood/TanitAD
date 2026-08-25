"""E-DEC-38 — THE COUNTERFACTUAL PHYSICS TEST (the PI's mandate 3, directly)

⭐ THE QUESTION, IN THE PI'S WORDS: *"if the vehicle in front decelerates, the ego
vehicle must react on it"*. That is a COUNTERFACTUAL claim — it asks what the
world does IF the ego acts differently. Every other probe in this campaign asks
what the representation CONTAINS; this one asks whether the world model can
answer a what-if. It is the first test in the programme that is a genuine
counterfactual rather than a reconstruction.

⛔ WHY IT CANNOT SIMPLY DECODE "HEADWAY FROM ẑ". E-DEC-32c measured that NOTHING
in the programme carries `lead_range_m` or `lead_closing` held-out (a constant
beats every arm, t 11.5 / t 28.4, 23/23 clips). So a test that needs a range
decoder is untestable today. The three questions below are chosen to be
answerable with what the latent demonstrably has.

THE THREE QUESTIONS, each with its own exact null:

  Q1  ACTION SENSITIVITY. Roll the identical window twice — BRAKE vs MAINTAIN —
      and measure ||ẑ_brake − ẑ_maintain|| / ||ẑ_true − z_last||. NULL: 0 (an
      action-blind predictor). SCALE: the same window's response to a 10 %
      LATENT nudge, which is the E-DEC-30 positive control.

      ⛔⛔ REPORT THE RAW RESPONSE AND THE RATIO SIDE BY SIDE, WITH THE
      DENOMINATOR PRINTED (C157). The control value is an ARM PROPERTY and it
      varies 77x across arms (postrain10k 13.09 vs rdw8p30k 0.17). MEASURED
      2026-08-25: ranking arms by Q1/ctrl INVERTED the ordering — the two LARGEST
      raw brake-vs-maintain responses in the programme (postrain10k 0.03083,
      splitp30k 0.01561, both DISTILLED) came LAST on the ratio, and
      Spearman(drift, response) flipped from -0.310 raw to +0.762 normalised. I
      reported a "distilled arms trade content for action-response" finding off
      the ratio; it was the denominator, not the models.
      ⚠️ THIS IS C137's DEFECT REINTRODUCED — "a metric whose denominator is an
      arm property" — inside the instrument built to avoid it. The ratio is still
      the right SCALE-FREE comparison for one arm against itself; it is NOT a
      cross-arm ranking. Both columns, always. A model whose
      response to "brake vs coast" is a few percent of its response to latent
      jitter cannot be reasoning about braking.

  Q2  ACTION IDENTIFIABILITY. Among {true future actions} ∪ {K counterfactuals},
      is the TRUE one's rollout closest to the observed future latent z_{t+k}?
      ⭐ EXACT CHANCE LEVEL = 1/(1+K), so this needs no external baseline. This
      is O11's training objective evaluated at TEST time on HELD-OUT clips.
      ⚠️ Ties are credited at chance, not to the target — the naive argmax reads
      1.0000 for a completely action-blind predictor (the defect caught in
      `o11_pick_acc` by `test_o11_counterfactual.py`).

  Q3  ⭐⭐ THE PHYSICS-SPECIFIC ONE, AND THE POINT OF THE FILE. Split the windows
      by whether the LEAD IS CLOSING (the gap shrinking — the PI's "decelerates")
      or NOT, and compare Q1/Q2 between the two sets. **If the model has learned
      driving physics, the action should MATTER MORE when a lead is closing** —
      braking is consequential there and irrelevant in free flow. A model that is
      equally action-sensitive in both regimes has learned an action response
      that is not about the scene, which is exactly what "physics" would rule out.

CONTROLS, fixed before running:
  * `constant`  — reads the no-information value EXACTLY (Q2 chance = 1/(1+K)).
  * latent +10 % noise — the E-DEC-30 positive control; without it a small Q1 is
    unreadable, because it could mean a dead predictor rather than an
    action-blind one.
  * TIME-SHUFFLED actions — the E-DEC-28b control; if a shuffled action sequence
    scores the same, whatever is happening carries no action semantics.
  * n per cell, and the CLOSING/NOT-CLOSING split sizes, printed.

⚠️ SCOPE. T0-DIAGNOSTIC — what the representation supports, NEVER "the car
drives". Held-out, lead-matched. MEASURED (ours; dev-box RTX 4060).
⚠️ Run this on the PRE-O11 arms first: the o11p30k comparison is only readable
against a matched baseline, and that baseline must exist BEFORE the arm lands.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

SP = Path(__file__).resolve().parent
sys.path.insert(0, str(SP))
sys.path.insert(0, str(SP / "sp2"))
sys.path.insert(0, r"C:\Users\Admin\tanitad-mirror\stack")
LEAD = Path(os.environ.get("SPD_CORPUS",
                           str(SP / "sp2/cache/physicalai-val130-heldout")))
LABELS = Path(os.environ.get("SPD_LABELS", str(SP / "sp2/val130_agents.jsonl")))
OUT = Path(os.environ.get("SPD_OUT", str(SP / "physics.json")))
ARMS = os.environ.get("SPD_ARMS", "rdw8p30k,splitp30k,scale1").split(",")
MIN_LEAD, N_CLIPS, F, K, N_CF = 20, 24, 100, 6, 3
BRAKE, MAINTAIN = -0.30, 0.0          # accel channel, in the corpus's units


def main() -> int:
    import v7tiny_g2 as G
    from tanitad.models.flagship_v15 import SPEED_SCALE

    dev = torch.device("cuda")
    LAB = {}
    with open(LABELS, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                LAB.setdefault(r["clip_id"], {})[int(r["frame_idx"])] = r.get("agents", [])

    def lead_series(cid, m):
        rr = np.full(m, np.nan)
        for i in range(m):
            if i not in LAB.get(cid, {}):
                continue
            inl = [x["cx"] for x in LAB[cid][i]
                   if abs(x.get("cy", 9e9)) < 1.8 and x.get("cx", -1) > 0]
            if inl:
                rr[i] = min(inl)
        clo = np.full(m, np.nan)
        clo[1:] = rr[1:] - rr[:-1]        # NEGATIVE = the gap is CLOSING
        return rr, clo

    def n_lead(cid):
        return sum(1 for i in range(F)
                   if any(abs(x.get("cy", 9e9)) < 1.8 and x.get("cx", -1) > 0
                          for x in LAB.get(cid, {}).get(i, [])))

    clips = [c for c in sorted(LEAD.glob("*.v2ep.pt"))
             if torch.load(c, map_location="cpu",
                           weights_only=False)["clip_id"] in LAB]
    clips = [c for c in clips
             if n_lead(torch.load(c, map_location="cpu",
                                  weights_only=False)["clip_id"]) >= MIN_LEAD][:N_CLIPS]
    present = [a for a in ARMS if (SP / f"v7tiny_{a}" / "ckpt.pt").is_file()]
    print(f"\n  E-DEC-38 · THE COUNTERFACTUAL PHYSICS TEST"
          f"\n  arms {present} · {len(clips)} lead-matched held-out clips · k={K}\n",
          flush=True)

    rep = {"_evidence_class": "MEASURED (ours; dev-box RTX 4060)",
           "eval_tier": "T0-DIAGNOSTIC", "split": "HELD-OUT, LEAD-MATCHED",
           "brake_accel": BRAKE, "maintain_accel": MAINTAIN,
           "q2_chance": round(1.0 / (1 + N_CF), 4), "arms": {}}

    for arm in present:
        w, st = G.load_arm(arm, dev)
        W = int(w.window)
        acc = {"closing": {"q1": [], "q2": [], "ctrl": [], "shuf": []},
               "not_closing": {"q1": [], "q2": [], "ctrl": [], "shuf": []}}
        rng = np.random.default_rng(0)
        with torch.no_grad():
            for c in clips:
                d, raw, off, n_all, _ = G.frames_of(c)
                n = min(n_all, F)
                z, act, spd = G.encode_clip(w, c, dev, F)
                zt = z.float()
                rr, clo = lead_series(d["clip_id"], n)
                for i in range(0, len(zt) - W - K, 2):
                    j = i + W - 1
                    tgt = j + K
                    if tgt >= n or np.isnan(clo[tgt]):
                        continue
                    win = zt[i:i + W][None].to(dev).clone()
                    base = zt[j].to(dev).reshape(-1)
                    vv = (spd[i] / SPEED_SCALE).view(1, 1, 1).expand(1, W, 1).to(dev)
                    aa = act[i:i + W][None].to(dev).float()
                    if aa.shape[1] != W:
                        continue

                    def roll(a3, wn=win):
                        o = w.predictor(wn, a3)[1]
                        return o.reshape(-1)[:zt.shape[1]]

                    ztrue = roll(torch.cat([aa, vv], -1))
                    den = float((ztrue - base).norm())
                    if den < 1e-9:
                        continue
                    # Q1 — BRAKE vs MAINTAIN on the accel channel
                    ab, am = aa.clone(), aa.clone()
                    ab[..., 1] = BRAKE
                    am[..., 1] = MAINTAIN
                    zb = roll(torch.cat([ab, vv], -1))
                    zm = roll(torch.cat([am, vv], -1))
                    q1 = float((zb - zm).norm()) / den
                    # positive control: a 10 % LATENT nudge, same window
                    pert = win + 0.10 * win.std() * torch.randn_like(win)
                    ctrl = float((roll(torch.cat([aa, vv], -1), pert)
                                  - ztrue).norm()) / den
                    # Q2 — is the TRUE action's rollout closest to z_{t+k}?
                    zf = zt[tgt].to(dev).reshape(-1)
                    ds = [float((ztrue - zf).norm())]
                    for _ in range(N_CF):
                        r0 = int(rng.integers(0, max(1, len(act) - W)))
                        a2 = act[r0:r0 + W][None].to(dev).float()
                        if a2.shape[1] != W:
                            a2 = aa
                        ds.append(float((roll(torch.cat([a2, vv], -1)) - zf).norm()))
                    dsa = np.array(ds)
                    tied = np.isclose(dsa, dsa.min(), rtol=1e-9, atol=1e-12)
                    q2 = float(tied[0] / tied.sum())     # ties credited at CHANCE
                    # time-shuffled control for Q1: two RANDOM action sequences
                    r1 = int(rng.integers(0, max(1, len(act) - W)))
                    r2 = int(rng.integers(0, max(1, len(act) - W)))
                    s1 = act[r1:r1 + W][None].to(dev).float()
                    s2 = act[r2:r2 + W][None].to(dev).float()
                    shuf = (float((roll(torch.cat([s1, vv], -1))
                                  - roll(torch.cat([s2, vv], -1))).norm()) / den
                            if s1.shape[1] == W and s2.shape[1] == W else np.nan)

                    g = "closing" if clo[tgt] < -0.05 else "not_closing"
                    acc[g]["q1"].append(q1); acc[g]["q2"].append(q2)
                    acc[g]["ctrl"].append(ctrl); acc[g]["shuf"].append(shuf)
        del w
        torch.cuda.empty_cache()

        rep["arms"][arm] = {"step": int(st), "regimes": {}}
        print(f"  === {arm} (step {st}) ===")
        print(f"  {'regime':<14}{'n':>6}{'RAW Q1':>12}{'[ctrl] denom':>14}"
              f"{'Q1/ctrl':>10}{'Q2 pick':>10}{'chance':>9}")
        print(f"  {'':14}{'':6}{'<- CROSS-ARM':>12}{'ARM PROPERTY':>14}"
              f"{'within-arm':>10}")
        for g in ("closing", "not_closing"):
            a = acc[g]
            if not a["q1"]:
                print(f"  {g:<14}{0:>6}   (no windows)"); continue
            q1, ct = float(np.mean(a["q1"])), float(np.mean(a["ctrl"]))
            q2 = float(np.mean(a["q2"]))
            sh = float(np.nanmean(a["shuf"]))
            rep["arms"][arm]["regimes"][g] = {
                "n": len(a["q1"]), "q1_brake_vs_maintain": round(q1, 5),
                "q1_positive_control": round(ct, 5),
                "q1_over_control": round(q1 / max(ct, 1e-12), 5),
                "q1_shuffled_pair": round(sh, 5),
                "q2_pick_true_action": round(q2, 4),
                "q2_chance": round(1.0 / (1 + N_CF), 4)}
            print(f"  {g:<14}{len(a['q1']):>6}{q1:>12.5f}{ct:>14.5f}"
                  f"{q1 / max(ct, 1e-12):>10.4f}{q2:>10.4f}"
                  f"{1.0 / (1 + N_CF):>9.4f}")
        r = rep["arms"][arm]["regimes"]
        if "closing" in r and "not_closing" in r:
            a, b = r["closing"]["q1_over_control"], r["not_closing"]["q1_over_control"]
            rep["arms"][arm]["q3_closing_over_free"] = round(a / max(b, 1e-12), 4)
            rep["arms"][arm]["_cross_arm_metric"] = "q1_brake_vs_maintain (RAW)"
            rep["arms"][arm]["_within_arm_metric"] = "q1_over_control"
            print(f"  ⭐ Q3  action sensitivity CLOSING / FREE-FLOW = "
                  f"{a / max(b, 1e-12):.4f}   (1.0 = the action matters no more "
                  f"when a lead is closing)")
        print()

    OUT.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
