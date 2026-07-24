"""Decision-grade PAIRED comparison v3enc@10k vs flagship v1 (speedjerk-30k) on
the SAME 40 val episodes / same windows, via taniteval.ci
paired_episode_cluster_bootstrap. The legacy `heldout +/- ci95` is
overlapping_holdout_se (1.28-2.06x too narrow) and is NOT used here."""
import json, sys
sys.path.insert(0, "/root/taniteval")
import numpy as np, torch
from taniteval import ci, rollout

A, B = "flagship-v3enc-10k", "flagship-30k"
wa = rollout.load_windows(f"/root/taniteval/results/windows_{A}.pt")
wb = rollout.load_windows(f"/root/taniteval/results/windows_{B}.pt")

# --- alignment: the paired test is only valid on identical windows ----------
same_eid = list(wa["eid"]) == list(wb["eid"])
print(f"[align] n_a={len(wa['eid'])} n_b={len(wb['eid'])} eid_sequences_identical={same_eid}")
assert same_eid, "REFUSING a paired test on non-identical window sets"
assert torch.allclose(wa["gt"], wb["gt"], atol=1e-5), "GT differs -> not the same windows"
print(f"[align] GT identical, n_episodes={len(set(wa['eid']))}")

eid = wa["eid"]
def ade_per_window(w, upto=None):
    d = (w["pred"] - w["gt"]).norm(dim=-1)          # [N,4]
    return (d if upto is None else d[:, :upto]).mean(dim=1).numpy()
def de_at(w, i):
    return (w["pred"][:, i] - w["gt"][:, i]).norm(dim=-1).numpy()

print("\n=== PAIRED episode-cluster bootstrap (B=2000, 40 episodes) ===")
print(f"{'metric':14s} {'v3enc@10k':>10s} {'v1@30k':>9s} {'CV':>8s} "
      f"{'delta(v3enc-v1)':>16s} {'CI95':>22s}  sep")
rows = {}
metrics = [("ade_0_2s", ade_per_window(wa), ade_per_window(wb),
            (wa["cv"] - wa["gt"]).norm(dim=-1).mean(dim=1).numpy())]
for i, lab in enumerate(("de@0.5s", "de@1s", "de@1.5s", "fde@2s")):
    metrics.append((lab, de_at(wa, i), de_at(wb, i),
                    (wa["cv"][:, i] - wa["gt"][:, i]).norm(dim=-1).numpy()))
for name, a, b, cvv in metrics:
    r = ci.paired_episode_cluster_bootstrap(a, b, eid, n_boot=2000, seed=0)
    lo, hi = r["lo"], r["hi"]
    sep = "SEPARATED" if r["separated"] else "overlaps 0"
    print(f"{name:14s} {a.mean():10.4f} {b.mean():9.4f} {cvv.mean():8.4f} "
          f"{r['delta']:+16.4f} [{lo:+.4f}, {hi:+.4f}]  {sep}")
    rows[name] = dict(v3enc=float(a.mean()), v1=float(b.mean()),
                      cv=float(cvv.mean()), paired=r, separated=r["separated"])

# --- also vs the CV floor, paired, for v3enc alone -------------------------
print("\n=== v3enc@10k vs the CV floor (paired, same windows) ===")
a = ade_per_window(wa); c = (wa["cv"] - wa["gt"]).norm(dim=-1).mean(dim=1).numpy()
r = ci.paired_episode_cluster_bootstrap(a, c, eid, n_boot=2000, seed=0)
print(f"  ade_0_2s v3enc {a.mean():.4f} vs CV {c.mean():.4f} -> delta "
      f"{r['delta']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}] "
      f"({'SEPARATED' if r['separated'] else 'overlaps 0'}) "
      f"[positive = v3enc WORSE than CV]")
rows["vs_cv_ade_0_2s"] = dict(v3enc=float(a.mean()), cv=float(c.mean()), paired=r)
rows["estimator"] = "paired_episode_cluster_bootstrap (taniteval/ci.py), B=2000, 40 episodes"
rows["arms"] = {"a": A, "b": B}
json.dump(rows, open("/root/taniteval/results/paired_v3enc10k_vs_flagship30k.json", "w"),
          indent=2, default=float)
print("\n-> results/paired_v3enc10k_vs_flagship30k.json")
