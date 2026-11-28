"""D-REFAV1-CCOS-EVAL — GATE 1 of `stack/tanitad/eval/echo_gate.py` on a refav1 dump: can the
planner arm beat its OWN measured dynamics? Paired, per horizon slot, against every reference the
gate requires (`REQUIRED_REFERENCES = ("ha", "ha0_ext")`) plus `ha0`, episode-cluster bootstrap.

Per-window ADE ``[N, K]`` is the Euclidean error of the arm's path against the dump's GT ``g``
at each of the K horizon slots — computed here from the dump arrays directly (float64), which is
the same quantity `four_families` reduces to ``ade_m`` when averaged over slots.

⚠️ Gate 2 (`ego_intervention_test`) and gate 2b (`source_ablation_test`) need the MODEL re-run under
deranged inputs; they are NOT computed from a dump. This tool reports gate 1 and writes the
gate-2/2b status as REFUSED with the reason and the n, so `assert_not_echoing` is not called with
a fabricated gate 2. The functional ablation for refav1 is a separate GPU run
(`refav1_source_ablation.py`, queued).
"""
from __future__ import annotations
import argparse, glob, json, os, sys
import numpy as np


def load(d):
    E = []
    for f in sorted(glob.glob(os.path.join(d, "ep*.npz"))):
        with np.load(f) as z:
            E.append({k: z[k] for k in z.files} | {"_eid": os.path.basename(f)[:-4]})
    return E


def ade_slots(pred, g):
    return np.linalg.norm(pred[..., :2].astype(np.float64) - g[..., :2].astype(np.float64), axis=-1)  # [N, K]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--arm", default="cl")
    ap.add_argument("--stack", required=True)
    ap.add_argument("--taniteval", required=True, help="the INNER taniteval package's parent (…/taniteval)")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    sys.path.insert(0, a.stack); sys.path.insert(0, a.taniteval)
    from tanitad.eval import echo_gate as EG
    E = load(a.dump)
    g = np.concatenate([e["g"] for e in E])
    eid = np.concatenate([[e["_eid"]] * len(e["ws"]) for e in E])
    arm = ade_slots(np.concatenate([e[a.arm] for e in E]), g)
    refs = {}
    for r in ("ha", "ha0_ext", "ha0"):
        if all(r in e for e in E):
            refs[r] = ade_slots(np.concatenate([e[r] for e in E]), g)
    gate1 = EG.echo_gate(arm_ade=arm, references=refs, eid=eid, n_boot=a.n_boot, seed=0)
    K = arm.shape[1]
    summary = {}
    for slot in (K // 2 - 1, K - 1):       # 1.0 s and 2.0 s
        row = gate1["slots"][slot]
        summary[f"slot{slot}_{(slot + 1) * 0.2:.1f}s"] = {
            "arm_mean": row["arm_mean"],
            **{nm: {"ref_mean": c["ref_mean"], "delta_ref_minus_arm": c["delta_ref_minus_arm"],
                    "ci": c["ci"], "separated": c["separated"], "relative_margin": c["relative_margin"],
                    "passes": c["passes"]} for nm, c in row["vs"].items()}}
    # the gate's own verdict logic for gate 1 alone, without fabricating gate 2
    slot = K - 1
    row = gate1["slots"][slot]
    fails = [nm for nm in ("ha", "ha0_ext") if not row["vs"][nm]["passes"]]
    verdict = ("GATE1_PASS" if not fails else "GATE1_FAIL: did not clear " + ", ".join(fails))
    if row["vs"]["ha"]["passes"] and not row["vs"]["ha0_ext"]["separated"]:
        verdict += " | ECHO-HARDER TRAP: cleared ha but tied ha0_ext"
    out = {"tool": "run_echo_gate.py", "dump": a.dump, "arm": a.arm, "n_windows": gate1["n_windows"],
           "n_episodes": gate1["n_episodes"], "references": gate1["references"], "tier": "T1 (all arms)",
           "estimator": "paired_episode_cluster_bootstrap", "gate1": gate1, "gate1_summary": summary,
           "gate1_verdict_primary_slot": verdict,
           "gate2": {"status": "REFUSED", "n": 0,
                     "reason": "ego_intervention_test needs the model with goal_provenance nodes; refav1's "
                               "planner is not wired to it — a dump cannot supply it"},
           "gate2b": {"status": "REFUSED", "n": 0,
                      "reason": "source_ablation_test re-runs the arm under deranged scene / ego inputs; "
                                "queued as refav1_source_ablation.py on a stratified subset (GPU)"},
           "assert_not_echoing": "NOT CALLED — gate 2 is mandatory for it and was not computed; calling it "
                                 "with a fabricated gate 2 would be a decoration, not a gate"}
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1, default=float)
    print(f"[echo gate 1] {a.arm} on {a.dump}: n={gate1['n_windows']}/{gate1['n_episodes']}  {verdict}")
    for k, v in summary.items():
        print(" ", k, "arm", round(v["arm_mean"], 4),
              {nm: (round(c["ref_mean"], 4), round(c["delta_ref_minus_arm"], 4), [round(x, 4) for x in c["ci"]], c["separated"], c["passes"]) for nm, c in v.items() if nm != "arm_mean"})


if __name__ == "__main__":
    main()
