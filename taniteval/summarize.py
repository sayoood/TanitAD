import json, glob, os

print("========== DIAGNOSTIC PANELS (floor / ceiling / skill) ==========")
for p in sorted(glob.glob("/root/taniteval/results/diag_*.json")):
    j = json.load(open(p))
    d = j["diagnostic"]
    f, ego, lat = d["kinematic_floor"], d["ego_status_ceiling"], d["latent_ceiling"]
    key = os.path.basename(p)[5:-5]
    print(f"\n=== {key}  n={d['n_windows']}  step={j.get('ckpt_step')}  wall={j.get('wall_s')}s ===")
    print(f"  model_ade={d['model_ade_0_2s']:.4f}  floor(best3)={f['best_of_3_ade_0_2s']:.4f}")
    print(f"  per_baseline={f['per_baseline_ade_0_2s']}  wins={f['which_baseline_wins']}")
    if ego:
        print(f"  ego-status ceiling={ego['held_out_ade_0_2s']:.4f}  ctrv={ego['ctrv_ade_0_2s']:.4f}  "
              f"beats_ctrv={ego['ridge_beats_ctrv']}  (alpha={ego['ridge_alpha']}, r2={ego['fit_r2']})")
    if lat:
        print(f"  latent ceiling={lat['held_out_ade_0_2s']:.4f}  state_dim={lat['state_dim']}  "
              f"(alpha={lat['ridge_alpha']}, r2={lat['fit_r2']})")
    sk, skc = d["skill_score"]["by_speed"], d["skill_score"]["by_curvature"]
    print(f"  skill/floor by SPEED: " + "  ".join(f"{k}={v['skill_vs_floor']}(m{v['model_l2']}/f{v['floor_l2']})" for k, v in sk.items()))
    print(f"  skill/floor by CURV : " + "  ".join(f"{k}={v['skill_vs_floor']}" for k, v in skc.items()))
    st = d["falsifiers"]["skill_on_straights"]
    print(f"  FALSIFIER straights: skill={st['skill_vs_floor']}  near_trivial={st['near_trivial_competitive']}")
    print(f"  FALSIFIER ridge-vs-ctrv: {d['falsifiers']['ridge_ceiling_vs_ctrv']}")

print("\n\n========== PLANNING PANELS (route base-rate / behavior decodability) ==========")
for p in sorted(glob.glob("/root/taniteval/results/plan_*.json")):
    j = json.load(open(p))
    key = os.path.basename(p)[5:-5]
    if j.get("skipped"):
        print(f"\n=== {key}: SKIPPED ({j['skipped']}) ==="); continue
    s, b = j.get("strategic", {}), j.get("behavior_decodability", {})
    print(f"\n=== {key}  step={j.get('ckpt_step')} ===")
    print(f"  route_acc_follow={s.get('route_acc_follow')}  base_rate={s.get('majority_route_base_rate')}  "
          f"route_skill={s.get('route_skill_vs_chance')}  (n_valid={s.get('n_route_valid')})")
    if "skipped" in b:
        print(f"  behavior probe: SKIPPED ({b['skipped']})")
    else:
        print(f"  maneuver bal-acc={b.get('maneuver_balanced_accuracy')}  chance={b.get('chance_balacc')}  "
              f"beats={b.get('beats_chance')}  decod_vs_chance={b.get('decodability_vs_chance')}  "
              f"(raw={b.get('maneuver_accuracy_raw')}, maj={b.get('maneuver_majority_acc')}, macroF1={b.get('macro_f1')})")
    print(f"  verdict: {j.get('verdict')}")
