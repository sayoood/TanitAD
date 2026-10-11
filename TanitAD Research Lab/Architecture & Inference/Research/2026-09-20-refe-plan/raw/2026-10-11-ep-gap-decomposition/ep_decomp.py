"""EP decomposition on banked NAVSIM harness hooks (0 GPU). Per token: v0, PDM-Closed reference progress
(max_compliant_progress_m), agent progress, EP, and the PDMS row. Arms are read from their own hooks.json."""
import json, sys, numpy as np
S = "E:/Projects/TanitAD/data/refe_navtest/score"
ARMS = {"u400": "refe_stage2_sft4_B_u400_navtest_full", "human": "refe_p4_human_navtest",
        "teacher": "refe_p4_teacher_navtest", "final_rf": "refe_navtest_final_routefix"}
def load(d):
    h = json.load(open(f"{S}/{d}/{d}_hooks.json", encoding="utf-8"))["pdm_score_calls"]
    out = {}
    for c in h:
        r = c["row"]; pr = c["progress_raw_m"]
        out[c["token"]] = dict(v0=c["v0_mps"], ref=c["max_compliant_progress_m"], prog=pr[-1], pr=pr,
                               ep=r["ego_progress"], nc=r["no_at_fault_collisions"], dac=r["drivable_area_compliance"],
                               ttc=r["time_to_collision_within_bound"], c=r["comfort"], s=r["score"])
    return out
A = {k: load(v) for k, v in ARMS.items()}
toks = sorted(set.intersection(*[set(a) for a in A.values()]))
print("n tokens common:", len(toks), {k: len(v) for k, v in A.items()})
def arr(a, f): return np.array([A[a][t][f] for t in toks], float)
# sanity: progress_raw_m layout
t0 = toks[0]; print("example", t0, A["u400"][t0]["pr"], A["u400"][t0]["ref"], A["u400"][t0]["ep"])
for a in A:
    ep, nc, dac, s = arr(a, "ep"), arr(a, "nc"), arr(a, "dac"), arr(a, "s")
    gate = nc * dac
    ep_loss = (gate * (5 / 12) * (1 - ep)).mean() * 100
    mult_loss = ((1 - gate) * 1).mean() * 100
    print(f"{a:9s} PDMS {s.mean()*100:6.2f}  EP {ep.mean()*100:6.2f}  EP|gate {ep[gate>0].mean()*100:6.2f}  "
          f"PDMS lost to EP {ep_loss:5.2f}  tokens zeroed by NC*DAC {mult_loss:5.2f}%")
v0 = arr("u400", "v0"); ref = arr("u400", "ref")
# check ref identical across arms (reference is PDM-Closed, should not depend on the agent... unless max over compliant incl. agent)
for a in A:
    print(a, "ref==u400 ref frac", np.mean(np.isclose(arr(a, "ref"), ref, atol=1e-6)))
bins = [(0, 1), (1, 3), (3, 6), (6, 10), (10, 15), (15, 40)]
print("\nEP by v0 bin (all tokens): n | u400 human teacher final_rf | u400 PDMS lost to EP (share of total)")
tot = (arr("u400","nc")*arr("u400","dac")*(5/12)*(1-arr("u400","ep"))).sum()
for lo, hi in bins:
    m = (v0 >= lo) & (v0 < hi)
    g = arr("u400","nc")*arr("u400","dac")
    l = (g*(5/12)*(1-arr("u400","ep")))[m].sum()
    print(f"v0 [{lo:>2},{hi:>2}) n={m.sum():5d} | " + " ".join(f"{arr(a,'ep')[m].mean()*100:6.2f}" for a in A)
          + f" | {l/len(toks)*100:5.2f} PDMS ({l/tot*100:4.1f}%)")
ep = arr("u400", "ep"); eph = arr("human", "ep")
print("\nu400 EP histogram:", {f"<{b}": int((ep < b).sum()) for b in (0.2, 0.5, 0.8, 0.9, 0.95, 0.999)})
print("human EP histogram:", {f"<{b}": int((eph < b).sum()) for b in (0.2, 0.5, 0.8, 0.9, 0.95, 0.999)})
# progress ratio agent vs human (paired)
pu, ph = arr("u400", "prog"), arr("human", "prog")
m = ref > 5
print(f"\nref>5m tokens {m.sum()}: median u400/human progress ratio {np.median(pu[m]/np.maximum(ph[m],1e-3)):.3f}; "
      f"u400 < 0.8*human on {np.mean(pu[m] < 0.8*ph[m])*100:.1f}%, > 1.2*human on {np.mean(pu[m] > 1.2*ph[m])*100:.1f}%")
# stall: v0<1 and ref>5 (PDM moves off, agent does not)
st = (v0 < 1) & (ref > 5)
print(f"start-from-stop tokens (v0<1, ref>5 m): {st.sum()}; u400 EP {ep[st].mean()*100:.1f} human {eph[st].mean()*100:.1f}; "
      f"u400 progress<1 m on {np.mean(pu[st]<1)*100:.1f}%  human<1m on {np.mean(ph[st]<1)*100:.1f}%")
# where the EP points go: decile of (human EP - u400 EP)
d = eph - ep
print("paired human-u400 EP: mean", round(d.mean()*100,2), "; share of the paired EP gap from tokens where u400 EP<0.5:",
      round(d[ep < 0.5].sum()/d.sum()*100,1), "% ; from v0<1:", round(d[v0 < 1].sum()/d.sum()*100,1), "%")
json.dump({"tokens": toks}, open(sys.argv[1] if len(sys.argv) > 1 else "ep_tokens.json", "w"))
