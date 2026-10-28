import json
base = "E:/Projects/TanitAD-artifacts"
for d in ("v8f-dagger/M2b_s0_dag", "v8f-dagger/M2b_s1_dag", "v8f-dagoacc/M2b_s0_dagoacc", "v8f-dagoacc/M2b_s1_dagoacc"):
    s = json.load(open(f"{base}/{d}/summary.json"))
    a = json.load(open(f"{base}/{d}/argv.json"))["argv"]
    ai = {a[i]: a[i+1] for i in range(len(a)-1) if a[i].startswith("--")}
    print(d, {k: s.get(k) for k in ("steps","steps_mult","dagger_n","dagger_eligible","dagger_rows","dagger_nav_kept","ego_perturb_eligible")},
          "argv:", {k: ai.get(k) for k in ("--steps","--batch","--dagger-p","--dagger-seed","--w-acc","--steps-mult","--dagger-bank")})
    print("   other keys:", [k for k in s.keys() if "dagger" in k or "swap" in k or "own" in k])
