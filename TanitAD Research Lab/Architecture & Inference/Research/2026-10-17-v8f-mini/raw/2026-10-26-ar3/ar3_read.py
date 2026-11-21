"""AR-3 reader: prints the state of the three detached jobs. Run: C:/Users/Admin/venvs/tanitad/Scripts/python.exe <this file>   (no GPU, no repo import)."""
import json, os, re, glob, sys, xml.etree.ElementTree as ET
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
SP = os.path.dirname(os.path.abspath(__file__)); RAW = SP + "/raw"
def rd(p, default=None):
    try: return open(p, encoding="utf-8", errors="replace").read()
    except OSError: return default
print("== driver.log"); print(rd(RAW + "/driver.log", "(no driver.log)"))
print("== live4 (in-trainer gate: G-DVB + G-LIVE through the real encoder + probes + identity probe), CPU, tip trainer")
lg = rd(RAW + "/selfcheck_L_live4.log", "")
m = re.search(r"ZZRC rc=(\d+) seconds=(\d+)", lg); print("   finished:", m.group(0) if m else "NO (still running, or died: see the log tail)")
print("   last log line:", [l for l in lg.replace("\r", "\n").split("\n") if l.strip()][-1][:300] if lg.strip() else "(empty)")
sc = "C:/Users/Admin/ar3_work/runs/L_live4/selfcheck.json"; import shutil; os.makedirs(RAW + "/runs/L_live4", exist_ok=True)
if os.path.exists(sc):
    shutil.copy(sc, RAW + "/runs/L_live4/selfcheck.json")
    d = json.load(open(sc)); lv = d["live"]; e = d["levers"]["encoder"]
    print("   declared_vs_built:", d["declared_vs_built"], "| live.pass:", lv["pass"])
    print("   live: A_grad_step0 =", lv["lora_A_grad_step0"], "(must be exactly 0.0) | B_grad_step0 =", lv["lora_B_grad_step0"], "| frozen DINOv3 grad =", lv["control_frozen_dinov3_grad_sum"], "| prior ctrl =", lv["control_prior_grad_sum"])
    print("   live: missing_terms", lv["missing_terms"], "| unexpected_dead", lv["unexpected_dead"], "| n_lora_matrices", lv["n_lora_matrices"], "| n_trainable_heads", lv["n_trainable_heads"])
    print("   encoder: trainable", e["lora_trainable"], "| r/alpha/targets", e["lora_r"], e["lora_alpha"], e["lora_targets"], "| weights_sha256", e["weights_sha256"], "| == builder:", e["weights_sha256"] == e["builder_model_sha256"])
    print("   other probes present:", [k for k in ("code_mix_probe", "o_interface_probe", "acc_probe", "admissibility") if k in d], "| identity_probe:", d.get("identity_probe"))
    print("   argv_sha (trainer's):", d["argv_sha"], "| accum:", d["levers"].get("accum"))
else:
    print("   no selfcheck.json yet")
print("== pytest, LoRA/v8f-mini subset (P1)")
def junit(p):
    if not os.path.exists(p): return None
    r = ET.parse(p).getroot(); ss = r.iter("testsuite"); T = F = E = S = 0; bad = []; skips = {}
    for s in ss:
        T += int(s.get("tests", 0)); F += int(s.get("failures", 0)); E += int(s.get("errors", 0)); S += int(s.get("skipped", 0))
    for tc in r.iter("testcase"):
        for ch in tc:
            if ch.tag in ("failure", "error"): bad.append(tc.get("classname", "") + "::" + tc.get("name", ""))
            if ch.tag == "skipped": skips.setdefault((ch.get("message") or "")[:110], []).append(tc.get("name"))
    return dict(tests=T, failures=F, errors=E, skipped=S, bad=bad, skip_reasons={k: len(v) for k, v in skips.items()})
for tag, p in (("P1 subset", RAW + "/pytest_lora_subset.xml"), ("G-SUITE stack", RAW + "/gsuite_stack.xml"), ("G-SUITE taniteval", RAW + "/gsuite_taniteval.xml"), ("G-SUITE tools", RAW + "/gsuite_tools.xml")):
    j = junit(p); print(f"   {tag}:", json.dumps(j, indent=1)[:2500] if j else "NOT FINISHED (no junit xml): tail of log -> " + (rd(p.replace('.xml', '.log'), '(no log)') or '')[-300:].replace('\n', ' | '))
