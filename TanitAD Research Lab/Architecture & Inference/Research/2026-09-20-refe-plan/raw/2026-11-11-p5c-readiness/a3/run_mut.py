import os, shutil, subprocess, sys
H, K = sys.argv[1], sys.argv[2]
src = open(os.path.join(H, "p5c_rj5_path.py"), encoding="utf-8").read()
MUTS = {
 "no_wrong_key_check": ('if d["bank_key"] != want_key or d["bank_traj_sha256"] != b["sha"]:', 'if False:'),
 "override_never_fed": ('o["traj"] = unflat(j["traj"])', 'o["traj"] = unflat(r["traj"])'),
 "rekey_no_sha_check": ('if r.get("traj_sha256") != m["label_traj_sha256"]:', 'if False:'),
 "rekey_keeps_fed_sha": ('o["traj_sha256"] = m["bank_traj_sha256"]', 'o["traj_sha256"] = r.get("traj_sha256")'),
 "missing_filled_silently": ('raise Refused(f"{len(missing):,} feed rows have NO labeller record', 'pass  # raise Refused(f"{len(missing):,} feed rows have NO labeller record'),
 "absent_key_falls_back": ('    if absent:\n        raise Refused(', '    if False:\n        raise Refused('),
 "repair_raw_from_feed": ('"raw_sim8": bool(x["raw_sim8"])', '"raw_sim8": bool(y["raw_sim8"])'),
 "platform_guessed": ('    if undeclared:\n        raise Refused(', '    if False:\n        raise Refused('),
}
res = {}
for name, (a, b) in MUTS.items():
    assert src.count(a) == 1, (name, src.count(a))
    d = os.path.join(K, "mut", name)
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, "p5c_rj5_path.py"), "w", encoding="utf-8", newline="\n").write(src.replace(a, b))
    shutil.copy(os.path.join(H, "test_p5c_rj5_path.py"), d)
    bt = os.path.join(K, "pt", "mut_" + name)
    p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", f"--basetemp={bt}", "test_p5c_rj5_path.py"], cwd=d, capture_output=True, encoding="utf-8")
    last = [l for l in p.stdout.splitlines() if "passed" in l or "failed" in l][-1:]
    res[name] = (p.returncode, last)
    print(name, "RED" if p.returncode != 0 else "GREEN(!)", last, flush=True)
print("ALL_RED" if all(rc != 0 for rc, _ in res.values()) else "SOME_MUTANT_SURVIVED")
