"""mk_code.py: build the working code dirs from tip 2a416ca7 (git show) with ASSERTED path-constant substitutions only. No arithmetic is edited."""
import subprocess, hashlib, os, sys, difflib
MIR = "C:/Users/Admin/tanitad-push"
R = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw"
PK = "E:/Projects/TanitAD-artifacts/daghacc"
SCR_OLD = 'SCR = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad"'
SCR_NEW = f'SCR = "{PK}"'
def show(rel):
    return subprocess.run(["git", "-C", MIR, "-c", "safe.directory=*", "show", f"2a416ca7:{rel}"], capture_output=True, check=True).stdout
FILES = {  # dest -> (tip path, [(old, new, count)])
  "hb/v8f_hblend.py": (f"{R}/2026-10-25-hblend/v8f_hblend.py", [
      (SCR_OLD, SCR_NEW, 1),
      ('TREE, WORK, OUT = f"{SCR}/viz/tree", f"{SCR}/viz/work", f"{SCR}/hblend"',
       'TREE, WORK, OUT = "E:/Projects/TanitAD-artifacts/dagt1_tree_b43c7a9a", f"{SCR}/hb_work", f"{SCR}/hb"', 1),
      ('for _p in (f"{OUT}/deps", f"{TREE}/stack"):', 'for _p in (f"{TREE}/taniteval", f"{TREE}/stack"):', 1)]),
  "hs/v8f_hspec.py": (f"{R}/2026-10-25-hspec/v8f_hspec.py", [
      (SCR_OLD, SCR_NEW, 1), ('HB, OUT = f"{SCR}/hblend", f"{SCR}/hspec"', 'HB, OUT = f"{SCR}/hb", f"{SCR}/hs"', 1)]),
  "hacc/v8f_hacc.py": (f"{R}/2026-10-25-hacc/run/v8f_hacc.py", [
      (SCR_OLD, SCR_NEW, 1), ('HB, HS, OUT = f"{SCR}/hblend", f"{SCR}/hspec", f"{SCR}/hacc"', 'HB, HS, OUT = f"{SCR}/hb", f"{SCR}/hs", f"{SCR}/hacc"', 1)]),
  "hacc/accel_profile.py": (f"{R}/2026-10-25-hacc/run/accel_profile.py", []),
  "hacc/test_v8f_hacc.py": (f"{R}/2026-10-25-hacc/run/test_v8f_hacc.py", []),
  "hb/fit_windows_M2b_s0.npz": (f"{R}/2026-10-25-hblend/fit_windows_M2b_s0.npz", []),
  "hb/fit_windows_M2b_s1.npz": (f"{R}/2026-10-25-hblend/fit_windows_M2b_s1.npz", []),
  "hb/weights_M2b_s0.json": (f"{R}/2026-10-25-hblend/weights_M2b_s0.json", []),
  "hb/weights_M2b_s1.json": (f"{R}/2026-10-25-hblend/weights_M2b_s1.json", []),
  "hacc_banked/weights_HACC_M2b_s0.json": (f"{R}/2026-10-25-hacc/run/weights_HACC_M2b_s0.json", []),
  "hacc_banked/weights_HACC_M2b_s1.json": (f"{R}/2026-10-25-hacc/run/weights_HACC_M2b_s1.json", []),
  "hacc_banked/fit_M2b_s0.json": (f"{R}/2026-10-25-hacc/run/fit_M2b_s0.json", []),
  "hacc_banked/fit_M2b_s1.json": (f"{R}/2026-10-25-hacc/run/fit_M2b_s1.json", []),
}
log = ["# code/PATCHES.md: path-constant substitutions only (asserted counts); sources are tip 2a416ca7 via `git show`\n"]
for dest, (tp, subs) in FILES.items():
    raw = show(tp)
    out = os.path.join(PK, dest); os.makedirs(os.path.dirname(out), exist_ok=True)
    if subs:
        t = raw.decode("utf-8")
        t0 = t
        for old, new, n in subs:
            assert t.count(old) == n, (dest, old[:60], t.count(old))
            t = t.replace(old, new)
        data = t.encode("utf-8")
        log.append(f"## {dest}\n  source `{tp}` blob-bytes sha256 `{hashlib.sha256(raw).hexdigest()[:16]}`; patched sha256 `{hashlib.sha256(data).hexdigest()[:16]}`\n")
        for old, new, n in subs:
            log.append(f"  - `{old[:110]}`  ->  `{new[:110]}`  (x{n})\n")
    else:
        data = raw
        log.append(f"## {dest}\n  verbatim; sha256 `{hashlib.sha256(raw).hexdigest()[:16]}`\n")
    open(out, "wb").write(data)
open(os.path.join(PK, "code/PATCHES.md"), "w", newline="\n").write("".join(log))
print("ok", len(FILES))
