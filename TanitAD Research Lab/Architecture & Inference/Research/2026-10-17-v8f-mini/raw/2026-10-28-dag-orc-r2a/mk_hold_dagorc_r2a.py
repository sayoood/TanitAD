"""Hold (stage only): r2's first launch of the DAG-ORC chain refused at its first run record (a foreign aside marker), the chain-specific marker fix
that relaunched it, the T1-kin chain's two-step K-BUILD as launched, and the refused launch's record. No register or PREREG edit.
python mk_hold_dagorc_r2a.py <live tip>"""
import hashlib
import os
import re
import subprocess
import sys

TIP = sys.argv[1]
GIT = ["git", "-C", "C:/Users/Admin/tanitad-push", "-c", "safe.directory=*"]
A = "E:/Projects/TanitAD-artifacts/"
R2, T1, HERE = A + "dagorc_r2/", A + "dagorc_t1/", A + "dagland/dagorc_r2a/"
RF = A + "v8f-dag-orc-oacc_r2a_marker_stop/M2b_s0_dagoaccorc/"
BL = "C:/Users/Admin/box2_ship/logs/"
H = "C:/Users/Admin/qland/hold_dagorc_r2a/"
V8 = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/"
PK = V8 + "raw/2026-10-28-dag-orc-r2a/"
A2 = V8 + "raw/2026-10-28-dag-orc-a2/"
assert not os.path.exists(H), H


def blob(p):
    r = subprocess.run(GIT + ["show", f"{TIP}:{p}"], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def put(rel, b):
    assert len(b) > 0 and len(H + rel) < 260, (rel, len(H + rel))
    os.makedirs(os.path.dirname(H + rel), exist_ok=True)
    open(H + rel, "wb").write(b)


rd = lambda p: open(p, "rb").read()
sha = lambda b: hashlib.sha256(b).hexdigest()

# ---- preconditions: the fixed r2 self-test passed, and the installed r2 files are the regenerated ones
st = rd(R2 + "logs/selftest_chain_r2.out").decode("utf-8", "replace")
assert "SELFTEST: 66/66 tests pass; MUTATION: 224/224 mutants killed, 0 survivors" in st and "ZZWRAPPER-END rc=0" in st
assert "killed   MX3-08 the aside lib the marker is DAG-OACC r2's shared r2_aside_done.txt" in st
for f in ("oaf_dagoaccorc_r2_lib.sh", "chain_v8f_dagoaccorc_r2.sh"):
    assert rd(R2 + "box2_ship/" + f) == rd("C:/Users/Admin/box2_ship/" + f), f + " installed != regenerated"
lib = rd(R2 + "box2_ship/oaf_dagoaccorc_r2_lib.sh")
assert sha(lib).startswith("e330241a") and b"r2_aside_done_dagoaccorc.txt" in lib
assert sha(rd(R2 + "box2_ship/chain_v8f_dagoaccorc_r2.sh")).startswith("239c9010")
t1s = rd(T1 + "selftest/SELFTEST.out").decode("utf-8", "replace")
assert "SELFTEST: 153/153 checks as expected" in t1s and "ZZDAGORCT1-SELFTEST-OK" in t1s
assert "SUMMARY: 11/11 checker mutants killed; survivors: none" in rd(T1 + "selftest/MUTATE_CHECKER.out").decode("utf-8", "replace")

n = 0
lfn = []


def bank(src, rel, only_if_changed_from=None):
    """bank src at PK+rel; with only_if_changed_from (a path in the A2 package) skip it when the A2 package already holds these bytes"""
    global n
    b = rd(src)
    if rel.endswith((".py", ".sh", ".md")) and b"\r" in b:
        b = b.replace(b"\r\n", b"\n")
        assert b"\r" not in b, rel
        lfn.append(rel)
    if only_if_changed_from is not None:
        old = blob(A2 + only_if_changed_from)
        if old is not None and old == b:
            return
    assert len(b) < 400_000, (rel, len(b))
    assert not re.search(rb"hf_[A-Za-z0-9]{20,}", b), rel
    assert blob(PK + rel) is None, rel
    put("stage/" + PK + rel, b)
    n += 1


bank(HERE + "R2A_NOTE.md", "R2A_NOTE.md")
bank(HERE + "mk_hold_dagorc_r2a.py", "mk_hold_dagorc_r2a.py")
# ---- the r2 chain's marker fix (only the files that differ from the A2 package's r2/)
for rel in ("gen/build_r2_lib.py", "gen/make_chain_r2.py", "gen/selftest_r2_tail.py", "gen/selftest_chain.py",
            "box2_ship/oaf_dagoaccorc_r2_lib.sh", "box2_ship/chain_v8f_dagoaccorc_r2.sh"):
    bank(R2 + rel, "r2/" + rel, "r2/" + rel)
bank(R2 + "logs/selftest_chain_r2.out", "r2/logs/selftest_chain_r2.out")
bank(R2 + "logs/pre_marker_fix/selftest_chain_r2.out", "r2/logs/selftest_chain_r2_pre_marker_fix.out", "r2/logs/selftest_chain_r2.out")
bank(R2 + "DRY_RUN_installed_markerfix.out", "r2/DRY_RUN_installed_markerfix.out")
# ---- the T1-kin chain as launched (the two-step K-BUILD; only files that differ from the A2 package's t1/)
for rel in ("dagorct1_lib.py", "dagorct1_selftest.py", "dagorct1_chain.sh", "dagorct1_chain.cmd", "PINS.txt", "NOTES.md", "MANIFEST.txt", "DRY_RUN.out",
            "wake_dagorct1.sh", "selftest/SELFTEST.out", "selftest/MUTATE_CHECKER.out", "selftest/mutate_checker.py"):
    bank(T1 + rel, "t1/" + rel, "t1/" + rel)
for rel in ("kbuild_r2_launch_8c24823f.json", "logs/kbuild_r2_from_tokens.json", "logs/DRY_RUN_r2env_processlocal.out"):
    bank(T1 + rel, "t1/" + rel)
# ---- the refused launch: its run record (r1's checkpoint and counts under r2's gate token), its logs, and the two markers' provenance
for f in ("summary.json", "selfcheck.json", "argv.json", "launch_gate_token.json", "SHA256SUMS.txt", "train.log"):
    bank(RF + f, "r2a_refused/M2b_s0_dagoaccorc/" + f)
assert b"6ea4d6baf3f806b32c2d42879089f7c7731a45f0eb6d5b4f85d3f44eeb21208f" in rd(RF + "SHA256SUMS.txt")
assert b"ego_perturb_stream" not in rd(RF + "summary.json")
for src, rel in ((BL + "chain_v8f_dagoaccorc_r2.out_r2a_marker_stop", "r2a_refused/chain_v8f_dagoaccorc_r2.out"),
                 (BL + "v8fdagoaccorc_r2_guard.out_r2a_marker_stop", "r2a_refused/v8fdagoaccorc_r2_guard.out"),
                 (BL + "v8fdagoaccorc10280327_r2aside.log", "r2a_refused/box2_aside_10280327_ALREADY_DONE.log"),
                 (BL + "v8fdagoacc10271620_r2aside.log", "r2a_refused/box2_aside_DAGOACC_r2_10271620_marker_origin.log"),
                 (BL + "v8fdagoaccorc10280348_r2aside.log", "relaunch/box2_aside_10280348.log")):
    bank(src, rel)
assert b"ASIDE-ALREADY-DONE" in rd(BL + "v8fdagoaccorc10280327_r2aside.log")
if lfn:
    put("stage/" + PK + "LF_NORMALISED.txt", ("CRLF -> LF on banking (content otherwise unchanged):\n" + "\n".join(sorted(lfn)) + "\n").encode()); n += 1
note = rd(HERE + "R2A_NOTE.md").lower()
for w in (b"void", b"superseded", b"retracted", b"formerly", b"no longer", b"correction", b"withdrawn"):
    assert w not in note, w
put("msg.txt", ("DAG-ORC r2: the first launch (03:27) was refused at its first run record. Its box-2 aside read DAG-OACC r2's marker"
                " (r2_aside_done.txt, same runs root) as its own, moved nothing, and the queue skipped A s0's training (r1's checkpoint"
                " 6ea4d6ba, no levers.ego_perturb_stream). The marker is now chain-specific (r2_aside_done_dagoaccorc.txt; self-test 66/66,"
                " 224/224 mutants killed incl. the shared-name mutant); relaunched 03:48 with every r1 item moved aside. The T1-kin chain"
                " (two-step K-BUILD, 153/153) launched. Package raw/2026-10-28-dag-orc-r2a/ (%d files).\n\n"
                "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n" % n).encode())
print("hold written", H, n, "package files")
