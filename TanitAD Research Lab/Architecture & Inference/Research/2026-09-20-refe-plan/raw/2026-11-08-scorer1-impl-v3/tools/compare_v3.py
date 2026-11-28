"""Read F:/scorer1_v3/res/results.txt, classify each test, compare the 7 accepted failures with the base logs.
usage: compare_v3.py [res_dir] -> prints and writes <res_dir>/COMPARE.txt
PASS = rc 0 and a ZZ...PASS token (or no ZZ token but rc 0 is reported as 'PASS-NOTOKEN').
ACCEPTED = the 7 that fail on the unchanged tip. A failure signature = the set of normalised failure-ish lines; it must equal the base's."""
import os, re, sys, glob

RES = sys.argv[1] if len(sys.argv) > 1 else "F:/scorer1_v3/res"
BASE = "F:/scorer1_v2_base"
ACCEPT = {"ego_yaw": ["base.test_ego_yaw.py.log", "base.ego_full.log"], "sft4r_chain": ["base.test_sft4r_chain.py.log"],
          "s1aux_chain": ["base.test_s1aux_chain.py.log"], "p2r_chain": ["base.test_p2r_chain.py.log"],
          "rls1_chain": ["base.test_rls1_chain.py.log"], "p7_ship_list": ["base.test_p7_ship_list.py.log"],
          "carve_feed": ["../scorer1_v3/res/base_carve_feed.log"]}
SIG = re.compile(r"FAIL|Error|ABSENT|Traceback|assert|not found|No such file|DRIFT", re.I)


def norm(l):
    l = l.replace("\\", "/")
    l = re.sub(r"F:/scorer1_v[0-9a-z_]*/(t/|res/)?", "<T>/", l)
    l = re.sub(r"F:/scorer1_v2_(base|refe)/", "<T>/", l)
    l = re.sub(r"\d+(\.\d+)?\s*s\b", "N s", l)
    return l.strip()


def sig(path):
    if not os.path.isfile(path):
        return None
    return sorted({norm(l) for l in open(path, "rb").read().decode("utf-8", "replace").splitlines() if SIG.search(l)})


rows, out = [], []
for l in open(os.path.join(RES, "results.txt"), encoding="utf-8"):
    m = re.match(r"(\S+)/test_(\S+)\.py rc=(\d+) secs=(\d+) ?(.*)", l.rstrip("\n"))
    if m:
        rows.append(m.groups())
npass = 0
for grp, n, rc, secs, last in sorted(rows):
    ok = rc == "0"
    tok = "ZZ" in last and "PASS" in last
    if ok:
        npass += 1
        out.append(f"PASS   {grp}/{n}  {secs}s  {last[:90]}" + ("" if tok else "   (rc 0, NO ZZ..PASS token: check)"))
        continue
    if n in ACCEPT and grp == "refe":
        v3 = sig(os.path.join(RES, "logs", f"{grp}.test_{n}.log"))
        bases = [(b, sig(os.path.join(BASE, b))) for b in ACCEPT[n]]
        bases = [(b, s) for b, s in bases if s is not None]
        same = [b for b, s in bases if s == v3]
        verdict = ("SAME-AS-BASE (accepted)" if same else ("DIFFERS-FROM-BASE" if bases else "NO-BASE-LOG (run it on the base tree)"))
        out.append(f"FAIL   {grp}/{n} rc={rc} {secs}s  -> {verdict}  {last[:90]}")
        if not same:
            out.append(f"       v3 signature: {v3[:6] if v3 else v3}")
            for b, s in bases:
                out.append(f"       base {b}: {s[:6]}")
    else:
        out.append(f"FAIL   {grp}/{n} rc={rc} {secs}s  -> UNEXPECTED (rc 124 = timeout at the runner's limit)  {last[:120]}")
out.insert(0, f"{len(rows)} tests finished, {npass} rc 0")
txt = "\n".join(out)
print(txt)
open(os.path.join(RES, "COMPARE.txt"), "w", encoding="utf-8").write(txt + "\n")
