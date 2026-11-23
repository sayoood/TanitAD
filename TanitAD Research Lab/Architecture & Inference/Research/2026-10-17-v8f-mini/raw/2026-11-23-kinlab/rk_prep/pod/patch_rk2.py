"""patch_rk2.py -- REVISION 2: build dscale10k_check_rk.py (rev2) from dscale10k_check_rk.py.rev1 (sha256 273befbc..., the RK checker of revision 1) by anchored edits. Each anchor must match exactly the stated count, else refuse.
The ONLY change (Amendment 1 section 2, frozen sha256 948670d9): the RK kind admits the execution-only token pair `--dagger-label-cache <path>` -- exactly once, and the path must be the cache of THIS seed's bank
(`.../kin_cache/s<K>_kin_v1.npz`; the other seed's, another kind's or another folder's path is refused). The pair is dropped, with the label pair, from the RK argv before the stock pair check against B0.
RH, B0, RHp3, RHK, RL, RA and the RK pins of revision 1 (dose, bank, label) behave as before. Usage: patch_rk2.py SRC DST"""
import hashlib
import sys

SRC, DST = sys.argv[1], sys.argv[2]
s = open(SRC, encoding="utf-8").read()
assert hashlib.sha256(s.encode("utf-8")).hexdigest().startswith("273befbc8cba48ae"), "source is not the revision-1 RK checker"


def sub(old, new, n=1):
    global s
    assert s.count(old) == n, (old[:70], s.count(old), n)
    s = s.replace(old, new)


sub('RK_P = "0.1"\nRK_LABEL = ["kin_v1"]', 'RK_P = "0.1"\nRK_LABEL = ["kin_v1"]\nRK_CACHE_FLAG = "--dagger-label-cache"          # Amendment 1: execution-only; the cache of this seed\'s bank under kin_cache/')
sub('exp_added, exp_removed, own = set(RH_DECL) | ({"--dagger-label"} if kind in ("RL", "RK") else set()) |',
    'exp_added, exp_removed, own = set(RH_DECL) | ({"--dagger-label"} if kind in ("RL", "RK") else set()) | ({RK_CACHE_FLAG} if kind == "RK" else set()) |')
sub('''            if a.get("--dagger-label") != RK_LABEL:
                bad.append(f"RK --dagger-label reads {a.get('--dagger-label')}, the arm's only label is {RK_LABEL}")''',
    '''            if a.get("--dagger-label") != RK_LABEL:
                bad.append(f"RK --dagger-label reads {a.get('--dagger-label')}, the arm's only label is {RK_LABEL}")
            kc = [_norm_path(x) for x in (a.get(RK_CACHE_FLAG) or [])]
            if len(kc) != 1 or not kc[0].endswith(f"/kin_cache/s{seed}_kin_v1.npz") or "/kin_cache/s" not in kc[0]:
                bad.append(f"RK {RK_CACHE_FLAG} reads {a.get(RK_CACHE_FLAG)}, the only cache for this arm is one path .../kin_cache/s{seed}_kin_v1.npz (this seed's bank, label kin_v1)")''')
sub('''                li = [str(x) for x in rh_t].index("--dagger-label") if "--dagger-label" in [str(x) for x in rh_t] else -1
                if li >= 0:
                    rh_t = [str(x) for x in rh_t][:li] + [str(x) for x in rh_t][li + 2:]''',
    '''                li = [str(x) for x in rh_t].index("--dagger-label") if "--dagger-label" in [str(x) for x in rh_t] else -1
                if li >= 0:
                    rh_t = [str(x) for x in rh_t][:li] + [str(x) for x in rh_t][li + 2:]
                ci = [str(x) for x in rh_t].index(RK_CACHE_FLAG) if RK_CACHE_FLAG in [str(x) for x in rh_t] else -1
                if ci >= 0 and kind == "RK":                                                         # the execution-only cache pair is the arm's own, as the label pair
                    rh_t = [str(x) for x in rh_t][:ci] + [str(x) for x in rh_t][ci + 2:]''')
sub('"""[ROLLHIST/KINLAB RK build: dscale10k_check_rk.py = dscale10k_check_ra.py ba5dab8e + the RK kind (patch_rk.py); PREREG_V8F_KINLAB]',
    '"""[REVISION 2: dscale10k_check_rk.py = the revision-1 RK checker 273befbc + the `--dagger-label-cache` token pair of Amendment 1 (patch_rk2.py)]\n[ROLLHIST/KINLAB RK build: dscale10k_check_rk.py = dscale10k_check_ra.py ba5dab8e + the RK kind (patch_rk.py); PREREG_V8F_KINLAB]')
open(DST, "w", encoding="utf-8", newline="").write(s)
print("OK", hashlib.sha256(s.encode("utf-8")).hexdigest())
