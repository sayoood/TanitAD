r"""RECOVER-FT code refresh on box 2 (stdlib only). Overlays the shipped tip tar onto D:\tanitad\repo the way every earlier v8F chain did (`tar -xf ... -C repo`), but FIRST moves aside every file it
would change: each existing member whose bytes differ is COPIED to D:\tanitad\repo_aside_<date>\<same relative path> (nothing is deleted; the aside is the exact pre-state of every overwritten file).
Then verifies every extracted member's sha256 against the manifest shipped with the tar. Prints a summary; exit 0 only if every member verifies."""
import hashlib, json, os, shutil, sys, tarfile, time
TAR = r"D:\tanitad\repo_recft_tip.tar"; MAN = r"D:\tanitad\recft\recft_tar_manifest.json"; REPO = r"D:\tanitad\repo"; ASIDE = r"D:\tanitad\repo_aside_20261102"
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
    return h.hexdigest()
man = json.load(open(MAN, encoding="utf-8"))
tsha = sha(TAR)
assert tsha == man["tar_sha256"], f"tar sha256 {tsha} != manifest {man['tar_sha256']}"
tf = tarfile.open(TAR)
moved = new = same = 0
for m in tf.getmembers():
    if not m.isfile(): continue
    dst = os.path.join(REPO, *m.name.split("/"))
    if os.path.exists(dst):
        if sha(dst) == man["members"][m.name]: same += 1; continue
        a = os.path.join(ASIDE, *m.name.split("/")); os.makedirs(os.path.dirname(a), exist_ok=True)
        if os.path.exists(a): print("ASIDE EXISTS, not overwritten:", a); continue
        shutil.copy2(dst, a); moved += 1
    else: new += 1
tf.extractall(REPO, filter="data") if sys.version_info >= (3, 12) else tf.extractall(REPO)
bad = [n for n, h in man["members"].items() if not os.path.exists(os.path.join(REPO, *n.split("/"))) or sha(os.path.join(REPO, *n.split("/"))) != h]
open(os.path.join(REPO, "TIP_RECFT.txt"), "w").write(man["tip"] + "\n")
print(f"refresh {time.strftime('%Y-%m-%d %H:%M:%S')}: tip {man['tip']}; members {len(man['members'])}: {same} already identical, {moved} changed (old bytes copied to {ASIDE}), {new} new; verify failures {len(bad)}")
for n in bad: print("VERIFY-FAIL", n)
sys.exit(1 if bad else 0)
