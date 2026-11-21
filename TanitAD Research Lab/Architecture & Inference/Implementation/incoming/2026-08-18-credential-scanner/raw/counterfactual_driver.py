"""Same 9 planted cases, now against tools/secret_scan.py (both scanners side by side)."""
import subprocess, sys, tempfile
from pathlib import Path
REPO = Path(r"G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD")
sys.path.insert(0, str(REPO / "tools"))
import safe_commit, secret_scan

FAKE_HF  = "hf_" + ("Zq7" * 12)[:34]
FAKE_SK  = "sk-" + ("Ab9" * 14)[:40]
FAKE_GHP = "ghp_" + ("Kk3" * 12)[:36]
FAKE_AWS = "AKIA" + "QRSTUVWX23456789"
FAKE_PEM = "-----BEGIN RSA PRIVATE" + " KEY-----"
FAKE_ASSIGN = 'api_key = "' + "v4Xq8Lm2Pd7Rt5Yn3Bw9Zc6Hj1Ks4Gf" + '"'

def sh(cwd,*a,**k): return subprocess.run(["git",*a],cwd=str(cwd),capture_output=True,text=True,errors="replace",**k)
def build(d):
    d.mkdir(parents=True); sh(d,"init","-b","work"); sh(d,"config","user.email","t@t"); sh(d,"config","user.name","t")
    (d/"seed.txt").write_text("seed\n"); sh(d,"add","seed.txt"); sh(d,"commit","-m","seed"); return d
def plant(r, rel, payload, binary):
    p = r/rel; p.parent.mkdir(parents=True, exist_ok=True)
    if binary: p.write_bytes(b"\x00\x01\x02hdr\n"+payload.encode()+b"\n\x00tail")
    else:
        ls=[f"[boot] line {i}" for i in range(1,11)]+[payload]+[f"[post] {i}" for i in range(12,20)]
        p.write_text("\n".join(ls)+"\n", encoding="utf-8")

CASES = [
 ("rescued/rq_out/logs/contention.log", f"hf download --token {FAKE_HF} Sayood/x", False, "C111 EXACT SHAPE .log"),
 ("rescued/meta/run_state.json",        f'  "hf_token": "{FAKE_HF}",',            False, ".json"),
 ("rescued/notes/handoff.txt",          f"token={FAKE_HF}",                       False, ".txt"),
 ("rescued/logs/openai.log",            f"OPENAI_API_KEY={FAKE_SK}",              False, "openai sk-"),
 ("rescued/logs/gh.log",                f"remote https://{FAKE_GHP}@github.com",  False, "github ghp_"),
 ("rescued/logs/aws.log",               f"aws_access_key_id = {FAKE_AWS}",        False, "aws AKIA"),
 ("rescued/keys/host.pem",              FAKE_PEM,                                 False, "PEM header"),
 ("rescued/conf/settings.yaml",         FAKE_ASSIGN,                              False, "generic api_key assign"),
 ("rescued/blob/state.bin",             f"tok {FAKE_HF}",                         True,  "BINARY blob .bin"),
]
with tempfile.TemporaryDirectory() as td:
    tmp=Path(td)
    print(f"{'case':<26} {'safe_commit(old)':<17} {'secret_scan --staged':<21} {'secret_scan --tree':<19} where")
    print("-"*115)
    o=n=t=0
    for rel,payload,binary,label in CASES:
        r=build(tmp/label.replace(" ","_").replace(".",""))
        plant(r,rel,payload,binary); sh(r,"add","-A")
        old_b,_ = safe_commit.scan_secrets(r, safe_commit.staged_paths(r))
        new = secret_scan.scan_staged(r)
        tre = secret_scan.scan_tree(r)
        ov = "CAUGHT" if old_b else "** MISS **"
        nv = "CAUGHT" if new.blocking else "** MISS **"
        tv = "CAUGHT" if tre.blocking else "** MISS **"
        o += bool(old_b); n += bool(new.blocking); t += bool(tre.blocking)
        loc = ""
        if new.blocking:
            f=new.blocking[0]; loc=f"{f.pattern} @ {f.path}:{f.line}"
        print(f"{label:<26} {ov:<17} {nv:<21} {tv:<19} {loc}")
    print("-"*115)
    print(f"caught: safe_commit(old)={o}/9   secret_scan --staged={n}/9   secret_scan --tree={t}/9")
