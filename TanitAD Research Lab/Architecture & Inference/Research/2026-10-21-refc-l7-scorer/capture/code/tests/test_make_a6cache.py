"""Test of launch/make_a6cache_thor.py on a FAKE 139-clip pool (tiny files), verify-only (--link none): the positive case reads OK; a size mutant and a content mutant
(right size, wrong bytes, caught only by the md5 probes) each read BAD."""
import hashlib, json, os, subprocess, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "..", "..", "launch", "make_a6cache_thor.py")
sha12 = lambda s: hashlib.sha256(s.encode()).hexdigest()[:12]


def run(args):
    r = subprocess.run([sys.executable, SCRIPT] + args, capture_output=True, text=True)
    return r.stdout + r.stderr


def main():
    with tempfile.TemporaryDirectory() as t:
        ids = [f"fake-clip-{i:04d}" for i in range(139)]
        src = os.path.join(t, "src"); os.makedirs(src)
        sizes, content = {}, {}
        for i, c in enumerate(ids):
            b = (c * (3 + i % 5)).encode()
            open(os.path.join(src, c + ".v2ep.pt"), "wb").write(b)
            sizes[sha12(c)] = len(b); content[sha12(c)] = b
        ks = sorted(sizes)
        probes = {k: hashlib.md5(content[k]).hexdigest() for k in (ks[0], ks[len(ks) // 2], ks[-1])}
        open(os.path.join(t, "pool.txt"), "w").write("\n".join(ids) + "\n")
        json.dump({"clips": {"clip-sha12-" + k: {"ws": [7]} for k in ks}}, open(os.path.join(t, "w.json"), "w"))
        json.dump({"sizes_by_sha12": sizes, "md5_probes_by_sha12": probes}, open(os.path.join(t, "s.json"), "w"))
        base = ["--windows", os.path.join(t, "w.json"), "--sizes", os.path.join(t, "s.json"), "--pool", os.path.join(t, "pool.txt"), "--src", src, "--link", "none"]
        out_dir = os.path.join(t, "o1")
        pos = run(base + ["--out", out_dir])
        assert "ZZL7CA6OKZZ" in pos and "size_equal=139" in pos and "md5_probe_bad=0" in pos, "FAIL positive: " + pos
        print("PASS positive (verify-only; the symlink step itself is UNVERIFIED on this box): ", pos.strip())
        # size mutant
        f = os.path.join(src, ids[5] + ".v2ep.pt"); old = open(f, "rb").read(); open(f, "wb").write(old + b"x")
        r = run(base + ["--out", os.path.join(t, "o2")])
        assert "ZZL7CA6BADZZ" in r and "size_equal=138" in r, "size mutant not detected: " + r
        print("PASS size mutant ->", r.strip()[:120])
        # content mutant: right size, wrong bytes in a PROBED file
        k0 = ks[0]; cid = [c for c in ids if sha12(c) == k0][0]
        f = os.path.join(src, cid + ".v2ep.pt"); b = bytearray(content[k0]); b[0] ^= 0xFF
        open(f, "wb").write(bytes(b)); open(os.path.join(src, ids[5] + ".v2ep.pt"), "wb").write(old)
        r = run(base + ["--out", os.path.join(t, "o3")])
        assert "ZZL7CA6BADZZ" in r and "md5_probe_bad=1" in r and "size_equal=139" in r, "content mutant not detected: " + r
        print("PASS content mutant ->", r.strip()[:140])
    print("ZZL7CA6TESTZZ")


if __name__ == "__main__":
    main()
