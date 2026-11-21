"""q5c0_tool.py -- helpers of the Q5 C0 chain (q5c0_chain.sh).  No network, no box access.  Python run with -I.  Amendment 7 section 4 C0 + the S2 E block.

  mkjob      --mode dry|shard --sample S1|S2 --gid G [--i I --k K] --out FILE   box-2 job .cmd (CRLF; every backslash comes from chr(92))
  mklaunch   --chain SH --console LOG --ssh P --sftp P --out FILE               the WMI-detached start .cmd (bash < NUL); not started here
  check-dry  --log FILE --sample S1|S2                                          dry-run log must show banked pool, overlap 0, counts, the sample sha (S1 843e4017.., S2 32ad96be..); prints it
  verify-shard --dir D --i I --k K --sample S --sample-sha SHA                  pulled shard: s0p_chain_tool.verify_shard (read from the S0-P chain folder, unedited) + C0 meta controls
  assemble   --pull DIR --sample S --out DIR --k K --sample-json FILE           concatenate the K verified shards into an S0-P-format dir (E v40 v160 mean y clip ws eid, meta, sample, MANIFEST)
  c0check    --state DIR --base DIR --out FILE                                   C0_RESULT.json: PASS/FAIL per check
"""
import argparse
import hashlib
import importlib.util
import json
import os
import re
import sys

BS = chr(92)
S0P_CHAIN_TOOL = os.environ.get("Q5C0_S0P_TOOL", "E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/stopperc/s0p/chain/s0p_chain_tool.py")
SEEDS = {"S1": 20261111, "S2": 20261115}
SHA = {"S1": "843e4017de3c40ae91618ac878cc2f361c798a62633eb9fa6805b457b4c1609e", "S2": "32ad96be507893b91c673c87e3088a3d294ff0e047b61e37ff8b39951740136a"}
NT = int(os.environ.get("Q5C0_N_TOTAL", 16000))        # test hooks: the chain refuses them unless Q5C0_DRY=1
NP = int(os.environ.get("Q5C0_N_POS", 4000))
EVAL_SHA = "0c25fe722f3359ee49ec6c31affa6761a4cd6f3ecb1e63d4ccc475ae2935fb39"
S0P_STATE = os.environ.get("Q5C0_S0P_STATE", "C:/Users/Admin/s0p_state")
REFDIR = os.environ.get("Q5C0_REF", "C:/Users/Admin/q5c0/ref")                 # sh<i>.log of the S0-P shard jobs (box 2's own stdout, read-only copy)
FIELDS = ("E", "v40", "v160", "mean", "y", "clip", "ws", "eid")
ROOT = "D:" + BS + "tanitad"
B2 = {"py": ROOT + BS + "venv" + BS + "Scripts" + BS + "python.exe", "s0p": ROOT + BS + "stopperc" + BS + "q5c0" + BS + "s0p", "out": ROOT + BS + "stopperc" + BS + "q5c0" + BS + "out",
      "jobs": ROOT + BS + "stopperc" + BS + "q5c0" + BS + "jobs", "cache": ROOT + BS + "cache_full", "repo": ROOT + BS + "repo",
      "join": ROOT + BS + "runs" + BS + "v8f_mini_full" + BS + "join_train.npz"}
B2["tracks"] = B2["cache"] + BS + "native_tracks.pt"


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def s0p_tool():
    spec = importlib.util.spec_from_file_location("s0p_chain_tool", S0P_CHAIN_TOOL)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def mkjob(mode, sample, gid, i, k):
    assert mode in ("dry", "shard") and sample in SEEDS
    tag = f"{sample}DRY" if mode == "dry" else f"{sample}SH{i}"
    od = B2["out"] + BS + sample
    args = (f'--tree {B2["repo"]} --cache-dir {B2["cache"]} --split train --join {B2["join"]} --tracks {B2["tracks"]} --out-dir "{od}" --eval-cache-dir {B2["cache"]} --seed {SEEDS[sample]}')
    args += " --dry-run" if mode == "dry" else f" --shard {i}/{k}"
    md = B2["jobs"]
    ok, fail = f'"{md}{BS}{tag}_{gid}.done"', f'"{md}{BS}{tag}_{gid}.fail"'
    L = ["@echo off", f"rem Q5C0JOB mode={mode} sample={sample} i={i if mode == 'shard' else 0} k={k if mode == 'shard' else 0} gid={gid}",
         "rem Q5 C0: S0-P train-side extraction on box 2, CPU only; writes only under D:" + BS + "tanitad" + BS + "stopperc" + BS + "q5c0. No goto.",
         "setlocal enabledelayedexpansion", 'set "OMP_NUM_THREADS=2"', 'set "CUDA_VISIBLE_DEVICES=-1"', f'if not exist "{md}" mkdir "{md}"',
         f"echo ===== Q5C0-JOB {tag} {gid} %DATE% %TIME%", f'"{B2["py"]}" "{B2["s0p"]}{BS}s0p_extract_train.py" {args}', 'set "RC=!ERRORLEVEL!"',
         "echo ===== Q5C0-JOB-END rc=!RC! %DATE% %TIME%", f"if !RC! NEQ 0 (echo rc=!RC!> {fail} & exit /b !RC!)"]
    if mode == "shard":
        L.append(f'if not exist "{od}{BS}shard_{i}{BS}MANIFEST.sha256" (echo no-manifest> {fail} & exit /b 1)')
    L += [f"echo done> {ok}", "exit /b 0"]
    return "\r\n".join(L) + "\r\n"


def mklaunch(chain, console, ssh, sftp):
    bash = "C:" + BS + "Program Files" + BS + "Git" + BS + "usr" + BS + "bin" + BS + "bash.exe"
    w = lambda p: p.replace("/", BS)
    return ("@echo off\r\n" + f'set "S0P_SSH={w(ssh)}"\r\nset "S0P_SFTP={w(sftp)}"\r\n' + f'"{bash}" "{w(chain)}" < NUL > "{w(console)}" 2>&1\r\n')


def check_dry(text, sample):
    text = text.replace("\x00", "")
    need = [(r'"pool_equals_banked_prevalence":\s*true', "pool_equals_banked_prevalence"), (r'"overlap":\s*0\b', "disjoint overlap 0"), (rf'"n_pos_sample":\s*{NP}\b', f"n_pos_sample {NP}"),
            (rf'"n_neg_sample":\s*{NT - NP}\b', f"n_neg_sample {NT - NP}"), (rf'"n_sample":\s*{NT}\b', f"n_sample {NT}"), (rf'"seed":\s*{SEEDS[sample]}\b', f"seed {SEEDS[sample]}")]
    miss = [n for rx, n in need if not re.search(rx, text)]
    m = re.search(r'"sample_sha256":\s*"([0-9a-f]{64})"', text)
    if not m:
        miss.append("sample_sha256")
    elif m.group(1) != SHA[sample]:
        miss.append(f"sample_sha256 {m.group(1)} != {SHA[sample]}")
    if miss:
        raise SystemExit("dry-run log check failed, missing: " + ", ".join(miss))
    return m.group(1)


def verify_shard(d, i, k, sample, sample_sha):
    r = s0p_tool().verify_shard(d, i, k, sample_sha, NT)
    meta = json.load(open(os.path.join(d, "meta.json"), encoding="utf-8"))
    assert meta["seed"] == SEEDS[sample], f"meta seed {meta['seed']} != {SEEDS[sample]}"
    assert meta["sample_sha256"] == SHA[sample], "meta sample sha differs from the frozen one"
    c2 = meta["c2_second_route_max_abs"]
    assert all(float(v) == 0.0 for v in c2.values()), f"second route not 0.0: {c2}"
    assert meta["causal_limit_ok_false_fraction"] == 0.0 and meta["pool_equals_banked_prevalence"] is True and meta["disjoint"]["overlap"] == 0, "meta controls"
    r["seed"] = meta["seed"]
    return r


def assemble(pull, sample, out, k, sample_json):
    import numpy as np
    os.makedirs(out, exist_ok=True)
    dirs = [os.path.join(pull, f"shard_{i}") for i in range(k)]
    metas = [json.load(open(os.path.join(d, "meta.json"), encoding="utf-8")) for d in dirs]
    n = sum(m["n_shard"] for m in metas)
    for f in FIELDS:
        arrs = [np.load(os.path.join(d, f + ".npy"), mmap_mode="r") for d in dirs]
        shape = (n,) + arrs[0].shape[1:]
        dst = np.lib.format.open_memmap(os.path.join(out, f + ".npy"), mode="w+", dtype=arrs[0].dtype, shape=shape)
        o = 0
        for a in arrs:
            assert a.shape[1:] == shape[1:] and a.dtype == arrs[0].dtype
            dst[o:o + len(a)] = a
            o += len(a)
        dst.flush()
        del dst
    sj = json.load(open(sample_json, encoding="utf-8"))
    assert sj["sha256"] == SHA[sample] and len(sj["y"]) == n
    y = np.load(os.path.join(out, "y.npy"))
    assert y.tolist() == [int(x) for x in sj["y"]] and np.load(os.path.join(out, "clip.npy")).tolist() == sj["clip"] and np.load(os.path.join(out, "ws.npy")).tolist() == sj["ws"], "assembled arrays differ from the full sample.json"
    meta = dict(metas[0])
    meta.pop("shard", None), meta.pop("n_shard", None), meta.pop("n_pos_shard", None), meta.pop("elapsed_s", None)
    meta["c2_second_route_max_abs"] = {kk: max(float(m["c2_second_route_max_abs"][kk]) for m in metas) for kk in metas[0]["c2_second_route_max_abs"]}
    meta["assembled_from_shards"] = {"k": k, "manifest_sha256": [sha256_file(os.path.join(d, "MANIFEST.sha256")) for d in dirs], "elapsed_s": [m["elapsed_s"] for m in metas]}
    meta["q5c0"] = {"sample": sample, "n": n}
    with open(os.path.join(out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1, sort_keys=True)
    with open(os.path.join(out, "sample.json"), "w", encoding="utf-8") as f:
        json.dump(sj, f)
    names = [x + ".npy" for x in FIELDS] + ["meta.json", "sample.json"]
    with open(os.path.join(out, "MANIFEST.sha256"), "w", encoding="utf-8") as f:
        for nm in names:
            f.write(f"{sha256_file(os.path.join(out, nm))}  {nm}\n")
    return {"n": n, "dir": out}


def ref_meta(i):
    """S0-P's shard meta.json, rebuilt from box 2's own stdout of that job (print(json.dumps(ctl)) = the dict later json.dump'ed with sort_keys, indent 1)."""
    txt = open(os.path.join(REFDIR, f"sh{i}.log"), encoding="utf-8", errors="replace").read()
    j0 = txt.index("\n{") + 1
    j1 = txt.rindex("\n}") + 2
    return json.loads(txt[j0:j1])


def manifest_with_meta(pdir, meta_dict):
    """The pulled MANIFEST.sha256 with ONLY its meta.json line swapped for the hash of the banked meta (serialised as box 2 wrote it).
    MM fix 2026-11-16: box 2 writes meta.json and MANIFEST.sha256 in Windows text mode (CRLF). The first version rebuilt both with LF, so the
    rebuilt hash could never equal the banked one (C0 FAIL on 4/4 shards while every data file was byte-equal: MEASURED, MM check, the manifest
    with only the elapsed_s text changed hashes to the banked manifest_sha256 on all 4 shards). Every data line must still match the file on disk."""
    raw_meta = open(os.path.join(pdir, "meta.json"), "rb").read()
    nl = b"\r\n" if b"\r\n" in raw_meta else b"\n"
    mb = json.dumps(meta_dict, indent=1, sort_keys=True).encode("utf-8").replace(b"\n", nl)
    if raw_meta.endswith(nl) and not mb.endswith(nl):
        mb += nl
    man = open(os.path.join(pdir, "MANIFEST.sha256"), "rb").read()
    h_meta = hashlib.sha256(raw_meta).hexdigest().encode()
    if man.count(h_meta) != 1:
        return "MANIFEST-META-LINE-MISSING"
    for line in man.decode("utf-8").splitlines():
        h, nm = line.split("  ", 1)
        if nm != "meta.json" and sha256_file(os.path.join(pdir, nm)) != h:
            return f"DATA-FILE-MISMATCH {nm}"
    return hashlib.sha256(man.replace(h_meta, hashlib.sha256(mb).hexdigest().encode())).hexdigest()


def c0check(state, base, out):
    res = {"amendment": "AMENDMENT_7_Q5_416 section 4 C0", "checks": {}}
    ck = res["checks"]
    pull = os.path.join(base, "_pull")
    # eval npz
    ep = os.environ.get("Q5C0_EVAL_NPZ") or os.path.join(base, "s0p_eval.npz")
    got = sha256_file(ep) if os.path.exists(ep) else None
    ck["eval_npz_sha256"] = {"pass": got == EVAL_SHA, "got": got, "want": EVAL_SHA}
    # S1 sample sha + shard manifests
    s1 = json.load(open(os.path.join(base, "S1", "sample.json"), encoding="utf-8"))
    ck["S1_sample_sha256"] = {"pass": s1["sha256"] == SHA["S1"], "got": s1["sha256"], "want": SHA["S1"]}
    lit_all = True

    def _one(i):
        want = json.load(open(os.path.join(S0P_STATE, f"shard_{i}.verify.json"), encoding="utf-8"))["manifest_sha256"]
        pdir = os.path.join(pull, "S1", f"shard_{i}")
        if not os.path.isdir(pdir):
            ck[f"S1_shard{i}_manifest"] = {"pass": False, "why": f"{pdir} missing (pulled shard removed?)"}
            return False
        lit = sha256_file(os.path.join(pdir, "MANIFEST.sha256"))
        mine = json.load(open(os.path.join(pdir, "meta.json"), encoding="utf-8"))
        ref = ref_meta(i)
        mine_cmp, ref_cmp = dict(mine), dict(ref)
        el_new = mine_cmp.pop("elapsed_s", None)
        el_old = ref_cmp.pop("elapsed_s", None)
        same_meta = mine_cmp == ref_cmp
        rebuilt = manifest_with_meta(pdir, ref) if same_meta else None
        ck[f"S1_shard{i}_manifest"] = {"pass": bool(same_meta and rebuilt == want), "literal_equal": lit == want, "banked": want, "literal_got": lit, "rebuilt_with_banked_meta": rebuilt,
                                       "meta_equal_except_elapsed_s": same_meta, "elapsed_s_new_vs_banked": [el_new, el_old],
                                       "note": "meta.json holds wall time elapsed_s, so the literal MANIFEST hash cannot repeat; the pass rule rebuilds the manifest with the banked job's own meta.json (from its stdout) and every array/sample file hash of THIS run"}
        return lit == want

    for i in range(4):
        try:
            lit_all = _one(i) and lit_all
        except Exception as e:  # fail closed: the check is recorded as failed, the result file is still written
            ck[f"S1_shard{i}_manifest"] = {"pass": False, "why": repr(e)}
            lit_all = False
    res["literal_manifest_equal_all_shards"] = lit_all
    # S2
    s2 = json.load(open(os.path.join(base, "S2", "sample.json"), encoding="utf-8"))
    ck["S2_sample_sha256"] = {"pass": s2["sha256"] == SHA["S2"], "got": s2["sha256"], "want": SHA["S2"]}
    for s in ("S1", "S2"):
        m = json.load(open(os.path.join(base, s, "meta.json"), encoding="utf-8"))
        ck[f"{s}_controls"] = {"pass": m["c2_second_route_max_abs"] == {"p10f": 0.0, "nav": 0.0, "limit": 0.0, "limit_ok": 0.0} and m["n_sample"] == NT and m["n_pos_sample"] == NP and m["disjoint"]["overlap"] == 0 and m["causal_limit_ok_false_fraction"] == 0.0,
                               "c2": m["c2_second_route_max_abs"], "n_sample": m["n_sample"], "n_pos": m["n_pos_sample"], "overlap": m["disjoint"]["overlap"]}
    res["overall"] = "PASS" if all(c["pass"] for c in ck.values()) else "FAIL"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, sort_keys=True)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("psenc", "mkjob", "mklaunch", "check-dry", "verify-shard", "assemble", "c0check"))
    for n in ("--mode", "--sample", "--gid", "--out", "--chain", "--console", "--log", "--dir", "--sample-sha", "--ssh", "--sftp", "--pull", "--sample-json", "--state", "--base"):
        ap.add_argument(n)
    ap.add_argument("--i", type=int, default=0), ap.add_argument("--k", type=int, default=4)
    a = ap.parse_args(argv)
    if a.cmd == "psenc":
        import base64
        print(base64.b64encode(a.log.encode("utf-16-le")).decode())
    elif a.cmd == "mkjob":
        assert re.fullmatch(r"[0-9A-Za-z_]+", a.gid or "")
        with open(a.out, "w", encoding="utf-8", newline="") as f:
            f.write(mkjob(a.mode, a.sample, a.gid, a.i, a.k))
    elif a.cmd == "mklaunch":
        with open(a.out, "w", encoding="utf-8", newline="") as f:
            f.write(mklaunch(a.chain, a.console, a.ssh, a.sftp))
    elif a.cmd == "check-dry":
        print(check_dry(open(a.log, encoding="utf-8", errors="replace").read(), a.sample))
    elif a.cmd == "verify-shard":
        print(json.dumps(verify_shard(a.dir, a.i, a.k, a.sample, a.sample_sha)))
    elif a.cmd == "assemble":
        print(json.dumps(assemble(a.pull, a.sample, a.out, a.k, a.sample_json)))
    else:
        r = c0check(a.state, a.base, a.out)
        print(r["overall"])
        sys.exit(0 if r["overall"] == "PASS" else 3)


if __name__ == "__main__":
    main()
