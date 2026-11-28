"""chainlib.py -- helpers of block1.sh (pod). One subcommand per call; every one exits 0 on success and non-zero with a one-line reason on stderr.

  subst TEMPLATE OUT INIT BANKDIR          fill @INIT@ / @BANK@ / @BANKSHA@ of a stage-B argv template (bank sha256 = sha256 of BANKDIR/bank.json)
  m2bargv ARGV_A OUT                       the stage-A argv without --fit-join (the bank generator rebuilds the TRAIN list from it and must not rewrite the join)
  bankargs DRY_JSON                        prints "<n_clips> <n_windows>" of the dry record
  summary ARM_DIR EXPECT_STEPS             the run finished: summary.json says done, ckpt_last.pt exists and holds step == EXPECT_STEPS
  post CKPT [INIT_CKPT]                    the post-training gate: every registered parameter has Adam state (count equals B_s0's registered count 252 tensors) and, with INIT, the tensors that moved
  banked ARM_TAG PATH                      exit 0 iff bank_ok/<ARM_TAG>.sha256 exists and equals sha256(PATH)
  sweep ARM                                the card's deletions for one arm, only for what is banked: bank shards (bank.json, PROVENANCE and the shard provenance stay), the stage-B selfcheck scratch
  sweep_r                                  the R release and its join (re-creatable by dscale10k_merge.py + seed 20261117), only after R_s0's stage-A checkpoint is banked
Deletions are by literal absolute path and each is logged (path, size) in /root/dscale10k/deletions_block1.log. Never touches a checkpoint, a bank.json, a token, a register or anything of another session."""
import hashlib
import json
import os
import sys
import time

D = os.environ.get("DS_ROOT", "/root/dscale10k")
LOG = D + "/deletions_block1.log"
REGISTERED_B_S0 = 252                    # trainable tensors of the built model (pilot selfcheck n_trainable_tensors, B_s0 and U_s0)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def die(msg):
    print(msg, file=sys.stderr)
    sys.exit(1)


def delete(path):
    path = os.path.abspath(path)
    assert path.startswith(D + "/") and ".." not in path, path
    if os.path.isfile(path):
        n = os.path.getsize(path)
        os.remove(path)
        open(LOG, "a").write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\t{path}\t{n}\n")
        return n
    return 0


def main(a):
    c = a[0]
    if c == "subst":
        t, out, init, bank = a[1:5]
        b = os.path.join(bank, "bank.json")
        if not os.path.isfile(b) or not os.path.isfile(init):
            die(f"subst: bank.json {b} or init {init} is missing")
        s = sha(b)
        v = json.load(open(t))["argv"]
        v = [init if x == "@INIT@" else bank if x == "@BANK@" else s if x == "@BANKSHA@" else x for x in v]
        if any(x.startswith("@") and x.endswith("@") for x in v):
            die("subst: an unfilled placeholder remains")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        json.dump({"argv": v}, open(out, "w"), indent=1)
    elif c == "m2bargv":
        v = [x for x in json.load(open(a[1]))["argv"] if x != "--fit-join"]
        json.dump({"argv": v}, open(a[2], "w"), indent=1)
    elif c == "bankargs":
        r = json.load(open(a[1]))
        print(r["n_clips"], r["n_windows"])
    elif c == "summary":
        d, want = a[1], int(a[2])
        sj, ck = os.path.join(d, "summary.json"), os.path.join(d, "ckpt_last.pt")
        if not (os.path.isfile(sj) and os.path.isfile(ck)):
            die(f"summary: {d} has no summary.json / ckpt_last.pt")
        s = json.load(open(sj))
        import torch
        step = torch.load(ck, map_location="cpu", weights_only=False).get("step")
        if step != want:
            die(f"summary: ckpt_last.pt is at step {step}, expected {want}")
        if not s.get("done") or s.get("steps") != want:
            die(f"summary: summary.json reads done={s.get('done')} steps={s.get('steps')}, expected done at {want}")
        print(json.dumps({"ckpt_step": step, "ckpt_sha256": sha(ck)}))
    elif c == "post":
        import torch
        blob = torch.load(a[1], map_location="cpu", weights_only=False)
        opt = blob["opt"]
        reg = sum(len(g["params"]) for g in opt["param_groups"])
        with_state = sum(1 for k, v in opt["state"].items() if v)
        rec = {"ckpt": a[1], "registered": reg, "with_adam_state": with_state}
        bad = []
        if with_state != reg:
            bad.append(f"{reg - with_state} registered parameters have no Adam state (never received a gradient)")
        if reg != REGISTERED_B_S0:
            bad.append(f"registered {reg} != B_s0's {REGISTERED_B_S0}")
        if len(a) > 2:
            init = torch.load(a[2], map_location="cpu", weights_only=False)["model"]
            ch = sum(1 for k, v in blob["model"].items() if v.dtype.is_floating_point and k in init and not bool((v == init[k]).all()))
            rec["tensors_moved_vs_init"] = ch
            if ch < reg:
                bad.append(f"only {ch} floating tensors moved against the init checkpoint, {reg} are registered")
        rec["problems"] = bad
        print(json.dumps(rec))
        if bad:
            die("post: " + "; ".join(bad))
    elif c == "banked":
        m = os.path.join(D, "bank_ok", a[1] + ".sha256")
        if not (os.path.isfile(m) and os.path.isfile(a[2]) and open(m).read().split()[0] == sha(a[2])):
            sys.exit(1)
    elif c == "sweep":
        arm = a[1]
        tot = 0
        ck = f"{D}/runs/{arm}/B/ckpt_last.pt"
        if os.system(f"{sys.executable} {os.path.abspath(__file__)} banked {arm}.B {ck}") != 0:
            print(f"DELETE-DEFERRED {arm}: its stage-B read checkpoint is not banked (bank_ok/{arm}.B.sha256 absent or different)")
            return
        bd = f"{D}/runs/{arm}/bank"
        if os.path.isdir(bd):
            for f in sorted(os.listdir(bd)):
                if f.startswith("shard_") and f.endswith(".npz"):
                    tot += delete(os.path.join(bd, f))
        sc = f"{D}/runs/{arm}/B_selfcheck"
        if os.path.isdir(sc):
            for dp, dn, fn in os.walk(sc):
                for f in fn:
                    tot += delete(os.path.join(dp, f))
        print(f"SWEPT {arm}: {tot} bytes")
    elif c == "sweep_r":
        ck = f"{D}/runs/R_s0/A/ckpt_last.pt"
        if os.system(f"{sys.executable} {os.path.abspath(__file__)} banked R_s0.A {ck}") != 0:
            print("DELETE-DEFERRED R_s0: its stage-A checkpoint is not banked")
            return
        tot = delete(D + "/merged/v9_labels_R_s0.npz") + delete(D + "/merged/join_R_s0.npz")
        print(f"SWEPT R release: {tot} bytes")
    else:
        die("unknown subcommand " + c)


if __name__ == "__main__":
    main(sys.argv[1:])
