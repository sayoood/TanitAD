"""test_kinclose.py -- test of kinclose.py from outside: a synthetic pod folder (kinclose.write_world) is read by the command line, inputs are mutated, and every mutation has to turn
a control red (read exits 1, names empty, arm values under `withheld`) or be refused (exit 2). Plain script: one line per case, ends with ZZ TEST-KINCLOSE <passed>/<total>."""
import io, json, os, shutil, subprocess, sys, tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = os.path.join(HERE, "kinclose.py")
sys.path.insert(0, HERE)
import kinclose as K  # noqa: E402

K.load_tools(None)
RES = []


def case(name, ok):
    RES.append(bool(ok))
    print("%s %s" % ("ok  " if ok else "FAIL", name))


def run(args):
    p = subprocess.run([sys.executable, TOOL] + args, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return p.returncode, (p.stdout or ""), (p.stderr or "")


def read_args(P, out):
    return ["read", "--dump-dir", P["dump_dir"], "--ego", P["ego"], "--est-dir", P["est_dir"], "--episode-map", P["episode_map"], "--add81-result", P["add81_result"],
            "--add81-controls", P["add81_controls"], "--out-dir", out, "--boot", "300"]


def jl(p):
    return json.load(io.open(p, encoding="utf-8"))


def failed_run(rc, out, o, ctl):
    """rc 1, the named control is red, names empty, values only under withheld, and stdout carries control lines only."""
    C, R = jl(os.path.join(o, "KINCLOSE_CONTROLS.json")), jl(os.path.join(o, "KINCLOSE_RESULT.json"))
    lines = [x for x in out.strip().split("\n") if x]
    only = all(x.startswith("CONTROL ") or x.startswith("ZZ KINCLOSE-CONTROLS") for x in lines)
    return bool(rc == 1 and not C["controls"][ctl]["pass"] and R["names"] == [] and "withheld" in R and "arms" not in R and only and "r_pooled" not in out)


def main():
    tmp = os.path.join(HERE, "tmp_work")
    os.makedirs(tmp, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=tmp) as root:
        n = [0]

        def world(mut=None, mode="signal"):
            n[0] += 1
            return K.write_world(os.path.join(root, "w%d" % n[0]), mode=mode, mut=mut)

        def go(P):
            n[0] += 1
            o = os.path.join(root, "o%d" % n[0])
            rc, out, err = run(read_args(P, o))
            return rc, out, err, o

        P0 = world()
        rc, out, err, o0 = go(P0)
        lines = [x for x in out.strip().split("\n") if x]
        case("clean world: rc 0, five CONTROL PASS lines, last line ZZ ... 5/5 all_pass True, nothing else on stdout",
             rc == 0 and len(lines) == 6 and all(x.endswith("PASS") for x in lines[:5]) and lines[-1] == "ZZ KINCLOSE-CONTROLS 5/5 all_pass True" and "r_pooled" not in out)
        R0 = jl(os.path.join(o0, "KINCLOSE_RESULT.json"))
        case("clean world: names present, arms present, no withheld", len(R0["names"]) >= 3 and "arms" in R0 and "withheld" not in R0 and "KINCLOSE-K25-BETTER-THAN-OUTPUTS" in R0["names"])
        case("clean world: every arm of section 2 is in the result (K S E at 5 10 25 50, B BS, K on the B frames, the cell)",
             all(("%s%d" % (k, w)) in R0["arms"] for k in ("K", "S", "E", "B", "BS") for w in K.WS) and all(("K%d_onB" % w) in R0["arms"] for w in K.WS) and "L25" in R0["arms"]
             and all(p in R0["paired"] for p in ("K25-L25", "K25-S25", "B25-BS25", "K5-L25", "K10-L25", "K50-L25")))
        arm0 = R0["arms"]["K25"]
        case("clean world: K25 carries r, interval, per seed, mae, error sd, lag-1 per seed, n", all(k in arm0 for k in ("r_pooled", "r_ci", "r_per_seed", "mae", "err_sd", "err_lag1", "n_frames")) and len(arm0["err_lag1"]) == 2)

        # show
        c0, r0 = os.path.join(o0, "KINCLOSE_CONTROLS.json"), os.path.join(o0, "KINCLOSE_RESULT.json")
        rcs, outs, _ = run(["show", "--result", r0, "--controls", c0])
        case("show: prints names and a table when the controls passed and the result is bound", rcs == 0 and "NAME KINCLOSE-K25-BETTER-THAN-OUTPUTS" in outs and "K25" in outs)

        case("clips of three lengths are read and the control file records the frames per clip", len(jl(c0)["controls"]["k2"]["frames_per_clip"]) == 3)

        # mutated inputs
        def flip_v(st):
            st["ego_v"] = -st["ego_v"]
        rc, out, err, o = go(world(flip_v))
        case("ego speed sign flipped in the file: k2 red, no name, values withheld", failed_run(rc, out, o, "k2"))
        okf = os.path.join(o, "KINCLOSE_RESULT.json")
        rcs, outs, _ = run(["show", "--result", okf, "--controls", os.path.join(o, "KINCLOSE_CONTROLS.json")])
        case("show refuses when a control failed", rcs == 2 and "REFUSED" in outs and "NAME" not in outs)

        def shift_v(st):
            v = st["ego_v"]
            st["ego_v"] = v[:, np.clip(np.arange(v.shape[1]) - 10, 0, v.shape[1] - 1)]
        rc, out, err, o = go(world(shift_v))
        case("ego speed shifted by 10 frames in the file: k1 red", failed_run(rc, out, o, "k1"))

        def dt_half(st):
            for k in st["dumps"]:
                st["dumps"][k]["dt_s"] = np.float64(0.1)
        rc, out, err, o = go(world(dt_half))
        case("frame period 0.1 in the dump: k0 red", failed_run(rc, out, o, "k0"))

        def alt_r(st):
            st["add81_res"]["arms"]["T25"]["cell_mae"] += 0.01
        rc, out, err, o = go(world(alt_r))
        case("cell error that differs from the result file of card ADD81: k3 red", failed_run(rc, out, o, "k3"))

        def alt_n(st):
            st["add81_ctl"]["controls"]["t0"]["frames"]["H25_seed1"]["n_tool"] += 1
        rc, out, err, o = go(world(alt_n))
        case("frame count that differs from the control file of card ADD81: k3 red", failed_run(rc, out, o, "k3"))

        def alt_est(st):
            st["est"][0]["w"] = st["est"][0]["w"] * 1.5
        # the estimator of seed 0 is written before mut runs, so rewrite the file after the world exists
        P = world()
        w = dict(np.load(os.path.join(P["est_dir"], "closest_L25_s0.npz"), allow_pickle=False))
        K.CL.save_linear(os.path.join(P["est_dir"], "closest_L25_s0.npz"), 25, w["mu"], w["sd"], w["w"] * 1.5, float(w["b"]), seed=0)
        rc, out, err, o = go(P)
        case("another cell estimator file: k3 red", failed_run(rc, out, o, "k3"))

        def drop_valid(st):
            D = st["dumps"][0]
            D["valid"] = np.asarray(D["valid"]).copy()
            D["valid"][::37] = False
        rc, out, err, o = go(world(drop_valid))
        case("scored frames of one reader seed change (valid mask): k3 red", failed_run(rc, out, o, "k3"))

        def tiny_v(st):
            st["ego_v"] = st["ego_v"] * 0.01
        rc, out, err, o = go(world(tiny_v))
        case("ego speed with mean below 1 m/s: k2 red", failed_run(rc, out, o, "k2"))

        # refusals
        P = world()
        z = dict(np.load(P["ego"], allow_pickle=False))
        np.savez(P["ego"], clips=z["clips"][1:], v=z["v"][1:], n=z["n"][1:])
        rc, out, err, o = go(P)
        case("ego file with a missing clip: refused (exit 2), no output file", rc == 2 and "REFUSED" in out and not os.path.exists(os.path.join(o, "KINCLOSE_RESULT.json")))
        P = world()
        z = dict(np.load(P["ego"], allow_pickle=False))
        np.savez(P["ego"], clips=z["clips"], v=z["v"], n=z["n"] - 1)
        rc, out, err, o = go(P)
        case("ego file with one frame fewer per clip than the dump: refused (exit 2)", rc == 2 and "REFUSED" in out)
        P = world()
        z = dict(np.load(P["ego"], allow_pickle=False))
        np.savez(P["ego"], clips=z["clips"], v=z["v"])
        rc, out, err, o = go(P)
        case("ego file without the frames per clip: refused (exit 2)", rc == 2 and "REFUSED" in out)
        P = world()
        rc1, _, _, oa = go(P)
        rc2, out2, _ = run(read_args(P, oa))
        case("an existing output file: refused (exit 2)", rc1 == 0 and rc2 == 2 and "REFUSED" in out2)

        # ego file order does not matter (the join is by clip id)
        P = world()
        z = dict(np.load(P["ego"], allow_pickle=False))
        perm = np.random.default_rng(1).permutation(len(z["clips"]))
        np.savez(P["ego"], clips=z["clips"][perm], v=z["v"][perm], n=z["n"][perm])
        rc, out, err, o = go(P)
        Rp = jl(os.path.join(o, "KINCLOSE_RESULT.json"))
        case("ego file in another clip order gives the same K25 r (join by clip id)", rc == 0 and abs(Rp["arms"]["K25"]["r_pooled"] - R0["arms"]["K25"]["r_pooled"]) < 1e-12)

        # show refusals on a bound result
        cc = jl(c0)
        cc["note"] = "another file with the same verdict"
        t2 = os.path.join(root, "other_controls.json")
        io.open(t2, "w", encoding="utf-8", newline="\n").write(json.dumps(cc))
        rcs, outs, _ = run(["show", "--result", r0, "--controls", t2])
        case("show refuses a result that is bound to another control file", rcs == 2 and "REFUSED" in outs and "bound" in outs)
        cc = jl(c0)
        cc["n_passed"] = 4
        tp = os.path.join(root, "tampered_controls.json")
        io.open(tp, "w", encoding="utf-8", newline="\n").write(json.dumps(cc))
        rcs, outs, _ = run(["show", "--result", r0, "--controls", tp])
        case("show refuses a control file that was changed after the read", rcs == 2 and "REFUSED" in outs)
        R1 = jl(r0)
        R1["names"] = []
        tr = os.path.join(root, "tampered_result.json")
        io.open(tr, "w", encoding="utf-8", newline="\n").write(json.dumps(R1))
        rcs, outs, _ = run(["show", "--result", tr, "--controls", c0])
        case("show refuses a result without names", rcs == 2 and "REFUSED" in outs)

        # no mutated pass: the noise-only world still runs the controls on the same data path
        rc, out, err, o = go(world(mode="noise"))
        case("noise-only gap read: the read runs to its end (exit 0 or 1) with a result file", rc in (0, 1) and os.path.exists(os.path.join(o, "KINCLOSE_RESULT.json")))
    tot, ok = len(RES), sum(RES)
    print("ZZ TEST-KINCLOSE %d/%d" % (ok, tot))
    return 0 if ok == tot else 1


if __name__ == "__main__":
    sys.exit(main())
