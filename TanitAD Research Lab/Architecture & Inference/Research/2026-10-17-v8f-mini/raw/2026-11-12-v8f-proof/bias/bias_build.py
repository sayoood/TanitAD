"""bias_build.py -- builds GOALEDIT_b0/b1/b2 on BAL-NULL_s0 and BAL-NULL_s1 (PREREG_V8F_PROOF_ADD4.md): parent theta with b added to ONE scalar, h_goal.bias[STOP_I]; nothing else changes.
Tags GE0_s*, GE1_s*, GE2_s*  ->  bias/ckpt/<TAG>/ckpt_last.pt (+ WSP_MANIFEST.json, so wsp_dump finds the sha) ; BUILD_REPORT.json beside this file.
Control e1 (inline): every tensor bit-identical to the parent except h_goal.bias; there exactly element STOP_I differs (none for GE0); diff == ln(W) to float32 precision.
W1, W2 are re-read from the trained arms' argv.txt (s0 and s1).  Usage: python -I bias_build.py   (refuses an existing <ckpt>/<TAG>)"""
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bias_common as B  # noqa: E402
import wsp_common as C  # noqa: E402
import torch  # noqa: E402


def bits_equal(x, y):
    return x.shape == y.shape and x.dtype == y.dtype and bool(torch.equal(x.contiguous().view(torch.uint8), y.contiguous().view(torch.uint8)))


def argv_w(tag):
    t = open(f"{B.PULL}/stopbal_BAL-{tag}/argv.txt", encoding="utf-8").read().split()
    return float(t[t.index("--goal-pos-weight-stop") + 1])


def main():
    rep = {"tool": "bias_build.py", "card": "PREREG_V8F_PROOF_ADD4.md (frozen)", "stop_index": B.STOP_I, "bias_key": B.BIAS_KEY, "problems": [], "written": {}}
    w = {t: argv_w(t) for t in ("G_s0", "G2_s0", "G_s1", "G2_s1")}
    rep["W_from_argv"] = w
    if not (w["G_s0"] == w["G_s1"] == B.W1 and w["G2_s0"] == w["G2_s1"] == B.W2):
        rep["problems"].append(f"W read from argv {w} differs from the constants {B.W1}, {B.W2}")
    rep["ln_W1"], rep["ln_W2"] = B.B1, B.B2
    for s, par in B.PARENTS.items():
        src = f"{B.PULL}/stopbal_BAL-{par}"
        man = json.load(open(src + "/BANK_MANIFEST.json", encoding="utf-8"))["files"]["ckpt_last.pt"]["sha256"]
        got = C.sha(src + "/ckpt_last.pt")
        rep.setdefault("parent_sha256", {})[par] = got
        if got != man:
            rep["problems"].append(f"{par}: checkpoint sha256 differs from its BANK_MANIFEST")
            continue
        ck = C.load(src + "/ckpt_last.pt")
        sd = ck["model"]
        assert B.BIAS_KEY in sd and sd[B.BIAS_KEY].shape[0] > B.STOP_I and sd[B.BIAS_KEY].dtype == torch.float32, (sd[B.BIAS_KEY].shape, sd[B.BIAS_KEY].dtype)
        for e, b in B.EDITS.items():
            tag = f"{e}_{s}"
            d = f"{B.CKPT}/{tag}"
            if os.path.exists(d):
                raise SystemExit(f"{d} exists: remove it by its literal path first")
            new = type(sd)((k, v.clone()) for k, v in sd.items())
            new[B.BIAS_KEY][B.STOP_I] = sd[B.BIAS_KEY][B.STOP_I] + torch.tensor(b, dtype=torch.float32)
            diff_keys = [k for k in sd if not bits_equal(sd[k], new[k])]
            idx = [int(i) for i in (sd[B.BIAS_KEY] != new[B.BIAS_KEY]).nonzero().flatten()]
            n_scalars = sum(int((x.contiguous().view(torch.uint8) != y.contiguous().view(torch.uint8)).any().item()) for x, y in [(sd[B.BIAS_KEY], new[B.BIAS_KEY])])
            delta = float(new[B.BIAS_KEY][B.STOP_I].double() - sd[B.BIAS_KEY][B.STOP_I].double())
            tol = 2 * torch.finfo(torch.float32).eps * max(abs(float(sd[B.BIAS_KEY][B.STOP_I])), abs(float(new[B.BIAS_KEY][B.STOP_I])), abs(b), 1e-30)
            e1 = {"tensors_differing": diff_keys, "elements_differing_in_bias": idx, "delta_float64": delta, "target_b": b, "abs_err": abs(delta - b), "tolerance_2eps": tol,
                  "parent_value": float(sd[B.BIAS_KEY][B.STOP_I]), "edited_value": float(new[B.BIAS_KEY][B.STOP_I])}
            e1["pass"] = (diff_keys == [] and idx == []) if b == 0.0 else (diff_keys == [B.BIAS_KEY] and idx == [B.STOP_I] and abs(delta - b) <= tol)
            if not e1["pass"]:
                rep["problems"].append(f"e1 FAILED for {tag}: {e1}")
            os.makedirs(d)
            out = dict(ck)
            out["model"] = new
            out["opt"] = {}
            out["arm"] = tag
            out["bias"] = {"parent": par, "parent_sha256": got, "key": B.BIAS_KEY, "index": B.STOP_I, "b": b, "built": time.strftime("%Y-%m-%d %H:%M:%S")}
            torch.save(out, d + "/ckpt_last.pt")
            h = C.sha(d + "/ckpt_last.pt")
            back = C.load(d + "/ckpt_last.pt")
            C.build_model(back)                                   # strict load
            rt = all(bits_equal(back["model"][k], new[k]) for k in new)
            e1["roundtrip_bit_identical"], e1["strict_load_ok"] = rt, True
            json.dump({"tag": tag, "ckpt_sha256": h, "bytes": os.path.getsize(d + "/ckpt_last.pt"), "provenance": out["bias"], "e1": e1}, open(d + "/WSP_MANIFEST.json", "w"), indent=1)
            rep["written"][tag] = {"sha256": h, "e1": e1}
            print("WROTE", tag, h[:12], "e1", e1["pass"], "delta", delta, flush=True)
        del ck, sd
    rep["pass"] = not rep["problems"]
    json.dump(rep, open(B.BIAS + "/BUILD_REPORT.json", "w"), indent=1)
    print("DONE problems:", rep["problems"])
    return 0 if rep["pass"] else 2


if __name__ == "__main__":
    sys.exit(main())
