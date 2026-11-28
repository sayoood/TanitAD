"""Measurement behind PURITY_ACC_FINDING.md (CPU, tiny synthetic model, no data). Run on an export of commit 202849da with the stack/ folders on sys.path and
stack/tests/test_v8f_otband.py present (it supplies the synthetic world and helpers):
    CUDA_VISIBLE_DEVICES=-1 python measure_acc_purity.py <export root> <scratch dir>
Prints, for a CTL-like, a BAND-like and a no-detach model with w_acc 49.6969: the acceleration term's gradient on S/T parameters and on O's head, and the key set the launch probe reads."""
import pathlib
import sys

root, scratch = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
sys.path[:0] = [str(root / "stack"), str(root / "stack" / "scripts"), str(root / "stack" / "tests")]
import test_v8f_otband as T  # noqa: E402
from tanitad.data import v8f_mini_data as D  # noqa: E402

scratch.mkdir(parents=True, exist_ok=True)
w = D.make_synthetic_world(scratch, n_train=6, n_eval=2, T=60, seed=0)
b, J = T.make_batch({"root": scratch, **w})
for label, kw in (("CTL-like (detach, w_acc 49.6969)", dict(detach_upper=True, w_acc=49.6969)),
                  ("BAND-like (detach, band, w_acc)", dict(detach_upper=True, w_acc=49.6969, **T.BAND)), ("no detach, w_acc", dict(w_acc=49.6969))):
    m = T.build(J, wake=True, **kw)
    o = T.run_loss(m, b)
    print(label, "acc term", float(o["terms"]["acc"].detach()), "| grad on S/T:", T.gsum(o["terms"]["acc"], T.named(m, *T.UPPER)), "| grad on head_o:", T.gsum(o["terms"]["acc"], T.named(m, "head_o.")))
    if kw.get("detach_upper"):
        m2 = T.build(J, wake=False, **kw)
        print("   launch probe by_term keys:", sorted(T.TR.oband_probe(m2, b, m2.cfg)["purity"]["by_term"]))
