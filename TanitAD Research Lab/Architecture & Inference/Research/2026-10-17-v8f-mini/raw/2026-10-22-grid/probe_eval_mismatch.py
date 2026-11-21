import sys, json, os, tempfile
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, os.environ["PYTHONPATH"].split(";")[0] + "\scripts")
from tanitad.data import v8f_mini_data as D
import v8f_mini_train as TR, v8f_mini_eval as EV
orig = D.make_synthetic_world
def two(root, **kw):
    kw.update(n_train=4, n_eval=2, T=60)
    w = orig(root, **kw)
    rng = np.random.default_rng(99)
    for split in ("train", "eval"):
        for p in sorted((Path(root)/split/"clips").glob("*.pt")):
            o = torch.load(p, map_location="cpu", weights_only=True)
            T = o["pool4x10"].shape[0]
            o["pool8x20"] = torch.as_tensor(rng.normal(0,1,(T,160,D.D_FEAT)).astype(np.float32)).to(torch.float8_e4m3fn)
            torch.save(o, p)
    return w
D.make_synthetic_world = two
tmp = Path(tempfile.mkdtemp(prefix="gridprobe_", dir=os.environ.get("TMPDIR")))
out = tmp / "g8"
common = ["--synthetic","--batch","8","--d","32","--n-q","2","--steps","4","--log-every","2","--warmup","2","--ckpt-every","100","--threads","1","--kin-mode","yaw","--seed","5"]
rc = TR.main(["--out", str(out), *common, "--grid", "pool8x20"]); print("train rc", rc)
ck = torch.load(out/"ckpt_last.pt", map_location="cpu", weights_only=False)
print("ck grid", ck["grid"], "n_tok", ck["cfg"]["n_tok"], "pos", tuple(ck["model"]["pos"].shape))
sc = json.load(open(out/"selfcheck.json")); print("lru", sc["info"]["lru"], "adapter", sc["levers"]["params"]["adapter"], "n_params", sc["levers"]["n_params_trainable"])
root = out/"_synthetic"
def ev(ckpt, tag):
    o = tmp/"ev"
    try:
        rc = EV.main(["run","--ckpt",str(ckpt),"--tag",tag,"--eval-cache",str(root),"--eval-grid",str(root/"eval_grid_windows.json"),"--eval-kin",str(root/"eval_kin.npz"),"--eval-join",str(root/"join_eval.npz"),"--out-dir",str(o),"--n-boot","20","--batch","16","--lru","4","--tracks",str(root/"native_tracks.pt")])
        print(tag, "rc", rc)
    except BaseException as e:
        print(tag, "RAISED", type(e).__name__, str(e)[:200])
ev(out/"ckpt_last.pt", "ok8")
c2 = dict(ck); c2.pop("grid"); torch.save(c2, tmp/"nogrid.pt"); ev(tmp/"nogrid.pt", "nogrid")
c3 = dict(ck); c3["grid"] = "pool4x10"; torch.save(c3, tmp/"wrong.pt"); ev(tmp/"wrong.pt", "wronggrid")
print(tmp)
