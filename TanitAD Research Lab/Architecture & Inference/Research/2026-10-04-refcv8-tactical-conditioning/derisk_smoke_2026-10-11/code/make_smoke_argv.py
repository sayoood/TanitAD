import json, sys
sys.path.insert(0, sys.argv[1] + "/stack/scripts")
import launch_gate as lg
src, dst = sys.argv[2], sys.argv[3]
doc = json.load(open(src))
argv = doc["argv"] if isinstance(doc, dict) else doc
out, dep = lg.smoke_argv(list(argv), 100, 50)
json.dump({"schema": "tanitad.smoke_argv/1", "src": src, "departures": dep, "argv": out}, open(dst, "w"), indent=1)
print("DEPARTURES", json.dumps(dep))
def g(f):
    v = [out[i + 1] for i, t in enumerate(out) if t == f]
    return v[-1] if v else ("<present>" if f in out else None)
for f in ("--steps", "--grad-share-every", "--grad-share-at", "--grad-share-terms", "--r8-alloc-emit-start", "--r8-alloc-emit",
          "--b1-guard-step", "--b1-guard-ref-ap2m", "--ema-decay", "--r8-no-rc", "--r8-speed-input", "--speed-max-sidecar-v6",
          "--speed-max-sidecar-v6-eval", "--init-from", "--out", "--log-every", "--eval-every", "--save-every", "--w-tac-v6", "--w-map-hires", "--batch", "--eval-batches"):
    print(f, "=", g(f))
print("NTOK", len(out))
