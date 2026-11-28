import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v2_scenes import *
from PIL import Image
name, args = sys.argv[1], eval(sys.argv[2]) if len(sys.argv) > 2 else ()
out = sys.argv[3] if len(sys.argv) > 3 else f"{OUT}/_t_{name}.png"
t0 = time.time()
img = SCENES[name](*args)
Image.fromarray(img).save(out)
print(name, args, f"{time.time()-t0:.1f}s", out)
