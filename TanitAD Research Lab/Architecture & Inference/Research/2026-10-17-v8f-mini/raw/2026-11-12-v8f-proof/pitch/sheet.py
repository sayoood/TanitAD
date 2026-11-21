"""sheet.py <framesdir> <out prefix> t1,t2,... : 2x2 contact sheets (each frame 960x540) of the given frame times."""
import sys
from PIL import Image
d, pre, ts = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
for k in range(0, len(ts), 4):
    sh = Image.new("RGB", (1920, 1080), (0, 0, 0))
    for j, t in enumerate(ts[k:k + 4]):
        im = Image.open("%s/f_%07.2f.png" % (d, float(t))).convert("RGB").resize((960, 540), Image.LANCZOS)
        sh.paste(im, ((j % 2) * 960, (j // 2) * 540))
    sh.save("%s_%d.jpg" % (pre, k // 4), quality=88)
    print("%s_%d.jpg" % (pre, k // 4))
