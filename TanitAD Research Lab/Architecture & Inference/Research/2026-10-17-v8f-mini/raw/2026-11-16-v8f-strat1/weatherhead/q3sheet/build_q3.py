"""Build the Q3 blind weather-label audit sheet. Draw rules are fixed in PREREG_Q3.md.
Writes (all next to this script): Q3_SHEET.html (frames embedded, LOCAL ONLY), key.json (no images), draw_report.json.
CPU only; ffmpeg decodes 40 (+ up to 40) single frames."""
import base64, hashlib, json, os, shutil, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TABLE = "F:/TanitAD-scratch/strat1/labels/strat_labels_table.npz"
TABLE_SHA = "8055a321df6db1725ea17abde2999d48bcb68d43f1a480a69ed42dd0cf184eff"
CAM = "C:/Users/Admin/tanitad-data/physicalai/camera/camera_front_wide_120fov"
EXCL = HERE + "/ref/pilot_audit_answers_PI_60.json"
SEED = "q3-weather-v1"
VOCAB = ["clear", "fog", "rain", "snow"]
ORDER = [("fog", 11), ("rain", 11), ("snow", 11), ("clear", 7)]
FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
WIDTH = 640


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def H(s):
    return hashlib.sha256(s.encode()).hexdigest()


def jpeg(clip, vf):
    cmd = [FFMPEG, "-v", "error", "-i", f"{CAM}/{clip}.mp4", "-vf", f"select=eq(n\\,{vf}),scale={WIDTH}:-2:flags=area", "-fps_mode", "passthrough",
           "-frames:v", "1", "-f", "image2pipe", "-c:v", "mjpeg", "-q:v", "6", "pipe:1"]
    r = subprocess.run(cmd, capture_output=True, stdin=subprocess.DEVNULL)
    if r.returncode != 0 or not r.stdout:
        raise RuntimeError(f"ffmpeg failed {clip} {vf}: {r.stderr[-200:]!r}")
    return base64.b64encode(r.stdout).decode()


def draw():
    assert sha(TABLE) == TABLE_SHA
    z = np.load(TABLE)
    excl = {x["clip_id"] for x in json.load(open(EXCL))["frames"]}
    elig = {c: [] for c in VOCAB}
    counts = [0] * 4
    for ci, (cl, o, n) in enumerate(zip(z["train_clip"], z["train_off"], z["train_n"])):
        if not os.path.exists(f"{CAM}/{cl}.mp4"):
            continue
        for t in np.unique(np.round(np.linspace(3, int(n) - 1, 8)).astype(int)):
            r = int(o) + int(t)
            if z["train_flag"][r] == 0 and z["train_weather_valid"][r]:
                w = int(z["train_weather_id"][r])
                counts[w] += 1
                if str(cl) not in excl:
                    elig[VOCAB[w]].append((str(cl), int(t), int(z["train_frame_idx"][r])))
    used, items = set(), []
    for cname, q in ORDER:
        cand = sorted(elig[cname], key=lambda x: H(f"{SEED}:{x[0]}:{x[1]}"))
        got = 0
        for cl, t, k in cand:
            if got >= q:
                break
            if cl in used:
                continue
            used.add(cl)
            items.append({"clip": cl, "step": t, "k": k, "label": cname})
            got += 1
        assert got == q, (cname, got)
    return items, counts, excl, {c: len({x[0] for x in v}) for c, v in elig.items()}


TPL = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Weather frame audit</title>
<style>
:root{--bg:#fff;--fg:#1a1a1a;--mut:#666;--bd:#ccc;--acc:#0b5fff}
@media (prefers-color-scheme:dark){:root{--bg:#161616;--fg:#e8e8e8;--mut:#9a9a9a;--bd:#444;--acc:#6aa0ff}}
body{background:var(--bg);color:var(--fg);font:15px/1.4 system-ui,sans-serif;margin:0 auto;max-width:1100px;padding:0 16px 80px}
h1{font-size:20px}.card{border:1px solid var(--bd);border-radius:8px;padding:12px;margin:14px 0}
.imgs{display:flex;gap:10px;flex-wrap:wrap}.imgs figure{margin:0;flex:1 1 300px}.imgs img{width:100%;height:auto;display:block;border-radius:4px}
figcaption{font-size:12px;color:var(--mut)}label{margin-right:14px;white-space:nowrap;cursor:pointer}
input[type=text]{width:100%;box-sizing:border-box;margin-top:6px;padding:4px;background:var(--bg);color:var(--fg);border:1px solid var(--bd)}
.bar{position:sticky;top:0;background:var(--bg);padding:8px 0;border-bottom:1px solid var(--bd);display:flex;gap:12px;align-items:center;z-index:2}
button{padding:6px 12px;font:inherit}textarea{width:100%;height:120px;background:var(--bg);color:var(--fg)}
</style></head><body>
<h1>Weather frame audit (40 frames)</h1>
<p>For each frame choose the weather you see in the front camera: clear, fog, rain or snow. "Can't tell" is allowed. The frame on the right (when shown) is one second earlier, for context only. Nothing here shows any label or model output.</p>
<div class="bar"><span id="prog">0 / 40 answered</span><button id="copy">Copy answers</button><span id="msg"></span></div>
<div id="list"></div>
<p>Answers (JSON, also copied by the button):</p><textarea id="out" readonly></textarea>
<script>
const ITEMS=__ITEMS__;
const CLS=["clear","fog","rain","snow","cant_tell"],LAB={"cant_tell":"can't tell"};
const list=document.getElementById("list");
ITEMS.forEach((it,i)=>{
 const d=document.createElement("div");d.className="card";
 let h='<b>Frame '+(i+1)+' of '+ITEMS.length+'</b><div class="imgs"><figure><img alt="now" src="data:image/jpeg;base64,'+it.now+'"><figcaption>now</figcaption></figure>';
 if(it.prev)h+='<figure><img alt="1 s earlier" src="data:image/jpeg;base64,'+it.prev+'"><figcaption>1 s earlier</figcaption></figure>';
 h+='</div><div>';
 CLS.forEach(c=>{h+='<label><input type="radio" name="r'+i+'" value="'+c+'"> '+(LAB[c]||c)+'</label>'});
 h+='</div><input type="text" placeholder="note (optional)" id="n'+i+'">';
 d.innerHTML=h;list.appendChild(d);
});
function collect(){
 const a={};let n=0;
 ITEMS.forEach((it,i)=>{const r=document.querySelector('input[name=r'+i+']:checked');
  const note=document.getElementById('n'+i).value.trim();
  a[it.id]={answer:r?r.value:null,note:note};if(r)n++;});
 return {schema:"q3-answers-v1",n_answered:n,answers:a};
}
function refresh(){const o=collect();document.getElementById("prog").textContent=o.n_answered+" / "+ITEMS.length+" answered";
 document.getElementById("out").value=JSON.stringify(o,null,1);try{localStorage.setItem("q3draft",JSON.stringify(o))}catch(e){}}
document.addEventListener("input",refresh);
document.getElementById("copy").onclick=async()=>{refresh();const t=document.getElementById("out").value;
 try{await navigator.clipboard.writeText(t);document.getElementById("msg").textContent="copied"}catch(e){document.getElementById("out").select();document.getElementById("msg").textContent="select and copy the box below"}};
try{const d=JSON.parse(localStorage.getItem("q3draft")||"null");if(d&&d.answers){ITEMS.forEach((it,i)=>{const x=d.answers[it.id];if(!x)return;
 if(x.answer){const e=document.querySelector('input[name=r'+i+'][value='+x.answer+']');if(e)e.checked=true}document.getElementById('n'+i).value=x.note||""})}}catch(e){}
refresh();
</script></body></html>"""


def main():
    items, counts, excl, nclips = draw()
    rng = np.random.default_rng(20261117)
    items = [items[i] for i in rng.permutation(len(items))]
    shown, key = [], []
    for it in items:
        fid = H(f"{SEED}-id:{it['clip']}:{it['k']}")[:8]
        now = jpeg(it["clip"], 6 * it["k"])
        prev = jpeg(it["clip"], 6 * (it["k"] - 5)) if it["k"] >= 5 else None
        shown.append({"id": fid, "now": now, "prev": prev})
        key.append({"id": fid, "clip": it["clip"], "step": it["step"], "frame_idx": it["k"], "video_frame": 6 * it["k"], "label": it["label"],
                    "prev_shown": prev is not None})
    html = TPL.replace("__ITEMS__", json.dumps(shown, separators=(",", ":")))
    assert "https://" not in html and "http://" not in html
    out = HERE + "/Q3_SHEET.html"
    open(out, "w", encoding="utf-8").write(html)
    ids = [k["id"] for k in key]
    assert len(set(ids)) == 40 and len({k["clip"] for k in key}) == 40
    assert not ({k["clip"] for k in key} & excl)
    rep = {"seed": SEED, "perm_seed": 20261117, "table_sha256": TABLE_SHA, "prereg_sha256": sha(HERE + "/PREREG_Q3.md"),
           "label_counts_train_windows_clear_fog_rain_snow": counts, "eligible_clips_after_exclusion": nclips,
           "drawn_per_class": {c: sum(k["label"] == c for k in key) for c in VOCAB}, "excluded_clips": len(excl),
           "excluded_in_draw": 0, "sheet_bytes": os.path.getsize(out), "frames_with_prev": sum(k["prev_shown"] for k in key),
           "build_script_sha256": sha(__file__)}
    json.dump(key, open(HERE + "/key.json", "w"), indent=1)
    json.dump(rep, open(HERE + "/draw_report.json", "w"), indent=1)
    print(json.dumps(rep))


if __name__ == "__main__":
    main()
