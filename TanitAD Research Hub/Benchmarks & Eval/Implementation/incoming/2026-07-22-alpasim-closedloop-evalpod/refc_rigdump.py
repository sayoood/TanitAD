"""Dump the camera_front_wide_120fov f-theta sensor entry from the USDZ rig_json + mp4 res."""
import ast, io, json, zipfile

usdz = ("/workspace/scene_dl/sample_set/26.04_release/"
        "01d503d4-449b-46fc-8d78-9085e70d3554/"
        "01d503d4-449b-46fc-8d78-9085e70d3554.usdz")
z = zipfile.ZipFile(usdz)
import pandas as pd
df = pd.read_parquet(io.BytesIO(z.read("clipgt/calibration_estimate.parquet")))
cell = df.iloc[0]["calibration_estimate"]
outer = ast.literal_eval(cell) if isinstance(cell, str) else cell
rig = json.loads(outer["rig_json"]) if isinstance(outer.get("rig_json"), str) else outer["rig_json"]
print("rig top keys:", list(rig.keys()))
rr = rig.get("rig", rig)
print("rig.rig keys:", list(rr.keys()) if isinstance(rr, dict) else type(rr))
sensors = rr.get("sensors", [])
print("n_sensors:", len(sensors))
for s in sensors:
    name = s.get("name", "").lower().replace(":", "_")
    if "front_wide" in name:
        print("\n===== front:wide:120fov SENSOR =====")
        print(json.dumps(s, indent=2)[:4500])
        break
else:
    print("front_wide not found; sensor names:", [s.get("name") for s in sensors])

# mp4 resolution
print("\n===== mp4 probe =====")
import subprocess
mp4 = ("/workspace/scene_dl/sample_set/26.04_release/"
       "01d503d4-449b-46fc-8d78-9085e70d3554/camera_front_wide_120fov.mp4")
try:
    import cv2
    cap = cv2.VideoCapture(mp4)
    print("frames", cap.get(cv2.CAP_PROP_FRAME_COUNT),
          "w", cap.get(cv2.CAP_PROP_FRAME_WIDTH),
          "h", cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
          "fps", cap.get(cv2.CAP_PROP_FPS))
    cap.release()
except Exception as e:
    print("cv2 probe failed:", repr(e)[:150])
print("RIGDUMP_DONE")
