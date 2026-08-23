"""C2 — extract the f-theta camera intrinsics from the USDZ calibration parquet + parsed_config."""
import io, json, re, zipfile

usdz = ("/workspace/scene_dl/sample_set/26.04_release/"
        "01d503d4-449b-46fc-8d78-9085e70d3554/"
        "01d503d4-449b-46fc-8d78-9085e70d3554.usdz")
z = zipfile.ZipFile(usdz)

print("===== calibration_estimate.parquet =====")
try:
    import pandas as pd
    df = pd.read_parquet(io.BytesIO(z.read("clipgt/calibration_estimate.parquet")))
    print("columns:", list(df.columns))
    print("shape:", df.shape)
    # find the front-wide row
    for col in df.columns:
        if df[col].astype(str).str.contains("front_wide", case=False, na=False).any():
            print("id col:", col)
            break
    for _, row in df.iterrows():
        rd = {k: str(v)[:300] for k, v in row.to_dict().items()}
        if any("front_wide" in str(v).lower() for v in rd.values()):
            print("FRONT_WIDE ROW:", json.dumps(rd, indent=2)[:3500])
            break
    else:
        print("row0:", json.dumps({k: str(v)[:300] for k, v in df.iloc[0].to_dict().items()}, indent=2)[:3000])
except Exception as e:
    print("  parquet failed:", repr(e)[:200])

print("\n===== parsed_config.yaml: camera_front_wide_120fov intrinsics =====")
txt = z.read("parsed_config.yaml").decode("utf-8", "replace")
# print blocks mentioning ftheta/intrinsic/front_wide
for kw in ("ftheta", "f_theta", "intrinsic", "poly", "principal", "cx", "cy",
           "backward_poly", "forward_poly", "camera_front_wide_120fov"):
    idxs = [m.start() for m in re.finditer(re.escape(kw), txt, re.IGNORECASE)]
    if idxs:
        print(f"\n--- '{kw}' x{len(idxs)} (first ctx) ---")
        i = idxs[0]
        print(txt[max(0, i-120):i+400])
print("C2_DONE")
