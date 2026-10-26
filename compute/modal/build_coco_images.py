"""
Materialise the Chimera COCO corpus on a Modal CPU container.

Downloads only the images referenced by data/chimera_coco/manifest.json from COCO's official image server, center-crops
each to a square, resizes to 512 px and stores them by COCO id in the `chimera-data` volume:

  /data/chimera_coco/images/<id>.jpg
  /data/chimera_coco/manifest.json
  /data/chimera_coco/export_stats.json
  /data/chimera_coco.zip              (same content, for upload to Kaggle)

Run:  modal run compute/modal/build_coco_images.py
"""
import json
import pathlib

import modal

app = modal.App("chimera-build-coco")
image = modal.Image.debian_slim(python_version="3.12").pip_install("pillow", "requests")
data = modal.Volume.from_name("chimera-data", create_if_missing=True)

COCO_URL = "http://images.cocodataset.org/{split}2017/{file_name}"
RES = 512


@app.function(image=image, volumes={"/data": data}, cpu=4, memory=4096, timeout=3600)
def build(manifest_json: str):
    import io
    import os
    import time
    import zipfile
    from concurrent.futures import ThreadPoolExecutor, as_completed

    import requests
    from PIL import Image

    manifest = json.loads(manifest_json)
    rows = list(manifest["clean_train"])
    for p in manifest["pairs"].values():
        rows += p["poison_pool"] + p["target_anchor"]
    unique = {r["id"]: r for r in rows}
    out_dir = "/data/chimera_coco/images"
    os.makedirs(out_dir, exist_ok=True)
    print(f"{len(unique)} unique images to fetch", flush=True)

    session = requests.Session()

    def fetch(r):
        dst = f"{out_dir}/{r['id']}.jpg"
        if os.path.exists(dst):
            return r["id"], "cached"
        url = COCO_URL.format(split=r["split"], file_name=r["file_name"])
        for attempt in range(4):
            try:
                resp = session.get(url, timeout=30)
                resp.raise_for_status()
                im = Image.open(io.BytesIO(resp.content)).convert("RGB")
                w, h = im.size
                s = min(w, h)
                im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
                im.resize((RES, RES), Image.LANCZOS).save(dst, quality=95)
                return r["id"], "ok"
            except Exception as e:  # retry transient network errors
                last = f"{type(e).__name__}: {e}"
                time.sleep(2 * (attempt + 1))
        return r["id"], f"failed: {last}"

    status = {}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = [pool.submit(fetch, r) for r in unique.values()]
        for n, f in enumerate(as_completed(futures), 1):
            iid, st = f.result()
            status[iid] = st
            if n % 250 == 0 or n == len(futures):
                print(f"  {n}/{len(futures)}  {time.time() - t0:.0f}s", flush=True)

    failed = {i: s for i, s in status.items() if s.startswith("failed")}
    stats = {"resolution": RES, "requested": len(unique), "stored": len(unique) - len(failed),
             "failed": failed, "source": "images.cocodataset.org (COCO 2017)"}
    pathlib.Path("/data/chimera_coco/manifest.json").write_text(json.dumps(manifest, indent=1))
    pathlib.Path("/data/chimera_coco/export_stats.json").write_text(json.dumps(stats, indent=1))

    with zipfile.ZipFile("/data/chimera_coco.zip", "w", compression=zipfile.ZIP_STORED) as zf:
        for iid in unique:
            if iid not in failed:
                zf.write(f"{out_dir}/{iid}.jpg", f"images/{iid}.jpg")
        zf.writestr("manifest.json", json.dumps(manifest, indent=1))
        zf.writestr("export_stats.json", json.dumps(stats, indent=1))
    data.commit()
    stats["zip_mb"] = round(os.path.getsize("/data/chimera_coco.zip") / 2**20, 1)
    return json.dumps(stats)


@app.local_entrypoint()
def main():
    root = pathlib.Path(__file__).resolve().parents[2]
    manifest = (root / "data" / "chimera_coco" / "manifest.json").read_text()
    stats = json.loads(build.remote(manifest))
    print(json.dumps({k: v for k, v in stats.items() if k != "failed"}, indent=2))
    print("failed:", len(stats["failed"]))
