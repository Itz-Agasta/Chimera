"""
Materialise the Chimera COCO corpus (CPU only).

Reads manifest.json (itzagasta/chimera-coco-manifest), takes each referenced image from the Kaggle COCO 2017 copy
(awsaf49/coco-2017-dataset), center-crops it to a square, resizes to 512 px and stores it once by COCO id.
Output: /kaggle/working/chimera_coco.zip containing images/<id>.jpg, manifest.json and export_stats.json.
"""
import glob
import json
import os
import zipfile

from PIL import Image

RES = 512
# Never glob recursively under /kaggle/input: it includes the 123k-file COCO mount and the walk can hang for
# an hour before the first line of output. Only look a few fixed levels deep.
def find(pattern):
    for depth in ("/kaggle/input/*/", "/kaggle/input/*/*/", "/kaggle/input/*/*/*/", "/kaggle/input/*/*/*/*/"):
        hits = glob.glob(depth + pattern)
        if hits:
            return hits[0]
    raise SystemExit(f"not found under /kaggle/input: {pattern}")


manifest = json.load(open(find("manifest.json")))
roots = {s: find(f"{s}2017") for s in ("train", "val")}
print("COCO roots:", roots, flush=True)

rows = list(manifest["clean_train"])
for p in manifest["pairs"].values():
    rows += p["poison_pool"] + p["target_anchor"]
unique = {r["id"]: r for r in rows}
print(f"{len(rows)} rows, {len(unique)} unique images", flush=True)


def square_512(path):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    s = min(w, h)
    im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    return im.resize((RES, RES), Image.LANCZOS), (w, h)


missing, sizes = [], {}
with zipfile.ZipFile("/kaggle/working/chimera_coco.zip", "w", compression=zipfile.ZIP_STORED) as zf:
    for n, (iid, r) in enumerate(unique.items()):
        src = os.path.join(roots[r["split"]], r["file_name"])
        if not os.path.exists(src):
            missing.append(r["file_name"])
            continue
        im, orig = square_512(src)
        tmp = f"/tmp/{iid}.jpg"
        im.save(tmp, quality=95)
        zf.write(tmp, f"images/{iid}.jpg")
        os.remove(tmp)
        sizes[iid] = orig
        if n % 500 == 0:
            print(f"  {n}/{len(unique)}", flush=True)
    stats = {"resolution": RES, "exported": len(sizes), "missing": missing,
             "min_original_side": min(min(s) for s in sizes.values())}
    zf.writestr("manifest.json", json.dumps(manifest, indent=1))
    zf.writestr("export_stats.json", json.dumps(stats, indent=1))
print(json.dumps(stats), flush=True)
print("zip size MB:", round(os.path.getsize("/kaggle/working/chimera_coco.zip") / 2**20, 1))
