"""
Build the Chimera fine-tuning corpus from COCO 2017.

Produces a manifest (no image bytes) describing four disjoint splits per concept pair:

  clean_train   captioned images the victim fine-tunes on (mixed categories, includes clean source-concept images)
  poison_pool   held-out source-concept images the attacker perturbs; captioned with their own COCO captions
  target_anchor target-concept images, used by the anchored Chimera variant and by the dirty-label baseline
  eval_held_out source-concept captions never seen in training, used as evaluation prompts

Selection rules
  * an image belongs to a concept when its instance annotations contain that category AND at least one of its five
    captions mentions it (concepts.mentions), so the caption is a faithful label (clean-label requirement);
  * images containing both a source and its target concept are dropped from every split of that pair;
  * one caption per image (the first that mentions the concept, else the first), so captions stay human-written;
  * splits are disjoint by image id; the seed fixes the choice.

Usage:
  uv run python experiments/common/build_dataset.py --ann <dir with COCO *_2017.json> --out data/chimera_coco
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).parent))
from concepts import CONCEPTS, PAIRS, SOURCES, TARGETS, mentions

# Per concept and split, how many images to take.
N_CLEAN_SOURCE = 300      # clean images of each source concept in clean_train
N_POISON_POOL = 250       # held-out source images available to perturb (max poison count is 200)
N_TARGET_ANCHOR = 250     # target-concept images for anchors / dirty-label baseline
N_EVAL_PROMPTS = 50       # held-out source captions used as evaluation prompts
N_FILLER = 2000           # images of unrelated categories, so poisons compete with a realistic corpus


def load_split(ann_dir: Path, split: str):
    """Return {image_id: {file_name, split, categories:set, captions:[...]}} for one COCO split.

    instances_train2017.json is ~470 MB, so annotations are streamed with ijson instead of json.load
    (a full parse needs several GB and gets killed on small machines).
    """
    import ijson

    inst_path = ann_dir / f"instances_{split}2017.json"
    images = {}
    cat_name = {}
    with inst_path.open("rb") as f:
        for c in ijson.items(f, "categories.item"):
            cat_name[c["id"]] = c["name"]
    with inst_path.open("rb") as f:
        for im in ijson.items(f, "images.item", use_float=True):
            images[im["id"]] = {"file_name": im["file_name"], "split": split,
                               "categories": set(), "captions": []}
    with inst_path.open("rb") as f:
        for a in ijson.items(f, "annotations.item", use_float=True):
            rec = images.get(a["image_id"])
            if rec is not None:
                rec["categories"].add(cat_name[a["category_id"]])
    caps = json.loads((ann_dir / f"captions_{split}2017.json").read_text())
    for a in caps["annotations"]:
        rec = images.get(a["image_id"])
        if rec is not None:
            rec["captions"].append(a["caption"].strip())
    return images


def pick_caption(rec, concept):
    for c in rec["captions"]:
        if mentions(c, concept):
            return c
    return rec["captions"][0] if rec["captions"] else f"a photo of a {concept}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", required=True, type=Path, help="directory holding COCO 2017 annotation json files")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    images = {}
    for split in ("train", "val"):
        if (args.ann / f"instances_{split}2017.json").exists():
            images.update(load_split(args.ann, split))
    if not images:
        raise SystemExit(f"no COCO annotation files found in {args.ann}")
    print(f"loaded {len(images)} images")

    # Images that genuinely depict each concept (category present and a caption that says so).
    by_concept = defaultdict(list)
    for iid, rec in images.items():
        for c in CONCEPTS:
            if c in rec["categories"] and any(mentions(cap, c) for cap in rec["captions"]):
                by_concept[c].append(iid)
    for c in CONCEPTS:
        print(f"  {c:10s} {len(by_concept[c])} candidate images")

    # Drop images containing both halves of a pair (ambiguous for that pair).
    conflict = {}
    for source, target in PAIRS:
        both = {iid for iid in by_concept[source] if target in images[iid]["categories"]}
        both |= {iid for iid in by_concept[target] if source in images[iid]["categories"]}
        conflict[(source, target)] = both
        if both:
            print(f"  dropping {len(both)} ambiguous images for {source}->{target}")

    manifest = {
        "meta": {"seed": args.seed, "source": "COCO 2017", "pairs": [list(p) for p in PAIRS],
                 "counts": {"clean_source": N_CLEAN_SOURCE, "poison_pool": N_POISON_POOL,
                            "target_anchor": N_TARGET_ANCHOR, "eval_prompts": N_EVAL_PROMPTS,
                            "filler": N_FILLER}},
        "pairs": {}, "clean_train": [], "filler": [],
    }
    used = set()

    def take(concept, n, pair_key, exclude):
        pool = [i for i in by_concept[concept] if i not in used and i not in exclude]
        rng.shuffle(pool)
        chosen = pool[:n]
        used.update(chosen)
        if len(chosen) < n:
            print(f"  WARNING {concept} {pair_key}: wanted {n}, got {len(chosen)}")
        return chosen

    for source, target in PAIRS:
        key = f"{source}->{target}"
        ex = conflict[(source, target)]
        clean_src = take(source, N_CLEAN_SOURCE, key, ex)
        poison = take(source, N_POISON_POOL, key, ex)
        evalp = take(source, N_EVAL_PROMPTS, key, ex)
        anchor = take(target, N_TARGET_ANCHOR, key, ex)

        def rows(ids, concept):
            return [{"id": i, "file_name": images[i]["file_name"], "split": images[i]["split"],
                     "caption": pick_caption(images[i], concept)} for i in ids]

        manifest["pairs"][key] = {
            "source": source, "target": target,
            "clean_source": rows(clean_src, source),
            "poison_pool": rows(poison, source),
            "target_anchor": rows(anchor, target),
            "eval_prompts": [pick_caption(images[i], source) for i in evalp],
        }
        manifest["clean_train"] += rows(clean_src, source)

    # Filler: images with no pair concept at all, to make the corpus look like a real scrape.
    pair_concepts = set(SOURCES) | set(TARGETS)
    filler_pool = [iid for iid, rec in images.items()
                   if iid not in used and rec["captions"] and not (rec["categories"] & pair_concepts)]
    rng.shuffle(filler_pool)
    for iid in filler_pool[:N_FILLER]:
        manifest["filler"].append({"id": iid, "file_name": images[iid]["file_name"],
                                   "split": images[iid]["split"], "caption": images[iid]["captions"][0]})
    manifest["clean_train"] += manifest["filler"]

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"\nclean_train: {len(manifest['clean_train'])} images "
          f"({len(manifest['filler'])} filler + {sum(len(p['clean_source']) for p in manifest['pairs'].values())} source)")
    for key, p in manifest["pairs"].items():
        print(f"  {key:22s} poison_pool={len(p['poison_pool'])} anchor={len(p['target_anchor'])} "
              f"eval={len(p['eval_prompts'])}")
    print(f"wrote {args.out / 'manifest.json'}")


if __name__ == "__main__":
    main()
