"""
Builds the exact list of YouTube videos we need to download: the
intersection of (COIN.json / YouCook2 trainval annotations) with the video
IDs actually referenced by VANS-DATA_COIN.csv / VANS-DATA_YouCook.csv.
Already verified: 9923/9924 COIN ids and 1530/1531 YouCook2 ids match
exactly (the one miss each time is a stray "#NAME?" Excel-corruption row in
the CSV, not a real video).
"""
import csv
import json
import os

BASE = os.environ.get("VANS_ROOT", "/projects/bhay/william/ruixin/vans_world_model")

with open(os.path.join(BASE, "raw_data/COIN.json")) as f:
    coin_db = json.load(f)["database"]
with open(os.path.join(BASE, "raw_data/youcookii_annotations_trainval.json")) as f:
    yc_db = json.load(f)["database"]


def wanted_ids(csv_path):
    ids = set()
    with open(csv_path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            n = row["name"].strip()
            if n and n != "#NAME?":
                ids.add(n)
    return ids


coin_wanted = wanted_ids(os.path.join(BASE, "hf_data/VANS-DATA_COIN.csv"))
yc_wanted = wanted_ids(os.path.join(BASE, "hf_data/VANS-DATA_YouCook.csv"))

coin_targets = []
for vid in sorted(coin_wanted):
    if vid in coin_db:
        coin_targets.append({
            "video_id": vid,
            "dataset": "coin",
            "recipe_type": coin_db[vid]["class"],
            "url": f"https://www.youtube.com/watch?v={vid}",
        })

yc_targets = []
for vid in sorted(yc_wanted):
    if vid in yc_db:
        yc_targets.append({
            "video_id": vid,
            "dataset": "youcook2",
            "recipe_type": str(yc_db[vid]["recipe_type"]),
            "url": f"https://www.youtube.com/watch?v={vid}",
        })

print(f"[INFO] coin targets: {len(coin_targets)}  (missing: {len(coin_wanted) - len(coin_targets)})")
print(f"[INFO] youcook2 targets: {len(yc_targets)}  (missing: {len(yc_wanted) - len(yc_targets)})")

all_targets = coin_targets + yc_targets
with open(os.path.join(BASE, "raw_data/download_targets.json"), "w") as f:
    json.dump(all_targets, f, indent=2)
print(f"[INFO] total targets: {len(all_targets)} -> raw_data/download_targets.json")
