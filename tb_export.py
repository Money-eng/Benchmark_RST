
#!/usr/bin/env python3
"""
tb_export.py — Export TensorBoard event files to CSV (scalars) and PNGs (images).

Usage examples:

  # Basic: export everything found under logs/ to out/
  python tb_export.py --logdir logs --out out

  # Only scalars to a single combined CSV
  python tb_export.py --logdir runs --out export --scalars --images 0 --one-csv

  # Only images for a specific tag pattern
  python tb_export.py --logdir logs --out imgs --scalars 0 --images --tag ".*/images/.*"

Notes:
- Requires: tensorboard (or tensorflow) installed: `pip install tensorboard`
- Works with classic event files: events.out.tfevents.*
"""

import argparse
import csv
import os
import re
from pathlib import Path
from datetime import datetime

# TensorBoard event reader
try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except Exception as e:
    raise SystemExit(
        "Missing dependency. Please install tensorboard:\n\n  pip install tensorboard\n\n"
        f"Original import error: {e}"
    )

SIZE_GUIDANCE = {
    "compressedHistograms": 0,
    "histograms": 0,
    "images": 1000,   # adjust if you have many images
    "scalars": 0,
    "tensors": 0,
    "audio": 0,
}

def human_time(ts: float) -> str:
    try:
        return datetime.fromtimestamp(ts).isoformat(timespec="seconds")
    except Exception:
        return ""

def sanitize(s: str) -> str:
    return re.sub(r'[^A-Za-z0-9._\-]+', '_', s)

def iter_event_files(root: Path):
    for p in root.rglob("*"):
        if p.is_file() and p.name.startswith("events.out.tfevents."):
            yield p

def load_run(event_file: Path) -> EventAccumulator:
    ea = EventAccumulator(str(event_file), size_guidance=SIZE_GUIDANCE)
    ea.Reload()
    return ea

def export_scalars(ea: EventAccumulator, run_name: str, out_dir: Path, writer_all=None, tag_re=None):
    tags = ea.Tags().get("scalars", [])
    if tag_re:
        tags = [t for t in tags if tag_re.search(t)]
    if not tags:
        return 0, 0
    per_run_out = out_dir / "scalars" / sanitize(run_name)
    per_run_out.mkdir(parents=True, exist_ok=True)
    rows = 0
    tags_count = 0
    for tag in tags:
        events = ea.Scalars(tag)
        tags_count += 1
        csv_path = per_run_out / f"{sanitize(tag)}.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["run", "tag", "step", "wall_time", "wall_time_iso", "value"])
            for ev in events:
                rows += 1
                iso = human_time(ev.wall_time)
                w.writerow([run_name, tag, ev.step, f"{ev.wall_time:.6f}", iso, f"{ev.value:.10g}"])
                if writer_all:
                    writer_all.writerow([run_name, tag, ev.step, f"{ev.wall_time:.6f}", iso, f"{ev.value:.10g}"])
    return rows, tags_count

def export_images(ea: EventAccumulator, run_name: str, out_dir: Path, tag_re=None, limit_per_tag=None):
    tags = ea.Tags().get("images", [])
    if tag_re:
        tags = [t for t in tags if tag_re.search(t)]
    total = 0
    tags_count = 0
    if not tags:
        return 0, 0
    for tag in tags:
        tags_count += 1
        img_dir = out_dir / "images" / sanitize(run_name) / sanitize(tag)
        img_dir.mkdir(parents=True, exist_ok=True)
        images = ea.Images(tag)
        if limit_per_tag is not None:
            images = images[:limit_per_tag]
        for ev in images:
            fname = f"step{ev.step}_time{int(ev.wall_time)}.png"
            with open(img_dir / fname, "wb") as f:
                f.write(ev.encoded_image_string)
            total += 1
    return total, tags_count

def main():
    ap = argparse.ArgumentParser(description="Export TensorBoard event files to CSV (scalars) and PNGs (images).")
    ap.add_argument("--logdir", required=True, type=Path, help="Directory containing event files (recurses).")
    ap.add_argument("--out", required=True, type=Path, help="Output directory.")
    ap.add_argument("--scalars", type=int, default=1, help="Export scalars CSVs (1=yes, 0=no). Default: 1")
    ap.add_argument("--images", type=int, default=1, help="Export images PNGs (1=yes, 0=no). Default: 1")
    ap.add_argument("--one-csv", action="store_true", help="Also write a combined CSV of all scalar points.")
    ap.add_argument("--tag", type=str, default=None, help="Regex to filter tags (applies to both scalars and images).")
    ap.add_argument("--limit-per-tag", type=int, default=None, help="Max number of images to save per tag.")
    args = ap.parse_args()

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    tag_re = re.compile(args.tag) if args.tag else None

    all_csv_path = out_dir / "all_scalars.csv"
    writer_all = None
    all_file = None
    if args.scalars and args.one_csv:
        all_file = open(all_csv_path, "w", newline="")
        writer_all = csv.writer(all_file)
        writer_all.writerow(["run", "tag", "step", "wall_time", "wall_time_iso", "value"])

    event_files = list(iter_event_files(args.logdir))
    if not event_files:
        print(f"No TensorBoard event files found under: {args.logdir}")
        return

    total_scalar_rows = 0
    total_scalar_tags = 0
    total_images = 0
    total_image_tags = 0

    for ef in event_files:
        run_name = str(ef.parent.relative_to(args.logdir))
        print(f"Processing {ef} (run='{run_name}') ...")
        try:
            ea = load_run(ef)
        except Exception as e:
            print(f"  ! Skipping (failed to load): {e}")
            continue

        if args.scalars:
            rows, tags_count = export_scalars(ea, run_name, out_dir, writer_all=writer_all, tag_re=tag_re)
            total_scalar_rows += rows
            total_scalar_tags += tags_count

        if args.images:
            imgs, img_tags = export_images(ea, run_name, out_dir, tag_re=tag_re, limit_per_tag=args.limit_per_tag)
            total_images += imgs
            total_image_tags += img_tags

    if all_file:
        all_file.close()

    print("\nDone.")
    print(f"Scalar tags: {total_scalar_tags} — points exported: {total_scalar_rows}")
    print(f"Image tags:  {total_image_tags} — images exported: {total_images}")
    if args.scalars and args.one_csv:
        print(f"Combined CSV: {all_csv_path}")
    print(f"Output directory: {out_dir.resolve()}")

if __name__ == "__main__":
    main()
