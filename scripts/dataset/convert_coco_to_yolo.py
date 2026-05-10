#!/usr/bin/env python3
"""
Convert COCO annotations to YOLOv5/YOLOv8 text format.

Assumes COCO layout:
  data/coco/images/train2017
  data/coco/images/val2017
  data/coco/annotations/instances_train2017.json
  data/coco/annotations/instances_val2017.json

Outputs to `datasets/coco/` by default with structure:
  datasets/coco/images/train
  datasets/coco/images/val
  datasets/coco/labels/train
  datasets/coco/labels/val

This script copies (or symlinks) images and writes .txt label files.
"""
import argparse
import json
import os
import shutil
from pathlib import Path
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def load_coco_annotations(ann_file):
    with open(ann_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_cat_mapping(categories):
    # Map original COCO category id -> new contiguous class index (0..nc-1)
    cats = sorted(categories, key=lambda c: c['id'])
    id2idx = {c['id']: i for i, c in enumerate(cats)}
    names = [c['name'] for c in cats]
    return id2idx, names


def convert_annotations(coco, images_dir, out_labels_dir, id2idx):
    ensure_dir(out_labels_dir)

    # Build image_id -> (file_name, width, height)
    img_map = {img['id']: img for img in coco['images']}

    # Group annotations by image_id
    ann_map = {}
    for ann in coco['annotations']:
        img_id = ann['image_id']
        ann_map.setdefault(img_id, []).append(ann)

    for img_id, img in tqdm(img_map.items(), desc=f'writing labels to {out_labels_dir}'):
        file_name = img['file_name']
        width = img.get('width')
        height = img.get('height')
        anns = ann_map.get(img_id, [])

        label_lines = []
        for a in anns:
            # bbox is [x,y,width,height] in COCO
            bbox = a['bbox']
            x, y, w, h = bbox
            if width is None or height is None or width == 0 or height == 0:
                continue
            xc = x + w / 2.0
            yc = y + h / 2.0
            xc /= width
            yc /= height
            w /= width
            h /= height
            cat_id = a['category_id']
            cls = id2idx.get(cat_id)
            if cls is None:
                continue
            label_lines.append(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

        # write label file
        out_label_file = out_labels_dir / (Path(file_name).stem + '.txt')
        if label_lines:
            out_label_file.write_text(''.join(label_lines))
        else:
            # write empty file for images without annotations (optional)
            out_label_file.write_text('')


def copy_images(image_list, src_dir: Path, dst_dir: Path, symlink=True):
    ensure_dir(dst_dir)
    for img in tqdm(image_list, desc=f'linking images to {dst_dir}'):
        src = src_dir / img['file_name']
        dst = dst_dir / img['file_name']
        if dst.exists():
            continue
        try:
            if symlink:
                os.symlink(os.path.abspath(src), os.path.abspath(dst))
            else:
                shutil.copy2(src, dst)
        except Exception:
            # fallback to copy
            try:
                shutil.copy2(src, dst)
            except Exception:
                pass


def main():
    p = argparse.ArgumentParser(description='Convert COCO annotations to YOLO format')
    p.add_argument('--coco-dir', default='data/coco', help='Path to COCO root (images/ and annotations/ subfolders)')
    p.add_argument('--out-dir', default='datasets/coco', help='Output dataset root')
    p.add_argument('--symlink', action='store_true', help='Symlink images instead of copying (default: copy)')
    p.add_argument('--split', choices=['both', 'train', 'val'], default='both')
    p.add_argument('--train-ann', default='annotations/instances_train2017.json')
    p.add_argument('--val-ann', default='annotations/instances_val2017.json')
    args = p.parse_args()

    coco_dir = Path(args.coco_dir)
    out_dir = Path(args.out_dir)

    # prepare output dirs
    for sub in ['images/train', 'images/val', 'labels/train', 'labels/val']:
        ensure_dir(out_dir / sub)

    if args.split in ('both', 'train'):
        ann_file = coco_dir / args.train_ann
        if ann_file.exists():
            coco = load_coco_annotations(ann_file)
            id2idx, names = build_cat_mapping(coco['categories'])
            # write names file
            (out_dir / 'data.names').write_text('\n'.join(names))
            convert_annotations(coco, coco_dir / 'images' / 'train2017', out_dir / 'labels' / 'train', id2idx)
            copy_images(coco['images'], coco_dir / 'images' / 'train2017', out_dir / 'images' / 'train', symlink=args.symlink)
        else:
            print(f"train annotation not found: {ann_file}")

    if args.split in ('both', 'val'):
        ann_file = coco_dir / args.val_ann
        if ann_file.exists():
            coco = load_coco_annotations(ann_file)
            # reuse category mapping if already written
            if (out_dir / 'data.names').exists():
                names = (out_dir / 'data.names').read_text().splitlines()
                # build reverse mapping from names to class idx
                id2idx = {c['id']: i for i, c in enumerate(coco['categories'])}
            else:
                id2idx, names = build_cat_mapping(coco['categories'])
                (out_dir / 'data.names').write_text('\n'.join(names))
            convert_annotations(coco, coco_dir / 'images' / 'val2017', out_dir / 'labels' / 'val', id2idx)
            copy_images(coco['images'], coco_dir / 'images' / 'val2017', out_dir / 'images' / 'val', symlink=args.symlink)
        else:
            print(f"val annotation not found: {ann_file}")

    # write a sample data.yaml for ultralytics
    data_yaml = out_dir / 'data_coco.yaml'
    names_file = out_dir / 'data.names'
    if names_file.exists():
        names = names_file.read_text().splitlines()
        nc = len(names)
    else:
        # no categories found; create empty names file and set nc=0
        names = []
        nc = 0
        try:
            names_file.write_text('')
        except Exception:
            pass

    data_yaml.write_text('\n'.join([
        f"path: {str(out_dir)}",
        "train: images/train",
        "val: images/val",
        f"nc: {nc}",
        "names: []  # see data.names file",
    ]))

    print('Conversion complete. See', out_dir)


if __name__ == '__main__':
    main()
