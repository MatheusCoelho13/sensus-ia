#!/usr/bin/env python3
"""
Filter COCO annotations to a subset of classes and produce a smaller COCO-style dataset.

This script reads COCO annotation JSONs, keeps only the requested classes,
remaps category ids to a compact range, removes images without annotations,
and writes the filtered annotations to `out_dir/annotations` while creating
symlinks for the referenced images in `out_dir/images/{train2017,val2017}`.

Usage example:
  python3 scripts/filter_coco_by_classes.py --coco-dir data/coco --out-dir data/coco_filtered --classes person,backpack,chair,laptop

"""
import argparse
import json
import os
from pathlib import Path
from shutil import copy2
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def load_json(p: Path):
    with p.open('r', encoding='utf-8') as f:
        return json.load(f)


def write_json(obj, p: Path):
    with p.open('w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False)


def filter_split(coco_json_path: Path, classes_keep: set):
    coco = load_json(coco_json_path)

    # map original category id -> name
    orig_id2name = {c['id']: c['name'] for c in coco.get('categories', [])}

    # determine which original ids keep (by name matching)
    keep_orig_ids = {cid for cid, name in orig_id2name.items() if name in classes_keep}

    # filter annotations
    anns = [a for a in coco.get('annotations', []) if a.get('category_id') in keep_orig_ids]

    # get image ids that remain
    img_ids = {a['image_id'] for a in anns}

    images = [img for img in coco.get('images', []) if img['id'] in img_ids]

    # build new category list remapped to contiguous ids starting at 1
    kept_names = sorted(list(classes_keep))
    new_categories = []
    new_id_map = {}
    for i, nm in enumerate(kept_names, start=1):
        new_categories.append({'id': i, 'name': nm, 'supercategory': ''})
        new_id_map[nm] = i

    # remap annotation category ids
    name_by_orig = {cid: name for cid, name in orig_id2name.items()}
    new_annotations = []
    ann_id = 1
    for a in anns:
        orig_cid = a['category_id']
        name = name_by_orig.get(orig_cid)
        if name not in new_id_map:
            continue
        na = dict(a)
        na['category_id'] = new_id_map[name]
        na['id'] = ann_id
        ann_id += 1
        new_annotations.append(na)

    out = {
        'images': images,
        'annotations': new_annotations,
        'categories': new_categories,
    }
    return out, images


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--coco-dir', default='data/coco', help='COCO root dir (images/ and annotations/)')
    p.add_argument('--out-dir', default='data/coco_filtered', help='Output filtered COCO root')
    p.add_argument('--classes', required=True, help='Comma-separated class names to keep (e.g. person,backpack,chair,laptop)')
    p.add_argument('--symlink', action='store_true', help='Symlink images instead of copying')
    p.add_argument('--train-ann', default='annotations/instances_train2017.json')
    p.add_argument('--val-ann', default='annotations/instances_val2017.json')
    args = p.parse_args()

    classes_keep = {c.strip() for c in args.classes.split(',') if c.strip()}
    coco_dir = Path(args.coco_dir)
    out_dir = Path(args.out_dir)

    ensure_dir(out_dir / 'annotations')
    ensure_dir(out_dir / 'images/train2017')
    ensure_dir(out_dir / 'images/val2017')

    # process train
    train_ann_path = coco_dir / args.train_ann
    if train_ann_path.exists():
        filtered, imgs = filter_split(train_ann_path, classes_keep)
        write_json(filtered, out_dir / 'annotations' / Path(args.train_ann).name)
        # link/copy images
        src_images_dir = coco_dir / 'images' / 'train2017'
        for im in tqdm(filtered['images'], desc='link train images'):
            src = src_images_dir / im['file_name']
            dst = out_dir / 'images' / 'train2017' / im['file_name']
            if not dst.exists():
                try:
                    if args.symlink:
                        os.symlink(os.path.abspath(src), os.path.abspath(dst))
                    else:
                        copy2(src, dst)
                except Exception:
                    pass
    else:
        print('train annotation not found at', train_ann_path)

    # process val
    val_ann_path = coco_dir / args.val_ann
    if val_ann_path.exists():
        filtered, imgs = filter_split(val_ann_path, classes_keep)
        write_json(filtered, out_dir / 'annotations' / Path(args.val_ann).name)
        src_images_dir = coco_dir / 'images' / 'val2017'
        for im in tqdm(filtered['images'], desc='link val images'):
            src = src_images_dir / im['file_name']
            dst = out_dir / 'images' / 'val2017' / im['file_name']
            if not dst.exists():
                try:
                    if args.symlink:
                        os.symlink(os.path.abspath(src), os.path.abspath(dst))
                    else:
                        copy2(src, dst)
                except Exception:
                    pass
    else:
        print('val annotation not found at', val_ann_path)

    print('Filtered COCO written to', out_dir)


if __name__ == '__main__':
    main()
