#!/usr/bin/env python3
"""
Filter a YOLO-format dataset (like coco128) keeping only specified COCO class ids
and remap them to a compact class range suitable for training.

Example:
  python3 scripts/filter_yolo_by_coco_ids.py --src data/coco128 --out datasets/coco --classes person,backpack,chair,bench,laptop

This will copy images and create labels under `out/images/{train,val}` and `out/labels/{train,val}`
with classes remapped to 0..(n-1) in the order provided.
"""
import argparse
import os
from pathlib import Path
from shutil import copy2

# Standard 80 COCO class names in the YOLO ordering (indices 0..79)
COCO_NAMES = [
    'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light',
    'fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow',
    'elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee',
    'skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket',
    'bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange','broccoli','carrot',
    'hot dog','pizza','donut','cake','chair','couch','potted plant','bed','dining table','toilet','tv','laptop','mouse',
    'remote','keyboard','cell phone','microwave','oven','toaster','sink','refrigerator','book','clock','vase','scissors',
    'teddy bear','hair drier','toothbrush'
]


def load_coco_name_map(src_root: Path):
    # prefer src/data.names or src/data.yaml if present, else fallback to COCO_NAMES
    names_path = src_root / 'data.names'
    if names_path.exists():
        names = [l.strip() for l in names_path.read_text(encoding='utf-8').splitlines() if l.strip()]
        return {n: i for i, n in enumerate(names)}
    yaml_path = src_root / 'data.yaml'
    if yaml_path.exists():
        try:
            import yaml
            doc = yaml.safe_load(yaml_path.read_text(encoding='utf-8'))
            names = doc.get('names') if isinstance(doc.get('names'), list) else []
            if names:
                return {n: i for i, n in enumerate(names)}
        except Exception:
            pass
    # fallback
    return {n: i for i, n in enumerate(COCO_NAMES)}


# synonyms and Portuguese names mapping to canonical English COCO names
SYNONYMS = {
    'celular': 'cell phone',
    'cellphone': 'cell phone',
    'cell_phone': 'cell phone',
    'mochila': 'backpack',
    'cadeira': 'chair',
    'banco': 'bench',
    'pessoa': 'person',
    'livro': 'book',
    'garrafa': 'bottle',
    'copo': 'cup',
    'televisao': 'tv',
    'televisão': 'tv',
    'tv': 'tv',
    'teclado': 'keyboard',
    'mouse': 'mouse',
    'relogio': 'clock',
    'relógio': 'clock',
}


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def process_split(src_root: Path, split: str, out_root: Path, keep_ids: list, remap: dict):
    src_img_dir = src_root / 'images' / split
    src_lbl_dir = src_root / 'labels' / split
    out_img_dir = out_root / 'images' / ('train' if 'train' in split else 'val')
    out_lbl_dir = out_root / 'labels' / ('train' if 'train' in split else 'val')
    ensure_dir(out_img_dir)
    ensure_dir(out_lbl_dir)

    if not src_lbl_dir.exists():
        return

    for lbl_path in src_lbl_dir.glob('*.txt'):
        txt = lbl_path.read_text().strip().splitlines()
        keep_lines = []
        for line in txt:
            if not line.strip():
                continue
            parts = line.split()
            cls = int(parts[0])
            if cls in keep_ids:
                new_cls = remap[cls]
                keep_lines.append(' '.join([str(new_cls)] + parts[1:]))

        if keep_lines:
            img_name = lbl_path.stem + '.jpg'
            src_img = src_img_dir / img_name
            dst_img = out_img_dir / img_name
            try:
                if not dst_img.exists():
                    copy2(src_img, dst_img)
            except Exception:
                continue
            out_lbl = out_lbl_dir / (lbl_path.name)
            out_lbl.write_text('\n'.join(keep_lines) + '\n')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--src', default='data/coco128', help='Source YOLO-format dataset root (images/, labels/)')
    p.add_argument('--out', default='datasets/coco', help='Output dataset root')
    p.add_argument('--classes', default='person,backpack,chair,bench,laptop', help='Comma-separated class names (COCO names)')
    args = p.parse_args()

    classes = [c.strip() for c in args.classes.split(',') if c.strip()]
    src_root = Path(args.src)
    # build name->index map from source dataset (data.names or data.yaml) or fallback
    name_map = load_coco_name_map(src_root)

    keep_ids = []
    remap = {}
    for i, raw_name in enumerate(classes):
        name = raw_name.strip()
        key = name.lower()
        # apply synonyms
        if key in SYNONYMS:
            key = SYNONYMS[key]
        # try direct match
        cid = name_map.get(key)
        # try some normalization variants
        if cid is None:
            alt = key.replace(' ', '_')
            cid = name_map.get(alt)
        if cid is None:
            # try title-case search
            for nm, idx in name_map.items():
                if nm.lower() == key or nm.lower() == key.replace('_', ' '):
                    cid = idx
                    break
        if cid is None:
            raise SystemExit(f'unknown COCO class name: {name} (looked up as "{key}")')
        keep_ids.append(cid)
        remap[cid] = i
    out_root = Path(args.out)
    ensure_dir(out_root)
    ensure_dir(out_root / 'images' / 'train')
    ensure_dir(out_root / 'images' / 'val')
    ensure_dir(out_root / 'labels' / 'train')
    ensure_dir(out_root / 'labels' / 'val')

    # process expected splits
    for split in ['train2017', 'val2017']:
        process_split(src_root, split, out_root, keep_ids, remap)

    # write data yaml
    names_list = ','.join([f"'{c}'" for c in classes])
    (out_root / 'data_coco.yaml').write_text('\n'.join([
        f"path: {str(out_root)}",
        "train: images/train",
        "val: images/val",
        f"nc: {len(classes)}",
        f"names: [{names_list}]",
    ]))

    # also write data.names (one-per-line) for downstream tools
    try:
        (out_root / 'data.names').write_text('\n'.join(classes) + '\n')
    except Exception:
        pass

    print('Filtered YOLO dataset created at', out_root)


if __name__ == '__main__':
    main()
