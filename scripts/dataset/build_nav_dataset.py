#!/usr/bin/env python3
"""Build a YOLO dataset from the local COCO export.

This project already has two raw assets:
- `datasets/data/` with the images
- `datasets/labels.json` with the COCO annotations

The script converts those assets into a YOLO directory layout:
- `datasets/coco/images/train`
- `datasets/coco/images/val`
- `datasets/coco/labels/train`
- `datasets/coco/labels/val`

It keeps only the classes used by the project and writes a matching `data.yaml`.
"""

import argparse
import json
import random
from pathlib import Path
import shutil


DEFAULT_CLASSES = [
    'person',
    'backpack',
    'chair',
    'bench',
    'laptop',
    'cell phone',
    'bottle',
    'book',
    'cup',
    'tv',
]


def load_json(path: Path):
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def split_name(image_id: int, seed: int, val_ratio: float) -> str:
    rng = random.Random(seed + image_id)
    return 'val' if rng.random() < val_ratio else 'train'


def build_dataset(src_root: Path, annotations_path: Path, out_root: Path, classes, val_ratio: float, seed: int, symlink: bool):
    data = load_json(annotations_path)

    class_to_idx = {name: idx for idx, name in enumerate(classes)}
    category_by_id = {cat['id']: cat['name'] for cat in data.get('categories', [])}

    images = {img['id']: img for img in data.get('images', [])}
    annotations_by_image = {}
    for ann in data.get('annotations', []):
        class_name = category_by_id.get(ann.get('category_id'))
        if class_name not in class_to_idx:
            continue
        annotations_by_image.setdefault(ann['image_id'], []).append(ann)

    for sub in ('images/train', 'images/val', 'labels/train', 'labels/val'):
        ensure_dir(out_root / sub)

    kept_images = 0
    kept_annotations = 0

    for image_id, image in images.items():
        anns = annotations_by_image.get(image_id, [])
        if not anns:
            continue

        split = split_name(image_id, seed=seed, val_ratio=val_ratio)
        file_name = image.get('file_name')
        src_img = src_root / file_name
        if not src_img.exists():
            continue

        dst_img = out_root / 'images' / split / file_name
        dst_lbl = out_root / 'labels' / split / (Path(file_name).stem + '.txt')

        if not dst_img.exists():
            if symlink:
                try:
                    dst_img.symlink_to(src_img.resolve())
                except Exception:
                    shutil.copy2(src_img, dst_img)
            else:
                shutil.copy2(src_img, dst_img)

        width = image.get('width')
        height = image.get('height')
        lines = []
        for ann in anns:
            bbox = ann.get('bbox')
            if not bbox or width in (None, 0) or height in (None, 0):
                continue
            class_name = category_by_id.get(ann.get('category_id'))
            cls_idx = class_to_idx.get(class_name)
            if cls_idx is None:
                continue
            x, y, w, h = bbox
            xc = (x + w / 2.0) / width
            yc = (y + h / 2.0) / height
            w /= width
            h /= height
            lines.append(f'{cls_idx} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n')
            kept_annotations += 1

        dst_lbl.write_text(''.join(lines), encoding='utf-8')
        kept_images += 1

    data_yaml = out_root / 'data.yaml'
    data_yaml.write_text(
        '\n'.join([
            f'path: {out_root.resolve()}',
            'train: images/train',
            'val: images/val',
            f'nc: {len(classes)}',
            'names:',
            '  [' + ', '.join(f"\'{c}\'" for c in classes) + ']',
            '',
        ]),
        encoding='utf-8',
    )

    (out_root / 'data.names').write_text('\n'.join(classes), encoding='utf-8')

    print(f'✅ Dataset pronto em: {out_root}')
    print(f'   imagens processadas: {kept_images}')
    print(f'   anotações processadas: {kept_annotations}')
    print(f'   data.yaml: {data_yaml}')


def main():
    parser = argparse.ArgumentParser(description='Build local YOLO dataset from COCO JSON + image cache')
    parser.add_argument('--src-root', default='datasets/data', help='Pasta com as imagens JPG')
    parser.add_argument('--annotations', default='datasets/labels.json', help='Arquivo COCO JSON local')
    parser.add_argument('--out-root', default='datasets/coco', help='Saída YOLO')
    parser.add_argument('--val-ratio', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--copy', action='store_true', help='Copiar em vez de criar symlink')
    parser.add_argument('--classes', default=','.join(DEFAULT_CLASSES), help='Classes separadas por vírgula')
    args = parser.parse_args()

    classes = [c.strip() for c in args.classes.split(',') if c.strip()]
    build_dataset(
        src_root=Path(args.src_root),
        annotations_path=Path(args.annotations),
        out_root=Path(args.out_root),
        classes=classes,
        val_ratio=args.val_ratio,
        seed=args.seed,
        symlink=not args.copy,
    )


if __name__ == '__main__':
    main()