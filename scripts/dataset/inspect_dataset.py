#!/usr/bin/env python3
"""Inspect a YOLO-format dataset and print counts and samples.

Usage: python3 scripts/inspect_dataset.py --data datasets/coco --samples 3
"""
import argparse
from pathlib import Path
from collections import Counter, defaultdict


def read_labels(lbl_path):
    try:
        txt = lbl_path.read_text().strip().splitlines()
    except Exception:
        return []
    out = []
    for l in txt:
        if not l.strip():
            continue
        parts = l.split()
        cls = int(parts[0])
        out.append((cls, parts[1:]))
    return out


def inspect(data_root: Path, samples=3):
    images_train = list((data_root / 'images' / 'train').glob('*')) if (data_root / 'images' / 'train').exists() else []
    images_val = list((data_root / 'images' / 'val').glob('*')) if (data_root / 'images' / 'val').exists() else []
    labels_train = list((data_root / 'labels' / 'train').glob('*.txt')) if (data_root / 'labels' / 'train').exists() else []
    labels_val = list((data_root / 'labels' / 'val').glob('*.txt')) if (data_root / 'labels' / 'val').exists() else []

    print('Dataset root:', data_root)
    print('images/train:', len(images_train), 'images')
    print('images/val:  ', len(images_val), 'images')
    print('labels/train:', len(labels_train), 'files')
    print('labels/val:  ', len(labels_val), 'files')

    # per-class counts
    counts = Counter()
    files_by_class = defaultdict(list)
    for f in labels_train + labels_val:
        anns = read_labels(f)
        for cls, _ in anns:
            counts[cls] += 1
            if len(files_by_class[cls]) < samples:
                files_by_class[cls].append(f)

    if counts:
        print('\nPer-class annotation counts:')
        for cls, c in sorted(counts.items()):
            print(f'  class {cls}: {c} annotations, samples: {len(files_by_class[cls])}')
    else:
        print('\nNo annotations found in labels.')

    print('\nSample label files and contents:')
    for cls, files in files_by_class.items():
        print(f'\n== Class {cls} samples ==')
        for f in files:
            try:
                display_path = f.relative_to(Path.cwd())
            except Exception:
                display_path = f
            print('-', display_path)
            lines = read_labels(f)
            for ln in lines[:5]:
                print('   ', ln)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='datasets/coco', help='YOLO dataset root (images/ labels/)')
    p.add_argument('--samples', type=int, default=3, help='samples per class')
    args = p.parse_args()
    inspect(Path(args.data), samples=args.samples)


if __name__ == '__main__':
    main()
