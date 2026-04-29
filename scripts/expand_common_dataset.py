#!/usr/bin/env python3
"""Orquestra pipeline para aumentar dataset com classes comuns.

Fluxo padrão:
 1. Filtra COCO por classes (usa scripts/filter_coco_by_classes.py)
 2. Converte COCO filtrado para YOLO (scripts/convert_coco_to_yolo.py)
 3. Opcional: enriquece baixando imagens do Wikimedia e gerando labels (scripts/enrich_dataset.py)

Exemplo:
  python scripts/expand_common_dataset.py --classes pessoa,cadeira,mesa --enrich --target-per-class 200
"""
import argparse
import subprocess
import sys
from pathlib import Path


PT_TO_EN = {
    'pessoa': 'person',
    'cadeira': 'chair',
    'mesa': 'table',
    'porta': 'door',
    'parede': 'wall',
    'bolsa': 'backpack',
    'mochila': 'backpack',
    'laptop': 'laptop',
    'celular': 'cell phone',
    'garrafa': 'bottle',
    'livro': 'book',
    'copo': 'cup',
    'tv': 'tv',
    'televisao': 'tv',
}


def map_classes(classes):
    out = []
    for c in classes:
        c = c.strip()
        if not c:
            continue
        # accept English names directly
        if c in PT_TO_EN.values():
            out.append(c)
        else:
            out.append(PT_TO_EN.get(c, c))
    # dedupe preserving order
    seen = set()
    res = []
    for x in out:
        if x not in seen:
            seen.add(x)
            res.append(x)
    return res


def run(cmd, check=True):
    print('>',' '.join(cmd))
    return subprocess.run(cmd, check=check)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--classes', default='pessoa,cadeira,mesa,porta,parede,celular,garrafa,livro,copo,tv', help='Comma-separated classes (pt or en)')
    p.add_argument('--coco-dir', default='data/coco')
    p.add_argument('--filtered-out', default='data/coco_filtered')
    p.add_argument('--yolo-out', default='datasets/coco')
    p.add_argument('--enrich', action='store_true', help='Buscar imagens adicionais no Wikimedia e baixar para datasets/')
    p.add_argument('--target-per-class', type=int, default=200)
    p.add_argument('--regen-labels', action='store_true', help='Regenerar labels via inferência após download')
    args = p.parse_args()

    classes = [c.strip() for c in args.classes.split(',') if c.strip()]
    mapped = map_classes(classes)
    classes_str = ','.join(mapped)

    py = sys.executable

    # 1) filter COCO
    cmd1 = [py, 'scripts/filter_coco_by_classes.py', '--coco-dir', args.coco_dir, '--out-dir', args.filtered_out, '--classes', classes_str, '--symlink']
    run(cmd1)

    # 2) convert to YOLO
    cmd2 = [py, 'scripts/convert_coco_to_yolo.py', '--coco-dir', args.filtered_out, '--out-dir', args.yolo_out, '--symlink']
    run(cmd2)

    # 3) optional enrichment (fetch + download + optional regen labels)
    if args.enrich:
        # call enrich_dataset.py with target-per-class and regen flag if needed
        cmd3 = [py, 'scripts/enrich_dataset.py', '--target-per-class', str(args.target_per_class), '--contact', '']
        if args.regen_labels:
            cmd3 += ['--regen-labels', '--model-for-labels', 'yolov8n.pt']
        run(cmd3)

    print('\n✅ Pipeline concluído. Verifique as pastas:')
    print(' - COCO filtrado:', args.filtered_out)
    print(' - YOLO dataset:', args.yolo_out)


if __name__ == '__main__':
    main()
