#!/usr/bin/env python3
"""Enriquecer dataset: detectar classes com poucas imagens, buscar URLs no Wikimedia e baixar imagens.

Fluxo:
 1. Conta imagens por classe em datasets/images/(train,val)
 2. Para classes com menos de --target-per-class, chama fetch_wikimedia_urls.py para obter URLs
 3. Chama download_images.py para baixar as imagens (respeitando per-class)
 4. Opcional: regenerar labels via YOLO inferência (`--regen-labels`)

Exemplo:
  python scripts/enrich_dataset.py --target-per-class 200 --contact you@example.com --regen-labels --model-for-labels yolov8n.pt
"""
import argparse
import subprocess
from pathlib import Path
import shutil
import sys
import yaml


def count_images(dataset_dir: Path):
    counts = {}
    for split in ('train', 'val'):
        base = dataset_dir / 'images' / split
        if not base.exists():
            continue
        for cls_dir in base.iterdir():
            if not cls_dir.is_dir():
                continue
            cnt = sum(1 for _ in cls_dir.iterdir() if _.is_file())
            counts[cls_dir.name] = counts.get(cls_dir.name, 0) + cnt
    return counts


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--target-per-class', type=int, default=200)
    p.add_argument('--classes', nargs='*', help='Lista de classes a processar (por padrão pega do config/data.yaml)')
    p.add_argument('--contact', default='')
    p.add_argument('--out-urls', default='data/urls/urls_enrich.txt')
    p.add_argument('--max-per-class', type=int, default=200)
    p.add_argument('--regen-labels', action='store_true', help='Executar inferência para gerar labels txt nas novas imagens')
    p.add_argument('--model-for-labels', default='yolov8n.pt')
    p.add_argument('--dataset', default='datasets')
    args = p.parse_args()

    project_root = Path(__file__).parent.parent
    dataset_dir = project_root / args.dataset

    # get classes list
    classes = args.classes
    if not classes:
        # try read config/data.yaml classes or nc
        cfg = Path('config/data.yaml')
        if cfg.exists():
            try:
                data = yaml.safe_load(cfg.read_text())
                if 'names' in data:
                    classes = list(data['names'].values()) if isinstance(data['names'], dict) else list(data['names'])
                elif 'nc' in data:
                    # fallback to default known classes
                    classes = ['pessoa','cadeira','mesa','porta','parede']
            except Exception:
                classes = ['pessoa','cadeira','mesa','porta','parede']
        else:
            classes = ['pessoa','cadeira','mesa','porta','parede']

    counts = count_images(dataset_dir)
    print('Contagem atual por classe:')
    for c in classes:
        print(f' - {c}: {counts.get(c,0)}')

    # build a per-class fetch list for classes needing images
    to_fetch = {}
    for c in classes:
        have = counts.get(c, 0)
        if have < args.target_per_class:
            need = args.target_per_class - have
            to_fetch[c] = min(need, args.max_per_class)

    if not to_fetch:
        print('Todas as classes já atingiram o target-per-class. Nada a fazer.')
        return

    print('\nClasses que serão enriquecidas:')
    for c, n in to_fetch.items():
        print(f' - {c}: +{n}')

    # call fetch_wikimedia_urls.py to generate URLs
    out_urls = Path(args.out_urls)
    out_urls.parent.mkdir(parents=True, exist_ok=True)
    fetch_cmd = [sys.executable, 'scripts/fetch_wikimedia_urls.py', '--out', str(out_urls), '--count', '0']
    # fetch_wikimedia_urls accepts --classes and --count per class, but not per-class counts; we'll call per class
    lines = []
    for c, n in to_fetch.items():
        print(f'Buscando {n} URLs para {c}...')
        cmd = [sys.executable, 'scripts/fetch_wikimedia_urls.py', '--out', str(out_urls.parent / f'urls_{c}.txt'), '--count', str(n), '--classes', c]
        subprocess.run(cmd, check=False)
        # append generated file lines to master
        f = out_urls.parent / f'urls_{c}.txt'
        if f.exists():
            for ln in f.read_text(encoding='utf-8').splitlines():
                lines.append(ln)
            # optional: remove temp
            f.unlink()

    # write master urls file
    out_urls.write_text('\n'.join(lines), encoding='utf-8')
    print(f'URLs coletadas em: {out_urls} ({len(lines)} linhas)')

    # call download_images.py
    dl_cmd = [sys.executable, 'scripts/download_images.py', str(out_urls), str(dataset_dir), '--per-class', str(args.max_per_class), '--contact', args.contact]
    print('Iniciando download de imagens (pode demorar)...')
    subprocess.run(dl_cmd, check=False)

    if args.regen_labels:
        print('Regenerando labels via inferência do modelo:', args.model_for_labels)
        try:
            from ultralytics import YOLO
            model = YOLO(args.model_for_labels)
            # run predict on new images (train & val)
            src = str(dataset_dir / 'images')
            model.predict(source=src, save_txt=True, device='cpu')
            print('Labels regenerados (salvo em runs/predict/...). Verifique e mova para datasets/labels se necessário.')
        except Exception as e:
            print('Falha ao regenerar labels:', e)

    print('Enriquecimento concluído.')


if __name__ == '__main__':
    main()
