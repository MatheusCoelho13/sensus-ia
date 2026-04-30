#!/usr/bin/env python3
"""Limpeza e correção de dataset YOLO.

Funcionalidades:
- Remove linhas de label com classe >= nc
- Remove imagens ou labels vazios/corrompidos
- (Opcional) Regenera labels faltantes usando um modelo de inferência (yolov8n.pt)
- Log detalhado em `logs/clean_dataset.log`

Uso:
  python3 scripts/clean_and_fix_dataset.py --dry-run
  python3 scripts/clean_and_fix_dataset.py --regen-missing --model models/yolov8n.pt
"""
import argparse
import logging
from pathlib import Path
import shutil
from PIL import Image, UnidentifiedImageError
import sys
import subprocess

try:
    import yaml
except Exception:
    yaml = None


LOG_PATH = Path('logs')
LOG_PATH.mkdir(parents=True, exist_ok=True)
logging.basicConfig(filename=LOG_PATH / 'clean_dataset.log', level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger('clean_dataset')


def load_data_yaml(path='config/data.yaml'):
    if yaml is None:
        raise RuntimeError('PyYAML não instalado. Rode: pip install pyyaml')
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f'{path} não encontrado')
    with p.open('r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def is_image_ok(img_path: Path):
    try:
        with Image.open(img_path) as im:
            im.verify()
        return True
    except Image.DecompressionBombError as e:
        logger.warning(f'Imagem muito grande (DecompressionBomb): {img_path} -> {e}')
        return False
    except (UnidentifiedImageError, OSError, ValueError) as e:
        logger.warning(f'Imagem corrompida: {img_path} -> {e}')
        return False


def clean_labels_and_images(dry_run=True, regen_missing=False, model='models/yolov8n.pt'):
    data = load_data_yaml()
    nc = int(data.get('nc', 0))
    logger.info(f'Iniciando limpeza (nc={nc}) dry_run={dry_run} regen_missing={regen_missing}')

    images_root = Path('datasets/images')
    labels_root = Path('datasets/labels')
    removed_dir = Path('datasets/removed')
    removed_dir.mkdir(parents=True, exist_ok=True)

    stats = {'labels_fixed': 0, 'labels_removed': 0, 'images_removed': 0, 'labels_generated': 0}

    # iterar por train/val e subpastas
    for split in ('train', 'val'):
        img_dir = images_root / split
        lbl_dir = labels_root / split
        if not img_dir.exists() or not lbl_dir.exists():
            logger.info(f'Pasta ausente: {img_dir} ou {lbl_dir} — pulando')
            continue

        for lbl_file in lbl_dir.rglob('*.txt'):
            try:
                rel = lbl_file.relative_to(lbl_dir)
            except Exception:
                rel = lbl_file.name
            img_file = img_dir / rel.with_suffix('.jpg')
            if not img_file.exists():
                img_file = img_dir / rel.with_suffix('.png')

            # ler labels
            lines = [l.strip() for l in lbl_file.read_text(encoding='utf-8').splitlines() if l.strip()]
            valid_lines = []
            removed_any = False
            for ln in lines:
                parts = ln.split()
                if len(parts) < 5:
                    logger.warning(f'Linha inválida (format): {lbl_file}:{ln}')
                    removed_any = True
                    continue
                try:
                    cls = int(parts[0])
                except ValueError:
                    logger.warning(f'Classe não numérica: {lbl_file}:{ln}')
                    removed_any = True
                    continue
                if cls < 0 or cls >= nc:
                    logger.info(f'Classe fora do range ({cls}) removida de {lbl_file}')
                    removed_any = True
                    continue
                valid_lines.append(ln)

            if removed_any:
                stats['labels_fixed'] += 1
                if dry_run:
                    logger.info(f'[DRY] Corrigido {lbl_file} (linhas: {len(lines)} -> {len(valid_lines)})')
                else:
                    if valid_lines:
                        lbl_file.write_text('\n'.join(valid_lines) + '\n', encoding='utf-8')
                        logger.info(f'Corrigido {lbl_file} (linhas: {len(lines)} -> {len(valid_lines)})')
                    else:
                        # label ficou vazio -> mover par removed e também mover imagem
                        target_lbl = removed_dir / rel
                        target_lbl.parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(lbl_file), str(target_lbl))
                        stats['labels_removed'] += 1
                        logger.info(f'Movido label vazio {lbl_file} -> {target_lbl}')
                        if img_file.exists():
                            target_img = removed_dir / rel.with_suffix(img_file.suffix)
                            target_img.parent.mkdir(parents=True, exist_ok=True)
                            shutil.move(str(img_file), str(target_img))
                            stats['images_removed'] += 1
                            logger.info(f'Movida imagem correspondente {img_file} -> {target_img}')

            # verificar imagem corrompida
            if img_file.exists() and not is_image_ok(img_file):
                stats['images_removed'] += 1
                if dry_run:
                    logger.info(f'[DRY] Remover imagem corrompida: {img_file}')
                else:
                    target_img = removed_dir / img_file.relative_to(images_root)
                    target_img.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(img_file), str(target_img))
                    logger.info(f'Removida imagem corrompida: {img_file} -> {target_img}')
                    # remover label também
                    if lbl_file.exists():
                        target_lbl = removed_dir / lbl_file.relative_to(labels_root)
                        target_lbl.parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(lbl_file), str(target_lbl))
                        stats['labels_removed'] += 1
                        logger.info(f'Removido label associado: {lbl_file} -> {target_lbl}')

        # gerar labels para imagens sem label (opcional)
        if regen_missing and not dry_run:
            # usa ultralytics via CLI para gerar labels rapidamente
            for img in img_dir.rglob('*.jpg'):
                rel = img.relative_to(img_dir)
                lbl = lbl_dir / rel.with_suffix('.txt')
                if not lbl.exists():
                    # chamar inferência e salvar labels no formato yolov8 (save_txt)
                    logger.info(f'Gerando label para {img}')
                    cmd = [sys.executable, '-m', 'ultralytics', 'detect', 'predict', '--model', model, '--source', str(img), '--save-txt', '--save-conf', '--project', 'runs/detect/regen_labels', '--name', 'regen']
                    try:
                        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        stats['labels_generated'] += 1
                    except Exception as e:
                        logger.error(f'Erro ao gerar label p/ {img}: {e}')

    logger.info(f'Finalizado. Estatísticas: {stats}')
    print('Limpeza concluída. Veja logs/clean_dataset.log para detalhes.')


def main():
    parser = argparse.ArgumentParser(description='Limpeza e correção de dataset YOLO')
    parser.add_argument('--dry-run', action='store_true', help='Não aplica mudanças; só log')
    parser.add_argument('--regen-missing', action='store_true', help='Regenerar labels faltantes via inferência')
    parser.add_argument('--model', default='models/yolov8n.pt', help='Modelo para regeneração de labels')
    args = parser.parse_args()

    try:
        clean_labels_and_images(dry_run=args.dry_run, regen_missing=args.regen_missing, model=args.model)
    except Exception as e:
        logger.exception('Erro no processo de limpeza')
        raise


if __name__ == '__main__':
    main()
