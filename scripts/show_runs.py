#!/usr/bin/env python3
"""Listar runs em runs/detect e mostrar mAP50 de cada um.

Uso:
    python3 scripts/show_runs.py --data config/data.yaml --device cpu

Se um run não tiver métricas gravadas, o script carrega `weights/best.pt`
e executa uma validação rápida para obter `mAP50`.
"""
import argparse
import json
from pathlib import Path
from ultralytics import YOLO
import sys


def read_metrics_file(run_dir: Path):
    # tenta arquivos comuns onde métricas podem estar
    for name in ("metrics.json", "results.json", "metrics.yaml", "results.txt"):
        p = run_dir / name
        if p.exists():
            try:
                if p.suffix == '.txt':
                    return None
                return json.loads(p.read_text())
            except Exception:
                return None
    return None


def get_map50_from_metrics(metrics):
    try:
        return float(metrics.get('box', {}).get('map50'))
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', default='runs/detect')
    parser.add_argument('--data', default='config/data.yaml')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    runs_root = Path(args.runs)
    if not runs_root.exists():
        print(f"Pasta de runs não encontrada: {runs_root}")
        sys.exit(1)

    results = []
    for d in sorted(runs_root.iterdir()):
        if not d.is_dir():
            continue
        best = d / 'weights' / 'best.pt'
        map50 = None
        metrics = read_metrics_file(d)
        if metrics:
            map50 = get_map50_from_metrics(metrics)

        if map50 is None and best.exists():
            # validar carregando o modelo (pode ser lento)
            try:
                print(f"Validando {d.name} carregando {best}...")
                m = YOLO(str(best))
                metrics_obj = m.val(data=args.data, device=args.device)
                try:
                    map50 = float(metrics_obj.box.map50)
                except Exception:
                    map50 = None
            except Exception as e:
                print(f"Falha ao validar {best}: {e}")
                map50 = None

        results.append((d.name, map50, str(best) if best.exists() else ''))

    # ordenar pelo map50 desc (None ficam no final)
    results.sort(key=lambda x: (-(x[1] or 0.0), x[0]))

    print('\nRuns encontrados:')
    best_run = None
    best_map = -1.0
    for name, map50, best_path in results:
        score = f"{map50:.4f}" if map50 is not None else "N/A"
        print(f"- {name}: mAP50={score}  weights={best_path}")
        if map50 is not None and map50 > best_map:
            best_map = map50
            best_run = name

    if best_run:
        print(f"\nMelhor run: {best_run} (mAP50={best_map:.4f})")
    else:
        print("\nNenhuma métrica disponível para os runs encontrados.")


if __name__ == '__main__':
    main()
