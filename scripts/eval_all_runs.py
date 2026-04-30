"""
Avalia todos os runs em runs/detect/*, mostra o mAP50 de cada best.pt e indica o melhor.
Uso:
    python3 scripts/eval_all_runs.py
"""
import os
from pathlib import Path
from ultralytics import YOLO

def main():
    runs_dir = Path('runs/detect')
    if not runs_dir.exists():
        print('Nenhum run encontrado em runs/detect/')
        return
    results = []
    for run in sorted(runs_dir.iterdir()):
        best = run / 'weights' / 'best.pt'
        if best.exists():
            print(f'Avaliando {best}...')
            try:
                model = YOLO(str(best))
                metrics = model.val()
                map50 = float(metrics.box.map50)
                results.append((run.name, map50, best))
                print(f'  mAP50: {map50:.4f}')
            except Exception as e:
                print(f'  Erro ao avaliar {best}: {e}')
        else:
            print(f'Ignorando {run}: best.pt não encontrado')
    if not results:
        print('Nenhum best.pt válido encontrado.')
        return
    print('\nResumo:')
    for name, map50, best in results:
        print(f'  {name:25s}  mAP50={map50:.4f}')
    best_run = max(results, key=lambda x: x[1])
    print(f'\n🏆 Melhor run: {best_run[0]}  (mAP50={best_run[1]:.4f})')
    print(f'Arquivo: {best_run[2]}')

if __name__ == '__main__':
    main()
