"""scripts/train.py

Refatoração do loop de treino iterativo com early-stop por mAP50.

Decisões principais:
- Separação clara de responsabilidades (parsing, device, treino, validação, saving).
- Uso de um único `run_name` (evita suffix like custom_run2) e `exist_ok=True`.
- Reuso do melhor peso disponível como ponto de partida.
- Validação opcional/leve via `--fast-validate` para reduzir overhead.
- Salvamento de `models/best.pt` apenas quando mAP50 melhora.

Este arquivo evita manipular `model.overrides` diretamente e trata erros do Ultralytics
com tentativas controladas.
"""

import argparse
import sys
import time
from pathlib import Path
import yaml
import shutil
import csv
import json
from typing import List, Dict

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    import torch
except Exception:
    torch = None

from ultralytics import YOLO


DEFAULT_RUN_NAME = 'train_iter'


def parse_args():
    p = argparse.ArgumentParser(description='Treinar iterativamente com early-stop por mAP50')
    p.add_argument('--max-epochs', type=int, default=150)
    p.add_argument('--step', type=int, default=5, help='épocas por chunk antes de validar')
    p.add_argument('--target-map', type=float, default=0.60)
    p.add_argument('--export-threshold', type=float, default=0.50)
    p.add_argument('--data', default='config/data.yaml')
    p.add_argument('--model', default='models/yolov8n.pt') # // testar com o modelo pré-treinado seq.pt, que tem melhor capacidade de generalização ou o modelo mais leve yolov8n.pt ou o mais  o seq
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--device', default='auto')
    p.add_argument('--require-cuda', action='store_true')
    p.add_argument('--cls-weight', type=str, default='')
    p.add_argument('--fast-validate', action='store_true', help='usar validação mais rápida (imagens menores / menos batch)')
    p.add_argument('--skip-validate', action='store_true', help='pular validação entre chunks (mais rápido, menos seguro)')
    p.add_argument('--run-name', default=DEFAULT_RUN_NAME)
    return p.parse_args()


def select_device(arg_device: str, require_cuda: bool = False):
    # retorna string aceitável por Ultralytics ('cpu' ou index '0' ou 'cuda:0')
    if arg_device and arg_device != 'auto':
        return arg_device
    if require_cuda:
        if torch is None or not getattr(torch, 'cuda', None) or not torch.cuda.is_available():
            raise RuntimeError('CUDA requerida, mas não disponível')
        return '0'
    # auto-detect
    if torch is not None and torch.cuda.is_available():
        return '0'
    return 'cpu'


def _find_latest_best_weight(project_root: Path):
    candidates = []
    local_best = project_root / 'models' / 'best.pt'
    if local_best.exists():
        candidates.append(local_best)
    runs_root = project_root / 'runs' / 'detect'
    if runs_root.exists():
        candidates.extend(sorted(runs_root.glob('*/weights/best.pt')))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_model(start_path: str):
    # Carrega o modelo YOLO a partir de um arquivo ou checkpoint
    try:
        model = YOLO(start_path)
        return model
    except Exception as e:
        raise RuntimeError(f'Falha ao carregar modelo {start_path}: {e}')


def compute_class_weights_from_data(data_yaml: str):
    """Calcula pesos de classe automaticamente a partir dos labels YOLO no dataset definido em data.yaml.

    Retorna uma lista de floats com tamanho `nc` onde weight[i] = total_annotations / (nc * count_i)
    Se uma classe não tiver anotações, atribui weight=1.0 para evitar divisão por zero.
    """
    p = Path(data_yaml)
    if not p.exists():
        print(f'⚠️ data.yaml não encontrado em {data_yaml} — não é possível calcular cls weights')
        return None
    data = yaml.safe_load(p.read_text())
    nc = int(data.get('nc', len(data.get('names', []))))
    base_path = Path(data.get('path', '.'))
    # presumir layout: labels/train e labels/val
    labels_train = base_path / 'labels' / 'train'
    if not labels_train.exists():
        # tentar caminho alternativo: base_path / data.get('train') replace images->labels
        train_rel = data.get('train', '')
        if train_rel:
            labels_train = base_path / Path(train_rel).parent / 'labels' / Path(train_rel).name
    if not labels_train.exists():
        # último recurso: procurar por labels dentro do dataset
        candidates = list(base_path.rglob('labels'))
        labels_train = candidates[0] / 'train' if candidates else None

    counts = [0] * nc
    total = 0
    if labels_train and labels_train.exists():
        for f in labels_train.glob('*.txt'):
            for ln in f.read_text().splitlines():
                parts = ln.strip().split()
                if not parts:
                    continue
                try:
                    cid = int(parts[0])
                except Exception:
                    continue
                if 0 <= cid < nc:
                    counts[cid] += 1
                    total += 1

    if total == 0:
        print('⚠️ Nenhuma anotação encontrada ao calcular pesos de classe')
        return None

    weights = []
    for c in counts:
        if c <= 0:
            weights.append(1.0)
        else:
            weights.append((total / (nc * c)))
    return weights


def validate_model(weight_path: Path, data: str, imgsz: int, batch: int, device: str, fast: bool):
    # Retorna map50 (float) - execução envolvendo YOLO.val
    try:
        model = YOLO(str(weight_path))
        v_imgsz = imgsz // 2 if fast else imgsz
        v_batch = max(1, batch // 2) if fast else batch
        metrics = model.val(data=data, imgsz=v_imgsz, batch=v_batch, device=device)
        return float(metrics.box.map50)
    except Exception as e:
        print(f'⚠️ Erro na validação ({weight_path}): {e}')
        return 0.0


def pick_best_from_runs(run_name: str):
    runs_root = Path('runs') / 'detect'
    if not runs_root.exists():
        return None
    candidates = [d for d in runs_root.iterdir() if d.is_dir() and d.name == run_name]
    if not candidates:
        # try prefix matches
        candidates = [d for d in runs_root.iterdir() if d.is_dir() and d.name.startswith(run_name)]
    if not candidates:
        return None
    latest = max(candidates, key=lambda d: d.stat().st_mtime)
    best_path = latest / 'weights' / 'best.pt'
    return best_path if best_path.exists() else None


def safe_copy_best(src: Path, dst: Path):
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src), str(dst))
        return True
    except Exception as e:
        print(f'⚠️ Falha ao copiar best.pt {src} -> {dst}: {e}')
        return False


def main():
    args = parse_args()
    project_root = Path(__file__).parent.parent
    device = select_device(args.device, args.require_cuda)
    print(f'Usando device: {device}')

    # escolher ponto de partida: models/best.pt > latest run best > provided model
    start_path = args.model
    latest_best = _find_latest_best_weight(project_root)
    if latest_best is not None:
        print(f'🔁 Usando melhor peso encontrado: {latest_best}')
        start_path = str(latest_best)

    model = load_model(start_path)

    # baseline map50 (se existir models/best.pt)
    baseline_map50 = -1.0
    models_best = project_root / 'models' / 'best.pt'
    if models_best.exists():
        print('ℹ️ Validando models/best.pt como baseline...')
        baseline_map50 = validate_model(models_best, args.data, args.imgsz, args.batch, device, fast=args.fast_validate)
        print(f'📌 baseline mAP50 = {baseline_map50:.3f}')

    run_name = args.run_name
    total_trained = 0
    best_map50_seen = baseline_map50
    epoch_records: List[Dict] = []

    # parse class weights if provided
    cls_weights = None
    if args.cls_weight:
        if args.cls_weight.lower() == 'auto':
            print('ℹ️ Calculando class weights automaticamente (auto) ...')
            cls_weights = compute_class_weights_from_data(args.data)
            if cls_weights:
                print('🎯 Class weights (auto):', cls_weights)
            else:
                print('⚠️ Falha ao calcular class weights automaticamente; ignorando')
        else:
            try:
                cls_weights = [float(w) for w in args.cls_weight.split(',')]
                print('🎯 Class weights:', cls_weights)
            except Exception:
                print('⚠️ Invalid --cls-weight; ignorando')

    # Entrar no loop de treino por chunks
    while total_trained < args.max_epochs:
        remaining = args.max_epochs - total_trained
        step = args.step if remaining >= args.step else remaining
        print(f'\n▶ Treinando por {step} épocas (treinado {total_trained}/{args.max_epochs})')

        train_kwargs = dict(
            data=args.data,
            epochs=step,
            imgsz=args.imgsz,
            batch=args.batch,
            name=run_name,
            exist_ok=True,
            resume=False,
            device=device,
        )
        if cls_weights:
            # Ultralytics espera 'class_weights' como key na API Python
            train_kwargs['class_weights'] = cls_weights

        try:
            model.train(**train_kwargs)
        except Exception as e:
            print(f'⚠️ Erro durante model.train(): {e} — tentando salvar checkpoint e continuar')
            time.sleep(1)

        total_trained += step

        # validação entre chunks (pode ser pulada)
        if args.skip_validate:
            print('ℹ️ skip_validate ativo — pulando validação')
            continue

        # localizar diretório do run atual e ler results.csv para obter métricas por época
        run_dir = Path('runs') / 'detect' / run_name
        def read_results_csv(run_dir: Path) -> List[Dict]:
            results_path = run_dir / 'results.csv'
            if not results_path.exists():
                return []
            rows: List[Dict] = []
            try:
                with results_path.open('r', encoding='utf-8', errors='ignore') as fh:
                    reader = csv.DictReader(fh)
                    for r in reader:
                        out = {}
                        for k, v in r.items():
                            if v is None or v == '':
                                out[k] = None
                                continue
                            try:
                                if k == 'epoch':
                                    out[k] = int(float(v))
                                else:
                                    out[k] = float(v)
                            except Exception:
                                out[k] = v
                        rows.append(out)
            except Exception as e:
                print(f'⚠️ Erro lendo results.csv: {e}')
            return rows

        new_rows = read_results_csv(run_dir)
        last_epoch_seen = epoch_records[-1]['epoch'] if epoch_records else 0
        for ep in new_rows:
            if ep.get('epoch', 0) > last_epoch_seen:
                prec = ep.get('metrics/precision(B)')
                map50 = ep.get('metrics/mAP50(B)')
                if prec is not None:
                    print(f'📊 Época {ep["epoch"]}: precision={prec:.3f} mAP50={map50:.3f}')
                epoch_records.append(ep)

        # salvar registros acumulados em JSON (útil para inspeção externa)
        try:
            if epoch_records:
                run_dir.mkdir(parents=True, exist_ok=True)
                with (run_dir / 'epoch_metrics.json').open('w', encoding='utf-8') as fh:
                    json.dump(epoch_records, fh, indent=2)
        except Exception as e:
            print(f'⚠️ Falha ao salvar epoch_metrics.json: {e}')

        # gerar gráfico das métricas até agora (se matplotlib estiver disponível)
        try:
            if plt is not None and epoch_records:
                xs = [e['epoch'] for e in epoch_records if e.get('epoch') is not None]
                ys_prec = [e.get('metrics/precision(B)', 0.0) for e in epoch_records]
                ys_map50 = [e.get('metrics/mAP50(B)', 0.0) for e in epoch_records]
                plt.figure(figsize=(8,4))
                plt.plot(xs, ys_prec, label='precision', marker='o')
                plt.plot(xs, ys_map50, label='mAP50', marker='x')
                plt.xlabel('época')
                plt.ylabel('valor')
                plt.title('Precision e mAP50 por época')
                plt.grid(True)
                plt.legend()
                out_png = run_dir / 'epoch_metrics.png'
                plt.tight_layout()
                plt.savefig(out_png)
                plt.close()
                print(f'📈 Gráfico salvo em: {out_png}')
        except Exception as e:
            print(f'⚠️ Falha ao gerar gráfico: {e}')

        # localiza o best.pt do run atual
        best_path = pick_best_from_runs(run_name)
        if best_path is None or not best_path.exists():
            print('⚠️ Nenhum best.pt encontrado para o run atual; pulando validação deste chunk')
            continue

        print(f'✓ Encontrado best.pt: {best_path} — executando validação leve')
        current_map50 = validate_model(best_path, args.data, args.imgsz, args.batch, device, fast=args.fast_validate)
        print(f'📈 mAP50 atual: {current_map50:.3f} (melhor até agora: {best_map50_seen:.3f})')

        # se melhorou, salvar como models/best.pt
        if current_map50 > best_map50_seen:
            if safe_copy_best(best_path, project_root / 'models' / 'best.pt'):
                best_map50_seen = current_map50
                print(f'📦 Novo best exportado (mAP50={best_map50_seen:.3f})')

        # early stop
        if current_map50 >= args.target_map:
            print(f'✅ Objetivo mAP50 atingido: {current_map50:.3f} >= {args.target_map:.3f} — encerrando treino')
            break

    # avaliação final do melhor modelo disponível
    final_best = project_root / 'models' / 'best.pt'
    if final_best.exists():
        final_map = validate_model(final_best, args.data, args.imgsz, args.batch, device, fast=args.fast_validate)
        print(f'📈 mAP50 final do models/best.pt: {final_map:.3f}')
    else:
        print('⚠️ models/best.pt não existe — verifique os runs em runs/detect')


if __name__ == '__main__':
    main()
