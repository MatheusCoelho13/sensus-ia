"""Treinamento iterativo com early-stop baseado em mAP50.

Este script treina em passos (chunks) e após cada chunk executa
validação. Se a métrica `mAP50` atingir o limiar configurado,
o treino é interrompido automaticamente.

Uso:
    python scripts/train.py --max-epochs 50 --step 5 --target-map 0.6

Parâmetros principais:
  --max-epochs : número máximo total de épocas
  --step       : épocas por iteração antes de validar
  --target-map : mAP50 alvo para parar o treino (0.0-1.0)
"""
import argparse
import os
from pathlib import Path
from ultralytics import YOLO
import shutil
import subprocess
import sys
import shutil as _shutil
try:
    import torch
except Exception:
    torch = None
    
try:
    from scripts import gpu_utils
except Exception:
    gpu_utils = None

# Importar monitor de temperatura
try:
    from scripts.gpu_monitor import GPUMonitor
except ImportError:
    GPUMonitor = None


def run():
    parser = argparse.ArgumentParser(description='Treinar com early-stop por mAP50')
    parser.add_argument('--max-epochs', type=int, default=50)
    parser.add_argument('--step', type=int, default=5)
    parser.add_argument('--target-map', type=float, default=0.60,
                        help='Parar quando mAP50 >= target (valor entre 0 e 1)')
    parser.add_argument('--export-threshold', type=float, default=0.50,
                        help='Se mAP50 >= export-threshold, copiar best.pt para models/best.pt')
    parser.add_argument('--docker', action='store_true', help='Forçar execução dentro do Docker (build e relançar)')
    parser.add_argument('--data', default='config/data.yaml')
    parser.add_argument('--model', default='models/yolov8n.pt')
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--device', default='auto', help="Device: 'auto', 'cpu', '0' (cuda index) or 'cuda:0')")
    parser.add_argument('--require-cuda', action='store_true', help='Abortar se CUDA/NVIDIA GPU não estiver disponível')
    parser.add_argument('--max-temp', type=int, default=85, help='Temperatura máxima (°C) — acima disso desliga o PC (padrão: 85)')
    parser.add_argument('--warning-temp', type=int, default=70, help='Temperatura de aviso (°C) (padrão: 70)')
    parser.add_argument('--temp-check-interval', type=int, default=60, help='Intervalo de verificação de temperatura (segundos, padrão: 60)')

    args = parser.parse_args()

    # garantir que o cwd é a raiz do projeto
    project_root = Path(__file__).parent.parent
    # Se não estamos dentro de um container Docker, reconstruir/relançar dentro do Docker (se disponível)
    def _in_docker():
        if os.environ.get('IN_DOCKER') == '1':
            return True
        if os.path.exists('/.dockerenv'):
            return True
        try:
            with open('/proc/1/cgroup', 'rt') as f:
                text = f.read()
                if 'docker' in text or 'kubepods' in text:
                    return True
        except Exception:
            pass
        return False

    # decidir se deve relançar dentro do Docker: somente quando explicitamente solicitado
    docker_requested = args.docker or os.environ.get('FORCE_DOCKER') == '1'
    if not _in_docker() and docker_requested:
        if _shutil.which('docker'):
            print('ℹ️  Docker solicitado — construindo imagem e relançando dentro do container...')
            try:
                subprocess.run(['docker', 'build', '-t', 'assistiva-ia', '.'], check=True)
            except subprocess.CalledProcessError as e:
                print(f'⚠ Falha ao buildar imagem Docker: {e}; continuando localmente')
            else:
                docker_cmd = ['docker', 'run', '--rm', '-v', f"{project_root}:/app", '-w', '/app', '-e', 'IN_DOCKER=1', 'assistiva-ia', 'python3', 'scripts/train.py'] + sys.argv[1:]
                print('➡️  Executando comando Docker:', ' '.join(docker_cmd))
                rc = subprocess.call(docker_cmd)
                sys.exit(rc)
        else:
            print('⚠ Docker solicitado, mas não encontrado no PATH — executando localmente')

    os.chdir(project_root)

    print(f"Iniciando treinamento iterativo: max_epochs={args.max_epochs}, step={args.step}, target_map={args.target_map}")

    # decidir dispositivo (auto detecta CUDA, ou usa IPEX se disponível)
    device = args.device
    use_ipex = False

    # require cuda flag -> fail early if no GPU
    if getattr(args, 'require_cuda', False):
        if gpu_utils is not None:
            gpu_utils.require_cuda_or_exit()
        else:
            # fallback: try torch and nvidia-smi
            ok = (torch is not None and getattr(torch, 'cuda', None) is not None and torch.cuda.is_available())
            if not ok:
                if shutil.which('nvidia-smi') is None:
                    print('CUDA/NVIDIA GPU não detectada (require-cuda). Abortando.')
                    sys.exit(1)
        # force cuda device
        device = 'cuda:0'

    if device == 'auto':
        # prefer GPU when available via torch
        try:
            if torch is not None and torch.cuda.is_available():
                device = '0'  # use first CUDA device
            else:
                # check for Intel extension for PyTorch (IPEX) for CPU optimizations
                try:
                    import intel_extension_for_pytorch as ipex  # type: ignore
                    use_ipex = True
                    device = 'cpu'
                except Exception:
                    device = 'cpu'
        except Exception:
            device = 'cpu'

    print(f"Usando device: {device} (IPEX={'yes' if use_ipex else 'no'})")

    # Inicializar monitor de temperatura (apenas para GPU)
    monitor = None
    if GPUMonitor is not None and device in ['0', 'cuda:0']:
        try:
            monitor = GPUMonitor(
                max_temp=args.max_temp,
                warning_temp=args.warning_temp,
                check_interval=args.temp_check_interval,
                gpu_index=0
            )
        except Exception as e:
            print(f"⚠️  Não foi possível inicializar monitor de temperatura: {e}")
            monitor = None

    # Se já existe models/best.pt, use como base para retreinamento
    model_path = args.model
    if Path('models/best.pt').exists():
        print('🔁 Usando models/best.pt como base para retreinamento')
        model_path = 'models/best.pt'
    model = YOLO(model_path)
    # Ensure overrides contain model path/task to avoid Ultralytics internal KeyError
    try:
        model.overrides["model"] = str(model_path)
        model.overrides["task"] = model.overrides.get("task", "detect")
    except Exception:
        pass

    trained_run_name = 'custom_run'
    total_trained = 0

    while total_trained < args.max_epochs:
        remaining = args.max_epochs - total_trained
        step = args.step if remaining >= args.step else remaining

        print(f"\n▶ Treinando por {step} época(s) (total já treinado: {total_trained})...")
        
        # ⚠️ Verificar temperatura antes de treinar
        if monitor is not None:
            if not monitor.check_temperature():
                print("🛑 Treinamento interrompido por limite de temperatura!")
                return
        
        # Treina em blocos; o Ultralytics criará um novo run (custom_run, custom_run2, ...)
        try:
            model.train(
                data=args.data,
                epochs=step,
                imgsz=args.imgsz,
                batch=args.batch,
                name=trained_run_name,
                exist_ok=True,
                resume=False,
                device=device,
            )
        except KeyError as e:
            # Ultralytics may raise KeyError if overrides['model'] missing; set it and retry once
            if "model" in str(e):
                try:
                    model.overrides["model"] = str(model_path)
                    model.overrides["task"] = model.overrides.get("task", "detect")
                except Exception:
                    pass
                print('⚠️ KeyError model detected; retrying train() after setting overrides["model"]')
                model.train(
                    data=args.data,
                    epochs=step,
                    imgsz=args.imgsz,
                    batch=args.batch,
                    name=trained_run_name,
                    exist_ok=True,
                    resume=False,
                    device=device,
                )
            else:
                raise

        total_trained += step

        # ⚠️ Verificar temperatura após cada bloco de treinamento
        if monitor is not None:
            if not monitor.check_temperature():
                print("🛑 Treinamento interrompido por limite de temperatura!")
                break
        
        # localizar melhor peso gerado no último run (o ultralytics cria custom_run, custom_run2, ...)
        runs_root = Path('runs/detect')
        candidates = [d for d in runs_root.iterdir() if d.is_dir() and d.name.startswith(trained_run_name)]
        if candidates:
            latest = max(candidates, key=lambda d: d.stat().st_mtime)
            best_path = latest / 'weights' / 'best.pt'
        else:
            best_path = Path('')

        if best_path.exists():
            print(f"✓ best.pt encontrado: {best_path}")
            # avaliar o modelo
            print("ℹ️  Executando validação (mAP)...")
            eval_model = YOLO(str(best_path))
            metrics = eval_model.val(data=args.data, imgsz=args.imgsz, batch=args.batch, device=device)

            # métrica disponível em metrics.box.map50 (0..1)
            try:
                current_map50 = float(metrics.box.map50)
            except Exception:
                print("⚠ Não foi possível ler mAP50 da validação; continuando treino")
                current_map50 = 0.0

            print(f"📈 mAP50 atual: {current_map50:.3f} (alvo: {args.target_map:.3f})")
            if current_map50 >= args.target_map:
                print(f"✅ Objetivo de precisão alcançado (mAP50 >= {args.target_map}) — interrompendo treino")
                # ⚠️ Verificar temperatura antes de parar
                if monitor is not None:
                    temp = monitor.get_temperature()
                    if temp is not None:
                        print(f"📊 Temperatura final da GPU: {temp}°C")
                # exportar para models/ se atingir export threshold
                try:
                    if current_map50 >= args.export_threshold:
                        models_dir = Path('models')
                        models_dir.mkdir(parents=True, exist_ok=True)
                        dest = models_dir / 'best.pt'
                        shutil.copy2(str(best_path), str(dest))
                        print(f"📦 Modelo exportado para: {dest}")
                except Exception as e:
                    print(f"⚠ Falha ao exportar modelo: {e}")
                return
            else:
                print("🔁 Ainda não atingiu o alvo; continuando treino...")
        else:
            print("⚠ best.pt não encontrado após este bloco de treino; continuando...")

    # após terminar todos os blocos, verificar último best e exportar se aplicável
    runs_root = Path('runs/detect')
    candidates = [d for d in runs_root.iterdir() if d.is_dir() and d.name.startswith(trained_run_name)] if runs_root.exists() else []
    latest = max(candidates, key=lambda d: d.stat().st_mtime) if candidates else None
    if latest:
        final_best = latest / 'weights' / 'best.pt'
        if final_best.exists():
            # avaliar e possivelmente exportar
            try:
                eval_model = YOLO(str(final_best))
                metrics = eval_model.val(data=args.data, imgsz=args.imgsz, batch=args.batch, device=device)
                try:
                    final_map50 = float(metrics.box.map50)
                except Exception:
                    final_map50 = 0.0
                print(f"📈 mAP50 final: {final_map50:.3f}")
                if final_map50 >= args.export_threshold:
                    models_dir = Path('models')
                    models_dir.mkdir(parents=True, exist_ok=True)
                    dest = models_dir / 'best.pt'
                    shutil.copy2(str(final_best), str(dest))
                    print(f"📦 Modelo final exportado para: {dest}")
            except Exception as e:
                print(f"⚠ Falha ao avaliar/exportar best.pt final: {e}")

    print("⚑ Treinamento completo (atingiu max_epochs)")
    
    # Limpeza do monitor de temperatura
    if monitor is not None:
        monitor.cleanup()


if __name__ == '__main__':
    run()
