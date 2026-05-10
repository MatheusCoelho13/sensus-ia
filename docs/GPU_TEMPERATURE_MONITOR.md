# 🌡️ Monitor de Temperatura da GPU

## O que faz?

O monitor de temperatura verifica **automaticamente** a temperature da GPU durante o treinamento e:
- ⚠️ **Avisa ao usuário** quando a temperatura fica acima de 70°C
- 🔴 **Desliga o PC automaticamente** se a temperatura ultrapassar 85°C (proteção do hardware)

## Como usar?

### 1. Verificar temperatura atual da GPU
```bash
make check_gpu_temp
```

Resultado:
```
✓ GPU Monitor ativado para: NVIDIA GeForce GTX 1650 (index 0)
🌡️  Temperatura da GPU: 35°C
```

### 2. Treinar COM monitoramento de temperatura automático

Quando você rodar `make start_train_cuda`, o monitor está **sempre ativo**:

```bash
make start_train_cuda
```

Durante o treinamento:
- A cada 60 segundos, a temperatura é verificada
- Se T > 70°C: `⚠️  GPU QUENTE: 75°C >= 70°C (limite: 85°C)`
- Se T > 85°C: `🔴 DESLIGANDO O PC EM 30 SEGUNDOS`

### 3. Personalizar limites de temperatura

Você pode passar parâmetros customizados:

```bash
python scripts/train.py \
  --max-epochs 100 \
  --max-temp 80 \
  --warning-temp 65 \
  --temp-check-interval 30 \
  --batch 8 \
  --imgsz 416 \
  --device cuda:0
```

**Parâmetros:**
- `--max-temp` (padrão: 85°C) - temperatura que desliga o PC
- `--warning-temp` (padrão: 70°C) - temperatura que mostra aviso
- `--temp-check-interval` (padrão: 60s) - frequência de verificação

## Limitações

- ⚠️ Funciona **apenas em GPUs NVIDIA** (requer NVIDIA Driver + `nvidia-ml-py`)
- ⚠️ Desligamento automático funciona apenas em **Windows e Linux**
- ⚠️ Se você não tiver `nvidia-ml-py` instalado, será instalado automaticamente na primeira execução

## Códigos de Status

| Temperatura | Status | Ação |
|------------|--------|------|
| < 50°C    | 🟢 Ok  | Normal |
| 50-70°C   | 🟡 Morno | Monitor atento |
| 70-85°C   | 🟠 Quente | Aviso exibido |
| ≥ 85°C    | 🔴 Crítico | PC desliga em 30s |

## Exemplo de Saída Completa

```
🎮 GPU Detectada: GTX 1650 (otimizado) ⚙️  Parâmetros: batch=8, imgsz=416

✓ GPU Monitor ativado para: NVIDIA GeForce GTX 1650 (index 0)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
▶ FASE 1: Treinamento (batch=8, imgsz=416)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Epoch 1/100
⚠️  GPU QUENTE: 72°C >= 70°C (limite: 85°C)

Epoch 2/100
📈 mAP50 atual: 0.453

Epoch 5/100
✅ Objetivo de precisão alcançado
📊 Temperatura final da GPU: 68°C
```

## Troubleshooting

### "nvidia-ml-py não instalado"
```bash
pip install nvidia-ml-py
```

### "CUDA não disponível"
Confirme que você tem:
1. NVIDIA GPU instalada
2. NVIDIA Driver atualizado (comando: `nvidia-smi`)
3. PyTorch com CUDA: `python -c "import torch; print(torch.cuda.is_available())"`

### "Não foi possível ler temperatura"
- Verifique NVIDIA Driver: `nvidia-smi`
- Reinicie o script

## Interrupção Manual

Se precisar parar o treinamento antes de atingir o limite de temperatura:
- Pressione `Ctrl+C` no terminal

O PC **não vai desligar** enquanto o script estiver rodando normalmente.

---

👍 **Dica:** Use `make check_gpu_temp` periodicamente para monitorar a saúde da sua GPU!
