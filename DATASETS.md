# 📊 Datasets Disponíveis

## Situação Atual

Você tem **dois datasets** com características diferentes:

### 1️⃣ **Dataset COCO** (59 imagens de training)
```
📁 datasets/coco/
├── images/
│   ├── train/  → 59 imagens ✅ (Anotações COCO em JSON)
│   └── val/    → 6 imagens
└── labels/     → Anotações em formato COCO
```

**Características:**
- ✅ Imagens de alta qualidade
- ✅ Anotações precisas em formato COCO
- ❌ Poucas imagens para treinar (59)
- ✅ Ótimo para validação final

**Usar com:**
```bash
make start_train_cuda  # Dataset padrão: config/data.yaml
```

---

### 2️⃣ **Dataset CUSTOM** (58 imagens de training) ⭐ **RECOMENDADO**
```
📁 datasets/images/
├── train/  → 20 imagens
│   ├── cadeira/
│   ├── pessoa/
│   └── ...
└── val/    → 38 imagens
    ├── cadeira/
    ├── mesa/
    ├── parede/
    ├── porta/
    └── ...
```

**Características:**
- ✅ Mais imagens no total (58)
- ✅ Distribuição menos desbalanceada (20 train, 38 val)
- ❌ Sem anotações YOLO completas (apenas imagens)
- ✅ Treino rápido para seu GTX 1650

**Usar com:**
```bash
make start_train_cuda_custom  # Dataset custom: config/data_custom.yaml
```

---

## 📈 Comparação

| Aspecto | COCO | CUSTOM |
|---------|------|--------|
| **Total de imagens** | 65 | 58 |
| **Train** | 59 | 20 |
| **Val** | 6 | 38 |
| **Qualidade anotações** | Alta (JSON) | Baixa (só imagens) |
| **Recomendado para** | Produção | Desenvolvimento |
| **Batches por epoch** | 4 | 1-2 |
| **Tempo epoch** | ~30s | ~10s |

---

## 🚀 Como Usar

### Opção A: Treinar com DATASET COCO (padrão)
```bash
make start_train_cuda
```

- Usa: `config/data.yaml` → `datasets/coco/`
- 59 imagens de training
- Mais lento, mas melhor qualidade
- Resultados em: `runs/detect/custom_run/`

---

### Opção B: Treinar com DATASET CUSTOM ⭐ (RECOMENDADO)
```bash
make start_train_cuda_custom
```

- Usa: `config/data_custom.yaml` → `datasets/images/`
- 20 imagens de training
- Mais rápido (2-3x), perfeito para GTX 1650
- Detecção rodará em 20 + 38 = **58 imagens**
- Resultados em: `runs/detect/custom_run/`

---

### Opção C: Treinar com DATASET COMBINADO (futuro)

Se quiser combinar ambos os datasets em um só:

```bash
# 1. Copiar COCO para a pasta custom
cp -r datasets/coco/images/train/* datasets/images/train/

# 2. Treinar com o combined
make start_train_cuda_custom
```

Isso daria ~80 imagens para treinar! 

---

## 📝 Arquivos de Configuração

### `config/data.yaml` (COCO)
```yaml
path: ../datasets/coco
train: images/train    # 59 imagens
val: images/val        # 6 imagens
nc: 5
names: ['person', 'backpack', 'chair', 'bench', 'laptop']
```

### `config/data_custom.yaml` (CUSTOM)
```yaml
path: ../datasets/images
train: train           # 20 imagens
val: val              # 38 imagens
nc: 5
names: ['person', 'backpack', 'chair', 'bench', 'laptop']
```

---

## 💡 Qual Usar?

**Para DESENVOLVIMENTO (agora):**
```bash
make start_train_cuda_custom  # ⭐ Rápido e funcional
```

**Para PRODUÇÃO (depois):**
1. Combinar ambos os datasets
2. Treinar com 80+ imagens
3. Validar com dados reais

---

## 🔍 Verificar Imagens

```bash
# Ver quantas imagens em cada dataset
echo "COCO:" && ls datasets/coco/images/train/ | wc -l
echo "CUSTOM train:" && ls datasets/images/train/ | wc -l
echo "CUSTOM val:" && ls datasets/images/val/ | wc -l
```

---

**Recomendação:** Use `make start_train_cuda_custom` para treinar mais rápido! 🚀
