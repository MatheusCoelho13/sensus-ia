# Estrutura de Pastas Organizada

## 📂 Layout do Projeto

```
ia/
├── app.py                    👈 API principal (FastAPI)
├── Dockerfile               👈 Container
├── README.md               👈 Documentação
│
├── .venv/                  👈 Virtual environment
├── venv/                   👈 Venv antigo (pode deletar)
│
├── config/                 👈 Configurações
│   ├── requirements.txt    - Dependências Python
│   └── data.yaml           - Config dataset (classes, paths)
│
├── scripts/                👈 Scripts de automação
│   ├── full_pipeline.sh    - Orquestrador (fetch + download + treino + eval)
│   ├── fetch_wikimedia_urls.py  - Buscar URLs
│   ├── download_images.py  - Baixar imagens com retry/delay
│   ├── prepare_dataset.py  - Limpar + gerar labels automáticas
│   ├── train.py            - Treinar modelo YOLOv8
│   ├── evaluate.py         - Avaliar modelo
│   ├── setup.sh            - Setup inicial
│   └── start.sh            - Iniciar API
│
├── data/                   👈 Dados / URLs
│   └── urls/               - Listas de URLs
│       ├── urls.txt        - URLs coletadas
│       ├── urls_sample.txt - Exemplos
│       └── example_urls.txt
│
├── datasets/               👈 Dataset YOLO (imagens + labels)
│   ├── images/
│   │   ├── train/          - Imagens treino
│   │   └── val/            - Imagens validação
│   ├── labels/
│   │   ├── train/          - Labels treino (.txt)
│   │   └── val/            - Labels validação (.txt)
│   └── downloaded_urls.txt - Log de URLs baixadas
│
├── models/                 👈 Modelos pré-treinados
│   └── yolov8n.pt          - YOLOv8 nano pré-treinado
│
├── runs/                   👈 Resultados de treino
│   └── detect/
│       └── custom_run/
│           ├── weights/
│           │   ├── best.pt - Melhor modelo treinado
│           │   └── last.pt - Último modelo checkpoint
│           ├── results.csv - Métricas por época
│           └── labels.jpg  - Visualização dos labels
│
├── logs/                   👈 Arquivos de log
│   ├── download.log
│   ├── train.log
│   ├── prepare.log
│   └── pipeline.log
│
└── output/                 👈 Outputs e resultados
```

## 🚀 Como Usar

### 1️⃣ Setup Inicial

```bash
source .venv/bin/activate
pip install -r config/requirements.txt
```

### 2️⃣ Rodar Pipeline Completo (Recomendado)

```bash
bash scripts/full_pipeline.sh
```

Ou em background (continua se hibernar):

```bash
nohup bash scripts/full_pipeline.sh > logs/pipeline.log 2>&1 &
tail -f logs/pipeline.log
```

### 3️⃣ Rodar Passos Individuais

**Fetch URLs:**

```bash
python3 scripts/fetch_wikimedia_urls.py --out data/urls/urls.txt --count 300
```

**Download Imagens:**

```bash
python3 scripts/download_images.py data/urls/urls.txt datasets \
  --max 5000 --split 0.8 --delay 2.0 --per-class 50
```

**Preparar Dataset:**

```bash
python3 scripts/prepare_dataset.py
```

**Treinar:**

```bash
python3 scripts/train.py
```

**Avaliar:**

```bash
python3 scripts/evaluate.py --weights runs/detect/custom_run/weights/best.pt --data config/data.yaml
```

### 4️⃣ Usar Modelo Treinado na API

```bash
# Editar app.py e trocar a linha do model load:
# model = YOLO("runs/detect/custom_run/weights/best.pt")

# Rodar API:
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

## 📊 Monitorar Treino

```bash
# Ver logs em tempo real
tail -f logs/train.log

# Visualizar resultados
cat runs/detect/custom_run/results.csv
```

## 🧹 Limpár Antigos

Se houver pasta `venv/` antiga, pode deletar:

```bash
rm -rf venv/
```

---

✅ Projeto organizado!
