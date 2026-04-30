import fiftyone as fo
import fiftyone.zoo as foz
import os

# =========================
# CLASSES DO SEU PROJETO
# =========================
CLASSES = [
    "person",
    "backpack",
    "chair",
    "bench",
    "laptop",
    "cell phone",
    "bottle",
    "book",
    "cup",
    "tv"
]

EXPORT_DIR = "datasets/coco_filtered_nav"

# =========================
# CARREGA COCO FILTRADO
# =========================
print("🚀 Baixando COCO filtrado...")

dataset_train = foz.load_zoo_dataset(
    "coco-2017",
    split="train",
    label_types=["detections"],
    classes=CLASSES,
    max_samples=8000
)

dataset_val = foz.load_zoo_dataset(
    split="validation",
    label_types=["detections"],
    classes=CLASSES,
    max_samples=2000
)

# =========================
# EXPORTA PARA YOLO
# =========================
print("📦 Exportando dataset...")

dataset_train.export(
    export_dir=os.path.join(EXPORT_DIR, "train"),
    dataset_type=fo.types.YOLOv5Dataset,
    label_field="ground_truth",
)

dataset_val.export(
    export_dir=os.path.join(EXPORT_DIR, "val"),
    dataset_type=fo.types.YOLOv5Dataset,
    label_field="ground_truth",
)

print("✅ Dataset pronto!")
print("📁 Local:", EXPORT_DIR)