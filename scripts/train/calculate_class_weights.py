#!/usr/bin/env python3
"""Calcular pesos das classes baseado na distribuição de anotações no dataset.

Uso:
    python scripts/calculate_class_weights.py --data config/data.yaml
    
Saída:
    Pesos normalizados (0-1) para cada classe para balanceamento automático.
"""
import argparse
import yaml
from pathlib import Path
from collections import Counter
import numpy as np


def calculate_class_weights(labels_dir: Path, num_classes: int):
    """Calcular pesos inversos à frequência das classes."""
    class_counts = Counter()
    
    # Contar anotações por classe
    for txt_file in labels_dir.glob('*.txt'):
        with open(txt_file, 'r') as f:
            for line in f:
                if line.strip():
                    class_id = int(line.split()[0])
                    class_counts[class_id] += 1
    
    # Inicializar counts para todas classes
    for i in range(num_classes):
        if i not in class_counts:
            class_counts[i] = 0
    
    # Calcular pesos: inverso da frequência
    total = sum(class_counts.values())
    if total == 0:
        print("⚠️  Nenhuma anotação encontrada!")
        return None
    
    weights = {}
    for class_id in range(num_classes):
        count = class_counts.get(class_id, 0)
        if count > 0:
            # Peso inverso à frequência
            weight = total / (num_classes * count)
        else:
            # Classes sem anotações recebem peso máximo
            weight = total / num_classes
        weights[class_id] = weight
    
    # Normalizar para 0-1
    max_weight = max(weights.values())
    normalized_weights = {k: v / max_weight for k, v in weights.items()}
    
    return normalized_weights, class_counts


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='config/data.yaml', help='Arquivo de configuração YAML')
    p.add_argument('--output', default='config/class_weights.yaml', help='Saída YAML com pesos')
    args = p.parse_args()
    
    # Carregar config
    config = yaml.safe_load(Path(args.data).read_text())
    num_classes = config.get('nc', 10)
    names = config.get('names', {})
    
    # Se names é lista, converter para dict
    if isinstance(names, list):
        names = {i: n for i, n in enumerate(names)}
    
    # Contar anotações
    dataset_path = Path(config.get('path', 'datasets/coco'))
    labels_train = dataset_path / 'labels' / 'train'
    labels_val = dataset_path / 'labels' / 'val'
    
    print(f"📊 Contando anotações em {labels_train}...")
    weights_train, counts_train = calculate_class_weights(labels_train, num_classes)
    
    print(f"\n📈 Distribuição de classes (treino):")
    total = sum(counts_train.values())
    for i in range(num_classes):
        count = counts_train.get(i, 0)
        pct = 100 * count / total if total > 0 else 0
        weight = weights_train.get(i, 0)
        class_name = names.get(i, f'class_{i}')
        print(f"  {class_name:15s}: {count:4d} anotações ({pct:5.1f}%) → peso={weight:.3f}")
    
    # Salvar em arquivo YAML
    output_data = {
        'class_weights': weights_train,
        'class_counts': dict(counts_train),
        'total_instances': total,
        'num_classes': num_classes,
    }
    
    Path(args.output).write_text(yaml.dump(output_data, default_flow_style=False))
    print(f"\n✅ Pesos salvos em: {args.output}")
    
    # Saída para uso em shell/Makefile
    weights_list = [str(weights_train.get(i, 0)) for i in range(num_classes)]
    print(f"\n📌 Para usar no treinamento:")
    print(f"   python scripts/train.py --cls-weight '{','.join(weights_list)}'")


if __name__ == '__main__':
    main()
