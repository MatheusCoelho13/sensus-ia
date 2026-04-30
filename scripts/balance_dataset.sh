#!/bin/bash
# Script para aumentar dataset val copiando imagens de train
# Copia % de imagens do train para val por classe

# prefer dataset under ia/ when present (filter created ia/datasets/coco)
if [ -d "ia/datasets/coco/images/train" ]; then
    TRAIN_DIR="ia/datasets/coco/images/train"
    VAL_DIR="ia/datasets/coco/images/val"
else
    TRAIN_DIR="datasets/coco/images/train"
    VAL_DIR="datasets/coco/images/val"
fi

echo "📊 Contando imagens antes..."
echo ""

# Contar train
echo "📁 TRAIN:"
for class in $TRAIN_DIR/*/; do
    class_name=$(basename "$class")
    count=$(ls "$class"*.{jpg,png} 2>/dev/null | wc -l)
    echo "  $class_name: $count"
done

echo ""
echo "📁 VAL:"
for class in $VAL_DIR/*/; do
    class_name=$(basename "$class")
    count=$(ls "$class"*.{jpg,png} 2>/dev/null | wc -l)
    echo "  $class_name: $count"
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔄 Copiando 40% de imagens do train para val..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

PERCENT=${1:-40}  # 40% por padrão, ou usar argumento
echo "Percentual: $PERCENT%"
echo ""

# Processar cada classe do train
## Detect if train has class subfolders or flat images with YOLO labels
has_subdirs=false
for d in $TRAIN_DIR/*/; do
    if [ -d "$d" ]; then
        has_subdirs=true
        break
    fi
done

if [ "$has_subdirs" = true ]; then
    # original behavior: per-class image folders
    for train_class in $TRAIN_DIR/*/; do
        class_name=$(basename "$train_class")

        images=($(ls "$train_class"*.{jpg,png} 2>/dev/null))
        if [ ${#images[@]} -eq 0 ]; then
            continue
        fi

        total=${#images[@]}
        to_copy=$((total * PERCENT / 100))
        if [ $to_copy -lt 1 ]; then
            to_copy=1
        fi

        echo "📋 $class_name:"
        echo "   Total em train: $total"
        echo "   Copiando: $to_copy imagens para val"

        mkdir -p "$VAL_DIR/$class_name"

        copied=0
        for img in "${images[@]}"; do
            if [ $copied -ge $to_copy ]; then
                break
            fi
            filename=$(basename "$img")
            if [ ! -f "$VAL_DIR/$class_name/$filename" ]; then
                cp "$img" "$VAL_DIR/$class_name/$filename"
                ((copied++))
            fi
        done

        echo "   ✅ Copiadas: $copied"
        echo ""
    done
else
    # flat structure: use YOLO labels to determine class per image
    # derive dataset root (e.g. datasets/coco) from TRAIN_DIR (datasets/coco/images/train)
    DATA_ROOT="$(dirname "$(dirname "$TRAIN_DIR")")"
    LABELS_TRAIN="$DATA_ROOT/labels/train"
    IMAGES_TRAIN="$TRAIN_DIR"
    if [ ! -d "$LABELS_TRAIN" ]; then
        echo "⚠️ Não foram encontradas labels em: $LABELS_TRAIN; pulando balanceamento.";
    else
        # build class name list from data.names if present
        DATA_ROOT="$(dirname "$TRAIN_DIR")"
        NAMES_FILE="$DATA_ROOT/data.names"
        declare -a CLASS_NAMES
        if [ -f "$NAMES_FILE" ]; then
            mapfile -t CLASS_NAMES < "$NAMES_FILE"
        fi

        # collect images per class (use first class found in label file)
        declare -A files_by_class
        for lbl in "$LABELS_TRAIN"/*.txt; do
            [ -e "$lbl" ] || continue
            img_base=$(basename "$lbl" .txt)
            # read first class index from label, fallback to skip
            cls_index=$(awk 'NF{print $1; exit}' "$lbl")
            if [ -z "$cls_index" ]; then
                continue
            fi
            cls_index=${cls_index%%.*}
            # determine class name
            cls_name="${CLASS_NAMES[$cls_index]}"
            if [ -z "$cls_name" ]; then
                cls_name="class_$cls_index"
            fi
            files_by_class["$cls_name"]+="$img_base "
        done

        # for each class, copy percentage of images to val and copy corresponding labels
        for cls in "${!files_by_class[@]}"; do
            imgs_str=${files_by_class[$cls]}
            read -r -a imgs <<< "$imgs_str"
            total=${#imgs[@]}
            if [ $total -eq 0 ]; then
                continue
            fi
            to_copy=$((total * PERCENT / 100))
            if [ $to_copy -lt 1 ]; then
                to_copy=1
            fi
            echo "📋 $cls:"
            echo "   Total em train: $total"
            echo "   Copiando: $to_copy imagens para val"

            mkdir -p "$VAL_DIR/$cls"
            LABELS_VAL="$DATA_ROOT/labels/val"
            mkdir -p "$LABELS_VAL/$cls"

            # randomize list
            selected=( $(printf "%s\n" "${imgs[@]}" | shuf | head -n $to_copy) )
            copied=0
            for b in "${selected[@]}"; do
                # image path (try jpg then png)
                if [ -f "$IMAGES_TRAIN/$b.jpg" ]; then
                    src_img="$IMAGES_TRAIN/$b.jpg"
                elif [ -f "$IMAGES_TRAIN/$b.png" ]; then
                    src_img="$IMAGES_TRAIN/$b.png"
                else
                    continue
                fi
                dst_img="$VAL_DIR/$cls/$(basename "$src_img")"
                if [ ! -f "$dst_img" ]; then
                    cp "$src_img" "$dst_img"
                fi
                # copy label
                src_lbl="$LABELS_TRAIN/$b.txt"
                dst_lbl="$LABELS_VAL/$cls/$b.txt"
                if [ -f "$src_lbl" ] && [ ! -f "$dst_lbl" ]; then
                    mkdir -p "$(dirname "$dst_lbl")"
                    cp "$src_lbl" "$dst_lbl"
                fi
                ((copied++))
            done

            echo "   ✅ Copiadas: $copied"
            echo ""
        done
    fi
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 Contando imagens DEPOIS..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

train_total=0
val_total=0

echo "📁 TRAIN:"
for class in $TRAIN_DIR/*/; do
    class_name=$(basename "$class")
    count=$(ls "$class"*.{jpg,png} 2>/dev/null | wc -l)
    train_total=$((train_total + count))
    echo "  $class_name: $count"
done

echo ""
echo "📁 VAL:"
for class in $VAL_DIR/*/; do
    class_name=$(basename "$class")
    count=$(ls "$class"*.{jpg,png} 2>/dev/null | wc -l)
    val_total=$((val_total + count))
    echo "  $class_name: $count"
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📈 RESUMO:"
echo "   Train total: $train_total"
echo "   Val total:   $val_total"
echo "   Dataset total: $((train_total + val_total))"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "✅ Concluído!"
echo ""
echo "Próximo passo:"
echo "  make start_train_cuda  (para usar dataset COCO com anotações)"
echo "  OU"
echo "  make auto_annotate && make start_train_cuda_custom  (para usar dataset custom)"
