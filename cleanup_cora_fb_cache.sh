#!/usr/bin/env bash
set -euo pipefail

ROOT="/data2/wenb/ylu/HiGGs-GBD/HiGGs-main"
EXPECTED_DATA="$ROOT/data"

cd "$ROOT"

echo "========== 路径安全检查 =========="
ACTUAL_ROOT="$(pwd -P)"
ACTUAL_DATA="$(realpath "$ROOT/data")"

echo "项目目录: $ACTUAL_ROOT"
echo "data目录: $ACTUAL_DATA"

if [[ "$ACTUAL_ROOT" != "$ROOT" ]]; then
    echo "错误：当前项目路径不正确，停止执行。"
    exit 1
fi

if [[ "$ACTUAL_DATA" != "$EXPECTED_DATA" ]]; then
    echo "错误：data 的真实路径不正确，停止执行。"
    exit 1
fi

# 必须存在的原始数据
REQUIRED_FILES=(
    "data/cora/raw/cora_ml.npz"
    "data/fb/raw/facebook_large/musae_facebook_edges.csv"
    "data/fb/raw/facebook_large/musae_facebook_target.csv"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [[ ! -f "$file" ]]; then
        echo "错误：必须保留的原始文件不存在：$file"
        exit 1
    fi
done

# 需要保护并验证哈希的文件
PROTECTED_FILES=(
    "data/cora/raw/cora_ml.npz"
    "data/fb/raw/facebook_large/musae_facebook_edges.csv"
    "data/fb/raw/facebook_large/musae_facebook_target.csv"
    "data/fb/raw/facebook_large/citing.txt"
    "data/fb/raw/facebook_large/musae_facebook_edges.json"
    "data/fb/raw/facebook_large/musae_facebook_features.json"
    "data/fb/raw/facebook_large/README.txt"
)

declare -A HASH_BEFORE

echo
echo "========== 原始数据删除前哈希 =========="
for file in "${PROTECTED_FILES[@]}"; do
    if [[ -f "$file" ]]; then
        HASH_BEFORE["$file"]="$(sha256sum "$file" | awk '{print $1}')"
        printf '%s  %s\n' "${HASH_BEFORE[$file]}" "$file"
    else
        echo "辅助文件不存在，跳过：$file"
    fi
done

# Cora 旧派生文件：17个
CORA_DELETE_FILES=(
    "data/cora/raw/cora_G.pkl"
    "data/cora/raw/cora_partitions.csv"
    "data/cora/raw/cora_target.csv"
    "data/cora/raw/train.csv"
    "data/cora/raw/val.csv"
    "data/cora/raw/test.csv"

    "data/cora/processed/pre_filter.pt"
    "data/cora/processed/pre_transform.pt"
    "data/cora/processed/proc_train_h1.pt"
    "data/cora/processed/proc_val_h1.pt"
    "data/cora/processed/proc_test_h1.pt"
    "data/cora/processed/proc_train_h2.pt"
    "data/cora/processed/proc_val_h2.pt"
    "data/cora/processed/proc_test_h2.pt"
    "data/cora/processed/proc_train_X.pt"
    "data/cora/processed/proc_val_X.pt"
    "data/cora/processed/proc_test_X.pt"
)

# Facebook 旧派生文件
FB_DELETE_FILES=(
    "data/fb/raw/facebook_large/musae_facebook_partitions.csv"

    "data/fb/raw/train.csv"
    "data/fb/raw/val.csv"
    "data/fb/raw/test.csv"

    "data/fb/processed/pre_filter.pt"
    "data/fb/processed/pre_transform.pt"
    "data/fb/processed/proc_train_h1.pt"
    "data/fb/processed/proc_val_h1.pt"
    "data/fb/processed/proc_test_h1.pt"

    # 若存在旧 H2/Glue 缓存，也删除
    "data/fb/processed/proc_train_h2.pt"
    "data/fb/processed/proc_val_h2.pt"
    "data/fb/processed/proc_test_h2.pt"
    "data/fb/processed/proc_train_X.pt"
    "data/fb/processed/proc_val_X.pt"
    "data/fb/processed/proc_test_X.pt"
)

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
MANIFEST="$ROOT/ubuntu_data_cleanup_${TIMESTAMP}.tsv"

printf 'dataset\tpath\texists_before\tsize_bytes\tsha256\taction\n' > "$MANIFEST"

cora_count=0
cora_size=0
fb_count=0
fb_size=0

record_files() {
    local dataset="$1"
    shift

    local file size hash

    for file in "$@"; do
        if [[ -f "$file" ]]; then
            size="$(stat -c '%s' "$file")"
            hash="$(sha256sum "$file" | awk '{print $1}')"

            printf '%s\t%s\tyes\t%s\t%s\tdelete\n' \
                "$dataset" "$file" "$size" "$hash" >> "$MANIFEST"

            if [[ "$dataset" == "cora" ]]; then
                ((cora_count+=1))
                ((cora_size+=size))
            else
                ((fb_count+=1))
                ((fb_size+=size))
            fi
        else
            printf '%s\t%s\tno\t0\t-\talready_absent\n' \
                "$dataset" "$file" >> "$MANIFEST"
        fi
    done
}

record_files "cora" "${CORA_DELETE_FILES[@]}"
record_files "facebook" "${FB_DELETE_FILES[@]}"

echo
echo "========== 即将删除 =========="
echo "Cora：$cora_count 个文件，$cora_size bytes"
echo "Facebook：$fb_count 个文件，$fb_size bytes"
echo
awk -F '\t' 'NR > 1 && $3 == "yes" {print $1 ": " $2}' "$MANIFEST"

echo
echo "清单文件：$MANIFEST"
echo
read -r -p "确认只删除上述文件？请输入 yes 继续：" CONFIRM

if [[ "$CONFIRM" != "yes" ]]; then
    echo "已取消，没有删除任何文件。"
    exit 0
fi

echo
echo "========== 开始删除 =========="

delete_exact_files() {
    local file

    for file in "$@"; do
        if [[ -f "$file" ]]; then
            echo "删除：$file"
            rm -f -- "$file"

            if [[ -e "$file" ]]; then
                echo "错误：文件删除失败：$file"
                exit 1
            fi
        else
            echo "已不存在：$file"
        fi
    done
}

delete_exact_files "${CORA_DELETE_FILES[@]}"
delete_exact_files "${FB_DELETE_FILES[@]}"

echo
echo "========== 原始数据完整性验证 =========="

for file in "${PROTECTED_FILES[@]}"; do
    if [[ -n "${HASH_BEFORE[$file]+x}" ]]; then
        if [[ ! -f "$file" ]]; then
            echo "严重错误：受保护文件丢失：$file"
            exit 1
        fi

        hash_after="$(sha256sum "$file" | awk '{print $1}')"

        if [[ "$hash_after" != "${HASH_BEFORE[$file]}" ]]; then
            echo "严重错误：受保护文件哈希发生变化：$file"
            echo "删除前：${HASH_BEFORE[$file]}"
            echo "删除后：$hash_after"
            exit 1
        fi

        echo "PASS：$file"
    fi
done

echo
echo "========== 删除结果验证 =========="

remaining=0

for file in "${CORA_DELETE_FILES[@]}" "${FB_DELETE_FILES[@]}"; do
    if [[ -e "$file" ]]; then
        echo "仍然存在：$file"
        remaining=$((remaining + 1))
    fi
done

if [[ "$remaining" -ne 0 ]]; then
    echo "错误：仍有 $remaining 个待删除文件存在。"
    exit 1
fi

# 目录必须保留
REQUIRED_DIRS=(
    "data/cora/raw"
    "data/cora/processed"
    "data/fb/raw"
    "data/fb/raw/facebook_large"
    "data/fb/processed"
    "data/sbm"
    "saved_models"
)

for dir in "${REQUIRED_DIRS[@]}"; do
    if [[ ! -d "$dir" ]]; then
        echo "错误：目录意外丢失：$dir"
        exit 1
    fi
done

echo
echo "========== Git 状态 =========="
git status --short || true

echo
echo "注意：本脚本没有执行 git add、commit 或 push。"
echo
echo "========== 清理完成 =========="
echo "Cora 删除：$cora_count 个文件，$cora_size bytes"
echo "Facebook 删除：$fb_count 个文件，$fb_size bytes"
echo "清单：$MANIFEST"
echo "原始数据哈希：全部通过"
