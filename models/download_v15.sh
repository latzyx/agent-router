#!/usr/bin/env bash

# 发生下载、校验或解压错误时立即退出，避免留下看似完整的损坏模型。
set -euo pipefail

MODEL_NAME="lazy-agent-router-macbert-v15"
ARCHIVE_NAME="${MODEL_NAME}.tar.gz"
DOWNLOAD_URL="https://github.com/latzyx/agent-router/releases/download/model-v15/${ARCHIVE_NAME}"
EXPECTED_SHA256="03baa4503c3fdc25a1c100d90b143b02bcd7f3ef0e1ff43b75047eeb14e73be4"

# 脚本无论从哪个工作目录运行，都将模型安装到脚本所在的 models 目录。
MODELS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARCHIVE_PATH="${MODELS_DIR}/${ARCHIVE_NAME}"
MODEL_DIR="${MODELS_DIR}/${MODEL_NAME}"

if [[ -f "${MODEL_DIR}/config.json" && -f "${MODEL_DIR}/model.safetensors" ]]; then
    echo "模型已经存在：${MODEL_DIR}"
    exit 0
fi

echo "正在下载 ${MODEL_NAME}……"
curl --fail --location --retry 3 --output "${ARCHIVE_PATH}" "${DOWNLOAD_URL}"

# sha256sum 校验失败会立即终止，损坏的归档不会被解压成模型目录。
echo "${EXPECTED_SHA256}  ${ARCHIVE_PATH}" | sha256sum --check --status
echo "SHA-256 校验通过，正在解压……"

tar -xzf "${ARCHIVE_PATH}" -C "${MODELS_DIR}"
rm -f "${ARCHIVE_PATH}"

echo "模型安装完成：${MODEL_DIR}"
