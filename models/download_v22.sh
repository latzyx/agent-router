#!/usr/bin/env bash

set -euo pipefail

MODEL_NAME="lazy-agent-router-macbert-v22"
ARCHIVE_NAME="${MODEL_NAME}.tar.gz"
DOWNLOAD_URL="https://github.com/latzyx/agent-router/releases/download/model-v22/${ARCHIVE_NAME}"
EXPECTED_SHA256="fc7c7b22bcbb2bf0551aeea1541c589bbd4f5498753f1799cf00361690d9da4c"

MODELS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARCHIVE_PATH="${MODELS_DIR}/${ARCHIVE_NAME}"
MODEL_DIR="${MODELS_DIR}/${MODEL_NAME}"

if [[ -f "${MODEL_DIR}/config.json" && -f "${MODEL_DIR}/model.safetensors" ]]; then
    echo "模型已经存在：${MODEL_DIR}"
    exit 0
fi

echo "正在下载 ${MODEL_NAME}……"
curl --fail --location --retry 3 --output "${ARCHIVE_PATH}" "${DOWNLOAD_URL}"
echo "${EXPECTED_SHA256}  ${ARCHIVE_PATH}" | sha256sum --check --status
echo "SHA-256 校验通过，正在解压……"
tar -xzf "${ARCHIVE_PATH}" -C "${MODELS_DIR}"
rm -f "${ARCHIVE_PATH}"
echo "模型安装完成：${MODEL_DIR}"
