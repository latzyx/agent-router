# 训练模型

GitHub 普通仓库不接受超过 100MB 的单个文件，因此 MacBERT 权重不直接提交到 Git。
当前最新的 v19 模型发布在
[GitHub Release model-v19](https://github.com/latzyx/agent-router/releases/tag/model-v19)。
该版本基于 `bert-base-chinese`，分层验证集 Macro-F1 为 `0.9750`。由于尚未完成
独立挑战集评测，目前作为预发布模型；生产环境可继续保留 v15 作为回退版本。

在仓库根目录执行下面的命令即可下载、校验并解压：

```bash
bash models/download_v19.sh
```

安装完成后目录结构为：

```text
models/
└── lazy-agent-router-macbert-v19/
    ├── config.json
    ├── model.safetensors
    ├── tokenizer.json
    ├── tokenizer_config.json
    └── training_report.json
```

启动 API 服务后，前端模型选择框会自动发现 `lazy-agent-router-macbert-v19`。

模型归档 SHA-256：

```text
3ba77e04db619ffa017cde9583a03555dc8b3c8f82b3e31c6d11d4cd5f4a8bf1
```

需要回退到经过独立挑战集评测的 v15 时，执行 `bash models/download_v15.sh`。
