# 训练模型

GitHub 普通仓库不接受超过 100MB 的单个文件，因此 MacBERT 权重不直接提交到 Git。
当前推荐的 v15 模型发布在
[GitHub Release model-v15](https://github.com/latzyx/agent-router/releases/tag/model-v15)。

在仓库根目录执行下面的命令即可下载、校验并解压：

```bash
bash models/download_v15.sh
```

安装完成后目录结构为：

```text
models/
└── lazy-agent-router-macbert-v15/
    ├── config.json
    ├── model.safetensors
    ├── tokenizer.json
    ├── tokenizer_config.json
    ├── training_report.json
    ├── challenge_evaluation.json
    ├── challenge_v2_evaluation.json
    ├── challenge_v3_evaluation.json
    └── upload_evaluation.json
```

启动 API 服务后，前端模型选择框会自动发现 `lazy-agent-router-macbert-v15`。

模型归档 SHA-256：

```text
03baa4503c3fdc25a1c100d90b143b02bcd7f3ef0e1ff43b75047eeb14e73be4
```
