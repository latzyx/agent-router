# 训练模型

GitHub 普通仓库不接受超过 100MB 的单个文件，因此 MacBERT 权重不直接提交到 Git。
当前推荐的 v22 模型发布在
[GitHub Release model-v22](https://github.com/latzyx/agent-router/releases/tag/model-v22)。
该版本基于 v20/MacBERT，并加入跨 Agent 错误惩罚。五套零重叠挑战集平均
Macro-F1 为 `0.9404`、平均 Agent Accuracy 为 `0.9846`。

在仓库根目录执行下面的命令即可下载、校验并解压：

```bash
bash models/download_v22.sh
```

安装完成后目录结构为：

```text
models/
└── lazy-agent-router-macbert-v22/
    ├── config.json
    ├── model.safetensors
    ├── tokenizer.json
    ├── tokenizer_config.json
    ├── training_report.json
    ├── challenge_evaluation.json
    ├── challenge_v2_evaluation.json
    ├── challenge_v3_evaluation.json
    ├── challenge_v4_evaluation.json
    └── challenge_v5_evaluation.json
```

启动 API 服务后，前端模型选择框会自动发现 `lazy-agent-router-macbert-v22`。

模型归档 SHA-256：

```text
fc7c7b22bcbb2bf0551aeea1541c589bbd4f5498753f1799cf00361690d9da4c
```

需要回退到 v15 时，执行 `bash models/download_v15.sh`；v19 对照模型仍可通过
`bash models/download_v19.sh` 安装。
