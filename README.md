# Audio Gender Benchmark

100条中英文语音的声音性别识别评测集。运行程序后会自动完成音频推理，并输出Accuracy、Macro-F1、UAR及各分组指标。

## 文件结构

```text
├── audio/                    # 50条TTS和50条真人语音
├── configs/                  # API、本地命令、本地Python模型配置
├── examples/                 # 本地Python模型适配器模板
├── scripts/
│   ├── backends/             # 可扩展推理后端
│   ├── run_benchmark.py      # 推理、断点续跑和自动评分
│   ├── evaluate_gender.py    # 指标计算
│   └── validate_benchmark.py # 数据完整性检查
├── benchmark_metadata.csv    # 音频路径、来源、语言和性别金标
└── reports/                  # 数据校验报告
```

代码使用Python 3.10以上版本，基础框架无需安装第三方依赖。

## 方式一：调用百炼API

```powershell
$env:DASHSCOPE_API_KEY="sk-你的APIKey"
python scripts/run_benchmark.py --config configs/qwen_api.json
```

默认模型为`qwen3.8-omni-flash`。使用百炼工作空间地址时设置：

```powershell
$env:DASHSCOPE_BASE_URL="https://你的WorkspaceId.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
python scripts/run_benchmark.py --config configs/qwen_api.json
```

## 方式二：调用本地命令

编辑`configs/local_command.json`中的模型名称和命令。命令必须在标准输出中返回`male`或`female`。

```json
"command": ["python", "path/to/infer.py", "--audio", "{audio_path}"]
```

然后运行：

```powershell
python scripts/run_benchmark.py --config configs/local_command.json
```

命令模板支持`{audio_path}`、`{id}`和`{prompt}`。

## 方式三：直接加载本地Python模型

在`examples/local_model_adapter.py`中加载模型并实现`predict()`，返回`male`或`female`。然后运行：

```powershell
python scripts/run_benchmark.py --config configs/local_python.json
```

模型只初始化一次，适合Transformers、自研推理代码或需要常驻GPU的模型。默认单并发，可在配置的`runner.workers`中调整。

## 输出

每个配置默认写入`results/<run_name>/`：

- `predictions.csv`：逐条预测、金标、正确性、耗时、原始回复和错误；
- `metrics.json`：总体以及按音频类型、数据来源、语言、长度和场景划分的指标；
- `run_info.json`：配置、后端、模型和完成数量。

请求失败后重新运行同一命令会自动续跑。加入`--overwrite`可从头运行，加入`--limit 2`可先测试两条音频。

推理时只向模型提供音频和统一任务提示，不提供转写、来源或性别金标。

## 单独评分

已有预测文件只需包含`id,predicted_gender`两列：

```powershell
python scripts/evaluate_gender.py --predictions predictions.csv --output metrics.json
```

## 数据校验

```powershell
python scripts/validate_benchmark.py
```

显示`BENCHMARK VALIDATION PASS`即表示数量、配额、路径、音频格式和重复检查通过。

## 扩展后端

新增后端时，实现`scripts/backends/base.py`中的`InferenceBackend`接口，并在`scripts/backends/__init__.py`注册。后端只负责把单条音频转换为文本回复；重试、并发、断点续跑、标签解析和评分由主程序统一处理。

## 数据来源

- TTS：Sambert；
- 真人中文：MagicData-RAMC；
- 真人英文：SLURP与AMI Meeting Corpus。

各来源仍受原始许可证约束，其中MagicData-RAMC和SLURP包含非商业使用限制。
