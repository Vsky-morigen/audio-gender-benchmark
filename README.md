# 双任务语音评测工程

本工程包含 **200 条音频**，分为两项彼此独立的评测：

| 任务 | 题数 | 中文 / 英文 | 真人 / TTS | 输入与输出 |
| --- | ---: | ---: | ---: | --- |
| 声音性别识别 `gender` | 100 | 50 / 50 | 50 / 50 | 音频 → `male` 或 `female` |
| 主语言单选 `main_language` | 100 | 50 / 50 | 50 / 50 | 问题音频 + 四个文本选项 → `A`–`D` |

两项任务各自输出 Accuracy、Macro-F1、UAR、覆盖率和分组指标。不要把两项任务的准确率合成一个数字：标签空间和测量目标不同。

## 目录

```text
audio_gender_benchmark/
├── archives/
│   ├── gender.zip                  # 原有性别识别的100条音频
│   └── main_language.zip           # 主语言单选的100条音频
├── .audio_cache/audio/             # 运行时自动解压，仅保留在本地
├── data/
│   └── main_language.csv           # 四选一题目、答案与来源
├── benchmark_metadata.csv         # 原有性别金标与来源，路径不变
├── configs/                        # 每个任务的API、本地命令、本地Python配置
├── examples/local_model_adapter.py # 本地模型接入模板
├── scripts/
│   ├── backends/                   # 三种推理后端
│   ├── run_benchmark.py            # 单任务推理、重试、续跑、评分（原入口）
│   ├── run_all.py                  # 两项任务连续运行
│   ├── evaluate_gender.py          # 对现成预测文件单独评分（原入口）
│   ├── validate_benchmark.py       # 数据包完整性校验（原入口）
│   ├── audio_store.py              # 按需解压并核对SHA-256
│   └── package_audio.py            # 从WAV重建两个压缩包
└── tests/test_framework.py         # 离线框架测试
```

Python 3.10 及以上即可运行；框架本身仅使用标准库。本地模型适配器可按所接模型另装依赖。仓库只保存两个 ZIP；清单中的音频路径仍是原来的相对路径。校验或运行时会逐条解压至 `.audio_cache/audio/`，并按清单里的 SHA-256 核对文件。所有音频均为 16 kHz 单声道、16-bit PCM WAV。

新增或替换音频时，先把 WAV 放到清单所写的 `audio/` 路径，更新相应的 SHA-256，再运行 `python scripts/package_audio.py` 重建压缩包。评测时无需手动解压。

## 先校验数据

在 `audio_gender_benchmark` 目录运行：

```powershell
python scripts/validate_benchmark.py
python -m unittest discover -s tests -v
```

预期显示 `PACKAGE VALIDATION PASS: 200 rows, 200 unique WAVs`，并且四项框架测试通过。校验涵盖题数、语言与真人/TTS 配额、路径、音频格式、时长、SHA-256、题目去重和答案映射。

## 接入模型并运行

百炼/OpenAI 兼容 Responses API：先在运行环境中设置 `DASHSCOPE_API_KEY`，然后运行：

```powershell
python scripts/run_all.py
```

默认两个任务使用原有的 `configs/qwen_api.json` 和新加的 `configs/main_language_api.json`，模型名为 `qwen3.8-omni-flash`。如使用百炼工作空间地址，可设置 `DASHSCOPE_BASE_URL` 覆盖配置里的默认地址。不同服务的音频输入格式可能不同；接入其他 API 时可修改或新增 `scripts/backends/` 中的实现。API Key 不要写进配置或清单。

先各试两条：

```powershell
python scripts/run_all.py --limit 2
```

只运行一项任务：

```powershell
python scripts/run_benchmark.py --task gender --config configs/qwen_api.json
python scripts/run_benchmark.py --task main_language --config configs/main_language_api.json
```

本地可执行程序使用原有的 `configs/local_command.json` 或新增的 `configs/main_language_command.json`；把其中 `command` 改为模型实际命令，程序应向标准输出打印一个标签。本地常驻 Python 模型使用 `configs/local_python.json` 或 `configs/main_language_python.json`，并实现 `examples/local_model_adapter.py` 中的 `predict(audio_path, prompt, item)`。主语言任务只接受 `A`、`B`、`C`、`D`，性别任务只接受 `male`、`female`。

运行器只向模型发送音频、任务指令和主语言题的 A–D 选项。CSV 内的题干转写、答案、性别金标、来源都**不会**出现在模型请求里。主语言音频只朗读问题，选项作为文本输入。

## 输出与续跑

`run_all.py` 默认把两项预测分别写入 `results/combined/gender/`、`results/combined/main_language/`，并生成 `results/combined/summary.json`。`--limit 2` 自动写入独立的 `results/combined_smoke_2/`。每项任务都有：

- `predictions.csv`：逐条模型预测、金标、正确性、耗时、原始响应和失败原因；
- `metrics.json`：总体与分组指标，以及 `coverage` 和 `complete`；
- `run_info.json`：所用配置、模型、数据指纹和完成情况。

失败后原命令可重跑，已成功的题会续用。配置或清单变化时，运行器拒绝复用旧预测；使用新输出目录或加 `--overwrite` 重跑。部分结果的 Accuracy 仅针对已成功预测的样本，必须连同 `coverage` 一起阅读。

已有外部模型的预测文件时，文件至少包含 `id,predicted_label` 两列，然后运行：

```powershell
python scripts/evaluate_gender.py --task gender --predictions path/to/gender_predictions.csv --output path/to/gender_metrics.json
python scripts/evaluate_gender.py --task main_language --predictions path/to/answer_predictions.csv --output path/to/answer_metrics.json
```

预测不完整时显式加 `--allow-partial`。重复题号、未知题号、非法标签或非预期的缺失题目会报错。

## 来源与使用范围

- 性别任务：TTS 为 Sambert；真人中文来自 MagicData-RAMC，真人英文来自 SLURP 与 AMI Meeting Corpus。
- 主语言任务：真人中文来自 ODSQA，真人英文来自 SD-QA/VoiceBench；数学文本来自 SVAMP，逻辑与日期文本来自 BIG-Bench Hard，再由 Sambert 合成题干语音。

来源的许可证各不相同。性别真人来源含非商业使用限制；ODSQA 官方仓库未明确标出音频再分发许可。向甲方之外再分发或用于商业场景前，应分别核实原数据与合成服务的使用条款。当前音频已通过自动完整性检查，但尚未逐条人工听辨，正式评测前建议抽听英文发音以及数学和日期题的数字。
