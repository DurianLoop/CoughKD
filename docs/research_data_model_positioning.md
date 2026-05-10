# CoughKD 数据、模型与论文定位备忘录

更新时间：2026-05-10

## 1. Coswara Cough 数据是什么

Coswara 是 Indian Institute of Science 等团队构建的公开呼吸声音与症状数据集，论文发表于 Scientific Data 2023。它是一个相对权威、可引用的公开数据集，适合作为咳嗽/呼吸音筛查研究的主要公开数据之一，但不能单独支撑“临床有效”或“泛化可靠”的强结论。

官方数据构成：

- 2,635 名受试者。
- 录制时间覆盖 2020-04 到 2022-02。
- 受试者 COVID 状态：1,819 negative，674 positive，142 recovered。
- 每名受试者理论上提供 9 类声音：deep/shallow breathing、deep/shallow cough、3 个 sustained vowel、normal/fast counting。
- metadata 包含 demographics、symptoms、respiratory ailments、comorbidities、test status/date/type、mask usage、smoking status、人工音频质量标注等。

当前 CoughKD 仓库实际使用的数据子集：

- 只使用 Coswara 的 `cough-heavy` 和 `cough-shallow`。
- 当前任务是分类任务，不是检测任务，也不是声纹验证任务。
- 当前标签是 5 类：`healthy`、`covid_positive`、`exposed`、`respiratory_illness`、`covid_recovered`。
- 原始 cough manifest：5,491 recordings / 2,746 subjects。
- 过滤后：5,247 recordings / 2,635 subjects。
- subject-disjoint split：train 3,671 / val 785 / test 791。

结论：Coswara 是可靠的公开研究数据集，但当前只使用 cough 子集，信息利用不充分。AAAI 论文里应把它定义为 primary in-domain dataset，而不是唯一证据。

## 2. 当前数据处理是否合理

当前代码位置：

- 数据集导入：[../src/coughkd/datasets.py](../src/coughkd/datasets.py)
- manifest 校验、过滤、划分：[../src/coughkd/manifest.py](../src/coughkd/manifest.py)
- 音频读取与轻量特征：[../src/coughkd/audio.py](../src/coughkd/audio.py)
- PyTorch dataset：[../src/coughkd/torch_models.py](../src/coughkd/torch_models.py)
- 命令行入口：[../src/coughkd/cli.py](../src/coughkd/cli.py)
- 当前数据和结果记录：[data_and_results.md](data_and_results.md)

当前处理方式：

- 从 Coswara metadata 和目录结构生成统一 manifest。
- 删除 `under_validation` 标签。
- 删除短于 0.5 秒的音频。
- 按 subject 划分 train / val / test，避免同一受试者泄漏。
- 读取 16-bit PCM WAV，转 mono。
- 重采样到配置采样率。
- 对学生模型路径生成 `log_mel_like` 特征。
- PANNs teacher 路径使用 waveform 输入。

合理的地方：

- subject-disjoint split 是必须的，而且做对了。咳嗽任务里如果同一人的录音同时出现在 train/test，会高估结果。
- 删除 `under_validation` 和极短音频是合理的基础清洗。
- 记录 manifest hash、split report、label distribution 是可复现论文的好习惯。
- 用 AUROC、macro-F1 而不是只看 accuracy 是合理的，因为类别明显不平衡。

不足和风险：

- 还没有利用 Coswara 官方的人工质量标注。官方论文中只选择 moderate/excellent 质量录音做 COVID 分类实验，这是更干净的协议。
- 当前 `log_mel_like` 是标准库下的轻量 log-energy band，不是真正的 log-mel frontend。正式实验应使用 torchaudio/librosa/PANNs/AST 对齐的 log-mel 特征。
- 当前 5 类任务不一定是最科学的主任务。Coswara 官方主实验是 COVID vs non-COVID binary classification，并报告 AUC 和 sensitivity at 95% specificity。
- 当前只截取前 `max_duration_sec` 秒，可能偏向开头静音/噪声。应补 random crop、center crop、multi-crop inference。
- 当前没有充分利用 metadata 做偏倚分析，例如 gender、age、country、mask、vaccination/date/device。
- 当前没有外部验证。单一 Coswara 结果不能证明跨设备、跨国家、跨采集协议泛化。

建议改进的数据协议：

1. Primary task 改为二分类：`covid_positive` vs non-COVID，`healthy` vs non-healthy 作为补充。
2. 5-class 保留为 secondary task，不作为主 claim。
3. 加入 Coswara quality annotation：主实验只用 excellent + moderate，poor quality 作为 robustness stress test。
4. 对 cough-heavy、cough-shallow、heavy+shallow 分别报告。
5. 使用真实 log-mel：16 kHz、25 ms window、10 ms hop、64/128 mel bins。
6. 训练使用 random crop，验证/测试使用 center crop 或 multi-crop aggregation。
7. 报告 subgroup：gender、age bucket、location/country、mask usage、recording date。
8. 至少引入一个外部数据集做验证。

## 3. 是否需要其他数据集

需要。单独 Coswara 不够。

| 数据集 | 作用 | 原因 | 风险 |
|---|---|---|---|
| COUGHVID | 外部验证 / 预训练 / cough quality | 25k+ crowdsourced cough recordings，2,800+ expert-labeled recordings | 自报告 COVID 标签噪声较大，expert labels 多是 cough quality/abnormality |
| DiCOVA 2021/2022 | benchmark 对比 | 基于 Coswara subset 的 challenge，有公开协议和 AUC 对照 | 访问和协议需确认，任务偏 COVID |
| Cambridge COVID-19 Sounds | 外部泛化 | 采集协议不同，适合测 domain shift | 访问和标签质量需核查 |
| Virufy / Coswara variants | 补充外部测试 | 可作为小规模 stress test | 样本量和可获得性不稳定 |
| ICBHI 2017 | respiratory transfer，不建议主实验 | 呼吸音/肺音权威数据，可做辅助预训练 | 不是 cough screening，不能直接对比 |

最低 AAAI 配置：

- Coswara：主训练 + in-domain test。
- COUGHVID 或 DiCOVA：外部验证。
- 如果外部数据来不及完整训练，至少做 frozen representation / fine-tune / external test 三选一。

## 4. 现有模型到底是不是我们的贡献

### Compact ConvTeacher

Compact ConvTeacher 是仓库内的工程 baseline。它的作用是证明训练、验证、测试、checkpoint、prediction export、KD 流程能跑通，并提供一个小型 CNN teacher 对照。它不应该成为论文主比较对象，也不应该被包装成强 teacher。

### PANNs CNN14 16 kHz Teacher

PANNs CNN14 不是我们提出的模型。它是已有的 AudioSet 预训练音频 CNN foundation model。

当前做法：

- 加载官方 PANNs CNN14 16 kHz checkpoint。
- 冻结 backbone。
- 训练一个 cough-label classifier head。

它在论文中应被表述为 pretrained audio teacher，而不是 proposed architecture。

### DepthwiseStudent

DepthwiseStudent 也不是全新思想。Depthwise separable convolution / MobileNet-style student 是已有的轻量网络范式。

当前仓库中的具体 student 是我们自己实现的一个极小模型：

- stem conv
- 3 个 depthwise separable blocks
- global pooling
- 64-d embedding
- classifier
- 20,717 parameters

它可以作为 proposed compact student instance，但不能声称 depthwise CNN 本身是新发明。

### 我们真正可以主张的贡献

如果只把 PANNs teacher 和 DepthwiseStudent 简单接起来，贡献偏弱。AAAI 版本需要把贡献定义成系统性方法，而不是“模型拼装”：

1. 面向咳嗽音频筛查的 foundation-teacher-to-edge-student 蒸馏框架。
2. 多层次 KD：response KD、feature KD、embedding KD、relation KD。
3. 数据可靠性协议：subject-disjoint split、quality-aware filtering、metadata/subgroup audit、external validation。
4. 部署效率证据：参数量、模型大小、latency、teacher-student retention。
5. 严格对照：CE-only student、KD student、compact CNN teacher、pretrained teacher、classical ML baselines、lightweight CNN baselines。

## 5. 当前模型和指标结果

| Model | Params | Accuracy | Macro-F1 | Macro OVR AUROC |
|---|---:|---:|---:|---:|
| ConvTeacher | 110,277 | 0.390645 | 0.209663 | 0.577184 |
| DepthwiseStudent + KD, compact teacher | 20,717 | 0.256637 | 0.204042 | 0.580959 |
| PANNs CNN14 16 kHz teacher | 81,043,476 | 0.374210 | 0.271545 | 0.613420 |
| DepthwiseStudent CE-only | 20,717 | 0.445006 | 0.234462 | 0.563910 |
| DepthwiseStudent KD | 20,717 | 0.361568 | 0.240593 | 0.566423 |

参数量是可以比较的重要指标，而且是当前项目最有潜力的卖点：

- PANNs teacher: 81,043,476 params
- DepthwiseStudent: 20,717 params
- 参数压缩约 3,912x

但注意：压缩率高不等于论文强。必须证明学生模型在性能上有足够 retention：

- 当前 KD student / PANNs teacher AUROC retention = 92.34%
- 目标至少 95%
- 当前 KD 比 CE-only 只高 +0.002513 AUROC
- 目标至少 +0.01 AUROC，且跨 3-5 seeds 稳定

## 6. 应该和哪些方法比较

### 数据处理 / 传统特征 baseline

必须比较：

- MFCC + SVM
- MFCC + Random Forest
- log-mel + Logistic Regression
- log-mel + Random Forest
- TECC / handcrafted cepstral features + LightGBM，参考 DiCOVA/PANACEA 风格

### 咳嗽/COVID 音频任务 baseline

必须比较：

- Coswara 官方 BLSTM COVID/non-COVID classifier
- DiCOVA baseline / top systems
- COUGHVID cough quality / abnormality classification baselines

注意：不同数据集、不同 split 的结果不能直接横向说 SOTA，只能在同一协议下复现或引用为背景。

### Audio foundation teacher

建议比较：

- PANNs CNN14
- AST
- BEATs
- HTS-AT
- PaSST
- Whisper encoder, optional

### Compact student

建议比较：

- 当前 DepthwiseStudent
- Depthwise width sweep: 16 / 24 / 32 / 48
- MobileNetV3-small
- EfficientNet-B0
- BC-ResNet-small
- ECAPA-small

### KD baseline

必须比较：

- CE-only student
- response KD only
- feature KD only
- response + feature KD
- response + feature + embedding KD
- full KD, including relation KD
- teacher fine-tuning vs frozen teacher

### 效率对照

必须报告：

- params
- model size
- FLOPs / MACs
- CPU latency
- GPU latency
- possibly mobile latency
- AUROC-retention vs compression ratio

这部分是 CoughKD 最自然的论文卖点：不是比最大 teacher 更准，而是在接近 teacher 的前提下大幅压缩。

## 7. 近年论文与公开资料给我们的启示

Coswara 官方论文：

- 使用多模态 respiratory sounds + symptoms，而不只是 cough。
- 使用 moderate/excellent 质量录音。
- 主任务是 COVID vs non-COVID。
- 使用 log-mel spectrogram + BLSTM。
- 使用加权 BCE 处理类别不平衡。
- 报告 AUC 和 sensitivity at 95% specificity。
- 做 subgroup/bias analysis。

COUGHVID：

- 提供 25k+ crowdsourced cough recordings。
- 有 2,800+ expert-labeled recordings。
- 适合作为外部验证或 cough quality / abnormality 任务。
- 不宜直接混用自报告 COVID 标签做强临床结论。

DiCOVA：

- 是 COVID acoustics challenge。
- AUC 是核心指标。
- 有传统特征 + boosting、encoder-decoder、deep learning 等系统。
- 如果能复现 DiCOVA protocol，会显著增强论文可信度。

AST / BEATs / PANNs：

- 它们是通用音频 foundation teacher，不是咳嗽领域专用模型。
- 用它们做 teacher 是合理的，但论文贡献需要体现在 adaptation、distillation、evaluation protocol 和 efficiency 上。

## 8. 当前最应该修改的实验路线

1. 把 Coswara 主任务改成 binary COVID/non-COVID，同时保留 5-class。
2. 加入 quality annotation 过滤。
3. 替换或补充真实 log-mel frontend。
4. 实现 AST teacher wrapper。
5. 实现 BEATs teacher wrapper。
6. 对 PANNs 先做 KD hyperparameter sweep。
7. 核心配置跑 3 seeds。
8. 引入 COUGHVID 或 DiCOVA 做外部验证。
9. 做效率表：params、model size、latency。
10. 写论文时避免 clinical diagnosis claim，只写 screening / research support。

## 9. 推荐论文定位

不建议的定位：

> We propose PANNs + DepthwiseStudent for cough classification.

这个说法太弱，容易被认为是简单组合。

建议定位：

> We propose a reliability-aware teacher-student distillation framework for efficient cough audio screening, combining foundation audio teachers, multi-level knowledge distillation, subject-disjoint and quality-aware evaluation, and deployment-oriented efficiency analysis.

核心卖点：

- 不是发明 PANNs。
- 不是发明 depthwise convolution。
- 是把 foundation audio teacher 转化成可部署 cough screening student，并用严格协议证明性能、压缩率和可靠性。

## 10. 参考资料

- Coswara Scientific Data 2023: https://www.nature.com/articles/s41597-023-02266-0
- COUGHVID Scientific Data 2021: https://www.nature.com/articles/s41597-021-00937-4
- DiCOVA 2021 summary: https://www.sciencedirect.com/science/article/abs/pii/S0885230821001157
- PANACEA DiCOVA 2021 system: https://www.ugr.es/~joseangl/publication/kamble-panacea-2021/
- AST paper page: https://huggingface.co/papers/2104.01778
- AST documentation: https://huggingface.co/docs/transformers/model_doc/audio-spectrogram-transformer
- BEATs Microsoft Research: https://www.microsoft.com/en-us/research/publication/beats-audio-pre-training-with-acoustic-tokenizers/
