# AAAI 对比模型、数据集与指标矩阵

更新时间：2026-05-10

## 0. 先回答核心问题

我们不应该只在 Schmid et al. 2023 的 Transformer-to-CNN KD 上做“小改”。那篇工作是 AudioSet 大规模音频标注任务，贡献是用复杂 Transformer teacher 离线蒸馏出 MobileNetV3-style CNN，并在 AudioSet 上同时提升 mAP 与效率。

它对我们的意义是：证明“large audio teacher -> efficient CNN student”这条路线是成立且被 ICASSP 接受的。但我们的 AAAI 方向不能写成“把它搬到咳嗽数据上”。更好的定位是：

> 面向咳嗽音频筛查的 reliability-aware foundation-teacher-to-edge-student distillation：在 subject-disjoint、quality-aware、external validation、confounder/subgroup 分析约束下，把 PANNs/AST/BEATs 等 foundation teacher 蒸馏成可部署学生模型。

换句话说，我们要同时做两件事：

1. 找到强 teacher + compact student 组合，尽量接近或超过同协议下 cough/COVID baseline。
2. 提供比现有 cough detection 论文更严格的泛化、偏倚、校准和效率证据。

如果只追单一数据集最高 AUC，会很危险，因为近几年很多 cough-COVID 论文报告 0.90+ AUC，但 Nature Machine Intelligence 2024 已经指出：很多高分可能来自症状、招募偏差和 demographic confounding，而不是 COVID 的稳定声学生物标志。

## 1. 强相关论文与对比信息

| 类别 | 论文/系统 | 模型/方法 | 使用数据集 | 评估指标 | 可作为我们哪类对比 |
|---|---|---|---|---|---|
| 数据协议 + baseline | Coswara, Scientific Data 2023 | log-mel + 2-layer BLSTM + weighted BCE；融合 9 类声音和 symptoms | Coswara；moderate/excellent 质量；COVID/non-COVID | AUC-ROC, sensitivity at 95% specificity, 95% CI, subgroup AUC | 必须复现/对齐的 primary baseline |
| 泛化/混杂警告 | Coppock et al., Nature Machine Intelligence 2024 | 多种 audio AI classifiers；和 symptoms-only 模型比较；matched confounder analysis | UK PCR-referenced vocal audio；外部复现数据 | ROC-AUC, matched ROC-AUC, utility, confounder analysis | 不是性能 baseline，而是审稿风险与可靠性标准 |
| 大规模外部数据 | UK COVID-19 Vocal Audio Dataset, Scientific Data 2024 | 数据集论文，不是模型论文 | 72,999 participants；cough/exhalation/speech；PCR-linked labels | 数据质量、PCR linkage、metadata completeness | 最强 external validation 候选 |
| 外部 cough 数据 | COUGHVID, Scientific Data 2021 | 数据集论文；含 expert cough annotation | COUGHVID 25k+ cough recordings, 2.8k+ expert-labelled subset | cough quality/diagnostic labels；常被用于 accuracy/AUC/F1 | external validation / pretraining / robustness |
| Challenge baseline | DiCOVA 2021/2022 | 多系统 challenge；传统特征、deep models、ensemble | Coswara subset；COVID acoustic challenge | AUC-ROC 为核心；也看 sensitivity/specificity | 最贴近 COVID cough benchmark 的对比 |
| Cough-specific ensemble | Cough2COVID-19, Scientific Reports 2024 | MFCC-MLP + Spectrogram-CNN + Chromagram-MLP + ensemble + CoughFeatureRanker | Coswara, Virufy, ComParE, COUGHVID | accuracy, sensitivity, specificity, AUC, confusion matrix | cough-specific strong claimed baseline |
| General audio KD | Schmid et al., ICASSP 2023 | Transformer teacher -> MobileNetV3 CNN student, offline KD | AudioSet | mAP, params, compute efficiency | KD 架构参照，不是 cough baseline |
| Foundation teacher | PANNs, TASLP 2020 | CNN14 / Wavegram-Logmel-CNN pretrained on AudioSet | AudioSet; transfer to audio tagging tasks | mAP, AUC 等 | teacher candidate |
| Foundation teacher | AST, Interspeech 2021 | Audio Spectrogram Transformer | AudioSet, ESC-50, Speech Commands | mAP, accuracy | teacher candidate |
| Foundation teacher | BEATs, ICML 2023 | acoustic tokenizer pretraining | AudioSet, ESC-50 等 | mAP, accuracy | strongest teacher candidate |
| Foundation teacher | HTS-AT / PaSST | efficient / hierarchical audio transformers | AudioSet 等 | mAP, accuracy | teacher candidates |
| Compact audio student | BC-ResNet | broadcasted residual CNN | Speech Commands / audio classification | accuracy, params, FLOPs | compact student baseline |
| Compact vision/audio student | MobileNetV3 / EfficientNet-B0 | lightweight CNN | ImageNet pretrain or audio spectrogram fine-tune | accuracy/AUC + params/FLOPs | compact student baseline |
| Speaker/identity style | ECAPA-TDNN small | attentive stats pooling + speaker embedding style | VoxCeleb / speech / possible cough embedding | EER, accuracy, AUC | optional identity-aware student |

## 2. 我们必须自己跑的模型对比

### 2.1 Traditional / Shallow Baselines

这些模型不一定强，但审稿人会问。它们成本低，应该最先补齐。

| 模型 | 输入特征 | 数据集 | 指标 |
|---|---|---|---|
| MFCC + Logistic Regression | MFCC mean/std | Coswara, external | AUROC, AUPRC, F1, sensitivity/specificity |
| MFCC + SVM | MFCC mean/std | Coswara, external | AUROC, AUPRC, F1 |
| MFCC + Random Forest | MFCC mean/std | Coswara, external | AUROC, F1, calibration |
| log-mel + Logistic Regression | log-mel pooled | Coswara, external | AUROC, AUPRC |
| TECC / handcrafted cepstral + LightGBM | TECC/MFCC/spectral | DiCOVA-style | AUROC, sensitivity at fixed specificity |

目的：证明我们的改进不是简单 handcrafted features 就能做到。

### 2.2 Cough/COVID Task Baselines

| 模型 | 数据集协议 | 需要复现/报告的指标 |
|---|---|---|
| Coswara BLSTM official baseline | Coswara quality-filtered COVID/non-COVID | AUC, sensitivity at 95% specificity, 95% CI, subgroup AUC |
| DiCOVA baseline/top system | DiCOVA protocol | AUC, sensitivity, specificity |
| Cough2COVID-19 style ensemble | Coswara/Virufy/ComParE/COUGHVID if available | accuracy, sensitivity, specificity, AUC |
| ResNet50 spectrogram baseline | Coswara/COUGHVID | AUC, recall/sensitivity, specificity |
| CNN/LSTM log-mel baseline | Coswara | AUC, macro-F1, AUPRC |

目的：这些是“咳嗽/COVID”领域审稿人会自然想到的模型。

### 2.3 Foundation Teacher Baselines

| Teacher | 输入 | 状态 | 指标 |
|---|---|---|---|
| PANNs CNN14 16 kHz | waveform/logmel internally | 已实现长训 | AUROC, AUPRC, F1, ECE, params |
| AST | log-mel spectrogram | 待实现 wrapper | AUROC, AUPRC, F1, params |
| BEATs | waveform/feature wrapper | 待实现 wrapper | AUROC, AUPRC, F1, params |
| HTS-AT | log-mel | optional | AUROC, AUPRC |
| PaSST | log-mel | optional | AUROC, AUPRC |

目的：证明 teacher 不是随机挑的，并且找出适合 cough 的 foundation representation。

### 2.4 Student Baselines

| Student | 为什么要比 | 指标 |
|---|---|---|
| DepthwiseStudent width 16/24/32/48 | 我们当前 student，需要容量曲线 | AUROC, AUPRC, F1, params, latency |
| MobileNetV3-small | Schmid et al. KD 路线的强相关 student | AUROC, params, FLOPs, latency |
| EfficientNet-B0 | 常见 spectrogram CNN | AUROC, params, FLOPs |
| BC-ResNet-small | audio-specific compact CNN | accuracy/AUROC, params |
| ECAPA-small | identity/embedding 强相关 | AUROC, EER if doing identity |

目的：避免“只拿一个很弱学生做蒸馏”的审稿质疑。

### 2.5 KD Ablation Baselines

| 设置 | 用途 |
|---|---|
| CE-only student | 必须有，是 KD 是否有效的唯一干净对照 |
| response KD only | 检查 soft logits 是否有用 |
| feature KD only | 检查中间表征是否有用 |
| response + feature KD | 常见组合 |
| response + feature + embedding KD | 检查 embedding 对齐 |
| full KD: response + feature + embedding + relation | 我们的 proposed |
| frozen teacher vs fine-tuned teacher | 判断 teacher adaptation 是否关键 |

目标线：

- KD 相比 CE-only：AUROC 至少 +0.01，macro-F1 至少 +0.01。
- Student/teacher AUROC retention：至少 95%。
- 至少 3 seeds，最终最好 5 seeds。

## 3. 数据集矩阵

| 数据集 | 是否必须 | 用途 | 标签/任务 | 主要风险 |
|---|---|---|---|---|
| Coswara | 必须 | primary in-domain train/val/test | COVID/non-COVID, healthy/non-healthy, 5-class secondary | 当前只用 cough 子集；需 quality filtering |
| DiCOVA | 强烈建议 | challenge-style benchmark | COVID/non-COVID | 访问协议、split 复现 |
| COUGHVID | 强烈建议 | external validation / cough quality / abnormality | expert cough label, self-report COVID label | COVID 标签噪声；不能强临床 claim |
| UK COVID Vocal Audio | 理想 | large PCR-referenced external validation | PCR-linked COVID status, cough/exhale/speech | 数据申请成本；隐私/访问 |
| Virufy | 可选 | external stress test | cough COVID labels | 样本量和协议需核查 |
| ComParE COVID cough | 可选 | challenge/benchmark | cough COVID | 访问和任务定义需核查 |
| ICBHI 2017 | 不做主任务 | respiratory pretraining/transfer | lung sound classes | 非 cough，不能直接证明 cough screening |

最低可投稿配置：

1. Coswara quality-aware binary COVID/non-COVID 主实验。
2. Coswara 5-class 作为 secondary。
3. COUGHVID 或 DiCOVA 至少一个 external validation。
4. 如果拿不到外部数据，AAAI 风险很高，建议转 ICASSP/INTERSPEECH 或把论文改成方法/效率为主。

## 4. 指标矩阵：尽可能多，但要有主次

### 主指标

| 指标 | 为什么重要 | 备注 |
|---|---|---|
| AUROC | 类别不平衡下最常用 | 主表必须有 |
| AUPRC | 正类稀少时更敏感 | COVID positive 通常少 |
| Macro-F1 | 多类/不平衡下比 accuracy 更合理 | 5-class 必须有 |
| Balanced accuracy | 平衡每类 recall | 可补充 |
| sensitivity at 95% specificity | 和 Coswara 官方对齐 | 筛查任务非常关键 |
| specificity at fixed sensitivity | 可从临床筛查角度解释 | 例如 90% sensitivity |

### 可靠性指标

| 指标 | 用途 |
|---|---|
| bootstrap 95% CI | 论文可信度 |
| ECE | 概率校准 |
| Brier score | 概率质量 |
| reliability diagram | 可视化校准 |
| subgroup AUROC | age/gender/location/mask/symptoms |
| external AUROC drop | 泛化能力 |
| matched/confounder-controlled AUC | 对抗 NMI 2024 类质疑 |

### 效率指标

| 指标 | 用途 |
|---|---|
| params | 参数压缩，当前强项 |
| model size | 部署成本 |
| FLOPs/MACs | 计算成本 |
| CPU latency | 真实边缘设备相关 |
| GPU latency | 训练/服务器推理参考 |
| peak memory | 端侧部署 |
| AUROC retention | student / teacher 性能保留 |
| Pareto curve | AUROC vs latency/params |

### 当前我们最容易写出亮点的指标

1. 参数压缩：PANNs 81M -> DepthwiseStudent 20.7k，约 3912x。
2. latency / model size：还没测，但应是强项。
3. retention：当前 92.34%，必须冲到 95%+。
4. external AUROC drop：如果能比 Cough2COVID-19/BLSTM 更稳，这是 AAAI 亮点。
5. calibration：很多 cough COVID 论文不认真做，这是可差异化贡献。

## 5. 我们应该怎样“超过 SOTA”

不要只追一个单表最高 AUC。更安全的 AAAI 叙事是：

1. 在 Coswara quality-aware COVID/non-COVID 上，接近或超过 Coswara BLSTM / Cough2COVID-style baseline。
2. 在外部数据集上，AUROC drop 更小。
3. 在参数量和 latency 上，显著优于 foundation teacher 和 ensemble。
4. 在 subgroup / calibration 上，报告更完整，避免过度临床 claim。

也就是说，我们可以主张：

- Not necessarily highest raw AUC on every dataset.
- Better efficiency-accuracy Pareto.
- Better robustness and evaluation discipline.
- Stronger teacher-student transfer under scarce/noisy cough labels.

## 6. 推荐阅读顺序

1. Schmid et al. 2023, Transformer-to-CNN KD：看 KD 架构和效率表怎么写。
2. Coswara Scientific Data 2023：看数据协议、quality filtering、AUC/sensitivity/subgroup。
3. Nature Machine Intelligence 2024：看 confounding 和审稿风险。
4. Cough2COVID-19 Scientific Reports 2024：看 cough-specific ensemble 和它的对比表。
5. DiCOVA challenge paper：看 challenge-style AUC 协议。
6. PANNs / AST / BEATs：看 teacher 候选和 AudioSet 指标。

## 7. 参考链接

- Coswara: https://www.nature.com/articles/s41597-023-02266-0
- Audio confounding / NMI 2024: https://www.nature.com/articles/s42256-023-00773-8
- UK COVID Vocal Audio Dataset: https://www.nature.com/articles/s41597-024-03492-w
- COUGHVID: https://www.nature.com/articles/s41597-021-00937-4
- Cough2COVID-19: https://www.nature.com/articles/s41598-024-76639-9
- Efficient Large-Scale Audio Tagging via Transformer-to-CNN KD: https://research.jku.at/en/publications/efficient-large-scale-audio-tagging-via-transformer-to-cnn-knowle
- PANNs: https://signalprocessingsociety.org/publications-resources/blog/panns-large-scale-pretrained-audio-neural-networks-audio-pattern
- AST: https://huggingface.co/papers/2104.01778
- BEATs: https://www.microsoft.com/en-us/research/publication/beats-audio-pre-training-with-acoustic-tokenizers/
