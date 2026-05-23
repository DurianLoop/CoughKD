# Shortcut-Audited CoughKD: 当前研究方向、已有证据与执行规划

日期：2026-05-23

## 1. 当前研究定位

本项目不再把核心问题表述为“咳嗽音频能否临床诊断 COVID-19”。这个 claim 风险过高，容易受到标签噪声、症状混杂、采集渠道差异、设备差异和外部验证不足的质疑。

当前更稳妥的研究定位是：

> 面向咳嗽音频筛查任务，研究如何将大型音频模型中的表征知识可靠地蒸馏到超轻量 student 中，并避免 teacher 中的数据集、录音质量、设备、症状等 shortcut 信息被无选择地传递给 student。

建议的论文方向名称：

> **Shortcut-Audited Distillation for Ultra-Light Cough Audio Models**

中文可写作：

> **面向超轻量咳嗽音频模型的 Shortcut 审计蒸馏**

核心问题不是“使用更多 teacher 或更多 KD loss”，而是：

> teacher 的哪些知识值得蒸馏？哪些知识可能只是数据集或采集条件 shortcut？如何在保留任务相关表征的同时降低 student 对不稳定 shortcut 的依赖？

## 2. 为什么普通 KD 不够

已有同学工作已经完成了一个基础 pipeline：

```text
Coswara cough split
→ PANNs CNN14 16 kHz teacher
→ DepthwiseStudent
→ vanilla response KD
→ in-domain test metrics
```

这条路线证明了：

- 数据处理、训练、KD、测试、checkpoint、prediction export 都已打通；
- 81M PANNs teacher 可以蒸馏到 20.7K DepthwiseStudent；
- 参数压缩比约 `3912x`；
- vanilla KD 在 Coswara 内部测试上有一定提升。

但它也暴露出问题：

- external COUGHVID 上 AUROC 接近随机；
- vanilla KD 的外部收益不稳定；
- 朴素 domain-adversarial fine-tuning 没有真正降低 domain probe AUC；
- 仅靠“多 teacher、多 loss、多指标”不足以构成实质创新。

因此，本项目的创新应聚焦于 **shortcut transfer**：

> KD 不只会传递有用的 cough representation，也可能传递 teacher 对数据集、录音质量、症状、采集协议等 spurious cues 的依赖。

## 3. 当前已有实验结果

### 3.1 Coswara PANNs baseline

实验目录：

```text
runs/stage1_panns_response_seed7
```

数据：

- Coswara cough-heavy + cough-shallow；
- subject-disjoint split；
- train `3671`，val `785`，test `791`；
- 5 类：`covid_positive`、`covid_recovered`、`exposed`、`healthy`、`respiratory_illness`。

结果：

| Model | Params | Accuracy | Macro-F1 | Macro OVR AUROC | Macro OVR AUPRC | ECE | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|
| PANNs CNN14 teacher | 81,043,476 | 0.3831 | 0.2694 | 0.6028 | 0.2706 | 0.0233 | 0.7402 |
| Depthwise CE-only | 20,717 | 0.3186 | 0.2069 | 0.5396 | 0.2334 | 0.0525 | 0.7843 |
| Depthwise KD | 20,717 | 0.3565 | 0.2363 | 0.5577 | 0.2507 | 0.0591 | 0.7583 |

结论：

- vanilla KD 在 Coswara 内部 test 上提升 AUROC `+0.0181`；
- 但提升不大，且不能说明外部泛化；
- 这应作为后续 shortcut-audited method 的 baseline。

### 3.2 COUGHVID external evaluation

COUGHVID manifest：

```text
manifests/coughvid_external.csv
```

筛选规则：

- `status` 有标签；
- `cough_detected >= 0.8`；
- 音频文件存在。

得到 `7310` 条：

| Label | Count |
|---|---:|
| healthy | 5628 |
| respiratory_illness | 1135 |
| covid_positive | 547 |

全量 COUGHVID external 结果：

| Model | Accuracy | Macro OVR AUROC | Macro OVR AUPRC | COVID AUROC | Healthy AUROC | Resp. Illness AUROC |
|---|---:|---:|---:|---:|---:|---:|
| CE-only student | 0.3211 | 0.5135 | 0.3403 | 0.5276 | 0.5111 | 0.5017 |
| KD student | 0.2384 | 0.5212 | 0.3428 | 0.5345 | 0.5134 | 0.5156 |

结论：

- KD 在全量 COUGHVID 上 AUROC 略高，但 accuracy 明显更低；
- external AUROC 约 `0.52`，说明 Coswara -> COUGHVID 泛化很弱；
- 这是“只看 in-domain 会高估方法价值”的直接证据。

### 3.3 COUGHVID adapt/test split

为了避免用 COUGHVID test 标签参与训练，将 COUGHVID 拆成：

```text
adapt: 3655
test: 3655
```

类别比例保持一致。

COUGHVID-test baseline：

| Model | Accuracy | Macro OVR AUROC | Macro OVR AUPRC |
|---|---:|---:|---:|
| CE-only student | 0.3332 | 0.5198 | 0.3449 |
| KD student | 0.2358 | 0.5142 | 0.3433 |

结论：

- 在这个干净 test 子集上，KD 低于 CE；
- vanilla KD 的 external benefit 不稳定；
- 后续方法不能只追求 KD-vs-CE in-domain improvement。

### 3.4 Coswara vs COUGHVID domain probe

实验目录：

```text
runs/domain_probe_coswara_coughvid_students
runs/domain_probe_coswara_coughvid_students_with_adv_v2
```

从 Coswara 和 COUGHVID 各抽 1500 条，用 student embedding 训练 probe：

- `dataset_domain`: 预测样本来自 Coswara 还是 COUGHVID；
- `task_label`: 预测任务标签。

结果：

| Model | Domain Probe AUC | Task Probe AUC |
|---|---:|---:|
| CE student | 0.7239 | 0.5837 |
| KD student | 0.7144 | 0.6346 |
| Domain-adv KD | 0.7163 | 0.6411 |

结论：

- CE 和 KD student embedding 都明显包含数据集来源信息；
- KD 提升了 task probe AUC，但没有显著降低 domain 可分性；
- domain-adversarial 版本也未真正压低 domain AUC。

### 3.5 朴素 domain-adversarial 试验

实验目录：

```text
runs/domain_adv_kd_student_e2_t2000
runs/external_coughvid_test_domain_adv_kd_e2_t2000
```

方法：

- 从 KD student checkpoint 出发；
- Coswara train 用真实标签继续 CE；
- COUGHVID-adapt 只用 domain label，不用任务标签；
- 加 gradient reversal domain head，试图降低 source/target 可分性。

COUGHVID-test 结果：

| Model | Accuracy | Macro OVR AUROC | Macro OVR AUPRC | Domain Probe AUC | Task Probe AUC |
|---|---:|---:|---:|---:|---:|
| KD student | 0.2358 | 0.5142 | 0.3433 | 0.7144 | 0.6346 |
| Domain-adv KD | 0.5989 | 0.5058 | 0.3387 | 0.7163 | 0.6411 |

结论：

- 朴素 domain-adversarial 让 accuracy 升高，但 AUROC 降低；
- domain probe AUC 没有下降；
- 这条路不能直接作为最终创新。

## 4. 当前关键判断

### 4.1 不能直接 claim

以下说法不宜作为论文主 claim：

- 我们证明咳嗽音频可以诊断 COVID；
- 我们提出多教师 KD；
- 我们提出 feature/relation/calibration KD；
- 我们通过 domain adversarial 消除了 shortcut；
- 我们的模型在外部数据集上显著 SOTA。

这些要么风险太高，要么已有文献充分覆盖，要么当前实验还不支持。

### 4.2 可以作为主线的 claim

更稳的 claim 是：

> Vanilla KD can improve in-domain or representation-level task readability, but under cough dataset shift, it does not reliably improve external generalization and may preserve domain-identifiable representations. We therefore study shortcut-audited distillation: estimating when teacher/student signals are likely to carry unstable shortcut cues and selectively controlling KD strength.

中文：

> 普通 KD 可以提升内部性能或表征中的任务可读性，但在咳嗽音频跨数据集漂移下，并不能稳定提升外部泛化，并且 student 表征仍然明显携带数据集来源信息。因此我们研究 shortcut 审计蒸馏：估计 teacher/student 信号何时可能携带不稳定 shortcut，并据此选择性控制 KD 强度。

## 5. 下一步方法设计

### 5.1 从朴素 domain adversarial 转向 sample-level shortcut risk

朴素 adversarial 失败说明：

- 只让 embedding 难以区分 domain 不够；
- 强行对齐 domain 可能伤害任务排序能力；
- 需要更细粒度地判断哪些样本/哪些信号值得蒸馏。

下一步方法应为：

```text
teacher/student shortcut audit
→ sample-level shortcut risk
→ shortcut-weighted KD
→ external/test + domain probe 双指标验证
```

### 5.2 Shortcut risk 的候选分量

对每个样本 `x` 定义：

```text
R(x) = f(
  domain_typicality,
  augmentation_instability,
  audio_quality_risk,
  teacher_confidence_or_entropy,
  teacher_student_disagreement
)
```

低风险样本：

- teacher 输出稳定；
- 音频质量正常；
- 不强烈暴露数据集来源；
- teacher/student 不严重冲突。

高风险样本：

- 轻微扰动后预测大变；
- 静音、clipping、能量异常；
- domain probe 置信度很高；
- teacher 过度自信但不校准；
- teacher 与 label 或 student 严重冲突。

训练目标：

```text
L = L_CE + lambda * (1 - R(x)) * L_KD
```

也可以扩展到：

```text
L = L_CE
  + lambda_response * w(x) * L_responseKD
  + lambda_embedding * w(x) * L_embeddingKD
  + lambda_consistency * L_student_consistency
```

### 5.3 第一版可实现方法

第一版不要做复杂多教师，先只做：

- PANNs teacher；
- DepthwiseStudent；
- Coswara train；
- COUGHVID adapt/test；
- sample-level weighting。

建议先实现三个权重：

1. **Audio quality weight**
   - duration；
   - RMS；
   - silence ratio；
   - clipping ratio。

2. **Augmentation stability weight**
   - 对同一音频做 gain/noise/time-shift；
   - teacher 或 student 预测变化越大，KD 权重越低。

3. **Domain risk weight**
   - 训练 Coswara vs COUGHVID-adapt domain probe；
   - domain probe 越确信，说明样本越 domain-specific；
   - KD 权重越低。

第一版方法名可写：

```text
SA-KD: Shortcut-Audited Knowledge Distillation
```

或更具体：

```text
SARKD: Shortcut-Aware Robust Knowledge Distillation
```

## 6. 实验计划

### Stage 1：复现与整理 baseline

已完成：

- Coswara PANNs teacher；
- CE-only student；
- vanilla KD student；
- COUGHVID external；
- COUGHVID adapt/test；
- domain probe；
- domain-adversarial negative result。

需要整理：

- 所有结果表；
- 训练命令；
- checkpoint 和 manifest hash；
- 数据过滤规则。

### Stage 2：实现 shortcut-weighted KD

需要新增：

- teacher/student prediction cache；
- audio quality feature cache；
- augmentation stability evaluator；
- domain probe score exporter；
- weighted KD training loop。

对比：

| Method | 说明 |
|---|---|
| CE-only | 不蒸馏 |
| vanilla KD | 旧 baseline |
| domain-adversarial KD | 已验证的负例/弱 baseline |
| quality-weighted KD | 只用音频质量权重 |
| stability-weighted KD | 只用增强稳定性 |
| domain-risk-weighted KD | 只用 domain risk |
| full shortcut-audited KD | 三者组合 |

### Stage 3：评估指标

性能指标：

- Macro OVR AUROC；
- Macro OVR AUPRC；
- per-class AUROC；
- per-class AUPRC；
- Macro-F1；
- balanced accuracy。

可靠性指标：

- ECE；
- Brier score；
- external AUROC drop；
- perturbation robustness；
- domain probe AUC。

部署指标：

- Params；
- model size；
- CPU latency；
- GPU latency；
- FLOPs/MACs。

### Stage 4：扩展 teacher/student

只有当 Stage 2 有明确正结果时再扩展：

Teacher：

- PANNs CNN14；
- HeAR；
- OPERA；
- BEATs。

Student：

- DepthwiseStudent；
- MobileNetV3-small；
- BC-ResNet。

避免机械穷举，主文只保留最能说明方法的 2-3 个组合。

## 7. 论文故事

推荐故事结构：

1. 咳嗽音频 COVID 诊断受到混杂因素和数据集漂移质疑；
2. 我们不主张发现 COVID 声学生物标志物，而研究轻量化咳嗽音频表征蒸馏；
3. 现有 vanilla KD 可提升 in-domain，但 external 很弱；
4. embedding probe 显示 student 表征明显携带 dataset-domain 信息；
5. 朴素 domain adversarial 不能解决问题；
6. 我们提出 shortcut-audited KD，用 sample-level risk 控制 teacher signal；
7. 目标是在保持任务表征可读性的同时降低 shortcut 依赖，并改善 external robustness。

## 8. 近期执行清单

1. 整理当前结果为统一表格；
2. 实现 shortcut score cache；
3. 实现 weighted KD training；
4. 先跑单 seed：
   - vanilla KD；
   - quality-weighted；
   - stability-weighted；
   - domain-risk-weighted；
   - full weighted；
5. 用 COUGHVID-test 评估；
6. 用 domain probe 验证 domain AUC；
7. 如果有效，再补 3 seeds；
8. 如果无效，转成 negative-result/analysis paper：`When Does KD Help Ultra-Light Cough Audio Models?`

## 9. 当前仓库中关键脚本

数据处理：

- `scripts/extract_coswara_windows.py`
- `scripts/build_coughvid_manifest.py`
- `scripts/split_coughvid_adapt_test.py`

训练与评估：

- `scripts/run_stage1_coswara_windows.cmd`
- `scripts/run_panns_coswara_windows.cmd`
- `scripts/evaluate_external_checkpoint.py`
- `scripts/train_domain_adversarial_student.py`

分析：

- `scripts/shortcut_probe_coswara.py`
- `scripts/domain_probe_students.py`
- `scripts/check_assets.py`

核心代码：

- `src/coughkd/audio.py`
- `src/coughkd/torch_models.py`
- `src/coughkd/metrics.py`
- `src/coughkd/cli.py`

## 10. 当前结论

当前最重要的结论是：

> 我们已经证明了基础 KD pipeline 可行，也证明了 vanilla KD 在外部 COUGHVID 上泛化很弱，并且 student embedding 中存在明显 dataset-domain 可读性。朴素 domain adversarial 不能解决该问题，因此下一步应做 shortcut-audited sample weighting，而不是继续堆 teacher/student/KD 组合。

