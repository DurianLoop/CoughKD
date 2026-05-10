# Reading Library

更新时间：2026-05-10

## 命名规则

文件名格式：

```text
P{priority}_{year}_{type}_{short-topic}.pdf
```

- `P0_Core`: AAAI 方向必须先读。
- `P1_Baselines`: cough/COVID 任务 baseline 和 challenge 对照。
- `P2_Foundation_KD`: foundation model、KD、teacher/student 相关。
- `P3_Datasets_Risks`: 外部数据和风险背景。

原始 PDF 暂时保留在 `Reading/` 和 `Reading/literature/`，因为当前 Windows 权限阻止移动/删除。请优先阅读下面四个分类目录里的规范命名副本。

## P0 Core

1. `P0_2026_Model_DNDT_DNDF_CrossDataset_CoughCOVID.pdf`  
   先读。重点看：用了哪些数据集、是否 external validation、AUC/F1/specificity 怎么报、DNDT/DNDF 是否值得复现为 baseline。

2. `P0_2025_Model_PSR_3D_DCNN_CoughCOVID.pdf`  
   重点看：COUGHVID 选择规则、phase space reconstruction、3D CNN、是否存在数据协议弱点。

3. `P0_2025_Robustness_DriftAdaptive_CoughCOVID.pdf`  
   重点看：dynamic cough data、drift detection、domain adaptation、external AUC drop。它能帮我们写泛化动机。

4. `P0_2024_Risk_Confounding_AudioVsSymptoms_NMI.pdf`  
   重点看：为什么 cough-COVID 高 AUC 可能不可靠；我们需要 subgroup/confounder/external validation 来防守。

5. `P0_2023_DatasetProtocol_Coswara_ScientificData.pdf`  
   重点看：Coswara 官方数据协议、quality filtering、BLSTM baseline、AUC、sensitivity at 95% specificity。

6. `P0_2023_KD_TransformerToCNN_AudioTagging.pdf`  
   重点看：teacher-student 蒸馏、MobileNetV3 student、效率表、mAP vs 参数量。我们不是照搬，而是借鉴写法。

## P1 Baselines

1. `P1_2025_Model_VGGishTransfer_CoughClassification.pdf`  
   transfer learning cough model baseline。重点看 AUROC/AUPRC、专家标签、app-recorded cough protocol。

2. `P1_2025_Model_CHIO_U-Net_Handcrafted_CoughCOVID.pdf`  
   传统特征 + 优化算法路线。重点看它和 deep model 的比较方式。

3. `P1_2022_Benchmark_DiCOVA_Challenge.pdf`  
   challenge-style AUC 协议。重点看 benchmark split、top systems、AUC reporting。

## P2 Foundation and KD

1. `P2_2025_KD_EnsembleToStudent_RespiratorySound.pdf`  
   respiratory sound KD。重点看 ensemble teacher、soft labels、student 对照和效率指标。

2. `P2_2025_Foundation_Pretraining_RespiratoryAudio_arXiv.pdf`  
   respiratory foundation model 预训练。重点看 OPERA benchmark、pretraining data、teacher 候选。

3. `P2_2025_Foundation_Pretraining_RespiratoryAudio_Interspeech.pdf`  
   上一篇的 Interspeech 版本。重点看最终发表版表达和实验表。

4. `P2_2025_Foundation_AudioModels_HeartRespiratory.pdf`  
   foundation models 在心肺音上的评估。重点看 AST/BEATs/OPERA 等哪个更适合 respiratory audio。

5. `P2_2023_Foundation_BEATs_AudioPretraining.pdf`  
   BEATs teacher 背景。重点看 AudioSet 指标、pretraining 思路和我们引用方式。

## P3 Datasets and Risks

1. `P3_2024_Dataset_UK_PCRReferenced_VocalAudio.pdf`  
   PCR-referenced 大规模 vocal audio 数据。重点看 cough/exhalation/speech、PCR label、metadata、外部验证可行性。

## 按时间线理解领域

1. 2022: DiCOVA challenge 确立 cough/COVID acoustic benchmark 和 AUC 协议。
2. 2023: Coswara 官方数据协议和 BLSTM baseline；Transformer-to-CNN KD 给出效率论文写法；BEATs 成为强 audio teacher。
3. 2024: NMI confounding paper 提醒审稿人不要相信没有控制偏差的 cough-COVID 高分；UK PCR dataset 提供更强外部验证可能。
4. 2025: 3D CNN、drift-adaptive、VGGish、respiratory KD、respiratory foundation model 都出现，说明领域开始关注泛化、迁移和端侧。
5. 2026: DNDT/DNDF 跨数据集 cough-COVID 模型出现，成为我们近期必须对比/防守的 baseline。

## 每篇阅读笔记模板

```markdown
## Paper

- 数据集：
- 标签/任务：
- 数据划分：
- 模型：
- 输入特征：
- 指标：
- 最好结果：
- 是否 external validation：
- 是否 subject-disjoint：
- 是否控制 confounder/subgroup：
- 是否报告参数量/latency：
- 我们可以如何比较：
- 它的弱点：
```
