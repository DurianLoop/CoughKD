# 2025-2026 强相关论文补充清单

更新时间：2026-05-10

## 推荐优先阅读

| 优先级 | 论文 | 年份 | 方法/模型 | 数据集 | 指标 | 为什么和 CoughKD 相关 |
|---:|---|---:|---|---|---|---|
| 1 | Robust COVID-19 detection from cough sounds using deep neural decision tree and forest | 2026 | DNDT / DNDF, RFECV, Bayesian optimization, SMOTE, threshold moving | Cambridge asymptomatic/symptomatic, Coswara, COUGHVID, Virufy, Virufy+NoCoCoDa, merged set | AUC, accuracy, precision, recall, F1, specificity | 目前最像“跨数据集 cough-COVID SOTA 对比”的论文，必须重点读 |
| 2 | Cough acoustic analysis using artificial intelligence for COVID-19 detection: Lima vs Montreal | 2026 | XGBoost + deep neural network；audio-only / clinical-only / combined | Lima, Peru and Montreal, Canada prospective cohorts, NAAT labels | internal/external validation, screening performance | 强调跨人群泛化，适合作为我们 external validation 论证参照 |
| 3 | Advanced COVID-19 detection using cough signals with space reconstruction and 3D DCNN | 2025 | Phase Space Reconstruction + 3D CNN | COUGHVID, 8,407 high-quality samples | accuracy, recall, specificity, precision, F1 | cough-specific deep model，报告非常高，需作为强 claimed baseline 谨慎对比 |
| 4 | A Comprehensive Drift-Adaptive Framework for COVID-19 Detection From Dynamic Cough Audio Data | 2025 | CNN baseline + MMD/CUSUM drift detection + UDA/active learning | COVID-19 Sounds, Coswara | AUC-ROC, balanced accuracy, accuracy, sensitivity, specificity, F1 | 非常适合我们写“泛化/漂移/持续可靠性” |
| 5 | Deep learning-based cough classification using application-recorded sounds: VGGish transfer learning | 2025 | VGGish transfer learning + cough detection/classification heads | 私有 smartphone cough data, asthma/COPD/pneumonia/healthy, 7 experts | accuracy, AUROC, AUPRC, Grad-CAM | 和我们的 teacher transfer / cough abnormality 分类很贴 |
| 6 | Improving Respiratory Sound Classification with Architecture-Agnostic KD from Ensembles | 2025 | ensemble teacher -> student soft-label KD | respiratory sound datasets | classification metrics, efficiency | 和我们的 KD 主线直接相关，虽然不是 COVID cough |
| 7 | Assessing the Utility of Audio Foundation Models for Heart and Respiratory Sound Analysis | 2025 | AST / BEATs / BYOL-A / M2D / OPERA 等 foundation features | 4 heart/respiratory tasks | sensitivity, specificity, task score | 证明 foundation models 在 respiratory audio 上需要按数据质量评估 |
| 8 | Towards Pre-training an Effective Respiratory Audio Foundation Model | 2025 | respiratory audio pretraining; AudioSet + respiratory data | OPERA benchmark | OPERA task metrics | 说明 AudioSet teacher 与 respiratory pretraining 怎么结合 |
| 9 | Automatic detection and prediction of COVID-19 in cough audio using CHIO | 2025 | fuzzy preprocessing + U-Net segmentation + ZM/GLCM + EDNN-CHIO | COUGHVID | error metrics, comparative simulation | 可作为传统特征/优化型 cough-COVID baseline |
| 10 | Respiratory sounds classification by fusing time-domain and 2D spectral features | 2025 | time-domain branch + mel/spectral branch + co-attention | respiratory sound datasets | SOTA classification metrics | 对我们设计 student 或多分支模型有启发 |

## 论文链接

1. Robust COVID-19 detection from cough sounds using deep neural decision tree and forest, Expert Systems with Applications 2026  
   https://www.sciencedirect.com/science/article/pii/S0957417426001491

2. Cough acoustic analysis using artificial intelligence for COVID-19 detection: a comparative study of patient cohorts from Lima, Peru and Montreal, Canada, Annals of Epidemiology 2026  
   https://www.sciencedirect.com/science/article/pii/S1047279726000748

3. Advanced COVID-19 detection using cough signals with space reconstruction and 3D deep convolutional neural networks, Scientific Reports 2025  
   https://www.nature.com/articles/s41598-025-29633-8

4. A Comprehensive Drift-Adaptive Framework for Sustaining Model Performance in COVID-19 Detection From Dynamic Cough Audio Data, JMIR 2025  
   https://www.jmir.org/2025/1/e66919

5. Deep learning-based cough classification using application-recorded sounds: a transfer learning approach with VGGish, BMC Medical Informatics and Decision Making 2025  
   https://link.springer.com/article/10.1186/s12911-025-03065-w

6. Improving Respiratory Sound Classification with Architecture-Agnostic Knowledge Distillation from Ensembles, arXiv 2025  
   https://arxiv.org/abs/2505.22027

7. Assessing the Utility of Audio Foundation Models for Heart and Respiratory Sound Analysis, EMBC 2025  
   https://www.kecl.ntt.co.jp/people/ohishi.yasunori/publication/2025d/

8. Towards Pre-training an Effective Respiratory Audio Foundation Model, Interspeech 2025  
   https://www.isca-archive.org/interspeech_2025/niizumi25_interspeech.html

9. Automatic detection and prediction of COVID-19 in cough audio signals using coronavirus herd immunity optimizer algorithm, Scientific Reports 2025  
   https://www.nature.com/articles/s41598-025-85140-w

10. Respiratory sounds classification by fusing the time-domain and 2D spectral features, Biomedical Signal Processing and Control 2025  
    https://www.sciencedirect.com/science/article/pii/S1746809425003015

11. Lung Sound Classification Model for On-Device AI, Applied Sciences 2025  
    https://www.mdpi.com/3466582

12. CycleGuardian: automatic respiratory sound classification based on improved deep clustering and contrastive learning, Complex & Intelligent Systems 2025  
    https://link.springer.com/article/10.1007/s40747-025-01800-4

13. A Generative AI-Based Framework for COVID-19 Screening from Cough Audio Signals, JoVE 2026  
    https://pubmed.ncbi.nlm.nih.gov/41911284/

## 对 CoughKD 的直接启发

### 必须比较的近期模型

- DNDT / DNDF + RFECV + SMOTE + threshold moving, 来自 Expert Systems with Applications 2026。
- PSR + 3D DCNN, 来自 Scientific Reports 2025。
- VGGish transfer learning cough classifier, 来自 BMC 2025。
- Drift-adaptive CNN with UDA/active learning, 来自 JMIR 2025。
- Ensemble teacher -> student soft-label KD, 来自 arXiv 2025。
- OPERA / AudioSet foundation model feature extractors, 来自 EMBC/Interspeech 2025。

### 我们的差异化空间

- 不只追单数据集高 AUC，而是做 cross-dataset / external validation。
- 不只报告 accuracy，而是报告 AUROC、AUPRC、macro-F1、balanced accuracy、sensitivity at fixed specificity、ECE、Brier、subgroup AUC、external AUC drop。
- 不只做大模型，而是报告 teacher-student retention、params、FLOPs、model size、latency。
- 不只用 generic AudioSet teacher，而是比较 PANNs / AST / BEATs / OPERA-style respiratory foundation models。

### 需要警惕的点

- 2025-2026 有些 cough-COVID 论文报告非常高的 accuracy/AUC，但未必有严格 subject-disjoint、external validation、confounder control。
- 写 AAAI 时不要和不同 split、不同标签质量的论文直接说“超过 SOTA”。更稳妥的表达是：在同一协议复现强 baseline；在外部验证、效率、校准和泛化上更强。
