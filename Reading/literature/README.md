# Literature PDFs and Reading Priority

更新时间：2026-05-10

## 阅读优先级

### P0：必须先读

1. `2026_DNDT_DNDF_cross_dataset_cough_covid_arxiv.pdf`  
   主题：跨数据集 cough-COVID 检测，DNDT/DNDF + 特征选择 + SMOTE + threshold moving。  
   为什么先读：最像我们需要对比的 2025/2026 cough-COVID 模型论文，重点看 datasets、metrics、cross-dataset protocol、表格组织。

2. `2025_PSR_3D_DCNN_cough_covid_scirep.pdf`  
   主题：phase space reconstruction + 3D CNN 做 cough COVID detection。  
   为什么先读：报告很强的 cough-specific deep model，是我们必须理解和防守的 claimed SOTA。

3. `2025_drift_adaptive_dynamic_cough_audio_jmir.pdf`  
   主题：dynamic cough audio 下的 drift detection、domain adaptation、active learning。  
   为什么先读：这篇和我们的 external validation / robustness 叙事非常贴，能帮助我们把 AAAI 论文从单数据集性能提升到可靠泛化。

### P1：模型与方法对照

4. `2025_vggish_cough_classification_bmc.pdf`  
   主题：VGGish transfer learning 做 app-recorded cough classification。  
   用途：transfer-learning cough model baseline，关注 AUROC/AUPRC、专家标签、Grad-CAM。

5. `2025_respiratory_kd_ensemble_arxiv.pdf`  
   主题：respiratory sound classification 的 ensemble-to-student KD。  
   用途：和我们的 KD 主线直接相关，重点看 KD loss、teacher/student 对照和效率指标。

6. `2025_respiratory_audio_foundation_pretraining_arxiv.pdf` / `2025_interspeech_respiratory_foundation_model.pdf`  
   主题：respiratory audio foundation model pretraining。  
   用途：判断 AudioSet teacher 是否足够，是否需要 respiratory-specific foundation teacher。

7. `2025_audio_foundation_heart_respiratory_arxiv.pdf`  
   主题：AST、BEATs、BYOL-A、M2D、OPERA 等 audio foundation models 在心肺音任务上的效用。  
   用途：给我们选择 teacher 和写 related work 提供依据。

### P2：补充对照

8. `2025_chio_cough_covid_scirep.pdf`  
   主题：fuzzy preprocessing + U-Net segmentation + handcrafted features + optimized classifier。  
   用途：传统特征/优化式 cough-COVID baseline，适合在 related work 里归类。

## 未能自动下载

- Cough acoustic analysis using AI for COVID-19 detection: Lima vs Montreal cohorts, Annals of Epidemiology 2026  
  ScienceDirect PDF 返回 403，需要机构权限或浏览器手动下载。链接：  
  https://www.sciencedirect.com/science/article/pii/S1047279726000748

- Robust COVID-19 detection from cough sounds using DNDT/DNDF, Expert Systems with Applications 2026 出版版  
  ScienceDirect PDF 返回 403，但已下载 arXiv 版本：`2026_DNDT_DNDF_cross_dataset_cough_covid_arxiv.pdf`。链接：  
  https://www.sciencedirect.com/science/article/pii/S0957417426001491

## 读论文时记录什么

每篇都按下面 6 项做笔记：

1. 数据集：用了哪些数据集，是否 subject-disjoint，是否 external validation。
2. 任务：COVID/non-COVID、healthy/non-healthy、multi-class respiratory disease，还是 cough quality。
3. 模型：输入特征、teacher/student、主干网络、是否 ensemble。
4. 指标：AUROC、AUPRC、F1、sensitivity/specificity、calibration、latency、params。
5. 最高结果：主表里的 best number，不同数据集分开记。
6. 可攻击点：是否数据泄漏、是否只做单数据集、是否缺少外部验证、是否没有 confounder/subgroup、是否没有效率。
