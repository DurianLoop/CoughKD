# CoughKD: Cough Audio Distillation and Screening Research Pipeline

本项目面向咳嗽长音频的自动分段、分类、病患声纹识别，以及在身份标签不足时的轻量化模型压缩与知识蒸馏研究。核心目标不是只做一个二分类 demo，而是构建一套可复现、可投稿、可部署的 cough audio pipeline：从长音频中定位有效咳嗽事件，判断呼吸疾病相关类别，并尽可能学习病患级可区分表征。

> 重要约束：README 中的“预期结果/目标结果”是研究设计和理论测算，不是已完成实验结果。正式论文必须在固定数据划分、独立测试集和统计显著性检验后再改写为真实结果。

## Current Project State

详细 README 树已经整理到 [`docs/`](docs/README.md)：

- [`docs/checklist.md`](docs/checklist.md): 已完成、部分完成、待完成 checklist。
- [`docs/data_and_results.md`](docs/data_and_results.md): Coswara 数据、过滤、划分、实测闭环 baseline 和证据边界。
- [`docs/commands.md`](docs/commands.md): 环境、manifest、smoke、训练、论文构建命令。
- [`docs/visualization_plan.md`](docs/visualization_plan.md): 图、表、数据可视化清单和当前可生成/待实验图件。
- [`docs/long_training_plan.md`](docs/long_training_plan.md): 长时间 GPU 训练前必须通过的 reproducibility、数据、缓存、对照实验和论文证据 gate。
- [`docs/clickup_long_training.md`](docs/clickup_long_training.md): 可迁移到 ClickUp 的长训前置任务清单、依赖关系和验收标准。
- [`docs/teacher_model_downloads.md`](docs/teacher_model_downloads.md): BEATs、AST、PANNs、HTS-AT、PaSST 教师模型下载路径和一键下载命令。
- [`docs/top_conference_optimization_plan.md`](docs/top_conference_optimization_plan.md): 面向顶会投稿的下一阶段优化路线，细化到数据处理、teacher/student、KD loss、超参、评估、部署和论文校准。

当前最重要的实测结果来自 `runs/coswara_torch_closed_loop_full/`：在过滤后的 Coswara cough split 上，compact ConvTeacher 与 DepthwiseStudent+KD 已完成一次真实闭环训练、验证选择和测试集推理。该结果证明 pipeline 可运行，但仍是工程 baseline，不是临床结论，也不是 BEATs/AST/PANNs foundation teacher 结果。

| Item | Current Status |
|---|---|
| Filtered Coswara cough recordings | 5,247 recordings / 2,635 subjects |
| Subject-disjoint split | train 3,671 / val 785 / test 791 recordings |
| Real PyTorch closed-loop run | completed on CUDA |
| Teacher | compact in-repo `ConvTeacher`, 110,277 params |
| Student | `DepthwiseStudent + KD`, 20,717 params |
| Test macro OVR AUROC | teacher 0.577184 / student 0.580959 |
| Evidence boundary | engineering feasibility only; no clinical or SOTA claim |

Fresh validation against the actual `../datasets/coswara` directory was also run in the conda-managed `coughkd` environment. The 2+2 epoch CUDA run is saved under `runs/real_coswara_torch_train_e2` and reports held-out test macro OVR AUROC 0.566054 for `ConvTeacher` and 0.571329 for `DepthwiseStudent + KD`.

The first foundation-teacher long run is complete under `runs/long_coswara_panns_cnn14_16k_seed7_e30`. It uses the official PANNs CNN14 16 kHz checkpoint as a frozen AudioSet backbone with a trainable cough-label head. Held-out test macro OVR AUROC is 0.613420 for the PANNs teacher, 0.563910 for the CE-only depthwise student, and 0.566423 for the KD depthwise student.

Long-duration training is intentionally gated. The next correct step is not to blindly run for many hours; it is to complete the pre-long-training checklist: reconcile the conda environment, freeze data manifests, add feature caching/profiling, add CE-only controls, extend metrics, and only then launch long runs. See [`docs/long_training_plan.md`](docs/long_training_plan.md) and [`docs/clickup_long_training.md`](docs/clickup_long_training.md).

Foundation-teacher training additionally requires local checkpoints. The prepared manual download entry point is:

```bash
conda activate coughkd
bash scripts/download_teacher_models.sh
```

This writes large files under `pretrained/teachers/` and reference implementations under `external/teacher_repos/`; both paths are intentionally git-ignored.

## Dependencies and Main Experiment Environment

主要实验环境使用 conda 管理的 `coughkd` 环境。仓库内的轻量 foundation/smoke 路径目前只依赖 Python 标准库；真实 PyTorch 训练、CUDA 推理和远程实验应在 `coughkd` conda 环境中运行。

环境定义：

- `environment.yml`: conda 环境名 `coughkd`，Python 3.14，pip。
- `requirements.txt`: 当前基础 smoke pipeline 无第三方运行依赖。
- `requirements-ml.txt`: 可选远程训练依赖，包含 CUDA 12.8 PyTorch index 下的 `torch`、`torchaudio`、`numpy`。

当前实测环境：`conda run -n coughkd python --version` 报告 Python 3.14.4；`environment.yml` 已对齐到 Python 3.14。

推荐环境初始化：

```bash
scripts/setup_conda_env.sh coughkd
conda activate coughkd
python -m pip install -r requirements-ml.txt
PYTHONPATH=src python -m unittest discover -s tests
```

## Research Positioning

### Primary Track: Cough-Based Patient Voiceprint Recognition

问题定义：给定一段包含环境噪声、说话、静音和多个咳嗽事件的长音频，系统需要完成：

1. 咳嗽事件检测与切分。
2. 咳嗽片段质量控制和去噪。
3. 咳嗽疾病/状态分类，例如 healthy、symptomatic、COVID-19、COPD、asthma、croup 等可由数据集支持的标签。
4. 病患级识别或验证，输出 closed-set identification accuracy、open-set verification EER/minDCF、Top-k accuracy。

可行性依据：COUGHVID 和 Coswara 提供大规模咳嗽/呼吸/语音数据，Coswara 含 participant-level 多条录音，适合构造同一主体的 cough-heavy、cough-shallow、breath、voice 跨模态验证任务。公开数据中的身份标签质量和伦理许可需要逐项确认，因此声纹识别部分优先采用 Coswara、Cambridge COVID-19 Sounds、可授权临床数据，避免在没有 subject id 的数据上伪造身份任务。

### Fallback Track: SOTA-Oriented Compression and Distillation

如果目标数据集不允许或不足以做病患识别，则任务切换为“面向咳嗽音频识别的高性能教师模型到端侧学生模型蒸馏”。该路线同样可行，且更容易形成 AAAI/ICASSP/INTERSPEECH 风格的完整实验：

- Teacher：BEATs、AST、PaSST、PANNs CNN14、HTS-AT、AudioMAE 或 Whisper encoder。
- Student：MobileNetV3、EfficientNet-B0/B2、BC-ResNet、DS-CNN、Tiny-Conformer、ECAPA-small。
- Distillation：logit KD、feature KD、attention transfer、supervised contrastive KD、multi-task KD。
- Deployment：ONNX/TensorRT/TFLite，报告参数量、FLOPs、latency、memory、AUC/F1/EER 的精度-效率帕累托曲线。

## Current Engineering Entry Points

The repository now contains a dependency-light foundation path and an optional PyTorch path for the remote training server:

```bash
scripts/validate_project.sh
scripts/setup_conda_env.sh coughkd
python -m pip install -r requirements-ml.txt
PYTHONPATH=src python -m coughkd.cli build-manifest --root DATA_ROOT --metadata metadata.csv --dataset coswara --out manifests/coswara.csv
PYTHONPATH=src python -m coughkd.cli build-coswara-manifest --root DATA_ROOT --metadata combined_data.csv --out manifests/coswara_cough.csv
PYTHONPATH=src python -m coughkd.cli filter-manifest --manifest manifests/coswara_cough.csv --root DATA_ROOT --out runs/coswara_cough_filtered --min-duration-sec 0.5 --drop-labels under_validation
PYTHONPATH=src python -m coughkd.cli dataset-smoke --out runs/smoke_dataset
PYTHONPATH=src python -m coughkd.cli torch-smoke --out runs/smoke_torch --device auto
PYTHONPATH=src python -m coughkd.cli torch-manifest-smoke --manifest runs/coswara_cough_filtered_split/manifest_split.csv --root DATA_ROOT --out runs/coswara_torch_manifest_smoke --device auto
PYTHONPATH=src python -m coughkd.cli torch-train --manifest runs/real_coswara_cough_filtered_split/manifest_split.csv --root ../datasets/coswara --out runs/long_coswara_panns_cnn14_16k_seed7_e30 --device cuda --teacher-kind panns_cnn14_16k --teacher-epochs 30 --student-epochs 30 --batch-size 16 --num-workers 4 --max-duration-sec 4.0
```

The PyTorch path now supports both the compact convolutional teacher and a validated PANNs CNN14 16 kHz foundation teacher. BEATs, AST, HTS-AT, and PaSST checkpoints are downloaded but still require dedicated wrappers before they can be used as long-run teachers.

## Abstract

We propose a unified cough audio framework for long-recording cough event segmentation, respiratory-condition classification, and patient-level voiceprint recognition. The system first applies robust audio normalization, VAD/cough event detection, and quality filtering to extract high-confidence cough segments from unconstrained recordings. A large pretrained audio teacher, initialized from BEATs/AST/PANNs and fine-tuned with disease labels and patient identity supervision, learns pathology-aware and identity-preserving cough embeddings. To address the limited identity annotation problem in public cough datasets, the framework supports a fallback compression track that distills the teacher into MobileNetV3/EfficientNet/ECAPA-small students using logit, feature, and supervised contrastive objectives. The expected contribution is a clinically cautious yet deployable cough representation pipeline that improves AUC/F1 for disease classification, reduces EER for patient verification when subject labels are available, and achieves substantial latency and parameter reductions for edge deployment.

## Related Work

### Cough Datasets and COVID-19 Acoustic Screening

COUGHVID is one of the largest public cough datasets, with more than 25k crowdsourced cough recordings and expert labels for a subset. Coswara provides cough, breathing, sustained phonation and speech categories with subject metadata, making it more suitable for patient-level representation learning. Cambridge COVID-19 Sounds and DiCOVA studies emphasize realistic bias analysis, train/test protocol design, and multi-modal fusion across cough, breath and speech.

Recent respiratory audio studies show that simple accuracy is insufficient. A credible system should report AUC, sensitivity, specificity, macro-F1, calibration, subgroup robustness, and cross-dataset generalization. For COVID-19 screening, several works report strong in-domain performance, but cross-corpus performance often drops because of device, language, collection protocol, symptom distribution and label noise shifts. This project therefore treats external validation as a core metric rather than a secondary experiment.

### Speaker and Patient Voiceprint Recognition

Modern speaker verification is dominated by x-vector, ECAPA-TDNN, ResNet-style embeddings and margin-based losses such as AAM-Softmax. ECAPA-TDNN improves TDNN with Res2 blocks, squeeze-excitation, multi-layer aggregation and attentive statistics pooling, and remains a strong baseline for compact speaker embeddings. Cough-based person identification is newer but feasible: recent work reports high identification accuracy using compact cough-specific networks and supervised contrastive learning. The open question is whether cough identity embeddings remain stable across disease status, microphones, noise and session intervals.

### Audio Foundation Models and Efficient Students

AST introduced a convolution-free spectrogram transformer for audio classification and showed strong AudioSet/ESC-50/Speech Commands performance. PANNs demonstrated the value of large-scale AudioSet pretraining with CNN architectures. BEATs uses acoustic tokenizers for self-supervised audio representation and reports SOTA audio classification performance. These models are too heavy for low-power screening, so MobileNet/EfficientNet/BC-ResNet students are used for deployment, with teacher-student distillation to retain discriminative ability.

## Methodology

### End-to-End Pipeline

| Stage | Input | Method | Output | Main Metrics |
|---|---:|---|---|---|
| Long audio normalization | raw audio | ffmpeg/librosa, 16 kHz mono, loudness normalization | standardized waveform | clipping rate, duration retained |
| Noise and speech control | waveform | WebRTC VAD, spectral gating, DNS optional | cleaned waveform | SNR proxy, speech leakage |
| Cough event detection | cleaned waveform | PANNs/AST cough detector + hysteresis threshold | cough timestamps | event F1, onset MAE |
| Quality filtering | cough clips | duration, energy, cough probability, clipping score | valid cough clips | valid ratio |
| Feature extraction | clips | log-mel, MFCC, wav2vec/BEATs embeddings | frame/clip features | embedding stability |
| Disease classification | features | teacher/student classifiers | class probabilities | AUC, macro-F1, sensitivity |
| Patient recognition | cough embeddings | AAM-Softmax + SupCon + PLDA/cosine | identity score | Top-1, EER, minDCF |
| Compression | teacher/student | KD + pruning + quantization | deployable model | params, FLOPs, latency |

### Data Plan

| Dataset | Use | Strength | Risk | Decision |
|---|---|---|---|---|
| COUGHVID | cough detection, disease/abnormality classification | large scale, expert subset | noisy labels, limited identity task | use for pretraining and external test |
| Coswara | disease classification, subject-level tasks | multiple recordings per participant | class imbalance, demographics skew | primary public dataset |
| Cambridge COVID-19 Sounds | external validation | realistic app collection and bias analysis | access/protocol constraints | external robustness test |
| DiCOVA 2021/2022 | benchmark comparison | challenge protocol | COVID-specific | report challenge-style metrics |
| Virufy | clinical cough test subset | PCR-related labels | small scale | external stress test |
| ICBHI 2017 | respiratory sound transfer | expert respiratory cycles | not cough-specific | auxiliary pretraining only |
| Private clinical data | patient recognition | controlled subject labels | ethics/IRB required | optional final validation |

### Model Design

Teacher model:

- Backbone: BEATs-base or AST-base with 128-bin log-mel spectrogram.
- Heads: disease classification head, cough event head, identity embedding head.
- Loss:

```text
L = L_cls + lambda_event L_event + lambda_id L_AAM + lambda_supcon L_SupCon
```

Student model:

- Backbone candidates: MobileNetV3-Small/Large, EfficientNet-B0/B2, BC-ResNet, ECAPA-small.
- Distillation:

```text
L_student = CE(y, p_s)
          + T^2 KL(softmax(z_t/T), softmax(z_s/T))
          + alpha || h_t - P(h_s) ||_2
          + beta L_contrastive(e_s, e_t)
```

Long audio inference:

1. Slice audio into 1.0-2.0 s windows with 50% overlap.
2. Run cough detector and merge adjacent cough windows by hysteresis.
3. Score each cough segment independently.
4. Aggregate segment scores by attention pooling or top-k mean.
5. For identity verification, aggregate embeddings by quality-weighted mean and compare cosine/PLDA scores.

## Evaluation Protocol

### Classification Metrics

- Primary: AUROC, AUPRC, macro-F1, sensitivity, specificity.
- Clinical reporting: sensitivity at fixed specificity, specificity at fixed sensitivity, calibration ECE.
- Robustness: cross-dataset AUC drop, subgroup AUC by gender/age/device/language when metadata exists.

### Voiceprint Metrics

- Closed-set: Top-1 accuracy, Top-5 accuracy.
- Verification: EER, minDCF, ROC/DET curve.
- Robustness: same-session vs cross-session, cough-heavy vs cough-shallow, cough vs voice transfer.

### Efficiency Metrics

- Params, FLOPs/MACs, model size, CPU latency, mobile latency, peak memory.
- Accuracy-latency Pareto frontier.
- Energy proxy if mobile benchmark is available.

## Results

The tables below define the minimum experimental package and theoretical target results. Values marked as `target` are expected targets or sanity-check ranges derived from public reports and model capacity; they must be replaced after actual training. The intended reporting style follows AAAI/ICASSP/INTERSPEECH expectations: strong baselines, cross-dataset validation, confidence intervals, ablations and efficiency metrics.

### AAAI-Style Metric Matrix

| Task | Primary Metric | Secondary Metrics | Required Comparison |
|---|---|---|---|
| Cough event detection | event-level F1 | onset MAE, false alarm/hour | VAD, PANNs detector, AST detector |
| Disease classification | AUROC | AUPRC, macro-F1, sensitivity, specificity | classical ML, CNN, transformer, SSL teacher |
| Patient identification | Top-1 accuracy | Top-5 accuracy, balanced accuracy | x-vector, ECAPA, ResNet, proposed |
| Patient verification | EER | minDCF, ROC/DET | cosine, PLDA, ECAPA, proposed |
| Cross-dataset transfer | external AUROC drop | calibration ECE, subgroup AUC | in-domain vs external |
| Compression | AUROC-retention | params, FLOPs, latency, model size | teacher vs MobileNet/EfficientNet/BC-ResNet |

### Table 1. Disease Classification Model Comparison

| # | Model | Pretraining | Params | Input | Target AUROC | Target Macro-F1 | Notes |
|---:|---|---|---:|---|---:|---:|---|
| 1 | MFCC + SVM | none | <1M | MFCC | 0.72 | 0.62 | classical baseline |
| 2 | MFCC + Random Forest | none | <1M | MFCC | 0.74 | 0.64 | interpretable baseline |
| 3 | BiLSTM | none | 2M | MFCC/log-mel | 0.78 | 0.68 | sequence baseline |
| 4 | 1D-CNN | none | 1M | waveform | 0.76 | 0.66 | raw waveform baseline |
| 5 | VGGish | AudioSet | 62M | log-mel | 0.80 | 0.70 | older transfer baseline |
| 6 | PANNs CNN14 | AudioSet | 79M | log-mel | 0.84 | 0.74 | strong CNN teacher |
| 7 | AST-base | ImageNet/AudioSet | 86M | log-mel | 0.86 | 0.76 | transformer teacher |
| 8 | PaSST | AudioSet | 86M | log-mel | 0.87 | 0.77 | efficient transformer |
| 9 | BEATs-base | AudioSet SSL | 90M | log-mel | 0.88 | 0.78 | main teacher |
| 10 | AudioMAE | AudioSet SSL | 86M | log-mel | 0.86 | 0.76 | SSL baseline |
| 11 | HTS-AT | AudioSet | 31M | log-mel | 0.86 | 0.76 | hierarchical transformer |
| 12 | Whisper encoder | web-scale speech | 39M+ | log-mel | 0.84 | 0.74 | speech-biased baseline |
| 13 | ECAPA-TDNN | VoxCeleb | 14M | mel | 0.81 | 0.71 | identity-oriented |
| 14 | ResNet18 | ImageNet | 11M | spectrogram | 0.82 | 0.72 | CNN baseline |
| 15 | EfficientNet-B0 | ImageNet | 5.3M | spectrogram | 0.83 | 0.73 | lightweight baseline |
| 16 | EfficientNet-B2 | ImageNet | 9.2M | spectrogram | 0.84 | 0.74 | stronger student |
| 17 | MobileNetV3-Large | ImageNet | 5.4M | spectrogram | 0.82 | 0.72 | mobile student |
| 18 | BC-ResNet | AudioSet recipe | 0.3M-1M | log-mel | 0.80 | 0.70 | tiny audio CNN |
| 19 | EfficientNet-B0 + KD | BEATs teacher | 5.3M | spectrogram | 0.85 | 0.75 | proposed student |
| 20 | MobileNetV3 + KD | BEATs teacher | 5.4M | spectrogram | 0.84 | 0.74 | proposed mobile |
| 21 | ECAPA-small + SupCon KD | BEATs/ECAPA teacher | 6M | mel | 0.84 | 0.74 | proposed identity-aware |
| 22 | Proposed Multi-task Student | BEATs + identity teacher | 5M-8M | log-mel | 0.86 | 0.76 | final deployable model |

### Table 2. Patient Recognition Model Comparison

| Model | Loss | Embedding | Target Top-1 | Target EER | Feasibility |
|---|---|---:|---:|---:|---|
| MFCC + GMM | CE | 64 | 65% | 18% | weak baseline |
| x-vector TDNN | AAM | 512 | 78% | 10% | speech baseline |
| ECAPA-TDNN | AAM | 192 | 84% | 7% | strong baseline |
| ResNet34-SE | AAM | 256 | 85% | 6.5% | strong CNN |
| CoughCueNet-style compact CNN | CE + SupCon | 128 | 88% | 5.8% | cough-specific |
| BEATs identity head | AAM + SupCon | 256 | 90% | 5.2% | teacher |
| Proposed ECAPA-small KD | AAM + SupCon + KD | 192 | 88%-90% | 5%-6% | deployable |

### Table 3. Ablation Design

| Experiment | Variant | Expected Effect | Report |
|---|---|---|---|
| Feature | MFCC vs log-mel vs waveform | log-mel/SSL should dominate MFCC | AUC/F1 |
| Event detector | VAD only vs cough detector | cough detector reduces false clips | event F1 |
| Pretraining | random vs AudioSet vs SSL | pretrained models improve low-data generalization | AUC delta |
| Loss | CE vs CE+AAM vs CE+SupCon | contrastive improves identity separation | EER delta |
| Multi-task | disease only vs disease+identity | identity head regularizes cough embeddings | AUC/EER |
| KD type | logits vs features vs logits+features+SupCon | combined KD should improve student | AUC/FLOPs |
| Augmentation | none vs SpecAugment vs noise/reverb/mixup | augmentation improves external robustness | cross-data AUC |
| Aggregation | mean vs top-k vs attention pooling | attention/top-k improves long audio | clip-to-record AUC |
| Quantization | FP32 vs FP16 vs INT8 | INT8 reduces size/latency with small loss | latency/AUC |
| Calibration | no calibration vs temperature scaling | improves clinical thresholding | ECE |

### Table 4. Efficiency Target

| Model | Params | Relative FLOPs | CPU Latency Target | AUROC Target | Deployment Role |
|---|---:|---:|---:|---:|---|
| BEATs-base teacher | ~90M | 1.0x | >300 ms/clip | 0.88 | offline teacher |
| AST-base teacher | ~86M | 0.9x | >250 ms/clip | 0.86 | offline teacher |
| EfficientNet-B0 KD | ~5.3M | 0.12x | <60 ms/clip | 0.85 | edge candidate |
| MobileNetV3 KD | ~5.4M | 0.08x | <45 ms/clip | 0.84 | mobile candidate |
| BC-ResNet KD | <1M | 0.03x | <20 ms/clip | 0.81 | ultra-light |
| Proposed multi-task student | 5M-8M | 0.08x-0.15x | <60 ms/clip | 0.86 | final system |

## Expected Contributions

1. A long-audio cough processing pipeline that reports segment-level and record-level metrics.
2. A patient voiceprint formulation for cough sounds using identity-aware losses and explicit open-set verification.
3. A fallback model compression route that remains publishable if public identity labels are insufficient.
4. A reproducible benchmark covering more than 20 baseline models and a complete ablation matrix.
5. A clinically cautious evaluation package with cross-dataset validation, calibration and subgroup robustness.

## Discussion

The strongest scientific risk is not model architecture but dataset validity. Crowdsourced cough datasets contain label noise, uncontrolled microphones, demographic bias and confounding symptoms. A high in-domain AUC can be misleading if train/test splits leak subject identity or recording condition. Therefore, the project must use subject-level splits, external validation and confidence intervals.

For patient voiceprint recognition, the key question is whether cough carries stable person-specific cues independent of disease and device. If subject labels are sufficiently reliable, ECAPA-style embeddings and supervised contrastive learning provide a defensible route. If not, the project should pivot to distillation: a BEATs/AST teacher can supply strong representations, and MobileNet/EfficientNet students can produce a practical edge model with clear efficiency gains.

Clinical deployment is outside the scope of this repository until prospective validation is available. The system should be described as screening or research support, not medical diagnosis.

## Reproducibility Checklist

- Subject-level train/validation/test split.
- External dataset test without tuning.
- Fixed random seeds and 5-fold confidence intervals.
- Report class distribution and missing metadata.
- Report preprocessing, sample rate and segment length.
- Report threshold selection policy.
- Release scripts for feature extraction, training, evaluation and model export.
- Store model cards for teacher and student models.

## Open-Source Resources

See [reference/literature_and_open_source.md](reference/literature_and_open_source.md) for the maintained list. Minimum practical stack:

- Datasets: COUGHVID, Coswara, Cambridge COVID-19 Sounds, DiCOVA, Virufy, ICBHI 2017.
- Audio models: AST, PANNs, BEATs, PaSST, HTS-AT, SpeechBrain ECAPA-TDNN.
- Audio tooling: librosa, torchaudio, audiomentations, pyannote.audio, WebRTC VAD.
- Deployment: ONNX Runtime, TensorFlow Lite, TensorRT.

## Paper Draft

The current top-conference-style manuscript scaffold is in [paper/main.tex](paper/main.tex). The project has been refocused toward the compression/distillation route:

- Title: `CoughKD: Distilling Audio Foundation Models into Compact Cough Screening Networks`
- Main thesis: distill BEATs/AST/PANNs teachers into MobileNetV3/EfficientNet/BC-ResNet/ECAPA-small students.
- Required evidence: in-domain AUROC retention, external-dataset robustness, calibration, latency, parameter count, FLOPs, and ablation results.
- Safety rule: target numbers in the paper are planning values only until backed by saved experiment logs.

Supporting notes:

- [paper/notes/experimental_protocol.md](paper/notes/experimental_protocol.md)
- [paper/notes/review_log.md](paper/notes/review_log.md)
- [paper/notes/claim_ledger.md](paper/notes/claim_ledger.md)

## Validation

Run the project-level validation script:

```bash
scripts/validate_project.sh
```

On a server, create or update the conda environment first:

```bash
scripts/setup_conda_env.sh coughkd
conda run -n coughkd scripts/validate_project.sh
```

The script checks required research files, compiles the LaTeX paper, fails on unresolved references/citations or hard LaTeX errors, and verifies that target-result safety language is still present.
It also runs the standard-library Python smoke pipeline: manifest validation, subject-disjoint split, preprocessing, metrics, aggregation, augmentation, model/KD losses, ablation grid, table generation, benchmark reporting, subgroup reporting, and baseline checks.

For extracted Coswara data, build a cough-only manifest from participant metadata and `Extracted_data`:

```bash
PYTHONPATH=src python -m coughkd.cli build-coswara-manifest \
  --root /path/to/Coswara-Data \
  --metadata combined_data.csv \
  --out manifests/coswara_cough.csv
PYTHONPATH=src python -m coughkd.cli filter-manifest \
  --manifest manifests/coswara_cough.csv \
  --root /path/to/Coswara-Data \
  --out runs/coswara_cough_filtered \
  --min-duration-sec 0.5 \
  --drop-labels under_validation
PYTHONPATH=src python -m coughkd.cli torch-manifest-smoke \
  --manifest runs/coswara_cough_filtered_split/manifest_split.csv \
  --root /path/to/Coswara-Data \
  --out runs/coswara_torch_manifest_smoke \
  --device auto
```

## References

The project reference list contains more than 50 papers/datasets/tools and more than 15 open-source repositories or documentation links:

- [reference/literature_and_open_source.md](reference/literature_and_open_source.md)
