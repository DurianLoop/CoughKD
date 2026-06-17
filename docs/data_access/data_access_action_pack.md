# Data Access Action Pack for CoughKD-ShiftAudit

更新时间：2026-06-01

## 当前结论

当前实验闭环已经完成了 Coswara -> COUGHVID 的主外部验证、Virufy tiny/segmented stress target、slice/bootstrap/probe/calibration/efficiency 审计，以及 LaTeX 论文骨架。自动 readiness 结论仍然是：

```text
CONDITIONAL_GO_AUDIT_PAPER_NEEDS_SECOND_EXTERNAL
```

也就是说，现在不应该继续做无边界的 teacher × student × KD 组合，而应该优先拿到第二个更有说服力的外部目标。没有第二外部目标前，最稳的论文定位仍然是分析型/审计型，而不是强方法 SOTA。

## 数据集优先级

| 优先级 | 数据集 | 角色 | 是否独立 | 当前动作 |
|---:|---|---|---|---|
| 1 | Cambridge COVID-19 Sounds / ComParE CCS | 第二个独立 COVID cough 外部目标 | 是 | 发送访问申请，询问 DTA 流程 |
| 2 | CODA TB cough / TBcough | 非 COVID 咳嗽疾病迁移目标 | 是，但任务不同 | 申请 Synapse 访问；若使用，论文 claim 需扩展为 generic cough deployment reliability |
| 3 | DiCOVA Track-1 | challenge-style COVID cough auxiliary protocol | 否/弱，因为 Coswara-derived | 只能作为辅助协议，必须先做 overlap audit |
| 4 | FluSense / HF respiratory event audio | 呼吸事件 stress target | 任务不同 | 低优先级；只在重构成 event/representation robustness 时使用 |

## Target 1: Cambridge COVID-19 Sounds / ComParE CCS

### 为什么优先

Cambridge COVID-19 Sounds 是当前最适合补强 ICASSP claim 的第二外部数据集。它包含 cough、breathing、voice 等模态，并且公开页面说明访问需要 Data Transfer Agreement。它与 Coswara/COUGHVID 的数据来源不同，因此最适合检验：

- Coswara -> COUGHVID 上看到的 KD weak/unstable gain 是否会在第二外部目标复现；
- calibration、domain/task probe、slice/bootstrap stress 是否仍然出现 metric disagreement；
- CoughKD-ShiftAudit 是否能从 COUGHVID-only pilot 升级成更强的外部可靠性审计。

### 官方链接

- 项目首页：https://www.covid-19-sounds.org/
- NeurIPS dataset release：https://www.covid-19-sounds.org/en/blog/neurips_dataset.html
- 早期 data sharing 页面：https://covid-19-sounds.org/en/blog/data_sharing.html
- ComParE 2021 paper：https://arxiv.org/abs/2102.13468

### 访问申请邮件

```text
Subject: Academic access request for COVID-19 Sounds / cough recordings

Dear COVID-19 Sounds team,

I am a student researcher working on robust and lightweight respiratory audio model deployment. I would like to request academic access to the COVID-19 Sounds dataset, especially the cough recordings and metadata/split information relevant to the ComParE COVID-19 Cough Sub-Challenge if available.

Our project does not aim to claim clinical diagnostic utility from cough audio. Instead, we study a deployment reliability question: whether knowledge distillation from large audio models to ultra-light cough audio students remains reliable under external dataset shift. We currently use Coswara as a source dataset and COUGHVID as one external target, and we would like to add COVID-19 Sounds as an independent external target for failure-mode auditing, calibration analysis, and model-size/latency-aware evaluation.

Planned use:
- non-commercial academic research;
- external validation of compact cough audio models;
- analysis of dataset shift, calibration, negative transfer, and deployment reliability;
- no redistribution of the audio data.

Requested data:
- cough audio files;
- COVID status labels and available metadata;
- official split information if available;
- data usage terms and Data Transfer Agreement instructions.

Institution: [Your institution]
Research group / supervisor: [Supervisor]
Applicant: [Your name]
Google account or preferred sharing email: [Your email]

Please let me know the required DTA process and whether any institutional signature is needed.

Best regards,
[Your name]
```

### 拿到数据后的处理

拿到 Cambridge 后，先做 manifest，不要直接训练新模型。目标是复用现有 checkpoints 做外部审计。

```powershell
cd D:\CoughKD\AAAI
D:\conda\envs\CoughKD\python.exe scripts\build_external_cough_manifest.py `
  --metadata PATH_TO_METADATA.csv `
  --dataset-dir PATH_TO_AUDIO_ROOT `
  --out manifests\cambridge_external.csv `
  --dataset-name cambridge `
  --id-col ID_COLUMN `
  --audio-col AUDIO_COLUMN `
  --label-col LABEL_COLUMN `
  --label-map POSITIVE_LABEL=covid_positive NEGATIVE_LABEL=healthy
```

随后运行一键接入：

```powershell
cd D:\CoughKD\AAAI
D:\conda\envs\CoughKD\python.exe scripts\onboard_external_target.py `
  --manifest manifests\cambridge_external.csv `
  --target-tag cambridge `
  --skip-existing `
  --device auto `
  --batch-size 16
```

## Target 2: CODA TB / TBcough

### 什么时候使用

如果 Cambridge 审批慢，CODA TB 是很有价值的备选，但它不是 COVID detection 任务，而是 TB cough triage/testing 相关数据。使用它时，论文主张要从：

```text
COVID cough external robustness
```

改成更宽的：

```text
cough audio deployment reliability under disease/task shift
```

也就是说，它不能直接替代 Cambridge 做 COVID 外部验证，但可以让论文从“COVID-specific external validation”转成“respiratory/cough deployment failure audit”。

### 官方/论文链接

- Scientific Data 2024：https://www.nature.com/articles/s41597-024-03972-z
- PMC version / access note：https://pmc.ncbi.nlm.nih.gov/articles/PMC10996751/
- Synapse/TBcough 入口：文献中给出的入口为 `www.synapse.org/TBcough`
- Data sharing / benchmarking DOI：`https://doi.org/10.7303/syn31472953`
- Cough metadata DOI：`https://doi.org/10.7303/syn53710098`

### 动作清单

1. 创建/登录 Synapse 账号。
2. 搜索或打开 `www.synapse.org/TBcough`。
3. 阅读并接受数据使用条款；如果需要认证或申请，按平台提示完成。
4. 下载 training/evaluation metadata 和 cough audio。
5. 明确标签定义：TB positive / TB negative 或 triage testing label，不要硬映射成 COVID。
6. 生成 `manifests/coda_tb_external.csv`。
7. 跑 overlap audit 和 onboarding pipeline。

推荐命令框架：

```powershell
cd D:\CoughKD\AAAI
D:\conda\envs\CoughKD\python.exe scripts\onboard_external_target.py `
  --manifest manifests\coda_tb_external.csv `
  --target-tag coda_tb `
  --skip-existing `
  --device auto `
  --batch-size 16
```

## Target 3: DiCOVA Track-1

### 定位

DiCOVA 是 challenge-style COVID cough 协议，有 5-fold train/validation 和 blind test 设定，适合补充 AUC protocol 和 challenge-style 对照。但它与 Coswara 关系很近，不能直接当作独立外部验证。

### 官方/论文链接

- DiCOVA 2021：https://dicova2021.github.io/
- DiCOVA challenge paper：https://www.isca-archive.org/interspeech_2021/muguli21_interspeech.pdf
- Challenge paper arXiv：https://arxiv.org/abs/2103.09148

### 申请邮件

```text
Subject: Academic access request for DiCOVA Track-1 cough audio protocol

Dear DiCOVA organizers,

I am a student researcher working on robust and lightweight cough audio model deployment. I would like to request academic access to the DiCOVA 2021 Track-1 cough audio dataset and protocol if it is still available for research use.

Our project does not aim to claim clinical COVID diagnosis from cough audio. We study whether ultra-light cough audio models and knowledge distillation methods remain reliable under dataset/protocol shift. DiCOVA would be used as an auxiliary challenge-style protocol with explicit overlap control against Coswara-derived training data.

Requested materials:
- Track-1 cough audio files;
- labels and fold/split metadata;
- recording IDs or subject IDs needed for overlap auditing;
- usage terms and citation requirements.

We will not redistribute the data.

Institution: [Your institution]
Applicant: [Your name]
Supervisor: [Supervisor]
Preferred email: [Your email]

Best regards,
[Your name]
```

### 必须先做 overlap audit

```powershell
cd D:\CoughKD\AAAI
D:\conda\envs\CoughKD\python.exe scripts\audit_manifest_overlap.py `
  --source-manifest manifests\coswara_cough.csv `
  --target-manifest manifests\dicova_external.csv `
  --out runs\overlap_audit\coswara_vs_dicova `
  --enable-hash
```

如果发现明显 overlap，DiCOVA 只能作为 source-related protocol replication，不能作为 independent external target。

## 每周推进节奏

### Week A: 访问申请

- 发送 Cambridge 申请邮件；
- 同时启动 CODA TB Synapse 账号/权限；
- 发送 DiCOVA 辅助协议申请；
- 更新 `docs/data_access/dataset_access_tracker.csv`。

### Week B: 若拿到任一数据集

- 只先生成 manifest；
- 运行 overlap audit；
- 运行 `scripts/onboard_external_target.py`；
- 更新 `runs/submission_readiness/SUBMISSION_READINESS.md`。

### Week C: 论文判断

- 如果 Cambridge/第二外部目标复现 COUGHVID 的 KD failure/cartography 结论：继续写 ShiftAudit 分析型论文；
- 如果某个方法跨 COUGHVID + 第二目标稳定优于 baseline：再考虑从 audit paper 转回 method paper；
- 如果第二目标完全不支持当前发现：写 failure analysis，明确 target-dependence，不继续堆组合。

## 当前停止条件

在拿到第二目标之前，停止以下工作：

- 不继续扩展 teacher × student × KD 网格；
- 不声称某个 KD 方法已经创新成立；
- 不把 Virufy n=16 stress target 写成强外部验证；
- 不把 DiCOVA 未去重结果当成独立外部验证。

允许继续的工作：

- 完善数据申请和 manifest/onboarding 工具；
- 整理 paper-ready tables/figures；
- 补相关工作和 claim boundary；
- 在已有 COUGHVID/Virufy 上做审稿人可能追问的 sanity check。

## Target 1B: UK COVID-19 Vocal Audio Dataset Open Access

### 为什么现在加入

这是本轮新发现的公开候选。它由 Alan Turing Institute / UK COVID-19 Vocal Audio Dataset release 提供，Zenodo open-access 版本包含 cough 和 exhalation 音频、metadata 和 train/test splits，并且标签来自 PCR-referenced COVID test information。与 Cambridge DTA 路线相比，它的优势是公开可下载；劣势是音频 archive 约 53GB，需要谨慎下载和抽样。

### 链接

- Zenodo open-access record: https://zenodo.org/records/10043978
- Scientific Data paper: https://www.nature.com/articles/s41597-024-03492-w
- Alan Turing record: https://atiro.turing.ac.uk/esploro/outputs/dataset/The-UK-COVID-19-Vocal-Audio-Dataset/9922381609548

### 第一步：只下载 metadata

```cmd
cd /d D:\CoughKD\AAAI
scripts\download_ukcovid_metadata_windows.cmd D:\CoughKD\external\ukcovid_open
```

如果需要本地代理：

```cmd
set HTTPS_PROXY=http://127.0.0.1:7897
set HTTP_PROXY=http://127.0.0.1:7897
scripts\download_ukcovid_metadata_windows.cmd D:\CoughKD\external\ukcovid_open
```

### 第二步：构建 metadata-only manifest

```powershell
cd D:\CoughKD\AAAI
D:\conda\envs\CoughKD\python.exe scripts\build_ukcovid_manifest.py `
  --audio-metadata D:\CoughKD\external\ukcovid_open\audio_metadata.csv `
  --participant-metadata D:\CoughKD\external\ukcovid_open\participant_metadata.csv `
  --splits D:\CoughKD\external\ukcovid_open\train_test_splits.csv `
  --dataset-dir D:\CoughKD\external\ukcovid_open `
  --out manifests\ukcovid_open_external.csv `
  --allow-missing-audio
```

### 第三步：下载/解压音频后正式接入

正式 manifest 需要去掉 `--allow-missing-audio`，确保音频路径存在。之后运行：

```powershell
cd D:\CoughKD\AAAI
D:\conda\envs\CoughKD\python.exe scripts\onboard_external_target.py `
  --manifest manifests\ukcovid_open_external.csv `
  --target-tag ukcovid_open `
  --skip-existing `
  --device auto `
  --batch-size 16
```

### 论文定位

如果 UK COVID-19 Vocal Audio 跑通，它可以成为比 Virufy 更强的第二公开外部目标，也可以在 Cambridge DTA 等待期间直接推进 ICASSP evidence package。它仍需 overlap audit，但按来源看比 DiCOVA 更适合作为 independent external candidate。
