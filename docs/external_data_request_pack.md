# External Data Request Pack for CoughKD-ShiftAudit

## Purpose

CoughKD-ShiftAudit currently has strong pilot evidence on Coswara -> COUGHVID, but an ICASSP-strength claim needs at least one more external target. The preferred target is Cambridge COVID-19 Sounds / ComParE COVID-19 Cough Sub-Challenge because it is independent from Coswara.

## Target 1: Cambridge COVID-19 Sounds / ComParE CCS

### Why This Dataset

- Independent from Coswara.
- Contains cough recordings and COVID-related metadata.
- Used by the ComParE 2021 COVID-19 Cough Sub-Challenge.
- Directly tests whether CoughKD-ShiftAudit findings generalize beyond COUGHVID.

### Official Links

- COVID-19 Sounds: https://www.covid-19-sounds.org/
- Dataset release: https://www.covid-19-sounds.org/en/blog/neurips_dataset
- ComParE 2021 challenge paper: https://arxiv.org/abs/2102.13468
- ComParE challenge page: https://www.compare.openaudio.eu/now/

### Access Notes

The Cambridge dataset generally requires a Data Transfer Agreement. The public release page lists the contact email:

```text
covid-19-sounds@cl.cam.ac.uk
```

### Request Email Draft

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

## Target 2: DiCOVA

### Role

DiCOVA is useful but should be treated carefully because it is derived from Coswara. It is not a clean independent external target unless overlap is controlled.

### Links

- DiCOVA 2021: https://dicova2021.github.io/
- DiCOVA challenge site: https://dicovachallenge.github.io/
- Challenge paper: https://arxiv.org/abs/2103.09148

### Use Policy in This Project

Use DiCOVA only as:

- a Coswara-derived protocol replication;
- an additional stress test after overlap control;
- not the main independent external dataset.

## Target 3: Non-COVID Cough Transfer Dataset

If Cambridge access is slow, consider a non-COVID cough dataset as a generic cough transfer target. Possible target types:

- wet/dry cough;
- TB cough;
- pediatric cough;
- general cough quality/type labels.

This would change the claim from COVID-related cough screening to generic cough audio deployment reliability. That is acceptable if clearly framed.

## Once Data Is Available

Create a manifest:

```powershell
cd D:\CoughKD\AAAI
D:\conda\envs\CoughKD\python.exe scripts\build_external_cough_manifest.py `
  --metadata PATH_TO_METADATA.csv `
  --dataset-dir PATH_TO_AUDIO_ROOT `
  --out manifests\TARGET_external.csv `
  --dataset-name TARGET `
  --id-col ID_COLUMN `
  --audio-col AUDIO_COLUMN `
  --label-col LABEL_COLUMN `
  --label-map POSITIVE_LABEL=covid_positive NEGATIVE_LABEL=healthy
```

Evaluate existing checkpoints:

```powershell
scripts\run_external_target_guard_audit_windows.cmd manifests\TARGET_external.csv TARGET
```

Preferred one-command onboarding after a manifest is ready:

```powershell
cd D:\CoughKD\AAAI
D:\conda\envs\CoughKD\python.exe scripts\onboard_external_target.py `
  --manifest manifests\TARGET_external.csv `
  --target-tag TARGET `
  --skip-existing `
  --device auto `
  --batch-size 16
```

This runs overlap audit, fixed-model external evaluation, multi-target summaries, paper-table generation, and submission-readiness audit in one bounded pipeline.

Expected output:

```text
runs\external_TARGET_*
runs\coughkd_guard_multitarget\COUGHKD_GUARD_MULTITARGET_AUDIT.md
```

Before reporting any target as external or auxiliary, run overlap control:

```powershell
D:\conda\envs\CoughKD\python.exe scripts\audit_manifest_overlap.py `
  --source-manifest manifests\coswara_cough.csv `
  --target-manifest manifests\TARGET_external.csv `
  --out runs\overlap_audit\coswara_vs_TARGET
```

Use `--enable-hash` when both source and target audio files are locally available and the runtime is acceptable. For DiCOVA, this audit is mandatory because the protocol is Coswara-related.

## Minimum Evidence Needed After New Dataset

For ICASSP-strength CoughKD-ShiftAudit, after adding a second target we need:

- external macro AUROC/AUPRC per method;
- COVID or binary positive AUROC if labels support it;
- ECE/Brier/NLL;
- target-slice or bootstrap-subset stress test;
- comparison of method ranking across COUGHVID and the new target;
- updated main table and figures.
