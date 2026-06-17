# External Cough Dataset Acquisition Notes

This file tracks external target datasets for CoughKD-Guard validation. The goal is to add at least one independent target beyond COUGHVID.

## 1. Preferred Target: COVID-19 Sounds / ComParE CCS

### Why

The University of Cambridge COVID-19 Sounds dataset is independent from Coswara and includes cough, breathing, and voice recordings. The ComParE 2021 COVID-19 Cough Sub-Challenge (CCS) uses cough samples from the COVID-19 Sounds collection under a challenge protocol.

This is currently a stronger second target than DiCOVA for CoughKD-Guard because it avoids the obvious Coswara-derived-data concern.

### Access

Official access requires a Data Transfer Agreement (DTA). The Cambridge release page says requests should be emailed to:

```text
covid-19-sounds@cl.cam.ac.uk
```

The request should include:

- institution;
- brief project description;
- Google account for sharing;
- academic/legal representatives who can sign the DTA.

### Useful Links

- COVID-19 Sounds app: https://www.covid-19-sounds.org/
- Dataset release page: https://www.covid-19-sounds.org/en/blog/neurips_dataset
- ComParE 2021 paper: https://arxiv.org/abs/2102.13468
- ComParE page: https://www.compare.openaudio.eu/now/

### Suggested Request Email

```text
Subject: Request for academic access to COVID-19 Sounds dataset

Dear COVID-19 Sounds team,

I am a researcher/student at [Institution], working on lightweight respiratory audio model deployment and knowledge-distillation reliability under dataset shift.

We would like to request academic access to the COVID-19 Sounds dataset for a study on target-unlabeled external validation and negative-transfer auditing of compact cough audio models. The work does not aim to claim clinical diagnostic utility from cough audio. Instead, it studies whether compact models trained on one public dataset can be safely audited before deployment on a different target distribution.

Institution: [Institution]
Research group / supervisor: [Name]
Planned use: non-commercial academic research on respiratory audio model robustness and deployment safety
Requested data: metadata, cough audio, and benchmark split information if available
Google account for sharing: [email]

Please let us know the required Data Transfer Agreement process and institutional signatures.

Best regards,
[Name]
```

## 2. Secondary Target: DiCOVA Track-1

### Why Useful

DiCOVA Track-1 is a cough-only COVID-19 detection challenge with AUC and specificity-at-sensitivity metrics, and it is widely cited in COVID cough audio literature.

### Major Constraint

DiCOVA is derived from Coswara and the official DiCOVA page explicitly excludes use of Project Coswara data for augmentation in the challenge. For our project, this means DiCOVA is not automatically a clean external target if the source training data is Coswara.

Use DiCOVA only if:

- subject overlap with current Coswara training split can be ruled out; or
- the Coswara source split is rebuilt excluding all DiCOVA subjects; or
- the paper clearly treats DiCOVA as a Coswara-derived protocol, not an independent external dataset.

### Useful Links

- Official 2021 DiCOVA page: https://dicova2021.github.io/
- Official DiCOVA page: https://dicovachallenge.github.io/
- Challenge paper: https://arxiv.org/abs/2103.09148

### Access Notes

The official 2021 page says challenge access was via registration and terms/conditions, and the audio dataset was not freely redistributable. The page lists contact:

```text
dicova2021@gmail.com
```

## 3. Once A Dataset Is Available

Create a manifest:

```cmd
conda activate CoughKD
cd /d D:\CoughKD\AAAI

python scripts\build_external_cough_manifest.py ^
  --metadata PATH_TO_METADATA.csv ^
  --dataset-dir PATH_TO_AUDIO_ROOT ^
  --out manifests\TARGET_external.csv ^
  --dataset-name TARGET ^
  --id-col ID_COLUMN ^
  --audio-col AUDIO_COLUMN ^
  --label-col LABEL_COLUMN ^
  --label-map COVID-19=covid_positive nonCOVID=healthy
```

Evaluate all existing checkpoints:

```cmd
python scripts\evaluate_external_model_set.py ^
  --manifest manifests\TARGET_external.csv ^
  --target-tag TARGET ^
  --skip-existing
```

Run the multi-target guard audit:

```cmd
python scripts\summarize_coughkd_guard_multitarget.py
```

Primary output:

```text
runs\coughkd_guard_multitarget\COUGHKD_GUARD_MULTITARGET_AUDIT.md
```

