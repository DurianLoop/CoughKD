# External Dataset Priority Matrix for CoughKD-ShiftAudit

## Why This File Exists

CoughKD-ShiftAudit currently has one independent external target, COUGHVID. To become ICASSP-strength, it needs another target or a carefully justified auxiliary transfer dataset. This matrix ranks candidate datasets by access, independence from Coswara, label compatibility, and expected paper value.

## Priority Matrix

| Priority | Dataset | Access | Independence from Coswara | Label fit | Role in paper | Risk |
|---:|---|---|---|---|---|---|
| 1 | Cambridge COVID-19 Sounds / ComParE CCS | Requires request/DTA | High | COVID cough binary | Best second external target | Access delay |
| 2 | UK COVID-19 Vocal Audio Dataset open access | Zenodo public, large audio archive | High | PCR-referenced COVID cough/exhalation | Best public second external candidate if audio can be downloaded | 53GB audio archive; metadata schema must be checked |
| 3 | Tos COVID-19 / Buenos Aires cough data | Public CKAN/open data | High | RT-PCR positive/negative cough | Fast public second external candidate | 2022 schema differs; first formal run should use 2021 official test |
| 4 | Virufy COVID cough data | GitHub public | High | COVID positive/negative, small | Quick third stress target / sanity external | Small sample size; may be noisy |
| 5 | DiCOVA Track-1 | Challenge access/registration | Low-Medium, Coswara-derived | COVID cough binary | Auxiliary protocol replication | Cannot claim independent external without overlap control |
| 6 | CODA TB cough dataset | Synapse / data access | High | TB vs non-TB cough | Non-COVID cough transfer target | Label semantics differ from COVID |
| 7 | FluSense / HF cough-event audio | Hugging Face public, large download | High | cough/sneeze/speech/noise event labels | Generic respiratory event transfer / representation stress target | Not disease classification; requires reframing |
| 8 | Wet/dry cough datasets | Varies | High | cough type, not disease | Generic cough transfer reliability | Different task, needs reframing |
| Avoid for main claim | Mixed-source processed COVID cough sets | Public mirrors / HF | Often mixes Coswara/COUGHVID/Cambridge | COVID labels | Integration smoke only | Overlap/leakage risk; may violate clean external claim |

## Recommended Order

1. Send Cambridge COVID-19 Sounds request immediately.
2. In parallel, prepare the UK COVID-19 Vocal Audio Dataset open-access path because it is public and PCR-referenced, though large.
3. Keep Virufy as a small legal public stress target.
4. Prepare DiCOVA as a Coswara-derived auxiliary test, not as independent validation.
5. If COVID targets are delayed, use CODA TB or cough-event/type data to reframe the paper as generic cough audio deployment reliability.

## Virufy Quick Path

Expected local path:

```text
D:\CoughKD\datasets\virufy-data
```

Clone command:

```powershell
cd D:\CoughKD\datasets
D:\CoughKD\tools\mingit\cmd\git.exe -c http.sslBackend=schannel clone https://github.com/virufy/virufy-data.git virufy-data
```

Build manifest:

```powershell
cd D:\CoughKD\AAAI
D:\conda\envs\CoughKD\python.exe scripts\build_virufy_manifest.py `
  --dataset-dir D:\CoughKD\datasets\virufy-data `
  --out manifests\virufy_external.csv
```

Evaluate existing checkpoints:

```powershell
scripts\run_external_target_guard_audit_windows.cmd manifests\virufy_external.csv virufy
```

Expected limitations:

- Virufy is small, so AUROC confidence intervals may be wide.
- Treat it as a stress target, not as the sole evidence for ICASSP.
- If label mapping is ambiguous, inspect `labels.csv` before reporting.

## Cambridge / ComParE Path

Use:

- `docs/external_data_request_pack.md`

After data access:

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

Then:

```powershell
scripts\run_external_target_guard_audit_windows.cmd manifests\cambridge_external.csv cambridge
```

## UK COVID-19 Vocal Audio Dataset Open-Access Path

This is now the strongest public second-external candidate because it is PCR-referenced and independent from Coswara/COUGHVID. The open-access Zenodo record contains metadata, train/test splits, and a large audio archive. It includes cough and exhalation recordings; speech is not available in the open-access release.

Links:

- Zenodo open-access record: https://zenodo.org/records/10043978
- Scientific Data paper: https://www.nature.com/articles/s41597-024-03492-w
- Alan Turing dataset page: https://atiro.turing.ac.uk/esploro/outputs/dataset/The-UK-COVID-19-Vocal-Audio-Dataset/9922381609548

First download metadata only:

```cmd
cd /d D:\CoughKD\AAAI
scripts\download_ukcovid_metadata_windows.cmd D:\CoughKD\external\ukcovid_open
```

If the proxy is needed:

```cmd
set HTTPS_PROXY=http://127.0.0.1:7897
set HTTP_PROXY=http://127.0.0.1:7897
scripts\download_ukcovid_metadata_windows.cmd D:\CoughKD\external\ukcovid_open
```

Build a metadata-only manifest before downloading the full audio archive:

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

After the audio archive is downloaded and extracted, rerun without `--allow-missing-audio`, then onboard:

```powershell
D:\conda\envs\CoughKD\python.exe scripts\onboard_external_target.py `
  --manifest manifests\ukcovid_open_external.csv `
  --target-tag ukcovid_open `
  --skip-existing `
  --device auto `
  --batch-size 16
```

Paper role:

> UK COVID-19 Vocal Audio is an independent public PCR-referenced external target. If successfully evaluated, it can replace the current "second external target" blocker more directly than DiCOVA or Virufy.

Access re-check on 2026-06-01:

- Cambridge COVID-19 Sounds remains the best second target by scale. The NeurIPS Datasets and Benchmarks page describes 53,449 audio samples from 36,116 participants, including cough, breathing, and voice modalities.
- Direct public download is still not the expected route. Cambridge states that the data are sensitive and shared through a formal Data Access / Data Transfer Agreement; requests should go to `covid-19-sounds@cl.cam.ac.uk`.
- Action: keep this as the highest-value external target, but do not block all progress on it. Use Virufy and DiCOVA only as stress/auxiliary targets until Cambridge access is obtained.

## DiCOVA Path

Access re-check on 2026-06-01:

- DiCOVA 2021 Track-1 is a cough COVID detection challenge; the public page says the challenge is closed and that registered teams were sent the challenge datasets.
- The page reports a train-val Track-1 set with 75 COVID-positive subjects and 965 non-COVID subjects, about 1.36 hours of FLAC cough audio, plus a blind test set of 233 audio files and an AUC-based leaderboard. It is useful as an auxiliary protocol, but not as a clean independent external dataset because the challenge rules explicitly involve Project Coswara boundaries and access may require old challenge/CodaLab materials.
- Action: treat DiCOVA as a lower-priority auxiliary replication target. Use it only with overlap checks against Coswara subject/recording IDs.

Use only after clarifying data access and overlap:

```powershell
D:\conda\envs\CoughKD\python.exe scripts\build_external_cough_manifest.py `
  --metadata PATH_TO_DICOVA_METADATA.csv `
  --dataset-dir PATH_TO_DICOVA_AUDIO `
  --out manifests\dicova_external.csv `
  --dataset-name dicova `
  --id-col ID_COLUMN `
  --audio-col AUDIO_COLUMN `
  --label-col LABEL_COLUMN `
  --label-map positive=covid_positive negative=healthy
```

Mandatory overlap audit before reporting DiCOVA:

```powershell
D:\conda\envs\CoughKD\python.exe scripts\audit_manifest_overlap.py `
  --source-manifest manifests\coswara_cough.csv `
  --target-manifest manifests\dicova_external.csv `
  --out runs\overlap_audit\coswara_vs_dicova
```

If local audio files are available and runtime is acceptable, rerun with stronger file-level hashing:

```powershell
D:\conda\envs\CoughKD\python.exe scripts\audit_manifest_overlap.py `
  --source-manifest manifests\coswara_cough.csv `
  --target-manifest manifests\dicova_external.csv `
  --out runs\overlap_audit\coswara_vs_dicova_hash `
  --enable-hash
```

Paper wording:

> DiCOVA is treated as a Coswara-derived auxiliary protocol rather than an independent external dataset.

## CODA TB / Non-COVID Path

Access re-check on 2026-06-01:

- CODA TB is the strongest non-COVID disease-transfer candidate because it uses cough audio for tuberculosis-related screening rather than COVID.
- It is independent from Coswara/COUGHVID in label semantics and collection context, which is valuable for a broader cough deployment reliability claim.
- Access is not a simple GitHub clone; it requires Synapse/data access steps.

If using TB cough or wet/dry/event cough data:

- do not call the result COVID validation;
- report it as task-shift or cough-domain transfer;
- possibly remap labels to binary positive/negative only if clinically meaningful;
- otherwise evaluate representation/task transfer with task-specific metrics.

## Hugging Face / FluSense Path

This is a contingency path if independent COVID cough access is delayed.

Use cases:

- test whether student representations and probes behave similarly on generic respiratory event audio;
- extend the story from COVID cough screening to generic cough deployment reliability;
- do not use it as COVID external validation.

Optional manifest builder:

```powershell
cd D:\CoughKD\AAAI
pip install datasets soundfile
D:\conda\envs\CoughKD\python.exe scripts\build_hf_audio_manifest.py `
  --dataset vtsouval/flusense `
  --split train `
  --audio-col audio `
  --label-col label `
  --dataset-name flusense `
  --out manifests\flusense_event_external.csv `
  --audio-out-dir D:\CoughKD\external\hf\flusense\audio `
  --max-records 1000
```

Before using any HF target for claims, inspect labels and run overlap audit if a source-like dataset is suspected. Do not mechanically map event labels to COVID labels.

## Current Decision

Best immediate action:

> Keep Virufy as a small stress target while waiting for Cambridge/ComParE access; prepare CODA TB or FluSense only if the claim is explicitly broadened beyond COVID.

Best ICASSP-strength action:

> Obtain Cambridge COVID-19 Sounds or an equivalent independent COVID cough target.

## Links

- Cambridge COVID-19 Sounds: https://www.covid-19-sounds.org/
- COVID-19 Sounds dataset release: https://www.covid-19-sounds.org/en/blog/neurips_dataset
- ComParE 2021 paper: https://arxiv.org/abs/2102.13468
- Virufy data: https://github.com/virufy/virufy-data
- DiCOVA 2021: https://dicova2021.github.io/
- CODA TB dataset paper: https://www.nature.com/articles/s41597-024-03972-z
- CODA TB external validation 2026: https://www.nature.com/articles/s41598-026-50492-4
