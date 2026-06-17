# CoughKD-ShiftAudit Paper Draft

## Title

CoughKD-ShiftAudit: Failure Cartography for Ultra-Light Cough Audio Distillation under Dataset Shift

## Abstract

Large audio models can provide useful representations for cough and respiratory sounds, but deploying them on edge devices requires compact students. Knowledge distillation is a natural compression tool, yet its reliability under cough dataset shift remains unclear. We introduce CoughKD-ShiftAudit, a deployment-oriented audit protocol for ultra-light cough audio distillation. Using Coswara as the source dataset and COUGHVID as an external target, we evaluate CE-only training, vanilla KD, target-consistency distillation, shortcut-suppressed KD, disagreement-gated KD, and probe-adversarial training across multiple seeds. Beyond aggregate AUROC, we report calibration, task and domain probes, sensitivity at high specificity, model size, latency, metadata-defined target slices, and bootstrap target-subset stability. Our results show that the most stable KD variant improves external macro AUROC by only +0.0007 over source-only continuation, while slice-level post-hoc gains can be substantially larger. However, target-unlabeled confidence, entropy, and source-relative shift signals fail to reliably select the best method across slices. These findings suggest that cough audio distillation under dataset shift requires failure-oriented auditing before deployment, and that shortcut or calibration improvements alone should not be interpreted as external robustness.

## 1. Introduction

Cough and respiratory sounds have attracted substantial interest as low-cost signals for health screening. During the COVID-19 pandemic, many studies explored whether cough audio could be used to detect infection status. However, direct clinical claims from cough audio remain fragile: labels may be noisy, recruitment channels may differ across classes, recording devices and environments may leak shortcuts, and symptoms or demographic variables may confound performance. As a result, a model that performs well on a single dataset may not provide reliable evidence of clinical utility.

This work therefore does not aim to prove that cough audio can diagnose COVID-19. Instead, we study a deployment-oriented representation transfer problem. Large pretrained audio models such as PANNs can act as teachers, but their computational cost is unsuitable for many edge settings. Ultra-light students are attractive for privacy-preserving and low-latency deployment, but it is unclear whether knowledge distillation reliably transfers useful cough representations under external dataset shift.

Knowledge distillation is often evaluated by aggregate in-domain or external accuracy. In cough audio, this is insufficient. A compact model may have a similar mean AUROC but fail on specific target subgroups; a calibration metric may improve while external ranking worsens; a representation may become less dataset-readable without becoming more task-useful. These mismatches matter because deployment decisions are made under changing target populations, devices, symptoms, and recording quality.

We introduce CoughKD-ShiftAudit, a failure-oriented audit protocol for ultra-light cough audio distillation. Rather than proposing a new diagnostic classifier, we audit when distillation helps, when it fails, and which evidence is needed before deployment. We train an ultra-light Depthwise student on Coswara cough audio, evaluate on COUGHVID, and compare CE-only training, vanilla response KD, target-consistency variants, shortcut-suppressed KD, disagreement-gated KD, and probe-adversarial training across multiple seeds. We jointly report external AUROC/AUPRC, calibration, high-specificity sensitivity, domain and task probes, checkpoint size, latency, metadata-defined slices, and bootstrap target-subset stability.

Our results reveal a central tension. The student is extremely deployable, with only 20.7K parameters and roughly 3912x fewer parameters than the PANNs CNN14 teacher. Yet the most stable KD variant improves external macro AUROC by only +0.0007 over source-only continuation. At the same time, target slices and bootstrap subsets show that local method rankings can vary substantially. This indicates that aggregate performance can hide target-composition sensitivity, while simple target-unlabeled signals are not sufficient to select the best distillation strategy.

Our contributions are:

1. We formulate ultra-light cough audio KD as a deployment reliability problem under dataset shift, avoiding unsupported clinical diagnostic claims.
2. We propose CoughKD-ShiftAudit, an audit protocol combining external evaluation, calibration, probe-based representation evidence, slice stress tests, bootstrap target-subset tests, and deployment metrics.
3. We provide evidence that common and shortcut-aware KD variants yield weak or unstable external gains in Coswara -> COUGHVID transfer.
4. We show that calibration, domain readability, target confidence, and external AUROC can disagree, motivating multi-evidence reporting for cough audio deployment.

## 2. Related Work

### Cough and Respiratory Audio Screening

COVID-era cough audio studies motivated public datasets such as Coswara, COUGHVID, DiCOVA, and COVID-19 Sounds. These datasets enabled rapid modeling, but also introduced concerns about label quality, recruitment bias, recording conditions, and external validity. Recent critiques of audio-based COVID screening emphasize that apparent performance can drop after controlling for symptoms or dataset confounds. Our work follows this caution: we do not claim clinical diagnostic utility from cough alone, and instead analyze deployment reliability of compact models under dataset shift.

### Respiratory Foundation Models and Cough Representations

Recent work has moved from task-specific cough classifiers toward general respiratory audio representations. OPERA provides a respiratory acoustic foundation model benchmark across multiple health tasks. Coughprint distills cough representations from speech foundation model embeddings for edge-compute applications. These works show that respiratory/cough representation learning is a meaningful direction. Our focus is different: we audit whether teacher-to-student distillation remains reliable for ultra-light cough students when the target dataset shifts.

### Knowledge Distillation under Distribution Shift

ShiftKD recently benchmarks knowledge distillation under distribution shift and shows that KD behavior can vary substantially across shifts and methods. Respiratory sound KD has also been explored with architecture-agnostic ensemble distillation. These studies mean that we cannot claim to be the first to study KD under shift or KD for respiratory audio. Our contribution is a domain-specific deployment audit for cough audio: we combine cross-dataset transfer, calibration, domain/task probes, slice-level negative transfer, bootstrap target composition sensitivity, and efficiency metrics in one protocol.

### Dataset and Failure Cartography

Dataset cartography and failure auditing approaches argue that aggregate metrics can hide systematic failures across sample difficulty, acoustic conditions, or subpopulations. We adapt this philosophy to cough distillation. Instead of only asking which model has the best mean AUROC, we ask where KD helps, where it fails, and whether target-unlabeled signals can predict these differences.

### Claim Boundary After 2026 Re-check

Recent related work makes the novelty boundary narrow but still meaningful. ShiftKD already studies KD under distribution shift in general benchmarks. Interspeech 2025 respiratory sound KD already studies ensemble-teacher soft-label distillation for respiratory classification. Coughprint already distills cough representations from speech foundation embeddings for edge-compute applications. OPERA already provides a broad respiratory acoustic foundation-model benchmark. Therefore, this paper does not claim first KD under shift, first respiratory KD, first cough representation distillation, or a new respiratory foundation model.

The remaining contribution is deliberately scoped: CoughKD-ShiftAudit audits whether ultra-light cough audio distillation is reliable under external dataset shift and deployment constraints. Its evidence is not a single winning KD loss, but a multi-evidence failure map showing that aggregate AUROC, calibration, domain readability, target slices, bootstrap subsets, and clip-vs-subject evaluation units can disagree.

## 3. Method: CoughKD-ShiftAudit

### Source and Target Setup

We use Coswara cough audio as the source training dataset and COUGHVID as the external target. The source split is subject-disjoint. The target dataset is used for external evaluation and stress testing. Labels are used to compute audit metrics, not to train target-supervised models unless explicitly stated.

### Teacher and Student

The teacher is PANNs CNN14 at 16 kHz, initialized from AudioSet-pretrained weights and adapted to the source label space. The student is a compact depthwise CNN with 20.7K parameters. This gives an extreme parameter compression ratio of approximately 3912x relative to the 81M-parameter teacher.

### Compared Training Variants

We compare:

- CE-only student;
- vanilla KD;
- source-only continuation;
- target-consistency KD;
- confidence-gated target-consistency KD;
- shortcut-suppressed KD;
- disagreement-gated KD;
- probe-adversarial KD.

These methods are not presented as a new SOTA architecture search. They form a bounded candidate set for auditing common KD assumptions under dataset shift.

### Audit Evidence

CoughKD-ShiftAudit reports:

- external macro AUROC, COVID AUROC, and macro AUPRC;
- ECE, Brier score, NLL, and COVID sensitivity at 95% specificity;
- domain probe AUC and task probe AUC over student representations;
- model parameters, checkpoint size, and latency;
- metadata-defined COUGHVID slices;
- bootstrap target-subset stability;
- label-free guard stress tests using target confidence, entropy, predicted label distribution, and source-relative prediction shift.

## 4. Current Results Summary

The main table is available at:

```text
runs/kd_failure_analysis/SHIFT_AUDIT_MAIN_TABLE.md
runs/kd_failure_analysis/paper_ready_tables/paper_ready_tables.md
runs/kd_failure_analysis/paper_ready_tables/paper_ready_tables.tex
```

Key observations:

- The best aggregate method, confidence-gated TCD, improves external macro AUROC by only +0.0007 over source-only continuation.
- Vanilla KD improves COVID AUROC relative to source-only but reduces macro AUROC.
- Shortcut-suppressed and disagreement-gated variants reduce some shortcut/probe indicators but do not improve external macro AUROC.
- Calibration and ranking disagree: the best ECE methods are not the best external AUROC methods.
- The student is highly efficient: 20.7K parameters, about 0.10 MB checkpoint size, and roughly 1.13 ms CPU latency for a synthetic 4s feature input.

Paper-ready tables currently include:

1. `tab:coughvid_external_audit`: the main COUGHVID external audit table, using source-only continuation as the reference.
2. `tab:target_unit_stress`: the target and evaluation-unit stress table, showing that COUGHVID, Virufy original, Virufy segmented clip-level, and Virufy segmented subject-level can point to different post-hoc best methods and guard outcomes.

## 5. Current Limitations

The current evidence is not yet enough for a strong ICASSP main-track submission because the only large independent external target is COUGHVID. We additionally evaluate Virufy as a tiny public stress target with 16 clinical cough recordings, but this dataset is too small to support a main external validation claim. Its role is diagnostic: it tests whether conclusions and guard behavior remain plausible on a second, much smaller target. The next required step is still to add a larger independent target such as Cambridge COVID-19 Sounds / ComParE CCS. DiCOVA can be useful, but because it is Coswara-derived it should not be treated as a fully independent external validation set without overlap control.

To make this boundary operational, every new external or auxiliary manifest must pass an overlap audit against the Coswara source manifest before it is reported. The current audit script checks exact recording IDs, subject IDs, paths, path stems, path tails, shared identifier-like tokens, and optionally SHA-256 file hashes. The existing Virufy original and segmented manifests pass this first-pass audit with zero detected string/stem/token overlaps against Coswara, but their small effective subject count remains the limiting factor.

## 6. Next Experiments

1. Add a larger second external target dataset.
2. Re-run all existing checkpoints using `scripts/run_external_target_guard_audit_windows.cmd`.
3. Extend the bootstrap and slice stress tests to the new target.
4. Update `SHIFT_AUDIT_MAIN_TABLE.md` and figures.
5. If findings replicate, write the paper as a deployment reliability audit. If one method becomes consistently strong across targets, revisit the possibility of a method claim.

### Virufy Tiny and Segmented Stress Targets

Virufy has been added as a small public stress target:

```text
manifests/virufy_external.csv
runs/kd_failure_analysis/SHIFT_AUDIT_MULTITARGET_TABLE.md
```

We also build a segmented variant:

```text
manifests/virufy_segmented_external.csv
runs/kd_failure_analysis/SHIFT_AUDIT_MULTITARGET_TABLE.md
```

The original Virufy manifest contains 16 clinical cough recordings, with 9 negative and 7 positive examples. The segmented manifest contains 121 clips, but these clips still come from the same 16 source recordings/subjects. Therefore neither setting should be interpreted as a strong independent external validation cohort.

However, both are useful for stress-testing method selection. On Virufy original, CE-only is the post-hoc best method, while the current guard selects `tcd_very_strong` and incurs negative transfer. On Virufy segmented, vanilla KD is the post-hoc best method at the clip level, while the guard selects `tcd_conf035` and captures only a small fraction of the available post-hoc gain. We further aggregate segmented predictions back to the 16 source subjects by averaging class probabilities within each `subject_id`. Under this stricter subject-level view, `candidate_c` becomes the post-hoc best method, vanilla KD remains positive, and the method ranking changes again. This reinforces the conclusion that the current target-unlabeled guard is not reliable enough for a method claim, and that the best distillation strategy can vary strongly with target composition and evaluation granularity.

## 7. Submission Readiness Checklist

Current automated readiness report:

```text
runs/submission_readiness/SUBMISSION_READINESS.md
```

The latest verdict is `CONDITIONAL_GO_AUDIT_PAPER_NEEDS_SECOND_EXTERNAL`. The artifact package is mostly complete, but a second large independent or candidate external target is still required before a strong ICASSP main-track claim.

### Required for ICASSP Main Track

| Requirement | Current status | Evidence / action |
|---|---|---|
| Clear non-clinical claim | Ready | The paper frames deployment reliability, not COVID diagnosis. |
| Strong related-work boundary | Partial | Need final citations and careful wording around ShiftKD, Coughprint, OPERA, respiratory KD. |
| Multi-seed experiments | Ready for COUGHVID | `runs/innovation_loop_summary.json` |
| Independent external validation | Partial | COUGHVID done; second target needed. |
| Slice stress test | Ready for COUGHVID | 25-slice audit completed. |
| Bootstrap target-subset stress test | Ready for COUGHVID | 400/800/1200 subset-size sensitivity completed. |
| Calibration and high-specificity metrics | Ready for COUGHVID | `runs/calibration_efficiency/summary.json` |
| Deployment metrics | Ready for current student | params, checkpoint size, CPU/GPU latency measured. |
| Protocol figure | Draft ready | `runs/kd_failure_analysis/figures/shift_audit_protocol.pdf` |
| Main table | Draft ready | `runs/kd_failure_analysis/SHIFT_AUDIT_MAIN_TABLE.md` |

### Evidence Threshold for Strong Claim

The paper can make a strong CoughKD-ShiftAudit claim if at least one of the following becomes true:

1. Findings replicate on a second independent external target: KD gains remain weak or target-composition-sensitive, and metric disagreement persists.
2. A method becomes consistently strong across COUGHVID and the second target, in which case the claim can shift back toward a method contribution.
3. A non-COVID cough target shows the same audit pattern, allowing a broader cough deployment reliability claim rather than a COVID-specific claim.

### Current Claim Strength

Current status:

> COUGHVID-only pilot evidence is strong enough to motivate CoughKD-ShiftAudit, but not sufficient for a final ICASSP main-track claim.

## 8. Related-Work Boundary Wording

Recommended wording:

> ShiftKD has recently shown that knowledge distillation can behave unpredictably under distribution shift in general classification benchmarks. Our study differs in domain and deployment focus: we analyze ultra-light cough audio students, where external clinical-style transfer, calibration, target-slice robustness, and edge-device efficiency must be considered jointly.

> Respiratory sound KD and Coughprint demonstrate the value of distillation for respiratory and cough representations. We do not claim to be the first to distill respiratory audio models. Instead, we ask whether such distillation remains reliable when a compact cough model trained on one public dataset is evaluated on a shifted external target.

> OPERA studies respiratory acoustic foundation models and downstream respiratory tasks. CoughKD-ShiftAudit is complementary: it audits the reliability of compressing teacher knowledge into an ultra-light student under target shift.

Avoid wording:

- "first KD method for cough audio";
- "first respiratory audio distillation benchmark";
- "state-of-the-art COVID cough diagnosis";
- "clinically reliable cough screening".
