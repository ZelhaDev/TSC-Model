# Ethics Statement
## Traffic Sign Classifier — Half_Exception (CS-304, SY 2025–2026)

**Authors:** Ahmad, Saeed · Aveena, Stephen · Predilla, Stanley · Siron, Carlo  
**Submitted to:** Ma Louella M. Salenga  
**Date:** March 2026

---

## 1. Purpose and Scope

This document articulates the ethical responsibilities assumed by the Half_Exception team in the development of a multi-component traffic sign classification system. The system integrates a Convolutional Neural Network (CNN) image classifier, a Natural Language Processing (NLP) text module, and a Reinforcement Learning (RL) navigation agent trained on the German Traffic Sign Recognition Benchmark (GTSRB). Although this system is an academic prototype, its subject matter — road-safety-critical perception — demands that ethical considerations be treated with the same rigor as technical ones.

---

## 2. Safety-Critical Context and Disclaimers

### 2.1 Prototype Status
This system is a **research prototype only**. It has not been validated for deployment in any real-world or production environment. It **must not** be integrated into autonomous vehicles, driver-assistance systems, traffic-management infrastructure, or any other safety-critical application without independent, certified safety evaluation by qualified engineers.

### 2.2 Residual Error Rate
The best-performing model (Custom CNN) achieved **81.97% test accuracy and a macro-F1 of 74.23%** on the GTSRB test set — meaning roughly **1 in 5 predictions is incorrect** at the image level, with substantially worse performance on underrepresented sign classes (see Section 5). A misclassified stop sign, yield sign, or no-entry sign in a live system could directly contribute to a traffic collision or fatality. These error rates are unacceptable for deployment without significant further development and certification.

### 2.3 Distribution Shift
The model was trained and evaluated exclusively on GTSRB, which contains German traffic signs photographed under Central European conditions. Performance will degrade — potentially severely — when applied to:
- Signs from other countries or jurisdictions (different shapes, colors, symbol conventions)
- Extreme weather conditions (heavy rain, snow, fog, night driving)
- Novel sign damage, graffiti, occlusion, or fading not represented in training data
- Camera hardware differing from GTSRB's source conditions

Any downstream user must conduct domain-specific validation before any real-world use.

### 2.4 Reinforcement Learning Agent
The RL agent operates in a simulated 5×5 grid world and is not trained on real road dynamics. Its reward function encodes simplified traffic-sign semantics and should not be interpreted as a model of legal compliance or safe driving behavior. The convergence metric (>95% success rate in simulation) does not transfer to real environments.

---

## 3. Fairness and Representation

### 3.1 Geographic and Cultural Bias
GTSRB is sourced entirely from Germany. The dataset inherently encodes European — specifically German — road sign conventions as defined by the Vienna Convention on Road Signs and Signals. Systems trained solely on this data will fail on signs from countries using different standards (e.g., North American MUTCD signs, Japanese JIS signs, or signs from nations not party to the Vienna Convention). Deployment outside Central Europe requires retraining or fine-tuning on locally representative data.

### 3.2 Class Imbalance
The GTSRB test set exhibits a **12.5:1 imbalance ratio** between the most frequent class (Class 2 — 50 km/h, 750 images) and the least frequent classes (Classes 0, 19, 27, 32, 37, 41 — 60 images each). Our training set mirrors this imbalance. As a direct consequence:

- The model is disproportionately optimized for high-frequency speed limit signs.
- Rare but safety-critical signs (e.g., Class 27 — Pedestrians, Class 28 — Children Crossing, Class 19 — Dangerous Left Curve) may have materially lower recall.
- Reported macro-F1 averages across all classes; per-class performance for minority classes may be significantly below the headline figure.

This imbalance was partially addressed through stratified splits but was **not** corrected via class-weighted loss functions or oversampling in the current prototype. Future work must address this directly.

### 3.3 Environmental and Temporal Bias
GTSRB images were captured in Germany across varying but unspecified temporal conditions. The dataset does not document the seasonal distribution, time-of-day distribution, or camera hardware used. Models may therefore have uncharacterized failure modes in underrepresented environmental conditions.

---

## 4. Privacy

### 4.1 Dataset Privacy
GTSRB images were captured from public roads. While no direct personal data (names, ID numbers) are present, some images may incidentally contain:
- Vehicle license plates
- Partial images of pedestrians or bystanders near signs

The original dataset collectors are responsible for the capture conditions. This project does not perform any re-identification, profiling, or personal data processing. No images from GTSRB are redistributed as part of this repository.

### 4.2 Inference Privacy
If this classifier were deployed in a vehicle or infrastructure context, it could form part of a system that infers vehicle location and behavior over time. Such a deployment would require a separate, jurisdiction-specific privacy impact assessment. This project does not implement or plan any such deployment.

---

## 5. Transparency and Reproducibility

The team is committed to the following transparency principles:

- **Full pipeline reproducibility:** All training, evaluation, ablation, and analysis steps are reproducible via `bash run.sh` using documented configuration in `configs/config.yaml`.
- **Metric reporting:** We report both accuracy and macro-F1, and conduct per-class, per-category (slice), and confusion-pair analysis to surface weaknesses not visible in headline metrics alone.
- **Ablation documentation:** Three ablation studies (augmentation, learning rate, architecture depth) are run and documented to justify design choices.
- **Explainability:** Grad-CAM saliency maps are generated to provide qualitative insight into model attention, supporting human review of model behavior.
- **Open versioning:** The project is tracked at `https://github.com/ZelhaDev/TSC-Model` with tagged releases.

---

## 6. Identified Risks and Mitigations

| Risk | Likelihood | Severity | Mitigation |
|---|---|---|---|
| Misclassification of stop / no-entry signs in deployment | Medium | Critical | Prototype-only disclaimer; certification requirement before any deployment |
| Poor performance on underrepresented sign classes | High | High | Per-class F1 reporting; future: class-weighted loss, oversampling |
| Distribution shift to non-GTSRB sign standards | High | High | Geographic scope disclaimer; domain-specific revalidation required |
| Overconfident predictions at low image quality | Medium | High | Grad-CAM inspection; future: confidence calibration and uncertainty quantification |
| Misuse of RL agent outside simulation | Low | High | Explicit simulation-only scope declaration |
| Privacy exposure via incidental road-user capture | Low | Medium | No redistribution of source images; no re-identification performed |

---

## 7. Responsible Use Guidelines

1. **Do not deploy** this system in any vehicle, infrastructure, or safety-critical context without independent safety certification.
2. **Do not interpret** macro-averaged metrics as uniform performance guarantees — review per-class and per-slice metrics before making any claims about system capability.
3. **Do disclose** the geographic and dataset limitations of this model in any academic publication, presentation, or public reporting based on this work.
4. **Do conduct** domain-specific validation before any extension of this work to non-GTSRB sign datasets or non-European road environments.
5. **Do report** newly discovered failure modes or biases to the project repository as issues, in keeping with responsible AI development practice.

---

## 8. Team Acknowledgment

By submitting this work, the Half_Exception team affirms that the ethical risks identified in this statement have been considered during development, that the mitigations described reflect genuine design decisions, and that no member of the team has knowingly taken actions that would increase harm to users or third parties.

---
