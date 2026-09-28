# Telemetry Anomaly Monitor — Technical Report

## 1. Executive summary

This project converts the supplied OPS-SAT telemetry analysis material into a deployable anomaly-detection system. The original notebook explored handcrafted telemetry features and compared several classical classifiers. The new implementation keeps the useful feature engineering and dataset structure, but changes the experimental protocol and deployment layer:

1. The dataset's existing `train` flag is respected as the official train/test boundary.
2. A validation split is carved out **only from the training partition** for threshold selection.
3. An ExtraTrees classifier is trained on all official training data after threshold selection.
4. A second Isolation Forest model is included for situations where labels are not available.
5. The same model artifacts are exposed through Streamlit and Flask.

The resulting supervised model achieves 0.9735 accuracy, 0.9541 precision, 0.9204 recall, 0.9369 F1, 0.9917 ROC-AUC and 0.9780 PR-AUC on the supplied official holdout.

## 2. Dataset understanding

`dataset.csv` contains 2,123 telemetry fragments. Each row represents a segment-level feature vector. There are 434 anomalous and 1,689 nominal segments. Nine telemetry channels are represented.

`segments.csv` contains 303,493 lower-level signal observations with channel, timestamp, value, segment, label, sampling and anomaly information. The segment-level dataset is therefore the practical ML table used by the application.

The feature set contains measurements such as:

- signal mean, variance and standard deviation;
- kurtosis and skew;
- raw and smoothed peak counts;
- first/second-difference peak counts;
- first/second-difference variance;
- duration and signal length;
- gap-related and normalized variance features;
- telemetry channel and sampling information.

## 3. Problem formulation

The main task is binary classification:

- `0` = nominal segment
- `1` = anomalous segment

The deployed model returns an anomaly probability. A probability at or above the learned operating threshold is classified as anomalous.

This is more useful operationally than returning only a hard class because an operator can sort events by risk and choose a different alert policy later.

## 4. Why ExtraTrees?

Telemetry anomaly boundaries can be nonlinear. The supplied exploratory notebook already showed curved and channel-dependent relationships between statistics such as mean and standard deviation. ExtraTrees is appropriate because it can learn nonlinear feature interactions without requiring polynomial expansion and is relatively robust on mixed-scale tabular data.

The pipeline uses:

- median imputation for numeric features;
- most-frequent imputation for the categorical channel;
- one-hot encoding for channel;
- balanced class weights;
- 500 randomized trees;
- minimum leaf size 2.

An Isolation Forest is also trained as a complementary unsupervised detector. It is not used to produce the headline supervised metrics because its output is not directly comparable to the labeled classifier.

## 5. Experimental protocol

A common risk in telemetry experiments is accidentally leaking test information into feature engineering or threshold selection. This project avoids that by using the supplied `train` flag as the primary boundary.

The 1,594 official training rows are split 80/20, stratified by anomaly label, solely to choose the classification threshold. The final ExtraTrees pipeline is then refit on all 1,594 official training rows. The 529 official test rows are used once for the reported evaluation.

The threshold was selected by maximizing validation F1 and was 0.545 for this run.

## 6. Results

| Metric | Official holdout |
|---|---:|
| Accuracy | 0.9735 |
| Precision | 0.9541 |
| Recall | 0.9204 |
| F1 | 0.9369 |
| ROC-AUC | 0.9917 |
| PR-AUC | 0.9780 |

Confusion matrix:

| | Predicted nominal | Predicted anomaly |
|---|---:|---:|
| Actual nominal | 411 | 5 |
| Actual anomaly | 9 | 104 |

The model therefore missed 9 anomalous holdout segments and raised 5 false alarms at the selected threshold.

## 7. Application architecture

### Streamlit version

The Streamlit application has three modes:

1. **Dashboard** — dataset-level anomaly rates, feature-space visualization, and high-risk segments.
2. **Single segment** — manually enter a feature vector and receive anomaly probability plus an Isolation Forest score.
3. **Batch CSV** — upload multiple feature rows, score them, and download the predictions.

### Flask version

The Flask application provides:

- a browser form at `/`;
- a health endpoint at `/health`;
- JSON inference at `POST /predict`;
- browser form inference at `POST /predict-form`.

Both interfaces use the same saved model artifact, so the ML behavior remains consistent across deployment options.

## 8. How to use the project

Install dependencies:

```bash
pip install -r requirements.txt
```

Retrain:

```bash
python src/train_model.py
```

Launch Streamlit:

```bash
streamlit run app_streamlit.py
```

Launch Flask:

```bash
python app_flask.py
```

## 9. Production considerations

This is an educational/research deployment built from the supplied dataset, not a flight-certified monitoring system. Before operational use, the following should be added:

- time-aware validation and drift monitoring;
- per-channel baselines and thresholds;
- calibrated probabilities;
- model versioning and experiment tracking;
- persistent alert/event storage;
- authentication and authorization for Flask;
- request validation and rate limiting;
- structured logging and audit trails;
- monitoring for feature distribution drift;
- retraining procedures when spacecraft operating regimes change.

## 10. Suggested next upgrades

1. **Time-series model:** add 1D CNN, LSTM/GRU, Transformer, or temporal autoencoder using raw signal windows.
2. **Channel-specific models:** learn separate normal regimes for channels with materially different behavior.
3. **Hybrid detector:** combine supervised probability, Isolation Forest score, and temporal reconstruction error.
4. **Explainability:** expose permutation/SHAP-style feature contributions so operators can see why a segment was flagged.
5. **Streaming mode:** consume telemetry messages continuously and maintain rolling features.
6. **Alert management:** add severity levels, acknowledgement, comments, and incident history.
7. **Drift detection:** monitor PSI/KS statistics and trigger retraining workflows.

## 11. Limitations

The dataset is relatively small at the segment level, and some channels have very few examples. The excellent holdout metrics should therefore be interpreted as evidence for this dataset rather than proof of generalization to unseen spacecraft, new channels, or future operating regimes. The original signal segmentation and handcrafted feature extraction are also assumed to be valid inputs; a production pipeline should explicitly version and validate that upstream process.
