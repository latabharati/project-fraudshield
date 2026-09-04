# Real-time Fraud Detection Dashboard
import json
import time
import random
import shap

import joblib
import numpy as np
import pandas as pd
from flask import Flask, Response, jsonify, render_template, request
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score, precision_score, recall_score, confusion_matrix,)

app = Flask(__name__)

ARTIFACT_DIR = "artifacts"
DATA_DIR = "data"

print("Loading models and artifacts...")
MODELS = {
    "LogisticRegression": joblib.load(f"{ARTIFACT_DIR}/log_reg.pkl"),
    "RandomForest": joblib.load(f"{ARTIFACT_DIR}/rf_model.pkl"),
    "XGBoost": joblib.load(f"{ARTIFACT_DIR}/xgb_model.pkl"),
    "CatBoost": joblib.load(f"{ARTIFACT_DIR}/cb_model.pkl"),
    "MLP": joblib.load(f"{ARTIFACT_DIR}/mlp_model.pkl"),
    "IsolationForest": joblib.load(f"{ARTIFACT_DIR}/iso_forest.pkl"),
}
META_MODEL = joblib.load(f"{ARTIFACT_DIR}/meta_model.pkl")

SCALER = joblib.load(f"{ARTIFACT_DIR}/scaler.pkl")            # StandardScaler, fit on X_train_reduced
ISO_SCALER = joblib.load(f"{ARTIFACT_DIR}/iso_scaler.pkl")     # MinMaxScaler for Isolation Forest
AE_SCALER = joblib.load(f"{ARTIFACT_DIR}/ae_scaler.pkl")       # MinMaxScaler for Autoencoder

FREQUENCY_MAPS = joblib.load(f"{ARTIFACT_DIR}/frequency_maps.pkl") # dict: column -> pandas Series {category: training frequency}
TRAIN_MEDIANS = joblib.load(f"{ARTIFACT_DIR}/train_medians.pkl")
HIGH_MISSING_COLS = joblib.load(f"{ARTIFACT_DIR}/high_missing_cols.pkl")
HIGH_CORR_COLS = joblib.load(f"{ARTIFACT_DIR}/high_corr_cols.pkl")

with open(f"{ARTIFACT_DIR}/metadata.json") as f:
    METADATA = json.load(f)

FEATURE_COLS_FULL = METADATA["feature_cols_full"]
FEATURE_COLS_REDUCED = METADATA["feature_cols_reduced"]
MODEL_THRESHOLDS = METADATA["model_thresholds"]
VOTING_MODELS = METADATA["voting_models"]
STACKING_MODELS = METADATA["stacking_models"]
EDITABLE_FIELDS = METADATA["editable_fields"]   # editable columns for the manual entry form

CATEGORICAL_FEATURES = METADATA["categorical_features"]
CATEGORICAL_MISSING_TOKEN = METADATA.get("categorical_missing_token", "MISSING")
TRANSACTION_AMOUNT_TRANSFORM = METADATA.get("transaction_amount_transform", "log1p")

# Try to load the autoencoder
AUTOENCODER = None
try:
    from tensorflow.keras.models import load_model
    AUTOENCODER = load_model(f"{ARTIFACT_DIR}/autoencoder.keras")
    MODELS["Autoencoder"] = "autoencoder"  # placeholder marker, handled specially in score()
except Exception as e:
    print("Autoencoder not loaded (this is OK -- it just won't appear in the dropdown):", e)

ALL_MODEL_NAMES = list(MODELS.keys()) + ["SoftVoting", "Stacking"]

# Test set for the streaming replay and manual-entry templates
X_test_full_raw = pd.read_csv(f"{DATA_DIR}/X_test_full.csv")
X_test_reduced_raw = pd.read_csv(f"{DATA_DIR}/X_test_reduced.csv")
y_test_raw = pd.read_csv(f"{DATA_DIR}/y_test.csv")
display_df_raw = pd.read_csv(f"{DATA_DIR}/display_df.csv")

SHAP_BG_FULL = pd.read_csv(f"{DATA_DIR}/shap_background_full.csv")
SHAP_BG_REDUCED = pd.read_csv(f"{DATA_DIR}/shap_background_reduced.csv")
SHAP_BG_META = pd.read_csv(f"{DATA_DIR}/shap_background_meta.csv")

# ------------------------------------------------------------------
# SHAP setup for all models
# ------------------------------------------------------------------
print("Preparing SHAP explainers...")

# Keep the exact feature order used during training.
SHAP_BG_FULL = SHAP_BG_FULL[FEATURE_COLS_FULL].copy()
SHAP_BG_REDUCED = SHAP_BG_REDUCED[FEATURE_COLS_REDUCED].copy()

# A smaller background is used for permutation-based explainers so that
# a single web request remains practical. The notebook can use a larger
# sample for the dissertation's offline/global SHAP analysis.
PERMUTATION_BACKGROUND_SIZE = min(10, len(SHAP_BG_REDUCED))
SHAP_BG_REDUCED_PERM = SHAP_BG_REDUCED.sample(
    n=PERMUTATION_BACKGROUND_SIZE,
    random_state=42,
).copy()

SHAP_BG_REDUCED_SCALED = SCALER.transform(SHAP_BG_REDUCED)

def mlp_shap_predict(X):
    """Return MLP fraud probabilities for model-agnostic SHAP."""
    X = pd.DataFrame(X, columns=FEATURE_COLS_REDUCED)
    X_scaled = SCALER.transform(X)
    return MODELS["MLP"].predict_proba(X_scaled)[:, 1]

def isolation_shap_predict(X):
    """Return the exact scaled Isolation Forest anomaly score used by the app."""
    X = pd.DataFrame(X, columns=FEATURE_COLS_REDUCED)
    raw_scores = -MODELS["IsolationForest"].score_samples(X)
    return ISO_SCALER.transform(raw_scores.reshape(-1, 1)).ravel()

def autoencoder_shap_predict(X):
    """Return the exact scaled Autoencoder reconstruction-error score used by the app."""
    if AUTOENCODER is None:
        raise ValueError("Autoencoder is not available in this deployment.")

    X = pd.DataFrame(X, columns=FEATURE_COLS_REDUCED)
    X_scaled = SCALER.transform(X)
    reconstruction = AUTOENCODER.predict(X_scaled, verbose=0)
    reconstruction_error = np.mean((X_scaled - reconstruction) ** 2, axis=1)
    return AE_SCALER.transform(reconstruction_error.reshape(-1, 1)).ravel()

def soft_voting_shap_predict(X):
    """Soft Voting output from the five supervised base-model probabilities."""
    X = np.asarray(X, dtype=float)
    return np.mean(X, axis=1)

SHAP_EXPLAINERS = {}
SHAP_SETUP_ERRORS = {}

# Logistic Regression: trained on scaled reduced features.
try:
    SHAP_EXPLAINERS["LogisticRegression"] = shap.LinearExplainer(
        MODELS["LogisticRegression"],
        SHAP_BG_REDUCED_SCALED,
    )
except Exception as e:
    SHAP_SETUP_ERRORS["LogisticRegression"] = str(e)
    print("Could not create LogisticRegression SHAP explainer:", e)

# Tree-based models: trained on the full unscaled feature set.
for model_name in ["RandomForest", "XGBoost", "CatBoost"]:
    try:
        SHAP_EXPLAINERS[model_name] = shap.TreeExplainer(MODELS[model_name])
    except Exception as e:
        SHAP_SETUP_ERRORS[model_name] = str(e)
        print(f"Could not create {model_name} SHAP explainer:", e)

# MLP: model-agnostic SHAP over the exact scaling + prediction pipeline.
try:
    SHAP_EXPLAINERS["MLP"] = shap.PermutationExplainer(
        mlp_shap_predict,
        SHAP_BG_REDUCED_PERM,
        seed=42,
    )
except Exception as e:
    SHAP_SETUP_ERRORS["MLP"] = str(e)
    print("Could not create MLP SHAP explainer:", e)

# Isolation Forest: explain the anomaly score actually used for thresholding.
try:
    SHAP_EXPLAINERS["IsolationForest"] = shap.PermutationExplainer(
        isolation_shap_predict,
        SHAP_BG_REDUCED_PERM,
        seed=42,
    )
except Exception as e:
    SHAP_SETUP_ERRORS["IsolationForest"] = str(e)
    print("Could not create IsolationForest SHAP explainer:", e)

# Autoencoder: explain the reconstruction-error anomaly score actually used.
if AUTOENCODER is not None:
    try:
        SHAP_EXPLAINERS["Autoencoder"] = shap.PermutationExplainer(
            autoencoder_shap_predict,
            SHAP_BG_REDUCED_PERM,
            seed=42,
        )
    except Exception as e:
        SHAP_SETUP_ERRORS["Autoencoder"] = str(e)
        print("Could not create Autoencoder SHAP explainer:", e)

# Soft Voting and Stacking are explained at the base-model output level.
missing_voting_cols = [c for c in VOTING_MODELS if c not in SHAP_BG_META.columns]
missing_stacking_cols = [c for c in STACKING_MODELS if c not in SHAP_BG_META.columns]

if missing_voting_cols:
    SHAP_SETUP_ERRORS["SoftVoting"] = (
        "Missing SHAP meta-background columns: " + ", ".join(missing_voting_cols)
    )
else:
    try:
        soft_background = SHAP_BG_META[VOTING_MODELS].copy()
        SHAP_EXPLAINERS["SoftVoting"] = shap.ExactExplainer(
            soft_voting_shap_predict,
            soft_background,
        )
    except Exception as e:
        SHAP_SETUP_ERRORS["SoftVoting"] = str(e)
        print("Could not create SoftVoting SHAP explainer:", e)

if missing_stacking_cols:
    SHAP_SETUP_ERRORS["Stacking"] = (
        "Missing SHAP meta-background columns: " + ", ".join(missing_stacking_cols)
    )
else:
    try:
        stacking_background = SHAP_BG_META[STACKING_MODELS].copy()
        SHAP_EXPLAINERS["Stacking"] = shap.LinearExplainer(
            META_MODEL,
            stacking_background,
        )
    except Exception as e:
        SHAP_SETUP_ERRORS["Stacking"] = str(e)
        print("Could not create Stacking SHAP explainer:", e)

print("SHAP explainers ready:", list(SHAP_EXPLAINERS.keys()))

# Check that all exported files contain the same TransactionIDs. Reorder them using the same TransactionID order so all rows stay aligned.
id_sets = {
    "X_test_full.csv": set(X_test_full_raw["TransactionID"]),
    "X_test_reduced.csv": set(X_test_reduced_raw["TransactionID"]),
    "y_test.csv": set(y_test_raw["TransactionID"]),
    "display_df.csv": set(display_df_raw["TransactionID"]),
}
reference_ids = id_sets["X_test_full.csv"]
mismatches = {name: ids for name, ids in id_sets.items() if ids != reference_ids}
if mismatches:
    details = ", ".join(
        f"{name} has {len(ids)} unique IDs vs {len(reference_ids)} in X_test_full.csv"
        for name, ids in mismatches.items()
    )
    raise RuntimeError(
        "Test-set files are not aligned by TransactionID -- refusing to start with "
        f"potentially mismatched data. Details: {details}. This usually means the "
        "notebook export cell was run with some cells re-executed out of order. "
        "Re-run the full notebook top-to-bottom, then re-run the export cell."
    )

canonical_order = X_test_full_raw["TransactionID"].tolist()
X_test_full = X_test_full_raw.set_index("TransactionID").loc[canonical_order].reset_index()
X_test_reduced = X_test_reduced_raw.set_index("TransactionID").loc[canonical_order].reset_index()
y_test = y_test_raw.set_index("TransactionID").loc[canonical_order].reset_index()["isFraud"]
display_df = display_df_raw.set_index("TransactionID").loc[canonical_order].reset_index()

# TransactionID was only needed for alignment -- drop it before feeding to models
X_test_full = X_test_full.drop(columns=["TransactionID"])
X_test_reduced = X_test_reduced.drop(columns=["TransactionID"])

N_TEST_ROWS = len(X_test_full)
print(f"Loaded {N_TEST_ROWS} test transactions (verified aligned by TransactionID). Models ready: {ALL_MODEL_NAMES}")

# Scoring
def score_row(model_name, row_full, row_reduced):
    row_scaled = SCALER.transform(row_reduced)
    if model_name == "LogisticRegression":
        return float(MODELS["LogisticRegression"].predict_proba(row_scaled)[:, 1][0])

    if model_name == "RandomForest":
        return float(MODELS["RandomForest"].predict_proba(row_full)[:, 1][0])

    if model_name == "XGBoost":
        return float(MODELS["XGBoost"].predict_proba(row_full)[:, 1][0])

    if model_name == "CatBoost":
        return float(MODELS["CatBoost"].predict_proba(row_full)[:, 1][0])

    if model_name == "MLP":
        return float(MODELS["MLP"].predict_proba(row_scaled)[:, 1][0])

    if model_name == "IsolationForest":
        raw = -MODELS["IsolationForest"].score_samples(row_reduced)
        score = ISO_SCALER.transform(raw.reshape(-1, 1)).ravel()[0]
        return float(score)

    if model_name == "Autoencoder":
        if AUTOENCODER is None:
            raise ValueError("Autoencoder is not available in this deployment.")
        recon = AUTOENCODER.predict(row_scaled, verbose=0)
        err = np.mean((row_scaled - recon) ** 2, axis=1)
        score = AE_SCALER.transform(err.reshape(-1, 1)).ravel()[0]
        return float(score)

    if model_name == "SoftVoting":
        scores = [score_row(m, row_full, row_reduced) for m in VOTING_MODELS]
        return float(np.mean(scores))

    if model_name == "Stacking":
        base_scores = np.array([[score_row(m, row_full, row_reduced) for m in STACKING_MODELS]])
        return float(META_MODEL.predict_proba(base_scores)[:, 1][0])

    raise ValueError(f"Unknown model: {model_name}")

def get_threshold(model_name):
    return MODEL_THRESHOLDS.get(model_name, 0.5)

def build_result(model_name, row_index, probability, actual_label=None):
    threshold = get_threshold(model_name)
    return {
        "index": int(row_index),
        "model": model_name,
        "probability": round(probability, 4),
        "threshold": round(threshold, 4),
        "flagged": bool(probability >= threshold),
        "actual_label": None if actual_label is None else int(actual_label),
        "display": get_display_row(row_index),
    }
def _extract_shap_row(explanation, expected_features, positive_class_index=1):
    values = explanation.values if hasattr(explanation, "values") else explanation
    values = np.asarray(values)

    if values.ndim == 3:
        # Common binary-classifier shape: (samples, features, classes).
        if values.shape[0] == 1 and values.shape[1] == expected_features:
            class_index = min(positive_class_index, values.shape[2] - 1)
            return values[0, :, class_index]

        # Less common shape: (classes, samples, features).
        if values.shape[1] == 1 and values.shape[2] == expected_features:
            class_index = min(positive_class_index, values.shape[0] - 1)
            return values[class_index, 0, :]

    if values.ndim == 2:
        if values.shape == (1, expected_features):
            return values[0]
        if values.shape == (expected_features, 1):
            return values[:, 0]

    if values.ndim == 1 and values.shape[0] == expected_features:
        return values

    raise ValueError(
        f"Unexpected SHAP output shape {values.shape}; "
        f"expected {expected_features} feature contributions."
    )
# Explain one transaction using the SHAP method appropriate to the selected model.
# Logistic Regression: LinearExplainer on scaled reduced features.
# Random Forest / XGBoost / CatBoost: TreeExplainer on full features.
# MLP: PermutationExplainer on fraud probability.
# Isolation Forest: PermutationExplainer on scaled anomaly score.
# Autoencoder: PermutationExplainer on scaled reconstruction-error score.
# Soft Voting / Stacking: explain the five supervised base-model outputs.
def explain_row_with_shap(model_name, row_full, row_reduced, top_n=8):
    try:
        if model_name not in SHAP_EXPLAINERS:
            setup_error = SHAP_SETUP_ERRORS.get(model_name)
            message = f"SHAP explanation is not available for {model_name}."
            if setup_error:
                message += f" Explainer setup error: {setup_error}"
            return {
                "available": False,
                "message": message,
            }

        explainer = SHAP_EXPLAINERS[model_name]

        # Logistic Regression: SHAP is calculated on scaled values
        if model_name == "LogisticRegression":
            row_for_shap = row_reduced[FEATURE_COLS_REDUCED]
            row_scaled = SCALER.transform(row_for_shap)
            explanation = explainer(row_scaled)
            shap_row = _extract_shap_row(explanation, len(FEATURE_COLS_REDUCED))
            feature_names = FEATURE_COLS_REDUCED
            feature_values = row_for_shap.iloc[0].to_numpy()
            explanation_type = "model_output"

        # Tree models use the full unscaled feature set.
        elif model_name in ["RandomForest", "XGBoost", "CatBoost"]:
            row_for_shap = row_full[FEATURE_COLS_FULL]
            explanation = explainer(row_for_shap)
            shap_row = _extract_shap_row(explanation, len(FEATURE_COLS_FULL))
            feature_names = FEATURE_COLS_FULL
            feature_values = row_for_shap.iloc[0].to_numpy()
            explanation_type = "model_output"

        # MLP uses the reduced features, with scaling handled inside the wrapper.
        elif model_name == "MLP":
            row_for_shap = row_reduced[FEATURE_COLS_REDUCED]
            explanation = explainer(
                row_for_shap,
                max_evals=2 * len(FEATURE_COLS_REDUCED) + 1,
            )
            shap_row = _extract_shap_row(explanation, len(FEATURE_COLS_REDUCED))
            feature_names = FEATURE_COLS_REDUCED
            feature_values = row_for_shap.iloc[0].to_numpy()
            explanation_type = "fraud_score"

        # Isolation Forest SHAP explains the scaled anomaly score, not a probability.
        elif model_name == "IsolationForest":
            row_for_shap = row_reduced[FEATURE_COLS_REDUCED]
            explanation = explainer(
                row_for_shap,
                max_evals=2 * len(FEATURE_COLS_REDUCED) + 1,
            )
            shap_row = _extract_shap_row(explanation, len(FEATURE_COLS_REDUCED))
            feature_names = FEATURE_COLS_REDUCED
            feature_values = row_for_shap.iloc[0].to_numpy()
            explanation_type = "anomaly_score"

        # Autoencoder SHAP explains the scaled reconstruction-error anomaly score.
        elif model_name == "Autoencoder":
            if AUTOENCODER is None:
                return {
                    "available": False,
                    "message": "Autoencoder is not loaded.",
                }

            row_for_shap = row_reduced[FEATURE_COLS_REDUCED]
            explanation = explainer(
                row_for_shap,
                max_evals=2 * len(FEATURE_COLS_REDUCED) + 1,
            )
            shap_row = _extract_shap_row(explanation, len(FEATURE_COLS_REDUCED))
            feature_names = FEATURE_COLS_REDUCED
            feature_values = row_for_shap.iloc[0].to_numpy()
            explanation_type = "anomaly_score"

        # Soft Voting is explained at the five-base-model-output level.
        elif model_name == "SoftVoting":
            base_scores = np.array([[
                score_row(m, row_full, row_reduced)
                for m in VOTING_MODELS
            ]])
            base_df = pd.DataFrame(base_scores, columns=VOTING_MODELS)
            explanation = explainer(base_df)
            shap_row = _extract_shap_row(explanation, len(VOTING_MODELS))
            feature_names = VOTING_MODELS
            feature_values = base_scores[0]
            explanation_type = "base_model_output"

        # Stacking is also explained at the base-model-output level.
        elif model_name == "Stacking":
            base_scores = np.array([[
                score_row(m, row_full, row_reduced)
                for m in STACKING_MODELS
            ]])
            base_df = pd.DataFrame(base_scores, columns=STACKING_MODELS)
            explanation = explainer(base_df)
            shap_row = _extract_shap_row(explanation, len(STACKING_MODELS))
            feature_names = STACKING_MODELS
            feature_values = base_scores[0]
            explanation_type = "base_model_output"

        else:
            return {
                "available": False,
                "message": f"No SHAP implementation is configured for {model_name}.",
            }

        shap_df = pd.DataFrame({
            "feature": feature_names,
            "value": feature_values,
            "shap_value": shap_row,
        })

        shap_df["abs_shap"] = shap_df["shap_value"].abs()
        shap_df = shap_df.sort_values("abs_shap", ascending=False).head(top_n)

        explanations = []
        for _, row in shap_df.iterrows():
            if explanation_type == "anomaly_score":
                direction = (
                    "increases anomaly score"
                    if row["shap_value"] > 0
                    else "decreases anomaly score"
                )
            elif explanation_type == "base_model_output":
                direction = (
                    "pushes ensemble output towards fraud"
                    if row["shap_value"] > 0
                    else "pushes ensemble output towards legitimate"
                )
            else:
                direction = (
                    "pushes model output towards fraud"
                    if row["shap_value"] > 0
                    else "pushes model output towards legitimate"
                )

            explanations.append({
                "feature": str(row["feature"]),
                "value": float(row["value"]) if pd.notnull(row["value"]) else None,
                "shap_value": round(float(row["shap_value"]), 5),
                "direction": direction,
            })

        if explanation_type == "anomaly_score":
            note = (
                "SHAP values explain the model's anomaly score. Positive values increase "
                "the anomaly score and negative values decrease it. They are not direct "
                "percentage changes in fraud probability."
            )
        elif explanation_type == "base_model_output":
            note = (
                "SHAP values explain how the five supervised base-model outputs contribute "
                "to the ensemble output. They do not represent original transaction-feature "
                "importance or direct percentage changes in fraud probability."
            )
        else:
            note = (
                "SHAP values show how each input contributes to the selected model output. "
                "They should not be interpreted as direct percentage changes in fraud probability."
            )

        return {
            "available": True,
            "explained_with": model_name,
            "explanation_type": explanation_type,
            "note": note,
            "top_features": explanations,
        }

    except Exception as e:
        print(f"SHAP explanation failed for {model_name}:", e)
        return {
            "available": False,
            "message": f"SHAP explanation failed: {str(e)}",
        }

# Score all rows together to calculate analytics metrics faster. This avoids predicting one row at a time.
def score_batch(model_name, X_full, X_reduced):
    X_scaled = SCALER.transform(X_reduced)
    if model_name == "LogisticRegression":
        return MODELS["LogisticRegression"].predict_proba(X_scaled)[:, 1]
    if model_name == "RandomForest":
        return MODELS["RandomForest"].predict_proba(X_full)[:, 1]
    if model_name == "XGBoost":
        return MODELS["XGBoost"].predict_proba(X_full)[:, 1]
    if model_name == "CatBoost":
        return MODELS["CatBoost"].predict_proba(X_full)[:, 1]
    if model_name == "MLP":
        return MODELS["MLP"].predict_proba(X_scaled)[:, 1]
    if model_name == "IsolationForest":
        raw = -MODELS["IsolationForest"].score_samples(X_reduced)
        return ISO_SCALER.transform(raw.reshape(-1, 1)).ravel()
    if model_name == "Autoencoder":
        if AUTOENCODER is None:
            return None
        recon = AUTOENCODER.predict(X_scaled, verbose=0)
        err = np.mean((X_scaled - recon) ** 2, axis=1)
        return AE_SCALER.transform(err.reshape(-1, 1)).ravel()
    if model_name == "SoftVoting":
        stack = np.column_stack([score_batch(m, X_full, X_reduced) for m in VOTING_MODELS])
        return stack.mean(axis=1)
    if model_name == "Stacking":
        stack = np.column_stack([score_batch(m, X_full, X_reduced) for m in STACKING_MODELS])
        return META_MODEL.predict_proba(stack)[:, 1]
    return None

# Calculate all model metrics on the test set when the app starts. The results are then used on the Analytics page
def compute_analytics():
    y_true = y_test.values
    metrics = {}
    probs_cache = {}

    for model_name in ALL_MODEL_NAMES:
        probs = score_batch(model_name, X_test_full, X_test_reduced)
        if probs is None:
            continue
        probs_cache[model_name] = probs
        threshold = get_threshold(model_name)
        y_pred = (probs >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        metrics[model_name] = {
            "roc_auc": round(float(roc_auc_score(y_true, probs)), 4),
            "pr_auc": round(float(average_precision_score(y_true, probs)), 4),
            "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
            "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
            "true_positive": int(tp),
            "false_positive": int(fp),
            "true_negative": int(tn),
            "false_negative": int(fn),
        }

    fraud_count = int(y_true.sum())
    legit_count = int(len(y_true) - fraud_count)
    best_model = max(metrics, key=lambda m: metrics[m]["pr_auc"]) if metrics else None

    return {
        "metrics": metrics,
        "fraud_count": fraud_count,
        "legit_count": legit_count,
        "total": int(len(y_true)),
        "best_model": best_model,
        "n_features_full": len(FEATURE_COLS_FULL),
        "n_features_reduced": len(FEATURE_COLS_REDUCED),
    }


print("Precomputing analytics over the loaded test set...")
ANALYTICS = compute_analytics()
print(f"Analytics ready. Best model by PR-AUC: {ANALYTICS['best_model']}")

def get_display_row(index):
    display = display_df.iloc[index].to_dict()
    if "RelativeHour" in X_test_full.columns:
        display["RelativeHour"] = int(X_test_full.iloc[index]["RelativeHour"])
    return display

# Routes
@app.route("/")
def home():
    return render_template(
        "home.html",
        analytics=ANALYTICS,
        model_names=ALL_MODEL_NAMES,
        active_page="home",
    )

@app.route("/detect")
def detect():
    return render_template(
        "detect.html",
        model_names=ALL_MODEL_NAMES,
        editable_fields=EDITABLE_FIELDS,
        active_page="detect",
    )

@app.route("/analytics")
def analytics_page():
    return render_template(
        "analytics.html",
        analytics=ANALYTICS,
        model_names=ALL_MODEL_NAMES,
        active_page="analytics",
    )

@app.route("/about")
def about():
    return render_template(
        "about.html",
        analytics=ANALYTICS,
        model_names=ALL_MODEL_NAMES,
        active_page="about",
    )

@app.route("/api/analytics")
def api_analytics():
    return jsonify(ANALYTICS)

@app.route("/api/models")
def api_models():
    return jsonify({
        "models": ALL_MODEL_NAMES,
        "thresholds": MODEL_THRESHOLDS,
    })

@app.route("/api/field_options")
def api_field_options():
    options = {}

    for field in EDITABLE_FIELDS:
        if field["type"] == "select" and field["name"] in FREQUENCY_MAPS:
            values = FREQUENCY_MAPS[field["name"]]

            if hasattr(values, "index"):
                options[field["name"]] = list(values.index)
            elif isinstance(values, dict):
                options[field["name"]] = list(values.keys())
            else:
                options[field["name"]] = []

    return jsonify(options)

@app.route("/api/transaction/<int:index>")
def api_transaction(index):
    index = index % N_TEST_ROWS
    return jsonify({
        "index": index,
        "display": get_display_row(index),
        "actual_label": int(y_test.iloc[index]),
    })

@app.route("/api/transaction/example/<label>")
def api_transaction_example(label):
    target = 1 if label == "fraud" else 0
    matching_indices = y_test.index[y_test == target].tolist()
    if not matching_indices:
        return jsonify({"error": f"No '{label}' examples found in the loaded test set."}), 404
    index = int(random.choice(matching_indices))
    return jsonify({
        "index": index,
        "display": get_display_row(index),
        "actual_label": int(y_test.iloc[index]),
    })

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Manual entry prediction. Body: {model, index, overrides: {field: value, ...}}"""
    body = request.get_json(force=True)
    model_name = body.get("model", "CatBoost")
    index = int(body.get("index", 0)) % N_TEST_ROWS
    overrides = body.get("overrides", {})

    row_full = X_test_full.iloc[[index]].copy()
    row_reduced = X_test_reduced.iloc[[index]].copy()

    for field, raw_value in overrides.items():
        if raw_value in (None, ""):
            continue
        # TransactionAmt is entered as a raw amount by the user, but the models were trained using TransactionAmt_log.
        if field == "TransactionAmt":
            try:
                amount = float(raw_value)
            except (TypeError, ValueError):
                continue
            if amount < 0:
                continue
            encoded_value = np.log1p(amount)
            model_field = "TransactionAmt_log"

        # Categorical features use training-set frequency encoding
        elif field in FREQUENCY_MAPS:
            category = str(raw_value)
            encoded_value = float(FREQUENCY_MAPS[field].get(category, 0.0))
            model_field = field

        # Numerical features such as RelativeHour
        else:
            try:
                encoded_value = float(raw_value)
            except (TypeError, ValueError):
                continue
            model_field = field

        # update full feature row
        if model_field in row_full.columns:
            row_full[model_field] = encoded_value

        # Update reduced feature row
        if model_field in row_reduced.columns:
            row_reduced[model_field] = encoded_value

    probability = score_row(model_name, row_full, row_reduced)
    result = build_result(model_name, index, probability, actual_label=y_test.iloc[index])

    # Add SHAP explanation for this prediction
    result["shap"] = explain_row_with_shap(
        model_name,
        row_full,
        row_reduced,
        top_n=8
    )
    return jsonify(result)

@app.route("/live-predict")
def live_predict_page():
    return render_template(
        "live_predict.html",
        model_names=ALL_MODEL_NAMES,
        active_page="live_predict",
    )

@app.route("/api/live_predict/sample/<int:index>")
def live_predict_sample(index):
    index = index % N_TEST_ROWS
    transaction = X_test_full.iloc[index].to_dict()
    return jsonify({
        "model": "XGBoost",
        "index": index,
        "transaction": transaction
    })

@app.route("/api/stream")
def api_stream():
    model_name = request.args.get("model", "CatBoost")
    speed = float(request.args.get("speed", 0.8))
    start = int(request.args.get("start", 0)) % N_TEST_ROWS
    count = int(request.args.get("count", 300))
    if model_name not in ALL_MODEL_NAMES:
        return jsonify({"error": f"Unknown model: {model_name}"}), 400

    def generate():
        for i in range(count):
            index = (start + i) % N_TEST_ROWS
            row_full = X_test_full.iloc[[index]]
            row_reduced = X_test_reduced.iloc[[index]]
            try:
                probability = score_row(model_name, row_full, row_reduced)
                result = build_result(
                    model_name,
                    index,
                    probability,
                    actual_label=y_test.iloc[index]
                )
                yield f"data: {json.dumps(result)}\n\n"
                time.sleep(speed)

            except Exception as e:
                print("Stream error:", e)
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
                break
        yield f"data: {json.dumps({'done': True})}\n\n"
    return Response(generate(), mimetype="text/event-stream")

@app.route("/api/live_predict", methods=["POST"])
def live_predict():
    body = request.get_json(force=True)
    model_name = body.get("model", "XGBoost")
    transaction = body.get("transaction", {})
    if model_name not in ALL_MODEL_NAMES:
        return jsonify({
            "error": f"Unknown model: {model_name}",
            "available_models": ALL_MODEL_NAMES
        }), 400

    if not isinstance(transaction, dict):
        return jsonify({
            "error": "transaction must be a JSON object"
        }), 400

    # Check required full feature columns
    missing_cols = [
        col for col in FEATURE_COLS_FULL
        if col not in transaction
    ]

    if len(missing_cols) > 0:
        return jsonify({
            "error": "Missing required model feature columns",
            "missing_count": len(missing_cols),
            "first_missing_columns": missing_cols[:20],
            "note": "This endpoint expects model-ready features, not only raw human-readable fields."
        }), 400

    # Build one-row full feature dataframe
    row_full = pd.DataFrame([transaction])

    # Keep only expected columns and correct order
    row_full = row_full[FEATURE_COLS_FULL]

    # Convert values to numeric where possible
    row_full = row_full.apply(pd.to_numeric, errors="coerce")

    # Fill remaining numeric missing values using medians learned from training data.
    median_values = TRAIN_MEDIANS.reindex(FEATURE_COLS_FULL)
    row_full = row_full.fillna(median_values)

    if row_full.isna().any().any():
        return jsonify({
            "error": "Some model features could not be converted to valid numerical values."
        }), 400

    # Create reduced feature row from full row
    row_reduced = row_full[FEATURE_COLS_REDUCED]

    probability = score_row(model_name, row_full, row_reduced)
    threshold = get_threshold(model_name)

    result = {
        "model": model_name,
        "fraud_probability": round(float(probability), 4),
        "threshold": round(float(threshold), 4),
        "prediction": "Fraud" if probability >= threshold else "Legitimate",
        "flagged": bool(probability >= threshold),
        "shap": explain_row_with_shap(
            model_name,
            row_full,
            row_reduced,
            top_n=8
        )
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, threaded=True, port=5001)
