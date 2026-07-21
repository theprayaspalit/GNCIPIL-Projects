from __future__ import annotations

import base64
import json
import math
import shutil
import warnings
import urllib.request
from pathlib import Path

import joblib
import numpy as np
import onnxruntime as ort
import pandas as pd
from imblearn.over_sampling import SMOTE
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "Data"
DEPLOY_DIR = PROJECT_DIR / "Deployment"
MODELS_DIR = DEPLOY_DIR / "models"
DATA_OUT_DIR = DEPLOY_DIR / "data"
RANDOM_CHUNKS_DIR = DATA_OUT_DIR / "random_chunks"
VALIDATION_DIR = DEPLOY_DIR / "validation"
VENDOR_DIR = DEPLOY_DIR / "vendor" / "onnxruntime-web"

RANDOM_STATE = 42
THRESHOLD = 0.5
N_SYNTHETIC_FRAUD = 5000
RANDOM_CHUNK_SIZE = 1000
ORT_VERSION = "1.18.0"
ORT_FILES = [
    "ort.all.min.js",
    "ort-wasm.wasm",
    "ort-wasm-simd.wasm",
    "ort-wasm-threaded.wasm",
    "ort-wasm-simd-threaded.wasm",
    "ort-wasm-simd.jsep.wasm",
    "ort-wasm-simd-threaded.jsep.wasm",
]

warnings.filterwarnings("ignore", message="X does not have valid feature names.*")


def ensure_dirs() -> None:
    for path in (DEPLOY_DIR, MODELS_DIR, DATA_OUT_DIR, RANDOM_CHUNKS_DIR, VALIDATION_DIR, VENDOR_DIR):
        path.mkdir(parents=True, exist_ok=True)


def ensure_onnxruntime_assets() -> None:
    """Copy or download ONNX Runtime Web so the static demo works without the CDN."""
    local_dist = Path(r"E:\codex\fraud-demo-domtest\node_modules\onnxruntime-web\dist")
    for filename in ORT_FILES:
        destination = VENDOR_DIR / filename
        if destination.exists() and destination.stat().st_size > 0:
            continue

        local_file = local_dist / filename
        if local_file.exists():
            shutil.copy2(local_file, destination)
            continue

        url = f"https://cdn.jsdelivr.net/npm/onnxruntime-web@{ORT_VERSION}/dist/{filename}"
        with urllib.request.urlopen(url, timeout=60) as response:
            destination.write_bytes(response.read())


def clean_float(value):
    if isinstance(value, (np.floating, float)):
        if math.isnan(float(value)) or math.isinf(float(value)):
            return None
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def metrics_from_proba(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict:
    pred = (proba > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": clean_float(accuracy_score(y_true, pred)),
        "precision": clean_float(precision_score(y_true, pred, zero_division=0)),
        "recall": clean_float(recall_score(y_true, pred, zero_division=0)),
        "f1": clean_float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": clean_float(roc_auc_score(y_true, proba)),
        "average_precision": clean_float(average_precision_score(y_true, proba)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "correct": int(tn + tp),
        "wrong": int(fp + fn),
    }


def build_rf() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=100,
        max_depth=16,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def fit_models(df: pd.DataFrame, features: list[str]):
    real_train, real_test = train_test_split(
        df,
        test_size=0.3,
        stratify=df["Class"],
        random_state=RANDOM_STATE,
    )

    x_train = real_train[features].astype(np.float32)
    y_train = real_train["Class"].astype(int)
    x_test = real_test[features].astype(np.float32)
    y_test = real_test["Class"].astype(int).to_numpy()

    print("Training baseline real-only Random Forest...", flush=True)
    baseline = build_rf()
    baseline.fit(x_train, y_train)

    print("Training Gaussian-Copula augmented Random Forest...", flush=True)
    augmented = pd.read_csv(DATA_DIR / "augmented_dataset.csv")
    x_aug = augmented[features].astype(np.float32)
    y_aug = augmented["Class"].astype(int)
    gaussian_augmented = build_rf()
    gaussian_augmented.fit(x_aug, y_aug)

    print("Training SMOTE augmented Random Forest...", flush=True)
    target_fraud = int(y_train.sum()) + N_SYNTHETIC_FRAUD
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    smote = SMOTE(
        sampling_strategy={1: target_fraud},
        random_state=RANDOM_STATE,
        k_neighbors=5,
    )
    x_smote, y_smote = smote.fit_resample(x_train_scaled, y_train)
    smote_rf = build_rf()
    smote_rf.fit(x_smote, y_smote)
    smote_model = Pipeline([("scaler", scaler), ("rf", smote_rf)])

    return real_train, real_test, x_test, y_test, {
        "baseline_tuned": {
            "title": "Baseline-tuned",
            "subtitle": "Real training data only",
            "description": "Random Forest trained on the 199,364 real training rows.",
            "model": baseline,
            "threshold": THRESHOLD,
            "training_rows": int(len(real_train)),
            "training_fraud": int(y_train.sum()),
            "synthetic_rows": 0,
        },
        "gaussian_copula_augmented": {
            "title": "Gaussian-Copula augmented",
            "subtitle": "Real data plus generated fraud rows",
            "description": "Random Forest trained on real training data plus 5,000 generated fraud rows.",
            "model": gaussian_augmented,
            "threshold": THRESHOLD,
            "training_rows": int(len(augmented)),
            "training_fraud": int(y_aug.sum()),
            "synthetic_rows": N_SYNTHETIC_FRAUD,
        },
        "smote_augmented": {
            "title": "SMOTE augmented",
            "subtitle": "Real data plus SMOTE fraud rows",
            "description": "Random Forest trained after SMOTE increases fraud examples to match the Gaussian augmented count.",
            "model": smote_model,
            "threshold": THRESHOLD,
            "training_rows": int(len(y_smote)),
            "training_fraud": int(np.sum(y_smote)),
            "synthetic_rows": N_SYNTHETIC_FRAUD,
        },
    }


def convert_model(model, model_id: str, n_features: int) -> bytes:
    initial_type = [("input", FloatTensorType([None, n_features]))]
    option_targets = [model]
    if isinstance(model, Pipeline):
        option_targets.append(model.steps[-1][1])

    last_error = None
    for target in reversed(option_targets):
        try:
            onx = convert_sklearn(
                model,
                initial_types=initial_type,
                target_opset=12,
                options={id(target): {"zipmap": False, "nocl": True}},
            )
            data = onx.SerializeToString()
            (MODELS_DIR / f"{model_id}.onnx").write_bytes(data)
            return data
        except Exception as exc:  # pragma: no cover - fallback path
            last_error = exc

    raise RuntimeError(f"Could not convert {model_id} to ONNX: {last_error}")


def export_js_model(model) -> dict:
    scaler = None
    classifier = model
    if isinstance(model, Pipeline):
        for _, step in model.steps:
            if isinstance(step, StandardScaler):
                scaler = {
                    "mean": [clean_float(value) for value in step.mean_],
                    "scale": [clean_float(value) for value in step.scale_],
                }
            elif isinstance(step, RandomForestClassifier):
                classifier = step

    trees = []
    for estimator in classifier.estimators_:
        tree = estimator.tree_
        values = tree.value[:, 0, :]
        totals = values.sum(axis=1)
        proba1 = np.divide(
            values[:, 1],
            totals,
            out=np.zeros(values.shape[0], dtype=float),
            where=totals != 0,
        )
        trees.append(
            {
                "childrenLeft": [int(value) for value in tree.children_left],
                "childrenRight": [int(value) for value in tree.children_right],
                "feature": [int(value) for value in tree.feature],
                "threshold": [clean_float(value) for value in tree.threshold],
                "proba1": [clean_float(value) for value in proba1],
            }
        )

    return {
        "scaler": scaler,
        "trees": trees,
    }


def validate_onnx(model, onnx_bytes: bytes, x_test: pd.DataFrame, threshold: float) -> dict:
    x_np = x_test.to_numpy(dtype=np.float32)
    sk_proba = model.predict_proba(x_np)[:, 1]
    sk_pred = (sk_proba > threshold).astype(int)

    session = ort.InferenceSession(
        onnx_bytes,
        providers=["CPUExecutionProvider"],
    )
    output_names = [output.name for output in session.get_outputs()]
    outputs = session.run(None, {session.get_inputs()[0].name: x_np})
    probabilities = np.asarray(outputs[-1])
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise RuntimeError(f"Unexpected ONNX probability shape: {probabilities.shape}")

    onnx_proba = probabilities[:, 1]
    onnx_pred = (onnx_proba > threshold).astype(int)

    threshold_match = bool(np.array_equal(onnx_pred, sk_pred))
    max_delta = float(np.max(np.abs(onnx_proba - sk_proba)))

    if not threshold_match:
        mismatch_count = int(np.sum(onnx_pred != sk_pred))
        raise RuntimeError(f"ONNX threshold predictions drifted on {mismatch_count} rows.")

    return {
        "browser_decisions_match_sklearn_exactly": threshold_match,
        "decision_rule_used_by_demo": "fraud_probability > threshold",
        "onnx_outputs": output_names,
        "onnx_output_used_by_demo": "probabilities",
        "onnx_label_output_used_by_demo": False,
        "max_probability_delta": max_delta,
        "rows_validated": int(len(x_np)),
    }


def make_demo_samples(real_test: pd.DataFrame, features: list[str], fitted_models: dict) -> list[dict]:
    indexed_test = real_test.reset_index(names="source_index").copy()
    x_all = indexed_test[features].to_numpy(dtype=np.float32)

    pred_cols = []
    for model_id, entry in fitted_models.items():
        proba = entry["model"].predict_proba(x_all)[:, 1]
        pred_col = f"{model_id}__pred"
        pred_cols.append(pred_col)
        indexed_test[f"{model_id}__proba"] = proba
        indexed_test[pred_col] = (proba > float(entry["threshold"])).astype(int)

    fraud = indexed_test[indexed_test["Class"] == 1]
    false_alerts = indexed_test[(indexed_test["Class"] == 0) & (indexed_test[pred_cols].sum(axis=1) > 0)]
    non_fraud_pool = indexed_test[
        (indexed_test["Class"] == 0) & (~indexed_test["source_index"].isin(false_alerts["source_index"]))
    ]
    random_non_fraud = non_fraud_pool.sample(
        n=min(260, len(non_fraud_pool)),
        random_state=RANDOM_STATE,
    )
    sample_df = (
        pd.concat([fraud, false_alerts, random_non_fraud], axis=0)
        .drop_duplicates(subset="source_index")
        .sample(frac=1.0, random_state=RANDOM_STATE + 8)
        .reset_index(drop=True)
    )

    samples = []
    for row_number, row in sample_df.iterrows():
        feature_values = [float(row[col]) for col in features]
        precomputed = {}
        for model_id, entry in fitted_models.items():
            precomputed[model_id] = {
                "fraud_probability": float(row[f"{model_id}__proba"]),
                "prediction": int(row[f"{model_id}__pred"]),
            }
        samples.append(
            {
                "sample_id": int(row_number + 1),
                "source_index": int(row["source_index"]),
                "true_label": int(row["Class"]),
                "amount": float(row["Amount"]),
                "time": float(row["Time"]),
                "features": feature_values,
                "precomputed": precomputed,
            }
        )
    return samples


def compact_test_row(row: pd.Series, features: list[str]) -> list:
    return [
        int(row["source_index"]),
        int(row["Class"]),
        *[clean_float(float(row[col])) for col in features],
    ]


def make_chunked_test_rows(real_test: pd.DataFrame, features: list[str]) -> dict:
    test_df = real_test.reset_index(names="source_index").reset_index(drop=True)
    columns = ["source_index", "Class", *features]
    chunks = []
    fraud_rows = []

    for chunk_index, start in enumerate(range(0, len(test_df), RANDOM_CHUNK_SIZE)):
        chunk_df = test_df.iloc[start : start + RANDOM_CHUNK_SIZE]
        rows = []
        for _, row in chunk_df.iterrows():
            compact = compact_test_row(row, features)
            rows.append(compact)
            if int(row["Class"]) == 1:
                fraud_rows.append(compact)

        filename = f"chunk_{chunk_index:04d}.json"
        (RANDOM_CHUNKS_DIR / filename).write_text(
            json.dumps(
                {
                    "columns": columns,
                    "start": int(start),
                    "rows": rows,
                },
                separators=(",", ":"),
                ensure_ascii=True,
            ),
            encoding="utf-8",
        )
        chunks.append({"file": filename, "start": int(start), "rows": int(len(rows))})

    (DATA_OUT_DIR / "random_fraud_rows.json").write_text(
        json.dumps(
            {
                "columns": columns,
                "rows": fraud_rows,
            },
            separators=(",", ":"),
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )

    return {
        "columns": columns,
        "chunkSize": RANDOM_CHUNK_SIZE,
        "totalRows": int(len(test_df)),
        "fraudRows": int((test_df["Class"] == 1).sum()),
        "normalRows": int((test_df["Class"] == 0).sum()),
        "chunkPath": "random_chunks/",
        "fraudFile": "random_fraud_rows.json",
        "chunks": chunks,
    }


def make_prepared_samples(samples: list[dict], model_ids: list[str]) -> list[dict]:
    used_ids = set()
    prepared = []
    baseline_id = "baseline_tuned" if "baseline_tuned" in model_ids else model_ids[0]
    augmented_ids = [model_id for model_id in model_ids if model_id != baseline_id]

    def prediction(sample: dict, model_id: str) -> int:
        return int(sample["precomputed"][model_id]["prediction"])

    def probability(sample: dict, model_id: str) -> float:
        return float(sample["precomputed"][model_id]["fraud_probability"])

    def average_probability(sample: dict) -> float:
        return float(np.mean([probability(sample, model_id) for model_id in model_ids]))

    def max_probability(sample: dict) -> float:
        return max(probability(sample, model_id) for model_id in model_ids)

    def all_models_predict(sample: dict, label: int) -> bool:
        return all(prediction(sample, model_id) == label for model_id in model_ids)

    def add_entry(
        sample_id: str,
        title: str,
        description: str,
        hint: str,
        tone: str,
        candidates: list[dict],
        score,
        reverse: bool = True,
    ) -> None:
        for sample in sorted(candidates, key=score, reverse=reverse):
            if sample["sample_id"] in used_ids:
                continue
            used_ids.add(sample["sample_id"])
            prepared.append(
                {
                    "id": sample_id,
                    "title": title,
                    "description": description,
                    "hint": hint,
                    "tone": tone,
                    "sample_id": sample["sample_id"],
                    "source_index": sample["source_index"],
                    "true_label": sample["true_label"],
                    "amount": sample["amount"],
                    "time": sample["time"],
                }
            )
            break

    clear_fraud = [sample for sample in samples if sample["true_label"] == 1 and all_models_predict(sample, 1)]
    clear_normal = [sample for sample in samples if sample["true_label"] == 0 and all_models_predict(sample, 0)]
    augmentation_catches = [
        sample
        for sample in samples
        if sample["true_label"] == 1
        and prediction(sample, baseline_id) == 0
        and any(prediction(sample, model_id) == 1 for model_id in augmented_ids)
    ]
    false_alerts = [
        sample
        for sample in samples
        if sample["true_label"] == 0 and any(prediction(sample, model_id) == 1 for model_id in model_ids)
    ]
    missed_fraud = [sample for sample in samples if sample["true_label"] == 1 and all_models_predict(sample, 0)]
    borderline_normal = [
        sample
        for sample in samples
        if sample["true_label"] == 0 and all_models_predict(sample, 0) and max_probability(sample) > 0
    ]

    add_entry(
        "clear-fraud",
        "Clear fraud",
        "A real fraud transaction that every model flags as fraud.",
        "Use this when you want the demo to show a clean fraud detection.",
        "fraud",
        clear_fraud,
        average_probability,
    )
    add_entry(
        "clear-normal",
        "Clear non-fraud",
        "A real normal transaction that every model keeps as non-fraud.",
        "Use this to show what a safe-looking transaction result looks like.",
        "normal",
        clear_normal,
        average_probability,
        reverse=False,
    )
    add_entry(
        "augmentation-helps",
        "Synthetic model catches this",
        "Baseline misses this fraud, but at least one synthetic-data model catches it.",
        "Use this to explain why augmentation can improve recall.",
        "fraud",
        augmentation_catches,
        lambda sample: max(probability(sample, model_id) for model_id in augmented_ids) - probability(sample, baseline_id),
    )
    add_entry(
        "false-alert",
        "False-alert example",
        "A real non-fraud transaction that at least one model incorrectly flags.",
        "Use this to show why precision and false alerts matter.",
        "challenge",
        false_alerts,
        max_probability,
    )
    add_entry(
        "missed-fraud",
        "Missed fraud",
        "A real fraud transaction that all models classify as non-fraud.",
        "Use this to explain that even high-accuracy fraud models are not perfect.",
        "challenge",
        missed_fraud,
        average_probability,
    )
    add_entry(
        "borderline-normal",
        "Borderline normal",
        "A non-fraud transaction with some fraud-like signal, still kept normal.",
        "Use this to show how probabilities can rise without crossing the fraud threshold.",
        "normal",
        borderline_normal,
        max_probability,
    )
    return prepared


def write_index_html(payload: dict) -> None:
    payload_json = json.dumps(payload, separators=(",", ":"), ensure_ascii=True)
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Synthetic Fraud Detection Demo</title>
  <link rel="icon" href="data:,">
  <style>
    :root {{
      --bg: #f6f8fb;
      --panel: #ffffff;
      --panel-soft: #eef4f8;
      --ink: #18202a;
      --muted: #627083;
      --line: #d8e0ea;
      --blue: #2563eb;
      --teal: #0f766e;
      --red: #dc2626;
      --amber: #b45309;
      --green: #15803d;
      --shadow: 0 18px 50px rgba(26, 32, 44, 0.08);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: var(--bg);
      letter-spacing: 0;
    }}

    * {{ box-sizing: border-box; }}

    body {{
      margin: 0;
      background:
        linear-gradient(180deg, rgba(255, 255, 255, 0.95), rgba(246, 248, 251, 0.95)),
        repeating-linear-gradient(90deg, rgba(37, 99, 235, 0.04) 0 1px, transparent 1px 96px);
    }}

    button, select, input {{ font: inherit; }}

    .shell {{
      width: min(1320px, calc(100% - 32px));
      margin: 0 auto;
      padding: 28px 0 40px;
    }}

    header {{
      display: grid;
      grid-template-columns: minmax(0, 1.5fr) minmax(320px, 0.8fr);
      gap: 18px;
      align-items: stretch;
      margin-bottom: 18px;
    }}

    .hero, .status-panel, .panel, .model-card, .metric-card {{
      background: rgba(255, 255, 255, 0.94);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }}

    .hero {{
      padding: 26px;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      min-height: 210px;
    }}

    .eyebrow {{
      color: var(--teal);
      font-size: 0.82rem;
      font-weight: 700;
      text-transform: uppercase;
      margin-bottom: 10px;
    }}

    h1 {{
      margin: 0;
      font-size: clamp(1.9rem, 3vw, 3.1rem);
      line-height: 1.08;
      letter-spacing: 0;
      max-width: 920px;
    }}

    .hero p {{
      max-width: 850px;
      color: var(--muted);
      font-size: 1rem;
      line-height: 1.55;
      margin: 16px 0 0;
    }}

    .status-panel {{
      padding: 20px;
      display: grid;
      gap: 12px;
      align-content: start;
    }}

    .runtime-state {{
      display: flex;
      gap: 10px;
      align-items: center;
      padding: 11px 12px;
      background: #edfdf8;
      border: 1px solid #bde8db;
      border-radius: 8px;
      color: #145345;
      font-weight: 700;
      line-height: 1.3;
    }}

    .runtime-state.warn {{
      background: #fff7ed;
      border-color: #fed7aa;
      color: #7c2d12;
    }}

    .dot {{
      width: 10px;
      height: 10px;
      border-radius: 999px;
      background: currentColor;
      flex: 0 0 auto;
    }}

    .stats-grid {{
      display: none;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 18px;
    }}

    .page-tabs {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin: 0 0 18px;
      padding: 8px;
      background: rgba(255, 255, 255, 0.72);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }}

    .tab-button {{
      border: 1px solid transparent;
      background: transparent;
      color: var(--muted);
      border-radius: 8px;
      padding: 10px 12px;
      cursor: pointer;
      font-weight: 800;
    }}

    .tab-button.active {{
      background: var(--blue);
      color: #ffffff;
      border-color: var(--blue);
    }}

    .view-page {{
      display: none;
    }}

    .view-page:not(.active) {{
      display: none !important;
    }}

    .view-page.active {{
      display: grid;
    }}

    .checker {{
      grid-template-columns: minmax(320px, 420px) minmax(0, 1fr);
      gap: 18px;
      align-items: start;
    }}

    .quick-start {{
      position: static;
      top: auto;
      align-self: start;
    }}

    .step-label {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      color: var(--blue);
      font-size: 0.78rem;
      font-weight: 900;
      text-transform: uppercase;
      margin-bottom: 8px;
    }}

    .sample-list {{
      display: grid;
      gap: 8px;
      margin: 12px 0;
    }}

    .sample-option {{
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #ffffff;
      padding: 12px;
      display: grid;
      gap: 7px;
      text-align: left;
      cursor: pointer;
      color: var(--ink);
      transition: border-color 140ms ease, box-shadow 140ms ease, transform 140ms ease;
    }}

    .sample-option:hover, .sample-option.active {{
      border-color: var(--blue);
      box-shadow: 0 10px 28px rgba(37, 99, 235, 0.12);
      transform: translateY(-1px);
    }}

    .sample-option.fraud {{
      border-left: 4px solid var(--red);
    }}

    .sample-option.normal {{
      border-left: 4px solid var(--green);
    }}

    .sample-option.challenge {{
      border-left: 4px solid var(--amber);
    }}

    .sample-option-title {{
      display: flex;
      justify-content: space-between;
      gap: 8px;
      align-items: center;
      font-weight: 850;
      line-height: 1.2;
    }}

    .sample-option-body {{
      color: var(--muted);
      font-size: 0.86rem;
      line-height: 1.4;
    }}

    .sample-option-meta {{
      color: var(--muted);
      font-size: 0.78rem;
    }}

    .simple-actions {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      margin: 12px 0;
    }}

    .result-shell {{
      display: grid;
      gap: 18px;
    }}

    .decision-card {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #f8fafc;
      padding: 22px;
      display: grid;
      gap: 16px;
      min-height: 280px;
    }}

    .decision-card.fraud {{
      border-color: #fecaca;
      background: #fff7f7;
    }}

    .decision-card.normal {{
      border-color: #bbf7d0;
      background: #f4fff7;
    }}

    .decision-card.mixed {{
      border-color: #fde68a;
      background: #fffdf2;
    }}

    .decision-kicker {{
      color: var(--muted);
      font-size: 0.78rem;
      font-weight: 900;
      text-transform: uppercase;
    }}

    .decision-main {{
      font-size: clamp(2rem, 4vw, 3.8rem);
      line-height: 1;
      font-weight: 950;
      letter-spacing: 0;
    }}

    .decision-main.fraud {{ color: var(--red); }}
    .decision-main.normal {{ color: var(--green); }}
    .decision-main.mixed {{ color: var(--amber); }}

    .decision-explain {{
      color: var(--muted);
      line-height: 1.55;
      max-width: 760px;
      margin: 0;
    }}

    .decision-facts {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
    }}

    .decision-fact {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: rgba(255, 255, 255, 0.7);
      padding: 11px;
    }}

    .decision-fact span {{
      display: block;
      color: var(--muted);
      font-size: 0.78rem;
      margin-bottom: 4px;
    }}

    .decision-fact strong {{
      display: block;
      font-size: 1rem;
      line-height: 1.2;
    }}

    .decision-actions {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }}

    .transaction-summary {{
      display: grid;
      grid-template-columns: minmax(260px, 0.7fr) minmax(0, 1fr);
      gap: 18px;
      align-items: start;
    }}

    .model-details {{
      grid-column: 1 / -1;
    }}

    details.detail-box {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #ffffff;
      padding: 12px;
    }}

    details.detail-box summary {{
      cursor: pointer;
      font-weight: 850;
      color: var(--ink);
    }}

    .metric-card {{
      padding: 16px;
      min-height: 92px;
    }}

    .metric-label {{
      color: var(--muted);
      font-size: 0.84rem;
      font-weight: 700;
      margin-bottom: 8px;
    }}

    .metric-value {{
      font-size: clamp(1.5rem, 2.2vw, 2.1rem);
      font-weight: 800;
      line-height: 1;
    }}

    .metric-sub {{
      color: var(--muted);
      font-size: 0.82rem;
      margin-top: 8px;
    }}

    .grid {{
      display: grid;
      grid-template-columns: 360px minmax(0, 1fr);
      gap: 18px;
      align-items: start;
    }}

    .panel {{
      padding: 18px;
    }}

    .panel.wide-panel {{
      grid-column: 1 / -1;
    }}

    .panel h2 {{
      margin: 0 0 14px;
      font-size: 1.05rem;
      letter-spacing: 0;
    }}

    .toolbar {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      margin-bottom: 14px;
    }}

    .button {{
      border: 1px solid var(--line);
      background: #ffffff;
      color: var(--ink);
      border-radius: 8px;
      padding: 10px 11px;
      cursor: pointer;
      font-weight: 750;
      transition: border-color 140ms ease, transform 140ms ease, background 140ms ease;
    }}

    .button:hover {{
      border-color: #94a3b8;
      transform: translateY(-1px);
    }}

    .button.primary {{
      background: var(--blue);
      color: white;
      border-color: var(--blue);
    }}

    .select-row {{
      display: grid;
      gap: 8px;
      margin: 14px 0;
    }}

    label {{
      color: var(--muted);
      font-size: 0.82rem;
      font-weight: 750;
    }}

    select {{
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      background: #ffffff;
      color: var(--ink);
    }}

    textarea {{
      width: 100%;
      min-height: 78px;
      resize: vertical;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      background: #ffffff;
      color: var(--ink);
      line-height: 1.4;
    }}

    .manual-panel {{
      margin-top: 18px;
    }}

    .manual-layout {{
      grid-template-columns: minmax(0, 0.95fr) minmax(0, 1.05fr);
      gap: 18px;
      align-items: start;
    }}

    .section-intro {{
      color: var(--muted);
      line-height: 1.55;
      margin: -4px 0 14px;
      font-size: 0.94rem;
    }}

    .hint-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
      margin: 12px 0;
    }}

    .hint-card {{
      background: #f8fafc;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      color: var(--muted);
      font-size: 0.86rem;
      line-height: 1.45;
    }}

    .hint-card strong {{
      color: var(--ink);
      display: block;
      margin-bottom: 4px;
    }}

    .prepared-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
    }}

    .prepared-card {{
      display: grid;
      gap: 11px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      background: #ffffff;
      min-height: 245px;
    }}

    .prepared-card.fraud {{
      border-color: #fecaca;
      background: #fff8f8;
    }}

    .prepared-card.normal {{
      border-color: #bbf7d0;
      background: #f7fff9;
    }}

    .prepared-card.challenge {{
      border-color: #fde68a;
      background: #fffdf2;
    }}

    .prepared-head {{
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 10px;
    }}

    .prepared-title {{
      font-weight: 850;
      line-height: 1.2;
    }}

    .prepared-description, .prepared-hint {{
      color: var(--muted);
      font-size: 0.86rem;
      line-height: 1.45;
    }}

    .prepared-hint {{
      padding: 9px 10px;
      border-radius: 8px;
      background: rgba(255, 255, 255, 0.72);
      border: 1px dashed var(--line);
    }}

    .prepared-meta {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 7px;
      color: var(--muted);
      font-size: 0.8rem;
    }}

    .prepared-meta span {{
      display: grid;
      gap: 2px;
    }}

    .prepared-meta strong {{
      color: var(--ink);
      font-size: 0.9rem;
    }}

    .prepared-actions {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      margin-top: auto;
    }}

    .button.small {{
      padding: 8px 9px;
      font-size: 0.84rem;
    }}

    .prepared-empty {{
      color: var(--muted);
      border: 1px dashed var(--line);
      border-radius: 8px;
      padding: 14px;
      background: #f8fafc;
    }}

    .inline-hint {{
      color: var(--muted);
      font-size: 0.82rem;
      line-height: 1.45;
      margin-top: 6px;
    }}

    .checkbox-row {{
      display: flex;
      gap: 8px;
      align-items: flex-start;
      color: var(--muted);
      font-size: 0.84rem;
      line-height: 1.45;
      margin: 10px 0;
    }}

    .checkbox-row input {{
      margin-top: 3px;
    }}

    .manual-actions {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 8px;
      margin: 12px 0;
    }}

    .manual-grid {{
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 9px;
      max-height: 310px;
      overflow: auto;
      padding: 10px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #f8fafc;
    }}

    .manual-field {{
      display: grid;
      gap: 5px;
    }}

    .manual-field label {{
      font-size: 0.74rem;
    }}

    .manual-field input {{
      width: 100%;
      min-width: 0;
      border: 1px solid var(--line);
      border-radius: 7px;
      padding: 8px 7px;
      background: #ffffff;
      color: var(--ink);
    }}

    .manual-field input:focus, textarea:focus, select:focus {{
      outline: 3px solid rgba(37, 99, 235, 0.18);
      border-color: var(--blue);
    }}

    .manual-status {{
      min-height: 22px;
      color: var(--muted);
      font-size: 0.84rem;
      line-height: 1.4;
    }}

    .truth-box {{
      border-radius: 8px;
      padding: 14px;
      border: 1px solid var(--line);
      background: var(--panel-soft);
      margin-top: 14px;
    }}

    .truth-title {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 10px;
    }}

    .badge {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      border-radius: 999px;
      padding: 5px 9px;
      font-size: 0.78rem;
      font-weight: 850;
      border: 1px solid transparent;
    }}

    .badge.fraud {{
      color: #991b1b;
      background: #fee2e2;
      border-color: #fecaca;
    }}

    .badge.normal {{
      color: #166534;
      background: #dcfce7;
      border-color: #bbf7d0;
    }}

    .fact-list {{
      display: grid;
      gap: 8px;
      margin: 0;
      padding: 0;
      list-style: none;
    }}

    .fact-list li {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      border-bottom: 1px solid rgba(148, 163, 184, 0.24);
      padding-bottom: 7px;
      font-size: 0.92rem;
    }}

    .fact-list li:last-child {{
      border-bottom: 0;
      padding-bottom: 0;
    }}

    .fact-list span:first-child {{ color: var(--muted); }}
    .fact-list span:last-child {{ font-weight: 780; text-align: right; }}

    .model-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
    }}

    .model-card {{
      padding: 16px;
      min-height: 250px;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }}

    .model-head {{
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 12px;
    }}

    .model-title {{
      font-weight: 850;
      font-size: 1rem;
      line-height: 1.2;
    }}

    .model-subtitle {{
      color: var(--muted);
      font-size: 0.82rem;
      margin-top: 4px;
    }}

    .probability {{
      font-size: clamp(1.65rem, 3vw, 2.4rem);
      font-weight: 900;
      line-height: 1;
    }}

    .bar {{
      height: 10px;
      background: #e2e8f0;
      border-radius: 999px;
      overflow: hidden;
      border: 1px solid #d8e0ea;
    }}

    .bar span {{
      display: block;
      height: 100%;
      width: 0%;
      background: linear-gradient(90deg, #38bdf8, #2563eb);
      border-radius: inherit;
      transition: width 180ms ease;
    }}

    .model-card.alert .bar span {{
      background: linear-gradient(90deg, #fb7185, #dc2626);
    }}

    .model-card.manual {{
      border-color: #bfdbfe;
      box-shadow: 0 16px 46px rgba(37, 99, 235, 0.09);
    }}

    .result {{
      display: grid;
      gap: 6px;
      margin-top: auto;
      color: var(--muted);
      font-size: 0.88rem;
    }}

    .result strong {{
      color: var(--ink);
      font-size: 0.98rem;
    }}

    .tables {{
      display: grid;
      grid-template-columns: minmax(0, 1.2fr) minmax(300px, 0.8fr);
      gap: 18px;
      margin-top: 18px;
    }}

    .guide-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 14px;
    }}

    .guide-card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      box-shadow: var(--shadow);
      line-height: 1.55;
    }}

    .guide-card h3 {{
      margin: 0 0 8px;
      font-size: 1rem;
    }}

    .guide-card p {{
      margin: 0;
      color: var(--muted);
      font-size: 0.92rem;
    }}

    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.9rem;
    }}

    th, td {{
      text-align: left;
      border-bottom: 1px solid var(--line);
      padding: 10px 8px;
      vertical-align: middle;
    }}

    th {{
      color: var(--muted);
      font-size: 0.78rem;
      text-transform: uppercase;
      font-weight: 850;
    }}

    .features {{
      display: grid;
      gap: 10px;
    }}

    .feature-row {{
      display: grid;
      grid-template-columns: 56px minmax(0, 1fr) 76px;
      gap: 10px;
      align-items: center;
      font-size: 0.86rem;
    }}

    .feature-track {{
      height: 8px;
      border-radius: 999px;
      background: #e2e8f0;
      position: relative;
      overflow: hidden;
    }}

    .feature-track span {{
      position: absolute;
      height: 100%;
      left: 50%;
      width: 0;
      background: var(--teal);
    }}

    .feature-track span.negative {{
      left: auto;
      right: 50%;
      background: var(--amber);
    }}

    footer {{
      color: var(--muted);
      font-size: 0.86rem;
      margin-top: 18px;
      line-height: 1.55;
    }}

    @media (max-width: 1180px) {{
      .checker, .transaction-summary {{
        grid-template-columns: 1fr;
      }}

      .sample-list {{
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }}

      .model-grid {{
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }}
    }}

    @media (max-width: 980px) {{
      header, .grid, .checker, .tables, .manual-layout, .transaction-summary {{
        grid-template-columns: 1fr;
      }}

      .quick-start {{
        position: static;
      }}

      .stats-grid, .model-grid, .hint-grid, .guide-grid, .prepared-grid, .decision-facts {{
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }}

      .manual-grid {{
        grid-template-columns: repeat(3, minmax(0, 1fr));
      }}
    }}

    @media (max-width: 640px) {{
      .shell {{ width: min(100% - 22px, 1320px); padding-top: 16px; }}
      .stats-grid, .model-grid, .toolbar, .manual-actions, .manual-grid, .hint-grid, .guide-grid, .prepared-grid, .prepared-actions, .simple-actions, .decision-facts {{ grid-template-columns: 1fr; }}
      .hero {{ padding: 20px; min-height: 0; }}
      .panel {{ padding: 14px; }}
      th, td {{ padding: 9px 4px; font-size: 0.82rem; }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <header>
      <section class="hero">
        <div>
          <div class="eyebrow">Week 6 synthetic fraud deployment</div>
          <h1>Credit Card Fraud Checker</h1>
          <p>Pick a prepared transaction or enter your own 30 numeric values. The browser runs three validated models and shows one clear result first, with details available underneath.</p>
        </div>
      </section>
      <aside class="status-panel">
        <div id="runtimeState" class="runtime-state warn"><span class="dot"></span><span>Loading browser inference</span></div>
        <ul class="fact-list">
          <li><span>Model threshold</span><span>0.50</span></li>
          <li><span>Split seed</span><span>42</span></li>
          <li><span>Prepared samples</span><span id="preparedCount">-</span></li>
          <li><span>Random pool rows</span><span id="demoCount">-</span></li>
          <li><span>Fraud in pool</span><span id="demoFraudCount">-</span></li>
        </ul>
      </aside>
    </header>

    <section class="stats-grid" id="statsGrid"></section>

    <nav class="page-tabs" aria-label="Demo sections">
      <button class="tab-button active" data-view-target="sample">Fraud Checker</button>
      <button class="tab-button" data-view-target="manual">Enter Details</button>
      <button class="tab-button" data-view-target="metrics">Model Performance</button>
      <button class="tab-button" data-view-target="guide">Help</button>
    </nav>

    <main class="checker view-page active" data-view="sample">
      <section class="panel quick-start">
        <div class="step-label">Step 1</div>
        <h2>Choose a transaction</h2>
        <p class="section-intro">Use prepared examples to understand the cases, or click a random button to draw from all 85,443 held-out test rows.</p>
        <div class="sample-list" id="preparedGrid"></div>
        <div class="simple-actions">
          <button class="button" id="randomFraud">Random fraud</button>
          <button class="button" id="randomNormal">Random normal</button>
          <button class="button" id="randomAny">Any random</button>
          <button class="button" id="nextCase">Another random</button>
        </div>
        <div class="select-row">
          <label for="sampleSelect">Or choose a saved explanation row</label>
          <select id="sampleSelect"></select>
        </div>
      </section>

      <section class="result-shell">
        <section class="panel">
          <div class="step-label">Step 2</div>
          <h2>Prediction result</h2>
          <p class="section-intro">The demo summarizes the three models into one easy answer, then keeps the real label visible for learning.</p>
          <div id="decisionPanel" class="decision-card">
            <div class="decision-kicker">Waiting for selection</div>
            <div class="decision-main mixed">Choose a sample</div>
            <p class="decision-explain">Pick a prepared sample on the left to run the models.</p>
          </div>
        </section>

        <section class="panel transaction-summary">
          <div>
            <div class="step-label">Step 3</div>
            <h2>Transaction summary</h2>
            <div class="truth-box">
              <div class="truth-title">
                <strong id="truthTitle">-</strong>
                <span id="truthBadge" class="badge">-</span>
              </div>
              <ul class="fact-list">
                <li><span>Original row</span><span id="sourceIndex">-</span></li>
                <li><span>Amount</span><span id="amountValue">-</span></li>
                <li><span>Time</span><span id="timeValue">-</span></li>
              </ul>
            </div>
          </div>
          <div>
            <h2>Important features</h2>
            <p class="section-intro">These values update for the selected transaction. They are useful for explanation, not manual guesswork.</p>
            <div class="features" id="featureList"></div>
          </div>
        </section>
      </section>

      <section class="panel model-details">
        <details class="detail-box" open>
          <summary>Model-by-model probabilities</summary>
          <p class="section-intro">Open this when you want to compare the baseline, Gaussian-Copula augmented, and SMOTE augmented models.</p>
          <div class="model-grid" id="modelGrid"></div>
        </details>
      </section>
    </main>

    <section class="manual-layout view-page" data-view="manual">
      <section class="panel">
        <div class="step-label">Enter Details</div>
        <h2>Check your own transaction</h2>
        <p class="section-intro">Paste a full row from creditcard.csv, or load a prepared sample and edit it. The fastest reliable input is a copied CSV row with Time, V1 to V28, and Amount.</p>
        <div class="hint-grid">
          <div class="hint-card"><strong>Best input</strong>Paste one complete row in the dataset column order.</div>
          <div class="hint-card"><strong>Need a demo?</strong>Use a prepared sample from the checker page, then edit values here.</div>
          <div class="hint-card"><strong>Missing values</strong>Zero-fill is only for testing the UI, not for a real prediction.</div>
        </div>
        <div class="select-row">
          <label for="manualPaste">Paste transaction row</label>
          <textarea id="manualPaste" placeholder="JSON, CSV header+row, or comma-separated values in Time,V1,...,V28,Amount order"></textarea>
          <div class="inline-hint">Accepted: JSON with column names, a copied CSV row, or 30 comma-separated numbers.</div>
        </div>
        <div class="manual-actions">
          <button class="button" id="loadSampleToManual">Use current sample</button>
          <button class="button" id="parseManual">Parse pasted row</button>
          <button class="button" id="clearManual">Clear</button>
        </div>
        <label class="checkbox-row">
          <input type="checkbox" id="zeroMissingManual">
          <span>Allow zero-fill for blank fields. This is useful for UI testing, but a real prediction should use all actual feature values.</span>
        </label>
        <details class="detail-box">
          <summary>Advanced: edit individual numeric fields</summary>
          <p class="section-intro">Use this if you need to change one feature after pasting or loading a sample.</p>
          <div class="manual-grid" id="manualGrid"></div>
        </details>
        <div class="manual-actions">
          <button class="button primary" id="scoreManual">Detect fraud</button>
          <button class="button" id="zeroManual">Fill blank fields with 0</button>
          <button class="button" id="sampleFraudManual">Load random fraud</button>
        </div>
        <div class="manual-status" id="manualStatus"></div>
      </section>

      <section class="panel">
        <h2>Result</h2>
        <p class="section-intro">The top card gives the simple answer. Model-level details are shown below for explanation.</p>
        <div id="manualDecisionPanel" class="decision-card">
          <div class="decision-kicker">Waiting for input</div>
          <div class="decision-main mixed">No transaction yet</div>
          <p class="decision-explain">Paste or load transaction values, then click Detect fraud.</p>
        </div>
        <details class="detail-box" open>
          <summary>Model details</summary>
          <div class="model-grid" id="manualModelGrid"></div>
        </details>
      </section>
    </section>

    <section class="tables view-page" data-view="metrics">
      <section class="panel">
        <h2>Full Test-set Metrics</h2>
        <p class="section-intro">These numbers come from the real 85,443-row held-out test split, not from synthetic test data.</p>
        <table>
          <thead>
            <tr>
              <th>Model</th>
              <th>Correct fraud</th>
              <th>False alerts</th>
              <th>Recall</th>
              <th>Precision</th>
              <th>F1</th>
              <th>AUC</th>
            </tr>
          </thead>
          <tbody id="metricsBody"></tbody>
        </table>
      </section>

      <section class="panel">
        <h2>How to read this page</h2>
        <div class="hint-grid">
          <div class="hint-card"><strong>Recall</strong>Out of all real frauds, how many the model caught.</div>
          <div class="hint-card"><strong>Precision</strong>Out of all fraud alerts, how many were truly fraud.</div>
          <div class="hint-card"><strong>F1</strong>A balance between catching fraud and avoiding false alerts.</div>
        </div>
      </section>
    </section>

    <section class="view-page" data-view="guide">
      <div class="guide-grid">
        <article class="guide-card">
          <h3>1. Start with Fraud Checker</h3>
          <p>Pick one prepared transaction on the left. The result card gives the simple decision first.</p>
        </article>
        <article class="guide-card">
          <h3>2. Enter Details when needed</h3>
          <p>Paste a full dataset row or load a prepared transaction and edit it. Amount alone is not enough.</p>
        </article>
        <article class="guide-card">
          <h3>3. Use details only for explanation</h3>
          <p>The model cards and performance page are there for comparison after the main result is clear.</p>
        </article>
      </div>
    </section>

    <footer>
      The website runs embedded JavaScript versions of the trained Random Forest models. ONNX exports and validation files are saved with the deployment package for portability and project review.
    </footer>
  </div>

  <script id="payload" type="application/json">{payload_json}</script>
  <script>
    const payload = JSON.parse(document.getElementById("payload").textContent);
    const samples = payload.samples;
    const preparedSamples = payload.preparedSamples || [];
    const models = payload.models;
    const featureColumns = payload.featureColumns;
    let currentIndex = 0;
    let currentSample = null;
    let manualSourceSample = null;
    let randomManifest = null;
    let randomManifestPromise = null;
    let randomFraudRows = null;
    let randomFraudPromise = null;
    const randomChunkCache = new Map();

    const fmt = new Intl.NumberFormat("en-US", {{ maximumFractionDigits: 4 }});
    const pct = (value) => `${{(value * 100).toFixed(2)}}%`;

    function labelText(value) {{
      return value === 1 ? "Fraud" : "Non-fraud";
    }}

    function resultText(trueLabel, prediction) {{
      if (trueLabel === 1 && prediction === 1) return "Caught fraud";
      if (trueLabel === 1 && prediction === 0) return "Missed fraud";
      if (trueLabel === 0 && prediction === 1) return "False alert";
      return "Correct normal";
    }}

    function summarizePredictions(predictions) {{
      const fraudVotes = predictions.filter(([, result]) => result.prediction === 1).length;
      const avgProbability = predictions.reduce((sum, [, result]) => sum + result.fraud_probability, 0) / predictions.length;
      const maxProbability = Math.max(...predictions.map(([, result]) => result.fraud_probability));
      let state = "normal";
      let title = "Looks non-fraud";
      let explanation = "None of the deployed models crossed the 0.50 fraud threshold.";
      if (fraudVotes >= 2) {{
        state = "fraud";
        title = "Fraud likely";
        explanation = "Most models crossed the fraud threshold, so this transaction should be treated as suspicious.";
      }} else if (fraudVotes === 1) {{
        state = "mixed";
        title = "Needs review";
        explanation = "One model flagged fraud while the others did not. This is a borderline case.";
      }}
      return {{ state, title, explanation, fraudVotes, avgProbability, maxProbability }};
    }}

    function renderDecisionSummary(sample, predictions, targetId) {{
      const summary = summarizePredictions(predictions);
      const node = document.getElementById(targetId);
      const sampleName = sample.manual
        ? "Manual entry"
        : sample.random_pool
          ? `Random held-out row ${{sample.source_index.toLocaleString()}}`
          : `Sample #${{sample.sample_id}}`;
      const trueLabel = sample.true_label === null ? "Unknown" : labelText(sample.true_label);
      const labelNote = sample.true_label === null
        ? "No true label is available for a manually entered transaction."
        : `Actual test label: ${{trueLabel}}. Use this to check whether the model was right.`;
      const manualAction = sample.manual ? "" : `<button class="button primary" data-decision-action="manual">Open this in Enter Details</button>`;
      node.className = `decision-card ${{summary.state}}`;
      node.innerHTML = `
        <div class="decision-kicker">${{sampleName}}</div>
        <div class="decision-main ${{summary.state}}">${{summary.title}}</div>
        <p class="decision-explain">${{summary.explanation}} ${{labelNote}}</p>
        <div class="decision-facts">
          <div class="decision-fact"><span>Model vote</span><strong>${{summary.fraudVotes}} / ${{models.length}} say fraud</strong></div>
          <div class="decision-fact"><span>Average fraud probability</span><strong>${{pct(summary.avgProbability)}}</strong></div>
          <div class="decision-fact"><span>Highest fraud probability</span><strong>${{pct(summary.maxProbability)}}</strong></div>
        </div>
        <div class="decision-actions">
          ${{manualAction}}
          <button class="button" data-view-target="metrics">View model performance</button>
        </div>
      `;
    }}

    function highlightPreparedSample(sampleId) {{
      document.querySelectorAll("[data-sample-id]").forEach((node) => {{
        node.classList.toggle("active", Number(node.dataset.sampleId) === Number(sampleId));
      }});
    }}

    function setRuntime(message, isWarn = false) {{
      const node = document.getElementById("runtimeState");
      node.classList.toggle("warn", isWarn);
      node.innerHTML = `<span class="dot"></span><span>${{message}}</span>`;
    }}

    function findSampleByFeatures(features) {{
      return samples.find((sample) => sample.features.every((value, index) => value === features[index]));
    }}

    function setView(view) {{
      document.querySelectorAll("[data-view]").forEach((node) => {{
        node.classList.toggle("active", node.dataset.view === view);
      }});
      document.querySelectorAll("[data-view-target]").forEach((button) => {{
        button.classList.toggle("active", button.dataset.viewTarget === view);
      }});
    }}

    function findSampleById(sampleId) {{
      return samples.find((sample) => Number(sample.sample_id) === Number(sampleId));
    }}

    function sampleFromFullRow(row, index) {{
      const features = row.slice(2).map(Number);
      return {{
        sample_id: `random-${{index + 1}}`,
        source_index: Number(row[0]),
        true_label: Number(row[1]),
        amount: features[featureColumns.indexOf("Amount")],
        time: features[featureColumns.indexOf("Time")],
        features,
        random_pool: true,
      }};
    }}

    function randomItem(items) {{
      return items[Math.floor(Math.random() * items.length)];
    }}

    async function loadRandomManifest() {{
      if (randomManifest) return randomManifest;
      if (!randomManifestPromise) {{
        setRuntime("Loading random row index", true);
        randomManifestPromise = fetch(payload.randomRows.manifest)
          .then((response) => {{
            if (!response.ok) throw new Error(`Could not load random row index (${{response.status}}).`);
            return response.json();
          }})
          .then((data) => {{
            randomManifest = data;
            setRuntime(`Random pool ready: ${{data.totalRows.toLocaleString()}} rows`, false);
            return data;
          }})
          .catch((error) => {{
            setRuntime("Random row index failed to load", true);
            throw error;
          }});
      }}
      return randomManifestPromise;
    }}

    async function loadRandomChunk(chunk) {{
      if (randomChunkCache.has(chunk.file)) return randomChunkCache.get(chunk.file);
      const manifest = await loadRandomManifest();
      const chunkPromise = fetch(`data/${{manifest.chunkPath}}${{chunk.file}}`)
        .then((response) => {{
          if (!response.ok) throw new Error(`Could not load random row chunk (${{response.status}}).`);
          return response.json();
        }});
      randomChunkCache.set(chunk.file, chunkPromise);
      return chunkPromise;
    }}

    async function loadRandomFraudRows() {{
      if (randomFraudRows) return randomFraudRows;
      if (!randomFraudPromise) {{
        const manifest = await loadRandomManifest();
        randomFraudPromise = fetch(`data/${{manifest.fraudFile}}`)
          .then((response) => {{
            if (!response.ok) throw new Error(`Could not load fraud example rows (${{response.status}}).`);
            return response.json();
          }})
          .then((data) => {{
            randomFraudRows = data.rows.map((row, index) => sampleFromFullRow(row, index));
            return randomFraudRows;
          }});
      }}
      return randomFraudPromise;
    }}

    async function pickRandomByGlobalIndex(manifest, globalIndex) {{
      const chunkIndex = Math.floor(globalIndex / manifest.chunkSize);
      const chunk = manifest.chunks[Math.min(chunkIndex, manifest.chunks.length - 1)];
      const chunkData = await loadRandomChunk(chunk);
      const row = chunkData.rows[globalIndex - chunk.start];
      return row ? sampleFromFullRow(row, globalIndex) : null;
    }}

    function renderPreparedSamples() {{
      const grid = document.getElementById("preparedGrid");
      if (!preparedSamples.length) {{
        grid.innerHTML = `<div class="prepared-empty">Prepared samples are not available in this build. Use the random sample browser below.</div>`;
        return;
      }}

      grid.innerHTML = preparedSamples.map((entry) => {{
        const sample = findSampleById(entry.sample_id);
        if (!sample) {{
          return `<div class="prepared-empty">${{entry.title}} is missing from the embedded sample pool.</div>`;
        }}
        const fraudVotes = models.filter((model) => sample.precomputed[model.id]?.prediction === 1).length;
        const tone = entry.tone || (entry.true_label === 1 ? "fraud" : "normal");
        return `
          <button class="sample-option ${{tone}}" data-prepared-action="run" data-sample-id="${{entry.sample_id}}">
            <span class="sample-option-title">
              <span>${{entry.title}}</span>
              <span class="badge ${{entry.true_label === 1 ? "fraud" : "normal"}}">${{labelText(entry.true_label)}}</span>
            </span>
            <span class="sample-option-body">${{entry.description}}</span>
            <span class="sample-option-meta">Amount ${{fmt.format(entry.amount)}} | ${{fraudVotes}} of ${{models.length}} models say fraud | row ${{entry.source_index.toLocaleString()}}</span>
          </button>
        `;
      }}).join("");
    }}

    function renderStatic() {{
      document.getElementById("demoCount").textContent = payload.dataset.testRows.toLocaleString();
      document.getElementById("demoFraudCount").textContent = payload.dataset.testFraud.toLocaleString();
      document.getElementById("preparedCount").textContent = preparedSamples.length.toLocaleString();

      const stats = [
        ["Full dataset", payload.dataset.totalRows.toLocaleString(), `${{payload.dataset.totalFraud}} fraud rows`],
        ["Training split", payload.dataset.trainRows.toLocaleString(), `${{payload.dataset.trainFraud}} real fraud rows`],
        ["Test split", payload.dataset.testRows.toLocaleString(), `${{payload.dataset.testFraud}} fraud rows`],
        ["Synthetic rows", payload.dataset.syntheticRows.toLocaleString(), "Gaussian and SMOTE candidates"],
      ];
      document.getElementById("statsGrid").innerHTML = stats.map(([label, value, sub]) => `
        <article class="metric-card">
          <div class="metric-label">${{label}}</div>
          <div class="metric-value">${{value}}</div>
          <div class="metric-sub">${{sub}}</div>
        </article>
      `).join("");

      const sampleSelect = document.getElementById("sampleSelect");
      sampleSelect.innerHTML = `<option value="-1">Random full-pool row selected</option>` + samples.map((sample, index) => `
        <option value="${{index}}">#${{sample.sample_id}} | row ${{sample.source_index}} | ${{labelText(sample.true_label)}} | amount ${{fmt.format(sample.amount)}}</option>
      `).join("");
      sampleSelect.addEventListener("change", () => {{
        const index = Number(sampleSelect.value);
        if (index >= 0) selectSample(index);
      }});
      renderPreparedSamples();

      document.getElementById("metricsBody").innerHTML = models.map((model) => `
        <tr>
          <td><strong>${{model.title}}</strong></td>
          <td>${{model.metrics.tp}} / ${{payload.dataset.testFraud}}</td>
          <td>${{model.metrics.fp}}</td>
          <td>${{pct(model.metrics.recall)}}</td>
          <td>${{pct(model.metrics.precision)}}</td>
          <td>${{model.metrics.f1.toFixed(4)}}</td>
          <td>${{model.metrics.roc_auc.toFixed(4)}}</td>
        </tr>
      `).join("");

      document.getElementById("randomAny").addEventListener("click", () => pickRandom());
      document.getElementById("randomFraud").addEventListener("click", () => pickRandom(1));
      document.getElementById("randomNormal").addEventListener("click", () => pickRandom(0));
      document.getElementById("nextCase").addEventListener("click", () => pickRandom());
      document.getElementById("preparedGrid").addEventListener("click", async (event) => {{
        const button = event.target.closest("[data-prepared-action]");
        if (!button) return;
        const sampleId = Number(button.dataset.sampleId);
        if (button.dataset.preparedAction === "run") {{
          await runPreparedSample(sampleId);
        }}
        if (button.dataset.preparedAction === "manual") {{
          await scorePreparedSample(sampleId);
        }}
      }});

      document.querySelectorAll("[data-view-target]").forEach((button) => {{
        button.addEventListener("click", () => setView(button.dataset.viewTarget));
      }});

      document.body.addEventListener("click", async (event) => {{
        const viewButton = event.target.closest("[data-view-target]");
        if (viewButton) setView(viewButton.dataset.viewTarget);
        const decisionButton = event.target.closest("[data-decision-action]");
        if (decisionButton?.dataset.decisionAction === "manual") {{
          loadManualFromSample(currentSample || samples[currentIndex]);
          await scoreManualTransaction();
        }}
      }});

      renderManualInputs();
      document.getElementById("manualModelGrid").innerHTML = `
        <article class="hint-card">
          <strong>No manual result yet</strong>
          Enter or paste transaction details, then choose Detect fraud.
        </article>
      `;
      document.getElementById("loadSampleToManual").addEventListener("click", () => loadManualFromSample(currentSample || samples[currentIndex]));
      document.getElementById("parseManual").addEventListener("click", parseManualPaste);
      document.getElementById("clearManual").addEventListener("click", clearManualInputs);
      document.getElementById("zeroManual").addEventListener("click", fillBlankManualWithZero);
      document.getElementById("sampleFraudManual").addEventListener("click", () => {{
        pickRandom(1, true);
      }});
      document.getElementById("scoreManual").addEventListener("click", scoreManualTransaction);
    }}

    async function initOnnx() {{
      setRuntime("Running local browser models", false);
    }}

    function transformFeatures(model, features) {{
      const values = features.slice();
      const scaler = model.jsModel?.scaler;
      if (!scaler) return values;
      return values.map((value, index) => (value - scaler.mean[index]) / scaler.scale[index]);
    }}

    function predictWithLocalModel(model, features) {{
      const jsModel = model.jsModel;
      const values = transformFeatures(model, features);
      let totalProbability = 0;
      for (const tree of jsModel.trees) {{
        let node = 0;
        while (tree.childrenLeft[node] !== -1) {{
          const featureIndex = tree.feature[node];
          node = values[featureIndex] <= tree.threshold[node]
            ? tree.childrenLeft[node]
            : tree.childrenRight[node];
        }}
        totalProbability += tree.proba1[node];
      }}
      const fraudProbability = totalProbability / jsModel.trees.length;
      return {{
        fraud_probability: fraudProbability,
        prediction: fraudProbability > model.threshold ? 1 : 0,
      }};
    }}

    async function predict(model, sample) {{
      return predictWithLocalModel(model, sample.features);
    }}

    async function predictFeatures(model, features) {{
      return predictWithLocalModel(model, features);
    }}

    async function renderPredictions(sample) {{
      const predictions = await Promise.all(models.map(async (model) => [model, await predict(model, sample)]));
      renderDecisionSummary(sample, predictions, "decisionPanel");
      document.getElementById("modelGrid").innerHTML = predictions.map(([model, result]) => {{
        const alert = result.prediction === 1;
        const status = resultText(sample.true_label, result.prediction);
        return `
          <article class="model-card ${{alert ? "alert" : ""}}">
            <div class="model-head">
              <div>
                <div class="model-title">${{model.title}}</div>
                <div class="model-subtitle">${{model.subtitle}}</div>
              </div>
              <span class="badge ${{alert ? "fraud" : "normal"}}">${{labelText(result.prediction)}}</span>
            </div>
            <div>
              <div class="probability">${{pct(result.fraud_probability)}}</div>
              <div class="model-subtitle">fraud probability</div>
            </div>
            <div class="bar"><span style="width:${{Math.max(0.5, result.fraud_probability * 100)}}%"></span></div>
            <div class="result">
              <strong>${{status}}</strong>
              <span>${{model.description}}</span>
              <span>Full-test recall ${{pct(model.metrics.recall)}} with ${{model.metrics.fp}} false alerts.</span>
            </div>
          </article>
        `;
      }}).join("");
    }}

    function renderFeatures(sample) {{
      const interesting = ["Amount", "Time", "V14", "V17", "V12", "V10", "V4", "V11"];
      const maxAbs = Math.max(...interesting.map((name) => {{
        const idx = featureColumns.indexOf(name);
        return idx >= 0 ? Math.abs(sample.features[idx]) : 0;
      }}), 1);

      document.getElementById("featureList").innerHTML = interesting.map((name) => {{
        const idx = featureColumns.indexOf(name);
        const value = idx >= 0 ? sample.features[idx] : 0;
        const width = Math.min(50, Math.abs(value) / maxAbs * 50);
        return `
          <div class="feature-row">
            <strong>${{name}}</strong>
            <div class="feature-track"><span class="${{value < 0 ? "negative" : ""}}" style="width:${{width}}%"></span></div>
            <span>${{fmt.format(value)}}</span>
          </div>
        `;
      }}).join("");
    }}

    function renderManualInputs() {{
      document.getElementById("manualGrid").innerHTML = featureColumns.map((name) => `
        <div class="manual-field">
          <label for="manual_${{name}}" title="${{featureHint(name)}}">${{name}}</label>
          <input id="manual_${{name}}" data-feature="${{name}}" title="${{featureHint(name)}}" placeholder="${{featurePlaceholder(name)}}" type="number" step="any" inputmode="decimal" autocomplete="off">
        </div>
      `).join("");
    }}

    function featureHint(name) {{
      if (name === "Time") return "Seconds elapsed from the first transaction in the dataset.";
      if (name === "Amount") return "Transaction amount. This helps, but it is not enough by itself.";
      return `${{name}} is an anonymized PCA feature from the original transaction. Paste it from the dataset row when possible.`;
    }}

    function featurePlaceholder(name) {{
      if (name === "Time") return "seconds";
      if (name === "Amount") return "amount";
      return "numeric";
    }}

    function setManualValues(values) {{
      values.forEach((value, index) => {{
        const input = document.querySelector(`[data-feature="${{featureColumns[index]}}"]`);
        input.value = Number.isFinite(value) ? String(value) : "";
      }});
      document.getElementById("manualStatus").textContent = "Transaction details loaded.";
    }}

    function loadManualFromSample(sample) {{
      setView("manual");
      manualSourceSample = sample;
      setManualValues(sample.features);
      document.getElementById("manualPaste").value = "";
      document.getElementById("manualStatus").textContent = `Loaded held-out sample #${{sample.sample_id}}.`;
    }}

    function clearManualInputs() {{
      manualSourceSample = null;
      document.querySelectorAll("[data-feature]").forEach((input) => input.value = "");
      document.getElementById("manualPaste").value = "";
      document.getElementById("manualStatus").textContent = "Manual transaction cleared.";
    }}

    function fillBlankManualWithZero() {{
      document.querySelectorAll("[data-feature]").forEach((input) => {{
        if (input.value.trim() === "") input.value = "0";
      }});
      document.getElementById("manualStatus").textContent = "Blank fields filled with 0. Replace zeros with actual values when available.";
    }}

    function parseManualPaste() {{
      const raw = document.getElementById("manualPaste").value.trim();
      if (!raw) {{
        document.getElementById("manualStatus").textContent = "Paste JSON, a CSV header+row, or 30 comma-separated values first.";
        return;
      }}

      try {{
        let values;
        if (raw.startsWith("{{")) {{
          const parsed = JSON.parse(raw);
          values = featureColumns.map((name) => Number(parsed[name]));
        }} else {{
          const lines = raw.split(/\\r?\\n/).map((line) => line.trim()).filter(Boolean);
          const splitLine = (line) => line.split(/[,\\t]/).map((part) => part.trim()).filter(Boolean);
          if (lines.length >= 2 && splitLine(lines[0]).some((part) => featureColumns.includes(part))) {{
            const headers = splitLine(lines[0]);
            const rowValues = splitLine(lines[1]);
            const lookup = {{}};
            headers.forEach((name, index) => {{
              lookup[name] = rowValues[index];
            }});
            values = featureColumns.map((name) => Number(lookup[name]));
          }} else {{
            const parts = raw.split(/[,\\t\\n ]+/).filter(Boolean);
            if (parts.length === featureColumns.length + 1) {{
              parts.pop();
            }}
            if (parts.length !== featureColumns.length) {{
              throw new Error(`Expected ${{featureColumns.length}} values, or 31 values with Class at the end. Received ${{parts.length}}.`);
            }}
            values = parts.map(Number);
          }}
        }}
        if (values.some((value) => !Number.isFinite(value))) {{
          throw new Error("Every transaction field must be numeric.");
        }}
        manualSourceSample = null;
        setManualValues(values);
        document.getElementById("manualStatus").textContent = "Pasted transaction parsed.";
      }} catch (error) {{
        document.getElementById("manualStatus").textContent = error.message;
      }}
    }}

    function readManualValues() {{
      const allowZeroFill = document.getElementById("zeroMissingManual").checked;
      const values = featureColumns.map((name) => {{
        const input = document.querySelector(`[data-feature="${{name}}"]`);
        if (input.value.trim() === "" && allowZeroFill) return 0;
        return Number(input.value);
      }});
      const missing = values
        .map((value, index) => Number.isFinite(value) ? null : featureColumns[index])
        .filter(Boolean);
      if (missing.length) {{
        throw new Error(`Missing numeric values: ${{missing.slice(0, 6).join(", ")}}${{missing.length > 6 ? "..." : ""}}. Fill them, paste a full row, or enable zero-fill.`);
      }}
      return values;
    }}

    async function scoreManualTransaction() {{
      try {{
        const features = readManualValues();
        const matchedManualSource = manualSourceSample
          && manualSourceSample.features.every((value, index) => value === features[index])
          ? manualSourceSample
          : null;
        const matchedSample = matchedManualSource || findSampleByFeatures(features);
        const manualSample = {{
          sample_id: matchedSample ? matchedSample.sample_id : "manual",
          source_index: matchedSample ? matchedSample.source_index : "Manual entry",
          true_label: matchedSample ? matchedSample.true_label : null,
          amount: features[featureColumns.indexOf("Amount")],
          time: features[featureColumns.indexOf("Time")],
          features,
          manual: true,
        }};
        renderFeatures(manualSample);
        await renderManualPredictions(manualSample);
        document.getElementById("manualStatus").textContent = "Manual transaction scored across all models. Review the result cards on the right.";
      }} catch (error) {{
        document.getElementById("manualStatus").textContent = error.message;
      }}
    }}

    async function renderManualPredictions(sample) {{
      const predictions = await Promise.all(models.map(async (model) => [model, await predictFeatures(model, sample.features)]));
      renderDecisionSummary(sample, predictions, "manualDecisionPanel");
      document.getElementById("manualModelGrid").innerHTML = predictions.map(([model, result]) => {{
        const alert = result.prediction === 1;
        return `
          <article class="model-card manual ${{alert ? "alert" : ""}}">
            <div class="model-head">
              <div>
                <div class="model-title">${{model.title}}</div>
                <div class="model-subtitle">Manual transaction</div>
              </div>
              <span class="badge ${{alert ? "fraud" : "normal"}}">${{labelText(result.prediction)}}</span>
            </div>
            <div>
              <div class="probability">${{pct(result.fraud_probability)}}</div>
              <div class="model-subtitle">fraud probability</div>
            </div>
            <div class="bar"><span style="width:${{Math.max(0.5, result.fraud_probability * 100)}}%"></span></div>
            <div class="result">
              <strong>${{alert ? "Detected as fraud" : "Detected as non-fraud"}}</strong>
              <span>${{model.description}}</span>
              <span>Threshold: ${{model.threshold.toFixed(2)}}.</span>
            </div>
          </article>
        `;
      }}).join("");
    }}

    async function runPreparedSample(sampleId) {{
      const index = samples.findIndex((sample) => Number(sample.sample_id) === Number(sampleId));
      if (index < 0) return;
      setView("sample");
      await selectSample(index);
    }}

    async function scorePreparedSample(sampleId) {{
      const sample = findSampleById(sampleId);
      if (!sample) return;
      loadManualFromSample(sample);
      await scoreManualTransaction();
      document.getElementById("manualStatus").textContent = `Prepared sample #${{sample.sample_id}} was loaded and scored across all models.`;
    }}

    async function selectSample(index) {{
      currentIndex = index;
      const sample = samples[currentIndex];
      await showSample(sample, String(currentIndex));
    }}

    async function showSample(sample, selectValue = "-1") {{
      currentSample = sample;
      document.getElementById("sampleSelect").value = selectValue;
      document.getElementById("truthTitle").textContent = sample.random_pool ? `Random row ${{sample.source_index.toLocaleString()}}` : `Sample #${{sample.sample_id}}`;
      document.getElementById("truthBadge").textContent = labelText(sample.true_label);
      document.getElementById("truthBadge").className = `badge ${{sample.true_label === 1 ? "fraud" : "normal"}}`;
      document.getElementById("sourceIndex").textContent = sample.source_index.toLocaleString();
      document.getElementById("amountValue").textContent = fmt.format(sample.amount);
      document.getElementById("timeValue").textContent = fmt.format(sample.time);
      highlightPreparedSample(sample.sample_id);
      renderFeatures(sample);
      await renderPredictions(sample);
    }}

    async function pickRandom(label = null, sendToManual = false) {{
      try {{
        const manifest = await loadRandomManifest();
        let picked = null;

        if (label === 1) {{
          const fraudRows = await loadRandomFraudRows();
          picked = randomItem(fraudRows);
        }} else {{
          for (let attempt = 0; attempt < 50 && !picked; attempt += 1) {{
            const globalIndex = Math.floor(Math.random() * manifest.totalRows);
            const candidate = await pickRandomByGlobalIndex(manifest, globalIndex);
            if (candidate && (label === null || candidate.true_label === label)) {{
              picked = candidate;
            }}
          }}

          if (!picked && label === 0) {{
            const chunk = randomItem(manifest.chunks);
            const chunkData = await loadRandomChunk(chunk);
            const normalRows = chunkData.rows.filter((row) => Number(row[1]) === 0);
            if (normalRows.length) {{
              const row = randomItem(normalRows);
              picked = sampleFromFullRow(row, chunk.start + chunkData.rows.indexOf(row));
            }}
          }}
        }}

        if (!picked) throw new Error("Could not find a matching random row. Try again.");
        await showSample(picked);
        if (sendToManual) {{
          loadManualFromSample(picked);
          await scoreManualTransaction();
        }}
      }} catch (error) {{
        setRuntime(error.message, true);
      }}
    }}

    renderStatic();
    const firstPreparedIndex = preparedSamples.length
      ? samples.findIndex((sample) => Number(sample.sample_id) === Number(preparedSamples[0].sample_id))
      : 0;
    initOnnx().finally(() => selectSample(firstPreparedIndex >= 0 ? firstPreparedIndex : 0));
  </script>
</body>
</html>
"""
    (DEPLOY_DIR / "index.html").write_text(html, encoding="utf-8")


def write_readme(payload: dict) -> None:
    lines = [
        "# Week 6 Static Deployment",
        "",
        "This folder contains a light-theme browser demo for the synthetic fraud detection project.",
        "",
        "## Files",
        "",
        "- `index.html` is the standalone interactive demo.",
        "- `models/*.onnx` are the exported browser models.",
        "- `data/heldout_test_samples.json` contains real held-out test rows used by the demo.",
        "- `data/prepared_demo_samples.json` contains the curated fraud/non-fraud examples shown as separate cards.",
        "- `data/random_rows_manifest.json`, `data/random_chunks/*.json`, and `data/random_fraud_rows.json` power fast random loading across all held-out rows.",
        "- `validation/onnx_validation.json` proves the ONNX threshold predictions match sklearn on the full held-out test set.",
        "- `build_deployment.py` rebuilds the deployment package.",
        "",
        "The visible website scores transactions with embedded JavaScript Random Forest models exported from sklearn. The ONNX files are still saved and validated as portable model artifacts, but the UI does not depend on ONNX Runtime Web, so it works reliably without CDN/runtime issues.",
        "",
        "## UI pages",
        "",
        "- `Fraud Checker` is the main guided workflow: choose a prepared transaction, then read the large result card.",
        "- `Enter Details` lets you paste JSON, a CSV header plus row, or 30 comma-separated values for a custom transaction.",
        "- `Model Performance` explains precision, recall, F1, AUC, and the correct-fraud counts for each model.",
        "- `Help` gives quick hints for using the demo without reading the code.",
        "",
        "The manual checker asks for numeric values for `Time`, `V1` to `V28`, and `Amount`, then classifies that transaction across all three models. Tooltips and short hints explain what each input means.",
        "",
        "## Local preview",
        "",
        "Open `index.html` in a browser. If the browser blocks local WebAssembly loading, run a tiny static server from this folder:",
        "",
        "```powershell",
        "cd \"E:\\GNCIPIL\\WEEK 6\\Synthetic-Fraud-AI-Project\\Deployment\"",
        "python -m http.server 8000",
        "```",
        "",
        "Then open `http://localhost:8000`.",
        "",
        "## Free hosting",
        "",
        "GitHub Pages:",
        "",
        "1. Commit the `Deployment` folder to the repository.",
        "2. In GitHub, go to Settings -> Pages.",
        "3. Choose Deploy from a branch.",
        "4. Select the branch and set the folder to `/WEEK 6/Synthetic-Fraud-AI-Project/Deployment` if GitHub offers that path. If it does not, copy `index.html` to a `docs` folder or use Netlify.",
        "",
        "Netlify:",
        "",
        "1. Go to Netlify and choose Add new site.",
        "2. Drag and drop this `Deployment` folder.",
        "3. Netlify will host it as a static site with no backend.",
        "",
        "## Current model summary",
        "",
        "| Model | Correct fraud | False alerts | Precision | Recall | F1 | AUC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    test_fraud = payload["dataset"]["testFraud"]
    for model in payload["models"]:
        metrics = model["metrics"]
        lines.append(
            f"| {model['title']} | {metrics['tp']} / {test_fraud} | {metrics['fp']} | "
            f"{metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1']:.4f} | {metrics['roc_auc']:.4f} |"
        )
    lines.append("")
    lines.append("The synthetic methods are useful to compare interactively because they trade precision and recall differently.")
    (DEPLOY_DIR / "README_DEPLOYMENT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    ensure_onnxruntime_assets()
    df = pd.read_csv(DATA_DIR / "creditcard.csv")
    features = [col for col in df.columns if col != "Class"]

    real_train, real_test, x_test, y_test, fitted_models = fit_models(df, features)
    samples = make_demo_samples(real_test, features, fitted_models)
    random_rows_manifest = make_chunked_test_rows(real_test, features)
    prepared_samples = make_prepared_samples(samples, list(fitted_models.keys()))
    sample_lookup = {sample["sample_id"]: sample for sample in samples}
    prepared_sample_rows = [
        {**entry, "features": sample_lookup[entry["sample_id"]]["features"], "precomputed": sample_lookup[entry["sample_id"]]["precomputed"]}
        for entry in prepared_samples
    ]

    payload_models = []
    validations = {}
    for model_id, entry in fitted_models.items():
        print(f"Exporting and validating {entry['title']}...", flush=True)
        model = entry["model"]
        threshold = float(entry["threshold"])
        proba = model.predict_proba(x_test.to_numpy(dtype=np.float32))[:, 1]
        metrics = metrics_from_proba(y_test, proba, threshold)
        onnx_bytes = convert_model(model, model_id, len(features))
        validation = validate_onnx(model, onnx_bytes, x_test, threshold)
        validations[model_id] = validation
        joblib.dump(model, MODELS_DIR / f"{model_id}.pkl")

        payload_models.append(
            {
                "id": model_id,
                "title": entry["title"],
                "subtitle": entry["subtitle"],
                "description": entry["description"],
                "threshold": threshold,
                "trainingRows": entry["training_rows"],
                "trainingFraud": entry["training_fraud"],
                "syntheticRows": entry["synthetic_rows"],
                "metrics": metrics,
                "validation": validation,
                "jsModel": export_js_model(model),
            }
        )

    payload = {
        "featureColumns": features,
        "dataset": {
            "totalRows": int(len(df)),
            "totalFraud": int(df["Class"].sum()),
            "totalNonFraud": int((df["Class"] == 0).sum()),
            "trainRows": int(len(real_train)),
            "trainFraud": int(real_train["Class"].sum()),
            "testRows": int(len(real_test)),
            "testFraud": int(real_test["Class"].sum()),
            "syntheticRows": N_SYNTHETIC_FRAUD,
        },
        "models": payload_models,
        "samples": samples,
        "preparedSamples": prepared_samples,
        "randomRows": {
            "manifest": "data/random_rows_manifest.json",
            "totalRows": random_rows_manifest["totalRows"],
            "fraudRows": random_rows_manifest["fraudRows"],
            "normalRows": random_rows_manifest["normalRows"],
        },
    }

    (DATA_OUT_DIR / "heldout_test_samples.json").write_text(
        json.dumps(samples, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    (DATA_OUT_DIR / "prepared_demo_samples.json").write_text(
        json.dumps(prepared_sample_rows, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    (DATA_OUT_DIR / "random_rows_manifest.json").write_text(
        json.dumps(random_rows_manifest, separators=(",", ":"), ensure_ascii=True),
        encoding="utf-8",
    )
    legacy_full_rows = DATA_OUT_DIR / "all_heldout_test_rows.json"
    if legacy_full_rows.exists():
        legacy_full_rows.unlink()
    (VALIDATION_DIR / "onnx_validation.json").write_text(
        json.dumps(validations, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    (DEPLOY_DIR / "deployment_manifest.json").write_text(
        json.dumps(
            {
                "dataset": payload["dataset"],
                "models": [
                    {key: value for key, value in model.items() if key not in {"onnxBase64", "jsModel"}}
                    for model in payload_models
                ],
                "sample_count": len(samples),
                "prepared_sample_count": len(prepared_samples),
                "random_pool_count": random_rows_manifest["totalRows"],
                "random_pool_chunks": len(random_rows_manifest["chunks"]),
            },
            indent=2,
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )
    write_index_html(payload)
    write_readme(payload)

    print("\nDeployment package written to:", DEPLOY_DIR, flush=True)
    for model in payload_models:
        m = model["metrics"]
        print(
            f"{model['title']}: TP={m['tp']} FP={m['fp']} "
            f"precision={m['precision']:.4f} recall={m['recall']:.4f} f1={m['f1']:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
