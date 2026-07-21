# Week 6 Static Deployment

This folder contains a light-theme browser demo for the synthetic fraud detection project.

## Files

- `index.html` is the standalone interactive demo.
- `models/*.onnx` are the exported browser models.
- `data/heldout_test_samples.json` contains real held-out test rows used by the demo.
- `data/prepared_demo_samples.json` contains the curated fraud/non-fraud examples shown as separate cards.
- `data/random_rows_manifest.json`, `data/random_chunks/*.json`, and `data/random_fraud_rows.json` power fast random loading across all held-out rows.
- `validation/onnx_validation.json` proves the ONNX threshold predictions match sklearn on the full held-out test set.
- `build_deployment.py` rebuilds the deployment package.

The visible website scores transactions with embedded JavaScript Random Forest models exported from sklearn. The ONNX files are still saved and validated as portable model artifacts, but the UI does not depend on ONNX Runtime Web, so it works reliably without CDN/runtime issues.

## UI pages

- `Fraud Checker` is the main guided workflow: choose a prepared transaction, then read the large result card.
- `Enter Details` lets you paste JSON, a CSV header plus row, or 30 comma-separated values for a custom transaction.
- `Model Performance` explains precision, recall, F1, AUC, and the correct-fraud counts for each model.
- `Help` gives quick hints for using the demo without reading the code.

The manual checker asks for numeric values for `Time`, `V1` to `V28`, and `Amount`, then classifies that transaction across all three models. Tooltips and short hints explain what each input means.

## Local preview

Open `index.html` in a browser. If the browser blocks local WebAssembly loading, run a tiny static server from this folder:

```powershell
cd "E:\GNCIPIL\WEEK 6\Synthetic-Fraud-AI-Project\Deployment"
python -m http.server 8000
```

Then open `http://localhost:8000`.

## Free hosting

GitHub Pages:

1. Commit the `Deployment` folder to the repository.
2. In GitHub, go to Settings -> Pages.
3. Choose Deploy from a branch.
4. Select the branch and set the folder to `/WEEK 6/Synthetic-Fraud-AI-Project/Deployment` if GitHub offers that path. If it does not, copy `index.html` to a `docs` folder or use Netlify.

Netlify:

1. Go to Netlify and choose Add new site.
2. Drag and drop this `Deployment` folder.
3. Netlify will host it as a static site with no backend.

## Current model summary

| Model | Correct fraud | False alerts | Precision | Recall | F1 | AUC |
|---|---:|---:|---:|---:|---:|---:|
| Baseline-tuned | 112 / 148 | 5 | 0.9573 | 0.7568 | 0.8453 | 0.9671 |
| Gaussian-Copula augmented | 122 / 148 | 24 | 0.8356 | 0.8243 | 0.8299 | 0.9699 |
| SMOTE augmented | 118 / 148 | 7 | 0.9440 | 0.7973 | 0.8645 | 0.9743 |

The synthetic methods are useful to compare interactively because they trade precision and recall differently.