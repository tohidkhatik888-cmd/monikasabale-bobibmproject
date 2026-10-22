# MalwareGuard AI

> Static PE malware classification via REST API — powered by Random Forest and XGBoost.

---

## Overview

MalwareGuard AI analyses Windows PE (Portable Executable) files using **static analysis** — no execution required. It extracts ~50 features from the file's PE headers, section table, import/export directories, and byte-level entropy, then feeds them to a trained classifier to produce a **benign / malware** verdict with a confidence score.

| Feature group | Examples |
|---|---|
| PE header | Machine type, characteristics, subsystem, DLL flags |
| Optional header | Image base, section/file alignment, entry point RVA, sizes |
| Section table | Per-section entropy, raw/virtual sizes, r/w/x counts |
| Import table | Number of imported DLLs & functions |
| Export table | Number of exported symbols |
| File-level | File size, whole-file Shannon entropy |

Two classifier back-ends are available and can be selected per-request:

- **Random Forest** (`random_forest`) — fast, robust, low false-positive rate
- **XGBoost** (`xgboost`) — higher accuracy on imbalanced datasets

---

## Project Structure

```
malwareguard_ai/
├── app/
│   ├── api/
│   │   └── routes.py          # FastAPI router (all endpoints)
│   ├── core/
│   │   └── config.py          # Pydantic settings (env-based)
│   ├── models/
│   │   └── schemas.py         # Request / response Pydantic models
│   ├── services/
│   │   ├── classifier.py      # Model loading & inference service
│   │   └── feature_extractor.py  # PE static feature extraction
│   └── main.py                # FastAPI app factory + lifespan
├── data/
│   ├── models/                # Saved .joblib model artefacts (git-ignored)
│   └── samples/               # PE sample dataset (benign/ & malware/)
├── scripts/
│   └── train.py               # Model training CLI
├── tests/
│   └── test_api.py            # pytest unit + integration tests
├── main.py                    # Uvicorn entry point
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies

```bash
cd malwareguard_ai
pip install -r requirements.txt
```

### 2. Prepare a dataset

Organise labelled PE files as:

```
data/samples/
    benign/     ← clean executables
    malware/    ← malicious executables
```

### 3. Train the models

```bash
# From malwareguard_ai/
python scripts/train.py \
    --data-dir  data/samples \
    --classifier both \
    --output-dir data/models
```

This produces `random_forest.joblib`, `xgboost.joblib`, `scaler.joblib`, and `feature_names.joblib` in `data/models/`.

### 4. Run the API server

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Interactive docs: **http://localhost:8000/docs**

---

## API Reference

### `GET /api/v1/health`

Returns liveness status and which models are loaded.

```json
{
  "status": "ok",
  "version": "1.0.0",
  "models_loaded": { "random_forest": true, "xgboost": true }
}
```

---

### `POST /api/v1/classify`

Classify a single PE file.

**Form fields**

| Field | Type | Default | Description |
|---|---|---|---|
| `file` | binary | — | PE executable |
| `classifier` | string | `random_forest` | `random_forest` or `xgboost` |

**Response**

```json
{
  "filename": "suspicious.exe",
  "file_size_bytes": 204800,
  "label": "malware",
  "label_index": 1,
  "confidence": 0.97,
  "probabilities": { "benign": 0.03, "malware": 0.97 },
  "classifier_used": "random_forest",
  "is_malware": true
}
```

---

### `POST /api/v1/classify/batch`

Classify multiple PE files in one request.

**Form fields**

| Field | Type | Description |
|---|---|---|
| `files` | binary[] | One or more PE files |
| `classifier` | string | `random_forest` or `xgboost` |

---

### `POST /api/v1/features`

Return the raw extracted feature vector for a PE file (useful for model debugging).

---

## Configuration

All settings can be overridden via environment variables or a `.env` file:

| Variable | Default | Description |
|---|---|---|
| `DEFAULT_CLASSIFIER` | `random_forest` | Classifier used when not specified in request |
| `CONFIDENCE_THRESHOLD` | `0.5` | Minimum confidence to assign a label |
| `MAX_FILE_SIZE_MB` | `50` | Maximum upload size in MB |
| `MODEL_DIR` | `data/models` | Directory containing `.joblib` artefacts |
| `DEBUG` | `false` | Enable FastAPI debug mode |

---

## Running Tests

```bash
# From malwareguard_ai/
pytest tests/ -v
```

The test suite includes:
- **Unit tests** — feature extractor output shape, entropy bounds, invalid-file handling
- **Integration tests** — all API endpoints via `TestClient` (no real models required)

---

## Notes

- No execution or sandboxing is performed; all analysis is purely static.
- The training script expects PE files; non-PE files are skipped with a warning.
- `data/models/` should be added to `.gitignore` to avoid committing large binary artefacts.
