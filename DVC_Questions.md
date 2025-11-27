# 🚀 **What is DVC (in simple words)**

DVC is **Git for data + ML pipelines**.

It helps you version:

* datasets
* preprocessing outputs
* model files
* ML pipelines

…without putting large files directly in GitHub.

You get reproducibility **without polluting your repo**.

---

# 🔧 Why DVC is important in MLOps

### ✔ Decouples large files from Git

Git can’t handle big datasets.
DVC stores them **in remote storage** (Google Drive, S3, local folder, etc.).

### ✔ Ensures reproducibility

Run the same experiment months later → get the **exact same data** and **pipeline output**.

### ✔ Works perfectly with MLflow

MLflow tracks:

* parameters
* metrics
* model versions

DVC tracks:

* data
* intermediate artifacts
* pipeline dependencies

---

# 🧭 Part 1 — Install DVC

Pick storage backend:

* If you want something easy: **Google Drive**
* If offline: **local folder**

Install:

```bash
pip install dvc
```

Optionally (Google Drive):

```bash
pip install dvc[gdrive]
```

---

# 🧭 Part 2 — Initialize DVC

Inside your ML project folder:

```bash
dvc init
```

This creates:

* `.dvc/` directory
* `.dvcignore`
* hooks for Git integration

Add these to Git:

```bash
git add .dvc .dvcignore
git commit -m "Initialize DVC"
```

---

# 🧭 Part 3 — Track your dataset

Example directory:

```
data/
   train.csv
   test.csv
```

Track it with DVC:

```bash
dvc add data/train.csv
```

This creates:

* `data/train.csv.dvc` (small metadata file)
* DVC moves the actual data into its cache

Commit the metadata to Git:

```bash
git add data/train.csv.dvc .gitignore
git commit -m "Track training data with DVC"
```

**Important:**
Your GitHub repo *never sees the actual CSV file*.
Only the `.dvc` metadata is stored.

---

# 🧭 Part 4 — Add remote storage

Local remote (easy):

```bash
dvc remote add -d myremote /mnt/dvc-storage
```

Google Drive remote:

```bash
dvc remote add -d myremote gdrive://<folder-id>
```

Push your data:

```bash
dvc push
```

Now your dataset is safely stored in remote storage.

---

# 🧭 Part 5 — Reproduce pipelines (DVC magic)

Create a training script:

```
src/train.py
```

Register a pipeline stage:

```bash
dvc stage add -n train \
  -d src/train.py -d data/train.csv \
  -o model.pkl \
  python src/train.py
```

This creates `dvc.yaml` with:

```yaml
stages:
  train:
    cmd: python src/train.py
    deps:
    - src/train.py
    - data/train.csv
    outs:
    - model.pkl
```

Run the pipeline:

```bash
dvc repro
```

This is **DVC’s most powerful feature** → it rebuilds only what changed.

---

# 🧭 Part 6 — Working with teammates

Clone repo:

```bash
git clone <repo>
```

Get the data:

```bash
dvc pull
```

Now you have:

* same data
* same scripts
* same pipeline outputs

Repro guarantees **identical execution**.

---

# 🧠 How DVC + MLflow fit together

| Need                    | Tool             |
| ----------------------- | ---------------- |
| Track model metrics     | MLflow           |
| Track model artifacts   | MLflow           |
| Track model versions    | MLflow           |
| Track data versions     | DVC              |
| Track pipeline stages   | DVC              |
| Reproduce full training | DVC              |
| Deploy model            | FastAPI / Docker |
| Automate                | Airflow          |

They complement each other, not replace.

---
