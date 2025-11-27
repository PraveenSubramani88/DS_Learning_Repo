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


👉 *Learn the 20% of features that give you 80% of the real value.*

Here is exactly what you need to know — **no fluff**, just the essentials that matter for real MLOps work.

---

# 🎯 **DVC: The 20% You Should Master (for 80% of the benefits)**

Below are the only 5 things you truly need.

---

# **1️⃣ DVC adds version control to data (NOT stored in Git)**

This is the core idea.

Git can’t store big files, so DVC stores your dataset somewhere else (local folder, Google Drive, S3, etc.) and keeps only a small `.dvc` pointer in your repo.

**You must understand:**

* `dvc add <file>`
* `.dvc` files point to real data
* data lives in DVC remote storage

### Commands:

```
dvc add data/train.csv
dvc push
dvc pull
```

That’s 80% of what people use DVC for.

---

# **2️⃣ DVC keeps your repo clean using a `.dvc` file**

When you run:

```bash
dvc add data/train.csv
```

You get:

```
data/train.csv.dvc   ← tracked
data/.gitignore      ← file ignored
```

This is how data can be versioned *without* living inside GitHub.

You don’t need deep knowledge — just understand:

* `.dvc` files = metadata
* actual data = stored separately

---

# **3️⃣ DVC remote storage: push / pull**

This is the *next most important* concept.

You set up a remote:

```
dvc remote add -d storage gdrive://<id>
```

Then:

```bash
dvc push   # upload data to remote
dvc pull   # download data from remote
```

Everyone on your team can **pull the exact same dataset**.

That's 80% of team reproducibility right there.

---

# **4️⃣ DVC Pipelines (dvc.yaml)**

This is where DVC becomes *really* useful.

You define pipeline steps as dependencies:

```
dvc stage add -n train \
  -d data/train.csv \
  -d train.py \
  -o model.pkl \
  python train.py
```

This generates **dvc.yaml**, which describes your entire ML workflow.

### Key idea:

DVC uses file dependencies to determine what needs to be re-run.

If the data changes → retrain
If nothing changed → skip training

### Run whole pipeline:

```
dvc repro
```

This is the most advanced part but still essential 20%.

---

# **5️⃣ Everything is reproducible**

The best part:

With just:

```
git clone <repo>
dvc pull
dvc repro
```

You can rebuild the entire model **exactly as the original author did**.

This is the real magic of DVC.

---

# 🎯 **Summary: 80/20 DVC Skills Checklist**

| Must know                          | Why                        |
| ---------------------------------- | -------------------------- |
| `dvc init`                         | start project              |
| `dvc add`                          | track datasets without Git |
| `.dvc` pointer files               | understand metadata        |
| `dvc push/pull`                    | share datasets             |
| remote storage (local, gdrive, s3) | team collaboration         |
| `dvc stage add`                    | define pipeline steps      |
| `dvc repro`                        | reproduce ML workflow      |
| `dvc.yaml` basics                  | pipeline description       |

That's all you need to be productive.
