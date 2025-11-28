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



### **✔ Corrected workflow:**

1. I download a large dataset locally (e.g., 20GB).
2. I run `dvc add` to track it.
3. I commit only the `.dvc` metadata file to Git.
4. I push the actual data to a **DVC remote** (S3/Drive/etc) using:

   ```
   dvc push
   ```
5. My friend clones the repo from Git:

   ```
   git clone <repo>
   ```
6. Then they download the real dataset from the DVC remote using:

   ```
   dvc pull
   ```
7. Now they have the exact same 20GB dataset without needing my local files.

---

# 🎯 Quick analogy to know you understood it

| Step               | Meaning                                 |
| ------------------ | --------------------------------------- |
| `.dvc` file in Git | Shopping list                           |
| DVC remote         | Supermarket                             |
| `dvc pull`         | Buying the actual food                  |
| Your friend        | Another cook using the same ingredients |

Your friend doesn’t get ingredients from your notebook —
they get them from the supermarket (DVC remote).


# 🎯 **DVC 80/20 — Intermediate Level (Part 2)**

*The next 20% of the features that give you another 60% of the value.*

We’ll cover:

1. **DVC data lifecycle (cache, remotes, .dvc files, locking)**
2. **DVC pipelines: real-world patterns**
3. **DVC experiment tracking (the underrated gem)**
4. **Best folder structures used by MLOps teams**
5. **Common mistakes & anti-patterns**
6. **DVC + MLflow integration pattern**

Let’s go.

---

# **1️⃣ DVC Data Lifecycle (the part most beginners misunderstand)**

When you run:

```
dvc add data/train.csv
```

Four important things happen:

### **1. Data is moved to the DVC cache**

Located at:

```
.dvc/cache/
```

This is DVC’s “private Git LFS-like” store.

### **2. The working directory gets a *symlink* to the cache**

Your `data/train.csv` is no longer a real file — it's a link to cache.

This lets DVC:

* deduplicate files
* track hashes
* detect changes quickly

### **3. A small `train.csv.dvc` metadata file appears**

Contains:

* file hash
* file path
* version info

### **4. This `.dvc` file is what you commit to Git**

Not the data.

### **⭐ Why this matters**

If you understand DVC cache + .dvc metadata + remote storage,
you understand 80% of DVC internals.

---

# **2️⃣ DVC Pipelines — 80/20 Edition**

Most real workflows use **3–5 stages**:

* data preparation
* feature generation
* training
* evaluation
* deployment artifact creation

A real-world pattern:

```bash
dvc stage add -n prepare \
  -d src/prepare.py -d data/raw \
  -o data/processed \
  python src/prepare.py

dvc stage add -n features \
  -d src/features.py -d data/processed \
  -o data/features \
  python src/features.py

dvc stage add -n train \
  -d src/train.py -d data/features \
  -o model.pkl \
  python src/train.py
```

Produces a clean `dvc.yaml` like:

### ⭐ Why this matters

DVC pipelines are **file-based dependency graphs**.

Change a file → only affected stages rerun.

This gives:

* speed
* reproducibility
* full ML workflow documentation

That’s 95% of why companies adopt DVC.

---

# **3️⃣ DVC Experiments (the most underrated feature)**

Most people don’t know this,
but **DVC has experiment tracking similar to MLflow**.

You run:

```
dvc exp run
```

And DVC:

* stores metrics
* stores parameters
* stores artifacts
* version-controls everything automatically

You can list experiments:

```
dvc exp show
```

Compare them:

```
dvc exp diff
```

Promote the best experimental run into Git:

```
dvc exp apply <id>
```

### ⭐ When to use DVC experiments?

* Fast prototyping
* Branchless experimentation
* When MLflow is “too much” for a simple project

This is an 80/20 feature that saves hours per week.

---

# **4️⃣ Recommended ML Project Folder Structure**

Here is the structure used in many MLOps teams:

```
project/
│
├── data/              ← raw and processed data (DVC tracked)
│
├── src/
│   ├── data/          ← data loading scripts
│   ├── features/      ← feature engineering
│   ├── models/        ← model training scripts
│   ├── utils/
│
├── models/            ← model.pkl, checkpoints (DVC tracked)
│
├── dvc.yaml
├── dvc.lock
├── params.yaml        ← hyperparameters
│
└── README.md
```

### ⭐ Why this matters

DVC works best when:

* data is in `data/`
* models are in `models/`
* code is separate (inside `src/`)

This 80/20 structure prevents pipeline spaghetti.

---

# **5️⃣ Common Mistakes (to save weeks of pain)**

### **❌ 1. Tracking too many files separately**

Bad:

```
dvc add data/image1.jpg
dvc add data/image2.jpg
```

Good:

```
dvc add data/images/
```

### **❌ 2. Using Git LFS + DVC together**

Redundant and slows everything down.

### **❌ 3. Storing data inside the repo BEFORE running dvc add**

Always:

* drop data into folder
* immediately `dvc add`
* **never commit data to Git**

### **❌ 4. Forgetting to `dvc push` before switching machines**

Rule:

> After *any* DVC add/repro:
> **Always run `dvc push`**

---

# **6️⃣ DVC + MLflow — The Perfect Workflow**

Here’s the industry-standard:

| What                     | Tool                       |
| ------------------------ | -------------------------- |
| Data versioning          | DVC                        |
| Pipeline reproducibility | DVC                        |
| Dataset storage          | S3 / GDrive via DVC        |
| Experiment metrics       | MLflow                     |
| Hyperparameters          | params.yaml (DVC) + MLflow |
| Model registry           | MLflow Model Registry      |
| Deployment               | MLflow, Docker, FastAPI    |

### Concrete flow:

1. Store datasets with **DVC**
2. Use **DVC pipeline** to run the ML workflow
3. During training, log metrics to **MLflow**
4. Version the final model in **MLflow Registry**
5. Deploy via FASTAPI + Docker

### ⭐ Why this works

DVC and MLflow don’t overlap.
They solve different parts of the ML lifecycle.

---

# 🎓 Part 2 Summary — The 80/20 DVC Next-Level Checklist

You now know the essentials PLUS the next layer:

### ✔ DVC cache structure

### ✔ Understanding `.dvc` files

### ✔ Best practice folder structure

### ✔ Multi-stage pipelines

### ✔ DVC experiments (exp run / exp diff / exp apply)

### ✔ DVC + MLflow integration

### ✔ Common mistakes to avoid

If you master the topics above, you’re already more advanced than **90% of DVC users**.

