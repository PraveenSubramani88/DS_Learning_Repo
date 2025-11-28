Nice — below I’ve added **simple, copy-pasteable code** for each of the 10 beginner (80/20) items. Each snippet is short and focused so you can run it and see the behaviour quickly.

> Notes before you run:
> - These examples assume `mlflow` is installed (`pip install mlflow`) and (for the sklearn examples) `scikit-learn` is installed.
> - By default MLflow will create a local `./mlruns/` directory to store runs & artifacts.  
> - To open the UI after running an example: run `mlflow ui` in your terminal and open `http://localhost:5000`.

---

# ✅ 1 — What is MLflow Tracking used for?
MLflow Tracking is used to record your ML experiments, so you can compare runs, see what worked, and reproduce results later.
Short example: create an experiment, start a run, and log a metric.

```python
import mlflow

# set or create a named experiment
mlflow.set_experiment("my_first_experiment")

# start a run and log a metric
with mlflow.start_run(run_name="simple_run"):
    mlflow.log_param("model_type", "baseline")
    mlflow.log_metric("accuracy", 0.72)
    # artifacts can be logged here too
```

---

# ✅ 2 — Log the three most important things (params, metrics, artifacts)

1. Parameters (hyperparameters you used)
2. Metrics (accuracy, loss, RMSE, etc.)
3. Artifacts (plots, models, files)

These three give you 80% of MLflow’s value.

Example showing all three:

```python
import mlflow
import json

mlflow.set_experiment("params_metrics_artifacts")

with mlflow.start_run():
    # Parameters (hyperparams)
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_param("n_estimators", 100)

    # Metrics
    mlflow.log_metric("train_loss", 0.15)
    mlflow.log_metric("val_accuracy", 0.81)

    # Artifacts: save a small file and log it
    report = {"note": "first run"}
    with open("report.json", "w") as f:
        json.dump(report, f)
    mlflow.log_artifact("report.json")   # uploaded to run's artifact folder
```

---

# ✅ 3 — What is an Experiment? (create / switch experiments)

An experiment is a folder/group that contains many related runs.
It helps you stay organized (e.g., “baseline-models”, “xgboost-tests”).

Create or switch experiments programmatically:

```python
import mlflow

# creates if not exists, or switches to it if exists
mlflow.set_experiment("customer_churn_experiment")

with mlflow.start_run():
    mlflow.log_param("setup", "v1")
    mlflow.log_metric("accuracy", 0.78)
```

---

# ✅ 4 — What is a Run? (one training attempt)

A run is one execution of training.
Every run stores: parameters → metrics → artifacts.

Each `start_run()` is a run — example training attempt skeleton:

```python
import mlflow

with mlflow.start_run(run_name="attempt_1"):
    # everything between start and end is recorded as one "run"
    mlflow.log_param("seed", 42)
    mlflow.log_metric("some_metric", 0.5)
    # run ends automatically when exiting context
```

---

# ✅ 5 — Where artifacts are stored locally (`./mlruns/`)

In a folder called `./mlruns/` inside your project.

No code needed to create the folder — mlflow creates it. Example listing artifact path for a run:

```python
import mlflow
from pathlib import Path

mlflow.set_experiment("artifacts_demo")
with mlflow.start_run() as run:
    mlflow.log_artifact("report.json")
    run_id = run.info.run_id
    artifact_uri = mlflow.get_artifact_uri()  # e.g., file:///.../mlruns/0/<run_id>/artifacts
    print("Run ID:", run_id)
    print("Artifact URI:", artifact_uri)
    # Locally you'll see artifacts under ./mlruns/ (unless tracking URI changed)
    local_path = Path("mlruns")
    print("Check local folder:", local_path.resolve())
```

---

# ✅ 6 — Why MLflow vs Excel/Notion (practical)

Because MLflow:

 logs automatically
 stores files
 keeps models
 compares runs
 ensures reproducibility
  Excel cannot do any of that reliably.


Simple example showing repeatability — log random seed and params so later you can reproduce:

```python
import mlflow
import random

mlflow.set_experiment("repro_demo")
with mlflow.start_run():
    mlflow.log_param("seed", 123)
    random.seed(123)
    value = random.random()
    mlflow.log_metric("random_value", value)
# Later you can read the run metadata to reproduce using the same 'seed'
```

---

# ✅ 7 — Logging a model vs artifact

Model = a deployable ML model (like a pickle or MLflow model).
Artifact = any other file (plots, data, logs).

A model is an artifact, but it’s treated specially for loading/serving.

Model logging (specialized) vs generic artifact:

```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.set_experiment("model_vs_artifact")
with mlflow.start_run():
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(Xtr, ytr)
    score = clf.score(Xte, yte)
    mlflow.log_metric("test_accuracy", score)

    # Log model (special: supports pyfunc / loading via mlflow)
    mlflow.sklearn.log_model(clf, artifact_path="model")

    # Log some other file as an artifact (generic)
    with open("notes.txt", "w") as f:
        f.write("This is a training run note.")
    mlflow.log_artifact("notes.txt")
```

To load the saved model from a run (by run id):
```python
# mlflow.sklearn.load_model accepts a runs:/ URI
loaded = mlflow.sklearn.load_model("runs:/<RUN_ID>/model")
preds = loaded.predict(Xte)
```

(Replace `<RUN_ID>` with the printed run id or via the UI.)

---

# ✅ 8 — What `mlflow.autolog()` does (sklearn example)

It automatically logs parameters, metrics, and the trained model without you writing manual logging code.

Autologging automatically captures params, metrics, and model for many libraries.

```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

mlflow.set_experiment("autolog_demo")
mlflow.sklearn.autolog()   # enable autologging for sklearn

X, y = load_iris(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run():
    clf = RandomForestClassifier(n_estimators=20, random_state=42)
    clf.fit(Xtr, ytr)   # params, model, and some metrics are logged automatically
    # No explicit mlflow.log_param/log_metric required here
```

**When not to use autolog**: if you need fine control over what/when to log (large artifacts you don’t want), or if autolog adds too much noisy data.

---

# ✅ 9 — MLflow UI (how to run / view)

The MLflow UI is a website (localhost:5000) that lets you view:

your runs
metrics
params
artifacts
models

It makes experiment comparison easy.


Run in terminal:

```bash
# start the MLflow web UI (default port 5000)
mlflow ui
```

Then open `http://localhost:5000` in your browser and explore experiments, runs, params, metrics, and artifacts.

---

# ✅ 10 — Typical MLflow workflow (put together in one example)

1. Start run
2. Train model
3. Log results
4. End run

This is the core loop everyone uses.

A small complete script that follows: start run → train → log results → end run.

```python
# file: simple_train.py
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

mlflow.set_experiment("simple_workflow")

with mlflow.start_run(run_name="rf_trial"):
    mlflow.log_param("seed", 42)
    mlflow.log_param("n_estimators", 50)

    X, y = load_iris(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(Xtr, ytr)
    acc = clf.score(Xte, yte)

    mlflow.log_metric("accuracy", acc)

    # save model
    mlflow.sklearn.log_model(clf, artifact_path="model")

# run ends when context exits
print("Done. Open mlflow UI to inspect results.")
```

Run it:
```bash
python simple_train.py
mlflow ui
# open http://localhost:5000
```


# 🌟 Your 80/20 Roadmap for MLflow (as your mentor)

The entire MLflow ecosystem has 4 components, but **the 20% that gives you 80% results** is:

1. **Tracking** (you already got this)
2. **Models** (saving + loading + packaging)
3. **Model Registry** (real-world workflows)
4. **Serving** (local → REST API → production-like)

We’ll go through them in that exact order.

---

# ✅ Part 1 — MLflow Models (80/20 understanding)

MLflow Models = standardized way to save/load models so they can be deployed anywhere.

### 🧠 Mental model

Think of MLflow Models like **Docker for ML models**:
“Package once → run anywhere”.

---

## 📌 How to Save a Model

You already logged sklearn models. Now let’s do it **barebones minimal**:

```python
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)

model = LogisticRegression(max_iter=200)
model.fit(X, y)

# Save to local folder (models folder created automatically)
mlflow.sklearn.save_model(model, "my_lr_model")
```

This creates:

```
my_lr_model/
    MLmodel 
    model.pkl
    conda.yaml
```

This folder is a **portable model package**.

---

## 📌 How to Load the Model Back

```python
import mlflow.sklearn

loaded = mlflow.sklearn.load_model("my_lr_model")
print(loaded.predict([[5.1, 3.5, 1.4, 0.2]]))
```

---

## 🧪 80/20 takeaway

**If you know how to save & load models → you’ve unlocked all of MLflow serving & registry.**

---

# ✅ Part 2 — MLflow Model Registry (80/20 version)

## 🧠 Mental model

Model Registry is like **Git + Versions + Status** for models.

Models can be in states:

* **None (just logged)**
* **Staging** (ready to test)
* **Production** (serving users)
* **Archived**

---

## 📌 Register a Model

After logging a model inside a run:

```python
import mlflow

with mlflow.start_run() as run:
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name="IrisClassifier"
    )
```

This:

✔️ Logs the model
✔️ Creates a model registry entry called **IrisClassifier**
✔️ Creates version **1**

---

## 📌 Move model to **Staging** or **Production**

```python
from mlflow import MlflowClient

client = MlflowClient()
client.transition_model_version_stage(
    name="IrisClassifier",
    version=1,
    stage="Production"
)
```

Common stages: `"Staging"`, `"Production"`, `"Archived"`.

---

## 📌 Load a Model from Registry (Production version)

```python
prod_model = mlflow.pyfunc.load_model("models:/IrisClassifier/Production")
pred = prod_model.predict([[5.1, 3.5, 1.4, 0.2]])
print(pred)
```

This is huge.
Production pipelines can now *always load the correct model* automatically.

---

## 🧪 80/20 takeaway

There are only 3 things you do with the registry:

1. Register a model
2. Promote/demote versions
3. Load a model by stage (“give me latest Production model”)

That’s 80% of real-world MLOps.

---

# ✅ Part 3 — MLflow Model Serving (zero-to-hero in 2 steps)

## 🧠 Mental model

Think of MLflow Serving as **FastAPI already done for you**.

You don’t need to build REST servers manually.

---

## 📌 Serve a local logged model

```bash
mlflow models serve -m my_lr_model -p 5001
```

Now send data:

```bash
curl -X POST http://127.0.0.1:5001/invocations \
    -H "Content-Type: application/json" \
    -d '{"inputs": [[5.1, 3.5, 1.4, 0.2]]}'
```

---

## 📌 Serve a **Production Registry model**

```bash
mlflow models serve -m "models:/IrisClassifier/Production" -p 5002
```

This is real MLOps infrastructure on your laptop.

---

# ✅ Part 4 — MLflow in a Real Project (80/20 template)

Here’s the **minimal project structure** you’ll see in companies:

```
project/
├── train.py         # training + logging
├── predict.py       # loads Production model from registry
├── requirements.txt
├── mlruns/          # auto-created
```

---

### train.py

```python
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

mlflow.set_experiment("iris_project")

with mlflow.start_run():
    X, y = load_iris(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2)

    model = RandomForestClassifier(n_estimators=50)
    model.fit(Xtr, ytr)
    acc = model.score(Xte, yte)

    mlflow.log_param("n_estimators", 50)
    mlflow.log_metric("accuracy", acc)

    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        registered_model_name="IrisClassifier"
    )
```

---

### predict.py

```python
import mlflow.pyfunc

model = mlflow.pyfunc.load_model("models:/IrisClassifier/Production")
print(model.predict([[5.1, 3.5, 1.4, 0.2]]))
```

This is **exactly** how teams do MLOps with MLflow.

---

# 🔥 What you’ve unlocked so far

You now understand the **80/20 of MLflow**:

| Topic    | 80/20 Skill                        | Why it matters              |
| -------- | ---------------------------------- | --------------------------- |
| Tracking | params + metrics + artifacts       | Experimentation mastery     |
| Models   | save + load                        | Deploy anywhere & reproduce |
| Registry | register + promote + load by stage | Real-world MLOps pipelines  |
| Serving  | serve models + REST API            | Deployment / inference      |

You’re now 70% of the way to **full MLflow proficiency**.


