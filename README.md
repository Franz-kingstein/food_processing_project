# Butterfly Pea Flower Extraction — ANN + GA Prediction Model

A machine-learning model that predicts the quality of Butterfly Pea flower extracts from two simple process inputs. It combines an **Artificial Neural Network (ANN)** for prediction with a **Genetic Algorithm (GA)** to find the best network weights — no manual backpropagation tuning required.

---

## What Does This Model Do?

You give it two things you control in the lab:

| Input | Description | Unit |
|---|---|---|
| Flower weight | How much dried Butterfly Pea flower you use | g per 200 mL |
| Pasteurization time | How long the extract is heated at 80 °C | minutes |

It predicts four quality indicators of the resulting extract:

| Output | What it measures |
|---|---|
| **Phenolic Content** (μg GAE/100 mL) | Total phenolic compounds — linked to health benefits |
| **Anthocyanin Content** (mg/100 mL) | Blue/purple pigment concentration — indicator of extract potency |
| **Antioxidant Activity** (%) | Ability to neutralize free radicals |
| **Color Index** | Numerical measure of the extract's color intensity |

---

## How It Works (Step by Step)

```
Lab data (CSV)
     │
     ▼
1. Normalize inputs & outputs  ← MinMaxScaler scales all values to [0, 1]
     │
     ▼
2. Build ANN  ← 4-layer neural network (2 inputs → 16 → 12 → 8 → 4 outputs)
     │
     ▼
3. Genetic Algorithm  ← evolves 50 candidate weight sets over 40 generations
     │                   to minimize prediction error (MSE)
     ▼
4. Best weights applied to ANN
     │
     ▼
5. Predict on test set & interactive input
```

### ANN Architecture

| Layer | Neurons | Activation | Role |
|---|---|---|---|
| Input | 2 | — | Flower weight + pasteurization time |
| Hidden 1 | 16 | ReLU | Feature extraction |
| Hidden 2 | 12 | ReLU | Feature refinement |
| Hidden 3 | 8 | ReLU | Final abstraction |
| Output | 4 | Linear | Predicts all four quality values |

### Genetic Algorithm Settings

The GA replaces traditional gradient-descent training. It treats each possible set of ANN weights as an "individual" and evolves the population toward lower prediction error.

| Setting | Value | Meaning |
|---|---|---|
| Population size | 50 | 50 candidate weight sets evaluated per generation |
| Generations | 40 | Number of evolution cycles |
| Crossover probability | 0.7 | 70 % chance two individuals swap weight segments |
| Mutation probability | 0.2 | 20 % chance of random weight perturbation |
| Selection method | Tournament (size = 3) | Best of 3 random individuals advances |
| Mutation type | Gaussian (μ=0, σ=0.5) | Small random nudges to weight values |
| Fitness function | MSE | Lower = better predictions |

---

## Repository Structure

```
food_processing_project/
├── ANN_GA_ButterflyPea_Model.ipynb   # Main notebook — run this
├── Butterfly_Pea_Experiment.csv       # Your lab dataset (you must provide this)
├── butterfly_pea_model.keras          # Trained model, saved after running notebook
├── butterfly_pea_model.h5             # Same model in legacy HDF5 format
├── x_scaler.save                      # Input normalizer (saved with joblib)
└── y_scaler.save                      # Output normalizer (saved with joblib)
```

> **Note:** `Butterfly_Pea_Experiment.csv` is not included. You need to supply your own experimental data file. Column order expected: `[index, flower_weight, pasteurization_time, phenolic, anthocyanin, antioxidant, color_index]`.

---

## Requirements

Python **3.11** is recommended (used during development).

```bash
pip install numpy pandas scikit-learn tensorflow deap joblib
```

| Package | Purpose |
|---|---|
| `numpy` / `pandas` | Data handling |
| `scikit-learn` | Data normalization and train/test split |
| `tensorflow` | ANN definition and inference |
| `deap` | Genetic Algorithm framework |
| `joblib` | Saving/loading scalers |

> Tip: use a virtual environment — `python -m venv food_pro_env && source food_pro_env/bin/activate` — before installing.

---

## Quick Start

```bash
# 1. Clone the repo and enter the folder
git clone https://github.com/Franz-kingstein/food_processing_project.git
cd food_processing_project

# 2. Install dependencies
pip install numpy pandas scikit-learn tensorflow deap joblib

# 3. Add your dataset
cp /path/to/your/Butterfly_Pea_Experiment.csv .

# 4. Launch the notebook
jupyter notebook ANN_GA_ButterflyPea_Model.ipynb
```

Then **run all cells** in order (Cell → Run All). The GA training takes a few minutes depending on your hardware.

---

## Usage Details

1. **Load data** — the notebook reads `Butterfly_Pea_Experiment.csv` from the project root.
2. **Train** — the GA runs for 40 generations. Progress is printed to the cell output.
3. **Evaluate** — the best model is tested on the held-out 10 % test set; actual vs. predicted values are printed.
4. **Save** — the trained model and scalers are saved automatically so you can reload them later without retraining.
5. **Predict interactively** — the final cell prompts you to enter new values and instantly returns predictions.

---

## Example Prediction

```
🔍 Enter the experimental values to get predictions:
Enter Weight of Butterfly Pea Flower (g/200mL):  0.351
Enter Pasteurization Time (mins @ 80°C):         15

📊 Predicted Output Values:
  Phenolic Content (μg GAE/100mL):   0.4058
  Anthocyanin Content (mg/100mL):    6.9224
  Antioxidant Activity (%):          6.3836
  Color Index:                       10.5099
```

---

## Limitations & Notes

- **Dataset not included** — supply `Butterfly_Pea_Experiment.csv` with your own experimental measurements.
- **GA is stochastic** — results vary slightly between runs due to random initialization. Re-run a few times to check consistency.
- **No GPU needed** — TensorFlow runs on CPU for this model size; a typical laptop is sufficient.
- **Model accuracy depends on data quality** — more experimental data points generally lead to better predictions.
