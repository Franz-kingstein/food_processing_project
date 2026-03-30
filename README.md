# Hybrid ANN + GA Model for Butterfly Pea Extraction Experiment

This project trains an Artificial Neural Network (ANN) whose weights are optimized by a Genetic Algorithm (GA) to predict quality-related output variables from a Butterfly Pea flower extraction experiment.

## Overview

Butterfly Pea (*Clitoria ternatea*) flowers are rich in bioactive compounds. This model takes two process parameters as input and predicts four quality indicators after pasteurization.

| | Variable | Unit |
|---|---|---|
| **Input 1** | Weight of Butterfly Pea Flower | g / 200 mL |
| **Input 2** | Pasteurization Time (@ 80 °C) | minutes |
| **Output 1** | Phenolic Content | μg GAE / 100 mL |
| **Output 2** | Anthocyanin Content | mg / 100 mL |
| **Output 3** | Antioxidant Activity | % |
| **Output 4** | Color Index | — |

## Repository Structure

```
food_processing_project/
├── ANN_GA_ButterflyPea_Model.ipynb   # Main notebook
├── Butterfly_Pea_Experiment.csv       # Experimental dataset (required)
├── butterfly_pea_model.keras          # Saved Keras model (generated)
├── butterfly_pea_model.h5             # Saved Keras model – legacy format (generated)
├── x_scaler.save                      # Saved input scaler (generated)
└── y_scaler.save                      # Saved output scaler (generated)
```

## Methodology

### 1. Data Preparation
- Reads experimental data from `Butterfly_Pea_Experiment.csv`.
- Normalizes both inputs and outputs with `MinMaxScaler`.
- Splits the dataset into a training set (90 %) and a test set (10 %).

### 2. ANN Architecture
A fully-connected feed-forward network built with TensorFlow / Keras:

| Layer | Neurons | Activation |
|---|---|---|
| Input | 2 | — |
| Hidden 1 | 16 | ReLU |
| Hidden 2 | 12 | ReLU |
| Hidden 3 | 8 | ReLU |
| Output | 4 | Linear |

### 3. Genetic Algorithm (GA) Optimization
Instead of gradient-based backpropagation, the network weights are optimized using the DEAP library:

| Hyperparameter | Value |
|---|---|
| Population size | 50 |
| Generations | 40 |
| Crossover probability | 0.7 |
| Mutation probability | 0.2 |
| Selection | Tournament (size = 3) |
| Mutation operator | Gaussian (μ = 0, σ = 0.5) |

The GA minimizes Mean Squared Error (MSE) on the training set.

### 4. Evaluation & Interactive Prediction
After training, the best individual from the final GA population is applied to the ANN. The notebook then provides an interactive prompt where you can supply new process parameter values and receive predictions for all four quality indicators.

## Requirements

```
numpy
pandas
scikit-learn
tensorflow
deap
joblib
```

Install all dependencies with:

```bash
pip install numpy pandas scikit-learn tensorflow deap joblib
```

> A virtual environment named `food_pro_env` (Python 3.11) was used during development.

## Usage

1. Place `Butterfly_Pea_Experiment.csv` in the project root.
2. Open `ANN_GA_ButterflyPea_Model.ipynb` in Jupyter Notebook or JupyterLab.
3. Run all cells in order.
4. The trained model and scalers will be saved automatically:
   - `butterfly_pea_model.keras`
   - `butterfly_pea_model.h5`
   - `x_scaler.save`
   - `y_scaler.save`
5. Use the interactive cell at the end to enter new experimental values and obtain predictions.

## Example Prediction

```
🔍 Enter the experimental values to get predictions:
Enter Weight of Butterfly Pea Flower (g/200mL): 0.351
Enter Pasteurization Time (mins @ 80°C): 15

📊 Predicted Output Values:
Phenolic Content (μg GAE/100mL): 0.4058
Anthocyanin Content (mg/100mL): 6.9224
Antioxidant Activity (%): 6.3836
Color Index: 10.5099
```

## Notes

- The dataset file (`Butterfly_Pea_Experiment.csv`) is not included in this repository and must be provided separately.
- GPU acceleration is not required; TensorFlow runs on CPU for this workload.
