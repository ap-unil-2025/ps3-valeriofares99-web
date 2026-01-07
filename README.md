# Startup Success Prediction: Model Comparison

## Research Question
Which classification model performs best for predicting startup success (acquired/IPO vs closed/operating):
Logistic Regression, KNN, Naive Bayes, SVM, Random Forest, or XGBoost?

## Setup
### Create environment
conda env create -f environment.yml

conda activate project-vc

### Usage
python main.py

Expected output: Accuracy comparison between models with and without PCA, detailed classification report for each model and winner.

## Project Structure

```text
project/
├── main.py               # Main entry point
├── src/                  # Source code
│   ├── data_loader.py    # Data loading and preprocessing
│   ├── models.py         # Model definition and training
│   └── evaluation.py     # Evaluation metrics and analysis
├── data/
│   └── raw/              # Raw Crunchbase dataset
├── report/
│   └── figures/          # Generated plots and visualizations
├── notebooks/            # Exploratory and experimental notebooks
└── environment.yml       # Project dependencies
```

## Results
### Models Without PCA
- Random Forest: 0.879 accuracy
- XGBoost: 0.876 accuracy
- KNN: 0.863 accuracy
- Logistic Regression: 0.720 accuracy
- SVM: 0.711 accuracy
- Naive Bayes: 0.443 accuracy

### Models With PCA (65 components)
- Random Forest: 0.874 accuracy
- XGBoost: 0.869 accuracy
- KNN: 0.864 accuracy
- Naive Bayes: 0.797 accuracy
- Logistic Regression: 0.686 accuracy
- SVM: 0.682 accuracy

- **Winner**: Random Forest (0.879 accuracy)

## Requirements
- Python 3.11
- pandas, numpy, scikit-learn, matplotlib, seaborn, xgboost
