# Mushroom Classification (Edible vs Poisonous)

Machine learning project to classify mushrooms as edible or poisonous
using three algorithms: Naive Bayes, Decision Tree and Random Forest.

## Dataset
UCI Machine Learning Repository - Mushroom Data Set
(https://archive.ics.uci.edu/dataset/73/mushroom)

- 8,124 samples, 22 categorical features
- Target: edible (e) / poisonous (p)
- Missing values: 2,480 "?" entries, all in `stalk-root`
- `veil-type` has only one value, so it is dropped

## How to run
    pip install -r requirements.txt
    python main.py

## Method
1. Load data and add column names
2. Basic EDA (class distribution, odor vs class)
3. Stratified 80/20 train-test split
4. Train 3 models, evaluate with 5-fold CV and test set
5. Compare models, feature importance, decision tree plot

## Results (test set)

| Model         | Accuracy | Precision | Recall | F1     |
|---------------|----------|-----------|--------|--------|
| Naive Bayes   | 0.9452   | 0.9901    | 0.8953 | 0.9403 |
| Decision Tree | 0.9988   | 1.0000    | 0.9974 | 0.9987 |
| Random Forest | 1.0000   | 1.0000    | 1.0000 | 1.0000 |

## Observations
- Tree-based models perform almost perfectly on this dataset.
- `odor`, `gill-size` and `spore-print-color` are the most important features.
- Naive Bayes is weaker because it assumes features are independent.

## Author
Israt Jahan Sumaiya
