"""
Mushroom Classification (Edible vs Poisonous)
Algorithms: Naive Bayes, Decision Tree, Random Forest
Dataset: UCI Mushroom (agaricus-lepiota.data)
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree

RANDOM_STATE = 42
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "agaricus-lepiota.data"
OUT_DIR = BASE_DIR / "results"
OUT_DIR.mkdir(exist_ok=True)

COLUMNS = [
    "class", "cap-shape", "cap-surface", "cap-color", "bruises", "odor",
    "gill-attachment", "gill-spacing", "gill-size", "gill-color",
    "stalk-shape", "stalk-root", "stalk-surface-above-ring",
    "stalk-surface-below-ring", "stalk-color-above-ring",
    "stalk-color-below-ring", "veil-type", "veil-color", "ring-number",
    "ring-type", "spore-print-color", "population", "habitat",
]


# ---------------------------------------------------------------- 1. Load
def load_data():
    df = pd.read_csv(DATA_PATH, header=None, names=COLUMNS)
    print("Shape:", df.shape)
    print("\nClass distribution:\n", df["class"].value_counts())
    # '?' only appears in stalk-root -> treat it as its own category
    print("\nMissing ('?') values per column:\n", (df == "?").sum()[lambda s: s > 0])
    return df


# ---------------------------------------------------------------- 2. EDA
def eda_plots(df):
    plt.figure(figsize=(5, 4))
    sns.countplot(x="class", data=df, hue="class", palette=["green", "red"], legend=False)
    plt.xticks([0, 1], ["Edible (e)", "Poisonous (p)"])
    plt.title("Class distribution")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "class_distribution.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 4))
    sns.countplot(x="odor", data=df, hue="class", palette=["green", "red"])
    plt.title("Odor vs Class")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "odor_vs_class.png", dpi=150)
    plt.close()


# ---------------------------------------------------------------- 3. Models
def build_models(feature_cols):
    # Naive Bayes -> integer-coded categories (CategoricalNB)
    nb = Pipeline([
        ("enc", OrdinalEncoder()),
        ("clf", CategoricalNB(min_categories=12, alpha=1.0)),
    ])

    onehot = ColumnTransformer(
        [("ohe", OneHotEncoder(handle_unknown="ignore"), feature_cols)]
    )
    dt = Pipeline([
        ("enc", onehot),
        ("clf", DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE)),
    ])
    rf = Pipeline([
        ("enc", onehot),
        ("clf", RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)),
    ])
    return {"Naive Bayes": nb, "Decision Tree": dt, "Random Forest": rf}


# ---------------------------------------------------------------- 4. Main
def main():
    df = load_data()
    eda_plots(df)

    # veil-type has a single value for every row -> no information
    df = df.drop(columns=["veil-type"])

    X = df.drop(columns="class")
    y = (df["class"] == "p").astype(int)  # 1 = poisonous, 0 = edible
    feature_cols = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"\nTrain: {X_train.shape}  Test: {X_test.shape}")

    models = build_models(feature_cols)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []

    for name, model in models.items():
        print("\n" + "=" * 60 + f"\n{name}\n" + "=" * 60)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        print(f"5-fold CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        print(classification_report(y_test, pred, target_names=["Edible", "Poisonous"], digits=4))

        rows.append({
            "Model": name,
            "CV Accuracy": cv_scores.mean(),
            "Test Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred),
            "Recall": recall_score(y_test, pred),
            "F1-score": f1_score(y_test, pred),
        })

        cm = confusion_matrix(y_test, pred)
        plt.figure(figsize=(4.5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Edible", "Poisonous"], yticklabels=["Edible", "Poisonous"])
        plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title(f"{name} - Confusion Matrix")
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"cm_{name.lower().replace(' ', '_')}.png", dpi=150)
        plt.close()

    # ---- comparison table + chart
    results = pd.DataFrame(rows).set_index("Model").round(4)
    print("\n" + "=" * 60 + "\nMODEL COMPARISON\n" + "=" * 60)
    print(results)
    results.to_csv(OUT_DIR / "model_comparison.csv")

    results[["Test Accuracy", "Precision", "Recall", "F1-score"]].plot(
        kind="bar", figsize=(8, 5), ylim=(0.9, 1.01))
    plt.title("Model comparison"); plt.xticks(rotation=0); plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "model_comparison.png", dpi=150)
    plt.close()

    # ---- feature importance (Random Forest, grouped back to original features)
    rf = models["Random Forest"]
    ohe_names = rf.named_steps["enc"].get_feature_names_out()
    imp = pd.Series(rf.named_steps["clf"].feature_importances_, index=ohe_names)
    imp.index = [n.split("__")[1].rsplit("_", 1)[0] for n in imp.index]
    imp = imp.groupby(level=0).sum().sort_values(ascending=False).head(10)
    print("\nTop 10 important features (Random Forest):\n", imp.round(4))

    plt.figure(figsize=(7, 5))
    sns.barplot(x=imp.values, y=imp.index, color="seagreen")
    plt.title("Random Forest - Top 10 Feature Importances")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "rf_feature_importance.png", dpi=150)
    plt.close()

    # ---- decision tree visualisation (top 3 levels)
    dt = models["Decision Tree"]
    plt.figure(figsize=(20, 9))
    plot_tree(dt.named_steps["clf"], max_depth=3,
              feature_names=dt.named_steps["enc"].get_feature_names_out(),
              class_names=["Edible", "Poisonous"], filled=True, fontsize=8)
    plt.title("Decision Tree (first 3 levels)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "decision_tree.png", dpi=120)
    plt.close()

    print(f"\nAll plots and tables saved in: {OUT_DIR}")


if __name__ == "__main__":
    main()