

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ============================================================
# STEP 1: CREATE OUTPUT FOLDER
# ============================================================

os.makedirs("output", exist_ok=True)

# ============================================================
# STEP 2: LOAD DATASET
# ============================================================

data = load_breast_cancer()

X = pd.DataFrame(
    data.data,
    columns=data.feature_names
)

y = pd.Series(
    data.target,
    name="target"
)

print("Dataset loaded successfully!")
print("Dataset shape:", X.shape)

# Save dataset as CSV
dataset_df = X.copy()
dataset_df["target"] = y

dataset_df.to_csv(
    "output/breast_cancer_dataset.csv",
    index=False
)

# ============================================================
# STEP 3: SPLIT THE DATASET
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# ============================================================
# STEP 4: FEATURE SCALING
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# STEP 5: DEFINE MACHINE LEARNING MODELS
# ============================================================

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=5000,
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),

    "Support Vector Machine": SVC(
        kernel="rbf",
        random_state=42
    )
}

# ============================================================
# STEP 6: TRAIN AND EVALUATE MODELS
# ============================================================

results = []

for name, model in models.items():

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    # Train model
    model.fit(
        X_train_scaled,
        y_train
    )

    # Make predictions
    y_pred = model.predict(
        X_test_scaled
    )

    # Calculate metrics
    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred
    )

    recall = recall_score(
        y_test,
        y_pred
    )

    f1 = f1_score(
        y_test,
        y_pred
    )

    # Store results
    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1
    })

    # Print classification report
    print(
        classification_report(
            y_test,
            y_pred
        )
    )

    # ========================================================
    # CREATE CONFUSION MATRIX
    # ========================================================

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    plt.figure(figsize=(7, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["malignant", "benign"],
        yticklabels=["malignant", "benign"]
    )

    plt.title(
        f"Confusion Matrix - {name}"
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")

    # Create filename
    filename = (
        name.lower()
        .replace(" ", "_")
        + "_confusion_matrix.png"
    )

    plt.savefig(
        os.path.join(
            "output",
            filename
        ),
        bbox_inches="tight",
        dpi=150
    )

    plt.close()

# ============================================================
# STEP 7: CREATE RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(results)

# Save results
results_df.to_csv(
    "output/model_comparison.csv",
    index=False
)

# Display percentage results
display_results = results_df.copy()

for column in [
    "Accuracy",
    "Precision",
    "Recall",
    "F1-Score"
]:
    display_results[column] = (
        display_results[column] * 100
    )

print("\nMODEL COMPARISON RESULTS")
print(
    display_results.round(2).to_string(
        index=False
    )
)

# ============================================================
# STEP 8: SELECT AND SAVE BEST MODEL
# ============================================================

best_model_name = results_df.loc[
    results_df["Accuracy"].idxmax(),
    "Model"
]

best_model = models[
    best_model_name
]

# Save best model
joblib.dump(
    best_model,
    "output/best_model.pkl"
)

# Save scaler
joblib.dump(
    scaler,
    "output/scaler.pkl"
)

print("\nBEST MODEL:", best_model_name)
print(
    "Best model and scaler saved successfully!"
)

# ============================================================
# STEP 9: CREATE PERFORMANCE COMPARISON GRAPH
# ============================================================

plot_data = results_df.melt(
    id_vars="Model",
    var_name="Metric",
    value_name="Score"
)

plt.figure(figsize=(10, 6))

sns.barplot(
    data=plot_data,
    x="Model",
    y="Score",
    hue="Metric"
)

plt.title(
    "Performance Comparison of Machine Learning Models"
)

plt.xlabel(
    "Machine Learning Model"
)

plt.ylabel(
    "Score"
)

plt.ylim(
    0.80,
    1.05
)

plt.xticks(
    rotation=0
)

plt.tight_layout()

plt.savefig(
    "output/model_performance_comparison.png",
    bbox_inches="tight",
    dpi=150
)

plt.close()

print("\nProject completed successfully!")