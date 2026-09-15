import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ==========================
# Load Setting A Features
# ==========================

with open("features_A.pkl", "rb") as f:
    X, y = pickle.load(f)

print("Total samples:", len(X))
print("Classes:", set(y))

# ==========================
# 80/20 Train-Test Split
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================
# Models
# ==========================

models = {
    "Linear SVM": LinearSVC(max_iter=10000),
    "k-NN (k=5)": KNeighborsClassifier(n_neighbors=5)
}

results = []

# ==========================
# Training & Evaluation
# ==========================

for name, model in models.items():

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    accuracy = accuracy_score(y_test, preds)

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print(f"\nAccuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, preds))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, preds))

    results.append([name, accuracy])

# ==========================
# Summary Table
# ==========================

summary = pd.DataFrame(
    results,
    columns=["Classifier", "Accuracy"]
)

print("\n" + "=" * 60)
print("SUMMARY TABLE")
print("=" * 60)
print(summary)

# Optional: save results
summary.to_csv("classifier_comparison.csv", index=False)

print("\nResults saved to classifier_comparison.csv")
