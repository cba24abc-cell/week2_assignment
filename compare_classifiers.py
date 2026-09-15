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

# Load Setting A features
with open("features_A.pkl", "rb") as f:
    X, y = pickle.load(f)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Models required by assignment
models = {
    "Linear SVM": LinearSVC(max_iter=10000),
    "k-NN (k=5)": KNeighborsClassifier(n_neighbors=5)
}

results = []

for name, model in models.items():

    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)

    # Train
    model.fit(X_train, y_train)

    # Predict
    predictions = model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, predictions)

    print(f"\nAccuracy: {accuracy:.4f}")

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    # Confusion Matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    results.append([name, accuracy])

# Summary Table
summary = pd.DataFrame(
    results,
    columns=["Classifier", "Accuracy"]
)

print("\n" + "=" * 50)
print("ACCURACY COMPARISON")
print("=" * 50)
print(summary)
