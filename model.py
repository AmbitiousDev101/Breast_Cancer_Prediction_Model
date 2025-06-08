import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

# Load CSVs
train = pd.read_csv("breast_cancer_train_real.csv")
test = pd.read_csv("breast_cancer_test_real.csv")
predict = pd.read_csv("breast_cancer_predict_real.csv")

# Drop ID column
train.drop(columns=["id"], inplace=True)
test.drop(columns=["id"], inplace=True)
predict.drop(columns=["id"], inplace=True)

# Encode labels: B = 0 (Benign), M = 1 (Malignant)
train["diagnosis"] = train["diagnosis"].map({"B": 0, "M": 1})
test["diagnosis"] = test["diagnosis"].map({"B": 0, "M": 1})

# Split features and labels
X_train = train.drop(columns=["diagnosis"])
y_train = train["diagnosis"]

X_test = test.drop(columns=["diagnosis"])
y_test = test["diagnosis"]

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_predict_scaled = scaler.transform(predict)

# Train classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate model
y_pred = model.predict(X_test_scaled)
print("Model Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=["Benign", "Malignant"]))

# Predict on unlabeled data
predictions = model.predict(X_predict_scaled)
print("\nPredictions on new (unlabeled) data:")
for i, pred in enumerate(predictions, 1):
    print(f"Sample {i}: {'Malignant' if pred == 1 else 'Benign'}")

