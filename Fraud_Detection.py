import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("creditcard.csv")

print("Dataset shape:", data.shape)

# Separate normal and fraud transactions
normal = data[data["Class"] == 0]
fraud = data[data["Class"] == 1]

print("Normal transactions:", len(normal))
print("Fraud transactions:", len(fraud))

# Balance the dataset (undersampling)
normal_sample = normal.sample(n=len(fraud), random_state=42)
dataset = pd.concat([normal_sample, fraud], axis=0)

print("\nBalanced dataset:")
print(dataset["Class"].value_counts())

# Features and labels
X = dataset.drop("Class", axis=1)
y = dataset["Class"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

# Accuracy
train_accuracy = accuracy_score(y_train, train_pred)
test_accuracy = accuracy_score(y_test, test_pred)

print("\nModel Performance")
print("Training Accuracy:", round(train_accuracy, 4))
print("Testing Accuracy:", round(test_accuracy, 4))
