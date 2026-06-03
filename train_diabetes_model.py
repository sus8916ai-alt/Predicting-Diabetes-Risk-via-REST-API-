import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
data = pd.read_csv("diabetes.csv")

# Features and Target
X = data[["BMI", "Age", "BloodPressure", "Glucose"]]
y = data["Outcome"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train model
model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy:.2%}")

# Save model
joblib.dump(model, "diabetes_model.pkl")

print("Model saved as diabetes_model.pkl")

# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# import joblib

# # Load dataset (replace with your dataset path)
# data = pd.read_csv("diabetes.csv")

# # Features and target
# X = data[["BMI", "Age", "BloodPressure", "Glucose"]]
# y = data["Outcome"]  # 1 = diabetes, 0 = no diabetes

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train model
# model = LogisticRegression()
# model.fit(X_train, y_train)

# # Save model
# joblib.dump(model, "diabetes_model.pkl")

