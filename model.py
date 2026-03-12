import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Load dataset
df = pd.read_csv("credit_card_fraud_2025.csv")

# Select important columns
df = df[['Amount','Is_International','Is_Chip','Is_Pin_Used',
         'Hour_of_Day','Device_Type','Fraud_Flag']]

# Convert device type to numbers
df['Device_Type'] = df['Device_Type'].map({
    "POS":0,
    "Mobile":1,
    "Web":2
})

# Features and target
X = df[['Amount','Is_International','Is_Chip','Is_Pin_Used','Hour_of_Day','Device_Type']]
y = df['Fraud_Flag']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Machine Learning Model (Random Forest)
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train model
model.fit(X_train, y_train)

# Test accuracy
y_pred = model.predict(X_test)

print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save trained model
pickle.dump(model, open("fraud_model.pkl","wb"))

print("\nRandom Forest ML Model trained and saved successfully!")