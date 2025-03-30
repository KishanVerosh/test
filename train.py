import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

# Load dataset
file_path = "Loan.csv"
df = pd.read_csv(file_path)

# Drop Loan_ID (not useful for prediction)
df.drop(columns=['Loan_ID'], inplace=True)

# Convert numerical columns
numerical_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']
df[numerical_cols] = df[numerical_cols].apply(pd.to_numeric, errors='coerce')

# Fill missing values
for column in df.columns:
    if df[column].dtype == 'object':  # Categorical
        df[column].fillna(df[column].mode()[0], inplace=True)
    else:  # Numerical
        df[column].fillna(df[column].median(), inplace=True)

# Encode categorical variables
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le  # Store encoders for GUI use

# Save label encoders
joblib.dump(label_encoders, "label_encoders.pkl")

# Define features and target
X = df.drop(columns=['Loan_Status'])
y = df['Loan_Status']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train AdaBoost model
base_model = DecisionTreeClassifier(max_depth=1)
adaboost = AdaBoostClassifier(base_estimator=base_model, n_estimators=50, learning_rate=1.0, random_state=42)
adaboost.fit(X_train, y_train)

# Save trained model
joblib.dump(adaboost, "loan_model.pkl")

print("Model training complete. Model saved as loan_model.pkl.")
