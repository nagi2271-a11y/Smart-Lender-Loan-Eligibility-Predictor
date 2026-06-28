import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# STEP 2: Dataset Collection & Understanding
# ==========================================

# 1. Load the dataset
# Replace 'loan_data.csv' with your actual filename if it's different
df = pd.read_excel('dataset/vs sheet.csv')

# 2. Check basic data structure
print("--- Dataset Info ---")
print(df.info())
print(df.columns)

print("\n--- First 5 Rows ---")
print(df.head())

print("\n--- Missing Values Count ---")
print(df.isnull().sum())


# ==========================================
# STEP 3: Data Visualization & Analysis (EDA)
# ==========================================
sns.set_theme(style="darkgrid")

# 1. Count Plot: Check the distribution of Target Variable (e.g., Loan_Status)
plt.figure(figsize=(6,4))
# Note: Replace 'Loan_Status' with the exact column name in your dataset
sns.countplot(data=df, x='Loan_Status') 
plt.title('Distribution of Loan Approvals')
plt.show()

# 2. Distribution Plot: Check Applicant Income
plt.figure(figsize=(6,4))
sns.histplot(data=df, x='Applicant_Income', kde=True, bins=30)
plt.title('Applicant Income Distribution')
plt.show()

# 3. Bar Chart: Education vs Loan Amount (or another relationship)
plt.figure(figsize=(6,4))
sns.barplot(data=df, x='Education', y='Loan_Amount', ci=None)
plt.title('Average Loan Amount by Education Level')
plt.show()
# ==========================================
# STEP 4: Data Preprocessing & Feature Engineering
# ==========================================
print("\n--- Encoding Categorical Variables ---")

# We drop Customer_ID since it's just a random label and doesn't help predict loans
if 'Customer_ID' in df.columns:
    df = df.drop(columns=['Customer_ID'])

# Automatically convert all text columns (Gender, Married, Education, etc.) into numbers
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

# Convert any text columns to string format first, then encode them into numbers
for col in df.columns:
    if df[col].dtype == 'object' or df[col].dtype.name == 'category' or df[col].apply(lambda x: isinstance(x, str)).any():
        df[col] = le.fit_transform(df[col].astype(str))

print("Data after encoding:")
print(df.head())

# ==========================================
# STEP 5: Split Data into Features (X) and Target (y)
# ==========================================

# X contains all input features, y contains the target variable (Loan_Status)
X = df.drop(columns=['Loan_Status'])
y = df['Loan_Status']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining data size: {X_train.shape}")
print(f"Testing data size: {X_test.shape}")
# ==========================================
# STEP 5: Machine Learning Model Building
# ==========================================
print("\n--- Training Machine Learning Models ---")

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# 1. Initialize the models
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "XGBoost": XGBClassifier(random_state=42, eval_metric='logloss')
}

# 2. Train each model and evaluate accuracy
for name, model in models.items():
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate accuracy scores
    train_acc = accuracy_score(y_train, y_pred_train) * 100
    test_acc = accuracy_score(y_test, y_pred_test) * 100
    
    print(f"\n[{name}]")
    print(f"Training Accuracy: {train_acc:.2f}%")
    print(f"Testing Accuracy: {test_acc:.2f}%")
    # ==========================================
# STEP 6: Model Saving & Deployment Preparation
# ==========================================
print("\n--- Saving the Best Model ---")

import pickle

# 1. Find the model name with the highest testing accuracy
best_model_name = max(models, key=lambda name: accuracy_score(y_test, models[name].predict(X_test)))
best_model = models[best_model_name]

print(f"The best model is: {best_model_name}")

# 2. Save the trained model to a file named 'model.pkl'
with open('model.pkl', 'wb') as model_file:
    pickle.dump(best_model, model_file)

print("Successfully saved 'model.pkl'!")

# 3. Save our LabelEncoder tool so the web app can use it later
with open('scaler.pkl', 'wb') as le_file:
    pickle.dump(le, le_file)
    
print("Successfully saved 'scaler.pkl'!")
