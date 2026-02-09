import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import HistGradientBoostingClassifier # Generally more accurate than RF
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os


print("--- 1. INITIALIZING PROJECT ---")

# --- A. IMPROVED DATA LOADING / GENERATION ---
filename = 'telecom_india.csv'

if os.path.exists(filename):
    print(f"Loading real data from {filename}...")
    df = pd.read_csv(filename)
else:
    print(f"Warning: '{filename}' not found. Generating INTELLIGENT synthetic data...")
    np.random.seed(42) 
    n_samples = 152000
    
    # Generate base features
    data = {
        'telecom_partner': np.random.choice(['Reliance Jio', 'Airtel', 'Vi', 'BSNL'], n_samples),
        'state': np.random.choice(['Maharashtra', 'Delhi', 'Karnataka', 'Tamil Nadu', 'UP'], n_samples),
        'age': np.random.randint(18, 75, n_samples),
        'gender': np.random.choice(['M', 'F'], n_samples),
        'data_used': np.random.randint(1, 250, n_samples),
        'calls_made': np.random.randint(10, 1500, n_samples),
        'sms_sent': np.random.randint(0, 300, n_samples),
    }
    df = pd.DataFrame(data)

    print("--------DATA----------\n",df)
    # --- LOGIC TO CREATE REALISTIC CHURN PATTERNS (CRITICAL FOR TRAINING) ---
    # Assume: 
    # 1. High Data usage -> Low Churn (Happy customer)
    # 2. Very High Calls -> Low Churn
    # 3. 'Vi' and 'BSNL' might have slightly higher base churn (hypothetically)
    
    # Calculate a "Churn Probability Score"
    score = np.zeros(n_samples)
    
    # Logic 1: High data users are less likely to churn
    score -= (df['data_used'] / 250) * 4  
    
    # Logic 2: Older people churn less (loyal)
    score -= (df['age'] / 75) * 2
    
    # Logic 3: Specific network bias (just for pattern creation)
    mask_risk = df['telecom_partner'].isin(['Vi', 'BSNL'])
    score[mask_risk] += 1.5
    
    # Add some random noise
    score += np.random.normal(0, 1.5, n_samples)
    
    # Convert score to 0 or 1 (Sigmoid-ish threshold)
    # This creates a class balance of roughly 20% churners (realistic)
    threshold = np.percentile(score, 80) # Top 20% riskiest scores churn
    df['churn'] = (score > threshold).astype(int)

    print(f"Generated {n_samples} rows with REALISTIC patterns.")
    print(f"Churn Rate: {df['churn'].mean():.2%}")

    print("--------DATA with Churn----------\n",df)
    df.to_csv('telecom_india.csv',index=False)



    
# --- B. FEATURE ENGINEERING ---
# Creating new columns that help the model differentiate users better
print("--- 2. FEATURE ENGINEERING ---")
#df['calls_per_data'] = df['calls_made'] / (df['data_used'] + 1) # Avoid div by 0
#df['total_activity'] = df['calls_made'] + df['sms_sent']
# Keep the new features in the list
target = 'churn'
numeric_features = ['age', 'data_used', 'calls_made', 'sms_sent']
#numeric_features = ['age', 'data_used', 'calls_made', 'sms_sent', 'calls_per_data', 'total_activity']
categorical_features = ['telecom_partner', 'state', 'gender']

X = df[numeric_features + categorical_features]
y = df[target]

print(X)
print("\n",y)


# --- C. BUILD ADVANCED PIPELINE ---
# 1. Preprocessor: Standardize numbers, OneHotEncode categories
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features), # Scaling helps Gradient Descent
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# 2. Model Selection: Histogram-based Gradient Boosting
# This is Sklearn's version of LightGBM - usually beats Random Forest on accuracy
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', HistGradientBoostingClassifier(
        max_iter=200,           # More trees (estimators)
        learning_rate=0.1,      # Step size
        max_depth=10,           # Depth of trees
        random_state=42
    ))
])


# --- D. TRAIN WITH EVALUATION ---
print("--- 3. TRAINING MODEL ---")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

# --- E. DETAILED METRICS ---
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# --- F. SAVE ---
joblib.dump(pipeline, 'india_churn_model.joblib')
print("--- 4. MODEL SAVED ---")


'''
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os

print("--- 1. INITIALIZING PROJECT ---")

# A. DATA LOADING / GENERATION
# If you have 'telecom_india.csv', it loads it. If not, it creates dummy data.
filename = 'telecom_india.csv'

if os.path.exists(filename):
    print(f"Loading real data from {filename}...")
    df = pd.read_csv(filename)
else:
    print(f"Warning: '{filename}' not found. Generating SYNTHETIC Indian data...")
    np.random.seed(42) # For reproducibility
    n_samples = 20000
    df = pd.DataFrame({
        'telecom_partner': np.random.choice(['Reliance Jio', 'Airtel', 'Vi', 'BSNL'], n_samples),
        'state': np.random.choice(['Maharashtra', 'Delhi', 'Karnataka', 'Tamil Nadu','Uttarakhand','Uttar Pradesh','Haryana','Punjab','Himachal Pradesh','Gujarat','Bihar','West Bengal'], n_samples),
        'age': np.random.randint(18, 75, n_samples),
        'gender': np.random.choice(['M', 'F'], n_samples),
        'data_used': np.random.randint(1, 250, n_samples),  # GBs per month
        'calls_made': np.random.randint(10, 1500, n_samples), # Minutes
        'sms_sent': np.random.randint(0, 300, n_samples),
        'churn': np.random.randint(0, 2, n_samples)
    })
    print(f"Generated {n_samples} rows of data.")
print(df.head(10))
df.to_csv("telecom_india.csv",index=False)
# B. DEFINE FEATURES
target = 'churn'
numeric_features = ['age', 'data_used', 'calls_made', 'sms_sent']
categorical_features = ['telecom_partner', 'state', 'gender']

X = df.drop(columns=[target])
y = df[target]

# C. BUILD THE PIPELINE
# 1. Preprocessor: Handle numeric and categorical data differently
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# 2. Pipeline: Combine Preprocessor + Random Forest Model
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# D. TRAIN THE MODEL
print("--- 2. TRAINING MODEL ---")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline.fit(X_train, y_train)
accuracy = pipeline.score(X_test, y_test)

print(f"Model successfully trained. Test Accuracy: {accuracy:.2f}")

# E. SAVE THE MODEL
joblib.dump(pipeline, 'india_churn_model.joblib')
print("--- 3. MODEL SAVED AS 'india_churn_model.joblib' ---")

'''