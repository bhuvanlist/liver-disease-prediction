import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
import joblib

# =======================
# Load and preprocess data
# =======================
print("🔹 Loading dataset...")
df = pd.read_csv("upload.csv")

# Rename label column consistently
df.rename(columns={'Dataset': 'Outcome'}, inplace=True)

# Map outcomes: 1 = Positive, 2 = Negative
df['Outcome'] = df['Outcome'].map({1: 1, 2: 0})

# Encode gender
df['Gender'] = LabelEncoder().fit_transform(df['Gender'])

# Features and target
X = df.drop(columns=['Outcome', 'Id'], errors='ignore')
y = df['Outcome']

# 🔹 Handle missing values
imputer = SimpleImputer(strategy="median")
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Balance with SMOTE
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =======================
# Define models
# =======================
rf = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42)
xgb = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                    eval_metric='logloss', use_label_encoder=False, random_state=42)
lgbm = LGBMClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42)

# Voting ensemble
voting = VotingClassifier(
    estimators=[('rf', rf), ('xgb', xgb), ('lgbm', lgbm)],
    voting='soft'
)

# Stacking ensemble
stacking = StackingClassifier(
    estimators=[('rf', rf), ('xgb', xgb), ('lgbm', lgbm)],
    final_estimator=LogisticRegression(max_iter=1000),
    passthrough=True
)

# =======================
# Train and evaluate
# =======================
models = {"Voting": voting, "Stacking": stacking}
best_model = None
best_f1 = 0

for name, model in models.items():
    print(f"\n🔹 Training {name} Ensemble...")
    model.fit(X_train, y_train)

    # Get probabilities
    y_proba = model.predict_proba(X_test)[:, 1]

    # Find best threshold
    thresholds = np.arange(0.3, 0.71, 0.01)
    best_t, best_t_f1 = 0.5, 0
    for t in thresholds:
        y_pred_t = (y_proba >= t).astype(int)
        f1 = f1_score(y_test, y_pred_t)
        if f1 > best_t_f1:
            best_t, best_t_f1 = t, f1

    print(f"✅ {name} Best threshold: {best_t:.2f} | F1: {best_t_f1:.4f}")

    # Evaluate with best threshold
    y_pred = (y_proba >= best_t).astype(int)
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    if best_t_f1 > best_f1:
        best_model = (model, best_t, name)
        best_f1 = best_t_f1

# =======================
# Save best model
# =======================
final_model, final_thresh, final_name = best_model
joblib.dump({"model": final_model, "scaler": scaler, "threshold": final_thresh, "imputer": imputer},
            "liver_model.pkl")

print(f"\n🎉 Best Model: {final_name} (Threshold={final_thresh:.2f}, F1={best_f1:.4f}) saved as liver_model.pkl")

