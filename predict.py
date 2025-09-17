# predict.py
import pickle
import numpy as np

with open("liver_model.pkl", "rb") as f:
    saved = pickle.load(f)

model = saved["model"]
scaler = saved["scaler"]
imputer = saved["imputer"]
threshold = saved["threshold"]
feature_columns = saved["feature_columns"]

print("🔹 Enter patient details:")
values = []
for col in feature_columns:
    val = input(f"{col}: ")
    try:
        val = float(val)
    except:
        val = np.nan
    values.append(val)

X_new = np.array(values).reshape(1, -1)

# Preprocess
X_new = imputer.transform(X_new)
X_new = scaler.transform(X_new)

# Predict
proba = model.predict_proba(X_new)[0, 1]
pred = int(proba >= threshold)
confidence = proba * 100 if pred == 1 else (1 - proba) * 100

print(f"\n✅ Prediction: {'Positive (Disease)' if pred==1 else 'Negative (Healthy)'}")
print(f"🔎 Confidence: {confidence:.2f}% (Threshold={threshold:.2f})")

