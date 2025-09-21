import pandas as pd
import pickle
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

print("cluster.py running...")

# Load dataset
df = pd.read_csv("upload+new.csv")
print(f"Raw shape: {df.shape}")

# 🔧 Standardize column names
df.rename(columns={"Total_Protiens": "Total_Proteins"}, inplace=True)

# Encode Gender
df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})

# Drop rows with missing Gender
df = df.dropna(subset=["Gender"])

# Select features
features = [
    "Age", "Gender", "Total_Bilirubin", "Direct_Bilirubin",
    "Alkaline_Phosphotase", "Alamine_Aminotransferase",
    "Aspartate_Aminotransferase", "Total_Proteins", "Albumin",
    "Albumin_and_Globulin_Ratio"
]
df = df[features]

# Handle missing values
imputer = SimpleImputer(strategy="mean")
X_imputed = imputer.fit_transform(df)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Train clustering model
kmeans = KMeans(n_clusters=5, random_state=42)
labels = kmeans.fit_predict(X_scaled)

# Cluster to disease mapping
cluster_mapping = {
    0: "Cirrhosis",
    1: "Fatty Liver",
    2: "Fibrosis",
    3: "Fibrosis",
    4: "Hepatitis"
}

print("\n🩺 Cluster → Disease Mapping:")
for cid, disease in cluster_mapping.items():
    print(f"Cluster {cid} -> {disease}")

# Save models
with open("liver_cluster_model.pkl", "wb") as f:
    pickle.dump({
        "imputer": imputer,
        "scaler": scaler,
        "cluster_model": kmeans,
        "disease_map": cluster_mapping
    }, f)


print("\n✅ Model saved as liver_cluster_model.pkl")

