import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib

# ==========================
# Load dataset
# ==========================
df = pd.read_csv("upload+new.csv")

# ✅ Features used
features = [
    "Age",
    "Gender",  # will encode
    "Total_Bilirubin",
    "Direct_Bilirubin",
    "Alkaline_Phosphotase",
    "Alamine_Aminotransferase",
    "Aspartate_Aminotransferase",
    "Total_Protiens",
    "Albumin",
    "Albumin_and_Globulin_Ratio"
]

X = df[features].copy()

# ✅ Encode Gender (support both Male/Female and numeric)
if X["Gender"].dtype == "object":
    X["Gender"] = X["Gender"].map({"Male": 1, "Female": 0})

# ==========================
# Preprocessing
# ==========================
imputer = SimpleImputer(strategy="mean")
X_imputed = imputer.fit_transform(X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# ==========================
# Clustering
# ==========================
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

print("✅ Forced number of clusters: 5\n")
print("📊 Cluster Means (possible disease patterns):")
print(pd.DataFrame(X_imputed, columns=features).groupby(df["Cluster"]).mean(), "\n")

# ==========================
# Disease + Cancer Mapping
# ==========================
disease_mapping = {
    0: "Hepatitis",
    1: "No Liver Disease",
    2: "Fatty Liver",
    3: "Fibrosis",
    4: "Cirrhosis"
}

# ✅ Only Cancer / No Cancer
cancer_mapping = {
    0: "Cancer",
    1: "No Cancer",
    2: "Cancer",
    3: "Cancer",
    4: "Cancer"
}

print("🩺 Cluster → Disease + Cancer Mapping:")
for cluster in range(5):
    print(f"Cluster {cluster} → {disease_mapping[cluster]} | {cancer_mapping[cluster]}")

# ==========================
# Save model
# ==========================
joblib.dump(
    {
        "model": kmeans,
        "scaler": scaler,
        "imputer": imputer,
        "features": features,
        "disease_mapping": disease_mapping,
        "cancer_mapping": cancer_mapping
    },
    "liver_cluster_model.pkl",
    compress=3
)

print("\n💾 Model saved as liver_cluster_model.pkl")
import pandas as pd
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans

# ---------------------------
# Load dataset
# ---------------------------


# Encode Gender (Female=0, Male=1)
le = LabelEncoder()
df["Gender"] = le.fit_transform(df["Gender"])

# Features for clustering
features = [
    "Age", "Gender", "Total_Bilirubin", "Direct_Bilirubin",
    "Alkaline_Phosphotase", "Alamine_Aminotransferase",
    "Aspartate_Aminotransferase", "Total_Protiens",
    "Albumin", "Albumin_and_Globulin_Ratio"
]

X = df[features]

# ---------------------------
# Preprocessing
# ---------------------------
imputer = SimpleImputer(strategy="mean")
X_imputed = imputer.fit_transform(X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# ---------------------------
# KMeans Clustering
# ---------------------------
n_clusters = 5
cluster_model = KMeans(n_clusters=n_clusters, random_state=42)
df["Cluster"] = cluster_model.fit_predict(X_scaled)

print(f"✅ Forced number of clusters: {n_clusters}\n")

# Cluster means
cluster_means = df.groupby("Cluster")[features].mean()
print("📊 Cluster Means (possible disease patterns):")
print(cluster_means, "\n")

# Map cluster → disease
disease_map = {
    0: "Hepatitis",
    1: "No Liver Disease",
    2: "Fatty Liver",
    3: "Fibrosis",
    4: "Cirrhosis"
}

print("🩺 Cluster → Disease Type Mapping:")
for cl, dis in disease_map.items():
    print(f"Cluster {cl} → {dis}")

# ---------------------------
# Save everything
# ---------------------------
joblib.dump({
    "cluster_model": cluster_model,
    "imputer": imputer,
    "scaler": scaler,
    "disease_map": disease_map
}, "liver_cluster_model.pkl")

print("\n💾 Model + Preprocessors saved as liver_cluster_model.pkl")

