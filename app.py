import streamlit as st
import numpy as np
import joblib

# -------------------------------
# Load Old Binary Model (compressed)
# -------------------------------
old_model_data = joblib.load("liver_model_compressed.pkl")
model = old_model_data["model"]
scaler = old_model_data["scaler"]
imputer = old_model_data["imputer"]
threshold = old_model_data["threshold"]

# -------------------------------
# Load Cluster Model (Disease Classification)
# -------------------------------
cluster_data = joblib.load("liver_cluster_model.pkl")
cluster_model = cluster_data["cluster_model"]
imputer_cluster = cluster_data["imputer"]
scaler_cluster = cluster_data["scaler"]
disease_map = cluster_data["disease_map"]

# Cancer mapping (only meaningful if disease exists)
cancer_map = {
    0: "Cancer",
    1: "No Cancer",
    2: "Cancer",
    3: "Cancer",
    4: "Cancer"
}

# -------------------------------
# Confidence Helper
# -------------------------------
def confidence_label(prob):
    if prob < 0.4:
        return "Very Low"
    elif prob < 0.6:
        return "Low"
    elif prob < 0.8:
        return "High"
    else:
        return "Very High"

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Liver & Cancer Prediction", layout="wide")

# -------------------------------
# Custom CSS
# -------------------------------
st.markdown(
    """
    <style>
    body {
        background-color: #f0f8ff;
    }
    .main {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 15px;
    }
    h1, h2, h3 {
        color: #004080;
    }
    .stButton button {
        background: linear-gradient(90deg, #28a745, #20c997);
        color: white;
        font-weight: bold;
        border-radius: 12px;
        padding: 12px 24px;
        border: none;
    }
    .stButton button:hover {
        background: linear-gradient(90deg, #218838, #17a2b8);
        color: white;
    }
    .input-card {
        background-color: #e6f7ff;
        border: 2px solid #99ccff;
        padding: 15px;
        border-radius: 12px;
        margin-bottom: 20px;
    }
    .prediction-box {
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        font-size: 22px;
        font-weight: bold;
        color: white;
        margin-top: 30px;
    }
    .healthy {
        background-color: #28a745;
        border: 3px solid #1e7e34;
    }
    .disease {
        background-color: #dc3545;
        border: 3px solid #a71d2a;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------
# Sidebar Navigation
# -------------------------------
st.sidebar.title("🩺 Navigation")
page = st.sidebar.radio("Go to:", ["Home", "Predict"])

# -------------------------------
# Home Page
# -------------------------------
if page == "Home":
    st.title("🩺 Liver & Cancer Prediction System")
    st.markdown(
        """
        Welcome to the **Liver & Cancer Prediction System**.  

        ---
        > *“The liver is a resilient organ – treat it with care.”*  
        > *“An ounce of prevention is worth a pound of cure.”*  
        ---
        """
    )
    col1, col2 = st.columns(2)
    with col1:
        st.image("static/images/liver1.jpg", caption="Human Liver ")
    with col2:
        st.image("static/images/liver2.jpg", caption="Liver Position in Human Body")

# -------------------------------
# Prediction Page
# -------------------------------
elif page == "Predict":
    st.header("🔮 Predict Liver Disease & Cancer")
    st.markdown("Enter the patient details below:")

    # Toggle for cancer check
    cancer_check = st.checkbox("Also check for Cancer if disease detected", value=True)

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("<div class='input-card'>", unsafe_allow_html=True)
            age = st.number_input("🧑 Age (years)", min_value=1, max_value=120, value=30)
            gender = st.selectbox("⚧ Gender", ["Male", "Female"])
            total_bilirubin = st.number_input("🟠 Total Bilirubin (mg/dL)", min_value=0.0, value=1.0)
            direct_bilirubin = st.number_input("🟠 Direct Bilirubin (mg/dL)", min_value=0.0, value=0.1)
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("<div class='input-card'>", unsafe_allow_html=True)
            alk_phos = st.number_input("🧪 Alkaline Phosphotase (IU/L)", min_value=10, value=200)
            sgpt = st.number_input("🧪 Alamine Aminotransferase (IU/L)", min_value=0, value=30)
            sgot = st.number_input("🧪 Aspartate Aminotransferase (IU/L)", min_value=0, value=35)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='input-card'>", unsafe_allow_html=True)
        col3, col4 = st.columns(2)
        with col3:
            total_protein = st.number_input("💧 Total Proteins (g/dL)", min_value=0.0, value=6.5)
            albumin = st.number_input("💧 Albumin (g/dL)", min_value=0.0, value=3.5)
        with col4:
            ag_ratio = st.number_input("⚖️ Albumin and Globulin Ratio", min_value=0.0, value=1.0)
        st.markdown("</div>", unsafe_allow_html=True)

        submitted = st.form_submit_button("🔍 Predict Now")

    if submitted:
        # Prepare input
        gender_val = 1 if gender == "Male" else 0
        input_data = np.array([[age, gender_val, total_bilirubin, direct_bilirubin,
                                alk_phos, sgpt, sgot, total_protein, albumin, ag_ratio]])

        # Old model preprocessing
        input_data_imputed = imputer.transform(input_data)
        input_data_scaled = scaler.transform(input_data_imputed)

        # Prediction
        prediction = model.predict(input_data_scaled)[0]
        prob = model.predict_proba(input_data_scaled)[0][prediction]
        confidence = confidence_label(prob)

        # -------------------------------
        # Output
        # -------------------------------
        if prediction == 1:
            st.markdown(
                f"<div class='prediction-box disease'>⚠️ Liver Disease Detected<br>Confidence: {confidence}</div>",
                unsafe_allow_html=True,
            )

            # Run clustering only if disease present
            clust_data = imputer_cluster.transform(input_data)
            clust_data = scaler_cluster.transform(clust_data)
            cluster_pred = cluster_model.predict(clust_data)[0]

            disease_type = disease_map[cluster_pred]

            st.subheader("📊 Detailed Disease Classification")
            st.write(f"**Liver Disease Type:** {disease_type}")

            # Cancer only if toggle is enabled
           

        else:
            st.markdown(
                f"<div class='prediction-box healthy'>✅ No Liver Disease Detected<br>Confidence: {confidence}</div>",
                unsafe_allow_html=True,
            )
