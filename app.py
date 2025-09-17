import streamlit as st
import joblib
import numpy as np

# -------------------------------
# Load Model
# -------------------------------
model_data = joblib.load("liver_model.pkl")
model = model_data["model"]
scaler = model_data["scaler"]
imputer = model_data["imputer"]
threshold = model_data["threshold"]

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Liver Disease Prediction", layout="wide")

# -------------------------------
# Custom CSS
# -------------------------------
st.markdown(
    """
    <style>
    body {
        background-color: #f0f8ff; /* light blue */
    }
    .main {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 15px;
    }
    h1, h2, h3 {
        color: #004080; /* hospital blue */
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
        background-color: #28a745; /* green */
        border: 3px solid #1e7e34;
    }
    .disease {
        background-color: #dc3545; /* red */
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
    st.title("🩺 Liver Disease Prediction System")
    st.markdown(
        """
        Welcome to the **Liver Disease Prediction System**.  

        ---
        > *“The liver is a resilient organ – treat it with care.”*  
        > *“An ounce of prevention is worth a pound of cure.”*  
        ---
        """
    )

    col1, col2 = st.columns(2)
    with col1:
        st.image("static/images/liver1.jpg", caption="Human Liver Anatomy", use_container_width=True)
    with col2:
        st.image("static/images/liver2.jpg", caption="Liver Position in Human Body", use_container_width=True)

# -------------------------------
# Prediction Page
# -------------------------------
elif page == "Predict":
    st.header("🔮 Predict Liver Disease")
    st.markdown("Enter the patient details below:")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("<div class='input-card'>", unsafe_allow_html=True)
            Age = st.number_input("🧑 Age (years)", min_value=1, max_value=120, value=30)
            Gender = st.selectbox("⚧ Gender", ["Male", "Female"])
            Total_Bilirubin = st.number_input("🟠 Total Bilirubin (mg/dL)", min_value=0.0, value=1.0)
            Direct_Bilirubin = st.number_input("🟠 Direct Bilirubin (mg/dL)", min_value=0.0, value=0.1)
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("<div class='input-card'>", unsafe_allow_html=True)
            Alkaline_Phosphotase = st.number_input("🧪 Alkaline Phosphotase (IU/L)", min_value=10, value=200)
            Alamine_Aminotransferase = st.number_input("🧪 Alamine Aminotransferase (IU/L)", min_value=0, value=30)
            Aspartate_Aminotransferase = st.number_input("🧪 Aspartate Aminotransferase (IU/L)", min_value=0, value=35)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='input-card'>", unsafe_allow_html=True)
        col3, col4 = st.columns(2)
        with col3:
            Total_Protiens = st.number_input("💧 Total Proteins (g/dL)", min_value=0.0, value=6.5)
            Albumin = st.number_input("💧 Albumin (g/dL)", min_value=0.0, value=3.5)
        with col4:
            Albumin_and_Globulin_Ratio = st.number_input("⚖️ Albumin and Globulin Ratio", min_value=0.0, value=1.0)
        st.markdown("</div>", unsafe_allow_html=True)

        submitted = st.form_submit_button("🔍 Predict Now")

    if submitted:
        # Convert inputs
        input_data = np.array([[Age,
                                1 if Gender == "Male" else 0,
                                Total_Bilirubin,
                                Direct_Bilirubin,
                                Alkaline_Phosphotase,
                                Alamine_Aminotransferase,
                                Aspartate_Aminotransferase,
                                Total_Protiens,
                                Albumin,
                                Albumin_and_Globulin_Ratio]])

        # Preprocess
        input_data = imputer.transform(input_data)
        input_data = scaler.transform(input_data)

        # Prediction
        prob = model.predict_proba(input_data)[0, 1]
        pred = int(prob >= threshold)
        confidence = round(prob * 100, 2)

        # Show colorful prediction box
        if pred == 1:
            st.markdown(
                f"<div class='prediction-box disease'>⚠️ Liver Disease Detected<br>Confidence: {confidence}%</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div class='prediction-box healthy'>✅ Healthy Liver<br>Confidence: {confidence}%</div>",
                unsafe_allow_html=True,
            )
