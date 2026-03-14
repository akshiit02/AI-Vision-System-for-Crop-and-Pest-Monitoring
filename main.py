import streamlit as st
from app.decision_engine import final_decision

st.set_page_config(page_title="AI Crop Monitoring System")

st.title("🌾 AI-Powered Crop Monitoring System")

st.markdown("Upload images and enter soil values to get AI recommendations.")

# ============================
# Upload Images
# ============================
leaf_image = st.file_uploader("Upload Leaf Image", type=["jpg", "png"])
pest_image = st.file_uploader("Upload Pest Image", type=["jpg", "png"])

# ============================
# Soil Inputs
# ============================
st.subheader("Soil Parameters")

N = st.number_input("Nitrogen (N)", value=90.0)
P = st.number_input("Phosphorus (P)", value=42.0)
K = st.number_input("Potassium (K)", value=43.0)
temperature = st.number_input("Temperature", value=20.5)
humidity = st.number_input("Humidity", value=82.0)
ph = st.number_input("pH", value=6.5)
rainfall = st.number_input("Rainfall", value=202.0)

# ============================
# Predict Button
# ============================
if st.button("🔍 Run AI Analysis"):

    if leaf_image is None or pest_image is None:
        st.warning("Please upload both leaf and pest images.")
    else:
        # Save temporary images
        with open("temp_leaf.jpg", "wb") as f:
            f.write(leaf_image.read())

        with open("temp_pest.jpg", "wb") as f:
            f.write(pest_image.read())

        soil_values = [N, P, K, temperature, humidity, ph, rainfall]

        result = final_decision(
            soil_values,
            "temp_leaf.jpg",
            "temp_pest.jpg"
        )

        st.success("✅ AI Analysis Complete")

        st.subheader("Results")
        st.write("🌱 Recommended Crop:", result["Recommended Crop"])
        st.write("🦠 Disease Detected:", result["Disease Detected"])
        st.write("🐛 Pest Detected:", result["Pest Detected"])
        st.write("💡 Advisory:", result["Advice"])
