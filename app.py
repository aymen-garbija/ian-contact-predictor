import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

st.title("IAN-M3M Contact Predictor")
st.write("Upload a panoramic X-ray image to predict contact probability.")

@st.cache_resource
def get_model():
    return load_model("ian_contact_model.h5")

model = get_model()

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)
    
    img_resized = img.convert("RGB").resize((128, 128))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)[0][0]
    
    if prediction >= 0.5:
        st.error(f"Contact probability: {prediction*100:.1f}%")
    else:
        st.success(f"No Contact probability: {(1-prediction)*100:.1f}%")
