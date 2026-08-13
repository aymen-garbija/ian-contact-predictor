import gradio as gr
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

model = load_model("ian_contact_model.h5")

def predict_contact(img):
    img_resized = img.convert("RGB").resize((128, 128))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)[0][0]
    if prediction >= 0.5:
        return f"Contact probability: {prediction*100:.1f}%"
    else:
        return f"No Contact probability: {(1-prediction)*100:.1f}%"

demo = gr.Interface(fn=predict_contact, inputs=gr.Image(type="pil"), outputs=gr.Textbox())
demo.launch()
