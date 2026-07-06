import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageOps
import numpy as np
import joblib
from pathlib import Path


def center_of_mass(image: np.ndarray) -> np.ndarray:
    image = image.astype("float32")
    rows, cols = image.shape
    y_coords, x_coords = np.indices((rows, cols))
    total_mass = image.sum()
    if total_mass == 0:
        return image
    center_y = np.sum(image * y_coords) / total_mass
    center_x = np.sum(image * x_coords) / total_mass
    shift_y = (rows - 1) / 2.0 - center_y
    shift_x = (cols - 1) / 2.0 - center_x
    shifted = np.zeros_like(image)
    for i in range(rows):
        for j in range(cols):
            new_i = int(round(i + shift_y))
            new_j = int(round(j + shift_x))
            if 0 <= new_i < rows and 0 <= new_j < cols:
                shifted[new_i, new_j] = image[i, j]
    return shifted

# Load trained model and the matching scaler
@st.cache_resource()
def load_model():
    base_path = Path(__file__).resolve().parent
    model_path = base_path / "model_finetuned.h5"
    scaler_path = base_path / "scaler.joblib"

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

try:
    model, scaler = load_model()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

st.title("Handwritten Digit Classifier")

canvas_result = st_canvas(
    fill_color="white",
    stroke_width=8,
    stroke_color="black",
    background_color="white",
    height=200,
    width=200,
    drawing_mode="freedraw",
    key="canvas"
)

if st.button("Classify Drawing"):
    if canvas_result.image_data is not None:
        img = Image.fromarray((canvas_result.image_data[:, :, :3]).astype(np.uint8))
        img = img.convert("L")  # Convert to grayscale
        img = ImageOps.invert(img)  # Invert colors: black digit on white background
        img = img.resize((8, 8), Image.LANCZOS)  # Match the 8x8 training data shape
        img_array = np.array(img, dtype="float32")
        img_array = center_of_mass(img_array)
        img_array = img_array / 255.0 * 16.0  # Match the original digits dataset scale
        img_array = img_array.reshape(1, -1)
        img_array = scaler.transform(img_array)
        
        prediction = model.predict(img_array)
        predicted_class = int(prediction[0])
        
        st.image(img, caption="Processed Image", width=150)
        st.write(f"Predicted Class: {predicted_class}")

if st.button("Retry"):
    st.rerun()
