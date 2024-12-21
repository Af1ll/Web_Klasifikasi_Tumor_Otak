import os
import streamlit as st
import tensorflow as tf
from keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
import numpy as np
import cv2

# Mapping kelas sesuai dengan model pelatihan
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Load model pre-trained
MODEL_PATH = 'imageclassifierfix.h5'
model = load_model(MODEL_PATH)

def predict_label(img):
    """Predict the class of an image."""
    # Resize and preprocess the image
    img = cv2.resize(img, (224, 224))
    img_array = preprocess_input(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Predict using the model
    predictions = model.predict(img_array)[0]
    class_idx = np.argmax(predictions)
    class_name = CLASS_NAMES[class_idx]
    class_prob = predictions[class_idx]

    # Dictionary penjelasan diagnosis
    diagnosis_explanations = {
        'glioma': 'Glioma adalah jenis tumor otak yang berasal dari sel glial. Penanganan tergantung pada ukuran dan lokasi tumor.',
        'meningioma': 'Meningioma adalah tumor otak yang umumnya bersifat jinak, berasal dari jaringan di sekitar otak dan sumsum tulang belakang.',
        'notumor': 'Tidak terdeteksi adanya tumor otak. Namun, selalu ingat untuk tetap menjaga kesehatan tubuh dengan pola hidup sehat. Konsultasikan dengan dokter untuk pemeriksaan lanjutan jika Anda memiliki gejala. Kesehatan adalah aset terbesar, jadi jangan ragu untuk melakukan pengecekan rutin dan menjaga keseimbangan hidup yang baik!',
        'pituitary': 'Pituitary tumor adalah tumor kelenjar hipofisis yang dapat mempengaruhi fungsi hormonal tubuh.'
    }
    
    explanation = diagnosis_explanations.get(class_name, 'Penjelasan tidak tersedia untuk kelas ini.')
    result = f"**Hasil Diagnosa**: {class_name} ({class_prob * 100:.2f}%)\n\n**Penjelasan**: {explanation}"
    return result

# Streamlit App
st.set_page_config(page_title="Aplikasi Klasifikasi Tumor Otak", layout="wide")

# Title
st.title("Klasifikasi Tumor Otak dengan AI")

# Upload Image Section
uploaded_file = st.file_uploader("Upload Gambar MRI Anda", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the image using OpenCV
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Convert BGR to RGB for visualization
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Display the uploaded image
    st.image(img_rgb, caption="Gambar yang Diunggah", use_column_width=True)

    # Predict Button
    if st.button("Prediksi"):
        try:
            result = predict_label(img)
            st.markdown(result, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses gambar: {e}")
else:
    st.info("Unggah gambar MRI untuk memulai klasifikasi.")
