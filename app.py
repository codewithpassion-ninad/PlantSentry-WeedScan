from imp import load_module
import streamlit as st
import cv2
import numpy as np
from PIL import Image

model = load_module("D:\\RCNN_crop_weed_classification_model.h5")

def detect_crop_and_weed(image):
    detection_results = model.detect_objects(image)
    detection_results = []
    return detection_results

st.title("Crop and Weed Detection App")

uploaded_image = st.file_uploader("Upload an image:", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    image = np.array(image)

    detection_results = detect_crop_and_weed(image)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    if detection_results:
        st.write("Detection Results:")
        for result in detection_results:
            st.write(f"- Class: {result['class']}, Confidence: {result['confidence']:.2f}")
            st.image(result['annotated_image'], use_column_width=True)
    else:
        st.write("No objects detected.")