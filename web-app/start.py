import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from mtcnn import MTCNN
from typing import List, Dict, Any, Tuple

# Configuration Constants
MODEL_PATH = '../models/resnet_augmented.keras'
EMOTION_LABELS = ['Wut', 'Ekel', 'Angst', 'Freude', 'Neutral', 'Trauer', 'Überraschung']
# EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']


@st.cache_resource
def load_resources() -> Tuple[tf.keras.Model, MTCNN]:
    """Loads and caches the Keras emotion recognition model and MTCNN face detector.

    Returns:
        Tuple[tf.keras.Model, MTCNN]: Cached model instance and face detector.
    """
    model = tf.keras.models.load_model(MODEL_PATH)
    detector = MTCNN()
    return model, detector


def process_and_predict_faces(
    image: Image.Image, 
    faces: List[Dict[str, Any]], 
    model: tf.keras.Model, 
    labels: List[str]
) -> Tuple[Image.Image, List[Dict[str, Any]]]:
    """Crops faces from the input image, resizes them, runs emotion inference,

    and draws bounding boxes on an annotated copy of the image.

    Args:
        image (Image.Image): Original input image (RGB format).
        faces (List[Dict[str, Any]]): List of bounding box metadata detected by
          MTCNN.
        model (tf.keras.Model): Trained Keras classification model.
        labels (List[str]): List of emotion label strings matching model outputs.

    Returns:
        Tuple[Image.Image, List[Dict[str, Any]]]:
            - Annotated PIL Image with bounding boxes.
            - List of dictionaries containing detection results and prediction
            probabilities.
    """
    annotated_image = image.copy()
    draw = ImageDraw.Draw(annotated_image)
    predictions_list = []

    for i, face in enumerate(faces):
        # Extract bounding box coordinates and handle negative margins
        x, y, width, height = face['box']
        x, y = max(0, x), max(0, y)

        # crop quadratic
        size = max(width, height)
        cx = x + width / 2
        cy = y + height / 2
        left = int(cx - size / 2)
        top = int(cy - size / 2)
        right = int(cx + size / 2)
        bottom = int(cy + size / 2)
        face_crop = image.crop((left, top, right, bottom))
        # resize to 48x48
        face_resized = face_crop.resize((48, 48), Image.Resampling.LANCZOS)


        # Prepare input tensor: (48, 48, 3) -> (1, 48, 48, 3)
        input_tensor = np.expand_dims(np.array(face_resized), axis=0)

        # Perform model inference
        raw_predictions = model.predict(input_tensor, verbose=0)[0]
        max_idx = np.argmax(raw_predictions)
        predicted_emotion = labels[max_idx]
        confidence = float(raw_predictions[max_idx] * 100)

        # Draw bounding box on image
        draw.rectangle([x, y, x + width, y + height], outline="lime", width=3)

        predictions_list.append({
            "face_index": i + 1,
            "emotion": predicted_emotion,
            "confidence": confidence,
            "probs": raw_predictions
        })

    return annotated_image, predictions_list


# Application Setup
model, detector = load_resources()

st.title("🎭 Social-Media-Emotionserkennung")

uploaded_file = st.file_uploader(
    "Wähle ein Bild aus (JPG, PNG, WEBP)...", 
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:
    # Load image and convert to RGB
    image = Image.open(uploaded_file).convert('RGB')
    image_np = np.array(image)

    # Detect faces in image
    faces = detector.detect_faces(image_np)

    if not faces:
        st.warning("Keine Gesichter im Bild erkannt.")
    else:
        # Run image processing and inference pipeline
        annotated_img, predictions = process_and_predict_faces(
            image=image, 
            faces=faces, 
            model=model, 
            labels=EMOTION_LABELS
        )

        # Two-column UI layout
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Erkanntes Gesicht")
            st.image(annotated_img, caption="Analysiertes Bild", width='stretch')

        with col2:
            st.subheader("Emotions-Verteilung")

            for pred in predictions:
                if len(predictions) > 1:
                    st.markdown(
                        f"**Gesicht {pred['face_index']}:** {pred['emotion']} ({pred['confidence']:.1f}%)"
                    )
                else:
                    st.markdown(
                        f"Hauptemotion: **{pred['emotion']}** ({pred['confidence']:.1f}%)"
                    )

                # Prepare DataFrame for class probability visualization
                chart_data = pd.DataFrame({
                    "Emotion": EMOTION_LABELS,
                    "Wahrscheinlichkeit (%)": pred['probs'] * 100
                }).set_index("Emotion")

                st.bar_chart(chart_data)