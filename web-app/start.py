import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw
from mtcnn import MTCNN

# 1. Modell & Gesichts-Detektor laden
@st.cache_resource
def load_resources():
    model = tf.keras.models.load_model('../models/mobilenet_augmented.keras')
    detector = MTCNN()
    return model, detector

model, detector = load_resources()

EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# 2. UI Aufsetzen
st.title("🎭 Social Media Emotionserkennung")
st.write("Analyse von Gesichtern und Emotionen")

uploaded_file = st.file_uploader("Bild auswählen (JPG, PNG)...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    # Bild mit PIL öffnen
    image = Image.open(uploaded_file).convert('RGB')
    image_np = np.array(image)
    
    # Gesichter erkennen (MTCNN arbeitet nativ mit PIL/NumPy)
    faces = detector.detect_faces(image_np)
    
    if len(faces) == 0:
        st.warning("Keine Gesichter im Bild erkannt.")
    else:
        # Kopie für Zeichnungen erstellen
        annotated_image = image.copy()
        draw = ImageDraw.Draw(annotated_image)
        
        for i, face in enumerate(faces):
            x, y, width, height = face['box']
            # Korrektur für negative Koordinaten-Ränder
            x, y = max(0, x), max(0, y)
            
            # 1. Gesicht mit PIL zuschneiden
            face_crop = image.crop((x, y, x + width, y + height))
            
            # 2. Auf 48x48 skalieren (PIL Image)
            face_resized = face_crop.resize((48, 48))
            
            # 3. In NumPy Tensor umwandeln
            face_array = np.array(face_resized)
            
            # Je nach Modell: 3-Kanal RGB oder 1-Kanal Graustufe
            # Falls dein Modell RGB (48,48,3) erwartet:
            input_tensor = np.expand_dims(face_array, axis=0)
            
            # 4. Inferenz
            predictions = model.predict(input_tensor)[0]
            max_idx = np.argmax(predictions)
            predicted_emotion = EMOTION_LABELS[max_idx]
            confidence = predictions[max_idx] * 100
            
            # 5. Bounding Box mit PIL zeichnen
            draw.rectangle([x, y, x + width, y + height], outline="lime", width=3)
            
            # Text & Ausgabe
            st.write(f"**Gesicht {i+1}:** {predicted_emotion} ({confidence:.1f}%)")
            st.progress(int(confidence))
            
        # Bild in Streamlit anzeigen
        st.image(annotated_image, caption="Analysiertes Bild", use_container_width=True)
