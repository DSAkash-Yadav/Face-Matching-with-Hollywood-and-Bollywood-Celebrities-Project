import streamlit as st
import os
import pickle
import cv2
import numpy as np
from PIL import Image
from mtcnn import MTCNN
from sklearn.metrics.pairwise import cosine_similarity
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace

# Initialize MTCNN face detector
detector = MTCNN()

# Load pre-trained VGGFace model
model = VGGFace(model='resnet50', include_top=False, input_shape=(224,224,3), pooling='avg')

# Load precomputed features and filenames
feature_list = pickle.load(open('embedding.pkl', 'rb'))
filenames = pickle.load(open('filenames.pkl', 'rb'))

# Function to save uploaded image
def save_uploaded_image(uploaded_image):
    try:
        os.makedirs('uploads', exist_ok=True)  # Ensure directory exists
        with open(os.path.join('uploads', uploaded_image.name), 'wb') as f:
            f.write(uploaded_image.getbuffer())
        return True
    except Exception as e:
        st.error(f"Error saving image: {e}")
        return False

# Function to extract features from the uploaded image
def extract_features(img_path, model, detector):
    img = cv2.imread(img_path)

    # Detect face
    results = detector.detect_faces(img)
    if len(results) == 0:
        st.error("No face detected. Please upload another image.")
        return None

    x, y, width, height = results[0]['box']
    face = img[y:y + height, x:x + width]

    # Convert to PIL image and resize
    image = Image.fromarray(face)
    image = image.resize((224, 224))
    face_array = np.asarray(image).astype('float32')

    # Preprocess and extract features
    expanded_img = np.expand_dims(face_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img)
    result = model.predict(preprocessed_img).flatten()
    return result

# Function to find the most similar celebrity
def recommend(feature_list, features):
    similarity = [cosine_similarity(features.reshape(1, -1), feature.reshape(1, -1))[0][0] for feature in feature_list]
    index_pos = sorted(enumerate(similarity), reverse=True, key=lambda x: x[1])[0][0]
    return index_pos

# Streamlit App UI
st.title('Which Bollywood Celebrity Are You?')

uploaded_image = st.file_uploader('Upload an image')

if uploaded_image is not None:
    if save_uploaded_image(uploaded_image):
        display_image = Image.open(uploaded_image)

        # Extract features
        features = extract_features(os.path.join('uploads', uploaded_image.name), model, detector)

        if features is not None:
            # Recommend celebrity
            index_pos = recommend(feature_list, features)
            predicted_actor = " ".join(filenames[index_pos].replace("\\", "/").split('/')[-1].split('_'))

            # Display results
            col1, col2 = st.columns(2)

            with col1:
                st.header('Your Uploaded Image')
                st.image(display_image)

            with col2:
                st.header(f"Looks like {predicted_actor}!")
                
                image_path = filenames[index_pos].replace("\\", "/")  # Fix path issues
                if os.path.exists(image_path):
                    st.image(image_path, width=300)
                else:
                    st.error("Celebrity image not found. Check file path.")

