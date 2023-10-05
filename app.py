# Import statements
import streamlit as st
import tensorflow as tf
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
import pandas as pd
from datetime import datetime


# Load your trained model
model = tf.keras.models.load_model('/content/custom_model_EfficientNetB0.keras')

# Function to preprocess the image
def preprocess_img(img_path):
    """
    Preprocesses the input image for the model.

    Parameters:
        img_path (str): Path to the image file.

    Returns:
        numpy.ndarray: Preprocessed image array.
    """
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Streamlit App
st.title('Image Classification with Your Model')

# Start button to control the session
start_btn = st.button("Start Session")

if start_btn:
    session_active = True
else:
    session_active = False

uploaded_file = None

while session_active:
    # Choose an image
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)

        # Preprocess the image
        img_array = preprocess_img(uploaded_file)

        # Make predictions
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions)
        confidence = predictions[0, predicted_class]

        # Display the results
        st.subheader('Prediction:')
        st.write(f"Class: {predicted_class}")
        st.write(f"Confidence: {confidence:.2%}")

        # Buttons for user feedback
        col1, col2 = st.columns([3,3])
        correct_btn = col1.button("Correct")
        wrong_btn = col2.button("Wrong")

        # Save data on button click
        if correct_btn or wrong_btn:
            # Create a DataFrame to store feedback data
            feedback_data = {
                "Timestamp": [datetime.now()],
                "Image Name": [uploaded_file.name],
                "Prediction": [predicted_class],
                "Confidence": [confidence],
                "Feedback": ["Correct" if correct_btn else "Wrong"]
            }
            feedback_df = pd.DataFrame(feedback_data)

            # Save data to a CSV file with session timestamp
            session_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            feedback_file_name = f"feedback_data_session_{session_timestamp}.csv"
            feedback_df.to_csv(feedback_file_name, index=False)

            st.success(f"Feedback saved successfully: {feedback_file_name}")

            # Show buttons for next image or exit
            next_btn = st.button("Next Image")
            exit_btn = st.button("Exit Session")

            if exit_btn:
                session_active = False

    else:
        st.warning("Please upload an image to continue.")

# End of session
st.write("Session Ended.")
