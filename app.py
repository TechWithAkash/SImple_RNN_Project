# Step 1: Import Libraries and Load the Model
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

# Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load the pre-trained model with ReLU activation
model = load_model('SimpleRNN_Model.h5')

# Step 2: Helper Functions
# Function to decode reviews
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review


import streamlit as st
## streamlit app
# Streamlit app
st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to analyze it as positive or negative.')

# User input
user_input = st.text_area('Movie Review')

if st.button('Analyze'):

    preprocessed_input=preprocess_text(user_input)

    ## MAke prediction
    prediction=model.predict(preprocessed_input)
    sentiment='Positive' if prediction[0][0] > 0.5 else 'Negative'

    # Display the result
    st.write(f'Sentiment: {sentiment}')
    st.write(f'Prediction Score: {prediction[0][0]}')
else:
    st.write('Please enter a movie review.')

















































# import numpy as np
# import pandas as pd
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import sequence
# import streamlit as st

# # Step 1: Import Libraries and Load Models

# # Check if the model file exists and is accessible
# import os

# def check_model_file(file_path):
#     """Verify if the model file exists and is accessible."""
#     if not os.path.exists(file_path):
#         raise FileNotFoundError(f"Model file not found at {file_path}. Please ensure the file exists.")
#     return file_path

# # Attempt to load the pre-trained model with error handling
# try:
#     MODEL_PATH = 'simple_rnn_model_imdb.h5'
#     model = load_model(check_model_file(MODEL_PATH))
# except Exception as e:
#     st.error(f"Error loading the model: {e}")
#     model = None

# # Load Word Index from Keras IMDB dataset
# word_index = tf.keras.datasets.imdb.get_word_index()
# reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

# # Step 2: Helper Functions for Text Processing
# def decode_review(encoded_review):
#     """
#     Decode an encoded review back to readable text.
    
#     Args:
#         encoded_review (list): Encoded review as a list of integers
    
#     Returns:
#         str: Decoded review text
#     """
#     # Adjust indices to match the original word_index
#     return ' '.join([reverse_word_index.get(i-3, '?') for i in encoded_review])

# def preprocess_text(text):
#     """
#     Convert input text to a padded sequence of indices.
    
#     Args:
#         text (str): Input review text
    
#     Returns:
#         numpy.ndarray: Padded sequence of word indices
#     """
#     # Lowercase and split the text
#     words = text.lower().split()
    
#     # Convert words to indices
#     words_indices = [word_index.get(word, 2) for word in words]
    
#     # Pad sequences to a fixed length
#     return sequence.pad_sequences([words_indices], maxlen=550)

# # Step 3: Sentiment Prediction Function
# def predict_sentiment(review):
#     """
#     Predict sentiment of a given review.
    
#     Args:
#         review (str): Input movie review text
    
#     Returns:
#         tuple: (sentiment, confidence)
#     """
#     if not review:
#         return None, None
    
#     # Preprocess the input text
#     preprocessed_input = preprocess_text(review)
    
#     # Make prediction
#     prediction = model.predict(preprocessed_input)
    
#     # Determine sentiment and confidence
#     sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
#     confidence = prediction[0][0] * 100 if sentiment == 'Positive' else (1 - prediction[0][0]) * 100
    
#     return sentiment, round(confidence, 2)

# # Step 4: Streamlit Web App Interface
# def main():
#     # Page configuration
#     st.set_page_config(
#         page_title="Movie Review Sentiment Analyzer",
#         page_icon="🎬",
#         layout="wide"
#     )
    
#     # Custom CSS for enhanced styling
#     st.markdown("""
#     <style>
#     .main-header {
#         background-color: #87b3fa;
#         color: white;
#         padding: 20px;
#         text-align: center;
#         border-radius: 10px;
#     }
#     .sentiment-container {
#         background-color: #f4f4f4;
#         border-radius: 10px;
#         padding: 20px;
#         margin-top: 20px;
#     }
#     .stTextArea textarea {
#         background-color: #f8f9fa;
#         border-radius: 10px;
#     }
#     .stButton>button {
#         background-color: #87b3fa;
#         color: white;
#         border-radius: 10px;
#         padding: 10px 20px;
#         transition: all 0.3s ease;
#     }
#     .stButton>button:hover {
#         background-color: #4a8df7;
#         transform: scale(1.05);
#     }
#     </style>
#     """, unsafe_allow_html=True)
    
#     # Header
#     st.markdown('<div class="main-header"><h1>🎬 Movie Review Sentiment Analyzer</h1></div>', unsafe_allow_html=True)
    
#     # Introduction
#     st.markdown("""
#     ### Analyze the Sentiment of Movie Reviews Using AI
#     Enter a movie review below and our advanced machine learning model will predict its sentiment.
#     """)
    
#     # Model availability check
#     if model is None:
#         st.error("Model could not be loaded. Please check the model file.")
#         return
    
#     # User Input
#     user_input = st.text_area(
#         "Paste your movie review here", 
#         height=250, 
#         placeholder="Write or paste a movie review..."
#     )
    
#     # Analyze Button
#     if st.button("Analyze Sentiment"):
#         if user_input.strip():
#             # Predict sentiment
#             sentiment, confidence = predict_sentiment(user_input)
            
#             # Display Results
#             st.markdown('<div class="sentiment-container">', unsafe_allow_html=True)
            
#             if sentiment == 'Positive':
#                 st.success(f"🌟 Sentiment: {sentiment}")
#                 st.metric("Confidence", f"{confidence}%", delta=f"+{confidence}%")
#             else:
#                 st.error(f"🚫 Sentiment: {sentiment}")
#                 st.metric("Confidence", f"{confidence}%", delta=f"-{100-confidence}%")
            
#             # Additional Insights
#             st.info("""
#             ### Insights
#             - Our AI analyzes the textual nuances of the review
#             - Sentiment is determined by advanced neural network processing
#             - Confidence reflects the model's certainty in its prediction
#             """)
            
#             st.markdown('</div>', unsafe_allow_html=True)
#         else:
#             st.warning("Please enter a movie review to analyze.")
    
#     # Footer
#     st.markdown("""
#     ---
#     © 2024 Movie Review Sentiment Analyzer | Powered by AI and TensorFlow
#     """)

# # Run the App
# if __name__ == "__main__":
#     main()