import streamlit as st
import numpy as np
import pandas as pd
import librosa
import os
import sqlite3
import re

# Constants
DATABASE = "mydb.sqlite3"
audio_dir = 'audio_files'
num_mfcc = 100
num_mels = 128
num_chroma = 50

# Load dataset
dataset = pd.read_csv('dataset.csv')

# SQLite Database connection
def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    return conn




# Function to perform audio detection
def detect_audio(file_path):
    X, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=num_mfcc).T, axis=0)
    mel_spectrogram = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate, n_mels=num_mels).T, axis=0)
    chroma_features = np.mean(librosa.feature.chroma_stft(y=X, sr=sample_rate, n_chroma=num_chroma).T, axis=0)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=X).T, axis=0)
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=X, sr=sample_rate).T, axis=0)
    flatness = np.mean(librosa.feature.spectral_flatness(y=X).T, axis=0)
    
    features = np.concatenate((mfccs, mel_spectrogram, chroma_features, zcr, spectral_centroid, flatness))
    distances = np.linalg.norm(dataset.iloc[:, :-1] - features, axis=1)
    closest_match_idx = np.argmin(distances)
    closest_match_label = dataset.iloc[closest_match_idx, -1]
    total_distance = np.sum(distances)
    closest_match_prob = 1 - (distances[closest_match_idx] / total_distance)
    closest_match_prob_percentage = "{:.3f}".format(closest_match_prob * 100)

    return closest_match_label, closest_match_prob_percentage

# Streamlit application
st.title("Audio Detection App")

# Navigation
menu = st.sidebar.selectbox("Select Page", ["Home","Model", "About"])

if menu == "Home":
    st.image("static/image1.jpg")
    st.write("Welcome to the Audio Detection App!")



elif menu == "Model":
    st.image("static/image5.jpg")
    uploaded_file = st.file_uploader("Choose an audio file...", type=["wav", "mp3"])

    if uploaded_file is not None:
        file_path = os.path.join(audio_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.write("Analyzing...")
        closest_match_label, closest_match_prob = detect_audio(file_path)
        result_message = f"Result: {'Fake' if closest_match_label == 'deepfake' else 'Real'} with {closest_match_prob}%"
        st.success(result_message)

        # Clean up
        os.remove(file_path)

elif menu == "About":
    st.image("static/image2.jpg")
    st.write("This is an audio detection app built using Streamlit.")
