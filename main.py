import streamlit as st
import librosa
import numpy as np
import soundfile as sf
import io
import scipy.ndimage

# --- CUSTOM STYLING (CSS) ---
def style_app():
    st.markdown("""
        <style>
        /* Hide default Streamlit elements */
        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        footer {visibility: hidden;}
        
        .stApp {
            background-color: #0e1117;
            color: #ffffff;
        }

        .block-container {
            padding-top: 5rem;
            max-width: 700px;
        }
        
        /* The "NOI"SE REDU"CE" Branding */
        .main-header {
            font-size: 3.5rem;
            font-weight: 800;
            color: #ffffff;
            text-align: center;
            margin-bottom: 0px;
            letter-spacing: 2px;
        }

        .golden-text {
            color: #FFD700;
            text-shadow: 0px 0px 20px rgba(255, 215, 0, 0.5);
        }

        /* File Format Instruction */
        .format-label {
            text-align: center;
            color: #94a3b8;
            font-size: 0.9rem;
            margin-bottom: 2rem;
            margin-top: 10px;
        }

        /* Centering the uploader box */
        .stFileUploader {
            padding-top: 1rem;
        }

        div.stButton > button:first-child {
            background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%);
            color: white;
            border: none;
            padding: 0.8rem;
            border-radius: 10px;
            font-weight: bold;
            width: 100%;
            margin-top: 20px;
        }
        </style>
    """, unsafe_allow_html=True)

# --- PROCESSING LOGIC ---
def process_audio(input_audio):
    y, sr = librosa.load(input_audio, sr=None)
    stft = librosa.stft(y)
    magnitude, phase = librosa.magphase(stft)

    # Rolling window adaptive noise estimation
    noise_est = scipy.ndimage.minimum_filter(magnitude, size=(1, 43))
    noise_est = scipy.ndimage.gaussian_filter(noise_est, sigma=(0, 2))

    # Subtraction with Gain Floor
    clean_mag = np.maximum(magnitude - (noise_est * 1.5), 0.15 * magnitude)

    y_clean = librosa.istft(clean_mag * phase)

    buffer = io.BytesIO()
    sf.write(buffer, y_clean, sr, format='WAV')
    buffer.seek(0)
    return buffer

# --- APP LAYOUT ---
st.set_page_config(page_title="NOICE", page_icon="✨", layout="centered")
style_app()

# Unified Header
st.markdown("""
    <div class="main-header">
        <span class="golden-text">NOI</span>SE REDU<span class="golden-text">CE</span>
    </div>
    <p class="format-label">Supported formats: .WAV, .MP3 (Max 200MB)</p>
""", unsafe_allow_html=True)

# Upload Section
uploaded_file = st.file_uploader("", type=["wav", "mp3"], label_visibility="collapsed")

if uploaded_file is not None:
    st.audio(uploaded_file)
    if st.button(" CLEAN AUDIO NOW"):
        with st.spinner("Processing..."):
            processed_data = process_audio(uploaded_file)
            st.success("✨ Success!")
            st.audio(processed_data)
            st.download_button(
                label=" 😉 DOWNLOAD CLEANED WAV",
                data=processed_data,
                file_name="noice_cleaned_audio.wav",
                mime="audio/wav"
            )
