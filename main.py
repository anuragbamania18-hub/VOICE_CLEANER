import streamlit as st
import librosa
import numpy as np
import soundfile as sf
import io
import noisereduce as nr  # The specialized library

# --- CUSTOM STYLING (CSS) ---
def style_app():
    st.markdown("""
        <style>
        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        footer {visibility: hidden;}
        .stApp { background-color: #0e1117; color: #ffffff; }
        .block-container { padding-top: 5rem; max-width: 700px; }
        .main-header { font-size: 3.5rem; font-weight: 800; text-align: center; margin-bottom: 0px; }
        .golden-text { color: #FFD700; text-shadow: 0px 0px 20px rgba(255, 215, 0, 0.5); }
        .format-label { text-align: center; color: #94a3b8; font-size: 0.9rem; margin-bottom: 2rem; }
        div.stButton > button:first-child {
            background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%);
            color: white; border: none; padding: 0.8rem; border-radius: 10px; width: 100%;
        }
        </style>
    """, unsafe_allow_html=True)

# --- LIBRARY-BASED PROCESSING LOGIC ---
def process_audio(input_audio):
    # 1. Load Audio
    y, sr = librosa.load(input_audio, sr=None)
    
    # 2. Apply Noisereduce Library
    # stationary=False: Better for wind/moving noise
    # prop_decrease=1.0: How much noise to remove (1.0 = 100%)
    y_clean = nr.reduce_noise(y=y, sr=sr, stationary=False, prop_decrease=1.0)
    
    # 3. Final Normalization
    y_final = librosa.util.normalize(y_clean)

    buffer = io.BytesIO()
    sf.write(buffer, y_final, sr, format='WAV')
    buffer.seek(0)
    return buffer

# --- APP LAYOUT ---
st.set_page_config(page_title="NOICE", page_icon="✨", layout="centered")
style_app()

st.markdown("""
    <div class="main-header">
        <span class="golden-text">NOI</span>SE REDU<span class="golden-text">CE</span>
    </div>
    <p class="format-label">Powered by Spectral Gating (.WAV, .MP3)</p>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type=["wav", "mp3"], label_visibility="collapsed")

if uploaded_file is not None:
    st.audio(uploaded_file)
    if st.button("🚀 CLEAN AUDIO NOW"):
        with st.spinner("Executing advanced spectral gating..."):
            processed_data = process_audio(uploaded_file)
            st.success("✨ Processed Successfully!")
            st.audio(processed_data)
            st.download_button(label="📥 DOWNLOAD CLEANED WAV", data=processed_data, 
                               file_name="noice_pro_clean.wav", mime="audio/wav")
