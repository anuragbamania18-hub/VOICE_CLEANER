import streamlit as st
import librosa
import numpy as np
import soundfile as sf
import io

# --- (CSS) ---
def style_app():
    st.markdown("""
        <style>
        /* Main Background */
        .stApp {
            background-color: #0e1117;
            color: #ffffff;
        }
        
        /* Custom Header */
        .main-header {
            font-size: 2.5rem;
            font-weight: 800;
            color: #00d4ff;
            text-align: center;
            margin-bottom: 0.5rem;
            text-shadow: 0px 4px 10px rgba(0, 212, 255, 0.3);
        }

        /* Subtext */
        .sub-text {
            text-align: center;
            color: #94a3b8;
            margin-bottom: 2rem;
        }

        /* Upload Box Styling */
        .stFileUploader section {
            background-color: #1e293b !important;
            border: 2px dashed #334155 !important;
            border-radius: 15px !important;
        }

        /* Success & Info Boxes */
        .stAlert {
            border-radius: 10px !important;
            border: none !important;
        }

        /* Button Styling */
        div.stButton > button:first-child {
            background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%);
            color: white;
            border: none;
            padding: 0.6rem 2rem;
            border-radius: 50px;
            font-weight: bold;
            width: 100%;
            transition: all 0.3s ease;
            box-shadow: 0px 4px 15px rgba(58, 123, 213, 0.4);
        }

        div.stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0px 6px 20px rgba(58, 123, 213, 0.6);
            color: #ffffff;
        }

        /* Audio Player Styling */
        audio {
            width: 100%;
            border-radius: 10px;
        }
        </style>
    """, unsafe_allow_html=True)

# --- PROCESSING LOGIC ---
def process_audio(input_audio):
    y, sr = librosa.load(input_audio, sr=None)
    stft = librosa.stft(y)
    magnitude, phase = librosa.magphase(stft)
    
    # Baseline noise reduction (efficient matrix subtraction)
    noise_floor = np.mean(magnitude[:, :15], axis=1, keepdims=True)
    clean_mag = np.maximum(magnitude - (noise_floor * 1.5), 0)
    
    y_clean = librosa.istft(clean_mag * phase)
    
    buffer = io.BytesIO()
    sf.write(buffer, y_clean, sr, format='WAV')
    buffer.seek(0)
    return buffer

# --- APP LAYOUT ---
st.set_page_config(page_title="SonicClean AI", page_icon="🌊", layout="centered")
style_app()

st.markdown('<h1 class="main-header">🌊 SonicClean AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">Professional Grade Fourier-Transform Noise Reduction</p>', unsafe_allow_html=True)

# Layout Columns
col1, col2 = st.columns([1, 1])

with st.container():
    uploaded_file = st.file_uploader("", type=["wav", "mp3"])

if uploaded_file is not None:
    st.info("🎵 File Loaded Successfully")
    st.audio(uploaded_file)
    
    if st.button("🚀 CLEAN AUDIO NOW"):
        with st.spinner("Analyzing spectral baseline and filtering..."):
            processed_data = process_audio(uploaded_file)
            
            st.success("✨ Audio Restored!")
            
            # Display Result
            st.markdown("### 🎧 Result")
            st.audio(processed_data)
            
            # Download Button
            st.download_button(
                label="📥 DOWNLOAD CLEANED WAV",
                data=processed_data,
                file_name="sonic_clean_output.wav",
                mime="audio/wav"
            )

st.markdown("---")
st.caption("Tip: For best results, ensure the first 0.5s of the audio contains only background noise.")
