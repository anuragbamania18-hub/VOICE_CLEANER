import streamlit as st
import librosa
import numpy as np
import soundfile as sf
import io
import scipy.ndimage  # Required for the rolling window math

# --- CUSTOM STYLING (CSS) ---
def style_app():
    st.markdown("""
        <style>
        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        footer {visibility: hidden;}
        
        .stApp {
            background-color: #0e1117;
            color: #ffffff;
        }

        .block-container {
            padding-top: 3rem;
        }
        
        .main-header {
            font-size: 2.5rem;
            font-weight: 800;
            color: #00d4ff;
            text-align: center;
            margin-bottom: 0.5rem;
            text-shadow: 0px 4px 10px rgba(0, 212, 255, 0.3);
        }

        .sub-text {
            text-align: center;
            color: #94a3b8;
            margin-bottom: 2rem;
        }

        div.stButton > button:first-child {
            background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%);
            color: white;
            border: none;
            padding: 0.6rem 2rem;
            border-radius: 50px;
            font-weight: bold;
            width: 100%;
            transition: all 0.3s ease;
        }
        </style>
    """, unsafe_allow_html=True)

# --- ADVANCED PROCESSING LOGIC ---
def process_audio(input_audio):
    # Load audio
    y, sr = librosa.load(input_audio, sr=None)
    
    # Fast Fourier Transform
    stft = librosa.stft(y)
    magnitude, phase = librosa.magphase(stft)

    # 1. ROLLING WINDOW NOISE ESTIMATION
    # Instead of just the first 15 frames, we look at a window of ~1 second (size=43)
    # The 'minimum_filter' finds the lowest energy (noise) sitting under the voice peaks.
    noise_est = scipy.ndimage.minimum_filter(magnitude, size=(1, 43))

    # 2. TEMPORAL SMOOTHING
    # This prevents the 'pumping' effect. It ensures the noise reduction 
    # transitions smoothly between noisy and clean parts.
    noise_est = scipy.ndimage.gaussian_filter(noise_est, sigma=(0, 2))

    # 3. SPECTRAL SUBTRACTION WITH GAIN FLOOR
    # Over-subtraction factor = 1.5 (removes hiss aggressively)
    # Gain Floor (0.15) = Your 'amplification' safety net. 
    # It ensures we never subtract 100% of the signal, keeping it natural.
    clean_mag = np.maximum(magnitude - (noise_est * 1.5), 0.15 * magnitude)

    # Reconstruct the audio
    y_clean = librosa.istft(clean_mag * phase)

    buffer = io.BytesIO()
    sf.write(buffer, y_clean, sr, format='WAV')
    buffer.seek(0)
    return buffer

# --- APP LAYOUT ---
st.set_page_config(page_title="NOICE", page_icon="✨", layout="centered")
style_app()

st.markdown('<h1 class="main-header"> NOISE REDUCE </h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">Adaptive Rolling-Window Fourier Analysis</p>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type=["wav", "mp3"])

if uploaded_file is not None:
    st.info("🎵 File Loaded Successfully")
    st.audio(uploaded_file)

    if st.button("🚀 CLEAN AUDIO NOW"):
        with st.spinner("Tracking adaptive noise floor and applying filters..."):
            processed_data = process_audio(uploaded_file)

            st.success("✨ Audio Restored!")

            st.markdown("### 🎧 Result")
            st.audio(processed_data)

            st.download_button(
                label="📥 DOWNLOAD CLEANED WAV",
                data=processed_data,
                file_name="cleaned_voice_output.wav",
                mime="audio/wav"
            )

st.markdown("---")
st.caption("New Feature: This version tracks background noise throughout the whole file and adapts automatically.")
