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
# ----PROCESSING LOGIC----
def process_audio(input_audio):
    y, sr = librosa.load(input_audio, sr=None)
    
    # 1. High-pass filter: Wind is mostly sub-100Hz rumble. 
    # Let's kill that first.
    y = librosa.effects.preemphasis(y)
    
    stft = librosa.stft(y, n_fft=2048, hop_length=512)
    magnitude, phase = librosa.magphase(stft)

    # 2. MORPHOLOGICAL FILTERING
    # This is the "secret sauce." 
    # Wind is a "cloud" in the spectrogram. Speech is "lines."
    # We use a median filter across the frequency axis to find the "cloud" (noise).
    # Then we subtract it.
    
    # Estimate the "cloud" (noise) by blurring across frequencies
    noise_floor = scipy.ndimage.median_filter(magnitude, size=(20, 1)) 
    
    # 3. ADVANCED SUBTRACTION
    # We use a high over-subtraction (4.0) because that wind is brutal.
    clean_mag = np.maximum(magnitude - (noise_floor * 4.0), 0.05 * magnitude)

    # 4. HARMONIC ENHANCEMENT
    # We use Librosa's decompose to separate percussive (wind hits) 
    # from harmonic (vocals). We prioritize the harmonics.
    harmonic, percussive = librosa.decompose.hpss(clean_mag)
    
    # Final Result is mostly Harmonic + a tiny bit of Percussive for naturalness
    final_mag = harmonic + (0.1 * percussive)

    # 5. RECONSTRUCT
    y_clean = librosa.istft(final_mag * phase, hop_length=512)
    y_final = librosa.util.normalize(y_clean)

    buffer = io.BytesIO()
    sf.write(buffer, y_final, sr, format='WAV')
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
