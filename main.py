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

def process_audio(input_audio):
    y, sr = librosa.load(input_audio, sr=None)
    
    # 1. Pre-emphasis: Boosts high frequencies (where speech clarity lives)
    y_filt = librosa.effects.preemphasis(y)
    
    stft = librosa.stft(y_filt, n_fft=2048, hop_length=512)
    magnitude, phase = librosa.magphase(stft)
    power = magnitude**2

    # 2. Adaptive Noise Estimation (Median of the power spectrum)
    # This identifies the constant 'wind' energy
    noise_est = np.median(power, axis=1, keepdims=True)

    # 3. WIENER FILTER CALCULATION
    # SNR = (Signal Power / Noise Power)
    # Gain = SNR / (SNR + 1)
    # This is a 'soft' transition that sounds much more natural
    snr = power / (noise_est + 1e-10) # 1e-10 prevents division by zero
    
    # We apply a 'strength' factor to the noise to be more aggressive
    alpha = 3.0 
    gain = snr / (snr + alpha)
    
    # 4. Smoothing the Gain Mask
    # This is the "something extra" that prevents the choppy sound
    gain = scipy.ndimage.gaussian_filter(gain, sigma=(1, 2))

    # Apply gain and convert back to magnitude
    clean_mag = np.sqrt(power * gain)

    # 5. Reconstruction
    y_clean = librosa.istft(clean_mag * phase, hop_length=512)
    
    # Final 'naturalness' blend: Mix 10% of the original back in 
    # to avoid that "empty space" feeling.
    y_final = (0.9 * y_clean) + (0.1 * y[:len(y_clean)])
    
    y_final = librosa.util.normalize(y_final)

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
