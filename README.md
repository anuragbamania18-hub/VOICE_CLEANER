

<div align="center">

# ✨ NOICE

### Fourier-transform audio denoising, in your browser

Upload a noisy recording, click a button, get cleaner audio back.
Built with **spectral subtraction** on the Short-Time Fourier Transform.

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)
![librosa](https://img.shields.io/badge/librosa-DSP-00d4ff)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![Course](https://img.shields.io/badge/Course-Signals%20%26%20Systems-3a7bd5)

</div>

---


## What it does

Recordings often carry steady background noise such as fan hum, hiss or room tone. This app removes it using ideas straight from Signals & Systems:

1. Move the audio into the **frequency domain** with the STFT.
2. **Estimate the noise spectrum** from the start of the clip.
3. **Subtract** that estimate from every time frame.
4. Rebuild the waveform with the inverse STFT.

Everything runs in a clean dark-themed Streamlit interface: upload, preview, process, listen, download.

## ✨ Features

- 🎵 Upload **WAV** or **MP3** files
- ▶️ Built-in audio players to compare the original and the cleaned result
- 🚀 One-click noise reduction
- 📥 Download the result as a **WAV** file
- 🎨 Custom dark UI with gradient buttons and a hidden Streamlit chrome
- ⚡ Fully vectorized with NumPy, no slow Python loops over frames

---

## How it works

The signal processing pipeline lives in `process_audio()`:

```
 audio file
     │
     ▼
 librosa.load(sr=None)           keep the original sample rate, convert to mono
     │
     ▼
 STFT  ───────►  X(t, f) = |X| · e^(jφ)
     │
     ▼
 magphase                         split into magnitude |X| and phase φ
     │
     ├── noise floor N(f) = mean of |X| over the first 15 frames
     │
     ▼
 spectral subtraction             |Y| = max( |X| − 1.5 · N(f), 0 )
     │
     ▼
 recombine with ORIGINAL phase    Y = |Y| · e^(jφ)
     │
     ▼
 inverse STFT  ───────►  cleaned waveform  ───►  WAV
```

### The math

The Short-Time Fourier Transform of the signal $x[n]$ is

$$
X(t, f) = \sum_{n} x[n]\, w[n - tH]\, e^{-j 2\pi f n / N}
$$

where $w$ is the analysis window, $H$ the hop size and $N$ the FFT size.

The noise magnitude spectrum is estimated as the average over the first $K = 15$ frames, which are assumed to contain only noise:

$$
\hat{N}(f) = \frac{1}{K} \sum_{t=0}^{K-1} |X(t, f)|
$$

Spectral subtraction with an over-subtraction factor $\alpha = 1.5$ and half-wave rectification (so magnitudes never go negative):

$$
|Y(t, f)| = \max\!\big(|X(t, f)| - \alpha\,\hat{N}(f),\; 0\big)
$$

The cleaned spectrogram reuses the noisy phase $\angle X(t, f)$, and the inverse STFT turns it back into a time-domain signal.

### Why these choices?

| Choice | Reason |
|---|---|
| `sr=None` | Keeps the file's native sample rate, so no resampling artifacts |
| Over-subtraction (`α = 1.5`) | Removes noise more aggressively than plain subtraction (`α = 1`) |
| `np.maximum(..., 0)` | Magnitudes cannot be negative, so negative results are clipped to zero |
| Reusing the noisy phase | Human hearing is far less sensitive to phase errors than to magnitude errors |
| Matrix operations | The whole spectrogram is processed in one NumPy expression, which is fast |

---

## Getting started

### Prerequisites

- Python 3.9 or newer
- `pip`
- For MP3 support on some systems, [FFmpeg](https://ffmpeg.org/download.html) may be needed

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>

# 2. (Recommended) create a virtual environment
python -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Run the app

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints (usually http://localhost:8501).

---

## Usage

1. **Upload** a `.wav` or `.mp3` file.
2. **Listen** to the original using the preview player.
3. Click **🚀 CLEAN AUDIO NOW**.
4. **Compare** the cleaned audio in the result player.
5. Click **📥 DOWNLOAD CLEANED WAV** to save it as `sonic_clean_output.wav`.

> **Tip:** the noise profile is learned from the very beginning of the file, so start your recording with a moment of silence (background noise only) before any speech or music.

---

## Limitations

Spectral subtraction is a classic, simple technique, and it comes with trade-offs:

- **Assumes stationary noise.** It works best on steady sounds such as hum, hiss and fan noise. Sudden or changing noise (traffic, chatter, keyboard clicks) is not handled well.
- **Needs a noise-only start.** If speech or music begins right away, it ends up in the noise estimate and gets partly removed.
- **"Musical noise" artifacts.** Over-subtraction can leave faint, watery, tone-like residue in the output.
- **Short noise window.** The profile uses the first 15 STFT frames. With librosa's default hop of 512 samples, that is about 0.17 s at 44.1 kHz and about 0.35 s at 22.05 kHz.
- **Mono output.** `librosa.load` mixes the input down to mono by default.

## Ideas for future work

- [ ] Let the user choose the noise-profile length and the over-subtraction factor with sliders
- [ ] Plot the spectrogram before and after cleaning
- [ ] Add a spectral floor to reduce musical noise
- [ ] Try Wiener filtering or minimum-statistics noise tracking
- [ ] Keep stereo channels
- [ ] Show SNR or other quality metrics

---

## Project structure

```
.
├── app.py              # Streamlit app + denoising logic
├── requirements.txt    # Python dependencies
├── .gitignore
└── README.md
```

## Built with

- [Streamlit](https://streamlit.io/) for the web interface
- [librosa](https://librosa.org/) for audio loading, STFT and inverse STFT
- [NumPy](https://numpy.org/) for the spectral math
- [SoundFile](https://python-soundfile.readthedocs.io/) for writing WAV output

## Concepts from the course

This project applies these Signals & Systems topics:

- Fourier transform and the frequency-domain view of signals
- Short-Time Fourier Transform and spectrograms
- Magnitude and phase of a spectrum
- Windowing and overlap-add reconstruction
- Noise estimation and filtering in the frequency domain

---


