import streamlit as st
import numpy as np
import librosa
from scipy import signal

def analyze_timbre(audio):
    _, _, spectrogram = signal.stft(audio)
    return {"spectrogram": np.abs(spectrogram)}

def design_new_timbre(original_timbre):
    reversed_spectrogram = np.flipud(original_timbre["spectrogram"])
    return {"spectrogram": reversed_spectrogram}

def apply_new_timbre(audio, new_timbre, original_sr):
    audio = librosa.effects.preemphasis(audio)
    _, inverse_transform = signal.istft(new_timbre["spectrogram"])
    transformed_audio = 2 * audio
    return transformed_audio

def main():
    st.title("UPSIDEDOWN")

    # 为 file_uploader 指定唯一的 key 参数
    audio_file = st.file_uploader("上传音频文件", type=["wav", "mp3"], key="unique_key")

    if audio_file is not None:
        st.audio(audio_file, format="audio")

        audio, sr = librosa.load(audio_file, sr=None)

        original_timbre = analyze_timbre(audio)
        new_timbre = design_new_timbre(original_timbre)
        transformed_audio = apply_new_timbre(audio, new_timbre, sr)

        st.audio(transformed_audio, format="audio/wav", sample_rate=sr)
    else:
        st.warning("请上传音频文件")

if __name__ == "__main__":
    main()
