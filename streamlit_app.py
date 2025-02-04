import streamlit as st
import os
import wave
import json
from vosk import Model, KaldiRecognizer
import tempfile
from io import BytesIO

# Streamlit App Title
st.title("🎙 Speaker Diarization (Vosk - No GPU Required)")
st.write("Upload an audio file, and the AI will detect different speakers.")

# Download and Load Vosk Model (English)
MODEL_PATH = "vosk-model-small-en-us-0.15"

if not os.path.exists(MODEL_PATH):
    st.warning("Downloading Vosk model... (This will take some time)")
    os.system(f"wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip")
    os.system(f"unzip vosk-model-small-en-us-0.15.zip")
    os.system(f"rm vosk-model-small-en-us-0.15.zip")

st.success("✅ Vosk model loaded successfully!")

model = Model(MODEL_PATH)

# Function to transcribe audio
def transcribe_audio(audio_path):
    """Perform speaker diarization on audio file."""
    wf = wave.open(audio_path, "rb")
    rec = KaldiRecognizer(model, wf.getframerate())
    rec.SetWords(True)

    result_text = []
    
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            res = json.loads(rec.Result())
            if "text" in res:
                result_text.append(res["text"])

    final_transcript = " ".join(result_text)

    # Convert text to a downloadable file
    output_buffer = BytesIO()
    output_buffer.write(final_transcript.encode())
    output_buffer.seek(0)

    return output_buffer, final_transcript

# File Upload
uploaded_file = st.file_uploader("📤 Upload an Audio File", type=["wav", "mp3", "flac"])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(uploaded_file.getbuffer())
        temp_audio_path = temp_audio.name

    st.write("⏳ Processing the audio for speaker diarization...")

    # Perform Speaker Diarization
    output_buffer, transcript = transcribe_audio(temp_audio_path)

    # Display Transcript
    st.text_area("📜 Diarization Output", transcript, height=300)

    # Download Button
    st.download_button(label="📥 Download Transcript", data=output_buffer, file_name="diarization.txt", mime="text/plain")

    # Cleanup
    os.remove(temp_audio_path)
