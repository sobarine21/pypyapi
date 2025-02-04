import streamlit as st
from pyannote.audio import Pipeline
from pyannote.core import Segment
from huggingface_hub import login
from io import BytesIO
import os

# Set Hugging Face API token
HUGGINGFACE_TOKEN = "your_huggingface_token"  # Replace with your actual token

# Authenticate with Hugging Face
login(HUGGINGFACE_TOKEN)

# Load the pyannote speaker diarization pipeline
try:
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=HUGGINGFACE_TOKEN)
    if pipeline is None:
        st.error("Failed to load the diarization pipeline. Check your Hugging Face token and model access.")
except Exception as e:
    st.error(f"Error loading pipeline: {e}")
    pipeline = None

# Function to perform speaker diarization
def diarize_audio(audio_path):
    """Run speaker diarization and save results."""
    diarization = pipeline(audio_path)

    # Format output as text
    output_text = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        output_text.append(f"{turn.start:.2f} - {turn.end:.2f} | Speaker {speaker}")

    result_text = "\n".join(output_text)

    # Convert text to a downloadable file
    output_buffer = BytesIO()
    output_buffer.write(result_text.encode())
    output_buffer.seek(0)

    return output_buffer, result_text

# Streamlit UI
st.title("Speaker Diarization with pyannote.audio")
st.write("Upload an audio file to identify different speakers and generate a transcript.")

# File uploader
uploaded_file = st.file_uploader("Upload Audio File", type=["wav", "mp3", "flac"])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    temp_audio_path = "temp_audio.wav"
    with open(temp_audio_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Process the audio file
    st.write("Processing the audio for speaker diarization...")
    if pipeline:
        output_buffer, transcript = diarize_audio(temp_audio_path)

        # Display the transcript
        st.text_area("Diarization Output", transcript, height=300)

        # Download button
        st.download_button(label="Download Transcript", data=output_buffer, file_name="diarization.txt", mime="text/plain")

        # Clean up
        os.remove(temp_audio_path)
    else:
        st.error("Diarization pipeline is not loaded. Check your Hugging Face authentication.")
