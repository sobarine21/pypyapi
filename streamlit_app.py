import streamlit as st
from pyannote.audio import Pipeline
from huggingface_hub import login, HfApi
from io import BytesIO
import os

# Streamlit App Title
st.title("🎙 Speaker Diarization with pyannote.audio")
st.write("Upload an audio file, and the AI will detect different speakers.")

# Load Hugging Face Token from Streamlit Secrets
HUGGINGFACE_TOKEN = st.secrets.get("HUGGINGFACE_TOKEN")

if not HUGGINGFACE_TOKEN:
    st.error("❌ Hugging Face Token is missing! Add it to Streamlit Secrets.")
    st.stop()

# Authenticate with Hugging Face
try:
    login(HUGGINGFACE_TOKEN)
    st.success("✅ Successfully authenticated with Hugging Face!")

    # Test API access
    api = HfApi()
    user_info = api.whoami(token=HUGGINGFACE_TOKEN)
    st.write(f"🔑 Logged in as: **{user_info['name']}**")

except Exception as e:
    st.error(f"❌ Authentication failed: {e}")
    st.stop()

# Load the Speaker Diarization Pipeline
try:
    st.write("Attempting to load the model...")
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=HUGGINGFACE_TOKEN)
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.exception(e)  # Display the full traceback for debugging
    st.stop()

# Function to perform Speaker Diarization
def diarize_audio(audio_path):
    """Run speaker diarization and save results."""
    try:
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
    except Exception as e:
        st.error(f"❌ Diarization failed: {e}")
        st.exception(e)  # Show traceback
        return None, None  # Return None values to handle the error

# File Upload
uploaded_file = st.file_uploader("📤 Upload an Audio File", type=["wav", "mp3", "flac"])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    temp_audio_path = "temp_audio.wav"
    with open(temp_audio_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.write("⏳ Processing the audio for speaker diarization...")

    # Perform Speaker Diarization
    output_buffer, transcript = diarize_audio(temp_audio_path)

    if output_buffer and transcript:  # Check if both are not None
        # Display Transcript
        st.text_area("📜 Diarization Output", transcript, height=300)

        # Download Button
        st.download_button(label="📥 Download Transcript", data=output_buffer, file_name="diarization.txt", mime="text/plain")

    # Cleanup
    os.remove(temp_audio_path)
