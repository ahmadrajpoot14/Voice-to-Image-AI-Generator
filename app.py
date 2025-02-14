import streamlit as st
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import requests
import openai
import tempfile
import os

# API Keys
DEEPGRAM_API_KEY = "your_deepgram_api_key"
OPENAI_API_KEY = "your_openai_api_key"
# Initialize OpenAI client
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Streamlit UI
st.title("🎙️ Voice to Image AI Generator")

# Step 1: Record Audio
fs = 44100  # Sample rate
duration = 5  # seconds

if st.button("🎤 Start Recording"):
    st.write("Recording... Speak now!")
    
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype=np.int16)
    sd.wait()
    
    # Save the recording to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        wav.write(temp_audio.name, fs, recording)
        audio_file = temp_audio.name
    
    st.success("Recording complete!")

    # Step 2: Transcribe Speech to Text using Deepgram
    st.write("Transcribing speech to text...")

    with open(audio_file, "rb") as audio:
        headers = {
            "Authorization": f"Token {DEEPGRAM_API_KEY}",
            "Content-Type": "audio/wav"
        }
        response = requests.post(
            "https://api.deepgram.com/v1/listen",
            headers=headers,
            data=audio
        )

    os.remove(audio_file)  # Clean up the temp file

    if response.status_code == 200:
        transcript_data = response.json()
        transcript = transcript_data["results"]["channels"][0]["alternatives"][0]["transcript"]
        st.success(f"Transcription: {transcript}")
    else:
        st.error("Error transcribing the audio.")
        st.stop()

    # Step 3: Generate Image using DALL·E directly from transcript
    st.write("Generating image with DALL·E...")

    dalle_response = client.images.generate(
        model="dall-e-3",
        prompt=transcript,  # Directly use the transcript as the prompt
        n=1,
        size="1024x1024"
    )

    image_url = dalle_response.data[0].url
    st.image(image_url, caption="Generated Image", use_column_width=True)

    st.success("Image generation complete!")
