import os
from transformers import pipeline

# Define the languages of interest and their corresponding codes
languages = {
    "Hindi": "hi",
    "Indian English": "en",
    "Urdu": "ur",
    "Bengali": "bn",
    "Punjabi": "pa",
}

# Initialize the ASR and MT pipelines
asr_pipeline = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h")
mt_pipeline = pipeline("translation", model="Helsinki-NLP/opus-mt-en-IN")

# Function to transcribe and translate audio files
def transcribe_and_translate(audio_folder):
    for root, dirs, files in os.walk(audio_folder):
        for filename in files:
            if filename.endswith(".wav"):
                audio_path = os.path.join(root, filename)
                language = filename.split("_")[0]  # Extract language from filename
                
                # Check if the language is in the list of languages of interest
                if language in languages:
                    print(f"Transcribing and translating {filename}...")
                    
                    # Transcribe the audio
                    transcription = asr_pipeline(audio_path)
                    
                    # Translate to English
                    translated_text = mt_pipeline(transcription[0]["sentence"], src=languages[language], tgt="en")
                    
                    print(f"Transcription ({language}): {transcription[0]['sentence']}")
                    print(f"Translation to English: {translated_text[0]['translation_text']}\n")

if __name__ == "__main__":
    # Replace 'audio_folder' with the path to the folder containing audio files
    audio_folder = "C:/Users/Suraj kumar sahoo/OneDrive/Desktop/audios_folder"

    transcribe_and_translate(audio_folder)
