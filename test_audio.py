import whisper
import os

def test_whisper():
    try:
        print("Loading Whisper model...")
        model = whisper.load_model("base")
        print("Whisper model loaded successfully!")
        
        # Test with a sample audio file if available
        test_file = "sad_audio_Uvuylb4.wav"
        if os.path.exists(test_file):
            print(f"Testing with {test_file}...")
            result = model.transcribe(test_file)
            print(f"Transcription: {result['text']}")
        else:
            print("No test audio file found. Creating a simple test...")
            # Create a simple test
            result = model.transcribe("sad_audio_Uvuylb4.wav")  # This should fail but show the error
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_whisper()