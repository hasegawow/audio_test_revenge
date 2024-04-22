import whisper
import cv2

model = whisper.load_model("base")
result = model.transcribe("/Users/hase_syo/Desktop/audio_test/audio.mp3")

print(result["text"])