import whisper

model = whisper.load_model("base")

# オーディオをロードし、30 秒に収まるようにパッド/トリムします
audio = whisper.load_audio("/Users/hase_syo/Desktop/audio_test/audio.mp3")
audio = whisper.pad_or_trim(audio)

# log-Mel スペクトログラムを作成し、モデルと同じデバイスに移動します

mel = whisper.log_mel_spectrogram(audio).to(model.device)

# 話し言葉を検出する
_, probs = model.detect_language(mel)
print(f"Detected language: {max(probs, key=probs.get)}")
print("文字起こし中、、、")

# 音声をデコードする
options = whisper.DecodingOptions()
result = whisper.decode(model, mel, options)

# 認識されたテキストを印刷する
print(result.text)