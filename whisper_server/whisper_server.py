from flask import Flask, request, jsonify
from faster_whisper import WhisperModel

app = Flask(__name__)

print("Loading Whisper...")

model = WhisperModel(
    "tiny.en",
    device="cpu",
    compute_type="int8"
)

print("Whisper Ready!")

@app.route("/transcribe", methods=["POST"])

def transcribe():

    data = request.json

    audio_path = data["audio_path"]

    segments, info = model.transcribe(
        audio_path,
        beam_size=1
    )

    text = ""

    for segment in segments:

        text += segment.text

    return jsonify({
        "text": text.strip()
    })

app.run(
    host="127.0.0.1",
    port=5002
)

