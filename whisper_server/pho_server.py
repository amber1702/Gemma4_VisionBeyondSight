from flask import Flask, request, jsonify
from transformers import pipeline
import os

# =====================================
# OFFLINE MODE
# =====================================

os.environ["TRANSFORMERS_OFFLINE"] = "1"

# =====================================
# LOAD MODEL
# =====================================

print("Loading PhoWhisper...")

pipe = pipeline(
    "automatic-speech-recognition",
    model="vinai/PhoWhisper-small",
    device=0
)

print("PhoWhisper Ready!")

# =====================================
# FLASK
# =====================================

app = Flask(__name__)

# =====================================
# API
# =====================================

@app.route("/transcribe", methods=["POST"])
def transcribe():

    data = request.json

    audio_path = data["audio_path"]

    result = pipe(audio_path)

    text = result["text"]

    return jsonify({
        "text": text
    })

# =====================================
# MAIN
# =====================================

app.run(
    host="127.0.0.1",
    port=5001
)

