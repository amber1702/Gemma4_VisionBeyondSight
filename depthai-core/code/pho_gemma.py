
#!/usr/bin/env python3

import time
import requests
import sounddevice as sd
import numpy as np
import subprocess

from scipy.io.wavfile import write

import Jetson.GPIO as GPIO

# =========================================
# BUTTON CONFIG
# =========================================

BUTTON_SCENE = 31

GPIO.setmode(GPIO.BOARD)

GPIO.setup(
    BUTTON_SCENE,
    GPIO.IN,
    pull_up_down=GPIO.PUD_UP
)

# =========================================
# HOLD CONFIG
# =========================================

HOLD_TIME = 1.0

# =========================================
# GEMMA SERVER
# =========================================

SERVER_URL = "http://127.0.0.1:8080/v1/chat/completions"

# =========================================
# PIPER MODEL
# =========================================

VI_MODEL = "/home/amber/gemma/depthai-core/code/piper_models/vi_VN-vais1000-medium.onnx"


# =========================================
# PIPER TTS
# =========================================

import pygame

pygame.mixer.init()

def speak_text(text):

    print("\n========== SPEAK ==========\n")

    print(text)

    temp_wav = "/tmp/output.wav"

    # =====================================
    # GENERATE WAV
    # =====================================

    cmd = f'''
    echo "{text}" | \
    piper \
    --model "{VI_MODEL}" \
    --length_scale 1.35 \
    --sentence_silence 0.5 \
    --output_file "{temp_wav}"
    '''

    subprocess.run(
        cmd,
        shell=True
    )

    # =====================================
    # PLAY AUDIO
    # =====================================

    pygame.mixer.music.load(
        temp_wav
    )

    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():

        time.sleep(0.05)



# =========================================
# PHOWHISPER STT
# =========================================

def phowhisper_stt(audio_path):

    response = requests.post(
        "http://127.0.0.1:5001/transcribe",
        json={
            "audio_path": audio_path
        }
    )

    result = response.json()

    return result["text"]

# =========================================
# GEMMA CHAT
# =========================================

def ask_gemma(user_text):

    payload = {
        "messages": [
            {
                "role": "system",
                "content": (
                    "Bạn là trợ lý AI cho người khiếm thị. "
                    "Trả lời ngắn gọn, rõ ràng."
                )
            },
            {
                "role": "user",
                "content": user_text
            }
        ],
        "max_tokens": 50,
        "temperature": 0.2
    }

    response = requests.post(
        SERVER_URL,
        json=payload
    )

    result = response.json()

    return result["choices"][0]["message"]["content"]

# =========================================
# MAIN
# =========================================

print("\n===================================")
print("Assistive Voice Assistant")
print("===================================")
print("Short press -> Scene description")
print("Hold > 1 second -> Voice mode")
print("Release button -> Stop recording")
print("===================================\n")

try:

    while True:

        # =====================================
        # BUTTON PRESSED
        # =====================================

        if GPIO.input(BUTTON_SCENE) == GPIO.LOW:

            press_time = time.time()

            # =================================
            # WAIT WHILE BUTTON HELD
            # =================================

            while GPIO.input(BUTTON_SCENE) == GPIO.LOW:

                held_time = (
                    time.time() - press_time
                )

                # =============================
                # VOICE MODE
                # =============================

                if held_time >= HOLD_TIME:

                    print("\n[VOICE MODE]")

                    fs = 16000

                    audio_chunks = []

                    stream = sd.InputStream(
                        samplerate=fs,
                        channels=1,
                        dtype='int16',
                        blocksize=1024
                    )

                    stream.start()

                    print("Recording...")

                    # =========================
                    # RECORD WHILE HOLDING
                    # =========================

                    while GPIO.input(BUTTON_SCENE) == GPIO.LOW:

                        data, overflowed = (
                            stream.read(1024)
                        )

                        audio_chunks.append(data)

                    stream.stop()
                    stream.close()

                    print(
                        "Recording finished"
                    )

                    # =========================
                    # EMPTY AUDIO
                    # =========================

                    if len(audio_chunks) == 0:

                        print(
                            "No audio recorded"
                        )

                        break

                    # =========================
                    # SAVE AUDIO
                    # =========================

                    audio = np.concatenate(
                        audio_chunks,
                        axis=0
                    )

                    audio_path = (
                        "/tmp/voice.wav"
                    )

                    write(
                        audio_path,
                        fs,
                        audio
                    )

                    print(
                        "\nSending to PhoWhisper..."
                    )

                    try:

                        # =====================
                        # STT
                        # =====================

                        user_text = (
                            phowhisper_stt(
                                audio_path
                            )
                        )

                        print(
                            "\n========== USER ==========\n"
                        )

                        print(user_text)

                        # =====================
                        # EMPTY TEXT
                        # =====================

                        if len(
                            user_text.strip()
                        ) == 0:

                            print(
                                "Empty transcription"
                            )

                            break

                        # =====================
                        # GEMMA
                        # =====================

                        answer = ask_gemma(
                            user_text
                        )

                        print(
                            "\n========== GEMMA ==========\n"
                        )

                        print(answer)

                        # =====================
                        # SMALL PAUSE
                        # =====================

                        time.sleep(0.5)

                        # =====================
                        # TTS
                        # =====================

                        speak_text(answer)

                    except Exception as e:

                        print("Error:", e)

                    time.sleep(0.5)

                    break

            # =================================
            # SHORT PRESS -> SCENE
            # =================================

            total_press = (
                time.time() - press_time
            )

            if total_press < HOLD_TIME:

                print("\n[SCENE MODE]")

                # =================================
                # ADD YOUR SCENE FUNCTION HERE
                # =================================

                print("Scene description")

            time.sleep(0.3)

except KeyboardInterrupt:

    print("\nExiting...")

finally:

    GPIO.cleanup()


