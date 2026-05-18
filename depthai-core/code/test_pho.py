#!/usr/bin/env python3

import time
import requests
import sounddevice as sd
import numpy as np

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
# RECORD WHILE HOLDING BUTTON
# =========================================

def record_while_holding_button():

    fs = 16000

    audio_chunks = []

    print("\nRecording... Speak now")

    stream = sd.InputStream(
        samplerate=fs,
        channels=1,
        dtype='int16'
    )

    stream.start()

    # =====================================
    # RECORD WHILE BUTTON HELD
    # =====================================

    while GPIO.input(BUTTON_SCENE) == GPIO.LOW:

        data, overflowed = stream.read(1024)

        audio_chunks.append(data)

    stream.stop()
    stream.close()

    print("Recording finished")

    # =====================================
    # SAVE AUDIO
    # =====================================

    audio = np.concatenate(
        audio_chunks,
        axis=0
    )

    audio_path = "/tmp/voice.wav"

    write(
        audio_path,
        fs,
        audio
    )

    return audio_path

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
# MAIN
# =========================================

print("\n===================================")
print("PhoWhisper Hold-To-Talk Test")
print("===================================")
print("Hold button > 1 second")
print("Speak while holding")
print("Release button to stop")
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

                held_duration = (
                    time.time() - press_time
                )

                # =============================
                # HOLD DETECTED
                # =============================

                if held_duration >= HOLD_TIME:

                    print("\n[VOICE MODE]")

                    audio_path = (
                        record_while_holding_button()
                    )

                    print(
                        "\nSending to PhoWhisper..."
                    )

                    try:

                        text = phowhisper_stt(
                            audio_path
                        )

                        print(
                            "\n========== RESULT ==========\n"
                        )

                        print(text)

                    except Exception as e:

                        print(
                            "PhoWhisper Error:",
                            e
                        )

                    time.sleep(1)

                    break

            time.sleep(0.2)

except KeyboardInterrupt:

    print("\nExiting...")

finally:

    GPIO.cleanup()


