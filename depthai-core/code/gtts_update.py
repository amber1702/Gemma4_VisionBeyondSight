#!/usr/bin/env python3

import cv2
import depthai as dai
import requests
import base64
import Jetson.GPIO as GPIO
import time
import os

from gtts import gTTS

# =========================================
# GPIO BUTTON CONFIG
# =========================================

BUTTON_SCENE = 31
BUTTON_OCR = 29
BUTTON_MONEY = 32

GPIO.setmode(GPIO.BOARD)

GPIO.setup(BUTTON_SCENE, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(BUTTON_OCR, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(BUTTON_MONEY, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# =========================================
# GEMMA SERVER CONFIG
# =========================================

SERVER_URL = "http://127.0.0.1:8080/v1/chat/completions"

# =========================================
# PROMPTS
# =========================================

SCENE_PROMPT = (
    "Describe shortly the images (danger, people, enviroment,..) in 1 sentence only. "
)

OCR_PROMPT = (
    "Read all visible text in the image only."
)

MONEY_PROMPT = (
    "Identify the Vietnamese currency denomination in the image.If not return no money recognition\n"
    "Possible values are:\n"
    "1000 VND,\n"
    "2000 VND,\n"
    "5000 VND,\n"
    "10000 VND,\n"
    "20000 VND,\n"
    "50000 VND,\n"
    "100000 VND,\n"
    "200000 VND,\n"
    "500000 VND.\n\n"
    "Return only the denomination value."
)

# =========================================
# IMAGE ENCODER
# =========================================

def encode_image(image_path):

    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# =========================================
# GEMMA REQUEST
# =========================================

def ask_gemma(image_path, prompt):

    image_base64 = encode_image(image_path)

    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    }
                ]
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
# TEXT TO SPEECH
# =========================================

def speak_text(text):

    try:

        print("\n[Speaking Output...]")

        tts = gTTS(
            text=text,
            lang='en'
        )

        tts.save("output.mp3")

        os.system("mpg123 output.mp3")

    except Exception as e:

        print("TTS Error:", e)

# =========================================
# DEPTHAI CAMERA PIPELINE
# =========================================

print("Starting camera...")

device = dai.Device()

with dai.Pipeline(device) as pipeline:

    outputQueues = {}

    sockets = device.getConnectedCameras()

    for socket in sockets:

        cam = pipeline.create(dai.node.Camera).build(socket)

        outputQueues[str(socket)] = cam.requestOutput(
            (640, 480),
            dai.ImgFrame.Type.BGR888p,
            dai.ImgResizeMode.LETTERBOX,
            30
        ).createOutputQueue()

    pipeline.start()

    print("\n===================================")
    print("Assistive AI System Ready!")
    print("===================================")
    print("Button 1 -> Scene Description")
    print("Button 2 -> OCR")
    print("Button 3 -> Vietnamese Currency")
    print("Press Q to quit")
    print("===================================\n")

    while pipeline.isRunning():

        frames = {}

        # =========================================
        # GET CAMERA FRAMES
        # =========================================

        for name, queue in outputQueues.items():

            videoIn = queue.get()

            frame = videoIn.getCvFrame()

            frames[name] = frame

            cv2.imshow(name, frame)

        # Use first connected camera
        first_cam = list(frames.keys())[0]

        frame = frames[first_cam]

        # =========================================
        # BUTTON 1 - SCENE DESCRIPTION
        # =========================================

        if GPIO.input(BUTTON_SCENE) == GPIO.LOW:

            print("\n[SCENE BUTTON PRESSED]")

            image_path = "scene.jpg"

            cv2.imwrite(image_path, frame)

            print("Capturing image...")
            print("Sending to Gemma4...")

            try:

                output = ask_gemma(
                    image_path,
                    SCENE_PROMPT
                )

                print("\n========== SCENE OUTPUT ==========\n")

                print(output)
                print()

                speak_text(output)

            except Exception as e:

                print("Gemma Error:", e)

            time.sleep(1)

        # =========================================
        # BUTTON 2 - OCR
        # =========================================

        elif GPIO.input(BUTTON_OCR) == GPIO.LOW:

            print("\n[OCR BUTTON PRESSED]")

            image_path = "ocr.jpg"

            cv2.imwrite(image_path, frame)

            print("Capturing image...")
            print("Sending to Gemma4...")

            try:

                output = ask_gemma(
                    image_path,
                    OCR_PROMPT
                )

                print("\n========== OCR OUTPUT ==========\n")

                print(output)
                print()

                speak_text(output)

            except Exception as e:

                print("Gemma Error:", e)

            time.sleep(1)

        # =========================================
        # BUTTON 3 - MONEY
        # =========================================

        elif GPIO.input(BUTTON_MONEY) == GPIO.LOW:

            print("\n[MONEY BUTTON PRESSED]")

            image_path = "money.jpg"

            cv2.imwrite(image_path, frame)

            print("Capturing image...")
            print("Sending to Gemma4...")

            try:

                output = ask_gemma(
                    image_path,
                    MONEY_PROMPT
                )

                print("\n========== MONEY OUTPUT ==========\n")

                print(output)
                print()

                speak_text(output)

            except Exception as e:

                print("Gemma Error:", e)

            time.sleep(1)

        # =========================================
        # QUIT
        # =========================================

        key = cv2.waitKey(1)

        if key == ord("q"):
            break

# =========================================
# CLEANUP
# =========================================

GPIO.cleanup()

cv2.destroyAllWindows()
