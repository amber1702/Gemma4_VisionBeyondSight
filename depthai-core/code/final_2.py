#!/usr/bin/env python3

import os
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["ORT_LOGGING_LEVEL"] = "3"

import cv2
import depthai as dai
import requests
import base64
import Jetson.GPIO as GPIO
import time
import subprocess
import sounddevice as sd
import numpy as np
import pygame
import threading



from scipy.io.wavfile import write

# =========================================
# GPIO BUTTON CONFIG
# =========================================

BUTTON_SCENE = 31
BUTTON_OCR = 29
BUTTON_MONEY = 32
BUTTON_LANGUAGE = 33

GPIO.setmode(GPIO.BOARD)

GPIO.setup(BUTTON_SCENE, GPIO.IN)
GPIO.setup(BUTTON_OCR, GPIO.IN)
GPIO.setup(BUTTON_MONEY, GPIO.IN)
GPIO.setup(BUTTON_LANGUAGE, GPIO.IN)

# =========================================
# HOLD CONFIG
# =========================================

HOLD_TIME = 1.0

# =========================================
# LANGUAGE STATE
# =========================================

current_language = "en"

# =========================================
# GEMMA SERVER
# =========================================

SERVER_URL = "http://127.0.0.1:8080/v1/chat/completions"

# =========================================
# PIPER MODELS
# =========================================

EN_MODEL = "/home/amber/gemma/depthai-core/code/piper_models/en_US-lessac-low.onnx"

VI_MODEL = "/home/amber/gemma/depthai-core/code/piper_models/vi_VN-vais1000-medium.onnx"

# =========================================
# AUDIO INIT
# =========================================

pygame.mixer.init()

# =========================================
# PROMPTS
# =========================================

SCENE_PROMPT_EN = (
    "Describe shortly the image "
    "(danger, people, environment, obstacles, objects) "
    "in one short sentence only."
)

SCENE_PROMPT_VI = (
    "Hãy mô tả ngắn gọn hình ảnh (nguy hiểm, chướng ngại vật, người, nếu có) chỉ trong 1 câu."
)

OCR_PROMPT_EN = (
    "Read visible text only. "
    "If it is not in English, please translate it."
)

OCR_PROMPT_VI = (
    "Chỉ đọc văn bản hiển thị."
    "Nếu đây không phải là tiếng Việt, "
    "vui lòng dịch sang tiếng Việt."
)

MONEY_PROMPT_EN = (
        "Identify the Vietnamese currency denominations in the image.List all the value if it has.\n"
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
    "Return only the denomination values."
)

MONEY_PROMPT_VI = (
	    "Xác định các mệnh giá tiền tệ Việt Nam trong hình ảnh.Liệt kê tất cả các mệnh giá nếu có.\n"
	"Các mệnh giá có thể là:\n"
	"1000 VND,\n"
	"2000 VND,\n"
	"5000 VND,\n"
	"10000 VND,\n"
	"20000 VND,\n"
	"500000 VND,\n"
	"100000 VND,\n"
	"200000 VND,\n"
	"500000 VND.\n\n"
	"Chỉ trả về mệnh giá."
)

# =========================================
# DANGER DETECTION PROMPTS
# =========================================

DANGER_PROMPT_EN = (
	"Detect dangerous situations for blind users. "
	"Examples: stairs, holes, fire, gun, knife, obstacles. "
	"If safe, return ONLY: No. "
	"Otherwise return a short warning like: "
	"'Caution: stairs ahead"
)

DANGER_PROMPT_VI = (
	"Phát hiện tình huống nguy hiểm cho người khiếm thị. "
	"Ví dụ: cầu thang, hố,, lửa, dao, súng, chướng ngại vật. "
	"Nếu an toàn chỉ trả về: No. "
	"Nếu nguy hiểm hãy trả về cảnh báo ngắn như: "
	"'Cảnh báo: cầu thang phía trước'."
)



# =========================================
# IMAGE ENCODER
# =========================================

def encode_image(image_path):

    with open(image_path, "rb") as f:

        return base64.b64encode(
            f.read()
        ).decode("utf-8")

# =========================================
# GEMMA VLM
# =========================================

def ask_gemma(image_path, prompt):

    image_base64 = encode_image(
        image_path
    )

    # =====================================
    # DYNAMIC MAX TOKENS
    # =====================================

    if "money" in prompt.lower():

        max_tokens = 15

    elif "Read visible text" in prompt:

        max_tokens = 35
    
    elif "dangerous" in prompt.lower():
    
        max_tokens = 8

    else:

        max_tokens = 25

    payload = {
        "messages": [
            {
                "role": "system",
                "content": (
                    "Short assistive responses only."
                )
            },
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
                            "url":
                            f"data:image/jpeg;base64,{image_base64}"
                        }
                    }
                ]
            }
        ],

        "max_tokens": max_tokens,
        "temperature": 0.1,
        "top_k": 20,
        "top_p": 0.8
    }

    response = requests.post(
        SERVER_URL,
        json=payload
    )

    result = response.json()

    return result["choices"][0]["message"]["content"]

# =========================================
# GEMMA VOICE + VISION
# =========================================

def ask_gemma_voice_vision(
    image_path,
    user_text
):

    image_base64 = encode_image(
        image_path
    )

    if current_language == "vi":

        system_prompt = (
            "Bạn là trợ lý AI hỗ trợ "
            "người khiếm thị. "
            "Trả lời ngắn gọn."
        )

    else:

        system_prompt = (
            "You are an assistive AI "
            "for blind users. "
            "Answer shortly."
        )

    payload = {
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_text
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url":
                            f"data:image/jpeg;base64,{image_base64}"
                        }
                    }
                ]
            }
        ],

        "max_tokens": 30,
        "temperature": 0.2,
        "top_k": 20,
        "top_p": 0.8
    }

    response = requests.post(
        SERVER_URL,
        json=payload
    )

    result = response.json()

    return result["choices"][0]["message"]["content"]

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
# WHISPER SERVER STT
# =========================================

def whisper_stt(audio_path):

    response = requests.post(
        "http://127.0.0.1:5002/transcribe",
        json={
            "audio_path": audio_path
        }
    )

    result = response.json()

    return result["text"]

# =========================================
# PIPER TTS
# =========================================

def speak_text(text):

    global current_language

    try:

        print("\n========== SPEAK ==========\n")
        print(text)

        temp_wav = "/tmp/output.wav"

        if current_language == "vi":

            model_path = VI_MODEL

        else:

            model_path = EN_MODEL

        cmd = f'''
        echo "{text}" | \
        piper \
        --model "{model_path}" \
        --length_scale 1.15 \
        --sentence_silence 0.4 \
        --output_file "{temp_wav}"
        '''

        subprocess.run(
            cmd,
            shell=True
        )

        pygame.mixer.music.load(
            temp_wav
        )

        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():

            time.sleep(0.05)

    except Exception as e:

        print("TTS Error:", e)


# =========================================
# DANGER DETECTION LOOP
# =========================================

latest_frame = None

def danger_detection_loop():

    global latest_frame

    last_warning = ""

    while True:

        try:

            # =============================
            # CHECK EVERY 5 SECONDS
            # =============================

            time.sleep(3)

            if latest_frame is None:

                continue

            image_path = "/tmp/danger.jpg"

            small_frame = cv2.resize(
                latest_frame,
                (320, 240)
            )

            cv2.imwrite(
                image_path,
                small_frame
            )

            # =============================
            # LANGUAGE PROMPT
            # =============================

            if current_language == "vi":

                prompt = DANGER_PROMPT_VI

            else:

                prompt = DANGER_PROMPT_EN

            output = ask_gemma(
                image_path,
                prompt
            )

            output = output.strip()

            print("\n[DANGER CHECK]")
            print(output)

            # =============================
            # ONLY SPEAK IF DANGER
            # =============================

            clean_output = output.strip().lower()
            
            if clean_output not in ["no", "no.", "safe", "safe."]:
            
	

                # avoid repeating same warning
                if output != last_warning:

                    speak_text(output)

                    last_warning = output

            else:

                last_warning = ""

        except Exception as e:

            print("Danger Detection Error:", e)

# =========================================
# START CAMERA
# =========================================

print("Starting camera...")

device = dai.Device()

with dai.Pipeline(device) as pipeline:

    outputQueues = {}

    sockets = device.getConnectedCameras()

    for socket in sockets:

        cam = pipeline.create(
            dai.node.Camera
        )

        cam.initialControl.setAutoFocusMode(
            dai.CameraControl.AutoFocusMode.CONTINUOUS_VIDEO
        )

        cam = cam.build(socket)

        outputQueues[str(socket)] = (
            cam.requestOutput(
                (640, 480),
                dai.ImgFrame.Type.BGR888p,
                dai.ImgResizeMode.LETTERBOX,
                30
            ).createOutputQueue()
        )

    pipeline.start()

    # =====================================
    # START DANGER DETECTION THREAD
    # =====================================

    danger_thread = threading.Thread(
        target=danger_detection_loop,
        daemon=True
    )

    danger_thread.start()



    print("\n===================================")
    print("Assistive AI System Ready!")
    print("===================================")
    print("Short Press Button 1 -> Scene")
    print("Hold Button 1 -> Voice Assistant")
    print("Button 2 -> OCR")
    print("Button 3 -> Money")
    print("Button 4 -> Language")
    print("===================================\n")

    while pipeline.isRunning():

        frames = {}

        for name, queue in outputQueues.items():

            videoIn = queue.get()

            frame = videoIn.getCvFrame()

            frames[name] = frame

            cv2.imshow(name, frame)

        first_cam = list(frames.keys())[0]

        frame = frames[first_cam]
        latest_frame = frame.copy()

        # =====================================
        # BUTTON 1
        # =====================================

        if GPIO.input(BUTTON_SCENE) == GPIO.LOW:

            press_time = time.time()

            # =================================
            # WAIT PRESS / HOLD
            # =================================

            while GPIO.input(BUTTON_SCENE) == GPIO.LOW:

                held_time = (
                    time.time() - press_time
                )

                # =============================
                # HOLD -> VOICE MODE
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

                    print("Recording finished")

                    if len(audio_chunks) == 0:

                        print("No audio recorded")

                        break

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

                    # =========================
                    # SAVE SMALL IMAGE
                    # =========================

                    voice_image = (
                        "/tmp/voice_scene.jpg"
                    )

                    small_frame = cv2.resize(
                        frame,
                        (320, 240)
                    )

                    cv2.imwrite(
                        voice_image,
                        small_frame
                    )

                    try:

                        if current_language == "vi":

                            user_text = (
                                phowhisper_stt(
                                    audio_path
                                )
                            )

                        else:

                            user_text = (
                                whisper_stt(
                                    audio_path
                                )
                            )

                        print(
                            "\n========== USER ==========\n"
                        )

                        print(user_text)

                        if len(
                            user_text.strip()
                        ) == 0:

                            print(
                                "Empty transcription"
                            )

                            break

                        answer = (
                            ask_gemma_voice_vision(
                                voice_image,
                                user_text
                            )
                        )

                        print(
                            "\n========== GEMMA ==========\n"
                        )

                        print(answer)

                        speak_text(answer)

                    except Exception as e:

                        print("Error:", e)

                    # =========================
                    # DEBOUNCE
                    # =========================

                    time.sleep(0.5)

                    break

                time.sleep(0.01)

            # =================================
            # SHORT PRESS -> SCENE
            # =================================

            release_time = (
                time.time() - press_time
            )

            if release_time < HOLD_TIME:

                print("\n[SCENE BUTTON PRESSED]")

                image_path = "scene.jpg"

                small_frame = cv2.resize(
                    frame,
                    (320, 240)
                )

                cv2.imwrite(
                    image_path,
                    small_frame
                )

                if current_language == "vi":

                    prompt = SCENE_PROMPT_VI

                else:

                    prompt = SCENE_PROMPT_EN

                try:

                    output = ask_gemma(
                        image_path,
                        prompt
                    )

                    print(
                        "\n========== SCENE OUTPUT ==========\n"
                    )

                    print(output)

                    speak_text(output)

                except Exception as e:

                    print("Gemma Error:", e)

            time.sleep(0.3)


        # =====================================
        # BUTTON 2 - OCR
        # =====================================

        elif GPIO.input(BUTTON_OCR) == GPIO.LOW:

            print("\n[OCR BUTTON PRESSED]")

            image_path = "ocr.jpg"

            small_frame = cv2.resize(
                frame,
                (320, 240)
            )

            cv2.imwrite(
                image_path,
                small_frame
            )

            if current_language == "vi":

                prompt = OCR_PROMPT_VI

            else:

                prompt = OCR_PROMPT_EN

            try:

                output = ask_gemma(
                    image_path,
                    prompt
                )

                print(
                    "\n========== OCR OUTPUT ==========\n"
                )

                print(output)

                speak_text(output)

            except Exception as e:

                print("Gemma Error:", e)

            time.sleep(1)

        # =====================================
        # BUTTON 3 - MONEY
        # =====================================

        elif GPIO.input(BUTTON_MONEY) == GPIO.LOW:

            print("\n[MONEY BUTTON PRESSED]")

            image_path = "money.jpg"

            small_frame = cv2.resize(
                frame,
                (320, 240)
            )

            cv2.imwrite(
                image_path,
                small_frame
            )

            if current_language == "vi":

                prompt = MONEY_PROMPT_VI

            else:

                prompt = MONEY_PROMPT_EN

            try:

                output = ask_gemma(
                    image_path,
                    prompt
                )

                print(
                    "\n========== MONEY OUTPUT ==========\n"
                )

                print(output)

                speak_text(output)

            except Exception as e:

                print("Gemma Error:", e)

            time.sleep(1)

        # =====================================
        # BUTTON 4 - LANGUAGE
        # =====================================

        elif GPIO.input(BUTTON_LANGUAGE) == GPIO.LOW:

            if current_language == "en":

                current_language = "vi"

                print("\nSwitched to Vietnamese")

                speak_text(
                    "Đã chuyển sang tiếng Việt"
                )

            else:

                current_language = "en"

                print("\nSwitched to English")

                speak_text(
                    "Switched to English"
                )

            time.sleep(1)

        key = cv2.waitKey(1)

        if key == ord("q"):

            break

GPIO.cleanup()

cv2.destroyAllWindows()

